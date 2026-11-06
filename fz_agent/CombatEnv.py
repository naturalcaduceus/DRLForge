
import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# ============================================================
# 2. 演示环境
#    真实使用时，可以用自己的仿真环境替换 CombatEnv。
# ============================================================

# 计算状态用的
class Unit:
    def __init__(
        self,
        x,
        y,
        hp=100.0,
        speed=50.0,
        attack_range=100.0,
        attack_damage=20.0,
    ):
        self.x = float(x)
        self.y = float(y)
        self.hp = float(hp)
        self.max_hp = float(hp)
        self.speed = float(speed)
        self.attack_range = float(attack_range)
        self.attack_damage = float(attack_damage)

    def distance_to(self, other):
        return float(np.hypot(self.x - other.x, self.y - other.y))

    def is_alive(self):
        return self.hp > 0.0


class CombatEnv:
    """
    动作：
        0：攻击最近的可见敌人
        1：靠近最近的可见敌人
        2：远离最近的可见敌人

    step() 返回 Gymnasium 风格的五元组：
        obs, reward, terminated, truncated, info
    """

    def __init__(
        self,
        num_enemies=3,
        map_size=2000.0,
        detect_range=1000.0,
        max_detected=10,
        max_episode_steps=300,
        seed=None,
    ):
        self.num_enemies = int(num_enemies)
        self.map_size = float(map_size)
        self.detect_range = float(detect_range)
        self.max_detected = int(max_detected)
        self.max_episode_steps = int(max_episode_steps)

        self.action_space_n = 3
        self.observation_shape = (self.max_detected, 3)

        self.norm_dist = self.map_size
        self.norm_hp = 100.0
        self.rng = np.random.default_rng(seed)

        self.agent = None
        self.enemies = []
        self.elapsed_steps = 0
        self.terminated = False
        self.truncated = False

    # 重开
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        center = self.map_size / 2.0
        self.agent = Unit(
            center,
            center,
            hp=100.0,
            speed=50.0,
            attack_range=100.0,
            attack_damage=20.0,
        )

        self.enemies = []
        for _ in range(self.num_enemies):
            # 将敌人初始化在侦察范围内，避免大量 Episode 完全没有交互。
            angle = self.rng.uniform(0.0, 2.0 * np.pi)
            radius = self.rng.uniform(150.0, min(700.0, self.detect_range * 0.8))
            x = np.clip(center + radius * np.cos(angle), 0.0, self.map_size)
            y = np.clip(center + radius * np.sin(angle), 0.0, self.map_size)

            self.enemies.append(
                Unit(
                    x,
                    y,
                    hp=50.0,
                    speed=30.0,
                    attack_range=80.0,
                    attack_damage=10.0,
                )
            )

        self.elapsed_steps = 0
        self.terminated = False
        self.truncated = False
        return self._get_observation(), {}

    def _get_observation(self):
        obs = np.zeros(self.observation_shape, dtype=np.float32)
        visible = []

        for enemy in self.enemies:
            if not enemy.is_alive():
                continue

            dist = self.agent.distance_to(enemy)
            if dist <= self.detect_range:
                dx = (enemy.x - self.agent.x) / self.norm_dist
                dy = (enemy.y - self.agent.y) / self.norm_dist
                hp_ratio = enemy.hp / max(enemy.max_hp, 1e-8)
                visible.append((dist, [dx, dy, hp_ratio]))

        visible.sort(key=lambda item: item[0])

        for i, (_, feature) in enumerate(visible[: self.max_detected]):
            obs[i] = np.asarray(feature, dtype=np.float32)

        return obs

    # 执行动作，返回结果
    def step(self, action):
        if self.terminated or self.truncated:
            raise RuntimeError("Episode 已结束，请先调用 reset()。")

        action = int(action)
        if action < 0 or action >= self.action_space_n:
            raise ValueError(f"非法动作：{action}")

        self.elapsed_steps += 1
        reward = -0.01  # 轻微步数惩罚，避免无意义拖延
        info = {"action_taken": action}

        visible_alive = [
            enemy
            for enemy in self.enemies
            if enemy.is_alive()
            and self.agent.distance_to(enemy) <= self.detect_range
        ]

        nearest_enemy = None
        if visible_alive:
            nearest_enemy = min(
                visible_alive,
                key=lambda enemy: self.agent.distance_to(enemy),
            )

        if nearest_enemy is not None:
            dx = nearest_enemy.x - self.agent.x
            dy = nearest_enemy.y - self.agent.y
            dist = max(float(np.hypot(dx, dy)), 1e-8)
            ux, uy = dx / dist, dy / dist

            if action == 0:  # 攻击
                if dist <= self.agent.attack_range:
                    old_hp = nearest_enemy.hp
                    nearest_enemy.hp = max(
                        0.0, nearest_enemy.hp - self.agent.attack_damage
                    )
                    reward += 1.0
                    if old_hp > 0.0 and not nearest_enemy.is_alive():
                        reward += 10.0
                        info["killed"] = True
                else:
                    reward -= 0.1  # 超出射程攻击，给予小惩罚

            elif action == 1:  # 靠近
                move_dist = min(self.agent.speed, max(0.0, dist - self.agent.attack_range))
                self.agent.x += ux * move_dist
                self.agent.y += uy * move_dist

            elif action == 2:  # 远离
                move_dist = self.agent.speed
                self.agent.x -= ux * move_dist
                self.agent.y -= uy * move_dist

        self.agent.x = float(np.clip(self.agent.x, 0.0, self.map_size))
        self.agent.y = float(np.clip(self.agent.y, 0.0, self.map_size))

        # 敌方反击
        incoming_damage = 0.0
        for enemy in self.enemies:
            if (
                enemy.is_alive()
                and enemy.distance_to(self.agent) <= enemy.attack_range
            ):
                incoming_damage += enemy.attack_damage

        if incoming_damage > 0.0:
            self.agent.hp -= incoming_damage
            reward -= 0.02 * incoming_damage

        alive_enemies = sum(enemy.is_alive() for enemy in self.enemies)

        if not self.agent.is_alive():
            self.terminated = True
            reward -= 10.0
            info["result"] = "agent_dead"
        elif alive_enemies == 0:
            self.terminated = True
            reward += 30.0
            info["result"] = "all_enemies_eliminated"

        # 时间上限属于 truncation，不等同于真正的终止状态。
        if not self.terminated and self.elapsed_steps >= self.max_episode_steps:
            self.truncated = True
            info["result"] = "time_limit"

        return (
            self._get_observation(),
            float(reward),
            self.terminated,
            self.truncated,
            info,
        )

