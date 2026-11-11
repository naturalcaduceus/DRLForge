"""简易对抗环境（可直接替换为真实仿真引擎）。

设计要点
--------
1. 单智能体控制"己方单位"，敌方单位由两种方式驱动：
   - 默认：脚本化反击（靠近/攻击），用于快速自测；
   - 对抗：外部传入 opponent_policy（可为另一个构建体），敌方按策略行动，
     从而支持"多个构建体同时参与对抗"的自博弈/联赛。
2. 动作空间同时支持离散与连续：
   - discrete：0=攻击最近敌人 1=靠近 2=远离
   - continuous：[dir_x, dir_y, attack]，attack>0.5 视为攻击，否则按朝向投影判进退
3. 观测统一为 (max_detected, 3) 的实体集合：每行 [dx, dy, hp_ratio]。

step() 返回 Gymnasium 风格五元组。
"""
from typing import Callable, Optional

import numpy as np

from .base import BaseEnv, IOSpec, ObsSpec, ActionSpec


class _DiscreteSpace:
    def __init__(self, n: int):
        self.n = int(n)

    def __repr__(self):
        return f"Discrete({self.n})"


class _BoxSpace:
    def __init__(self, shape, low, high):
        self.shape = tuple(shape)
        self.low = low
        self.high = high

    def __repr__(self):
        return f"Box({self.shape})"


class Unit:
    def __init__(self, x, y, hp=100.0, speed=50.0,
                 attack_range=100.0, attack_damage=20.0):
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


class CombatEnv(BaseEnv):
    """简易 2D 对抗环境。"""

    ACTION_ATTACK = 0
    ACTION_APPROACH = 1
    ACTION_RETREAT = 2

    def __init__(self, num_enemies=3, map_size=2000.0, detect_range=1000.0,
                 max_detected=10, max_episode_steps=300, seed=None,
                 action_space_type="discrete", action_dim=3):
        self.num_enemies = int(num_enemies)
        self.map_size = float(map_size)
        self.detect_range = float(detect_range)
        self.max_detected = int(max_detected)
        self.max_episode_steps = int(max_episode_steps)

        self.action_space_n = 3
        self.action_space_type = str(action_space_type).lower()
        self.observation_shape = (self.max_detected, 3)

        if self.action_space_type == "continuous":
            self.action_space = _BoxSpace((int(action_dim),), -1.0, 1.0)
        else:
            self.action_space = _DiscreteSpace(self.action_space_n)

        self.io_spec = IOSpec(
            obs=ObsSpec(
                shape=list(self.observation_shape),
                dim=3,
                entity_set=True,
                names=["dx", "dy", "hp_ratio"],
                normalize=True,
            ),
            action=ActionSpec(
                space_type=self.action_space_type,
                dim=int(action_dim) if self.action_space_type == "continuous"
                else self.action_space_n,
            ),
        )

        self.norm_dist = self.map_size
        self.rng = np.random.default_rng(seed)

        self.agent = None
        self.enemies = []
        self.elapsed_steps = 0
        self.terminated = False
        self.truncated = False

        # 对手策略：obs(enemy 视角) -> 队伍动作 0/1/2
        self.opponent_policy: Optional[Callable] = None
        self.opponent_last_action = None

    # ---------------- 生命周期 ----------------

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        center = self.map_size / 2.0
        self.agent = Unit(center, center, hp=100.0, speed=50.0,
                          attack_range=100.0, attack_damage=20.0)

        self.enemies = []
        for _ in range(self.num_enemies):
            angle = self.rng.uniform(0.0, 2.0 * np.pi)
            radius = self.rng.uniform(150.0, min(700.0, self.detect_range * 0.8))
            x = np.clip(center + radius * np.cos(angle), 0.0, self.map_size)
            y = np.clip(center + radius * np.sin(angle), 0.0, self.map_size)
            self.enemies.append(
                Unit(x, y, hp=50.0, speed=30.0,
                     attack_range=80.0, attack_damage=10.0)
            )

        self.elapsed_steps = 0
        self.terminated = False
        self.truncated = False
        return self._get_observation(), {}

    def close(self):
        pass

    # ---------------- 对手接入 ----------------

    def set_opponent(self, policy: Optional[Callable]):
        """注入对手策略；传入 None 表示恢复脚本化敌人。"""
        self.opponent_policy = policy

    # ---------------- 观测 ----------------

    def _get_observation(self):
        """己方视角：可见敌人集合。"""
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

    def _get_enemy_observation(self):
        """敌方视角：以最近存活敌人为观测者，描述己方单位。"""
        obs = np.zeros(self.observation_shape, dtype=np.float32)
        alive = [e for e in self.enemies if e.is_alive()]
        if not alive or not self.agent.is_alive():
            return obs
        observer = min(alive, key=lambda e: e.distance_to(self.agent))
        obs[0] = np.asarray([
            (self.agent.x - observer.x) / self.norm_dist,
            (self.agent.y - observer.y) / self.norm_dist,
            self.agent.hp / max(self.agent.max_hp, 1e-8),
        ], dtype=np.float32)
        return obs

    # ---------------- 动作归一化 ----------------

    def _resolve_action(self, action, nearest_enemy) -> int:
        """把外部动作（离散/连续）统一成 0/1/2 语义。"""
        if self.action_space_type == "discrete":
            arr = np.asarray(action).reshape(-1)
            if arr.size != 1:
                raise ValueError(
                    f"离散动作必须是标量，实得 shape={np.asarray(action).shape}"
                )
            a = int(arr[0])
            if a < 0 or a >= self.action_space_n:
                raise ValueError(f"非法动作：{a}（离散动作范围 0..{self.action_space_n - 1}）")
            return a

        vec = np.asarray(action, dtype=np.float32).reshape(-1)
        if vec.size < 3:
            raise ValueError(f"连续动作维度不足：期望 3，实得 {vec.size}")
        if vec[2] > 0.5:
            return self.ACTION_ATTACK
        if nearest_enemy is None:
            return self.ACTION_APPROACH
        dx = nearest_enemy.x - self.agent.x
        dy = nearest_enemy.y - self.agent.y
        norm = max(float(np.hypot(dx, dy)), 1e-8)
        proj = float(vec[0] * dx / norm + vec[1] * dy / norm)
        return self.ACTION_APPROACH if proj >= 0.0 else self.ACTION_RETREAT

    # ---------------- 敌方行动 ----------------

    def _enemy_team_act(self, team_action: int) -> float:
        """敌方全队执行同一动作，返回对己方造成的伤害。"""
        damage = 0.0
        for enemy in self.enemies:
            if not enemy.is_alive():
                continue
            dx = self.agent.x - enemy.x
            dy = self.agent.y - enemy.y
            dist = max(float(np.hypot(dx, dy)), 1e-8)
            ux, uy = dx / dist, dy / dist

            if team_action == self.ACTION_ATTACK:
                if dist <= enemy.attack_range:
                    self.agent.hp -= enemy.attack_damage
                    damage += enemy.attack_damage
            elif team_action == self.ACTION_APPROACH:
                move = min(enemy.speed, max(0.0, dist - enemy.attack_range))
                enemy.x += ux * move
                enemy.y += uy * move
            elif team_action == self.ACTION_RETREAT:
                enemy.x -= ux * enemy.speed
                enemy.y -= uy * enemy.speed

            enemy.x = float(np.clip(enemy.x, 0.0, self.map_size))
            enemy.y = float(np.clip(enemy.y, 0.0, self.map_size))
        return damage

    def _coerce_team_action(self, team_action) -> int:
        """把对手输出统一成 0/1/2 队伍动作。

        兼容离散 int/tensor 与连续 [dir_x, dir_y, attack] 向量。
        """
        if isinstance(team_action, (int, np.integer)):
            return int(team_action) % self.action_space_n
        if hasattr(team_action, "item") and getattr(team_action, "numel", lambda: 0)() == 1:
            return int(team_action.item()) % self.action_space_n
        arr = np.asarray(team_action, dtype=np.float64).reshape(-1)
        if arr.size >= 3:
            if arr[2] > 0.5:
                return self.ACTION_ATTACK
            return self.ACTION_APPROACH if arr[0] >= 0.0 else self.ACTION_RETREAT
        if arr.size >= 1:
            return int(arr[0]) % self.action_space_n
        return self.ACTION_APPROACH

    # ---------------- 单步 ----------------

    def step(self, action):
        if self.terminated or self.truncated:
            raise RuntimeError("Episode 已结束，请先调用 reset()。")

        self.elapsed_steps += 1
        reward = -0.01
        info = {}

        visible_alive = [
            e for e in self.enemies
            if e.is_alive() and self.agent.distance_to(e) <= self.detect_range
        ]
        nearest_enemy = None
        if visible_alive:
            nearest_enemy = min(visible_alive,
                                key=lambda e: self.agent.distance_to(e))

        action_id = self._resolve_action(action, nearest_enemy)
        info["action_taken"] = action_id

        if nearest_enemy is not None:
            dx = nearest_enemy.x - self.agent.x
            dy = nearest_enemy.y - self.agent.y
            dist = max(float(np.hypot(dx, dy)), 1e-8)
            ux, uy = dx / dist, dy / dist

            if action_id == self.ACTION_ATTACK:
                if dist <= self.agent.attack_range:
                    old_hp = nearest_enemy.hp
                    nearest_enemy.hp = max(0.0, nearest_enemy.hp - self.agent.attack_damage)
                    reward += 1.0
                    if old_hp > 0.0 and not nearest_enemy.is_alive():
                        reward += 10.0
                        info["killed"] = True
                else:
                    reward -= 0.1
            elif action_id == self.ACTION_APPROACH:
                move = min(self.agent.speed, max(0.0, dist - self.agent.attack_range))
                self.agent.x += ux * move
                self.agent.y += uy * move
            elif action_id == self.ACTION_RETREAT:
                self.agent.x -= ux * self.agent.speed
                self.agent.y -= uy * self.agent.speed

        self.agent.x = float(np.clip(self.agent.x, 0.0, self.map_size))
        self.agent.y = float(np.clip(self.agent.y, 0.0, self.map_size))

        # 敌方行动：有对手策略则按策略，否则脚本化反击
        if self.opponent_policy is not None:
            enemy_obs = self._get_enemy_observation()
            team_action = self._coerce_team_action(self.opponent_policy(enemy_obs))
            self.opponent_last_action = team_action
            damage = self._enemy_team_act(team_action)
            info["enemy_action"] = team_action
        else:
            damage = 0.0
            for enemy in self.enemies:
                if enemy.is_alive() and enemy.distance_to(self.agent) <= enemy.attack_range:
                    damage += enemy.attack_damage
            self.agent.hp -= damage

        if damage > 0.0:
            reward -= 0.02 * damage

        alive_enemies = sum(e.is_alive() for e in self.enemies)
        if not self.agent.is_alive():
            self.terminated = True
            reward -= 10.0
            info["result"] = "agent_dead"
        elif alive_enemies == 0:
            self.terminated = True
            reward += 30.0
            info["result"] = "all_enemies_eliminated"

        if not self.terminated and self.elapsed_steps >= self.max_episode_steps:
            self.truncated = True
            info["result"] = "time_limit"

        return (self._get_observation(), float(reward),
                self.terminated, self.truncated, info)


class CombatEnvProxy(CombatEnv):
    """仿真引擎接入适配层。

    真实引擎（网络/进程/本地 DLL）只需在这层实现
    reset()/step() 并维护 io_spec，即可被训练与推理流程直接复用。
    当前内置实现为简易对抗环境，开箱即用。
    """

    def io_dict(self) -> dict:
        return self.io_spec.to_dict()
