# -*- coding: utf-8 -*-
"""智能体平台的纯逻辑测试：配置组装、校验、dry-run、训练输出解析、曲线读取。"""
import csv
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from fz_agent import studio


class BuildConfigTest(unittest.TestCase):
    def test_chess_preset_is_valid(self):
        form = dict(studio.PRESETS["chess_ppo"])
        cfg, warns = studio.build_config(form)
        self.assertEqual(cfg["env"]["type"], "chess_remote")
        self.assertEqual(cfg["env"]["io"]["obs"]["shape"], [12, 10])
        self.assertEqual(cfg["networks"]["policy"]["head"], "actor_critic")
        self.assertEqual(cfg["algo"]["name"], "ppo")
        errs, _ = studio.validate_config(cfg)
        self.assertEqual(errs, [], "预设配置应通过静态校验")
        self.assertIn("policy", cfg["networks"])

    def test_combat_preset_env(self):
        cfg, _ = studio.build_config(dict(studio.PRESETS["combat_ppo"]))
        self.assertEqual(cfg["env"]["type"], "combat")
        self.assertEqual(cfg["env"]["config"]["num_enemies"], 3)
        errs, _ = studio.validate_config(cfg)
        self.assertEqual(errs, [])

    def test_continuous_algorithm(self):
        form = dict(studio.PRESETS["chess_ppo"])
        form["algo"] = "ddpg"
        form["io"] = dict(form["io"], space_type="continuous", action_dim=3)
        cfg, _ = studio.build_config(form)
        self.assertEqual(set(cfg["networks"]), {"actor", "critic"})
        errs, _ = studio.validate_config(cfg)
        self.assertEqual(errs, [])

    def test_wrong_space_for_algorithm(self):
        form = dict(studio.PRESETS["chess_ppo"])
        form["algo"] = "ddpg"  # 连续算法却给离散动作
        cfg, _ = studio.build_config(form)
        errs, _ = studio.validate_config(cfg)
        self.assertTrue(any("连续" in e for e in errs))

    def test_rainbow_blocked(self):
        form = dict(studio.PRESETS["chess_ppo"])
        form["algo"] = "rainbow"
        cfg, _ = studio.build_config(form)
        errs, _ = studio.validate_config(cfg)
        self.assertTrue(any("待补齐" in e for e in errs))

    def test_stack_input_alignment(self):
        form = dict(studio.PRESETS["chess_ppo"])
        form["stack"] = [{"type": "mlp", "config": {
            "in_dim": 3, "hidden_dims": [64], "out_dim": 64, "activation": "ReLU"}}]
        cfg, warns = studio.build_config(form)
        self.assertEqual(cfg["networks"]["policy"]["network_stack"][0]["config"]["in_dim"], 10)
        self.assertTrue(any("对齐" in w for w in warns))

    def test_all_algos_buildable_metadata(self):
        self.assertGreaterEqual(len(studio.ALGOS), 11, "算法库不少于 11 种")
        self.assertGreaterEqual(len(studio.NET_MODULES), 8, "网络积木不少于 8 种")
        self.assertEqual(len(studio.STRATEGIES) - 1, 6, "演化训练 6 种（不含'无'）")

    def test_strategy_attached(self):
        form = dict(studio.PRESETS["chess_ppo"])
        form["strategy"] = {"name": "adversarial_train", "config": {"epsilon": 0.02}}
        cfg, _ = studio.build_config(form)
        self.assertEqual(cfg["strategy"]["name"], "adversarial_train")


class ValidateAndDryRunTest(unittest.TestCase):
    def test_dry_run_mlp(self):
        cfg, _ = studio.build_config(dict(studio.PRESETS["chess_ppo"]))
        ok, msgs = studio.dry_run(cfg)
        self.assertTrue(ok)
        self.assertTrue(any("前向 OK" in m for m in msgs))

    def test_dry_run_dqn(self):
        cfg, _ = studio.build_config(dict(studio.PRESETS["chess_dqn"]))
        ok, msgs = studio.dry_run(cfg)
        self.assertTrue(ok)

    def test_signature_stable(self):
        cfg, _ = studio.build_config(dict(studio.PRESETS["chess_ppo"]))
        self.assertEqual(studio.config_signature(cfg), studio.config_signature(cfg))
        cfg2, _ = studio.build_config(dict(studio.PRESETS["chess_dqn"]))
        self.assertNotEqual(studio.config_signature(cfg), studio.config_signature(cfg2))


class TrainParseTest(unittest.TestCase):
    def test_parse_lines(self):
        self.assertEqual(studio.parse_train_line("Epoch 003/020 | eval=1.2")[0], "epoch")
        self.assertEqual(studio.parse_train_line("step 01000/05000 | eval=0.5")[0], "step")
        self.assertEqual(studio.parse_train_line("[bundle] 构建体已导出: /x")[0], "bundle")
        self.assertEqual(studio.parse_train_line("Traceback (most recent call last)")[0], "error")

    def test_read_curves(self):
        d = tempfile.mkdtemp()
        try:
            with open(os.path.join(d, "train_epoch.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["epoch", "train_reward_mean20", "policy_loss"])
                w.writeheader()
                w.writerow({"epoch": 1, "train_reward_mean20": -1.5, "policy_loss": 0.3})
                w.writerow({"epoch": 2, "train_reward_mean20": -1.0, "policy_loss": 0.2})
            with open(os.path.join(d, "eval.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["epoch", "mean_reward", "win_rate"])
                w.writeheader()
                w.writerow({"epoch": 1, "mean_reward": -2.0, "win_rate": 0.0})
            curves = studio.read_curves(d)
            self.assertEqual(curves["epochs"], ["1", "2"])
            self.assertEqual(curves["train_reward"], [-1.5, -1.0])
            self.assertEqual(curves["eval_reward"], [-2.0])
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_save_and_load_config(self):
        cfg, _ = studio.build_config(dict(studio.PRESETS["combat_ppo"]))
        cfg["meta"]["name"] = "_unittest_studio_"
        path = studio.save_config(cfg)
        try:
            self.assertTrue(os.path.exists(path))
            again = studio.load_config(path)
            self.assertEqual(again["meta"]["name"], "_unittest_studio_")
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()