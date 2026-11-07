"""训练过程记录器。

一次训练 = 一个 run 目录，内容：
    runs/<run_name>_<时间戳>/
        config.json       训练配置快照
        train_epoch.csv   每个 epoch 的训练指标
        train_episode.csv 每个 episode 的回报 / 长度 / 结果
        eval.csv          每次评估的指标
        events.jsonl      结构化事件流（checkpoint、配置、状态等）
        train.log         完整文本日志
        summary.json      训练结束汇总
        curves.png        训练曲线图
        tensorboard/      TensorBoard event 文件

使用：
    logger = TrainLogger("ppo_combat")
    logger.log_config(cfg)
    logger.log_epoch(epoch, metrics, extra={...})
    logger.log_episode(episode_id, reward, length, result)
    logger.log_eval(epoch, eval_metrics)
    logger.close()
"""

import csv
import json
import logging
import sys
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np

from ..config import RUNS_DIR


# ============================================================
# CSV 增量写入（列可以动态增加，每次重写保证列对齐）
# ============================================================

class CsvTable:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.rows = []
        self.fieldnames = []

    def append(self, row: dict):
        for key in row:
            if key not in self.fieldnames:
                self.fieldnames.append(key)
        self.rows.append(dict(row))
        self.flush()

    def flush(self):
        if not self.rows:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
            for row in self.rows:
                writer.writerow({k: row.get(k, "") for k in self.fieldnames})

    def __len__(self):
        return len(self.rows)


# ============================================================
# 训练记录器
# ============================================================

class TrainLogger:
    """统一记录训练过程：CSV + JSONL + TensorBoard + 文本日志 + 曲线图。"""

    def __init__(self, run_name: str = "run", log_dir=None,
                 use_tensorboard: bool = True, plot: bool = True):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(
            c if c.isalnum() or c in "-_." else "_" for c in run_name
        )
        base_dir = Path(log_dir) if log_dir else RUNS_DIR
        self.run_dir = base_dir / f"{safe_name}_{stamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # 数据文件
        self.epoch_table = CsvTable(self.run_dir / "train_epoch.csv")
        self.episode_table = CsvTable(self.run_dir / "train_episode.csv")
        self.eval_table = CsvTable(self.run_dir / "eval.csv")
        self.events_path = self.run_dir / "events.jsonl"
        self.config_path = self.run_dir / "config.json"
        self.summary_path = self.run_dir / "summary.json"
        self.curves_path = self.run_dir / "curves.png"
        self.log_path = self.run_dir / "train.log"

        # 文本日志（控制台 + 文件）
        self.logger = self._build_logger()

        # TensorBoard（可选）
        self.writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(str(self.run_dir / "tensorboard"))
            except Exception as e:                      # noqa: BLE001
                self.logger.warning("TensorBoard 不可用，已跳过: %s", e)

        self.plot_enabled = plot
        self.start_time = time.time()
        self.episode_count = 0
        self.closed = False

        self.logger.info("Run 目录: %s", self.run_dir)

    # ---------------- 内部工具 ----------------

    def _build_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"fz_agent_run_{id(self)}")
        logger.setLevel(logging.INFO)
        logger.propagate = False

        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-5s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(fmt)
        file_handler = logging.FileHandler(self.log_path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(stream)
        logger.addHandler(file_handler)
        return logger

    def _tb_scalars(self, prefix: str, values: dict, step: int):
        if self.writer is None:
            return
        for key, value in values.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                if np.isfinite(float(value)):
                    self.writer.add_scalar(f"{prefix}/{key}", float(value), step)

    def _json_default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return str(obj)

    # ---------------- 配置 & 事件 ----------------

    def log_config(self, config: dict, name: str = "config"):
        payload = dict(config)
        payload["_run_dir"] = str(self.run_dir)
        payload["_created_at"] = datetime.now().isoformat(timespec="seconds")
        self.config_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False,
                       default=self._json_default),
            encoding="utf-8",
        )
        self.log_event(f"{name}_saved", path=str(self.config_path))

    def log_event(self, event: str, **payload):
        record = {
            "time": datetime.now().isoformat(timespec="milliseconds"),
            "event": event,
            **payload,
        }
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False,
                               default=self._json_default) + "\n")

    def info(self, message: str, *args):
        self.logger.info(message, *args)

    def warning(self, message: str, *args):
        self.logger.warning(message, *args)

    # ---------------- 指标记录 ----------------

    def log_epoch(self, epoch: int, metrics: dict, extra: dict = None):
        """记录一个 epoch 的训练指标。"""
        row = OrderedDict()
        row["epoch"] = int(epoch)
        row["wall_time_sec"] = round(time.time() - self.start_time, 3)

        for key, value in metrics.items():
            if isinstance(value, (np.integer, np.floating)):
                value = value.item()
            row[key] = value
        if extra:
            for key, value in extra.items():
                row[key] = value

        self.epoch_table.append(row)
        self._tb_scalars("train", row, epoch)
        self.log_event("epoch", **row)
        return row

    def log_episode(self, reward: float, length: int, result: str = None,
                    env_steps: int = None, **extra):
        """记录一个 episode。"""
        self.episode_count += 1
        row = OrderedDict()
        row["episode"] = self.episode_count
        row["env_steps"] = env_steps if env_steps is not None else ""
        row["reward"] = round(float(reward), 6)
        row["length"] = int(length)
        row["result"] = result or ""
        for key, value in extra.items():
            row[key] = value

        self.episode_table.append(row)
        self._tb_scalars(
            "episode",
            {"reward": reward, "length": length},
            self.episode_count,
        )
        return row

    def log_eval(self, epoch: int, metrics: dict):
        """记录一次评估。"""
        row = OrderedDict()
        row["epoch"] = int(epoch)
        row["wall_time_sec"] = round(time.time() - self.start_time, 3)
        for key, value in metrics.items():
            if isinstance(value, (np.integer, np.floating)):
                value = value.item()
            row[key] = value

        self.eval_table.append(row)
        self._tb_scalars("eval", metrics, epoch)
        self.log_event("eval", **row)
        return row

    # ---------------- 曲线图 ----------------

    def plot_curves(self):
        """根据已记录数据画训练曲线，保存 curves.png。"""
        if not self.plot_enabled or not self.episode_table.rows:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:                          # noqa: BLE001
            self.logger.warning("matplotlib 不可用，跳过绘图: %s", e)
            return

        try:
            fig, axes = plt.subplots(2, 2, figsize=(13, 8))
            fig.suptitle(f"Training Curves - {self.run_dir.name}", fontsize=12)

            episodes = self.episode_table.rows
            rewards = np.array([float(r["reward"]) for r in episodes])
            ax = axes[0][0]
            ax.plot(rewards, alpha=0.35, label="episode reward")
            if len(rewards) >= 5:
                window = min(20, len(rewards))
                smooth = np.convolve(
                    rewards, np.ones(window) / window, mode="valid",
                )
                ax.plot(range(window - 1, len(rewards)), smooth,
                        linewidth=2, label=f"moving avg({window})")
            ax.set_title("Episode Reward")
            ax.set_xlabel("episode")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)

            epochs = self.epoch_table.rows
            ax = axes[0][1]
            if epochs:
                xs = [r["epoch"] for r in epochs]
                for key in ("train_reward_mean20", "train_reward"):
                    if key in epochs[0] or any(key in r for r in epochs):
                        ys = [r.get(key, np.nan) for r in epochs]
                        ax.plot(xs, ys, marker="o", ms=3, label=key)
                evals = self.eval_table.rows
                if evals:
                    ax.plot(
                        [r["epoch"] for r in evals],
                        [r["mean_reward"] for r in evals],
                        marker="s", ms=4, label="eval mean_reward",
                    )
            ax.set_title("Reward: Train vs Eval")
            ax.set_xlabel("epoch")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)

            ax = axes[1][0]
            if epochs:
                xs = [r["epoch"] for r in epochs]
                for key in ("policy_loss", "value_loss"):
                    if any(key in r for r in epochs):
                        ax.plot(xs, [r.get(key, np.nan) for r in epochs],
                                marker="o", ms=3, label=key)
            ax.set_title("Losses")
            ax.set_xlabel("epoch")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)

            ax = axes[1][1]
            if epochs:
                xs = [r["epoch"] for r in epochs]
                if any("entropy" in r for r in epochs):
                    ax.plot(xs, [r.get("entropy", np.nan) for r in epochs],
                            marker="o", ms=3, label="entropy")
                evals = self.eval_table.rows
                if evals and any("win_rate" in r for r in evals):
                    ax2 = ax.twinx()
                    ax2.plot(
                        [r["epoch"] for r in evals],
                        [r["win_rate"] for r in evals],
                        color="tab:red", marker="s", ms=4, label="eval win_rate",
                    )
                    ax2.set_ylabel("win_rate", color="tab:red")
                    ax2.tick_params(axis="y", labelcolor="tab:red")
                    ax2.set_ylim(-0.02, 1.02)
            ax.set_title("Entropy & Win Rate")
            ax.set_xlabel("epoch")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8, loc="upper left")

            fig.tight_layout()
            fig.savefig(self.curves_path, dpi=140)
            plt.close(fig)
            self.logger.info("训练曲线已保存: %s", self.curves_path)
        except Exception as e:                          # noqa: BLE001
            self.logger.warning("绘制曲线失败: %s", e)

    # ---------------- 结束 ----------------

    def close(self, extra_summary: dict = None):
        if self.closed:
            return
        self.closed = True

        self.plot_curves()

        summary = {
            "run_dir": str(self.run_dir),
            "total_epochs": len(self.epoch_table),
            "total_episodes": len(self.episode_table),
            "total_evals": len(self.eval_table),
            "elapsed_sec": round(time.time() - self.start_time, 2),
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        }
        if self.eval_table.rows:
            best = max(self.eval_table.rows, key=lambda r: r.get("mean_reward", -1e18))
            summary["best_eval"] = {
                "epoch": best.get("epoch"),
                "mean_reward": best.get("mean_reward"),
                "win_rate": best.get("win_rate"),
            }
        if self.epoch_table.rows:
            summary["last_epoch_metrics"] = dict(self.epoch_table.rows[-1])
        if extra_summary:
            summary.update(extra_summary)

        self.summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False,
                       default=self._json_default),
            encoding="utf-8",
        )
        self.log_event("run_finished", **summary)

        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

        self.logger.info("训练记录已保存: %s", self.run_dir)
        for handler in list(self.logger.handlers):
            handler.flush()
            handler.close()
            self.logger.removeHandler(handler)
