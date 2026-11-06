"""全局配置 / 路径常量。"""
import os
from pathlib import Path

# 项目根目录（fz_agent 所在的那一层）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FZ_AGENT_ROOT = Path(__file__).resolve().parent

# 输出目录
OUTPUT_DIR = FZ_AGENT_ROOT / "output"
CHECKPOINT_DIR = FZ_AGENT_ROOT / "checkpoints"

# 自动创建
for d in (OUTPUT_DIR, CHECKPOINT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# 演示用 JSON 目录
DOCS_DIR = PROJECT_ROOT / "docs"
