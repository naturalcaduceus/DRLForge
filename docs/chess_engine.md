# 接入 chess-engine2（Windows 环境机）

`fz_agent.env.ChessEnv.ChessRemoteEnv` 把 chess-engine2 的 `tools.net_service`
当作训练环境：`reset(seed)` / `step(action)` / `set_opponent(policy)` 与内置
`CombatEnv` 完全同形，训练循环、构建体导出、`serve` 部署不需要任何改动。

完整跨系统说明（WSL ↔ Windows 地址、防火墙、四种用法）见环境仓库：
`chess-engine2/docs/REMOTE_RL.md`。

## 1. 启动链路

```bash
# Windows（环境机）
python -m tools.net_service --scenario scenarios/square_battle.json \
    --host 0.0.0.0 --port 8000 --auto blue --max-actions 32 --obs-entities 12

# WSL（训练机）
export DRL_CHESS_URL=http://<Windows宿主机IP>:8000
python -m fz_agent.main rl-train -c fz_agent/configs/ppo_chess.json
```

同机调试可直接用 `http://127.0.0.1:8000`；WSL 镜像网络下也可直接用回环。

## 2. 配置写法

```json
"env": {
  "type": "chess_remote",
  "url": "http://127.0.0.1:8000",
  "config": {"side": "red", "max_entities": 12, "timeout": 15.0},
  "action_space_type": "discrete",
  "action_dim": 32,
  "io": {
    "obs": {"shape": [12, 10], "dim": 10, "entity_set": true,
            "names": ["dx", "dy", "hp_ratio", "atk_ratio", "armor",
                      "move_ratio", "is_enemy", "is_self", "alive", "tick_ratio"]},
    "action": {"space_type": "discrete", "dim": 32}
  }
}
```

约束（三处必须相等）：

| 位置 | 训练侧 | 环境侧 |
|------|--------|--------|
| 动作数 | `io.action.dim` = `algo.config.action_dim` | `--max-actions` |
| 实体行数 | `io.obs.shape[0]` | `--obs-entities` |
| 特征维 | `io.obs.shape[1]`、网络 `in_dim` | `envi/rl_adapter.OBS_DIM` |

已内置配置：`fz_agent/configs/ppo_chess.json`、`fz_agent/configs/dqn_chess.json`。

## 3. 会话与自博弈

- 每个环境实例用独立 `sid`（默认 `drlforge-<side>-<seed>`），
  所以 `main.py` 同时创建 train（seed）与 eval（seed+1000）两个环境不会互相踩状态。
- 服务端 `--auto blue` 时，训练端只发红方动作；
  `env.set_opponent(policy)` 时改为训练端本地给蓝方出招
  （`policy(obs_blue) -> 动作下标`），通过 `/rl/step` 的 `actions` 同时提交——这就是
  `SelfPlayEnv` / `ModelPool` 的对弈通道。
- 训练端越界动作回退为 `wait`，与 chess-engine2 的 `legal_actions` 语义一致。

## 4. 部署对战

```bash
# WSL：把构建体挂成推理服务（/act）
python -m fz_agent.serve --bundle-dir fz_agent/bundles --host 0.0.0.0 --port 8765

# Windows：神经网络棋手作为蓝方
python -m tools.play --red human --blue neural \
    --neural-url http://<WSL地址>:8765 --bundle ppo_chess
```

`/act` 请求 `{"bundle_id","obs":[[...]]}`，响应 `{"action": 7}`；
`bundle_id` 省略时，服务里只有一个构建体就自动使用它。
`GET /health` 额外返回每个构建体的 `io`，棋手据此对齐观测形状与动作维度。

## 5. 冒烟检查

```bash
# 环境侧（不依赖 torch）
curl http://<环境机>:8000/rl/spec?side=red

# 训练侧（不需要 GPU）
python - <<'PY'
from fz_agent.env.ChessEnv import ChessRemoteEnv
env = ChessRemoteEnv("http://<环境机>:8000", side="red", max_actions=32, max_entities=12)
obs, info = env.reset(seed=1)
print(obs.shape, len(info["legal_actions"]))
print(env.step(1)[1:4])
PY
```