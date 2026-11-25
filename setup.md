下面这套流程完全不需要 WSL 图形界面：Windows 只负责“环境服务 + 看回放 +（可选）开 GUI 生成配置”，WSL 只负责“训练”。



Windows                         WSL（无界面）
┌──────────────────┐            ┌───────────────────────────┐
│ chess-engine2     │  HTTP:8000 │ DRLForge                  │
│ tools.net_service ├───────────▶│ fz_agent.main rl-train    │
│ 自动写 logs/runs  │            │ 产出 bundles/ runs/       │
│ replay_viewer 看  │            │ 终端 logs.replay 也能看    │
└──────────────────┘            └───────────────────────────┘
        ▲ 构建智能体 GUI（Windows）只生成/校验 JSON，训练仍在 WSL
0. 一次性准备
Windows（环境机 + 看回放）

chess-engine2 仓库；Python 3（回放播放器需要 tkinter，Windows 自带 Python 通常有）。

放行端口（管理员 PowerShell，一次即可）：
powershell

New-NetFirewallRule -DisplayName "chess-engine2" -Direction Inbound -Protocol TCP -LocalPort 8000,8765 -Action Allow
WSL（训练机）
bash


python3 -m pip install numpy torch      # DRLForge 训练依赖
cd ~/RTS/DRLForge
要让 Windows 上的 GUI 生成配置 → WSL 训练，两边必须是同一份 DRLForge 目录（推荐把仓库放在 D 盘，WSL 用 /mnt/d/... 访问）；否则把生成的 fz_agent/configs/xxx.json 复制到 WSL 对应目录。

1. 启动环境（Windows，先跑起来）
powershell


cd C:\Users\user\1projects\chess-engine2
python -m tools.net_service --scenario scenarios/geo_sim_demo.json --host 0.0.0.0 --port 8000 --auto blue --auto-agent script

--auto blue：蓝方由托管棋手行动，训练只控制红方；
想换行为树 AI：--auto-agent bt:behavior/programs/scout_attack.json；
默认就在 logs/runs/ 记录每局回放（不用额外参数）；
启动成功会打印本机 IPv4 和 [自检] LISTENING ✓。

WSL 里先自检：
bash

cd ~/RTS/DRLForge
python -m fz_agent.net_utils --port 8000 --url http://127.0.0.1:8000

====xport WIN_HOST=$(ip route show default | awk '{print $3}')
====python -m fz_agent.net_utils --port 8000 --url http://$WIN_HOST:8000
看到 通信成功 ✓ 再往下。

2. 构建智能体（两种方式，都不用 WSL 图形界面）
方式 A：Windows GUI 只负责生成配置（推荐给需要调参数时）
powershell

cd \\wsl.localhost\Ubuntu_New\home\cq\RTS\DRLForge
python gui_agent_studio.py
「智能体构建」→ 预设 chess_ppo；

改名称（名称 = 训练产出的构建体名）、算法/网络/训练参数；
点「静态校验」（不依赖 torch，Windows 没装 torch 也能用）；
「dry-run 前向」需要 torch，没装就跳过；
「保存到 configs」→ 生成 fz_agent/configs/<名称>.json。
GUI 的「智能体训练」页是给有 torch 的机器用的；你的训练在 WSL，直接用命令行启动（下一步）。

方式 B：WSL 无界面，直接用 JSON（完全够用）
bash

cd ~/RTS/DRLForge
cp fz_agent/configs/ppo_chess.json fz_agent/configs/g.json

python3 - <<'PY'
import json
p = "fz_agent/configs/g.json"
cfg = json.load(open(p, encoding="utf-8"))
cfg["meta"]["name"] = "my_agent"          # 产出 fz_agent/bundles/my_agent
cfg["train"].update({"total_epochs": 30, "rollout_steps": 512,
                     "eval_interval": 5})
json.dump(cfg, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("ok:", cfg["meta"]["name"])
PY

或者用 studio 直接生成（可选，会打印校验结果）：
bash

python3 - <<'PY'
from fz_agent import studio
form = dict(studio.PRESETS["chess_ppo"])
form["name"] = "my_agent"
form["train"].update({"total_epochs": 30, "rollout_steps": 512})
cfg, warns = studio.build_config(form)
print("errors:", studio.validate_config(cfg))
print("saved:", studio.save_config(cfg, "g.json"))
PY

维度必须与服务端一致（默认配置已一致）：
io.obs.shape=[12,10] ↔ --obs-entities 12；io.action.dim=32 ↔ --max-actions 32。

3. 选择构建体开始训练（WSL）
3.1 从零训练
bash
cd ~/RTS/DRLForge
====export DRL_CHESS_URL=http://$(ip route show default | awk '{print $3}'):8000
export DRL_CHESS_URL=http://127.0.0.1:8000
python -m fz_agent.main rl-train -c fz_agent/configs/g.json

后台跑（推荐）：
bash

nohup env DRL_CHESS_URL=http://$WIN_HOST:8000 \
  python -m fz_agent.main rl-train -c fz_agent/configs/g.json \
  > /tmp/train_my_agent.log 2>&1 &
tail -f /tmp/train_my_agent.log
停止：kill %1（或 pkill -f "rl-train -c fz_agent/configs/g.json"）。

3.2 用已有构建体做“热启动”（选择构建体）
把已有 bundle 作为初始化权重（迁移学习）：
bash

python3 - <<'PY'
import json
p = "fz_agent/configs/g.json"
cfg = json.load(open(p, encoding="utf-8"))
cfg["strategy"] = {
    "name": "transfer_learning",
    "config": {"pretrained": "fz_agent/bundles/g20261125", "freeze": False},
    "extra": {}
}
json.dump(cfg, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
PY

pretrained 可以是 bundle 目录，也可以是 .pt 检查点（fz_agent/checkpoints/*.pth）。
注意：这是“加载权重重新训练”，不是恢复优化器/步数的断点续训；当前 incremental_continual 还是空实现。需要真正的续训功能可以再说，我补。

3.3 看训练进度
bash


tail -f /tmp/train_my_agent.log
RUN=$(ls -td fz_agent/runs/* | head -1)
echo $RUN
tail -n 5 "$RUN/train_epoch.csv"; tail -n 5 "$RUN/eval.csv"
结束产物：


fz_agent/bundles/my_agent/        # bundle.json + weights.pt（构建体）
fz_agent/runs/my_agent_<时间>/     # 日志/曲线 CSV
fz_agent/checkpoints/*.pth         # 检查点
4. 观看回放
回放由 Windows 的环境服务自动写到 chess-engine2/logs/runs/（每局一个 JSONL）。

4.1 Windows 图形回放（推荐，能看棋盘/单位/事件）
powershell
cd D:\...\chess-engine2
python -m tools.replay_viewer --dir logs/runs
支持播放/暂停、上下一帧、速度、进度条、事件面板；

训练进行中也能看（文件边写边追加，点“刷新目录”）。

4.2 WSL 终端回放（没有界面也能看）
前提：WSL 能访问 chess-engine2 目录（同一份代码或 /mnt/d/...）。
bash

cd /mnt/d/.../chess-engine2
LATEST=$(ls -t logs/runs/*.jsonl | head -1)
python -m logs.replay "$LATEST" --delay 0.2       # 自动播放字符画
python -m logs.replay "$LATEST" --step            # 回车逐帧
如果 WSL 里没有 chess-engine2：把 logs/runs/*.jsonl 复制到 WSL 再跑上面的命令。

4.3 不播放，只看摘要（批量分析）
bash

python3 -c "
from logs.replay_index import summarize_dir
import json
for s in summarize_dir('logs/runs')[:10]:
    print(s['name'], s['ticks'], s['winner'], s['frames'])
"
5.（可选）训练完成后上场
bash


# WSL：把构建体挂成推理服务
python -m fz_agent.serve --bundle-dir fz_agent/bundles --host 0.0.0.0 --port 8765

# Windows：人机对战（tkinter）或 Qt 播放器
python -m tools.play --red human --blue neural ^
  --neural-url http://<WSL_IP>:8765 --bundle my_agent
常见问题（按出现概率）
训练第一步连不上：Windows 服务必须 --host 0.0.0.0；WSL 用网关 IP；防火墙放行 8000；先跑 python -m fz_agent.net_utils --port 8000。
make_env 打印维度警告：把 --max-actions/--obs-entities 与配置 io 对齐（默认 32 / 12）。
回放目录是空的：确认 Windows 服务没加 --no-record，且训练确实发生了 /reset、/step（每局一个文件）。
Windows 上 GUI 的 dry-run 失败：那是没装 torch；只做「静态校验 + 保存」即可，dry-run 放 WSL 里跑。
训练太慢：每个 step 一次 HTTP。先 total_epochs=1, rollout_steps=64 跑通，再放大。
想换 GIS 经纬度场景：Windows 服务加 --scenario scenarios/geo_sim_demo.json；配置可继续用 ppo_chess.json（实体 6 个会补到 12 行）。