# DRLForge
A project try to cover a universal calling framework for neural networks and reinforcement learning as much as possible.

把"智能体"定义成一个四元组，全部走配置驱动：
```
agent:
  backbone: TransformerEncoder   # 来自模型库（8选1）
  heads: [policy_head, value_head]  # 任务头
  algorithm: PPO                 # 来自算法库（11选1）
  training_method:               # 演化训练（6选1或多选）
    - transfer: { init_from: pretrained.pt, freeze: [embedding] }
    - self_play: { opponent_pool_size: 20 }
```

# 离谱的要求
多种模型库、算法库，支持自助决策模型开发调试和参数导出。
典型深度强化学习算法数量不少于11种 DQN、DDQN、Dueling DQN、Rainbow、DDPG、PPO
基础神经模型数量不少于8种 GRU、LSTM、Transformer、Attention、ResNet
演化训练方法  迁移学习、对抗训练、增量持续训练、模型蒸馏轻量化、多任务联合、集成投票

# 文件结构
```
fz_agent/
├── __init__.py
├── backbone/       # 深度学习的基础网络自由构建
│   ├── __init__.py
│   ├── base.py          # BaseBackbone
│   ├── factory.py       # 网络构建器工厂
│   └── net_blocks.py    # 注册中心统一管理 8 种网络
├── rl/             # 强化学习方法
│   ├── __init__.py
│   ├── base.py          # BaseRLAlgorithm
│   ├── algos.py         # DQN PPO DDQN DDPG...实现
│   └── strategies.py    # 迁移学习、蒸馏等训练策略
├── stra/             # 训练策略
│   └── strategies.py    # 迁移学习、蒸馏等训练策略
├── core/
│   ├── __init__.py
│   └── ......           # 简易对抗环境
│
├── factory.py           # RLAssemblyFactory 总工厂
├── checkpoint.py        # save/load 带架构hash校验
├── api_demo.py          # 模拟后端接收JSON入口（可改成FastAPI接口）
└── gui_json_builder.py      # 模拟前端拖拽生成JSON（可改成Vue.js等前端框架）基于tinker写的简单生成json界面
```
对应：
深度学习网络
强化学习训练
训练策略


# 代码实现
在深度学习中，这种需求被称为神经架构搜索（NAS）的子集或动态模型构建。你需要一个“网络构建器”，能够像搭积木一样逐层堆叠，同时还要能将这种拓扑结构“序列化”记录下来，以便推理时复现。

我先为你解答**“现成的库”，然后给出定制化的高阶代码实现**。

一、 现成的库推荐
如果你不想自己造轮子，目前主流框架中有现成的解决方案：

PyTorch 原生 torch.nn.Sequential + add_module
最基础的堆叠方式，但缺乏灵活的动态配置和结构序列化能力。
NVIDIA 的 PyTorch Lightning
提供了比较方便的模块组织能力，但对于逐层动态配置也没有直接的开箱即用API。
AutoGluon / NNI (Microsoft)
这是专业的 NAS 库，支持以字典或配置文件定义网络结构并自动构建。但它们偏重于“搜索”，对于你这种只需要“组合与记录”的需求来说太重了。
Configurable MLP 库 (如 configargparse 结合自定义)
强化学习领域（如 CleanRL, SB3）通常自己实现一套 NatureCNN 或 MLP，但不支持异构层堆叠。
结论：现成的轻量级库无法完美契合“异构层自由堆叠+结构记录+强化学习适配”，自己写一套基于配置驱动的网络构建器是最优解，代码量不大且完全可控。

二、 定制化高阶架构：动态网络构建器
下面我为你设计一个 NetworkArchitect 类。它具备以下能力：
自由堆叠：通过配置列表，逐层添加不同的网络（LSTM, Attention, ResNet, MLP等）。
隐藏层与激活函数：动态指定每层的 hidden_dim 和 activation。
结构签名：自动生成网络结构的哈希值或JSON，用于模型存档与加载。
维度自动推导：层与层之间的 input_dim 和 output_dim 自动对齐，无需手动计算。

三、 结合之前的RL架构：如何锁定网络型号并复现
你提到“后续需要记录网络形状组合才能用上训练好的模型”，这正是 architecture_signature 的作用。
我们可以将 NetworkArchitect 无缝嵌入到上一轮的 BaseBackbone 接口中，并在保存模型时连带配置一起导出：


四、 设计亮点总结
维度自动推导：在 NetworkArchitect 的循环中，current_dim = hidden_dim 自动接管了层与层之间的维度对齐，你只需关心每一层的输出维度（即隐藏层维度），不用去算 LSTM 输出 64 后接 Attention 的输入是不是 64。
结构即配置：网络结构被完全抽象为 JSON 列表。你可以把这份 JSON 存入数据库，配合你的“自助决策模型开发调试”，实现前端拖拽生成 JSON，后端动态生成模型。
防呆机制：通过 MD5 哈希签名，在加载模型权重时进行严格的拓扑校验，避免因为前端传错配置导致 RuntimeError: size mismatch 破坏已有的训练成果。
 

## 执行问题解决
```python
# 方式1（开发时推荐）把当前目录加入python路径
export PYTHONPATH=/home/cq/RTS:$PYTHONPATH
python api_demo.py
# 或者
python rl_json_gui_full.py
```

## 解决问题记录
1. 让接受深度学习的模型还能接收强化学习，问题在batch这个维度上的输入。
2. 强化学习方法对特定神经网络是有适配的，并不能随便设置神经网络来进行强化学习。


## 输入输出契约与多模型接入

### 1. I/O 契约（训练与推理共用）
`fz_agent/env/base.py` 用 `IOSpec = (ObsSpec, ActionSpec)` 描述一个智能体看到什么、能做什么：

```json
"io": {
  "obs":    {"shape": [10, 3], "dim": 3, "entity_set": true,
             "names": ["dx", "dy", "hp_ratio"], "normalize": true, "clip": null},
  "action": {"space_type": "discrete", "dim": 3, "low": -1.0, "high": 1.0}
}
```
- `space_type` 决定算法族：`discrete` → PPO/A2C/REINFORCE/DQN 系；`continuous` → DDPG/TD3/SAC。
- 环境把 `io_spec` 交给 `main.make_env`，网络输入维度、动作头维度都由它推导，不再硬编码。
- 同一份 `io` 段落既写在训练配置里，也随构建体一起导出，保证"训练时怎么喂，部署时就怎么喂"。

### 2. 构建体（AgentBundle）：一个可独立加载/部署的模型单元
一个构建体 = **输入输出规格 + 网络结构/算法配置 + 权重文件**，落盘为一个目录：

```
bundles/<bundle_id>/
├── bundle.json    # {"version", "io_spec", "config", "metadata", "kind"}
└── weights.pt     # 各网络（policy/critic/q1/q2/target_*）的 state_dict
```

```python
from fz_agent.construct import AgentBundle, ModelPool, SelfPlayEnv

bundle = AgentBundle.from_agent(agent, cfg, env.io_spec, bundle_id="ppo_v3")
bundle.save("fz_agent/bundles/ppo_v3")

b = AgentBundle.load("fz_agent/bundles/ppo_v3")   # 构建网络 + 恢复权重
action = b.predict(obs, deterministic=True)        # 推理，返回 JSON 兼容值
```
`bundle.json` 里保存了完整 `config`，所以加载端不需要再传配置文件；`io_spec` 让部署端可校验输入 shape。

### 3. 多构建体同时接入（这是 2026.11.11 TODO 的答案）
对抗训练/自博弈/联赛里，同一时刻要跑"我方 N 个版本 + 对手 M 个版本"，不能再写死一个全局模型。

`ModelPool` 按 `bundle_id` 同时持有多个构建体，各自独立的网络、权重、动作空间：

```python
pool = ModelPool(device="cpu")
pool.load_dir("fz_agent/bundles")     # 递归扫描所有含 bundle.json 的目录
pool.ids()                            # ['ppo_v3', 'ddpg_v2', ...]

env = SelfPlayEnv(CombatEnv(), pool, opponent_id="ddpg_v2")
env.set_opponent("ppo_v3")            # 训练中途换对手，无需重建环境
for _ in range(steps):
    obs = env.reset()
    next_obs, reward, done, info = env.step(policy(obs))
```
关键点：
- 对手策略是 `lambda obs: pool.predict(opponent_id, obs)`，对手自己的 `io_spec` 与动作空间在池内独立解析，主环境只负责把对手动作转成战斗语义（`CombatEnv._coerce_team_action` 兼容离散 int 与连续 3 维向量）。
- 池子懒加载/常驻可选：`register(bundle)` 常驻，`get(id)` 取用时加载，显存/内存不够时可自行淘汰。
- 训练产物 `main.py rl-train` 结束会自动导出到 `fz_agent/bundles/<run_name>`，直接可被别的训练任务当对手使用。

### 4. 训练与利用流程
```bash
# 训练（按配置里的 algo / io 自动分派到离散/连续/off-policy/监督循环，结束后导出构建体）
python -m fz_agent.main rl-train -c fz_agent/configs/ppo_combat.json
python -m fz_agent.main rl-train -c fz_agent/configs/ddpg_combat.json
python -m fz_agent.main dl-train -c fz_agent/configs/dl_regression.json

# 利用（加载构建体或 config+weights 跑回合 / 跑预测）
python -m fz_agent.main rl-exp   --bundle fz_agent/bundles/ppo_combat --episodes 3
python -m fz_agent.main dl-exp   --model fz_agent/output/dl_models/dl_regression.pt --input data.npz --limit 5
```

### 5. 部署适配
内置一个零依赖（Python 标准库 `http.server`）的多构建体推理服务，用于把构建体挂成 HTTP 接口：

```bash
python -m fz_agent.serve --bundle-dir fz_agent/bundles --host 0.0.0.0 --port 8765
```
```bash
GET  /health                        # {"status":"ok","bundles":[...],"count":N}
GET  /bundles                       # 构建体列表与各自的 io_spec
POST /act                           # {"bundle_id":"ppo_v3","obs":[[...]]}
POST /act  {"requests":[{...},{...}]}  # 批量，返回 {"responses":[...]}
```
对应关系：`/act` → `InferenceService.handle` → `ModelPool.predict` → `AgentBundle.predict`。要换成 FastAPI/gRPC 只需复写 `make_handler`；接真实 C++ 引擎时，实现 `CombatEnvProxy` 的 `reset/step/io_dict` 即可，训练侧代码无需改动。

## 日志
2026.11.5 确定强化学习与深度学习分别实现的方案。
2026.11.6 完成模拟环境的构建，实现DRL跑通有效果。
2026.11.9 框架还有一些填充工作，比如所有强化学习算法实现细节的完成。后续是输入的向量化怎么做一下，自己调优即可，不暴露接口。TODO 训练策略还差些东西怎么给原上。
2026.11.10 完成强化学习算法的json定义，后续是C++接口的实现、python环境部署。
2026.11.11 TODO 之前隐隐有的问题，如何同时处理多个模型的接入？？？ → 已解决，见上节「多构建体同时接入」：用 AgentBundle 把「I/O 规格 + 网络结构配置 + 权重」打包，用 ModelPool 按 bundle_id 并发持有，SelfPlayEnv 动态换对手，serve.py 对外提供多模型推理接口。