# DRLForge
A project try to cover a universal calling framework for neural networks and reinforcement learning as much as possible.


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


## 日志
2026.11.5 确定强化学习与深度学习分别实现的方案。
2026.11.6 完成模拟环境的构建，实现DRL跑通有效果。
2026.11.9 框架还有一些填充工作，比如所有强化学习算法实现细节的完成。后续是输入的向量化怎么做一下，自己调优即可，不暴露接口。