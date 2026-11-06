import json
import torch
from net_blocks import build_network_by_type
from fz_agent.rl.algos import build_rl_algorithm
from fz_agent.stra.strategy_wrapper import wrap_agent_with_strategy

# 先堆叠 net_blocks，再实例化 RL 算法，再可选套训练策略包装器，**完整链路 JSON → 网络组装 → RL Agent → 训练策略包装**
def build_rl_from_json_config(raw_cfg: dict):
    """
    核心组装入口，给FastAPI / Tk前端生成的JSON调用
    raw_cfg结构约定：
    {
        "network_stack": [ {"type":"mlp", "config": {...}}, ... ],
        "algo": {"name":"ppo", "config": {...}},
        "strategy": {"name": "distill_light", "config": {...}, "extra": {...}} # 可选
    }
    """
    # Step1：顺序堆叠网络块（你低代码层堆叠核心）
    blocks = []
    for block_spec in raw_cfg["network_stack"]:
        net = build_network_by_type(block_spec["type"], block_spec["config"])
        blocks.append(net)
    # 这里你可以自行实现 Sequential 串联；复杂分支（Dueling、Actor-Critic）在algo内部拆解
    # TODO 这里没有做网络分支的拆解，界面上的json可以后续再扩展
    backbone = torch.nn.Sequential(*blocks)

    # Step2：构建RL算法实例
    algo_cfg = raw_cfg["algo"]["config"]
    algo_name = raw_cfg["algo"]["name"]
    agent = build_rl_algorithm(algo_name, algo_cfg, backbone)

    # Step3：可选包装演化训练策略
    if "strategy" in raw_cfg and raw_cfg["strategy"]["name"]:
        st_cfg = raw_cfg["strategy"]["config"]
        extra_args = raw_cfg["strategy"].get("extra", {})
        agent = wrap_agent_with_strategy(raw_cfg["strategy"]["name"], agent, st_cfg, **extra_args)

    return agent, backbone, blocks
