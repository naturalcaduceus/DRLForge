"""
模拟后端接口，接收前端低代码平台POST过来的JSON配置
如果接入FastAPI，把这个函数改成 @app.post("/build_rl_model")
"""
from config import gopt

from fz_agent.backbone.factory import build_rl_from_json_config
from checkpoint import save_checkpoint, load_checkpoint
import torch

'''
# =====================
# 📌前端传给后端的JSON样例（低代码拖拽生成）
# =====================
REQUEST_JSON = {
    "input_dim": 16,
    "output_dim": 4,
    "layer_configs": [
        {"type": "LSTM", "hidden_dim": 64, "activation": "Tanh"},
        {"type": "Attention", "hidden_dim": 32, "activation": "ReLU", "nhead":4},
        {"type": "MLP", "hidden_dim": 128, "activation": "GELU"}
    ],
    "algorithm": {
        "name": "DQN",
        "params": {"lr":1e-4, "gamma":0.98}
    },
    "strategies":[
        {
            "name":"TransferLearningStrategy",
            "params":{"freeze_layers":2}
        }
    ]
}
'''

import json
import os
# 定义文件路径
json_file_path = os.path.join(os.path.dirname(__file__), '..', 'docs', '1.json')

try:
    with open(json_file_path, 'r', encoding='utf-8') as f:
        REQUEST_JSON = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"找不到指定的JSON文件，请检查路径是否正确：{json_file_path}")
except json.JSONDecodeError:
    raise ValueError(f"JSON文件格式有误，请检查文件内容：{json_file_path}")




def handle_build_model(req_json: dict):
    """模拟后端接口处理函数"""
    agent = build_rl_from_json_config(req_json)
    dummy_x = torch.randn(4, req_json["input_dim"])
    out = agent.backbone(dummy_x)
    return {
        "ok": True,
        "arch_hash": agent.backbone.architecture_signature,
        "arch_config": agent.backbone.get_config_dict(),
        "output_test_shape": list(out.shape)
    }


if __name__ == "__main__":
    resp = handle_build_model(REQUEST_JSON)
    print("后端返回结果：")
    import json
    print(json.dumps(resp, indent=2, ensure_ascii=False))
    agent = build_rl_from_json_config(REQUEST_JSON)
    save_checkpoint(agent, gopt.outputpath + "agent.ckpt")
    print("\n模型保存完成")
    
    # 输出模型配置
    print("\n模型配置：")
    print(agent.backbone.get_config_dict())
