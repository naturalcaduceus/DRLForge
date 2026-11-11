JSON 配置“用哪个 backbone + 哪个算法 + 哪种训练策略”

顶层字段：
  meta      运行名
  io        输入输出契约：obs(shape/dim/entity_set/names/normalize/clip) + action(space_type/dim/low/high)
            space_type=discrete → PPO/A2C/REINFORCE/DQN 系；continuous → DDPG/TD3/SAC
  env       环境参数（type/config/action_space_type/action_dim，可选 io 覆盖）
  networks  网络角色：每个角色可用 network_stack(顺序堆叠) 或 network_graph(DAG 非堆叠)
            角色名跟算法走：policy / q_net / actor+critic / q1+q2 ...
  algo      算法名与超参（device/learning_rate/gamma/batch_size ...）
  strategy  可选：迁移学习 / 对抗训练 / 蒸馏 / 集成 / 持续训练
  train     训练循环参数（total_epochs / total_steps / rollout_steps / eval_interval ...）
  data      仅 dl-train：{path, x_key, y_key, task, batch_size, epochs, lr}

network_graph 示例：
  {"network_graph": {
     "inputs": {"obs": ["enc"]},
     "nodes": [
       {"id": "enc",  "type": "mlp", "config": {...}, "inputs": ["__input__"]},
       {"id": "res",  "type": "resnet", "config": {...}, "inputs": ["enc"]},
       {"id": "merge","type": "add", "inputs": ["enc", "res"]},
       {"id": "out",  "type": "mlp", "config": {...}, "inputs": ["merge"]}
     ],
     "output": "out",
     "head": "actor_critic"
  }}

产物：
  训练 → runs/<name>_<时间戳>/（日志/CSV/曲线）
       → fz_agent/bundles/<name>/（bundle.json + weights.pt，可直接部署）
  深度学习 → fz_agent/output/dl_models/<name>.pt
