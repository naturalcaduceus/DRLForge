import tkinter as tk
from tkinter import ttk, messagebox
import json
# import pyperclip

class RLNetJsonBuilder:
    def __init__(self, root):
        self.root = root
        self.root.title("低代码强化学习模型配置生成器｜自助决策模型开发调试")
        self.root.geometry("1150x820")

        # -------------------------- 配置常量（和后端对齐） --------------------------
        self.rl_algo_list = [
            "DQN", "DDQN", "DuelingDQN", "Rainbow",
            "DDPG", "TD3", "SAC", "PPO",
            "A2C", "A3C", "REINFORCE"
        ]
        self.net_module_list = [
            "MLP", "LSTM", "GRU", "SelfAttention",
            "TransformerBlock", "ResBlock", "Conv1d", "Conv2d"
        ]
        self.strategy_all = [
            ("transfer", "迁移学习"),
            ("adversarial", "对抗训练"),
            ("continual", "增量持续训练"),
            ("distill", "模型蒸馏轻量化"),
            ("multitask", "多任务联合"),
            ("ensemble", "集成投票")
        ]
        self.act_options = ["ReLU", "GELU", "SiLU", "Tanh", "Sigmoid", "LeakyReLU"]

        self.layer_list = []

        # 算法参数缓存
        self.algo_params_var = {}
        # 策略勾选变量
        self.strategy_vars = {}
        for sid, name in self.strategy_all:
            self.strategy_vars[sid] = tk.BooleanVar(value=False)

        # ===================== 顶部：基础环境配置 =====================
        frame_env = ttk.LabelFrame(root, text="环境 & 基础输入输出")
        frame_env.pack(fill=tk.X, padx=8, pady=4)

        ttk.Label(frame_env, text="obs_dim(观测):").grid(row=0, column=0, padx=4, pady=4)
        self.var_obs_dim = tk.StringVar(value="16")
        ttk.Entry(frame_env, textvariable=self.var_obs_dim, width=10).grid(row=0, column=1, padx=4)

        ttk.Label(frame_env, text="act_dim(动作):").grid(row=0, column=2, padx=4, pady=4)
        self.var_act_dim = tk.StringVar(value="4")
        ttk.Entry(frame_env, textvariable=self.var_act_dim, width=10).grid(row=0, column=3, padx=4)

        ttk.Label(frame_env, text="max_episode:").grid(row=0, column=4, padx=4, pady=4)
        self.var_max_ep = tk.StringVar(value="10000")
        ttk.Entry(frame_env, textvariable=self.var_max_ep, width=10).grid(row=0, column=5, padx=4)

        # ===================== RL算法选择 + 动态参数面板 =====================
        frame_algo = ttk.LabelFrame(root, text="强化学习算法选择（共11种）")
        frame_algo.pack(fill=tk.X, padx=8, pady=4)

        ttk.Label(frame_algo, text="算法:").grid(row=0, column=0, padx=4)
        self.var_algo_name = tk.StringVar(value="PPO")
        cb_algo = ttk.Combobox(frame_algo, textvariable=self.var_algo_name, values=self.rl_algo_list, width=14)
        cb_algo.grid(row=0, column=1, padx=4)
        cb_algo.bind("<<ComboboxSelected>>", self.refresh_algo_panel)

        self.algo_panel = ttk.Frame(frame_algo)
        self.algo_panel.grid(row=0, column=2, padx=10)
        self.refresh_algo_panel()

        # ===================== 训练策略（多选） =====================
        frame_strategy = ttk.LabelFrame(root, text="演化训练策略（可多选）")
        frame_strategy.pack(fill=tk.X, padx=8, pady=4)
        col = 0
        for sid, name in self.strategy_all:
            cb = ttk.Checkbutton(frame_strategy, text=name, variable=self.strategy_vars[sid])
            cb.grid(row=0, column=col, padx=3)
            col += 1

        # ===================== 添加网络层模块 =====================
        frame_layer_add = ttk.LabelFrame(root, text="添加网络积木（8种基础模块）")
        frame_layer_add.pack(fill=tk.X, padx=8, pady=4)

        ttk.Label(frame_layer_add, text="模块类型:").grid(row=0, column=0, padx=4)
        self.var_module_type = tk.StringVar(value="MLP")
        cb_mod = ttk.Combobox(frame_layer_add, textvariable=self.var_module_type, values=self.net_module_list, width=16)
        cb_mod.grid(row=0, column=1, padx=4)
        cb_mod.bind("<<ComboboxSelected>>", self.refresh_layer_param_panel)

        self.layer_param_frame = ttk.Frame(frame_layer_add)
        self.layer_param_frame.grid(row=0, column=2, padx=8)
        self.layer_param_vars = {}
        self.refresh_layer_param_panel()

        ttk.Button(frame_layer_add, text="添加到堆叠", command=self.add_layer).grid(row=0, column=12, padx=6)

        # ===================== 层堆叠列表 =====================
        frame_tree = ttk.LabelFrame(root, text="网络堆叠顺序：从上到下 = 输入流向输出")
        frame_tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        cols = ("type", "hidden", "act", "extra")
        self.tree = ttk.Treeview(frame_tree, columns=cols, show="headings", height=9)
        self.tree.heading("type", text="模块类型")
        self.tree.heading("hidden", text="hidden/channel")
        self.tree.heading("act", text="激活")
        self.tree.heading("extra", text="额外参数")
        self.tree.column("type", width=130)
        self.tree.column("hidden", width=90)
        self.tree.column("act", width=90)
        self.tree.column("extra", width=240)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll = ttk.Scrollbar(frame_tree, orient=tk.VERTICAL, command=self.tree.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=scroll.set)

        frame_op = ttk.Frame(root)
        frame_op.pack(fill=tk.X, padx=8)
        ttk.Button(frame_op, text="上移", command=self.move_up).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame_op, text="下移", command=self.move_down).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame_op, text="删除选中", command=self.del_layer).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame_op, text="清空堆叠", command=self.clear_layers).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame_op, text="载入演示样例", command=self.load_demo).pack(side=tk.LEFT, padx=10)

        # ===================== JSON预览 =====================
        frame_json = ttk.LabelFrame(root, text="导出配置JSON（直接传入后端build_rl_from_json_config）")
        frame_json.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        self.txt_json = tk.Text(frame_json, height=14, font=("Consolas", 9))
        self.txt_json.pack(fill=tk.BOTH, expand=True)

        frame_bottom = ttk.Frame(root)
        frame_bottom.pack(pady=6)
        ttk.Button(frame_bottom, text="生成JSON", command=self.gen_json).grid(row=0, column=0, padx=6)
        ttk.Button(frame_bottom, text="复制JSON", command=self.copy_json).grid(row=0, column=1, padx=6)
        ttk.Button(frame_bottom, text="【演示】直接POST后端", command=self.call_backend).grid(row=0, column=2, padx=6)
        self.var_api_url = tk.StringVar(value="http://127.0.0.1:8000/build_rl_model")
        ttk.Entry(frame_bottom, textvariable=self.var_api_url, width=36).grid(row=0, column=3, padx=8)

    def refresh_algo_panel(self, evt=None):
        for w in self.algo_panel.winfo_children():
            w.destroy()
        self.algo_params_var.clear()
        algo = self.var_algo_name.get()
        # 每个算法专属参数定义
        algo_cfg = {
            "DQN": [("lr", "1e-4"), ("gamma", "0.98"), ("epsilon_start", "0.9"), ("epsilon_end", "0.05")],
            "DDQN": [("lr", "1e-4"), ("gamma", "0.98")],
            "DuelingDQN": [("lr", "1e-4"), ("gamma", "0.98")],
            "Rainbow": [("lr", "1e-4"), ("gamma", "0.99"), ("n_step", "3")],
            "DDPG": [("lr_actor", "1e-4"), ("lr_critic", "3e-4"), ("gamma", "0.98"), ("tau", "0.005")],
            "TD3": [("lr_actor", "1e-4"), ("lr_critic", "3e-4"), ("gamma", "0.98"), ("tau", "0.005"), ("policy_noise", "0.2")],
            "SAC": [("lr", "3e-4"), ("gamma", "0.99"), ("alpha", "0.2")],
            "PPO": [("lr", "3e-4"), ("gamma", "0.99"), ("gae_lambda", "0.95"), ("clip_ratio", "0.2")],
            "A2C": [("lr", "3e-4"), ("gamma", "0.99")],
            "A3C": [("lr", "3e-4"), ("gamma", "0.99"), ("num_workers", "4")],
            "REINFORCE": [("lr", "1e-3"), ("gamma", "0.98")]
        }
        items = algo_cfg.get(algo, [])
        for idx, (k, v) in enumerate(items):
            ttk.Label(self.algo_panel, text=f"{k}:").grid(row=0, column=idx*2)
            var = tk.StringVar(value=v)
            self.algo_params_var[k] = var
            ttk.Entry(self.algo_panel, textvariable=var, width=9).grid(row=0, column=idx*2+1)

    def refresh_layer_param_panel(self, evt=None):
        for w in self.layer_param_frame.winfo_children():
            w.destroy()
        self.layer_param_vars.clear()
        mod = self.var_module_type.get()
        cfg = {
            "MLP": [("hidden_dim", "64"), ("activation", "ReLU")],
            "LSTM": [("hidden_dim", "64"), ("num_layers", "1"), ("activation", "Tanh")],
            "GRU": [("hidden_dim", "64"), ("num_layers", "1")],
            "SelfAttention": [("hidden_dim", "64"), ("nhead", "4")],
            "TransformerBlock": [("hidden_dim", "128"), ("nhead", "4"), ("ff_dim", "256")],
            "ResBlock": [("hidden_dim", "64"), ("activation", "ReLU")],
            "Conv1d": [("out_channels", "32"), ("kernel_size", "3"), ("stride", "1")],
            "Conv2d": [("out_channels", "32"), ("kernel_size", "3"), ("stride", "1")]
        }
        items = cfg.get(mod, [])
        for idx, (k, v) in enumerate(items):
            ttk.Label(self.layer_param_frame, text=f"{k}:").grid(row=0, column=idx*2)
            var = tk.StringVar(value=v)
            self.layer_param_vars[k] = var
            if k == "activation":
                cb = ttk.Combobox(self.layer_param_frame, textvariable=var, values=self.act_options, width=8)
                cb.grid(row=0, column=idx*2+1)
            else:
                ttk.Entry(self.layer_param_frame, textvariable=var, width=8).grid(row=0, column=idx*2+1)

    def add_layer(self):
        mod_type = self.var_module_type.get()
        d = {"type": mod_type}
        try:
            for k, var in self.layer_param_vars.items():
                val = var.get().strip()
                if k == "activation":
                    d[k] = val
                else:
                    d[k] = int(val)
        except Exception as e:
            messagebox.showerror("参数错误", "数值必须是整数！")
            return
        self.layer_list.append(d)
        self.refresh_tree()

    def refresh_tree(self):
        for i in self.tree.get_children():
            self.tree.delete(i)
        for item in self.layer_list:
            t = item["type"]
            hidden = item.get("hidden_dim", item.get("out_channels", "-"))
            act = item.get("activation", "-")
            others = ",".join([f"{k}={v}" for k,v in item.items() if k not in ["type","hidden_dim","out_channels","activation"]])
            self.tree.insert("", tk.END, values=(t, hidden, act, others))

    def get_selected_idx(self):
        sel = self.tree.selection()
        if not sel: return None
        return self.tree.index(sel[0])
    def move_up(self):
        i = self.get_selected_idx()
        if i is None or i <= 0: return
        self.layer_list[i], self.layer_list[i-1] = self.layer_list[i-1], self.layer_list[i]
        self.refresh_tree()
    def move_down(self):
        i = self.get_selected_idx()
        if i is None or i >= len(self.layer_list)-1: return
        self.layer_list[i], self.layer_list[i+1] = self.layer_list[i+1], self.layer_list[i]
        self.refresh_tree()
    def del_layer(self):
        i = self.get_selected_idx()
        if i is None: return
        del self.layer_list[i]
        self.refresh_tree()
    def clear_layers(self):
        self.layer_list.clear()
        self.refresh_tree()

    def gen_json(self):
        # 读取选中策略
        selected_strategies = []
        strategy_meta = {
            "transfer": {"name":"TransferLearning", "params":{"freeze_n_layer":2}},
            "adversarial": {"name":"AdversarialTrain", "params":{"eps":0.01}},
            "continual": {"name":"ContinualTrain", "params":{"buffer_size":10000}},
            "distill": {"name":"Distillation", "params":{"temp":2.0, "alpha":0.7}},
            "multitask": {"name":"MultiTaskJoint", "params":{"task_num":2}},
            "ensemble": {"name":"EnsembleVote", "params":{"model_count":3}}
        }
        for sid, meta in strategy_meta.items():
            if self.strategy_vars[sid].get():
                selected_strategies.append(meta)

        # 算法参数
        algo_param_dict = {}
        for k, var in self.algo_params_var.items():
            v = var.get().strip()
            try:
                if "." in v or "e-" in v:
                    algo_param_dict[k] = float(v)
                else:
                    algo_param_dict[k] = int(v)
            except:
                algo_param_dict[k] = v

        payload = {
            "obs_dim": int(self.var_obs_dim.get()),
            "act_dim": int(self.var_act_dim.get()),
            "max_episode": int(self.var_max_ep.get()),
            "layer_configs": self.layer_list,
            "algorithm": {
                "name": self.var_algo_name.get(),
                "params": algo_param_dict
            },
            "train_strategies": selected_strategies
        }
        text = json.dumps(payload, indent=2, ensure_ascii=False)
        self.txt_json.delete(1.0, tk.END)
        self.txt_json.insert(tk.END, text)
        return text

    def copy_json(self):
        txt = self.gen_json()
        try:
            pyperclip.copy(txt)
            messagebox.showinfo("OK", "JSON已复制")
        except:
            messagebox.showwarning("失败", "pyperclip不可用，请手动复制文本框")

    def call_backend(self):
        import requests
        txt = self.gen_json()
        url = self.var_api_url.get()
        try:
            resp = requests.post(url, json=json.loads(txt), timeout=10)
            messagebox.showinfo("后端返回", f"code={resp.status_code}\n{resp.text[:500]}")
        except Exception as e:
            messagebox.showerror("请求失败", str(e))

    def load_demo(self):
        self.var_obs_dim.set("24")
        self.var_act_dim.set("6")
        self.var_max_ep.set("20000")
        self.var_algo_name.set("PPO")
        # 默认勾选迁移+蒸馏
        self.strategy_vars["transfer"].set(True)
        self.strategy_vars["distill"].set(True)
        self.layer_list = [
            {"type":"LSTM", "hidden_dim":64, "num_layers":1, "activation":"Tanh"},
            {"type":"TransformerBlock", "hidden_dim":128, "nhead":4, "ff_dim":256},
            {"type":"ResBlock", "hidden_dim":128, "activation":"GELU"},
            {"type":"MLP", "hidden_dim":64, "activation":"ReLU"}
        ]
        self.refresh_tree()
        self.refresh_algo_panel()
        self.gen_json()

if __name__ == "__main__":
    root = tk.Tk()
    app = RLNetJsonBuilder(root)
    root.mainloop()
