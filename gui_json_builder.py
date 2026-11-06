"""低代码强化学习模型配置生成器（Tkinter）。

生成的 JSON 可以直接传给后端 `fz_agent.dl.factory.build_rl_from_json_config`。

JSON 输出格式约定（与后端对齐）：
{
    "network_stack": [                        # 堆叠式网络
        {"type": "mlp", "config": {...}},
        {"type": "lstm", "config": {...}},
    ],
    "network_graph": {...},                   # （预留）DAG/分支式，暂不生成
    "algo": {
        "name": "ppo",                        # 小写，与 rl/__init__.py 的 registry 对齐
        "config": {...}
    },
    "strategy": {                             # 可选
        "name": "distill_light",
        "config": {...},
        "extra": {...}
    }
}
"""

import json
import tkinter as tk
from tkinter import ttk, messagebox


# ===================== 后端对齐的注册表 =====================

# 与 fz_agent/dl/blocks/__init__.py 的 NET_REGISTRY key 对齐
NET_MODULES = [
    ("MLP",          "mlp"),
    ("CNN",          "cnn"),
    ("Conv1d",       "conv1d"),
    ("LSTM",         "lstm"),
    ("GRU",          "gru"),
    ("SelfAttention","attention"),
    ("Transformer",  "transformer"),
    ("ResNet",       "resnet"),
]

# 每个模块的参数定义（参数名 → (类型, 默认值)）
# type 可选: int / float / str
NET_PARAMS = {
    "mlp": [
        ("in_dim",     int,  16),
        ("hidden_dims",str,  "64,64"),   # 逗号分隔列表，生成 JSON 时转 list
        ("out_dim",    int,  4),
        ("activation", str,  "ReLU"),
    ],
    "cnn": [
        ("in_channels",int,  3),
        ("channels",   str,  "32,64"),
        ("kernel_sizes",str,  "3,3"),
        ("strides",    str,  "1,1"),
        ("padding",    int,  0),
        ("activation", str,  "ReLU"),
    ],
    "conv1d": [
        ("in_channels",int,  8),
        ("channels",   str,  "32,32"),
        ("kernel_sizes",str,  "3,3"),
        ("strides",    str,  "1,1"),
        ("activation", str,  "ReLU"),
    ],
    "lstm": [
        ("input_size", int,  16),
        ("hidden_size",int,  64),
        ("num_layers", int,  1),
        ("out_dim",    int,  32),
        ("batch_first",str,  "True"),
        ("bidirectional",str,"False"),
    ],
    "gru": [
        ("input_size", int,  16),
        ("hidden_size",int,  64),
        ("num_layers", int,  1),
        ("out_dim",    int,  32),
        ("batch_first",str,  "True"),
        ("bidirectional",str,"False"),
    ],
    "attention": [
        ("embed_dim", int,  64),
        ("num_heads", int,  4),
    ],
    "transformer": [
        ("d_model",       int,  64),
        ("nhead",         int,  4),
        ("dim_feedforward",int,128),
        ("num_layers",    int,  2),
        ("dropout",       float,0.1),
    ],
    "resnet": [
        ("in_channels", int,  32),
        ("out_channels",int,  64),
        ("stride",      int,  1),
    ],
}

# 算法列表（小写，与 rl/__init__.py 的 ALGO_REGISTRY 对齐）
RL_ALGOS = [
    "ppo", "dqn", "ddqn", "dueling_dqn", "rainbow",
    "ddpg", "sac", "td3", "a2c", "reinforce", "iqn",
]

# 每个算法的参数定义
ALGO_PARAMS = {
    "ppo": [
        ("learning_rate", float, 3e-4),
        ("gamma",         float, 0.99),
        ("gae_lambda",    float, 0.95),
        ("clip_coef",     float, 0.2),
        ("ent_coef",      float, 0.01),
        ("vf_coef",       float, 0.5),
        ("max_grad_norm", float, 0.5),
    ],
    "dqn": [
        ("learning_rate",  float, 1e-3),
        ("gamma",          float, 0.99),
        ("tau",            float, 0.005),
        ("epsilon_start",  float, 1.0),
        ("epsilon_end",    float, 0.05),
        ("epsilon_decay",  float, 0.995),
        ("batch_size",     int,   64),
    ],
    "ddqn":               ALGO_PARAMS["dqn"],
    "dueling_dqn":        ALGO_PARAMS["dqn"],
    "rainbow":            [("learning_rate", float, 1e-4), ("gamma", float, 0.99)],
    "ddpg":               [("learning_rate", float, 1e-4), ("gamma", float, 0.98)],
    "sac":                [("learning_rate", float, 3e-4), ("gamma", float, 0.99), ("alpha", float, 0.2)],
    "td3":                [("learning_rate", float, 1e-4), ("gamma", float, 0.98)],
    "a2c":                [("learning_rate", float, 3e-4), ("gamma", float, 0.99)],
    "reinforce":          [("learning_rate", float, 1e-3), ("gamma", float, 0.98)],
    "iqn":                [("learning_rate", float, 1e-4), ("gamma", float, 0.99)],
}

# 训练策略（可选，与 stra/strategy_wrapper.py 的 STRATEGY_REGISTRY 对齐）
TRAIN_STRATEGIES = [
    ("transfer_learning",    "迁移学习"),
    ("adversarial_train",    "对抗训练"),
    ("incremental_continual","增量持续训练"),
    ("distill_light",        "模型蒸馏轻量化"),
    ("multitask_joint",      "多任务联合"),
    ("ensemble_vote",        "集成投票"),
]

ACT_OPTIONS = ["ReLU", "GELU", "SiLU", "Tanh", "Sigmoid", "LeakyReLU"]


# ===================== GUI 主体 =====================

class RLNetJsonBuilder:
    def __init__(self, root):
        self.root = root
        self.root.title("低代码 RL 模型配置生成器")
        self.root.geometry("1200x850")

        # 内部状态
        self.layer_list = []                      # [{"type": "mlp", "config": {...}}]
        self.algo_param_vars = {}
        self.strategy_var = tk.StringVar(value="")
        self.layer_param_vars = {}

        # ===================== RL 算法选择 =====================
        frame_algo = ttk.LabelFrame(root, text="强化学习算法")
        frame_algo.pack(fill=tk.X, padx=8, pady=4)

        ttk.Label(frame_algo, text="算法:").grid(row=0, column=0, padx=4)
        self.var_algo = tk.StringVar(value="ppo")
        cb_algo = ttk.Combobox(
            frame_algo, textvariable=self.var_algo,
            values=RL_ALGOS, width=14, state="readonly",
        )
        cb_algo.grid(row=0, column=1, padx=4)
        cb_algo.bind("<<ComboboxSelected>>", self._refresh_algo_panel)

        self.algo_panel = ttk.Frame(frame_algo)
        self.algo_panel.grid(row=0, column=2, padx=10)
        self._refresh_algo_panel()

        # ===================== 训练策略（可选） =====================
        frame_strategy = ttk.LabelFrame(root, text="训练策略（可选，最多选一个）")
        frame_strategy.pack(fill=tk.X, padx=8, pady=4)
        ttk.Radiobutton(
            frame_strategy, text="(无)",
            variable=self.strategy_var, value="",
        ).grid(row=0, column=0, padx=3)
        for i, (sid, name) in enumerate(TRAIN_STRATEGIES, start=1):
            ttk.Radiobutton(
                frame_strategy, text=name,
                variable=self.strategy_var, value=sid,
            ).grid(row=0, column=i, padx=3)

        # ===================== 添加网络层 =====================
        frame_layer_add = ttk.LabelFrame(root, text="添加网络积木（后端按堆叠顺序从前向后传递）")
        frame_layer_add.pack(fill=tk.X, padx=8, pady=4)

        ttk.Label(frame_layer_add, text="模块:").grid(row=0, column=0, padx=4)
        self.var_module = tk.StringVar(value="MLP")
        cb_mod = ttk.Combobox(
            frame_layer_add, textvariable=self.var_module,
            values=[name for name, _ in NET_MODULES],
            width=16, state="readonly",
        )
        cb_mod.grid(row=0, column=1, padx=4)
        cb_mod.bind("<<ComboboxSelected>>", self._refresh_layer_panel)

        self.layer_param_frame = ttk.Frame(frame_layer_add)
        self.layer_param_frame.grid(row=0, column=2, padx=8)
        self._refresh_layer_panel()

        ttk.Button(frame_layer_add, text="添加到堆叠", command=self._add_layer).grid(row=0, column=12, padx=6)

        # ===================== 层堆叠列表 =====================
        frame_tree = ttk.LabelFrame(root, text="网络堆叠顺序（上→下 = 输入→输出）")
        frame_tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        cols = ("type", "params")
        self.tree = ttk.Treeview(frame_tree, columns=cols, show="headings", height=8)
        self.tree.heading("type", text="模块类型")
        self.tree.heading("params", text="config")
        self.tree.column("type", width=120)
        self.tree.column("params", width=800)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll = ttk.Scrollbar(frame_tree, orient=tk.VERTICAL, command=self.tree.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=scroll.set)

        frame_op = ttk.Frame(root)
        frame_op.pack(fill=tk.X, padx=8)
        for text, cmd in [
            ("上移", self._move_up),
            ("下移", self._move_down),
            ("删除选中", self._del_layer),
            ("清空堆叠", self._clear_layers),
            ("载入演示", self._load_demo),
        ]:
            ttk.Button(frame_op, text=text, command=cmd).pack(side=tk.LEFT, padx=2)

        # ===================== JSON 预览 =====================
        frame_json = ttk.LabelFrame(root, text="导出 JSON（直接传给 build_rl_from_json_config）")
        frame_json.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        self.txt_json = tk.Text(frame_json, height=14, font=("Consolas", 9))
        self.txt_json.pack(fill=tk.BOTH, expand=True)

        frame_bottom = ttk.Frame(root)
        frame_bottom.pack(pady=6)
        ttk.Button(frame_bottom, text="生成 JSON", command=self._gen_json).grid(row=0, column=0, padx=6)
        ttk.Button(frame_bottom, text="复制到剪贴板", command=self._copy_json).grid(row=0, column=1, padx=6)
        ttk.Button(frame_bottom, text="保存为文件", command=self._save_json).grid(row=0, column=2, padx=6)

    # ==============================================================
    # 刷新面板
    # ==============================================================

    def _refresh_algo_panel(self, evt=None):
        for w in self.algo_panel.winfo_children():
            w.destroy()
        self.algo_param_vars.clear()
        algo = self.var_algo.get()
        params = ALGO_PARAMS.get(algo, [])
        for idx, (k, typ, default) in enumerate(params):
            ttk.Label(self.algo_panel, text=f"{k}:").grid(row=0, column=idx * 2)
            var = tk.StringVar(value=str(default))
            self.algo_param_vars[k] = (typ, var)
            ttk.Entry(self.algo_panel, textvariable=var, width=9).grid(row=0, column=idx * 2 + 1)

    def _refresh_layer_panel(self, evt=None):
        for w in self.layer_param_frame.winfo_children():
            w.destroy()
        self.layer_param_vars.clear()
        label, net_key = self._current_net_key()
        params = NET_PARAMS.get(net_key, [])
        for idx, (k, typ, default) in enumerate(params):
            ttk.Label(self.layer_param_frame, text=f"{k}:").grid(row=0, column=idx * 2)
            var = tk.StringVar(value=str(default))
            self.layer_param_vars[k] = (typ, var)
            if k == "activation":
                cb = ttk.Combobox(
                    self.layer_param_frame, textvariable=var,
                    values=ACT_OPTIONS, width=8, state="readonly",
                )
                cb.grid(row=0, column=idx * 2 + 1)
            else:
                ttk.Entry(self.layer_param_frame, textvariable=var, width=10).grid(row=0, column=idx * 2 + 1)

    def _current_net_key(self):
        """从下拉框选中的显示名拿到后端 registry key。"""
        display = self.var_module.get()
        for name, key in NET_MODULES:
            if name == display:
                return name, key
        return display, display.lower()

    # ==============================================================
    # 层操作
    # ==============================================================

    def _parse_param_value(self, raw, typ):
        """把字符串按 typ 解析成正确的 Python 类型。"""
        s = raw.strip()
        if typ is int:
            return int(s)
        if typ is float:
            return float(s)
        if typ is str:
            # 布尔值
            if s.lower() in ("true", "false"):
                return s.lower() == "true"
            # 逗号分隔的列表
            if "," in s:
                parts = [p.strip() for p in s.split(",")]
                # 尝试每个元素为 int 或 float
                result = []
                for p in parts:
                    try:
                        result.append(int(p))
                    except ValueError:
                        try:
                            result.append(float(p))
                        except ValueError:
                            result.append(p)
                return result
            return s
        return s

    def _add_layer(self):
        display_name, net_key = self._current_net_key()
        config = {}
        for k, (typ, var) in self.layer_param_vars.items():
            try:
                config[k] = self._parse_param_value(var.get(), typ)
            except Exception as e:
                messagebox.showerror("参数错误", f"{k} 解析失败: {e}")
                return
        self.layer_list.append({"type": net_key, "config": config})
        self._refresh_tree()

    def _refresh_tree(self):
        for i in self.tree.get_children():
            self.tree.delete(i)
        for item in self.layer_list:
            self.tree.insert(
                "", tk.END,
                values=(item["type"], json.dumps(item["config"], ensure_ascii=False)),
            )

    def _selected_idx(self):
        sel = self.tree.selection()
        if not sel:
            return None
        return self.tree.index(sel[0])

    def _move_up(self):
        i = self._selected_idx()
        if i is None or i <= 0:
            return
        self.layer_list[i], self.layer_list[i - 1] = self.layer_list[i - 1], self.layer_list[i]
        self._refresh_tree()

    def _move_down(self):
        i = self._selected_idx()
        if i is None or i >= len(self.layer_list) - 1:
            return
        self.layer_list[i], self.layer_list[i + 1] = self.layer_list[i + 1], self.layer_list[i]
        self._refresh_tree()

    def _del_layer(self):
        i = self._selected_idx()
        if i is None:
            return
        del self.layer_list[i]
        self._refresh_tree()

    def _clear_layers(self):
        self.layer_list.clear()
        self._refresh_tree()

    def _load_demo(self):
        self.var_algo.set("ppo")
        self.strategy_var.set("distill_light")
        self.layer_list = [
            {"type": "mlp",       "config": {"in_dim": 16, "hidden_dims": [64, 64], "out_dim": 32, "activation": "ReLU"}},
            {"type": "lstm",      "config": {"input_size": 32, "hidden_size": 64, "num_layers": 1, "out_dim": 64, "batch_first": True, "bidirectional": False}},
            {"type": "attention", "config": {"embed_dim": 64, "num_heads": 4}},
        ]
        self._refresh_tree()
        self._refresh_algo_panel()
        self._gen_json()

    # ==============================================================
    # JSON 生成
    # ==============================================================

    def _gen_json(self):
        # 算法参数
        algo_cfg = {}
        for k, (typ, var) in self.algo_param_vars.items():
            try:
                algo_cfg[k] = self._parse_param_value(var.get(), typ)
            except Exception:
                algo_cfg[k] = var.get()

        payload = {
            "network_stack": self.layer_list,
            "algo": {
                "name": self.var_algo.get(),
                "config": algo_cfg,
            },
        }

        st_name = self.strategy_var.get()
        if st_name:
            payload["strategy"] = {
                "name": st_name,
                "config": {},
                "extra": {},
            }

        text = json.dumps(payload, indent=2, ensure_ascii=False)
        self.txt_json.delete(1.0, tk.END)
        self.txt_json.insert(tk.END, text)
        return text

    def _copy_json(self):
        txt = self._gen_json()
        self.root.clipboard_clear()
        self.root.clipboard_append(txt)
        messagebox.showinfo("OK", "JSON 已复制到剪贴板")

    def _save_json(self):
        from tkinter import filedialog
        txt = self._gen_json()
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            initialfile="rl_config.json",
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(txt)
            messagebox.showinfo("OK", f"已保存到 {path}")


if __name__ == "__main__":
    root = tk.Tk()
    RLNetJsonBuilder(root)
    root.mainloop()
