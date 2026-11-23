# -*- coding: utf-8 -*-
"""智能体构建与训练一体化平台（tkinter）。

把原来两个界面合并成一个：
  · 智能体构建（原 gui_agent_platform 的构建体管理 + gui_json_builder 的网络画布）
  · 智能体训练（真实启动 fz_agent.main rl-train，子进程 + 实时日志 + 曲线）
  · 模型与回放（bundles / runs / chess-engine2 的 logs 回放）

设计原则：
  · 界面只做编排，配置组装/校验/解析全部在 fz_agent/studio.py（可单测）；
  · 生成的 JSON 直接是 fz_agent.main 能解释的完整配置，不再是"模拟"；
  · 训练是子进程，可停止；环境是否在线可一键检测（无代理探测，见 net_utils）。

运行（WSL 需 WSLg 才能显示窗口；Windows 上可直接运行）：
    python gui_agent_studio.py
    python -m fz_agent.studio        # 无界面：只跑纯逻辑自检
"""
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 涉及界面，改用相对路径，获取当前文件所在目录并添加到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fz_agent import studio
from fz_agent.net_utils import candidate_hosts, probe, request_json


ACT_OPTIONS = ["ReLU", "GELU", "SiLU", "Tanh", "Sigmoid", "LeakyReLU"]


def open_path(path):
    """用系统默认程序打开文件/目录。"""
    path = str(path)
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception as exc:  # noqa: BLE001
        messagebox.showwarning("打开失败", "%s\n%s" % (path, exc))


class ScrollFrame(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.inner.bind("<Configure>",
                        lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.vbar.set)
        self.canvas.bind_all("<MouseWheel>",
                             lambda e: self.canvas.yview_scroll(int(-e.delta / 120), "units"))
        self.canvas.pack(side="left", fill="both", expand=True)
        self.vbar.pack(side="right", fill="y")


class AgentStudio(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DRLForge · 智能体构建与训练平台")
        self.geometry("1360x860")
        self.stack_list = []
        self.net_param_vars = {}
        self.strategy_param_vars = {}
        self.algo_param_vars = {}
        self.train_proc = None
        self.train_queue = queue.Queue()
        self.svc_queue = queue.Queue()
        self.tasks = {}
        self.serve_proc = None
        self.last_run_dir = None
        self.replay_dir = str(studio.DEFAULT_REPLAY_DIR)
        self._build_ui()
        self.after(150, self._poll_all)

    # ================================================================
    # 骨架
    # ================================================================
    def _build_ui(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=6, pady=6)
        self.nb = nb
        self.tab_build = ttk.Frame(nb)
        self.tab_train = ttk.Frame(nb)
        self.tab_model = ttk.Frame(nb)
        nb.add(self.tab_build, text=" 智能体构建 ")
        nb.add(self.tab_train, text=" 智能体训练 ")
        nb.add(self.tab_model, text=" 模型与回放 ")
        self._build_tab_build()
        self._build_tab_train()
        self._build_tab_model()

    # ================================================================
    # 页面一：智能体构建
    # ================================================================
    def _build_tab_build(self):
        bar = ttk.Frame(self.tab_build)
        bar.pack(fill="x", padx=6, pady=4)
        ttk.Label(bar, text="预设").pack(side="left")
        self.v_preset = tk.StringVar(value="chess_ppo")
        ttk.Combobox(bar, textvariable=self.v_preset, width=14, state="readonly",
                     values=list(studio.PRESETS)).pack(side="left", padx=4)
        ttk.Button(bar, text="载入预设", command=lambda: self._apply_preset(self.v_preset.get())).pack(side="left", padx=2)
        ttk.Button(bar, text="载入配置文件…", command=self._load_config_file).pack(side="left", padx=2)
        ttk.Button(bar, text="保存到 configs", command=self._save_config).pack(side="left", padx=2)
        ttk.Label(bar, text="结构签名").pack(side="left", padx=(16, 2))
        self.lbl_sig = ttk.Label(bar, text="-", foreground="#06a")
        self.lbl_sig.pack(side="left")

        pan = ttk.PanedWindow(self.tab_build, orient="horizontal")
        pan.pack(fill="both", expand=True)
        left = ScrollFrame(pan)
        pan.add(left, weight=3)
        f = left.inner

        row = 0
        # ① 基本信息
        ttk.Label(f, text="① 基本信息", font=("", 11, "bold")).grid(row=row, column=0, sticky="w", pady=(4, 2))
        row += 1
        base = ttk.LabelFrame(f)
        base.grid(row=row, column=0, sticky="ew", pady=2)
        row += 1
        self.v_name = tk.StringVar(value="ppo_chess_gui")
        self.v_desc = tk.StringVar(value="")
        self.v_paradigm = tk.StringVar(value="rl")
        ttk.Label(base, text="名称").grid(row=0, column=0, sticky="e", padx=4, pady=2)
        ttk.Entry(base, textvariable=self.v_name, width=26).grid(row=0, column=1, sticky="w")
        ttk.Label(base, text="描述").grid(row=0, column=2, sticky="e", padx=4)
        ttk.Entry(base, textvariable=self.v_desc, width=34).grid(row=0, column=3, sticky="w")
        ttk.Label(base, text="范式").grid(row=1, column=0, sticky="e", padx=4)
        for i, (val, text) in enumerate([("rl", "强化学习"), ("dl", "深度学习")]):
            ttk.Radiobutton(base, text=text, value=val, variable=self.v_paradigm,
                            command=self._on_paradigm).grid(row=1, column=1 + i, sticky="w")

        # ② 算法与超参
        self.v_space = tk.StringVar(value="discrete")   #特意提前（因为会被用到，仅此参数提前初始化-cq20261120）
        ttk.Label(f, text="② 算法（11 种）", font=("", 11, "bold")).grid(row=row, column=0, sticky="w", pady=(8, 2))
        row += 1
        algo = ttk.LabelFrame(f)
        algo.grid(row=row, column=0, sticky="ew", pady=2)
        row += 1
        self.v_algo_label = tk.StringVar(value="PPO")
        ttk.Label(algo, text="算法").grid(row=0, column=0, sticky="e", padx=4, pady=2)
        self.cmb_algo = ttk.Combobox(algo, textvariable=self.v_algo_label, width=22,
                                     state="readonly",
                                     values=[a["label"] for a in studio.ALGOS])
        self.cmb_algo.grid(row=0, column=1, sticky="w")
        self.cmb_algo.bind("<<ComboboxSelected>>", lambda e: self._on_algo())
        self.lbl_algo_hint = ttk.Label(algo, foreground="#666")
        self.lbl_algo_hint.grid(row=0, column=2, columnspan=5, sticky="w", padx=6)
        ttk.Label(algo, text="学习率").grid(row=1, column=0, sticky="e", padx=4)
        self.v_lr = tk.StringVar(value="3e-4")
        ttk.Entry(algo, textvariable=self.v_lr, width=10).grid(row=1, column=1, sticky="w")
        ttk.Label(algo, text="γ").grid(row=1, column=2, sticky="e")
        self.v_gamma = tk.StringVar(value="0.99")
        ttk.Entry(algo, textvariable=self.v_gamma, width=8).grid(row=1, column=3, sticky="w")
        ttk.Label(algo, text="batch").grid(row=1, column=4, sticky="e")
        self.v_batch = tk.StringVar(value="64")
        ttk.Entry(algo, textvariable=self.v_batch, width=8).grid(row=1, column=5, sticky="w")
        self.algo_extra = ttk.Frame(algo)
        self.algo_extra.grid(row=2, column=0, columnspan=6, sticky="w", padx=4, pady=2)
        self._on_algo()

        # ③ 网络积木
        ttk.Label(f, text="③ 网络积木（8 种，可堆叠）", font=("", 11, "bold")).grid(row=row, column=0, sticky="w", pady=(8, 2))
        row += 1
        net = ttk.LabelFrame(f)
        net.grid(row=row, column=0, sticky="ew", pady=2)
        row += 1
        self.v_module = tk.StringVar(value="MLP")
        ttk.Label(net, text="模块").grid(row=0, column=0, sticky="e", padx=4)
        cmb_mod = ttk.Combobox(net, textvariable=self.v_module, width=16, state="readonly",
                               values=[m["label"] for m in studio.NET_MODULES])
        cmb_mod.grid(row=0, column=1, sticky="w")
        cmb_mod.bind("<<ComboboxSelected>>", lambda e: self._refresh_net_params())
        self.net_param_frame = ttk.Frame(net)
        self.net_param_frame.grid(row=0, column=2, padx=6)
        ttk.Button(net, text="添加到堆叠", command=self._add_layer).grid(row=0, column=12, padx=4)
        self._refresh_net_params()

        self.tree_stack = ttk.Treeview(net, columns=("type", "config"), show="headings", height=5)
        self.tree_stack.heading("type", text="模块")
        self.tree_stack.heading("config", text="config")
        self.tree_stack.column("type", width=110)
        self.tree_stack.column("config", width=680)
        self.tree_stack.grid(row=1, column=0, columnspan=13, sticky="ew", pady=4)
        ops = ttk.Frame(net)
        ops.grid(row=2, column=0, columnspan=13, sticky="w")
        for text, cmd in [("上移", self._move_up), ("下移", self._move_down),
                          ("删除选中", self._del_layer), ("清空", self._clear_layers),
                          ("按观测维度对齐", self._align_stack)]:
            ttk.Button(ops, text=text, command=cmd).pack(side="left", padx=2)

        # ④ 观测 / 动作 / 环境
        ttk.Label(f, text="④ 观测 / 动作 / 环境", font=("", 11, "bold")).grid(row=row, column=0, sticky="w", pady=(8, 2))
        row += 1
        io = ttk.LabelFrame(f)
        io.grid(row=row, column=0, sticky="ew", pady=2)
        row += 1
        self.v_obs_dim = tk.StringVar(value="10")
        self.v_action_dim = tk.StringVar(value="32")
        self.v_max_entities = tk.StringVar(value="12")
        self.v_entity_set = tk.BooleanVar(value=True)
        # self.v_space = tk.StringVar(value="discrete")
        self.v_obs_names = tk.StringVar(value="dx,dy,hp_ratio,atk_ratio,armor,move_ratio,is_enemy,is_self,alive,tick_ratio")
        ttk.Label(io, text="观测特征维").grid(row=0, column=0, sticky="e", padx=4)
        ttk.Entry(io, textvariable=self.v_obs_dim, width=6).grid(row=0, column=1, sticky="w")
        ttk.Label(io, text="实体行数").grid(row=0, column=2, sticky="e", padx=4)
        ttk.Entry(io, textvariable=self.v_max_entities, width=6).grid(row=0, column=3, sticky="w")
        ttk.Checkbutton(io, text="实体集合", variable=self.v_entity_set).grid(row=0, column=4, padx=4)
        ttk.Label(io, text="动作空间").grid(row=0, column=5, sticky="e", padx=4)
        ttk.Combobox(io, textvariable=self.v_space, width=10, state="readonly",
                     values=["discrete", "continuous"]).grid(row=0, column=6, sticky="w")
        ttk.Label(io, text="动作维度").grid(row=0, column=7, sticky="e", padx=4)
        ttk.Entry(io, textvariable=self.v_action_dim, width=6).grid(row=0, column=8, sticky="w")
        ttk.Label(io, text="观测名称(逗号分隔)").grid(row=1, column=0, sticky="e", padx=4, pady=2)
        ttk.Entry(io, textvariable=self.v_obs_names, width=90).grid(row=1, column=1, columnspan=8, sticky="w")

        env = ttk.LabelFrame(f)
        env.grid(row=row, column=0, sticky="ew", pady=2)
        row += 1
        self.v_env_type = tk.StringVar(value="chess_remote")
        self.v_env_url = tk.StringVar(value="")
        self.v_env_side = tk.StringVar(value="red")
        self.v_env_port = tk.StringVar(value="8000")
        self.v_num_enemies = tk.StringVar(value="3")
        self.v_max_detected = tk.StringVar(value="10")
        self.v_max_steps = tk.StringVar(value="300")
        ttk.Label(env, text="环境类型").grid(row=0, column=0, sticky="e", padx=4)
        cmb_env = ttk.Combobox(env, textvariable=self.v_env_type, width=14, state="readonly",
                               values=["chess_remote", "combat"])
        cmb_env.grid(row=0, column=1, sticky="w")
        cmb_env.bind("<<ComboboxSelected>>", lambda e: self._refresh_env_fields())
        ttk.Label(env, text="服务地址").grid(row=0, column=2, sticky="e", padx=4)
        ttk.Entry(env, textvariable=self.v_env_url, width=30).grid(row=0, column=3, sticky="w")
        ttk.Label(env, text="（留空=自动探测 Windows 主机）", foreground="#666").grid(row=0, column=4, sticky="w")
        ttk.Label(env, text="控制阵营").grid(row=1, column=0, sticky="e", padx=4, pady=2)
        ttk.Entry(env, textvariable=self.v_env_side, width=8).grid(row=1, column=1, sticky="w")
        ttk.Label(env, text="端口").grid(row=1, column=2, sticky="e", padx=4)
        ttk.Entry(env, textvariable=self.v_env_port, width=8).grid(row=1, column=3, sticky="w")
        self.combat_box = ttk.Frame(env)
        self.combat_box.grid(row=2, column=0, columnspan=5, sticky="w", padx=4, pady=2)
        ttk.Label(self.combat_box, text="敌人数").pack(side="left")
        ttk.Entry(self.combat_box, textvariable=self.v_num_enemies, width=5).pack(side="left", padx=2)
        ttk.Label(self.combat_box, text="最大探测").pack(side="left")
        ttk.Entry(self.combat_box, textvariable=self.v_max_detected, width=5).pack(side="left", padx=2)
        ttk.Label(self.combat_box, text="最大步数").pack(side="left")
        ttk.Entry(self.combat_box, textvariable=self.v_max_steps, width=6).pack(side="left", padx=2)
        self._refresh_env_fields()

        # ⑤ 训练参数
        ttk.Label(f, text="⑤ 训练参数", font=("", 11, "bold")).grid(row=row, column=0, sticky="w", pady=(8, 2))
        row += 1
        tr = ttk.LabelFrame(f)
        tr.grid(row=row, column=0, sticky="ew", pady=2)
        row += 1
        self.v_seed = tk.StringVar(value="42")
        self.v_epochs = tk.StringVar(value="10")
        self.v_rollout = tk.StringVar(value="512")
        self.v_update_epochs = tk.StringVar(value="4")
        self.v_minibatch = tk.StringVar(value="128")
        self.v_eval_interval = tk.StringVar(value="5")
        self.v_total_steps = tk.StringVar(value="5000")
        self.v_capacity = tk.StringVar(value="20000")
        self.v_warmup = tk.StringVar(value="500")
        self.v_eval_steps = tk.StringVar(value="1000")
        self._entry(tr, 0, 0, "seed", self.v_seed, 6)
        self._entry(tr, 0, 2, "total_epochs", self.v_epochs, 6)
        self._entry(tr, 0, 4, "rollout_steps", self.v_rollout, 8)
        self._entry(tr, 0, 6, "update_epochs", self.v_update_epochs, 6)
        self._entry(tr, 1, 0, "minibatch", self.v_minibatch, 6)
        self._entry(tr, 1, 2, "eval_interval", self.v_eval_interval, 6)
        self._entry(tr, 1, 4, "total_steps(离线)", self.v_total_steps, 8)
        self._entry(tr, 2, 0, "capacity", self.v_capacity, 8)
        self._entry(tr, 2, 2, "warmup_steps", self.v_warmup, 8)
        self._entry(tr, 2, 4, "eval_interval_steps", self.v_eval_steps, 8)

        # ⑥ 演化训练策略
        ttk.Label(f, text="⑥ 演化训练策略（6 种，最多选一个）", font=("", 11, "bold")).grid(row=row, column=0, sticky="w", pady=(8, 2))
        row += 1
        st = ttk.LabelFrame(f)
        st.grid(row=row, column=0, sticky="ew", pady=2)
        row += 1
        self.v_strategy = tk.StringVar(value="")
        ttk.Label(st, text="策略").grid(row=0, column=0, sticky="e", padx=4)
        cmb_st = ttk.Combobox(st, textvariable=self.v_strategy, width=20, state="readonly",
                              values=[s["label"] for s in studio.STRATEGIES])
        cmb_st.grid(row=0, column=1, sticky="w")
        cmb_st.bind("<<ComboboxSelected>>", lambda e: self._on_strategy())
        self.strategy_box = ttk.Frame(st)
        self.strategy_box.grid(row=0, column=2, padx=8, sticky="w")
        self._on_strategy()

        # ⑦ 深度学习范式（仅 dl 使用）
        ttk.Label(f, text="⑦ 深度学习", font=("", 11, "bold")).grid(row=row, column=0, sticky="w", pady=(8, 2))
        row += 1
        dl = ttk.LabelFrame(f)
        dl.grid(row=row, column=0, sticky="ew", pady=2)
        row += 1
        self.v_dl_path = tk.StringVar(value="")
        self.v_dl_task = tk.StringVar(value="regression")
        self.v_dl_epochs = tk.StringVar(value="20")
        self.v_dl_lr = tk.StringVar(value="1e-3")
        self.v_dl_batch = tk.StringVar(value="64")
        ttk.Label(dl, text="数据集").grid(row=0, column=0, sticky="e", padx=4)
        ttk.Entry(dl, textvariable=self.v_dl_path, width=40).grid(row=0, column=1, columnspan=3, sticky="w")
        ttk.Button(dl, text="选择…", command=self._pick_dl_data).grid(row=0, column=4, padx=4)
        ttk.Label(dl, text="任务").grid(row=1, column=0, sticky="e", padx=4)
        ttk.Combobox(dl, textvariable=self.v_dl_task, width=14, state="readonly",
                     values=["regression", "classification"]).grid(row=1, column=1, sticky="w")
        self._entry(dl, 1, 2, "epochs", self.v_dl_epochs, 6)
        self._entry(dl, 1, 4, "lr", self.v_dl_lr, 8)
        self._entry(dl, 2, 0, "batch", self.v_dl_batch, 6)

        # 操作条
        bar2 = ttk.Frame(f)
        bar2.grid(row=row, column=0, sticky="ew", pady=10)
        row += 1
        self.lbl_check = ttk.Label(bar2, text="尚未校验", foreground="#a60")
        self.lbl_check.pack(side="left", padx=4)
        ttk.Button(bar2, text="静态校验", command=self._validate_static).pack(side="right", padx=3)
        ttk.Button(bar2, text="dry-run 前向", command=self._dry_run).pack(side="right", padx=3)
        ttk.Button(bar2, text="生成 JSON", command=self._gen_json).pack(side="right", padx=3)
        ttk.Button(bar2, text="保存并去训练", command=self._save_and_train).pack(side="right", padx=3)

        # 右侧：JSON 预览
        right = ttk.Frame(pan)
        pan.add(right, weight=2)
        ttk.Label(right, text="配置 JSON（可直接被 fz_agent.main 执行）").pack(anchor="w")
        self.txt_json = tk.Text(right, font=("Consolas", 9), wrap="none")
        self.txt_json.pack(fill="both", expand=True)

        self._apply_preset("chess_ppo")

    def _entry(self, parent, r, c, label, var, width=8):
        ttk.Label(parent, text=label).grid(row=r, column=c, sticky="e", padx=4, pady=2)
        ttk.Entry(parent, textvariable=var, width=width).grid(row=r, column=c + 1, sticky="w")

    # ---------------- 构建页：网络堆叠 ----------------
    def _current_module(self):
        label = self.v_module.get()
        for m in studio.NET_MODULES:
            if m["label"] == label:
                return m
        return studio.NET_MODULES[0]

    def _refresh_net_params(self):
        for w in self.net_param_frame.winfo_children():
            w.destroy()
        self.net_param_vars.clear()
        mod = self._current_module()
        for i, (key, typ, default) in enumerate(mod["params"]):
            ttk.Label(self.net_param_frame, text=key).grid(row=0, column=i * 2, padx=2)
            var = tk.StringVar(value=str(default))
            self.net_param_vars[key] = (typ, var)
            if key == "activation":
                ttk.Combobox(self.net_param_frame, textvariable=var, values=ACT_OPTIONS,
                             width=9, state="readonly").grid(row=0, column=i * 2 + 1)
            else:
                ttk.Entry(self.net_param_frame, textvariable=var, width=9).grid(row=0, column=i * 2 + 1)

    def _add_layer(self):
        mod = self._current_module()
        cfg = {}
        try:
            for key, (typ, var) in self.net_param_vars.items():
                cfg[key] = studio.parse_value(var.get(), typ)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("参数错误", "%s 解析失败: %s" % (key, exc))
            return
        self.stack_list.append({"type": mod["key"], "config": cfg})
        self._refresh_stack_tree()

    def _refresh_stack_tree(self):
        self.tree_stack.delete(*self.tree_stack.get_children())
        import json
        for layer in self.stack_list:
            self.tree_stack.insert("", "end",
                                   values=(layer["type"],
                                           json.dumps(layer["config"], ensure_ascii=False)))

    def _stack_idx(self):
        sel = self.tree_stack.selection()
        return self.tree_stack.index(sel[0]) if sel else None

    def _move_up(self):
        i = self._stack_idx()
        if i not in (None, 0):
            self.stack_list[i - 1], self.stack_list[i] = self.stack_list[i], self.stack_list[i - 1]
            self._refresh_stack_tree()

    def _move_down(self):
        i = self._stack_idx()
        if i is not None and i < len(self.stack_list) - 1:
            self.stack_list[i + 1], self.stack_list[i] = self.stack_list[i], self.stack_list[i + 1]
            self._refresh_stack_tree()

    def _del_layer(self):
        i = self._stack_idx()
        if i is not None:
            del self.stack_list[i]
            self._refresh_stack_tree()

    def _clear_layers(self):
        self.stack_list.clear()
        self._refresh_stack_tree()

    def _align_stack(self):
        try:
            obs_dim = int(self.v_obs_dim.get())
        except ValueError:
            messagebox.showwarning("提示", "观测特征维不是整数")
            return
        warnings = []
        self.stack_list, warnings = studio.align_stack(self.stack_list, obs_dim)
        self._refresh_stack_tree()
        messagebox.showinfo("已对齐", "\n".join(warnings) or "所有入口维度已对齐到 %d" % obs_dim)

    # ---------------- 构建页：其他 ----------------
    def _refresh_env_fields(self):
        if self.v_env_type.get() == "combat":
            for w in self.combat_box.winfo_children():
                try:
                    w.configure(state="normal")
                except tk.TclError:
                    pass
        else:
            for w in self.combat_box.winfo_children():
                try:
                    w.configure(state="disabled")
                except tk.TclError:
                    pass

    def _on_paradigm(self):
        is_rl = self.v_paradigm.get() == "rl"

        def set_state(w, state):
            for child in w.winfo_children():
                try:
                    child.configure(state=state)
                except tk.TclError:
                    pass
                if child.winfo_children():
                    set_state(child, state)
        for frame in (self.algo_extra,):
            set_state(frame, "normal" if is_rl else "disabled")

    def _algo_by_label(self, label):
        for a in studio.ALGOS:
            if a["label"] == label:
                return a
        return studio.ALGOS[0]

    def _on_algo(self):
        algo = self._algo_by_label(self.v_algo_label.get())
        family = {"on_policy": "on-policy", "off_policy": "off-policy",
                  "continuous": "连续动作"}[algo["family"]]
        tip = "%s · %s · 角色: %s" % (algo["name"], family, "/".join(algo["roles"]))
        if not algo.get("ok", True):
            tip += " · 后端待补齐，无法训练"
        self.lbl_algo_hint.config(text=tip)
        if algo["family"] == "continuous":
            self.v_space.set("continuous")
        elif self.v_space.get() == "continuous":
            self.v_space.set("discrete")
        # 算法专属超参
        for w in self.algo_extra.winfo_children():
            w.destroy()
        self.algo_param_vars.clear()
        defaults = studio.ALGO_DEFAULTS.get(algo["name"], {})
        skip = {"learning_rate", "gamma", "batch_size", "device", "action_dim"}
        i = 0
        for key, val in defaults.items():
            if key in skip:
                continue
            ttk.Label(self.algo_extra, text=key).grid(row=0, column=i * 2, padx=2)
            var = tk.StringVar(value=str(val))
            self.algo_param_vars[key] = (type(val), var)
            ttk.Entry(self.algo_extra, textvariable=var, width=9).grid(row=0, column=i * 2 + 1)
            i += 1

    def _on_strategy(self):
        for w in self.strategy_box.winfo_children():
            w.destroy()
        self.strategy_param_vars.clear()
        label = self.v_strategy.get()
        meta = next((s for s in studio.STRATEGIES if s["label"] == label), None)
        if not meta or not meta.get("fields"):
            return
        for i, (key, typ, default) in enumerate(meta["fields"]):
            ttk.Label(self.strategy_box, text=key).grid(row=0, column=i * 2, padx=2)
            var = tk.StringVar(value=str(default))
            self.strategy_param_vars[(meta["name"], key)] = (typ, var)
            ttk.Entry(self.strategy_box, textvariable=var, width=14).grid(row=0, column=i * 2 + 1)

    def _pick_dl_data(self):
        p = filedialog.askopenfilename(filetypes=[("npz", "*.npz"), ("全部", "*.*")])
        if p:
            self.v_dl_path.set(p)

    def _apply_preset(self, key):
        preset = studio.PRESETS.get(key)
        if not preset:
            return
        self.v_name.set(preset["name"])
        self.v_desc.set(preset.get("desc", ""))
        self.v_paradigm.set("rl")
        io = preset["io"]
        self.v_obs_dim.set(str(io["obs_dim"]))
        self.v_entity_set.set(bool(io["entity_set"]))
        self.v_space.set(io["space_type"])
        self.v_action_dim.set(str(io["action_dim"]))
        env = preset["env"]
        self.v_env_type.set(env["type"])
        self.v_env_url.set(env.get("url", ""))
        self.v_env_side.set(env.get("side", "red"))
        self.v_env_port.set(str(env.get("port", 8000)))
        self.v_max_entities.set(str(env.get("max_entities", 12)))
        self.v_num_enemies.set(str(env.get("num_enemies", 3)))
        self.v_max_detected.set(str(env.get("max_detected", 10)))
        self.v_max_steps.set(str(env.get("max_episode_steps", 300)))
        if io.get("names"):
            self.v_obs_names.set(",".join(io["names"]))
        self.v_algo_label.set(self._algo_by_label("PPO")["label"] if preset["algo"] == "ppo"
                              else next(a["label"] for a in studio.ALGOS if a["name"] == preset["algo"]))
        self._on_algo()
        tr = preset["train"]
        self.v_seed.set(str(tr.get("seed", 42)))
        self.v_epochs.set(str(tr.get("total_epochs", 10)))
        self.v_rollout.set(str(tr.get("rollout_steps", 512)))
        self.v_update_epochs.set(str(tr.get("update_epochs", 4)))
        self.v_minibatch.set(str(tr.get("minibatch_size", 128)))
        self.v_eval_interval.set(str(tr.get("eval_interval", 5)))
        self.v_total_steps.set(str(tr.get("total_steps", 5000)))
        self.v_capacity.set(str(tr.get("capacity", 20000)))
        self.v_warmup.set(str(tr.get("warmup_steps", 500)))
        self.v_eval_steps.set(str(tr.get("eval_interval_steps", 1000)))
        self.v_strategy.set("（无）")
        self._on_strategy()
        self.stack_list = [dict(layer) for layer in preset["stack"]]
        self._refresh_stack_tree()
        self._refresh_env_fields()
        self._gen_json()

    def _collect_form(self):
        algo = self._algo_by_label(self.v_algo_label.get())
        try:
            obs_dim = int(self.v_obs_dim.get())
            action_dim = int(self.v_action_dim.get())
            max_entities = int(self.v_max_entities.get())
        except ValueError:
            messagebox.showwarning("提示", "观测/动作/实体行数必须是整数")
            raise

        algo_cfg = studio.default_algo_config(algo["name"], action_dim, self.v_space.get())
        try:
            algo_cfg["learning_rate"] = float(self.v_lr.get())
            algo_cfg["gamma"] = float(self.v_gamma.get())
            if "batch_size" in algo_cfg:
                algo_cfg["batch_size"] = int(self.v_batch.get())
        except ValueError:
            messagebox.showwarning("提示", "学习率/γ/batch 必须是数字")
            raise
        for key, (typ, var) in self.algo_param_vars.items():
            algo_cfg[key] = studio.parse_value(var.get(), typ)

        strategy = {"name": "", "config": {}}
        label = self.v_strategy.get()
        for s in studio.STRATEGIES:
            if s["label"] == label and s["name"]:
                strategy = {"name": s["name"], "config": {}}
                for key, (typ, var) in self.strategy_param_vars.items():
                    if key[0] == s["name"]:
                        strategy["config"][key[1]] = studio.parse_value(var.get(), typ)
                break

        names = [n.strip() for n in self.v_obs_names.get().split(",") if n.strip()]
        form = {
            "name": self.v_name.get(), "desc": self.v_desc.get(),
            "paradigm": self.v_paradigm.get(),
            "algo": algo["name"], "algo_config": algo_cfg,
            "io": {"obs_dim": obs_dim, "entity_set": bool(self.v_entity_set.get()),
                   "space_type": self.v_space.get(), "action_dim": action_dim,
                   "names": names},
            "env": {"type": self.v_env_type.get(), "url": self.v_env_url.get().strip(),
                    "side": self.v_env_side.get().strip() or "red",
                    "port": int(self.v_env_port.get() or 8000),
                    "max_entities": max_entities,
                    "num_enemies": int(self.v_num_enemies.get() or 3),
                    "max_detected": int(self.v_max_detected.get() or 10),
                    "max_episode_steps": int(self.v_max_steps.get() or 300)},
            "stack": self.stack_list,
            "train": {
                "seed": int(self.v_seed.get() or 42),
                "total_epochs": int(self.v_epochs.get() or 10),
                "rollout_steps": int(self.v_rollout.get() or 512),
                "update_epochs": int(self.v_update_epochs.get() or 4),
                "minibatch_size": int(self.v_minibatch.get() or 128),
                "eval_interval": int(self.v_eval_interval.get() or 5),
                "total_steps": int(self.v_total_steps.get() or 5000),
                "capacity": int(self.v_capacity.get() or 20000),
                "warmup_steps": int(self.v_warmup.get() or 500),
                "eval_interval_steps": int(self.v_eval_steps.get() or 1000),
            },
            "strategy": strategy,
            "dl": {"path": self.v_dl_path.get(), "task": self.v_dl_task.get(),
                   "epochs": int(self.v_dl_epochs.get() or 20),
                   "lr": float(self.v_dl_lr.get() or 1e-3),
                   "batch_size": int(self.v_dl_batch.get() or 64)},
        }
        return form

    def _build_cfg(self):
        form = self._collect_form()
        cfg, warns = studio.build_config(form)
        return cfg, warns

    def _validate_static(self):
        try:
            cfg, warns = self._build_cfg()
        except Exception:  # noqa: BLE001
            return False
        errs, more = studio.validate_config(cfg)
        warns += more
        if errs:
            self.lbl_check.config(text="校验未通过：%s" % "；".join(errs), foreground="#c22")
            messagebox.showwarning("校验未通过", "\n".join(errs))
            return False
        self.lbl_check.config(text="校验通过（签名 %s）" % studio.config_signature(cfg),
                              foreground="#0a6")
        if warns:
            messagebox.showinfo("校验通过（有提示）", "\n".join(warns))
        return True

    def _dry_run(self):
        if self.v_paradigm.get() == "dl":
            messagebox.showinfo("dry-run", "深度学习配置请在训练页直接启动 dl-train；\n"
                                           "dry-run 目前只做强化学习网络前向。")
            return
        if not self._validate_static():
            return
        cfg, _ = self._build_cfg()
        self.lbl_check.config(text="dry-run 中…", foreground="#a60")
        self.update_idletasks()

        def work():
            try:
                _ok, msgs = studio.dry_run(cfg)
            except Exception as exc:  # noqa: BLE001
                self.after(0, lambda e=exc: messagebox.showerror("dry-run 失败", str(e)))
                self.after(0, lambda: self.lbl_check.config(text="dry-run 失败",
                                                            foreground="#c22"))
                return
            self.after(0, lambda: self.lbl_check.config(
                text="dry-run 通过（签名 %s）" % studio.config_signature(cfg),
                foreground="#0a6"))
            self.after(0, lambda: messagebox.showinfo("dry-run 通过", "\n".join(msgs)))

        threading.Thread(target=work, daemon=True).start()

    def _gen_json(self):
        try:
            cfg, _ = self._build_cfg()
        except Exception:  # noqa: BLE001
            return None
        import json
        text = json.dumps(cfg, ensure_ascii=False, indent=2)
        self.txt_json.delete("1.0", "end")
        self.txt_json.insert("end", text)
        self.lbl_sig.config(text=studio.config_signature(cfg))
        return text

    def _save_config(self):
        try:
            cfg, _ = self._build_cfg()
        except Exception:  # noqa: BLE001
            return None
        errs, _ = studio.validate_config(cfg)
        if errs:
            messagebox.showwarning("无法保存", "\n".join(errs))
            return None
        path = studio.save_config(cfg)
        self._refresh_config_list()
        messagebox.showinfo("已保存", "配置已写入：\n%s" % path)
        return path

    def _save_and_train(self):
        path = self._save_config()
        if path:
            self._refresh_config_list()
            self.v_train_cfg.set(Path(path).name)
            self.notebook_select_train()

    def _load_config_file(self):
        p = filedialog.askopenfilename(initialdir=str(studio.CONFIG_DIR),
                                       filetypes=[("JSON", "*.json"), ("全部", "*.*")])
        if not p:
            return
        cfg = studio.load_config(p)
        self._load_cfg_into_form(cfg)

    def _load_cfg_into_form(self, cfg):
        meta = cfg.get("meta") or {}
        self.v_name.set(meta.get("name", ""))
        self.v_desc.set(meta.get("desc", ""))
        env = cfg.get("env") or {}
        io = (env.get("io") or {})
        obs = io.get("obs") or {}
        act = io.get("action") or {}
        self.v_env_type.set(env.get("type", "chess_remote"))
        self.v_env_url.set(env.get("url", ""))
        self.v_env_side.set((env.get("config") or {}).get("side", "red"))
        shape = obs.get("shape") or [12, 10]
        self.v_max_entities.set(str(shape[0]))
        self.v_obs_dim.set(str(obs.get("dim", shape[-1])))
        self.v_entity_set.set(bool(obs.get("entity_set", True)))
        self.v_obs_names.set(",".join(obs.get("names") or []))
        self.v_space.set(act.get("space_type", "discrete"))
        self.v_action_dim.set(str(act.get("dim", 32)))
        algo = (cfg.get("algo") or {}).get("name", "ppo")
        self.v_algo_label.set(next((a["label"] for a in studio.ALGOS if a["name"] == algo),
                                   "PPO"))
        self._on_algo()
        algo_cfg = (cfg.get("algo") or {}).get("config") or {}
        self.v_lr.set(str(algo_cfg.get("learning_rate", 3e-4)))
        self.v_gamma.set(str(algo_cfg.get("gamma", 0.99)))
        self.v_batch.set(str(algo_cfg.get("batch_size", 64)))
        tr = cfg.get("train") or {}
        for var, key in ((self.v_seed, "seed"), (self.v_epochs, "total_epochs"),
                         (self.v_rollout, "rollout_steps"),
                         (self.v_update_epochs, "update_epochs"),
                         (self.v_minibatch, "minibatch_size"),
                         (self.v_eval_interval, "eval_interval"),
                         (self.v_total_steps, "total_steps"),
                         (self.v_capacity, "capacity"),
                         (self.v_warmup, "warmup_steps"),
                         (self.v_eval_steps, "eval_interval_steps")):
            if key in tr:
                var.set(str(tr[key]))
        roles = cfg.get("networks") or {}
        first = next(iter(roles.values()), {}) if roles else {}
        self.stack_list = [dict(x) for x in (first.get("network_stack") or [])]
        self._refresh_stack_tree()
        self._refresh_env_fields()
        self._gen_json()

    # ================================================================
    # 页面二：智能体训练
    # ================================================================
    def _build_tab_train(self):
        form = ttk.LabelFrame(self.tab_train, text="训练任务")
        form.pack(fill="x", padx=6, pady=6)
        self.v_train_cfg = tk.StringVar()
        ttk.Label(form, text="配置").grid(row=0, column=0, sticky="e", padx=4, pady=4)
        self.cmb_train_cfg = ttk.Combobox(form, textvariable=self.v_train_cfg, width=40,
                                          state="readonly")
        self.cmb_train_cfg.grid(row=0, column=1, sticky="w")
        ttk.Button(form, text="刷新", command=self._refresh_config_list).grid(row=0, column=2, padx=4)
        ttk.Button(form, text="检测环境连通", command=self._check_env).grid(row=0, column=3, padx=4)
        self.btn_start = ttk.Button(form, text="启动训练", command=self._start_train)
        self.btn_start.grid(row=0, column=4, padx=4)
        self.btn_stop = ttk.Button(form, text="停止", command=self._stop_train, state="disabled")
        self.btn_stop.grid(row=0, column=5, padx=4)
        ttk.Button(form, text="打开 runs 目录", command=lambda: open_path(studio.RUNS_DIR)).grid(row=0, column=6, padx=4)
        ttk.Button(form, text="打开 bundles 目录", command=lambda: open_path(studio.BUNDLE_DIR)).grid(row=0, column=7, padx=4)
        ttk.Button(form, text="启动推理服务", command=self._start_serve).grid(row=0, column=8, padx=4)
        self.btn_stop_serve = ttk.Button(form, text="停止服务", command=self._stop_serve, state="disabled")
        self.btn_stop_serve.grid(row=0, column=9, padx=4)

        pan = ttk.PanedWindow(self.tab_train, orient="horizontal")
        pan.pack(fill="both", expand=True, padx=6, pady=4)
        left = ttk.Frame(pan)
        pan.add(left, weight=2)
        ttk.Label(left, text="任务列表").pack(anchor="w")
        self.tree_tasks = ttk.Treeview(left, columns=("cfg", "status", "prog", "bundle"),
                                       show="headings", height=8)
        for cid, txt, w in (("cfg", "配置", 220), ("status", "状态", 90),
                            ("prog", "进度", 70), ("bundle", "产物", 220)):
            self.tree_tasks.heading(cid, text=txt)
            self.tree_tasks.column(cid, width=w)
        self.tree_tasks.pack(fill="both", expand=True)
        self.pg = ttk.Progressbar(left, mode="determinate")
        self.pg.pack(fill="x", pady=4)
        ttk.Label(left, text="训练曲线").pack(anchor="w")
        self.canvas_curve = tk.Canvas(left, height=190, bg="#fbfbf7")
        self.canvas_curve.pack(fill="x", pady=2)
        ttk.Button(left, text="刷新曲线（最近一次 run）",
                   command=self._draw_latest_curve).pack(anchor="e")

        right = ttk.Frame(pan)
        pan.add(right, weight=3)
        self._refresh_config_list()
        ttk.Label(right, text="训练日志 / 服务输出").pack(anchor="w")
        self.txt_log = tk.Text(right, state="disabled", bg="#111", fg="#8f8")
        self.txt_log.pack(fill="both", expand=True)

    def notebook_select_train(self):
        if getattr(self, "nb", None) is not None:
            self.nb.select(1)

    def _refresh_config_list(self):
        names = [Path(p).name for p in studio.list_configs()]
        self.cmb_train_cfg.config(values=names)
        if names and not self.v_train_cfg.get():
            self.v_train_cfg.set(names[0])

    def _selected_config_path(self):
        name = self.v_train_cfg.get()
        if not name:
            return None
        p = studio.CONFIG_DIR / name
        if not p.exists():
            messagebox.showwarning("提示", "配置不存在：%s" % p)
            return None
        return str(p)

    def _check_env(self):
        path = self._selected_config_path()
        if not path:
            return
        cfg = studio.load_config(path)
        env = cfg.get("env") or {}
        if env.get("type") != "chess_remote":
            messagebox.showinfo("环境检查", "当前配置使用内置环境（%s），无需联网" % env.get("type"))
            return
        port = int(env.get("port", 8000))
        url = (env.get("url") or "").rstrip("/")
        try:
            if not url:
                host = None
                for h in candidate_hosts():
                    if probe(h, port):
                        host = h
                        break
                if host is None:
                    raise RuntimeError("候选地址均不可达：%s" % candidate_hosts())
                url = "http://%s:%d" % (host, port)
            health = request_json(url + "/health", timeout=5)
            self._log("[环境] %s ✓ scenario=%s tick=%s sessions=%s" % (
                url, health.get("scenario"), health.get("tick"), health.get("sessions")))
            messagebox.showinfo("环境连通", "已连接 %s\nscenario=%s" % (url, health.get("scenario")))
        except Exception as exc:  # noqa: BLE001
            self._log("[环境] 检测失败：%s" % exc)
            messagebox.showerror("环境连通失败", "%s" % exc)

    def _start_train(self):
        if self.train_proc is not None and self.train_proc.poll() is None:
            messagebox.showinfo("提示", "已有训练在运行，请先停止")
            return
        path = self._selected_config_path()
        if not path:
            return
        cfg = studio.load_config(path)
        name = (cfg.get("meta") or {}).get("name", Path(path).stem)
        cmd = studio.training_command(
            path, python=sys.executable) if cfg.get("algo") else [
            sys.executable, "-m", "fz_agent.main", "dl-train", "-c", path]
        tid = "T%03d" % (len(self.tasks) + 1)
        self.tasks[tid] = {"cfg": name, "status": "运行中", "prog": 0.0, "bundle": ""}
        self.tree_tasks.insert("", "end", iid=tid, values=(name, "运行中", "0%", ""))
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        try:
            self.train_proc = subprocess.Popen(
                cmd, cwd=str(studio.PROJECT_ROOT), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        except OSError as exc:
            self._log("[训练] 启动失败：%s" % exc)
            return
        self.train_queue_done = False
        self._run_name = name
        self._log("[训练] %s 启动：%s" % (tid, " ".join(cmd)))
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self._tid = tid
        threading.Thread(target=self._reader, args=(self.train_proc, self.train_queue),
                         daemon=True).start()

    def _reader(self, proc, q):
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                q.put(line.rstrip("\n"))
        finally:
            q.put(None)

    def _stop_train(self):
        if self.train_proc is not None and self.train_proc.poll() is None:
            self.train_proc.terminate()
            self._log("[训练] 已请求停止")
        self.btn_stop.config(state="disabled")

    def _poll_all(self):
        self._drain(self.train_queue, self._on_train_line)
        self._drain(self.svc_queue, self._log)
        if self.train_proc is not None and self.train_proc.poll() is not None \
                and getattr(self, "train_queue_done", False):
            self._finish_train()
        self.after(150, self._poll_all)

    def _drain(self, q, handler):
        while True:
            try:
                item = q.get_nowait()
            except queue.Empty:
                return
            if item is None:
                self.train_queue_done = True
                continue
            handler(item)

    def _on_train_line(self, line):
        self._log(line)
        kind, payload = studio.parse_train_line(line)
        tid = getattr(self, "_tid", None)
        if not tid or tid not in self.tasks:
            return
        if kind in ("epoch", "step"):
            cur, total = payload
            prog = cur / max(1, total)
            self.tasks[tid]["prog"] = prog
            self.tree_tasks.set(tid, "prog", "%.0f%%" % (prog * 100))
            self.pg["value"] = prog * 100
        elif kind == "bundle":
            self.tasks[tid]["bundle"] = payload.split(":", 1)[-1].strip()
            self.tree_tasks.set(tid, "bundle", payload.split(":", 1)[-1].strip())

    def _finish_train(self):
        tid = getattr(self, "_tid", None)
        code = self.train_proc.returncode if self.train_proc else None
        if tid and tid in self.tasks:
            ok = code == 0
            self.tasks[tid]["status"] = "已完成" if ok else "失败(%s)" % code
            self.tree_tasks.set(tid, "status", self.tasks[tid]["status"])
            if ok:
                self.tree_tasks.set(tid, "prog", "100%")
                self.pg["value"] = 100
                name = getattr(self, "_run_name", "")
                bundle = studio.BUNDLE_DIR / name if name else None
                if bundle and bundle.exists():
                    self.tasks[tid]["bundle"] = str(bundle)
                    self.tree_tasks.set(tid, "bundle", str(bundle))
                self._draw_latest_curve()
        self._log("[训练] 进程结束，返回码 %s" % code)
        self.train_proc = None
        self.train_queue_done = False
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")

    # ---------------- 服务 / 曲线 ----------------
    def _start_serve(self):
        if self.serve_proc is not None and self.serve_proc.poll() is None:
            messagebox.showinfo("提示", "推理服务已在运行")
            return
        cmd = studio.serve_command(str(studio.BUNDLE_DIR), "0.0.0.0", 8765,
                                   python=sys.executable)
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        self.serve_proc = subprocess.Popen(cmd, cwd=str(studio.PROJECT_ROOT),
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, bufsize=1, env=env)
        self._log("[服务] 启动：%s" % " ".join(cmd))
        self.btn_stop_serve.config(state="normal")
        threading.Thread(target=lambda: [self.svc_queue.put(l.rstrip())
                                        for l in self.serve_proc.stdout],
                         daemon=True).start()

    def _stop_serve(self):
        if self.serve_proc is not None and self.serve_proc.poll() is None:
            self.serve_proc.terminate()
            self._log("[服务] 已停止")
        self.btn_stop_serve.config(state="disabled")

    def _draw_latest_curve(self):
        runs = studio.list_runs()
        if not runs:
            return
        self.last_run_dir = runs[0]["path"]
        self._draw_curve(self.last_run_dir)

    def _draw_curve(self, run_dir):
        c = self.canvas_curve
        c.delete("all")
        curves = studio.read_curves(run_dir)
        series = [(curves.get("train_reward"), "#2a9d5c", "train_reward_mean20"),
                  (curves.get("eval_reward"), "#d97706", "eval_mean_reward")]
        pts = [(v, col, label) for v, col, label in series
               if any(x is not None for x in v)]
        if not pts:
            c.create_text(10, 10, anchor="nw", text="暂无曲线数据（run: %s）"
                          % Path(run_dir).name, fill="#666")
            return
        w = max(c.winfo_width(), 400)
        h = max(c.winfo_height(), 160)
        pad = 28
        vals = [x for v, _c, _l in pts for x in v if x is not None]
        lo, hi = min(vals), max(vals)
        if hi == lo:
            hi, lo = hi + 1, lo - 1
        n = max(len(v) for v, _c, _l in pts)
        c.create_line(pad, h - pad, w - 6, h - pad, fill="#999")
        c.create_line(pad, 6, pad, h - pad, fill="#999")
        c.create_text(pad, 2, anchor="nw", text="%s  hi=%.2f lo=%.2f"
                      % (Path(run_dir).name, hi, lo), fill="#333")
        for v, col, label in pts:
            seq = [(i, x) for i, x in enumerate(v) if x is not None]
            if len(seq) < 2:
                continue
            xy = []
            for i, x in seq:
                px = pad + (w - pad - 8) * (i / max(1, n - 1))
                py = h - pad - (h - pad - 12) * ((x - lo) / (hi - lo))
                xy.extend([px, py])
            c.create_line(*xy, fill=col, width=2)
            c.create_text(w - 8, 14 + 14 * (label == "eval_mean_reward"), anchor="ne",
                          text=label, fill=col)

    def _log(self, msg):
        self.txt_log.config(state="normal")
        self.txt_log.insert("end", msg + "\n")
        self.txt_log.see("end")
        self.txt_log.config(state="disabled")

    # ================================================================
    # 页面三：模型与回放
    # ================================================================
    def _build_tab_model(self):
        pan = ttk.PanedWindow(self.tab_model, orient="vertical")
        pan.pack(fill="both", expand=True, padx=6, pady=6)

        top = ttk.LabelFrame(pan, text="构建体（bundles）")
        pan.add(top, weight=1)
        self.tree_bundles = ttk.Treeview(top, columns=("id", "algo", "created", "path"),
                                         show="headings", height=6)
        for cid, txt, w in (("id", "构建体", 180), ("algo", "算法", 80),
                            ("created", "创建时间", 160), ("path", "路径", 520)):
            self.tree_bundles.heading(cid, text=txt)
            self.tree_bundles.column(cid, width=w)
        self.tree_bundles.pack(fill="both", expand=True)
        b = ttk.Frame(top)
        b.pack(fill="x")
        ttk.Button(b, text="刷新", command=self._refresh_bundles).pack(side="left", padx=2)
        ttk.Button(b, text="打开目录", command=lambda: open_path(studio.BUNDLE_DIR)).pack(side="left", padx=2)
        ttk.Button(b, text="运行推理（rl-exp 1 局）", command=self._run_exp_selected).pack(side="left", padx=2)

        mid = ttk.LabelFrame(pan, text="训练记录（runs）")
        pan.add(mid, weight=1)
        self.tree_runs = ttk.Treeview(mid, columns=("name", "path"), show="headings", height=5)
        self.tree_runs.heading("name", text="run")
        self.tree_runs.heading("path", text="路径")
        self.tree_runs.column("name", width=260)
        self.tree_runs.column("path", width=640)
        self.tree_runs.pack(fill="both", expand=True)
        r = ttk.Frame(mid)
        r.pack(fill="x")
        ttk.Button(r, text="刷新", command=self._refresh_runs).pack(side="left", padx=2)
        ttk.Button(r, text="打开目录", command=lambda: open_path(studio.RUNS_DIR)).pack(side="left", padx=2)
        ttk.Button(r, text="查看曲线", command=self._view_selected_run).pack(side="left", padx=2)

        bot = ttk.LabelFrame(pan, text="对局回放（chess-engine2/logs）")
        pan.add(bot, weight=1)
        row = ttk.Frame(bot)
        row.pack(fill="x")
        ttk.Label(row, text="回放目录").pack(side="left")
        self.v_replay_dir = tk.StringVar(value=self.replay_dir)
        ttk.Entry(row, textvariable=self.v_replay_dir, width=70).pack(side="left", padx=4)
        ttk.Button(row, text="选择…", command=self._pick_replay_dir).pack(side="left", padx=2)
        ttk.Button(row, text="刷新", command=self._refresh_replays).pack(side="left", padx=2)
        ttk.Button(row, text="打开目录", command=lambda: open_path(self.v_replay_dir.get())).pack(side="left", padx=2)
        self.tree_replays = ttk.Treeview(bot, columns=("name", "size", "path"),
                                         show="headings", height=6)
        for cid, txt, w in (("name", "文件", 300), ("size", "大小KB", 80), ("path", "路径", 620)):
            self.tree_replays.heading(cid, text=txt)
            self.tree_replays.column(cid, width=w)
        self.tree_replays.pack(fill="both", expand=True)
        ttk.Button(bot, text="复制选中回放路径",
                   command=self._copy_replay_path).pack(anchor="e", padx=2, pady=2)

        self._refresh_bundles()
        self._refresh_runs()
        self._refresh_replays()

    def _refresh_bundles(self):
        self.tree_bundles.delete(*self.tree_bundles.get_children())
        for b in studio.list_bundles():
            self.tree_bundles.insert("", "end",
                                     values=(b["id"], b["algo"], b["created"], b["path"]))

    def _refresh_runs(self):
        self.tree_runs.delete(*self.tree_runs.get_children())
        for r in studio.list_runs():
            self.tree_runs.insert("", "end", values=(r["name"], r["path"]))

    def _refresh_replays(self):
        self.tree_replays.delete(*self.tree_replays.get_children())
        for item in studio.list_replays(self.v_replay_dir.get() or None):
            self.tree_replays.insert("", "end",
                                     values=(item["name"], item["size_kb"], item["path"]))
        self._log("[回放] %s 下找到 %d 个 JSONL"
                  % (self.v_replay_dir.get(), len(self.tree_replays.get_children())))

    def _pick_replay_dir(self):
        d = filedialog.askdirectory(initialdir=self.v_replay_dir.get() or str(ROOT))
        if d:
            self.v_replay_dir.set(d)
            self._refresh_replays()

    def _copy_replay_path(self):
        sel = self.tree_replays.selection()
        if not sel:
            messagebox.showinfo("提示", "先选中一个回放文件")
            return
        path = self.tree_replays.item(sel[0], "values")[2]
        self.clipboard_clear()
        self.clipboard_append(path)
        messagebox.showinfo("已复制", "%s\n\nWindows 上用：python -m tools.replay_viewer \"%s\""
                            % (path, path))

    def _view_selected_run(self):
        sel = self.tree_runs.selection()
        if not sel:
            messagebox.showinfo("提示", "先选中一个 run")
            return
        path = self.tree_runs.item(sel[0], "values")[1]
        self._draw_curve(path)

    def _run_exp_selected(self):
        sel = self.tree_bundles.selection()
        if not sel:
            messagebox.showinfo("提示", "先选中一个构建体")
            return
        path = self.tree_bundles.item(sel[0], "values")[3]
        cmd = studio.exp_command(path, 1, python=sys.executable)
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        proc = subprocess.Popen(cmd, cwd=str(studio.PROJECT_ROOT), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        self._log("[推理] %s" % " ".join(cmd))
        threading.Thread(target=lambda: [self.svc_queue.put("[推理] " + l.rstrip())
                                        for l in proc.stdout], daemon=True).start()


def main():
    app = AgentStudio()
    app.mainloop()


if __name__ == "__main__":
    main()