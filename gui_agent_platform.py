# -*- coding: utf-8 -*-
# 智能体构建与训练平台 —— tkinter 原型
# 范围：构建体编辑 + 训练管理 + 模型库（对抗评估页不在本次范围）
# 说明：数据暂存内存；训练过程为线程模拟，接入真实引擎时替换 _train_worker 即可

import tkinter as tk
from tkinter import ttk, messagebox
import threading, time, random, queue

UNIT_TYPES = ["坦克", "步兵分队", "无人机", "舰艇", "导弹发射车"]

OBS_LIBRARY = {
    "坦克":       ["可见敌方单位", "单位位置", "自身血量", "弹药量", "地形通行性", "最近掩体距离"],
    "步兵分队":   ["可见敌方单位", "单位位置", "兵力状态", "士气", "掩体分布", "通讯状态"],
    "无人机":     ["可见敌方单位", "单位位置", "油量", "高度", "雷达信号", "禁飞区"],
    "舰艇":       ["可见敌方单位", "单位位置", "舰体损伤", "导弹库存", "海况", "声呐信号"],
    "导弹发射车": ["可见敌方单位", "单位位置", "发射准备状态", "装填状态", "阵地掩蔽度"],
}

ACT_LIBRARY = {
    "坦克":       ["机动", "开火", "隐蔽", "撤退", "呼叫支援"],
    "步兵分队":   ["机动", "开火", "隐蔽", "构筑工事", "撤退"],
    "无人机":     ["机动", "侦察", "电子干扰", "返航", "降落"],
    "舰艇":       ["机动", "导弹攻击", "近防", "电子战", "撤离"],
    "导弹发射车": ["机动转移", "展开阵地", "发射", "装填", "隐蔽"],
}

ALGORITHMS = ["DQN", "DDQN", "Dueling DQN", "Rainbow", "C51", "DDPG", "TD3", "SAC", "PPO", "A2C", "TRPO"]
BACKBONES   = ["MLP", "CNN", "GRU", "LSTM", "Attention", "Transformer", "ResNet", "GNN"]
SCENARIOS   = ["平原遭遇战", "城镇攻防", "山地伏击", "岛屿登陆", "要地防空"]

REWARD_ITEMS = [("击毁目标", 100.0), ("被击毁", -200.0), ("到达点位", 50.0),
                ("保持存活(每步)", 1.0), ("弹药消耗", -2.0)]

ADV_METHODS = ["迁移学习", "对抗训练", "增量续训", "模型蒸馏", "多任务联合", "集成投票"]

ALGO_STRUCTURAL = {
    "Rainbow":     "结构性组件自动附加：Dueling + Distributional(C51) + NoisyNet + 优先回放",
    "Dueling DQN": "结构性组件自动附加：Dueling（价值流 + 优势流）",
    "DDQN":        "双 Q 网络评估（Target 网络默认开启）",
}

DL_TASKS  = ["分类", "回归", "时序预测"]
MT_GROUPS = ["宏观决策 + 微观操控", "侦察 + 打击", "机动 + 防御"]


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


def set_state(w, state):
    for child in w.winfo_children():
        try:
            child.configure(state=state)
        except tk.TclError:
            pass
        if child.winfo_children():
            set_state(child, state)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("智能体构建与训练平台（原型）")
        self.geometry("1200x780")
        self.builds = {}   # 名称 -> [配置字典, ...] 版本列表
        self.tasks = {}    # 任务ID -> 记录
        self.queues = {}   # 任务ID -> Queue
        self.model_rows = []
        self._build_ui()

    # ---------------- 界面骨架 ----------------
    def _build_ui(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=6, pady=6)
        self.tab_build = ttk.Frame(nb);  nb.add(self.tab_build, text=" 构建体管理 ")
        self.tab_train = ttk.Frame(nb);  nb.add(self.tab_train, text=" 训练管理 ")
        self.tab_model = ttk.Frame(nb);  nb.add(self.tab_model, text=" 模型库 ")
        self._build_tab_build()
        self._build_tab_train()
        self._build_tab_model()

    # ================= 构建体管理 =================
    def _build_tab_build(self):
        pan = ttk.PanedWindow(self.tab_build, orient="horizontal")
        pan.pack(fill="both", expand=True)

        # 左：构建体列表
        left = ttk.Frame(pan, width=300); pan.add(left, weight=1)
        ttk.Label(left, text="构建体列表", font=("", 11, "bold")).pack(anchor="w", pady=4)
        cols = ("version", "para", "unit")
        self.tree_builds = ttk.Treeview(left, columns=cols, show="tree headings", height=18)
        self.tree_builds.heading("#0", text="名称")
        self.tree_builds.heading("version", text="版本")
        self.tree_builds.heading("para", text="范式")
        self.tree_builds.heading("unit", text="单位类型")
        self.tree_builds.column("#0", width=120); self.tree_builds.column("version", width=40, anchor="center")
        self.tree_builds.pack(fill="both", expand=True)
        btns = ttk.Frame(left); btns.pack(fill="x", pady=4)
        ttk.Button(btns, text="新建", command=self._new_form).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(btns, text="载入编辑", command=self._load_selected).pack(side="left", expand=True, fill="x", padx=2)

        # 右：编辑表单（可滚动）
        right = ScrollFrame(pan); pan.add(right, weight=3)
        f = right.inner
        row = 0
        ttk.Label(f, text="① 基本信息", font=("", 11, "bold")).grid(row=row, column=0, sticky="w", pady=(4, 2)); row += 1
        base = ttk.LabelFrame(f); base.grid(row=row, column=0, sticky="ew", pady=2); row += 1
        self.v_name = tk.StringVar(); self.v_desc = tk.StringVar(); self.v_unit = tk.StringVar()
        ttk.Label(base, text="名称").grid(row=0, column=0, sticky="e", padx=4, pady=2)
        ttk.Entry(base, textvariable=self.v_name, width=24).grid(row=0, column=1, sticky="w")
        ttk.Label(base, text="描述").grid(row=0, column=2, sticky="e", padx=4)
        ttk.Entry(base, textvariable=self.v_desc, width=30).grid(row=0, column=3, sticky="w")
        ttk.Label(base, text="单位类型").grid(row=1, column=0, sticky="e", padx=4, pady=2)
        self.cmb_unit = ttk.Combobox(base, textvariable=self.v_unit, values=UNIT_TYPES,
                                     state="readonly", width=22)
        self.cmb_unit.grid(row=1, column=1, sticky="w")
        self.cmb_unit.bind("<<ComboboxSelected>>", lambda e: self._refresh_spaces())
        self.v_para = tk.StringVar(value="强化学习")
        ttk.Label(base, text="决策范式").grid(row=1, column=2, sticky="e", padx=4)
        for i, p in enumerate(["强化学习", "深度学习"]):
            ttk.Radiobutton(base, text=p, value=p, variable=self.v_para,
                            command=self._on_para).grid(row=1, column=3 + i, sticky="w")

        # RL 区
        ttk.Label(f, text="② 网络结构（强化学习）", font=("", 11, "bold")).grid(row=row, column=0, sticky="w", pady=(8, 2)); row += 1
        self.frm_rl = ttk.LabelFrame(f); self.frm_rl.grid(row=row, column=0, sticky="ew", pady=2); row += 1
        self.v_algo = tk.StringVar(value="PPO"); self.v_backbone = tk.StringVar(value="Transformer")
        ttk.Label(self.frm_rl, text="算法").grid(row=0, column=0, sticky="e", padx=4, pady=2)
        self.cmb_algo = ttk.Combobox(self.frm_rl, textvariable=self.v_algo, values=ALGORITHMS,
                                     state="readonly", width=18)
        self.cmb_algo.grid(row=0, column=1, sticky="w")
        self.cmb_algo.bind("<<ComboboxSelected>>", lambda e: self._on_algo())
        ttk.Label(self.frm_rl, text="骨干网络").grid(row=0, column=2, sticky="e", padx=4)
        self.cmb_bb = ttk.Combobox(self.frm_rl, textvariable=self.v_backbone, values=BACKBONES,
                                   state="readonly", width=18)
        self.cmb_bb.grid(row=0, column=3, sticky="w")
        ttk.Button(self.frm_rl, text="自定义骨干…",
                   command=lambda: messagebox.showinfo(
                       "自定义骨干", "进入网络画布：仅中间堆叠区可编辑，决策头由算法锁定。\n（原型未含画布组件，正式版接入）")
                   ).grid(row=0, column=4, padx=6)
        self.lbl_struct = ttk.Label(self.frm_rl, foreground="#0a6")
        self.lbl_struct.grid(row=1, column=0, columnspan=5, sticky="w", padx=4)
        self._on_algo()

        # DL 区
        ttk.Label(f, text="② 网络结构（深度学习）", font=("", 11, "bold")).grid(row=row, column=0, sticky="w", pady=(8, 2)); row += 1
        self.frm_dl = ttk.LabelFrame(f); self.frm_dl.grid(row=row, column=0, sticky="ew", pady=2); row += 1
        self.v_dl_task = tk.StringVar(value="时序预测"); self.v_dl_tpl = tk.StringVar(value="LSTM")
        self.v_dl_data = tk.StringVar(value="tank_log_2026q3.csv")
        ttk.Label(self.frm_dl, text="任务类型").grid(row=0, column=0, sticky="e", padx=4, pady=2)
        ttk.Combobox(self.frm_dl, textvariable=self.v_dl_task, values=DL_TASKS,
                     state="readonly", width=16).grid(row=0, column=1, sticky="w")
        ttk.Label(self.frm_dl, text="网络模板").grid(row=0, column=2, sticky="e", padx=4)
        ttk.Combobox(self.frm_dl, textvariable=self.v_dl_tpl, values=BACKBONES,
                     state="readonly", width=16).grid(row=0, column=3, sticky="w")
        ttk.Label(self.frm_dl, text="数据集").grid(row=1, column=0, sticky="e", padx=4, pady=2)
        ttk.Entry(self.frm_dl, textvariable=self.v_dl_data, width=30).grid(row=1, column=1, columnspan=3, sticky="w")

        # 输入/输出
        ttk.Label(f, text="③ 观测 / 动作空间", font=("", 11, "bold")).grid(row=row, column=0, sticky="w", pady=(8, 2)); row += 1
        io = ttk.LabelFrame(f); io.grid(row=row, column=0, sticky="ew", pady=2); row += 1
        ttk.Label(io, text="观测项（多选）").grid(row=0, column=0, sticky="nw", padx=4, pady=2)
        self.lst_obs = tk.Listbox(io, selectmode="multiple", exportselection=False, height=6, width=28)
        self.lst_obs.grid(row=0, column=1, padx=4)
        ttk.Label(io, text="动作项（多选）").grid(row=0, column=2, sticky="nw", padx=4)
        self.lst_act = tk.Listbox(io, selectmode="multiple", exportselection=False, height=6, width=28)
        self.lst_act.grid(row=0, column=3, padx=4)
        ttk.Label(io, text="离散动作：勾选即动作清单；连续动作在正式版中\n此处切换为维度与上下界配置。",
                  foreground="#666").grid(row=1, column=1, columnspan=3, sticky="w")

        # 奖励函数
        ttk.Label(f, text="④ 奖励函数（强化学习）", font=("", 11, "bold")).grid(row=row, column=0, sticky="w", pady=(8, 2)); row += 1
        self.frm_rw = ttk.LabelFrame(f); self.frm_rw.grid(row=row, column=0, sticky="ew", pady=2); row += 1
        self.reward_vars = {}
        for i, (item, default) in enumerate(REWARD_ITEMS):
            bv = tk.BooleanVar(value=True); wv = tk.StringVar(value=str(default))
            ttk.Checkbutton(self.frm_rw, text=item, variable=bv).grid(row=i, column=0, sticky="w", padx=4)
            ttk.Entry(self.frm_rw, textvariable=wv, width=8).grid(row=i, column=1, sticky="w")
            self.reward_vars[item] = (bv, wv)
        ttk.Button(self.frm_rw, text="保存为模板…",
                   command=lambda: messagebox.showinfo("奖励模板", "已保存到资源库（原型为模拟）")
                   ).grid(row=0, column=2, rowspan=2, padx=10)

        # 高级训练选项（演化训练）
        ttk.Label(f, text="⑤ 高级训练选项（演化训练）", font=("", 11, "bold")).grid(row=row, column=0, sticky="w", pady=(8, 2)); row += 1
        adv = ttk.LabelFrame(f); adv.grid(row=row, column=0, sticky="ew", pady=2); row += 1
        self.adv_vars = {}
        self.adv_widgets = {}
        for i, m in enumerate(ADV_METHODS):
            bv = tk.BooleanVar(value=(m == "增量续训"))
            cb = ttk.Checkbutton(adv, text=m, variable=bv, command=lambda mm=m: self._on_adv(mm))
            cb.grid(row=i, column=0, sticky="w", padx=4)
            self.adv_vars[m] = bv
            self.adv_widgets[m] = []
        # 迁移学习：预训练模型
        self.v_pretrain = tk.StringVar()
        w = ttk.Combobox(adv, textvariable=self.v_pretrain, values=["imagenet_resnet50.pt", "moba_macro_v3.pt", "bc_tank_v1.pt"], width=22)
        w.grid(row=0, column=1, sticky="w"); self.adv_widgets["迁移学习"].append(w)
        # 对抗训练：对手池规模
        self.sp_pool = ttk.Spinbox(adv, from_=2, to=50, width=6)
        self.sp_pool.set(10)
        self.sp_pool.grid(row=1, column=1, sticky="w")
        self.adv_widgets["对抗训练"].append(self.sp_pool)
        # 增量续训：说明标签
        lb = ttk.Label(adv, text="自动落盘 checkpoint，默认开启", foreground="#666")
        lb.grid(row=2, column=1, sticky="w"); self.adv_widgets["增量续训"].append(lb)
        # 蒸馏：teacher
        self.v_teacher = tk.StringVar()
        self.cmb_teacher = ttk.Combobox(adv, textvariable=self.v_teacher, width=22)
        self.cmb_teacher.grid(row=3, column=1, sticky="w"); self.adv_widgets["模型蒸馏"].append(self.cmb_teacher)
        # 多任务
        self.v_mt = tk.StringVar(value=MT_GROUPS[0])
        w = ttk.Combobox(adv, textvariable=self.v_mt, values=MT_GROUPS, width=22)
        w.grid(row=4, column=1, sticky="w"); self.adv_widgets["多任务联合"].append(w)
        # 集成
        self.v_ens = tk.StringVar(value="置信度加权")
        w = ttk.Combobox(adv, textvariable=self.v_ens, values=["投票", "置信度加权"], width=22)
        w.grid(row=5, column=1, sticky="w")
        lb = ttk.Label(adv, text="（仅推理生效）", foreground="#666")
        lb.grid(row=5, column=2, sticky="w")
        self.adv_widgets["集成投票"] += [w, lb]
        for m in ADV_METHODS:
            self._on_adv(m)

        # 校验 + 保存
        bar = ttk.Frame(f); bar.grid(row=row, column=0, sticky="ew", pady=10); row += 1
        self.lbl_check = ttk.Label(bar, text="尚未校验", foreground="#a60")
        self.lbl_check.pack(side="left", padx=6)
        ttk.Button(bar, text="校验", command=self._validate).pack(side="right", padx=4)
        ttk.Button(bar, text="保存构建体", command=self._save_build).pack(side="right", padx=4)

        self._refresh_spaces()
        self._on_para()

    # ---------------- 构建页逻辑 ----------------
    def _refresh_spaces(self):
        unit = self.v_unit.get() or UNIT_TYPES[0]
        for lst, lib in ((self.lst_obs, OBS_LIBRARY[unit]), (self.lst_act, ACT_LIBRARY[unit])):
            lst.delete(0, "end")
            for it in lib:
                lst.insert("end", it)

    def _on_para(self):
        is_rl = self.v_para.get() == "强化学习"
        set_state(self.frm_rl, "normal" if is_rl else "disabled")
        set_state(self.frm_dl, "disabled" if is_rl else "normal")
        set_state(self.frm_rw, "normal" if is_rl else "disabled")
        for m, bv in self.adv_vars.items():
            if m == "增量续训" and not is_rl:
                pass
            self._on_adv(m)

    def _on_algo(self):
        algo = self.v_algo.get()
        self.lbl_struct.config(text=ALGO_STRUCTURAL.get(algo, "决策头（actor/critic 或 Q 头）由算法模板自动生成，不可编辑"))

    def _on_adv(self, m):
        on = self.adv_vars[m].get()
        for w in self.adv_widgets[m]:
            try:
                w.configure(state="normal" if on else "disabled")
            except tk.TclError:
                pass

    def _new_form(self):
        self.v_name.set(""); self.v_desc.set("")
        self.tree_builds.selection_remove(self.tree_builds.selection())
        self.lbl_check.config(text="尚未校验", foreground="#a60")

    def _collect_cfg(self):
        cfg = {"名称": self.v_name.get().strip(), "描述": self.v_desc.get(),
               "单位类型": self.v_unit.get(), "范式": self.v_para.get()}
        if cfg["范式"] == "强化学习":
            cfg["算法"] = self.v_algo.get(); cfg["骨干"] = self.v_backbone.get()
            cfg["观测"] = [self.lst_obs.get(i) for i in self.lst_obs.curselection()]
            cfg["动作"] = [self.lst_act.get(i) for i in self.lst_act.curselection()]
            cfg["奖励"] = {item: self.reward_vars[item][1].get()
                           for item, (bv, wv) in self.reward_vars.items() if bv.get()}
            cfg["高级选项"] = {m: bv.get() for m, bv in self.adv_vars.items()}
        else:
            cfg["任务类型"] = self.v_dl_task.get(); cfg["网络模板"] = self.v_dl_tpl.get()
            cfg["数据集"] = self.v_dl_data.get()
        return cfg

    def _validate(self, silent=False):
        errs = []
        cfg = self._collect_cfg()
        if not cfg["名称"]:
            errs.append("名称为空")
        if not cfg["单位类型"]:
            errs.append("未选择单位类型")
        if cfg["范式"] == "强化学习":
            if not cfg["观测"]:
                errs.append("未选择观测项")
            if not cfg["动作"]:
                errs.append("未选择动作项")
            if not cfg["奖励"]:
                errs.append("至少勾选一个奖励项")
            for it, w in cfg["奖励"].items():
                try:
                    float(w)
                except ValueError:
                    errs.append(f"奖励项「{it}」权重不是数字")
            if cfg["高级选项"].get("迁移学习") and not self.v_pretrain.get():
                errs.append("勾选了迁移学习但未选预训练模型")
            if cfg["高级选项"].get("模型蒸馏") and not self.v_teacher.get():
                errs.append("勾选了模型蒸馏但未选 teacher 模型")
        if errs:
            self.lbl_check.config(text="校验未通过：" + "；".join(errs), foreground="#c22")
            if not silent:
                messagebox.showwarning("校验未通过", "\n".join(errs))
        else:
            self.lbl_check.config(text="校验通过：契约检查 + dry-run 均成功（模拟）", foreground="#0a6")
            if not silent:
                messagebox.showinfo("校验通过", "观测/动作/奖励契约检查通过，\ndry-run 前向正常。")
        return errs

    def _save_build(self):
        if self._validate(silent=True):
            messagebox.showwarning("无法保存", "请先通过校验")
            return
        cfg = self._collect_cfg()
        name = cfg["名称"]
        versions = self.builds.setdefault(name, [])
        cfg["版本"] = len(versions) + 1
        cfg["保存时间"] = time.strftime("%H:%M:%S")
        versions.append(cfg)
        self._refresh_build_tree()
        self._refresh_train_combos()
        messagebox.showinfo("已保存", f"构建体「{name}」已保存为 v{cfg['版本']}")

    def _refresh_build_tree(self):
        self.tree_builds.delete(*self.tree_builds.get_children())
        for name, vers in self.builds.items():
            for cfg in vers:
                self.tree_builds.insert("", "end", iid=f"{name}|v{cfg['版本']}",
                                        text=name, values=(f"v{cfg['版本']}", cfg["范式"], cfg["单位类型"]))

    def _load_selected(self):
        sel = self.tree_builds.selection()
        if not sel:
            messagebox.showinfo("提示", "先在左侧选中一个构建体版本")
            return
        name, v = sel[0].split("|")
        ver = int(v[1:])
        cfg = self.builds[name][ver - 1]
        self.v_name.set(cfg["名称"]); self.v_desc.set(cfg.get("描述", ""))
        self.v_unit.set(cfg["单位类型"]); self.v_para.set(cfg["范式"])
        self._refresh_spaces()
        if cfg["范式"] == "强化学习":
            self.v_algo.set(cfg["算法"]); self.v_backbone.set(cfg["骨干"])
            self._on_algo()
            items = list(self.lst_obs.get(0, "end"))
            for it in cfg["观测"]:
                if it in items:
                    self.lst_obs.selection_set(items.index(it))
            items = list(self.lst_act.get(0, "end"))
            for it in cfg["动作"]:
                if it in items:
                    self.lst_act.selection_set(items.index(it))
            for item, (bv, wv) in self.reward_vars.items():
                bv.set(item in cfg["奖励"])
                if item in cfg["奖励"]:
                    wv.set(cfg["奖励"][item])
            for m, bv in self.adv_vars.items():
                bv.set(cfg["高级选项"].get(m, False))
                self._on_adv(m)
        else:
            self.v_dl_task.set(cfg["任务类型"]); self.v_dl_tpl.set(cfg["网络模板"])
            self.v_dl_data.set(cfg["数据集"])
        self._on_para()
        self.lbl_check.config(text=f"已载入 {name} v{ver}（保存将产生新版本）", foreground="#06a")

    # ================= 训练管理 =================
    def _build_tab_train(self):
        form = ttk.LabelFrame(self.tab_train, text="创建训练任务")
        form.pack(fill="x", padx=6, pady=6)
        self.v_t_build = tk.StringVar(); self.v_t_scene = tk.StringVar(value=SCENARIOS[0])
        self.v_t_start = tk.StringVar(value="从零训练")
        self.v_t_ckpt = tk.StringVar(); self.v_t_reward = tk.StringVar(value="使用构建体配置")
        self.v_lr = tk.StringVar(value="3e-4"); self.v_batch = tk.StringVar(value="256"); self.v_gamma = tk.StringVar(value="0.99")
        ttk.Label(form, text="构建体").grid(row=0, column=0, sticky="e", padx=4, pady=4)
        self.cmb_t_build = ttk.Combobox(form, textvariable=self.v_t_build, state="readonly", width=28)
        self.cmb_t_build.grid(row=0, column=1, sticky="w")
        ttk.Label(form, text="场景").grid(row=0, column=2, sticky="e", padx=4)
        ttk.Combobox(form, textvariable=self.v_t_scene, values=SCENARIOS, state="readonly", width=16).grid(row=0, column=3, sticky="w")
        ttk.Label(form, text="训练起点").grid(row=1, column=0, sticky="e", padx=4)
        for i, s in enumerate(["从零训练", "增量续训", "迁移初始化"]):
            ttk.Radiobutton(form, text=s, value=s, variable=self.v_t_start,
                            command=self._on_start).grid(row=1, column=1 + i, sticky="w")
        ttk.Label(form, text="历史模型").grid(row=2, column=0, sticky="e", padx=4)
        self.cmb_t_ckpt = ttk.Combobox(form, textvariable=self.v_t_ckpt, state="disabled", width=28)
        self.cmb_t_ckpt.grid(row=2, column=1, sticky="w")
        ttk.Label(form, text="奖励函数").grid(row=2, column=2, sticky="e", padx=4)
        ttk.Combobox(form, textvariable=self.v_t_reward,
                     values=["使用构建体配置", "模板A（进攻）", "模板B（防守）"], state="readonly", width=16).grid(row=2, column=3, sticky="w")
        ttk.Label(form, text="学习率").grid(row=3, column=0, sticky="e", padx=4)
        ttk.Entry(form, textvariable=self.v_lr, width=10).grid(row=3, column=1, sticky="w")
        ttk.Label(form, text="batch").grid(row=3, column=1, sticky="e", padx=4)
        ttk.Entry(form, textvariable=self.v_batch, width=8).grid(row=3, column=2, sticky="w")
        ttk.Label(form, text="γ").grid(row=3, column=2, sticky="e")
        ttk.Entry(form, textvariable=self.v_gamma, width=6).grid(row=3, column=3, sticky="w")
        ttk.Button(form, text="启动训练", command=self._start_train).grid(row=3, column=4, rowspan=2, padx=8, pady=4)

        pan = ttk.PanedWindow(self.tab_train, orient="horizontal")
        pan.pack(fill="both", expand=True, padx=6, pady=6)
        mid = ttk.Frame(pan); pan.add(mid, weight=2)
        ttk.Label(mid, text="任务列表", font=("", 10, "bold")).pack(anchor="w")
        cols = ("build", "scene", "status", "prog")
        self.tree_tasks = ttk.Treeview(mid, columns=cols, show="headings")
        for cid, txt, w in (("build", "构建体", 200), ("scene", "场景", 100),
                            ("status", "状态", 80), ("prog", "进度", 70)):
            self.tree_tasks.heading(cid, text=txt)
            self.tree_tasks.column(cid, width=w, anchor="center")
        self.tree_tasks.pack(fill="both", expand=True)
        self.pg = ttk.Progressbar(mid, length=200, mode="determinate")
        self.pg.pack(fill="x", pady=4)

        right = ttk.Frame(pan); pan.add(right, weight=3)
        ttk.Label(right, text="训练日志", font=("", 10, "bold")).pack(anchor="w")
        self.txt_log = tk.Text(right, height=20, state="disabled", bg="#111", fg="#8f8", insertbackground="#8f8")
        self.txt_log.pack(fill="both", expand=True)

    def _refresh_train_combos(self):
        labels = [f"{n} v{c['版本']}" for n, vers in self.builds.items() for c in vers]
        self.cmb_t_build.config(values=labels)
        self.cmb_teacher.config(values=labels + ["外部模型…"])
        self._on_start()

    def _on_start(self):
        on = self.v_t_start.get() != "从零训练"
        self.cmb_t_ckpt.config(state="readonly" if on else "disabled")
        if on:
            labels = [f"{n} v{c['版本']}" for n, vers in self.builds.items() for c in vers]
            self.cmb_t_ckpt.config(values=labels)

    def _start_train(self):
        label = self.v_t_build.get()
        if not label:
            messagebox.showwarning("提示", "请先选择一个构建体版本")
            return
        tid = f"T{len(self.tasks) + 1:03d}"
        rec = {"id": tid, "build": label, "scene": self.v_t_scene.get(),
               "start": self.v_t_start.get(), "status": "采样中", "prog": 0.0}
        self.tasks[tid] = rec
        self.tree_tasks.insert("", "end", iid=tid, values=(label, rec["scene"], "采样中", "0%"))
        self._log(f"[{tid}] 任务创建：{label} / {rec['scene']} / {rec['start']}")
        q = queue.Queue()
        self.queues[tid] = q
        threading.Thread(target=self._train_worker, args=(tid, q), daemon=True).start()
        self.after(150, lambda: self._poll(tid))

    # 接入真实引擎时替换本方法：向引擎发训练指令，从引擎收指标
    def _train_worker(self, tid, q):
        loss, reward = 2.5, -50.0
        for i in range(1, 31):
            time.sleep(0.25)
            loss *= random.uniform(0.85, 0.97)
            reward += random.uniform(2, 12)
            q.put(("log", f"[{tid}] iter {i * 100:>5}  loss={loss:8.4f}  mean_reward={reward:8.1f}"))
            q.put(("prog", i / 30))
        q.put(("log", f"[{tid}] 训练完成：胜率 {random.uniform(55, 85):.1f}%（评估 100 局）"))
        q.put(("done", f"{tid}_model.pt"))

    def _poll(self, tid):
        q = self.queues.get(tid)
        if q is None:
            return
        try:
            while True:
                kind, payload = q.get_nowait()
                if kind == "log":
                    self._log(payload)
                elif kind == "prog":
                    self.tasks[tid]["prog"] = payload
                    self.tree_tasks.set(tid, "prog", f"{payload:.0%}")
                    self.pg["value"] = payload * 100
                elif kind == "done":
                    self.tasks[tid]["status"] = "已完成"
                    self.tree_tasks.set(tid, "status", "已完成")
                    name = f"{self.tasks[tid]['build'].replace(' ', '')}_{payload}"
                    self.model_rows.append((name, self.tasks[tid]["build"], self.tasks[tid]["scene"], "胜率 71.3%"))
                    self._refresh_models()
                    self._log(f"[{tid}] 模型已导出：{name}（ONNX + .pt）")
                    del self.queues[tid]
                    return
        except queue.Empty:
            pass
        self.after(150, lambda: self._poll(tid))

    def _log(self, msg):
        self.txt_log.config(state="normal")
        self.txt_log.insert("end", msg + "\n")
        self.txt_log.see("end")
        self.txt_log.config(state="disabled")

    # ================= 模型库 =================
    def _build_tab_model(self):
        cols = ("name", "build", "scene", "metric")
        self.tree_models = ttk.Treeview(self.tab_model, columns=cols, show="headings")
        for cid, txt, w in (("name", "模型文件", 260), ("build", "所属构建体", 200),
                            ("scene", "训练场景", 120), ("metric", "指标", 140)):
            self.tree_models.heading(cid, text=txt)
            self.tree_models.column(cid, width=w, anchor="center")
        self.tree_models.pack(fill="both", expand=True, padx=6, pady=6)
        bar = ttk.Frame(self.tab_model); bar.pack(fill="x", padx=6)
        ttk.Button(bar, text="导出 ONNX",
                   command=lambda: messagebox.showinfo("导出", "已导出 ONNX（原型为模拟）")).pack(side="left", padx=4)
        ttk.Button(bar, text="查看回放",
                   command=lambda: messagebox.showinfo("回放", "接入引擎回放后提供（原型为模拟）")).pack(side="left", padx=4)

    def _refresh_models(self):
        self.tree_models.delete(*self.tree_models.get_children())
        for row in self.model_rows:
            self.tree_models.insert("", "end", values=row)


if __name__ == "__main__":
    App().mainloop()
