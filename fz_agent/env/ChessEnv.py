# -*- coding: utf-8 -*-
"""chess-engine2 远端环境：把 Windows 上的棋类内核当作 WSL 侧的训练环境。

链路（与 chess-engine2 的 tools/net_service 配套）：
    DRLForge(训练/推理) --HTTP/JSON--> chess-engine2 /rl/reset /rl/step

为什么环境侧编码：观测特征只保留一份（chess 的 envi.rl_adapter），
训练端只消费数值；对战时神经网络棋手复用同一份编码，避免「训练/部署漂移」。

接口语义：
    reset(seed)        → (obs[N, F], info)
    step(action:int)   → (obs, reward, terminated, truncated, info)
    action             legal_actions 的下标；越界由环境侧回退为 wait
    set_opponent(p)    p(obs_opp) → 动作下标/指令；用于自博弈与联赛

配置示例（fz_agent/configs/ppo_chess.json）：
    "env": {"type": "chess_remote",
            "config": {"url": "http://172.20.0.1:8000", "side": "red",
                       "max_actions": 32, "max_entities": 12}}
"""
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable, Optional

import numpy as np

from .base import ActionSpec, BaseEnv, IOSpec, ObsSpec

DEFAULT_OBS_FEATURES = ("dx", "dy", "hp_ratio", "atk_ratio", "armor",
                        "move_ratio", "is_enemy", "is_self", "alive", "tick_ratio")


class ChessRemoteError(RuntimeError):
    pass


class ChessRemoteEnv(BaseEnv):
    """通过 HTTP 驱动的 chess-engine2 环境（Gymnasium 风格五元组 step）。"""

    def __init__(self, url="http://127.0.0.1:8000", side="red", timeout=15.0,
                 max_actions=32, max_entities=12, obs_dim=None, names=None,
                 sid=None, seed=None, retries=2, retry_wait=0.2,
                 auto_spec=True, max_ticks=None):
        self.base_url = url.rstrip("/")
        self.side = side
        self.timeout = float(timeout)
        self.retries = max(0, int(retries))
        self.retry_wait = float(retry_wait)
        self.max_actions = int(max_actions) if max_actions else 32
        self.max_entities = int(max_entities) if max_entities else 12
        self.obs_dim = int(obs_dim) if obs_dim else len(DEFAULT_OBS_FEATURES)
        self.names = list(names or DEFAULT_OBS_FEATURES)
        self.max_ticks = max_ticks
        self.seed = seed
        self.sid = sid or "drlforge-%s-%s" % (side, seed if seed is not None else 0)

        self.opponent_policy: Optional[Callable] = None
        self.opponent_side: Optional[str] = None
        self._last_legal = [{"type": "wait"}]
        self._sides = [side]
        self._last_obs = None

        self.io_spec = self._make_spec()
        if auto_spec:
            self.refresh_spec()

    # ---------- 规格 ----------
    def _make_spec(self):
        return IOSpec(
            obs=ObsSpec(shape=[self.max_entities, self.obs_dim],
                        dim=self.obs_dim, entity_set=True,
                        names=list(self.names), normalize=True),
            action=ActionSpec(space_type="discrete", dim=self.max_actions),
        )

    def refresh_spec(self):
        """从环境服务读取观测/动作规格；失败时保留配置默认值。"""
        try:
            spec = self._request("/rl/spec", params={"side": self.side, "sid": self.sid})
        except ChessRemoteError as exc:
            print("[chess_remote] 读取 /rl/spec 失败，沿用配置默认规格：%s" % exc)
            return self.io_spec
        obs = spec.get("obs") or {}
        act = spec.get("action") or {}
        shape = obs.get("shape") or []
        if shape:
            self.max_entities = int(shape[0])
            self.obs_dim = int(obs.get("dim") or (shape[-1] if len(shape) > 1 else self.obs_dim))
        self.names = list(obs.get("names") or self.names)
        if act.get("dim"):
            self.max_actions = int(act["dim"])
        self.max_ticks = spec.get("max_ticks", self.max_ticks)
        self._sides = list(spec.get("sides") or self._sides)
        self.opponent_side = next((s for s in self._sides if s != self.side), None)
        self.io_spec = self._make_spec()
        return self.io_spec

    def io_dict(self):
        return self.io_spec.to_dict()

    # ---------- 传输 ----------
    def _request(self, path, payload=None, method=None, params=None):
        url = self.base_url + path
        if params:
            query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
            if query:
                url += ("&" if "?" in url else "?") + query
        data = None
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        method = method or ("POST" if data is not None else "GET")
        last = None
        for attempt in range(self.retries + 1):
            try:
                req = urllib.request.Request(
                    url, data=data, method=method,
                    headers={"Content-Type": "application/json; charset=utf-8"})
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                if isinstance(body, dict) and body.get("ok") is False:
                    raise ChessRemoteError(body.get("error") or "环境返回失败")
                return body
            except (urllib.error.URLError, OSError, ValueError) as exc:
                last = exc
                if attempt < self.retries and self.retry_wait > 0:
                    time.sleep(self.retry_wait)
        raise ChessRemoteError("chess-engine2 不可用（%s）：%s" % (url, last))

    # ---------- 观测/动作 ----------
    def _to_obs(self, payload):
        rows = payload.get("obs")
        arr = np.asarray(rows if rows is not None else [], dtype=np.float32)
        if arr.ndim == 1 and arr.size == self.max_entities * self.obs_dim:
            arr = arr.reshape(self.max_entities, self.obs_dim)
        return arr

    @staticmethod
    def _coerce_action(value):
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, dict):
            return 0
        arr = np.asarray(value).reshape(-1)
        return int(arr[0]) if arr.size else 0

    def _opponent_payload(self):
        """对手策略在本地出招：返回 {"actions": {...}} 或 {"commands": {...}}。"""
        if self.opponent_policy is None or not self.opponent_side:
            return {}
        out = self._request("/rl/observe", params={"side": self.opponent_side,
                                                   "sid": self.sid})
        decision = self.opponent_policy(self._to_obs(out))
        if isinstance(decision, dict) and decision.get("type"):
            return {"commands": {self.opponent_side: [dict(decision)]}}
        if isinstance(decision, (list, tuple)) and decision \
                and isinstance(decision[0], dict):
            return {"commands": {self.opponent_side: [dict(c) for c in decision]}}
        return {"actions": {self.opponent_side: self._coerce_action(decision)}}

    # ---------- RL 三件套 ----------
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed = seed
        out = self._request("/rl/reset", {"sid": self.sid, "seed": seed,
                                          "side": self.side})
        self._last_legal = out.get("legal") or [{"type": "wait"}]
        obs = self._to_obs(out)
        self._last_obs = obs
        return obs, {"legal_actions": self._last_legal, "tick": out.get("tick"),
                     "done": bool(out.get("done")), "sides": list(self._sides)}

    def step(self, action):
        payload = {"sid": self.sid, "side": self.side}
        try:
            idx = int(np.asarray(action).reshape(-1)[0])
        except (TypeError, ValueError, IndexError):
            idx = 0
        payload.update(self._opponent_payload())
        actions = dict(payload.get("actions") or {})
        actions[self.side] = idx
        payload["actions"] = actions

        out = self._request("/rl/step", payload)
        obs = self._to_obs(out)
        self._last_obs = obs
        self._last_legal = out.get("legal") or [{"type": "wait"}]
        self._tick = out.get("tick")

        terminated = bool(out.get("terminated"))
        truncated = bool(out.get("truncated"))
        winner = out.get("winner")
        reward = float(out.get("reward") or 0.0)
        info = dict(out.get("info") or {})
        info.update({
            "reward": {self.side: reward},
            "events": out.get("events") or [],
            "done": bool(out.get("done")),
            "winner": winner,
            "tick": out.get("tick"),
            "legal_actions": self._last_legal,
            "result": self._result(terminated, truncated, winner),
        })
        return obs, reward, terminated, truncated, info

    def _result(self, terminated, truncated, winner):
        if terminated:
            if winner in (None, ""):
                return "ended"
            if winner == self.side:
                return "win"
            if winner == "draw":
                return "draw"
            return "loss"
        if truncated:
            return "time_limit"
        return "running"

    # ---------- 对抗接入 ----------
    def set_opponent(self, policy):
        """对手策略由训练端本地前向；None 表示交回环境服务端托管（--auto）。"""
        self.opponent_policy = policy
        if policy is not None and not self.opponent_side:
            self.opponent_side = next((s for s in self._sides if s != self.side), None)

    def legal_actions(self):
        return list(self._last_legal)

    def render(self):
        print("[chess_remote] sid=%s tick=%s legal=%d"
              % (self.sid, getattr(self, "_tick", None), len(self._last_legal)))

    def close(self):
        try:
            self._request("/session/close", {"sid": self.sid})
        except ChessRemoteError:
            pass