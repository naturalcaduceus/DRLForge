# -*- coding: utf-8 -*-
"""ChessRemoteEnv 协议测试：用标准库起一个假环境服务，不需要 chess 仓库与 torch。

    python -m unittest discover -s tests -p "test_*.py"
"""
import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fz_agent.env.ChessEnv import ChessRemoteEnv

OBS_SHAPE = [5, 4]
ACTIONS = 7


class _Stub(BaseHTTPRequestHandler):
    tick = 0
    received = []

    def log_message(self, fmt, *args):
        pass

    def _send(self, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n).decode("utf-8")) if n else {}

    def _obs(self):
        return [[float(i + j) for j in range(OBS_SHAPE[1])] for i in range(OBS_SHAPE[0])]

    def do_GET(self):
        cls = type(self)
        if self.path.startswith("/rl/spec"):
            self._send({"ok": True, "sid": "s", "side": "red",
                        "obs": {"shape": OBS_SHAPE, "dim": OBS_SHAPE[1],
                                "names": ["a", "b", "c", "d"], "entity_set": True},
                        "action": {"space_type": "discrete", "dim": ACTIONS},
                        "sides": ["red", "blue"], "max_ticks": 100})
        elif self.path.startswith("/rl/observe"):
            self._send({"ok": True, "side": "blue", "obs": self._obs(),
                        "legal": [{"type": "wait"}], "legal_count": 1})
        else:
            self._send({"ok": False, "error": "not found"})

    def do_POST(self):
        cls = type(self)
        body = self._body()
        cls.received.append((self.path, body))
        if self.path == "/rl/reset":
            cls.tick = 0
            self._send({"ok": True, "sid": body.get("sid"), "side": body.get("side"),
                        "obs": self._obs(), "obs_shape": OBS_SHAPE, "reward": 0.0,
                        "terminated": False, "truncated": False, "done": False,
                        "tick": 0, "legal": [{"type": "wait"}], "legal_count": 1})
        elif self.path == "/rl/step":
            cls.tick += 1
            done = cls.tick >= 2
            self._send({"ok": True, "sid": body.get("sid"), "side": body.get("side"),
                        "obs": self._obs(), "obs_shape": OBS_SHAPE, "reward": 0.25,
                        "terminated": done, "truncated": False, "done": done,
                        "winner": "red" if done else None, "tick": cls.tick,
                        "legal": [{"type": "wait"}], "legal_count": 1,
                        "events": [{"type": "damage"}]})
        elif self.path == "/session/close":
            self._send({"ok": True, "closed": True})
        else:
            self._send({"ok": False, "error": "not found"})


class ChessRemoteEnvTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Stub)
        cls.url = "http://127.0.0.1:%d" % cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def setUp(self):
        _Stub.tick = 0
        _Stub.received.clear()

    def _env(self, **kwargs):
        return ChessRemoteEnv(self.url, side="red", seed=42, **kwargs)

    def test_spec_from_server(self):
        env = self._env()
        self.assertEqual(env.max_entities, OBS_SHAPE[0])
        self.assertEqual(env.obs_dim, OBS_SHAPE[1])
        self.assertEqual(env.max_actions, ACTIONS)
        self.assertEqual(env.io_spec.obs.shape, OBS_SHAPE)
        self.assertEqual(env.io_spec.action.dim, ACTIONS)
        self.assertEqual(env.opponent_side, "blue")
        env.close()

    def test_reset_returns_entity_set(self):
        env = self._env()
        obs, info = env.reset(seed=1)
        self.assertEqual(obs.shape, tuple(OBS_SHAPE))
        self.assertEqual(obs.dtype, np.float32)
        self.assertEqual(info["tick"], 0)
        self.assertEqual(info["legal_actions"][0]["type"], "wait")
        env.close()

    def test_step_reward_and_result(self):
        env = self._env()
        env.reset(seed=1)
        obs, r, term, trunc, info = env.step(1)
        self.assertEqual(obs.shape, tuple(OBS_SHAPE))
        self.assertAlmostEqual(r, 0.25)
        self.assertFalse(term)
        self.assertEqual(info["result"], "running")
        self.assertEqual(info["reward"], {"red": 0.25})
        obs, r, term, trunc, info = env.step(1)
        self.assertTrue(term)
        self.assertEqual(info["result"], "win")
        self.assertEqual(info["winner"], "red")
        env.close()

    def test_out_of_range_action_is_sent_as_index(self):
        env = self._env()
        env.reset(seed=1)
        env.step(999)
        path, body = _Stub.received[-1]
        self.assertEqual(path, "/rl/step")
        self.assertEqual(body["actions"]["red"], 999)
        env.close()

    def test_opponent_policy_dual_actions(self):
        env = self._env()
        env.set_opponent(lambda obs: np.asarray([3]))
        env.reset(seed=1)
        env.step(2)
        path, body = _Stub.received[-1]
        self.assertEqual(path, "/rl/step")
        self.assertEqual(body["actions"], {"blue": 3, "red": 2})
        env.close()

    def test_opponent_policy_command_dict(self):
        env = self._env()
        env.set_opponent(lambda obs: [{"type": "wait"}])
        env.reset(seed=1)
        env.step(0)
        _, body = _Stub.received[-1]
        self.assertEqual(body["commands"]["blue"], [{"type": "wait"}])
        self.assertEqual(body["actions"]["red"], 0)
        env.close()


if __name__ == "__main__":
    unittest.main()