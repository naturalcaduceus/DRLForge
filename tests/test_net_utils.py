# -*- coding: utf-8 -*-
"""WSL 侧通信工具测试：无代理、候选顺序、端口探测、自动定位。"""
import json
import os
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from fz_agent.net_utils import (candidate_hosts, probe, request_json,
                                resolve_host)


class _Stub(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        body = json.dumps({"ok": True}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _serve():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, "http://127.0.0.1:%d" % httpd.server_address[1]


class NetUtilsTest(unittest.TestCase):
    def test_request_json_no_proxy(self):
        httpd, base = _serve()
        keys = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY")
        old = {k: os.environ.get(k) for k in keys}
        try:
            for k in keys:
                os.environ[k] = "http://127.0.0.1:9"
            self.assertTrue(request_json(base + "/x", timeout=3)["ok"])
        finally:
            for k, v in old.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
            httpd.shutdown()
            httpd.server_close()

    def test_candidate_order(self):
        old_url = os.environ.get("DRL_CHESS_URL")
        try:
            os.environ["DRL_CHESS_URL"] = "http://192.168.50.7:8000"
            hosts = candidate_hosts()
            self.assertEqual(hosts[0], "192.168.50.7", "显式 URL 的 host 优先")
            self.assertIn("127.0.0.1", hosts)
            self.assertEqual(len(hosts), len(set(hosts)))
        finally:
            if old_url is None:
                os.environ.pop("DRL_CHESS_URL", None)
            else:
                os.environ["DRL_CHESS_URL"] = old_url

    def test_probe_and_resolve(self):
        httpd, base = _serve()
        try:
            port = httpd.server_address[1]
            self.assertTrue(probe("127.0.0.1", port))
            self.assertEqual(resolve_host(port, verbose=False), "127.0.0.1")
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_resolve_fails_with_clear_error(self):
        with self.assertRaises(RuntimeError):
            resolve_host(1, env_url="http://127.0.0.1:1", verbose=False)


if __name__ == "__main__":
    unittest.main()