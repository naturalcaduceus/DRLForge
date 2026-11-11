"""部署：多构建体推理 HTTP 服务（仅用标准库，无额外依赖）。

启动：
    python -m fz_agent.serve --bundle-dir fz_agent/bundles --port 8000
    python -m fz_agent.main serve --bundle fz_agent/bundles/ppo_combat --port 8000

接口：
    GET  /health           → {"status":"ok", ...}
    GET  /bundles          → {"bundles":[...], "count":N}
    POST /act              → 请求体 {"bundle_id":"...", "obs":[...], "deterministic":true}
                             或批量 {"requests":[{...}, {...}]}
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .construct import InferenceService, ModelPool


def build_service(bundle_dir=None, bundles=None) -> InferenceService:
    pool = ModelPool()
    if bundle_dir:
        loaded = pool.load_dir(bundle_dir)
        print(f"[serve] 从 {bundle_dir} 加载构建体: {loaded}")
    for path in bundles or []:
        from .construct import AgentBundle
        bundle = AgentBundle.load(path)
        pool.register(bundle)
        print(f"[serve] 加载构建体: {bundle.bundle_id} ({path})")
    if len(pool) == 0:
        raise RuntimeError("没有可用构建体，请指定 --bundle-dir 或 --bundle")
    return InferenceService(pool)


def make_handler(service: InferenceService):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                self._send(200, {"status": "ok", **service.list_models()})
            elif self.path == "/bundles":
                self._send(200, service.list_models())
            else:
                self._send(404, {"error": f"unknown path {self.path}"})

        def do_POST(self):  # noqa: N802
            if self.path != "/act":
                self._send(404, {"error": f"unknown path {self.path}"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length else b"{}"
                payload = json.loads(raw.decode("utf-8"))
            except Exception as e:  # noqa: BLE001
                self._send(400, {"error": f"invalid json: {e}"})
                return
            result = service.handle(payload)
            code = 200 if "error" not in result else 400
            self._send(code, result)

        def log_message(self, fmt, *args):  # 降噪
            return

    return Handler


def run_server(bundle_dir=None, bundles=None, host="0.0.0.0", port=8000):
    service = build_service(bundle_dir, bundles)
    httpd = ThreadingHTTPServer((host, port), make_handler(service))
    print(f"[serve] 监听 http://{host}:{port}  构建体={service.list_models()}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] 收到中断，正在退出")
    finally:
        httpd.server_close()


def main():
    parser = argparse.ArgumentParser(description="多构建体推理服务")
    parser.add_argument("--bundle-dir", default=None)
    parser.add_argument("--bundle", action="append", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    run_server(args.bundle_dir, args.bundle, args.host, args.port)


if __name__ == "__main__":
    main()
