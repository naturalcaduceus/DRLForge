# -*- coding: utf-8 -*-
"""WSL 侧跨机通信工具：无代理 HTTP + Windows 主机自动探测（与 chess-engine2 配套）。

三个实测坑：

1) **代理劫持**：Clash / 公司代理会拦截 `127.0.0.1`、`172.x.x.x` 的请求并返回 502。
   这里用 `ProxyHandler({})` 构建**局部** opener，只影响本项目请求，不改全局设置。

2) **地址漂移**：WSL2 NAT 模式下 Windows 主机 = WSL 的默认网关，重启会变。
   候选顺序：127.0.0.1（镜像网络）→ 默认网关 → /etc/resolv.conf → $WIN_HOST
   → $DRL_CHESS_URL 里的 host（若已显式配置则最先用）。

3) **探测而不是猜**：逐个 TCP 探测端口，选第一个能连上的地址，并打印过程。

诊断：

    python -m fz_agent.net_utils --port 8000
    python -m fz_agent.net_utils --url http://172.20.0.1:8000
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def make_opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def request_json(url, payload=None, method=None, timeout=5.0, headers=None):
    """发 JSON 请求并解析响应；绕过系统代理（避免 502 劫持）。"""
    data = None
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
    method = method or ("POST" if data is not None else "GET")
    hdrs = {"Content-Type": "application/json; charset=utf-8"}
    hdrs.update(headers or {})
    req = urllib.request.Request(url, data=data, method=method, headers=hdrs)
    with _OPENER.open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def probe(host, port, timeout=1.5):
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def _default_gateway():
    try:
        out = subprocess.check_output(["ip", "route", "show", "default"],
                                      text=True, timeout=2)
    except (OSError, subprocess.SubprocessError):
        return None
    parts = out.split()
    return parts[parts.index("via") + 1] if "via" in parts else None


def _resolv_nameserver():
    try:
        with open("/etc/resolv.conf", "r", encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("nameserver"):
                    return line.split()[1]
    except OSError:
        pass
    return None


def _host_of_url(url):
    try:
        return urllib.parse.urlparse(url).hostname
    except ValueError:
        return None


def candidate_hosts(env_url=None):
    """按优先级返回候选 Windows 主机地址（去重）。"""
    hosts = []
    for ip in (_host_of_url(env_url) if env_url else None,
               os.environ.get("DRL_CHESS_URL") and _host_of_url(os.environ["DRL_CHESS_URL"]),
               "127.0.0.1",
               _default_gateway(),
               _resolv_nameserver(),
               os.environ.get("WIN_HOST")):
        if ip and ip not in hosts:
            hosts.append(ip)
    return hosts


def resolve_host(port, env_url=None, verbose=True):
    """逐个探测候选地址，返回第一个可连的（用于未显式给 URL 时的自动探测）。"""
    hosts = candidate_hosts(env_url)
    for host in hosts:
        ok = probe(host, port)
        if verbose:
            print("  %s %s:%s" % ("✓" if ok else "✗", host, port))
        if ok:
            return host
    raise RuntimeError(
        "WSL 连不上 Windows 环境服务 %s 端口（候选: %s）。\n"
        "排查：1) Windows 端 python -m tools.net_service 是否显示 LISTENING ✓；"
        "2) 是否 --host 0.0.0.0；3) Windows 防火墙是否放行该端口。" % (port, ", ".join(hosts)))


def main(argv=None):
    ap = argparse.ArgumentParser(description="WSL→Windows 通信诊断")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--url", default=None, help="显式指定 http://host:port（优先）")
    ap.add_argument("--path", default="/health")
    args = ap.parse_args(argv)

    if args.url:
        base = args.url.rstrip("/")
        print("使用显式地址: %s" % base)
    else:
        print("候选地址:")
        try:
            host = resolve_host(args.port)
        except RuntimeError as exc:
            print("\n[失败] %s" % exc)
            return 1
        base = "http://%s:%d" % (host, args.port)
    url = base + args.path
    print("\nGET %s" % url)
    try:
        print(json.dumps(request_json(url, timeout=5), ensure_ascii=False)[:600])
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print("请求失败: %s" % exc)
        return 1
    print("\n通信成功 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())