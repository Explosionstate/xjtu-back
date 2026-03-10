from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import urllib.request


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")


def _listener_pids(port: int) -> list[int]:
    proc = _run(["netstat", "-ano"])
    pids: set[int] = set()
    for line in proc.stdout.splitlines():
        text = line.strip()
        if not text or "LISTENING" not in text:
            continue
        parts = text.split()
        if len(parts) < 5:
            continue
        local_addr = parts[1]
        pid_text = parts[-1]
        if local_addr.endswith(f":{port}") and pid_text.isdigit():
            pids.add(int(pid_text))
    return sorted(pids)


def _pid_command_map(pids: list[int]) -> dict[int, str]:
    if not pids:
        return {}
    where = " OR ".join([f"ProcessId={pid}" for pid in pids])
    cmd = [
        "powershell",
        "-NoProfile",
        "-Command",
        (
            f'Get-CimInstance Win32_Process -Filter "{where}" | '
            "Select-Object ProcessId,CommandLine | ConvertTo-Json -Depth 3"
        ),
    ]
    proc = _run(cmd)
    if proc.returncode != 0 or not proc.stdout.strip():
        return {}
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {}

    rows = payload if isinstance(payload, list) else [payload]
    output: dict[int, str] = {}
    for row in rows:
        try:
            pid = int(row.get("ProcessId"))
            output[pid] = (row.get("CommandLine") or "").strip()
        except Exception:
            continue
    return output


def _port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def _print_conflicts(port: int) -> None:
    pids = _listener_pids(port)
    if not pids:
        print(f"[INFO] Port {port} is free")
        return
    print(f"[WARN] Port {port} is occupied by PIDs: {pids}")
    mapping = _pid_command_map(pids)
    for pid in pids:
        print(f"  - PID {pid}: {mapping.get(pid, '<command line unavailable>')}")


def cmd_start(args: argparse.Namespace) -> int:
    if _port_in_use(args.host, args.port):
        _print_conflicts(args.port)
        if args.force_stop:
            stop_args = argparse.Namespace(port=args.port, all_python=False)
            code = cmd_stop(stop_args)
            if code != 0:
                return code
        else:
            print("[ABORT] Use --force-stop to kill conflicting listeners first.")
            return 1

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if args.reload:
        cmd.append("--reload")
    print("[INFO] Starting:", " ".join(cmd))
    return subprocess.call(cmd)


def cmd_stop(args: argparse.Namespace) -> int:
    pids = _listener_pids(args.port)
    if args.all_python:
        proc = _run(["taskkill", "/F", "/IM", "python.exe"])
        print(proc.stdout.strip() or proc.stderr.strip() or "[INFO] taskkill finished")
        return 0 if proc.returncode == 0 else 1

    if not pids:
        print(f"[INFO] No listener found on {args.port}")
        return 0

    mapping = _pid_command_map(pids)
    print(f"[INFO] Stopping listeners on {args.port}: {pids}")
    for pid in pids:
        print(f"  - PID {pid}: {mapping.get(pid, '')}")
        proc = _run(["taskkill", "/F", "/PID", str(pid)])
        message = proc.stdout.strip() or proc.stderr.strip()
        if message:
            print(f"    {message}")
    return 0


def _fetch_json(url: str, timeout: int) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="ignore"))


def cmd_check(args: argparse.Namespace) -> int:
    _print_conflicts(args.port)
    if not args.probe:
        return 0

    base = args.base.rstrip("/")
    try:
        health = _fetch_json(f"{base}/health", timeout=args.timeout)
        print("[OK] /health", health)
        openapi = _fetch_json(f"{base}/openapi.json", timeout=args.timeout)
        print(
            "[OK] /openapi.json",
            {"openapi": openapi.get("openapi"), "paths": len(openapi.get("paths", {}))},
        )
        return 0
    except Exception as exc:
        print(f"[FAIL] probe failed: {exc}")
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="xjtu-back operations")
    sub = parser.add_subparsers(dest="command", required=True)

    start = sub.add_parser("start", help="start backend service")
    start.add_argument("--host", default="127.0.0.1")
    start.add_argument("--port", type=int, default=8000)
    start.add_argument("--reload", action="store_true")
    start.add_argument("--force-stop", action="store_true")
    start.set_defaults(func=cmd_start)

    stop = sub.add_parser("stop", help="stop backend service")
    stop.add_argument("--port", type=int, default=8000)
    stop.add_argument("--all-python", action="store_true")
    stop.set_defaults(func=cmd_stop)

    check = sub.add_parser("check", help="check port and health")
    check.add_argument("--port", type=int, default=8000)
    check.add_argument("--probe", action="store_true")
    check.add_argument("--base", default="http://127.0.0.1:8000")
    check.add_argument("--timeout", type=int, default=8)
    check.set_defaults(func=cmd_check)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
