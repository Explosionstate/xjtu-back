from __future__ import annotations

import argparse
import json
import urllib.request


def fetch_json(url: str, timeout: int) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        payload = resp.read().decode("utf-8", errors="ignore")
        return json.loads(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="xjtu-back health check")
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=int, default=8)
    args = parser.parse_args()

    base = args.base.rstrip("/")
    health = fetch_json(f"{base}/health", timeout=args.timeout)
    print("[OK] /health:", health)

    openapi = fetch_json(f"{base}/openapi.json", timeout=args.timeout)
    version = openapi.get("openapi")
    paths = len(openapi.get("paths", {}))
    print("[OK] /openapi.json:", {"openapi": version, "paths": paths})

    if not isinstance(version, str) or not version.startswith("3.0"):
        print("[WARN] OpenAPI version is not 3.0.x; docs UI may be incompatible")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
