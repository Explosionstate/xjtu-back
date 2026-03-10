from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ops import main as ops_main


def main() -> int:
    sys.argv = ["ops.py", "check", "--probe", *sys.argv[1:]]
    return ops_main()


if __name__ == "__main__":
    raise SystemExit(main())
