from __future__ import annotations

import json
import sys

from app.core.errors import BusinessError
from app.services.local_transformer_service import _generate_answer_in_process


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        answer, model_reference, metrics = _generate_answer_in_process(
            question=str(payload.get("question") or ""),
            contexts=list(payload.get("contexts") or []),
            model_name=payload.get("model_name"),
            temperature=payload.get("temperature"),
            max_new_tokens=payload.get("max_new_tokens"),
            system_instruction=payload.get("system_instruction"),
            kb_hit=payload.get("kb_hit"),
        )
        sys.stdout.write(
            json.dumps(
                {
                    "answer": answer,
                    "model_reference": model_reference,
                    "metrics": metrics,
                },
                ensure_ascii=False,
            )
        )
        return 0
    except BusinessError as exc:
        sys.stderr.write(str(exc))
        return 2
    except Exception as exc:  # pragma: no cover
        sys.stderr.write(f"{exc.__class__.__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
