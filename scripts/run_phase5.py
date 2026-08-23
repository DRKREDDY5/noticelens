"""Run the frozen Phase 5 final RAG-core evaluation."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from noticelens.phase5 import load_phase5_secrets  # noqa: E402
from noticelens.phase5_evaluation import run_phase5  # noqa: E402


def main() -> int:
    try:
        secrets = load_phase5_secrets(project_root=PROJECT_ROOT)
        result = run_phase5(PROJECT_ROOT, secrets)
    except Exception as exc:
        # Provider wrappers suppress remote exception text. This outer guard
        # also avoids printing arbitrary request/client objects or credentials.
        print(
            json.dumps(
                {
                    "phase": 5,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "message": "Phase 5 stopped at a fail-closed gate; no credentials were logged",
                }
            ),
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "phase": 5,
                "status": "complete",
                "reports": result["reports"],
                "generation_model": result["model_selection"]["selected_model"],
                "identity": result["identity"],
                "formal_faithfulness": result["faithfulness"]["formal_faithfulness"],
                "citation_support_rate": result["faithfulness"]["citation_support_rate"],
                "refusal": result["refusal"],
                "warm_end_to_end_p95_seconds": result["latency"]["end_to_end"]["p95_seconds"],
                "latency_target_met": result["latency_target_met"],
                "tests": result["tests"],
                "frozen_composite": result["frozen_composite"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
