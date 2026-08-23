"""Run the frozen Phase 5.1 generation comparison and faithfulness closure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from noticelens.phase5 import load_phase5_secrets  # noqa: E402
from noticelens.phase5_1 import recover_phase51_evaluator, run_phase51  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recover-evaluator",
        action="store_true",
        help="replay only the pinned unjudged blinded-evaluator claims; no retrieval or generation",
    )
    args = parser.parse_args()
    try:
        secrets = load_phase5_secrets(project_root=PROJECT_ROOT)
        result = (
            recover_phase51_evaluator(PROJECT_ROOT, secrets)
            if args.recover_evaluator
            else run_phase51(PROJECT_ROOT, secrets)
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "phase": "5.1",
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "message": "Phase 5.1 stopped at a fail-closed gate; no credentials were logged",
                }
            ),
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "phase": "5.1",
                "status": result["status"],
                "mode": result.get("mode", "full_comparison"),
                "models_tested": result["models_tested"],
                "selected_model": result["selected_model"],
                "effective_configured_model": result["effective_configured_model"],
                "config_changed": result["config_changed"],
                "decision": result["decision"],
                "tests": result["tests"],
                "frozen_retrieval_unchanged": result["frozen_retrieval_unchanged"],
                "reports": result["reports"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
