"""Run NoticeLens Phase 3 from a source checkout."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from noticelens.phase3 import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
