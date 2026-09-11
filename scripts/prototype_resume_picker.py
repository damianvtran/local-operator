"""PROTOTYPE launcher — throwaway. See ~/workspace/PROPOSAL-resume-picker.md."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 1. Capture the REAL store BEFORE isolation re-homes HOME.
REAL_STORE = Path(
    os.environ.get("LOCAL_OPERATOR_CONFIG_DIR") or (Path.home() / ".local-operator")
)

# 2. Isolation MUST precede every local_operator import. This is why the entry
#    point is a script and not `python -m local_operator...`: -m imports the
#    parent packages before the module body runs, so isolation can never be first.
import scripts.probe_isolation  # noqa: F401,E402

# 3. Only now may local_operator be imported.
from local_operator.tui.widgets.prototype_resume_picker import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(REAL_STORE, sys.argv[1:]))
