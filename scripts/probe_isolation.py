"""Import this FIRST in any ad-hoc script that touches local-operator.

    import scripts.probe_isolation  # noqa: F401  -- must be the first import

Importing it re-homes the process: ``HOME`` and ``LOCAL_OPERATOR_CONFIG_DIR``
point at a fresh ``mkdtemp`` before anything under ``local_operator`` can
resolve the operator's real ``~/.local-operator``. Import-time rather than a
function to call, because the function was already there
(``visual_capture.isolate_capture``) and the incident happened anyway: an
ad-hoc probe written from a worktree imported the app first and ran a
config migration against the operator's LIVE config, which un-guarded the
installed runtime's reaper and cost a session (PR #645, round 5). A helper
that must be remembered is a helper that will be forgotten once; one that
does its work on import is a single line at the top of the file and fails
loudly if any of the package has already been imported.

Nothing here is clever. It is the rule from the incident record, made
mechanical:

* ``HOME`` and ``LOCAL_OPERATOR_CONFIG_DIR`` → a fresh temp dir, always,
  even if the caller set them (a stale export pointing at the real dir is
  exactly the mistake).
* ``NO_COLOR`` unset, ``TERM=xterm-256color``, shimmer/notifications/title
  off — the same sandbox ``visual_capture.isolate_capture`` builds, so the
  two cannot drift.
* If ``local_operator`` (or any submodule) is ALREADY in ``sys.modules``
  the import raises: isolation after the fact protects nothing.

The sandbox path is exported as :data:`SANDBOX` for scripts that want to
seed it.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_already = sorted(name for name in sys.modules if name.split(".")[0] == "local_operator")
if _already:
    raise RuntimeError(
        "scripts.probe_isolation must be imported BEFORE any local_operator module; "
        f"already imported: {', '.join(_already[:5])}" + (" ..." if len(_already) > 5 else "")
    )

_TEMP = tempfile.TemporaryDirectory(prefix="lop-probe-")
SANDBOX = Path(_TEMP.name)
os.environ["HOME"] = str(SANDBOX)
os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = str(SANDBOX / "config")
os.environ.pop("NO_COLOR", None)
os.environ["TERM"] = "xterm-256color"
os.environ["LOCAL_OPERATOR_NO_SHIMMER"] = "1"
os.environ["LOCAL_OPERATOR_NO_NOTIFICATIONS"] = "1"
os.environ["LOCAL_OPERATOR_NO_TERMINAL_TITLE"] = "1"
(SANDBOX / "config").mkdir(parents=True, exist_ok=True)
