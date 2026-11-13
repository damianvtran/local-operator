"""C3 of remote files parity: family discovery lives outside the desktop module.

docs/design/remote-files-parity.md §2 "Owner execution": the relay answers
``net_session_files`` with the SAME family definition the desktop route uses, and
importing ``server.utils.desktop_sessions`` into the relay measured +27 MiB RSS and
+0.8 s and pulled in ``fastapi``. So the code moved; these cells pin that it is ONE
implementation (re-exported, not copied) and that the relay's import path stays light.
"""

from __future__ import annotations

import subprocess
import sys

from local_operator.server.utils import desktop_sessions
from local_operator.session_files import family


def test_the_desktop_module_re_exports_the_moved_names_not_copies() -> None:
    for name in (
        "contained_descendants",
        "_contained_child_dir",
        "_persisted_children",
        "_prove_child",
        "SubagentChildUnavailable",
        "FAMILY_MAX_DEPTH",
        "FAMILY_MAX_CHILDREN",
    ):
        assert getattr(desktop_sessions, name) is getattr(family, name), name


def test_the_relay_import_path_leaves_fastapi_and_the_desktop_module_out() -> None:
    probe = (
        "import sys\n"
        "import local_operator.network.files\n"
        "import local_operator.session_files.family\n"
        "heavy = [m for m in ('fastapi', 'local_operator.server.utils.desktop_sessions')"
        " if m in sys.modules]\n"
        "print(','.join(heavy))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=120, check=True
    )
    assert result.stdout.strip() == "", f"the relay import path loaded: {result.stdout}"
