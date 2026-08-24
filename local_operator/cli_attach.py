"""The CLI's owned-resume branch: attach to a live owner instead of racing it.

``lop --resume <id>`` where another live process owns ``<id>`` used to fall
through to the session factory, claim the directory, and silently become a
second writer on a transcript the owner was still appending to — the cold-boot
half of the corruption hole the TUI's in-app guard already refuses. This module
is the CLI edge of the attach design: discover the owner's control-socket
record, and either run the standalone attach app (a follower view; the factory
never runs) or print the TUI's refusal copy and exit 1 when the owner is not
attachable (old binary, registrant failure, rebind race).

Kept as its own module (not inline in ``cli.py``) because everything here is
import-heavy (Textual, the mobile package) and the branch is rare: the CLI's
startup import-cost guard stays green because none of this loads unless an
owned resume actually happens. ``local_operator.resume`` stays stdlib-only —
this module imports it, never the reverse.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _refuse(session_id: str, owner: int) -> int:
    """Today's refusal line, verbatim with the TUI's, plus exit 1."""
    print(
        f"\033[31msession {session_id} is already open in another process "
        f"(pid {owner}) — watch and steer it there, or from the phone "
        f"session list\033[0m",
        file=sys.stderr,
    )
    return 1


def run_owned_resume_attach(config_dir: Path, session_id: str, owner: int) -> int:
    """Attach to the live owner of ``session_id``, or refuse exactly as before.

    Exit codes: 0 after a clean detach; 75 when the user chose 'resume here'
    after the owner died mid-attach (the caller relaunches without ``--resume``
    — by then the claim marker names a dead pid, so the relaunch is the
    legitimate first writer); 1 when the owner is not attachable.
    """
    from local_operator.mobile.attach_client import find_owner_record

    record, found_owner = find_owner_record(config_dir, session_id)
    if record is None or found_owner != owner:
        return _refuse(session_id, owner)

    from local_operator.tui.attach_screen import run_attach_app

    code = run_attach_app(record, session_id)
    if code == 75:
        # 'resume here': the owner is gone (the attach screen only offers this
        # after owner death), so re-enter the CLI with the resume flag dropped.
        # A fresh process is the honest shape: this one already decided NOT to
        # build a session when it took the attach branch. ``-m local_operator.cli``
        # because the package has no ``__main__``; the module's own tail runs
        # ``main()``.
        import subprocess

        return subprocess.call([sys.executable, "-m", "local_operator.cli"])
    return code
