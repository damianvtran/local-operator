"""The post-sign-in no-spend setup run's journal, and whether it is running now.

Its own module because the polled READ paths need the answer and must not load
`tunnels.setup` (the CLI, installers and mobile bundle behind it):
`report.local_payload` (`GET /v1/desktop/tunnel`) and the desktop setup route
(`GET`/`POST /v1/desktop/tunnel/setup`). Only the standard library,
`tunnels.config` and the kernel-claim helper are imported.
"""

from __future__ import annotations

import json
import os
from typing import Any

from local_operator.config_writer import nonblocking_file_claim
from local_operator.procstate import pid_alive
from local_operator.tunnels import config

#: The unattended run's journal (`tunnels/auto-setup.json`): ``running`` while a
#: `--no-spend` apply is in flight, then the step and remedy kind it stopped at.
#: It is NOT a second setup store — the records stay the journal of what exists
#: (module docstring) — only the provenance that tells a desktop poll "the
#: background reconnect owns `setup.lock` right now" apart from "another
#: window's run does", which the lock alone cannot say.
AUTO_JOURNAL = "auto-setup.json"


def setup_lock_held() -> bool:
    """Whether some process holds `setup.lock` now, without creating it.

    A probe, not a claim: an absent file is "nobody" (no write to find out), and
    an existing one is tried non-blockingly and released at once. The kernel lock
    dies with its owner, so a crashed run can never read as live — which is why
    liveness comes from the lock rather than from the journal's pid.
    """
    path = config.directory() / "setup.lock"
    if not path.exists():
        return False
    try:
        with nonblocking_file_claim(path) as acquired:
            return not acquired
    except OSError:
        return False


def auto_reconnecting() -> bool:
    """Whether the post-sign-in no-spend run is reconnecting this tunnel NOW.

    All three facts are required (#3945 round 1, R1): the journal says the
    unattended run wrote ``running`` (it does so only once it HOLDS
    `setup.lock`, `setup.command`), its recorded pid is still alive, and some
    process holds the lock. The pid check is what a crash needs: a SIGKILLed or
    rebooted run leaves ``running`` behind, and without it the next lock holder
    (the desktop's own run) read as "reconnecting" with no button. Read by
    `report.local_payload` (`GET /v1/desktop/tunnel`) and the setup route
    (U10), so the background heal is visible instead of a stale "paused" step,
    and by the setup POST, which answers this state rather than a 409.
    """
    journal = _read_journal()
    return (
        journal is not None
        and journal.get("state") == "running"
        and _pid_alive(journal.get("pid"))
        and setup_lock_held()
    )


def _read_journal() -> dict[str, Any] | None:
    try:
        journal = json.loads((config.directory() / AUTO_JOURNAL).read_text())
    except (OSError, ValueError):
        return None
    return journal if isinstance(journal, dict) else None


def _pid_alive(pid: object) -> bool:
    """Whether ``pid`` names a live process (a journal without one is not live).

    Delegates to :func:`local_operator.procstate.pid_alive` rather than calling
    ``os.kill(pid, 0)`` here: on Windows signal 0 goes to ``TerminateProcess``,
    so the "probe" would KILL the setup run it is asking about (the xplat
    battery's ``static.posix_attributes`` fatal on #3945). The shared probe asks
    ``OpenProcess`` on win32 and keeps the POSIX ESRCH/EPERM answers unchanged;
    its fail-closed doubt (alive) only means a journal is treated as live, and
    ``is_running`` still requires the setup lock to be held as well.
    """
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    return pid_alive(pid)


def write_journal(record: dict[str, Any], *, unless_live_other: bool = False) -> None:
    """Write the journal (best-effort).

    ``unless_live_other``: skip the write while ANOTHER live pid's ``running``
    record stands — a run that never claimed the lock (it stopped at a quote,
    say) must not erase the claimant's state from the desktop's view.
    """
    if unless_live_other:
        current = _read_journal()
        if (
            current is not None
            and current.get("state") == "running"
            and current.get("pid") != os.getpid()
            and _pid_alive(current.get("pid"))
        ):
            return
    # Best-effort: the journal is a visibility aid, and failing to write it must
    # not stop the reconnect it describes.
    try:
        config.private_write(config.directory() / AUTO_JOURNAL, json.dumps(record) + "\n")
    except OSError:
        pass


#: Which Radient credentials the TUI's first-sign-in phone access offer was
#: already shown for (`tunnels/login-offer.json`, ``{"shown": [ids]}``).
LOGIN_OFFER = "login-offer.json"


def _offered() -> set[int]:
    try:
        record = json.loads((config.directory() / LOGIN_OFFER).read_text())
    except (OSError, ValueError):
        return set()
    shown = record.get("shown") if isinstance(record, dict) else None
    if not isinstance(shown, list):
        return set()
    return {i for i in shown if isinstance(i, int) and not isinstance(i, bool)}


def login_offer_due() -> int | None:
    """The credential a first-time phone access offer should name, or ``None``.

    The TUI shows ONE line after `/login radient` on a machine that has NO phone
    access, pointing at `/mobile enable` (the existing in-app setup verb; nothing
    is created automatically — the post-sign-in run covers only machines that
    already wanted it). Shown once per credential: recording it at show time is
    also its dismissal, so an offer the user saw, closed or ignored never returns
    for that credential. A newer sign-in row (a different credential) may see it
    once. Read-only store access; never raises.
    """
    from contextlib import closing

    from local_operator.providers.auth_store import AuthStore, AuthStoreError
    from local_operator.tunnels.install import phone_access_wanted

    try:
        if phone_access_wanted():
            return None
        with closing(AuthStore(read_only=True)) as store:
            rows = [r for r in store.list_credentials("radient") if r.credential_type == "oauth"]
    except (AuthStoreError, OSError, ValueError):
        return None
    if not rows:
        return None
    newest = max(rows, key=lambda r: (r.updated_at, r.id))
    return None if newest.id in _offered() else newest.id


def record_login_offer(credential_id: int) -> None:
    """Remember the offer was shown for ``credential_id`` (best-effort)."""
    shown = sorted(_offered() | {int(credential_id)})
    try:
        config.private_write(config.directory() / LOGIN_OFFER, json.dumps({"shown": shown}) + "\n")
    except OSError:
        pass
