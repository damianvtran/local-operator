"""The post-sign-in no-spend setup run's journal, and whether it is running now.

Its own module because two READ paths need the answer and neither may load
`tunnels.setup` (the CLI, installers and mobile bundle behind it): the desktop's
`GET /v1/desktop/tunnel` and `GET /v1/auth/status`, both polled. Only the
standard library, `tunnels.config` and the kernel-claim helper are imported.
"""

from __future__ import annotations

import json
from typing import Any

from local_operator.config_writer import nonblocking_file_claim
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

    Both facts are required: the journal says the run that holds the lock is the
    unattended one, and the held lock says it is still alive. Read by both
    desktop GETs (U10) so the background heal is visible instead of looking like
    a stuck "not active" tunnel, and by the setup POST, which answers this state
    rather than a `setup_in_progress` refusal.
    """
    try:
        journal = json.loads((config.directory() / AUTO_JOURNAL).read_text())
    except (OSError, ValueError):
        return False
    return isinstance(journal, dict) and journal.get("state") == "running" and setup_lock_held()


def write_journal(record: dict[str, Any]) -> None:
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
