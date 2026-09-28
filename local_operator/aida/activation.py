"""Where Aida may auto-activate herself at boot (R17/R21, slice B).

WHAT THIS ANSWERS. Auto-activation is the one thing nobody asked for: the boot
hooks (the TUI, the ``lop serve`` lifespan) CREATE her session, pin it, arm
her cadence and install the wake supervisor without a user keystroke. R17 says
harness-only / cloud automation installs (agent-runtime-svc; no project
management, no TUI, no UI) must pay none of that; R21 fixes the DEFAULT — an
interactive install gets her with no configuration, a cloud/automation install
is not auto-activated.

THE SIGNAL, and why these two halves. The design named this predicate but left
its signal open (design §2.9 flags it; slice B proposes the smallest rule and
says so on the PR). A human surface present in THIS process is the honest
question, and there are exactly two ways this process can have one:

1. **A terminal** — ``stdin`` and ``stdout`` are both ttys, the same test
   ``local_operator.network.relay._has_terminal`` uses for "this process can
   ask a human a question". The TUI path always satisfies it (a full-screen
   terminal UI is on screen), and so does a ``lop serve`` a person started in
   their own shell.
2. **The desktop app** — a daemon the Electron main process spawned carries
   ``LOCAL_OPERATOR_DESKTOP_TOKEN`` (``server/desktop.py``). That env is read
   ONLY through :func:`desktop.desktop_posture`, the module's single-reader
   invariant (`tests/unit/server/test_desktop_claim.py` enforces it), so this
   module asks the ONE predicate rather than growing a second opinion.

Neither half holds for the harness-only shape — a supervisor-driving install
runs ``lop exec`` under pipes, and the exec path never calls the boot ensure at
all (design §2.3) — so cloud/automation installs are not auto-activated by
default. They are not LOCKED OUT either: an explicit surface (``/aida``,
``POST /v1/desktop/aida {op:"open"}``) still ensures her, and a deployer who
wants no trace of her sets ``aida.enabled=false`` or
``LOCAL_OPERATOR_NO_AIDA=1`` (R17/R18's documented switches).

THE CONSERVATIVE GAP, STATED. A human install that runs ``lop serve`` under
launchd/nohup (no tty, no token, later claimed by the desktop app) is
classified automation at boot: she is created on the first explicit open
instead of at daemon start. Direction of the failure: a session that starts a
click later, never a wake cost on a harness-only install. Chosen deliberately —
the operator's own framing was "agent-runtime-svc installs would set it off (or
we detect their runtime profile)", and profile detection needs deployer
cooperation that does not exist yet.

THE PERMISSIVE RESIDUAL, stated for the same reason: an orchestrator that
allocates a PTY for ``lop serve`` satisfies both tty checks and is classified
interactive. That direction cannot be detected from inside the process — a pty
is what the check IS — so the signal is best-effort in both directions and the
explicit off switches above are the guarantee (review round 1, F5).

IMPORT-LIGHT by the package's contract: stdlib only at module scope; the
desktop read is a lazy import behind a try/except (the TUI and the daemon load
this on boot paths, and a failure to read the posture must answer "no surface",
not raise).
"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)


def _has_terminal() -> bool:
    """Whether this process can show a terminal UI / talk to a human directly.

    Mirrors ``network.relay._has_terminal`` (stdin AND stdout are ttys) rather
    than inventing a second dialect: "cannot tell" is a ``False``, because the
    wrong ``True`` here starts a conversation nobody asked for.
    """
    try:
        return bool(
            sys.stdin is not None
            and sys.stdin.isatty()
            and sys.stdout is not None
            and sys.stdout.isatty()
        )
    except (ValueError, OSError):  # closed or replaced streams
        return False


def _desktop_plane_open() -> bool:
    """Whether this daemon was spawned by (or claimed by) the desktop app.

    Reads through :func:`local_operator.server.desktop.desktop_posture`, the
    module's single reader for the desktop environment, so this predicate
    cannot drift from the gates that govern the plane itself. The import is
    lazy and the failure answers ``False``: a boot hook must degrade to "no
    surface", never raise.
    """
    try:
        from local_operator.server.desktop import desktop_posture

        return bool(desktop_posture().token)
    except Exception:  # noqa: BLE001 — "cannot tell" means "no human surface"
        logger.debug("aida: could not read the desktop posture", exc_info=True)
        return False


def human_surface_present() -> bool:
    """Whether boot may auto-activate her in THIS process (R17/R21).

    True = an interactive install: a terminal is attached, or the desktop app
    governs this daemon. False = a cloud/automation install, where she waits
    for an explicit open instead. The disable switch is deliberately NOT part
    of this predicate: it is the separate, first gate
    (``bootstrap.config_enabled``), so "may a surface mention her" and "may
    this boot create her" cannot disagree about a switch they share.
    """
    return _has_terminal() or _desktop_plane_open()
