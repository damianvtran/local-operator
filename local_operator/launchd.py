"""Shared LaunchAgent plist helpers: addressability, staleness, and the repair.

WHY THIS MODULE EXISTS
----------------------
Three of the four installers (``mobile``, ``tunnels``, ``browser_bridge``) wrote
their plist at install time and never looked at it again, so a plist written by
an older build kept running that older build's interpreter for as long as the
job existed. ``wakes`` is the one that already repaired itself, "idempotent by
CONTENT, not by existence"; this module is that idea lifted out of it so the
other three can share the parts that are the same — and, more importantly, so
the parts that are dangerous are written once:

- **The addressability guard.** ``launchctl`` has no notion of a sandbox: it
  always addresses the calling user's live session, whatever ``Path.home()``
  has been redirected to. A test that patches ``home`` to a tmpdir — the
  ordinary way to test an installer — would otherwise rewrite and restart the
  operator's REAL daemon. ``wakes/install.py`` documents the incident that put
  this guard in (a live unit pointed at a pytest tmpdir). The test is
  IDENTITY, not location: the plist path must be the one the passwd home
  produces, because a containment test fails open whenever a redirected home
  lands inside the real one (``TMPDIR`` under ``$HOME`` is not exotic) — see
  :func:`is_own_plist`.

- **The config-dir guard.** A plist that records a config dir (``wakes``,
  ``tunnels``) must not be rewritten to point at a store that is outside the
  real home: that is how a sandbox run plants a supervised unit pointed at a
  directory that vanishes when the sandbox ends.

- **Bootout + bootstrap, never ``kickstart -k``, after a rewrite.** MEASURED on
  macOS with a scratch label, ``Program`` = the branded hardlink and
  ``ProgramArguments[0]`` = a label: after rewriting the plist on disk,
  ``launchctl kickstart -k`` returned 0, started a NEW pid, and kept running
  the PREVIOUS argv — its marker file was never written, the old one still was.
  launchd restarts from the in-memory job definition, so a re-write followed by
  a kickstart silently repairs nothing. ``bootout`` + ``bootstrap`` on the same
  scratch label did pick up the new argv. ``wakes``'s kickstart repair is
  correct for the different case it handles (the plist is right and the job is
  stopped) and is left alone.

WHY THE REPAIR MUST RUN IN A NEW PROCESS
----------------------------------------
``lop update`` upgrading the wheel does not change the code its own process has
already imported. A repair run in-process would render the OLD plist shape and
therefore rewrite nothing — the pre-fix behaviour with more code. Everything in
this module is executed by a child started from the NEW wheel; see
``update.refresh_daemons_after_upgrade``.
"""

from __future__ import annotations

import logging
import os
import plistlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

#: Where a user-level LaunchAgent lives. Both halves are needed by
#: :func:`is_own_plist`, which builds the expected path from the passwd home.
_PLIST_DIR = ("Library", "LaunchAgents")

PlistRefreshKind = Literal[
    "unsupported", "not-installed", "not-addressable", "current", "repaired", "failed"
]


@dataclass(frozen=True)
class PlistRefresh:
    """What a per-daemon repair did. Never an exception — see ``update.py``.

    ``name`` is the daemon's human name ("mobile", "browser bridge", "tunnel",
    "wakes supervisor") because this is what the upgrade summary prints; the
    caller owns the vocabulary, this type only carries it.

    ``unsupported`` is the honest answer on a platform with no LaunchAgent to
    repair (Linux, where the unit is re-read on every start) and on a machine
    with no ``launchctl``; it is deliberately silent, because a platform that
    cannot have this problem must not read as a failure on every upgrade.
    """

    name: str
    kind: PlistRefreshKind
    detail: str = ""

    def summary(self) -> str:
        """The upgrade-summary line, or ``""`` when there is nothing to say.

        Only a repair speaks. A daemon that is absent, unaddressable from here
        or already current is the normal state of a machine and printing a line
        for it would turn every upgrade into a list of non-events — the same
        reason the mobile refresh stays silent when it skips.
        """
        if self.kind == "repaired":
            return f"{self.name} daemon: refreshed a stale LaunchAgent and restarted it"
        return ""

    def warning(self) -> str:
        """The line for the upgrade summary's stderr, or ``""``.

        A failure DETAIL is a whole sentence rather than a token: the reload
        failures name the command that restores a stopped daemon, and that
        sentence is the point of the line, so it is printed as it stands.
        """
        if self.kind == "failed":
            return f"warning: {self.name} daemon was not refreshed: {self.detail}"
        return ""


def reload_failure(name: str, path: Path, recovery: str, error: str) -> PlistRefresh:
    """Outcome for a repair that rewrote the plist but could not reload the job.

    THE ONE FAILURE IN THE LADDER THAT LEAVES THE MACHINE CHANGED **AND** THE
    SERVICE DOWN: ``bootout`` succeeded, so the daemon the operator had is no
    longer running, and a bare ``launchctl`` error reads like a stale-file
    problem rather than a stopped service. The ``recovery`` command named here
    is that daemon's own installer (``lop mobile install``, ``lop browser
    install``, ``lop tunnel install``, ``lop wake install``), which rewrites
    the same plist and loads it again. One helper so all four installers cannot
    drift in how they say this.
    """
    return PlistRefresh(
        name=name,
        kind="failed",
        detail=(
            f"rewrote {path} but launchctl could not load it: {error} "
            f"— the daemon is now STOPPED; run `{recovery}` to reinstall it"
        ),
    )


def recorded_install_prefix(data: object) -> Path | None:
    """The install prefix the plist's interpreter lives in, or ``None``.

    Reads the interpreter path from either plist shape: the branded ``Program``
    (``<prefix>/bin/Local Operator``) or, on a plist written before that key
    existed, ``ProgramArguments[0]`` when it is a path rather than a label.
    Used to answer "is this the SAME installation?" — see
    :func:`update.daemons_refresh_command`, where the repair may change how a
    daemon is NAMED but must never change WHICH INSTALL it runs.

    ``None`` means "cannot tell", which callers must treat as no objection: a
    plist this code cannot read is not evidence of a different install.
    """
    if not isinstance(data, dict):
        return None
    program = data.get("Program")
    candidates: list[str] = []
    if isinstance(program, str):
        candidates.append(program)
    argv = data.get("ProgramArguments")
    if isinstance(argv, list) and argv and isinstance(argv[0], str):
        candidates.append(argv[0])
    for candidate in candidates:
        if not candidate.startswith("/"):
            # A label, not a path: the branded shape carries the image in
            # ``Program``, and anything else here is not a candidate prefix.
            continue
        prefix = Path(candidate).parent.parent
        return prefix
    return None


def real_home() -> Path | None:
    """The uid's passwd home, or ``None`` when it cannot be read.

    NOT ``Path.home()``, and that difference is the whole guard: ``Path.home()``
    reads ``$HOME``, so an isolated run would compare its own redirected home
    against itself and conclude it is the real one.
    """
    import pwd

    try:
        return Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()
    except (KeyError, OSError):
        return None


def is_own_plist(path: Path, label: str) -> bool:
    """Whether ``launchctl``'s answer about ``path`` would be about THIS run.

    True only when ``path`` is the plist path the real passwd home produces for
    ``label``. A redirected ``HOME`` therefore refuses: the file half of an
    installer stays fully testable, while the half that reaches outside the
    process declines to run. Unreadable passwd entry degrades to False, which
    is the safe direction — the cost of a wrongly-skipped repair is a stale
    plist; the cost of a wrongly-executed one is the operator's live daemon.
    """
    home = real_home()
    if home is None:
        return False
    try:
        expected = (home.joinpath(*_PLIST_DIR) / f"{label}.plist").resolve()
        return Path(path).resolve() == expected
    except (OSError, ValueError):
        return False


def config_lives_in_real_home(config_dir: Path) -> bool:
    """Whether a unit supervising ``config_dir`` would outlive this process.

    CONTAINMENT here, not the identity test :func:`is_own_plist` uses, and the
    difference is deliberate: the config dir is an ordinary path the user may
    legitimately place anywhere under their home, so only dirs OUTSIDE it are
    the sandbox shape (``/tmp``, a throwaway home). A unit pointed at one of
    those is a live launchd job watching a store that is deleted when the
    sandbox ends.
    """
    home = real_home()
    if home is None:
        return False
    try:
        return Path(config_dir).resolve().is_relative_to(home)
    except (OSError, ValueError):
        return False


def load(path: Path) -> dict[str, object] | None:
    """The parsed plist, or ``None`` for absent/corrupt/unreadable.

    One answer for all three on purpose: every caller wants "can I read what is
    there?", and an unreadable plist is a stale one.
    """
    try:
        parsed = plistlib.loads(Path(path).read_bytes())
    except Exception:  # noqa: BLE001 — a corrupt plist is "no evidence", not a crash
        return None
    return parsed if isinstance(parsed, dict) else None


def arg_value(plist: dict[str, object] | None, flag: str) -> str | None:
    """The value following ``flag`` in ``ProgramArguments``, if present.

    Reads argv for its ARGUMENTS, not for an interpreter: the plist's element 0
    is a label in the current shape and an interpreter path in the pre-branding
    one, which is exactly why nothing may parse it positionally (see
    ``procname.launchd_job``). ``--flag=value`` is accepted as well as
    ``--flag value``.
    """
    if not plist:
        return None
    argv = plist.get("ProgramArguments")
    if not isinstance(argv, list):
        return None
    for index, item in enumerate(argv):
        if not isinstance(item, str):
            continue
        if item == flag:
            following = argv[index + 1] if index + 1 < len(argv) else None
            return following if isinstance(following, str) else None
        if item.startswith(f"{flag}="):
            return item.partition("=")[2] or None
    return None


def int_arg(plist: dict[str, object] | None, flag: str, default: int) -> int:
    """``arg_value`` as an int, falling back to ``default``.

    A repair must never CHANGE a setting: a daemon installed on a non-default
    port is repaired on THAT port, which is only knowable from the plist being
    replaced. Anything unparseable keeps the default rather than inventing one.
    """
    raw = arg_value(plist, flag)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def config_dir_from_plist(plist: dict[str, object] | None) -> Path | None:
    """The config dir a plist's own ``EnvironmentVariables`` record, if any.

    Read from the file being repaired rather than from this process's ambient
    environment: the repair's job is to bring a unit up to date IN PLACE, not to
    migrate it to whatever store happens to be current here.
    """
    from local_operator.paths import CONFIG_DIR_ENV

    if not plist:
        return None
    environment = plist.get("EnvironmentVariables")
    if not isinstance(environment, dict):
        return None
    value = environment.get(CONFIG_DIR_ENV)
    return Path(value) if isinstance(value, str) and value else None


def rewrite_if_stale(*, name: str, path: Path, rendered: dict[str, object]) -> PlistRefresh:
    """Rewrite ``path`` when it does not already say what ``rendered`` says.

    The caller has ALREADY established that it may act on this path
    (:func:`is_own_plist`) and that the store it names is durable
    (:func:`config_lives_in_real_home`); this function only compares content and
    writes. It never restarts anything — the per-daemon module owns that,
    because the restart command differs per unit.

    Never raises: writing is done in the caller's try/except too, but a
    permission error on ``~/Library/LaunchAgents`` must not escape from here
    either, since the only caller that matters is an upgrade that has already
    succeeded.
    """
    if not path.exists():
        return PlistRefresh(name=name, kind="not-installed")
    current = load(path)
    if current == rendered:
        return PlistRefresh(name=name, kind="current")
    try:
        # ``write_bytes`` over an existing file keeps its mode, which is the
        # mode the installer that wrote it chose (0600 for the tunnel plist).
        # Nothing here chmods: a repair must not change permissions it was not
        # asked about.
        path.write_bytes(plistlib.dumps(rendered))
    except Exception as exc:  # noqa: BLE001 — a stale plist is not a failure
        logger.debug("could not rewrite %s", path, exc_info=True)
        return PlistRefresh(name=name, kind="failed", detail=f"could not rewrite {path}: {exc}")
    return PlistRefresh(name=name, kind="repaired")
