"""tmux / zellij / screen / wezterm: publish a discoverable per-pane marker.

WHY THESE ARE ONE MODULE AND NOT FOUR
-------------------------------------
None of these multiplexers has a resume-binding API — there is nothing to hand
a command to and nothing that will re-run it after a crash. What they DO have
in common is a stable per-pane identity, so all four publish the same fact in
the same shape and differ only in where it is written. Splitting that into
four near-identical modules would hide how small the actual difference is.

THE CONTRACT (this is the public part — restore is scripted against it)
----------------------------------------------------------------------
Two facts are published per pane, under these exact names:

    @lop_session          the 12-hex session id, i.e. what --resume takes
    @lop_resume_command   the full restore-and-idle argv, shell-quoted

tmux and wezterm store them as native pane options, so they are readable with
the multiplexer's own tooling and vanish with the pane:

    tmux show-options -pv @lop_session
    tmux show-options -pv @lop_resume_command

zellij and screen have no per-pane key/value store, so the same two facts go
in a JSON state file, one per pane, under::

    ~/.local-operator/multiplexer/<backend>-<pane-id>.json
    {"session_id": "...", "command": [...], "cwd": "...", "updated_at": 1.0}

``<pane-id>`` is whatever identifies a pane on that multiplexer, which is not
always one variable: zellij's is ``<session-name>-<pane-id>`` because its pane
ids are only unique within a session. screen's is ``<sty>-<window>`` when
``WINDOW`` is exported and falls back to ``<sty>`` when it is not — the one
case where a marker is per SESSION rather than per window, and a second ``lop``
in that session overwrites it.

A restore script's job is the same in both cases: for each pane, read
``@lop_session`` (or the file), and if it names a session directory that still
exists, launch ``@lop_resume_command`` in that pane. Nothing here runs that
script — these multiplexers cannot, which is precisely the difference from
cmux — so the marker is written for a human, a shell function, or the user's
own session-restore tool to consume.

The state files are deliberately NOT cleaned up by age: a pane id is reused,
so a stale file is overwritten by the next session in that pane, and a file
whose pane is gone is harmless (a restore script keys off panes that exist,
not off files it finds). A clean exit removes its own file; a crash leaves it,
which is the entire point.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import subprocess
import time
from pathlib import Path

from local_operator.multiplexer.types import EnvMap, SessionBinding

logger = logging.getLogger(__name__)

#: The two option names, spelled once. tmux requires the ``@`` prefix for
#: user options; the same names are reused for the file-backed backends so the
#: contract reads identically whichever multiplexer a user is on.
SESSION_OPTION = "@lop_session"
COMMAND_OPTION = "@lop_resume_command"

#: Same short leash as the cmux backend: these are fire-and-forget subprocesses
#: on a worker thread, and a wedged multiplexer socket must not accumulate one
#: stuck process per session.
CALL_TIMEOUT_S = 5.0

#: Pane identifiers are interpolated into a FILENAME, so they are restricted to
#: characters that cannot escape the directory. A pane id that fails this is
#: treated as "no identity" (nothing is published) rather than sanitised into
#: a different pane's file — writing the right data about the wrong pane is
#: worse than writing nothing.
_SAFE_PANE_ID = re.compile(r"\A[A-Za-z0-9._-]{1,128}\Z")


def marker_dir() -> Path:
    """Where the file-backed markers live.

    Under the config dir rather than a temp directory: these must survive the
    reboot that follows a crash, and ``/tmp`` is exactly what does not.
    """
    from local_operator.paths import config_dir

    return config_dir() / "multiplexer"


def _run(argv: list[str]) -> bool:
    """Run a multiplexer command. True on success, never raises.

    Same best-effort contract as everything else here: the multiplexer may
    have exited between detection and this call, and a user quitting tmux is
    not an error this app reports.
    """
    try:
        completed = subprocess.run(  # noqa: S603 — fixed argv, no shell
            argv,
            capture_output=True,
            text=True,
            timeout=CALL_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        logger.debug("multiplexer command failed to spawn: %s", argv[0], exc_info=True)
        return False
    if completed.returncode != 0:
        logger.debug("%s exited %s: %s", argv[0], completed.returncode, completed.stderr[:200])
        return False
    return True


def _which(binary: str) -> str | None:
    """Resolve a multiplexer binary, or None.

    The binary is the gate for the same reason it is in the cmux backend: the
    ``TMUX``/``ZELLIJ`` variables are inherited by descendants that may have
    crossed into a container where the binary does not exist.
    """
    import shutil

    return shutil.which(binary)


class _OptionBackend:
    """Shared body for multiplexers with a native per-pane option store."""

    #: Subclass fills these in. ``binary`` is what must exist on PATH;
    #: ``pane_env`` is the variable naming this pane.
    name = ""
    binary = ""
    pane_env = ""

    def detect(self, env: EnvMap) -> bool:
        if not (env.get(self.pane_env) or "").strip():
            return False
        return _which(self.binary) is not None

    def _set_option(self, pane: str, option: str, value: str) -> bool:
        raise NotImplementedError

    def _unset_option(self, pane: str, option: str) -> bool:
        raise NotImplementedError

    def publish(self, binding: SessionBinding, env: EnvMap) -> bool:
        pane = (env.get(self.pane_env) or "").strip()
        if not pane or _which(self.binary) is None:
            return False
        # shlex.join, not " ".join: the command is stored as a STRING that a
        # restore script will hand to a shell, and a cwd or launcher path
        # containing a space would otherwise restore as two arguments.
        command = shlex.join(binding.argv)
        wrote_session = self._set_option(pane, SESSION_OPTION, binding.session_id)
        wrote_command = self._set_option(pane, COMMAND_OPTION, command)
        return wrote_session and wrote_command

    def retire(self, binding: SessionBinding, env: EnvMap) -> bool:
        pane = (env.get(self.pane_env) or "").strip()
        if not pane or _which(self.binary) is None:
            return False
        # Both are unset even if the first fails: a pane left advertising a
        # session id with no command (or vice versa) is a half-marker a
        # restore script has to special-case.
        cleared_session = self._unset_option(pane, SESSION_OPTION)
        cleared_command = self._unset_option(pane, COMMAND_OPTION)
        return cleared_session and cleared_command


class TmuxBackend(_OptionBackend):
    """tmux, via ``set-option -p`` (pane-scoped user options)."""

    name = "tmux"
    binary = "tmux"
    #: ``TMUX_PANE`` and not ``TMUX``: ``TMUX`` proves a server connection but
    #: names no pane, and the option must be set on the pane holding THIS
    #: session or a second session in the same window would overwrite it.
    pane_env = "TMUX_PANE"

    def detect(self, env: EnvMap) -> bool:
        # Both markers: TMUX is what proves a live server (TMUX_PANE alone
        # survives into a process that left tmux behind).
        if not (env.get("TMUX") or "").strip():
            return False
        return super().detect(env)

    def _set_option(self, pane: str, option: str, value: str) -> bool:
        return _run([self.binary, "set-option", "-p", "-t", pane, option, value])

    def _unset_option(self, pane: str, option: str) -> bool:
        # ``-u`` unsets. Without it the option would be set to the empty
        # string, which a restore script reads as "a session with a blank id"
        # rather than as absence.
        return _run([self.binary, "set-option", "-p", "-t", pane, "-u", option])


class WezTermBackend(_OptionBackend):
    """wezterm, via ``cli set-user-var`` (per-pane user variables)."""

    name = "wezterm"
    binary = "wezterm"
    pane_env = "WEZTERM_PANE"

    def _set_option(self, pane: str, option: str, value: str) -> bool:
        return _run([self.binary, "cli", "set-user-var", "--pane-id", pane, option, value])

    def _unset_option(self, pane: str, option: str) -> bool:
        # wezterm has no unset; an empty value is how a user var is cleared.
        # Documented here because it is the one place the marker contract is
        # weaker than tmux's: readers must treat "" as absent.
        return _run([self.binary, "cli", "set-user-var", "--pane-id", pane, option, ""])


class _FileBackend:
    """Shared body for multiplexers with no per-pane key/value store.

    zellij and screen can both identify the pane/window a process is in, and
    neither offers anywhere to hang a value off it, so the marker goes in a
    file named for that identity.
    """

    name = ""
    #: Environment variables that identify this pane, in preference order.
    #: Each entry is a tuple of variables that are joined with ``-`` to form
    #: one identity, so a multiplexer whose pane is only identified by
    #: (session, pane) can name both. ALL variables in a tuple must be present
    #: and safe for that candidate to be used; the next candidate is then
    #: tried, which is what lets a backend degrade to a coarser identity on a
    #: host that does not export the finer one.
    pane_envs: tuple[tuple[str, ...], ...] = ()

    def _pane_id(self, env: EnvMap) -> str | None:
        for variables in self.pane_envs:
            parts = [(env.get(variable) or "").strip() for variable in variables]
            # Every component is validated separately rather than validating
            # the joined string: the join character is legal inside a
            # component, so checking only the result would let a component
            # containing a separator forge a different pane's filename.
            if all(part and _SAFE_PANE_ID.match(part) for part in parts):
                return "-".join(parts)
        return None

    def detect(self, env: EnvMap) -> bool:
        return self._pane_id(env) is not None

    def _path(self, pane: str) -> Path:
        return marker_dir() / f"{self.name}-{pane}.json"

    def publish(self, binding: SessionBinding, env: EnvMap) -> bool:
        pane = self._pane_id(env)
        if pane is None:
            return False
        payload = {
            "backend": self.name,
            "pane": pane,
            "session_id": binding.session_id,
            "command": list(binding.argv),
            "cwd": binding.cwd,
            "updated_at": time.time(),
        }
        try:
            directory = marker_dir()
            directory.mkdir(parents=True, exist_ok=True)
            # Written via a temp file and renamed: a restore script may read
            # this at any moment (including while a second session is starting
            # in a neighbouring pane), and a half-written JSON file would read
            # as a corrupt marker rather than as an older valid one.
            temporary = self._path(pane).with_suffix(".json.tmp")
            temporary.write_text(json.dumps(payload), encoding="utf-8")
            os.replace(temporary, self._path(pane))
        except OSError:
            logger.debug("%s marker write failed", self.name, exc_info=True)
            return False
        return True

    def retire(self, binding: SessionBinding, env: EnvMap) -> bool:
        pane = self._pane_id(env)
        if pane is None:
            return False
        try:
            self._path(pane).unlink(missing_ok=True)
        except OSError:
            logger.debug("%s marker removal failed", self.name, exc_info=True)
            return False
        return True


class ZellijBackend(_FileBackend):
    """zellij, keyed by session name AND pane id.

    Both halves are required for correctness, not for readability.
    ``ZELLIJ_SESSION_NAME`` names a SESSION, and a session holds many panes, so
    keying on it alone gave every ``lop`` pane in one zellij session the same
    marker file: the last publisher won, and one pane's clean exit deleted its
    siblings' markers, so a restore reopened the wrong conversation or none at
    all. Verified on zellij 0.42.2: two panes of one session export
    ``ZELLIJ_SESSION_NAME=<same>`` with ``ZELLIJ_PANE_ID=0`` and ``=1``.

    The session name stays in the key because a pane id is only unique WITHIN
    a session — pane 0 exists in every one of them.
    """

    name = "zellij"
    #: (session, pane) is the only identity that satisfies the one-file-per-pane
    #: contract at the top of this module. No coarser fallback is offered: a
    #: zellij that exports the session name but not the pane id would collide
    #: silently, and publishing nothing is the better failure.
    pane_envs = (("ZELLIJ_SESSION_NAME", "ZELLIJ_PANE_ID"),)


class ScreenBackend(_FileBackend):
    """GNU screen, keyed by session (``STY``) and window index (``WINDOW``).

    Same collision as zellij's and fixed the same way: ``STY`` names the
    session, so several ``lop`` windows in one screen session shared a marker
    file. Verified on screen 4.00.03: a process inside a session exports both
    ``STY=<pid>.<tty>`` and ``WINDOW=0``.

    ``STY`` alone is kept as a fallback — unlike zellij's pane id, ``WINDOW``
    is set by the shell's startup rather than by screen in every configuration,
    and a per-session marker still restores correctly for the one-window-per-
    session case that is how screen is usually run. When it is used, the marker
    is per session and a second ``lop`` in that session overwrites it; the
    module contract above and the docs table say so.
    """

    name = "screen"
    pane_envs = (("STY", "WINDOW"), ("STY",))
