"""``lop permissions status`` / ``lop permissions setup``: the runtime's macOS privacy identity.

WHY THIS COMMAND EXISTS. Every Local Operator runtime (TUI/UI session
runtimes, ``lop exec`` workers, eval workers, launchd daemons) executes from a
bundle-wrapped image (see "ONE MACOS PRIVACY IDENTITY" in ``procname``), so
macOS keys its privacy grants by a bundle id instead of a path. Which id
depends on the host: ``com.local-operator.runtime`` ("Local Operator
Runtime", ad-hoc, every host) or ``com.local-operator`` (the desktop app's own
row, signed with the operator's Developer ID after they opt in here). An
operator looking at System Settings needs to know which row to enable;
``status`` answers that and ``setup`` is the ONLY path that records the opt-in.

CONSTRAINTS, each from an incident or a measurement:

- ``status`` is read-only and NEVER prompts: no plant, no signing, no keychain
  read, and Screen Recording is checked with ``CGPreflightScreenCaptureAccess``
  only (never the request call, which would put a dialog on screen). The check
  runs in a DISCLAIMED child executed through the runtime image, because the
  ``lop`` process itself is attributed to the terminal that started it (its
  responsible process), so a preflight here would report the TERMINAL's grant.
- ``setup`` refuses without a TTY and in an isolated HOME/config/runtime root
  (``procname.team_signing_refusal``): Developer ID signing can raise the
  keychain dialog, and under a scratch ``HOME`` macOS instead asks to CREATE a
  login keychain (the 2026-09-18 rig incident).
- ``security find-identity`` lists certificates and whether a private key is
  present; it does not use the key, so it does not prompt. The first USE of
  the key is the ``codesign`` inside ``publish_team_runtime``, after the
  operator has read the explanation and said yes.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

from local_operator import procname
from local_operator.i18n.keys import cli_permissions as _k
from local_operator.i18n.text import tr

_t = tr("cli.permissions")

#: The probe a runtime image runs for ``status``: preflight ONLY. ``-I`` keeps
#: the user's site/env out of a three-line ctypes call.
_PREFLIGHT_PROBE = (
    "import ctypes;"
    "cg=ctypes.CDLL('/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics');"
    "f=cg.CGPreflightScreenCaptureAccess;f.restype=ctypes.c_bool;"
    "print('granted' if f() else 'denied')"
)


def _runtime_preflight() -> str:
    """``granted``/``denied``/``unknown`` for Screen Recording, as a runtime sees it.

    Spawned exactly like a runtime: through :func:`procname.spawn_identity`
    (the bundle image, labelled) and :func:`macos_disclaim.spawn_disclaimed`
    (out of this process's responsibility scope), so tccd evaluates the
    runtime's identity, not the terminal's. Preflight never prompts.
    """
    try:
        from local_operator import macos_disclaim

        argv0, executable = procname.spawn_identity(procname.LABEL_DAEMONS_REFRESH)
        if executable is None:
            return "unknown"
        read_fd, write_fd = os.pipe()
        try:
            child = macos_disclaim.spawn_disclaimed(
                [argv0, "-I", "-c", _PREFLIGHT_PROBE],
                executable=executable,
                stdin=subprocess.DEVNULL,
                stdout=write_fd,
                stderr=subprocess.DEVNULL,
            )
        finally:
            os.close(write_fd)
        with os.fdopen(read_fd, "r", encoding="utf-8") as reader:
            answer = reader.read().strip()
        child.wait(timeout=30)
        return answer if answer in {"granted", "denied"} else "unknown"
    except Exception:  # noqa: BLE001 — a status read never fails
        return "unknown"


def _status() -> int:
    if sys.platform != "darwin":
        print(_t(_k.status_not_macos()))
        return 0
    info = procname.runtime_identity()
    flavor = info.get("flavor")
    identifier = str(info.get("identifier") or "")
    if not info.get("planted"):
        print(_t(_k.status_identity_none()))
    elif flavor == procname.FLAVOR_TEAM:
        print(_t(_k.status_identity_team(identifier=identifier or procname.APP_BUNDLE_ID)))
    elif flavor == procname.FLAVOR_ADHOC:
        print(_t(_k.status_identity_adhoc(identifier=identifier or procname.RUNTIME_BUNDLE_ID)))
    else:
        print(_t(_k.status_identity_legacy()))
    if info.get("image"):
        print(_t(_k.status_image(path=str(info["image"]))))
    app = procname.installed_app()
    if app is None:
        print(_t(_k.status_app_missing()))
    else:
        team = procname.code_identity(app).get("TeamIdentifier", "?")
        print(_t(_k.status_app_found(path=str(app), team=team)))
    refusal = procname.team_signing_refusal()
    record = procname.read_signing_optin()
    if refusal is not None:
        print(_t(_k.status_optin_blocked(reason=refusal)))
    elif record is not None:
        print(_t(_k.status_optin_on(identity=str(record["identity"]), team=str(record["team"]))))
    else:
        print(_t(_k.status_optin_off()))
    # Only through an image that already exists: spawning through
    # ``spawn_identity`` would otherwise PLANT one, and status is read-only.
    preflight = _runtime_preflight() if info.get("planted") else "unknown"
    if preflight == "granted":
        print(_t(_k.status_screen_granted()))
    elif preflight == "denied":
        row = "Local Operator" if flavor == procname.FLAVOR_TEAM else procname.RUNTIME_DISPLAY_NAME
        print(_t(_k.status_screen_denied(row=row)))
    else:
        print(_t(_k.status_screen_unknown()))
    print(_t(_k.status_guide()))
    return 0


def _developer_identities(team: str) -> list[tuple[str, str]]:
    """``(sha1, name)`` of valid "Developer ID Application" identities for ``team``."""
    try:
        result = subprocess.run(
            ["/usr/bin/security", "find-identity", "-v", "-p", "codesigning"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    found: list[tuple[str, str]] = []
    for match in re.finditer(r'^\s*\d+\)\s+([0-9A-F]{40})\s+"([^"]+)"', result.stdout, re.M):
        sha1, name = match.groups()
        if name.startswith("Developer ID Application:") and name.endswith(f"({team})"):
            found.append((sha1, name))
    return found


def _ask(prompt: str) -> bool:
    try:
        return input(prompt).strip().lower() in {"y", "yes"}
    except EOFError:
        return False


def _setup() -> int:
    if sys.platform != "darwin":
        print(_t(_k.status_not_macos()))
        return 0
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        print(_t(_k.setup_refused_tty()), file=sys.stderr)
        return 2
    refusal = procname.team_signing_refusal()
    if refusal is not None:
        print(_t(_k.setup_refused_isolated(reason=refusal)), file=sys.stderr)
        return 2
    app = procname.installed_app()
    team = procname.code_identity(app).get("TeamIdentifier") if app else None
    if app is None or not team or team == "not set":
        print(_t(_k.setup_no_app()), file=sys.stderr)
        return 1
    identities = _developer_identities(team)
    if not identities:
        print(_t(_k.setup_no_identity(team=team)), file=sys.stderr)
        return 1
    sha1, name = identities[0]
    print(_t(_k.setup_explain(name=name)))
    if not _ask(_t(_k.setup_confirm())):
        print(_t(_k.setup_cancelled()))
        return 0
    # Recorded BEFORE signing because ``publish_team_runtime`` reads the record;
    # ``always_allow_confirmed`` stays False until the operator says otherwise,
    # which keeps agent-run (non-TTY) updates from signing until then.
    path = procname.write_signing_optin(sha1, team, always_allow_confirmed=False)
    image = procname.publish_team_runtime(sys.executable, interactive=True)
    if image is None:
        path.unlink(missing_ok=True)
        print(_t(_k.setup_failed()), file=sys.stderr)
        return 1
    always = _ask(_t(_k.setup_always_allow()))
    procname.write_signing_optin(sha1, team, always_allow_confirmed=always)
    # Repoint this venv's link now, so the next runtime from it is the team image.
    procname.ensure_branded_interpreter()
    print(_t(_k.setup_done(identifier=procname.APP_BUNDLE_ID)))
    return 0


def permissions_command(args: argparse.Namespace) -> int:
    """Dispatch ``lop permissions <status|setup>``; bare ``permissions`` is ``status``."""
    action = getattr(args, "permissions_command", None) or "status"
    if action == "setup":
        return _setup()
    return _status()


def register(
    subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]", help_locale: str | None
) -> None:
    """Add the ``permissions`` command group to the top-level parser."""
    parser = subparsers.add_parser(
        "permissions", help=_k.help_permissions().envelope(locale=help_locale)["text"]
    )
    actions = parser.add_subparsers(dest="permissions_command")
    actions.add_parser("status", help=_k.help_status().envelope(locale=help_locale)["text"])
    actions.add_parser("setup", help=_k.help_setup().envelope(locale=help_locale)["text"])


__all__ = ["permissions_command", "register"]
