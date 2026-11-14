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

- ``status`` is read-only and NEVER prompts: no plant, no link, no signing,
  no keychain key use (``security find-identity`` only lists certificates),
  and Screen Recording is checked with ``CGPreflightScreenCaptureAccess``
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
import signal
import subprocess
import sys
from typing import Any

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


#: Upper bound on the preflight child, in seconds (review round 1, N2: the
#: read used to have no bound, so a hung probe hung ``status``). The probe is
#: one ctypes call; see the PR thread for the measured n/median/max this cap
#: sits far above.
_PREFLIGHT_TIMEOUT_S = 10.0

#: ``procname.team_signing_refusal`` codes → the operator-facing reason (UX
#: round 1, U8: the bare machine code read as jargon).
_REFUSAL_REASONS = {
    "isolated-home": _k.status_reason_isolated_home,
    "runtime-root-override": _k.status_reason_runtime_root_override,
    "config-dir-override": _k.status_reason_config_dir_override,
    "not-darwin": _k.status_reason_not_darwin,
}


def _refusal_text(code: str) -> str:
    message = _REFUSAL_REASONS.get(code)
    return _t(message()) if message is not None else code


def _guide_path() -> str:
    """The shipped guide file, which a shell user can open directly.

    ``guide://permissions`` resolves only inside a Local Operator session, so
    the status footer names the packaged file too (UX round 1, U3).
    """
    from pathlib import Path

    return str(Path(procname.__file__).resolve().parent / "guides" / "permissions" / "GUIDE.md")


def _runtime_preflight(image: str) -> str:
    """``granted``/``denied``/``unknown`` for Screen Recording, as a runtime sees it.

    Executes ``image`` — the image :func:`procname.runtime_identity` already
    selected — DIRECTLY, labelled like a runtime and through
    :func:`macos_disclaim.spawn_disclaimed` (out of this process's
    responsibility scope), so tccd evaluates the runtime's identity, not the
    terminal's. Never through ``procname.spawn_identity``: that runs the plant
    ladder, which links this venv and can even build an image, so ``status``
    would write (review round 1, R3 / QA Q1) and could preflight a different
    image from the one it reported. Preflight never prompts.
    """
    child = None
    try:
        from local_operator import macos_disclaim

        argv0 = procname.branded_argv0(procname.LABEL_DAEMONS_REFRESH)
        read_fd, write_fd = os.pipe()
        try:
            child = macos_disclaim.spawn_disclaimed(
                [argv0, "-I", "-c", _PREFLIGHT_PROBE],
                executable=image,
                stdin=subprocess.DEVNULL,
                stdout=write_fd,
                stderr=subprocess.DEVNULL,
            )
        finally:
            os.close(write_fd)
        with os.fdopen(read_fd, "r", encoding="utf-8") as reader:
            # The answer is one word, far below the pipe buffer, so the child
            # never blocks on the write: waiting first and reading after is
            # what puts the bound on a hung probe.
            child.wait(timeout=_PREFLIGHT_TIMEOUT_S)
            answer = reader.read().strip()
        return answer if answer in {"granted", "denied"} else "unknown"
    except Exception:  # noqa: BLE001 — a status read never fails
        if child is not None and child.poll() is None:
            try:
                os.kill(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
            except Exception:  # noqa: BLE001 — best effort reap of our own child
                pass
        return "unknown"


def _optin_line(app_team: str | None) -> str:
    """The "Team signing" line: on, off (setup can work), or why it cannot."""
    refusal = procname.team_signing_refusal()
    if refusal is not None:
        return _t(_k.status_optin_blocked(reason=_refusal_text(refusal)))
    record = procname.read_signing_optin()
    if record is not None:
        return _t(_k.status_optin_on(identity=str(record["identity"]), team=str(record["team"])))
    # Suggest ``setup`` only when it can succeed: it needs a Developer ID key
    # of the APP's team, which in practice only the app's publisher holds, so
    # pointing everyone else at it ended in "no signing identity" (UX round 1,
    # U2). ``security find-identity`` lists certificates without using a key,
    # so it does not prompt; the refusal above keeps it out of isolated rigs.
    if not app_team:
        return _t(_k.status_optin_no_app())
    if not _developer_identities(app_team):
        return _t(_k.status_optin_unavailable(team=app_team))
    return _t(_k.status_optin_off())


def _status() -> int:
    if sys.platform != "darwin":
        print(_t(_k.status_not_macos()))
        return 0
    info = procname.runtime_identity()
    flavor = info.get("flavor")
    identifier = str(info.get("identifier") or "")
    planted = bool(info.get("planted"))
    if not planted:
        print(_t(_k.status_identity_none()))
    elif flavor == procname.FLAVOR_TEAM:
        print(_t(_k.status_identity_team(identifier=identifier or procname.APP_BUNDLE_ID)))
    elif flavor == procname.FLAVOR_ADHOC:
        print(_t(_k.status_identity_adhoc(identifier=identifier or procname.RUNTIME_BUNDLE_ID)))
    else:
        print(_t(_k.status_identity_legacy()))
    # Only an image that exists: before the first plant ``image`` names the
    # path one WOULD be built at (UX round 1, U6).
    if planted and info.get("image"):
        print(_t(_k.status_image(path=str(info["image"]))))
    app = procname.installed_app()
    app_team: str | None = None
    if app is None:
        print(_t(_k.status_app_missing()))
    else:
        team = procname.code_identity(app).get("TeamIdentifier", "?")
        app_team = team if team not in {"?", "not set"} else None
        print(_t(_k.status_app_found(path=str(app), team=team)))
    print(_optin_line(app_team))
    if not planted or not info.get("image"):
        print(_t(_k.status_screen_no_image()))
    else:
        preflight = _runtime_preflight(str(info["image"]))
        if preflight == "granted":
            print(_t(_k.status_screen_granted()))
        elif preflight == "denied":
            # Only the ad-hoc bundle carries the "Local Operator Runtime" name;
            # the team image shares the app's row and a path-keyed (legacy)
            # image shows as its executable, "Local Operator" (review N1).
            row = (
                procname.RUNTIME_DISPLAY_NAME
                if flavor == procname.FLAVOR_ADHOC
                else "Local Operator"
            )
            print(_t(_k.status_screen_denied(row=row)))
        else:
            print(_t(_k.status_screen_unknown()))
    print(_t(_k.status_scope()))
    print(_t(_k.status_guide(path=_guide_path())))
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


def _restore_optin(prior: dict[str, object] | None) -> None:
    """Put the opt-in back as it was before this setup run (review round 1, N3).

    A failed re-run used to delete the record outright, silently dropping a
    consent recorded by an earlier, successful setup.
    """
    path = procname._optin_path()
    if prior is None:
        if path is not None:
            path.unlink(missing_ok=True)
        return
    procname.write_signing_optin(
        str(prior["identity"]),
        str(prior["team"]),
        always_allow_confirmed=bool(prior.get("always_allow_confirmed")),
    )


def _setup() -> int:
    if sys.platform != "darwin":
        print(_t(_k.status_not_macos()))
        return 0
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        print(_t(_k.setup_refused_tty()), file=sys.stderr)
        return 2
    refusal = procname.team_signing_refusal()
    if refusal is not None:
        print(_t(_k.setup_refused_isolated(reason=_refusal_text(refusal))), file=sys.stderr)
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
    prior = procname.read_signing_optin()
    # A prior "Always Allow" answer for the SAME key still describes that
    # key's access list; any other key starts unconfirmed.
    kept = bool(
        prior is not None
        and str(prior["identity"]) == sha1.upper()
        and prior.get("always_allow_confirmed")
    )
    # Recorded BEFORE signing because ``publish_team_runtime`` reads the record;
    # ``always_allow_confirmed`` stays False until the operator says otherwise,
    # which keeps unattended updates from signing until then.
    procname.write_signing_optin(sha1, team, always_allow_confirmed=kept)
    result = procname.publish_team_runtime_result(sys.executable, interactive=True)
    if result.image is None:
        _restore_optin(prior)
        print(_t(_k.setup_failed(detail=result.detail or str(result.reason))), file=sys.stderr)
        return 1
    if result.signed:
        always = _ask(_t(_k.setup_always_allow()))
        procname.write_signing_optin(sha1, team, always_allow_confirmed=always)
    else:
        # The image already existed, so no codesign ran and no keychain
        # dialog was answered: asking now would record "Always Allow" for an
        # access list nobody exercised (review round 1, R4). Keep what is known.
        always = kept
        print(_t(_k.setup_already_signed()))
        # Only claim the earlier answer survived when it did: with no prior
        # confirmed "Always Allow" for this key, saying "kept" and then printing
        # the allow-once cost contradicted itself (review round 2, m1/Q2/U9).
        if kept:
            print(_t(_k.setup_always_allow_kept()))
    # Repoint this venv's link now, so the next runtime from it is the team image.
    procname.ensure_branded_interpreter()
    print(_t(_k.setup_done(identifier=procname.APP_BUNDLE_ID)))
    if not always:
        # Say what the "no" costs (UX round 1, U5): otherwise this was guide-only.
        print(_t(_k.setup_done_allow_once()))
    return 0


def permissions_command(args: argparse.Namespace) -> int:
    """Dispatch ``lop permissions <status|setup>``; bare ``permissions`` is ``status``."""
    action = getattr(args, "permissions_command", None) or "status"
    if action == "setup":
        return _setup()
    return _status()


def register(subparsers: "argparse._SubParsersAction[Any]", help_locale: str | None) -> None:
    """Add the ``permissions`` command group to the top-level parser."""
    parser = subparsers.add_parser(
        "permissions", help=_k.help_permissions().envelope(locale=help_locale)["text"]
    )
    actions = parser.add_subparsers(dest="permissions_command")
    actions.add_parser("status", help=_k.help_status().envelope(locale=help_locale)["text"])
    actions.add_parser("setup", help=_k.help_setup().envelope(locale=help_locale)["text"])


__all__ = ["permissions_command", "register"]
