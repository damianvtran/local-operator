"""The OS-owned gesture for a device-admission write.

This module launches a command; it never accepts or resolves an admin password.
Onboarding keeps its existing printed/sudo routes. Admission uses this separate
entry point because a stored password is not proof that a person confirmed a
phone. Real UAC, polkit and SecurityAgent behaviour requires attended host QA.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass

from local_operator.operator import trust


@dataclass(frozen=True)
class ElevationResult:
    """The launcher's outcome, including the cancellation the caller must preserve.

    ``refusal`` is the SENTENCE explaining that no gesture could be asked for at all
    (UAC off or set never-notify, no polkit and no sudo). It is carried rather than
    printed here because this module only launches a command; the caller owns the
    stream it reports on — and it is carried at all because discarding it made every
    one of those states exit silently while the pairing card promised a prompt
    (design round 1, D2).

    ``diagnosis`` is the privileged command's OWN error output, on macOS only, for a
    run that failed without being cancelled. macOS captures osascript's stderr to read
    the cancel number, so the install script's refusal ("the installed anchor changed
    after it was read; nothing was installed") no longer reached the terminal and the
    user saw only the generic "did not complete" (agent review round 4, m3). It is
    carried for the same reason ``refusal`` is, and is empty on a cancel: a cancel
    already has its own sentence and osascript's "User canceled. (-128)" adds nothing.
    """

    exit_code: int
    cancelled: bool = False
    refusal: str = ""
    diagnosis: str = ""
    #: A machine code for a refusal a caller must branch on without parsing the
    #: sentence (today only :data:`REFUSAL_NO_PASSWORD_PROMPT`); ``""`` otherwise.
    refusal_code: str = ""


def _without_sudo(command: list[list[str]]) -> list[list[str]]:
    """Remove sudo: the OS launcher, not the inner command, supplies elevation."""
    return [argv[1:] if argv and argv[0] == "sudo" else argv for argv in command]


#: WHY the OS is asking, as one of a FIXED set (design v2 §2.4, D1). The macOS
#: prompt used to say "…to confirm a phone" for every privileged anchor write, so a
#: person removing a phone, or turning phone approvals on, was asked to approve
#: something they were not doing — the one sentence they read before typing their
#: computer's password named the wrong action. The set is closed on purpose: the
#: reason is chosen by the CALLER'S code path, never from request data, so no
#: client can put words of its own into an OS password dialog.
PURPOSE_CONFIRM_PHONE = "confirm_phone"
PURPOSE_REMOVE_PHONE = "remove_phone"
PURPOSE_PHONE_APPROVALS = "phone_approvals"
PURPOSES = (PURPOSE_CONFIRM_PHONE, PURPOSE_REMOVE_PHONE, PURPOSE_PHONE_APPROVALS)


def _purpose_copy(purpose: str) -> tuple[str, str]:
    """The OS-prompt reason and the card title for ``purpose``, in the active locale.

    Keyed (``wire.operator_setup``) rather than literal, because the prompt reason is
    product copy a person reads inside the OS dialog. Rendered at call time, never
    cached at import, so a language change reaches the next prompt. An unknown
    purpose is a programming error and raises rather than falling back to "confirm a
    phone" — the silent fallback is exactly the mislabelled prompt this replaces.
    """
    from local_operator.i18n.keys import wire_operator_setup as _ws
    from local_operator.i18n.text import tr

    t = tr("wire.operator_setup")
    if purpose == PURPOSE_CONFIRM_PHONE:
        return t(_ws.prompt_confirm_phone()), t(_ws.title_confirm_phone())
    if purpose == PURPOSE_REMOVE_PHONE:
        return t(_ws.prompt_remove_phone()), t(_ws.title_remove_phone())
    if purpose == PURPOSE_PHONE_APPROVALS:
        return t(_ws.prompt_phone_approvals()), t(_ws.title_phone_approvals())
    raise ValueError(f"unknown elevation purpose: {purpose!r}")


def _applescript_literal(text: str) -> str:
    """``text`` escaped for the inside of an AppleScript double-quoted string."""
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _macos_prompt(command: list[list[str]], purpose: str = PURPOSE_CONFIRM_PHONE) -> list[str]:
    from local_operator.operator import format_install_commands

    script = format_install_commands(_without_sudo(command))
    # The script is a shell-quoted argv, then an AppleScript string literal. Both
    # quoting layers are required; interpolating either one alone is an injection.
    quoted = _applescript_literal(script)
    # The reason is catalogue copy, and a translation may contain a quote or a
    # backslash: it gets the same AppleScript escaping as the script, never raw.
    reason = _applescript_literal(_purpose_copy(purpose)[0])
    return [
        "osascript",
        "-e",
        f'do shell script "{quoted}" with administrator privileges with prompt "{reason}"',
    ]


def windows_uac_refusal() -> str | None:
    """Why UAC cannot ask for a gesture here, as the sentence a person reads, or ``None``.

    ONE REMEDY PER BRANCH (design round 2, D11). An unreadable policy is not proof that
    UAC is off — it may well be on — so telling that person to "turn it back on" sent
    them to fix something that was not broken. Each branch now says what it knows.
    """
    unreadable = (
        "Windows did not say whether it will ask you for approval, so no phone was "
        "confirmed. Check that User Account Control is on, then try again."
    )
    # A STATIC platform check, not only the ImportError below: pyright narrows on
    # ``sys.platform`` and so type-checks the registry calls against the real
    # ``winreg`` stubs instead of reporting them unknown off Windows. Off Windows the
    # answer is the same as an unreadable policy -- nothing proves UAC will ask.
    if sys.platform != "win32":
        return unreadable
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\Policies\System",
        ) as key:
            enabled = winreg.QueryValueEx(key, "EnableLUA")[0]
            consent = winreg.QueryValueEx(key, "ConsentPromptBehaviorAdmin")[0]
    except (ImportError, OSError):
        # A missing/unreadable policy must not become a claim that UAC will ask.
        return unreadable
    if not enabled or consent == 0:
        return (
            "Windows User Account Control is set not to ask for approval, so this "
            "computer cannot confirm a phone. Nothing was confirmed. Turn User Account "
            "Control back on and try again."
        )
    return None


#: The sentence for a host with no launcher that can ask a person (design round 2,
#: D11): it names a next step, and not "polkit", a word a phone owner never met.
NO_LAUNCHER_REFUSAL = (
    "This computer has no way to ask you for an administrator's approval, so nothing "
    "was confirmed. Install sudo (or ask whoever runs this computer to), then try again."
)


def prompt_refusal(*, windows: bool | None = None) -> str | None:
    """Why no in-person prompt CAN appear on this host, or ``None`` when one can.

    SPLIT OUT OF ``run_elevated`` SO A CALLER CAN ASK BEFORE IT PROMISES (design round
    2, D8). The CLI prints the gesture it is about to ask for ("Your Mac will ask for
    your admin password."); printed before this question was answered, that promise
    sat one line above the sentence admitting no prompt could happen. Both the caller's
    pre-check and ``run_elevated`` read this one function, so they cannot disagree.
    """
    is_windows = trust._on_windows() if windows is None else windows
    if is_windows:
        return windows_uac_refusal()
    if sys.platform == "darwin":
        return None
    if shutil.which("pkexec") is not None or shutil.which("sudo") is not None:
        return None
    return NO_LAUNCHER_REFUSAL


#: osascript's error number for "User canceled." on an administrator-privileges
#: prompt. osascript exits 1 for EVERY script error — including the install script
#: refusing because the anchor changed (its `exit 3`) — so the exit code alone cannot
#: tell a cancel from a failure (agent review round 3, m2); this number can.
_OSASCRIPT_USER_CANCELED = "(-128)"


#: The machine code a ``run_elevated(..., terminal_sudo=False)`` refusal carries in
#: ``ElevationResult.refusal_code`` when Linux has no pkexec (design v2 §6 item 2).
REFUSAL_NO_PASSWORD_PROMPT = "no_password_prompt"


def no_password_prompt_refusal() -> str:
    """The U18 sentence: this host can show no GUI password prompt (A2, U18)."""
    from local_operator.i18n.keys import wire_operator_setup as _ws
    from local_operator.i18n.text import tr

    return tr("wire.operator_setup")(_ws.refused_no_password_prompt())


def gui_prompt_refusal(*, windows: bool | None = None) -> str | None:
    """Like :func:`prompt_refusal`, but a terminal ``sudo`` does not count.

    FOR CALLERS WITH NO TERMINAL A PERSON IS LOOKING AT — the desktop server (design
    v2 §6 item 2). ``prompt_refusal`` accepts a bare ``sudo`` because a TTY-owning
    caller (the CLI, the TUI under ``App.suspend()``) gets a real in-person password
    prompt from it. A server has no TTY, so ``sudo`` there either fails or — under
    passwordless sudo — SUCCEEDS WITH NO HUMAN AT ALL, which would turn "an OS prompt
    a person must answer" into "any bearer holder installs authority". So on Linux
    only pkexec (a polkit dialog) counts here; macOS and Windows have their own GUI
    prompts and answer exactly as :func:`prompt_refusal` does.
    """
    is_windows = trust._on_windows() if windows is None else windows
    refusal = prompt_refusal(windows=is_windows)
    if refusal is not None:
        return refusal
    if is_windows or sys.platform == "darwin":
        return None
    if shutil.which("pkexec") is None:
        return no_password_prompt_refusal()
    return None


def run_elevated(
    command: list[list[str]],
    *,
    windows: bool | None = None,
    purpose: str = PURPOSE_CONFIRM_PHONE,
    terminal_sudo: bool = True,
) -> ElevationResult:
    """Launch the platform's in-person admin gesture, without a shell wrapper.

    ``purpose`` picks the reason the OS prompt shows (one of :data:`PURPOSES`); the
    default keeps every existing caller's "confirm a phone" wording.

    ``terminal_sudo=False`` REFUSES the headless ``sh -c sudo`` fallback instead of
    running it, for callers with no terminal (the desktop server, design v2 §6
    item 2). It is enforced HERE, where the launcher is chosen, not only by the
    caller's precheck, so the pkexec answer that was checked is the one acted on.
    """
    is_windows = trust._on_windows() if windows is None else windows
    refusal = prompt_refusal(windows=is_windows)
    if refusal is not None:
        # The honest account of why no prompt can appear travels to the caller
        # instead of the bare exit code that used to say nothing at all (design
        # round 1, D2).
        return ElevationResult(2, refusal=refusal)
    if purpose not in PURPOSES:
        raise ValueError(f"unknown elevation purpose: {purpose!r}")
    if is_windows:
        # install_commands' Windows command is ShellExecute runas via PowerShell;
        # running it here (unlike onboarding) raises UAC and waits for its result.
        argv = command[0]
    elif sys.platform == "darwin":
        argv = _macos_prompt(command, purpose)
    elif shutil.which("pkexec") is not None:
        from local_operator.operator import format_install_commands

        argv = ["pkexec", "sh", "-c", format_install_commands(_without_sudo(command))]
    elif not terminal_sudo:
        return ElevationResult(
            2, refusal=no_password_prompt_refusal(), refusal_code=REFUSAL_NO_PASSWORD_PROMPT
        )
    else:
        # Headless Linux has no polkit agent. sudo's terminal prompt is still an
        # in-person gesture; its password never passes through this process.
        argv = ["sh", "-c", " && ".join(shlex.join(row) for row in command)]
    darwin = sys.platform == "darwin" and not is_windows
    try:
        # Only osascript's stderr is captured: it carries the cancel number and no
        # secret, while sudo/pkexec keep the terminal they prompt on. ``text=True``
        # everywhere so the captured stream is always ``str`` (with nothing captured
        # it changes nothing).
        result = subprocess.run(argv, check=False, capture_output=darwin, text=True)
    except OSError:
        return ElevationResult(1)
    cancelled = (
        (darwin and result.returncode == 1 and _OSASCRIPT_USER_CANCELED in (result.stderr or ""))
        or (argv[0] == "pkexec" and result.returncode in (126, 127))
        or (argv[0] == "sh" and result.returncode in (1, 130, -2))
        or (is_windows and result.returncode in (1223, -1073740771))
    )
    diagnosis = ""
    if darwin and result.returncode != 0 and not cancelled:
        # Bounded: it is printed to a terminal, and a runaway script must not flood it.
        diagnosis = (result.stderr or "").strip()[:2000]
    return ElevationResult(result.returncode, cancelled, diagnosis=diagnosis)


def confirm_card_copy(
    *, windows: bool | None = None, purpose: str = PURPOSE_CONFIRM_PHONE
) -> tuple[str, str, str]:
    """The confirming card's words: title, body, and the gesture it promises.

    ONE source for the fleet promise, because two copies of it drift exactly where a
    person cannot check them (decision 9).

    IT HAS TWO CONSUMERS, and each takes the part it owns. The desktop card renders
    title+body from the mesh lane's contract. The CLI takes the GESTURE sentence and
    prints it before it asks, so the promise printed on the verb and the prompt that
    follows cannot disagree — and when no prompt can be asked for at all, the same
    run prints why (``handlers._admission_install``); agent review round 2, m6.

    ``windows`` defaults to the host, matching :func:`run_elevated`, so a caller
    cannot promise the wrong platform's gesture by forgetting to pass it.

    ``purpose`` is the SAME closed set the OS prompt uses (design v2 §2.4), so the
    card a person reads and the dialog that follows name the same action. The title
    follows the purpose; the confirm body is unchanged; the turn-on-approvals body is
    the setup offer's own sentence. The removal body is left to the revoke surface
    that owns it (C2), so it returns the confirm body's neutral absence: ``""``.
    """
    is_windows = trust._on_windows() if windows is None else windows
    title = _purpose_copy(purpose)[1]
    if purpose == PURPOSE_PHONE_APPROVALS:
        from local_operator.i18n.keys import wire_operator_setup as _ws
        from local_operator.i18n.text import tr

        body = tr("wire.operator_setup")(_ws.body_phone_approvals())
    elif purpose == PURPOSE_REMOVE_PHONE:
        body = ""
    else:
        body = (
            "Confirming lets this phone approve things on this computer. It will also be "
            "trusted on any computer that joins this one later. Computers that joined "
            "earlier keep their own list."
        )
    if is_windows:
        gesture = "Windows will ask you to allow this."
    elif sys.platform == "darwin":
        gesture = "Your Mac will ask for your admin password."
    else:
        gesture = "Your computer will ask for your password."
    return title, body, gesture


__all__ = [
    "NO_LAUNCHER_REFUSAL",
    "PURPOSES",
    "PURPOSE_CONFIRM_PHONE",
    "PURPOSE_PHONE_APPROVALS",
    "PURPOSE_REMOVE_PHONE",
    "REFUSAL_NO_PASSWORD_PROMPT",
    "ElevationResult",
    "confirm_card_copy",
    "gui_prompt_refusal",
    "no_password_prompt_refusal",
    "prompt_refusal",
    "run_elevated",
    "windows_uac_refusal",
]
