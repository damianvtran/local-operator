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
    """

    exit_code: int
    cancelled: bool = False
    refusal: str = ""


def _without_sudo(command: list[list[str]]) -> list[list[str]]:
    """Remove sudo: the OS launcher, not the inner command, supplies elevation."""
    return [argv[1:] if argv and argv[0] == "sudo" else argv for argv in command]


def _macos_prompt(command: list[list[str]]) -> list[str]:
    from local_operator.operator import format_install_commands

    script = format_install_commands(_without_sudo(command))
    # The script is a shell-quoted argv, then an AppleScript string literal. Both
    # quoting layers are required; interpolating either one alone is an injection.
    quoted = script.replace("\\", "\\\\").replace('"', '\\"')
    return [
        "osascript",
        "-e",
        f'do shell script "{quoted}" with administrator privileges '
        'with prompt "Local Operator needs your approval to confirm a phone on this computer."',
    ]


def windows_uac_refusal() -> str | None:
    """Why UAC cannot ask for a gesture here, as the sentence a person reads, or ``None``.

    ONE REMEDY PER BRANCH (design round 2, D11). An unreadable policy is not proof that
    UAC is off — it may well be on — so telling that person to "turn it back on" sent
    them to fix something that was not broken. Each branch now says what it knows.
    """
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
        return (
            "Windows did not say whether it will ask you for approval, so no phone was "
            "confirmed. Check that User Account Control is on, then try again."
        )
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


def run_elevated(command: list[list[str]], *, windows: bool | None = None) -> ElevationResult:
    """Launch the platform's in-person admin gesture, without a shell wrapper."""
    is_windows = trust._on_windows() if windows is None else windows
    refusal = prompt_refusal(windows=is_windows)
    if refusal is not None:
        # The honest account of why no prompt can appear travels to the caller
        # instead of the bare exit code that used to say nothing at all (design
        # round 1, D2).
        return ElevationResult(2, refusal=refusal)
    if is_windows:
        # install_commands' Windows command is ShellExecute runas via PowerShell;
        # running it here (unlike onboarding) raises UAC and waits for its result.
        argv = command[0]
    elif sys.platform == "darwin":
        argv = _macos_prompt(command)
    elif shutil.which("pkexec") is not None:
        from local_operator.operator import format_install_commands

        argv = ["pkexec", "sh", "-c", format_install_commands(_without_sudo(command))]
    else:
        # Headless Linux has no polkit agent. sudo's terminal prompt is still an
        # in-person gesture; its password never passes through this process.
        argv = ["sh", "-c", " && ".join(shlex.join(row) for row in command)]
    darwin = sys.platform == "darwin" and not is_windows
    try:
        # Only osascript's stderr is captured: it carries the cancel number and no
        # secret, while sudo/pkexec keep the terminal they prompt on.
        result = subprocess.run(argv, check=False, capture_output=darwin, text=darwin)
    except OSError:
        return ElevationResult(1)
    cancelled = (
        (darwin and result.returncode == 1 and _OSASCRIPT_USER_CANCELED in (result.stderr or ""))
        or (argv[0] == "pkexec" and result.returncode in (126, 127))
        or (argv[0] == "sh" and result.returncode in (1, 130, -2))
        or (is_windows and result.returncode in (1223, -1073740771))
    )
    return ElevationResult(result.returncode, cancelled)


def confirm_card_copy(*, windows: bool | None = None) -> tuple[str, str, str]:
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
    """
    is_windows = trust._on_windows() if windows is None else windows
    title = "Confirm this phone on this computer"
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
    "ElevationResult",
    "confirm_card_copy",
    "prompt_refusal",
    "run_elevated",
    "windows_uac_refusal",
]
