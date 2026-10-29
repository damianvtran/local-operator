"""``lop operator`` verbs. Imported only after an ``operator`` verb is dispatched.

THE COPY RULES THESE HANDLERS FOLLOW, all of them from the design round and the
acceptance matrix:

* ``init`` reports the LEVEL it achieved. A setup step that printed "operator key
  created" without saying whether a signature costs a human gesture would be
  exactly the overclaim §2.2 exists to prevent, so the level line is the verb's
  main output rather than a footnote.
* ``sign`` prints machine-readable JSON on stdout and every human word on
  stderr, because its consumer is the TUI's attached pane or the desktop backend
  (and, in stage D, the relay) — a banner line on stdout would corrupt the one
  value crossing that pipe, the same failure ``lop secret get`` documents.
* nothing here ever prints key material: the private half never leaves the
  backend, and the anchor holds public data by construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from local_operator.operator import (
    LEVEL_ANCHOR_UNPINNED,
    LEVEL_OPERATOR_FILE_ONLY,
    LEVEL_OPERATOR_PRESENCE,
    LEVEL_SPAWN_ONLY,
    OperatorAnchor,
    anchor_bytes,
    anchor_path,
    create_key,
    format_install_commands,
    install_commands,
    key_id_for,
    load_anchor,
    load_staged_anchor,
    operator_authority_report,
    sign_challenge,
    staging_path,
)
from local_operator.operator import trust as trust_module
from local_operator.operator.keychain import (
    FILE_ONLY,
    SECURE_ENCLAVE,
    KeyBackendError,
    KeyHandle,
)
from local_operator.operator.sign import anchor_for_handle, describe_level
from local_operator.operator.verify import spki_fp
from local_operator.paths import config_dir


def dispatch(args: argparse.Namespace) -> int:
    command = getattr(args, "operator_command", None)
    if command == "init":
        return _init(args)
    if command == "install":
        return _install(args)
    if command == "anchor":
        return _anchor(args)
    if command == "setup":
        return _setup(args)
    if command == "trust":
        return _trust()
    if command == "status":
        return _status()
    if command == "sign":
        return _sign(args)
    if command == "devices":
        from local_operator.operator.pair_handlers import describe_devices

        return describe_devices(args)
    print(
        "usage: lop operator {init|install|anchor|setup|trust|sign|status|devices}",
        file=sys.stderr,
    )
    return 2


def _is_privileged() -> bool:
    """Whether this process may already write the anchor's root-owned path.

    POSIX answers with ``os.geteuid``; Windows has no such attribute (reading it
    raises ``AttributeError``, which the probe battery counts as a fatal
    unguarded POSIX attribute), and there is no cheap, dependency-free way to ask
    whether the token is elevated — so Windows answers ``False`` and takes the
    print-the-command path. That is the honest outcome: a Windows operator runs
    the PowerShell step from an elevated shell, and a `lop` process that guessed
    "elevated" would write a non-administrator-owned file into ``%PROGRAMDATA%``
    and call it an anchor.
    """
    geteuid = getattr(os, "geteuid", None)
    return bool(geteuid is not None and geteuid() == 0)


def stage_anchor(root: Path, anchor: Any) -> Path:
    """Write an anchor statement to the staging path, 0600 under a 0700 directory.

    Split out of ``init`` (and reused by ``devices --revoke``) so the STAGING
    half of onboarding exists once. Both callers produce a statement the one
    privileged install step then moves, which is what keeps "root-owned or
    nothing" true for every anchor write rather than only for the first one.
    """
    staged = staging_path(root)
    staged.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(staged.parent, 0o700)
    staged.write_bytes(anchor_bytes(anchor))
    os.chmod(staged, 0o600)
    return staged


#: The ``purpose`` of an anchor write that ADDS a device to the operator's list
#: (S4a pair-op calls ``install_anchor`` with it). See ``install_anchor`` (F5).
PURPOSE_ADMIT_DEVICE = "admit_device"

#: WHY an admission write did not happen — ONE code per cause. A single
#: ``pair_device_not_admitted`` covered a cancelled prompt, a host that cannot ask
#: for approval at all, the migration guard, and "this phone has no pending
#: request", and its one remedy ("confirm the phone here and try again") is right
#: for the first and wrong — for the no-pending case a dead end — for the rest
#: (agent review round 2, m2). ``admit_device``/``confirm_devices`` and
#: ``_admission_install`` return these, and ``pair_op`` turns one into the refusal
#: code and sentence the card shows.
ADMIT_OK = 0
ADMIT_NO_INSTALLED_ANCHOR = 1
ADMIT_MIGRATION_GUARD = 2
ADMIT_NO_PENDING_REQUEST = 3
ADMIT_REVOKED = 4
ADMIT_CANCELLED = 5
ADMIT_NO_PROMPT = 6
ADMIT_INSTALL_FAILED = 7
#: `--confirm`/`--approve` named no paired phone, or one this computer does not hold.
#: Its own code (agent review round 3, m5): it used to borrow
#: ``ADMIT_NO_INSTALLED_ANCHOR``, so a typo'd id exited as "no operator statement".
#: CLI-only — ``admit_device`` never returns it, so ``ADMIT_REFUSED`` has no row.
ADMIT_NO_SELECTION = 8

#: The refusal code and sentence the pairing card shows for each cause, keyed by
#: code. ONE map, so a new cause cannot ship with a neighbour's remedy copied into
#: it — which is what one "not confirmed, confirm it and try again" sentence did for
#: all of them (agent review round 2, m2). The sentences name a route that exists:
#: `--authorise` is the only way back from a revocation, and `--confirm` is the only
#: way to end the migration guard.
ADMIT_REFUSED: dict[int, tuple[str, str]] = {
    ADMIT_NO_INSTALLED_ANCHOR: (
        "pair_device_not_admitted",
        "This computer could not read its own operator statement, so no certificate was "
        "issued. Ask Local Operator to set up operator authority here, then pair the "
        "phone again; nothing was installed.",
    ),
    ADMIT_MIGRATION_GUARD: (
        "pair_device_not_admitted",
        "This phone was not confirmed on this computer: confirm the phones you paired "
        "before this version first (`lop operator devices --confirm`), then try the "
        "pairing again; nothing was installed.",
    ),
    ADMIT_NO_PENDING_REQUEST: (
        "pair_device_not_admitted",
        "This phone has no pairing request waiting on this computer, so no certificate "
        "was issued — start the pairing again from the phone; nothing was installed.",
    ),
    ADMIT_REVOKED: (
        "pair_device_revoked",
        # No id here (design round 2, D14): this map holds only fixed strings, so
        # the card names the verb and the CLI's own refusal names the id.
        "This phone was removed on this computer, so no certificate was issued. Bring "
        "it back on this computer first with `lop operator devices --authorise` and "
        "the phone's id (`lop operator devices` lists it), then pair the phone again; "
        "nothing was installed.",
    ),
    ADMIT_CANCELLED: (
        "pair_device_not_admitted",
        "You cancelled the approval, so no certificate was issued. Try the pairing "
        "again when you are ready; nothing was installed.",
    ),
    ADMIT_NO_PROMPT: (
        "pair_device_not_admitted",
        "This computer has no way to ask you for an administrator's approval, so no "
        "certificate was issued. Install sudo (or ask whoever runs this computer to), "
        "then try the pairing again; nothing was installed.",
    ),
    ADMIT_INSTALL_FAILED: (
        "pair_device_not_admitted",
        "The approval did not finish, so no certificate was issued. Try the pairing "
        "again; nothing was installed.",
    ),
}


def _unrevoked_devices(installed: Any, staged: Any) -> list[Any]:
    """Ids whose installed revocation the incoming statement would lift (R1)."""
    staged_revoked = trust_module.revoked_device_ids(staged)
    return [d for d in trust_module.revoked_device_ids(installed) if d not in staged_revoked]


def _admitted_devices(installed: Any, staged: Any) -> list[str]:
    """Device ids the STAGED anchor lets in that the INSTALLED one keeps out (R1-4).

    Tightening (a new revocation) is not an admission. Raw ``==`` is deliberate:
    the runtime compares the JSON values without coercing their types.
    """
    installed_open = trust_module.open_device_ids(installed)
    added = [d for d in trust_module.open_device_ids(staged) if d not in installed_open]
    changed = _unrevoked_devices(installed, staged) + added
    return sorted({repr(d) if not isinstance(d, str) else d for d in changed})


def install_anchor(
    root: Path, *, print_only: bool = False, sudo_secret: str = "", purpose: str = ""
) -> int:
    """The ONE privileged step: move the staged anchor to where the runtime reads it.

    Returns a process exit code rather than raising, because every caller is a
    CLI verb whose contract is an exit status. ``print_only`` is what
    ``lop operator install --print-only`` prints, unchanged from when this lived
    inline in that verb. ``sudo_secret`` is the INTERIM ask-sudo path (design
    §3.7/OQ15): the admin password is resolved in place from the secret store and
    fed to ``sudo -S`` on stdin for this one step, never printed or logged.

    ``purpose=PURPOSE_ADMIT_DEVICE`` REFUSES ``sudo_secret`` (F5). Admitting a
    device is the write whose whole value is that a HUMAN approved it at an OS
    admin prompt the agent cannot answer. A stored admin password is readable by
    any same-user process (``lop secret get``), so letting this write take one
    would let an agent admit its own device with nothing but files it can read —
    the presence proof would be the agent's own keystroke. Onboarding's installs
    (no ``purpose``) keep the interim path unchanged; narrowing those is the
    mesh lane's call, not this guard's.

    THE GUARD DOES NOT RELY ON THE CALLER'S ``purpose`` ALONE (R1-4): a future
    admit path that forgets the keyword would bypass it. So ``sudo_secret`` is
    ALSO refused whenever the staged anchor would admit a device the installed
    anchor keeps out (see :func:`_admitted_devices`) — the write is judged by
    what it does, not by what it says it is. A host with no usable installed
    anchor (first onboarding) has nothing to compare and is unchanged.

    THE BYTES INSTALLED ARE THE BYTES READ (the staging-swap race): the staged
    file is read once here, its sha256 travels in the privileged command's argv,
    and the privileged step installs only content with that digest — a swap of
    the user-writable staged file in between is refused, not installed.
    """
    if purpose == PURPOSE_ADMIT_DEVICE and sudo_secret:
        print(
            "confirming a phone needs an administrator to approve it in person, so a "
            "stored admin password cannot be used for it; nothing was installed",
            file=sys.stderr,
        )
        return 2
    staged = staging_path(root)
    target = anchor_path()
    refusal = trust_module.install_refusal()
    if refusal is not None:
        # R2-1: no root (or no PowerShell) means no command — print the reason,
        # never an elevated line whose ``$root`` is the drive root.
        print(
            f"the operator anchor cannot be installed on this computer: {refusal}; "
            "nothing was installed",
            file=sys.stderr,
        )
        return 1
    # Q3: ONE read. The device guard, the digest and the install all use these
    # bytes, so a swap of the user-writable staged file between two reads can no
    # longer show the guard a benign copy and install another.
    try:
        staged_bytes: bytes | None = staged.read_bytes()
    except OSError:
        staged_bytes = None
    if sudo_secret and staged_bytes is not None:
        installed = load_anchor()
        staged_anchor = trust_module.parse_staged_anchor(staged_bytes)
        if installed.usable and installed.anchor is not None and staged_anchor is not None:
            if trust_module.admission_enforced(
                installed.anchor
            ) and not trust_module.admission_enforced(staged_anchor):
                # Dropping the admission marker turns the rule off. It needs the
                # same in-person gesture as admitting a phone; otherwise a stored
                # password could quietly undo every confirmed list.
                print(
                    "turning off phone admission needs an administrator to approve it "
                    "in person, so a stored admin password cannot be used; nothing was installed",
                    file=sys.stderr,
                )
                return 2
            admitted = _admitted_devices(installed.anchor, staged_anchor)
            if admitted:
                print(
                    "this change confirms a phone ("
                    + ", ".join(admitted)
                    + "), which needs an administrator to approve it in person, so a "
                    "stored admin password cannot be used for it; nothing was installed",
                    file=sys.stderr,
                )
                return 2
    if staged_bytes is None:
        # R1-4 (§4's rule applied to the init-shaped refusals): the remedy names the
        # product action, not the terminal verb that performs it.
        print(
            f"nothing staged at {staged}; ask Local Operator to set up operator "
            "authority (the setup stages the key)",
            file=sys.stderr,
        )
        return 1
    command = install_commands(staged, target, hashlib.sha256(staged_bytes).hexdigest())
    if print_only:
        print(format_install_commands(command))
        return 0
    if _is_privileged():
        # Already root: write the bytes READ above rather than copying the path
        # again, so a swap after the read cannot change what lands.
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(staged_bytes)
        os.chmod(target, 0o644)
    else:
        # Windows always prints: the elevated step raises a UAC prompt, and lop
        # never starts one on the operator's behalf — they run it themselves.
        if trust_module._on_windows() or shutil.which("sudo") is None:
            print(
                "installing the anchor needs administrator rights. Run this by hand:\n  "
                + format_install_commands(command).replace(" && ", "\n  "),
                file=sys.stderr,
            )
            return 1
        if sudo_secret:
            return _install_with_sudo_stdin(command, sudo_secret)
        print(
            "installing the anchor as root — this is the one privileged step, and sudo "
            "will ask for your password:",
            file=sys.stderr,
        )
        for argv in command:
            result = subprocess.run(argv, check=False)
            if result.returncode != 0:
                return result.returncode
    loaded = load_anchor()
    print(f"anchor installed at {loaded.path} (trusted: {loaded.usable})")
    if not loaded.usable:
        print(f"  the runtime will NOT trust it: {loaded.reason}", file=sys.stderr)
        return 1
    return 0


def _resolve_secret_bytes(name: str) -> bytes | None:
    """Resolve a secret-store name to bytes, IN PLACE, in memory only.

    ``lop secret get`` is the documented in-place reader (``guide://credentials``);
    the CLI is located the way self-invocations in this tree locate it — ``which``
    first, else this interpreter running the same entry point — so a checkout's
    editable venv and an installed tool both work. Nothing here prints the value.
    """
    cli = shutil.which("lop")
    argv = (
        [cli]
        if cli
        else [
            sys.executable,
            "-c",
            "import sys; from local_operator.cli import main; sys.exit(main())",
        ]
    )
    try:
        proc = subprocess.run([*argv, "secret", "get", name], capture_output=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0 or not proc.stdout:
        return None
    return proc.stdout


def _install_with_sudo_stdin(command: list[list[str]], secret_name: str) -> int:
    """Feed a stored admin password to ``sudo -S`` for the ONE privileged step.

    The password is written only to the child's stdin and never printed, logged
    or receipted — the receipt names the secret, never a value. ``-p ''``
    suppresses sudo's own prompt text so a captured transcript cannot look like
    a fresh ask. It is re-fed per command because a sudo timestamp needs a
    terminal to be cached, and a terminal is exactly what this path lacks.
    """
    password = _resolve_secret_bytes(secret_name)
    if password is None:
        print(
            f"the admin password named by {secret_name} could not be resolved, so "
            "nothing was installed",
            file=sys.stderr,
        )
        return 1
    for argv in command:
        elevated = (
            [argv[0], "-S", "-p", "", *argv[1:]] if argv and argv[0] == "sudo" else list(argv)
        )
        result = subprocess.run(elevated, input=password + b"\n", check=False)
        if result.returncode != 0:
            return result.returncode
    return 0


def install_anchor_from(
    source: Path,
    root: Path,
    *,
    print_only: bool = False,
    sudo_secret: str = "",
    allow_revocation_lift: bool = False,
) -> int:
    """Install a statement RECEIVED from another machine (remote onboarding §3.3 step 7).

    ``allow_revocation_lift`` IS A TEST SEAM, NOT AN OPERATOR CONTROL. It exists so
    the plant's own cell can prove the allow-control behind R1 toggles exactly one
    thing; no product caller passes it (`git grep` finds only the test), and there is
    no verb, flag or surface that reaches it. Recorded here rather than left to be
    discovered, so a future caller does not reach for a bypass where the product
    route is `lop operator devices --authorise` followed by a fresh plant.

    THE POINT IS THE BYTES. The approval that authorises this install mints the
    digest of the exact statement the operator's machine holds (F4b), and the
    runner re-checks that digest before planting — so this path REFUSES anything
    that is not the canonical form: a re-serialised, reformatted or hand-edited
    file would land bytes nobody approved. The canonical form is
    :func:`anchor_bytes`, the same builder the staged install writes.
    """
    try:
        raw = source.read_bytes()
    except OSError:
        # DESIGN ROUND 1, D4: no Python errno text in the reader's sentence (the
        # verb has no ``--json`` detail channel; the action is the sentence).
        print(
            f"the anchor statement at {source} could not be read; ask for a fresh "
            "export from the machine that set it up",
            file=sys.stderr,
        )
        return 1
    try:
        anchor = OperatorAnchor.from_json(json.loads(raw.decode("utf-8")))
    except (UnicodeDecodeError, ValueError, TypeError):
        print(
            f"the anchor statement at {source} is not one this build recognises; "
            "nothing was installed — ask for a fresh export from the machine that "
            "set it up",
            file=sys.stderr,
        )
        return 1
    if anchor is None:
        # ``from_json`` refuses by returning None (it is weighted toward
        # refusal); name that case rather than crashing on it.
        print(
            f"the anchor statement at {source} is not one this build recognises; "
            "nothing was installed — ask for a fresh export from the machine that "
            "set it up",
            file=sys.stderr,
        )
        return 1
    if anchor_bytes(anchor) != raw:
        print(
            f"the anchor statement at {source} isn't the one the operator's computer "
            "exported (it looks edited or re-saved), so nothing was installed — ask "
            "for a fresh export",
            file=sys.stderr,
        )
        return 1
    installed = load_anchor()
    if installed.usable and installed.anchor is not None:
        lifted = _unrevoked_devices(installed.anchor, anchor)
        if lifted and not allow_revocation_lift:
            # A plant REPLACES the receiving list with the approved bytes, but it
            # must not resurrect a phone this computer revoked. Adding a phone is
            # the explicit control: re-list it locally before retrying the plant.
            names = ", ".join(str(item) for item in lifted)
            print(
                f"the incoming anchor would lift this computer's revocation of {names}; "
                "nothing was staged or installed. Allow that phone here first if you "
                "intend to trust it again",
                file=sys.stderr,
            )
            return 2
    # No merge: the statement's bytes are exactly what the sending operator
    # approved, so the receiving list is replaced rather than partly preserved.
    staged = stage_anchor(root, anchor)
    print(f"anchor statement accepted from {source} (staged at {staged})", file=sys.stderr)
    return install_anchor(root, print_only=print_only, sudo_secret=sudo_secret)


#: D13 (design round 2): "operator anchor" is build vocabulary; the person needs the
#: verb that says why.
_UNREADABLE_SETUP = (
    "this computer's operator setup could not be read, so nothing was confirmed. "
    "Run `lop operator status` to see why."
)


def _admission_install(
    root: Path,
    expected_old: bytes,
    *,
    print_only: bool,
    sudo_secret: str,
    runner: Any = None,
    announce: bool = False,
) -> int:
    """Install with both old and new digests pinned across the OS prompt.

    ``announce`` prints the gesture sentence on stdout just before the prompt. Only the
    CLI verb asks for it: the desktop apply route runs in a server whose card already
    renders the same sentence from ``confirm_card_copy``.
    """
    if sudo_secret:
        # `install_anchor` prints its own refusal and answers with its own codes; map
        # them onto THIS function's one-code-per-cause set, so a caller cannot read a
        # stored-password refusal as the migration guard or a cancelled prompt.
        return (
            ADMIT_OK
            if install_anchor(root, sudo_secret=sudo_secret, purpose=PURPOSE_ADMIT_DEVICE) == 0
            else ADMIT_INSTALL_FAILED
        )
    from local_operator.operator.elevate import run_elevated

    target = anchor_path()
    refusal = trust_module.install_refusal()
    if refusal is not None:
        # The same precheck `install_anchor` runs (R2-1), and it has to run HERE
        # too: `install_commands` raises ValueError on a Windows host with no
        # PowerShell, and that exception used to escape `admit_device` into the
        # desktop apply route as a 500 instead of the typed refusal (agent review
        # round 2, m5).
        print(
            f"the operator anchor cannot be installed on this computer: {refusal}; "
            "nothing was confirmed",
            file=sys.stderr,
        )
        return ADMIT_INSTALL_FAILED
    try:
        staged_bytes = trust_module.staging_path(root).read_bytes()
    except OSError:
        return ADMIT_INSTALL_FAILED
    command = trust_module.install_commands(
        trust_module.staging_path(root),
        target,
        hashlib.sha256(staged_bytes).hexdigest(),
        hashlib.sha256(expected_old).hexdigest(),
    )
    if print_only:
        print(format_install_commands(command))
        return ADMIT_OK
    if runner is None:
        from local_operator.operator.elevate import confirm_card_copy, prompt_refusal

        # THE PROMISE IS PRINTED ONLY WHERE IT IS KEPT (design round 2, D8). Every
        # check that can stop the prompt — revoked, unknown id, `install_refusal`,
        # idempotence above, and the host's ability to ask at all here — has run, so
        # "Your Mac will ask for your admin password." is never one line above the
        # sentence saying no prompt will appear, nor above a success that asked for
        # nothing. The words still come from `confirm_card_copy`, the one source the
        # card renders from.
        cannot = prompt_refusal()
        if cannot is not None:
            try:
                trust_module.staging_path(root).unlink()
            except OSError:
                pass
            print(cannot, file=sys.stderr)
            return ADMIT_NO_PROMPT
        if announce:
            print(confirm_card_copy()[2])
    result = run_elevated(command) if runner is None else runner(command)
    if result.cancelled or result.exit_code != 0:
        # S3 signs the certificate only after a zero result, so a failed prompt can
        # never reach certificate issuance. The staged statement goes on ANY
        # non-zero result, not only on a cancellation (agent review round 2, m4):
        # on macOS `osascript` exits 1 for a script error as well as a cancel, and
        # on pkexec/sh a real failure is not a cancel either — and a statement left
        # behind is exactly what the next plain `lop operator install` would apply
        # after an unrelated admin prompt.
        try:
            trust_module.staging_path(root).unlink()
        except OSError:
            pass
        # EVERY failed privileged step says something (design round 1, D2). The
        # card promises a gesture; a run in which none happened and nothing was
        # said is the one outcome a person cannot tell from success.
        if result.refusal:
            print(result.refusal, file=sys.stderr)
            return ADMIT_NO_PROMPT
        # NO CLAIM ABOUT WHAT THE PHONE CAN DO NOW (design round 2, D10): after a
        # cancelled migration `--confirm` the pre-rule phones keep working, and on a
        # presence host an unlisted phone is admitted — so the sentence names only
        # the fact and the retry.
        if result.cancelled:
            print(
                "You cancelled the approval, so nothing was confirmed. Run the same "
                "command again when you are ready.",
                file=sys.stderr,
            )
            return ADMIT_CANCELLED
        # The script's own reason first, then the retry (agent review round 4, m3).
        if result.diagnosis:
            print(result.diagnosis, file=sys.stderr)
        print(
            "The approval did not complete, so nothing was confirmed. Run the same "
            "command again to retry.",
            file=sys.stderr,
        )
        return ADMIT_INSTALL_FAILED
    try:
        installed_raw = target.read_bytes()
        loaded = trust_module.OperatorAnchor.from_json(json.loads(installed_raw.decode("utf-8")))
    except (OSError, UnicodeDecodeError, ValueError):
        return ADMIT_INSTALL_FAILED
    return (
        ADMIT_OK
        if loaded is not None and trust_module.anchor_bytes(loaded) == staged_bytes
        else ADMIT_INSTALL_FAILED
    )


def _admission_anchor(installed: Any, ids: list[str]) -> Any:
    """Installed statement plus marker and selected ids; never a fresh anchor.

    A REVOCATION IS NOT LIFTED BY AN ADMISSION (agent review round 2, M1). This
    builder used to strip the entry and append a bare ``{"device_id": id}``, so
    admitting a phone the installed statement revokes REWROTE the revocation — a
    phone revoked while its pairing card was open came back admitted, signed,
    installed and honoured. Both callers now refuse a revoked phone before they
    get here (``--authorise`` is the one route back); this is the second lock, so
    no future caller can lift a revocation by admitting.
    """
    revoked = trust_module.revoked_device_ids(installed)
    entries = [dict(entry) for entry in installed.devices]
    for device_id in ids:
        if device_id in revoked:
            continue
        entries = [entry for entry in entries if entry.get("device_id") != device_id]
        entries.append({"device_id": device_id})
    return replace(
        installed,
        admission=trust_module.ADMISSION_LISTED,
        devices=tuple(entries),
    )


def _installed_admission_anchor() -> tuple[Any, bytes] | None:
    """The canonical installed statement, only after its ownership gate passes."""
    loaded = load_anchor()
    try:
        raw = anchor_path().read_bytes()
        installed = trust_module.OperatorAnchor.from_json(json.loads(raw.decode("utf-8")))
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    if (
        not loaded.usable
        or installed is None
        or loaded.anchor != installed
        or trust_module.anchor_bytes(installed) != raw
    ):
        return None
    return installed, raw


def _revoked_selection(root: Path, ids: list[str]) -> list[str]:
    """Which of ``ids`` this computer has revoked — the ANCHOR's list or the record's.

    ``devices.is_revoked`` is the product predicate (both lists, in the order that
    costs least when the answer is yes), so ``admit_device`` (one id) and
    ``confirm_devices`` (a selection) cannot disagree about the rule — and neither
    can miss the revocation a `--revoke --print-only` left in the local record
    before the privileged anchor write landed. Separate from the refusal sentence so
    the words live in one place.
    """
    from local_operator.operator.devices import is_revoked

    return [device_id for device_id in ids if is_revoked(root, device_id)]


def _revoked_in(installed: Any, ids: list[str]) -> list[str]:
    """Which of ``ids`` the statement just read revokes — independent of ``devices``.

    THE SECOND LOCK REFUSES, NOT ONLY REFRAINS (agent review round 3, m3).
    ``_admission_anchor`` leaves a revoked entry's bytes alone, so for a lone revoked
    id the proposal equalled the installed bytes and the idempotence branch answered
    ``ADMIT_OK`` — and ``pair_op`` signs on ``ADMIT_OK``. Reading the revocation off
    the SAME statement the proposal is built from means a caller that skips (or a
    future change that weakens) ``_revoked_selection`` still gets a refusal.
    """
    revoked = trust_module.revoked_device_ids(installed)
    return [device_id for device_id in ids if device_id in revoked]


def _print_revoked_refusal(ids: list[str]) -> None:
    """The one sentence for "this phone was removed on this computer".

    NAMES THE ROUTE BACK, because a revoked phone can never be admitted by
    confirming it: ``--authorise`` clears the entry, and only then does a pairing
    work. Saying "confirm the phone" here would be an instruction that cannot
    work (agent review round 2, m1/m2).
    """
    from local_operator.operator.devices import AUTHORISE_COMMAND

    names = ", ".join(ids)
    print(
        f"this phone was removed on this computer, so it cannot be confirmed here: "
        f"{names}. Bring it back first with "
        f"`{AUTHORISE_COMMAND.format(device_id=ids[0])}` (this asks for your "
        "approval), then pair the phone again; nothing was installed",
        file=sys.stderr,
    )


def confirm_devices(
    root: Path,
    ids: list[str],
    *,
    print_only: bool = False,
    sudo_secret: str = "",
    runner: Any = None,
) -> int:
    """Confirm the human's chosen phones and end the pre-rule migration state."""
    from local_operator.operator.devices import list_devices

    current = _installed_admission_anchor()
    if current is None:
        # Never derive an admission from the user-writable staged statement.
        print(_UNREADABLE_SETUP, file=sys.stderr)
        return ADMIT_NO_INSTALLED_ANCHOR
    installed, old = current
    known = {device.device_id for device in list_devices(root)}
    selected = list(dict.fromkeys(ids))
    if not selected or any(device_id not in known for device_id in selected):
        print(
            "choose one or more paired phones to confirm; nothing was installed",
            file=sys.stderr,
        )
        return ADMIT_NO_SELECTION
    revoked = _revoked_selection(root, selected) or _revoked_in(installed, selected)
    if revoked:
        _print_revoked_refusal(revoked)
        return ADMIT_REVOKED
    proposed = _admission_anchor(installed, selected)
    if trust_module.anchor_bytes(proposed) == old:
        return ADMIT_OK  # Idempotence: an already confirmed selection needs no prompt.
    stage_anchor(root, proposed)
    return _admission_install(
        root,
        old,
        print_only=print_only,
        sudo_secret=sudo_secret,
        runner=runner,
        announce=True,
    )


def admit_device(
    root: Path,
    device_id: str,
    *,
    print_only: bool = False,
    sudo_secret: str = "",
    runner: Any = None,
) -> int:
    """Confirm exactly one new phone before S3 is allowed to issue its cert.

    A phone this computer revoked is refused FIRST (agent review round 2, M1): a
    revocation that lands while a pairing card is open used to be rewritten by the
    admission that followed, so the phone was signed, installed and honoured.
    """
    from local_operator.operator.devices import list_devices

    current = _installed_admission_anchor()
    if current is None:
        print(_UNREADABLE_SETUP, file=sys.stderr)
        return ADMIT_NO_INSTALLED_ANCHOR
    installed, old = current
    if _revoked_selection(root, [device_id]) or _revoked_in(installed, [device_id]):
        # A REVOCATION OUTRANKS ADMISSION (agent review round 2, M1). Without this
        # the admission REWROTE the revocation a moment later and the phone was
        # signed, installed and honoured — a phone revoked while its card was open
        # came back by finishing that card. `--authorise` is the one route back.
        _print_revoked_refusal([device_id])
        return ADMIT_REVOKED
    if not trust_module.admission_enforced(installed) and any(
        device.device_id not in trust_module.open_device_ids(installed)
        for device in list_devices(root)
    ):
        print(
            "confirm your existing phones before adding another; adding one must not "
            "un-confirm the phones you already paired",
            file=sys.stderr,
        )
        return ADMIT_MIGRATION_GUARD
    if device_id not in {device.device_id for device in list_devices(root)}:
        # A first-time phone has no stored cert until S3 signs it, AFTER this
        # write. Its pending request, not the cert store, is what validates it.
        from local_operator.operator.devices import list_pending

        if device_id not in {str(row.get("device_id") or "") for row in list_pending(root)}:
            print(
                "the phone has no pending pairing request; nothing was confirmed — "
                "start the pairing again from the phone",
                file=sys.stderr,
            )
            return ADMIT_NO_PENDING_REQUEST
    proposed = _admission_anchor(installed, [device_id])
    if trust_module.anchor_bytes(proposed) == old:
        return ADMIT_OK
    stage_anchor(root, proposed)
    return _admission_install(
        root,
        old,
        print_only=print_only,
        sudo_secret=sudo_secret,
        runner=runner,
    )


def _existing_key(root: Path, preference: str) -> KeyHandle | None:
    """The operator key already in this host's store, as a handle, or ``None``.

    A HANDLE, NOT A SIGNER — the contract its callers already read it under, and the
    one this function's docstring claimed before it did. ``backend.load()`` returns a
    ``Signer | None`` (the loaded private half, with the handle at ``signer.handle``),
    while every consumer of this probe wants the public handle: ``_report_existing_key``
    and ``_setup`` dereference ``.key_id`` / ``.spki`` and print a level, and
    ``_setup`` uses the result as the ``anchor_for_handle`` input. Unwrapping and
    closing HERE is what stopped the two halves of ``_setup`` disagreeing about the
    type — the resume path (key present, anchor not installed) and the already-installed
    short-circuit both died on ``'_KeyagentSigner' object has no attribute 'key_id'``.
    Closing is safe: it is a no-op on the key-agent signer and only drops a reference on
    the software signer, and the handle is inert public material while the key stays on
    disk.

    BEST EFFORT, deliberately: a store that raises on a probe (a locked presence
    store, a platform whose load path is unimplemented) reports ``None`` here and
    lets ``create`` produce its own diagnosis, which is the message an operator can
    act on. Swallowing the probe's error into a sentence of its own would be a
    second, worse explanation of the same state (R9-2/Q9-2).
    """
    from local_operator.operator.keychain import choose_backend

    try:
        backend = choose_backend(preference, config_root=root)
        signer = backend.load()
    except (OSError, KeyBackendError):
        return None
    if signer is None:
        return None
    handle = signer.handle
    signer.close()
    return handle


def _print_failure(headline: str, exc: Exception) -> None:
    """Print ``headline`` and a backend failure, on the shape the message is written in.

    SOME OF THESE MESSAGES ARE BLOCKS, not sentences: a key-agent state carries an
    aligned ``label : value`` remedy block (design round 1, D4), and a ``refused``
    key-generation message has carried one line per diagnosis since round 1. Joining
    either to the headline with ``": "`` put the first remedy at the end of a 200+
    character line, so a block gets the headline on its own line and keeps its
    indentation, while a one-sentence message (a cancelled prompt, a missing key) stays
    where it reads best — on the headline's line.
    """
    text = str(exc)
    if "\n" in text:
        print(headline, file=sys.stderr)
        print(text, file=sys.stderr)
    else:
        print(f"{headline}: {text}", file=sys.stderr)


def _report_existing_key(root: Path, handle: KeyHandle, label: str) -> int:
    """Report the operator key already in this host's store, and complete what is missing.

    IDEMPOTENT RATHER THAN FATAL (R9-2/Q9-2). Measured: a second ``init`` on a file-only
    host died on an uncaught ``FileExistsError`` from the O_EXCL create, printing a
    traceback and nothing an operator could act on — and on a presence host the same
    second run is a duplicate-item create. Neither the key nor the anchor may be silently
    replaced (a new anchor invalidates every device certificate signed under the old
    one), so ``init`` reports the state it found and completes only the part that is
    missing.

    Reached from two places, which is why it is a function: the probe at the top of
    ``_init``, and a second probe in the branch where the create came back ``reused``
    — the race where a key appeared between the probe and the create (agent review
    round 1, R1-5). Both call sites pass an ``_existing_key`` result, never a signer.
    """
    staged = staging_path(root)
    loaded = load_anchor()
    staged_now = False
    if not staged.exists() and not loaded.exists:
        # New statements are born enforcing; only an existing pre-rule installed
        # file gets the migration guard. Constructing from this installed/created
        # statement preserves the key and backend rather than minting a new key.
        staged = stage_anchor(
            root,
            replace(
                anchor_for_handle(handle, label=label),
                admission=trust_module.ADMISSION_LISTED,
            ),
        )
        staged_now = True
    print(f"operator key already exists in the {handle.backend} store; nothing replaced")
    print(f"  key id : {handle.key_id}")
    print(f"  level  : {describe_level(handle)}")
    if staged_now:
        print(f"  staged : {staged} (the anchor statement was missing, so it was written)")
    elif staged.exists():
        print(f"  staged : {staged}")
    else:
        # EVERY OTHER STATE PRINTS A VALUE (UX round 10, U4): with the anchor
        # installed and no carrier on disk this slot was simply absent, which reads
        # as "not reported" rather than as "there is none". Nothing is wrong with
        # that state — `--revoke` and `--authorise` re-stage on demand — so it says
        # so instead of leaving a gap.
        print(f"  staged : (none at {staged} — `--revoke` or `--authorise` re-stages)")
    if loaded.exists:
        print(f"  anchor : {loaded.path} (installed: {loaded.usable})")
    print()
    from local_operator.operator.devices import AUTHORISE_COMMAND

    if handle.backend == FILE_ONLY:
        # NAME THE FILE, NOT "that store" (design round 1, D5): the file-only store IS a
        # path under the config dir, and the copy used to send an operator looking for it
        # without saying where. Read from the handle's own definition rather than written
        # as a literal, so a moved store moves the sentence with it.
        from local_operator.operator.keychain import default_file_path

        print(
            f"To replace this key, remove {default_file_path(root)} — this host's "
            "file-only store is that 0600 file — yourself first; a new anchor "
            "invalidates every paired phone."
        )
    else:
        print("To replace this key, remove it from that store yourself first — a new")
        print("anchor invalidates every paired phone.")
    print("To lift a revocation instead:")
    print(f"  {AUTHORISE_COMMAND.format(device_id='<device id>')}")
    return 0


def _init(args: argparse.Namespace) -> int:
    root = config_dir()
    existing = _existing_key(root, args.backend)
    if existing is not None:
        return _report_existing_key(root, existing, args.label)
    try:
        handle = create_key(config_root=root, preference=args.backend)
    except KeyBackendError as exc:
        _print_failure("could not create the operator key", exc)
        return 1
    except FileExistsError as exc:
        # The race the probe above cannot close: something created the key between
        # the check and the create. Named rather than raised, because the operator's
        # next step is the same either way (R9-2/Q9-2).
        print(
            f"an operator key already exists at {exc.filename or root} — nothing was "
            "replaced. `lop operator init` is idempotent: run it again to see what is "
            "there, and `lop operator trust` to see whether the runtime honours it.",
            file=sys.stderr,
        )
        return 1
    if handle.reused:
        # THE EXPLICIT-REUSE HALF OF THE SAME RACE (agent review round 1, R1-5). The probe
        # above could not see a key that appeared in the microseconds before the helper
        # ran, and on the presence path the helper now answers that case with the tag's
        # key and ``reused:true`` rather than with a second key. Reporting "created" there
        # would be the one thing this verb's idempotence exists to prevent — a report that
        # does not match what is on the machine — so the same block a second `init` prints
        # is printed here, and the anchor staged below is the one for the key it names.
        again = _existing_key(root, args.backend)
        if again is not None:
            return _report_existing_key(root, again, args.label)
    anchor = replace(
        anchor_for_handle(handle, label=args.label),
        admission=trust_module.ADMISSION_LISTED,
    )
    staged = stage_anchor(root, anchor)

    print(f"operator key created in the {handle.backend} store")
    print(f"  key id : {handle.key_id}")
    print(f"  level  : {describe_level(handle)}")
    print(f"  staged : {staged}")
    print()
    print("The runtime trusts ONLY the root-owned anchor, and landing it is one privileged")
    print("step: set up operator authority for this machine (one approval and one admin")
    print("password prompt). If you are working from this terminal, the install verb below")
    print("is the same step and keeps its own sudo prompt:")
    print("  lop operator install")
    if handle.backend == "file-only":
        print()
        print(
            "WARNING: this install cannot reach a presence store, so loosening is "
            "authorised by a key any process running as you can read. That is reported "
            "as a lower level and is NOT a boundary.",
            file=sys.stderr,
        )
    return 0


def _install(args: argparse.Namespace) -> int:
    from_file = str(getattr(args, "from_file", "") or "")
    if from_file:
        return install_anchor_from(Path(from_file), config_dir(), print_only=bool(args.print_only))
    return install_anchor(config_dir(), print_only=bool(args.print_only))


def _anchor(args: argparse.Namespace) -> int:
    """``lop operator anchor export [--file P] [--json]`` — the PUBLIC statement.

    THE SOURCE IS THE LOCAL STATEMENT, in trust order: the STAGED statement
    first (it is the one `init`/`setup` produced and the one install lands), the
    INSTALLED anchor as the fallback so a machine whose staging area was cleaned
    can still hand its key to a peer. It never invents a statement: a machine
    with no operator key says so and points at the product action that creates
    one, and the sentence names a product action, not a terminal command.
    """
    if getattr(args, "anchor_command", None) != "export":
        print("usage: lop operator anchor export [--file PATH] [--json]", file=sys.stderr)
        return 2
    root = config_dir()
    anchor = load_staged_anchor(root)
    loaded = load_anchor()
    if anchor is None:
        anchor = loaded.anchor
    if anchor is None:
        staged_file = staging_path(root)
        if staged_file.exists() or loaded.exists:
            # WHICH statement is present-but-unusable, and ONE remedy (F5 review,
            # D1/D2/R-M1): the staged file counts too — export prefers to hand
            # over the artifact ``init``/``setup`` produced, so "no key yet" must
            # not be printed while one is sitting on disk — and the key-id
            # refusal's reason is designed standalone and already ends in the
            # product action, so export must not stack its own on top of it (that
            # composition ran 464 characters with the action twice). The artifact
            # and the cost clause match the no-key branch (§2.9: no terminal
            # commands in copy).
            path = staged_file if staged_file.exists() else loaded.path
            reason = "" if staged_file.exists() else loaded.reason
            remedy = (
                " — ask Local Operator to set up operator authority again for this "
                "machine (one approval and one admin password prompt)"
            )
            suffix = "" if "set up operator authority" in reason else remedy
            print(
                f"the operator key at {path} exists but is not usable"
                + (f" ({reason})" if reason else "")
                + f", so there is nothing to hand to another machine{suffix}",
                file=sys.stderr,
            )
            return 1
        print(
            "this machine has no operator key yet, so there is nothing to hand to "
            "another machine — ask Local Operator to set it up for you (one approval "
            "and one admin password prompt)",
            file=sys.stderr,
        )
        return 1
    statement = anchor_bytes(anchor)
    target = str(getattr(args, "file", "") or "")
    written = ""
    if target:
        path = Path(target).expanduser()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(statement)
            os.chmod(path, 0o644)
        except OSError as exc:
            print(f"the statement could not be written to {path} ({exc})", file=sys.stderr)
            return 1
        written = str(path)
    payload: dict[str, Any] = {
        "ok": True,
        "path": written,
        "key_id": key_id_for(anchor.spki),
        "spki_fp": spki_fp(anchor.spki),
    }
    if not target:
        # NO FILE MEANS STDOUT IS THE TRANSFER FORM, so the bytes ride the stream
        # (human mode) and the payload too (--json): a --json caller without a
        # file has nowhere else to read the statement from.
        payload["statement"] = statement.decode("utf-8")
    if bool(getattr(args, "json", False)):
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if target:
        print(f"anchor statement written to {written}")
        print(f"  key id     : {payload['key_id']}")
        # N1 (design round 1): spki_fp is THE value a human compares, so it is
        # labelled as a fingerprint and the comparison is named.
        print(
            f"  fingerprint: {payload['spki_fp']}  (the value to compare before "
            "this key is set up anywhere else)"
        )
    else:
        sys.stdout.write(statement.decode("utf-8"))
    return 0


def _setup(args: argparse.Namespace) -> int:
    """The agent-runnable self-install for THIS machine (design §3.7).

    RECEIPTS, IN THE CARD'S OWN VOCABULARY: ``proposed → consent → generated →
    installed → verified``. The privileged write is the SAME one root-owned file
    of public data ``init``/``install`` already land; what this verb adds is the
    agent-usable shape:

    * no step requires the reader to run a command — the consent gesture is the
      host's own (the sudo prompt here; the desktop sheet is slice (c)), and an
      admin password CAN be supplied once through the credential store for the
      one privileged step (``--sudo-secret``), never printed or logged;
    * a refused or failed gesture leaves the machine exactly as it was, with the
      outcome recorded on ``installed`` and the state reading not-installed —
      the "Finish setup" affordance, never a dead end;
    * the level is reported AS ACHIEVED, so a host whose key is ``file-only`` is
      told that rather than promised a presence check it cannot deliver.
    """
    root = config_dir()
    receipts: list[dict[str, Any]] = []

    def receipt(step: str, ok: bool, detail: str, **data: Any) -> None:
        row: dict[str, Any] = {"step": step, "ok": ok, "detail": detail, "at": time.time()}
        row.update(data)
        receipts.append(row)

    def finish(code: int, **extra: Any) -> int:
        report = operator_authority_report()
        payload: dict[str, Any] = {"ok": code == 0, "receipts": receipts, **extra}
        payload.setdefault("level", report["level"])
        if bool(getattr(args, "json", False)):
            print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        else:
            for row in receipts:
                mark = "ok" if row["ok"] else "blocked"
                print(f"[{mark}] {row['step']}: {row['detail']}")
            print(f"level: {payload.get('level')}")
        return code

    receipt(
        "proposed",
        True,
        # D5 (design round 1): #1877 settled this vocabulary — the same step is
        # "set up operator authority", not the engineer's account of the files.
        "set up operator authority for this machine — one approval and one admin "
        "password prompt",
    )
    loaded = load_anchor()
    existing = _existing_key(root, "auto")
    if existing is not None and loaded.usable:
        receipt("consent", True, "nothing to raise: the anchor is already installed")
        receipt(
            "generated",
            True,
            f"operator key already exists ({existing.key_id}); nothing replaced",
        )
        receipt("installed", True, f"anchor already at {loaded.path} (root-owned)")
        receipt("verified", True, describe_level(existing, with_remedy=False))
        return finish(0, state="installed", anchor=str(loaded.path))

    sudo_secret = str(getattr(args, "sudo_secret", "") or "")
    # THE ADMIN ROUTE IS COMPUTED ONCE, BEFORE THE CONSENT ROW: it decides both
    # the row's MARK and whether the privileged step has any way to run at all
    # (design round 1, D1 — the mark must match the sentence).
    if _is_privileged():
        admin_route = "privileged"
    elif sudo_secret:
        admin_route = "ask-once"
    elif shutil.which("sudo") is not None:
        admin_route = "prompt"
    else:
        admin_route = "unavailable"
    if admin_route == "privileged":
        consent_line = "this process already holds administrator rights"
    elif admin_route == "ask-once":
        consent_line = (
            f"the admin password is used once, from the secret named {sudo_secret} "
            "(never printed or logged)"
        )
    elif admin_route == "prompt":
        consent_line = "the admin password will be asked for once, at the install step"
    else:
        consent_line = (
            "this machine needs an admin approval for the anchor, and no "
            "administrator tool Local Operator can use is available here"
        )
    receipt("consent", admin_route != "unavailable", consent_line)

    try:
        # A probed handle is used as-is rather than re-created: the key belongs to a
        # previous run and stays on disk, and the staged statement below carries
        # everything this verb needs — so there is nothing to release here.
        handle = (
            existing if existing is not None else create_key(config_root=root, preference="auto")
        )
    except (KeyBackendError, FileExistsError) as exc:
        receipt("generated", False, f"the operator key could not be created: {exc}")
        return finish(1, state="not_installed")
    anchor = anchor_for_handle(handle, label="")
    staged = stage_anchor(root, anchor)
    receipt(
        "generated",
        True,
        f"operator key ready ({handle.key_id}); anchor statement staged at {staged}",
    )

    if admin_route == "unavailable":
        # D1: setup never shows the direct verb's by-hand block (it cannot be
        # followed where sudo is absent, and setup is not the documentation
        # register). The reason is a receipt, in product words.
        receipt(
            "installed",
            False,
            "this machine has no administrator tool Local Operator can use, so "
            "nothing was installed. The key exists but is not trusted yet — ask "
            "Local Operator to finish setup where an admin approval is possible.",
        )
        return finish(1, state="not_installed")
    rc = install_anchor(root, print_only=False, sudo_secret=sudo_secret)
    if rc != 0:
        receipt(
            "installed",
            False,
            "the admin approval was not completed, so the anchor is not installed "
            "yet. The key exists but is not trusted until setup finishes — ask "
            "Local Operator to run setup again when an admin approval is possible.",
        )
        return finish(rc, state="not_installed")
    receipt("installed", True, f"anchor landed root-owned at {anchor_path()}")

    loaded = load_anchor()
    verified = bool(loaded.usable)
    level_line = describe_level(handle, with_remedy=False)
    receipt(
        "verified",
        verified,
        # D5: the level is REPEATED AS ACHIEVED, in the same sentence `lop operator
        # status` prints — a file-only host is told what that means. The remedy half
        # is dropped here because it names a terminal command and this is a receipt
        # surface (§2.9); the status/init registers keep the full sentence.
        level_line if verified else f"{level_line.rstrip('.')}; {loaded.reason}",
    )
    return finish(0 if verified else 1, state="installed" if verified else "not_installed")


def _trust() -> int:
    loaded = load_anchor()
    report = operator_authority_report()
    print(f"anchor    : {loaded.path}")
    print(f"installed : {loaded.exists}")
    print(f"trusted   : {loaded.usable}")
    if loaded.anchor:
        print(f"key id    : {loaded.anchor.key_id}")
        print(f"backend   : {loaded.anchor.backend} (presence: {loaded.anchor.presence})")
    if not loaded.usable:
        print(f"reason    : {loaded.reason}", file=sys.stderr)
    print(f"level     : {report['level']}")
    return 0 if loaded.usable else 1


def _keyagent_state(backend: str) -> tuple[bool, str] | None:
    """The macOS key agent's state, or ``None`` when it is not this host's private half.

    Asked ONLY where it can matter — on macOS, and only when the recorded backend is the
    presence store or nothing is recorded at all — so a Linux or Windows report and a
    ``file-only`` macOS report are unchanged. This is the one place the runtime asks the
    key agent about ITSELF rather than about a key, and it is a diagnostic verb rather
    than a frame: two short subprocesses (a pre-flight and a ``doctor``) on the command an
    operator runs when something is already wrong.
    """
    if sys.platform != "darwin" or backend not in ("", SECURE_ENCLAVE):
        return None
    from local_operator.operator.keychain import SecureEnclaveBackend

    return SecureEnclaveBackend().health()


def _status() -> int:
    report = operator_authority_report()
    staged = staging_path(config_dir())
    staged_here = staged.exists() and not report["anchor_installed"]
    print(f"level                  : {report['level']}")
    print(f"anchor path            : {report['anchor_path']}")
    print(f"anchor installed       : {report['anchor_installed']}")
    print(f"anchor root-owned      : {report['anchor_root_owned']}")
    backend_field = report["backend"] or "(none)"
    if staged_here and not report["backend"]:
        # A BARE ``(none)`` READS AS "MY KEY VANISHED" ONE COMMAND AFTER `init` REPORTED
        # CREATING IT (design round 1, D1). The authority report only ever describes the
        # INSTALLED anchor, so a staged-but-uninstalled key — the state every macOS user
        # is in until `lop operator install` succeeds, and the state this PR's own
        # fallback advice lands them in — has no backend to name. The staged anchor does:
        # it says a key exists and what the one pending step is.
        backend_field = "(none — an operator key is staged; the install step has not run yet)"
    print(f"private-half backend   : {backend_field}")
    print(f"presence per signature : {report['presence_enforced_by_os']}")
    # THE RULE, STATED WHERE AN OPERATOR WILL READ IT. One sentence shared with
    # `lop operator devices` (`trust.admission_sentence`), because two hand-written
    # copies of one rule drifted exactly where nobody could check them: both claimed
    # a list requirement that `anchor.presence` exempts before the marker is read, so
    # a Secure-Enclave host would have honoured an unlisted phone while both surfaces
    # said it would not (agent review round 1, m1). The anchor is read once here for
    # both facts; the report above describes the LEVEL, which is a different question.
    loaded_anchor = trust_module.load_anchor()
    anchor = loaded_anchor.anchor if loaded_anchor.usable else None
    print(f"phone admission        : {trust_module.admission_sentence(anchor)}")
    print(f"spawn-capable guarantee: {report['capability_guarantee']}")

    # THE KEY AGENT'S FAULT STANDS BESIDE THE AUTHORITY REASON, NEVER IN PLACE OF IT
    # (design round 1, D1). It used to REPLACE the reason on the "nothing recorded"
    # branch, which collapsed two independent facts into one field: the level is
    # `spawn-capability-only` because NO ANCHOR IS INSTALLED, and the key agent's state
    # does not move the level at all — while `reason: … (broken install)` one command after
    # a successful `init` reads as "my key vanished and my installation is broken", and the
    # remedy a reader derives from it (reinstall) neither installs the anchor nor changes
    # the level.
    agent = _keyagent_state(report["backend"])
    fault: str | None = None
    if agent is not None:
        ok, note = agent
        if not ok:
            fault = note
            print(f"key agent              : {note}")
            # D3: `status`'s ONLY named remedy in this state used to be `lop operator
            # init`, which in exactly this state exits 1 — a closed loop whose working
            # remedy appeared only in the failure message of the command it sent the
            # reader to. The fix line names the reinstall, and the file-only alternative
            # only where there is nothing staged yet to fall back on.
            #
            # AND THE ALTERNATIVE GETS ITS OWN LABEL (design round 2, D6). Appending it to
            # this line made it 166 characters — the widest field on the surface, three
            # display lines at 80 columns, with the remedy a reader can act on IMMEDIATELY
            # arriving last. The `init` refusal block already carries one labelled field
            # per remedy (`fix  :` / `or   :` / `note :`), so `status` uses the same shape
            # and the alternative is reachable by label rather than by reading to the end
            # of a wrapped line.
            print(
                "fix                    : reinstall the macOS wheel — "
                "`uv tool install local-operator --force`"
            )
            if not staged_here:
                print(
                    "or                     : take a file-backed key now: "
                    "`lop operator init --backend file-only`"
                )
    print(f"reason                 : {report['reason']}")
    # THE ONE STEP THAT IS STILL PENDING, named rather than left for the reader to
    # infer from a level that looks lower than what `init` just printed. The two
    # lines are about different things on purpose: `init` reports the KEY it
    # created, this reports what the RUNTIME will honour, and between them sits
    # exactly one privileged command.
    if staged_here:
        print(f"staged anchor          : {staged} (not trusted yet — the install step has not run)")
    if report["level"] == LEVEL_OPERATOR_PRESENCE:
        print("loosening: authorised by a signature that costs a human gesture")
    elif report["level"] == LEVEL_OPERATOR_FILE_ONLY:
        print("loosening: authorised by a key ANY process running as you can read — not a boundary")
    elif report["level"] == LEVEL_ANCHOR_UNPINNED:
        print("loosening: refused — a file sits where the anchor belongs and is not root-owned")
    elif report["level"] == LEVEL_SPAWN_ONLY:
        # TRUE OF THIS LEVEL AND ONLY THIS LEVEL, which is why it is worded as the
        # state rather than the rule (agent review round 9, R9-1): with no operator
        # authority on the host, the spawn capability really is the only source a
        # running runtime will accept. The second sentence names the way out, because a
        # reader in this state meets no working lever anywhere else — EXCEPT when the key
        # agent is broken, where `lop operator init` refuses and the `fix` line above is
        # the way out (design round 1, D3).
        #
        # ONE SENTENCE PER print(), NO MANUAL INDENT (design round 10, D3). The first
        # version hand-wrapped inside a single string with an 11-space continuation
        # indent, and below ~79 columns the terminal soft-wrapped those lines AGAIN,
        # stranding the indent mid-answer and breaking words across lines — visible at
        # 60 and 44 columns. Every other field in this report is one physical line and
        # lets the terminal do the wrapping; this one now behaves the same way.
        print("loosening: no operator authority on this host.")
        if fault is None:
            print(
                "Only the process that started the session can loosen it; `lop operator init` "
                "adds the operator key that lets a gesture or a paired phone do it."
            )
    _print_paired_devices()
    return 0


def _print_paired_devices() -> None:
    """The phones this machine knows, from the store ``lop operator devices`` reads.

    On ``status`` because the level ABOVE says how strong authority is on this
    host and says nothing about WHICH devices hold it. An operator deciding
    whether to revoke a phone should not have to know a second verb exists.

    REVOKED DEVICES ARE PRINTED TOO (UX round 10, U1). They were dropped from both
    listings: the revocation clears the certificate, so `paired` cannot hold one, and
    nothing printed the anchor-only revocations — which left the id the remedy names
    unreadable anywhere but a receipt from an earlier day.

    UNDER ENFORCEMENT EACH PHONE SAYS WHETHER THE COMPUTER HAS CONFIRMED IT. A phone
    whose certificate this machine holds but whose id the anchor does not list is
    refused by the runtime, and without this the listing above would name it as a
    phone that can act — the one reading a human would take from a line of ids. The
    marker appears only on hosts that enforce, because on a host that does not, an
    unlisted phone IS admitted and calling it "not confirmed" would describe a
    boundary the host does not apply.
    """
    from local_operator.operator.devices import (
        AUTHORISE_COMMAND,
        list_devices,
        revoked_ids,
    )
    from local_operator.operator.trust import (
        admission_enforced,
        load_anchor,
        open_device_ids,
    )

    paired = list_devices(config_dir())
    revoked = revoked_ids(config_dir())
    loaded = load_anchor()
    anchor = loaded.anchor if loaded.usable else None
    enforcing = bool(anchor is not None and admission_enforced(anchor))
    # A PRESENCE HOST EXEMPTS THE LIST BEFORE THE MARKER IS READ
    # (`trust.device_is_admitted`): the key asks for a gesture per signature there,
    # so an unlisted phone IS admitted and the marker below would be a boundary this
    # host does not apply (agent review round 1, m1).
    presence = bool(anchor is not None and anchor.presence)
    listed = open_device_ids(anchor) if anchor is not None else []
    if paired:
        print(f"paired devices         : {len(paired)}")
        for device in paired:
            # Under enforcement a phone this computer holds a certificate for but does
            # not list is REFUSED by the runtime, so the row has to say so — otherwise
            # a line of ids reads as a list of phones that can act.
            admitted = not enforcing or presence or device.device_id in listed
            pending = "" if admitted else "  (not confirmed on this computer)"
            print(f"  - {device.device_id}  {device.name or '(unnamed)'}{pending}")
    if revoked:
        print(f"revoked devices        : {len(revoked)}")
        for device_id in revoked:
            print(
                f"  - {device_id}  (revoked — "
                f"`{AUTHORISE_COMMAND.format(device_id=device_id)}` brings this device "
                "back, then pair the phone again)"
            )


def _sign(args: argparse.Namespace) -> int:
    root = config_dir()
    # THE SHEET A PERSON ANSWERS NAMES THE SESSION AND THE EFFECT — on the channel
    # this verb has for a human (UX round 6, U3 = design round 6, D3). stdout is the
    # value and nothing else (see the module docstring), so this goes to stderr.
    #
    # This is also where the design's mitigation for its prompt-misread residual
    # actually lands for a CLI caller: the OS sheet raised by the presence backend
    # carries the OS's own wording, because `SecKeyCreateSignature` takes no
    # parameters dictionary and `kSecUseOperationPrompt` was deprecated in macOS 11
    # (see `operator/keychain._KeyagentSigner`). Printing the sentence
    # before the gesture is what makes the human's decision an informed one.
    from local_operator.operator.sign import effect_copy

    print(
        effect_copy(
            purpose=args.purpose,
            session_id=getattr(args, "session", "") or "",
            request_id=getattr(args, "request_id", "") or "",
        ),
        file=sys.stderr,
    )
    try:
        signature = sign_challenge(
            challenge=args.challenge,
            purpose=args.purpose,
            config_root=root,
            session_id=args.session,
            request_id=args.request_id,
            timeout=getattr(args, "timeout", None),
        )
    except KeyBackendError as exc:
        _print_failure("could not sign", exc)
        return 1
    # stdout carries the value and nothing else: see the module docstring.
    sys.stdout.write(json.dumps(signature.as_json(), separators=(",", ":")) + "\n")
    return 0
