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
import json
import os
import shutil
import subprocess
import sys
import time
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
    install_commands,
    key_id_for,
    load_anchor,
    load_staged_anchor,
    operator_authority_report,
    sign_challenge,
    staging_path,
)
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


def install_anchor(root: Path, *, print_only: bool = False, sudo_secret: str = "") -> int:
    """The ONE privileged step: move the staged anchor to where the runtime reads it.

    Returns a process exit code rather than raising, because every caller is a
    CLI verb whose contract is an exit status. ``print_only`` is what
    ``lop operator install --print-only`` prints, unchanged from when this lived
    inline in that verb. ``sudo_secret`` is the INTERIM ask-sudo path (design
    §3.7/OQ15): the admin password is resolved in place from the secret store and
    fed to ``sudo -S`` on stdin for this one step, never printed or logged.
    """
    staged = staging_path(root)
    target = anchor_path()
    if not staged.exists():
        # R1-4 (§4's rule applied to the init-shaped refusals): the remedy names the
        # product action, not the terminal verb that performs it.
        print(
            f"nothing staged at {staged}; ask Local Operator to set up operator "
            "authority (the setup stages the key)",
            file=sys.stderr,
        )
        return 1
    command = install_commands(staged, target)
    if print_only:
        print(" && ".join(" ".join(argv) for argv in command))
        return 0
    if _is_privileged():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(staged, target)
        os.chmod(target, 0o644)
    else:
        if shutil.which("sudo") is None:
            print(
                "installing the anchor needs administrator rights. Run this by hand:\n  "
                + "\n  ".join(" ".join(argv) for argv in command),
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
    source: Path, root: Path, *, print_only: bool = False, sudo_secret: str = ""
) -> int:
    """Install a statement RECEIVED from another machine (remote onboarding §3.3 step 7).

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
            f"the anchor statement at {source} isn't the one the operator's machine "
            "exported (it looks edited or re-saved), so nothing was installed — ask "
            "for a fresh export",
            file=sys.stderr,
        )
        return 1
    staged = stage_anchor(root, anchor)
    print(f"anchor statement accepted from {source} (staged at {staged})", file=sys.stderr)
    return install_anchor(root, print_only=print_only, sudo_secret=sudo_secret)


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
    ``_init``, and the create that came back ``reused`` — the race where a key appeared
    between the probe and the create (agent review round 1, R1-5). Both callers pass the
    handle ``_existing_key`` / ``create_key`` returned, never a signer.
    """
    staged = staging_path(root)
    loaded = load_anchor()
    staged_now = False
    if not staged.exists() and not loaded.exists:
        staged = stage_anchor(root, anchor_for_handle(handle, label=label))
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
    anchor = anchor_for_handle(handle, label=args.label)
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
    """
    from local_operator.operator.devices import (
        AUTHORISE_COMMAND,
        list_devices,
        revoked_ids,
    )

    paired = list_devices(config_dir())
    revoked = revoked_ids(config_dir())
    if paired:
        print(f"paired devices         : {len(paired)}")
        for device in paired:
            print(f"  - {device.device_id}  {device.name or '(unnamed)'}")
    if revoked:
        print(f"revoked devices        : {len(revoked)}")
        for device_id in revoked:
            print(
                f"  - {device_id}  (revoked — "
                f"`{AUTHORISE_COMMAND.format(device_id=device_id)}` brings it back)"
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
