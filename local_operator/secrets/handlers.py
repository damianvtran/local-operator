"""Handlers behind ``lop secret`` — everything that touches the store.

Split from :mod:`local_operator.secrets.cli`, which registers the arguments,
because that module is imported on EVERY ``lop`` invocation to build
``--kind``'s choices. Keeping the crypto and SQLite imports here means
``lop --version`` never loads an OpenSSL binding; ``cli.main`` imports this
module only once a ``secret`` verb has actually been dispatched.
``tests/unit/secrets/test_startup_cost.py`` pins that arrangement.

The stdout-purity contract these implement is documented in
:mod:`local_operator.secrets.cli`; the short version is that ``get`` writes the
exact stored bytes to the stdout buffer and every diagnostic goes to stderr.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from local_operator.secrets.access import open_store, session_id
from local_operator.secrets.errors import SecretStoreError
from local_operator.secrets.keys import DIR_MODE, FILE_MODE, key_mode, secrets_dir
from local_operator.secrets.store import SecretRecord

#: How long `broker stop`/`restart` waits for the daemon's socket to go away.
#: Generous: the broker drains in-flight requests before exiting (§13), and a
#: restart that raced a dying broker would leave the operator on the old one.
_BROKER_STOP_TIMEOUT_S = 10.0


def dispatch(args: argparse.Namespace) -> int:
    """Dispatch a ``lop secret`` invocation; returns the process exit code.

    Every :class:`SecretStoreError` becomes a one-line stderr message and exit
    2. A traceback here would be worse than useless: this command is routinely
    run inside ``$( )``, where a traceback on stderr buries the one line the
    operator needs, and the failures are all ordinary operator conditions
    (no such secret, store not created, permissions loosened).
    """
    command = getattr(args, "secret_command", None)
    if command is None:
        _err("usage: lop secret {get,set,update,list,describe,rm,rotate,status,audit,file,run}")
        return 2

    handlers = {
        "get": _get,
        "set": _set,
        "update": _update,
        "list": _list,
        "describe": _describe,
        "rm": _remove,
        "delete": _remove,
        "rotate": _rotate,
        "status": _status,
        "audit": _audit,
        "file": _file,
        "run": _run,
        "harden": _harden,
        "unlock": _unlock,
        "broker": _broker,
    }
    try:
        return handlers[command](args)
    except SecretStoreError as exc:
        _err(str(exc))
        return 2
    except KeyboardInterrupt:
        return 130


# --- helpers ----------------------------------------------------------------


def _err(message: str) -> None:
    """Diagnostics go to stderr, always.

    Never stdout: see the module docstring — stdout is a data channel for
    ``get`` and a machine-parsed one for every ``--json`` verb.
    """
    print(message, file=sys.stderr)


def _read_value(args: argparse.Namespace, verb: str) -> bytes:
    """Read a secret value from ``--from-file`` or stdin, as exact bytes.

    ``sys.stdin.buffer`` rather than ``sys.stdin``: the value may be binary,
    and text mode would mangle it through the locale codec and universal
    newline translation — a CRLF in a stored PEM would come back as LF and the
    key would no longer verify.

    A single trailing newline is stripped, because ``echo`` and every heredoc
    add one and an operator piping ``echo hunter2`` means five characters, not
    six. Anything more (a value that genuinely ends in blank lines, or a
    multi-line PEM) is preserved exactly — only the last ``\\n`` goes.
    """
    from_file = getattr(args, "from_file", None)
    if from_file is not None:
        return Path(from_file).read_bytes()
    if sys.stdin is None or sys.stdin.isatty():
        _err(
            f"lop secret {verb}: the value is read from stdin, never from the command line "
            "(argv is readable by any process running as you).\n"
            f"  printf %s 'the-value' | lop secret {verb} NAME"
        )
        raise SystemExit(2)
    data = sys.stdin.buffer.read()
    if data.endswith(b"\n"):
        data = data[:-1]
    return data


def _format_time(value: float | None) -> str:
    if value is None:
        return "never"
    import datetime

    return datetime.datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def _record_dict(record: SecretRecord) -> dict[str, Any]:
    """A record as JSON-safe data. Contains no value, by construction."""
    return {
        "id": record.record_id,
        "name": record.name,
        "description": record.description,
        "kind": record.kind,
        "key_generation": record.key_generation,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "last_used_at": record.last_used_at,
    }


# --- verbs ------------------------------------------------------------------


def _get(args: argparse.Namespace) -> int:
    """Write the exact stored bytes to stdout and nothing else.

    Straight to ``sys.stdout.buffer``: the value is bytes, and going through
    the text layer would re-encode it in the locale's codec and translate
    newlines. No trailing newline is added — the caller asked for a value, and
    ``$( )`` stripping the newline we add is luck, not contract, in every other
    consumer (a Python ``subprocess.check_output``, a here-string, a file
    redirect).
    """
    value = open_store().get(args.name, session_id=session_id())
    sys.stdout.buffer.write(value)
    sys.stdout.buffer.flush()
    return 0


def _set(args: argparse.Namespace) -> int:
    value = _read_value(args, "set")
    record = open_store(create=True).set(
        args.name,
        value,
        description=args.description,
        kind=args.kind,
        session_id=session_id(),
    )
    # The confirmation names the secret and its size, never its value. Size is
    # genuinely useful — it is how an operator notices they stored the shell
    # prompt instead of the token — and reveals nothing.
    _err(f"stored {record.name} ({len(value)} bytes, kind={record.kind})")
    return 0


def _update(args: argparse.Namespace) -> int:
    value = _read_value(args, "update")
    record = open_store().update(
        args.name, value, description=args.description, session_id=session_id()
    )
    _err(f"updated {record.name} ({len(value)} bytes)")
    return 0


def _list(args: argparse.Namespace) -> int:
    """Names, kinds and descriptions. NEVER a value — there is no flag for it.

    Retrieval is ``get``, one secret at a time, so that every value leaving the
    store is one audited event attributable to one name. A ``--values`` flag
    here would turn a single audit entry into a bulk export.
    """
    records = open_store().list()
    if args.json:
        print(json.dumps([_record_dict(record) for record in records], indent=2))
        return 0
    if not records:
        _err("no secrets stored")
        return 0
    width = max(len(record.name) for record in records)
    for record in records:
        suffix = f"  {record.description}" if record.description else ""
        print(f"{record.name:<{width}}  {record.kind:<6}{suffix}")
    return 0


def _describe(args: argparse.Namespace) -> int:
    record = open_store().describe(args.name)
    if args.json:
        print(json.dumps(_record_dict(record), indent=2))
        return 0
    print(f"name         {record.name}")
    print(f"id           {record.record_id}")
    print(f"kind         {record.kind}")
    print(f"description  {record.description or '(none)'}")
    print(f"created      {_format_time(record.created_at)}")
    print(f"updated      {_format_time(record.updated_at)}")
    print(f"last used    {_format_time(record.last_used_at)}")
    print(f"key gen      {record.key_generation}")
    return 0


def _remove(args: argparse.Namespace) -> int:
    """Delete a secret, confirming first unless ``--yes``.

    The value is not recoverable afterwards and there is no undo, so an
    interactive run asks. A non-tty run (an agent's bash call, a script) cannot
    answer a prompt, so it requires ``--yes`` explicitly rather than defaulting
    to either answer.
    """
    if not args.yes:
        if not sys.stdin or not sys.stdin.isatty():
            _err(f"refusing to delete {args.name} without --yes (stdin is not a terminal)")
            return 2
        _err(f"Delete {args.name} permanently? This cannot be undone. [y/N] ")
        if input().strip().lower() not in ("y", "yes"):
            _err("cancelled")
            return 1
    record = open_store().delete(args.name, session_id=session_id())
    _err(f"deleted {record.name}")
    return 0


def _rotate(args: argparse.Namespace) -> int:
    """Re-seal every record under a fresh master key.

    The new key file is installed only AFTER the database transaction commits.
    A crash between the two leaves the old key matching an unmodified database,
    which is recoverable; the other order would leave a new key against records
    still sealed under the old one, which is not.
    """
    from local_operator.secrets.crypto import generate_master_key
    from local_operator.secrets.keys import replace_master_key

    store = open_store()
    new_key = generate_master_key()
    moved = store.rotate(new_key, session_id=session_id())
    replace_master_key(None, new_key)
    _err(f"rotated {moved} secret(s) to key generation {store.key_generation()}")
    return 0


def _status(args: argparse.Namespace) -> int:
    """Where the store is, what mode it is in, and how many records it holds.

    Deliberately does not overclaim. The ``mode`` line says ``keyfile`` and the
    note says what that means: the master key is on disk next to the database,
    which stops opportunistic credential-scanning malware and does not stop an
    attacker who reads the key file. This is the text an operator forms their
    mental model from, so it states the residual risk (design §9) rather than
    implying a vault.
    """
    from local_operator.secrets.client import broker_status

    directory = secrets_dir()
    mode = key_mode()
    exists = (directory / "store.db").exists()
    # Open the store FIRST when there is one, then sample the broker (QA Q3).
    # `open_store` lazily starts a broker, so sampling first reported "not
    # running" while this very command was starting one — wrong precisely on
    # the first run after a reboot, which is when an operator most needs this
    # diagnostic to be true.
    #
    # A failure to open is NOT fatal here, and that matters more than the
    # ordering: `status` is the command an operator runs when the store is
    # already misbehaving — locked, hardened-but-not-unlocked, permissions
    # wrong — and it has to keep reporting the directory, the tier and the
    # broker state in exactly those cases. Letting the exception escape would
    # print nothing at all and exit 2 on a locked hardened store, which is the
    # single case where the operator most needs to be told "locked — run
    # `lop secret unlock`".
    store = None
    store_error: str | None = None
    if exists:
        try:
            store = open_store()
        except SecretStoreError as exc:
            store_error = str(exc)
    broker = broker_status()
    payload: dict[str, Any] = {
        "directory": str(directory),
        "exists": exists,
        "key_mode": mode,
        "broker_running": broker is not None,
        "broker_pid": (broker or {}).get("pid"),
        "broker_locked": (broker or {}).get("locked"),
    }
    if store is not None:
        payload["secrets"] = len(store.list())
        payload["key_generation"] = store.key_generation()
        ok, position, message = store.verify_audit()
        payload["audit_ok"] = ok
        payload["audit_message"] = message
        payload["audit_break_at"] = position
    elif store_error is not None:
        payload["store_error"] = store_error

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"directory   {payload['directory']}")
    print(f"key mode    {mode}")
    if broker is None:
        print("broker      not running (it starts on demand)")
    else:
        locked = " (locked — run `lop secret unlock`)" if payload["broker_locked"] else ""
        print(f"broker      running, pid {payload['broker_pid']}{locked}")
    if not payload["exists"]:
        print("store       not created yet (lop secret set NAME creates it)")
        return 0
    if store is None:
        # The store exists but could not be opened — almost always a hardened
        # store that has not been unlocked this boot. Everything above still
        # printed, which is the point: the operator learns the tier and the
        # broker state, which is what tells them what to do next.
        print(f"store       present but not readable: {store_error}")
        return 0
    print(f"secrets     {payload['secrets']}")
    print(f"key gen     {payload['key_generation']}")
    print(f"audit       {payload['audit_message']}")
    # The note is the text an operator builds their mental model from, so it
    # states the residual risk (design §9) per tier rather than implying a
    # vault. Neither line may be widened into a stronger claim.
    if mode == "passphrase":
        print(
            "note        passphrase mode keeps the master key on disk only scrypt-wrapped; "
            "the\n            unwrapped copy lives in the broker's memory until it exits. "
            "Anything\n            running as you that can run `lop` can still read your "
            "secrets while\n            the broker is unlocked."
        )
    else:
        print(
            "note        keyfile mode keeps the master key on disk beside the store. That "
            "defeats\n            malware that scans for credential files; it does not "
            "defeat an attacker\n            who reads the key file itself. "
            "`lop secret harden` moves it into\n            passphrase mode."
        )
    return 0


def _audit(args: argparse.Namespace) -> int:
    store = open_store()
    if args.verify:
        ok, position, message = store.verify_audit()
        if args.json:
            print(json.dumps({"ok": ok, "broken_at": position, "message": message}, indent=2))
        else:
            print(message)
        return 0 if ok else 1
    records = store.audit_entries(limit=args.limit)
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "ts": entry.ts,
                        "event": entry.event,
                        "secret_id": entry.secret_id,
                        "session_id": entry.session_id,
                        "pid": entry.pid,
                        "outcome": entry.outcome,
                        "hash": entry.hash.hex(),
                    }
                    for entry in records
                ],
                indent=2,
            )
        )
        return 0
    for entry in records:
        print(
            f"{_format_time(entry.ts)}  {entry.event:<7} {entry.outcome:<5} "
            f"pid={entry.pid or '-':<7} {entry.secret_id or ''}"
        )
    return 0


def _file(args: argparse.Namespace) -> int:
    """Materialise a file secret at a private path for one command (design §7).

    A 0600 file inside a fresh 0700 directory under ``$TMPDIR``, removed in a
    ``finally``. The design measured the two tidier-looking alternatives and
    both are wrong on macOS: a FIFO is not seekable, and ``/dev/fd/N`` on a
    deleted file is a DUP with a *shared offset*, so a consumer that opens it
    twice — which google-auth does — reads zero bytes the second time.

    Stated plainly because the guidance has to say it: the plaintext IS on disk
    for the lifetime of this command. Randomly named, in a directory only this
    user can traverse, and gone afterwards — far better than a permanent
    well-known path, but not zero exposure.
    """
    command = [argument for argument in args.command if argument != "--"]
    if not command:
        _err("usage: lop secret file NAME [--env-var VAR] -- COMMAND...")
        return 2

    value = open_store().get(args.name, session_id=session_id())
    directory = Path(tempfile.mkdtemp(prefix="lop-secret-"))
    os.chmod(directory, DIR_MODE)
    target = directory / args.name.replace(os.sep, "_")
    try:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, FILE_MODE)
        try:
            os.write(descriptor, value)
        finally:
            os.close(descriptor)
        os.chmod(target, FILE_MODE)
        environment = os.environ.copy()
        environment[args.env_var] = str(target)
        return subprocess.run(command, env=environment, check=False).returncode
    finally:
        # A `finally` covers a normal exit, an exception and a SIGINT that
        # Python turns into KeyboardInterrupt. It does NOT cover SIGKILL or a
        # SIGTERM with no handler, so the plaintext can outlive a hard kill —
        # the broker PR, which owns process lifecycle, installs the signal
        # handler that narrows this. Named rather than papered over.
        shutil.rmtree(directory, ignore_errors=True)


def _run(args: argparse.Namespace) -> int:
    """Run a command with named secrets exported into its environment.

    The child's environment is readable by any same-uid process while it runs
    (design §2.2, spike 2), so this is strictly weaker than ``$(lop secret
    get)`` interpolation and exists only for consumers that read a fixed
    variable and offer no other input. The help text says so; so does the
    guidance in PR 4.
    """
    command = [argument for argument in args.command if argument != "--"]
    if not command:
        _err("usage: lop secret run --secret NAME[=VAR] -- COMMAND...")
        return 2
    if not args.secret:
        _err("lop secret run: name at least one secret with --secret NAME")
        return 2

    store = open_store()
    environment = os.environ.copy()
    for specification in args.secret:
        name, _, variable = specification.partition("=")
        environment[variable or name] = store.get(name, session_id=session_id()).decode(
            "utf-8", errors="strict"
        )
    return subprocess.run(command, env=environment, check=False).returncode


def _read_passphrase(prompt: str) -> str:
    """Read a passphrase from the terminal, never from argv or a flag.

    ``getpass`` reads from ``/dev/tty`` where it can, so this still works with
    stdin redirected — and it keeps the passphrase off the command line, which
    any same-uid process can read through ``ps`` and ``KERN_PROCARGS2``
    (design §2.2, spike 2). A passphrase in argv is a passphrase already
    leaked, exactly as a value in argv is.
    """
    import getpass

    try:
        return getpass.getpass(prompt)
    except (EOFError, OSError) as exc:
        raise SecretStoreError(
            "A passphrase must be typed at a terminal; there is no flag for it "
            "because a passphrase on the command line is readable by any process "
            "running as you."
        ) from exc


def _harden(args: argparse.Namespace) -> int:
    """Move the store from ``keyfile`` mode to ``passphrase`` mode (design §2.3).

    What this buys, stated exactly: afterwards the master key exists on disk
    only scrypt-wrapped, and the unwrapped copy lives solely in the broker's
    memory — which a same-uid attacker cannot read without tripping a macOS
    authorization prompt (spikes 3, 5). It converts "a script reads the key
    file and decrypts the store" into "a script must impersonate a lop session
    while the broker is unlocked, or raise a password dialog on the operator's
    screen". It does NOT protect a store while it is unlocked and in use
    (design §9.4), and this is the one prompt the operator accepted: once per
    boot, never per access.
    """
    from local_operator.secrets.keys import key_mode as current_mode
    from local_operator.secrets.keys import load_master_key, wrap_master_key

    if current_mode() == "passphrase":
        _err("This store is already hardened. Use `lop secret unlock` to unlock it.")
        return 2
    # Read the key BEFORE prompting: a store that cannot be opened should fail
    # before the operator types a passphrase they then discover was pointless.
    key = load_master_key()
    passphrase = _read_passphrase("New passphrase for the secret store: ")
    if not passphrase:
        _err("Empty passphrase; nothing changed.")
        return 2
    if passphrase != _read_passphrase("Repeat the passphrase: "):
        _err("The passphrases did not match; nothing changed.")
        return 2
    wrap_master_key(None, key, passphrase)
    _err(
        "Store hardened. The master key is now wrapped with your passphrase and the "
        "plaintext key file is gone.\n"
        "Run `lop secret unlock` once after each reboot; retrievals are silent after that.\n"
        "If you forget this passphrase the secrets cannot be recovered — there is no "
        "escrow copy."
    )
    return 0


def _unlock(args: argparse.Namespace) -> int:
    """Unlock a hardened store for this boot by unwrapping the key into the broker."""
    from local_operator.secrets.client import ensure_broker, unlock
    from local_operator.secrets.keys import key_mode as current_mode

    if current_mode() != "passphrase":
        _err(
            "This store is in keyfile mode, so there is nothing to unlock. "
            "Run `lop secret harden` to move it to passphrase mode."
        )
        return 2
    if not ensure_broker(None):
        _err("Could not start the secret broker, so there is nowhere to hold the unlocked key.")
        return 2
    unlock(_read_passphrase("Passphrase for the secret store: "))
    _err("Unlocked. The broker holds the key in memory until it exits or the machine reboots.")
    return 0


def _stop_broker(client: Any) -> bool | None:
    """Stop a running broker. ``True`` stopped, ``None`` none ran, ``False`` failed.

    Three outcomes rather than a bool because ``restart`` must continue after
    "none was running" but abort after a real failure, and ``stop`` reports
    those two differently.

    Stopping waits for the socket to go away instead of returning the moment
    SIGTERM is delivered: ``restart`` immediately starts another, and the lazy
    start would otherwise find the dying broker's socket still present, decide
    one is already running, and leave the operator on the OLD process — the
    exact skew this verb exists to clear.
    """
    status = client.broker_status(None)
    if status is None:
        # Exit 0, deliberately (QA Q6). `stop` names a desired END STATE, and
        # that state already holds — the daemon is started lazily and exits on
        # its own idle timer, so "no broker" is its normal resting condition
        # rather than an error. Returning non-zero would make every teardown
        # script that stops a broker it did not start report a failure.
        _err("No secret broker is running.")
        return None
    pid = status.get("pid")
    if not isinstance(pid, int):
        _err("The broker did not report a pid; not killing anything.")
        return False
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as exc:
        _err(f"Could not stop the secret broker (pid {pid}): {exc}")
        return False
    deadline = time.monotonic() + _BROKER_STOP_TIMEOUT_S
    while time.monotonic() < deadline and client.is_running(None):
        time.sleep(0.05)
    if client.is_running(None):
        _err(f"The secret broker (pid {pid}) did not exit within {_BROKER_STOP_TIMEOUT_S:g}s.")
        return False
    _err(f"stopped secret broker (pid {pid})")
    return True


def _broker(args: argparse.Namespace) -> int:
    """``lop secret broker {status,start,stop,restart,run}``."""
    from local_operator.secrets import broker as broker_module
    from local_operator.secrets import client

    command = getattr(args, "broker_command", None) or "status"

    if command == "run":
        # Foreground, for debugging. The normal path is the lazy start in
        # client.ensure_broker; this exists so an operator can watch one.
        return broker_module.run_broker()

    if command == "start":
        if client.ensure_broker(None):
            status = client.broker_status(None) or {}
            _err(f"secret broker running (pid {status.get('pid', '?')})")
            return 0
        _err("Could not start the secret broker.")
        return 2

    if command in ("stop", "restart"):
        stopped = _stop_broker(client)
        if stopped is False:
            return 2
        if command == "stop":
            return 0
        # restart: the reason this verb exists is version skew, which
        # AGENTS.md calls routine here because `lop-update` runs under live
        # sessions. The broker's own protocol-mismatch error names it, so it
        # has to exist (QA Q4).
        if client.ensure_broker(None):
            status = client.broker_status(None) or {}
            _err(f"secret broker restarted (pid {status.get('pid', '?')})")
            return 0
        _err("Could not start the secret broker.")
        return 2

    status = client.broker_status(None)
    if getattr(args, "json", False):
        print(json.dumps(status or {"running": False}, indent=2))
        return 0
    if status is None:
        print("broker      not running (it starts on demand)")
        return 0
    print(f"broker      running (pid {status.get('pid')})")
    print(f"protocol    {status.get('protocol')}")
    print(f"locked      {status.get('locked')}")
    print(f"sessions    {len(status.get('sessions') or [])} registered")
    return 0
