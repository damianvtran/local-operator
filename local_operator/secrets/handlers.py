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
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from local_operator.secrets.access import open_store, session_id
from local_operator.secrets.errors import SecretStoreError
from local_operator.secrets.keys import DIR_MODE, FILE_MODE, key_mode, secrets_dir
from local_operator.secrets.store import SecretRecord


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
        "harden": _needs_broker,
        "unlock": _needs_broker,
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
    directory = secrets_dir()
    mode = key_mode()
    payload: dict[str, Any] = {
        "directory": str(directory),
        "exists": (directory / "store.db").exists(),
        "key_mode": mode,
        "broker": "not available in this version",
    }
    if payload["exists"]:
        store = open_store()
        payload["secrets"] = len(store.list())
        payload["key_generation"] = store.key_generation()
        ok, position, message = store.verify_audit()
        payload["audit_ok"] = ok
        payload["audit_message"] = message
        payload["audit_break_at"] = position

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"directory   {payload['directory']}")
    print(f"key mode    {mode}")
    if not payload["exists"]:
        print("store       not created yet (lop secret set NAME creates it)")
        return 0
    print(f"secrets     {payload['secrets']}")
    print(f"key gen     {payload['key_generation']}")
    print(f"audit       {payload['audit_message']}")
    print(
        "note        keyfile mode keeps the master key on disk beside the store. That "
        "defeats\n            malware that scans for credential files; it does not "
        "defeat an attacker\n            who reads the key file itself."
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


def _needs_broker(args: argparse.Namespace) -> int:
    """``harden`` / ``unlock`` — real verbs whose mechanism is not here yet.

    They report that honestly instead of pretending. Passphrase mode wraps the
    master key with scrypt and keeps the unwrapped copy only in the broker's
    memory, which is the entire point of it (design §2.3): implementing the
    wrap without the daemon would leave the unwrapped key on disk anyway and
    the operator would believe they had hardened something.
    """
    _err(
        f"lop secret {args.secret_command}: passphrase mode needs the secret broker, which "
        "this version does not ship yet.\n"
        "The store is in keyfile mode: the master key is on disk beside it, 0600 in a 0700 "
        "directory."
    )
    return 2
