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
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from local_operator.secrets.access import open_store, retrieve_secret, session_id
from local_operator.secrets.errors import BrokerIncompatible, SecretStoreError
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

    The same treatment is extended to the ordinary I/O and decoding failures
    that are NOT ``SecretStoreError`` — a missing ``--from-file``, a store file
    that is not a database, a value that is not UTF-8 under ``run``. Those used
    to escape to the generic CLI handler, which prints an ANSI ``Error:`` line
    and a stack-trace box, and — worse for a scripted caller — exited **1**,
    which ``audit --verify`` already uses to mean "the chain is broken". A
    crash and a detected tamper must not be indistinguishable to a script, so
    they are mapped to the same one-line stderr and rc=2 as everything else.
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
    except (OSError, UnicodeError, sqlite3.DatabaseError) as exc:
        # Deliberately narrow. These are the conditions an operator can act on
        # (a path that does not exist, a permission, a damaged database file, a
        # value that is not text); a genuine bug in this module still raises
        # and still gets its traceback, because that one nobody can act on
        # without it.
        _err(f"{type(exc).__name__}: {exc}")
        return 2


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


class _Terminated(SystemExit):
    """Raised in the main thread when SIGTERM arrives, to run ``finally``.

    A :class:`SystemExit` subclass so it unwinds the stack — which is the whole
    point — without being swallowed by an ``except Exception``. The exit code
    is the shell's convention for a signal death, 128 + SIGTERM.
    """

    def __init__(self) -> None:
        super().__init__(128 + signal.SIGTERM)


def _install_sigterm_cleanup() -> Any:
    """Make SIGTERM unwind the stack so ``finally`` blocks actually run.

    Python's default SIGTERM disposition kills the interpreter outright: no
    ``finally``, no ``atexit``. For ``lop secret file`` that means a routine
    ``kill`` — a shell logging out, a supervisor reaping a stuck command,
    ``lop-update`` stopping a runtime — strands DECRYPTED plaintext (a service
    account key, a private key) in ``$TMPDIR`` until the machine reboots. That
    directly contradicts design §7's "gone afterwards", and it is a handful of
    lines to close.

    Raising rather than cleaning up inside the handler is deliberate: the
    handler runs on whatever the main thread was doing, and ``shutil.rmtree``
    there would race the ``finally`` that may already be running. Unwinding
    lets the single existing cleanup path do the work exactly once.

    Returns the previous handler so it can be restored; signal state is
    process-global, and this function is called from a library.
    """

    def _on_sigterm(signum: int, frame: Any) -> None:
        raise _Terminated()

    try:
        return signal.signal(signal.SIGTERM, _on_sigterm)
    except ValueError:
        # Only the main thread may install a handler. An embedder calling this
        # off-thread still gets the `finally`; it just does not get the signal
        # coverage, which is strictly better than refusing to run.
        return None


def _restore_sigterm(previous: Any) -> None:
    """Put the process's SIGTERM disposition back."""
    if previous is None:
        return
    try:
        signal.signal(signal.SIGTERM, previous)
    except ValueError:
        pass


def _child_command(tokens: list[str]) -> list[str]:
    """The child command, stripping only a LEADING ``--`` separator.

    Filtering out every ``--`` corrupts child commands that legitimately
    contain one, silently and with no error — ``git log -- path``,
    ``kubectl exec pod -- cmd``, a ``bash -c`` script whose own arguments carry
    a separator. The command still runs; it just means something different from
    what the operator wrote.

    Only the leading token can be OUR separator, and even that is usually gone
    before we see it: with ``nargs="*"`` argparse consumes the first ``--`` it
    uses to stop option parsing. A second one survives into this list and
    belongs to the child, so exactly one leading occurrence is removed here.
    """
    if tokens and tokens[0] == "--":
        return tokens[1:]
    return list(tokens)


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

    Routed through :func:`~local_operator.secrets.access.retrieve_secret` so
    the §6 notice fires BEFORE these bytes exist here (QA Q3). This verb is the
    headline path — ``$(lop secret get NAME)`` is what the credentials guide
    tells agents to write — so it is the one that most needs the value already
    registered for redaction by the time it can be printed. Nothing about the
    stdout contract changes: same bytes, no trailing newline.
    """
    value = retrieve_secret(args.name)
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
    store = open_store()
    records = store.list()
    # Damaged rows are named on STDERR, never mixed into stdout: `--json` is
    # machine-parsed and the plain listing is read by eye. Reporting them is
    # what keeps `list` both honest and usable when the store is damaged --
    # it used to raise on the first one, which took down the only command that
    # could have shown the operator what survived.
    damaged = store.damaged_records()
    if args.json:
        print(json.dumps([_record_dict(record) for record in records], indent=2))
        _warn_damaged(damaged)
        return 0
    if not records:
        _err("no secrets stored")
        _warn_damaged(damaged)
        return 0
    width = max(len(record.name) for record in records)
    for record in records:
        suffix = f"  {record.description}" if record.description else ""
        print(f"{record.name:<{width}}  {record.kind:<6}{suffix}")
    _warn_damaged(damaged)
    return 0


def _warn_damaged(damaged: list[str]) -> None:
    """Tell the operator about unreadable rows, and how to remove one.

    On stderr with the id and the exact command, because a damaged record is
    actionable but only if the operator is given the one thing `rm NAME` cannot
    give them: the name is inside the ciphertext they cannot open.
    """
    if not damaged:
        return
    _err(
        f"warning: {len(damaged)} record(s) in this store cannot be decrypted and are not "
        "listed above. They were sealed under a key that no longer exists, or they have "
        "been altered."
    )
    for record_id in damaged:
        _err(f"  {record_id}  (remove with: lop secret rm --id {record_id} --yes)")


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
    target = args.id or args.name
    if not target:
        _err("usage: lop secret rm NAME | lop secret rm --id RECORD_ID")
        return 2
    if args.id and args.name:
        _err("lop secret rm: give a NAME or --id, not both")
        return 2

    if not args.yes:
        if not sys.stdin or not sys.stdin.isatty():
            _err(f"refusing to delete {target} without --yes (stdin is not a terminal)")
            return 2
        _err(f"Delete {target} permanently? This cannot be undone. [y/N] ")
        if input().strip().lower() not in ("y", "yes"):
            _err("cancelled")
            return 1

    if args.id:
        # The repair path for a record `rm NAME` cannot reach: a row whose
        # ciphertext will not open has no readable name and no valid blind
        # index, so deleting by primary key is the only way out short of
        # hand-editing SQLite.
        if not open_store().delete_record_id(args.id, session_id=session_id()):
            _err(f"no record with id {args.id} in this store")
            return 2
        _err(f"deleted record {args.id}")
        return 0

    record = open_store().delete(args.name, session_id=session_id())
    _err(f"deleted {record.name}")
    return 0


def _rotate(args: argparse.Namespace) -> int:
    """Re-seal every record under a fresh master key.

    Three steps, in this order, because a crash between any two of them must
    still leave the store openable by a key that exists on disk:

    1. STAGE the new key beside the old one. Inert until the database moves.
    2. COMMIT the re-seal. From here the database needs the new key, and the
       staged file is the copy of it that survives this process dying.
    3. INSTALL the staged key as ``master.key`` and remove the staging file,
       but ONLY if the database is still sealed under it.

    The previous order committed first and wrote the key afterwards, with a
    docstring claiming a crash there "leaves the old key matching an unmodified
    database". It does not: the database is fully re-sealed at that point and
    the only copy of its key is in this process's memory, so an ordinary power
    cut destroyed every secret in the store. ``resolve_master_key`` recovers a
    crash between 2 and 3 by matching the staged key's fingerprint against the
    one the database records.

    **Step 3 is conditional, and that is what makes concurrent rotation safe.**
    The epoch guard in step 2 serialises the COMMITs but imposed no order on the
    installs, so a rotator that committed earlier could install its superseded
    key later and leave the database sealed under a key held nowhere on disk —
    with both processes reporting success.
    :func:`local_operator.secrets.store.install_master_key_if_current` carries
    the full analysis; it re-checks the fingerprint under the database's write
    lock and refuses rather than clobbering.

    A rotator that loses says so and exits non-zero. The loss is not a fault:
    the operator's secrets are intact under the winner's key, this rotation's
    re-seal was superseded by an equally valid one, and the honest report is
    "another rotation completed first, run it again if you still want one".
    Reporting success would be the actual defect — it is what the unguarded
    install did while destroying the store.

    **Every step above is TIER-AWARE, and it was not (QA Q10).** The three steps
    are the same in both tiers, but "the key" names a different file in each:
    ``master.key`` in the keyfile tier, ``master.key.wrapped`` in the hardened
    one. This function staged and installed the plaintext file unconditionally,
    so a single rotation of a hardened store re-sealed the database under a new
    key while ``unlock`` kept unwrapping the OLD one — every secret
    undecryptable, exit code 0, and a plaintext master key left sitting beside
    the database while ``status`` still reported ``passphrase``. That second
    consequence is the worse one: the tier's entire claim (§2.3) is that
    nothing on disk decrypts the store on its own.

    The hardened path is therefore delegated to :func:`_rotate_hardened`, which
    keeps the identical stage → commit → install ordering over the wrapped file.
    """
    from local_operator.secrets.crypto import generate_master_key
    from local_operator.secrets.keys import (
        assert_key_of_record_invariant,
        discard_staged_master_key,
        key_of_record_inconsistency,
        stage_master_key,
    )
    from local_operator.secrets.store import install_master_key_if_current

    if key_mode() == "passphrase":
        return _rotate_hardened()

    # **Sampled BEFORE the rotation, because the answer afterwards is the same
    # and the meaning is not (review R4-2).** A store carrying both key files is
    # Q10 damage that predates this command. Rotating it is correct — the
    # plaintext key IS the live one, so the keyfile path is the right path — but
    # the stale wrapped file survives the rotation, and the post-condition below
    # then reported this command as having failed after it had fully succeeded.
    # Recording the state up front is what lets the exit status distinguish "I
    # installed the wrong file" from "I finished on a store that was already
    # damaged".
    inherited_damage = key_of_record_inconsistency() is not None

    store = open_store()
    new_key = generate_master_key()
    stage_master_key(None, new_key)
    try:
        moved = store.rotate(new_key, session_id=session_id())
    except BaseException:
        # Nothing committed, so the staged key is not the store's key and
        # leaving it would be a stray copy of key material for no benefit.
        # Scoped to THIS rotation's key: a concurrent rotator that won the
        # epoch race may be committed and not yet installed, and its staged
        # file is the only on-disk copy of the key its database now needs.
        discard_staged_master_key(None, new_key)
        raise
    if not install_master_key_if_current(new_key):
        # Superseded between this rotation's COMMIT and its install. The store
        # is fully usable under the winner's key; only this rotation is void.
        # Safe to discard THIS key now: the compare-and-swap read the
        # fingerprint under the write lock and it was not ours, so no committed
        # database needs it and no later one can (the key is fresh randomness).
        discard_staged_master_key(None, new_key)
        _err(
            "Another key rotation completed first, so this one was not installed. "
            "Your secrets are intact and readable under the key that rotation "
            "installed; nothing was lost. Run `lop secret rotate` again if you "
            "still want a fresh key."
        )
        return 2
    assert_key_of_record_invariant(None, "keyfile", stale_wrapped_ok=inherited_damage)
    _err(f"rotated {moved} secret(s) to key generation {store.key_generation()}")
    if inherited_damage:
        # Said plainly and separately from the success line: the rotation did
        # work, and the store still carries damage this verb cannot repair —
        # only `harden` holds the passphrase needed to re-wrap the live key.
        # Exit 0 regardless, because retrying `rotate` never fixes it and a
        # non-zero status here is what sent the operator into that loop.
        _err(
            "This store still has a plaintext master key beside a stale wrapped one, so it "
            "is NOT protected at rest the way passphrase mode claims. The rotation above "
            "succeeded; run `lop secret harden` to re-wrap the live key and remove the "
            "plaintext copy."
        )
    return 0


def _rotate_hardened() -> int:
    """``rotate`` on a passphrase-hardened store (QA Q10).

    Same three steps and same ordering as :func:`_rotate`, over the wrapped file
    instead of the plaintext one, so the crash-safety argument carries over
    unchanged: at every instant a key that opens the store exists on disk, and
    the window between the COMMIT and the install is recovered — here by
    :func:`~local_operator.secrets.keys.unwrap_master_key_matching` at the next
    ``unlock``, because a wrapped staged key can only be tested when the
    passphrase is present.

    **The passphrase is re-typed rather than taken from the broker, and that is
    deliberate.** The broker holds the unwrapped KEY, never the passphrase that
    wraps it — by design, since the passphrase is the one secret §2.3 keeps off
    disk and out of every process but the one the operator typed it into. So
    there is nothing to reuse: re-wrapping needs the passphrase itself. Asking
    for it also means a rotation cannot be performed by something that merely
    inherited this terminal's standing, which is a narrower authority than the
    unlock grant — appropriate for the verb that replaces the key of record.

    Confirmed rather than assumed once, because a mistyped new passphrase here
    would seal the store under a key the operator cannot unwrap, which is the
    same unrecoverable loss by another route.
    """
    from local_operator.secrets.crypto import generate_master_key
    from local_operator.secrets.keys import (
        assert_key_of_record_invariant,
        discard_staged_wrapped_key,
        install_staged_wrapped_key,
        stage_wrapped_master_key,
    )
    from local_operator.secrets.store import install_key_of_record_if_current

    # Open the store BEFORE prompting: a locked or unreadable store must fail
    # before the operator types a passphrase they then learn was pointless.
    store = open_store()
    passphrase = _read_passphrase("Passphrase for the secret store: ")
    if not passphrase:
        _err("Empty passphrase; nothing changed.")
        return 2
    if passphrase != _read_passphrase("Repeat the passphrase: "):
        _err("The passphrases did not match; nothing changed.")
        return 2

    new_key = generate_master_key()
    staged = stage_wrapped_master_key(None, new_key, passphrase)
    try:
        moved = store.rotate(new_key, session_id=session_id())
    except BaseException:
        # Nothing committed, so this staged blob wraps a key no database needs.
        discard_staged_wrapped_key(None, staged)
        raise

    # The compare-and-swap `install_master_key_if_current` performs for the
    # keyfile tier, over the wrapped file. A rotation that was superseded
    # between its COMMIT and this point must NOT install, for the reason that
    # function documents at length: the winner's key would be replaced by a
    # stale one and the store sealed under a key held nowhere.
    installed = install_key_of_record_if_current(
        new_key, None, lambda: install_staged_wrapped_key(None, staged)
    )
    if not installed:
        discard_staged_wrapped_key(None, staged)
        _err(
            "Another key rotation completed first, so this one was not installed. "
            "Your secrets are intact and readable under the key that rotation "
            "installed; nothing was lost. Run `lop secret rotate` again if you "
            "still want a fresh key."
        )
        return 2

    assert_key_of_record_invariant(None, "passphrase")
    _err(f"rotated {moved} secret(s) to key generation {store.key_generation()}")
    _err(
        "The broker still holds the previous key, so run `lop secret broker restart` "
        "and `lop secret unlock` to serve the new one."
    )
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
    from local_operator.secrets.keys import key_of_record_inconsistency

    directory = secrets_dir()
    mode = key_mode()
    exists = (directory / "store.db").exists()
    # A store carrying BOTH key files reports its tier honestly above
    # (``keyfile``, because a plaintext key IS on disk), but the operator
    # hardened it and has no reason to look. Naming the state is the difference
    # between a silent downgrade and an actionable one — QA Q10's second and
    # worse consequence, where `status` kept printing `key mode passphrase`
    # while the live key sat unwrapped beside the database.
    inconsistency = key_of_record_inconsistency()
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
        "key_inconsistency": inconsistency,
    }
    if store is not None:
        payload["secrets"] = len(store.list())
        payload["damaged"] = store.damaged_records()
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
    if inconsistency is not None:
        print(f"WARNING     {inconsistency}")
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
    if payload["damaged"]:
        print(f"damaged     {len(payload['damaged'])} unreadable record(s)")
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
    command = _child_command(args.command)
    if not command:
        _err("usage: lop secret file NAME [--env-var VAR] -- COMMAND...")
        return 2

    # Same announcement seam as `_get`: the plaintext is about to be written
    # to a path the child reads, so the session must already be scrubbing it.
    value = retrieve_secret(args.name)
    directory = Path(tempfile.mkdtemp(prefix="lop-secret-"))
    os.chmod(directory, DIR_MODE)
    target = directory / args.name.replace(os.sep, "_")
    previous_sigterm = _install_sigterm_cleanup()
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
        # A `finally` covers a normal exit, an exception, and — because of the
        # handler installed above — a SIGTERM. It does NOT cover SIGKILL, which
        # is unblockable by definition: the plaintext outlives a `kill -9` and
        # no in-process design can change that. Narrowing THAT needs an
        # out-of-process owner, so it belongs to the broker PR. Named rather
        # than papered over.
        shutil.rmtree(directory, ignore_errors=True)
        _restore_sigterm(previous_sigterm)


def _run(args: argparse.Namespace) -> int:
    """Run a command with named secrets exported into its environment.

    The child's environment is readable by any same-uid process while it runs
    (design §2.2, spike 2), so this is strictly weaker than ``$(lop secret
    get)`` interpolation and exists only for consumers that read a fixed
    variable and offer no other input. The help text says so; so does the
    guidance in PR 4.
    """
    command = _child_command(args.command)
    if not command:
        _err("usage: lop secret run --secret NAME[=VAR] -- COMMAND...")
        return 2
    if not args.secret:
        _err("lop secret run: name at least one secret with --secret NAME")
        return 2

    environment = os.environ.copy()
    for specification in args.secret:
        name, _, variable = specification.partition("=")
        # Through the announcement seam, per secret. These values go into a
        # child's environment, which any same-uid process can read while it
        # runs, so they need the redaction notice at least as much as `_get`'s
        # do. One store handle is no longer reused across the loop: each
        # retrieval is its own broker round trip, which is also what makes the
        # audit trail record them individually.
        environment[variable or name] = retrieve_secret(name).decode("utf-8", errors="strict")
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

    **This is also the REPAIR path for a store damaged by the pre-fix ``rotate``
    (QA Q10).** That bug left a plaintext ``master.key`` beside a stale
    ``master.key.wrapped``; :func:`~local_operator.secrets.keys.key_mode` now
    reports such a store as ``keyfile`` — truthfully, since a plaintext key is
    on disk — so it reaches this verb instead of being turned away with
    "already hardened", which was the state with no CLI way out. Re-running
    ``harden`` re-wraps the LIVE key and removes the plaintext copy, which is
    exactly the repair. No separate ``--force`` flag is introduced: the
    condition it would guard is precisely "this store is not hardened right
    now", which is what this verb already means.

    The key is taken from :func:`resolve_master_key` rather than read straight
    off disk, so the one that actually opens the DATABASE is the one wrapped.
    On a damaged store the plaintext file is the live key and a stale wrapping
    sits beside it; wrapping the file blindly would be right by luck there and
    wrong after an interrupted rotation, where the live key is a staged one.
    """
    from local_operator.secrets.access import resolve_master_key
    from local_operator.secrets.keys import assert_key_of_record_invariant
    from local_operator.secrets.keys import key_mode as current_mode
    from local_operator.secrets.keys import key_of_record_inconsistency, wrap_master_key

    if current_mode() == "passphrase":
        _err("This store is already hardened. Use `lop secret unlock` to unlock it.")
        return 2
    repairing = key_of_record_inconsistency() is not None
    if repairing:
        _err(
            "This store has a plaintext master key beside a stale wrapped one — an "
            "earlier rotation undid its hardening. Re-wrapping the live key now."
        )
    # Read the key BEFORE prompting: a store that cannot be opened should fail
    # before the operator types a passphrase they then discover was pointless.
    key = resolve_master_key()
    passphrase = _read_passphrase("New passphrase for the secret store: ")
    if not passphrase:
        _err("Empty passphrase; nothing changed.")
        return 2
    if passphrase != _read_passphrase("Repeat the passphrase: "):
        _err("The passphrases did not match; nothing changed.")
        return 2
    wrap_master_key(None, key, passphrase)
    assert_key_of_record_invariant(None, "passphrase")
    _err(
        "Store hardened. The master key is now wrapped with your passphrase and the "
        "plaintext key file is gone.\n"
        "Run `lop secret unlock` once after each reboot; retrievals are silent after that.\n"
        "If you forget this passphrase the secrets cannot be recovered — there is no "
        "escrow copy."
    )
    return 0


def _unlock(args: argparse.Namespace) -> int:
    """Unlock a hardened store for this boot by unwrapping the key into the broker.

    Also tells the operator that this terminal gained standing, which is the
    part of the mechanism they can act on (review R9).
    """
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
    # **The grant is stated at the moment it starts applying (review R9).** The
    # operator has just given this terminal standing to read every secret for
    # as long as the shell lives, and that is a materially different mechanism
    # from "a script that runs `lop`" — anything here can talk to the broker
    # socket directly. Learning it from §9 of a design document is learning it
    # too late, so it is said here, in two lines, where the decision is made.
    _err("Unlocked. The broker holds the key in memory until it exits or the machine reboots.")
    _err(
        "This terminal is now authorized: anything you run in it can read every secret "
        "until this shell exits. Other terminals are not. Run `lop secret broker stop` to "
        "revoke it sooner."
    )
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

    # `is_running` propagates a version refusal (Q4), but here the only
    # question is whether the socket still answers AT ALL — a stale daemon that
    # has not died yet is still running for the purpose of waiting it out, and
    # it is precisely the daemon this verb was invoked to clear.
    def still_up() -> bool:
        try:
            return client.is_running(None)
        except SecretStoreError:
            return True

    deadline = time.monotonic() + _BROKER_STOP_TIMEOUT_S
    while time.monotonic() < deadline and still_up():
        time.sleep(0.05)
    if still_up():
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
        # A stale daemon holds the socket, so `ensure_broker` cannot start over
        # it and now says so instead of timing out (Q4). The refusal already
        # names the fix, and `restart` is a verb away.
        try:
            started = client.ensure_broker(None)
        except BrokerIncompatible as exc:
            _err(str(exc))
            return 2
        if started:
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
