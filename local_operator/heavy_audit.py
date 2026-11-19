"""The heavy-job lease's audit trail: ``<lease dir>/heavy-lease.jsonl``.

WHY THIS EXISTS. The machine-wide heavy-job lease (:mod:`local_operator.heavy_lease`)
left nothing behind but ``slot-N.lock`` and the current holder's ``slot-N.json``,
which is deleted on release. So after the fact nobody could PROVE that two heavy
jobs serialized rather than overlapped (the 2026-11-19 11:50Z finding): the only
evidence was the absence of an OOM. This module appends one JSON line per lease
event so that question is answered by reading a file:

* ``take``          a slot was acquired (``waited_s`` = time spent queueing);
* ``release``       it was given back (``held_s``, and the command's ``exit`` code
                    when the caller knows it);
* ``timeout``       a waiter gave up when its queue bound ran out (no slot);
* ``stale-reclaim`` a slot was taken over from a holder that died without
                    releasing (its record was still on disk when the flock was
                    free, so the kernel, not the holder, dropped the lock).

Two intervals serialized iff, for each slot, every ``take`` is followed by its
``release`` (or a ``stale-reclaim``) before the next ``take``.

WHY A SEPARATE FILE, NOT THE MESH AUDIT STREAM. ``network/audit.py``'s
:class:`AuditLog` is the mesh's tamper-evident log: a hash-chained schema
(``lop.mesh.audit.v1``) whose events need a mesh actor/actor-kind, written in
batches through a buffered writer with publication tracking. A lease event has
no mesh actor, would sit in a batch buffer (the opposite of "prove it after a
kill -9"), and would put a hot local path behind mesh machinery. What IS reused
is its retention pattern: a byte cap on the live file (cf. ``AUDIT_MAX_BYTES``),
rotation into numbered generations, and pruning by generation count.

CONSTRAINTS. A write must never break, block or slow the lease:

* best-effort — every failure is swallowed (a read-only config dir, a full disk);
* one small line through one ``O_APPEND`` ``write(2)``, so concurrent writers
  from several sessions interleave whole lines, never torn ones;
* ``release`` is written AFTER the flock is dropped (see ``Lease.release``), and
  ``take`` is a single append after the slot record, so the slot is held no
  longer than before this file existed;
* rotation is attempted only under a NON-blocking flock on a sidecar lock file:
  whoever loses the race just appends to the file the winner left, and a slow
  rotation can never make an acquirer wait. Rotated generations are renamed,
  not gzipped (unlike the mesh log): compression would put a disk-bound copy on
  the lease's path, and the cap is small enough that compressing buys little.

PRIVACY. ``label`` is ``argv[0]``'s basename plus at most ONE subcommand-shaped
word (see :func:`command_label`) — never the full argv and never the environment,
because a heavy command line routinely carries tokens (a ``--token`` flag,
a ``*_TOKEN=`` prefix) and this file outlives the session.

USAGE. :func:`record` is called by ``heavy_lease`` only; readers use
:func:`tail` (``lop services heavy status``) and :func:`last_event` (the
``lop services status`` audit row).
"""

from __future__ import annotations

import json
import os
import re
import shlex
import time
from pathlib import Path
from typing import Any

try:  # POSIX only; on Windows the lease itself is unsupported.
    import fcntl
except ImportError:  # pragma: no cover - exercised on Windows only
    fcntl = None  # type: ignore[assignment]

AUDIT_FILENAME = "heavy-lease.jsonl"

#: Bytes in the live file before it rotates. Lease events are ~250 bytes, so
#: 1 MiB holds ~4000 events — weeks of heavy jobs on a busy node — while the
#: whole trail stays a few MiB. A constant, like the mesh log's cap: a knob here
#: would only ever be turned the wrong way.
AUDIT_MAX_BYTES = 1024 * 1024
#: Rotated generations kept after the live file (``heavy-lease.jsonl.1`` newest).
AUDIT_GENERATIONS = 3

EVENTS = ("take", "release", "timeout", "stale-reclaim")

#: The longest label part kept. Subcommands are short words; a long token in
#: subcommand position is far more likely to be an identifier or a secret.
_LABEL_PART_MAX = 32
#: What a subcommand looks like: a lowercase word (``test``, ``run``, ``build``,
#: ``nx``). Uppercase/mixed case, digits-first, ``=``, ``/``, ``:`` and ``@`` are
#: rejected on purpose — that is the shape of tokens, URLs, paths and KEY=VALUE.
_SUBCOMMAND = re.compile(r"[a-z][a-z0-9_.-]*")
#: Leading words that are setup, not the program the lease is for.
_SETUP_WORDS = frozenset({"cd", "export", "set", "source", ".", "pushd", "unset"})


def audit_path(directory: Path) -> Path:
    return directory / AUDIT_FILENAME


def command_label(command: str) -> str:
    """``argv[0]`` basename plus one subcommand word, from a shell command line.

    ``env GITHUB_TOKEN=x make test`` → ``make test``; ``cd web && pnpm build``
    → ``pnpm build``; ``tool --token SECRET deploy`` → ``tool`` (an uppercase
    word is never a subcommand);
    ``curl https://u:p@host`` → ``curl``. Anything unparsable yields ``?``.
    """
    from local_operator import heavy_lease

    try:
        segments = heavy_lease._segments(command)
    except Exception:  # noqa: BLE001 - a label must never fail the lease
        segments = []
    if not segments:
        try:
            segments = [shlex.split(command)]
        except ValueError:
            return "?"
    for tokens in segments:
        try:
            tokens = heavy_lease._strip_wrappers(list(tokens))
        except Exception:  # noqa: BLE001
            pass
        # Drop leading KEY=VALUE assignments the wrapper stripper left behind:
        # they are exactly where env secrets sit.
        while tokens and "=" in tokens[0] and not tokens[0].startswith(("-", "/", ".")):
            tokens = tokens[1:]
        if not tokens or tokens[0] in _SETUP_WORDS:
            continue
        head = os.path.basename(tokens[0])
        if not head or "=" in head or len(head) > _LABEL_PART_MAX:
            return "?"
        # Only argv[1], and only when it is word-shaped: the first non-flag
        # word anywhere would pick up a flag's VALUE (``--token abc deploy``).
        sub = tokens[1] if len(tokens) > 1 else ""
        if sub and len(sub) <= _LABEL_PART_MAX and _SUBCOMMAND.fullmatch(sub):
            return f"{head} {sub}"
        return head
    return "?"


def record(directory: Path, event: str, **fields: Any) -> None:
    """Append one event; never raises, never blocks on another writer."""
    try:
        entry: dict[str, Any] = {"ts": round(time.time(), 3), "event": event}
        entry.update({k: v for k, v in fields.items() if v is not None})
        line = (json.dumps(entry, separators=(",", ":"), sort_keys=False) + "\n").encode()
        path = audit_path(directory)
        _rotate_if_needed(path)
        fd = os.open(
            path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_CLOEXEC", 0), 0o600
        )
        try:
            os.write(fd, line)
        finally:
            os.close(fd)
    except Exception:  # noqa: BLE001 - the trail is evidence, never a gate
        pass


def _rotate_if_needed(path: Path) -> None:
    try:
        if path.stat().st_size < AUDIT_MAX_BYTES:
            return
    except OSError:
        return
    if fcntl is None:
        return
    lock = os.open(path.with_name(path.name + ".lock"), os.O_RDONLY | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return  # someone else is rotating; append to whatever file is live
        try:
            # Re-check under the lock: the winner of a race already rotated.
            if path.stat().st_size < AUDIT_MAX_BYTES:
                return
        except OSError:
            return
        # Shift .N-1 -> .N (the oldest is overwritten: the count prune), then
        # live -> .1. Renames only, so this costs a few metadata ops.
        for generation in range(AUDIT_GENERATIONS, 0, -1):
            source = path.with_name(f"{path.name}.{generation - 1}") if generation > 1 else path
            target = path.with_name(f"{path.name}.{generation}")
            try:
                os.replace(source, target)
            except FileNotFoundError:
                continue
    finally:
        os.close(lock)


def _read_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []


def tail(directory: Path, count: int) -> list[dict[str, Any]]:
    """The last ``count`` events, oldest first; malformed lines are skipped.

    Reads the live file and, when it alone is short, the newest rotated one, so
    a rotation that just happened does not blank ``status``.
    """
    if count <= 0:
        return []
    path = audit_path(directory)
    lines = _read_lines(path)
    if len(lines) < count:
        lines = _read_lines(path.with_name(f"{path.name}.1")) + lines
    events: list[dict[str, Any]] = []
    for raw in lines[-count * 2 :]:
        try:
            item = json.loads(raw)
        except ValueError:
            continue
        if isinstance(item, dict) and isinstance(item.get("event"), str):
            events.append(item)
    return events[-count:]


def last_event(directory: Path) -> dict[str, Any] | None:
    events = tail(directory, 1)
    return events[-1] if events else None


#: Field order for :func:`describe`; anything else in a line is appended after.
_DESCRIBE_ORDER = (
    "slot",
    "pid",
    "session",
    "label",
    "waited_s",
    "held_s",
    "exit",
    "memory_mb",
    "reclaimed_by",
    "holders",
)


def describe(entry: dict[str, Any]) -> str:
    """One event as ``<event> key=value …`` (logfmt), for status surfaces.

    The keys are the JSONL field names and the event names are the log's own
    identifiers, so this rendering is a technical excerpt of the file, not
    prose: the surrounding sentence is the localized part.
    """
    parts = [str(entry.get("event", "?"))]
    keys = [k for k in _DESCRIBE_ORDER if k in entry]
    keys += [k for k in entry if k not in keys and k not in ("ts", "event")]
    for key in keys:
        value = entry[key]
        if isinstance(value, list):
            text = ",".join(str(v) for v in value) or "-"
        else:
            text = str(value)
        if not text or any(c.isspace() for c in text) or '"' in text:
            text = json.dumps(text, ensure_ascii=False)
        parts.append(f"{key}={text}")
    return " ".join(parts)


def iso_time(ts: Any) -> str:
    """``2026-11-19T11:50:02Z`` for a line's ``ts`` (``?`` when unreadable)."""
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(ts)))
    except (TypeError, ValueError, OverflowError, OSError):
        return "?"
