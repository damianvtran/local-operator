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
                    free, so the kernel, not the holder, dropped the lock);
* ``nested``        a descendant of a holder ran under the holder's slot
                    (``holder`` = its pid) instead of queueing behind it.

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

#: The event names this trail writes. Documentation, not a whitelist: record()
#: accepts any name and describe() renders any event, so a reader built before
#: a name existed still shows the row. ``late-start`` is ``heavy run
#: --start-by`` refusing a run whose window passed while it queued (PR #4279);
#: hyphenated like ``stale-reclaim`` and like run_bounded's
#: ``termination_cause=late-start`` for the same breach, so one grep for
#: ``late-start`` finds it on either trail. ``nested`` is a descendant of a
#: holder running under that holder's slot instead of queueing behind it
#: (``heavy_lease.inherited_lease``, the 2026-11-20 self-deadlock); ``holder``
#: names the ancestor pid whose ``take`` covers it, and there is no ``release``
#: for it because no slot changed hands. Event names are trail VALUES that
#: readers match on, so they change only with every reader.
EVENTS = ("take", "release", "timeout", "stale-reclaim", "late-start", "nested")

#: The longest label part kept. Subcommands are short words; a long token in
#: subcommand position is far more likely to be an identifier or a secret.
_LABEL_PART_MAX = 32
#: The longest subcommand kept. Real subcommands (``install``, ``typecheck``,
#: ``run-script``) are well under this; a longer lowercase word in argv[1] is
#: more often an identifier, a hash or a key (PR #4264 R2/Q2).
_SUBCOMMAND_MAX = 20
#: What a subcommand looks like: a lowercase word (``test``, ``run``, ``build``,
#: ``nx``, ``s3``), with ``-``/``_`` only BETWEEN letter/digit runs. Uppercase or
#: mixed case, digits-first, ``.``, ``=``, ``/``, ``:`` and ``@`` are rejected on
#: purpose — that is the shape of tokens, file names, URLs, paths and KEY=VALUE.
_SUBCOMMAND = re.compile(r"[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*")
#: More digits than this in one word reads as an id or a key, not a verb:
#: subcommands carry at most a version-ish digit or two (``s3``, ``ec2``,
#: ``k8s``), while lowercase hex (``a1b2c3d4…``) or ``build20261119`` do not.
_SUBCOMMAND_MAX_DIGITS = 2
#: Well-known credential prefixes that are themselves lowercase and word-shaped
#: (``sk-live-abcdef`` passes the pattern above), so they are refused by name.
_SECRET_PREFIXES = (
    "sk-",
    "sk_",
    "pk-",
    "pk_",
    "rk_",
    "ghp_",
    "gho_",
    "ghs_",
    "ghu_",
    "ghr_",
    "github_pat_",
    "glpat-",
    "xoxb-",
    "xoxp-",
    "xoxa-",
    "xoxs-",
    "whsec_",
    "npm_",
)
#: Leading words that are setup, not the program the lease is for.
_SETUP_WORDS = frozenset({"cd", "export", "set", "source", ".", "pushd", "unset"})
#: How many nested ``bash -c '…'`` strings a label looks into.
_MAX_LABEL_DEPTH = 3


def audit_path(directory: Path) -> Path:
    return directory / AUDIT_FILENAME


def command_label(command: str, _depth: int = 0) -> str:
    """``argv[0]`` basename plus one subcommand word, from a shell command line.

    ``env GITHUB_TOKEN=x make test`` → ``make test``; ``cd web && pnpm build``
    → ``pnpm build``; ``tool --token [redacted] deploy`` → ``tool`` (an uppercase
    word is never a subcommand);
    ``curl https://u:p@host`` → ``curl``. Anything unparsable yields ``?``.

    The label names the HEAVY part of a line: ``grep -v x f | pytest tests/unit``
    is ``pytest``, and ``bash -c '<script>'`` is labelled by what its script
    runs. ``$(…)``/backtick bodies are masked first, so a substitution's inner
    commands never name the line. A line with no heavy segment keeps the first
    non-setup command.
    """
    from local_operator import heavy_lease

    try:
        # Top-level commands only: a `$(grep … | tr …)` body is not what runs.
        masked, _ = heavy_lease._mask_substitutions(command)
        segments = heavy_lease._segments(masked)
    except Exception:  # noqa: BLE001 - a label must never fail the lease
        segments = []
    if not segments:
        try:
            segments = [shlex.split(command)]
        except ValueError:
            return "?"
    heavy_at = _first_heavy_segment(segments)
    if heavy_at is not None:
        label = _heavy_segment_label(segments[heavy_at], _depth)
        if label is not None:
            return label
    for tokens in segments:
        label = _segment_label(tokens)
        if label is not None:
            return label
    return "?"


def _first_heavy_segment(segments: list[list[str]]) -> int | None:
    """Index of the first segment the lease classifier calls heavy (or long-lived)."""
    from local_operator import heavy_lease

    for index, tokens in enumerate(segments):
        try:
            if heavy_lease._classify_segment(list(tokens), None) is not None:
                return index
        except Exception:  # noqa: BLE001 - a label must never fail the lease
            continue
    return None


def _heavy_segment_label(tokens: list[str], depth: int) -> str | None:
    """The label of a heavy segment; a shell ``-c`` string is labelled by its script."""
    from local_operator import heavy_lease

    try:
        stripped = heavy_lease._strip_wrappers(list(tokens))
        if (
            depth < _MAX_LABEL_DEPTH
            and stripped
            and os.path.basename(stripped[0]) in heavy_lease._SHELLS
        ):
            script = heavy_lease._shell_c_string(stripped[1:])
            if script is not None:
                return command_label(script, depth + 1)
    except Exception:  # noqa: BLE001
        pass
    return _segment_label(tokens)


def _segment_label(tokens: list[str]) -> str | None:
    """One segment's label; None for an empty or setup segment (skip it)."""
    from local_operator import heavy_lease

    try:
        tokens = heavy_lease._strip_wrappers(list(tokens))
    except Exception:  # noqa: BLE001
        pass
    # Drop leading KEY=VALUE assignments the wrapper stripper left behind:
    # they are exactly where env secrets sit.
    while tokens and "=" in tokens[0] and not tokens[0].startswith(("-", "/", ".")):
        tokens = tokens[1:]
    if not tokens or tokens[0] in _SETUP_WORDS:
        return None
    head = os.path.basename(tokens[0])
    if not head or "=" in head or len(head) > _LABEL_PART_MAX:
        return "?"
    # Only argv[1], and only when it is word-shaped: the first non-flag
    # word anywhere would pick up a flag's VALUE (``--token [redacted] deploy``).
    sub = tokens[1] if len(tokens) > 1 else ""
    if _is_subcommand(sub):
        return f"{head} {sub}"
    return head


def _is_subcommand(word: str) -> bool:
    """Whether ``word`` is short and verb-shaped enough to log next to argv[0].

    Deliberately conservative: a false negative only drops a word from the
    label (``tool`` instead of ``tool frobnicate``), while a false positive
    writes a possible secret into a file that outlives the session.
    """
    if not word or len(word) > _SUBCOMMAND_MAX or not _SUBCOMMAND.fullmatch(word):
        return False
    if sum(ch.isdigit() for ch in word) > _SUBCOMMAND_MAX_DIGITS:
        return False
    return not word.startswith(_SECRET_PREFIXES)


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


#: Keys whose value is an epoch instant rather than a count or duration.
_EPOCH_KEYS = frozenset({"start_by"})


def describe(entry: dict[str, Any]) -> str:
    """One event as ``<event> key=value …`` (logfmt), for status surfaces.

    The keys are the JSONL field names and the event names are the log's own
    identifiers, so this rendering is a technical excerpt of the file, not
    prose: the surrounding sentence is the localized part.
    """
    parts = [str(entry.get("event", "?"))]
    # Absolute instants render as ISO UTC (``…Z``), the format the time column
    # beside each row already uses, so a reader compares ``start_by`` with that
    # column by eye. Only the excerpt changes: the JSONL keeps the raw epoch.
    # Durations (``*_s``) stay numbers.
    keys = [k for k in _DESCRIBE_ORDER if k in entry]
    keys += [k for k in entry if k not in keys and k not in ("ts", "event")]
    for key in keys:
        value = entry[key]
        if isinstance(value, list):
            text = ",".join(str(v) for v in value) or "-"
        elif key in _EPOCH_KEYS:
            text = iso_time(value)
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
