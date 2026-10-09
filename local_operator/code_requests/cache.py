"""The fetch cache: what a forge last told us, and when we may ask again.

THREE LAYERS, one entry shape.

1. **Memory** — a per-host LRU of :data:`MEM_ENTRIES_PER_HOST` entries. The
   heaviest real session's scan found 158 distinct refs (design §D.2); 256
   holds that whole with headroom, and the eviction is per HOST so one noisy
   host cannot empty another's cache.
2. **Disk** — one file per ref at ``<config>/cache/code_requests/<host>/
   <project>__<n>.json`` (the design's path; the project is percent-encoded so
   a GitLab subgroup cannot nest directories). It carries the validators
   (``etag``/``last_modified``) so a RESTART revalidates with conditional
   requests instead of refetching blind. Writes are atomic (pid-suffixed temp
   + ``os.replace``, the ``transcript_index`` convention). Entries untouched
   for :data:`SWEEP_AGE_S` are swept opportunistically.
3. **Session dirty marks** — ``<config>/cache/code_requests/.dirty/
   <session_id>.json``. Written by the SESSION side (an acted event, a turn
   end, a wake/monitor delivery) and read by whichever process fetches for
   that session: the two are usually different processes (a runtime writes,
   the desktop backend refreshes), so the mark is a file, not memory.

WHAT A DIRTY MARK MEANS. "The next GET may revalidate this ref regardless of
its TTL" — it does not fetch anything by itself, and the design's words are
exact (''Events mark rows dirty; they do not fetch immediately''). Consumption
is on the refresh ATTEMPT: a failure is not re-armed here, it enters the
per-key backoff below, so a lying refresh cannot become a storm.

COOLING vs BACKOFF vs TTL (three throttles, three scopes, and they are not
interchangeable):

* **TTL** is per REF and per state — the steady-state pacing (design §D.3:
  open+CI-pending is 60 s on GitHub and 90 s on GitLab because a GitLab 304
  still costs quota, open+settled is 5 min, merged/closed is 24 h because
  post-merge comments still arrive).
* **Cooling** is per HOST, for a 403/429: until the ``Reset`` header (or
  ``Retry-After``), with an exponential fallback 30 s -> 15 min when neither
  header says anything usable. A cooling host is reported in the route payload
  (``cooling: {host: until}``) so a UI can say when it will retry.
* **Backoff** is per KEY, for a 5xx or a network failure: exponential 30 s ->
  15 min. One broken ref must not stall the others.

All three live in THIS process's memory except the TTL clock (stored on the
entry: ``checked_at``) and the dirty marks (files). Cross-process duplicates
(daemon + runtime) are accepted per the design: they cost GitHub nothing (304s
are free there) and GitLab's 2000/min budget absorbs them.

STDLIB ONLY, deliberately: this module is imported by the live hook path (the
acted-event dirty mark) and by cold readers, and neither may pull httpx or the
session runtime in. The adapters live beside the service, which is where the
network is.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import quote

from local_operator.code_requests.adapters.base import FetchOutcome
from local_operator.code_requests.refs import Ref

logger = logging.getLogger(__name__)

#: Bumped when the entry shape changes incompatibly; an older entry is ignored
#: (treated as absent) rather than misread.
ENTRY_SCHEMA = 1

#: Per-host memory LRU size (design §D.2).
MEM_ENTRIES_PER_HOST = 256

#: The entry bound: comments are trimmed to at most this many convention
#: comments of at most COMMENT_BODY_MAX each, which keeps a normal entry well
#: under the design's ~32 KB envelope. The bound is a sanity guard for a
#: pathological thread, not a routine cut — real entries measured well below.
COMMENT_KEEP_MAX = 40
COMMENT_BODY_MAX = 4 * 1024
ENTRY_SERIALIZED_MAX = 256 * 1024

#: TTLs (seconds), design §D.3. GitHub's own Cache-Control is max-age=60 and
#: is where the 60 comes from; GitLab's 90 s is the design's answer to its
#: 304s costing quota.
TTL_OPEN_PENDING_GITHUB_S = 60.0
TTL_OPEN_PENDING_GITLAB_S = 90.0
TTL_OPEN_SETTLED_S = 5 * 60.0
TTL_SETTLED_S = 24 * 3600.0

#: Cooling fallback growth for a 403/429 with no usable header: 30 s doubling
#: to 15 min. Backoff for 5xx/network is the same ladder on the key.
BACKOFF_BASE_S = 30.0
BACKOFF_MAX_S = 15 * 60.0

#: How old an entry must be before the opportunistic sweep removes it. Also
#: the maximum age a ``Reset`` header may claim (a bogus year-3000 reset must
#: not cool a host forever).
SWEEP_AGE_S = 30 * 24 * 3600.0
MAX_COOLING_S = 24 * 3600.0

_DIRTY_DIRNAME = ".dirty"
_HOST_SAFE = re.compile(r"[^a-z0-9.-]+")

_LOCK = threading.Lock()


def fetch_dir(config_dir: str | Path) -> Path:
    """The disk tier's root: ``<config>/cache/code_requests`` (the scan cache's home too)."""
    return Path(config_dir) / "cache" / "code_requests"


def _safe_host(host: str) -> str:
    """One path segment for a host, safe against traversal and dot-names.

    Hosts come from parsed URLs (lowercased, no port), so the only job here is
    defence in depth: strip anything but ``[a-z0-9.-]``, and refuse a leading
    dot (``.dirty``'s namespace). A host sanitized to empty becomes ``_host``.
    """
    name = _HOST_SAFE.sub("-", host.strip().lower())
    if not name:
        return "_host"
    if name.startswith("."):
        name = "_" + name.lstrip(".")
    return name


def entry_path(config_dir: str | Path, *, host: str, project: str, number: int) -> Path:
    filename = f"{quote(project, safe='')}__{int(number)}.json"
    return fetch_dir(config_dir) / _safe_host(host) / filename


def dirty_path(config_dir: str | Path, session_id: str) -> Path:
    return fetch_dir(config_dir) / _DIRTY_DIRNAME / f"{session_id}.json"


# ---------------------------------------------------------------------------
# Entry IO (memory + disk)
# ---------------------------------------------------------------------------

_MEMORY: dict[str, "OrderedDict[str, dict[str, Any]]"] = {}


def _memory_get(host: str, key: str) -> dict[str, Any] | None:
    with _LOCK:
        bucket = _MEMORY.get(host)
        if bucket is None:
            return None
        entry = bucket.get(key)
        if entry is None:
            return None
        bucket.move_to_end(key)
        return entry


def _memory_put(entry: Mapping[str, Any], *, host: str, key: str) -> None:
    with _LOCK:
        bucket = _MEMORY.setdefault(host, OrderedDict())
        bucket[key] = dict(entry)
        bucket.move_to_end(key)
        while len(bucket) > MEM_ENTRIES_PER_HOST:
            bucket.popitem(last=False)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _write_json(path: Path, payload: Mapping[str, Any]) -> bool:
    """Atomic write (pid-suffixed temp + replace). False when it could not land."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"))
        os.replace(tmp, path)
        return True
    except OSError:
        logger.debug("could not write code-request cache entry %s", path, exc_info=True)
        return False


def read_entry(config_dir: str | Path, ref: Ref) -> dict[str, Any] | None:
    """One ref's entry: memory first, then disk (which fills memory back)."""
    in_memory = _memory_get(ref.host, ref.key)
    if in_memory is not None and in_memory.get("schema") == ENTRY_SCHEMA:
        return in_memory
    data = _read_json(entry_path(config_dir, host=ref.host, project=ref.project, number=ref.number))
    if data is None or data.get("schema") != ENTRY_SCHEMA:
        return None
    _memory_put(data, host=ref.host, key=ref.key)
    return data


def write_entry(config_dir: str | Path, entry: Mapping[str, Any], *, ref: Ref) -> bool:
    """Bound, persist and memoize one entry. Disk first, memory second."""
    prepared = dict(entry)
    prepared["schema"] = ENTRY_SCHEMA
    prepared = _bound(prepared)
    ok = _write_json(
        entry_path(config_dir, host=ref.host, project=ref.project, number=ref.number), prepared
    )
    _memory_put(prepared, host=ref.host, key=ref.key)
    return ok


def _bound(entry: dict[str, Any]) -> dict[str, Any]:
    """Apply the size bounds: comment count, comment body length, total size.

    The comment arrays live INSIDE the piece mapping (``comments``/``reviews``
    on GitHub, ``notes`` on GitLab), and each is trimmed independently:
    bodies are capped, and over-cap lists keep the newest
    :data:`COMMENT_KEEP_MAX` (the round parser reads the newest passes for
    every lane decision, and old rounds are exactly what no reader needs).
    This is what keeps a normal entry far under :data:`ENTRY_SERIALIZED_MAX`;
    the last-resort pass below is for a pathological thread, not routine work.
    """
    pieces = entry.get("pieces")
    if isinstance(pieces, dict):
        for name, value in list(pieces.items()):
            if isinstance(value, list):
                pieces[name] = _trim_comments(value)
    try:
        size = len(json.dumps(entry, separators=(",", ":")))
    except (TypeError, ValueError):
        return entry
    if size <= ENTRY_SERIALIZED_MAX:
        return entry
    # Last-resort bound: drop the body of every comment but keep the parsed
    # lanes (which is what the surfaces render). A reader that wanted the
    # bodies gets the note instead of a blown entry.
    pieces = entry.get("pieces")
    if isinstance(pieces, dict):
        for name, value in pieces.items():
            if isinstance(value, list):
                pieces[name] = [
                    {**item, "body": ""} if isinstance(item, Mapping) else item for item in value
                ]
        entry["comments_truncated"] = True
    return entry


def _trim_comments(items: list[Any]) -> list[dict[str, Any]]:
    """One comment list bounded: body caps first, then the newest kept."""
    trimmed: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        body = str(item.get("body") or "")
        if len(body) > COMMENT_BODY_MAX:
            item = {**item, "body": body[:COMMENT_BODY_MAX] + "\n[truncated]"}
        trimmed.append(dict(item))
    if len(trimmed) > COMMENT_KEEP_MAX:
        trimmed = trimmed[-COMMENT_KEEP_MAX:]
    return trimmed


def drop_entry(config_dir: str | Path, ref: Ref) -> None:
    """Remove one entry from memory and disk (best-effort)."""
    with _LOCK:
        bucket = _MEMORY.get(ref.host)
        if bucket is not None:
            bucket.pop(ref.key, None)
    try:
        entry_path(config_dir, host=ref.host, project=ref.project, number=ref.number).unlink()
    except OSError:
        pass


def merge_pieces(
    stored: Mapping[str, Any] | None,
    outcome: FetchOutcome,
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    """Combine stored pieces with a fetch outcome: replaced + revalidated + kept.

    The rule is per ENDPOINT: a piece in ``outcome.pieces`` replaces; a piece
    in ``outcome.not_modified`` keeps the stored value verbatim; anything else
    the adapter did not mention keeps the stored value too (a partial fetch
    must not erase a piece it did not touch).
    """
    pieces: dict[str, Any] = {}
    stored_pieces = stored.get("pieces") if isinstance(stored, Mapping) else None
    if isinstance(stored_pieces, Mapping):
        pieces.update(stored_pieces)
    for piece, value in outcome.pieces.items():
        pieces[piece] = value
    validators: dict[str, dict[str, str]] = {}
    stored_validators = stored.get("validators") if isinstance(stored, Mapping) else None
    if isinstance(stored_validators, Mapping):
        for piece, value in stored_validators.items():
            if isinstance(value, Mapping):
                validators[str(piece)] = {str(k): str(v) for k, v in value.items()}
    for piece, value in outcome.validators.items():
        validators[piece] = {str(k): str(v) for k, v in value.items()}
    return pieces, validators


# ---------------------------------------------------------------------------
# Freshness (TTL) and dirty marks
# ---------------------------------------------------------------------------


def ttl_seconds(entry: Mapping[str, Any] | None, *, forge: str) -> float | None:
    """The ref's steady-state revalidation interval, from its stored state.

    ``None`` means "never on a TTL" — which only the never-fetched case hits
    (the caller fetches it on first need, not on a timer).
    """
    if entry is None:
        return None
    state = str(entry.get("state") or "")
    if state in ("merged", "closed"):
        return TTL_SETTLED_S
    ci = entry.get("ci")
    ci_status = str(ci.get("status") or "") if isinstance(ci, Mapping) else ""
    if ci_status == "pending":
        return TTL_OPEN_PENDING_GITLAB_S if forge == "gitlab" else TTL_OPEN_PENDING_GITHUB_S
    return TTL_OPEN_SETTLED_S


def is_expired(entry: Mapping[str, Any] | None, *, forge: str, now: float | None = None) -> bool:
    """Whether the entry's TTL has elapsed since its last successful check."""
    if entry is None:
        return True
    moment = time.time() if now is None else now
    checked = entry.get("checked_at")
    if not isinstance(checked, (int, float)):
        return True
    ttl = ttl_seconds(entry, forge=forge)
    if ttl is None:
        return True
    return (moment - float(checked)) >= ttl


def mark_dirty(
    config_dir: str | Path,
    session_id: str,
    *,
    keys: Iterable[str] = (),
    all_rows: bool = False,
) -> bool:
    """Record that ''this session's rows may have moved''; no fetch happens here.

    Read-modify-write so an acted event's key and a turn-end's all-flag cannot
    clobber one another; best-effort on failure (a dirty mark is an
    optimisation over the TTL, never correctness).

    An ``all_rows`` mark with no keys is a sweep over whatever the session has,
    so it is a no-op for a session with no rows: writing it for every session
    on every turn end would litter the store with files no reader ever needs.
    """
    if not session_id:
        return False
    if all_rows and not keys:
        from local_operator.code_requests import ledger

        entry = ledger.read_index(config_dir, session_id)
        if not entry or not entry.get("rows"):
            return False
    path = dirty_path(config_dir, session_id)
    try:
        current = _read_json(path) or {}
        merged_keys = [str(item) for item in current.get("keys") or [] if str(item)]
        for key in keys:
            text = str(key or "")
            if text and text not in merged_keys:
                merged_keys.append(text)
        payload = {
            "keys": merged_keys[-512:],
            "all": bool(all_rows or current.get("all")),
            "at": round(time.time(), 3),
        }
        return _write_json(path, payload)
    except Exception:  # noqa: BLE001 - a mark never breaks a turn
        logger.debug("could not mark code-request rows dirty", exc_info=True)
        return False


def read_dirty(config_dir: str | Path, session_id: str) -> dict[str, Any]:
    """The session's dirty marks, normalised: ``{keys: [...], all: bool, at: float}``."""
    data = _read_json(dirty_path(config_dir, session_id)) or {}
    keys = [str(item) for item in data.get("keys") or [] if str(item)]
    at = data.get("at")
    return {
        "keys": keys,
        "all": bool(data.get("all")),
        "at": float(at) if isinstance(at, (int, float)) else 0.0,
    }


def clear_dirty(
    config_dir: str | Path,
    session_id: str,
    *,
    keys: Iterable[str] | None = None,
    since: float | None = None,
) -> None:
    """Consume consumed keys, or the whole mark when ``keys is None``.

    Called on the refresh ATTEMPT (not its success): a failed refresh lands in
    the per-key backoff, and re-arming the dirty mark would bypass that bound.

    ``since`` guards the consume-then-a-new-mark race: the file is only
    removed/edited when its own ``at`` is not NEWER than the snapshot the
    refresher took, so a mark that landed while the refresh ran survives.
    """
    path = dirty_path(config_dir, session_id)
    try:
        data = _read_json(path)
        if data is None:
            return
        if since is not None:
            at = data.get("at")
            if isinstance(at, (int, float)) and at > since:
                return
        if keys is None:
            path.unlink()
            return
        drop = {str(item) for item in keys}
        remaining = [str(item) for item in data.get("keys") or [] if str(item) not in drop]
        if not remaining and not data.get("all"):
            path.unlink()
            return
        data["keys"] = remaining
        _write_json(path, data)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Cooling (per host) and backoff (per key)
# ---------------------------------------------------------------------------

_COOLING: dict[str, float] = {}
_COOLING_STEP: dict[str, int] = {}
_KEY_BACKOFF: dict[str, "tuple[float, float]"] = {}


def cooling_until(host: str, *, now: float | None = None) -> float | None:
    """When ``host`` becomes usable again, or ``None`` while it is usable."""
    moment = time.time() if now is None else now
    until = _COOLING.get(host)
    if until is None or until <= moment:
        return None
    return until


def cooling_map() -> dict[str, float]:
    """Every cooling host and its until-epoch, for the route payload."""
    moment = time.time()
    return {host: until for host, until in sorted(_COOLING.items()) if until > moment}


def note_rate_limited(
    host: str,
    *,
    reset_at: float | None = None,
    retry_after: float | None = None,
    now: float | None = None,
) -> float:
    """Cool ``host`` after a 403/429; returns the until-epoch.

    Priority: ``Reset`` (epoch seconds; used only when it is plausibly in the
    future and within :data:`MAX_COOLING_S`), then ``Retry-After`` seconds,
    then the exponential fallback 30 s doubling to 15 min. The step survives
    within the process so repeated offences lengthen the pause; a success
    clears it.
    """
    moment = time.time() if now is None else now
    if reset_at is not None and moment < reset_at <= moment + MAX_COOLING_S:
        until = float(reset_at)
    elif retry_after is not None and 0 < retry_after <= MAX_COOLING_S:
        until = moment + float(retry_after)
    else:
        step = _COOLING_STEP.get(host, 0)
        until = moment + min(BACKOFF_BASE_S * (2**step), BACKOFF_MAX_S)
        _COOLING_STEP[host] = min(step + 1, 20)
    _COOLING[host] = until
    return until


def note_host_success(host: str) -> None:
    """A usable response from ``host`` clears its cooling and reset its step."""
    _COOLING.pop(host, None)
    _COOLING_STEP.pop(host, None)


def key_backoff_until(key: str, *, now: float | None = None) -> float | None:
    moment = time.time() if now is None else now
    entry = _KEY_BACKOFF.get(key)
    if entry is None or entry[0] <= moment:
        return None
    return entry[0]


def note_key_failure(key: str, *, now: float | None = None) -> float:
    """Raise one key's exponential backoff after a 5xx or network failure.

    The stored pair is ``(until, delay)``: keeping the DELAY beside the
    deadline is what makes the next offence double it, without a parallel step
    dict to prune. 30 s -> 15 min, then flat.
    """
    moment = time.time() if now is None else now
    previous = _KEY_BACKOFF.get(key)
    if previous is not None and previous[0] > moment:
        delay = min(BACKOFF_MAX_S, max(BACKOFF_BASE_S, previous[1] * 2))
    else:
        delay = BACKOFF_BASE_S
    until = moment + delay
    _KEY_BACKOFF[key] = (until, delay)
    return until


def clear_key_backoff(key: str) -> None:
    _KEY_BACKOFF.pop(key, None)


def _reset_for_tests() -> None:
    """Drop every in-memory layer. Tests that force cooling/backoff call this."""
    with _LOCK:
        _MEMORY.clear()
    _COOLING.clear()
    _COOLING_STEP.clear()
    _KEY_BACKOFF.clear()


# ---------------------------------------------------------------------------
# The opportunistic sweep
# ---------------------------------------------------------------------------

_SWEPT_AT = 0.0


def sweep(
    config_dir: str | Path, *, now: float | None = None, min_interval_s: float = 3600.0
) -> int:
    """Remove entries untouched for :data:`SWEEP_AGE_S`, at most hourly.

    Bounded on purpose: the walk stops after :data:`_SWEEP_LIMIT` files so a
    pathological store cannot make a routine fetch pay a full-tree scan. The
    sweep is an optimisation; skipping files is always safe.
    """
    global _SWEPT_AT
    moment = time.time() if now is None else now
    if moment - _SWEPT_AT < min_interval_s:
        return 0
    _SWEPT_AT = moment
    removed = 0
    seen = 0
    root = fetch_dir(config_dir)
    try:
        hosts = [item for item in root.iterdir() if item.is_dir() and not item.name.startswith(".")]
    except OSError:
        return 0
    for host_dir in hosts:
        try:
            files = list(host_dir.iterdir())
        except OSError:
            continue
        for path in files:
            seen += 1
            if seen > _SWEEP_LIMIT:
                return removed
            try:
                stat = path.stat()
            except OSError:
                continue
            if moment - stat.st_mtime > SWEEP_AGE_S:
                try:
                    path.unlink()
                    removed += 1
                except OSError:
                    continue
    return removed


_SWEEP_LIMIT = 2000


__all__ = [
    "COMMENT_BODY_MAX",
    "COMMENT_KEEP_MAX",
    "ENTRY_SCHEMA",
    "MEM_ENTRIES_PER_HOST",
    "clear_dirty",
    "clear_key_backoff",
    "cooling_map",
    "cooling_until",
    "dirty_path",
    "drop_entry",
    "entry_path",
    "fetch_dir",
    "is_expired",
    "key_backoff_until",
    "mark_dirty",
    "merge_pieces",
    "note_host_success",
    "note_key_failure",
    "note_rate_limited",
    "read_dirty",
    "read_entry",
    "sweep",
    "ttl_seconds",
    "write_entry",
]
