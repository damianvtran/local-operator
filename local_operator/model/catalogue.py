"""Disk-cached model catalogue for providers with no static registry entry.

WHY THIS EXISTS
---------------
``model/registry.py`` hard-codes windows and prices for providers whose
catalogues are small and stable (Anthropic, OpenAI, Kimi, ...). Aggregators
cannot work that way: OpenRouter alone routes hundreds of models that change
weekly, so its registry entry is a placeholder with ``context_window = -1``
("no data") and zero prices.

``configure_model`` has always been able to read the real numbers from the
provider's ``list_models()`` payload, but only when handed a
``model_info_client`` — and the session factory never handed it one. Every
OpenRouter and Radient session therefore ran on the fallbacks in
``configure.py``, which is wrong in three ways that all matter at runtime:

* **Context window** fell back to ``UNKNOWN_CONTEXT_WINDOW`` (128k). Auto
  compaction derives its threshold from the window, so a 1M-context model
  compacted at ~102k instead of ~600k (`min(0.8 * window, 600_000)`, the cap in
  `compaction/thresholds.py`) — a ~5.9x premature summarisation of history the
  model could still hold, on every long session.
* **Prices** stayed 0.0, so the status band could only report ``$—``. Cost is
  one of the few numbers an operator steers by mid-task.
* **``supports_prompt_cache``** stayed False, which gates ``cache_control``
  emission. Models reached THROUGH OpenRouter that need explicit breakpoints
  (the Anthropic family) silently never got them.

WHY IT IS CACHED
----------------
The fix is to call the listing — but a naive call adds a synchronous HTTP
round trip to every session start, and makes an offline start fail for a
reason the user cannot act on. Model catalogues change on the order of days,
so the payload is cached on disk with a TTL and a stale entry is preferred to
no entry:

1. Fresh cache (< soft TTL)       -> use it, no network.
2. Ageing cache (soft <= age < hard TTL) -> use it NOW, and refresh it in a
   background thread (stale-while-revalidate). The calling path pays nothing;
   the NEXT call sees the new document. This is what keeps "a model released
   today is offered today" true without putting a request on a session start
   or a TUI repaint. See :func:`read_listing` and :func:`_schedule_revalidate`.
3. Expired or missing (>= hard TTL) -> fetch synchronously, rewrite, use it.
4. Fetch fails but a stale cache exists -> use the stale copy. A window that
   is a week old is enormously better than pretending a 1M model holds 128k.
5. Fetch fails with no cache      -> return None; the caller keeps its static
   fallbacks. A session MUST still start with no network.

The cache holds the RAW payload under a key the caller chooses, so mapping stays
the caller's business and one provider can hold differently-shaped documents
without either overwriting the other.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("local_operator.model.catalogue")

#: The HARD TTL: how old a document may be before a read blocks on a fetch.
#: Model listings move on the order of days; a week would hide a genuinely new
#: model for too long, and anything shorter puts a synchronous request on the
#: boot path more often for no new information. Also the width of the in-process
#: memo bucket in ``configure._resolve_model_info_cached``, so keep the name.
DEFAULT_TTL_S = 24 * 60 * 60

#: The SOFT TTL: age past which a served document is ALSO refreshed in the
#: background. An hour bounds "released today, offered today" for the next call
#: without putting a request on the calling path; below it a listing endpoint
#: would be asked repeatedly for information that changes on the order of days.
#: The incident this exists for: a 22h-old Anthropic document (inside the hard
#: TTL, so never refetched) hid a model published that morning from ``/model``
#: and from the cost band for a whole working day.
SOFT_TTL_S = 60 * 60

#: A document younger than this that lacks a requested id is believed: the id is
#: genuinely unknown, not newly released. Bounds the miss-triggered refetch
#: (see ``discovery.available_models(want_id=...)``) so a typo cannot refetch on
#: every resolution — one refetch per key per ten minutes, however many ids miss.
MISS_REFETCH_MIN_AGE_S = 10 * 60

#: Per-process floor between background attempts on one key, so an offline
#: machine does not spawn a thread every time a stale document is served. The
#: sync miss path is unaffected: it has its own stale-beats-absent rule.
REVALIDATE_BACKOFF_S = 5 * 60


def default_cache_dir() -> Path:
    """Same cache root the skills index uses, so there is one place to clear."""
    return Path("~/.local-operator/cache").expanduser()


#: Document names written by earlier layouts of this cache. Two generations are
#: dead: ``<provider>.models.json``, from when the caller's key was the bare
#: provider id, and ``<provider>.models.models.json``, from when the caller's key
#: and :func:`_cache_path` each appended ``.models``. One pattern matches both,
#: and neither can match a current ``<key>.json`` whose key ends in ``.listing``.
_LEGACY_DOCUMENT_GLOB = "*.models.json"


def _cache_path(key: str, cache_dir: Path | None) -> Path:
    """The document for ``key``; the key is the WHOLE stem, suffix added once.

    This used to append ``.models`` on top of a caller key that already ended in
    ``.models``, which is how ``openrouter.models.models.json`` reached disk.
    """
    return (cache_dir or default_cache_dir()) / f"{key}.json"


def purge_legacy_documents(cache_dir: Path | None = None) -> None:
    """Delete catalogue documents no reader will ever open again.

    Measured on one install: ``openrouter.models.json`` (569,845 bytes) and
    ``radient.models.json`` (239,019 bytes) left behind by a document-name
    change, beside the names in use. A cache with no index has no other way to
    notice that an old entry became unreachable, so without this it is ~800 KB
    per install that nothing reads and nothing ever removes.

    Stateless on purpose -- no "migration done" marker. The legacy patterns
    cannot match a current document name, so the sweep is idempotent by
    construction rather than by remembering that it ran, and it stays correct if
    the cache directory is cleared or copied between machines. A missing
    directory globs to nothing rather than raising, so a first run is a no-op.
    """
    directory = cache_dir or default_cache_dir()
    try:
        for stale in directory.glob(_LEGACY_DOCUMENT_GLOB):
            stale.unlink(missing_ok=True)
    except OSError as exc:  # pragma: no cover - read-only or unreadable cache dir
        # Reclaiming disk is never worth failing a session start over.
        logger.debug("could not purge legacy catalogue documents: %s", exc)


def _read_cache(path: Path) -> tuple[dict[str, Any] | None, float]:
    """Return ``(payload, age_seconds)``; ``(None, inf)`` when unusable.

    A corrupt cache is treated as absent rather than raised: this is an
    optimisation store, and a half-written file from a killed process must not
    stop a session from starting.

    A timestamp in the FUTURE also counts as unusable. Clamping the age to zero
    instead (the obvious move) makes such an entry look permanently fresh, so a
    single clock skew — an NTP correction, a suspend/resume, a file copied from
    another machine — would pin the catalogue forever with no way to recover
    short of deleting the file by hand. Refetching once is the cheap direction
    to be wrong in.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        payload = raw["payload"]
        fetched_at = float(raw["fetched_at"])
    except (OSError, ValueError, KeyError, TypeError):
        return None, float("inf")
    if not isinstance(payload, dict):
        return None, float("inf")
    age = time.time() - fetched_at
    # `not (age >= 0)` rather than `age < 0`, so NaN lands here too. A NaN
    # `fetched_at` is reachable: `json.loads` accepts the bare literal, and
    # `float("NaN")` accepts the string. Under `age < 0` it fell through as a NaN
    # age, which then compared False against the TTL and so happened to be
    # treated as stale — the right outcome by accident of IEEE comparison rather
    # than by this function's stated rule, and it printed "using stale catalogue
    # (nans old)" on the way past.
    if not (age >= 0):
        return payload, float("inf")
    return payload, age


def _write_cache(path: Path, payload: dict[str, Any]) -> None:
    """Persist a payload, best-effort.

    Written to a temp file and renamed, because ``rename`` within a directory is
    atomic: a concurrent reader sees either the old document or the new one,
    never a partial write.

    The temp name comes from :func:`tempfile.mkstemp`, which is the only way to
    get a name no other writer holds. Every derived name has a sharing bug of
    the same shape: the obvious ``path.with_suffix('.tmp')`` is shared by every
    concurrent writer, and adding the PID still leaves it shared by every THREAD
    in one process — which is the case that actually occurs here, since
    ``configure_model`` is called from request handlers in a long-lived server.
    Two writers on one temp file interleave their bytes and then rename the
    result into place, turning the atomic rename into a guarantee that the
    corruption is delivered intact. It parses as JSON and passes every shape
    check, so the damage is silently-wrong prices and context windows served for
    the full TTL. A ~190 KB listing spans many write syscalls, so this is
    routine rather than a narrow race.

    ``mkstemp`` creates in the target directory (not the system temp dir) so the
    rename stays within one filesystem, and at mode 0600 — the caller's umask is
    reapplied after the rename, because a cache of public model metadata that
    only the owner can read would break a shared install.
    """
    fd: int | None = None
    tmp: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, name = tempfile.mkstemp(dir=str(path.parent), prefix=f"{path.name}.", suffix=".tmp")
        fd, tmp = handle, Path(name)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            fd = None  # fdopen owns it now; closing the wrapper closes the fd
            json.dump({"fetched_at": time.time(), "payload": payload}, stream)
        os.chmod(tmp, 0o644 & ~_umask())
        tmp.replace(path)
        tmp = None
    except OSError as exc:  # pragma: no cover - disk full, read-only home
        logger.debug("could not cache %s catalogue: %s", path.name, exc)
    finally:
        # A failed write can strand the temp file; it would otherwise accumulate
        # one per failing start.
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if tmp is not None:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass


def _umask() -> int:
    """The process umask, read without leaving it changed.

    ``os.umask`` has no getter: the only way to read it is to set it, which is
    why this is a helper rather than an inline call. Racy in principle against
    another thread setting the umask in the same instant; that is a
    process-global setting no library should be touching, and the consequence
    here is at worst one cache file with slightly different permissions.
    """
    current = os.umask(0o022)
    os.umask(current)
    return current


#: How long a catalogue fetch lease stands before a peer may steal it. A
#: listing fetch is bounded by the caller's own timeout (tens of seconds at
#: worst), so a lease comfortably longer covers a slow response, and a
#: process that dies mid-fetch cannot strand the document: the lease lapses
#: and the next session fetches.
_LISTING_LEASE_S = 60.0

#: How long a lease LOSER waits for the winner's write before re-reading.
#: Bounded hard because the caller can be a session boot: this is a
#: courtesy pause, not a barrier.
_LISTING_LEASE_WAIT_S = 2.0


class _ListingFetchLease:
    """A best-effort cross-process lock on one catalogue document.

    Modelled on the usage cache's SQLite lease but lockfile-based, because
    this module's store is plain files and pulling SQLite in here would put
    a database dependency on the model-picker path. ``O_CREAT | O_EXCL`` is
    the atomic take; the file carries the holder's identity and an expiry
    so a crashed holder cannot block the next refresh. EVERY failure mode
    degrades to "no lease": an unwritable directory, a race lost, an
    unreadable holder file — the caller then just fetches, which is the
    pre-lease behaviour and always correct, merely less polite to the
    endpoint.
    """

    def __init__(self, document: Path) -> None:
        self._path = document.with_name(document.name + ".fetching")
        self._token = f"{os.getpid()}:{uuid.uuid4().hex[:8]}"
        self._held = False

    def acquire(self) -> bool:
        """Take the lease if free or expired. True means THIS process fetches."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self._path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            # Standing lease. Legitimate only while unexpired; a stale one
            # (holder crashed mid-fetch) is stolen by replacing it, where the
            # same O_EXCL race decides the single winner among the stealers.
            try:
                raw = self._path.read_text(encoding="utf-8", errors="replace")
                expires = float(json.loads(raw).get("expires_at", 0.0))
            except (OSError, ValueError):
                expires = 0.0
            if expires > time.time():
                return False
            try:
                self._path.unlink()
            except OSError:
                return False
            return self.acquire()
        except OSError:
            # Unwritable cache dir or worse: fetch without a lease. Correct,
            # just unpolite.
            return True
        try:
            payload = json.dumps(
                {"holder": self._token, "expires_at": time.time() + _LISTING_LEASE_S}
            )
            os.write(fd, payload.encode("utf-8"))
        finally:
            os.close(fd)
        self._held = True
        return True

    def await_peer_briefly(self) -> None:
        """Give the lease holder a moment, then return whatever is on disk.

        Polls the DOCUMENT, not the lease: the winner's atomic rename is the
        event we are waiting for, and a holder that fails leaves the lease
        standing until expiry — waiting for the lease would mean waiting out
        the full lease TTL on every failed peer fetch.
        """
        document = self._path.with_name(self._path.name[: -len(".fetching")])
        deadline = time.monotonic() + _LISTING_LEASE_WAIT_S
        while time.monotonic() < deadline:
            time.sleep(0.05)
            if document.exists():
                try:
                    if time.time() - document.stat().st_mtime < _LISTING_LEASE_WAIT_S * 2:
                        return
                except OSError:
                    return

    def release(self) -> None:
        """Free the lease if this process holds it."""
        if not self._held:
            return
        self._held = False
        try:
            raw = self._path.read_text(encoding="utf-8", errors="replace")
            if json.loads(raw).get("holder") == self._token:
                self._path.unlink()
        except (OSError, ValueError):
            # Already gone or unreadable: nothing to free.
            pass


@dataclasses.dataclass(frozen=True)
class Listing:
    """What :func:`read_listing` found, and how.

    ``fetched`` is how a live answer is told from a cached one — the whole
    difference between discovery's ``"ok"`` and ``"cached"`` statuses. It used
    to be smuggled out of the fetch thunk through a ``nonlocal`` flag, which
    could not express "the fetch was attempted and FAILED" (the ``"stale"``
    status): a thunk that raises never gets to set anything. ``failed`` carries
    that now. ``refreshing`` says a background revalidation was scheduled by
    THIS call, which tests use to join it and callers may use to annotate.
    """

    payload: dict[str, Any] | None
    #: ``inf`` when there is no usable document.
    age_s: float
    #: A live fetch produced ``payload`` on this call.
    fetched: bool = False
    #: A live fetch was attempted on this call and raised; ``payload`` is the
    #: stale document (or ``None`` when there was none to fall back on).
    failed: bool = False
    #: A background revalidation thread was started by this call.
    refreshing: bool = False


#: Background revalidations in flight in THIS process, keyed by document PATH
#: (not by key: two cache directories are two documents). A second read of the
#: same ageing document must not start a second thread; the cross-process lease
#: already dedupes between processes but not between the threads of one, and
#: the TUI resolves the same provider from the picker worker and the 1 Hz cost
#: poll at once.
_revalidating: set[str] = set()
#: ``time.monotonic()`` of the last background attempt per document, for the backoff.
_last_attempt: dict[str, float] = {}
#: Live revalidation threads, so tests can join them instead of sleeping.
_threads: dict[str, threading.Thread] = {}
_revalidate_lock = threading.Lock()


def _revalidation_threads() -> list[threading.Thread]:
    """The background revalidations currently alive — a TEST hook.

    AGENTS.md forbids waiting on the clock for asynchronous work; a test that
    wants to see the rewritten document joins these instead.
    """
    with _revalidate_lock:
        return [thread for thread in _threads.values() if thread.is_alive()]


def _schedule_revalidate(
    key: str, fetch: Callable[[], dict[str, Any]], cache_dir: Path | None
) -> bool:
    """Refresh ``key`` off the calling path. True when a thread was started.

    Plain ``threading.Thread``, not the asyncio loop: this module is synchronous
    and is entered from the CLI, the server, ``asyncio.to_thread`` workers and
    ``run_in_executor`` threads alike, so there is no loop to hand the work to
    that every caller has. ``refresh_model_info_background`` in ``configure`` is
    the same shape one layer up. Daemon, so a short-lived ``lop --model ...``
    process is not held open by a listing fetch: a daemon killed mid-write leaves
    nothing half-written (``_write_cache`` is mkstemp+rename) and at worst a lease
    that lapses in ``_LISTING_LEASE_S`` — the existing crash contract.

    Two dedupes. In-process: a key already in flight, or attempted within
    ``REVALIDATE_BACKOFF_S``, gets no new thread — an offline machine serving a
    stale document on every repaint must not spawn a thread per repaint.
    Cross-process: the thread takes the same ``_ListingFetchLease`` as the sync
    path and, unlike it, simply EXITS when a peer holds it; there is nothing to
    wait for because the caller already has its answer.
    """
    path = _cache_path(key, cache_dir)
    slot = str(path)
    now = time.monotonic()
    with _revalidate_lock:
        if slot in _revalidating:
            return False
        last = _last_attempt.get(slot)
        if last is not None and now - last < REVALIDATE_BACKOFF_S:
            return False
        _revalidating.add(slot)
        _last_attempt[slot] = now

    def run() -> None:
        lease = _ListingFetchLease(path)
        try:
            if not lease.acquire():
                return
            try:
                fresh = fetch()
            except Exception as exc:  # noqa: BLE001 - the stale document stays
                logger.debug("background %s revalidation failed: %s", key, exc)
                return
            _write_cache(path, fresh)
        finally:
            lease.release()
            with _revalidate_lock:
                _revalidating.discard(slot)
                _threads.pop(slot, None)

    thread = threading.Thread(target=run, name=f"catalogue-revalidate:{key}", daemon=True)
    with _revalidate_lock:
        _threads[slot] = thread
    thread.start()
    return True


def peek_listing(key: str, *, cache_dir: Path | None = None) -> Listing:
    """The document on disk, read-only: no fetch, no revalidation, no sweep.

    For a fetch thunk that wants to send ``If-None-Match``: it reads the previous
    payload's stored ETag through this and, on a 304, hands the same payload
    back so ``_write_cache`` re-stamps ``fetched_at`` — which is exactly
    "validated just now". Kept separate from :func:`read_listing` so the thunk
    cannot recurse into the reader that is calling it.
    """
    payload, age = _read_cache(_cache_path(key, cache_dir))
    return Listing(payload=payload, age_s=age)


def read_listing(
    key: str,
    fetch: Callable[[], dict[str, Any]],
    *,
    soft_ttl_s: float = SOFT_TTL_S,
    ttl_s: float = DEFAULT_TTL_S,
    cache_dir: Path | None = None,
    refetch_if: Callable[[dict[str, Any], float], bool] | None = None,
    revalidate: Callable[[], dict[str, Any]] | None = None,
) -> Listing:
    """A provider's ``list_models()`` payload, stale-while-revalidate.

    ``key`` names the document, not the provider: the caller owns the naming, so
    two differently-shaped listings for one provider cannot land on one file.

    ``fetch`` returns the payload as a plain dict (``model_dump()`` of the
    client's response). ``payload`` is None only when there is no cache AND the
    fetch failed, which is the one case where the caller must keep its static
    fallbacks.

    ``revalidate`` is the thunk the BACKGROUND refresh runs; it defaults to
    ``fetch``. They differ in budget, not in what they fetch: ``fetch`` runs on
    the calling path and carries the caller's ceiling (2 s from a repaint), while
    the background thread is off-path and may take the provider's full default.
    A background thunk that inherited the on-path budget failed every time on a
    link slower than that budget, backed off five minutes, and the document only
    ever advanced through the 24 h synchronous path — the pre-revalidation
    behaviour, silently restored.

    State machine, by document age:

    * ``age < soft_ttl_s`` — served, nothing else happens.
    * ``soft_ttl_s <= age < ttl_s`` — served IMMEDIATELY, and a background
      refresh is scheduled (:func:`_schedule_revalidate`). The calling path never
      waits on the network for a document it already has.
    * ``age >= ttl_s`` or no document — the synchronous fetch, under the
      cross-process lease. This is the offline/cold contract and is unchanged.

    ``soft_ttl_s >= ttl_s`` disables the background state entirely, which is
    how :func:`cached_listing` keeps its original three-state behaviour.

    ``refetch_if(payload, age_s)`` lets the caller declare a document inside
    the TTL EXPIRED anyway — discovery's "the id I am about to resolve is not
    in this document and the document is old enough to predate it" rule. It is
    a predicate here rather than a second call after the fact because the two
    would race: a document in the soft window has ALREADY had a background
    refresh scheduled by the time a caller could inspect it, and a synchronous
    refetch behind that loses the lease to its own process's thread, waits on
    it, and cannot tell the result apart from a plain cache hit. Deciding
    before either path starts means one fetch, on the calling thread, with a
    ``fetched``/``failed`` flag that means what it says.

    A cross-process fetch lease guards the miss path: several lop sessions
    cold-starting together all miss in the same instant, and without the
    lease each fires its own live listing request at the same public
    endpoint — the thundering herd that earns a 429 for data one request
    would have fetched. See :class:`_ListingFetchLease` for the degradation
    contract; nothing here can block a session start.
    """
    # Swept here because this module owns the naming scheme: a caller cannot know
    # which document names went dead, and a dead name has no reader left to
    # notice it. Costs one directory read once the sweep has nothing to match.
    purge_legacy_documents(cache_dir)
    path = _cache_path(key, cache_dir)
    payload, age = _read_cache(path)
    if payload is not None and age < ttl_s:
        if refetch_if is None or not refetch_if(payload, age):
            refreshing = False
            if age >= soft_ttl_s:
                refreshing = _schedule_revalidate(key, revalidate or fetch, cache_dir)
            return Listing(payload=payload, age_s=age, refreshing=refreshing)
        logger.debug("%s catalogue (%.0fs old) declared expired by its reader", key, age)

    # CROSS-PROCESS FETCH LEASE (see the class docstring): the winner fetches
    # and writes; losers give the winner a brief window, re-read, and serve
    # whatever is on disk — stale included, which is already this module's
    # stale-beats-absent rule. Every failure degrades to fetching.
    lease = _ListingFetchLease(path)
    if not lease.acquire():
        lease.await_peer_briefly()
        payload, age = _read_cache(path)
        if payload is not None:
            return Listing(payload=payload, age_s=age)
        # No document even after the peer's window: fall through and fetch.
        # The lease is lost or stale; a cold machine must still start.

    try:
        fresh = fetch()
    except Exception as exc:  # noqa: BLE001 - any client/transport error degrades
        lease.release()
        if payload is not None:
            # Stale beats absent: the numbers are days old at worst, whereas
            # the static fallback is wrong by nearly a factor of six.
            logger.debug("using stale %s catalogue (%.0fs old): %s", key, age, exc)
            return Listing(payload=payload, age_s=age, failed=True)
        logger.debug("no %s catalogue available: %s", key, exc)
        return Listing(payload=None, age_s=float("inf"), failed=True)

    _write_cache(path, fresh)
    lease.release()
    return Listing(payload=fresh, age_s=0.0, fetched=True)


def cached_listing(
    key: str,
    fetch: Callable[[], dict[str, Any]],
    *,
    ttl_s: float = DEFAULT_TTL_S,
    cache_dir: Path | None = None,
) -> dict[str, Any] | None:
    """The original three-state reader, kept for callers that want no threads.

    Equivalent to :func:`read_listing` with the soft TTL pinned to the hard one:
    fresh is served, expired is fetched synchronously, and nothing runs in the
    background. Callers that want stale-while-revalidate use :func:`read_listing`
    and read the ``Listing`` flags.
    """
    return read_listing(key, fetch, soft_ttl_s=ttl_s, ttl_s=ttl_s, cache_dir=cache_dir).payload


def invalidate(key: str, *, cache_dir: Path | None = None) -> None:
    """Drop the ``key`` document so the next :func:`cached_listing` refetches.

    Needed because the payload is written BEFORE anything tries to interpret it.
    A document that validates but cannot be mapped (a ``models`` key that is not
    an array, a context length arriving as an object) would otherwise be served
    as a fresh cache hit on every start for the whole TTL, repeating the same
    failure for a day with no way to recover but deleting the file by hand.

    Called by ``discovery._available_models`` -- the one place that maps a cached
    payload -- when a document it just read yielded no usable rows.
    """
    try:
        _cache_path(key, cache_dir).unlink(missing_ok=True)
    except OSError as exc:  # pragma: no cover - read-only cache dir
        logger.debug("could not invalidate %s catalogue: %s", key, exc)
