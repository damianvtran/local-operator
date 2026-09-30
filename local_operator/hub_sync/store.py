"""Availability / failure state (design B4): ``<config_dir>/hub/status.json``.

A CACHE of derivable state plus retry bookkeeping — never a source of truth for
content. If the file is lost or corrupt the next check re-derives every item; the
only thing that is genuinely lost is backoff memory, which errs toward checking
sooner. That is why a corrupt file is quarantined and treated as empty instead of
blocking anything.

Concurrency: the sidebar polls this constantly (a plain read, no lock), while the
runner, a CLI run and a route may all write. Writes are read-modify-write under
an ``flock`` with a bounded wait — a lock timeout degrades to "skip this write,
retry next tick" and never blocks a request — and land with temp + ``os.replace``
so a reader never sees a torn file.

``config_dir`` is always passed in (never ``Path.home()``): AGENTS.md "Isolating a
run".
"""

from __future__ import annotations

import contextlib
import errno
import json
import logging
import os
import random
import tempfile
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from local_operator.hub_sync.provenance import hub_root, now_iso

logger = logging.getLogger(__name__)

SCHEMA = 1
LOCK_WAIT_S = 2.0
#: A merge marked ``updating`` longer ago than this is a crashed writer (B4.2).
UPDATING_STALE_S = 300
#: Backoff for counted failures: ``min(6h, 15min * 2**(attempts-1)) * U(0.9, 1.1)``.
BACKOFF_BASE_S = 15 * 60
BACKOFF_CAP_S = 6 * 3600
BACKOFF_JITTER = 0.1
#: After this many counted attempts stop nagging the model (B4.3); manual Retry or a
#: new remote fingerprint re-arms.
MAX_ATTEMPTS = 6
MODEL_UNAVAILABLE_RETRY_S = 3600
MISSING_RETRY_S = 24 * 3600
#: Crash-recovery bound only: a live holder's heartbeat renews every ``TTL / 3``, so
#: a short TTL costs a healthy run nothing and shortens how long a HARD-KILLED
#: daemon wedges every writer (``lop agents|teams sync``, the routes, the tool).
#: Not shorter than a few heartbeat periods: a holder whose thread is starved (or
#: whose host slept) for a whole TTL loses the lease to a contender.
LEASE_TTL_S = 180.0

#: Classes that are decided by a human, not a timer (B4.3).
NO_AUTO_RETRY = {"merge-refused", "prompt-too-long"}
#: Classes that are not counted attempts.
UNCOUNTED = {"no-credential", "concurrent-edit"}
#: Systemic: one item hitting these stops an apply-all run (B4.4).
SYSTEMIC = {"no-credential", "model-unavailable", "provider-error/quota"}

STATES = ("up-to-date", "available", "updating", "applied", "failed")


def item_key(kind: str, local_id: str) -> str:
    return f"{kind}:{local_id}"


def _parse(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _empty() -> dict[str, Any]:
    return {"schema": SCHEMA, "updated_at": now_iso(), "last_tick": None, "items": {}}


def scrub(text: str, limit: int = 300) -> str:
    """The user-facing error sentence: one line, bounded, credential-scrubbed.

    Scrubs by credential SHAPE (bearer headers, ``sk-`` keys, JWTs...): this text
    is built from hub and PROVIDER exceptions, none of which pass through the
    Radient client's own body scrubbing, and it is persisted to ``status.json``
    and served by the updates route and ``lop hub status``. There is no known
    secret VALUE to pass here, so a value-only redactor would replace nothing.
    The shapes run BEFORE the whitespace collapse and the length cut, so a token
    is never left half-masked by the truncation.
    """

    try:
        from local_operator.redaction_shapes import scrub_secrets

        text = scrub_secrets(str(text))
    except Exception:  # noqa: BLE001 - scrubbing is best-effort; length bound still applies
        text = str(text)
    return " ".join(text.split())[:limit]


class StatusStore:
    """Read (lock-free) and read-modify-write (locked, atomic) access to status.json."""

    def __init__(self, config_dir: Path) -> None:
        self.config_dir = Path(config_dir)
        self.path = hub_root(self.config_dir) / "status.json"
        self._lock_path = hub_root(self.config_dir) / ".status.lock"

    # -- reads ---------------------------------------------------------------

    def load(self) -> dict[str, Any]:
        """The document, or an empty one. Corrupt files are quarantined, not fatal."""

        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return _empty()
        except (OSError, ValueError):
            self._quarantine()
            return _empty()
        if (
            not isinstance(raw, dict)
            or raw.get("schema") != SCHEMA
            or not isinstance(raw.get("items"), dict)
        ):
            self._quarantine()
            return _empty()
        return raw

    def _quarantine(self) -> None:
        target = self.path.with_name(f"status.json.corrupt-{int(time.time())}")
        try:
            os.replace(self.path, target)
            logger.warning("hub status file was unreadable; moved to %s", target.name)
        except OSError:
            pass

    # -- writes --------------------------------------------------------------

    @contextlib.contextmanager
    def _locked(self) -> Iterator[bool]:
        """Yield True holding the lock, False on a bounded-wait timeout."""

        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        acquired = False
        try:
            import fcntl

            deadline = time.monotonic() + LOCK_WAIT_S
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except OSError as exc:
                    if exc.errno not in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
                        raise
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(0.05)
            yield acquired
        finally:
            if acquired:
                try:
                    import fcntl

                    fcntl.flock(fd, fcntl.LOCK_UN)
                except OSError:
                    pass
            os.close(fd)

    def mutate(self, fn: Callable[[dict[str, Any]], None]) -> bool:
        """Apply ``fn`` to the document under the lock. False = skipped (lock or IO failure)."""

        try:
            with self._locked() as held:
                if not held:
                    logger.debug("hub status lock busy; skipping a write")
                    return False
                doc = self.load()
                fn(doc)
                doc["updated_at"] = now_iso()
                self._write(doc)
                return True
        except OSError:
            logger.warning("could not write the hub status file", exc_info=True)
            return False

    def _write(self, doc: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".status.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(doc, handle, ensure_ascii=False, indent=1, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise


# -- item bookkeeping (pure functions over the document) -------------------------------------


def blank_item(
    kind: str, local_id: str, name: str, hub_id: str, tenant: str | None
) -> dict[str, Any]:
    return {
        "kind": kind,
        "local_id": local_id,
        "name": name,
        "hub_id": hub_id,
        "tenant_id": tenant,
        "state": "up-to-date",
        "remote_fingerprint": None,
        "local_fingerprint": None,
        "baseline": "known",
        "classification": None,
        "summary": {},
        "attempts": 0,
        "error_class": None,
        "error_subclass": None,
        "last_error": None,
        "first_seen_available_at": None,
        "last_checked_at": None,
        "last_applied_at": None,
        "next_retry_at": None,
        "auto_retry": True,
        "applied_backup": None,
    }


def effective_state(item: Mapping[str, Any], now: datetime | None = None) -> str:
    """The state, with a crashed ``updating`` writer read as ``failed`` (B4.2)."""

    state = str(item.get("state") or "up-to-date")
    if state == "updating":
        since = _parse(item.get("updating_since"))
        if (
            since is None
            or ((now or datetime.now(timezone.utc)) - since).total_seconds() > UPDATING_STALE_S
        ):
            return "failed"
    return state


def backoff_delay_s(attempts: int, rng: Callable[[float, float], float] = random.uniform) -> float:
    return min(BACKOFF_CAP_S, BACKOFF_BASE_S * (2 ** max(0, attempts - 1))) * rng(
        1 - BACKOFF_JITTER, 1 + BACKOFF_JITTER
    )


def apply_check(
    doc: dict[str, Any],
    *,
    kind: str,
    local_id: str,
    name: str,
    hub_id: str,
    tenant_id: str | None,
    verdict: str,
    classification: str | None,
    baseline: str,
    local_fp: str | None,
    remote_fp: str | None,
    reason: str = "",
    detail: str = "",
    now: datetime | None = None,
) -> dict[str, Any]:
    """Fold one check result into the document; returns the item."""

    now = now or datetime.now(timezone.utc)
    key = item_key(kind, local_id)
    items = doc.setdefault("items", {})
    item = items.get(key) or blank_item(kind, local_id, name, hub_id, tenant_id)
    item.update(name=name, hub_id=hub_id, tenant_id=tenant_id, last_checked_at=_iso(now))
    item["local_fingerprint"] = local_fp
    previous_remote = item.get("remote_fingerprint")
    if remote_fp is not None:
        if previous_remote and previous_remote != remote_fp:
            # A new remote text re-arms everything a human/attempt cap had stopped (B4.3).
            item["attempts"], item["auto_retry"], item["next_retry_at"] = 0, True, None
        item["remote_fingerprint"] = remote_fp
    if verdict == "unavailable":
        if reason == "no-credential":
            # Not a failure (no attempt is counted, no retry is scheduled, the state is
            # left as it was) but it IS a fact the person can act on: this item cannot be
            # reached until they sign in. It is recorded as the item's ``error_class`` so
            # the snapshot can carry it - including for an item whose state is
            # ``up-to-date``, which is exactly the org-linked user with no login and would
            # otherwise be dropped from the list with no way to learn why nothing
            # updates (UX round 2, U11). ONE derivation: the UI reads this field and
            # never re-derives "needs the login" from the credential or the tenant.
            item["error_class"], item["last_error"] = "no-credential", None
            item["error_subclass"] = None
            # It also RETIRES the schedule the class it replaced had armed (agent review
            # round 4, M2): an apply this item could not do is not worth retrying, and a
            # stale ``next_retry_at`` would keep answering "due" for a merge that cannot
            # run. We do not touch ``auto_retry`` or ``attempts``: the class is uncounted
            # and a successful fetch restores everything through ``settle_applied`` /
            # the up-to-date arm.
            item["next_retry_at"] = None
        else:
            record_failure(
                item, reason or "hub-error", detail, now=now, keep_state=reason == "hub-error"
            )
    elif verdict == "up-to-date":
        # ``applied`` is kept for exactly ONE check cycle ("Updated just now"), so
        # the check that first observes it only stamps ``applied_seen`` and the
        # next one settles to ``up-to-date``. Without the stamp it was sticky.
        keep_applied = item.get("state") == "applied" and not item.get("applied_seen")
        item["applied_seen"] = keep_applied
        item.update(
            state="applied" if keep_applied else "up-to-date",
            classification=None,
            error_class=None,
            last_error=None,
            attempts=0,
            next_retry_at=None,
            auto_retry=True,
            first_seen_available_at=None,
            summary={},
        )
        item["baseline"] = baseline
    else:
        item.update(state="available", classification=classification, baseline=baseline)
        item["first_seen_available_at"] = item.get("first_seen_available_at") or _iso(now)
        # A successful fetch retires the two classes a fetch can leave behind.
        if item.get("error_class") in ("hub-item-missing", "no-credential"):
            item["error_class"], item["last_error"] = None, None
    items[key] = item
    return item


def settle_applied(
    item: dict[str, Any],
    summary: Mapping[str, int],
    backup: str | None,
    now: datetime | None = None,
) -> None:
    now = now or datetime.now(timezone.utc)
    item["applied_seen"] = False
    item.update(
        state="applied",
        classification=None,
        error_class=None,
        last_error=None,
        attempts=0,
        next_retry_at=None,
        auto_retry=True,
        summary=dict(summary),
        last_applied_at=_iso(now),
        applied_backup=backup,
        first_seen_available_at=None,
    )
    item.pop("updating_since", None)


def settle_unchanged(item: dict[str, Any]) -> None:
    """The merge produced exactly the local text: nothing to write, nothing pending."""

    item.update(
        state="up-to-date",
        classification=None,
        error_class=None,
        error_subclass=None,
        last_error=None,
        attempts=0,
        next_retry_at=None,
        auto_retry=True,
        first_seen_available_at=None,
        summary={},
    )
    item.pop("updating_since", None)


def mark_updating(item: dict[str, Any], now: datetime | None = None) -> None:
    item["state"] = "updating"
    item["updating_since"] = _iso(now or datetime.now(timezone.utc))


def record_failure(
    item: dict[str, Any],
    error_class: str,
    message: str,
    *,
    now: datetime | None = None,
    keep_state: bool = False,
    summary: Mapping[str, int] | None = None,
    rng: Callable[[float, float], float] = random.uniform,
) -> None:
    """Set ``error_class``/``last_error``, the state and the retry schedule per B4.3."""

    now = now or datetime.now(timezone.utc)
    cls, _, sub = error_class.partition("/")
    item.pop("updating_since", None)
    item["error_class"], item["error_subclass"] = cls, sub or None
    item["last_error"] = scrub(message)
    if summary is not None:
        item["summary"] = dict(summary)
    if cls not in UNCOUNTED:
        item["attempts"] = int(item.get("attempts") or 0) + 1
    if not keep_state:
        had_check = item.get("first_seen_available_at") is not None or item.get("classification")
        if cls in ("prompt-too-long", "hub-item-missing"):
            item["state"] = "failed"
        elif cls == "provider-error" and not had_check:
            item["state"] = "failed"
        else:
            item["state"] = "available"
    delay: float | None
    if cls in NO_AUTO_RETRY:
        item["auto_retry"], delay = False, None
    elif cls == "concurrent-edit":
        delay = None  # next tick
    elif cls == "hub-item-missing":
        delay = MISSING_RETRY_S
    elif cls == "model-unavailable":
        delay = MODEL_UNAVAILABLE_RETRY_S
    else:
        delay = backoff_delay_s(int(item["attempts"]), rng)
    if item.get("attempts", 0) >= MAX_ATTEMPTS and cls not in UNCOUNTED:
        item["auto_retry"], delay = False, None
    item["next_retry_at"] = _iso(now + timedelta(seconds=delay)) if delay is not None else None


def clear_failure(item: dict[str, Any]) -> None:
    """A manual retry proved the update can be computed: retire the stale failure.

    Without this a manual-mode item that failed while the hub was down stayed
    ``available`` + ``hub-error`` forever: the retry answered "ready to update"
    (a dry run - manual mode never writes on a retry) but left the failure on the
    row, so the UI kept offering Retry and the update itself was unreachable
    (UX round 2, U10). The state moves to ``available`` (the item IS available: the
    check just fetched it) so the row offers the update instead.
    """

    item.update(error_class=None, error_subclass=None, last_error=None, next_retry_at=None)
    if item.get("state") == "failed":
        item["state"] = "available"


def clear_retry(item: dict[str, Any]) -> None:
    """Manual Retry: user intent outranks the schedule (B4.3)."""

    item.update(attempts=0, next_retry_at=None, auto_retry=True)


def auto_apply_due(
    item: Mapping[str, Any], now: datetime | None = None, *, manual: bool = False
) -> bool:
    """May a timer tick try to apply this item now?"""

    if manual:
        return True
    if not item.get("auto_retry", True):
        return False
    due = _parse(item.get("next_retry_at"))
    return due is None or due <= (now or datetime.now(timezone.utc))


def check_due(item: Mapping[str, Any], now: datetime | None = None) -> bool:
    """Should a timer tick re-fetch this item? Only a 404'd item is throttled (24 h)."""

    if item.get("error_class") == "hub-item-missing":
        due = _parse(item.get("next_retry_at"))
        return due is None or due <= (now or datetime.now(timezone.utc))
    return True


def prune_items(doc: dict[str, Any], live: set[str]) -> None:
    """Drop items whose local row or baseline link is gone (B4.2)."""

    for key in [k for k in doc.get("items", {}) if k not in live]:
        del doc["items"][key]


# -- cross-process lease (B3.3 layer 2) -------------------------------------------------


class RunnerLease:
    """``O_EXCL`` + expiry lease so a second daemon or a CLI run does not merge concurrently.

    The ``model/catalogue._ListingFetchLease`` shape: a crashed holder cannot
    block forever because the lease carries an expiry, and every IO failure
    degrades to "no lease" for a READ-ONLY caller. Here a failure to take the
    lease means SKIP (a merge that ignored a lost race would be the double-apply
    the lease exists to prevent).

    EVERY WRITER TAKES IT: the runner tick, the desktop routes (through
    ``HubSyncRunner.exclusive``), ``lop agents|teams sync`` and the ``agent``
    tool's sync. A holder is kept alive by a heartbeat thread started in
    :meth:`acquire`, because a tick of fifty items with model calls of up to two
    minutes each outlives any fixed TTL; the TTL is only the bound on how long a
    CRASHED holder blocks everyone else.
    """

    #: Poll period while waiting for a held lease.
    _POLL_S = 0.25

    def __init__(self, config_dir: Path, ttl_s: float = LEASE_TTL_S) -> None:
        self._path = hub_root(Path(config_dir)) / ".runner.lease"
        self._ttl = ttl_s
        self._token = f"{os.getpid()}:{uuid.uuid4().hex[:8]}"
        self._held = False
        self._beat_stop: threading.Event | None = None
        # Serialises ``renew`` against ``release``: without it the heartbeat can pass
        # its ``_held`` check, lose the CPU to ``release`` (which unlinks the file),
        # then ``os.replace`` a fresh lease file with OUR token back into place --
        # a lease nobody holds that blocks every writer for a full TTL.
        self._guard = threading.Lock()

    def acquire(self, wait_s: float = 0.0) -> bool:
        """Take the lease, waiting up to ``wait_s`` for a live holder; start the heartbeat."""

        deadline = time.monotonic() + max(0.0, wait_s)
        while True:
            if self._try_acquire():
                self._start_heartbeat()
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(self._POLL_S)

    def _start_heartbeat(self) -> None:
        stop = threading.Event()
        self._beat_stop = stop

        def beat() -> None:
            while not stop.wait(self._ttl / 3):
                if not self.renew():
                    return

        threading.Thread(target=beat, name="hub-lease-heartbeat", daemon=True).start()

    def _try_acquire(self) -> bool:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self._path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            try:
                expires = float(json.loads(self._path.read_text("utf-8")).get("expires_at", 0.0))
            except (OSError, ValueError):
                expires = 0.0
            if expires > time.time():
                return False
            try:
                self._path.unlink()
            except OSError:
                return False
            return self._try_acquire()
        except OSError:
            return False
        try:
            os.write(
                fd,
                json.dumps({"holder": self._token, "expires_at": time.time() + self._ttl}).encode(),
            )
        finally:
            os.close(fd)
        self._held = True
        return True

    def renew(self) -> bool:
        """Push the expiry out again; False when the lease is no longer ours.

        The TTL is a crash-recovery bound, not a budget: a tick of up to fifty
        items with model calls of up to two minutes each can outlive any fixed
        TTL, and a second process acquiring the lease mid-tick is the double
        apply the lease exists to prevent. So the heartbeat renews it.

        The rewrite is atomic (temp + replace): a reader that saw a half-written
        file would parse it as "expired" and steal a live lease.
        """

        with self._guard:
            if not self._held:
                return False
            try:
                data = json.loads(self._path.read_text("utf-8"))
                if data.get("holder") != self._token:
                    self._held = False
                    return False
                data["expires_at"] = time.time() + self._ttl
                tmp = self._path.with_name(f"{self._path.name}.{self._token.replace(':', '-')}.tmp")
                tmp.write_text(json.dumps(data), "utf-8")
                os.replace(tmp, self._path)
            except (OSError, ValueError):
                return False
            return True

    def remaining_s(self) -> float | None:
        """Seconds until the CURRENT holder's lease expires; ``None`` if unreadable/free.

        Lets a refused caller say how long a crashed holder can still block it,
        rather than a vague "try again".
        """

        try:
            expires = float(json.loads(self._path.read_text("utf-8")).get("expires_at", 0.0))
        except (OSError, ValueError):
            return None
        return max(0.0, expires - time.time())

    def release(self) -> None:
        if self._beat_stop is not None:
            self._beat_stop.set()
            self._beat_stop = None
        with self._guard:
            if not self._held:
                return
            # Flipped under the guard, so a heartbeat already inside ``renew`` finishes
            # first and any later one sees ``_held`` False and never rewrites the file.
            self._held = False
            try:
                holder = json.loads(self._path.read_text("utf-8")).get("holder")
                if holder == self._token:
                    self._path.unlink()
            except (OSError, ValueError):
                pass

    def __enter__(self) -> "RunnerLease":
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()
