"""Durable per-session settings intents, and the background applier that lands them.

WHY THIS EXISTS. A model/effort/fast/working-directory pick in the desktop used
to be ONE HTTP request that attached to the session's owner, refreshed the watch
lease and waited for the runtime's slash ack (``POST …/commands``): up to 23 s of
budgets against a renderer deadline of 20 s, and a runtime busy with a long
synchronous step answered late or not at all. The user could not switch the
model, go to the next conversation and do the same there — every pick was a
foreground wait that a conversation switch or an app restart threw away.

THE SHAPE (design: "Durable background settings intents", PR2):

* The pick is an INTENT, written to ``sessions/<id>/settings-intent.json`` (one
  slot per kind: ``model | effort | fast | cwd``) and answered at once (202). The
  route never calls the runtime.
* :class:`SettingsApplier` — one per server process, on ``app.state`` — applies
  every non-terminal slot in the background through the existing bridge, with
  the retry schedule below, and publishes each state change on the machine-wide
  desktop feed as a ``settings_intent`` frame.
* The FILE is the authority, not the renderer: the desktop backend does not
  reliably outlive an app restart (a self-spawned daemon is SIGTERMed on quit),
  so the applier resumes every non-terminal slot in ``lifespan``.

COALESCING IS LAST-WINS. A POST for kind K replaces slot K and takes a new
``generation`` from a SESSION-WIDE monotonic counter (kept in the file, so a
DELETE followed by a new POST can never reuse a generation an in-flight attempt
still carries). An attempt in flight for an older generation runs to completion;
its result is discarded when the slot's generation moved, and the worker runs
the newer one straight away.

SUPERSEDE. Before every attempt (and therefore on resume) the applier reads the
session's current selection from its journal (``read_model_selection`` — memoised
and sidecar-backed, so no transcript scan in the common case) or the cwd from the
desktop marker. A value equal to neither the slot's ``base`` nor its ``value``
means someone else changed the setting (a TUI ``/model``, the phone): the slot
becomes ``superseded``, is published once and pruned, so a stale pick never
overwrites a newer choice made elsewhere.

WHAT THIS DELIBERATELY DOES NOT DO:

* It never calls ``refresh_watch``: the watch is a presence lease, unrelated to
  settings, and its chained 5 s bound is part of the latency this removes.
* It does not classify busy vs unreachable on its own terms beyond mapping the
  exceptions the bridge already raises; that classification is owned elsewhere
  (``routes/desktop_sessions.py``'s ``errors()``), and both are retryable here.
* It does not admit a switch before ``frontend_sync`` — retries cover a fresh
  attach to a starved runtime.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import os
import re
import tempfile
import time
import uuid
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol

logger = logging.getLogger(__name__)

SCHEMA = 1
INTENT_FILE = "settings-intent.json"

Kind = Literal["model", "effort", "fast", "cwd"]
KINDS: tuple[Kind, ...] = ("model", "effort", "fast", "cwd")

#: The fixed apply order across kinds. A model switch RESETS fast mode (and may
#: re-seat effort onto the new model's ladder), so fast is applied after the
#: model, and the effort slot is paired with the model rather than racing it;
#: cwd retires the runtime, so it goes last to let the settings land on the
#: runtime the user was looking at.
APPLY_ORDER: tuple[Kind, ...] = ("model", "effort", "fast", "cwd")

ACTIVE_STATES = frozenset({"pending", "applying", "queued_next_step", "waiting_idle"})
TERMINAL_STATES = frozenset({"applied", "failed", "superseded"})

#: Attempt delays for model/effort/fast, then every ``RETRY_TAIL_S``. Measured
#: inputs (design §c): owner silence while working reached 205.8 s (heartbeat
#: gaps, n=2), a busy refusal hints 2 s, and a pre-lane parked switch was first
#: seen after 31.3 s — so a fast front for the common blip and a 30 s tail for a
#: long synchronous step.
RETRY_SCHEDULE_S: tuple[float, ...] = (0.0, 1.0, 2.0, 4.0, 8.0, 15.0)
RETRY_TAIL_S = 30.0
#: Wall budget from the LATEST POST: ~3x the worst observed owner silence.
DEFAULT_BUDGET_S = 600.0
#: Test/QA-only override of the budget (seconds).
BUDGET_ENV = "LOP_SETTINGS_INTENT_BUDGET_S"
#: A cwd move refused because a turn is running waits for the turn to END —
#: turns run for hours, so there is no wall budget; the user can cancel.
CWD_IDLE_RETRY_S = 5.0
#: A move refused because another viewer holds the session is retried this long,
#: then fails as ``move_exclusive`` naming the remedy.
CWD_EXCLUSIVE_BUDGET_S = 120.0
#: Machine-wide cap on concurrent attempts. Each attempt holds a bridge; a burst
#: across many sessions must not walk the pool towards its ``BRIDGE_COUNT``
#: ceiling (64), where evictions start hurting sessions the user has open.
MAX_IN_FLIGHT = 4
#: An applied slot is kept this long so a renderer that reconnects shortly after
#: still sees the confirmation, then pruned (a ``null`` frame announces it).
PRUNE_APPLIED_AFTER_S = 60.0
#: After a ``queued`` ack the applier does NOT resend; it polls the journal for
#: the target at this interval, and resends once if the target is not confirmed
#: within ``CONFIRM_WINDOW_S`` (safe: the operation is set-to-target).
CONFIRM_POLL_S = 2.0
CONFIRM_WINDOW_S = 60.0

_SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def budget_seconds() -> float:
    raw = os.environ.get(BUDGET_ENV, "").strip()
    if raw:
        try:
            value = float(raw)
        except ValueError:
            logger.warning("ignoring non-numeric %s=%r", BUDGET_ENV, raw)
        else:
            if value > 0:
                return value
    return DEFAULT_BUDGET_S


def retry_delay(attempts: int) -> float:
    """Delay before the attempt that follows ``attempts`` failed ones."""
    if attempts < len(RETRY_SCHEDULE_S):
        return RETRY_SCHEDULE_S[attempts]
    return RETRY_TAIL_S


def valid_session_id(session_id: str) -> bool:
    return bool(_SESSION_ID.match(session_id or ""))


def _iso(epoch: float | None) -> str | None:
    if epoch is None:
        return None
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _epoch(value: Any) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


# -- the store ---------------------------------------------------------------


class IntentStore:
    """``sessions/<id>/settings-intent.json``: schema 1, written atomically.

    tmp + fsync + ``os.replace`` in the SAME directory, the way ``desktop.json``
    is written, so a process killed mid-write leaves either the old file or the
    new one. A file that cannot be parsed reads as empty: an intent is a request,
    and losing an unreadable one costs a re-pick, never the session. Retention
    deletes the session directory and this file with it.
    """

    def __init__(self, config_dir: Path) -> None:
        self.root = Path(config_dir) / "sessions"

    def session_dir(self, session_id: str) -> Path:
        return self.root / session_id

    def path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / INTENT_FILE

    def exists(self, session_id: str) -> bool:
        return valid_session_id(session_id) and self.session_dir(session_id).is_dir()

    def load(self, session_id: str) -> dict[str, Any]:
        try:
            raw = self.path(session_id).read_text(encoding="utf-8")
            payload = json.loads(raw)
        except (OSError, ValueError):
            return {"schema": SCHEMA, "generation": 0, "slots": {}, "seen": {}}
        if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
            return {"schema": SCHEMA, "generation": 0, "slots": {}, "seen": {}}
        slots = payload.get("slots")
        payload["slots"] = {
            kind: slot
            for kind, slot in (slots.items() if isinstance(slots, dict) else ())
            if kind in KINDS and isinstance(slot, dict)
        }
        payload["generation"] = int(payload.get("generation") or 0)
        seen = payload.get("seen")
        payload["seen"] = {
            str(rid): str(kind)
            for rid, kind in (seen.items() if isinstance(seen, dict) else ())
            if kind in KINDS
        }
        return payload

    def save(self, session_id: str, document: Mapping[str, Any]) -> None:
        directory = self.session_dir(session_id)
        target = self.path(session_id)
        if not document.get("slots"):
            # Nothing left to apply or show, but keep the GENERATION counter: a
            # later POST must not reuse a number an in-flight attempt carries.
            # ``seen`` goes with the slots: see ``SettingsApplier.submit``.
            document = {"schema": SCHEMA, "generation": document.get("generation", 0), "slots": {}}
        data = json.dumps(document, sort_keys=True, indent=1).encode("utf-8")
        fd, tmp = tempfile.mkstemp(prefix=".settings-intent.", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(tmp, 0o600)
            os.replace(tmp, target)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    def sessions_with_intents(self) -> list[str]:
        """Every session with an intent file: a glob, ~200 stats at most."""
        try:
            return sorted(
                path.parent.name
                for path in self.root.glob(f"*/{INTENT_FILE}")
                if valid_session_id(path.parent.name)
            )
        except OSError:
            return []


# -- the runtime side --------------------------------------------------------


@dataclass(frozen=True)
class Current:
    """The session's settings as its durable records say they are NOW.

    ``None`` fields are "no reading", never "no value": the supersede rule only
    fires on a reading that positively differs from both ``base`` and ``value``.
    """

    model: str | None = None
    effort: str | None = None
    fast: bool | None = None
    cwd: str | None = None
    settings_result: Mapping[str, Any] | None = None

    def of(self, kind: str) -> Any:
        return getattr(self, kind, None)


class RequestReplaced(Exception):
    """A replayed ``request_id`` whose slot a newer pick has already replaced."""


def public_slot(slot: Mapping[str, Any]) -> dict[str, Any]:
    """The slot as clients see it: ``_``-prefixed bookkeeping stays server-side.

    ``_predecessors`` / ``_paired_generation`` are the applier's own memory
    (review n1); a client that started switching on them would pin internals.
    """
    return {key: value for key, value in slot.items() if not str(key).startswith("_")}


@dataclass(frozen=True)
class Outcome:
    """One attempt's classified result.

    ``status``:
      * ``applied`` — the runtime answered definitively and the change is in.
      * ``queued`` — a deferred ack ("applies at the next step"); confirmed later.
      * ``retry`` — transient (busy, unreachable, retiring, a capped pool).
      * ``waiting_idle`` — a cwd move refused by a running turn.
      * ``exclusive`` — a cwd move refused by another viewer.
      * ``terminal`` — refused for a reason a retry cannot change.
    """

    status: Literal["applied", "queued", "retry", "waiting_idle", "exclusive", "terminal"]
    code: str = ""
    message: str = ""


class SettingsDriver(Protocol):
    """What the applier needs from the runtime side; faked in tests."""

    async def current(self, session_id: str) -> Current: ...

    async def apply(
        self, session_id: str, kind: str, value: Any, effort: str | None, *, request_id: str
    ) -> Outcome: ...


# -- the slot ---------------------------------------------------------------


def new_slot(
    *,
    kind: str,
    value: Any,
    effort: str | None,
    request_id: str,
    generation: int,
    base: Any,
    now: float,
    budget: float,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "value": value,
        "effort": effort,
        "request_id": request_id,
        "generation": generation,
        "base": base,
        "state": "pending",
        "attempts": 0,
        "created_at": _iso(now),
        "updated_at": _iso(now),
        "next_attempt_at": _iso(now),
        "deadline_at": _iso(now + budget),
        "error": None,
    }


def _error(code: str, message: str, *, retryable: bool, **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {"code": code, "message": message, "retryable": retryable}
    body.update({key: value for key, value in extra.items() if value is not None})
    return body


@dataclass
class _SessionState:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task[None] | None = None
    #: Clock time a cwd move was first refused by another viewer, per
    #: generation. In memory only: a restart resetting the 120 s window is the
    #: same "the user asked, apply is idempotent" rule as the attempt counter.
    exclusive_since: dict[int, float] = field(default_factory=dict)
    #: Clock time a queued ack was received, per (kind, generation).
    queued_since: dict[tuple[str, int], float] = field(default_factory=dict)
    resent: set[tuple[str, int]] = field(default_factory=set)


Publisher = Callable[[str, str, "dict[str, Any] | None"], None]


class SettingsApplier:
    """Applies every session's settings intents in the background.

    ONE PER PROCESS (``app.state.settings_applier``). Per session a single worker
    task applies slots in :data:`APPLY_ORDER`, serially — a session's runtime has
    one session loop and two concurrent switches against it would only queue
    there. Across sessions, at most :data:`MAX_IN_FLIGHT` attempts run at once.

    ``clock`` is wall time (the slot's timestamps, deadlines) and ``sleep`` the
    waiting primitive; both are injectable so the retry schedule is testable
    with a fake clock.
    """

    def __init__(
        self,
        config_dir: Path,
        driver: SettingsDriver,
        *,
        publish: Publisher | None = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], Awaitable[Any]] | None = None,
        budget: float | None = None,
        max_in_flight: int = MAX_IN_FLIGHT,
    ) -> None:
        self.store = IntentStore(config_dir)
        self.driver = driver
        self._publish = publish
        self.clock = clock
        self._sleep = sleep
        self._budget = budget
        self._limit = asyncio.Semaphore(max_in_flight)
        self._sessions: dict[str, _SessionState] = {}
        self._closed = False
        self.in_flight = 0
        self.peak_in_flight = 0

    # -- public API (routes) ------------------------------------------------

    @property
    def budget(self) -> float:
        return self._budget if self._budget is not None else budget_seconds()

    def _state(self, session_id: str) -> _SessionState:
        state = self._sessions.get(session_id)
        if state is None:
            state = self._sessions[session_id] = _SessionState()
        return state

    async def _load(self, session_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(self.store.load, session_id)

    async def _save(self, session_id: str, document: Mapping[str, Any]) -> None:
        await asyncio.to_thread(self.store.save, session_id, dict(document))

    async def submit(
        self,
        session_id: str,
        *,
        kind: str,
        value: Any,
        effort: str | None,
        request_id: str,
        base: Any,
    ) -> tuple[bool, dict[str, Any], dict[str, Any]]:
        """Record a pick; returns ``(created, slot, all slots)``.

        ``created=False`` is the idempotent replay: the ``request_id`` equals the
        slot's current one, so the stored slot is answered unchanged (200).
        Raises :class:`RequestReplaced` for a ``request_id`` this session already
        accepted whose slot a newer pick has since replaced (review m3).

        SEEN REQUEST IDS. ``document["seen"]`` maps every accepted
        ``request_id`` to its kind for exactly as long as the document holds any
        slot — the store drops it with the last slot (``IntentStore.save``), so
        it shares the intents' own lifetime instead of adding a count or age
        cap. Its size is therefore the number of picks made while some slot of
        this session was still pending or showing, i.e. human picking speed over
        at most the retry budget plus the 60 s applied display. A replay that
        arrives after everything was pruned is accepted as a new pick, which is
        the same answer it got before this record existed.
        """
        state = self._state(session_id)
        async with state.lock:
            document = await self._load(session_id)
            slots = document["slots"]
            existing = slots.get(kind)
            if existing is not None and existing.get("request_id") == request_id:
                return False, existing, slots
            if request_id in document["seen"]:
                raise RequestReplaced(request_id)
            document["seen"][request_id] = kind
            document["generation"] = int(document.get("generation", 0)) + 1
            now = self.clock()
            slot = new_slot(
                kind=kind,
                value=value,
                effort=effort,
                request_id=request_id,
                generation=document["generation"],
                base=base,
                now=now,
                budget=self.budget,
            )
            if existing and existing.get("state") in ACTIVE_STATES:
                # Our own earlier generation may still land after this POST.
                # Retain that target as an owned predecessor across restart so
                # the supersede rule cannot mistake it for an external change.
                # Deduplicated (review n1): A→B→A picks must not grow it by two
                # entries each time; membership is all the supersede rule reads.
                owned: list[Any] = []
                for item in (
                    *existing.get("_predecessors", []),
                    existing["value"],
                    existing.get("base"),
                ):
                    if item not in owned:
                        owned.append(item)
                slot["_predecessors"] = owned
            slots[kind] = slot
            dropped: list[str] = []
            older_effort = slots.get("effort")
            if (
                kind == "model"
                and effort is not None
                and older_effort is not None
                and older_effort.get("state") in ACTIVE_STATES
            ):
                # Review m2: a model pick that NAMES its effort is the user's
                # newest effort choice too. Left in place, the older effort slot
                # would be re-based after the model lands and then applied over
                # it. Same-kind last-wins replaces silently; so does this. An
                # effort attempt already in flight runs before the model's
                # (one serial worker per session) and its result is discarded
                # because the slot is gone.
                del slots["effort"]
                dropped.append("effort")
            await self._save(session_id, document)
        for gone in dropped:
            self._emit(session_id, gone, None)
        self._emit(session_id, kind, slot)
        self._kick(session_id)
        return True, slot, slots

    async def dismiss(self, session_id: str, kind: str, generation: int) -> bool | None:
        """Remove slot ``kind``: ``True`` done, ``False`` generation moved, ``None`` absent.

        An attempt already in flight is not recalled; its result is discarded
        because the slot (and so its generation) is gone.
        """
        state = self._state(session_id)
        async with state.lock:
            document = await self._load(session_id)
            slot = document["slots"].get(kind)
            if slot is None:
                return None
            if int(slot.get("generation", -1)) != generation:
                return False
            del document["slots"][kind]
            await self._save(session_id, document)
        self._emit(session_id, kind, None)
        return True

    async def seen_request(self, session_id: str, request_id: str) -> bool:
        """Whether ``request_id`` was accepted while this session's slots lived."""
        return request_id in (await self._load(session_id))["seen"]

    async def snapshot(self, session_id: str | None = None) -> dict[str, dict[str, Any]]:
        if session_id is not None:
            ids = [session_id] if valid_session_id(session_id) else []
        else:
            ids = await asyncio.to_thread(self.store.sessions_with_intents)
        result: dict[str, dict[str, Any]] = {}
        for sid in ids:
            slots = (await self._load(sid))["slots"]
            if slots:
                result[sid] = slots
        return result

    async def resume(self) -> int:
        """Re-arm every non-terminal slot on disk; called once from ``lifespan``.

        ``attempts`` resets and the deadline is recomputed from NOW: the user
        asked, the apply is idempotent, and the supersede rule (run before the
        first attempt) guards against landing a pick that is stale by now.
        Returns the number of sessions re-armed.
        """
        ids = await asyncio.to_thread(self.store.sessions_with_intents)
        armed = 0
        for session_id in ids:
            state = self._state(session_id)
            async with state.lock:
                document = await self._load(session_id)
                now = self.clock()
                changed = False
                for slot in document["slots"].values():
                    if slot.get("state") in ACTIVE_STATES:
                        slot["attempts"] = 0
                        slot["deadline_at"] = _iso(now + self.budget)
                        slot["next_attempt_at"] = _iso(now)
                        if slot["state"] in {"applying", "queued_next_step"}:
                            slot["state"] = "pending"
                        changed = True
                if changed:
                    await self._save(session_id, document)
            if document["slots"]:
                armed += 1
                self._kick(session_id)
        return armed

    async def close(self) -> None:
        self._closed = True
        tasks = [s.task for s in self._sessions.values() if s.task is not None]
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    # -- the worker -----------------------------------------------------------

    def _emit(self, session_id: str, kind: str, slot: dict[str, Any] | None) -> None:
        if self._publish is None:
            return
        try:
            self._publish(session_id, kind, public_slot(slot) if slot is not None else None)
        except Exception:  # noqa: BLE001 — a frame is advisory; GET is the baseline
            logger.debug("settings_intent frame not published", exc_info=True)

    def _kick(self, session_id: str) -> None:
        if self._closed:
            return
        state = self._state(session_id)
        state.wake.set()
        if state.task is None or state.task.done():
            state.task = asyncio.get_running_loop().create_task(
                self._run(session_id), name=f"settings-intents:{session_id}"
            )

    async def _sleep_until(self, state: _SessionState, delay: float) -> None:
        """Wait ``delay`` seconds or until a new POST wakes the worker."""
        if delay <= 0:
            return
        if self._sleep is not None:
            # The injected (fake-clock) sleep: a wake does not shorten it, which
            # keeps the schedule deterministic under test.
            await self._sleep(delay)
            return
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(state.wake.wait(), timeout=delay)

    async def _run(self, session_id: str) -> None:
        state = self._state(session_id)
        try:
            while not self._closed:
                state.wake.clear()
                delay = await self._step(session_id)
                if delay is None:
                    return
                await self._sleep_until(state, delay)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — one session's defect must not stop others
            logger.exception("settings intent worker for %s failed", session_id)

    async def _step(self, session_id: str) -> float | None:
        """Do whatever is due for ``session_id``; return the next wait or ``None`` to stop."""
        state = self._state(session_id)
        now = self.clock()
        async with state.lock:
            document = await self._load(session_id)
            slots = document["slots"]
            # Prune: superseded at once (published already), applied after 60 s.
            pruned = []
            for kind, slot in list(slots.items()):
                done_at = _epoch(slot.get("updated_at")) or now
                if slot.get("state") == "superseded" or (
                    slot.get("state") == "applied" and now - done_at >= PRUNE_APPLIED_AFTER_S
                ):
                    pruned.append(kind)
                    del slots[kind]
            if pruned:
                await self._save(session_id, document)
        for kind in pruned:
            self._emit(session_id, kind, None)

        active = [
            (kind, slots[kind])
            for kind in APPLY_ORDER
            if kind in slots and slots[kind].get("state") in ACTIVE_STATES
        ]
        if not active:
            applied = [
                _epoch(slot.get("updated_at")) or now
                for slot in slots.values()
                if slot.get("state") == "applied"
            ]
            if applied:
                return max(0.0, min(applied) + PRUNE_APPLIED_AFTER_S - now)
            return None

        kind, slot = active[0]
        due = _epoch(slot.get("next_attempt_at")) or now
        if due > now:
            # A later kind cannot overtake the first (model resets fast and
            # cwd retires the runtime). Waiting on its already-due timestamp
            # instead would spin at zero delay for the entire model backoff.
            return due - now
        if slot.get("state") == "queued_next_step":
            await self._confirm(session_id, kind, slot)
        else:
            await self._attempt(session_id, kind, slot)
        return 0.0

    async def _update(
        self, session_id: str, kind: str, generation: int, mutate: Callable[[dict[str, Any]], None]
    ) -> dict[str, Any] | None:
        """Apply ``mutate`` to slot ``kind`` iff its generation is still ``generation``."""
        state = self._state(session_id)
        async with state.lock:
            document = await self._load(session_id)
            slot = document["slots"].get(kind)
            if slot is None or int(slot.get("generation", -1)) != generation:
                return None
            mutate(slot)
            slot["updated_at"] = _iso(self.clock())
            await self._save(session_id, document)
        self._emit(session_id, kind, slot)
        return slot

    def _superseded(self, kind: str, slot: Mapping[str, Any], current: Current) -> bool:
        reading = current.of(kind)
        # No baseline at POST time (a session with no journal selection yet)
        # means nothing to compare against: never guess a supersede.
        if reading is None or slot.get("base") is None:
            return False
        allowed = [slot.get("base"), slot["value"], *slot.get("_predecessors", [])]
        if kind == "model":
            # Model IDs are case-sensitive; do not lowercase a user's selector.
            return reading not in allowed
        return reading not in allowed

    def _confirmed(self, kind: str, slot: Mapping[str, Any], current: Current) -> bool:
        reading = current.of(kind)
        if reading is None:
            return False
        if kind == "model":
            if str(reading).lower() != str(slot["value"]).lower():
                return False
            wanted = slot.get("effort")
            if not wanted or current.effort == wanted:
                return True
            # Review m5: the owner CLAMPS a paired level onto the new model's
            # ladder (``serving.set_model_effort`` → ``resolve_effort_in``), so an
            # exact match can be impossible and the slot would poll out the whole
            # budget. The model and its level travel in ONE ``set_model`` frame
            # and are journalled in one row, so once the reading has moved off
            # ``base`` onto our model, the level beside it IS the admitted
            # outcome. With the model unchanged (base == value) the old row
            # proves nothing, so the exact level is still required there.
            base = slot.get("base")
            return base is not None and str(base).lower() != str(slot["value"]).lower()
        return reading == slot["value"]

    async def _attempt(self, session_id: str, kind: str, slot: dict[str, Any]) -> None:
        generation = int(slot["generation"])
        try:
            current = await self.driver.current(session_id)
        except Exception:  # noqa: BLE001 — no reading is not a refusal
            logger.debug("settings intent: no current reading for %s", session_id, exc_info=True)
            current = Current()
        # BEFORE EVERY ATTEMPT, not only the first: a TUI ``/model`` that lands
        # between two of our retries is exactly the newer choice this must not
        # overwrite. Our own landed-but-unacknowledged attempt reads as
        # ``value``, which the rule accepts.
        if self._superseded(kind, slot, current):
            await self._update(session_id, kind, generation, lambda s: s.update(state="superseded"))
            return
        # Paired effort: an effort slot pending beside a model slot rides the
        # model's attempt (model always sorts first, so reaching an effort slot
        # here means no model slot is active).
        pending_effort: tuple[int, str] | None = None
        effort = slot.get("effort") if kind == "model" else None
        if kind == "model" and effort is None:
            pending_effort = await self._paired_effort(session_id)
            if pending_effort is not None:
                effort = pending_effort[1]

        await self._update(session_id, kind, generation, lambda s: s.update(state="applying"))
        async with self._limit:
            self.in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
            try:
                outcome = await self.driver.apply(
                    session_id, kind, slot["value"], effort, request_id=slot["request_id"]
                )
            except asyncio.CancelledError:
                raise
            except Exception as error:  # noqa: BLE001 — the driver classifies; this is the net
                logger.debug("settings intent attempt raised", exc_info=True)
                outcome = Outcome("retry", "runtime_unreachable", str(error))
            finally:
                self.in_flight -= 1
        if outcome.status == "applied":
            # A transport ACK is not proof: old runtimes can acknowledge before
            # their journal projection catches up. Confirm only from durable
            # selection (or the request-correlated result field below).
            try:
                after = await self.driver.current(session_id)
            except Exception:  # noqa: BLE001 — poll again, never strand applying
                after = Current()
            target = {**slot, "effort": effort} if kind == "model" else slot
            result = after.settings_result
            confirmed_result = (
                result
                and result.get("request_id") == slot["request_id"]
                and result.get("ok") is True
            )
            if not self._confirmed(kind, target, after) and not confirmed_result:
                outcome = Outcome("queued")
            elif (
                pending_effort is not None
                and after.effort is not None
                and after.effort != pending_effort[1]
            ):
                # The paired level was clamped (m5): the model landed, but the
                # separate effort slot did NOT get what it asked for. Leave it
                # active so its own attempt reaches the owner and is either
                # applied or refused there — never marked applied by proxy.
                pending_effort = None
        await self._settle(
            session_id,
            kind,
            generation,
            outcome,
            effort=effort,
            value=slot["value"],
            paired_generation=pending_effort[0] if pending_effort else None,
        )

    async def _paired_effort(self, session_id: str) -> tuple[int, str] | None:
        document = await self._load(session_id)
        slot = document["slots"].get("effort")
        if slot is None or slot.get("state") not in ACTIVE_STATES:
            return None
        return int(slot["generation"]), str(slot["value"])

    async def _settle(
        self,
        session_id: str,
        kind: str,
        generation: int,
        outcome: Outcome,
        *,
        effort: str | None = None,
        value: Any = None,
        paired_generation: int | None = None,
    ) -> None:
        now = self.clock()
        state = self._state(session_id)

        def mutate(slot: dict[str, Any]) -> None:
            slot["attempts"] = int(slot.get("attempts", 0)) + 1
            if outcome.status == "applied":
                slot.update(state="applied", error=None, next_attempt_at=None)
                return
            if outcome.status == "queued":
                if kind == "model" and effort is not None:
                    slot["effort"] = effort
                    slot["_paired_generation"] = paired_generation
                state.queued_since[(kind, generation)] = self.clock()
                slot.update(state="queued_next_step", error=None, next_attempt_at=_iso(now))
                return
            if outcome.status == "terminal":
                slot.update(
                    state="failed",
                    next_attempt_at=None,
                    error=_error(outcome.code, outcome.message, retryable=False),
                )
                return
            if outcome.status == "waiting_idle":
                slot.update(
                    state="waiting_idle",
                    error=None,
                    next_attempt_at=_iso(now + CWD_IDLE_RETRY_S),
                )
                return
            if outcome.status == "exclusive":
                since = state.exclusive_since.setdefault(generation, self.clock())
                if self.clock() - since >= CWD_EXCLUSIVE_BUDGET_S:
                    slot.update(
                        state="failed",
                        next_attempt_at=None,
                        error=_error("move_exclusive", outcome.message, retryable=True),
                    )
                    return
                slot.update(state="pending", next_attempt_at=_iso(now + CWD_IDLE_RETRY_S))
                return
            # retryable
            delay = retry_delay(slot["attempts"])
            deadline = _epoch(slot.get("deadline_at")) or now
            if now + delay > deadline:
                from local_operator.i18n.keys import wire_errors_desktop as _wd

                # NO NEW COPY: the sentence is the last attempt's own (the
                # runtime's or the bridge's, passed through verbatim), or the
                # catalogued "owner unavailable" one when it had none. The UI
                # switches on the code and composes its own chip from it.
                envelope = (
                    _wd.desktop_refusal(error=outcome.message)
                    if outcome.message
                    else _wd.runtime_unreachable()
                ).envelope(code="retry_budget_exhausted")
                slot.update(
                    state="failed",
                    next_attempt_at=None,
                    error=_error(
                        "retry_budget_exhausted",
                        envelope["text"],
                        retryable=True,
                        last_code=outcome.code or "runtime_unreachable",
                        attempts=slot["attempts"],
                    ),
                )
                return
            slot.update(
                state="pending",
                next_attempt_at=_iso(now + delay),
                error=_error(
                    outcome.code or "runtime_unreachable", outcome.message, retryable=True
                ),
            )

        updated = await self._update(session_id, kind, generation, mutate)
        if updated is None:
            # A newer pick replaced this slot while the attempt was in flight:
            # the result is discarded. But if OUR older attempt landed, the
            # session now reads the older value, which is neither the newer
            # slot's base nor its value — and the supersede rule would read our
            # own landing as "changed elsewhere" and drop the user's latest pick.
            # Re-base the newer slot on what we just applied.
            if outcome.status in {"applied", "queued"} and value is not None:
                await self._rebase(session_id, kind, value)
            return
        if kind == "model" and effort is not None:
            # The paired effort landed (or failed) with the model.
            paired = await self._paired_effort(session_id)
            if paired is not None and paired[1] == effort and paired[0] == paired_generation:
                await self._update(
                    session_id,
                    "effort",
                    paired[0],
                    lambda s: s.update(
                        state=updated["state"],
                        error=updated.get("error"),
                        next_attempt_at=updated.get("next_attempt_at"),
                        attempts=updated.get("attempts", 0),
                    ),
                )
        if kind == "model" and updated["state"] == "applied":
            await self._rebase_after_model(session_id)

    async def _rebase(self, session_id: str, kind: str, base: Any) -> None:
        state = self._state(session_id)
        async with state.lock:
            document = await self._load(session_id)
            slot = document["slots"].get(kind)
            if slot is not None and slot.get("state") in ACTIVE_STATES:
                slot["base"] = base
                await self._save(session_id, document)

    async def _rebase_after_model(self, session_id: str) -> None:
        """A model switch moves effort and fast: re-base those slots on the result.

        Without this the supersede rule would read the model's new default level
        (or fast reset to off) as "changed elsewhere" and drop the user's pick.
        """
        try:
            current = await self.driver.current(session_id)
        except Exception:  # noqa: BLE001
            return
        state = self._state(session_id)
        changed: list[tuple[str, dict[str, Any]]] = []
        async with state.lock:
            document = await self._load(session_id)
            for kind in ("effort", "fast"):
                slot = document["slots"].get(kind)
                if slot is not None and slot.get("state") in ACTIVE_STATES:
                    reading = current.of(kind)
                    if reading is not None:
                        slot["base"] = reading
                        changed.append((kind, slot))
            if changed:
                await self._save(session_id, document)

    async def _confirm(self, session_id: str, kind: str, slot: dict[str, Any]) -> None:
        """A queued ack: poll the journal until the target shows, resend once at 60 s."""
        generation = int(slot["generation"])
        state = self._state(session_id)
        key = (kind, generation)
        try:
            current = await self.driver.current(session_id)
        except Exception:  # noqa: BLE001
            current = Current()
        result = current.settings_result
        if result and result.get("request_id") == slot["request_id"]:
            if result.get("ok") is False:
                await self._settle(
                    session_id,
                    kind,
                    generation,
                    Outcome(
                        "terminal",
                        str(result.get("code") or "refused"),
                        str(result.get("message") or ""),
                    ),
                )
                return
        if (
            result and result.get("request_id") == slot["request_id"] and result.get("ok") is True
        ) or self._confirmed(kind, slot, current):
            await self._update(
                session_id, kind, generation, lambda s: s.update(state="applied", error=None)
            )
            if kind == "model":
                paired = await self._paired_effort(session_id)
                if (
                    paired is not None
                    and paired[1] == slot.get("effort")
                    and paired[0] == slot.get("_paired_generation")
                    # A clamped level (m5) leaves the effort slot to its own
                    # attempt rather than claiming it applied by proxy.
                    and (current.effort is None or current.effort == paired[1])
                ):
                    await self._update(
                        session_id, "effort", paired[0], lambda s: s.update(state="applied")
                    )
                await self._rebase_after_model(session_id)
            return
        if self._superseded(kind, slot, current):
            await self._update(session_id, kind, generation, lambda s: s.update(state="superseded"))
            return
        now = self.clock()
        deadline = _epoch(slot.get("deadline_at")) or now
        since = state.queued_since.setdefault(key, self.clock())
        if self.clock() - since >= CONFIRM_WINDOW_S and key not in state.resent:
            state.resent.add(key)
            await self._attempt(session_id, kind, slot)
            return
        if now + CONFIRM_POLL_S > deadline:
            await self._settle(
                session_id, kind, generation, Outcome("retry", "unconfirmed", ""), effort=None
            )
            return
        await self._update(
            session_id,
            kind,
            generation,
            lambda s: s.update(next_attempt_at=_iso(now + CONFIRM_POLL_S)),
        )


# -- the production driver ----------------------------------------------------


#: The runtime's own refusal sentence for ``/fast on`` on a model whose route
#: has no fast tier: ``serving._fast_slash`` and the TUI's ``/fast`` both start
#: with it (and, unlike every other refusal, style it ``info``). Matching the
#: runtime's own words is the only signal a released runtime gives; a
#: ``data.code`` of ``fast_mode_unsupported`` is honoured too, so a runtime that
#: starts tagging the refusal needs no change here.
FAST_UNSUPPORTED_PREFIX = "fast mode: not available"


def _slash_outcome(kind: str, payload: Any, *, requested: Any = None) -> Outcome:
    """Classify a routed slash's typed result (``SlashResult`` dict).

    ``queued`` is the deferred ack a runtime with ``defer_ok`` support answers
    (PR "settings-deferred-ack"); an older runtime answers a plain notice. A
    notice styled ``error`` (model) or ``warning`` (effort/fast) is the owner's
    refusal sentence — terminal, a retry would get the same answer.
    """
    if payload is None:
        return Outcome("applied")
    data = payload if isinstance(payload, Mapping) else getattr(payload, "__dict__", {})
    raw_fields = data.get("fields")
    raw_inner = data.get("data")
    fields: Mapping[str, Any] = raw_fields if isinstance(raw_fields, Mapping) else {}
    inner: Mapping[str, Any] = raw_inner if isinstance(raw_inner, Mapping) else {}
    if data.get("detail") == "queued" or fields.get("queued") or inner.get("queued"):
        return Outcome("queued")
    style = str(data.get("style") or "")
    text = str(data.get("text") or "")
    if (
        kind == "fast"
        and requested is not False
        and (
            inner.get("code") == "fast_mode_unsupported"
            or text.strip().lower().startswith(FAST_UNSUPPORTED_PREFIX)
        )
    ):
        # QA Q1: an ``info`` notice, but a refusal. Read as "applied" it fell to
        # the journal confirmation, which never sees fast on, and failed only at
        # the end of the budget as "owner unavailable". Terminal with the
        # runtime's own sentence and the draft boundary's existing code.
        return Outcome("terminal", "fast_mode_unsupported", text)
    if data.get("kind") == "error" or style == "error":
        return Outcome("terminal", "model_unknown" if kind == "model" else "refused", text)
    if style == "warning":
        code = {"model": "model_unknown", "effort": "effort_unsupported"}.get(kind, "refused")
        return Outcome("terminal", code, text)
    return Outcome("applied")


def classify_exception(error: BaseException) -> Outcome:
    """Map what the bridge raises onto retryable vs terminal.

    Busy and unreachable are BOTH retryable within the budget; which of the two
    a given failure is belongs to ``routes/desktop_sessions.py``'s classifier and
    is only carried here as a code for the UI.
    """
    from fastapi import HTTPException

    from local_operator.model.configure import ModelSelectionRefused
    from local_operator.session.attached import RuntimeUnresponsiveError
    from local_operator.session.errors import RuntimeRetiring

    if isinstance(error, RuntimeRetiring):
        return Outcome("retry", "runtime_retiring", str(error))
    if isinstance(error, ModelSelectionRefused):
        return Outcome("terminal", "model_unknown", error.message)
    if isinstance(error, RuntimeUnresponsiveError):
        return Outcome("retry", "runtime_busy", str(error))
    if isinstance(error, ConnectionError) and getattr(error, "owner_alive", False):
        return Outcome("retry", "runtime_busy", str(error))
    if isinstance(error, HTTPException):
        detail = error.detail if isinstance(error.detail, Mapping) else {}
        code = str(detail.get("code") or "")
        text = str(detail.get("text") or detail.get("message") or error.detail)
        if error.status_code == 404:
            return Outcome("terminal", "session_unknown", text)
        if error.status_code >= 500 or code == "runtime_busy":
            return Outcome("retry", code or "runtime_unreachable", text)
        return Outcome("terminal", code or "refused", text)
    if isinstance(error, (ConnectionError, TimeoutError, OSError)):
        return Outcome("retry", "runtime_unreachable", str(error))
    if isinstance(error, ValueError):
        return Outcome("terminal", "refused", str(error))
    return Outcome("retry", "runtime_unreachable", str(error))


class BridgeDriver:
    """Applies a slot through the session pool's bridge — the desktop's own path.

    Model and effort use the public attach client's setters after the bridge
    binds, with the same owner-provider admission as routed ``/model``. The
    model's effort rides ONE frame, so no provider call can see half the pair.
    Fast uses the client's typed slash result (the old routed seam when that
    client predates deferred kwargs). cwd uses ``move_session``, the existing
    transaction behind the TUI and desktop move. No watch refresh is involved.
    """

    def __init__(self, pool_of: Callable[[], Any]) -> None:
        self._pool_of = pool_of

    async def current(self, session_id: str) -> Current:
        from local_operator.server.utils.desktop_sessions import (
            read_desktop_marker,
            stored_draft_model,
        )
        from local_operator.session.cold_model import configured_model_pair
        from local_operator.session.model_selection import read_model_selection

        pool = self._pool_of()
        directory = Path(pool.root) / "sessions" / session_id

        def read() -> Current:
            marker = read_desktop_marker(directory) or {}
            selection = read_model_selection(directory, read_only=True)
            if selection is not None:
                # Reuse the journal's own selection; no bridge or transcript
                # hydration is needed, and this read has a stat-keyed memo.
                return Current(
                    model=selection.selector,
                    effort=selection.effort,
                    fast=selection.fast,
                    cwd=marker.get("cwd"),
                )
            birth = stored_draft_model(marker)
            # No selection yet: cold_model supplies the configured pair. Do not
            # hydrate model metadata here: unlike the actual cold-open painter,
            # the intent POST only needs identity and must never discover over
            # HTTP. A draft's explicit birth selection outranks machine config.
            provider, model_id = configured_model_pair(Path(pool.root))
            return Current(
                model=(
                    f"{birth['provider']}/{birth['model_id']}"
                    if birth
                    else f"{provider}/{model_id}"
                ),
                effort=birth.get("reasoning_effort") if birth else None,
                fast=birth.get("fast_mode", False) if birth else None,
                cwd=marker.get("cwd"),
            )

        current = await asyncio.to_thread(read)
        bridge = pool.bridges.get(session_id)
        if bridge is not None and bridge.remote is not None:
            # Only read an ALREADY ATTACHED bridge, never acquire one for a GET
            # or confirmation poll. PR1's canonical result is request-correlated.
            try:
                result = getattr(bridge.remote.frontend_state, "settings_result", None)
                if result is not None:
                    current = Current(
                        **{
                            **current.__dict__,
                            "settings_result": (
                                result.model_dump() if hasattr(result, "model_dump") else result
                            ),
                        }
                    )
            except RuntimeError:
                pass  # cold / not yet synchronized; the journal is sufficient
        return current

    @staticmethod
    def _deferred(method: Any, request_id: str) -> dict[str, Any]:
        # PR1 is additive. Old Python clients don't accept these kwargs even
        # though old RUNTIMES ignore unknown wire fields, so probe the client
        # signature rather than catching TypeError after a mutation may run.
        parameters = inspect.signature(method).parameters
        return {"defer_ok": True, "request_id": request_id} if "defer_ok" in parameters else {}

    async def apply(
        self, session_id: str, kind: str, value: Any, effort: str | None, *, request_id: str
    ) -> Outcome:
        # Review m1: an owner's explicit error FRAME reaches us as a bare
        # ``RuntimeError`` from the client call, and that is a refusal. The same
        # class is raised LOCALLY before any frame is sent (``frontend state has
        # not synchronized`` from the admission read on a fresh attach), which
        # is transient. Only errors raised once the owner call has started are
        # read as the owner's answer.
        owner_call = False
        pool = self._pool_of()
        try:
            async with pool.session(session_id) as bridge:
                if kind == "cwd":
                    return await self._move(bridge, str(value))
                remote = bridge.remote
                if remote is None:
                    return Outcome("retry", "runtime_unreachable", "")
                await remote.bind_runtime()
                client = remote._client
                if client is None:
                    return Outcome("retry", "runtime_unreachable", "")
                if kind == "model":
                    # Retain the routed model admission check (older owners may
                    # lack it), but send ONE set_model frame so effort lands
                    # atomically with the chosen identity, never as a later op.
                    remote._require_routed_model_admission(str(value))
                    provider, _, model_id = str(value).partition("/")
                    kwargs = self._deferred(client.set_model, request_id)
                    if effort is not None:
                        kwargs["effort"] = effort
                    owner_call = True
                    detail = await client.set_model(provider, model_id, **kwargs)
                    return Outcome("queued" if detail == "queued" else "applied")
                if kind == "effort":
                    owner_call = True
                    detail = await client.set_effort(
                        str(value), **self._deferred(client.set_effort, request_id)
                    )
                    return Outcome("queued" if detail == "queued" else "applied")
                if kind == "fast":
                    kwargs = self._deferred(client.slash_result, request_id)
                    owner_call = True
                    if kwargs:
                        payload = await client.slash_result(
                            "fast", "on" if value else "off", **kwargs
                        )
                    else:
                        payload = await remote.route_shared_slash("fast", "on" if value else "off")
                    return _slash_outcome("fast", payload, requested=value)
                return Outcome("terminal", "unknown_kind", kind)
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 — every result is persisted
            from local_operator.session.errors import RuntimeRetiring

            # Older attach clients decode an explicit owner error frame as
            # RuntimeError. That is a refusal, not a lost connection. Preserve
            # typed retirement as retryable; never resend an unknown-model or
            # invalid-effort refusal until the user makes a new request.
            if isinstance(error, RuntimeError) and not isinstance(error, RuntimeRetiring):
                if not owner_call:
                    return Outcome("retry", "runtime_busy", str(error))
                # Unrecognised owner text is a ``refused``, not a claim that the
                # model does not exist (``ModelSelectionRefused`` owns that code).
                code = "effort_unsupported" if kind == "effort" else "refused"
                return Outcome("terminal", code, str(error))
            if (
                isinstance(error, KeyError)
                and not (Path(pool.root) / "sessions" / session_id).is_dir()
            ):
                # Review m4: ``pool.session`` raises ``KeyError("Unknown
                # session")`` for a session deleted after the POST. Retrying
                # cannot bring it back, so fail now instead of spending the
                # budget. The directory check keeps an unrelated KeyError on a
                # live session on the retry path below.
                from local_operator.i18n.keys import wire_errors_desktop as _wd

                return Outcome(
                    "terminal",
                    "session_unknown",
                    _wd.requested_not_found().envelope(code="session_unknown")["text"],
                )
            if kind == "effort" and isinstance(error, ValueError):
                return Outcome("terminal", "effort_unsupported", str(error))
            return classify_exception(error)

    async def _move(self, bridge: Any, target: str) -> Outcome:
        # SHIELDED: a cancelled move (server shutdown) must not release
        # ``move_lock`` while its marker write and retire are in flight — the
        # working-directory route's ``_join_owned`` states the same hazard.
        from local_operator.owned_tasks import join_owned
        from local_operator.server.utils.desktop_sessions import move_session
        from local_operator.session.errors import MoveIndeterminate
        from local_operator.session.runtime.types import VIEWED_MOVE_REFUSAL

        async def transaction() -> dict[str, Any]:
            return (await move_session(bridge, target)).model_dump(mode="json")

        task = asyncio.create_task(transaction())
        try:
            await join_owned(task)
        except MoveIndeterminate as error:
            # The owner may have accepted: the next attempt either finds the
            # session already there (``unchanged`` → applied) or retries.
            return Outcome("retry", "runtime_unreachable", str(error))
        except RuntimeError as error:
            text = str(error)
            if "working right now" in text:
                return Outcome("waiting_idle", "runtime_busy", text)
            if text.startswith(VIEWED_MOVE_REFUSAL[:40]) or "older desktop window" in text:
                return Outcome("exclusive", "move_exclusive", text)
            return Outcome("terminal", "move_refused", text)
        return Outcome("applied")


def new_request_id() -> str:
    return str(uuid.uuid4())


def iter_kinds(slots: Mapping[str, Any]) -> Iterable[str]:
    return (kind for kind in KINDS if kind in slots)
