"""HTTP viewers of canonical runtimes, never a second execution host.

One bridge is shared by concurrent HTTP operations and event subscribers. Its
receipt sequence is deliberately independent of the runtime's frontend revision:
a snapshot covers paint state, not semantic receipts such as steering delivery.
The last reader detaches; neither socket disposal nor HTTP shutdown stops work.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import os
import re
import sqlite3
import time
import uuid
from collections import deque
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from anyio import CancelScope
from fastapi import HTTPException

from local_operator.harness.types import ModelSpec
from local_operator.resume import (
    ORIGIN_SUBAGENT,
    is_user_session,
    read_session_attachment,
    session_origin,
    session_preview,
    write_session_attachment,
)
from local_operator.server.models.desktop_sessions import MoveReceipt
from local_operator.server.retire import RETIRING_MESSAGE, DaemonRetiring
from local_operator.session.attached import AttachedSession
from local_operator.session.attachments import ATTACHMENTS_DIRNAME, AttachmentStore
from local_operator.session.attention import AttentionStore
from local_operator.session.catalog import load_catalog
from local_operator.session.cold_model import resolve_birth_effort
from local_operator.session.frontend_state import (
    FrontendSync,
    FrontendUpdate,
    sync_wire_payload,
)
from local_operator.session.restored_rows import record_field, roster_records
from local_operator.session.retention import DESKTOP_MARKER_NAME
from local_operator.session.transcript import (
    TRANSCRIPT_FILENAME,
    Transcript,
    read_transcript_page,
)

# The move shares the TUI's own `/move` machinery rather than a second resolver:
# `expand_path` resolves a relative target against the SESSION's directory and
# keeps symlinks, which is the rule a move must follow (`resolve_working_directory`
# below deliberately resolves against THIS process's cwd, which is right for
# `sessions.create`, whose session does not exist yet, and wrong for a move).
# The module is widget-free by design, so a non-Textual frontend can reuse it.
from local_operator.tui.move_targets import (
    expand_path,
    format_label,
    remember_recent,
    validate_target,
)

logger = logging.getLogger(__name__)

#: The page ceiling one child read may ask for, in ONE place. The route declares
#: it on the wire (FastAPI answers a bigger ``limit`` with 422 before the
#: handler runs) and the adapter refuses a direct caller the same way, so the
#: sentence and the number cannot drift apart (review round 1, R1-6).
CHILD_PAGE_LIMIT = 500

SESSION_ID = re.compile(r"^[a-f0-9]{12}$")
REPLAY_COUNT = 256
REPLAY_BYTES = 8 * 1024 * 1024
SUBSCRIBER_COUNT = 32
BRIDGE_COUNT = 64
#: The BRIDGE's subscription lease: what the renderer renews with a heartbeat,
#: and how long a bridge-backed warm intent lives with no beat behind it.
#:
#: NOT the runtime's lifetime, which is the runtime-side
#: ``DESKTOP_WATCH_LEASE_S`` (``session/runtime/types.py``), read a third time by
#: the dial (``attached.py::_dial``). The two are 45 s by agreement rather than by
#: construction, and since a live visible lease now CREATES the runtime, a
#: mismatch is not cosmetic: if this one were raised alone the bridge would keep
#: a lease it calls live while the reaper had already stopped counting the
#: viewer, so the warmed runtime would idle out under a window still waiting to
#: use it. Change them together, or make one derive from the other (review round
#: 1: the architect's cross-reference nit on these two constants).
WATCH_TTL = 45.0

#: Pace of the lease-driven warm, in three parts, because a warm that cannot
#: succeed must not become a spawn per heartbeat.
#:
#: A heartbeat is 15 s and a lease is 45 s, so an attempt left on that cadence
#: is a child spawn every beat for a session whose runtime cannot start
#: (provider credential gone, an MCP hang, an unwritable config dir). That is
#: not a latency optimisation any more, it is an unattended loop, so a FAILED
#: attempt — one that really did try to spawn and left the viewer cold — waits
#: out ``_LEASE_WARM_BACKOFF_S`` and doubles from there to the ceiling. The
#: honest cost of the doubling is that a transient failure is not retried for
#: 30 s; the fallback is the cold bind that shipped before this feature, while
#: the alternative is a machine running out of RAM because a window is open.
#: The ceiling is not a give-up point: the loop keeps re-asking at 120 s for as
#: long as the lease lives, because that is the same intent the beat expressed,
#: at 1/120 of the rate — and it is logged, since a bind that cannot start has
#: no other surface on this path.
#:
#: An attempt that did NO WORK (the facade was in owner recovery, or an engage
#: was already in flight) is explicitly not a failure and does not spend the
#: backoff: it is retried at ``_LEASE_WARM_POLL_S``, which is what carries the
#: intent across the ~8 s recovery window that used to swallow a warm until the
#: next beat. Keep it well under a heartbeat: a click inside that window pays
#: the cold bind this feature exists to remove.
_LEASE_WARM_BACKOFF_S = 30.0
_LEASE_WARM_BACKOFF_CAP_S = 120.0
_LEASE_WARM_POLL_S = 1.0

#: The completion kinds the DESKTOP BRIDGE may put on the wire as a
#: ``notification`` frame. Narrower than ``NotificationKind`` on purpose.
#:
#: ``ask``/``approval`` are absent because they already reach the desktop as
#: ``pending_gate`` in the snapshot and update frames, and a second channel for
#: the same card is the duplicate this whole contract exists to prevent.
#:
#: ``interrupted`` is absent because the user pressed Ctrl+C or Esc a moment
#: ago and already knows — telling them their own stop worked is the definition
#: of a notification nobody wants, which is the same call the TUI already
#: makes. The counter-argument (on the desktop an interruption can come from
#: another surface) is real but undecidable here: ``AgentEndEvent`` carries
#: ``aborted`` with no actor. One frozenset entry away if that ever changes.
BRIDGE_NOTIFIABLE_KINDS = frozenset({"complete", "error"})


async def _no_takeover() -> None:
    raise RuntimeError("Desktop viewers cannot own a runtime")


def _never_retiring() -> bool:
    """The default admission probe: a host with no retirement watching it.

    Returning False is the whole of the contract — a ``DesktopSessions`` built
    by a test or a reduced app (there are many) must behave exactly as it did
    before this probe existed, and only the daemon's lifespan wires the real
    one (``routes/desktop_sessions.py::host``).
    """
    return False


def resolve_working_directory(cwd: str) -> Path:
    """The directory ``cwd`` names, or ``ValueError`` (→ 409) if it is not one.

    Shared by ``DesktopSessions.create`` and the desktop's draft preview, so the
    SAME body gets the same answer from either route. A working directory that
    does not exist cannot start a session, so a strip describing one would be
    reporting readings for a session that could never be created — which is
    exactly what the preview's own contract refuses for an unresolvable profile.
    """
    directory = Path(cwd).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError("Choose an existing working directory")
    return directory


#: The ``desktop.json`` key carrying the model a draft was CREATED on. Additive
#: on purpose: a marker written before this key existed (``{"version": 1,
#: "cwd": ...}``) must keep loading exactly as it did, and every other reader of
#: the marker (``session.catalog``, ``DesktopSessionBridge``'s cwd lookup) reads
#: the keys it knows and ignores the rest.
DRAFT_MODEL_KEY = "model"


#: The three fields the draft selection travels with, on the HTTP wire and inside
#: the marker: the same shape the canonical frontend state publishes for a
#: conversation's model, so the picker's row can be handed back unmodified.
DRAFT_MODEL_FIELDS = ("provider", "model_id", "reasoning_effort")


def read_desktop_marker(session_dir: Path) -> dict[str, Any] | None:
    """Parse ``desktop.json``, or ``None`` when it is absent or unusable.

    Tolerant on the same terms as :func:`local_operator.resume.read_session_attachment`
    and for the same reason: this file is written by a process that can be killed
    mid-write, and a marker that cannot be read must cost the caller the value it
    was after — never the session. A caller that has no use for an unreadable
    marker (a cwd lookup) falls back exactly as it did when the file was missing.
    """
    try:
        raw = (session_dir / DESKTOP_MARKER_NAME).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    try:
        payload = json.loads(raw)
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def stored_draft_model(marker: dict[str, Any] | None) -> dict[str, str | None] | None:
    """The draft selection a marker carries, or ``None`` when it carries none.

    Every field is re-checked here rather than trusted: the marker is a plain
    JSON document that a hand edit, an interrupted write or an older build can
    shape arbitrarily, and the value this feeds (a model the child runtime will
    run on) must not depend on it having been written by this code.
    """
    choice = (marker or {}).get(DRAFT_MODEL_KEY)
    if not isinstance(choice, dict):
        return None
    provider = choice.get("provider")
    model_id = choice.get("model_id")
    if not isinstance(provider, str) or not provider:
        return None
    if not isinstance(model_id, str) or not model_id:
        return None
    effort = choice.get("reasoning_effort")
    return {
        "provider": provider,
        "model_id": model_id,
        "reasoning_effort": effort if isinstance(effort, str) and effort else None,
    }


def draft_birth_selection(root: Path, session_id: str) -> ModelSpec | None:
    """The selection a session was CREATED on, or ``None`` to use today's answer.

    This is the ONE place the birth choice is turned into a spec, and it returns
    ``None`` — meaning "no seed, resolve from config/journal exactly as before" —
    in every case where seeding would be wrong or impossible:

    * the marker carries no selection (every session created by an older build,
      by the TUI, or by a desktop create that omitted ``model``): the launch is
      then byte-for-byte today's;
    * the session's own journal already owns a selection. ``read_model_selection``
      is consulted here rather than left to :func:`cold_model.resolve_conversation_model`
      because the birth seed travels with ``model_selection_override``, and an
      override applied to a conversation that later switched models would drag it
      back to its birth model on the next open. The journal's precedence is only
      real if the override is never set in the first place;
    * the stored provider is gone from the registry, or the stored model id is no
      longer served by that provider's catalogue: a vanished pair must degrade to
      the configured default, never fail a resume or 400 the first turn;
    * the pair cannot be resolved into a spec at all (metadata is best-effort by
      contract, and a missing window is not worth a failed open).

    A stored level of ``null`` is the one case that seeds the PAIR and the
    machine's CONFIGURED level: the marker records a model the user picked and no
    choice about its reasoning effort, so the level is the one every launch that
    named no level resolves (R1). The marker itself keeps the ``null`` — the
    reading here is what the first turn will RUN at, not a choice that was made.

    Runs OFF the event loop: it reads the marker, the journal and — for an
    unshipped model — the provider's cached listing. The journal scan is guarded
    by the marker check above, so a session that carries no birth choice (the
    overwhelming majority) pays nothing for this.
    """
    choice = stored_draft_model(read_desktop_marker(root / "sessions" / session_id))
    if choice is None:
        return None
    from local_operator.providers.registry import get_provider_definition

    provider = str(choice["provider"])
    model_id = str(choice["model_id"])
    if get_provider_definition(provider) is None:
        logger.info("draft birth model names an unknown provider; using the default")
        return None
    from local_operator.model.discovery import offered_model_ids

    known = offered_model_ids(provider)
    if known is not None and model_id not in known:
        logger.info("draft birth model is no longer served; using the default")
        return None
    from local_operator.session.model_selection import read_model_selection

    if read_model_selection(root / "sessions" / session_id) is not None:
        return None
    from local_operator.model.configure import build_model_spec

    try:
        spec = build_model_spec(provider, model_id)
    except Exception:  # noqa: BLE001 — metadata is never worth a failed open
        logger.debug("draft birth model could not be resolved", exc_info=True)
        return None
    # The level the first turn will RUN at: the stored choice clipped to today's
    # ladder, or — for a marker that stored ``null`` ("this model, no level") — the
    # machine's configured level, and the model's own seed only when the config has
    # no opinion. See :func:`session.cold_model.resolve_birth_effort` — the ONE
    # resolver, which the preview answers through the same synthesis, so the pane and
    # the first cold frame cannot disagree.
    #
    # ``spec`` still carries its SEED here (it is ``build_model_spec``'s result),
    # which that function's third case requires.
    resolved = resolve_birth_effort(spec, choice["reasoning_effort"], root)
    if resolved != spec.reasoning_effort:
        spec = spec.model_copy(update={"reasoning_effort": resolved})
    return spec


def write_desktop_marker(
    path: Path, directory: Path, *, model: dict[str, Any] | None = None
) -> None:
    """Write the desktop draft marker in the session directory ``path``.

    ONE WRITER FUNCTION, TWO CALLERS, and that is why it exists as a function.
    ``catalog.py`` justifies skipping a stat in its scan with "``desktop.json``
    has exactly ONE writer — ``DesktopSessions.create``"; a move is the second
    CALLER, and what keeps that scan's saving sound is that no other module ever
    writes this file. Factoring the write out keeps the claim true in the form
    the scan can still benefit from — and the comment there now states it that
    way, because a stale "exactly one writer" is a lie the next reader believes.

    Synchronous, because both callers already hop to a worker thread for it
    (``asyncio.to_thread``): the desktop route is on the event loop and this is
    a filesystem write. The bytes are the contract ``locate()`` reads back
    (``{"version": 1, "cwd": …}``) and 0600 is its permission — a directory
    name is the user's data like any other.

    ``model`` is the DRAFT's chosen model (:data:`DRAFT_MODEL_KEY`), written only
    when the caller has one. A move has no opinion about the model and passes the
    one it read back, because ``cwd`` is the only field it is changing: a writer
    that reproduced the whole document from its own arguments would silently drop
    the choice the window made, which is the failure the additive-key comment on
    :data:`DRAFT_MODEL_KEY` exists to prevent.
    """
    marker = path / DESKTOP_MARKER_NAME
    payload: dict[str, Any] = {"version": 1, "cwd": str(directory)}
    if model is not None:
        payload[DRAFT_MODEL_KEY] = {field: model.get(field) for field in DRAFT_MODEL_FIELDS}
    marker.write_text(json.dumps(payload))
    marker.chmod(0o600)


async def move_session(bridge: DesktopSessionBridge, requested: str) -> MoveReceipt:
    """Point a desktop session at ``requested``; own every side effect of doing so.

    Validation, durability and the retire are ONE ordering, which is why this is
    a function rather than three calls in the route. The order is the rule
    :meth:`AttachedSession.set_working_directory` already states for ``_cwd``
    ("the field is set FIRST, before joining… so the successor cannot be engaged
    before the field it reads is set") applied to the two other copies of that
    field: the session's durable marker and the bridge's own ``cwd``. A
    successor spawned mid-move reads the marker when this bridge has been
    evicted and the bridge field when it has not, so both must be set before the
    retire makes either reachable.

    The three copies are kept AGREEING, and the previous bytes are what makes
    that possible: a refused retire (a turn that arrived during it, a runtime
    too old to move, a lost socket) restores the marker and the bridge field, so
    a failure leaves the session working where it did rather than half moved.
    """
    remote = bridge.remote
    assert remote is not None, "a move runs against an acquired bridge"
    # The VIEWER's live value, not ``bridge.cwd``: the bridge field is written by
    # THIS function and read at `acquire()`, so after a first move it is the
    # older of the two and a relative path resolved from it would name the wrong
    # sibling.
    previous = remote.cwd
    try:
        directory = validate_target(expand_path(requested, cwd=previous))
    except OSError as error:
        # A path on an unmounted volume, or a symlink loop. The TUI answers this
        # class of case in the same words (``_apply_move``), and it is NOT left to
        # the route's ``errors()`` ladder: that ladder has no OSError clause, so
        # an unmounted volume would reach the user as a 500.
        raise HTTPException(409, f"cannot move to {requested}: {error}") from None

    resolved = str(directory)
    label = format_label(directory)
    if os.path.normpath(resolved) == os.path.normpath(previous):
        # NOTHING is written and nothing is retired. This is the TUI's "already in
        # ~/x", and it is also what makes a retried move idempotent: the receipt
        # journal admits a second POST with the same request id, and a move that
        # had already landed must not retire the runtime a second time.
        # ``will_wait`` is False rather than sampled: no transition is about to
        # happen, so there is no wait for it to have been a hint about.
        return MoveReceipt(cwd=resolved, label=label, outcome="unchanged", will_wait=False)

    # Sampled BEFORE the call, exactly as ``_apply_move`` does, because after it
    # the answer is about a transition that has already happened — and this is a
    # HINT for an operator reading a receipt, never a gate.
    will_wait = remote.move_will_wait()

    # DURABILITY FIRST, the marker before the bridge field, and both before the
    # retire. `locate()` prefers the marker over the canonical checkpoint when
    # this bridge has been evicted or the HTTP server restarted; the bridge field
    # is what a re-``acquire()`` on THIS bridge passes to ``cold(cwd=…)``.
    marker_dir = bridge.root / "sessions" / bridge.session_id

    def read_marker() -> bytes | None:
        try:
            return (marker_dir / DESKTOP_MARKER_NAME).read_bytes()
        except OSError:
            # A session old enough to predate the marker, or an unreadable
            # directory. `None` is the honest "there was nothing there", and the
            # rollback below then removes what this call created.
            return None

    def restore_marker(previous_bytes: bytes | None) -> None:
        marker = marker_dir / DESKTOP_MARKER_NAME
        try:
            if previous_bytes is None:
                marker.unlink(missing_ok=True)
                return
            marker.write_bytes(previous_bytes)
            marker.chmod(0o600)
        except OSError:
            # A rollback that cannot run is worth a line in the log and NOT a
            # second exception: the refusal that caused it is what the user has
            # to see, and raising here would replace that sentence with a
            # filesystem error. The consequence is named because it is real —
            # the durable copy now says where the session is NOT, and only a
            # successful move will fix it.
            logger.warning(
                "could not restore the desktop marker for %s; a later acquire may "
                "resume in the refused directory",
                bridge.session_id,
                exc_info=True,
            )

    # The draft's stored model is CARRIED ACROSS, never re-derived: a move
    # changes ``cwd`` and nothing else, and the model key is the window's choice
    # for a conversation that has not run yet (``draft_birth_selection``). Read
    # through the same helpers every other marker reader uses, so a marker this
    # route could not parse degrades here exactly as it does there.
    previous_model = await asyncio.to_thread(
        lambda: stored_draft_model(read_desktop_marker(marker_dir))
    )
    previous_marker = await asyncio.to_thread(read_marker)
    await asyncio.to_thread(write_desktop_marker, marker_dir, directory, model=previous_model)
    bridge.cwd = resolved
    try:
        # The receipt's ``outcome`` IS this call's return vocabulary: the facade
        # documents exactly ``"cold"`` and ``"rebound"`` and nothing else, and
        # ``MoveReceipt`` spells the same two words plus the route-level
        # ``"unchanged"`` above. The cast narrows a ``str`` the protocol cannot
        # express (``RemoteSession.set_working_directory -> str``) to the Literal
        # the wire model owns, rather than the model being widened to ``str`` and
        # losing the fact a renderer switches on.
        outcome = cast(Literal["cold", "rebound"], await remote.set_working_directory(resolved))
    except BaseException:
        # BaseException, not Exception: a CANCELLED move is the same hazard as a
        # refused one — the retire may already be in flight while this call's
        # caller went away — and `set_working_directory` rolls its own field back
        # on the same terms (review MINOR-2 there). Restoring the OTHER two copies
        # is this function's half of that invariant.
        await asyncio.to_thread(restore_marker, previous_marker)
        bridge.cwd = previous
        raise

    # Best effort by contract, and off the loop because it is a read, a write and
    # an atomic replace. Sharing the recents list with the TUI is a bonus of
    # sharing the file, not the point of writing it here.
    await asyncio.to_thread(remember_recent, bridge.root, directory)

    return MoveReceipt(cwd=resolved, label=label, outcome=outcome, will_wait=will_wait)


@dataclass(eq=False)
class DesktopSubscription:
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    queue: asyncio.Queue[tuple[dict[str, Any], int] | None] = field(
        default_factory=lambda: asyncio.Queue(maxsize=REPLAY_COUNT)
    )
    queued_bytes: int = 0
    visible: bool = False
    can_notify: bool = False
    expires: float = 0.0
    overflow: bool = False


class DesktopSessionBridge:
    def __init__(
        self,
        root: Path,
        session_id: str,
        cwd: str,
        *,
        retiring: Callable[[], bool] | None = None,
    ) -> None:
        self.root, self.session_id, self.cwd = root, session_id, cwd
        # Why this bridge must not start a runtime, asked of the daemon's own
        # state rather than cached: the flag flips ONCE, mid-life, when the
        # retirement poll runs (``server/retire.py``), and a bridge built before
        # that must see it too. Defaulted so every direct construction — the
        # tests', a reduced app's — is an ordinary admitting bridge.
        self.retiring_probe = retiring or _never_retiring
        self.remote: AttachedSession | None = None
        self.epoch = uuid.uuid4().hex
        self.sequence = 0
        self.replay: deque[tuple[dict[str, Any], int]] = deque()
        self.replay_bytes = 0
        self.subscribers: dict[str, DesktopSubscription] = {}
        self.users = 0
        self.touched = time.monotonic()
        self.lock = asyncio.Lock()
        self.watch_lock = asyncio.Lock()
        self.unsubscribers: list[Any] = []
        self.watch_task: asyncio.Task[None] | None = None
        #: The in-flight speculative engage started by :meth:`warm`, held so
        #: the event loop keeps a strong reference to it. A bare
        #: ``create_task`` with no referent may be garbage collected mid-flight
        #: (asyncio only holds a weak reference), which would make the warm
        #: silently do nothing on an arbitrary subset of requests — the worst
        #: possible failure for a latency optimisation, because the slow path
        #: it leaves behind is the correct one.
        self.warm_task: asyncio.Task[None] | None = None
        #: The lease-driven warm's retry loop (see :meth:`_lease_warm_loop`).
        #: Separate from ``warm_task`` because it outlives one engage: it holds
        #: the LEASE's intent across attempts, while ``warm_task`` is the single
        #: engage it (or the ``/warm`` route) has in flight at any moment.
        self.lease_warm_task: asyncio.Task[None] | None = None
        #: Pace for the next lease-driven attempt, valid only between an attempt
        #: that failed and the retry it earned; ``warm_not_before`` is the
        #: monotonic deadline, ``warm_backoff_s`` the value it was set from so
        #: the next failure can double it. Both reset when the intent is served
        #: or the bridge detaches, so a fresh intent starts from the base.
        self.warm_backoff_s = 0.0
        self.warm_not_before = 0.0
        self.attention_task: asyncio.Task[None] | None = None
        self.attention: dict[str, Any] = {}
        self.attention_poll_key: tuple[tuple[int, int, int], bool] | None = None

    async def acquire(self) -> AttachedSession:
        async with self.lock:
            self.users += 1
            self.touched = time.monotonic()
            try:
                if self.remote is None:
                    # The birth selection this draft was created with, and the
                    # deliberate override that makes the child PIN it (a config
                    # edit must not re-select a conversation the user chose a
                    # model for). Both are ``None``/``False`` for every session
                    # that carries no stored choice, which is every session an
                    # older build created — and for one whose own journal already
                    # owns a selection, so a switched conversation is never
                    # dragged back to the model it was born on (see
                    # :func:`draft_birth_selection`).
                    birth = await asyncio.to_thread(
                        draft_birth_selection, self.root, self.session_id
                    )
                    remote = await AttachedSession.cold(
                        self.session_id,
                        config_dir=self.root,
                        cwd=self.cwd,
                        takeover_factory=_no_takeover,
                        surface="desktop",
                        initial_model=birth,
                        model_selection_override=birth is not None,
                    )
                    self.remote = remote
                    # A detached interval has no receipt feed. A new epoch makes
                    # that gap explicit even when the runtime itself never died.
                    self.epoch = uuid.uuid4().hex
                    self.sequence = 0
                    self.replay.clear()
                    self.replay_bytes = 0
                    # The engage's refresh hook, installed HERE because this is the only
                    # seam that owns this facade for its whole life.
                    #
                    # WHY IT IS NEEDED AT ALL: `retiring` means "a successor is owed,
                    # engage one" — the frame a move ends with, and a client-side build
                    # refresh too — and the facade answers it with `_go_cold(refresh=True)`,
                    # which fires this callback. The TUI installs one
                    # (`_on_runtime_refreshed`); without one the desktop viewer simply sat
                    # cold until the user's next send engaged, so a moved session's chip
                    # stayed on the OLD directory with nothing to settle it. (A
                    # WHOLE-DAEMON retirement is a different case with its own mechanism,
                    # `server/retire.py`: the callback still fires there and declines in
                    # `_schedule_warm`, because the daemon leaving needs no successor
                    # spawned inside it.) Nothing else can take this job:
                    # `attach_existing()` only adopts an EXISTING owner record (there is
                    # none after a retire), and warm()/prompt()/command only re-engage when
                    # the user next acts.
                    #
                    # WHY THE CALLBACK AND NOT "warm() after the move route returns": at
                    # the moment `set_working_directory` returns, the outgoing client is
                    # usually STILL connected (`retire_now` is acked before the EOF), so an
                    # engage issued there samples `is_cold` as False and returns without
                    # doing anything — silently. This callback runs on the exact frame
                    # that flips the viewer cold, which is the only moment that is not a
                    # race. It fires from `_on_disconnected` inside the client's pump, i.e.
                    # on the event loop, so `_schedule_warm` may create its task directly.
                    remote.set_refresh_callback(self._on_runtime_retired)
                    self.unsubscribers = [
                        remote.subscribe(self._event),
                        remote.subscribe_frontend(self._frontend).unsubscribe,
                    ]
                await self.remote.attach_existing()
                if self.attention_task is None:
                    self.attention_task = asyncio.create_task(self._poll_attention())
                return self.remote
            except BaseException:
                self.users -= 1
                if self.users == 0:
                    await self._detach()
                raise

    async def release(self) -> None:
        async with self.lock:
            self.users -= 1
            self.touched = time.monotonic()
            if self.users == 0:
                await self._detach()

    async def _detach(self) -> None:
        if self.watch_task is not None:
            self.watch_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.watch_task
            self.watch_task = None
        # THE LOOP GOES FIRST, before the engage it may be awaiting. Cancelling
        # the engage alone would resume the loop on a cancellation it did not
        # ask for and leave it to interpret that; cancelling the observer first
        # removes the question. `BaseException` rather than `CancelledError`
        # because awaiting a task that already died re-raises its exception, and
        # neither shape may abort teardown midway.
        #
        # Its backoff is forgotten here too: the facade it was armed against is
        # being disposed, so the next life of this bridge starts a fresh intent
        # from the base pace rather than inheriting a dead viewer's.
        if self.lease_warm_task is not None:
            self.lease_warm_task.cancel()
            with contextlib.suppress(BaseException):
                await self.lease_warm_task
            self.lease_warm_task = None
        self.warm_backoff_s = 0.0
        self.warm_not_before = 0.0
        # BEFORE `dispose()`, and suppressing the task's own failure as well as
        # the cancellation: an engage that lands after the facade is gone would
        # otherwise hold a freshly spawned runtime resident with no viewer to
        # release it. The facade already refuses a disposed bind and closes a
        # client that arrives late, so this is belt-and-braces — but the TUI
        # needed exactly this cancel for exactly this reason (a swap's engage
        # landing afterwards kept the old runtime up for the process's life),
        # and a bridge detaching mid-warm is the same shape by a different
        # route.
        #
        # KNOWN COST, not fixable from here (review round 1, MINOR-2). A
        # cancel that lands mid-engage leaks the spawn's capture tempfile
        # (``lop-runtime-*.log``): ``engage_runtime`` unlinks it on every
        # normal exit but has no ``finally``, and the path is a local of that
        # function -- nothing outside it can see, let alone unlink, the file.
        # The mechanism is pre-existing and its own docstring names it ("a
        # cancelled task would leak that tempfile"); the TUI's engage cancel
        # is the other trigger. Closing it means a ``try/finally`` inside the
        # shared launch loop, which would fix both surfaces at once -- and is
        # deliberately NOT done from this PR's new path, because a partial fix
        # here would be a second unlink site that disagrees with the first.
        # Bounded to one small file per navigate-away-during-warm, never a
        # live process (the spawned child is left to the residency drain).
        if self.warm_task is not None:
            self.warm_task.cancel()
            with contextlib.suppress(BaseException):
                await self.warm_task
            self.warm_task = None
        for unsubscribe in self.unsubscribers:
            unsubscribe()
        self.unsubscribers.clear()
        remote, self.remote = self.remote, None
        if remote is not None:
            await remote.dispose()
        # LAST, and suppressing the task's OWN failure rather than only
        # CancelledError: awaiting a task that already died re-raises its
        # exception, and with this block ahead of `dispose()` a store error
        # aborted teardown midway -- leaking the runtime's session and its
        # subscriptions while `users` had already reached 0, so a later
        # `acquire()` reused a half-torn bridge. A read receipt must never
        # strand a session runtime.
        if self.attention_task is not None:
            self.attention_task.cancel()
            with contextlib.suppress(BaseException):
                await self.attention_task
            self.attention_task = None

    async def close(self) -> None:
        for sub in self.subscribers.values():
            self._disconnect(sub)
        async with self.lock:
            await self._detach()

    def _disconnect(self, sub: DesktopSubscription) -> None:
        sub.overflow = True
        sub.visible = sub.can_notify = False
        while not sub.queue.empty():
            sub.queue.get_nowait()
        sub.queued_bytes = 0
        sub.queue.put_nowait(None)

    def publish(self, kind: str, payload: dict[str, Any], *, replay: bool = True) -> None:
        """Put one frame on every live subscriber, and normally into replay.

        ``replay=False`` publishes LIVE ONLY. It exists for the ``notification``
        frame, whose whole value is timeliness: replaying it on reconnect toasts
        the user about a turn that finished while their laptop lid was shut,
        possibly hours later. The durable signal for "you missed something" is
        not lost — the ``attention`` frame and the sidebar's unseen mark both
        survive a reconnect and are the right surface for it.

        THE SEQUENCE STILL ADVANCES for a non-replayed frame, and that is
        load-bearing rather than incidental. :meth:`events` computes ``gap``
        from ``after_seq < first - 1`` where ``first`` is the oldest RETAINED
        frame's seq, so a skipped seq simply never becomes ``first`` and a
        client reconnecting at the notification's own cursor still satisfies
        the test against the next retained frame. Not incrementing would
        instead make ``seq`` non-monotonic across the two paths and break the
        receipt cursor every renderer keeps.
        """
        self.sequence += 1
        frame = {
            "session_id": self.session_id,
            "epoch": self.epoch,
            "seq": self.sequence,
            "type": kind,
            "payload": payload,
        }
        size = len(json.dumps(frame, separators=(",", ":")).encode())
        if replay:
            self.replay.append((frame, size))
            self.replay_bytes += size
            while self.replay and (
                len(self.replay) > REPLAY_COUNT or self.replay_bytes > REPLAY_BYTES
            ):
                _, removed = self.replay.popleft()
                self.replay_bytes -= removed
        for sub in self.subscribers.values():
            if sub.overflow:
                continue
            if sub.queue.full() or sub.queued_bytes + size > REPLAY_BYTES:
                # Never silently discard a semantic event. Closing forces an
                # authoritative gap snapshot on reconnect, and revokes presence.
                self._disconnect(sub)
            else:
                sub.queue.put_nowait((frame, size))
                sub.queued_bytes += size

    def _event(self, event: Any) -> None:
        self.publish("event", event.model_dump(mode="json"))

    def _frontend(self, update: FrontendUpdate) -> None:
        # Keep the runtime's field deltas, not a full snapshot per streamed token.
        # Trajectories are intentionally opt-in on the runtime and absent here;
        # large roster/usage fields still pass through the shared wire budget.
        payload = update.model_dump(mode="json")
        # Receipt revisions outlive a runtime epoch. Only the independent durable
        # projection below may update them; a delayed runtime delta must not undo
        # a read made through another process while this stream stays mounted.
        payload["changes"].pop("attention", None)
        payload["job_trajectory_appends"] = {}
        payload["job_trajectory_replacements"] = []
        if {"jobs", "usage_components"} & update.changes.keys():
            bounded = self.state()["snapshot"]
            for key in ("jobs", "usage_components"):
                if key in payload["changes"]:
                    payload["changes"][key] = bounded[key]
        self.publish("frontend.update", payload)

    async def refresh_attention(self) -> dict[str, Any]:
        state = await asyncio.to_thread(
            AttentionStore(self.root / "attention.db").state, f"session/{self.session_id}"
        )
        remote = self.remote
        state["supported"] = bool(
            remote is not None
            and (remote.is_cold or getattr(remote, "supports_completion_ack", False))
        )
        if state != self.attention:
            previous = self.attention
            self.attention = state
            # The initial snapshot owns the baseline; later changes have their
            # own receipt clock rather than borrowing a runtime sequence.
            if previous:
                self.publish("attention", state)
                # THE NOTIFICATION EDGE, published AFTER the attention frame so
                # a reader that toasts already holds the receipt state that
                # explains the toast. The same `previous` baseline rule governs
                # both: a bridge's FIRST read is the session's history, not
                # news, and opening a conversation must not announce the
                # completion it ended on last week.
                await self._maybe_publish_notification(previous, state)
        return state

    async def _maybe_publish_notification(
        self, previous: dict[str, Any], state: dict[str, Any]
    ) -> None:
        """Turn a newly published, unseen completion into one notification frame.

        THE AUTHORITY IS THE ATTENTION PUBLICATION, not any engine event. A
        `completions` row exists only because ``Session._publish_attention_
        outcome`` decided the turn produced a notifiable outcome — in the
        process that owns the job manager, using the same delegated-children
        check the TUI uses (``job.type == "task" and job.status == "running"``).
        A delegating parent's premature ``agent_end`` writes an ``eligible:
        False`` marker and publishes nothing, so there is simply no row for the
        bridge to see, and each settled child re-enters as a fresh turn whose
        own completion publishes normally.

        That is why this method asks no questions about jobs, ``agent_end`` or
        ``turn_end``: reconstructing the decision here would mean making it
        again in a process with less information, which is how a frontend ends
        up disagreeing with the TUI about whether a turn finished. The bridge
        OBSERVES the decision; it does not judge it.

        NO CLAIM IS TAKEN HERE. ``claim_delivery`` is claim-then-deliver, and
        the claimant must be the deliverer — between this frame and an OS
        banner lie an SSE socket, the Electron main process, a support check
        and a focus gate. A claim taken here that the renderer then suppresses
        would mark the completion delivered while nobody was told, for good. A
        frame is an OFFER; the renderer claims through ``POST /notified``
        immediately before it shows the banner.

        Guarded end to end: a notification is chrome, and this runs inside the
        1 s attention poll whose loop already treats a store error as costing
        one tick rather than the feature.
        """
        token = state.get("completion_token")
        if (
            not token
            or token == previous.get("completion_token")
            or not state.get("unseen")
            or state.get("kind") not in BRIDGE_NOTIFIABLE_KINDS
        ):
            return
        try:
            from local_operator.notifications import (
                NOTIFICATION_CONTRACT_VERSION,
                compose,
            )

            # The LIVE name wins over the sidecar: a rename reaches frontend
            # state before it reaches `title.json`, and `compose` falls back to
            # the stored title on its own when this is empty (a cold bridge has
            # no runtime to ask).
            remote = self.remote
            session_name = ""
            if remote is not None:
                session_name = getattr(remote.frontend_state, "conversation_title", "") or ""
            # `compose` reads up to 128 KiB for the title and 64 KB for the
            # preview, and `refresh_attention` runs on the event loop. Off-loop
            # for the same reason the store read above is.
            composed = await asyncio.to_thread(
                compose,
                state["kind"],
                session_dir=self.root / "sessions" / self.session_id,
                session_name=session_name,
            )
            self.publish(
                "notification",
                {
                    "contract": NOTIFICATION_CONTRACT_VERSION,
                    "kind": composed.kind,
                    "title": composed.title,
                    "status": composed.status,
                    "body": composed.body,
                    "body_is_snippet": composed.body_is_snippet,
                    # Additive since the first draft of this frame; a renderer
                    # that does not know the field simply shows the body, which
                    # is already the right thing to do with it.
                    "body_is_failure": composed.body_is_failure,
                    "title_is_session_name": composed.title_is_session_name,
                    # Keyed on the DURABLE completion token rather than on this
                    # bridge's sequence: `acquire()` mints a new epoch and
                    # resets `sequence` to 0 after a detached interval, so a
                    # seq-keyed dedupe re-toasts the same completion on every
                    # reconnect. The prefix is the frame's own kind (round 1,
                    # n1): a token has exactly one kind, so it costs nothing,
                    # and a store or dedupe-map dump no longer reads as an
                    # error banner mislabelled `complete:`.
                    "dedupe_key": f"{composed.kind}:{self.session_id}:{token}",
                    "completion_token": token,
                    "session_name": composed.title if composed.title_is_session_name else None,
                    "focus_policy": "when_unfocused",
                },
                replay=False,
            )
        except Exception:  # noqa: BLE001 — chrome must not cost the attention poll
            logger.debug("notification compose failed for %s", self.session_id, exc_info=True)

    async def _poll_attention(self) -> None:
        # Read-only polling is shared by every subscriber of this bridge and
        # independent of watch leases. It also works while no runtime is running.
        #
        # The body is guarded because this store has other writers: a `database
        # is locked` that outlives its 2 s timeout is routine contention, and
        # letting it end the loop stopped cross-process read sync for the life
        # of the bridge -- the phone and the TUI would clear an unread
        # completion while the desktop kept showing it, silently and forever.
        # A transient store error must cost one poll, not the feature. Matches
        # the suppression `_expire_watches` already uses for the same reason.
        store = AttentionStore(self.root / "attention.db")
        failing = 0
        while True:
            try:
                # `revision()` exists for exactly this loop and is far cheaper
                # than the full per-conversation read; the steady state is a
                # store nothing has written since the last tick.
                #
                # The runtime's own state is part of the key because `supported`
                # is derived from it, not from the store: a runtime starting or
                # going cold changes that answer while the store is untouched,
                # so gating on the revision alone would pin `supported` to
                # whatever happened to be true when the bridge attached.
                remote = self.remote
                key = (
                    await asyncio.to_thread(store.revision),
                    remote is not None
                    and (remote.is_cold or getattr(remote, "supports_completion_ack", False)),
                )
                if key != self.attention_poll_key:
                    await self.refresh_attention()
                    self.attention_poll_key = key
                if failing:
                    logger.info(
                        "attention poll recovered for %s after %d failure(s)",
                        self.session_id,
                        failing,
                    )
                    failing = 0
            except Exception as error:
                # Log the TRANSITION, not the tick. A transient error costs one
                # line, but a persistent one (corrupt schema, permissions, full
                # disk) would otherwise write ~3,600 identical warnings an hour
                # per bridge, across up to BRIDGE_COUNT bridges, burying
                # whatever else the operator needs to read. Recovery is logged
                # too, so the pair brackets the outage rather than leaving a
                # single warning of unknown duration.
                failing += 1
                if failing == 1:
                    logger.warning(
                        "attention poll failed for %s (further failures quiet "
                        "until it recovers): %s",
                        self.session_id,
                        error,
                    )
            await asyncio.sleep(1)

    def state(self) -> dict[str, Any]:
        assert self.remote is not None
        state = self.remote.frontend_state.model_copy(update={"attention": self.attention})
        return sync_wire_payload(
            FrontendSync(
                epoch=state.epoch,
                sequence=state.sequence,
                snapshot=state,
                live_cursor=state.history_cursor,
            )
        )

    async def snapshot(self) -> dict[str, Any]:
        # Decorative, so a busy or damaged receipt sidecar cannot stop a
        # conversation from OPENING. Before this field existed the snapshot
        # never touched `attention.db`; letting it raise here turned routine
        # write contention into a failure of the primary read path. The last
        # known state is kept rather than blanked -- it is what the previous
        # successful poll actually saw.
        with contextlib.suppress(sqlite3.Error, OSError):
            await self.refresh_attention()
        state = self.state()
        seq, epoch = self.sequence, self.epoch
        cursor = state["snapshot"].get("history_cursor")
        history: dict[str, Any] = {"entries": [], "has_more": False, "cursor_missing": False}
        # The gate is about the STATE, not about bounding the page, and the empty
        # page it produces is a signal a reader ACTS on: "no history cursor, so
        # reconcile through /history". That promise lives in the other repository
        # (``local-operator-ui`` reconciles on an empty page or ``cursor_missing``),
        # which is why it is stated here and in ``docs/DESKTOP_API.md`` rather than
        # left to be inferred -- and why this branch is the one place where the
        # snapshot still serves something not derived from the journal.
        if cursor:
            # THE PAGE IS THE JOURNAL'S TAIL. Its upper bound is NOT the frontend
            # cursor above, and that is the fix rather than a detail: this is a
            # read of the TRANSCRIPT, while ``history_cursor`` is a FRONTEND
            # refresh watermark -- ``transcript.entries()[-1].id`` as of the
            # owning store's last ``refresh_from_session``
            # (``frontend_state.py``), which a turn advances only at its message,
            # tool and turn boundaries and which a checkpoint persists verbatim.
            #
            # Bounding one source's read by another source's watermark silently
            # LOSES rows. Any row durable past the watermark is outside the page,
            # and because the bound row itself is still on disk
            # ``read_transcript_page`` reports no ``cursor_missing`` -- so a
            # reader that reconciles only for an empty page or a missing cursor
            # (the desktop client does exactly that) accepts the short page as
            # complete and never learns the rows exist. Measured on the reported
            # flow: a steer drained mid-turn pins the watermark at the steer row,
            # every later row of that turn is then outside the page, and the user
            # sees a transcript that stops at their own steer. A viewer that lost
            # its owner (`_can_go_cold`) or a state restored from a checkpoint
            # that predates the rows reaches the same short page with no error.
            #
            # The cursor keeps its real job, and one job only: it is the
            # ``live_cursor`` DEDUPE watermark on the wire, telling a reader which
            # rows the paired frontend state has already accounted for. It is not
            # a pairing field for this read and never was one that needed the
            # truncation -- ``/history`` serves this very same unbounded tail, so
            # the page and the state were already paired on rows, not on the
            # bound. Nothing is dropped from the contract by not cutting at it,
            # and reads stay bounded by their OWN source's cut: see
            # ``read_transcript_page`` for the inclusive-boundary rule it still
            # applies when a caller asks for one.
            history = await self.history()
        return {
            "session_id": self.session_id,
            "epoch": epoch,
            "seq": seq,
            "type": "snapshot",
            "payload": {
                "frontend": state,
                "history": history,
                "cold": self.remote is None or self.remote.is_cold,
            },
        }

    async def history(
        self, *, before_id: str | None = None, through_id: str | None = None, limit: int = 100
    ) -> dict[str, Any]:
        """One page of the durable journal.

        ``through_id`` is the transcript-level inclusive cut and stays part of
        this method's contract, but NO DESKTOP CALLER PASSES IT ANY MORE: the
        snapshot used to bind the page to the paired frontend ``history_cursor``
        and that bound was the defect (a state watermark is not a visibility
        boundary over the journal -- see :meth:`snapshot`). Do not restore it
        here without that argument; a page that stops short of the journal loses
        rows silently, because the bound row is still on disk so
        ``read_transcript_page`` cannot report them missing. ``before_id``
        backward paging and this cut's direct use by
        ``read_transcript_page``'s own tests are what keep the parameter alive.
        """
        try:
            page = await asyncio.to_thread(
                read_transcript_page,
                self.root / "sessions" / self.session_id,
                before_id=before_id,
                through_id=through_id,
                limit=limit,
            )
        except FileNotFoundError:
            return {
                "entries": [],
                "has_more": False,
                "cursor_missing": bool(before_id or through_id),
            }
        return {
            "entries": [json.loads(row.to_json()) for row in page.entries],
            "has_more": page.has_more,
            "cursor_missing": page.reconciled,
        }

    async def watch(self, subscription_id: str, *, visible: bool, can_notify: bool) -> None:
        sub = self.subscribers.get(subscription_id)
        if sub is None or sub.overflow:
            raise KeyError("This event subscription is no longer connected")
        sub.visible, sub.can_notify = visible, can_notify
        sub.expires = time.monotonic() + WATCH_TTL
        await self.refresh_watch()
        if self.watch_task is None or self.watch_task.done():
            self.watch_task = asyncio.create_task(self._expire_watches())

    def _live_leases(self) -> list[DesktopSubscription]:
        """The subscriptions holding a LIVE lease. Caller holds ``watch_lock``.

        Extracted rather than inlined into :meth:`refresh_watch`, because the
        lease-driven warm's retry loop has to re-ask exactly this question on
        every pass — a second copy of the filter is how the trigger and its
        retry would come to disagree about what "a live visible lease" means.
        """
        now = time.monotonic()
        return [s for s in self.subscribers.values() if not s.overflow and s.expires > now]

    async def refresh_watch(self) -> None:
        """Recompute the aggregate watch lease, and warm for a VISIBLE one.

        Two jobs, because they are one policy: what the owner is told about
        presence, and — for a live VISIBLE lease on a viewer with no runtime
        yet — that a runtime is created for it, off the request path. The
        second is the change argued in the branch below; the first is the
        existing contract, and the record is written on EVERY beat rather than
        only on the way into a warm (see the comment at the write).

        Two things this beat does NOT do, both of which used to be wrong: it
        creates nothing for a lease that is not live and visible (unchanged),
        and it creates nothing for a session someone deliberately STOPPED — a
        stopped session stays stopped until a user action re-opens it
        (round-2 review MAJOR-1, argued at the gate below).
        """
        async with self.watch_lock:
            live = self._live_leases()
            remote = self.remote
            if remote is None:
                return
            visible = any(s.visible for s in live)
            can_notify = any(s.can_notify for s in live)
            # RECORDED ON EVERY BEAT, COLD OR NOT, and gating only the WARM on
            # `visible` is what makes the record correct rather than sloppy.
            # `_dial` re-asserts whatever was recorded last (TTL-bounded), so a
            # cold facade that skipped the write leaves the PREVIOUS pair
            # standing: hide the window during the ~1 s a spawn takes and the
            # runtime this warm creates counts a viewer who has gone, until the
            # next beat corrects it (≤15 s) or the runtime-side lease expires
            # (≤45 s, then the 3 s drain) — one idle runtime (~82 MB) held for
            # an absent viewer. Against that, the write is a field assignment
            # while cold (its RPC half is guarded by a connected client), and it
            # keeps `_desktop_seen` fresh as well as truthful.
            await remote.update_desktop_watch(visible=visible, can_notify=can_notify)
            if not visible:
                # NO LIVE VISIBLE LEASE, so the intent that earned any standing
                # pace is gone with it and the pace must not charge the next
                # one. Cleared HERE and not only in the loop, because the loop
                # that earned it has usually already returned by the time the
                # viewer leaves: a served attempt keeps its charge standing
                # (see `_lease_warm_loop`), and this beat — the first with no
                # live visible lease — is the moment a fresh intent can be told
                # apart from the last one's continuation (QA round 2, Q2).
                self._clear_warm_backoff()
                return
            if not remote.is_cold:
                return
            # A DELIBERATE STOP IS NOT A COLD VIEWER TO BE WARMED (review
            # round 2, MAJOR-1). `_recover_runtime` already refuses on this
            # fact, with the rationale "it is what keeps the takeover from
            # resurrecting a session a kill switch just ended"
            # (`session/attached.py`), and the desktop stop's own copy promises
            # the same thing — `/resume` reopens a stopped conversation, so
            # nothing else may. Without this guard the user stops a session in a
            # focused window, the runtime exits, and the next beat — within
            # 15 s, at ~82 MB idle — silently starts a fresh runtime for the
            # session they just ended, which also clears the `stopped_at`
            # marker the stop wrote.
            #
            # ITS LIMIT, stated so the guard is not read as complete: this is
            # "not proven stopped", not proof of life — the marker is this
            # facade's own flag, OR the durable `stopped_at` written only for a
            # session that HAS wakes (see `session_was_stopped`'s docstring). A
            # stop this facade issued is always caught; a stop from another
            # surface on a wake-less session is not. Closing that arm means
            # stamping the marker unconditionally in the stop path, which is a
            # change to the stop contract rather than to the warm.
            if await remote.session_was_stopped():
                self._clear_warm_backoff()
                return
            # A VISIBLE LEASED VIEWER CREATES RESIDENCY, it no longer only
            # preserves it, and that is the whole policy change here.
            #
            # Term 3 of `process._should_exit` already argues from "a user
            # looking at the session is about to type", and a desktop
            # viewer counts as one only while this lease is live AND the
            # window says visible (`server.py::attach_clients`). That
            # premise used to reach only a runtime that ALREADY existed, so
            # the first session-scoped action after one exited paid the
            # whole child spawn + handshake inline inside the user's click
            # (measured 1415 ms median against 119 ms warm). Warming on the
            # same lease closes that gap with the policy's own signal
            # rather than a new one: the viewer the reaper would have kept
            # alive now also causes one.
            #
            # PRESENCE IS RECORDED BEFORE THE WARM IS ARMED, and that is
            # correctness, not bookkeeping. `_ensure_bound`'s dial
            # re-asserts whatever `update_desktop_watch` last recorded, so
            # recording it here is what makes the runtime this warm is
            # about to create count the viewer from its FIRST tick.
            # `update_desktop_watch` needs no client to record it (the RPC
            # half is skipped while cold). Without it, the new runtime's
            # `attach_clients()` is 0 for the whole handshake, the 3 s idle
            # drain (`DEFAULT_GRACE_S`) runs against a viewer nothing ever
            # asserted, and the runtime exits moments after the bind
            # returns — whereupon the renderer's next 15 s heartbeat starts
            # another. A spawn/exit cycle per heartbeat is strictly worse
            # than the stall this removes, and the RAM it costs is the part
            # that actually shows up.
            #
            # BOUNDS, because "create a runtime for anyone watching" is a
            # residency change and unbounded residency is the failure mode.
            # The first two are the existing ones; the third is this trigger's
            # own, and the fourth is what keeps a failure from becoming a loop.
            #
            # * ONE RUNTIME PER SESSION, held by the EXISTING lock. Every
            #   attempt is `warm()` — the single engage path that is
            #   `_ensure_bound`, the only place a viewer creates a process — so
            #   a command arriving mid-warm and two attempts contending all
            #   serialise on `_bind_lock` and the loser returns at its own
            #   `is_cold` check. No second spawn path, and no per-subscriber
            #   multiplication: the bridge is per-session and one task per
            #   bridge is `warm()`'s own rule.
            # * THE LEASE IS THE LIFETIME, and the constant that decides it is
            #   the RUNTIME-side `DESKTOP_WATCH_LEASE_S`
            #   (`session/runtime/types.py`) — the same value as `WATCH_TTL`
            #   (45 s) by agreement today rather than by construction, read a
            #   third time by the dial (`attached.py::_dial`). `WATCH_TTL` here
            #   is the BRIDGE's subscription lease, which is what the renderer
            #   renews. Stop heartbeating — window closed, killed, navigated
            #   away — and the lease expires, the runtime falls out of term 3,
            #   and the existing drain reaps it exactly as it reaps one this
            #   change did not start; nothing here holds a process past the
            #   lease.
            # * THE AGGREGATE IS THE FOCUSED WINDOW, NOT ONE RUNTIME. One
            #   runtime per session says nothing about how many SESSIONS can be
            #   warm at once: `BRIDGE_COUNT` (64) sessions can hold a bridge
            #   while their event stream is open, and only eviction at
            #   `users == 0` bounds them. What makes "no cap" safe is that
            #   `visible` is `visibilityState === "visible" && hasFocus()` and
            #   the app mounts ONE lease-bearing chat view per focused window,
            #   so the warm is one runtime at a time — measured here at ~82 MB
            #   idle for the runtime, against the ~283 MB `process.py` budgets
            #   for one. A future that mounts several lease-bearing views at
            #   once (a split pane, a per-pane lease, a "watching" surface that
            #   asserts `visible` without focus) multiplies that by the number
            #   of panes and NEEDS a real cap; the cap today is this gate, and
            #   it is a consequence of the UI rather than of this code.
            # * AN ATTEMPT THAT FAILED IS PACED. The retry loop (see
            #   `_lease_warm_loop`) keeps a live lease's intent across attempts,
            #   but a bind that could not start — the failing-spawn shape — waits
            #   out a doubling backoff rather than re-engaging on every beat,
            #   and an attempt that was refused before doing any work is retried
            #   at the poll pace instead. Both numbers are argued on their
            #   constants.
            # * NOTHING ELSE CREATES ANYTHING. A hidden or notify-only
            #   viewer is the `not visible` return above: `can_notify` is
            #   delivery reachability, not attention, and a window nobody
            #   is looking at is not about to type. Deliberately NARROWER
            #   than term 3, which still counts `visible or can_notify` to
            #   preserve an existing runtime — that policy is untouched.
            #
            # OFF THE REQUEST PATH: `_arm_lease_warm` schedules a task and
            # returns, so `/watch` answers in the ~10 ms it always did and the
            # heartbeat never pays the engage it triggers.
            self._arm_lease_warm(remote)

    def _arm_lease_warm(self, remote: AttachedSession) -> None:
        """Start the lease-driven warm's retry loop, unless it is already running.

        Called by :meth:`refresh_watch` while it holds ``watch_lock``. The
        ``create_task`` has no ``await`` in front of it, which is what keeps the
        heartbeat off the engage it triggers; the loop's first step engages.

        ONE TASK PER BRIDGE, AND IT OUTLIVES THE BEAT THAT ARMED IT: the intent
        it holds is the LEASE's, not the request's, so a heartbeat arriving
        while an attempt is in flight — or inside a failure's backoff — must
        leave the running task alone rather than start a second one. The
        freshness of the lease is not sampled here for the same reason: the loop
        re-asks it every pass, which is what lets a withdrawn lease end the
        retries instead of being noticed only at the next beat.
        """
        if self.lease_warm_task is not None and not self.lease_warm_task.done():
            return
        self.lease_warm_task = asyncio.create_task(self._lease_warm_loop(remote))

    def _clear_warm_backoff(self) -> None:
        """Forget the pace of an intent that is over, so the next one is fresh.

        Called when the intent is ABANDONED rather than served: the lease that
        held it lapsed or withdrew, the facade was replaced, `_detach` dropped
        the bridge, or a deliberate stop ended it. Cleared rather than left to
        expire so the NEXT intent — a fresh cold period, on the same or a new
        runtime — starts from the base backoff instead of from whatever the last
        one had grown to (QA round 2, Q2: a viewer returning 12 s into a 30 s
        pace paid the remaining 15.9 s before its first child appeared).

        DELIBERATELY NOT called when an attempt left the viewer bound. That
        charge is what bounds a runtime that boots and then dies — the
        crash-loop of review round 2 MINOR-1 — and dropping it on the way out
        would re-spawn one per heartbeat, which is the unattended loop this
        pacing exists to prevent. A beat with no live visible lease clears it
        instead (`refresh_watch`), so a viewer who genuinely leaves is not
        charged for a runtime that died while it was still looking.
        """
        self.warm_backoff_s = 0.0
        self.warm_not_before = 0.0

    async def _lease_warm_loop(self, remote: AttachedSession) -> None:
        """Keep a live VISIBLE lease's warm until it is served or withdrawn.

        WHY A LOOP RATHER THAN ONE ATTEMPT PER BEAT, which is how this started.
        Two ways one attempt is silently lost, both ending in the cold bind the
        warm exists to remove:

        * **The facade cannot engage yet.** `_ensure_bound` returns at its own
          ``_recovering`` guard — no error, and no task to report it — and a
          viewer that has just lost its runtime sits there for up to
          ``COLD_FALLBACK_S`` (8 s; ~9.4 s measured end to end on this path,
          because the loop pays dial pacing on the way out). An attempt landing
          in that window does NOTHING, and the renderer's next beat is 15 s
          away while the user usually clicks first. What marks this as not a
          failure is exactly that it did no work.
        * **An engage is already in flight** (`engage_in_flight`) — the lock
          held by another subscriber's `attach_existing`, say — so `warm()`
          starts no task and, again, nothing retries for a whole beat.

        AN ATTEMPT THAT ACTUALLY RAN is the third case and is handled
        differently, because it DID work: it spawned. Retrying that on the beat
        is the spawn loop a heartbeat alone could drive, so it pays a doubling
        backoff (base and ceiling argued on the constants) and the failure is
        logged, since a bind that keeps failing has no other surface on this
        path. The charge follows the ATTEMPT, not its outcome at the post-await
        check: a runtime that comes up and then dies passes that check and would
        otherwise be re-spawned by the next beat, one failure shape over from
        the case this pacing was built for (review round 2, MINOR-1).

        NOT A SECOND SPAWN PATH: every attempt is `warm()`, the ordinary
        background engage through `_ensure_bound` and the one `_bind_lock`. This
        loop decides only WHEN to ask, never how.

        BOUNDED BY THE LEASE, WHICH IS THE POINT. It exits the moment the viewer
        is bound, the facade is replaced, no live visible lease remains, or the
        session has been deliberately stopped — and it re-asks all four on every
        pass rather than trusting the state at arm time, which is also why a
        backoff is waited out in slices rather than in one long sleep: a lease
        withdrawn during the wait ends the retries within one slice instead of
        after up to two minutes of them.

        EVERY EXIT THAT IS NOT "the viewer is bound" IS AN ABANDONED INTENT and
        drops the pace with it; the bound exit keeps its charge, for the
        crash-loop reason above.
        """
        while True:
            if self.remote is not remote:
                # A replacement viewer owns the bridge now: the intent (and its
                # pace) belonged to the facade that just left.
                self._clear_warm_backoff()
                return
            if not remote.is_cold:
                self._clear_warm_backoff()
                return
            async with self.watch_lock:
                visible = any(s.visible for s in self._live_leases())
            if not visible:
                # ABANDONED, NOT SERVED: the lease that expressed this intent
                # has lapsed or withdrawn, so the intent ends and its pace goes
                # with it — see `_clear_warm_backoff`.
                self._clear_warm_backoff()
                return
            if await remote.session_was_stopped():
                # The same question `refresh_watch` asks before it arms, where
                # the rationale and the marker's limit are written (review
                # round 2, MAJOR-1). Re-asked here because a stop can land while
                # the loop is still pacing, and a loop armed before the stop
                # must not be the thing that resurrects the stopped session.
                self._clear_warm_backoff()
                return
            remaining = self.warm_not_before - time.monotonic()
            if remaining > 0:
                await asyncio.sleep(min(remaining, _LEASE_WARM_POLL_S))
                continue
            if remote.recovering or remote.engage_in_flight:
                # REFUSED, NOT FAILED. No task was created (or one is already
                # running that an attempt would only join) and no process was
                # started, so this does not spend the failure backoff: the poll
                # is what carries the intent across a recovery window.
                await asyncio.sleep(_LEASE_WARM_POLL_S)
                continue
            try:
                await self.warm()
            except DaemonRetiring:
                # The daemon announced its retirement while this lease was live.
                # A warm is not admissible any more and the refusal is one-way,
                # so the intent ENDS here: looping would spin a spawn attempt a
                # beat against a refusal that cannot resolve, and the retry
                # exists to serve a viewer, not to keep a dying daemon busy.
                self._clear_warm_backoff()
                return
            task = self.warm_task
            if task is not None:
                # Cancellation is deliberately NOT suppressed: `_detach` cancels
                # this loop and the engage under it, and a cancelled engage must
                # end this loop rather than be read as a settled failure.
                with contextlib.suppress(Exception):
                    await task
            # THE ATTEMPT IS CHARGED, NOT ITS OUTCOME (review round 2, MINOR-1).
            # Charging only when the post-await check finds the viewer cold let a
            # runtime that boots and then dies — a late boot failure, an OOM, a
            # build-stamp restart gone wrong — be re-spawned on every beat, with
            # `warm_backoff_s` still 0.0 (reproduced: 4 beats -> 4 attempts).
            # The pace prices the spawn that was actually made, whether or not
            # the check below happens to catch the shape. A runtime that STAYS
            # up is unaffected: this loop returns and no other arms until a beat
            # finds the viewer cold with a live visible lease, and the charge is
            # dropped by the first beat with no live visible lease
            # (`refresh_watch`), so a window that leaves and returns starts from
            # the base.
            self.warm_backoff_s = min(
                self.warm_backoff_s * 2 if self.warm_backoff_s else _LEASE_WARM_BACKOFF_S,
                _LEASE_WARM_BACKOFF_CAP_S,
            )
            self.warm_not_before = time.monotonic() + self.warm_backoff_s
            if not remote.is_cold:
                # SERVED: the runtime is up. The charge stands (above) so a
                # runtime that dies in the next few seconds is paced rather than
                # re-spawned at the next beat.
                return
            logger.debug(
                "lease-driven warm for %s left the viewer cold; next attempt in %.0fs",
                self.session_id,
                self.warm_backoff_s,
            )

    def assert_admitting(self) -> None:
        """Raise ``DaemonRetiring`` when the daemon serving this bridge has LATCHED.

        THE SAME QUESTION the pool's door asks (``DesktopSessions.session``, which
        every route comes through), kept as a bridge method for the callers that
        never arrive through a route: ``warm``'s own speculation, the lease-warm
        loop, and anything else this process starts on its own behalf. Those may
        not delegate to a route's refusals, so the question has to be askable
        here.

        The per-route history is why the DOOR, not this method, is now the
        enforcement: this check reached the three routes that called it
        (``/messages``, ``/commands``, ``/answers``) and the routes that did not
        (``/mcp``, ``/credentials``, ``/fork``, ``/asides``, ``/adopt``) reached
        ``bind_runtime()`` on a latched daemon instead (review round 2, MAJOR-1).
        The bridge-level call is a second pair of eyes, not the mechanism.

        Delegates the QUESTION to the pool's probe rather than reading a flag of
        its own, for the reason the probe exists: the answer changes once, mid-
        life, in the daemon rather than in any bridge.
        """
        if self.retiring_probe():
            raise DaemonRetiring(RETIRING_MESSAGE)

    async def warm(self) -> str:
        """Start a runtime for this session without submitting any work.

        REFUSED WHILE LATCHED, before anything else: this is the one path that
        SPAWNS a session runtime from this process (``warm_runtime`` below), and
        a daemon that is about to exit must not start a runtime whose viewer
        would follow it onto a dead address. The refusal is typed so the route
        answers a named 503 rather than a 500 (see ``DaemonRetiring``), and it
        is checked before the ``remote is None``/cold branches so a latched
        daemon refuses uniformly rather than only when it happens to be cold.
        The decision itself lives in :meth:`assert_admitting`, which this bridge
        shares with the pool's door (``DesktopSessions.session``) — the routes
        reach that door, this loop reaches this method, and both read one probe.

        Returns the state at RETURN TIME — ``"warm"``, ``"warming"`` — never the
        eventual outcome, because every caller fires this speculatively (a
        keystroke, or a live visible watch lease) and has nothing to do with an
        answer either way.

        FIRE AND FORGET, DELIBERATELY. The engage runs in a detached task so the
        HTTP response returns in the ~12-40 ms a warm send costs while the spawn
        proceeds behind it. Awaiting the engage here would not remove the
        ~1.15 s cold cost, it would only move it from the send to the warm — and
        onto a request the renderer issues while the user is still typing.

        IDEMPOTENT, AND ITS SAFETY IS THE LOCK'S, NOT THIS CHECK'S. Both early
        returns are cost avoidance: an already-bound viewer needs no task, and
        an engage already in flight needs no second one. If two warms raced past
        ``engage_in_flight`` anyway, ``_ensure_bound``'s ``_bind_lock``
        serialises them and the loser returns at its own ``is_cold`` check, so
        two warms can never spawn two runtimes. Do not "strengthen" this into a
        lock of its own: a second lock beside the one that already decides the
        question is how the two answers drift apart.

        THE ENGAGE LIVES ONLY AS LONG AS THE BRIDGE, which constrains the
        CALLER and is not visible from this method alone. A bridge is
        reference-counted; ``_detach()`` cancels the warm below so a spawn
        cannot outlive the facade it was started against. The warm request is
        itself a user of that bridge, so a warm issued while nobody else holds
        one is cancelled the instant its own request releases — correct, and
        also useless. It is not a problem for the real caller because the
        renderer warms from a composer inside a mounted session panel, which
        holds an events subscription for its whole life. A future change that
        moves the warm outside that panel, or a probe that warms with no
        subscription open, gets a warm that does nothing and a send that still
        pays the full cold engage.

        TWO CALLERS, ONE SPAWN PATH, AND ONE RETRY RULE. The renderer asks for
        it explicitly (``POST /warm``, first keystroke — a user action, so this
        path is never paced) and the lease-driven retry loop
        (:meth:`_lease_warm_loop`, armed by :meth:`refresh_watch`) asks on
        behalf of a live VISIBLE watch lease, so a session being looked at is
        warm before the first click rather than after it. Both go through the
        guards below and through ``_ensure_bound``; a third one is how two
        answers to "is a runtime needed" would drift apart. The ``done()``
        clause below is what lets the loop tell a settled attempt from a live
        one, and it is deliberately NOT where the pacing lives: a retry that
        no user is behind belongs to the loop, which owns the backoff.

        THERE IS NO ``retire_if_unused`` COUNTERPART HERE, and its absence is a
        decision rather than an oversight. The TUI offers its runtime back
        because the TUI QUITS and must hand over before its socket dies. The
        desktop app does not: a desktop attach only counts as an interactive
        viewer while its watch lease is live AND the window says visible or
        notifiable, so a warmed session the user navigates away from stops
        counting and the runtime's own residency drain reaps it seconds later.
        Calling ``retire_if_unused`` here would be a second mechanism beside a
        working one, and a strictly worse one — it answers "no runtime attached"
        whenever the client is None, which is precisely the state a bridge in
        the middle of a warm is in.
        """
        self.assert_admitting()
        remote = self.remote
        assert remote is not None
        if not remote.is_cold:
            return "warm"
        self._schedule_warm()
        return "warming"

    def _schedule_warm(self) -> None:
        """Start this session's speculative engage, unless one is already owed.

        THE BODY OF :meth:`warm`, factored out for its SECOND caller: the retire
        frame (:meth:`_on_runtime_retired`). Both callers need exactly the same
        guards and the same one-task discipline — a copy of them beside
        ``warm()`` is how two answers to "does this session need a runtime"
        drift apart. A method of its own rather than a flag on ``warm()``
        because a retire has no response to compose and no state to report: its
        caller is a frame handler, not a route.

        Returns nothing, deliberately. What an engage becomes is not knowable
        here (that is `warm()`'s own docstring), and the retire path has nobody
        to tell either way.

        IT ASKS :meth:`assert_admitting` ITSELF rather than relying on its
        callers, because one of them is a frame handler with no refusal to
        compose: a daemon latched for retirement must not start a session
        runtime from EITHER caller (``warm``'s own docstring states why). The
        ``DaemonRetiring`` that raises is each caller's to handle — the route
        answers its named 503, the lease loop ends its intent, and
        :meth:`_on_runtime_retired` declines, because a daemon being replaced
        owes this viewer no successor.
        """
        self.assert_admitting()
        remote = self.remote
        # ``None`` only while detached. `_detach()` clears the facade and cancels
        # any task this could have started, so a frame arriving after it must not
        # re-arm a spawn with no viewer left to release it — the leak the
        # ``warm_task`` field is spent to prevent.
        if remote is None or not remote.is_cold:
            return
        # TWO conditions, because they answer different questions and the
        # second is not implied by the first. `engage_in_flight` samples the
        # facade's bind lock; this one asks whether THIS BRIDGE already owns a
        # live warm task. A second warm arriving while the first task exists
        # but has not yet taken the lock -- a second HTTP request resumed out
        # of `acquire()` ahead of the first task's first step, which two tabs
        # make ordinary -- passes the predicate, and overwriting `warm_task`
        # would orphan the first: it escapes `_detach()`'s cancel and is left
        # to the weak-reference hazard the field's own comment names. Keeping
        # exactly one referenced task is the point; `done()` lets a settled
        # warm be retried, which matters because a failed engage leaves the
        # viewer cold and the next keystroke should be free to try again.
        if remote.engage_in_flight or (self.warm_task is not None and not self.warm_task.done()):
            return
        self.warm_task = asyncio.create_task(remote.warm_runtime())

    def _on_runtime_retired(self) -> None:
        """The runtime retired itself; engage its successor now.

        Reached from ``AttachedSession._go_cold(refresh=True)``, i.e. from the
        ``retiring`` frame, which the runtime sends with THE SAME MEANING for a
        move and for a client-side build refresh: "a successor is owed; engage
        one". The TUI has answered it since the build-refresh work landed
        (``_on_runtime_refreshed``, installed at the same seam); the desktop had
        no callback installed at all, so a retired runtime left the viewer cold
        and the chip showing the OLD directory until the user's next send
        happened to engage — the gap this feature's own move exposed.

        A MOVE IS THE CASE THIS EXISTS FOR, AND IT IS THE ONE NOBODY ELSE
        COVERS. A whole-daemon retirement (``server/retire.py``, a build update)
        has its own mechanism and this callback deliberately declines during it:
        ``_schedule_warm`` asks ``assert_admitting`` and the daemon being
        replaced needs no successor spawned inside it. A session runtime retired
        by the VIEWER — which is exactly what a move is — has nothing else: the
        daemon is healthy, nobody is going to replace it, and the successor is
        owed by this frame alone.

        EAGER rather than lazy, for the reason the TUI is: the next prompt would
        engage anyway (``_ensure_bound``), but nothing on the desktop surface
        repaints the chip in the meantime, and the successor's own bind is what
        publishes the new ``frontend.cwd``. Deferring the engage defers the one
        event that settles the chip.

        Runs ON THE EVENT LOOP, which is what lets it create a task directly:
        it is called by the attach client's pump (`_on_disconnected` → this),
        an async method on the loop thread. Cancellation stays correct because
        the task it may create is the bridge's own ``warm_task``, which
        ``_detach()`` cancels.
        """
        try:
            self._schedule_warm()
        except DaemonRetiring:
            # The DAEMON is going, not just this runtime: it latched for
            # retirement and a replacement is on its way. Engaging here would
            # spawn a runtime whose viewer follows it onto a dead address, and
            # the app reconnects to whatever replaces the daemon instead.
            logger.debug("not re-engaging %s; the daemon is retiring", self.session_id)

    async def _expire_watches(self) -> None:
        while True:
            remaining = [
                s.expires
                for s in self.subscribers.values()
                if not s.overflow and s.expires > time.monotonic()
            ]
            if not remaining:
                # LAST lease has expired. Returning here without a final refresh
                # left the runtime holding whatever presence the previous pass
                # asserted -- visible, notifiable -- for the rest of the
                # session, because nothing else recomputes it once the loop is
                # gone. The expiry that ends the loop is exactly the one the
                # runtime still needs to be told about.
                with contextlib.suppress(ConnectionError, RuntimeError):
                    await self.refresh_watch()
                return
            await asyncio.sleep(max(0, min(remaining) - time.monotonic()))
            with contextlib.suppress(ConnectionError, RuntimeError):
                await self.refresh_watch()

    def subscribe(self) -> DesktopSubscription:
        if len(self.subscribers) >= SUBSCRIBER_COUNT:
            raise ValueError("Too many event subscribers")
        sub = DesktopSubscription()
        self.subscribers[sub.id] = sub
        return sub

    async def events(
        self, sub: DesktopSubscription, *, epoch: str | None, after_seq: int
    ) -> AsyncGenerator[dict[str, Any], None]:
        try:
            cutoff = self.sequence
            first = self.replay[0][0]["seq"] if self.replay else cutoff + 1
            gap = epoch != self.epoch or after_seq < first - 1 or after_seq > cutoff
            replay = (
                [f for f, _ in self.replay if after_seq < f["seq"] <= cutoff] if not gap else []
            )
            snapshot = await self.snapshot()
            yield {
                "session_id": self.session_id,
                "epoch": self.epoch,
                "seq": cutoff,
                "type": "open",
                "payload": {
                    "subscription_id": sub.id,
                    "gap": gap,
                    "watch_ttl_seconds": WATCH_TTL,
                },
            }
            # Replay receipts BEFORE the authoritative snapshot so cumulative
            # record updates cannot repaint newer snapshot text with old deltas.
            # The open frame is metadata, NOT permission to skip this replay.
            for frame in replay:
                yield frame
            yield snapshot
            while True:
                try:
                    item = await asyncio.wait_for(sub.queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield {"type": "heartbeat", "session_id": self.session_id}
                    continue
                if item is None:
                    yield {"type": "gap", "session_id": self.session_id}
                    return
                frame, size = item
                sub.queued_bytes -= size
                if frame["seq"] > cutoff:
                    yield frame
        finally:
            self.subscribers.pop(sub.id, None)
            # ASGI disconnect runs inside a cancelled anyio scope. Cleanup must
            # still reach the runtime; otherwise a dead renderer leaves presence
            # asserted until TTL expiry and the bridge never releases its socket.
            with CancelScope(shield=True), contextlib.suppress(ConnectionError, RuntimeError):
                await self.refresh_watch()


class SubagentChildUnavailable(Exception):
    """A child-route URL does not name a readable child of that conversation.

    ONE refusal for every containment failure — a malformed id, a parent that
    is not the user's session, a child the parent never launched, a directory
    whose origin marker is not ``subagent``, a record pointing outside
    ``sessions/``. The caller learns that this pair is unreadable and nothing
    else, which is the point: separate refusals would let an authenticated
    renderer walk ids and use the difference between "no such session" and "not
    a child of this one" to enumerate the machine's session store.

    ``child_not_found`` is the code design § 9.1 fixes, and it is RETRYABLE: a
    runtime snapshots its roster asynchronously, so a child launched inside the
    last write window is briefly missing from the persisted record. A reader
    that re-probes on its next pulse (the sidebar's 1 Hz child read) resolves
    that case without any special handling, which is why the route does not
    distinguish it from a permanent refusal.
    """

    code = "child_not_found"

    def __init__(self) -> None:
        super().__init__("That subagent does not belong to this conversation.")


def _persisted_children(parent_dir: Path) -> list[Any]:
    """The records of the children a conversation launched, from its own store.

    The roster is the ownership record for a child read (design § 9.1), and it
    is read the way every other reader reads it rather than through a query of
    this module's invention:

    * the roster SIDECAR, replaced atomically by
      ``Session._persist_subagent_roster`` on every roster move — the store a
      current runtime writes; then
    * the legacy ``subagent_roster`` transcript custom entry, written once by
      builds that predate the sidecar, so a conversation last run by one of
      those still opens its children — **accepted only when it was appended
      after this session's fork boundary**.

    That is exactly ``Session._load_subagent_roster``'s order AND its fork
    guard, deliberately not re-derived here: two readers of one ownership
    record that disagree is the defect ``session/restored_rows.py`` exists to
    prevent, and the fork case is where this reader used to be the one that
    disagreed (review round 1, R1-1). ``fork_session`` CLONES the parent's
    transcript — so the parent's entry is present VERBATIM in the fork — while
    ``fork.EXCLUDED_SIDECARS`` leaves the sidecar behind. Without the guard a
    fork inherits the original's children as its own and can read them through
    its own route; with it, only a fork's OWN roster (written after
    ``forked_at``) counts. Re-stamping the sidecar happens on the fork's first
    roster move, so the fallback is what a fresh fork rides until then.

    The design's ``subagent_roster`` custom-entry wording predates the v0.40.0
    sidecar; the sidecar holds the same ``records`` list and is newer, so it is
    consulted first and the entry is the fallback, never the other way round.

    Runs off the event loop like every other reader here: the legacy fallback
    constructs a ``Transcript``, which parses the parent's whole journal, and a
    roster read must not block the loop a streaming turn is using.
    """
    from local_operator.fork import fork_instant
    from local_operator.session.session import (
        SUBAGENT_ROSTER_CUSTOM_TYPE,
        SUBAGENT_ROSTER_SIDECAR,
        _read_roster_sidecar,
    )

    payload = _read_roster_sidecar(parent_dir / SUBAGENT_ROSTER_SIDECAR)
    if payload is None:
        # The entry's TIMESTAMP is the half of the fork rule the sidecar makes
        # unnecessary: an entry at or before ``forked_at`` belongs to the
        # conversation this one was cloned from.
        entry = Transcript(parent_dir).latest_custom_entry(SUBAGENT_ROSTER_CUSTOM_TYPE)
        forked_at = fork_instant(parent_dir)
        if entry is None or (forked_at is not None and not entry.ts > forked_at):
            return []
        payload = dict(entry.payload.get("details", {}))
    return list(roster_records(payload))


def _contained_child_dir(root: Path, session_id: str, child_id: str) -> Path:
    """The directory of ``child_id`` as a child of ``session_id``, or refuse.

    This is the whole containment proof of the child read route (design § 9.1),
    and every clause carries weight. It is a new READ PATH ACROSS A TRUST
    BOUNDARY — the renderer holds an absolute ``session_dir`` on the wire and
    must never be able to ask for one — so the route proves membership here, in
    the server, and the caller asserts nothing:

    * **Both ids are ids, never paths.** The same 12-hex-character shape the
      whole desktop surface validates a session on. Checked before any path is
      built, so a crafted value cannot reach ``sessions/`` through ``..`` (a
      directory name is not a legal id).
    * **The parent is the user's own conversation.** A child route scoped to a
      subagent or a fork would make the graph reachable from either end; only a
      conversation the user opened may name its children (``is_user_session``).
    * **The parent's persisted roster names this child.** Membership is not
      inferred from the directory layout, because every session directory looks
      alike on disk; the parent has to have recorded the launch. The record's
      ``session_dir`` must point at exactly ``sessions/<child_id>`` with the
      resolved sessions root as its parent, so a hand-edited record cannot
      redirect a read outside the store.
    * **The target resolves inside the store.** The id cannot be a path, but the
      DIRECTORY it names can be a link, and the store is writable by anything
      running as the user — so a symlinked ``sessions/<12-hex>`` would take the
      read (and the checks below) out of the store while every clause above
      still passed (review round 1, R1-2). The path is resolved and the
      resolved path is what the rest of this function — and the caller — then
      treats as the child, so the thing checked is the thing read. That is the
      same gate ``session/cleanup.py`` applies before it removes a directory
      and the legacy chat route applies before it opens a file.
    * **A child that is still on disk is marked a subagent.** Reversing the
      parent's rule: a user conversation or a fork must be unreadable through
      this route even if it somehow appears in a roster. A directory that is
      ABSENT is not refused here — the caller answers that with the derived
      ``gone`` state, which is the one case where the absence itself is the
      answer. A path that EXISTS and is not a directory is not a child session
      at all, so it is refused too rather than reported as ``gone``: it claims
      a deletion that never happened (review round 1, R1-5).

    Deliberately does NOT acquire a desktop bridge: answering must never start,
    attach to or wake a runtime. The live comms graph may *confirm* membership
    when the runtime happens to be attached, but nothing here reads it — a
    route that a paused session answers identically is a route with no second
    behaviour to test.
    """
    sessions = root / "sessions"
    if not SESSION_ID.fullmatch(session_id) or not SESSION_ID.fullmatch(child_id):
        raise SubagentChildUnavailable()
    parent_dir = sessions / session_id
    if not parent_dir.is_dir() or not is_user_session(parent_dir):
        raise SubagentChildUnavailable()
    resolved_root = sessions.resolve()
    try:
        child_dir = (sessions / child_id).resolve()
    except (OSError, RuntimeError):
        # A path that cannot be resolved (a symlink loop) is not a child.
        raise SubagentChildUnavailable() from None
    if not child_dir.is_relative_to(resolved_root):
        raise SubagentChildUnavailable()
    named = False
    for record in _persisted_children(parent_dir):
        raw_dir = record_field(record, "session_dir")
        if not raw_dir:
            continue
        candidate = Path(str(raw_dir).rstrip("/"))
        if candidate.name == child_id and candidate.parent.resolve() == resolved_root:
            named = True
            break
    if not named:
        raise SubagentChildUnavailable()
    if child_dir.exists() and (
        not child_dir.is_dir() or session_origin(child_dir) != ORIGIN_SUBAGENT
    ):
        raise SubagentChildUnavailable()
    return child_dir


def _absent_child_page(state: str, *, before_id: str | None = None) -> dict[str, Any]:
    """The envelope for a child with no readable rows: ``pending`` or ``gone``.

    ``cursor_missing`` mirrors ``DesktopSessionBridge.history``'s
    ``FileNotFoundError`` branch instead of being hardcoded ``False``: a caller
    that paged backwards from a cursor into a transcript that is no longer
    there is in the same position as one whose cursor a compaction replaced,
    and the envelope's documented answer to both is "re-read the tail and
    dedupe by id".
    """
    return {
        "entries": [],
        "has_more": False,
        "cursor_missing": bool(before_id),
        "state": state,
    }


class DesktopSessions:
    """Bounded adapter cache; canonical identity lives in the session directory."""

    def __init__(self, root: Path, *, retiring: Callable[[], bool] | None = None) -> None:
        self.root = root
        self.bridges: dict[str, DesktopSessionBridge] = {}
        self.lock = asyncio.Lock()
        # Whether the DAEMON this pool serves has LATCHED against new work, asked
        # rather than cached: the answer changes once, mid-life, and both the
        # refusal (``assert_admitting``) and every bridge this pool hands out
        # must see it. Defaulted so the many reduced ``DesktopSessions(root)``
        # constructions (tests, embedded apps) behave exactly as before.
        self.retiring_probe = retiring or _never_retiring

    def assert_admitting(self) -> None:
        """Raise ``DaemonRetiring`` when this daemon has LATCHED against new work.

        THE ADMISSION PATH for everything session-scoped, and the only one: this
        pool's ``session()`` — the door EVERY desktop route obtains its bridge
        through, so the whole plane is covered by construction (review round 2,
        MAJOR-1) — and ``create`` (a new session, which needs no bridge) ask this
        question, so "what does a retiring daemon refuse" has exactly one answer.
        ``DesktopSessionBridge.assert_admitting`` is the same question for the
        callers that never come through a route (``warm``'s lease loop).

        NOT raised while the daemon is merely ANNOUNCED: the announcement's only
        job is to tell an attached client to let go, and a daemon that refused
        work the instant it announced would be unusable for however long that
        client took to react. The latch follows the empty drain
        (``server/retire.py``).

        ONCE LATCHED IT REFUSES READS AS WELL, because a session-scoped request
        handed to a process that has told its readers to leave can only be served
        by the build it is leaving; the record plane (``GET /v1/desktop/sessions``,
        ``GET /health``, the record file) is a different surface and keeps
        answering until the clean exit removes it, which is what lets a reader
        observe the handover.
        """
        if self.retiring_probe():
            raise DaemonRetiring(RETIRING_MESSAGE)

    def in_flight_reason(self) -> str | None:
        """Why the DESKTOP plane is still using this daemon, or ``None``.

        Multiple terms, all things an exit would CUT rather than pause, all read
        off the live bridges:

        * **An in-flight HTTP operation** — any bridge with ``users > 0``. Every
          desktop route runs inside ``session()``, which brackets the request
          with ``acquire()``/``release()`` (``DesktopSessionBridge.acquire``),
          so a non-zero count means a client is holding this bridge open RIGHT
          NOW. For a request that is building a response that is a few tens of
          milliseconds; for the app's event stream it is the whole life of the
          view (see below), and this docstring used to claim the first while
          meaning only it.
        * **A STANDING attachment** — the same ``users`` term, held by
          ``GET /v1/desktop/sessions/{id}/events``, which acquires the bridge
          before it returns response headers and releases it only when the
          stream tears down: an unbounded, replayable relay with no turn
          boundary and no TTL. This is the term that decides the SHAPE of the
          daemon's retirement — the announcement has to precede the drain,
          because this stream is only released when the client decides to
          (``server/retire.py``'s module docstring).
        * **An open attach with a LIVE watch lease** — a window is looking at
          this session (``DesktopSessionBridge._live_leases``, renewed by
          ``watch`` and expiring after ``WATCH_TTL``, which the app renews every
          15 s): the operator's "a viewer is never pulled out from under" rule
          applied to the process that serves it. Counts whether or not the
          window is visible or focused.
        * **A runtime being started** — a bridge with a warm task still
          running (``DesktopSessionBridge.warm_task``, set by ``warm`` and by
          the lease-driven retry loop). A spawn is a handshake with a child
          process that takes ~1.2 s; exiting in the middle of one leaves the
          engage unfinished against a successor that has no idea it was
          running. Bounded by the engage's own attempt budget, and not covered
          by ``users`` — ``warm`` is fire-and-forget and answers the request
          before the handshake completes.

        Read WITHOUT ``watch_lock`` on purpose, and that is safe: this is a
        synchronous filter over an in-process dict on the event loop, so it
        cannot interleave with a mutation. The lock ``_live_leases``'s other
        callers take guards the ACTIONS they then take on the result, not the
        read itself — and taking it here would let a busy warm hold up the
        retirement poll.
        """
        for bridge in list(self.bridges.values()):
            if bridge.users:
                return f"{bridge.users} in-flight desktop request(s) on {bridge.session_id}"
            if bridge._live_leases():
                return f"a desktop window watching session {bridge.session_id}"
            if bridge.warm_task is not None and not bridge.warm_task.done():
                return f"a runtime being started for session {bridge.session_id}"
        return None

    async def acknowledge_attention(self, session_id: str, token: str) -> dict[str, Any]:
        """A read receipt never admits work, binds a viewer, or starts a runtime.

        Validate the same durable user-session namespace as the bridge, but do
        not enter its acquire path: a completed cold conversation is readable
        even when its runtime and the mobile daemon are both stopped.
        """

        def acknowledge() -> dict[str, Any]:
            if not SESSION_ID.fullmatch(session_id):
                raise KeyError("Unknown session")
            path = self.root / "sessions" / session_id
            if not path.is_dir() or not is_user_session(path):
                raise KeyError("Unknown session")
            return AttentionStore(self.root / "attention.db").acknowledge(
                f"session/{session_id}", token
            )

        return await asyncio.to_thread(acknowledge)

    async def claim_notification(self, session_id: str, token: str) -> bool:
        """Claim the right to TOAST ``token``; exactly one surface ever wins.

        NOTIFYING IS NOT READING, and this is the boundary that keeps the two
        watermarks apart. ``claim_delivery`` writes ``deliveries`` only: the
        sidebar's unseen mark and ``receipts.acknowledged`` are untouched, so a
        session the user was merely *told about* stays unread until they
        actually open it. Routing this through :meth:`acknowledge_attention`
        instead would clear the mark for a conversation nobody looked at, which
        is the one thing ``docs/ATTENTION.md`` forbids outright.

        Cold path, exactly like :meth:`acknowledge_attention`: same session-id
        validation, no bridge acquire, no runtime spawn. A completion worth
        announcing is usually one whose owner has already exited, and a banner
        is never a reason to start a process.

        ``backend="desktop"`` names the claimant. The column is diagnostics
        only — no decision may read it, because a claim that consulted anything
        beyond the monotonic sequence would stop being clock-free — but naming
        it correctly is what makes a store dump readable when two surfaces
        disagree about who toasted.

        Returns ``False`` for an unknown or foreign token rather than raising:
        the caller's next step is "show or do not show a banner", and a
        surface that cannot claim simply stays quiet.
        """

        def claim() -> bool:
            if not SESSION_ID.fullmatch(session_id):
                raise KeyError("Unknown session")
            path = self.root / "sessions" / session_id
            if not path.is_dir() or not is_user_session(path):
                raise KeyError("Unknown session")
            return AttentionStore(self.root / "attention.db").claim_delivery(
                f"session/{session_id}", token, "desktop"
            )

        return await asyncio.to_thread(claim)

    async def attachment(self, session_id: str, digest: str) -> tuple[bytes, str]:
        """Decoded bytes and mime type for one content-addressed attachment.

        Durable transcript rows reference images by digest, not by payload:
        ``transcript._externalize_attachments`` strips ``data`` from any block
        over 1 KiB of base64 and leaves ``{"attachment": <digest>,
        "mime_type": ...}`` behind. ``/history`` serves those rows verbatim,
        so a reading surface can see that an image WAS there and has no way to
        fetch it. This is that way.

        Deliberately outside :meth:`session`, exactly like
        :meth:`acknowledge_attention` and for the same reason: reading a
        screenshot out of a finished conversation must not start a runtime
        process. The session id is still validated against the same durable
        user-session namespace, so the route cannot be used to probe arbitrary
        directories, and the store is shared rather than per-session because
        the digest IS the content key.

        ``KeyError`` for an unknown session or an unresolvable digest — the
        store's own contract is that a miss is ordinary (an interrupted write,
        a hand-pruned store) and callers degrade to a placeholder rather than
        treating it as a fault.

        The session id is an EXISTENCE check, not a binding: it proves *a* user
        conversation by that name is on this machine, never that this digest
        belongs to it. The store is content-addressed and shared across
        conversations by design, so any valid user session id resolves any
        digest in it. The bearer already authorises the whole desktop surface,
        so this is not an escalation — but it is not per-session scoping
        either, and the URL shape reads as though it were.
        """

        def read() -> tuple[bytes, str]:
            # Both halves of this gate carry weight and neither is redundant.
            # The shape check keeps a crafted id from escaping the sessions
            # namespace through ``..`` before a path is ever built; the origin
            # check keeps this route out of SUBAGENT conversations, which are a
            # machine's delegated runs the user never opened and which the
            # desktop surface does not list. Dropping either is a one-token
            # edit, so each has a named test standing on it.
            if not SESSION_ID.fullmatch(session_id):
                raise KeyError("Unknown session")
            path = self.root / "sessions" / session_id
            if not path.is_dir() or not is_user_session(path):
                raise KeyError("Unknown session")
            resolved = AttachmentStore(self.root / ATTACHMENTS_DIRNAME).get(digest)
            if resolved is None:
                raise KeyError("Unknown attachment")
            data_b64, mime_type = resolved
            return base64.b64decode(data_b64), mime_type

        return await asyncio.to_thread(read)

    async def child_transcript(
        self,
        session_id: str,
        child_id: str,
        *,
        before_id: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """One page of a child's RAW transcript, in the parent's envelope.

        The rows come from ``read_transcript_page`` on the child's own
        directory and go out VERBATIM, which is what lets the renderer fold
        them through the same reducer it uses for the parent's history. Not
        ``hub op='peek'``: peek renders numbered single-string steps under a
        parent-agent context budget (``PEEK_MAX_STEPS = 50``,
        ``PEEK_STEP_CHARS = 600``) and DROPS compaction and bookkeeping rows,
        so a sidebar built on it would show neither the child's real
        conversation nor its structure (design § 9.1).

        The derived ``state`` is the only thing added, and it is derived from
        the FILESYSTEM because only the filesystem separates the two absences:
        ``pending`` (directory present, ``transcript.jsonl`` not written yet)
        is a child that may still speak, while ``gone`` (directory missing) is
        final. A row's persisted status cannot tell them apart, and a reader
        that guesses either says "gone" about a child that has not started or
        promises a transcript that will never come.

        NO bridge is acquired and no runtime is started — the containment proof
        refuses before anything else runs, and the whole body rides a worker
        thread because it stats, reads and parses files.

        ``limit`` is checked here as well as in the route's ``Query``: a route
        is not the only caller of an adapter, and a page ceiling that exists
        only in a declaration is one a second caller can walk around.
        """
        if not 1 <= limit <= CHILD_PAGE_LIMIT:
            raise ValueError(f"limit must be between 1 and {CHILD_PAGE_LIMIT}")

        def read() -> dict[str, Any]:
            child_dir = _contained_child_dir(self.root, session_id, child_id)
            if not child_dir.is_dir():
                # `gone` carries the cursor exactly as `pending` does below and
                # as `/history` does when the whole file is absent: a caller
                # that paged into a transcript which is no longer there is in
                # the same position either way, and the envelope's answer to
                # both is "re-read the tail and dedupe by id" (review round 1,
                # R1-3).
                return _absent_child_page("gone", before_id=before_id)
            if not (child_dir / TRANSCRIPT_FILENAME).exists():
                return _absent_child_page("pending", before_id=before_id)
            try:
                page = read_transcript_page(child_dir, before_id=before_id, limit=limit)
            except FileNotFoundError:
                # Vanished between the check above and the open: the same fact
                # as "never written", and an ordinary race rather than a 500.
                return _absent_child_page("pending", before_id=before_id)
            return {
                "entries": [json.loads(row.to_json()) for row in page.entries],
                "has_more": page.has_more,
                "cursor_missing": page.reconciled,
                "state": "ready",
            }

        return await asyncio.to_thread(read)

    async def child_attachment(
        self, session_id: str, child_id: str, digest: str
    ) -> tuple[bytes, str]:
        """Decoded bytes and mime type for one attachment in a CHILD transcript.

        The mirror of :meth:`attachment`, and the same containment proof runs
        before the store is touched (:func:`_contained_child_dir`) so a child
        read cannot become a way to enumerate another conversation's media. The
        store itself is content-addressed and shared across conversations by
        design (see the parent route's docstring), so this is a membership gate,
        not a per-conversation partition: what it rejects is an unknown parent,
        a child its parent never launched, and a child that is still on disk
        without the subagent marker.
        """

        def read() -> tuple[bytes, str]:
            _contained_child_dir(self.root, session_id, child_id)
            resolved = AttachmentStore(self.root / ATTACHMENTS_DIRNAME).get(digest)
            if resolved is None:
                raise KeyError("Unknown attachment")
            data_b64, mime_type = resolved
            return base64.b64decode(data_b64), mime_type

        return await asyncio.to_thread(read)

    async def create(
        self,
        cwd: str,
        *,
        target: dict[str, str] | None = None,
        model: dict[str, str | None] | None = None,
    ) -> str:
        """Create a draft session's record.

        ``model`` is the caller-validated birth selection (see the route), stored
        in the session's own marker so the FIRST turn can be born on it. It is
        additive and optional: an omitted ``model`` writes the marker byte-for-byte
        as before, which is what makes an older client's create identical.
        """
        self.assert_admitting()
        directory = resolve_working_directory(cwd)
        binding = {"agent": "", "team": ""}
        if target:
            from local_operator.agents import AgentRegistry
            from local_operator.server.utils.desktop_profiles import validate_target
            from local_operator.teams import TeamRegistry

            binding[target["kind"]] = await asyncio.to_thread(
                validate_target,
                AgentRegistry(self.root),
                TeamRegistry(self.root),
                target["kind"],
                target["name"],
            )
        session_id = uuid.uuid4().hex[:12]
        path = self.root / "sessions" / session_id

        def persist() -> None:
            path.mkdir(parents=True, mode=0o700)
            if target:
                write_session_attachment(path, **binding, goal="")
                stored = read_session_attachment(path)
                if (
                    stored is None
                    or stored.agent != binding["agent"]
                    or stored.team != binding["team"]
                ):
                    # Never publish desktop.json after a best-effort writer lost
                    # the attachment. No possibly admitted work is deleted.
                    raise ValueError(
                        "The selected profile could not be saved. Retry after checking storage."
                    )
            # An explicitly created desktop draft needs an identity after an
            # HTTP restart, unlike the TUI's uncommitted welcome-screen draft.
            # The chosen model is ADDITIVE: a marker written without it is the
            # document every earlier build wrote, and every reader here reads
            # ``cwd`` by key. A stored pair is what makes the choice survive the
            # window that chose it — the record outlives the request, and the
            # first turn is born from it (``draft_birth_selection``).
            #
            # Through the shared writer, which carries the model key too, so the
            # MOVE route's second call site and this one cannot disagree about
            # the bytes, the mode or the fields.
            write_desktop_marker(path, directory, model=model)

        await asyncio.to_thread(persist)
        return session_id

    async def binding(self, session_id: str) -> dict[str, str | None]:
        def read() -> dict[str, str | None]:
            stored = read_session_attachment(self.root / "sessions" / session_id)
            return {
                "agent": stored.agent or None if stored else None,
                "team": stored.team or None if stored else None,
            }

        return await asyncio.to_thread(read)

    async def list(self, limit: int) -> list[dict[str, Any]]:
        def rows() -> list[dict[str, Any]]:
            entries = load_catalog(self.root, limit=limit)[:limit]
            attention: dict[str, dict[str, Any]] = {}
            with contextlib.suppress(sqlite3.Error, OSError):
                attention = AttentionStore(self.root / "attention.db").state_many(
                    f"session/{entry.id}" for entry in entries
                )
            result = []
            for entry in entries:
                row = entry.row._asdict()
                stored = read_session_attachment(self.root / "sessions" / entry.id)
                row.update(
                    {
                        "active": entry.active,
                        "status": {"code": entry.status_code, "label": entry.status},
                        "binding": {
                            "agent": stored.agent or None if stored else None,
                            "team": stored.team or None if stored else None,
                        },
                        "preview": session_preview(self.root / "sessions" / entry.id),
                    }
                )
                if f"session/{entry.id}" in attention:
                    row["attention"] = attention[f"session/{entry.id}"]
                result.append(row)
            return result

        return await asyncio.to_thread(rows)

    @contextlib.asynccontextmanager
    async def session(self, session_id: str) -> AsyncIterator[DesktopSessionBridge]:
        """Hand out this session's bridge — or refuse, once the daemon has LATCHED.

        THE GATE IS HERE, AT THE DOOR, AND THAT IS THE WHOLE MECHANISM (review
        round 2, MAJOR-1). Every desktop route obtains its bridge here and this
        method builds every bridge the process hands out (the only
        ``DesktopSessionBridge(...)`` construction in the tree — pinned by
        ``test_serve_retire.py``'s walk), so one refusal covers every path that
        can admit or start work: the five round 1 gated, the five
        ``routes/desktop_lifecycle.py`` handlers review round 2 measured reaching
        ``bind_runtime()`` unrefused (``/mcp``, ``/credentials``, ``/fork`` and
        its child admission, ``/asides``, ``/adopt``), and every route a later
        edit adds.

        WHY THE DOOR RATHER THAN THE ROUTES. A list of gated routes is the defect
        this replaces, not the fix: ``/messages`` was gated, ``/commands`` was
        gated, and ``/mcp`` was not — three answers to one question, drifting
        apart exactly as fast as routes are added. The route-by-route
        enumeration survives only as a TEST (the refusal matrix and the walk in
        ``tests/unit/server/test_serve_retire.py``), where a new route shows up
        as a missing row instead of as a silently ungated path.

        READS ARE REFUSED TOO, and that is a deliberate narrowing of what this
        module used to claim. A session-scoped request handed to a process that
        has already told its readers to leave can only be served by the build it
        is leaving, and the typed refusal is the one answer that moves the client
        on (``DaemonRetiring``'s message says to reconnect to the successor).
        What stays readable is the RECORD plane, which is not this method and not
        this pool: ``GET /v1/desktop/sessions``, ``GET /health`` and the record
        file itself keep answering until the clean exit removes them, which is
        what lets any reader observe the handover.

        THE 404 BELONGS TO THE LOOKUP, NOT TO THE REFUSAL, so it is decided
        first: an unknown session is unknown whether or not this daemon is
        leaving, and keeping that answer stable is what makes "the 503 is the
        LATCH answering" a readable control in the evidence rather than an
        artefact of routing.
        """
        if not SESSION_ID.fullmatch(session_id):
            raise KeyError("Unknown session")
        async with self.lock:
            bridge = self.bridges.get(session_id)
            if bridge is None:
                path = self.root / "sessions" / session_id

                def locate() -> str:
                    if not path.is_dir() or not is_user_session(path):
                        raise KeyError("Unknown session")
                    # Through the TOLERANT reader, not ``json.loads``: a marker this
                    # code cannot parse (a hand edit, an interrupted write, a
                    # directory where the document should be) is a document with no
                    # cwd, and a session whose marker has no readable cwd still opens
                    # here — on the checkpoint fallback below — instead of failing the
                    # open with a 409/404 raised out of a parse error. Round 1 of
                    # #1110 wrote the coverage for a malformed marker and found the
                    # strict read behind it (R3).
                    stored = read_desktop_marker(path)
                    marker_cwd = (stored or {}).get("cwd")
                    if isinstance(marker_cwd, str) and marker_cwd:
                        return marker_cwd
                    # The cold facade restores cwd from the durable canonical
                    # checkpoint. This fallback is only used by pre-checkpoint
                    # transcripts, whose historical launch directory is unknown.
                    from local_operator.session.frontend_state import (
                        FRONTEND_CHECKPOINT_CUSTOM_TYPE,
                    )
                    from local_operator.session.transcript import Transcript

                    checkpoint = Transcript(path).latest_custom(FRONTEND_CHECKPOINT_CUSTOM_TYPE)
                    return str((checkpoint or {}).get("state", {}).get("cwd") or self.root.parent)

                # THE LOOKUP FIRST, so an unknown session stays 404 on a latched
                # daemon too: that is what lets "the 503 is the LATCH answering" be
                # read as a control in the evidence rather than as an artefact of
                # routing.
                cwd = await asyncio.to_thread(locate)
                self.assert_admitting()  # THE REFUSAL, before anything is built
                if len(self.bridges) >= BRIDGE_COUNT:
                    idle = [b for b in self.bridges.values() if b.users == 0]
                    if not idle:
                        raise ValueError("Too many active desktop sessions")
                    oldest = min(idle, key=lambda b: b.touched)
                    del self.bridges[oldest.session_id]
                bridge = DesktopSessionBridge(
                    self.root, session_id, cwd, retiring=self.retiring_probe
                )
                self.bridges[session_id] = bridge
            else:
                # Asked on the WARM path too, and that is not redundancy: the cache
                # is a cache of the same door, so without this a refusal would be
                # one a client could walk past by never having gone cold.
                self.assert_admitting()
            # Reserve under the pool lock; eviction must not remove a bridge
            # between lookup and its first acquire.
            await bridge.acquire()
        try:
            yield bridge
        finally:
            with CancelScope(shield=True):
                await bridge.release()

    async def close(self) -> None:
        await asyncio.gather(*(bridge.close() for bridge in self.bridges.values()))
        self.bridges.clear()
