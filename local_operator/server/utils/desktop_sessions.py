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
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Sequence
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
from local_operator.session.attached import READ_ATTACH_BUDGET_S, AttachedSession
from local_operator.session.attachments import ATTACHMENTS_DIRNAME, AttachmentStore
from local_operator.session.attention import AttentionStore
from local_operator.session.catalog import DECORATION_ATTENTION, load_catalog
from local_operator.session.cold_model import resolve_birth_effort
from local_operator.session.errors import MoveIndeterminate
from local_operator.session.frontend_state import (
    FrontendSync,
    FrontendUpdate,
    sync_wire_payload,
)
from local_operator.session.page_cache import load_transcript_page
from local_operator.session.restored_rows import record_field, roster_records
from local_operator.session.retention import DESKTOP_MARKER_NAME
from local_operator.session.runtime import registry
from local_operator.session.session_search import search_store
from local_operator.session.transcript import (
    TRANSCRIPT_FILENAME,
    read_latest_custom,
    read_latest_custom_entry,
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

# The pin store is the sidebar's OWN module, reused rather than re-implemented —
# for the reason the `move_targets` import above cites, which is also that
# module's stated model: it imports no Textual, so a non-Textual frontend can
# read the pins without a terminal. A second pin format here would be two
# surfaces disagreeing about which conversations are pinned, and the file would
# have two writers with two sets of rules for the cap and the prune.
# `tests/unit/test_import_graph.py` pins the absence of `textual`/`rich` on this
# module's own import graph, so the reuse cannot quietly start costing the
# server a terminal stack.
#
# ``set_pin`` is aliased only because this adapter's own method of that name is
# the caller's entry point; the store function stays the single writer.
from local_operator.tui.sidebar_pins import read_pins
from local_operator.tui.sidebar_pins import set_pin as set_sidebar_pin

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
    # ATOMIC, and the mode is applied BEFORE publication (review R2). The old
    # ``write_text`` then ``chmod`` truncated the authoritative file in place,
    # so a failure between the two left a world-readable marker and a crash
    # mid-write left it torn; and because the truncate happens first, a reader
    # racing the write sees an empty document rather than either answer.
    _stage_and_replace(marker, json.dumps(payload).encode())


def write_desktop_marker_bytes(marker: Path, data: bytes) -> None:
    """Atomically replace ``marker`` with already-rendered bytes, mode 0600.

    The ROLLBACK half of the marker contract, and a function rather than two
    lines inline because the ordering it enforces is the same one
    :func:`write_desktop_marker` needs (they share
    :func:`_stage_and_replace`): stage in the SAME directory, apply 0600 BEFORE
    publication, then ``os.replace``. A rollback that truncated the live marker
    and then chmodded it would leave a window where the authoritative file is
    empty or world-readable — and the rollback is exactly the moment nothing
    else is watching (review R2 / QA Q2).
    """
    _stage_and_replace(marker, data)


def _stage_and_replace(marker: Path, data: bytes) -> None:
    """Publish ``data`` at ``marker`` atomically, 0600, temp in the SAME dir.

    The one writer both halves of the marker contract go through, so the
    ordering is stated once: write the whole document to a unique staging file
    in the marker's own directory (``os.replace`` is only atomic within a
    filesystem), ``flush`` and ``fsync`` it so the rename cannot be ordered
    ahead of the bytes on a crash, apply 0600 BEFORE the file is reachable
    under the authoritative name, replace, then fsync the directory itself so
    the rename survives a power loss — the discipline ``config.py`` states for
    its own file. The directory fsync is best-effort: it is a durability
    upgrade attempted after the replacement ALREADY succeeded, so a filesystem
    that refuses ``O_RDONLY`` on a directory must not turn a good write into a
    failure.

    A stage that fails is removed rather than left behind: it is ours and
    unreferenced, it would otherwise accumulate in the session directory, and
    the marker name it is prefixed with is one a reader globs for.
    """
    staged = marker.parent / f".{marker.name}.{uuid.uuid4().hex}.tmp"
    try:
        # CREATED 0600, not chmodded afterwards (review round 2, N5): ``open``
        # applies the umask, so the staging file was briefly group/other
        # readable and its content is this session's working directory. The
        # exclusive create also cannot collide with a concurrent writer's stage.
        descriptor = os.open(staged, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staged, marker)
    except BaseException:
        with contextlib.suppress(OSError):
            staged.unlink(missing_ok=True)
        raise
    try:
        directory_fd = os.open(str(marker.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        pass


def _same_directory(left: str, right: str) -> bool:
    """Whether two paths name the SAME DIRECTORY, symlinks and all.

    A move to where the session already is must be a no-op, and comparing the
    spellings alone misses the case that matters on macOS: ``/tmp/x`` and
    ``/private/tmp/x`` are one directory, so a user who types the other spelling
    got a full retire-and-respawn — a ~1-3 s rebuild of a runtime that had not
    moved anywhere (QA round 1, Q3 on the shared move route). The fast path stays
    a string compare because that is the common case and it costs nothing; the
    identity check is ``os.path.samefile``, which is the filesystem's own answer
    rather than a second path-normalisation rule to keep in step with
    ``expand_path``'s deliberate symlink preservation.

    Both paths are directories that exist by the time this is asked (the target
    is validated first, the session's own is where it works), but ``samefile``
    raises ``OSError`` when one of them has vanished underneath us — in which
    case the honest answer is "not the same": the move proceeds and its own
    validation decides.
    """
    if not left or not right:
        return False
    if os.path.normpath(left) == os.path.normpath(right):
        return True
    try:
        return os.path.samefile(left, right)
    except OSError:
        return False


def _live_owner_cwd(root: Path, session_id: str) -> str | None:
    """The directory the LIVE owner of ``session_id`` reports, or ``None``.

    Discovery IS the owner's own account of where it works: ``SessionRecord.cwd``
    is written by the runtime process at publish and carried on every heartbeat,
    so it is not a value this server wrote about itself. That is exactly why the
    settlement below asks it — the copies a failed operation leaves behind are
    all its own, and comparing them to each other can only ever confirm its own
    belief (review round 2, N1).

    A record whose pid is gone is dropped as stale, and a wedged owner still
    holds its control socket and its directory, so it counts: the question is
    "which directory is this session working in", not "is it healthy".
    """
    for record, state in registry.scan(root):
        if state != "stale" and record.session_id == session_id:
            return str(getattr(record, "cwd", "") or "")
    return None


def _cwd_is_unconfirmed(root: Path, session_id: str, marker_cwd: str | None, resolved: str) -> bool:
    """Whether a bridge being BUILT must treat ``resolved`` as unconfirmed.

    WHY THIS EXISTS AT ALL (review round 3, MAJOR-1). The doubt used to live
    only in the bridge that watched the unknown outcome fail, and that is a copy
    in one process's memory: an eviction at ``BRIDGE_COUNT`` or a plain server
    restart rebuilt the bridge from the MARKER — the very copy the failed
    operation wrote — so the doubt was discarded on exactly the path the finding
    named, and the next move resolved a relative target against it and rewrote
    the durable marker from that base. A successor could then be engaged in a
    directory no party confirmed, while the docs promised the doubt was settled
    against the owner.

    The signature of a half-applied move is durable and readable here: a marker
    that names a DIFFERENT directory from the one the session's LIVE owner
    reports in its own record (``registry.scan``; not a value this server wrote).
    An owner still running in ``before`` while the marker says ``after`` can only
    mean the move did not take — a move is honoured by retiring the owner, and a
    definite refusal restores every copy — so the marker is a failed write and
    the bridge must reconcile before it acts on it.

    Deliberately NARROW, because a false positive costs a move: the doubt is only
    reconstructed when a marker exists and carries a cwd, and when a live owner
    disagrees with the resolved directory. A cold session (no record), a settled
    session (record == marker) and a checkpoint-only directory (no marker to
    doubt) all stay confirmed. The one conservative direction is an owner in the
    seconds of its own retirement: the outgoing record can still name the old
    directory while the successor has not published yet, which flags the doubt
    for that window. That refuses the next move with a reconcile sentence instead
    of acting on a possibly-stale copy, and it clears as soon as the successor's
    own record lands — the safe side of the line this finding is about.

    No separate read is needed for the two arguments: ``marker_cwd`` is the
    cwd the caller already got from the marker (``None`` when the directory came
    from the checkpoint fallback), and ``resolved`` is the directory the bridge
    will run with, so this asks only the question the marker cannot answer.
    """
    if not marker_cwd:
        return False
    owner = _live_owner_cwd(root, session_id)
    return bool(owner) and not _same_directory(owner, resolved)


async def _settle_unconfirmed_move(
    bridge: DesktopSessionBridge, marker_dir: Path, read_marker: Callable[[], bytes | None]
) -> None:
    """Settle which directory is in force after an UNKNOWN or half-applied move.

    Contract §A: an outcome whose answer never came back, or a rollback that
    could not run, is NEITHER a refusal nor a success, so the next operation must
    finish that reconciliation under the move lock instead of acting on an
    optimistic ``_cwd``.

    WHY THE OWNER IS ASKED, AND WHY COMPARING OUR OWN COPIES WAS NOT ENOUGH. The
    durable marker and ``bridge.cwd`` are BOTH written by the same failed
    operation from the same ``resolved`` value, so they agree by construction —
    a guard comparing only those two always reads "settled" and never sees the
    divergence it exists for. The facade is the second self-written copy, but it
    is the one that carries the OWNER'S answer (a definite refusal restores it to
    the directory the owner kept), so it disagrees far more often than the marker
    does. The authoritative account of where a live session works is the owner's
    own record, which is what :func:`_live_owner_cwd` reads.

    Three outcomes, and every one of them acts only on what a party confirms:

    * **The live owner confirms the facade, and the durable copy is the odd one
      out.** A move is honoured only by retiring the owner, so an owner that is
      still live — and whose own record names the directory the facade rolled
      back to — never accepted it. Both parties agree against the marker, which
      is therefore PROVABLY stale: it is repaired to the owner's directory, under
      the move lock, through the same atomic writer the move path uses, and the
      move proceeds. This is the case that used to leave a restart or a bridge
      eviction spawning a successor in a directory the move was REFUSED for.
    * **Every party agrees.** The durable target governs the successor and the
      base a relative new target resolves against; the doubt is settled and the
      flag clears.
    * **Anything else** — a live owner that disagrees with the marker while the
      facade also disagrees, an unreadable payload, an owner-side observation
      that matches neither copy — is genuinely unresolved. It is REPORTED for the
      caller to reconcile (503, the indeterminate class) and never resolved by
      preferring one copy, which is how the previous behaviour overwrote a
      committed move with a stale one.

    The three readbacks are logged at the refusal, so the state that needed
    reconciling is recorded rather than reconstructed later.
    """
    if not bridge.cwd_unconfirmed:
        return
    raw = await asyncio.to_thread(read_marker)
    durable = ""
    if raw is not None:
        try:
            payload = json.loads(raw)
        except ValueError:
            payload = None
        if isinstance(payload, dict):
            durable = str(payload.get("cwd", ""))
    remote = bridge.remote
    facade = str(getattr(remote, "cwd", "") or "") if remote is not None else ""
    owner = await asyncio.to_thread(_live_owner_cwd, bridge.root, bridge.session_id)
    observed = f"marker={durable!r} bridge={bridge.cwd!r} facade={facade!r} owner={owner!r}"

    if owner and owner == facade and owner != durable:
        # The write is wrapped because this branch is entered PRECISELY when the
        # same file could not be written a moment ago: a second failure escaped
        # as a raw ``OSError`` out of the route, which answers 500 in a ladder
        # where every neighbour is a mapped 409/503 and the route's own contract
        # is that the failure classes below are the ladder's (review round 3,
        # MINOR-1). The state is unresolved either way, so it is reported as
        # such — never as a refusal, which would tell the client to act on a
        # directory nobody confirmed.
        try:
            await asyncio.to_thread(
                write_desktop_marker,
                marker_dir,
                Path(owner),
                model=stored_draft_model(read_desktop_marker(marker_dir)),
            )
        except OSError as error:
            logger.error(
                "could not repair the unconfirmed move for %s in %s: %s",
                bridge.session_id,
                marker_dir,
                observed,
                exc_info=True,
            )
            raise MoveIndeterminate(observed) from error
        bridge.cwd = owner
        bridge.cwd_unconfirmed = False
        logger.warning(
            "unconfirmed move settled from the live owner's own record for %s: %s",
            bridge.session_id,
            observed,
        )
        return

    if (
        bool(durable)
        and all(copy == durable for copy in (bridge.cwd, facade))
        and (owner is None or owner == durable)
    ):
        bridge.cwd_unconfirmed = False
        return

    logger.error("unconfirmed move left unresolved for %s: %s", bridge.session_id, observed)
    raise MoveIndeterminate(
        observed,
        message=(
            "The session's working directory could not be confirmed after an "
            "interrupted move. Reconnect, then reconcile it before moving again."
        ),
    )


async def move_session(bridge: DesktopSessionBridge, requested: str) -> MoveReceipt:
    # The facade's bind lock does not cover the marker write or its rollback.
    # Serialize the whole transaction so a refused request cannot restore its
    # old marker over a different request's successful move.
    async with bridge.move_lock:
        # THE FENCE'S LIFETIME IS THE TRANSACTION'S (contract §C): set before the
        # first mutation and cleared only after the publication below, so a
        # legacy viewer can neither mount across the replacement frame nor be
        # admitted behind the precondition check that refused the move.
        bridge.move_in_progress = True
        try:
            return await _move_session(bridge, requested)
        finally:
            bridge.move_in_progress = False


async def _move_session(bridge: DesktopSessionBridge, requested: str) -> MoveReceipt:
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
    that possible: a DEFINITE refusal (a turn that arrived during the retire, a
    runtime too old to move, a missing capability) restores the marker and the
    bridge field, so a failure leaves the session working where it did rather
    than half moved. An UNKNOWN owner outcome is deliberately the other case —
    the owner may already have accepted, so nothing is restored and the bridge
    is marked unconfirmed for the next operation to reconcile (contract §A).
    """
    remote = bridge.remote
    assert remote is not None, "a move runs against an acquired bridge"
    # DURABILITY FIRST, the marker before the bridge field, and both before the
    # retire. `locate()` prefers the marker over the canonical checkpoint when
    # this bridge has been evicted or the HTTP server restarted; the bridge field
    # is what a re-``acquire()`` on THIS bridge passes to ``cold(cwd=…)``.
    # Defined ABOVE the resolution base below because an unconfirmed previous
    # move has to be settled before anything reads the optimistic cwd.
    marker_dir = bridge.root / "sessions" / bridge.session_id
    marker_path = marker_dir / DESKTOP_MARKER_NAME

    def read_marker() -> bytes | None:
        """The existing marker bytes, or ``None`` ONLY when it is not there.

        ``FileNotFoundError`` is the sole "absent" answer (contract §A). Every
        other read failure is raised with its real cause: the previous
        ``except OSError`` treated an unreadable or permission-denied marker as
        absence, so the rollback below would then DELETE the authoritative
        record of where the session works (review R2 / QA Q2).
        """
        try:
            return marker_path.read_bytes()
        except FileNotFoundError:
            return None
        except OSError as error:
            raise HTTPException(
                409, f"cannot read the session's working-directory marker: {error}"
            ) from None

    def restore_marker(previous_bytes: bytes | None) -> None:
        """Put the marker back atomically, and REPORT a failure to do so.

        A rollback that cannot run is not a log line beside an ordinary refusal:
        the durable copy would then name a directory the session is not in, so
        the caller turns this into an explicit indeterminate-state error rather
        than a warning followed by a confident-looking refusal (contract §A).
        ``MoveIndeterminate`` is the honest class: the filesystem and the owner
        now disagree and only a reconciliation settles which one is in force.
        """
        try:
            if previous_bytes is None:
                marker_path.unlink(missing_ok=True)
                return
            write_desktop_marker_bytes(marker_path, previous_bytes)
        except OSError as error:
            # LOGGED HERE, where the errno and the path still exist as a cause:
            # the class keeps the detail off the wire (it names directories), so
            # without this the operator sees a generic "reconcile" and no trace
            # of a failed rollback (review round 2, N3).
            logger.error(
                "could not restore the working-directory marker for %s: %s",
                bridge.session_id,
                error,
                exc_info=True,
            )
            raise MoveIndeterminate(
                f"could not restore the working-directory marker for {bridge.session_id}: {error}"
            ) from error

    await _settle_unconfirmed_move(bridge, marker_dir, lambda: read_marker())
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
        # the route's ``errors()`` ladder: that ladder's ``OSError`` arm claims
        # ONLY the disk-full errnos (ENOSPC/EDQUOT) and re-raises the rest, so
        # this one would reach the user as a 500 rather than as the path it is.
        raise HTTPException(409, f"cannot move to {requested}: {error}") from None

    resolved = str(directory)
    if _same_directory(resolved, previous):
        # NOTHING is written and nothing is retired. This is the TUI's "already in
        # ~/x", and it is also what makes a retried move idempotent: the receipt
        # journal admits a second POST with the same request id, and a move that
        # had already landed must not retire the runtime a second time.
        # ``will_wait`` is False rather than sampled: no transition is about to
        # happen, so there is no wait for it to have been a hint about.
        #
        # The receipt reports the SESSION's own spelling and label, not the
        # typed one: for a no-op through a symlink those are two names for one
        # directory, and the stream the renderer reconciles against reports the
        # session's. The TUI's symlink-PRESERVING spelling is still what a real
        # move is labelled with, below — this is only what "you are already
        # here" answers with.
        return MoveReceipt(
            cwd=previous,
            label=format_label(previous),
            outcome="unchanged",
            will_wait=False,
        )

    # Sampled BEFORE the call, exactly as ``_apply_move`` does, because after it
    # the answer is about a transition that has already happened — and this is a
    # HINT for an operator reading a receipt, never a gate.
    will_wait = remote.move_will_wait()
    # The TUI's own home-aware rendering of the directory being moved TO, kept
    # symlink-preserving on purpose: a user who moved into `~/current-project`
    # means the symlink, and printing its target back at them would read as the
    # move having gone somewhere else.
    label = format_label(directory)

    # PRECONDITIONS, BEFORE ANY MUTATION (contract §A). Both refuse while the
    # durable copy and the owner are still untouched, so a refusal costs the user
    # a sentence and never a half-applied move:
    #
    # * the OWNER must advertise the exclusivity fence, or it would ignore the
    #   flag and retire unguarded while a sibling facade is attached (review R3);
    # * no MOUNTED desktop viewer may predate ``frontend.replace``, because that
    #   frame is the only thing that repaints an already-mounted widget (review
    #   R4).
    if not remote.supports_exclusive_move:
        raise RuntimeError("this session's runtime is too old to be moved; /reload first")
    bridge.refuse_if_incompatible_subscribers()

    # The draft's stored model is CARRIED ACROSS, never re-derived: a move
    # changes ``cwd`` and nothing else, and the model key is the window's choice
    # for a conversation that has not run yet (``draft_birth_selection``). Read
    # through the same helpers every other marker reader uses, so a marker this
    # route could not parse degrades here exactly as it does there.
    previous_model = await asyncio.to_thread(
        lambda: stored_draft_model(read_desktop_marker(marker_dir))
    )
    previous_marker = await asyncio.to_thread(read_marker)
    try:
        await asyncio.to_thread(write_desktop_marker, marker_dir, directory, model=previous_model)
    except OSError as error:
        # THE FIRST MUTATION REFUSES WITH ITS REAL CAUSE, and there is nothing
        # to roll back (review R2). ``_stage_and_replace`` writes the whole
        # document to a staging file and only then replaces the marker, so the
        # authoritative bytes are never partially written and a failed write
        # cannot have changed them; the stage is removed on the way out. That
        # is exactly why this is a 409 refusal carrying the filesystem's own
        # message rather than a 500: the user's session is intact and the
        # actionable fact is that the marker could not be written. It is
        # deliberately NOT ``MoveIndeterminate`` either — nothing reached an
        # owner and no state is in doubt.
        raise HTTPException(
            409, f"cannot write the session's working-directory marker: {error}"
        ) from None
    bridge.cwd = resolved
    try:
        # The receipt's ``outcome`` IS this call's return vocabulary: the facade
        # documents exactly ``"cold"`` and ``"rebound"`` and nothing else, and
        # ``MoveReceipt`` spells the same two words plus the route-level
        # ``"unchanged"`` above. The cast narrows a ``str`` the protocol cannot
        # express (``RemoteSession.set_working_directory -> str``) to the Literal
        # the wire model owns, rather than the model being widened to ``str`` and
        # losing the fact a renderer switches on.
        #
        # ``exclusive=True`` is the desktop's endorsement of the owner fence
        # checked above: this call is what makes the move REFUSE while another
        # actual attach is registered rather than racing it.
        outcome = cast(
            Literal["cold", "rebound"],
            await remote.set_working_directory(resolved, exclusive=True),
        )
    except MoveIndeterminate:
        # NO ROLLBACK, and that is the whole point of the class: the owner may
        # already have accepted the move, so restoring the previous bytes would
        # overwrite a committed move with a stale one and the successor could
        # then spawn in the old path while the receipt said otherwise. The
        # marker deliberately stays at the accepted target and the route answers
        # "reconcile" (contract §A).
        #
        # The bridge is marked UNCONFIRMED rather than being trusted: the next
        # operation reconciles the durable copy against this belief under the
        # move lock before it resolves anything against it.
        bridge.cwd_unconfirmed = True
        raise
    except BaseException:
        # BaseException, not Exception: a CANCELLED move is the same hazard as a
        # refused one — the retire may already be in flight while this call's
        # caller went away — and `set_working_directory` rolls its own field back
        # on the same terms (review MINOR-2 there). Restoring the OTHER two copies
        # is this function's half of that invariant. A cancelled HTTP waiter no
        # longer reaches here at all (the route owns and joins this operation),
        # so this is the shutdown path.
        try:
            await asyncio.to_thread(restore_marker, previous_marker)
        except MoveIndeterminate:
            # The rollback could not run, so the durable copy and this bridge
            # now disagree about where the session works. That is the same
            # reconcile-before-acting state the unknown owner outcome produces,
            # and the next move settles it the same way.
            bridge.cwd_unconfirmed = True
            raise
        bridge.cwd = previous
        raise

    # Best effort by contract, and off the loop because it is a read, a write and
    # an atomic replace. Sharing the recents list with the TUI is a bonus of
    # sharing the file, not the point of writing it here.
    await asyncio.to_thread(remember_recent, bridge.root, directory)

    # A DEFINITE accepted move is itself the reconciliation: both the durable
    # copy and this bridge now name one directory, so any earlier doubt is
    # settled.
    bridge.cwd_unconfirmed = False
    return MoveReceipt(cwd=resolved, label=label, outcome=outcome, will_wait=will_wait)


class LegacySubscriberDuringMove(Exception):
    """A legacy viewer tried to mount during a move; see ``subscribe``.

    Its own class rather than a ``ValueError`` because the events route must
    answer it differently: a full subscriber table is a 404-shaped "no room",
    while this is "your build cannot follow a move in flight" — a 409 the
    renderer can act on by updating rather than by retrying blindly.
    """


@dataclass(eq=False)
class DesktopSubscription:
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    queue: asyncio.Queue[tuple[dict[str, Any], int] | None] = field(
        default_factory=lambda: asyncio.Queue(maxsize=REPLAY_COUNT)
    )
    queued_bytes: int = 0
    visible: bool = False
    can_notify: bool = False
    #: Whether this subscriber's renderer can consume ``frontend.replace``.
    #: False for every existing caller, which is what makes the move path
    #: refuse rather than silently leave a mounted viewer stale.
    frontend_replace: bool = False
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
        cwd_unconfirmed: bool = False,
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
        self.move_lock = asyncio.Lock()
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
        #: Set for the duration of ``_move_session`` (contract §C). Read by
        #: :meth:`subscribe` so a legacy viewer can neither be left stale by an
        #: in-flight move nor admitted behind its precondition check.
        self.move_in_progress = False
        #: Set when a move ended with an UNKNOWN owner outcome (contract §A), so
        #: the next operation settles which directory is in force before it
        #: resolves anything against this bridge: see
        #: :func:`_settle_unconfirmed_move`. Never a retry token — the receipt
        #: journal is what makes the request at-most-once.
        #:
        #: ALSO RECONSTRUCTED AT CONSTRUCTION by the pool, from the durable
        #: evidence (``marker != the live owner's record``, see
        #: :func:`_cwd_is_unconfirmed`), because a doubt that lives only in the
        #: bridge that watched the failure is discarded by the eviction and
        #: restart paths this flag exists to cover (review round 3, MAJOR-1).
        self.cwd_unconfirmed = cwd_unconfirmed

    def has_legacy_subscriber(self) -> bool:
        """Whether any LIVE subscriber cannot consume ``frontend.replace``."""
        return any(not sub.frontend_replace for sub in self.subscribers.values())

    def refuse_if_incompatible_subscribers(self) -> None:
        """Refuse a move, before it mutates, if a mounted viewer cannot repaint.

        The replacement frame is the ONLY thing that carries the accepted
        directory to an already-mounted desktop viewer in the COLD case — there
        is no successor runtime to publish it — so moving with such a subscriber
        attached would leave it showing the old directory while the receipt
        claimed success (review R4). Refusing is the honest answer, and the
        sentence names the action.
        """
        if self.has_legacy_subscriber():
            raise RuntimeError(
                "This session is open in an older desktop window. Update the desktop "
                "app, then move again."
            )

    def publish_frontend_replace(self) -> None:
        """Publish this bridge's CURRENT frontend projection as a replacement.

        Desktop-only, and NOT an ordinary ``frontend.update`` — that is the whole
        point (review R4). A local move installs state without touching the
        owner's clock, so a delta would carry the owner's UNCHANGED epoch and
        sequence, and the shipped renderer rejects a same-sequence update as
        stale (measured against the real reducer, which drops it and leaves a
        cold viewer on the old directory forever). This frame is ordered by the
        BRIDGE's own outer cursor instead, and the renderer consumes it as the
        authoritative projection. It carries no history field, so it neither
        creates a gap nor invalidates a history cursor.
        """
        if self.remote is None:
            return
        self.publish(
            # The whole bounded ``FrontendSync`` from ``state()``, not just its
            # ``snapshot``: the consumer needs the owner's epoch alongside the
            # state, and sending the sync payload keeps this frame's ``frontend``
            # field the same shape as the bootstrap snapshot's.
            "frontend.replace",
            {"frontend": self.state(), **self._cold_fields()},
        )

    def _cold_fields(self) -> dict[str, Any]:
        """The cold contract as the wire states it — see ``docs/DESKTOP_API.md``.

        ``cold`` is the boolean every existing renderer reads; ``cold_reason`` is
        the TOKEN that says which of the three cases it is, and ``attaching`` says
        an authenticated dial is retained and its canonical state has not arrived
        yet. Computed together, deliberately, so no frame can state one and
        contradict another — a frame claiming ``cold: false`` while a dial sat
        unsynced is the exact conflation these tokens exist to remove.

        Three facts the bridge can actually establish, and no copy: the token is
        the contract and the sentence belongs to the surface (the same discipline
        the routes' error ``code`` already follows). ``no-runtime`` for a facade
        with no remote at all; otherwise the facade classifies it (see
        ``AttachedSession.cold_reason``), defaulting to ``no-runtime`` for a cold
        facade no read has classified — which is what makes the field safely
        ADDITIVE for a reader that predates it, since
        ``cold ? "no-runtime" : null`` is the documented fallback.
        """
        remote = self.remote
        if remote is None:
            return {"cold": True, "cold_reason": "no-runtime", "attaching": False}
        return {
            "cold": remote.is_cold,
            "cold_reason": remote.cold_reason,
            "attaching": remote.attaching,
        }

    async def acquire(self, *, read: bool = False) -> AttachedSession:
        """Take one reference on this bridge, binding the owner if it is cold.

        ``read`` is the READ envelope, and it is a property of the ROUTE rather
        than of the facade: reading state must never be able to fail because an
        existing runtime was too busy to answer, because the durable answer is on
        disk in the same process (``snapshot``/``history``). Read mode therefore
        bounds its one attempt at ``READ_ATTACH_BUDGET_S`` and answers cold when
        it does not land, keeping the authenticated dial for the rollover. The
        CONTROL envelope (the default) is unchanged: one attempt on the foreground
        envelope, and a raise the route ladder turns into a named refusal — a
        write that was not admitted must say so rather than be reported as a
        served read.
        """
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
                    # THE LOCAL REPLACEMENT SEAM (contract §C). A move installs an
                    # accepted directory on the facade WITHOUT the owner's
                    # epoch/sequence moving, so telling desktop subscribers as an
                    # ordinary ``frontend.update`` would present a same-sequence
                    # delta the renderer discards — the frame is published here
                    # instead, through the bridge's own outer cursor.
                    remote.set_local_cwd_callback(lambda _cwd: self.publish_frontend_replace())
                    self.unsubscribers = [
                        remote.subscribe(self._event),
                        remote.subscribe_frontend(self._frontend).unsubscribe,
                    ]
                await self.remote.attach_existing(budget=READ_ATTACH_BUDGET_S if read else None)
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
        # THE COLD PAIR RIDES THIS FRAME TOO, and it is the frame that makes the
        # difference: a read that attached cold and retained its dial learns the
        # owner came back through the ROLLOVER this store publishes on
        # ``_install_frontend(..., publish=True)`` (a new epoch, full changes). A
        # renderer told only by the opening snapshot would keep painting the
        # cold/attaching row over a live conversation for the rest of its life.
        # Additive: the same three fields the snapshot carries, so a renderer that
        # knows the pair reads it here and one that does not ignores them.
        payload.update(self._cold_fields())
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
            # The payload builder calls the PUBLIC composer
            # (`notifications.compose`), which is the seam T-B13 patches, so a
            # composer that raises is caught here and costs the BANNER — never
            # the attention frame published above, which is the sync this bridge
            # exists for.
            from local_operator.notifications import notification_payload

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
            #
            # THE PAYLOAD IS BUILT BY THE SHARED BUILDER, not inline. The feed
            # ships the same payload for the same completion and the desktop
            # collapses the pair on `dedupe_key`; two dict literals would be one
            # field away from two banners for one turn. The bridge takes the
            # default `focus_policy` (`when_unfocused`) because this stream
            # exists only while an app is DISPLAYING the session — the card is
            # on screen already, so a banner on top of it is the interruption
            # the policy exists to prevent. The feed derives `always` instead:
            # its frames are about sessions nobody is looking at.
            payload = await asyncio.to_thread(
                notification_payload,
                state["kind"],
                session_dir=self.root / "sessions" / self.session_id,
                token=token,
                session_id=self.session_id,
                session_name=session_name,
            )
            self.publish("notification", payload, replay=False)
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
                **self._cold_fields(),
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

        Through ``load_transcript_page`` rather than a bare ``to_thread`` around
        the reader: the SSE open frame asks for THIS read seconds after this
        method already did (see ``events``), and the renderer's reconcile walk
        asks again for pages it has, so the second request for unchanged rows
        should cost a dict lookup rather than a second decode of the same bytes.
        The reader's contract, its special returns and its ``FileNotFoundError``
        are unchanged; only who pays for the read is.
        """
        try:
            page = await load_transcript_page(
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
            # NO ``timeout`` HERE, and that is the documented envelope rather
            # than an oversight: the re-assert's bound is ``_DESKTOP_WATCH_ACK_BOUND_S``
            # (5 s), which belongs to the LEASE — its TTL is 45 s and the beat that
            # renews it is 15 s, so this hint's patience is the lease's business and
            # not a read's. A read narrows it through ``update_desktop_watch``'s
            # ``timeout`` so the hint can never lengthen a read; a BEAT is not a
            # read, and clamping it to the remainder of one request's budget would
            # make the renewal's patience depend on which request happened to
            # arrive first. ``docs/DESKTOP_API.md`` states the resulting envelope
            # (≤ 2 s attach + ≤ 5 s hint) for this one route.
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

    def subscribe(self, *, frontend_replace: bool = False) -> DesktopSubscription:
        """Register one event subscriber.

        ``frontend_replace`` records whether this subscriber's renderer
        understands the desktop-only ``frontend.replace`` frame. It defaults to
        False, so a caller that does not know the flag negotiates nothing and
        the move path treats it as a viewer that must not be left stale.
        """
        if len(self.subscribers) >= SUBSCRIBER_COUNT:
            raise ValueError("Too many event subscribers")
        if self.move_in_progress and not frontend_replace:
            # FENCE DURING A MOVE (contract §C). A legacy subscriber arriving
            # while the move is between its preconditions and its publication
            # would be mounted across the one frame that tells a viewer the
            # accepted directory, and would then be silently stale for the rest
            # of its life. Refused rather than admitted: after publication the
            # move is over and a fresh subscription gets the normal snapshot.
            raise LegacySubscriberDuringMove(
                "A working-directory move is in progress; this viewer cannot follow it. "
                "Update the desktop app, then reconnect."
            )
        sub = DesktopSubscription(frontend_replace=frontend_replace)
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

    Runs off the event loop like every other reader here. It used to construct a
    ``Transcript`` for the legacy fallback, which parsed the parent's whole
    journal; it now takes the same backward one-row read the rest of this module
    does (``read_latest_custom_entry``), so the roster read is bounded by the
    distance from EOF to the newest roster row rather than by the journal, and a
    roster read must not block the loop a streaming turn is using either way.
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
        entry = read_latest_custom_entry(parent_dir, SUBAGENT_ROSTER_CUSTOM_TYPE)
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


@dataclass(frozen=True)
class SessionPage:
    """One page of rows, and the pinned conversations the page does not carry.

    WHY A SECOND POPULATION EXISTS. Older than the page means unrendered, and
    a pin is the one fact in a listing that the user chose rather than the
    recency window produced: on a store bigger than the client's page — which
    is the ordinary case, not an edge one — a pin made on an older conversation
    would otherwise have no row anywhere in the app, with no count and no trace
    of it. The operator's store is the measured case: 5,267 sessions against a
    500-row page.

    The TUI reports the same FACT differently, which is why this is not simply
    the desktop half of one promise. A TUI SIDEBAR holds its whole listing and
    draws only a window of it, so an off-page pin is COUNTED there —
    ``+N more pinned — scroll`` — and reachable by scrolling; it does not draw
    its row either. A client that holds one page has no scroll to offer and no
    listing to count against, so the row itself has to come with the answer.
    (``pinned_hidden_ids``, the parameter that sentence is usually about, is the
    other axis entirely: it keeps a HIDDEN session resolvable in the entries a
    sidebar scrolls, where a hidden id is absent because the catalogue never
    built it. An off-page id IS built and is dropped by the ``limit`` slice.)

    TWO LISTS, NOT ONE, and the route concatenates them for the wire. They are
    kept apart here because they answer different questions and only the caller
    knows how it wants them framed: ``rows`` is the page a ``limit`` describes,
    ``pinned_off_page`` is everything below it that the user pinned. The wire
    answer is their concatenation, which is also RANK ORDER — every extra ranks
    below every page row by construction — so a client that renders the two as
    one list needs no sort and no marker field.

    ``truncated`` keeps its original meaning — the ranking held more rows than
    the page — and deliberately says nothing about the extras: a client uses it
    to know whether more history exists, and an appended pin is a row it may
    already be holding rather than history it has not seen.
    """

    rows: list[dict[str, Any]]
    pinned_off_page: list[dict[str, Any]]
    truncated: bool


class DesktopSessions:
    """Bounded adapter cache; canonical identity lives in the session directory."""

    def __init__(self, root: Path, *, retiring: Callable[[], bool] | None = None) -> None:
        self.root = root
        self.bridges: dict[str, DesktopSessionBridge] = {}
        self.lock = asyncio.Lock()
        #: Session ids with at least one caller inside the HANDOUT window — from
        #: the moment this pool resolves a bridge for them to the moment that
        #: caller's first ``acquire()`` returns. Eviction skips a session in this
        #: window (``_evictable``), which is what replaced "the pool lock is held
        #: across ``acquire()``, so no other caller can run evict". Counted per
        #: session rather than flagged on the bridge because a caller is in the
        #: window BEFORE its bridge exists (the cold path) and because two
        #: concurrent callers for one session must not clear each other's claim.
        #: Mutated only from the event loop, and never across an ``await``.
        self._handouts: dict[str, int] = {}
        #: The COLD LOOKUP single-flight: ``session id -> (loop, task)`` for a
        #: ``locate()`` that is already running. The task is the pool's own
        #: ``asyncio.to_thread(locate)``, so a second caller for the same cold
        #: session awaits the read instead of starting a second one — and, since
        #: the read is what constructs the opening ``cwd``, one session cannot be
        #: built twice. The loop is carried so an entry left behind by a torn-down
        #: test loop reads as ABSENT rather than as a future bound to a dead loop
        #: (the ``RuntimeError`` ``tests/unit/tui`` would otherwise hit
        #: intermittently); the done-callback removes the entry, which is what
        #: makes a cancelled LEADER harmless — the task itself is never cancelled
        #: by a waiter (``asyncio.shield``), so it still settles and still
        #: publishes.
        self._locate_flights: dict[
            str, tuple[asyncio.AbstractEventLoop, asyncio.Task[tuple[str, str | None]]]
        ] = {}
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

    async def set_pin(self, session_id: str, pinned: bool) -> dict[str, Any]:
        """Put a session's pin into the state the caller asked for.

        DESIRED STATE RATHER THAN A TOGGLE, which is the whole reason this takes
        a flag: this backs an HTTP route, and a toggle is not idempotent over a
        link that can drop a response and retry. A retried toggle flips the pin
        BACK, which the user reports as "the pin keeps un-pinning itself" — a bug
        in the one feature whose entire value is that the pin stays put. A retry
        of this call lands on the same state.

        VALIDATION DELIBERATELY DIFFERS FROM ``acknowledge_attention`` ABOVE,
        which is the closest neighbour and the trap here. That method requires
        ``is_user_session(path)``; this one must NOT, because the sidebar pins
        DELEGATED RUNS too — a delegated run is a HIDDEN session, and pins are
        kept resolvable across BOTH visibility axes (``pinned_hidden_ids`` for
        the hidden one, ``pinned_off_page`` for a visible session outside a
        page) — and a desktop pin the user cannot remove is the worst shape of
        bug in this feature: the remedy for an unwanted pin is the thing such a
        check would refuse. A delegated run lives in `sessions/` like every other
        session, so the id-shape check plus the is-dir check is the whole
        admission test.

        Cold like its neighbour: no bridge, no runtime, no receipt. The write is
        a small file replace, and a receipt would buy at-most-once for a call
        that is already idempotent by construction.

        Returns the answer the route publishes, whose ``pinned`` is the state the
        store settled on. Today that is always the requested state — with ONE
        exception worth naming, because the response reads as a durability claim
        and is not one: on a config root this process cannot write,
        ``_write_pins`` swallows its ``OSError`` by the never-raise contract the
        store inherits from ``toggle_pin``, so this echoes the request over a file
        that did not change and the client renders a pin the store does not hold
        until its next catalogue read settles the row. Deliberately not fixed
        here: escaping the failure would break that pinned contract, and a
        read-back would reintroduce the race between the two writers that the
        store documents as accepted. A backend whose config root is read-only
        cannot serve this feature at all, and "the state the store settled on"
        is not a claim that every ``os.replace`` succeeded.
        """

        def apply() -> dict[str, Any]:
            if not SESSION_ID.fullmatch(session_id):
                raise KeyError("Unknown session")
            if not (self.root / "sessions" / session_id).is_dir():
                raise KeyError("Unknown session")
            return {
                "session_id": session_id,
                "pinned": set_sidebar_pin(self.root, session_id, pinned),
            }

        return await asyncio.to_thread(apply)

    async def acknowledge_attention_many(self, items: Sequence[tuple[str, str]]) -> dict[str, Any]:
        """Clear the unread completion marks a CLIENT enumerated, in one write.

        The machine-wide sibling of :meth:`acknowledge_attention`, and cold for
        the same reason: a read receipt never admits work, binds a viewer or
        starts a runtime, so this takes no bridge and no runtime spawn. It exists
        because the per-session route can only clear what it is told, one call at
        a time, and the desktop sidebar's pile of marks is thousands of
        conversations the user would have to open one by one.

        NOT a sweep, and the distinction is the whole safety story. The caller
        sends the completions it actually RENDERED, each ``(session_id,
        token)``; the store's own compare (:meth:`AttentionStore.
        acknowledge_many`) runs inside the write transaction against each
        item's CURRENT token, so a completion published after the client's
        render is not in the batch and stays unread. There is deliberately no
        "mark everything read" form: that would advance watermarks by the
        daemon's MAX(sequence) and silently clear results nobody saw.

        The conversation identity is DERIVED here (``session/<id>``) and never
        taken from the caller -- the client names a session id and nothing else,
        so it cannot write receipts for identities it cannot enumerate. An item
        whose id is malformed, whose directory is gone, or which is not one of
        the user's own sessions is answered ``unknown`` FOR THAT ITEM rather
        than refused for the call: one stale row in a batch of forty must not
        cost the other thirty-nine their receipt.

        Returns the three verdict buckets, in input order within each bucket::

            {"read": [state, ...], "superseded": [session_id, ...],
             "unknown": [session_id, ...]}

        ``read`` entries are the store's own post-write state dicts, which is
        what the list route already publishes per row -- the caller needs no
        second read to learn what its rows now say. ``superseded`` and
        ``unknown`` are the two verdicts that mean "not cleared", so a consumer
        can name the remainder instead of reporting a clean sweep it did not
        get.

        ONE worker hop for the whole batch, and ONE transaction inside it: the
        per-item validation is a directory stat, so doing it per item on the
        event loop would be a blocking ladder, while splitting the write would
        expose a partially applied batch. A ``sqlite3.Error`` propagates to the
        shared failure ladder (``server/utils/store_failures``), which splits it
        by condition -- contention answers the retryable 503, an unreadable or
        corrupt store the 500 that says retrying will not help -- and the single
        transaction guarantees nothing was written on either path.
        """

        def acknowledge() -> dict[str, Any]:
            store = AttentionStore(self.root / "attention.db")
            # Every item gets a slot first, so the buckets below can be filled in
            # INPUT order -- including the items rejected before the store is
            # ever consulted, which is also the order a caller's own listing is
            # in.
            outcomes: list[tuple[str, dict[str, Any] | None]] = [("unknown", None) for _ in items]
            batched: list[int] = []
            for index, (session_id, token) in enumerate(items):
                path = self.root / "sessions" / session_id
                if (
                    not SESSION_ID.fullmatch(session_id)
                    or not path.is_dir()
                    or not is_user_session(path)
                ):
                    continue
                batched.append(index)
            verdicts = store.acknowledge_many(
                [(f"session/{items[index][0]}", items[index][1]) for index in batched]
            )
            for index, verdict in zip(batched, verdicts):
                outcomes[index] = (verdict["status"], verdict["state"])
            result: dict[str, Any] = {"read": [], "superseded": [], "unknown": []}
            for (session_id, _token), (status, state) in zip(items, outcomes):
                if status == "read":
                    result["read"].append(state)
                else:
                    result[status].append(session_id)
            return result

        return await asyncio.to_thread(acknowledge)

    def bridged_notify_sessions(self) -> set[str]:
        """The FEED's key domain for sessions whose bridge will announce them.

        REVIEW ROUND 1, R10. The feed's ``bridged`` hook compares against
        ``session/<id>`` keys, and the route used to hand it ``set(pool.bridges)``
        — bare session ids. The two never intersected, so the exclusion was
        silently dead and every bridged session got a second, machine-wide
        banner composed for it. Returning the prefixed form from HERE rather
        than letting the feed normalise is deliberate: the pool is the thing that
        knows its own keys, and a conversion at the consumer would have to be
        repeated for every future consumer that gets it wrong the same way.

        A BRIDGE ALONE IS NOT ENOUGH, and that is the second half of the finding.
        A pooled bridge can be retained after its last subscriber left — that is
        what ``BRIDGE_COUNT`` and ``bridge.users`` exist for — and such a bridge
        publishes to nobody, so excluding it would open a background-notification
        HOLE where the prefix fix had just closed a duplicate-banner one. The
        predicate is therefore a LIVE, non-overflowing subscriber that can
        actually notify: the same filter ``_live_leases`` applies, plus
        ``can_notify``. Deliberately NOT ``_live_leases`` itself — that one
        requires ``watch_lock`` and is read under it, while this runs from the
        feed's poller, which must never take a lock the runtime's watch path can
        hold.
        """
        now = time.monotonic()
        return {
            f"session/{session_id}"
            for session_id, bridge in list(self.bridges.items())
            if any(
                not sub.overflow and sub.expires > now and sub.can_notify
                for sub in list(bridge.subscribers.values())
            )
        }

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
        refuses before anything else runs, and the FILESYSTEM half (containment,
        the two absence probes) still rides a worker thread because it stats
        paths. The PAGE itself now comes from ``load_transcript_page``, which
        runs the read in a worker of its own and shares the result with any other
        surface asking for the same rows — the child panel here is on a 1 Hz
        timer per open child, and re-decoding an unchanged journal sixty times a
        minute to answer the same question is what that seam exists to stop.

        ``limit`` is checked here as well as in the route's ``Query``: a route
        is not the only caller of an adapter, and a page ceiling that exists
        only in a declaration is one a second caller can walk around.
        """
        if not 1 <= limit <= CHILD_PAGE_LIMIT:
            raise ValueError(f"limit must be between 1 and {CHILD_PAGE_LIMIT}")

        def probe() -> Path | dict[str, Any]:
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
            return child_dir

        child_dir = await asyncio.to_thread(probe)
        if isinstance(child_dir, dict):
            return child_dir
        try:
            page = await load_transcript_page(child_dir, before_id=before_id, limit=limit)
        except FileNotFoundError:
            # Vanished between the probe above and the open: the same fact as
            # "never written", and an ordinary race rather than a 500. The probe
            # is deliberately KEPT rather than left to the reader's own absent
            # answer, because it is what distinguishes `pending` from `gone`.
            return _absent_child_page("pending", before_id=before_id)
        return {
            "entries": [json.loads(row.to_json()) for row in page.entries],
            "has_more": page.has_more,
            "cursor_missing": page.reconciled,
            "state": "ready",
        }

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

    async def list(
        self, limit: int, status_stamps: tuple[str, dict[str, int]] | None = None
    ) -> SessionPage:
        """One page of rows, plus the pinned rows the page does not carry.

        ``limit`` IS THE PAGE SIZE and the truncation verdict is computed here,
        because the two are one question: the caller used to ask for
        ``limit + 1`` rows and slice them itself to learn whether the store held
        more, and that trick cannot survive rows that are deliberately appended
        BEYOND the page (see ``SessionPage.pinned_off_page``) — the count would
        no longer mean what it is read to mean.

        The catalogue itself refuses rather than lying when the store cannot be
        walked (``load_catalog`` is strict), so everything below is about the
        DECORATION: a row that could not have its live state or wake index read
        says so in ``degraded`` instead of presenting the defaults as verdicts.
        The field is always present and always a list, so a client can read it
        without a presence check; ``attention`` below stays sparse because it
        is a per-row fact rather than a read-level one — and when THAT read is
        the one that failed, the affected rows name it in ``degraded``, so an
        absent ``attention`` key is never published as "nothing unread".

        ``status_stamps`` is ``(epoch, {session_id: revision})`` from the desktop
        feed, and it exists because TWO writers now ship the same fact: this row's
        ``status`` and the feed's ``session_status`` frame. A response computed
        before a frame this client already applied would otherwise clobber it —
        and the in-app marker effect fires a list on exactly the transition the
        frames speed up, so the race is the normal path rather than a
        hypothetical. A row whose session the feed has never published for is
        stamped with NEITHER key, and a client reads that as "no comparison
        available, take the list's value".

        The two facts TRAVEL TOGETHER and neither may displace the other:
        ``degraded`` says which read behind a row failed, the stamps say whether
        the row predates a frame this client has already applied, and a listing
        can be fully stamped and still be degraded — the stamps are per-row and
        optional, the marker is per-row and always present.

        Additive and defaulted: nothing else that lists sessions reads it, and an
        unstamped response is byte-identical to what this returned before.
        ``SessionRow`` is ``extra="allow"``, so the two keys serialize without a
        model change.
        """

        def rows() -> SessionPage:
            # ONE ``read_pins`` per request, and it is read BEFORE the catalogue
            # so the catalogue can resolve the pins the page will not carry.
            # `read_pins` already applies both the store's own read-time prune
            # (an id whose directory is gone is not a pin) and the id-shape
            # rule, so neither is re-implemented here — and one read means every
            # row of one answer describes the same pin set, which two reads a
            # millisecond apart would not guarantee.
            pins = set(read_pins(self.root))
            # ``limit + 1`` is the truncation PROBE and nothing else: one row
            # beyond the page is enough to answer "did the ranking hold more",
            # and asking for it here rather than at the route keeps the answer
            # from needing a second scan to interpret. Nothing in the store's
            # scan is bounded by this number (it is limit-independent), so the
            # extra row costs one rank position.
            entries = load_catalog(self.root, limit=limit + 1, pinned_off_page=tuple(pins))
            page_entries = entries[:limit]
            # A PINNED ROW THE PAGE DOES NOT CARRY, and the filter is on the id
            # rather than on the projected row's flag so it runs before the
            # projection. Everything from the page bound onwards is a candidate:
            # the probe row belongs here when it is itself pinned, because the
            # page does not carry it either. A pinned id that resolved to no
            # entry at all — a deleted directory, or a hidden delegated run the
            # catalogue never builds — is simply not among them, which is the
            # outcome `load_catalog` documents and the one a listing wants.
            extra_entries = [entry for entry in entries[limit:] if entry.id in pins]
            attention: dict[str, dict[str, Any]] = {}
            # NOT ``contextlib.suppress``: the suppression was silent, so this
            # route answered ``degraded: []`` -- "everything about this page was
            # read" -- while the ``attention`` key it could not build was simply
            # absent, which a client renders as "nothing unread". That is the
            # same confidently-wrong negative the catalogue's own attention read
            # is written to stop (``session.catalog.load_catalog``), one read
            # further out: the catalogue reads this store for the ROW's unseen
            # mark, and this is a second read of it for the wire's per-row
            # ``attention`` object, so a transient SQLITE_BUSY can hit one and
            # not the other.
            #
            # The failure is named on the rows rather than carried beside them
            # for the reason the catalogue states: every consumer walks the rows,
            # and a sibling value is a second channel a caller can forget.
            attention_degraded = False
            try:
                attention = AttentionStore(self.root / "attention.db").state_many(
                    f"session/{entry.id}" for entry in (*page_entries, *extra_entries)
                )
            except (sqlite3.Error, OSError):
                logger.warning("desktop listing could not read attention state", exc_info=True)
                attention_degraded = True
            # ONE PROJECTION for both populations, keyed by id: a pinned row off
            # the page carries exactly the fields a page row carries -- the same
            # attention object, the same status stamps, the same binding and
            # preview -- because the client renders them side by side in one
            # list and a second projector would be a second place for them to
            # disagree.
            projected: dict[str, dict[str, Any]] = {}
            for entry in (*page_entries, *extra_entries):
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
                        # ALWAYS PRESENT, BOTH VALUES. See `SessionRow.pinned`:
                        # the renderer's merge reads an absent key as "no claim",
                        # so a `false` here is load-bearing and omitting it would
                        # let a stale optimistic pin outlive a successful unpin.
                        "pinned": entry.id in pins,
                        # ``_asdict`` already carried this through as a tuple;
                        # spelled as a list here rather than left to the
                        # serializer, because JSON has one array type and a
                        # client deriving the listing-level ``degraded`` should
                        # not have to handle both.
                        "degraded": list(entry.row.degraded),
                    }
                )
                if attention_degraded and DECORATION_ATTENTION not in row["degraded"]:
                    # Deduped rather than appended blindly: the catalogue's own
                    # read of this store may already have named it on the row,
                    # and a source listed twice is a renderer that has to guess
                    # whether it means anything.
                    row["degraded"].append(DECORATION_ATTENTION)
                if f"session/{entry.id}" in attention:
                    row["attention"] = attention[f"session/{entry.id}"]
                if status_stamps is not None:
                    epoch, revisions = status_stamps
                    revision = revisions.get(entry.id)
                    if revision is not None:
                        row["status_epoch"] = epoch
                        row["status_revision"] = revision
                projected[entry.id] = row
            return SessionPage(
                rows=[projected[entry.id] for entry in page_entries],
                pinned_off_page=[projected[entry.id] for entry in extra_entries],
                # MORE ROWS THAN THE PAGE, which is the original question and
                # is still the right one: `entries` is the probe window plus
                # whatever was appended beyond it, and the window is full
                # exactly when the ranking held more than `limit` rows. Derived
                # from `entries` rather than from the page alone because the
                # probe row was ASKED for and correctly answered this.
                truncated=len(entries) > limit,
            )

        return await asyncio.to_thread(rows)

    async def search(self, query: str, limit: int) -> list[dict[str, Any]]:
        """Past conversations matching ``query``, each carrying its pin state.

        The projection lives here rather than in the route for the reason
        :meth:`list`'s does: the adapter is what owns the config root, and this
        is the only place that knows a row's wire shape needs the pin store read
        at all.

        ONE ``read_pins`` PER REQUEST — a membership test per match, not a file
        read per match — and both reads sit in the SAME worker thread as the
        scan, so a request costs one store walk and one pin-file read however
        many rows it answers. ``read_pins`` already applies the store's read-time
        prune (an id whose directory is gone is not a pin) and the id-shape rule,
        so neither is re-implemented here, and a pinned-but-unresolvable id
        simply does not match anything to begin with.
        """

        def rows() -> list[dict[str, Any]]:
            matches = search_store(self.root, query, limit=limit)
            pins = set(read_pins(self.root))
            return [
                {
                    "id": match.row.id,
                    "name": match.row.name,
                    "mtime": match.row.mtime,
                    "forked": match.row.forked,
                    "rank": match.rank,
                    "body_match": match.body_match,
                    # ALWAYS PRESENT, BOTH VALUES — see `SessionSearchRow.pinned`:
                    # a client synthesising a row from this answer reads an
                    # absent key as "no claim", and a pinned conversation would
                    # then render outside the Pinned section with no way back.
                    "pinned": match.row.id in pins,
                }
                for match in matches
            ]

        return await asyncio.to_thread(rows)

    def _take_handout(self, session_id: str) -> None:
        """Enter this session's eviction-protected window. Call under the pool lock."""
        self._handouts[session_id] = self._handouts.get(session_id, 0) + 1

    def _end_handout(self, session_id: str) -> None:
        """Leave it. Safe to call unconditionally, including after a failed lock."""
        remaining = self._handouts.get(session_id, 1) - 1
        if remaining > 0:
            self._handouts[session_id] = remaining
        else:
            self._handouts.pop(session_id, None)

    def _evictable(self, bridge: DesktopSessionBridge) -> bool:
        """Whether eviction may take ``bridge`` — asked with the pool lock held.

        ``users == 0`` alone is not enough once the slow halves run outside the
        lock: a bridge that has been resolved and not yet acquired also has
        ``users == 0``, and evicting it hands the next request for that session a
        SECOND bridge, since the pool no longer holds the first one. The handout
        count is that window, and asking here keeps the reservation explicit
        instead of leaving it implied by lock ownership.
        """
        return bridge.users == 0 and not self._handouts.get(bridge.session_id)

    def _locate_flight(
        self, session_id: str, locate: Callable[[], tuple[str, str | None]]
    ) -> asyncio.Task[tuple[str, str | None]]:
        """The cold LOOKUP for ``session_id``, shared by every concurrent caller.

        Call under the pool lock, so two callers cannot both decide they are the
        leader. IT MUST STAY AWAIT-FREE AND NON-BLOCKING: the caller holds the
        pool-wide lock across this call, so an ``await`` or a blocking syscall
        added here would hold every other session's open behind one session's
        cold lookup — the defect this method exists to remove, reintroduced one
        level down. ``locate`` is the caller's own closure (it is defined beside
        the path it resolves), and the task is created on the RUNNING loop — an
        entry whose loop is not this one is treated as absent rather than
        awaited, see ``_locate_flights``.
        """
        loop = asyncio.get_running_loop()
        entry = self._locate_flights.get(session_id)
        if entry is not None and entry[0] is loop:
            return entry[1]
        task = asyncio.create_task(asyncio.to_thread(locate))
        self._locate_flights[session_id] = (loop, task)

        def forget(settled: asyncio.Task[tuple[str, str | None]]) -> None:
            # Identity-checked: a later caller may already have replaced this
            # entry on a fresh loop, and dropping THAT one would lose its flight.
            current = self._locate_flights.get(session_id)
            if current is not None and current[1] is settled:
                del self._locate_flights[session_id]

        task.add_done_callback(forget)
        return task

    @contextlib.asynccontextmanager
    async def session(
        self, session_id: str, *, read: bool = False
    ) -> AsyncIterator[DesktopSessionBridge]:
        """Hand out this session's bridge — or refuse, once the daemon has LATCHED.

        ``read`` says the caller is READING state (``snapshot``, ``history``,
        ``events``, the ``/watch`` presence beat), and it is threaded down to
        ``bridge.acquire(read=True)`` so the one attempt to attach to an existing
        owner is bounded and its failure is a cold answer rather than a refusal.
        It belongs to the door rather than to a flag each route sets on its own:
        the envelope is a property of WHAT THE CALLER IS DOING, and this method is
        where every desktop route already declares that.

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

        WHAT THE POOL LOCK DECIDES, AND WHAT IT NO LONGER DOES. ``self.lock`` is
        the pool's single answer to one question — *which bridges are resident,
        and which of them may be evicted* — and it is held only for the frames
        that read or change that: the handout reservation, the cached lookup, the
        ``BRIDGE_COUNT`` eviction and the insertion. It is deliberately NOT held
        across either slow half of opening a session. Holding it there is what let
        one conversation's open become every other conversation's wait: measured on
        the operator's live backend, a 642-byte session's snapshot answered in
        **25 ms alone but 829 ms** while the 261 MB session's open was in flight,
        and the 261 MB open itself took 32.1 s (with a 96 MB open returning 503
        after 25.4 s) because a cold open queues behind the pool. Two mechanisms
        replace the lock's old reach here, and neither is a second lock: the
        ``_handouts`` count keeps a resolved-but-not-yet-acquired bridge out of the
        eviction path (``_evictable``), and ``_locate_flights`` single-flights a
        cold session's lookup so racing callers share one journal read. The bridge
        keeps its OWN lock for its own state — that one answers a different
        question and always did.

        Reproduced in isolation against a COPY of that 261 MB journal (never the
        operator's store): the same 642-byte session's snapshot took 29 ms alone
        and **7909 ms** — the whole 7.9 s open — while the cold open ran, and
        exactly ONE request completed in that window. On this change the open
        takes 2453 ms and **24** requests complete inside it (median 70 ms, worst
        308 ms). See ``docs/evidence/session-load-central-cache``.
        """
        if not SESSION_ID.fullmatch(session_id):
            raise KeyError("Unknown session")
        bridge: DesktopSessionBridge | None = None
        flight: asyncio.Task[tuple[str, str | None]] | None = None
        # ``taken`` rather than an unconditional release in the ``finally``: a
        # caller whose LOCK ACQUISITION is cancelled never incremented the count,
        # and decrementing anyway would release ANOTHER caller's reservation —
        # the count is per session, not per caller.
        taken = False
        try:
            async with self.lock:
                # THE HANDOUT IS TAKEN FIRST, while the pool lock is still held,
                # and it is what the eviction path reads instead of the lock
                # itself: a bridge this pool has resolved must survive until its
                # caller's first ``acquire()`` returns, even though both halves of
                # that wait — the cold lookup and the attach — now run with the
                # pool lock RELEASED. Taken inside this ``try`` for a reason the
                # review found: the WARM path's refusal (``assert_admitting``)
                # raises from inside this block, and a reservation stranded there
                # would make that session's bridge permanently unevictable and
                # turn the pool into one that can only refuse at ``BRIDGE_COUNT``.
                self._take_handout(session_id)
                taken = True
                bridge = self.bridges.get(session_id)
                if bridge is None:
                    path = self.root / "sessions" / session_id

                    def locate() -> tuple[str, str | None]:
                        """This session's opening directory, and the MARKER's own value.

                        The second element is provenance, not decoration: it is what
                        lets the caller ask
                        :func:`_cwd_is_unconfirmed` whether the directory is a
                        durable claim a failed move could have written (the marker),
                        or the checkpoint fallback for a pre-checkpoint transcript,
                        which has no marker to doubt.
                        """
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
                            return marker_cwd, marker_cwd
                        # The cold facade restores cwd from the durable canonical
                        # checkpoint. This fallback is only used by pre-checkpoint
                        # transcripts, whose historical launch directory is unknown.
                        from local_operator.session.frontend_state import (
                            FRONTEND_CHECKPOINT_CUSTOM_TYPE,
                        )

                        # A ONE-ROW read, not a ``Transcript(path)``: constructing the
                        # transcript JSON-decodes the whole journal (2059.5 ms on the
                        # operator's 261 MB conversation — the A/B table in
                        # ``docs/evidence/session-load-central-cache``), and this branch
                        # runs on every bridge creation for a session that carries no
                        # ``desktop.json`` marker. The reader answers from the tail
                        # backward and never creates the directory.
                        checkpoint = read_latest_custom(path, FRONTEND_CHECKPOINT_CUSTOM_TYPE)
                        return (
                            str((checkpoint or {}).get("state", {}).get("cwd") or self.root.parent),
                            None,
                        )

                    # THE LOOKUP FIRST, so an unknown session stays 404 on a latched
                    # daemon too: that is what lets "the 503 is the LATCH answering" be
                    # read as a control in the evidence rather than as an artefact of
                    # routing. It is also the read that parses the journal when the
                    # session has no marker, so it is SHARED rather than repeated: a
                    # second request for the same cold session awaits this one read
                    # instead of paying a second full parse, and one session therefore
                    # cannot be built twice.
                    flight = self._locate_flight(session_id, locate)
                else:
                    # Asked on the WARM path too, and that is not redundancy: the cache
                    # is a cache of the same door, so without this a refusal would be
                    # one a client could walk past by never having gone cold.
                    self.assert_admitting()
            if flight is not None:
                # THE SLOW HALVES, BOTH OUTSIDE THE POOL LOCK. This is the change
                # that stops one conversation's open from being every other
                # conversation's wait: a 25 ms request for a 642-byte session was
                # measured at 829 ms while the 261 MB session's open was in flight
                # — 33x for a read that is trivial — because the whole lookup and
                # attach held the pool-wide lock. Nothing below touches pool state
                # except in the guarded section, and the handout taken above is
                # what keeps this bridge out of the eviction path meanwhile.
                cwd, marker_cwd = await asyncio.shield(flight)
                self.assert_admitting()  # THE REFUSAL, before anything is built
                # THE BRIDGE THE SHARED LOOKUP WENT ON TO BUILD, read WITHOUT the
                # pool lock — and that read is safe precisely because of the
                # handout taken above: a bridge this pool has handed out cannot
                # be evicted out from under it, so no lock is needed to keep the
                # answer true. A FOLLOWER therefore stops here: no second lookup,
                # no second confirmation read, no lock at all. Sharing the flight
                # is what buys that; sharing the lock never could.
                bridge = self.bridges.get(session_id)
                if bridge is None:
                    # Reconstructed from the durable evidence BEFORE the bridge can
                    # be handed out, so the doubt survives the eviction and restart
                    # paths that rebuild it (review round 3, MAJOR-1). Off the loop
                    # — it reads the run directory through discovery — and off the
                    # lock, because it is I/O on the path this method exists to
                    # shorten. Only reached when this caller is the one that builds
                    # the bridge (or lost the race to build it).
                    unconfirmed = await asyncio.to_thread(
                        _cwd_is_unconfirmed, self.root, session_id, marker_cwd, cwd
                    )
                    async with self.lock:
                        # Re-read rather than assume: two concurrent cold callers both
                        # reach here and only the first may build. The second finds the
                        # bridge and reuses it, which is what makes the single-flight a
                        # single flight rather than a single LOOKUP.
                        bridge = self.bridges.get(session_id)
                        if bridge is None:
                            if len(self.bridges) >= BRIDGE_COUNT:
                                idle = [b for b in self.bridges.values() if self._evictable(b)]
                                if not idle:
                                    raise ValueError("Too many active desktop sessions")
                                oldest = min(idle, key=lambda b: b.touched)
                                del self.bridges[oldest.session_id]
                            bridge = DesktopSessionBridge(
                                self.root,
                                session_id,
                                cwd,
                                retiring=self.retiring_probe,
                                cwd_unconfirmed=unconfirmed,
                            )
                            self.bridges[session_id] = bridge
            assert bridge is not None  # resolved above: either found or built
            await bridge.acquire(read=read)
        finally:
            # The window closes when acquire() returns, success or failure: from
            # here on the bridge is an ordinary resident one, and ``users`` (which
            # acquire/release maintain) is what says whether anyone is using it.
            if taken:
                self._end_handout(session_id)
        try:
            yield bridge
        finally:
            with CancelScope(shield=True):
                await bridge.release()

    async def close(self) -> None:
        await asyncio.gather(*(bridge.close() for bridge in self.bridges.values()))
        self.bridges.clear()
