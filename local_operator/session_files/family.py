"""Family discovery: which subagent children a conversation may name, proven on disk.

Moved out of ``server/utils/desktop_sessions.py`` (docs/design/remote-files-parity.md
§2, C3) so the RELAY can list a session's family for ``net_session_files`` without
importing the desktop module: that import measured +27 MiB RSS and +0.8 s and pulled
``fastapi`` into the relay. ``desktop_sessions`` re-exports every name, so callers and
tests are unchanged. Pure move apart from :func:`family_for`, lifted from the files
route's ``_family``.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from local_operator.resume import ORIGIN_SUBAGENT, is_user_session, session_origin
from local_operator.session.restored_rows import record_field, roster_records
from local_operator.session.transcript import read_latest_custom_entry

#: ``desktop_sessions.SESSION_ID``'s shape, restated rather than imported so this
#: module stays off the desktop module's import path (the reason it exists).
SESSION_ID = re.compile(r"^[a-f0-9]{12}$")


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


def _persisted_children(
    parent_dir: Path, *, indexed_roster: dict[str, Any] | None = None, indexed: bool = False
) -> list[Any]:
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
    from local_operator.session.roster_sidecar import (
        SUBAGENT_ROSTER_CUSTOM_TYPE,
        SUBAGENT_ROSTER_SIDECAR,
        read_roster_sidecar,
    )

    payload = read_roster_sidecar(parent_dir / SUBAGENT_ROSTER_SIDECAR)
    if payload is None and indexed:
        # Files already scanned the journal under its request budget, including a durable
        # negative result. Never fall back to a second unbounded journal read here.
        forked_at = fork_instant(parent_dir)
        if indexed_roster is None or (
            forked_at is not None and indexed_roster.get("ts", 0) <= forked_at
        ):
            return []
        payload = indexed_roster.get("details", {})
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
    return _prove_child(root, child_id, _persisted_children(parent_dir))


def _prove_child(root: Path, child_id: str, records: Sequence[Any]) -> Path:
    """The tail of :func:`_contained_child_dir`, over ALREADY-READ roster ``records``.

    Split out so :func:`contained_descendants` can prove many children against ONE read of a
    roster (a heavy session's sidecar is several MB; re-parsing it per child would turn a
    64-child family walk into 64 parses). The clauses are exactly the ones documented above -
    id shape and the parent's own-conversation rule stay in the caller, the roster naming, the
    resolved-inside-the-store check and the subagent-marker check live here - and there is one
    copy of them, so the single-child route and the family walk cannot disagree about who is a
    child.
    """
    sessions = root / "sessions"
    resolved_root = sessions.resolve()
    try:
        child_dir = (sessions / child_id).resolve()
    except (OSError, RuntimeError):
        # A path that cannot be resolved (a symlink loop) is not a child.
        raise SubagentChildUnavailable() from None
    if not child_dir.is_relative_to(resolved_root):
        raise SubagentChildUnavailable()
    named = False
    for record in records:
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


#: Bounds on a conversation's FAMILY walk (``session_files``): how deep a subagent chain is
#: followed and how many children are read. Past either, a child is COUNTED as skipped and the
#: response says so - a cap that is invisible reads as "that is everything".
FAMILY_MAX_DEPTH = 3
FAMILY_MAX_CHILDREN = 64


def contained_descendants(
    root: Path,
    session_id: str,
    *,
    indexed_roster: dict[str, Any] | None = None,
    indexed: bool = False,
) -> tuple[list[Any], int]:
    """``(children, skipped)``: the subagent sessions ``session_id`` launched, each PROVEN.

    ``children`` are :class:`~local_operator.session_files.service.ChildRef` values. A child is
    listed only if :func:`_prove_child` accepts it against the parent's persisted roster -
    the same containment proof the child transcript route applies, run once per record over one
    roster read. The parent's roster is FLAT (a grandchild is a record with ``parent_job_id``
    and ``depth`` 2 in the root's own sidecar), so one read covers the whole tree.

    ``skipped`` counts records dropped by the depth/count caps or refused by the proof; both are
    "the list may not be complete" and the caller reports them as one number. Never raises for
    a missing or unreadable roster (that is an empty family), and does not touch a session's
    runtime: it reads files.
    """
    from local_operator.session_files.service import ChildRef

    if not SESSION_ID.fullmatch(session_id):
        return [], 0
    parent_dir = root / "sessions" / session_id
    if not parent_dir.is_dir() or not is_user_session(parent_dir):
        return [], 0
    try:
        records = _persisted_children(parent_dir, indexed_roster=indexed_roster, indexed=indexed)
    except (OSError, ValueError):
        return [], 0
    children: list[Any] = []
    seen: set[str] = set()
    skipped = 0
    for record in records:
        raw_dir = record_field(record, "session_dir")
        child_id = Path(str(raw_dir or "").rstrip("/")).name
        if not SESSION_ID.fullmatch(child_id) or child_id == session_id or child_id in seen:
            continue
        seen.add(child_id)
        depth = record_field(record, "depth")
        depth = depth if isinstance(depth, int) and not isinstance(depth, bool) else 1
        if depth > FAMILY_MAX_DEPTH or len(children) >= FAMILY_MAX_CHILDREN:
            skipped += 1
            continue
        try:
            child_dir = _prove_child(root, child_id, records)
        except SubagentChildUnavailable:
            skipped += 1
            continue
        label = record_field(record, "label") or record_field(record, "agent_role")
        children.append(
            ChildRef(
                child_id=child_id,
                label=str(label) if label else None,
                depth=max(depth, 1),
                session_dir=child_dir,
            )
        )
    return children, skipped


def family_blocking(
    config_dir: Path,
    session_id: str,
    directory: Path,
    *,
    scope: str = "family",
    budget_s: float | None = None,
) -> tuple[Any, Any]:
    """``(service.Family, ledger.Snapshot | None)`` for one session's files index. Blocking.

    Lifted from ``routes/desktop_session_files._family`` so the desktop route and the
    relay's ``net_session_files`` (which runs on a relay op thread, no event loop) answer
    from ONE definition of a family.
    """
    from local_operator.session_files import ledger, service

    if scope == "self":
        return service.Family(session_id, directory), None
    # Roster absence is indexed during the SAME resumable pass as files. A sidecar still
    # wins; the indexed fallback cannot reparse a huge no-roster journal before the budget.
    snapshot = ledger.refresh(
        config_dir,
        session_id,
        directory,
        cwd=service.session_cwd(config_dir, session_id, directory),
        home=str(Path.home()),
        deny=service.make_deny(config_dir),
        budget_s=budget_s,
    )
    children, skipped = contained_descendants(
        config_dir,
        session_id,
        indexed_roster=snapshot.scan.roster if snapshot else None,
        indexed=True,
    )
    return (
        service.Family(
            session_id=session_id,
            session_dir=directory,
            children=tuple(children),
            skipped=skipped,
        ),
        snapshot,
    )


async def family_for(
    config_dir: Path,
    session_id: str,
    directory: Path,
    *,
    scope: str = "family",
    budget_s: float | None = None,
) -> tuple[Any, Any]:
    """:func:`family_blocking` off the event loop (the desktop route's form)."""
    return await asyncio.to_thread(
        family_blocking, config_dir, session_id, directory, scope=scope, budget_s=budget_s
    )
