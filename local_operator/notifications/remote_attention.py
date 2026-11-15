"""A ``notify`` on ANOTHER device, raised on the session's origin (turn-ends are not).

WHY THIS MODULE EXISTS (remote-session-parity, 2026-11-11). A session the
operator starts on their head device and runs on a mesh node (``/new remote``,
an offload, a move) finishes its turns and calls ``notify`` ON THE NODE: the
completion lands in the node's ``attention.db`` and the item in the node's
``notifications.db``, where nobody is looking. The measured case: the head's
``notifications.db`` held 0 rows for the node's ``notify`` item
``n_xr44lx4zb5gi3mfn``, and its ``attention.db`` 0 completions for any node
session. A local session raises both on the operator's own machine; this is the
hop that makes a remote one do the same.

IT EXTENDS THE PARK PATH, never a second one. A remote approval park already
reaches the origin (``notifications/remote_park.py`` plus
``session/peer_rows.park_edges``) by reading the SAME peer listing the sidebar
polls; this adds two more facts to those rows and two more edge detectors
beside ``park_edges``:

* OWNER SIDE (:func:`stamp_rows_for_viewer`, called from the relay's catalogue
  op): a row gains ``attention: {token, outcome}`` for the owner's newest
  UNACKNOWLEDGED completion and ``notices: [{key, revision, priority, title},
  ...]`` for the newest :data:`NOTICES_ON_WIRE` unread ``notify`` items — ONLY
  when the asking link's device is the session's origin
  (``stamp.origin.source_device``), and only on a link whose viewer advertised
  ``wire.REMOTE_ATTENTION_V1``. Another member of the mesh
  sees neither: a completion is the operator's fact about THEIR session, and a
  co-member listing the node's sessions is not that operator.
* ORIGIN SIDE (:class:`RemoteAttentionWatcher`): one edge per (owner item,
  revision) — completions are no longer raised, see below — written through the SAME
  ``NotificationStore.notify`` door every other item uses — the kill switch
  (``notify_enabled``), the read/dismiss marks, the priority ladder and the
  store's retention are the shipped ones.

THE CLEAR IS THE EXISTING RECEIPT. Opening the session from the origin already
routes its read receipt to the owner (``net_session_receipt`` →
``attention.acknowledge_session_receipt``); an acknowledged completion is no
longer "unacknowledged", so the owner stops stamping it, and any listing read
that shows the session without that claim marks the origin's done row READ (the
operator saw it elsewhere), never deletes it. The settle reads the store and
the CURRENT listing, never process memory, so a receipt that landed while no
watcher ran still clears the row on the next read (QA round 1, Q1).

IDEMPOTENT BY CONSTRUCTION, which is what lets two hosts run it. The TUI and the
always-on ``lop serve`` process (and the desktop's own backend) may all watch at
once, and every one of them starts with an empty episode map after a restart.
Each row records its episode identity — the completion token, or ``rev:<N>``
for a notice — as its ``anchor_id`` and the store's ``episode``; a writer whose
episode is already the row's (or, for a notice, older than it) returns
``unchanged`` BEFORE writing — no new row, no unread flip, no banner — and only a
NEW token or a HIGHER revision moves the row. The check reads the row rather
than relying on the content digest alone, because the digest also covers the
title and the "runs on <device>" body, which a renamed session or a host that
read a nameless row would change (review round 1, m1). No cross-process lock or
ledger is needed.

REMOTE TURN-ENDS ARE NOT NOTIFICATIONS (operator rule, 2026-11-13). This module
used to raise a ``remote-done:<device>:<session>`` row (plus an OS banner) for
every finished turn the owner stamped. The first serve poll after v0.102.2
installed wrote 127 such rows in four seconds — one per historic node session,
because every completion that predated the feature was "unacknowledged" — and
34 ``remote-notice:`` rows for notices months old, all unread and stamped with
the sync time. The operator's rule: the inbox carries ONLY what an agent
deliberately posted (``notify``) or asked, so the completion producer and its
banner are gone. The owner still stamps ``attention`` (the sidebar and the
settle below read it), and :func:`settle_unclaimed_done_items` stays so a
``remote-done:`` row an older build wrote is still marked read once the owner
acknowledges it — it only ever marks read, never writes a row.

A FORWARDED NOTICE MUST BE NEW AND THE OPERATOR'S. Three filters, same date:

* only an explicit ``notify`` item (``NOT_FORWARDED_PREFIXES`` keeps parks,
  this module's own rows and bridge notices out — unchanged);
* never from a session another AGENT created on the owner (``origin.kind ==
  "agent"`` on the mesh stamp, :data:`AGENT_ORIGIN_KIND`): a reviewer/QA/probe
  child reports to the agent that spawned it, not to the operator — dropped by
  the owner in :func:`stamp_rows_for_viewer`, save its operator-signed site
  cards, which reach the verified operator link only
  (:func:`signed_card_target`); and, DEFAULT-DENY, a ``notify``
  item travels only when the stamp positively says the operator started the
  session (``started_by: operator``, :data:`STARTED_BY_FIELD`) — the origin
  re-applies this to every listing (``session/peer_rows``), so an owner too old
  to filter cannot reopen the flood. A session STARTED on the node and bound to
  the operator's device (``network.operator_binding``) carries its ``notify``
  items only when no agent started it (:func:`is_non_agent_session`, 2026-11-15).
  Open asks are exempt;
* NO BACKFILL: the owner carries each notice's own ``updated_at`` (``at``, ms)
  and the origin raises only a notice newer than its persisted first-contact
  baseline for that device (:func:`origin_baseline_ms`), writing the row with
  the owner's timestamp rather than "now". A notice with no ``at`` (an owner
  older than this rule) cannot prove it is new, so it is not raised. A
  forwarded OPEN ASK (``remote-ask:``) is exempt: it is still awaiting the
  operator, not history (QA round 3, Q3-1; see ``RemoteAttentionWatcher.observe``).

A notice passes its own priority through, so a ``high`` notice on the node is a
banner-eligible ``high`` row here, raised through the item ladder
(``notifications.delivery.deliver_item``) whose per-revision claim keeps the
TUI, the desktop feed and this writer to one banner between them.

WHAT THIS DOES NOT DO. Phone push reads ``attention.db``, not this inbox
(``mobile/push_worker.py``: its completion scan is built on ``AttentionStore``
records, cursor and deferral), and nothing here writes the origin's
``attention.db`` — a remote completion does not reach the phone yet; the
follow-up is a second push source in that worker reading ``remote-done:`` rows.
The completion carries no end time: the owner's completion store records
none, so no ``ended_at`` is invented here.

BEST-EFFORT BY CONTRACT, like the park module: nothing here raises.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Key namespaces of the two origin-side rows. ``remote-gate`` is the park's
#: (``remote_park.KEY_PREFIX``); these sit beside it so a reader can tell a
#: forwarded row from a local one by its key alone.
DONE_PREFIX = "remote-done"
NOTICE_PREFIX = "remote-notice"

#: Owner-side items NEVER forwarded as a notice. ``gate:`` rows are parks, which
#: reach the origin through ``pending`` (``remote_park``) — forwarding them here
#: too would put two rows on one park. ``remote-*`` rows are THIS module's and
#: the park module's own writes on a device that is itself an origin: echoing
#: them back would loop between two devices that each started a session on the
#: other. ``team-bridge-`` rows are the node's agent-message notices
#: (``builtin._RESERVED_NOTIFY_KEY_PREFIX``), which belong to that device's
#: connector, not to the session's operator.
NOT_FORWARDED_PREFIXES: tuple[str, ...] = (
    "gate:",
    "remote-gate:",
    f"{DONE_PREFIX}:",
    f"{NOTICE_PREFIX}:",
    "team-bridge-",
    # Reserved for :func:`open_ask_notices` below: an agent ``notify`` that
    # happened to use this key would otherwise be mistaken on the origin for a
    # forwarded ask, and retired when it left the listing.
    "remote-ask:",
)

#: ``origin.kind`` on a mesh stamp for a session an AGENT created on this device
#: (``lop network sessions --create`` run from an agent's shell, or the
#: ``sessions spawn peer=`` tool; ``network/cli.py`` sets it). Its notices are
#: never forwarded: see the module doc, "A FORWARDED NOTICE MUST BE NEW AND THE
#: OPERATOR'S".
AGENT_ORIGIN_KIND = "agent"

#: DEFAULT-DENY ATTESTATION (operator rule, 2026-11-14): "notifications reach the
#: dashboard ONLY when an agent deliberately calls notify (or asks) ... remote
#: mesh session turn-ends must not become notifications." ``origin.kind`` cannot
#: carry that promise: the owner's create op has defaulted a missing kind to
#: ``user`` since the mesh core, so every session an agent placed on a peer before
#: #3902 (and every one an older origin places) reads ``user`` — five agent-lane
#: workers on lop-remote-node-2 reached the inbox that way after v0.103.0. So a
#: second field, written ONLY by a build that knows who asked
#: (``relay._ctl_peer_create`` derives it from the attributed ``origin``), says
#: ``started_by: operator``; its absence — a legacy stamp, an older origin, a move
#: or a replica — denies. Asks are exempt: a question still waiting on the
#: operator always travels (:func:`stamp_rows_for_viewer`).
STARTED_BY_FIELD = "started_by"

#: The ``origin.kind`` values a peer CREATE writes (the owner defaults a missing
#: kind to ``user``). Only these are subject to the ``started_by`` default-deny on
#: the node's ``notify`` mount: a ``moved``/``fork`` stamp is a session the
#: operator carried to this device, and the owner never forwards its items anyway.
PEER_CREATE_ORIGIN_KINDS: frozenset[str] = frozenset({"user", AGENT_ORIGIN_KIND})
OPERATOR_STARTED = "operator"


def started_by_for_origin(origin_kind: str) -> str:
    """The ``started_by`` attestation for an ATTRIBUTED create (``user``/``agent``).

    Called only where the caller's identity was actually decided (the origin's
    control op, after ``network/cli._create_origin`` or the desktop route), never
    on the owner from a frame that may predate the field.
    """
    # Anything but an exact ``user`` — ``agent``, an empty or a future kind —
    # deliberately maps to ``agent``: default-deny means an attestation is only
    # ever OPERATOR when the caller was positively identified as the operator.
    return OPERATOR_STARTED if origin_kind == "user" else AGENT_ORIGIN_KIND


def origin_is_operator_started(origin: Any) -> bool:
    """Whether a mesh stamp's ``origin`` POSITIVELY says the operator started it.

    Default-deny: a missing/odd origin, a missing ``started_by`` (legacy stamps,
    moves, replicas) or an ``agent`` kind all answer NO. See
    :data:`STARTED_BY_FIELD` for why ``kind == "user"`` alone is not enough.
    """
    if not isinstance(origin, dict):
        return False
    if str(origin.get("kind") or "") == AGENT_ORIGIN_KIND:
        return False
    return str(origin.get(STARTED_BY_FIELD) or "") == OPERATOR_STARTED


#: ``path -> (stamp, {child: origin} | None)`` for the lineage registry: a
#: listing classifies every row, and the registry is append-only, so it is
#: re-read only when its stat stamp moves. ``None`` is a registry that EXISTS
#: but cannot be trusted (S6-r3), which classifies every session as an agent's.
#: The stamp carries ``st_ctime_ns`` and ``st_mode`` because a ``chmod`` changes
#: neither size nor mtime, and a memo keyed on those alone would keep serving
#: the rows read before the registry was made unreadable.
_lineage_memo: dict[str, tuple[tuple[int, int, int, int, int], dict[str, str] | None]] = {}


def _read_lineage_strict(path: Path) -> dict[str, str] | None:
    """``{child: origin}`` from an EXISTING registry, or ``None`` if it is not trustworthy.

    ``provenance.read_lineage`` deliberately swallows every error (a spawn's
    adoption logic must never fail on it), which here would read "no agent rows"
    — the permissive answer. This reader distinguishes the cases S6-r3 needs:
    an open/read error (permission, IO, a symlink refused by ``O_NOFOLLOW``),
    a 0-byte file and a file with no parseable row all answer ``None``.
    ``record_lineage`` only ever appends complete JSON rows and never truncates,
    so none of those is a state the harness itself leaves behind. Same
    first-row-wins semantics as ``provenance.read_lineage``.
    """
    out: dict[str, str] = {}
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                child = row.get("child") if isinstance(row, dict) else None
                if isinstance(child, str) and child and child not in out:
                    out[child] = str(row.get("origin") or "")
    except OSError:
        return None
    return out or None


def _lineage_says_agent(root: Path, session_id: str) -> bool:
    """The harness lineage registry records ``session_id`` as agent-opened (S6).

    FAIL CLOSED (security review S6-r3, manager direction): a registry that
    EXISTS but cannot be read, is empty, or has no parseable row is evidence of
    tampering — the session's own approved interpreter call can reach the file
    (HIGH), and emptying or ``chmod 000``-ing it would otherwise erase every
    agent row and let an agent session's ``notify`` travel. Such a registry
    answers "agent" for every session. Only an ABSENT registry (a fresh install
    whose first spawn has not happened) says nothing, and the ``origin.json``
    rule decides as in round 2. Asks never reach this function's verdict.
    """
    from local_operator.guardrails import provenance

    path = provenance.lineage_path(root)
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError:
        return True
    stamp = (info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_mode, info.st_ino)
    cached = _lineage_memo.get(str(path))
    if cached is None or cached[0] != stamp:
        if len(_lineage_memo) > 16:
            _lineage_memo.clear()
        _lineage_memo[str(path)] = (stamp, _read_lineage_strict(path))
        cached = _lineage_memo[str(path)]
    origins = cached[1]
    if origins is None:
        return True
    origin = origins.get(session_id)
    return origin is not None and origin not in provenance._HUMAN_ORIGINS  # noqa: SLF001


def is_non_agent_session(root: Path, session_id: str, origin: Any, *, started_here: bool) -> bool:
    """Whether ``session_id``'s ``notify`` items may reach the operator's devices.

    THE ONE PREDICATE FOR "NOT AN AGENT'S SESSION" (manager rule, 2026-11-15,
    confirming two operator rules): asks from any session always travel; a
    ``notify`` item travels only from a session NO AGENT started, because
    "child/reviewer/QA sessions never notify the operator directly"
    (2026-11-13) and "remote mesh session turn-ends must not become
    notifications" (#4001, 2026-11-14). There is no carve-out by notify kind:
    the store's session-scoped, un-prefixed items ARE the ``notify`` tool's
    (every other writer keys under :data:`NOT_FORWARDED_PREFIXES`), so this
    decides by who started the session, nothing else. Kept in one place so the
    pending operator decision (ask a-1075) can flip it here.

    * STAMPED (placed here by another device): #4001's default-deny unchanged —
      only a stamp attesting ``started_by: operator``
      (:func:`origin_is_operator_started`).
    * STARTED HERE (``started_here``: no foreign origin, bound to the operator's
      device by :func:`notice_target`): no agent child (``origin.kind``), no
      ``started_by: agent``, and the local ``origin.json`` marker absent (a
      session a human opened) or ``fork`` (the human's own branch). Every other
      marker — ``subagent``, ``agent-shell``, ``agent-workstream``,
      ``agent-config``, a future value, or one that cannot be parsed — is an
      agent's or unknown, and DENIES: the default-deny direction of #4001.
      The marker is written by the session's creator on this device, before
      the notify tool is ever offered (``session._session_was_agent_spawned``
      reads the same marker to withhold the tool), so a stampless human
      session and an agent-started one ARE distinguishable here.
    * THE LINEAGE REGISTRY FIRST (security review S6): ``origin.json`` is
      reachable by the session's own approved interpreter call (HIGH, not
      CRITICAL), and a deleted marker read as "human". The harness-written
      registry ``<config>/guardrails/lineage.jsonl``
      (``guardrails.provenance.record_lineage``, appended at the spawn sites by
      ``resume.mark_session_origin``; CRITICAL to write from the shell and file
      tools) records every session opened on an agent's behalf, so any row
      there with a non-human origin is an agent's, whatever ``origin.json``
      says or whether it exists.
    """
    if not started_here:
        return origin_is_operator_started(origin)
    if isinstance(origin, dict) and (
        str(origin.get("kind") or "") == AGENT_ORIGIN_KIND
        or str(origin.get(STARTED_BY_FIELD) or "") == AGENT_ORIGIN_KIND
    ):
        return False
    import json

    from local_operator.resume import ORIGIN_FORK, ORIGIN_NAME

    if not session_id:
        return False
    if _lineage_says_agent(Path(root), session_id):
        return False
    marker = Path(root) / "sessions" / session_id / ORIGIN_NAME
    try:
        raw = marker.read_text(encoding="utf-8")
    except FileNotFoundError:
        return True
    except OSError:
        return False
    try:
        payload = json.loads(raw)
    except ValueError:
        return False
    if not isinstance(payload, dict):
        return False
    return str(payload.get("origin") or "") in ("", ORIGIN_FORK)


def is_operator_signed_card(record: Any) -> bool:
    """Whether a queued ask record is a card ONLY the operator's signature answers.

    Today that is exactly a browser site card: a question whose ``question_id``
    is a site digest (``browser_bridge.site_approval.is_site_question_id``).
    ``asks/decision.py`` refuses a plain answer for one and routes it through
    ``verify_site_card`` and the signed ``bridge_decide`` door, so no agent —
    the session's own parent included — can answer it. Deliberately narrow:
    ``human_required`` alone is NOT this (any human at any surface may answer
    such an ask), and no other signature-only card exists to include.
    """
    if not isinstance(record, dict):
        return False
    from local_operator.browser_bridge.site_approval import is_site_question_id

    return any(
        isinstance(question, dict) and is_site_question_id(question.get("question_id"))
        for question in record.get("questions") or ()
    )


def signed_card_target(origin: Any, *, operator_device_id: str = "") -> str:
    """Where an AGENT CHILD's operator-signed cards go: the verified operator, or nowhere.

    THE ONE EXCEPTION TO "AN AGENT'S CHILD NEVER NOTIFIES THE OPERATOR"
    (2026-11-13), and only for :func:`is_operator_signed_card` asks (manager
    rule, 2026-11-15: "human-only signed cards ... forward to the bound operator
    device even from agent-origin and stampless sessions. Ordinary agent-child
    asks keep their current routing"). The parent agent cannot answer a site
    card — only the operator's signature can — so dropping it strands the agent
    until the request lapses (sessions 1c31cfff075b's a-652b/a-d271 on
    lop-remote-node-2, spawned by an agent on the operator's Mac, lapsed with
    nothing on the Mac).

    Only for a row :func:`notice_origin` calls an agent child (the set the
    catalogue leg drops; every other row's asks already travel by
    :func:`notice_target`), and only to ``operator_device_id`` — the caller's
    VERIFIED binding (``is_operator_link`` on the catalogue leg). Never to "the
    viewer the origin names": a co-member that placed the child here, even the
    origin itself, gets nothing unless it is the bound operator device. ``""``
    when the row is no agent child or no operator is verified.
    """
    if not operator_device_id or not notice_origin(origin)[1]:
        return ""
    return operator_device_id


#: The OWNER-side key namespace of a forwarded open ask, inside the existing
#: ``notices`` wire block: ``remote-ask:<ask id>``. Distinct from the store's
#: ``ask:`` read-mark refs (``store.ASK_REF_PREFIX``) and from every prefix
#: above. The origin row is therefore ``remote-notice:<dev>:<sess>:remote-ask:<id>``
#: and :func:`settle_unlisted_ask_items` finds it by this infix.
ASK_NOTICE_PREFIX = "remote-ask:"

#: How many notices one row carries for its origin, newest first (review round 1,
#: M2: carrying only the newest lost every earlier unread notice, ``high`` ones
#: included). WHY 8: the owner's inbox is never read on the node, so its unread
#: items persist and EVERY catalogue read for the origin carries the full list
#: for every peer-originated session — at most 8 x (the owner store's 120-char
#: ``KEY_MAX`` key + 120-char ``TITLE_MAX`` title + small fields), measured at
#: ≤2.5 KB of JSON per session (2,453 B worst case), which keeps a listing of tens
#: of such sessions well inside a relay frame. A notice is lost only when more
#: than 8 newer forwardable notices land between two origin reads; the origin
#: reads every 5 s against a 20 s listing cache, so that is >8 notices inside
#: ~20 s from one session — a runaway loop the store's own
#: ``NEW_ITEMS_PER_HOUR`` (30) refuses within the hour anyway. An origin that
#: was OFFLINE across more than 8 notices does miss the oldest (stated in the PR's
#: known limits). ``projection.NOTICES_MAX`` is the read-side twin; a test pins
#: them equal.
NOTICES_ON_WIRE = 8

#: How many unread items per session the owner scans to fill that list. Larger
#: than the list so a burst of EXCLUDED rows (parks, bridge notices) cannot crowd
#: the forwardable ones out, and bounded so the catalogue op never reads a whole
#: inbox.
_NOTICE_SCAN = 4 * NOTICES_ON_WIRE


def done_key(device_id: str, session_id: str) -> str:
    """One row per (device, session): a new completion UPDATES it, never stacks.

    The device is in the key for the reason ``remote_park.item_key`` gives: a
    session id meant to be mesh-unique is still not assumed to be.
    """
    return f"{DONE_PREFIX}:{device_id}:{session_id}"


def notice_key(device_id: str, session_id: str, key: str) -> str:
    """One row per OWNER item: a keyed ``notify`` that updates there updates here."""
    return f"{NOTICE_PREFIX}:{device_id}:{session_id}:{key}"


# ---------------------------------------------------------------------------
# Owner side: what a catalogue row carries for its origin
# ---------------------------------------------------------------------------


def notice_origin(origin: Any) -> tuple[str, bool]:
    """``(source_device, is_agent_child)`` of a stamp's ``origin`` mapping.

    THE ONE READING OF "WHO WOULD THIS SESSION'S NOTICES GO TO" (PR #4029
    review R1-1). Both the owner's catalogue leg (:func:`stamp_rows_for_viewer`)
    and the ``notify`` result (:func:`notice_forwarding`) decide on it, so the
    tool can never say "not forwarded" about a row the catalogue does put on
    the wire, or the reverse. It reads ``origin.source_device`` ONLY — never
    ``placement.mode``: the catalogue row carries ``stamp.origin`` whatever the
    mode, and a replica recovered by ``sync.promote_replica`` is stamped
    ``mode="local"`` with its old owner as ``source_device``, so its notices
    ARE forwarded to that owner. ``is_agent_child`` is the agent-origin drop
    (:data:`AGENT_ORIGIN_KIND`): such a row may still match a viewer for its
    completion badge, but its notices are never put on the wire. A non-mapping
    (no stamp, no origin) is ``("", False)``.
    """
    if not isinstance(origin, dict):
        return "", False
    return (
        str(origin.get("source_device") or ""),
        str(origin.get("kind") or "") == AGENT_ORIGIN_KIND,
    )


def notice_target(
    origin: Any, *, self_device_id: str = "", operator_device_id: str = ""
) -> tuple[str, bool, bool]:
    """``(target_device, is_agent_child, via_operator_binding)`` for one session.

    :func:`notice_origin` extended by the OPERATOR BINDING (2026-11-15), and,
    like it, the one reading BOTH legs use (:func:`stamp_rows_for_viewer` with
    the link's verdict, :func:`notice_forwarding` with the stored proof's) so the
    tool result and the catalogue cannot disagree.

    * A session with an origin other than this device goes to that origin —
      unchanged, whatever the binding says (a row co-member X placed here is
      X's alone).
    * A session STARTED here (no origin, or this device's own id) goes to the
      verified operator device when there is one (``network.operator_binding``:
      the operator's signed onboarding decision, checked against this node's
      root-owned anchor), else nowhere.
    * An agent's child goes nowhere new: it reports to that agent.
    """
    source, agent_child = notice_origin(origin)
    if source and source != self_device_id:
        return source, agent_child, False
    if agent_child or not operator_device_id:
        return "", agent_child, False
    return operator_device_id, False, True


def stamp_rows_for_viewer(
    root: Path,
    rows: Sequence[dict[str, Any]],
    viewer_device_id: str,
    *,
    operator_viewer: bool = False,
    self_device_id: str = "",
) -> None:
    """Add ``attention``/``notices`` to the rows whose ORIGIN is the viewer. In place.

    The caller (``RelayServer._op_catalog``) has already checked the viewer's
    capability; this decides per row. A row qualifies only when its stamp's
    ``origin.source_device`` equals ``viewer_device_id`` — equality on the
    authenticated link's device id, so a member cannot ask for another device's
    completions by naming it. One ``attention.db`` snapshot for every qualifying
    row (``state_many``) and one bounded inbox page per qualifying row; a device
    with no peer-originated sessions pays one dict scan and nothing else.

    ``operator_viewer`` (the relay's :func:`network.operator_binding.is_operator_link`
    verdict for this link) widens it to sessions STARTED on this device
    (:func:`notice_target`): their open asks — site cards included — and
    ``notify`` items reach the operator's device. Their COMPLETIONS do not: the
    request was "asks and notices", and a long-lived node session finishing a
    turn is not something the operator asked to be paged for.

    An AGENT'S CHILD (wherever it was placed from) puts only its site cards on
    the operator link's wire, and nothing on any other link's beyond what it
    did before (:func:`signed_card_target`).

    Never raises: a listing is not an error path, and a store that cannot be
    read leaves the rows exactly as the build before this one wrote them.
    """
    if not viewer_device_id:
        return
    # The caller checked the viewer's capability, so this origin collects
    # forwarded items: recorded for the node's ``notify`` sentence (m2).
    record_forwarding_origin(Path(root), viewer_device_id)
    operator = viewer_device_id if operator_viewer else ""
    via_operator: set[str] = set()
    cards_only: set[str] = set()
    mine = []
    for row in rows:
        target, _agent, bound = notice_target(
            row.get("origin"), self_device_id=self_device_id, operator_device_id=operator
        )
        if not str(row.get("session_id") or ""):
            continue
        if target != viewer_device_id:
            # An agent child placed by ANOTHER device (or started here) still
            # owes the operator its site cards (:func:`signed_card_target`);
            # nothing else of it — no completion badge — reaches this viewer.
            if signed_card_target(row.get("origin"), operator_device_id=operator) != (
                viewer_device_id
            ):
                continue
            cards_only.add(str(row["session_id"]))
        mine.append(row)
        if bound:
            via_operator.add(str(row["session_id"]))
    if not mine:
        return
    try:
        from local_operator.session.attention import AttentionStore

        identities = [f"session/{row['session_id']}" for row in mine]
        states = AttentionStore(Path(root) / "attention.db").state_many(identities)
    except Exception:  # noqa: BLE001 — a listing without attention is still a listing
        logger.debug("remote attention: completion read failed", exc_info=True)
        states = None
    store: Any = None
    try:
        from local_operator.notifications.store import NotificationStore

        store = NotificationStore(root)
    except Exception:  # noqa: BLE001
        logger.debug("remote attention: inbox open failed", exc_info=True)
    for row in mine:
        session_id = str(row["session_id"])
        state: dict[str, Any] = {}
        if session_id in via_operator or session_id in cards_only:
            # Notices only (see the docstring): no ``attention``, and no
            # ``attention_unknown`` either — the origin keeps no done row for a
            # session it never started, so there is nothing to protect.
            pass
        elif states is None:
            # A FAILED READ IS NOT "NOTHING UNSEEN" (review round 2, R2-m2): a
            # row with no ``attention`` is what an acknowledgement looks like,
            # and the origin settles its done row on it — losing the badge of a
            # completion nobody saw. Saying "unknown" makes the origin leave
            # its row alone until a read that could see the store.
            row["attention_unknown"] = True
        else:
            state = states.get(f"session/{session_id}") or {}
        token = state.get("completion_token")
        # ``notify`` False is the completion's own "do not tell anyone" (§14 of
        # the attention contract): honoured here exactly as the local legs do.
        if token and state.get("unseen") and state.get("notify", True):
            row["attention"] = {"token": str(token), "outcome": str(state.get("kind") or "")}
        if notice_origin(row["origin"])[1]:
            # AN AGENT'S CHILD NEVER NOTIFIES THE OPERATOR (2026-11-13): a
            # session another agent created here (a reviewer, QA or probe run
            # spawned with ``peer=``) reports to that agent, so its ``notify``
            # items stay in this device's inbox and are never put on the wire.
            # NARROW EXCEPTION (:func:`signed_card_target`): its site cards go
            # to the VERIFIED operator link, because only the operator can sign
            # one and the parent agent waiting on it cannot. Its ordinary asks
            # and notify items stay here, exactly as before.
            if signed_card_target(row.get("origin"), operator_device_id=operator) == (
                viewer_device_id
            ):
                cards = open_ask_notices(Path(root), session_id, signed_only=True)
                if cards is None:
                    row["asks_unknown"] = True
                elif cards:
                    row["notices"] = cards[:NOTICES_ON_WIRE]
            continue
        # Open asks FIRST: a question the agent is waiting on outranks an
        # informational notice for the bounded slots (``NOTICES_ON_WIRE``).
        asks = open_ask_notices(Path(root), session_id)
        if asks is None:
            # A FAILED ASK READ IS NOT "NO OPEN ASKS" (PR #3934 review R1-2),
            # the ``attention_unknown`` rule: the origin retires a forwarded ask
            # by its absence, so a missing ask here must say "unknown" or an
            # EIO would dismiss a question that is still owed — for good, since
            # the row's revision never changes and re-raising it is a no-op.
            row["asks_unknown"] = True
        notices = list(asks or [])
        # NOTIFY ITEMS ONLY FROM A NON-AGENT SESSION (:func:`is_non_agent_session`:
        # #4001's attestation for a stamped row, the local marker for one started
        # here); the open asks above travel either way.
        if store is not None and is_non_agent_session(
            Path(root), session_id, row.get("origin"), started_here=session_id in via_operator
        ):
            notices += _newest_notices(store, session_id)
        if notices:
            row["notices"] = notices[:NOTICES_ON_WIRE]


def _newest_notices(store: Any, session_id: str) -> list[dict[str, Any]]:
    """The newest unread, forwardable items of ``session_id``, as wire blocks.

    At most :data:`NOTICES_ON_WIRE`, newest content first (the store's
    ``updated_at DESC`` order), so a reader that cuts the list keeps the newest.
    """
    try:
        items, _cursor = store.list_items(state="unread", session_id=session_id, limit=_NOTICE_SCAN)
    except Exception:  # noqa: BLE001
        logger.debug("remote attention: inbox read failed for %s", session_id, exc_info=True)
        return []
    out: list[dict[str, Any]] = []
    for item in items:
        key = str(item.get("key") or "")
        if key.startswith(NOT_FORWARDED_PREFIXES):
            continue
        out.append(
            {
                # A keyless item is its own topic on the owner (every write is a new
                # row), so its id stands in for the key — the same "one row per
                # owner item" on the origin.
                "key": key or str(item.get("id") or ""),
                "revision": int(item.get("revision") or 1),
                "priority": str(item.get("priority") or "normal"),
                "title": str(item.get("title") or ""),
                # The item's OWN time (ms, this device's clock), so the origin
                # can refuse a backfill and keep the original timestamp rather
                # than stamping the sync moment (module doc, "NO BACKFILL").
                "at": int(item.get("updated_at") or 0),
            }
        )
        if len(out) >= NOTICES_ON_WIRE:
            break
    return out


def open_ask_notices(
    root: Path, session_id: str, *, signed_only: bool = False
) -> list[dict[str, Any]] | None:
    """``session_id``'s OPEN queued asks as ``notices`` wire blocks, oldest first.

    ``None`` — not ``[]`` — when the ask log could not be read: the caller
    stamps ``asks_unknown`` so the origin settles nothing on that listing.

    WHY (2026-11-13). A queued ask is not a ``NotificationStore`` row on the
    device it was asked on — the local inbox merges open asks from the ask
    index (``notifications/items.py``) — so :func:`_newest_notices` never saw
    one, and a session running on a mesh node asked its question where nobody
    was looking. Riding the EXISTING ``notices`` block (rather than a new wire
    field) means the origin's shipped watcher (:func:`raise_notice_item`, its
    ``high`` banner, the inbox row that deep-links to the session) carries it
    with no new parse, and an origin that predates this change still shows it.

    READ FROM DISK, not from the runtime: the session's own ``asks.jsonl``
    folded at "now", so an ask stays forwarded while the runtime is cold and a
    past-deadline ask folds to ``timed_out`` and drops out like an answered one.
    NOT through ``Session.notify``: the open-ask dedupe (``asks/dedupe.py``)
    would fold it into the ask and mark it read, and read items never travel.

    NO NEW AUTHORITY: the block is a fact (id + question text). Answering
    happens in the session the row opens, over the permissioned ``ask_respond``
    wire op (or the signed ``bridge_decide`` for a mirrored ask) — never here.
    Generic on purpose: any open ask in the queue (a browser-site question
    included) reaches the origin through this one function. Never raises.

    The title is the local inbox's own (``items.ask_item``: the first question,
    excerpted to the store's 120-char cap) and so is the priority: the wire
    sizing under ``NOTICES_ON_WIRE`` assumes bounded titles (review R1-4).
    """
    try:
        from local_operator.asks import store as ask_store
        from local_operator.notifications.items import (
            ASK_PRIORITY,
            excerpt,
            first_question,
        )

        events = ask_store.read_events(ask_store.session_dir(root, session_id), strict=True)
    except Exception:  # noqa: BLE001 — unknown, never "none" (R1-2)
        logger.debug("remote attention: ask read failed for %s", session_id, exc_info=True)
        return None
    try:
        if not events:
            return []
        records = ask_store.fold(events, ask_store.now_ms(), session_id=session_id)
        out: list[dict[str, Any]] = []
        for record in ask_store.outstanding_asks(records):
            # ``signed_only``: an agent child's leg (:func:`signed_card_target`)
            # carries its operator-signed cards and nothing else.
            if signed_only and not is_operator_signed_card(record):
                continue
            ask_id = str(record.get("ask_id") or "")
            title = excerpt(first_question(record), 120)
            if not ask_id or not title:
                continue
            out.append(
                {
                    "key": f"{ASK_NOTICE_PREFIX}{ask_id}",
                    # An ask's question does not change once queued (a
                    # revision rewrites the ANSWER), so one episode per ask.
                    "revision": 1,
                    "priority": ASK_PRIORITY,
                    "title": title,
                    # The ask's OWN queue time, like ``_newest_notices``'s
                    # ``at``: the origin drops a notice with no time (or one
                    # older than its first contact, "NO BACKFILL", PR #3902), so
                    # an ask block without it would never be raised there.
                    "at": int(record.get("created_at") or 0),
                }
            )
        return out[:NOTICES_ON_WIRE]
    except Exception:  # noqa: BLE001 — a listing whose asks did not fold says "unknown"
        logger.debug("remote attention: ask fold failed for %s", session_id, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Origin side: the rows
# ---------------------------------------------------------------------------


def _inbox_reads_here() -> bool:
    """The ``notify`` kill switch, asked the way ``remote_park`` asks it."""
    from local_operator.tools import builtin

    return builtin.notify_enabled()


def _location(device_label: str) -> str:
    """ "runs on <device>" — the park card's own floor sentence, reused (keyed copy)."""
    from local_operator.tui.notify import remote_session_location

    return remote_session_location(device_label)


def settle_done_item(root: Path, *, session_id: str, device_id: str, token: str) -> bool:
    """Mark the done row READ when its completion was acknowledged on the owner.

    READ, not dismissed: the operator saw the conversation (that is what an
    acknowledgement is), and a read row stays in the reel's history exactly as a
    locally-read one does. Bound to ``token``: a row that already describes a
    NEWER completion is left alone.
    """
    try:
        from local_operator.notifications.store import NotificationStore

        store = NotificationStore(root)
        row = store.get_by_key(session_id, done_key(device_id, session_id))
        if row is None or row["read"] or str(row.get("anchor_id") or "") != token:
            return False
        result: Any = store.apply(
            "read", ids=[(str(row["id"]), int(row["revision"]))], via="remote-attention"
        )
        return bool(result.changed)
    except Exception:  # noqa: BLE001
        logger.debug(
            "remote attention: could not settle done row for %s", session_id, exc_info=True
        )
        return False


def settle_unclaimed_done_items(
    root: Path, rows: Iterable[Any], *, listed_at_ms: int | None = None
) -> int:
    """Mark READ every unread done row the CURRENT listing no longer claims.

    QA round 1, Q1: the settle used to fire only when a key left the watcher's
    in-memory map, so a receipt that landed while no watcher ran (the TUI
    closed, ``lop serve`` restarting) left the row unread for good — a fresh
    process starts with an empty map and never sees the key leave. This reads
    the store's unread ``remote-done:`` rows (one query) and settles each whose
    session IS in this listing with no ``attention`` claim for the row's token:
    the owner acknowledged it. A session absent from the listing — a silent
    device, a deleted session — is left alone, because absence is not an
    acknowledgement. A row whose session claims a DIFFERENT token is also left:
    the same read raises that newer completion as an update. Returns how many
    rows were settled. Never raises.

    ONLY A LISTING NEWER THAN THE ROW MAY SETTLE IT (QA round 2, Q3).
    ``listed_at_ms`` is the wall-clock moment the listing's relay read STARTED
    (``peer_rows.listing_read_at_ms``); a row whose ``updated_at`` is not older
    than that was written from a later read than this one, so this listing's
    "no claim" predates the completion rather than acknowledging it. Without
    the check, ``lop serve``'s 20 s-cached listing, fetched just before a turn
    finished, marked the row the TUI had just raised READ ~5 s later. Both
    times are this device's own clock, so no skew between devices enters.
    ``None`` — a listing of unknown age — settles nothing. A row whose owner
    could not read its attention store (``remote_attention_unknown``) is left
    alone too (R2-m2).
    """
    if listed_at_ms is None:
        return 0
    try:
        listed: dict[str, str] = {}
        for row in rows:
            device_id = str(getattr(row, "owner_device", "") or "")
            session_id = str(getattr(row, "id", "") or "")
            if not device_id or not session_id:
                continue
            if getattr(row, "remote_attention_unknown", False):
                continue
            claim = getattr(row, "remote_attention", None)
            listed[done_key(device_id, session_id)] = str(claim[0]) if claim else ""
        if not listed:
            return 0
        from local_operator.notifications.store import NotificationStore

        store = NotificationStore(root)
        settled = 0
        for item in store.unread_with_key_prefix(f"{DONE_PREFIX}:"):
            key = str(item.get("key") or "")
            if key not in listed:
                continue
            if listed[key]:
                # Still claimed (this token, or a newer one): not acknowledged.
                continue
            if int(item.get("updated_at") or 0) >= int(listed_at_ms):
                # Written from a read newer than this listing (Q3, above).
                continue
            result: Any = store.apply(
                "read",
                ids=[(str(item["id"]), int(item["revision"]))],
                via="remote-attention",
            )
            settled += int(bool(result.changed))
        return settled
    except Exception:  # noqa: BLE001
        logger.debug("remote attention: done settle failed", exc_info=True)
        return 0


def settle_unlisted_ask_items(
    root: Path, rows: Iterable[Any], *, listed_at_ms: int | None = None
) -> int:
    """Retire (dismiss) every unread forwarded ASK row the listing no longer carries.

    A forwarded notice is never settled by its absence — the owner's unread
    notice just aged out of the bounded list — but an ask's absence IS its
    answer (or decline, dismissal, timeout): it is no longer owed, and the row
    must stop asking, the way a local answered ask leaves the inbox. Dismissed
    rather than read for that reason.

    Absence is trusted only when it can mean that: the session IS in this
    listing (a silent device or a deleted session settles nothing), the owner
    could read its ask log (``asks_unknown``, R1-2), its ``notices`` list is
    not full OF ASKS (asks are stamped first, so a full list that still carries
    any non-ask claim proves every open ask fit — and a busy session's list is
    full on every read, its notifies never being read on the node; R1-1), and
    the listing's read started after the row was last written
    (``listed_at_ms``; the done settle's Q3 rule — a cached listing older than
    the row predates the ask rather than answering it). ``None`` settles
    nothing. Returns how many rows were dismissed. Never raises.
    """
    if listed_at_ms is None:
        return 0
    try:
        listed: dict[tuple[str, str], set[str]] = {}
        for row in rows:
            device_id = str(getattr(row, "owner_device", "") or "")
            session_id = str(getattr(row, "id", "") or "")
            if not device_id or not session_id:
                continue
            if getattr(row, "remote_asks_unknown", False):
                continue
            claims = tuple(getattr(row, "remote_notices", None) or ())
            if len(claims) >= NOTICES_ON_WIRE and all(
                claim and str(claim[0]).startswith(ASK_NOTICE_PREFIX) for claim in claims
            ):
                continue
            listed[(device_id, session_id)] = {str(claim[0]) for claim in claims if claim}
        if not listed:
            return 0
        from local_operator.notifications.store import NotificationStore

        store = NotificationStore(root)
        settled = 0
        for item in store.unread_with_key_prefix(f"{NOTICE_PREFIX}:"):
            key = str(item.get("key") or "")
            infix = key.find(f":{ASK_NOTICE_PREFIX}")
            if infix < 0:
                continue
            # ``remote-notice:<device>:<session>:remote-ask:<id>`` — the
            # session is the row's own column, so the device is what precedes it.
            owner_key = key[infix + 1 :]
            session_id = str(item.get("session_id") or "")
            middle = key[len(NOTICE_PREFIX) + 1 : infix]
            if not session_id or not middle.endswith(f":{session_id}"):
                continue
            device_id = middle[: -(len(session_id) + 1)]
            still = listed.get((device_id, session_id))
            if still is None or owner_key in still:
                continue
            if int(item.get("updated_at") or 0) >= int(listed_at_ms):
                continue
            result: Any = store.apply(
                "dismiss",
                ids=[(str(item["id"]), int(item["revision"]))],
                via="remote-attention",
            )
            settled += int(bool(result.changed))
        return settled
    except Exception:  # noqa: BLE001
        logger.debug("remote attention: ask settle failed", exc_info=True)
        return 0


def raise_notice_item(
    root: Path,
    *,
    session_id: str,
    device_id: str,
    key: str,
    revision: int,
    priority: str,
    title: str,
    device_name: str = "",
    name: str = "",
    deliver: bool = True,
    at_ms: int = 0,
) -> str | None:
    """Mirror one owner ``notify`` item onto the origin's inbox.

    The title is the agent's own (already redacted by the owner's ``notify``
    before it was stored there); the body says where it ran. ``priority`` passes
    through — an unknown word degrades to ``normal`` rather than failing the
    write. A deliverable ``high`` write is handed to the item ladder off-thread
    (``deliver``), so it banners exactly once machine-wide.
    """
    if not session_id or not device_id or not key or not title:
        return None
    try:
        if not _inbox_reads_here():
            return None
        from local_operator.notifications.store import PRIORITIES, NotificationStore

        if priority not in PRIORITIES:
            priority = "normal"
        label = device_name or device_id
        store = NotificationStore(root)
        row_key = notice_key(device_id, session_id, key)
        existing = store.get_by_key(session_id, row_key)
        if existing is not None and _notice_rev(existing) >= int(revision):
            # THIS revision (or a newer one) is already the row's: a restarted
            # watcher, a second host, or a host reading a stale listing. Decided
            # on the revision BEFORE the write rather than left to the digest
            # (review round 1, m1): the digest also covers the "runs on
            # <device>" body, so a host that read a nameless row would turn a
            # repeat into an update and flip a READ row back to unread, or raise
            # a second ``high`` banner revision for one owner revision.
            return "unchanged"
        result = store.notify(
            session_id=session_id,
            title=title,
            body_md=_location(label),
            priority=priority,
            key=row_key,
            anchor_id=f"rev:{int(revision)}",
            session_title=name,
            episode=f"rev:{int(revision)}",
            # THE OWNER'S TIME, not the sync moment (2026-11-13): the row sorts
            # where the agent posted it. ``None`` (no ``at``) keeps "now".
            now=int(at_ms) if int(at_ms or 0) > 0 else None,
        )
        # An ``absorbed`` notice (the operator dismissed this row and the owner's
        # item was updated without escalating) STAYS dismissed: that is the
        # store's rule for a local keyed ``notify`` too, and a forwarded one must
        # not be harder to set aside than a local one. An escalation revives it
        # through the store itself.
        if deliver and priority == "high":
            _deliver_async(root, result, session_id=session_id, title=title, body=_location(label))
        return str(result.action)
    except Exception:  # noqa: BLE001
        logger.warning("remote attention: could not raise notice for %s", session_id, exc_info=True)
        return None


def _notice_rev(row: dict[str, Any]) -> int:
    """The owner revision a notice row describes (its ``rev:<N>`` anchor), else 0.

    0 for a row written before the anchor was recorded, so the first write
    after an upgrade still goes through the store's digest as before.
    """
    anchor = str(row.get("anchor_id") or "")
    if not anchor.startswith("rev:"):
        return 0
    try:
        return int(anchor[4:])
    except ValueError:
        return 0


def _deliver_async(root: Path, outcome: Any, *, session_id: str, title: str, body: str) -> None:
    """Hand a ``high`` revision to the item ladder WITHOUT blocking the watcher.

    ``deliver_item`` sleeps out the banner grace before it claims, so it runs on
    a daemon thread. ``route_item`` refuses a non-deliverable action itself
    (``unchanged`` from a second watcher never banners), and the store's
    per-revision claim makes the TUI's sweep, the desktop feed and this one
    raise at most one banner between them.
    """

    def run() -> None:
        try:
            from local_operator.notifications.delivery import deliver_item

            deliver_item(
                root,
                item_id=str(outcome.item_id),
                revision=int(outcome.revision),
                priority=str(outcome.priority),
                action=str(outcome.action),
                session_id=session_id,
                title=title,
                body=body,
            )
        except Exception:  # noqa: BLE001
            logger.debug("remote attention: banner delivery failed", exc_info=True)

    threading.Thread(target=run, name="remote-notice-banner", daemon=True).start()


# ---------------------------------------------------------------------------
# The watcher both hosts share
# ---------------------------------------------------------------------------


class RemoteAttentionWatcher:
    """Edge state for one host process; :meth:`observe` is the whole surface.

    ONE helper for both hosts (the TUI's park poll and ``lop serve``'s
    :func:`run_forever`), so they cannot drift. The maps are per process by
    design — the store's ``episode`` digest is what makes a fresh map (a restart,
    a second host) harmless.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._notice: dict[tuple[str, str, str], int] = {}

    def observe(
        self,
        rows: Iterable[Any],
        unanswered: Iterable[str] = (),
        *,
        listed_at_ms: int | None = None,
    ) -> None:
        """Apply one listing read's completion and notice edges. Never raises.

        ``listed_at_ms`` is when the listing's read started (wall clock, ms);
        omitted, it is looked up for ``rows`` from ``peer_rows``' cache, which
        answers only for the very tuple ``read_listing`` handed out — so both
        TUI callers and ``lop serve`` get it without threading a timestamp
        through the sidebar. Unknown means no settle (Q3).
        """
        try:
            from local_operator.session.peer_rows import (
                listing_read_at_ms,
                notice_edges,
            )

            if listed_at_ms is None:
                listed_at_ms = listing_read_at_ms(self.root, rows)
            rows = tuple(rows)
            silent = tuple(unanswered)
            # No completion edges: a remote turn-end is never an inbox row
            # (module doc, operator rule 2026-11-13). The settle stays — it
            # only marks an older build's ``remote-done:`` row read once the
            # owner acknowledged it, and never writes one.
            settle_unclaimed_done_items(self.root, rows, listed_at_ms=listed_at_ms)
            # Before raising: a row raised by THIS read is newer than the
            # listing, so the settle could not touch it anyway (Q3 rule).
            settle_unlisted_ask_items(self.root, rows, listed_at_ms=listed_at_ms)
            notices, notice_state = notice_edges(self._notice, rows, unanswered=silent)
            self._notice = notice_state
            # FIRST CONTACT is the listing, not the first notice (R1-1): every
            # owner device named here gets its baseline before any notice is
            # judged, so the first deliberate notice after contact is raised.
            baselines = origin_baselines_ms(
                self.root, (str(getattr(row, "owner_device", "") or "") for row in rows)
            )
            for notice in notices:
                baseline = baselines.get(notice.device_id)
                if baseline is None:
                    baseline = origin_baseline_ms(self.root, notice.device_id)
                if not notice.key.startswith(ASK_NOTICE_PREFIX) and (
                    notice.at_ms <= 0 or notice.at_ms < baseline
                ):
                    # NO BACKFILL: posted before this device first watched
                    # that one (or by an owner too old to say when). Skipped
                    # deterministically, so a restart skips it again.
                    #
                    # A FORWARDED OPEN ASK IS EXEMPT (QA round 3, Q3-1). The
                    # rule targets FINISHED notices — history the operator
                    # never asked to see; an ask in the listing is still
                    # waiting on the operator now, however long ago it was
                    # queued. Judged by the ask's ``created_at`` it was dropped
                    # forever whenever it predated the first listing (the
                    # baseline only ever moves earlier) or whenever the owner's
                    # clock ran behind this one — the time comparison mixes two
                    # clocks, so the exemption is also what makes asks
                    # skew-immune. Bounded without it: an ask rides the wire
                    # only while the owner lists it as outstanding, the settle
                    # above retires it once a fresh listing drops it, and
                    # ``raise_notice_item``'s revision check keeps one row per
                    # ask across watchers and restarts.
                    continue
                raise_notice_item(
                    self.root,
                    session_id=notice.session_id,
                    device_id=notice.device_id,
                    key=notice.key,
                    revision=notice.revision,
                    priority=notice.priority,
                    title=notice.title,
                    device_name=notice.device_name,
                    name=notice.name,
                    at_ms=notice.at_ms,
                )
        except Exception:  # noqa: BLE001 — a poll that cannot forward is still a poll
            logger.debug("remote attention watch failed", exc_info=True)


#: ``lop serve``'s poll cadence: the TUI park watch's (``REMOTE_PARK_POLL_S``),
#: over the same 20 s-cached listing, so serve adds no relay load the TUI did
#: not already pay — the two share ``peer_rows``' process-local cache only within
#: one process, so with both up the relay sees at most one listing per TTL each.
SERVE_POLL_S = 5.0


async def run_forever(root: Path, *, interval_s: float = SERVE_POLL_S) -> None:
    """The always-on host: ``lop serve`` runs this so no TUI has to be open.

    A device in no mesh short-circuits in ``read_listing`` (no relay record, no
    call), so every install outside a mesh pays one directory check per poll.
    Cancelled by the server's lifespan teardown.
    """
    from local_operator.session.peer_rows import read_listing

    watcher = RemoteAttentionWatcher(root)
    while True:
        try:
            rows, unanswered = await asyncio.to_thread(read_listing, Path(root))
            await asyncio.to_thread(watcher.observe, rows, [peer.device_id for peer in unanswered])
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.debug("remote attention serve poll failed", exc_info=True)
        await asyncio.sleep(interval_s)


# ---------------------------------------------------------------------------
# The node's notify sentence
# ---------------------------------------------------------------------------


#: Where an OWNER records the origins that have asked for these fields over a
#: ``REMOTE_ATTENTION_V1`` link: ``{device_id: first-seen epoch}`` directly
#: under the config root — NOT in the networks directory, whose every ``*.json``
#: is read (and a non-record quarantined) as a network record. WHY A FILE
#: (review round 1, m2): a link's capabilities live only in the relay process,
#: while the ``notify`` sentence is composed in the session process; the
#: sentence must promise forwarding only to an origin that has shown it
#: collects it. Written once per (process, origin) — a new
#: origin, not every catalogue read — so it costs no steady-state I/O.
ORIGINS_FILE = "remote-attention-origins.json"
_recorded_origins: set[tuple[str, str]] = set()
_origins_lock = threading.Lock()


def _origins_path(root: Path) -> Path:
    return Path(root) / ORIGINS_FILE


def record_forwarding_origin(root: Path, device_id: str) -> None:
    """Note that ``device_id`` reads forwarded completions/notices. Never raises."""
    if not device_id:
        return
    marker = (str(Path(root)), device_id)
    if marker in _recorded_origins:
        return
    try:
        import json
        import os
        import time

        with _origins_lock:
            path = _origins_path(root)
            known = forwarding_origins(root)
            if device_id not in known:
                known[device_id] = time.time()
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
                # 0600 like its siblings (``attention.db``, ``notifications.db``):
                # it names the mesh devices this one serves (review round 2,
                # R2-m3). Created with the mode rather than chmod-ed after, so
                # no umask-wide window exists; ``os.replace`` keeps the mode.
                fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(json.dumps(known, sort_keys=True))
                os.chmod(tmp, 0o600)
                os.replace(tmp, path)
            _recorded_origins.add(marker)
    except Exception:  # noqa: BLE001 — a marker is a hint for one sentence
        logger.debug("remote attention: could not record origin", exc_info=True)


def forwarding_origins(root: Path) -> dict[str, float]:
    """The recorded origins (see :data:`ORIGINS_FILE`), or ``{}``. Never raises."""
    try:
        import json

        data = json.loads(_origins_path(root).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — absent or unreadable is "none recorded"
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): float(v) for k, v in data.items() if isinstance(v, (int, float))}


#: The ORIGIN's record of when it first watched each owner device:
#: ``{device_id: first-seen epoch ms}``, directly under the config root beside
#: :data:`ORIGINS_FILE` (its owner-side twin) and for the same reason it is not
#: in the networks directory. WHY PERSISTED (2026-11-13): the backfill flood came
#: from a FIRST CONTACT treating every historic unread item as news; a baseline
#: kept in memory would be "first contact" again after every restart. Written
#: once per device, the first listing that names it; never moved later,
#: so reconnects and restarts compare against the same instant. The comparison
#: is against the OWNER's clock (``at``): skew between the two devices shifts the
#: cut-off by that skew, which only matters for a notice posted within seconds of
#: the very first contact.
BASELINES_FILE = "remote-attention-baselines.json"
_baselines_lock = threading.Lock()


def _valid_baseline(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def origin_baselines_ms(root: Path, device_ids: Iterable[str]) -> dict[str, int]:
    """This origin's first-contact instant (ms) for each of ``device_ids``.

    A device not yet recorded is recorded NOW. :meth:`RemoteAttentionWatcher.observe`
    calls this for every owner device in a listing BEFORE it looks at notices
    (review round 1, R1-1): recording lazily, on the first notice, stamped that
    notice's own observation instant as the baseline, so the first deliberate
    notice from a newly watched device was always older than its baseline and
    was dropped for good. Recorded at first contact, a notice posted after the
    first listing that named the device is raised; one posted before is the
    backfill this file exists to skip.

    CROSS-PROCESS (R1-4): the TUI and ``lop serve`` both run the watcher, so the
    read-merge-replace runs under an ``fcntl.flock`` on a sidecar lock file (the
    ``network/approvals.py`` pattern) as well as the in-process lock, and a
    device already in the file keeps its EARLIEST value — two writers can never
    move a baseline later and silently swallow the notices in between.

    Never raises: anything unreadable or unwritable answers "now" for the
    devices it could not settle, the conservative direction — a notice that
    cannot be shown to be new is not raised, which costs at worst a missed
    notice, never a flood.
    """
    import json
    import os
    import time

    wanted = sorted({str(d) for d in device_ids if d})
    now = int(time.time() * 1000)
    if not wanted:
        return {}
    try:
        import fcntl

        with _baselines_lock:
            path = Path(root) / BASELINES_FILE
            path.parent.mkdir(parents=True, exist_ok=True)
            lock_fd = os.open(path.with_name(f".{path.name}.lock"), os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                try:
                    try:
                        known = json.loads(path.read_text(encoding="utf-8"))
                    except (FileNotFoundError, ValueError):
                        # Absent, or unparseable (rewritten below rather than
                        # leaving every later poll to answer "now" forever).
                        known = {}
                    if not isinstance(known, dict):
                        known = {}
                    missing = [d for d in wanted if not _valid_baseline(known.get(d))]
                    if missing:
                        for device_id in missing:
                            known[device_id] = now
                        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
                        # 0600 like ``ORIGINS_FILE``: it names the mesh devices this one reads.
                        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                        with os.fdopen(fd, "w", encoding="utf-8") as handle:
                            handle.write(json.dumps(known, sort_keys=True))
                        os.replace(tmp, path)
                    return {d: int(known[d]) for d in wanted}
                finally:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)
    except Exception:  # noqa: BLE001 — see the docstring: unknown is "now"
        logger.debug("remote attention: baselines unavailable", exc_info=True)
        return {d: now for d in wanted}


def origin_baseline_ms(root: Path, device_id: str) -> int:
    """:func:`origin_baselines_ms` for one device (recording it now if new)."""
    import time

    return origin_baselines_ms(root, (device_id,)).get(device_id, int(time.time() * 1000))


#: :func:`notice_forwarding` verdicts. ``FORWARDED``: an origin known to collect
#: forwarded items will raise it. ``ORIGIN_NOT_COLLECTING``: the session has an
#: origin device (:func:`notice_origin`, any placement mode), but that origin is
#: not recorded as collecting. ``LOCAL_ONLY``: nothing will carry the item to any
#: other device — a session of an agent's child (its notices report to that
#: agent, never the wire, whatever the mesh), or one with no origin device on a
#: mesh node (started here). ``NOT_OPERATOR_STARTED``: the origin collects, but
#: the stamp lacks the operator attestation (:func:`origin_is_operator_started`),
#: so the owner's default-deny (#4001) keeps the item here. ``NOT_APPLICABLE``:
#: no clause at all (no mesh here, or a key this module never forwards).
FORWARDED = "forwarded"
ORIGIN_NOT_COLLECTING = "origin_not_collecting"
NOT_OPERATOR_STARTED = "not_operator_started"
LOCAL_ONLY = "local_only"
NOT_APPLICABLE = "not_applicable"


def notice_forwarding(root: Path, session_id: str, key: str = "") -> tuple[str, str]:
    """Will a ``notify`` from ``session_id`` reach another of the operator's devices?

    Returns ``(verdict, origin_device_id)``; the verdict is one of the module
    constants above and the origin is ``""`` unless the session carries one.

    THE ONE PLACE THIS IS DECIDED (2026-11-15). A ``notify`` always lands in
    THIS device's inbox; whether it travels further is the forwarding leg's
    business (:func:`stamp_rows_for_viewer` keeps only stamped, non-agent rows).
    The ``notify`` tool result must say which happened, or the model tells the
    operator they were told when the item never left this node — the bug this
    fixes: a session STARTED on a mesh node has no stamp, so the old sentence
    was empty and read as delivered. Keep every "will it be forwarded?" rule
    here so the surfaces cannot drift; the operator-binding work extends THIS
    function (e.g. ``FORWARDED`` only to a verified operator device, else
    ``LOCAL_ONLY``) rather than adding a second predicate beside it.

    A device in no mesh (no trusted network with another active member) gets
    ``NOT_APPLICABLE``: a single-device operator has nowhere else to be told,
    so their result stays byte-for-byte what it was. Never raises.

    THE OPERATOR BINDING (2026-11-15): a session STARTED here is ``FORWARDED``
    (to the operator's device id) exactly when the catalogue leg would put its
    notices on that device's link — a stored proof that still verifies against
    this node's root-owned anchor for an active member
    (``network.operator_binding.operator_device``), and that device recorded as
    collecting forwarded items, and no agent started the session
    (:func:`is_non_agent_session`, the catalogue leg's own predicate). Otherwise
    it stays ``LOCAL_ONLY``.
    """
    verdict, device, _via_operator = _notice_decision(root, session_id, key)
    return verdict, device


def _notice_decision(root: Path, session_id: str, key: str = "") -> tuple[str, str, bool]:
    """:func:`notice_forwarding` plus whether the target came from the operator binding."""
    try:
        if key and key.startswith(NOT_FORWARDED_PREFIXES):
            # Parks, gates and this module's own rows travel by their own
            # routes (or deliberately not at all); the clause is unchanged.
            return NOT_APPLICABLE, "", False
        from local_operator.session.placement import read_stamp

        stamp = read_stamp(Path(root), session_id)
        origin_map = stamp.origin if stamp is not None else None
        source, agent_child = notice_origin(origin_map)
        self_id = ""
        operator = ""
        if not agent_child:
            # Only a session with no foreign origin can ride the binding, so the
            # (cached) proof check runs only for those.
            self_id = _self_device_id(Path(root))
            if not source or source == self_id:
                from local_operator.network.operator_binding import operator_device

                operator, _network = operator_device(Path(root))
        # Decided exactly as the catalogue leg is (notice_target over
        # notice_origin), NOT on placement.mode: a promote_replica stamp is
        # mode="local" yet its notices are forwarded to the old owner (PR #4029
        # review R1-1).
        target, agent_child, via_operator = notice_target(
            origin_map, self_device_id=self_id, operator_device_id=operator
        )
        if agent_child:
            # Mirrors the drop in stamp_rows_for_viewer: an agent's child
            # reports to that agent, so nothing is put on the wire.
            return LOCAL_ONLY, source, False
        if target and not _is_active_peer(Path(root), target):
            # PR #4029 review R2-1: an origin (or bound device) that is no longer
            # an active member of an active, trusted network here has no link
            # that will carry the item, whatever the collector marker remembers.
            # Fall through to the honest node-only answer.
            target = ""
            via_operator = False
            source = ""
        if via_operator and not is_non_agent_session(
            Path(root), session_id, origin_map, started_here=True
        ):
            # Started here BY AN AGENT: its asks still reach the bound device,
            # its notify items stay here (the catalogue leg's own predicate).
            return LOCAL_ONLY, "", False
        if target:
            if target not in forwarding_origins(Path(root)):
                if via_operator:
                    # Bound, but that device has not read this node's catalogue
                    # with this build yet: nothing is carrying it today.
                    return LOCAL_ONLY, "", False
                # Main's precedence since #4001, kept byte for byte: an origin
                # that does not collect gets the not-collecting sentence whatever
                # the attestation (#4001's "brought home" test pins it).
                return ORIGIN_NOT_COLLECTING, target, False
            # For the bound leg (``via_operator``) this re-check is redundant — the
            # agent case already returned above — and always passes; it decides
            # only STAMPED rows (review R2-n1).
            if not is_non_agent_session(
                Path(root), session_id, origin_map, started_here=via_operator
            ):
                # Mirrors #4001's default-deny in stamp_rows_for_viewer: notify
                # items travel only for a stamp attesting ``started_by:
                # operator`` — the SAME predicate on the SAME mapping.
                return NOT_OPERATOR_STARTED, target, False
            return FORWARDED, target, via_operator
        if _is_mesh_node(Path(root)):
            # Started here (or no origin) and no verified operator device: the
            # forwarding leg drops the row, so the item stays on this node.
            return LOCAL_ONLY, "", False
        return NOT_APPLICABLE, "", False
    except Exception:  # noqa: BLE001 — a tool result never fails on this
        return NOT_APPLICABLE, "", False


def _self_device_id(root: Path) -> str:
    """This install's mesh id, or ``""``; never mints one."""
    from local_operator.network import identity

    try:
        mine = identity.load(root)
    except (OSError, ValueError):
        return ""
    return mine.device_id if mine is not None else ""


def _is_active_peer(root: Path, device_id: str) -> bool:
    """``device_id`` is an ACTIVE member of an ACTIVE (trusted) network here."""
    from local_operator.network import store as net_store

    for record in net_store.list_networks(root):
        if record.trust != "active" or record.is_burned(device_id):
            continue
        if any(m.device_id == device_id for m in record.active_members()):
            return True
    return False


def _is_mesh_node(root: Path) -> bool:
    """This device is in a trusted network that has at least one OTHER active member.

    Only then could the operator expect a notice on another device; a network
    created here that nobody has joined yet has no "other devices" to mention.
    """
    from local_operator.network import store as net_store

    for record in net_store.list_networks(root):
        if record.trust != "active":
            continue
        if any(m.device_id != record.self_device_id for m in record.active_members()):
            return True
    return False


def forwarded_notice_sentence(root: Path, session_id: str, key: str = "") -> str:
    """The clause a ``notify`` result adds on a mesh device, or ``""``.

    Model-facing English (a tool result, like the sentence it extends), so not
    catalogued. A bare "Notified" on the node read as if the operator had been
    told; the item is in THIS device's inbox and reaches the operator only when
    the origin forwards it, so the model is told which of the two happens.
    "Will be forwarded" is said ONLY for an origin recorded as collecting
    forwarded items (:func:`record_forwarding_origin`); any other origin — an
    older build, or one that has not read this device since updating — gets
    "it stays on <this device>" (review round 1, m2), and so does a session whose
    stamp lacks the operator attestation (:func:`origin_is_operator_started`, the
    owner's own forward filter — QA round 2, Q1). ``""`` for every session
    on a device in no mesh, and for a ``key`` this module never forwards — a
    single-device operator's result is unchanged byte for byte. A session on a
    mesh node whose item goes nowhere else (started here, or an agent's child)
    says so: "Notified" alone read as if the operator had been told (2026-11-15).
    The decision is :func:`notice_forwarding`'s; this only words it.
    """
    try:
        verdict, origin, via_operator = _notice_decision(Path(root), session_id, key)
        if verdict == NOT_APPLICABLE:
            return ""
        here, there = _device_names(Path(root), origin)
        if verdict == LOCAL_ONLY:
            return f" Shown on {here} only; it was not forwarded to your other devices."
        if verdict == NOT_OPERATOR_STARTED:
            # #4001's wording, byte for byte: the origin collects, so the
            # reason the item stays is the missing operator attestation.
            return (
                f" Posted on {here}; it stays on {here}, because only notices from"
                f" sessions the operator started on {there} are forwarded there, and"
                " this one was moved, forked or placed here otherwise."
            )
        if verdict == FORWARDED and via_operator:
            # A session started HERE, bound to the operator's verified device.
            return f" Posted on {here}; it will be forwarded to {there}, the operator's device."
        if verdict == FORWARDED:
            return (
                f" Posted on {here}; it will be forwarded to {there}, where this session started."
            )
        return (
            f" Posted on {here}; it stays on {here}, because {there}, where this session"
            " started, is not known to collect forwarded notices."
        )
    except Exception:  # noqa: BLE001
        return ""


def _device_names(root: Path, origin: str) -> tuple[str, str]:
    """``(this device's name, the origin's name)`` from the membership records.

    An origin that is no member here (it left, or was never in this network)
    reads "the device where this session started" rather than its raw device id
    (QA round 1, Q2): an id names nothing the operator would recognise.
    """
    here = "this device"
    there = "its origin device"
    try:
        from local_operator.network import store as net_store

        for record in net_store.list_networks(root):
            for member in record.active_members():
                if member.device_id == record.self_device_id and member.name:
                    here = member.name
                if member.device_id == origin and member.name:
                    there = member.name
    except Exception:  # noqa: BLE001
        pass
    return here, there
