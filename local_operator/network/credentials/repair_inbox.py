"""The owner's inbox item for an open credential repair — the ASKING half.

WHY THIS EXISTS (owner-relogin design, §0/§3). When a borrowed credential dies and
no refresh can save it, the one repair is a fresh sign-in ON THE OWNER. The
borrower already reads a sentence naming the owner and the command; the owner's
operator was never asked. ``credentials/repair.py`` derives "is a repair open, and
for which key" from the audit log, but its two surfaces — the ``credential_repair``
row of ``lop network doctor`` and the ``/network`` panel — are PULL surfaces: seen
only by somebody already looking. This module turns that same derivation into ONE
durable ``notifications.db`` item per open ``(network, key)``, the place the
operator already looks for "something needs me", with ``high`` priority on its
first raise so the existing banner ladder (``notifications/delivery.py``) tells
them once.

WHAT IT IS NOT. No wire frame, op or field; no signature or approval; no new store
(the item lives in ``notifications.db`` and its revival token comes from the audit
log); no token material anywhere — the item is built from a key NAME, device names
and the repair command. It grants nothing: the repair restores the owner's own
login, and the peer's next borrow is authorised by the placement row that already
authorised the last one.

WHERE IT RUNS, AND WHAT IT COSTS IDLE. :func:`tick` is called from the credentials
cadence (``sync.credentials_sync_step``), once per member per 15 s tick, NEVER from
the borrow path: ``notify`` is a SQLite write that can raise, and a borrow must not
gain a way to fail. The cost is bounded in three layers: an ``os.stat`` doorbell on
the live audit file (unchanged → nothing else happens, so N members cost N stats);
a byte scan of only the APPENDED tail for the two event names the derivation folds
(other traffic — placement pulls, sync rows — rings the doorbell but never pays for
a derivation, and grant rows, which every successful borrow writes, pay for one only
while an item is open for a grant to close); and an in-process memo of the last
published plan and of every item's raised text, so a derivation that changes nothing
touches no store. Measured on the two-relay rig: 20 idle ticks cost 20 stats, no
scan, no derivation and no store write; one derivation of a full 8.6 MB audit file
costs 75-110 ms of CPU, which is what the scan exists to avoid paying per tick.

COPY. The title and body are built ONLY from the derivation's own sentences —
``RepairNotice.remedy`` (the doctor row's remedy) and ``RepairNotice.sentence``
(``messages.render_repair_notice``) — so the doctor row, the panel and the item
cannot drift, and no new English enters the product. The overflow mark is a bare
ellipsis for the same reason; the session label is the ``/network`` command, a
technical identifier, which is where the full repair rows are.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping

from local_operator.network.credentials.repair import (
    REPAIR_TAIL_ROWS,
    OwnerLogins,
    RepairNotice,
    repair_notices,
    repaired_keys,
)

if TYPE_CHECKING:
    from local_operator.network.relay import RelayServer

logger = logging.getLogger(__name__)

#: The owner-scoped pseudo-session every repair item is filed under. Not a
#: conversation: no session directory exists, so the reel falls back to
#: :data:`SESSION_TITLE` (``items.SessionFacts.get``) and offers only read/dismiss.
SESSION_ID = "network-credentials"

#: The label the reel shows for :data:`SESSION_ID` — the in-app command that shows
#: the full repair rows. A command is a technical identifier (exempt from the copy
#: catalogue by AGENTS.md's i18n rule), so the label adds no untranslated prose.
SESSION_TITLE = "/network"

#: The item key's namespace: ``mesh-repair:<network>:<credential key>:<generation>``.
KEY_PREFIX = "mesh-repair"

#: ``dismissed_via`` for an item THIS module closed because the window shows the
#: repair (a later ``credential.grant``). The store's column is free text; the
#: value is what distinguishes "repaired" from "the operator dismissed it".
DISMISSED_VIA = "mesh_repair"

#: How many askers the body names, sorted by name. More are marked by a bare
#: ellipsis rather than an "and N more" sentence (no new copy, see the docstring);
#: the action — the title — is the same however many devices asked.
NAMED_ASKERS_MAX = 3
OVERFLOW_MARK = "…"

#: The byte spellings of the two events :func:`repair.open_reports` folds. The
#: audit writer renders ``"event": "<name>"`` with ``json.dumps`` defaults, so a
#: row carrying either event contains one of these exactly.
_REPORT_EVENT = b'"credential.report"'
_GRANT_EVENT = b'"credential.grant"'

#: How many standing items one pass will page through. One item per open key, so
#: a real owner holds a handful; the bound only stops a damaged store from
#: turning a tick into an unbounded scan.
_STANDING_PAGES_MAX = 5


def item_key(network_id: str, key: str, generation: str) -> str:
    """The item's collapse key: one row per ``(network, credential key, episode)``."""
    return f"{KEY_PREFIX}:{network_id}:{key}:{generation}"


def _generation_rank(generation: str) -> int:
    """``g<seq>`` as an orderable int; anything else ranks below every real token."""
    digits = generation[1:] if generation.startswith("g") else ""
    return int(digits) if digits.isdigit() else -1


def _split_item_key(value: str) -> tuple[str, str, str] | None:
    """``(network, credential key, generation)`` back out of an item key, or ``None``.

    The credential key may itself contain ``:`` (``mcp:https://…``), so the network
    is split off the front and the generation off the back.
    """
    if not value.startswith(KEY_PREFIX + ":"):
        return None
    rest = value[len(KEY_PREFIX) + 1 :]
    network_id, _, tail = rest.partition(":")
    key, _, generation = tail.rpartition(":")
    if not network_id or not key or _generation_rank(generation) < 0:
        return None
    return network_id, key, generation


@dataclass(frozen=True)
class PlannedItem:
    """One item this device's operator should be holding, fully rendered."""

    network_id: str
    key: str
    generation: str
    title: str
    body: str


def _server_host(notice: RepairNotice) -> str:
    """The MCP server's host, for a key this device cannot NAME (#3522 D3/U5).

    ``/mcp login <name>`` is a form, not a command, when the local config no longer
    maps the URL to a name — and then nothing on the item said WHICH server died,
    so two such keys rendered as two identical rows. The host is a technical
    identifier (no catalogue key); a named login needs nothing added.
    """
    if not notice.key.startswith("mcp:") or notice.login_name:
        return ""
    from urllib.parse import urlsplit

    from local_operator.network.credentials.types import mcp_url_from_key

    try:
        return urlsplit(mcp_url_from_key(notice.key)).hostname or ""
    except ValueError:
        return ""


def _display_url(url: str) -> str:
    """``scheme://host[:port]/path`` — the server, without anything that can be a secret.

    THE BODY CAN REACH THE LOCK SCREEN (#3522 follow-up, review m2): it is the one
    field the banner ladder shows when the privacy flag allows it, and some MCP
    endpoints carry an access token in the query string or credentials in the
    userinfo. The operator needs WHICH server, which scheme, host and path say;
    userinfo, query and fragment are dropped. The item KEY keeps the full URL (it
    is the collapse identity and is never rendered on a banner), as does the
    doctor row — both out of this function's scope.
    """
    from urllib.parse import urlsplit

    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
        port = parts.port
    except ValueError:
        # An unparseable URL (a bad port, a broken IPv6 literal) shows nothing
        # rather than its raw text — the raw text is what may hold a secret.
        return ""
    if not host:
        return ""
    if ":" in host:
        host = f"[{host}]"  # an IPv6 literal; ``hostname`` strips the brackets
    if port is not None:
        host = f"{host}:{port}"
    return f"{parts.scheme}://{host}{parts.path}"


def render_title(notice: RepairNotice) -> str:
    """The doctor row's remedy, as a headline: the action, the same for every asker.

    QUOTED AS THE BODY QUOTES IT (#3522 D2/N1): the remedy wraps the command in
    backticks for the CLI, but a title is drawn as plain text — in the reel, the
    pane header and the OS banner — so the backticks showed literally beside a
    body that says ``'/login openai'``. Same words, the body's quoting.

    THE HOST LEADS WHEN THERE IS ONE (#3522 round 2, D7). An unnamed MCP key put
    its host after "on this device", where both the 140- and the 80-column list
    truncate it — two such rows read identically. First, it survives the cut and
    the rows are distinguishable; the action still reads in full wherever it fits.
    """
    remedy = notice.remedy.replace("`", "'")
    host = _server_host(notice)
    if host:
        return f"{host}: {remedy}"
    return remedy[:1].upper() + remedy[1:]


def render_body(askers: Iterable[tuple[str, str]], notice: RepairNotice) -> str:
    """One producer sentence per asker, sorted by name, capped — as a Markdown list.

    A LIST, because the pane collapses a blank line and two wrapped sentences then
    read as one block (#3522 D5/N2); the sentences themselves stay byte-identical
    to ``render_repair_notice``. An MCP key with no local name gets its URL as the
    last list item (see :func:`_server_host`), REDACTED to scheme, host and path
    (:func:`_display_url`) and PLAIN — the body is rendered as
    Markdown but the reel is not, so a backtick-wrapped URL reached the operator
    with its formatting characters showing (#3522 round 2, D6/U7).

    STABLE FACTS ONLY — device names and the key's command, no timestamps and no
    counters — so a retry renders byte-identical text, the store's content hash
    answers ``unchanged``, and nothing is written.
    """
    labels = sorted({name or device for device, name in askers}, key=lambda s: (s.casefold(), s))
    lines = [f"- {notice.sentence(label)}" for label in labels[:NAMED_ASKERS_MAX]]
    if len(labels) > NAMED_ASKERS_MAX:
        lines.append(f"- {OVERFLOW_MARK}")
    if _server_host(notice):
        from local_operator.network.credentials.types import mcp_url_from_key

        shown = _display_url(mcp_url_from_key(notice.key))
        if shown:
            lines.append(f"- {shown}")
    return "\n".join(lines)


def plan_items(
    root: Path, records: list[Mapping[str, Any]], *, logins: OwnerLogins | None = None
) -> dict[tuple[str, str], PlannedItem]:
    """Every item this device should hold now, keyed ``(network, credential key)``.

    A STRONGER BAR THAN THE DOCTOR ROW, deliberately: a pull surface may list any
    open report, but a push that asks the operator to sign in must be about a login
    THIS device owns (its placement entry names it as owner) and must name only
    devices that still hold the share. A key owned elsewhere, or a report whose
    every asker has since lost the share, raises nothing; the doctor row is not
    touched by this filter.
    """
    from local_operator.network import store
    from local_operator.network.credentials.placement import PlacementDocument

    plan: dict[tuple[str, str], PlannedItem] = {}
    for record in store.list_networks(root):
        notices = repair_notices(record, records, logins=logins)
        if not notices:
            continue
        document = PlacementDocument.load(
            record.network_id, root, self_device=record.self_device_id
        )
        for notice in notices:
            entry = document.entry(notice.key)
            if entry is None or entry.owner_device != record.self_device_id:
                continue
            askers = [(device, name) for device, name in notice.askers if entry.is_holder(device)]
            if not askers:
                continue
            plan[(notice.network_id, notice.key)] = PlannedItem(
                network_id=notice.network_id,
                key=notice.key,
                generation=notice.generation,
                title=render_title(notice),
                body=render_body(askers, notice),
            )
    return plan


@dataclass
class PublishResult:
    """What one publication pass did, for the caller's log and the tests."""

    created: int = 0
    updated: int = 0
    unchanged: int = 0
    reopened: int = 0
    closed: int = 0
    #: ``""`` on a completed pass; else why it stopped (``busy``/``refused``/``failed``).
    stopped: str = ""
    #: The exception's type and text when ``stopped == "failed"``.
    error: str = ""


def publish_notices(
    config_dir: Path | str,
    plan: Mapping[tuple[str, str], PlannedItem],
    repaired: Iterable[str],
    *,
    episodes: dict[tuple[str, str], tuple[str, str, str]] | None = None,
) -> PublishResult:
    """Make ``notifications.db`` hold exactly ``plan``'s items. NEVER RAISES.

    ``repaired`` is :func:`repair.repaired_keys` — the keys the window shows
    repaired; a standing item for one of them is dismissed with
    ``dismissed_via=mesh_repair``, leaving the unread reel and staying readable
    under ``state="dismissed"`` as the record. An item for a key that merely left
    the window is left alone: closing needs positive evidence.

    ``episodes`` is the caller's memory, per key with a live item, of the
    ``(generation, title, body)`` it last raised (``""`` text for an item a previous
    process raised). Two jobs. An item already raised with exactly this text is
    not handed to the store again — ``notify`` takes a write transaction even to
    answer ``unchanged``. And GENERATIONS ONLY MOVE FORWARD while a key stays
    open: the token is the seq of the grant before the episode, and that grant can
    age out of the 500-row window while the episode is still open, which would
    otherwise read as ``g0`` — a NEW item for an old episode, a second banner, and
    the first item stranded. A token lower than the one already raised (in ``episodes`` or on a
    standing row) is that aging, never a new episode, because a grant cannot
    un-happen; a HIGHER token is a real repair-then-break, and the older standing
    row is closed as repaired.

    ``NotifyRefused`` (the store's hourly new-item cap) and
    ``NotificationStoreBusy`` stop the pass with a named outcome; anything else is
    logged and swallowed. A cadence reports; it never raises into the tick.
    """
    from local_operator.notifications.store import (
        NotificationStore,
        NotificationStoreBusy,
        NotifyRefused,
    )

    result = PublishResult()
    episodes = episodes if episodes is not None else {}
    repaired_set = set(repaired)
    try:
        notifications = NotificationStore(config_dir)
        standing = _standing_items(notifications)
        # THE FLOOR ALSO COUNTS DISMISSED ROWS (#3522 round 1, M1). Without them a
        # restart with the episode's grant aged out of the window read ``g0``, found
        # no live row (the operator had dismissed it) and raised a NEW high item and
        # a banner for an episode they had already said they know about. With the
        # dismissed row's token as the floor, the raise lands on that row: same
        # content → ``unchanged``, new asker → ``absorbed`` silently (contract §1.3).
        dismissed = (
            _standing_items(notifications, state="dismissed") if (plan or repaired_set) else {}
        )
        for (network_id, key), item in plan.items():
            rows = standing.get((network_id, key), [])
            generation = item.generation
            known = [gen for gen, _ in rows]
            known.extend(gen for gen, _ in dismissed.get((network_id, key), []))
            remembered = episodes.get((network_id, key))
            if remembered is not None:
                known.append(remembered[0])
            newest = max(known, key=_generation_rank, default="")
            if newest and _generation_rank(generation) < _generation_rank(newest):
                generation = newest
            for gen, row in rows:
                if _generation_rank(gen) < _generation_rank(generation):
                    result.closed += _close(notifications, row)
            if remembered == (generation, item.title, item.body):
                result.unchanged += 1
                continue
            action = _raise(notifications, item, generation)
            episodes[(network_id, key)] = (generation, item.title, item.body)
            if action == "created":
                result.created += 1
            elif action == "reopened":
                result.reopened += 1
            elif action in ("updated", "revived"):
                result.updated += 1
            else:
                result.unchanged += 1
        for (network_id, key), rows in standing.items():
            if (network_id, key) in plan:
                continue
            if key in repaired_set:
                for _, row in rows:
                    result.closed += _close(notifications, row)
                episodes.pop((network_id, key), None)
            else:
                # Standing but no longer open in the window (it aged out, or a
                # previous process raised it): remembered, so that the grant which
                # eventually repairs it still wakes a derivation (see ``_tick``).
                newest = max((gen for gen, _ in rows), key=_generation_rank)
                episodes.setdefault((network_id, key), (newest, "", ""))
        # A RENEWAL REPAIRS AN OPERATOR-DISMISSED ROW TOO (#3522 round 2, review m1).
        # Without this, "operator dismisses → signs in → the same login breaks again"
        # raised nothing when no peer borrowed in between: the dismissed row still
        # held the current generation, so the new episode's raise landed on it and
        # was absorbed silently, while the doctor row showed the break. Marking the
        # row REPAIRED (once the owner's own sign-in proves it was) makes the next
        # break a new episode, which reopens the row at ``high``.
        for episode_key, rows in dismissed.items():
            if episode_key[1] not in repaired_set:
                continue
            for _, row in rows:
                result.closed += _mark_repaired(notifications, row)
            episodes.pop(episode_key, None)
        # Nothing to close and nothing open: forgotten, so a lingering key cannot keep
        # every later grant row paying for a derivation.
        for stale in [k for k in episodes if k not in plan and k not in standing]:
            episodes.pop(stale, None)
    except NotifyRefused:
        logger.debug("credential repair inbox: the store refused a new item (hourly cap)")
        result.stopped = "refused"
    except NotificationStoreBusy:
        logger.debug("credential repair inbox: the store was busy; retrying next tick")
        result.stopped = "busy"
    except Exception as exc:  # noqa: BLE001 — a lost item must never cost a tick or a borrow
        # Debug here: the cadence says it ONCE per failure episode (``_failed``).
        logger.debug("credential repair inbox: publication failed", exc_info=True)
        result.stopped = "failed"
        result.error = f"{type(exc).__name__}: {exc}"
    return result


def _standing_items(
    notifications: Any, state: str = "all"
) -> dict[tuple[str, str], list[tuple[str, dict[str, Any]]]]:
    """The repair items in ``state``, grouped ``(network, key) → [(generation, row)]``.

    ``all`` is the NOT-dismissed rows (the store's own meaning); ``dismissed`` is
    read only for the generation floor in :func:`publish_notices`.
    """
    grouped: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]] = {}
    cursor: str | None = None
    for _ in range(_STANDING_PAGES_MAX):
        rows, cursor = notifications.list_items(
            state=state, session_id=SESSION_ID, cursor=cursor, limit=100
        )
        for row in rows:
            parts = _split_item_key(str(row.get("key") or ""))
            if parts is None:
                continue
            network_id, key, generation = parts
            grouped.setdefault((network_id, key), []).append((generation, row))
        if cursor is None:
            break
    return grouped


def _close(notifications: Any, row: Mapping[str, Any]) -> int:
    """Dismiss one standing item as REPAIRED, token-bound (``store.apply``).

    The ``(id, revision)`` pair is the store's optimistic token, so an operator's
    concurrent read or dismiss is never overwritten blindly: a stale token is
    skipped, and the next pass sees the row's new state.
    """
    applied = notifications.apply(
        "dismiss", ids=[(str(row["id"]), int(row["revision"]))], via=DISMISSED_VIA
    )
    return len(applied.changed)


def _mark_repaired(notifications: Any, row: Mapping[str, Any]) -> int:
    """Re-file a DISMISSED row as repaired (``dismissed_via=mesh_repair``); 0 if already.

    ONE token-bound store write (``refile_dismissal``), never an undismiss and a
    dismiss: between those two commits the row was live and unread, and a row whose
    first revision was never bannered was claimable there — a banner for a login
    that had just been repaired (#3522 follow-up, review m1 = QA Q4). A row this
    module already closed needs nothing.
    """
    if str(row.get("dismissed_via") or "") == DISMISSED_VIA:
        return 0
    refiled = notifications.refile_dismissal(str(row["id"]), int(row["revision"]), DISMISSED_VIA)
    return 1 if refiled else 0


def _raise(notifications: Any, item: PlannedItem, generation: str) -> str:
    """Create or update one item; ``high`` only when the operator has not been told.

    A FIRST raise is ``high`` — the banner ladder's one banner for it. A later
    revision (a second asker joined) is ``normal``: the reel and the badge carry
    it, and the operator is not bannered again for the same dead login. A retry
    with the same text keeps the row's own priority, so the content hash matches
    and the store writes nothing.
    """
    key = item_key(item.network_id, item.key, generation)
    existing = notifications.get_by_key(SESSION_ID, key)
    repaired_before = existing is not None and (
        existing.get("dismissed") and str(existing.get("dismissed_via") or "") == DISMISSED_VIA
    )
    if existing is None or repaired_before:
        priority = "high"
    elif existing.get("title") == item.title and existing.get("body_md") == item.body:
        priority = str(existing.get("priority") or "normal")
    else:
        priority = "normal"
    outcome = notifications.notify(
        SESSION_ID,
        title=item.title,
        body_md=item.body,
        priority=priority,
        key=key,
        session_title=SESSION_TITLE,
    )
    if repaired_before and outcome.action == "absorbed":
        # A key this module closed as repaired is open again under the SAME token
        # (only possible for ``g0``, once the closing grant has itself left the
        # window): a new episode, so the row comes back — undismissed, unread AND
        # bannerable, in one transaction (``store.reopen``). The revision is the
        # one the previous episode's banner already claimed, so an undismiss alone
        # left a live row no banner leg would raise (#3522 follow-up, QA Q3).
        notifications.reopen(outcome.item_id, outcome.revision)
        return "reopened"
    return str(outcome.action)


# ---------------------------------------------------------------------------
# The cadence: doorbell, memo, gates


@dataclass
class _Doorbell:
    """One audit file's last-seen shape, the last plan published, and the episodes."""

    inode: int = -1
    offset: int = -1
    #: The plan (and repaired set) the store was last brought in line with; a
    #: derivation that reproduces it writes nothing.
    published: tuple[Any, ...] | None = None
    episodes: dict[tuple[str, str], tuple[str, str, str]] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    #: A derivation is OWED: a pass consumed the doorbell's change and did not finish
    #: (gated, busy, refused, failed). Without this the offset had already moved past
    #: the closing grant and a repair made while notifications were off stayed open
    #: (#3522 round 1, m1/Q1). Cleared only by a completed publication.
    pending: bool = False
    #: ``auth.db``/``-wal`` stat fingerprint taken before the last derivation, kept
    #: only while an item is open: the owner's own sign-in writes no audit row (U1).
    login_fp: tuple[Any, ...] | None = None
    #: Consecutive failed passes, and the monotonic time before which none is retried.
    failures: int = 0
    retry_at: float = 0.0


#: One doorbell per audit file (two relays in one process — the test rig — are two
#: files), created on first use and never dropped: one small object per relay root.
_DOORBELLS: dict[str, _Doorbell] = {}
_DOORBELLS_LOCK = threading.Lock()

#: Counters the idle-cost evidence reads (stats taken, derivations run, passes that
#: touched the store). Process-wide and monotonic; tests diff them.
COUNTERS: dict[str, int] = {
    "stats": 0,
    "scans": 0,
    "login_stats": 0,
    "derivations": 0,
    "publications": 0,
}

#: Backoff for a pass that keeps failing (a corrupt ``notifications.db``, a
#: regression in the derivation): 15 s, doubling, capped at 15 min. Before this a
#: failing store re-derived (75-110 ms) and logged a traceback on every member call
#: of every tick (#3522 round 1, m2). Contention (``busy``) is not a failure.
BACKOFF_BASE_S = 15.0
BACKOFF_MAX_S = 900.0


def _doorbell_for(path: Path) -> _Doorbell:
    with _DOORBELLS_LOCK:
        return _DOORBELLS.setdefault(str(path), _Doorbell())


def _relevant_change(path: Path, bell: _Doorbell) -> str:
    """What the audit file gained since the last pass: ``none``, ``grant``, ``report``, ``full``.

    One ``stat`` when nothing changed. When it grew, read ONLY the appended bytes and
    look for the two folded event names — a relay writes many other rows (sync,
    placement, links), and none of them can move a repair. ``grant`` means grant rows
    and no report row: those can only CLOSE an item, so the caller derives on them
    only while something is open. A new inode, a shrunk file or a first pass is
    ``full`` (everything is re-derived); a missing file is ``none``. The offset only
    advances past complete lines, so a row the writer is midway through is read whole
    next time.
    """
    COUNTERS["stats"] += 1
    try:
        stat = os.stat(path)
    except OSError:
        return "none"
    if stat.st_ino == bell.inode and stat.st_size == bell.offset:
        return "none"
    if stat.st_ino != bell.inode or stat.st_size < bell.offset or bell.offset < 0:
        # First pass, a rotation or a re-arm: derive from scratch, and resume the
        # cheap scan from the last COMPLETE line the derivation could have read.
        bell.inode, bell.offset = stat.st_ino, _last_line_end(path, stat.st_size)
        return "full"
    COUNTERS["scans"] += 1
    with path.open("rb") as handle:
        handle.seek(bell.offset)
        chunk = handle.read(stat.st_size - bell.offset)
    complete = chunk.rfind(b"\n") + 1
    if not complete:
        return "none"
    appended = chunk[:complete]
    bell.offset += complete
    if _REPORT_EVENT in appended:
        return "report"
    return "grant" if _GRANT_EVENT in appended else "none"


def _last_line_end(path: Path, size: int) -> int:
    """The offset just past the last newline at or before ``size`` (0 for none).

    Audit rows are small (``MAX_DETAIL_BYTES`` caps the variable part), so the last
    64 KiB always holds a line end on a file that has one; a writer's half-appended
    batch past it is scanned whole on the next pass.
    """
    if size <= 0:
        return 0
    base = max(0, size - 65536)
    try:
        with path.open("rb") as handle:
            handle.seek(base)
            chunk = handle.read(size - base)
    except OSError:
        return 0
    end = chunk.rfind(b"\n")
    return base + end + 1 if end >= 0 else base


def _login_fingerprint() -> tuple[Any, ...]:
    """``(mtime_ns, size)`` of this device's ``auth.db`` and its ``-wal``.

    The owner's own ``/login`` or ``/mcp login`` writes the auth store and no audit
    row, so the audit doorbell alone could not see the one action the item asks for
    (#3522 round 1, U1). Two stats, taken only while an item is open. Any other
    write to the store (a routine token refresh) also moves it; that costs one
    derivation on the next tick, only while an item is open.
    """
    from local_operator.providers.auth_store import default_db_path

    out: list[Any] = []
    for path in (default_db_path(), Path(f"{default_db_path()}-wal")):
        COUNTERS["login_stats"] += 1
        try:
            stat = os.stat(path)
            out.append((stat.st_mtime_ns, stat.st_size))
        except OSError:
            out.append(None)
    return tuple(out)


def _owe(bell: _Doorbell) -> None:
    bell.pending = True
    bell.published = None


def _failed(bell: _Doorbell, what: str, error: str = "") -> str:
    """Record a failed pass: owed, backed off, and said ONCE per failure episode."""
    _owe(bell)
    bell.failures += 1
    delay = min(BACKOFF_BASE_S * 2 ** (bell.failures - 1), BACKOFF_MAX_S)
    bell.retry_at = time.monotonic() + delay
    if bell.failures == 1:
        logger.warning(
            "credential repair inbox: %s (%s); retrying with backoff up to %.0f s",
            what,
            error or "no detail",
            BACKOFF_MAX_S,
        )
    else:
        logger.debug("credential repair inbox: %s again (%s)", what, error)
    return "failed"


def tick(server: "RelayServer") -> str:
    """One pass of the owner's asking channel, for the credentials cadence. NEVER RAISES.

    Answers a word for the debug log: ``idle`` (nothing to do), ``gated``
    (notifications are off, or this process is a rig under a foreign HOME — the
    banner ladder's own two refusals, ``delivery.gates_refuse``), ``backoff``,
    ``same`` (re-derived, nothing to change), ``published`` or a stop reason.
    """
    try:
        return _tick(server)
    except Exception:  # noqa: BLE001 — a cadence reports; it never raises into the tick
        # Only reachable before a doorbell exists (resolving the server's paths);
        # everything after is accounted for, owed and backed off inside ``_tick``.
        logger.debug("credential repair inbox: tick failed", exc_info=True)
        return "failed"


def _tick(server: "RelayServer") -> str:
    from local_operator.network.store import audit_path

    log = getattr(server, "audit", None)
    if log is None:
        return "no_audit"
    root = Path(server.root)
    path = audit_path(root)
    bell = _doorbell_for(path)
    if not bell.lock.acquire(blocking=False):
        # Another member's step in this tick is already deriving; it covers this one.
        return "busy"
    try:
        return _locked_tick(log, root, path, bell)
    except Exception as exc:  # noqa: BLE001 — owed and backed off, never raised
        logger.debug("credential repair inbox: pass raised", exc_info=True)
        return _failed(bell, "a pass raised", f"{type(exc).__name__}: {exc}")
    finally:
        bell.lock.release()


def _locked_tick(log: Any, root: Path, path: Path, bell: _Doorbell) -> str:
    change = _relevant_change(path, bell)
    login_moved = False
    if bell.episodes and bell.login_fp is not None:
        login_moved = _login_fingerprint() != bell.login_fp
    wanted = (
        bell.pending
        or login_moved
        or change in ("report", "full")
        or (change == "grant" and bool(bell.episodes))
    )
    if not wanted:
        # Nothing new, or only grants while no item is open (a grant can only close).
        return "idle"
    if bell.failures and time.monotonic() < bell.retry_at:
        _owe(bell)
        return "backoff"
    from local_operator.notifications.delivery import gates_refuse

    if gates_refuse():
        # OWED, not dropped (#3522 round 1, m1/Q1): the doorbell has already moved
        # past whatever rang it, so the pass that finds the gate open again must
        # derive from scratch. While owed, a tick costs this gate check and no
        # derivation; the doctor row never depended on any of this.
        _owe(bell)
        return "gated"
    fingerprint_before = _login_fingerprint()
    records = log.tail(REPAIR_TAIL_ROWS)
    COUNTERS["derivations"] += 1
    logins = OwnerLogins()
    try:
        plan = plan_items(root, records, logins=logins)
        repaired = repaired_keys(records, logins=logins)
    finally:
        logins.close()
    fingerprint = (tuple(sorted(plan.items())), tuple(sorted(repaired)))
    if fingerprint != bell.published:
        COUNTERS["publications"] += 1
        result = publish_notices(root, plan, repaired, episodes=bell.episodes)
        if result.stopped == "busy":
            # Contention clears in seconds: owed, retried next tick, no backoff.
            _owe(bell)
            return "busy"
        if result.stopped:
            # ``failed`` (anything the store raised) or ``refused`` (its hourly cap):
            # owed, and backed off so a store that keeps failing is not re-derived
            # and re-reported on every member call of every tick.
            _failed(bell, f"publication {result.stopped}", result.error)
            return result.stopped
        logger.debug("credential repair inbox: %s", result)
        outcome = "published"
    else:
        outcome = "same"
    if bell.failures:
        logger.info("credential repair inbox: recovered after %d failed pass(es)", bell.failures)
    bell.failures, bell.retry_at, bell.pending = 0, 0.0, False
    bell.published = fingerprint
    bell.login_fp = fingerprint_before if bell.episodes else None
    return outcome
