"""The fetch service: eligibility, the refresh pass, and the read-side merge.

WHERE THIS SITS. ``cache.py`` owns state; ``adapters/`` owns one forge each;
``credentials.py`` owns logins. This module is the only place they meet, and it
is deliberately NOT on the live hook path (the hook writes a dirty MARK; the
network only runs from here).

THE THREE READ PATHS, and what each is allowed to do:

* **The route's GET** never blocks on the network (design §D.6): it merges the
  stored entries into the rows and, when :func:`plan` says something is
  eligible, KICKS a background refresh (single-flight per session) whose
  completion rewrites the index's ``updated_at`` — which is the feed frame's
  revision, so the client refetches and sees the result. That is the whole
  "completion is announced" story: nothing polls, the frame is the poke.
* **The route's POST /refresh** runs the same pass as a background task and
  answers 202 (``force`` bypasses TTLs but never cooling, per design §D.5).
* **The model tool's ``show``** is the one path allowed to AWAIT a fetch: it is
  a model call answering a model's question, not a UI poll. ``list`` reads
  stored state only (``view_rows``), so a bare listing never waits.

ELIGIBILITY (:func:`plan`) composes the design's three throttles: dirty marks
and TTL expiry decide NEED, cooling refuses the HOST, per-key backoff refuses
the KEY. Missing entries are always eligible (first sight of a ref), and a
link-only row (no adapter, unconfirmed host) is never eligible — "never
fetched" is its whole contract.

FAILURE POSTURE (design §D.4). A failed refresh never erases what was known:
the entry keeps its pieces and gets ``refresh_error`` plus ``stale`` so a
reader can tell "this is old" from "this is broken". A 401 survives one fresh
credential re-resolve before the row degrades to ``link_only`` with
"credential rejected"; a rate limit cools the host; a 5xx backs off the key.

REMOTE TEXT IS DATA. Comment bodies reach this module from the network and are
stored and excerpted as quoted data. Nothing here interpolates them into a
prompt, and the classification layer's rule (remote text never reaches option
descriptions) is untouched because this module never feeds it.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from local_operator.code_requests import cache as fetch_cache
from local_operator.code_requests import ledger, rounds
from local_operator.code_requests.adapters import FULL_FORGES, adapter_for, state_of
from local_operator.code_requests.adapters.base import (
    FetchOutcome,
    ForgeHTTPError,
    merge_comments,
)
from local_operator.code_requests.credentials import CredentialError, Token, resolve
from local_operator.code_requests.refs import Ref

logger = logging.getLogger(__name__)

#: How many refs one session refresh pass will fetch before it stops and lets
#: the next pass continue. A session's row list is normally a handful; the cap
#: bounds a pathological one (a script that opened fifty PRs) without failing.
MAX_REFS_PER_PASS = 24

#: The piece names whose content the round parser reads (GitHub's
#: ``comments``/``reviews``, GitLab's ``notes``). A pass that fetched ANY of
#: these has full bodies in hand and (re)parses; a pass that fetched none of
#: them replays the stored ``convention`` parse instead of re-reading bodies
#: the size bound has truncated (cross-round finding X1).
_COMMENT_PIECES = frozenset({"comments", "reviews", "notes"})

#: The state words whose lane's "awaiting review" default applies (the parser's
#: ``is_open`` argument): a merged/closed request is not awaiting anything.
_OPEN_STATES = frozenset({"open", "draft"})


@dataclass
class Plan:
    """What a refresh pass should attempt, and what it is deliberately not doing."""

    refs: list[Ref] = field(default_factory=list)
    #: Host -> until, for hosts that are cooling (the pass skips them entirely).
    cooling: dict[str, float] = field(default_factory=dict)
    #: key -> until, for keys in per-key backoff.
    backing_off: dict[str, float] = field(default_factory=dict)
    #: Why each eligible ref is eligible (``missing``/``dirty``/``expired``/``force``),
    #: for tests and for a log line that can explain a refresh wave.
    reasons: dict[str, str] = field(default_factory=dict)


def plan(
    config_dir: Any,
    session_id: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    force: bool = False,
    keys: Sequence[str] | None = None,
    now: float | None = None,
) -> Plan:
    """Decide which of ``rows`` a refresh pass may fetch, and why.

    Pure with respect to the network; reads the cache's disk tier and the
    session's dirty marks. ``keys`` filters to named refs (the POST body's
    ``keys``); ``force`` bypasses TTLs and dirty marks but NOT cooling or
    backoff — a force must never defeat the host's own rate limit.
    """
    moment = time.time() if now is None else now
    out = Plan()
    selected = {str(item) for item in keys} if keys else None
    dirty = fetch_cache.read_dirty(config_dir, session_id)
    dirty_keys = {str(item) for item in dirty.get("keys") or []}
    dirty_all = bool(dirty.get("all"))
    for raw in rows:
        ref_raw = raw.get("ref")
        ref = Ref.from_payload(ref_raw if isinstance(ref_raw, Mapping) else None)
        if ref is None or not ref.full or ref.forge not in FULL_FORGES:
            continue
        if selected is not None and ref.key not in selected:
            continue
        cooling_until = fetch_cache.cooling_until(ref.host, now=moment)
        if cooling_until is not None:
            out.cooling[ref.host] = cooling_until
            continue
        backoff_until = fetch_cache.key_backoff_until(ref.key, now=moment)
        if backoff_until is not None:
            out.backing_off[ref.key] = backoff_until
            continue
        entry = fetch_cache.read_entry(config_dir, ref)
        is_dirty = dirty_all or ref.key in dirty_keys
        if force:
            reason = "force"
        elif entry is None:
            reason = "missing"
        elif is_dirty:
            reason = "dirty"
        elif fetch_cache.is_expired(entry, forge=ref.forge, now=moment):
            reason = "expired"
        else:
            continue
        out.refs.append(ref)
        out.reasons[ref.key] = reason
        if len(out.refs) >= MAX_REFS_PER_PASS:
            break
    return out


def link_only_hint(ref: Mapping[str, Any] | None) -> str:
    """One line of remedy copy for a link-only row, per FORGE (finding X3).

    A UI cannot infer this: a gitea row is not "sign in with gh" — there is no
    CLI in scope for a detect-and-link-only forge at all — and naming the
    wrong CLI teaches the wrong fix. gh and glab are the two logins this slice
    can actually consume, so only they are named, each against its own family.
    """
    data = ref if isinstance(ref, Mapping) else {}
    forge = str(data.get("forge") or "")
    host = str(data.get("host") or "")
    if forge == "github":
        remedy = (
            "sign in with the gh CLI"
            if host == "github.com"
            else f"sign in with the gh CLI (`gh auth login --hostname {host}`)"
        )
        return f"Link only — {remedy} to track this one."
    if forge == "gitlab":
        remedy = (
            "sign in with the glab CLI"
            if host == "gitlab.com"
            else f"sign in with the glab CLI (`glab auth login --hostname {host}`)"
        )
        return f"Link only — {remedy} to track this one."
    return "Link only — this host isn't tracked yet."


def view_row(
    raw: Mapping[str, Any],
    entry: Mapping[str, Any] | None,
    *,
    cooling: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """One index row merged with its stored fetch entry, as the wire draws it.

    The index row's fields pass through; the entry contributes ``summary``,
    ``lanes``, ``fetched_at``, ``stale`` and ``refresh_error``. ``link_only``
    is TRUE unless there is real fetched data to draw: an entry with pieces.
    The reason string follows the server model's precedence — the ref's own
    sentence first (an unconfirmed host, a detect-and-link forge), then the
    entry's refresh error, then the scanner's classification note.
    """
    ref = raw.get("ref")
    ref_map: Mapping[str, Any] = ref if isinstance(ref, Mapping) else {}
    row: dict[str, Any] = {}
    for key in ("key", "relation", "relations", "acts", "mentions", "first_at", "last_at"):
        if key in raw:
            row[key] = raw[key]
    for key in ("via", "inherited_from", "unknown_reason", "evidence"):
        if raw.get(key) is not None:
            row[key] = raw[key]
    row.setdefault("key", str(raw.get("key") or ""))
    # The wire row draws the ref's FLAT fields (the route model's required
    # shape); the index keeps them nested under ``ref``. Copied here so both
    # the route and the tool read one merged dict rather than each unfolding
    # the ref by hand — the kind of second copy that drifts.
    for key in ("url", "forge", "host", "project", "number"):
        if ref_map.get(key) is not None:
            row[key] = ref_map.get(key)
    skip_reason = str(ref_map.get("reason") or "")
    has_pieces = bool(entry and entry.get("pieces"))
    if entry is None or not has_pieces:
        row["link_only"] = True
        row["summary"] = None
        row["lanes"] = None
        # The remedy copy, per forge (finding X3) — a flat row field the UI
        # renders instead of guessing a CLI from the forge id itself.
        row["link_only_hint"] = link_only_hint(ref_map)
        reason = skip_reason or str((entry or {}).get("refresh_error") or "")
        reason = reason or str(raw.get("unknown_reason") or "")
        if reason:
            row["reason"] = reason
        if entry and entry.get("refresh_error"):
            row["refresh_error"] = str(entry.get("refresh_error"))
    else:
        row["link_only"] = False
        summary = entry.get("summary")
        if isinstance(summary, Mapping):
            row["summary"] = {
                k: summary.get(k)
                for k in (
                    "state",
                    "draft",
                    "title",
                    "head_sha",
                    "head_ref",
                    "base_ref",
                    "author",
                    "created_at",
                    "updated_at",
                )
            }
            row["summary"]["ci"] = entry.get("ci")
            row["summary"]["url"] = str(ref_map.get("url") or "")
            # The UI's row contract reads ``summary.comments``
            # (DesktopCodeRequestSummary.comments in the desktop contract): the
            # count the host reported, null when a host did not — never a
            # rendered 0 (QA round 1, Q7).
            total = entry.get("comments_total")
            row["summary"]["comments"] = int(total) if isinstance(total, (int, float)) else None
        lanes = entry.get("lanes")
        row["lanes"] = lanes if isinstance(lanes, list) else None
        if entry.get("fetched_at") is not None:
            row["fetched_at"] = entry.get("fetched_at")
        if entry.get("stale"):
            row["stale"] = True
        if entry.get("refresh_error"):
            row["refresh_error"] = str(entry.get("refresh_error"))
    return row


def _pick_lane(entry: Mapping[str, Any], lane: str = "agent") -> Mapping[str, Any] | None:
    lanes = entry.get("lanes")
    if not isinstance(lanes, list):
        return None
    for item in lanes:
        if isinstance(item, Mapping) and item.get("lane") == lane:
            return item
    return None


def lane_freshness_note(entry: Mapping[str, Any] | None, head_sha: str = "") -> str:
    """The merge-note's "where is this up to" fragment, or ``""``.

    Built from the parser's own ``state_copy`` (so the note and the UI cannot
    disagree about a state word) plus the two SHAs the reviewer compares. The
    design's example: ``agent review r2 clean, fresh on 9d29452``.
    """
    if entry is None:
        return ""
    lane = _pick_lane(entry)
    if lane is None:
        return ""
    state_copy = str(lane.get("state_copy") or lane.get("state") or "")
    if not state_copy:
        return ""
    number = lane.get("round")
    label = f"agent review r{number}" if number is not None else "agent review"
    freshness = str(lane.get("freshness") or "")
    if freshness == rounds.FRESHNESS_FRESH and head_sha:
        return f"{label} {state_copy} on {head_sha[:7]}"
    if freshness == rounds.FRESHNESS_STALE and head_sha:
        return f"{label} {state_copy}, head {head_sha[:7]}"
    return f"{label} {state_copy}"


def view_for_ref(config_dir: Any, ref: Ref) -> dict[str, Any] | None:
    """The tool-facing view of one ref from the cache alone, or ``None``.

    Same merge as :func:`view_row`, but for a ref that has no index row: the
    tool's ``show`` may name a code request this session never saw, and that
    row is drawn from the cache without entering any ledger. An entry with no
    pieces (a failed first fetch) renders as the link-only view whose reason
    is the failure copy.
    """
    entry = fetch_cache.read_entry(config_dir, ref)
    if entry is None:
        return None
    has_pieces = bool(entry.get("pieces"))
    summary = entry.get("summary")
    view: dict[str, Any] = {
        "key": ref.key,
        "url": ref.url,
        "forge": ref.forge,
        "host": ref.host,
        "project": ref.project,
        "number": ref.number,
        "link_only": not has_pieces,
        "link_only_reason": ref.reason,
        # Same per-forge remedy copy the route's row carries (finding X3).
        "link_only_hint": link_only_hint({"forge": ref.forge, "host": ref.host}),
        "summary": None,
        "ci": None,
        "lanes": None,
        "comments": [],
        "comments_total": entry.get("comments_total"),
        "fetched_at": entry.get("fetched_at"),
        "checked_at": entry.get("checked_at"),
        "stale": bool(entry.get("stale")),
        "refresh_error": entry.get("refresh_error"),
    }
    if has_pieces:
        view["summary"] = dict(summary) if isinstance(summary, Mapping) else None
        view["ci"] = entry.get("ci")
        view["lanes"] = entry.get("lanes") if isinstance(entry.get("lanes"), list) else None
        view["comments"] = convention_comments(entry)
    else:
        view["link_only_reason"] = ref.reason or str(
            entry.get("refresh_error") or "no state has been fetched yet"
        )
    return view


def convention_comments(entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The stored comments that ARE convention comments, oldest first.

    Derived from the stored piece arrays, never stored twice: the pieces keep
    the (bounded) bodies a later revalidation needs, and this filter is what
    the tool renders. ``parse_comment`` is a first-line regex — cheap enough
    for a per-``show`` call.
    """
    pieces = entry.get("pieces")
    raw_lists: list[Any] = []
    if isinstance(pieces, Mapping):
        for name in ("comments", "reviews", "notes"):
            value = pieces.get(name)
            if isinstance(value, list):
                raw_lists.extend(value)
    merged = merge_comments([item for item in raw_lists if isinstance(item, Mapping)], [])
    out: list[dict[str, Any]] = []
    for item in merged:
        comment = rounds.Comment.from_payload(item)
        if comment is None:
            continue
        if rounds.parse_comment(comment) is not None:
            out.append(item)
    return out


def view_rows(config_dir: Any, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Merge every index row with its stored entry (no network, no writes)."""
    out: list[dict[str, Any]] = []
    for raw in rows:
        ref = Ref.from_payload(raw.get("ref") if isinstance(raw.get("ref"), Mapping) else None)
        entry = fetch_cache.read_entry(config_dir, ref) if ref is not None else None
        out.append(view_row(raw, entry))
    return out


async def show(
    config_dir: Any,
    ref: Ref,
    *,
    force: bool = False,
    timeout_s: float = 25.0,
    session_id: str = "",
) -> dict[str, Any]:
    """Fetch-on-demand for the tool's ``show``: AWAIT a real fetch when needed.

    This is the one path allowed to block on the network (see the module
    docstring): a model asked a question about a specific ref, and an answer
    from a two-hour-old TTL is worse than a 2 s wait. Cooling and per-key
    backoff still refuse (a rate-limited host must not be hammered by a
    model's curiosity); the caller renders whatever is stored, which is the
    same degraded answer a UI gets. A timeout degrades the same way.

    ``session_id`` makes the session's own DIRTY MARKS honour-able here too
    (review round 1, F6): the acted seam marks the ref it just touched, and a
    ``show`` of that ref must not answer from the pre-act state until the TTL
    lapses. The key's mark is consumed on the attempt, under the same ``since``
    race guard the pass uses.
    """
    entry = fetch_cache.read_entry(config_dir, ref)
    needs = force or entry is None or fetch_cache.is_expired(entry, forge=ref.forge)
    dirty_at: float | None = None
    if not needs and session_id:
        dirty = fetch_cache.read_dirty(config_dir, session_id)
        at = dirty.get("at")
        dirty_at = at if isinstance(at, (int, float)) else None
        marked = bool(dirty.get("all")) or ref.key in {str(k) for k in dirty.get("keys") or ()}
        needs = marked
    if needs and adapter_for(ref) is not None:
        cooling = fetch_cache.cooling_until(ref.host)
        backing_off = fetch_cache.key_backoff_until(ref.key)
        if cooling is None and backing_off is None:
            try:
                await asyncio.wait_for(
                    refresh_keys(config_dir, [ref], force=force), timeout=timeout_s
                )
            except asyncio.TimeoutError:
                logger.debug("code-request show timed out for %s", ref.key)
            if session_id:
                fetch_cache.clear_dirty(config_dir, session_id, keys=[ref.key], since=dirty_at)
    view = view_for_ref(config_dir, ref)
    if view is not None:
        return view
    return {
        "key": ref.key,
        "url": ref.url,
        "forge": ref.forge,
        "host": ref.host,
        "project": ref.project,
        "number": ref.number,
        "link_only": True,
        "link_only_reason": ref.reason or "no state has been fetched yet",
        "summary": None,
        "ci": None,
        "lanes": None,
        "comments": [],
        "fetched_at": None,
        "checked_at": None,
        "stale": False,
        "refresh_error": None,
    }


@dataclass
class RefreshReport:
    """What one refresh pass did, honestly: attempted, changed, refused, failed."""

    attempted: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    cooling: dict[str, float] = field(default_factory=dict)
    backing_off: dict[str, float] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)


#: In-flight refresh PASSES, keyed by ``(config_dir, session_id)``. Single-flight
#: per session: a GET that arrives while a pass runs joins it rather than
#: queueing another. Strong references are held here on purpose — a bare
#: ``create_task`` has only a weak referent and can be collected mid-flight
#: (the route's own scan latch documents the same trap).
_SESSION_TASKS: dict[tuple[str, str], "asyncio.Task[RefreshReport]"] = {}

#: Per-KEY single-flight locks (design §D.2: "single-flight per key per
#: process"), pruned opportunistically so the map cannot grow without bound.
_KEY_LOCKS: dict[tuple[str, str], asyncio.Lock] = {}


def _key_lock(config_dir: Any, key: str) -> asyncio.Lock:
    ident = (str(config_dir), key)
    lock = _KEY_LOCKS.get(ident)
    if lock is None:
        if len(_KEY_LOCKS) > 512:
            for stale in [item for item, value in _KEY_LOCKS.items() if not value.locked()][:256]:
                _KEY_LOCKS.pop(stale, None)
        lock = asyncio.Lock()
        _KEY_LOCKS[ident] = lock
    return lock


async def probe_shorthand(
    config_dir: Any, ref: Ref, *, timeout_s: float = 15.0
) -> tuple[str, Ref | None]:
    """Resolve an ambiguous ``owner/repo#N`` for ``show``: ``(verdict, ref)``.

    A qualified ref cannot say whether #N is an issue or a pull request, and
    the tool doc + guide advertise the form, so ``show`` makes it work rather
    than advertise dead syntax (QA round 1, Q3). One pull request probe: 200
    promotes the ref to full — the normal show path then fetches and caches it
    — and a 404 reports ``"issue"`` (GitHub numbers issues and pull requests
    from one sequence, so the number exists as an issue, or is invisible to
    this login). Rate limits cool the host; every other outcome is
    ``"unknown"`` and the caller keeps the link-only row it already had. The
    probe never writes the ledger, and the credential gate inside
    :func:`resolve` means an unauthenticated host makes ZERO requests.
    """
    adapter = adapter_for(ref)
    probe = getattr(adapter, "probe_pull", None) if adapter is not None else None
    if probe is None:
        return "unknown", None
    try:
        token = await asyncio.to_thread(resolve, ref.host, ref.forge)
    except CredentialError:
        return "unknown", None
    try:
        verdict = await asyncio.wait_for(probe(ref, token.value), timeout=timeout_s)
    except ForgeHTTPError as exc:
        if exc.kind == "rate_limited":
            fetch_cache.note_rate_limited(
                ref.host, reset_at=exc.reset_at, retry_after=exc.retry_after
            )
        return "unknown", None
    except asyncio.TimeoutError:
        return "unknown", None
    if verdict == "pull":
        # The same identity, now certain: a full ref enters the normal fetch
        # and cache path (and only there does anything get stored).
        return "pull", Ref(ref.forge, ref.host, ref.project, ref.number, ref.url, full=True)
    return (verdict, None) if verdict == "issue" else ("unknown", None)


async def refresh_keys(
    config_dir: Any, refs: Sequence[Ref], *, force: bool = False
) -> RefreshReport:
    """Fetch ``refs`` under the key locks, writing entries; never raises for a ref.

    SINGLE-FLIGHT, in the strong sense: a second caller queued on a key's lock
    whose entry was checked by a pass that STARTED after this call began is
    skipped rather than refetched. That is what makes "one pass per key per
    process" true rather than "serialised passes" — the GET that schedules,
    the POST that refreshes and a concurrent ``show`` all collapse into one
    network call when they overlap. A ``force`` pass never skips, and a stale
    entry never skips either (a failed pass must be retried under the key's
    backoff, not treated as fresh).
    """
    started = time.time()
    report = RefreshReport()
    for ref in refs:
        async with _key_lock(config_dir, ref.key):
            # RE-CHECK the throttles per ref: the plan's snapshot is taken once,
            # but a 429 on an earlier ref sets HOST cooling mid-pass, and the
            # remaining refs on that host must make zero calls (review round 1,
            # F4a / QA round 1, Q4 — a sibling's success used to clear the
            # cooling within the same pass and the host got hammered anyway).
            # Unconditional, force included: a force must never defeat the
            # host's own rate limit.
            cooling_until = fetch_cache.cooling_until(ref.host)
            if cooling_until is not None:
                report.cooling[ref.host] = cooling_until
                continue
            backoff_until = fetch_cache.key_backoff_until(ref.key)
            if backoff_until is not None:
                report.backing_off[ref.key] = backoff_until
                continue
            if not force:
                current = await asyncio.to_thread(fetch_cache.read_entry, config_dir, ref)
                checked = (current or {}).get("checked_at")
                if (
                    isinstance(checked, (int, float))
                    and checked >= started
                    and not (current or {}).get("stale")
                ):
                    continue
            report.attempted.append(ref.key)
            try:
                changed = await _fetch_one(config_dir, ref, force=force)
            except Exception:  # noqa: BLE001 - one ref never fails the pass
                logger.warning("code-request refresh failed for %s", ref.key, exc_info=True)
                report.failed[ref.key] = "refresh failed"
                continue
            if changed:
                report.changed.append(ref.key)
    return report


async def refresh_session(
    config_dir: Any,
    session_id: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    keys: Sequence[str] | None = None,
    force: bool = False,
) -> RefreshReport:
    """Plan and run one session's refresh pass; touch the index when content moved.

    Consumption rule: the session's dirty marks are cleared on the ATTEMPT (a
    failure lands in the per-key backoff, which is the throttle that must not
    be bypassed). The ``since`` guard keeps a mark that landed DURING the pass.
    """
    moment = time.time()
    dirty = fetch_cache.read_dirty(config_dir, session_id) if session_id else {}
    dirty_at = dirty.get("at")
    planned = plan(config_dir, session_id, rows, force=force, keys=keys, now=moment)
    report = RefreshReport(cooling=dict(planned.cooling), backing_off=dict(planned.backing_off))
    if planned.refs:
        report = _merge_reports(report, await refresh_keys(config_dir, planned.refs, force=force))
    if session_id:
        attempted_keys = [ref.key for ref in planned.refs]
        # Consume the turn-end/wake ``all`` mark on THIS pass too: the plan above
        # selected every fetchable row (a plan truncated at MAX_REFS_PER_PASS
        # does not pass the flag on), so the mark has been serviced — leaving it
        # would refetch the session on every GET (QA round 1, Q2). Guarded by
        # the same ``since`` snapshot as the keys.
        consume_all = bool(dirty.get("all")) and len(planned.refs) < MAX_REFS_PER_PASS
        fetch_cache.clear_dirty(
            config_dir,
            session_id,
            keys=attempted_keys,
            since=dirty_at if isinstance(dirty_at, (int, float)) else None,
            consume_all=consume_all,
        )
        if report.changed:
            # The completion signal (design §D.6): the index's own ``updated_at``
            # is what the feed frame and the route both report as ``revision``,
            # so bumping it IS the "fetch finished" announcement. Cache first,
            # index second — the feed watches the index, so its frame never
            # announces data that is not on disk yet.
            try:
                await asyncio.to_thread(ledger.touch_index, config_dir, session_id)
            except Exception:  # noqa: BLE001 - a missing index is not a failure
                logger.debug("could not touch the code-request index", exc_info=True)
    fetch_cache.sweep(config_dir)
    return report


def _merge_reports(base: RefreshReport, fresh: RefreshReport) -> RefreshReport:
    base.attempted.extend(fresh.attempted)
    base.changed.extend(fresh.changed)
    base.failed.update(fresh.failed)
    return base


def schedule_session_refresh(
    config_dir: Any,
    session_id: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    keys: Sequence[str] | None = None,
    force: bool = False,
) -> bool:
    """Kick a background refresh pass (single-flight); True when one is now running.

    Called from an event loop only (both routes are async). A pass already in
    flight for this session means "joined", not "skip": the caller still
    returns its cached read immediately, which is the GET contract.
    """
    ident = (str(config_dir), session_id)
    existing = _SESSION_TASKS.get(ident)
    if existing is not None and not existing.done():
        return True

    async def _run() -> RefreshReport:
        try:
            return await refresh_session(config_dir, session_id, rows, keys=keys, force=force)
        finally:
            _SESSION_TASKS.pop(ident, None)

    try:
        _SESSION_TASKS[ident] = asyncio.get_running_loop().create_task(
            _run(), name=f"code-requests-refresh:{session_id}"
        )
    except RuntimeError:  # no loop: nothing to schedule on, and that is fine
        return False
    return True


async def _fetch_one(config_dir: Any, ref: Ref, *, force: bool = False) -> bool:
    """Fetch one ref under the design's failure policy. Returns whether the row changed.

    Never raises for an expected condition: every branch here either writes a
    new entry (returns True) or records why not and returns False. The caller
    (a background pass or the tool's ``show``) treats both as "the pass ran".
    """
    adapter = adapter_for(ref)
    if adapter is None:
        return False
    prior = fetch_cache.read_entry(config_dir, ref)
    try:
        token = await asyncio.to_thread(resolve, ref.host, ref.forge)
    except CredentialError as exc:
        changed = _record_failure(config_dir, ref, prior, exc.message)
        # A gentle per-key backoff so a poll cannot re-spawn the login probe on
        # every GET; a sign-in is picked up on the next attempt after it.
        fetch_cache.note_key_failure(ref.key)
        return changed
    try:
        outcome = await adapter.fetch(
            ref,
            _validators_of(prior),
            token.value,
            stored=(prior or {}).get("pieces") or {},
        )
    except ForgeHTTPError as exc:
        return await _handle_fetch_error(config_dir, ref, prior, exc, token)
    fetch_cache.clear_key_backoff(ref.key)
    fetch_cache.note_host_success(ref.host)
    entry = _build_entry(ref, prior, outcome, adapter)
    fetch_cache.write_entry(config_dir, entry, ref=ref)
    return bool(outcome.pieces)


async def _handle_fetch_error(
    config_dir: Any,
    ref: Ref,
    prior: Mapping[str, Any] | None,
    exc: ForgeHTTPError,
    token: Token,
) -> bool:
    """The failure policy, one branch per ``kind`` (design §D.4)."""
    if exc.kind == "unauthorized":
        # ONE fresh re-resolve, then degrade to link_only. Never an error wall.
        try:
            fresh = await asyncio.to_thread(resolve, ref.host, ref.forge, fresh=True)
        except CredentialError:
            changed = _record_failure(config_dir, ref, prior, _REJECTED)
            fetch_cache.note_key_failure(ref.key)
            return changed
        adapter = adapter_for(ref)
        assert adapter is not None
        try:
            outcome = await adapter.fetch(
                ref,
                _validators_of(prior),
                fresh.value,
                stored=(prior or {}).get("pieces") or {},
            )
        except ForgeHTTPError as second:
            if second.kind == "unauthorized":
                changed = _record_failure(config_dir, ref, prior, _REJECTED)
                fetch_cache.note_key_failure(ref.key)
                return changed
            return await _handle_fetch_error(config_dir, ref, prior, second, fresh)
        fetch_cache.clear_key_backoff(ref.key)
        fetch_cache.note_host_success(ref.host)
        fetch_cache.write_entry(config_dir, _build_entry(ref, prior, outcome, adapter), ref=ref)
        return bool(outcome.pieces)
    if exc.kind == "rate_limited":
        fetch_cache.note_rate_limited(ref.host, reset_at=exc.reset_at, retry_after=exc.retry_after)
        # The row keeps its last known data; cooling is a HOST state and is
        # surfaced in the payload, not per row.
        return False
    if exc.kind in ("server", "network"):
        fetch_cache.note_key_failure(ref.key)
        suffix = f" {exc.status}" if exc.status else ""
        return _record_failure(
            config_dir,
            ref,
            prior,
            f"refresh failed ({exc.kind}{suffix}); keeping the last known data",
        )
    if exc.kind == "not_found":
        # A 404 is a row-level condition and a THROTTLED one (review round 1,
        # F3): the row keeps its last known data with the reason, ``checked_at``
        # moves so the TTL paces the retry, and the key backs off — a repeating
        # 404 used to return "changed" every pass, which moved the feed
        # revision and drove a client refetch loop.
        fetch_cache.note_key_failure(ref.key)
        return _record_failure(config_dir, ref, prior, "not found at the forge")
    fetch_cache.note_key_failure(ref.key)
    return _record_failure(config_dir, ref, prior, f"refresh failed (HTTP {exc.status})")


_REJECTED = "credential rejected — sign in again with gh/glab, then refresh"


def _validators_of(prior: Mapping[str, Any] | None) -> dict[str, dict[str, str]]:
    raw = (prior or {}).get("validators")
    if not isinstance(raw, Mapping):
        return {}
    return {
        str(piece): {str(k): str(v) for k, v in value.items()}
        for piece, value in raw.items()
        if isinstance(value, Mapping)
    }


def _record_failure(
    config_dir: Any, ref: Ref, prior: Mapping[str, Any] | None, message: str
) -> bool:
    """Keep the last known row, add ``stale`` + ``refresh_error``; True when the
    RENDERED row changed.

    A failure is a cache state like any other (review round 1, F3):
    ``checked_at`` moves so the TTL paces retries instead of every poll
    refetching, and the return value is a CONTENT comparison — a repeated
    identical error must not report "changed", or ``refresh_session`` would
    touch the index and drive the client's refetch loop off a failure that is
    not moving (QA round 1: 5 passes against a 404 moved the revision every
    time). A failure on a never-fetched ref still records the entry: the route
    renders its ``refresh_error`` as the link-only reason, which is exactly the
    "sign in with gh/glab" copy the design asks for.
    """
    entry = dict(prior or {})
    entry.setdefault("key", ref.key)
    entry.setdefault("forge", ref.forge)
    entry.setdefault("host", ref.host)
    entry.setdefault("project", ref.project)
    entry.setdefault("number", ref.number)
    changed = (
        not prior or not prior.get("stale") or str(prior.get("refresh_error") or "") != message
    )
    entry["refresh_error"] = message
    entry["stale"] = True
    entry["checked_at"] = round(time.time(), 3)
    fetch_cache.write_entry(config_dir, entry, ref=ref)
    return changed


def _build_entry(
    ref: Ref,
    prior: Mapping[str, Any] | None,
    outcome: FetchOutcome,
    adapter: Any,
) -> dict[str, Any]:
    """The next stored entry: merged pieces, derived state/CI/lanes/comments.

    The bulky comment arrays are DROPPED from the stored pieces after the
    derivation: the top-level ``comments`` (convention bodies only) and
    ``lanes`` are what every reader renders, and keeping a second copy of
    every body would double the entry for the same bytes.

    THE PARSE IS A FETCH-TIME ARTEFACT (cross-round finding X1). It runs on
    the FULL bodies whenever this pass fetched a comment-bearing piece, and its
    result is STORED (``convention``); a rebuild that did not refetch the
    comments replays that stored parse instead of re-parsing bodies the size
    bound has since truncated. Before this, a verdict past the 4 KiB cap —
    #2106's round-2 review keeps its verdict at char 6708 of 6979 — parsed as
    ``terminal`` on the fetch that saw the full text and as ``unstated`` on the
    next 304 that re-read the capped copy.
    """
    moment = round(time.time(), 3)
    pieces, validators = fetch_cache.merge_pieces(prior, outcome)
    state = state_of(adapter, pieces) or str((prior or {}).get("state") or "")
    all_comments = adapter.comments(pieces)
    head_sha = ""
    summary_piece = pieces.get("summary")
    if isinstance(summary_piece, Mapping):
        head_sha = str(summary_piece.get("head_sha") or "")
    # The state word is the SERVICE's to spell (the adapters keep the host's
    # raw fields): the stored summary carries it so every reader of
    # ``summary.state`` sees one answer, and ``states(head_sha=...)`` is what
    # turns a reviewed-SHA into fresh/stale — calling it without the head is
    # how every lane silently reads "freshness unknown".
    summary = dict(summary_piece) if isinstance(summary_piece, Mapping) else {}
    summary["state"] = state
    fresh_comments = bool(_COMMENT_PIECES & set(outcome.pieces))
    report: Any = None
    if not fresh_comments and isinstance(prior, Mapping):
        stored = prior.get("convention")
        if isinstance(stored, Mapping):
            report = rounds.RoundReport.from_payload(stored)
    if report is None:
        report = rounds.parse(
            all_comments, head_sha=head_sha or None, is_open=state in _OPEN_STATES
        )
    states = report.states(head_sha or None, is_open=state in _OPEN_STATES)
    lanes = [item.to_payload() for item in states]
    entry: dict[str, Any] = {
        "key": ref.key,
        "forge": ref.forge,
        "host": ref.host,
        "project": ref.project,
        "number": ref.number,
        # The pieces KEEP their comment arrays: a 304 on a later refresh must
        # still answer `comments()` from what is stored, and ``cache._bound``
        # is what keeps those arrays small (bodies capped, newest kept).
        "pieces": dict(pieces),
        "validators": validators,
        "summary": summary,
        "state": state,
        "ci": adapter.ci(pieces) if state else None,
        "lanes": lanes,
        # The fetch-time parse, replayed across non-refetching rebuilds; see
        # the docstring. Small: passes carry verdict text, not bodies.
        "convention": {
            "passes": [item.to_payload() for item in report.passes],
            "ignored": list(report.ignored),
        },
        "comments_total": len(all_comments),
        "checked_at": moment,
        "fetched_at": moment if outcome.pieces else (prior or {}).get("fetched_at", moment),
        "refresh_error": None,
        "stale": False,
    }
    return entry


__all__ = [
    "MAX_REFS_PER_PASS",
    "Plan",
    "RefreshReport",
    "convention_comments",
    "lane_freshness_note",
    "plan",
    "refresh_keys",
    "refresh_session",
    "schedule_session_refresh",
    "show",
    "view_for_ref",
    "view_row",
    "view_rows",
]
