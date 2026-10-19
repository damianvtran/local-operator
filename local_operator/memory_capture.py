"""Best-effort capture is speculative; ONLY a committed marker authorizes writes.

No inference fork: snapcompact has no semantic model summary. Locators work for
both strategies; context-full can also reuse its already-paid semantic sections.
Explicit stores remain the guaranteed route for decisions, especially long runs.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, NamedTuple

from local_operator.memory import (
    MAX_REVISIONS,
    MemoryError,
    MemoryStore,
    _dump,
    guard,
    validate_fact,
    validate_op,
)

logger = logging.getLogger(__name__)

# Shipped ON: a dark-launched default would leave the feature undelivered. Flipped only
# after the isolated pilot (store->compact->restart->retrieve, wire-prefix, no-wake,
# privacy, mesh) passed. settings_io reads this constant too so CLI/TUI declarations and
# consumers share ONE default authority; `memory.enabled: false` remains the kill switch
# and never erases retained facts.
MEMORY_DEFAULT_ENABLED = True
# Every quantifier is bounded and the email alternative is anchored so a scan cannot
# restart inside a word run: the old `[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}` took 3.7 s on 32K
# letters, GIL-held, so no "2 s budget" between matches could interrupt it (review R1).
# Hex ids stop at 12 characters (short git SHAs) and need a digit: longer bare hex is
# indistinguishable from an API key and is deliberately NOT captured (review R4).
_LOCATOR = re.compile(
    # Consume other URI schemes atomically too: ignoring guide:// must not
    # leave its //tail available to the path alternative. A window is <=4096.
    # A label colon (URL:https://...) is punctuation, not a URI tail. Real
    # scheme:// tokens are consumed by this alternative before their contents.
    r"(?<![\w/])(?P<uri>[A-Za-z][A-Za-z0-9+.-]{0,31}://[^\s<>\"'`]{1,4096})"
    r"|(?<![\w.+-])[\w.+-]{1,64}@[\w.-]{1,255}\.[A-Za-z]{2,24}"
    # A slash inside a ratio, URI or another path is not a new absolute root.
    r"|(?<![\w:/~.-])(?:~?/|\.\.?/|(?:src|docs|tests|local_operator)/)[^\s<>\"'`]{1,400}"
    r"|\bsecret\s{1,8}[A-Z][A-Z0-9_]{2,63}\b|(?<!\w)#[\w-]{2,64}"
    r"|\b(?=[0-9a-f]{7,12}\b)[0-9a-f]*\d[0-9a-f]*\b"
    # The finite bound is a rejection limit, not permission to publish a
    # prefix of a longer branch/thread identifier into memory or associations.
    r"|(?P<reference>\b(?:branch|thread)(?:\s{0,3}:\s{0,3}|\s{1,3})[\w./-]{3,128}(?![\w./-]))"
)
#: The regex never sees more than this many characters at once. Bounded quantifiers make
#: each call linear; windowing makes the worst case independent of input size, and lets
#: the time budget be checked between windows (a single re call cannot be interrupted).
SCAN_WINDOW = 4096
#: How long the scan may look before it stops and hands back what it found. Named because
#: two things must agree: this value, and ``finish``'s await, which has to be STRICTLY
#: longer than it (see ``_SCAN_GRACE``) or the partial rows never reach the write.
SCAN_BUDGET = 2.0
#: Grace on the await in ``finish``. The scan checks its own budget between windows, so it
#: normally raises with its partial rows a moment after ``SCAN_BUDGET``; the await only has
#: to outlast that hand-off. It stays a bound, not a promise: a wedged thread still times
#: out here, and that path records ``timeout`` without rows (nothing was observable).
_SCAN_GRACE = 0.5
#: Coarse, content-free causes recorded BESIDE a ``captures`` row (in ``meta``), so a
#: pass that captured less than everything is attributable later. Literal members only:
#: this record exists so a failed pass is attributable at all, so it must never carry
#: free text (a body, a locator or an exception message) out of the store.
#: Each literal names the GUARD that dropped candidates, never a verdict about them:
#: ``admission-skip`` = ``extract``'s admission guard skipped a candidate, which never
#: reached ``persist``; ``write-refused`` = ``persist``'s write belt refused a row;
#: ``timeout`` = the scan budget ran out; ``quota`` = the store refused growth;
#: ``store-error`` = anything else. The old ``refused-row`` was a VERDICT that named
#: neither guard, so a record reading ``refused=0, skipped=18`` was misread as an
#: over-refusing write guard when it was an admission skip; it is deliberately not
#: accepted any more. Records written before :data:`ATTRIBUTION_STARTED_KEY` stay
#: unclassifiable rather than being assigned a cause they never recorded.
CAPTURE_REASONS = ("admission-skip", "write-refused", "timeout", "quota", "store-error")
#: ``meta`` key stamping the first capture written by a build that records guard-named
#: reasons and the ``declined`` status. Reporting must cohort on it (rows older than the
#: stamp predate the split and cannot be reclassified honestly), and it is never
#: backfilled onto a store whose rows were written before the split.
ATTRIBUTION_STARTED_KEY = "attribution_started_at"
_HEADINGS = re.compile(r"^#{1,4}\s*(?:Goal|Key Decisions|Progress|Critical Context)\b", re.I)


class CaptureRows(list[dict[str, Any]]):
    """Candidate rows plus the ADMISSION skips that produced them.

    A list SUBCLASS on purpose: ``extract``'s result crosses a task hand-off
    (``session._begin_memory_capture`` builds the task and ``finish`` awaits it), and
    every existing consumer reads it as a list. The count has to travel WITH the rows
    or the attribution is lost by the time ``persist`` writes.

    ``skipped`` counts candidates the admission guard refused inside ``extract``; they
    never reach ``persist``, whose ``refused`` count is the WRITE-time belt. The two are
    recorded separately so a stricter ``guard()`` rule is visible instead of silently
    shrinking the capture (review round 1, MINOR 2).
    """

    __slots__ = ("skipped",)

    def __init__(self, rows: Iterable[dict[str, Any]] = (), skipped: int = 0):
        super().__init__(rows)
        self.skipped = skipped


class BudgetExceeded(TimeoutError):
    """The scan ran out of its budget; ``rows`` is every candidate found so far.

    A TimeoutError SUBCLASS on purpose: an existing caller that only knows "capture
    timed out" keeps working, while ``finish`` can tell this apart and persist the partial
    rows instead of recording a ``failed``/0 marker for a pass that merely ran long.
    """

    def __init__(self, rows: CaptureRows):
        super().__init__("memory capture budget")
        self.rows = rows


class _QuotaExceeded(MemoryError):
    """``MemoryStore._quota`` refused a capture write; ``finish`` records ``quota``.

    Subclasses MemoryError so a caller that only handles the store's refusal keeps
    working; the subclass exists solely so the row's reason is not mislabelled as a
    generic store error.
    """


def policy(config: Path) -> dict[str, Any]:
    from local_operator.config import read_config_values

    values = read_config_values(config) or {}
    memory = values.get("memory", {})
    memory = memory if isinstance(memory, dict) else {}
    return {
        key: memory.get(key, MEMORY_DEFAULT_ENABLED) is True
        for key in ("enabled", "capture", "suggest")
    }


def import_capture_disabled(session_dir: str | Path | None) -> bool:
    """True for a session an import marked ``capture_memory: false``.

    The import service writes ``harness-import.json`` into every target session
    and sets that flag because an imported conversation has already been mined
    by its SOURCE harness; re-mining it here would duplicate foreign facts into
    this store under this harness's provenance. Absence of the manifest is the
    ordinary live-session case and stays enabled. A malformed manifest also
    stays enabled: this flag can only ever SUPPRESS capture, so failing open
    preserves existing behaviour rather than silently disabling memory.
    """
    if not session_dir:
        return False
    path = Path(session_dir) / "harness-import.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(manifest, dict) and manifest.get("capture_memory") is False


def carry_rows(carry: str, *, limit: int = 12) -> list[dict[str, Any]]:
    """Memory candidates from an archive's carry digest.

    WHY the carry, when ``extract`` already scans a locator set: the scan reads
    ``recovery_text``, which samples only a bounded head and tail of the pre-cut
    prefix. Measured on the operator's own session (439818272d84, pass 45): that
    sample covered 10.4% of a 9.6 M-char prefix and MISSED 2,584 of 3,524
    locators (73%) — and missed every operator ask and decision in the middle
    entirely, because prose carries no locator. The carry is the pass's own
    deterministic digest of exactly the material that was about to be dropped
    (see :mod:`local_operator.compaction.mine`), so feeding it here closes that
    gap with no extra model call and no second scan of the transcript.

    Every candidate is put through :func:`guard` and a refused one is SKIPPED,
    exactly as ``extract`` guards each item it emits. This is not decoration:
    the miner mines tool results as well as prose, so an ordinary path like
    ``docs/.env`` or ``~/.aws/credentials`` can appear in a carry, and without a
    per-item guard a single refused substring would make ``validate_fact`` raise
    and take the WHOLE capture down to a ``failed`` terminal with zero rows
    (review round 1, BLOCKER 1).

    Bounded and de-duplicated like every other capture source: at most ``limit``
    rows, whitespace-collapsed, each capped well under the fact body limit.
    """
    if not carry:
        return []
    from local_operator.compaction.mine import parse_carry

    parsed = parse_carry(carry)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for tag, items in (
        ("ask", parsed.asks),
        ("decision", parsed.decisions),
        ("locator", parsed.identifiers),
    ):
        for item in items:
            if len(rows) >= limit:
                break
            body = " ".join(item.split())[:280]
            if not body or body in seen:
                continue
            seen.add(body)
            try:
                # ``opaque=True``: ``validate_fact`` applies the opaque guard to
                # a fact body, so a candidate that clears only the DEFAULT guard
                # would still raise inside ``persist`` and take the whole capture
                # to a ``failed`` terminal. Matching the guard the write applies
                # is what actually closes BLOCKER 1 (review round 2, MAJOR C).
                guard(body, opaque=True)
            except MemoryError:
                # A refused candidate must never fail the whole capture: this is
                # the BLOCKER-1 fix — one unguardable locator skips, the rest of
                # the pass's facts still land.
                continue
            rows.append({"title": body[:100], "body": body, "tags": [tag]})
    return rows[:limit]


def carry_digest(carry: str) -> str:
    """The carry rendered as SEMANTIC sections, for callers that parse a summary
    rather than a tier list.

    Renders the carry under the headings ``extract`` already recognises
    (``Goal``/``Key Decisions``/``Critical Context``), so the semantic arm works
    UNCHANGED for a pass that has no model summary (snapcompact) and no new fact
    grammar is introduced. Deliberately NOT fed to capture alongside
    :func:`carry_rows` in the same pass — writing the same bodies twice under two
    tag sets is the duplication a reviewer found (round 1, MAJOR 7); degree
    callers pick one. Retained because ``extract`` is the path that also handles a
    context-full summary, and a caller that only has a summary string still needs
    the carry reachable through its own parser.
    """
    if not carry:
        return ""
    from local_operator.compaction.mine import parse_carry

    parsed = parse_carry(carry)
    lines: list[str] = []
    if parsed.asks:
        lines.append("# Goal")
        lines.extend(f"- {item}" for item in parsed.asks)
    if parsed.decisions:
        lines.append("# Key Decisions")
        lines.extend(f"- {item}" for item in parsed.decisions)
    if parsed.identifiers:
        lines.append("# Critical Context")
        lines.extend(f"- {item}" for item in parsed.identifiers)
    return "\n".join(lines)


def snapshot(messages: list[Any]) -> str:
    """Bound before serialising; images/tool objects/environment are never ingested."""
    head: list[str] = []
    tail: list[str] = []
    used = 0
    # Bound traversal as well as the final string: iterating every old block
    # just to discard its text would still stall a very long compaction plan.
    selected = messages if len(messages) <= 64 else messages[:16] + messages[-48:]
    for message in selected:
        for content in getattr(message, "content", ())[:128]:
            text = getattr(content, "text", None)
            if not isinstance(text, str):
                continue
            # Each message can be enormous. Both ends, not an unbounded join.
            bounded = text[:16384] + (text[-16384:] if len(text) > 16384 else "")
            if used < 512000:
                head.append(bounded)
                used += len(bounded)
            tail.append(bounded)
            if sum(map(len, tail)) > 512000:
                tail.pop(0)
    return ("\n".join(head) + "\n" + "\n".join(tail)).encode()[:2000000].decode(errors="ignore")


def windows(text: str) -> Iterator[str]:
    """Split ``text`` into <= SCAN_WINDOW pieces, preferring whitespace boundaries."""
    start = 0
    size = len(text)
    while start < size:
        end = min(size, start + SCAN_WINDOW)
        if end < size:
            cut = max(text.rfind(" ", start, end), text.rfind("\n", start, end))
            if cut > start:
                end = cut
        yield text[start:end]
        start = end


def extract_messages(messages: list[Any]) -> list[dict[str, Any]]:
    return extract(snapshot(messages))


def _ranked(counts: Counter[str], last: dict[str, int]) -> list[dict[str, Any]]:
    """The bounded candidate list, ranked by last occurrence then frequency.

    ONE renderer for both exits (the completed scan and the budget-exceeded one), so a
    partial result is shaped and ordered exactly like a complete one.
    """
    return [
        {"title": locator[:100], "body": locator, "tags": ["locator"]}
        for locator in sorted(counts, key=lambda x: (last[x], counts[x], x), reverse=True)[:40]
    ]


def extract(text: str, summary: str = "") -> CaptureRows:
    """CPU-only bounded scan; returns values, never opens a store or writes state.

    Raises :class:`BudgetExceeded` when the scan budget runs out, carrying the rows found
    up to that point: running out of time is a degraded result, never a lost one.

    The returned list carries ``skipped``, the number of candidates the admission guard
    refused. They are counted rather than silently dropped: an admission rule stricter
    than the write's would otherwise shrink a pass with no trace anywhere (review round
    1, MINOR 2).
    """
    started = time.monotonic()
    counts: Counter[str] = Counter()
    last: dict[str, int] = {}
    skipped = 0
    index = 0
    offset = 0
    for window in windows(text[:2000000]):
        if time.monotonic() - started > SCAN_BUDGET:
            raise BudgetExceeded(CaptureRows(_ranked(counts, last), skipped=skipped))
        start = offset
        offset += len(window)
        if start and not text[start - 1].isspace():
            # A forced window cut can bisect an oversized URI/path. Do not
            # reinterpret its continuation as an independent locator; traversal
            # is still bounded by the unchanged scan window, with no I/O.
            prefix = next((i for i, char in enumerate(window) if char.isspace()), len(window))
            window = window[prefix:]
        for match in _LOCATOR.finditer(window):
            index += 1
            if match.end() == len(window) and offset < len(text) and not text[offset].isspace():
                continue  # neither a forced cut nor the input cap publishes a whole token
            locator = match.group().rstrip(".,;:)]}")
            if len(locator) > 200:
                continue  # dropping an oversized candidate is safer than persisting a fragment
            if match.lastgroup == "uri" and not locator.lower().startswith(("http://", "https://")):
                continue
            if match.lastgroup == "reference" and ":" not in locator:
                identifier = locator.split(maxsplit=1)[1]
                # Without an explicit delimiter, ordinary prose ("branch that",
                # "thread about") is not evidence of an identifier. Slash refs
                # and numeric thread IDs remain useful without inventing facts.
                if "/" not in identifier and not any(char.isdigit() for char in identifier):
                    continue
            try:
                # WHY opaque=True: ``persist`` re-guards every free-text field with
                # ``guard(x, opaque=True)`` via ``validate_fact``, and that enables the
                # long-hex/base64 rule this default guard does NOT. Admitting a locator
                # here that the write will refuse used to mean the row was accepted by the
                # scanner and then zeroed the whole pass (221 of 587 rows, 2026-10). A
                # candidate that would be refused is SKIPPED at admission instead — the
                # same pattern ``carry_rows`` already uses for the same reason.
                guard(locator, opaque=True)
            except MemoryError:
                # Counted, not just dropped: see the docstring.
                skipped += 1
                continue
            counts[locator] += 1
            last[locator] = index
    found = _ranked(counts, last)
    semantic = False
    semantic_count = 0
    for line in summary.splitlines()[:1000]:
        if _HEADINGS.match(line):
            semantic = True
            continue
        if line.startswith("#"):
            semantic = False
        if semantic and line.strip().startswith(("- ", "* ")):
            text = line.strip()[2:202]
            try:
                # Same reason as the locator arm: match the guard the WRITE applies, so a
                # semantic bullet carrying a long hex/base64 run is skipped here instead of
                # being refused (fatally, before the belt in ``persist``) later.
                guard(text, opaque=True)
            except MemoryError:
                skipped += 1
                continue
            found.append({"title": text[:100], "body": text, "tags": ["summary"]})
            semantic_count += 1
            if semantic_count >= 12:
                break
    return CaptureRows(found, skipped=skipped)


class CaptureDetail(NamedTuple):
    """One capture's terminal row plus the coarse attribution stored beside it."""

    status: str
    n: int
    reason: str | None
    refused: int
    skipped: int


def _capture_key(store: MemoryStore, marker_id: str) -> str:
    """The ``meta`` key holding one capture's coarse attribution.

    NOT a ``captures`` column on purpose: that table keeps its exact 4-column shape so
    the previous release's positional ``INSERT INTO captures VALUES (?,?,?,?)`` keeps
    working against a store this build has opened. Running sessions still hold the old
    code after an upgrade, and an added column silently stopped their captures until
    restart (review round 1, MINOR 3). ``meta`` is the store's existing key/value table
    and needs no migration at all.

    ``session``/``compaction`` are the same pair as the table's primary key; the session
    id is a hex slug, so the prefix is unambiguous.
    """
    return f"capture.{store.identity.session_id}.{marker_id}"


def _count(value: Any) -> int:
    """A non-negative count from an untrusted-in-shape meta record; else zero."""
    return value if isinstance(value, int) and value >= 0 else 0


def capture_detail(store: MemoryStore, marker_id: str) -> CaptureDetail | None:
    """The terminal row plus ``reason``/``refused``/``skipped`` from ``meta``.

    One reader for every surface that needs the attribution, so ``hint`` and any future
    status view cannot disagree about what a pass recorded. A malformed meta value
    degrades to "no attribution" rather than breaking the hint: the row itself is the
    truth, and a hint must never cost a turn.
    """
    if not store.path.exists():
        return None
    with store.connection() as db:
        row = db.execute(
            "SELECT status,n FROM captures WHERE session=? AND compaction=?",
            (store.identity.session_id, marker_id),
        ).fetchone()
        if row is None:
            return None
        record = db.execute(
            "SELECT value FROM meta WHERE key=?", (_capture_key(store, marker_id),)
        ).fetchone()
    reason: str | None = None
    refused = 0
    skipped = 0
    if record is not None:
        try:
            parsed = json.loads(record[0])
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, dict):
            if parsed.get("reason") in CAPTURE_REASONS:
                reason = parsed["reason"]
            refused = _count(parsed.get("refused"))
            skipped = _count(parsed.get("skipped"))
    return CaptureDetail(str(row[0]), int(row[1]), reason, refused, skipped)


def watermark(store: MemoryStore, compaction_id: str) -> tuple[str, int] | None:
    detail = capture_detail(store, compaction_id)
    return None if detail is None else (detail.status, detail.n)


def persist(
    store: MemoryStore,
    marker_id: str,
    rows: list[dict[str, Any]],
    status: str,
    committed_at: float | None = None,
    reason: str | None = None,
    skipped: int | None = None,
) -> tuple[str, str | None]:
    """Atomic exact-candidate refresh; never overwrite a correction or resurrect a forget.

    ``reason`` is the coarse cause recorded beside the row, so a pass that captured less
    than everything is attributable later (see :data:`CAPTURE_REASONS`). ``skipped`` is
    the admission-skip count from :func:`extract`; when omitted it is taken from a
    :class:`CaptureRows` result, so a direct caller cannot silently lose it. Write-time
    refusals are counted here. Both counts, and the reason, are stored in ``meta`` under
    :func:`_capture_key` — ``captures`` keeps its 4-column shape for mixed-version
    rollout.

    A pass that captured nothing is recorded explicitly rather than as an ``ok``/0 that
    would hide the outcome (review round 1, MAJOR): ``failed`` when a fault or the write
    belt ended it (a budget stop with no rows, a store that refused every write), and
    ``declined`` when the admission guard skipped everything the scan found. An admission
    skip is a benign decline — nothing the store should have taken was lost — and must
    not be counted as a stage fault beside timeouts and store errors.

    Returns the terminal ``(status, reason)`` actually recorded, so ``finish`` reports
    the same outcome the store holds.
    """
    guard(marker_id)
    if skipped is None:
        skipped = getattr(rows, "skipped", 0)
    if status not in ("ok", "failed", "declined") or len(rows) > 52:
        raise MemoryError("Invalid capture result.")
    if reason is not None and reason not in CAPTURE_REASONS:
        raise MemoryError("Invalid capture reason.")
    if not isinstance(skipped, int) or skipped < 0:
        raise MemoryError("Invalid capture skip count.")
    timestamp = time.time() if committed_at is None else committed_at
    facts = {}
    refused = 0
    for row in rows:
        try:
            fact = validate_fact(
                dict(
                    row,
                    status="candidate",
                    created_at=timestamp,
                    source_time=timestamp,
                    source={
                        "session": store.identity.session_id,
                        "entry": marker_id,
                        "kind": "compaction",
                        "author": "deterministic-capture",
                    },
                )
            )
        except (MemoryError, TypeError, ValueError):
            # BELT at the write. The admit site now matches ``validate_fact``'s stricter
            # guard, but that is a rule the scanner KNOWS; a future field-shape rule it
            # does not yet know would otherwise reach here and — because the loop below
            # is inside the transaction — raise out of ``persist`` and leave a
            # ``failed``/0 row, which is how ONE bad row erased 221 passes. A row shape
            # the store refuses now costs one candidate and is counted. TypeError and
            # ValueError cover ``dict(row, ...)`` itself: a non-mapping row is the same
            # one-candidate contract, not a reason to abort the pass (review NIT 1).
            refused += 1
            continue
        facts[_dump([fact["body"], fact["tags"]])] = fact
    if refused:
        # Count only, never the row: content is untrusted and must not be echoed. The
        # count itself is also recorded in ``meta`` below, so a pass's loss is queryable
        # rather than only a line in a log.
        logger.warning("memory capture skipped %d refused candidate(s)", refused)
    if reason is None and (refused or skipped):
        # Either guard losing a candidate makes the pass partial. ``skipped`` arrived
        # from admission, where the row never reached this function; counting only the
        # write belt is how a stricter ``extract`` rule became invisible. The literal
        # names the guard that fired: sharing one verdict-shaped literal between the two
        # is how 41 admission skips read as write refusals (``refused`` summed to zero
        # across every record). When BOTH fired the write belt is named — it is the
        # later, stricter guard — while ``refused``/``skipped`` in ``meta`` keep both
        # counts.
        reason = "write-refused" if refused else "admission-skip"
    # A moved session's full portable truth must be indexed before a capture
    # exports anything, otherwise the new candidates would replace its old facts.
    store.import_session()
    with store.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        if db.execute(
            "SELECT 1 FROM captures WHERE session=? AND compaction=?",
            (store.identity.session_id, marker_id),
        ).fetchone():
            return status, reason
        origin = store._origin(db)
        seq = db.execute(
            "SELECT coalesce(max(seq),0) FROM ops WHERE origin=?", (origin,)
        ).fetchone()[0]
        previous_clock = db.execute("SELECT value FROM meta WHERE key='clock'").fetchone()
        clock = max(int(time.time() * 1000), int(previous_clock[0]) if previous_clock else 0)
        captured = 0
        for key, fact in facts.items():
            # Same exact body/tags/session, not fuzzy consolidation. Stable
            # entities prevent repeated compactions filling the ACTIVE bank;
            # each committed marker still has a distinct idempotent causal op.
            entity = hashlib.sha256((store.identity.session_id + "|" + key).encode()).hexdigest()
            operation_id = hashlib.sha256((entity + "|" + marker_id).encode()).hexdigest()
            ops = store._ops(db, entity)
            if any(op["kind"] == "tombstone" for op in ops):
                continue
            if any(op["op_id"] == operation_id for op in ops):
                captured += 1
                continue
            heads = store._heads(ops)
            if heads:
                current = heads[0]["fact"]
                # Capture may not resolve conflicts, replace an explicit edit,
                # or let an older slow pass overwrite newer source provenance.
                if (
                    len(heads) != 1
                    or current["status"] != "candidate"
                    or current["source"].get("kind") != "compaction"
                    or (current["source_time"] or 0) > timestamp
                    or len(ops) >= MAX_REVISIONS
                ):
                    continue
                fact["created_at"] = current["created_at"]
            seq += 1
            clock += 1
            op = {
                "v": 1,
                "op_id": operation_id,
                "entity_id": entity,
                "origin_device": origin,
                "seq": seq,
                "hlc": [clock, 0, origin],
                "parents": [heads[0]["op_id"]] if heads else [],
                "kind": "put",
                "scope": "session",
                "owner_key": store.identity.session_id,
                "fact": fact,
                "body_sha": hashlib.sha256(_dump(fact).encode()).hexdigest(),
            }
            validate_op(op)
            db.execute(
                "INSERT INTO ops VALUES (?,?,?,?,?,?,?,?)",
                (
                    operation_id,
                    entity,
                    origin,
                    seq,
                    "session",
                    store.identity.session_id,
                    "put",
                    _dump(op),
                ),
            )
            store._reindex(db, entity)
            captured += 1
        # A pass that captured nothing must not present as a completed capture: the
        # hint would claim success and a health metric would count it as a good pass.
        # An admission-only skip is DECLINED (benign: the guard rejected everything the
        # scan found), while a fault or a write-belt total loss stays FAILED. `quota`
        # and `store-error` normally arrive through ``_record_failure`` with status
        # already `failed`; they are listed here so no path can land an `ok`/0 for them.
        if status == "ok" and not captured:
            if reason == "admission-skip":
                status = "declined"
            elif reason in ("timeout", "write-refused", "quota", "store-error"):
                status = "failed"
        if facts:
            try:
                store._quota(db, "session", store.identity.session_id)
            except MemoryError as exc:
                # Re-raised under a subclass so ``finish`` can record ``quota`` rather than
                # a generic ``store-error``. The message is the store's own (content-free).
                raise _QuotaExceeded(str(exc)) from None
        db.execute("INSERT OR REPLACE INTO meta VALUES ('clock',?)", (str(clock),))
        # Cohort marker for the guard-named split: written once by the first capture
        # this build persists, and never backfilled onto rows written before it. Old
        # rows carry the verdict-shaped literal (or none), so reporting ages them
        # against this stamp instead of inventing a cause for them.
        db.execute(
            "INSERT OR IGNORE INTO meta VALUES (?,?)",
            (ATTRIBUTION_STARTED_KEY, _dump({"at": timestamp})),
        )
        # Attribution lives in ``meta``, not in ``captures``: the 4-column table keeps
        # the previous release's positional INSERT working for sessions still running
        # old code during rollout. Written only when there is something to attribute,
        # so a clean pass adds no meta row.
        if reason is not None or refused or skipped:
            db.execute(
                "INSERT OR REPLACE INTO meta VALUES (?,?)",
                (
                    _capture_key(store, marker_id),
                    _dump({"reason": reason, "refused": refused, "skipped": skipped}),
                ),
            )
        # Explicit columns on the 4-column table: valid against a store the previous
        # release wrote, and against one a 6-column pre-fix build already migrated.
        db.execute(
            "INSERT INTO captures(session,compaction,status,n) VALUES (?,?,?,?)",
            (store.identity.session_id, marker_id, status, captured),
        )
        db.execute(
            "DELETE FROM captures WHERE session=? AND compaction NOT IN "
            "(SELECT compaction FROM captures WHERE session=? ORDER BY rowid DESC LIMIT 32)",
            (store.identity.session_id, store.identity.session_id),
        )
        # Drop attribution whose capture row was just pruned; ``meta`` must not grow
        # without bound beside a table that keeps only the last 32 passes per session.
        prefix = f"capture.{store.identity.session_id}."
        db.execute(
            "DELETE FROM meta WHERE substr(key,1,?)=? AND key NOT IN "
            "(SELECT 'capture.'||session||'.'||compaction FROM captures WHERE session=?)",
            (len(prefix), prefix, store.identity.session_id),
        )
        if store.session_dir:
            store._export(db, store.identity.session_id)
        db.commit()
    return status, reason


#: Rows held for the carry arm inside the shared 52-row capture cap. The pass's
#: carry is its marquee retention (the asks/decisions/identifiers a resumed
#: session cannot re-derive), so it must not be starved by a locator scan that
#: can fill the whole cap alone (review round 2, MINOR 3).
_CARRY_ROW_RESERVE = 12


async def finish(
    store: MemoryStore,
    marker_id: str,
    task: asyncio.Task[Any],
    summary: str,
    committed_at: float | None = None,
    carry: str = "",
) -> str:
    try:
        scan_rows = await asyncio.wait_for(asyncio.shield(task), timeout=SCAN_BUDGET + _SCAN_GRACE)
        # Summary parsing is deterministic, not another provider request. The
        # carry rows are added ONCE (see ``carry_rows``); the digest is NOT also
        # fed through ``extract`` in the same pass, which would persist the same
        # bodies twice under two tag sets. Admission skips travel with each arm
        # (``CaptureRows.skipped``) and are summed before the list concatenation
        # below flattens them into a plain list.
        semantic_rows = await asyncio.to_thread(extract, "", summary)
        skipped = getattr(scan_rows, "skipped", 0) + getattr(semantic_rows, "skipped", 0)
        rows = scan_rows + semantic_rows
        carry_arm = carry_rows(carry) if carry else []
        # The carry is the pass's MARQUEE retention, so it gets its own bounded
        # reserve inside the shared cap instead of sitting behind a locator scan
        # that can fill all 52 rows by itself (measured: ``rows[:52]`` kept 0 of
        # 3 carry rows; review round 2, MINOR 3). The scan keeps the remainder.
        reserved = min(len(carry_arm), _CARRY_ROW_RESERVE)
        scan_rows = rows[: 52 - reserved] if reserved else rows[:52]
        rows = (scan_rows + carry_arm[:_CARRY_ROW_RESERVE])[:52]
        terminal, _ = await asyncio.to_thread(
            persist, store, marker_id, rows, "ok", committed_at, skipped=skipped
        )
        return terminal
    except asyncio.CancelledError:
        raise
    except BudgetExceeded as partial:
        # The scan budget is a DEGRADATION, not a failure: persist what was found and
        # record why the result is partial. The summary arm belongs to a completed
        # pass, but the CARRY is not the work that ran out of budget — it is the
        # pass's marquee retention (the asks/decisions a resumed session cannot
        # re-derive), so it is merged here with the same bounded reserve. Dropping it
        # exactly when the budget is tight lost the highest-value rows (review round
        # 1, MINOR 1). A zero-row stop still lands ``failed``/``timeout`` inside
        # ``persist``.
        carry_arm = carry_rows(carry) if carry else []
        reserved = min(len(carry_arm), _CARRY_ROW_RESERVE)
        scan_keep = partial.rows[: 52 - reserved] if reserved else partial.rows[:52]
        rows = (scan_keep + carry_arm[:_CARRY_ROW_RESERVE])[:52]
        try:
            terminal, _ = await asyncio.to_thread(
                persist,
                store,
                marker_id,
                rows,
                "ok",
                committed_at,
                reason="timeout",
                skipped=getattr(partial.rows, "skipped", 0),
            )
        except Exception as exc:
            # A partial write that could not land is still not a success.
            await _record_failure(
                store,
                marker_id,
                "quota" if isinstance(exc, _QuotaExceeded) else "store-error",
                committed_at,
            )
            return "failed"
        return terminal
    except asyncio.TimeoutError:
        # The await expired before the scan reported (a wedged thread): no rows were
        # observable, so this stays a failure — but recorded as ``timeout``, which is
        # the attribution the meta record exists for.
        await _record_failure(store, marker_id, "timeout", committed_at)
        return "failed"
    except Exception as exc:
        await _record_failure(
            store,
            marker_id,
            "quota" if isinstance(exc, _QuotaExceeded) else "store-error",
            committed_at,
        )
        return "failed"


async def _record_failure(
    store: MemoryStore, marker_id: str, reason: str, committed_at: float | None
) -> None:
    """Write the terminal ``failed`` marker; never raise and never log a payload.

    Failure is a safe terminal state, never a new turn. Nothing here may escape: a
    store that refused the failure marker (full, locked, corrupt) has already cost the
    pass, and ``finish``'s caller only needs the terminal string.
    """
    try:
        await asyncio.to_thread(persist, store, marker_id, [], "failed", committed_at, reason)
    except Exception:
        pass


def recovery_text(transcript: Any, marker: Any) -> str:
    """Reconstruct only bounded pre-cut text; no archive/image decoding or model."""
    kept = marker.payload.get("first_kept_entry_id")
    head: list[str] = []
    tail: list[str] = []
    size = 0
    for entry in transcript.entries():
        if entry.id in (marker.id, kept):
            break
        payload = getattr(entry, "payload", {})
        content = payload.get("content", [])
        for part in content if isinstance(content, list) else []:
            if not isinstance(part, dict) or not isinstance(part.get("text"), str):
                continue
            text = part["text"][:16000]
            if size < 500000:
                head.append(text)
                size += len(text)
            tail.append(text)
            if sum(map(len, tail)) > 500000:
                tail.pop(0)
    return "\n".join(head + tail)[:1000000]


_warned = False


async def session_tail(session: Any) -> str:
    """Observe state only: never wake, start a turn or change prior prompt bytes.

    Advisory memory must never be on the critical path of a provider request: any
    store failure (FTS5 missing, locked, corrupt, disk full) degrades to "no hint".
    The failure is logged ONCE per process at WARNING with the exception type only;
    memory content and paths are untrusted and never logged.
    """
    global _warned
    try:
        return await _session_tail(session)
    except Exception as exc:
        if not _warned:
            _warned = True
            logger.warning("memory prompt tail disabled for this request: %s", type(exc).__name__)
        return ""


async def _session_tail(session: Any) -> str:
    if not getattr(session, "_config_dir", None):
        return ""
    enabled = policy(session._config_dir)
    if not enabled["enabled"] or not session._session_id:
        return ""
    from local_operator.tools.memory_tools import store_for

    store = store_for(session._build_tool_context())
    marker = session._transcript.latest_entry("compaction")
    memory_hint = ""
    if enabled["capture"] and marker is not None:
        memory_hint = await asyncio.to_thread(hint, store, marker.id)
        if not memory_hint and getattr(session, "_memory_terminal", None) == (marker.id, "failed"):
            memory_hint = (
                f'<memory_hint compaction="{marker.id}">Automatic memory capture '
                "did not complete; store key decisions explicitly.</memory_hint>"
            )
        if not memory_hint and getattr(session, "_memory_recovery_marker", None) != marker.id:
            session._memory_recovery_marker = marker.id

            async def recover() -> None:
                text = await asyncio.to_thread(recovery_text, session._transcript, marker)
                task = asyncio.create_task(asyncio.to_thread(extract, text))
                payload = marker.payload
                preserve = payload.get("preserve_data") or {}
                summary = "" if preserve.get("snapcompact") else payload.get("summary", "")
                snap = preserve.get("snapcompact") if isinstance(preserve, dict) else None
                carry_text = str(snap.get("carry") or "") if isinstance(snap, dict) else ""
                terminal = await finish(store, marker.id, task, summary, marker.ts, carry_text)
                session._memory_terminal = (marker.id, terminal)

            session._spawn_background(recover())
    suggestions = ""
    if enabled["suggest"]:
        from local_operator.compaction.tokens import count_text_tokens
        from local_operator.memory import (
            SUGGESTION_TOKENS,
            suggest_line,
            suggestion_block,
        )
        from local_operator.session_factory import _payload_user_text

        user = session._transcript.latest_user_entry()
        query = _payload_user_text(user.payload)[:2048] if user else ""
        # A separate total 180-token budget; titles cannot become instructions.
        # No whole transcript/summary is used as a query and there are no hops.
        try:
            items = await asyncio.to_thread(
                store.search, query, tokens=SUGGESTION_TOKENS, limit=3, suggest=True
            )
        except MemoryError:
            items = []
        if items:
            admitted = ""
            for item in items:
                # The SAME renderer and the SAME wrapper the admission budget in
                # ``MemoryStore.search`` measured against, so the 180-token cap
                # that admitted these entries is the cap the wire enforces — two
                # rulers here is how the budget drifted from the block it was
                # documented to bound (lines-only charging rendered 2 of a
                # documented 3 once titles got long enough for the 19-token
                # wrapper to decide the last admission).
                line = suggest_line(item)
                if count_text_tokens(suggestion_block(admitted + line)) <= SUGGESTION_TOKENS:
                    admitted += line
            # Guard the empty case: ``suggestion_block("")`` still renders the
            # ~19-token wrapper, and a header-only section is strictly worse
            # than shipping nothing. The ``if items`` above and this guard are
            # two rulers over the same admission decision — if they ever
            # diverge (an item filtered to no line), the divergence must
            # degrade to no block, which is the regression this file exists
            # to prevent.
            suggestions = suggestion_block(admitted) if admitted else ""
    return "\n\n".join(x for x in (suggestions, memory_hint) if x)


def hint(store: MemoryStore, marker_id: str) -> str:
    """The post-compaction hint, including WHY a pass was partial or failed.

    ``reason`` used to be written and read by nobody, which is how 221 failed passes
    became unattributable. This is the operator surface that carries it: the hint is
    rendered into the next turn, so a partial capture says so and a total loss says
    which coarse cause ended it. Reasons are literals, never content. A ``declined``
    pass is reported as a benign decline (nothing worth keeping), not as a failure.
    """
    state = capture_detail(store, marker_id)
    if state is None:
        return ""
    if state.status == "ok":
        if state.reason == "timeout":
            text = (
                f"Context compacted; {state.n} candidates captured to session memory "
                "before the scan budget ran out (partial capture). "
                "Search/read memory:// for paths and links, retained asks and decisions. "
                "Store goals and decisions explicitly."
            )
        elif state.reason == "write-refused":
            text = (
                f"Context compacted; {state.n} candidates captured to session memory; "
                "some candidates were withheld by the memory guard at the write. "
                "Search/read memory:// for paths and links, retained asks and decisions. "
                "Store goals and decisions explicitly."
            )
        elif state.reason == "admission-skip":
            text = (
                f"Context compacted; {state.n} candidates captured to session memory; "
                "some candidates were declined by the admission guard. "
                "Search/read memory:// for paths and links, retained asks and decisions. "
                "Store goals and decisions explicitly."
            )
        else:
            text = (
                f"Context compacted; {state.n} candidates captured to session memory. "
                "Search/read memory:// for paths and links, retained asks and decisions. "
                "Store goals and decisions explicitly."
            )
    elif state.status == "declined":
        # A benign decline, deliberately NOT worded as a failure: admission refused
        # everything the scan found, and nothing the store should have taken was lost.
        text = (
            "Context compacted; automatic memory capture declined every candidate "
            "(nothing in the compacted summary was worth keeping as memory). "
            "Store goals and decisions explicitly."
        )
    else:
        cause = {
            "timeout": "the capture scan timed out",
            "write-refused": "every candidate was refused by the memory store",
            "quota": "the memory store is at its quota",
            "store-error": "the memory store raised an error",
        }.get(state.reason or "")
        detail = f" ({cause})" if cause else ""
        text = (
            f"Context compacted; automatic memory capture did not complete{detail}. "
            "Store key decisions explicitly."
        )
    return f'<memory_hint compaction="{marker_id}">\n{text}\n</memory_hint>'
