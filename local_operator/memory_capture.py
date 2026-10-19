"""Best-effort capture is speculative; ONLY a committed marker authorizes writes.

Knowledge first: heading sections (Goal / Constraints / Key Decisions /
Progress / Next Steps / Critical Context) are the primary fact source, from a
context-full summary and from the transcript itself; snapcompact's carry digest
supplies asks/decisions with no inference fork. A locator is never a fact — it is
supporting context on a real memory, and `memory.capture.locators` (default off)
governs even that. Explicit stores remain the guaranteed route for decisions,
especially long runs.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import re
import time
from bisect import bisect_right
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
#: `memory.capture.locators` defaults OFF. Measured on the operator's store (18,571
#: session facts): 13,175 (70.9%) were locator debris, because a locator found by the
#: bounded scan became a standalone fact. Knowledge is the primary source now; this
#: setting only re-enables the CONTEXT arm (a locator attached to a decision/goal in
#: the same markdown section), never a standalone locator fact. Off is the shipped
#: default so a new session's memory is knowledge-first without a code change; the
#: legacy rows stay untouched and are the operator's to forget (`--tag locator`).
LOCATORS_DEFAULT_ENABLED = False
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
#: The compaction prompt is HARNESS SCAFFOLDING, never conversation. A recovery
#: scan and an LLM-history snapshot both see it as an ordinary user message (the
#: pass that produced the marker wrote it). Its template headings and the four
#: description lines under them read exactly like knowledge, so on a real
#: transcript they were captured as goal/decision/progress facts. Dropping the
#: part is the narrow fix: the actual archived conversation is other messages.
_PROMPT_SIGNATURE = "You are summarizing the conversation so far"
#: The prompt's template conditional, seen as a captured "Critical Context"
#: bullet when the prompt block itself was parsed. Bounded literal, not a grammar.
_TEMPLATE_MARKERS = ("{{#if previous_summary}}", "{{/if}}")

#: The semantic headings the compaction prompt asks for (see
#: ``prompts_md/compaction_summary.md``): all six, not four. `Constraints` and
#: `Next Steps` were being silently dropped even though the prompt mandates them
#: (QA m2); the tag vocabulary names each section honestly.
_HEADINGS = re.compile(
    r"^#{1,4}\s*(?P<name>Goal|Constraints|Key Decisions|Progress|Next Steps|Critical Context)\b",
    re.I,
)
_HEADING_TAGS = {
    "goal": "goal",
    "constraints": "constraint",
    "key decisions": "decision",
    "progress": "progress",
    "next steps": "next-step",
    "critical context": "context",
}
#: Bounds for knowledge capture. A compaction summary has a handful of sections;
#: 40 rows is well inside the 52-row persist cap while leaving room for the carry
#: reserve, and a longer transcript section is truncated rather than parsed whole.
_KNOWLEDGE_ROWS = 40
_KNOWLEDGE_BULLET_CHARS = 200
_KNOWLEDGE_PARAGRAPH_CHARS = 400
#: A context suffix rides on the section's own knowledge fact. It stays well under
#: the fact body's 256-token limit (600 chars ~= 150 tokens before title/tags).
_CONTEXT_BODY_CHARS = 600
_CONTEXT_LOCATORS = 8
#: A word, after locators are removed, is what distinguishes a decision/goal line
#: from a bare path. Unicode letters in ANY script: `[^\W\d_]` is a letter (not a
#: digit or underscore), so CJK/Cyrillic/Greek knowledge is not silently English-only
#: (QA Q8). Two characters because scripts such as CJK have one-character words.
_WORD = re.compile(r"[^\W\d_]{2,}")
#: Locator shapes that `_LOCATOR` deliberately does NOT own (that grammar is pinned
#: byte-identical to ``compaction/mine.py``): Windows drive paths, UNC shares, bare
#: domains and `mailto:`. They are context like every other locator, so they must not
#: make a locator-only bullet look substantive (QA m1). Admission-only pattern.
_ADJACENT_LOCATOR = re.compile(
    r"\b[A-Za-z]:[\\/][^\s<>\"']{1,400}"
    r"|\\\\[^\s<>\"']{1,400}"
    r"|(?<![\w@.-])mailto:[^\s<>\"']{0,400}"
    r"|(?<![\w@.-])(?:[A-Za-z0-9-]{1,63}\.)+[A-Za-z]{2,24}(?![\w.-])"
)


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
        "enabled": memory.get("enabled", MEMORY_DEFAULT_ENABLED) is True,
        "capture": memory.get("capture", MEMORY_DEFAULT_ENABLED) is True,
        "suggest": memory.get("suggest", MEMORY_DEFAULT_ENABLED) is True,
        # The setting's KEY is `memory.capture.locators`; its path is flat beside
        # `capture` because `memory.capture` is a BOOL and cannot also hold a
        # mapping. `settings_io` is the authority for the dotted name.
        "locators": memory.get("capture_locators", LOCATORS_DEFAULT_ENABLED) is True,
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


def carry_rows(carry: str, *, limit: int = 12, locators: bool = False) -> list[dict[str, Any]]:
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

    The carry's ``Identifiers`` tier is locators, so it follows the same
    knowledge-first rule as :func:`extract`: OFF unless ``locators`` is set, and
    never a standalone fact even then — the identifiers ride as `— context: …`
    on the first ask/decision the carry also retained. A carry that is nothing
    but identifiers yields ZERO facts.
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
    ):
        for item in items:
            if len(rows) >= limit:
                break
            body = " ".join(item.split())[:280]
            if (
                not body
                or body in seen
                or _PROMPT_SIGNATURE in body
                or any(marker in body for marker in _TEMPLATE_MARKERS)
            ):
                continue
            seen.add(body)
            try:
                # The same fact-shaped admission the WRITE applies (see
                # :func:`_admitted`): a candidate that clears only the body
                # guard can still be refused once ``validate_op`` re-dumps the
                # fact, which used to crash the whole pass on one carry line.
                _admitted(body, tag)
            except MemoryError:
                # A refused candidate must never fail the whole capture: one
                # unguardable candidate skips, the rest of the pass still lands.
                continue
            rows.append({"title": body[:100], "body": body, "tags": [tag]})
    if locators and rows and parsed.identifiers:
        # Identifiers are supporting detail, not facts. They are attached to the
        # first retained ask/decision and never emitted as their own row; a carry
        # with no real memory above admits none of them.
        extra: list[str] = []
        for item in parsed.identifiers[:_CONTEXT_LOCATORS]:
            body = " ".join(item.split())[:280]
            if (
                not body
                or body in seen
                or len(rows[0]["body"]) + len(" — context: ") + len(", ".join(extra + [body]))
                > _CONTEXT_BODY_CHARS
            ):
                continue
            try:
                _admitted(body, "context")
            except MemoryError:
                continue
            seen.add(body)
            extra.append(body)
        if extra:
            candidate = rows[0]["body"] + " — context: " + ", ".join(extra)
            try:
                _admitted(candidate, rows[0]["tags"][0])
            except MemoryError:
                pass
            else:
                rows[0] = dict(rows[0], body=candidate)
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
            if _PROMPT_SIGNATURE in text:
                # Harness scaffolding, not conversation (see _PROMPT_SIGNATURE).
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


def extract_messages(messages: list[Any], *, locators: bool = False) -> list[dict[str, Any]]:
    return extract(snapshot(messages), locators=locators)


class _Section(NamedTuple):
    """One recognised knowledge heading and the half-open span of its content."""

    tag: str
    start: int
    end: int


def _sections(
    source: str,
    *,
    deadline: float | None = None,
    rows: list[dict[str, Any]] | None = None,
    skipped: int = 0,
) -> list[_Section]:
    """Bounded heading scan: the knowledge regions of a summary or transcript.

    A markdown heading that is NOT one of the SIX recognised ones CLOSES the
    region; fenced code is skipped so a `# Key Decisions` line inside a tool
    output or a shell snippet cannot invent a section. The walk is lazy and
    checks the shared deadline every 512 lines, so a line-heavy 2 MB input is
    bounded by TIME, not only by the work it happens to find (QA n2).
    """
    sections: list[_Section] = []
    current: _Section | None = None
    fenced = False
    offset = 0
    for number, line in enumerate(io.StringIO(source)):
        if deadline is not None and number % SCAN_WINDOW == 0 and time.monotonic() > deadline:
            raise BudgetExceeded(CaptureRows(list(rows or ()), skipped=skipped))
        stripped = line.lstrip()
        if stripped.startswith("```"):
            fenced = not fenced
        elif not fenced:
            heading = _HEADINGS.match(line)
            if heading:
                if current is not None:
                    sections.append(current._replace(end=offset))
                tag = _HEADING_TAGS[heading.group("name").lower()]
                current = _Section(tag, offset + len(line), len(source))
            elif re.match(r"^#{1,4}\s", line):
                if current is not None:
                    sections.append(current._replace(end=offset))
                    current = None
        offset += len(line)
    if current is not None:
        sections.append(current)
    return sections


def _without_prompt(source: str) -> str:
    """Drop a logged compaction-prompt block before any heading is trusted.

    ``snapshot`` and ``recovery_text`` already drop the prompt PART; this covers
    a direct ``extract`` call that hands over a prompt string whole (the template
    repeats the knowledge headings, so its description lines otherwise become
    goal/decision facts). The conversation the prompt summarizes is separate
    messages, never a suffix of the prompt itself.
    """
    index = source.find(_PROMPT_SIGNATURE)
    return source if index < 0 else source[:index]


def _substantive(body: str) -> bool:
    """Whether a candidate still says something after its locators are removed.

    This is the admission rule that makes a transcript of URLs and paths yield
    ZERO facts: `- ~/tools/x/proof.md` is a locator with no memory around it,
    while `- Store evidence at ~/tools/x/proof.md` is a decision whose path is
    supporting detail. Locator grammar is removed first, then the adjacent shapes
    `_LOCATOR` deliberately does not own (Windows/UNC, bare domains, mailto) —
    they are context too (QA m1). A candidate with nothing left is COUNTED at
    admission, never silently dropped as an `ok`/0 pass (QA Q8).
    """
    remaining = _LOCATOR.sub(" ", body)
    remaining = _ADJACENT_LOCATOR.sub(" ", remaining)
    return bool(_WORD.search(remaining))


def _admitted(body: str, tag: str) -> None:
    """Raise ``MemoryError`` when the WRITE's own shape check would refuse the row.

    ``guard(body, opaque=True)`` is not sufficient on its own. ``persist`` builds a
    fact (defaults plus source) and ``validate_op`` re-dumps THAT mapping; a
    credential-shaped run can appear only across the JSON boundary once those
    keys are present. Measured on a real carry line containing the literal
    ``secret: <value>``: the body guard passed, then ``validate_op`` raised and
    took the whole pass down with zero rows. Probing the same ``validate_fact``
    output the write will re-dump closes that gap at admission, where the
    candidate is one skipped row instead of a lost pass.
    """
    guard(body, opaque=True)
    probe = validate_fact(
        {
            "title": body[:100],
            "body": body,
            "tags": [tag],
            "status": "candidate",
            "created_at": 0,
            "source_time": 0,
            "source": {"session": "probe", "entry": "probe", "kind": "compaction"},
        }
    )
    guard(_dump(probe), opaque=True)


def _emit(
    body: str,
    tag: str,
    rows: list[dict[str, Any]],
    seen: set[str],
    skipped: int,
    section_of_row: dict[int, int] | None,
    section_index: int,
) -> int:
    """Admit one knowledge candidate; returns the updated admission-skip count."""
    if len(rows) >= _KNOWLEDGE_ROWS:
        return skipped
    body = " ".join(body.split())
    if not body or body in seen:
        return skipped
    if not _substantive(body):
        # A section of bare locators (or text no script rule can judge) is an
        # ADMISSION skip and is counted. Before Q8 this returned silently, so a
        # whole non-Latin knowledge section landed as `ok`/0 with no attribution.
        return skipped + 1
    if _PROMPT_SIGNATURE in body or any(marker in body for marker in _TEMPLATE_MARKERS):
        return skipped
    try:
        # Same guard the WRITE applies: a candidate whose body — or whose
        # fact-shaped dump — would be refused is skipped here instead of aborting
        # the pass later.
        _admitted(body, tag)
    except MemoryError:
        return skipped + 1
    seen.add(body)
    if section_of_row is not None:
        section_of_row[len(rows)] = section_index
    rows.append({"title": body[:100], "body": body, "tags": [tag]})
    return skipped


def _knowledge_rows(
    source: str,
    rows: list[dict[str, Any]],
    seen: set[str],
    skipped: int,
    deadline: float,
    section_of_row: dict[int, int] | None = None,
) -> tuple[list[_Section], int]:
    """Append knowledge facts from recognised heading sections, bounded.

    Bullets are atomic facts (the old semantic arm's shape). Prose under a heading
    is captured too: the compaction prompt does not require bullets, so a
    context-full summary whose Goal/Progress is a paragraph would otherwise
    contribute nothing. Locator-only lines are not facts here — they may only be
    attached as context to a section fact (see :func:`_attach_context`).
    """
    sections = _sections(source, deadline=deadline, rows=rows, skipped=skipped)
    for index, section in enumerate(sections):
        if len(rows) >= _KNOWLEDGE_ROWS:
            break
        paragraph: list[str] = []
        length = 0
        fenced = False

        def flush() -> None:
            nonlocal skipped, paragraph, length
            if paragraph:
                body = " ".join(paragraph)[:_KNOWLEDGE_PARAGRAPH_CHARS]
                skipped = _emit(body, section.tag, rows, seen, skipped, section_of_row, index)
            paragraph = []
            length = 0

        for number, line in enumerate(io.StringIO(source[section.start : section.end])):
            # Every 64 lines is frequent enough for a 2 s budget on the bounded
            # input and coarse enough that normal summaries never pay for it.
            if number % 64 == 0 and time.monotonic() > deadline:
                flush()
                raise BudgetExceeded(CaptureRows(list(rows), skipped=skipped))
            stripped = line.strip()
            if stripped.startswith("```"):
                # A fenced block is DATA (a tool result or a snippet), not the
                # section's own prose. Its locators remain eligible as context,
                # but its lines never become facts: that is how a heading over
                # pasted output stops minting junk paragraphs.
                flush()
                fenced = not fenced
                continue
            if fenced:
                continue
            if not stripped:
                flush()
                continue
            if stripped.startswith(("- ", "* ")):
                flush()
                skipped = _emit(
                    stripped[2:][:_KNOWLEDGE_BULLET_CHARS],
                    section.tag,
                    rows,
                    seen,
                    skipped,
                    section_of_row,
                    index,
                )
                continue
            if len(stripped) > 2000:
                # Tool output or a pasted document, not a summarised sentence.
                flush()
                continue
            paragraph.append(stripped)
            length += len(stripped) + 1
            if length >= _KNOWLEDGE_PARAGRAPH_CHARS:
                flush()
        flush()
    return sections, skipped


def _attach_context(
    text: str,
    sections: list[_Section],
    rows: list[dict[str, Any]],
    section_of_row: dict[int, int],
    deadline: float,
    skipped: int,
) -> int:
    """Attach in-section locators to that section's knowledge fact.

    The legacy sweep minted every locator as its own fact, which is where 70.9%
    of the measured store came from. Here a locator is only considered when it
    falls inside a recognised heading section, and it is never stored alone: it
    is appended to the section's first knowledge fact as `— context: …`. A
    section with no fact (e.g. a bare list of paths) contributes nothing even
    when the setting is on, which is the adjacency rule the setting cannot
    disable.

    The windowing and per-token guards are UNCHANGED from the legacy sweep: 4096-
    char windows, the 2 s budget checked between them, the forced-cut guard, and
    the 200-char locator cap. Only the admission destination changed.
    """
    if not sections or not section_of_row:
        return skipped
    starts = [section.start for section in sections]
    buckets: dict[int, list[str]] = {}
    seen_locators: dict[int, set[str]] = {}
    offset = 0
    for window in windows(text):
        if time.monotonic() > deadline:
            raise BudgetExceeded(CaptureRows(list(rows), skipped=skipped))
        start = offset
        offset += len(window)
        base = start
        if start and not text[start - 1].isspace():
            # A forced window cut can bisect an oversized URI/path. Do not
            # reinterpret its continuation as an independent locator; traversal
            # is still bounded by the unchanged scan window, with no I/O.
            prefix = next((i for i, char in enumerate(window) if char.isspace()), len(window))
            window = window[prefix:]
            base = start + prefix
        for match in _LOCATOR.finditer(window):
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
            position = base + match.start()
            index = bisect_right(starts, position) - 1
            if index < 0 or position >= sections[index].end:
                # Outside every knowledge section: never a candidate. This is what
                # makes a URL-only transcript yield zero, with the setting on too.
                continue
            try:
                _admitted(locator, "context")
            except MemoryError:
                skipped += 1
                continue
            bucket = buckets.setdefault(index, [])
            seen_here = seen_locators.setdefault(index, set())
            if locator in seen_here or len(bucket) >= _CONTEXT_LOCATORS:
                continue
            seen_here.add(locator)
            bucket.append(locator)
    for index, locators in buckets.items():
        row_index = next((n for n, section in section_of_row.items() if section == index), None)
        if row_index is None or row_index >= len(rows):
            continue
        body = rows[row_index]["body"]
        # Skip anything a fact in THIS section already says: appending a locator
        # that another section fact carries would duplicate it onto the first
        # one, which is the two-rulers mistake this pass exists to avoid.
        section_bodies = " ".join(
            rows[n]["body"] for n, section in section_of_row.items() if section == index
        )
        extra = [locator for locator in locators if locator not in section_bodies]
        while (
            extra and len(body) + len(" — context: ") + len(", ".join(extra)) > _CONTEXT_BODY_CHARS
        ):
            extra.pop()
        if not extra:
            continue
        candidate = body + " — context: " + ", ".join(extra)
        try:
            _admitted(candidate, rows[row_index]["tags"][0])
        except MemoryError:
            continue
        # Title stays the knowledge statement; the locator is context in the body.
        rows[row_index] = dict(rows[row_index], body=candidate)
    return skipped


def extract(text: str, summary: str = "", *, locators: bool = False) -> CaptureRows:
    """CPU-only bounded scan; returns values, never opens a store or writes state.

    KNOWLEDGE FIRST. The semantic heading sections (``Goal``/``Key Decisions``/
    ``Progress``/``Critical Context``) are the primary source of facts, from the
    compaction summary and from the transcript itself. The transcript's summary
    is parsed FIRST so its curated wording wins a de-duplication tie against the
    same sentence in raw history. A locator is never a fact: it may only ride as
    `— context: …` on a section fact, and only when ``locators`` is true. This is
    the measured fix — the old sweep minted every URL/path/id as its own fact and
    70.9% of the operator's session store was exactly that debris.

    Raises :class:`BudgetExceeded` when the scan budget runs out, carrying the rows found
    up to that point: running out of time is a degraded result, never a lost one.

    The returned list carries ``skipped``, the number of candidates the admission guard
    refused. They are counted rather than silently dropped: an admission rule stricter
    than the write's would otherwise shrink a pass with no trace anywhere (review round
    1, MINOR 2).
    """
    started = time.monotonic()
    deadline = started + SCAN_BUDGET
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    skipped = 0
    # The scan cap and the 1,000-line summary cap are unchanged from the legacy
    # scan; only what gets admitted out of them changed.
    summary_text = _without_prompt("\n".join(summary.splitlines()[:1000]))
    _summary_sections, skipped = _knowledge_rows(summary_text, rows, seen, skipped, deadline)
    section_of_row: dict[int, int] = {}
    text = _without_prompt(text[:2000000])
    text_sections, skipped = _knowledge_rows(text, rows, seen, skipped, deadline, section_of_row)
    # Locator work exists only when there is a real memory to attach it to. With
    # no knowledge facts the scan returns immediately — a URL-only transcript
    # costs no locator walk at all, not even with the setting on.
    if locators and rows and text_sections:
        skipped = _attach_context(text, text_sections, rows, section_of_row, deadline, skipped)
    else:
        # Keep the TIME bound over the input even when no knowledge section and
        # no locator walk consumed it. The scan is bounded by the budget, not
        # only by whatever work it happened to find, so a flat transcript still
        # stops at the same ceiling as a heading-rich one.
        for _ in windows(text):
            if time.monotonic() > deadline:
                raise BudgetExceeded(CaptureRows(rows[:_KNOWLEDGE_ROWS], skipped=skipped))
    return CaptureRows(rows[:_KNOWLEDGE_ROWS], skipped=skipped)


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


def _meta_fields(value: Any) -> tuple[str | None, int, int]:
    """``(reason, refused, skipped)`` from one stored attribution value.

    One parser for every reader, so ``capture_detail`` and ``persist``'s duplicate
    path cannot disagree about what a record says. A reason outside
    :data:`CAPTURE_REASONS` — ``refused-row`` from a pre-split build — is reported as
    ``None`` because it named no guard, and a malformed value degrades to no
    attribution rather than breaking the hint: the row itself is the truth.
    """
    try:
        parsed = json.loads(value) if value is not None else None
    except (TypeError, ValueError):
        parsed = None
    if not isinstance(parsed, dict):
        return None, 0, 0
    reason = parsed.get("reason")
    return (
        reason if reason in CAPTURE_REASONS else None,
        _count(parsed.get("refused")),
        _count(parsed.get("skipped")),
    )


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
        reason, refused, skipped = _meta_fields(record[0])
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
        existing = db.execute(
            "SELECT status FROM captures WHERE session=? AND compaction=?",
            (store.identity.session_id, marker_id),
        ).fetchone()
        if existing is not None:
            # A duplicate pass must report the terminal pair the store ALREADY holds:
            # ``declined``/``failed`` are derived below from what the pass wrote, so
            # echoing the caller's requested status here would contradict both the
            # row a resumed caller reads next and this function's returned contract.
            # The reason is read back for the same reason.
            record = db.execute(
                "SELECT value FROM meta WHERE key=?", (_capture_key(store, marker_id),)
            ).fetchone()
            stored_reason, _, _ = _meta_fields(record[0] if record is not None else None)
            return str(existing[0]), stored_reason
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
        # Guard losses and stage faults must not present as a completed capture: the
        # hint would claim success and a health metric would count them as good passes.
        # An admission-only skip is DECLINED (benign: the guard rejected everything the
        # scan found), while a fault or a write-belt total loss stays FAILED. `quota`
        # and `store-error` normally arrive through ``_record_failure`` with status
        # already `failed`; they are listed here so no guarded/faulted path lands
        # `ok`/0. A pass whose facts were all dropped by a tombstone or supersede
        # ``continue`` is NOT a loss — a forget/edit was respected — and stays `ok`/0.
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
        # old code during rollout. This attribution row is written only when there is
        # something to attribute; the cohort stamp above is not per-pass attribution.
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
    # Read outside the try so every exit (including the budget-partial one) uses
    # the ONE policy answer this pass started with; a torn config read falls back
    # to the shipped default rather than re-deciding mid-pass.
    locators = policy(store.root.parent)["locators"]
    try:
        scan_rows = await asyncio.wait_for(asyncio.shield(task), timeout=SCAN_BUDGET + _SCAN_GRACE)
        # Summary parsing is deterministic, not another provider request. The
        # carry rows are added ONCE (see ``carry_rows``); the digest is NOT also
        # fed through ``extract`` in the same pass, which would persist the same
        # bodies twice under two tag sets. Admission skips travel with each arm
        # (``CaptureRows.skipped``) and are summed before the list concatenation
        # below flattens them into a plain list.
        semantic_rows = await asyncio.to_thread(extract, "", summary, locators=locators)
        skipped = getattr(scan_rows, "skipped", 0) + getattr(semantic_rows, "skipped", 0)
        # SUMMARY FIRST. The semantic branch is the primary knowledge source, and
        # the 52-row persist cap below truncates from the END: a transcript scan
        # that filled its own 40-row cap would otherwise push every curated
        # summary fact out of the pass. Within each arm the same 40-row cap applies.
        rows = semantic_rows + scan_rows
        carry_arm = carry_rows(carry, locators=locators) if carry else []
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
        carry_arm = carry_rows(carry, locators=locators) if carry else []
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
            if _PROMPT_SIGNATURE in text:
                # The compaction prompt logged in the prefix is scaffolding; its
                # template headings must never become memories.
                continue
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
                task = asyncio.create_task(
                    asyncio.to_thread(extract, text, locators=enabled["locators"])
                )
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
                "Search/read memory:// for retained asks, decisions and goals. "
                "Store goals and decisions explicitly."
            )
        elif state.reason == "write-refused":
            text = (
                f"Context compacted; {state.n} candidates captured to session memory; "
                "some candidates were withheld by the memory guard at the write. "
                "Search/read memory:// for retained asks, decisions and goals. "
                "Store goals and decisions explicitly."
            )
        elif state.reason == "admission-skip":
            text = (
                f"Context compacted; {state.n} candidates captured to session memory; "
                "some candidates were declined by the admission guard. "
                "Search/read memory:// for retained asks, decisions and goals. "
                "Store goals and decisions explicitly."
            )
        else:
            # A pre-split record's reason reads None because ``refused-row`` named no
            # guard; its stored counts still prove the pass was partial. Report that
            # evidence without naming a guard: an unqualified success here would hide
            # the withheld candidates on exactly the records that can never be
            # reclassified (review round 1, MINOR 1).
            withheld = state.refused + state.skipped
            caveat = (
                f"; {withheld} candidates were withheld (cause not recorded)" if withheld else ""
            )
            text = (
                f"Context compacted; {state.n} candidates captured to session memory"
                f"{caveat}. "
                "Search/read memory:// for retained asks, decisions and goals. "
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
