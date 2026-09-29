"""Checkpoint naming — a short name and summary for a completed conversation turn.

The checkpoint rail's hover card wants more than the raw text of a turn: a
3-6 word name and one sentence saying what the turn accomplished. That is a
model call, and this module owns the whole of it — what gets asked, how the
answer is parsed, when a call is spent, how the result is cached, and what
state the manifest reports while a call is pending or after one fails.

WHAT GOVERNS THE DESIGN, each because the obvious implementation gets it wrong:

- **Never eager.** A name is bought by a user GESTURE (the rail opening, a
  hover over a tick), never for every turn — eager naming is LLM spend without
  an ask. :func:`warm_checkpoints` is the only entry point, and it is bounded:
  at most :data:`MAX_WARM_IDS` ids per call.
- **Never a turn's problem.** Mirroring ``session/naming.py``: every failure
  resolves to "no name", never raises, and one 15 s budget bounds the whole
  call. A checkpoint whose name never arrives costs the card its name and
  nothing else — the rail still draws and the jump still works.
- **A bad name is worse than no name.** Over-long answers are REJECTED, not
  truncated (the naming module's rule), and the model is offered a sentinel
  for "nothing here is worth naming" so noise turns do not mint names.
- **One call per checkpoint, cached.** The name is stored in the transcript
  index's ``naming`` section keyed by the TURN KEY — the opening user entry
  id, stable across re-settles and tail extensions — together with the hash
  of the digest the answer was generated from. A warm regenerates only when
  that hash changed (the turn grew) or the section's prompt version moved.
- **Failure is a state, not silence.** A failed call persists
  ``{state: "unavailable", failed_ts, ...}`` under the turn key, and warm
  skips that turn for :data:`NAMING_UNAVAILABLE_COOLDOWN_S`. The manifest
  derives its ``naming.state`` from the same section through
  :func:`naming_state`, so "unavailable" is visible to the rail (it can stop
  polling) from a process that is not the owner — the manifest route reads
  the cache file, the generation runs where the session does.

WHERE IT RUNS. Generation has to execute where the session lives: provider
credentials and the errand tier are session state, and a viewer facade refuses
errands outright ("provider errands run on the session owner"). The desktop
route therefore routes ``sessions.checkpoints.warm`` to the serving runtime
over the shared-slash seam, and :func:`warm_checkpoints` schedules its work as
background tasks on that runtime's loop — holding a strong reference itself,
because a bare task is only weakly held by the loop and can be collected
before it runs.

TOOL-ACTION COUNTS, stated because the design's digest sketch names them: the
shipped index stores none (tool rows are classified and dropped by the
scanner, and no record it keeps carries a count), so the digest carries the
user text and the final answer. Recorded as a slice deviation rather than
inventing a heuristic that would mislabel its own input.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from local_operator.session import transcript_index

# Borrowed, not re-implemented: these are the thinking-envelope and
# untagged-reply disciplines ``parse_title`` is built from, and a second copy
# of them here would drift from the parser the session titles use — a leaked
# ``<think>`` block must strip identically on both surfaces or a checkpoint
# name becomes whatever the model was muttering to itself. Promote them to
# public names (or a shared ``naming_parse`` leaf) if a third consumer
# appears.
from local_operator.session.naming import (
    _LOGGED_ERROR_CHARS,
    _QUOTE_CHARS,
    _THINKING_PREAMBLE_RE,
    _TRAILING_PUNCTUATION,
    _cut_unclosed_thinking,
    _sentence_case,
    _strip_thinking_envelopes,
    _untagged_candidate,
    cut_on_a_word,
)

logger = logging.getLogger(__name__)

#: How long ONE checkpoint naming call may run before it is abandoned. The
#: budget mirrors ``naming.TITLE_TIMEOUT_S``'s reasoning (2.5x the slowest
#: title call measured, 5.4-5.8 s against the real providers): a decoration
#: nobody is waiting on that has not answered in 15 s has nothing left to
#: win. Single attempt — there is no retry behind it, so this is the whole
#: budget.
CHECKPOINT_NAME_TIMEOUT_S = 15.0

#: How long a failed turn is skipped before warm may try it again. A provider
#: that just failed usually fails again immediately, so a hover loop must not
#: re-spend on every poll; ten minutes keeps recovering within a session's
#: life. The marker is PERSISTED (see the module docstring), so a runtime
#: restart inside the window does not re-spend either.
NAMING_UNAVAILABLE_COOLDOWN_S = 600.0

#: Bound on ids one warm call may own, and the default selection size when the
#: caller names no ids (D9: ``ids`` max 16; D2: the rail's open gesture warms
#: the most recent <= 8 checkpoints missing names).
MAX_WARM_IDS = 16
DEFAULT_WARM_LIMIT = 8

#: Budget on the condensed turn digest. 2,000 characters is roughly 500 input
#: tokens — enough for the ask, the answer and their labels, and still the
#: same cost class as a title call. The per-section budgets below are chosen
#: so their sum cannot exceed it (6 + 900 + 1 + 8 + 900 + margin).
DIGEST_MAX_CHARS = 2000
_USER_TEXT_BUDGET = 900
_ANSWER_TEXT_BUDGET = 900

#: Caps on a name, enforced as REJECTION (see the module docstring). The ask
#: is 3-6 words; 8 words is the review-copy ceiling (a nine-word "name" is a
#: sentence someone has to sit through), and 80 characters matches
#: ``naming.MAX_TITLE_CHARS`` so the two naming surfaces agree about when a
#: label has stopped being a label.
MAX_NAME_WORDS = 8
MAX_NAME_CHARS = 80

#: Cap on the summary, enforced by the one "shorten" definition in the
#: product (``naming.cut_on_a_word``). A summary is prose, so cutting it on a
#: word boundary reads as a shortening; the NAME is rejected instead when it
#: is over its caps, because a name cut mid-word reads as a bug.
SUMMARY_MAX_CHARS = 160

#: The system block for a checkpoint naming call. Terse like
#: ``naming.TITLE_SYSTEM_PROMPT`` and for the same reason: it rides every call
#: and is the half of the request we control, so every clause earns its
#: tokens. The ``<name/>`` sentinel exists so "nothing worth naming" is
#: answerable rather than malformed — without it, models invent a name for
#: "ok, thanks".
CHECKPOINT_NAME_SYSTEM_PROMPT = (
    "Name this completed exchange from a conversation.\n"
    "Reply with only <name>3 to 6 words</name>"
    "<summary>one sentence, at most 160 characters</summary>.\n"
    "Nothing worth naming: reply exactly <name/>.\n"
    "No quotes, no trailing punctuation."
)

_NAME_TAG_RE = re.compile(r"<name\s*>(.*?)</name\s*>", re.IGNORECASE | re.DOTALL)
_EMPTY_NAME_RE = re.compile(r"<name\s*/\s*>", re.IGNORECASE)
#: Stray / unclosed ``<name>`` fragments, stripped before the untagged path so
#: a truncated ``<name>the login fix`` never stores the markup (parse_title's
#: rule, same regex shape).
_STRAY_NAME_TAG_RE = re.compile(r"</?name\s*/?>", re.IGNORECASE)
_SUMMARY_TAG_RE = re.compile(r"<summary\s*>(.*?)</summary\s*>", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class CheckpointName:
    """One parsed answer: the name, plus its (possibly empty) summary."""

    name: str
    summary: str


def _normalise_name_body(body: str) -> str | None:
    """Cap / quote / punct rejection for a candidate name (parse_title's rule).

    A name that fails any cap is REJECTED, never truncated — see the module
    docstring. Sentence case only lifts an all-lower-case first word, so a
    name like "gRPC startup crash" keeps its casing.
    """
    first_line = next((line for line in body.splitlines() if line.strip()), "")
    cleaned = " ".join(first_line.split()).strip(_QUOTE_CHARS + " ")
    cleaned = cleaned.rstrip(_TRAILING_PUNCTUATION).strip()
    # Strip once more: a quoted name with trailing punctuation leaves a stray
    # quote after the punctuation pass (parse_title's measured case).
    cleaned = cleaned.strip(_QUOTE_CHARS + " ")
    if not cleaned:
        return None
    if len(cleaned) > MAX_NAME_CHARS:
        return None
    words = cleaned.split()
    if len(words) > MAX_NAME_WORDS:
        return None
    return _sentence_case(words)


def _normalise_summary_body(body: str) -> str:
    """A summary is best-effort: absent or malformed yields ``""``.

    The name is the load-bearing half of the card, so a bad summary never
    rejects a good name; over-long prose is cut with the product's one
    shortening rule rather than discarded, because half of a summary still
    tells the reader what the turn accomplished.
    """
    cleaned = " ".join(body.split()).strip(_QUOTE_CHARS + " ")
    if not cleaned:
        return ""
    return cut_on_a_word(cleaned, SUMMARY_MAX_CHARS)


def parse_checkpoint(raw: str) -> CheckpointName | None:
    """Extract and normalise a name+summary from a naming call's raw reply.

    Sibling of :func:`naming.parse_title`, and it inherits that parser's
    policies: the last *visible* marked ``<name>`` wins (a draft tag before
    the real one is common, and a tag inside a thinking envelope was already
    stripped or cut); an unclosed ``<think>`` envelope discards the reply
    unless a closed visible name already won; over-cap names are rejected.

    ``None`` means "no name from this reply" — the model's own ``<name/>``
    sentinel, an empty or unparseable answer, or a name over its caps. The
    caller folds all of them onto the same unavailable state: a retry cannot
    turn a decline into a name, and one dead reply is not worth a second
    spend.
    """
    if not raw:
        return None
    visible = _cut_unclosed_thinking(_strip_thinking_envelopes(raw))
    matches = list(_NAME_TAG_RE.finditer(visible))
    if matches:
        name = _normalise_name_body(_unwrap_json_name(matches[-1].group(1)))
        if name is None:
            return None
        summary = ""
        summaries = list(_SUMMARY_TAG_RE.finditer(visible))
        if summaries:
            summary = _normalise_summary_body(summaries[-1].group(1))
        return CheckpointName(name=name, summary=summary)
    if _EMPTY_NAME_RE.search(visible):
        # The model's own "nothing worth naming": a decline, and the caller
        # treats it exactly like a failure for retry purposes (see above).
        return None
    # Untagged path: non-Anthropic models (Grok, DeepSeek, Kimi, most local
    # OpenAI-compat servers) often answer without the tags, so rejecting
    # those replies is how they would never name a checkpoint at all. The
    # NAME is recovered the way parse_title recovers its title; the summary
    # is NOT guessed — a wrong split would mislabel half the card, and a
    # missing summary is a state the card already renders.
    stripped = _STRAY_NAME_TAG_RE.sub("", visible)
    stripped = _SUMMARY_TAG_RE.sub("", stripped)
    if _THINKING_PREAMBLE_RE.search(stripped.lstrip()):
        return None
    candidate = _untagged_candidate(stripped).strip()
    name = _normalise_name_body(_unwrap_json_name(candidate))
    if name is None:
        return None
    return CheckpointName(name=name, summary="")


def _unwrap_json_name(candidate: str) -> str:
    """``{"name": "..."}`` (optionally fenced) -> the inner string.

    The two-part answer is a structured shape, and some models emit the JSON
    they were trained on for it instead of the tags; without this the raw
    JSON is what gets normalised (usually rejected as over-long, which is how
    a shape-ignoring model would never produce a name). Only the ``name`` key
    is unwrapped — the summary around it stays untrusted.
    """
    text = candidate.strip()
    fenced = re.match(r"^```[^\n]*\s*(.*?)\s*```$", text, re.IGNORECASE | re.DOTALL)
    if fenced is not None and fenced.group(1).strip().startswith("{"):
        text = fenced.group(1).strip()
    if not text.startswith("{"):
        return candidate
    quoted = re.search(r'"name"\s*:\s*("(?:[^"\\]|\\.)*")', text)
    if quoted is None:
        return candidate
    import json

    try:
        salvaged = json.loads(quoted.group(1))
    except json.JSONDecodeError:
        return candidate
    return salvaged.strip() if isinstance(salvaged, str) else candidate


def _condense(text: str) -> str:
    """Collapse whitespace — the digest reads as one flowing prompt, not rows."""
    return " ".join((text or "").split())


def build_turn_digest(index: "transcript_index.TranscriptIndex", turn: int) -> str | None:
    """The condensed prompt input for one turn, or ``None`` when there is none.

    Built from the index's stored docs — never from the journal, which this
    module does not read: the turn's user text and its final answer, each
    bounded by its budget and cut on a word boundary, labelled so the model
    can tell the ask from the result. The output is a pure function of the
    two stored strings (same content, same digest, same hash), which is what
    makes the cache's hash-comparison meaningful across processes and
    restarts.
    """
    user = next(
        (c for c in index.checkpoints if c.kind == transcript_index.KIND_USER and c.turn == turn),
        None,
    )
    completion = next(
        (
            c
            for c in index.checkpoints
            if c.kind == transcript_index.KIND_COMPLETION and c.turn == turn
        ),
        None,
    )
    if user is None or completion is None:
        return None
    user_text = _condense(user.text)
    answer_text = _condense(completion.text)
    if not user_text and not answer_text:
        # Nothing to name: a turn whose rows carried no text at all. Do not
        # spend a call asking a model to name emptiness.
        return None
    parts = []
    if user_text:
        parts.append("User: " + cut_on_a_word(user_text, _USER_TEXT_BUDGET))
    if answer_text:
        parts.append("Answer: " + cut_on_a_word(answer_text, _ANSWER_TEXT_BUDGET))
    digest = "\n".join(parts)
    # Belt and braces: the section budgets already keep this under the cap,
    # but a future edit to either budget must not silently grow the prompt.
    return digest[:DIGEST_MAX_CHARS]


def _digest_hash(digest: str) -> str:
    """The cache's invalidation key: sha256 of the digest, full hex.

    Full length rather than truncated: it is an equality check, not an index,
    and there is no reason to hand a cost-bearing decision a narrower key.
    """
    return hashlib.sha256(digest.encode("utf-8")).hexdigest()


def _failure_cooling(item: Any, *, now: float | None = None) -> bool:
    """Whether ``item``'s last failed attempt is still inside its cooldown.

    ONE definition for both readers of that window: :func:`naming_state`
    (which reports ``unavailable`` for a turn that never got a name) and
    warm's stale-name arm (which must not re-spend on a NAMED item whose
    regeneration just failed — the last good pair keeps being served, and the
    retry waits out the window exactly as a fresh name's does).
    """
    if not isinstance(item, dict):
        return False
    failed = item.get("failed_ts")
    if not isinstance(failed, (int, float)) or isinstance(failed, bool):
        return False
    moment = time.time() if now is None else now
    return moment - float(failed) < NAMING_UNAVAILABLE_COOLDOWN_S


def naming_state(item: Any, *, now: float | None = None) -> str:
    """The manifest's naming state for one ``naming.items`` entry.

    One of ``"ready" | "pending" | ``"unavailable"`` — the enum the desktop
    models ship (``CheckpointNamingState``). This is the single source of
    truth for BOTH readers of that state: the manifest builder
    (``transcript_index``) and warm's own skip decision, so a turn cannot
    read "unavailable" on the card while the next warm happily re-spends on
    it, or vice versa.

    ``"ready"`` = a name is present; the digest hash is NOT consulted here,
    and neither is ``failed_ts`` — a NAMED item whose regeneration failed
    keeps showing its last good pair (the failure rides there only as warm's
    retry gate; see :func:`_failure_cooling`). The manifest shows the name it
    has, and staleness is warm's business (it compares the hash, regenerates,
    and overwrites). ``"unavailable"`` = the last attempt failed inside
    :data:`NAMING_UNAVAILABLE_COOLDOWN_S` and NO name was ever bought; once
    the window passes, the same item reads "pending" again and a warm may
    retry it. Everything else is ``"pending"``.
    """
    if isinstance(item, dict) and item.get("name"):
        return "ready"
    if _failure_cooling(item, now=now):
        return "unavailable"
    return "pending"


@dataclass(frozen=True)
class _TurnTarget:
    """One resolved unit of naming work.

    ``request_id`` is the id the CALLER understands (echoed back in the
    response so the rail can poll what it asked for); ``turn_key`` is the
    cache key — the turn's opening user entry id (D2).
    """

    request_id: str
    turn_key: str
    turn: int
    digest: str


@dataclass
class _SessionState:
    """Per-session warm bookkeeping (module state; loop thread only).

    ``lock`` enforces per-session concurrency 1 (D2): a rail opening on a
    long conversation may accept several turns at once, and provider errands
    are serialized so one session does not open a burst of simultaneous
    requests. ``active`` doubles as the strong-reference set — a bare task is
    only weakly held by the loop and can be collected before it runs.
    """

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    active: dict[str, "asyncio.Task[None]"] = field(default_factory=dict)


_STATE: dict[tuple[str, str], _SessionState] = {}


def _session_state(config_dir: str | Path, session_id: str) -> _SessionState:
    key = (str(config_dir), session_id)
    state = _STATE.get(key)
    if state is None:
        state = _SessionState()
        _STATE[key] = state
    return state


def _naming_items(index: "transcript_index.TranscriptIndex") -> dict[str, Any]:
    """The servable ``naming.items`` mapping, or ``{}``.

    Mirrors the manifest's own gate: a section written under another prompt
    version is not served (``transcript_index`` owns that check for both
    consumers).
    """
    naming = index.naming if isinstance(index.naming, dict) else {}
    items = (
        naming.get("items")
        if naming.get("prompt_version") == transcript_index.NAMING_PROMPT_VERSION
        else {}
    )
    return items if isinstance(items, dict) else {}


def _targets_from_ids(
    index: "transcript_index.TranscriptIndex", ids: Sequence[str]
) -> list[_TurnTarget]:
    """Resolve caller-named ids to turns, bounded and deduped.

    Both kinds of checkpoint id are accepted, because the hover gesture sends
    whichever tick is under the pointer: a completion id names its own turn;
    a USER id is the turn key itself and names the turn it opens (the name
    lands on that turn's completion tick — the only surface that wears one).
    Ids that resolve to nothing, to a turn with no completion tick, or to the
    same turn twice are dropped rather than refused — the caller is asking
    for decoration, and a stale id is the rail's normal state after a fork.
    """
    by_id = {c.id: c for c in index.checkpoints}
    user_key = {c.turn: c.id for c in index.checkpoints if c.kind == transcript_index.KIND_USER}
    completions = {c.turn for c in index.checkpoints if c.kind == transcript_index.KIND_COMPLETION}
    targets: list[_TurnTarget] = []
    seen_ids: set[str] = set()
    seen_turns: set[str] = set()
    for raw_id in ids:
        request_id = str(raw_id)
        if not request_id or request_id in seen_ids:
            continue
        seen_ids.add(request_id)
        checkpoint = by_id.get(request_id)
        if checkpoint is None:
            continue
        turn_key = (
            checkpoint.id
            if checkpoint.kind == transcript_index.KIND_USER
            else user_key.get(checkpoint.turn, "")
        )
        if not turn_key or turn_key in seen_turns or checkpoint.turn not in completions:
            continue
        digest = build_turn_digest(index, checkpoint.turn)
        if digest is None:
            continue
        seen_turns.add(turn_key)
        targets.append(
            _TurnTarget(
                request_id=request_id,
                turn_key=turn_key,
                turn=checkpoint.turn,
                digest=digest,
            )
        )
        if len(targets) >= MAX_WARM_IDS:
            break
    return targets


def _targets_default(
    index: "transcript_index.TranscriptIndex", limit: int | None
) -> list[_TurnTarget]:
    """The rail-open selection: most recent settled turns missing names (D2).

    Newest first, because the rail's recent ticks are the ones a user is
    looking at and the ones whose names will be read; a live ``open`` tail is
    skipped — its text is still moving, so a name bought now would be
    regenerated on the next warm anyway. Turns already named are skipped —
    even when their text moved: this batch buys names for turns that have
    NONE, and refreshing a stale name is the explicit hover's job (D2's
    "missing names" wording; agent review round 1, NIT-1). Turns inside their
    failure cooldown are still SELECTED (they show up as accepted but never
    pending, so the rail can render their unavailable state without waiting
    through a poll).
    """
    bound = DEFAULT_WARM_LIMIT if limit is None else max(1, min(int(limit), MAX_WARM_IDS))
    items = _naming_items(index)
    user_key = {c.turn: c.id for c in index.checkpoints if c.kind == transcript_index.KIND_USER}
    completions = sorted(
        (c for c in index.checkpoints if c.kind == transcript_index.KIND_COMPLETION),
        key=lambda c: c.turn,
        reverse=True,
    )
    targets: list[_TurnTarget] = []
    for checkpoint in completions:
        if len(targets) >= bound:
            break
        if checkpoint.outcome == transcript_index.OUTCOME_OPEN:
            continue
        turn_key = user_key.get(checkpoint.turn, "")
        if not turn_key or naming_state(items.get(turn_key)) == "ready":
            continue
        digest = build_turn_digest(index, checkpoint.turn)
        if digest is None:
            continue
        targets.append(
            _TurnTarget(
                request_id=checkpoint.id,
                turn_key=turn_key,
                turn=checkpoint.turn,
                digest=digest,
            )
        )
    return targets


async def _run_one(
    state: _SessionState,
    target: _TurnTarget,
    *,
    config_dir: str | Path,
    session_id: str,
    complete_fn: Callable[[str, str], Awaitable[str]],
) -> None:
    """One bounded naming call, serialized per session, with a marker either way.

    The lock is the per-session concurrency 1 (D2). Every path ends in a
    persisted item — the parsed name on success, an ``unavailable`` marker on
    a timeout, a provider error, or a reply that parses to nothing — because
    the marker is what carries the cooldown across processes and restarts and
    what the manifest reads. A CANCELLATION is the one path that leaves no
    marker: the task is detached decoration and its cancel is a routine
    shutdown, not a failure to charge a cooldown for.
    """
    async with state.lock:
        try:
            raw = await asyncio.wait_for(
                complete_fn(CHECKPOINT_NAME_SYSTEM_PROMPT, target.digest),
                CHECKPOINT_NAME_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "checkpoint naming call timed out after %.0fs (session %s, turn key %s)",
                CHECKPOINT_NAME_TIMEOUT_S,
                session_id,
                target.turn_key,
            )
            item: dict[str, Any] = await _failure_item(
                target, config_dir=config_dir, session_id=session_id
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — any provider failure means "no name"
            # The exception TYPE and a bounded excerpt of the PROVIDER's own
            # message, mirroring ``naming._ask_for_title``: "checkpoints never
            # get names" is otherwise diagnosable only by provider archaeology,
            # while a provider that echoes its request in the error body must
            # not leak the digest into the log through the message.
            message = str(exc)
            if len(message) > _LOGGED_ERROR_CHARS:
                message = message[:_LOGGED_ERROR_CHARS] + "…"
            logger.warning(
                "checkpoint naming call failed (session %s, turn key %s): %s: %s",
                session_id,
                target.turn_key,
                type(exc).__name__,
                message,
            )
            item = await _failure_item(target, config_dir=config_dir, session_id=session_id)
        else:
            parsed = parse_checkpoint(str(raw or ""))
            if parsed is None:
                # A decline or an unparseable reply: no name will come from a
                # second identical spend, so it takes the unavailable state
                # (and its cooldown) exactly like a failure.
                item = await _failure_item(target, config_dir=config_dir, session_id=session_id)
            else:
                item = {
                    "name": parsed.name,
                    "summary": parsed.summary,
                    "text_hash": _digest_hash(target.digest),
                    "generated_ts": time.time(),
                }
        # Off the loop: the cache can be megabytes (the operator's 272 MB
        # journal yields a ~7 MB index) and patch_naming re-reads and rewrites
        # it whole. Best-effort by contract — patch_naming swallows an
        # unwritable cache, which costs a later regeneration, never a failure.
        await asyncio.to_thread(
            transcript_index.patch_naming, config_dir, session_id, {target.turn_key: item}
        )


def _unavailable_item(target: _TurnTarget) -> dict[str, Any]:
    """The persisted failure marker (see :func:`naming_state` for its window).

    ``text_hash`` rides along for the same reason it rides on a success: it is
    the record of WHICH content the attempt was made against. This shape is
    for turns that NEVER had a name; a turn that already carries one keeps it
    through :func:`_failure_item` instead.
    """
    return {
        "state": "unavailable",
        "text_hash": _digest_hash(target.digest),
        "failed_ts": time.time(),
    }


async def _failure_item(
    target: _TurnTarget, *, config_dir: str | Path, session_id: str
) -> dict[str, Any]:
    """The persisted item for a failed or declined attempt.

    A turn that ALREADY carries a name KEEPS it: the attempt failed, but the
    previous name+summary is still the best answer the rail has, and replacing
    it with the marker would make a refresh visibly degrade what it was
    refreshing (agent review round 1, MINOR-1). The failure rides beside the
    pair as ``failed_ts``, which warm reads as the regeneration cooldown
    (:func:`_failure_cooling`) and the manifest ignores while a name is
    present. ``text_hash`` stays the OLD hash on purpose: it describes the
    content the kept name was generated for, so the item keeps reading as
    stale — and, once the cooldown passes, a later warm regenerates it.

    The read is an extra cache parse, but this path is a provider FAILURE:
    rare, off the loop, and already about to rewrite the same document.
    """
    index = await asyncio.to_thread(transcript_index.read_index, config_dir, session_id)
    existing = _naming_items(index).get(target.turn_key) if index is not None else None
    if isinstance(existing, dict) and existing.get("name"):
        kept = dict(existing)
        kept["failed_ts"] = time.time()
        return kept
    return _unavailable_item(target)


def _schedule(
    state: _SessionState,
    target: _TurnTarget,
    *,
    config_dir: str | Path,
    session_id: str,
    complete_fn: Callable[[str, str], Awaitable[str]],
) -> None:
    """Start one background generation, with the strong reference the loop does not keep."""
    task = asyncio.get_running_loop().create_task(
        _run_one(
            state,
            target,
            config_dir=config_dir,
            session_id=session_id,
            complete_fn=complete_fn,
        ),
        name=f"checkpoint-name:{session_id}:{target.turn_key}",
    )
    state.active[target.turn_key] = task

    def _forget(settled: "asyncio.Task[None]") -> None:
        if state.active.get(target.turn_key) is settled:
            state.active.pop(target.turn_key, None)
        if not settled.cancelled():
            # Consume a stray exception so the loop does not warn about an
            # unretrieved one; the persisted marker (or its absence, on a
            # cancel) is the outcome this module keeps.
            settled.exception()

    task.add_done_callback(_forget)


async def warm_checkpoints(
    config_dir: str | Path,
    session_id: str,
    *,
    ids: Sequence[str] | None = None,
    limit: int | None = None,
    complete_fn: Callable[[str, str], Awaitable[str]],
) -> dict[str, list[str]]:
    """Accept naming work for one session; never blocks on a call.

    Returns ``{"accepted": [...], "pending": [...]}`` (D9): ``accepted`` are
    the ids this call took ownership of — the request's ids echoed back, or
    the selected ones when the caller sent none — and ``pending`` are those
    still waiting for a name: queued, in flight, or named-but-stale and
    regenerating. An id already named (same digest), or a turn inside its
    failure cooldown, is accepted but NOT pending: the rail's poll (while any
    requested id is pending) has nothing left to wait for on it.

    The digest comes from the transcript index — the cache the manifest route
    refreshed. This function never scans the journal itself: a rail gesture
    must not pay a 22 s scan, and an unbuilt index simply means there is
    nothing to name yet (an empty answer, not an error).

    Idempotent by construction: a turn already queued or in flight is
    answered from the active set without a second task, and a current cached
    name is answered without work. Bounded: at most :data:`MAX_WARM_IDS` ids
    per call.
    """
    index = await asyncio.to_thread(transcript_index.read_index, config_dir, session_id)
    if index is None:
        return {"accepted": [], "pending": []}
    items = _naming_items(index)
    state = _session_state(config_dir, session_id)
    targets = _targets_from_ids(index, ids) if ids is not None else _targets_default(index, limit)
    accepted: list[str] = []
    pending: list[str] = []
    for target in targets:
        accepted.append(target.request_id)
        item = items.get(target.turn_key)
        current = naming_state(item)
        if current == "ready":
            if isinstance(item, dict) and item.get("text_hash") == _digest_hash(target.digest):
                # Cached and current: the idempotent no-op arm.
                continue
            if _failure_cooling(item):
                # A stale name whose regeneration just failed: keep serving the
                # LAST GOOD pair and let the cooldown gate the next attempt —
                # without this every warm would re-spend on a turn that keeps
                # failing to regenerate.
                continue
            # Stale name (the turn grew since it was named): fall through and
            # regenerate over it — D2's "regenerate only when the hash
            # changes" is exactly this case.
        elif current == "unavailable":
            continue
        if target.turn_key in state.active:
            pending.append(target.request_id)
            continue
        _schedule(
            state,
            target,
            config_dir=config_dir,
            session_id=session_id,
            complete_fn=complete_fn,
        )
        pending.append(target.request_id)
    return {"accepted": accepted, "pending": pending}


def _reset_for_tests() -> None:
    """Drop the module's loop state (test isolation; never called in production).

    Tasks are cancelled rather than left running: their writes land in a cache
    keyed by config root, and a suite that re-creates roots under fresh
    ``tmp_path``s must not have a previous test's task settle into the next
    one's fixtures.
    """
    for state in _STATE.values():
        for task in state.active.values():
            task.cancel()
    _STATE.clear()
