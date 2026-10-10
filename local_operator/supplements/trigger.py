"""The "real user turn" predicate and the frozen per-run provenance (memo §2.1, §2.2).

WHY THIS EXISTS. A supplement is a callout under the answer to a PERSON'S request. There is
no single origin enum for "a person asked": the session's trigger set (``_run_triggers``)
records ``user | wake_prompt | monitor_prompt | internal`` keyed on ``custom_type`` and is
INSUFFICIENT alone, because goal-loop / judge continuations and spooled owner chrome call
``prompt(..., harness_injected=True)`` and classify as ``user``. The session therefore keeps
two more facts per run (``_run_typed_user`` / ``_run_last_trigger``, written by
``Session._note_run_input``) and freezes them, with the LOGICAL turn's messages, onto a
:class:`RunProvenance` when the end event is emitted. The runtime subscriber reads that
record -- the trigger set itself is reset at the next pipeline head, so by the time anything
could ask, the facts are gone.

WHY THE MESSAGES ARE SNAPSHOTS, NOT REFERENCES (memo round-1 R2). A turn that compacted and
auto-continued emits ONE end per loop run and the held end is replaced each run, so the
emitted end carries only the LAST run's messages: the file writes and tool results of the
pre-compaction half would silently vanish from the pre-filter. The session accumulates every
run's messages for the pipeline (reset at the pipeline head, NOT in ``_flush_held_end``,
which clears the generation before the ``_emit`` that freezes this record -- mirroring that
neighbouring clear freezes an empty accumulator and reproduces R2 silently). They are copied
into the small :class:`TurnItem` form HERE, at accumulation time, because compaction pruning
mutates tool-result messages IN PLACE (``compaction/pruning.py``): a held reference would see
the "[pruned]" notice instead of the result the evidence extractor needs.

LEAF ON PURPOSE: standard library only, duck-typed over the harness message classes, so the
session can import it at module scope for free.
"""

from dataclasses import dataclass, field
from typing import Any, Final, Iterable, Mapping

#: ``Session._run_last_trigger`` values. Wake / monitor deliveries use their own custom-type
#: strings (``wake_prompt`` / ``monitor_prompt``); these two name the user-role classes.
TRIGGER_TYPED: Final = "typed"
TRIGGER_INJECTED: Final = "injected"

#: Per-item text bound. The pre-filter only ever reads the head of a result (its own scan cap
#: is smaller), and the final answer is cut at the decision's own bound later, so a 50 MB tool
#: output costs the provenance 64 KiB, not 50 MB.
ITEM_TEXT_MAX_CHARS: Final = 65_536
#: The typed user text kept for the decision state (which truncates again to 1.5k).
USER_TEXT_MAX_CHARS: Final = 4_000
#: Tool-call argument keys worth keeping, with a bound each. Everything else (above all a
#: ``write`` call's ``content``, which can be the whole file) is dropped: candidates need the
#: PATH, the shell/code string and the intent, nothing else.
_KEPT_ARGS: Final = ("path", "command", "code", "i")
_ARG_MAX_CHARS: Final = 8_192


@dataclass(frozen=True)
class ToolCallItem:
    id: str
    name: str
    args: Mapping[str, str]


@dataclass(frozen=True)
class TurnItem:
    """One message of the logical turn, reduced to what the pre-filter reads."""

    id: str
    role: str
    text: str = ""
    tool_calls: tuple[ToolCallItem, ...] = ()
    tool_call_id: str = ""
    tool_name: str = ""
    is_error: bool = False


@dataclass(frozen=True)
class RunProvenance:
    """Facts about the run that just ended, frozen when its end event was emitted."""

    typed_user: bool
    last_trigger: str | None
    user_text: str
    items: tuple[TurnItem, ...]
    job_id: str | None
    one_shot: bool
    #: ``Session.turns_settled`` at freeze time. The runner waits until the counter EXCEEDS
    #: this, i.e. until THIS turn's pipeline ``finally`` has finished -- a counter and not a
    #: bare ``Event`` so a next turn that clears the event first cannot strand the waiter.
    settled_mark: int = 0
    triggers: frozenset[str] = field(default_factory=frozenset)


def _clip(value: Any, limit: int) -> str:
    return value[:limit] if isinstance(value, str) else ""


def _text_head(message: Any, limit: int) -> str:
    """The first ``limit`` characters of a message's text blocks, without joining them all.

    ``Message.text`` concatenates every block, which for a 50 MB tool result allocates 50 MB
    on the event loop just to keep 64 KiB of it; walking the blocks stops at the bound.
    """
    parts: list[str] = []
    remaining = limit
    for block in getattr(message, "content", None) or ():
        text = getattr(block, "text", None)
        if not isinstance(text, str) or getattr(block, "type", "text") != "text":
            continue
        parts.append(text[:remaining])
        remaining -= len(parts[-1])
        if remaining <= 0:
            break
    return "".join(parts)


def snapshot_message(message: Any, *, text_budget: int = ITEM_TEXT_MAX_CHARS) -> TurnItem:
    """Reduce one harness message to a :class:`TurnItem` (duck-typed, never raises)."""
    calls: list[ToolCallItem] = []
    for call in getattr(message, "tool_calls", None) or ():
        arguments = getattr(call, "arguments", None)
        kept = (
            {
                key: _clip(arguments.get(key), _ARG_MAX_CHARS)
                for key in _KEPT_ARGS
                if isinstance(arguments.get(key), str)
            }
            if isinstance(arguments, Mapping)
            else {}
        )
        calls.append(
            ToolCallItem(
                id=str(getattr(call, "id", "") or ""),
                name=str(getattr(call, "name", "") or ""),
                args=kept,
            )
        )
    return TurnItem(
        id=str(getattr(message, "id", "") or ""),
        role=str(getattr(message, "role", "") or ""),
        text=_text_head(message, min(ITEM_TEXT_MAX_CHARS, max(0, text_budget))),
        tool_calls=tuple(calls),
        tool_call_id=str(getattr(message, "tool_call_id", "") or ""),
        tool_name=str(getattr(message, "tool_name", "") or ""),
        is_error=bool(getattr(message, "is_error", False)),
    )


#: Total text kept across one logical turn's snapshots (all items, all loop runs). Past it,
#: further items keep their ids and tool calls but no text -- a runaway loop cannot make the
#: accumulator unbounded, and the file evidence (tool-call args) survives regardless.
TURN_TEXT_BUDGET: Final = 2_000_000


def snapshot_messages(
    messages: Iterable[Any], *, text_budget: int = TURN_TEXT_BUDGET
) -> list[TurnItem]:
    """Snapshot a run's messages, spending at most ``text_budget`` characters of text.

    Returns the items; the caller subtracts :func:`items_text_size` to carry the remainder
    into the next run of the same logical turn.
    """
    items: list[TurnItem] = []
    for message in messages:
        item = snapshot_message(message, text_budget=text_budget)
        text_budget -= len(item.text)
        items.append(item)
    return items


def items_text_size(items: Iterable[TurnItem]) -> int:
    return sum(len(item.text) for item in items)


def refusal(
    provenance: RunProvenance | None,
    *,
    error: bool,
    aborted: bool,
    cut_off_cause: str,
    goal_loop_running: bool,
) -> str:
    """Why this run is NOT a real user turn, or ``""`` when it is (memo §2.2 rules 1-6).

    A reason string and not a bool so the refusal is observable: tests pin each rule by name
    (a guard proven only by the all-clear path could be deleted unnoticed), and the DEBUG
    counter line says which rule absorbed a turn. First failing rule wins.
    """
    if provenance is None:
        return "no-provenance"
    # 6. A headless one-shot host (``run_print_mode``): its --json arm dumps EVERY event, so
    # this rule is load-bearing there, not belt-and-braces.
    if provenance.one_shot:
        return "one-shot-host"
    # 3. A subagent session.
    if provenance.job_id is not None:
        return "subagent"
    # 4. The run completed cleanly.
    if error or aborted or cut_off_cause:
        return "unclean-end"
    # 1. A typed user row exists in this run.
    if not provenance.typed_user:
        return "no-typed-user-row"
    # 2. ...and it is the LAST non-internal trigger: a courtesy wake folded in after the
    # user's message makes the final response mostly an answer to the wake.
    if provenance.last_trigger != TRIGGER_TYPED:
        return "user-not-last-trigger"
    # 5. No goal loop is running (covers ``/loop`` turns typed as the loop seed).
    if goal_loop_running:
        return "goal-loop"
    return ""


def final_answer(items: Iterable[TurnItem], *, persisted: Any) -> TurnItem | None:
    """The logical turn's final assistant message: last assistant item with text whose id is
    durable (``persisted(id)``), which is also the attention anchor rule (memo §2.4).

    Walks backwards because the final answer is last; an assistant message the transcript
    does not hold (aborted tail, never persisted) cannot be an anchor a surface can find.
    """
    for item in reversed(tuple(items)):
        if item.role == "assistant" and item.text.strip() and persisted(item.id):
            return item
    return None
