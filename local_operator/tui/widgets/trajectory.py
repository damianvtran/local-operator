"""The subagent trajectory viewer — click-through for one task job.

A subagent's child session streams the same ``AgentEvent`` taxonomy the main
session does; the engine retains a bounded serialization of them on the job
(``AsyncJob.trajectory``, entries are ``model_dump(mode="json")`` dicts) so
the TUI can replay what the child did without holding the child session. This
module folds those events into transcript-style rows the same way
``AgentEventBridge`` folds them for the server UI: one long-lived assistant
record per message id (deltas accumulate, the end adopts the authoritative
text) and one record per tool call keyed by ``tool_call_id``.

The viewer is a modal over the app: the band row is the summary, this screen
is the detail, and Esc returns to the band. Rendering is a PURE function of
the event list (:func:`render_trajectory`) so the fold can be tested without
a mounted screen.

Entries arrive from another session's internals and are read defensively —
a malformed or partial dict degrades to a skipped row, never a crash: this
screen is observability, and observability must not be able to take the app
down.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rich.cells import cell_len
from rich.style import Style
from rich.text import Text
from textual.binding import Binding
from textual.containers import Container, ScrollableContainer
from textual.screen import ModalScreen
from textual.widgets import Static

from local_operator.ansi import strip_control_sequences
from local_operator.tui import theme as theme_mod

#: Events rendered per trajectory. The engine caps the retained list at the
#: same number (the job keeps at most 500 child events), so this is the TUI's
#: half of one shared bound: a viewer handed a longer list by a future engine
#: still paints a bounded screen instead of measuring an unbounded one. Read
#: by the fold AND pinned by a test, so the two halves of the contract cannot
#: drift silently.
TRAJECTORY_MAX_EVENTS = 500

#: Cells of tool-argument summary a trajectory row may spend. The row leads
#: with the tool NAME — the arguments are context, not the headline.
_ARGS_CAP = 60


def _as_dict(event: Any) -> dict[str, Any]:
    """One trajectory entry as a dict.

    Entries are normally serialized dicts already; an engine that hands over
    live event objects is tolerated by dumping them, and anything neither is
    skipped (empty dict) rather than raised on.
    """
    if isinstance(event, dict):
        return event
    dump = getattr(event, "model_dump", None)
    if callable(dump):
        try:
            value = dump(mode="json")
            return value if isinstance(value, dict) else {}
        except Exception:
            return {}
    return {}


def _message_text(message: Any) -> str:
    """Text of a serialized message dict (assistant text blocks only)."""
    if not isinstance(message, dict):
        return ""
    parts: list[str] = []
    for block in message.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)


def _tool_args_summary(args: Any) -> str:
    """``key=value`` argument digest, one line, bounded."""
    if not isinstance(args, dict) or not args:
        return ""
    rendered = ", ".join(f"{key}={value!r}" for key, value in args.items())
    rendered = " ".join(rendered.split())  # one line, never a reflow
    if cell_len(rendered) > _ARGS_CAP:
        rendered = rendered[: _ARGS_CAP - 1] + "…"
    return rendered


def render_trajectory(events: Sequence[Any]) -> list[Text]:
    """Fold serialized child events into transcript-style rows.

    Mirrors ``AgentEventBridge``'s record model: assistant messages
    accumulate per message id (start resets, update appends the delta, end
    adopts the authoritative text) and tool rows are keyed by
    ``tool_call_id`` (start announces, end settles with ✓/✗). Everything else
    — turn boundaries, retries, notices — is noise at this zoom and dropped,
    except notices, which carry words the child's operator wrote.
    """
    fg = Style(color=theme_mod.semantic_color("fg"))
    muted = Style(color=theme_mod.semantic_color("muted"))
    dim = Style(color=theme_mod.semantic_color("dim"))
    danger = Style(color=theme_mod.semantic_color("danger"))

    # Creation-ordered records: ("text", message_id) or ("tool", call_id).
    ordered: list[tuple[str, str]] = []
    streams: dict[str, str] = {}  # message id -> accumulated text
    tools: dict[str, tuple[str, str, str | None]] = {}  # id -> (name, args, outcome)

    for raw in list(events)[:TRAJECTORY_MAX_EVENTS]:
        event = _as_dict(raw)
        etype = event.get("type")
        if etype in ("message_start", "message_update", "message_end"):
            message = event.get("message") or {}
            if message.get("role") != "assistant":
                continue
            message_id = str(message.get("id") or id(raw))
            if etype == "message_start":
                streams[message_id] = ""
                if not any(kind == "text" and key == message_id for kind, key in ordered):
                    ordered.append(("text", message_id))
            elif etype == "message_update":
                if message_id not in streams:
                    streams[message_id] = ""
                    ordered.append(("text", message_id))
                streams[message_id] += str(event.get("delta") or "")
            else:  # message_end adopts the authoritative text
                text = _message_text(message) or streams.get(message_id, "")
                if message_id not in streams and not text:
                    continue
                streams[message_id] = text
                if not any(kind == "text" and key == message_id for kind, key in ordered):
                    ordered.append(("text", message_id))
        elif etype == "tool_execution_start":
            call_id = str(event.get("tool_call_id") or id(raw))
            name = str(event.get("tool_name") or "tool")
            tools[call_id] = (name, _tool_args_summary(event.get("args")), None)
            ordered.append(("tool", call_id))
        elif etype == "tool_execution_end":
            call_id = str(event.get("tool_call_id") or "")
            if call_id not in tools:
                continue  # end without a start: nothing to settle
            name, args, _outcome = tools[call_id]
            result = event.get("result") or {}
            outcome = "error" if (event.get("is_error") or result.get("is_error")) else "done"
            tools[call_id] = (name, args, outcome)
        elif etype == "notice":
            message_id = f"notice-{len(ordered)}"
            streams[message_id] = "· " + str(event.get("text") or "")
            ordered.append(("text", message_id))

    rows: list[Text] = []
    for kind, key in ordered:
        if kind == "text":
            text = strip_control_sequences(streams.get(key, "")).strip()
            if not text:
                continue  # a tool-use message carries no prose — spend no row
            line = Text(no_wrap=False)
            line.append(text, style=fg)
            rows.append(line)
        else:
            name, args, outcome = tools.get(key, ("tool", "", None))
            row = Text(no_wrap=True, overflow="ellipsis")
            row.append("▸ ", style=dim)
            row.append(strip_control_sequences(name), style=muted)
            if args:
                row.append("  ", style=dim)
                row.append(strip_control_sequences(args), style=dim)
            if outcome == "done":
                row.append("  ✓", style=dim)
            elif outcome == "error":
                row.append("  ✗", style=danger)
            rows.append(row)
    return rows


class TrajectoryScreen(ModalScreen[None]):
    """The modal replay of one subagent's retained events.

    Opened from a band row (click or Enter), closed by Esc. The header names
    the subagent and its outcome; the body scrolls the folded rows. A
    snapshot, deliberately: the viewer reads the job's trajectory once at
    open, so reading it never races the child still writing it.
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Close", show=False),
    ]

    def __init__(self, label: str, status: str, events: Sequence[Any]) -> None:
        super().__init__()
        self._label = strip_control_sequences(label or "subagent")
        self._status = status
        self._events = list(events or [])

    def compose(self):  # type: ignore[override]
        dim = Style(color=theme_mod.semantic_color("dim"))
        header = Text(no_wrap=True, overflow="ellipsis")
        header.append(self._label, style=Style(color=theme_mod.semantic_color("fg")))
        header.append(f"  {self._status}", style=dim)
        with Container(classes="trajectory"):
            yield Static(header, id="trajectory-header")
            self._body = ScrollableContainer(id="trajectory-scroll")
            yield self._body

    def on_mount(self) -> None:
        # Rows mount AFTER the container tree exists (compose only yields;
        # mounting into a not-yet-mounted container would race layout). The
        # fold runs once against the snapshot; an empty trajectory still
        # shows the header and one honest line — the row is the news.
        dim = Style(color=theme_mod.semantic_color("dim"))
        rows = render_trajectory(self._events)
        if not rows:
            rows = [Text("no trajectory retained", style=dim)]
        for row in rows:
            self._body.mount(Static(row, classes="trajectory-row"))
