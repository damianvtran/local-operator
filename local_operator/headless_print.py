"""Headless print rendering — the non-TUI subscriber for ``AgentEvent``s.

Two consumers:

- ``run_print_mode`` — the one-shot ``exec`` / ``exec_worker`` runner that
  mirrors omp print-mode semantics (subscribe FIRST because session
  persistence hangs off the subscription, prompt each message sequentially,
  print the last assistant text in text mode, one JSON line per event in
  json mode, exit 1 on error/aborted).
- the headless REPL in ``cli.py`` — attaches a streaming ``PrintRenderer``
  to a long-lived session.

Minimalism rules match the TUI (docs/REWRITE.md section D): one line per
tool action, assistant text streamed plainly, errors in red. Everything
renders to the NORMAL screen — no alt-screen, no cursor tricks — so output
pipes cleanly. Progress chrome goes to stderr; stdout carries only the
machine-consumable payload (final text or JSON lines).
"""

from __future__ import annotations

import json
import sys
from typing import Any

from rich.console import Console

from local_operator.harness.types import AgentEvent, Message
from local_operator.session.protocol import SessionProtocol

#: One-line tool rows get truncated to this many columns (TUI minimalism).
_TOOL_LINE_WIDTH = 100


def _message_text(message: Any) -> str:
    """Text of a harness message; custom entries render as empty (they are
    context plumbing, never user-facing output)."""
    if isinstance(message, Message):
        return message.text
    return ""


def strip_provider_payload(data: Any) -> Any:
    """Recursively drop ``provider_payload`` keys from a dumped event.

    The payload is transport-native replay state (encrypted reasoning items
    etc.) — opaque and useless outside the process that produced it, and it
    can be enormous. Ported from omp's ``stripProviderPayload``.
    """
    if isinstance(data, dict):
        return {
            key: strip_provider_payload(value)
            for key, value in data.items()
            if key != "provider_payload"
        }
    if isinstance(data, list):
        return [strip_provider_payload(item) for item in data]
    return data


def printable_event(event: AgentEvent) -> dict[str, Any]:
    """Shape an event for ``--json`` output.

    Removes two classes of bloat so transcripts grow linearly with
    conversation size instead of quadratically (a single long turn used to
    re-serialize its whole in-progress message on every streamed delta,
    producing multi-GB logs — the omp fix, ported):

    - ``message_update`` full-message snapshots are dropped; only the
      incremental ``delta`` is printed. The authoritative message follows in
      ``message_end``.
    - ``provider_payload`` is stripped everywhere it appears.
    """
    if event.type == "message_update":
        return {"type": "message_update", "delta": getattr(event, "delta", "")}
    data = event.model_dump(mode="json")
    return strip_provider_payload(data)


class PrintRenderer:
    """Subscribes to ``AgentEvent``s and renders them with rich on the normal
    screen.

    Modes:

    - default: progress chrome (tool rows, notices, errors) to STDERR via a
      rich console, keeping stdout clean for the final text / JSON payload;
    - ``stream_text=True`` (headless REPL): assistant text deltas are written
      to STDOUT as they arrive, terminated by a newline on ``message_end``;
    - ``json_mode=True``: no chrome at all — one ``printable_event`` JSON
      line per event on stdout.

    ``failed`` flips True on an errored or aborted ``agent_end``; callers
    turn that into exit code 1. ``last_assistant_text`` tracks the final
    assistant message for text-mode output.
    """

    def __init__(
        self,
        console: Console | None = None,
        *,
        stream_text: bool = False,
        json_mode: bool = False,
    ) -> None:
        # stderr console: progress chrome must not pollute the payload stream.
        self.console = console or Console(stderr=True, highlight=False)
        self.stream_text = stream_text
        self.json_mode = json_mode
        self.failed: bool = False
        self.last_assistant_text: str = ""
        self._streaming_assistant: bool = False

    # -- subscription entry point -------------------------------------------

    def handle(self, event: AgentEvent) -> None:
        """Event handler for ``session.subscribe`` (sync; the harness accepts
        sync or async handlers)."""
        if self.json_mode:
            sys.stdout.write(json.dumps(printable_event(event), ensure_ascii=False) + "\n")
            sys.stdout.flush()
            self._track_outcome(event)
            return
        self._render(event)
        self._track_outcome(event)

    def attach(self, session: SessionProtocol) -> Any:
        """Subscribe to a session, returning the unsubscribe callable."""
        return session.subscribe(self.handle)

    # -- internals -----------------------------------------------------------

    def _track_outcome(self, event: AgentEvent) -> None:
        """Record error/abort outcome and the last assistant text."""
        if event.type == "agent_end":
            if getattr(event, "error", None) or getattr(event, "aborted", False):
                self.failed = True
        elif event.type == "message_end":
            message = getattr(event, "message", None)
            if isinstance(message, Message) and message.role == "assistant":
                text = message.text
                if text:
                    self.last_assistant_text = text

    def _render(self, event: AgentEvent) -> None:
        event_type = event.type
        if event_type == "message_start":
            message = getattr(event, "message", None)
            if self.stream_text and isinstance(message, Message) and message.role == "assistant":
                self._streaming_assistant = True
        elif event_type == "message_update":
            if self.stream_text and self._streaming_assistant:
                delta = getattr(event, "delta", "")
                if delta:
                    # Plain write, no markup interpretation, immediate flush:
                    # this is the "streamed via print" path.
                    sys.stdout.write(delta)
                    sys.stdout.flush()
        elif event_type == "message_end":
            if self.stream_text and self._streaming_assistant:
                self._streaming_assistant = False
                sys.stdout.write("\n")
                sys.stdout.flush()
        elif event_type == "tool_execution_start":
            name = getattr(event, "tool_name", "tool")
            summary = getattr(event, "intent", None) or _args_summary(getattr(event, "args", {}))
            line = f"● {name} {summary}".rstrip()
            self.console.print(f"[dim]{line[:_TOOL_LINE_WIDTH]}[/dim]", highlight=False)
        elif event_type == "tool_execution_end":
            if getattr(event, "is_error", False):
                name = getattr(event, "tool_name", "tool")
                self.console.print(f"[red]✗ {name} failed[/red]", highlight=False)
        elif event_type == "notice":
            text = getattr(event, "text", "")
            kind = getattr(event, "kind", "info")
            style = {"error": "red", "warning": "yellow"}.get(kind, "dim")
            self.console.print(f"[{style}]{text}[/{style}]", highlight=False)
        elif event_type == "retry_start":
            error = getattr(event, "error", "")
            attempt = getattr(event, "attempt", 0)
            self.console.print(f"[dim]retry {attempt}: {error}[/dim]", highlight=False)
        elif event_type == "compaction_start":
            self.console.print("[dim]compacting context…[/dim]", highlight=False)
        elif event_type == "agent_end":
            error = getattr(event, "error", None)
            if error:
                self.console.print(f"[red]Error: {error}[/red]", highlight=False)
            elif getattr(event, "aborted", False):
                self.console.print("[red]aborted[/red]", highlight=False)


def _args_summary(args: dict[str, Any]) -> str:
    """One-line, privacy-minded summary of tool args: first scalar value.

    Mirrors the TUI's one-line-per-action minimalism — enough to recognize
    what the tool is doing, never a dump.
    """
    for value in args.values():
        if isinstance(value, str) and value.strip():
            return value.replace("\n", " ").strip()
        if isinstance(value, (int, float, bool)):
            return str(value)
    return ""


async def run_print_mode(
    session: SessionProtocol, messages: list[str], json_mode: bool = False
) -> int:
    """One-shot headless run mirroring omp print-mode semantics.

    Subscribe FIRST (session persistence depends on the subscription being
    active during ``prompt``), then prompt each message sequentially. Text
    mode prints the last assistant text to stdout; json mode already emitted
    one line per event. Returns 0 on success, 1 when any turn errored or was
    aborted. Disposes the session before returning (one-shot by contract).
    """
    renderer = PrintRenderer(stream_text=False, json_mode=json_mode)
    unsubscribe = renderer.attach(session)
    try:
        for message in messages:
            await session.prompt(message)
            if renderer.failed:
                break
        if not json_mode and renderer.last_assistant_text:
            sys.stdout.write(renderer.last_assistant_text + "\n")
            sys.stdout.flush()
        return 1 if renderer.failed else 0
    finally:
        if callable(unsubscribe):
            unsubscribe()
        await session.dispose()
