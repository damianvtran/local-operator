"""Headless print rendering — the non-TUI subscriber for ``AgentEvent``s.

Two consumers:

- ``run_print_mode`` — the one-shot ``exec`` / ``exec_worker`` runner that
  mirrors the one-shot print-mode semantics (subscribe FIRST because session
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
from typing import Any, Callable

from rich.console import Console

from local_operator.ansi import sanitize_prompt_line, strip_control_sequences
from local_operator.harness.types import (
    AgentEndEvent,
    AgentEvent,
    CompactionStartEvent,
    Message,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ModelChangeEvent,
    NoticeEvent,
    RetryStartEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
)
from local_operator.session.protocol import SessionProtocol

#: One-line tool rows get truncated to this many columns (TUI minimalism).
_TOOL_LINE_WIDTH = 100


def strip_provider_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively drop ``provider_payload`` keys from a dumped event.

    The payload is transport-native replay state (encrypted reasoning items
    etc.) — opaque and useless outside the process that produced it, and it
    can be enormous. The stripping mirrors the provider-payload sanitation
    applied upstream so local rendering never carries that replay state.
    """
    return {key: _stripped(value) for key, value in data.items() if key != "provider_payload"}


def _stripped(value: Any) -> Any:
    """Recurse into a dumped value. ``Any`` is honest here: this walks
    arbitrary JSON produced by ``model_dump``."""
    if isinstance(value, dict):
        return strip_provider_payload(value)
    if isinstance(value, list):
        return [_stripped(item) for item in value]
    return value


def printable_event(event: AgentEvent) -> dict[str, Any]:
    """Shape an event for ``--json`` output.

    Removes two classes of bloat so transcripts grow linearly with
    conversation size instead of quadratically (a single long turn used to
    re-serialize its whole in-progress message on every streamed delta,
    producing multi-GB logs — fixed by forwarding only deltas):

    - ``message_update`` full-message snapshots are dropped; only the
      incremental ``delta`` is printed. The authoritative message follows in
      ``message_end``.
    - ``provider_payload`` is stripped everywhere it appears.
    """
    if isinstance(event, MessageUpdateEvent):
        return {
            "type": "message_update",
            "message_id": event.message.id,
            "delta": event.delta,
        }
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
        #: The attached session, held so an auth-error line can name the active
        #: provider in its recovery hint. ``None`` until :meth:`attach`.
        self._session: SessionProtocol | None = None

    @property
    def session_id(self) -> str | None:
        """The attached session's id, or ``None`` before :meth:`attach`.

        Read defensively: a test double satisfying only the parts of
        ``SessionProtocol`` a renderer touches may not carry an id, and a
        missing id must degrade to an unstamped line rather than break the
        stream that is the run's only output.
        """
        session = self._session
        if session is None:
            return None
        value = getattr(session, "session_id", None)
        return value if isinstance(value, str) and value else None

    # -- subscription entry point -------------------------------------------

    def handle(self, event: AgentEvent) -> None:
        """Event handler for ``session.subscribe`` (sync; the harness accepts
        sync or async handlers)."""
        if self.json_mode:
            payload = printable_event(event)
            # Stamp the session on EVERY line rather than once in a header.
            # External supervisors parse this stream line-by-line and
            # statelessly (Minerva's sentinel runner is a per-line jq filter),
            # so a header they happened to start after is unrecoverable — and
            # the id is what lets them resume the session later.
            session_id = self.session_id
            if session_id:
                payload.setdefault("session_id", session_id)
            sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
            sys.stdout.flush()
            self._track_outcome(event)
            return
        self._render(event)
        self._track_outcome(event)

    def attach(self, session: SessionProtocol) -> Callable[[], None]:
        """Subscribe to a session, returning the unsubscribe callable."""
        self._session = session
        return session.subscribe(self.handle)

    # -- internals -----------------------------------------------------------

    def _track_outcome(self, event: AgentEvent) -> None:
        """Record error/abort outcome and the last assistant text."""
        if isinstance(event, AgentEndEvent):
            if event.error or event.aborted:
                self.failed = True
        elif isinstance(event, MessageEndEvent):
            message = event.message
            if isinstance(message, Message) and message.role == "assistant":
                text = message.text
                if text:
                    self.last_assistant_text = text

    def _render(self, event: AgentEvent) -> None:
        if isinstance(event, MessageStartEvent):
            message = event.message
            if self.stream_text and isinstance(message, Message) and message.role == "assistant":
                self._streaming_assistant = True
        elif isinstance(event, MessageUpdateEvent):
            if self.stream_text and self._streaming_assistant and event.delta:
                # Plain write, no markup interpretation, immediate flush:
                # this is the "streamed via print" path.
                sys.stdout.write(event.delta)
                sys.stdout.flush()
        elif isinstance(event, MessageEndEvent):
            if self.stream_text and self._streaming_assistant:
                self._streaming_assistant = False
                sys.stdout.write("\n")
                sys.stdout.flush()
        elif isinstance(event, ToolExecutionStartEvent):
            # Sanitised for the same reason the TUI card is: tool_name, intent
            # and args are all model-controlled, and an erase-display escape in
            # any of them clears the operator's terminal. This is the non-JSON
            # headless renderer, so it writes real text to a real terminal;
            # `exec --json` is unaffected because json.dumps escapes it.
            summary = strip_control_sequences(event.intent or _args_summary(event.args))
            name = strip_control_sequences(event.tool_name)
            line = f"● {name} {summary}".rstrip()
            # `markup=False` and the style passed as a style, for the reason the
            # notice branch below states at length: the tool NAME is chosen by
            # the model, and a `[` in it is Rich markup — `[/red]x` raises
            # `MarkupError` here, which `session._emit` swallows, so the row
            # vanishes and the operator watches a tool run with no line at all.
            # Pre-existing on `main`; fixed here because the notice fix
            # generalised the rule and it should hold for every branch that
            # renders model-controlled text, not just the newest one (R14-2,
            # agent review round 14).
            self.console.print(line[:_TOOL_LINE_WIDTH], style="dim", highlight=False, markup=False)
        elif isinstance(event, ToolExecutionEndEvent):
            if event.is_error:
                name = strip_control_sequences(event.tool_name)
                self.console.print(f"✗ {name} failed", style="red", highlight=False, markup=False)
        elif isinstance(event, NoticeEvent):
            style = {"error": "red", "warning": "yellow"}.get(event.kind, "dim")
            # A GLYPH carries the severity, not just the colour. This renderer
            # writes to a real terminal but its output is also piped into logs
            # and read under NO_COLOR, where an ansi-stripped error notice was
            # indistinguishable from an informational one — and the `✗` line it
            # replaced for unrunnable tool calls did carry a marker, so dropping
            # it was a regression in exactly the case that matters (D11, design
            # round 3). `info` stays bare: a marker on every routine line is
            # noise, and it is the one kind with nothing to warn about.
            glyph = {"error": "✗ ", "warning": "! "}.get(event.kind, "")
            # SANITIZED, like every other line this renderer writes. Notice text
            # is no longer only ours: the unrunnable-call diagnostic carries a
            # model-chosen tool name, so an erase-display escape inside it would
            # clear the operator's terminal — and the `✗ <name> failed` line that
            # diagnostic replaced was stripped for exactly that reason, two
            # branches up. Moving the message onto a notice moved it off the
            # guard (R7-1, agent review round 7). Applied to every notice rather
            # than to that one call site, because the next notice to carry
            # untrusted text should not have to remember this.
            # `sanitize_prompt_line`, not bare stripping, and the style applied
            # as a Rich STYLE rather than as inline markup. Three hazards, and
            # the tool name inside this text is model-chosen (D14/D15, design
            # round 4):
            #
            # 1. Control sequences repaint the terminal — what `strip` covered.
            # 2. Newlines SURVIVE stripping by design (tool output is
            #    multi-line and the renderers want it), so a name containing one
            #    forges a second, unmarked row that can read as a clean success.
            #    `sanitize_prompt_line` collapses whitespace runs, which is the
            #    same reason it exists for approval prompts.
            # 3. Square brackets are Rich MARKUP: `[bold]x` silently renders the
            #    wrong name, and `[/red]oops` raises `MarkupError` inside the
            #    renderer, which `session._emit` swallows — so the notice
            #    vanishes entirely. That is precisely the silence this
            #    diagnostic was added to prevent, reachable from a hallucinated
            #    tool name. `markup=False` makes the text data rather than code.
            #
            # Pre-existing on the `✗ <name> failed` line this replaced; fixed
            # here rather than deferred because the notice is now the only
            # report an operator gets.
            text = sanitize_prompt_line(event.text)
            self.console.print(f"{glyph}{text}", style=style, highlight=False, markup=False)
        elif isinstance(event, RetryStartEvent):
            self.console.print(f"[dim]retry {event.attempt}: {event.error}[/dim]", highlight=False)
        elif isinstance(event, ModelChangeEvent):
            # The route edge in one line, both directions — the exec-mode
            # counterpart of the TUI band repaint: a reader of a long headless
            # run needs to know which model produced the output from here on.
            # The verbs pair with the failure notice's "falling back to"
            # (design D2): "serving from" reads as a location, not a route.
            selector = f"{event.provider}/{event.model_id}"
            self.console.print(
                f"[dim]{'fell back to' if event.is_fallback else 'back to'} " f"{selector}[/dim]",
                highlight=False,
            )
        elif isinstance(event, CompactionStartEvent):
            self.console.print("[dim]compacting context…[/dim]", highlight=False)
        elif isinstance(event, AgentEndEvent):
            if event.error:
                # ``markup=False`` for the same reason as the notice branch
                # above: ``error`` now carries model-authored prose (a
                # provider's refusal message), and adversarial text like
                # ``[/see policy]`` raised MarkupError inside this subscriber —
                # BEFORE ``_track_outcome`` ran, so the process printed a
                # traceback instead of the error line and exited 0. The text is
                # data, never markup. (Review R1-1.)
                #
                # Append the local recovery to an auth-classified failure so the
                # headless user learns they can `local-operator login` / rotate
                # the key, not just the provider's opaque refusal. No-op for
                # every other kind. Provider is the first segment of the active
                # model label.
                from local_operator.providers.failover import append_auth_recovery

                provider = ""
                if self._session is not None:
                    try:
                        provider = (self._session.model_label or "").partition("/")[0]
                    except Exception:
                        provider = ""
                self.console.print(
                    f"Error: {append_auth_recovery(event.error, provider or None)}",
                    style="red",
                    highlight=False,
                    markup=False,
                )
            elif event.aborted:
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
    """One-shot headless run mirroring the print-mode semantics.

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
