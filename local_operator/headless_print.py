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
from typing import Any, Awaitable, Callable

from rich.console import Console

from local_operator.ansi import sanitize_prompt_line, strip_control_sequences
from local_operator.harness.types import (
    FAULT_KEY,
    INTERRUPTED_FAULTS,
    AgentEndEvent,
    AgentEvent,
    CompactionStartEvent,
    Message,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ModelChangeEvent,
    NoticeEvent,
    OutputValidationEvent,
    ReasoningDeltaEvent,
    RetryStartEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
)
from local_operator.harness.wire import bound_agent_end_for_wire
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


def printable_event(event: AgentEvent, *, session_id: str | None = None) -> dict[str, Any]:
    """Shape an event for ``--json`` output.

    Removes three classes of bloat so transcripts grow linearly with
    conversation size instead of quadratically (a single long turn used to
    re-serialize its whole in-progress message on every streamed delta,
    producing multi-GB logs — fixed by forwarding only deltas):

    - ``message_update`` full-message snapshots are dropped; only the
      incremental ``delta`` is printed. The authoritative message follows in
      ``message_end``.
    - ``provider_payload`` is stripped everywhere it appears.
    - an ``agent_end`` frame — which carries the turn's WHOLE conversation — is
      bounded to ``harness.wire.AGENT_END_FRAME_BUDGET_BYTES``. Its tool rows
      are already on this stream as ``tool_execution_end`` events, so shipping
      them again in the aggregate doubles the turn's bytes, and on this path
      the size is not a matter of taste: supervisors read this stream with a
      4 MiB ``bufio.Scanner`` and fail the whole run on ``scanner.Err()``.
      Elided tool rows say so and name the transcript entry holding the text.

    ``session_id`` is stamped onto every emitted line — see the call site for
    why the stamp is per line rather than per stream — and is also what lets an
    elision marker name the session whose transcript holds what was cut.
    """
    if isinstance(event, MessageUpdateEvent):
        payload: dict[str, Any] = {
            "type": "message_update",
            "message_id": event.message.id,
            "delta": event.delta,
        }
    elif isinstance(event, ReasoningDeltaEvent):
        # Explicit for the reason the branch above is: a supervisor's per-line
        # filter reads these fields by NAME, and the reasoning channel is the
        # one they use to see the model working before it answers. There is no
        # accumulated payload to strip here -- the event carries one fragment --
        # so the shaping is the field list itself.
        payload: dict[str, Any] = {
            "type": "reasoning_delta",
            "message_id": event.message_id,
            "delta": event.delta,
        }
    else:
        payload = strip_provider_payload(event.model_dump(mode="json"))
    # THE STAMP GOES ON WHATEVER THIS FUNCTION RETURNS, on every branch. It used
    # to live at the call site, after shaping; moving it in here without moving
    # it past the early return above dropped the session id from exactly the
    # most frequent line type on the stream — the one a supervisor's per-line
    # filter sees most — which is unrecoverable for a stateless reader and is
    # what `test_every_emitted_line_carries_the_session_id` now pins (it would
    # have caught that).
    if session_id:
        payload.setdefault("session_id", session_id)
    return bound_agent_end_for_wire(payload, session_id=session_id)


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
    assistant message for text-mode output, and ``last_output_payload``
    tracks the VALIDATED payload span when an output contract ran —
    ``output_contract_failed`` says a contract ran and exhausted its retries,
    in which case stdout carries nothing (a wrong payload is worse than an
    empty one for ``… | jq``).
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
        #: The payload span the output contract validated, or "" when no
        #: contract ran (the field stays empty and ``run_print_mode`` falls
        #: back to ``last_assistant_text``, which is what keeps an unenforced
        #: run byte-identical to today).
        self.last_output_payload: str = ""
        #: True once a contract-checked turn exhausted its retries: the run
        #: failed AND its stdout must stay empty.
        self.output_contract_failed: bool = False
        self._streaming_assistant: bool = False
        #: Whether this model call's reasoning phase has already announced
        #: itself on stderr. One line per phase, not per fragment -- see
        #: :meth:`_render` for why exec does not stream the text itself.
        self._reasoning_announced: bool = False
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
            # The session id is stamped on EVERY line rather than once in a
            # header, inside ``printable_event``. External supervisors parse
            # this stream line-by-line and statelessly (Minerva's sentinel
            # runner is a per-line jq filter), so a header they happened to
            # start after is unrecoverable — and the id is what lets them
            # resume the session later. It is also what makes an elided tool
            # row's marker resolvable, which is why the shaping function is the
            # one that stamps it.
            payload = printable_event(event, session_id=self.session_id)
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
        elif isinstance(event, OutputValidationEvent):
            if event.ok:
                self.last_output_payload = event.payload_text
            elif event.exhausted:
                # Only an EXHAUSTED rejection silences stdout. A rejected
                # attempt with retries left is followed by another final
                # answer, and the last one wins: a recovery must not be
                # overwritten by the failure before it.
                self.output_contract_failed = True
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
        elif isinstance(event, ReasoningDeltaEvent):
            # The phase is ANNOUNCED once, dim, on stderr -- NOT streamed.
            #
            # Two contracts decide this. stdout is the run's payload (the answer
            # in text mode, one JSON line per event with ``--json``), so
            # reasoning must never touch it: a caller piping the answer into a
            # file would get the model's private thoughts concatenated to it.
            # And stderr is progress chrome read by a human, where one line per
            # TOKEN would bury every tool row in a wall of thinking -- the TUI
            # gets away with a live block because it repaints a region instead of
            # appending lines. The reasoning CONTENT is fully available on the
            # JSON channel, one ``reasoning_delta`` line per fragment, which is
            # where a program that wants it should be reading.
            #
            # ``markup=False`` and the style passed as a style for the same
            # reason the notice branch states at length: this text is
            # model-authored, and a ``[`` in it would otherwise be Rich markup.
            if event.delta and not self._reasoning_announced:
                self._reasoning_announced = True
                self.console.print("· thinking…", style="dim", highlight=False, markup=False)
        elif isinstance(event, MessageEndEvent):
            if self.stream_text and self._streaming_assistant:
                self._streaming_assistant = False
                sys.stdout.write("\n")
                sys.stdout.flush()
            # The phase is over for this call, so the next one announces itself
            # again (a tool-loop turn reasons before each of its calls).
            self._reasoning_announced = False
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
                # A MARKED abort/skip is not a failure: the user stopped it or
                # steering redirected it, and the line must not send a piped
                # log's reader hunting a tool that never had a chance to work
                # (the same {skipped, aborted} → interrupted rule the TUI and
                # the phone apply). Marker only — never the wording — so a
                # genuine `exit 3` that happens to read like an abort stays
                # red.
                details = getattr(event.result, "details", None)
                fault = details.get(FAULT_KEY) if isinstance(details, dict) else None
                if fault in INTERRUPTED_FAULTS:
                    self.console.print(
                        f"⊘ {name} interrupted", style="dim", highlight=False, markup=False
                    )
                else:
                    self.console.print(
                        f"✗ {name} failed", style="red", highlight=False, markup=False
                    )
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
        elif isinstance(event, OutputValidationEvent):
            # One dim stderr line per REJECTED attempt, and nothing on success
            # (success speaks through stdout, the payload). The text is
            # model-influenced (a schema violation message can quote the
            # payload at fault), so it takes the same rendering discipline as
            # every other line here: sanitized, whitespace-collapsed by
            # ``sanitize_prompt_line``, and ``markup=False`` so a ``[`` in it
            # is data rather than Rich markup that would vanish the line.
            if not event.ok:
                reason = sanitize_prompt_line(event.error)
                self.console.print(
                    f"! output check failed ({event.format}, attempt "
                    f"{event.attempt}/{event.max_attempts}): {reason}",
                    style="dim",
                    highlight=False,
                    markup=False,
                    # One stderr line per rejected attempt, literally: rich
                    # would otherwise reflow a long schema reason at the
                    # console width, splitting the pinned prefix from its
                    # reason for anything grepping stderr.
                    soft_wrap=True,
                )
        elif isinstance(event, RetryStartEvent):
            self.console.print(f"[dim]retry {event.attempt}: {event.error}[/dim]", highlight=False)
        elif isinstance(event, ModelChangeEvent) and event.context_metadata:
            # A context-metadata refresh (turn start/end, or a per-request
            # window read) re-announces the model already in force. It is a
            # display refresh, not a route edge: printed, it read as a recovery
            # that never happened ("back to X") or a repeat of a pin already
            # narrated ("fell back to X"). Keyed on the flag alone, NOT on an
            # empty ``reason``: a route edge that forgot its reason must still
            # print, not silently vanish from the only headless record of it.
            pass
        elif isinstance(event, ModelChangeEvent):
            # The route edge in one line, both directions — the exec-mode
            # counterpart of the TUI band repaint: a reader of a long headless
            # run needs to know which model produced the output from here on.
            # The verbs pair with the failure notice's "falling back to"
            # (design D2): "serving from" reads as a location, not a route.
            selector = f"{event.provider}/{event.model_id}"
            if event.is_fallback:
                verb = "fell back to"
            elif event.reason == "model switched":
                # A deliberate switch (`/model`, the phone, a peer) is not a
                # recovery: "back to" would claim a model this run never left.
                verb = "switched to"
            else:
                verb = "back to"
            self.console.print(f"[dim]{verb} {selector}[/dim]", highlight=False)
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
                from local_operator.providers.radient_recovery import (
                    append_usage_limit_recovery,
                )

                provider = ""
                if self._session is not None:
                    try:
                        provider = (self._session.model_label or "").partition("/")[0]
                    except Exception:
                        provider = ""
                # The same two additive remedies the TUI's helper applies, so
                # headless and TUI agree about what one failure says. This is
                # the ONE surface allowed to block on the Radient probe: the
                # module documents the three access patterns, and the renderer
                # is the bounded-sync arm's sole caller — it cannot await, and
                # a cache-only answer would render the generic fallback
                # forever because every headless run is a fresh process that
                # exits with this line. The probe's wall envelope is ~5.5s
                # worst case; the module swallows every failure and nothing
                # on this path may raise.
                sentence = append_auth_recovery(event.error, provider or None)
                sentence = append_usage_limit_recovery(sentence, provider or None)
                # ``soft_wrap`` ONLY for the contract-exhaustion end, and the
                # flag says which end this is without a string test: an
                # exhausted ``OutputValidationEvent`` sets it (see
                # ``_track_outcome``) and the loop emits that event and this
                # error together, by construction. It is needed there because
                # rich otherwise reflows the pinned "… after N attempts: …"
                # sentence at the console width (80 when stderr is not a
                # terminal), splitting it mid-phrase — measured. Every OTHER
                # error keeps its historical wrapping: review R-1 measured
                # that scoping this to all errors changed unenforced-run stderr,
                # and an unenforced run must stay byte-identical.
                self.console.print(
                    f"Error: {sentence}",
                    style="red",
                    highlight=False,
                    markup=False,
                    soft_wrap=self.output_contract_failed,
                )
            elif event.aborted:
                self.console.print("[red]aborted[/red]", highlight=False)


async def _call_before_dispose(hook: Callable[..., Awaitable[None]], failed: bool) -> None:
    """Call the teardown hook with the verdict when it wants it.

    Inspected rather than always-passed so the existing zero-argument hooks
    (and test doubles) are unaffected by a caller that needs the outcome.
    """
    import inspect

    try:
        takes_verdict = bool(inspect.signature(hook).parameters)
    except (TypeError, ValueError):  # builtins/C callables report no signature
        takes_verdict = False
    await (hook(failed) if takes_verdict else hook())


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
    session: SessionProtocol,
    messages: list[str],
    json_mode: bool = False,
    before_dispose: Callable[..., Awaitable[None]] | None = None,
    prompt_handler: Callable[[str], Awaitable[bool | None]] | None = None,
    continuation: Callable[[], Awaitable[bool]] | None = None,
) -> int:
    """One-shot headless run mirroring the print-mode semantics.

    Subscribe FIRST (session persistence depends on the subscription being
    active during ``prompt``), then prompt each message sequentially. Text
    mode prints the last assistant text to stdout; json mode already emitted
    one line per event. Returns 0 on success, 1 when any turn errored or was
    aborted. Disposes the session before returning (one-shot by contract).

    ``prompt_handler`` replaces the direct ``session.prompt`` call for hosts
    that own a prompt queue (``exec`` submits through the runtime so a live
    viewer cannot race it). Returning ``False`` from it marks the run failed;
    ``None`` (what ``session.prompt`` returns) leaves the verdict to the
    events, which is where a provider error already reports itself.

    ``before_dispose`` is awaited in the teardown, after the last event and
    BEFORE the session is disposed. It exists because the dispose is this
    function's own contract and a caller therefore cannot sequence anything
    ahead of it from the outside: ``exec --control`` needs its control surface
    announced-and-closed while the session is still whole, so an attached
    supervisor reads a deliberate end rather than a dropped socket, and the
    runtime's heartbeat is not still reading through a handle into a session
    being torn down. Awaited on every exit path including a raising prompt,
    and its own failure must not mask that error — the callable owns swallowing
    what it can (see ``ExecControl.aclose``).
    """
    renderer = PrintRenderer(stream_text=False, json_mode=json_mode)
    unsubscribe = renderer.attach(session)
    # ONE-SHOT DECLARATION (v3, 2026-09-30): this function disposes the session
    # itself (below), and its OWN arming of the departure pair lands
    # ~100-160 ms after the last turn's ``finally`` — too late for an arrival
    # queued on ``_turn_lock`` at the release, which was admitted 6-14 ms after
    # the final completion and cut, cancelled, by the disposal (nine exec
    # sessions on 0.64.8/0.64.9). The declaration lets the session arm the
    # same latches inside the last turn's own ``finally`` (``Session._run_turn``,
    # one host-hop earlier than this function's); the arming below stays as the
    # belt. ``--loop``/goal runs
    # (``continuation`` set) are excluded: their turns are the run's own work,
    # not teardown leftovers.
    if continuation is None:
        declare = getattr(session, "declare_one_shot_exit", None)
        if callable(declare):
            declare()
    try:
        for message in messages:
            # A handler may REPORT a failed turn rather than raise, because the
            # provider's own error arrives as an event the renderer prints. An
            # explicit False still fails the run, so a turn that died without
            # emitting an error event cannot be mistaken for success.
            if await (prompt_handler or session.prompt)(message) is False:
                renderer.failed = True
            if renderer.failed:
                break
        # Keep one subscription and one disposal across initial work and the
        # owner-local loop: repeated one-shot runs would dispose between turns.
        if not renderer.failed and continuation is not None:
            if not await continuation():
                renderer.failed = True
        if not json_mode and not renderer.output_contract_failed:
            # The validated payload span when a contract ran (byte-faithful to
            # what validated, so ``--output-format json | jq .`` works even
            # when the payload arrived inside a fence or prose), the last
            # assistant text otherwise. An exhausted turn prints NOTHING here:
            # a wrong payload on stdout is worse than an empty one for every
            # consumer of this stream.
            final_text = renderer.last_output_payload or renderer.last_assistant_text
            if final_text:
                sys.stdout.write(final_text + "\n")
                sys.stdout.flush()
        return 1 if renderer.failed else 0
    finally:
        if callable(unsubscribe):
            unsubscribe()
        # THE ONE-SHOT EXIT CLOSES THE DOOR BEFORE ITS DISPOSE (664a234ec561,
        # 2026-09-28): run_print_mode is how exec ends, and no departure rung
        # covers that path — ``begin_drain``/``begin_retire`` install these
        # latches and nothing on the exec side ever arms one — so a batch a
        # last turn deferred was spawned into a delivery run that the
        # following dispose aborted as a zero-work turn: an
        # ``error|cause=disposed`` row that superseded the completed turn's
        # own marker for every latest-wins reader and made the successor
        # journal a ``[session incident]`` about work that had finished.
        #
        # Armed HERE — before ``before_dispose`` (exec's close() owns that
        # await window on a real run) and before the dispose — so a spawn
        # that slipped past the spawn-time check is held at admission by
        # ``_prompt_messages`` instead of opening a turn whose only possible
        # end is the disposal's abort.
        #
        # NOT armed when ``continuation`` is set: a --loop/goal run is exec's
        # continuation mechanism, its turns are the run's own work, and the
        # latches belong to the one-shot contract this function disposes on.
        # The disposal's own evidence gate (``Session.dispose``) is the net
        # for anything that slips there.
        #
        # getattr-guarded: the departure pair lives on the one-shot hosts
        # (the real ``Session``, ``ServingSessionHandle``) rather than on
        # ``SessionProtocol``, which test doubles implement to the letter.
        if continuation is None:
            retire_jobs = getattr(session, "retire_job_deliveries_to_transcript", None)
            retire_wakes = getattr(session, "retire_wakes_to_inbox", None)
            if callable(retire_jobs):
                retire_jobs()
            if callable(retire_wakes):
                retire_wakes()
        if before_dispose is not None:
            # Handed the run's own verdict so a teardown that must publish a
            # terminal outcome (exec's browser scope) uses THIS value rather
            # than deriving a second, possibly disagreeing, one. Optional-arg
            # so existing zero-argument hooks keep working.
            await _call_before_dispose(before_dispose, renderer.failed)
        await session.dispose()
