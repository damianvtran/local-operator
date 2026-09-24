"""Building blocks shared by the end-to-end tests: real app, real session.

The rule this module exists to hold is that as little as possible is faked.
The tests here are the answer to "the whole unit suite was green while the app
was completely frozen", so anything replaced by a double is a place the stage
cannot see. What IS replaced, and why:

* **The provider stream.** A scripted ``stream_fn`` replaces the network, not
  the harness: the real :class:`~local_operator.session.session.Session` still
  runs its real agent loop over the scripted events, executes real tools and
  writes a real transcript. This is what makes the stage deterministic enough
  to run on every PR including forks, which is where the regression under test
  has to be caught. See ``tests/e2e/test_tui_e2e.py``'s module docstring for
  the full split.

* **OAuth endpoint discovery.** Two HTTP round trips to a server that does not
  exist. Replaced by the SDK's own documented fallback shape so the refresh
  path proceeds to the lock — which is the part under test — without a network.

Everything else is production code: ``OperatorApp``, ``Session``,
``Transcript``, ``McpManager``, the real tool implementations, and the real
cross-process OAuth refresh lock.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from local_operator.harness.rows import is_harness_notice_text
from local_operator.harness.types import (
    AbortSignal,
    ChatRequest,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    TextContent,
)
from local_operator.session.goal_judge import goal_continuation_prompt
from local_operator.session.goal_loop import LOOP_JUDGE_PROMPT, LOOP_PROMPT
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.tui.notify import ENV_DISABLE, ENV_DISABLE_VALUE
from local_operator.tui.resume_click import DESKTOP_LAUNCH_REFUSED_ENV

#: The notification kill switch every bespoke child-environment builder in this
#: suite must re-assert, spelled ONCE so it cannot drift between them.
#:
#: WHY A CONSTANT RATHER THAN TRUSTING THE AMBIENT ENVIRONMENT. The builders
#: here deliberately hand a child a FILTERED environment — dropping ``CMUX_*``
#: and ``LOP_*`` so a runtime cannot address the operator's live panes or adopt
#: their session — and several of them build the mapping with no reference to
#: ``os.environ`` at all. Those children are real ``lop`` runtimes: with no kill
#: switch one announces a parked gate through ``tui.notify.detached_notify``,
#: which on darwin is a genuine ``osascript display notification``, and with a
#: mock model it announces a completion whose body is the mock's own reply.
#: Both have fired from this suite before.
#:
#: WHY NOT ``agent_shell.harness_child_env``, which carries the same gate: that
#: helper is for a SCRIPT driving the real CLI, and it also injects the
#: nested-session marker — a cell that boots a TUI or a runtime child must not
#: have that waiver in its environment. So the pairs are separate on purpose,
#: sourced from the product constants, and pinned together by
#: ``tests/unit/test_notification_isolation.py``.
#:
#: ``tests/conftest.py`` also arms it at import time, which covers the builders
#: that copy ``os.environ``; this mapping is for the ones that do not, and is
#: guarded by ``tests/unit/test_notification_isolation.py``, which walks the
#: ``env=`` builders under ``tests/e2e/`` and ``scripts/`` and fails on one that
#: spawns a local-operator child without it.
NO_NOTIFY_ENV: dict[str, str] = {
    ENV_DISABLE: ENV_DISABLE_VALUE,
    DESKTOP_LAUNCH_REFUSED_ENV: "1",
}


#: A model spec no provider is ever asked about. ``provider="test"`` keeps the
#: pricing and discovery paths on their unknown-model branches instead of
#: reaching the model registry over the network.
TEST_MODEL = ModelSpec(provider="test", model_id="e2e-model", context_window=100_000)

#: The spec for a cell that must NOT be on the test hosting.
#:
#: A session whose recorded selection is the test hosting is skipped by the
#: notification surfaces on purpose (a test session is not news anybody can act
#: on; see ``session.model_selection.session_uses_test_hosting``), so a cell that
#: pins those surfaces' OWN contract — the frame counts, the dedupe — has to
#: drive a session that is not test-hosted. ``openai`` is a real provider and
#: the model id is deliberately one the registry has never heard of: every cell
#: using this passes its own scripted stream, so no client is ever built for it,
#: and an unknown model keeps the pricing and discovery paths offline exactly as
#: ``TEST_MODEL`` does.
E2E_ORACLE_MODEL = ModelSpec(provider="openai", model_id="e2e-oracle-model", context_window=100_000)

#: How long the pilot may wait for the app to adopt its session before the
#: test calls it a failure. Generous relative to the real cost (adoption is a
#: handful of loop turns) because a slow CI runner must not flake; short enough
#: that a genuinely stuck boot is reported by the assertion rather than by the
#: watchdog, which gives a clearer message.
ADOPT_TIMEOUT_S = 20.0


def _user_row_texts(request: ChatRequest | Mapping[str, Any]) -> list[str]:
    """The text of every USER-role row in one recorded call, in order.

    TWO SHAPES, because two harnesses record different things. The in-process
    cells hold ``ChatRequest``s. The exec cells drive a real ``lop`` subprocess
    against a real loopback provider, so the call happened in ANOTHER PROCESS
    and all there is to read is the raw wire body — there is no request object
    to hold. One reader for both, so the census below labels them identically.

    Only TEXT blocks are read, and the two skips are separate facts. ``Content``
    is ``TextContent | ImageContent`` and an image block carries no ``text`` at
    all, so a user row holding a pasted attachment is skipped by SHAPE — asked of
    the type rather than probed for the attribute, because a text-free block is
    the expected case and not a malformed one. A ``TextContent`` whose ``text``
    is empty is then skipped by VALUE: it would contribute no words while still
    costing a separator. Both are load-bearing for the census that labels calls
    by their question, so neither may be folded into the other.
    """
    if isinstance(request, Mapping):
        rows = [
            message
            for message in request.get("messages") or ()
            if isinstance(message, Mapping) and message.get("role") == "user"
        ]
        texts: list[str] = []
        for row in rows:
            content = row.get("content")
            if isinstance(content, str):
                texts.append(content)
                continue
            texts.append(
                " ".join(
                    block.get("text") or ""
                    for block in content or ()
                    if isinstance(block, Mapping) and block.get("type") == "text"
                )
            )
        return texts
    return [
        " ".join(
            block.text for block in message.content if isinstance(block, TextContent) and block.text
        )
        for message in request.messages
        if message.role == "user"
    ]


def last_user_text(request: ChatRequest | Mapping[str, Any]) -> str:
    """The question a provider call is ASKING: its last user-role row that is not STATE.

    A provider call carries the whole conversation, so the last user row is what
    distinguishes the calls a goal-owning session makes from one another — the
    human's own words for their turn, the judge's forked question for its aside,
    one of the two loop prompts for a self-continuation. Reading the WHOLE
    message list instead would match the goal transcript every time and label
    every call the same.

    STATE ROWS ARE SKIPPED, and that is a correction rather than a nicety: the
    runtime appends its ``[session-state]`` records, todo reminders and notices
    AFTER the turn's own prompt, so on those calls the raw last row is a state
    record and the question sits one row above it. Measured: both of the count
    loop's iterations in the exec cell end with ``[session-state]``, and a reader
    that took the raw last row labelled them user turns. The heads come from
    ``harness.rows`` — the module that already decides what counts as a
    harness-minted row — rather than from a second list here, because a second
    list is how a relabelled call goes unnoticed.

    A call whose rows are ALL state (or that has none) still reads as its last
    row, so the skip can only ever sharpen a label, never blank one.
    """
    texts = _user_row_texts(request)
    if not texts:
        return ""
    questions = [text for text in texts if not is_harness_notice_text(text)]
    return (questions or texts)[-1]


def provider_call_kinds(
    requests: Sequence[ChatRequest | Mapping[str, Any]],
    *,
    goal: str,
    extra: Mapping[str, str] | None = None,
) -> list[str]:
    """Label every provider call by WHAT IT IS, not by how many there were.

    ``/goal <text>`` no longer costs one provider call. It starts the user's own
    turn, and beside it the judged-goal machinery forks a judge AND may admit a
    continuation turn — all of it on the OWNER's host, so all of it lands in the
    same scripted stream. A bare ``len(stream.requests)`` therefore cannot tell a
    second user turn (the bug these cells exist to catch: an argument submitted
    twice, or a command that re-asks) from the harness legitimately working the
    goal. The property that survives the feature is that a command starts exactly
    ONE user-authored turn and that every other call is attributable, and this is
    the instrument that can state it. It reads EITHER recorded shape — a
    ``ChatRequest`` or a raw wire body — so the exec cells, which drive a real
    ``lop`` subprocess and can only see the wire, are censused by this instrument
    rather than by a second one.

    Labels, from the question each call asks:

    * ``goal`` — the row IS the standing goal text, i.e. the ``/goal`` argument
      submitted as an ordinary turn. Counting THIS is how a cell states that a
      command submits its argument once, and a retry does not submit it again;
    * ``judge`` — ``LOOP_JUDGE_PROMPT.format(goal=...)``, the forked aside. The
      goal judge and the ``/loop`` driver's judge ask the SAME question (one
      policy, two triggers), so they share a label on purpose;
    * ``continuation`` — ``goal_continuation_prompt(goal)``, the turn the judge
      admitted;
    * ``loop`` — ``LOOP_PROMPT``, the count loop's own next iteration, which is
      app-authored chrome rather than anybody's words;
    * whatever ``extra`` names, for the questions a cell drives itself (a
      goal-mode loop's ``LOOP_GOAL_PROMPT``, an aside's own text);
    * ``user`` — THE DEFAULT: a provider call whose question is none of the
      above is a human's own turn, because the harness's questions are the few
      texts this module knows by name and a person's turn is whatever they
      typed. A cell that needs an absolute census therefore asserts its exact
      counts (how many user turns it drove, and how many of each harness call)
      rather than a floor.
    """
    judge_question = LOOP_JUDGE_PROMPT.format(goal=goal)
    continuation = goal_continuation_prompt(goal)
    known = {
        goal: "goal",
        judge_question: "judge",
        continuation: "continuation",
        LOOP_PROMPT: "loop",
    }
    if extra:
        known.update(extra)
    return [known.get(last_user_text(request), "user") for request in requests]


def user_turns(kinds: Sequence[str]) -> int:
    """The user-authored turns in a census: the ``/goal`` argument and the rest.

    Named rather than inlined because the pair is easy to get subtly wrong at a
    call site: a ``/goal`` turn is a user turn, it just happens to be the one
    whose text a cell also asserts separately.
    """
    return sum(kind in {"user", "goal"} for kind in kinds)


class ScriptedStreamExhausted(RuntimeError):
    """A provider call arrived that the script had no turn for.

    Raised in place of an ``IndexError``, and the difference is not cosmetic.
    The session's turn machinery CATCHES an exception out of the stream and
    continues on a failed turn ("model stream failed: …"), so an exhausted
    script does not fail the test — it silently shifts every LATER call's
    conversation, which is precisely how a census label goes wrong while the
    cell's assertions stay green. Naming the call, the script's length and the
    question it was asking is what makes the mis-script diagnosable at all; the
    stream also records the exhaustion in ``exhausted_at``, so a cell can assert
    its tape held (``assert stream.exhausted_at is None``).
    """

    def __init__(self, index: int, scripted: int, question: str) -> None:
        super().__init__(
            f"provider call #{index} has no scripted turn ({scripted} scripted), "
            f"and it asked {question[:120]!r} — the tape is short, so every call "
            "after this one would be answered from the wrong turn"
        )


class ScriptedStream:
    """Replays one canned event list per model call; records the requests.

    Mirrors ``tests/unit/session/test_session.py``'s stream double deliberately
    — the session's contract with a provider is one place, and a second shape
    for it here would be a second thing to keep in step.
    """

    def __init__(self, turns: Sequence[Sequence[StreamEvent]]) -> None:
        self.turns = [list(turn) for turn in turns]
        self.requests: list[ChatRequest] = []
        #: The call index a script ran out at, or ``None`` while it held. Recorded
        #: as well as raised (see :class:`ScriptedStreamExhausted`): the raise is
        #: swallowed by the turn machinery, so this is the only thing a cell can
        #: assert on.
        self.exhausted_at: int | None = None

    def __call__(
        self, request: ChatRequest, signal: AbortSignal | None = None
    ) -> AsyncIterator[StreamEvent]:
        self.requests.append(request)
        # A call past the end of the script is a test bug (the loop re-entered
        # when the author expected it to stop, or the tape missed a harness call
        # like a judge), and answering with a bare stop would hide the extra
        # turn. The tape is indexed by CALL rather than by how many turns the
        # conversation happened to need, so the error names the call that had
        # nothing scripted for it.
        index = len(self.requests)
        if index > len(self.turns):
            self.exhausted_at = index
            raise ScriptedStreamExhausted(index, len(self.turns), last_user_text(request))
        turn = self.turns[index - 1]

        async def gen() -> AsyncIterator[StreamEvent]:
            for event in turn:
                yield event

        return gen()


def tool_call_turn(
    *,
    text: str,
    tool_name: str,
    tool_call_id: str,
    arguments: dict[str, Any],
) -> list[StreamEvent]:
    """One model turn that says something and then asks for one tool call."""
    return [
        StreamTextDelta(delta=text),
        StreamToolCallDelta(
            index=0,
            id=tool_call_id,
            name=tool_name,
            argument_delta=json.dumps(arguments),
        ),
        StreamEndEvent(stop_reason="toolUse"),
    ]


def text_turn(text: str) -> list[StreamEvent]:
    """One model turn that just answers and stops."""
    return [StreamTextDelta(delta=text), StreamEndEvent(stop_reason="stop")]


def build_session(
    directory: Path,
    stream: Any,
    *,
    tools: Iterable[Any] = (),
    cwd: Path | None = None,
    model: ModelSpec = TEST_MODEL,
) -> Session:
    """A REAL session over a real transcript directory.

    ``yolo=True`` because these tests drive turns, not the approval prompt:
    with the gate armed every tool call would park waiting for a keypress the
    test is not sending, and the approval surface has its own dedicated unit
    coverage (``tests/unit/tui/test_approvals_ux.py``).

    ``model`` exists for the cells that must NOT look like a test session to the
    notification surfaces — see ``E2E_ORACLE_MODEL``. It changes nothing else:
    the stream is scripted either way, so the spec is only ever read as a label.
    """
    # ``variables`` is wired the way ``session_factory`` wires it in
    # production. Without it ``session.variables`` is None, and a test driving
    # ``/credential`` over the runtime would exercise a session shape no real
    # `lop` ever builds — which is exactly the substitution that let the viewer
    # capability gap ship (see ``test_viewer_attach_e2e``'s module docstring).
    from local_operator.variables import VariableStore

    return Session(
        model=model,
        stream_fn=stream,
        tools=list(tools),
        transcript=Transcript(directory),
        system_blocks_provider=lambda *_args: [],
        yolo=True,
        cwd=str(cwd) if cwd is not None else None,
        variables=VariableStore(cwd=str(cwd) if cwd is not None else str(directory)),
    )


async def seed_transcript(directory: Path, messages: Sequence[Message]) -> Transcript:
    """Write ``messages`` to a session directory as a prior conversation.

    This is what makes ``/resume`` a real resume rather than a screen swap:
    the resumed session is built over a directory that already holds a
    transcript on disk, exactly as a session closed yesterday would.
    """
    directory.mkdir(parents=True, exist_ok=True)
    transcript = Transcript(directory)
    for message in messages:
        await transcript.append_message(message)
    transcript.flush()
    return transcript


def user_message(text: str) -> Message:
    return Message(role="user", content=[TextContent(text=text)])


def assistant_message(text: str) -> Message:
    return Message(role="assistant", content=[TextContent(text=text)], stop_reason="stop")


async def wait_for_adoption(app: Any, pilot: Any, timeout_s: float = ADOPT_TIMEOUT_S) -> None:
    """Pump the pilot until the app has adopted a session, or fail saying so.

    Polling rather than awaiting an event because adoption happens inside a
    Textual worker the test does not own a handle to; this is the same shape
    ``tests/unit/tui/test_resumed_conversation_name.py`` uses.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if app._session is not None:
            return
        await pilot.pause()
    raise AssertionError(
        f"the app never adopted a session within {timeout_s:g}s; "
        f"transcript on screen: {transcript_text(app)!r}"
    )


def transcript_text(app: Any) -> str:
    """Every transcript block flattened to the plain text a user would read.

    Reads what was actually COMPOSED rather than what the session holds, which
    is the distinction the resume test needs: a session can carry history that
    never reached the screen, and that is precisely the bug shape here.
    """
    from local_operator.tui.widgets.transcript import TranscriptView
    from tests.unit.tui.test_app_pilot import _renderable_plain

    try:
        view = app.query_one(TranscriptView)
    except Exception:  # noqa: BLE001 — a screen with no transcript reads as empty
        return ""
    return "\n".join(_renderable_plain(getattr(block, "renderable", "")) for block in view.blocks())


async def drain(pilot: Any, cycles: int = 30) -> None:
    """Let the app settle: mount, layout, paint, and any queued workers."""
    for _ in range(cycles):
        await pilot.pause()


class LoopLiveness:
    """Counts how many times the event loop came back to us, and how late.

    The distinction this class exists to draw: a test that merely waits for a
    frame to appear passes on a loop that painted once and then died. What has
    to be asserted after ``/resume`` is that the loop is STILL SCHEDULING —
    that it keeps handing control back at roughly the cadence asked for. So
    this records both the number of resumptions and the worst gap between them.

    Note that on a fully deadlocked process this object never gets to report
    anything at all: the freeze under test parks the loop thread inside a
    syscall, so no counter advances and nothing raises. That is not a gap in
    this class — it is exactly why :mod:`tests.e2e.watchdog` exists and why the
    liveness assertion must run INSIDE a ``bounded`` block.
    """

    def __init__(self) -> None:
        self.resumptions = 0
        self.worst_gap_s = 0.0

    async def observe(self, pilot: Any, seconds: float, interval_s: float = 0.02) -> None:
        """Pump the pilot for ``seconds``, recording cadence as it goes."""
        deadline = time.monotonic() + seconds
        last = time.monotonic()
        while time.monotonic() < deadline:
            await pilot.pause(interval_s)
            now = time.monotonic()
            self.worst_gap_s = max(self.worst_gap_s, now - last)
            last = now
            self.resumptions += 1

    def assert_alive(self, *, minimum: int, ceiling_s: float, context: str) -> None:
        """The loop kept scheduling, and never went quiet for ``ceiling_s``."""
        assert self.resumptions >= minimum, (
            f"the event loop scheduled only {self.resumptions} times after {context} "
            f"(expected at least {minimum}): the loop stopped servicing work"
        )
        assert self.worst_gap_s < ceiling_s, (
            f"the event loop went quiet for {self.worst_gap_s:.2f}s after {context} "
            f"(ceiling {ceiling_s:g}s): the app froze rather than kept painting"
        )


async def dispose_quietly(*sessions: Any) -> None:
    """Tear sessions down without letting teardown noise mask a real failure.

    Bounded, because a session whose MCP manager is wedged is exactly what
    these tests construct on purpose, and an unbounded dispose in a teardown
    path would convert an informative assertion failure into a hang.
    """
    for session in sessions:
        if session is None:
            continue
        try:
            await asyncio.wait_for(session.dispose(), timeout=10.0)
        except BaseException:  # noqa: BLE001 — teardown must never mask the real error
            pass
