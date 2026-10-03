"""The completion-time project check: the session half of it.

``test_projects_store.py`` pins the store and ``test_project_tool.py`` the
tool; these pin WHEN the project guardrail nudges — and, more importantly,
when it must stay silent. The failure being guarded against is expensive in
both directions: a nudge that fires on a stale set nobody moved burns the
loop's continuation budget and asks the model to re-report nothing, while a
nudge that never fires leaves the operator's project rows quietly rotting —
the exact thing the completion check exists to prevent. The mechanism mirrors
the todo guardrail (``test_todo_guardrail.py``), so these tests are shaped
like its: a scripted provider, the REAL :class:`Session`, the REAL
:class:`AgentLoop`, and (where the turn must have done work) the REAL ``todo``
or ``project`` tool.

The worked-turn guard is the one place the two guardrails deliberately
differ: the project check fires only after at least one tool execution landed
this turn, because a turn that ran no tools cannot have moved a project's
record and a nudge would be asking for a report on nothing.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Literal

import pytest

from local_operator.harness.message_types import (
    PROJECT_REMINDER_MESSAGE_TYPE,
    TODO_REMINDER_MESSAGE_TYPE,
)
from local_operator.harness.render import _default_convert_to_llm
from local_operator.harness.types import (
    AbortSignal,
    ChatRequest,
    CustomMessage,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    TextContent,
    ToolContext,
)
from local_operator.projects import (
    PROJECT_PROGRESS_STALE_S,
    Project,
    ProjectEdit,
    ProjectRegistry,
    stale_projects_fingerprint,
    stale_projects_for_session,
)
from local_operator.session.session import Session, _project_reminder_text
from local_operator.session.transcript import Transcript
from local_operator.tools import builtin
from local_operator.tools.registry import create_tools

MODEL = ModelSpec(provider="test", model_id="m", context_window=100_000)

#: The row validator only stores 12-hex ids (``_SESSION_ID_RE``), so the test
#: session is shaped like one a real transcript directory would have.
SESSION = "abcdef012345"
OTHER_SESSION = "ffffffffffff"


@pytest.fixture(autouse=True)
def clean_todo_store():
    """The todo store is process-global; a leaked list would decide the next
    test's guardrail outcome."""
    builtin.TODO_STORE.clear()
    yield
    builtin.TODO_STORE.clear()


class ScriptedStream:
    """Replays a per-call event script; records every request it received."""

    def __init__(self, turns: list[list[StreamEvent]]) -> None:
        self.turns = turns
        self.requests: list[ChatRequest] = []

    def __call__(self, request: ChatRequest, signal: AbortSignal | None):
        self.requests.append(request)
        # Past the script: answer with a bare stop so a guardrail that nudges
        # more than the test expects fails on the assertion, not an IndexError.
        turn = self.turns[len(self.requests) - 1] if len(self.requests) <= len(self.turns) else []

        async def gen():
            for event in turn:
                yield event
            if not turn:
                yield StreamEndEvent(stop_reason="stop")

        return gen()

    def request_texts(self, index: int) -> list[str]:
        """Every text block the provider was handed on request ``index``."""
        return [
            block.text
            for message in self.requests[index].messages
            for block in message.content
            if isinstance(block, TextContent)
        ]

    def reminders(self, index: int) -> list[str]:
        return [text for text in self.request_texts(index) if "<system-reminder>" in text]


def tool_call(call_id: str, name: str, payload: dict[str, Any]) -> list[StreamEvent]:
    return [
        StreamToolCallDelta(index=0, id=call_id, name=name),
        StreamToolCallDelta(index=0, argument_delta=json.dumps(payload)),
    ]


def prose(text: str) -> list[StreamEvent]:
    return [StreamTextDelta(delta=text), StreamEndEvent(stop_reason="stop")]


def worked_turn(call_id: str = "c-work") -> list[StreamEvent]:
    """A turn that executes a real, harmless tool call — the worked-turn guard
    is the thing under test in most files here, so the work must be REAL."""
    return tool_call(call_id, "project", {"op": "list"})


def reminder_rows(text: str) -> list[str]:
    """The project names a reminder text lists, in order."""
    return [
        line[len("- ") :].split(" ", 1)[0] for line in text.splitlines() if line.startswith("- ")
    ]


def registered(store: ProjectRegistry, name: str, *, session: str = SESSION) -> str:
    """Create a linked project and return its id."""
    project = store.create_project(ProjectEdit(name=name), sessions=[session])
    return project.id


def backdate(store: ProjectRegistry, project_id: str, age_s: float) -> None:
    """Rewrite the row on disk with an older progress stamp.

    The store has no API for "make this stale" (correctly — only time makes a
    record stale), so the test writes the state a real clock would have left
    and re-reads it through the constructor. This is the same disk-patch move
    ``test_projects_store.py`` uses for its refresh-amendment cases.
    """
    path = store.projects_dir / f"{project_id}.json"
    payload = json.loads(path.read_text())
    payload["progress_updated_at"] = time.time() - age_s
    path.write_text(json.dumps(payload))


def stale_store(tmp_path, *names: str, session: str = SESSION) -> ProjectRegistry:
    """A registry whose named projects are linked and stale (a report older
    than the window), returned freshly loaded so every read sees the patched
    rows."""
    store = ProjectRegistry(tmp_path / "config")
    for name in names:
        project_id = registered(store, name, session=session)
        store.update_project(
            project_id, ProjectEdit(progress=f"2026-09-26: {name} moved"), reporter=session
        )
        backdate(store, project_id, PROJECT_PROGRESS_STALE_S + 60)
    return ProjectRegistry(tmp_path / "config")


def make_session(
    tmp_path,
    stream: ScriptedStream,
    *,
    store: ProjectRegistry | None,
    tools: list[Any] | None = None,
    session_id: str = SESSION,
) -> Session:
    if tools is None:
        context = ToolContext(cwd=str(tmp_path), session_id=session_id, project_registry=store)
        tools = create_tools(context, ["project"] if store is not None else ["todo"])
    return Session(
        model=MODEL,
        stream_fn=stream,
        tools=tools,
        transcript=Transcript(tmp_path / "sess"),
        session_id=session_id,
        cwd=str(tmp_path),
        system_blocks_provider=lambda: ["stable"],
        project_registry=store,
    )


# -- the text and the renderer (pure) ---------------------------------------


def test_the_reminder_text_is_the_designed_template() -> None:
    """The discriminating snapshot: a reader comparing the nudge against
    §V2.C.3 must find the exact template, substitutions and all — the label
    that stops the model answering the user about a message they never sent,
    the two row shapes, and the four honest exits plus ``ask``."""
    now = 1_800_000_000.0
    stale = [
        Project(
            id="a" * 12,
            name="payments-migration",
            status="active",
            progress="2026-09-26: landed the store; next is the tool surface.",
            progress_updated_at=now - 7_200,  # 2h
            progress_reported_by=SESSION,
        ),
        Project(id="b" * 12, name="website", status="active"),
    ]

    assert _project_reminder_text(stale, now=now) == (
        "<system-reminder>\n"
        "Injected by the harness at the turn boundary. Not from the user, and "
        "not shown to them.\n"
        "This session is linked to projects whose recorded progress is stale:\n"
        f"- payments-migration [active] — progress last reported 2h ago by session {SESSION}: "
        '"2026-09-26: landed the store; next is the tool surface."\n'
        "- website [active] — no progress recorded\n"
        "Keep the record true: if this turn's work moved a project on, write one "
        "dated line with `project op='update' name='<name>' progress='<line>'`; "
        "if its state changed, `project op='update' name='<name>' "
        "status='paused|done'`. If the recorded progress still describes reality, "
        "`project op='refresh' name='<name>'` records that you checked; it does "
        "not reset the staleness clock, and a reworded re-send is a NEW line, "
        "not a refresh. If this session no longer belongs "
        "to a project, `project op='unlink' name='<name>'`. If a decision here "
        "is the user's to make, put it to them with the `ask` tool.\n"
        "</system-reminder>"
    )


def test_the_reminder_text_caps_rows_and_truncates_the_excerpt() -> None:
    """The nudge stays bounded however many projects are linked and however
    long a progress line has grown: ≤ 4 rows then ``… and N more``, one line
    per row (newlines collapsed), and the excerpt capped with an ellipsis."""
    now = 1_800_000_000.0
    long_line = "x" * 200 + "\nsecond line"
    stale = [
        Project(
            id=f"{index:02d}" + "a" * 10,
            name=f"project-{index}",
            status="active",
            progress=long_line if index == 0 else "a line",
            progress_updated_at=now - 100,
            progress_reported_by=SESSION,
        )
        for index in range(6)
    ]

    text = _project_reminder_text(stale, now=now)

    assert "… and 2 more" in text
    assert "project-4" not in text and "project-5" not in text
    excerpt = text.split('"')[1]
    assert len(excerpt) <= 80 and excerpt.endswith("…")
    assert "\nsecond line" not in text, "a newline in a row would break the frame"
    # The four rows are ONE line each (plus the collapsed extra line, which is
    # not a row); count the rows.
    assert len(reminder_rows(text)) == 4


def test_renderer_passes_the_project_reminder_through() -> None:
    """``_default_convert_to_llm`` is an ALLOW-LIST: an unlisted custom type is
    dropped as bookkeeping, which would leave the loop re-entering with
    nothing for the model to react to."""
    reminder = CustomMessage(
        custom_type=PROJECT_REMINDER_MESSAGE_TYPE,
        attribution="system",
        details={"text": "stale: payments-migration", "fingerprint": ()},
    )

    rendered = _default_convert_to_llm([Message.user("go"), reminder])

    assert [message.role for message in rendered] == ["user", "user"]
    assert rendered[-1].text == "stale: payments-migration"
    assert rendered[-1].id == reminder.id  # entry id preserved, like every branch


def test_renderer_keeps_the_newest_of_each_reminder_type() -> None:
    """The two reminder types are independent claims about different stores,
    so each keeps its own newest: the newest project nudge and the newest todo
    nudge both reach the model, while an older project nudge — whose stale set
    has since moved — drops."""
    old_project = CustomMessage(
        custom_type=PROJECT_REMINDER_MESSAGE_TYPE, attribution="system", details={"text": "old"}
    )
    todo = CustomMessage(
        custom_type=TODO_REMINDER_MESSAGE_TYPE, attribution="system", details={"text": "todos"}
    )
    new_project = CustomMessage(
        custom_type=PROJECT_REMINDER_MESSAGE_TYPE, attribution="system", details={"text": "new"}
    )

    rendered = _default_convert_to_llm([Message.user("go"), old_project, todo, new_project])

    assert [message.text for message in rendered] == ["go", "todos", "new"]


def test_stamped_project_fingerprint_round_trips_and_expires_garbage() -> None:
    """The stamp arrives from a plain dict, where JSON has turned the nested
    tuples into lists; a stamp that cannot be normalised must compare equal to
    nothing (and so expire) rather than raise on a render path.

    FOUR fields since schema 2 — ``(id, status, content stamp, assertion
    stamp)`` — and the arity is load-bearing: a 3-field stamp from a reminder
    built by an older build (or a 5-field one from a newer build) can describe
    nothing this build can compare against, so it must expire rather than
    suppress the nudge it cannot vouch for.
    """
    from local_operator.session.session import _stamped_project_fingerprint

    details = {"fingerprint": [["a" * 12, "active", 123, 456]]}
    assert _stamped_project_fingerprint(details) == (("a" * 12, "active", 123, 456),)
    assert _stamped_project_fingerprint({"fingerprint": [["a" * 12, "active", 123]]}) == ()
    assert _stamped_project_fingerprint({"fingerprint": [["a" * 12, "active", "nope", 1]]}) == ()
    assert _stamped_project_fingerprint({}) == ()


# -- the predicate, driven through the real session -------------------------


@pytest.mark.asyncio
async def test_a_stale_linked_project_is_nudged_at_the_yield(tmp_path) -> None:
    """THE positive case. The turn works (a real tool call), the model then
    tries to stop twice; the first yield re-asserts the stale row, the second
    is released by the latch (§12 slice 4a (i) and (ii))."""
    store = stale_store(tmp_path, "payments-migration")
    stream = ScriptedStream([worked_turn(), prose("Nothing else needed."), prose("Still done.")])
    session = make_session(tmp_path, stream, store=store)

    await session.prompt("check the project")

    assert len(stream.requests) == 3, "one nudge, then the turn is allowed to end"
    nudges = stream.reminders(2)
    assert len(nudges) == 1
    assert "payments-migration" in nudges[0]
    assert "progress last reported" in nudges[0] and "ago" in nudges[0]
    await session.dispose()


@pytest.mark.asyncio
async def test_the_reminder_leaves_no_trace_in_the_transcript_or_the_event_stream(
    tmp_path,
) -> None:
    """The nudge is model-visible and user-invisible (§12 slice 4a (i)): nothing
    persists it, so a resume never replays a stale claim about a record that
    has moved on, and no event hands it to a viewer."""
    store = stale_store(tmp_path, "payments-migration")
    stream = ScriptedStream([worked_turn(), prose("Nothing else needed."), prose("Still done.")])
    session = make_session(tmp_path, stream, store=store)
    events: list[object] = []
    session.subscribe(events.append)

    await session.prompt("check the project")

    assert len(stream.reminders(2)) == 1, "the nudge did fire — so a leak would be meaningful"
    raw = (tmp_path / "sess" / "transcript.jsonl").read_text()
    assert "project_reminder" not in raw
    assert "<system-reminder>" not in raw
    # Every event shape this session emits is checked by repr, not by one known
    # attribute: a nudge leaking through any field is the failure, not just a
    # nudge leaking through `text`.
    assert not any("<system-reminder>" in repr(event) for event in events)
    # Nor does it survive into a replayed history.
    assert not any(
        "<system-reminder>" in getattr(block, "text", "")
        for message in Transcript(tmp_path / "sess").build_llm_history()
        for block in getattr(message, "content", [])
    )
    await session.dispose()


@pytest.mark.asyncio
async def test_a_project_with_no_progress_is_stale_by_construction(tmp_path) -> None:
    """A linked project that has never carried a line is stale by
    construction (§V2.C): the first honest line is still owed, and the nudge
    asks for it explicitly rather than quoting an age that does not exist."""
    store = ProjectRegistry(tmp_path / "config")
    registered(store, "brand-new")
    stream = ScriptedStream([worked_turn(), prose("Nothing to add.")])
    session = make_session(tmp_path, stream, store=store)

    await session.prompt("check")

    assert len(stream.requests) == 3
    nudges = stream.reminders(2)
    assert len(nudges) == 1
    assert "- brand-new [active] — no progress recorded" in nudges[0]
    await session.dispose()


@pytest.mark.asyncio
async def test_a_turn_with_no_tool_calls_is_not_nudged(tmp_path) -> None:
    """The worked-turn guard: prose cannot have moved a project's record, so
    even a stale linked project is not worth a re-entry (§12 slice 4a (v))."""
    store = stale_store(tmp_path, "payments-migration")
    stream = ScriptedStream([prose("All good."), prose("Really all good.")])
    session = make_session(tmp_path, stream, store=store)

    await session.prompt("just chatting")

    assert len(stream.requests) == 1
    assert not any(stream.reminders(index) for index in range(len(stream.requests)))
    await session.dispose()


@pytest.mark.asyncio
async def test_a_freshly_reported_project_is_not_nudged(tmp_path) -> None:
    """A report inside the staleness window is current by definition; nudging
    it would ask the model to repeat what it just wrote."""
    store = ProjectRegistry(tmp_path / "config")
    project_id = registered(store, "payments-migration")
    store.update_project(
        project_id, ProjectEdit(progress="2026-09-26: just reported"), reporter=SESSION
    )
    stream = ScriptedStream([worked_turn(), prose("Done.")])
    session = make_session(tmp_path, stream, store=store)

    await session.prompt("check")

    assert len(stream.requests) == 2
    assert not any(stream.reminders(index) for index in range(len(stream.requests)))
    await session.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["paused", "done", "archived"])
async def test_settled_projects_are_never_named(
    tmp_path, status: Literal["paused", "done", "archived"]
) -> None:
    """paused/done/archived are deliberate statements that the record is
    settled — a reminder about one would nag the session to revive it."""
    store = ProjectRegistry(tmp_path / "config")
    project_id = registered(store, "payments-migration")
    store.update_project(
        project_id, ProjectEdit(progress="2026-09-26: wrapped up"), reporter=SESSION
    )
    store.update_project(project_id, ProjectEdit(status=status))
    backdate(store, project_id, PROJECT_PROGRESS_STALE_S + 60)
    store = ProjectRegistry(tmp_path / "config")
    stream = ScriptedStream([worked_turn(), prose("Done.")])
    session = make_session(tmp_path, stream, store=store)

    await session.prompt("check")

    assert len(stream.requests) == 2
    assert not any(stream.reminders(index) for index in range(len(stream.requests)))
    await session.dispose()


@pytest.mark.asyncio
async def test_a_project_linked_to_another_session_is_not_nudged(tmp_path) -> None:
    """The reverse lookup is by THIS session's id: another session's project
    is not this session's row to keep current."""
    store = stale_store(tmp_path, "payments-migration", session=OTHER_SESSION)
    stream = ScriptedStream([worked_turn(), prose("Done.")])
    session = make_session(tmp_path, stream, store=store)

    await session.prompt("check")

    assert len(stream.requests) == 2
    assert not any(stream.reminders(index) for index in range(len(stream.requests)))
    await session.dispose()


@pytest.mark.asyncio
async def test_a_host_with_no_project_registry_is_not_nudged(tmp_path) -> None:
    """No registry means the host keeps no projects: there is nothing to look
    up and nothing to refresh, and the hook must return clean the same way the
    todo one does for an empty list."""
    stream = ScriptedStream([tool_call("c1", "todo", {"op": "view"}), prose("Done.")])
    session = make_session(tmp_path, stream, store=None)

    await session.prompt("check")

    assert len(stream.requests) == 2
    assert not any(stream.reminders(index) for index in range(len(stream.requests)))
    await session.dispose()


@pytest.mark.asyncio
async def test_an_identical_stale_set_is_nudged_exactly_once(tmp_path) -> None:
    """The latch. A model that yields twice on a byte-identical stale set is
    stuck; nudging it again would spend the loop's continuation budget, and
    only the reminder's own exits (update / refresh / unlink) move the set."""
    store = stale_store(tmp_path, "payments-migration")
    stream = ScriptedStream(
        [worked_turn(), prose("Nothing to report."), prose("Still nothing."), prose("Stopping.")]
    )
    session = make_session(tmp_path, stream, store=store)

    await session.prompt("check")

    assert len(stream.requests) == 3, "one nudge, then the turn ends — not re-nudged every yield"
    live = [
        message
        for message in session._context.messages
        if isinstance(message, CustomMessage)
        and message.custom_type == PROJECT_REMINDER_MESSAGE_TYPE
    ]
    assert len(live) == 1
    assert session._project_reminder_fingerprint == stale_projects_fingerprint(
        stale_projects_for_session(store, SESSION)
    )
    await session.dispose()


@pytest.mark.asyncio
async def test_progress_earns_another_nudge_for_the_remaining_projects(tmp_path) -> None:
    """Movement re-arms the guardrail: an update moves the stale set, so the
    REMAINING stale project earns a fresh nudge in the same turn — and the
    superseded reminder expires from the render (§12 slice 4a (vi))."""
    store = stale_store(tmp_path, "alpha", "beta")
    stream = ScriptedStream(
        [
            worked_turn(),
            prose("Holding."),
            tool_call(
                "c-update",
                "project",
                {"op": "update", "name": "alpha", "progress": "2026-09-26: alpha moved"},
            ),
            prose("Alpha updated."),
            prose("Nothing else."),
        ]
    )
    session = make_session(tmp_path, stream, store=store)

    await session.prompt("check")

    assert len(stream.requests) == 5, "nudge, update, nudge for what remains, then released"
    assert sorted(reminder_rows(stream.reminders(2)[0])) == ["alpha", "beta"]
    assert stream.reminders(3) == [], "the superseded reminder expires the moment the set moves"
    assert reminder_rows(stream.reminders(4)[0]) == ["beta"]
    await session.dispose()


@pytest.mark.asyncio
async def test_a_fresh_user_turn_rearms_the_latch(tmp_path) -> None:
    """The latch is per user turn: a user who says "carry on" after a released
    turn must get the guardrail back, even though the stale set is unchanged."""
    store = stale_store(tmp_path, "payments-migration")
    stream = ScriptedStream(
        [
            worked_turn(),
            prose("Nothing written."),
            prose("Still nothing."),
            worked_turn("c-work-2"),
            prose("Nothing again."),
            prose("Stopping for real."),
        ]
    )
    session = make_session(tmp_path, stream, store=store)

    await session.prompt("check")
    assert len(stream.requests) == 3
    assert len(stream.reminders(2)) == 1

    await session.prompt("carry on")

    assert len(stream.requests) == 6, "the new turn must be nudged despite the same stale set"
    assert len(stream.reminders(5)) == 1
    await session.dispose()


@pytest.mark.asyncio
async def test_the_reminder_expires_when_the_record_moves(tmp_path) -> None:
    """A reminder is a point-in-time assertion. Once the record is refreshed it
    must stop being SENT — as a render decision, never a rewrite of the live
    list behind the loop's back."""
    store = stale_store(tmp_path, "payments-migration")
    stream = ScriptedStream([worked_turn(), prose("Nothing."), prose("Nothing.")])
    session = make_session(tmp_path, stream, store=store)
    await session.prompt("check")
    project_id = stale_projects_for_session(store, SESSION)[0].id

    # While the record stands still the assertion is still true, so it is still sent.
    still_sent = session._render_history(session._context.messages)
    assert any("<system-reminder>" in message.text for message in still_sent)

    store.update_project(
        project_id, ProjectEdit(progress="2026-09-26: refreshed"), reporter=SESSION
    )

    rendered = session._render_history(session._context.messages)
    assert not any("<system-reminder>" in message.text for message in rendered)
    # Expiry is a RENDER decision: the live list is never rewritten.
    assert any(
        isinstance(message, CustomMessage) and message.custom_type == PROJECT_REMINDER_MESSAGE_TYPE
        for message in session._context.messages
    )
    await session.dispose()


@pytest.mark.asyncio
async def test_a_mixed_todos_and_projects_batch_reenters_exactly_once(tmp_path) -> None:
    """The budget case (§V2.C.5): both guardrails fire at the same yield, the
    batch is ONE re-entry carrying both reminders, and the loop's charging
    rule — a follow-up in the batch wins, charged to the follow-up budget —
    is exactly what the single wrapper hook preserves."""
    store = stale_store(tmp_path, "payments-migration")
    builtin.TODO_STORE[SESSION] = [{"text": "decide the domain", "status": "pending"}]
    stream = ScriptedStream([worked_turn(), prose("Nothing final."), prose("Still nothing.")])
    session = make_session(tmp_path, stream, store=store)

    await session.prompt("check everything")

    assert len(stream.requests) == 3, "todos + projects in one batch re-enter ONCE"
    both = stream.reminders(2)
    assert len(both) == 2, "both reminders ride the one re-entry"
    assert any("decide the domain" in text for text in both)
    assert any("payments-migration" in text for text in both)
    await session.dispose()


@pytest.mark.asyncio
async def test_no_nudge_fires_while_the_turn_is_parked_on_an_ask(tmp_path, monkeypatch) -> None:
    """The ``ask`` interaction. A parked question means the loop has not
    reached a yield, so no nudge can fire while the turn waits on the user;
    once the question settles, the boundary nudge still works.

    PARKING IS THE KILL-SWITCH ARM: the queued ask (the shipped default since
    2026-10-03) returns a receipt and the turn carries on, so there is no parked
    turn for this cell to assert about. Selecting the blocking arm explicitly is
    what keeps the pinned property pinned — it is a property of the arm that
    stops the loop, not of ``ask`` in general.
    """
    from local_operator.asks import policy

    monkeypatch.setattr(policy, "NONBLOCKING_ASK", False)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def ask_user(questions):
        entered.set()
        await release.wait()
        return {}

    store = stale_store(tmp_path, "payments-migration")
    context = ToolContext(
        cwd=str(tmp_path), session_id=SESSION, project_registry=store, ask_user=ask_user
    )
    tools = create_tools(context, ["ask", "project"])
    stream = ScriptedStream(
        [
            tool_call(
                "c-ask",
                "ask",
                {
                    "questions": [
                        {
                            "id": "pick",
                            "question": "Which one?",
                            "options": [{"label": "a"}, {"label": "b"}],
                        }
                    ]
                },
            ),
            prose("No answer — carrying on."),
            prose("Nothing else."),
        ]
    )
    session = make_session(tmp_path, stream, store=store, tools=tools)
    session.set_ask_handler(ask_user)

    task = asyncio.create_task(session.prompt("ask then work"))
    await asyncio.wait_for(entered.wait(), 30)

    # Parked on the human: the tool batch has not settled, so there is no yield
    # and no nudge — and no second provider request.
    assert len(stream.requests) == 1
    assert not any(
        isinstance(message, CustomMessage) and message.custom_type == PROJECT_REMINDER_MESSAGE_TYPE
        for message in session._context.messages
    )

    release.set()
    await asyncio.wait_for(task, 60)

    assert len(stream.requests) == 3, "after the ask settles, the boundary nudge still fires"
    assert len(stream.reminders(2)) == 1
    await session.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["planning", "qa", "validation"])
async def test_lifecycle_in_flight_statuses_are_nudged(tmp_path, status) -> None:
    """planning/qa/validation are work in flight: the completion check watches
    them exactly as it watches active."""
    store = ProjectRegistry(tmp_path / "config")
    project_id = registered(store, "payments-migration")
    store.update_project(project_id, ProjectEdit(progress="2026-09-26: moved"), reporter=SESSION)
    store.update_project(project_id, ProjectEdit(status=status))
    backdate(store, project_id, PROJECT_PROGRESS_STALE_S + 60)
    store = ProjectRegistry(tmp_path / "config")
    stream = ScriptedStream([worked_turn(), prose("Done.")])
    session = make_session(tmp_path, stream, store=store)

    await session.prompt("check")

    assert any(stream.reminders(index) for index in range(len(stream.requests)))
    await session.dispose()
