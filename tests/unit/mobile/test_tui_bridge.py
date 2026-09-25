"""The TUI auto-registers with the mobile control plane when a session is
adopted — this pins that contract: record published, control socket answers,
slash commands land through the app's own dispatch, and unmount unpublishes.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.harness.types import (
    AgentStartEvent,
    Message,
    MessageUpdateEvent,
    TextContent,
    ToolExecutionEndEvent,
    ToolResult,
)
from local_operator.mobile.tui_handle import (
    TuiSessionHandle,
    _DetailChangedDuringHydration,
)
from tests.unit.tui.test_app_pilot import FakeSession, _factory

#: The generated formatter fixture, one directory up from the bundle it pins.
FORMATTER_PARITY = (
    Path(__file__).resolve().parents[3] / "local_operator/mobile/web/src/lib/format.parity.json"
)


@pytest.mark.asyncio
async def test_tui_same_id_concurrent_steers_cross_thread_once() -> None:
    class Session(FakeSession):
        def __init__(self) -> None:
            super().__init__()
            self.steer_calls: list[tuple[str, str | None]] = []

        def steer(self, text, images=None, *, message_id=None):  # noqa: ANN001, ANN202
            self.steer_calls.append((text, message_id))

    class App:
        def __init__(self, session) -> None:  # noqa: ANN001
            self._session = session
            self._loop_lock = threading.Lock()

        def call_from_thread(self, callback) -> None:  # noqa: ANN001
            # The callback is the Textual loop's atomic admission section. This
            # fake runs it inline while concurrent bridge tasks contend for it,
            # so it must hold a lock to keep it atomic: ``_on_app`` performs the
            # enqueue on a WORKER thread (``asyncio.to_thread``), which means two
            # concurrent hops call this from two threads at once — where the
            # real ``App.call_from_thread`` cannot, because a Textual app runs
            # its callbacks one at a time on one thread. Without the lock the
            # double lets two steers into the admission section simultaneously
            # and fails a test about a guarantee the product keeps.
            with self._loop_lock:
                callback()

    session = Session()
    handle = TuiSessionHandle(App(session))  # type: ignore[arg-type]
    receipts = await asyncio.gather(
        handle.steer("correction", command_id="same-id"),
        handle.steer("correction", command_id="same-id"),
    )

    assert receipts == ["steering queued", "already admitted"]
    assert session.steer_calls == [("correction", "same-id")]
    assert [row.text for row in handle._fold.projection.transcript] == ["correction"]


@pytest.mark.asyncio
async def test_tui_stalled_steers_apply_owner_loop_backpressure() -> None:
    class Session(FakeSession):
        def __init__(self) -> None:
            super().__init__()
            self.steer_calls: list[str] = []

        def steer(self, text, images=None, *, message_id=None):  # noqa: ANN001, ANN202
            assert isinstance(message_id, str)
            self.steer_calls.append(message_id)

    class App:
        def __init__(self, session) -> None:  # noqa: ANN001
            self._session = session

        def call_from_thread(self, callback) -> None:  # noqa: ANN001
            callback()

    session = Session()
    handle = TuiSessionHandle(App(session))  # type: ignore[arg-type]
    for index in range(32):
        assert await handle.steer(str(index), command_id=f"id-{index}") == "steering queued"
    assert await handle.steer("duplicate", command_id="id-0") == "already admitted"
    with pytest.raises(RuntimeError, match=r"steering queue is full \(32\)"):
        await handle.steer("overflow", command_id="overflow")
    assert len(session.steer_calls) == 32


@pytest.mark.asyncio
async def test_tui_hosted_aside_applies_the_instruction_only_when_asked() -> None:
    """The TUI-HOSTED seam wraps by default and leaves a declared ask alone.

    Same rule as ``ServingSessionHandle``, at the other remote hop: a remote
    caller that sends RAW turns (the phone's quick-ask, the desktop app attached
    to this TUI) is instructed HERE, while a caller that supplied its own — the
    TUI's ``/btw`` overlay, its goal judge — says so and gets its turns
    untouched. Both directions are asserted against the turns the owner session
    actually received, because a double wrap is invisible in the answer.
    """
    from local_operator.session.aside import ASIDE_PROMPT

    class Session(FakeSession):
        def __init__(self) -> None:
            super().__init__()
            self.turns: list[list[Any]] = []

        async def complete_aside(  # noqa: ANN001, ANN202
            self, turns, *, aside_instruction=True, on_delta=None, on_usage=None
        ):
            self.turns.append(list(turns))
            return "aside answer"

    class App:
        """Runs ``call_from_thread`` callbacks ON the loop thread, as Textual does.

        The hopped work here is not just any callback: ``TuiSessionHandle`` hands
        over ``asyncio.get_running_loop`` itself, so a double that runs it inline
        on the worker thread (the other tests' fakes do, and their callbacks are
        loop-independent) makes that call raise. ``call_soon_threadsafe`` plus a
        latch reproduces the real handoff: the callback executes on the loop
        thread and the caller blocks until it is done.
        """

        def __init__(self, session) -> None:  # noqa: ANN001
            self._session = session
            self._loop = asyncio.get_running_loop()

        def call_from_thread(self, callback) -> None:  # noqa: ANN001
            done = threading.Event()

            def run() -> None:
                try:
                    callback()
                finally:
                    done.set()

            self._loop.call_soon_threadsafe(run)
            assert done.wait(timeout=5.0), "the app loop never ran the hop"

    session = Session()
    handle = TuiSessionHandle(App(session))  # type: ignore[arg-type]
    already = ASIDE_PROMPT.format(question="why?")

    await handle.complete_aside(
        [{"role": "user", "content": [{"type": "text", "text": already}]}],
        aside_instruction=False,
    )
    assert session.turns[-1][-1].text == already
    assert session.turns[-1][-1].text.count("<aside>") == 1

    await handle.complete_aside([{"role": "user", "content": [{"type": "text", "text": "why?"}]}])
    assert session.turns[-1][-1].text == ASIDE_PROMPT.format(question="why?")


@pytest.mark.asyncio
async def test_nested_child_detail_events_refresh_after_warm(monkeypatch) -> None:
    class Comms:
        def __init__(self) -> None:
            self.listener = None
            self.child = SimpleNamespace(
                job_id="nested",
                label="nested",
                session_dir=Path("/tmp/nested"),
                parent_job_id="parent",
                session_id="nested",
                prompt="",
                effort="",
                agent_role="task",
                launch_message_id=None,
            )

        def roster_pass(self):  # noqa: ANN201
            # The fold reads one pass; this fake IS the pass, so the same three
            # members answer (see ``SubagentComms.roster_pass``).
            return self

        def roster(self):  # noqa: ANN201
            return []

        def nodes(self):  # noqa: ANN201
            return [self.child]

        def job(self, job_id):  # noqa: ANN001, ANN201
            return None

        def node(self, job_id):  # noqa: ANN001, ANN201
            return self.child if job_id == "nested" else None

        def subscribe_detail_changes(self, listener):  # noqa: ANN001, ANN201
            self.listener = listener
            return lambda: None

    comms = Comms()
    session = FakeSession()
    setattr(session, "_subagent_comms", comms)
    app = SimpleNamespace(_session=session)
    handle = TuiSessionHandle(app)  # type: ignore[arg-type]
    invalidated: list[str] = []
    monkeypatch.setattr(handle, "_invalidate_subagent_detail", invalidated.append)
    handle.subscribe(lambda: None)
    assert invalidated == ["nested"]  # initial warm
    assert comms.listener is not None
    comms.listener("nested")
    assert invalidated == ["nested", "nested"]  # later nested mutation


@pytest.mark.asyncio
async def test_dirty_hydration_retries_without_later_event(monkeypatch) -> None:
    node = SimpleNamespace(job_id="nested", session_dir=Path("/tmp/nested"))
    comms = SimpleNamespace(node=lambda job_id: node)
    session = FakeSession()
    setattr(session, "_subagent_comms", comms)
    handle = TuiSessionHandle(SimpleNamespace(_session=session))  # type: ignore[arg-type]
    handle._detail_generations["nested"] = 1
    calls = 0

    async def fake_to_thread(fn, session_dir):  # noqa: ANN001, ANN202
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _DetailChangedDuringHydration
        return (1, 1), []

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(handle._fold, "set_subagent_hydrated_details", lambda *args: True)
    await handle._hydrate_subagent_detail("nested")
    assert calls == 2


@pytest.mark.asyncio
async def test_tui_auto_registers_and_answers_control() -> None:
    from local_operator.tui.app import OperatorApp

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        for _ in range(50):
            if app._mobile_registrant is not None:
                break
            await pilot.pause(0.1)
        assert app._mobile_registrant is not None, "mobile registrant never started"

        from local_operator.session.runtime import registry

        # The registrant EXISTING and its record being on disk are two
        # different async steps: publication is a thread hop, so on a slow
        # runner the scan lands in between and reads an empty store. Poll for
        # the state the assertion is about (CI shard 0, 3.12) rather than
        # betting the first scan wins the race.
        records = registry.scan()
        for _ in range(50):
            if records:
                break
            await pilot.pause(0.1)
            records = registry.scan()
        assert records, "no discovery record published"
        record, state = records[0]
        assert state == "live"
        assert record.kind == "tui"

        reader, writer = await asyncio.open_connection("127.0.0.1", record.control_port)
        writer.write(json.dumps({"key": record.control_key}).encode() + b"\n")
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout=5)
        frame = json.loads(line)
        assert frame["op"] == "projection"
        assert frame["data"]["model_label"] == record.model_label

        writer.write(
            json.dumps({"op": "slash", "req": 1, "command": "goal", "args": "test goal"}).encode()
            + b"\n"
        )
        await writer.drain()
        acked = None
        for _ in range(10):
            line = await asyncio.wait_for(reader.readline(), timeout=5)
            frame = json.loads(line)
            if frame.get("op") == "ack":
                acked = frame
                break
        assert acked is not None and "goal" in acked["detail"]
        writer.close()

    assert not registry.scan(), "record was not unpublished on exit"


def test_the_started_hook_is_wired_and_reseeded_on_rebind(tmp_path) -> None:
    """Q2: ``TuiSessionHandle`` must wire ``session._publish_session_started``
    the way ``ServingSessionHandle`` does, or a TUI-owned ``kind="tui"`` record
    stays ``started=False`` forever. Rebind must RE-wire it on the new session
    object and re-seed the registrant's bit from the new session's own durable
    history: a ``/new`` (no message rows) drops back to False — the composer
    window — while a ``/resume`` (message rows on disk) reads True."""

    class _Session(FakeSession):
        # The attribute ServingSessionHandle/TuiSessionHandle wire onto a real
        # Session; the fake does not carry it, so the test stands it up.
        def __init__(self, session_id: str, transcript_rows: list[str]) -> None:  # noqa: ANN001
            super().__init__()
            self._publish_session_started: Any = None
            self._sid = session_id
            path = tmp_path / session_id / "transcript.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(row + "\n" for row in transcript_rows))
            self.transcript_path = path

        # FakeSession pins a constant "sess"; the rebind path copies this
        # onto the projection, so the test needs per-session values.
        @property
        def session_id(self) -> str:
            return self._sid

    class _App:
        def __init__(self, session) -> None:  # noqa: ANN001
            self._session = session

    class _Registrant:
        def __init__(self) -> None:
            self.resets: list[bool] = []
            self.started_calls: list[bool] = []

        def reset_record_started(self, started: bool) -> None:
            self.resets.append(started)

        def set_record_started(self, started: bool) -> None:
            self.started_calls.append(started)

    fresh = _Session("fresh-id", [])
    resumed = _Session(
        "resumed-id",
        [
            '{"id":"m1","ts":1,"type":"custom","payload":{"custom_type":"title"}}',
            '{"id":"m2","ts":2,"type":"message","payload":{"role":"user"}}',
        ],
    )
    registrant = _Registrant()
    handle = TuiSessionHandle(_App(fresh))  # type: ignore[arg-type]
    handle._registrant = registrant  # type: ignore[attr-defined]

    # Wired at construction: the session's hook reaches the handle. Bound
    # methods are compared with == (each attribute access mints a new object).
    assert fresh._publish_session_started == handle._publish_session_started
    fresh._publish_session_started()
    assert registrant.started_calls == [True]

    # /resume: message history on disk re-seeds the bit True even though THIS
    # process has not run a turn for it.
    handle._app = _App(resumed)  # type: ignore[assignment]
    handle.rebind()
    assert resumed._publish_session_started == handle._publish_session_started
    assert registrant.resets == [True]

    # /new after that working conversation: no message rows, bit drops False.
    newer = _Session("newer-id", ['{"id":"t1","ts":3,"type":"custom","payload":{}}'])
    handle._app = _App(newer)  # type: ignore[assignment]
    handle.rebind()
    assert newer._publish_session_started == handle._publish_session_started
    assert registrant.resets == [True, False]

    # QA Q4: a transcript whose only message rows are quiet-dialled peer
    # notes (``peer_message`` CustomMessages, persisted without a turn) is
    # NOT durable history — the bit must stay False on rebind to it.
    noted = _Session(
        "noted-id",
        [
            '{"id":"p1","ts":4,"type":"message","payload":{"kind":"custom",'
            '"custom_type":"peer_message","attribution":"user","details":{"text":"hi"}}}',
        ],
    )
    handle._app = _App(noted)  # type: ignore[assignment]
    handle.rebind()
    assert registrant.resets == [True, False, False]


@pytest.mark.asyncio
async def test_the_attach_seeds_a_live_calls_duration_from_the_owner() -> None:
    """The WIRING, not just the fold: attaching must hand over the instants.

    ``ProjectionFold.reconcile_clocks`` consuming a producer's start instant is
    pinned in ``test_projection.py``, and that is not enough on its own: the
    seed lives at the CALL SITE, beside ``reconcile_streaming``, so a refactor
    that drops the call leaves every fold test green while the reported defect —
    a phone attaching onto work in flight measuring it from the attach — comes
    straight back.

    So this drives the real handle over the real attach path and asserts the
    reading ONLY the seed can produce: the call ends with no ``duration_s`` of
    its own (optional on the wire), which leaves the fold measuring from the
    instant the attach seeded. Without the seed that reading is ~0s.
    """

    class Epochs(FakeSession):
        """A session publishing the two folded anchors, as ``Session`` does."""

        epochs: dict[str, float | None] = {}
        phase: tuple[str, float | None] = ("", None)

        def live_tool_start_epochs(self) -> dict[str, float | None]:
            return dict(self.epochs)

        def activity_phase_clock(self) -> tuple[str, float | None]:
            return self.phase

    class App:
        def __init__(self, session: Any) -> None:
            self._session = session

        def call_from_thread(self, callback: Any) -> None:
            callback()

    session = Epochs()
    session.streaming = True
    session.epochs = {"c1": time.time() - 180.0}
    handle = TuiSessionHandle(App(session))  # type: ignore[arg-type]
    handle.subscribe(lambda: None)

    session.emit(
        ToolExecutionEndEvent(
            tool_call_id="c1",
            tool_name="bash",
            result=ToolResult(tool_call_id="c1", content=[TextContent(text="4")]),
        )
    )
    row = [entry for entry in handle.session_projection_seed.transcript if entry.kind == "tool"][0]
    assert row.tool_state == "done"
    assert row.elapsed_s == pytest.approx(
        180.0, abs=2.0
    ), "a call that began before the phone attached must not be measured from the attach"


def test_the_phone_formatter_fixture_still_matches_the_tuis_formatter() -> None:
    """The other half of the formatter bridge (review round 2, MINOR 4).

    `local_operator/mobile/web/src/lib/format.ts::formatElapsed` is a port of
    `tui/widgets/tool_card.py::format_duration`, and the phone's band, its tool
    rows and its subagent rows all print its output. The two are pinned by ONE
    generated artifact — `local_operator/mobile/web/src/lib/format.parity.json`,
    written by `scripts/generate_clock_format_parity.py` — which the vitest suite
    asserts `formatElapsed` against and this test asserts `format_duration`
    against. So a change to either formatter fails a suite in its own tree, and
    re-aligning them means regenerating the fixture and then making the other
    side agree; neither can drift in silence while both suites stay green.

    The fixture lives under the web bundle on purpose: that path is the
    mobile-web workflow's own filter, so regenerating it is what makes the vitest
    half run in CI.
    """
    from local_operator.tui.widgets.tool_card import format_duration

    fixture = json.loads(FORMATTER_PARITY.read_text())
    cases = fixture["cases"]
    assert len(cases) > 40, "the fixture must keep covering every branch and crossing"
    for seconds, expected in cases:
        assert format_duration(float(seconds)) == expected, seconds


@pytest.mark.asyncio
async def test_the_hand_off_redates_the_bands_age_from_the_fold_that_is_fed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review round 3, MAJOR 1 — the hand-off half, through the attach path.

    The seed a viewer gets IS the object the runtime serializes, and it carries
    the age as of the phase's last edge. A second viewer — a reconnect, a second
    phone, "resume that session" — attaching to a live fold mid-phase therefore
    seeded on that stale number: at a known zero, `0s` counting from the new
    viewer's own mount, which is the operator-reported defect rendered as a
    fabricated zero on the surface this PR exists for.

    `redate_from_phase` is the hand-off the runtime performs before it serializes
    a frame, and it is deliberately NOT the seed property: that one is read for
    identity too, and a mutating getter is what round 4's NIT 2 asked to remove.
    The pushed-frame half of the same fix is pinned end to end, over a real
    runtime and a real daemon dial, in
    `tests/unit/session/runtime/test_server.py::test_a_pushed_frame_carries_the_bands_age_from_the_fold_events_reach`.

    Driven with the fold's own clock, and with the deltas the reviewer measured
    this window with: prose streams them continuously, and none re-enters the
    label's arm, so the stored snapshot does not move on its own.
    """
    import local_operator.mobile.projection as projection_module
    from tests.unit.mobile.test_projection import _StubClock

    class App:
        def __init__(self, session: Any) -> None:
            self._session = session

        def call_from_thread(self, callback: Any) -> None:
            callback()

    clock = _StubClock()
    monkeypatch.setattr(projection_module, "time", clock)
    session = FakeSession()
    session.streaming = True
    handle = TuiSessionHandle(App(session))  # type: ignore[arg-type]
    handle.subscribe(lambda: None)

    # The fold watches this turn begin, so the prose edge is one it witnessed —
    # its own true zero, not a number counted from the attach.
    session.emit(AgentStartEvent(generation=1))
    session.emit(MessageUpdateEvent(message=Message.assistant(), delta="Here "))
    assert handle.session_projection_seed.activity == "responding"
    assert handle.session_projection_seed.activity_started_s == 0.0, "the viewer's own edge"

    clock.advance(45)
    session.emit(MessageUpdateEvent(message=Message.assistant(), delta="more prose "))
    assert handle.session_projection_seed.activity_started_s == 0.0, "the stored snapshot is stale"
    handle.redate_from_phase()  # what the runtime does before serializing a frame
    assert handle.session_projection_seed.activity_started_s == pytest.approx(
        45.0, abs=0.2
    ), "a viewer attaching 45s into the phase must be served the phase's age, not zero"


def test_the_formatter_fixture_is_what_its_generator_produces() -> None:
    """Review round 3, NIT 1: the fixture is pinned to its GENERATOR, not just
    to its content.

    Both suites assert the fixture's content against their own formatter, which
    catches a formatter drifting but says nothing about the FILE: a hand edit
    that rewrote the cases while keeping the list longer than the suites' ``>40``
    bound would shrink coverage with every test still green. ``render()`` is the
    generator's own bytes, so comparing against it is the provenance half — and
    it is here rather than only behind the script's ``--check`` flag because CI
    runs this suite.
    """
    from scripts.generate_clock_format_parity import FIXTURE, render

    assert FORMATTER_PARITY == FIXTURE, "the generator writes the file this suite reads"
    assert FORMATTER_PARITY.read_text() == render()


@pytest.mark.asyncio
async def test_tui_hop_never_parks_the_runtime_loop() -> None:
    """No code on the runtime's loop may perform a synchronous cross-thread wait.

    The invariant behind the operator's "a busy session reads as wedged": a TUI
    runtime serves its control socket, its heartbeat, its attention ticks and
    its projection pushes from ONE loop, and ``app.call_from_thread`` does not
    merely enqueue — it enqueues and BLOCKS until Textual runs the callback. So a
    hop taken directly from a coroutine parks that whole loop behind the app's
    queue. Measured on 2026-09-18 against a real ``OperatorApp``: with one client
    registered and the app loop held 50.5 s, the control thread sat 50.1 s
    inside the enqueue, a fresh dial got no welcome within 20 s, and the
    discovery record aged to 45.1 s and read ``wedged`` on a live idle pid.

    Asserted STRUCTURALLY, as thread identity, because that cannot flake: the
    thread that performs the blocking enqueue must not be the loop's thread.
    The liveness half below is the same fact stated positively — a task on that
    loop makes progress while the app still holds the callback — and both are
    deterministic, since the fake's own wait is bounded rather than counted on.
    """
    loop_thread = threading.get_ident()
    entered = threading.Event()
    release = threading.Event()
    hop_threads: list[int] = []

    class Session(FakeSession):
        def refresh_attention(self) -> dict[str, Any]:
            # The exact call the runtime's ``_attention_loop`` makes once per
            # second for as long as any client is registered.
            return {"unseen": 0}

    class App:
        def __init__(self, session: Any) -> None:
            self._session = session

        def call_from_thread(self, callback: Any) -> None:
            hop_threads.append(threading.get_ident())
            entered.set()
            # Bounded so the pre-fix shape reports the assertion below instead
            # of hanging the suite; the assertion is the thread identity, not
            # this wait.
            release.wait(10.0)
            callback()

    handle = TuiSessionHandle(App(Session()))  # type: ignore[arg-type]
    hop = asyncio.create_task(handle.refresh_attention())
    assert await asyncio.to_thread(entered.wait, 5), "the hop never reached the app"

    assert hop_threads and hop_threads[0] != loop_thread, (
        "the cross-thread hop performed its blocking enqueue on the runtime's own "
        "loop — every accept, ping and heartbeat behind it stops for as long as "
        "the app is busy"
    )

    ticks = 0

    async def spin() -> None:
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0)

    ticker = asyncio.create_task(spin())
    # Bounded by loop TURNS, not seconds: a free loop needs a handful of them,
    # and a parked one never reaches the count at all.
    for _ in range(200):
        if ticks >= 5:
            break
        await asyncio.sleep(0)
    assert ticks >= 5, "the runtime's loop made no progress while the hop was parked"

    release.set()
    assert await asyncio.wait_for(hop, timeout=5) == {"unseen": 0}
    ticker.cancel()


@pytest.mark.asyncio
async def test_tui_hop_awaits_an_awaitable_result() -> None:
    """An async callback's coroutine is awaited, never returned as its result.

    THE DEFECT THIS PINS (review round 1, U1). Several callbacks handed to
    ``_on_app`` are async on the session — ``Session.refresh_attention``,
    ``acknowledge_attention`` and ``record_shell`` all are — so ``fn()`` returns
    a coroutine OBJECT. The first version of the hop returned that object as the
    answer, and the cost was not a cosmetic one: ``refresh_attention`` stores its
    answer in ``Projection.attention``, so the projection carried a coroutine
    and every push then failed to serialize — reproduced 2026-09-19 with the
    product's own attach clients, where the FIRST viewer was dropped the moment
    a SECOND dialled ("owner closed the connection"), and no artificial hold was
    involved. ``record_shell``'s write silently never ran at all.

    Asserted as the TYPE that reaches the projection rather than as an internal
    call count, because the projection is what the wire serializes.
    """

    class Session(FakeSession):
        async def refresh_attention(self) -> dict[str, Any]:
            # The real signature: a coroutine, not a value.
            return {"unseen": 3}

    class App:
        def __init__(self, session: Any, loop: Any) -> None:
            self._session = session
            self.owner_loop = loop

        def call_from_thread(self, callback: Any) -> None:
            # Runs the callback the way Textual does — on the app's own loop —
            # and hands back whatever it returned.
            asyncio.run_coroutine_threadsafe(_call(callback), self.owner_loop).result()

    app = App(Session(), asyncio.get_running_loop())
    handle = TuiSessionHandle(app)  # type: ignore[arg-type]

    state = await asyncio.wait_for(handle.refresh_attention(), timeout=5)
    assert state == {"unseen": 3}
    assert handle._projection.attention == {"unseen": 3}, (
        "the projection holds something other than the callback's RESULT — a "
        "coroutine object here fails every projection push"
    )


async def _call(callback: Any) -> Any:
    """Run ``callback`` and await it if it handed back a coroutine."""
    outcome = callback()
    if asyncio.iscoroutine(outcome):
        return await outcome
    return outcome


@pytest.mark.asyncio
async def test_tui_hop_is_bounded_by_one_budget(monkeypatch) -> None:
    """The budget covers the ENQUEUE as well as the callback's result.

    Review round 1, MAJOR 1: ``call_from_thread`` returns only once Textual has
    RUN the callback, so a ``wait_for`` around the future alone bounded the
    scheduling tail and nothing else — the await that actually parks was the
    enqueue. This pins the corrected shape: with the app loop unavailable, the
    caller is told within the budget instead of waiting out the client's own
    15 s timeout (``attach_client.ACK_TIMEOUT_S``) in silence.
    """
    from local_operator.mobile import tui_handle as mod

    monkeypatch.setattr(mod, "_APP_HOP_TIMEOUT_S", 0.3)
    entered = threading.Event()

    class Session(FakeSession):
        def refresh_attention(self) -> dict[str, Any]:
            return {"unseen": 0}

    class App:
        def __init__(self, session: Any) -> None:
            self._session = session

        def call_from_thread(self, callback: Any) -> None:
            entered.set()
            # The app is busy: the callback is queued and not run. The fake
            # waits long enough that the budget must expire first, then runs it
            # anyway — which is exactly what Textual does, and why the
            # docstring says the bound does not RECALL an enqueue.
            time.sleep(1.0)
            callback()

    handle = TuiSessionHandle(App(Session()))  # type: ignore[arg-type]
    started = time.monotonic()
    with pytest.raises(TimeoutError) as caught:
        await handle.refresh_attention()
    elapsed = time.monotonic() - started
    assert entered.is_set(), "the hop never reached the app"
    assert elapsed < 0.9, f"the caller waited {elapsed:.2f}s past its budget"
    assert "did not answer within" in str(caught.value)


@pytest.mark.asyncio
async def test_tui_stop_receipt_does_not_wait_for_the_app_loop() -> None:
    """``request_stop`` answers from identity reads, not from the app loop.

    Review round 1, UX U2: the receipt used to be built INSIDE the hop, so a
    busy app meant the ladder's rung 1 timed out at 15 s
    (``OwnerAckTimeout``) and ``lop stop`` escalated to the signal rung —
    killing a host whose hook would have ended the session politely (exit 143).
    A receipt is a statement about what is about to happen, so it must not need
    the work it announces to have started.
    """
    ran = threading.Event()

    class Session(FakeSession):
        def __init__(self) -> None:
            super().__init__()
            # Instance attributes, not class ones: ``FakeSession`` declares both
            # as read-only properties, so assigning here must go through the
            # object rather than shadowing them on the class.
            self.owner_identity = "sess"

        @property
        def session_id(self) -> str:
            return "sess"

        @property
        def conversation_name(self) -> str:
            return "the conversation"

    class App:
        def __init__(self, session: Any) -> None:
            self._session = session

        def call_from_thread(self, callback: Any) -> None:
            # The app is busy: the scheduling step is queued, not run yet.
            time.sleep(0.4)
            callback()
            ran.set()

        def run_worker(self, coro: Any, **kwargs: Any) -> None:
            # ``_stop_local_session`` is a coroutine the real app owns; the
            # receipt must not depend on it running.
            coro.close()

    handle = TuiSessionHandle(App(Session()))  # type: ignore[arg-type]
    started = time.monotonic()
    receipt = await asyncio.wait_for(handle.request_stop(), timeout=5)
    elapsed = time.monotonic() - started
    assert receipt == 'stopping "the conversation" — /resume sess reopens it'
    assert elapsed < 0.3, f"the receipt waited {elapsed:.2f}s for the app loop"
    assert await asyncio.to_thread(ran.wait, 5), "the stop hop never reached the app"


@pytest.mark.asyncio
async def test_tui_hop_names_itself_on_all_three_expiry_paths(monkeypatch) -> None:
    """Every way a hop can run out of budget reports a sentence, not silence.

    Review round 2, MINOR 1. There are three expiries: the enqueue does not come
    back inside the budget; the enqueue comes back with nothing left of the
    budget for the callback's result; and — the one that was unnamed — the
    enqueue is serviced inside the budget but the callback's AWAITABLE does not
    resolve in what remains. That third path raised a bare ``TimeoutError`` with
    an empty message (reproduced by the reviewer with a 0.3 s budget), and an
    empty message is the one shape a caller cannot act on: the sentence naming
    the busy terminal is the whole difference between "retry when it settles"
    and a mystery.
    """
    from local_operator.mobile import tui_handle as mod

    monkeypatch.setattr(mod, "_APP_HOP_TIMEOUT_S", 0.3)
    messages: list[str] = []

    class Session(FakeSession):
        pass

    class App:
        def __init__(self, session: Any, loop: Any) -> None:
            self._session = session
            self.owner_loop = loop

        def call_from_thread(self, callback: Any) -> None:
            # SERVICED AT ONCE, on the app's own loop: the enqueue costs nothing,
            # which is what isolates the third path — the callback's awaitable is
            # the only thing that consumes the budget. It must run ON that loop
            # for the same reason the real app does: ``wrapped`` schedules the
            # awaitable there.
            asyncio.run_coroutine_threadsafe(_call(callback), self.owner_loop).result()

    async def slow_refresh() -> dict[str, Any]:
        await asyncio.sleep(1.0)
        return {"unseen": 0}

    class SlowSession(Session):
        async def refresh_attention(self) -> dict[str, Any]:
            return await slow_refresh()

    app = App(SlowSession(), asyncio.get_running_loop())
    handle = TuiSessionHandle(app)  # type: ignore[arg-type]
    with pytest.raises(TimeoutError) as caught:
        await handle.refresh_attention()
    messages.append(str(caught.value))
    assert messages[0], "the third expiry path reported an empty message"
    assert "did not answer within" in messages[0]
    assert handle._late_hop_tasks, "the in-flight hop is not held by the handle"


#: The four verbs the review named for a phone, and the lane each must reach.
#: ``terminal`` is the owner's own dispatch (``OperatorApp._run_slash_command``);
#: ``dispatcher`` is the per-verb-gated typed seam it must be ROUTED to, which is
#: what keeps the phone working without re-opening ``/move``.
_PHONE_COMMANDS: tuple[tuple[str, str, str], ...] = (
    ("rename", "probe title", "dispatcher"),
    ("model", "", "dispatcher"),
    ("mcp", "login notion", "dispatcher"),
    ("stop", "", "terminal"),
)

#: And the verbs a phone must NOT reach, with the lane that answers them. The
#: delete-scoped pair is refused by the CAPABILITY gate before any lane; the rest
#: are refused by the dispatcher itself, which has no branch for them — so the
#: "no" is the app's own sentence rather than a second copy of it here.
_PHONE_REFUSED: tuple[tuple[str, str, str], ...] = (
    # V4-1: `/stop`'s ARGUMENT forms are the fleet and other sessions, so they take
    # the dispatcher's refusal like every other verb the carrier does not own. The
    # bare form is the four-verb table above; the line is what the lane is chosen by.
    ("stop", "all", "dispatcher"),
    ("stop", "62181", "dispatcher"),
    ("move", "abc --to evil", "dispatcher"),
    ("archive", "", "refused"),
    ("delete", "yes", "refused"),
    ("exit", "", "dispatcher"),
    ("update", "", "dispatcher"),
    ("btw", "", "dispatcher"),
)


@pytest.mark.asyncio
async def test_a_phone_shaped_slash_op_runs_its_commands_and_refuses_the_rest() -> None:
    """R3-1: the phone sends every typed ``/…`` line down this op, so ROUTE, don't refuse.

    The phone composer sends any ``/…`` line as ``{"op": "slash"}`` and the phone
    daemon authenticates with ``"locality": "remote"`` and no capabilities, so the
    round-2 allowlist turned away the operator's own phone: ``/model``, ``/rename``
    and ``/mcp login`` were refused on a TUI-hosted session where the same session
    on a runtime host answers them. Locality alone cannot tell a phone from a mesh
    member, so the fix is to route: a relayed command outside the terminal set goes
    to the per-verb-gated dispatcher (``run_slash_authoritative``), which has no
    branch for ``/move``, ``/exit`` or ``/update`` and therefore cannot run them.

    ``stop`` is the one verb that stays in the terminal set: it is session-scoped,
    the caller is looking at the session it stops, and the dispatcher has no branch
    for it — routing it would break the phone's ``/stop`` for nothing.
    """
    ran: list[str] = []
    routed: list[tuple[str, str, str | None, frozenset[str] | None]] = []

    class App:
        def __init__(self) -> None:
            self._session = FakeSession()
            self._loop = asyncio.get_running_loop()

        def call_from_thread(self, callback) -> None:  # noqa: ANN001
            # The REAL ``App.call_from_thread`` runs the callback on Textual's
            # loop, where a running loop exists and ``_on_app``'s ``wrapped`` can
            # create the task that carries the answer back. Calling it inline
            # instead runs it on the worker thread ``_on_app`` hops through, which
            # has no running loop — measured here as ``RuntimeError: no running
            # event loop``, the failure ``tests/unit/mobile/test_tui_ask.py``
            # documents for same-thread calls.
            self._loop.call_soon_threadsafe(callback)

        def _run_slash_command(self, line, attachments=None) -> None:  # noqa: ANN001
            ran.append(line)

        async def run_slash_authoritative(  # noqa: ANN201
            self, command, args, images=None, *, locality=None, capabilities=None, **kw
        ):
            routed.append((command, args, locality, capabilities))
            return {"kind": "notice", "text": f"owner answered /{command}", "style": "info"}

    handle = TuiSessionHandle(App())  # type: ignore[arg-type]
    # The projection push is not what this test is about, and a fake app has no
    # runtime to fold.
    handle._refresh_state = lambda: None  # type: ignore[method-assign]

    # THE PHONE'S OWN FRAME: ``locality="remote"`` and no capability claim at all.
    for command, args, lane in _PHONE_COMMANDS:
        receipt = await handle.slash(command, args, locality="remote")
        if lane == "terminal":
            assert receipt == f"ran /{command}" + (f" {args}" if args else ""), receipt
        else:
            assert receipt == f"owner answered /{command}", receipt
    assert ran == ["/stop"], ran
    assert [row[0] for row in routed] == ["rename", "model", "mcp"], routed

    # AND NOTHING DANGEROUS GETS THERE. The delete-scoped pair is refused before a
    # lane is chosen; the rest reach the dispatcher, whose refusal is its own.
    ran.clear()
    routed.clear()
    for command, args, lane in _PHONE_REFUSED:
        receipt = await handle.slash(command, args, locality="remote")
        assert not receipt.startswith("ran "), receipt
        if lane == "refused":
            assert "delete" in receipt, receipt
        else:
            assert receipt == f"owner answered /{command}", receipt
    assert ran == [], ran
    assert [row[0] for row in routed] == ["stop", "stop", "move", "exit", "update", "btw"], routed

    # A RELAYED MESH MEMBER WITH ``delete`` KEEPS ITS TERMINAL LANE (round 1's
    # negative control, and the one receipt shape routing must not change): the
    # routed dispatcher would answer with a typed notice instead of ``ran /…``.
    ran.clear()
    routed.clear()
    receipt = await handle.slash(
        "archive", "", locality="remote", capabilities=frozenset({"slash", "delete"})
    )
    assert receipt == "ran /archive", receipt
    assert ran == ["/archive"] and routed == []

    # A LOCAL PANE IS UNTOUCHED, `/stop all` INCLUDED: the fan-out is this machine's
    # own kill switch and the user at this terminal is the one entitled to it.
    assert await handle.slash("stop", "all", locality="local") == "ran /stop all"
    if ran:
        ran.clear()

    # A LOCAL PANE IS UNTOUCHED: the user's own terminal keeps every verb,
    # delete-scoped and move included, exactly as it always has.
    ran.clear()
    routed.clear()
    moved = await handle.slash("move", "abc --to evil", locality="local")
    assert moved == "ran /move abc --to evil", moved
    assert await handle.slash("archive", "", locality="local") == "ran /archive"
    assert ran == ["/move abc --to evil", "/archive"] and routed == []

    # R5-2: THE SCOPE ARGUMENT IS KEYWORD-ONLY, so a future caller cannot reach the
    # bare, session-scoped form by accident — the failure mode is a TypeError at the
    # call site rather than a silently permissive lane, which is the class rounds 2-5
    # kept finding in this feature. A default only some callers honour is how a
    # forgotten forward goes unnoticed; this one cannot be omitted positionally.
    from local_operator.network.types import may_run_slash_in_the_owners_terminal

    with pytest.raises(TypeError):
        may_run_slash_in_the_owners_terminal("stop", "remote", None, "all")  # type: ignore[misc]

    # THE DEFAULT IS STILL FAIL-CLOSED, and what it proves is narrower than
    # "a forgotten forward is refused" (round 3 R3-5, corrected in round 4 V4-5):
    # with no facts the caller is routed to the DISPATCHER rather than into the
    # terminal. Restoring the permissive default puts the line back in the terminal —
    # mutating ``locality: str | None = None`` to ``locality: str = "local"`` on
    # ``slash`` makes THESE cells report ``ran /move …``, and the cells below pin the
    # same default on the other entry point, ``slash_images``, which ``slash`` does
    # not exercise. What it does NOT prove is that a future carrier passing
    # ``"local"`` while holding no facts is safe: that caller reaches the terminal
    # lane, and only the terminal set plus the delete gate bound what it can do.
    cells: tuple[dict[str, Any], ...] = (
        {},
        {"locality": "remote"},
        {"capabilities": frozenset({"slash", "delete"})},
    )
    for facts in cells:
        ran.clear()
        routed.clear()
        receipt = await handle.slash("move", "abc --to evil", **facts)
        assert not receipt.startswith("ran "), (facts, receipt)
        assert ran == [], (facts, ran)
    # The half-forwarded case is the one a WRONG locality cannot cover: the gate is
    # not "did the caller say remote", it is "is the caller provably local".
    assert routed and all(row[2] != "local" for row in routed), routed

    # THE IMAGED CARRIER IS ITS OWN ENTRY POINT WITH ITS OWN DEFAULT. Round 4 (V4-5)
    # measured that mutating only ``slash_images``'s default left every cell above
    # green, because they all call ``slash`` — so the twin is pinned here directly.
    # In production both callers pass ``locality`` explicitly (``slash`` above, and
    # ``server.py``'s forward for the imaged half), which is why this is a
    # forgotten-forward guard rather than a live path; it is fail-closed all the same.
    image_cells: tuple[dict[str, Any], ...] = ({}, {"locality": "remote"})
    for facts in image_cells:
        ran.clear()
        routed.clear()
        receipt = await handle.slash_images("move", "abc --to evil", None, **facts)
        assert not receipt.startswith("ran "), (facts, receipt)
        assert ran == [], (facts, ran)
    ran.clear()
    routed.clear()
    local_images = await handle.slash_images("move", "abc --to evil", None, locality="local")
    assert local_images == "ran /move abc --to evil", local_images
    assert ran == ["/move abc --to evil"], ran
