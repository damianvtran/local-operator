"""Eager runtime start, and the retirement that pays for it.

A viewer used to hold no runtime until the user's first keystroke, so a freshly
opened TUI painted a band with no MCP roster, no context/token reading and no
effective model — indistinguishable from a status bar that had failed to load.
The viewer now engages at mount instead.

That trades a lazy start for a process per opened terminal, and these tests pin
the two halves of the bargain:

1. The engage happens with NO input at all (``test_the_tui_engages_a_runtime_…``).
2. A viewer that leaves without using the session offers the runtime back, and
   the RUNTIME decides — refusing whenever another viewer is attached or
   anything durable exists. The refusals are the load-bearing half: a wrong
   "retire" ends a session someone is using, which is why every uncertain
   answer here must be "keep".
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

import pytest

from local_operator.session.protocol import RuntimeLocality
from local_operator.session.runtime.server import RuntimeServer, _ClientConn
from tests.unit.session.runtime.test_server import FakeHandle

#: Upper bound on an awaited event, never a budget to sleep through.
DEADLOCK_GUARD_S = 30.0


def _configure_provider(config_dir: Path) -> None:
    """Make the temp config look like a configured machine (see the
    no-provider test for why an empty one must NOT engage)."""
    from local_operator.config import ConfigManager

    ConfigManager(config_dir=config_dir).update_config(
        {"hosting": "anthropic", "model_name": "claude-opus-5"}
    )


class PristineHandle(FakeHandle):
    """A handle that can answer the retirement probe, like a real runtime.

    ``FakeHandle`` deliberately stays as it is — it stands in for a reduced or
    older runtime, and the test below uses it unmodified to pin the
    "cannot judge itself" refusal.
    """

    def __init__(self, *, pristine: bool = True) -> None:
        super().__init__()
        self.pristine = pristine
        self.stopped = False

    def is_pristine(self) -> bool:
        return self.pristine

    def request_stop(self) -> None:
        self.stopped = True


class ExplodingHandle(PristineHandle):
    """A runtime whose pristine probe raises. Uncertainty must keep it alive."""

    def is_pristine(self) -> bool:
        raise RuntimeError("state is unreadable")


def _conn(kind: str) -> _ClientConn:
    """A registered connection of the given kind, with a stand-in writer.

    The real dataclass rather than a look-alike, so the fields ``_on_request``
    reads (``kind``, ``locality``, ``watched_jobs``, ``writer`` as the registry
    key) are the production ones. Nothing is ever written to the socket: the
    test swaps ``_send_to`` for a capture.
    """
    return _ClientConn(writer=cast(Any, object()), kind=cast(Any, kind))


async def _retire(server: RuntimeServer, conn: _ClientConn) -> str:
    """Drive the op the way a viewer does and return the ack detail."""
    sent: list[dict[str, Any]] = []

    async def capture(target, frame):  # noqa: ANN001
        sent.append(frame)

    async def noop_broadcast(frame):  # noqa: ANN001
        return None

    server._send_to = capture  # type: ignore[assignment]
    server._broadcast = noop_broadcast  # type: ignore[assignment]
    await server._on_request({"op": "retire_if_pristine", "req": 1}, conn)
    assert sent, "the op never replied"
    reply = sent[-1]
    assert reply.get("op") == "ack", f"unexpected reply: {reply}"
    return str(reply.get("detail", ""))


def _register(server: RuntimeServer, conn: _ClientConn) -> None:
    server._clients[id(conn.writer)] = conn


@pytest.mark.asyncio
async def test_a_pristine_unobserved_runtime_retires() -> None:
    """The case eager start creates: opened, never used, viewer leaving."""
    handle = PristineHandle(pristine=True)
    server = RuntimeServer(handle, kind="tui")
    leaving = _conn("attach")
    _register(server, leaving)

    detail = await _retire(server, leaving)

    assert detail == "retired"
    assert handle.stopped is True, "a pristine unobserved runtime must stop"


@pytest.mark.asyncio
async def test_a_runtime_with_history_is_kept() -> None:
    """Not-pristine outranks everything: a real conversation is never dropped."""
    handle = PristineHandle(pristine=False)
    server = RuntimeServer(handle, kind="tui")
    leaving = _conn("attach")
    _register(server, leaving)

    detail = await _retire(server, leaving)

    assert detail == "kept: session has work or history"
    assert handle.stopped is False


@pytest.mark.asyncio
async def test_a_runtime_another_viewer_is_watching_is_kept() -> None:
    """The forgotten-TUI case the retirement exists to NOT break.

    A session left open for hours stays pristine forever, so emptiness alone
    would happily retire the runtime behind a second terminal that is still
    attached — and that terminal is exactly where a later instruction arrives.
    """
    handle = PristineHandle(pristine=True)
    server = RuntimeServer(handle, kind="tui")
    leaving = _conn("attach")
    observer = _conn("attach")
    _register(server, leaving)
    _register(server, observer)

    detail = await _retire(server, leaving)

    assert detail == "kept: 1 viewer(s) still attached"
    assert handle.stopped is False, "a runtime under observation must survive"


@pytest.mark.asyncio
async def test_the_leaving_viewer_does_not_count_as_its_own_observer() -> None:
    """The off-by-one that would make every retirement refuse itself.

    The leaving viewer's connection is still registered while its op is
    dispatched, so counting it would make the observer term never reach zero.
    """
    handle = PristineHandle(pristine=True)
    server = RuntimeServer(handle, kind="tui")
    leaving = _conn("attach")
    _register(server, leaving)

    assert server.attach_clients() == 1, "the leaving viewer is still registered"
    detail = await _retire(server, leaving)

    assert detail == "retired"


@pytest.mark.asyncio
async def test_a_daemon_connection_does_not_hold_a_runtime_open() -> None:
    """``daemon`` clients are not attention, exactly as the reaper reads them.

    The mobile daemon adopts EVERY session on the machine, so counting its
    connection would mean nothing is ever retired on a machine running
    ``lop mobile``.
    """
    handle = PristineHandle(pristine=True)
    server = RuntimeServer(handle, kind="tui")
    leaving = _conn("attach")
    daemon = _conn("daemon")
    _register(server, leaving)
    _register(server, daemon)

    detail = await _retire(server, leaving)

    assert detail == "retired"


@pytest.mark.asyncio
async def test_an_unreadable_pristine_probe_keeps_the_runtime() -> None:
    """Uncertainty is never a licence to stop a session."""
    handle = ExplodingHandle()
    server = RuntimeServer(handle, kind="tui")
    leaving = _conn("attach")
    _register(server, leaving)

    detail = await _retire(server, leaving)

    assert detail.startswith("kept: pristine probe failed")
    assert handle.stopped is False


@pytest.mark.asyncio
async def test_a_runtime_that_cannot_judge_itself_is_kept() -> None:
    """An older runtime, or a reduced handle, is left to the residency drain."""
    handle = FakeHandle()  # no is_pristine
    server = RuntimeServer(handle, kind="tui")
    leaving = _conn("attach")
    _register(server, leaving)

    detail = await _retire(server, leaving)

    assert detail == "kept: this runtime cannot judge itself pristine"


@pytest.mark.asyncio
async def test_is_pristine_reads_the_attachment_sidecar(tmp_path: Path, monkeypatch) -> None:
    """#624 review round 2, R7: a routed `/team <name>` with no request.

    ``Session.attach_team`` journals ``attachment.json`` and prints "team X is
    ready" — and wrote no transcript row, so ``is_pristine`` said True and a
    quit retired the runtime the receipt had just vouched for. The sidecar is
    what a resume re-stamps the team from; it is as durable as a row.
    """
    from local_operator.resume import ATTACHMENT_SIDECAR_NAME
    from local_operator.session.runtime.serving import ServingSessionHandle
    from local_operator.session.transcript import Transcript

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    directory = tmp_path / "sessions" / "s1"
    directory.mkdir(parents=True)
    transcript = Transcript(directory)

    class _Session:
        # Runtime role (SessionProtocol). This fake stands in for an OWNER:
        # it carries no attached runtime, which is what the absent legacy
        # `is_remote` meant.
        owns_runtime = True
        outcome_is_synchronous = True
        runtime_locality: RuntimeLocality = "this-process"

        def __init__(self) -> None:
            self._transcript = transcript
            self.wake_scheduler = None
            self.session_id = "s1"

        def history(self):  # noqa: ANN202
            return []

        # ``SessionProtocol.credential_op``: the REAL verb table against a
        # memory-only store (the ``test_app_pilot.FakeSession`` pattern), so a
        # credential probe of this double answers the way the owner session it
        # stands in for does instead of silently refusing — a double that
        # swallows the verb is how #891 passed review on an unreachable path.
        @property
        def variables(self) -> Any:
            store = getattr(self, "_variables", None)
            if store is None:
                from local_operator.variables import VariableStore

                store = self._variables = VariableStore(cwd="/tmp", env={})
            return store

        async def credential_op(
            self, action: str, key: str = "", value: str = ""
        ) -> dict[str, Any]:
            from local_operator.session.credential_ops import run_credential_verb

            return await run_credential_verb(
                self.variables, getattr(self, "journal_credential_change", None), action, key, value
            )

    handle = object.__new__(ServingSessionHandle)
    handle._session = _Session()  # type: ignore[attr-defined]
    object.__setattr__(handle, "is_busy", lambda: False)

    assert handle.is_pristine() is True, "nothing has happened yet"
    (directory / ATTACHMENT_SIDECAR_NAME).write_text('{"team": "lopdev"}', encoding="utf-8")
    assert handle.is_pristine() is False, "an attached team is durable state the user asked for"


@pytest.mark.asyncio
async def test_is_pristine_reads_durable_rows_not_the_model_window(
    tmp_path: Path,
) -> None:
    """A compacted conversation is idle, and emphatically not disposable.

    ``is_busy`` answers "may this exit later", which a finished conversation
    satisfies. ``is_pristine`` must answer "did this session ever exist", so it
    reads the durable transcript rather than the model-facing history that
    compaction shrinks.
    """
    from local_operator.harness.types import Message
    from local_operator.session.runtime.serving import ServingSessionHandle
    from local_operator.session.transcript import Transcript

    directory = tmp_path / "sessions" / "s1"
    directory.mkdir(parents=True)
    transcript = Transcript(directory)
    await transcript.append_message(Message.user("we talked about something"))

    class _Session:
        # Runtime role (SessionProtocol). This fake stands in for an OWNER:
        # it carries no attached runtime, which is what the absent legacy
        # `is_remote` meant.
        owns_runtime = True
        outcome_is_synchronous = True
        runtime_locality: RuntimeLocality = "this-process"

        def __init__(self) -> None:
            self._transcript = transcript
            self.wake_scheduler = None

        def history(self):  # noqa: ANN202 — compaction emptied the window
            return []

        # ``SessionProtocol.credential_op``: the REAL verb table against a
        # memory-only store (the ``test_app_pilot.FakeSession`` pattern), so a
        # credential probe of this double answers the way the owner session it
        # stands in for does instead of silently refusing — a double that
        # swallows the verb is how #891 passed review on an unreachable path.
        @property
        def variables(self) -> Any:
            store = getattr(self, "_variables", None)
            if store is None:
                from local_operator.variables import VariableStore

                store = self._variables = VariableStore(cwd="/tmp", env={})
            return store

        async def credential_op(
            self, action: str, key: str = "", value: str = ""
        ) -> dict[str, Any]:
            from local_operator.session.credential_ops import run_credential_verb

            return await run_credential_verb(
                self.variables, getattr(self, "journal_credential_change", None), action, key, value
            )

    handle = object.__new__(ServingSessionHandle)
    handle._session = _Session()  # type: ignore[attr-defined]

    # is_busy is stubbed: the point here is the DURABLE probe, and the real
    # one reads a dozen session internals this reduced double does not have.
    object.__setattr__(handle, "is_busy", lambda: False)

    assert handle.is_pristine() is False, "a transcript row means the session is real"


@pytest.mark.asyncio
async def test_the_tui_engages_a_runtime_without_any_input(tmp_path: Path, monkeypatch) -> None:
    """The reported bug, end to end: no keystroke, and a runtime starts anyway.

    Driven through the real ``OperatorApp`` boot path rather than by calling
    the helper, because the regression being pinned is one of ORDERING — the
    engage has to be reached from boot, after adoption, with nothing typed.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    directory = tmp_path / "sessions" / "s1"
    directory.mkdir(parents=True)
    (directory / "transcript.jsonl").write_text("", encoding="utf-8")

    _configure_provider(tmp_path)

    from local_operator.session.attached import AttachedSession
    from local_operator.tui.app import OperatorApp

    engaged = asyncio.Event()

    async def fake_engage(
        session_id, cwd, work, *, config_dir, deadline_s=30.0, preempt=None, preempt_budget_s=0.0
    ):  # noqa: ANN001
        engaged.set()
        raise ConnectionError("no runtime in this test")

    monkeypatch.setattr("local_operator.session.runtime.launch.engage_runtime", fake_engage)

    async def _never():
        raise AssertionError("takeover was not expected")

    viewer = await AttachedSession.cold(
        "s1", config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never
    )

    async def factory():
        return viewer

    app = OperatorApp(factory)
    try:
        async with app.run_test(size=(100, 30)) as pilot:
            await asyncio.wait_for(engaged.wait(), timeout=DEADLOCK_GUARD_S)
            await pilot.pause()
    finally:
        await viewer.dispose()


@pytest.mark.asyncio
async def test_leaving_a_session_offers_its_runtime_back(tmp_path: Path) -> None:
    """``AttachedSession.retire_if_unused`` asks, and reports what it was told."""
    from local_operator.session.attached import AttachedSession

    viewer = object.__new__(AttachedSession)
    viewer._snapshot_clients = {}

    class _Client:
        connected = True

        def __init__(self) -> None:
            self.asked = 0

        async def retire_if_pristine(self) -> str:
            self.asked += 1
            return "retired"

    client = _Client()
    viewer._client = client  # type: ignore[attr-defined]

    assert await viewer.retire_if_unused() == "retired"
    assert client.asked == 1


@pytest.mark.asyncio
async def test_a_cold_viewer_has_no_runtime_to_offer_back() -> None:
    """Quitting a viewer that never engaged must not raise on the way out."""
    from local_operator.session.attached import AttachedSession

    viewer = object.__new__(AttachedSession)
    viewer._snapshot_clients = {}
    viewer._client = None  # type: ignore[attr-defined]

    assert await viewer.retire_if_unused() == "no runtime attached"


@pytest.mark.asyncio
async def test_a_failed_offer_is_swallowed_on_the_way_out() -> None:
    """Teardown must not fail over a courtesy the residency drain also covers."""
    from local_operator.session.attached import AttachedSession

    viewer = object.__new__(AttachedSession)
    viewer._snapshot_clients = {}

    class _Client:
        connected = True

        async def retire_if_pristine(self) -> str:
            raise ConnectionError("socket already gone")

    viewer._client = _Client()  # type: ignore[attr-defined]

    assert (await viewer.retire_if_unused()).startswith("request failed:")


@pytest.mark.asyncio
async def test_an_engage_that_lands_after_dispose_does_not_bind(
    tmp_path: Path, monkeypatch
) -> None:
    """Review round 1, MAJOR-1: the swap-during-engage race.

    `/resume` typed in the first second of a fresh `lop` disposes the viewer
    while its mount engage is still spawning. If the engage then binds, a
    live `attach` socket hangs off a dead facade: nobody closes it, so the
    old runtime stays resident for the life of the process and never gets a
    retire offer. The facade must refuse to bind once disposed.
    """
    from local_operator.session.attached import AttachedSession

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)

    parked = asyncio.Event()
    release = asyncio.Event()
    looked_for_record = False

    async def fake_engage(
        session_id, cwd, work, *, config_dir, deadline_s=30.0, preempt=None, preempt_budget_s=0.0
    ):  # noqa: ANN001
        parked.set()
        await release.wait()

    def fake_find(config_dir, session_id, **_probe):  # noqa: ANN001
        nonlocal looked_for_record
        looked_for_record = True
        return None, None

    monkeypatch.setattr("local_operator.session.runtime.launch.engage_runtime", fake_engage)
    monkeypatch.setattr("local_operator.mobile.attach_client.find_runtime_record", fake_find)

    async def _never():
        raise AssertionError

    viewer = await AttachedSession.cold(
        "s1", config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never
    )
    engage = asyncio.ensure_future(viewer._ensure_bound())
    await asyncio.wait_for(parked.wait(), timeout=DEADLOCK_GUARD_S)
    await viewer.dispose()
    release.set()
    await asyncio.wait_for(engage, timeout=DEADLOCK_GUARD_S)

    assert viewer.is_cold, "a disposed viewer must not come out of the engage bound"
    assert viewer._client is None
    assert not looked_for_record, "a disposed viewer must stop before dialling anything"


@pytest.mark.asyncio
async def test_a_session_swap_cancels_the_engage_in_flight(tmp_path: Path, monkeypatch) -> None:
    """The app-side half of MAJOR-1: `/resume` cancels the mount engage.

    Even with the facade refusing to bind, a worker left running after the
    swap would hold the band's "starting…" and spend a spawn on a session
    the user has already left.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    (tmp_path / "sessions" / "s1" / "transcript.jsonl").write_text("", encoding="utf-8")

    _configure_provider(tmp_path)

    from local_operator.session.attached import AttachedSession
    from local_operator.tui.app import OperatorApp

    parked = asyncio.Event()
    cancelled = asyncio.Event()

    async def fake_engage(
        session_id, cwd, work, *, config_dir, deadline_s=30.0, preempt=None, preempt_budget_s=0.0
    ):  # noqa: ANN001
        parked.set()
        try:
            await asyncio.Event().wait()  # park until cancelled
        except asyncio.CancelledError:
            cancelled.set()
            raise

    monkeypatch.setattr("local_operator.session.runtime.launch.engage_runtime", fake_engage)

    async def _never():
        raise AssertionError

    async def make(session_id: str):
        return await AttachedSession.cold(
            session_id, config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never
        )

    first = await make("s1")

    async def factory():
        return first

    app = OperatorApp(factory)
    async with app.run_test(size=(100, 30)) as pilot:
        await asyncio.wait_for(parked.wait(), timeout=DEADLOCK_GUARD_S)
        await pilot.pause()
        assert app._starting_runtime is True, "the band should say starting… mid-engage"

        app._session_factory = lambda: make("s2")
        await app._reload_session()
        await asyncio.wait_for(cancelled.wait(), timeout=DEADLOCK_GUARD_S)
        await pilot.pause()

        assert first._disposed
        assert app._session is not first
        assert getattr(app._session, "session_id", "") == "s2"
        # And the NEW session got its own engage on the same terms (the stub
        # parks it, so the band is starting… again for s2, not stuck from s1).
        assert app._warm_engage_started is True
    await first.dispose()


@pytest.mark.asyncio
async def test_no_provider_configured_skips_the_mount_engage(tmp_path: Path, monkeypatch) -> None:
    """Review round 1, MAJOR-2: the first-run screen must not spawn anything.

    With no `hosting`/`model_name` a runtime exits rc=2 on construction; the
    engage loop respawned three of those and then sat on "starting…" for its
    30 s deadline on the very screen that tells the user to `/login`.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions").mkdir(parents=True)

    from local_operator.session.attached import AttachedSession
    from local_operator.tui.app import OperatorApp

    engaged = False

    async def fake_engage(
        session_id, cwd, work, *, config_dir, deadline_s=30.0, preempt=None, preempt_budget_s=0.0
    ):  # noqa: ANN001
        nonlocal engaged
        engaged = True
        raise AssertionError("must not be reached")

    monkeypatch.setattr("local_operator.session.runtime.launch.engage_runtime", fake_engage)

    async def _never():
        raise AssertionError

    # An EMPTY config dir: the cold state synthesises an empty model spec.
    viewer = await AttachedSession.cold(
        "s1", config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never
    )
    assert viewer.frontend_state.effective_model is not None
    assert viewer.frontend_state.effective_model.provider == ""

    async def factory():
        return viewer

    app = OperatorApp(factory)
    try:
        async with app.run_test(size=(100, 30)) as pilot:
            for _ in range(40):
                await pilot.pause()
            assert app._session is viewer
            assert engaged is False, "an unconfigured viewer must not spawn a runtime"
            assert app._warm_engage_started is False
            assert app._starting_runtime is False, "no spinner on the onboarding screen"
            # And the honesty rule does not fire here either: the skip happens
            # BEFORE the worker, so there is no failed engage to report. An
            # unconfigured first run must not open with a warning about a
            # runtime it was never going to start.
            assert _failure_notices(app) == [], "a deliberate skip must stay silent"
    finally:
        await viewer.dispose()


@pytest.mark.asyncio
async def test_is_pristine_reads_the_wake_index_not_only_the_live_scheduler(
    tmp_path: Path, monkeypatch
) -> None:
    """Review round 1, MINOR-4: a wake row on disk alone makes a session real."""
    from local_operator.session.runtime.serving import ServingSessionHandle
    from local_operator.session.transcript import Transcript
    from local_operator.wakes.store import write_entry

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    directory = tmp_path / "sessions" / "s1"
    directory.mkdir(parents=True)
    write_entry(
        tmp_path,
        "s1",
        cwd=str(tmp_path),
        schedules=[
            {"id": "w1", "message": "check", "next_due_at": 4_102_444_800_000, "created_at": 1}
        ],
    )

    class _Session:
        # Runtime role (SessionProtocol). This fake stands in for an OWNER:
        # it carries no attached runtime, which is what the absent legacy
        # `is_remote` meant.
        owns_runtime = True
        outcome_is_synchronous = True
        runtime_locality: RuntimeLocality = "this-process"

        session_id = "s1"

        def __init__(self) -> None:
            self._transcript = Transcript(directory, defer_materialise=True)
            self.wake_scheduler = None  # disposed / absent: reports no wakes

        def history(self):  # noqa: ANN202
            return []

        # ``SessionProtocol.credential_op``: the REAL verb table against a
        # memory-only store (the ``test_app_pilot.FakeSession`` pattern), so a
        # credential probe of this double answers the way the owner session it
        # stands in for does instead of silently refusing — a double that
        # swallows the verb is how #891 passed review on an unreachable path.
        @property
        def variables(self) -> Any:
            store = getattr(self, "_variables", None)
            if store is None:
                from local_operator.variables import VariableStore

                store = self._variables = VariableStore(cwd="/tmp", env={})
            return store

        async def credential_op(
            self, action: str, key: str = "", value: str = ""
        ) -> dict[str, Any]:
            from local_operator.session.credential_ops import run_credential_verb

            return await run_credential_verb(
                self.variables, getattr(self, "journal_credential_change", None), action, key, value
            )

    handle = object.__new__(ServingSessionHandle)
    handle._session = _Session()  # type: ignore[attr-defined]
    object.__setattr__(handle, "is_busy", lambda: False)

    assert handle.next_wake_due_at() is None, "the live scheduler sees nothing"
    assert handle.is_pristine() is False, "the index row must still count"


@pytest.mark.asyncio
async def test_a_provider_without_a_model_name_still_engages(tmp_path: Path, monkeypatch) -> None:
    """Review round 2, MAJOR-1: an empty `model_name` is not "unconfigured".

    The runtime's resolver falls back to the provider's default model, so a
    config naming only `hosting` boots fine; the gate must agree with the
    resolver rather than regress that config to never engaging.
    """
    from local_operator.config import ConfigManager
    from local_operator.session.attached import AttachedSession
    from local_operator.tui.app import OperatorApp

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    (tmp_path / "sessions" / "s1" / "transcript.jsonl").write_text("", encoding="utf-8")
    ConfigManager(config_dir=tmp_path).update_config({"hosting": "anthropic", "model_name": ""})

    engaged = asyncio.Event()

    async def fake_engage(
        session_id, cwd, work, *, config_dir, deadline_s=30.0, preempt=None, preempt_budget_s=0.0
    ):  # noqa: ANN001
        engaged.set()
        raise ConnectionError("no runtime in this test")

    monkeypatch.setattr("local_operator.session.runtime.launch.engage_runtime", fake_engage)

    async def _never():
        raise AssertionError

    viewer = await AttachedSession.cold(
        "s1", config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never
    )
    assert viewer.frontend_state.effective_model is not None
    from local_operator.model.defaults import default_model_for

    # The cold viewer now samples the same concrete default as its owner;
    # first engagement must not defer this choice to a later config snapshot.
    assert viewer.frontend_state.effective_model.model_id == default_model_for("anthropic")

    async def factory():
        return viewer

    app = OperatorApp(factory)
    try:
        async with app.run_test(size=(100, 30)) as pilot:
            await asyncio.wait_for(engaged.wait(), timeout=DEADLOCK_GUARD_S)
            await pilot.pause()
    finally:
        await viewer.dispose()


@pytest.mark.asyncio
async def test_a_viewer_attaching_during_the_announcement_keeps_the_runtime() -> None:
    """Review round 2, MINOR-2: the observer count is re-asked after the broadcast."""
    handle = PristineHandle(pristine=True)
    server = RuntimeServer(handle, kind="tui")
    leaving = _conn("attach")
    _register(server, leaving)
    late = _conn("attach")

    sent: list[dict[str, Any]] = []

    async def capture(target, frame):  # noqa: ANN001
        sent.append(frame)

    async def attach_during_broadcast(frame):  # noqa: ANN001
        _register(server, late)  # a second terminal opens the session right now

    server._send_to = capture  # type: ignore[assignment]
    server._broadcast = attach_during_broadcast  # type: ignore[assignment]
    await server._on_request({"op": "retire_if_pristine", "req": 1}, leaving)

    assert sent[-1]["detail"] == "kept: 1 viewer(s) attached while stopping was announced"
    assert handle.stopped is False


# --- the failure the band cannot explain (mount engage honesty) ----------------
#
# The mount/draft engage is silent on failure BY DESIGN — a warm-up nobody asked
# for must not print an error at a user who has not sent anything. That design
# assumed the failure would be quick. When it is not, the band is the whole of
# what the user was told, and clearing it without a word renders "a runtime is
# coming up" identically to "the runtime never came up". These tests pin the
# three gates that decide which case is which, and the per-binding rule that
# keeps a machine which cannot start a runtime from growing a notice per
# keystroke.


async def _pump_until(pilot, predicate, timeout: float = DEADLOCK_GUARD_S) -> bool:
    """Pump the app's loop until ``predicate`` holds, or the guard expires.

    A predicate poll rather than a sleep of a fixed length: the engage runs in
    a worker, and the number of loop turns before its ``except`` arm runs is not
    something a test should assume.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        await pilot.pause()
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return predicate()


def _failure_notices(app) -> list[str]:  # noqa: ANN001 — a Textual app, untyped here
    """Every transcript notice that names the runtime-start failure."""
    from local_operator.tui.widgets.transcript import NoticeBlock

    return [
        str(block._text)
        for block in app.query(NoticeBlock)
        if "no runtime came up for this session" in str(block._text)
    ]


async def _app_with_engage(tmp_path: Path, monkeypatch, engage) -> Any:  # noqa: ANN001
    """A real app over a cold viewer whose engage is ``engage``.

    Shared by the tests below because they differ only in what the engage does
    and how long it takes — which is exactly the axis the patience gate reads.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True, exist_ok=True)
    _configure_provider(tmp_path)

    from local_operator.session.attached import AttachedSession
    from local_operator.tui.app import OperatorApp

    monkeypatch.setattr("local_operator.session.runtime.launch.engage_runtime", engage)

    async def _never():
        raise AssertionError("takeover was not expected")

    viewer = await AttachedSession.cold(
        "s1", config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never
    )

    async def factory():
        return viewer

    return OperatorApp(factory), viewer


@pytest.mark.asyncio
async def test_a_watched_mount_failure_is_reported_once_with_the_next_step(
    tmp_path: Path, monkeypatch
) -> None:
    """Past the patience threshold the transcript owes the user a sentence.

    The copy half is asserted too, and deliberately: this notice exists because
    the band clearing silently is unreadable, so a notice that does not name
    what to do next would leave the user exactly where they started — staring at
    a splash with no runtime. The measured next step is a prompt (a later bind
    lands in ~0.7 s once the child has finished constructing).
    """
    # Patience is POLICY, so the test sets its own value and drives the engage
    # past it. Pinning the production number here would test the constant rather
    # than the comparison.
    monkeypatch.setattr("local_operator.tui.app.START_ENGAGE_PATIENCE_S", 0.05)

    async def slow_failure(
        session_id, cwd, work, *, config_dir, deadline_s=30.0, preempt=None, preempt_budget_s=0.0
    ):  # noqa: ANN001
        await asyncio.sleep(0.2)  # over the patched patience above
        raise ConnectionError("the runtime is reconnecting")

    app, viewer = await _app_with_engage(tmp_path, monkeypatch, slow_failure)
    try:
        async with app.run_test(size=(100, 30)) as pilot:
            assert await _pump_until(
                pilot, lambda: bool(_failure_notices(app))
            ), "a mount engage that failed after the patience threshold said nothing"
            (notice,) = _failure_notices(app)
            assert "send a message" in notice, (
                "the notice must name the next step; it was the only account of "
                f"what happened: {notice!r}"
            )
            assert app._start_engage_notice_shown is True
    finally:
        await viewer.dispose()


@pytest.mark.asyncio
async def test_a_quick_mount_failure_stays_silent(tmp_path: Path, monkeypatch) -> None:
    """The existing contract, kept: a blip nobody watched reports nothing.

    A refused socket fails in milliseconds, the latch is cleared, and the
    message the user sends next engages again and owns the report. Announcing
    this case would put a warning on screen for a start the user never asked
    for and that cost them nothing.
    """
    monkeypatch.setattr("local_operator.tui.app.START_ENGAGE_PATIENCE_S", 5.0)

    async def fast_failure(
        session_id, cwd, work, *, config_dir, deadline_s=30.0, preempt=None, preempt_budget_s=0.0
    ):  # noqa: ANN001
        raise ConnectionError("the runtime is reconnecting")

    app, viewer = await _app_with_engage(tmp_path, monkeypatch, fast_failure)
    try:
        async with app.run_test(size=(100, 30)) as pilot:
            # Wait for the failure to have definitely run: the latch is cleared
            # by the same `except` arm that would have reported it.
            assert await _pump_until(pilot, lambda: app._warm_engage_started is False)
            for _ in range(20):
                await pilot.pause()
            assert _failure_notices(app) == [], "a sub-patience failure must stay quiet"
    finally:
        await viewer.dispose()


@pytest.mark.asyncio
async def test_a_vetted_configuration_failure_is_left_to_the_prompt(
    tmp_path: Path, monkeypatch
) -> None:
    """``ActionableConnectionError`` carries a sentence the PROMPT relays.

    The type is the permission slip for echoing the message verbatim, and the
    prompt already does that with its own reporting path. Predicting it here
    would either duplicate the sentence or paraphrase a vetted one, which is how
    "it is running in the background" got shipped over three cases where it was
    false.
    """
    from local_operator.session.runtime.launch import ActionableConnectionError

    monkeypatch.setattr("local_operator.tui.app.START_ENGAGE_PATIENCE_S", 0.05)

    async def actionable_failure(
        session_id, cwd, work, *, config_dir, deadline_s=30.0, preempt=None, preempt_budget_s=0.0
    ):  # noqa: ANN001
        await asyncio.sleep(0.2)
        raise ActionableConnectionError("no API key is stored for this provider")

    app, viewer = await _app_with_engage(tmp_path, monkeypatch, actionable_failure)
    try:
        async with app.run_test(size=(100, 30)) as pilot:
            assert await _pump_until(pilot, lambda: app._warm_engage_started is False)
            for _ in range(20):
                await pilot.pause()
            assert _failure_notices(app) == [], "a vetted configuration sentence is the prompt's"
    finally:
        await viewer.dispose()


@pytest.mark.asyncio
async def test_the_notice_is_per_binding_not_per_keystroke(tmp_path: Path, monkeypatch) -> None:
    """One notice per binding, however many times the engage is re-armed.

    A failed engage clears the latch by design, so the first keystroke retries
    (``_warm_runtime_for_draft``). Without the per-binding record a machine that
    cannot start a runtime would answer every letter typed with another copy of
    the same sentence — the retry loop the operator described as "stuck
    forever", now with a transcript full of duplicates.
    """
    monkeypatch.setattr("local_operator.tui.app.START_ENGAGE_PATIENCE_S", 0.05)
    attempts: list[int] = []

    async def slow_failure(
        session_id, cwd, work, *, config_dir, deadline_s=30.0, preempt=None, preempt_budget_s=0.0
    ):  # noqa: ANN001
        attempts.append(1)
        await asyncio.sleep(0.2)
        raise ConnectionError("the runtime is reconnecting")

    app, viewer = await _app_with_engage(tmp_path, monkeypatch, slow_failure)
    try:
        async with app.run_test(size=(100, 30)) as pilot:
            assert await _pump_until(pilot, lambda: len(attempts) == 1)
            assert await _pump_until(pilot, lambda: bool(_failure_notices(app)))
            assert len(_failure_notices(app)) == 1

            # The keystroke path, for real: the same trigger the user has.
            from local_operator.tui.widgets.editor import Editor

            editor = app.query_one(Editor)
            editor.focus()
            await pilot.press("h")
            assert await _pump_until(
                pilot, lambda: len(attempts) == 2
            ), "the keystroke must still retry the engage"
            for _ in range(20):
                await pilot.pause()
            assert (
                len(_failure_notices(app)) == 1
            ), "a second failure on the SAME binding repeated the notice"
    finally:
        await viewer.dispose()


@pytest.mark.asyncio
async def test_a_new_binding_reports_again(tmp_path: Path, monkeypatch) -> None:
    """Per BINDING, not per app: `/new` and `/resume` are cold again and owe it.

    The flag is reset on the one edge that changes the binding
    (``_adopt_session``), which is what keeps the suppression from outliving the
    session whose failure it described.
    """
    monkeypatch.setattr("local_operator.tui.app.START_ENGAGE_PATIENCE_S", 0.05)
    attempts: list[str] = []

    async def slow_failure(
        session_id, cwd, work, *, config_dir, deadline_s=30.0, preempt=None, preempt_budget_s=0.0
    ):  # noqa: ANN001
        from local_operator.session.attached import RuntimeUnresponsiveError

        attempts.append(str(session_id))
        await asyncio.sleep(0.2)
        if str(session_id) == "s2":
            # A DIFFERENT failure class on the second binding, so the assertion
            # below can tell the new binding's notice from the old one's rather
            # than counting rows: the two ceilings get two sentences, and a
            # stale suppression flag would leave only the first on screen.
            raise RuntimeUnresponsiveError("the runtime is not responding")
        raise ConnectionError("the runtime is reconnecting")

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True, exist_ok=True)
    (tmp_path / "sessions" / "s2").mkdir(parents=True, exist_ok=True)
    _configure_provider(tmp_path)

    from local_operator.session.attached import AttachedSession
    from local_operator.tui.app import OperatorApp

    monkeypatch.setattr("local_operator.session.runtime.launch.engage_runtime", slow_failure)

    async def _never():
        raise AssertionError("takeover was not expected")

    async def make(session_id: str):
        return await AttachedSession.cold(
            session_id, config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never
        )

    first = await make("s1")

    app = OperatorApp(lambda: _make_coro(first))
    second = None
    try:
        async with app.run_test(size=(100, 30)) as pilot:
            assert await _pump_until(
                pilot,
                lambda: any(
                    "no runtime came up for this session" in text for text in _notice_texts(app)
                ),
            )

            second = await make("s2")
            app._session_factory = lambda: _make_coro(second)
            await app._reload_session()
            assert await _pump_until(
                pilot,
                lambda: any("did not answer in " in text for text in _notice_texts(app)),
            ), (
                "the swapped-in binding's own failure was suppressed by the old " "binding's notice"
            )
    finally:
        await first.dispose()
        if second is not None:
            await second.dispose()


async def _make_coro(viewer: Any) -> Any:
    """An awaited-once factory that returns an already-built viewer."""
    return viewer


def _notice_texts(app) -> list[str]:  # noqa: ANN001 — a Textual app, untyped here
    """Every transcript notice, whatever it says."""
    from local_operator.tui.widgets.transcript import NoticeBlock

    return [str(block._text) for block in app.query(NoticeBlock)]


@pytest.mark.asyncio
async def test_a_prompt_in_flight_owns_the_failure_not_the_band(
    tmp_path: Path, monkeypatch
) -> None:
    """A preempted background engage is a bound, not a failure, and says nothing.

    The mount engage yields its place the moment a prompt arrives — that is what
    ``_BACKGROUND_YIELD_BUDGET_S`` is for — so a user who sends a message into a
    slow start makes this engage give up by design. Reporting that gave the
    measured result: submit at t+20.1 s, band cleared 1.5 s later with "no
    runtime came up for this session in 17s — send a message to start one", and
    the session then bound on the prompt's own engage at t+47.0 s and ran the
    turn. The prompt's bind reports what it finds; the band must not answer the
    message the user just sent with an instruction to send one.
    """
    monkeypatch.setattr("local_operator.tui.app.START_ENGAGE_PATIENCE_S", 0.05)

    async def preempted_failure(
        session_id, cwd, work, *, config_dir, deadline_s=30.0, preempt=None, preempt_budget_s=0.0
    ):  # noqa: ANN001
        # The state `_bind_lock_for` publishes BEFORE a foreground caller waits on
        # the bind lock: it is what makes the background holder cut its own
        # deadline, and it is STILL SET when that holder raises — the foreground
        # caller only clears it in its own `finally`, one task later. Deliberately
        # left set here for the same reason; the counter is per-facade state on a
        # viewer this test disposes.
        app._session._foreground_waiting += 1
        await asyncio.sleep(0.2)
        raise ConnectionError("the runtime is reconnecting")

    app, viewer = await _app_with_engage(tmp_path, monkeypatch, preempted_failure)
    try:
        async with app.run_test(size=(100, 30)) as pilot:
            assert await _pump_until(pilot, lambda: app._warm_engage_started is False)
            for _ in range(20):
                await pilot.pause()
            assert (
                _failure_notices(app) == []
            ), "a prompt in flight owns the outcome; the band must stay quiet"
    finally:
        await viewer.dispose()
