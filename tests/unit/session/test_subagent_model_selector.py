"""Per-run selectors are relationships or pins, never profile/config edits."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from local_operator.harness.comms import SubagentComms
from local_operator.harness.types import Message, ModelSpec, TextContent
from local_operator.session.transcript import Transcript
from local_operator.tools.builtin import (
    TaskParams,
    build_task_tool,
    execute_hub,
    execute_task,
)
from tests.unit.session.test_pinned_subagent_model import (
    MODEL,
    make_session,
    wait_for,
    write_model_choice_config,
)

PIN = ModelSpec(provider="deepseek", model_id="deepseek-chat")
NEXT = ModelSpec(provider="ollama", model_id="next-parent")


def setup(tmp_path, monkeypatch, *, choice="model"):
    from local_operator.agents import AgentRegistry
    from local_operator.tools.agent_tool import AgentParams, write_profile

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    write_model_choice_config(tmp_path / "config", {"hi": "deepseek/deepseek-chat"}, choice=choice)
    registry = AgentRegistry(tmp_path / "agents")
    write_profile(
        registry,
        AgentParams(op="create", name="designer", instructions="Design it", effort="hi"),
        creating=True,
    )
    session = make_session(tmp_path)
    session.agent_registry = registry
    return session


def row(session, job_id):
    result = session.jobs.get(job_id)
    assert result is not None
    return result


def details(result):
    value = result.details
    assert value is not None
    return value


def text(result):
    return "".join(c.text for c in result.content if isinstance(c, TextContent))


async def launch(session, **options):
    result = await execute_task(
        "launch",
        {"label": "design", "prompt": "go", "agent": "designer", **options},
        None,
        None,
        session._build_tool_context(),
    )
    assert not result.is_error, text(result)
    job_id = details(result)["jobs"][0]["job_id"]
    await wait_for(lambda: row(session, job_id).status == "completed")
    return job_id, result


@pytest.mark.asyncio
async def test_role_pin_repro_then_explicit_inherit_launch_and_resume(tmp_path, monkeypatch):
    session = setup(tmp_path, monkeypatch, choice="operator")
    try:
        resolved = session._resolve_subagent_model("designer", None, strict=True)
        assert resolved is not None and resolved.model_id == PIN.model_id
        job_id, result = await launch(session, model="inherit")
        assert row(session, job_id).model_label == f"{MODEL.provider}/{MODEL.model_id}"
        assert "selection: explicit_inherit" in text(result)
        session.set_model(NEXT, explicit=True)
        resumed = await execute_hub(
            "resume",
            {"op": "resume", "to": [job_id], "message": "continue"},
            None,
            None,
            session._build_tool_context(),
        )
        new_id = details(resumed)["job_ids"][0]
        await wait_for(lambda: row(session, new_id).status == "completed")
        assert row(session, new_id).model_label == "ollama/next-parent"
        assert session.subagent_comms.snapshot()[0]["model_selector"] == "inherit"
        assert "selection: explicit_inherit" in text(resumed)
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_exact_pin_survives_restart_and_operator_policy_without_new_choice(
    tmp_path, monkeypatch
):
    session = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "local_operator.mobile.peer_model.validate_peer_selection", lambda p, m: PIN
    )
    try:
        job_id, _ = await launch(session, model="deepseek/deepseek-chat")
        saved = session.subagent_comms.snapshot()
        assert saved[0]["model_selector"] == "deepseek/deepseek-chat"
        session._subagent_comms = SubagentComms(session)
        session.subagent_comms.restore(saved)
        session.set_model(NEXT, explicit=True)
        write_model_choice_config(tmp_path / "config", {}, choice="operator")
        resumed = await execute_hub(
            "resume",
            {"op": "resume", "to": [job_id], "message": "continue"},
            None,
            None,
            session._build_tool_context(),
        )
        new_id = details(resumed)["job_ids"][0]
        await wait_for(lambda: row(session, new_id).status == "completed")
        assert row(session, new_id).model_label == "deepseek/deepseek-chat"
        assert row(session, new_id).owns_model is True
        # A deliberate inherit override drops this run's old pin, durably.
        overridden = await execute_hub(
            "override",
            {"op": "resume", "to": [new_id], "message": "continue", "model": "inherit"},
            None,
            None,
            session._build_tool_context(),
        )
        final_id = details(overridden)["job_ids"][0]
        await wait_for(lambda: row(session, final_id).status == "completed")
        assert row(session, final_id).owns_model is False
        assert session.subagent_comms.snapshot()[0]["model_selector"] == "inherit"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_batch_and_conflicting_fields_refuse_before_launch(tmp_path, monkeypatch):
    session = setup(tmp_path, monkeypatch)
    try:
        with pytest.raises(ValidationError, match="either model or effort"):
            TaskParams.model_validate(
                {"tasks": [{"label": "bad", "prompt": "go", "model": "inherit", "effort": "hi"}]}
            )
        result = await execute_task(
            "batch",
            {
                "tasks": [
                    {"label": "a", "prompt": "go", "model": "inherit"},
                    {"label": "b", "prompt": "go", "model": "inherit"},
                ]
            },
            None,
            None,
            session._build_tool_context(),
        )
        assert len(details(result)["jobs"]) == 2
        assert all(
            j["model_selection_source"] == "explicit_inherit" for j in details(result)["jobs"]
        )
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_nested_inherit_replays_nearest_direct_pin_after_restart(tmp_path, monkeypatch):
    """A resumed worker must not adopt the root's model past a pinned manager."""
    from local_operator.harness.types import StreamEndEvent, StreamTextDelta

    session = setup(tmp_path, monkeypatch)
    release = __import__("asyncio").Event()

    def transport(request, signal):
        async def stream():
            if any("HOLD_MANAGER" in str(m) for m in request.messages):
                await release.wait()
            yield StreamTextDelta(delta="done")
            yield StreamEndEvent(stop_reason="stop")

        return stream()

    session._stream_fn = transport
    monkeypatch.setattr(
        "local_operator.mobile.peer_model.validate_peer_selection", lambda p, m: PIN
    )
    try:
        manager_id = session._launch_subagent(
            "manager", "HOLD_MANAGER", agent="manager", model="deepseek/deepseek-chat"
        )
        await wait_for(lambda: session.subagent_comms._record(manager_id).child is not None)
        manager = session.subagent_comms._record(manager_id).child
        result = await execute_task(
            "worker",
            {"label": "worker", "prompt": "go", "model": "inherit"},
            None,
            None,
            manager._build_tool_context(),
        )
        worker_id = details(result)["jobs"][0]["job_id"]
        await wait_for(lambda: row(manager, worker_id).status == "completed")
        release.set()
        await wait_for(lambda: row(session, manager_id).status == "completed")
        saved = session.subagent_comms.snapshot()
        session._subagent_comms = SubagentComms(session)
        session.subagent_comms.restore(saved)
        session.set_model(NEXT, explicit=True)
        result = await execute_hub(
            "resume-worker",
            {"op": "resume", "to": [worker_id], "message": "continue"},
            None,
            None,
            session._build_tool_context(),
        )
        resumed_id = details(result)["job_ids"][0]
        await wait_for(lambda: row(session, resumed_id).status == "completed")
        assert row(session, resumed_id).model_label == "deepseek/deepseek-chat"
        assert row(session, resumed_id).owns_model is False
        assert "its parent's model" in text(result)
    finally:
        release.set()
        await session.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["task", "hub", "ancestor"])
async def test_exact_selector_validation_does_not_block_shared_loop(tmp_path, monkeypatch, surface):
    """The loop must run while validation is held, not merely meet a timing budget."""
    import asyncio
    import threading

    session = setup(tmp_path, monkeypatch)
    entered = threading.Event()
    release = threading.Event()
    worker_threads: list[int] = []
    loop_thread = threading.get_ident()
    monkeypatch.setattr(
        "local_operator.mobile.peer_model.validate_peer_selection", lambda p, m: PIN
    )
    job_id, _ = await launch(session, model="deepseek/deepseek-chat")
    if surface == "ancestor":
        record = session.subagent_comms._record(job_id)
        assert record is not None
        session.subagent_comms.record_launch(
            "worker",
            "worker",
            parent_job_id=job_id,
            model_selector="inherit",
            model_selection_source="explicit_inherit",
        )
        worker = session.subagent_comms._record("worker")
        assert worker is not None
        worker.session_dir = record.session_dir.parent / "worker-transcript"
        await Transcript(worker.session_dir).append_message(Message.user("go"))
        job_id = "worker"

    def held_validation(provider, model_id):
        worker_threads.append(threading.get_ident())
        entered.set()
        # Safety bound only: assertions use synchronization, not elapsed time.
        if not release.wait(5):
            raise ValueError("loop could not release metadata validation")
        return PIN

    monkeypatch.setattr("local_operator.mobile.peer_model.validate_peer_selection", held_validation)
    before = len(session.subagent_comms.snapshot())
    if surface == "task":
        operation = execute_task(
            "held",
            {"label": "held", "prompt": "go", "model": "deepseek/deepseek-chat"},
            None,
            None,
            session._build_tool_context(),
        )
    else:
        operation = execute_hub(
            "held",
            {"op": "resume", "to": [job_id], "message": "go"},
            None,
            None,
            session._build_tool_context(),
        )

    async def run_operation():
        return await operation

    pending = asyncio.create_task(run_operation())
    try:
        async with asyncio.timeout(10):
            while not entered.is_set():
                await asyncio.sleep(0)
        # This code can execute only if validation yielded the loop. No new
        # job may register before the worker produces its validated spec.
        assert not pending.done()
        assert len(session.subagent_comms.snapshot()) == before
        assert worker_threads == [worker_threads[0]] and worker_threads[0] != loop_thread
        release.set()
        result = await pending
        assert not result.is_error, text(result)
    finally:
        release.set()
        await pending
        await session.dispose()


@pytest.mark.asyncio
async def test_cancelled_exact_validation_never_registers_a_child(tmp_path, monkeypatch):
    import asyncio
    import threading

    session = setup(tmp_path, monkeypatch)
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def held(provider, model_id):
        entered.set()
        release.wait(5)
        finished.set()
        return PIN

    monkeypatch.setattr("local_operator.mobile.peer_model.validate_peer_selection", held)

    async def run():
        return await execute_task(
            "cancel",
            {"label": "cancel", "prompt": "go", "model": "deepseek/deepseek-chat"},
            None,
            None,
            session._build_tool_context(),
        )

    pending = asyncio.create_task(run())
    try:
        async with asyncio.timeout(10):
            while not entered.is_set():
                await asyncio.sleep(0)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        release.set()
        async with asyncio.timeout(10):
            while not finished.is_set():
                await asyncio.sleep(0)
        assert not session.subagent_comms.snapshot()
    finally:
        release.set()
        await session.dispose()


def test_schema_optionals_and_operator_inherit_only(tmp_path, monkeypatch):
    session = setup(tmp_path, monkeypatch, choice="operator")
    tool = build_task_tool(session._build_tool_context())
    assert tool is not None
    schema = tool.parameters
    assert "model" not in schema.get("required", [])
    assert schema["properties"]["model"]["anyOf"][0]["enum"] == ["inherit"]
    assert "model" not in TaskParams(label="x", prompt="go").model_dump(exclude_unset=True)
    with pytest.raises(ValidationError, match="model_choice=model"):
        TaskParams(label="x", prompt="go", model="deepseek/deepseek-chat")


@pytest.mark.asyncio
@pytest.mark.parametrize("selector", ["nosuchprovider/x", "deepseek/not-a-model", "deepseek/", ""])
async def test_invalid_exact_selectors_do_not_register_jobs(tmp_path, monkeypatch, selector):
    session = setup(tmp_path, monkeypatch)
    try:
        result = await execute_task(
            "bad",
            {"label": "bad", "prompt": "go", "model": selector},
            None,
            None,
            session._build_tool_context(),
        )
        assert result.is_error, text(result)
        assert not session.subagent_comms.snapshot()
    finally:
        await session.dispose()
