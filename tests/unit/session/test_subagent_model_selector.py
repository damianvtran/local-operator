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
        scope="global",
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


# ---------------------------------------------------------------------------
# the inherit sentinel against an explicit per-run model pick
# ---------------------------------------------------------------------------


def setup_inherit_pinned(tmp_path, monkeypatch, *, choice="model"):
    """A session whose ``coder`` role is the PACKAGED seed, pin and all.

    The seed is the published definition of the role, so the pin under test is
    the one a teammate who pulls the role actually gets — not a row written by
    hand for the occasion.
    """
    from local_operator.agent_profiles import install_seed
    from local_operator.agents import AgentRegistry

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    write_model_choice_config(tmp_path / "config", {"hi": "deepseek/deepseek-chat"}, choice=choice)
    registry = AgentRegistry(tmp_path / "agents")
    assert install_seed("coder", registry=registry, scope="global") is not None
    session = make_session(tmp_path)
    session.agent_registry = registry
    return session


@pytest.mark.asyncio
async def test_an_explicit_model_pick_cannot_move_a_role_that_pins_the_session_model(
    tmp_path, monkeypatch
):
    """The other model-reachable layer: the per-run ``model=`` pick.

    A role that declares "always run on the launching session's model" is not
    moved by a delegating model naming a different model either — the same
    reason the launch tier does not move it. The receipt names the role as the
    decider, so the pick is visible as unused rather than silently dropped.
    """
    session = setup_inherit_pinned(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "local_operator.mobile.peer_model.validate_peer_selection", lambda p, m: PIN
    )
    try:
        spec, source = session._resolve_subagent_selection(
            "coder", None, strict=True, model="deepseek/deepseek-chat"
        )
        assert spec is not None
        assert (spec.provider, spec.model_id) == (MODEL.provider, MODEL.model_id)
        assert source == "role_inherit"

        # And the launch says so on the receipt the delegating model reads.
        job_id, result = await launch(session, agent="coder", model="deepseek/deepseek-chat")
        assert row(session, job_id).model_label == f"{MODEL.provider}/{MODEL.model_id}"
        assert "selection: role_inherit" in text(result)
        assert "NOT applied" in text(result)
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_role_with_no_pin_still_takes_an_explicit_model_pick(tmp_path, monkeypatch):
    """The negative half: the rule is narrow, and an ordinary role is untouched.

    A role with no pin is the delegating model's to place, exactly as before.
    """
    session = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "local_operator.mobile.peer_model.validate_peer_selection", lambda p, m: PIN
    )
    try:
        spec, source = session._resolve_subagent_selection(
            "scout", None, strict=True, model="deepseek/deepseek-chat"
        )
        assert spec is not None
        assert (spec.provider, spec.model_id) == (PIN.provider, PIN.model_id)
        assert source == "explicit_model"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_malformed_pick_is_refused_even_for_a_role_that_pins_the_session_model(
    tmp_path, monkeypatch
):
    """A malformed selector is malformed whether or not the role would use it.

    The SHAPE check stays in front of the role's pin (it is the same check the
    ``task`` tool's own argument validation makes), so a role that pins the
    session's model does not become a way to launch on a typo.
    """
    session = setup_inherit_pinned(tmp_path, monkeypatch)
    try:
        result = await execute_task(
            "bad",
            {"label": "bad", "prompt": "go", "agent": "coder", "model": "deepseek/"},
            None,
            None,
            session._build_tool_context(),
        )
        assert result.is_error
        assert not session.jobs.list()
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_pick_this_machine_cannot_use_is_not_applied_rather_than_refused(
    tmp_path, monkeypatch
):
    """A pick that is well-formed but unusable here is NOT applied — and is not
    the reason the launch fails.

    The role's pin means the pick will not run, so requiring it to be usable
    on this machine would fail launches over a credential nothing was going to
    use. The pick is reported as not applied on the receipt, which is the
    honest account of what happened; what it must never be is silently
    honoured.
    """
    session = setup_inherit_pinned(tmp_path, monkeypatch)
    try:
        job_id, result = await launch(session, agent="coder", model="nosuchprovider/x")
        assert row(session, job_id).model_label == f"{MODEL.provider}/{MODEL.model_id}"
        assert "NOT applied" in text(result)
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_hub_resume_of_a_pinned_role_says_the_pick_was_not_applied(tmp_path, monkeypatch):
    """R1-2: a resume is a launch, so its receipt owes the same note.

    ``SubagentComms.resume`` re-resolves the model from current state under the
    launch rule, which means a role that pins the session's model ignores a
    ``model=`` passed to ``hub op='resume'`` exactly as it ignores one passed to
    ``task``. The task path already names the dropped pick; a resume that
    stayed silent would be the one surface where it reads as honoured — and a
    resume spends money on the model it actually chose.
    """
    session = setup_inherit_pinned(tmp_path, monkeypatch)
    try:
        job_id, _ = await launch(session, agent="coder")

        resumed = await execute_hub(
            "resume",
            {
                "op": "resume",
                "to": [job_id],
                "message": "continue",
                "model": "deepseek/deepseek-chat",
            },
            None,
            None,
            session._build_tool_context(),
        )

        new_id = details(resumed)["job_ids"][0]
        await wait_for(lambda: row(session, new_id).status == "completed")
        assert row(session, new_id).model_label == f"{MODEL.provider}/{MODEL.model_id}"
        assert "selection: role_inherit" in text(resumed)
        assert "NOT applied" in text(resumed)
        assert "model='deepseek/deepseek-chat'" in text(resumed)
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_an_inherit_pick_on_a_pinned_role_is_not_reported_as_dropped(tmp_path, monkeypatch):
    """R1-3: ``model='inherit'`` is not a pick that failed to apply.

    The wire spelling means "no specific model", and the child a sentinel role
    launches DOES inherit the session's model — the outcome the caller asked
    for. Calling that "NOT applied" invents a substitution nobody made, and it
    would teach a reading model to distrust a note that is true everywhere it
    actually fires.
    """
    session = setup_inherit_pinned(tmp_path, monkeypatch)
    try:
        job_id, result = await launch(session, agent="coder", model="inherit")
        assert row(session, job_id).model_label == f"{MODEL.provider}/{MODEL.model_id}"
        assert "selection: role_inherit" in text(result)
        assert "NOT applied" not in text(result)
    finally:
        await session.dispose()
