"""A resumed subagent runs on the model the LAUNCH rule picks, not its old one.

The incident: an operator moved a session to a cheaper model, then resumed a
paused child with ``hub op='resume'``. The child ran 26 calls on the model it
was born on, because ``Session._restore_selected_model`` replayed the child's
own ``selected_model`` journal row over the spec the resume had just resolved.

These drive real ``Session`` objects end to end: a real launch, a real settle,
a real ``hub`` resume, and the model named on the request the resumed child
actually sent. A top-level resume is pinned beside them, because only the
child source was meant to change.
"""

from __future__ import annotations

import asyncio

import pytest
import yaml

from local_operator.harness.types import (
    AbortSignal,
    ChatRequest,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    TextContent,
)
from local_operator.session.model_selection import read_model_selection
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.tools.builtin import execute_hub

BIRTH = ModelSpec(
    provider="openrouter",
    model_id="anthropic/claude-opus-5-5",
    context_window=100_000,
    max_output_tokens=4_096,
)
SWITCHED = BIRTH.model_copy(update={"model_id": "deepseek/deepseek-flash"})

RESUME_PROMPT = "carry on from where you stopped"


class RecordingStream:
    """Answers every call with one text delta and records the request."""

    def __init__(self) -> None:
        self.requests: list[ChatRequest] = []

    def __call__(self, request: ChatRequest, signal: AbortSignal | None):
        self.requests.append(request)

        async def gen():
            yield StreamTextDelta(delta="done")
            yield StreamEndEvent(stop_reason="stop")

        return gen()

    def selectors_for(self, needle: str) -> list[str]:
        """The model of every request whose conversation contains ``needle``.

        The parent and its child share this stream, so the resumed child's
        calls are picked out by the instruction only the resume delivered.
        """
        found = []
        for request in self.requests:
            text = " ".join(
                block.text
                for message in request.messages
                for block in (getattr(message, "content", None) or [])
                if isinstance(block, TextContent)
            )
            if needle in text:
                found.append(f"{request.model.provider}/{request.model.model_id}")
        return found


async def wait_for(predicate, timeout: float = 10.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("timed out waiting for condition")
        await asyncio.sleep(0.005)


def _parent(tmp_path, stream: RecordingStream) -> Session:
    return Session(
        model=BIRTH,
        stream_fn=stream,
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["stable"],
    )


def _write_tiers(config_dir, **tiers: str) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.yml").write_text(
        yaml.safe_dump({"values": {"subagents": {"models": tiers}}})
    )


def _completed(session: Session, job_id: str) -> bool:
    row = session.jobs.get(job_id)
    return row is not None and row.status == "completed"


async def _hub_resume(parent: Session, job_id: str) -> str:
    result = await execute_hub(
        "call-resume",
        {"op": "resume", "to": [job_id], "message": RESUME_PROMPT},
        None,
        None,
        parent._build_tool_context(),
    )
    assert not result.is_error, result
    block = result.content[0]
    assert isinstance(block, TextContent)
    return block.text


def _resumed_id(parent: Session, text: str) -> str:
    marker = "resumed as job "
    assert marker in text, text
    new_id = text.split(marker, 1)[1].split()[0]
    assert parent.jobs.get(new_id) is not None
    return new_id


@pytest.mark.asyncio
async def test_an_inheriting_child_resumes_on_the_parents_current_model(tmp_path, monkeypatch):
    """The incident's exact shape: the parent switched, the child is resumed,
    and the child's calls must go to the model the parent is on NOW."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    stream = RecordingStream()
    parent = _parent(tmp_path, stream)
    try:
        job_id = parent._launch_subagent(label="explore", prompt="look around")
        await wait_for(lambda: _completed(parent, job_id))
        first = parent.jobs.get(job_id)
        assert first is not None and first.model_label == "openrouter/anthropic/claude-opus-5-5"
        child_dir = parent.subagent_comms.session_dir_of(job_id)
        assert child_dir is not None
        # The child journalled its birth model; that row is what used to win.
        born = read_model_selection(child_dir)
        assert born is not None and born.selector == "openrouter/anthropic/claude-opus-5-5"

        parent.set_model(SWITCHED, explicit=True)
        text = await _hub_resume(parent, job_id)
        new_id = _resumed_id(parent, text)
        await wait_for(lambda: _completed(parent, new_id))

        # The request the resumed child SENT, not a label about it.
        assert stream.selectors_for(RESUME_PROMPT) == ["openrouter/deepseek/deepseek-flash"]
        row = parent.jobs.get(new_id)
        assert row is not None and row.model_label == "openrouter/deepseek/deepseek-flash"
        # The receipt names the inherited model and the change from last time.
        assert (
            "on this session's model (openrouter/deepseek/deepseek-flash)"
            " (its previous run was on openrouter/anthropic/claude-opus-5-5)"
        ) in text
        # The journal now records the model the resumed run actually used, so a
        # reader of the child's transcript is not told the old one.
        after = read_model_selection(child_dir)
        assert after is not None and after.selector == "openrouter/deepseek/deepseek-flash"
    finally:
        await parent.dispose()


@pytest.mark.asyncio
async def test_a_pinned_child_resumes_on_its_re_resolved_tier(tmp_path, monkeypatch):
    """A pinned child re-resolves its tier from CURRENT config on resume, like a
    launch does, and the receipt says the model moved."""
    config_dir = tmp_path / "config"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))
    _write_tiers(config_dir, hi="openrouter/moonshotai/kimi-k3")
    stream = RecordingStream()
    parent = _parent(tmp_path, stream)
    try:
        job_id = parent._launch_subagent(label="review", prompt="review it", effort="hi")
        await wait_for(lambda: _completed(parent, job_id))
        first = parent.jobs.get(job_id)
        assert first is not None and first.model_label == "openrouter/moonshotai/kimi-k3"

        # The operator re-points the tier while the child is stopped. The parent
        # switching too proves the pin, not the parent, decides.
        _write_tiers(config_dir, hi="openrouter/qwen/qwen3.8-max")
        parent.set_model(SWITCHED, explicit=True)
        text = await _hub_resume(parent, job_id)
        new_id = _resumed_id(parent, text)
        await wait_for(lambda: _completed(parent, new_id))

        assert stream.selectors_for(RESUME_PROMPT) == ["openrouter/qwen/qwen3.8-max"]
        row = parent.jobs.get(new_id)
        assert row is not None and row.model_label == "openrouter/qwen/qwen3.8-max"
        assert row.owns_model is True
        assert (
            "on openrouter/qwen/qwen3.8-max"
            " (its previous run was on openrouter/moonshotai/kimi-k3)"
        ) in text
    finally:
        await parent.dispose()


@pytest.mark.asyncio
async def test_a_resume_on_an_unchanged_model_names_it_without_a_change_note(tmp_path, monkeypatch):
    """The change note appears only when the model changed, so it stays a signal."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    stream = RecordingStream()
    parent = _parent(tmp_path, stream)
    try:
        job_id = parent._launch_subagent(label="explore", prompt="look around")
        await wait_for(lambda: _completed(parent, job_id))
        text = await _hub_resume(parent, job_id)
        new_id = _resumed_id(parent, text)
        await wait_for(lambda: _completed(parent, new_id))
        assert "on this session's model (openrouter/anthropic/claude-opus-5-5)" in text
        assert "previous run" not in text
    finally:
        await parent.dispose()


@pytest.mark.asyncio
async def test_a_top_level_resume_still_restores_its_journalled_model(tmp_path):
    """Only the child source changed. An ordinary ``--resume`` still opens on
    the model the conversation last selected, whatever today's default is."""
    directory = tmp_path / "sessions" / "top"
    stream = RecordingStream()
    owner = Session(
        model=BIRTH,
        stream_fn=stream,
        tools=[],
        transcript=Transcript(directory),
        system_blocks_provider=lambda: ["stable"],
    )
    await owner.prompt("seed")
    owner.set_model(SWITCHED, explicit=True)
    await owner.dispose()

    resumed_stream = RecordingStream()
    resumed = Session(
        model=BIRTH,
        model_source="config",
        stream_fn=resumed_stream,
        tools=[],
        transcript=Transcript(directory),
        system_blocks_provider=lambda: ["stable"],
    )
    try:
        assert resumed.model_label == "openrouter/deepseek/deepseek-flash"
        await resumed.prompt("resume")
        assert resumed_stream.requests[-1].model.model_id == "deepseek/deepseek-flash"
    finally:
        await resumed.dispose()


# ---------------------------------------------------------------------------
# Nested children (D9.1, D9.2): "inherit" means the REAL parent, not the root
# ---------------------------------------------------------------------------

FLASH = "openrouter/deepseek/deepseek-flash"
OPUS = "openrouter/anthropic/claude-opus-5-5"


class HangingThenDoneStream(RecordingStream):
    """Holds the MANAGER's first turn open so it stays a live session while its
    worker runs, settles, and is resumed. Every other request answers at once."""

    def __init__(self, hold: str) -> None:
        super().__init__()
        self.hold = hold
        self.release = asyncio.Event()

    def __call__(self, request: ChatRequest, signal: AbortSignal | None):
        self.requests.append(request)
        text = " ".join(
            block.text
            for message in request.messages
            for block in (getattr(message, "content", None) or [])
            if isinstance(block, TextContent)
        )
        holding = self.hold in text and not self.release.is_set()

        async def gen():
            if holding:
                await self.release.wait()
            yield StreamTextDelta(delta="done")
            yield StreamEndEvent(stop_reason="stop")

        return gen()


async def _manager_with_worker(tmp_path, stream: RecordingStream):
    """Root on Opus; a manager pinned by ``effort='lo'`` to flash; a worker the
    manager launched, which inherits flash. Returns ``(root, mgr_id, worker_id)``
    with the manager's session still live."""
    config_dir = tmp_path / "config"
    _write_tiers(config_dir, lo=FLASH)
    root = _parent(tmp_path, stream)
    mgr_id = root._launch_subagent(label="mgr", prompt="MANAGE-HOLD", effort="lo")
    await wait_for(lambda: (r := root.subagent_comms._record(mgr_id)) is not None and r.child)
    manager = root.subagent_comms._record(mgr_id).child
    assert manager.model_label == FLASH
    worker_id = manager._launch_subagent(label="worker", prompt="do the work")
    await wait_for(lambda: _completed(manager, worker_id))
    assert stream.selectors_for("do the work") == [FLASH]
    return root, manager, mgr_id, worker_id


@pytest.mark.asyncio
async def test_a_managers_worker_resumes_on_the_managers_live_model(tmp_path, monkeypatch):
    """R1/Q1: the root resumes a worker whose manager is still running. It must
    come back on the manager's flash, not the root's Opus."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    stream = HangingThenDoneStream(hold="MANAGE-HOLD")
    root, manager, mgr_id, worker_id = await _manager_with_worker(tmp_path, stream)
    try:
        text = await _hub_resume(root, worker_id)
        new_id = _resumed_id(root, text)
        await wait_for(lambda: _completed(root, new_id))
        assert stream.selectors_for(RESUME_PROMPT) == [FLASH]
        row = root.jobs.get(new_id)
        assert row is not None and row.model_label == FLASH
        # Inherited, not pinned: the attribution the launch line reads.
        assert row.owns_model is False
        assert f"on its parent's model ({FLASH})" in text
        assert "previous run" not in text
    finally:
        stream.release.set()
        await root.dispose()


@pytest.mark.asyncio
async def test_a_managers_worker_follows_the_managers_switch(tmp_path, monkeypatch):
    """The manager's CURRENT model wins over its launch tier: a parent switch
    moves a resumed child, at every depth."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    stream = HangingThenDoneStream(hold="MANAGE-HOLD")
    root, manager, mgr_id, worker_id = await _manager_with_worker(tmp_path, stream)
    try:
        qwen = BIRTH.model_copy(update={"model_id": "qwen/qwen3.8-max"})
        manager.set_model(qwen, explicit=True)
        text = await _hub_resume(root, worker_id)
        new_id = _resumed_id(root, text)
        await wait_for(lambda: _completed(root, new_id))
        assert stream.selectors_for(RESUME_PROMPT) == ["openrouter/qwen/qwen3.8-max"]
        assert f"(its previous run was on {FLASH})" in text
    finally:
        stream.release.set()
        await root.dispose()


@pytest.mark.asyncio
async def test_a_managers_worker_resumes_on_the_managers_recorded_model_after_restart(
    tmp_path, monkeypatch
):
    """Q1 across a restart, plus Q2: the manager is gone and its session is not
    rebuilt, so the worker takes the model RECORDED for the manager. The tier is
    re-pointed first, which proves the recorded model (not a re-resolved tier)
    is read. The receipt names the worker's previous model from its saved record,
    because the worker's own job row lived on the manager's job manager."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    stream = HangingThenDoneStream(hold="MANAGE-HOLD")
    root, manager, mgr_id, worker_id = await _manager_with_worker(tmp_path, stream)
    stream.release.set()
    await wait_for(lambda: _completed(root, mgr_id))
    await root._persist_subagent_roster()
    await root.dispose()

    # Restarted: the manager ran on flash; the operator now switches the ROOT's
    # model AND the manager's tier. Neither may move the worker off the model
    # its manager actually ran on.
    _write_tiers(tmp_path / "config", lo="openrouter/moonshotai/kimi-k3")
    after = RecordingStream()
    revived = _parent(tmp_path, after)
    try:
        assert revived.jobs.get(worker_id) is None, "precondition: the worker's row is gone"
        assert revived.subagent_comms.last_model_label(worker_id) == FLASH
        revived.set_model(SWITCHED.model_copy(update={"model_id": "openai/gpt-6"}), explicit=True)
        text = await _hub_resume(revived, worker_id)
        new_id = _resumed_id(revived, text)
        await wait_for(lambda: _completed(revived, new_id))
        assert after.selectors_for(RESUME_PROMPT) == [FLASH]
        assert f"on its parent's model ({FLASH})" in text
    finally:
        await revived.dispose()


@pytest.mark.asyncio
async def test_a_nested_child_after_restart_still_reports_its_previous_model(tmp_path, monkeypatch):
    """Q2 when the model DID change: a PINNED worker of the manager, whose tier
    is re-pointed across a restart. Its job row lived on the manager's job
    manager and is gone, so the "(its previous run was on …)" note can only come
    from the worker's own saved record."""
    config_dir = tmp_path / "config"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))
    _write_tiers(config_dir, lo=FLASH, hi="openrouter/moonshotai/kimi-k3")
    stream = HangingThenDoneStream(hold="MANAGE-HOLD")
    root = _parent(tmp_path, stream)
    mgr_id = root._launch_subagent(label="mgr", prompt="MANAGE-HOLD", effort="lo")
    await wait_for(lambda: (r := root.subagent_comms._record(mgr_id)) is not None and r.child)
    manager = root.subagent_comms._record(mgr_id).child
    worker_id = manager._launch_subagent(label="worker", prompt="do the work", effort="hi")
    await wait_for(lambda: _completed(manager, worker_id))
    stream.release.set()
    await wait_for(lambda: _completed(root, mgr_id))
    await root._persist_subagent_roster()
    await root.dispose()

    _write_tiers(config_dir, lo=FLASH, hi="openrouter/qwen/qwen3.8-max")
    after = RecordingStream()
    revived = _parent(tmp_path, after)
    try:
        assert revived.jobs.get(worker_id) is None, "precondition: the worker's row is gone"
        text = await _hub_resume(revived, worker_id)
        new_id = _resumed_id(revived, text)
        await wait_for(lambda: _completed(revived, new_id))
        assert after.selectors_for(RESUME_PROMPT) == ["openrouter/qwen/qwen3.8-max"]
        assert (
            "on openrouter/qwen/qwen3.8-max (its previous run was on openrouter/moonshotai/kimi-k3)"
        ) in text
    finally:
        await revived.dispose()


@pytest.mark.asyncio
async def test_a_nested_child_whose_parent_is_lost_keeps_its_model_and_says_so(
    tmp_path, monkeypatch
):
    """D9.2: the parent's model cannot be found (a legacy sidecar with no saved
    labels and no parent record). The child keeps what it last ran on, never
    silently the root's, and the receipt says why."""
    import json

    from local_operator.session.session import SUBAGENT_ROSTER_SIDECAR

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    stream = HangingThenDoneStream(hold="MANAGE-HOLD")
    root, manager, mgr_id, worker_id = await _manager_with_worker(tmp_path, stream)
    stream.release.set()
    await wait_for(lambda: _completed(root, mgr_id))
    await root._persist_subagent_roster()
    await root.dispose()

    # Rewrite the sidecar into the pre-fix shape: no saved labels, no manager.
    sidecar = tmp_path / "sess" / SUBAGENT_ROSTER_SIDECAR
    details = json.loads(sidecar.read_text())
    details["records"] = [
        {k: v for k, v in row.items() if k != "model_label"}
        for row in details["records"]
        if row.get("job_id") != mgr_id
    ]
    details["jobs"] = [row for row in details["jobs"] if row.get("id") != mgr_id]
    sidecar.write_text(json.dumps(details))

    after = RecordingStream()
    revived = _parent(tmp_path, after)
    try:
        text = await _hub_resume(revived, worker_id)
        new_id = _resumed_id(revived, text)
        await wait_for(lambda: _completed(revived, new_id))
        assert after.selectors_for(RESUME_PROMPT) == [FLASH]
        assert "its parent's model could not be found; kept its previous model" in text
        assert f"on {FLASH}" in text
    finally:
        await revived.dispose()


@pytest.mark.asyncio
async def test_a_direct_child_still_takes_the_roots_model(tmp_path, monkeypatch):
    """D9.2 condition 1: the fallback is for depth 2+ only. A direct child's
    parent IS the root, so it still follows the root's switch (see the first
    test) and carries no fallback note."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    stream = RecordingStream()
    parent = _parent(tmp_path, stream)
    try:
        job_id = parent._launch_subagent(label="explore", prompt="look around")
        await wait_for(lambda: _completed(parent, job_id))
        parent.subagent_comms._record(job_id).model_label = ""
        parent.set_model(SWITCHED, explicit=True)
        text = await _hub_resume(parent, job_id)
        new_id = _resumed_id(parent, text)
        await wait_for(lambda: _completed(parent, new_id))
        assert stream.selectors_for(RESUME_PROMPT) == [FLASH]
        assert "could not be found" not in text
    finally:
        await parent.dispose()
