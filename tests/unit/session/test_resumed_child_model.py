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
