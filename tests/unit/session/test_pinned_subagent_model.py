"""A pinned subagent tier fails the launch loudly instead of inheriting.

The incident these pin: a reviewer role bound to a cross-family model through
its ``effort`` tier launched, the model 403'd, the author retried at a
different tier that resolved to nothing, and the "independent" review ran on
the author's own model and approved a wrong fix. Three seams close it:

* ``_resolve_subagent_model(strict=True)`` raises ``SubagentModelUnavailable``
  for a tier that is named but unresolvable — the launch path is strict, the
  session-naming path is not.
* the ``task`` tool names the tier and says not to retry elsewhere;
* ``SubagentStartEvent.model`` states the model a child really runs on, and a
  pinned child that dies of access names its pinned model in the error.
"""

from __future__ import annotations

import pytest
import yaml

from local_operator.harness.subagent import (
    SubagentModelUnavailable,
    _describe_child_failure,
)
from local_operator.harness.types import (
    AbortSignal,
    AgentEvent,
    ChatRequest,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    SubagentStartEvent,
    TextContent,
    ToolContext,
)
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.tools.builtin import execute_task

MODEL = ModelSpec(
    provider="openrouter",
    model_id="qwen/qwen3.8-max",
    context_window=100_000,
    max_output_tokens=4_096,
)


class OneShotStream:
    def __init__(self) -> None:
        self.requests: list[ChatRequest] = []

    def __call__(self, request: ChatRequest, signal: AbortSignal | None):
        self.requests.append(request)

        async def gen():
            yield StreamTextDelta(delta="child did the work")
            yield StreamEndEvent(stop_reason="stop")

        return gen()


def make_session(tmp_path) -> Session:
    return Session(
        model=MODEL,
        stream_fn=OneShotStream(),
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["stable", "env"],
    )


def write_tiers(config_dir, **tiers: str) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.yml").write_text(
        yaml.safe_dump({"values": {"subagents": {"models": tiers}}})
    )


async def wait_for(predicate, timeout: float = 5.0) -> None:
    import asyncio

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.01)


# ---------------------------------------------------------------------------
# strict resolution
# ---------------------------------------------------------------------------


def test_strict_launch_fails_on_a_tier_with_no_model(tmp_path, monkeypatch):
    """The exact shape of the incident: ``effort='hi'`` is asked for, nothing
    is configured at ``subagents.models.hi``. The launch must raise, not
    inherit the parent's model."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    write_tiers(tmp_path / "config", lo="openrouter/moonshotai/kimi-k3")
    session = make_session(tmp_path)

    with pytest.raises(SubagentModelUnavailable) as caught:
        session._launch_subagent(label="review", prompt="review it", effort="hi")
    assert caught.value.tier == "hi"
    assert "no model configured at subagents.models.hi" in caught.value.reason
    # No job row for a child that would have run on the wrong model.
    assert not session.jobs.list()


def test_strict_launch_fails_on_a_malformed_selector(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    write_tiers(tmp_path / "config", hi="just-a-model-no-provider")
    session = make_session(tmp_path)

    with pytest.raises(SubagentModelUnavailable) as caught:
        session._launch_subagent(label="review", prompt="review it", effort="hi")
    assert caught.value.tier == "hi"
    assert "lacks provider/model" in caught.value.reason


def test_a_tier_outside_the_registry_set_never_launches(tmp_path, monkeypatch):
    """The launch half of the reader's narrowing (review R1-F2). A hand-added
    ``xl:`` is not advertised by the schema because the config watcher's
    per-registry-key diff could never re-render it; leaving it launchable
    through a role pin would make the schema and the launch path disagree in
    exactly the direction this PR exists to close."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    write_tiers(tmp_path / "config", xl="openrouter/moonshotai/kimi-k3")
    session = make_session(tmp_path)

    with pytest.raises(SubagentModelUnavailable) as caught:
        session._launch_subagent(label="review", prompt="review it", effort="xl")
    assert caught.value.tier == "xl"
    assert "no model configured at subagents.models.xl" in caught.value.reason
    assert not session.jobs.list()


def test_a_padded_selector_resolves_to_a_clean_provider(tmp_path, monkeypatch):
    """Q-1: ``lop config edit`` preserves surrounding whitespace verbatim, so
    a padded selector was advertised by the schema (which stripped) but the
    strict launch path (which did not) built ``ModelSpec(provider='  openai')``
    and died on the first provider call. The strip now lives once in
    ``read_effort_tier_selectors``, so both consumers see the same value."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    write_tiers(tmp_path / "config", hi="  openrouter/moonshotai/kimi-k3  ")
    session = make_session(tmp_path)

    spec = session._resolve_subagent_model("task", "hi")
    assert spec is not None
    assert spec.provider == "openrouter"
    assert spec.model_id == "moonshotai/kimi-k3"


@pytest.mark.asyncio
async def test_no_effort_still_inherits_the_parent(tmp_path, monkeypatch):
    """The ordinary case must be untouched: a plain child with no tier and no
    config at all launches on the parent's model."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    session = make_session(tmp_path)

    job_id = session._launch_subagent(label="sub", prompt="go")
    assert session.jobs.get(job_id) is not None
    await session.dispose()


def test_lenient_resolution_still_inherits_for_naming(tmp_path, monkeypatch, caplog):
    """Session naming prefers ``lo`` but has a sound fallback of its own; it
    must keep getting ``None`` (inherit) rather than an exception when the
    tier is unset -- and SILENTLY, because "no subagents.models at all" is the
    default configuration and a warning on every naming call for a default
    is noise (review R1). A malformed selector still warns."""
    import logging

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    session = make_session(tmp_path)
    with caplog.at_level(logging.WARNING):
        assert session._resolve_subagent_model("task", "lo") is None
        assert session._resolve_subagent_model("task", "lo", strict=False) is None
    assert not [r for r in caplog.records if "subagent model tier" in r.getMessage()]

    write_tiers(tmp_path / "config", lo="no-provider-here")
    with caplog.at_level(logging.WARNING):
        assert session._resolve_subagent_model("task", "lo") is None
    assert [r for r in caplog.records if "lacks provider/model" in r.getMessage()]


@pytest.mark.asyncio
async def test_resume_refuses_a_tier_that_broke_since_launch(tmp_path, monkeypatch):
    """A resume is a second launch (review R2): a child launched on ``hi``
    whose tier has since been removed must NOT come back on the parent's
    model with the panel still saying ``hi``. The refusal lands in the same
    ``(None, reason)`` slot every other resume refusal uses."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    write_tiers(tmp_path / "config", hi=f"{MODEL.provider}/{MODEL.model_id}")
    session = make_session(tmp_path)
    job_id = session._launch_subagent(label="review", prompt="review it", effort="hi")
    await wait_for(lambda: (j := session.jobs.get(job_id)) is not None and j.status == "completed")

    # The tier disappears between launch and resume.
    (tmp_path / "config" / "config.yml").write_text("values: {}\n")
    new_id, reason = session.subagent_comms.resume(job_id, "carry on")
    assert new_id is None
    assert reason is not None and "effort tier 'hi' is unavailable" in reason
    await session.dispose()


@pytest.mark.asyncio
async def test_start_event_names_the_model_the_child_runs_on(tmp_path, monkeypatch):
    """A consumer of the stream (the Axis runner) can state which model a
    delegated review ran on from ``subagent_start`` alone."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    write_tiers(tmp_path / "config", hi="openrouter/moonshotai/kimi-k3")
    session = make_session(tmp_path)
    events: list[AgentEvent] = []
    session.subscribe(events.append)

    session._launch_subagent(label="review", prompt="review it", effort="hi")
    await wait_for(lambda: any(isinstance(e, SubagentStartEvent) for e in events))
    [start] = [e for e in events if isinstance(e, SubagentStartEvent)]
    assert start.model == "openrouter/moonshotai/kimi-k3"
    # And the wire form carries it, since exec --json is model_dump.
    assert start.model_dump(mode="json")["model"] == "openrouter/moonshotai/kimi-k3"
    await session.dispose()


# ---------------------------------------------------------------------------
# the task tool's wording
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wait_names_the_model_a_task_child_ran_on(tmp_path, monkeypatch):
    """The parent model's own view: ``wait`` reports the harness-recorded
    model of a settled task child in its header, so a launcher can state
    which model produced a delegated verdict instead of assuming it."""
    from local_operator.tools.builtin import execute_wait

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    write_tiers(tmp_path / "config", hi=f"{MODEL.provider}/{MODEL.model_id}")
    session = make_session(tmp_path)
    job_id = session._launch_subagent(label="review", prompt="review it", effort="hi")
    await wait_for(lambda: (j := session.jobs.get(job_id)) is not None and j.status == "completed")

    context = ToolContext(jobs=session.jobs, subagent_comms=session.subagent_comms)
    result = await execute_wait("call-1", {"job_id": job_id, "wait_ms": 1000}, None, None, context)
    text = "".join(block.text for block in result.content if isinstance(block, TextContent))
    assert f"[completed] model={MODEL.provider}/{MODEL.model_id}" in text
    await session.dispose()


@pytest.mark.asyncio
async def test_task_tool_tells_the_model_not_to_retry_on_another_tier(tmp_path, monkeypatch):
    """The launch-time backstop. ``hi`` IS configured here, so the tool's own
    argument validation lets it through, and the launcher's refusal stands in
    for the case validation cannot see (a selector that no longer builds)."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    write_tiers(tmp_path / "config", hi="anthropic/claude-opus-5")

    def launcher(label, prompt, *, agent="task", effort=None):
        raise SubagentModelUnavailable("hi", "no model configured at subagents.models.hi")

    context = ToolContext(subagent_launcher=launcher)
    result = await execute_task(
        "call-1", {"label": "review", "prompt": "review it", "effort": "hi"}, None, None, context
    )
    text = "".join(block.text for block in result.content if isinstance(block, TextContent))
    assert result.is_error
    assert "effort tier 'hi' is unavailable" in text
    assert "Do NOT retry at a different effort tier" in text
    assert "subagents.models.hi" in text


# ---------------------------------------------------------------------------
# a pinned child that dies of access
# ---------------------------------------------------------------------------


def test_pinned_child_auth_failure_names_the_pinned_model():
    spec = ModelSpec(
        provider="openrouter", model_id="meta/muse-spark-1.2", context_window=1, max_output_tokens=1
    )
    rendered = (
        "authentication failed (HTTP 403): This model requires you to complete "
        "18+ age confirmation."
    )
    out = _describe_child_failure(rendered, spec)
    assert out.startswith(rendered)
    assert "pinned model openrouter/meta/muse-spark-1.2 is unavailable" in out
    assert "do not re-run it at another effort tier" in out


def test_pinned_child_transient_failure_is_left_alone():
    """A 5xx on a pinned child should be retried on the SAME model; the
    suffix would argue against that, so it must not appear."""
    spec = ModelSpec(
        provider="openrouter", model_id="moonshotai/kimi-k3", context_window=1, max_output_tokens=1
    )
    rendered = "transient provider error (HTTP 502): upstream unavailable"
    assert _describe_child_failure(rendered, spec) == rendered


def test_unpinned_child_failure_is_left_alone():
    rendered = "authentication failed (HTTP 401): bad key"
    assert _describe_child_failure(rendered, None) == rendered
