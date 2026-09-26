"""Pod leads drive their own subagents; nested replies reach the real parent.

BEN-7-D5. Before it, a root -> lead -> worker tree was broken three ways
(architect's probe, driving ``SubagentComms`` with the ``test_comms`` fakes):
the worker's reply landed on the ROOT, ``cancel`` on the worker was
``unknown job``, and ``live_ids`` did not list the running worker. A lead also
held only the message-only ``hub``, so it could not supervise its pod.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.comms import SubagentComms
from local_operator.harness.loop import validate_tool_arguments
from local_operator.harness.types import ModelSpec, TextContent, ToolContext, ToolResult
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.tools.builtin import (
    HubChildParams,
    HubParams,
    build_hub_tool,
    execute_hub,
)
from tests.unit.harness.test_comms import FakeChild, FakeJobs, FakeParent

MODEL = ModelSpec(provider="test", model_id="m", context_window=100_000)


class FakeLead(FakeChild):
    """A live mid-tier child: it owns a job manager, like a real Session."""

    def __init__(self) -> None:
        super().__init__()
        self.jobs = FakeJobs()


def tree(tmp_path: Path) -> tuple[SubagentComms, FakeParent, FakeLead, FakeChild]:
    """root -> lead -> worker, plus a sibling of the lead (``other``) with its
    own child (``cousin``), each job living in its REAL parent's manager."""
    root_jobs = FakeJobs()
    root = FakeParent(root_jobs)
    comms = SubagentComms(root)  # type: ignore[arg-type]
    lead = FakeLead()
    other = FakeLead()
    worker = FakeChild()
    cousin = FakeChild()
    root_jobs.add("lead")
    root_jobs.add("other")
    lead.jobs.add("w")
    other.jobs.add("cousin")
    comms.record_launch("lead", "lead", agent_role="team:pod")
    comms.record_launch("other", "other")
    comms.record_launch("w", "worker", parent_job_id="lead", agent_role="coder")
    comms.record_launch("cousin", "cousin", parent_job_id="other")
    for job_id, session in (("lead", lead), ("other", other), ("w", worker), ("cousin", cousin)):
        comms.attach(job_id, session, tmp_path / job_id)  # type: ignore[arg-type]
    return comms, root, lead, worker


def test_a_workers_reply_reaches_its_lead_not_the_root(tmp_path):
    comms, root, lead, _worker = tree(tmp_path)
    comms.reply_to_parent("w", "blocked on the schema")
    assert len(lead.asides) == 1 and root.asides == []
    # A direct child's reply still reaches the root, as before.
    comms.reply_to_parent("lead", "pod done")
    assert len(root.asides) == 1


def test_a_reply_falls_back_to_the_root_once_the_lead_settled(tmp_path):
    comms, root, lead, _worker = tree(tmp_path)
    comms.detach("lead")
    comms.reply_to_parent("w", "late note")
    assert lead.asides == [] and len(root.asides) == 1


def test_a_running_grandchild_is_running(tmp_path):
    comms, _root, _lead, _worker = tree(tmp_path)
    assert set(comms.live_ids()) == {"lead", "other", "w", "cousin"}
    assert comms.live_ids(scope="lead") == ["w"]


@pytest.mark.asyncio
async def test_cancel_and_pause_reach_the_manager_that_owns_the_job(tmp_path):
    comms, _root, lead, _worker = tree(tmp_path)
    delivery = await comms.cancel("w")
    assert delivery.outcome == "cancelled" and lead.jobs.cancelled == ["w"]
    lead.jobs.add("w2")
    comms.record_launch("w2", "w2", parent_job_id="lead")
    comms.attach("w2", FakeChild(), tmp_path / "w2")  # type: ignore[arg-type]
    paused = await comms.pause("w2")
    assert paused.outcome == "paused" and lead.jobs.cancelled == ["w", "w2"]


def test_a_scoped_address_outside_the_subtree_is_refused(tmp_path):
    comms, _root, _lead, _worker = tree(tmp_path)
    assert comms.resolve("w", scope="lead") == (["w"], None)
    assert comms.resolve("worker", scope="lead") == (["w"], None)
    for target in ("cousin", "other", "lead"):
        ids, error = comms.resolve(target, scope="lead")
        assert ids == [] and error is not None and "not your subagent" in error
    # Unscoped (the top session) still sees the whole tree.
    assert comms.resolve("cousin") == (["cousin"], None)


def _settle_with_transcript(comms: SubagentComms, jobs: FakeJobs, job_id: str) -> None:
    jobs.jobs[job_id].status = "cancelled"
    record = comms._records[job_id]
    assert record.session_dir is not None
    record.session_dir.mkdir(parents=True, exist_ok=True)
    (record.session_dir / "transcript.jsonl").write_text("{}\n", encoding="utf-8")
    comms.detach(job_id)


def test_resume_launches_under_the_live_lead(tmp_path, monkeypatch):
    comms, _root, lead, _worker = tree(tmp_path)
    _settle_with_transcript(comms, lead.jobs, "w")
    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        "local_operator.harness.subagent.run_subagent",
        lambda **kwargs: (seen.update(kwargs), "w-2")[1],
    )
    monkeypatch.setattr(
        "local_operator.harness.subagent.resolve_launch_target",
        lambda agent, parent, carried=None: seen.setdefault("target_parent", parent),
    )
    new_id, error = comms.resume("w", "carry on")
    assert error is None and new_id == "w-2"
    assert seen["parent_session"] is lead and seen["jobs_manager"] is lead.jobs
    assert seen["target_parent"] is lead


def test_resume_of_an_orphan_falls_back_to_the_root(tmp_path, monkeypatch):
    """The other half of the routing contract (D5 addendum 2, manager ruling
    2026-09-26): a live parent owns the continuation, a GONE parent does not
    refuse it.

    Refusing would delete the recovery ``_inherited_model`` was built for — it
    walks the recorded lineage precisely so a nested child can be resumed after
    a restart, when no parent session exists. The fallback re-parents the JOB
    only; the child's team rides its own record, which
    ``test_nested_teams.test_an_orphaned_pod_worker_resumes_on_the_root_but_keeps_its_pod``
    pins.
    """
    comms, root, lead, _worker = tree(tmp_path)
    _settle_with_transcript(comms, lead.jobs, "w")
    comms.detach("lead")
    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        "local_operator.harness.subagent.run_subagent",
        lambda **kwargs: (seen.update(kwargs), "w-2")[1],
    )
    new_id, error = comms.resume("w", "carry on")
    assert error is None and new_id == "w-2"
    assert seen["parent_session"] is root and seen["jobs_manager"] is root.jobs


# -- the tool: shape and scope -------------------------------------------------


def body(result: ToolResult) -> str:
    block = result.content[0]
    assert isinstance(block, TextContent)
    return block.text


def context_for(comms: SubagentComms, job_id: str | None, *, may_delegate: bool) -> ToolContext:
    return ToolContext(cwd=".", subagent_comms=comms, job_id=job_id, may_delegate=may_delegate)


def test_a_delegating_child_gets_the_parent_shape_and_others_keep_messages(tmp_path):
    """Who gets which hub, and the two schemas that must not move.

    The message-only child tool and the TOP session's parent tool are
    cache-prefix strings: both are asserted byte-identical to the schema their
    params model renders, so the lead's variant cannot leak into either
    (BEN-7-D5 addendum, manager ruling 2026-09-26).
    """
    comms, _root, _lead, _worker = tree(tmp_path)
    lead_tool = build_hub_tool(context_for(comms, "lead", may_delegate=True))
    worker_tool = build_hub_tool(context_for(comms, "w", may_delegate=False))
    top_tool = build_hub_tool(context_for(comms, None, may_delegate=True))
    assert lead_tool is not None and worker_tool is not None and top_tool is not None

    assert worker_tool.parameters == HubChildParams.model_json_schema()
    assert top_tool.parameters == HubParams.model_json_schema()

    # The lead's variant: the parent's ops, with ``op`` optional so its own
    # reply validates, and ``to`` naming the parent.
    assert set(lead_tool.parameters["properties"]) == set(
        HubParams.model_json_schema()["properties"]
    )
    assert lead_tool.parameters["required"] == []
    assert {"type": "null"} in lead_tool.parameters["properties"]["op"]["anyOf"]
    assert "parent" in lead_tool.parameters["properties"]["to"]["description"]
    assert "delegated to you" in lead_tool.description


@pytest.mark.asyncio
async def test_a_leads_bare_reply_validates_and_reaches_its_parent(tmp_path):
    """The regression the schema variant exists for, pinned at the loop's gate.

    ``comms.TO_CHILD_INSTRUCTIONS["ask"]`` tells EVERY child to "Answer it now
    with the ``hub`` tool", and ``loop.validate_tool_arguments`` runs BEFORE
    ``execute``. Against the plain parent schema that call was rejected as
    invalid arguments, so a lead could never answer its own parent: the ask
    timed out and the child died on the repeated-error guard. Asserted at the
    validator because that is where it was failing, and at the executor because
    passing validation is not the same as routing.
    """
    comms, root, lead, _worker = tree(tmp_path)
    tool = build_hub_tool(context_for(comms, "lead", may_delegate=True))
    assert tool is not None
    args = {"message": "not stuck; fixtures are slow"}
    assert validate_tool_arguments(tool, args, json.dumps(args)) == []

    ctx = context_for(comms, "lead", may_delegate=True)
    result = await execute_hub("c1", args, None, None, ctx)
    assert not result.is_error
    assert "delivered to the parent" in body(result)
    # The lead's parent IS the root here; the nested hop (worker -> lead) is
    # ``test_a_workers_reply_reaches_its_lead_not_the_root``.
    assert len(root.asides) == 1 and lead.asides == []

    # The top session's tool still requires ``op``: the variant is the lead's.
    top = build_hub_tool(context_for(comms, None, may_delegate=True))
    assert top is not None
    assert validate_tool_arguments(top, args, json.dumps(args)) != []


@pytest.mark.asyncio
async def test_a_lead_drives_only_its_own_pod(tmp_path):
    comms, root, lead, worker = tree(tmp_path)
    ctx = context_for(comms, "lead", may_delegate=True)

    listed = await execute_hub("c1", {"op": "list"}, None, None, ctx)
    assert "worker (w)" in body(listed) and "cousin" not in body(listed)

    sent = await execute_hub(
        "c2", {"op": "send", "to": ["worker"], "message": "hi"}, None, None, ctx
    )
    assert "1/1" in body(sent) and len(worker.asides) == 1

    for op in ("send", "steer", "cancel", "pause", "peek", "resume", "ask"):
        refused = await execute_hub(
            "c3", {"op": op, "to": ["cousin"], "message": "x"}, None, None, ctx
        )
        assert refused.is_error and "not your subagent" in body(refused), op

    everyone = await execute_hub(
        "c4", {"op": "send", "to": ["all"], "message": "x"}, None, None, ctx
    )
    assert "1/1" in body(everyone)

    up = await execute_hub(
        "c5", {"op": "send", "to": ["parent"], "message": "pod done"}, None, None, ctx
    )
    assert "delivered to the parent" in body(up) and len(root.asides) == 1

    cancelled = await execute_hub("c6", {"op": "cancel", "to": ["w"]}, None, None, ctx)
    assert "cancelled" in body(cancelled) and lead.jobs.cancelled == ["w"]


@pytest.mark.asyncio
async def test_a_non_delegating_child_still_only_messages_its_parent(tmp_path):
    comms, _root, lead, _worker = tree(tmp_path)
    ctx = context_for(comms, "w", may_delegate=False)
    result = await execute_hub(
        "c1", {"op": "cancel", "to": ["lead"], "message": "hi"}, None, None, ctx
    )
    assert "delivered to the parent" in body(result) and len(lead.asides) == 1


# -- the real session: a pod lead is built holding the parent shape ------------


class NoStream:
    def __call__(self, request, signal):  # pragma: no cover - never called
        raise AssertionError("no provider turn expected")


@pytest.mark.asyncio
async def test_a_built_manager_child_holds_the_scoped_parent_hub(tmp_path, monkeypatch):
    from local_operator.agent_profiles import load_seed
    from local_operator.harness import subagent as subagent_mod

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    root = Session(
        model=MODEL,
        stream_fn=NoStream(),
        tools=[],
        transcript=Transcript(tmp_path / "root"),
        system_blocks_provider=lambda: ["stable", "env"],
    )
    root.subagent_comms.record_launch("job-mgr", "mgr", agent_role="manager")
    root.subagent_comms.record_launch("job-cod", "cod", agent_role="coder")
    manager = await subagent_mod._build_child_session(
        label="mgr",
        prompt="lead",
        parent_session=root,
        model_spec=None,
        job_id="job-mgr",
        agent="manager",
        profile=load_seed("manager"),
    )
    coder = await subagent_mod._build_child_session(
        label="cod",
        prompt="code",
        parent_session=root,
        model_spec=None,
        job_id="job-cod",
        agent="coder",
        profile=load_seed("coder"),
    )
    hub = {tool.name: tool for tool in manager._tools}["hub"]
    assert "op" in hub.parameters["properties"]
    coder_hub = {tool.name: tool for tool in coder._tools}["hub"]
    assert "op" not in coder_hub.parameters["properties"]
    # Nothing the prune removed came back with the rebuild.
    assert "wake" not in {tool.name for tool in manager._tools}
    assert "task" not in {tool.name for tool in coder._tools}
    for session in (coder, manager, root):
        await session.dispose()
