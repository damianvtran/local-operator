"""``hub op='set_model'``, steer-as-re-probe, served-model readings, subtree pause.

The 2026-11 quota incident: a fable primary hit quota, a manager's children
fell back to sol, and when quota returned they stayed on sol — a subagent's
task is one user message, so the message-boundary re-probe never fired, and
the only lever (pause + resume) cancelled the manager's children, changed
every job id and orphaned owned tabs. These tests drive ``SubagentComms`` and
the ``hub``/``jobs`` tools with the ``test_comms`` fakes, extended with the
model surface a real child ``Session`` exposes (``model``,
``effective_model_label``, ``set_model``, ``request_route_reprobe``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.comms import SUBTREE_RESUME_MESSAGE, SubagentComms
from local_operator.harness.subagent import LaunchTarget
from local_operator.harness.types import ModelSpec, TextContent, ToolContext, ToolResult
from local_operator.tools.builtin import HubParams, execute_hub, execute_jobs
from tests.unit.harness.test_comms import FakeChild, FakeJobs, FakeParent

FABLE = ModelSpec(provider="anthropic", model_id="fable-5.2", context_window=200_000)
SOL = ModelSpec(provider="openai", model_id="gpt-6.1-sol", context_window=200_000)
OPUS = ModelSpec(provider="anthropic", model_id="opus-5-5", context_window=200_000)


class ModelChild(FakeChild):
    """A running child with the model/route surface ``set_model`` drives."""

    def __init__(self, model: ModelSpec, *, fallback: ModelSpec | None = None) -> None:
        super().__init__()
        self.model = model
        self.fallback = fallback
        self.jobs = FakeJobs()
        self.switches: list[tuple[ModelSpec, bool, Any]] = []
        self.reprobes: list[dict[str, Any]] = []

    @property
    def effective_model_label(self) -> str:
        spec = self.fallback or self.model
        return f"{spec.provider}/{spec.model_id}"

    def set_model(self, spec: ModelSpec, *, explicit: bool = False, effort_choice: Any = None):
        # Mirrors ``Session.set_model(explicit=True)``: an explicit choice
        # withdraws an active fallback, same pair or not.
        self.switches.append((spec, explicit, effort_choice))
        self.model = spec
        if explicit:
            self.fallback = None

    def request_route_reprobe(self, *, withdraw: bool = False, launch_pin: object = None):
        self.reprobes.append({"withdraw": withdraw, "launch_pin": launch_pin})
        if withdraw:
            self.fallback = None


class ModelParent(FakeParent):
    def __init__(self, jobs: FakeJobs, model: ModelSpec) -> None:
        super().__init__(jobs)
        self.model = model


def body(result: ToolResult) -> str:
    block = result.content[0]
    assert isinstance(block, TextContent)
    return block.text


def incident_tree(tmp_path: Path):
    """root(fable) -> manager -> {worker-a, worker-b}; every child fell to sol."""
    root_jobs = FakeJobs()
    root = ModelParent(root_jobs, FABLE)
    comms = SubagentComms(root)  # type: ignore[arg-type]
    manager = ModelChild(FABLE, fallback=SOL)
    a = ModelChild(FABLE, fallback=SOL)
    b = ModelChild(FABLE, fallback=SOL)
    root_jobs.add("mgr")
    manager.jobs.add("wa")
    manager.jobs.add("wb")
    comms.record_launch("mgr", "manager", agent_role="manager")
    comms.record_launch("wa", "worker-a", parent_job_id="mgr")
    comms.record_launch("wb", "worker-b", parent_job_id="mgr")
    for job_id, child in (("mgr", manager), ("wa", a), ("wb", b)):
        comms.attach(job_id, child, tmp_path / job_id)  # type: ignore[arg-type]
    return comms, root, manager, a, b


# -- set_model -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_inherit_with_cascade_returns_the_subtree_to_the_primary_in_place(tmp_path):
    comms, _root, manager, a, b = incident_tree(tmp_path)
    assert comms.model_reading("wa") == ("openai/gpt-6.1-sol", "anthropic/fable-5.2")

    out = await comms.set_model(["mgr"], model="inherit", effort=None, cascade=True)

    assert [(d.job_id, d.outcome) for d in out] == [
        ("mgr", "switched"),
        ("wa", "switched"),
        ("wb", "switched"),
    ]
    for child in (manager, a, b):
        assert child.effective_model_label == "anthropic/fable-5.2"
        # Re-probed with the fallback withdrawn and the launch pin CLEARED:
        # the child now inherits rather than owns a tier.
        assert child.reprobes == [{"withdraw": True, "launch_pin": ""}]
    # Same job ids, nothing paused, nothing cancelled: no teardown at all.
    assert manager.jobs.cancelled == [] and comms._records["wa"].paused is False
    assert comms.model_reading("wa") == ("anthropic/fable-5.2", "")


@pytest.mark.asyncio
async def test_without_cascade_only_the_named_child_moves(tmp_path):
    """Negative control for cascade: the manager's workers stay on sol."""
    comms, _root, manager, a, _b = incident_tree(tmp_path)
    await comms.set_model(["mgr"], model="inherit", effort=None)
    assert manager.effective_model_label == "anthropic/fable-5.2"
    assert a.effective_model_label == "openai/gpt-6.1-sol" and a.reprobes == []


@pytest.mark.asyncio
async def test_cascade_inherit_reads_each_parents_new_model(tmp_path):
    """Top-down order: a worker inheriting from a manager just moved to opus
    lands on opus, not on the manager's pre-switch model."""
    comms, _root, manager, a, _b = incident_tree(tmp_path)
    manager.model = OPUS  # the manager was switched to opus by its own /model
    await comms.set_model(["wa"], model="inherit", effort=None)
    assert a.model == OPUS


@pytest.mark.asyncio
async def test_a_role_pin_is_kept_on_cascade_and_reported(tmp_path, monkeypatch):
    comms, _root, _manager, a, _b = incident_tree(tmp_path)
    comms._records["wa"].model_selection_source = "role"
    a.model = OPUS  # the reviewer role pins opus

    out = await comms.set_model(["mgr"], model="inherit", effort="low", cascade=True)

    receipt = {d.job_id: d for d in out}
    assert receipt["wa"].outcome == "switched"
    assert receipt["wa"].error is not None and "kept its role pin anthropic/opus-5-5" in (
        receipt["wa"].error
    )
    # Kept AND re-probed back onto its own pin; the launch pin untouched.
    assert a.model == OPUS and a.reprobes == [{"withdraw": True, "launch_pin": None}]
    assert a.effective_model_label == "anthropic/opus-5-5"


@pytest.mark.asyncio
async def test_the_top_session_naming_a_pinned_child_directly_overrides(tmp_path):
    comms, _root, _manager, a, _b = incident_tree(tmp_path)
    comms._records["wa"].model_selection_source = "role"
    a.model = OPUS
    out = await comms.set_model(["wa"], model="inherit", effort=None)
    assert out[0].error is None and a.model == FABLE


@pytest.mark.asyncio
async def test_a_manager_cannot_override_a_role_pin_even_naming_it(tmp_path):
    comms, _root, _manager, a, _b = incident_tree(tmp_path)
    comms._records["wa"].model_selection_source = "team"
    a.model = OPUS
    out = await comms.set_model(["wa"], model="inherit", effort=None, scope="mgr")
    assert a.model == OPUS and "only the top session" in (out[0].error or "")


@pytest.mark.asyncio
async def test_effort_clamps_onto_the_models_ladder(tmp_path):
    comms, _root, manager, _a, _b = incident_tree(tmp_path)
    manager.model = FABLE.model_copy(
        update={"reasoning_efforts": ["low", "high"], "reasoning_default_effort": "high"}
    )
    await comms.set_model(["mgr"], model=None, effort="low")
    spec, explicit, choice = manager.switches[-1]
    assert spec.reasoning_effort == "low" and explicit and choice == "low"


@pytest.mark.asyncio
async def test_a_settled_child_is_refused_with_the_resume_route(tmp_path):
    comms, _root, manager, _a, _b = incident_tree(tmp_path)
    manager.jobs.jobs["wa"].status = "completed"
    comms.detach("wa")
    out = await comms.set_model(["wa"], model="inherit", effort=None)
    assert out[0].outcome == "failed" and "hub op='resume'" in (out[0].error or "")


@pytest.mark.asyncio
async def test_hub_tool_set_model_receipt_names_the_served_model(tmp_path):
    comms, *_ = incident_tree(tmp_path)
    ctx = ToolContext(cwd=".", subagent_comms=comms)
    result = await execute_hub(
        "c1",
        {"op": "set_model", "to": ["mgr"], "model": "inherit", "cascade": True},
        None,
        None,
        ctx,
    )
    text = body(result)
    assert "set_model: 3/3 subagent(s)" in text
    assert "- worker-a (wa): switched, on anthropic/fable-5.2" in text
    # effort/cascade are set_model-only: elsewhere they are refused, not ignored.
    refused = await execute_hub(
        "c2", {"op": "send", "to": ["mgr"], "message": "hi", "cascade": True}, None, None, ctx
    )
    assert "only valid for op='set_model'" in body(refused)


def test_the_schema_adds_only_effort_and_cascade():
    properties = HubParams.model_json_schema()["properties"]
    assert set(properties) == {
        "model",
        "op",
        "to",
        "message",
        "timeout_ms",
        "range",
        "steps",
        "effort",
        "cascade",
    }
    assert "set_model" in properties["op"]["enum"]


# -- served model on list surfaces ---------------------------------------------


@pytest.mark.asyncio
async def test_hub_list_and_jobs_show_pinned_x_ran_on_y(tmp_path):
    comms, root, *_ = incident_tree(tmp_path)
    # The ``jobs`` renderer reads a few more row fields than ``FakeJob`` models.
    row: Any = root.jobs.jobs["mgr"]
    row.label, row.type, row.settled_at, row.start_time = "manager", "task", None, 0.0
    root.jobs.list = lambda: list(root.jobs.jobs.values())  # type: ignore[attr-defined]
    ctx = ToolContext(cwd=".", subagent_comms=comms, jobs=root.jobs)  # type: ignore[arg-type]

    # R7: the manager INHERITS (no pin), so its fallback names the primary
    # without claiming a pin nobody set.
    listed = body(await execute_hub("c0", {"op": "list"}, None, None, ctx))
    assert "on openai/gpt-6.1-sol (primary anthropic/fable-5.2)" in listed
    assert "pinned anthropic/fable-5.2" not in listed
    # A real pin (the launch recorded ``owns_model``) keeps the pin wording.
    row.owns_model = True
    listed = body(await execute_hub("c1", {"op": "list"}, None, None, ctx))
    assert "pinned anthropic/fable-5.2, ran on openai/gpt-6.1-sol" in listed
    jobs = body(await execute_jobs("c2", {}, None, None, ctx))
    assert "manager  [pinned anthropic/fable-5.2, ran on openai/gpt-6.1-sol]" in jobs

    await comms.set_model(["mgr"], model="inherit", effort=None, cascade=True)
    listed = body(await execute_hub("c3", {"op": "list"}, None, None, ctx))
    assert "pinned " not in listed and "    on anthropic/fable-5.2" in listed
    assert "manager  [on anthropic/fable-5.2]" in body(
        await execute_jobs("c4", {}, None, None, ctx)
    )


# -- steer is a re-probe boundary ----------------------------------------------


def test_steer_requests_a_reprobe_without_withdrawing(tmp_path):
    comms, _root, _manager, a, b = incident_tree(tmp_path)
    assert comms.steer("wa", "switch to the API half").outcome == "injected"
    assert a.reprobes == [{"withdraw": False}] or a.reprobes == [
        {"withdraw": False, "launch_pin": None}
    ]
    # Negative control: a plain send is NOT a boundary.
    comms.send("wb", "fyi")
    assert b.reprobes == []


# -- pause a manager pauses its subtree; resume restores it ----------------------


@pytest.mark.asyncio
async def test_pausing_a_manager_pauses_its_subtree_and_resume_restores_it(tmp_path, monkeypatch):
    comms, root, manager, _a, _b = incident_tree(tmp_path)

    delivery = await comms.pause("mgr", by="parent-hub")

    assert delivery.outcome == "paused"
    assert "worker-a (wa)" in (delivery.error or "") and "worker-b (wb)" in (delivery.error or "")
    # Paused deepest-first BEFORE the manager, not cancelled by its teardown.
    assert manager.jobs.cancelled == ["wa", "wb"] and root.jobs.cancelled == ["mgr"]
    assert comms._records["wa"].paused and comms._records["wa"].subtree_paused_by == "mgr"

    # Settle all three on disk, as the runners' teardown would.
    for job_id in ("wa", "wb", "mgr"):
        session_dir = comms._records[job_id].session_dir
        assert session_dir is not None
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "transcript.jsonl").write_text("{}\n", encoding="utf-8")
        comms.detach(job_id)

    launched: list[dict[str, Any]] = []

    def run_subagent(**kwargs: Any) -> str:
        launched.append(kwargs)
        return f"{kwargs['label']}-2"

    monkeypatch.setattr("local_operator.harness.subagent.run_subagent", run_subagent)
    monkeypatch.setattr(
        "local_operator.harness.subagent.resolve_launch_target",
        lambda agent, parent, *, carried=None: LaunchTarget(
            role=agent, team=None, team_lineage=(), depth=1, reports_to="", is_team_launch=False
        ),
    )
    # The operator resumes ONLY the manager...
    new_id, error = comms.resume("mgr", "carry on")
    assert error is None and new_id == "manager-2"
    assert len(launched) == 1  # ...children wait for the manager to be live.
    # ...and its continuation attaches (same transcript directory: a fold).
    manager2 = ModelChild(FABLE)
    root.jobs.add("manager-2")
    comms.record_launch("manager-2", "manager", agent_role="manager")
    comms.attach("manager-2", manager2, tmp_path / "mgr")  # type: ignore[arg-type]
    import asyncio

    await asyncio.sleep(0)  # the deferred subtree resume runs on the next tick

    resumed = sorted(kwargs["label"] for kwargs in launched[1:])
    assert resumed == ["worker-a", "worker-b"]
    for kwargs in launched[1:]:
        # Homed under the LIVE manager continuation, with the canned message.
        assert kwargs["parent_session"] is manager2
        assert kwargs["prompt"] == SUBTREE_RESUME_MESSAGE
    assert comms._records["wa"].subtree_paused_by == ""


@pytest.mark.asyncio
async def test_pausing_a_leaf_touches_nothing_else(tmp_path):
    """Negative control: a childless pause carries no subtree clause."""
    comms, root, manager, _a, _b = incident_tree(tmp_path)
    delivery = await comms.pause("wa")
    assert delivery.error is None and manager.jobs.cancelled == ["wa"]
    assert root.jobs.cancelled == []


# -- review/QA round 1 remediation ---------------------------------------------


def _settle_paused(comms: SubagentComms, job_ids: tuple[str, ...]) -> None:
    """Write each paused child's transcript and detach it, as teardown would."""
    for job_id in job_ids:
        session_dir = comms._records[job_id].session_dir
        assert session_dir is not None
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "transcript.jsonl").write_text("{}\n", encoding="utf-8")
        comms.detach(job_id)


def _capture_launches(monkeypatch) -> list[dict[str, Any]]:
    launched: list[dict[str, Any]] = []

    def run_subagent(**kwargs: Any) -> str:
        launched.append(kwargs)
        return f"{kwargs['label']}-2"

    monkeypatch.setattr("local_operator.harness.subagent.run_subagent", run_subagent)
    monkeypatch.setattr(
        "local_operator.harness.subagent.resolve_launch_target",
        lambda agent, parent, *, carried=None: LaunchTarget(
            role=agent, team=None, team_lineage=(), depth=1, reports_to="", is_team_launch=False
        ),
    )
    return launched


@pytest.mark.asyncio
async def test_a_child_cancelled_while_its_manager_is_paused_stays_cancelled(tmp_path, monkeypatch):
    """R1: the cancel is the newer word; the manager's resume must not relaunch it."""
    import asyncio

    comms, root, _manager, _a, _b = incident_tree(tmp_path)
    assert (await comms.pause("mgr")).outcome == "paused"
    _settle_paused(comms, ("wa", "wb", "mgr"))

    assert (await comms.cancel("wa")).outcome == "cancelled"
    record = comms._records["wa"]
    assert (record.outcome, record.subtree_paused_by) == ("cancelled", "")

    launched = _capture_launches(monkeypatch)
    assert comms.resume("mgr", "carry on")[1] is None
    root.jobs.add("manager-2")
    comms.record_launch("manager-2", "manager", agent_role="manager")
    comms.attach("manager-2", ModelChild(FABLE), tmp_path / "mgr")  # type: ignore[arg-type]
    await asyncio.sleep(0)
    assert sorted(kwargs["label"] for kwargs in launched[1:]) == ["worker-b"]


@pytest.mark.asyncio
async def test_a_failed_manager_pause_names_its_paused_subtree(tmp_path):
    """R2: the children's pauses landed; the failure receipt must say so."""
    comms, root, _manager, _a, _b = incident_tree(tmp_path)

    async def refuse(job_id: str, **_kwargs: Any) -> bool:
        return False

    root.jobs.cancel = refuse  # type: ignore[method-assign]
    delivery = await comms.pause("mgr")
    assert delivery.outcome == "failed"
    error = delivery.error or ""
    assert "worker-a (wa)" in error and "worker-b (wb)" in error
    assert "hub op='resume'" in error
    assert comms._records["mgr"].paused is False


@pytest.mark.asyncio
async def test_pause_all_counts_cascade_paused_children_as_paused(tmp_path):
    """Q1: one batch reaching the manager first pauses its workers; the batch
    must report them as paused with it, not 'failed — already gone'."""
    comms, *_ = incident_tree(tmp_path)
    ctx = ToolContext(cwd=".", subagent_comms=comms)
    text = body(await execute_hub("c1", {"op": "pause", "to": ["all"]}, None, None, ctx))
    assert "3/3" in text, text
    assert "worker-a (wa): paused — already paused with its parent manager" in text
    assert "failed" not in text


@pytest.mark.asyncio
async def test_unknown_effort_is_refused_and_the_receipt_names_the_applied_level(tmp_path):
    """R4/Q2: an unknown word no longer clamps silently to the default."""
    comms, _root, manager, _a, _b = incident_tree(tmp_path)
    out = await comms.set_model(["mgr"], model=None, effort="turbo")
    assert out[0].outcome == "failed" and "unknown effort 'turbo'" in (out[0].error or "")
    assert manager.switches == []

    manager.model = FABLE.model_copy(
        update={"reasoning_efforts": ["low", "high"], "reasoning_default_effort": "high"}
    )
    out = await comms.set_model(["mgr"], model=None, effort="medium")
    # medium clamps onto the ladder; the receipt says the rung actually applied.
    applied = manager.switches[-1][0].reasoning_effort
    assert applied in ("low", "high") and out[0].error == f"effort {applied}"

    out = await comms.set_model(["mgr"], model=None, effort="auto")
    spec, _explicit, choice = manager.switches[-1]
    assert spec.reasoning_effort == "high" and choice is None
    assert "effort auto" in (out[0].error or "")


@pytest.mark.asyncio
async def test_a_tier_word_resolves_the_tier_model_as_task_does(tmp_path, monkeypatch):
    """R4: lo|med|hi is task()'s vocabulary — it picks the tier's MODEL."""
    comms, root, manager, _a, _b = incident_tree(tmp_path)
    monkeypatch.setattr(
        "local_operator.harness.subagent.model_may_choose_tier", lambda *a, **k: False
    )
    out = await comms.set_model(["mgr"], model=None, effort="hi")
    assert out[0].outcome == "failed" and "model_choice=model" in (out[0].error or "")

    monkeypatch.setattr(
        "local_operator.harness.subagent.model_may_choose_tier", lambda *a, **k: True
    )
    calls: list[tuple[Any, ...]] = []

    def resolve(role: str, tier: str | None, **kwargs: Any) -> tuple[ModelSpec, str]:
        calls.append((role, tier, kwargs.get("model")))
        return OPUS, "effort"

    root._resolve_subagent_selection = resolve  # type: ignore[attr-defined]
    out = await comms.set_model(["mgr"], model=None, effort="HI")
    assert calls == [("manager", "hi", None)]
    assert manager.model == OPUS and out[0].error == "tier hi"
    assert manager.reprobes[-1] == {"withdraw": True, "launch_pin": "anthropic/opus-5-5"}
    both = await comms.set_model(["mgr"], model="inherit", effort="lo")
    assert both[0].outcome == "failed" and "omit model=" in (both[0].error or "")
