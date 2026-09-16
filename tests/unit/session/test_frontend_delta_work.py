"""A scalar viewer edge must not walk retained child histories or rebuild facades.

These are work/ownership invariants, not machine-dependent latency ceilings.
The real AttachedSession callback is included: a cheap store with an expensive
compatibility facade would still freeze the production TUI after the unit test.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from local_operator.harness.types import Usage
from local_operator.session import frontend_state as module
from local_operator.session.attached import AttachedSession
from local_operator.session.frontend_state import (
    FrontendSessionState,
    FrontendStateStore,
    FrontendUpdate,
    JobState,
    SnapshotJobs,
)


def state() -> FrontendSessionState:
    value = FrontendSessionState(
        session_id="benchmark-parent",
        epoch="owner",
        jobs=[
            JobState(
                id="child",
                type="task",
                trajectory=[{"type": "message_update", "delta": "retained"}],
                trajectory_length=500,
                latest_details={"nested": {"status": "running"}},
                todos=[{"name": "Build", "items": [{"text": "keep", "status": "pending"}]}],
            )
        ],
    )
    return value.model_copy(update={"future_owner_field": {"keep": True}})


def delta(**changes: Any) -> FrontendUpdate:
    return FrontendUpdate(epoch="owner", sequence=1, changes=changes)


def test_scalar_update_does_not_serialize_or_refreeze_existing_jobs() -> None:
    store = FrontendStateStore(state())
    owned_job = store._state.jobs[0]
    with (
        patch.object(FrontendSessionState, "model_dump", side_effect=AssertionError("full dump")),
        patch.object(module, "_freeze_job", side_effect=AssertionError("retained job freeze")),
    ):
        result = store.apply_update(delta(activity_phase="responding"))
    assert store._state.jobs[0] is owned_job
    assert result.jobs[0] is not owned_job
    assert result.jobs[0].trajectory_length == 500
    assert result.jobs[0].trajectory[0]["delta"] == "retained"
    assert result.activity_phase == "responding"


def test_scalar_patch_preserves_model_validators_extras_and_public_isolation() -> None:
    store = FrontendStateStore(state())
    original = store.state
    incoming = delta(
        last_usage=Usage(input_tokens=12, output_tokens=3),
        future_new_field={"nested": ["new"]},
    )
    result = store.apply_update(incoming)
    assert result.last_usage is not None and result.last_usage.input_tokens == 12
    assert result.model_extra == {
        "future_owner_field": {"keep": True},
        "future_new_field": {"nested": ["new"]},
    }
    incoming.changes["future_new_field"]["nested"].append("caller mutation")
    assert result.model_extra is not None
    result.model_extra["future_new_field"]["nested"].append("public mutation")
    # Frozen models can still have their owning __dict__ edited by consumers;
    # canonical state must not share that mutable Pydantic shell.
    cast(dict[str, Any], result.jobs[0].__dict__)["label"] = "public mutation"
    current = store.state.model_extra
    assert current is not None and current["future_new_field"] == {"nested": ["new"]}
    assert store.state.jobs[0].label != "public mutation"
    assert original.sequence == 0 and store.state.sequence == 1


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("cumulative_cost", 42),
        ("model_label", "new-wire-label"),
        ("model_dump", {"nested": [{"value": "future"}]}),
        ("model_fields", {"nested": [{"value": "future"}]}),
        ("model_extra", {"nested": [{"value": "future"}]}),
        ("model_fields_set", {"nested": [{"value": "future"}]}),
    ],
)
def test_reserved_extra_names_preserve_wire_values_and_nested_isolation(
    name: str, value: Any
) -> None:
    # A newer runtime is allowed to introduce a field whose name happens to be
    # a property/method in this viewer. Attribute lookup is not value lookup.
    original_extra = {"nested": [{"value": "retained"}]}
    seed = state().model_copy(update={"existing_extra": original_extra})
    store = FrontendStateStore(seed)
    publications: list[FrontendUpdate] = []
    store.subscribe(publications.append)
    incoming = delta(**{name: copy.deepcopy(value)})
    result = store.apply_update(incoming)
    assert result.model_dump(mode="json")[name] == value
    assert result.model_extra is not None
    assert result.model_extra["existing_extra"] == {"nested": [{"value": "retained"}]}

    original_extra["nested"][0]["value"] = "original caller mutation"
    result.model_extra["existing_extra"]["nested"][0]["value"] = "public mutation"
    if isinstance(value, dict):
        incoming.changes[name]["nested"][0]["value"] = "incoming mutation"
        result.model_extra[name]["nested"][0]["value"] = "public collision mutation"
        publications[0].changes[name]["nested"][0]["value"] = "subscriber mutation"
    # Unchanged nested extras survive another scalar edge and neither their
    # original caller nor a public snapshot/subscriber owns canonical storage.
    current = store.apply_update(
        FrontendUpdate(epoch="owner", sequence=2, changes={"activity_phase": "responding"})
    ).model_dump(mode="json")
    assert current[name] == value
    assert current["existing_extra"] == {"nested": [{"value": "retained"}]}


def test_invalid_job_patch_does_not_commit_state_or_todo_watermarks() -> None:
    store = FrontendStateStore(state())
    original = store.state.model_dump(mode="json")
    watermarks = dict(store._todo_sequences)
    published: list[FrontendUpdate] = []
    store.subscribe(published.append)
    invalid = delta(jobs=[{"id": "child", "type": "task"}], context_tokens="invalid")
    invalid.job_todo_updates = {"child": [{"name": "New", "items": []}]}
    with pytest.raises(ValidationError):
        store.apply_update(invalid)
    assert store.state.model_dump(mode="json") == original
    assert store._todo_sequences == watermarks
    assert published == []
    invalid.changes["context_tokens"] = 123
    result = store.apply_update(invalid)
    assert result.sequence == 1
    plan = result.jobs[0].todos
    assert plan is not None and plan[0]["name"] == "New"
    assert store._todo_sequences["child"] == 1
    assert [update.sequence for update in published] == [1]


def test_explicit_jobs_clear_still_drops_rows_and_todo_watermarks() -> None:
    store = FrontendStateStore(state())
    store._todo_sequences["child"] = 0
    # Null is NOT a clear in the existing wire contract; only [] is.
    with pytest.raises(TypeError):
        store.apply_update(delta(jobs=None))
    assert len(store.state.jobs) == 1
    assert store._todo_sequences == {"child": 0}
    result = store.apply_update(delta(jobs=[]))
    assert list(result.jobs) == []
    assert store._todo_sequences == {}


def test_snapshot_job_get_uses_index_and_preserves_detached_owners() -> None:
    jobs = SnapshotJobs(FrontendStateStore(state()).state.jobs)

    class NoTraversal(list[JobState]):
        def __iter__(self):
            raise AssertionError("get() walked the roster")

    jobs._values = NoTraversal(jobs._values)
    first = jobs.get("child")
    assert first is not None
    cast(dict[str, Any], first.__dict__)["label"] = "changed outside"
    second = jobs.get("child")
    assert second is not None and second.label != first.label
    assert jobs.get("absent") is None
    jobs.replace([JobState(id="replacement", type="task")])
    assert jobs.get("child") is None
    assert jobs.get("replacement") is not None


def test_snapshot_duplicate_ids_keep_first_lookup_semantics() -> None:
    jobs = SnapshotJobs(
        [
            JobState(id="same", type="task", label="first"),
            JobState(id="same", type="task", label="second"),
        ]
    )
    first = jobs.get("same")
    assert first is not None and first.label == "first"
    assert [j.label for j in jobs.list()] == ["first", "second"]


def test_attached_scalar_callback_preserves_unrelated_collection_facades(tmp_path: Path) -> None:
    async def never():
        raise AssertionError("no takeover expected")

    viewer = AttachedSession(
        config_dir=tmp_path, session_id="benchmark-parent", takeover_factory=never
    )
    viewer._install_frontend(state())
    with (
        patch.object(viewer.jobs, "replace", wraps=viewer.jobs.replace) as jobs,
        patch.object(
            viewer._subagent_comms, "replace", wraps=viewer._subagent_comms.replace
        ) as comms,
        patch.object(
            viewer.wake_scheduler, "replace", wraps=viewer.wake_scheduler.replace
        ) as wakes,
        patch.object(viewer.mcp_manager, "replace", wraps=viewer.mcp_manager.replace) as mcp,
    ):
        viewer._on_frontend_update(delta(activity_phase="responding").model_dump(mode="json"))
        for spy in (jobs, comms, wakes, mcp):
            spy.assert_not_called()
        update = FrontendUpdate(epoch="owner", sequence=2, changes={"jobs": []})
        viewer._on_frontend_update(update.model_dump(mode="json"))
        jobs.assert_called_once()
        comms.assert_called_once()
        wakes.assert_not_called()
        mcp.assert_not_called()
        assert viewer.jobs.list() == []
        # Full snapshots always refresh, even after a sequence reset/reconnect.
        viewer._install_frontend(state())
        assert jobs.call_count == 2
        assert comms.call_count == 2
        wakes.assert_called_once()
        mcp.assert_called_once()
