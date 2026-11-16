"""A per-delta viewer edge clones only the fields its caller reads.

WHY. ``AttachedSession._on_frontend_update`` runs on the serve event loop once
per canonical delta of every attached conversation. It used to receive
``FrontendStateStore.apply_update``'s WHOLE caller-owned clone (every job shell,
``usage_components``, ``child_costs``...) and read about a dozen fields of it.
Measured 2026-11-14 on a store seeded from a live 346-job orchestrator
checkpoint (3.16 MB of state JSON), that clone was 13.6 ms p50 / 42.4 ms p95 CPU
of a 14.3 ms / 43.4 ms delta, and the serve loop-watchdog caught the event-loop
thread inside it during 23-37 s stalls while ~13 conversations streamed.

``PublicStateView`` now clones field by field on read. These tests pin both
halves: the caller does NOT pay the whole clone (structural call counts, per
AGENTS.md "Timing, flakes": no wall-clock bound), and the view keeps every
guarantee ``_public_state`` documents — one canonical reference, no shared
owning instance, stable objects per snapshot.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from local_operator.harness.types import Usage
from local_operator.session import frontend_state as module
from local_operator.session.attached import AttachedSession
from local_operator.session.frontend_state import (
    FrontendModelSpec,
    FrontendSessionState,
    FrontendStateStore,
    FrontendUpdate,
    JobState,
    PublicStateView,
)


def _state(jobs: int = 3) -> FrontendSessionState:
    value = FrontendSessionState(
        session_id="view-parent",
        epoch="owner",
        selected_model=FrontendModelSpec(provider="anthropic", model_id="m-1"),
        mcp_startup={"configured": ["a"], "failures": {"b": "boom"}},
        jobs=[
            JobState(
                id=f"child-{index}",
                type="task",
                usage=Usage(input_tokens=10, output_tokens=5),
                trajectory=[{"type": "message_update", "delta": "retained"}],
                latest_details={"nested": {"status": "running"}},
            )
            for index in range(jobs)
        ],
    )
    return value.model_copy(update={"future_owner_field": {"keep": True}})


def _delta(store: FrontendStateStore, **changes: Any) -> FrontendUpdate:
    return FrontendUpdate(epoch="owner", sequence=store._state.sequence + 1, changes=changes)


# -- the caller does not pay the whole-state clone ----------------------------


def _viewer(tmp_path: Path, jobs: int = 3) -> AttachedSession:
    async def never():
        raise AssertionError("no takeover expected")

    viewer = AttachedSession(config_dir=tmp_path, session_id="view-parent", takeover_factory=never)
    viewer._install_frontend(_state(jobs))
    return viewer


def test_attached_scalar_delta_clones_no_job_and_not_the_whole_state(tmp_path: Path) -> None:
    """Fails on the pre-view code: ``apply_update`` returned ``self.state``,
    which detached every job (``_public_job``) for a delta that moved none."""
    viewer = _viewer(tmp_path, jobs=5)
    store = viewer._frontend_store
    assert store is not None
    with (
        patch.object(module, "_public_job", wraps=module._public_job) as detach,
        patch.object(
            FrontendSessionState, "model_copy", autospec=True, side_effect=_real_model_copy
        ) as model_copy,
    ):
        viewer._on_frontend_update(_delta(store, activity_phase="responding").model_dump())
    assert detach.call_count == 0, "a scalar delta must not detach any job"
    deep = [call for call in model_copy.call_args_list if call.kwargs.get("deep")]
    assert deep == [], "a scalar delta must not deep-copy the whole state"
    # The facade still saw the delta.
    assert store.read_field("activity_phase") == "responding"


_ORIGINAL_MODEL_COPY = FrontendSessionState.model_copy


def _real_model_copy(self: Any, *args: Any, **kwargs: Any) -> Any:
    return _ORIGINAL_MODEL_COPY(self, *args, **kwargs)


def test_attached_roster_delta_detaches_only_the_moved_row(tmp_path: Path) -> None:
    """The jobs facades get the CANONICAL rows, so their identity memo works.

    Fed the view's detached ``jobs`` (fresh objects every delta) the memo in
    ``SnapshotJobs.replace`` never hit and a roster tick re-detached the whole
    roster; on the pre-view code it detached it twice (clone + facade)."""
    viewer = _viewer(tmp_path, jobs=4)
    store = viewer._frontend_store
    assert store is not None
    roster = [job.model_dump(mode="json") for job in store._state.jobs]
    # The first delta seeds the follower's wire-body memo and the facades'.
    viewer._on_frontend_update(_delta(store, jobs=roster).model_dump())
    roster[1]["status"] = "completed"
    with patch.object(module, "_public_job", wraps=module._public_job) as detach:
        viewer._on_frontend_update(_delta(store, jobs=roster).model_dump())
    assert detach.call_count == 1, "only the one moved row may be detached"
    assert [job.status for job in viewer.jobs.list()][1] == "completed"
    assert viewer._subagent_comms._nodes["child-1"].status == "completed"


# -- the view keeps _public_state's guarantees --------------------------------


def test_view_is_one_canonical_reference_unmoved_by_later_publishes() -> None:
    store = FrontendStateStore(_state())
    view = store.apply_update_view(_delta(store, conversation_title="first"))
    store.apply_update(
        _delta(
            store,
            conversation_title="second",
            selected_model={"provider": "openai", "model_id": "m-2"},
            jobs=[],
        )
    )
    # Every field read AFTER the later publish still describes the state the
    # view captured — never a mix of two published states.
    assert view.sequence == 1
    assert view.conversation_title == "first"
    assert view.selected_model.model_id == "m-1"
    assert [job.id for job in view.jobs] == ["child-0", "child-1", "child-2"]
    assert store.read_field("conversation_title") == "second"


def test_view_never_shares_an_owning_instance_with_the_store() -> None:
    store = FrontendStateStore(_state())
    view = store.apply_update_view(_delta(store, activity_phase="responding"))
    canonical = store._state
    assert view.selected_model is not canonical.selected_model
    assert view.mcp_startup is not canonical.mcp_startup
    assert view.jobs[0] is not canonical.jobs[0]

    view.selected_model.model_id = "mutated"
    view.mcp_startup["failures"]["b"] = "mutated"
    view.future_owner_field["keep"] = False

    after = store.state
    assert after.selected_model is not None and after.selected_model.model_id == "m-1"
    assert after.mcp_startup == {"configured": ["a"], "failures": {"b": "boom"}}
    assert after.model_extra is not None and after.model_extra["future_owner_field"] == {
        "keep": True
    }


def test_view_clones_each_field_once_and_shares_only_immutable_scalars() -> None:
    store = FrontendStateStore(_state())
    view = store.apply_update_view(_delta(store, conversation_title="t"))
    assert view.selected_model is view.selected_model
    assert view.jobs is view.jobs
    for name in module._SHAREABLE_STATE_FIELDS:
        assert getattr(view, name) is getattr(store._state, name)


@pytest.mark.parametrize("name", ["model_dump", "model_copy", "model_label", "not_a_field"])
def test_view_refuses_methods_properties_and_unknown_names(name: str) -> None:
    store = FrontendStateStore(_state())
    view = store.apply_update_view(_delta(store, activity_phase="responding"))
    with pytest.raises(AttributeError):
        getattr(view, name)


def test_view_matches_the_whole_clone_field_for_field() -> None:
    whole = FrontendStateStore(_state())
    lazy = FrontendStateStore(_state())
    changes: list[dict[str, Any]] = [
        {"activity_phase": "responding", "streaming": True},
        {"selected_model": {"provider": "openai", "model_id": "m-2"}},
        {"jobs": [{"id": "child-9", "type": "task", "status": "running"}]},
        {"conversation_title": "done", "streaming": False},
    ]
    for change in changes:
        expected = whole.apply_update(_delta(whole, **change))
        view = lazy.apply_update_view(_delta(lazy, **change))
        assert isinstance(view, PublicStateView)
        assert view.materialize().model_dump(mode="json") == expected.model_dump(mode="json")
        for field in FrontendSessionState.model_fields:
            value = getattr(view, field)
            other = getattr(expected, field)
            if field == "jobs":
                assert [job.model_dump(mode="json") for job in value] == [
                    job.model_dump(mode="json") for job in other
                ]
            else:
                assert value == other, field


def test_view_path_refuses_an_out_of_order_delta_without_committing() -> None:
    store = FrontendStateStore(_state())
    published: list[FrontendUpdate] = []
    store.subscribe(published.append)
    with pytest.raises(ValueError, match="next state sequence"):
        store.apply_update_view(FrontendUpdate(epoch="owner", sequence=5, changes={}))
    assert store._state.sequence == 0
    assert published == []
    view = store.apply_update_view(_delta(store, activity_phase="responding"))
    assert view.sequence == 1
    assert [update.sequence for update in published] == [1]
