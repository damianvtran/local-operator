"""Cleanup-ref determinism and the rescue-from-descriptor-alone guarantee.

This is the C2 guarantee stated as tests: the same episode ID always produces
the same refs, client token, and cleanup_plan_id; and a ``CleanupRefs``
reconstructed ONLY from a round-tripped ``RescueDescriptor`` equals the
original. A rescue worker has never run ``prepare``, so teardown must be
possible from the descriptor's resource_refs alone.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from lop_osworld_v2_adapter import cleanup

from local_operator.evaluation.adapters.api import (
    ADAPTER_SCHEMA_VERSION,
    AdapterCapabilities,
    AdapterMetadata,
    AdapterSelector,
    Handshake,
    PythonRuntime,
    RescueDescriptor,
)


def _refs(episode_id: str) -> cleanup.CleanupRefs:
    return cleanup.CleanupRefs.mint(episode_id)


def test_refs_are_deterministic_per_episode() -> None:
    a = _refs("ep-1")
    b = _refs("ep-1")
    assert a == b
    assert a.client_token == b.client_token


def test_refs_differ_per_episode() -> None:
    assert _refs("ep-1") != _refs("ep-2")
    assert _refs("ep-1").client_token != _refs("ep-2").client_token


def test_cleanup_plan_id_is_deterministic() -> None:
    a = cleanup.build_cleanup_plan("ep-1", _refs("ep-1"))
    b = cleanup.build_cleanup_plan("ep-1", _refs("ep-1"))
    assert a.cleanup_plan_id == b.cleanup_plan_id


def test_plan_carries_the_three_symbolic_actions() -> None:
    plan = cleanup.build_cleanup_plan("ep-1", _refs("ep-1"))
    by_kind = {a.kind: a.resource_ref for a in plan.actions}
    assert by_kind == {
        "release_instance": "lop-ep-ep-1",
        "revoke_lease": "lop-ttl-ep-1",
        "close_session": "ep-1",
    }


def test_no_delete_volume_action() -> None:
    # The root volume dies with the instance (DeleteOnTermination=true), so a
    # delete_volume action could only ever report not_needed — pure noise.
    plan = cleanup.build_cleanup_plan("ep-1", _refs("ep-1"))
    assert all(a.kind != "delete_volume" for a in plan.actions)


def test_refs_round_trip_through_a_descriptor() -> None:
    """The rescue guarantee: refs rebuilt from a persisted descriptor alone."""
    episode_id = "ep-rescue"
    original = _refs(episode_id)
    plan = cleanup.build_cleanup_plan(episode_id, original)

    # The Handshake validator requires the runtime executable to equal the
    # selector's python_executable; use the running interpreter for both.
    import sys

    executable = str(Path(sys.executable).resolve())
    selector = AdapterSelector(
        schema_version=ADAPTER_SCHEMA_VERSION,
        adapter_id="osworld-v2",
        distribution="lop-osworld-v2-adapter",
        version="0.1.0",
        entry_point="lop_osworld_v2_adapter:create",
        package_digest="a" * 64,
        release_digest="b" * 64,
        python_executable=executable,
        workspace="/tmp/workspace",
        workspace_digest="c" * 64,
        route_capability="computer",
    )
    handshake = Handshake(
        selector=selector,
        metadata=AdapterMetadata(
            adapter_id="osworld-v2",
            distribution="lop-osworld-v2-adapter",
            version="0.1.0",
            entry_point="lop_osworld_v2_adapter:create",
            package_digest="a" * 64,
            release_digest="b" * 64,
            schema_version=ADAPTER_SCHEMA_VERSION,
            capabilities=AdapterCapabilities(routes=("computer",), ask_user=True, scoring=True),
        ),
        python=PythonRuntime.current(),
        workspace_digest="c" * 64,
        selected_route="computer",
    )
    descriptor = RescueDescriptor(
        schema_version=ADAPTER_SCHEMA_VERSION,
        selector=selector,
        handshake=handshake,
        episode_id=episode_id,
        cleanup_plan=plan,
        secret_refs=(),
        infra_values=(),
        artifact_root="/tmp/artifacts",
    )

    # Round-trip the descriptor through its canonical wire form so the test
    # proves the refs survive persistence, not just in-memory equality.
    reloaded = RescueDescriptor.from_canonical_json(descriptor.to_canonical_json())
    rebuilt = cleanup.CleanupRefs.from_descriptor_actions(reloaded.cleanup_plan.actions)
    assert rebuilt == original
    assert rebuilt.client_token == original.client_token


def test_from_descriptor_actions_rejects_an_incomplete_plan() -> None:
    plan = cleanup.build_cleanup_plan("ep-1", _refs("ep-1"))
    incomplete = [a for a in plan.actions if a.kind != "revoke_lease"]
    with pytest.raises(ValueError):
        cleanup.CleanupRefs.from_descriptor_actions(tuple(incomplete))
