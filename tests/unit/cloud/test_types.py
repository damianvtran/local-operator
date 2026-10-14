"""``PoolRequest`` and ``ComputeTask``: the two records, and their readers.

The three properties these pin are the ones §3.5 states so that two slices cannot
diverge: the keys are exactly the dataclass field names, ABSENT IS NOT A CLAIM
(a `version: 1` body from `mesh-compute-pool.md` §4.1 parses with the cloud delta
defaulted), and a value this build does not know is refused BY NAME rather than
coerced — because silently reading an unknown size class as the default `m`
provisions the wrong machine.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest

from local_operator.cloud.provider import CloudRefused, ComputeStatus
from local_operator.cloud.types import (
    COMPUTE_TASK_VERSION,
    POOL_REQUEST_VERSION,
    SIZES,
    ComputeTask,
    PoolRequest,
    member_lifecycle_for,
)

#: `mesh-compute-pool.md` §4.1's literal request body — the shape this slice's
#: request has to keep reading. Its `size_class` is the one key renamed (`size`),
#: and its `state`/`grant_id` moved to the task, so it is also the record that
#: proves those two are DROPPED rather than refused.
POOL_CONTRACT_BODY: dict[str, Any] = {
    "version": 1,
    "request_id": "req_8c31",
    "network_id": "n_4a1c",
    "size_class": "m",
    "max_hours": 4,
    "requested_at": 1789399800.0,
    "requested_by": "d_6c1f",
    "state": "queued",
    "grant_id": "g_71f0",
}


def test_pool_request_round_trips(request_factory: Callable[..., PoolRequest]) -> None:
    """Every field, through ``to_json``/``from_json``, equality preserved."""
    request = request_factory(
        version=POOL_REQUEST_VERSION,
        size="l",
        max_hours=4.0,
        tier="k8s",
        budget_usd=12.5,
        spot_ok=True,
        hosting="byo",
        max_sessions=3,
        include_local=True,
    )
    body = request.to_json()
    assert set(body) == set(PoolRequest.__dataclass_fields__), (
        "the JSON keys must be exactly the dataclass field names (§3.5), or the "
        "reader and the writer disagree about one field's spelling"
    )
    assert PoolRequest.from_json(body) == request


def test_compute_task_round_trips() -> None:
    """The mutable record too: it is the one that has to survive a restart."""
    task = ComputeTask(
        version=COMPUTE_TASK_VERSION,
        request_id="req_1a2b3c4d",
        tier="fargate",
        provider="fargate",
        provider_ref="arn:aws:ecs:ca-central-1:325492156725:task/lop-poc/abc",
        state="running",
        size="m",
        spot=True,
        created_at=1000.0,
        submitted_at=1001.0,
        started_at=1010.0,
        deadline_at=1010.0 + 2 * 3600,
        session_id="s_9f2a",
        budget_usd=5.0,
        repo_url="https://example.invalid/fixture.git",
        repo_sha="69db7e55",
        branch="lop/req_1a2b3c4d",
        artifacts={"results": "/tmp/results.tar.gz"},
        checkpoint={"branch": "lop/req_1a2b3c4d", "commit": "deadbeef"},
        events=[{"at": 1010.0, "phase": "running", "detail": {"message": "up"}}],
    )
    body = task.to_json()
    assert set(body) == set(ComputeTask.__dataclass_fields__)
    assert ComputeTask.from_json(body) == task


@pytest.mark.parametrize("size_class", ["m", "l"])
def test_a_version_1_request_from_the_pool_contract_still_parses(size_class: str) -> None:
    """A ``version: 1`` body parses, with the cloud delta defaulted.

    Parametrised over two size classes on purpose: the default is ``m``, so a
    reader that ignored ``size_class`` entirely would pass the ``m`` case by
    coincidence and ship a request that provisions a different machine than the
    one that asked for it.
    """
    request = PoolRequest.from_json({**POOL_CONTRACT_BODY, "size_class": size_class})
    assert request.version == 1
    assert request.request_id == "req_8c31"
    assert request.network_id == "n_4a1c"
    assert request.size == size_class
    assert request.max_hours == 4.0
    assert request.requested_at == 1789399800.0
    assert request.requested_by == "d_6c1f"
    # The cloud delta's defaults: absent is not a claim.
    assert request.tier == "fargate"
    assert request.budget_usd == 0.0
    assert request.spot_ok is False
    assert request.repo_url == ""
    assert request.repo_sha == ""
    assert request.prompt == ""
    assert request.hosting == "hosted"
    assert request.model == ""
    assert request.labels == {}
    assert request.max_sessions == 1
    assert request.include_local is False
    assert request.mock is False


def test_unknown_keys_are_dropped_not_refused() -> None:
    """A newer writer's extra keys must not stop an older reader mid-upgrade."""
    request = PoolRequest.from_json(
        {
            **POOL_CONTRACT_BODY,
            "version": POOL_REQUEST_VERSION,
            "a_key_from_the_future": {"nested": [1, 2]},
        }
    )
    assert request.request_id == "req_8c31"
    assert not hasattr(request, "a_key_from_the_future")
    # The two keys §3.3 moved ONTO the task are dropped here, not read and not
    # refused: the request is immutable intent and neither is intent.
    assert not hasattr(request, "state")
    assert not hasattr(request, "grant_id")


def test_a_wrong_typed_key_is_coerced_by_the_reader_that_owns_the_field() -> None:
    """§3.5: a key of the wrong type is coerced, not refused and not fatal."""
    request = PoolRequest.from_json(
        {
            "version": POOL_REQUEST_VERSION,
            "request_id": 17,
            "max_hours": "4",
            "budget_usd": "2.5",
            "spot_ok": "true",
            "mock": "1",
            "labels": {"purpose": 3},
        }
    )
    assert request.request_id == "17"
    assert request.max_hours == 4.0
    assert request.budget_usd == 2.5
    assert request.spot_ok is True
    assert request.mock is True
    assert request.labels == {"purpose": "3"}


def test_a_version_we_cannot_read_is_refused_by_name() -> None:
    """A shape older or newer than this build knows is unusable, not guessable."""
    with pytest.raises(CloudRefused) as refusal:
        PoolRequest.from_json({**POOL_CONTRACT_BODY, "version": 9})
    assert refusal.value.code == "version_unknown"
    assert "9" in refusal.value.sentence

    with pytest.raises(CloudRefused) as task_refusal:
        ComputeTask.from_json({"version": COMPUTE_TASK_VERSION + 1, "request_id": "req_1"})
    assert task_refusal.value.code == "version_unknown"

    # A version that is present but is not a number at all is refused too: the
    # alternative is reading it as "absent" and treating a foreign record as ours.
    with pytest.raises(CloudRefused):
        PoolRequest.from_json({**POOL_CONTRACT_BODY, "version": "two"})


def test_a_vocabulary_this_build_does_not_know_is_refused_not_defaulted() -> None:
    """An unknown size, tier, hosting mode or state fails closed, by name."""
    assert set(SIZES) == {"s", "m", "l"}
    for body, code in (
        ({"size": "xl"}, "size_unknown"),
        ({"tier": "lambda"}, "tier_unknown"),
        ({"hosting": "cloud"}, "hosting_unknown"),
    ):
        with pytest.raises(CloudRefused) as refusal:
            PoolRequest.from_json({**POOL_CONTRACT_BODY, "version": POOL_REQUEST_VERSION, **body})
        assert refusal.value.code == code, body

    with pytest.raises(CloudRefused) as state_refusal:
        ComputeTask.from_json({"version": COMPUTE_TASK_VERSION, "state": "sprinting"})
    assert state_refusal.value.code == "version_unknown"


def test_a_provider_reading_survives_a_round_trip() -> None:
    """``ComputeStatus`` crosses the seam, so it serialises like the other records."""
    status = ComputeStatus(
        provider="fargate",
        provider_ref="arn:aws:ecs:ca-central-1:325492156725:task/lop-poc/abc",
        phase="stopped",
        exit_code=1,
        stop_reason="failed",
        started_at=10.0,
        stopped_at=20.0,
        node="",
        detail={"last_status": "STOPPED"},
    )
    assert ComputeStatus.from_json(status.to_json()) == status
    assert status.stopped is True
    # Absent phase reads as the conservative "not observable yet", never as running.
    assert ComputeStatus.from_json({"provider_ref": "x"}).phase == "pending"


def test_member_lifecycle_for_refuses_a_state_it_does_not_know() -> None:
    """A silent ``None`` for an unknown state would read as "there was no member"."""
    with pytest.raises(ValueError):
        member_lifecycle_for("sprinting")
