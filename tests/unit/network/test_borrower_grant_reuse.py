"""The borrower's hot read: a lent provider grant reused without dialling the owner.

WHY THIS FILE EXISTS. ``MeshCredentialClient._serve_cached`` lets a covered provider
key skip the owner round trip while the owner's announced lifecycle for that key is
unchanged and the grant is younger than ``grant_ttl_s/2`` (operator-approved C5
window, 2026-11-07; design ``mesh-credentials.md`` §3.5 as-built note, §3.7).
These cells are the proposal's acceptance list. Each one pins a fence whose
absence would let a revoked, withdrawn, logged-out, rotated, cooling or
re-steered credential keep serving from memory; the coverage cells pin that an
owner without lifecycle evidence keeps today's dial-per-call.

The relay hop is faked with a counting ``control_request``: the property under
test is "how many times does the borrower ask", and every owner-side behaviour
has its own suite (``test_credentials_owner``).
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from local_operator.network.credentials import client as client_mod
from local_operator.network.credentials import lifecycle
from local_operator.network.credentials.client import MeshCredentialClient, key_for
from local_operator.network.credentials.github import GITHUB_KEY, GITHUB_KIND
from local_operator.network.credentials.held_policy import HeldPolicyStore
from local_operator.network.credentials.state import PlacementState
from local_operator.network.credentials.types import BrokerError, CredentialRef, Grant
from tests.unit.network.test_credentials_placement import (
    NETWORK,
    OWNER,
    _bridge_before_placement,
    placement_mod,
)

SESSION = "synthetic-session"
SCOPE = "native:openai:c3ludGhldGlj"


def _grant(token: str, credential_id: int = 1, provider: str = "openai") -> Grant:
    return Grant(
        access_token=token,
        kind="bearer",
        token_expires_at_ms=0,
        grant_expires_at_ms=int((time.time() + 900) * 1000),
        credential_ref=CredentialRef(
            owner_device=OWNER,
            owner_device_name="owner",
            provider=provider,
            kind="oauth",
            credential_id=credential_id,
        ),
        served_by=OWNER,
    )


def _client(root: Path, *keys: str) -> MeshCredentialClient:
    """A joined recipient holding ``keys`` (default ``openai``), placement ON DISK.

    On disk so a revoke written by another process reaches this client the way it
    does in production: through ``placement()``'s stat check, not a shared object.
    """
    _bridge_before_placement(root, "recipient")
    document = placement_mod.PlacementDocument(NETWORK, root=root, written_by=OWNER)
    for key in keys or ("openai",):
        document.declare(key, owner_device=OWNER, provider=key, by=OWNER)
        # A forge loan is device-scoped by rule (``placement.grant`` refuses session).
        scope = "device" if key == GITHUB_KEY else "session"
        document.grant(key, "recipient", scope=scope, by=OWNER)
    document.save()
    return MeshCredentialClient(
        root=root,
        self_device="recipient",
        network_id=NETWORK,
        placement=placement_mod.PlacementDocument.load(NETWORK, root, self_device="recipient"),
    )


def _observe(
    root: Path, *, key: str = "openai", generation: int = 1, state: str = "available"
) -> None:
    """Apply an owner lifecycle announcement as ``member_lifecycle_announce`` would."""
    result = lifecycle.LifecycleStore(NETWORK, root).observe(
        lifecycle.Observation(
            owner=OWNER,
            key=key,
            account_scope_ref=SCOPE,
            login_chain_revision=1,
            material_generation=generation,
            state=state,
        )
    )
    assert result.reason in (lifecycle.OBSERVED, lifecycle.OBSERVED_REPLAY), result


class _Owner:
    """A counting leg-1 relay: every ``credential_grant`` is one owner dial."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, offline: bool = False) -> None:
        self.grants: list[dict[str, Any]] = []
        self.reports: list[dict[str, Any]] = []
        self.offline = offline
        self.refuse = ""
        monkeypatch.setattr("local_operator.network.store.find_own_relay", lambda _root: {})
        monkeypatch.setattr("local_operator.network.relay.control_request", self.control)

    def control(self, _record: Any, op: str, **fields: Any) -> Any:
        if op == "credential_report":
            self.reports.append(fields)
            return {"op": "ack"}
        self.grants.append(fields)
        if self.offline or self.refuse:
            code = self.refuse or "owner_offline"
            return {"op": "ack", "detail": {"kind": "error", "code": code}}
        excluded = set(fields.get("exclude_credential_ids") or ())
        credential_id = 2 if 1 in excluded else 1
        grant = _grant(f"synthetic-bearer-{len(self.grants)}", credential_id)
        return {"op": "ack", "detail": grant.to_detail()}


class _Clock:
    """``client_mod.time`` with three movable offsets; everything else is real ``time``.

    ``offset`` is time spent AWAKE: every clock moves. ``asleep`` is time the machine
    spent sleeping: ``time.monotonic()`` (macOS ``mach_absolute_time``) does NOT see
    it, while ``clock_gettime`` (``CLOCK_MONOTONIC`` on Darwin, ``CLOCK_BOOTTIME`` on
    Linux) and wall time do. ``wall_step`` moves only ``time.time()`` (an NTP or
    manual step). Only ``client.py`` sees this shim; other modules keep real time.
    """

    def __init__(self) -> None:
        self.offset = 0.0
        self.asleep = 0.0
        self.wall_step = 0.0

    def __getattr__(self, name: str) -> Any:
        return getattr(time, name)

    def time(self) -> float:
        return time.time() + self.offset + self.asleep + self.wall_step

    def monotonic(self) -> float:
        return time.monotonic() + self.offset

    def clock_gettime(self, clock_id: int) -> float:
        return time.clock_gettime(clock_id) + self.offset + self.asleep


def _resolve(client: MeshCredentialClient, key: str = "openai", **kw: Any) -> Any:
    return client.request_grant_sync(key, session_id=SESSION, provider=key, **kw)


def _warm(client: MeshCredentialClient, owner: _Owner, **kw: Any) -> Grant:
    """Two resolves, ONE dial: the reuse that the fence under test must then break.

    Every invalidation cell starts here so it fails on a build without the hot
    read (which dials twice) and passes only where the fence, not the absence of
    a cache, is what makes the next resolve dial.
    """
    first = _resolve(client, **kw)
    assert isinstance(first, Grant) and _resolve(client, **kw) is first
    assert len(owner.grants) == 1, owner.grants
    return first


# 1 -------------------------------------------------------------------------


def test_covered_provider_dials_the_owner_once_for_sequential_resolves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client(tmp_path)
    _observe(tmp_path)
    owner = _Owner(monkeypatch)
    first = _resolve(client)
    second = _resolve(client)
    assert isinstance(first, Grant) and second is first
    assert len(owner.grants) == 1
    client.close()


# 2 -------------------------------------------------------------------------


def test_a_lifecycle_generation_bump_between_resolves_dials_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client(tmp_path)
    _observe(tmp_path)
    owner = _Owner(monkeypatch)
    first = _warm(client, owner)
    _observe(tmp_path, generation=2)  # owner rotated / re-logged in
    second = _resolve(client)
    assert isinstance(second, Grant) and second.access_token != first.access_token
    assert len(owner.grants) == 2
    client.close()


def test_the_fence_is_taken_before_the_dial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A logout announced WHILE the owner answers must not be masked by that answer."""
    client = _client(tmp_path)
    _observe(tmp_path)
    owner = _Owner(monkeypatch)
    real = owner.control

    def racing(record: Any, op: str, **fields: Any) -> Any:
        reply = real(record, op, **fields)
        if len(owner.grants) == 1:
            _observe(tmp_path, generation=2)
        return reply

    monkeypatch.setattr("local_operator.network.relay.control_request", racing)
    _resolve(client)
    second = _resolve(client)  # the fence stored with the first grant predates the bump
    assert len(owner.grants) == 2
    assert _resolve(client) is second and len(owner.grants) == 2  # the new fence is reused
    client.close()


# 3 -------------------------------------------------------------------------


@pytest.mark.parametrize("state", ["absent", "withdrawn"])
def test_absent_or_withdrawn_observation_is_never_served_from_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    client = _client(tmp_path)
    _observe(tmp_path)
    owner = _Owner(monkeypatch)
    _warm(client, owner)
    # A logout moves the generation; a sticky withdraw restricts at the same one.
    _observe(tmp_path, generation=2 if state == "absent" else 1, state=state)
    second = _resolve(client)
    assert len(owner.grants) == 2
    if state == "withdrawn":
        assert isinstance(second, BrokerError) and second.code == "consent_withdrawn"
        assert client.grants.peek("openai", SESSION) is None
    client.close()


# 4 -------------------------------------------------------------------------


def test_a_revoke_from_another_process_clears_the_cache_and_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client(tmp_path)
    _observe(tmp_path)
    owner = _Owner(monkeypatch)
    _warm(client, owner)
    # The borrower's placement pull, a different object/process, writes the revoke.
    other = placement_mod.PlacementDocument.load(NETWORK, tmp_path, self_device="recipient")
    assert other is not None
    other.revoke("openai", "recipient", by=OWNER)
    other.save()
    refused = _resolve(client)
    assert isinstance(refused, BrokerError) and refused.code == "not_a_holder"
    assert client.grants.peek("openai", SESSION) is None
    assert len(owner.grants) == 1
    client.close()


# 5 -------------------------------------------------------------------------


def test_lost_membership_drops_borrowed_grants_without_serving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client(tmp_path)
    _observe(tmp_path)
    owner = _Owner(monkeypatch)
    _warm(client, owner)
    monkeypatch.setattr(MeshCredentialClient, "_self_membership_active", lambda _self: False)
    refused = _resolve(client)
    assert isinstance(refused, BrokerError) and refused.code == "not_a_holder"
    assert client.grants.keys() == []
    assert len(owner.grants) == 1  # answered locally, as both other paths would
    client.close()


# 6 -------------------------------------------------------------------------


def test_a_held_policy_stamp_change_is_not_served(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client(tmp_path)
    _observe(tmp_path)
    owner = _Owner(monkeypatch)
    _warm(client, owner)
    policy = HeldPolicyStore(NETWORK, "recipient", tmp_path)
    with policy.transaction() as document:  # another session's persisted deny
        policy.deny(document, OWNER, "openai", "consent_withdrawn")
    _resolve(client)
    assert len(owner.grants) == 2
    client.close()


# 7 is the parametrised ``test_native_hot_access``
# ``test_live_grant_and_old_login_refusal_do_not_short_circuit_owner``.

# 8 -------------------------------------------------------------------------


def test_a_grant_older_than_half_the_ttl_dials_and_holds_if_the_owner_is_gone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.network.credentials import grant_ttl_s

    clock = _Clock()
    monkeypatch.setattr(client_mod, "time", clock)
    client = _client(tmp_path)
    _observe(tmp_path)
    owner = _Owner(monkeypatch)
    first = _resolve(client)
    clock.offset = grant_ttl_s(tmp_path) / 2 - 1
    assert _resolve(client) is first and len(owner.grants) == 1
    clock.offset = grant_ttl_s(tmp_path) / 2
    owner.offline = True
    held = _resolve(client)
    assert len(owner.grants) == 2
    assert isinstance(held, Grant) and held.held_offline
    assert held.access_token == first.access_token
    client.close()


@pytest.mark.parametrize("moved", ["asleep", "wall_step_forward", "wall_step_back"])
def test_reuse_age_counts_sleep_and_wall_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, moved: str
) -> None:
    """R1-B1 / QA Q1: a borrower that slept through a revoke must not wake up reusing.

    ``time.monotonic()`` stands still here, as it does across macOS sleep. The
    reuse age must still reach ``grant_ttl_s/2`` through the sleep-inclusive clock,
    and the wall-clock age is checked too: either clock past the bound, or a wall
    clock stepped behind the stamp, ends reuse. The owner's update is "lost"
    (nothing changes on disk), so only the age can stop the serve.
    """
    from local_operator.network.credentials import grant_ttl_s

    clock = _Clock()
    monkeypatch.setattr(client_mod, "time", clock)
    client = _client(tmp_path)
    _observe(tmp_path)
    owner = _Owner(monkeypatch)
    _warm(client, owner)
    half = grant_ttl_s(tmp_path) / 2
    if moved == "asleep":
        clock.asleep = half
    elif moved == "wall_step_forward":
        clock.wall_step = half
    else:
        clock.wall_step = -60.0
    _resolve(client)
    assert len(owner.grants) == 2, f"{moved}: a stale slot was reused without a dial"
    client.close()


def test_a_lifecycle_bump_with_the_owner_refusing_drops_the_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1-m2: the other half of keeping the slot on a fence change.

    The slot survives a fence change only so ``_held_or`` can hold it while the
    owner is UNREACHABLE. A reachable owner that refuses (logged out) must leave
    nothing behind to serve, on this call or the next.
    """
    client = _client(tmp_path)
    _observe(tmp_path)
    owner = _Owner(monkeypatch)
    _warm(client, owner)
    _observe(tmp_path, generation=2, state="absent")
    owner.refuse = "no_local_credential"
    refused = _resolve(client)
    assert isinstance(refused, BrokerError) and refused.code == "no_local_credential"
    assert client.grants.peek("openai", SESSION) is None
    assert not isinstance(_resolve(client), Grant)
    assert len(owner.grants) == 3
    client.close()


def test_a_lifecycle_bump_while_the_owner_is_unreachable_still_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fence makes the next resolve DIAL; it does not shrink the offline hold.

    Measured on the two-relay rig: the owner's own token refresh after the first
    lend bumps ``material_generation``, and that announce can land while the owner
    is frozen. Dropping the slot there turned a ``held_offline`` serve (today's
    contract) into a refusal; the dial must reach ``_held_or`` with the slot intact.
    """
    client = _client(tmp_path)
    _observe(tmp_path)
    owner = _Owner(monkeypatch)
    first = _warm(client, owner)
    _observe(tmp_path, generation=2)
    owner.offline = True
    held = _resolve(client)
    assert len(owner.grants) == 2
    assert isinstance(held, Grant) and held.held_offline
    assert held.access_token == first.access_token
    client.close()


# 9 -------------------------------------------------------------------------


def test_force_refresh_always_dials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(tmp_path)
    _observe(tmp_path)
    owner = _Owner(monkeypatch)
    _warm(client, owner)
    forced = _resolve(client, force_refresh=True)
    assert isinstance(forced, Grant) and len(owner.grants) == 2
    assert owner.grants[1]["force_refresh"] is True
    client.close()


# 10 ------------------------------------------------------------------------


def test_a_lent_row_cooling_in_another_process_dials_with_exclude(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client(tmp_path)
    _observe(tmp_path)
    owner = _Owner(monkeypatch)
    _warm(client, owner, model_id="gpt-6")
    sibling = PlacementState.load(NETWORK, tmp_path)  # another session's process
    sibling.cool_down(
        "openai",
        owner_device=OWNER,
        credential_id=1,
        block_scope="",
        until=time.time() + 600,
        label="a",
    )
    sibling.save()
    second = _resolve(client, model_id="gpt-6")
    assert len(owner.grants) == 2
    assert owner.grants[1]["exclude_credential_ids"] == [1]
    assert isinstance(second, Grant) and second.credential_ref.credential_id == 2
    client.close()


# 11, 12 --------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["unauthorized", "invalid", "quota", "unavailable"])
def test_any_failure_report_makes_the_next_resolve_dial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    client = _client(tmp_path)
    _observe(tmp_path)
    owner = _Owner(monkeypatch)
    first = _warm(client, owner, model_id="gpt-6")
    client.report_sync("openai", kind=kind, session_id=SESSION, model_id="gpt-6")
    assert owner.reports and owner.reports[0]["credential_id"] == 1
    second = _resolve(client, model_id="gpt-6")
    assert len(owner.grants) == 2
    assert isinstance(second, Grant) and second.access_token != first.access_token
    if kind == "quota":
        # The 429 cooled the lent row: the dial hides it and lands on another account.
        assert owner.grants[1]["exclude_credential_ids"] == [1]
        assert second.credential_ref.credential_id == 2
    client.close()


# 13 ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "first,second",
    [
        ({}, {"affinity": (1, 100)}),
        ({"affinity": (1, 100)}, {"affinity": (2, 200)}),
        ({}, {"demote_credential_ids": [1]}),
    ],
)
def test_changed_selection_inputs_dial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, first: Any, second: Any
) -> None:
    client = _client(tmp_path)
    _observe(tmp_path)
    owner = _Owner(monkeypatch)
    _resolve(client, **first)
    _resolve(client, **first)
    assert len(owner.grants) == 1
    _resolve(client, **second)
    assert len(owner.grants) == 2
    client.close()


# 14 ------------------------------------------------------------------------


MCP_KEY = key_for(mcp_url="https://mcp.example.test/sse")


def test_mcp_key_frames_are_unchanged_by_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Coverage changes nothing for an MCP key: the same frames, a dial per call."""
    key = MCP_KEY
    frames = []
    for covered in (False, True):
        root = tmp_path / str(covered)
        root.mkdir()
        client = _client(root, key)
        if covered:
            _observe(root, key=key)
        owner = _Owner(monkeypatch)
        outcomes = [type(_resolve(client, key)).__name__ for _ in range(3)]
        frames.append((owner.grants, outcomes))
        client.close()
    assert frames[0] == frames[1]
    assert len(frames[1][0]) == 3


def test_device_bound_key_is_never_served_from_the_hot_read(tmp_path: Path) -> None:
    """``placement`` refuses to lend ``kimi`` at all; the hot read refuses it too.

    Belt and braces for an alias or an older document that slips past placement:
    even a fenced slot for a device-bound key is never handed out without a dial.
    """
    client = _client(tmp_path)
    _observe(tmp_path, key="kimi")
    fence = client._lifecycle_fence("kimi")
    client.grants.put(
        "kimi",
        SESSION,
        _grant("synthetic-kimi", provider="kimi"),
        lifecycle_fence=fence or frozenset({("s", (1, 1), "available")}),
        selection={},
    )
    assert client._serve_cached("kimi", SESSION, model_id="", selection={}) is None
    client.close()


def test_github_keeps_its_own_cache_window_regardless_of_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client(tmp_path, GITHUB_KEY)
    owner = _Owner(monkeypatch)
    grant = _grant("synthetic-forge", provider=GITHUB_KEY)
    grant.credential_ref = replace(grant.credential_ref, kind=GITHUB_KIND)
    client.grants.put(GITHUB_KEY, SESSION, grant)  # no lifecycle fence at all
    assert _resolve(client, GITHUB_KEY) is grant
    assert owner.grants == []
    client.close()
