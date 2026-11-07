"""What a mesh sends at first contact, and what it stops asking an owner-less peer.

WHY THIS FILE EXISTS (mesh perf matrix, "pairing vs live"). Two findings, each pinned
here and each failing on the build before the fix:

* PAIRING → FIRST DEFINITIONS SYNC waited for the definitions syncer's next timed tick
  (15 s): nothing told the cadence a member had appeared, so a freshly paired device
  got its agents, teams and MCP servers 0.8–13.6 s later depending only on the tick
  phase. A peer link coming up now nudges the syncer, which serves that member at once
  through the ordinary tick body.
* AN OWNER-LESS PEER WAS ASKED FOR ITS PLACEMENT ON EVERY TICK, FOREVER: a device that
  owns no credential answers ``net_broker`` with ``not_implemented``, and the credential
  step's answer is discarded by the tick, so nothing parked it. The step now parks a
  refusing peer per relay instance and build, and re-asks when either changes.

Real relays on loopback (``devices``/``_pair``, the share-push file's ``drive_mesh``);
only the syncer's debounce is driven with an injected clock.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.network import definitions, store
from local_operator.network.credentials import client as client_mod
from local_operator.network.credentials import sync as cred_sync
from tests.unit.network import conftest as net_fixtures
from tests.unit.network.test_credentials_owner import (  # noqa: F401 — fixtures by import
    idp,
    stub_provider,
)
from tests.unit.network.test_credentials_share_push import (  # noqa: F401 — fixtures by import
    _relink,
    _start_b,
    drive_mesh,
)
from tests.unit.network.test_definitions import _make_agent, _make_team
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixtures by import
    _pair,
    devices,
)

#: The real hook, captured at import: the directory's autouse pin
#: (``conftest._no_first_contact_nudge``) replaces it per test. ``getattr`` so a build
#: without the nudge still collects this file and fails its cells on their assertions.
_REAL_ON_LINK_UP = getattr(definitions, "_on_link_up", None)


@pytest.fixture()
def first_contact_nudge(monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-arm the first-contact nudge for the cell that pins it."""
    if _REAL_ON_LINK_UP is not None:
        monkeypatch.setattr(definitions, "_on_link_up", _REAL_ON_LINK_UP)


# ---------------------------------------------------------------------------
# First contact: a paired member is served without waiting for the timed tick
# ---------------------------------------------------------------------------


def test_a_freshly_paired_member_gets_definitions_without_waiting_for_the_tick(
    first_contact_nudge: None,
    devices: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A's syncer started with A (``devices``), so its first timed tick is ~15 s away
    when the pairing completes; the definitions must reach B well before that."""
    from local_operator.agents import AgentRegistry
    from local_operator.teams import TeamRegistry

    server_a, server_b, host, port = devices
    _make_agent(server_a.root, "reviewer")
    _make_team(server_a.root, "first-contact-team")
    record, _host, _port = _pair(devices, monkeypatch, role="drive")
    started = time.monotonic()
    host_b, port_b = server_b.bind()
    server_b.bind_control()
    with store.mutate(record.network_id, server_b.root) as copy:
        owner_row = copy.member(server_a.identity.device_id)
        assert owner_row is not None
        owner_row.endpoints = [f"{host}:{port}"]
        store.save(copy, server_b.root)
    with store.mutate(record.network_id, server_a.root) as copy:
        member_row = copy.member(server_b.identity.device_id)
        assert member_row is not None
        member_row.endpoints = [f"{host_b}:{port_b}"]
        store.save(copy, server_a.root)
    server_b.start()

    def _arrived() -> bool:
        return AgentRegistry(server_b.root).get_agent_by_name("reviewer") is not None and any(
            str(team.name) == "first-contact-team"
            for team in TeamRegistry(server_b.root).list_teams()
        )

    assert net_fixtures.wait_for(_arrived), "definitions never reached B"
    # THE BOUND IS THE ASSERTION here, the wait only a backstop: B's relay links within
    # ~1.1 s of starting and the push is one round trip (bench median 1.22 s), while the
    # pre-fix build could not deliver before A's first timed tick, ~15 s after A started.
    elapsed = time.monotonic() - started
    assert elapsed < 6.0, f"definitions waited for the timed tick ({elapsed:.1f} s)"


def test_first_contact_nudges_are_debounced_per_member(tmp_path: Path) -> None:
    """A flapping link must not wake the syncer on every re-link (one per minute per
    member), a different member is never held back by another's debounce, and a member
    already brought in sync is the timed cadence's, not the nudge's."""
    syncer = definitions.DefinitionsSyncer(SimpleNamespace(root=tmp_path))  # type: ignore[arg-type]
    peer, other, synced = "d_" + "1" * 32, "d_" + "2" * 32, "d_" + "4" * 32
    syncer._last_ok[synced] = True  # noqa: SLF001 — its last push succeeded
    assert syncer.nudge(synced, now=1000.0) is False
    assert syncer.nudge(peer, now=1000.0) is True
    assert syncer.nudge(peer, now=1010.0) is False
    assert syncer.nudge(other, now=1010.0) is True
    assert syncer.nudge(peer, now=1000.0 + definitions.NUDGE_MIN_INTERVAL_S) is True
    assert syncer._take_nudged() == frozenset({peer, other})  # noqa: SLF001
    assert syncer._take_nudged() == frozenset()  # noqa: SLF001


def test_a_nudged_member_is_served_by_the_running_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The nudge is served by ONE tick narrowed to that member, on the syncer's own
    thread, long before the timed tick (an hour away here) would have run."""
    monkeypatch.setattr(definitions, "_tick_seconds", lambda _root: 3600.0)
    syncer = definitions.DefinitionsSyncer(SimpleNamespace(root=tmp_path))  # type: ignore[arg-type]
    calls: list[Any] = []
    monkeypatch.setattr(syncer, "tick", lambda *, now=None, only=None: calls.append(only) or [])
    syncer.start()
    try:
        peer = "d_" + "3" * 32
        assert syncer.nudge(peer)
        assert net_fixtures.wait_for(lambda: calls == [frozenset({peer})]), calls
    finally:
        syncer.stop()
        syncer.join(5.0)
    assert not syncer.is_alive()


# ---------------------------------------------------------------------------
# An owner-less peer is parked, and re-armed by a new instance or build
# ---------------------------------------------------------------------------


def _count_pulls_to(monkeypatch: pytest.MonkeyPatch, asker: Any, owner: str) -> list[str]:
    """Record the reply code of every placement pull ``asker`` sends ``owner``."""
    seen: list[str] = []
    original = client_mod.MeshCredentialClient._ask_owner_directly  # noqa: SLF001

    def _spy(self: Any, target: str, frame: dict[str, Any], **kwargs: Any) -> Any:
        reply = original(self, target, frame, **kwargs)
        if (
            getattr(self, "_relay", None) is asker
            and target == owner
            and frame.get("kind") == "placement"
            and frame.get("want") == "pull"
        ):
            detail = (reply or {}).get("detail") if isinstance(reply, dict) else None
            code = (reply or {}).get("code") if isinstance(reply, dict) else None
            if not code and isinstance(detail, dict):
                code = detail.get("code")
            seen.append(str(code or ""))
        return reply

    monkeypatch.setattr(client_mod.MeshCredentialClient, "_ask_owner_directly", _spy)
    return seen


def test_an_owner_less_peer_is_asked_once_not_on_every_tick(
    drive_mesh: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A (admin, so not capability-blocked toward B) runs its credential step toward B
    20 times — 300 s of 15 s ticks. B owns nothing and answers ``not_implemented``;
    pre-fix that was 20 refusals, one per tick."""
    _start_b(drive_mesh)
    pulls = _count_pulls_to(monkeypatch, drive_mesh.a, drive_mesh.borrower)
    for _ in range(20):
        cred_sync.credentials_sync_step(drive_mesh.a, drive_mesh.borrower)
    assert pulls == ["not_implemented"], pulls


def test_a_peer_with_a_new_relay_instance_is_asked_again(
    drive_mesh: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An update restarts B's relay: the park belonged to the instance that refused, so
    the first step that sees the new instance asks again."""
    from local_operator.network import relay as relay_mod

    _start_b(drive_mesh)
    pulls = _count_pulls_to(monkeypatch, drive_mesh.a, drive_mesh.borrower)
    cred_sync.credentials_sync_step(drive_mesh.a, drive_mesh.borrower)
    cred_sync.credentials_sync_step(drive_mesh.a, drive_mesh.borrower)
    assert len(pulls) == 1, pulls

    drive_mesh.b.instance_id = relay_mod.mint_instance_id()
    _relink(drive_mesh)
    cred_sync.credentials_sync_step(drive_mesh.a, drive_mesh.borrower)
    assert len(pulls) == 2, pulls


def test_a_peer_whose_build_changed_is_asked_again(
    drive_mesh: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The park also names the BUILD that refused (``not_implemented`` is a fact about
    one build): a different build stamp behind the live link re-arms the pull."""
    _start_b(drive_mesh)
    pulls = _count_pulls_to(monkeypatch, drive_mesh.a, drive_mesh.borrower)
    cred_sync.credentials_sync_step(drive_mesh.a, drive_mesh.borrower)
    assert len(pulls) == 1, pulls
    live = [
        link
        for link in drive_mesh.a.links.values()
        if link.device_id == drive_mesh.borrower and link.alive
    ]
    assert live, "A has no live link to B"
    live[0].peer_build = {**dict(live[0].peer_build or {}), "version": "99.0.0"}
    cred_sync.credentials_sync_step(drive_mesh.a, drive_mesh.borrower)
    assert len(pulls) == 2, pulls
