"""``_ensure_link`` is single-flight per peer.

THE RACE (#3849's gate, ``test_two_config_roots_the_node_holds_the_share_before_the_run_returns``):
the definitions syncer and the CLI's ``definitions_sync`` both found no live link
to the node and BOTH dialled it. The node's duplicate-identity rule keeps the
NEWEST link and closes the other (``"replaced"``), so whichever push held the
loser sent ``state`` into a closing link and reported ``no_answer`` for a peer
that was up. The suite lane measured 5 of 6 concurrent pushes with no link.

The first cell is that race on two real relays, made deterministic by holding
both dials at a barrier so they overlap; the rest pin the single-flight contract
on a relay whose dial is a fake, so the counts are exact.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.network import definitions, relay
from tests.unit.network import conftest as net_fixtures
from tests.unit.network.test_provision_step import two_roots  # noqa: F401 — the fixture
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — two_roots reaches for it
    devices,
)

#: How long a dial is held so concurrent callers are provably inside it at once.
OVERLAP_S = 1.0


def test_concurrent_pushes_with_no_link_share_one_dial_and_both_land(
    two_roots: Any,  # noqa: F811 — the imported fixture
) -> None:
    """Two pushes to a peer with no live link: one dial, both answered.

    RED before the fix: both pushes dialled (held at the barrier so they overlap,
    as the syncer and the CLI did on the gate), the node replaced one link with
    the other, and the push on the replaced link answered ``no_answer``.
    """
    a = two_roots.a
    node = two_roots.node_id
    definitions.stop_syncer(a)  # the CLI push and a cadence push, as two callers
    assert net_fixtures.wait_for(
        lambda: a._link_for(node) is None  # noqa: SLF001
    ), "the pairing's link never closed, so there is no race to have"

    real_dial = a.dial
    dials: list[float] = []
    overlap = threading.Barrier(2)

    def _held_dial(*args: Any, **kwargs: Any) -> Any:
        dials.append(time.monotonic())
        try:
            overlap.wait(OVERLAP_S)  # both dials in flight together, if both dial
        except threading.BrokenBarrierError:
            pass
        return real_dial(*args, **kwargs)

    a.dial = _held_dial  # type: ignore[method-assign]
    results: list[dict[str, Any]] = []

    def _push() -> None:
        results.append(definitions.push_to_peer(a, node))

    threads = [threading.Thread(target=_push) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)

    assert len(results) == 2, results
    assert [row.get("code") for row in results if not row.get("ok")] == [], results
    assert all(row.get("ok") for row in results), results
    assert len(dials) == 1, f"{len(dials)} concurrent dials to one peer"


# A relay whose dial is a fake ------------------------------------------------


def _fake_relay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, members: list[str], dial: Any
) -> Any:
    """``object.__new__``, as the endpoint-probe cells build one: no ``__init__``."""
    record = SimpleNamespace(
        network_id="n_synthetic",
        epoch=1,
        member=lambda device: (
            SimpleNamespace(active=True, device_id=device) if device in members else None
        ),
    )
    monkeypatch.setattr(relay.store, "list_networks", lambda root: [record])
    monkeypatch.setattr(
        relay,
        "probe_candidates",
        lambda candidates, **kwargs: relay.CandidateProbe(
            object(), candidates[0], [], True, ""  # type: ignore[arg-type]
        ),
    )
    server = object.__new__(relay.RelayServer)
    server.root = tmp_path
    object.__setattr__(
        server,
        "settings",
        SimpleNamespace(handshake_timeout_s=5, port=4097, listen_address="0.0.0.0"),
    )
    monkeypatch.setattr(server, "_link_for", lambda device: None)
    monkeypatch.setattr(server, "_redial_candidates", lambda member: ["127.0.0.1:9"])
    monkeypatch.setattr(server, "dial", dial)
    return server


def _together(calls: list[Any]) -> list[Any]:
    """Run every call on its own thread, all at once; the results in call order."""
    out: list[Any] = [None] * len(calls)
    start = threading.Barrier(len(calls))

    def _run(index: int, call: Any) -> None:
        start.wait(5)
        try:
            out[index] = call()
        except BaseException as exc:  # noqa: BLE001 — surfaced as the result
            out[index] = exc

    threads = [threading.Thread(target=_run, args=(i, c)) for i, c in enumerate(calls)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert not any(thread.is_alive() for thread in threads), "a caller hung"
    return out


def test_n_callers_for_one_peer_share_one_dial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dials: list[str] = []
    link = SimpleNamespace(name="the-one-link")

    def _dial(network_id: str, **kwargs: Any) -> Any:
        dials.append(str(kwargs.get("expected_device")))
        time.sleep(0.3)  # long enough that every caller arrives while it runs
        return link, ""

    server = _fake_relay(tmp_path, monkeypatch, ["d_peer"], _dial)

    results = _together([lambda: server._ensure_link("d_peer") for _ in range(6)])

    assert dials == ["d_peer"], f"{len(dials)} dials for 6 same-peer callers"
    assert all(result is link for result in results), results


def test_callers_for_different_peers_are_never_serialised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """N peers, N dials, and all N in flight at the same moment."""
    peers = [f"d_peer{i}" for i in range(4)]
    inside = threading.Barrier(len(peers))
    dials: list[str] = []

    def _dial(network_id: str, **kwargs: Any) -> Any:
        dials.append(str(kwargs.get("expected_device")))
        inside.wait(5)  # BREAKS (raises) if any dial waited for another to finish
        return SimpleNamespace(peer=kwargs.get("expected_device")), ""

    server = _fake_relay(tmp_path, monkeypatch, peers, _dial)

    results = _together([lambda p=p: server._ensure_link(p) for p in peers])

    assert sorted(dials) == peers
    assert [getattr(r, "peer", r) for r in results] == peers


def test_a_failed_dial_releases_every_waiter_with_its_code_and_the_next_call_redials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dials: list[float] = []

    def _dial(network_id: str, **kwargs: Any) -> Any:
        dials.append(time.monotonic())
        time.sleep(0.3)
        return None, "handshake_refused: synthetic"

    server = _fake_relay(tmp_path, monkeypatch, ["d_peer"], _dial)

    results = _together([lambda: server._ensure_link_with_reason("d_peer") for _ in range(5)])

    assert len(dials) == 1, f"{len(dials)} dials"
    assert results == [(None, "handshake_refused: synthetic")] * 5, results
    # Nothing is left behind: the next caller dials fresh.
    assert server._ensure_link_with_reason("d_peer") == (None, "handshake_refused: synthetic")
    assert len(dials) == 2, "a failed flight was cached instead of redialled"


def test_a_dial_that_raises_releases_its_waiters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    def _dial(network_id: str, **kwargs: Any) -> Any:
        calls.append(1)
        time.sleep(0.3)
        raise OSError("synthetic dial failure")

    server = _fake_relay(tmp_path, monkeypatch, ["d_peer"], _dial)

    results = _together([lambda: server._ensure_link("d_peer") for _ in range(3)])

    assert len(calls) == 1
    assert all(isinstance(r, OSError) for r in results), results
    calls.clear()
    with pytest.raises(OSError):
        server._ensure_link("d_peer")
    assert calls == [1], "the next call must dial again"


def test_a_waiter_never_waits_past_its_own_probe_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = threading.Event()
    entered = threading.Event()

    def _dial(network_id: str, **kwargs: Any) -> Any:
        entered.set()
        release.wait(3)
        return SimpleNamespace(), ""

    server = _fake_relay(tmp_path, monkeypatch, ["d_peer"], _dial)
    leader = threading.Thread(target=lambda: server._ensure_link("d_peer"))
    leader.start()
    try:
        assert entered.wait(5), "the leader never dialled"
        started = time.monotonic()
        link, reason, _ = server._ensure_link_with_probe("d_peer", probe_timeout_s=0.2)
        waited = time.monotonic() - started
    finally:
        release.set()
        leader.join(10)
    assert link is None and reason == relay.NOT_ATTEMPTED_REASON
    assert waited < 2.0, f"a 0.2 s probe waited {waited:.2f} s behind another dial"


def test_an_unbounded_caller_does_not_inherit_a_listing_probes_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A session op behind a budget-limited listing probe dials for itself.

    A CONTROL, green before and after: it pins that sharing never hands a real
    session op (no deadline) the failure a listing's budget produced.
    """
    calls: list[float | None] = []
    entered = threading.Event()

    def _dial(network_id: str, **kwargs: Any) -> Any:
        calls.append(kwargs.get("timeout_s"))
        entered.set()
        if len(calls) == 1:
            time.sleep(0.3)
            return None, "connect_failed: synthetic budget"
        return SimpleNamespace(name="dialled-for-itself"), ""

    server = _fake_relay(tmp_path, monkeypatch, ["d_peer"], _dial)
    listing = threading.Thread(
        target=lambda: server._ensure_link_with_probe("d_peer", probe_timeout_s=5.0)
    )
    listing.start()
    assert entered.wait(5), "the listing never dialled"
    link = server._ensure_link("d_peer")
    listing.join(10)

    assert getattr(link, "name", None) == "dialled-for-itself"
    assert len(calls) == 2
