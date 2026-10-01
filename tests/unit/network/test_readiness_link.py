"""Peer readiness over two REAL relays on one host.

The unit file (``test_readiness``) pins the verdict matrix; THIS file pins the
wire and the drive: the real ``lop network ready`` parser against a real
relay-to-relay link, the capability gate that decides between asking and
degrading to ``peer_too_old`` rows, and the discriminator cells the offload run
needed — a stopped relay reads "refused" (the host is up; nothing is
listening), a black-holed endpoint reads "nothing answered", and a second
endpoint that answers is reported as such.

ONE PROCESS, TWO ROOTS, with the constraints the other two-relay files record
(``test_credentials_real_link``, ``test_relay_e2e``): the CLI resolves its
config dir from the AMBIENT environment, so each command runs with the ambient
root pointed at the device the operator would be on — here, the viewer A. A
peer's facts are read under an EXPLICIT root (``readiness._open_store`` passes
``db_path``, so no ambient root can leak into them), and every root used here
is the per-test ``root`` fixture's.
"""

from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import cli as net_cli
from local_operator.network import identity as identity_mod
from local_operator.network import readiness, relay, store, wire
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixtures by import
    _pair,
    devices,
)

Devices = tuple[relay.RelayServer, relay.RelayServer, str, int]


def _parser() -> argparse.ArgumentParser:
    """The shipped ``lop network`` parser, built the way ``lop`` builds it."""
    parser = argparse.ArgumentParser(prog="lop")
    net_cli.add_parser(parser.add_subparsers(dest="command"))
    return parser


def _ready(*argv: str) -> int:
    return net_cli.main(_parser().parse_args(["network", "ready", *argv]))


def _ready_json(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict[str, Any]]:
    rc = _ready(*argv, "--json")
    payload = json.loads(capsys.readouterr().out)
    return rc, payload


def _set_peer_endpoints(
    server: relay.RelayServer, record: Any, device_id: str, endpoints: list[str]
) -> None:
    """Point a member row at what the test actually started.

    The pairing records the joiner's ADVERTISED endpoints, which in this rig is
    a relay that had not bound yet; the tests below set the address the peer is
    truly listening on, the same seam ``test_credentials_real_link`` uses.
    """
    with store.mutate(record.network_id, server.root) as copy:
        row = copy.member(device_id)
        assert row is not None
        row.endpoints = list(endpoints)
        store.save(copy, server.root)


def _rows(payload: dict[str, Any], check: str, **match: Any) -> list[dict[str, Any]]:
    return [
        row
        for row in payload.get("checks") or []
        if row.get("check") == check and all(row.get(key) == value for key, value in match.items())
    ]


def _capability(payload: dict[str, Any], name: str) -> dict[str, Any]:
    found = _rows(payload, "readiness", capability=name)
    assert found, (name, payload.get("checks"))
    return found[0]


def _bind_and_start(server_b: relay.RelayServer, record: Any) -> str:
    """Bind the peer's listener and make ITS OWN row advertise the real address.

    The joiner's record still carries what it advertised mid-ceremony —
    loopback with port 0, because a relay that has not bound yet has no port to
    name — and a running relay re-syncs its own row, so a row A fixed locally
    would be corrected back to the bogus endpoint on the next link. Give B's
    own row the truth first; A's copy is set by the cell and kept by the sync.
    """
    host, port = server_b.bind()
    server_b.bind_control()
    server_b.start()
    live = f"{host}:{port}"
    with store.mutate(record.network_id, server_b.root) as copy:
        own = copy.member(server_b.identity.device_id)
        if own is not None:
            own.endpoints = [live]
            store.save(copy, server_b.root)
    return live


def _settle_the_peer_link(
    server_a: relay.RelayServer, server_b: relay.RelayServer, record: Any, live: str
) -> None:
    """Establish A's link to the live peer BEFORE a report composes, and settle it.

    WHY THE REPORT MUST NOT DIAL LAZILY (CI run 36518398815). A report that finds no
    link DIALS the declared endpoint, and that handshake spends the report's own
    budget: on a loaded shard the dial can fail, and the rows then degrade to
    ``not_asked`` ("the peer did not answer") instead of the row under test — which
    is how ``test_an_old_peer_degrades_to_peer_too_old_and_is_never_asked`` failed
    with ``{'not_asked'}`` against an expected ``{'peer_too_old'}``. The two cells
    that already pinned their link (``test_a_link_that_cannot_be_pinned_…`` and
    ``test_an_unpinnable_link_…``) dial explicitly and assert it; this is that same
    step, shared. With the link held, the report takes the existing-link path and
    asks the peer nothing over a fresh handshake.

    ``_await_membership_pull`` follows the dial because admitting a link runs the
    peer's own membership writers on the accept thread, and a member-row edit that
    lands inside their window is silently dropped (review round 3, R3-1).

    ORDER MATTERS AGAINST A CAPABILITY NARROWING: a link's capability set is exchanged
    in its HANDSHAKE (``PeerLink.capabilities`` is the peer's advertised set), so a
    cell that narrows ``wire.LINK_CAPABILITIES`` to model an old peer must warm the
    link AFTER the narrowing — a link warmed before it still advertises the
    capability, and the report asks a peer the cell means to leave unasked.
    """
    dialed, reason = server_a.dial(
        record.network_id,
        host=live,
        epoch=record.epoch,
        expected_device=server_b.identity.device_id,
    )
    assert dialed is not None, reason
    link = server_a._link_for(server_b.identity.device_id)  # noqa: SLF001 — the one link seam
    assert link is not None
    _await_membership_pull(link)


# ---------------------------------------------------------------------------
# The wire: registered, asked, and the three reachability readings
# ---------------------------------------------------------------------------


def test_both_ends_serve_the_readiness_op(request: pytest.FixtureRequest) -> None:
    pair_devices: Devices = request.getfixturevalue("devices")
    server_a, server_b, _host, _port = pair_devices
    for server in (server_a, server_b):
        assert "net_readiness" in server._handlers  # noqa: SLF001 — the slice's seat
        assert "peer_readiness" in server._local_slice_handlers  # noqa: SLF001
    assert wire.PEER_READINESS_V1 in wire.LINK_CAPABILITIES


def test_ready_reads_a_stopped_relay_as_refused_and_a_black_hole_as_silent(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The operator's discriminator, driven through the real parser and a real probe.

    A closed loopback port REFUSES (the host is up; nothing is listening there)
    and a TEST-NET endpoint answers nothing; the black-holed row carries the
    route this device would have taken, observed rather than inferred. Neither
    cell needs the peer's relay to exist, and both keep the raw vocabulary in
    ``--json`` while the human rows read as sentences.
    """
    pair_devices: Devices = request.getfixturevalue("devices")
    server_a, server_b, _host, _port = pair_devices
    record, _h, _p = _pair(pair_devices, monkeypatch)
    # The joining half prints its own receipt JSON; a cell that parsed a
    # later capture must not find two documents in it.
    capsys.readouterr()
    # The join's client link on this (accepting) side is reaped asynchronously;
    # a report composed inside that window takes the link path and renders
    # `unknown` instead of `not asked` (Q-O1). Wait for the reap first.
    _await_link_reaped(server_a, server_b.identity.device_id)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_a.root))
    _set_peer_endpoints(server_a, record, server_b.identity.device_id, ["127.0.0.1:1"])

    rc, payload = _ready_json(capsys, "--peer", server_b.identity.name)
    assert rc == 1
    reach = _rows(payload, "reachability", device_id=server_b.identity.device_id)
    assert len(reach) == 1, reach
    assert reach[0]["ok"] is False
    assert reach[0]["detail"].startswith(relay.CONNECT_FAILED_PREFIX)
    assert reach[0]["observed"]["outcome"] == "refused"
    assert reach[0]["observed"]["attempted"] is True
    # No link, so nothing was asked and no capability could have passed.
    assert all(row["ok"] is False for row in _rows(payload, "readiness"))
    assert {row["code"] for row in _rows(payload, "readiness")} == {readiness.CODE_NOT_ASKED}

    human = _human(capsys, "--peer", server_b.identity.name)
    assert "ConnectionRefusedError" not in human
    assert "something answered this address and refused the connection" in human

    _set_peer_endpoints(
        server_a,
        record,
        server_b.identity.device_id,
        ["127.0.0.1:1", "203.0.113.9:9"],
    )
    rc, payload = _ready_json(capsys, "--peer", server_b.identity.name)
    assert rc == 1
    by_endpoint = {
        row["endpoint"]: row
        for row in _rows(payload, "reachability", device_id=server_b.identity.device_id)
    }
    silent = by_endpoint["203.0.113.9:9"]
    assert silent["ok"] is False
    assert silent["observed"]["outcome"] in {"no_answer", "no_route"}
    if silent["observed"]["outcome"] == "no_answer":
        # The observed route facts: on a host that has a route they are filled,
        # and when they are not the outcome says so instead of guessing.
        assert silent["observed"]["source_address"] or silent["observed"]["interface"]


def _human(capsys: pytest.CaptureFixture[str], *argv: str) -> str:
    rc = _ready(*argv)
    assert rc == 1
    return capsys.readouterr().out


def test_a_second_endpoint_that_answers_is_reported_while_the_dead_one_is_named(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """One address answered, another refused — the row says which.

    The state the offload run actually produced on the cloud node: a member row
    whose FIRST endpoint was dead while the peer was up on the second. The dead
    row must read as "the peer is up (it answered <winner>); this address did
    not answer", and the remedy must point at the address rather than send
    someone to restart a running relay.
    """
    pair_devices: Devices = request.getfixturevalue("devices")
    server_a, server_b, _host, _port = pair_devices
    record, _h, _p = _pair(pair_devices, monkeypatch)
    # The joining half prints its own receipt JSON; a cell that parsed a
    # later capture must not find two documents in it.
    capsys.readouterr()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_a.root))
    live = _bind_and_start(server_b, record)
    _settle_the_peer_link(server_a, server_b, record, live)
    _set_peer_endpoints(server_a, record, server_b.identity.device_id, ["127.0.0.1:1", live])

    rc, payload = _ready_json(capsys, "--peer", server_b.identity.name)
    assert rc == 1  # a fresh peer is unreachable-then-unready: the install gaps fail
    reach = _rows(payload, "reachability", device_id=server_b.identity.device_id)
    by_endpoint = {row["endpoint"]: row for row in reach}
    assert by_endpoint[live]["ok"] is True
    dead = by_endpoint["127.0.0.1:1"]
    assert dead["ok"] is False
    assert dead["observed"]["outcome"] == "refused"
    assert dead["observed"]["winner"] == live
    assert dead["remedies"] and "working address" in dead["remedies"][0]
    # And the link that the winner produced is what the capability rows were
    # asked over: the build row is a real comparison, not a not-asked placeholder.
    build = _capability(payload, readiness.CAPABILITY_BUILD)
    assert build["ok"] is True and build.get("code", "") == ""
    server_b.stop()


def test_a_non_peer_listener_is_never_reported_as_the_peer(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Design round 1, D1, driven live: something else accepts the peer's address.

    The reviewer's reproduction on this host — a non-peer process holding a
    candidate port — reduced to its mechanism: the first report DIALS the peer's
    real address (identifying it, so a link exists), then a bare listener holds
    a second declared address. In the second report the listener's accept must
    not read as the peer (the row renders the observed fact), while the peer's
    OWN address keeps its claim — "answered" when it wins the race, "the link
    runs there" when it does not, which is why the outcomes are asserted as a
    set rather than a winner this test cannot schedule.
    """
    pair_devices: Devices = request.getfixturevalue("devices")
    server_a, server_b, _host, _port = pair_devices
    record, _h, _p = _pair(pair_devices, monkeypatch)
    capsys.readouterr()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_a.root))
    live = _bind_and_start(server_b, record)
    _settle_the_peer_link(server_a, server_b, record, live)
    _set_peer_endpoints(server_a, record, server_b.identity.device_id, [live])

    # First report: the dial identifies the peer at ``live`` and installs the link.
    rc, _payload = _ready_json(capsys, "--peer", server_b.identity.name)
    assert rc == 1

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(8)  # the backlog completes accepts; nobody here is the peer
        fake = f"127.0.0.1:{listener.getsockname()[1]}"
        _set_peer_endpoints(server_a, record, server_b.identity.device_id, [fake, live])

        rc, payload = _ready_json(capsys, "--peer", server_b.identity.name)
        assert rc == 1
        reach = {row["endpoint"]: row for row in _rows(payload, "reachability")}

        imposter = reach[fake]
        assert imposter["ok"] is False
        assert imposter["observed"]["outcome"] in ("connected_unverified", "connected_elsewhere")
        reading = readiness.reachability_reading(imposter)
        assert "was not identified as the peer" in reading
        assert "the peer answered at this address" not in reading
        assert not reading.startswith("the peer")

        # THE COUNTER-DIRECTION: the peer's own address keeps its claim.
        real = reach[live]
        assert real["ok"] is True
        assert real["observed"]["outcome"] in ("connected", "connected_link")
    finally:
        listener.close()
    server_b.stop()


def test_a_verified_peer_answer_still_claims_the_peer(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The counter-cell to D1: the socket this report DIALED and identified.

    No link exists yet, so the winner socket goes through the real handshake —
    and only then does the row say the peer answered.
    """
    pair_devices: Devices = request.getfixturevalue("devices")
    server_a, server_b, _host, _port = pair_devices
    record, _h, _p = _pair(pair_devices, monkeypatch)
    capsys.readouterr()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_a.root))
    live = _bind_and_start(server_b, record)
    # NO LINK IS WARMED HERE, deliberately: this cell is the counter-cell to D1 and its
    # subject IS the report's own dial — the row must read ``observed.outcome ==
    # "connected"``, the socket THIS report handshook, not ``connected_link``.
    _set_peer_endpoints(server_a, record, server_b.identity.device_id, [live])

    rc, payload = _ready_json(capsys, "--peer", server_b.identity.name)
    assert rc == 1
    row = _rows(payload, "reachability")[0]
    assert row["ok"] is True
    assert row["observed"]["outcome"] == "connected"
    assert row["observed"]["winner_verified"] is True
    assert readiness.reachability_reading(row) == "the peer answered at this address"
    server_b.stop()


def _await_membership_pull(link: relay.PeerLink, *, settle_s: float = 0.2) -> None:
    """Wait out a just-admitted link's membership writers before editing a copy.

    Admitting a link runs the accepting side's ``_note_peer_endpoints`` and
    ``_pull_members`` -> ``_learn_members_from`` on its accept thread: both
    rewrite and persist this device's copy of the peer's row, and the pull's
    persist trails its ``member_pulled_at`` stamp (stamped BEFORE the merge).
    An edit landing in that window is silently dropped — measured as 7 false
    reds in 31 runs (review round 3, R3-1). So: wait for the pull's stamp (both
    writers run before it), then settle past the bounded save. The deadline
    spans the pull's OWN timeout (``MEMBERSHIP_PULL_TIMEOUT_S``) plus margin: a
    narrower deadline would let a slow-but-successful pull land after the
    helper returned and re-open the window (review round 4, R4-1).
    """
    deadline = time.monotonic() + relay.MEMBERSHIP_PULL_TIMEOUT_S + 1.0
    while not link.member_pulled_at and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(settle_s)


def _await_link_reaped(
    server: relay.RelayServer, device_id: str, *, deadline_s: float = 5.0
) -> None:
    """Wait out a finally-completed join's async link teardown.

    Admitting the join leaves a client link on the accepting side that is torn
    down asynchronously — still registered 10-18 ms after ``_pair`` returns in
    40/60 probed runs. A report composed inside that window takes the link
    path and renders ``unknown`` capability rows instead of the expected
    ``not_asked`` (observed once in 101 file-level runs; QA round 4, Q-O1).
    The loop is bounded so a torn fixture cannot hang the suite; the poll
    reads ``_link_for``, which reports only ALIVE links.
    """
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        link = server._link_for(device_id)  # noqa: SLF001 — the one link seam
        if link is None:
            return
        time.sleep(0.01)


def test_a_link_that_cannot_be_pinned_never_names_its_source_socket(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Round 2, R2-1/Q-3, inbound direction: the peer dialled US.

    An inbound link records the dialer's SOURCE socket, so that address may
    not be published as "the peer's link runs at …" (it dies with the
    connection), and the peer's own reachable address must not read unverified
    merely because the link's address cannot be named. The link fact — the peer
    is up — is presented with no address at all; the ephemeral appears nowhere.
    """
    pair_devices: Devices = request.getfixturevalue("devices")
    server_a, server_b, host, port = pair_devices
    record, _h, _p = _pair(pair_devices, monkeypatch)
    capsys.readouterr()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_a.root))
    live = _bind_and_start(server_b, record)
    # The PEER is the dialer: B completes the handshake into A, so A's link
    # records B's ephemeral source socket rather than any declared address.
    peer_link, reason = server_b.dial(record.network_id, host=f"{host}:{port}", epoch=record.epoch)
    assert peer_link is not None, reason
    link = server_a._link_for(server_b.identity.device_id)  # noqa: SLF001 — the one link seam
    assert link is not None
    ephemeral = str(link.peer_addr)
    # Both membership writers run before the pull's stamp; the stamp + settle
    # then pass the pull's own save (R3-1) — only after that is the edit safe.
    _await_membership_pull(link)
    _set_peer_endpoints(server_a, record, server_b.identity.device_id, [live, "127.0.0.1:1"])

    rc, payload = _ready_json(capsys, "--peer", server_b.identity.name)
    # The dead sibling address keeps the report red; the peer's own address
    # does not.
    assert rc == 1
    rows = {row["endpoint"]: row for row in _rows(payload, "reachability")}
    live_row = rows[live]
    assert live_row["ok"] is True
    assert live_row["observed"]["outcome"] == "connected_unpinned"
    reading = readiness.reachability_reading(live_row)
    assert "its link is live" in reading
    assert "not identified" not in reading
    assert "link_address" not in live_row["observed"]
    dead_row = rows["127.0.0.1:1"]
    assert dead_row["ok"] is False
    assert "its link is live" in dead_row["remedies"][0]
    # The ephemeral is nowhere in the payload: not in observed, not in the
    # summary message, not in a remedy. (Guarded only for the theoretical
    # port-reuse collision, where the premise — an unpinnable address — is
    # gone and the row legitimately reads connected_link instead.)
    if ephemeral not in (live, "127.0.0.1:1"):
        assert ephemeral not in json.dumps(payload)
    server_b.stop()


def test_an_unpinnable_link_carries_no_direction_claim(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Round 3, R3-2: "unpinnable" is the condition, not a direction.

    An OUTBOUND link whose recorded address has left the declared set is
    unpinned too — the row and its remedy must say so without claiming the
    peer dialled us.
    """
    pair_devices: Devices = request.getfixturevalue("devices")
    server_a, server_b, _host, _port = pair_devices
    record, _h, _p = _pair(pair_devices, monkeypatch)
    capsys.readouterr()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_a.root))
    live = _bind_and_start(server_b, record)
    # A dials B: OUTBOUND, recorded at ``live``.
    dialed, reason = server_a.dial(
        record.network_id,
        host=live,
        epoch=record.epoch,
        expected_device=server_b.identity.device_id,
    )
    assert dialed is not None, reason
    link = server_a._link_for(server_b.identity.device_id)  # noqa: SLF001 — the one link seam
    assert link is not None
    _await_membership_pull(link)
    # The declared set no longer contains the recorded address (as if the peer
    # had re-published elsewhere): the link is unpinned while still outbound.
    _set_peer_endpoints(server_a, record, server_b.identity.device_id, ["127.0.0.1:1"])

    rc, payload = _ready_json(capsys, "--peer", server_b.identity.name)
    assert rc == 1
    row = _rows(payload, "reachability")[0]
    assert row["endpoint"] == "127.0.0.1:1"
    assert row["ok"] is False
    assert row["observed"]["outcome"] == "refused"
    assert row["observed"].get("link_unpinned") is True
    assert "link_address" not in row["observed"]
    assert "its link is live" in row["remedies"][0]
    # NO DIRECTION CLAIM anywhere: the field names the condition (R3-2).
    rendered = json.dumps(payload)
    assert "link_inbound" not in rendered
    assert "inbound" not in rendered
    server_b.stop()


# ---------------------------------------------------------------------------
# The ask: a real answer, and the capability gate for a peer that predates it
# ---------------------------------------------------------------------------


def test_ready_flips_a_blocked_peer_to_ready_as_each_condition_is_fixed(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """The headline: every offload blocker the 2026-09-28 run hit, one row each.

    BEFORE: no operator authority, no git identity, no MCP surface, no default
    model — four rows, four codes, and the remedy each one names. AFTER fixing
    each ON THE PEER (the files its own processes read), the same report is
    ready: every row ok and rc 0. The operator step is stubbed at the
    authority-report seam rather than install a launchd anchor, because the
    claim under test is what a given LEVEL renders as, not how a host came to
    have it.
    """
    pair_devices: Devices = request.getfixturevalue("devices")
    server_a, server_b, _host, _port = pair_devices
    record, _h, _p = _pair(pair_devices, monkeypatch)
    # The joining half prints its own receipt JSON; a cell that parsed a
    # later capture must not find two documents in it.
    capsys.readouterr()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_a.root))
    live = _bind_and_start(server_b, record)
    _settle_the_peer_link(server_a, server_b, record, live)
    _set_peer_endpoints(server_a, record, server_b.identity.device_id, [live])

    # The peer's own environment: a scratch HOME with no git config, and the
    # authority report stubbed to the not-installed level.
    home = tmp_path / "peer-home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(
        "local_operator.operator.operator_authority_report",
        lambda **kwargs: {
            "level": "spawn-capability-only",
            "reason": "no anchor is installed, so the runtime trusts no key yet",
            "anchor_installed": False,
            "anchor_root_owned": False,
            "presence": False,
        },
    )

    rc, payload = _ready_json(capsys, "--peer", server_b.identity.name)
    assert rc == 1
    operator_row = _capability(payload, readiness.CAPABILITY_OPERATOR_AUTHORITY)
    assert (operator_row["ok"], operator_row["code"]) == (
        False,
        readiness.CODE_NOT_INSTALLED,
    )
    assert "ask Local Operator to set up" in " ".join(operator_row["remedies"])
    git_row = _capability(payload, readiness.CAPABILITY_GIT)
    assert (git_row["ok"], git_row["code"]) == (False, readiness.CODE_NO_GIT_IDENTITY)
    assert "git config --global user.name" in " ".join(git_row["remedies"])
    mcp_row = _capability(payload, readiness.CAPABILITY_MCP_SERVERS)
    assert (mcp_row["ok"], mcp_row["code"]) == (False, readiness.CODE_NO_MCP_SERVERS)
    assert "/mcp add" in " ".join(mcp_row["remedies"])
    model_row = _capability(payload, readiness.CAPABILITY_MODEL_CREDENTIAL)
    assert (model_row["ok"], model_row["code"]) == (False, readiness.CODE_NOT_CONFIGURED)
    assert "/model default" in " ".join(model_row["remedies"])

    # --- the fixes, each on the PEER's side --------------------------------
    monkeypatch.setattr(
        "local_operator.operator.operator_authority_report",
        lambda **kwargs: {
            "level": "operator-presence",
            "reason": "the anchor is pinned and the key is in the system store",
            "anchor_installed": True,
            "anchor_root_owned": True,
            "presence": True,
        },
    )
    (home / ".gitconfig").write_text(
        "[user]\n\tname = Cloud Node\n\temail = cloud-node@example.test\n", encoding="utf-8"
    )
    server_b.root.joinpath("mcp.json").write_text(
        json.dumps({"mcpServers": {"files": {"command": "npx", "args": ["-y", "files"]}}}),
        encoding="utf-8",
    )
    server_b.root.joinpath("config.yml").write_text(
        'version: "0.0.0"\nvalues:\n  hosting: openai\n  model_name: gpt-5\n',
        encoding="utf-8",
    )
    from local_operator.providers.auth_store import AuthStore

    auth = AuthStore(db_path=server_b.root / "auth.db", config_dir=server_b.root)
    auth.upsert_credential("openai", {"type": "api_key", "key": "local-test"})
    auth.close()

    rc, payload = _ready_json(capsys, "--peer", server_b.identity.name)
    ok_rows = [
        row
        for row in payload.get("checks") or []
        if row.get("check") in {"reachability", "readiness"}
    ]
    failures = [row for row in ok_rows if not row.get("ok")]
    assert not failures, failures
    assert rc == 0, payload
    assert payload["ok"] is True
    server_b.stop()


def test_an_old_peer_degrades_to_peer_too_old_and_is_never_asked(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Capability-absent degrades; asking anyway would be a request the far side
    answers with a refusal it had to compose, so the gate is asserted by CALL
    COUNT: the peer's handler was never entered.

    The build row still works — its evidence rides the handshake, which is what
    makes it the one check an old peer can pass — and the four peer-side checks
    come back ``peer_too_old`` with the remedy that clears it.
    """
    pair_devices: Devices = request.getfixturevalue("devices")
    server_a, server_b, _host, _port = pair_devices
    record, _h, _p = _pair(pair_devices, monkeypatch)
    # The joining half prints its own receipt JSON; a cell that parsed a
    # later capture must not find two documents in it.
    capsys.readouterr()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_a.root))
    live = _bind_and_start(server_b, record)
    _set_peer_endpoints(server_a, record, server_b.identity.device_id, [live])
    calls: list[int] = []
    original = server_b._handlers["net_readiness"]  # noqa: SLF001 — the slice's seat
    server_b._handlers["net_readiness"] = lambda link, frame: (  # noqa: SLF001
        calls.append(1),
        original(link, frame),
    )[1]

    monkeypatch.setattr(
        wire,
        "LINK_CAPABILITIES",
        tuple(c for c in wire.LINK_CAPABILITIES if c != wire.PEER_READINESS_V1),
    )
    # THE LINK IS ESTABLISHED AFTER THE NARROWING, and the order is the whole point: a
    # link's capability set is EXCHANGED IN ITS HANDSHAKE (``PeerLink.capabilities`` is
    # the peer's advertised set), so a link warmed before this monkeypatch would still
    # carry ``PEER_READINESS_V1`` and the report would ASK — which is exactly what this
    # cell counts calls to forbid.
    _settle_the_peer_link(server_a, server_b, record, live)
    rc, payload = _ready_json(capsys, "--peer", server_b.identity.name)
    assert rc == 1
    assert calls == [], "the viewer asked a peer that never advertised the capability"
    too_old = [
        row
        for row in _rows(payload, "readiness")
        if row["capability"] != readiness.CAPABILITY_BUILD
    ]
    assert {row["code"] for row in too_old} == {readiness.CODE_PEER_TOO_OLD}
    assert {row["capability"] for row in too_old} == set(readiness.PEER_SIDE_CHECKS)
    assert all(
        any("ask Local Operator to update it" in remedy for remedy in row["remedies"])
        for row in too_old
    )
    build = _capability(payload, readiness.CAPABILITY_BUILD)
    assert build["ok"] is True, build
    server_b.stop()


def test_the_report_is_read_only_on_both_roots(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No new files appear under either device's root across a report.

    Warm first (one run, so anything lazily created exists before the
    snapshot), then run again and compare the file SETS: the read-only claim is
    about creation, and it is asserted over both roots because the viewer's
    relay and the peer's handler both read on this path.
    """
    pair_devices: Devices = request.getfixturevalue("devices")
    server_a, server_b, _host, _port = pair_devices
    record, _h, _p = _pair(pair_devices, monkeypatch)
    # The joining half prints its own receipt JSON; a cell that parsed a
    # later capture must not find two documents in it.
    capsys.readouterr()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_a.root))
    live = _bind_and_start(server_b, record)
    _settle_the_peer_link(server_a, server_b, record, live)
    _set_peer_endpoints(server_a, record, server_b.identity.device_id, [live])

    _ready_json(capsys, "--peer", server_b.identity.name)

    def snapshot(root: Path) -> set[str]:
        return {str(path.relative_to(root)) for path in root.rglob("*")}

    before = {"a": snapshot(server_a.root), "b": snapshot(server_b.root)}
    _ready_json(capsys, "--peer", server_b.identity.name)
    after = {"a": snapshot(server_a.root), "b": snapshot(server_b.root)}
    for side in ("a", "b"):
        created = after[side] - before[side]
        assert not created, f"a readiness report created files on {side}: {sorted(created)}"
    # And the two files a writer would leave behind are still absent on the viewer.
    assert not (server_a.root / "auth.db").exists()
    assert not (server_a.root / "mcp.json").exists()
    server_b.stop()


def test_peer_status_rows_carry_the_peers_build_and_capabilities(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The UI's version line: read off the link the listing's probe already made.

    ``peer_status`` publishes both keys with both values on every row; an old
    peer would send ``{}`` / ``[]`` and a dark peer the same — the point pinned
    here is that a live one sends its real stamp and the capability list this
    whole slice gates on, with no extra dial.
    """
    pair_devices: Devices = request.getfixturevalue("devices")
    server_a, server_b, _host, _port = pair_devices
    record, _h, _p = _pair(pair_devices, monkeypatch)
    live = _bind_and_start(server_b, record)
    _settle_the_peer_link(server_a, server_b, record, live)
    _set_peer_endpoints(server_a, record, server_b.identity.device_id, [live])

    rows = [
        row for row in server_a.peer_status() if row["device_id"] == server_b.identity.device_id
    ]
    assert rows, "the peer was not in the table"
    row = rows[0]
    assert row["reachable"] is True
    assert row["build"].get("version") == server_b.build.get("version")
    assert row["build"] == dict(server_b.build)
    assert wire.PEER_READINESS_V1 in row["capabilities"]
    assert wire.MESH_NET_V1 in row["capabilities"]
    server_b.stop()


def test_ready_names_no_peer_when_the_filter_matches_nothing(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--peer`` resolves names; an unknown one refuses rather than answering
    about every member — the failure direction that matters is the loud one.

    The refusal is the family's shape (``ok``/``code``/``message``), so an agent
    branches on it the same way it branches on every other `lop network`
    refusal; and the empty report for a name nobody holds never silently reads
    as "all clean".
    """
    pair_devices: Devices = request.getfixturevalue("devices")
    server_a, server_b, _host, _port = pair_devices
    _pair(pair_devices, monkeypatch)
    capsys.readouterr()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_a.root))

    rc = _ready("--peer", "not-a-device", "--json")
    payload = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert payload["ok"] is False and payload["code"]
    assert "not-a-device" in payload["message"]


def test_the_viewer_reports_its_own_facts_the_same_way(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``lop network ready --peer <B>`` run ON B describes A identically.

    Readiness is symmetric: each side reports the other's install, so the file
    facts a peer cannot see from outside (its operator level, its git identity)
    are still the facts a report about it carries — driven here from the other
    end so a one-sided implementation cannot pass both directions.
    """
    pair_devices: Devices = request.getfixturevalue("devices")
    server_a, server_b, _host, _port = pair_devices
    record, _h, _p = _pair(pair_devices, monkeypatch)
    # The joining half prints its own receipt JSON; a cell that parsed a
    # later capture must not find two documents in it.
    capsys.readouterr()
    live = _bind_and_start(server_b, record)
    _settle_the_peer_link(server_a, server_b, record, live)
    _set_peer_endpoints(server_a, record, server_b.identity.device_id, [live])
    # B dials A: give B A's real address.
    with store.mutate(record.network_id, server_b.root) as copy:
        owner_row = copy.member(server_a.identity.device_id)
        assert owner_row is not None
        owner_row.endpoints = [f"{_host}:{_port}"]
        store.save(copy, server_b.root)

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_b.root))
    rc = _ready("--peer", server_a.identity.name, "--json")
    payload = json.loads(capsys.readouterr().out)
    # A is not a ready peer on this host either (no operator authority here), so
    # rc follows the rows the same way round from this direction.
    assert rc == 1 and payload["ok"] is False
    assert _capability(payload, readiness.CAPABILITY_OPERATOR_AUTHORITY)["ok"] is False
    assert _capability(payload, readiness.CAPABILITY_BUILD)["ok"] is True
    server_b.stop()


def test_a_fresh_device_with_no_networks_remembers_nothing_to_report(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No networks: the report is empty and honest, not a failure with no cause."""
    pair_devices: Devices = request.getfixturevalue("devices")
    server_a, server_b, _host, _port = pair_devices
    fresh = server_b.root / "fresh"
    fresh.mkdir()
    identity_mod.mint(fresh, name="fresh-device")
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(fresh))
    rc, payload = _ready_json(capsys)
    assert rc == 0
    assert payload["checks"] == [] or all(row.get("ok") for row in payload["checks"])
    assert payload["identity_present"] is True
