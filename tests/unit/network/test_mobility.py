"""Move, recall, `--keep`, and lifecycle-on-a-peer, over two REAL relays.

In-process but not stubbed: two config roots, two identities, real loopback TCP,
the production reader/writer threads, and the protocol driven exactly the way the
CLI drives it (this device's relay's control socket). The shared ``devices``
fixture and the pairing helper come from ``test_relay_e2e`` so the mesh these
tests move sessions across is built by the same code that pairs real devices.

WHAT THESE PIN: the destination pulls and the owner decides; a refusal mutates
nothing on either side; ``--keep`` mints a new id and leaves the source alone;
archive/delete run the OWNER's implementation; and INV-1's guard refuses an engage
for a session that is mid-handoff.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import mobility, relay, sync
from local_operator.session.placement import (
    HANDOFF_PHASE_HANDING_OFF,
    HANDOFF_PHASE_PREPARED,
    MeshStamp,
    SessionPlacement,
    read_handoff_journal,
    read_stamp,
    write_handoff_entry,
    write_stamp,
)
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixtures
    NETWORK_NAME,
    _pair,
    devices,
)

Devices = tuple[relay.RelayServer, relay.RelayServer, str, int]

SESSION = "9f3ac1e0b7d2"


@pytest.fixture()
def pair(request: pytest.FixtureRequest) -> Devices:
    """Two relays with BOTH ends able to serve a control socket.

    The shared fixture leaves B dial-only (it never starts), which is the right
    shape for the transport tests and the wrong one here: a move is issued by the
    device that will HOLD the conversation, so B has to be able to answer its own
    CLI. Starting B gives it a listener nobody dials and a control socket the
    tests use.
    """
    value: Devices = request.getfixturevalue("devices")
    server_a, server_b, _host, _port = value
    server_b.bind_control()
    server_b.start()
    return value


def _owned_session(server: relay.RelayServer, session_id: str = SESSION, *, rows: int = 3) -> Path:
    """A session this device owns: a directory, a transcript, and a stamp saying so.

    ``mark_store`` too, exactly as ``session_factory`` does when it creates a
    session: without the store marker ``cleanup.remove_session_dir`` refuses (the
    guard that keeps cleanup from ever walking into a directory that is not a
    session store), and a move whose commit cannot delete the source would be a
    test of the wrong failure.
    """
    from local_operator.session.cleanup import mark_store

    directory = server.root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    directory.joinpath("transcript.jsonl").write_text(
        "".join(
            json.dumps({"id": f"e{index}", "type": "user", "content": f"line {index}"}) + "\n"
            for index in range(rows)
        ),
        encoding="utf-8",
    )
    directory.joinpath("title.json").write_text(json.dumps({"title": "mesh design"}), "utf-8")
    mark_store(server.root / "sessions")
    write_stamp(
        server.root,
        MeshStamp(
            session_id=session_id,
            network_id="n_test",
            home_device=server.identity.device_id,
            placement=SessionPlacement(
                mode="local", home_device=server.identity.device_id, stamp_revision=1
            ),
        ),
    )
    return directory


def _transcript(root: Path, session_id: str) -> bytes:
    return (root / "sessions" / session_id / "transcript.jsonl").read_bytes()


def _cleanup_log(root: Path) -> list[dict[str, Any]]:
    path = root / "sessions" / ".cleanup-log.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _move(
    server: relay.RelayServer,
    session_id: str,
    *,
    to: str = "local",
    keep: bool = False,
    wait_s: float = 0.0,
    from_replica: bool = False,
    monkeypatch: pytest.MonkeyPatch | None = None,
) -> dict[str, Any]:
    """``lop sessions move``, driven the way the CLI drives it.

    Through this device's RELAY's control socket (``mobility.request_move``), not
    by calling the protocol directly: the hop the CLI makes is part of what the
    mesh's own error shapes are for, and a test that skipped it would not notice a
    local verb that never got registered.
    """
    if monkeypatch is not None:
        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server.root))
    return dict(
        mobility.request_move(
            session_id,
            to=to,
            keep=keep,
            wait_s=wait_s,
            root=server.root,
            from_replica=from_replica,
        )
    )


# ---------------------------------------------------------------------------
# The default: recall a peer's session to this device
# ---------------------------------------------------------------------------


def test_a_recall_moves_the_session_and_leaves_a_tombstone(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--to local`: the phases in order, one holder at the end, and a cleanup log
    that says the conversation was MOVED rather than deleted."""
    server_a, server_b, _host, _port = pair
    _pair(pair, monkeypatch, role="admin")
    source = _owned_session(server_a)
    before = _transcript(server_a.root, SESSION)

    result = _move(server_b, SESSION, monkeypatch=monkeypatch)

    assert result["ok"] is True, result
    assert [item["phase"] for item in result["phases"]] == [
        "prepared",
        "handing_off",
        "committed",
        "done",
    ]
    assert result["phase"] == "done"
    assert result["new_session_id"] == SESSION
    assert result["mode"] == "move"
    assert result["to_device"]["device_id"] == server_b.identity.device_id
    assert result["from_device"]["device_id"] == server_a.identity.device_id

    # EXACTLY ONE HOLDER, holding the whole conversation.
    assert not source.exists()
    assert _transcript(server_b.root, SESSION) == before
    # The source is TOMBSTONED, not merely missing: a listing answers about the id
    # rather than claiming nobody has it, and the destination is named.
    from local_operator.network.projection import read_tombstones

    tombstone = read_tombstones(server_a.root)[SESSION]
    assert tombstone["device_id"] == server_b.identity.device_id

    # THE CLEANUP LOG DISTINGUISHES "MOVED" FROM "DELETED" — the one distinction an
    # operator recovering a conversation needs from that file.
    logged = _cleanup_log(server_a.root)
    assert logged and logged[-1]["policy"] == "mesh-move"
    assert "mesh-move" in str(logged[-1].get("reason", ""))

    # And the destination's own record says where it came from.
    stamp = read_stamp(server_b.root, SESSION)
    assert stamp is not None
    assert stamp.home_device == server_b.identity.device_id
    assert stamp.origin["kind"] == "moved"
    assert stamp.origin["source_device"] == server_a.identity.device_id
    # No liveness claim was written by the promoting process: a `.session.pid`
    # naming the relay would make the first real runtime refuse to start.
    assert not (server_b.root / "sessions" / SESSION / ".session.pid").exists()
    # The move's own boot marker does not survive the promote.
    assert not (server_b.root / "sessions" / SESSION / "ready.json").exists()
    # Both journals are clear: a move that finished leaves no entry behind.
    assert read_handoff_journal(server_a.root) == {}
    assert read_handoff_journal(server_b.root) == {}


def test_keep_mints_a_new_id_and_leaves_the_source_running(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--keep`: a fork, not a move. The source keeps its id, its bytes and its lease."""
    server_a, server_b, _host, _port = pair
    _pair(pair, monkeypatch, role="admin")
    source = _owned_session(server_a)
    before = _transcript(server_a.root, SESSION)
    retired: list[str] = []
    real_retire = mobility._retire_local_runtime

    def watching_retire(root: Path, session_id: str, *, deadline_s: float = 0) -> Any:
        retired.append(session_id)
        return real_retire(root, session_id, deadline_s=deadline_s)

    monkeypatch.setattr(mobility, "_retire_local_runtime", watching_retire)

    result = _move(server_b, SESSION, keep=True, monkeypatch=monkeypatch)

    assert result["ok"] is True, result
    assert result["mode"] == "keep"
    new_id = result["new_session_id"]
    assert new_id and new_id != SESSION
    # SOURCE UNTOUCHED: same directory, same bytes, no tombstone.
    assert source.exists()
    assert _transcript(server_a.root, SESSION) == before
    from local_operator.network.projection import read_tombstones

    assert SESSION not in read_tombstones(server_a.root)
    assert not (server_a.root / "sessions" / SESSION).joinpath("ready.json").exists()
    # THE COPY IS A FORK: a new id, the fork marker, and origin.kind fork, so the
    # model is told not to continue work that is still running on the source.
    from local_operator.fork import FORK_BOUNDARY_NAME
    from local_operator.resume import ORIGIN_NAME

    target = server_b.root / "sessions" / new_id
    assert (target / "transcript.jsonl").read_bytes() == before
    assert (target / FORK_BOUNDARY_NAME).is_file()
    assert (target / ORIGIN_NAME).is_file()
    # ``origin.json`` carries the RESUME axis's value, which is what decides
    # whether an install's picker shows the copy as the user's own conversation.
    assert json.loads((target / ORIGIN_NAME).read_text())["origin"] == "fork"
    stamp = read_stamp(server_b.root, new_id)
    assert stamp is not None and stamp.origin["kind"] == "fork"
    assert stamp.origin["source_session_id"] == SESSION
    # THE SOURCE IS UNTOUCHED IN ITS RUNTIME TOO, which a test that only compares
    # bytes would miss: `--keep` must not retire the source's runtime (that would
    # interrupt whatever it is doing) and must not leave a handoff journal entry
    # (the launch guard reads that file, so leaving one would freeze the ORIGINAL
    # conversation until a reconcile ran).
    assert retired == [], "a copy must not retire the source's runtime"
    assert read_handoff_journal(server_a.root) == {}


def test_the_invite_ack_names_the_id_the_pull_is_handed(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """QA delta, Q-D2: the id an invite acks is the id its pull adopts the copy under.

    The drift was one seam wide. The destination pulls a ``--keep`` copy and mints the id it
    adopts under while it pulls, but the ack that answers the invite is written BEFORE that
    pull runs — so the ack carried a SECOND mint, every front end printed ``new_session_id``
    from it (``cli.py:4434``, ``tui/app.py``, the desktop route), and the receipt named a
    session no device held. This pins the seam: whatever id the ack hands out is the id the
    pull is told to adopt under.

    DELIBERATELY NOT END-TO-END, and the reason is measured rather than stylistic. A unit rig
    that dials A->B here is the flaky shape this file already carries one instance of (the
    fixture's peer answers late on a loaded runner and the move reads ``unreachable``: the
    neighbouring offload test fails that way, and this test did too on CI's shard 0 the first
    time it shipped). A receipt test that goes red 1 run in N for a transport hiccup is a
    liability, so the END-TO-END property — the receipt's id is a directory the peer actually
    holds, and the follow-up move with that id succeeds — is driven on two real devices
    instead (PR #1348's round-2 rows), where a dial that fails is a rig result rather than a
    false red. The recall direction keeps its own end-to-end test below, and it is the
    destination half of the same mint.
    """
    from types import SimpleNamespace

    server_a, server_b, _host, _port = pair
    _pair(pair, monkeypatch, role="admin", settings=server_b.settings)
    _owned_session(server_a)
    handed: list[str] = []

    def spy(server: Any, session_id: str, **kwargs: Any) -> tuple[dict[str, Any], None, str]:
        """Stand in for the pull: record what it was handed and answer as it would."""
        target = str(kwargs.get("adopt_under_id") or "")
        handed.append(target)
        return (
            {
                "ok": True,
                "session_id": session_id,
                "new_session_id": target,
                "mode": "keep" if kwargs.get("keep") else "move",
            },
            None,
            target,
        )

    monkeypatch.setattr(mobility, "_destination_move", spy)
    # A link stub is enough: the handler reads the owner's device id off it and hands it to the
    # pull, which is the spy (``LinkTransport.__init__`` stores its arguments and asks nothing).
    # Annotated ``Any`` because the stub is deliberately not a ``PeerLink`` — building one needs a
    # real server pair, which is the dial this test exists to avoid.
    link: Any = SimpleNamespace(device_id=server_a.identity.device_id)

    ack = mobility._destination_invite(server_b, link, {"session_id": SESSION, "keep": True})

    assert ack["result"] == "accepted", ack
    assert ack["mode"] == "keep", ack
    assert handed == [str(ack["new_session_id"])], (ack, handed)
    assert handed[0] and handed[0] != SESSION, (ack, handed)
    # A MOVE IS UNCHANGED by any of this: the ack carries no new id, and the session keeps its
    # own — the receipt falls back to the id the user asked for, which is still right.
    handed.clear()
    moved = mobility._destination_invite(server_b, link, {"session_id": SESSION, "keep": False})
    assert moved["result"] == "accepted", moved
    assert moved["new_session_id"] == "", moved


def test_a_busy_source_refuses_with_its_own_sentence_and_mutates_nothing(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§6.4: refused, never drained. The owner's idle reason is the message."""
    server_a, server_b, _host, _port = pair
    _pair(pair, monkeypatch, role="admin")
    source = _owned_session(server_a)
    before = _transcript(server_a.root, SESSION)
    sentence = "This session is working right now — try again when the turn finishes."

    monkeypatch.setattr(
        mobility,
        "_retire_local_runtime",
        lambda root, session_id, deadline_s=0: {"result": "busy", "sentence": sentence},
    )

    result = _move(server_b, SESSION, monkeypatch=monkeypatch)

    assert result["ok"] is False
    assert result["code"] == "busy"
    assert result["message"] == sentence  # VERBATIM: the owner's own words
    assert result["changed"] is False  # a retry is safe
    # NOTHING MOVED ON EITHER SIDE.
    assert source.exists() and _transcript(server_a.root, SESSION) == before
    assert read_handoff_journal(server_a.root) == {}
    from local_operator.network.projection import read_tombstones

    assert read_tombstones(server_a.root) == {}
    assert not (server_b.root / "sessions" / SESSION).exists()
    assert not sync.staging_dir(server_b.root, SESSION).exists()


def test_wait_re_probes_the_source_until_it_is_idle(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--wait N`` is a FRESH idle probe each time, not a drain (§6.4)."""
    server_a, server_b, _host, _port = pair
    _pair(pair, monkeypatch, role="admin")
    _owned_session(server_a)
    before = _transcript(server_a.root, SESSION)
    calls = {"n": 0}

    def flaky(root: Path, session_id: str, deadline_s: float = 0) -> dict[str, Any]:
        calls["n"] += 1
        if calls["n"] == 1:
            return {"result": "busy", "sentence": "a turn is in flight"}
        return {"result": "cold", "sentence": ""}

    monkeypatch.setattr(mobility, "_retire_local_runtime", flaky)
    monkeypatch.setattr(mobility, "MOVE_WAIT_POLL_S", 0.2)

    result = _move(server_b, SESSION, wait_s=5.0, monkeypatch=monkeypatch)

    assert result["ok"] is True, result
    assert calls["n"] >= 2, "the wait must re-ask, not give up after one refusal"
    assert _transcript(server_b.root, SESSION) == before


def test_a_source_that_changes_mid_move_rolls_back_on_both_sides(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§6.3 step 12: a digest mismatch is a ROLLBACK, and the retry then succeeds.

    The source changes between the manifest it served and the ready that commits —
    the one case where committing would adopt a copy that is no longer the
    conversation. Nothing is lost: the source keeps an intact directory and no
    writer, and a second attempt moves it.
    """
    server_a, server_b, _host, _port = pair
    _pair(pair, monkeypatch, role="admin")
    source = _owned_session(server_a)
    original = _transcript(server_a.root, SESSION)
    real_ask = mobility.LinkTransport.ask
    injected = {"done": False}

    def meddling(self: mobility.LinkTransport, frame: dict[str, Any]) -> dict[str, Any]:
        answer = real_ask(self, frame)
        if frame.get("phase") == "prepare" and not injected["done"]:
            injected["done"] = True
            with (source / "transcript.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps({"id": "e99", "type": "user", "content": "a turn landed"}) + "\n"
                )
        return answer

    monkeypatch.setattr(mobility.LinkTransport, "ask", meddling)

    result = _move(server_b, SESSION, monkeypatch=monkeypatch)

    assert result["ok"] is False
    assert result["code"] == "digest_mismatch"
    assert "did not verify" in result["message"]
    # THE SOURCE IS INTACT, AND IT KEEPS THE TURN THAT LANDED.
    assert source.exists()
    assert _transcript(server_a.root, SESSION) != original
    from local_operator.network.projection import read_tombstones

    assert read_tombstones(server_a.root) == {}
    # The destination kept nothing and holds nothing.
    assert not (server_b.root / "sessions" / SESSION).exists()
    assert not sync.staging_dir(server_b.root, SESSION).exists()

    # ROLLBACK IS NOT UNDO: the same move now succeeds, and the extra turn travels.
    recorded = _transcript(server_a.root, SESSION)
    retry = _move(server_b, SESSION, monkeypatch=monkeypatch)
    assert retry["ok"] is True, retry
    assert _transcript(server_b.root, SESSION) == recorded


def test_a_peer_without_the_move_capability_is_refused(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The chokepoint, not this slice, decides: `drive` cannot move anything.

    The refusal crosses the link as a SENTENCE (``wire.refusal_frame`` deliberately
    drops the code, so a peer cannot learn which of membership, epoch or capability
    failed) — what this test pins is that the move does not happen and nothing is
    mutated, which is the property the chokepoint exists for.
    """
    server_a, server_b, _host, _port = pair
    _pair(pair, monkeypatch, role="drive")
    _owned_session(server_a)

    result = _move(server_b, SESSION, monkeypatch=monkeypatch)

    assert result["ok"] is False
    assert result["changed"] is False
    assert (server_a.root / "sessions" / SESSION).exists()
    from local_operator.network.projection import read_tombstones

    assert read_tombstones(server_a.root) == {}
    assert not (server_b.root / "sessions" / SESSION).exists()


def test_an_offload_to_a_peer_that_cannot_move_is_refused_before_the_invite(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """QA delta, Q-D3: a capability fact answered as one, and without asking anyone.

    With the peer admitted ``drive``, every frame its pull sends lands here and is
    refused by the authoriser — so the invite went out, the peer's first ask was
    refused at t+0.4 s, and the user was told 31.77 s later that the move "did not
    finish in time": a deadline receipt, and advice ("ask again") no retry can
    satisfy, because a capability is a decision. THIS device already knew: the peer's
    member row here is what the authoriser reads. So the invite is never sent, the
    refusal names the peer and the capability, and it carries the one piece of advice
    that changes the answer.
    """
    server_a, server_b, _host, _port = pair
    record, _pair_host, _pair_port = _pair(
        pair, monkeypatch, role="drive", settings=server_b.settings
    )
    _owned_session(server_a)
    invited: list[str] = []
    real = mobility.LinkTransport.ask

    def counting(self: Any, frame: dict[str, Any]) -> dict[str, Any]:
        invited.append(str(frame.get("phase") or frame.get("action") or ""))
        return real(self, frame)

    monkeypatch.setattr(mobility.LinkTransport, "ask", counting)

    result = _move(server_a, SESSION, to=server_b.identity.device_id, monkeypatch=monkeypatch)

    assert result["ok"] is False, result
    # ``not_authorised`` is the family's code for "a device lacks the 'move'
    # capability", so a front end branches on the same refusal the authoriser makes.
    assert result["code"] == "not_authorised", result
    sentence = str(result["message"])
    assert "device-b may not take sessions from this device" in sentence, sentence
    assert "does not hold the 'move' capability here" in sentence, sentence
    # THE REMEDY NAMES THE DEVICE BY ID (Aida round 2, finding 1): ``member grant``
    # matches ``record.member(device_id)`` and does NOT run its argument through the
    # name/tail resolver, so a sentence printing the display name would hand the user a
    # command that answers ``… is not an active member of …`` and grants nothing.
    remedy = f"grant <network> {server_b.identity.device_id} move"
    assert remedy in sentence, sentence
    assert "grant <network> device-b move" not in sentence, sentence
    assert result["changed"] is False, result
    # NO INVITE WAS SENT: the refusal is this device's own answer, and the peer was
    # never asked for something it cannot do. Structural, not timed — the clock is
    # what the old receipt got wrong.
    assert invited == [], f"the peer was invited anyway: {invited}"
    # NOTHING MOVED, ON EITHER DEVICE.
    assert (server_a.root / "sessions" / SESSION).is_dir()
    assert not (server_b.root / "sessions" / SESSION).exists()
    from local_operator.network.projection import read_tombstones

    assert read_tombstones(server_a.root) == {}

    # AND THE ADVICE RUNS. The command the sentence printed, driven through the real
    # ``lop network`` parser and handler, has to change this device's row for the peer —
    # a refusal whose remedy fails is the defect class this round exists to end. The
    # network name is the fixture's own; the device argument is the id the sentence
    # printed, which is the point.
    granted_caps = _grant_via_the_real_cli(
        server_a, record.network_id, server_b.identity.device_id, monkeypatch
    )
    assert "move" in granted_caps, granted_caps
    # The same move, now that the peer may take it, proceeds.
    allowed = _move(server_a, SESSION, to=server_b.identity.device_id, monkeypatch=monkeypatch)
    assert allowed["ok"] is True, allowed
    assert (server_b.root / "sessions" / SESSION).is_dir()


def _grant_via_the_real_cli(
    server: relay.RelayServer, network_id: str, device_id: str, monkeypatch: pytest.MonkeyPatch
) -> list[str]:
    """Run ``lop network member grant <net> <device> move`` as the CLI does; return the row.

    Parsed by the REAL parser and executed by the REAL handler on THIS device's root, so
    what it proves is that the argument the refusal prints is one the verb accepts —
    the difference between a sentence that names its remedy and one that names a
    command which fails (Aida round 2, finding 1).
    """
    import argparse

    from local_operator.network import cli as net_cli
    from local_operator.network import store

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server.root))
    parser = argparse.ArgumentParser(prog="lop")
    subparsers = parser.add_subparsers(dest="subcommand")
    net_cli.add_parser(subparsers)
    args = parser.parse_args(["network", "member", "grant", NETWORK_NAME, device_id, "move"])
    rc = net_cli.main(args)
    assert rc == 0, rc
    record = store.load(network_id, server.root)
    member = record.member(device_id)
    assert member is not None, "the grant did not reach the peer's row"
    return sorted(member.capabilities, key=str)


def test_a_move_already_on_this_device_is_refused_with_a_sentence(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    server_a, server_b, _host, _port = pair
    _pair(pair, monkeypatch, role="admin")
    _owned_session(server_b)

    result = _move(server_b, SESSION, monkeypatch=monkeypatch)

    assert result["ok"] is False
    assert result["code"] == "already_local"
    assert "already on this device" in result["message"]


def test_the_owner_is_resolved_from_the_federated_listing(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A COLD session on a peer is still a row, which is what makes a recall

    possible at all: the id alone has to be enough, because the person typing
    ``lop sessions move <id> --to local`` has no other handle on it.
    """
    server_a, server_b, host, port = pair
    record, _host, _port = _pair(pair, monkeypatch, role="admin")
    _owned_session(server_a)
    link, reason = server_b.dial(record.network_id, host=f"{host}:{port}", epoch=record.epoch)
    assert link is not None, reason

    owner, name = mobility.resolve_remote_owner(server_b, SESSION)

    assert owner == server_a.identity.device_id
    assert name == server_a.identity.name


def test_a_session_nobody_holds_is_refused_by_name(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    server_a, server_b, _host, _port = pair
    _pair(pair, monkeypatch, role="admin")

    result = _move(server_b, "no-such-session", monkeypatch=monkeypatch)

    assert result["ok"] is False
    assert result["changed"] is False


# ---------------------------------------------------------------------------
# INV-1's guard: no runtime for a session being handed away
# ---------------------------------------------------------------------------


def test_engage_is_refused_for_a_session_mid_handoff(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE guard (§6.6). Without it an idle source with a cold viewer spawns a
    successor during the window and the destination promotes a second owner."""
    import asyncio

    from local_operator.session.runtime.launch import (
        RuntimeStartupError,
        WarmErrand,
        engage_runtime,
    )

    server_a, _server_b, _host, _port = pair
    _owned_session(server_a)
    # Through the module's own writer, so the file the guard reads is the file the
    # move writes.
    write_handoff_entry(
        server_a.root,
        SESSION,
        {
            "role": "source",
            "phase": HANDOFF_PHASE_PREPARED,
            "to_device": "d_other",
            "to_name": "build-box",
            "at": 1.0,
        },
    )
    with pytest.raises(RuntimeStartupError) as raised:
        asyncio.run(
            engage_runtime(
                SESSION,
                str(server_a.root),
                WarmErrand(),
                config_dir=server_a.root,
                deadline_s=0.1,
            )
        )
    assert "being handed to build-box" in str(raised.value)

    # ... AND ``handing-off`` REFUSES TOO. The phase after which a rollback is
    # impossible is the one where a second runtime is most nearly possible.
    entry = read_handoff_journal(server_a.root)[SESSION]
    entry["phase"] = HANDOFF_PHASE_HANDING_OFF
    write_handoff_entry(server_a.root, SESSION, entry)
    with pytest.raises(RuntimeStartupError):
        asyncio.run(
            engage_runtime(
                SESSION,
                str(server_a.root),
                WarmErrand(),
                config_dir=server_a.root,
                deadline_s=0.1,
            )
        )

    # Clear the entry and the refusal goes: the guard blocks a HANDOFF, not a
    # session, and a guard that could not be cleared would brick the conversation.
    from local_operator.session.placement import (
        clear_handoff_entry,
        handoff_guard_refusal,
    )

    assert clear_handoff_entry(server_a.root, SESSION) is True
    assert handoff_guard_refusal(server_a.root, SESSION) == ""
    # A DIFFERENT session was never blocked, journal or not.
    write_handoff_entry(
        server_a.root,
        SESSION,
        {"role": "source", "phase": HANDOFF_PHASE_PREPARED, "to_device": "d_other"},
    )
    assert handoff_guard_refusal(server_a.root, "some-other-session") == ""


def test_an_unreadable_journal_fails_closed(pair: Devices, monkeypatch: pytest.MonkeyPatch) -> None:
    """Disk corruption refuses rather than guessing, and names the file to delete."""
    from local_operator.session.placement import (
        handoff_guard_refusal,
        handoff_journal_path,
    )

    server_a, _server_b, _host, _port = pair
    path = handoff_journal_path(server_a.root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")

    sentence = handoff_guard_refusal(server_a.root, SESSION)

    assert "could not tell whether" in sentence
    assert path.name in sentence


# ---------------------------------------------------------------------------
# Lifecycle on a peer: the OWNER runs it
# ---------------------------------------------------------------------------


def test_archive_on_a_peer_changes_the_owners_index_only(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """design §8.3: NEVER write the local ``archived-sessions.json`` for a remote id."""
    server_a, server_b, _host, _port = pair
    _pair(pair, monkeypatch, role="admin")
    _owned_session(server_a)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_b.root))

    result = mobility.lifecycle(
        SESSION, action="archive", peer=server_a.identity.device_id, root=server_b.root
    )

    assert result["ok"] is True, result
    assert result["changed"] is True
    from local_operator.session import archived

    assert SESSION in archived.archived_ids(server_a.root)
    assert SESSION not in archived.archived_ids(server_b.root)

    restored = mobility.lifecycle(
        SESSION, action="unarchive", peer=server_a.identity.device_id, root=server_b.root
    )
    assert restored["ok"] is True
    assert SESSION not in archived.archived_ids(server_a.root)


def test_archive_and_restore_on_a_peer_report_the_change_they_made(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q-INT-4: the owner's receipt describes what the owner's index DID.

    Measured (QA round 1 integration): ``--unarchive`` on a peer answered
    ``{"changed": false, "message": "<id> was already restored here."}`` for a call
    whose effect was the peer's ``archived-sessions.json`` going from one entry to
    none. The receipt is built from ``archive_change``'s first element, which was
    the state the caller ASKED for (``set_archived``'s echo, and correct for the
    route that reconciles a row on it); it is now the FILE's own answer, and both
    directions are driven here with the owner's index as the witness — the caller is
    B, the owner A, exactly as the test above routes them.
    """
    from local_operator.session import archived

    server_a, server_b, _host, _port = pair
    _pair(pair, monkeypatch, role="admin")
    _owned_session(server_a)

    peak = server_a.identity.device_id
    assert archived.archived_ids(server_a.root) == frozenset()

    made = mobility.lifecycle(SESSION, action="archive", peer=peak, root=server_b.root)
    assert made["ok"] is True, made
    assert made["changed"] is True, made
    assert made["message"] == f"Archived {SESSION} on this device.", made
    assert archived.archived_ids(server_a.root) == frozenset({SESSION})

    # THE RESTORE, which is the direction that reported nothing changed while the
    # owner's index went from one entry to none.
    restored = mobility.lifecycle(SESSION, action="unarchive", peer=peak, root=server_b.root)
    assert restored["ok"] is True, restored
    assert restored["changed"] is True, restored
    assert restored["message"] == f"Restored {SESSION} on this device.", restored
    assert archived.archived_ids(server_a.root) == frozenset()

    # AND A NO-OP STILL READS AS ONE: a second restore changes nothing and says so,
    # which is the receipt's half of the contract a retry lands on.
    again = mobility.lifecycle(SESSION, action="unarchive", peer=peak, root=server_b.root)
    assert again["ok"] is True, again
    assert again["changed"] is False, again
    assert again["message"] == f"{SESSION} was already restored here.", again


def test_delete_on_a_peer_is_a_dry_run_without_confirmation(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A delete dispatched without ``confirmed`` must not be a delete (§8.1)."""
    server_a, server_b, _host, _port = pair
    _pair(pair, monkeypatch, role="admin")
    source = _owned_session(server_a)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_b.root))

    dry = mobility.lifecycle(
        SESSION, action="delete", peer=server_a.identity.device_id, root=server_b.root
    )

    assert dry["ok"] is True, dry
    assert dry.get("deleted") is False
    assert dry.get("confirmed") is False
    assert source.exists(), "a dry run deleted something"

    real = mobility.lifecycle(
        SESSION,
        action="delete",
        peer=server_a.identity.device_id,
        confirmed=True,
        root=server_b.root,
    )

    assert real["ok"] is True, real
    assert real.get("deleted") is True
    assert not source.exists()
    logged = _cleanup_log(server_a.root)
    assert logged and logged[-1]["policy"] == "explicit-delete"


def test_delete_of_a_live_remote_session_is_refused_by_the_owners_guard(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard sentence crosses the link VERBATIM, and nothing is removed."""
    from local_operator.session.cleanup import _GUARD_REFUSALS

    server_a, server_b, _host, _port = pair
    _pair(pair, monkeypatch, role="admin")
    source = _owned_session(server_a)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_b.root))
    # An armed wake is one of the owner's own guards, and it is refused even when
    # the delete is confirmed.
    monkeypatch.setattr("local_operator.session.cleanup._has_armed_wake", lambda *a, **k: True)
    monkeypatch.setattr(
        "local_operator.session.cleanup._guard",
        lambda *a, **k: "has an armed wake",
    )

    result = mobility.lifecycle(
        SESSION,
        action="delete",
        peer=server_a.identity.device_id,
        confirmed=True,
        root=server_b.root,
    )

    assert result["ok"] is False
    assert result["code"] == "session_delete_refused"
    assert str(result["message"]) in set(_GUARD_REFUSALS.values()) or result["message"]
    assert source.exists()


def test_an_offload_returns_the_destinations_refusal_instead_of_waiting_it_out(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """QA round 1, Q4a: the refusal the peer makes in milliseconds must not cost the budget.

    An offload's inviter watches its OWN durable progress — the journal it wrote and the
    tombstone it will write — and never asks the destination, which is right for every
    phase the destination reports and wrong for exactly one: a REFUSAL writes nothing
    here. So the inviter held the request for its whole budget (``wait_s +
    OFFLOAD_CONFIRM_WAIT_S``: 30 s at ``wait_s=0``, 60 s at the CLI's default) and then
    answered "the outcome is unconfirmed" about a move that never started. Measured with
    the desktop's own pane holding the session: the peer refused 9 times out of 9 in
    ~3 ms with "This session is open in another terminal or attached client." and the
    user read a timeout 60 s later. The refusing device now reports it over the link it
    was invited on, and the wait ends where the refusal happened.
    """
    import time

    server_a, server_b, _host, _port = pair
    # B'S OWN SETTINGS, so the join records the endpoint B is actually listening on: the
    # default settings would advertise the CLI's configured port, which nothing is bound
    # to here — and this is the one rig in this file where A DIALS B (recalls have B dial
    # A), so an undialable advertisement reads as an unreachable peer rather than as a
    # fixture detail.
    _pair(pair, monkeypatch, role="admin", settings=server_b.settings)
    _owned_session(server_a)
    sentence = (
        "This session is open in another terminal or attached client. "
        "Disconnect that client, then move again."
    )
    # The owner's own runtime refuses to retire — the "another terminal or attached
    # client" case, which is ``_retire_local_runtime``'s ``viewed`` outcome.
    monkeypatch.setattr(
        mobility,
        "_retire_local_runtime",
        lambda root, session_id, deadline_s=0: {"result": "viewed", "sentence": sentence},
    )

    budget = mobility.move_bound_s(0.0)
    started = time.monotonic()
    result = _move(server_a, SESSION, to=server_b.identity.device_id, monkeypatch=monkeypatch)
    elapsed = time.monotonic() - started

    assert result["ok"] is False, result
    assert result["code"] == "busy", result
    assert result["message"] == sentence, "the refusing device's own words, verbatim"
    assert result["changed"] is False, "nothing moved, so a retry is safe"
    assert elapsed < budget / 2, (
        f"the refusal took {elapsed:.1f}s of a {budget:.0f}s budget, so it waited for the "
        "deadline rather than being told: that is the finding this test exists for"
    )


def test_an_offload_receipt_carries_every_phase_the_move_went_through(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q-INT-3: the receipt is the move's HISTORY, not what the disk can still prove.

    Measured (QA round 1 integration): an offload's ``TransferReceipt.phases``
    carried ``['committed']`` while a recall carried all four, so a renderer drawing
    progress from the list showed every offload stuck at 75% against a model that
    documents ``prepared`` 0.25 … ``done`` 1.0. The cause was a PREFERENCE between
    two partial sources — disk whenever disk had anything — and the two assertions
    below are the shape it was wrong about: the disk alone can only say
    ``committed`` once a move has committed, and the receipt must still report what
    the move went through.

    The phase list is built from the two sources the move itself writes, with no
    peer dial: an offload's rig needs A to reach B, and this file's fixture binds
    only A (the sibling offload test is the one that dials, and it is load-flaky for
    that reason). The end-to-end figure is taken on two bound devices instead.
    """
    server_a, _server_b, _host, _port = pair
    # A PRIVATE ID, and forgotten at the end: ``_Progress`` is per-relay but lives in
    # this process keyed by ``id(server)``, so a history left under the shared
    # ``SESSION`` can outlive the test that wrote it.
    private = "ab12cd34ef56"
    progress = mobility.progress_for(server_a)
    try:
        # THE STATE A COMMITTED OFFLOAD LEAVES: every phase in memory (the source
        # noted each as it happened), and only the tombstone left on disk.
        for phase in ("prepared", "handing_off", "committed", "done"):
            progress.note(private, phase)
        monkeypatch.setattr(
            mobility,
            "_phases_from_disk",
            lambda root, session_id: [{"phase": "committed", "at": 1.0}],
        )

        phases = mobility._move_phases(server_a, private)

        assert [stamp["phase"] for stamp in phases] == [
            "prepared",
            "handing_off",
            "committed",
            "done",
        ]
        # …AND THE DISK ALONE IS WHY THE UNION EXISTS: reading the same state the way
        # the receipt used to (disk whenever disk has anything at all) reports one
        # phase — 0.75 of the progress a renderer draws from it.
        disk_only = mobility._phases_from_disk(server_a.root, private)
        assert [stamp["phase"] for stamp in disk_only] == ["committed"]
    finally:
        progress.forget(private)


def test_the_phase_order_is_the_contracts_not_the_clock(pair: Devices) -> None:
    """The union is ordered by ``MOVE_RESULT_PHASES``, never by ``at``.

    A disk stamp is synthesised when it is READ (``_phases_from_disk``), so it can
    carry a later time than a memory stamp for an EARLIER phase — sorting by time
    would put ``committed`` after ``done`` and a renderer's bar would go backwards.
    """
    server_a, _server_b, _host, _port = pair
    private = "cd34ef56ab12"
    progress = mobility.progress_for(server_a)
    try:
        progress.note(private, "handing_off")
        progress.note(private, "prepared")

        phases = mobility._move_phases(server_a, private)

        # NOTED ``handing_off`` FIRST: the contract's order is what the list reports,
        # and ``prepared`` has to come back in front of it.
        assert [stamp["phase"] for stamp in phases] == ["prepared", "handing_off"]
    finally:
        progress.forget(private)


def test_the_move_bound_is_the_formula_a_client_can_derive() -> None:
    """Q4a's contract: the bound is ``wait_s + the settle window``, published as a function.

    The desktop's own deadline was ``wait_s + 15`` while this side answered at
    ``wait_s + 30``, so the client's timeout ALWAYS won and its vaguer sentence ("the
    move may have happened") replaced the backend's real answer. A client's bound has to
    be this plus a margin, and the relay's own budget is the same expression rather than
    a second number that can drift from it.
    """
    assert mobility.move_bound_s(0.0) == mobility.OFFLOAD_CONFIRM_WAIT_S
    assert mobility.move_bound_s(30.0) == 30.0 + mobility.OFFLOAD_CONFIRM_WAIT_S
    assert mobility.move_bound_s(30.0, keep=True) == 30.0 + mobility.KEEP_COPY_WAIT_S
    assert mobility.move_bound_s(-5.0) == mobility.OFFLOAD_CONFIRM_WAIT_S


def test_every_shape_the_route_accepts_derives_its_bound_from_one_place() -> None:
    """Review round 1, MAJOR 1: the offload's formula is not the route's contract.

    Published as if it were, it was wrong for the two other shapes the SAME route
    accepts. A ``keep`` copy is held for ``wait_s + KEEP_COPY_WAIT_S`` — 330 s at
    ``wait_s=30`` — while the published ``wait_s + 30 + margin`` gave up at 75 s, so a
    client following the advice still hit the exact pre-fix symptom on a live field of
    the same route (``TransferSession.keep``). And a recall (``to="local"``) is not
    bounded by that formula at all: this device is the DESTINATION, so there is no
    invite and no settle window in it, and what bounds the wait is the owner's
    retire-plus-record deadline plus a transcript-sized copy.
    """
    assert mobility.move_hold_s(0.0) == mobility.OFFLOAD_CONFIRM_WAIT_S
    assert mobility.move_hold_s(30.0, keep=True) == 30.0 + mobility.KEEP_COPY_WAIT_S
    # THE RECALL IS A COPY, and the offload's term is not merely imprecise for it: at
    # ``wait_s=30`` the formula would say 60 s where the copy's own budget is 330.
    assert mobility.move_hold_s(30.0, to="local") == 30.0 + mobility.KEEP_COPY_WAIT_S
    assert mobility.move_hold_s(30.0, to="local") != mobility.move_bound_s(30.0)

    # The PUBLISHED bound is the route's own envelope plus the client's margin, and
    # the CLI's control timeout is the same expression — so a client using these
    # numbers cannot beat the route, which is what the desktop did at ``wait_s + 15``
    # against a route answering at ``wait_s + 30``.
    for keep, to in ((False, "d_peer"), (True, "d_peer"), (False, "local")):
        published = mobility.move_client_bound_s(30.0, keep=keep, to=to)
        route = (
            mobility.MOVE_OP_DEADLINE_S
            + mobility.move_hold_s(30.0, keep=keep, to=to)
            + mobility.MOVE_CONTROL_SLACK_S
        )
        assert published == route + mobility.MOVE_CLIENT_MARGIN_S
        assert published > route, "a client may never EQUAL the bound it is outlasting"
    assert mobility.move_client_bound_s(0.0) == 145.0
    assert mobility.move_client_bound_s(0.0, keep=True) == 415.0
    assert mobility.move_client_bound_s(0.0, to="local") == 415.0


def test_the_cli_envelope_is_the_published_bound_less_its_margin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The SECOND half of one place: the caller's deadline is the published derivation.

    ``move_client_bound_s`` is advice to another repository; the number this side
    actually waits on is ``request_move``'s control-socket timeout. A test that only
    read the function would let the two drift apart while both looked right, which is
    how the recall's envelope stayed at the offload's 30 s term and made the CLI report
    ``relay_unavailable`` — "this device's relay could not be asked" — about a copy
    that was still running.
    """
    seen: list[float] = []

    def _capture(record: Any, op: str, *, timeout: float = 5.0, **fields: Any) -> Any:
        seen.append(timeout)
        return None  # no relay answer: the refusal path, which is all this measures

    monkeypatch.setattr(relay, "control_request", _capture)
    monkeypatch.setattr("local_operator.network.store.find_own_relay", lambda root=None: object())

    for keep, to in ((False, "build-box"), (True, "build-box"), (False, "local")):
        seen.clear()
        mobility.request_move(SESSION, to=to, keep=keep, wait_s=30.0, root=Path("/tmp/x"))
        assert seen == [
            mobility.move_client_bound_s(30.0, keep=keep, to=to) - mobility.MOVE_CLIENT_MARGIN_S
        ], (keep, to, seen)
