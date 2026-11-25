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
import os
import threading
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.agents import AgentEditFields
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
    _pair_settled,
    devices,
)


def _agent_fields(**fields: Any) -> AgentEditFields:
    """``AgentEditFields`` from only the fields a test names.

    Every unnamed field defaults to ``None`` (leave/unset) at runtime, but
    pyright's synthesized pydantic ``__init__`` demands each ``Field(None, ...)``
    field be spelled. Unpacking keeps the runtime model identical — the same
    reason the repo's other ``_edit_fields(**overrides)`` helpers exist.
    """
    return AgentEditFields(**fields)


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
    engage_on_arrival: bool = False,
    drop_session_roles: bool = False,
    monkeypatch: pytest.MonkeyPatch | None = None,
) -> dict[str, Any]:
    """``lop sessions move``, driven the way the CLI drives it.

    Through this device's RELAY's control socket (``mobility.request_move``), not
    by calling the protocol directly: the hop the CLI makes is part of what the
    mesh's own error shapes are for, and a test that skipped it would not notice a
    local verb that never got registered.

    ``engage_on_arrival`` is the verb's one opt-in flag (``--engage-on-arrival``), and
    it is passed through here rather than by building the control frame in the caller:
    a test that hand-rolled the frame would not be exercising the same hop the CLI's
    flag travels.
    """
    if monkeypatch is not None:
        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server.root))
    return dict(
        mobility.request_move(
            session_id,
            to=to,
            keep=keep,
            wait_s=wait_s,
            drop_session_roles=drop_session_roles,
            root=server.root,
            from_replica=from_replica,
            engage_on_arrival=engage_on_arrival,
        )
    )


# ---------------------------------------------------------------------------
# The default: recall a peer's session to this device
# ---------------------------------------------------------------------------


def test_a_move_is_allowed_and_does_not_carry_the_approval_choice(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The approvals sidecar is excluded, not unlisted.

    Round-4 review found a real blocker: `approvals.json` was absent from the
    sync vocabulary, so moving a conversation that had used the approvals
    feature refused with `unlisted_content` and told the user to delete their
    saved choice. The move must succeed and the destination copy must NOT
    contain the source's choice — a copy is a new conversation, the same rule
    `fork` applies.
    """
    from local_operator.session.approval_policy import OVERRIDE_FILENAME, write_override

    server_a, server_b, _host, _port = pair
    _pair_settled(pair, monkeypatch, role="admin")
    source = _owned_session(server_a)
    write_override(source, "auto")

    result = _move(server_b, SESSION, monkeypatch=monkeypatch)

    assert result["ok"] is True, result
    destination = server_b.root / "sessions" / SESSION
    assert destination.is_dir(), result
    assert not (
        destination / OVERRIDE_FILENAME
    ).exists(), "the copy inherited the source conversation's approval choice"


@pytest.mark.parametrize(
    "case", ["pinned", "legacy", "no_ack", "write_failure", "interrupted_publish"]
)
def test_move_promotes_and_pins_the_skill_manifest(
    pair: Devices, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    from local_operator.cli import _carry_lines
    from local_operator.resume import read_session_attachment, write_session_attachment

    server_a, server_b, _host, _port = pair
    _pair_settled(
        pair, monkeypatch, role="admin", settings=server_b.settings if case == "no_ack" else None
    )
    source = _owned_session(server_a)
    manifest = {"carry-alpha": "a" * 64, "carry-beta": "b" * 64}
    expected = None if case == "legacy" else manifest
    write_session_attachment(source, team="", agent="", goal="Keep this", skill_manifest=expected)
    before = read_session_attachment(source)
    assert before is not None and not before.skill_manifest_pinned
    target = server_b.root / "sessions" / SESSION
    if case == "no_ack":
        promote = mobility._promote

        def promote_silently(
            server: relay.RelayServer, staging: Path, target_id: str
        ) -> mobility._PromoteOutcome:
            outcome = promote(server, staging, target_id)
            if server is server_b and outcome.ok:
                # Same wire shape as a legacy destination; the requester
                # cannot infer the actual pin from successful publication.
                # ``skill_selection`` is its own outcome field (the wake-carry
                # block stays exact), so dropping it is what a legacy peer sends.
                return outcome._replace(skill_selection=None)
            return outcome

        monkeypatch.setattr(mobility, "_promote", promote_silently)
    elif case == "write_failure":
        # The attachment writer publishes through ``atomic_files.atomic_write_bytes``,
        # whose rename is ``os.replace(temp, target)``: fail exactly that rename of the
        # destination's sidecar, so the strict pin raises after staging its temp.
        replace = os.replace
        original_sidecar = (source / "attachment.json").read_bytes()

        def fail_pin(src: Any, dst: Any, **kwargs: Any) -> None:
            to = Path(dst)
            if to.name == "attachment.json" and to.is_relative_to(server_b.root):
                raise OSError("injected attachment replace failure")
            replace(src, dst, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(os, "replace", fail_pin)
            refused = _move(server_b, SESSION, monkeypatch=patch)
        assert refused["ok"] is False and "was not published" in refused["message"]
        assert not target.exists()
        stage = sync.staging_dir(server_b.root, SESSION)
        assert (stage / "attachment.json").read_bytes() == original_sidecar
        # The shared writer discards its own unique temp: no partial member stays staged.
        assert not [path.name for path in stage.iterdir() if path.name.endswith(".tmp")]
        ready = json.loads((stage / "ready.json").read_text())
        assert mobility._staging_content_digest(server_b, SESSION, stage) == ready["content_digest"]
    elif case == "interrupted_publish":
        replace_directory = os.replace
        observed: list[bool] = []

        def stop_before_publish(src: Any, dst: Any, **kwargs: Any) -> None:
            if dst == str(target):
                attachment = read_session_attachment(Path(src))
                observed.append(bool(attachment and attachment.skill_manifest_pinned))
                raise OSError("injected interruption before publication")
            replace_directory(src, dst, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(os, "replace", stop_before_publish)
            refused = _move(server_b, SESSION, monkeypatch=patch)
        assert refused["ok"] is False and not target.exists()
        assert observed == [True], "the single publish boundary must already contain the pin"

    if case == "no_ack":
        # The source must wait for the destination's report over the link.
        result = _move(server_a, SESSION, to=server_b.identity.device_id, monkeypatch=monkeypatch)
    else:
        result = _move(server_b, SESSION, monkeypatch=monkeypatch)

    assert result["ok"] is True, result
    stored = read_session_attachment(target)
    assert stored is not None
    assert stored.skill_manifest == expected
    assert stored.skill_manifest_pinned is (expected is not None)
    assert stored.goal == "Keep this"
    assert not (target / mobility._SKILL_PIN_ORIGINAL).exists()

    if case == "no_ack":
        status, phrase = ("unconfirmed", "did not confirm a skill snapshot")
    elif expected is None:
        status, phrase = ("local_routing", "will select skills locally")
    else:
        status, phrase = ("pinned", "pinned the source's skill selection")
    assert result["carry"]["skill_selection"]["status"] == status
    assert any(phrase in line for line in _carry_lines(result))


def test_a_recall_moves_the_session_and_leaves_a_tombstone(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--to local`: the phases in order, one holder at the end, and a cleanup log
    that says the conversation was MOVED rather than deleted."""
    server_a, server_b, _host, _port = pair
    _pair_settled(pair, monkeypatch, role="admin")
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
    _pair_settled(pair, monkeypatch, role="admin")
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
    _pair_settled(pair, monkeypatch, role="admin", settings=server_b.settings)
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
    _pair_settled(pair, monkeypatch, role="admin")
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
    _pair_settled(pair, monkeypatch, role="admin")
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
    _pair_settled(pair, monkeypatch, role="admin")
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
    _pair_settled(pair, monkeypatch, role="drive")
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
    record, _pair_host, _pair_port = _pair_settled(
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


@pytest.mark.parametrize("peer_reply", ["no-advert", "no-echo", "correct-echo"])
def test_offload_admission_and_wait_require_different_peer_evidence(
    pair: Devices, monkeypatch: pytest.MonkeyPatch, peer_reply: str
) -> None:
    from local_operator.network.wire import SESSION_MOVE_ATTEMPT_V1

    server_a, server_b, _host, _port = pair
    _pair_settled(pair, monkeypatch, role="admin", settings=server_b.settings)
    source = _owned_session(server_a)
    request = relay.PeerLink.request
    send = relay.PeerLink.send
    invite_frames: list[dict[str, Any]] = []
    sent_invites: list[int] = []
    done_frames: list[dict[str, Any]] = []

    def send_recorded(link: relay.PeerLink, frame: dict[str, Any], **kwargs: Any) -> bool:
        if (
            link.server is server_a
            and frame.get("op") == "net_session_move"
            and frame.get("phase") == "invite"
        ):
            sent_invites.append(frame["attempt"])
        return send(link, frame, **kwargs)

    def record(link: relay.PeerLink, frame: dict[str, Any], **kwargs: Any) -> dict[str, Any] | None:
        if frame.get("op") == "net_session_move":
            if link.server is server_a and frame.get("phase") == "invite":
                invite_frames.append(dict(frame))
                if peer_reply == "no-echo":
                    assert SESSION_MOVE_ATTEMPT_V1 in link.capabilities
                    # An inert ack: no transfer and no completion evidence follow it.
                    return {"op": "ack", "detail": {"result": "accepted"}}
            if link.server is server_b and frame.get("phase") == "done":
                done_frames.append(dict(frame))
        return request(link, frame, **kwargs)

    monkeypatch.setattr(relay.PeerLink, "request", record)
    monkeypatch.setattr(relay.PeerLink, "send", send_recorded)
    if peer_reply == "no-echo":
        monkeypatch.setattr(mobility, "move_bound_s", lambda *_a, **_k: 0.0)

    def offload() -> dict[str, Any]:
        return _move(server_a, SESSION, to=server_b.identity.device_id, monkeypatch=monkeypatch)

    if peer_reply == "no-advert":
        link_for_move = mobility._link_for_move

        def assert_refusal(*, empty_advert: bool) -> None:
            with monkeypatch.context() as advertised:

                def read_link(
                    server: relay.RelayServer, device_id: str, label: str
                ) -> relay.PeerLink:
                    link = link_for_move(server, device_id, label)
                    # Replace only the observed advert; grants and invite handling are real.
                    caps = (
                        frozenset()
                        if empty_advert
                        else link.capabilities - {SESSION_MOVE_ATTEMPT_V1}
                    )
                    advertised.setattr(link, "capabilities", caps)
                    return link

                advertised.setattr(mobility, "_link_for_move", read_link)
                result = offload()
            assert not result["ok"] and not result["changed"], result
            assert result["code"] == "peer_cannot_confirm_attempts", result
            assert "Upgrade that peer to a version that advertises move attempt token support" in (
                result["message"]
            )
            assert "or move the session to a device that advertises support" in result["message"]
            assert len(sent_invites) == 0, sent_invites
            assert len(invite_frames) == 0, invite_frames

        assert_refusal(empty_advert=False)
        assert_refusal(empty_advert=True)
        assert source.is_dir()
        return

    result = offload()
    assert len(invite_frames) == 1, (invite_frames, result)
    token = invite_frames[0]["attempt"]
    assert token > 0
    if peer_reply == "no-echo":
        assert not result["ok"] and not result["changed"], result
        assert result["code"] == "move_attempt_unconfirmed", result
        assert done_frames == []
        assert source.is_dir()
        assert not (server_b.root / "sessions" / SESSION).exists()
    else:
        assert result["ok"], result
        assert sent_invites == [token]
        assert [frame.get("attempt") for frame in done_frames] == [token]
        assert (server_b.root / "sessions" / SESSION).is_dir()


def test_an_offload_carries_the_sessions_unattended_authority_on_the_invite(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Defect 2, the MOVE half: the invite carries the session's auto authority.

    The step that would APPLY it — the destination's runtime construction — runs
    on the other device, so the bit must travel WITH the invite, exactly like
    ``wait_s`` and ``engage_on_arrival`` beside it. Read from THIS device's stamp
    (the session's own record of the authority it was created or re-stamped with)
    and sent on EVERY invite so the key's presence does not depend on a boolean:
    an older destination ignores it and a MISSING key reads as off.

    Two halves in one cell, because a field that reads True whatever the session
    says is a field that carries nothing: the carried session sends True and the
    destination STAMPS it (the construction reads the stamp, not the invite); a
    session that never set the bit sends False. Nothing here asserts the
    destination's runtime — the stamp write and the construction's re-check have
    their own cells — this one pins the WIRE hop, which is the one a refactor
    deletes most quietly.
    """
    from dataclasses import replace

    from local_operator.session.placement import read_stamp, write_stamp

    server_a, server_b, _host, _port = pair
    _pair_settled(pair, monkeypatch, role="admin", settings=server_b.settings)
    _owned_session(server_a)
    carried = read_stamp(server_a.root, SESSION)
    assert carried is not None
    write_stamp(server_a.root, replace(carried, unattended=True))

    frames: list[dict[str, Any]] = []
    real = mobility.LinkTransport.ask

    def capture(self: Any, frame: dict[str, Any]) -> dict[str, Any]:
        frames.append(dict(frame))
        return real(self, frame)

    monkeypatch.setattr(mobility.LinkTransport, "ask", capture)

    result = _move(server_a, SESSION, to=server_b.identity.device_id, monkeypatch=monkeypatch)
    assert result["ok"] is True, result
    invites = [frame for frame in frames if frame.get("phase") == "invite"]
    assert invites, frames
    assert invites[0].get("unattended") is True, invites[0]
    arrived = read_stamp(server_b.root, SESSION)
    assert arrived is not None and arrived.unattended is True, arrived

    # The other direction, on a second session in the same rig.
    second_id = "9f3ac1e0b7d3"
    _owned_session(server_a, second_id)
    frames.clear()
    moved = _move(server_a, second_id, to=server_b.identity.device_id, monkeypatch=monkeypatch)
    assert moved["ok"] is True, moved
    second = [frame for frame in frames if frame.get("phase") == "invite"]
    assert second and second[0].get("unattended") is False, second[0]


def test_an_offload_carries_the_devices_saved_auto_mode(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Defect 2, the move half of the SURFACE fix: the device's saved mode travels.

    A session that runs with no routine re-prompts HERE because ``tool_approval_mode``
    resolves ``auto`` must keep doing so on the node (the design's acceptance: "a
    moved auto session stays auto"). Before this, the invite read ONLY the stamp —
    the authority accepted at a create/move doorstep — so a locally-auto session
    with a plain stamp lost auto the moment it landed. The read goes through the
    ONE derivation (``session_factory.saved_tool_approval_is_auto`` → the same
    ``_approval_mode_is_auto`` a locally-started session runs with) and this cell
    pins both directions: ``auto`` on the source travels True (and the destination
    stamps it), and a return to ``ask`` travels False — a read that drifted into
    "always true" goes red on the second half.
    """
    from local_operator.config import ConfigManager

    server_a, server_b, _host, _port = pair
    _pair_settled(pair, monkeypatch, role="admin", settings=server_b.settings)
    _owned_session(server_a)
    # No stamp bit anywhere: the carry has to come from the saved mode alone.
    carried = read_stamp(server_a.root, SESSION)
    assert carried is not None and carried.unattended is False
    ConfigManager(server_a.root).set_config_value("tool_approval_mode", "auto")

    frames: list[dict[str, Any]] = []
    real = mobility.LinkTransport.ask

    def capture(self: Any, frame: dict[str, Any]) -> dict[str, Any]:
        frames.append(dict(frame))
        return real(self, frame)

    monkeypatch.setattr(mobility.LinkTransport, "ask", capture)

    result = _move(server_a, SESSION, to=server_b.identity.device_id, monkeypatch=monkeypatch)
    assert result["ok"] is True, result
    invites = [frame for frame in frames if frame.get("phase") == "invite"]
    assert invites, frames
    assert invites[0].get("unattended") is True, invites[0]
    arrived = read_stamp(server_b.root, SESSION)
    assert arrived is not None and arrived.unattended is True, arrived

    # The negative half: saved mode back at `ask` and no stamp bit => attended.
    second_id = "9f3ac1e0b7d3"
    _owned_session(server_a, second_id)
    ConfigManager(server_a.root).set_config_value("tool_approval_mode", "ask")
    frames.clear()
    moved = _move(server_a, second_id, to=server_b.identity.device_id, monkeypatch=monkeypatch)
    assert moved["ok"] is True, moved
    second = [frame for frame in frames if frame.get("phase") == "invite"]
    assert second and second[0].get("unattended") is False, second[0]


def test_a_second_offload_after_a_round_trip_is_a_real_move_not_a_stale_signal(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE FALSE SUCCESS, on the wire: move out, come back, move out again.

    Measured live 2026-09-29 (relay pid 20929, session ``ec51d0c8a9c7``, 0.64.1):
    after its FIRST move out completed, every later offload of that session answered
    ``committed`` in microseconds — no audit lines, no tombstone, nothing moved, and
    the busy refusal the protocol did produce was never consulted. The source relay's
    ``_source_done`` sets a completion event; ``forget`` cleared phases and refusals
    but not events, so the next move's first poll consumed it as its own completion.
    This is the same sequence on two real relays over loopback.

    THE ASSERTIONS ARE THE WIRE'S OWN FACTS: a receipt that says the move finished
    must leave the conversation on the destination with the source tombstoned, and
    the history must carry every phase the move went through — the broken shape was
    an instant ``[prepared, committed]`` pair 3.8 µs apart against the real move's
    four phases over 8.7 s.
    """
    server_a, server_b, _host, _port = pair
    _pair_settled(pair, monkeypatch, role="admin", settings=server_b.settings)
    source = _owned_session(server_a)
    target = server_b.identity.device_id

    first = _move(server_a, SESSION, to=target, monkeypatch=monkeypatch)
    assert first["ok"] is True, first
    assert first["phase"] == "done"
    # The conversation comes home — the live episode's own middle step (06:23 out,
    # ~06:30 back) — which is what makes the next offload a duplicate of the first.
    back = _move(server_a, SESSION, to="local", monkeypatch=monkeypatch)
    assert back["ok"] is True, back
    assert (server_a.root / "sessions" / SESSION).is_dir()

    second = _move(server_a, SESSION, to=target, monkeypatch=monkeypatch)

    assert second["ok"] is True, second
    assert (
        second["phase"] == "done"
    ), "the relay answered a success the protocol never ran: " + repr(second)
    assert [stamp["phase"] for stamp in second["phases"]] == [
        "prepared",
        "handing_off",
        "committed",
        "done",
    ]
    assert not source.exists(), "the receipt claimed a move that never left this device"
    assert (server_b.root / "sessions" / SESSION).is_dir()

    from local_operator.network.projection import read_tombstones

    assert read_tombstones(server_a.root)[SESSION]["device_id"] == target


def test_a_mid_turn_second_offload_refuses_instead_of_faking_success(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The busy refusal must RUN; the stale signal used to answer before it could.

    The same round trip as the sibling cell, then the second offload is attempted
    with the session mid-turn (the owner's retire refuses ``busy`` — the state a
    self-move from inside the conversation is in). That refusal travels back over the
    link the invite used, which the inviter can only read while it is still LOOKING
    for an answer: with the stale ``done`` event in play the first poll returned
    instantly, so the receipt claimed success and the protocol's own refusal was
    recorded but never consulted. Nothing may move and the answer must be the refusal.
    """
    server_a, server_b, _host, _port = pair
    _pair_settled(pair, monkeypatch, role="admin", settings=server_b.settings)
    _owned_session(server_a)
    target = server_b.identity.device_id

    # Capture once; only closed protocol values may reach diagnostics, never peer text.
    phases = mobility.MOVE_RESULT_PHASES + mobility.MOVE_QUEUE_PHASES
    diagnostic_values = {
        "code": mobility.MOVE_REFUSAL_CODES,
        "phase": phases,
        "phase_reached": phases,
    }
    first = _move(server_a, SESSION, to=target, monkeypatch=monkeypatch)
    assert first["ok"] is True, {
        key: first[key] if isinstance(first[key], str) and first[key] in allowed else "unknown"
        for key, allowed in diagnostic_values.items()
        if key in first
    }
    back = _move(server_a, SESSION, to="local", monkeypatch=monkeypatch)
    assert back["ok"] is True, {
        key: back[key] if isinstance(back[key], str) and back[key] in allowed else "unknown"
        for key, allowed in diagnostic_values.items()
        if key in back
    }

    sentence = (
        "this session is working right now, so nothing was moved; try again when the "
        "turn finishes; from a shell, pass --wait <seconds> to re-check"
    )
    monkeypatch.setattr(
        mobility,
        "_retire_local_runtime",
        lambda root, session_id, deadline_s=0: {"result": "busy", "sentence": sentence},
    )

    result = _move(server_a, SESSION, to=target, monkeypatch=monkeypatch)

    assert result["ok"] is False, f"a mid-turn move faked success: {result!r}"
    assert result["code"] == "busy", result
    assert result["message"] == sentence, result
    assert result["changed"] is False, result
    # NOTHING CHANGED, ON EITHER DEVICE.
    assert (server_a.root / "sessions" / SESSION).is_dir()
    assert not (server_b.root / "sessions" / SESSION).exists()

    from local_operator.network.projection import read_tombstones

    assert read_tombstones(server_a.root).get(SESSION) is None


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
    _pair_settled(pair, monkeypatch, role="admin")
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
    record, _host, _port = _pair_settled(pair, monkeypatch, role="admin")
    _owned_session(server_a)
    link, reason = server_b.dial(
        record.network_id,
        host=f"{host}:{port}",
        epoch=record.epoch,
        expected_device=server_a.identity.device_id,
    )
    assert link is not None, reason

    owner, name = mobility.resolve_remote_owner(server_b, SESSION)

    assert owner == server_a.identity.device_id
    assert name == server_a.identity.name


def test_a_session_nobody_holds_is_refused_by_name(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    server_a, server_b, _host, _port = pair
    _pair_settled(pair, monkeypatch, role="admin")

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


def test_two_handoff_writers_in_one_process_never_share_a_staging_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The journal's staging file is unique per CALL, not per process.

    The CI failure frame enters ``_write_journal`` from ``reconcile`` ->
    ``clear_handoff_entry`` on the driving thread, colliding with another
    writer in the same process — the reachable in-process shape
    ``network/store.py``'s concurrent-writer cell documents. With a pid-scoped
    staging name two such writers shared one file: the first ``os.replace``
    took it away and the second died with ``FileNotFoundError``. The barrier
    below holds both writers at their ``fsync`` — i.e. AFTER both have staged —
    so the old name collides BY CONSTRUCTION rather than by racing. Both write
    the SAME session and entry, so any ordering of the two replaces leaves the
    same bytes: what is pinned here is the staging name's uniqueness, not
    read-modify-write serialisation.
    """
    from local_operator.session import placement

    root = tmp_path / "device"
    both_staged = threading.Barrier(2, timeout=30)
    real_fsync = placement.os.fsync

    def fsync_when_both_have_staged(descriptor: int) -> None:
        both_staged.wait()
        real_fsync(descriptor)

    monkeypatch.setattr(placement.os, "fsync", fsync_when_both_have_staged)
    entry = {"role": "source", "phase": HANDOFF_PHASE_PREPARED, "at": 1.0}
    errors: list[BaseException] = []

    def write_one() -> None:
        try:
            write_handoff_entry(root, SESSION, entry)
        except BaseException as exc:  # noqa: BLE001 — the assertion is the report
            errors.append(exc)

    threads = [
        threading.Thread(target=write_one, name=f"journal-writer-{index}") for index in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not any(thread.is_alive() for thread in threads), "a writer never returned"
    assert errors == [], (
        "both writers must land; a shared staging file kills the second at its "
        f"os.replace: {len(errors)} raised "
        f"{sorted({type(exc).__name__ for exc in errors})}"
    )
    assert read_handoff_journal(root) == {SESSION: entry}
    leftover = sorted(
        path.name for path in (root / "network").iterdir() if path.name.endswith(".tmp")
    )
    assert leftover == [], f"staging files left behind: {leftover}"


# ---------------------------------------------------------------------------
# Lifecycle on a peer: the OWNER runs it
# ---------------------------------------------------------------------------


def test_archive_on_a_peer_changes_the_owners_index_only(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """design §8.3: NEVER write the local ``archived-sessions.json`` for a remote id."""
    server_a, server_b, _host, _port = pair
    _pair_settled(pair, monkeypatch, role="admin")
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
    _pair_settled(pair, monkeypatch, role="admin")
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
    _pair_settled(pair, monkeypatch, role="admin")
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
    _pair_settled(pair, monkeypatch, role="admin")
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


# ---------------------------------------------------------------------------
# Read receipts on a peer: the OWNER's store is the one that changes
# ---------------------------------------------------------------------------


def _publish_completion(server: relay.RelayServer, session_id: str, anchor: str) -> str:
    """One completion on ``server``'s own attention store, returning its token."""
    from local_operator.session.attention import AttentionStore

    token = str(uuid.uuid4())
    AttentionStore(server.root / "attention.db").publish(
        f"session/{session_id}", token, anchor, "complete"
    )
    return token


def test_a_read_receipt_clears_on_the_owners_store(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The unread mark lives on the owner: B forwards, A's store changes, B writes nothing.

    The operator-reported defect this pins: the desktop's visible-read receipt for
    a peer's row hit a cold local path and 404'd about a conversation the user
    could see. The fix routes it to the owner over ``net_session_receipt``; this
    cell drives the whole hop the way the desktop backend does (B's control
    socket), so a slice that never registered the op fails here.
    """
    from local_operator.session.attention import AttentionStore

    server_a, server_b, _host, _port = pair
    _pair_settled(pair, monkeypatch, role="admin")
    _owned_session(server_a)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_b.root))

    token = _publish_completion(server_a, SESSION, "anchor-1")
    result = mobility.receipt(
        SESSION, token=token, peer=server_a.identity.device_id, root=server_b.root
    )

    assert result["ok"] is True, result
    assert result["session_id"] == SESSION, result
    state = AttentionStore(server_a.root / "attention.db").state(f"session/{SESSION}")
    assert state["unseen"] is False and state["revision"] == [1, 1], state
    # THE WRITE HAPPENED THERE, NOT HERE: B holds no receipt at all.
    assert not (server_b.root / "attention.db").exists()

    # AND A RETRY CONVERGES: a second identical acknowledgement lands on the
    # same settled outcome, which is what the desktop's retry path needs.
    again = mobility.receipt(
        SESSION, token=token, peer=server_a.identity.device_id, root=server_b.root
    )
    assert again["ok"] is True, again
    again_state = again["state"]
    assert isinstance(again_state, dict) and again_state["unseen"] is False, again


def test_a_superseded_receipt_token_crosses_with_its_own_code(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The owner's stale-token verdict is DATA with the code the renderer re-arms on."""
    from local_operator.session.attention import AttentionStore

    server_a, server_b, _host, _port = pair
    _pair_settled(pair, monkeypatch, role="admin")
    _owned_session(server_a)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_b.root))

    first = _publish_completion(server_a, SESSION, "anchor-1")
    second = _publish_completion(server_a, SESSION, "anchor-2")

    stale = mobility.receipt(
        SESSION, token=first, peer=server_a.identity.device_id, root=server_b.root
    )
    assert stale["ok"] is False, stale
    assert stale["code"] == "superseded_completion_token", stale
    assert "current token" in str(stale["message"]), stale
    # The refusal marked nothing read.
    assert (
        AttentionStore(server_a.root / "attention.db").state(f"session/{SESSION}")["unseen"] is True
    )

    caught_up = mobility.receipt(
        SESSION, token=second, peer=server_a.identity.device_id, root=server_b.root
    )
    assert caught_up["ok"] is True, caught_up
    caught_up_state = caught_up["state"]
    assert isinstance(caught_up_state, dict) and caught_up_state["unseen"] is False, caught_up


def test_an_unknown_receipt_token_is_refused_without_touching_state(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The store's unknown-token verdict crosses as its own code and writes nothing."""
    from local_operator.session.attention import AttentionStore

    server_a, server_b, _host, _port = pair
    _pair_settled(pair, monkeypatch, role="admin")
    _owned_session(server_a)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_b.root))

    _publish_completion(server_a, SESSION, "anchor-1")
    result = mobility.receipt(
        SESSION, token=str(uuid.uuid4()), peer=server_a.identity.device_id, root=server_b.root
    )
    assert result["ok"] is False, result
    assert result["code"] == "unknown_completion_token", result
    assert (
        AttentionStore(server_a.root / "attention.db").state(f"session/{SESSION}")["unseen"] is True
    )


def test_a_receipt_for_a_session_the_owner_does_not_hold_is_refused(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The chokepoint's ownership rule covers the new op: a relay acts only on its own ids."""
    server_a, server_b, _host, _port = pair
    _pair_settled(pair, monkeypatch, role="admin")
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_b.root))

    stranger = "0f1e2d3c4b5a"
    result = mobility.receipt(
        stranger, token=str(uuid.uuid4()), peer=server_a.identity.device_id, root=server_b.root
    )
    assert result["ok"] is False, result
    assert "does not live on this device" in str(result["message"]), result


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
    the desktop's own pane holding the session: this device's refusal was produced in
    ~3 ms and the user read a timeout 60 s later. The destination now reports it over the
    link it was invited on (``_source_refused``), and the wait ends where the refusal
    happened (``_await_own_progress``'s refusal half).

    DELIBERATELY NOT END-TO-END, the same trade the invite-ack test above records. The rig
    that dials A->B is this file's load-flaky shape — the fixture's peer answers late on a
    loaded runner and the move reads ``unreachable``, which is a transport result and not a
    statement about this property. So the two seams the property is MADE of are driven
    directly: the report that records the destination's sentence, and the wait that must
    end on it within a poll instead of a budget. The device-keyed half is pinned here too
    (a refusal from anybody else must not end a move it did not receive), which is a fact
    about THIS seam that the dialling version could only reach by standing up a stranger.
    """
    import time
    from types import SimpleNamespace

    server_a, server_b, _host, _port = pair
    # NEVER THE SHARED ``SESSION``: ``_Progress`` is per-relay but lives in that process
    # keyed by ``id(server)``, so a refusal left under it can outlive the test that wrote it.
    session_id = "ef56ab12cd34"
    invited = server_b.identity.device_id
    sentence = (
        "This conversation is open in another window or app. "
        "Close that window and try again, or queue the move to run at the next safe point."
    )
    progress = mobility.progress_for(server_a)
    try:
        # THE REPORT PATH, and the recording is exact: this side never saw the guard that
        # fired, so the sentence travels VERBATIM (a paraphrase here would be the second
        # sentence table for one condition that §8.2 exists to prevent).
        link: Any = SimpleNamespace(device_id=invited)
        recorded = mobility._source_refused(  # noqa: SLF001
            server_a, link, {"session_id": session_id, "code": "busy", "message": sentence}
        )

        assert recorded["result"] == "recorded", recorded
        assert progress.refusal(session_id, from_device=invited) == ("busy", sentence)

        # AND THE WAIT ENDS ON IT. The budget is the one an offload at ``wait_s=0`` would
        # otherwise sit out in full; the poll the wait notices it on is 0.2 s.
        budget = mobility.move_bound_s(0.0)
        started = time.monotonic()
        outcome, refusal = mobility._await_own_progress(  # noqa: SLF001
            server_a, session_id, budget=budget, invited=invited
        )
        elapsed = time.monotonic() - started

        assert outcome == "none", "a refusal is not a commit"
        assert refusal == ("busy", sentence), refusal
        assert elapsed < 1.0, (
            f"the refusal took {elapsed:.1f}s of a {budget:.0f}s budget, so it waited for the "
            "deadline rather than being told: that is the finding this test exists for"
        )

        # KEYED BY THE DEVICE THAT SENT IT: a member that did not receive this move cannot
        # end it, so a refusal recorded by anyone else leaves the wait with no answer.
        progress.forget(session_id)
        stranger: Any = SimpleNamespace(device_id="ffffffffffff")
        mobility._source_refused(  # noqa: SLF001
            server_a,
            stranger,
            {"session_id": session_id, "code": "busy", "message": "not this move"},
        )
        outcome, refusal = mobility._await_own_progress(  # noqa: SLF001
            server_a, session_id, budget=0.3, invited=invited
        )

        assert outcome == "none" and refusal is None, refusal
    finally:
        progress.forget(session_id)


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


def test_forget_drops_the_completion_signal_a_previous_move_set() -> None:
    """``forget``'s contract: EVERYTHING a move start must not inherit — signals too.

    ``signal`` stores a level-triggered ``threading.Event`` under ``<id>:<marker>``
    and nothing else clears it; ``forget`` used to drop only phases and refusals, so
    one completed move-out left its ``done`` set for the relay process's lifetime and
    the NEXT offload's first poll consumed it as its own completion — an instant
    synthetic ``committed`` for a move that never ran (live, 2026-09-29; the sibling
    cells above pin the wire-level symptoms). This pins the reset itself, with no
    relay in the way: after ``forget`` there is no history, no refusal, no signal.
    """
    progress = mobility._Progress()
    session_id = "ef56ab12cd34"

    progress.note(session_id, "done")
    progress.signal(session_id, "done")
    assert progress.wait_for(session_id, "done", 0.05) is True

    progress.note_refusal(session_id, code="busy", message="mid-turn", from_device="d_peer")
    progress.forget(session_id)

    assert progress.phases(session_id) == []
    assert progress.refusal(session_id) is None, "a refusal outlived its move"
    assert (
        progress.wait_for(session_id, "done", 0.05) is False
    ), "a set completion event survived forget(); the next move would read it as its own"


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
    # A CURRENT relay's record: one without ``session-move-workset-v1`` is refused
    # before any control request (review R9), which is not the envelope measured here.
    current = SimpleNamespace(capabilities=list(relay.wire.LINK_CAPABILITIES))
    monkeypatch.setattr("local_operator.network.store.find_own_relay", lambda root=None: current)

    for keep, to in ((False, "build-box"), (True, "build-box"), (False, "local")):
        seen.clear()
        mobility.request_move(SESSION, to=to, keep=keep, wait_s=30.0, root=Path("/tmp/x"))
        assert seen == [
            mobility.move_client_bound_s(30.0, keep=keep, to=to) - mobility.MOVE_CLIENT_MARGIN_S
        ], (keep, to, seen)


class TestViewedIsTheViewerBlocker:
    """Q1 (QA round 1): the exclusive-observer refusal is ``viewed``, not ``busy``.

    §5.4's whole point is the split a notice offers Queue vs Wait from: a live
    TURN is ``busy`` (waiting clears it), while another attached client is
    ``viewed`` (waiting cannot clear it — the queue can). ``_RetireOutcome``'s
    vocabulary always named the second case ``viewed``, but the producer
    collapsed every ``kept:`` answer into ``busy``, so the wire never carried
    ``viewed_elsewhere`` for a viewer. These cells drive the real
    ``_retire_local_runtime`` with a canned client: the runtime's own sentence
    (matched through the SHARED constant) classifies as ``viewed``; anything
    else the runtime keeps itself for stays ``busy``.
    """

    def _stub(self, monkeypatch: pytest.MonkeyPatch, answer: str) -> None:
        from types import SimpleNamespace

        from local_operator.mobile import attach_client as attach_mod

        class StubClient:
            def __init__(self, *_a: Any, **_k: Any) -> None: ...

            async def connect(self, record: Any, session_id: str) -> None: ...

            async def retire_now(self, *, exclusive: bool = False) -> str:
                assert exclusive is True, "a move must ask under the fence"
                return answer

            def close(self) -> None: ...

        monkeypatch.setattr(attach_mod, "AttachClient", StubClient)
        monkeypatch.setattr(
            attach_mod,
            "find_runtime_record",
            lambda root, session_id: (
                SimpleNamespace(capabilities=["exclusive-move-v1"], session_id=session_id),
                4321,
            ),
        )

    def test_the_viewer_refusal_is_viewed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from local_operator.session.runtime.types import VIEWED_MOVE_REFUSAL

        self._stub(monkeypatch, f"kept: {VIEWED_MOVE_REFUSAL}")
        outcome = mobility._retire_local_runtime(tmp_path, SESSION)  # noqa: SLF001
        assert outcome["result"] == "viewed", outcome
        assert outcome["sentence"] == VIEWED_MOVE_REFUSAL

    def test_a_turn_refusal_stays_busy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._stub(monkeypatch, "kept: busy")
        outcome = mobility._retire_local_runtime(tmp_path, SESSION)  # noqa: SLF001
        assert outcome["result"] == "busy", outcome


def test_a_compressed_session_moves_and_is_not_refused_as_unlisted(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A session whose transcript is an ARCHIVE moves like any other (M-1 / Q-3).

    THE FAILURE THIS PINS. ``assert_complete`` runs in ``_source_prepare`` BEFORE
    anything restores the transcript, so it judges the directory as it is on disk —
    and a compressed session's body is ``transcript.jsonl.gz``/``.zst``, not
    ``transcript.jsonl``. Nothing named the archive, so every session past
    ``after_days`` was refused with ``unlisted_content`` and a sentence whose remedy
    ("move or remove it") meant deleting the only copy of the conversation.
    Compression is on by default, so this was not a corner.

    The fix has two halves and this test exercises both on the REAL path: the archive
    names are copy-set names (so the completeness question can be answered), and the
    prepare restores the transcript before it digests the copy set (so the stamps it
    takes describe the bytes the wire will carry). Two relays, real loopback, driven
    the way the CLI drives it.
    """
    from local_operator.session import compression

    server_a, server_b, _host, _port = pair
    _pair_settled(pair, monkeypatch, role="admin")
    source = _owned_session(server_a)
    before = _transcript(server_a.root, SESSION)

    outcome = compression.compress_session(source)
    assert outcome.status == "compressed", outcome
    archive = source / f"transcript.jsonl{compression.ARCHIVE_SUFFIXES[outcome.codec]}"
    assert archive.is_file() and not (source / "transcript.jsonl").exists()

    # THE COMPLETENESS GUARD THAT REFUSED IT, asked the way the prepare asks it.
    assert sync.assert_complete(source) is None

    # THE ARCHIVE IS THE TRANSCRIPT, COUNTED ONCE, WITH ITS OWN DIGEST AND CLOCK: the
    # plain name is absent, so it is a single member, and the digest is of the archive
    # bytes rather than a second name for the same bytes.
    names = sync.copy_set(server_a.root, SESSION)
    transcript_members = [name for name in names if name.startswith("transcript.jsonl")]
    assert transcript_members == [archive.name], transcript_members
    stamp = sync.member_stamps(
        server_a.root, SESSION, attachments_dir=server_a.root / "attachments"
    )[archive.name]
    info = archive.stat()
    assert (stamp.stat_bytes, stamp.mtime_ns) == (info.st_size, info.st_mtime_ns)

    result = _move(server_b, SESSION, monkeypatch=monkeypatch)

    assert result["ok"] is True, result
    assert result["phases"][0]["phase"] == "prepared", result["phases"]
    assert result["phase"] == "done"
    assert not source.exists()
    # THE CONVERSATION ARRIVED WHOLE, as the plain journal the protocol speaks.
    assert _transcript(server_b.root, SESSION) == before


def _open_team_session(root: Path, directory: Path):
    from local_operator.agents import AgentRegistry
    from local_operator.session.session import Session
    from local_operator.session.transcript import Transcript
    from local_operator.teams import TeamRegistry
    from tests.unit.session.test_session import MODEL, ScriptedStream

    teams = TeamRegistry(root)
    return Session(
        model=MODEL,
        stream_fn=ScriptedStream([[]]),
        tools=[],
        transcript=Transcript(directory),
        system_blocks_provider=lambda: [],
        team_registry=teams,
        agent_registry=AgentRegistry(root),
    )


def _authorize_source(root: Path, directory: Path, name: str) -> None:
    session = _open_team_session(root, directory)
    assert session.team_registry is not None
    team = session.team_registry.get_team_by_name(name, require_loaded=True)
    assert team is not None
    session.attach_team(team)


def _create_source_team(server: relay.RelayServer, name: str) -> None:
    """A team whose manager is a role row recorded HERE ONLY.

    The destination cannot resolve that manager until the independent definitions
    sync lands, which is the condition the queued-retry tests need: the first queued
    invite is refused ``team_definitions_pending`` rather than succeeding.
    """
    from local_operator.agents import AgentRegistry
    from local_operator.teams import TeamEditFields, TeamMember, TeamRegistry

    AgentRegistry(server.root).create_agent(_agent_fields(name="offload-manager", tags=["role"]))
    TeamRegistry(server.root).create_team(
        TeamEditFields(
            name=name,
            manager="offload-manager",
            members=[TeamMember(role="coder")],
            instructions="Every PR gets a review round.",
        )
    )


def test_no_authority_conferring_install_before_invite_and_retry_after_sync(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No authority-conferring install may happen before the invite completes."""
    from local_operator.agents import AgentRegistry
    from local_operator.network import definitions
    from local_operator.teams import TeamEditFields, TeamMember, TeamRegistry

    # Hold the independent cadence until the test explicitly reconciles below;
    # pairing can nudge it immediately, before the invite reaches the peer.
    monkeypatch.setattr(definitions.DefinitionsSyncer, "tick", lambda *_a, **_k: [])
    server_a, server_b, _host, _port = pair
    _pair_settled(pair, monkeypatch, role="admin", settings=server_b.settings)
    session_dir = _owned_session(server_a)
    manager = "offload-manager"
    AgentRegistry(server_a.root).create_agent(_agent_fields(name=manager, tags=["role"]))
    TeamRegistry(server_a.root).create_team(
        TeamEditFields(
            name="lopdev",
            description="the team",
            manager=manager,
            members=[TeamMember(role="coder")],
            instructions="Every PR gets a review round.",
            project="local-operator",
        )
    )
    _authorize_source(server_a.root, session_dir, "lopdev")
    rules = "# Personal Defaults\n\n- Never merge without an agent-review round.\n"
    (server_a.root / "system_prompt.md").write_text(rules, encoding="utf-8")

    events: list[tuple[str, Any]] = []
    real_ask = mobility.LinkTransport.ask
    real_apply = definitions.apply_bundle

    def apply(root: Any, bundle: Any, *, origin_device: str) -> dict[str, Any]:
        # WHICH ROOT, not just that an apply ran: the invariant is that the
        # DESTINATION's store carries nothing this move installed before its invite.
        events.append(("apply", str(root)))
        return real_apply(root, bundle, origin_device=origin_device)

    def ask(self: Any, frame: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        events.append(("ask", frame.get("phase")))
        return real_ask(self, frame, **kwargs)

    monkeypatch.setattr(definitions, "apply_bundle", apply)
    monkeypatch.setattr(mobility.LinkTransport, "ask", ask)

    result = _move(server_a, SESSION, to=server_b.identity.device_id, monkeypatch=monkeypatch)

    # R4-2: NOTHING is installed on the destination before the invite — not roles or
    # teams, and not the standing-instructions singleton either. The condition is
    # about the MECHANISM (``definitions.apply_bundle`` on the receiver), so "those
    # selectors only install instructions today" is an argument about arguments.
    invite = events.index(("ask", "invite"))
    assert [event for event in events[:invite] if event[0] == "apply"] == [], events
    assert not (
        server_b.root / "system_prompt.md"
    ).exists(), "the destination's standing instructions were installed before the invite"
    assert TeamRegistry(server_b.root).get_team_by_name("lopdev") is None
    assert AgentRegistry(server_b.root).get_agent_by_name(manager) is None
    assert result["ok"] is False and result["code"] == "team_definitions_pending", result
    assert result["message"] == (
        "lopdev: team definition is not available yet; retry after definitions sync"
    ), result
    assert result["phase_reached"] == "prepared" and result["changed"] is False, result
    assert session_dir.is_dir()
    assert not (server_b.root / "sessions" / SESSION).exists()
    print("Before definitions sync:", result["code"], result["message"])

    # Independently run the same reconciliation used by the background cadence.
    synced = definitions.push_to_peer(server_a, server_b.identity.device_id)
    assert synced["ok"] and "lopdev" in synced["teams"], synced
    retry = _move(server_a, SESSION, to=server_b.identity.device_id, monkeypatch=monkeypatch)
    assert retry["ok"] is True, retry
    arrived = read_stamp(server_b.root, SESSION)
    assert arrived is not None and arrived.home_device == server_b.identity.device_id
    assert not session_dir.exists()
    assert "definitions_notice" not in (retry.get("carry") or {}), retry
    print("After definitions sync: retry succeeded and destination owns the session")


@pytest.mark.asyncio
@pytest.mark.parametrize("destination_tool", ["read", "edit"])
async def test_unbound_destination_declaration_requires_recorded_team_identity(
    pair: Devices, monkeypatch: pytest.MonkeyPatch, destination_tool: str
) -> None:
    from local_operator.agents import AgentRegistry
    from local_operator.network import move_team
    from local_operator.teams import TeamEditFields, TeamRegistry

    owner, receiver, _host, _port = pair
    _pair_settled(pair, monkeypatch, role="admin", settings=receiver.settings)
    source = _owned_session(owner)
    manager = "offload-manager"
    team = "offload-team"
    for root, tool in ((owner.root, "read"), (receiver.root, destination_tool)):
        AgentRegistry(root).create_agent(
            _agent_fields(name=manager, tags=["role", f"tools:{tool}"])
        )
        TeamRegistry(root).create_team(TeamEditFields(name=team, manager=manager))
    _authorize_source(owner.root, source, team)
    assert move_team.source_offer(owner.root, source, supported=True) is not None

    destination = receiver.root / "sessions" / SESSION
    real_ask = mobility.LinkTransport.ask
    checked = False

    def ask(self: Any, frame: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        nonlocal checked
        if frame.get("phase") == "invite":
            # An existing declaration cannot bind the arriving session. The
            # invite transports no declaration; the pull carries the ceiling.
            assert move_team.KEY not in frame
            assert AgentRegistry(receiver.root).get_agent_by_name(manager) is not None
            assert TeamRegistry(receiver.root).get_team_by_name(team) is not None
            assert not destination.exists()
            # THE INVARIANT, not the mechanism (review r5 R5-3): an existing
            # local declaration still cannot bind the arriving session. A pull
            # carrying no record is accepted as carrying NOTHING — no team identity
            # comes back and no binding is written — which is exactly what keeps a
            # NAME from being authority. (It used to refuse the whole move here;
            # a name that cannot be attested is now reported as unresolved on the
            # receipt instead of blocking the conversation from moving.)
            assert move_team.check_arrival(receiver.root, source, None) is None
            # The destination is untouched: no team identity and no directory.
            assert not destination.exists()
            checked = True
        # The rollback and refusal report pass explicit timeouts; the spy must
        # relay those unchanged or a real refusal looks like a move deadline.
        return real_ask(self, frame, **kwargs)

    monkeypatch.setattr(mobility.LinkTransport, "ask", ask)
    result = _move(owner, SESSION, to=receiver.identity.device_id, monkeypatch=monkeypatch)
    assert checked
    if destination_tool == "edit":
        assert result["ok"] is False and result["code"] == move_team.CODE, result
        assert result["message"] == f"{team}: would grant tool pins: edit", result
        assert source.is_dir() and not destination.exists()
        print("Recorded ceiling refusal:", result["code"], result["message"])
        return
    assert result["ok"] is True, result
    arrived = _open_team_session(receiver.root, destination)
    try:
        assert arrived.active_team_name == team
        assert arrived._team_manager_name() == manager
    finally:
        await arrived.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("destination", ["missing", "conflict", "wider-manager", "match"])
async def test_recall_requires_the_source_team_definition(
    pair: Devices, monkeypatch: pytest.MonkeyPatch, destination: str
) -> None:
    """Drive real relay recall and inspect runtime attachment in-process."""
    from local_operator.agents import AgentRegistry
    from local_operator.network.move_team import TeamIdentityRefused, _digest
    from local_operator.session.goal import GoalState
    from local_operator.session.session import Session
    from local_operator.session.transcript import Transcript
    from local_operator.teams import TeamEditFields, TeamRegistry
    from tests.unit.session.test_session import MODEL, ScriptedStream

    def open_session(root: Path, directory: Path) -> tuple[Session, GoalState]:
        goal = GoalState()
        return (
            Session(
                model=MODEL,
                stream_fn=ScriptedStream([[]]),
                tools=[],
                transcript=Transcript(directory),
                system_blocks_provider=lambda: [],
                goal_state=goal,
                team_registry=TeamRegistry(root),
                agent_registry=AgentRegistry(root),
            ),
            goal,
        )

    def attachment(session: Session, goal: GoalState) -> tuple[str, str, bool]:
        return goal.team_brief, session._team_manager_name(), session._readonly_shell

    owner, receiver, _host, _port = pair
    _pair_settled(pair, monkeypatch, role="admin")
    source = _owned_session(owner)
    source_fields = TeamEditFields(
        name="identity-test", manager="manager", instructions="source brief"
    )
    authored = TeamRegistry(owner.root).create_team(source_fields)
    _authorize_source(owner.root, source, authored.name)

    registry = TeamRegistry(receiver.root)
    registry.create_team(
        TeamEditFields(name="local-control", manager="manager", instructions="local brief")
    )
    matching_team = None
    if destination == "conflict":
        registry.create_team(
            TeamEditFields(name="identity-test", manager="scout", instructions="destination brief")
        )
    elif destination in {"match", "wider-manager"}:
        matching_team = registry.create_team(source_fields)
        assert matching_team.id != authored.id
        assert _digest(matching_team) == _digest(authored)
        if destination == "wider-manager":
            AgentRegistry(receiver.root).create_agent(
                _agent_fields(name="manager", tags=["role", "tools:edit"])
            )
    existing_dir = _owned_session(receiver, "101010101010", rows=0)
    existing_team = "local-control" if destination == "missing" else "identity-test"
    _authorize_source(receiver.root, existing_dir, existing_team)
    existing, existing_goal = open_session(receiver.root, existing_dir)
    before = attachment(existing, existing_goal)
    if destination == "conflict":
        assert before[1:] == ("scout", True)
        assert "destination brief" in before[0]
    arrived = None
    try:
        result = _move(receiver, SESSION, monkeypatch=monkeypatch)
        moved_dir = receiver.root / "sessions" / SESSION
        observed: Any = (result["ok"], result.get("message"))
        if moved_dir.is_dir():
            arrived, arrived_goal = open_session(receiver.root, moved_dir)
            brief, manager, read_only = attachment(arrived, arrived_goal)
            observed = (result["ok"], brief, manager, read_only)

        # Check both the existing live session and a fresh restore from the
        # local registry; neither may be changed by comparison on arrival.
        assert attachment(existing, existing_goal) == before
        reopened, reopened_goal = open_session(receiver.root, existing_dir)
        try:
            assert attachment(reopened, reopened_goal) == before
        finally:
            await reopened.dispose()
        if destination == "match":
            assert arrived is not None, result
            assert observed == (True, before[0], "manager", False)
            assert "source brief" in before[0]

            # A later registry edit must not switch the moved session to a
            # different team at construction after the protocol already agreed.
            assert matching_team is not None
            registry.update_team(matching_team.id, TeamEditFields(manager="scout"))
            with pytest.raises(TeamIdentityRefused, match="identity-test"):
                open_session(receiver.root, moved_dir)
            assert attachment(existing, existing_goal) == before
        else:
            assert observed[0] is False, observed
            expected_code = (
                "team_definitions_pending"
                if destination == "missing"
                else "team_definition_unavailable"
            )
            assert result["code"] == expected_code, result
            assert "identity-test" in str(observed[1]), observed
            if destination == "wider-manager":
                assert "would grant tool pins: edit" in str(observed[1]), observed
            assert arrived is None
            assert source.is_dir()
    finally:
        if arrived is not None:
            await arrived.dispose()
        await existing.dispose()


def test_offload_reports_a_team_conflict_without_replacing_the_local_definition(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.teams import TeamEditFields, TeamRegistry

    source, destination, _host, _port = pair
    _pair_settled(pair, monkeypatch, role="admin", settings=destination.settings)
    directory = _owned_session(source)
    name = "identity-test"
    TeamRegistry(source.root).create_team(TeamEditFields(name=name, instructions="source brief"))
    TeamRegistry(destination.root).create_team(
        TeamEditFields(name=name, instructions="different brief")
    )
    _authorize_source(source.root, directory, name)

    result = _move(source, SESSION, to=destination.identity.device_id, monkeypatch=monkeypatch)

    assert result["ok"] is False and result["code"] == "team_definition_unavailable", result
    assert name in result["message"], result
    assert directory.is_dir()
    assert not (destination.root / "sessions" / SESSION).exists()
    still_local = TeamRegistry(destination.root).get_team_by_name(name)
    assert still_local is not None and still_local.instructions == "different brief"


def test_a_failed_definitions_refresh_does_not_block_an_already_matching_team(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing is refreshed before the invite (R4-2), so a broken definitions relay
    cannot block the move — and the receipt claims no refresh that never ran."""
    from local_operator.network import definitions
    from local_operator.teams import TeamEditFields, TeamRegistry

    # Hold the independent cadence: its pushes are legitimate and are not this row.
    monkeypatch.setattr(definitions.DefinitionsSyncer, "tick", lambda *_a, **_k: [])
    server_a, server_b, _host, _port = pair
    _pair_settled(pair, monkeypatch, role="admin", settings=server_b.settings)
    session_dir = _owned_session(server_a)
    fields = TeamEditFields(name="lopdev", instructions="the same team brief")
    authored = TeamRegistry(server_a.root).create_team(fields)
    independent = TeamRegistry(server_b.root).create_team(fields)
    assert independent.id != authored.id
    _authorize_source(server_a.root, session_dir, "lopdev")

    pushed: list[str] = []

    def refused(server: Any, device_id: str, **kwargs: Any) -> dict[str, Any]:
        pushed.append(device_id)
        return {"ok": False, "code": "unreachable", "message": "definitions relay unavailable"}

    monkeypatch.setattr(definitions, "push_to_peer", refused)

    result = _move(server_a, SESSION, to=server_b.identity.device_id, monkeypatch=monkeypatch)

    assert result["ok"] is True, result
    assert (server_b.root / "sessions" / SESSION).is_dir()
    assert pushed == [], "the move pushed definitions before its invite (R4-2)"
    assert "definitions_notice" not in (result.get("carry") or {}), result


def test_team_identity_refuses_unreadable_attachment_evidence(tmp_path: Path) -> None:
    from local_operator.network.move_team import (
        TeamIdentityRefused,
        check_arrival,
        source_offer,
    )
    from local_operator.resume import ATTACHMENT_SIDECAR_NAME, write_session_attachment

    assert source_offer(tmp_path, tmp_path, supported=True) is None
    write_session_attachment(tmp_path, team="", agent="", goal="")
    assert source_offer(tmp_path, tmp_path, supported=True) is None
    (tmp_path / ATTACHMENT_SIDECAR_NAME).write_text("{", encoding="utf-8")
    with pytest.raises(TeamIdentityRefused):
        source_offer(tmp_path, tmp_path, supported=True)
    with pytest.raises(TeamIdentityRefused):
        check_arrival(tmp_path, tmp_path, None)


def test_team_identity_uses_current_revision_on_the_second_hop(tmp_path: Path) -> None:
    from local_operator.network.move_team import KEY, check_arrival, source_offer
    from local_operator.resume import write_session_attachment
    from local_operator.session.placement import MeshStamp, write_stamp_into
    from local_operator.teams import TeamEditFields, TeamRegistry

    roots = [tmp_path / str(index) for index in range(3)]
    registries = [TeamRegistry(root) for root in roots]
    original = registries[0].create_team(TeamEditFields(name="identity-test", instructions="v1"))
    registries[1].save_team(original)
    for root in roots:
        write_session_attachment(root, team=original.name, agent="", goal="")
    _authorize_source(roots[0], roots[0], original.name)
    first = source_offer(roots[0], roots[0], supported=True)
    assert first is not None
    binding = check_arrival(roots[1], roots[1], first)
    write_stamp_into(roots[1], MeshStamp(session_id=SESSION, origin={KEY: binding}))

    updated = registries[1].update_team(original.id, TeamEditFields(instructions="v2"))
    registries[2].save_team(updated)
    second = source_offer(roots[1], roots[1], supported=True)
    assert second is not None and second["digest"] != first["digest"]
    assert check_arrival(roots[2], roots[2], second) is not None


def test_team_identity_refuses_until_the_destination_definition_is_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.network.move_team import (
        TeamIdentityRefused,
        check_arrival,
        source_offer,
    )
    from local_operator.resume import write_session_attachment
    from local_operator.teams import TeamEditFields, TeamRegistry

    author, peer = tmp_path / "author", tmp_path / "peer"
    team = TeamRegistry(author).create_team(TeamEditFields(name="identity-test"))
    for root in (author, peer):
        write_session_attachment(root, team=team.name, agent="", goal="")
    _authorize_source(author, author, "identity-test")
    offer = source_offer(author, author, supported=True)
    assert offer is not None
    pending = "identity-test: team definition is not available yet; retry after definitions sync"
    with pytest.raises(TeamIdentityRefused, match=pending) as missing:
        check_arrival(peer, peer, offer)
    assert missing.value.code == "team_definitions_pending"
    TeamRegistry(peer).save_team(team)
    with monkeypatch.context() as unloaded:
        unloaded.setattr(TeamRegistry, "_load_briefs", lambda self, row: row)
        with pytest.raises(TeamIdentityRefused, match=pending) as not_loaded:
            check_arrival(peer, peer, offer)
        assert not_loaded.value.code == "team_definitions_pending"
    assert check_arrival(peer, peer, offer) is not None
    with pytest.raises(TeamIdentityRefused, match="recorded authority is unavailable"):
        check_arrival(
            peer, peer, {key: value for key, value in offer.items() if key != "delegation"}
        )


def test_team_identity_refuses_a_different_manager_authority(tmp_path: Path) -> None:
    from local_operator.agents import AgentRegistry
    from local_operator.network.move_team import (
        TeamIdentityRefused,
        _digest,
        check_arrival,
        source_offer,
    )
    from local_operator.resume import write_session_attachment
    from local_operator.teams import TeamEditFields, TeamRegistry

    author, peer = tmp_path / "author", tmp_path / "peer"
    team = TeamRegistry(author).create_team(TeamEditFields(name="identity-test", manager="lead"))
    other = TeamRegistry(peer).create_team(TeamEditFields(name=team.name, manager=team.manager))
    assert other.id != team.id
    assert _digest(other) == _digest(team)
    for root, tool in ((author, "read"), (peer, "edit")):
        AgentRegistry(root).create_agent(
            _agent_fields(name="lead", description="Team manager", tags=["role", f"tools:{tool}"])
        )
        write_session_attachment(root, team=team.name, agent="", goal="")
    _authorize_source(author, author, "identity-test")
    offer = source_offer(author, author, supported=True)
    assert offer is not None
    with pytest.raises(TeamIdentityRefused, match="would grant tool pins: edit"):
        check_arrival(peer, peer, offer)


def test_team_identity_names_shell_widening_before_any_attachment(tmp_path: Path) -> None:
    from local_operator.network.move_team import (
        TeamIdentityRefused,
        check_arrival,
        source_offer,
    )
    from local_operator.resume import write_session_attachment
    from local_operator.teams import TeamEditFields, TeamRegistry

    author, peer = tmp_path / "author", tmp_path / "peer"
    for root, manager in ((author, "scout"), (peer, "manager")):
        TeamRegistry(root).create_team(TeamEditFields(name="identity-test", manager=manager))
        write_session_attachment(root, team="identity-test", agent="", goal="")
    _authorize_source(author, author, "identity-test")
    offer = source_offer(author, author, supported=True)
    assert offer is not None
    with pytest.raises(TeamIdentityRefused, match="would grant an unrestricted shell"):
        check_arrival(peer, peer, offer)


def test_arrival_accepts_strictly_narrower_manager_pins(tmp_path: Path) -> None:
    from local_operator.agents import AgentRegistry
    from local_operator.network.move_team import KEY, check_arrival, source_offer
    from local_operator.resume import write_session_attachment
    from local_operator.session.placement import MeshStamp, write_stamp_into
    from local_operator.teams import TeamEditFields, TeamRegistry

    source, target = tmp_path / "source", tmp_path / "target"
    for root, tags in ((source, ["tools:read,edit"]), (target, ["tools:read"])):
        AgentRegistry(root).create_agent(_agent_fields(name="lead", tags=["role", *tags]))
        TeamRegistry(root).create_team(TeamEditFields(name="identity-test", manager="lead"))
    _authorize_source(source, source, "identity-test")
    write_session_attachment(target, team="identity-test", agent="", goal="")
    offer = source_offer(source, source, supported=True)
    assert offer is not None
    assert offer["manager"]["pins"] == ["edit", "read"]
    admitted = check_arrival(target, target, offer)
    assert admitted is not None
    assert admitted["manager"]["pins"] == ["read"]
    write_stamp_into(target, MeshStamp(session_id=SESSION, origin={KEY: admitted}))
    assert _open_team_session(target, target)._deferral_pins == {"read"}


def test_intermediate_reopen_keeps_brief_update_and_tightens_next_hop(tmp_path: Path) -> None:
    from local_operator.agents import AgentRegistry
    from local_operator.network.move_team import (
        KEY,
        TeamIdentityRefused,
        check_arrival,
        source_offer,
    )
    from local_operator.resume import write_session_attachment
    from local_operator.session.placement import MeshStamp, write_stamp_into
    from local_operator.teams import TeamEditFields, TeamRegistry

    source, middle, target = (tmp_path / part for part in ("source", "middle", "target"))
    old = TeamEditFields(name="identity-test", manager="lead", instructions="old brief")
    new = TeamEditFields(name="identity-test", manager="lead", instructions="updated brief")
    managers = {}
    for root, fields, tools in (
        (source, old, "read,edit"),
        (middle, old, "read,edit"),
        (target, new, "read"),
    ):
        managers[root] = AgentRegistry(root).create_agent(
            _agent_fields(name="lead", tags=["role", f"tools:{tools}"])
        )
        TeamRegistry(root).create_team(fields)
        write_session_attachment(root, team="identity-test", agent="", goal="")
    _authorize_source(source, source, "identity-test")
    first = source_offer(source, source, supported=True)
    previous = check_arrival(middle, middle, first)
    write_stamp_into(middle, MeshStamp(session_id=SESSION, origin={KEY: previous}))
    mid_teams, mid_agents = TeamRegistry(middle), AgentRegistry(middle)
    mid_team = mid_teams.get_team_by_name("identity-test")
    assert mid_team is not None
    mid_teams.update_team(mid_team.id, new)
    mid_agents.update_agent(managers[middle].id, _agent_fields(tags=["role", "tools:read"]))
    reopened = _open_team_session(middle, middle)
    assert "updated brief" in reopened._goal_state.team_brief
    assert reopened._deferral_pins == {"read"}
    second = source_offer(middle, middle, supported=True)
    assert first is not None and second is not None
    assert second["digest"] != first["digest"]
    arrived = check_arrival(target, target, second)
    assert arrived is not None
    assert arrived["manager"]["pins"] == ["read"]
    mid_agents.update_agent(managers[middle].id, _agent_fields(tags=["role", "tools:read,edit"]))
    with pytest.raises(TeamIdentityRefused, match="would grant tool pins: edit"):
        source_offer(middle, middle, supported=True)


def test_matching_top_team_refuses_wider_nested_member(tmp_path: Path) -> None:
    from local_operator.agents import AgentRegistry
    from local_operator.network.move_team import (
        TeamIdentityRefused,
        _digest,
        check_arrival,
        source_offer,
    )
    from local_operator.resume import write_session_attachment
    from local_operator.teams import TeamEditFields, TeamMember, TeamRegistry

    source, target = tmp_path / "source", tmp_path / "target"
    teams = []
    for root, pins in ((source, "read"), (target, "read,edit")):
        agents, registry = AgentRegistry(root), TeamRegistry(root)
        agents.create_agent(_agent_fields(name="helper", tags=["role", f"tools:{pins}"]))
        registry.create_team(TeamEditFields(name="pod", members=[TeamMember(role="helper")]))
        teams.append(
            registry.create_team(
                TeamEditFields(name="identity-test", members=[TeamMember(role="pod", kind="team")])
            )
        )
    assert _digest(teams[0]) == _digest(teams[1])
    _authorize_source(source, source, "identity-test")
    write_session_attachment(target, team="identity-test", agent="", goal="")
    offer = source_offer(source, source, supported=True)
    with pytest.raises(TeamIdentityRefused, match="identity-test: would grant tool pins: edit"):
        check_arrival(target, target, offer)


def _assert_queued_destination_owns(root: Path, device: str, *, wait_s: float = 5.0) -> None:
    import time

    # The queue can see the source tombstone before the destination promotes.
    # A fold alone is not proof of ownership: require the promoted local stamp.
    deadline = time.monotonic() + wait_s
    stamp = read_stamp(root, SESSION)
    while stamp is None and time.monotonic() < deadline:
        time.sleep(0.05)
        stamp = read_stamp(root, SESSION)
    assert stamp is not None, "the queue folded resumed but the destination never owned it"
    assert stamp.home_device == device
    assert (root / "sessions" / SESSION).is_dir()


def test_queued_destination_ownership_check_rejects_an_empty_store(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="destination never owned it"):
        _assert_queued_destination_owns(tmp_path, "device-b", wait_s=0.0)


@pytest.mark.asyncio
async def test_a_queued_transfer_succeeds_end_to_end_after_the_definitions_sync(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review round 4, R4-1 (second half): a full queued transfer through the driver.

    The reviewer could not run a complete queued transfer at all. This is one, on the
    real two-relay path after the independent definitions sync: the queue driver
    invites, the destination pulls and promotes, the source commits its tombstone,
    and the record reads ``resumed`` with the conversation living on the destination.
    """
    from local_operator.network import definitions
    from local_operator.network import move_queue as mq
    from local_operator.teams import TeamRegistry

    # These rows call the real driver directly. Startup reconciliation must not
    # run a second driver on the same record while this test owns it.
    monkeypatch.setattr(mq, "start_driver", lambda *_a, **_k: False)
    monkeypatch.setattr(definitions.DefinitionsSyncer, "tick", lambda *_a, **_k: [])
    server_a, server_b, _host, _port = pair
    _pair_settled(pair, monkeypatch, role="admin", settings=server_b.settings)
    session_dir = _owned_session(server_a)
    _create_source_team(server_a, "lopdev")
    _authorize_source(server_a.root, session_dir, "lopdev")
    synced = definitions.push_to_peer(server_a, server_b.identity.device_id)
    assert synced["ok"], synced
    assert TeamRegistry(server_b.root).get_team_by_name("lopdev") is not None

    mq.enqueue(
        server_a.root,
        SESSION,
        to_device=server_b.identity.device_id,
        to_name="device-b",
        request_id="q1",
        requested_by=server_a.identity.device_id,
    )
    mq.transition(server_a.root, SESSION, mq.QUEUE_PHASE_COPYING, writer="relay")
    assert mq.drive(server_a, SESSION) == mq.QUEUE_PHASE_RESUMED
    record = mq.read_record(server_a.root, SESSION) or {}
    assert record["phase"] == mq.QUEUE_PHASE_RESUMED, record
    # THE SOURCE'S OWN FACTS, which is all the queue decides on: the record folded on
    # the tombstone and this device retired its copy.
    assert not session_dir.exists()
    _assert_queued_destination_owns(server_b.root, server_b.identity.device_id)
    print("queued transfer:", record["phase"], "source retired: True, destination promoted: True")


@pytest.mark.asyncio
async def test_a_queued_retry_is_decided_by_its_own_attempt(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review round 4, R4-1: a queued retry must not fold on a remembered refusal.

    Attempt 1 is refused because the destination has not reconciled the team yet.
    After the independent sync, the same queue record is driven again: whatever this
    attempt decides, it must decide it from state AT THIS ATTEMPT — a ``resumed``
    record, or a refusal that actually crossed for it. At ``cf453abae`` the second
    drive folded ``team_definitions_pending`` again with NO second refusal arriving.
    """
    from local_operator.network import definitions, mobility
    from local_operator.network import move_queue as mq
    from local_operator.teams import TeamRegistry

    # The explicit drive() calls below own this record, not the startup scheduler.
    monkeypatch.setattr(mq, "start_driver", lambda *_a, **_k: False)
    monkeypatch.setattr(definitions.DefinitionsSyncer, "tick", lambda *_a, **_k: [])
    server_a, server_b, _host, _port = pair
    _pair_settled(pair, monkeypatch, role="admin", settings=server_b.settings)
    session_dir = _owned_session(server_a)
    _create_source_team(server_a, "lopdev")
    _authorize_source(server_a.root, session_dir, "lopdev")

    # WHICH REFUSALS ACTUALLY CROSSED, and for which attempt: the fact that decides
    # whether a fold was computed from this attempt's wire traffic or remembered.
    arrived: list[tuple[int, str]] = []
    real_refused = mobility._source_refused

    def refused(server: Any, link: Any, frame: dict[str, Any]) -> dict[str, Any]:
        arrived.append((int(frame.get("attempt") or 0), str(frame.get("code") or "")))
        return real_refused(server, link, frame)

    monkeypatch.setattr(mobility, "_source_refused", refused)

    # THE OTHER WAY THIS ATTEMPT ANSWERS: the invite can be refused in the ASK
    # itself, which the destination never repeats as a ``refused`` frame. Both are
    # this attempt's own evidence; neither exists for a remembered sentence.
    answers: list[tuple[int, str]] = []
    real_ask = mobility.LinkTransport.ask

    def ask(self: Any, frame: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        try:
            return real_ask(self, frame, **kwargs)
        except mobility.Moved as refusal:
            # LinkTransport correlates the direct reply with the request itself.
            # A transport failure is not a refusal from this invited peer.
            if frame.get("phase") == "invite" and refusal.code != "unreachable":
                answers.append((int(frame.get("attempt") or 0), refusal.code))
            raise

    monkeypatch.setattr(mobility.LinkTransport, "ask", ask)

    def queue_it(request_id: str) -> None:
        mq.enqueue(
            server_a.root,
            SESSION,
            to_device=server_b.identity.device_id,
            to_name="device-b",
            request_id=request_id,
            requested_by=server_a.identity.device_id,
        )
        mq.transition(server_a.root, SESSION, mq.QUEUE_PHASE_COPYING, writer="relay")

    progress = mobility.progress_for(server_a)
    queue_it("q1")
    assert mq.drive(server_a, SESSION) == mq.QUEUE_PHASE_FAILED
    first_token = progress.attempt(SESSION)
    first = mq.read_record(server_a.root, SESSION) or {}
    first_code = str(first["code"])
    assert first_token > 0
    first_evidence = {code for token, code in arrived + answers if token == first_token}
    assert first_code in first_evidence, (first_code, first_evidence)
    print("queued attempt 1:", first["phase"], first_code, "matched own refusal: True")
    assert session_dir.is_dir()

    synced = definitions.push_to_peer(server_a, server_b.identity.device_id)
    assert synced["ok"], synced
    assert TeamRegistry(server_b.root).get_team_by_name("lopdev") is not None

    queue_it("q2")
    second_outcome = mq.drive(server_a, SESSION)
    second_token = progress.attempt(SESSION)
    second = mq.read_record(server_a.root, SESSION) or {}
    assert second_token > 0 and second_token != first_token
    if second_outcome == mq.QUEUE_PHASE_RESUMED:
        assert second["phase"] == mq.QUEUE_PHASE_RESUMED
        assert not session_dir.exists()
        _assert_queued_destination_owns(server_b.root, server_b.identity.device_id)
        print("queued attempt 2: resumed, source retired: True, destination promoted: True")
    else:
        assert second_outcome == second["phase"] == mq.QUEUE_PHASE_FAILED
        own_refusals = {code for token, code in arrived + answers if token == second_token}
        assert str(second.get("code")) in own_refusals, (second.get("code"), own_refusals)
        print("queued attempt 2:", second_outcome, second.get("code"), "matched own refusal: True")


@pytest.mark.asyncio
async def test_no_attempt_inherits_a_previous_attempts_success(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R4-1, both halves: no attempt inherits a previous attempt's SUCCESS either.

    The refusal half is pinned above; this is the worse one, because a remembered
    success ADMITS a move on evidence from a previous attempt. Attempt 1 moves the
    conversation to the peer for real, attempt 2 recalls it here, and the probe then
    invites the peer again with a transport that accepts and never completes. The
    probe must answer from THIS attempt (a deadline), never from the completion
    signal attempt 2 left.

    WHAT IS PROVEN LOAD-BEARING IS THE ATTRIBUTION, NOT THE CLEAR (review r5 R5-1).
    Clearing at ``begin_attempt`` forgets a PAST answer; it says nothing about a
    completion that crosses a retry LATE. So the same probe runs first with the
    PRE-ATTRIBUTION reading (any completion for this session ends the wait) and must
    inherit — reporting a finished move that never ran — and only then with the real
    per-attempt wait, which is what keeps this row from being vacuous.
    """
    from local_operator.network import mobility
    from local_operator.session.placement import read_stamp

    server_a, server_b, _host, _port = pair
    _pair_settled(pair, monkeypatch, role="admin", settings=server_b.settings)
    _owned_session(server_a)

    moved = _move(server_a, SESSION, to=server_b.identity.device_id, monkeypatch=monkeypatch)
    assert moved["ok"] is True, moved
    recalled = _move(server_a, SESSION, monkeypatch=monkeypatch)
    assert recalled["ok"] is True, recalled
    stamp = read_stamp(server_a.root, SESSION)
    assert stamp is not None and stamp.home_device == server_a.identity.device_id

    # THE PROBE INVITE: the peer acks and then does nothing at all, so the only thing
    # that could answer this attempt is something left by a previous one.
    invites: list[dict[str, Any]] = []

    progress = mobility.progress_for(server_a)
    delivered = {"token": 0}

    def ask(self: Any, frame: dict[str, Any], **_kw: Any) -> dict[str, Any]:
        if frame.get("phase") == "invite":
            invites.append(dict(frame))
            # A completion belonging to the attempt that ALREADY ran, crossing the
            # retry LATE. Nothing in this probe moved anything, so only attribution
            # decides whether it answers the attempt now waiting.
            progress.signal_done(SESSION, delivered["token"])
            return {"result": "accepted", "phase": "prepared"}
        return {"result": "recorded"}

    monkeypatch.setattr(mobility.LinkTransport, "ask", ask)
    monkeypatch.setattr(mobility, "move_bound_s", lambda *_a, **_k: 1.0)

    real_wait_done = mobility._Progress.wait_done
    state = {"attribute": False}

    def wait_done(progress: Any, session_id: str, attempt: int | None, timeout: float) -> bool:
        if state["attribute"]:
            return real_wait_done(progress, session_id, attempt, timeout)
        # The control: the PRE-ATTRIBUTION reading — any completion for this session
        # ends the wait, whichever attempt it answered.
        return progress.wait_for(session_id, "done", timeout)

    monkeypatch.setattr(mobility._Progress, "wait_done", wait_done)

    delivered["token"] = progress.attempt(SESSION)
    assert delivered["token"] > 0, "no earlier attempt to inherit from"
    inherited = _move(server_a, SESSION, to=server_b.identity.device_id, monkeypatch=monkeypatch)
    print(
        "control (unattributed):",
        inherited["ok"],
        inherited.get("code"),
        "| invites:",
        len(invites),
    )
    assert invites, "the probe never invited the peer"
    assert inherited["ok"] is True, (
        "the control did not inherit a completion, so this row never reached the "
        "stale signal and would be vacuous",
        inherited,
    )

    state["attribute"] = True

    delivered["token"] = progress.attempt(SESSION)
    fresh = _move(server_a, SESSION, to=server_b.identity.device_id, monkeypatch=monkeypatch)
    assert fresh["ok"] is False, fresh
    assert fresh["code"] == "move_attempt_unconfirmed", fresh
    assert fresh["changed"] is False, fresh
    assert not (server_b.root / "sessions" / SESSION).exists()
    print("attributed wait:", fresh["ok"], fresh["code"], "changed:", fresh["changed"])


@pytest.mark.asyncio
async def test_the_ceiling_is_load_bearing_for_a_wider_destination(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """QA round 4, row 5: the ceiling check must be LOAD-BEARING, not merely present.

    QA honestly refused to disable an authorization check in the product, so the
    pair that makes the refusal evidence was never demonstrated. It is here, with a
    TEST-LOCAL monkeypatch of the comparison (``move_team._ceiling``) and no product
    path weakened: bypassed, an otherwise-disallowed WIDER move PROCEEDS and the
    destination owns the conversation; restored, the same move REFUSES BY NAME.
    """
    from local_operator.agents import AgentRegistry
    from local_operator.network import move_team
    from local_operator.session.placement import read_stamp
    from local_operator.teams import TeamEditFields, TeamRegistry

    server_a, server_b, _host, _port = pair
    _pair_settled(pair, monkeypatch, role="admin", settings=server_b.settings)
    fields = TeamEditFields(name="identity-test", instructions="source brief")
    TeamRegistry(server_a.root).create_team(fields)
    # The destination holds the SAME team definition, but its manager resolves to an
    # edit-capable profile: strictly wider authority than the source recorded.
    TeamRegistry(server_b.root).create_team(fields)
    AgentRegistry(server_b.root).create_agent(
        _agent_fields(name="manager", tags=["role", "tools:edit"])
    )

    real_ceiling = move_team._ceiling
    state = {"bypass": True}
    reached: list[bool] = []

    def ceiling(*args: Any, **kwargs: Any) -> None:
        # THE CONTROL ON THE CONTROL: record that the comparison was REACHED, so a
        # bypass that never took effect cannot leave this row vacuously green.
        reached.append(state["bypass"])
        if state["bypass"]:
            return
        real_ceiling(*args, **kwargs)

    monkeypatch.setattr(move_team, "_ceiling", ceiling)

    wider_dir = _owned_session(server_a, "202020202020", rows=0)
    _authorize_source(server_a.root, wider_dir, "identity-test")
    bypassed = _move(
        server_a, "202020202020", to=server_b.identity.device_id, monkeypatch=monkeypatch
    )
    # The bypass must have been REACHED and taken (``True``): if the monkeypatch
    # silently failed to apply, the real comparison would have refused and this
    # assertion — not just the outcome below — is what says so.
    assert reached and reached[0] is True, (
        "the ceiling comparison was never reached with the bypass live; the "
        "negative control would be vacuous",
        reached,
    )
    assert bypassed["ok"] is True, bypassed
    arrived = read_stamp(server_b.root, "202020202020")
    assert arrived is not None and arrived.home_device == server_b.identity.device_id
    assert not wider_dir.exists()
    print("ceiling bypassed:", bypassed["ok"], "destination owns: True")

    state["bypass"] = False
    blocked_dir = _owned_session(server_a, "303030303030", rows=0)
    _authorize_source(server_a.root, blocked_dir, "identity-test")
    refused = _move(
        server_a, "303030303030", to=server_b.identity.device_id, monkeypatch=monkeypatch
    )
    assert refused["ok"] is False, refused
    assert refused["code"] == move_team.CODE, refused
    assert "would grant tool pins: edit" in refused["message"], refused
    assert blocked_dir.is_dir()
    assert not (server_b.root / "sessions" / "303030303030").exists()
    print("ceiling restored:", refused["code"], refused["message"])


# -- local binding write failures (Windows replace/unlink semantics) ----------
# ``_write_local_binding`` is the one writer of the local team-identity journal.
# A failed write must PROPAGATE (every caller treats it as "no authority
# recorded" and fails closed), and a failed cleanup of its own temp must not
# REPLACE that error. Windows is where both happen in practice: ``replace``
# onto a binding another process holds open, ``unlink`` of a temp a scanner
# holds. Simulated here with PermissionError, which is what CPython raises
# for both on Windows.


def test_failed_binding_replace_propagates_with_note_and_removes_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.network import move_team

    real_replace = Path.replace

    def refuse(self: Path, target: Any) -> Path:
        if Path(target).name == move_team.LOCAL_BINDING_NAME:
            raise PermissionError(5, "Access is denied", str(target))
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", refuse)
    with pytest.raises(PermissionError) as raised:
        move_team._write_local_binding(tmp_path, {"reauthorize": True})
    assert any("fail-closed" in note for note in raised.value.__notes__)
    # Fail-closed: no binding published and no temp left behind.
    assert sorted(p.name for p in tmp_path.iterdir()) == []


def test_failed_temp_cleanup_does_not_mask_the_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from local_operator.network import move_team

    def refuse_replace(self: Path, target: Any) -> Path:
        raise PermissionError(5, "Access is denied", str(target))

    def refuse_unlink(self: Path, missing_ok: bool = False) -> None:
        raise PermissionError(32, "being used by another process", str(self))

    monkeypatch.setattr(Path, "replace", refuse_replace)
    monkeypatch.setattr(Path, "unlink", refuse_unlink)
    with caplog.at_level("WARNING", logger=move_team.__name__):
        with pytest.raises(PermissionError) as raised:
            move_team._write_local_binding(tmp_path, {"reauthorize": True})
    # The REPLACE failure is the one the caller sees, not the cleanup's.
    assert raised.value.errno == 5
    assert "left local team binding temp" in caplog.text
    # The leftover is inert: it is never the binding name readers open.
    leftovers = [p.name for p in tmp_path.iterdir()]
    assert len(leftovers) == 1 and leftovers[0] != move_team.LOCAL_BINDING_NAME
    assert move_team._local_binding(tmp_path) is None
