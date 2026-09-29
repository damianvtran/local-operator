"""Recall recovery: the bare remnant a viewer left must not strand the recall.

THE STRAND (measured live 2026-09-29, the seamless-remote-offload E2E; the fix's
PR carries the live investigation): a session moved to a peer, stopped there,
then recalled home — and the recall answered twice. First ``"<id> could not be
adopted here because a session with that id already exists on this device"``
(the promote collision), then ``"<id> is already on this device, so nothing was
moved"`` (``_owned_here`` claiming a stamp-less directory), and ``lop sessions
sync`` denied the id existed anywhere — while the only complete copy sat in
this device's own staging area. The directory in the way was a BARE REMNANT:
``created_at.json`` and nothing else, materialised when a viewer ON THIS DEVICE
ran its owner-lost verdict for the peer's session (the writer's own cell lives
in ``test_remote_viewer.py``; the predicate that recognises the corpse and the
promote that now lands over it are in ``network/mobility.py``).

WHAT THESE CELLS PIN, one per state the operator hit:

* a recall lands over a bare remnant already sitting at its target;
* a recall lands when the remnant ARRIVES while the verified copy is in flight
  (the first refusal, rebuilt by position in the flow rather than by seconds);
* §6.5's staged-copy recovery adopts over the remnant too — the state the
  operator cleared by hand, with the removal now happening where it belongs;
* ``lop sessions sync`` names the verified staged copy instead of denying it;
* a directory that holds a real conversation is still refused untouched, so
  the promote's clearing rule cannot widen into "overwrite anything" —
  including one stamped for another device (the crash-window leftover).
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import mobility, sync
from local_operator.network.projection import write_tombstone
from local_operator.resume import ORIGIN_FORK, mark_session_origin
from local_operator.session.placement import (
    MeshStamp,
    SessionPlacement,
    read_stamp,
    write_stamp,
    write_stamp_into,
)
from tests.unit.network.test_mobility import (  # noqa: F401 — fixtures and helpers
    SESSION,
    Devices,
    _move,
    _owned_session,
    pair,
)
from tests.unit.network.test_relay_e2e import _pair_settled, devices  # noqa: F401


def _plant_remnant(root: Path, session_id: str) -> Path:
    """The EXACT corpse the viewer left: ``sessions/<id>/created_at.json``.

    Same value shape the operator's snapshot holds (17 B, the writer's
    ``time.time()``), written the way ``ensure_session_created_at`` publishes it.
    """
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "created_at.json").write_text(json.dumps(1790663404.232095), encoding="utf-8")
    return directory


def _offload_to_peer(pair_devices: Devices, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
    """Pair, own a session on A, and move it to B the way `--to <peer>` does."""
    server_a, server_b, _host, _port = pair_devices
    _pair_settled(pair_devices, monkeypatch, role="admin", settings=server_b.settings)
    _owned_session(server_a)
    moved = _move(server_a, SESSION, to=server_b.identity.device_id, monkeypatch=monkeypatch)
    assert moved["ok"] is True, moved
    assert (server_b.root / "sessions" / SESSION / "transcript.jsonl").is_file()
    return server_a, server_b


def _assert_adopted_home(server_a: Any, server_b: Any, before: bytes) -> None:
    """The conversation is WHOLE on A, and one holder exists — INV-1's own check."""
    home = server_a.root / "sessions" / SESSION
    assert (
        home / "transcript.jsonl"
    ).read_bytes() == before, "the conversation that landed is not the bytes the peer verified"
    stamp = read_stamp(server_a.root, SESSION)
    assert stamp is not None and stamp.home_device == server_a.identity.device_id
    assert not (
        server_b.root / "sessions" / SESSION
    ).exists(), "the device that handed the session back still holds a copy: two devices, one id"


# ---------------------------------------------------------------------------
# The refusal the operator hit second: the remnant is already at the target
# ---------------------------------------------------------------------------


def test_a_recall_lands_over_a_bare_remnant_already_at_its_target(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pre-fix this answered ``already_local`` — "already on this device" — about
    a directory holding nothing, dead-ending the recall of the id whose only
    complete copy was on the peer."""
    server_a, server_b = _offload_to_peer(request.getfixturevalue("pair"), monkeypatch)
    before = (server_b.root / "sessions" / SESSION / "transcript.jsonl").read_bytes()
    remnant = _plant_remnant(server_a.root, SESSION)
    assert sorted(p.name for p in remnant.iterdir()) == ["created_at.json"]

    recalled = _move(server_a, SESSION, monkeypatch=monkeypatch)

    assert recalled["ok"] is True, recalled
    assert recalled["new_session_id"] == SESSION
    _assert_adopted_home(server_a, server_b, before)
    assert not (
        remnant / "created_at.json"
    ).exists(), "the remnant outlived the promote that landed on it"


def test_a_failed_corpse_clear_refuses_the_recall_and_keeps_the_remnant(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed clear is a REFUSED promote, never a raise.

    Pre-delta ``_promote`` wrapped its own ``rmtree``, warned and returned
    ``False``. Routing the clear through ``remove_session_dir`` (whose
    ``rmtree`` is deliberately unguarded — ``apply_cleanup`` wraps the same call
    in the same shape) briefly let a real I/O failure — the target racing away,
    a permission error — escape the promote before its ``os.replace``, where the
    relay's control loop reads it as a dead connection and answers
    ``relay_unavailable`` about a relay that is actually serving (PR #1756
    round 3, MINOR). The remnant must survive and the recall must come back as
    the refusal it always was.
    """
    server_a, server_b = _offload_to_peer(request.getfixturevalue("pair"), monkeypatch)
    remnant = _plant_remnant(server_a.root, SESSION)
    before = (remnant / "created_at.json").read_bytes()

    real_rmtree = shutil.rmtree

    def fail_for_the_remnant_only(path: Any, *args: Any, **kwargs: Any) -> None:
        # Scoped to THIS target on purpose: any other rmtree a relay thread
        # happens to run during the window keeps its real behaviour, so the
        # fault injected is exactly the corpse-clear's.
        if Path(path) == remnant:
            raise OSError("simulated I/O failure while clearing the remnant")
        real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", fail_for_the_remnant_only)

    recalled = _move(server_a, SESSION, monkeypatch=monkeypatch)

    assert recalled["ok"] is False, recalled
    # THE PRE-DELTA CONTRACT, exactly: a failed clear reaches the caller as the
    # ordinary "could not be adopted" refusal — not as ``relay_unavailable``
    # ("a wedged relay"), which is what an escape past the promote produced.
    assert (
        recalled["code"] == "in_progress"
    ), f"an I/O failure in the corpse-clear must refuse, not read as a dead relay: {recalled}"
    assert (remnant / "created_at.json").read_bytes() == before, (
        "the remnant must survive a failed clear — a promote never writes over "
        "what it could not clear"
    )
    assert not (remnant / "transcript.jsonl").exists()


# ---------------------------------------------------------------------------
# The refusal the operator hit FIRST: the remnant lands while the copy is in flight
# ---------------------------------------------------------------------------


def test_a_recall_lands_when_the_remnant_arrives_before_its_promote(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live ordering: the viewer's write and the recall raced, and the corpse
    won the window between the owner's commit and the destination's ``promote``.

    Rebuilt by POSITION IN THE FLOW rather than by seconds: ``_verify_staging`` is
    the step between the ready-ack and the one ``os.replace``, so planting the
    remnant there is the same window the operator's viewer raced — pre-fix this
    answered ``"<id> could not be adopted here because a session with that id
    already exists on this device"`` while the verified copy sat staged."""
    pair_devices: Devices = request.getfixturevalue("pair")
    server_a, server_b = _offload_to_peer(pair_devices, monkeypatch)
    before = (server_b.root / "sessions" / SESSION / "transcript.jsonl").read_bytes()

    original_verify = mobility._verify_staging  # noqa: SLF001 — the ordered step

    def planting_verify(server: Any, session_id: str, staging: Path, ready: dict[str, Any]) -> str:
        result = original_verify(server, session_id, staging, ready)
        # THE REMNANT LANDS HERE (the viewer's write, mid-flight).
        _plant_remnant(Path(server.root), session_id)
        return result

    monkeypatch.setattr(mobility, "_verify_staging", planting_verify)

    recalled = _move(server_a, SESSION, monkeypatch=monkeypatch)

    assert recalled["ok"] is True, recalled
    _assert_adopted_home(server_a, server_b, before)


# ---------------------------------------------------------------------------
# §6.5 row 5, over the remnant: the state that needed manual surgery
# ---------------------------------------------------------------------------


def test_a_staged_copy_is_adopted_over_the_remnant_it_stranded_behind(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The operator's exact end state, constructed: the peer committed the move
    back to this device (its tombstone names A, its directory is gone), the
    verified copy sits in A's staging, and the viewer's remnant is at the target.

    The recovery the operator had to perform by hand — snapshot the remnant,
    delete it, recall — with the deletion now happening where it belongs (the
    promote), and the tombstone-and-staging proof unchanged (the digest here is
    the product's own, so the cell cannot accidentally test the digest gate)."""
    server_a, server_b = _offload_to_peer(request.getfixturevalue("pair"), monkeypatch)
    before = (server_b.root / "sessions" / SESSION / "transcript.jsonl").read_bytes()

    # THE CRASH WINDOW, constructed: B's copy goes back into A's staging with the
    # ready.json the receiver writes, and B's tombstone names A. The digest is the
    # product's own, computed before the stamp and lineage land — the order
    # ``_destination_move`` itself uses — so the cell cannot accidentally test the
    # digest gate.
    staged = sync.staging_dir(server_a.root, SESSION)
    staged.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(server_b.root / "sessions" / SESSION), str(staged))
    content_digest = mobility._staging_content_digest(server_a, SESSION, staged)  # noqa: SLF001
    # THE DESTINATION'S OWN STAMP AND LINEAGE, inside the staging directory: this
    # is what ``_destination_move`` leaves when it dies between its copy and its
    # one ``os.replace`` — the copy under A's ownership, not B's.
    write_stamp_into(
        staged,
        MeshStamp(
            session_id=SESSION,
            network_id="n_test",
            home_device=server_a.identity.device_id,
            placement=SessionPlacement(
                mode="peer",
                network_id="n_test",
                home_device=server_a.identity.device_id,
                policy="pinned",
                stamp_revision=1,
            ),
            origin={
                "kind": "moved",
                "source_device": server_b.identity.device_id,
                "source_session_id": SESSION,
                "moved_at": time.time(),
            },
        ),
    )
    mark_session_origin(
        staged, ORIGIN_FORK, parent=SESSION, source_device=server_b.identity.device_id
    )
    (staged / "ready.json").write_text(
        json.dumps(
            {
                "version": 1,
                "lease_epoch": "e_constructed",
                "content_digest": content_digest,
                "plan_id": "",
                "mode": "move",
                "owner_device": server_b.identity.device_id,
                "source_session_id": SESSION,
                "archived": False,
                "promoted": False,
                "at": time.time(),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    write_tombstone(
        SESSION,
        device_id=server_a.identity.device_id,
        device_name=server_a.identity.name,
        network_id="n_test",
        config_dir=server_b.root,
    )
    _plant_remnant(server_a.root, SESSION)

    recalled = _move(server_a, SESSION, monkeypatch=monkeypatch)

    assert recalled["ok"] is True, recalled
    assert recalled.get("recovered") is True, recalled
    _assert_adopted_home(server_a, server_b, before)
    assert not staged.exists(), "the promote consumes the staging directory"


# ---------------------------------------------------------------------------
# `lop sessions sync`: never deny bytes this device is holding
# ---------------------------------------------------------------------------


def test_a_sync_refusal_names_the_verified_staged_copy(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pre-fix this answered ``"no device in this network holds <id>, so nothing
    was moved"`` about bytes in this device's own staging area — the operator's
    attempt 3. The clause appears only for a VERIFIED copy (``ready.json``)."""
    server_a, server_b = _offload_to_peer(request.getfixturevalue("pair"), monkeypatch)

    staged = sync.staging_dir(server_a.root, SESSION)
    staged.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(server_b.root / "sessions" / SESSION), str(staged))
    (staged / "ready.json").write_text(json.dumps({"at": time.time()}), encoding="utf-8")
    write_tombstone(
        SESSION,
        device_id=server_a.identity.device_id,
        device_name=server_a.identity.name,
        network_id="n_test",
        config_dir=server_b.root,
    )
    _plant_remnant(server_a.root, SESSION)

    refused = sync.request_sync(SESSION, root=server_a.root)

    assert refused["ok"] is False, refused
    # The refusal's OWN code, split off ``unreachable`` (R1 design finding D1) so
    # the clause gate can key on it without parsing the sentence.
    assert refused["code"] == "no_holder", refused
    message = str(refused.get("message") or "")
    assert f"network/staging/{SESSION}" in message, message
    assert f"lop sessions move {SESSION} --to local" in message, message
    assert staged.is_dir(), "naming the copy must not consume it"

    # AND WITH NO VERIFIED COPY THE CLAUSE IS ABSENT: nothing points at a
    # staging directory that holds nothing.
    ready = staged / "ready.json"
    ready.unlink()
    plain = sync.request_sync(SESSION, root=server_a.root)
    assert plain["ok"] is False, plain
    assert f"network/staging/{SESSION}" not in str(plain.get("message") or ""), plain


def test_the_peer_unreachable_refusal_gets_no_staged_copy_clause(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Design review round 1, D1: the clause is gated to the NO-HOLDER refusal.

    Both ``resolve_remote_owner`` refusals used to carry the code
    ``unreachable``, so the round-1 clause also composed onto "cloud-node-1 is
    not answering right now (last seen 42s ago); nothing was changed" — a state
    where ``lop sessions move <id> --to local`` cannot complete while the peer
    stays down (adoption still needs the owner through
    ``_reconcile_destination``), so the advice promised what nothing could keep.
    The state fed here is §8.3's shape for a peer that stopped answering — a
    peer block with ``reachable: false`` — crafted to the contract because the
    live fan-out emits NO rows for such a peer ("a listing must not show
    phantom rows", relay.py's ``_fan_out_catalog``), which is exactly why the
    gate must not depend on the live fan-out being able to produce it. With a
    verified staged copy present, the refusal must still arrive WITHOUT the
    clause; the no-row case (the same device, no row at all) must keep it.
    """
    server_a, server_b = _offload_to_peer(request.getfixturevalue("pair"), monkeypatch)

    staged = sync.staging_dir(server_a.root, SESSION)
    staged.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(server_b.root / "sessions" / SESSION), str(staged))
    (staged / "ready.json").write_text(json.dumps({"at": time.time()}), encoding="utf-8")

    unreachable_row = {
        "session_id": SESSION,
        "locality": "remote",
        "peer": {
            "device_id": server_b.identity.device_id,
            "name": "cloud-node-1",
            "reachable": False,
            "age_s": 42,
        },
    }
    monkeypatch.setattr(
        server_a, "federated_rows", lambda: {"sessions": [unreachable_row], "peers": {}}
    )
    refused = sync.local_sync_handler(server_a)({"session_id": SESSION})

    assert refused["ok"] is False, refused
    assert refused["code"] == "unreachable", refused
    message = str(refused.get("message") or "")
    assert "not answering right now" in message, message
    assert f"network/staging/{SESSION}" not in message, (
        "the clause composed onto the peer-unreachable refusal, where the named "
        "verb cannot complete while the peer stays down"
    )

    # THE NO-ROW CASE STILL NAMES IT: same device, same verified copy, and its
    # own code — the other half of the gate, read in one cell so the decision is
    # the pair rather than either half alone.
    monkeypatch.delattr(server_a, "federated_rows")
    write_tombstone(
        SESSION,
        device_id=server_a.identity.device_id,
        device_name=server_a.identity.name,
        network_id="n_test",
        config_dir=server_b.root,
    )
    named = sync.request_sync(SESSION, root=server_a.root)
    assert named["ok"] is False, named
    assert named["code"] == "no_holder", named
    assert f"network/staging/{SESSION}" in str(named.get("message") or ""), named


# ---------------------------------------------------------------------------
# The promote's old contract, unchanged for anything that could be a session
# ---------------------------------------------------------------------------


def test_a_recall_refuses_over_a_directory_that_holds_a_conversation(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transcript-bearing directory at the target — here one stamped for the
    PEER, the crash-window leftover §6.5 describes — is not a bare remnant: the
    recall refuses exactly as it always has and touches nothing. This is the
    guard against the clearing rule widening into "overwrite anything"."""
    server_a, server_b = _offload_to_peer(request.getfixturevalue("pair"), monkeypatch)

    foreign = server_a.root / "sessions" / SESSION
    foreign.mkdir(parents=True)
    (foreign / "transcript.jsonl").write_text(
        json.dumps({"id": "x", "type": "user", "content": "not this device's"}) + "\n",
        encoding="utf-8",
    )
    write_stamp(
        server_a.root,
        MeshStamp(
            session_id=SESSION,
            network_id="n_test",
            home_device=server_b.identity.device_id,
            placement=SessionPlacement(
                mode="peer",
                network_id="n_test",
                home_device=server_b.identity.device_id,
                stamp_revision=1,
            ),
        ),
    )
    before = (foreign / "transcript.jsonl").read_bytes()

    recalled = _move(server_a, SESSION, monkeypatch=monkeypatch)

    assert recalled["ok"] is False, recalled
    assert (
        foreign / "transcript.jsonl"
    ).read_bytes() == before, "the refusal must leave the directory exactly as it found it"
    assert read_stamp(server_a.root, SESSION) is not None
