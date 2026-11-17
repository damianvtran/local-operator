"""G5/G6 on the relay: per-link ``peer_verified``, the peer block, the viewer drop,
the create carriage, and the promote downgrades — plus the end-to-end relayed
prompt from a REAL second relay (design §5 tests 3b, 8, 9b, 15, 16).

Design: ``docs/design/mesh-origin-stamp.md``.
"""

from __future__ import annotations

import ast
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from local_operator.network import mobility, origin, relay, sync
from local_operator.network.types import MeshRefusal
from tests.unit.network.test_audit_io import _pair

RELAY_SOURCE = Path(relay.__file__).read_text(encoding="utf-8")


def _links(server_a: Any, server_b: Any, network_id: str) -> tuple[Any, Any]:
    """(A's INBOUND listener link from B, B's OUTBOUND dialer link to A)."""
    inbound = next(
        held
        for held in server_a.links.values()
        if held.device_id == server_b.identity.device_id and held.alive
    )
    outbound = next(
        held for held in server_b.links.values() if held.network_id == network_id and held.alive
    )
    return inbound, outbound


# ---------------------------------------------------------------------------
# 3b: the three PeerLink build sites
# ---------------------------------------------------------------------------


def test_3b_listener_link_is_peer_verified_and_dialer_link_is_not(root: Path) -> None:
    server_a, server_b, record = _pair(root)
    try:
        inbound, outbound = _links(server_a, server_b, record.network_id)
        assert inbound.outbound is False and inbound.peer_verified is True  # relay.py listener
        assert outbound.outbound is True and outbound.peer_verified is False  # dialer, pre-G1
        block = server_a._peer_block(inbound)
        assert block == {
            "device_id": server_b.identity.device_id,
            "name": block["name"],
            "network_id": record.network_id,
            "link_id": inbound.link_id,
            "peer_verified": True,
        }
        assert server_b._peer_block(outbound)["peer_verified"] is False
    finally:
        server_b.stop()
        server_a.stop()


def _peer_link_calls() -> list[tuple[str, ast.Call]]:
    tree = ast.parse(RELAY_SOURCE)
    calls: list[tuple[str, ast.Call]] = []
    for function in ast.walk(tree):
        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for node in ast.walk(function):
                if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "PeerLink":
                    calls.append((function.name, node))
    return calls


def test_3b_every_peer_link_site_states_peer_verified_explicitly() -> None:
    """Three sites (R2-4): listener register and pair listener take the handshake's
    proved fact; the dialer pins False until G1 verifies the listener."""
    sites = {}
    for name, call in _peer_link_calls():
        values = {kw.arg: ast.unparse(kw.value) for kw in call.keywords}
        assert "peer_verified" in values, f"{name} builds a PeerLink without peer_verified"
        sites.setdefault(name, []).append(values["peer_verified"])
    assert len(_peer_link_calls()) == 3
    dialer = [value for value in sites.values() if value == ["False"]]
    listeners = [value for value in sites.values() if value == ["result.peer_verified"]]
    assert len(dialer) == 1 and len(listeners) == 2, sites


def test_handshake_result_carries_peer_verified_from_verify_auth() -> None:
    from local_operator.network import handshake, types

    assert types.HandshakeResult.__dataclass_fields__["peer_verified"].default is False
    source = Path(handshake.__file__).read_text(encoding="utf-8")
    assert "peer_verified=self.auth_verified" in source


# ---------------------------------------------------------------------------
# 9b: a prompt relayed from device B over a real link verifies as B
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_9b_relayed_prompt_from_b_verifies_as_b_end_to_end(
    root: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """B's mobile daemon, pilot, ``network sessions --send`` and agent ``send`` all
    reach A as ``net_forward``/``net_stream`` over B's link, and A's relay dials its
    runtime with ``_peer_block(link)``. Driven here through ``_op_forward``."""
    from tests.unit.session.runtime.test_approval_authority_seam import _serve
    from tests.unit.session.runtime.test_mesh_origin_runtime import _recorder

    server_a, server_b, record = _pair(root)
    live = None
    try:
        inbound, _outbound = _links(server_a, server_b, record.network_id)
        # A's runtime shares A's config root, exactly as on a real device: it is how
        # both sides derive the same K_relay from the one device seed.
        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_a.root))
        live = await _serve(tmp_path, operator_cap=None)
        seen = _recorder(live)
        session_id = live.record.session_id
        reply = await asyncio.to_thread(
            server_a._op_forward,
            inbound,
            {"req": 1, "frame": {"op": "prompt", "req": 9, "session_id": session_id, "text": "hi"}},
        )
        assert reply, reply
        stamp = seen[-1]["mesh_origin"]
        assert stamp["verified"] is True
        assert stamp["origin_device_id"] == server_b.identity.device_id
        assert stamp["network_id"] == record.network_id
        assert stamp["link_id"] == inbound.link_id
        assert stamp["stamped_by"] == server_a.identity.device_id
        assert "operator_origin" not in stamp
    finally:
        if live is not None:
            await live.close(server_a.root)
        server_b.stop()
        server_a.stop()


# ---------------------------------------------------------------------------
# 15: linkless dial vs create; 16: viewer attest dropped
# ---------------------------------------------------------------------------


class _Dial:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.eof = False

    def exchange(self, frame: dict[str, Any], timeout_s: float = 0.0) -> Any:
        self.sent.append(frame)
        return {"op": "ack", "req": frame.get("req"), "detail": "prompt admitted"}

    def send(self, frame: dict[str, Any]) -> None:
        self.sent.append(frame)

    def close(self) -> None:
        return None


def test_15_create_passes_its_link_to_the_first_prompt_dial(root: Path) -> None:
    server = relay.RelayServer(
        root=root, settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1")
    )
    calls: list[dict[str, Any]] = []
    dial = _Dial()

    def dialled(session_id: str, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return dial

    server._dial_owned = dialled  # type: ignore[assignment]
    link = SimpleNamespace(device_id="dev-b")
    server._prompt_on("sess00000001", "hello", None, link=cast(Any, link))
    assert calls[-1]["link"] is link
    server._prompt_on("sess00000001", "hello", None)
    assert calls[-1].get("link") is None  # linkless ⇒ no peer block ⇒ unverified

    tree = ast.parse(RELAY_SOURCE)
    create = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_op_session_create"
    )
    prompt_calls = [
        node
        for node in ast.walk(create)
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "_prompt_on"
    ]
    assert prompt_calls and all(
        any(kw.arg == "link" for kw in call.keywords) for call in prompt_calls
    )


def test_15_a_linkless_dial_sends_no_peer_block() -> None:
    server = object.__new__(relay.RelayServer)
    server.root = Path("/nonexistent")
    seen: dict[str, Any] = {}

    def fake_dial_owner(root: Any, session_id: str, **kwargs: Any) -> Any:
        seen.update(kwargs)
        return _Dial(), {}

    server._session_record = lambda session_id: None  # type: ignore[method-assign]
    original = relay.session_dial.dial_owner
    relay.session_dial.dial_owner = fake_dial_owner  # type: ignore[assignment]
    try:
        server._dial_owned("sess00000001", capabilities=())
    finally:
        relay.session_dial.dial_owner = original  # type: ignore[assignment]
    assert seen["peer"] is None


def test_16_a_viewer_relay_attest_is_dropped_by_the_owner_relay() -> None:
    server = object.__new__(relay.RelayServer)
    dial = _Dial()
    stream = SimpleNamespace(closed=False, dial=dial)
    server._stream_for = lambda link, stream_id: stream  # type: ignore[method-assign]
    link = SimpleNamespace(device_id="dev-b", context=SimpleNamespace(capabilities={"prompt"}))
    with pytest.raises(MeshRefusal):
        server._accept_stream_frame(
            cast(Any, link), {"frame": {"op": origin.RELAY_ATTEST_OP, "proof": "00" * 32}}, "s1"
        )
    assert dial.sent == []


# ---------------------------------------------------------------------------
# 8: move and replica promote downgrade EVERY imported stamp
# ---------------------------------------------------------------------------

OWN = {
    "origin_device_id": "dev-b",
    "network_id": "net-1",
    "link_id": "ab",
    "capabilities": ["prompt"],
    "verified": True,
    "stamped_by": "dev-a",
}


def _stamped_row() -> str:
    return json.dumps(
        {
            "id": "e-mesh",
            "type": "message",
            "payload": {"role": "user", "provider_payload": {"lop_mesh_origin": OWN}},
        }
    )


def test_8_move_promote_downgrades_imported_stamps(tmp_path: Path) -> None:
    root = tmp_path / "config"
    staged = root / "network" / "staging" / "staged"
    staged.mkdir(parents=True)
    (staged / "transcript.jsonl").write_text(_stamped_row() + "\n", encoding="utf-8")
    (staged / "asks.jsonl").write_text(
        json.dumps({"kind": "answered", "by": {"surface": "remote", "mesh_origin": OWN}}) + "\n",
        encoding="utf-8",
    )
    outcome = mobility._promote(cast(Any, SimpleNamespace(root=root)), staged, "moved0000001")
    assert outcome.ok is True, outcome
    target = root / "sessions" / "moved0000001"
    row = json.loads((target / "transcript.jsonl").read_text(encoding="utf-8"))
    assert row["payload"]["provider_payload"]["lop_mesh_origin"] == {**OWN, "verified": False}
    answer = json.loads((target / "asks.jsonl").read_text(encoding="utf-8"))
    assert answer["by"]["mesh_origin"]["verified"] is False
    assert not origin.mesh_origin_counts(answer["by"]["mesh_origin"])


def test_8_replica_promote_downgrades_imported_stamps(tmp_path: Path) -> None:
    from tests.unit.network.test_sync import _ask, seed

    root = tmp_path / "holder"
    root.mkdir()
    seed(root, "9f3ac1e0b7d2")
    transcript = root / "sessions" / "9f3ac1e0b7d2" / "transcript.jsonl"
    with transcript.open("a", encoding="utf-8") as handle:
        handle.write(_stamped_row() + "\n")
    sync.sync_from(
        root,
        "9f3ac1e0b7d2",
        ask=_ask(root),
        owner_device="d_owner",
        into=sync.replica_dir(root, "9f3ac1e0b7d2"),
    )
    (root / "sessions" / "9f3ac1e0b7d2").rename(tmp_path / "elsewhere")
    promoted = sync.promote_replica(root, "9f3ac1e0b7d2")
    target = root / "sessions" / promoted["session_id"]
    stamps = [
        json.loads(line)["payload"]["provider_payload"]["lop_mesh_origin"]
        for line in (target / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
        if "lop_mesh_origin" in line
    ]
    assert stamps and all(stamp["verified"] is False for stamp in stamps)


# ---------------------------------------------------------------------------
# Security review round 1 (S-1, S-2): BOTH import doors fail closed on ANY error
# in the downgrade, and on a compressed journal it cannot lower. "Closed" means:
# a refusal (not an exception escaping into the relay's frame handler), nothing
# under ``sessions/`` that still reads ``verified: true``, and no leaked claim.
# ---------------------------------------------------------------------------


def _deep_row() -> str:
    deep: Any = []
    for _ in range(1500):
        deep = [deep]
    row = json.loads(_stamped_row())
    row["payload"]["deep"] = deep
    return json.dumps(row)


def _assert_no_verified_copy(root: Path) -> None:
    import gzip

    sessions = root / "sessions"
    leftovers: list[str] = []
    for path in sessions.rglob("*") if sessions.exists() else ():
        if path.is_symlink() or not path.is_file():
            continue
        if path.name == ".session.pid":
            leftovers.append(f"claim {path.relative_to(root)}")
            continue
        raw = path.read_bytes()
        if path.name.endswith(".gz"):
            raw = gzip.decompress(raw)
        if b'"verified": true' in raw:
            leftovers.append(f"verified {path.relative_to(root)}")
    assert leftovers == []


def _fail_on_second_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lower the first stamped file for real, then raise: a walk failing PARTWAY."""
    real = origin._downgrade_file
    seen: list[Path] = []

    def flaky(path: Path) -> int:
        seen.append(path)
        if len(seen) > 1:
            raise OSError("injected: disk went away mid-downgrade")
        return real(path)

    monkeypatch.setattr(origin, "_downgrade_file", flaky)


def _staged_move(root: Path, *, deep: bool = False, archive: str = "") -> Path:
    import gzip

    staged = root / "network" / "staging" / "staged"
    staged.mkdir(parents=True)
    row = (_deep_row() if deep else _stamped_row()) + "\n"
    if archive:
        body = row.encode()
        (staged / archive).write_bytes(gzip.compress(body) if archive.endswith(".gz") else body)
    else:
        (staged / "transcript.jsonl").write_text(row, encoding="utf-8")
    (staged / "asks.jsonl").write_text(
        json.dumps({"kind": "answered", "by": {"surface": "remote", "mesh_origin": OWN}}) + "\n",
        encoding="utf-8",
    )
    return staged


@pytest.mark.parametrize(
    "case", ["deep", "oserror_partway", "transcript.jsonl.gz", "transcript.jsonl.zst"]
)
def test_s1_s2_move_promote_refuses_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    root = tmp_path / "config"
    archive = case if case.startswith("transcript") else ""
    staged = _staged_move(root, deep=case == "deep", archive=archive)
    if case == "deep":
        # The deep row must not refuse on its own: it is downgraded and promoted.
        outcome = mobility._promote(cast(Any, SimpleNamespace(root=root)), staged, "moved0000001")
        assert outcome.ok is True, outcome
        _assert_no_verified_copy(root)
        return
    if case == "oserror_partway":
        _fail_on_second_file(monkeypatch)
    outcome = mobility._promote(cast(Any, SimpleNamespace(root=root)), staged, "moved0000001")
    # Q-1: refused with its OWN outcome, so the reply names the real cause.
    assert outcome == mobility._PROMOTE_STAMPS_REFUSED
    assert not (root / "sessions" / "moved0000001").exists()
    _assert_no_verified_copy(root)


def _replica(tmp_path: Path, *, deep: bool = False) -> tuple[Path, str]:
    from tests.unit.network.test_sync import _ask, seed

    root = tmp_path / "holder"
    root.mkdir()
    seed(root, "9f3ac1e0b7d2")
    transcript = root / "sessions" / "9f3ac1e0b7d2" / "transcript.jsonl"
    with transcript.open("a", encoding="utf-8") as handle:
        handle.write((_deep_row() if deep else _stamped_row()) + "\n")
    asks = root / "sessions" / "9f3ac1e0b7d2" / "asks.jsonl"
    asks.write_text(
        json.dumps({"kind": "answered", "by": {"surface": "remote", "mesh_origin": OWN}}) + "\n",
        encoding="utf-8",
    )
    sync.sync_from(
        root,
        "9f3ac1e0b7d2",
        ask=_ask(root),
        owner_device="d_owner",
        into=sync.replica_dir(root, "9f3ac1e0b7d2"),
    )
    (root / "sessions" / "9f3ac1e0b7d2").rename(tmp_path / "elsewhere")
    return root, "9f3ac1e0b7d2"


def test_s1_replica_promote_downgrades_a_1500_deep_row(tmp_path: Path) -> None:
    root, session_id = _replica(tmp_path, deep=True)
    promoted = sync.promote_replica(root, session_id, new_id="recovered0001")
    assert promoted["session_id"] == "recovered0001"
    _assert_no_verified_copy(root)


@pytest.mark.parametrize("case", ["oserror_partway", "recursion", "transcript.jsonl.gz"])
def test_s1_s2_replica_promote_refuses_and_quarantines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    import gzip

    root, session_id = _replica(tmp_path)
    if case == "oserror_partway":
        _fail_on_second_file(monkeypatch)
    elif case == "recursion":
        # Any exception, not only OSError: the class S-1 escaped the handler with.
        def boom(path: Path) -> int:
            raise RecursionError("injected")

        monkeypatch.setattr(origin, "_downgrade_file", boom)
    else:
        # A HOSTILE owner's plan names the archive, so its cursor covers it (an
        # honest owner restores the plain file first; a real sync_from of a gz
        # session lands ``transcript.jsonl``). The cursor check is stubbed to stand
        # in for that hostile plan: what is under test is the door, not the check.
        replica = sync.replica_dir(root, session_id)
        (replica / case).write_bytes(gzip.compress((_stamped_row() + "\n").encode()))
        monkeypatch.setattr(sync, "_verify_replica_against_cursor", lambda *a, **k: None)
    with pytest.raises(sync.SyncRefused):
        sync.promote_replica(root, session_id, new_id="recovered0001")
    assert not (root / "sessions" / "recovered0001").exists()
    _assert_no_verified_copy(root)
    # The refused copy is kept for inspection, OUTSIDE ``sessions/`` and unclaimed.
    quarantined = sync.network_dir(root) / sync.QUARANTINE_DIRNAME / "recovered0001"
    assert quarantined.is_dir()
    assert not (quarantined / ".session.pid").exists()


# ---------------------------------------------------------------------------
# Review round 2 (R2-1): the decoder's depth limit follows the CALLER's stack
# depth (a C recursion counter on 3.12, the stack pointer on 3.14), so a line too
# deep for the import door can still decode for a reader running shallower. The
# door must refuse such a line, never skip it with ``verified: true`` in place.
# ---------------------------------------------------------------------------

#: C frames the import runs under. Each level goes through ``map`` (a C call back
#: into Python), which lowers the decoder's limit on both 3.12 and 3.14; plain
#: Python recursion does not on 3.12. 50 levels measured ~100 decode levels of gap
#: on 3.12.15 and ~350 on 3.14.9, far more than the door's own extra frames need.
_IMPORT_C_DEPTH = 50


def _c_nested(levels: int, fn: Any) -> Any:
    if levels == 0:
        return fn()
    return list(map(lambda _: _c_nested(levels - 1, fn), [0]))[0]


def _decode_limit() -> int:
    """Deepest ``[[...]]`` json.loads accepts at the CURRENT stack depth."""

    def decodes(depth: int) -> bool:
        try:
            json.loads("[" * depth + "]" * depth)
        except RecursionError:
            return False
        return True

    low, high = 1, 1_000_000
    while low < high:
        middle = (low + high + 1) // 2
        if decodes(middle):
            low = middle
        else:
            high = middle - 1
    return low


def _line_too_deep_at_the_door() -> str:
    """A verified stamp row that fails to decode at the door yet decodes at the top."""
    limit = _c_nested(_IMPORT_C_DEPTH, _decode_limit)
    # The outer object is one level, so ``limit`` list levels total limit + 1.
    head = json.dumps({"lop_mesh_origin": OWN})[:-1]
    line = head + ', "pad": ' + "[" * limit + "]" * limit + "}"

    def reader() -> Any:
        return json.loads(line)

    with pytest.raises(RecursionError):
        _c_nested(_IMPORT_C_DEPTH, reader)
    # The hazard itself: a shallower reader (this frame) decodes it as verified.
    assert reader()["lop_mesh_origin"]["verified"] is True
    return line


def test_r2_1_move_door_refuses_a_line_too_deep_at_its_depth(tmp_path: Path) -> None:
    root = tmp_path / "config"
    staged = _staged_move(root)
    line = _line_too_deep_at_the_door()
    with (staged / "transcript.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")

    outcome = _c_nested(
        _IMPORT_C_DEPTH,
        lambda: mobility._promote(cast(Any, SimpleNamespace(root=root)), staged, "moved0000001"),
    )
    assert outcome == mobility._PROMOTE_STAMPS_REFUSED
    assert not (root / "sessions" / "moved0000001").exists()
    _assert_no_verified_copy(root)


def test_r2_1_replica_door_refuses_a_line_too_deep_at_its_depth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, session_id = _replica(tmp_path)
    line = _line_too_deep_at_the_door()
    with (sync.replica_dir(root, session_id) / "transcript.jsonl").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(line + "\n")
    # A hostile owner's plan covers the line it wrote; the door is under test here.
    monkeypatch.setattr(sync, "_verify_replica_against_cursor", lambda *a, **k: None)

    def recover() -> None:
        sync.promote_replica(root, session_id, new_id="recovered0001")

    with pytest.raises(sync.SyncRefused, match="too deep"):
        _c_nested(_IMPORT_C_DEPTH, recover)
    assert not (root / "sessions" / "recovered0001").exists()
    _assert_no_verified_copy(root)
    assert (sync.network_dir(root) / sync.QUARANTINE_DIRNAME / "recovered0001").is_dir()


# ---------------------------------------------------------------------------
# R2-2: the refused copy leaves ``sessions/`` WHILE STILL CLAIMED, so nothing ever
# sees an unclaimed copy there; if it cannot leave, the claim stays.
# ---------------------------------------------------------------------------


def _refusing_replica(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str]:
    root, session_id = _replica(tmp_path)

    def boom(path: Path) -> int:
        raise origin.ImportStampsRefused("injected")

    monkeypatch.setattr(origin, "_downgrade_file", boom)
    return root, session_id


def test_r2_2_quarantine_moves_the_copy_before_releasing_its_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    root, session_id = _refusing_replica(tmp_path, monkeypatch)
    real_rename = os.rename
    claimed_at_move: list[bool] = []

    def watching_rename(src: Any, dst: Any) -> None:
        if sync.QUARANTINE_DIRNAME in Path(dst).parts:
            claimed_at_move.append((Path(src) / ".session.pid").exists())
        real_rename(src, dst)

    monkeypatch.setattr(sync.os, "rename", watching_rename)
    with pytest.raises(sync.SyncRefused):
        sync.promote_replica(root, session_id, new_id="recovered0001")
    assert claimed_at_move == [True]
    quarantined = sync.network_dir(root) / sync.QUARANTINE_DIRNAME / "recovered0001"
    assert quarantined.is_dir()
    assert not (quarantined / ".session.pid").exists()  # released once it was out
    _assert_no_verified_copy(root)


def test_r2_2_a_failed_quarantine_keeps_the_claim_and_still_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    root, session_id = _refusing_replica(tmp_path, monkeypatch)
    real_rename = os.rename

    def failing_rename(src: Any, dst: Any) -> None:
        if sync.QUARANTINE_DIRNAME in Path(dst).parts:
            raise OSError("injected: quarantine unavailable")
        real_rename(src, dst)

    monkeypatch.setattr(sync.os, "rename", failing_rename)
    with pytest.raises(sync.SyncRefused):
        sync.promote_replica(root, session_id, new_id="recovered0001")
    # Still under sessions/, so it stays claimed: listings and sweeps skip it.
    assert (root / "sessions" / "recovered0001" / ".session.pid").exists()


# The guarded quarantine (cleanup.quarantine_claimed_session_dir) that the
# refused-recovery path routes through: it moves only a directory THIS process
# still claims, never deletes, and refuses everything else in place.


def _claimed_dir(root: Path, name: str, pid: str) -> Path:
    target = root / "sessions" / name
    target.mkdir(parents=True)
    (target / ".session.pid").write_text(pid, encoding="utf-8")
    (target / "transcript.jsonl").write_text("{}\n", encoding="utf-8")
    return target


def test_quarantine_moves_only_a_dir_this_process_claims(tmp_path: Path) -> None:
    import os

    from local_operator.session import cleanup

    holding = tmp_path / "network" / "quarantine"
    mine = _claimed_dir(tmp_path, "mine0001", str(os.getpid()))
    landing = cleanup.quarantine_claimed_session_dir(
        mine, config_dir=tmp_path, holding_dir=holding, reason="t", actor="t"
    )
    assert landing == holding / "mine0001"
    assert not mine.exists()
    assert (landing / "transcript.jsonl").read_text(encoding="utf-8") == "{}\n"
    assert not (landing / ".session.pid").exists()


@pytest.mark.parametrize("claim", ["1", "", None])
def test_quarantine_refuses_a_dir_another_process_or_nobody_claims(
    tmp_path: Path, claim: str | None
) -> None:
    from local_operator.session import cleanup

    target = _claimed_dir(tmp_path, "other0001", claim or "")
    if claim is None:
        (target / ".session.pid").unlink()
    landing = cleanup.quarantine_claimed_session_dir(
        target,
        config_dir=tmp_path,
        holding_dir=tmp_path / "network" / "quarantine",
        reason="t",
        actor="t",
    )
    assert landing is None
    assert (target / "transcript.jsonl").is_file()  # left exactly where it was


def test_quarantine_refuses_a_holding_dir_inside_sessions_or_a_foreign_store(
    tmp_path: Path,
) -> None:
    import os

    from local_operator.session import cleanup

    target = _claimed_dir(tmp_path, "mine0002", str(os.getpid()))
    for holding in (tmp_path / "sessions" / "q", tmp_path.parent / "elsewhere-q"):
        assert (
            cleanup.quarantine_claimed_session_dir(
                target, config_dir=tmp_path, holding_dir=holding, reason="t", actor="t"
            )
            is None
        )
    other_store = tmp_path / "other"
    (other_store / "sessions").mkdir(parents=True)
    assert (
        cleanup.quarantine_claimed_session_dir(
            target,
            config_dir=other_store,
            holding_dir=other_store / "network" / "quarantine",
            reason="t",
            actor="t",
        )
        is None
    )
    assert (target / ".session.pid").is_file()  # still claimed, still in place


# ---------------------------------------------------------------------------
# Q-1: a downgrade refusal is not an id collision; the reply says which it was.
# ---------------------------------------------------------------------------


def test_q1_refusal_reply_names_the_real_cause() -> None:
    stamps = mobility._promote_refusal_detail("moved0000001", mobility._PROMOTE_STAMPS_REFUSED)
    assert "origin stamps could not be lowered" in stamps
    assert "already exists" not in stamps
    collision = mobility._promote_refusal_detail("moved0000001", mobility._PROMOTE_REFUSED)
    assert "already exists on this device" in collision


# ---------------------------------------------------------------------------
# Review R3-1: a stamped line the strict parser rejects (a stray ``\r``, one
# invalid UTF-8 byte) refuses the import at BOTH doors; before, the door skipped
# it and ``asks.store.read_events`` read ``verified: true`` from the copy.
# ---------------------------------------------------------------------------

_ANSWERED = json.dumps(
    {"kind": "answered", "ask_id": "a-1", "by": {"surface": "remote", "mesh_origin": OWN}}
).encode()
_R3_1_LINES = {
    "carriage_return": b"garbage\r" + _ANSWERED,
    "invalid_utf8": _ANSWERED.replace(b'"remote"', b'"remote\xff"'),
}
#: An unstamped undecodable line: must stay byte-identical wherever it lands.
_CONTROL = b"junk\r\xff{torn"


def _no_verified_answer(directory: Path) -> None:
    from local_operator.asks import store as ask_store

    for event in ask_store.read_events(directory):
        assert event["by"]["mesh_origin"]["verified"] is not True


@pytest.mark.parametrize("case", sorted(_R3_1_LINES))
def test_r3_1_move_door_refuses_a_stamped_line_it_cannot_parse(tmp_path: Path, case: str) -> None:
    root = tmp_path / "config"
    staged = _staged_move(root)
    body = _CONTROL + b"\n" + _R3_1_LINES[case] + b"\n"
    (staged / "asks.jsonl").write_bytes(body)
    outcome = mobility._promote(cast(Any, SimpleNamespace(root=root)), staged, "moved0000001")
    assert outcome == mobility._PROMOTE_STAMPS_REFUSED
    assert not (root / "sessions" / "moved0000001").exists()
    _assert_no_verified_copy(root)
    # Refused before any rewrite of this file; the refused copy stays in staging.
    assert (staged / "asks.jsonl").read_bytes() == body
    _no_verified_answer(staged)


@pytest.mark.parametrize("case", sorted(_R3_1_LINES))
def test_r3_1_replica_door_refuses_a_stamped_line_it_cannot_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    root, session_id = _replica(tmp_path)
    (sync.replica_dir(root, session_id) / "asks.jsonl").write_bytes(
        _CONTROL + b"\n" + _R3_1_LINES[case] + b"\n"
    )
    # A hostile owner's plan covers the bytes it wrote; the door is under test.
    monkeypatch.setattr(sync, "_verify_replica_against_cursor", lambda *a, **k: None)
    with pytest.raises(sync.SyncRefused):
        sync.promote_replica(root, session_id, new_id="recovered0001")
    assert not (root / "sessions" / "recovered0001").exists()
    _assert_no_verified_copy(root)
    # R2-2: quarantined outside ``sessions/`` and unclaimed once there.
    quarantined = sync.network_dir(root) / sync.QUARANTINE_DIRNAME / "recovered0001"
    assert quarantined.is_dir()
    assert not (quarantined / ".session.pid").exists()
    _no_verified_answer(quarantined)


def test_r3_1_an_unstamped_undecodable_line_lands_byte_identical(tmp_path: Path) -> None:
    root = tmp_path / "config"
    staged = _staged_move(root)
    with (staged / "asks.jsonl").open("ab") as handle:
        handle.write(_CONTROL + b"\n")
    outcome = mobility._promote(cast(Any, SimpleNamespace(root=root)), staged, "moved0000001")
    assert outcome.ok is True, outcome
    landed = (root / "sessions" / "moved0000001" / "asks.jsonl").read_bytes().split(b"\n")
    assert _CONTROL in landed
    _assert_no_verified_copy(root)
