"""A create whose reply is lost must never become two sessions, nor "unreachable".

WHAT THIS PINS (2026-11-11, live): ``sessions op=spawn peer=lop-remote-node-2``
answered "stopped answering before it replied" while the node HAD created
59b1f39ac5ee — and a caller that believed it and retried would have made a second
session running the same first prompt. See ``network/create_keys.py``.

THE RIG is two in-process relays: an OWNER whose real ``_op_session_create`` /
``_op_session_create_lookup`` run against its own disk, and a REQUESTER whose real
``_ctl_peer_create`` → ``_local_peer_call`` path talks to the owner through a
fake link. The link can DROP a reply after the owner has acted, which is the
production failure exactly (the frame arrived, the answer did not). The warm-up
is stubbed: a spawned runtime belongs to the e2e cell, not here.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.network import definitions, relay, types
from local_operator.network.create_keys import CreateKeyIndex

REQUESTER_DEVICE = "d_" + "b" * 32
NETWORK = "n_0123456789abcdef0123456789abcdef"


def _server(root: Path) -> relay.RelayServer:
    return relay.RelayServer(
        root=root, settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1")
    )


def _owner_link(device_id: str = REQUESTER_DEVICE) -> Any:
    """What the owner's create op reads off the link the requester came in on."""
    return SimpleNamespace(
        device_id=device_id,
        network_id=NETWORK,
        epoch=1,
        context=SimpleNamespace(
            capabilities=frozenset({"prompt"}),
            device_id=device_id,
            network_id=NETWORK,
            epoch=1,
        ),
    )


class _Wire:
    """The requester's link to the owner: dispatches into the owner's real ops.

    ``drop_create_replies`` replies to that many ``net_session_create`` frames
    with ``None`` AFTER the owner ran them — the lost-reply case. ``lookup`` set
    to ``"unknown_op"`` plays an owner older than this build.
    """

    def __init__(self, owner: relay.RelayServer, *, drop_create_replies: int = 0) -> None:
        self.owner = owner
        self.drop = drop_create_replies
        self.lookup = "ok"
        self.sent: list[dict[str, Any]] = []

    def request(self, frame: dict[str, Any], timeout: float | None = None) -> dict[str, Any] | None:
        self.sent.append(dict(frame))
        op = frame["op"]
        if op == "net_session_create_lookup" and self.lookup == "unknown_op":
            return {"op": "error", "code": "unknown_op", "message": "not an operation"}
        # A create MUST reconcile definitions first, even with no named profile.
        # Exercise that real preflight rather than letting the fake raise KeyError
        # and the production caller report state_failed before the create proceeds.
        handler = {
            "net_definitions": definitions.make_handler(self.owner),
            "net_session_create": self.owner._op_session_create,  # noqa: SLF001
            "net_session_create_lookup": self.owner._op_session_create_lookup,  # noqa: SLF001
        }[op]
        try:
            detail = handler(_owner_link(), frame)
        except types.MeshRefusal as refusal:
            return {"op": "error", "code": refusal.code, "message": refusal.sentence}
        if op == "net_session_create" and self.drop > 0:
            self.drop -= 1
            return None
        return {"op": "ack" if op == "net_definitions" else "reply", "detail": detail}


@pytest.fixture
def pair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any, _Wire]:
    monkeypatch.setattr(relay.RelayServer, "_warm_after_create", lambda self, *a, **k: None)
    owner = _server(tmp_path / "owner")
    requester = _server(tmp_path / "requester")
    wire = _Wire(owner)
    monkeypatch.setattr(requester, "_resolve_peer", lambda peer: peer)
    monkeypatch.setattr(requester, "_ensure_link", lambda _peer: wire)
    return owner, requester, wire


def _sessions(owner: relay.RelayServer) -> list[str]:
    directory = owner.root / "sessions"
    return sorted(p.name for p in directory.iterdir()) if directory.is_dir() else []


def test_a_dropped_create_reply_reports_the_id_and_a_retry_returns_the_same_one(
    pair: tuple[Any, Any, _Wire],
) -> None:
    """The acceptance cell: drop the reply, retry with the key, ONE session."""
    owner, requester, wire = pair
    wire.drop = 1
    first = requester._ctl_peer_create({"peer": "node-2"})  # noqa: SLF001
    # Settled by the lookup, not reported as unreachable.
    assert first["create_unconfirmed"] is True, first
    minted = first["session_id"]
    assert _sessions(owner) == [minted]
    assert "lost" in first["detail"] and minted in first["detail"], first
    key = first["create_key"]
    # The mandatory preflight predates this cell (24b6de4c86). The original
    # assertion in 6ccc448f63 omitted it; 18318597a only edits later cells/copy.
    assert [f["op"] for f in wire.sent] == [
        "net_definitions",
        "net_session_create",
        "net_session_create_lookup",
    ]
    assert wire.sent[0]["phase"] == "state"
    create_frames = [f for f in wire.sent if f["op"].startswith("net_session_create")]
    assert [f["create_key"] for f in create_frames] == [key, key]

    retry = requester._ctl_peer_create({"peer": "node-2", "create_key": key})  # noqa: SLF001
    assert retry["session_id"] == minted, retry
    assert retry["replayed"] is True, retry
    assert _sessions(owner) == [minted], "a retry with the same key made a second session"


def test_without_a_key_two_creates_are_two_sessions(
    pair: tuple[Any, Any, _Wire],
) -> None:
    """Control: the dedup is the key's doing — distinct requests stay distinct."""
    owner, requester, _wire = pair
    one = requester._ctl_peer_create({"peer": "node-2"})  # noqa: SLF001
    two = requester._ctl_peer_create({"peer": "node-2"})  # noqa: SLF001
    assert one["session_id"] != two["session_id"]
    assert len(_sessions(owner)) == 2


def test_an_old_owner_that_cannot_be_asked_says_may_have_been_created(
    pair: tuple[Any, Any, _Wire],
) -> None:
    """No lookup answer is NOT "not created": the sentence says to check first."""
    _owner, requester, wire = pair
    wire.drop = 1
    wire.lookup = "unknown_op"
    with pytest.raises(types.MeshRefusal) as refused:
        requester._ctl_peer_create({"peer": "node-2"})  # noqa: SLF001
    assert refused.value.code == "create_unconfirmed", refused.value
    sentence = refused.value.sentence
    assert "may have been created" in sentence, sentence
    # Both real spellings (R1-n1 / QA Q2): the tool's and the CLI's.
    assert "`sessions op=list peer=node-2`" in sentence, sentence
    assert "`lop network sessions --peer node-2`" in sentence, sentence
    assert "unreachable" not in sentence


def test_a_pending_lookup_names_the_full_retry_command(
    pair: tuple[Any, Any, _Wire], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1-n1: the retry sentence is a command an agent can run as written."""
    owner, requester, wire = pair
    wire.drop = 1
    monkeypatch.setattr(
        owner._create_keys, "lookup", lambda _device, _key: ("pending", "")  # noqa: SLF001
    )
    with pytest.raises(types.MeshRefusal) as refused:
        requester._ctl_peer_create({"peer": "node-2", "create_key": "a" * 32})  # noqa: SLF001
    sentence = refused.value.sentence
    assert f"`lop network sessions --peer node-2 --create --create-key {'a' * 32}`" in sentence


def test_images_without_a_prompt_are_refused_before_the_key_is_reserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1-m4: the refusal depends on the frame alone, so it reserves (and binds)
    nothing — a keyed retry must not be answered ``replayed`` for a refused create."""
    monkeypatch.setattr(relay.RelayServer, "_warm_after_create", lambda self, *a, **k: None)
    owner = _server(tmp_path / "owner")
    key = "9" * 32
    frame = {
        "op": "net_session_create",
        "req": 1,
        "cwd": "",
        "create_key": key,
        "images": [{"data_b64": "AA==", "mime_type": "image/png"}],
    }
    with pytest.raises(types.MeshRefusal) as refused:
        owner._op_session_create(_owner_link(), frame)  # noqa: SLF001
    assert refused.value.code == "protocol_error"
    assert owner._create_keys.lookup(REQUESTER_DEVICE, key) == ("unknown", "")  # noqa: SLF001
    assert _sessions(owner) == [], "nothing was minted for a refused frame"


def test_an_unreachable_link_is_still_unreachable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Never asked is still ``peer_unreachable``: nothing could have been created."""
    requester = _server(tmp_path / "requester")
    monkeypatch.setattr(requester, "_resolve_peer", lambda peer: peer)
    monkeypatch.setattr(requester, "_ensure_link", lambda _peer: None)
    with pytest.raises(types.MeshRefusal) as refused:
        requester._ctl_peer_create({"peer": "node-2"})  # noqa: SLF001
    assert refused.value.code == "peer_unreachable"


def test_keys_are_scoped_to_the_device_that_sent_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Another member's identical key neither replays nor reveals the session."""
    monkeypatch.setattr(relay.RelayServer, "_warm_after_create", lambda self, *a, **k: None)
    owner = _server(tmp_path / "owner")
    frame = {"op": "net_session_create", "req": 1, "cwd": "", "create_key": "k" * 32}
    mine = owner._op_session_create(_owner_link(), dict(frame))  # noqa: SLF001
    theirs = owner._op_session_create(_owner_link("d_" + "c" * 32), dict(frame))  # noqa: SLF001
    assert mine["session_id"] != theirs["session_id"]
    looked = owner._op_session_create_lookup(  # noqa: SLF001
        _owner_link("d_" + "e" * 32), {"create_key": "k" * 32}
    )
    assert looked["state"] == "unknown" and looked["session_id"] == ""


def test_a_refused_create_frees_its_key_and_a_bound_key_survives_a_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refused before the mkdir → key free again; bound → persisted for a new relay."""
    monkeypatch.setattr(relay.RelayServer, "_warm_after_create", lambda self, *a, **k: None)
    owner = _server(tmp_path / "owner")
    key = "f" * 32
    bad = {
        "op": "net_session_create",
        "req": 1,
        "cwd": str(tmp_path / "nope"),
        "create_key": key,
    }
    with pytest.raises(types.MeshRefusal):
        owner._op_session_create(_owner_link(), bad)  # noqa: SLF001
    assert owner._create_keys.lookup(REQUESTER_DEVICE, key) == ("unknown", "")  # noqa: SLF001
    made = owner._op_session_create(  # noqa: SLF001
        _owner_link(),
        {"op": "net_session_create", "req": 2, "cwd": "", "create_key": key},
    )
    reborn = CreateKeyIndex(owner.root)
    assert reborn.lookup(REQUESTER_DEVICE, key) == ("created", made["session_id"])
    assert reborn.reserve(REQUESTER_DEVICE, key) == ("existing", made["session_id"])


def test_a_concurrent_replay_while_pending_creates_nothing(tmp_path: Path) -> None:
    index = CreateKeyIndex(tmp_path)
    assert index.reserve("d", "k" * 16) == ("new", "")
    assert index.reserve("d", "k" * 16) == ("pending", "")
    assert index.lookup("d", "k" * 16) == ("pending", "")
