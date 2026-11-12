"""``net_session_peek`` / ``peer_session_peek`` at the handler level.

The parity proof (remote text == the owner's local peek) lives beside the
local peek tests in ``tests/unit/tools/test_sessions_tool.py``, and the real
two-relay run in ``test_sessions_pilot_cli.py``; this file pins the op's
contract edges: authority, wire typing, the old-peer answer and the regex caps.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.network import peek as net_peek
from local_operator.network import relay, types, wire
from local_operator.network.types import MeshRefusal

SESSION = "0123456789ab"


def _journal(root: Path, rows: int = 4) -> Path:
    directory = root / "sessions" / SESSION
    directory.mkdir(parents=True)
    lines = [
        json.dumps(
            {
                "id": f"{i:032x}",
                "ts": i,
                "type": "message",
                "payload": {
                    "kind": "message",
                    "role": "user",
                    "content": [{"text": f"row {i}"}],
                },
            }
        )
        for i in range(1, rows + 1)
    ]
    (directory / "transcript.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return directory


def test_the_op_is_registered_with_view_authority_and_advertised() -> None:
    """``view`` like ``net_session_history``; a slice op with a local half; the
    capability string is in the handshake list a viewer's relay checks."""
    assert types.OP_CAPABILITY["net_session_peek"] == "view"
    assert "net_session_peek" in types.NET_OPS
    assert "peer_session_peek" in types.LOCAL_OPS
    assert "peer_session_peek" not in types.OP_CAPABILITY
    assert "net_session_peek" in relay.SLICE_PEER_OPS
    assert "peer_session_peek" in relay.SLICE_LOCAL_OPS
    assert "local_operator.network.peek" in relay.SLICE_MODULES
    assert wire.SESSION_PEEK_V1 in wire.LINK_CAPABILITIES


def test_the_owner_serves_a_stored_session_from_its_journal(tmp_path: Path) -> None:
    """No runtime anywhere: the owner reads its disk and answers the window."""
    _journal(tmp_path)
    answer = net_peek.serve(tmp_path, {"session_id": SESSION, "head": 2}, own_label="b")
    assert answer["mode"] == "head" and answer["session_id"] == SESSION
    assert [step["body"] for step in answer["window"]["steps"]] == ["row 1", "row 2"]
    assert answer["request"] == {"head": 2}


@pytest.mark.parametrize(
    ("frame", "code", "words"),
    [
        ({}, "protocol_error", "must name a session_id"),
        ({"session_id": "ffffffffffff"}, "unknown_session", "does not hold a session"),
        ({"session_id": SESSION, "head": "2"}, "invalid_request", "whole number"),
        ({"session_id": SESSION, "head": True}, "invalid_request", "whole number"),
        ({"session_id": SESSION, "query": 7}, "invalid_request", "must be text"),
        ({"session_id": SESSION, "steps": 51}, "invalid_request", "too many for one peek"),
        ({"session_id": SESSION, "steps": 2, "head": 2}, "invalid_request", "one window"),
        ({"session_id": SESSION, "before_id": "nope"}, "invalid_request", "no transcript entry"),
    ],
)
def test_the_owner_refuses_bad_frames_by_name(
    tmp_path: Path, frame: dict[str, Any], code: str, words: str
) -> None:
    """A granted peer is still untrusted input: wrong types and over-budget
    windows are named refusals, never a TypeError on the link thread."""
    _journal(tmp_path)
    with pytest.raises(MeshRefusal) as caught:
        net_peek.serve(tmp_path, frame, own_label="b")
    assert caught.value.code == code and words in caught.value.sentence


def test_a_session_with_no_transcript_is_a_named_refusal(tmp_path: Path) -> None:
    (tmp_path / "sessions" / SESSION).mkdir(parents=True)
    with pytest.raises(MeshRefusal) as caught:
        net_peek.serve(tmp_path, {"session_id": SESSION}, own_label="b")
    assert caught.value.code == "no_transcript"


def _fake_server(capabilities: frozenset[str], calls: list[Any]) -> Any:
    def _call(op: str, peer: str, *, timeout: float, **fields: Any) -> dict[str, Any]:
        calls.append((op, peer, timeout, fields))
        return {"mode": "tail", "window": {"steps": []}}

    return SimpleNamespace(
        _resolve_peer=lambda peer: peer,
        _ensure_link=lambda peer: SimpleNamespace(capabilities=capabilities),
        _peer_label=lambda peer: f"device {peer}",
        _local_peer_call=_call,
    )


def test_an_old_peer_is_answered_unsupported_with_the_reason_and_never_asked() -> None:
    """Acceptance: a peer without the capability is NOT sent the op; the
    answer names why so the caller degrades to the tail and says so."""
    calls: list[Any] = []
    handler = net_peek.local_handler(_fake_server(frozenset({"mesh-net-v1"}), calls))
    answer = handler({"peer": "b", "session_id": SESSION, "head": 2})
    assert answer["unsupported"] is True
    assert "device b" in answer["reason"] and wire.SESSION_PEEK_V1 in answer["reason"]
    assert calls == []


def test_a_current_peer_is_asked_with_only_the_named_fields() -> None:
    calls: list[Any] = []
    handler = net_peek.local_handler(_fake_server(frozenset({wire.SESSION_PEEK_V1}), calls))
    handler({"peer": "b", "session_id": SESSION, "query": "x", "regex": True})
    ((op, peer, timeout, fields),) = calls
    assert op == "net_session_peek" and peer == "b"
    assert fields == {"session_id": SESSION, "query": "x", "regex": True}
    # The requester outwaits the owner's deadline by the slow-reply margin.
    assert timeout == net_peek.SESSION_PEEK_DEADLINE_S + relay.SLOW_REPLY_MARGIN_S


@pytest.mark.parametrize(
    ("query", "words"),
    [
        ("a" * 257, "capped at 256"),
        ("(a+)+$", "nest quantifiers"),
        ("(\\w*)*x", "nest quantifiers"),
        ("(ab){2,3}", ""),  # bounded repetition of a plain group: allowed
        ("row \\d+", ""),
    ],
)
def test_the_owner_bounds_a_peer_regex(tmp_path: Path, query: str, words: str) -> None:
    """A peer-supplied regex runs on the owner's CPU and ``re`` cannot be
    interrupted, so length and the catastrophic shape are refused there."""
    _journal(tmp_path)
    frame = {"session_id": SESSION, "query": query, "regex": True}
    if words:
        with pytest.raises(MeshRefusal, match=words):
            net_peek.serve(tmp_path, frame, own_label="b")
    else:
        net_peek.serve(tmp_path, frame, own_label="b")
