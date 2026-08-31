"""The shared send-side core (``mobile/peer_send.py``).

The CLI and the in-session ``send`` tool both resolve targets and validate
bodies through this module; these tests pin the shared decisions with fake
records (no socket), so a drift in resolution priority or body validation is
caught once for both callers.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from local_operator.mobile import peer_send, registry


class _Record:
    """The minimal SessionRecord shape the resolver and identity read."""

    def __init__(
        self,
        pid: int,
        *,
        session_id: str = "s1",
        conversation_name: str = "peer",
        model_label: str = "test/model",
        cwd: str = "/tmp",
    ) -> None:
        self.pid = pid
        self.session_id = session_id
        self.conversation_name = conversation_name
        self.model_label = model_label
        self.cwd = cwd
        self.control_port = 1
        self.control_key = "k"


def _scan(records: "list[tuple[Any, str]]"):
    def scan(root=None):
        return records

    return scan


@pytest.fixture
def fake_scan(monkeypatch):
    def install(records):
        monkeypatch.setattr(peer_send.registry, "scan", _scan(records))

    return install


def test_pid_wins_over_everything(fake_scan) -> None:
    a = _Record(10, conversation_name="alpha")
    b = _Record(20, conversation_name="beta")
    fake_scan([(a, "live"), (b, "live")])
    record, candidates, error = peer_send.resolve_peer_target(pid=20, target="alpha")
    assert record is b
    assert candidates == []
    assert error == ""


def test_session_id_matches_exactly(fake_scan) -> None:
    a = _Record(10, session_id="exact-id")
    fake_scan([(a, "live")])
    record, _c, error = peer_send.resolve_peer_target(session="exact-id")
    assert record is a
    assert error == ""


def test_substring_matches_name_session_and_cwd(fake_scan) -> None:
    by_name = _Record(10, conversation_name="release cutter")
    by_cwd = _Record(20, conversation_name="other", cwd="/home/u/ingest")
    fake_scan([(by_name, "live"), (by_cwd, "live")])
    record, _c, _e = peer_send.resolve_peer_target(target="release")
    assert record is by_name
    record, _c, _e = peer_send.resolve_peer_target(target="ingest")
    assert record is by_cwd


def test_ambiguous_substring_returns_candidates(fake_scan) -> None:
    a = _Record(10, conversation_name="multi one")
    b = _Record(20, conversation_name="multi two")
    fake_scan([(a, "live"), (b, "live")])
    record, candidates, error = peer_send.resolve_peer_target(target="multi")
    assert record is None
    assert error == ""
    assert candidates == [a, b]
    lines = peer_send.candidate_lines(candidates, indent="  ", prefix="pid")
    assert lines == [
        "  pid 10  multi one  (test/model)",
        "  pid 20  multi two  (test/model)",
    ]


def test_only_live_records_are_eligible(fake_scan) -> None:
    wedged = _Record(10, conversation_name="slow")
    fake_scan([(wedged, "wedged")])
    record, _c, error = peer_send.resolve_peer_target(target="slow")
    assert record is None
    assert "not responding" in error


def test_no_target_is_a_clean_error(fake_scan) -> None:
    fake_scan([])
    _r, _c, error = peer_send.resolve_peer_target()
    assert "no target given" in error


def test_validate_body_rejects_empty_and_oversized() -> None:
    assert peer_send.validate_peer_body("   ") == "message is empty"
    big = "x" * (peer_send.PEER_MESSAGE_MAX_BYTES + 1)
    error = peer_send.validate_peer_body(big)
    assert error is not None
    assert "too large" in error
    assert peer_send.validate_peer_body("fine") is None


def test_sender_identity_copies_the_matching_record(fake_scan) -> None:
    rec = _Record(42, conversation_name="me", session_id="me-id")
    fake_scan([(rec, "live")])
    sender = peer_send.peer_sender_identity(42)
    assert sender["pid"] == 42
    assert sender["conversation_name"] == "me"
    assert sender["session_id"] == "me-id"
    assert sender["model_label"] == "test/model"


def test_sender_identity_falls_back_to_pid_alone(fake_scan) -> None:
    fake_scan([])
    sender = peer_send.peer_sender_identity(999)
    assert sender == {"pid": 999}


def test_registry_scan_sees_a_published_record(tmp_path) -> None:
    # Round-trip through the REAL registry so the core's scan contract holds.
    rec = registry.SessionRecord(
        pid=os.getpid(),
        kind="tui",
        session_id="rt",
        conversation_name="roundtrip",
        cwd="/tmp",
        model_label="test/model",
        control_port=1,
        control_key="k",
    )
    registry.publish(rec, root=tmp_path)
    found = registry.scan(root=tmp_path)
    assert any(r.pid == os.getpid() and state == "live" for r, state in found)
