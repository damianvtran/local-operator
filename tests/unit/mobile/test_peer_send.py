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


def test_identity_walks_up_to_a_grandparent_that_owns_the_record(monkeypatch) -> None:
    """`lop send` is not always a direct child of the TUI.

    Run from a subagent's bash tool, through a shell wrapper, or under nohup,
    the session is a grandparent or higher — testing only the immediate parent
    missed it and the card rendered `peer message from (pid 1)`.
    """
    session_rec = _Record(500, conversation_name="owning session", session_id="own-id")
    monkeypatch.setattr(peer_send.registry, "scan", _scan([(session_rec, "live")]))
    # 100 (lop send) -> 200 (shell wrapper) -> 500 (the session that owns a record)
    tree = {100: 200, 200: 500, 500: 1}
    monkeypatch.setattr(peer_send, "_parent_pid", lambda pid: tree.get(pid))

    sender = peer_send.peer_sender_identity(100)
    # The pid reported is the SESSION's, not the transient shell's: the card has
    # to name a session the reader can go and talk to.
    assert sender["pid"] == 500
    assert sender["conversation_name"] == "owning session"
    assert sender["session_id"] == "own-id"


def test_identity_degrades_gracefully_when_no_ancestor_owns_a_record(monkeypatch) -> None:
    """The reparented case (ppid 1, nothing published): still deliverable, just
    less labelled — identity is advisory and must never block a send."""
    monkeypatch.setattr(peer_send.registry, "scan", _scan([]))
    monkeypatch.setattr(peer_send, "_parent_pid", lambda pid: 1 if pid != 1 else None)
    assert peer_send.peer_sender_identity(4242) == {"pid": 4242}


def test_the_ancestry_walk_is_bounded(monkeypatch) -> None:
    """A pathological tree must not turn identity lookup into a long walk."""
    monkeypatch.setattr(peer_send.registry, "scan", _scan([]))
    seen: list[int] = []

    def parent(pid: int) -> int:
        seen.append(pid)
        return pid + 1  # an infinite chain that never reaches a record

    monkeypatch.setattr(peer_send, "_parent_pid", parent)
    assert peer_send.peer_sender_identity(10) == {"pid": 10}
    assert len(seen) <= peer_send._ANCESTRY_MAX_HOPS


def test_a_parent_lookup_failure_ends_the_walk_without_raising(monkeypatch) -> None:
    monkeypatch.setattr(peer_send.registry, "scan", _scan([]))
    monkeypatch.setattr(peer_send, "_parent_pid", lambda pid: None)
    assert peer_send.peer_sender_identity(77) == {"pid": 77}


def test_receiver_resolves_a_pid_only_sender_from_the_registry(monkeypatch) -> None:
    """OP2: the receive side must not depend on the sender's self-report.

    A sender whose ancestry walk found nothing arrives as ``{"pid": N}``; the
    local registry is the authoritative answer to "who is pid N"."""
    rec = _Record(321, conversation_name="release cutter", session_id="rc-id")
    monkeypatch.setattr(peer_send.registry, "scan", _scan([(rec, "live")]))
    resolved = peer_send.resolve_sender_identity({"pid": 321})
    assert resolved["conversation_name"] == "release cutter"
    assert resolved["model_label"] == "test/model"
    assert resolved["session_id"] == "rc-id"


def test_receiver_keeps_what_the_sender_actually_supplied(monkeypatch) -> None:
    """A session that renamed itself mid-flight is right about its own name, so
    only ABSENT or blank fields are filled in."""
    rec = _Record(321, conversation_name="stale name")
    monkeypatch.setattr(peer_send.registry, "scan", _scan([(rec, "live")]))
    resolved = peer_send.resolve_sender_identity(
        {"pid": 321, "conversation_name": "fresh name", "model_label": ""}
    )
    assert resolved["conversation_name"] == "fresh name"
    # The blank one is still filled from the record.
    assert resolved["model_label"] == "test/model"


def test_receiver_enrichment_never_raises_on_a_junk_sender(monkeypatch) -> None:
    monkeypatch.setattr(peer_send.registry, "scan", _scan([]))
    assert peer_send.resolve_sender_identity(None) == {}
    assert peer_send.resolve_sender_identity({}) == {}
    # A non-int pid cannot be looked up and must pass through untouched.
    assert peer_send.resolve_sender_identity({"pid": "nope"}) == {"pid": "nope"}


def test_the_core_stays_import_light() -> None:
    """NIT-1: the module docstring promises it never pulls the heavyweight
    Session graph, and that promise is what keeps it importable from a tool.
    A comment cannot enforce it; this does."""
    import ast
    from pathlib import Path

    source = Path(peer_send.__file__).read_text()
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any(name.startswith("local_operator.session") for name in imported), imported
    assert not any(name.startswith("local_operator.tui") for name in imported), imported


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
