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
        started: bool = True,
    ) -> None:
        self.pid = pid
        self.session_id = session_id
        self.conversation_name = conversation_name
        self.model_label = model_label
        self.cwd = cwd
        self.control_port = 1
        self.control_key = "k"
        self.started = started


def _scan(records: "list[tuple[Any, str]]"):
    def scan(root=None):
        return records

    return scan


@pytest.fixture
def fake_scan(monkeypatch):
    def install(records):
        monkeypatch.setattr(peer_send.registry, "scan", _scan(records))

    return install


def test_pid_with_a_target_substring_is_refused_as_ambiguous(fake_scan) -> None:
    """A selector AND a substring name two different sessions, so the resolver
    refuses instead of applying precedence.

    This used to assert the opposite ("pid wins over everything"), which is the
    behaviour that made `lop send other-name "body" --pid N` deliver to pid N
    and report success while reading as though it addressed other-name. The
    precedence is fine when only one address is given; silently picking a winner
    between two is the wrong-recipient hazard. See docs/design/peer-send.md
    §4.3.
    """
    a = _Record(10, conversation_name="alpha")
    b = _Record(20, conversation_name="beta")
    fake_scan([(a, "live"), (b, "live")])
    record, candidates, error = peer_send.resolve_peer_target(pid=20, target="alpha")
    assert record is None
    assert candidates == []
    assert "not both" in error


def test_session_with_a_target_substring_is_refused_as_ambiguous(fake_scan) -> None:
    a = _Record(10, conversation_name="alpha", session_id="exact-id")
    fake_scan([(a, "live")])
    record, _c, error = peer_send.resolve_peer_target(session="exact-id", target="alpha")
    assert record is None
    assert "not both" in error


def test_pid_and_session_together_is_refused(fake_scan) -> None:
    """The CLI can never reach this (argparse's mutually-exclusive group rejects
    the pair at parse time), but the tool has no parser in front of it."""
    a = _Record(10, session_id="exact-id")
    fake_scan([(a, "live")])
    record, _c, error = peer_send.resolve_peer_target(pid=10, session="exact-id")
    assert record is None
    assert "not both" in error


def test_conflict_error_uses_the_callers_own_grammar(fake_scan) -> None:
    """The CLI's hints must survive into the pid+session wording, the same way
    they already do for "no target given"."""
    fake_scan([])
    _r, _c, error = peer_send.resolve_peer_target(
        pid=10, session="exact-id", pid_hint="--pid", session_hint="--session"
    )
    assert error == "pass either --pid or --session, not both"


def test_a_conflict_is_refused_before_the_registry_is_scanned(monkeypatch) -> None:
    """The refusal has to be free: nothing resolved, nothing read, nothing
    dialled on an ambiguously addressed call."""

    def _boom(*_a, **_k):
        raise AssertionError("registry must not be scanned on a conflicting address")

    monkeypatch.setattr(peer_send.registry, "scan", _boom)
    record, _c, error = peer_send.resolve_peer_target(pid=20, target="alpha")
    assert record is None
    assert "not both" in error


def test_pid_still_resolves_when_it_is_the_only_address(fake_scan) -> None:
    """The precedence path itself is untouched for a single-address call."""
    a = _Record(10, conversation_name="alpha")
    b = _Record(20, conversation_name="beta")
    fake_scan([(a, "live"), (b, "live")])
    record, candidates, error = peer_send.resolve_peer_target(pid=20)
    assert record is b
    assert candidates == []
    assert error == ""


def test_a_blank_target_beside_a_selector_is_not_a_conflict(fake_scan) -> None:
    """An empty/whitespace target is absence, not a competing address — the tool
    may pass target="" rather than omitting the field."""
    b = _Record(20, conversation_name="beta")
    fake_scan([(b, "live")])
    record, _c, error = peer_send.resolve_peer_target(pid=20, target="   ")
    assert record is b
    assert error == ""


def test_an_all_digit_target_resolves_as_a_pid(fake_scan) -> None:
    """The pid every listing shows (`lop sessions`, the picker, the
    disambiguation rows) resolves when typed back as a bare target — for
    `send` and `stop` alike (U2 / Q1-4)."""
    a = _Record(48213, conversation_name="alpha")
    b = _Record(20, conversation_name="beta", session_id="48213")
    fake_scan([(a, "live"), (b, "live")])
    record, candidates, error = peer_send.resolve_peer_target(target="48213")
    assert record is a and candidates == [] and error == ""
    # A wedged pid is reported as such through the pid path, not as "no match".
    fake_scan([(a, "wedged")])
    record, _c, error = peer_send.resolve_peer_target(target="48213")
    assert record is None and "wedged" in error
    record, _c, error = peer_send.resolve_peer_target(target="48213", include_wedged=True)
    assert record is a


def test_digits_that_are_not_a_pid_fall_through_to_the_substring_match(fake_scan) -> None:
    """A numeric session id or name still resolves when no record has that pid."""
    b = _Record(20, conversation_name="beta", session_id="777")
    fake_scan([(b, "live")])
    record, _c, error = peer_send.resolve_peer_target(target="777")
    assert record is b and error == ""


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


def test_a_broadcast_substring_skips_an_unstarted_session(fake_scan) -> None:
    """A ``/new`` session sitting in the composer (``started=False``) is
    invisible to a substring/broadcast match: an interrupt there would drive a
    turn into a session whose owner has not started it."""
    fresh = _Record(10, conversation_name="release", started=False)
    working = _Record(20, conversation_name="release cutter", started=True)
    fake_scan([(fresh, "live"), (working, "live")])
    record, candidates, error = peer_send.resolve_peer_target(target="release")
    assert record is working
    assert candidates == []
    assert error == ""


def test_a_broadcast_matching_only_unstarted_sessions_finds_nothing(fake_scan) -> None:
    fresh = _Record(10, conversation_name="release", started=False)
    fake_scan([(fresh, "live")])
    record, _c, error = peer_send.resolve_peer_target(target="release")
    assert record is None
    assert "no live session matches" in error


def test_an_exact_pid_send_still_resolves_an_unstarted_session(fake_scan) -> None:
    """The sender deliberately NAMED this session by pid, so it resolves — the
    delivery layer spools rather than drives a turn (see deliver_peer_message)."""
    fresh = _Record(10, conversation_name="release", started=False)
    fake_scan([(fresh, "live")])
    record, _c, error = peer_send.resolve_peer_target(pid=10)
    assert record is fresh
    assert error == ""


def test_an_exact_session_send_still_resolves_an_unstarted_session(fake_scan) -> None:
    fresh = _Record(10, session_id="fresh-id", started=False)
    fake_scan([(fresh, "live")])
    record, _c, error = peer_send.resolve_peer_target(session="fresh-id")
    assert record is fresh
    assert error == ""


def test_no_target_is_a_clean_error(fake_scan) -> None:
    fake_scan([])
    _r, _c, error = peer_send.resolve_peer_target()
    assert "no target given" in error


# --- Stored-session discovery (resolve_stored_target) ----------------------
#
# A note addressed by the name a session had before its terminal was closed
# must still deliver. These pin the fallback's contract: live wins, ambiguity
# refuses with candidates, and a single stored match resolves to the id that
# feeds the unchanged cold-delivery path. The store is faked by patching
# ``resume.recent_session_rows`` — the same scan the picker uses — so the test
# pins the matching rule, not the filesystem walk.


class _StoredRow:
    """The ``SessionRow`` fields ``resolve_stored_target`` reads."""

    def __init__(self, session_id: str, name: str, mtime: float = 0.0) -> None:
        self.id = session_id
        self.name = name
        self.mtime = mtime


def _stored(monkeypatch, rows):
    monkeypatch.setattr(
        "local_operator.resume.recent_session_rows",
        lambda directory, limit=None: rows,
    )


def test_stored_match_by_name_resolves_when_no_live_record(monkeypatch, fake_scan) -> None:
    fake_scan([])
    _stored(monkeypatch, [_StoredRow("abc123def456", "Improve /credential skill")])
    session_id, candidates, error = peer_send.resolve_stored_target("credential")
    assert error == ""
    assert candidates == []
    assert session_id == "abc123def456"


def test_stored_match_is_case_insensitive(monkeypatch, fake_scan) -> None:
    fake_scan([])
    _stored(monkeypatch, [_StoredRow("abc123def456", "Release Cutter")])
    session_id, _c, _e = peer_send.resolve_stored_target("release cutter")
    assert session_id == "abc123def456"


def test_live_ids_exclude_a_running_session_s_own_stored_row(monkeypatch, fake_scan) -> None:
    """The exclusion is by ID, not by name, and exists for the id collision.

    The fallback is entered only after the live scan matched nothing (see
    ``live_scan_found_nothing``), so a live session sharing the stored row's
    NAME cannot be what this guards: both read their name from the same
    transcript. What the exclusion exists for is a stored row whose ID belongs
    to a session that currently has a runtime — delivering cold to it would
    queue behind a process that could have been dialled. Review round 1,
    MINOR-3: an earlier draft of this test passed a live NAME and asserted the
    stored row still returned, which pinned nothing the production callers
    exercise.
    """
    live = _Record(10, session_id="shared-id", conversation_name="live name")
    fake_scan([(live, "live")])
    _stored(monkeypatch, [_StoredRow("shared-id", "old name"), _StoredRow("other-id", "old name")])
    session_id, candidates, error = peer_send.resolve_stored_target(
        "old name", live_ids={"shared-id"}
    )
    assert error == ""
    assert candidates == []
    # The live-owned row is skipped; the distinct stored row still resolves.
    assert session_id == "other-id"


def test_ambiguous_stored_matches_are_refused_with_candidates(monkeypatch, fake_scan) -> None:
    fake_scan([])
    _stored(
        monkeypatch,
        [_StoredRow("id1", "multi one"), _StoredRow("id2", "multi two")],
    )
    session_id, candidates, error = peer_send.resolve_stored_target("multi")
    assert session_id is None
    assert error == ""
    assert [c.session_id for c in candidates] == ["id1", "id2"]
    lines = peer_send.stored_candidate_lines(candidates, indent="  ", prefix="session")
    assert lines == [
        "  session id1  multi one  (not running)",
        "  session id2  multi two  (not running)",
    ]


def test_no_stored_match_is_a_clean_no_match(monkeypatch, fake_scan) -> None:
    fake_scan([])
    _stored(monkeypatch, [_StoredRow("id1", "something else")])
    session_id, candidates, error = peer_send.resolve_stored_target("nothing-here")
    assert (session_id, candidates, error) == (None, [], "")


def test_an_unreadable_store_never_refuses_a_send(monkeypatch, fake_scan) -> None:
    fake_scan([])

    def _boom(directory, limit=None):
        raise OSError("disk gone")

    monkeypatch.setattr("local_operator.resume.recent_session_rows", _boom)
    session_id, candidates, error = peer_send.resolve_stored_target("anything")
    assert (session_id, candidates, error) == (None, [], "")


def test_stored_resolution_flows_into_spool_delivery(monkeypatch, tmp_path) -> None:
    """A stored substring match delivers through the EXISTING cold path.

    ``resolve_stored_target`` decides WHO; ``deliver_peer_message`` still owns
    HOW. This pins that a resolved stored id spools to the inbox on a quiet
    mailbox (wake=False) exactly as an exact-id cold send does — the fallback
    does not rebuild delivery.
    """
    import asyncio

    sid = "deadbeef0123"
    directory = tmp_path / "sessions" / sid
    directory.mkdir(parents=True)
    monkeypatch.setattr(peer_send, "config_dir", lambda: tmp_path)

    receipt = asyncio.run(
        peer_send.deliver_peer_message(
            None,
            session_id=sid,
            text="hello from the past",
            mode="mailbox",
            wake=False,
            sender={},
        )
    )
    assert receipt == "spooled (will be read when the session next opens)"
    assert (directory / "inbox.jsonl").is_file()


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

    # ``local_operator.session.runtime.*`` is EXEMPT, and the exemption does not
    # weaken this guard. The registry this module imports is the same
    # stdlib-only record layer it always used — it merely moved from
    # ``mobile/registry.py`` into the session runtime package, whose ``types``
    # and ``registry`` modules are import-light by contract precisely because
    # they sit on the CLI startup path (see session/runtime/types.py and
    # tests/unit/test_import_graph.py). What this test actually exists to
    # forbid is the heavyweight Session graph, so that is now asserted by
    # name rather than inferred from a path prefix that stopped tracking it.
    heavy = {
        "local_operator.session.session",
        "local_operator.session_factory",
    }
    assert not (imported & heavy), imported
    assert not any(
        name.startswith("local_operator.session")
        and not name.startswith("local_operator.session.runtime")
        for name in imported
    ), imported
    assert not any(name.startswith("local_operator.tui") for name in imported), imported


def test_the_core_really_does_not_load_the_session_graph() -> None:
    """The static check above reads imports; this one measures what loading
    the module actually costs, in a FRESH interpreter.

    The AST check can only see this file's own import statements, so it would
    miss a heavyweight module pulled in transitively by something it imports —
    which is exactly the regression the docstring's promise is about. Run in a
    subprocess because pytest has already imported half the tree in-process.
    """
    import json
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    probe = (
        "import json, importlib, sys; "
        "importlib.import_module('local_operator.mobile.peer_send'); "
        "print(json.dumps(sorted(sys.modules)))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, cwd=str(repo)
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    modules = set(json.loads(proc.stdout.strip().splitlines()[-1]))

    for heavy in ("local_operator.session.session", "local_operator.session_factory"):
        assert heavy not in modules, f"{heavy} is back on the peer-send import path"
    # pydantic is the tell that the harness's model layer arrived; textual is
    # the TUI. A `send` tool import must pay for neither.
    for heavy in ("pydantic", "textual"):
        assert not any(m == heavy or m.startswith(heavy + ".") for m in modules), heavy


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


def test_enrichment_ignores_wedged_and_stale_records(monkeypatch) -> None:
    """Only LIVE records may name a sender (round 2, MINOR-4).

    ``scan`` also returns ``wedged`` (pid alive, heartbeat aged out) and
    ``stale`` (pid gone) entries. Enriching from those attributes a message to
    whichever session happens to hold a reused pid, and that attribution reaches
    the model-visible provenance envelope — so enrichment must be no laxer than
    ``resolve_peer_target``, which filters to live twenty lines up.
    """
    for state in ("wedged", "stale"):
        rec = _Record(4242, conversation_name="not really here")
        monkeypatch.setattr(peer_send.registry, "scan", _scan([(rec, state)]))
        resolved = peer_send.resolve_sender_identity({"pid": 4242})
        assert resolved == {"pid": 4242}, f"{state} record was used to label a sender"
        # The send-side walk must not accept it either.
        monkeypatch.setattr(peer_send, "_parent_pid", lambda pid: None)
        assert peer_send.peer_sender_identity(4242) == {"pid": 4242}

    live = _Record(4242, conversation_name="genuinely live")
    monkeypatch.setattr(peer_send.registry, "scan", _scan([(live, "live")]))
    assert peer_send.resolve_sender_identity({"pid": 4242})["conversation_name"] == (
        "genuinely live"
    )


def test_enrichment_never_raises_whatever_the_registry_does(monkeypatch) -> None:
    """The docstring's "never raises" has to be true, not aspirational.

    This runs on the receive path AHEAD of the transcript write, on a message
    the wire has already accepted, so an escaping exception DROPS a delivered
    message. Previously only OSError was caught and ValueError/RuntimeError
    propagated (round 2, MINOR-5).
    """
    for boom in (ValueError("torn record"), RuntimeError("wedged"), OSError("gone")):

        def scan(root=None, _exc=boom):
            raise _exc

        monkeypatch.setattr(peer_send.registry, "scan", scan)
        # Degrades to the unenriched dict rather than propagating.
        assert peer_send.resolve_sender_identity({"pid": 5}) == {"pid": 5}
        monkeypatch.setattr(peer_send, "_parent_pid", lambda pid: None)
        assert peer_send.peer_sender_identity(5) == {"pid": 5}

    # A record whose attributes explode is survivable too.
    class _Hostile:
        pid = 5

        def __getattr__(self, name):
            raise RuntimeError("hostile record")

    monkeypatch.setattr(peer_send.registry, "scan", _scan([(_Hostile(), "live")]))
    assert peer_send.resolve_sender_identity({"pid": 5}) == {"pid": 5}


@pytest.mark.asyncio
async def test_the_ancestry_walk_has_an_off_loop_entry_point(monkeypatch) -> None:
    """The walk runs a registry scan and a ``ps`` per hop, so callers inside a
    running loop must not do it inline (round 2, MINOR-7)."""
    rec = _Record(900, conversation_name="owning session")
    monkeypatch.setattr(peer_send.registry, "scan", _scan([(rec, "live")]))
    monkeypatch.setattr(peer_send, "_parent_pid", lambda pid: 900 if pid != 900 else 1)

    resolved = await peer_send.peer_sender_identity_async(100)
    assert resolved["conversation_name"] == "owning session"
    assert resolved["pid"] == 900


def test_a_non_dict_sender_cannot_escape_the_handler(monkeypatch) -> None:
    """The recovery path must not re-run the expression that threw.

    ``sender`` is whatever the wire's JSON decoded to, so it is not necessarily
    a dict. Building the fallback INSIDE the except arm meant ``dict(sender)``
    raised in the try and raised again in the handler, so the exception escaped
    to the receive path ahead of the transcript write and dropped a message the
    peer had already accepted (round 3, MINOR-8).
    """
    monkeypatch.setattr(peer_send.registry, "scan", _scan([]))
    # Deliberately ill-typed: the wire hands us whatever JSON decoded to, so the
    # runtime contract is wider than the annotation.
    hostile_inputs: "list[Any]" = [["not", "a", "dict"], "a string", 42, 3.5, object(), (1, 2, 3)]
    for hostile in hostile_inputs:
        assert peer_send.resolve_sender_identity(hostile) == {}, hostile

    class _HostileMapping(dict[str, Any]):
        """A mapping whose copy misbehaves — the handler is guarded for it too."""

        def keys(self):  # noqa: ANN201
            raise RuntimeError("hostile keys")

    assert peer_send.resolve_sender_identity(_HostileMapping()) == {}


# --- ``started`` gating on the DELIVERY path ---------------------------------
#
# ``resolve_peer_target`` excludes an unstarted session from a broadcast, but
# an exact-address send still resolves to it. The spool-below is the belt-and-
# braces that makes "no turn is ever driven into an unstarted session" hold even
# for a caller that bypassed resolution.


def _unstarted_record(session_id: str = "fresh-id") -> registry.SessionRecord:
    return registry.SessionRecord(
        pid=4242,
        kind="tui",
        session_id=session_id,
        conversation_name="fresh",
        cwd="/tmp",
        model_label="test/model",
        control_port=1,
        control_key="k",
        started=False,
    )


@pytest.mark.asyncio
async def test_an_unstarted_session_is_spooled_not_dialled(monkeypatch, tmp_path) -> None:
    """mailbox/wake/steer all degrade to a quiet spool for an unstarted session;
    the socket is never dialled, so no turn can be driven."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    record = _unstarted_record()
    dialed = False

    async def _dial(*_a: Any, **_k: Any) -> str:
        nonlocal dialed
        dialed = True
        return "delivered"

    monkeypatch.setattr("local_operator.mobile.peer_client.send_peer_message", _dial, raising=True)
    from local_operator.session.runtime.inbox import drain_inbox

    for mode, wake in (("mailbox", False), ("mailbox", True), ("steer", False), ("steer", True)):
        detail = await peer_send.deliver_peer_message(
            record,
            session_id=record.session_id,
            text=f"note-{mode}-{wake}",
            mode=mode,
            wake=wake,
            sender={"pid": 1},
        )
        assert detail == "spooled (session not started yet; will be read when it next opens)"
    assert not dialed, "an unstarted session must never be dialled"
    # Every spooled line is drained by the existing inbox path on first open.
    directory = tmp_path / "sessions" / record.session_id
    assert len(drain_inbox(directory)) == 4


@pytest.mark.asyncio
async def test_a_started_session_still_dials_the_socket(monkeypatch) -> None:
    """The new branch must not change the normal path: a started session is
    dialled and its ack detail is returned verbatim."""
    record = _unstarted_record()
    record.started = True

    async def _dial(rec: Any, *, text: str, mode: str, wake: bool, sender: Any) -> str:
        assert rec is record
        return "delivered and woke the session" if wake else "delivered to the mailbox"

    monkeypatch.setattr("local_operator.mobile.peer_client.send_peer_message", _dial, raising=True)
    detail = await peer_send.deliver_peer_message(
        record,
        session_id=record.session_id,
        text="hi",
        mode="mailbox",
        wake=True,
        sender={"pid": 1},
    )
    assert detail == "delivered and woke the session"


def test_a_record_without_the_started_key_reads_as_unstarted() -> None:
    """Mixed-version: a record an OLDER binary wrote has no ``started`` key, so
    the dataclass default ``False`` applies — the safe direction (excluded from
    broadcasts, spooled on an exact send) rather than risk driving a turn into a
    fresh ``/new`` session. See ``SessionRecord.from_json``."""
    record = registry.SessionRecord.from_json(
        {
            "pid": 4242,
            "kind": "tui",
            "session_id": "old-id",
            "conversation_name": "old",
            "cwd": "/tmp",
            "model_label": "test/model",
            "control_port": 1,
            "control_key": "k",
        }
    )
    assert record.started is False
