"""The shared send-side core (``mobile/peer_send.py``).

The CLI and the in-session ``send`` tool both resolve targets and validate
bodies through this module; these tests pin the shared decisions with fake
records (no socket), so a drift in resolution priority or body validation is
caught once for both callers.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.mobile import peer_send, registry
from local_operator.session.transcript import TRANSCRIPT_FILENAME


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
        #: Records written before the peer-message-id capability existed carry
        #: no string here, which is exactly what the retry gate reads.
        self.capabilities: list[str] = []
        # The resolver refuses a record that has stopped reporting, and it words
        # that refusal with the measured age, so the double carries the stamp
        # every real record has. Four minutes is past ``HEARTBEAT_TIMEOUT_S``
        # (45 s) — the state a plain send refuses.
        self.heartbeat_at = time.time() - 240


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
    assert record is None and "has not reported for 4m" in error
    record, _c, error = peer_send.resolve_peer_target(target="48213", include_wedged=True)
    assert record is a


def test_numeric_pid_target_uses_one_registry_scan(monkeypatch) -> None:
    """A PID typed as a bare target is resolved from its existing snapshot."""
    record = _Record(48213, conversation_name="alpha")
    scan_calls = 0

    def scan(root=None):
        nonlocal scan_calls
        scan_calls += 1
        return [(record, "live")]

    monkeypatch.setattr(peer_send.registry, "scan", scan)

    resolved, candidates, error = peer_send.resolve_peer_target(target="48213")

    assert resolved is record
    assert candidates == []
    assert error == ""
    assert scan_calls == 1


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
    # The measured age, and no promise about when the silence ends: the old
    # sentence said "not responding ... try again shortly", which invented both
    # a cause and a timetable (see ``_not_dialable``).
    assert "has not reported for 4m" in error, error
    assert "shortly" not in error, error


def test_a_broadcast_substring_skips_an_unstarted_session(fake_scan) -> None:
    """A ``/new`` session sitting in the composer (``started=False``) is
    invisible to a substring/broadcast match: an interrupt there would drive a
    turn into a session whose owner has not started it, and a quiet note there
    would become the opening row of that history."""
    fresh = _Record(10, conversation_name="release", started=False)
    working = _Record(20, conversation_name="release cutter", started=True)
    fake_scan([(fresh, "live"), (working, "live")])
    record, candidates, error = peer_send.resolve_peer_target(target="release")
    assert record is working
    assert candidates == []
    assert error == ""


def test_a_broadcast_matching_only_unstarted_sessions_is_refused(fake_scan) -> None:
    """THE NEW RULE: a substring/broadcast match that reached only a session
    nobody has typed in is REFUSED, and the refusal must not read as "nothing".

    The refused record is live and the scan DID reach it, so the error must not
    contain ``no live session matches``: that phrasing is what opens the stored
    fallback (``live_scan_found_nothing``), and a stored namesake would then be
    spooled to — a recipient this call never named (BLOCKER-1).
    """
    fresh = _Record(10, conversation_name="release", started=False)
    fake_scan([(fresh, "live")])
    record, _c, error = peer_send.resolve_peer_target(target="release")
    assert record is None
    assert "has not been engaged yet" in error, error
    assert "pid 10" in error, error
    assert peer_send.live_scan_found_nothing(error) is False, error
    assert "no live session matches" not in error, error


def test_an_exact_pid_send_refuses_an_unstarted_session(fake_scan) -> None:
    """THE NEW RULE: naming a session by pid does not make it a recipient.

    This used to resolve (the delivery layer degraded to a quiet dial). A peer
    row written into a composer window becomes the OPENING row of a
    conversation its owner never started, which is the reported symptom, so the
    refusal happens before delivery can consider it. The session becomes
    eligible when it runs its first turn — see ``require_started=False`` for the
    kill switch, the one caller that must still resolve it.
    """
    fresh = _Record(10, conversation_name="release", started=False)
    fake_scan([(fresh, "live")])
    record, _c, error = peer_send.resolve_peer_target(pid=10)
    assert record is None
    assert "pid 10 has not been engaged yet" in error, error


def test_an_exact_session_send_refuses_an_unstarted_session(fake_scan) -> None:
    """THE NEW RULE: same for an exact session id (and for an all-digit target
    tried as a pid, which routes through the same branch)."""
    fresh = _Record(10, session_id="fresh-id", started=False)
    fake_scan([(fresh, "live")])
    record, _c, error = peer_send.resolve_peer_target(session="fresh-id")
    assert record is None
    assert "session 'fresh-id' has not been engaged yet" in error, error

    # The bare all-digit spelling reaches the pid branch, so it is refused too.
    digits = _Record(20123, session_id="digit-id", started=False)
    fake_scan([(digits, "live")])
    record, _c, error = peer_send.resolve_peer_target(target="20123")
    assert record is None
    assert "pid 20123 has not been engaged yet" in error, error


def test_the_kill_switch_still_resolves_an_unstarted_session(fake_scan) -> None:
    """``require_started=False`` is the KILL-SWITCH carve-out: ``/stop <target>``
    and ``lop stop <target>`` must resolve a composer window in every address
    form, because a session someone needs to stop is not a delivery."""
    fresh = _Record(10, session_id="fresh-id", conversation_name="release", started=False)
    fake_scan([(fresh, "live")])
    addresses: list[dict[str, Any]] = [
        {"pid": 10},
        {"session": "fresh-id"},
        {"target": "release"},
    ]
    for kwargs in addresses:
        record, candidates, error = peer_send.resolve_peer_target(require_started=False, **kwargs)
        assert record is fresh, f"kwargs={kwargs} error={error}"
        assert candidates == []
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


def _write_transcript(root: Path, session_id: str, *, engaged: bool = True) -> Path:
    """Materialise one session's directory, with or without real history.

    Both new gates read the DISK rather than the fakes: ``resolve_stored_target``
    skips a stored row whose session has no durable history, and
    ``deliver_peer_message`` refuses a cold target on the same signal. So a
    stored row that is meant to resolve needs a transcript holding a real turn,
    and a row that is meant to be skipped needs a bare directory. Written as the
    raw JSONL shape the gate parses (``type: message`` + ``kind: message``)
    because going through ``Transcript`` would make the fixture's writer rather
    than the gate's reader the thing under test.
    """
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    if engaged:
        (directory / TRANSCRIPT_FILENAME).write_text(
            json.dumps(
                {
                    "id": "h1",
                    "ts": 1,
                    "type": "message",
                    "payload": {
                        "kind": "message",
                        "role": "user",
                        "content": [{"type": "text", "text": "hello"}],
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
    return directory


def _stored(monkeypatch, rows, *, root: Path, unengaged: "set[str] | None" = None):
    """Install the faked store scan AND the on-disk history it now depends on.

    ``unengaged`` names the rows that must stay bare — no transcript — which is
    the state the new skip exists for.
    """
    monkeypatch.setattr(
        "local_operator.resume.recent_session_rows",
        lambda directory, limit=None: rows,
    )
    monkeypatch.setattr(peer_send, "config_dir", lambda: root)
    for row in rows:
        _write_transcript(root, row.id, engaged=row.id not in (unengaged or set()))


def test_stored_match_by_name_resolves_when_no_live_record(
    monkeypatch, tmp_path, fake_scan
) -> None:
    fake_scan([])
    _stored(
        monkeypatch,
        [_StoredRow("abc123def456", "Improve /credential skill")],
        root=tmp_path,
    )
    session_id, candidates, error = peer_send.resolve_stored_target("credential")
    assert error == ""
    assert candidates == []
    assert session_id == "abc123def456"


def test_stored_match_is_case_insensitive(monkeypatch, tmp_path, fake_scan) -> None:
    fake_scan([])
    _stored(monkeypatch, [_StoredRow("abc123def456", "Release Cutter")], root=tmp_path)
    session_id, _c, _e = peer_send.resolve_stored_target("release cutter")
    assert session_id == "abc123def456"


def test_a_stored_row_with_no_durable_history_is_skipped(monkeypatch, tmp_path, fake_scan) -> None:
    """THE NEW RULE, stored half: a session that never ran a turn is not a
    recipient, so the fallback must not resolve onto one — a broadcast would
    otherwise spool a peer row into a conversation nobody started, and an exact
    cold send would be refused only one layer later.

    Both directions are pinned: the unengaged row is skipped while an engaged
    namesake still resolves, and a match on an unengaged row ALONE comes back as
    a REFUSAL naming it rather than as an empty no-match — the row DOES answer
    to the name, so "no session matches" would be a false statement about a
    session the user can see on the picker (review round 1, F-4).
    """
    fake_scan([])
    _stored(
        monkeypatch,
        [_StoredRow("dead0001", "new session"), _StoredRow("used0002", "new session")],
        root=tmp_path,
        unengaged={"dead0001"},
    )
    session_id, candidates, error = peer_send.resolve_stored_target("new session")
    assert (session_id, candidates, error) == ("used0002", [], "")

    _stored(
        monkeypatch,
        [_StoredRow("dead0001", "brand new")],
        root=tmp_path,
        unengaged={"dead0001"},
    )
    session_id, candidates, error = peer_send.resolve_stored_target("brand new")
    assert session_id is None
    assert candidates == []
    assert "the only stored match for 'brand new' (session 'dead0001')" in error, error
    assert "has not been engaged yet" in error, error

    # TWO withheld rows name the count instead of a single id, and neither form
    # may read as the no-match the callers compose for a store that answered
    # nothing.
    _stored(
        monkeypatch,
        [_StoredRow("dead0001", "brand new"), _StoredRow("dead0002", "brand new")],
        root=tmp_path,
        unengaged={"dead0001", "dead0002"},
    )
    session_id, candidates, error = peer_send.resolve_stored_target("brand new")
    assert (session_id, candidates) == (None, [])
    assert "2 stored matches for 'brand new' have not been engaged yet" in error, error
    # D3: the count is stated ONCE and the tail agrees with it — the form this
    # replaced (``every stored match for 'x' (2 of them) … so it cannot``) said
    # the count twice and then spoke about one session.
    assert "every stored match" not in error and "(2 of them)" not in error, error
    assert "no user message has been sent in any of them" in error, error
    assert "so they cannot receive peer messages" in error, error
    # D5: a stored row has no window anyone can type into, so the remedy is the
    # conversation being opened, not an owner sending into one that is already
    # there.
    assert "they become recipients once someone opens them" in error, error

    # A TRUE no-match is still the silent empty answer the callers compose their
    # own "searched live and stored sessions" sentence over.
    _stored(monkeypatch, [_StoredRow("other0001", "something else")], root=tmp_path)
    assert peer_send.resolve_stored_target("brand new") == (None, [], "")


def test_a_partly_delivered_broadcast_reports_the_matches_it_skipped(fake_scan) -> None:
    """D1 (design round 1): a name/substring send that REACHES a recipient may
    still have passed over matches held back for being unengaged. The sender
    typed one command believing it reached its needle, so the receipt has to
    say how many were left out — silence here is what made the GUIDE's "skips
    such a session and says so" a claim the product did not keep.

    ``skipped`` is the out-parameter the two surfaces read; the wording they
    append comes from ``skipped_clause`` so both say it the same way.
    """
    fake_scan(
        [
            (_Record(10, conversation_name="release build", started=False), "live"),
            (_Record(11, conversation_name="release notes", started=False), "live"),
            (_Record(20, conversation_name="release cutter", started=True), "live"),
        ]
    )
    skipped: list[Any] = []
    record, candidates, error = peer_send.resolve_peer_target(target="release", skipped=skipped)
    assert record is not None and record.pid == 20
    assert (candidates, error) == ([], "")
    assert [rec.pid for rec in skipped] == [10, 11]
    assert peer_send.skipped_clause(skipped) == "; 2 matches skipped (not engaged yet)"


def test_the_skipped_clause_is_silent_when_nothing_was_skipped() -> None:
    """The clause is the empty string for an empty list — that is what lets
    both receipts append it unconditionally — and it agrees with itself at one.
    """
    assert peer_send.skipped_clause([]) == ""
    assert peer_send.skipped_clause([_Record(10)]) == "; 1 match skipped (not engaged yet)"


def test_no_receipt_means_nothing_is_reported_as_skipped(fake_scan) -> None:
    """Only a DELIVERY prints a receipt, so only a delivery is handed the
    skipped matches: a refusal and an ambiguity resolve nothing, and a caller
    that appended a clause there would qualify a line that is already the
    error."""
    fake_scan([(_Record(10, conversation_name="release", started=False), "live")])
    skipped: list[Any] = []
    record, _c, error = peer_send.resolve_peer_target(target="release", skipped=skipped)
    assert record is None and "has not been engaged yet" in error
    assert skipped == []

    fake_scan(
        [
            (_Record(20, conversation_name="release a"), "live"),
            (_Record(21, conversation_name="release b"), "live"),
        ]
    )
    skipped = []
    record, candidates, _error = peer_send.resolve_peer_target(target="release", skipped=skipped)
    assert record is None and [c.pid for c in candidates] == [20, 21]
    assert skipped == []


def test_the_kill_switch_never_skips_a_match(fake_scan) -> None:
    """``require_started=False`` withholds nothing — the whole carve-out is
    that a composer window RESOLVES for ``/stop`` — so the kill switch's own
    callers keep working unchanged and never grow a receipt clause."""
    fresh = _Record(10, session_id="fresh-id", conversation_name="release", started=False)
    fake_scan([(fresh, "live")])
    skipped: list[Any] = []
    record, _c, error = peer_send.resolve_peer_target(
        target="release", require_started=False, skipped=skipped
    )
    assert record is fresh and error == ""
    assert skipped == []


def test_a_batched_refusal_agrees_with_its_own_count(fake_scan) -> None:
    """D3: a refusal about SEVERAL sessions has to read as a plural. The form
    this replaced said the count twice (``every … (3 of them)``) and then spoke
    about a single owner (``so it cannot … its owner``), which a reader can take
    as one owner's remedy for a batch."""
    fake_scan(
        [
            (_Record(10, conversation_name="release a", started=False), "live"),
            (_Record(11, conversation_name="release b", started=False), "live"),
            (_Record(12, conversation_name="release c", started=False), "live"),
        ]
    )
    record, _c, error = peer_send.resolve_peer_target(target="release")
    assert record is None
    assert error.startswith("3 live matches for 'release' have not been engaged yet"), error
    assert "no user message has been sent in any of them" in error, error
    assert "so they cannot receive peer messages" in error, error
    assert "their owners have to send a first message" in error, error
    assert "every live match" not in error and "(3 of them)" not in error, error


def test_a_single_match_refusal_still_reads_as_one_session(fake_scan) -> None:
    """The other half of D3: the singular forms are right and must stay — a
    pid-addressed refusal names the pid and talks about one owner."""
    fake_scan([(_Record(10, conversation_name="release", started=False), "live")])
    _r, _c, error = peer_send.resolve_peer_target(target="release")
    assert error.startswith("the only live match for 'release' (pid 10) has not been engaged yet")
    assert "no user message has been sent in it" in error, error
    assert "its owner has to send a first message" in error, error


def test_live_ids_exclude_a_running_session_s_own_stored_row(
    monkeypatch, tmp_path, fake_scan
) -> None:
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
    _stored(
        monkeypatch,
        [_StoredRow("shared-id", "old name"), _StoredRow("other-id", "old name")],
        root=tmp_path,
    )
    session_id, candidates, error = peer_send.resolve_stored_target(
        "old name", live_ids={"shared-id"}
    )
    assert error == ""
    assert candidates == []
    # The live-owned row is skipped; the distinct stored row still resolves.
    assert session_id == "other-id"


def test_ambiguous_stored_matches_are_refused_with_candidates(
    monkeypatch, tmp_path, fake_scan
) -> None:
    fake_scan([])
    _stored(
        monkeypatch,
        [_StoredRow("id1", "multi one"), _StoredRow("id2", "multi two")],
        root=tmp_path,
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


def test_no_stored_match_is_a_clean_no_match(monkeypatch, tmp_path, fake_scan) -> None:
    fake_scan([])
    _stored(monkeypatch, [_StoredRow("id1", "something else")], root=tmp_path)
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
    does not rebuild delivery. The session has real history, which is what
    makes it a recipient at all now (a never-engaged stored row is skipped by
    the resolver AND refused by the cold gate).
    """
    import asyncio

    sid = "deadbeef0123"
    directory = _write_transcript(tmp_path, sid)
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
    from local_operator.session.runtime.inbox import SPOOL_RECEIPT_NOTE

    assert receipt == SPOOL_RECEIPT_NOTE
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
# THE RULE: a session that has not run a real turn yet is NOT a peer-message
# recipient. Resolution refuses it in every address form and skips it in a
# broadcast; delivery refuses it AGAIN here, because a caller can bypass
# resolution and because a COLD target has no ``started`` bit to read — only its
# durable history. Nothing may be dialled, spooled or persisted for such a
# session: the peer row would become the OPENING row of a conversation its owner
# never started. This replaced a quiet dial that persisted exactly that row
# ("delivered to the mailbox (session not started yet; no turn driven)").


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
async def test_an_unstarted_live_session_is_refused_with_no_dial_and_no_spool(
    monkeypatch, tmp_path
) -> None:
    """THE NEW RULE, live half: whatever mode/wake the sender asked for, an
    unstarted record raises the refusal — no dial, no spool, nothing persisted.

    Every shape is exercised because the old behaviour degraded all four to a
    quiet dial that wrote the row; the point is that NONE of them writes now.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    record = _unstarted_record()
    dialled: list[str] = []

    async def _dial(*_a: Any, **_k: Any) -> str:
        dialled.append("dialled")
        raise AssertionError("an unengaged session must never be dialled")

    monkeypatch.setattr("local_operator.mobile.peer_client.send_peer_message", _dial, raising=True)

    for mode, wake in (("mailbox", False), ("mailbox", True), ("steer", False), ("steer", True)):
        with pytest.raises(RuntimeError) as excinfo:
            await peer_send.deliver_peer_message(
                record,
                session_id=record.session_id,
                text=f"note-{mode}-{wake}",
                mode=mode,
                wake=wake,
                sender={"pid": 1},
            )
        assert "pid 4242 has not been engaged yet" in str(excinfo.value), str(excinfo.value)
        assert "no user message has been sent in it" in str(excinfo.value)
        # A LIVE record is not the cold shape: the remedy stays the owner
        # sending a first message into the window that is already open (D5).
        assert "its owner has to send a first message" in str(excinfo.value)

    assert dialled == []
    # Nothing was written anywhere — no spool row, no session directory at all.
    assert not (tmp_path / "sessions").exists()


@pytest.mark.asyncio
async def test_a_cold_target_without_durable_history_is_refused(monkeypatch, tmp_path) -> None:
    """THE NEW RULE, cold half: no live record and no transcript means nobody
    ever started this conversation, so neither cold branch may run — no spool
    (it would become the head of that history) and no ``engage_runtime`` (a
    ``--wake`` would OPEN A TURN in a session whose owner is not there).

    The directory EXISTS, which is the state ``resolve_cold_session`` still
    hands over: a ``/new`` that was abandoned before any message leaves one.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _write_transcript(tmp_path, "cold-fresh", engaged=False)
    engaged: list[str] = []

    async def _engage(*_a: Any, **_k: Any) -> Any:
        engaged.append("engaged")
        raise AssertionError("a wake must not open a turn in a session nobody started")

    monkeypatch.setattr(
        "local_operator.session.runtime.launch.engage_runtime", _engage, raising=True
    )

    for mode, wake in (("mailbox", False), ("mailbox", True), ("steer", False)):
        with pytest.raises(RuntimeError) as excinfo:
            await peer_send.deliver_peer_message(
                None,
                session_id="cold-fresh",
                text="note",
                mode=mode,
                wake=wake,
                sender={"pid": 1},
            )
        assert "session 'cold-fresh' has not been engaged yet" in str(excinfo.value)
        # D5: no runtime means no window to send into, so the remedy names
        # opening the conversation rather than typing into it.
        assert "it becomes a recipient once someone opens it and sends a first message" in str(
            excinfo.value
        ), str(excinfo.value)

    assert engaged == []
    assert not (tmp_path / "sessions" / "cold-fresh" / "inbox.jsonl").exists()


@pytest.mark.asyncio
async def test_a_cold_target_with_history_still_spools_and_engages(monkeypatch, tmp_path) -> None:
    """The unchanged half: a cold target that HAS durable history keeps both
    behaviours exactly as they were — the quiet note spools for the next open,
    and a wake engages a runtime."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _write_transcript(tmp_path, "used-session")

    detail = await peer_send.deliver_peer_message(
        None,
        session_id="used-session",
        text="quiet",
        mode="mailbox",
        wake=False,
        sender={"pid": 1},
    )
    from local_operator.session.runtime.inbox import SPOOL_RECEIPT_NOTE

    assert detail == SPOOL_RECEIPT_NOTE
    assert (tmp_path / "sessions" / "used-session" / "inbox.jsonl").exists()

    engaged: list[tuple[str, str]] = []

    class _Outcome:
        detail = "delivered and woke the session"

    async def _engage(session_id: str, cwd: str, errand: Any, *, config_dir: Any) -> Any:
        engaged.append((session_id, errand.text))
        return _Outcome()

    monkeypatch.setattr(
        "local_operator.session.runtime.launch.engage_runtime", _engage, raising=True
    )
    woken = await peer_send.deliver_peer_message(
        None,
        session_id="used-session",
        text="act",
        mode="mailbox",
        wake=True,
        sender={"pid": 1},
    )
    assert woken == "delivered and woke the session"
    assert engaged == [("used-session", "act")]


def test_the_durable_history_signal_is_a_real_turn_not_a_peer_note(tmp_path) -> None:
    """``session_has_durable_history`` is the cold gate's ONLY signal, and it
    must answer the same question the record's ``started`` bit is seeded from:
    a transcript whose rows are quiet-dialled peer notes (kind ``custom``) is
    NOT history. An absent directory answers False too — the conservative
    unengaged direction, which the owner's first real turn corrects."""
    assert peer_send.session_has_durable_history("never-existed", root=tmp_path) is False

    directory = _write_transcript(tmp_path, "only-notes", engaged=False)
    (directory / TRANSCRIPT_FILENAME).write_text(
        json.dumps(
            {
                "id": "p1",
                "ts": 1,
                "type": "message",
                "payload": {
                    "kind": "custom",
                    "custom_type": "peer_message",
                    "attribution": "user",
                    "details": {"text": "<peer-session-message>hi</peer-session-message>"},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert peer_send.session_has_durable_history("only-notes", root=tmp_path) is False

    _write_transcript(tmp_path, "ran-a-turn")
    assert peer_send.session_has_durable_history("ran-a-turn", root=tmp_path) is True


@pytest.mark.asyncio
async def test_a_started_session_still_dials_the_socket(monkeypatch) -> None:
    """The new branch must not change the normal path: a started session is
    dialled with the sender's own mode/wake and its ack detail is returned
    verbatim."""
    record = _unstarted_record()
    record.started = True

    async def _dial(
        rec: Any,
        *,
        text: str,
        mode: str,
        wake: bool,
        sender: Any,
        message_id: str | None = None,
        deadline_s: float | None = None,
    ) -> str:
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


def test_a_record_without_the_started_key_reads_as_started() -> None:
    """Mixed-version: a record an OLDER (pre-field) binary wrote has no
    ``started`` key, and such a session had no composer gate — so the absent
    key reads True and old-peer behaviour is preserved (broadcasts reach a
    working old runtime; exact sends dial it). A CONSTRUCTED record keeps the
    dataclass default False. See ``SessionRecord.from_json``."""
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
    assert record.started is True
    # The dataclass default is untouched: a fresh this-binary record (the
    # composer window) is still constructed unstarted.
    assert (
        registry.SessionRecord(
            pid=4242,
            kind="tui",
            session_id="new-id",
            conversation_name="new",
            cwd="/tmp",
            model_label="test/model",
            control_port=1,
            control_key="k",
        ).started
        is False
    )
    # An explicit False on the wire still round-trips as False.
    explicit = registry.SessionRecord.from_json(
        {
            "pid": 4242,
            "kind": "tui",
            "session_id": "fresh-wire",
            "conversation_name": "fresh",
            "cwd": "/tmp",
            "model_label": "test/model",
            "control_port": 1,
            "control_key": "k",
            "started": False,
        }
    )
    assert explicit.started is False


def test_the_cold_gate_accepts_only_forms_where_nothing_owns_the_session() -> None:
    """The exact-``session`` cold gate answers one question — may the store be
    asked again? (review round 4, MINOR-1 and NIT-4).

    Two forms qualify, because in both no process is behind the conversation:
    an id the scan never knew, and a record whose pid is gone. The other three
    are answers ABOUT a session, and re-asking the store would deliver around
    the answer the caller just received: a wedged pid still owns the session, an
    unengaged one is deliberately not a recipient, and a refused target+selector
    pair never named one session at all.
    """
    accepted = [
        "no session found with session id 'abc'",
        "target session abc is stale (its pid no longer exists), so nothing can read it",
    ]
    refused = [
        "session 'abc' has not been engaged yet (no user message has been sent in it), "
        "so it cannot receive peer messages — its owner has to send a first message",
        "pid 42 has not reported for 4m, so a plain send will not dial it; it may report "
        "again on its own",
        "pass either a target substring or an exact pid/session, not both",
        "no live session matches 'abc'",
    ]
    for error in accepted:
        assert peer_send.session_id_unowned(error) is True, error
    for error in refused:
        assert peer_send.session_id_unowned(error) is False, error


# -- the delivery outcome (design note A.3) ------------------------------------
#
# Everything below drives ``deliver_peer_message_outcome`` against a REAL
# loopback control server, with the retry bounds monkeypatched down so a 5 s
# production deadline costs milliseconds here (the timing rules in AGENTS.md:
# never spend the real budget in a test). What is under test is the
# CLASSIFICATION -- which observation produces which state -- and the one rule
# the incident turned on: a send that got no answer is never reported as failed.


def _fast_retry(monkeypatch: pytest.MonkeyPatch, *, deadline: float = 0.15) -> None:
    """Shrink the retry bounds; the loop's SHAPE is what these tests exercise."""
    monkeypatch.setattr(peer_send, "PEER_SEND_DEADLINE_S", deadline)
    monkeypatch.setattr(peer_send, "PEER_SEND_RETRY_SLEEPS", (0.0, 0.0))


def _capable(record: _Record) -> _Record:
    record.capabilities = [peer_send.PEER_MESSAGE_ID_CAPABILITY]
    return record


class _FakeControlPeer:
    """A loopback control server that answers one op per connection.

    ``on_op`` gets the parsed op frame and the writer, and decides everything:
    write the row into the target's transcript, sleep past the sender's
    deadline, reply with an ack, an error frame, or nothing at all.
    """

    def __init__(self, on_op: Any) -> None:
        self._on_op = on_op
        self.connections = 0
        self.frames: list[dict[str, Any]] = []
        self._tasks: list[asyncio.Task[Any]] = []

    async def __aenter__(self) -> "_FakeControlPeer":
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        port = self._server.sockets[0].getsockname()[1]
        self.record = _Record(
            os.getpid(), session_id="probe-target", conversation_name="probe-target"
        )
        self.record.control_port = port
        self.record.control_key = "k"
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self._server.close()
        await self._server.wait_closed()
        for task in self._tasks:
            task.cancel()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.connections += 1
        try:
            await reader.readline()  # the bare-key auth frame
            frame = json.loads(await reader.readline())
            self.frames.append(frame)
            reply = await self._on_op(frame, writer)
            if reply is not None:
                writer.write(json.dumps(reply).encode() + b"\n")
                await writer.drain()
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (OSError, ConnectionError):
                pass


def _append_row(root: Path, session_id: str, message_id: str) -> None:
    """Append the row a receiver would write for ``message_id`` to the spool.

    Written as raw JSONL, the shape the sender's disk probe scans for: the id is
    the transcript entry id and the probe looks for that literal.
    """
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / TRANSCRIPT_FILENAME).open("a", encoding="utf-8") as handle:
        row = json.dumps({"id": message_id, "ts": 1, "type": "message", "payload": {}})
        handle.write(row + "\n")


@pytest.mark.asyncio
async def test_the_incident_shape_settles_as_mailbox_with_one_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The reported incident, at the classification layer: the receiver commits
    the row and its ack arrives after the sender's read deadline.

    The message LANDED. The verdict must say so -- ``mailbox``, ``is_error``
    False, the amber ``partial_result`` flag -- and it must be settled by the
    DISK PROBE on the first attempt rather than by spending the whole retry
    budget, which is what keeps the sender's latency where it always was.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _fast_retry(monkeypatch)

    async def on_op(frame: dict[str, Any], writer: Any) -> dict[str, Any]:
        _append_row(tmp_path, "probe-target", frame["message_id"])
        await asyncio.sleep(0.4)  # longer than the sender's read deadline
        return {"op": "ack", "req": frame["req"], "detail": "delivered and woke the session"}

    async with _FakeControlPeer(on_op) as peer:
        outcome = await peer_send.deliver_peer_message_outcome(
            _capable(peer.record),
            session_id="probe-target",
            text="did this land?",
            mode="mailbox",
            wake=True,
            sender={"pid": 1},
        )
        # ONE attempt: the probe found the row between attempts and stopped the
        # loop, which is the whole reason it runs before the sleep.
        assert outcome.attempts == 1, outcome
        assert peer.connections == 1

    assert outcome.state == peer_send.DELIVERY_MAILBOX, outcome
    assert outcome.is_error is False
    assert outcome.partial is True
    assert outcome.wake == peer_send.WAKE_UNCONFIRMED
    assert outcome.cause == "no_answer"
    assert outcome.route == "live"
    assert "delivered to its mailbox" in outcome.text
    assert outcome.message_id in outcome.text
    assert "do not send it again" in outcome.text
    # The identity the sender minted is the one it put on the wire, and it is
    # what the receiver's row is named by.
    assert peer.frames[0]["message_id"] == outcome.message_id
    rows = (tmp_path / "sessions" / "probe-target" / TRANSCRIPT_FILENAME).read_text().splitlines()
    assert [json.loads(row)["id"] for row in rows] == [outcome.message_id]


@pytest.mark.asyncio
async def test_a_silent_receiver_is_unconfirmed_never_failed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No ack and no evidence on disk after every attempt is the HONEST
    residual: the message may still arrive, so the sender may not claim it did
    not. Exactly ``PEER_SEND_ATTEMPTS`` tries against a capable receiver."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _fast_retry(monkeypatch)

    async def on_op(frame: dict[str, Any], writer: Any) -> None:
        await asyncio.sleep(5.0)  # never answers inside the test's lifetime
        return None

    async with _FakeControlPeer(on_op) as peer:
        outcome = await peer_send.deliver_peer_message_outcome(
            _capable(peer.record),
            session_id="probe-target",
            text="anyone home?",
            mode="mailbox",
            wake=True,
            sender={"pid": 1},
        )
        assert peer.connections == peer_send.PEER_SEND_ATTEMPTS

    assert outcome.state == peer_send.DELIVERY_UNCONFIRMED, outcome
    assert outcome.is_error is False
    assert outcome.partial is True
    assert outcome.attempts == peer_send.PEER_SEND_ATTEMPTS
    assert outcome.cause == "no_answer"
    # Sentence case (design N1) and a reader-neutral next step (UX U4): this one
    # string is printed by the tool result, the journal notice and `lop send`
    # stderr alike, and `sessions(op="peek", …)` is not a thing a person at a
    # terminal can run.
    assert "delivery unconfirmed" in outcome.text
    assert "sessions(op=" not in outcome.text
    assert "Check the target's transcript before resending" in outcome.text
    assert "may still arrive" in outcome.text
    # The clause claims only what this route's probe can establish: the receiver
    # advertised the carriage, so its row would be named with THIS id and a miss
    # means the row is genuinely absent (round 1, MAJOR).
    assert "and the message is not yet in its transcript" in outcome.text
    # The copy is pluralised for the number of tries that really happened.
    assert f"after {peer_send.PEER_SEND_ATTEMPTS} attempts" in outcome.text


@pytest.mark.asyncio
async def test_an_old_receiver_is_told_it_was_not_probed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A receiver from before the carriage gets one attempt and no probe, and the
    sentence must not imply a transcript was read (round 1, MAJOR). Its miss
    would prove nothing there: an old receiver names its row with its own id."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _fast_retry(monkeypatch)

    async def on_op(frame: dict[str, Any], writer: Any) -> None:
        await asyncio.sleep(5.0)
        return None

    async with _FakeControlPeer(on_op) as peer:
        record = peer.record
        record.capabilities = [c for c in (record.capabilities or []) if "message-id" not in c]
        outcome = await peer_send.deliver_peer_message_outcome(
            record,
            session_id="probe-target",
            text="anyone home?",
            mode="mailbox",
            wake=True,
            sender={"pid": 1},
        )
        assert peer.connections == 1, "an old receiver gets exactly one attempt"

    assert outcome.state == peer_send.DELIVERY_UNCONFIRMED, outcome
    assert outcome.is_error is False, "a timeout still never reads as failed"
    assert "refus" not in outcome.text.lower()
    assert (
        "could not be confirmed in its transcript" not in outcome.text
    ), "no probe ran at all, so the copy must not imply one did"
    assert "not yet in its transcript" not in outcome.text


@pytest.mark.asyncio
async def test_the_engaged_route_never_claims_the_transcript_is_clear(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The cold-target route cannot read a capability, so a probe miss there is
    UNKNOWN, not absence (round 1, MAJOR): the engaged arm must say so rather
    than asserting the message is not in a transcript it cannot interpret."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    from local_operator.mobile import peer_send as ps

    async def _boom(*args: Any, **kwargs: Any) -> Any:
        raise TimeoutError("the spawned runtime never answered")

    # The arm itself, not the resolver above it: the cold-target path is entered
    # through ``engage_runtime``, whose import is function-local, so the double
    # is installed on the launch module (and the gate that refuses a
    # never-engaged target is above this unit and not what is under test).
    monkeypatch.setattr("local_operator.session.runtime.launch.engage_runtime", _boom)
    outcome = await ps._engaged_outcome(
        "cold-target",
        target="cold-target (not running)",
        message_id="peer-" + "a" * 32,
        text="anyone home?",
        mode="mailbox",
        wake=True,
        sender={"pid": 1},
        cwd=str(tmp_path),
    )
    assert outcome.route == "engaged", outcome
    assert outcome.state == ps.DELIVERY_UNCONFIRMED, outcome
    assert "could not be confirmed in its transcript" in outcome.text
    assert "not yet in its transcript" not in outcome.text


@pytest.mark.asyncio
async def test_a_refused_dial_is_the_one_proven_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A socket that never opened wrote nothing, so this is the arm allowed to
    say nothing was delivered -- and it says it with the id and the retry."""
    _fast_retry(monkeypatch)
    record = _capable(_Record(os.getpid(), session_id="gone"))
    # A port nothing is listening on: ``open_connection`` raises at once, which
    # is the ControlDialFailed class rather than the ambiguous transport one.
    record.control_port = 1

    outcome = await peer_send.deliver_peer_message_outcome(
        record, session_id="gone", text="hello?", mode="mailbox", wake=True, sender={"pid": 1}
    )

    assert outcome.state == peer_send.DELIVERY_FAILED, outcome
    assert outcome.is_error is True
    assert outcome.partial is False
    assert outcome.cause == "dial_refused"
    assert outcome.attempts == peer_send.PEER_SEND_ATTEMPTS
    assert "Nothing was delivered" in outcome.text
    assert outcome.message_id in outcome.text
    assert "retry the send" in outcome.text


@pytest.mark.asyncio
async def test_a_peer_error_frame_is_a_refusal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The peer ANSWERED no (an older registrant, a handle that cannot
    receive): nothing landed, and the peer's own sentence is the reason."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _fast_retry(monkeypatch)

    async def on_op(frame: dict[str, Any], writer: Any) -> dict[str, Any]:
        return {
            "op": "error",
            "req": frame["req"],
            "message": "this session cannot receive peer messages",
        }

    async with _FakeControlPeer(on_op) as peer:
        outcome = await peer_send.deliver_peer_message_outcome(
            _capable(peer.record),
            session_id="probe-target",
            text="hello",
            mode="mailbox",
            wake=False,
            sender={"pid": 1},
        )
        assert peer.connections == 1

    assert outcome.state == peer_send.DELIVERY_FAILED, outcome
    assert outcome.cause == "peer_refused"
    assert "this session cannot receive peer messages" in outcome.text
    assert "Nothing was delivered" in outcome.text


@pytest.mark.asyncio
async def test_a_lost_ack_on_a_quiet_send_is_delivered(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Q3: a QUIET send has no wake to be unconfirmed, so once the probe finds
    the row the outcome is a plain delivery -- with the cause recorded, because
    the receipt really was lost."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _fast_retry(monkeypatch)

    async def on_op(frame: dict[str, Any], writer: Any) -> dict[str, Any]:
        _append_row(tmp_path, "probe-target", frame["message_id"])
        await asyncio.sleep(0.4)
        return {"op": "ack", "req": frame["req"], "detail": "delivered to the mailbox"}

    async with _FakeControlPeer(on_op) as peer:
        outcome = await peer_send.deliver_peer_message_outcome(
            _capable(peer.record),
            session_id="probe-target",
            text="fyi",
            mode="mailbox",
            wake=False,
            sender={"pid": 1},
        )

    assert outcome.state == peer_send.DELIVERY_DELIVERED, outcome
    assert outcome.is_error is False
    assert outcome.partial is False
    assert outcome.wake == peer_send.WAKE_NOT_REQUESTED
    assert outcome.cause == "ack_lost"
    assert "its receipt was lost" in outcome.text


@pytest.mark.asyncio
async def test_a_receiver_without_the_capability_gets_one_attempt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The back-compat gate (design note C). An OLD receiver has no dedupe, so a
    retry would write the message TWICE -- the sender therefore makes one
    attempt, never probes, and a timeout is still UNC0NFIRMED rather than
    failed."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _fast_retry(monkeypatch)

    async def on_op(frame: dict[str, Any], writer: Any) -> None:
        await asyncio.sleep(5.0)
        return None

    async with _FakeControlPeer(on_op) as peer:
        assert peer_send.peer_message_id_capable(peer.record) is False
        outcome = await peer_send.deliver_peer_message_outcome(
            peer.record,  # capabilities deliberately empty: an older build
            session_id="probe-target",
            text="anyone home?",
            mode="mailbox",
            wake=True,
            sender={"pid": 1},
        )
        assert peer.connections == 1

    assert outcome.state == peer_send.DELIVERY_UNCONFIRMED, outcome
    assert outcome.attempts == 1
    assert outcome.is_error is False
    assert "after 1 attempt" in outcome.text


@pytest.mark.asyncio
async def test_a_duplicate_ack_makes_a_wake_send_a_mailbox(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A prior attempt's ack was the lost one, and the receiver proves it by
    answering the re-send with the id it already owns. The row is durable; what
    the sender cannot know is whether the wake that shared the lost ack ran."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _fast_retry(monkeypatch)
    seen = {"n": 0}

    async def on_op(frame: dict[str, Any], writer: Any) -> dict[str, Any] | None:
        seen["n"] += 1
        if seen["n"] == 1:
            await asyncio.sleep(5.0)  # the first ack never arrives
            return None
        return {
            "op": "ack",
            "req": frame["req"],
            "detail": "duplicate — this message is already delivered",
            "delivery": {
                "message_id": frame["message_id"],
                "committed": True,
                "queued": False,
                "duplicate": True,
            },
        }

    async with _FakeControlPeer(on_op) as peer:
        outcome = await peer_send.deliver_peer_message_outcome(
            _capable(peer.record),
            session_id="probe-target",
            text="gates are green",
            mode="mailbox",
            wake=True,
            sender={"pid": 1},
        )

    assert outcome.state == peer_send.DELIVERY_MAILBOX, outcome
    assert outcome.wake == peer_send.WAKE_UNCONFIRMED
    assert outcome.cause == "ack_lost"
    assert outcome.is_error is False


@pytest.mark.asyncio
async def test_an_in_flight_duplicate_is_not_reported_as_delivered(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The receiver answers ``duplicate`` from TWO places (round 1, MINOR-4).

    ``committed=True`` means its transcript index owns the id -- the durable
    "already delivered" sentence is then true. ``committed=False`` means its
    IN-FLIGHT set does: an earlier attempt of this same send is still mid-hop, so
    nothing is on disk and claiming delivery would state a fact the in-flight
    window has not established. The two are reachable in the incident shape
    (attempt 2 lands while attempt 1 is still inside the receiver), so they must
    not classify alike.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _fast_retry(monkeypatch)
    seen = {"n": 0}

    async def on_op(frame: dict[str, Any], writer: Any) -> dict[str, Any] | None:
        seen["n"] += 1
        if seen["n"] == 1:
            await asyncio.sleep(5.0)  # the first ack never arrives
            return None
        return {
            "op": "ack",
            "req": frame["req"],
            "detail": "duplicate — this message is already delivered",
            "delivery": {
                "message_id": frame["message_id"],
                "committed": False,  # in flight, NOT on disk
                "queued": False,
                "duplicate": True,
            },
        }

    async with _FakeControlPeer(on_op) as peer:
        outcome = await peer_send.deliver_peer_message_outcome(
            _capable(peer.record),
            session_id="probe-target",
            text="gates are green",
            mode="mailbox",
            wake=True,
            sender={"pid": 1},
        )

    assert outcome.state == peer_send.DELIVERY_UNCONFIRMED, outcome
    assert outcome.cause == "in_flight"
    assert outcome.wake == peer_send.WAKE_UNCONFIRMED
    assert outcome.is_error is False
    assert "still being delivered" in outcome.text
    assert "do not send it again" not in outcome.text


@pytest.mark.asyncio
async def test_a_queued_ack_keeps_the_receivers_own_sentence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The hop-timeout ack answered -- with "queued, the terminal is busy" -- so
    the sender must report the wake as UNANSWERED (not ``acked``) and keep the
    receiver's accurate reason instead of replacing it with its own no-answer
    copy (round 1, MINOR-1)."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _fast_retry(monkeypatch)

    async def on_op(frame: dict[str, Any], writer: Any) -> dict[str, Any]:
        return {
            "op": "ack",
            "req": frame["req"],
            "detail": "queued — the terminal is busy; it lands when its turn settles",
            "delivery": {
                "message_id": frame["message_id"],
                "committed": False,
                "queued": True,
                "duplicate": False,
            },
        }

    async with _FakeControlPeer(on_op) as peer:
        outcome = await peer_send.deliver_peer_message_outcome(
            _capable(peer.record),
            session_id="probe-target",
            text="gates are green",
            mode="mailbox",
            wake=True,
            sender={"pid": 1},
        )

    assert outcome.state == peer_send.DELIVERY_UNCONFIRMED, outcome
    assert outcome.cause == "queued_busy"
    assert outcome.wake == peer_send.WAKE_UNCONFIRMED, "a queued wake did NOT answer"
    assert outcome.is_error is False
    assert "the terminal is busy" in outcome.text
    assert "no answer within 5s" not in outcome.text


@pytest.mark.asyncio
async def test_the_receipt_line_is_unchanged_for_a_plain_delivery(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The common path stays byte-stable: an ack is the receipt, and the target
    prefix is the one every send has always used."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _fast_retry(monkeypatch)

    async def on_op(frame: dict[str, Any], writer: Any) -> dict[str, Any]:
        return {
            "op": "ack",
            "req": frame["req"],
            "detail": "delivered and woke the session",
            "delivery": {
                "message_id": frame["message_id"],
                "committed": True,
                "queued": False,
                "duplicate": False,
            },
        }

    async with _FakeControlPeer(on_op) as peer:
        outcome = await peer_send.deliver_peer_message_outcome(
            _capable(peer.record),
            session_id="probe-target",
            text="gates are green",
            mode="mailbox",
            wake=True,
            sender={"pid": 1},
        )

    assert outcome.state == peer_send.DELIVERY_DELIVERED, outcome
    assert outcome.wake == peer_send.WAKE_ACKED
    assert outcome.cause == ""
    assert outcome.is_error is False
    assert outcome.partial is False
    assert outcome.text == (f"→ probe-target (pid {os.getpid()}): delivered and woke the session")


def test_the_delivered_line_is_the_receipt_and_the_failed_line_is_the_only_claim() -> None:
    """The two text shapes, asserted on the builder's own output: every state
    but ``failed`` keeps the arrow receipt, and only ``failed`` may say nothing
    was delivered."""
    delivered = peer_send.DeliveryOutcome(
        peer_send.DELIVERY_DELIVERED,
        "ack",
        "peer-0",
        peer_send.WAKE_ACKED,
        1,
        "",
        "live",
        "p (pid 1)",
    )
    failed = peer_send.DeliveryOutcome(
        peer_send.DELIVERY_FAILED,
        "pid 1 refused the connection",
        "peer-0",
        peer_send.WAKE_UNCONFIRMED,
        3,
        "dial_refused",
        "live",
        "p (pid 1)",
    )
    assert delivered.text == "→ p (pid 1): ack"
    assert delivered.is_error is False and delivered.partial is False
    assert failed.text.startswith("could not deliver to p (pid 1): pid 1 refused the connection.")
    assert "Nothing was delivered (id peer-0)" in failed.text
    assert "retry" in failed.text
    assert failed.is_error is True
    # The persisted payload and the live one are the SAME dict shape.
    assert set(failed.details()) == {
        "state",
        "message_id",
        "wake",
        "attempts",
        "cause",
        "route",
        # The human cause clause (design D3).
        "reason",
    }


def test_probe_transcript_for_reads_a_bounded_tail(tmp_path: Path) -> None:
    """The probe's bound is real: an id older than the tail is reported ABSENT,
    which reads as unknown, never as failed."""
    directory = tmp_path / "sessions" / "s"
    directory.mkdir(parents=True)
    path = directory / TRANSCRIPT_FILENAME
    path.write_text(json.dumps({"id": "peer-" + "a" * 32}) + "\n")
    assert peer_send.probe_transcript_for("peer-" + "a" * 32, "s", root=tmp_path) is True
    assert peer_send.probe_transcript_for("peer-" + "b" * 32, "s", root=tmp_path) is False
    missing = peer_send.probe_transcript_for("peer-" + "a" * 32, "no-such-session", root=tmp_path)
    assert missing is False
    # Past the bound the old row is out of the window.
    with path.open("a", encoding="utf-8") as handle:
        handle.write(("x" * 1024 + "\n") * (peer_send.PEER_SEND_PROBE_TAIL_BYTES // 1024 + 2))
    assert peer_send.probe_transcript_for("peer-" + "a" * 32, "s", root=tmp_path) is False
