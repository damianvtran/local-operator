"""``net_session_peek`` / ``peer_session_peek`` at the handler level.

The parity proof (remote text == the owner's local peek) lives beside the
local peek tests in ``tests/unit/tools/test_sessions_tool.py``, and the real
two-relay run in ``test_sessions_pilot_cli.py``; this file pins the op's
contract edges: authority, wire typing, the old-peer answer and the regex caps.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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
    """A peer-supplied regex runs on the owner's CPU, so the length cap and the
    classic catastrophic shape are refused up front (the cheap pre-check; the
    match timeout below is the real bound)."""
    _journal(tmp_path)
    frame = {"session_id": SESSION, "query": query, "regex": True}
    if words:
        with pytest.raises(MeshRefusal, match=words):
            net_peek.serve(tmp_path, frame, own_label="b")
    else:
        net_peek.serve(tmp_path, frame, own_label="b")


def _row(directory: Path, text: str) -> None:
    row = {
        "id": f"{99:032x}",
        "ts": 99,
        "type": "message",
        "payload": {"kind": "message", "role": "user", "content": [{"text": text}]},
    }
    with (directory / "transcript.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


#: PR #3831 review M1 / QA Q1: every pattern that slipped past the shape
#: heuristic, plus the reviewer's polynomial one. Under ``re`` each of these
#: ran for seconds-to-forever on a row like the one below.
_HEURISTIC_BYPASSES = [
    "((a+))+$",
    "(?:(a+))+$",
    "(a|a)+$",
    "(-|--)+$",
    "(a+){,30}$",
    "\\w*\\w*\\w*\\w*\\w*\\w*!",
    "((-+))+$",
]


def test_a_catastrophic_peer_regex_is_cut_by_the_deadline_and_frees_its_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The MATCH is bounded, not the shape: each bypass returns (a result or the
    named ``regex_timeout`` refusal) inside the slow-lane deadline, and ONE
    worker serves all of them and then an ordinary peek — the slot a pattern
    held is back in the pool, which is what four such peeks used to take from
    every peer."""
    directory = _journal(tmp_path)
    _row(directory, "a" * 40 + "-" * 40 + "!" + "a" * 60)
    monkeypatch.setattr(net_peek, "PEER_REGEX_REPLY_RESERVE_S", 0.4)
    monkeypatch.setattr(net_peek, "_owner_record", lambda session_id: None)
    deadline_s = 1.0

    budget_s = deadline_s - net_peek.PEER_REGEX_REPLY_RESERVE_S

    def _serve(frame: dict[str, Any]) -> tuple[float, str]:
        # ``regex`` bounds PROCESS CPU time (see PEER_REGEX_REPLY_RESERVE_S);
        # this thread's CPU can only be at or under that, so it is a safe upper
        # clock to assert on, and wall time on a loaded CI box stretches.
        cpu = time.thread_time()
        try:
            with relay._slow_deadline(time.monotonic() + deadline_s):
                answer = net_peek.serve(tmp_path, frame, own_label="device-b")
        except MeshRefusal as refusal:
            assert refusal.code == "regex_timeout", refusal.sentence
            assert "ran out of time on device-b" in refusal.sentence
            return time.thread_time() - cpu, refusal.code
        return time.thread_time() - cpu, str(answer["mode"])

    outcomes = []
    with ThreadPoolExecutor(max_workers=1) as pool:
        for query in _HEURISTIC_BYPASSES:
            frame = {"session_id": SESSION, "query": query, "regex": True}
            # A worker that never returned fails here (generous for a loaded
            # box), instead of hanging the suite as the ``re`` version would.
            cpu_s, outcome = pool.submit(_serve, frame).result(timeout=30)
            # The budget plus the walk's own row parsing/imports.
            assert cpu_s < budget_s + 0.5, (query, cpu_s)
            outcomes.append(outcome)
        _elapsed, mode = pool.submit(_serve, {"session_id": SESSION, "head": 1}).result(timeout=5)
    assert mode == "head"
    # The discriminating rows really did hit the bound (not just happen to be fast).
    assert outcomes.count("regex_timeout") >= 3, outcomes


class _PeekLink:
    """The parts of a ``PeerLink`` ``_dispatch_slow``/``_run_handler`` read."""

    def __init__(self, device_id: str) -> None:
        self.device_id = device_id
        self.link_id = f"l_{device_id}"
        self.network_id = "n_test"
        self.epoch = 1
        self.sent: list[dict[str, Any]] = []
        self.got = threading.Event()

    def send(self, frame: dict[str, Any]) -> bool:
        self.sent.append(frame)
        self.got.set()
        return True


def _peek_server(root: Path) -> relay.RelayServer:
    return relay.RelayServer(
        root=root, settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1")
    )


def test_the_regex_budget_is_capped_well_inside_the_deadline() -> None:
    """QA Q5/Q6: a full slow-lane deadline no longer becomes an ~18 s regex
    budget. The cap holds a 3.5x load stretch (QA r2 measured 62.8 s wall for
    18 s CPU) inside the deadline, so the owner's named refusal comes first."""
    with relay._slow_deadline(time.monotonic() + net_peek.SESSION_PEEK_DEADLINE_S):
        assert net_peek._regex_budget_s() == net_peek.PEER_REGEX_BUDGET_S
    # A frame that queued for a worker gets only what is left, minus the reserve.
    with relay._slow_deadline(time.monotonic() + 3.0):
        assert net_peek._regex_budget_s() <= 3.0 - net_peek.PEER_REGEX_REPLY_RESERVE_S
    assert net_peek.PEER_REGEX_BUDGET_S * 3.5 < (
        net_peek.SESSION_PEEK_DEADLINE_S - net_peek.PEER_REGEX_REPLY_RESERVE_S
    )


def test_a_regex_burst_from_one_device_leaves_the_slow_lane_to_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """QA Q5: five concurrent bad peeks from ONE device used to take all four
    slow workers, and a borrow/move/sync arriving next queued past its
    deadline. Now the first runs, the other four are refused AT ONCE with the
    named sentence (no slot, no worker), a different slow op from another
    device starts immediately, and once the first finishes the device may
    search again."""
    directory = _journal(tmp_path)
    _row(directory, "a" * 80 + "-" * 80 + "!")
    monkeypatch.setattr(net_peek, "PEER_REGEX_BUDGET_S", 1.5)
    monkeypatch.setattr(net_peek, "_owner_record", lambda session_id: None)
    server = _peek_server(tmp_path)
    monkeypatch.setattr(server, "_own_label", lambda: "device-b")
    peek_handler = net_peek.make_handler(server)
    deadline_s = net_peek.SESSION_PEEK_DEADLINE_S
    member = _PeekLink("device-a")
    bad = {"op": "net_session_peek", "session_id": SESSION, "query": "(a|a)+$", "regex": True}
    release = threading.Event()
    sync_started = threading.Event()

    def held_sync(link: Any, frame: dict[str, Any]) -> dict[str, Any]:
        sync_started.set()
        release.wait(10)
        return {}

    try:
        replies = [
            server._dispatch_slow(  # noqa: SLF001
                member,  # type: ignore[arg-type]
                {**bad, "req": n},
                peek_handler,
                types.Granted(action="net_session_peek"),
                deadline_s,
            )
            for n in range(5)
        ]
        assert replies[0] is None, "the first search is admitted"
        for reply in replies[1:]:
            assert reply is not None and reply["op"] == "error", reply
            assert "search from this device is already running on device-b" in reply["message"]
        # A literal peek from the same device is not limited (scan-bounded only).
        literal = {"op": "net_session_peek", "session_id": SESSION, "query": "row"}
        assert (
            server._dispatch_slow(  # noqa: SLF001
                member,  # type: ignore[arg-type]
                {**literal, "req": 9},
                peek_handler,
                types.Granted(action="net_session_peek"),
                deadline_s,
            )
            is None
        )
        other = _PeekLink("device-c")
        assert (
            server._dispatch_slow(  # noqa: SLF001
                other,  # type: ignore[arg-type]
                {"op": "net_sync", "req": 50},
                held_sync,
                types.Granted(action="net_sync"),
                10.0,
            )
            is None
        )
        # Started while the bad search still holds its worker (budget 1.5 s).
        assert sync_started.wait(1.0), "the other slow op waited behind the regex burst"
        # The first search ends with the OWNER's named refusal, and the slot is
        # given back: the same device may search again.
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not any(f.get("req") == 0 for f in member.sent):
            time.sleep(0.05)
        first = next(f for f in member.sent if f.get("req") == 0)
        assert first.get("code") == "regex_timeout", first
        time.sleep(0.1)  # the worker's ``finally`` releases just after it sends
        again = server._dispatch_slow(  # noqa: SLF001
            member,  # type: ignore[arg-type]
            {**bad, "query": "row \\d+", "req": 7},
            peek_handler,
            types.Granted(action="net_session_peek"),
            deadline_s,
        )
        assert again is None
        # Let that search answer so no worker outlives the test.
        while time.monotonic() < deadline and not any(f.get("req") == 7 for f in member.sent):
            time.sleep(0.05)
        assert any(f.get("req") == 7 and f.get("op") != "error" for f in member.sent)
    finally:
        release.set()
        server.stop()


def test_a_timed_out_regex_reaches_the_viewer_as_the_owners_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """QA Q6: over the real slow lane, at the shipped 20 s deadline, a bad
    pattern is answered with the owner's ``regex_timeout`` sentence — not the
    relay's generic "did not finish within 20 s ... ask for its status". (Before
    the cap the budget was ~18 s, so this waited out the deadline instead.)
    The cap is shrunk only to keep the test quick."""
    directory = _journal(tmp_path)
    _row(directory, "a" * 80 + "-" * 80 + "!")
    monkeypatch.setattr(net_peek, "PEER_REGEX_BUDGET_S", 0.5)
    monkeypatch.setattr(net_peek, "_owner_record", lambda session_id: None)
    server = _peek_server(tmp_path)
    monkeypatch.setattr(server, "_own_label", lambda: "device-b")
    member = _PeekLink("device-a")
    registered = server.slow_op_deadline("net_session_peek")
    assert registered is not None and registered == net_peek.SESSION_PEEK_DEADLINE_S
    deadline_s = registered
    try:
        started = time.monotonic()
        assert (
            server._dispatch_slow(  # noqa: SLF001
                member,  # type: ignore[arg-type]
                {
                    "op": "net_session_peek",
                    "req": 1,
                    "session_id": SESSION,
                    "query": "(a|a)+$",
                    "regex": True,
                },
                net_peek.make_handler(server),
                types.Granted(action="net_session_peek"),
                deadline_s,
            )
            is None
        )
        assert member.got.wait(deadline_s + 5)
        reply = member.sent[0]
        assert reply.get("code") == "regex_timeout", reply
        assert "ran out of time on device-b" in reply["message"]
        assert "did not finish within" not in reply["message"]
        assert time.monotonic() - started < deadline_s
    finally:
        server.stop()


@pytest.mark.parametrize("session_id", ["../outside", "a/b", "a\\b", ".", "..", "/etc"])
def test_the_owner_refuses_a_session_id_with_path_syntax(tmp_path: Path, session_id: str) -> None:
    """PR #3831 review m1: the path is built from a peer's string, so it must be
    one safe component whatever the chokepoint decided — even when it names a
    directory that exists."""
    _journal(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "transcript.jsonl").write_text("SECRET\n", encoding="utf-8")
    with pytest.raises(MeshRefusal) as caught:
        net_peek.serve(tmp_path, {"session_id": session_id}, own_label="b")
    assert caught.value.code == "invalid_request"
    assert "without path syntax" in caught.value.sentence


def test_the_header_prefers_the_live_conversation_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PR #3831 review m2: parity with the local peek's ``_peek_name`` — a live
    record's ``conversation_name`` first, then the stored title."""
    _journal(tmp_path)
    live = SimpleNamespace(conversation_name="renamed-live")
    monkeypatch.setattr(net_peek, "_owner_record", lambda session_id: live)
    answer = net_peek.serve(tmp_path, {"session_id": SESSION, "head": 1}, own_label="b")
    assert answer["name"] == "renamed-live"
    monkeypatch.setattr(net_peek, "_owner_record", lambda session_id: None)
    answer = net_peek.serve(tmp_path, {"session_id": SESSION, "head": 1}, own_label="b")
    # Not live: the stored title, exactly what the local peek falls back to.
    assert answer["name"] == "row 1"


def _silent_server(calls: list[Any]) -> Any:
    def _call(op: str, peer: str, *, timeout: float, **fields: Any) -> dict[str, Any]:
        calls.append(op)
        raise MeshRefusal("peer_unreachable", f"{peer} stopped answering before it replied")

    return SimpleNamespace(
        _resolve_peer=lambda peer: peer,
        _ensure_link=lambda peer: SimpleNamespace(capabilities=frozenset({wire.SESSION_PEEK_V1})),
        _peer_label=lambda peer: "device-b",
        _local_peer_call=_call,
    )


def test_a_peek_that_outlasts_its_deadline_says_the_owner_ran_out_of_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PR #3831 QA Q3: silence for the whole budget is the owner's read
    overrunning, worded by the member's NAME — not a raw device id "stopped
    answering", which read as a liveness failure."""
    monkeypatch.setattr(net_peek, "SESSION_PEEK_DEADLINE_S", 0.0)
    handler = net_peek.local_handler(_silent_server([]))
    with pytest.raises(MeshRefusal) as caught:
        handler({"peer": "d_raw", "session_id": SESSION, "head": 1})
    assert caught.value.code == "peer_timeout"
    assert caught.value.sentence.startswith("device-b did not finish this peek within")
    assert "ran out of time" in caught.value.sentence


def test_a_quick_transport_failure_stays_unreachable() -> None:
    """Failing before the deadline is a reachability fact, kept verbatim."""
    handler = net_peek.local_handler(_silent_server([]))
    with pytest.raises(MeshRefusal) as caught:
        handler({"peer": "d_raw", "session_id": SESSION, "head": 1})
    assert caught.value.code == "peer_unreachable"


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        # The relay's own sentence reaches the operator unchanged.
        (
            "device-b runs a build without the remote peek (session_peek/1); "
            "update it there to read its journal",
            "device-b runs a build without the remote peek (session_peek/1); "
            "update it there to read its journal",
        ),
        # A machine token from another build is said in words, never printed raw.
        ("connect_failed:ConnectionRefusedError", "it did not answer"),
        # A missing reason still gets this project's fallback sentence.
        ("", "that device cannot serve a journal peek"),
    ],
)
def test_an_unsupported_answers_reason_is_glossed_before_the_cli_prints_it(
    monkeypatch: pytest.MonkeyPatch, reason: str, expected: str
) -> None:
    """PR #3831 CI fix: the peer-supplied ``reason`` goes through
    ``resume.peer_reason_words`` (test_reason_surfaces.py's guard)."""
    from local_operator.network import cli as net_cli

    monkeypatch.setattr(
        net_cli, "_relay_answer", lambda *a, **k: {"unsupported": True, "reason": reason}
    )
    assert net_cli._peek_journal_call("device-b", "s-1", {}) == (None, expected)
