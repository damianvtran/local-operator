"""Q-R8-1 — the control socket's own line bound, and a refusal that names the frame.

WHAT THIS FILE IS ABOUT, and why the three claims are separate. `lop sessions
--all-peers`, `lop sessions --peer <dev>` and `lop network sessions --all-peers`
all read ONE reply, `peer_session_rows`, whose size GROWS WITH THE MESH: a row per
session on every device. Measured on this tree at 381 bytes a row, so ~17 KB at 45
sessions; QA's round-8 rig measured 19,743 bytes on a busier network. The client
half used to read that reply under ``wire.MAX_HANDSHAKE_LINE`` (16 KiB) — a
PRE-AUTH bound whose whole purpose is to stop an unauthenticated peer making this
process buffer without limit. Over the bound the reader raised ``LinkCryptoError``,
``relay.control_request`` swallowed it into ``None``, and ``None`` means "no relay
answered me", which every caller renders as a WEDGED RELAY with the remedy
``lop network restart``. The relay was healthy and answering in milliseconds.

Any one claim alone can pass while the defect stands, so:

1. the payload really does cross the old cap AND the shipped reader is what refused
   it — proven together, on the same live reply, with the pre-fix reader as the
   control;
2. the control reader's own bound is exact at its edge and refuses BY SIZE, so the
   next ceiling is a named one rather than a silent drop;
3. every refusal on this path names the frame it refused and never the wedge.
"""

from __future__ import annotations

import os
import re
import socket
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator import cli as main_cli
from local_operator.network import dial as session_dial
from local_operator.network import relay, store, types, wire
from local_operator.network.types import MeshRefusal

#: Enough stored sessions that a REAL relay's catalogue reply crosses the 16 KiB
#: handshake cap on this tree (45 x ~381 bytes = ~17 KB). Asserted against the
#: measured size in the test that uses it, so a row that shrinks says so instead
#: of quietly stopping being about this bound.
SESSIONS_OVER_THE_CAP = 45

#: A small stand-in for the control socket's bound. The shipped value is 8 MiB
#: (``dial.MAX_SESSION_FRAME_BYTES``) and a unit test cannot stand up a mesh that
#: large; the code path is the same one, and the caller reads the constant at call
#: time precisely so a test can shrink it.
TEST_BOUND = 4096


# ---------------------------------------------------------------------------
# The two halves of a control conversation, without a mesh
# ---------------------------------------------------------------------------


class _StubControl:
    """A loopback server that answers one control request with a canned line.

    The CLIENT half is what is under test, and its failure mode is the SIZE of what
    comes back — so the reply is canned and only the reader is real. ``payload=None``
    is the other shape an operator meets: a socket that accepts, receives the op and
    answers NOTHING.
    """

    def __init__(self, payload: bytes | None, *, hold_s: float = 0.5) -> None:
        self._payload = payload
        self._hold_s = hold_s
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.bind(("127.0.0.1", 0))
        self._server.listen(1)
        self.port = int(self._server.getsockname()[1])
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        try:
            conn, _addr = self._server.accept()
        except OSError:
            return
        with conn:
            conn.settimeout(5.0)
            seen = b""
            deadline = time.monotonic() + 5.0
            while seen.count(b"\n") < 2 and time.monotonic() < deadline:
                try:
                    chunk = conn.recv(1 << 16)
                except OSError:
                    return
                if not chunk:
                    return
                seen += chunk
            if self._payload is None:
                time.sleep(self._hold_s)
                return
            try:
                conn.sendall(self._payload)
            except OSError:
                pass

    def close(self) -> None:
        self._server.close()
        self._thread.join(timeout=5.0)


def _record(port: int) -> types.PeerRecord:
    """The two fields ``control_request`` reads off a relay's discovery record."""
    return types.PeerRecord(pid=os.getpid(), control_port=port, control_key="probe-key")


def _line_of(total: int) -> bytes:
    """A well-formed reply line of EXACTLY ``total`` bytes, newline included."""
    body: dict[str, Any] = {"op": "ack", "req": 1, "detail": {"sessions": [], "pad": ""}}
    head = len(wire.encode_line(body))
    assert total > head, f"{total} is too small for a reply line ({head} bytes of scaffolding)"
    body["detail"]["pad"] = "x" * (total - head)
    line = wire.encode_line(body)
    assert len(line) == total
    return line


def _live_relay(root: Path, monkeypatch: pytest.MonkeyPatch) -> relay.RelayServer:
    """A REAL relay on this root, publishing the record a CLI dials.

    The relay is the only process that speaks the mesh, so "the relay answered"
    cannot be tested at a lesser boundary — this is the same construction
    ``tests/unit/network/test_refusals.py::_live_relay`` uses, and the ambient
    config dir follows it because the CLI resolves its root from the environment.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    server = relay.RelayServer(
        root=root, settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1")
    )
    server.start()
    return server


def _seed_stored_sessions(root: Path, count: int) -> None:
    """``count`` conversations on disk, through the store the catalogue walks."""
    for index in range(count):
        directory = root / "sessions" / f"bound{index:04d}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "transcript.jsonl").write_text("", encoding="utf-8")


def _read_reply_raw(record: types.PeerRecord, op: str) -> bytes:
    """The reply LINE as it goes on the wire, read with no client bound at all."""
    with socket.create_connection(("127.0.0.1", record.control_port), timeout=10.0) as sock:
        sock.sendall(wire.encode_line({"key": record.control_key, "client": "probe"}))
        sock.sendall(wire.encode_line({"op": op, "req": 1}))
        buffer = b""
        deadline = time.monotonic() + 20.0
        while b"\n" not in buffer and time.monotonic() < deadline:
            chunk = sock.recv(1 << 20)
            if not chunk:
                break
            buffer += chunk
    line, _newline, _rest = buffer.partition(b"\n")
    return line


def _diagnose_a_silent_socket_with_the_pre_fix_reader(record: types.PeerRecord, op: str) -> str:
    """The client half as it shipped: ``wire.FrameReader`` under the 16 KiB cap."""
    with socket.create_connection(("127.0.0.1", record.control_port), timeout=10.0) as sock:
        sock.sendall(wire.encode_line({"key": record.control_key, "client": "probe"}))
        sock.sendall(wire.encode_line({"op": op, "req": 1}))
        with pytest.raises(wire.LinkCryptoError) as excinfo:
            wire.FrameReader(sock).read_line(wire.deadline_in(20.0))
    return str(excinfo.value)


def _sessions_args(**fields: object) -> Any:
    """`lop sessions`'s flag set, as `main` hands it to `sessions_command`."""
    flags: dict[str, object] = {
        "json": True,
        "sessions_command": None,
        "all": False,
        "limit": None,
        "peer": "",
        "all_peers": True,
    }
    flags.update(fields)
    from argparse import Namespace

    return Namespace(**flags)


# ---------------------------------------------------------------------------
# 1. The reply, over the cap, read by the shipped client
# ---------------------------------------------------------------------------


def test_a_federated_listing_over_the_handshake_cap_is_read_not_refused(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE DEFECT AT THE REAL BOUNDARY: a live relay's big catalogue, read whole.

    A real relay, a real store with enough conversations to cross the old cap, and
    the reply read with the client the CLI uses. Before the fix this reply was
    ``None`` (the reader raised, ``control_request`` swallowed it) and the operator
    was told the relay was wedged.
    """
    server = _live_relay(root, monkeypatch)
    try:
        _seed_stored_sessions(root, SESSIONS_OVER_THE_CAP)
        record = store.find_own_relay(root)
        assert record is not None, "the live relay published no record to dial"

        reply = relay.control_request(record, "peer_session_rows", timeout=20.0)
        assert reply is not None, "a healthy relay's listing was reported as no answer"
        assert reply["op"] == "ack"
        rows = (reply.get("detail") or {}).get("sessions") or []
        assert len(rows) == SESSIONS_OVER_THE_CAP

        size = len(_read_reply_raw(record, "peer_session_rows"))
        assert size > wire.MAX_HANDSHAKE_LINE, (
            f"{SESSIONS_OVER_THE_CAP} stored sessions produced only {size} bytes, so this "
            "payload no longer crosses the cap it is about; raise SESSIONS_OVER_THE_CAP"
        )
    finally:
        server.stop()


def test_the_pre_fix_reader_is_what_refuses_that_same_reply(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE CAUSE, ON IDENTICAL DATA: the reader's cap, not the relay.

    The control for the test above. The same live reply, read by the client half as
    it shipped — ``wire.FrameReader`` under ``MAX_HANDSHAKE_LINE`` — raises, and the
    exception names the 16 KiB number. That is what makes the pass above
    attributable to the reader change rather than to a listing that happens to be
    small enough.
    """
    server = _live_relay(root, monkeypatch)
    try:
        _seed_stored_sessions(root, SESSIONS_OVER_THE_CAP)
        record = store.find_own_relay(root)
        assert record is not None
        assert len(_read_reply_raw(record, "peer_session_rows")) > wire.MAX_HANDSHAKE_LINE

        refusal = _diagnose_a_silent_socket_with_the_pre_fix_reader(record, "peer_session_rows")
        assert str(wire.MAX_HANDSHAKE_LINE) in refusal, refusal
    finally:
        server.stop()


# ---------------------------------------------------------------------------
# 2. The control reader's own bound, at its edge
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("over", [False, True])
def test_the_control_reader_bounds_a_complete_line_at_its_edge(socketpair: Any, over: bool) -> None:
    """One byte under the bound parses; one byte over is refused WITH ITS SIZE.

    The line is delivered in one write, so the whole line is in hand when the bound
    is applied and the size in the exception is the line's length rather than a
    lower bound — which is the number an operator needs to know what was refused.
    """
    client, server = socketpair
    limit = 64
    line = _line_of(limit + 1 + (1 if over else 0))  # the reader measures the line WITHOUT its \n
    server.sendall(line)
    reader = session_dial.LineReader(client, limit, report_bad_frames=True)

    if not over:
        frame = reader.read_frame(5.0)
        assert frame is not None and frame["op"] == "ack"
        return
    with pytest.raises(session_dial.FrameTooLarge) as excinfo:
        reader.read_frame(5.0)
    assert excinfo.value.limit == limit
    assert excinfo.value.size == len(line) - 1


def test_a_line_that_crosses_the_bound_before_its_newline_reports_a_lower_bound(
    socketpair: Any,
) -> None:
    """THE OTHER HALF OF THE SWEEP: a line whose newline is still in flight.

    The reader refuses the moment its buffer passes the bound with no newline in
    sight, and it does NOT keep reading to find out the line's true length — that
    would be the unbounded allocation the bound exists to prevent. So the size it
    reports is a lower bound, and this test pins both ends of it: over the bound,
    and no more than one read past it.
    """
    client, server = socketpair
    limit = 4096
    line = _line_of(limit + 1 + 96 * 1024)
    server.sendall(line)
    reader = session_dial.LineReader(client, limit, report_bad_frames=True)

    with pytest.raises(session_dial.FrameTooLarge) as excinfo:
        reader.read_frame(5.0)
    assert excinfo.value.limit == limit
    assert limit < excinfo.value.size <= limit + (1 << 16)


def test_a_stream_reader_still_drops_an_oversized_line_and_keeps_the_next(
    socketpair: Any,
) -> None:
    """THE PROPERTY THE REPORT MODE MUST NOT COST: a session's later frames survive.

    The default reader is the one a viewer's stream uses, and its contract is the
    opposite of the control client's — one junk frame may not cost the session. So
    the default drops the oversized line and returns the NEXT frame.
    """
    client, server = socketpair
    server.sendall(_line_of(4096 + 1 + 96 * 1024) + wire.encode_line({"op": "welcome"}))

    reader = session_dial.LineReader(client, 4096)  # report_bad_frames off
    frame = reader.read_frame(10.0)
    assert frame is not None and frame["op"] == "welcome"


# ---------------------------------------------------------------------------
# 3. What the client refuses with — by name, and never the wedge
# ---------------------------------------------------------------------------


def test_an_over_bound_control_reply_refuses_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """``frame_too_large``, naming the op, the bound and the byte count."""
    monkeypatch.setattr(session_dial, "MAX_SESSION_FRAME_BYTES", TEST_BOUND)
    stub = _StubControl(_line_of(TEST_BOUND + 2))
    try:
        with pytest.raises(MeshRefusal) as excinfo:
            relay.control_request(_record(stub.port), "peer_session_rows", timeout=5.0)
    finally:
        stub.close()

    refusal = excinfo.value
    assert refusal.code == "frame_too_large", refusal.code
    assert "peer_session_rows" in refusal.sentence
    match = re.search(r"at least (\d+) bytes, over the (\d+)-byte bound", refusal.sentence)
    assert match, refusal.sentence
    size, bound = int(match.group(1)), int(match.group(2))
    assert bound == TEST_BOUND
    # A lower bound on the line: the read that crossed the bound is what is counted.
    assert TEST_BOUND < size <= TEST_BOUND + (1 << 16)
    # THE LIE, REFUSED: this is not a relay that stayed silent, and no caller may
    # render it as one (Q-R8-1's operator-visible half).
    assert "wedged" not in refusal.sentence.lower()
    assert "did not answer" not in refusal.sentence


def test_an_unreadable_control_reply_is_named_rather_than_read_as_a_silent_relay() -> None:
    """A line that is not a JSON object is ``frame_unreadable``, not ``None``."""
    stub = _StubControl(b"this is not a frame\n")
    try:
        with pytest.raises(MeshRefusal) as excinfo:
            relay.control_request(_record(stub.port), "peer_session_rows", timeout=5.0)
    finally:
        stub.close()

    assert excinfo.value.code == "frame_unreadable"
    assert "peer_session_rows" in excinfo.value.sentence
    assert "wedged" not in excinfo.value.sentence.lower()
    assert "did not answer" not in excinfo.value.sentence


def test_a_socket_that_answers_nothing_is_still_no_answer() -> None:
    """``None`` KEEPS its one meaning: no relay answered me.

    The distinction the whole fix rests on. A socket that accepts, takes the op and
    says nothing is the case every caller's sentence is written for, and it must
    stay ``None`` rather than acquiring a named refusal of its own.
    """
    stub = _StubControl(None, hold_s=0.5)
    try:
        assert relay.control_request(_record(stub.port), "peer_session_rows", timeout=0.2) is None
    finally:
        stub.close()


def test_the_ordinary_listing_refuses_with_the_named_code_not_the_wedged_sentence(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """THE OPERATOR'S SURFACE: `lop sessions --all-peers` reports the real incident.

    Driven through ``main_cli.sessions_command`` — the function `lop sessions` runs —
    so the code and the sentence that reach stdout are the ones an operator and a
    ``--json`` consumer actually get. Before the fix this was rc 1 with
    ``relay_unavailable`` and a sentence about a wedged relay.
    """
    monkeypatch.setattr(session_dial, "MAX_SESSION_FRAME_BYTES", TEST_BOUND)
    stub = _StubControl(_line_of(TEST_BOUND + 2))
    monkeypatch.setattr(store, "find_own_relay", lambda root=None: _record(stub.port))
    try:
        rc = main_cli.sessions_command(_sessions_args())
        captured = capsys.readouterr()
    finally:
        stub.close()

    assert rc == 1
    assert '"code": "frame_too_large"' in captured.out
    assert '"code": "relay_unavailable"' not in captured.out
    assert "wedged" not in captured.out.lower()
