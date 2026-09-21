"""Real Session → authenticated HTTP → a disposable console peer.

What this module is for, and its one honest limit. The console's real counterpart
is the desktop app, whose `console_*` methods belong to the UI half of the split
(`local-operator-ui`, PR A) and do not exist yet. So the peer here is a
disposable loopback server that speaks the FROZEN vocabulary from the design's
§10.2 — the same wire, the same 0600 discovery record, the same key header — and
these cells prove the Python half end to end: the gate, the transport, the typed
refusals, the secret path's containment, and the record's permissions.

Every cell that touches a surface therefore runs against the peer rather than the
app. When PR A lands, the cells marked `RE-RUN AGAINST THE APP` in the docstrings
must be re-run against a real app host; the vocabulary is identical, so the
difference is which process answers, not which assertions hold.
"""

from __future__ import annotations

import json
import os
import secrets
import signal
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_operator.browser_bridge.protocol import PROTO_VERSION
from local_operator.harness.types import Message, ModelSpec, TextContent
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.session_lease import acquire_session_lease
from local_operator.tools.builtin import build_console_tool, execute_console
from local_operator.ui_console import state
from local_operator.ui_console.backend import ui_console_advertisable
from local_operator.variables import VariableStore
from tests.e2e.watchdog import bounded

SECRET = "hunter2-three-blind-mice"

#: The names the app's encoder has, with the byte each one sends (design §10.5;
#: `local-operator-ui` `src/main/console/keys.ts`, whose own conformance test pins the
#: table byte for byte against the pane's DOM handler). A MIRROR for this peer, never a
#: second source of truth: it is here so the peer can answer a `console_keys` call the
#: way the app does — the whole list encoded before any of it is written, and a name
#: outside the vocabulary refused with the accepted set — which is what lets the cells
#: below assert that a SPELLING lands on a name this table has, and that the name
#: encodes to the byte the key means.
_PEER_KEY_BYTES: dict[str, bytes] = {
    "enter": b"\r",
    "tab": b"\t",
    "shift+tab": b"\x1b[Z",
    "backspace": b"\x7f",
    "escape": b"\x1b",
    "space": b" ",
    "up": b"\x1b[A",
    "home": b"\x1b[H",
    "pageup": b"\x1b[5~",
    "pagedown": b"\x1b[6~",
    "insert": b"\x1b[2~",
    "delete": b"\x1b[3~",
    "f5": b"\x1b[15~",
    "ctrl+a": b"\x01",
    "ctrl+c": b"\x03",
    "ctrl+d": b"\x04",
    "ctrl+z": b"\x1a",
    "ctrl+[": b"\x1b",
}

#: One row per spelling a model may write, with the name the encoder has and the byte
#: that name sends. Q-2's whole point: the guide shipped `ctrl-c`/`shift-tab` and the
#: encoder spells them `ctrl+c`/`shift+tab`, so every control key a model read about
#: was refused by the only real host.
_KEY_SPELLINGS: tuple[tuple[str, str, bytes], ...] = (
    ("ctrl+c", "ctrl+c", b"\x03"),
    ("ctrl-c", "ctrl+c", b"\x03"),
    ("CTRL+C", "ctrl+c", b"\x03"),
    ("ctrl_c", "ctrl+c", b"\x03"),
    ("^c", "ctrl+c", b"\x03"),
    ("ctrl+a", "ctrl+a", b"\x01"),
    ("shift+tab", "shift+tab", b"\x1b[Z"),
    ("shift-tab", "shift+tab", b"\x1b[Z"),
    ("escape", "escape", b"\x1b"),
    ("esc", "escape", b"\x1b"),
    ("enter", "enter", b"\r"),
    ("return", "enter", b"\r"),
    ("cr", "enter", b"\r"),
    ("pageup", "pageup", b"\x1b[5~"),
    ("pgup", "pageup", b"\x1b[5~"),
    ("page-up", "pageup", b"\x1b[5~"),
    ("pagedown", "pagedown", b"\x1b[6~"),
    ("pgdn", "pagedown", b"\x1b[6~"),
    ("page-down", "pagedown", b"\x1b[6~"),
    ("insert", "insert", b"\x1b[2~"),
    ("ins", "insert", b"\x1b[2~"),
    ("delete", "delete", b"\x1b[3~"),
    ("del", "delete", b"\x1b[3~"),
    ("backspace", "backspace", b"\x7f"),
    ("back-space", "backspace", b"\x7f"),
    ("up", "up", b"\x1b[A"),
    ("f5", "f5", b"\x1b[15~"),
)


class _Rpc(BaseHTTPRequestHandler):
    """The app's RPC leg, with the four safety rules a peer must keep too."""

    key: str = ""
    #: method -> result payload
    replies: dict[str, dict[str, Any]] = {}
    #: method -> (code, message, data); takes precedence over ``replies``
    refusals: dict[str, tuple[str, str, dict[str, Any]]] = {}
    received: list[tuple[str, dict[str, Any]]] = []
    #: the bytes each accepted `console_keys` call encoded, in call order
    encoded: list[list[bytes]] = []

    def do_POST(self) -> None:  # noqa: N802 - http.server's spelling
        # The body is read BEFORE the key is checked, which is what the app's own
        # listener does (it parses the frame, then authorizes the request) and also
        # what keeps this peer testable: a 401 returned with an unconsumed request
        # body resets the connection, so a client sees a transport error instead of
        # the status the cell is asserting.
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        if self.headers.get("X-Bridge-Key") != type(self).key:
            self.send_response(401)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        request = json.loads(raw or b"{}")
        method = str(request.get("method"))
        params = request.get("params") or {}
        type(self).received.append((method, params))
        if method in type(self).refusals:
            code, message, data = type(self).refusals[method]
            body: dict[str, Any] = {
                "id": request.get("id"),
                "ok": False,
                "error": {"code": code, "message": message, "data": data},
            }
        elif method == "console_keys":
            # The app's own encoder path in miniature (`keys.ts`, §10.5): the WHOLE
            # list is encoded before any of it is written, and a name outside the
            # vocabulary is a typed `unknown_key` carrying the accepted set. Both
            # halves matter to the cells below, because what they test is which NAME
            # the session put on the wire: a fold onto a name this table does not
            # have has to fail here rather than pass quietly.
            names = [str(name) for name in (params.get("keys") or [])]
            unknown = next((name for name in names if name not in _PEER_KEY_BYTES), None)
            if unknown is not None:
                body = {
                    "id": request.get("id"),
                    "ok": False,
                    "error": {
                        "code": "unknown_key",
                        "message": f"Unknown key name: {unknown}.",
                        "data": {"accepted": sorted(_PEER_KEY_BYTES), "key": unknown},
                    },
                }
            else:
                type(self).encoded.append([_PEER_KEY_BYTES[name] for name in names])
                body = {
                    "id": request.get("id"),
                    "ok": True,
                    "result": {"accepted": True, "encoded": names},
                }
        else:
            body = {
                "id": request.get("id"),
                "ok": True,
                "result": type(self).replies.get(method, {}),
            }
        payload = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802 - http.server's spelling
        payload = json.dumps(
            {"host": "ui", "pid": os.getpid(), "console": True, "proto": PROTO_VERSION}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - the base name
        # Overridden only to keep a test peer quiet on stderr. The parameter is
        # named `format` because the base class names it that way and pyright
        # checks override compatibility; nothing here formats anything.
        return


class Peer:
    """A disposable console host: record on disk, listener on loopback."""

    def __init__(self) -> None:
        self.key = secrets.token_urlsafe(32)
        handler = type(
            "_PeerHandler",
            (_Rpc,),
            {"key": self.key, "replies": {}, "refusals": {}, "received": [], "encoded": []},
        )
        self.handler = handler
        self.server = HTTPServer(("127.0.0.1", 0), handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    # -- the record the app would publish ---------------------------------
    @property
    def received(self) -> list[tuple[str, dict[str, Any]]]:
        return self.handler.received

    @property
    def encoded(self) -> list[list[bytes]]:
        """The bytes each accepted `console_keys` call put on the pty, in order."""
        return self.handler.encoded

    def serve(self, **replies: dict[str, Any]) -> None:
        self.handler.replies.update(replies)

    def refuse(self, method: str, code: str, message: str = "", data: dict[str, Any] | None = None):
        self.handler.refusals[method] = (code, message, data or {})

    def publish(self, root: Path | None = None, **updates: Any) -> state.ConsoleHostState:
        values: dict[str, Any] = {
            "pid": os.getpid(),
            "port": self.port,
            "session_key": self.key,
            "proto": PROTO_VERSION,
            "app_version": "0.26.0",
            "console": True,
            "console_surfaces": 1,
        }
        values.update(updates)
        record = state.ConsoleHostState.model_validate(values)
        state.publish(record, root)
        return record

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def bound_console_run() -> Iterator[None]:
    with bounded(60, "console RPC fixture"):
        yield


@pytest.fixture
def peer(bound_console_run: None, monkeypatch: pytest.MonkeyPatch) -> Iterator[Peer]:
    """One disposable peer per test, reachable through the REAL discovery record.

    The CMUX_* strip is the rule anything that drives a session inherits: an
    inherited ``CMUX_WORKSPACE_ID`` has already let a test run rename the
    operator's real cmux workspaces, so the variables are removed here even though
    this module starts no TUI and forks nothing.
    """
    for key in tuple(os.environ):
        if key.startswith(("CMUX_", "LOP_")):
            monkeypatch.delenv(key, raising=False)
    instance = Peer()
    try:
        yield instance
    finally:
        instance.stop()


@pytest.fixture
def broker_reaped() -> Iterator[None]:
    """Stop the broker this test's retrieval starts, and wait for it to go.

    `retrieve_secret` deliberately starts a DETACHED broker (it must outlive its
    starter, and it is what makes the §6 redaction notice reach the owning
    session), and it reaps itself on an idle window measured in minutes. A test
    suite that left one behind per run would put a process — and a socket
    directory under `/tmp` keyed by uid and store digest — on a machine already
    running dozens of sessions, so the process is stopped by pid here. That is the
    same "reclaim what you start" discipline the repository asks of every rig, and
    it also keeps a stale broker from ever answering a later run's retrieval (a
    broker that answers "no such secret" is BELIEVED by `retrieve_secret`, which is
    exactly right in production and surprising in a test).
    """
    yield
    from local_operator.secrets import client

    status = client.broker_status()
    pid = (status or {}).get("pid")
    if not isinstance(pid, int):
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except OSError:
            return
        time.sleep(0.1)


def _session(root: Path) -> Session:
    """A session built the way the PRODUCT builds one.

    The two details that matter here: a real `VariableStore` (the product's
    `_build_variable_store` gives every session one, and it is the redaction sink
    the secret path needs — a session without one refuses a `secret_ref` rather
    than typing an uncontained value into a pty), and a session id, which is what
    a surface is keyed by.
    """
    directory = root / "sessions" / "synthetic-console"
    lease = acquire_session_lease(directory)

    def stream(_request: Any, _signal: Any) -> Any:
        async def events() -> Any:
            if False:
                yield None

        return events()

    session = Session(
        model=ModelSpec(provider="test", model_id="synthetic", context_window=1000),
        stream_fn=stream,
        tools=[],
        transcript=Transcript(directory),
        variables=VariableStore(cwd=str(root)),
        system_blocks_provider=lambda: [],
    )
    session.add_dispose_hook(lease.release)
    return session


@pytest.mark.asyncio
async def test_the_record_is_what_opens_and_closes_the_tool(peer: Peer) -> None:
    """The gate, end to end, off a REAL file through the REAL predicate.

    Three states, and the two that must NOT advertise matter as much as the one
    that must: no record (no app), a record whose console bit is off (a running app
    with the feature disabled — the state a bare liveness check would get wrong),
    and a record that says both. The record is written where the builder reads it,
    which is the default discovery path rather than a root the test picked: the
    tool takes no root argument, and that is the property being asserted.
    """
    assert build_console_tool(None) is None  # nothing published anywhere

    peer.publish(console=False)
    assert ui_console_advertisable() is False
    assert build_console_tool(None) is None

    peer.publish()
    assert ui_console_advertisable() is True
    assert build_console_tool(None) is not None

    path = state.state_path()
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700


@pytest.mark.asyncio
async def test_create_read_and_close_over_the_real_wire(peer: Peer, tmp_path: Path) -> None:
    """RE-RUN AGAINST THE APP: this is the whole round trip.

    The peer answers the frozen vocabulary; the assertions are about what the tool
    SENT (method, params, session) and what the model READ back.
    """
    peer.publish()
    peer.serve(
        **{
            "console_create": {
                "surface": "con:1:9f2a",
                "cols": 100,
                "rows": 30,
                "pid": 4242,
                "live": True,
                "revealed": False,
            },
            "console_read": {
                "text": "$ echo hi\nhi\n$ ",
                "cols": 100,
                "rows": 30,
                # The FROZEN shape, and the only one a host may emit: §5.4's
                # emulator cursor, `x` the column and `y` the row (§10.2). This cell
                # sent the legacy `{row, col}` spelling, which the renderer accepts
                # defensively — so the shape the contract actually freezes was never
                # exercised on the wire, and the literal `row None, column None` that
                # spelling used to produce for the frozen one was invisible here.
                "cursor": {"x": 2, "y": 2},
                "truncated": False,
                "live": True,
                "mode": "viewport",
            },
            "console_close": {"closed": True, "exit_code": 0},
        }
    )
    session = _session(tmp_path)
    try:
        context = session._build_tool_context()
        created = await execute_console(
            "c-create", {"method": "create", "command": "zsh"}, None, None, context
        )
        assert created.is_error is False, created.text
        assert "con:1:9f2a" in created.text

        read = await execute_console(
            "c-read", {"method": "read", "surface": "con:1:9f2a"}, None, None, context
        )
        assert read.is_error is False, read.text
        assert "echo hi" in read.text
        assert "cursor row 2, column 2" in read.text, read.text
        assert "None" not in read.text, read.text

        closed = await execute_console(
            "c-close", {"method": "close", "surface": "con:1:9f2a"}, None, None, context
        )
        assert closed.is_error is False, closed.text

        methods = [method for method, _ in peer.received]
        assert methods == ["console_create", "console_read", "console_close"]
        # The session's OWN id rides every call: a surface belongs to the session
        # that created it, and the tool never lets the model choose whose.
        assert {params["session_id"] for _, params in peer.received if "session_id" in params} == {
            session.session_id
        }
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_request_without_the_key_is_refused(peer: Peer, tmp_path: Path) -> None:
    """The 401 cell, shared with the browser's own disposable-peer suite.

    The four safety rules are the host's, not the console's, and this is the cell
    that proves the console did not accidentally add a second, unauthenticated
    path to the same listener.
    """
    peer.publish()
    denied = httpx.post(f"http://127.0.0.1:{peer.port}/rpc", json={})
    assert denied.status_code == 401
    wrong = httpx.post(
        f"http://127.0.0.1:{peer.port}/rpc",
        headers={"X-Bridge-Key": "not-the-key"},
        json={"id": "r-1", "method": "console_list", "params": {}},
    )
    assert wrong.status_code == 401
    # And the real client, holding the record's key, is not refused.
    ok = httpx.post(
        f"http://127.0.0.1:{peer.port}/rpc",
        headers={"X-Bridge-Key": peer.key},
        json={"id": "r-2", "method": "console_list", "params": {}},
    )
    assert ok.status_code == 200


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("unsupported_method", "This app version has no console"),
        ("surface_unavailable", "No console surface named"),
        ("surface_not_owned", "belongs to another session"),
        ("process_exited", "has exited"),
        ("input_queue_full", "not draining it"),
        ("unknown_key", "Unknown key name"),
        ("secure_input_active", "refuses to read"),
        ("invalid_grid", "outside what the app will honour"),
        ("console_capture_full", "one at a time"),
        # The app's own addition to §10.6 (Q-1), over the real wire: it must validate
        # as a MODELLED code, so the forward-compat hook never fires and the model
        # reads the condition rather than "the app is newer than this session".
        ("capture_unavailable", "no pane is displaying it"),
        ("console_unavailable", "console feature as unavailable"),
        # The one code the first round left out of this list, which is how the copy
        # that interpolated the host's own sentence survived a green suite: a test
        # that covers ten of eleven codes cannot see the eleventh's defect. It is
        # asserted like its siblings now — harness copy, both numbers from `data`,
        # and the host's message nowhere in the result.
        ("proto_mismatch", "speaks protocol 7"),
    ],
)
@pytest.mark.asyncio
async def test_every_typed_refusal_of_the_vocabulary_reaches_the_model(
    peer: Peer, tmp_path: Path, code: str, expected: str
) -> None:
    """The §10.6 taxonomy, refused over the real wire and rendered from the CODE.

    The peer deliberately sends a host message that must NOT be what the model
    reads: the copy is the harness's, so a reworded app cannot change a session's
    behaviour, and a refusal the model can act on is the difference between a
    retry loop and a diagnosis.
    """
    peer.publish()
    peer.refuse(
        "console_status",
        code,
        "a host sentence the model must not be shown",
        {
            "surface": "con:1:a",
            "handle": "con:1:a",
            "exit_code": 3,
            "count": 2,
            "key": "ctrl-pageup",
            # The peer's own revision, deliberately not this session's: the copy
            # must name BOTH numbers, which it can only do from `data`.
            "proto": 7,
        },
    )
    session = _session(tmp_path)
    try:
        result = await execute_console(
            "c-status",
            {"method": "status", "surface": "con:1:a"},
            None,
            None,
            session._build_tool_context(),
        )
    finally:
        await session.dispose()
    assert result.is_error is True
    assert expected in result.text, result.text
    assert "a host sentence the model must not be shown" not in result.text


@pytest.mark.asyncio
async def test_the_secret_ref_cell_against_the_real_store(
    peer: Peer, tmp_path: Path, broker_reaped: None
) -> None:
    """RE-RUN AGAINST THE APP: the containment claim, asserted byte for byte.

    The value is stored in the REAL encrypted store (this test's HOME is a scratch
    dir, so nothing near the operator's own store is touched), typed into the pty
    through `secret_ref`, and then looked for everywhere a model or a reader could
    see it: the tool result, the arguments the model emitted, the transcript file
    on disk, and the session's own redaction seam. It IS in the peer's received
    params — the accepted v1 disclosure, stated in §11.3 rather than glossed — and
    that is asserted here too so the claim cannot quietly drift into a stronger
    one.
    """
    from local_operator.secrets.access import open_store

    peer.publish()
    peer.serve(**{"console_input": {"accepted": True, "bytes": len(SECRET)}})
    store = open_store(create=True)
    store.set("SUDO_PASSWORD", SECRET.encode(), description="", session_id=None)

    session = _session(tmp_path)
    try:
        args = {"method": "input", "surface": "con:1:a", "secret_ref": "SUDO_PASSWORD"}
        result = await execute_console("c-input", args, None, None, session._build_tool_context())
        assert result.is_error is False, result.text

        # The app receives the value: the one accepted disclosure (§11.3).
        assert peer.received == [("console_input", {"surface": "con:1:a", "text": SECRET})]

        # The model's result does not, and neither do the arguments it emitted.
        assert SECRET not in result.text
        assert SECRET not in repr(args)
        assert "SUDO_PASSWORD" in result.text

        # The session's own redaction seam — the exact function `session.py` hands
        # the loop as `redact_tool_result` — masks it, so an echoed value cannot
        # survive into a later result.
        redactor = session._redact_tool_result_text
        assert SECRET not in redactor(f"the app printed {SECRET} on its own")
        assert "[redacted]" in redactor(f"the app printed {SECRET} on its own")

        # And the TRANSCRIPT on disk, written through the real writer, holds no
        # trace of it: grep the bytes rather than trusting the seam's return value.
        transcript = session.transcript
        visible = redactor(result.text)
        await transcript.append_message(
            Message(role="user", content=[TextContent(text=f"console said: {visible}")])
        )
        assert SECRET.encode() not in transcript.path.read_bytes()
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_surface_the_user_opened_is_readable_and_named_as_theirs(
    peer: Peer, tmp_path: Path
) -> None:
    """R18's flow: the user says they ran something in the console.

    `list` must show the surface with `origin: user` and its command and cwd, so
    the agent reads THAT surface instead of asking the user to repeat their
    output — and the handle's `con:` prefix is what lets it be sure this is the
    Local Operator console rather than another terminal on the machine.
    """
    peer.publish()
    peer.serve(
        **{
            "console_list": {
                "surfaces": [
                    {
                        "surface": "con:4:beef",
                        "session_id": "the-users-session",
                        "origin": "user",
                        "command": "pytest",
                        "argv_tail": ["-x", "tests/e2e"],
                        "cwd": "/Users/someone/project",
                        "cols": 120,
                        "rows": 40,
                        "running": True,
                        "live": True,
                        "last_activity": "2s ago",
                    }
                ]
            }
        }
    )
    session = _session(tmp_path)
    try:
        result = await execute_console(
            "c-list", {"method": "list"}, None, None, session._build_tool_context()
        )
    finally:
        await session.dispose()
    assert result.is_error is False, result.text
    assert "con:4:beef" in result.text
    assert "user" in result.text
    assert "pytest -x tests/e2e" in result.text
    # N-4: §6.5/§13.4 put the owning session IN the listing, and this id is
    # deliberately not this session's — the field is what lets an agent check the
    # app's session filtering instead of taking it on trust, so the row that
    # demonstrates "a surface the user opened" is the row that has to carry it.
    assert "session the-users-session" in result.text
    assert "/Users/someone/project" in result.text


@pytest.mark.parametrize(
    ("spelling", "canonical", "byte"),
    _KEY_SPELLINGS,
    ids=[row[0] for row in _KEY_SPELLINGS],
)
@pytest.mark.asyncio
async def test_every_key_spelling_is_one_name_and_one_byte(
    peer: Peer, tmp_path: Path, spelling: str, canonical: str, byte: bytes
) -> None:
    """RE-RUN AGAINST THE APP (Q-2): one spelling per cell, as a real key press.

    Three facts per row, and all three are the defect Q-2 found: the session folds
    the SPELLING onto the name the encoder has (asserted on the wire, in full —
    surface and keys, so nothing else rides along), the name is one the encoder
    actually has (the peer refuses anything else with `unknown_key`, exactly as the
    app does, so a fold onto an invented name fails HERE rather than silently), and
    that name encodes to the byte the key means (`\x03` for `ctrl+c`, `\x1b[Z` for
    `shift+tab`) — which is the thing the model was failing to reach when the guide
    told it to write `ctrl-c`.

    The byte is asserted against a literal, not against the mirror table's own
    lookup: the table is what the peer encodes WITH, so asserting through it would
    only prove the dict is consistent with itself.
    """
    peer.publish()
    session = _session(tmp_path)
    try:
        result = await execute_console(
            "c-keys",
            {"method": "keys", "surface": "con:1:a", "keys": [spelling]},
            None,
            None,
            session._build_tool_context(),
        )
    finally:
        await session.dispose()

    assert result.is_error is False, (spelling, result.text)
    assert peer.received[-1] == (
        "console_keys",
        {"surface": "con:1:a", "keys": [canonical]},
    ), peer.received[-1]
    assert peer.encoded == [[byte]], (spelling, peer.encoded)


@pytest.mark.asyncio
async def test_a_name_the_encoder_does_not_have_is_refused_with_its_own_words(
    peer: Peer, tmp_path: Path
) -> None:
    """Q-2's other half: the fold accepts SPELLINGS, it never invents names.

    `meta+c` is not in the encoder's vocabulary — there is no modifier-chord
    vocabulary in it (design §10.5 says so now) — and the app answers with the
    accepted set in `data.accepted`. A session-side grammar that guessed would
    replace that authoritative list with a staler one, which is why an unrecognised
    name reaches the wire exactly as it arrived.
    """
    peer.publish()
    session = _session(tmp_path)
    try:
        result = await execute_console(
            "c-keys",
            {"method": "keys", "surface": "con:1:a", "keys": ["meta+c"]},
            None,
            None,
            session._build_tool_context(),
        )
    finally:
        await session.dispose()

    assert result.is_error is True
    assert peer.received[-1][1] == {"surface": "con:1:a", "keys": ["meta+c"]}
    assert "Unknown key name: meta+c" in result.text, result.text
    assert "ctrl+c" in result.text, result.text
    # Nothing was written: the app encodes the whole list before any of it, and this
    # cell is the one that can see the peer keep that rule.
    assert peer.encoded == []


@pytest.mark.asyncio
async def test_the_console_off_refusal_names_the_reason_the_app_sent(
    peer: Peer, tmp_path: Path
) -> None:
    """RE-RUN AGAINST THE APP (Q-3): the app's own payload is `{"reason": "disabled"}`.

    Launched with `LOCAL_OPERATOR_UI_CONSOLE_HOST=0`, the real app answers a console
    method with exactly this body. The gate normally stops the call first — a record
    saying `console: false` is what keeps the tool out of the inventory (§15) — so
    this cell drives the case the record did not show: a caller that got through
    anyway, which §15 keeps as the second line of defence. The copy must name the
    condition the app named, not disjoin §10.1's three.
    """
    peer.publish()
    peer.refuse(
        "console_status",
        "console_unavailable",
        "this app's console is not available (disabled): LOCAL_OPERATOR_UI_CONSOLE_HOST is off",
        {"reason": "disabled"},
    )
    session = _session(tmp_path)
    try:
        result = await execute_console(
            "c-status",
            {"method": "status", "surface": "con:1:a"},
            None,
            None,
            session._build_tool_context(),
        )
    finally:
        await session.dispose()

    assert result.is_error is True
    assert "LOCAL_OPERATOR_UI_CONSOLE_HOST" in result.text, result.text
    assert "this app's console is not available (disabled)" not in result.text, result.text
    assert "settings toggle" not in result.text, result.text


@pytest.mark.asyncio
async def test_an_app_that_quits_mid_call_gets_the_honest_answer_and_never_hangs(
    peer: Peer, tmp_path: Path
) -> None:
    """§15's last row: the surfaces ended with the app.

    The peer is stopped after its record is published, which is what a quit looks
    like from the session's side — the record outlives the process for exactly as
    long as it takes the heartbeat to age out. The answer names the app rather than
    reporting a hanging call, and it arrives without touching the socket the server
    used to own.
    """
    peer.publish()
    peer.stop()
    session = _session(tmp_path)
    try:
        result = await execute_console(
            "c-list", {"method": "list"}, None, None, session._build_tool_context()
        )
    finally:
        await session.dispose()
    assert result.is_error is True
    assert "browser" not in result.text.lower()
    assert ("not running" in result.text) or ("no longer answering" in result.text)
