"""The control plane's authority seam (issue #1310).

The invariant, on the real socket:

    a constrained subject must not be able to mint the authority
    that removes its own approval requirement

A running session publishes ``control_key`` mode 0600 in its discovery record.
Any same-uid process may read it — including a model-authored ``bash`` call,
which runs as exactly that uid — so the record key alone cannot authorise
``/approvals auto`` or an approved card. Those two are
:func:`local_operator.harness.approval.transition_authority`'s
authority-INCREASING class, and they additionally demand the per-session
operator capability, which exists only in the memory of the process that started
the runtime and of the console that typed the command.

This module is deliberately NOT the place for the predicate's own unit tests
(``tests/unit/harness/test_approval_authority.py``) — everything here drives the
REAL ``ServingSessionHandle`` behind a REAL ``RuntimeServer`` on a REAL loopback
socket, because the defect is a wire defect and the acceptance criteria are
about what a same-uid process can actually reach.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import inspect
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

import pytest

from local_operator.config import ConfigManager
from local_operator.harness.approval import (
    AUTHORITY_OPS,
    OPERATOR_AUTHORITY_REQUIRED_NOTICE,
    handshake_proof_ok,
    is_wire_hex,
    mint_operator_cap,
    operator_cap_for,
    operator_nonce,
    remember_operator_cap,
    request_proof,
    reset_operator_caps_for_tests,
)
from local_operator.operator.verify import key_id_for
from local_operator.paths import config_dir
from local_operator.session.runtime import launch as launch_module
from local_operator.session.runtime import registry
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.runtime.serving import ServingSessionHandle
from tests.unit.session.runtime.test_serving import FakeSession

_TESTS_ROOT = Path(__file__).resolve().parents[4]

#: The sentence's opening, matched as a substring so the assertion is about the
#: refusal rather than about the exact wrapping of one constant.
_REFUSAL = "this session's gate is still at ask"


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class _Live:
    """One real runtime serving a real owned-session handle."""

    def __init__(self, handle: ServingSessionHandle, runtime: RuntimeServer, record: Any) -> None:
        self.handle = handle
        self.runtime = runtime
        self.record = record

    async def close(self, scratch: Path) -> None:
        # Order matters: the runtime owns the record, and the handle's own
        # dispose unsubscribes the config watcher next to it.
        self.runtime.close()
        dispose = getattr(self.handle, "dispose", None)
        if callable(dispose):
            outcome = dispose()
            if inspect.isawaitable(outcome):
                await outcome
        registry.scan(scratch)


async def _serve(
    tmp_path: Path, *, operator_cap: bytes | None, operator_anchor: Any | None = None
) -> _Live:
    """Start a real ``RuntimeServer`` over a real ``ServingSessionHandle``.

    The handle is the PRODUCTION sink (``_approvals_slash`` writing
    ``_auto_approve``, ``_approval_answer`` resolving the parked future), not a
    double: the assertions below are about the gate's real value, and a double
    would let the seam be correct while the sink moved.
    """
    root = config_dir()
    ConfigManager(root).set_config_value("tool_approval_mode", "ask")
    session = _AttachableSession()
    handle = ServingSessionHandle(
        session, asyncio.get_running_loop(), cwd=str(tmp_path), auto_approve=False
    )
    runtime = RuntimeServer(
        handle, kind="tui", operator_cap=operator_cap, operator_anchor=operator_anchor
    )
    # ``start_in_process``, not ``start``: the handle validates that it is used
    # from the loop it was built on (``_check_loop_thread``), and a thread-hosted
    # runtime would dispatch on its own loop and be refused by the handle itself.
    await runtime.start_in_process()
    for _ in range(250):
        found = registry.scan(root)
        if found and found[0][1] == "live":
            return _Live(handle, runtime, found[0][0])
        await asyncio.sleep(0.02)
    runtime.close()
    raise AssertionError("the runtime never published a live record")


class _Conn:
    """One authenticated connection, plus the handshake material it settled on.

    ``nonce`` is what this client offered and ``salt`` is what the runtime
    answered with; together they are what a proof is bound to, so a test that
    wants to present one needs both. ``salt`` is EMPTY when no handshake
    completed — which is the state every non-console connection is in, and the
    state the refusal tests drive.
    """

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.reader = reader
        self.writer = writer
        self.nonce = ""
        self.salt = ""

    def proof(self, cap: bytes) -> str:
        """This connection's proof for an authority-increasing frame."""
        assert self.nonce and self.salt, "no handshake completed on this connection"
        return request_proof(cap, client_nonce=self.nonce, server_salt=self.salt)

    def close(self) -> None:
        self.writer.close()


async def _dial(record: Any, *, cap: bytes | None = None, **fields: Any) -> _Conn:
    """One authenticated connection; the welcome frame is consumed and verified.

    ``cap`` is what THIS process holds for the runtime behind ``record``: given
    it, the dial offers a nonce and VERIFIES the runtime's proof — the same two
    steps ``AttachClient.connect`` takes — so a test that then presents a proof
    is driving the production handshake rather than a shortcut around it.
    """
    reader, writer = await asyncio.open_connection("127.0.0.1", record.control_port, limit=1 << 20)
    conn = _Conn(reader, writer)
    auth: dict[str, Any] = {"key": record.control_key, **fields}
    if cap is not None:
        conn.nonce = operator_nonce()
        auth["operator_nonce"] = conn.nonce
    writer.write(json.dumps(auth).encode() + b"\n")
    await writer.drain()
    welcome = json.loads(await asyncio.wait_for(reader.readline(), timeout=10))
    assert welcome["op"] == "projection", welcome
    if cap is not None:
        conn.salt = str(welcome.get("operator_salt") or "")
        assert handshake_proof_ok(
            supplied=welcome.get("operator_proof"),
            held=cap,
            client_nonce=conn.nonce,
            server_salt=conn.salt,
        ), "the runtime did not prove it holds the capability we dialled for"
        assert cap.hex() not in json.dumps(welcome)
    return conn


async def _send(
    reader: asyncio.StreamReader | _Conn, writer: asyncio.StreamWriter | None, frame: dict[str, Any]
) -> dict[str, Any]:
    """Send one request frame and return its reply.

    Accepts a ``_Conn`` for the new call shape and the old ``(reader, writer)``
    pair, so the tests that never hand-shook (every refusal control) read the
    same as they did.
    """
    if isinstance(reader, _Conn):
        conn = reader
        reader, writer = conn.reader, conn.writer
    assert writer is not None
    writer.write(json.dumps({"req": 1, **frame}).encode() + b"\n")
    await writer.drain()
    for _ in range(40):
        raw = await asyncio.wait_for(reader.readline(), timeout=10)
        if not raw.strip():
            continue
        reply = json.loads(raw)
        if reply.get("req") == 1:
            return reply
    raise AssertionError("no reply to the request")


async def _park_a_card(handle: ServingSessionHandle) -> asyncio.Task[Any]:
    """Park one approval card and return the task waiting on it."""
    task = asyncio.ensure_future(handle._approval_gate("bash", "rm -rf build/"))
    await asyncio.sleep(0)
    assert handle._fold.projection.pending is not None, "the fixture did not park a card"
    return task


# ---------------------------------------------------------------------------
# Negative controls
# ---------------------------------------------------------------------------

#: The same-uid attempt, as its own PROGRAM rather than ``python -c``: the child
#: runs with the argv a model's ``bash`` call would have, imports nothing from
#: this repository, and uses only the two things such a process can reach — the
#: discovery record under ``<config>/run/mobile/`` and the loopback port it
#: names. The op is ``slash_result`` because that is the op that reaches the
#: sink (a bare ``slash`` answers "terminal-only here"); the second request is
#: the card half, with the ``request_id`` read out of the welcome snapshot this
#: connection was just sent.
#:
#: It never writes the capability anywhere and never carries one, which is the
#: point: it cannot, because none was ever written down.
_ATTEMPT_PROGRAM = r"""
import glob, json, os, socket, sys

for path in sorted(glob.glob(os.path.join(sys.argv[1], "run", "mobile", "*.json"))):
    record = json.loads(open(path).read())
    sock = socket.create_connection(("127.0.0.1", record["control_port"]), timeout=10)
    stream = sock.makefile("rwb")
    stream.write(json.dumps({"key": record["control_key"], "client": "daemon"}).encode() + b"\n")
    stream.flush()
    welcome = json.loads(stream.readline())
    print("welcome", welcome["op"])
    pending = welcome.get("data", {}).get("pending") or {}
    print("saw-request", bool(pending.get("request_id")))
    for frame in (
        {"op": "slash_result", "req": 1, "command": "approvals", "args": "auto", "images": []},
        {
            "op": "approval_answer",
            "req": 2,
            "request_id": pending.get("request_id", ""),
            "approved": True,
        },
    ):
        stream.write(json.dumps(frame).encode() + b"\n")
        stream.flush()
        for _ in range(20):
            raw = stream.readline()
            if not raw.strip():
                continue
            reply = json.loads(raw)
            if reply.get("req") == frame["req"]:
                print(frame["op"], reply.get("op"), reply.get("message") or "")
                break
    sock.close()
"""


@pytest.mark.asyncio
@pytest.mark.slow
async def test_a_model_authored_subprocess_cannot_loosen_the_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE ISSUE'S REGRESSION SHAPE, driven from a real same-uid process.

    The runtime HOLDS a capability here — it was started by a console — and the
    subprocess has none, because nothing about the capability is readable: not
    the record it just read, not the environment, not argv. That is the whole
    claim, so the control is only meaningful when there IS a value to fail to
    guess.

    Asserted: the refusal reply, ``_auto_approve`` still False, and the next
    gated call still PARKING (then denied) — the last one because "the flag says
    ask" and "the gate really is armed" are different claims, and only the
    parked card distinguishes them.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    cap = mint_operator_cap()
    live = await _serve(tmp_path, operator_cap=cap)
    try:
        # The card is parked BEFORE the attempt connects, so the child can read
        # its id out of the welcome snapshot it is sent — which is what makes
        # the card half of the control real rather than a guessed id.
        parked = await _park_a_card(live.handle)
        pending = live.handle._fold.projection.pending
        assert pending is not None
        request_id = pending.request_id

        script = tmp_path / "attempt.py"
        script.write_text(_ATTEMPT_PROGRAM, encoding="utf-8")
        attempt = await asyncio.to_thread(
            subprocess.run,
            [sys.executable, str(script), str(tmp_path)],
            capture_output=True,
            check=False,
        )
        out = attempt.stdout.decode()
        assert attempt.returncode == 0, out + attempt.stderr.decode()
        # The child really did reach this session and really did see the card:
        # without these two lines a refusal below would be satisfied by a child
        # that never connected at all.
        assert "welcome projection" in out, out
        assert "saw-request True" in out, out
        # Both increasing requests refused, each with the remedy named — not an
        # EOF, not a crash, and not a silent no-op. The two refusals are NOT the
        # same sentence and that is deliberate: the command's copy names the
        # remedies for a gate, and the CARD's tells the reader the question is
        # still parked and that a deny works from where they are (UX round 2,
        # U8 — a phone user was handed advice they could not take).
        from local_operator.harness.approval import CARD_APPROVAL_REFUSED_NOTICE

        assert out.count(_REFUSAL) == 1, out
        assert out.count(CARD_APPROVAL_REFUSED_NOTICE) == 1, out
        assert "slash_result error" in out, out
        assert "approval_answer error" in out, out

        assert live.handle._auto_approve is False, "a same-uid subprocess loosened the gate"
        # The card is still the human's to answer: the answering half was
        # refused too, which is the "or, worse, answer the parked card" half of
        # the issue.
        still_pending = live.handle._fold.projection.pending
        assert still_pending is not None, "the card was answered anyway"
        assert still_pending.request_id == request_id
        # ...and that card IS the evidence that the gate is still armed: it is a
        # real gated call that PARKS because the refusal left `ask` in force.
        # Denying it resolves the parked call, which is the human's own answer
        # and must keep working.
        await live.handle.approval_answer(request_id, False, False)
        assert await parked is False
    finally:
        await live.close(tmp_path)


@pytest.mark.asyncio
async def test_a_phone_shaped_dial_cannot_loosen_the_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The relay's shape WITHOUT the relay's capability, i.e. a forged one.

    ``locality="remote"`` plus a valid record key is everything a remote client
    can present on its own. It must not be enough: the relay is a local process
    that attaches the capability itself when it is the one that started the
    runtime, and a remote frame that carries none is indistinguishable from an
    attempt, so it is refused.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    live = await _serve(tmp_path, operator_cap=mint_operator_cap())
    try:
        conn = await _dial(live.record, client="attach", locality="remote")
        reply = await _send(
            conn,
            None,
            {"op": "slash_result", "command": "approvals", "args": "auto", "images": []},
        )
        assert reply["op"] == "error", reply
        assert _REFUSAL.split(" — ")[0] in reply["message"], reply
        assert live.handle._auto_approve is False
        conn.close()
    finally:
        await live.close(tmp_path)


@pytest.mark.asyncio
async def test_the_desktop_command_op_is_refused_without_the_capability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The desktop route's frame, on a runtime that backend did not start.

    ``POST /v1/desktop/sessions/{id}/commands`` reaches this runtime as an
    attach client and sends exactly this op (``slash_result`` — see
    ``session/attached.py::route_shared_slash``). A desktop backend that did NOT
    spawn the session has no capability for it, which is surface row five of the
    design doc: the app may report and tighten a running gate, never loosen one
    it did not open.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    live = await _serve(tmp_path, operator_cap=mint_operator_cap())
    try:
        conn = await _dial(live.record, client="attach", surface="desktop")
        reply = await _send(
            conn,
            None,
            {"op": "slash_result", "command": "approvals", "args": "auto", "images": []},
        )
        assert reply["op"] == "error", reply
        assert _REFUSAL.split(" — ")[0] in reply["message"], reply
        assert live.handle._auto_approve is False
        conn.close()
    finally:
        await live.close(tmp_path)


@pytest.mark.asyncio
async def test_a_runtime_with_no_capability_refuses_even_a_well_formed_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail-closed for a runtime nobody handed a capability to.

    The state a background spawn with no console produces, and the state a
    rolling upgrade produces when an older console starts a newer runtime. A
    frame that presents a plausible 64-hex string is refused, because there is
    nothing here to match it against — the alternative reading ("no capability,
    so anything passes") would reopen the defect for exactly the spawns nobody is
    watching.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    live = await _serve(tmp_path, operator_cap=None)
    try:
        conn = await _dial(live.record, client="daemon")
        reply = await _send(
            conn,
            None,
            {
                "op": "slash_result",
                "command": "approvals",
                "args": "auto",
                "images": [],
                # A well-formed proof of a capability this runtime does not
                # hold, on a connection that never even offered a nonce: the
                # refusal is about possession, not about shape.
                "operator_cap": request_proof(
                    mint_operator_cap(), client_nonce=operator_nonce(), server_salt=operator_nonce()
                ),
            },
        )
        assert reply["op"] == "error", reply
        assert live.handle._auto_approve is False
        conn.close()
    finally:
        await live.close(tmp_path)


@pytest.mark.asyncio
async def test_an_unapproved_card_answer_is_refused_but_a_deny_is_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The asymmetric half of the class, on the real card.

    ``approved=True`` concedes the parked call, so it is authority-increasing;
    ``approved=False`` settles it the safe way, so every surface that can reach
    the session keeps that route. A change that refused both would have walled
    off the only answer a follower could safely give.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    live = await _serve(tmp_path, operator_cap=mint_operator_cap())
    try:
        parked = await _park_a_card(live.handle)
        pending = live.handle._fold.projection.pending
        assert pending is not None
        request_id = pending.request_id
        conn = await _dial(live.record, client="attach")
        refused = await _send(
            conn,
            None,
            {"op": "approval_answer", "request_id": request_id, "approved": True},
        )
        assert refused["op"] == "error", refused
        assert live.handle._fold.projection.pending is not None, "the card was resolved anyway"

        allowed = await _send(
            conn,
            None,
            {"op": "approval_answer", "request_id": request_id, "approved": False},
        )
        assert allowed["op"] == "ack", allowed
        assert await parked is False
        assert live.handle._fold.projection.pending is None
        conn.close()
    finally:
        await live.close(tmp_path)


@pytest.mark.asyncio
async def test_tightening_still_works_from_a_non_capable_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``auto -> ask`` is not in the class, so it must keep working everywhere.

    The mirror of the negative controls: a phone, a desktop app, a peer or a
    stray terminal may all make the session SAFER. If this ever requires the
    capability, the fix has been widened past the invariant it exists for.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    live = await _serve(tmp_path, operator_cap=mint_operator_cap())
    try:
        live.handle._auto_approve = True
        conn = await _dial(live.record, client="attach", locality="remote")
        reply = await _send(
            conn,
            None,
            {"op": "slash_result", "command": "approvals", "args": "ask", "images": []},
        )
        assert reply["op"] == "result", reply
        assert live.handle._auto_approve is False, "a tightening was refused"
        # ...and the report is ordinary too, for the same reason.
        report = await _send(
            conn, None, {"op": "slash_result", "command": "approvals", "args": "", "images": []}
        )
        assert report["op"] == "result", report
        conn.close()
    finally:
        await live.close(tmp_path)


# ---------------------------------------------------------------------------
# Positive controls (non-vacuous: #1291's lesson)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_console_holding_the_capability_loosens_the_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The POSITIVE CONTROL for every refusal above.

    Without it, "the same-uid subprocess could not loosen the gate" is satisfied
    by a gate frozen for everyone. This is the console that started the runtime
    presenting the capability, and it must move the gate in one step — the
    operator's acceptance criterion (2).
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    cap = mint_operator_cap()
    live = await _serve(tmp_path, operator_cap=cap)
    try:
        # Dialled WITH the capability, so the handshake completes and this
        # connection is what the runtime considers its console.
        conn = await _dial(live.record, client="daemon", cap=cap)
        reply = await _send(
            conn,
            None,
            {
                "op": "slash_result",
                "command": "approvals",
                "args": "auto",
                "images": [],
                "operator_cap": conn.proof(cap),
            },
        )
        assert reply["op"] == "result", reply
        assert live.handle._auto_approve is True, "the owning console could not loosen its own gate"
        # The gate really moved, not merely the flag: the next decision answers
        # inline and parks nothing.
        assert await live.handle._approval_gate("bash", "rm -rf build/") is True
        assert live.handle._fold.projection.pending is None
        conn.close()
    finally:
        await live.close(tmp_path)


@pytest.mark.asyncio
async def test_the_console_holding_the_capability_resolves_the_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other increasing request, from the process that may make it."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    cap = mint_operator_cap()
    live = await _serve(tmp_path, operator_cap=cap)
    try:
        parked = await _park_a_card(live.handle)
        pending = live.handle._fold.projection.pending
        assert pending is not None
        request_id = pending.request_id
        conn = await _dial(live.record, client="attach", cap=cap)
        reply = await _send(
            conn,
            None,
            {
                "op": "approval_answer",
                "request_id": request_id,
                "approved": True,
                "operator_cap": conn.proof(cap),
            },
        )
        assert reply["op"] == "ack", reply
        assert await parked is True
        assert live.handle._fold.projection.pending is None
        conn.close()
    finally:
        await live.close(tmp_path)


@pytest.mark.asyncio
async def test_the_attach_client_presents_the_capability_for_a_runtime_it_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operator_cap: bytes
) -> None:
    """The CLIENT half of the rule, driven through ``AttachClient`` for real.

    Every console route — the TUI's routed slash, the desktop command surface,
    the phone relay — reaches the runtime through ``AttachClient``, so the
    capability has to be attached there, and only for the frames the same
    classification marks decreasing. Both halves are asserted: the frame for a
    runtime THIS process started carries the field, and a frame for a runtime it
    did not carries nothing (so an ordinary command stays wire-identical).
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    from local_operator.mobile.attach_client import AttachClient

    class _Writer:
        def __init__(self) -> None:
            self.sent: list[dict[str, Any]] = []

        def write(self, data: bytes) -> None:
            self.sent.append(json.loads(data))

        async def drain(self) -> None:
            return None

    client = AttachClient(lambda _projection: None, lambda _reason: None)
    client._connected = True
    writer = _Writer()
    client._writer = cast(Any, writer)

    # A runtime THIS process started: ``operator_cap_for(record.pid)`` answers,
    # which is what ``connect`` resolves against — and whose handshake this test
    # stands in for, because the point here is what the PRESENTATION adds to a
    # frame, not how the handshake that gates it is verified (that has its own
    # test, and the real dial in ``_dial`` drives it for every other case).
    client._operator_cap = operator_cap
    client._operator_nonce = operator_nonce()
    client._operator_salt = operator_nonce()
    client._authority_bearing = True
    writer.sent.clear()
    try:
        await asyncio.wait_for(
            client._request_frame(
                "slash_result", command="approvals", args="auto", images=[], deadline_s=0.05
            ),
            timeout=0.2,
        )
    except (TimeoutError, asyncio.TimeoutError):
        pass
    assert writer.sent[-1]["op"] == "slash_result", writer.sent
    assert writer.sent[-1]["operator_cap"] == request_proof(
        operator_cap, client_nonce=client._operator_nonce, server_salt=client._operator_salt
    ), writer.sent[-1]
    # THE SECRET IS NOT ON THE WIRE (agent review round 1, R1-1): the field is a
    # proof over this connection, so a same-uid impostor that rewrote
    # ``control_port`` in the record and read this frame learns nothing it can
    # replay at the real runtime.
    assert operator_cap.hex() not in json.dumps(writer.sent[-1]), writer.sent[-1]

    # The SAME client on a runtime another process started presents nothing.
    writer.sent.clear()
    client._operator_cap = None
    client._authority_bearing = False
    try:
        await asyncio.wait_for(
            client._request_frame(
                "slash_result", command="approvals", args="auto", images=[], deadline_s=0.05
            ),
            timeout=0.2,
        )
    except (TimeoutError, asyncio.TimeoutError):
        pass
    assert "operator_cap" not in writer.sent[-1], writer.sent[-1]

    # ...and an ordinary op never carries it, at all: the field is on exactly the
    # frames the runtime demands it on, which is why the wire needs no version
    # bump and an old runtime keeps serving every other request unchanged.
    writer.sent.clear()
    client._operator_cap = operator_cap
    client._authority_bearing = True
    try:
        await asyncio.wait_for(
            client._request_frame("ping", deadline_s=0.05),
            timeout=0.2,
        )
    except (TimeoutError, asyncio.TimeoutError):
        pass
    assert "operator_cap" not in writer.sent[-1], writer.sent[-1]


# ---------------------------------------------------------------------------
# The class pin: the op set, and the sinks behind it
# ---------------------------------------------------------------------------

#: The sink methods themselves, named as they appear at a call site.
SINK_NAMES = ("_approvals_slash", "_set_approve_all", "_resolve_pending")

#: The handle methods that can reach an authority-increasing sink. Named
#: positively so a new one has to be considered rather than silently scanned.
_SINK_REACHING_METHODS = frozenset(
    {"slash", "slash_images", "run_slash_authoritative", "approval_answer"}
)

#: Modules allowed to call the sinks directly. Every one of them is downstream of
#: the guarded seam (``RuntimeServer._on_request``) or is a host that IS the
#: operator's keyboard: ``serving.py`` holds the runtime's own sinks and
#: ``app.py`` the TUI-hosted ones, and nothing else in the tree may reach them.
_SINK_CALLERS = frozenset(
    {
        "local_operator/session/runtime/serving.py",
        "local_operator/tui/app.py",
    }
)


def _dispatch_ops_reaching_the_sinks(source: str) -> set[str]:
    """Every op in ``server.py``'s dispatch whose body calls a sink-reaching method.

    Read out of the source rather than from a list beside it, because the
    failure this pins is a route ADDED later: a coder who wires a new op to
    ``h.slash`` and forgets the authority class would otherwise ship an
    unguarded way to loosen a running gate with the whole suite green.
    """
    tree = ast.parse(source)
    ops: set[str] = set()
    for node in ast.walk(tree):
        # ``if op == "x":`` (or ``if op in ("x", "y"):``) — the dispatch shape
        # this module uses in both ``_dispatch`` and ``_dispatch_payload``.
        if not isinstance(node, ast.If):
            continue
        test = node.test
        names: set[str] = set()
        if isinstance(test, ast.Compare) and isinstance(test.ops[0], (ast.Eq, ast.In)):
            comparators = test.comparators
            if comparators:
                first = comparators[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    names.add(first.value)
                elif isinstance(first, (ast.Tuple, ast.Set)):
                    names.update(
                        element.value
                        for element in first.elts
                        if isinstance(element, ast.Constant) and isinstance(element.value, str)
                    )
        if not names:
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Attribute)
                and inner.func.attr in _SINK_REACHING_METHODS
            ):
                ops |= names
                continue
            # ``getattr(h, "run_slash_authoritative", None)`` — the optional-
            # capability probe this dispatch uses for the ops an older handle may
            # not implement. Matched by its STRING LITERAL, because that is where
            # the method name is written: a scan that only looked for attribute
            # calls would miss ``slash_result`` entirely, which is the op that
            # actually reaches ``_approvals_slash`` today.
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Name)
                and inner.func.id == "getattr"
                and len(inner.args) >= 2
            ):
                target = inner.args[1]
                if (
                    isinstance(target, ast.Constant)
                    and isinstance(target.value, str)
                    and target.value in _SINK_REACHING_METHODS
                ):
                    ops |= names
    return ops


def test_the_authority_op_set_matches_the_dispatch_source() -> None:
    """The set the seam reads and the routes the dispatch offers must be one set."""
    source = (_TESTS_ROOT / "local_operator/session/runtime/server.py").read_text(encoding="utf-8")
    derived = _dispatch_ops_reaching_the_sinks(source)
    assert derived, "the dispatch shape changed; this pin no longer reads it"
    assert derived == set(AUTHORITY_OPS), (
        f"server.py dispatches {sorted(derived)} to a sink-reaching handle method while the "
        f"seam classifies {sorted(AUTHORITY_OPS)}. A route that reaches "
        "`_approvals_slash`/`_set_approve_all`/a card approval without being in the class "
        "is an unguarded way to loosen a running gate."
    )


def test_only_the_guarded_hosts_call_the_authority_sinks() -> None:
    """The sinks appear in exactly two modules, and no new caller can appear quietly.

    A source scan rather than a behavioural probe, for the reason #1291's
    boundary test gives: a probe can only catch a path someone thought to
    exercise. Every route into the runtime reaches a sink through
    ``RuntimeServer._on_request`` (or through ``TuiSessionHandle`` in the TUI's
    own process), so a THIRD module calling one of these methods directly is
    either a second dispatch route — which the test above would then have to
    classify — or a host that has bypassed the seam entirely.
    """
    callers: dict[str, set[str]] = {}
    for module in sorted((_TESTS_ROOT / "local_operator").rglob("*.py")):
        source = module.read_text(encoding="utf-8")
        if not any(f"{sink}(" in source for sink in SINK_NAMES):
            continue
        callers[module.relative_to(_TESTS_ROOT).as_posix()] = {
            sink for sink in SINK_NAMES if f"{sink}(" in source
        }
    assert set(callers) == _SINK_CALLERS, (
        f"the authority-increasing sinks are referenced in {sorted(callers)} but expected "
        f"{sorted(_SINK_CALLERS)}. A new caller is a new way to loosen a running gate: if it "
        "is a dispatch route, add it to `harness/approval.AUTHORITY_OPS`; if it is a host, "
        "it must reach the runtime through the guarded seam instead."
    )


def test_the_capability_is_never_a_record_field() -> None:
    """Not-at-rest, on the artifact a same-uid process actually reads.

    The defect was a credential published in the discovery record. The
    capability must not become the same thing in a new field, so the record
    dataclass is asserted to have no field that names it — the behavioural half
    (the published JSON) is asserted in
    ``test_a_live_runtime_publishes_no_capability_anywhere``.
    """
    from dataclasses import fields

    from local_operator.session.runtime.types import SessionRecord

    # ``capabilities`` itself is excluded by name: it is the runtime's feature
    # advertisement and always was. Anything else that names a capability or an
    # operator is the field that must not exist.
    names = {field.name for field in fields(SessionRecord)} - {"capabilities"}
    assert not [
        name for name in names if "operator" in name or "capab" in name
    ], f"SessionRecord grew a capability-shaped field: {sorted(names)}"


@pytest.mark.asyncio
async def test_a_live_runtime_publishes_no_capability_anywhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The REAL record file, the REAL welcome frame, and the session's own export.

    Three artifacts a same-uid process can read: the record on disk, the frame
    every client is sent on connect, and the projection the desktop/info surfaces
    render. The capability must appear in none of them — it is held in memory or
    it is not a boundary.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    cap = mint_operator_cap()
    live = await _serve(tmp_path, operator_cap=cap)
    try:
        record_file = tmp_path / "run" / "mobile" / f"{live.record.pid}.json"
        assert record_file.exists(), f"no record at {record_file}"
        assert cap.hex() not in record_file.read_text(encoding="utf-8")

        # The welcome is THE frame every client is sent on connect, so it is
        # dialled by hand here rather than through the helper, which consumes it.
        reader, writer = await asyncio.open_connection(
            "127.0.0.1", live.record.control_port, limit=1 << 20
        )
        writer.write(
            json.dumps({"key": live.record.control_key, "client": "daemon"}).encode() + b"\n"
        )
        await writer.drain()
        welcome = await asyncio.wait_for(reader.readline(), timeout=10)
        assert json.loads(welcome)["op"] == "projection"
        assert cap.hex() not in welcome.decode()
        # The session's own rendered export, which is what `lop info` and the
        # desktop panels read.
        assert cap.hex() not in json.dumps(live.handle._fold.projection.to_json())
        writer.close()
    finally:
        await live.close(tmp_path)


def test_the_refusal_copy_names_the_remedies_and_not_a_rule() -> None:
    """The copy is part of the contract, so its content is pinned.

    Following #1291's precedent: it names the one-step remedy rather than the
    rule, and it must NOT promise a boundary (the residual section of the design
    doc is the honest statement of what this does not cover). Four clauses are
    load-bearing after revision 2: WHO has the authority (the operator, not a
    window), how the local reader exercises it (a presence gesture), where the
    other lever is (a paired phone), and that tightening still works here. The
    retired third clause — retiring the runtime so this window could own the next
    one — is asserted ABSENT, because it is the capability loss this revision
    exists to remove.
    """
    copy = OPERATOR_AUTHORITY_REQUIRED_NOTICE
    # WHAT CHANGED IN REVISION 2, and this first assertion is the change: the
    # copy used to answer "who may loosen?" with a WINDOW ("the window that
    # started this session") and send the reader to make their own window the
    # right one. Authority is now a fact about the operator, verifiable from any
    # surface, so the copy names the OPERATOR and the two places a person can
    # answer a prompt for them.
    assert "the operator's own consent" in copy
    assert "Touch ID" in copy, "the local remedy must name the gesture"
    assert "paired phone" in copy
    # THE REMEDY REVISION 2 DELETES, asserted ABSENT rather than merely not
    # asserted present. "Let its runtime retire and reopen it here — the window
    # that opens a runtime owns its gate" was true under the spawn-capability
    # model and is the capability loss this revision exists to remove: it made the
    # reader chase a window instead of authorising, and a background-started
    # runtime has no window to reopen it from at all. Pinning the absence is what
    # stops it creeping back as a "helpful" extra sentence.
    assert "retire" not in copy
    assert "reopen it here" not in copy
    assert "the window that opens a runtime owns its gate" not in copy
    assert "the window that started this session" not in copy
    # The remedies that are true for the NEXT session, and where they live.
    assert "--yolo" in copy
    assert "tool_approval_mode: auto" in copy
    # What still works from here.
    assert "/approvals ask" in copy
    # NOT `/approvals default ...`: the runtime refuses that command from the
    # very pane this notice is printed in (UX round 1 U1), so a remedy that
    # cannot work where it is read must not be in it.
    assert "/approvals default" not in copy
    # No promise of a boundary this host may not have. The residual is that a
    # HOST with no presence store (Linux today) signs without a gesture, and
    # that is `lop operator status`'s job to say — a refusal copy that claimed a
    # boundary everywhere would be wrong on exactly that host.
    assert re.search(r"safe|secure|protected|cannot be read", copy) is None
    # Not a visual twin of #1291's notice, which is about a config write
    # arriving from outside this session rather than a command typed where the
    # gate is not owned (design round 1 D4).
    from local_operator.harness.approval import LOOSENING_REFUSED_NOTICE

    assert copy.split(":")[0] != LOOSENING_REFUSED_NOTICE.split(":")[0]

    # IT HAS TO TRAVEL WHOLE, and it has to survive the narrowest frame the
    # product has. The runtime slices a raised exception's text to 400 characters
    # (``server.py``) before it answers, so a longer copy reaches a raw client as
    # a sentence that stops mid-remedy (QA round 2, Q3); and 44 columns with an
    # 11-row viewport is the smallest surface the response is rendered on, where
    # the previous 537-character copy pushed the reason and the remedy off the
    # top of the view (design round 2, D8). Both are pinned as NUMBERS because
    # both were regressions of exactly that kind.
    assert len(copy) <= 400, len(copy)
    # HOW MANY ROWS IT TAKES TO READ IS NOT THIS FILE'S NUMBER. The notice block
    # wraps at its own content width (40 cells at a 44-column terminal), so a
    # wrap of the COPY at 44 said "9 rows" while the rendered block was 12 —
    # a pin that measured a wrapping the frame does not do (design round 3, D14;
    # agent R3-5). The rendered bound lives where the renderer is:
    # ``tests/unit/tui/test_approvals_ux.py``
    # ``::test_the_refused_card_notice_reaches_the_screen``.


def test_the_card_refusal_is_its_own_sentence() -> None:
    """A refused CARD is not a refused command (UX round 2, U8).

    The card's reader never typed ``/approvals auto``: they pressed a key on a
    parked question. Answering them with the command's copy told a phone user to
    "type it in the terminal or app window that started this session" — advice
    they cannot take — and never said whether the question survived. So the op
    the seam refused picks the sentence, the token travels (never the prose), and
    the far side rebuilds the same one locally.
    """
    from local_operator.harness.approval import (
        CARD_APPROVAL_REFUSED_NOTICE,
        OPERATOR_AUTHORITY_REQUIRED_NOTICE,
    )
    from local_operator.session.errors import OperatorAuthorityRequired, admission_error

    assert CARD_APPROVAL_REFUSED_NOTICE != OPERATOR_AUTHORITY_REQUIRED_NOTICE
    # The card's reader is told the question SURVIVED and that a deny works from
    # where they are — the one action that does.
    assert "still waiting" in CARD_APPROVAL_REFUSED_NOTICE
    assert "Denying it works from here" in CARD_APPROVAL_REFUSED_NOTICE
    # It names no command they never typed.
    assert "/approvals auto" not in CARD_APPROVAL_REFUSED_NOTICE
    assert len(CARD_APPROVAL_REFUSED_NOTICE) <= 400

    # The seam's refusal carries the op as a TOKEN, and the decoder turns a token
    # back into the sentence — no prose crosses the wire.
    raised = OperatorAuthorityRequired(trigger="approval_answer")
    assert str(raised) == CARD_APPROVAL_REFUSED_NOTICE
    assert raised.trigger == "approval_answer"
    rebuilt = admission_error("operator_authority_required", None, "approval_answer")
    assert isinstance(rebuilt, OperatorAuthorityRequired)
    assert str(rebuilt) == CARD_APPROVAL_REFUSED_NOTICE
    # An unknown token is not prose either: it falls back to the command's copy.
    assert str(OperatorAuthorityRequired(trigger="../../etc/passwd")) == (
        OPERATOR_AUTHORITY_REQUIRED_NOTICE
    )
    assert str(admission_error("operator_authority_required")) == OPERATOR_AUTHORITY_REQUIRED_NOTICE


# ---------------------------------------------------------------------------
# The real spawn: the handoff, end to end, on a real detached runtime
# ---------------------------------------------------------------------------

#: The session this module's real-child test spawns, kept distinct from the
#: detachment module's id so the two cannot find each other's record.
_REAL_SESSION_ID = "approvalauth01"


def _seed_real(config_dir: Path) -> Path:
    """A resumable session on the mock provider, its gate at ``ask``.

    Mirrors ``test_runtime_detachment._seed`` (no API key, no network) with ONE
    difference that is the point of this file: the mode is ``ask``, so a
    loosening is observable as a transition rather than as a no-op.
    """
    directory = config_dir / "sessions" / _REAL_SESSION_ID
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_text(
        '{"id": "seed", "ts": 1, "type": "message", "payload": {"kind": "message", '
        '"role": "user", "content": [{"type": "text", "text": "seed"}]}}\n',
        encoding="utf-8",
    )
    (config_dir / "config.yml").write_text(
        "values:\n  hosting: test\n  model_name: mock\n  tool_approval_mode: ask\n",
        encoding="utf-8",
    )
    return directory


def _wait_for_real_record(config_dir: Path, *, timeout: float = 30.0) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for record, _state in registry.scan(config_dir):
            if getattr(record, "session_id", "") == _REAL_SESSION_ID:
                return record
        time.sleep(0.05)
    raise AssertionError(f"no record for {_REAL_SESSION_ID} within {timeout}s")


@pytest.mark.asyncio
@pytest.mark.slow
async def test_a_real_detached_runtime_gets_the_capability_and_refuses_a_peer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE WHOLE MECHANISM, over a real fork and a real ``exec``.

    Everything else in this file runs the runtime in-process, which cannot
    observe the properties that only exist across a spawn: the descriptor
    surviving ``exec``, the child reading it before it serves, the spawner
    resolving it against the child's pid, and the refusals holding on a session
    with a REAL session directory and a real tool-approval mode.

    Three claims, in order:

    1. the child booted WITH a capability — the spawner, which minted it, can
       loosen the gate through the socket, and the loosening is visible in the
       child's own runtime log;
    2. a same-uid process that reads only the record cannot (the issue's
       regression shape, on the real thing);
    3. the guarantee level is reported in that log, so the boundary's strength on
       this host is a fact an operator can read rather than an assumption.

    Uses the detachment module's own isolation helpers: ``CMUX_*``/``LOP_*``
    stripped, ``LOCAL_OPERATOR_CONFIG_DIR`` redirected, a long residency grace so
    the reaper is not what ends the child.
    """
    from tests.unit.session.runtime import test_runtime_detachment as detachment

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    _seed_real(config_dir)
    detachment._isolate(monkeypatch, config_dir)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))
    reset_operator_caps_for_tests()

    child = None
    try:
        child = launch_module._spawn_runtime(
            _REAL_SESSION_ID, str(config_dir), defer_materialise=False
        )
        record = _wait_for_real_record(config_dir)
        assert int(record.pid) == child.pid, (record.pid, child.pid)

        # (1) THE SPAWNER HOLDS IT. The capability was keyed on the child's pid
        # by the spawn itself, so a console dialling this record can present it.
        held = operator_cap_for(record.pid)
        assert held is not None, "the spawn did not register the capability it minted"
        assert len(held) == 32
        conn = await _dial(record, client="daemon", cap=held)
        loosened = await _send(
            conn,
            None,
            {
                "op": "slash_result",
                "command": "approvals",
                "args": "auto",
                "images": [],
                "operator_cap": conn.proof(held),
            },
        )
        assert loosened["op"] == "result", loosened
        conn.close()

        # (2) A PEER WITH ONLY THE RECORD CANNOT. The same attempt program the
        # in-process control uses, against the real child.
        script = tmp_path / "attempt.py"
        script.write_text(_ATTEMPT_PROGRAM, encoding="utf-8")
        attempt = await asyncio.to_thread(
            subprocess.run,
            [sys.executable, str(script), str(config_dir)],
            capture_output=True,
            check=False,
        )
        out = attempt.stdout.decode()
        assert attempt.returncode == 0, out + attempt.stderr.decode()
        assert "welcome projection" in out, out
        # The gate is `auto` now (claim 1), so tighten it back first: the point
        # below is that the peer cannot LOOSEN, and a frame that changed nothing
        # would prove nothing.
        conn = await _dial(record, client="daemon")
        tightened = await _send(
            conn,
            None,
            {"op": "slash_result", "command": "approvals", "args": "ask", "images": []},
        )
        assert tightened["op"] == "result", tightened
        conn.close()

        attempt = await asyncio.to_thread(
            subprocess.run,
            [sys.executable, str(script), str(config_dir)],
            capture_output=True,
            check=False,
        )
        out = attempt.stdout.decode()
        assert "welcome projection" in out, out
        assert "slash_result error" in out, out
        assert _REFUSAL in out, out
        # Read back through the socket that the gate really is still `ask`: the
        # report is an ordinary op, so this is the same channel a user would use.
        conn = await _dial(record, client="daemon")
        report = await _send(
            conn, None, {"op": "slash_result", "command": "approvals", "args": "", "images": []}
        )
        assert report["op"] == "result", report
        assert "tool approvals: ask" in json.dumps(report), report
        conn.close()

        # (3) THE LEVEL IS REPORTED IN THE CHILD'S OWN LOG. The phrase is pinned
        # as a STEM because the level itself is a property of the host: the child
        # reports whatever this machine is, and the assertion is that it reported
        # SOMETHING, which is what makes the not-at-rest claims below meaningful.
        text = detachment._log_text(config_dir)
        assert "operator authority:" in text, text[-2000:]
    finally:
        if child is not None:
            detachment._reap(child, config_dir)
        reset_operator_caps_for_tests()


class _AttachableSession(FakeSession):
    """A ``FakeSession`` a real follower can attach to.

    ``AttachedSession`` — the client the desktop command route actually uses —
    negotiates ``frontend_state``, so the runtime asks this session for a
    frontend seed and a subscription. ``test_serving.FakeSession`` predates that
    negotiation, and the double below adds the two members from the same store
    ``test_server.FakeHandle`` uses, so the follower's dial completes for the
    real reason rather than because the test switched frontend state off.
    """

    def __init__(self) -> None:
        super().__init__()
        from local_operator.session.frontend_state import (
            FrontendModelSpec,
            FrontendSessionState,
            FrontendStateStore,
        )

        spec = FrontendModelSpec(provider="test", model_id="model", context_window=1_000_000)
        self._frontend = FrontendStateStore(
            FrontendSessionState(
                session_id=self.session_id,
                epoch="approval-seam",
                cwd="/tmp",
                conversation_title="seam",
                selected_model=spec,
                effective_model=spec,
                context_window=1_000_000,
            )
        )

    @property
    def frontend_state_seed(self) -> Any:
        return self._frontend.state

    def subscribe_frontend(self, on_update: Any, *, display_window: bool = False) -> Any:
        return self._frontend.subscribe(on_update)


async def _never_take_over() -> Any:
    raise AssertionError("a live owner must not be taken over in these tests")


async def _follower(tmp_path: Path, record: Any) -> Any:
    """A REAL ``AttachedSession`` on the record — the desktop route's own client.

    ``POST /v1/desktop/sessions/{id}/commands`` does not dispatch a slash
    itself: it calls ``bridge.remote.route_shared_slash(...)``, and
    ``bridge.remote`` is exactly this object. So driving it here exercises the
    desktop route's body — its op, its client, its presentation — without
    standing up the FastAPI app and a bound bridge pool, which the desktop
    suite covers separately.
    """
    from local_operator.session.attached import AttachedSession

    remote = await AttachedSession.connect(
        record, "sess-1", config_dir=tmp_path, takeover_factory=_never_take_over
    )
    remote.subscribe(lambda _event: None)
    return remote


@pytest.mark.asyncio
async def test_the_desktop_route_cannot_loosen_a_runtime_this_backend_did_not_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The desktop route's real code path, on a backend that did not spawn it.

    ``DesktopSessionBridge`` engages a runtime for a session the app is asked to
    open; for a session another process started (a terminal, launchd, a wake) the
    pool attaches to THAT record and this process holds no capability for it, so
    the loosening must be refused with the copy naming the remedies. Nothing here
    weakens the app: reports and tightenings keep working (their own test is
    above), and a backend that DID spawn the runtime keeps loosening, which is
    the positive control that follows the shape of this one.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "sess-1").mkdir(parents=True, exist_ok=True)
    reset_operator_caps_for_tests()  # nobody here spawned this runtime
    live = await _serve(tmp_path, operator_cap=mint_operator_cap())
    remote = None
    try:
        remote = await _follower(tmp_path, live.record)
        # THE TYPED CATEGORY, not a bare RuntimeError: the desktop route (and the
        # relay, and the attach screen) carry this by its ``code`` and render its
        # copy, which is what stops the desktop answering 503
        # "runtime_unreachable" for a deliberate refusal (agent review round 1,
        # R1-2 = design D1 = UX U4 = QA Q1).
        from local_operator.session.errors import OperatorAuthorityRequired

        with pytest.raises(OperatorAuthorityRequired) as refused:
            await remote.route_shared_slash("approvals", "auto")
        assert refused.value.code == "operator_authority_required"
        assert OPERATOR_AUTHORITY_REQUIRED_NOTICE in str(refused.value)
        assert live.handle._auto_approve is False
    finally:
        if remote is not None:
            await remote.dispose()
        await live.close(tmp_path)


@pytest.mark.asyncio
async def test_the_desktop_route_loosens_a_runtime_this_backend_did_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operator_cap: bytes
) -> None:
    """The POSITIVE CONTROL for the negative above, through the same path.

    Without it, "the desktop route is refused" is satisfiable by a route that
    refuses everything. Here this process DID start the runtime (the fixture
    registers the capability against this process's pid, which is the record's),
    so the app's own command route keeps loosening in one step — the operator's
    criterion (2), on the client the desktop actually uses.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "sess-1").mkdir(parents=True, exist_ok=True)
    live = await _serve(tmp_path, operator_cap=operator_cap)
    remote = None
    try:
        remote = await _follower(tmp_path, live.record)
        result = await remote.route_shared_slash("approvals", "auto")
        assert live.handle._auto_approve is True, result
        assert "auto" in json.dumps(result), result
    finally:
        if remote is not None:
            await remote.dispose()
        await live.close(tmp_path)


#: Modules allowed to name the capability at all. Every one of them either mints
#: it, hands it over, demands it, presents it, or writes the sentence about it:
#: anything else naming it is a place the value could come to rest — a record
#: field, an `info` export, a projection, a log formatter.
_CAPABILITY_MODULES = frozenset(
    {
        "local_operator/harness/approval.py",
        "local_operator/mobile/attach_client.py",
        "local_operator/mobile/daemon.py",
        "local_operator/mobile/types.py",
        "local_operator/operator/__init__.py",
        "local_operator/session/runtime/exec_control.py",
        "local_operator/session/runtime/launch.py",
        "local_operator/session/runtime/process.py",
        "local_operator/session/runtime/server.py",
        "local_operator/tui/app.py",
    }
)

#: Why ``local_operator/operator/__init__.py`` is on that list, stated rather
#: than assumed: revision 2 made ``operator_authority_level()`` ABSORB the
#: capability guarantee, so the level report quotes it (``capability_guarantee``)
#: beside the anchor facts. That module asks the host-level question and never
#: receives the VALUE — it is the reporter, not a carrier, and the pin's concern
#: (a serializer, an export, or a log path learning the capability) does not apply
#: to it. Recorded here so the next reader can check the claim instead of
#: re-deriving it.


def test_the_capability_name_appears_only_where_it_has_to() -> None:
    """A source pin on the VALUE'S blast radius, not on a route.

    The capability is a boundary only while it stays in memory. Every module in
    the tree is checked for naming it, and the allowlist is the set that must:
    the mint/handoff/comparison, the child's read, the seam that demands it, the
    client that presents it, the HTTP boundary that drops it, the validator that
    types it, and the TUI host that mints its own. A module `lop info` renders,
    a serialization helper, or any other surface can therefore never quietly
    start carrying it.

    Read from the tree rather than from a list of imports, so a NEW module is
    caught by being new rather than by being reviewed.
    """
    offenders: dict[str, int] = {}
    for module in sorted((_TESTS_ROOT / "local_operator").rglob("*.py")):
        relative = module.relative_to(_TESTS_ROOT).as_posix()
        if relative in _CAPABILITY_MODULES:
            continue
        hits = module.read_text(encoding="utf-8").count("operator_cap")
        if hits:
            offenders[relative] = hits
    assert offenders == {}, (
        f"the operator capability is named in {sorted(offenders)}. It may only exist in the "
        "handful of modules that mint, hand over, demand or present it: "
        f"{sorted(_CAPABILITY_MODULES)}. "
        "If a new module needs it, add it here deliberately — and check it is not a serializer, an "
        "export, or a log path, which is where it would stop being a boundary."
    )


@pytest.mark.asyncio
@pytest.mark.slow
async def test_a_real_runtime_writes_the_capability_nowhere_it_could_be_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The not-at-rest claim on the REAL artifacts of a real spawn.

    Holds a capability whose value this test knows, boots a real detached runtime
    that received it, and then greps everything the child produced or published:
    the discovery record, the child's own ``runtime.log``, and the stdio capture
    the spawn keeps. Nothing may contain it — that is what "held only in memory"
    has to mean for the claim to be worth anything.
    """
    from tests.unit.session.runtime import test_runtime_detachment as detachment

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    _seed_real(config_dir)
    detachment._isolate(monkeypatch, config_dir)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))
    reset_operator_caps_for_tests()

    child = None
    try:
        child = launch_module._spawn_runtime(
            _REAL_SESSION_ID, str(config_dir), defer_materialise=False
        )
        record = _wait_for_real_record(config_dir)
        held = operator_cap_for(record.pid)
        assert held is not None, "no capability was registered for the child we just spawned"
        needle = held.hex()

        record_file = config_dir / "run" / "mobile" / f"{record.pid}.json"
        assert record_file.exists(), f"no record at {record_file}"
        assert needle not in record_file.read_text(encoding="utf-8")

        # The child's own log: give it a moment to have written its boot lines,
        # bounded by the file appearing rather than by a wall-clock guess.
        log_path = config_dir / "logs" / "runtime.log"
        for _ in range(200):
            if log_path.exists() and "session runtime started" in detachment._log_text(config_dir):
                break
            time.sleep(0.05)
        log_text = detachment._log_text(config_dir)
        assert "session runtime started" in log_text, log_text[-2000:]
        assert needle not in log_text
        # ...and the level IS reported there, so the file is a real artifact of
        # this child rather than an empty one this assertion passes on. Pinned as
        # a stem for the reason the sibling test records: the level is a property
        # of the host, not a constant.
        assert "operator authority:" in log_text, log_text[-2000:]

        capture = getattr(child, "lop_capture_path", None)
        if capture is not None and Path(capture).exists():
            assert needle not in Path(capture).read_text(encoding="utf-8", errors="replace")

        # THE ENVIRONMENT, PROBED RATHER THAN ASSERTED (agent review round 1,
        # MINOR). ``ps -E`` shows a same-uid process its own child's environment
        # on macOS, so this reads the REAL child's rather than reasoning about
        # what the spawn passed. Skipped where the platform cannot answer, with
        # the reason named instead of a silent pass.
        if sys.platform == "darwin":
            shown = subprocess.run(
                ["ps", "-Eww", "-o", "command=", "-p", str(child.pid)],
                capture_output=True,
                text=True,
                check=False,
            )
            assert shown.returncode == 0, shown.stderr
            assert (
                needle not in shown.stdout
            ), "the capability is visible in the child's environment"
    finally:
        if child is not None:
            detachment._reap(child, config_dir)
        reset_operator_caps_for_tests()


# ---------------------------------------------------------------------------
# The record-rewriting attack
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_impostor_endpoint_learns_nothing_it_can_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE HARVEST, driven end to end with the production clients (R1-1).

    The discovery record is 0600 — and WRITABLE by anything running under this
    uid, which is the model-run tool the gate exists to constrain. So
    ``control_port`` is not a trusted pointer: an attacker rewrites it to its
    own listener and waits for the console to dial. The console here is a real
    ``AttachClient`` holding a real capability for the runtime at that pid, and
    the property under test is that a same-uid endpoint answering in the
    runtime's place learns NOTHING it can use:

    * the frame it receives carries no credential — not the capability, and (as
      the assertion on the raw frames shows) not even a proof, because the
      client withholds both from an endpoint that did not prove itself;
    * what the impostor answers flips nothing: replayed at the REAL runtime, the
      gate stays at ``ask``.

    The regression this pins was measured before the fix: with the value sent as
    the frame's field, the same rig printed ``harvested the capability: True``
    and the replay set ``_auto_approve`` to True.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    from dataclasses import replace

    from local_operator.mobile.attach_client import AttachClient

    cap = mint_operator_cap()
    live = await _serve(tmp_path, operator_cap=cap)
    try:
        # What the SPAWN would have done in this process for the runtime it
        # started, so the console legitimately holds a capability for this pid.
        remember_operator_cap(live.record.pid, cap)

        seen: list[dict[str, Any]] = []

        async def impostor(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            """Answer the auth, look as much like a runtime as it can, collect."""
            await reader.readline()
            writer.write(
                json.dumps(
                    {
                        "op": "projection",
                        "data": {
                            "session_id": live.record.session_id,
                            "pid": live.record.pid,
                            "kind": "tui",
                        },
                    }
                ).encode()
                + b"\n"
            )
            await writer.drain()
            while True:
                line = await reader.readline()
                if not line:
                    break
                try:
                    frame = json.loads(line)
                except ValueError:
                    continue
                seen.append(frame)
                writer.write(
                    json.dumps(
                        {"op": "result", "req": frame.get("req"), "data": {"kind": "notice"}}
                    ).encode()
                    + b"\n"
                )
                await writer.drain()

        server = await asyncio.start_server(impostor, "127.0.0.1", 0)
        fake_port = server.sockets[0].getsockname()[1]
        try:
            forged = replace(live.record, control_port=fake_port)
            client = AttachClient(lambda _projection: None, lambda _reason: None, locality="local")
            await client.connect(forged, live.record.session_id)
            try:
                with contextlib.suppress(Exception):
                    await client._request_payload(
                        "slash_result", command="approvals", args="auto", images=[]
                    )
                # The handshake never completed, so this connection may not
                # present anything — the whole of the defence, in one flag.
                assert client._authority_bearing is False
            finally:
                client.close()
        finally:
            server.close()

        assert seen, "the impostor was never dialled; the rig proves nothing"
        wire = json.dumps(seen)
        assert cap.hex() not in wire, "the capability VALUE crossed to an impostor endpoint"
        assert (
            "operator_cap" not in wire
        ), "a proof was presented to an endpoint that proved nothing"

        # Whatever the impostor collected, replayed at the REAL runtime, changes
        # nothing: the gate is still armed and still parks.
        replay = await _dial(live.record, client="attach")
        reply = await _send(
            replay,
            None,
            {
                "op": "slash_result",
                "command": "approvals",
                "args": "auto",
                "images": [],
                "operator_cap": cap.hex(),
            },
        )
        assert reply["op"] == "error", reply
        assert live.handle._auto_approve is False
        replay.close()
    finally:
        await live.close(tmp_path)


# ---------------------------------------------------------------------------
# Stage D: the PHONE, through the relay, over the relay's real HTTP surface
# ---------------------------------------------------------------------------
#
# WHAT USED TO BE HERE, AND WHY IT IS GONE. The cell that lived at this point was
# ``test_the_relay_keeps_its_handshake_across_a_repaint``, and it FABRICATED the
# thing it claimed to measure: it called ``remember_operator_cap(record.pid, cap)``
# itself and then asserted that a relay holding that capability could loosen. No
# production path ever registers a capability for a relay-spawned runtime —
# ``mobile/daemon.py``'s own spawn builds
# ``python -m local_operator.session.runtime.process`` and passes no
# ``--operator-fd``, so ``entry.operator_cap`` is ALWAYS ``None`` there — which
# means the test proved that a hand-built state works, not that the phone works.
#
# Revision 2 removes the need for that state entirely: the phone's authority is a
# DEVICE SIGNATURE under an operator-signed certificate, and it therefore does not
# depend on who spawned the runtime. So the fabrication is deleted and replaced by
# the cell below, which drives the REAL path — the real runtime, the relay's real
# dial, the relay's real HTTP surface, a real ES256 device key — with NO
# ``remember_operator_cap`` anywhere in it. ``operator_cap_for`` returns ``None``
# for the runtime behind it, exactly as it does in production.
#
# The negative fact is pinned too (:func:`test_the_relays_own_spawn_path_registers_`
# ``no_capability``), because "the relay's capability is always None" is now a
# deliberate property rather than an accident: if a future change DID hand the
# relay a capability, the phone's cell below would still pass and nothing would
# notice that the design's justification had changed.


async def _new_device_key() -> tuple[Any, bytes]:
    """A device's ES256 private key and its uncompressed public point.

    Stands in for WebCrypto's non-extractable key: the POINT is what the operator
    certifies and what the runtime verifies against, and the private half is used
    here only to produce the signature the phone would produce. The portal's own
    half of this is covered by vitest (``operator-device.test.ts``); what cannot
    be covered there is the WIRE, which is what this rig drives.
    """
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    key = ec.generate_private_key(ec.SECP256R1())
    point = key.public_key().public_bytes(Encoding.X962, PublicFormat.UncompressedPoint)
    return key, point


def _store_device(
    config_root: Path,
    *,
    device_spki: bytes,
    certificate: str,
    name: str = "test phone",
) -> Any:
    """Install a device certificate through the PRODUCTION store writer.

    Not a hand-written JSON file: the on-disk shape, the 0644-under-0700 modes and
    the derived id are all facts the relay reads, and a test that wrote its own
    file would prove nothing about the writer the pairing flow actually uses.
    """
    from local_operator.operator import devices
    from local_operator.operator.verify import read_device_cert

    parsed = read_device_cert(certificate)
    assert parsed is not None, "the rig signed a certificate it cannot parse"
    return devices.write_device_cert(
        config_root,
        certificate=certificate,
        parsed=parsed,
        operator_key_id="",
        name=name,
    )


def _issue_device_certificate(config_root: Path, device_spki: bytes, *, label: str) -> str:
    """An operator-signed device certificate, from the REAL signer.

    ``load_signer`` + ``issue_device_cert`` are the pairing flow's own two steps,
    and the ``file-only`` backend is the only one a test may create (see
    ``_install_operator_key``). Nothing here touches the operator's login keychain.
    """
    from local_operator.operator.sign import issue_device_cert, load_signer
    from local_operator.operator.verify import key_id_for

    signer = load_signer(config_root=config_root, backend_name="file-only")
    assert signer is not None, "the rig's operator key did not load"
    try:
        return issue_device_cert(
            device_spki=device_spki,
            device_id=key_id_for(device_spki),
            label=label,
            signer=signer,
        )
    finally:
        signer.close()


def _device_signature(
    key: Any, *, action: str, session_id: str, request_id: str, challenge: str
) -> str:
    """What the phone produces: ES256 over the domain-separated, bound message."""
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec

    from local_operator.operator.verify import signed_message

    message = signed_message(
        action=action, session_id=session_id, request_id=request_id, challenge=challenge
    )
    return key.sign(message, ec.ECDSA(hashes.SHA256())).hex()


@pytest.mark.asyncio
async def test_a_phone_signature_loosens_and_approves_for_a_session_the_relay_did_not_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE HEADLINE SURFACE (design §3, phone row; matrix cells P1 and P2).

    A phone, reaching the machine through the relay's real HTTP surface, both
    LOOSENS a running gate and APPROVES a parked card — for a session this
    process never spawned and the relay never spawned either. Every hop is the
    production one:

    * the runtime mints the challenge on the relay's own control connection
      (``daemon.request`` -> the runtime's ``operator_challenge`` op);
    * the relay hands the challenge back over HTTP and forwards the command the
      phone signs;
    * the runtime verifies an ES256 signature from a device key against an
      operator-SIGNED certificate under its anchored operator key.

    THE RELAY HOLDS NOTHING HERE, and that is the point rather than a detail:
    ``reset_operator_caps_for_tests()`` leaves ``operator_cap_for`` empty, so
    ``entry.operator_cap`` is ``None`` and the relay's dial offers no nonce — the
    exact state production is in. The old cell for this surface fabricated a
    capability instead of accepting that.
    """
    import httpx

    from local_operator.harness.approval import reset_operator_caps_for_tests
    from local_operator.mobile.daemon import (
        MobileDaemon,
        SessionEntry,
        _dial,
        build_app,
    )
    from local_operator.operator.verify import key_id_for

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    reset_operator_caps_for_tests()  # NOBODY spawned this runtime
    anchor = _install_operator_key(config_dir())
    live = await _serve(tmp_path, operator_cap=mint_operator_cap(), operator_anchor=anchor)
    session_id = live.record.session_id
    try:
        assert operator_cap_for(live.record.pid) is None, "the rig spawned it after all"

        device_key, device_point = await _new_device_key()
        certificate = _issue_device_certificate(config_dir(), device_point, label="test phone")
        stored = _store_device(config_dir(), device_spki=device_point, certificate=certificate)
        # The relay declares the certificate on its auth frame, which is how the
        # runtime can answer "this connection may loosen" before a frame arrives.
        assert stored.device_id == key_id_for(device_point)

        daemon = MobileDaemon(port=0, password="pw")
        entry = SessionEntry(live.record)
        daemon.table.entries[live.record.pid] = entry
        dial = asyncio.ensure_future(_dial(daemon, entry))
        try:
            # Wait for the relay's connection to be ESTABLISHED before asking it to
            # mint a challenge: ``_dial`` publishes the writer only after it has
            # authenticated and read the welcome, and a request racing that is
            # answered 409 — the correct answer to a question asked too early, not
            # a defect.
            for _ in range(300):
                if entry.writer is not None and entry.projection is not None:
                    break
                await asyncio.sleep(0.02)
            assert entry.writer is not None, "the relay never completed its dial"
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=build_app(daemon)), base_url="http://relay.local"
            ) as relay:
                login = await relay.post("/login", data={"password": "pw"})
                assert login.status_code in (200, 303), login.text

                async def phone_frame(*, action: str, request_id: str, body: dict[str, Any]) -> Any:
                    """Challenge over HTTP, signature off the loop, command over HTTP."""
                    challenge_reply = await relay.post(
                        f"/api/sessions/{session_id}/operator/challenge",
                        json={"action": action, "request_id": request_id},
                    )
                    assert challenge_reply.status_code == 200, challenge_reply.text
                    challenge = challenge_reply.json()["challenge"]
                    signature = _device_signature(
                        device_key,
                        action=action,
                        session_id=session_id,
                        request_id=request_id,
                        challenge=challenge,
                    )
                    return await relay.post(
                        f"/api/sessions/{session_id}/command",
                        json={
                            "operator_sig": signature,
                            "operator_key_id": key_id_for(device_point),
                            "operator_cert": certificate,
                            **body,
                        },
                    )

                # P2 — the phone LOOSENS the gate.
                #
                # ``slash_result``, NOT ``slash``: the session-level ``slash`` op is
                # the off-terminal SUBSET (``/goal``, ``/compact``) and refuses
                # ``/approvals`` with "/approvals is terminal-only here" — the dead
                # end the design deletes. ``slash_result`` is the ROUTED op the
                # runtime's authority seam was built for and the one the desktop
                # backend and the TUI's attached pane already use, so the phone is
                # taking the same road rather than a second one.
                loosened = await phone_frame(
                    action="loosen",
                    request_id="",
                    body={
                        "op": "slash_result",
                        "command": "approvals",
                        "args": "auto",
                        "images": [],
                    },
                )
                assert loosened.status_code == 200, loosened.text
                assert live.handle._auto_approve is True, "the phone's signature did not loosen it"
                assert await live.handle._approval_gate("bash", "rm -rf build/") is True

                # P1 — the phone APPROVES a parked card, on a fresh challenge.
                live.handle._auto_approve = False
                parked = await _park_a_card(live.handle)
                pending = live.handle._fold.projection.pending
                assert pending is not None, "no card parked, so the approval proves nothing"
                approved = await phone_frame(
                    action="approve",
                    request_id=pending.request_id,
                    body={
                        "op": "approval_answer",
                        "request_id": pending.request_id,
                        "approved": True,
                        "remember": False,
                    },
                )
                assert approved.status_code == 200, approved.text
                assert await parked is True, "the phone's approval did not resolve the card"
                assert live.handle._fold.projection.pending is None
        finally:
            dial.cancel()
    finally:
        await live.close(tmp_path)


class _PhoneOverRelay:
    """A paired phone driving one live runtime through a real relay.

    The negative controls below all need the same four hops standing up — a live
    runtime, a relay that really dialled it, a paired device key, and the relay's
    real HTTP surface with a session cookie — and each of them then differs in ONE
    place. Building that once is what keeps each control to the single line that
    makes it a control, which is the difference between a negative test that
    proves something and a negative test that proves the rig works.

    Not a fixture: the runtime and the relay both belong to the test's own event
    loop (``_serve`` and ``_dial`` both run on it), so the setup has to happen
    inside the test rather than in a synchronous fixture.
    """

    def __init__(self, *, live: Any, daemon: Any, entry: Any, relay: Any, **parts: Any) -> None:
        self.live = live
        self.daemon = daemon
        self.entry = entry
        self.relay = relay
        self.device_key = parts["device_key"]
        self.device_point = parts["device_point"]
        self.certificate = parts["certificate"]
        self.session_id = parts["session_id"]
        self.tmp_path = parts["tmp_path"]
        self._dial = parts["dial"]
        self._client = parts["client"]

    async def challenge(self, *, action: str, request_id: str = "") -> Any:
        return await self.relay.post(
            f"/api/sessions/{self.session_id}/operator/challenge",
            json={"action": action, "request_id": request_id},
        )

    def sign(
        self,
        *,
        action: str,
        request_id: str,
        challenge: str,
        key: Any | None = None,
    ) -> str:
        return _device_signature(
            key if key is not None else self.device_key,
            action=action,
            session_id=self.session_id,
            request_id=request_id,
            challenge=challenge,
        )

    async def command(self, body: dict[str, Any]) -> Any:
        return await self.relay.post(f"/api/sessions/{self.session_id}/command", json=body)

    async def aclose(self) -> None:
        self._dial.cancel()
        await self._client.aclose()
        await self.live.close(self.tmp_path)


async def _phone_over_relay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    anchor: Any = None,
    certificate: str | None = None,
    device_point: bytes | None = None,
) -> _PhoneOverRelay:
    """Stand up the four hops. ``certificate``/``device_point`` let a control
    substitute its own key material without rebuilding the rig."""
    import httpx

    from local_operator.harness.approval import reset_operator_caps_for_tests
    from local_operator.mobile.daemon import (
        MobileDaemon,
        SessionEntry,
        _dial,
        build_app,
    )

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    reset_operator_caps_for_tests()  # the relay never spawned this runtime
    if anchor is None:
        anchor = _install_operator_key(config_dir())
    live = await _serve(tmp_path, operator_cap=mint_operator_cap(), operator_anchor=anchor)
    if device_point is None:
        device_key, device_point = await _new_device_key()
    else:
        device_key = None
    if certificate is None:
        certificate = _issue_device_certificate(config_dir(), device_point, label="test phone")
        stored = _store_device(config_dir(), device_spki=device_point, certificate=certificate)
        assert stored.device_id == key_id_for(device_point)

    daemon = MobileDaemon(port=0, password="pw")
    entry = SessionEntry(live.record)
    daemon.table.entries[live.record.pid] = entry
    dial = asyncio.ensure_future(_dial(daemon, entry))
    for _ in range(300):
        if entry.writer is not None and entry.projection is not None:
            break
        await asyncio.sleep(0.02)
    assert entry.writer is not None, "the relay never completed its dial"
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=build_app(daemon)), base_url="http://relay.local"
    )
    await client.__aenter__()
    login = await client.post("/login", data={"password": "pw"})
    assert login.status_code in (200, 303), login.text
    return _PhoneOverRelay(
        live=live,
        daemon=daemon,
        entry=entry,
        relay=client,
        device_key=device_key,
        device_point=device_point,
        certificate=certificate,
        session_id=live.record.session_id,
        tmp_path=tmp_path,
        dial=dial,
        client=client,
    )


@pytest.mark.asyncio
async def test_a_forged_device_certificate_cannot_loosen_the_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Matrix cell N3, the half that is about SUBSTITUTION rather than ownership.

    A relay that has been taken over — a same-uid child that read the portal
    password and drives the local HTTP surface — can present any certificate it
    likes. This one is well-formed, names a real P-256 point, and is signed by a
    key the machine has never seen: ``verify_device_cert`` checks it against the
    ANCHORED operator key and refuses. The gate therefore does not move, and the
    refusal is the runtime's rather than the relay's — which is why the relay must
    stay a forwarder and cannot be the thing that decides.
    """
    attacker_key, attacker_point = await _new_device_key()
    # Signed by the ATTACKER, not the operator: the certificate's shape is right
    # and its signature is not.
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec

    from local_operator.operator.verify import DeviceCert, key_id_for

    now = int(time.time())
    statement = DeviceCert(
        device_id=key_id_for(attacker_point),
        spki=attacker_point,
        label="not my phone",
        issued_at=now - 10,
        not_after=now + 3600,
    )
    forged = statement.encode(
        signature=attacker_key.sign(statement.payload(), ec.ECDSA(hashes.SHA256()))
    )
    # The attacker also needs the certificate to be PRESENT in the store, because
    # the relay declares one from there; that is exactly the substitution the
    # config-root store permits and the signature is what defeats.
    rig = await _phone_over_relay(
        tmp_path, monkeypatch, certificate=forged, device_point=attacker_point
    )
    try:
        challenge = (await rig.challenge(action="loosen")).json()["challenge"]
        signature = _device_signature(
            attacker_key,
            action="loosen",
            session_id=rig.session_id,
            request_id="",
            challenge=challenge,
        )
        reply = await rig.command(
            {
                "op": "slash_result",
                "command": "approvals",
                "args": "auto",
                "images": [],
                "operator_sig": signature,
                "operator_key_id": key_id_for(attacker_point),
                "operator_cert": forged,
            }
        )
        assert reply.status_code != 200, reply.text
        assert rig.live.handle._auto_approve is False, "a forged certificate loosened the gate"
    finally:
        await rig.aclose()


@pytest.mark.asyncio
async def test_a_replayed_phone_signature_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Matrix cell N8: one challenge, one use.

    The runtime POPS the challenge before it verifies, so the second presentation
    of a captured ``operator_sig`` finds no live challenge at all and is refused —
    regardless of the fact that the signature itself is perfectly valid, which is
    the point: replay is removed as a category rather than mitigated by a nonce
    that could be raced.
    """
    rig = await _phone_over_relay(tmp_path, monkeypatch)
    try:
        challenge = (await rig.challenge(action="loosen")).json()["challenge"]
        signature = rig.sign(action="loosen", request_id="", challenge=challenge)
        body = {
            "op": "slash_result",
            "command": "approvals",
            "args": "auto",
            "images": [],
            "operator_sig": signature,
            "operator_key_id": key_id_for(rig.device_point),
            "operator_cert": rig.certificate,
        }
        first = await rig.command(body)
        assert first.status_code == 200, first.text
        assert rig.live.handle._auto_approve is True

        # Re-arm the gate so a successful replay would be OBSERVABLE rather than
        # answering a question that was already settled.
        rig.live.handle._auto_approve = False
        second = await rig.command(body)
        assert second.status_code != 200, second.text
        assert rig.live.handle._auto_approve is False, "the replay was accepted"
    finally:
        await rig.aclose()


@pytest.mark.asyncio
async def test_the_relay_holds_no_key_material_and_cannot_mint_a_signature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Matrix cell N2, stated as the two facts that make the relay a courier.

    1. A caller holding the portal password CAN drive the relay's local HTTP
       surface — that is the threat model, not a hypothetical. What it cannot do
       is make the relay produce authority: ``operator_cap`` is dropped from the
       body (the relay mints its own, and it has none for this runtime), and the
       only fields it forwards are the phone's unforgeable, single-use ones.
    2. Nothing in the machine's operator/device store is a signing key. The
       certificate is public data and the private half is on the phone, so "steal
       the store" is not a path to a signature at all.
    """
    rig = await _phone_over_relay(tmp_path, monkeypatch)
    try:
        gate_before = rig.live.handle._auto_approve
        forged_proof = mint_operator_cap().hex()
        reply = await rig.command(
            {
                "op": "slash_result",
                "command": "approvals",
                "args": "auto",
                "images": [],
                "operator_cap": forged_proof,
            }
        )
        assert reply.status_code != 200, reply.text
        assert rig.live.handle._auto_approve is gate_before, "a pushed capability was honoured"

        # And the store: a certificate, an id, a scope and a public point — no
        # private half anywhere, and no operator key beside them.
        from local_operator.operator import devices as device_store

        root = config_dir()
        stored = device_store.read_device(root, key_id_for(rig.device_point))
        assert stored is not None
        record = json.loads(device_store.device_path(root, stored.device_id).read_text())
        assert set(record) == {
            "v",
            "kind",
            "device_id",
            "name",
            "spki",
            "key_id",
            "scope",
            "iat",
            "exp",
            "operator_key_id",
            "certificate",
        }
        assert "PRIVATE" not in record["certificate"].upper()
        for path in device_store.devices_dir(root).iterdir():
            assert "private" not in path.read_text(encoding="utf-8").lower()
    finally:
        await rig.aclose()


@pytest.mark.asyncio
async def test_a_revoked_device_cannot_sign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The revocation list is consulted at the point the device id is known.

    ``verify_device_cert`` deliberately knows nothing about revocation — it checks
    a signature and an expiry — so the check lives where the identity becomes
    available (``server._device_cert_point``), against the ROOT-OWNED anchor's
    list. An operator who revokes a lost phone therefore loses nothing to the
    certificate still sitting readable under the config root.
    """
    from dataclasses import replace

    rig = await _phone_over_relay(tmp_path, monkeypatch)
    try:
        device_id = key_id_for(rig.device_point)
        anchor = rig.live.runtime._anchor_cache.get().anchor
        assert anchor is not None
        revoking = replace(anchor, devices=(({"device_id": device_id, "revoked": True}),))
        # The runtime reads the anchor through ONE cache, pinned at first need, so
        # the revocation is installed the way a real one lands: by making the
        # cached load itself carry the revoked list.
        from local_operator.operator.trust import AnchorLoad, anchor_path

        rig.live.runtime._anchor_cache.load = AnchorLoad(
            anchor=revoking,
            path=anchor_path(),
            root_owned=True,
            reason="",
            exists=True,
        )
        challenge = (await rig.challenge(action="loosen")).json()["challenge"]
        signature = rig.sign(action="loosen", request_id="", challenge=challenge)
        reply = await rig.command(
            {
                "op": "slash_result",
                "command": "approvals",
                "args": "auto",
                "images": [],
                "operator_sig": signature,
                "operator_key_id": device_id,
                "operator_cert": rig.certificate,
            }
        )
        assert reply.status_code != 200, reply.text
        assert rig.live.handle._auto_approve is False, "a revoked device loosened the gate"
    finally:
        await rig.aclose()


@pytest.mark.asyncio
async def test_a_paired_phone_is_offered_the_loosening_in_its_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The report may not deny a command the connection could carry out.

    ``_connection_may_loosen`` used to answer ``local`` only, with a comment
    naming this as the single line stage D would widen. The phone declares its
    PAIRED certificate on the relay's auth frame, the runtime verifies it under
    the anchor, and the ``/approvals`` report then offers ``ask|auto`` — instead of
    sending the phone after a window it does not have, which was the dead end
    (UX round 2, U7/U9).

    Asserted on the PREDICATE rather than on the rendered sentence, and that is
    deliberate: the predicate is the widening, and the sentence it selects is
    pinned separately by ``test_the_capability_name_appears_only_where_it_has_to``
    (which drives the same sentence through both branches). Two pins, one fact.
    """
    from local_operator.harness.approval import approvals_default_notice

    rig = await _phone_over_relay(tmp_path, monkeypatch)
    try:
        remote = [c for c in rig.live.runtime._clients.values() if c.locality == "remote"]
        assert remote, "the relay's connection is not on the runtime"
        conn = remote[0]
        assert conn.device_certificate, "the relay declared no certificate"
        frame = {"op": "slash_result", "command": "approvals", "args": "default auto", "images": []}
        assert rig.live.runtime._connection_may_loosen(frame, conn) is True

        # AND IT IS THE CERTIFICATE THAT DID IT: the same connection with the
        # declaration removed is the conservative branch, so the widening is not
        # ``remote`` becoming unconditionally optimistic.
        conn.device_certificate = ""
        # ``is not True`` rather than ``is False``: this predicate never answers
        # ``False`` — a follower and a capable console must not be
        # indistinguishable by accident, so the distinction it draws is "proved"
        # versus "did not say".
        assert rig.live.runtime._connection_may_loosen(frame, conn) is not True
        offered = approvals_default_notice(may_loosen=False)
        assert "has to come from the window" not in offered, offered
        assert "needs the operator's own consent" in offered, offered
    finally:
        await rig.aclose()


def test_the_relays_own_spawn_path_registers_no_capability() -> None:
    """A SOURCE PIN on the fact the phone cell above no longer depends on.

    ``mobile/daemon.py``'s spawn builds the session runtime by hand and passes no
    ``--operator-fd``, so the relay's ``entry.operator_cap`` is always ``None``.
    Under the spawn-authority model that was a reachability GAP — the surface
    table promised a phone that could loosen and the code could not deliver it.
    Under revision 2 it is a property the design relies on, and the reason the
    device tier exists: the phone is a signer, not a capability holder.

    Pinned as a source fact rather than as a process, because the claim is about
    what the argv contains and a spawn would only show the symptom. If a future
    change DOES hand the relay a capability, this fails and the design document's
    phone row has to be re-argued deliberately.
    """
    source = (_TESTS_ROOT / "local_operator" / "mobile" / "daemon.py").read_text(encoding="utf-8")
    # Parsed rather than substring-matched: the module's PROSE explains why this
    # path has no descriptor ("it passes no `--operator-fd`"), and a naive
    # `"--operator-fd" not in source` would fail on the explanation instead of on
    # the code. An AST walk sees only what the interpreter would execute.
    tree = ast.parse(source)
    argv_literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "--operator-fd" not in argv_literals, (
        "the relay's spawn now passes an operator descriptor; the phone's authority is "
        "supposed to be spawn-independent — re-argue the design before changing this"
    )
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "OPERATOR_FD_FLAG" not in names, "the relay names the descriptor flag in code"

    # And the ONE production caller of ``remember_operator_cap`` is still the real
    # spawner, so a capability cannot be registered anywhere else by accident.
    callers: list[str] = []
    for module in sorted((_TESTS_ROOT / "local_operator").rglob("*.py")):
        text = module.read_text(encoding="utf-8")
        if (
            re.search(r"\bremember_operator_cap\s*\(", text)
            and "def remember_operator_cap" not in text
        ):
            callers.append(module.relative_to(_TESTS_ROOT).as_posix())
    assert callers == ["local_operator/session/runtime/launch.py"], callers


@pytest.mark.asyncio
async def test_a_refused_card_reply_reaches_the_pane(tmp_path: Path) -> None:
    """The third door, closed (design round 2, D9).

    A pane that may not answer a card used to press APPROVE and watch nothing
    happen: the refusal arrives AFTER its approval handler has returned, and the
    arm that treats "the owner answered first" as an ordinary race swallowed it —
    no exception, no message, the card still parked and the tool call still
    blocked. Measured once with a real client on a real socket; pinned here by
    driving that arm against the same real client and the real runtime, because
    the arm is the unit that was wrong.

    The delivery half (a parked card reaching a follower's projection) is
    exercised by the desktop-route tests in this module and by
    ``tests/unit/server/test_desktop_sessions.py``; what those could not see is
    that the reply's refusal vanished, which is what this asserts — including
    that the sentence the operator gets is the CARD's, not a command's (UX U8).
    """
    from local_operator.harness.approval import (
        mint_operator_cap,
        reset_operator_caps_for_tests,
    )
    from local_operator.mobile.types import PendingRequest
    from local_operator.session.errors import OperatorAuthorityRequired

    reset_operator_caps_for_tests()  # nobody here spawned this runtime
    live = await _serve(tmp_path, operator_cap=mint_operator_cap())
    remote = None
    try:
        remote = await _follower(tmp_path, live.record)
        asked = asyncio.Event()
        surfaced: list[BaseException] = []
        answers: list[bool] = []

        async def answer_as_the_card_does(
            tool_name: str, description: str, job_id: str | None = None
        ) -> bool:
            """The card's two keystrokes, then it waits for a third.

            Call 1 is the operator pressing ALLOW — refused by the owner. Call 2
            is the same operator pressing DENY on the card that came back, and it
            has to reach the owner with NO repaint in between (agent review round
            4, R4-1 = QA Q8 = design D17 = UX U12: the re-armed card looked
            answerable and swallowed every answer). Anything after that waits,
            which is what a card does between keystrokes — a handler that
            answered immediately forever would spin the re-armed gate.
            """
            call = len(answers)
            answers.append(call != 1)
            if call == 0:
                asked.set()
            if call >= 2:
                await asyncio.Event().wait()
            return answers[-1]

        remote.set_approval_handler(answer_as_the_card_does)
        remote.set_gate_refusal_handler(surfaced.append)
        card = await _park_a_card(live.handle)
        parked = live.handle._fold.projection.pending
        assert parked is not None and parked.kind == "approval"
        pending = PendingRequest(
            request_id=parked.request_id,
            kind="approval",
            title=parked.title,
            detail=parked.detail,
        )
        # The state a delivered gate leaves behind: this pane owns the reply, and
        # the task answering it is the one under test.
        remote._gate_key = remote._gate_identity(pending)
        remote._gate_task = asyncio.current_task()
        await asyncio.wait_for(remote._run_approval(pending), 10)
        assert asked.is_set(), "the pane's handler was never reached — the pin is vacuous"
        assert surfaced, "the refusal vanished: the pane pressed APPROVE and saw nothing"
        refusal = surfaced[0]
        assert isinstance(refusal, OperatorAuthorityRequired)
        assert "still waiting" in str(refusal)
        assert "Denying it works from here" in str(refusal)
        # The sentence is not a consolation: the card really is still parked.
        assert live.handle._fold.projection.pending is not None
        assert not card.done()

        # AND THE PANE GETS ITS CARD BACK (UX review round 3, U12). The dock card
        # resolves on the keypress, so a refused Allow used to leave the pane with
        # no card, inert keys and a blocked tool call — while the copy told the
        # operator to deny from there. The gate re-arms, which is what makes that
        # sentence true rather than aspirational.
        for _ in range(100):
            if remote._gate_task is not None and not remote._gate_task.done():
                break
            await asyncio.sleep(0.02)
        assert remote._gate_task is not None, "the refused pane never got its card back"
        assert not remote._gate_task.done(), "the re-armed gate settled without an answer"
        assert remote._gate_task is not asyncio.current_task()
        assert remote._gate_key == remote._gate_identity(
            pending
        ), "the re-armed card's key was not restored, so an answer it posts is dropped"
        for _ in range(100):
            if len(answers) == 2:
                break
            await asyncio.sleep(0.02)
        assert len(answers) == 2, answers

        # AND THE DENY IT NAMES ACTUALLY LANDS — with NO push in between. The
        # key has to come back WITH the arm: an answer posted by the re-armed
        # task is discarded before it reaches the owner if ``_gate_key`` is left
        # unset, and no repaint is owed after a refusal (agent review round 4,
        # R4-1 = QA Q8 = design D17 = UX U12). The handler above answered DENY on
        # its second call, which is that keystroke; nothing here pushes a
        # projection, so the only way the owner's card can clear is the reply
        # this gate posts.
        for _ in range(200):
            if live.handle._fold.projection.pending is None:
                break
            await asyncio.sleep(0.02)
        assert (
            live.handle._fold.projection.pending is None
        ), "the deny from the re-armed card never reached the owner"
        # A deny is ordinary: it surfaced no refusal, and the card is settled.
        assert len(surfaced) == 1, surfaced
        assert card.done()

        # THE POSITIVE CONTROL (agent review round 2, R2-4; #1291's lesson that
        # a refusal nobody can turn into an acceptance proves nothing). Same
        # client, same arm, same card — the ONLY difference is that this process
        # now holds the capability for the runtime, i.e. it is the process that
        # started it. The reply goes through and the card resolves.
        from local_operator.harness.approval import remember_operator_cap

        await remote.dispose()
        held = live.runtime._operator_cap
        assert held is not None
        remember_operator_cap(live.record.pid, held)
        remote = await _follower(tmp_path, live.record)
        assert remote._client._authority_bearing, "the console did not present its capability"
        surfaced.clear()

        async def allow_again(tool_name: str, description: str, job_id: str | None = None) -> bool:
            return True

        # A SECOND card, because the deny above settled the first: the control is
        # about the arm and the capability, not about that card.
        second = await _park_a_card(live.handle)
        remote.set_approval_handler(allow_again)
        remote.set_gate_refusal_handler(surfaced.append)
        parked_second = live.handle._fold.projection.pending
        assert parked_second is not None
        pending = PendingRequest(
            request_id=parked_second.request_id,
            kind="approval",
            title=parked_second.title,
            detail=parked_second.detail,
        )
        remote._gate_key = remote._gate_identity(pending)
        remote._gate_task = asyncio.current_task()
        await asyncio.wait_for(remote._run_approval(pending), 10)
        assert not surfaced[1:], surfaced
        assert live.handle._fold.projection.pending is None, "the card was not answered"
        assert second.done()
    finally:
        if remote is not None:
            await remote.dispose()
        await live.close(tmp_path)


@pytest.mark.asyncio
async def test_the_routed_report_speaks_for_the_connection_that_asks(tmp_path: Path) -> None:
    """Who the report is FOR, driven through the runtime (agent R3-1 = U10 = Q6).

    The first version of this asked the seam's own predicate about a frame the
    seam built, so it read a proof off a frame that had none and answered a
    constant "this connection may not loosen" — measured on production objects: a
    console that had just been told ``/approvals auto`` succeeded was then told
    by ``/approvals default auto`` that loosening "has to come from the window
    that started it", and offered retirement as an alternative.

    What the runtime can actually verify is the connection's HANDSHAKE proof, so
    the two directions are driven here on real sockets: a console offers a nonce,
    verifies the runtime's proof and presents its own; a follower does neither.
    The sentence each one gets is the one that is true for it.
    """
    from local_operator.harness.approval import handshake_proof, mint_operator_cap

    cap = mint_operator_cap()
    live = await _serve(tmp_path, operator_cap=cap)
    console = follower = None
    try:
        console = await _dial(live.record, cap=cap, client="attach")
        assert console.salt, "the console's handshake did not complete"
        capable = await _send(
            console,
            None,
            {
                "op": "slash_result",
                "command": "approvals",
                "args": "default auto",
                "images": [],
                "operator_handshake": handshake_proof(
                    cap, client_nonce=console.nonce, server_salt=console.salt
                ),
            },
        )
        capable_text = str(capable.get("data", {}).get("text", ""))
        # The console IS the window that started this session, so the report may
        # offer it both directions.
        assert "/approvals ask|auto switches this session now" in capable_text, capable_text
        assert "has to come from the window" not in capable_text, capable_text

        follower = await _dial(live.record, client="attach")
        unproved = await _send(
            follower,
            None,
            {"op": "slash_result", "command": "approvals", "args": "default auto", "images": []},
        )
        unproved_text = str(unproved.get("data", {}).get("text", ""))
        # THE CONSERVATIVE BRANCH NAMES THE REAL LEVERS, NOT THE SPAWNER (stage F).
        # It used to say "has to come from the window that started it", which under
        # revision 2 is the user-visible regression this redesign deletes: a
        # background-started runtime HAS no window that started it. The pin is on
        # the clause that identifies the branch, so the two branches stay
        # distinguishable while the remedies inside are free to move.
        assert "needs the operator's own consent" in unproved_text, unproved_text
        assert "has to come from the window" not in unproved_text, unproved_text
        assert "/approvals ask switches this session now" in unproved_text, unproved_text

        # A WRONG proof is not a proof: it names the same frame the receipt does,
        # on a connection that did not hand-shake, and must stay conservative.
        forged = await _dial(live.record, client="attach")
        try:
            reply = await _send(
                forged,
                None,
                {
                    "op": "slash_result",
                    "command": "approvals",
                    "args": "default auto",
                    "images": [],
                    "operator_handshake": handshake_proof(
                        mint_operator_cap(),
                        client_nonce=console.nonce,
                        server_salt=console.salt,
                    ),
                },
            )
            forged_text = str(reply.get("data", {}).get("text", ""))
            assert "needs the operator's own consent" in forged_text, forged_text
        finally:
            forged.close()
    finally:
        for conn in (console, follower):
            if conn is not None:
                conn.close()
        await live.close(tmp_path)


# ---------------------------------------------------------------------------
# Revision 2: the OPERATOR source, on the real socket
# ---------------------------------------------------------------------------

#: The domain the revision-2 source is exercised in. Everything here drives the
#: production seams — the real runtime, the real ``operator_challenge`` op, the
#: real signer — because the claim is about what a surface that did NOT spawn the
#: runtime can do over the wire, and a fabricated frame would prove nothing about
#: the challenge binding.


def _install_operator_key(config_root: Path) -> Any:
    """A real operator key in this process's store, plus the anchor for it.

    The ``file-only`` backend, deliberately: it is a real ES256 backend rather
    than a double, and it is the only one a test may use — the Secure Enclave
    cannot even be CREATED in a throwaway keychain (measured: every attribute
    shape returns ``errSecParam``; see ``operator/keychain.py``), and writing the
    operator's login keychain is forbidden. The staged anchor is written exactly
    as ``lop operator init`` leaves it, which is what the signer resolves its
    backend from.
    """
    from local_operator.operator.keychain import FILE_ONLY
    from local_operator.operator.sign import anchor_for_handle, create_key
    from local_operator.operator.trust import anchor_bytes, staging_path

    handle = create_key(config_root=config_root, preference=FILE_ONLY)
    anchor = anchor_for_handle(handle, label="seam-test")
    staged = staging_path(config_root)
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_bytes(anchor_bytes(anchor))
    return anchor


def _sign_for(config_root: Path, challenge: str, purpose: str, session_id: str) -> dict[str, str]:
    from local_operator.operator.sign import sign_challenge

    return sign_challenge(
        challenge=challenge,
        purpose=purpose,
        config_root=config_root,
        session_id=session_id,
    ).as_json()


async def _challenge(conn: _Conn, *, action: str, request_id: str = "") -> str:
    reply = await _send(
        conn, None, {"op": "operator_challenge", "action": action, "request_id": request_id}
    )
    # AN ``ack`` FRAME, and the op name is asserted because it is load-bearing:
    # the client's reader routes replies only for ``ack``/``error``/``result``, so
    # a novel op name here would tear down the whole connection and take the
    # caller's in-flight request with it.
    assert reply["op"] == "ack", reply
    assert reply["expires_s"] == 30, reply
    challenge = reply["challenge"]
    assert is_wire_hex(challenge), reply
    return challenge


@pytest.mark.asyncio
async def test_the_record_advertises_the_operator_signature_capability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The negotiation string a client gates its challenge request on.

    Advertised UNCONDITIONALLY, including on a host with no anchor installed:
    verification needs only the anchor's public half, so the honest answer to
    "can you check a signature" is yes everywhere, and an owner that answered
    "unsupported" would send a reader to a remedy that does not exist.
    """
    from local_operator.session.runtime.types import OPERATOR_SIGNATURE_CAPABILITY

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    live = await _serve(tmp_path, operator_cap=mint_operator_cap())
    try:
        assert OPERATOR_SIGNATURE_CAPABILITY in live.record.capabilities
    finally:
        await live.close(tmp_path)


@pytest.mark.asyncio
async def test_an_operator_signature_loosens_a_runtime_this_process_never_spawned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE CAPABILITY RESTORATION, at the seam.

    A connection with NO spawn capability — the state an attached pane, the
    desktop backend for a session it did not engage, the CLI on a
    background-started run and (stage D) the phone are all in — loosens the gate
    by signing the runtime's own per-action challenge. The gate really moves: the
    assertion is on the next decision parking nothing, not on the flag alone.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    reset_operator_caps_for_tests()  # nobody here spawned this runtime
    anchor = _install_operator_key(config_dir())
    live = await _serve(tmp_path, operator_cap=mint_operator_cap(), operator_anchor=anchor)
    try:
        conn = await _dial(live.record)  # no capability, no handshake
        challenge = await _challenge(conn, action="loosen")
        signature = _sign_for(config_dir(), challenge, "loosen", live.record.session_id)
        reply = await _send(
            conn,
            None,
            {
                "op": "slash_result",
                "command": "approvals",
                "args": "auto",
                "images": [],
                "operator_sig": signature["sig"],
                "operator_key_id": signature["key_id"],
            },
        )
        assert reply["op"] == "result", reply
        assert live.handle._auto_approve is True, "the signature did not loosen the gate"
        assert await live.handle._approval_gate("bash", "rm -rf build/") is True
        assert live.handle._fold.projection.pending is None
        conn.close()
    finally:
        await live.close(tmp_path)


@pytest.mark.asyncio
async def test_a_replayed_signature_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A captured signature has exactly ONE use, because its challenge is spent.

    Replay is refused as a category rather than case by case: the challenge entry
    is popped by the first frame that presents a signature for it, before the
    signature is even checked, so the second presentation finds nothing to verify
    against and is refused. The gate must be UNMOVED afterwards — a refusal that
    still loosened would be the worst of both.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    reset_operator_caps_for_tests()
    anchor = _install_operator_key(config_dir())
    live = await _serve(tmp_path, operator_cap=mint_operator_cap(), operator_anchor=anchor)
    try:
        conn = await _dial(live.record)
        challenge = await _challenge(conn, action="loosen")
        signature = _sign_for(config_dir(), challenge, "loosen", live.record.session_id)
        frame = {
            "op": "slash_result",
            "command": "approvals",
            "args": "auto",
            "images": [],
            "operator_sig": signature["sig"],
            "operator_key_id": signature["key_id"],
        }
        first = await _send(conn, None, dict(frame))
        assert first["op"] == "result", first
        assert live.handle._auto_approve is True

        # Put the gate back so the SECOND frame has something to change: if the
        # replay were honoured, this assertion below would catch it rather than
        # being masked by the state the first frame left.
        live.handle._auto_approve = False
        second = await _send(conn, None, dict(frame))
        assert second["op"] == "error", second
        assert second.get("error_code") == "operator_authority_required", second
        assert live.handle._auto_approve is False, "a replayed signature moved the gate"
        conn.close()
    finally:
        await live.close(tmp_path)


@pytest.mark.asyncio
async def test_a_signature_is_bound_to_the_action_it_was_minted_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A challenge asked for one action cannot authorise the other.

    The ``action`` is derived from the FRAME by the runtime and bound into the
    signed message, so a client cannot ask for an ``approve`` challenge and
    present it on a loosening command — and the challenge is keyed by
    ``(action, request_id)``, so there is not even an entry to find.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    reset_operator_caps_for_tests()
    anchor = _install_operator_key(config_dir())
    live = await _serve(tmp_path, operator_cap=mint_operator_cap(), operator_anchor=anchor)
    try:
        conn = await _dial(live.record)
        challenge = await _challenge(conn, action="approve")
        signature = _sign_for(config_dir(), challenge, "approve", live.record.session_id)
        reply = await _send(
            conn,
            None,
            {
                "op": "slash_result",
                "command": "approvals",
                "args": "auto",
                "images": [],
                "operator_sig": signature["sig"],
                "operator_key_id": signature["key_id"],
            },
        )
        assert reply["op"] == "error", reply
        assert live.handle._auto_approve is False, "an approve signature loosened the gate"
        conn.close()
    finally:
        await live.close(tmp_path)


@pytest.mark.asyncio
async def test_a_forged_device_certificate_cannot_approve_a_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The device tier's forgery case, end to end: a certificate with no operator signature.

    The attacker's certificate is not malformed — it is a correct envelope over
    their OWN real P-256 key — and it dies where it must: ``verify_device_cert``
    checks an operator signature the attacker cannot produce, so the card stays
    parked and the next gated call parks again.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    reset_operator_caps_for_tests()
    from local_operator.operator.keychain import FileKeyBackend
    from local_operator.operator.verify import DeviceCert

    anchor = _install_operator_key(config_dir())
    live = await _serve(tmp_path, operator_cap=mint_operator_cap(), operator_anchor=anchor)
    try:
        parked = await _park_a_card(live.handle)
        pending = live.handle._fold.projection.pending
        assert pending is not None
        attacker = FileKeyBackend(config_dir() / "attacker.pem").create()
        forged = DeviceCert(
            device_id="stolen-phone",
            spki=attacker.spki,
            label="not the operator's device",
            issued_at=int(time.time()) - 5,
            not_after=int(time.time()) + 3600,
        ).encode(signature=b"\x30\x06\x02\x01\x01\x02\x01\x01")

        conn = await _dial(live.record)
        challenge = await _challenge(conn, action="approve", request_id=pending.request_id)
        signature = _sign_for(config_dir(), challenge, "approve", live.record.session_id)
        reply = await _send(
            conn,
            None,
            {
                "op": "approval_answer",
                "request_id": pending.request_id,
                "approved": True,
                "operator_sig": signature["sig"],
                "operator_key_id": signature["key_id"],
                "operator_cert": forged,
            },
        )
        assert reply["op"] == "error", reply
        assert reply.get("error_code") == "operator_authority_required", reply
        assert live.handle._fold.projection.pending is not None, "the forged card was answered"
        assert not parked.done()
        conn.close()
    finally:
        await live.close(tmp_path)


@pytest.mark.asyncio
async def test_a_challenge_cannot_be_spent_on_another_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Per-CONNECTION, like the capability proof it sits beside.

    A relay — or an attacker who merely read the record and dialled its own
    socket — can have a challenge minted for ITSELF, but the signature harvested
    on one connection is presented on another with no challenge behind it. That is
    what stops the relay becoming a minting service even before stage D narrows
    its body scrub.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    reset_operator_caps_for_tests()
    anchor = _install_operator_key(config_dir())
    live = await _serve(tmp_path, operator_cap=mint_operator_cap(), operator_anchor=anchor)
    try:
        # ATTACH, not the default daemon kind: the runtime keeps exactly ONE
        # daemon connection and REPLACES it on a second dial (measured: the first
        # dial is dropped with reason "daemon replaced"), which would make this
        # test read a torn-down socket rather than a cross-connection refusal.
        # Attach connections multiplex up to ``ATTACH_MAX_CLIENTS``, which is the
        # shape a second surface actually has.
        first = await _dial(live.record, client="attach")
        second = await _dial(live.record, client="attach")
        challenge = await _challenge(first, action="loosen")
        signature = _sign_for(config_dir(), challenge, "loosen", live.record.session_id)
        reply = await _send(
            second,
            None,
            {
                "op": "slash_result",
                "command": "approvals",
                "args": "auto",
                "images": [],
                "operator_sig": signature["sig"],
                "operator_key_id": signature["key_id"],
            },
        )
        assert reply["op"] == "error", reply
        assert live.handle._auto_approve is False
        first.close()
        second.close()
    finally:
        await live.close(tmp_path)


@pytest.mark.asyncio
async def test_a_connection_to_a_runtime_with_no_anchor_is_refused_with_the_typed_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No anchor means no operator source, and the refusal is the typed one.

    The runtime must keep serving a host that has not run ``lop operator
    install`` — that is a fresh install — and the answer to a signature offered
    there is a refusal, not a crash and not a silent pass.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    reset_operator_caps_for_tests()
    anchor = _install_operator_key(config_dir())
    live = await _serve(tmp_path, operator_cap=mint_operator_cap(), operator_anchor=None)
    try:
        conn = await _dial(live.record)
        challenge = await _challenge(conn, action="loosen")
        signature = _sign_for(config_dir(), challenge, "loosen", live.record.session_id)
        reply = await _send(
            conn,
            None,
            {
                "op": "slash_result",
                "command": "approvals",
                "args": "auto",
                "images": [],
                "operator_sig": signature["sig"],
                "operator_key_id": signature["key_id"],
            },
        )
        assert reply["op"] == "error", reply
        assert reply.get("error_code") == "operator_authority_required", reply
        assert live.handle._auto_approve is False
        conn.close()
    finally:
        await live.close(tmp_path)
        del anchor


@pytest.mark.asyncio
async def test_the_report_offers_loosening_to_a_local_connection_that_can_sign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sentence half of the capability restoration (revision 2, §3).

    ``_connection_may_loosen`` decides which remedy a report may name, and under
    revision 1 it answered from the spawn capability alone — so a pane attached
    to a runtime another process started was told loosening "has to come from the
    window that started it" while the revision-2 client was, at that very moment,
    about to loosen it with a signature. With an anchor installed and a LOCAL
    connection, the report must offer the loosening route; without one it must
    keep the conservative sentence, because there is genuinely nothing to sign
    with and naming a route that cannot work is the dead end UX round 2 removed.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    reset_operator_caps_for_tests()
    anchor = _install_operator_key(config_dir())
    lived = await _serve(tmp_path, operator_cap=mint_operator_cap(), operator_anchor=anchor)
    bare = await _serve(tmp_path, operator_cap=mint_operator_cap(), operator_anchor=None)
    try:
        for live, expect_offer in ((lived, True), (bare, False)):
            conn = await _dial(live.record, client="attach")
            try:
                reply = await _send(
                    conn,
                    None,
                    {
                        "op": "slash_result",
                        "command": "approvals",
                        "args": "default auto",
                        "images": [],
                    },
                )
                text = str(reply.get("data", {}).get("text", ""))
                offered = "needs the operator's own consent" not in text
                assert offered is expect_offer, (expect_offer, text)
                # And the one remedy that is true on BOTH is named either way. The
                # sentence is spelled differently in the two cases — `ask|auto`
                # when auto is offered, `ask` alone when it is not — so the pin is
                # on the part that does not move, plus the presence of `auto`.
                assert "/approvals ask" in text, text
                assert ("ask|auto switches this session now" in text) is expect_offer, text
            finally:
                conn.close()
    finally:
        await bare.close(tmp_path)
        await lived.close(tmp_path)
