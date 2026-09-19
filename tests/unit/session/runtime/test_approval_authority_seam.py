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
    OPERATOR_CAP_REQUIRED_NOTICE,
    handshake_proof_ok,
    mint_operator_cap,
    operator_cap_for,
    operator_nonce,
    remember_operator_cap,
    request_proof,
    reset_operator_caps_for_tests,
)
from local_operator.paths import config_dir
from local_operator.session.runtime import launch as launch_module
from local_operator.session.runtime import registry
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.runtime.serving import ServingSessionHandle
from tests.unit.session.runtime.test_serving import FakeSession

_TESTS_ROOT = Path(__file__).resolve().parents[4]

#: The sentence's opening, matched as a substring so the assertion is about the
#: refusal rather than about the exact wrapping of one constant.
_REFUSAL = "tool approvals stay at ask"


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


async def _serve(tmp_path: Path, *, operator_cap: bytes | None) -> _Live:
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
    runtime = RuntimeServer(handle, kind="tui", operator_cap=operator_cap)
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
        # EOF, not a crash, and not a silent no-op.
        assert out.count(_REFUSAL) == 2, out
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
    doc is the honest statement of what this does not cover). Three clauses are
    load-bearing: where to type it, how to make it true for new sessions, and
    that tightening still works here.
    """
    copy = OPERATOR_CAP_REQUIRED_NOTICE
    # Where to type it, in words the operator can resolve (design round 1 D4:
    # "the session's console" was vocabulary to nobody).
    assert "the terminal or app window that started this session" in copy
    # The mechanism that works when there IS no such window: let the runtime go
    # idle and re-open the session here, which makes THIS window the one that
    # starts the next runtime (UX round 1 U2).
    assert "go idle" in copy and "open the session again in this window" in copy
    # The remedies that are true for the NEXT session, and where they live.
    assert "--yolo" in copy
    assert "tool_approval_mode: auto" in copy
    # What still works from here.
    assert "/approvals ask" in copy
    # NOT `/approvals default ...`: the runtime refuses that command from the
    # very pane this notice is printed in (UX round 1 U1), so a remedy that
    # cannot work where it is read must not be in it.
    assert "/approvals default" not in copy
    # No promise of a boundary this host may not have (see the design doc's
    # residual section — the copy must not overclaim).
    assert re.search(r"safe|secure|protected|cannot be read", copy) is None
    # Not a visual twin of #1291's notice, which is about a config write
    # arriving from outside this session rather than a command typed where the
    # gate is not owned (design round 1 D4).
    from local_operator.harness.approval import LOOSENING_REFUSED_NOTICE

    assert copy.split(":")[0] != LOOSENING_REFUSED_NOTICE.split(":")[0]


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

        # (3) THE GUARANTEE IS REPORTED IN THE CHILD'S OWN LOG.
        text = detachment._log_text(config_dir)
        assert "operator capability boundary on this host" in text, text[-2000:]
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
        assert OPERATOR_CAP_REQUIRED_NOTICE in str(refused.value)
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
        "local_operator/session/runtime/launch.py",
        "local_operator/session/runtime/process.py",
        "local_operator/session/runtime/server.py",
        "local_operator/tui/app.py",
    }
)


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
        # ...and the guarantee level IS reported there, so the file is a real
        # artifact of this child rather than an empty one this assertion passes on.
        assert "operator capability boundary on this host" in log_text, log_text[-2000:]

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
