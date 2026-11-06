"""What became of a create's FIRST TURN — carried instead of assumed.

WHAT THIS FILE EXISTS TO PIN. ``lop network sessions --peer <device> --create
--prompt "…"`` mints a session on that device and admits the prompt there. The
admission is a durable append, NOT the turn: ``serving.ServingSessionHandle.
prompt`` returns its receipt at the append and the turn runs behind it. So the
create's answer said "prompt admitted" whether the turn then succeeded or died a
second later on a model the requesting device cannot see — and the person who
typed the create read a silent conversation for as long as they were willing to
poll.

MEASURED, on a real peer (2026-11-06): the user row landed at 1.0 s, the turn's
provider failure at 2.2 s, and the creating device was told none of it while a
``--send`` to the same session DID report "ran the turn and it failed". That
difference is a viewer stream (``attached.prompt_and_wait``) versus a create,
which has none — so the OWNER has to settle its own turn and say what happened.

THE FACTS ASSERTED HERE, and the cell that fails without each:

1. the runtime's ``prompt`` op, asked with ``wait_complete``, answers a FAILED
   turn as a typed error carrying the cause (``turn_failed``) instead of its
   admission ack — ``test_a_failed_first_turn_crosses_as_its_own_code``;
2. the owning relay reads that as ``admitted: true`` plus a ``turn`` block whose
   state is ``failed`` and whose detail is the cause — it must not become an
   admission refusal — ``test_a_failed_turn_is_not_reported_as_a_refusal``;
3. the create's reply carries that block —
   ``test_the_create_reply_carries_the_turn_block``;
4. the receipt prints it — ``test_the_receipt_prints_the_failure``;
5. NEGATIVE CONTROL: a create with no prompt starts no turn, reports no turn,
   and still says the session exists cold —
   ``test_a_promptless_create_starts_no_turn``.

The rigs are the production ones: a real ``Session`` behind the real
``ServingSessionHandle`` and ``RuntimeServer`` for the wire cell, the real
``_prompt_on`` over a scripted dial for the relay cells, and the real
``_cmd_sessions`` handler for the receipt cell.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from local_operator.network import identity, relay
from local_operator.session.errors import FirstTurnFailed
from local_operator.session.runtime.server import RuntimeServer, _ClientConn
from local_operator.session.runtime.serving import ServingSessionHandle
from tests.e2e.harness import build_session

# ---------------------------------------------------------------------------
# 1. The wire: a failed first turn is a typed error, not an admission ack
# ---------------------------------------------------------------------------


def _wire_rig(root: Path) -> tuple[RuntimeServer, list[dict[str, Any]], _ClientConn, Any]:
    """A REAL session under the production handle and server, writes captured.

    ``MockClient``'s ``[refuse]`` marker is what makes the turn fail on the real
    path (a provider refusal, the same event a 401 or a rate limit produces), so
    nothing here substitutes for the product's own turn machinery.
    """
    from local_operator.providers.clients import MockClient

    directory = root / "sessions" / "sess00000001"
    directory.mkdir(parents=True, exist_ok=True)
    session = build_session(directory, MockClient().stream, cwd=root)
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(root))
    server = RuntimeServer(handle, kind="tui")
    sent: list[dict[str, Any]] = []

    async def capture(_target: Any, frame: dict[str, Any]) -> None:
        sent.append(frame)

    server._send_to = capture  # type: ignore[assignment]
    conn = _ClientConn(writer=cast(Any, object()), kind=cast(Any, "attach"))
    server._clients[id(conn.writer)] = conn
    return server, sent, conn, session


@pytest.mark.asyncio
async def test_a_failed_first_turn_crosses_as_its_own_code(tmp_path: Path) -> None:
    """The op the create's prompt rides must answer the FAILURE, not the append.

    Without ``wait_complete`` reaching the handle this reply is the string
    "prompt admitted" and the failure exists only on this device — the silence
    this file is about.
    """
    server, sent, conn, _session = _wire_rig(tmp_path)
    await server._on_request(
        {
            "op": "prompt",
            "req": 1,
            "command_id": str(uuid.uuid4()),
            "text": "Reply with exactly MESH_OK [refuse]",
            "images": [],
            "wait_complete": True,
        },
        conn,
    )
    errors = [frame for frame in sent if frame.get("op") == "error"]
    assert errors, f"a failed turn answered as if nothing went wrong: {sent}"
    # Round-trip through JSON, because the frame really goes over a socket.
    frame = cast(dict[str, Any], json.loads(json.dumps(errors[0])))
    assert frame["error_code"] == FirstTurnFailed.code, frame
    # THE CAUSE, not just the verdict: a model that refused and a login that
    # expired need different actions from the person reading the receipt.
    assert "I can't help with that request." in frame["message"], frame["message"]


@pytest.mark.asyncio
async def test_a_settling_owner_marks_its_ack_so_skew_cannot_masquerade(tmp_path: Path) -> None:
    """A settled turn keeps the sentence AND adds a field no old owner can emit.

    The sentence alone is not evidence: ``"prompt admitted"`` is what an owner
    that answered at the durable append also says, byte for byte. ``settled`` is
    the whole discriminator the caller may key "the turn finished" off (agent
    review round 1, BLOCKER-1).
    """
    server, sent, conn, _session = _runtime_ok_rig(tmp_path)
    await server._on_request(
        {
            "op": "prompt",
            "req": 1,
            "command_id": str(uuid.uuid4()),
            "text": "Reply with exactly MESH_OK",
            "images": [],
            "wait_complete": True,
        },
        conn,
    )
    replies = [frame for frame in sent if frame.get("op") != "error"]
    assert replies, sent
    ack = cast(dict[str, Any], json.loads(json.dumps(replies[0])))
    assert ack["detail"] == "prompt admitted", ack
    assert ack["settled"] is True, ack


def _runtime_ok_rig(root: Path) -> tuple[RuntimeServer, list[dict[str, Any]], _ClientConn, Any]:
    from local_operator.providers.clients import MockClient

    directory = root / "sessions" / "sess00000002"
    directory.mkdir(parents=True, exist_ok=True)
    session = build_session(directory, MockClient().stream, cwd=root)
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(root))
    server = RuntimeServer(handle, kind="tui")
    sent: list[dict[str, Any]] = []

    async def capture(_target: Any, frame: dict[str, Any]) -> None:
        sent.append(frame)

    server._send_to = capture  # type: ignore[assignment]
    conn = _ClientConn(writer=cast(Any, object()), kind=cast(Any, "attach"))
    server._clients[id(conn.writer)] = conn
    return server, sent, conn, session


# ---------------------------------------------------------------------------
# 2/3. The owning relay: the outcome is a turn fact, never a refusal
# ---------------------------------------------------------------------------


class _ScriptedDial:
    """One dial whose exchange answers ``reply`` (or nothing) and reports ``eof``."""

    def __init__(self, reply: dict[str, Any] | None, *, eof: bool = False) -> None:
        self.reply = reply
        self.eof = False
        self._eof_after = eof
        self.sent: list[dict[str, Any]] = []

    def exchange(self, frame: dict[str, Any], timeout_s: float = 0.0) -> Any:
        self.sent.append(frame)
        # The real dial sets ``eof`` while it reads; a closed socket is the one
        # case that must NOT read as "admitted and still running".
        self.eof = self._eof_after
        return self.reply

    def close(self) -> None:
        return None


def _server(root: Path) -> relay.RelayServer:
    return relay.RelayServer(
        root=root, settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1")
    )


def _dialed(
    server: relay.RelayServer, reply: dict[str, Any] | None, *, eof: bool = False
) -> _ScriptedDial:
    dial = _ScriptedDial(reply, eof=eof)
    server._dial_owned = lambda *_args, **_kwargs: dial  # type: ignore[assignment]
    return dial


def test_a_failed_turn_is_not_reported_as_a_refusal(root: Path) -> None:
    """``admitted`` is about the MESSAGE; the outcome is ``turn``.

    Reading a failed turn as ``admitted: false`` would tell the user their words
    were dropped — the very mistake in the other direction.
    """
    server = _server(root)
    _dialed(
        server,
        {
            "op": "error",
            "req": 1,
            "message": "The admitted loop turn did not complete: 'anthropic' is rate-limited",
            "error_code": FirstTurnFailed.code,
        },
    )
    outcome = server._prompt_on("sess00000001", "hello", None)  # noqa: SLF001
    assert outcome["admitted"] is True, outcome
    assert outcome["state"] == "failed", outcome
    assert "rate-limited" in outcome["detail"], outcome


def test_a_settled_turn_reports_settled_only_on_the_owners_marker(root: Path) -> None:
    """The OWNER's ``settled`` marker is what may claim completion.

    This is the arm BLOCKER-1 was about: a reply that carries the sentence but no
    marker is an owner that answered at the append, and it must read ``running``.
    """
    server = _server(root)
    _dialed(server, {"op": "ack", "req": 1, "detail": "prompt admitted", "settled": True})
    outcome = server._prompt_on("sess00000001", "hello", None)  # noqa: SLF001
    assert (outcome["admitted"], outcome["state"]) == (True, "settled"), outcome


def test_an_older_owner_that_acks_at_the_append_is_running_never_ran(root: Path) -> None:
    """VERSION SKEW, pinned to the exact answer: an old peer is NOT a completed turn.

    An owner from before ``wait_complete`` answers ``{"op": "ack", "detail":
    "prompt admitted"}`` — byte-identical to a settling owner's success. Keying
    "ran" off this side's request printed ``turn: completed`` for a turn nobody
    waited for (agent review round 1, BLOCKER-1), so the assertion is the exact
    state, not a set of acceptable ones.
    """
    server = _server(root)
    dial = _dialed(server, {"op": "ack", "req": 1, "detail": "prompt admitted"})
    outcome = server._prompt_on("sess00000001", "hello", None)  # noqa: SLF001
    assert outcome == {
        "admitted": True,
        "state": "running",
        "detail": "prompt admitted",
    }, outcome
    # ...and it did ask, so the field travelled even to an owner that ignores it.
    assert dial.sent and dial.sent[0]["wait_complete"] is True, dial.sent


def test_a_turn_that_outlives_the_budget_is_running_not_refused(root: Path) -> None:
    """An owner that never answered is NOT an owner that refused.

    A ``wait_complete`` reply arrives only after the turn, so silence means the
    turn outlived this device's op budget — the message is durable and still
    going. The old reading called that "did not admit", which is the same class
    of untruth as the silence itself.
    """
    server = _server(root)
    _dialed(server, None)
    outcome = server._prompt_on("sess00000001", "hello", None)  # noqa: SLF001
    assert (outcome["admitted"], outcome["state"]) == (True, "running"), outcome


def test_a_dead_owner_socket_is_unreported_not_admitted_running(root: Path) -> None:
    """A closed socket is not a timeout (agent review round 1, MINOR-4).

    ``OwnerDial.exchange`` answers ``None`` for both, and ``dial.eof`` is the one
    thing that distinguishes them: claiming durable admission for an owner that
    died mid-turn asserts something nobody observed.
    """
    server = _server(root)
    _dialed(server, None, eof=True)
    outcome = server._prompt_on("sess00000001", "hello", None)  # noqa: SLF001
    assert outcome["admitted"] is False, outcome
    assert outcome["state"] == "unreported", outcome
    assert "unknown" in outcome["detail"], outcome


def test_the_create_reply_carries_the_turn_block(root: Path) -> None:
    """The peer's answer must name what became of the turn it ran for us."""
    identity.mint(root, name="probe-node")
    server = _server(root)
    server._engage_locally = lambda *_a, **_k: ""  # type: ignore[assignment]
    server._prompt_on = lambda *_a, **_k: {  # type: ignore[assignment]
        "admitted": True,
        "state": "failed",
        "detail": "The admitted loop turn did not complete: 'anthropic' is rate-limited",
    }
    detail = server._op_session_create(  # noqa: SLF001
        _link(),
        {
            "op": "net_session_create",
            "req": 7,
            "locality": "remote",
            "cwd": str(root),
            "prompt": "Reply with exactly MESH_OK",
        },
    )
    assert detail["admitted"] is True, detail
    assert detail["turn"]["state"] == "failed", detail
    assert "rate-limited" in detail["turn"]["detail"], detail


def test_a_promptless_create_starts_no_turn(root: Path) -> None:
    """NEGATIVE CONTROL: no prompt, no turn, and nothing claims otherwise.

    A promptless create warms the runtime in the background and answers before
    it is up (``_warm_after_create``), so the reply must carry no ``turn`` block
    at all — a consumer seeing one would go looking for a turn that was never
    asked for.
    """
    identity.mint(root, name="probe-node")
    server = _server(root)
    server._warm_after_create = lambda *_a, **_k: None  # type: ignore[assignment]
    detail = server._op_session_create(  # noqa: SLF001
        _link(),
        {"op": "net_session_create", "req": 7, "locality": "remote", "cwd": str(root)},
    )
    assert detail["admitted"] is False, detail
    assert detail.get("warming") is True, detail
    assert "turn" not in detail, detail


def _link() -> Any:
    """A stub link carrying exactly what ``_op_session_create`` reads."""
    peer = "d_" + "a" * 32
    network = "n_0123456789abcdef0123456789abcdef"
    return SimpleNamespace(
        device_id=peer,
        network_id=network,
        epoch=1,
        context=SimpleNamespace(
            capabilities=frozenset({"prompt"}),
            device_id=peer,
            network_id=network,
            epoch=1,
        ),
    )


# ---------------------------------------------------------------------------
# 4. The receipt the person reads
# ---------------------------------------------------------------------------


def _args(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> Any:
    from local_operator.network import cli as network_cli

    monkeypatch.setattr(
        network_cli,
        "_relay_answer",
        lambda *_a, **_k: {"ok": True, "session_id": "sess00000001", **overrides},
    )
    return SimpleNamespace(
        peer="build-box",
        create=True,
        name="",
        prompt="Reply with exactly MESH_OK",
        cwd="",
        model="",
        profile="",
        agent="",
        agent_id="",
        team="",
        effort="",
        yolo=False,
        json=False,
        **{key: "" for key in ("stop", "engage", "send", "steer", "slash", "peek")},
    )


def test_the_receipt_prints_the_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The one line that ends the silence: the turn failed, and why."""
    from local_operator.network import cli as network_cli

    args = _args(
        monkeypatch,
        admitted=True,
        detail="",
        turn={
            "state": "failed",
            "detail": "The admitted loop turn did not complete: 'anthropic' is rate-limited",
        },
    )
    assert network_cli._cmd_sessions(args) == 0  # noqa: SLF001
    printed = capsys.readouterr().out
    assert "turn failed on build-box" in printed, printed
    assert "rate-limited" in printed, printed


def test_the_receipt_says_a_running_turn_is_running(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Running says it cannot see the end — and names no read that cannot show one.

    ``--peek`` was offered here until agent review round 1 (MAJOR-2): over the
    real journal a failed turn's assistant row is ``content: null`` and its cause
    is an incident record with no ``.text``, so the window filters both and shows
    the user their own message. The receipt must not point at that dead end.
    """
    from local_operator.network import cli as network_cli

    args = _args(monkeypatch, admitted=True, detail="", turn={"state": "running", "detail": ""})
    assert network_cli._cmd_sessions(args) == 0  # noqa: SLF001
    printed = capsys.readouterr().out
    assert "running on build-box" in printed, printed
    assert "cannot see how it ends" in printed, printed
    assert "--peek" not in printed, printed


def test_the_receipt_never_completes_a_turn_the_owner_did_not_settle(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """BLOCKER-1's user-visible half: no ``completed`` without the owner's marker.

    The old-peer outcome is what ``_prompt_on`` now returns for that ack
    (``running``); this pins the receipt's half of the pair, so neither the relay
    nor the printer can reintroduce "turn: completed" for a turn nobody waited
    for.
    """
    from local_operator.network import cli as network_cli

    args = _args(
        monkeypatch,
        admitted=True,
        detail="prompt admitted",
        turn={"state": "running", "detail": ""},
    )
    assert network_cli._cmd_sessions(args) == 0  # noqa: SLF001
    printed = capsys.readouterr().out
    assert "completed" not in printed, printed


def test_the_receipt_says_completed_when_the_owner_settled(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from local_operator.network import cli as network_cli

    args = _args(monkeypatch, admitted=True, detail="", turn={"state": "settled", "detail": ""})
    assert network_cli._cmd_sessions(args) == 0  # noqa: SLF001
    printed = capsys.readouterr().out
    assert "turn: completed on build-box" in printed, printed


def test_a_promptless_receipt_prints_no_turn(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from local_operator.network import cli as network_cli

    args = _args(monkeypatch, admitted=False, warming=True, detail="")
    args.prompt = ""
    assert network_cli._cmd_sessions(args) == 0  # noqa: SLF001
    printed = capsys.readouterr().out
    assert "no prompt sent" in printed, printed
    assert "turn" not in printed, printed
