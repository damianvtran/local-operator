"""The runtime stamps who answered a queued ask; the frame's ``by`` is never trusted.

THE DEFECT (pre-existing on main @ 49af370). ``ask_respond``/``ask_revise``/
``ask_decline``/``ask_dismiss`` copied ``by`` from the frame, and the legacy
``ask_answer`` mirror stamped ``"mirror"`` for any caller. The control socket
authenticates only with the record's ``control_key`` (readable by any same-uid
process, a model's ``bash`` included) and a mesh peer with ``prompt`` reaches
the same arm through ``net_forward``. ``desktop`` and ``mirror`` are in
``guardrails.provenance.TRUSTED_ASK_SURFACES``, and an approved trusted ask that
names a path settles a dangerous delete BEFORE any judge — so a forged answer
became delete consent with zero judge calls (scenarios D and E of the probe).

THE FIX under test: a trusted label only for a frame that proved the per-
connection operator capability (``request_proof_ok``); otherwise ``remote`` or
``unverified``. The answer still settles the ask; it is just not consent.

Every case drives the REAL ``RuntimeServer._dispatch`` arm, the REAL
``ServingSessionHandle`` methods, the REAL ``AskQueue`` log, and the REAL
``check_tool_call`` on ``rm -rf <home>/victim`` (HIGH: a top-level home
directory) with a judge that always answers BLOCK and counts its calls. Nothing
is executed: the guard is only evaluated.
"""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import pytest

from local_operator.asks import store
from local_operator.asks.queue import AskQueue
from local_operator.asks.store import mirror_request_id
from local_operator.asks.surfaces import stamp_answer_surface
from local_operator.guardrails.check import CallContext, GuardSettings, check_tool_call
from local_operator.harness.approval import (
    ASK_ANSWER_OPS,
    AUTHORITY_OPS,
    mint_operator_cap,
    operator_nonce,
    request_proof,
)
from local_operator.session.runtime.server import RuntimeServer, _ConnectionAuthority
from local_operator.session.runtime.serving import ServingSessionHandle


class _QueueOwnerStub:
    """The queue's session back-reference; delivery is not under test."""

    def __getattr__(self, name: str) -> Any:
        return lambda *args, **kwargs: None

    def _spawn_background(self, coro: Any) -> None:
        # Close rather than drop the scheduled reconcile: delivery is not under
        # test, and an un-awaited coroutine only adds a RuntimeWarning.
        coro.close()


class _Session:
    """The four session methods the serving handle's ask arms call, on a REAL queue."""

    def __init__(self, queue: AskQueue) -> None:
        self._queue = queue

    def ask_queue(self) -> AskQueue:
        return self._queue

    def respond_ask(self, ask_id: str, answers: Any, by: str = "unknown", **_: Any) -> Any:
        return self._queue.respond(ask_id, answers, by=by)

    def answer_ask_question(self, ask_id: str, key: str, values: Any, by: str = "unknown") -> Any:
        return self._queue.answer_one(ask_id, key, values, by=by)

    def decline_ask(self, ask_id: str, by: str = "unknown") -> Any:
        return self._queue.decline(ask_id, by=by)

    def dismiss_ask(self, ask_id: str, by: str = "unknown", reason: str | None = None) -> Any:
        return self._queue.dismiss(ask_id, by=by, reason=reason or "moot")

    async def reconcile_asks(self) -> None:
        return None


class _Rig:
    """One session's real ask log, a real serving handle over it, and a victim dir."""

    def __init__(self, home: Path, config: Path, tag: str) -> None:
        self.home = home
        self.config = config
        self.session_id = f"stamp{tag}"
        self.victim = home / f"victim-{tag}"
        self.victim.mkdir()
        (self.victim / "f.txt").write_text("x")
        self.scratchpad = config / "sessions" / self.session_id / "scratchpad"
        self.scratchpad.mkdir(parents=True)
        self.queue = AskQueue(_QueueOwnerStub(), config_dir=config, session_id=self.session_id)
        receipt = self.queue.enqueue(
            [
                {
                    "id": "q1",
                    "question": f"Delete {self.victim}?",
                    "options": [{"label": "Yes"}, {"label": "No, keep it"}],
                }
            ],
            "1h",
        )
        assert receipt.get("ok"), receipt
        self.ask_id = str(self.queue.records()[-1]["ask_id"])
        handle = ServingSessionHandle.__new__(ServingSessionHandle)
        handle._session = _Session(self.queue)  # type: ignore[attr-defined]
        handle._loop = None  # type: ignore[attr-defined]
        self.handle = handle
        # ``_dispatch`` reads only ``self._handle`` on these arms.
        self.server: Any = types.SimpleNamespace(_handle=handle)

    async def dispatch(self, op: str, frame: dict[str, Any], authority: Any) -> Any:
        return await RuntimeServer._dispatch(self.server, op, {"op": op, **frame}, authority)

    def record(self) -> dict[str, Any]:
        folded = store.fold(
            store.read_events(store.session_dir(self.config, self.session_id)), store.now_ms()
        )
        return next(r for r in folded if r["ask_id"] == self.ask_id)

    async def guard(self) -> tuple[Any, int]:
        calls: list[int] = []

        async def judge(system: str, user: str) -> str:
            calls.append(1)
            return "BLOCK no consent (test judge)"

        env = {
            "HOME": str(self.home),
            "PATH": "/usr/bin:/bin",
            "LOCAL_OPERATOR_CONFIG_DIR": str(self.config),
        }
        ctx = CallContext(
            cwd=str(self.home),
            env=env,
            scratchpad=str(self.scratchpad),
            session_id=self.session_id,
            intent="Cleaning up",
        )
        outcome = await check_tool_call(
            "bash",
            {"command": f"rm -rf {self.victim}"},
            ctx,
            judge=judge,
            settings=GuardSettings(),
            forked_judge=judge,
        )
        return outcome, len(calls)


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    home = tmp_path / "home"
    config = tmp_path / "config"
    home.mkdir()
    config.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    return home, config


_REMOTE_PROMPT = _ConnectionAuthority(
    locality="remote",
    capabilities=frozenset({"prompt"}),
    peer={"device_id": "dev-synthetic-peer", "name": "peer"},
)
#: A local dial that read the record key — a model's own ``bash`` — and claims
#: the desktop surface, with no capability proof.
_LOCAL_UNPROVEN = _ConnectionAuthority(locality="local", surface="desktop", kind="attach")
_PROVED_DESKTOP = _ConnectionAuthority(
    locality="local", surface="desktop", kind="attach", operator_proved=True
)


def _assert_judged_and_blocked(outcome: Any, judge_calls: int, victim: Path) -> None:
    assert outcome is not None
    assert judge_calls >= 1, "the forged answer settled the delete before any judge"
    assert outcome.blocked is True, outcome
    assert victim.exists()


# ---------------------------------------------------------------------------
# Regression: the probe's scenarios D and E (0 judge calls, allowed, at 49af370)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scenario_d_remote_ask_respond_claiming_desktop_is_not_consent(
    home: tuple[Path, Path],
) -> None:
    rig = _Rig(*home, tag="D")
    reply = await rig.dispatch(
        "ask_respond",
        {"ask_id": rig.ask_id, "answers": {"q1": ["Yes"]}, "by": "desktop"},
        _REMOTE_PROMPT,
    )
    assert reply == "answered"
    record = rig.record()
    # The answer still settles the ask for the agent...
    assert record["status"] == "answered"
    # ...but its surface is the runtime's word, not the frame's.
    assert record["answered_by"] == {"surface": "remote"}
    outcome, judge_calls = await rig.guard()
    _assert_judged_and_blocked(outcome, judge_calls, rig.victim)


@pytest.mark.asyncio
async def test_scenario_e_unproven_mirror_answer_is_not_consent(
    home: tuple[Path, Path],
) -> None:
    rig = _Rig(*home, tag="E")
    reply = await rig.dispatch(
        "ask_answer",
        {"request_id": mirror_request_id(rig.ask_id, 0), "value": "Yes"},
        _LOCAL_UNPROVEN,
    )
    assert reply == "answered"
    record = rig.record()
    assert record["status"] == "answered"
    assert record["answered_by"] == {"surface": "unverified"}
    outcome, judge_calls = await rig.guard()
    _assert_judged_and_blocked(outcome, judge_calls, rig.victim)


@pytest.mark.asyncio
async def test_remote_mirror_answer_is_labelled_remote(home: tuple[Path, Path]) -> None:
    rig = _Rig(*home, tag="E2")
    await rig.dispatch(
        "ask_answer",
        {"request_id": mirror_request_id(rig.ask_id, 0), "value": "Yes"},
        _REMOTE_PROMPT,
    )
    assert rig.record()["answered_by"] == {"surface": "remote"}


@pytest.mark.asyncio
async def test_a_local_key_holder_cannot_claim_desktop(home: tuple[Path, Path]) -> None:
    rig = _Rig(*home, tag="L")
    await rig.dispatch(
        "ask_respond",
        {"ask_id": rig.ask_id, "answers": {"q1": ["Yes"]}, "by": "desktop"},
        _LOCAL_UNPROVEN,
    )
    assert rig.record()["answered_by"] == {"surface": "unverified"}
    outcome, judge_calls = await rig.guard()
    _assert_judged_and_blocked(outcome, judge_calls, rig.victim)


@pytest.mark.asyncio
@pytest.mark.parametrize("op", ["ask_decline", "ask_dismiss"])
async def test_decline_and_dismiss_record_the_stamped_surface(
    home: tuple[Path, Path], op: str
) -> None:
    """Not consent either way; the audit line must still name the truth."""
    rig = _Rig(*home, tag=op)
    await rig.dispatch(op, {"ask_id": rig.ask_id, "by": "desktop"}, _REMOTE_PROMPT)
    events = store.read_events(store.session_dir(rig.config, rig.session_id))
    settled = [e for e in events if e.get("kind") in ("declined", "dismissed")]
    assert settled, events
    assert "desktop" not in str(settled[-1]), settled[-1]
    assert "remote" in str(settled[-1]), settled[-1]


# ---------------------------------------------------------------------------
# The legitimate paths still lift
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_proved_desktop_answer_still_lifts(home: tuple[Path, Path]) -> None:
    rig = _Rig(*home, tag="P")
    # A frame ``by`` is ignored even when proved: the connection's surface wins.
    await rig.dispatch(
        "ask_respond",
        {"ask_id": rig.ask_id, "answers": {"q1": ["Yes"]}, "by": "tui"},
        _PROVED_DESKTOP,
    )
    assert rig.record()["answered_by"] == {"surface": "desktop"}
    outcome, judge_calls = await rig.guard()
    assert judge_calls == 0
    assert outcome is None or outcome.blocked is False, outcome


@pytest.mark.asyncio
async def test_a_proved_mirror_answer_still_lifts(home: tuple[Path, Path]) -> None:
    rig = _Rig(*home, tag="PM")
    await rig.dispatch(
        "ask_answer",
        {"request_id": mirror_request_id(rig.ask_id, 0), "value": "Yes"},
        _PROVED_DESKTOP,
    )
    assert rig.record()["answered_by"] == {"surface": "mirror"}
    outcome, judge_calls = await rig.guard()
    assert judge_calls == 0
    assert outcome is None or outcome.blocked is False, outcome


@pytest.mark.asyncio
async def test_an_in_process_tui_answer_still_lifts(home: tuple[Path, Path]) -> None:
    """The owning TUI answers its own session's ask without crossing the socket
    (``tui/app.py`` calls ``Session.respond_ask(..., by="terminal")`` directly),
    so no stamp applies and the label stays trusted."""
    rig = _Rig(*home, tag="T")
    outcome = _Session(rig.queue).respond_ask(rig.ask_id, {"q1": ["Yes"]}, by="terminal")
    assert outcome.get("ok"), outcome
    assert rig.record()["answered_by"] == {"surface": "terminal"}
    guarded, judge_calls = await rig.guard()
    assert judge_calls == 0
    assert guarded is None or guarded.blocked is False, guarded


# ---------------------------------------------------------------------------
# The proof is computed per connection, from the frame, at the reader seam
# ---------------------------------------------------------------------------


def _bare_server(cap: bytes | None) -> RuntimeServer:
    server = RuntimeServer.__new__(RuntimeServer)
    server._operator_cap = cap  # type: ignore[attr-defined]
    return server


def _conn(*, surface: str = "desktop", kind: str = "attach") -> Any:
    return types.SimpleNamespace(
        locality="local",
        capabilities=frozenset(),
        peer={},
        surface=surface,
        kind=kind,
        operator_nonce=operator_nonce(),
        operator_salt=operator_nonce(),
    )


@pytest.mark.asyncio
async def test_the_reader_seam_proves_only_a_valid_capability_proof() -> None:
    cap = mint_operator_cap()
    server = _bare_server(cap)
    seen: list[_ConnectionAuthority] = []

    async def capture(op: str, frame: dict[str, Any], authority: Any = None) -> str:
        seen.append(authority)
        return "ok"

    server._dispatch = capture  # type: ignore[method-assign]
    conn = _conn()
    good = request_proof(cap, client_nonce=conn.operator_nonce, server_salt=conn.operator_salt)
    other = _conn()
    replayed = request_proof(
        cap, client_nonce=other.operator_nonce, server_salt=other.operator_salt
    )
    forged = request_proof(
        mint_operator_cap(), client_nonce=conn.operator_nonce, server_salt=conn.operator_salt
    )
    frame = {"op": "ask_respond", "ask_id": "a-1234", "answers": {}}
    await server._dispatch_with_authority("ask_respond", {**frame, "operator_cap": good}, conn)
    await server._dispatch_with_authority("ask_respond", frame, conn)
    await server._dispatch_with_authority("ask_respond", {**frame, "operator_cap": replayed}, conn)
    await server._dispatch_with_authority("ask_respond", {**frame, "operator_cap": forged}, conn)
    # A proof on a non-ask op does not mark the authority (it is read nowhere else).
    await server._dispatch_with_authority("ping", {"op": "ping", "operator_cap": good}, conn)
    assert [a.operator_proved for a in seen] == [True, False, False, False, False]
    assert seen[0].surface == "desktop" and seen[0].kind == "attach"
    # A runtime holding no capability proves nobody.
    unheld = _bare_server(None)
    unheld._dispatch = capture  # type: ignore[method-assign]
    await unheld._dispatch_with_authority("ask_respond", {**frame, "operator_cap": good}, conn)
    assert seen[-1].operator_proved is False


@pytest.mark.asyncio
async def test_the_attach_client_presents_the_proof_on_ask_answers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The console that started the runtime keeps its answers trusted, prompt-free."""
    import asyncio
    import json

    from local_operator.mobile.attach_client import AttachClient

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))

    class _Writer:
        def __init__(self) -> None:
            self.sent: list[dict[str, Any]] = []

        def write(self, data: bytes) -> None:
            self.sent.append(json.loads(data))

        async def drain(self) -> None:
            return None

    cap = mint_operator_cap()
    client = AttachClient(lambda _projection: None, lambda _reason: None)
    client._connected = True
    writer = _Writer()
    client._writer = writer  # type: ignore[assignment]
    client._operator_cap = cap
    client._operator_nonce = operator_nonce()
    client._operator_salt = operator_nonce()
    client._authority_bearing = True
    expected = request_proof(
        cap, client_nonce=client._operator_nonce, server_salt=client._operator_salt
    )
    fields: dict[str, Any]
    for op, fields in (
        ("ask_respond", {"ask_id": "a-1", "answers": {"q1": ["Yes"]}}),
        ("ask_answer", {"request_id": "a-1.0", "value": "Yes"}),
        ("ask_decline", {"ask_id": "a-1"}),
    ):
        try:
            await asyncio.wait_for(
                client._request_frame(op, deadline_s=0.05, _operator_ask=True, **fields),
                timeout=0.2,
            )
        except (TimeoutError, asyncio.TimeoutError):
            pass
        assert writer.sent[-1]["op"] == op
        assert writer.sent[-1]["operator_cap"] == expected, writer.sent[-1]
    # Not authority-bearing (a runtime another process started): nothing attached.
    client._authority_bearing = False
    try:
        await asyncio.wait_for(
            client._request_frame(
                "ask_respond", deadline_s=0.05, _operator_ask=True, ask_id="a-1", answers={}
            ),
            timeout=0.2,
        )
    except (TimeoutError, asyncio.TimeoutError):
        pass
    assert "operator_cap" not in writer.sent[-1]


def test_the_ask_ops_are_not_authority_increasing() -> None:
    """An unproven surface may still ANSWER; only the consent label is gated."""
    assert not (ASK_ANSWER_OPS & AUTHORITY_OPS)


@pytest.mark.parametrize(
    ("proved", "surface", "locality", "kind", "mirror", "expected"),
    [
        (True, "desktop", "local", "attach", False, "desktop"),
        (True, "terminal", "local", "daemon", False, "remote"),
        (True, "desktop", "remote", "attach", False, "remote"),
        (True, "desktop", "remote", "attach", True, "remote"),
        (True, "desktop", "local", "attach", True, "mirror"),
        (True, "", "local", "attach", False, "terminal"),
        (False, "desktop", "local", "attach", False, "unverified"),
        (False, "desktop", "local", "attach", True, "unverified"),
        (False, "terminal", "local", "daemon", False, "remote"),
        (False, "terminal", "remote", "attach", True, "remote"),
    ],
)
def test_stamp_answer_surface(
    proved: bool, surface: str, locality: str, kind: str, mirror: bool, expected: str
) -> None:
    assert (
        stamp_answer_surface(
            proved=proved, surface=surface, locality=locality, kind=kind, mirror=mirror
        )
        == expected
    )


def test_profile_trust_refuses_an_unproven_label() -> None:
    from local_operator.session.session import _profile_input_class

    def _answer(surface: str) -> Any:
        return types.SimpleNamespace(
            custom_type="ask_response",
            details={"answered_by": {"surface": surface}},
            role="user",
            content="Yes",
        )

    for label in ("unverified", "remote", "mirror", "terminal"):
        assert _profile_input_class(_answer(label)) == "untrusted", label


# ---------------------------------------------------------------------------
# On the REAL socket: the production handshake decides the label
# ---------------------------------------------------------------------------


async def _serve_asks(tmp_path: Path, cap: bytes, queue: AskQueue) -> Any:
    """A real ``RuntimeServer`` over a real ``ServingSessionHandle`` whose session owns ``queue``.

    The seam module's ``_serve`` with one change: its session double gains the
    ask methods, backed by the real queue, so the frame crosses the real auth
    handshake, the real reader loop (``_dispatch_with_authority``, where the
    proof is checked) and the real dispatch arm.
    """
    import asyncio

    from local_operator.session.runtime import registry
    from tests.unit.session.runtime.test_approval_authority_seam import (
        _AttachableSession,
        _Live,
    )

    class _AskSession(_AttachableSession):
        def ask_queue(self) -> AskQueue:
            return queue

        def respond_ask(self, ask_id: str, answers: Any, by: str = "unknown", **_: Any) -> Any:
            return queue.respond(ask_id, answers, by=by)

        async def reconcile_asks(self) -> None:
            return None

    session = _AskSession()
    handle = ServingSessionHandle(
        session, asyncio.get_running_loop(), cwd=str(tmp_path), auto_approve=False
    )
    runtime = RuntimeServer(handle, kind="tui", operator_cap=cap)
    await runtime.start_in_process()
    root = Path(str(tmp_path))
    for _ in range(250):
        found = registry.scan(root)
        if found and found[0][1] == "live":
            return _Live(handle, runtime, found[0][0])
        await asyncio.sleep(0.02)
    runtime.close()
    raise AssertionError("the runtime never published a live record")


@pytest.mark.asyncio
@pytest.mark.parametrize("present_proof", [True, False])
async def test_the_real_socket_stamps_desktop_only_for_a_proved_frame(
    home: tuple[Path, Path], present_proof: bool
) -> None:
    """The desktop backend that started the runtime vs. a key-holder claiming desktop.

    Both dial ``client=attach, surface=desktop`` and send ``by=desktop``. Only
    the one that completed the capability handshake AND presents the request
    proof records ``desktop`` (and lifts the delete with no judge); the other
    records ``unverified`` and the judge decides.
    """
    from tests.unit.session.runtime.test_approval_authority_seam import _dial, _send

    home_dir, config = home
    cap = mint_operator_cap()
    rig = _Rig(home_dir, config, tag=f"sock{int(present_proof)}")
    live = await _serve_asks(config, cap, rig.queue)
    try:
        conn = await _dial(
            live.record,
            cap=cap if present_proof else None,
            client="attach",
            surface="desktop",
        )
        frame: dict[str, Any] = {
            "op": "ask_respond",
            "ask_id": rig.ask_id,
            "answers": {"q1": ["Yes"]},
            "by": "desktop",
        }
        if present_proof:
            frame["operator_cap"] = conn.proof(cap)
        reply = await _send(conn, None, frame)
        assert reply["op"] == "ack", reply
        conn.close()
    finally:
        await live.close(config)
    record = rig.record()
    assert record["status"] == "answered"
    outcome, judge_calls = await rig.guard()
    if present_proof:
        assert record["answered_by"] == {"surface": "desktop"}
        assert outcome is not None and outcome.blocked is False, outcome
        assert judge_calls == 0
    else:
        assert record["answered_by"] == {"surface": "unverified"}
        _assert_judged_and_blocked(outcome, judge_calls, rig.victim)


@pytest.mark.asyncio
async def test_a_mesh_relayed_answer_is_remote_even_with_a_valid_proof(
    home: tuple[Path, Path],
) -> None:
    """MESH FORGERY: ``net_forward``/``stream_send`` dial with ``locality=remote``.

    The relaying runtime may itself hold a capability for a session it owns, so
    this drives the worst case: a relayed ``attach`` dial claiming ``desktop``
    that ALSO completes the handshake and presents a valid proof. The far-end
    person is still another device's, so the label is ``remote`` and the
    delete goes to the judge.
    """
    from tests.unit.session.runtime.test_approval_authority_seam import _dial, _send

    home_dir, config = home
    cap = mint_operator_cap()
    rig = _Rig(home_dir, config, tag="mesh")
    live = await _serve_asks(config, cap, rig.queue)
    try:
        conn = await _dial(
            live.record,
            cap=cap,
            client="attach",
            surface="desktop",
            locality="remote",
            capabilities=["prompt"],
            peer={"device_id": "dev-synthetic-peer", "name": "peer"},
        )
        reply = await _send(
            conn,
            None,
            {
                "op": "ask_respond",
                "ask_id": rig.ask_id,
                "answers": {"q1": ["Yes"]},
                "by": "desktop",
                "operator_cap": conn.proof(cap),
            },
        )
        assert reply["op"] == "ack", reply
        conn.close()
    finally:
        await live.close(config)
    assert rig.record()["answered_by"] == {"surface": "remote"}
    outcome, judge_calls = await rig.guard()
    _assert_judged_and_blocked(outcome, judge_calls, rig.victim)


@pytest.mark.asyncio
async def test_a_capability_holding_client_presents_no_proof_unless_asked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The phone daemon can hold a capability (it engages runtimes) and must not
    lend it to a portal answer: the proof rides an ask frame ONLY on opt-in."""
    import asyncio
    import json

    from local_operator.mobile.attach_client import AttachClient

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))

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
    client._writer = writer  # type: ignore[assignment]
    client._operator_cap = mint_operator_cap()
    client._operator_nonce = operator_nonce()
    client._operator_salt = operator_nonce()
    client._authority_bearing = True
    for call in (
        client.ask_respond("a-1", {"q1": ["Yes"]}, by="desktop"),
        client.ask_answer("a-1.0", "Yes"),
        client.ask_decline("a-1"),
    ):
        try:
            await asyncio.wait_for(call, timeout=0.2)
        except (TimeoutError, asyncio.TimeoutError, Exception):  # noqa: BLE001
            pass
        assert "operator_cap" not in writer.sent[-1], writer.sent[-1]
        assert "_operator_ask" not in writer.sent[-1]


def test_an_unverified_answer_reads_unattributed_in_the_refusal() -> None:
    """``unverified`` names a missing proof, not a place: the loser reads no label."""
    from local_operator.asks.render import refusal_copy

    unverified = {"status": "answered", "answered_by": {"surface": "unverified"}}
    assert refusal_copy(unverified) == "already answered."
    desktop = {"status": "answered", "answered_by": {"surface": "desktop"}}
    assert refusal_copy(desktop) == "already answered by desktop."
