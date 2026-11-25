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
from asyncio import StreamWriter
from pathlib import Path
from typing import Any, cast

import pytest

from local_operator.asks import store
from local_operator.asks.queue import AskQueue
from local_operator.asks.store import mirror_request_id
from local_operator.asks.surfaces import (
    GUARD_TRUSTED_SURFACES,
    PROFILE_CONSENT_SURFACES,
    stamp_answer_surface,
    weakest_surface,
)
from local_operator.guardrails.check import CallContext, GuardSettings, check_tool_call
from local_operator.guardrails.provenance import UNPROVEN_SURFACE_LABEL
from local_operator.harness.approval import (
    ASK_ANSWER_OPS,
    AUTHORITY_OPS,
    mint_operator_cap,
    operator_nonce,
    request_proof,
)
from local_operator.session.runtime.server import (
    RuntimeServer,
    _ClientConn,
    _ConnectionAuthority,
)
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

    def revise_ask(self, ask_id: str, answers: Any, by: str = "unknown", **_: Any) -> Any:
        return self._queue.revise(ask_id, answers, by=by)

    def decline_ask(self, ask_id: str, by: str = "unknown") -> Any:
        return self._queue.decline(ask_id, by=by)

    def dismiss_ask(self, ask_id: str, by: str = "unknown", reason: str | None = None) -> Any:
        return self._queue.dismiss(ask_id, by=by, reason=reason or "moot")

    async def reconcile_asks(self) -> None:
        return None


class _Rig:
    """One session's real ask log, a real serving handle over it, and a victim dir."""

    def __init__(
        self, home: Path, config: Path, tag: str, *, second_question: bool = False
    ) -> None:
        self.home = home
        self.config = config
        self.session_id = f"stamp{tag}"
        self.victim = home / f"victim-{tag}"
        self.victim.mkdir()
        (self.victim / "f.txt").write_text("x")
        self.scratchpad = config / "sessions" / self.session_id / "scratchpad"
        self.scratchpad.mkdir(parents=True)
        self.queue = AskQueue(_QueueOwnerStub(), config_dir=config, session_id=self.session_id)
        questions = [
            {
                "id": "q1",
                "question": f"Delete {self.victim}?",
                "options": [{"label": "Yes"}, {"label": "No, keep it"}],
            }
        ]
        if second_question:
            # Scenario M's harmless companion: the model writes the ask, so it
            # can pair the delete with a question the operator will happily tap.
            questions.append(
                {
                    "id": "q2",
                    "question": "Also tidy the build logs?",
                    "options": [{"label": "Yes, tidy"}, {"label": "No"}],
                }
            )
        receipt = self.queue.enqueue(questions, "1h")
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

    async def guard(self, *, verdict: str = "BLOCK: no consent (test judge)") -> tuple[Any, int]:
        # ``verdict`` lets a positive control hand the guard a judge that would
        # ALLOW, so an unblocked outcome cannot come from a judge that never ran
        # being asked the wrong question; the default BLOCK judge is the probe's.
        calls: list[int] = []

        async def judge(system: str, user: str) -> str:
            calls.append(1)
            return verdict

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


#: The deterministic refusal #3966 (bd8755e8) writes for an ask that names the
#: path but whose answer is not the operator's (``provenance._not_consent_label``).
_NOT_BY_OPERATOR = "not by the operator"


def _assert_refused_not_consent(
    outcome: Any, judge_calls: int, victim: Path, *, surface: str
) -> None:
    """A forged/unproven answer never settles the delete; it is refused or judged.

    WHY the judge is optional (the #3966 x #3993 interaction). #3993 (this file)
    landed asserting ``judge_calls >= 1``: with the stamp in place a forged answer
    no longer lifted the delete, so it fell through to the (BLOCK) judge. #3966
    rode a different release train and made the guard STRICTER: an ask that
    names the path but was answered on a surface outside
    ``TRUSTED_ASK_SURFACES`` (``remote``/``unverified`` -- exactly what the stamp
    writes) is "answered ... not by the operator" and refused deterministically
    (``DeleteCase.lapsed_only``) before any judge. Both outcomes keep the victim;
    what this file guards is the zero-judge ALLOW of scenarios D/E at 49af370.

    So: blocked, victim intact, and EITHER a judge ran OR the block is #3966's
    not-the-operator refusal -- never some unrelated zero-judge block that would
    let a regression of the stamp hide behind another rule.
    """
    assert outcome is not None
    assert outcome.blocked is True, outcome
    assert victim.exists()
    if judge_calls == 0:
        # ``unverified`` has its own wording since incident a-48e3 (2026-11-16):
        # it says the answer could not be proven the operator's rather than
        # implying there was none. Both are the same deterministic refusal, but
        # each surface must carry ITS wording, never the other's (R1-n2).
        if surface == "unverified":
            assert UNPROVEN_SURFACE_LABEL in outcome.text, outcome
            assert _NOT_BY_OPERATOR not in outcome.text, outcome
        else:
            assert _NOT_BY_OPERATOR in outcome.text, outcome
            assert UNPROVEN_SURFACE_LABEL not in outcome.text, outcome


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
    _assert_refused_not_consent(outcome, judge_calls, rig.victim, surface="remote")


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
    _assert_refused_not_consent(outcome, judge_calls, rig.victim, surface="unverified")


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
    _assert_refused_not_consent(outcome, judge_calls, rig.victim, surface="unverified")


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
@pytest.mark.parametrize("verdict", ["BLOCK: no consent (test judge)", "ALLOW"])
async def test_positive_control_a_proved_answer_is_consent_not_the_3966_refusal(
    home: tuple[Path, Path], verdict: str
) -> None:
    """The inverse of every refusal above, on the same ask and the same delete.

    #3966 refuses an ask "answered ... not by the operator"; that predicate reads
    ``answered_by.surface`` against ``TRUSTED_ASK_SURFACES``, and #3993's stamp
    writes ``desktop`` only for a frame that proved the operator capability. If
    the two ever disagree, operator approval itself stops working: a PROVED
    "Yes" would read as unanswered and be refused. Pin it: the proved answer is
    consent, lifts before the judge (so even a BLOCK judge cannot stop it, and
    an ALLOW judge is never needed), and the refusal text never appears.
    """
    rig = _Rig(*home, tag="PC")
    reply = await rig.dispatch(
        "ask_respond",
        {"ask_id": rig.ask_id, "answers": {"q1": ["Yes"]}, "by": "desktop"},
        _PROVED_DESKTOP,
    )
    assert reply == "answered"
    assert rig.record()["answered_by"] == {"surface": "desktop"}
    outcome, judge_calls = await rig.guard(verdict=verdict)
    assert outcome is None or outcome.blocked is False, outcome
    assert outcome is None or _NOT_BY_OPERATOR not in (outcome.text or ""), outcome
    assert judge_calls == 0
    assert rig.victim.exists()  # the guard is only evaluated; nothing ran


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


def _conn() -> _ClientConn:
    # The reader seam uses the authenticated connection's defaults but never writes to it.
    return _ClientConn(
        writer=cast(StreamWriter, object()),
        kind="attach",
        surface="desktop",
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
    assert all(not authority.inline_ui for authority in seen)
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
    # The production invariant the minted-here rule reads (F2 Part C): a session
    # owns the ask queue for its OWN id, so its ``session_id`` is the queue's.
    session.session_id = queue._session_id
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
        # F2: the ack names the label the runtime stamped, on the real wire.
        assert reply.get("surface") == ("desktop" if present_proof else "unverified"), reply
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
        _assert_refused_not_consent(outcome, judge_calls, rig.victim, surface="unverified")


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
    _assert_refused_not_consent(outcome, judge_calls, rig.victim, surface="remote")


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


# ---------------------------------------------------------------------------
# Agent review round 1: an unproven write must not inherit a proved label
# (scenarios R, R2 and M: allowed with 0 judge calls at 0587523)
# ---------------------------------------------------------------------------


async def _proved_no_then_revise(rig: _Rig, reviser: Any) -> None:
    """A proved desktop "No, keep it", then a revision to "Yes" from ``reviser``."""
    await rig.dispatch(
        "ask_respond", {"ask_id": rig.ask_id, "answers": {"q1": ["No, keep it"]}}, _PROVED_DESKTOP
    )
    assert rig.record()["answered_by"] == {"surface": "desktop"}
    # The frame's ``by`` claims desktop; only the connection's stamp counts.
    await rig.dispatch(
        "ask_revise", {"ask_id": rig.ask_id, "answers": {"q1": ["Yes"]}, "by": "desktop"}, reviser
    )
    assert rig.record()["answers"] == {"q1": ["Yes"]}


@pytest.mark.asyncio
async def test_scenario_r_an_unproven_revise_does_not_keep_the_proved_label(
    home: tuple[Path, Path],
) -> None:
    """R1-1 (R): the fold kept the FIRST answer's ``desktop`` over the revised
    "Yes" a local key-holder wrote, so the delete lifted with no judge."""
    rig = _Rig(*home, tag="R")
    await _proved_no_then_revise(rig, _LOCAL_UNPROVEN)
    assert rig.record()["answered_by"] == {"surface": "unverified"}
    outcome, judge_calls = await rig.guard()
    _assert_refused_not_consent(outcome, judge_calls, rig.victim, surface="unverified")


@pytest.mark.asyncio
async def test_scenario_r2_a_mesh_revise_does_not_keep_the_proved_label(
    home: tuple[Path, Path],
) -> None:
    """R1-1 (R2): the same laundering from a mesh peer holding ``prompt``."""
    rig = _Rig(*home, tag="R2")
    await _proved_no_then_revise(rig, _REMOTE_PROMPT)
    assert rig.record()["answered_by"] == {"surface": "remote"}
    outcome, judge_calls = await rig.guard()
    _assert_refused_not_consent(outcome, judge_calls, rig.victim, surface="remote")


@pytest.mark.asyncio
async def test_a_proved_revise_of_a_proved_answer_still_lifts(home: tuple[Path, Path]) -> None:
    """The honest change of mind: every write proved, so the label stays trusted."""
    rig = _Rig(*home, tag="RP")
    await _proved_no_then_revise(rig, _PROVED_DESKTOP)
    assert rig.record()["answered_by"] == {"surface": "desktop"}
    outcome, judge_calls = await rig.guard()
    assert judge_calls == 0
    assert outcome is None or outcome.blocked is False, outcome


@pytest.mark.asyncio
async def test_a_proved_revise_cannot_wash_an_earlier_unproven_one(
    home: tuple[Path, Path],
) -> None:
    """Fail-closed over EVERY revision: unproven "Yes", then a proved revise to
    the same "Yes" still reads unverified (the weakest write shaped the log)."""
    rig = _Rig(*home, tag="RW")
    await _proved_no_then_revise(rig, _LOCAL_UNPROVEN)
    await rig.dispatch(
        "ask_revise", {"ask_id": rig.ask_id, "answers": {"q1": ["Yes"]}}, _PROVED_DESKTOP
    )
    assert rig.record()["answered_by"] == {"surface": "unverified"}
    outcome, judge_calls = await rig.guard()
    _assert_refused_not_consent(outcome, judge_calls, rig.victim, surface="unverified")


@pytest.mark.asyncio
async def test_scenario_m_an_unproven_cell_is_not_settled_by_a_proved_tap(
    home: tuple[Path, Path],
) -> None:
    """R1-2 (M): an unproven tap answers the delete, a proved tap answers the
    harmless q2; the draft settled with the LAST tap's ``mirror`` label."""
    rig = _Rig(*home, tag="M", second_question=True)
    await rig.dispatch(
        "ask_answer",
        {"request_id": mirror_request_id(rig.ask_id, 0), "value": "Yes"},
        _LOCAL_UNPROVEN,
    )
    await rig.dispatch(
        "ask_answer",
        {"request_id": mirror_request_id(rig.ask_id, 1), "value": "No"},
        _PROVED_DESKTOP,
    )
    record = rig.record()
    assert record["status"] == "answered"
    assert record["answers"] == {"q1": ["Yes"], "q2": ["No"]}
    assert record["answered_by"] == {"surface": "unverified"}
    outcome, judge_calls = await rig.guard()
    _assert_refused_not_consent(outcome, judge_calls, rig.victim, surface="unverified")


@pytest.mark.asyncio
async def test_a_fully_proved_multi_question_answer_still_lifts(home: tuple[Path, Path]) -> None:
    rig = _Rig(*home, tag="MP", second_question=True)
    for index, value in ((0, "Yes"), (1, "No")):
        await rig.dispatch(
            "ask_answer",
            {"request_id": mirror_request_id(rig.ask_id, index), "value": value},
            _PROVED_DESKTOP,
        )
    assert rig.record()["answered_by"] == {"surface": "mirror"}
    outcome, judge_calls = await rig.guard()
    assert judge_calls == 0
    assert outcome is None or outcome.blocked is False, outcome


def test_the_guard_trust_set_mirror_matches_the_guard() -> None:
    """``weakest_surface`` reasons over a COPY of the guard's set; pin them equal."""
    from local_operator.guardrails.provenance import TRUSTED_ASK_SURFACES

    assert GUARD_TRUSTED_SURFACES == TRUSTED_ASK_SURFACES


@pytest.mark.parametrize(
    "labels,expected",
    [
        (("desktop",), "desktop"),
        (("desktop", "desktop"), "desktop"),
        (("desktop", "unverified"), "unverified"),
        (("remote", "desktop"), "remote"),
        (("desktop", "mirror"), "mirror"),
        (("desktop", "terminal"), "terminal"),
        (("tui", "desktop"), "tui"),
        # Each trusted by one reader only: no input is safe, so neither trusts it.
        (("chat", "terminal"), "unverified"),
        ((), "unverified"),
    ],
)
def test_weakest_surface_never_exceeds_any_input(labels: tuple[str, ...], expected: str) -> None:
    result = weakest_surface(*labels)
    assert result == expected
    for reader in (GUARD_TRUSTED_SURFACES, PROFILE_CONSENT_SURFACES):
        if result in reader:
            assert all(label in reader for label in labels)


# ---------------------------------------------------------------------------
# Incident a-48e3 (2026-11-16): an operator's desktop-card answer to a runtime
# the wake supervisor started was stamped ``unverified``, and nothing recorded
# which connection answered or told the card it could not authorise a delete.
# The fail-closed stays; these pin the attribution log and the proof probe the
# answer route reports as ``consent_bearing``.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("authority", "label", "proved"),
    [
        (_LOCAL_UNPROVEN, "unverified", False),
        (_PROVED_DESKTOP, "desktop", True),
        (_REMOTE_PROMPT, "remote", False),
    ],
)
async def test_each_stamp_logs_the_connection_facts_without_the_answer(
    home: tuple[Path, Path],
    caplog: pytest.LogCaptureFixture,
    authority: _ConnectionAuthority,
    label: str,
    proved: bool,
) -> None:
    rig = _Rig(*home, tag=f"log{label}")
    with caplog.at_level("INFO", logger="local_operator.session.runtime.server"):
        await rig.dispatch(
            "ask_respond",
            {"ask_id": rig.ask_id, "answers": {"q1": ["Yes-secret-words"]}, "by": "desktop"},
            authority,
        )
    lines = [r.getMessage() for r in caplog.records if "ask answer stamped" in r.getMessage()]
    assert len(lines) == 1, lines
    assert f"ask={rig.ask_id!r}" in lines[0]
    assert f"surface={label}" in lines[0]
    assert f"proved={proved}" in lines[0]
    assert f"locality={authority.locality}" in lines[0]
    # Connection facts only: never the answer's words, never the frame's ``by``.
    assert "Yes-secret-words" not in lines[0]
    assert rig.record()["answered_by"] == {"surface": label}


@pytest.mark.asyncio
async def test_the_mirror_stamp_logs_too(
    home: tuple[Path, Path], caplog: pytest.LogCaptureFixture
) -> None:
    rig = _Rig(*home, tag="logmirror")
    request_id = mirror_request_id(rig.ask_id, 0)
    with caplog.at_level("INFO", logger="local_operator.session.runtime.server"):
        await rig.dispatch(
            "ask_answer", {"request_id": request_id, "value": "Yes"}, _LOCAL_UNPROVEN
        )
    [line] = [r.getMessage() for r in caplog.records if "ask answer stamped" in r.getMessage()]
    assert f"ask={request_id!r}" in line and "surface=unverified" in line and "proved=False" in line


@pytest.mark.asyncio
async def test_a_newline_in_the_ask_id_cannot_forge_a_second_log_line(
    home: tuple[Path, Path], caplog: pytest.LogCaptureFixture
) -> None:
    """R1-m1: the ask id is client-supplied and logged before validation; a
    newline in it must not mint a fake ``surface=desktop proved=True`` line."""
    rig = _Rig(*home, tag="loginject")
    forged = "a-x\nask answer stamped: op=ask_respond ask=a-y surface=desktop proved=True"
    with caplog.at_level("INFO", logger="local_operator.session.runtime.server"):
        try:
            await rig.dispatch(
                "ask_respond", {"ask_id": forged, "answers": {"q1": ["Yes"]}}, _LOCAL_UNPROVEN
            )
        except Exception:  # an unknown ask is refused AFTER the stamp is logged
            pass
    [line] = [r.getMessage() for r in caplog.records if "ask answer stamped" in r.getMessage()]
    assert "\n" not in line
    assert f"ask={forged!r}" in line
    assert line.endswith("proved=False")


@pytest.mark.parametrize(
    ("cap", "bearing", "held"),
    [
        (True, True, True),  # this process started the runtime and it proved itself
        (False, True, False),  # no capability: not this process's runtime (a-48e3)
        (True, False, False),  # capability, but the runtime never proved the same one
        (False, False, False),
    ],
)
def test_the_proof_probe_matches_what_an_opted_in_ask_answer_presents(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cap: bool, bearing: bool, held: bool
) -> None:
    """``operator_ask_proof_held`` is True exactly when ``authority_proof`` would
    attach a proof to an opted-in ask answer, i.e. when the runtime will stamp a
    trusted surface. It is the one source of the route's ``consent_bearing``."""
    from local_operator.mobile.attach_client import AttachClient

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    client = AttachClient(lambda _projection: None, lambda _reason: None)
    client._operator_cap = mint_operator_cap() if cap else None
    client._operator_nonce = operator_nonce()
    client._operator_salt = operator_nonce()
    client._authority_bearing = bearing
    assert client.operator_ask_proof_held() is held
    presented = client.authority_proof("ask_respond", {}, operator_ask=True)
    assert (presented is not None) is held


class _SendClient:
    """A bound client double: records whether its proof was held when called."""

    def __init__(self, held: bool, *, connected: bool = True, probe: bool = True) -> None:
        self.connected = connected
        self.held = held
        self.sent: list[str] = []
        if probe:
            self.operator_ask_proof_held = lambda: self.held

    async def ask_respond(self, ask_id: str, body: Any, *, by: str) -> str:
        self.sent.append(ask_id)
        return "answered"


def _facade_with(client: Any) -> Any:
    from local_operator.session.attached import AttachedSession
    from local_operator.session.owner import SessionOwner
    from local_operator.session.placement import SessionPlacement

    facade = AttachedSession.__new__(AttachedSession)
    # These contracts exercise proof on the existing bound desktop client,
    # including reconnects. A peer owner keeps that production path; a local
    # desktop owner instead opens a separate one-shot control connection, covered
    # in test_ask_control_client.py rather than by this sending-client double.
    facade._surface = "desktop"
    facade._owner = cast(
        SessionOwner, types.SimpleNamespace(placement=SessionPlacement(mode="peer"))
    )
    facade._client = client  # type: ignore[attr-defined]
    facade._ask_proof_at_send = False  # type: ignore[attr-defined]

    async def _bound() -> None:
        return None

    facade._ensure_bound = _bound  # type: ignore[attr-defined]
    return facade


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("client", "expected"),
    [
        (_SendClient(True), True),
        (_SendClient(False), False),
        # A client predating the probe: fail closed.
        (_SendClient(True, probe=False), False),
    ],
)
async def test_the_facade_reports_the_proof_its_sending_client_held(
    client: Any, expected: bool
) -> None:
    facade = _facade_with(client)
    assert facade.ask_consent_bearing() is False  # nothing sent yet
    await facade._ask_client_method("ask_respond", "a-1", {"q": ["Yes"]}, by="desktop")
    assert client.sent == ["a-1"]
    assert facade.ask_consent_bearing() is expected


@pytest.mark.asyncio
async def test_a_reconnect_after_the_send_does_not_change_the_reported_proof() -> None:
    """R1-m2: the route reads the probe after ``ask_respond`` returns. A reconnect
    in between swaps the bound client; the receipt must still describe the
    connection that CARRIED the answer, in both directions."""
    old = _SendClient(False)
    facade = _facade_with(old)
    await facade._ask_client_method("ask_respond", "a-1", {"q": ["Yes"]}, by="desktop")
    facade._client = _SendClient(True)  # a fresh proving connection after the send
    assert facade.ask_consent_bearing() is False
    proving = _SendClient(True)
    facade = _facade_with(proving)
    await facade._ask_client_method("ask_respond", "a-2", {"q": ["Yes"]}, by="desktop")
    facade._client = None  # dropped after the send
    assert facade.ask_consent_bearing() is True


@pytest.mark.asyncio
async def test_a_send_refused_mid_reconnect_reports_no_proof() -> None:
    facade = _facade_with(_SendClient(True, connected=False))
    facade._ask_proof_at_send = True  # a stale value from an earlier answer

    async def _refusal(_reason: Any) -> Exception:
        return RuntimeError("reconnecting")

    facade._refusal_while_reconnecting = _refusal  # type: ignore[attr-defined]
    facade._unavailable_reason = lambda: "reconnecting"  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError):
        await facade._ask_client_method("ask_respond", "a-3", {"q": ["Yes"]}, by="desktop")
    assert facade.ask_consent_bearing() is False


@pytest.mark.asyncio
@pytest.mark.parametrize("bearing", [True, False], ids=["desktop-started", "terminal-started"])
async def test_the_signature_free_probe_matches_when_the_real_client_would_sign(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bearing: bool
) -> None:
    """PR #4208 S-1 (decision a-856f): ``bridge_decide_signature_free`` is False
    EXACTLY when the real ``AttachClient`` would sign a ``bridge_decide`` frame
    (the Touch ID path), so the desktop route reroutes precisely those cards."""
    from local_operator.mobile.attach_client import AttachClient

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    client = AttachClient(lambda _projection: None, lambda _reason: None)
    client._operator_cap = mint_operator_cap()
    client._operator_nonce = operator_nonce()
    client._operator_salt = operator_nonce()
    client._authority_bearing = bearing
    client._operator_signature_supported = True
    monkeypatch.setattr(type(client), "connected", property(lambda _self: True))
    signed: list[tuple[str, str]] = []

    async def signer(action: str, request_id: str) -> Any:
        signed.append((action, request_id))
        return None

    client._operator_signature = signer  # type: ignore[method-assign]
    facade = _facade_with(client)
    frame = {"op": "bridge_decide", "ask_id": "a-1", "question_id": "bmsg_x", "word": "approve"}
    await client._present_operator_signature(frame)
    assert await facade.bridge_decide_signature_free() is bearing
    assert signed == ([] if bearing else [("decide", "a-1#bmsg_x#approve")])
