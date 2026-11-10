"""``peer_set_model`` over a REAL in-process registrant: the op, the sender, the tool.

Mirrors ``test_peer_client.py``'s loopback approach. The receive handle is a
``FakeHandle`` whose ``receive_peer_model`` answers like the real hosts (their
own logic is pinned in ``test_peer_model.py`` and the TUI pilot), so what lives
here is the wire: the frame shape, the engagement gate, the capability probe,
and how each sender surface renders every kind of answer — including an OLDER
registrant that does not know the op (design D7).
"""

from __future__ import annotations

import json
import os

import pytest

from local_operator.harness.types import ToolContext
from local_operator.mobile.peer_send import (
    PeerModelUnconfirmed,
    parse_model_selector,
    switch_peer_model,
)
from local_operator.mobile.types import validate_control_frame
from local_operator.session.runtime import registry
from local_operator.session.runtime.server import _LOCAL_AUTHORITY, RuntimeServer
from local_operator.tools.builtin import _describe_send_approval, execute_send
from tests.unit.session.runtime.test_server import (
    FakeHandle,
    _dial,
    _until,
    _wait_record,
)


class _ModelHandle(FakeHandle):
    """Answers ``receive_peer_model`` like a real host: refuses an unknown provider."""

    async def receive_peer_model(  # noqa: ANN001, ANN202
        self, provider, model_id, *, sender=None
    ) -> str:
        self.calls.append(("receive_peer_model", (provider, model_id), {"sender": sender}))
        if provider == "nosuchprov":
            raise ValueError("refused: 'nosuchprov' is not a known provider; still on test/model")
        return f"switched to {provider}/{model_id} (was test/model)\nits next turn runs on it"


class _OldHandle(FakeHandle):
    """A handle without the capability (a reduced or older host)."""


class _OldRuntime(RuntimeServer):
    """A registrant built before the op: its dispatch answers unknown-op."""

    # The override MIRRORS the base signature, ``authority`` included, even
    # though this host never carries a connection authority of its own: it
    # stands in for the real dispatch, so a parameter the base gained and the
    # override dropped would be a call this host cannot be handed at all.
    async def _dispatch(  # noqa: ANN001, ANN202
        self, op, frame, authority=_LOCAL_AUTHORITY, *, deliver=None
    ):
        if op == "peer_set_model":
            raise ValueError(f"unknown op: {op!r}")
        return await super()._dispatch(op, frame, authority, deliver=deliver)


def test_the_frame_validator_refuses_a_malformed_switch() -> None:
    validate_control_frame(
        {"op": "peer_set_model", "provider": "deepseek", "model_id": "deepseek-flash"}
    )
    for bad in (
        {"op": "peer_set_model", "provider": "", "model_id": "x"},
        {"op": "peer_set_model", "provider": "deepseek"},
        {"op": "peer_set_model", "provider": "deepseek", "model_id": 3},
        {"op": "peer_set_model", "provider": "a", "model_id": "b", "sender": "me"},
    ):
        with pytest.raises(ValueError):
            validate_control_frame(bad)


@pytest.mark.parametrize(
    "selector,expected",
    [
        ("deepseek/deepseek-flash", ("deepseek", "deepseek-flash")),
        (" DeepSeek / deepseek-flash ", ("deepseek", "deepseek-flash")),
        # First slash only, like /model: the id keeps its own slash.
        ("openrouter/deepseek/deepseek-chat", ("openrouter", "deepseek/deepseek-chat")),
    ],
)
def test_the_selector_splits_on_the_first_slash(selector, expected) -> None:
    assert parse_model_selector(selector) == expected


@pytest.mark.parametrize("selector", ["", "deepseek", "/x", "deepseek/", "  "])
def test_a_malformed_selector_is_refused_with_the_form(selector) -> None:
    refusal = parse_model_selector(selector)
    assert isinstance(refusal, str) and "<provider>/<model-id>" in refusal


@pytest.mark.asyncio
async def test_the_op_reaches_the_handle_and_its_sentence_comes_back() -> None:
    handle = _ModelHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    runtime.set_record_started(True)
    try:
        record = await _wait_record()
        detail = await switch_peer_model(
            record, provider="deepseek", model_id="deepseek-flash", sender={"pid": 7}
        )
        assert detail == (
            "switched to deepseek/deepseek-flash (was test/model)\nits next turn runs on it"
        )
        assert handle.calls[-1] == (
            "receive_peer_model",
            ("deepseek", "deepseek-flash"),
            {"sender": {"pid": 7}},
        )
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_a_refusal_rides_the_error_frame_unchanged() -> None:
    handle = _ModelHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    runtime.set_record_started(True)
    try:
        record = await _wait_record()
        with pytest.raises(RuntimeError) as caught:
            await switch_peer_model(record, provider="nosuchprov", model_id="x", sender={})
        assert str(caught.value) == (
            "refused: 'nosuchprov' is not a known provider; still on test/model"
        )
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_an_unengaged_session_is_refused_at_the_receiver() -> None:
    """The receive-side gate: holds against a sender that skipped resolution."""
    handle = _ModelHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    writer = None
    try:
        record = await _wait_record()
        assert record.started is False
        reader, writer = await _dial(record)
        writer.write(
            json.dumps(
                {"op": "peer_set_model", "req": 3, "provider": "deepseek", "model_id": "x"}
            ).encode()
            + b"\n"
        )
        await writer.drain()
        err = await _until(reader, "error", 3)
        assert f"pid {record.pid} has not been engaged yet" in err["message"]
        assert "cannot be switched remotely" in err["message"]
        assert handle.calls == []
    finally:
        if writer is not None:
            writer.close()
        runtime.close()


@pytest.mark.asyncio
async def test_a_handle_without_the_capability_says_so() -> None:
    runtime = RuntimeServer(_OldHandle(), kind="tui")
    runtime.start()
    runtime.set_record_started(True)
    try:
        record = await _wait_record()
        with pytest.raises(RuntimeError) as caught:
            await switch_peer_model(record, provider="deepseek", model_id="x", sender={})
        assert str(caught.value) == "this session cannot switch models remotely"
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_an_older_registrant_is_named_as_older_and_nothing_changed() -> None:
    """D7: the raw ``unknown op`` becomes the sentence that says what to do."""
    runtime = _OldRuntime(_ModelHandle(), kind="tui")
    runtime.start()
    runtime.set_record_started(True)
    try:
        record = await _wait_record()
        with pytest.raises(RuntimeError) as caught:
            await switch_peer_model(record, provider="deepseek", model_id="x", sender={})
        # No pid inside the sentence: every surface prints the address beside
        # it, and naming it twice was QA Q1 / UX U6.
        assert str(caught.value) == (
            "older lop: it cannot switch models remotely; nothing changed — "
            "update it (lop update) or run /model in that session"
        )
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_a_lost_ack_is_unconfirmed_not_failed(monkeypatch) -> None:
    import local_operator.mobile.peer_client as peer_client

    async def silent(*_args, **_kwargs):
        raise TimeoutError

    monkeypatch.setattr(peer_client, "send_control_op", silent)
    record = registry.SessionRecord(
        pid=4321,
        kind="tui",
        session_id="s",
        conversation_name="n",
        cwd="/tmp",
        model_label="test/model",
        control_port=1,
        control_key="k",
        started=True,
    )
    with pytest.raises(PeerModelUnconfirmed) as caught:
        await switch_peer_model(record, provider="deepseek", model_id="x", sender={})
    assert "unconfirmed and may still apply" in str(caught.value)
    assert "lop sessions" in str(caught.value)


@pytest.mark.asyncio
async def test_a_refused_dial_is_not_delivered_and_nothing_changed() -> None:
    """N3: the socket never opened, so no byte of the op was sent."""
    import socket

    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    free_port = probe.getsockname()[1]
    probe.close()  # nothing listens here now
    record = registry.SessionRecord(
        pid=4321,
        kind="tui",
        session_id="s",
        conversation_name="n",
        cwd="/tmp",
        model_label="test/model",
        control_port=free_port,
        control_key="k",
        started=True,
    )
    with pytest.raises(RuntimeError) as caught:
        await switch_peer_model(record, provider="deepseek", model_id="x", sender={})
    assert not isinstance(caught.value, PeerModelUnconfirmed)
    assert str(caught.value).startswith("could not reach that session; nothing changed (")


@pytest.mark.asyncio
async def test_the_sender_waits_longer_than_the_tui_hop(monkeypatch) -> None:
    """N2: a busy TUI target has 10 s to answer; the sender must outwait it."""
    import local_operator.mobile.peer_client as peer_client
    from local_operator.mobile.tui_handle import _APP_HOP_TIMEOUT_S

    seen: dict[str, float] = {}

    async def fake(record, op, fields, *, deadline_s=5.0, **_kw):  # noqa: ANN001
        seen["deadline_s"] = deadline_s
        return "ok"

    monkeypatch.setattr(peer_client, "send_control_op", fake)
    record = registry.SessionRecord(
        pid=4321,
        kind="tui",
        session_id="s",
        conversation_name="n",
        cwd="/tmp",
        model_label="test/model",
        control_port=1,
        control_key="k",
        started=True,
    )
    await switch_peer_model(record, provider="deepseek", model_id="x", sender={})
    assert seen["deadline_s"] > _APP_HOP_TIMEOUT_S


# ---------------------------------------------------------------------------
# The ``send`` tool's model mode
# ---------------------------------------------------------------------------


async def _start_peer(handle: FakeHandle, runtime_cls: type[RuntimeServer] = RuntimeServer):
    """A live, engaged registrant plus an ALIAS record under a pid that is not
    this process's, so the tool's self-guard does not fire (see test_send_tool)."""
    runtime = runtime_cls(handle, kind="tui")
    runtime.start()
    runtime.set_record_started(True)
    own = await _wait_record()
    alias = registry.SessionRecord(
        pid=os.getppid(),
        kind="tui",
        session_id="alias-session",
        conversation_name="peer-target",
        cwd="/tmp",
        model_label="test/model",
        control_port=own.control_port,
        control_key=own.control_key,
        started=True,
    )
    registry.publish(alias)
    return runtime, alias


def _context() -> ToolContext:
    return ToolContext(cwd="/tmp")


@pytest.mark.asyncio
async def test_the_tool_switches_a_peer_and_echoes_its_sentence() -> None:
    handle = _ModelHandle()
    runtime, alias = await _start_peer(handle)
    try:
        result = await execute_send(
            "m1",
            {"target": "peer-target", "model": "deepseek/deepseek-flash"},
            None,
            None,
            _context(),
        )
        assert result.is_error is False, result.text
        # Outcome first, address last in `lop send`'s grammar (D1, D4).
        assert result.text.splitlines() == [
            "switched to deepseek/deepseek-flash (was test/model)",
            "its next turn runs on it",
            f"→ peer-target (pid {alias.pid})",
        ]
        assert (result.details or {})["outcome"] == "switched"
        assert handle.calls[-1][0:2] == ("receive_peer_model", ("deepseek", "deepseek-flash"))
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_the_tool_reports_a_refusal_as_an_error() -> None:
    runtime, alias = await _start_peer(_ModelHandle())
    try:
        result = await execute_send(
            "m2", {"pid": alias.pid, "model": "nosuchprov/x"}, None, None, _context()
        )
        assert result.is_error is True
        # The reason LEADS so the collapsed error slot shows it, not the address (D2).
        assert result.text.startswith(
            "refused: 'nosuchprov' is not a known provider; still on test/model\n→ "
        ), result.text
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_the_tool_names_an_older_peer() -> None:
    runtime, alias = await _start_peer(_ModelHandle(), _OldRuntime)
    try:
        result = await execute_send(
            "m3", {"pid": alias.pid, "model": "deepseek/deepseek-flash"}, None, None, _context()
        )
        assert result.is_error is True
        assert result.text.startswith("older lop: it cannot switch models remotely"), result.text
        assert result.text.count(str(alias.pid)) == 1, "the pid is printed once (Q1/U6)"
    finally:
        runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "args,needle",
    [
        (
            {"target": "peer-target", "model": "deepseek/x", "message": "hi"},
            "pass either message or model, not both — send the note in a second call",
        ),
        ({"target": "peer-target", "model": "deepseek/x", "now": True}, "now=True does not apply"),
        ({"target": "peer-target", "model": "deepseek"}, "<provider>/<model-id>"),
        ({"target": "peer-target"}, "pass a message (or model="),
    ],
)
async def test_the_tool_refuses_a_malformed_call_before_any_dial(args, needle) -> None:
    handle = _ModelHandle()
    runtime, _alias = await _start_peer(handle)
    try:
        result = await execute_send("m4", args, None, None, _context())
        assert result.is_error is True
        assert needle in result.text
        assert not any(call[0] == "receive_peer_model" for call in handle.calls)
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_the_tool_refuses_its_own_session() -> None:
    handle = _ModelHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    runtime.set_record_started(True)
    try:
        own = await _wait_record()
        result = await execute_send(
            "m5", {"pid": own.pid, "model": "deepseek/x"}, None, None, _context()
        )
        assert result.is_error is True
        assert "use /model" in result.text
        assert handle.calls == []
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_a_stored_session_is_named_not_running(monkeypatch, tmp_path) -> None:
    """Live only (D4): a session that exists on disk gets the two ways to switch it."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "coldsess1").mkdir(parents=True)
    result = await execute_send(
        "m6", {"session": "coldsess1", "model": "deepseek/deepseek-flash"}, None, None, _context()
    )
    assert result.is_error is True
    assert "session 'coldsess1' is not running — open it and use /model" in result.text
    assert "lop --resume coldsess1 --hosting <provider> --model <model-id>" in result.text


def test_the_approval_line_names_the_billing_change() -> None:
    assert _describe_send_approval({"pid": 48213, "model": "deepseek/deepseek-flash"}, "/") == (
        "switch pid 48213's model to deepseek/deepseek-flash (changes that session's billing)"
    )
    # The message form is unchanged.
    assert _describe_send_approval({"pid": 1, "message": "hi"}, "/") == "to pid 1 (wake): hi"


def test_the_json_payload_is_one_field_wider() -> None:
    """Keeps the send schema's footprint claims honest.

    D4 added ``model``; the proactive class (design §8.2.5) adds ``patience``
    as the second deliberate field, and PR-B (``sessions-remote-tools.md``
    §3B) adds ``peer`` as the third — the mesh delivery path: a send to a
    session on another device drives a turn there and returns the owner's
    reply. ``repeat`` is the fourth intentional addition: only an explicit
    user-requested force send bypasses confirmed same-turn receipt reuse.
    All are pinned EXACTLY, so a further field cannot arrive without this
    line moving and a reviewer seeing why.
    """
    from local_operator.tools.builtin import SendParams

    fields = set(SendParams.model_json_schema()["properties"])
    assert fields == {
        "target",
        "pid",
        "session",
        "message",
        "model",
        "wake",
        "now",
        "patience",
        "peer",
        "repeat",
    }
    repeat = SendParams.model_json_schema()["properties"]["repeat"]
    assert repeat["type"] == "boolean"
    assert repeat["default"] is False
    assert "confirmed same-turn" in repeat["description"]
    assert "only on explicit user request" in repeat["description"]


# ---------------------------------------------------------------------------
# ``AttachClient.set_model``/``set_effort`` carry ``defer_ok`` + ``request_id``.
#
# The client half of the runtime's deferred settings ack: the keys reach the
# owner only when asked for (an un-opted caller's frame is byte-identical), and
# a busy owner's ``queued`` answer comes back as ``QUEUED_DETAIL``.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_attach_client_sends_defer_keys_only_when_asked(monkeypatch) -> None:
    from local_operator.mobile.attach_client import AttachClient

    frames: list[tuple[str, dict[str, object]]] = []

    async def capture(self, op, **fields):  # noqa: ANN001, ANN202
        frames.append((op, fields))
        return "ok"

    monkeypatch.setattr(AttachClient, "_request", capture)
    client = AttachClient(lambda p: None, lambda reason: None)
    await client.set_model("openai", "m")
    await client.set_effort("high")
    await client.set_model("openai", "m", "low", defer_ok=True, request_id="r1")
    await client.set_effort("high", defer_ok=True, request_id="r2")
    assert frames == [
        ("set_model", {"provider": "openai", "model_id": "m"}),
        ("set_effort", {"effort": "high"}),
        (
            "set_model",
            {
                "provider": "openai",
                "model_id": "m",
                "effort": "low",
                "defer_ok": True,
                "request_id": "r1",
            },
        ),
        ("set_effort", {"effort": "high", "defer_ok": True, "request_id": "r2"}),
    ]


@pytest.mark.asyncio
async def test_a_busy_owner_answers_the_attach_client_queued(monkeypatch) -> None:
    """Over a real runtime: the held switch is acked ``queued`` and lands later."""
    import asyncio
    import threading

    from local_operator.mobile.attach_client import QUEUED_DETAIL, AttachClient

    release = threading.Event()
    landed: list[tuple[str, str | None]] = []

    class Held(FakeHandle):
        async def set_effort(self, effort, *, request_id=None):  # noqa: ANN001, ANN202
            assert await asyncio.to_thread(release.wait, 10)
            landed.append((effort, request_id))
            return f"effort: {effort}"

    runtime = RuntimeServer(Held(), kind="tui")
    runtime.start()
    client = AttachClient(lambda p: None, lambda reason: None)
    try:
        record = await _wait_record()
        await client.connect(record, "s1")
        detail = await asyncio.wait_for(
            client.set_effort("high", defer_ok=True, request_id="rid-9"), timeout=5
        )
        assert detail == QUEUED_DETAIL
        assert landed == []
        release.set()
        deadline = asyncio.get_running_loop().time() + 5
        while not landed and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.02)
        assert landed == [("high", "rid-9")]
        await client.detach()
    finally:
        release.set()
        runtime.close()


@pytest.mark.asyncio
async def test_slash_result_deferred_acceptance_is_explicit_and_legacy_unchanged(
    monkeypatch,
) -> None:
    from local_operator.mobile.attach_client import AttachClient

    seen = []

    async def frame(self, op, **fields):
        seen.append((op, fields))
        return {"op": "ack", "detail": "queued", "queued": True}

    async def payload(self, op, **fields):
        seen.append((op, fields))
        return {"kind": "notice", "text": "applied"}

    monkeypatch.setattr(AttachClient, "_request_frame", frame)
    monkeypatch.setattr(AttachClient, "_request_payload", payload)
    client = AttachClient(lambda p: None, lambda reason: None)
    result = await client.slash_result("model", "test/new", defer_ok=True, request_id="slash-1")
    assert result["data"] == {"queued": True, "request_id": "slash-1"}
    assert seen[-1][1]["defer_ok"] is True
    assert seen[-1][1]["request_id"] == "slash-1"
    result = await client.slash_result("model", "test/legacy")
    assert result["text"] == "applied"
    assert "defer_ok" not in seen[-1][1] and "request_id" not in seen[-1][1]
