"""Payload handling, settings, lop config and event projection for the runtime.

``SessionPool.run_turn`` is tested against a fake session object that mimics
the SDK session's ``prompt`` + subscription contract, so the per-session
serialisation and stream termination are covered without a model. The real
lop session path is exercised end to end in the deployed runtime (README
transcripts).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import lop_runtime
import pytest
from lop_runtime import (
    PayloadError,
    RuntimeSettings,
    SessionPool,
    Turn,
    parse_payload,
    project_event,
    write_lop_config,
)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"prompt": "hi"}, Turn("hi")),
        ({"message": "hi", "model": "m.x"}, Turn("hi", "m.x")),
        (b'{"prompt": "bytes"}', Turn("bytes")),
        ("plain text prompt", Turn("plain text prompt")),
    ],
)
def test_parse_payload_accepts_documented_shapes(payload: Any, expected: Turn) -> None:
    assert parse_payload(payload) == expected


@pytest.mark.parametrize("payload", [{}, {"prompt": "   "}, {"prompt": 3}, [1, 2]])
def test_parse_payload_rejects_non_prompts(payload: Any) -> None:
    with pytest.raises(PayloadError):
        parse_payload(payload)


def test_settings_require_a_model_and_parse_tools() -> None:
    with pytest.raises(RuntimeError, match="LOP_MODEL_ID"):
        RuntimeSettings.from_env({})
    settings = RuntimeSettings.from_env(
        {"LOP_MODEL_ID": "m.x", "LOP_TOOLS": "read, bash,,", "LOP_HOME": "/srv/lop"}
    )
    assert settings.tools == ("read", "bash")
    assert settings.approvals == "auto"
    assert settings.home == Path("/srv/lop")
    assert RuntimeSettings.from_env({"LOP_MODEL_ID": "m.x", "LOP_TOOLS": ""}).tools is None


def test_lop_config_is_under_values_and_points_at_the_bridge(tmp_path: Path) -> None:
    """lop ignores top-level keys; the provider block must sit under ``values``."""
    settings = RuntimeSettings(model_id="m.x", home=tmp_path)
    path = write_lop_config(tmp_path / "cfg", base_url="http://127.0.0.1:9/v1", settings=settings)
    values = json.loads(path.read_text())["values"]
    assert values["hosting"] == "openai-compatible"
    assert values["model_name"] == "m.x"
    provider = values["providers"]["openai-compatible"]
    assert provider["base_url"] == "http://127.0.0.1:9/v1"
    assert provider["models"]["m.x"]["supports_tools"] is True


def test_lop_config_is_read_by_lops_own_config_manager(tmp_path: Path, monkeypatch: Any) -> None:
    """Pin the shape against the installed lop, not just against our expectation."""
    config = pytest.importorskip("local_operator.config")
    settings = RuntimeSettings(model_id="m.x", home=tmp_path)
    write_lop_config(tmp_path, base_url="http://127.0.0.1:9/v1", settings=settings)
    values = config.ConfigManager(tmp_path).get_config().values
    assert values["providers"]["openai-compatible"]["base_url"] == "http://127.0.0.1:9/v1"
    assert values["hosting"] == "openai-compatible"


def test_session_model_override_is_declared_in_lop_config(tmp_path: Path) -> None:
    """Undeclared models get lop's 4K fallback window and every prompt is refused."""
    pool = SessionPool(RuntimeSettings(model_id="m.x", home=tmp_path))
    with pytest.raises(RuntimeError, match="configure"):
        pool.declare_model("m.y")
    path = pool.configure("http://127.0.0.1:9/v1")
    assert pool.declare_model("m.x") is False
    assert pool.declare_model("m.y") is True
    assert pool.declare_model("m.y") is False
    models = json.loads(path.read_text())["values"]["providers"]["openai-compatible"]["models"]
    assert set(models) == {"m.x", "m.y"}
    assert models["m.y"]["context_window"] == 200_000


def test_approval_policy_names() -> None:
    sdk = pytest.importorskip("local_operator.sdk")
    assert isinstance(lop_runtime.approval_policy("auto"), sdk.ApprovalPolicy)
    assert isinstance(lop_runtime.approval_policy("read,grep"), sdk.ApprovalPolicy)
    with pytest.raises(RuntimeError):
        lop_runtime.approval_policy(" , ")


def test_project_event_forwards_text_tools_and_end_only() -> None:
    assert project_event({"type": "message_update", "delta": "hi"}) == {
        "type": "text",
        "text": "hi",
    }
    assert project_event({"type": "message_update", "delta": ""}) is None
    assert project_event({"type": "usage", "tokens": 3}) is None
    start = project_event(
        {"type": "tool_execution_start", "tool_name": "bash", "args": {"command": "uname -m"}}
    )
    assert start == {"type": "tool_start", "tool": "bash", "args": {"command": "uname -m"}}
    end = project_event(
        {
            "type": "tool_execution_end",
            "tool_name": "bash",
            "result": {"content": [{"type": "text", "text": "aarch64"}], "is_error": False},
        }
    )
    assert end == {"type": "tool_end", "tool": "bash", "is_error": False, "result": "aarch64"}
    long = project_event({"type": "tool_execution_end", "tool_name": "read", "result": "x" * 5000})
    assert long is not None and len(long["result"]) == 2000
    assert project_event({"type": "agent_end", "error": "boom"}) == {"type": "end", "error": "boom"}


# ---------------------------------------------------------------------------
# SessionPool with a fake SDK session
# ---------------------------------------------------------------------------


class FakeEvent:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload


_CLOSED = FakeEvent({"type": "__closed__"})


class FakeStream:
    """Mimics ``SessionEventStream``: its own queue per subscription, and
    ``close()`` queues a sentinel that ends iteration after queued events."""

    def __init__(self) -> None:
        self.queue: asyncio.Queue[FakeEvent] = asyncio.Queue()
        self.closed = False

    def __aiter__(self) -> "FakeStream":
        return self

    async def __anext__(self) -> FakeEvent:
        event = await self.queue.get()
        if event is _CLOSED:
            raise StopAsyncIteration
        return event

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.queue.put_nowait(_CLOSED)

    async def aclose(self) -> None:
        self.close()


class FakeSession:
    """Emits a text delta and agent_end per prompt, like a one-step lop turn."""

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.streams: list[FakeStream] = []

    def emit(self, payload: dict[str, Any]) -> None:
        for stream in self.streams:
            if not stream.closed:
                stream.queue.put_nowait(FakeEvent(payload))

    async def prompt(self, text: str) -> None:
        self.prompts.append(text)
        self.emit({"type": "message_update", "delta": f"echo {text}"})
        await asyncio.sleep(0)
        self.emit({"type": "agent_end", "error": None})


class TurnInFlight(RuntimeError):
    """Stand-in for the SDK's refusal to start a second concurrent turn."""


class RefusingSession(FakeSession):
    """``prompt`` raises before emitting any event, as lop's does when it refuses a turn."""

    async def prompt(self, text: str) -> None:
        self.prompts.append(text)
        raise TurnInFlight("a turn is already running")


@pytest.fixture
def pool(tmp_path: Path, monkeypatch: Any) -> SessionPool:
    pool = SessionPool(RuntimeSettings(model_id="m.x", home=tmp_path))
    sessions: dict[str, FakeSession] = {}

    async def fake_session(key: str, model: str | None) -> Any:
        if key not in sessions:
            sessions[key] = RefusingSession() if key.startswith("refuse") else FakeSession()
        return lop_runtime._Entry(session=sessions[key], model=model or "m.x")

    def fake_events(session: FakeSession) -> FakeStream:
        stream = FakeStream()
        session.streams.append(stream)
        return stream

    entries: dict[str, Any] = {}

    async def cached(key: str, model: str | None) -> Any:
        if key not in entries:
            entries[key] = await fake_session(key, model)
        return entries[key]

    monkeypatch.setattr(pool, "_session", cached)
    monkeypatch.setattr("local_operator.sdk.events", fake_events)
    monkeypatch.setattr(
        "local_operator.headless_print.printable_event", lambda event, **_: event.payload
    )
    pool.fake_sessions = sessions  # type: ignore[attr-defined]
    return pool


def test_run_turn_streams_until_agent_end_and_unsubscribes(pool: SessionPool) -> None:
    async def run() -> list[dict[str, Any]]:
        return [event async for event in pool.run_turn("s1", Turn("hello"))]

    events = asyncio.run(run())
    assert events == [
        {"type": "session", "session_id": "s1", "model": "m.x"},
        {"type": "text", "text": "echo hello"},
        {"type": "end", "error": None},
    ]
    session = pool.fake_sessions["s1"]  # type: ignore[attr-defined]
    assert session.streams[0].closed


def test_turns_in_one_session_are_serialised_and_sessions_stay_separate(
    pool: SessionPool,
) -> None:
    async def run() -> None:
        async def collect(key: str, text: str) -> list[dict[str, Any]]:
            return [event async for event in pool.run_turn(key, Turn(text))]

        a1, a2, b1 = await asyncio.gather(
            collect("a", "one"), collect("a", "two"), collect("b", "three")
        )
        # Each turn saw exactly its own reply, despite sharing a's event queue.
        assert a1[1]["text"] == "echo one" and a2[1]["text"] == "echo two"
        assert b1[1]["text"] == "echo three"

    asyncio.run(run())
    assert pool.fake_sessions["a"].prompts == ["one", "two"]  # type: ignore[attr-defined]
    assert pool.fake_sessions["b"].prompts == ["three"]  # type: ignore[attr-defined]


def test_a_prompt_that_raises_before_emitting_surfaces_instead_of_hanging(
    pool: SessionPool,
) -> None:
    """R1: a refused turn must raise out of run_turn, not wait for an agent_end."""
    seen: list[dict[str, Any]] = []

    async def run() -> None:
        async for event in pool.run_turn("refuse-1", Turn("hello")):
            seen.append(event)

    async def bounded() -> None:
        # 5 s is far above the microseconds this takes; the pre-fix code never returned.
        await asyncio.wait_for(run(), timeout=5)

    with pytest.raises(TurnInFlight, match="already running"):
        asyncio.run(bounded())
    assert [event["type"] for event in seen] == ["session"]
    session = pool.fake_sessions["refuse-1"]  # type: ignore[attr-defined]
    assert session.streams[0].closed
    # The per-session lock is released, so the next turn is not queued forever.
    with pytest.raises(TurnInFlight):
        asyncio.run(bounded())


def test_a_model_on_an_existing_session_is_reported_not_silently_ignored(
    pool: SessionPool,
) -> None:
    """R2: the session frame names the model in effect and flags an ignored request."""

    async def first_frame(turn: Turn) -> dict[str, Any]:
        return [event async for event in pool.run_turn("s1", turn)][0]

    opened = asyncio.run(first_frame(Turn("one", "m.first")))
    assert opened == {"type": "session", "session_id": "s1", "model": "m.first"}
    same = asyncio.run(first_frame(Turn("two", "m.first")))
    assert "notice" not in same
    changed = asyncio.run(first_frame(Turn("three", "m.other")))
    assert changed["model"] == "m.first"
    assert "'m.other' ignored" in changed["notice"]


def test_session_event_without_a_requested_model_has_no_notice() -> None:
    assert lop_runtime.session_event("k", "m.x", None) == {
        "type": "session",
        "session_id": "k",
        "model": "m.x",
    }
