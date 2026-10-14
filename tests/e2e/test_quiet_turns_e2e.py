"""End-to-end: the quiet turn over REAL sessions — and over a REAL kill.

Two facts are asserted here and nowhere else (docs/design/quiet-turns.md §8):

* **The scripted-provider real path.** A peer message arrives at an idle
  session, the model answers it with ``no_reply``, and the run leaves the
  system the way the design promises: the transcript holds the PAIR (the
  assistant's call and its stamped result), the attention store holds NO
  completion row — no unread mark, no banner — and no notifier was called.

* **The kill-between case (§7's accepted exposure).** The runtime dies with
  the pair persisted and the outcome marker unwritten — the window the design
  names "between result persistence and outcome publication". The successor's
  classification must report an ERROR (``runtime-killed``) that notifies: a
  death outranks a would-have-been silence, so a quiet turn is never silently
  dropped by a crash.

The provider is scripted for the reason ``test_ask_tool_e2e`` gives: both
stages' failure signals are about what the runtime leaves behind, and live
model latency inside the bounds would only make them slower to read. The kill
cell additionally runs the session in a CHILD process, because only a real
SIGKILL produces the on-disk state the successor has to classify.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import QUIET_TURN_KEY, AgentEndEvent
from local_operator.session.attention import AttentionStore, conversation_identity
from tests.e2e.harness import (
    NO_NOTIFY_ENV,
    ScriptedStream,
    build_session,
    dispose_quietly,
    text_turn,
    tool_call_turn,
)
from tests.e2e.watchdog import bounded

pytestmark = pytest.mark.e2e


def _payloads(directory: Path) -> list[dict[str, Any]]:
    """Every transcript row's payload, in order."""
    path = directory / "transcript.jsonl"
    if not path.exists():
        return []
    return [json.loads(line)["payload"] for line in path.read_text().splitlines() if line]


async def _wait_until(predicate: Any, timeout: float = 30.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"timed out waiting for {predicate!r}")


@pytest.mark.asyncio
async def test_a_peer_message_into_an_idle_session_can_end_quietly(
    tmp_path: Path, monkeypatch
) -> None:
    """§8's scripted-provider real path, whole: peer message in, quiet end out.

    The session is real (the REAL tool inventory, the REAL ``no_reply`` tool
    built the way a session builds it, a real transcript on disk); only the
    provider stream is scripted.
    """
    import local_operator.tui.notify as notify_module

    calls: list[tuple[str, str]] = []

    def record_notify(title: str, body: str, **kwargs: Any) -> bool:
        calls.append((title, body))
        return True

    monkeypatch.setattr(notify_module, "detached_notify", record_notify)
    directory = tmp_path / "sess"
    directory.mkdir(parents=True, exist_ok=True)
    stream = ScriptedStream(
        [tool_call_turn(text="", tool_name="no_reply", tool_call_id="q1", arguments={})]
    )
    session = build_session(directory, stream)
    # The inventory is the one a real session builds — mounted by the
    # constructor's own capability merge, NO hand-splice (round-1 M2: a spliced
    # inventory proves the tool works but never that a session has it, which is
    # exactly how B1 shipped). Asserted before the first prompt.
    assert "no_reply" in {tool.name for tool in session._tools}
    ends: list[Any] = []
    session.subscribe(
        lambda event: ends.append(event) if isinstance(event, AgentEndEvent) else None
    )
    with bounded(60, "quiet e2e: peer message into an idle session"):
        try:
            await session.receive_peer_message(
                "child reporting in",
                mode="mailbox",
                wake=True,
                sender={"pid": 42, "conversation_name": "child"},
            )
            await _wait_until(lambda: bool(ends))
            await _wait_until(lambda: session._attention_run_settled)

            assert len(stream.requests) == 1, "the quiet end bought no further call"
            assert ends[-1].notify is False, "the value every notifier reads"

            # The transcript holds the PAIR: the assistant's call...
            rows = _payloads(directory)
            assert any(
                call.get("name") == "no_reply"
                for payload in rows
                for call in (payload.get("tool_calls") or ())
            ), "the assistant call is persisted"
            # ...and its result, with the marker.
            assert any(
                ((payload.get("provider_payload") or {}).get("details") or {}).get(QUIET_TURN_KEY)
                is True
                for payload in rows
                if payload.get("role") == "tool"
            ), "the stamped result is persisted"

            # The store holds NO completion row — no unread mark, no banner.
            state = await session.refresh_attention()
            assert state["completion_token"] is None, "a quiet end publishes nothing"
            assert state["unseen"] is False

            # And nothing asked a notifier to say anything.
            assert calls == [], "no notifier call"
        finally:
            await dispose_quietly(session)


@pytest.mark.asyncio
async def test_a_peer_message_answered_with_text_stays_quiet(tmp_path: Path, monkeypatch) -> None:
    """The TEXT twin of the cell above: the reply is ordinary prose, not
    ``no_reply`` — the write that used to re-arm the banner for every peer
    turn. Over the same real path: the end carries notify=False, the reply is
    persisted for the reader, the completion row exists and stays unread (the
    discovery surface is deliberately separate from the announcement), and no
    notifier is called.
    """
    import local_operator.tui.notify as notify_module

    calls: list[tuple[str, str]] = []

    def record_notify(title: str, body: str, **kwargs: Any) -> bool:
        calls.append((title, body))
        return True

    monkeypatch.setattr(notify_module, "detached_notify", record_notify)
    directory = tmp_path / "sess-text"
    directory.mkdir(parents=True, exist_ok=True)
    stream = ScriptedStream([text_turn("the peer's answer, in prose")])
    session = build_session(directory, stream)
    ends: list[Any] = []
    session.subscribe(
        lambda event: ends.append(event) if isinstance(event, AgentEndEvent) else None
    )
    with bounded(60, "quiet e2e: peer message answered with text"):
        try:
            await session.receive_peer_message(
                "child reporting in",
                mode="mailbox",
                wake=True,
                sender={"pid": 42, "conversation_name": "child"},
            )
            await _wait_until(lambda: bool(ends))
            await _wait_until(lambda: session._attention_run_settled)

            assert len(stream.requests) == 1, "the reply bought no further call"
            assert ends[-1].notify is False, "a text reply to a peer raises no banner"

            # The reply is persisted — the operator opens the session and reads it.
            rows = _payloads(directory)
            assert any(
                payload.get("role") == "assistant"
                and any(
                    "prose" in str(part.get("text", "")) for part in (payload.get("content") or [])
                )
                for payload in rows
            ), "the text reply is persisted"

            # The row EXISTS and is unread: the unread mark is the discovery
            # surface, and it is deliberately independent of `notify`.
            state = await session.refresh_attention()
            assert state["completion_token"] is not None, "a quiet row for the unread mark"
            assert state["notify"] is False, "the one value every notifier reads"
            assert state["unseen"] is True, "still unread; not announced"

            # And nothing asked a notifier to say anything.
            assert calls == [], "no notifier call"
        finally:
            await dispose_quietly(session)


#: The kill cell's child driver. Spawned via ``sys.executable``; it builds a
#: REAL session over the same directory (the real ``no_reply`` tool, a real
#: discovery record as the runtime publishes for itself) and PARKS inside the
#: window the cell is about: its subscriber receives the end event and never
#: returns, so the pair is on disk (the flush runs after ``_persist_new_
#: messages``) while the outcome marker is still unpublished (the publish is
#: the pipeline's NEXT step). The parent's SIGKILL then reproduces "died
#: between result and marker" exactly.
_DRIVER = """
import asyncio
import os
import sys
from pathlib import Path

from local_operator.harness.types import (
    AgentEndEvent,
    ModelSpec,
    StreamEndEvent,
    StreamToolCallDelta,
)
from local_operator.session.runtime.registry import RecordPublisher
from local_operator.session.runtime.types import SessionRecord
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript

directory = Path(sys.argv[1])


def stream_fn(request, signal=None):
    async def gen():
        yield StreamToolCallDelta(index=0, id="q1", name="no_reply", argument_delta="{}")
        yield StreamEndEvent(stop_reason="toolUse")

    return gen()


async def main():
    session = Session(
        model=ModelSpec(provider="test", model_id="quiet-kill", context_window=100_000),
        stream_fn=stream_fn,
        tools=[],
        transcript=Transcript(directory),
        system_blocks_provider=lambda: ["stable"],
        cwd=str(directory),
        yolo=True,
    )
    await session.async_init()
    # Mounted by the constructor's own merge (round-1 M2) — no hand-splice;
    # this is the inventory a real runtime boots with, and the cell's premise
    # depends on it holding the tool.
    assert "no_reply" in {tool.name for tool in session._tools}
    publisher = RecordPublisher(
        SessionRecord(
            pid=os.getpid(),
            kind="exec",
            session_id=directory.name,
            conversation_name="",
            cwd=str(directory),
            model_label="",
            control_port=0,
            control_key="k",
        )
    )
    assert publisher.path.exists()

    async def park_on_end(event):
        # The kill window itself: the pair is already persisted (the end emit
        # is the pipeline's flush, which runs after _persist_new_messages) and
        # _publish_attention_outcome has not run yet.
        if isinstance(event, AgentEndEvent):
            await asyncio.Event().wait()

    session.subscribe(park_on_end)
    await session.receive_peer_message(
        "beep", mode="mailbox", wake=True, sender={"pid": 1, "conversation_name": "driver"}
    )
    await asyncio.Event().wait()


asyncio.run(main())
"""


def _child_env(config: Path) -> dict[str, str]:
    """The driver's environment: a real runtime's, minus the pane families.

    ``CMUX_*`` and the ``LOP_*`` runtime families are stripped (a child that
    inherited a workspace id could address the operator's live window; an
    inherited ``LOP_RUNTIME_*`` could make it ADOPT a session), then the two
    values this cell means to set are set — the config root and the
    notification gate, re-asserted AFTER the strip so nothing inherited can
    put the gate back.
    """
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("CMUX_", "LOP_MOBILE_CHILD_", "LOP_RUNTIME_"))
    }
    env.update(NO_NOTIFY_ENV)
    env["LOCAL_OPERATOR_CONFIG_DIR"] = str(config)
    return env


async def _wait_for_pair(directory: Path, child: "subprocess.Popen[bytes]") -> None:
    """Wait until the quiet pair — and nothing after it — is on disk."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 30.0
    while loop.time() < deadline:
        if child.poll() is not None:
            raise AssertionError(f"the driver exited early (rc={child.returncode})")
        path = directory / "transcript.jsonl"
        if path.exists() and QUIET_TURN_KEY in path.read_text():
            return
        await asyncio.sleep(0.05)
    raise AssertionError("the quiet pair never reached the transcript")


@pytest.mark.asyncio
async def test_a_kill_between_the_result_and_the_marker_still_reports_an_error(
    headless_tui_env: Path, tmp_path: Path
) -> None:
    """§7's kill-between case with REAL processes.

    The runtime dies between the persisted pair and the unwritten outcome
    marker; the successor boot classifies the orphaned run as
    ``error``/``runtime-killed`` and NOTIFIES. This is the design's recorded
    exposure pinned from the accepting side: the death is never silent.
    """
    config = headless_tui_env
    session_id = "quietkill01"
    directory = config / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    driver = tmp_path / "quiet_driver.py"
    driver.write_text(_DRIVER)
    child = subprocess.Popen(
        [sys.executable, str(driver), str(directory)],
        env=_child_env(config),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        with bounded(120, "quiet e2e: kill between result and marker"):
            await _wait_for_pair(directory, child)
            # The premise of the cell, asserted rather than assumed: the pair
            # is on disk and the marker is NOT — that is the window.
            await asyncio.sleep(0.5)
            text = (directory / "transcript.jsonl").read_text()
            assert QUIET_TURN_KEY in text
            assert "completion_attention" not in text, (
                "the kill must land before the outcome marker; if this fires, the "
                "driver did not park and the cell reproduces nothing"
            )

            os.kill(child.pid, 9)
            child.wait(timeout=10)
            await asyncio.sleep(0.2)

            # The successor boot is what classifies and narrates the orphaned
            # run (the same two-step the cut-off suite uses).
            session = build_session(directory, ScriptedStream([text_turn("ok")]))
            await session.async_init()
            try:
                state = AttentionStore().state(conversation_identity(directory))
                assert state["kind"] == "error", state
                assert state["cause"] == "runtime-killed", state
                assert state["notify"] is True, "a cut-off always notifies"
                assert state["reason"], "a cut-off must name a reason"
                assert state["completion_token"] is not None
            finally:
                await dispose_quietly(session)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
