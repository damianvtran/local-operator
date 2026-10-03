"""End-to-end: answering a queued ask on a session whose runtime is STOPPED.

Why this file exists
--------------------

The operator's report was a phone reaching a stopped session: the ask card said
"session not connected", the answer was accepted by the surface, and no model
was ever told. The fix is that a queued ask's answer ENGAGES the session through
the same path a prompt uses (design ``docs/design/ask-nonblocking.md`` §2.4).

``tests/unit/mobile/test_asks_relay.py`` pins the routing at the seam (the right
errand, the right op). THIS file is the other half and the one that would have
caught the bug: a REAL relay daemon, over a REAL HTTP request, engaging a REAL
runtime child that boots the mock provider, against a REAL ``asks.jsonl`` and
transcript on disk — then the response row the model was waiting for must
actually land. A stubbed engage cannot prove that boot reconcile plus the op
deliver anything.

Isolation: the autouse ``headless_tui_env`` fixture points
``LOCAL_OPERATOR_CONFIG_DIR`` at a per-test directory and the root conftest
repoints ``HOME``, and the child inherits both, so nothing here reads or writes
the operator's own store. The runtime this engages is a real process and is
reaped by pid in the ``finally`` (see ``_reap``).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import time
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from local_operator.asks import store as ask_store
from local_operator.harness.types import Message
from local_operator.mobile.daemon import MobileDaemon, build_app
from local_operator.providers.registry import is_mock_provider
from local_operator.session.runtime import registry
from local_operator.session.transcript import Transcript
from tests.e2e.watchdog import bounded

pytestmark = pytest.mark.e2e

SESSION = "coldask00001"
ASK_ID = "a-coldask00001"
QUESTION = "ship the queued-ask engage?"

#: Generous, because it covers a real process spawn and a real boot reconcile
#: on a fleet that runs dozens of suites at once (AGENTS.md). The watchdog
#: turns a hang into a named ``TimeoutError`` rather than a silent stall.
BOUND_S = 120.0


def _seed(config_dir: Path) -> Path:
    """A durable user session with one OPEN queued ask and no runtime.

    The transcript is seeded (not just the directory) because the relay's own
    gate is ``_durable_user_session_dir``: an engage is only possible for a
    conversation that exists, and a test that skipped the transcript would be
    asserting the "gone" refusal while believing it tested the engage.
    """
    # The child resolves the mock provider from the config file rather than from
    # a live API key, so this test spawns a runtime that cannot reach the
    # network (the shape ``tests/e2e/test_desktop_sessions.py`` seeds).
    (config_dir / "config.yml").write_text(
        "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n"
    )
    directory = config_dir / "sessions" / SESSION
    directory.mkdir(parents=True, exist_ok=True)
    asyncio.run(Transcript(directory).append_message(Message.user("hello", id="seed-user")))
    now = int(time.time() * 1000)
    ask_store.append_event(
        directory,
        {
            "v": ask_store.EVENT_SCHEMA,
            "kind": ask_store.EVENT_QUEUED,
            "ask_id": ASK_ID,
            "at": now,
            "created_at": now,
            # Well inside the window, so the answer is an ordinary answer and
            # not a late one — the late arm is covered by the queue's own e2e.
            "expires_at": now + 3_600_000,
            "timeout_s": 3600,
            "urgent": False,
            "tool_call_id": "",
            "questions": [
                {
                    "id": "q1",
                    "question": QUESTION,
                    "options": [{"label": "yes", "description": ""}],
                    "multi": False,
                    "secret": False,
                    "persist": False,
                }
            ],
        },
    )
    return directory


def _client() -> TestClient:
    """A logged-in client over the REAL relay app (its own isolated store)."""
    client = TestClient(build_app(MobileDaemon(port=0, password="pw123")), follow_redirects=False)
    assert client.post("/login", data={"password": "pw123"}).status_code in (200, 303)
    return client


def _events(directory: Path) -> list[dict[str, object]]:
    return ask_store.read_events(directory)


def _transcript_text(directory: Path) -> str:
    path = directory / "transcript.jsonl"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _wait_until(predicate, *, timeout_s: float, what: str) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.1)
    raise AssertionError(f"timed out after {timeout_s}s waiting for {what}")


def _reap() -> None:
    """Kill every runtime this test engaged, by exact pid.

    The engage deliberately leaves its runtime serving (that is what an engage
    is), so a test that did not reap would leave a real child process holding
    the transcript lease on the fleet — the leak the team brief forbids. Scoped
    to the test's own config root so a sibling session's runtimes are never
    touched.
    """
    for record, _state in registry.scan():
        pid = getattr(record, "pid", None)
        if isinstance(pid, int) and pid > 0:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(pid, signal.SIGTERM)
    # Give the children a beat to leave, then make sure.
    time.sleep(0.5)
    for record, _state in registry.scan():
        pid = getattr(record, "pid", None)
        if isinstance(pid, int) and pid > 0:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(pid, signal.SIGKILL)


def test_answering_a_queued_ask_on_a_stopped_session_engages_and_delivers(
    headless_tui_env: Path,
) -> None:
    """The reported bug, end to end: stopped session, pending ask, phone answer.

    Asserts the THREE things the engaging answer must produce: the durable log
    records the winner, the engaged runtime injects the response row the model
    was waiting for, and the HTTP reply reports delivery rather than pretending
    while nothing reads it.
    """
    directory = _seed(headless_tui_env)
    assert not (directory / ".session.pid").exists(), "the session must start STOPPED"
    client = _client()
    try:
        with bounded(BOUND_S, "cold relay answer engages a runtime and delivers"):
            reply = client.post(
                f"/api/sessions/{SESSION}/command",
                json={"op": "ask_respond", "ask_id": ASK_ID, "answers": {"q1": ["yes"]}},
            )
            assert reply.status_code == 200, reply.text
            assert reply.json().get("ok") is True, reply.text

            # (1) The durable log carries the single winner for this ask.
            _wait_until(
                lambda: any(
                    event.get("kind") == ask_store.EVENT_ANSWERED and event.get("ask_id") == ASK_ID
                    for event in _events(directory)
                ),
                timeout_s=60,
                what="the answered event to land in asks.jsonl",
            )

            # (2) THE POINT: the response row reached the transcript, which is
            # what the model reads. Without the engage there is no runtime to
            # write it and the answer stays durable but unread.
            row_id = ask_store.response_row_id(ASK_ID)
            _wait_until(
                lambda: row_id in _transcript_text(directory),
                timeout_s=60,
                what="the ask-response row to reach the transcript",
            )

        # (3) The row the model sees carries the question and the answer, so the
        # injected turn is the response to the ask rather than an anonymous
        # message (the shared render, not a surface-local reconstruction).
        row = next(
            json.loads(line)
            for line in _transcript_text(directory).splitlines()
            if json.loads(line).get("id") == row_id
        )
        payload = json.dumps(row)
        assert QUESTION in payload
        assert "yes" in payload
    finally:
        client.close()
        _reap()


def test_answering_a_vanished_conversation_says_it_can_never_be_read(
    headless_tui_env: Path,
) -> None:
    """THE NEGATIVE, over the real route: no transcript, so no engage can help.

    A deleted session must not read as a delivered answer. The refusal names the
    reason in one sentence instead of the generic "session not connected", which
    the phone would otherwise render as "reopen it" — advice that cannot work,
    because there is nothing left to reopen.
    """
    client = _client()
    try:
        reply = client.post(
            f"/api/sessions/{SESSION}/command",
            json={"op": "ask_respond", "ask_id": ASK_ID, "answers": {"q1": ["yes"]}},
        )
        assert reply.status_code == 409, reply.text
        assert reply.json()["code"] == "ask_session_gone"
        assert "no longer exists" in reply.json()["error"]
    finally:
        client.close()


def test_the_mock_hosting_is_what_the_child_would_run() -> None:
    """Guard the fixture's premise, cheaply.

    The engage test's runtime child must answer from the mock rather than reach
    a provider; if the registry stopped classifying ``test`` as mock, the cell
    above would silently start making real model calls. Asserted here so the
    isolation is a checked claim rather than an assumption.
    """
    assert is_mock_provider("test")
