"""End-to-end: a credential printed by a REAL command never reaches the model.

**Why this file and not only the unit corpus.** The unit cases prove the rules
mask a spelling; they cannot prove the WIRING — that a real ``Session`` running
the real loop, through the real ``bash`` tool over a real subprocess, with a real
transcript on disk, actually routes every one of those surfaces through the
scrubber. The defect this closes was not a missing rule but a missing seam: the
masking that existed lived on the HTTP clients, and the harness had no fallback,
so a production DSN printed by ``kubectl exec … env`` was persisted in full.

So this drives the assembled path: a scripted provider asks for one bash command
that prints an incident-shaped line, the real tool runs it, and the assertions
read what was actually produced — the request the provider was handed on the
NEXT turn, the transcript file on disk, and the incident row. Nothing is
asserted from an intermediate object the production code would not itself use.

**The command carries the credential too.** It is both printed by the command
and typed into the call, so this one turn exercises the result path AND the
journaled/replayed tool-call arguments — the second of which had no redaction on
it at all before this change.

The credential here is synthetic and its value is never printed: the assertions
are about its ABSENCE, which is the property under test.
"""

from __future__ import annotations

import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any, Iterator

import pytest

from local_operator.harness.types import (
    AbortSignal,
    AgentToolUpdate,
    NoticeEvent,
    TextContent,
    ToolContext,
)
from local_operator.tools.builtin import build_bash_tool, execute_bash
from local_operator.variables import VariableStore
from tests.e2e.harness import (
    ScriptedStream,
    build_session,
    dispose_quietly,
    text_turn,
    tool_call_turn,
)

pytestmark = pytest.mark.e2e

#: The credential's password half. Synthetic, and never asserted FOR: every
#: assertion below is that this string is absent from something.
SENTINEL_PW = "sh4pedE2eSentinelPw"

#: The incident's shape, verbatim apart from the credential: a remote pod's
#: environment printed by a command whose own masking had no ``DSN`` pattern.
SENTINEL_LINE = (
    f"MONGO_DSN=mongodb+srv://agent_runtime_model_worker:{SENTINEL_PW}"
    "@mongodb-prod.example.net/agent_runtime"
)


@pytest.fixture(autouse=True)
def _isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Strip the inherited multiplexer/session identity and pin a scratch home.

    An inherited ``CMUX_WORKSPACE_ID`` once let a headless test rename the
    operator's real cmux workspaces, and an inherited ``LOP_*`` can point a
    child at a live session. This file boots a real Session over a real tool, so
    it strips both families and uses synthetic ids only.
    """
    for name in list(os.environ):
        if name.startswith(("CMUX_", "LOP_")):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    yield


def _provider_saw(stream: ScriptedStream) -> str:
    """Every byte the provider was handed on the LAST request, as JSON.

    The strongest available reading of "model-visible": this is the request
    object itself, after the production converter, so a credential surviving in
    a tool result, in an assistant turn's tool-call arguments, or in a replayed
    message all show up here.
    """
    assert stream.requests, "the scripted provider was never called"
    return json.dumps(stream.requests[-1].model_dump(mode="json"), default=str)


def _transcript_text(directory: Path) -> str:
    return (directory / "transcript.jsonl").read_text()


@pytest.mark.asyncio
async def test_a_real_command_that_prints_a_dsn_never_reaches_the_model(
    headless_tui_env: Path, workspace: Path
) -> None:
    directory = headless_tui_env / "sessions" / "shape-e2e"
    command = f"echo {shlex.quote(SENTINEL_LINE)}"
    stream = ScriptedStream(
        [
            tool_call_turn(
                text="checking the environment",
                tool_name="bash",
                tool_call_id="call-1",
                arguments={"command": command},
            ),
            text_turn("done"),
        ]
    )
    session = build_session(directory, stream, tools=[build_bash_tool()], cwd=workspace)
    events: list[Any] = []
    session.subscribe(events.append)
    await session.async_init()
    try:
        await session.prompt("print the environment")
        # The command really ran: a masked DSN in the result proves the tool
        # executed it AND that the result was rewritten rather than withheld.
        assert stream.requests, "the loop never reached the provider"
        assert len(stream.requests) >= 2, "the tool batch never produced a second call"
    finally:
        await dispose_quietly(session)

    # 1. The request the provider was handed: no credential anywhere in it.
    assert SENTINEL_PW not in _provider_saw(stream)

    # 1b. ...and the operator was TOLD, live. The incident row's whole purpose is
    # the rotation ticket, and a row that reaches only the model is not one:
    # measured before this wiring, the persisted row painted on no operator
    # surface at all (design round 1, D1).
    notices = [e for e in events if isinstance(e, NoticeEvent)]
    assert notices, "no live receipt for the masked credential"
    assert notices[0].kind == "warning"
    assert "dsn-password" in notices[0].text, notices[0].text

    # 2. The transcript on disk: the persistence surface, and the incident row.
    body = _transcript_text(directory)
    assert SENTINEL_PW not in body, "the credential reached transcript.jsonl"
    assert "session_incident" in body, "no rotation ticket was journalled"
    assert "dsn-password" in body, "the incident must name the shape that fired"

    # 3. The masking is visible rather than silent: the model is told the value
    #    was hidden, so it cannot misread an empty-looking result.
    assert "[redacted]" in body
    assert "rotate" in body


def _collect(sink: list[str]) -> Any:
    """An ``on_update`` sink that flattens a streamed update to its text.

    Text only: the surface under test is the bytes a UI paints, and the
    ``details`` of an update are not painted.
    """

    def sink_update(update: AgentToolUpdate) -> None:
        sink.append(
            "".join(block.text for block in update.content if isinstance(block, TextContent))
        )

    return sink_update


@pytest.mark.asyncio
async def test_the_live_stream_of_a_real_command_carries_no_shape(
    tmp_path: Path,
) -> None:
    """The live surface, on a real subprocess and a real pipe.

    The pipe filter is the only guard on the bytes a UI paints while a command
    is still running — there is no finished result to scrub yet. The command
    prints its line in three writes with a flush between them, so the value
    straddles reads exactly as a real child's buffering would, and then KEEPS
    RUNNING past the tool's 0.5 s update tick: the live snapshot is emitted on
    that cadence, so a command that finished in 100 ms would leave ``updates``
    empty and this test asserting nothing at all. The non-empty assertion below
    is what keeps the guard from being vacuous — measured: without the trailing
    sleep the collected updates are ``[]``.
    """
    updates: list[str] = []
    script = (
        "import sys, time\n"
        f"sys.stdout.write({SENTINEL_LINE[:20]!r}); sys.stdout.flush(); time.sleep(0.05)\n"
        f"sys.stdout.write({SENTINEL_LINE[20:]!r}); sys.stdout.flush(); time.sleep(0.05)\n"
        "sys.stdout.write('\\n'); sys.stdout.flush()\n"
        "time.sleep(1.2)\n"
    )
    context = ToolContext(
        cwd=str(tmp_path), variables=VariableStore(cwd=str(tmp_path)), session_id="s"
    )
    result = await execute_bash(
        "bash-live",
        {"command": f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"},
        AbortSignal(),
        _collect(updates),
        context,
    )
    assert not result.is_error, result.text

    assert updates, "the tool never painted a live update — this guard is vacuous"
    live = "\n".join(updates)
    assert SENTINEL_PW not in live, "the live stream painted the credential"
    assert SENTINEL_PW not in result.text, "the finished result carried the credential"
    assert "[redacted]" in result.text


@pytest.mark.e2e
async def test_an_output_only_credential_still_files_its_incident(
    headless_tui_env: Path, workspace: Path
) -> None:
    """The incident this feature exists for: the credential is ONLY in the output.

    The production shape — ``kubectl exec … env`` — has the credential in the
    command's OUTPUT and nowhere in the command text. The pipe filter masks those
    bytes while the command is still running, so by the time the loop's
    ``redact_tool_result`` hook reads the finished result the credential is gone,
    the shape pass finds nothing, and no incident is filed: measured before this
    test existed, the row count for exactly this case was ZERO.

    The command therefore carries a FILENAME and nothing else; the credential is
    read out of the file by the child, which is the position the real incident was
    in.
    """
    directory = headless_tui_env / "sessions" / "output-only"
    password = "qa-output-only-4b7f"
    (workspace / "agent.env").write_text(f"MONGO_DSN=mongodb+srv://svc:{password}@db.invalid/x\n")
    stream = ScriptedStream(
        [
            tool_call_turn(
                text="reading the environment",
                tool_name="bash",
                tool_call_id="call-out",
                arguments={"command": "grep MONGO_DSN agent.env"},
            ),
            text_turn("done"),
        ]
    )
    session = build_session(directory, stream, tools=[build_bash_tool()], cwd=workspace)
    events: list[Any] = []
    session.subscribe(events.append)
    await session.async_init()
    try:
        await session.prompt("print the environment")
    finally:
        await dispose_quietly(session)

    body = (directory / "transcript.jsonl").read_text()
    assert password not in body, "the output-only credential reached the transcript"
    assert "[redacted]" in body, "the mask did not happen at all"
    rows = [line for line in body.splitlines() if "session_incident" in line]
    assert rows, "an output-only credential filed no incident row"
    notices = [event for event in events if isinstance(event, NoticeEvent)]
    assert notices, "an output-only credential produced no live notice"
    assert notices[0].kind == "warning"
