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
from local_operator.redaction_shapes import REDACTION_MARKER
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

    # 1b. ...and NOTHING is indicated, live or on disk. This result was masked
    # WHOLE, which is the contained case: the value never entered the model's
    # context, so there is nothing to rotate and nothing the operator has to be
    # told — "as long as something wasn't actually leaked to the transcript we
    # shouldn't get a session incident indicated anywhere". Measured before this
    # change: a warning notice AND a journalled incident row, for a value the
    # model never read.
    notices = [e for e in events if isinstance(e, NoticeEvent)]
    assert not notices, [notice.text for notice in notices]

    # 2. The transcript on disk: the persistence surface, and no notice row.
    body = _transcript_text(directory)
    assert SENTINEL_PW not in body, "the credential reached transcript.jsonl"
    assert "session_incident" not in body, "a contained value filed an incident"
    assert "session_credential_redaction" not in body, "a contained value filed a notice"

    # 3. The instrument is not dead. The mask really was written into the
    #    transcript, so the absence above cannot be a session that never ran the
    #    pass — and the word the rotation demand used is gone with it.
    assert "[redacted]" in body
    assert "rotate" not in body, "a contained value demanded a rotation"


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
@pytest.mark.asyncio
async def test_an_output_only_credential_files_nothing_when_it_is_masked_whole(
    headless_tui_env: Path, workspace: Path
) -> None:
    """The output-only shape, contained: masked, and NOT an incident.

    The production shape — ``kubectl exec … env`` — has the credential in the
    command's OUTPUT and nowhere in the command text. The pipe filter masks those
    bytes while the command is still running, so by the time the loop's
    ``redact_tool_result`` hook reads the finished result the credential is gone.
    The pipe is therefore the only layer that can report such a hit — and this
    value is masked WHOLE, so it reports nothing: no row, no live notice. What the
    test still pins is that the mask HAPPENED, because an absence assertion that
    cannot tell "silent" from "never ran" is no evidence at all. The opposite
    direction is the next test.

    The command therefore carries a FILENAME and nothing else; the credential is
    read out of the file by the child, which is the position the real incident was
    in.
    """
    directory = headless_tui_env / "sessions" / "output-only"
    (workspace / "agent.env").write_text(
        f"MONGO_DSN=mongodb+srv://svc:{SENTINEL_PW}@db.invalid/x\n"
    )
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
    assert SENTINEL_PW not in body, "the output-only credential reached the transcript"
    assert REDACTION_MARKER in body, "the mask did not happen at all"
    rows = [line for line in body.splitlines() if "session_credential_redaction" in line]
    assert not rows, "a contained output-only credential filed a notice"
    notices = [event for event in events if isinstance(event, NoticeEvent)]
    assert not notices, "a contained output-only credential produced a live notice"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_an_output_only_EXPOSURE_still_files_its_incident(
    headless_tui_env: Path, workspace: Path
) -> None:
    """The other direction, and the one a too-eager gate would eat.

    A DSN whose username IS its password is the documented escalating shape: the
    DSN rule keeps the userinfo username readable by design, so the value's own
    characters end up in the text the model reads. Fed in as OUTPUT ONLY — the
    credential is in the file the child reads, never in the command — through the
    same pipe, so this exercises the pipe's own report on a real subprocess.

    The incident this feature was written for is this one. A gate that swallowed
    every output-only hit would look exactly like a working one from the test
    above, which is why both directions are pinned.
    """
    directory = headless_tui_env / "sessions" / "output-only-exposed"
    payload = "amqp://guest:guest@rabbit.invalid:5672/"
    (workspace / "exposed.env").write_text(f"AMQP_DSN={payload}\n")
    stream = ScriptedStream(
        [
            tool_call_turn(
                text="reading the environment",
                tool_name="bash",
                tool_call_id="call-exposed",
                arguments={"command": "grep AMQP_DSN exposed.env"},
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
    rows = [line for line in body.splitlines() if "session_credential_redaction" in line]
    assert rows, "an output-only exposure filed no notice row"
    notices = [event for event in events if isinstance(event, NoticeEvent)]
    assert notices, "an output-only exposure produced no live notice"
    assert notices[0].kind == "warning"
    assert "rotate" in notices[0].text, notices[0].text

    # THE MODEL MUST NOT SEE IT, and this is the real-command half of the
    # operator's instruction: the notice is the OPERATOR's ticket, and the value
    # it names was masked out of the text the model reads on the next turn. The
    # request the provider was handed is the strongest reading of "model-visible"
    # (after the production converter), so a notice surviving in the next turn's
    # context shows up here. Measured before this change: 1,493 unnamed notices
    # across 1,080 sessions rode exactly this path into the model's context.
    seen = _provider_saw(stream)
    assert "[credential redaction]" not in seen, "the notice reached the model"
    assert "rotate it" not in seen, "the notice reached the model"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_the_publish_script_an_agent_authors_reaches_the_model_readable(
    headless_tui_env: Path, workspace: Path
) -> None:
    """The operator's workflow end to end: author a script from what was displayed.

    Reported 2026-09-22. ``lop secret run`` with a store NAME is how
    ``guide://credentials`` hands a stored secret to a child, and an operator names
    an entry after the SYSTEM it belongs to — the reported one ends in USERNAME,
    which is not one of the credential words the pass used to require. So every tool
    result masked that name, and the script the agent then authored from the text it
    read asked the store for a secret literally named ``[redacted]``, which cannot
    work. This drives the real loop over the real ``bash`` tool and reads what the
    provider was handed, what the transcript holds, and what the file holds.

    The control lives in a file the TEST seeds, so it is in the command's OUTPUT
    while never being in the model's own call: a synthetic issuer token, which must
    still be masked. That is what makes the NAME assertion evidence of a pass that
    ran rather than of text nothing looked at.
    """
    directory = headless_tui_env / "sessions" / "name-e2e"
    store_name = "_".join(("MINERVA", "UI", "NPROD", "USERNAME"))
    control = "ghp" + "_AbCd1234EfGhIjKlMnOpQr"
    # Assembled so no literal in this SOURCE is a flag followed by its argument.
    command = "lop secret run " + "--" + "secret " + store_name + " -- npm publish"
    script = "#!/bin/sh" + chr(10) + "# publish the UI package" + chr(10) + command + chr(10)
    (workspace / "secrets.env").write_text("API_TOKEN=" + control + chr(10))

    stream = ScriptedStream(
        [
            tool_call_turn(
                text="writing the publish script",
                tool_name="bash",
                tool_call_id="call-1",
                arguments={
                    "command": "cat > publish.sh <<'SH'" + chr(10) + script + "SH" + chr(10)
                },
            ),
            tool_call_turn(
                text="reading it back",
                tool_name="bash",
                tool_call_id="call-2",
                arguments={"command": "cat publish.sh secrets.env"},
            ),
            text_turn("done"),
        ]
    )
    session = build_session(directory, stream, tools=[build_bash_tool()], cwd=workspace)
    events: list[Any] = []
    session.subscribe(events.append)
    await session.async_init()
    try:
        await session.prompt("write the publish script and read it back")
    finally:
        await dispose_quietly(session)

    # 1. The NAME survives where it is a NAME — the thing the agent has to copy.
    assert store_name in _provider_saw(stream), "the store NAME never reached the model"
    body = _transcript_text(directory)
    assert store_name in body, "the store NAME was rewritten in transcript.jsonl"

    # 2. The script on disk, as the next `cat` will render it.
    assert store_name in (workspace / "publish.sh").read_text()

    # 3. The control, in the SAME result: the pass ran, and it still masks a value.
    assert control not in _provider_saw(stream), "the synthetic token reached the model"
    assert control not in body, "the synthetic token reached the transcript"
    assert "[redacted]" in body, "the control was not masked: the pass never ran"

    # 4. A whole mask is the contained case, so nothing is indicated and nothing is
    #    demanded: the failure this pins cost no incident, which is why only the
    #    workflow found it. Both spellings are asserted because BOTH records now
    #    carry their own type: ``session_incident`` covers a real failed turn and
    #    ``session_credential_redaction`` covers the shape notice that left it.
    assert not [event for event in events if isinstance(event, NoticeEvent)]
    assert "session_incident" not in body
    assert "session_credential_redaction" not in body
