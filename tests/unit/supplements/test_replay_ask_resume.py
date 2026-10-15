"""Replay of a real turn that Highlights missed: resumed after an answered ask, wrote four
deliverables from bash, linked one of them only in prose.

THE INCIDENT (fixture ``replay/ask_resume_turn.json``, a trimmed and identifier-scrubbed copy of
transcript idx 1232-1272). The turn was opened by an ``ask_response`` the operator answered;
its work was ``cd ~/workspace/<pkg>; cp ...; python3 - <<PY ... open('RUN-RECORD.md','w')``,
a ``json.dump(d, open(p, 'w'))`` on a variable path and ``shasum ... > SHA256SUMS``; its final
answer named ``RUN-RECORD.md`` only inside backticks. Three independent defects hid it: the
ask-opened turn was refused as ``no-typed-user-row``; the bash parser could not lex the
heredoc (so it named nothing and ignored the ``cd``); and the linked file was deduped as
"already visible". Each test below fails on the pre-fix tree for exactly one of those.

REAL FILES, REAL SESSION. The pre-filter ``stat``s, so the four files are created in a tmp
HOME with the cwd at ``~`` (as in the incident), and the end-to-end cell drives a real
``Session`` + ``ServingSessionHandle`` through ``prompt`` -> ``deliver_ask_messages``.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import types
from pathlib import Path
from typing import Any

import pytest

from local_operator.classification.types import Answer
from local_operator.harness.message_types import ASK_RESPONSE_MESSAGE_TYPE
from local_operator.harness.types import (
    AgentTool,
    CustomMessage,
    StreamEndEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    TextContent,
    ToolResult,
)
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.supplements import decision as dec
from local_operator.supplements.candidates import prefilter
from local_operator.supplements.contract import SUPPLEMENT_CUSTOM_TYPE
from local_operator.supplements.trigger import snapshot_message
from tests.unit.session.test_session import ScriptedStream, wait_for
from tests.unit.supplements.test_runner import _await_row, _make_session

from .conftest import load

PACKAGE = "acme-bulk-closeout-20261014"
DELIVERABLES = (
    "RUN-RECORD.md",
    "SHA256SUMS",
    "reversal-ids.json",
    "write-ledger-supplementary-comments.jsonl",
)


def _messages() -> list[dict[str, Any]]:
    return load("replay/ask_resume_turn.json")["messages"]


def _item(row: dict[str, Any]):
    """A fixture row as the session's own ``snapshot_message`` reduces a harness message."""
    return snapshot_message(
        types.SimpleNamespace(
            id=row["id"],
            role=row["role"],
            content=[types.SimpleNamespace(type="text", text=row["text"] or "")],
            tool_calls=[
                types.SimpleNamespace(id=c["id"], name=c["name"], arguments=c["arguments"])
                for c in row["tool_calls"]
            ],
            tool_call_id=row["tool_call_id"],
            tool_name="",
            # The ``ask`` call the model got wrong first is a real errored result in the turn.
            is_error=row["role"] == "tool" and (row["text"] or "").startswith("invalid arguments"),
        )
    )


def _turn_after_the_ask():
    """The assistant/tool messages from the ask answer on, and the final answer text."""
    rows = _messages()
    start = next(i for i, r in enumerate(rows) if r["custom_type"] == "ask_response")
    body = [r for r in rows[start + 1 :] if r["role"] in ("assistant", "tool")]
    return [_item(r) for r in body[:-1]], rows[-1]["text"]


@pytest.fixture
def home(tmp_path: Path, monkeypatch) -> Path:
    """The incident's disk: cwd == ``~``, the package under ``~/workspace``."""
    root = tmp_path / "home"
    package = root / "workspace" / PACKAGE
    package.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(root))
    old = time.time() - 7200
    for name in ("rationale-gap-226-profiles.csv", "OLDER-NOTES.md"):  # before the turn
        (package / name).write_text("earlier\n")
        os.utime(package / name, (old, old))
    return root


def _write_deliverables(home: Path) -> None:
    package = home / "workspace" / PACKAGE
    for name in DELIVERABLES:
        (package / name).write_text(f"{name}\n" * 20)
    (package / ".env").write_text("TOKEN=not-for-callouts\n")  # written, but never offered
    (package / "debug.log").write_text("noise\n")


def test_the_replayed_turn_surfaces_all_four_deliverables_with_workspace_paths(
    home: Path,
) -> None:
    items, answer = _turn_after_the_ask()
    started = time.time() - 5
    _write_deliverables(home)
    out = prefilter(items, answer, cwd=str(home), home=str(home), since=started)

    by_name = {c.name: c for c in out.candidates}
    assert set(by_name) == set(DELIVERABLES), sorted(by_name)
    assert {c.path for c in out.candidates} == {f"workspace/{PACKAGE}/{n}" for n in DELIVERABLES}
    assert all(not os.path.isabs(c.path) for c in out.candidates)
    # RUN-RECORD.md is linked in the answer's prose and is STILL offered, flagged.
    assert by_name["RUN-RECORD.md"].linked_in_answer is True
    assert by_name["SHA256SUMS"].kind == "text" and by_name["SHA256SUMS"].deliverable
    # Negative controls on the same directory: dotfiles, logs and pre-turn files stay out.
    assert not {".env", "debug.log", "OLDER-NOTES.md", "rationale-gap-226-profiles.csv"} & set(
        by_name
    )
    assert out.skipped is None


def test_this_turns_answer_yields_table_only_labelled_summary(
    home: Path,
) -> None:
    """Heterogeneous operational counts form a labelled table, not a comparable series."""
    items, answer = _turn_after_the_ask()
    _write_deliverables(home)
    out = prefilter(items, answer, cwd=str(home), home=str(home), since=time.time() - 5)
    assert out.evidence.structured is True
    assert out.evidence.forms == ("labelled-summary",)
    dataset = out.evidence.datasets[0]
    assert dataset.presentation == "table-only"
    assert {row[0]: row[1] for row in dataset.rows} == {
        "comments": "296",
        "profiles": "225",
        "links": "325",
        "matches": "3194",
    }


class _Vendor:
    """Records the questions asked and answers the files question with every option."""

    vendor_name = "radient"

    def __init__(self) -> None:
        self.asked: list[str] = []

    async def decide(self, *, state: str, question: Any):
        self.asked.append(question.id)
        if question.id == dec.FILES_QUESTION_ID:
            ids = [k for k in question.criteria if k != "none"]
            return Answer(
                id=question.id, kind="choice", value=ids[0], probabilities={k: 0.9 for k in ids}
            )
        return Answer(id=question.id, kind="noul", value=0.99)


@pytest.mark.asyncio
async def test_the_decision_features_files_and_can_select_the_labelled_table(home: Path) -> None:
    items, answer = _turn_after_the_ask()
    _write_deliverables(home)
    out = prefilter(items, answer, cwd=str(home), home=str(home), since=time.time() - 5)
    vendor = _Vendor()
    decision = await dec.decide(
        vendor,
        user_text="close the gap",
        answer_text=answer,
        candidates=out.candidates,
        evidence=out.evidence,
        want_files=True,
        want_graphics=True,
        max_featured=4,
    )
    assert {c.name for c in decision.featured} == set(DELIVERABLES)
    # The classifier may now choose a table over this evidence; it still controls selection.
    assert set(vendor.asked) == {dec.FILES_QUESTION_ID, dec.GRAPHICS_QUESTION_ID}
    assert decision.graphics is True


def _bash_tool(home: Path, calls: list[str]) -> AgentTool:
    """A ``bash`` tool whose EFFECT is the incident's writes (the command text is the real one)."""

    async def execute(  # type: ignore[no-untyped-def]
        tool_call_id, args, signal, on_update, context
    ):
        calls.append(args["command"])
        _write_deliverables(home)
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name="bash",
            content=[TextContent(text="exit code: 0\n--- stdout ---\n296\n")],
        )

    return AgentTool(
        name="bash",
        parameters={
            "type": "object",
            "properties": {"command": {"type": "string"}, "i": {"type": "string"}},
            "required": ["command"],
        },
        interruptible=True,
        execute=execute,
    )


def _incident_command() -> str:
    rows = _messages()
    last_call = [c for r in rows for c in r["tool_calls"] if c["name"] == "bash"][-1]
    return last_call["arguments"]["command"]


@pytest.mark.asyncio
async def test_a_turn_resumed_after_an_answered_ask_journals_the_four_files(
    home: Path, tmp_path: Path
) -> None:
    """End to end through the real Session: prompt -> ask answer -> bash writes -> row."""
    answer = _messages()[-1]["text"]
    calls: list[str] = []
    stream = ScriptedStream(
        [
            [StreamTextDelta(delta="Asked."), StreamEndEvent(stop_reason="stop")],
            [
                StreamToolCallDelta(
                    index=0,
                    id="b1",
                    name="bash",
                    argument_delta=json.dumps({"command": _incident_command(), "i": "Updating"}),
                ),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamTextDelta(delta=answer), StreamEndEvent(stop_reason="stop")],
        ]
    )
    session = _make_session(tmp_path / "sessions" / "ask", stream, [_bash_tool(home, calls)], home)
    handle = ServingSessionHandle(
        session, asyncio.get_running_loop(), install_gates=False, cwd=str(home)
    )
    handle.subscribe(lambda: None)
    try:
        await session.prompt("close the rationale gap")
        await wait_for(lambda: session._turns_settled == 1, timeout=10)
        assert session.transcript.latest_custom(SUPPLEMENT_CUSTOM_TYPE) is None  # nothing yet
        await session.deliver_ask_messages(
            [
                CustomMessage(
                    custom_type=ASK_RESPONSE_MESSAGE_TYPE,
                    attribution="user",
                    details={
                        "ask_id": "a-1",
                        "status": "answered",
                        "questions": [{"id": "gap_fix", "question": "Post them?"}],
                        "answers": {"gap_fix": ["Yes, with the supplementary header"]},
                    },
                )
            ]
        )
        row = await _await_row(session.transcript)
        assert len(calls) == 1
        assert row["state"] == "done"
        assert {f["path"] for f in row["files"]} == {
            f"workspace/{PACKAGE}/{n}" for n in DELIVERABLES
        }
        # The answer code-spans RUN-RECORD.md / SHA256SUMS / reversal-ids.json / the ledger by
        # name: every one is flagged, and (the point of the fix) every one is still offered.
        assert all(f.get("linked_in_answer") is True for f in row["files"])
        assert session.transcript.has_entry(row["anchor"])
    finally:
        await handle.dispose()
        await session.dispose()


# -- the graphics default flip: what still turns it off ------------------------------------------


def test_graphics_default_on_and_every_off_switch_still_works(monkeypatch) -> None:
    from local_operator.supplements import policy

    assert policy.DEFAULT_GRAPHICS is True
    assert policy.SupplementSettings().graphics is True
    assert policy.SupplementSettings.from_values({}).graphics is True
    assert policy.SupplementSettings.from_values({"supplements": {"unrelated": 1}}).graphics is True
    off = policy.SupplementSettings.from_values({"supplements": {"graphics": False}})
    assert off.graphics is False and off.active, "files stay on when only graphics is off"
    master = policy.SupplementSettings.from_values({"supplements": {"enabled": False}})
    assert master.enabled is False and not master.active
    both = policy.SupplementSettings.from_values(
        {"supplements": {"files": False, "graphics": False}}
    )
    assert not both.active
    # YAML-1.1 spellings a human types still opt out; a typo is not an opt-out.
    assert (
        policy.SupplementSettings.from_values({"supplements": {"graphics": "off"}}).graphics
        is False
    )
    assert (
        policy.SupplementSettings.from_values({"supplements": {"graphics": "nope"}}).graphics
        is True
    )


@pytest.mark.parametrize("value", ["0", "false", "no", "off", " OFF "])
def test_the_env_kill_switch_still_removes_the_feature_with_graphics_on(value: str) -> None:
    """``LOP_SUPPLEMENTS`` is read once at import, so check it in a fresh interpreter."""
    import subprocess
    import sys

    code = "from local_operator.supplements import policy; print(policy.enabled())"
    env = {
        "PATH": os.environ["PATH"],
        "HOME": os.environ.get("HOME", "/"),
        "LOP_SUPPLEMENTS": value,
    }
    run = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60
    )
    assert run.stdout.strip() == "False", run.stderr


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command,expected",
    [
        ("cd ~/workspace/a; cd ~/workspace/b; cd -; echo REAL > dash.md", set()),
        ("cd ~; cd ..; echo REAL > parent.md", set()),
        ('echo "see <<EOF notes"; echo REAL > proper.md', {"proper.md"}),
    ],
)
async def test_real_session_bash_never_promotes_stale_directory_decoys(
    tmp_path, monkeypatch, command, expected
):
    """R2/Q1: actual bash and Session+runner, not a hand-built refusal or fake write.

    The shell writes A/dash.md after cd -, never the old ~/dash.md. We cannot
    infer OLDPWD statically, so the parser must decline it. Parent-of-home is
    outside both allowed roots, not a session-relative output. Quoted << is text.
    """
    from local_operator.supplements import candidates

    home = tmp_path / "home"
    cwd = home / "workspace" / "y"
    for name in ("a", "b", "y"):
        (home / "workspace" / name).mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    # Synthetic home is under pytest's temporary tree, unlike a production home.
    monkeypatch.setattr(candidates, "_scratch_roots", lambda: [])
    (home / "dash.md").write_text("STALE HOME DECOY")
    (cwd / "parent.md").write_text("STALE CWD DECOY")
    executed = []

    async def execute(tool_call_id, args, signal, on_update, context):
        process = await asyncio.create_subprocess_exec(
            "/bin/bash",
            "-c",
            args["command"],
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        executed.append((process.returncode, stdout.decode(), stderr.decode()))
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name="bash",
            is_error=process.returncode != 0,
            content=[TextContent(text=f"exit code: {process.returncode}\\n" + stdout.decode())],
        )

    tool = AgentTool(
        name="bash",
        parameters={
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
        execute=execute,
    )
    stream = ScriptedStream(
        [
            [
                StreamToolCallDelta(
                    index=0,
                    id="real-shell",
                    name="bash",
                    argument_delta=json.dumps({"command": command}),
                ),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamTextDelta(delta="Finished."), StreamEndEvent(stop_reason="stop")],
        ]
    )
    session = _make_session(tmp_path / "sess", stream, [tool], cwd)
    handle = ServingSessionHandle(
        session, asyncio.get_running_loop(), install_gates=False, cwd=str(cwd)
    )
    handle.subscribe(lambda: None)
    try:
        await session.prompt("Write the deliverable.")
        await wait_for(lambda: session.turns_settled == 1, timeout=10)
        await wait_for(
            lambda: handle._supplements._task is not None and handle._supplements._task.done(),
            timeout=10,
        )
        assert executed[0][0] == 0, executed
        row = session.transcript.latest_custom(SUPPLEMENT_CUSTOM_TYPE)
        assert ({f["path"] for f in row["files"]} if row else set()) == expected
        assert (home / "dash.md").read_text() == "STALE HOME DECOY"
        assert (cwd / "parent.md").read_text() == "STALE CWD DECOY"
        if "cd -;" in command:
            assert (home / "workspace" / "a" / "dash.md").read_text() == "REAL\n"
        elif "cd ..;" in command:
            assert (tmp_path / "parent.md").read_text() == "REAL\n"
        else:
            assert (cwd / "proper.md").read_text() == "REAL\n"
    finally:
        await handle.dispose()
        await session.dispose()
