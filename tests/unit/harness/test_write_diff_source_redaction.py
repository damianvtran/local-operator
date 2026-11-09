"""write/edit diffs are published only after their COMPLETE sources were judged.

Two reproduced disclosures on dfc7411 (merged #3695), both with the production
``VariableStore`` redactor and a SYNTHETIC registered multi-line literal:

* a 10-line literal whose middle line is edited: the hunk's two context lines
  carry 5 of its 10 lines, the literal never appears whole in the diff, and the
  per-hunk seam published all 5 (architect proof, frag_spike.log);
* a 3-line literal at rows 196-198 of a 600-line write: the 200-row preview cut
  kept its first line verbatim in the transcript and live event.

These drive the REAL Session with the real write/edit tools and the session's
own redactor, and inspect every live end event and the persisted transcript.
"""

from __future__ import annotations

import copy
import json
import pickle
from pathlib import Path
from typing import Any, cast

import pytest

from local_operator.harness.types import (
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    ToolContext,
    ToolExecutionEndEvent,
    ToolResult,
)
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.tools import builtin, diff_limits
from local_operator.tools.builtin import (
    DIFF_TRUNCATED_ROWS_KEY,
    build_edit_tool,
    build_write_tool,
)
from local_operator.variables import VariableStore

pytestmark = pytest.mark.asyncio

WITHHELD = diff_limits.DIFF_WITHHELD_TEXT


def _literal(height: int, tag: str) -> list[str]:
    """A synthetic registered literal (not a credential shape), as its lines."""
    return [f"SYNTH-{tag}-{n:02d}-QKV{n * 7919:06d}" for n in range(height)]


def _provider(calls: list[tuple[str, str, dict[str, Any]]]):
    requests: list[Any] = []

    def provider(request, signal=None):
        index = len(requests)
        requests.append(request)

        async def generate():
            if index < len(calls):
                call_id, name, args = calls[index]
                yield StreamToolCallDelta(
                    index=0, id=call_id, name=name, argument_delta=json.dumps(args)
                )
                yield StreamEndEvent(stop_reason="toolUse")
            else:
                yield StreamTextDelta(delta="done")
                yield StreamEndEvent(stop_reason="stop")

        return generate()

    return provider


async def _run(tmp_path: Path, calls, *, secrets: tuple[str, ...] = ()):
    variables = VariableStore(cwd=str(tmp_path))
    for value in secrets:
        assert variables.register_redaction(value)
    session = Session(
        model=ModelSpec(provider="test", model_id="synthetic", context_window=100000),
        stream_fn=_provider(calls),
        tools=[build_write_tool(), build_edit_tool()],
        transcript=Transcript(tmp_path / "sessions" / "srcredact0001"),
        system_blocks_provider=lambda *_: [],
        yolo=True,
        cwd=str(tmp_path),
        variables=variables,
    )
    events: list[Any] = []
    session.subscribe(events.append)
    await session.prompt("go")
    return session, events


def _published(session: Session, events: list[Any]) -> str:
    """Every live end event plus every persisted TOOL-RESULT row, verbatim.

    The assistant row's tool-call ARGUMENTS are a different channel (the
    model's own input, scrubbed by ``_scrub_history_arguments``); an ``edit``
    whose ``old_text`` is one line of the literal carries that line there, and
    this change does not touch it.
    """
    live = [e.model_dump_json() for e in events if isinstance(e, ToolExecutionEndEvent)]
    assert live, "no live end event captured"
    rows = [
        line
        for line in session.transcript.path.read_text().splitlines()
        if json.loads(line).get("payload", {}).get("role") == "tool"
    ]
    assert rows, "no persisted tool row"
    return "\n".join(rows + live)


def _tool_details(session: Session) -> list[dict[str, Any]]:
    return [
        cast(dict[str, Any], m.provider_payload)["details"]
        for m in session.transcript.build_llm_history()
        if m.role == "tool"
    ]


def _assert_no_fragment(blob: str, parts: list[str]) -> None:
    leaked = [index for index, part in enumerate(parts) if part in blob]
    assert leaked == [], f"literal lines published: {leaked}"


@pytest.fixture(autouse=True)
def _config(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))


@pytest.mark.parametrize("edited", [0, 5, 9])
async def test_edit_inside_tall_literal_publishes_no_fragment(tmp_path, edited):
    """Architect case A, plus the literal's first and last line edited."""
    parts = _literal(10, "A")
    pad = [f"filler line {index}" for index in range(40)]
    target = tmp_path / "cfg.txt"
    original = "\n".join(pad + parts + pad) + "\n"
    target.write_text(original)
    session, events = await _run(
        tmp_path,
        [("a-1", "edit", {"path": str(target), "old_text": parts[edited], "new_text": "EDITED"})],
        secrets=("\n".join(parts),),
    )
    assert target.read_text() == original.replace(parts[edited], "EDITED")
    _assert_no_fragment(_published(session, events), parts)
    (details,) = _tool_details(session)
    diff = details["diff"]
    assert diff[0] != WITHHELD, "masked, not withheld"
    assert "+EDITED" in diff and details["added"] == 1 and details["removed"] == 1
    assert [row[:1] for row in diff[3:]].count("+") == 1
    assert [row[:1] for row in diff[3:]].count("-") == 1


@pytest.mark.parametrize("tool", ["write", "edit"])
@pytest.mark.parametrize("at", [196, 396])
async def test_literal_at_preview_cut_publishes_no_fragment(tmp_path, tool, at):
    parts = _literal(3, "C")
    base = [f"line_{index}" for index in range(600)]
    lines = list(base)
    lines[at : at + 3] = parts
    target = tmp_path / "s.txt"
    if tool == "edit":
        # Every line changes, so the real edit transaction's diff is cut.
        lines = [f"{line} edited" for line in base]
        lines[at : at + 3] = parts
        target.write_text("\n".join(base) + "\n")
        args = {"path": str(target), "old_text": "\n".join(base), "new_text": "\n".join(lines)}
    else:
        args = {"path": str(target), "content": "\n".join(lines) + "\n"}
    session, events = await _run(tmp_path, [("c-1", tool, args)], secrets=("\n".join(parts),))
    assert target.read_text() == "\n".join(lines) + "\n"
    _assert_no_fragment(_published(session, events), parts)
    (details,) = _tool_details(session)
    assert details["diff"][0] != WITHHELD and details[DIFF_TRUNCATED_ROWS_KEY] > 0


async def test_duplicate_filler_around_literal_over_masks_never_under(tmp_path):
    parts = _literal(6, "D")
    pad = ["}", "", "}", ""] * 5
    target = tmp_path / "dup.txt"
    original = "\n".join(pad + parts + pad) + "\n"
    target.write_text(original)
    session, events = await _run(
        tmp_path,
        [("d-1", "edit", {"path": str(target), "old_text": parts[3], "new_text": "X"})],
        secrets=("\n".join(parts),),
    )
    _assert_no_fragment(_published(session, events), parts)


async def test_credential_free_diff_is_byte_identical_to_the_producer(tmp_path):
    target = tmp_path / "plain.py"
    before = "".join(f"v{index} = {index}\n" for index in range(300))
    target.write_text(before)
    after = before.replace("v150 = 150", "v150 = -150")
    session, _ = await _run(
        tmp_path,
        [
            (
                "p-1",
                "edit",
                {"path": str(target), "old_text": "v150 = 150", "new_text": "v150 = -150"},
            )
        ],
    )
    (details,) = _tool_details(session)
    assert details["diff"] == builtin._diff_details(str(target), before, after)["diff"]
    assert "_diff_source" not in session.transcript.path.read_text()


async def test_source_over_bound_is_withheld_with_exact_counters(tmp_path, monkeypatch):
    monkeypatch.setattr(diff_limits, "DIFF_SOURCE_MAX_BYTES", 1024)
    target = tmp_path / "big.txt"
    content = "".join(f"row {index}\n" for index in range(400))
    session, events = await _run(
        tmp_path, [("o-1", "write", {"path": str(target), "content": content})]
    )
    (details,) = _tool_details(session)
    assert details["diff"] == [WITHHELD]
    assert details["added"] == 400 and details["removed"] == 0
    ends = [e for e in events if isinstance(e, ToolExecutionEndEvent)]
    live = ends[0].result.details or {}
    assert live["diff"] == [WITHHELD] and live["added"] == 400
    assert "row 17" not in _published(session, events)


async def test_result_that_never_reaches_the_seam_publishes_the_receipt(tmp_path):
    """A direct ``execute`` caller (or a copy) sees the receipt, never raw rows."""
    target = tmp_path / "direct.txt"
    tool = build_write_tool()
    result = await tool.execute(
        "x-1",
        {"path": str(target), "content": "RAW-ROW\n"},
        None,
        None,
        ToolContext(cwd=str(tmp_path)),
    )
    assert (result.details or {})["diff"] == [WITHHELD]
    assert result._diff_source is not None
    for clone in (
        result.model_copy(),
        result.model_copy(update={"details": {}}),
        result.model_copy(deep=True),
        copy.copy(result),
        copy.deepcopy(result),
        pickle.loads(pickle.dumps(result)),
    ):
        assert clone._diff_source is None
    event = ToolExecutionEndEvent(tool_call_id="x-1", tool_name="write", result=result)
    assert "RAW-ROW" not in event.model_dump_json()
    assert "RAW-ROW" not in result.model_dump_json()
    assert result.take_diff_source() is not None and result._diff_source is None


def test_private_carrier_default_is_empty():
    result = ToolResult(tool_call_id="c", tool_name="write", content=[], details={})
    assert result._diff_source is None and result.take_diff_source() is None
