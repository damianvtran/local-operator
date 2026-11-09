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


# R1-4 (PR #3702): a coroutine, so the module-wide asyncio mark applies to it
# instead of warning that a sync test carries it.
async def test_private_carrier_default_is_empty():
    result = ToolResult(tool_call_id="c", tool_name="write", content=[], details={})
    assert result._diff_source is None and result.take_diff_source() is None


def _live_and_persisted(session: Session, events: list[Any]) -> tuple[Any, Any]:
    (end,) = [e for e in events if isinstance(e, ToolExecutionEndEvent)]
    (details,) = _tool_details(session)
    return end, details


@pytest.mark.parametrize("tool", ["write", "edit"])
@pytest.mark.parametrize("copier", ["guard_warn", "hook_note"])
async def test_copying_paths_keep_the_settled_credential_free_diff(
    tmp_path, monkeypatch, tool, copier
):
    """R1-1 / Q2: a WARN guardrail and a PostToolUse note both ``model_copy``
    the result; settlement must already have happened on the original, so the
    ordinary diff is published (not the withheld receipt) live and persisted.
    """
    token = "QAWARNTOKEN"
    if copier == "guard_warn":
        rules = tmp_path / "config" / "rules"
        rules.mkdir(parents=True)
        (rules / "synthetic-warn.md").write_text(
            "---\nname: synthetic-warn\nscope: tool:"
            + tool
            + "\naction: warn\ncondition: ["
            + token
            + "]\n---\nSynthetic warning.\n"
        )
    else:

        async def note(self, *_args, **_kwargs):
            return ["formatter ran"]

        monkeypatch.setattr(Session, "_run_post_tool_hooks", note)
    target = tmp_path / "plain.py"
    before = "".join(f"v{index} = {index}\n" for index in range(30))
    if tool == "edit":
        target.write_text(before)
        after = before.replace("v15 = 15", f"v15 = 15  # {token}")
        args = {"path": str(target), "old_text": "v15 = 15", "new_text": f"v15 = 15  # {token}"}
    else:
        before, after = "", f"hello\n{token} world\n"
        args = {"path": str(target), "content": after}
    session, events = await _run(tmp_path, [("w-1", tool, args)])
    assert target.read_bytes() == after.encode()
    end, details = _live_and_persisted(session, events)
    assert not end.is_error
    marker = "Synthetic warning" if copier == "guard_warn" else "formatter ran"
    assert marker in end.result.text, "the copying path did not run"
    expected = builtin._diff_details(str(target), before, after)["diff"]
    assert details["diff"] == expected
    assert (end.result.details or {})["diff"] == expected


_SEPARATORS = ["\r\n", "\r", "\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029"]


@pytest.mark.parametrize("sep", _SEPARATORS, ids=[repr(s) for s in _SEPARATORS])
@pytest.mark.parametrize("edited", [0, 5, 9])
async def test_literal_registered_with_the_files_own_separator(tmp_path, sep, edited):
    """R1-2 / Q1: the source pass normalized every break to ``"\n"`` before
    redaction, so a literal registered with the file's own separator never
    matched and its fragments published. Every ``splitlines`` separator."""
    parts = _literal(10, "S")
    pad = [f"filler line {index}" for index in range(40)]
    target = tmp_path / "sep.txt"
    original = sep.join(pad + parts + pad) + sep
    target.write_bytes(original.encode())
    session, events = await _run(
        tmp_path,
        [("s-1", "edit", {"path": str(target), "old_text": parts[edited], "new_text": "EDITED"})],
        secrets=(sep.join(parts),),
    )
    assert target.read_bytes() == original.replace(parts[edited], "EDITED").encode()
    end, details = _live_and_persisted(session, events)
    assert not end.is_error
    _assert_no_fragment(_published(session, events), parts)
    assert details["diff"][0] != WITHHELD, "masked, not withheld"
    assert (end.result.details or {})["diff"] == details["diff"]
    assert details["added"] == 1 and details["removed"] == 1


async def test_crlf_literal_among_duplicate_fillers_over_masks(tmp_path):
    """R1-2: the exact-source pass inherits the occurrence anchoring, so a
    duplicate filler row can never anchor a masked CRLF literal row back out."""
    parts = _literal(6, "E")
    pad = ["}", "", "}", ""] * 5
    target = tmp_path / "dupcrlf.txt"
    original = "\r\n".join(pad + parts + pad) + "\r\n"
    target.write_bytes(original.encode())
    session, events = await _run(
        tmp_path,
        [("e-1", "edit", {"path": str(target), "old_text": parts[3], "new_text": "X"})],
        secrets=("\r\n".join(parts),),
    )
    assert target.read_bytes() == original.replace(parts[3], "X").encode()
    _assert_no_fragment(_published(session, events), parts)


async def test_settle_uses_explicit_call_records_after_they_were_popped(tmp_path):
    """R1-3: the eval bridge pops ``original_call_args`` and the exemption
    verdict before its settle call; explicit values must reach the redactor's
    call identity and exemption scope instead of the emptied records."""
    from local_operator.harness.guard_area import source_is_exempt
    from local_operator.harness.loop import AgentLoop, LoopContext
    from local_operator.harness.redaction import current_tool_source

    target = tmp_path / "meta.txt"
    result = await build_write_tool().execute(
        "m-1",
        {"path": str(target), "content": "META-ROW\n"},
        None,
        None,
        ToolContext(cwd=str(tmp_path)),
    )
    seen: list[tuple[tuple[str, str], bool]] = []

    def redact(text: str) -> str:
        seen.append((current_tool_source(), source_is_exempt()))
        return text

    await AgentLoop._settle_write_diff(
        cast(Any, None),
        result,
        LoopContext(),
        redact,
        arguments={"path": str(target)},
        verdict=True,
    )
    assert seen, "redactor never ran"
    assert all(name == "write" and "meta.txt" in summary for (name, summary), _ in seen)
    assert all(exempt for _, exempt in seen)
    assert (result.details or {})["diff"] != [WITHHELD]
