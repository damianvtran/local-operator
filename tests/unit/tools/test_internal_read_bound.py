"""Bounds on ``read``'s internal-URL and directory branches.

The defect these pin: ``execute_read``'s internal-URL branch returned the
resolver's content with no cap at all, while every other read outcome was
bounded at 8 KiB — and it is the branch that serves the LARGEST documents in
the system. One ``read skill://minerva-software-development`` injected 78,320
chars (~27k billed tokens), and because ``_is_prunable`` exempts ``skill://``
reads from pruning it stayed resident for the whole session.

The tests that matter most here are NOT the size assertions. They are the ones
that pin the anti-regression properties, because the cheap fix — truncate the
head and stop — is a silent behavioural regression: in that skill the binding
``## Mandatory Agent Review Gate`` sits at char offset 30,343 while the
frontmatter phrase "agent-review round" at offset 420 survives any head cap,
so a head-truncated result READS complete while every operative rule is gone.
An agent acting on it merges without a review round. Hence
``test_review_gate_heading_survives_shaping_and_its_range_resolves``, which is
modelled on the real document's structure rather than on a generic large blob.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator.compaction.pruning import _is_prunable
from local_operator.harness.types import (
    AgentTool,
    Message,
    TextContent,
    ToolContext,
    ToolResult,
)
from local_operator.tools import builtin, spill
from local_operator.tools.registry import create_tools


@pytest.fixture(autouse=True)
def isolated_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every spill write inside the test's tmp dir, never the real home."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.delenv(spill.SPILL_MAX_BYTES_ENV, raising=False)
    monkeypatch.delenv(builtin.INTERNAL_READ_LIMIT_ENV, raising=False)


def _context(tmp_path: Path, doc: str, url_prefix: str = "skill://") -> ToolContext:
    """A context whose resolver serves ``doc`` for any URL under ``url_prefix``."""
    return ToolContext(
        cwd=str(tmp_path),
        session_id="internal-read-test",
        resolve_internal_url=lambda url: doc if url.startswith(url_prefix) else None,
    )


async def _call(context: ToolContext, args: dict[str, Any]) -> ToolResult:
    tools: dict[str, AgentTool] = {tool.name: tool for tool in create_tools(context)}
    read = tools["read"]
    return await read.execute("call-1", args, None, None, context)  # type: ignore[operator]


def _text_of(result: ToolResult) -> str:
    return "".join(b.text for b in result.content if isinstance(b, TextContent))


# A fixture modelled on the real skill: the operative rule sits DEEP, far past
# any plausible head budget, with a decoy near the top that a head-only
# truncation would leave behind to make the result read complete.
_GATE_HEADING = "Mandatory Agent Review Gate"
_GATE_BODY = "Every MR opened by an agent MUST carry a completed agent-review round."


def _skill_like_document() -> str:
    parts = [
        "---",
        "name: fixture-skill",
        "description: mentions an agent-review round in the frontmatter decoy.",
        "---",
        "",
        "# Fixture Skill",
        "",
        "## Start Every Task",
        "",
    ]
    parts += [f"Routing line {i}." for i in range(40)]
    # Filler sections push the gate well past the 6 KiB head budget, the way
    # the real document's 30 KB offset does.
    for section in range(12):
        parts += ["", f"## Filler Section {section}", ""]
        parts += [f"Filler body line {i} of section {section}." for i in range(40)]
    parts += ["", f"## {_GATE_HEADING}", "", _GATE_BODY, ""]
    parts += [f"Gate detail line {i}." for i in range(20)]
    parts += ["", "### Comment format", "", "Round comments look like this.", ""]
    parts += ["", "## Cross-Skill Routing", "", "Route elsewhere when needed.", ""]
    return "\n".join(parts)


@pytest.mark.asyncio
async def test_review_gate_heading_survives_shaping_and_its_range_resolves(
    tmp_path: Path,
) -> None:
    """THE regression test for this defect, not a generic size test.

    Two assertions, and both are needed: the deep operative heading must be
    VISIBLE in the outline (so the agent knows the rule exists), and the line
    range printed beside it must actually RESOLVE through the spill store (so
    knowing is actionable). A heading listed with a range that does not resolve
    is worse than no outline: it tells the agent expansion works when it does
    not.
    """
    doc = _skill_like_document()
    assert (
        doc.index(_GATE_BODY) > builtin.INTERNAL_READ_HEAD_CHARS
    ), "fixture must place the gate past the head budget, or it proves nothing"
    context = _context(tmp_path, doc)

    result = await _call(context, {"path": "skill://fixture-skill"})
    shaped = _text_of(result)

    assert _GATE_HEADING in shaped, "the review-gate heading vanished from the outline"
    assert _GATE_BODY not in shaped, "fixture is too small to have been shaped"

    # Pull the range the outline printed beside the gate heading and run it.
    entry = next(line for line in shaped.splitlines() if _GATE_HEADING in line)
    span = entry.rsplit("[lines ", 1)[1].rstrip("]")
    handle = result.details["spill"]["handle"]  # type: ignore[index]

    expanded = _text_of(await _call(context, {"path": handle, "range": span}))
    assert _GATE_BODY in expanded, f"range {span} did not resolve to the gate section"


@pytest.mark.asyncio
async def test_shaped_result_keeps_the_skill_url_so_pruning_stays_exempt(
    tmp_path: Path,
) -> None:
    """``_is_prunable`` keys the skill exemption off
    ``details['url'].startswith('skill://')``. An implementation that
    helpfully rewrote ``url`` to the spill handle would silently disable that
    exemption — and post-shaping the exempted result is what carries the handle
    the agent needs, so blanking it strands the expansion. Pinned here because
    the failure is invisible: nothing errors, sessions just start losing skills.
    """
    context = _context(tmp_path, _skill_like_document())
    result = await _call(context, {"path": "skill://fixture-skill"})

    assert result.details is not None
    assert result.details["url"] == "skill://fixture-skill"
    assert result.details["spill"]["handle"].startswith("spill://")

    message = Message(
        role="tool",
        content=[TextContent(text=_text_of(result))],
        tool_name="read",
        tool_call_id="call-1",
        provider_payload={"details": result.details},
    )
    assert _is_prunable(message) is False


@pytest.mark.asyncio
async def test_shaped_result_announces_partiality_at_the_top(tmp_path: Path) -> None:
    """The banner leads the result, not just trails it: the model reads
    top-down and may act on the head before ever reaching a footer."""
    context = _context(tmp_path, _skill_like_document())
    shaped = _text_of(await _call(context, {"path": "skill://fixture-skill"}))

    first = shaped.splitlines()[0]
    assert "PARTIAL" in first
    assert "NOT read the whole document" in first
    assert "skill://fixture-skill" in first


@pytest.mark.asyncio
async def test_footer_names_an_expansion_call_that_actually_resolves(
    tmp_path: Path,
) -> None:
    """Same discipline as ``_spill_footer``: run the printed call verbatim and
    require content back. A footer that describes expansion instead of spelling
    out a working call teaches the model that expansion does not work."""
    context = _context(tmp_path, _skill_like_document())
    result = await _call(context, {"path": "skill://fixture-skill"})
    shaped = _text_of(result)

    printed = next(line for line in shaped.splitlines() if 'range="' in line)
    handle = printed.split('read(path="', 1)[1].split('"', 1)[0]
    span = printed.split('range="', 1)[1].split('"', 1)[0]

    expanded = await _call(context, {"path": handle, "range": span})
    assert not expanded.is_error
    assert "beyond the end of" not in _text_of(expanded)


@pytest.mark.asyncio
async def test_documents_under_the_threshold_are_returned_byte_identical(
    tmp_path: Path,
) -> None:
    """Most guides fit. The MUST-read guide rule stays untouched for them, and
    a shaped result for a document that fits would be a pure regression."""
    doc = "# Small Guide\n\n" + "\n".join(f"line {i}" for i in range(200))
    assert len(doc) < builtin.INTERNAL_READ_LIMIT_CHARS
    context = _context(tmp_path, doc, url_prefix="guide://")

    result = await _call(context, {"path": "guide://small"})
    assert _text_of(result) == doc
    assert "spill" not in (result.details or {})


@pytest.mark.asyncio
async def test_heading_poor_documents_fall_back_to_plain_head_and_tail(
    tmp_path: Path,
) -> None:
    """Below the heading minimum there is no structure to index, so the
    document takes the same uniform ``spill_truncate`` every other oversized
    output gets rather than an outline of two entries."""
    doc = "# Only Heading\n\n" + "\n".join(f"prose line {i}" for i in range(4000))
    assert len(doc) > builtin.INTERNAL_READ_LIMIT_CHARS
    context = _context(tmp_path, doc, url_prefix="guide://")

    shaped = _text_of(await _call(context, {"path": "guide://flat"}))
    assert builtin.BASH_TRUNCATION_MARKER.strip() in shaped
    assert "is SAVED at spill://" in shaped
    assert len(shaped) <= builtin.INTERNAL_READ_LIMIT_CHARS + 1024


@pytest.mark.asyncio
async def test_the_outline_covers_every_line_past_the_head(tmp_path: Path) -> None:
    """No line may fall in a gap between the head and the outline's spans.

    This is completeness stated as an invariant rather than as a spot check on
    one heading: whatever the head cut, the outline's first span must resume at
    the very next line and its last must reach EOF. The awkward case is a
    document whose headings all cluster at the top followed by bulk prose — if
    the outline dropped the heading the head stopped at, those 4,000 lines
    would be addressable by nothing and would vanish unannounced.
    """
    doc = "\n".join(
        ["# One", "text", "## Two", "text", "## Three", "text", ""]
        + [f"bulk prose line {i}" for i in range(4000)]
    )
    assert len(doc) > builtin.INTERNAL_READ_LIMIT_CHARS
    context = _context(tmp_path, doc, url_prefix="guide://")

    result = await _call(context, {"path": "guide://clustered"})
    shaped = _text_of(result)

    spans = [
        (
            int(line.rsplit("[lines ", 1)[1].rstrip("]").split("-")[0]),
            int(line.rsplit("[lines ", 1)[1].rstrip("]").split("-")[1]),
        )
        for line in shaped.splitlines()
        if "  [lines " in line
    ]
    assert spans, "the outline indexed nothing"

    head_lines = len(shaped.split("\n\n[... ", 1)[0].splitlines()) - 2  # minus banner+blank
    assert spans[0][0] <= head_lines + 1, "a gap sits between the head and the first span"
    assert spans[-1][1] == len(doc.splitlines()), "the outline stops short of EOF"

    # And the bulk prose really is reachable through the last span.
    handle = result.details["spill"]["handle"]  # type: ignore[index]
    tail = _text_of(await _call(context, {"path": handle, "range": f"{spans[-1][1] - 5}-"}))
    assert "bulk prose line 3999" in tail


@pytest.mark.asyncio
async def test_kill_switch_restores_unbounded_passthrough(tmp_path: Path) -> None:
    """Rollout escape hatch: if shaping degrades behaviour in practice the
    operator turns it off in one variable rather than waiting for a release."""
    doc = _skill_like_document()
    context = _context(tmp_path, doc)

    import os

    os.environ[builtin.INTERNAL_READ_LIMIT_ENV] = "0"
    try:
        assert _text_of(await _call(context, {"path": "skill://fixture-skill"})) == doc
    finally:
        del os.environ[builtin.INTERNAL_READ_LIMIT_ENV]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "guide_md",
    # Keyed on the GUIDE.md file, not on "is a directory": the package ships a
    # __pycache__ beside the guides once it has been imported, and collecting
    # that would fail at read time rather than assert anything.
    sorted((Path(builtin.__file__).parents[1] / "guides").glob("*/GUIDE.md")),
    ids=lambda p: p.parent.name,
)
@pytest.mark.asyncio
async def test_every_bundled_guide_read_is_bounded(tmp_path: Path, guide_md: Path) -> None:
    """Drives the real bundled guides, so this guard cannot rot: the guides
    ship in the repo, and a future 80 KB guide fails here rather than in a
    session. This is the test that makes the original defect unrepeatable."""
    name = guide_md.parent.name
    context = _context(tmp_path, guide_md.read_text(), url_prefix="guide://")

    shaped = _text_of(await _call(context, {"path": f"guide://{name}"}))
    ceiling = builtin.INTERNAL_READ_LIMIT_CHARS + builtin.INTERNAL_READ_SHAPE_SLACK_CHARS
    assert len(shaped) <= ceiling, f"guide://{name} returned {len(shaped)} chars"


@pytest.mark.asyncio
async def test_no_read_branch_returns_an_unbounded_result(tmp_path: Path) -> None:
    """The test that would have caught the original hole, across all four
    branches at once — including the directory listing, which had the same
    defect and is fixed in the same change."""
    ceiling = builtin.INTERNAL_READ_LIMIT_CHARS + builtin.INTERNAL_READ_SHAPE_SLACK_CHARS

    big = "\n".join(f"filler line {i}" for i in range(6000))
    (tmp_path / "big.txt").write_text(big)
    wide = tmp_path / "wide"
    wide.mkdir()
    for i in range(4000):
        (wide / f"entry-with-a-longish-name-{i:05d}.txt").touch()

    context = _context(tmp_path, _skill_like_document())

    file_result = await _call(context, {"path": str(tmp_path / "big.txt")})
    dir_result = await _call(context, {"path": str(wide)})
    url_result = await _call(context, {"path": "skill://fixture-skill"})
    page_result = await _call(
        context, {"path": url_result.details["spill"]["handle"]}  # type: ignore[index]
    )

    for name, result in (
        ("file", file_result),
        ("directory", dir_result),
        ("internal-url", url_result),
        ("spill-page", page_result),
    ):
        assert len(_text_of(result)) <= ceiling, f"{name} branch returned an unbounded result"

    # The directory listing must stay expandable, not merely clipped.
    assert "spill://" in _text_of(dir_result)
