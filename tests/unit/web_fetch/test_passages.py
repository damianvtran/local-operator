"""Query-focused passage previews: the ranker and its ``run_fetch`` wiring.

Small synthetic pages only — the property under test is that an answer buried
in the MIDDLE of an oversized page reaches the first preview, with line labels
that resolve verbatim through ``read spill://<h> range=``.
"""

from __future__ import annotations

import json
import re
import time

import httpx
import pytest

from local_operator.harness.types import ToolContext
from local_operator.tools import spill
from local_operator.tools.builtin import TOOL_OUTPUT_LIMIT_CHARS, execute_read
from local_operator.web_fetch import passages, service
from local_operator.web_fetch.passages import chunk_lines, select_passages
from local_operator.web_fetch.tool import execute_web_fetch, run_fetch

NEEDLE = "The default value of `max_connections` is 100 connections."


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.delenv(spill.SPILL_MAX_BYTES_ENV, raising=False)
    monkeypatch.setattr(service, "_resolve_host_ips", lambda host: ["93.184.216.34"])
    monkeypatch.setattr(
        service, "DEFAULT_WEB_FETCH_CONFIG", {**service.DEFAULT_WEB_FETCH_CONFIG, "enrich": False}
    )


@pytest.fixture
def context(tmp_path) -> ToolContext:
    return ToolContext(cwd=str(tmp_path), session_id="passage-test")


def _long_markdown(sections: int = 120, needle_at: int = 60) -> str:
    """A markdown page whose only answer sits in a middle section."""
    out = ["# Server Configuration", "", "Reference for every server setting.", ""]
    for i in range(sections):
        out += [f"## Setting group {i}", ""]
        out.append(
            f"Group {i} settings control unrelated behaviour such as logging and "
            "locale; they are listed here for completeness and padding."
        )
        out.append("")
        if i == needle_at:
            out += ["### Connections", "", NEEDLE, ""]
    return "\n".join(out)


def _transport(text: str, ctype: str = "text/markdown") -> httpx.MockTransport:
    return httpx.MockTransport(
        lambda req: httpx.Response(200, text=text, headers={"content-type": ctype})
    )


# -- the ranker --------------------------------------------------------------


def test_ranker_finds_the_buried_identifier() -> None:
    lines = _long_markdown().splitlines()
    sel = select_passages(lines, "max_connections default", budget=5000, skip_through=4)
    assert sel is not None
    assert any(NEEDLE in p.text for p in sel.passages)
    # Document order, never ranking order: the preview must read like the page.
    starts = [p.start for p in sel.passages]
    assert starts == sorted(starts)
    assert all(p.start > 4 for p in sel.passages), "passages never repeat the lead"


def test_passage_text_is_exactly_its_line_range() -> None:
    lines = _long_markdown().splitlines()
    sel = select_passages(lines, "connections default", budget=5000)
    assert sel is not None
    for p in sel.passages:
        assert p.text == "\n".join(lines[p.start - 1 : p.end])


def test_ranker_is_deterministic_and_none_on_no_match() -> None:
    lines = _long_markdown().splitlines()
    a = select_passages(lines, "max_connections", budget=4000)
    b = select_passages(lines, "max_connections", budget=4000)
    assert a == b
    assert select_passages(lines, "zyzzyva quokka", budget=4000) is None
    # A query of nothing but stopwords carries no signal at all.
    assert select_passages(lines, "what is the", budget=4000) is None


def test_ranker_respects_the_budget() -> None:
    lines = _long_markdown().splitlines()
    sel = select_passages(lines, "group settings logging", budget=1500)
    assert sel is not None
    assert sum(len(p.text) for p in sel.passages) <= 1500
    assert sel.more > 0, "a common term matches more chunks than fit"


def test_tables_stay_whole() -> None:
    table = ["| name | default |", "| --- | --- |"] + [f"| opt_{i} | {i} |" for i in range(10)]
    lines = ["# T", "", "prose " * 80, ""] + table + ["", "after " * 80]
    chunks = chunk_lines(lines)
    table_start = lines.index(table[0]) + 1
    owning = [c for c in chunks if c.start <= table_start <= c.end]
    assert owning and owning[0].start == table_start
    assert owning[0].end == table_start + len(table) - 1


def test_chunking_is_linear_on_a_five_megabyte_page(monkeypatch: pytest.MonkeyPatch) -> None:
    """Structural first: one tokenisation per chunk, never per pair.

    ``_terms`` is called once per chunk (+ once per heading), so a quadratic
    regression — re-tokenising neighbours, scoring pairs — shows up as a
    call count, not as a flaky wall-clock number. The CPU bound is a
    catastrophe ceiling only (AGENTS.md "Timing, flakes").
    """
    para = "Lorem ipsum dolor sit amet, consectetur adipiscing elit sed do. " * 6
    lines: list[str] = []
    while sum(len(x) + 1 for x in lines) < 5 * 1024 * 1024:
        lines += ["## Heading", "", para, "", para, ""]
    lines[len(lines) // 2] = NEEDLE

    calls = {"n": 0}
    real = passages._terms

    def counting(text: str) -> list[str]:
        calls["n"] += 1
        return real(text)

    monkeypatch.setattr(passages, "_terms", counting)
    started = time.thread_time()
    sel = select_passages(lines, "max_connections default", budget=5000)
    cpu = time.thread_time() - started
    monkeypatch.undo()
    chunks = len(chunk_lines(lines))
    assert sel is not None and any(NEEDLE in p.text for p in sel.passages)
    assert calls["n"] <= 2 * chunks + 2
    assert cpu < 60, f"catastrophe ceiling only; took {cpu:.1f}s CPU"


# -- run_fetch wiring --------------------------------------------------------


@pytest.mark.asyncio
async def test_query_preview_surfaces_the_middle_and_labels_resolve(
    context: ToolContext,
) -> None:
    page = _long_markdown()
    assert len(page) > TOOL_OUTPUT_LIMIT_CHARS
    plain, _d, _e = await run_fetch(
        "https://example.com/a", tool_name="web_fetch", context=context, transport=_transport(page)
    )
    assert NEEDLE not in plain, "fixture must hide the answer from head/tail"

    preview, details, is_error = await run_fetch(
        "https://example.com/b",
        tool_name="web_fetch",
        context=context,
        transport=_transport(page),
        query="max_connections default",
    )
    assert is_error is False
    assert NEEDLE in preview
    assert len(preview) <= TOOL_OUTPUT_LIMIT_CHARS
    handle = details["spill"]["handle"]
    assert handle in preview and "?q=" in preview or details["query_more"] == 0
    # Every printed label must name a range whose read-back IS the text shown:
    # the stored lines for that range equal the passage verbatim, and the
    # model-facing ``read`` of that range carries every one of them.
    assert details["query_passages"]
    store = spill.get_store()
    for start, end in details["query_passages"]:
        label = f"[lines {start}-{end}"
        assert label in preview
        stored, _total = store.read_lines(handle, start, end)
        shown = "\n".join(stored)
        assert f"]\n{shown}\n" in preview
        back = await execute_read(
            "r", {"path": handle, "range": f"{start}-{end}"}, None, None, context
        )
        for line in stored:
            assert line in back.content[0].text


@pytest.mark.asyncio
async def test_query_without_match_falls_back_to_head_tail_and_says_so(
    context: ToolContext,
) -> None:
    page = _long_markdown()
    plain, _d, _e = await run_fetch(
        "https://example.com/c", tool_name="web_fetch", context=context, transport=_transport(page)
    )
    fallback, details, _e = await run_fetch(
        "https://example.com/c",
        tool_name="web_fetch",
        context=context,
        transport=_transport(page),
        query="zyzzyva quokka",
    )
    assert "No passage matched the query" in fallback
    header, body = plain.split("\n\n", 1)
    assert fallback.endswith(body), "fallback keeps the exact head/tail body"
    assert details["query_passages"] == []


@pytest.mark.asyncio
async def test_json_and_inline_pages_ignore_the_query(context: ToolContext) -> None:
    doc = json.dumps({"rows": [{"max_connections": i, "pad": "x" * 40} for i in range(400)]})
    plain, _d, _e = await run_fetch(
        "https://example.com/j.json",
        tool_name="web_fetch",
        context=context,
        transport=_transport(doc, "application/json"),
    )
    queried, d2, _e = await run_fetch(
        "https://example.com/j.json",
        tool_name="web_fetch",
        context=context,
        transport=_transport(doc, "application/json"),
        query="max_connections",
    )
    assert "query_passages" not in d2
    # Same content, so the second call is a cache hit; only the cache marker differs.
    assert queried == plain.replace("cache miss", "cache hit")

    small = "# Tiny\n\nmax_connections is 100.\n"
    inline_plain, _d, _e = await run_fetch(
        "https://example.com/s", tool_name="web_fetch", context=context, transport=_transport(small)
    )
    inline_q, d3, _e = await run_fetch(
        "https://example.com/s",
        tool_name="web_fetch",
        context=context,
        transport=_transport(small),
        query="max_connections",
        refresh=True,
    )
    assert inline_q == inline_plain
    assert "query_passages" not in d3


@pytest.mark.asyncio
async def test_query_reshapes_a_cache_hit_without_network(context: ToolContext) -> None:
    page = _long_markdown()
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, text=page, headers={"content-type": "text/markdown"})

    transport = httpx.MockTransport(handler)
    first, _d, _e = await run_fetch(
        "https://example.com/p", tool_name="web_fetch", context=context, transport=transport
    )
    second, d2, _e = await run_fetch(
        "https://example.com/p",
        tool_name="web_fetch",
        context=context,
        transport=transport,
        query="max_connections",
    )
    third, _d3, _e = await run_fetch(
        "https://example.com/p", tool_name="web_fetch", context=context, transport=transport
    )
    assert calls["n"] == 1, "the query must not enter the cache variant"
    assert d2["cache"] == "hit"
    assert NEEDLE in second and NEEDLE not in first
    assert third == first.replace("cache miss", "cache hit")


@pytest.mark.asyncio
async def test_tool_entry_point_threads_query(context: ToolContext) -> None:
    page = _long_markdown()
    await run_fetch(
        "https://example.com/t", tool_name="web_fetch", context=context, transport=_transport(page)
    )
    result = await execute_web_fetch(
        "c1", {"url": "https://example.com/t", "query": "max_connections"}, context=context
    )
    assert NEEDLE in result.content[0].text


def test_markdown_escaped_identifier_still_scores_as_identifier() -> None:
    """markdownify writes ``SQLITE\\_MAX\\_COLUMN``; the query names it plainly."""
    lines = ["# Limits", ""]
    for i in range(40):
        lines += [f"## Limit {i}", "", f"The default for limit {i} is configurable.", ""]
    # Decoys carry every query WORD, more often than the target does, but never
    # the identifier itself: only the identifier bonus can rank the target first,
    # and that bonus only fires if the escaped spelling is unescaped for scoring.
    for i in range(3):
        lines += [
            f"## Display {i}",
            "",
            "The default SQLite shell max column width; the default column is the max "
            "SQLite column by default.",
            "",
        ]
    # Its own section, so the target is not merged into a decoy's chunk.
    lines += ["## Columns", ""]
    target = len(lines) + 1
    lines += [r"The setting for SQLITE\_MAX\_COLUMN is 2000.", ""]
    # One passage only, so the assertion is about the TOP-ranked chunk.
    sel = select_passages(lines, "SQLITE_MAX_COLUMN default", budget=300, max_passages=1)
    assert sel is not None
    assert sel.passages[0].start <= target <= sel.passages[0].end
    assert "\\_" in sel.passages[0].text, "shown text stays the stored text"


@pytest.mark.asyncio
async def test_more_matches_hint_resolves_through_read(context: ToolContext) -> None:
    page = _long_markdown()
    preview, details, _e = await run_fetch(
        "https://example.com/h",
        tool_name="web_fetch",
        context=context,
        transport=_transport(page),
        query="group settings logging",
    )
    assert details["query_more"] > 0
    match = re.search(r'\[\d+ more passages? match — read\(path="([^"]+)"\)\]', preview)
    assert match, preview[-600:]
    found = await execute_read("r", {"path": match.group(1)}, None, None, context)
    assert found.is_error is False
    assert "Group" in found.content[0].text
