"""Compaction memory follows committed history, not speculative model calls."""

import asyncio
import json
import sqlite3
import threading
import time
import types
from itertools import count
from typing import Any

import pytest

from local_operator.harness.types import Message, TextContent
from local_operator.memory import MemoryError, MemoryIdentity, MemoryStore, guard
from local_operator.memory_capture import (
    ATTRIBUTION_STARTED_KEY,
    LOCATORS_DEFAULT_ENABLED,
    capture_detail,
    extract,
    finish,
    hint,
    persist,
    policy,
    snapshot,
    watermark,
)


async def _await_capture_row(memory: MemoryStore, marker: str, *, timeout: float = 10.0) -> Any:
    """The marker's capture row, waiting for a worker that has not landed yet.

    THE FAILURE THIS REPLACES was a bare ``assert all(watermark(...))``: a False
    with no marker, no status and no reason (run 39383034418, job 118496365150,
    ``[context-full]`` at :1162). A capture row is written by
    ``complete_memory_capture`` after it awaits the scan task, so the honest wait
    is for the ROW, not for a snapshot of ``_background_tasks`` — a completion
    task that is spawned or rescheduled after the snapshot is invisible to it.

    Returns ``capture_detail`` (``status``/``n``/``reason``/``refused``/
    ``skipped``) or ``None``. The caller asserts on the value, so a timeout is a
    named reading rather than a lost assertion.
    """
    deadline = time.monotonic() + timeout
    while True:
        detail = capture_detail(memory, marker)
        if detail is not None or time.monotonic() >= deadline:
            return detail
        await asyncio.sleep(0.05)


NOISY_LOCATORS = (
    "Ratio 9/1, see guide://credentials and skill://software-development. "
    "The branch that we discussed is ready."
)
USEFUL_LOCATORS = (
    "Evidence saved at ~/tools/widget/docs/proof.md; contact user@example.invalid; "
    "ticket https://example.invalid/task/25; branch: fix/memory-capture."
)
USEFUL_BODIES = {
    "~/tools/widget/docs/proof.md",
    "user@example.invalid",
    "https://example.invalid/task/25",
    "branch: fix/memory-capture",
}

#: The one knowledge fact the locator-grammar fixtures hang their context on.
CONTEXT_FACT = "The evidence is recorded."


def _decision_section(body: str) -> str:
    """A decision heading with one fact and `body` as fenced DATA beneath it."""
    return f"## Key Decisions\n- {CONTEXT_FACT}\n{body}\n"


# --- Knowledge-first admission --------------------------------------------
#
# Measured on the operator's store (18,571 session facts): 13,175 (70.9%) were
# locator debris — URLs, paths, hex ids and bare numbers — because the scan
# minted every locator it found as its own fact. Only 8 lessons and 3 decisions
# existed. These tests pin the inversion: headings/decisions are the primary
# source, and a locator is admitted only as context of a real memory, never on
# its own. The default is OFF and the setting can restore the context arm
# without a code change.


def test_a_transcript_that_is_only_locators_yields_no_facts():
    transcript = " ".join(
        (
            USEFUL_LOCATORS,
            "/src/task.py",
            "https://example.invalid/task/25",
            "thread: 1700000000.123456",
            "~/.aws/credentials",
            "guide://memory/docs/example",
        )
    )
    assert extract(transcript) == []
    # Even with the context arm enabled, adjacency is the admission rule: a
    # locator with no real memory beside it is still refused.
    assert extract(transcript, locators=True) == []


def test_a_decision_heading_keeps_a_path_only_as_its_context():
    with_path_in_the_decision = extract(
        "## Key Decisions\n- Store evidence at ~/tools/widget/docs/proof.md\n"
    )
    assert [row["tags"] for row in with_path_in_the_decision] == [["decision"]]
    assert "~/tools/widget/docs/proof.md" in with_path_in_the_decision[0]["body"]
    # Never a second, standalone locator fact beside the decision.
    assert len(with_path_in_the_decision) == 1


def test_a_locator_on_its_own_line_is_context_of_the_section_fact():
    text = (
        "## Key Decisions\n"
        "- Store evidence in the widget vault.\n"
        "~/tools/widget/docs/proof.md\n"
    )
    assert [row["body"] for row in extract(text)] == ["Store evidence in the widget vault."]
    contextual = extract(text, locators=True)
    assert len(contextual) == 1 and contextual[0]["tags"] == ["decision"]
    assert contextual[0]["body"].startswith("Store evidence in the widget vault.")
    assert "~/tools/widget/docs/proof.md" in contextual[0]["body"]


def test_the_locator_setting_defaults_off_and_reads_the_config(tmp_path):
    assert LOCATORS_DEFAULT_ENABLED is False
    assert policy(tmp_path)["locators"] is False
    (tmp_path / "config.yml").write_text(
        "values:\n  memory:\n    capture_locators: true\n", encoding="utf-8"
    )
    assert policy(tmp_path)["locators"] is True


def test_the_compaction_prompt_template_is_never_captured():
    """The prompt's own headings and descriptions are scaffolding, not memories.

    Measured on session 439818272d84: parsing the logged compaction prompt
    produced four template "facts" ("What the user wants...", "Decisions made
    during...", "Anything that does not fit above...") plus a `{{#if
    previous_summary}}` row. The template repeats the knowledge headings, so it
    is exactly the shape the parser must not trust as conversation.
    """
    prompt = (
        "You are summarizing the conversation so far so the agent can continue.\n\n"
        "## Goal\n\nWhat the user wants, in one or two sentences, in the user's terms.\n\n"
        "## Key Decisions\n\nDecisions made during the conversation and why each "
        "was chosen, especially where alternatives were rejected.\n\n"
        "## Critical Context\n\nAnything that does not fit above.\n\n"
        "{{#if previous_summary}}\n{{/if}}\n"
    )
    assert extract(prompt) == []
    messages = [Message(role="user", content=[TextContent(text=prompt)])]
    assert snapshot(messages).strip() == ""


def test_a_candidate_whose_fact_dump_is_refused_is_skipped_not_crashed(store):
    """ADMISSION matches ``validate_op``'s re-dump, not only the body guard.

    A real carry line containing the literal ``secret: <value>`` passed
    ``guard(body)`` and ``validate_fact``, then raised from ``validate_op`` when
    the fact's full JSON was re-dumped — taking the whole pass down with zero
    rows. The candidate is skipped and counted instead.
    """
    body = (
        "- The mask is **label-dependent**. `key: <value>`, `token: <value>`, "
        "`secret: <value>` and `key=<value>` are masked; `name: <value>`, "
        "`id: <value>` and the bare value print unchanged."
    )
    rows = extract("## Key Decisions\n- " + body)
    assert list(rows) == [] and rows.skipped == 1
    assert persist(store, "dump-shape", rows, "ok") == ("declined", "admission-skip")


def test_a_carry_line_whose_fact_dump_is_refused_does_not_take_the_pass(store):
    from local_operator.memory_capture import carry_rows

    carry = (
        "- Asks: The mask is **label-dependent**. `key: <value>`, `token: <value>`, "
        "`secret: <value>` and `key=<value>` are masked; `name: <value>`, "
        "`id: <value>` and the bare value print unchanged.\n"
        "- Asks: keep the launch plan\n"
    )
    rows = carry_rows(carry)
    assert [row["body"] for row in rows] == ["keep the launch plan"]


def test_carry_identifiers_are_context_only_and_never_standalone():
    from local_operator.memory_capture import carry_rows

    assert carry_rows("- Identifiers: /src/only.py\n", locators=True) == []
    assert carry_rows("- Identifiers: /src/only.py\n") == []
    with_memory = "- Asks: keep the plan\n- Identifiers: /src/detail.py\n"
    assert [row["body"] for row in carry_rows(with_memory)] == ["keep the plan"]
    rows = carry_rows(with_memory, locators=True)
    assert len(rows) == 1 and rows[0]["tags"] == ["ask"]
    assert rows[0]["body"].startswith("keep the plan")
    assert "/src/detail.py" in rows[0]["body"]


@pytest.mark.parametrize(
    "heading,body,tag",
    [
        ("Key Decisions", "采用标题解析，而不是定位符扫描。", "decision"),
        ("Progress", "已经运行了定向测试。", "progress"),
        ("Key Decisions", "Используем заголовки вместо сканирования локаторов.", "decision"),
        ("Progress", "Проверка завершена успешно.", "progress"),
        ("Key Decisions", "Deploy в staging через the new parser.", "decision"),
        ("Progress", "好", "progress"),
    ],
)
def test_knowledge_in_any_script_is_captured(heading, body, tag):
    """Q8: non-Latin knowledge must not be a silent English-only `ok`/0."""
    rows = extract(f"## {heading}\n- {body}")
    assert [row["body"] for row in rows] == [body]
    assert rows[0]["tags"] == [tag]


def test_a_section_with_no_judgeable_prose_is_counted_not_silent_ok_zero(store):
    """A locator-only section is DECLINED at admission, never a clean `ok`/0."""
    rows = extract("## Progress\n- https://example.invalid/only-a-locator")
    assert list(rows) == [] and rows.skipped == 1
    assert persist(store, "locator-only-section", rows, "ok") == (
        "declined",
        "admission-skip",
    )


#: The residue table. The verdict is what remains after every locator token is
#: removed, so the same table covers M1/MS2 and the round-2/3 cases; that is the
#: point of the design (QA round 4).
RESIDUE_DEBRIS = [
    # M1: trailing separators, spaced and unspaced
    "C:\\Program Files\\app\\",
    "C:\\Users\\First Last\\",
    "C:\\repo\\My App\\",
    "\\\\server\\Long Folder\\",
    "C:/Program Files/app/",
    # spaced final segment with no separator of its own
    "C:\\Program Files",
    "C:\\repo\\My App",
    "\\\\server\\Long Folder",
    # round 2/3 path-only shapes
    "C:\\Program Files\\Common Files\\proof.txt",
    "C:\\Users\\First Last\\file.txt",
    "C:\\Users\\damian\\file.txt",
    "\\\\server\\share\\file.txt",
    "C:\\Program Files (x86)\\app",
    "C:\\Program Files (x86)\\Common Files\\proof.txt",
    "C:\\Program Files (x86)\\Steam\\steamapps\\common",
    "C:\\Microsoft VS Code\\app",
    "C:\\My Documents\\app",
    "C:\\My Documents Old\\app",
    "C:\\My Documents Old Files\\app",
    "/srv/app",
    "/Users/First Last/app",
    "www.example.com",
    "www.example.com/path",
    "C:\\repo\\app example.com/docs",
    "mailto:ops@example.com",
    "MAILTO:ops@example.com",
    "https://example.invalid/a",
    # other locator-only tokens
    "#2652",
    "44fa24a65c",
    "branch fix/memory-capture-knowledge-first",
    "branch: fix/memory-capture,",
    "(branch fix/memory-capture)",
    "thread: fix/x;",
    "44fa24a65c.",
    "secret QA_SAMPLE_KEY",
    "tests/a.py tests/b.py",
    "A",
]
RESIDUE_KNOWLEDGE = [
    # M2: prose after a path, including a later separator that used to absorb it
    "C:\\repo\\app is the build root at C:\\repo\\out",
    "/srv/app needs read/write",
    "/opt/app handles input/output",
    # leading path plus prose
    "C:\\repo\\app is the build root",
    "C:\\Users\\First Last\\repo\\app is where we deploy from",
    "\\\\server\\share is the evidence root",
    "C:\\Program Files\\app is installed for all users",
    "\\\\server\\share\\dir is where evidence lives",
    "/srv/app is the build root",
    "/Users/First Last/repo/app is where we deploy from",
    "C:\\repo\\My App is the build root",
    # trailing and middle paths
    "The build root is C:\\repo\\app",
    "Store evidence at C:\\Program Files\\app for later",
    "C:\\repo\\app done",
    "/srv/app read-only",
    "C:\\repo\\app OK",
    "C:\\repo\\app Minerva",
    "TCP/IP",
    "and/or",
    "read/write",
    "rotate the dashboard key each quarter",
    "tests/a.py moved to tests/b.py",
    "scripts/a.mjs moved to scripts/b.mjs",
    "src/foo.py now imports src/bar.py",
    "lib/foo.py calls lib/bar.py",
    "C:\\repo\\app docs at example.com/docs",
    "C:\\repo\\app see www.example.com/path for details",
    # prose-first controls
    "We deploy at example.com/docs every Friday.",
    "We deploy from C:\\Program Files\\app every Friday.",
    "Use the docs at https://example.invalid/runbook when shipping.",
]


@pytest.mark.parametrize("body", RESIDUE_DEBRIS)
def test_residue_is_empty_for_a_locator_only_body(body):
    """B1/round 4: no prose left after locator removal means debris, not a fact."""
    rows = extract("## Progress\n- " + body)
    assert list(rows) == [] and rows.skipped == 1


@pytest.mark.parametrize("body", RESIDUE_KNOWLEDGE)
def test_residue_is_prose_for_a_body_that_contains_a_locator(body):
    """B1/round 4: prose on either side of a path keeps the fact."""
    rows = extract("## Key Decisions\n- " + body)
    assert [row["body"] for row in rows] == [body]
    assert rows[0]["tags"] == ["decision"]
    assert rows.skipped == 0


@pytest.mark.parametrize(
    "body",
    [
        r"C:\Users\damian\file.txt",
        r"C:\Program Files\Common Files\proof.txt",
        r"C:\Users\First Last\file.txt",
        r"\\server\share\file.txt",
        "www.example.com",
        "www.example.com/path",
        "mailto:ops@example.com",
        "MAILTO:ops@example.com",
        "/srv/app",
        "/Users/First Last/app",
    ],
)
def test_adjacent_locator_shapes_are_context_not_facts(body):
    """MINOR 1: a candidate that IS a locator shape is context, not a fact."""
    rows = extract("## Progress\n- " + body)
    assert list(rows) == [] and rows.skipped == 1


@pytest.mark.parametrize(
    "body",
    [
        r"C:\repo\app is the build root",
        r"C:\Users\First Last\repo\app is where we deploy from",
        r"\\server\share is the evidence root",
        r"C:\Program Files\app is installed for all users",
        r"\\server\share\dir is where evidence lives",
        "/srv/app is the build root",
        "/Users/First Last/repo/app is where we deploy from",
    ],
)
def test_a_leading_locator_plus_prose_is_knowledge(body):
    """B1: a path that starts the body must not swallow the prose after it."""
    rows = extract("## Key Decisions\n- " + body)
    assert [row["body"] for row in rows] == [body]
    assert rows[0]["tags"] == ["decision"]
    assert rows.skipped == 0


@pytest.mark.parametrize(
    "body",
    [
        "We deploy at example.com/docs every Friday.",
        r"We deploy from C:\Program Files\app every Friday.",
        "Use the docs at https://example.invalid/runbook when shipping.",
    ],
)
def test_a_decision_that_mentions_a_locator_is_still_knowledge(body):
    """The control for MINOR 1: the fix must not delete knowledge that names a path."""
    rows = extract("## Key Decisions\n- " + body)
    assert [row["body"] for row in rows] == [body]
    assert rows[0]["tags"] == ["decision"]
    assert rows.skipped == 0


def test_all_six_prompt_sections_are_recognised():
    """m2: the capture scheme must agree with what the prompt mandates."""
    summary = (
        "# Goal\n- Ship the memory fix.\n"
        "# Constraints\n- Do not touch the write path.\n"
        "# Progress\n- Wrote the parser and ran the suite.\n"
        "# Key Decisions\n- Keep locators as context only.\n"
        "# Next Steps\n- Open the PR and run QA.\n"
        "# Critical Context\n- The live store is 54 MB.\n"
    )
    rows = extract("", summary)
    assert len(rows) == 6
    assert {row["tags"][0] for row in rows} == {
        "goal",
        "constraint",
        "progress",
        "decision",
        "next-step",
        "context",
    }
    assert "Do not touch the write path." in {row["body"] for row in rows}
    assert "Open the PR and run QA." in {row["body"] for row in rows}


def test_the_prompt_exclusion_and_heading_vocabulary_are_pinned_to_the_template():
    """m3/MINOR 2: the pin is EQUALITY, so a new prompt section cannot pass unseen."""
    from local_operator.compaction.api import _load_instruction_template
    from local_operator.memory_capture import _HEADING_TAGS, _PROMPT_SIGNATURE

    template = _load_instruction_template()
    assert template.splitlines()[0].startswith(_PROMPT_SIGNATURE)
    mandated = {
        line[3:].strip().lower() for line in template.splitlines() if line.startswith("## ")
    }
    # The two headings that are prompt machinery, not summary sections the model
    # is asked to fill. Kept explicit so a new scaffolding heading is a red test.
    scaffolding = {
        "previous summary (fold into the sections above, then drop this section)",
        "conversation to summarize",
    }
    assert mandated - scaffolding == set(_HEADING_TAGS), sorted(
        (mandated - scaffolding) ^ set(_HEADING_TAGS)
    )
    assert mandated - set(_HEADING_TAGS) == scaffolding, sorted(
        (mandated - set(_HEADING_TAGS)) - scaffolding
    )


def test_carry_rows_never_overshoots_the_context_body_cap():
    """n1: the candidate and its separator must count toward the 600-char cap."""
    from local_operator.memory_capture import carry_rows

    segment = "/src/" + "/".join(["dir"] * 65) + "/proof"
    carry = (
        "- Asks: "
        + "x" * 280
        + "\n"
        + "".join(f"- Identifiers: {segment}{n}.md\n" for n in range(3))
    )
    rows = carry_rows(carry, locators=True)
    assert len(rows) == 1
    # The first identifier fits; the second would overshoot once its own
    # characters and separator are counted. Before the fix the second was
    # admitted and the body reached 855.
    assert 280 < len(rows[0]["body"]) <= 600


def test_line_heavy_input_stops_inside_the_heading_walk(monkeypatch):
    """n2: the first budget check must happen inside the lazy line walk."""
    import local_operator.memory_capture as capture

    ticks = count(0)
    monkeypatch.setattr(
        capture,
        "time",
        types.SimpleNamespace(monotonic=lambda: next(ticks) * 0.5, time=time.time),
    )
    with pytest.raises(TimeoutError):
        extract("plain prose\n" * 200_000)


@pytest.mark.parametrize(
    "text",
    [
        NOISY_LOCATORS,
        "See guide://memory/docs/example and scratchpad://notes/proof.md",
        "The thread about the branch that we discussed is ready.",
        "Ratios 9/1 and 3/4 are not absolute paths.",
    ],
)
def test_capture_ignores_prose_ratios_and_internal_uri_fragments(text):
    assert extract(text) == []


@pytest.mark.parametrize(
    "text,expected",
    [
        (USEFUL_LOCATORS, USEFUL_BODIES),
        ("URL:https://example.invalid/colon", {"https://example.invalid/colon"}),
        ("docs/説明.md branch: 修正/測定", {"docs/説明.md", "branch: 修正/測定"}),
        (
            "(/repo/proof.md) docs/proof.md src/task.py",
            {"/repo/proof.md", "docs/proof.md", "src/task.py"},
        ),
        ("./proof.md ../docs/proof.md", {"./proof.md", "../docs/proof.md"}),
        (
            "branch fix/capture thread: 1700000000.123456",
            {"branch fix/capture", "thread: 1700000000.123456"},
        ),
        ("thread 1700000000.123456 branch: main", {"thread 1700000000.123456", "branch: main"}),
        ("secret GITHUB_TOKEN #LOH-25 abc123def", {"secret GITHUB_TOKEN", "#LOH-25", "abc123def"}),
    ],
)
def test_capture_keeps_complete_useful_locators_as_context(text, expected):
    """The grammar is unchanged; only the ADMISSION destination is.

    A locator used to be a fact of its own. It now rides as `— context: …` on
    the section's decision, and the fenced block keeps the fixture's prose from
    becoming a separate paragraph fact. A locator-only section still yields
    zero facts, which the next test pins.
    """
    rows = extract(_decision_section(f"```\n{text}\n```"), locators=True)
    assert len(rows) == 1 and rows[0]["tags"] == ["decision"]
    assert rows[0]["body"].startswith(CONTEXT_FACT)
    for locator in expected:
        assert locator in rows[0]["body"], (locator, rows[0]["body"])
    assert all(row["body"] != locator for row in rows for locator in expected)


@pytest.mark.parametrize("kind,part", [("branch", "feature/"), ("thread", "topic/")])
def test_oversized_reference_never_persists_or_recalls_a_prefix(store, kind, part):
    text = f"{kind}: " + part * 24 + "trailing"
    rows = extract(_decision_section(f"```\n{text}\n```"), locators=True)
    persist(store, "oversized-reference", rows, "ok")
    # Exercise retention before asserting: an earlier parser really wrote and
    # recalled an invented prefix, not just a different intermediate list.
    assert store.search("trailing " + part.rstrip("/")) == []
    assert watermark(store, "oversized-reference") == ("ok", 1)
    assert [row["body"] for row in rows] == [CONTEXT_FACT]


def test_reference_at_the_identifier_limit_stays_complete():
    text = "branch: " + "r" * 128
    rows = extract(_decision_section(f"```\n{text}\n```"), locators=True)
    assert [row["body"] for row in rows] == [f"{CONTEXT_FACT} — context: {text}"]


def test_capture_noise_does_not_displace_useful_locators():
    noise = " ".join(f"guide://example/item-{n} ratio 9/{n}" for n in range(100))
    rows = extract(_decision_section(f"```\n{USEFUL_LOCATORS} {noise}\n```"), locators=True)
    assert len(rows) == 1
    for body in USEFUL_BODIES:
        assert body in rows[0]["body"], (body, rows[0]["body"])


def test_capture_never_turns_a_long_token_or_scan_cut_into_a_path():
    from local_operator.memory_capture import SCAN_WINDOW

    # Put a slash at a forced window boundary, then genuine source after the
    # oversized token. The continuation is data, not a fresh absolute path.
    long_uri = "guide://" + "x" * (SCAN_WINDOW - len("guide://")) + "/invented/proof.md"
    rows = extract(_decision_section(f"```\n{long_uri} docs/real.md\n```"), locators=True)
    assert len(rows) == 1
    assert "docs/real.md" in rows[0]["body"]
    assert "invented" not in rows[0]["body"]
    for oversized in ("https://example.invalid/" + "x" * 500, "/repo/" + "x" * 500):
        rows = extract(_decision_section(f"```\n{oversized}\n```"), locators=True)
        assert [row["body"] for row in rows] == [CONTEXT_FACT]


# Value-free stage witness for the restart-recovery diagnostic below. It only
# counts entries/returns/raises at existing function boundaries and records a
# fixed exception category plus already-sized row counts. It must never record
# text, IDs, exception messages, locals or timestamps: captured content is
# untrusted, and printing it on failure would leak exactly what capture refuses.
_STAGE_NAMES = ("recovery_text", "extract_text", "extract_summary", "persist_ok", "persist_failed")


def _exception_category(exc: BaseException) -> str:
    if isinstance(exc, asyncio.CancelledError):
        return "CancelledError"
    if isinstance(exc, MemoryError):
        return "SensitiveMemoryError"
    if isinstance(exc, TimeoutError):
        return "TimeoutError"
    if isinstance(exc, sqlite3.Error):
        return "SQLiteError"
    return "Other"


class _StageWitness:
    """Thread-safe finite observer; recovery/extract run in worker threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events = {name: {"enter": 0, "return": 0, "raise": 0} for name in _STAGE_NAMES}
        self._category = {name: "none" for name in _STAGE_NAMES}
        self._rows: dict[str, int | None] = {name: None for name in _STAGE_NAMES}

    def enter(self, stage: str, rows: int | None = None) -> None:
        with self._lock:
            self._events[stage]["enter"] += 1
            if rows is not None:
                self._rows[stage] = rows

    def complete(self, stage: str, rows: int | None = None) -> None:
        with self._lock:
            self._events[stage]["return"] += 1
            if rows is not None:
                self._rows[stage] = rows

    def fail(self, stage: str, exc: BaseException) -> None:
        with self._lock:
            self._events[stage]["raise"] += 1
            self._category[stage] = _exception_category(exc)

    def report(self) -> str:
        # Evaluated only as an assert message, so a passing run stays silent.
        with self._lock:
            return "stages=" + ";".join(
                f"{name}:enter={self._events[name]['enter']},"
                f"return={self._events[name]['return']},raise={self._events[name]['raise']},"
                f"category={self._category[name]},rows={self._rows[name]}"
                for name in _STAGE_NAMES
            )


def _witnessed_recovery(witness: _StageWitness, real):
    def call(*args, **kwargs):
        witness.enter("recovery_text")
        try:
            result = real(*args, **kwargs)
        except BaseException as exc:
            witness.fail("recovery_text", exc)
            raise
        witness.complete("recovery_text")
        return result

    return call


def _witnessed_extract(witness: _StageWitness, real):
    def call(*args, **kwargs):
        text = args[0] if args else kwargs.get("text", "")
        # finish() is the only extract("", summary) caller; recovery text is the
        # only non-empty first argument. Classify by that boundary, not content.
        stage = "extract_summary" if text == "" else "extract_text"
        witness.enter(stage)
        try:
            result = real(*args, **kwargs)
        except BaseException as exc:
            witness.fail(stage, exc)
            raise
        witness.complete(stage, len(result) if isinstance(result, list) else None)
        return result

    return call


def _witnessed_persist(witness: _StageWitness, real):
    def call(*args, **kwargs):
        status = args[3] if len(args) > 3 else kwargs.get("status")
        rows = args[2] if len(args) > 2 else kwargs.get("rows")
        stage = "persist_ok" if status == "ok" else "persist_failed"
        witness.enter(stage, len(rows) if isinstance(rows, list) else None)
        try:
            result = real(*args, **kwargs)
        except BaseException as exc:
            witness.fail(stage, exc)
            raise
        witness.complete(stage)
        return result

    return call


@pytest.fixture
def store(tmp_path):
    return MemoryStore(
        tmp_path / "cfg",
        MemoryIdentity("capture-session", "t", "a"),
        tmp_path / "sessions" / "capture-session",
        origin="capture-device",
    )


def test_extractor_returns_data_only_and_refuses_sensitive_locators(tmp_path):
    rows = extract(
        _decision_section(
            "```\nSee https://example.invalid/thread/123 and /src/task.py and "
            "~/.ssh/id_rsa; secret PASSPORT_NO\n```"
        ),
        locators=True,
    )
    assert len(rows) == 1
    assert "https://example.invalid/thread/123" in rows[0]["body"]
    assert "PASSPORT_NO" in rows[0]["body"]
    assert "id_rsa" not in rows[0]["body"]
    assert list(tmp_path.iterdir()) == []
    assert extract(" ".join("/src/item%d.py" % n for n in range(100))) == []


def test_bounded_snapshot_ignores_images():
    messages = [
        Message(role="user", content=[TextContent(text="/src/needle.py " + "x" * 1000000)])
    ] * 100
    bounded = snapshot(messages)
    assert len(bounded.encode()) <= 2000000
    assert "/src/needle.py" in bounded


def test_explicit_store_associations_keep_whole_uris_and_relative_paths(store):
    # _reindex also consumes the shared matcher. Explicit facts may reference an
    # internal resource even though best-effort auto-capture ignores those URIs.
    store.put(
        "session",
        {
            "title": "Resource evidence",
            "body": "guide://memory/docs/example ./proof.md URL:https://example.invalid/colon",
        },
    )
    with store.connection() as db:
        assert {row[0] for row in db.execute("SELECT locator FROM associations")} == {
            "guide://memory/docs/example",
            "./proof.md",
            "https://example.invalid/colon",
        }


def test_capture_idempotence_watermark_and_semantic_reuse(store):
    rows = extract(
        "",
        "# Goal\n- Complete release\n# Key Decisions\n- Keep stable prefix\n"
        "# Other\n- not semantic",
    )
    persist(store, "marker-1", rows, "ok")
    count = store.inventory()["ops"]
    persist(store, "marker-1", rows, "ok")
    assert store.inventory()["ops"] == count
    assert watermark(store, "marker-1") == ("ok", 2)
    assert watermark(store, "uncommitted") is None
    assert hint(store, "uncommitted") == ""
    assert "2 candidates" in hint(store, "marker-1")
    assert store.search("stable prefix")[0]["status"] == "candidate"
    assert {row["tags"][0] for row in rows} == {"goal", "decision"}


@pytest.mark.asyncio
async def test_curated_summary_facts_survive_a_full_transcript_scan(store):
    """SUMMARY FIRST: the 52-row cap truncates from the end, not the knowledge."""

    async def scan():
        return _scan_rows(40)

    summary = "## Key Decisions\n" + "\n".join(f"- Keep curated fact number {n}" for n in range(40))
    terminal = await finish(store, "summary-first", asyncio.create_task(scan()), summary)
    assert terminal == "ok"
    assert watermark(store, "summary-first") == ("ok", 52)
    assert store.search("curated fact number 39"), "the scan must not truncate the summary"
    assert store.search("curated fact number 0")


@pytest.mark.asyncio
async def test_capture_late_completion_never_blocks_commit_or_wakes(store):
    gate = asyncio.Event()

    async def capture():
        await gate.wait()
        return extract("## Progress\n- Wrote /src/late.py and verified it")

    task = asyncio.create_task(capture())
    writer = asyncio.create_task(finish(store, "committed-marker", task, ""))
    await asyncio.sleep(0)
    assert not writer.done()
    assert not store.path.exists()  # speculative extractor and waiting writer wrote nothing
    gate.set()
    await writer
    assert watermark(store, "committed-marker") == ("ok", 1)
    assert store.search("late")


@pytest.mark.asyncio
async def test_failure_is_terminal_not_false_capture_success(store):
    async def broken():
        raise RuntimeError("synthetic extractor failure")

    await finish(store, "failed-marker", asyncio.create_task(broken()), "")
    assert watermark(store, "failed-marker") == ("failed", 0)
    assert "did not complete" in hint(store, "failed-marker")
    assert not store.search("synthetic")


@pytest.mark.asyncio
async def test_real_committed_compaction_finishes_capture_while_idle_without_waking(
    tmp_path, monkeypatch
):
    import threading

    import local_operator.memory_capture as capture
    from local_operator.compaction.api import CompactionSettings
    from local_operator.harness.types import ModelSpec, StreamEndEvent, StreamTextDelta
    from local_operator.session.session import Session
    from local_operator.session.transcript import Transcript

    root = tmp_path / "config"
    root.mkdir()
    (root / "config.yml").write_text(
        '{"values":{"memory":{"enabled":true,"capture":true,"suggest":true}}}'
    )
    gate = threading.Event()
    entered = threading.Event()
    original = capture.extract_messages

    def gated(messages, **kwargs):
        entered.set()
        assert gate.wait(5), "capture setup gate was never released"
        return original(messages, **kwargs)

    monkeypatch.setattr(capture, "extract_messages", gated)
    requests = []

    def stream(request, signal):
        requests.append(request)

        async def events():
            yield StreamTextDelta(delta="## Progress\n- Captured docs/idle.md")
            yield StreamEndEvent(stop_reason="stop")

        return events()

    transcript = Transcript(root / "sessions" / "idle-native")
    session = Session(
        model=ModelSpec(
            provider="test", model_id="deterministic", context_window=100000, supports_images=False
        ),
        stream_fn=stream,
        tools=[],
        transcript=transcript,
        system_blocks_provider=lambda: ["stable"],
        compaction_settings=CompactionSettings(keep_recent_tokens=40, strategy="context-full"),
    )
    session._config_dir = root
    events = []
    session.subscribe(events.append)
    try:
        for n in range(3):
            await session.prompt(f"Turn {n} docs/idle.md " + "detail " * 35)
        outcome = await session.compact_now()
        assert outcome.ran
        marker = transcript.latest_entry("compaction")
        assert marker is not None
        assert await asyncio.to_thread(entered.wait, 2)
        memory = MemoryStore(root, MemoryIdentity("idle-native", "", "main"), transcript.directory)
        assert watermark(memory, marker.id) is None
        before = (len(requests), sum(event.type == "agent_start" for event in events))
        gate.set()
        pending = list(session._background_tasks)
        if pending:
            await asyncio.wait_for(asyncio.gather(*pending), 3)
        assert (state := watermark(memory, marker.id)) is not None
        assert state[0] == "ok"
        assert before == (len(requests), sum(event.type == "agent_start" for event in events))
        assert memory.search("idle")
        await session.prompt("Deliberate wake instrument control")
        with pytest.raises(AssertionError):
            assert before == (len(requests), sum(event.type == "agent_start" for event in events))
    finally:
        gate.set()
        await session.dispose()


@pytest.mark.asyncio
async def test_restart_recovers_committed_marker_without_waking(tmp_path, monkeypatch):
    import local_operator.memory_capture as capture
    from local_operator.harness.types import ModelSpec, StreamEndEvent
    from local_operator.session.session import Session
    from local_operator.session.transcript import Transcript

    # Diagnostic only: transparent wrappers always call the real functions once
    # and return/re-raise unchanged. They do not preseed a terminal identity,
    # retry, poll or force "ok"; the strict assertions below remain the oracle.
    witness = _StageWitness()
    monkeypatch.setattr(
        capture, "recovery_text", _witnessed_recovery(witness, capture.recovery_text)
    )
    monkeypatch.setattr(capture, "extract", _witnessed_extract(witness, capture.extract))
    monkeypatch.setattr(capture, "persist", _witnessed_persist(witness, capture.persist))
    root = tmp_path / "config"
    root.mkdir()
    (root / "config.yml").write_text(
        '{"values":{"memory":{"enabled":true,"capture":true,"suggest":true}}}'
    )
    transcript = Transcript(root / "sessions" / "restart-session")
    await transcript.append_message(
        Message.user("## Key Decisions\n- Keep evidence at ~/tools/widget/docs/proof.md")
    )
    kept = await transcript.append_message(Message.user("Kept turn"))
    marker = await transcript.append_compaction("# Goal\n- Recover committed goal", kept.id, 100)
    # No capture writes: this is the exact crash window after marker fsync.
    requests = []

    def stream(request, signal):
        requests.append(request)

        async def events():
            yield StreamEndEvent(stop_reason="stop")

        return events()

    reopened = Session(
        model=ModelSpec(provider="test", model_id="deterministic", context_window=100000),
        stream_fn=stream,
        tools=[],
        transcript=Transcript(transcript.directory),
        system_blocks_provider=lambda: ["stable"],
    )
    reopened._config_dir = root
    assert not (root / "memory").exists()
    assert await capture.session_tail(reopened) == ""
    pending = list(reopened._background_tasks)
    await asyncio.wait_for(asyncio.gather(*pending), 3)
    memory = MemoryStore(root, MemoryIdentity("restart-session", "", "main"), transcript.directory)
    assert (state := watermark(memory, marker.id)) is not None, witness.report()
    assert state[0] == "ok", witness.report()
    assert memory.search("recovered")
    assert state == ("ok", 2)  # recovered decision + reused semantic goal
    assert any("~/tools/widget/docs/proof.md" in item["body"] for item in memory.search("proof"))
    assert not memory.search("software-development")
    assert not memory.search("branch that")
    assert not memory.search("/1")
    assert requests == []
    assert "memory_hint" in await capture.session_tail(reopened)
    await reopened.dispose()


def test_suggestion_guard_suppresses_generic_one_word_and_filters_scope(store):
    store.put(
        "session",
        {
            "title": "Mercury release",
            "body": "Mercury deployment decisions",
            "tags": ["mercury", "deployment"],
        },
    )
    assert store.search("Mercury deployment", suggest=True, tokens=180)
    assert not store.search("how do I create a new branch", suggest=True, tokens=180)
    foreign = MemoryStore(store.root.parent, MemoryIdentity("other", "other", "other"))
    assert not foreign.search("Mercury deployment", suggest=True)


def test_suggestion_slot_is_one_per_fact_and_budget_is_the_rendered_line(store):
    """Two rows carrying one body cost one slot, and 180 tokens means the WIRE.

    Both halves were measured defects: capture writes the same body under
    several titles ("... corrected", "... again"), so a literal per-row cap
    spent two of three slots on one fact and crowded out a distinct matching
    fact. And the budget counted ``_dump(item)`` (~87-90 tokens per candidate)
    while the rendered line costs 32-35, so ``tokens=180, limit=3`` was really a
    two-entry cap in every fixture. Both are pinned here so a future ruler
    change cannot silently restore either.
    """
    from local_operator.compaction.tokens import count_text_tokens
    from local_operator.memory import (
        SUGGESTION_TOKENS,
        _dump,
        suggest_line,
        suggestion_block,
    )

    for suffix in ("", " corrected", " again"):
        store.put(
            "session",
            {
                "title": f"Mercury deployment{suffix}",
                "body": "Mercury deployment proof docs/proof.md",
                "tags": ["mercury", "deployment", "proof"],
            },
        )
    store.put(
        "session",
        {
            "title": "Mercury rollout owner",
            "body": "Mercury deployment owner is the platform team",
            "tags": ["mercury", "deployment"],
        },
    )
    for n in range(4):  # distinct facts that all match the same two terms
        store.put(
            "session",
            {
                "title": f"Mercury release note {n}",
                "body": f"Mercury deployment detail number {n} docs/note{n}.md",
                "tags": ["mercury", "deployment", f"note{n}"],
            },
        )

    items = store.search("Mercury deployment proof", tokens=180, limit=3, suggest=True)
    titles = [str(item["title"]) for item in items]
    assert len(items) == 3, titles
    # The three duplicate-body rows occupy ONE slot...
    assert sum(1 for title in titles if title.startswith("Mercury deployment")) == 1, titles
    # ...which is what lets two DISTINCT facts through, the ones the per-row
    # budget/cap could not reach (whichever two the ranking puts next).
    assert sum(1 for title in titles if not title.startswith("Mercury deployment")) == 2, titles
    # The BLOCK the tail emits, wrapper included — the same accounting the tail
    # applies. A lines-only sum here would pass a set the tail then truncates,
    # which is exactly the long-title defect the fixture below pins.
    rendered = suggestion_block("".join(suggest_line(item) for item in items))
    assert count_text_tokens(rendered) <= SUGGESTION_TOKENS
    # The old ruler would have refused three entries on its own: ~87-90
    # tokens per dumped record against a 180 budget.
    assert sum(count_text_tokens(_dump(item)) for item in items) > 180

    # Explicit search still returns every revision: collapsing is a suggestion
    # convenience, not a fact-visibility rule.
    explicit = store.search("Mercury deployment proof", tokens=2000, limit=20)
    assert len([i for i in explicit if i["body"] == "Mercury deployment proof docs/proof.md"]) == 3


@pytest.mark.asyncio
async def test_long_titles_admit_only_the_block_the_tail_renders(tmp_path, monkeypatch):
    """The suggestion budget is the emitted BLOCK; the wrapper is not free.

    The admission ruler and the renderer's ruler sat 19 tokens apart (header 14
    + closing 5) because ``search`` charged ``suggest_line`` alone. Ordinary
    titles never reached that gap; long ones do: three ~150-character titles
    cost 165-175 as LINES and 184-194 as the block the tail emits, so
    ``search(tokens=180, limit=3)`` admitted 3 and the wire rendered 2 — the
    documented ``limit=3`` yielding 2 again, confined to long titles. The
    assertion that binds the two rulers is the block the tail actually emits,
    never a lines-only sum (review round 1, MINOR 1), and the control shows a
    lines-only ruler really would admit all three on this fixture.
    """
    import local_operator.memory_capture as capture
    from local_operator.compaction.tokens import count_text_tokens
    from local_operator.harness.types import ModelSpec, StreamEndEvent, StreamTextDelta
    from local_operator.memory import (
        RECALL_HARD_TOKENS,
        SUGGESTION_TOKENS,
        suggest_line,
        suggestion_block,
    )
    from local_operator.session.session import Session
    from local_operator.session.transcript import Transcript

    # UUIDv7 randomness changes the tokenizer's hex segmentation, moving this
    # boundary fixture across its ruler even when rendered content is unchanged.
    ids = count(1)
    monkeypatch.setattr(
        "local_operator.memory._id",
        lambda: f"019f4a82-fc3b-7dab-9aaf-{next(ids):012x}",
    )

    titles = [
        "Mercury deployment proof runbook for the platform team covering rollback, canary "
        "checks and the on call escalation path used during a release",
        "Mercury deployment proof checklist for db migration v14 with backfills, index builds "
        "and the verification queries the operator runs before traffic shifts",
        "Mercury deployment proof note on the rollback window decision 2026-10-09, rejected "
        "shorter variant included, and the incident review follow ups",
    ]
    assert all(140 <= len(title) <= 160 for title in titles)

    root = tmp_path / "config"
    root.mkdir()
    (root / "config.yml").write_text(
        '{"values":{"memory":{"enabled":true,"capture":false,"suggest":true}}}'
    )
    transcript = Transcript(root / "sessions" / "long-titles")

    def stream(request, signal):
        async def events():
            yield StreamTextDelta(delta="ok")
            yield StreamEndEvent(stop_reason="stop")

        return events()

    session = Session(
        model=ModelSpec(provider="test", model_id="d", context_window=100000),
        stream_fn=stream,
        tools=[],
        transcript=transcript,
        system_blocks_provider=lambda: ["stable"],
        session_id="long-titles",
    )
    session._config_dir = root
    store = MemoryStore(root, MemoryIdentity("long-titles", "", "main"), transcript.directory)
    for n, title in enumerate(titles):
        store.put(
            "session",
            {
                "title": title,
                "body": f"Mercury deployment proof detail {n}",
                "tags": ["mercury", "deployment", "proof"],
            },
        )
    try:
        await session.prompt("Mercury deployment proof")
        items = store.search(
            "Mercury deployment proof", tokens=SUGGESTION_TOKENS, limit=3, suggest=True
        )
        lines = "".join(suggest_line(item) for item in items)
        # The ruler the tail applies: what search admitted must fit the block.
        assert count_text_tokens(suggestion_block(lines)) <= SUGGESTION_TOKENS
        assert len(items) == 2, [item["title"] for item in items]

        # Reachability control: all three facts match, and a LINES-ONLY ruler —
        # the defect — admits all three at this budget, so the tail would have
        # had to drop one and the wire would contradict the return value.
        unbounded = store.search(
            "Mercury deployment proof", tokens=RECALL_HARD_TOKENS, limit=3, suggest=True
        )
        all_lines = "".join(suggest_line(item) for item in unbounded)
        assert len(unbounded) == 3
        assert count_text_tokens(all_lines) <= SUGGESTION_TOKENS
        assert count_text_tokens(suggestion_block(all_lines)) > SUGGESTION_TOKENS

        tail = await capture.session_tail(session)
        assert tail.count("- memory://") == len(items) == 2, tail
        for item in items:
            assert f"memory://{item['scope']}/{item['id']}" in tail
    finally:
        await session.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "strategy,images",
    [("context-full", False), ("snapcompact", True)],
    ids=["context-full", "snapcompact"],
)
async def test_three_chained_real_compactions_capture_store_restart_retrieve(
    tmp_path, strategy, images
):
    """store -> compact x3 -> restart -> retrieve over a REAL Session, both strategies.

    context-full's summary is model-written; snapcompact keeps no model summary, so
    its knowledge comes from the carry and from heading sections in the transcript
    itself. The explicit store must survive either way, and no chained pass may wake
    the owner.
    """
    from local_operator.compaction.api import CompactionSettings
    from local_operator.harness.types import ModelSpec, StreamEndEvent, StreamTextDelta
    from local_operator.session.session import Session
    from local_operator.session.transcript import Transcript

    root = tmp_path / "config"
    root.mkdir()
    (root / "config.yml").write_text(
        '{"values":{"memory":{"enabled":true,"capture":true,"suggest":true}}}'
    )
    requests = []

    def stream(request, signal):
        requests.append(request)

        async def events():
            yield StreamTextDelta(delta="# Goal\n- Ship release\n# Key Decisions\n- Keep prefix")
            yield StreamEndEvent(stop_reason="stop")

        return events()

    directory = root / "sessions" / f"chain-{strategy}"
    transcript = Transcript(directory)
    model = ModelSpec(
        provider="test", model_id="deterministic", context_window=100000, supports_images=images
    )
    session = Session(
        model=model,
        stream_fn=stream,
        tools=[],
        transcript=transcript,
        system_blocks_provider=lambda: ["stable"],
        session_id=f"chain-{strategy}",
        compaction_settings=CompactionSettings(keep_recent_tokens=40, strategy=strategy),
    )
    session._config_dir = root
    memory = MemoryStore(root, MemoryIdentity(f"chain-{strategy}", "", "main"), directory)
    explicit = memory.put("session", {"title": "Release decision", "body": "Decision: ship Friday"})
    markers = []
    readings: list[tuple[str, Any]] = []
    try:
        for n in range(3):
            for m in range(3):
                await session.prompt(
                    f"Round {n}.{m}\n## Progress\n- Reviewed docs/chain{n}.md " + "detail " * 35
                )
            outcome = await session.compact_now()
            assert outcome.ran, (strategy, n)
            entry = transcript.latest_entry("compaction")
            assert entry is not None
            markers.append(entry.id)
            # Control that the arms really differ: snapcompact's summary is the
            # deterministic archive (no model text); context-full is model-written.
            archived = str(entry.payload.get("summary", "")).startswith("Resume prior conversation")
            assert archived == (strategy == "snapcompact"), entry.payload.get("summary", "")[:80]
            for task in list(session._background_tasks):
                await asyncio.wait_for(task, 5)
            # WAIT FOR THE ROW, and carry its fields into the failure. A chained
            # pass's completion can be scheduled around the snapshot above; the
            # postcondition is the persisted row, which is what the rest of this
            # cell (and a resumed session) actually reads.
            readings.append((entry.id, await _await_capture_row(memory, entry.id)))
        assert len(set(markers)) == 3
        missing = [marker for marker, detail in readings if detail is None]
        assert not missing, (
            "a chained compaction left no capture row: "
            + "; ".join(
                f"{marker}: status={detail.status if detail else None} "
                f"n={detail.n if detail else None} reason={detail.reason if detail else None} "
                f"refused={detail.refused if detail else None} "
                f"skipped={detail.skipped if detail else None}"
                for marker, detail in readings
            )
            + f"; pending_background={len(session._background_tasks)} "
            f"last_terminal={getattr(session, '_memory_terminal', None)!r}"
        )
        before = len(requests)
    finally:
        await session.dispose()
    # Restart: a fresh store handle over the same disk state still retrieves.
    reopened = MemoryStore(root, MemoryIdentity(f"chain-{strategy}", "", "main"), directory)
    assert reopened.show("session", explicit["id"])["fact"]["body"] == "Decision: ship Friday"
    assert reopened.search("chain0")
    assert before >= 9
    print("CHAIN", strategy, "markers", len(markers), "captured", len(reopened.search("docs")))


@pytest.mark.asyncio
async def test_cancelled_capture_is_retried_at_the_next_boundary_without_a_restart(
    tmp_path, monkeypatch
):
    """QA Q5: a capture cancelled in-process left no watermark until the process restarted.

    The committed marker is the authority, so the next prompt boundary re-runs the same
    recovery a restart would. No wake: the retry rides an ordinary user turn.
    """
    import local_operator.memory_capture as capture
    from local_operator.compaction.api import CompactionSettings
    from local_operator.harness.types import ModelSpec, StreamEndEvent, StreamTextDelta
    from local_operator.session.session import Session
    from local_operator.session.transcript import Transcript

    root = tmp_path / "config"
    root.mkdir()
    (root / "config.yml").write_text('{"values":{"memory":{"enabled":true,"capture":true}}}')

    def stream(request, signal):
        async def events():
            yield StreamTextDelta(delta="ok")
            yield StreamEndEvent(stop_reason="stop")

        return events()

    directory = root / "sessions" / "cancel-s"
    transcript = Transcript(directory)
    session = Session(
        model=ModelSpec(provider="test", model_id="d", context_window=100000),
        stream_fn=stream,
        tools=[],
        transcript=transcript,
        system_blocks_provider=lambda: ["stable"],
        session_id="cancel-s",
        compaction_settings=CompactionSettings(keep_recent_tokens=40, strategy="context-full"),
    )
    session._config_dir = root
    real_finish = capture.finish
    calls = {"n": 0}

    async def cancelled_once(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise asyncio.CancelledError
        return await real_finish(*args, **kwargs)

    monkeypatch.setattr(capture, "finish", cancelled_once)
    store = MemoryStore(root, MemoryIdentity("cancel-s", "", "main"), directory)
    try:
        for m in range(4):
            await session.prompt(f"Round {m} see docs/cancel.md " + "detail " * 35)
        assert (await session.compact_now()).ran
        marker = transcript.latest_entry("compaction")
        assert marker is not None
        for task in list(session._background_tasks):
            try:
                await asyncio.wait_for(task, 5)
            except asyncio.CancelledError:
                pass
        assert calls["n"] == 1 and watermark(store, marker.id) is None, "cancelled: no watermark"
        await session.prompt("Ordinary next turn")  # the next boundary re-arms recovery
        for task in list(session._background_tasks):
            await asyncio.wait_for(task, 5)
        assert calls["n"] == 2
        assert watermark(store, marker.id) is not None, "retried without a process restart"
    finally:
        await session.dispose()


# --- The admission/write invariant and the ``meta`` attribution record --------
#
# DEFECT (2026-10, measured on the operator's store): ``extract`` admitted candidates
# with the DEFAULT guard while ``persist`` re-guards title/body through ``validate_fact``
# with ``opaque=True``, which ALSO refuses long hex/base64 runs. A row the scanner had
# accepted was refused at the write, ``persist`` raised out of the transaction, and
# ``finish``'s bare handler recorded ``failed``/0 — one bad row zeroed the whole pass on
# 221 of 587 captures. These tests pin the three halves of the fix: admission matches the
# write, the write survives a row it refuses, and a budget-stopped scan lands its partial
# rows. The ``meta`` attribution record is what makes the remainder attributable.

#: 40 hex chars: the shape the opaque rule refuses as an unnamed key.
HEX = "0123456789abcdef0123456789abcdef01234567"
#: A 40-char mixed-case token with digits: the base64 shape the opaque rule refuses.
BASE64 = "AbCdEfGhIjKlMnOpQrStUvWxYz0123456789AbCd"

CLEAN_CORPUS = (
    USEFUL_LOCATORS,
    "Evidence at ~/docs/proof.md, https://example.invalid/task/25",
    "secret API_TOKEN #LOH-25 abc123def",
    "branch: fix/capture thread: 1700000000.123456",
    "docs/説明.md src/task.py ../docs/proof.md",
)

#: (text, does a candidate survive the stricter guard?) — the false entries are the
#: shapes the DEFAULT guard admits and the WRITE's opaque guard refuses.
GUARD_CORPUS = tuple((text, True) for text in CLEAN_CORPUS) + (
    (f"/docs/{HEX}", False),
    (f"https://example.invalid/{HEX}", False),
    (f"src/{BASE64}.py", False),
)


@pytest.mark.parametrize("text,expect_rows", GUARD_CORPUS)
def test_every_candidate_extract_returns_passes_the_guard_the_write_applies(text, expect_rows):
    """PROPERTY: everything ``extract`` emits survives ``guard(x, opaque=True)``.

    This is the invariant that was violated, asserted with the exact call
    ``validate_fact`` makes on title and body, over ordinary locators AND the long
    hex/base64 shapes only the opaque rule refuses. The corpus rides in a
    knowledge bullet because a bare locator is no longer a candidate at all.
    """
    rows = extract("## Key Decisions\n- Record " + text)
    for row in rows:
        guard(row["body"], opaque=True)
        guard(row["title"], opaque=True)
    assert bool(rows) is expect_rows, (
        "a candidate the write would refuse was admitted anyway"
        if expect_rows
        else "nothing was emitted"
    )


@pytest.mark.parametrize(
    "text,offending",
    [
        (f"/docs/{HEX}", HEX),
        (f"https://example.invalid/{HEX}", HEX),
        (f"src/{BASE64}.py", BASE64),
    ],
)
def test_a_candidate_the_write_would_refuse_is_skipped_not_admitted(text, offending):
    """A refused shape is skipped at admission; the pass keeps the rest of its rows."""
    rows = extract("## Key Decisions\n" + f"- Review {text}\n" + "- Keep ~/docs/proof.md")
    bodies = {row["body"] for row in rows}
    assert not any(offending in body for body in bodies)
    assert "Keep ~/docs/proof.md" in bodies, "the benign neighbour in the same section still lands"


def _capture_row(store, marker_id):
    """The capture's coarse attribution from ``meta``.

    ``captures`` deliberately keeps its 4-column shape (mixed-version rollout), so the
    reason and counts live in the store's existing key/value table.
    """
    with store.connection() as db:
        row = db.execute(
            "SELECT value FROM meta WHERE key=?",
            (f"capture.{store.identity.session_id}.{marker_id}",),
        ).fetchone()
        return json.loads(row[0]) if row else None


def _reason(store, marker_id):
    record = _capture_row(store, marker_id)
    return record["reason"] if record else None


def _detail(store, marker_id):
    """The capture detail, asserting a terminal row exists (the hint's path)."""
    detail = capture_detail(store, marker_id)
    assert detail is not None, "a terminal capture row must exist"
    return detail


def _scan_rows(count_):
    return [
        {"title": f"/src/item{n}.py", "body": f"/src/item{n}.py", "tags": ["locator"]}
        for n in range(count_)
    ]


def _refused_row():
    body = f"https://example.invalid/{HEX}"
    return {"title": body, "body": body, "tags": ["locator"]}


def test_a_clean_pass_records_no_reason(store):
    rows = extract("", "# Goal\n- Ship release\n# Key Decisions\n- Keep prefix")
    persist(store, "clean-marker", rows, "ok")
    assert watermark(store, "clean-marker") == ("ok", 2)
    assert _capture_row(store, "clean-marker") is None
    assert _detail(store, "clean-marker").reason is None


def test_the_write_skips_a_row_it_refuses_and_counts_it(store):
    """BELT: a row shape the admit site does not know costs one candidate, not the pass."""
    persist(store, "belt-marker", _scan_rows(39) + [_refused_row()], "ok")
    assert watermark(store, "belt-marker") == ("ok", 39)
    assert _capture_row(store, "belt-marker") == {
        "reason": "write-refused",
        "refused": 1,
        "skipped": 0,
    }
    assert store.search("item38")


@pytest.mark.asyncio
async def test_one_unwritable_row_cannot_zero_the_whole_capture(store):
    """REGRESSION: 39 good rows beside one 40-hex URI land 39, never ``failed``/0."""

    async def scan():
        return _scan_rows(39) + [_refused_row()]

    terminal = await finish(store, "one-bad-row", asyncio.create_task(scan()), "")
    assert (terminal, watermark(store, "one-bad-row")) == ("ok", ("ok", 39))
    assert _reason(store, "one-bad-row") == "write-refused"


def _fake_scan_clock(monkeypatch, step):
    """Advance the scan clock one ``step`` per reading; keep ``time.time`` real.

    ``extract`` reads ``time.monotonic`` before the scan and on every 64th section
    line, so a per-reading fake clock provokes a budget stop without a multi-second
    test. ``persist`` stamps rows with ``time.time``, which stays the real function.
    """
    import local_operator.memory_capture as capture

    ticks = count(0)
    monkeypatch.setattr(
        capture,
        "time",
        types.SimpleNamespace(monotonic=lambda: next(ticks) * step, time=time.time),
    )
    return "## Key Decisions\n" + "\n".join(
        f"- chose item{n} at /src/item{n}.py" for n in range(3000)
    )


def test_a_scan_that_runs_out_of_budget_hands_back_the_rows_it_found(monkeypatch):
    text = _fake_scan_clock(monkeypatch, 0.5)
    with pytest.raises(TimeoutError) as stopped:
        extract(text)
    assert getattr(
        stopped.value, "rows", None
    ), "a budget stop must carry the rows found so far, not just an exception"


@pytest.mark.asyncio
async def test_budget_exhaustion_lands_a_partial_result_instead_of_failure(store, monkeypatch):
    """PARTIAL BUDGET: the rows found before the stop land, attributed ``timeout``."""
    text = _fake_scan_clock(monkeypatch, 0.5)
    task = asyncio.create_task(asyncio.to_thread(extract, text))
    terminal = await finish(store, "budget-marker", task, "")
    assert (terminal, watermark(store, "budget-marker")) == ("ok", ("ok", 40))
    assert _reason(store, "budget-marker") == "timeout"
    assert "scan budget" in hint(store, "budget-marker")


@pytest.mark.asyncio
async def test_a_budget_stop_with_no_rows_is_a_failure(store, monkeypatch):
    """TOTAL LOSS: a budget stop that found nothing must not read as a completed pass."""
    _fake_scan_clock(monkeypatch, 0.5)
    # No locator in any window: the scan stops having found nothing at all.
    task = asyncio.create_task(asyncio.to_thread(extract, "plain prose " * 1200))
    terminal = await finish(store, "budget-empty", task, "")
    assert (terminal, watermark(store, "budget-empty")) == ("failed", ("failed", 0))
    assert _detail(store, "budget-empty").reason == "timeout"
    message = hint(store, "budget-empty")
    assert "did not complete" in message and "timed out" in message


@pytest.mark.asyncio
async def test_carry_rows_survive_a_budget_stop(store, monkeypatch):
    """MINOR 1: the marquee retention is not the arm that ran out of budget."""
    _fake_scan_clock(monkeypatch, 0.5)
    task = asyncio.create_task(asyncio.to_thread(extract, "plain prose " * 1200))
    terminal = await finish(
        store,
        "budget-carry",
        task,
        "",
        carry="- Asks: keep the launch plan",
    )
    assert (terminal, watermark(store, "budget-carry")) == ("ok", ("ok", 1))
    assert _detail(store, "budget-carry").reason == "timeout"
    assert store.search("launch plan")


@pytest.mark.asyncio
async def test_a_quota_refusal_is_attributed_and_still_records_the_marker(store, monkeypatch):
    """``quota`` is not a generic store error, and the failure marker still lands."""

    def refuse(db, scope, owner, *, relief=False):
        raise MemoryError("Memory quota reached (synthetic).")

    monkeypatch.setattr(store, "_quota", refuse)

    async def scan():
        return _scan_rows(1)

    assert await finish(store, "quota-marker", asyncio.create_task(scan()), "") == "failed"
    assert watermark(store, "quota-marker") == ("failed", 0)
    assert _reason(store, "quota-marker") == "quota"
    assert "quota" in hint(store, "quota-marker")


#: The exact 4-column shape every released store has. A 5th/6th column here is the
#: regression this test exists for: it breaks the PREVIOUS release's positional write.
CAPTURES_SHAPE = ["session", "compaction", "status", "n"]


def test_a_previous_release_store_round_trips_without_a_schema_change(tmp_path):
    """COMPATIBILITY: a 4-column store written positionally still round-trips.

    Running sessions keep the old binary in memory after an upgrade. Its ``persist``
    does ``INSERT INTO captures VALUES (?,?,?,?)`` with no column list, so adding a
    column here would make every one of those sessions' captures fail until restart.
    The reason/counts live in ``meta`` instead, and this test opens a store created and
    written by that old shape through the current reader and writer.
    """
    path = tmp_path / "cfg" / "memory" / "memory.db"
    path.parent.mkdir(parents=True)
    old = sqlite3.connect(path)
    old.executescript(
        "CREATE TABLE captures(session TEXT,compaction TEXT,status TEXT,n INTEGER,"
        " PRIMARY KEY(session,compaction));"
    )
    # The previous release's write, verbatim: no column list, four values.
    old.execute("INSERT INTO captures VALUES ('capture-session','old-marker','failed',0)")
    old.commit()
    old.close()

    store = MemoryStore(
        tmp_path / "cfg",
        MemoryIdentity("capture-session", "t", "a"),
        origin="capture-device",
    )
    with store.connection() as db:
        assert [row[1] for row in db.execute("PRAGMA table_info(captures)")] == CAPTURES_SHAPE
    assert watermark(store, "old-marker") == ("failed", 0)
    persist(store, "new-marker", extract("## Progress\n- Landed /src/after-rollout.py"), "ok")
    assert watermark(store, "new-marker") == ("ok", 1)
    assert _detail(store, "new-marker").reason is None


@pytest.mark.asyncio
async def test_a_pass_whose_every_candidate_is_refused_is_a_failure(store):
    """TOTAL LOSS: all-refused is a failure, not ``ok``/0."""

    async def scan():
        return [_refused_row()]

    terminal = await finish(store, "all-refused", asyncio.create_task(scan()), "")
    assert terminal == "failed"
    assert watermark(store, "all-refused") == ("failed", 0)
    assert _detail(store, "all-refused").refused == 1
    assert _detail(store, "all-refused").reason == "write-refused"
    assert "did not complete" in hint(store, "all-refused")


def test_admission_skips_are_counted_distinctly_from_write_refusals(store):
    """MINOR 2: a stricter admission rule leaves a count, not a silent gap."""
    rows = extract("## Key Decisions\n" f"- Review the credential {HEX}\n" "- Keep ~/docs/proof.md")
    assert rows.skipped == 1
    assert [row["body"] for row in rows] == ["Keep ~/docs/proof.md"]
    assert persist(store, "admission-marker", rows, "ok") == ("ok", "admission-skip")
    assert watermark(store, "admission-marker") == ("ok", 1)
    assert _capture_row(store, "admission-marker") == {
        "reason": "admission-skip",
        "refused": 0,
        "skipped": 1,
    }
    assert "declined by the admission guard" in hint(store, "admission-marker")


def test_a_pass_whose_every_candidate_is_declined_at_admission_is_not_a_failure(store):
    """An admission-only total loss is a benign DECLINE, distinct from a stage fault.

    Nothing the store should have taken was lost: every candidate was noise the
    admission guard refused before the store saw it. Recording ``failed`` here made a
    quiet pass indistinguishable from a timeout or a store error at the status column.
    """
    rows = extract(f"## Key Decisions\n- Review the credential {HEX}")
    assert list(rows) == [] and rows.skipped == 1
    assert persist(store, "admission-empty", rows, "ok") == ("declined", "admission-skip")
    assert watermark(store, "admission-empty") == ("declined", 0)
    assert "declined" in hint(store, "admission-empty")
    assert "did not complete" not in hint(store, "admission-empty")


def test_a_non_mapping_row_costs_one_candidate_not_the_pass(store):
    """NIT 1: the belt's contract holds for a malformed row, not just ``MemoryError``."""
    malformed: Any = ["not-a-mapping"]  # the shape a future producer bug would emit
    terminal = persist(store, "malformed-row", malformed, "ok")
    assert terminal == ("failed", "write-refused")
    assert watermark(store, "malformed-row") == ("failed", 0)
    assert _detail(store, "malformed-row").refused == 1


def test_both_guards_are_counted_and_the_write_belt_is_named(store):
    """BOTH GUARDS: the literal names the write belt; ``meta`` keeps both counts.

    The write belt is the later, stricter guard, and its refusal is the one an
    operator investigating a lost candidate needs named; the admission count stays
    visible beside it rather than being folded into the literal.
    """
    rows = extract("## Key Decisions\n" f"- Review the credential {HEX}\n" "- Keep ~/docs/proof.md")
    assert rows.skipped == 1
    terminal = persist(
        store, "both-guards", list(rows) + [_refused_row()], "ok", skipped=rows.skipped
    )
    assert terminal == ("ok", "write-refused")
    assert _capture_row(store, "both-guards") == {
        "reason": "write-refused",
        "refused": 1,
        "skipped": 1,
    }


def _meta_value(store, key):
    with store.connection() as db:
        row = db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None


def test_the_cohort_marker_starts_at_the_first_attributed_capture(store):
    """The split is dated in ``meta`` so reporting can age rows honestly.

    Rows written before this stamp carry the verdict-shaped literal (or none at all)
    and cannot be reclassified by guard; the stamp is the boundary between the two
    cohorts, written once and never moved.
    """
    assert _meta_value(store, ATTRIBUTION_STARTED_KEY) is None
    persist(store, "cohort-one", extract("## Progress\n- Landed /src/cohort.py"), "ok")
    first = _meta_value(store, ATTRIBUTION_STARTED_KEY)
    assert first is not None
    assert isinstance(json.loads(first)["at"], float)
    persist(store, "cohort-two", extract("## Progress\n- Landed /src/cohort2.py"), "ok")
    assert _meta_value(store, ATTRIBUTION_STARTED_KEY) == first, "first write wins"


def test_a_legacy_verdict_literal_is_unclassifiable_and_never_rewritten(store):
    """NO BACKFILL: a pre-split record keeps its bytes and yields no guard cause.

    ``refused-row`` named no guard, so reading it as either new literal would invent
    a cause the writer never recorded. The raw record is left exactly as written.
    """
    legacy = json.dumps({"reason": "refused-row", "refused": 0, "skipped": 18})
    with store.connection() as db:
        db.execute(
            "INSERT INTO meta VALUES (?,?)",
            (f"capture.{store.identity.session_id}.legacy-marker", legacy),
        )
        db.execute(
            "INSERT INTO captures(session,compaction,status,n) VALUES (?,?,?,?)",
            (store.identity.session_id, "legacy-marker", "failed", 0),
        )
        db.commit()
    detail = _detail(store, "legacy-marker")
    assert detail.reason is None
    assert (detail.status, detail.n, detail.refused, detail.skipped) == ("failed", 0, 0, 18)
    assert _capture_row(store, "legacy-marker") == {
        "reason": "refused-row",
        "refused": 0,
        "skipped": 18,
    }


def test_a_legacy_partial_pass_still_reports_withheld_candidates(store):
    """MINOR 1: an unclassifiable record must not render as a clean success.

    ``capture_detail`` refuses the pre-split verdict literal, so the reason reads
    ``None``; the stored counts are the only surviving evidence that the pass was
    partial. The hint reports them without naming a guard — the one thing it must not
    do is invent a cause or claim a clean pass on the records this change cannot
    reclassify.
    """
    meta = json.dumps({"reason": "refused-row", "refused": 0, "skipped": 18})
    with store.connection() as db:
        db.execute(
            "INSERT INTO meta VALUES (?,?)",
            (f"capture.{store.identity.session_id}.legacy-partial", meta),
        )
        db.execute(
            "INSERT INTO captures(session,compaction,status,n) VALUES (?,?,?,?)",
            (store.identity.session_id, "legacy-partial", "ok", 5),
        )
        db.commit()
    message = hint(store, "legacy-partial")
    assert "18 candidates were withheld (cause not recorded)" in message
    assert "declined by the admission guard" not in message
    assert "withheld by the memory guard at the write" not in message
    # A genuinely clean pass keeps the unqualified wording.
    persist(store, "legacy-clean", _scan_rows(2), "ok")
    assert "withheld" not in hint(store, "legacy-clean")


def test_a_duplicate_persist_reports_the_terminal_pair_the_store_holds(store):
    """MINOR 2: a replayed marker reports the recorded status, not the requested one.

    ``declined`` is derived inside ``persist`` from what the pass wrote, so the
    duplicate early-return must read the row back. Echoing the caller's ``ok`` would
    contradict the row a resumed caller reads next and the docstring that promises
    the terminal pair actually recorded.
    """
    rows = extract(f"## Key Decisions\n- Review the credential {HEX}")
    assert persist(store, "duplicate-marker", rows, "ok") == ("declined", "admission-skip")
    assert persist(store, "duplicate-marker", rows, "ok") == ("declined", "admission-skip")
    assert watermark(store, "duplicate-marker") == ("declined", 0)
    assert persist(store, "duplicate-clean", _scan_rows(2), "ok") == ("ok", None)
    assert persist(store, "duplicate-clean", _scan_rows(2), "ok") == ("ok", None)


#: The four shapes whose reason/status mapping is contractual. ``admission-skip`` and
#: ``write-refused`` are different guards; only the latter is a total loss when every
#: candidate lands there. ``quota`` and ``timeout`` are stage faults and stay ``failed``
#: when nothing was captured. The write refusal is the shape no live record produced
#: (``refused`` summed to zero across all 48), so it is proven here, not assumed.
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "shape,reason,status",
    [
        ("admission-skip", "admission-skip", "declined"),
        ("write-refused", "write-refused", "failed"),
        ("quota", "quota", "failed"),
        ("timeout", "timeout", "failed"),
    ],
)
async def test_the_four_attribution_shapes_map_to_exact_literal_and_status(
    store, monkeypatch, shape, reason, status
):
    if shape == "admission-skip":
        terminal = persist(
            store, shape, extract(f"## Key Decisions\n- Review the credential {HEX}"), "ok"
        )
    elif shape == "write-refused":
        terminal = persist(store, shape, [_refused_row()], "ok")
    elif shape == "quota":

        def refuse(db, scope, owner, *, relief=False):
            raise MemoryError("Memory quota reached (synthetic).")

        monkeypatch.setattr(store, "_quota", refuse)

        async def scan():
            return _scan_rows(1)

        terminal = (await finish(store, shape, asyncio.create_task(scan()), ""), "quota")
    else:
        terminal = persist(store, shape, [], "ok", None, "timeout")
    assert terminal == (status, reason)
    detail = _detail(store, shape)
    assert (detail.status, detail.reason) == (status, reason)
