"""The credential-SHAPE pass, on every surface a model can read.

**What this file is for.** ``test_agent_surfaces.py`` guards the redaction of
values the session KNOWS; this one guards the other half — a credential
recognised by how it is SPELLED, which is the only thing that can catch a secret
this session was never told. That distinction is the whole defect it closes: a
subagent ran ``kubectl exec … env`` against a production pod, the masking in the
pipeline was the agent's own ``sed`` (whose pattern had no ``DSN``), and the
harness had no fallback, so ``MONGO_DSN=mongodb+srv://user:<pw>@host`` landed in
a transcript on disk in full.

**Why the corpus is parametrised over surfaces.** The failure this file exists
to prevent is a case that passes on one path and leaks on another, so every
corpus case is run through ALL of the model-visible paths the tree has:
the pattern pass, the composed pass, the variable store's ``redact`` (the loop's
hook and the live-text path), the pipe filter (live stream and background-job
peek), the session's ``redact_tool_result`` hook, and the journaled/replayed
tool-call arguments. A new surface that forgets one of them fails here rather
than in production.

The NEGATIVE half is not a courtesy: this pass runs over every tool result, so a
rule that masks ordinary output blinds the agent to the text it is reading. Both
halves carry their reasons in ``credential_shape_corpus``.
"""

from __future__ import annotations

import asyncio
import json
import re
import shlex
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Callable

import pytest

from local_operator.harness.redaction import (
    current_tool_source,
    summarize_arguments,
    tool_source,
)
from local_operator.harness.types import (
    AbortSignal,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    TextContent,
    ToolCall,
    ToolContext,
)
from local_operator.redaction_shapes import (
    REDACTION_MARKER,
    credential_dump_notice,
    match_shape_names,
    scrub_secrets,
    scrub_shapes,
)
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.tools import builtin
from local_operator.variables import VariableStore
from tests.unit.secrets.credential_shape_corpus import (
    DUMP_COMMAND_CASES,
    NEGATIVE_CASES,
    POSITIVE_CASES,
)

#: A synthetic credential. Never a real one, and never asserted into anything a
#: human reads: the corpus carries the shapes, this carries the containment.
SENTINEL = "mongodb+srv://svc_user:sh4pedSentinelPw@db.internal/app"


def _store(text: str) -> str:
    return VariableStore(cwd=".").redact(text)


def _live_text(text: str) -> str:
    """The UI-facing live text path (bash stream, job peek, abort receipt)."""
    return builtin._redact_tool_text(text, ToolContext(cwd=".", variables=VariableStore(cwd=".")))


def _pipe_whole(text: str) -> str:
    """The pipe filter, one feed plus the end-of-stream flush."""
    redactor = builtin._PipeRedactor([])
    return (redactor.feed(text.encode()) + redactor.feed(b"", final=True)).decode()


def _session_hook(text: str) -> str:
    """The loop's ``redact_tool_result`` hook, over a real Session.

    A FRESH store per call, because the store CONTAINS: a value the shape pass
    matched is registered for the rest of the session, so a shared store would
    let one corpus case mask another's text — correct behaviour, wrong
    measurement (the negative half has to survive in a session that never saw
    that value). The Session itself is reused; only the store is replaced.
    """
    session = _session()
    session._variables = VariableStore(cwd=".")
    return session._redact_tool_result_text(text)


def _journaled(text: str) -> str:
    """The copy of an assistant turn that history STORES and REPLAYS."""
    from local_operator.harness.loop import _scrub_history_arguments

    message = Message(
        role="assistant",
        content=[TextContent(text="")],
        tool_calls=[ToolCall(id="c1", name="bash", arguments={"command": text})],
    )
    scrubbed = _scrub_history_arguments(message, scrub_secrets)
    assert isinstance(scrubbed, Message)
    return str(scrubbed.tool_calls[0].arguments["command"])


#: Every model-visible path, by name. The invariant the corpus pins: a case that
#: is masked on one of these is masked on all of them.
SURFACES: dict[str, Callable[[str], str]] = {
    "shapes": scrub_shapes,
    "composed": scrub_secrets,
    "store.redact": _store,
    "live-text": _live_text,
    "pipe-filter": _pipe_whole,
    "session-hook": _session_hook,
    "journaled-arguments": _journaled,
}


async def _never_streams(_request: Any, _signal: Any = None) -> AsyncIterator[Any]:
    """A provider stream these tests never reach: they call the hook directly.

    An async GENERATOR rather than a plain coroutine because that is what the
    loop's contract requires; a stub with the wrong shape would be a typing
    error here and a ``TypeError`` the day one of these tests grew a turn.
    """
    if False:  # pragma: no cover - makes this an async generator
        yield None
    raise AssertionError("the redaction tests never reach the provider stream")


_SESSION: Session | None = None


def _session() -> Session:
    """One real Session, built the way ``session_factory`` builds one.

    Cached because Session construction is not free and this file builds a lot
    of them; the object is only ever asked to redact, so nothing it accumulates
    across cases matters.
    """
    global _SESSION
    if _SESSION is None:
        cwd = Path.cwd() / ".shape-test-cwd"
        _SESSION = Session(
            model=ModelSpec(provider="test", model_id="unit-model", context_window=1000),
            stream_fn=_never_streams,
            tools=[],
            transcript=Transcript(cwd / "session"),
            system_blocks_provider=lambda *_a: [],
            yolo=True,
            cwd=str(cwd),
            variables=VariableStore(cwd=str(cwd)),
        )
    return _SESSION


# --- the corpus, over every surface -----------------------------------------


@pytest.mark.parametrize("surface", sorted(SURFACES))
@pytest.mark.parametrize("case", POSITIVE_CASES, ids=lambda c: c.reason[:44])
def test_a_credential_shape_is_masked_on_every_surface(
    surface: str, case: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    scrubbed = SURFACES[surface](case.text)
    assert REDACTION_MARKER in scrubbed, f"{surface} left {case.reason} readable"
    assert scrubbed != case.text, f"{surface} did not rewrite {case.reason}"


@pytest.mark.parametrize("surface", sorted(SURFACES))
@pytest.mark.parametrize("case", NEGATIVE_CASES, ids=lambda c: c.reason[:44])
def test_ordinary_text_is_left_byte_identical_on_every_surface(
    surface: str, case: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Over-masking is a defect: these must survive untouched, everywhere."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    assert SURFACES[surface](case.text) == case.text, f"{surface} rewrote {case.reason}"


def test_the_corpus_is_big_enough_to_be_evidence() -> None:
    """A guard on the corpus itself, so it cannot be quietly trimmed.

    The counts are the contract the PR body states; asserting them here means a
    later edit that removes cases fails rather than passing with less coverage.
    """
    assert len(POSITIVE_CASES) >= 150
    assert len(NEGATIVE_CASES) >= 50
    assert len(DUMP_COMMAND_CASES) >= 60
    # Both halves must carry a reason: a case without one cannot be argued with.
    assert all(case.reason.strip() for case in (*POSITIVE_CASES, *NEGATIVE_CASES))


def test_the_incidents_own_line_is_a_regression_case() -> None:
    """The measured incident, verbatim apart from the credential itself.

    Kept as its own test rather than only a corpus row because it is the reason
    the module exists, and a future narrowing that leaves the generic cases
    passing while dropping this one would be a false green.
    """
    line = (
        "MONGO_DSN=mongodb+srv://agent_runtime_model_worker:"
        "<pw>@mongodb-prod.example.net/agent_runtime"
    ).replace("<pw>", "Sup3rSecretPw")
    scrubbed = scrub_secrets(line)
    assert "Sup3rSecretPw" not in scrubbed
    assert "agent_runtime_model_worker" in scrubbed, "the user must stay readable"
    assert "mongodb-prod.example.net" in scrubbed, "the host must stay readable"
    assert scrubbed.startswith("MONGO_DSN=mongodb+srv://agent_runtime_model_worker:")


# --- the pattern pass alone --------------------------------------------------


def test_match_shape_names_reports_labels_and_never_values() -> None:
    labels = match_shape_names(SENTINEL)
    assert labels == ["dsn-password"]

    combined = "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG\nPASSWORD=hunter2hunter2"
    assert set(match_shape_names(combined)) == {"credential-assignment"}


def test_match_shape_names_is_empty_for_ordinary_text() -> None:
    assert match_shape_names("max_tokens=4096 and SERVICE_URL=https://api.example.com") == []


def test_scrub_shapes_and_scrub_secrets_agree_on_the_shape_half() -> None:
    """The composed pass is the shape pass over the value pass — not a parallel
    table. Two tables is how one surface ends up masking a DSN and another not."""
    text = f"plain kh17 value: {SENTINEL}"
    assert scrub_secrets(text, ["kh17"]) == scrub_shapes(text.replace("kh17", REDACTION_MARKER))


def test_every_positive_case_trips_the_gate() -> None:
    """The prefilter must not be able to hide a rule.

    ``redaction_shapes`` skips the whole table when a text carries none of its
    anchor substrings, because each rule is a full scan and a tool result can be
    megabytes. A subset of the anchors would therefore fail SILENTLY — a shape
    that no longer fires because nothing in the gate noticed it. This is the test
    that makes adding such a rule fail loudly instead, in CI, against the corpus.
    """
    from local_operator.redaction_shapes import has_shape_anchor

    misses = [case.text for case in POSITIVE_CASES if not has_shape_anchor(case.text)]
    assert not misses, f"these shapes would be skipped by the gate: {misses[:5]}"


def test_the_gate_does_not_trip_on_ordinary_text() -> None:
    """The other half of the gate: it exists to skip work, so it has to skip.

    Asserted on the corpus negatives as a whole (they are ordinary text by
    construction) and on a realistic tool result, because a gate that trips on
    everything costs the table's full price on every result.
    """
    from local_operator.redaction_shapes import has_shape_anchor

    body = ("GET /v1/items 200 12ms\n" * 200) + '{"port": 8080, "status": "ok"}\n' * 100
    assert not has_shape_anchor(body), "an ordinary tool result must skip the table"
    assert not has_shape_anchor("total 12\ndrwxr-xr-x  4 user staff 128 Sep 18 09:12 src")
    assert not has_shape_anchor("SELECT id, name FROM users WHERE active = true;")
    # Many corpus NEGATIVES DO trip it (`max_tokens`, `cache_key`, prose with
    # "password"), and that is by design rather than a leak in the gate: the
    # anchors are deliberately broad because a MISSING one would silently stop a
    # rule from firing, while a FALSE POSITIVE only costs the table's price on
    # that one line — and the corpus asserts the outcome there is unchanged. The
    # count is recorded so a widening that makes the gate useless is visible.
    tripped = [case.text for case in NEGATIVE_CASES if has_shape_anchor(case.text)]
    assert len(tripped) < len(NEGATIVE_CASES) * 0.8, len(tripped)


def test_the_gated_pass_is_fast_on_ordinary_text() -> None:
    """A per-byte budget for the common case, with the margin stated.

    Why a budget at all: this pass runs on every tool result AND on every live
    pipe chunk, so its cost is loop-thread CPU — the resource a TUI freeze is
    made of. The gate exists because the ungated table measured 1.3 µs/byte
    (5 s of blocked loop for a 4 MB result), and a future rule that is
    accidentally quadratic would otherwise be invisible until a user saw a
    frozen screen. Measured here: 0.09 µs/byte for ordinary text after the gate
    and the cheap guards, against 0.4 µs/byte for the same text before them.

    The ceiling is 10x the measured value and is a CATASTROPHE bound, not a
    precision one — the same convention AGENTS.md requires for timing guards on
    a shared machine: it catches an order-of-magnitude regression (a re-added
    per-position loop, a lost gate) without flaking when a loaded runner makes
    this slower by a constant factor. The ratios in
    ``test_the_shape_pass_is_linear_in_the_size_of_the_text`` are the half that
    survives load.
    """
    body = ("GET /v1/items 200 12ms\n" * 2000) + '{"port": 8080, "status": "ok"}\n' * 1000
    elapsed = _scrub_time(body)
    per_byte = elapsed / len(body)
    assert per_byte < 0.9e-6, f"ordinary text costs {per_byte * 1e6:.2f} µs/byte"


def test_no_entropy_heuristic_is_applied() -> None:
    """A bare high-entropy fragment with no spelling around it is NOT masked.

    Stated as a test because it is a deliberate boundary rather than an
    oversight: an entropy rule on this surface rewrites build ids, content
    hashes and base64 thumbnails the agent must read, and still misses a secret
    whose spelling the table does not know. The documented residual.
    """
    for bare in ("a3f5c1d9e7b24680a3f5c1d9e7b24680a3f5c1d9", "01ARZ3NDEKTSV4RRFFQ69G5FAV"):
        assert scrub_shapes(bare) == bare


# --- the pipe filter: splits, bounds, and no stall ---------------------------


def test_a_shape_split_at_every_offset_is_still_masked() -> None:
    """Every split point, not a sampled one.

    The filter's contract is about what it publishes when a child's writes land
    mid-credential; a test that tried three offsets would pass with a lookbehind
    that happens to be long enough for those three. The change from a
    fixed-window holdback to whole lines makes this the property that could
    regress silently, so it is asserted exhaustively.
    """
    raw = f"MONGO_DSN={SENTINEL}\n".encode()
    for offset in range(len(raw) + 1):
        redactor = builtin._PipeRedactor([])
        published = redactor.feed(raw[:offset]) + redactor.feed(raw[offset:])
        published += redactor.feed(b"", final=True)
        text = published.decode()
        assert "sh4pedSentinelPw" not in text, f"shape leaked when split at {offset}"
        assert REDACTION_MARKER in text, f"shape not masked when split at {offset}"


def test_a_known_value_split_at_every_offset_is_still_masked() -> None:
    """The value half of the same property, kept from the behaviour that was here."""
    secret = "value-sentinel-9f3e1a"
    raw = f"before {secret} after\n".encode()
    for offset in range(len(raw) + 1):
        redactor = builtin._PipeRedactor([secret])
        published = redactor.feed(raw[:offset]) + redactor.feed(raw[offset:])
        published += redactor.feed(b"", final=True)
        assert secret not in published.decode(), f"value leaked when split at {offset}"


def test_a_line_with_no_terminator_is_released_and_stays_bounded() -> None:
    """A 10 MB single line neither stalls nor grows the held buffer.

    Asserted on the BOUND rather than the wall clock: the filter's memory must
    not be a function of what the child prints, and a timing assertion here
    would be a laptop-calibrated ceiling on a shared runner (see AGENTS.md,
    "Calibrate ceilings from CI"). The structural assertion is the one that
    holds anywhere: nothing is retained past the cap, and the bytes still come
    out.
    """
    line = ("x" * 1024) + f"MONGO_DSN={SENTINEL} " + ("y" * (10 * 1024 * 1024))
    redactor = builtin._PipeRedactor([])
    published = 0
    limit = builtin._PIPE_DEFERRAL_LIMIT
    for start in range(0, len(line), 64 * 1024):
        chunk = line[start : start + 64 * 1024]
        published += len(redactor.feed(chunk.encode()))
        assert (
            len(redactor.pending) <= limit + 4096
        ), "the undecided buffer must not grow with the child's output"
    published += len(redactor.feed(b"", final=True))
    assert redactor.pending == ""
    assert published >= len(line) - 4 * limit, "the line must actually be published"


def test_the_pipe_holds_a_partial_line_and_releases_it_at_the_terminator() -> None:
    """The documented trade, asserted rather than implied: a partial line waits."""
    redactor = builtin._PipeRedactor([])
    assert redactor.feed(b"partial line") == b""
    assert redactor.feed(b" continues\n") == b"partial line continues\n"


def test_the_pipe_releases_at_a_carriage_return_too() -> None:
    """A progress bar rewrites its line with ``\\r`` and may not emit ``\\n``.

    Without this the live view of a long build's progress would be withheld
    until the command finished, which is the regression the filter's whole-line
    rule could otherwise introduce.
    """
    redactor = builtin._PipeRedactor([])
    assert redactor.feed(b"step 1/3\r") == b"step 1/3\r"


# --- the store: containment, and the incident path ---------------------------


def test_a_matched_shape_value_is_registered_for_the_rest_of_the_session() -> None:
    """Containment: the credential is masked even in a form the table has no rule for.

    The DSN's password is matched once by a shape; registering it means the
    exact-value pass (which runs first) catches the same secret quoted alone or
    concatenated into something else later in the session.
    """
    store = VariableStore(cwd=".")
    assert store.redact(SENTINEL) != SENTINEL
    assert "sh4pedSentinelPw" in store.redaction_values()
    assert store.redact("the password is sh4pedSentinelPw, plainly") == (
        f"the password is {REDACTION_MARKER}, plainly"
    )


def test_a_registered_shape_value_is_never_readable() -> None:
    """Registering for SCRUBBING must not make a value readable or injectable."""
    store = VariableStore(cwd=".")
    store.redact(SENTINEL)
    assert "sh4pedSentinelPw" not in store.credential_names()
    assert "sh4pedSentinelPw" not in store.credential_env().values()
    with pytest.raises(KeyError):
        store.read("sh4pedSentinelPw")


def test_the_session_hook_reports_the_shapes_that_fired() -> None:
    session = _session()
    session._pending_shape_incidents.clear()
    session._reported_shape_incidents.clear()
    text = session._redact_tool_result_text(SENTINEL)
    assert "sh4pedSentinelPw" not in text
    assert [labels for _tool, labels, _summary in session._pending_shape_incidents] == [
        ["dsn-password"]
    ]


def test_the_incident_names_the_tool_and_carries_no_value() -> None:
    """The notice must be actionable and must not be the next place the value lands."""
    from local_operator.incidents import format_shape_incident_message

    text = format_shape_incident_message("bash", ["dsn-password"], "kubectl exec api -- env")
    assert "bash" in text
    assert "dsn-password" in text
    assert "rotate" in text
    assert "sh4pedSentinelPw" not in text


def test_one_incident_per_tool_and_shape_set() -> None:
    """A command that echoes the same credential ten times is one fact."""
    session = _session()
    session._pending_shape_incidents.clear()
    session._reported_shape_incidents.clear()
    for _ in range(5):
        session._redact_tool_result_text(SENTINEL)
    assert len(session._pending_shape_incidents) == 1


@pytest.mark.asyncio
async def test_the_queued_incident_reaches_the_transcript(tmp_path: Path) -> None:
    """Flushed at the boundary, and PERSISTED: a resumed session still knows.

    Persisted rather than live-only because what it records — a credential
    reached a tool result and must be rotated — is still true tomorrow, which is
    the opposite of the MCP-recovery record's reason for not persisting.
    """
    session = Session(
        model=ModelSpec(provider="test", model_id="unit-model", context_window=1000),
        stream_fn=_never_streams,
        tools=[],
        transcript=Transcript(tmp_path / "incident"),
        system_blocks_provider=lambda *_a: [],
        yolo=True,
        cwd=str(tmp_path),
        variables=VariableStore(cwd=str(tmp_path)),
    )
    session._redact_tool_result_text(SENTINEL)
    await session._flush_shape_incidents()

    body = (tmp_path / "incident" / "transcript.jsonl").read_text()
    assert "session_incident" in body
    assert "sh4pedSentinelPw" not in body
    assert "dsn-password" in body


def test_the_tool_identity_travels_with_the_redaction() -> None:
    """The hook is called with text alone; the identity rides the contextvar."""
    assert current_tool_source() == ("", "")
    with tool_source("bash", {"command": "kubectl exec api -- env"}):
        assert current_tool_source() == ("bash", "kubectl exec api -- env")
    assert current_tool_source() == ("", "")


def test_the_argument_summary_is_bounded_and_scrubbed() -> None:
    """The summary is stored in a transcript, so it cannot carry a credential."""
    summary = summarize_arguments({"command": f"mysql -phunter2hunter2 {SENTINEL}"})
    assert "sh4pedSentinelPw" not in summary
    assert len(summarize_arguments({"command": "x" * 5000})) <= 200
    assert summarize_arguments(None) == ""
    assert summarize_arguments({"unknown": 1}) == '{"unknown": 1}'


# --- journaled / replayed tool-call arguments --------------------------------


def test_the_executed_command_is_unchanged_while_the_journaled_copy_is_scrubbed() -> None:
    """Both halves of the seam, and this is the pair that has to hold together.

    Scrubbing the arguments the executor reads would change the command; leaving
    them alone in the copy history persists and replays is what put a credential
    in the transcript even though the RESULT of the call was masked.
    """
    from local_operator.harness.loop import _scrub_history_arguments

    command = "mysql -u root -pSuperSecret1 -e 'select 1'"
    original = Message(
        role="assistant",
        content=[TextContent(text="")],
        tool_calls=[ToolCall(id="c1", name="bash", arguments={"command": command})],
    )
    stored = _scrub_history_arguments(original, scrub_secrets)

    assert isinstance(stored, Message)
    assert original.tool_calls[0].arguments["command"] == command
    assert stored.tool_calls[0].arguments["command"] != command
    assert "SuperSecret1" not in stored.tool_calls[0].arguments["command"]
    assert stored is not original


def test_the_history_copy_is_the_same_object_when_nothing_changed() -> None:
    """Identity is load-bearing: the loop retracts a refused turn by identity."""
    from local_operator.harness.loop import _scrub_history_arguments

    message = Message(
        role="assistant",
        content=[TextContent(text="")],
        tool_calls=[ToolCall(id="c1", name="bash", arguments={"command": "git status"})],
    )
    assert _scrub_history_arguments(message, scrub_secrets) is message
    assert _scrub_history_arguments(message, None) is message


def test_nested_argument_values_are_scrubbed_too() -> None:
    from local_operator.harness.loop import _scrub_history_arguments

    message = Message(
        role="assistant",
        content=[TextContent(text="")],
        tool_calls=[
            ToolCall(
                id="c1",
                name="bash",
                arguments={"command": "ok", "env": {"DSN": SENTINEL}, "list": [SENTINEL]},
                raw_arguments=json.dumps({"command": SENTINEL}),
            )
        ],
    )
    stored = _scrub_history_arguments(message, scrub_secrets)
    assert isinstance(stored, Message)
    assert "sh4pedSentinelPw" not in json.dumps(stored.model_dump(mode="json"))


def test_both_assistant_append_sites_route_through_the_history_copy() -> None:
    """A pin on the plumbing, because behaviour tests cannot see an append site
    the loop happens not to reach in a given scenario.

    Both sites that put an assistant turn into ``context.messages`` must build
    the stored copy first; a third one added without it is the regression this
    names.
    """
    source = (Path(__file__).resolve().parents[3] / "local_operator/harness/loop.py").read_text()
    appends = re.findall(
        r"context\.messages\.append\((\w+)\)\n\s*new_messages\.append\(\1\)", source
    )
    # The other pairs in that file are the loop's own notices; what must not
    # exist is an assistant turn appended RAW.
    assert "assistant" not in appends, appends
    assert appends.count("stored") == 2, appends


# --- the credential-printing CLI detector ------------------------------------


@pytest.mark.parametrize("case", DUMP_COMMAND_CASES, ids=lambda c: c.command[:44])
def test_the_dump_detector_matches_the_decision_table(case: Any) -> None:
    notice = credential_dump_notice(case.command)
    if case.fires:
        assert notice is not None, f"should have fired: {case.reason}"
        assert case.command.split()[0] not in ("",)
        assert "credential guard" in notice
    else:
        assert notice is None, f"should not have fired ({case.reason}): {notice}"


def test_the_notice_suggests_a_safer_form_without_repeating_the_command() -> None:
    notice = credential_dump_notice("kubectl exec -n backend-services api-0 -- env")
    assert notice is not None
    assert notice.count("\n") == 0, "the notice is ONE line in the result"
    assert "cut -d= -f1" in notice or "lop secret get" in notice


def test_the_notice_carries_no_value_from_the_command() -> None:
    notice = credential_dump_notice("env | grep MONGO_DSN")
    assert notice is not None
    assert "MONGO_DSN" not in notice.replace("NAME", "")


@pytest.mark.asyncio
async def test_a_credential_dumping_bash_result_carries_the_notice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The notice on the real tool result, which is what the model reads."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    context = ToolContext(
        cwd=str(tmp_path), variables=VariableStore(cwd=str(tmp_path)), session_id="s"
    )
    result = await builtin.execute_bash(
        "bash-notice", {"command": "env"}, AbortSignal(), None, context
    )
    assert not result.is_error, result.text
    assert "credential guard" in result.text

    quiet = await builtin.execute_bash(
        "bash-quiet", {"command": "printenv PATH"}, AbortSignal(), None, context
    )
    assert "credential guard" not in quiet.text, "an ordinary command must not be nagged about"


# --- performance: no catastrophic backtracking -------------------------------


def _scrub_time(text: str) -> float:
    start = time.perf_counter()
    scrub_shapes(text)
    return time.perf_counter() - start


def test_the_shape_pass_is_linear_in_the_size_of_the_text() -> None:
    """A structural bound instead of a laptop-calibrated ceiling.

    The rules are context-anchored regexes, so the risk is a nested quantifier
    that backtracks exponentially on a near-miss (a long run of digits with no
    ``Bearer`` after it, say). An absolute millisecond budget measured here would
    be a number from a dev box and would flake on a loaded CI runner — AGENTS.md
    records three PRs lost to exactly that mistake. What holds anywhere is the
    RATIO: eight times the text must not cost far more than eight times the
    time. The 40x allowance is enormous on purpose: it catches exponential
    blowup, which is orders of magnitude, not a constant factor.
    """
    body = (
        "GET /v1/items?id=42 HTTP/1.1\n"
        "Bearer \n"
        "password: \n"
        "mongodb://host/db?\n"
        + "".join(
            f"line {index}: max_tokens=4096 api_key= value {index}\n" for index in range(2000)
        )
    )
    small = _scrub_time(body)
    large = _scrub_time(body * 8)
    # A floor keeps a sub-millisecond measurement from making the ratio noise.
    assert large <= max(small, 1e-4) * 40, f"shape pass looks super-linear: {small} -> {large}"


# --- subagents ---------------------------------------------------------------
#
# The defect this module closes was found in a SUBAGENT's tool result, so the
# child path is not an afterthought: it is where a remote host's environment
# reaches a transcript. A subagent is a real ``Session`` built in-process
# (``harness/subagent.run_subagent``) that inherits the parent's ``VariableStore``
# and therefore the parent's ``redact_tool_result`` hook. This drives the real
# launch path and asserts on what the child actually WROTE.


async def _wait_for(predicate: Any, timeout: float = 20.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("timed out waiting for the child run")
        await asyncio.sleep(0.01)


class _ChildStream:
    """Serves the child's one tool call, then its final answer."""

    def __init__(self, command: str) -> None:
        self.command = command
        self.requests: list[Any] = []

    def __call__(self, request: Any, signal: Any = None) -> Any:
        self.requests.append(request)
        turn = len(self.requests)

        async def gen() -> Any:
            if turn == 1:
                yield StreamToolCallDelta(
                    index=0,
                    id="call-1",
                    name="bash",
                    argument_delta=json.dumps({"command": self.command}),
                )
                yield StreamEndEvent(stop_reason="toolUse")
            else:
                yield StreamTextDelta(delta="child done")
                yield StreamEndEvent(stop_reason="stop")

        return gen()


@pytest.mark.asyncio
async def test_a_subagents_transcript_never_carries_the_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The child path, exercised end to end through the real launch machinery.

    ``run_subagent`` builds a child ``Session`` in-process with
    ``variables=parent_session._variables``, so the child's loop config is built
    from the same store — which is the claim this test checks by searching every
    transcript the child wrote rather than by reading that line of code.
    """
    from local_operator.tools.builtin import build_bash_tool

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    secret = SENTINEL
    child_stream = _ChildStream(f"echo {shlex.quote(f'MONGO_DSN={secret}')}")
    parent = Session(
        model=ModelSpec(provider="test", model_id="unit-model", context_window=10_000),
        stream_fn=child_stream,
        tools=[build_bash_tool()],
        transcript=Transcript(tmp_path / "parent"),
        system_blocks_provider=lambda: ["stable"],
        yolo=True,
        cwd=str(tmp_path),
        variables=VariableStore(cwd=str(tmp_path)),
    )

    job_id = parent._launch_subagent(label="sub", prompt="print the environment")
    assert isinstance(job_id, str) and job_id
    await _wait_for(lambda: (parent.jobs.get(job_id) or None) is not None)
    job = parent.jobs.get(job_id)
    await _wait_for(lambda: job is not None and job.status == "completed")

    transcripts = sorted(tmp_path.rglob("transcript.jsonl"))
    assert transcripts, "the child wrote no transcript — the harness is not exercising it"
    bodies = {path: path.read_text() for path in transcripts}
    for path, body in bodies.items():
        assert "sh4pedSentinelPw" not in body, f"the credential reached {path}"
    assert any(
        "session_incident" in body for body in bodies.values()
    ), "the child's redaction was never reported"
    # And the command really ran: the masked value is what the child wrote.
    assert any(REDACTION_MARKER in body for body in bodies.values())
    await parent.dispose()
