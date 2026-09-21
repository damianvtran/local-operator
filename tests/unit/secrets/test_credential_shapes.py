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
import hashlib
import json
import re
import shlex
import tempfile
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Callable, cast

import pytest

from local_operator import redaction_shapes
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
    scrub_shapes_with_hits,
)
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.tools import builtin
from local_operator.variables import VariableStore
from tests.unit.secrets.credential_shape_corpus import (
    DUMP_COMMAND_CASES,
    NEGATIVE_CASES,
    POSITIVE_CASES,
    Case,
)

#: A synthetic credential. Never a real one, and never asserted into anything a
#: human reads: the corpus carries the shapes, this carries the containment.
SENTINEL = "mongodb+srv://svc_user:sh4pedSentinelPw@db.internal/app"


#: The credential INSIDE ``SENTINEL`` — the part a mask has to remove. Derived
#: from the sentinel rather than written again, so no test in this file has to
#: spell a credential-shaped literal of its own.
SENTINEL_FRAGMENT = SENTINEL.split(":", 1)[1].rsplit("@", 1)[0]


def _exposed_text() -> str:
    """The corpus's own ESCALATING case: a DSN with the username as its password.

    The pair of fixtures this file needs is "masked whole" (``SENTINEL``) and
    "readable material left behind", and only the corpus knows which spellings
    grade as the second: the DSN rule deliberately keeps the userinfo username, so
    ``amqp://guest:guest@…`` reads its own password back and the classification
    escalates. Derived rather than written out, so this file still spells no
    credential-shaped literal of its own, and so the day the corpus changes its
    mind the tests follow it instead of disagreeing with it.
    """
    return next(case.text for case in POSITIVE_CASES if case.reason == "amqp DSN")


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


def _session(tmp: Path | None = None) -> Session:
    """A real Session, built the way ``session_factory`` builds one.

    FRESH per call rather than cached, and that is not merely tidy: the store
    CONTAINS — a value the shape pass matched is registered for the rest of the
    session and masked in every later result — so one cached session let a corpus
    case mask another case's text, and the incident dedupe keyed on
    ``(tool, labels)`` was reachable from a neighbouring test. Isolated cases are
    what makes a per-surface parametrisation meaningful.
    """
    cwd = tmp or Path(tempfile.mkdtemp(prefix="shape-test-"))
    return Session(
        model=ModelSpec(provider="test", model_id="unit-model", context_window=1000),
        stream_fn=_never_streams,
        tools=[],
        transcript=Transcript(cwd / "session"),
        system_blocks_provider=lambda *_a: [],
        yolo=True,
        cwd=str(cwd),
        variables=VariableStore(cwd=str(cwd)),
    )


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


def test_a_shape_that_cannot_render_a_mask_is_rejected_at_import() -> None:
    """The table's one unacceptable state, refused where it would be written.

    A shape with neither a replacement nor a guard matches text and leaves it
    alone — it would DETECT a credential and publish it, which is exactly the
    failure this module exists to prevent and the one a reviewer cannot see by
    reading a diff. It is therefore an import-time error, not a runtime
    surprise on whichever result happens to match first.
    """
    from local_operator.redaction_shapes import Shape

    with pytest.raises(ValueError, match="neither a replacement nor a guard"):
        Shape("publishes-everything", re.compile(r"x"), None, None, None)
    with pytest.raises(ValueError, match="both a replacement and a guard"):
        Shape("dead-replacement", re.compile(r"x"), "y", None, lambda m: True)
    with pytest.raises(ValueError, match="names no secret_group"):
        Shape("guarded-without-group", re.compile(r"x"), None, None, lambda m: True)


def test_every_guard_rendered_shape_masks_its_credential_and_keeps_the_rest() -> None:
    """A guarded rule's mask comes from the group, so the readable parts survive.

    Asserted per guarded rule rather than once: the two rules keep different
    things (an assignment keeps its name and separator, a URL-valued name keeps
    the name and the password's user/host), and a future guarded rule that
    replaced the whole match would take back text the operator needs to read.
    """
    from local_operator.redaction_shapes import CREDENTIAL_SHAPES

    guarded = [shape for shape in CREDENTIAL_SHAPES if shape.guard is not None]
    assert {shape.label for shape in guarded} == {
        "credential-assignment",
        "credential-url-value",
        "cli-credential-flag",
        "vendor-prefixed-token",
        "gcp-service-account-value",
        "gcp-service-account-value-open",
        "authorization-basic-bare",
    }

    assignment = scrub_shapes("AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG")
    assert assignment == f"AWS_SECRET_ACCESS_KEY={REDACTION_MARKER}"

    dsn = scrub_shapes(SENTINEL)
    assert "svc_user" in dsn, "the user must stay readable"
    assert "db.internal" in dsn, "the host must stay readable"
    assert "sh4pedSentinelPw" not in dsn
    assert dsn.startswith("mongodb+srv://svc_user:")


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


def test_a_shape_straddling_the_deferral_boundary_is_masked_and_registered() -> None:
    """A cap-forced cut must not publish the two halves of a credential.

    The release point has always refused to cut through a KNOWN value; the same
    rule was missing for a SHAPE, and the cap is the one release chosen without
    regard to the text around it. A DSN sitting on that boundary came out as an
    unmasked head in one released slice and an unmasked tail in the next —
    neither half carries the spelling the pattern needs — and nothing later can
    repair it, because the live stream is the one surface no pass re-reads.

    Swept rather than sampled: with 4 KiB reads the cap puts a slice boundary
    every 4096 bytes, so the offsets below walk the credential across it. The
    real store is the sink, because the second half of the property is that a
    value the pipe masks is CONTAINED — masked later in a form no rule knows.
    """
    store = VariableStore(cwd=".")
    # The mask removes the whole userinfo, but the value the shape REGISTERS is the
    # password group — so the reuse below spells only that, which is the form no
    # rule in the table has a spelling for.
    password = SENTINEL_FRAGMENT.rsplit(":", 1)[-1]
    for offset in range(4080, 4112):
        body = "." * offset + SENTINEL + "." * 20000
        redactor = builtin._PipeRedactor([], contain=store.register_shape_hits_for_containment)
        raw = body.encode()
        published = [redactor.feed(raw[i : i + 4096]) for i in range(0, len(raw), 4096)]
        published.append(redactor.feed(b"", final=True))
        text = b"".join(published).decode()
        assert SENTINEL_FRAGMENT not in text, f"the password was published, cut at {offset}"
        assert REDACTION_MARKER in text, f"the shape was not masked, cut at {offset}"
    assert store.redact(f"prefix {password} suffix") == (
        f"prefix {REDACTION_MARKER} suffix"
    ), "the value the pipe masked was not registered for containment"


def test_no_chunk_boundary_publishes_what_one_pass_would_mask() -> None:
    """The invariant the boundary fix exists for, stated as the property itself.

    Whatever the filter publishes, concatenated, must be what a single pass over
    the same bytes produces. That is stronger than "the password is absent": it
    also fails if a fix buys the mask by dropping, duplicating or reordering
    output, which is the way this class of change usually goes wrong.

    The filler is a character a DSN spelling follows: a WORD character runs into
    the scheme and defeats the pattern's leading ``\\b``, so the table matches
    neither release and the property is vacuous there. That spelling's own
    behaviour — a slice boundary CREATING the boundary word the one-piece text
    lacks, so the streamed pass masks MORE than the single pass — is pre-existing,
    identical before and after this change, and in the safe direction; it is not
    this fix's to move.
    """
    for offset in (4090, 4094, 4095, 4096, 4097, 8190, 8191, 8192):
        body = "." * offset + SENTINEL + "." * 20000
        for chunk in (7, 4096, 8192, 8193):
            redactor = builtin._PipeRedactor([])
            raw = body.encode()
            published = [redactor.feed(raw[i : i + chunk]) for i in range(0, len(raw), chunk)]
            published.append(redactor.feed(b"", final=True))
            assert b"".join(published).decode() == scrub_secrets(body), (
                f"streamed output diverged from the one-piece pass at offset {offset}, "
                f"chunk {chunk}"
            )


def test_the_boundary_hold_is_bounded_and_still_streams() -> None:
    """The fix holds bytes back, so the hold is asserted rather than implied.

    Moving a cut out of a shape means publishing less per read, and the whole
    reason the cap exists is that the held buffer must not be a function of what
    the child prints. Both halves are asserted on the STRUCTURE (a peak and a
    bound), never on a wall clock — see AGENTS.md, "Calibrate ceilings from CI".
    The peak assertion is what keeps this test honest: it fails if the hold stops
    being exercised, which is how the bound would go stale without anyone
    noticing.
    """
    line = "." * 4095 + SENTINEL + "." * (4 * 1024 * 1024)
    redactor = builtin._PipeRedactor([])
    peak = 0
    published = 0
    for start in range(0, len(line), 4096):
        published += len(redactor.feed(line[start : start + 4096].encode()))
        peak = max(peak, len(redactor.pending))
        assert (
            len(redactor.pending) <= builtin._PIPE_HOLD_LIMIT
        ), "the hold must not grow with the child's output"
    published += len(redactor.feed(b"", final=True))
    assert redactor.pending == ""
    assert peak > builtin._PIPE_DEFERRAL_LIMIT, "the boundary hold was never exercised"
    assert published >= len(line) - builtin._PIPE_HOLD_LIMIT, "the line must still stream"


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


def test_a_contained_hit_is_not_an_incident_and_is_still_contained() -> None:
    """The operator's rule, both halves, in one test because they are one pair.

    "As long as something wasn't actually leaked to the transcript we shouldn't get
    a session incident indicated anywhere": a value masked WHOLE files nothing at
    all. The second claim is the one that must NOT move with it — the hit is still
    registered for containment, so the same secret in a spelling the table has no
    rule for is masked later in the session. The incident is the INDICATOR; the
    registration is the PROTECTION, and only the indicator was asked to go.
    """
    session = _session()
    session._pending_shape_incidents.clear()
    session._reported_shape_incidents.clear()
    text = session._redact_tool_result_text(SENTINEL)
    assert SENTINEL_FRAGMENT not in text, "the value stopped being masked"
    assert session._pending_shape_incidents == [], "a contained hit filed an incident"
    # The protection, exercised through the shipped hook rather than the store: the
    # value comes back in a form the shape table has no rule for.
    later = session._redact_tool_result_text(f"prefix {SENTINEL_FRAGMENT} suffix")
    assert SENTINEL_FRAGMENT not in later, "the masked value was not registered"


def test_the_incident_names_the_tool_and_carries_no_value() -> None:
    """The notice must be actionable and must not be the next place the value lands."""
    from local_operator.incidents import format_shape_incident_message

    text = format_shape_incident_message("bash", ["dsn-password"], "kubectl exec api -- env")
    assert "bash" in text
    assert "dsn-password" in text
    assert "rotate" in text
    assert "sh4pedSentinelPw" not in text


def test_an_exposed_hit_still_files_one_incident_per_tool_and_shape_set() -> None:
    """A command that echoes the same credential ten times is one fact.

    Driven with the EXPOSED case, which is the only one that reaches the queue now:
    the dedupe has to keep working for the event that is still filed.
    """
    session = _session()
    session._pending_shape_incidents.clear()
    session._reported_shape_incidents.clear()
    for _ in range(5):
        session._redact_tool_result_text(_exposed_text())
    assert [flag for _t, _l, _s, flag in session._pending_shape_incidents] == [True]


@pytest.mark.asyncio
async def test_the_queued_incident_reaches_the_transcript(tmp_path: Path) -> None:
    """Flushed at the boundary, and PERSISTED: a resumed session still knows.

    Persisted rather than live-only because what it records is still true
    tomorrow, which is the opposite of the MCP-recovery record's reason for not
    persisting. Driven with the EXPOSED case, which is the only one that files
    now: a contained hit has nothing to persist, and its own end-to-end absence
    test is below.
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
    session._redact_tool_result_text(_exposed_text())
    await session._flush_shape_incidents()

    body = (tmp_path / "incident" / "transcript.jsonl").read_text()
    assert "session_incident" in body
    assert "rotate" in body
    assert REDACTION_MARKER not in body


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


# --- the credential-printing detector: one bounded window, no unbounded run ------
#
# The dump table is a SECOND pattern table, on the same hot path, and the file rule
# in it was quadratic in the length of the line: an unbounded lazy gap followed by an
# unbounded ``[^\s]*`` makes the engine walk the rest of the line at every gap length.
# Measured on ``"head " + "x" * 140000``: 114.4 s of CPU for ONE ``search``, against
# 0.0009 s once both runs are bounded. Real commands reach 30,849 characters (p99
# 4.9 KB over 39,111 harvested commands), so the blowup was latent rather than live —
# and latent is not safe: the same class of unbounded run had frozen six sessions on
# this fleet hours earlier. The three tests below pin the cost three ways, none of
# them a stopwatch reading taken on this machine.


def _dump_cpu(command: str) -> float:
    """CPU seconds for one detection, best of three samples.

    ``process_time`` and not wall time, and the minimum of three and not one
    sample, for the reasons AGENTS.md records under "If you must measure, measure
    CPU, not wall time": a wall reading on this host conflates "the engine worked"
    with "the OS did not schedule this process" (measured there: 525-668 ms of
    apparent stall with the loop idle), and the minimum is the sample that saw the
    least of it. A single wall sample of a sub-millisecond call is noise — this test
    WAS written that way first and failed once in a full-tier run while passing
    alone, which is the flake the docstring above warns about.
    """
    best = float("inf")
    for _ in range(3):
        start = time.process_time()
        credential_dump_notice(command)
        best = min(best, time.process_time() - start)
    return best


#: Written to be the worst case for the OLD shape: a reading verb, then a long
#: separator-free run, and NO match in it, so every gap length and every path length
#: has to be tried before the search gives up. 8,000 characters is the size that
#: makes the arithmetic work in both directions: the pre-fix rule needs 0.58 s here
#: and 37 s at eight times the text (quadratic), while the fixed rule needs 14 ms and
#: 110 ms (linear) — so the 40x allowance sits between two numbers it cannot confuse,
#: and both are far above the clock's noise floor.
_PATHOLOGICAL_LINE = "head " * 1_600


def test_no_dump_rule_may_walk_a_long_line_super_linearly() -> None:
    """The RATIO, for the reason the shape pass's twin test carries.

    An absolute millisecond budget here would be a number off this host, and this
    host runs ~25 sibling sessions (AGENTS.md records three PRs lost to exactly that
    mistake). What holds anywhere is the ratio: eight times the text must not cost
    eight times the time per character. The 40x allowance is enormous on purpose —
    the regression it catches is quadratic, which is 64x at this size, and a constant
    factor is not it.
    """
    small = _dump_cpu(_PATHOLOGICAL_LINE)
    large = _dump_cpu(_PATHOLOGICAL_LINE * 8)
    assert large <= max(small, 1e-4) * 40, f"the dump rule looks super-linear: {small} -> {large}"


def test_the_file_rule_may_not_look_beyond_a_bounded_window() -> None:
    """The gap bound, pinned from both sides as behaviour rather than as constants.

    A credential filename has to be an argument OF the reading verb, which is what
    bounds the rule: ~90 characters out still fires (a path built through a loop
    variable, which is how one real harness command spelled it), while 5,000 does
    not. The far probe is the one the OLD rule failed — with an unbounded gap it
    matched a bare ``.pem`` five thousand characters away from the verb.
    """
    near = "head -c 200 " + "x" * 80 + "/key.pem"
    far = "head -c 200 " + "x" * 5_000 + "/key.pem"
    assert credential_dump_notice(near) is not None, "the window is too tight to be usable"
    assert credential_dump_notice(far) is None, "the rule walked past its own window"


def test_the_file_rule_is_written_without_an_unbounded_run() -> None:
    r"""A STRUCTURAL pin on the cost, independent of any measurement.

    The rule's own source is asserted to contain no unbounded quantifier, so the
    property cannot come back by an edit that looks innocent and measures fine on a
    short line: ``[^\n]*?`` and ``[^\s]*`` are exactly what made the search
    quadratic, and both are fixed-width now. Every other rule in the table is checked
    for the same reason it is cheap — its unbounded gap is followed by alternations
    that are literal-anchored, so a failing position costs a constant — which is why
    this pin is per-rule rather than over the table.
    """
    from local_operator.redaction_shapes import DUMP_SHAPES

    rule = next(shape for shape in DUMP_SHAPES if shape.label == "credential-file-read")
    source = rule.pattern.pattern
    assert "*" not in source, f"an unbounded ``*`` run is back: {source}"
    assert "+" not in source, f"an unbounded ``+`` run is back: {source}"
    assert "{" in source, "the bounds were removed rather than expressed"


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
    # And the command really ran: the masked value is what the child wrote. This is
    # the liveness evidence now that a contained hit files nothing at all — an
    # absence assertion alone could not tell "quiet" from "never exercised".
    assert any(REDACTION_MARKER in body for body in bodies.values())
    assert not any(
        "session_incident" in body for body in bodies.values()
    ), "a contained hit was reported"
    await parent.dispose()


# --- the review round that this file's guarantees are now pinned to ----------


def test_every_rule_that_fires_trips_the_gate() -> None:
    """Derive the gate requirement from the table, so a rule cannot lose its anchor.

    ``_SHAPE_ANCHORS`` gates the whole pass: text carrying none of the anchors
    skips the table entirely. A rule whose spelling has no anchor is therefore a
    rule that fires in ``_run_shapes`` and is skipped in the shipped path — which
    is exactly what happened to the `pk-`, `rk-`, `hf-` and `npm-` spellings of
    ``vendor-prefixed-token``, and the corpus could not see it because it had no
    ``-`` variant of those four prefixes.

    This test asks the question directly, per rule, using the corpus as the
    source of spellings: for every case a rule actually rewrites, the gate must
    be open for that case. A new prefix, a new alternation arm or a new rule
    without an anchor fails HERE.
    """
    from local_operator.redaction_shapes import CREDENTIAL_SHAPES, has_shape_anchor

    missing: dict[str, str] = {}
    for shape in CREDENTIAL_SHAPES:
        for case in POSITIVE_CASES:
            if _run_one_shape(shape, case.text) != case.text and not has_shape_anchor(case.text):
                missing.setdefault(shape.label, case.text)
    assert not missing, f"rules that fire on a case the gate skips: {missing}"


def _run_one_shape(shape: Any, text: str) -> str:
    from local_operator.redaction_shapes import _run_shapes

    return _run_shapes((shape,), text, [])


def test_every_notice_fits_one_narrow_card_row() -> None:
    """The advisory has to be READABLE, not merely correct.

    The notice lands in a result the operator reads through the tool card, which
    renders one row per line and ellipsises at the row width — 92 cells at a
    100-column frame. Measured before this budget existed: every one of the
    sixteen rules produced 120-282 cells, so the actionable half (the safer
    form, which is the point of the line) was behind an ellipsis at every width
    a terminal is used at. 88 leaves the card's own padding room.
    """
    from local_operator.redaction_shapes import DUMP_SHAPES

    probes = {
        "environment-dump": "env | sort",
        "named-variable-dump": "printenv MONGO_DSN",
        "kubectl-exec-env": "kubectl exec x -- env",
        "kubectl-secret-read": "kubectl get secret n -o yaml",
        "docker-inspect": "docker inspect api",
        "docker-exec-env": "docker exec x env",
        "docker-compose-config": "docker compose config",
        "aws-credentials-read": "aws secretsmanager get-secret-value",
        "gcloud-token": "gcloud auth print-access-token",
        "github-auth-token": "gh auth token",
        "gitlab-auth-token": "glab auth status -t",
        "vault-read": "vault kv get secret/x",
        "heroku-config": "heroku config",
        "terraform-output": "terraform output",
        "npm-token-list": "npm token list",
        "credential-file-read": "cat ~/.netrc",
    }
    assert set(probes) == {shape.label for shape in DUMP_SHAPES}, "a rule has no probe"
    for label, command in probes.items():
        notice = credential_dump_notice(command)
        assert notice, f"{label} did not fire on its own probe"
        assert len(notice) <= 88, f"{label} notice is {len(notice)} cells: {notice}"
        assert "`" in notice, f"{label} offers no copy-pasteable form"


def test_the_advisory_survives_a_long_result_and_is_not_last() -> None:
    """The notice must not be the first thing the 40-line head crop drops."""
    from local_operator.tools import builtin

    # Read the MODULE, not ``inspect.getsource(execute_bash)``: the tool is
    # wrapped by a decorator, so getsource returns the wrapper and the assertion
    # sees none of the body it is meant to pin.
    source = Path(builtin.__file__).read_text()
    assert "parts.insert(1, notice)" in source, "the advisory went back to the tail"


def test_the_live_pending_text_matches_the_card() -> None:
    """The tool says "no output yet"; the card must say the same words.

    The live card asserted the SETTLED ``(empty)`` while bytes were arriving and
    the pipe was withholding them. The card owns the open-state wording, and the
    tool layer cannot import the TUI layer, so the two constants are kept in step
    here rather than by comment.
    """
    from local_operator.tools import builtin
    from local_operator.tui.widgets.tool_card import LIVE_HEADER_PENDING

    assert builtin._LIVE_PENDING_TEXT == LIVE_HEADER_PENDING


def test_a_hit_is_graded_before_it_is_announced() -> None:
    """Never announce a masking that did not happen.

    The containment row tells the operator a credential "was masked before you saw
    it", and the operator acts on that by NOT cleaning the copy up. A false
    all-clear is therefore worse than silence. This was live: a DSN password
    containing ``@`` was masked to the first ``@`` while the row promised the whole
    thing was gone.

    Under the current policy the surviving hit is not silent either: it is the one
    case that escalates (readable material in the model's context), while the
    withheld-claim-but-contained case — a truncated PEM — announces nothing at all.
    """
    from local_operator.redaction_shapes import ShapeHit, _only_fully_masked

    surviving = ShapeHit(
        label="credential-assignment", value="hunter2hunter2", window="hunter2hunter2"
    )
    kept = _only_fully_masked([surviving], "PASSWORD=hunter2hunter2")
    # KEPT for containment, flagged out of the notice: suppressing the claim is
    # the point, and refusing to register the value would give up the protection
    # the hit is worth (round 2, N2-4).
    assert len(kept) == 1 and kept[0].complete is False
    masked = ShapeHit(
        label="credential-assignment", value="hunter2hunter2", window="hunter2hunter2"
    )
    assert _only_fully_masked([masked], "PASSWORD=[redacted]") == [masked]

    # ...and the Q1 shape, end to end: a hit is reported ONLY because the whole
    # password is gone. No fragment of the value may survive the mask.
    scrubbed, hits = scrub_shapes_with_hits("MONGO_DSN=mongodb+srv://svc:p@ssw0rd@db.invalid/x")
    assert [hit.value for hit in hits] == ["p@ssw0rd"]
    for fragment in ("p@ssw0rd", "ssw0rd"):
        assert fragment not in scrubbed, scrubbed


@pytest.mark.parametrize(
    ("reached_model", "marker"),
    [(True, "rotate"), (False, "no exposure")],
    ids=["reached-the-model", "contained"],
)
@pytest.mark.asyncio
async def test_the_incident_row_reaches_the_operator_live_and_on_replay(
    tmp_path: Path, reached_model: bool, marker: str
) -> None:
    """Both halves of "the operator sees it", which is the row's whole purpose.

    Live: ``journal_shape_incident`` must emit a receipt an attached TUI paints.
    Replay: the fold must have a branch for the record, or a resumed session
    shows nothing (a custom message falls through every role-based branch).
    Measured before this wiring: the row reached the model and painted nowhere.

    Parametrised over the CLASSIFICATION, because the contained text is a second
    new string on the same path and the failure mode this test exists for — a row
    that paints nowhere — does not care which wording it is carrying.
    """
    from local_operator.harness.message_types import SESSION_INCIDENT_MESSAGE_TYPE
    from local_operator.harness.types import NoticeEvent

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
    events: list[Any] = []
    session.subscribe(events.append)
    await session.journal_shape_incident(
        "bash", ["dsn-password"], "kubectl exec api -- env", reached_model=reached_model
    )

    notices = [event for event in events if isinstance(event, NoticeEvent)]
    assert notices, "the incident emitted no live receipt"
    assert notices[0].kind == "warning"
    assert marker in notices[0].text

    # Replay: the same record, folded through the real settlement path.
    rows = _fold_incident_row(SESSION_INCIDENT_MESSAGE_TYPE, notices[0].text)
    assert rows, "the incident row folded to nothing"
    assert any(marker in row for row in rows)


def _fold_incident_row(custom_type: str, text: str) -> list[str]:
    """Fold one incident row through ``project_settled_rows`` and read the rows.

    The fold reads a lot of bookkeeping off its target; this stand-in answers
    every attribute it has not been given explicitly and records the BLOCKS,
    which is the only thing the incident branch is responsible for. Anything
    that must be a real dict or set is set here, because ``dict(...)`` and ``in``
    over a ``MagicMock`` would raise or lie.
    """
    from unittest.mock import MagicMock

    from local_operator.harness.types import CustomMessage
    from local_operator.tui import session_presentation as presentation

    class _Target:
        def __init__(self) -> None:
            self.blocks: list[str] = []
            self._resume_results: dict[str, Any] = {}
            self._resume_mounted_ids: set[str] = set()
            self._live_wake_receipts: set[str] = set()
            self._live_peer_receipts: set[str] = set()
            self._replay_bang_pending: dict[str, Any] = {}
            self._block_sink: list[Any] = []

        def __getattr__(self, name: str) -> Any:
            value = MagicMock()
            setattr(self, name, value)
            return value

        def _append_block(self, block: Any, **_kwargs: Any) -> None:
            # ``NoticeBlock`` keeps its text on ``_text``; ``.text`` is a method.
            self.blocks.append(str(getattr(block, "_text", "") or type(block).__name__))

    target = _Target()
    message = CustomMessage(
        custom_type=custom_type,
        attribution="system",
        details={"text": text},
    )
    # The double answers the protocol dynamically (see ``__getattr__`` above), so
    # the cast is what tells the type checker what the runtime already knows.
    presentation.project_settled_rows(cast(Any, target), [message], fold_width=80)
    return target.blocks


#: The rules whose value class can still stop at a quote INJECTED INSIDE the token,
#: with the measured number of corpus cases that expose it. The invariant below is
#: the property this PR exists to hold — **a mask is all of the credential or none of
#: it, never a prefix with the remainder readable** — and it holds for the whole
#: table except these. They are FROZEN, not excused: a new rule that joins this set,
#: or a change that makes any of them worse, fails the test (`..._ratchet_...` below).
#:
#: Why each remaining rule is allowed to stand:
#:
#: * ``pem-private-key`` / ``gcp-service-account-key`` — a quote can only land inside a
#:   PEM dash-run or a base64 body in this measurement, and NEITHER ALPHABET CONTAINS
#:   ONE: a PEM header is dashes, spaces and capitals (`-----BEGIN RSA PRIVATE KEY-----`)
#:   and a PEM body is base64 (`A-Za-z0-9+/=`). A quote inside either cannot occur in
#:   real output, so these two counts are the synthetic case only — the proof, not a
#:   claim that the rule is right.
#: * ``credential-url-value`` — the matched value is a whole URL: the password inside it
#:   IS masked, and what survives is the host and path, which stay readable BY DESIGN
#:   (the operator needs to see which endpoint was called). Not secret material.
_PARTIAL_MASK_RESIDUAL = {
    # The two PEM/body classes are the SYNTHETIC case only, and the proof is the
    # same one: the quote is injected INSIDE the fixed marker phrase (`-----B'EGIN
    # RSA PRIVATE KEY-----`) or inside a PEM body, and NEITHER ALPHABET CONTAINS A
    # QUOTE — a PEM header is dashes, spaces and capitals, and a body is base64
    # (`A-Za-z0-9+/=`). A quote cannot occur there in real output, which is why
    # these counts are allowed to stand rather than fixed.
    # `credential-url-value` is NOT this case — its value class is a whole URL and
    # DOES contain a quote — and it has its own reason in the bullet above (the
    # password inside it IS masked; the host and path stay readable BY DESIGN).
    # Splitting the two was a correction: this comment used to claim all three
    # classes rested on the alphabet argument we could not make for the third
    # (agent review R2, finding 4).
    #
    # The numbers fall as real fixes land and must be updated in the SAME commit as
    # the fix that moves them (the ratchet asserts exact equality): `pem-private-key`
    # 152 → 89 and `gcp-service-account-key` 44 → 0 came from masking an anchored
    # value to its closing quote (M6-1), and `gcp-service-account-value` entered at 22
    # with that mask.
    # Re-measured after requiring the closing quote on the anchored rule: the counts
    # moved (pem 89 → 119, value 22 → 42) because a truncated value now falls through to
    # the line rules, which mask the marker and expose the same synthetic quote-in-marker
    # case. They still all fall in the "a quote cannot occur here in real output" class.
    "pem-private-key": 119,
    "gcp-service-account-value": 42,
    "credential-url-value": 10,
}


def _readable_fragments(value: str) -> set[str]:
    """Every six-character window of a credential — what must not survive a mask."""
    return {value[i : i + 6] for i in range(0, len(value) - 5)}


def _partial_masks(text: str) -> list[tuple[str, str, str]]:
    """Quote insertions inside a credential that leave a readable fragment.

    Two exclusions keep the measurement SOUND rather than merely loud: a fragment of
    the redaction marker itself is not a credential, and a fragment that is readable
    in the unmodified text was never the mask's to remove (the same credential can
    appear twice and only one occurrence may be in scope). Without them the sweep
    reported the marker's own letters and a URL's host as leaks.
    """
    import local_operator.redaction_shapes as rs

    baseline = rs.scrub_shapes(text)
    _, hits = rs.scrub_shapes_with_hits(text)
    found: list[tuple[str, str, str]] = []
    for hit in hits:
        value = hit.value
        if len(value) < 6 or rs.REDACTION_MARKER in value:
            continue
        already = {f for f in _readable_fragments(value) if f in baseline}
        for quote in ("'", '"'):
            for pos in range(1, len(value)):
                variant_value = value[:pos] + quote + value[pos:]
                variant = text.replace(value, variant_value, 1)
                masked = rs.scrub_shapes(variant)
                fragments = _readable_fragments(variant_value)
                surviving = {
                    f
                    for f in fragments
                    if f in masked and f not in already and rs.REDACTION_MARKER not in f
                }
                # A PROPER SUBSET surviving is the failure. All of it surviving means
                # the rule never saw the credential (legal); none of it is the
                # invariant holding.
                if surviving and len(surviving) < len(fragments):
                    found.append((hit.label, variant, masked))
    return found


@pytest.mark.parametrize("case", POSITIVE_CASES, ids=lambda c: c.reason[:48])
def test_a_quote_inside_a_credential_never_leaves_a_readable_fragment(case: Case) -> None:
    """THE INVARIANT: a mask covers the whole credential or none of it.

    Measured by inserting ``'`` and ``"`` at every position inside every credential
    the corpus masks, then checking that no 6-character fragment of the credential is
    readable afterwards. A masked prefix beside a readable tail is worse than no mask
    at all, because the notice tells the operator the credential was contained.

    The residual set above is the part of the table the closure does not reach yet;
    the assertion on it is what stops the class recurring silently — a rule that
    regresses, or a new one that joins the set, reds this test.
    """
    offenders = _partial_masks(case.text)
    labels = {label for label, _, _ in offenders}
    unexpected = labels - set(_PARTIAL_MASK_RESIDUAL)
    assert not unexpected, (
        f"{sorted(unexpected)} published a readable fragment of a credential whose "
        f"mask stopped at a quote; first: {offenders[0][1][:80]!r} -> "
        f"{offenders[0][2][:80]!r}"
    )


def test_the_partial_mask_ratchet_only_ever_tightens() -> None:
    """The frozen counts are an EXACT ceiling, in both directions.

    A number that RISES fails — a regression, or a rule that joined the class
    undeclared. A number that FALLS also fails, because a fall that nobody records
    is a fix nobody can see: it must come with the code change that caused it and an
    update to the frozen table in the same commit (red before, green after). That is
    what makes this a ratchet rather than a floor.

    SCOPE, stated because the measurement is corpus-scoped and not universal: this
    covers the cases in `POSITIVE_CASES` only. A rule the corpus does not exercise
    with a credential is not measured here, and the corpus is the specification for
    what must be masked — a case added to the corpus can only tighten this test.
    """
    measured: dict[str, int] = {}
    for case in POSITIVE_CASES:
        for label, _, _ in _partial_masks(case.text):
            measured[label] = measured.get(label, 0) + 1
    for label, count in measured.items():
        assert (
            label in _PARTIAL_MASK_RESIDUAL
        ), f"{label} joined the partial-mask class without being declared: {count} cases"
        assert count == _PARTIAL_MASK_RESIDUAL[label], (
            f"{label}: measured {count}, frozen {_PARTIAL_MASK_RESIDUAL[label]} — a rise "
            "is a regression, and a fall has to update the frozen number in the same "
            "commit as the fix that caused it"
        )
    for label, frozen in _PARTIAL_MASK_RESIDUAL.items():
        assert (
            measured.get(label, 0) == frozen
        ), f"{label} is frozen at {frozen} but now measures {measured.get(label, 0)}"


#: Every PEM spelling the corpus and real tooling use. Completeness (R5-2) and the
#: anchored-value mask (M6-1) both alternated silently all night because nothing in the
#: suite asserted `complete` at all — one direction for the claim, one for the hold.
_PEM_SPELLINGS = (
    "RSA PRIVATE KEY",
    "PRIVATE KEY",
    "OPENSSH PRIVATE KEY",
    "EC PRIVATE KEY",
    "DSA PRIVATE KEY",
    "ENCRYPTED PRIVATE KEY",
)
_PEM_BODY = "MIIEowIBAAKCAQEA1234abcd5678efgh"


@pytest.mark.parametrize("spelling", _PEM_SPELLINGS)
def test_a_complete_block_of_every_spelling_claims_its_mask(spelling: str) -> None:
    """The claim direction: a finished block must file `complete=True`.

    The literal `END PRIVATE KEY` matched PKCS#8 only, so RSA and OPENSSH — the
    commonest spellings in real tool output — withheld the rotation notice entirely
    (R5-2). Parametrised over the spellings so a future one cannot join them silently.
    """
    import local_operator.redaction_shapes as rs

    text = f"-----BEGIN {spelling}-----\n{_PEM_BODY}\n-----END {spelling}-----"
    scrubbed, hits = rs.scrub_shapes_with_hits(text)
    assert _PEM_BODY not in scrubbed, "the body was published"
    assert hits, "a complete block filed no hit"
    assert any(hit.complete for hit in hits), f"{spelling} withheld its completion claim"


@pytest.mark.parametrize("spelling", _PEM_SPELLINGS)
def test_a_truncated_block_of_every_spelling_claims_nothing(spelling: str) -> None:
    """The hold direction: a block with no END must never file a completion claim."""
    import local_operator.redaction_shapes as rs

    text = f'{{"private_key": "-----BEGIN {spelling}-----\\n{_PEM_BODY}\\n"}}'
    scrubbed, hits = rs.scrub_shapes_with_hits(text)
    assert _PEM_BODY not in scrubbed, "the body was published"
    assert hits, "a truncated block filed no hit"
    assert not any(hit.complete for hit in hits), f"{spelling} claimed a completed mask"


def test_an_anchored_block_with_a_same_line_continuation_masks_whole() -> None:
    """M6-1: a body line whose pad is followed by more text still masks the whole value.

    The line-bounded iteration failed its first step on `…efgh, note` (and on an ANSI
    reset), collapsed the alternation to the header, and published the entire body with
    nothing flagged complete. Anchored values are masked to the VALUE's closing quote,
    which is a real delimiter — the reason `[^\\r\\n]*` was the wrong remedy (it reopens
    B4-1's eaten anchor).
    """
    import local_operator.redaction_shapes as rs

    for continuation in (", note", "\x1b[0m", " and more"):
        text = (
            '{"private_key": "-----BEGIN RSA PRIVATE KEY-----\\n'
            + _PEM_BODY
            + continuation
            + '\\n"}'
        )
        scrubbed, hits = rs.scrub_shapes_with_hits(text)
        assert _PEM_BODY not in scrubbed, f"body published for continuation {continuation!r}"
        assert scrubbed == '{"private_key": "[redacted]"}', scrubbed
        assert not any(hit.complete for hit in hits)


@pytest.mark.parametrize("tail", ["NORMAL, more text", "INFO, starting", "done. next"])
def test_a_following_word_line_keeps_its_word(tail: str) -> None:
    """M6-2: a line that is a word plus punctuation must not lose its leading word."""
    import local_operator.redaction_shapes as rs

    text = f"-----BEGIN RSA PRIVATE KEY-----\n{_PEM_BODY}\n-----END RSA PRIVATE KEY-----\n{tail}"
    scrubbed = rs.scrub_shapes(text)
    assert scrubbed.splitlines()[-1] == tail, scrubbed


@pytest.mark.parametrize(
    "tail",
    [
        "NORMAL, more text\n",
        '", "other": "value"}',
        "prose that is not credential-shaped\n",
    ],
)
def test_an_unterminated_anchored_value_stops_before_a_non_credential_line(tail: str) -> None:
    """Rule 2 of the round-7 direction: mask the credential-SHAPED run, then stop.

    An anchored value with no closing quote used to mask "the remainder", which is not
    all credential: `…<BODY>\nNORMAL, more text\n` lost that whole line, and
    `…<BODY>\n", "other": "value"}` lost the following JSON. A line counts as
    credential-shaped only when, stripped, it is empty, a PEM header/footer, or solely
    `[A-Za-z0-9+/=]` plus at most one trailing comma — and the mask must stop before the
    first line that is not, byte-identical. Not "stop at the first newline": a bare
    newline body would then publish the rest of a real key.
    """
    import local_operator.redaction_shapes as rs

    header = "-----BEGIN RSA PRIVATE KEY-----"
    text = '{"private_key": "' + header + "\n" + _PEM_BODY + "\n" + tail
    scrubbed, hits = rs.scrub_shapes_with_hits(text)
    assert _PEM_BODY not in scrubbed, "the credential run was not masked"
    assert scrubbed.endswith(tail), f"the following line was eaten: {scrubbed!r}"
    assert not any(hit.complete for hit in hits), "an unterminated value claimed completion"


def test_a_credential_shaped_run_is_masked_to_its_end() -> None:
    """The other half of rule 2: a multi-line base64 body IS masked, all of it.

    The naive alternative — stop at the first newline — would publish every line after
    the header, which is the whole body of a real key.
    """
    import local_operator.redaction_shapes as rs

    header = "-----BEGIN RSA PRIVATE KEY-----"
    text = '{"private_key": "' + header + "\n" + _PEM_BODY + "\n" + _PEM_BODY + "\n"
    scrubbed, hits = rs.scrub_shapes_with_hits(text)
    assert _PEM_BODY not in scrubbed
    # The body is gone and only the value's prefix and the separators remain; the exact
    # tail is a separator rather than a fixed string, which is why this asserts the
    # CONTENT instead of an equality.
    assert scrubbed.startswith('{"private_key": "[redacted]')
    assert not any(hit.complete for hit in hits)


@pytest.mark.parametrize(
    "prefix",
    ["", "    ", "\t", "1| ", "2|     ", "\t3| "],
)
@pytest.mark.parametrize("suffix", ["", " ", ","])
def test_a_body_line_is_masked_as_tools_print_it(prefix: str, suffix: str) -> None:
    """R8-1: indentation, tabs, `N| ` line numbers and trailing padding are body lines.

    The run used to be spelled at line start with no whitespace, so an indented or
    line-numbered body ended it and every remaining line was published, silently — and a
    `N| ` prefix is exactly how this product's own `read` tool renders a file, so an agent
    reading a `.pem` hits it. Each variant must mask the whole body and leave the line
    after the block byte-identical.
    """
    import local_operator.redaction_shapes as rs

    header = "-----BEGIN RSA PRIVATE KEY-----"
    text = (
        '{"private_key": "' + header + "\n" + prefix + _PEM_BODY + suffix + "\nNORMAL, more text\n"
    )
    scrubbed, hits = rs.scrub_shapes_with_hits(text)
    assert _PEM_BODY not in scrubbed, f"body published for prefix={prefix!r} suffix={suffix!r}"
    assert scrubbed.endswith("NORMAL, more text\n"), scrubbed
    assert not any(hit.complete for hit in hits)


@pytest.mark.parametrize(
    "prefix",
    [
        "1| ",  # this product's own `read`
        "1\t",  # `cat -n`
        "1:",  # `grep -n`
        "1>",
        "1->",
        "1 -> ",
        "3| 4| ",  # a doubled prefix
        "12:\t",
    ],
)
def test_a_line_number_prefix_never_ends_the_run(prefix: str) -> None:
    """Q9-F1: every way a tool numbers a line, not just `N| `.

    The prefix was one numeric spelling plus one space, so `cat -n`'s `number<TAB>`,
    `grep -n`'s `number:`, an arrow prefix and a doubled prefix each ended the run and
    published the WHOLE body on six of seven surfaces. Enumerating spellings is what cost
    four rounds; the pattern now takes an optional, REPEATED prefix with any of the
    separators these tools emit, spaces or a TAB around them.
    """
    import local_operator.redaction_shapes as rs

    header = "-----BEGIN RSA PRIVATE KEY-----"
    text = '{"private_key": "' + header + "\n" + prefix + _PEM_BODY + "\nNORMAL, more text\n"
    scrubbed, hits = rs.scrub_shapes_with_hits(text)
    assert _PEM_BODY not in scrubbed, f"body published for prefix {prefix!r}"
    assert scrubbed.endswith("NORMAL, more text\n"), scrubbed
    assert not any(hit.complete for hit in hits)


@pytest.mark.parametrize(
    "prose",
    ["12| done", "12| 42", "5| NORMAL, more text", "1|", "|"],
)
def test_a_numbered_prose_line_survives(prose: str) -> None:
    """The other half of the generalisation: a numbered line of ORDINARY text stays.

    A prefix must not turn prose into a credential-shaped line — `12| done` is a log
    line, not key material.
    """
    import local_operator.redaction_shapes as rs

    header = "-----BEGIN RSA PRIVATE KEY-----"
    text = '{"private_key": "' + header + "\n" + _PEM_BODY + "\n" + prose + "\n"
    scrubbed = rs.scrub_shapes(text)
    assert prose in scrubbed, scrubbed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    [
        "cat key.pem",
        "head -n 6 key.pem",
        "sed -n '1,12p' key.pem",
        "cat -n key.pem",
        "grep -n -e '' key.pem",
    ],
)
async def test_a_real_bash_command_never_publishes_an_open_key_body(
    tmp_path: Path, command: str
) -> None:
    """The PIPE path, through the real tool call — the shape nothing covered before.

    `_in_key_block` had no test at all, which is how this layer's body masking went
    unreachable while `tests/unit/secrets` stayed green: the header test was aliased from
    the shape table (whose anchors are per-line) and searched against a WHOLE multi-line
    chunk, so the block opened only when the chunk was exactly one header line — which the
    release point's hold makes impossible. Measured then: `head -n 6 key.pem` published 5
    of 25 body lines through the real tool call. This drives the real command and asserts
    the body is gone from the tool result.
    """
    from local_operator.harness.types import ToolContext
    from local_operator.tools.builtin import execute_bash
    from local_operator.variables import VariableStore

    body = "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQ"
    key = "-----BEGIN RSA PRIVATE KEY-----\n" + "\n".join([body] * 6) + "\n"
    (tmp_path / "key.pem").write_text(key)
    context = ToolContext(cwd=str(tmp_path), variables=VariableStore(cwd=str(tmp_path)))

    result = await execute_bash("call-pipe-key", {"command": command}, None, None, context)

    text = "".join(getattr(part, "text", "") for part in result.content)
    assert body not in text, f"the key body reached the tool result for {command!r}"
    assert "[redacted]" in text


def test_two_stores_in_one_process_are_independent() -> None:
    """The registration set and its cap are PER STORE, not per process.

    They were declared in the class body, and `.add()` on a class-level set cannot
    create an instance attribute — so every `VariableStore` in the process shared one
    set. One session's masked credentials were then scanned in another session's
    results, and after 64 values process-wide containment stopped for everybody. It
    surfaced as a red CI shard (an earlier test in the shard had filled the shared set)
    and as the cross-session coupling the design note forbids.
    """
    a = VariableStore(cwd=".")
    b = VariableStore(cwd=".")
    a.redact("MONGO_DSN=mongodb+srv://u:sh4pedSentinelPw@host/db")

    assert "sh4pedSentinelPw" in a.redaction_values()
    assert b.redaction_values() == [], "a fresh store already holds another store's values"
    assert a._shape_registrations is not b._shape_registrations


def test_the_registration_cap_is_per_store() -> None:
    """Filling one store past the cap must not stop containment in another."""
    a = VariableStore(cwd=".")
    b = VariableStore(cwd=".")
    for index in range(VariableStore.MAX_DETECTED_REGISTRATIONS + 5):
        a.redact(f"PASSWORD=secretValue{index:03d}xx")

    assert len(a.redaction_values()) == VariableStore.MAX_DETECTED_REGISTRATIONS
    b.redact("PASSWORD=otherStoreSecret9x")
    assert "otherStoreSecret9x" in b.redaction_values(), "the cap leaked across stores"


def test_a_fresh_store_does_not_inherit_another_stores_containment() -> None:
    """What the per-store fix GUARANTEES, and what it deliberately does not.

    Guaranteed: store A's registrations and A's cap are A's alone. A fresh store B
    registers its own values normally, is not masked by A's set, and is not silenced by
    A's cap — which is what the class-level set broke, deterministically, because the
    sink it guards (`self._redactions`) was per-instance while the guard was not: the
    first store in a process registered a value and every later store skipped it, so a
    value a later store should have contained came back IN THE CLEAR.

    NOT guaranteed, and by design: B does not inherit A's containment. Two stores in one
    process are two security domains — the product creates ONE store per session, so
    per-session containment is the intended scope, and a process-wide set is exactly the
    coupling that made one session's reads affect another's text.
    """
    a = VariableStore(cwd=".")
    b = VariableStore(cwd=".")
    a.redact("MONGO_DSN=mongodb+srv://svc:storeASecret1234@host/db")

    assert "storeASecret1234" in a.redaction_values()
    # B is fresh: it holds nothing of A's, and it still masks what it registers itself.
    assert b.redaction_values() == []
    assert "storeBSecret5678" in (b.redact("PASSWORD=storeBSecret5678") and b.redaction_values())
    # The boundary, stated as an assertion so it cannot drift into a claim: A's value is
    # not contained in B, because containment is per store by design.
    assert "storeASecret1234" not in b.redaction_values()


# --- the classification: contained, or in the model's context -----------------


def test_the_grading_separates_contained_from_exposed() -> None:
    """``exposed`` is the severity, ``complete`` is the claim, and they differ.

    Three outcomes are possible for one hit and the design needs all three: a value
    masked whole (contained — the ordinary case), a value whose mask was withheld
    because its extent cannot be proven (a truncated PEM: contained, unclaimable),
    and a value with readable material still in the text (exposed — the one case
    that is a compromise). Collapsing the first two into the third is what made
    every masking event look like a compromise, and having no third at all is what
    kept the real one silent.
    """
    import local_operator.redaction_shapes as rs

    value = "regional-opensearch-admin-9"
    hit = rs.ShapeHit(label="credential-assignment", value=value, window=value)

    contained = rs._only_fully_masked([hit], "OPENSEARCH_PASSWORD=[redacted]")[0]
    assert (contained.complete, contained.exposed) == (True, False)

    exposed = rs._only_fully_masked([hit], f"OPENSEARCH_PASSWORD=x  # kept: {value}")[0]
    assert (exposed.complete, exposed.exposed) == (False, True)

    # The third outcome: a key whose extent is unknown. Everything visible is
    # masked, so nothing reached the model, and the claim is still withheld.
    truncated = f'{{"private_key": "-----BEGIN RSA PRIVATE KEY-----\n{_PEM_BODY}\n"}}'
    _, pem_hits = rs.scrub_shapes_with_hits(truncated)
    assert pem_hits, "the truncated block filed no hit"
    assert not any(h.complete for h in pem_hits), "a truncated key claimed a whole mask"
    assert not any(h.exposed for h in pem_hits), "a masked truncated key is not an exposure"


#: The positives that ESCALATE, named by the case's own reason string.
#:
#: An EXACT set, in both directions, like the partial-mask ratchet further down: a
#: case that joins it is a new false rotation demand, and one that leaves it without
#: the code change being recorded in the same commit is a fix nobody can see.
#:
#: ``amqp DSN`` (``amqp://guest:guest@rabbit.internal:5672/``) is the one that must
#: STAY. Its username and its password are the same five characters, and the DSN rule
#: deliberately keeps ``amqp://user:`` readable — so the password's own characters
#: genuinely sit in the text the model reads, ``value in text`` finds them, and the
#: hit escalates. That is the CONSERVATIVE direction and it is the point of this
#: change: base ``4a16dd62`` filed NOTHING here (the silent hole this PR closes), and
#: a rotation demand on a conceivably-readable secret is recoverable where a missed
#: leak is not.
#:
#: Do NOT silence it by excluding the marker or the mask-kept neighbours before
#: testing. QA round 1 proposed exactly that, and it suppresses this genuine
#: survivor along with the three false ones — the two ``.npmrc`` cases and the
#: cookie-header case, which are the marker-identity false positives pinned below.
_ESCALATING_POSITIVE_CASES = frozenset({"amqp DSN"})


def test_only_the_documented_positive_case_escalates() -> None:
    """The corpus is the referee: one escalation, NAMED, with 179/179 still masked.

    QA round 1 (Q1) measured four positives escalating at head ``91d70791``, three of
    them on nothing readable at all. Anchoring the check on the hit's own VALUE —
    and stripping the redaction marker before taking the text's runs — takes those
    three out and leaves the one survivor the frozen set above argues for.

    A structural pin rather than three point tests, so a corpus case that starts
    escalating cannot arrive silently: a rise fails HERE, and a fall fails too until
    the frozen set is updated in the commit that caused it.
    """
    import local_operator.redaction_shapes as rs

    escalating: set[str] = set()
    unmasked: list[str] = []
    for case in POSITIVE_CASES:
        masked, hits = scrub_shapes_with_hits(case.text)
        if REDACTION_MARKER not in masked:
            unmasked.append(case.reason)
        if rs.shape_report(hits).reached_model:
            escalating.add(case.reason)

    assert not unmasked, f"the mask stopped holding for: {unmasked[:5]}"
    assert escalating == set(_ESCALATING_POSITIVE_CASES), (
        f"the escalating positives changed: {sorted(escalating)} — a case that JOINED "
        "is a new false rotation demand, and one that LEFT has to update this set in "
        "the same commit as the fix that caused it"
    )


def test_a_marker_valued_hit_no_longer_escalates() -> None:
    """The half of Q1 QA got right, driven through the shipped path.

    Both ``.npmrc`` ``_authToken`` spellings and the cookie-header case are runs where
    a SECOND rule matched the marker the first rule had just inserted, so the exposed
    hit's own value IS (or contains) ``[redacted]``. The old check searched windows of
    the matched REGION against the masked text, found the literal marker it had just
    written, and filed a rotation demand for a value that was never readable.
    """
    import local_operator.redaction_shapes as rs

    for reason in (
        "the .npmrc auth token, bare name",
        "the .npmrc auth token with a registry path",
        "a cookie header inside a curl invocation",
    ):
        case = next(c for c in POSITIVE_CASES if c.reason == reason)
        masked, hits = scrub_shapes_with_hits(case.text)
        assert REDACTION_MARKER in masked, f"{reason} is no longer masked"
        assert (
            rs.shape_report(hits).reached_model is False
        ), f"{reason} files a rotation demand again: {masked!r}"
        # The mechanism the case exists for, pinned rather than inferred: without a
        # hit whose own value carries the marker this case stops exercising anything.
        assert any(
            REDACTION_MARKER in hit.value for hit in hits
        ), f"{reason} no longer produces a marker-valued hit"


def test_a_password_equal_to_its_own_username_still_escalates() -> None:
    """The ``amqp``/``postgres`` username==password class, which must NOT be silenced.

    Escalating is the conservative direction and this is a real survivor, not a
    misfire: the DSN mask keeps ``amqp://user:`` readable by design, so the
    credential's own characters are in the model-visible text. The assertion below is
    the measurement — the password's value read straight out of the masked output.
    """
    import local_operator.redaction_shapes as rs

    case = next(c for c in POSITIVE_CASES if c.reason == "amqp DSN")
    masked, hits = scrub_shapes_with_hits(case.text)
    assert REDACTION_MARKER in masked, "the DSN password is no longer masked"
    (hit,) = [h for h in hits if h.label == "dsn-password"]
    assert hit.value and hit.value in masked, (
        "the credential's own characters are no longer readable in the masked text, "
        "so escalating would be a misfire and this case belongs in the frozen set "
        "above as a documented false positive instead"
    )
    assert hit.exposed is True
    assert rs.shape_report(hits).reached_model is True


def test_a_marker_inside_the_credentials_own_value_is_a_recorded_limit() -> None:
    """The limit recorded on ``_credential_fragments_survive``, pinned so it is seen.

    A credential that itself contains ``[redacted]`` and survives only PARTIALLY is
    ungradeable: the surviving fragment is spelled exactly like the marker a mask
    writes, and no reading of the text can separate the two. That identity is why the
    marker-identity cases above escalate wrongly under a region search AND why this
    one cannot be graded by inspection. The wholly-surviving copy is still caught, so
    only the partial case is given up — recorded here rather than paid for, because
    reaching it needs an operator secret containing the harness's marker string.
    """
    import local_operator.redaction_shapes as rs

    value = f"tok{REDACTION_MARKER}tail"  # an operator secret that contains the marker
    whole = rs.ShapeHit(label="credential-assignment", value=value, window=value)
    assert rs._only_fully_masked([whole], f"PASSWORD={value}")[0].exposed is True

    partial = rs.ShapeHit(label="credential-assignment", value=value, window=value)
    assert rs._only_fully_masked([partial], f"PASSWORD=tok{REDACTION_MARKER}")[0].exposed is False


def test_a_mask_that_stopped_inside_a_credential_is_an_exposure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partial mask is an EXPOSURE, and only the window half can see it.

    The shape this grading exists for is a mask that stopped INSIDE a credential:
    the pattern captured less than the credential, so the marker was written over
    the captured part and the rest stayed readable. The whole-value question cannot
    see that — it reads the text as delivered, and the marker sits where the head of
    the value would be — so the window half is the only thing between that mask and
    a silent compromise, and a case that pins the whole-value half alone leaves it
    deletable (agent review R1, F1: of the corpus's 183 graded hits, not one needs
    this half).

    The mirror is asserted with it, so the case cannot be satisfied by a guard that
    escalates on any mask at all.
    """
    import local_operator.redaction_shapes as rs

    # Both window arms must answer this case, so pin the OTHER one here: at this
    # value's 15 windows the crossover picks the per-window search arm, which left
    # `_present_windows` (the gated pass, and the arm the incident itself took)
    # pinned by no test at all (agent review R2, C2).
    monkeypatch.setattr(rs, "_searches_are_cheaper", lambda keys: False)

    value = "p@ssw0rd-at-the-tail"
    hit = rs.ShapeHit(label="dsn-password", value=value, window=value)
    tail = value[6:]  # what a mask that stopped after six characters left readable

    # Only a run of the value is readable, and the whole value is not in the text
    # because the marker is where its head was: the seam is the whole test.
    partial = rs._only_fully_masked([hit], f"PASSWORD={REDACTION_MARKER}{tail}")[0]
    assert (partial.complete, partial.exposed) == (False, True)
    # And the exposure is a compromise the session reports as one: this is the
    # predicate behind the escalated notice, so the case cannot pass by grading a
    # hurt hit that nothing acts on.
    assert rs.shape_report([partial]).reached_model is True

    # The mirror: the marker and nothing else of this value — contained, claimable.
    contained = rs._only_fully_masked([hit], "PASSWORD=" + REDACTION_MARKER)[0]
    assert (contained.complete, contained.exposed) == (True, False)


def test_the_fragment_floor_is_pinned_from_both_sides() -> None:
    """``_FRAGMENT_WINDOW`` is the floor this check is built on, and it is pinned.

    It is the shortest run of a credential the pass will call readable material, so
    the constant is load-bearing in both directions and nothing else in the suite
    can see either end: the corpus contains no case whose grading depends on it
    (agent review R1, F3 — the whole test file stays green at 7, and the predicate
    restatement cannot disagree about a case the corpus does not hold). The pin is
    therefore behavioural rather than an equality against the constant: a run of
    exactly the floor survives a mask seam, and a run one character shorter does
    not, so the pair fails if the floor moves either way.
    """
    import local_operator.redaction_shapes as rs

    six = "abcdef"
    # The value straddles the mask: the whole-value half cannot see it (the marker
    # is between its halves) and the window half can (its characters are back
    # together once the marker is stripped). At a floor of 7 the value has no window
    # and this becomes contained.
    hit = rs.ShapeHit(label="password", value=six, window=six)
    assert rs._only_fully_masked([hit], f"k=abc{REDACTION_MARKER}def\n")[0].exposed is True

    # One character below the floor: the same shape has no window at all, so nothing
    # of it is readable material. At a floor of 5 this becomes an exposure.
    five = "abcde"
    shorter = rs.ShapeHit(label="password", value=five, window=five)
    assert rs._only_fully_masked([shorter], f"k=ab{REDACTION_MARKER}cde\n")[0].exposed is False


# --- what the classification COSTS --------------------------------------------
#
# The half of this pass that grades a hit used to ask the text one question per
# hit — ``value in text`` — and the text is megabytes, so the price was
# ``hits x bytes``: five session runtimes on this machine were found frozen for 1.5
# to 7.2 hours with 100% of their event-loop thread sampled inside its C-level
# search, heartbeats stale for hours, and every control call (``lop stop``,
# ``steer``, ``cancel``) refusing, because they all marshal onto the busy loop. The
# tests below are the structural half of the fix: they count the WORK, in bytes
# walked, and assert that the hit count does not appear in it — never a stopwatch,
# for the reason AGENTS.md records under "Prefer a structural invariant to a
# numeric one".

#: One credential row, repeated. A large tool result or transcript body carries the
#: same command and the same result over and over, so this is the shape the pass is
#: expensive on — and the one the old per-hit scan made quadratic, because the hits
#: multiply while the text stays a single string. Built from ``SENTINEL`` so this
#: file still spells no credential-shaped literal of its own.
_HIT_ROW = "MONGO_DSN=" + SENTINEL + "\n"


def _distinct_row(index: int) -> str:
    """One row carrying a credential no other row carries.

    The USERNAME is deliberately the same in every row and the VALUE's tail is what
    varies: the credential the pass grades is the value, so distinct usernames would
    still be one credential to grade and would never reach the key count this fixture
    exists for.
    """
    tailed = SENTINEL.replace(SENTINEL_FRAGMENT, f"{SENTINEL_FRAGMENT}-{index:04d}")
    return "MONGO_DSN=" + tailed + "\n"


#: Filler with no anchor in it at all, so the credential rows are the only thing
#: the pass has to grade.
_PAD_ROW = "a line of ordinary text with nothing shaped in it whatsoever\n"


def _rows_text(rows: int, size: int, *, distinct: bool = False) -> str:
    """``rows`` credential rows inside about ``size`` bytes of text."""
    body = "".join(_distinct_row(index) for index in range(rows)) if distinct else _HIT_ROW * rows
    return body + _PAD_ROW * max(0, (size - len(body)) // len(_PAD_ROW))


#: A credential long enough to carry many WINDOWS, which is the half of the check the
#: cost tests could not see: the sentinel's own value asks 11 distinct windows of the
#: text, the credential below asks 51, and the per-window search arm walks the text
#: once for each. The tail is a counting sequence rather than a repeated character
#: because the windows are what that arm searches for, and a run of one character
#: collapses into a single key. Derived from ``SENTINEL`` again, so the file still
#: spells no credential-shaped literal of its own.
_LONG_VALUE = SENTINEL_FRAGMENT + "".join(f"{index:02d}" for index in range(20))
_LONG_ROW = "MONGO_DSN=" + SENTINEL.replace(SENTINEL_FRAGMENT, _LONG_VALUE) + "\n"


def _long_rows_text(rows: int, size: int) -> str:
    """``rows`` copies of the LONG credential inside about ``size`` bytes of text."""
    body = _LONG_ROW * rows
    return body + _PAD_ROW * max(0, (size - len(body)) // len(_PAD_ROW))


#: The real ``_SurvivalIndex._readable_text``, captured here so the wrapper
#: :func:`_grading_work` installs can call it without chaining onto a wrapper an
#: earlier call left on the class.
_ORIGINAL_READABLE_TEXT = redaction_shapes._SurvivalIndex._readable_text


class _CountingText(str):
    """A str that counts the BYTES searched on it by the whole-text arms.

    ``value in text`` is a C-level walk of the whole string, and that walk is the
    cost this file measures — the same measure for the old code (one walk per hit)
    and for the new one (one walk per key), which is what lets the before/after
    claim be one number rather than two mechanisms.

    A search can land on a DERIVED string rather than on the object the caller
    holds: the window half reads the text with the marker stripped, which is a
    different string whenever the text carries a marker — and every graded fixture
    here does, because a mask is what put the marker there. Such a copy is built
    with the caller's ``sink`` so its walks are counted with the caller's; a private
    counter on the copy would be read by nobody, which is exactly how the window
    half's searches went unmeasured (agent review R1, F2).
    """

    #: Declared on the CLASS, not set only per instance: the derived copies built in
    #: ``__new__`` are typed from this annotation, and without it pyright reports
    #: three errors on the sink reads below (agent review R2, C1).
    _sink: list[int]

    def __new__(cls, value: str, sink: list[int] | None = None) -> "_CountingText":
        text = super().__new__(cls, value)
        text._sink = sink if sink is not None else [0]
        return text

    @property
    def scanned(self) -> int:
        """The bytes searched on this object and on every copy sharing its sink."""
        return self._sink[0]

    def __contains__(self, key: str) -> bool:
        self._sink[0] += len(self)
        return super().__contains__(key)


def _grading_work(text: str, monkeypatch: pytest.MonkeyPatch) -> tuple[int, int, int]:
    """The work grading ``text`` costs: ``(searched, passed over, credentials)``.

    Both arms are counted, so neither can hide from the assertion: the whole-text
    searches through :class:`_CountingText` — including the ones the window half
    makes on the marker-stripped copy of the text, which is where most of its cost
    went unmeasured — and the one-pass arms by wrapping the two helpers that read the
    text at once. The hits come from the real pass over the real fixture, so what is
    measured is the shipped composition rather than a hand-built approximation of it.
    The third number is how many DISTINCT credentials were graded, which is what
    decides which arm runs.
    """
    from local_operator import redaction_shapes as rs

    scrubbed, hits = scrub_shapes_with_hits(text)
    assert hits, "the fixture graded no hits, so measuring it would prove nothing"
    # The fixture has to reach BOTH halves: an EXPOSED hit is answered by the
    # whole-value half alone, and the window half — where the quadratic term lived
    # — would never be measured. Every hit this shape produces is contained, so the
    # fixture is the shape the pass is expensive on rather than a special case.
    assert not any(hit.exposed for hit in hits), "the fixture short-circuits the window half"
    counted = _CountingText(scrubbed)

    # The window half asks its questions of ``_readable_text()``, which is the text
    # with the marker stripped — a COPY, when there is a marker to strip, and one the
    # caller never holds. That lookup is the window half's whole cost, so the copy is
    # given the sink of the text THIS call handed in: without it the long-credential
    # fixture below reported 1.86 walks while its window half was asking 51 questions
    # of the text (agent review R1, F2).
    #
    # The original is taken from the module rather than from the class: this helper is
    # called several times per test with one monkeypatch, and a wrapper that captured
    # the class attribute would chain onto the previous call's wrapper and hand its
    # answer back — measuring the second fixture with the first fixture's meter, which
    # is the same blindness one layer down.
    def counted_readable(index: Any) -> str:
        readable = _ORIGINAL_READABLE_TEXT(index)
        if isinstance(readable, _CountingText):
            # No marker to strip: the readable text IS the counted text the caller
            # handed in, and counting it counts these searches with it.
            return readable
        sink = getattr(getattr(index, "_text", None), "_sink", None)
        if sink is None:
            # A text this meter did not hand in: the wrapper stays on the class for
            # the rest of the test, and ``scrub_shapes_with_hits`` reads its own text
            # before every measured call. Those searches are not the measurement, so
            # the text is handed back untouched.
            return readable
        return _CountingText(readable, sink)

    monkeypatch.setattr(rs._SurvivalIndex, "_readable_text", counted_readable)
    passed = 0

    def account(original: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal passed
            passed += len(args[0])
            return original(*args, **kwargs)

        return wrapper

    for name in ("_present_windows", "_present_heads"):
        original = getattr(rs, name, None)
        if original is None:
            # A tree with no one-pass arm at all: its per-hit searches are then the
            # whole measurement, which is exactly what this test exists to catch,
            # and the assertion below still reports the bytes it walked.
            continue
        monkeypatch.setattr(rs, name, account(original), raising=False)
    rs._only_fully_masked(hits, counted)
    return counted.scanned, passed, len({hit.value for hit in hits})


def test_grading_never_costs_a_walk_of_the_text_per_hit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Four times the hits over the same text must not cost four times the work.

    This is the defect itself, measured the way the freeze measured it. The grading
    half used to be one whole-text search per hit, so the same bytes walked four
    times as far when the hits quadrupled — which is what turned a 50 s replay into
    a runtime that had to be SIGKILLed by hand. The bound is on WORK (bytes walked
    by a search or a pass), which is a fact about the code rather than about this
    machine, so it cannot flake on a loaded runner.
    """
    size = 256_000
    few_text = _rows_text(rows=200, size=size)
    many_text = _rows_text(rows=800, size=size)
    few = _grading_work(few_text, monkeypatch)
    many = _grading_work(many_text, monkeypatch)
    assert sum(many[:2]) <= sum(few[:2]) * 1.5, (
        f"grading got more expensive with the hit count: {few} -> {many} bytes walked "
        f"over {size} bytes of text"
    )

    # And at a FIXED hit density, four times the text costs about four times the
    # work, not sixteen: cost per byte is what rose with size in the measurements
    # this change came from (1.19 -> 2.63 us per byte from 64 KB to 512 KB), and it
    # rose because the hits rose with the size.
    quarter_text = _rows_text(rows=200, size=64_000)
    whole_text = _rows_text(rows=800, size=256_000)
    quarter = _grading_work(quarter_text, monkeypatch)
    whole = _grading_work(whole_text, monkeypatch)
    quarter_rate = sum(quarter[:2]) / len(quarter_text)
    whole_rate = sum(whole[:2]) / len(whole_text)
    assert whole_rate <= quarter_rate * 2, (
        f"the cost per byte of grading rises with the size of the text: "
        f"{quarter_rate} -> {whole_rate}"
    )


def test_grading_walks_a_text_at_most_its_keys_worth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The absolute bound, at both ends of the key count.

    One credential repeated: the whole-value half is one search and the window half
    is one search per window of that credential — the arm chosen for a small key
    count, and at most ``(1 + windows)`` walks of the text for the fixture below.
    Four hundred DISTINCT credentials: both halves are past the crossover, so each
    reads the text once whatever the key count — the arm that keeps a text full of
    credentials from costing a walk per credential.
    """
    repeated = _rows_text(rows=400, size=256_000)
    searched, passed, distinct = _grading_work(repeated, monkeypatch)
    assert distinct == 1, f"the repeated fixture graded {distinct} credentials"
    assert passed == 0, "the one-credential fixture took the one-pass arm"
    limit = 40 * len(repeated)
    assert searched <= limit, f"{searched} bytes walked over {len(repeated)}, one credential"

    many = _rows_text(rows=400, size=256_000, distinct=True)
    searched, passed, distinct = _grading_work(many, monkeypatch)
    assert distinct == 400, f"the distinct fixture graded {distinct} credentials"
    # One walk of the text is the marker probe that decides whether the window half
    # has anything to strip at all; the arms themselves are the one-pass ones.
    assert searched <= len(many), f"{searched} bytes searched for {len(many)}"
    assert passed <= 4 * len(many), (
        f"the many-credential fixture walked {searched + passed} bytes over "
        f"{len(many)}: the one-pass arms must not scale with the key count"
    )


def test_a_long_credential_is_measured_where_its_cost_is(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The window half's walks are counted, not just the whole-value half's.

    The arm the window half takes while its key count is small is one C-level search
    per WINDOW of the credential, and those searches run on the marker-stripped copy
    of the text — so a meter that reads only the object the caller holds sees none of
    them. The sentinel's own value carries 11 distinct windows; the long credential
    below carries 51, and it is the one whose cost the meter has to show.

    The floor asserted here is the window count itself, which is what makes this a
    test of the instrument rather than of the code: the fixture below reported 1.86
    walks on the old meter, and no such report can reach 51 windows' worth of walks
    without counting those searches (agent review R1, F2).
    """
    import local_operator.redaction_shapes as rs

    text = _long_rows_text(rows=400, size=256_000)
    searched, passed, distinct = _grading_work(text, monkeypatch)
    # The credential as the pass grades it — the value ON the hit, which is what the
    # window half keys its windows on — rather than the literal the row was built
    # from: the DSN rule hands over part of what the row carries.
    (value,) = {hit.value for hit in scrub_shapes_with_hits(text)[1]}
    windows = len(value) - rs._FRAGMENT_WINDOW + 1

    assert distinct == 1, f"the long-credential fixture graded {distinct} credentials"
    assert passed == 0, "the long-credential fixture took the one-pass arm"
    # The floor is the window count, at half the fixture's length: not a byte-exact
    # claim about which strings get walked, but one only a meter that counts the
    # window half's searches can reach.
    assert searched >= windows * len(text) // 2, (
        f"{searched} bytes searched over {len(text)} for {windows} windows: the meter "
        "is not counting the window half's walks"
    )
    # And it is still bounded by the windows, the whole-value search and the marker
    # probe: the 400 hit rows do not appear in it.
    assert searched <= (windows + 4) * len(
        text
    ), f"{searched} bytes walked for {windows} windows over {len(text)} bytes of text"


def test_the_single_pass_arm_verifies_the_whole_value_it_claims() -> None:
    """The one-pass arm's key is only a CANDIDATE: the value is checked in full.

    ``_present_heads`` is the arm taken once the key count passes the crossover, and
    it is keyed on each value's own first ``_FRAGMENT_WINDOW`` characters. A text can
    carry that head without carrying the credential — a name that starts like a
    token, a shorter value that prefixes a longer one — so every candidate is sliced
    against the whole value before it enters the answer. Deleting that slice leaves
    the whole suite green (agent review R1, F5), which is why it is pinned here
    directly: through the composed pass the window half usually reaches the same
    verdict by its own route, so the difference only shows on the narrow
    marker-bearing shape that ``_credential_fragments_survive`` records as a limit.
    """
    import local_operator.redaction_shapes as rs

    head = "verification-head-9"
    # The head's six characters, and nothing after them: a candidate, not a hit.
    assert rs._present_heads("text with verification-XXXX nothing else", [head]) == set()
    # The same value in full IS a hit, and a value shorter than the window is looked
    # up at its own length rather than by a window it does not have.
    assert rs._present_heads(f"text with {head} in it", [head]) == {head}
    assert rs._present_heads("PASSWORD=abcde tail", ["abcde"]) == {"abcde"}


def test_the_single_pass_arm_is_total_over_an_empty_value() -> None:
    """The empty value is skipped rather than raising ``IndexError``.

    An empty value has no character to key on, so its key would be ``""`` and the
    head lookup inside ``_present_heads`` would index it at 0. Both callers filter
    it out today (``_SurvivalIndex`` drops it, and ``_credential_fragments_survive``
    returns before the index), which is why this was latent instead of live — but a
    helper whose declared input is a sequence of strings should not depend on two
    callers to stay total (agent review R1, F4).
    """
    import local_operator.redaction_shapes as rs

    assert rs._present_heads("aaa", [""]) == set()
    assert rs._present_heads("aaa", ["", "aaa"]) == {"aaa"}


def test_the_two_arms_of_the_index_are_interchangeable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The arm is a COST decision, so it may never move a mask or a grading.

    Both arms are the same two predicates — one C-level search per key, or one
    gated pass over the text — and which one runs is decided from two measured
    per-byte costs. Forcing each arm over the whole corpus and over the incident's
    own shape is what keeps that a fact about the code rather than an argument.
    """
    from local_operator import redaction_shapes as rs

    texts = [
        *(case.text for case in (*POSITIVE_CASES, *NEGATIVE_CASES)),
        (_HIT_ROW + _PAD_ROW) * 40,
        _HIT_ROW * 40 + _PAD_ROW * 400,
        "".join(_distinct_row(index) for index in range(200)),
    ]

    def grading() -> list[Any]:
        return [
            (
                scrub_shapes_with_hits(text)[0],
                [
                    (hit.label, hit.value, hit.window, hit.complete, hit.exposed)
                    for hit in scrub_shapes_with_hits(text)[1]
                ],
            )
            for text in texts
        ]

    monkeypatch.setattr(rs, "_searches_are_cheaper", lambda keys: True)
    searches = grading()
    monkeypatch.setattr(rs, "_searches_are_cheaper", lambda keys: False)
    passes = grading()
    assert searches == passes, "the arm chosen for cost changed a mask or a grading"


def _corpus_grading() -> str:
    """The corpus's masked text and full hit set, serialised as one digest."""
    canonical = []
    for case in (*POSITIVE_CASES, *NEGATIVE_CASES):
        masked, hits = scrub_shapes_with_hits(case.text)
        canonical.append(
            [
                case.reason,
                masked,
                [[hit.label, hit.value, hit.window, hit.complete, hit.exposed] for hit in hits],
            ]
        )
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


#: What the whole corpus produces TODAY, values included: every masked text, every
#: hit and every grading. Regenerate by printing ``_corpus_grading()`` — and only in
#: the commit that argues why the behaviour moved.
_CORPUS_GRADING_DIGEST = "8895d033a508776b8f2822477eded754315af15ea85db4a496233f4539f79316"


def test_the_corpus_masks_and_grades_byte_for_byte_as_it_always_has() -> None:
    """A security control is not allowed to move one byte because it got faster.

    The pass that decides whether a credential is masked, and how badly it leaked
    if it was not, was re-expressed (one index over the text instead of one search
    per hit). This is the corpus as the referee, in both directions: a mask that
    weakened, a hit that stopped being filed, a severity that changed — any of
    those moves the digest, and none of them may move it silently.
    """
    assert _corpus_grading() == _CORPUS_GRADING_DIGEST, (
        "the corpus no longer produces the same masked text and hit set. That is a "
        "BEHAVIOUR change on a credential control, not a fixture update: the change "
        "that moves this constant has to say which case moved and why it is safe."
    )


def test_the_grading_of_every_corpus_hit_matches_the_predicate() -> None:
    """``exposed`` restated in the test, so a case the digest cannot see still holds.

    The digest pins the corpus; this pins the PREDICATE the corpus is evidence for:
    readable material from the credential is in the text the model reads when the
    whole value is in the text as delivered, or when the value is at least a window
    long and one of its six-character windows is in the text with the redaction
    marker stripped. Restating it is the point — a re-implementation that agrees
    with the corpus for the wrong reason fails on the next case.
    """
    checked = 0
    for case in (*POSITIVE_CASES, *NEGATIVE_CASES):
        masked, hits = scrub_shapes_with_hits(case.text)
        readable = masked.replace(REDACTION_MARKER, "")
        for hit in hits:
            value = hit.value
            exposed = bool(value) and value != REDACTION_MARKER
            if exposed:
                exposed = value in masked or (
                    len(value) >= 6
                    and any(value[start : start + 6] in readable for start in range(len(value) - 5))
                )
            assert hit.exposed is exposed, (case.reason, hit.label)
            checked += 1
    assert checked > 150, f"the corpus graded only {checked} hits: it is not evidence"


def test_the_contained_notice_names_the_tool_and_carries_no_value() -> None:
    """Contained: it happened, it was handled, there is no exposure, clean up.

    This is the ordinary outcome — a credential reached a tool and was masked
    before the model could read it — so the notice may not use the escalation's
    words: no rotation demand, no "compromised". What it owes instead is the one
    action this path has: delete any plaintext copy WITHOUT reading it, stated
    explicitly because reading it is what would turn this case into the other one.
    """
    from local_operator.incidents import format_shape_incident_message

    text = format_shape_incident_message(
        "bash", ["dsn-password"], "kubectl exec api -- env", reached_model=False
    )
    assert "bash" in text
    assert "dsn-password" in text
    assert "no exposure" in text
    assert "rm -f" in text and "not to be done" in text
    assert "rotate" not in text
    assert "compromised" not in text
    assert "sh4pedSentinelPw" not in text


def test_the_escalated_notice_still_demands_a_rotation() -> None:
    """A value in the MODEL's context keeps the rotate-it severity, and says why.

    The one exposure this design cannot undo: the text the model reads is journaled
    in plain text, replays into later requests and may reach training data. The
    notice names that fact rather than the masking, because the fact is what the
    rotation decision turns on — and the default (no argument) is this case, so a
    caller that cannot classify cannot quietly file the quieter notice.
    """
    from local_operator.incidents import format_shape_incident_message

    text = format_shape_incident_message("bash", ["dsn-password"], "kubectl exec api -- env")
    assert "rotate" in text
    assert "context" in text
    assert "compromised" in text
    assert "sh4pedSentinelPw" not in text
    # One action, not two: a cleanup line here would dilute the sentence that
    # matters, and the copy on disk is the least of this case's problems.
    assert "rm -f" not in text


@pytest.mark.asyncio
async def test_a_contained_result_files_nothing_anywhere(tmp_path: Path) -> None:
    """End to end: masked whole means NO incident on any surface the operator sees.

    The three artefacts have to agree, and they are the whole of "indicated": the
    queued flag, the live receipt and the journaled row. Measured before this
    change: all three fired, for a value the model never read, which is the
    indicator the operator asked to be rid of. The masking itself is asserted in
    the same breath — an absence test that cannot tell "quiet" from "not running"
    is no evidence at all (AGENTS.md: a dead instrument returns a reading, not an
    error).
    """
    from local_operator.harness.types import NoticeEvent

    session = _session(tmp_path)
    session._pending_shape_incidents.clear()
    session._reported_shape_incidents.clear()
    events: list[Any] = []
    session.subscribe(events.append)

    assert SENTINEL_FRAGMENT not in session._redact_tool_result_text(SENTINEL)
    assert session._pending_shape_incidents == [], "a contained hit filed an incident"

    await session._flush_shape_incidents()

    assert not [
        event for event in events if isinstance(event, NoticeEvent)
    ], "a contained hit emitted a live receipt"
    # The transcript is written lazily — on a turn, not by the redaction hook —
    # so "no row" is asserted over whatever exists rather than over a file the
    # harness was never asked to create.
    transcript = tmp_path / "session" / "transcript.jsonl"
    body = transcript.read_text() if transcript.exists() else ""
    assert "session_incident" not in body, "a contained hit was journaled"
    # The instrument is not dead: the mask really is what the hook returned, and
    # that is what a reader of this session sees.
    assert REDACTION_MARKER in session._redact_tool_result_text(SENTINEL)


@pytest.mark.asyncio
async def test_an_exposure_with_no_contained_label_still_files_the_rotation(tmp_path: Path) -> None:
    """The case that used to be silent, and the one a labels-only gate would drop.

    A hit that left readable material has ``complete=False``, so it contributes no
    LABEL — the labels mean "masked whole". A notice gated on labels alone therefore
    goes quiet on the single event that must be raised, which is why the
    classification travels beside the labels and why ``report_shape_hits`` files an
    empty label list when the exposure flag is set.
    """
    from local_operator.harness.redaction import (
        report_shape_hits,
        reset_shape_hit_reporter,
        set_shape_hit_reporter,
    )
    from local_operator.harness.types import NoticeEvent

    session = _session(tmp_path)
    session._pending_shape_incidents.clear()
    session._reported_shape_incidents.clear()
    events: list[Any] = []
    session.subscribe(events.append)
    token = set_shape_hit_reporter(session._queue_shape_incident)
    try:
        report_shape_hits([], reached_model=True)
    finally:
        reset_shape_hit_reporter(token)
    assert [flag for _t, _l, _s, flag in session._pending_shape_incidents] == [True]

    await session._flush_shape_incidents()

    notices = [event for event in events if isinstance(event, NoticeEvent)]
    assert notices, "the exposure emitted no receipt"
    assert "rotate" in notices[0].text


def test_the_live_stream_files_only_an_exposure() -> None:
    """The pipe filter is a producer, and the sink's gate covers it too.

    It is the only layer that sees a credential existing ONLY in a command's
    output — the production case this whole feature was written for — and it masks
    the bytes before any result exists, so nothing later can report it. Its
    ordinary hit is CONTAINED (the bytes are masked before publication) and files
    nothing; an exposure inside the same stream still has to file, which is the
    second half of the test because a gate that swallows both is indistinguishable
    from a broken reporter.
    """
    from local_operator.harness.redaction import (
        reset_shape_hit_reporter,
        set_shape_hit_reporter,
    )

    def _feed(session: Any, payload: str) -> list[bool]:
        session._pending_shape_incidents.clear()
        session._reported_shape_incidents.clear()
        token = set_shape_hit_reporter(session._queue_shape_incident)
        try:
            redactor = builtin._PipeRedactor([])
            redactor.feed(payload.encode())
            redactor.feed(b"", final=True)
        finally:
            reset_shape_hit_reporter(token)
        return [flag for _t, _l, _s, flag in session._pending_shape_incidents]

    session = _session()
    assert _feed(session, SENTINEL) == [], "a contained stream hit filed an incident"
    assert _feed(session, _exposed_text()) == [True], "an exposure stopped being filed"


def test_a_value_the_pipe_masked_is_registered_for_the_rest_of_the_session() -> None:
    """The pipe is the ONE masking surface the store never sees raw text from.

    Agent review R1/E1. ``_PipeRedactor`` masks a running command's bytes, so the
    text the result path later hands ``VariableStore.redact_with_report`` already
    carries the marker: that pass matches nothing, which left the value contained
    for exactly one result and a later bare reuse of it — a spelling the shape
    table has no rule for — readable. Measured before the fix, on the shape this
    feature exists for (a credential in the command's OUTPUT only and nowhere in
    the command): ``registered values: 0`` and the reuse in the clear. The MASK
    never moved — only the registration was missing — so this test pins the
    registration and the reuse, not the mask.
    """
    # Assembled rather than written whole, so this file keeps spelling no
    # credential-shaped literal of its own (the same convention as SENTINEL
    # above). Fed in two halves so the pipe's release point is what completes the
    # line: registration has to survive the chunk boundary, not just one read.
    password = "unit" + "pipe" + "9x7"
    payload = f"MONGO_DSN=mongodb+srv://svc:{password}@db.invalid/x\n".encode()
    half = len(payload) // 2
    store = VariableStore(cwd=".")
    redactor = builtin._PipeRedactor([], contain=builtin._shape_containment_sink(store))
    masked = redactor.feed(payload[:half]) + redactor.feed(payload[half:])
    masked += redactor.feed(b"", final=True)
    assert REDACTION_MARKER in masked.decode(), "the pipe stopped masking its own hit"
    assert password in store.redaction_values(), "the pipe masked a value it did not contain"
    assert store.redact(f"prefix-{password}-suffix") == f"prefix-{REDACTION_MARKER}-suffix"


def test_a_pipe_redactor_with_no_store_still_masks_and_contains_nothing() -> None:
    """The historic single-argument construction: masking unchanged, no sink.

    ``contain`` is optional because a third-party embedder's store — and every
    bare tool test — constructs this filter with values alone. Those callers get
    exactly the behaviour they had before E1's fix: the bytes are masked and
    nothing is registered, which is a masking-only contract rather than a fault.
    """
    payload = f"MONGO_DSN=mongodb+srv://svc:{'unit' + 'pipe' + '9x7'}@db.invalid/x\n".encode()
    masked = builtin._PipeRedactor([]).feed(payload)
    assert REDACTION_MARKER in masked.decode()
    assert builtin._shape_containment_sink(object()) is None


def test_a_contained_report_cannot_file_by_any_route() -> None:
    """The gate is the SINK's, so every producer inherits it — including a new one.

    The policy has to hold for the result hook (tested above), for the pipe filter
    (tested above) and for anything that reports through
    ``harness.redaction.report_shape_hits``, which is the shape every other surface
    uses. Driving the sink directly is the point: a route that appears later cannot
    file a contained hit by forgetting a gate, and an exposure from the same tool is
    still filed exactly once.
    """
    from local_operator.harness.redaction import (
        report_shape_hits,
        reset_shape_hit_reporter,
        set_shape_hit_reporter,
    )

    session = _session()
    session._pending_shape_incidents.clear()
    session._reported_shape_incidents.clear()
    session._queue_shape_incident(["dsn-password"], False)
    assert session._pending_shape_incidents == [], "the sink filed a contained hit"

    token = set_shape_hit_reporter(session._queue_shape_incident)
    try:
        report_shape_hits(["dsn-password"], reached_model=False)
        assert session._pending_shape_incidents == [], "the reporter filed a contained hit"
        report_shape_hits([], reached_model=True)
        report_shape_hits([], reached_model=True)
    finally:
        reset_shape_hit_reporter(token)
    assert [flag for _t, _l, _s, flag in session._pending_shape_incidents] == [True]


def test_a_store_without_the_report_view_keeps_the_escalated_reading() -> None:
    """A store that cannot classify must not be read as a containment.

    ``VariableStore`` always reports, but ``_redact_tool_text`` composes with any
    store offering the older labels-only view (the broker work shipped separately).
    Taking that list as "nothing was exposed" would silently downgrade a real
    compromise to an informational notice, so the labels-only path keeps the
    escalated text it filed before the classification existed.
    """
    from local_operator.harness.redaction import (
        reset_shape_hit_reporter,
        set_shape_hit_reporter,
    )

    class _LabelsOnly:
        """The pre-classification surface: labels, and no way to say "exposed"."""

        def redact(self, text: str) -> str:
            return text.replace(SENTINEL_FRAGMENT, REDACTION_MARKER)

        def redact_with_hits(self, text: str) -> tuple[str, list[str]]:
            if SENTINEL_FRAGMENT not in text:
                return text, []
            return text.replace(SENTINEL_FRAGMENT, REDACTION_MARKER), ["dsn-password"]

    session = _session()
    session._pending_shape_incidents.clear()
    session._reported_shape_incidents.clear()
    # ``model_construct`` because ``ToolContext.variables`` is validated against the
    # store PROTOCOL, and the point of this stub is that it is NOT a
    # ``VariableStore``: it is the older surface, with the two methods the live path
    # looks for by name.
    context = ToolContext.model_construct(cwd=".", variables=_LabelsOnly())
    token = set_shape_hit_reporter(session._queue_shape_incident)
    try:
        assert SENTINEL_FRAGMENT not in builtin._redact_tool_text(SENTINEL, context)
    finally:
        reset_shape_hit_reporter(token)
    assert [flag for _t, _l, _s, flag in session._pending_shape_incidents] == [True]


def test_prose_after_a_flag_can_match_but_may_never_demand_a_rotation() -> None:
    """The other half of the measured misfire, pinned rather than declared away.

    ``--token was masked`` reads as a flag with a value, and the word after the flag
    is masked. That is an OVER-mask rather than an alarm, and it is left in place
    deliberately: the value is contained before the model sees it, so all it can
    produce is an informational notice — while a flag whose value really is a short
    word (``--password swordfish``) is a leak if the rule stops firing on
    word-shaped values. The asymmetry is what decides it: one masked English word
    costs a reader nothing, and the opposite mistake is unrecoverable.

    What it may never do is ask for a rotation, and that is asserted here because it
    is the thing the operator acts on.
    """
    from local_operator.incidents import format_shape_incident_message

    prose = "the " + "--" + "token" + " was masked before it reached bash"
    assert "cli-credential-flag" in match_shape_names(prose)

    notice = format_shape_incident_message(
        "bash", ["cli-credential-flag"], "cat WATCH.md", reached_model=False
    )
    assert "rotate" not in notice
    assert "no exposure" in notice


def test_a_flag_whose_value_is_a_name_is_not_a_credential() -> None:
    """The production misfire: a flag naming a stored secret.

    ``lop secret run --secret [redacted] -- <command>`` is the documented way to hand a
    stored secret to a child, and the token after that flag is the secret's NAME —
    the one thing an operator needs to be able to read. On 2026-09-19 a watch-log
    entry quoting that command was masked, and filed a rotation ticket in a
    production transcript for a credential that was not in the text at all. A value
    spelled as an environment variable AND ending in a credential word is now read
    as the reference it is.
    """
    # Joined from its segments so the literal never exists as a flag value in this
    # SOURCE: "flag followed by a value" is precisely the shape a redaction pass
    # rewrites, and this test's own text would otherwise be a casualty of it.
    name = "_".join(("OS", "PROD2", "ADMIN", "PASSWORD"))
    quoted = (
        "The correct mechanism was available and I did not use it: "
        f"`lop secret run --secret {name} -- <command>` and "
        "`lop secret file NAME -- <command>` keep the value inside the broker"
    )
    assert "cli-credential-flag" not in match_shape_names(quoted)
    assert scrub_shapes(quoted) == quoted

    # DRIVEN END TO END, not merely inspected. The ticket was filed by the session's
    # result hook over the whole watch-log entry, so the regression this test exists
    # for — an ESCALATED row for this text — can only be caught by driving that hook.
    # Asserting that the table is silent is a different assertion.
    entry = (
        "## INCIDENT — 2026-09-19 23:0x UTC — CREDENTIAL EXPOSURE TO BASH,"
        " ROTATION REQUIRED\n" + quoted + "\n"
    )
    session = _session()
    session._pending_shape_incidents.clear()
    session._reported_shape_incidents.clear()
    assert session._redact_tool_result_text(entry) == entry, "the entry was rewritten"
    assert session._pending_shape_incidents == [], "the entry filed an incident"

    # ...and a value that could be a credential is still masked, which is what the
    # corpus's own flag cases pin (they run over every surface above).
    assert "cli-credential-flag" in match_shape_names("server --token=" + "Sup3rTokenValue91")
