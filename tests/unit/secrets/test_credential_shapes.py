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
    AgentTool,
    LoopConfig,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    TextContent,
    ToolCall,
    ToolContext,
    ToolResult,
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
    COMPACT_TOKEN_PAIR,
    COUNT_QUALIFIER_NAMES,
    COUNT_TAIL_RELEASED_NAMES,
    COUNTER_USAGE_LINE,
    DUMP_COMMAND_CASES,
    FIXTURE_VALUE,
    NEGATIVE_CASES,
    POSITIVE_CASES,
    PRE_ESCAPED_LINE,
    TYPE_ANNOTATION_NEGATIVES,
    TYPE_ANNOTATION_POSITIVES,
    Case,
)

#: The angle brackets, assembled rather than spelled: this file is read by agents
#: THROUGH the pass it describes, so the spellings below are built from ordinals and
#: never appear as literals in the source.
_LT, _GT = chr(60), chr(62)

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


#: The corpus rows that make the MASK MARKER their own subject, named by REASON so
#: that a rename or a removal fails loudly instead of quietly checking nothing.
MASK_MARKER_NEGATIVE_REASONS: tuple[str, ...] = (
    "the mask marker alone: the pass's own output is not its own input",
    "the marker inside prose, in a credential position: a peer's evidence",
    "the marker inside a JSON tool argument: the shape a bash call journals",
)


def test_the_mask_marker_neither_masks_nor_labels_nor_escalates() -> None:
    """A message carrying the marker files no incident — the peer's report, pinned.

    The third of three "credential reached X" reports was this claim: a message that
    CONTAINS the mask marker re-fires a fresh incident, so "a detector whose output
    is its own input cannot settle". It is measured false, and the corpus could not
    settle it because the marker appeared in no row on either half; a claim that
    travels as prose is a claim the next session re-litigates, so it takes rows.

    What this test adds over the rows is the rest of the claim. The parametrised
    negative test asserts the BYTE half — the marker survives every surface — and
    the corpus digest pins the label and severity of what the table MATCHES, which
    for these rows is nothing. The three assertions below are stated together
    because they are one finding: no mask, no label, no escalation. ``reached_model``
    is derived from the hit list (:func:`shape_report`), so the empty hit list is
    what keeps the escalation half from being vacuously true — a rule that matched
    would file a hit, and the ``.npmrc`` ``_authToken=`` spelling is one of this
    table's own measured examples of a hit that files, labels, and still does not
    escalate — eight labels over the 23 corpus rows that re-fire, enumerated where
    the marker rows argue their own boundary in ``credential_shape_corpus``.
    """
    cases = [case for case in NEGATIVE_CASES if case.reason in MASK_MARKER_NEGATIVE_REASONS]
    missing = set(MASK_MARKER_NEGATIVE_REASONS) - {case.reason for case in cases}
    assert not missing, f"the marker rows were renamed or removed: {sorted(missing)}"

    for case in cases:
        # Anti-vacuity first: a row that no longer carries the marker would satisfy
        # every assertion below while pinning nothing at all.
        assert REDACTION_MARKER in case.text, f"{case.reason} no longer carries the marker"
        changed = [name for name, surface in SURFACES.items() if surface(case.text) != case.text]
        assert not changed, f"{case.reason} was rewritten on {changed}"
        assert match_shape_names(case.text) == [], f"{case.reason} was labelled"
        _, hits = scrub_shapes_with_hits(case.text)
        assert not hits, f"{case.reason} filed a hit"
        report = redaction_shapes.shape_report(hits)
        assert report.reached_model is False, f"{case.reason} escalated"


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


#: What the type clause is worth, as COUNTS rather than as a claim, and measured
#: the way the rest of this file measures a guard: neutralise the arm, count the rows
#: whose reading moves.
#:
#: Measured 2026-09-23 (``_corpus_grading()`` for the corpus under both modules). The
#: first is how many of ``TYPE_ANNOTATION_NEGATIVES`` come back MASKED when the arm
#: is removed — the rows that make the arm load-bearing at all; the second is how
#: many of ``TYPE_ANNOTATION_POSITIVES`` come back READABLE when the arm is widened
#: to everything — the rows that catch the clause over-reaching. Both are floors, not
#: equalities: adding rows may raise them, and a change that lowers one has taken the
#: arm's reach off a case that used to need it.
#:
#: The five negatives the first count does not include are the rows another rule
#: already releases (a bare primitive below the value floor, an annotation carrying
#: ``[``, a keyword argument in a function signature); they are in the corpus as
#: regression rows, and the count says so rather than leaving a reader to wonder.
#:
#: The SECOND count is an EQUALITY since agent review R1-1, and deliberately: its
#: number is the whole size of ``TYPE_ANNOTATION_POSITIVES``, because a wide arm must
#: release every one of them. It was ``8`` while the table held eight rows only
#: because a wide arm happened to release the eight the table happened to hold — the
#: assertion proved nothing about the rows it did not name. Pinning it to
#: ``len(TYPE_ANNOTATION_POSITIVES)`` means a row added to the table without a
#: discriminating reading FAILS here rather than sitting inert in the corpus.
_TYPE_CLAUSE_ROWS_THE_ARM_HOLDS = 15
#: Asserted below as ``len(TYPE_ANNOTATION_POSITIVES)``: every positive must be
#: released by a wide arm, so the row count IS the floor and no stale constant can
#: drift away from the table it measures.
_TYPE_CLAUSE_ROWS_THAT_CATCH_A_WIDE_ARM = len(TYPE_ANNOTATION_POSITIVES)

#: The spellings the confinement must MASK, and the arm each one is confined by.
#: They are the digit-carrying half of the residual class that had no row before
#: agent review R1-1, and they are asserted as VALUES rather than left to the corpus
#: because the corpus only proves the whole-line reading: this blocks SAYS which
#: condition of the documented release each spelling violates.
_CONFINEMENT_MUST_MASK: tuple[tuple[str, str], ...] = (
    ("Pass" + _LT + "Word" + "7", "a digit in the ARGUMENT"),
    ("Pass7" + _LT + "Word", "a digit in the BASE"),
    ("Abc" + _LT + "Xyz" + "1", "both capitals, a digit in the argument"),
    ("Correct" + "horse" + _LT + "Battery" + "7", "a digit, underscore removed"),
    ("foo" + _LT + "Bar" + _GT, "a LOWERCASE base under a generic"),
    ("abc" + "::" + "def" + _LT + "Bar" + _GT, "a lowercase-qualified leaf under a generic"),
    ("Option" + _LT + "Sha256" + _GT, "a digit inside a NON-primitive type name"),
    ("Pass" + _LT + "int" + _GT, "a credential-stem BASE over a bare primitive"),
    ("Pass" + _LT + "any" + _GT, "the same: ``any`` is English, not a type"),
    ("Secret" + _LT + "str" + _GT, "a credential WORD base over a primitive"),
    ("Token" + _LT + "void" + _GT, "the same with ``void``"),
    ("Sv" + "::" + "Secret", "a CamelCase MODULE segment in a bare path"),
    ("Camel" + "::" + "Word9", "the same convention violation with a digit"),
)


def test_the_type_annotation_clause_is_load_bearing_in_both_directions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Neither half of this corpus can pass without the arm, and each half proves it.

    This is the "prove the test can still fail" test for the type clause, and it
    exists because the clause is the kind of guard that is easy to believe without
    evidence: it sits inside a value predicate, it releases rather than masks, and a
    release is invisible unless something asserts the text survived.

    Three readings of the same rows:

    * IN PLACE: every negative comes back byte-identical, which is the fix;
    * REMOVED: the arm returns ``False`` and the rows that depend on it come back
      MASKED — so those rows are evidence for the arm and not for some other rule;
    * WIDENED: the arm returns ``True`` and the positives come back READABLE — the
      direction where the clause stops protecting a credential, which is the failure
      a reader of the corpus alone cannot see.
    """
    import local_operator.redaction_shapes as rs

    in_place = [
        case for case in TYPE_ANNOTATION_NEGATIVES if rs.scrub_shapes(case.text) == case.text
    ]
    assert len(in_place) == len(TYPE_ANNOTATION_NEGATIVES), [
        case.reason for case in TYPE_ANNOTATION_NEGATIVES if case not in in_place
    ]

    monkeypatch.setattr(rs, "_is_type_expression", lambda value: False)
    held = [case for case in TYPE_ANNOTATION_NEGATIVES if rs.scrub_shapes(case.text) != case.text]
    assert len(held) >= _TYPE_CLAUSE_ROWS_THE_ARM_HOLDS, [case.reason for case in held]

    monkeypatch.setattr(rs, "_is_type_expression", lambda value: True)
    released = [
        case for case in TYPE_ANNOTATION_POSITIVES if rs.scrub_shapes(case.text) == case.text
    ]
    assert len(released) == _TYPE_CLAUSE_ROWS_THAT_CATCH_A_WIDE_ARM, [
        case.reason for case in TYPE_ANNOTATION_POSITIVES if case not in released
    ]


def test_the_type_clause_masks_a_digit_carrying_spelling() -> None:
    """The confinement the module documents is the confinement it ENFORCES (R1-1).

    The documented release is the digit-free, symbol-free, single-token spelling
    (``Ident<Ident>``). The first implementation documented that and enforced only
    the symbol half: a digit on EITHER side released the value whole, with NO hit at
    all — a real credential released with nothing in the report to notice, which is
    the dangerous direction the corpus could not see because it pinned only the
    digit-free spelling.

    Each row below is a spelling the arm RELEASED before this round, measured on
    that revision, and each names the condition of the documented release it
    violates. They are asserted as VALUES rather than as corpus lines because this
    block is what says WHICH condition failed — a corpus row proves only that the
    whole line comes back unchanged.

    The last row is the price of the digit rule and is asserted the other way: a
    digit-carrying NON-primitive type name is now MASKED, so ``Option<Sha256>`` is
    the false positive this re-admits. That is the safe direction — masking a type
    is strictly better than releasing a credential — and pinning it here keeps the
    trade visible instead of letting a later change discover it.
    """
    import local_operator.redaction_shapes as rs

    for value, condition in _CONFINEMENT_MUST_MASK:
        assert rs._is_type_expression(value) is False, condition
        text = "DB" + "_PASSWORD=" + value
        assert rs.scrub_shapes(text) != text, f"released with no hit: {condition}"

    # The primitives are UNCHANGED by the digit rule: they are type names by fiat.
    for value in (
        "Option" + _LT + "Vec" + _LT + "u8" + _GT + _GT,
        "Vec" + _LT + "u8" + _GT,
        "Cow" + _LT + "'a" + _GT,
        "Arc" + _LT + "Mutex" + _LT + "T" + _GT + _GT,
        "Option" + _LT + "str" + _GT,
    ):
        assert rs._is_type_expression(value) is True, value
        assert rs._carries_a_non_primitive_digit(value) is False, value

    # A bare primitive is proven by a NON-credential base and refused by a
    # credential-stem one — the boundary R1-2 was about, in both directions.
    assert rs._is_type_expression("Vec" + _LT + "u8" + _GT) is True
    assert rs._is_type_expression("HashMap" + _LT + "u8,u8" + _GT) is True
    assert rs._is_type_expression("Pass" + _LT + "int" + _GT) is False
    assert rs._base_is_a_credential_stem("Pass") is True
    assert rs._base_is_a_credential_stem("Vec") is False
    assert rs._has_a_bare_primitive_argument("Vec" + _LT + "u8" + _GT) is True
    assert (
        rs._has_a_bare_primitive_argument("Option" + _LT + "Vec" + _LT + "u8" + _GT + _GT) is False
    )


def test_the_digit_rule_ignores_a_bare_integer_argument() -> None:
    """A const generic's integer is not a NAME, so the digit rule must not read it.

    ``ArrayVec<u8, 32>`` is a real annotation and its second argument is a bare
    integer, which belongs to no :data:`_TYPE_NAME_TOKEN`. A digit rule that flagged
    any digit in the value would mask every const-generic annotation — and one of
    those is a corpus row that ESCALATED before the fix, so the regression would be
    the exact incident this clause exists to remove.

    The const-generic spelling is NOT released by the arm (the parser reads ``32`` as
    a const argument and the corpus row is contained by another rule), so this test
    asserts the digit rule's own reading and the primitives' rather than claiming the
    row is arm-released. ``Sha256`` is the counter-example: a digit inside a NAME is
    flagged, which is what the enforcement is for.
    """
    import local_operator.redaction_shapes as rs

    assert rs._carries_a_non_primitive_digit("ArrayVec" + _LT + "u8, 32" + _GT) is False
    assert rs._carries_a_non_primitive_digit("u8") is False
    assert rs._carries_a_non_primitive_digit("f64") is False
    assert rs._carries_a_non_primitive_digit("Sha256") is True
    assert rs._carries_a_non_primitive_digit("Ident7") is True
    assert rs._carries_a_non_primitive_digit("Option" + _LT + "Sha256" + _GT) is True


def test_the_type_clause_needs_a_type_only_marker() -> None:
    """The three markers, and the spelling the clause must NOT reach.

    A type-only character — an angle bracket, a path separator, a reference — is what
    makes a value decidable as a type WITHOUT any allowlist of type names, which is
    what lets a custom annotation (``Option<SomeVeryLongTypeName>``) be released. The
    other half of this test is the load-bearing one: a bare word or a bare path is not
    decidable as a type, so ``API_KEY=averylonglowercasename`` keeps masking. A clause
    that released any long lowercase value would be a leak dressed as a fix, and this
    is the assertion that says so.
    """
    import local_operator.redaction_shapes as rs

    assert rs._is_type_expression("Option<String>") is True
    assert rs._is_type_expression("std::collections::HashMap") is True
    assert rs._is_type_expression("&SomeVeryLongEnumName") is True
    # ...and no marker at all is not a type, whatever it is spelled like.
    assert rs._is_type_expression("averylonglowercasename") is False
    assert rs._is_type_expression("public-catalogue-read") is False


def test_the_type_clauses_arguments_must_be_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The argument half of the clause, and the list it rests on.

    A generic application is only a type if what is INSIDE the angles is types. That
    rule is what keeps ``API_KEY=MyPass<secret>`` masked — the spelling a person
    reaches for when they write a passphrase with angle brackets in it — and it is
    carried by two things that can each silently stop working: the argument test, and
    ``_TYPE_PRIMITIVES``, which is what lets a lowercase argument (``u8``, ``str``,
    ``string``) prove itself a type without admitting every lowercase word.

    The primitive list being load-bearing is asserted rather than assumed: emptied,
    the same spelling stops being a type, so a future edit that trims the list loses
    ``Vec<u8>`` and fails here.
    """
    import local_operator.redaction_shapes as rs

    assert rs._is_type_expression("Vec<u8>") is True
    # A lifetime argument, in the spelling the value group actually delivers: the
    # annotation ``Cow<'a, str>`` is cut at the comma-space by the assigned-value
    # class, so the clause reads ``Cow<'a`` — the whole spelling never reaches it,
    # and a test that fed it the whole one would be asserting about dead input.
    assert rs._is_type_expression("Cow<'a") is True
    # A generic whose only argument is a const integer is not a type application, and
    # neither is one whose argument is a word.
    assert rs._is_type_expression("PassWord<1>") is False
    assert rs._is_type_expression("MyPass<secret>") is False

    monkeypatch.setattr(rs, "_TYPE_PRIMITIVES", frozenset())
    assert rs._is_type_expression("Vec<u8>") is False


def test_the_type_clauses_accepted_residual_is_a_row(monkeypatch: pytest.MonkeyPatch) -> None:
    """The one release this clause makes that is not a type annotation.

    ``API_KEY=Pass<Word>`` is a credential spelled exactly as ``Ident<Ident>``, and no
    spelling test separates that from a type application — a type name and a chosen
    password use the same alphabet. It is pinned as a corpus row rather than
    described in prose so a later change has to meet it, and the two assertions here
    are what make it a boundary of THIS clause: the row survives today, and it is
    released by this arm (removed, the same value masks).

    The boundary is now on BOTH sides, and this test asserts the widening that used
    to sit beside it: the residual is the digit-free, symbol-free, single-token
    spelling, so the digit-carrying sibling of the same row MASKS. Before the
    confinement was enforced that sibling was released whole, which is why a test
    that only asserted the residual would have passed while releasing credentials.
    """
    import local_operator.redaction_shapes as rs

    residual = [case for case in NEGATIVE_CASES if "Ident<Ident>" in case.reason]
    assert len(residual) == 1, [case.reason for case in residual]
    case = residual[0]
    assert rs.scrub_shapes(case.text) == case.text

    monkeypatch.setattr(rs, "_is_type_expression", lambda value: False)
    assert rs.scrub_shapes(case.text) != case.text, "the residual must be this arm's"

    # ...and the sibling that a wide arm used to take with it.
    monkeypatch.undo()
    sibling = "DB" + "_PASSWORD=" + "Pass" + _LT + "Word" + "7"
    assert rs.scrub_shapes(sibling) != sibling, "the digit-carrying sibling must MASK"


def test_an_escape_is_neither_a_name_character_nor_a_line_the_value_may_cross() -> None:
    """The rendering a tool call is JOURNALLED in is a surface, and it has escapes.

    Four claims, each measured, all of them about the same incident — a ``write`` of
    ordinary Python source whose file on disk holds no shape at all, whose arguments
    are scrubbed in their JSON spelling where every newline is the two characters
    ``\\`` and ``n``:

    * the escape's own letter is not part of the NAME, so the count trap still
      counts (the first block);
    * the escape's letters are detached only when they ARE an escape's — in either
      spelling of a literal backslash — so a literal backslash before a name does
      not eat the name (the second);
    * the escape is a line break for the JUDGEMENT, so an ordinary constant is not a
      13-character value (the first block again) — and the MASK may only ever cover
      more than the line the real spelling masks, never less (the third);
    * and the MASK, unlike the judgement, keeps every byte the rendering gave it: a
      value whose own bytes carry an escaped break is masked WHOLE, tail included
      (the fourth) — the direction agent review R1-2 measured the other way round.

    Kept as its own test rather than only corpus rows because the corpus pins the
    SPELLINGS while this pins the INVARIANTS they rest on — a future edit can satisfy
    every row by widening a rule somewhere else and still move this. Every assertion
    here discriminates against at least ONE of the two revisions under review (agent
    review R1-6, R2-F3), and which one it fails on is stated beside it: the count-trap
    block fails on ``origin/main``, while the literal-backslash block, the tail and
    the short-run assertions PASS there and fail only on ``5f757d9c``. Summing them
    into "each one fails on ``origin/main``" was a claim the measurements do not
    support.
    """
    # 1. The escape's letter is the newline's, not the name's: the count trap applies.
    # The prefix matters and is not decoration: a name the count trap covers is spared
    # by ``is_count_shaped``, which keys on the name's FIRST segment, so only a name
    # with something glued to its front can be a false positive here — and the only
    # thing that glues itself there is the escape letter of the line break before it.
    escaped_constructions = (
        PRE_ESCAPED_LINE
        + "MAX"
        + "_TOKENS = 4096"
        + "\\n" * 3
        + "def load_vendor_keys() -> dict[str, str]:",
        PRE_ESCAPED_LINE + "context_tokens=12345678" + "\\n" * 3 + "def run() -> None:",
        PRE_ESCAPED_LINE + "API" + "_KEY = PLACEHOLDER" + "\\n" * 3 + "def run() -> None:",
        # The same trap in the spellings ``json.dumps`` writes for a break it cannot
        # spell in two characters, which is what a payload carrying a raw U+2028
        # arrives as (agent review R1-4).
        PRE_ESCAPED_LINE.replace("\\n", "\\u2028")
        + "MAX"
        + "_TOKENS = 4096"
        + "\\u2028" * 3
        + "def load_vendor_keys() -> dict[str, str]:",
    )
    for text in escaped_constructions:
        assert scrub_shapes(text) == text, f"an escaped break invented a credential in {text!r}"
        assert match_shape_names(text) == []

    # 2. Only an ESCAPE's letters may be detached from a name. A literal backslash is
    # not one, and eating a letter made a credential name unreadable as a credential:
    # the mask was lost where ``origin/main`` kept it (agent review R1-3).
    literal_backslash = "\\" + "PASSWORD=" + "corr" + "ect_horse_bat" + "tery"
    assert "corr" + "ect_horse_bat" + "tery" not in scrub_secrets(literal_backslash)
    assert scrub_shapes(literal_backslash) != literal_backslash

    # 2b. ...and the DOUBLED spelling of the same literal backslash, which is how a
    # rendering writes one. The backslash before the name is itself escaped, so
    # nothing may be detached: R1-3's answer checked only that SOMETHING backslashish
    # preceded the name, and a name that IS a credential word lost its mask here with
    # no hit and nothing registered (agent review R2-F2). This is the half that fails
    # at the revision under review, where the block above passes there and fails on
    # ``5f757d9c``.
    escaped_literal_backslash = "\\\\" + "token=" + FIXTURE_VALUE
    assert FIXTURE_VALUE not in scrub_secrets(escaped_literal_backslash)
    assert scrub_shapes(escaped_literal_backslash) != escaped_literal_backslash

    # 3. A rendered break is a break for the JUDGEMENT, and the MASK may only ever
    # cover MORE than the line the real spelling masks — never less. The two
    # directions on one value: on a REAL break the line after it is a line like any
    # other and stays readable; in the rendering the same assignment is masked whole.
    carried = (
        "CLIENT" + "_SECRET=" + "alpha" + "_run" + "_body" + "\\n" + "more" + "_body" + "_material"
    )
    on_a_real_break = carried.replace("\\n", "\n")
    real_line = "alpha" + "_run" + "_body"
    tail = "more" + "_body" + "_material"
    assert real_line not in scrub_shapes(on_a_real_break), "both surfaces mask the value"
    assert tail in scrub_shapes(on_a_real_break), "the real spelling keeps the next line"

    # 4. ...and it is NOT a break for the MASK: the rendering masks the WHOLE run,
    # where the revision under review left the tail readable under a hit still graded
    # ``complete=True``.
    scrubbed = scrub_secrets(carried)
    assert real_line not in scrubbed, "the key must still be masked"
    assert tail not in scrubbed, "the tail stayed readable"
    assert scrubbed == "CLIENT" + "_SECRET=[redacted]"

    short_run = "CLIENT" + "_SECRET=" + "run" + ">" + "\\n" + "zip" + "tail" + "material"
    scrubbed = scrub_secrets(short_run)
    assert "zip" + "tail" + "material" not in scrubbed, "a short run still crosses the break"

    # The escape is not the name's, so an assignment that begins right after one is
    # still an assignment — judged on the name the escape actually belongs to.
    after_an_escape = PRE_ESCAPED_LINE + "OPENROUTER_API_KEY=" + "QA-fixture-9c1f4a"
    scrubbed = scrub_secrets(after_an_escape)
    assert "QA-fixture-9c1f4a" not in scrubbed
    assert scrubbed.startswith(PRE_ESCAPED_LINE), "the escaped rendering must round-trip"


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

    The credential assertion comes FIRST, and deliberately: on the pre-fix source
    this row has to fail for the reason it exists — the value being published —
    and not because ``_PIPE_HOLD_LIMIT`` does not exist there. A test whose only
    pre-fix failure is a missing symbol pins the seam rather than the property.
    """
    line = "." * 4095 + SENTINEL + "." * (4 * 1024 * 1024)
    redactor = builtin._PipeRedactor([])
    peak = 0
    published = 0
    chunks: list[bytes] = []
    for start in range(0, len(line), 4096):
        chunk = redactor.feed(line[start : start + 4096].encode())
        chunks.append(chunk)
        published += len(chunk)
        peak = max(peak, len(redactor.pending))
    tail = redactor.feed(b"", final=True)
    chunks.append(tail)
    published += len(tail)
    assert (
        SENTINEL_FRAGMENT not in b"".join(chunks).decode()
    ), "the credential straddling the boundary was published"
    assert peak <= builtin._PIPE_HOLD_LIMIT, "the hold must not grow with the child's output"
    assert peak > builtin._PIPE_DEFERRAL_LIMIT, "the boundary hold was never exercised"
    assert redactor.pending == ""
    assert published >= len(line) - builtin._PIPE_HOLD_LIMIT, "the line must still stream"


def test_the_hold_is_the_max_of_two_rules_and_stays_bounded() -> None:
    """``_PIPE_HOLD_LIMIT`` bounds the SHAPE rule, not the filter's whole hold.

    The older KNOWN-value rule holds whatever a registered value needs, because a
    registered value is a credential the session was told about and publishing it
    in two halves is the leak that rule exists to prevent. So the filter's total
    hold is ``max(_PIPE_HOLD_LIMIT, len(value) + _PIPE_DEFERRAL_LIMIT)`` — a bound
    over what the SESSION knows, never over what the child prints, which is the
    property the cap is for. Asserted with a NON-EMPTY secret on purpose: an empty
    one exercises only the first term, which is how this limit came to be
    documented as the whole bound in the first place.

    ``getattr`` because this row is a guard on the filter as a whole rather than a
    discriminator for this change — the rule it measures predates it and the
    numbers are the same on the pre-fix source, so it must not fail there for a
    missing symbol.
    """
    limit = getattr(builtin, "_PIPE_HOLD_LIMIT", builtin._PIPE_DEFERRAL_LIMIT)
    worst = 0
    for size, read in ((1_000, 4096), (5_000, 4096), (24_576, 65536), (30_000, 65536)):
        secret = ("q7Xk2m" * (size // 6 + 1))[:size]
        line = "." * 100_000 + secret + "." * 100_000
        raw = line.encode()
        redactor = builtin._PipeRedactor([secret])
        published: list[bytes] = []
        peak = 0
        for start in range(0, len(raw), read):
            published.append(redactor.feed(raw[start : start + read]))
            peak = max(peak, len(redactor.pending))
        published.append(redactor.feed(b"", final=True))
        assert (
            secret not in b"".join(published).decode()
        ), f"a registered value of {size} B was published at {read} B reads"
        assert peak <= max(
            limit, size + builtin._PIPE_DEFERRAL_LIMIT
        ), f"holding a {size} B value at {read} B reads took {peak} B"
        worst = max(worst, peak)
    assert (
        worst > limit
    ), "the second term must actually bind at some size, or this test measures the first one twice"


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


@pytest.mark.asyncio
async def test_the_advisory_survives_a_long_result_and_is_not_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The notice must not be the first thing the 40-line head crop drops.

    This used to pin the INSERT SPELLING — ``"parts.insert(1, notice)" in the
    module source`` — and round 2's Q5 fix broke it by computing the index from
    the exit-code line instead (the TIMEOUT head is inserted at 0 before the
    advisories, so a literal index put them ABOVE ``exit code:`` on that path).
    A source-text pin cannot tell a repositioned notice from an equivalent one,
    so it is a behavioural row now: a command that dumps a credential-shaped
    line and then hundreds of lines of output must still carry the notice inside
    the head window a card keeps, UNDER the exit code. That is the property the
    old assertion stood in for, and it fails for the reason that matters — the
    notice being at the tail — rather than on a rename.
    """
    from local_operator.harness.types import AbortSignal, ToolContext
    from local_operator.tools import builtin
    from local_operator.variables import VariableStore

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    context = ToolContext(
        cwd=str(tmp_path),
        variables=VariableStore(cwd=str(tmp_path)),
        session_id="advisory-head",
    )
    # ``env`` is credential-shaped, and the loop puts the notice's position under
    # the same crop a real 200-line result would.
    command = 'env; for i in $(seq 1 200); do echo "line $i of the report"; done'
    result = await builtin.execute_bash(
        "cred-long", {"command": command}, AbortSignal(), None, context
    )

    lines = result.text.splitlines()
    advisory = next((index for index, line in enumerate(lines) if "credential guard" in line), None)
    assert advisory is not None, result.text[:400]
    # Inside the head window, which is what "not the first thing dropped" means,
    # and below the exit code, so the shape is the same on every path.
    assert advisory < 5, lines[:8]
    assert lines[0].startswith("exit code: "), lines[:3]


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


# --- the pipe's OPEN BLOCK: nothing inside it is published -------------------
#
# The three mechanisms that defeated this state, each measured at the revision
# before this section existed, and each one a REGRESSION ARM below:
#
# 1. the header's own terminator was offered to the line loop, whose prose test
#    read a bare separator as prose and CLOSED the block one line into itself —
#    every later release then had no header to reopen it. Measured: 143 body
#    lines (the whole 8 KiB ``pending``) from PR #1427's case, one call.
# 2. a 512-line bound released the body verbatim past it — measured 1,488 body
#    lines of a 2,000-line block.
# 3. a cap-forced cut could land INSIDE a line, and the fragment is what the loop
#    classified: a four-character fragment is below the body grammar's floor, so
#    it read as prose and closed the block — measured 1,092 body lines published
#    on the next release.
#
# The outcome all three reach is the same publish, and it is also reachable when
# RETENTION drops one marker line from a >cap stream: the settled shape pass
# cannot repair it because ``pem-private-key`` spans BEGIN to END. That route
# measured 1,420 raw body lines in the call-site spill of a >4 MiB stream through
# the real tool, served over ``read spill://``.
#
# The armour below is built from PARTS on purpose: these tests are about a
# literal's LENGTH as much as its spelling, and a display filter that rewrote the
# dashes would leave every arm here passing vacuously. ``_PEM_GRAMMAR_IS_LIVE``
# is asserted first by every arm that depends on it.

_PEM_DASHES = "-" * 5
_PEM_HEADER = f"{_PEM_DASHES}BEGIN RSA PRIVATE KEY{_PEM_DASHES}\n"
_PEM_END = f"{_PEM_DASHES}END RSA PRIVATE KEY{_PEM_DASHES}\n"
#: A body line at a real PEM's width, and UNIQUE per line so a publish is
#: countable and locatable rather than merely detectable.
_PEM_BODY_STEM = "MIIEowIBAAKCAQEA" + "bKdFgHjLmNpQrStUvWxYzAbCdEfGhJkLmNoP"


def _body_lines(count: int) -> str:
    return "".join(f"{_PEM_BODY_STEM}{index:04d}\n" for index in range(count))


def _published_body_lines(text: str) -> list[str]:
    """Every body line still readable in ``text`` — the property under test."""
    return [line for line in text.splitlines() if line.startswith(_PEM_BODY_STEM)]


def _pem_grammar_is_live() -> None:
    """Fail LOUDLY if an armour literal stopped matching the classifiers.

    A rewritten header is not a masked one: it publishes the whole body, so an arm
    whose header no longer matches would assert the wrong thing about the wrong text.
    """
    assert builtin._PEM_HEADER_LINE.match(_PEM_HEADER.rstrip("\n")), "header literal is not PEM"
    assert builtin._PEM_END_LINE.match(_PEM_END.rstrip("\n")), "END literal is not PEM"
    assert builtin._PEM_BODY_LINE.match(_PEM_BODY_STEM), "body line is not PEM-shaped"


def test_an_unterminated_block_past_the_deferral_limit_publishes_no_body() -> None:
    """PR #1427's case, through the pipe: header + 200 body lines and no END.

    One call, and the whole 8 KiB ``pending`` used to go out verbatim — 143 body
    lines of a 57-byte body, or 31-32 of a 256-byte one: the byte budget is what
    is fixed, and the body width only chooses how many lines it buys.
    """
    _pem_grammar_is_live()
    payload = _PEM_HEADER + _body_lines(200)
    published = _pipe_whole(payload)

    published_lines = _published_body_lines(published)
    assert not published_lines, f"{len(published_lines)} body lines of an open block went out"
    assert REDACTION_MARKER in published, "the block was not masked at all"


def test_a_chunk_boundary_just_after_the_header_keeps_the_block_open() -> None:
    """The header in one release and its body in the next: the block must stay open.

    This is the separator bug at its narrowest. The header's own terminator used to
    be the line loop's first element, where the prose test closed the block — so a
    read boundary that lands after the header (any slow child that prints the
    header first) published everything after it.
    """
    _pem_grammar_is_live()
    redactor = builtin._PipeRedactor([])
    published = redactor.feed(_PEM_HEADER.encode())
    published += redactor.feed(_body_lines(200).encode())
    published += redactor.feed(b"", final=True)
    text = published.decode()

    assert not _published_body_lines(
        text
    ), f"{len(_published_body_lines(text))} body lines were published across the boundary"
    assert REDACTION_MARKER in text, "the block was not masked at all"


def test_a_block_longer_than_the_old_line_bound_is_masked_to_its_end() -> None:
    """The bound that released the body: 1,488 of these 2,000 lines used to go out raw.

    An unterminated block cannot hold memory — a masked line is DROPPED, and
    ``pending`` is capped separately — so the state is not bounded by lines any more.
    """
    _pem_grammar_is_live()
    published = _pipe_whole(_PEM_HEADER + _body_lines(2000))

    assert not _published_body_lines(
        published
    ), f"{len(_published_body_lines(published))} body lines past the old line bound"


def test_a_forced_cut_inside_a_line_is_held_to_the_line_boundary() -> None:
    """A fragment of a line cannot be classified, so it must not be classified.

    The cap can cut mid-line, and a fragment shorter than the body grammar's floor
    (``PEM_BODY_FLOOR``) is read as PROSE by the loop — which CLOSES an open block.
    Measured: the four-character fragment ``MIIE`` closed one, and the next release
    (all body, no header left) published 1,092 lines of a 2,000-line block.
    """
    _pem_grammar_is_live()
    redactor = builtin._PipeRedactor([])
    # 400 lines is past the deferral limit, so the cap decides the cut. The tail
    # length is SEARCHED, not guessed: which byte the cap lands on depends on the
    # line width, and the case only exists when the remainder is shorter than the
    # grammar's floor. A guessed length that happens to land at a line boundary
    # would assert nothing while looking like it did.
    # The floor is the code's own name for it, with the historical value as the
    # fallback: an arm run against a tree that predates the shared constant still
    # tests the MECHANISM (where the cut lands) instead of failing on the lookup.
    floor = getattr(builtin, "PEM_BODY_FLOOR", 8)
    for tail in range(1, 60):
        text = _PEM_HEADER + _body_lines(400) + "M" * tail
        cap_cut = len(text) - builtin._PIPE_DEFERRAL_LIMIT
        line_start = max(text.rfind("\n", 0, cap_cut), text.rfind("\r", 0, cap_cut)) + 1
        fragment = cap_cut - line_start
        if 0 < fragment < floor:
            break
    else:  # pragma: no cover - the fixture could not reach the case
        pytest.fail("no tail length puts a sub-floor fragment at the cap's cut")

    cut = redactor._release_point(text, final=False)

    assert cut == line_start, (
        f"the release ends {cut - line_start} bytes into a line, where the classifier "
        "reads the fragment as prose (and closes the block)"
    )


def test_a_terminated_block_is_masked_whole_and_its_end_line_closes_the_state() -> None:
    """The preservation arm: the fix must not start eating real key blocks.

    Asserted at the MASK rather than on the pipe's whole output, and that is
    deliberate: for a complete block the shape table ALSO masks the span from the
    header to the terminator, so a product-level assertion here would be satisfied by
    either layer and could not be reddened by breaking one of them. The mask's own
    contract is the narrower, provable one — body gone, END line out, state closed —
    and `pem-private-key` is already pinned elsewhere as the second layer.
    """
    _pem_grammar_is_live()
    redactor = builtin._PipeRedactor([])
    masked = redactor._mask_open_key_block(_PEM_HEADER + _body_lines(3) + _PEM_END)

    assert not _published_body_lines(masked), "a complete block published its body"
    assert _PEM_END.rstrip("\n") in masked, "the END line was eaten"
    assert redactor._in_key_block is False, "the END line did not close the block"
    # And the product-level outcome for the same text, which is the marker.
    assert REDACTION_MARKER in _pipe_whole(_PEM_HEADER + _body_lines(400) + _PEM_END)


def test_prose_after_a_stray_header_is_released_and_closes_the_block() -> None:
    """The over-mask guard, unchanged by this fix: prose is not key material.

    A header quoted in a doc or matched by ``grep`` opens the state, and the FIRST
    ordinary line closes it — so the fixture text after it stays readable.
    """
    _pem_grammar_is_live()
    published = _pipe_whole(_PEM_HEADER + "ordinary prose line\n" + _PEM_BODY_STEM + "0000\n")

    assert "ordinary prose line" in published, "prose was eaten by a stray header"
    # The one body-shaped line after the prose is masked: a close is not a licence
    # to publish, and this arm pins the direction the layer must fail in.
    assert not _published_body_lines(published)


def test_a_numeric_table_without_a_header_is_left_alone() -> None:
    """The negative half: a table of numbers is output a human has to read.

    ``10000000 10000001`` IS body-shaped — the line grammar reads the first run as a
    line-number prefix, which is how the old ambiguous ``LINE_PREFIX`` cost seconds
    on exactly this row (PR #1427). What keeps such a table readable is that NO
    HEADER is open: the state is the only thing that makes a body-shaped line a body
    line. The plausible bug this pins is the one that masks body-shaped lines with no
    header at all, which is a single mutation away in the loop below. (Inside an open
    block these rows ARE masked: an over-mask is the direction this layer must fail
    in, and it is recorded in the module comment.)
    """
    _pem_grammar_is_live()
    table = "10000000 10000001\n" * 20
    published = _pipe_whole(table)

    assert published == table, "a table with no header was rewritten by the pipe filter"


def _pipe_chunks(raw: bytes, chunk: int = 16384) -> str:
    """The pipe filter over a WHOLE stream, in chunks, as ``_pump`` drives it."""
    redactor = builtin._PipeRedactor([])
    published = [
        redactor.feed(raw[index : index + chunk]).decode() for index in range(0, len(raw), chunk)
    ]
    published.append(redactor.feed(b"", final=True).decode())
    return "".join(published)


def _retention_stream(cap: int, variant: str, block_lines: int) -> tuple[bytes, int]:
    """A >cap stream whose retention window drops exactly ONE marker line.

    Two things here are deliberate. The feed is CHUNKED, like the product's: a
    single call would hand the shape pass a complete BEGIN … END and it would mask
    the whole block, so there would be no fragment to measure and no case to test.
    And the layout is SEARCHED against the pipe's own output rather than computed
    from the raw bytes, because the masked length of the block is exactly what
    differs between the revision that leaks and the revision that does not — a
    layout derived from the raw stream would place the marker differently on each
    and the arm would prove nothing. Returns the raw stream and the window size.
    """
    marker = "BEGIN" if variant == "begin" else "END"
    marker_line = _PEM_HEADER if variant == "begin" else _PEM_END
    pad_line = "filler " + "x" * 40 + "\n"
    target = cap // 2
    window = 512

    def prose(total: int) -> str:
        """At least ``total`` bytes, always ending on a line break (a glued header is
        not the line-anchored spelling the mask looks for)."""
        return pad_line * (-(-total // len(pad_line)))

    def build(pre: int, post: int) -> bytes:
        block = _PEM_HEADER + _body_lines(block_lines) + _PEM_END
        return (prose(max(pre, 0)) + block + prose(max(post, 0))).encode()

    pre = target + 64 if variant == "begin" else target - 64
    for _ in range(30):
        measured = _pipe_chunks(build(pre, window * 4))
        at = measured.find(marker)
        if at < 0:  # the layout put the block inside a masked run: shift and retry
            pre += 4096
            continue
        line_start = measured.rfind("\n", 0, at) + 1
        if target <= line_start < target + window - len(marker_line):
            break
        pre += max(min(target - line_start, 4096), -4096)
    else:  # pragma: no cover - the search could not place the marker
        pytest.fail("could not place the marker line inside the retention window")

    # Size the tail so the omitted window is the width this arm needs. The tail
    # moves in whole pad lines, so the window lands in [window, window + a line):
    # the assertion is that band, and the ACTUAL width is what the arm compares
    # against, rather than a number the fixture would have to hit exactly.
    post = window * 4 + window - len(measured)
    raw = build(pre, post)
    omitted = len(measured) - cap
    for _ in range(4):
        raw = build(pre, post)
        measured = _pipe_chunks(raw)
        omitted = len(measured) - cap
        if window <= omitted < window + len(pad_line):
            break
        post += window - omitted
    assert window <= omitted < window + len(pad_line), f"the retention window is {omitted} bytes"
    line_start = measured.rfind("\n", 0, measured.find(marker)) + 1
    assert (
        0 <= line_start - target < omitted - len(marker_line)
    ), "the marker line is not inside the omitted window"
    return raw, omitted


@pytest.mark.parametrize("variant", ["begin", "end"])
def test_a_retention_split_marker_still_publishes_no_body(variant: str) -> None:
    """Retention drops ONE marker line; nothing in the body may reach any surface.

    This is the route a reviewer found independently, from the retention side, and it
    is the one the settled pass cannot repair: the operator's copy is built from the
    retained text, and ``pem-private-key`` needs a complete BEGIN … END, so a fragment
    matches no rule at all. Measured at the base: 1,420 raw body lines in the call-site
    spill of a >4 MiB stream through the real tool, served over ``read spill://``.
    """
    _pem_grammar_is_live()
    cap = 65536
    raw, window = _retention_stream(cap, variant, 700)

    sink = builtin._BashOutput(limit=cap)
    redactor = builtin._PipeRedactor([])
    live = []
    for index in range(0, len(raw), 16384):
        piece = redactor.feed(raw[index : index + 16384])
        sink.append(piece)
        live.append(piece.decode())
    piece = redactor.feed(b"", final=True)
    sink.append(piece)
    live.append(piece.decode())

    assert sink.omitted_bytes == window, "retention did not drop the marker line"
    for surface, text in (
        ("the live pipe", "".join(live)),
        ("the retained copy", sink.decode()),
        ("the settled pass", _live_text(sink.decode())),
    ):
        published = _published_body_lines(text)
        assert not published, f"{surface} published {len(published)} body lines of a split block"


def test_a_block_truncated_by_the_cap_still_publishes_no_body() -> None:
    """A block whose body is cut off mid-line by the cap, with no END at all.

    The cap is the mechanism that releases a block's middle, and the fragment it
    leaves is the one the loop used to read as prose; this is that shape end to
    end, through the pipe, at a small forced deferral.
    """
    _pem_grammar_is_live()
    original = builtin._PIPE_DEFERRAL_LIMIT
    try:
        builtin._PIPE_DEFERRAL_LIMIT = 256
        redactor = builtin._PipeRedactor([])
        # Chunks of 100 bytes: the cut lands inside lines repeatedly, and the
        # third line's remainder is a sub-floor fragment.
        payload = (_PEM_HEADER + _body_lines(60)).encode()
        published = b""
        for index in range(0, len(payload), 100):
            published += redactor.feed(payload[index : index + 100])
        published += redactor.feed(b"", final=True)
    finally:
        builtin._PIPE_DEFERRAL_LIMIT = original

    text = published.decode()
    assert not _published_body_lines(
        text
    ), f"{len(_published_body_lines(text))} body lines escaped a cap-truncated block"


def test_an_unterminated_block_with_a_second_block_present_publishes_no_body() -> None:
    """An END dropped while ANOTHER block follows: the first must not leak either.

    ``_release_point``'s hold looks for the NEXT terminator, so a second block's END
    can satisfy the first block's search. Whatever that does to the deferral, the
    body of the unterminated one must not reach the pipe's output.
    """
    _pem_grammar_is_live()
    # 700 lines in the unterminated block, not 300: below the old 512-line bound
    # this arm would have passed at the revision that published, which is how a
    # boundary arm becomes a vacuous one.
    payload = _PEM_HEADER + _body_lines(700) + _PEM_HEADER + _body_lines(200) + _PEM_END
    # CHUNK-fed: in one call the shape pass would match BEGIN(1) … END(2) as a single
    # span and mask the lot, so the arm would pass at the revision that published.
    published = _pipe_chunks(payload.encode())

    published_lines = _published_body_lines(published)
    # The second block is terminated, so ITS body is masked by the settled pass as
    # well; what this arm pins is that no body line survives the pipe.
    assert not published_lines, f"{len(published_lines)} body lines were published"
    assert REDACTION_MARKER in published


# --- the round-2 remediation: the cap's OWN split, and the armour's spellings -----
#
# Three holes round 1 found in the section above, all measured on the revision before
# these arms existed, and all three in the SAME publish direction:
#
# 1. the cap can land inside the ARMOUR LINE. The hold above finds a block by its
#    opening marker, but the cap is applied after that hold and recomputes the cut
#    from the buffer length, so the marker itself is what gets split: a bare marker
#    fragment goes out, the rest of the marker stays in ``pending`` where its line no
#    longer STARTS with it, and the classifier can never match it again. The state
#    never opens, so the whole flush is body nothing masks — measured 138 raw body
#    lines, and the alignment is fixed per stream (a repeated read leaks every time).
# 2. an armour line ending in ``\r`` / ``\r\n`` / trailing whitespace never opened the
#    block at all: ``$`` cannot match in front of a ``\r``, so an ORDINARY key file
#    written on Windows (or quoted by a wiki, or left with a trailing space by an
#    editor) published its body — measured 5 of 25 lines for ``head -n 6`` on a
#    complete CRLF key, and 200 lines of an unterminated one on every surface.
# 3. a second block's header arriving in the SAME read as the first block's END was
#    released without re-opening the state, so the second block's body went out raw —
#    measured 138 lines on the shape below.
#
# The armour literals in this section come from the shared ones above (``_PEM_HEADER``,
# ``_PEM_END``, ``_PEM_DASHES``), which are assembled from parts for the same reason.

#: The line-break spellings an armour line arrives with in the wild: LF, CRLF, a bare CR
#: (a key that came off an old Mac, or through a filter that normalised to CR), and the
#: two trailing-whitespace forms a copy-paste or an editor leaves behind.
_ARMOUR_BREAKS = (
    ("lf", "\n", "\n"),
    ("crlf", "\r\n", "\r\n"),
    ("cr", "\r", "\r"),
    ("trailing-spaces-lf", "   \n", "\n"),
    ("trailing-tab-crlf", "\t\r\n", "\r\n"),
    ("trailing-space-cr", " \r", "\r"),
)


def test_a_cap_that_splits_the_armour_line_publishes_no_body() -> None:
    """The cap's cut landing INSIDE the armour line is the one split the mask cannot survive.

    The hold in ``_release_point`` defers a block by its opening marker, and the cap is
    applied AFTER that hold and recomputes the cut from the buffer length — so an
    alignment that puts the cap a few bytes into the armour line publishes a marker
    FRAGMENT and leaves the rest of the marker in ``pending``, where the line no longer
    starts with it. From then on the classifier cannot match it, the state never opens,
    and every later release is body that nothing masks: 138 raw body lines measured at
    this exact alignment, identical at the base tree.

    WHAT DECIDES IT IS THE ALIGNMENT, NOT THE READ SIZE — which is why the sweep is over
    both: the same payload fed whole, in 4 KiB reads, in 1 KiB reads and in 100 B reads
    all published the body before the fix, and a session that runs the same truncated
    read twice leaks twice, because the alignment is fixed per stream.
    """
    _pem_grammar_is_live()
    # The sweep is DERIVED from the cap, not hard-coded: the cut lands ``limit`` bytes
    # before the end, so a total that puts it 8-27 bytes into a 32-byte armour line is
    # ``limit + 8`` to ``limit + 27``. Width-checked, because a wider armour literal
    # would move the window and quietly turn this arm into one that sweeps nothing.
    assert len(_PEM_HEADER) == 32, "the armour literal changed width; re-derive the sweep"
    limit = builtin._PIPE_DEFERRAL_LIMIT
    for size in (0, 100, 1024, 4096):  # 0 is one feed + the flush
        for total in range(limit + 8, limit + 28):
            raw = (_PEM_HEADER + _body_lines(300))[:total].encode()
            published = _pipe_whole(raw.decode()) if size == 0 else _pipe_chunks(raw, size)
            leaked = _published_body_lines(published)
            assert not leaked, (
                f"a {'whole-stream' if size == 0 else f'{size} B'} feed of {total} B "
                f"published {len(leaked)} body lines of a block whose marker the cap split"
            )


def test_a_second_block_after_a_terminated_one_publishes_no_body() -> None:
    """A terminated block AND a second header in ONE read: the state has to be RE-opened.

    The second-header arm treats a header inside an open block as armour rather than
    prose, but this loop reaches that arm with the state already CLOSED whenever the same
    release carried the earlier block's END. Releasing the header without re-opening it
    left the second block's body to go out raw on the next release — measured 138 raw
    body lines at this shape (12,071 B, one read), the whole flush.
    """
    _pem_grammar_is_live()
    payload = _PEM_HEADER + _body_lines(3) + _PEM_END + _PEM_HEADER + _body_lines(200)

    redactor = builtin._PipeRedactor([])
    redactor.feed(payload.encode())
    assert redactor._in_key_block is True, "the second block's header did not re-open the state"

    published = _pipe_whole(payload)
    leaked = _published_body_lines(published)
    assert not leaked, f"{len(leaked)} body lines of the second block were published"


@pytest.mark.parametrize(
    ("name", "armour_break", "body_break"),
    _ARMOUR_BREAKS,
    ids=[case[0] for case in _ARMOUR_BREAKS],
)
def test_an_armour_line_with_a_cr_or_a_trailing_space_is_still_armour(
    name: str, armour_break: str, body_break: str
) -> None:
    """The armour line's own terminator spelling must not decide whether a key is masked.

    ``$`` cannot match in front of a ``\\r``, so the classifier did not see
    ``...KEY-----\\r\\n`` (or ``...\\r``, or the trailing-whitespace forms) as armour at
    all, and an unterminated block then went out in the clear: 200 of 200 body lines on
    the transcript, the raw spill file and ``read spill://``. Base and head were
    identical there, so this is pre-existing — but it is the branch's own title claim
    unmet for an ORDINARY spelling, and the pipe is the only layer that can hide an
    unterminated view (``pem-private-key`` needs a complete BEGIN … END). Every surface
    below is built from the pipe's bytes, so the pipe's output is the first thing to pin
    and the store's copy and the settled pass follow it.
    """
    _pem_grammar_is_live()
    armour = _PEM_HEADER.rstrip("\n") + armour_break
    # (1) an UNTERMINATED block, 200 lines, which is the shape the retention route leaks.
    unterminated = armour + _body_lines(200).replace("\n", body_break)
    published = _pipe_whole(unterminated)
    leaked = _published_body_lines(published)
    assert not leaked, f"{name}: {len(leaked)} body lines of an unterminated block went out"

    # (2) the same bytes through the retention route: the store's retained copy and the
    # settled pass over it, neither of which can repair a headerless fragment.
    sink = builtin._BashOutput(limit=65536)
    redactor = builtin._PipeRedactor([])
    raw = unterminated.encode()
    for index in range(0, len(raw), 4096):
        sink.append(redactor.feed(raw[index : index + 4096]))
    sink.append(redactor.feed(b"", final=True))
    for surface, text in (
        ("the retained copy", sink.decode()),
        ("the settled pass", _live_text(sink.decode())),
    ):
        leaked = _published_body_lines(text)
        assert not leaked, f"{name}: {surface} published {len(leaked)} body lines"

    # (3) ``head -n 6`` on a COMPLETE key of this spelling: five body lines, no END in the
    # text at all, so nothing downstream can repair it. Measured 5 of 25 published before.
    truncated = armour + _body_lines(5).replace("\n", body_break)
    leaked = _published_body_lines(_pipe_whole(truncated))
    assert not leaked, f"{name}: {len(leaked)} body lines of a truncated complete key went out"


def test_a_complete_key_with_crlf_endings_is_masked_to_its_end() -> None:
    """The whole block path for the CRLF spelling: body masked, END out, state closed.

    Asserted at the MASK rather than on the pipe's whole output, because for a COMPLETE
    block the shape table masks the span as well and a product-level assertion would be
    satisfied by either layer (the same reason the LF arm above is written this way).
    """
    _pem_grammar_is_live()
    block = _PEM_HEADER.rstrip("\n") + "\r\n" + _body_lines(3).replace("\n", "\r\n")
    block += _PEM_END.rstrip("\n") + "\r\n"
    redactor = builtin._PipeRedactor([])
    masked = redactor._mask_open_key_block(block)

    assert not _published_body_lines(masked), "a complete CRLF block published its body"
    assert _PEM_END.rstrip("\n") in masked, "the END line was eaten"
    assert redactor._in_key_block is False, "the END line did not close the state"


def test_the_body_floor_is_one_definition_for_the_grammar_and_the_hold() -> None:
    """The floor the grammar accepts and the floor the hold holds at are ONE number.

    ``PEM_BODY_FLOOR`` is read twice: by the body grammar (through its two quantifier
    spellings) and by the release hold, which keeps a cap-forced cut off a fragment
    shorter than it — because the line loop reads such a fragment as PROSE, which CLOSES
    an open block. The relationship is a BOUNDARY, so it is pinned from both sides here:
    a merger who moves either side — including the linear deciders #1427 substitutes at
    the pipe's call sites, which spell an ``8``/``7`` pair by hand — makes this arm
    disagree with itself rather than let the hold under-hold, which is the leak
    direction.
    """
    floor = redaction_shapes.PEM_BODY_FLOOR
    # The grammar's side of the boundary.
    assert builtin._PEM_BODY_LINE.match("M" * floor), "the grammar rejects a run at its own floor"
    assert not builtin._PEM_BODY_LINE.match("M" * (floor - 1)), "a lone sub-floor line is prose"

    # The hold's side, at the SAME boundary: build a text whose cap-forced cut leaves a
    # fragment of exactly ``fragment`` bytes in an open block's last line, and read the
    # cut back. Held iff the fragment is shorter than the floor.
    original = builtin._PIPE_DEFERRAL_LIMIT
    try:
        builtin._PIPE_DEFERRAL_LIMIT = 256
        for fragment in (floor - 1, floor):
            text = _PEM_HEADER + _body_lines(20) + "M" * (256 + fragment)
            line_start = len(text) - (256 + fragment)
            cut = builtin._PipeRedactor([])._release_point(text, final=False)
            held = cut == line_start
            assert held is (fragment < floor), (
                f"a {fragment}-byte fragment at the cut was "
                f"{'held' if held else 'released'} against a floor of {floor}"
            )
    finally:
        builtin._PIPE_DEFERRAL_LIMIT = original


def test_a_certificate_banner_does_not_open_a_private_key_block() -> None:
    """The widened armour tail must not open a block on a PUBLIC key's banner.

    Tolerating ``\\r`` and trailing whitespace removed the whitespace as a discriminator,
    so the phrase is now the only thing separating a private-key banner from any other
    PEM banner — and this is the arm that pins it, including for the spellings the
    widening was for: a certificate banner followed by base64-shaped lines stays readable
    byte for byte, because ``PRIVATE KEY`` is missing.
    """
    _pem_grammar_is_live()
    banner = _PEM_DASHES + "BEGIN CERTIFICATE" + _PEM_DASHES
    for ending in ("\n", "\r\n", "  \r\n", " \r"):
        body_break = "\n" if ending.endswith("\n") else "\r"
        text = banner + ending + _body_lines(20).replace("\n", body_break)
        published = _pipe_whole(text)
        assert _published_body_lines(
            published
        ), f"a public banner ending {ending!r} opened a private-key block"
        assert published == text, f"a public banner ending {ending!r} was rewritten by the pipe"


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
#:
#: Moved on 2026-09-21 by the false-positive fix, and the argument is short because
#: the measurement is: **no pre-existing case moved at all.** The corpus as it stood
#: at ``origin/main`` (308 cases) produces byte-identical masked text, labels, values,
#: windows, ``complete`` and ``exposed`` under this module — measured case by case
#: against ``git show origin/main:local_operator/redaction_shapes.py`` loaded beside
#: it. The digest moves for one reason only: the corpus GREW, to 335 cases, and the
#: added cases are the specification for the fix — 5 compact-JSON positives, 4 counter
#: negatives, one positive per name in ``COUNT_QUALIFIER_NAMES`` (the credential names
#: an any-segment count sweep released in the first version of this change: agent
#: review R1-1, QA round 1 Q-1), and one negative per name in
#: ``COUNT_TAIL_RELEASED_NAMES`` (the boundary the tail arm draws, pinned in the half
#: it releases: QA round 2, Q2-1).
#:
#: MOVED ONCE MORE on 2026-09-21, in the commit that answers agent review R1-1, and
#: the argument is again a measurement rather than a claim: no production code
#: changed in that commit — the diff to ``redaction_shapes.py`` is comments only, so
#: nothing could move — and the digest delta is exactly the one added row. Measured
#: by recomputing this function over the corpus WITHOUT that row under the same
#: module: 341 cases produce ``d76461eb…``, the constant this commit replaces, and
#: 342 produce the value below. The row is ``glpat-lowercase-token-value``, the
#: boundary agent review R1-1 named: an all-lowercase, separator-carrying tail is
#: read as a NAME, and ``origin/main`` masked this one as ``vendor-prefixed-token``.
#: It sits in the NEGATIVE half on purpose, so the accepted residual is a test that
#: fails if a later rule narrows it, rather than a paragraph someone has to trust.
#:
#: Moved AGAIN on 2026-09-21, by the fix for a vendor-prefixed FALSE POSITIVE, and
#: the argument is the same measurement rather than a claim: the 335-case corpus
#: described just above produces a BYTE-IDENTICAL grading under the fixed module
#: (``git show origin/main:local_operator/redaction_shapes.py`` loaded beside it, and
#: ``_corpus_grading()`` computed for that corpus under both modules and compared),
#: so nothing already in the table moved — not a masked text, not a label, not a
#: value, not a window, not a severity. The digest moves because the corpus grew to
#: 341: five negatives for the spellings an ordinary env-var NAME takes in prose
#: (``<prefix>_<name>=<value>``, the dash-joined form, and a fixed-prefix name), and
#: one positive for the npm token's own hex-and-dash spelling, which pins the
#: boundary the new tail predicate must leave alone.
#:
#: Moved on 2026-09-21 a fourth time, by the commit that pinned the MASK MARKER as
#: corpus negatives, and the argument is once again a measurement: this commit's diff
#: to ``redaction_shapes.py`` is EMPTY (tests only), so nothing could move — and the
#: digest delta is exactly the three added rows, measured the same way as the round
#: before it: recomputing ``_corpus_grading()`` over the 342 rows the constant above
#: covered, i.e. this corpus minus the three marker rows, produces that constant
#: byte for byte, and 345 produce the value below. What the three rows change is
#: coverage rather than behaviour: the marker appeared in no row on either half, and
#: its claim — that a message CONTAINING the detector's own output re-fires an
#: incident — now fails against a table instead of against a paragraph.
#:
#: Moved on 2026-09-21 a FIFTH time, by the commit answering agent review R1-1/R1-3 on the
#: marker pin, and the argument is the same measurement rather than a claim:
#: ``redaction_shapes.py`` is untouched in this commit as well (its diff against
#: the merge base ``2a9a737a`` is empty — this is a tests-and-comments change), so
#: nothing could move,
#: and the digest delta is exactly the ONE added row. Measured the same way: recomputing the
#: grading over the 345 rows the constant above covered, i.e. this corpus minus
#: ``npm-config-manage-package-manager-versions=false``, reproduces that constant byte for
#: byte, and 346 produce the value below. The row closes the combination agent review R1-3
#: named — the reported family's dash join CARRYING a value, the one spelling #1399 moved,
#: which the four rows beside it covered only by intersection.
#:
#: Moved on 2026-09-21 a SIXTH time, by the second-wave false-positive fix, and the
#: argument is the same measurement rather than a claim. The 346 rows the constant
#: above covered produce that digest BYTE FOR BYTE under the fixed module —
#: recomputed through this very function with the eleven added rows filtered out, so
#: not a masked text, not a label, not a value, not a window, not a severity moved;
#: and a second, independent check agrees: origin/main's module and this branch's
#: grade all 346 pre-existing rows identically, field for field. The digest moves
#: because the corpus grew to 357, and the added rows are the specification of the
#: fix. Eight negatives: the count trap one ESCAPED newline away (three spellings,
#: the JSON rendering's own), an identifier assigned to another identifier (two
#: rows verbatim from the file whose ``read`` reported them), the guide's
#: two-part secret-renaming form, and a vendor-looking prefix in front of an
#: org/repo PATH — the slash spelling of a string the ``_`` and ``-`` spellings
#: already spared. Three positives, because a narrowing that also stopped masking
#: the credentials arriving IN an escaped rendering would be a leak: a real key one
#: escaped newline after its name, a real key whose line ends where the rendering
#: says it does, and a slash-joined tail that is a token (case plus a digit).
#: MOVED ONCE MORE on 2026-09-21, in the commit that answers agent review R1-1,
#: R1-2, R1-3 and R1-4 — and this one moves for TWO reasons, not one.
#:
#: The corpus grew from 357 rows to 422: 59 positives and 6 negatives, every one of
#: them on a class R1-1 named. ``IDENTIFIER_ARM_NAMES`` crossed with
#: ``IDENTIFIER_ARM_SPELLINGS`` and the digit-free underscore alphabet IS the
#: released class: each of those rows was MASKED at ``origin/main`` and came back
#: with NO HIT AT ALL — nothing registered, so nothing the later exact-value pass
#: could contain — at the revision under review. The hyphenated and digit-carrying
#: neighbours are pinned beside them, on the surface that carries them most often.
#: The six negatives pin what the narrowed arm releases on purpose (three names
#: whose tail is a quantity noun), what the six-character escape spellings do (two),
#: and the one value class the pre-escape judgement releases on the escaped surface
#: BECAUSE it releases it on the real one (one: a type-name-looking first line).
#:
#: AND ONE PRE-EXISTING ROW MOVES, which is the part a digest argument has to name.
#: It is the positive the previous commit added for "a real key whose line ends where
#: the rendering says it does": it masked up to the escape then, and it masks the
#: WHOLE run now, because R1-2 measured that stopping there left a credential's tail
#: readable under a hit still graded ``complete=True`` — and left it readable with no
#: hit and no registration at all when the run before the escape was shorter than the
#: floor. Exactly ONE of the 357 rows the previous constant covered moves, measured
#: by grading all 357 through that revision's module and this one, field for field;
#: it moves in the direction that keeps a credential out of the context window.
#:
#: MOVED AGAIN on 2026-09-21, in the commit that answers agent review R2-F2, and this
#: one has the shape of a measurement too: the corpus grew from 422 rows to 423 — ONE
#: added positive, the doubled-backslash spelling of a literal backslash before a
#: credential word — and the 422 rows the constant above covered produce THAT digest
#: byte for byte under the fixed module, recomputed through this very function with
#: the added row filtered out. So not a masked text, not a label, not a value, not a
#: window, not a ``complete``, not an ``exposed`` moved for any pre-existing row. The
#: added row is the specification of the fix and it is the only row whose grading
#: moves: at the revision above it came back with NO hit and nothing registered (the
#: value readable), and it now carries ``credential-assignment`` as complete and
#: contained. The class is a credential that LOST its mask, which is why the row is
#: pinned in the POSITIVE half rather than argued about in prose.
#: MOVED ON 2026-09-22, in the commit that stops the pass masking a store NAME in a
#: credential-flag position, and the argument is once again a measurement rather than a
#: claim — with a particular shape this time, because the module change is INVISIBLE to
#: the corpus that existed. Grading the 423 rows the constant above covered under the
#: ``origin/main`` module and under this one, field for field, produces byte-identical
#: digests (``2a29fe4c…`` both times, measured beside `git show
#: origin/main:local_operator/redaction_shapes.py` loaded as a second module), and the
#: digest moves for one reason only: the corpus GREW, 423 -> 438, and the 15 added rows
#: are the specification for the fix. Five positives are the VALUE side the release must
#: not reach — an issuer token, the two underscore-joined phrases of the identifier arm
#: (the class R1-1 measured), a single unseparated token, and a padded base64 value — and
#: ten negatives are a store NAME in that position: the guide's own publish command and
#: its ``cat``/``grep`` renderings, the same name under four other flag spellings, the
#: ``=`` spelling, and the two-part form whose right half is not a credential word
#: either.
#:
#: **The behaviour change the digest is too coarse to see, stated here instead.** Twelve
#: argument spellings stop being masked — the store NAME under each flag in the rule's
#: vocabulary, both separators, the two-part form, and the ``--token ABC_123_XYZ``-shaped
#: residual the corpus pins as a negative — and NO row anywhere gains a mask. The corpus
#: could not see any of them because every flag-carrying row it already had was either a
#: VALUE (which still masks) or a NAME whose tail was a credential word (which was
#: already released), which is exactly why the rows were added rather than argued about.
#:
#: **The constant the recovered commit carried was STALE, and it is re-derived here
#: rather than trusted.** That commit's module and corpus were recovered from a subagent
#: killed mid-task, and no test had been run against them before it was committed. Graded
#: as they stand, the corpus produces ``946670a4…``, not the ``4cc31872…`` the commit
#: recorded — written before its last corpus edit, and invisible to the suite for exactly
#: the reason this constant exists: the one arm that would have caught it is the arm the
#: constant belongs to, and a wrong constant fails only when someone runs it.
#: The ARGUMENT above is what makes the correction safe, and it survives re-derivation
#: unchanged: replaying it through this same function with the base module loaded beside
#: the head one reproduces ``2a29fe4c…`` for the 423 rows that existed at ``bf48ca47``
#: under BOTH modules, field for field, so the move is the corpus's growth (423 -> 438)
#: and not a behaviour change on any pre-existing row.
#:
#: MOVED ONCE MORE ON 2026-09-22, in the commit that refuses the EXPOSURE CLAIM for the
#: flag rule's word-shaped over-mask, and the argument is the same shape: no pre-existing
#: row moved — the 438 rows above produce ``946670a4…`` byte for byte under this module
#: too, measured by grading every one of them field for field with
#: ``git show 1e8d33e6:local_operator/redaction_shapes.py`` loaded as a second module —
#: and the digest moves for the corpus's growth alone, 438 -> 439: one positive that puts
#: the word TWICE in the line, so the whole-value half of the exposure question answers
#: yes for reasons that have nothing to do with the mask. That row is the second specimen
#: the operator reported — a 33 KB documentation read escalated to a "rotate it" demand
#: for the word ``when`` — and ``test_only_the_documented_positive_case_escalates`` now
#: referees it like every other positive.
#:
#: MOVED ONCE MORE ON 2026-09-23, in the round-1 remediation, and here the argument has
#: TWO halves because two different things happened in one commit.
#:
#: 1. **The module change is INVISIBLE to the corpus, measured.** Narrowing the flag
#:    rule's prose refusal from ``len < _ASSIGNED_VALUE_MIN_CHARS`` (eight) to
#:    ``len <= _FLAG_PROSE_MAX_CHARS`` (four) restores the escalation for every
#:    five-, six- and seven-character value, and the corpus's only flag-position
#:    word is the FOUR-character ``when`` row — refused by both bounds. So: the 442
#:    rows below graded with ``git show f6f58eb7:local_operator/redaction_shapes.py``
#:    loaded as a second module produce this same ``a755ab0e…`` byte for byte, and the
#:    bound is separately shown to be insensitive across the whole range — grading the
#:    439-row corpus with the refusal refusing bare words up to 4, 5, 6, 7, 8, 11 and 15
#:    characters all give ``ff40e831…``, while refusing only up to 3 gives ``96341322…``
#:    (it stops refusing the ``when`` row). Four is the narrowest bound that keeps that
#:    row contained, which is why the boundary is pinned by a test rather than by a row:
#:    the rows that would separate five from nine have to ESCALATE, and the corpus holds
#:    exactly one escalating case by construction.
#:
#: 2. **The corpus grew by three rows, which is what moves the constant.** Measured by
#:    recomputing over the corpus WITHOUT them under this same module: the 439 rows it
#:    had produce ``ff40e831…`` field for field, so nothing pre-existing moved. The new
#:    rows are two negatives and one positive from agent review R1-3 and R1-4 — the
#:    TWO-PART spelling of the accepted residual (once with a separator in each half and
#:    once with none, because the two halves are read by shape alone and so need no
#:    separator at all: wider than the one-part release, and it had no row), and the
#:    ONE-WORD store name the release does NOT reach (``normalize_credential_key("prod")``
#:    is ``PROD``, so the arm's separator requirement leaves it masked). Both were found
#:    by a differential rather than stated by the table, which is the thing this constant
#:    exists to stop.
_CORPUS_GRADING_DIGEST = "a755ab0e8960419f719323ae343ef725e9f8662f278b1bfc66ba0eef58406f57"
#: MOVED on 2026-09-23 by the fix for the TYPE-ANNOTATION false positive, and the
#: argument is the measurement this constant's history always asks for, taken the
#: same way: the 423 rows the constant above covered produce
#: ``2a29fe4c…`` BYTE FOR BYTE under the fixed module (``git show
#: origin/main:tests/unit/secrets/credential_shape_corpus.py`` loaded beside
#: ``origin/main``'s module and this one, ``_corpus_grading()`` computed for that
#: corpus under both and compared) — so not a masked text, not a label, not a
#: value, not a window, not a ``complete``, not an ``exposed`` moved for any
#: pre-existing row. The digest moves because the corpus grew to 452: 20 negatives
#: for the type annotations the assignment rule was masking, 8 positives for the
#: credentials spelled like them, and one negative for the residual this release
#: accepts.
#:
#: FIFTEEN of the 20 added negatives move, and they all move the SAME way — from
#: masked to readable — which is the whole claim of the fix; the other five were
#: already released by another rule (a bare primitive below the value floor, an
#: annotation carrying ``[``, a keyword argument in a function signature), so they
#: are regression rows rather than behaviour rows. TWO of the fifteen ESCALATED
#: before the fix, and that is the cost this change removes: a two-argument Rust
#: generic and a two-primitive TypeScript generic both graded ``exposed``, because
#: the mask covered the truncation of the type and left its tail readable — which
#: is a rotation demand for a type annotation, and one of them stopped a release
#: pending a verdict. NONE of the 8 added positives moves: they were masked before
#: and they are masked now, which is what makes them evidence that the clause is
#: scoped rather than broad.
#: MOVED AGAIN on 2026-09-23, in the R1-1 remediation, and the argument has the same
#: two halves as the move above it.
#:
#: 1. **The MODULE change moved no pre-existing row, measured.** The confinement is
#:    now enforced (a digit outside a primitive name, and a lowercase base, each stop
#:    the release — see :func:`_carries_a_non_primitive_digit`), and the 471 rows of
#:    the recovered corpus graded with ``git show 7a0d3cbc:local_operator/redaction_shapes.py``
#:    loaded as a second module produce ``bd30424e…`` BYTE FOR BYTE under BOTH modules.
#:    That is the whole safety claim of this round: the enforcement narrowed the
#:    release class the branch introduced and touched nothing the branch had not.
#:
#: 2. **The corpus grew by seven rows, which is what moves the constant.** Five are the
#:    DIGIT-CARRYING and LOWERCASE-BASE half of the residual class — spellings the arm
#:    released WHOLE with no hit at all before this round, and which the corpus had no
#:    row for because it pinned only the digit-free spelling (agent review R1-1). They
#:    sit in ``TYPE_ANNOTATION_POSITIVES`` because they MUST mask. The other two are
#:    negatives: the same digit-free passphrase with its underscore removed, and the
#:    ``Pass<int>`` primitive-argument spelling — both released, so both are boundaries
#:    of the confined arm. The table's own counts moved with them, and the wide-arm
#:    assertion is now pinned to the table's length rather than to a stale ``8``, so a
#:    positive row with no discriminating reading fails instead of sitting inert.
_CORPUS_GRADING_DIGEST = "bee10950878787b704adc228b84b2de62ba10098bed85e332554aff3cf6ffb99"


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

    ONE EXCLUSION, and it is part of the predicate rather than an exception to it:
    the flag rule's word-shaped over-mask (``--api-key [redacted] you need to``) is graded
    CONTAINED whatever the text says, because a value that is a bare lowercase word of
    FOUR characters or fewer answers both questions YES for reasons that have nothing to
    do with the mask — every English word recurs in prose, which is how a 33 KB
    documentation read filed an ESCALATED rotation demand for the word ``when``
    (2026-09-22). It is restated here, not imported, so the exclusion cannot drift from
    the implementation without failing this arm.

    THE RESTATED BOUND IS THE IMPLEMENTATION'S OWN FOUR, and that is the correction agent
    review R1-2 asked for. It was eight — the module's masked-VALUE floor, which answers a
    different question — and a restatement at eight could not catch drift across 5..9,
    because the corpus holds no row that separates those bounds: measured, the grading is
    identical for every bound at or above four. The rows that WOULD separate them have to
    escalate, and the corpus holds exactly one escalating case by construction (the
    reason ``test_a_genuinely_exposed_compact_credential_still_files_an_incident`` gives),
    so the boundary is pinned in ``test_the_flag_prose_refusal_stops_at_four_characters``
    instead, in both directions.

    WHAT THIS RESTATEMENT CAN AND CANNOT CATCH, stated rather than implied: it fails on
    drift DOWNWARD. An implementation refusing three characters or fewer leaves the
    four-character ``when`` row escalating, where this arm — restating a bound of four —
    computes ``exposed`` False for it, and the mismatch reds on the assertion below. Drift
    UPWARD across five to nine is invisible to the corpus (no row of that width is a bare
    word), so it is not this arm's job: it is the boundary test's.
    """
    short_word_floor = 4
    checked = 0
    for case in (*POSITIVE_CASES, *NEGATIVE_CASES):
        masked, hits = scrub_shapes_with_hits(case.text)
        readable = masked.replace(REDACTION_MARKER, "")
        for hit in hits:
            value = hit.value
            exposed = bool(value) and value != REDACTION_MARKER
            word_shaped = (
                len(value) <= short_word_floor
                and value.isascii()
                and value.isalpha()
                and value.islower()
            )
            if hit.label == "cli-credential-flag" and word_shaped:
                exposed = False
            elif exposed:
                exposed = value in masked or (
                    len(value) >= 6
                    and any(value[start : start + 6] in readable for start in range(len(value) - 5))
                )
            assert hit.exposed is exposed, (case.reason, hit.label)
            checked += 1
    assert checked > 150, f"the corpus graded only {checked} hits: it is not evidence"


def test_the_flag_prose_refusal_stops_at_four_characters() -> None:
    """The refusal's BOUNDARY, both ways, where the corpus cannot pin it.

    ``_is_prose_after_a_flag`` refuses the exposure claim for a bare lowercase word in a
    credential flag's argument position, and the bound is FOUR characters. Nothing in the
    corpus separates a bound of five from one of nine (measured: the grading is identical
    for every bound at or above four) and the rows that WOULD have to escalate, which the
    corpus may not hold — so the boundary lives here, on the specimens the refusal was
    written for and on the short values it must NOT swallow (agent review R1-2, measuring
    R1-1).

    An implementation whose bound drifted to five, six, seven, eight or nine fails the
    second loop; one that drifted to three or less fails the first. Both directions are
    the point: the narrowing exists to stop manufacturing rotation demands for English
    words, and it may not buy that by giving up the escalation for a short credential
    printed a second time in the clear.
    """
    import local_operator.redaction_shapes as rs

    def escalates(text: str) -> bool:
        return rs.shape_report(scrub_shapes_with_hits(text)[1]).reached_model

    # Assembled from its segments, like every other flag/value pair in this file: the flag
    # followed by a value is the exact shape the pass rewrites, so no literal here is one.
    flag = "--" + "api-key" + " "

    # The word case the refusal exists for — the 33 KB documentation line whose second
    # ``when`` in free prose filed a rotation demand — and the shorter specimen the suite
    # already pins. Both stay CONTAINED, and these are the only two words in this test
    # that are read as prose rather than as a value.
    for word in ("when", "was"):
        prose = f"{flag}{word} you need it, and {word} the flag is set it wins"
        assert REDACTION_MARKER in scrub_shapes_with_hits(prose)[0], "the over-mask stopped"
        assert not escalates(prose), f"the prose word {word!r} demanded a rotation again"

    # Three and four characters: still contained. This is the STATED LIMIT of the
    # refusal, pinned so a later widening of the bound is a decision rather than a
    # differential — at that width a word cannot be told from a credential anywhere in
    # the text, which is the whole of the reason the claim is refused.
    for word in ("was", "hunt"):
        short = f"{flag}{word} and the {word} is set"
        assert not escalates(short), f"{word!r} (len {len(word)}) stopped being refused"

    # Five, six and seven characters: escalated AGAIN, which is the half agent review
    # R1-1 measured as lost. ``hunter`` (six) and ``letmein`` (seven) are the canonical
    # short weak passwords; ``grace`` is the five-character edge of the same class. The
    # BOUND is what is pinned here rather than the spelling, so any bare lowercase run of
    # that width would do — what matters is that a credential-shaped value printed twice
    # in the text the model reads keeps its escalation.
    for word in ("grace", "hunter", "letmein"):
        repeated = f"{flag}{word} and the {word} is set"
        assert (
            REDACTION_MARKER in scrub_shapes_with_hits(repeated)[0]
        ), f"{word!r} stopped being masked"
        assert escalates(repeated), (
            f"a {len(word)}-character value printed twice no longer escalates: the "
            "refusal is wider than its four-character specimens"
        )


def test_the_count_judgement_sees_every_segment_of_a_name() -> None:
    """The name half of the assignment rule, at the segment level.

    A count word is the QUALIFIER, and it is not always the first segment:
    ``ephemeral_5m_input_tokens`` is a counter whose first segment is a MODE. The
    judgement read ``segments[0]`` only, so the tail ``tokens`` won it, a counter
    was masked, and on the compact spelling — where the next field's key sits
    inside the match — the grader then found a fragment of the swallowed text in
    the unmasked first key and demanded a rotation for a NUMBER.

    Both halves, and the second half is asserted over NAMES rather than over
    spellings: an any-segment sweep released the ``COUNT_QUALIFIER_NAMES`` family
    — ``REDIS_CACHE_PASSWORD``, ``FACEBOOK_PAGE_ACCESS_TOKEN`` — which share the
    counters' segment-level shape and are credentials, not counts (agent review
    R1-1, QA round 1 Q-1). The first version of this test asserted four one- and
    two-segment names with no count word in them, which is exactly why it could
    report "nothing loses a mask" while a real name did: a name has to be driven
    through BOTH the judgement and the pass, so that is what happens here.
    """
    for counter in (
        "ephemeral_5m_input_tokens",
        "ephemeral_1h_input_tokens",
        "max_tokens",
        "context_tokens",
        "num_tokens",
    ):
        assert redaction_shapes.is_credential_name(counter), counter
        assert redaction_shapes.is_count_shaped(counter), counter
    for credential in ("access_token", "refresh_token", "client_secret", "api_key"):
        assert redaction_shapes.is_credential_name(credential), credential
        assert not redaction_shapes.is_count_shaped(credential), credential
    for name in COUNT_TAIL_RELEASED_NAMES:
        # The boundary the tail arm draws, asserted in the OTHER direction: a
        # qualified quantity tail is a count, so these stay readable — and the
        # corpus carries the case for each, so neither direction can drift alone
        # (QA round 2, Q2-1).
        assert redaction_shapes.is_credential_name(name), name
        assert redaction_shapes.is_count_shaped(name), name
    for name in COUNT_QUALIFIER_NAMES:
        # Both readings, because either one alone can pass for the wrong reason:
        # the judgement must call it a credential, and the pass must mask it.
        assert redaction_shapes.is_credential_name(name), name
        assert not redaction_shapes.is_count_shaped(name), name
        masked = scrub_shapes(f"{name}={FIXTURE_VALUE}")
        assert REDACTION_MARKER in masked, f"{name} left its value readable"
        assert FIXTURE_VALUE not in masked, name


def test_a_compact_json_pair_keeps_its_neighbouring_key_and_files_nothing() -> None:
    """The measured over-mask, pinned in both directions.

    On a compact pair the greedy value crossed the closing quote into the NEXT
    field: the mask replaced the neighbour's key and value, and the grader —
    correctly, for that match — found a fragment of the swallowed text inside the
    UNMASKED first key and filed a rotation demand for a credential that had been
    covered whole. So this asserts the shape of the right answer, not a count:
    both keys stay readable, both values go, and nothing escalates.
    """
    text = COMPACT_TOKEN_PAIR
    masked, hits = scrub_shapes_with_hits(text)
    assert "access_token" in masked, "the credential's own key was masked away"
    assert "refresh_token" in masked, "the neighbouring KEY was masked away"
    assert masked.count(REDACTION_MARKER) == 2, masked
    report = redaction_shapes.shape_report(hits)
    assert not report.reached_model, report


def test_a_genuinely_exposed_compact_credential_still_files_an_incident() -> None:
    """The OVER-REACH direction: narrowing the value grammar may not stop filing.

    The fix stops a fragment of a SWALLOWED neighbouring field grading as exposed.
    It must not stop the other case, which is the one the whole control exists for:
    a credential whose own characters are readable in the text the model gets.

    Built from the corpus's compact pair — the same value echoed back in the clear
    beside it, which is the spelling a response that quotes its own token has — and
    asserted in BOTH directions: the pair is masked, and the hit is graded as having
    reached the model BECAUSE the echo is still readable (the echo is not under a
    credential name, so the pass leaves it alone; that survivor is the exposure).
    Kept out of the corpus rather than added to it because it legitimately
    ESCALATES, and the corpus holds exactly one escalating case by construction
    (``_ESCALATING_POSITIVE_CASES``); this is the test that pins the direction
    without widening that set.
    """
    obj = json.loads(COMPACT_TOKEN_PAIR)
    value = obj["access_token"]
    text = COMPACT_TOKEN_PAIR[:-1] + f',"echo":"{value}"' + "}"
    masked, hits = scrub_shapes_with_hits(text)
    assert masked.count(REDACTION_MARKER) == 2, masked
    assert f'"echo":"{value}"' in masked, "the readable copy is the exposure"
    assert redaction_shapes.shape_report(hits).reached_model, hits


def test_the_measured_counter_line_is_not_an_incident() -> None:
    """The operator's Bedrock evidence line, driven end to end.

    The false positive was a rotation demand for a usage COUNTER, filed from a
    ``write`` of the Bedrock cost-tracking evidence file. The line is in the
    negative corpus; this drives it through the pass and asserts the two things
    the notice depends on — the text is untouched, and nothing is graded as
    exposed — so a future widening that re-classifies a counter reds here even if
    the corpus row is edited away.
    """
    text = COUNTER_USAGE_LINE
    masked, hits = scrub_shapes_with_hits(text)
    assert masked == text, masked
    assert hits == [], hits
    assert not redaction_shapes.shape_report(hits).reached_model


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

    # ...and the GRADING has to say the same thing, because that wording is only reached
    # when it does. The word occurs TWICE in this line, which is the shape of the
    # 2026-09-22 documentation read: for a word, the whole-value half of the exposure
    # question answers YES because the word is simply repeated in the prose around it, so
    # a 33 KB read filed the ESCALATED notice ("rotate it") for the word ``when``. The
    # withholding is what `_is_prose_after_a_flag` is for, and this is where it is pinned.
    import local_operator.redaction_shapes as rs

    repeated = prose + ", which was the whole of it"
    masked, hits = scrub_shapes_with_hits(repeated)
    assert REDACTION_MARKER in masked, "the over-mask stopped holding"
    assert rs.shape_report(hits).reached_model is False, "a prose word demanded a rotation"


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


def test_the_documented_publish_workflow_survives_every_surface() -> None:
    """The operator's workflow, driven: a script authored from what was displayed.

    Reported 2026-09-22. ``lop secret run --secret NAME -- npm publish`` is the way
    ``guide://credentials`` teaches an agent to hand a stored secret to a child, and an
    operator names an entry after the SYSTEM it belongs to: this one's tail is USERNAME,
    which is not one of the credential words the guard required. So EVERY tool result
    masked the name, and the script the agent then authored from the displayed text
    asked the store for a secret literally named ``[redacted]`` — the command failed
    against a name that does not exist, which is the failure the operator reported.

    Nothing escalated it, and that is why this test drives the WORKFLOW rather than the
    rule: a whole mask is the contained case, so it files no incident, and a unit
    assertion that the table is silent would not have seen the ``cat`` either. What is
    asserted here is the property that broke — the text survives byte for byte — on
    every model-visible surface, and the next assertion is the other half of it: the
    values that must still mask, so a widening cannot pass by releasing everything.
    """
    # Assembled from pieces so no literal in this SOURCE is a flag followed by a value:
    # this file is read by agents through the very pass it asserts about.
    store_name = "MINERVA_UI_NPROD_USERNAME"
    command = "--" + "secret " + store_name + " --" + "secret " + store_name + " -- npm publish"
    script = "#!/bin/sh" + chr(10) + "# release the UI package" + chr(10) + command

    # The three renderings the agent reads back: the command as typed, the file it
    # wrote, the ``cat`` of that file, and the ``grep`` of it with a line number.
    renderings = (
        command,
        script,
        "cat publish.sh" + chr(10) + command,
        "grep -n secret publish.sh" + chr(10) + "4:" + command,
    )
    for surface, scrub in sorted(SURFACES.items()):
        for text in renderings:
            assert scrub(text) == text, f"{surface} rewrote the workflow text"

    # ...and the session's own result hook, over the whole entry, files nothing: the
    # mask this test forbids is the one that used to happen here.
    session = _session()
    session._pending_shape_incidents.clear()
    entry = "## WATCH — 2026-09-22 — a script that publishes" + chr(10) + script
    assert session._redact_tool_result_text(entry) == entry, "the entry was rewritten"
    assert session._pending_shape_incidents == [], "the entry filed an incident"

    # THE VALUE SIDE, beside it, because that is the regression this fix could have
    # introduced: a release that widened one more step would eat all four of these.
    issuer = "ghp" + "_AbCd1234EfGhIjKlMnOpQr"
    lowercase_phrase = "_".join(("correct", "horse", "battery"))
    caps_run = "DBPASSWORD"
    armed = "Sup3rTokenValue91"
    for value in (issuer, lowercase_phrase, caps_run, armed):
        assert "cli-credential-flag" in match_shape_names(
            "server --" + "token " + value
        ), f"a value of the shape {value[:3]}… stopped being masked"
    # ...and the DSN spelling, which no flag guard may swallow.
    dsn = "mongodb://svc:" + "p" + chr(64) + "ssw0rd" + chr(64) + "db.example.net/app"
    assert REDACTION_MARKER in scrub_shapes("tool --" + "password " + dsn)

    # The instrument is alive: the control is a value under the SAME flag, in the same
    # text, and it must still be masked.
    mixed = "lop secret run --" + "secret " + store_name + " -- npm publish --" + "token "
    assert REDACTION_MARKER in scrub_shapes(mixed + armed)
    assert scrub_shapes(mixed + store_name) == mixed + store_name


# ---------------------------------------------------------------------------
# Step cost: the pass is handed ONE STEP, never the conversation
# ---------------------------------------------------------------------------
#
# The requirement these pin: redaction must be step by step, with NOTHING whose
# cost grows with how long the session has been running. That is a claim about
# WHAT EACH PASS CALL IS HANDED, so it is measured at the single funnel every
# redaction path shares — ``redaction_shapes.scrub_secrets_with_hits`` — rather
# than argued from the call graph. A future path that handed the pass the whole
# conversation would leave every other test in this file green, which is exactly
# why these exist.

#: What one settled tool result may be. The arms below use the PRODUCTION number
#: rather than a test-sized one, so the shape measured is the shipped one while
#: the whole test stays `steps x 8 KiB` of work.

_STEP_RESULT_BYTES = builtin.TOOL_OUTPUT_LIMIT_CHARS


def _record_funnel(monkeypatch) -> list[tuple[int, int]]:
    """Record ``(bytes_in, hit_count)`` for every call into the ONE table entry.

    ``scrub_shapes_with_hits`` is the single place the rule table runs: every
    other view onto it (``scrub_secrets``, ``scrub_secrets_with_hits``,
    ``scrub_shapes``, ``match_shape_names``) is a caller of it in the same
    module, so every module-level ``from … import`` binding in the tree routes
    here and ONE patch counts them all. An earlier revision instead patched the
    ``scrub_secrets_with_hits`` binding in three modules, which is not the same
    set: ``harness/redaction.summarize_arguments`` reaches the table through
    ``scrub_shapes``, so the journaled tool-call argument summary was invisible
    to the count while the docstring claimed every path was covered (R1-3).

    What this therefore guarantees: a table call SOMEWHERE in the tree is
    counted, whichever path made it. It does NOT say which path made it — a path
    that stopped scrubbing altogether counts zero calls and reads as "nothing to
    do" — so the arms below pair it with a growth assertion on the step count.
    """
    records: list[tuple[int, int]] = []
    real = redaction_shapes.scrub_shapes_with_hits

    def wrapper(text):
        scrubbed, hits = real(text)
        records.append((len(text), len(hits)))
        return scrubbed, hits

    monkeypatch.setattr(redaction_shapes, "scrub_shapes_with_hits", wrapper)
    return records


def _step_text() -> str:
    """One settled tool result: ordinary, anchor-bearing log text.

    The anchor is deliberate. A result with no anchor at all returns from the
    funnel before the table runs, so a measurement over anchor-free text would
    pin the gate rather than the pass.
    """
    line = "2026-09-21T00:00:00Z INFO build step completed, token budget nominal\n"
    return (line * ((_STEP_RESULT_BYTES // len(line)) + 1))[:_STEP_RESULT_BYTES]


def _step_tool() -> AgentTool:
    async def execute(tool_call_id, args, signal, on_update, context):
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name="emit",
            content=[TextContent(text=_step_text())],
        )

    return AgentTool(
        name="emit",
        parameters={"type": "object", "properties": {}},
        execute=execute,
    )


def _step_stream(steps: int):
    """A provider that asks for ``steps`` tool calls and then stops."""
    issued = [0]

    def stream_fn(request, signal=None):
        n = min(issued[0], steps)
        issued[0] += 1

        async def gen():
            if n < steps:
                yield StreamToolCallDelta(index=0, id=f"c{n}", name="emit", argument_delta="{}")
                yield StreamEndEvent(stop_reason="toolUse")
            else:
                yield StreamTextDelta(delta="done")
                yield StreamEndEvent(stop_reason="stop")

        return gen()

    return stream_fn


async def _drive_steps(steps: int) -> None:
    """Run the REAL loop for ``steps`` tool-calling turns."""
    from local_operator.harness.loop import AgentLoop, LoopContext

    store = VariableStore(cwd=".")
    loop = AgentLoop()
    context = LoopContext(system_blocks=["sys"], tools=[_step_tool()])
    config = LoopConfig(
        model=ModelSpec(provider="test", model_id="unit-model"),
        convert_to_llm=lambda messages: [m for m in messages if isinstance(m, Message)],
        stream_fn=_step_stream(steps),
        # The production hook on the production wiring: ``Session`` hands
        # ``_redact_tool_result_text`` here, which funnels to
        # ``redact_with_report``. A bare ``redact`` would measure the same pass,
        # but the report-carrying entry point is what ships.
        redact_tool_result=lambda text: store.redact_with_report(text)[0],
    )
    async for _ in loop.run([Message.user("go")], context, config, None):
        pass


@pytest.mark.asyncio
async def test_the_pass_is_handed_one_step_and_the_largest_call_does_not_grow(monkeypatch):
    """The structural statement of "step by step only".

    The MAXIMUM is what a session-scoped scan would move: a pass handed the whole
    history has a largest call that climbs with the session, while a per-step
    pass is flat. The TOTAL is expected to climb — that is the step count, which
    is the point — so only the maximum is asserted.
    """
    records = _record_funnel(monkeypatch)

    records.clear()
    await _drive_steps(3)
    short = list(records)

    records.clear()
    await _drive_steps(30)
    long = list(records)

    assert short and long
    assert max(b for b, _ in short) == _STEP_RESULT_BYTES
    assert max(b for b, _ in long) == _STEP_RESULT_BYTES, (
        "the largest single pass call grew with session length: some path is "
        "handing the funnel more than one step"
    )
    # Growth is in CALLS and linear in steps, which is the shape asked for.
    assert len(long) > len(short)
    assert len(long) <= 4 * 30, "implausibly many calls per step"


@pytest.mark.asyncio
async def test_a_resume_replay_makes_no_pass_calls_at_all(monkeypatch, tmp_path):
    """The replay is a PASSTHROUGH, and that is measured rather than argued.

    A resumed child is built on the stopped child's directory: ``Transcript``
    reads the file back and ``Session.__init__`` seeds its context from
    ``build_llm_history()``. No scrub runs on that path — no module under
    ``session/`` references the redaction table at all. Containment is a property
    of the WRITE path (the journaled-arguments scrub and the result hook, both
    upstream of the file), so a replay has nothing left to remove and costs
    nothing. A figure like "49.7 s of scrub CPU across 22 child transcripts" is
    arithmetic on the assumption that this path scrubs; it does not.
    """
    records = _record_funnel(monkeypatch)

    # Written straight to disk, bypassing the write-path scrub on purpose: the
    # replay's behaviour over credential-shaped bytes is the thing under test,
    # so the bytes have to be there.
    directory = tmp_path / "stopped-child"
    transcript = Transcript(directory)
    await transcript.append_message(
        Message.assistant(
            "calling the API",
            tool_calls=[
                ToolCall(
                    name="bash",
                    arguments={"command": "curl -u svc:hunter2swordfish https://example.test"},
                )
            ],
        )
    )
    on_disk = list(Transcript(directory).build_llm_history())

    records.clear()
    resumed = Session(
        model=ModelSpec(provider="test", model_id="unit-model", context_window=1000),
        stream_fn=_never_streams,
        tools=[],
        transcript=Transcript(directory),
        system_blocks_provider=lambda *_a: [],
        yolo=True,
        cwd=str(tmp_path),
        variables=VariableStore(cwd=str(tmp_path)),
    )
    assert records == [], f"the replay called the pass {len(records)} time(s)"

    # ...and it did not rewrite the bytes it replayed. Asserted on the message the
    # new session seeded its context with, because "no funnel calls" alone would
    # also be true of a replay that quietly dropped the history.
    replayed = [m for m in resumed._context.messages if isinstance(m, Message)]
    assert [m.model_dump() for m in replayed] == [m.model_dump() for m in on_disk]
    assert "hunter2swordfish" in json.dumps([m.model_dump() for m in replayed])


def _decoded_stream_whose_block_contains_the_cut(
    cap: int, body_lines: int = 4
) -> tuple[str, str, str]:
    """The round-1 review's construction at ``cap``: ``(decoded, body_line, block)``.

    A `bash` stream carrying a PEM-shaped block whose END line is the LAST line
    of the retained head, so the cap's elision snaps to a body-line newline
    INSIDE the block: the BEGIN line and body lines are kept, the END line falls
    in the dropped middle. ``body_line`` is a line of the key body, and its
    presence anywhere downstream is the leak these arms exist to catch.

    The placement is arithmetic, not a lucky string: ``_clip_head_tail`` snaps
    the head cut back to the last newline at or before ``(cap - marker) // 2``
    and ``_BashOutput`` keeps exactly the first ``cap // 2`` bytes as its head,
    so ending the block at ``cap // 2`` puts its 30-byte END line in the last 30
    bytes of that head and the body newline before it exactly at the snap point.
    The retention notice is what makes the elision fire at all: ``decode()`` is
    ``cap + len(notice)`` characters whenever anything was omitted.
    """
    # The PEM header and footer are BUILT rather than written out, and asserted
    # below, because a PEM written whole is itself a credential shape: the
    # harness's own content filter rewrote an earlier spelling of them on the
    # way into this file, which left the arm testing a block no rule matched.
    dashes = "-" * 5
    begin = dashes + "BEGIN RSA PRIVATE KEY" + dashes + "\n"
    end = dashes + "END RSA PRIVATE KEY" + dashes + "\n"
    body = "MIIEowIBAAKCAQEA" * 3
    pad = "plain output line of the alignment run, no anchors in it\n"
    block = begin + (body + "\n") * body_lines + end
    start = cap // 2 - len(block)
    total = cap + 4096
    filler = (pad * ((start // len(pad)) + 2))[:start]
    rest_len = total - start - len(block)
    raw = filler + block + (pad * ((rest_len // len(pad)) + 2))[:rest_len]
    assert scrub_shapes(block).count(REDACTION_MARKER) >= 1, "not a shape the table masks"
    assert scrub_shapes(body) == body, "the body line is masked on its own"
    chunks = builtin._BashOutput(limit=cap)
    data = raw.encode()
    for at in range(0, len(data), 65536):
        chunks.append(data[at : at + 65536])
    return chunks.decode(), body, block


def _assert_the_cut_lands_inside(decoded: str, block: str, cap: int) -> None:
    """The construction is non-vacuous: the snapped cut is strictly INSIDE it.

    Without this the arms below could pass while testing nothing — a block the
    elision keeps whole is masked under either order, and the leak only exists
    when the cut falls between the block's first and last line.
    """
    snap = decoded[: (cap - len(builtin.BASH_TRUNCATION_MARKER)) // 2].rfind("\n") + 1
    start = decoded.index(block)
    assert start < snap < start + len(block), (
        f"the cut at {snap} is not inside the block [{start}, {start + len(block)}): "
        "this arm would test nothing"
    )


def test_a_cut_inside_a_multi_line_match_masks_the_kept_side(monkeypatch):
    """R1-1: the pass sees the WHOLE stream, and the cut cannot hide a fragment.

    Two assertions, one per half of the defect. First the ORDER: the text handed
    to the pass is byte-for-byte the decoded stream, so an elision can never
    remove bytes the mask has not seen. The round-1 revision elided first and
    failed here — it handed the pass ``cap`` characters of a larger stream —
    while every other arm in this file stayed green. Second the OUTCOME: the
    block's kept side comes back masked, not raw, because the mask runs while
    the block is still whole.

    Capped at 4 KiB so the arm costs kilobytes; ``cap`` is the only input the
    construction varies with, and the shipped number is exercised below.
    """
    cap = 4096
    monkeypatch.setattr(builtin, "_REDACT_STREAM_LIMIT_CHARS", cap)
    decoded, body, block = _decoded_stream_whose_block_contains_the_cut(cap)
    assert len(decoded) > cap, "the elision must bite for this arm to mean anything"
    _assert_the_cut_lands_inside(decoded, block, cap)

    seen: list[int] = []
    real = builtin._redact_tool_text

    def spy(text, context):
        seen.append(len(text))
        return real(text, context)

    monkeypatch.setattr(builtin, "_redact_tool_text", spy)
    # A store, not ``ToolContext(cwd=".")``: with no ``variables`` the pass is a
    # no-op (``_redact_tool_text`` returns its input untouched), so the OUTCOME
    # assertion below would read a raw body line on a pass that never ran.
    context = ToolContext(cwd=".", variables=VariableStore(cwd="."))
    out = builtin._redact_settled_stream(decoded, context)

    assert seen == [len(decoded)], (
        f"the pass was handed {seen} of {len(decoded)} characters: an elision ran "
        "before the mask, so a cut inside a multi-line match can publish the kept "
        "side raw"
    )
    assert body not in out, "a raw key body line survived the settled pass"
    assert "PRIVATE KEY" not in out, "the block's header survived the settled pass"
    assert out.count(REDACTION_MARKER) >= 1, "the block was dropped, not masked"


def test_the_shipped_cap_never_publishes_a_split_match(tmp_path):
    """R1-1 at the SHIPPED cap, through the spill the model can `read` later.

    The reviewer's exact case at ``_REDACT_STREAM_LIMIT_CHARS`` (4 MiB): a
    multi-line block wholly retained, ending in the last bytes of the retained
    head, with the snapped cut inside it. Checked on both surfaces a settled
    `bash` call publishes — the text the call site spills, and the bytes
    ``read spill://`` serves back out of that file. The elide-before-mask order
    failed both: 2,000 raw body lines in the spill, ``BEGIN`` line present, and
    the same fragment served to the model on `read`.
    """
    from local_operator.tools.spill import get_store

    cap = builtin._REDACT_STREAM_LIMIT_CHARS
    decoded, body, block = _decoded_stream_whose_block_contains_the_cut(cap)
    _assert_the_cut_lands_inside(decoded, block, cap)
    context = ToolContext(cwd=str(tmp_path), variables=VariableStore(cwd=str(tmp_path)))
    masked = builtin._redact_settled_stream(decoded, context)
    assert body not in masked, "the pass published a raw key body line"

    budget = builtin.TOOL_OUTPUT_LIMIT_CHARS - 2 * len(builtin.BASH_TRUNCATION_MARKER)
    out, _err, _footer, details = builtin._bash_oversized_streams(
        masked, "", budget, False, context
    )
    assert body not in out
    # ``details`` is optional on that return type and the spill payload is
    # ``Any``; both are narrowed here rather than unwrapped with a default that
    # pyright reads as an attribute access on ``None``.
    assert details is not None, "a spilled tail must report its details"
    spilled = details.get("spill")
    handle = spilled["handle"] if isinstance(spilled, dict) else None
    assert handle, "an over-budget settled stream must spill; without it this arm checks nothing"
    stored = get_store().read_lines(handle, 1, 10**7)
    assert stored is not None, "the handle must resolve in the store that wrote it"
    served, total = stored
    assert total > 1, "the spill is empty; the handle proves nothing"
    assert body not in "\n".join(served), "the spill the model can `read` carries a raw key line"


def test_the_settled_cap_bounds_what_is_published_not_the_pass():
    """The cap is an OUTPUT bound; the mask is what it is applied to.

    Pins the halves that need no fixture: the cap is the retention limit, so the
    only text it can drop is the retention notice, and it is larger than the
    display budget, so it cannot skip bytes the spill still publishes. What it is
    NOT — a bound on the pass input — is measured by the arms above; the round-1
    claim that an elision-first order "bounds the pass" is not in this file
    anywhere.
    """
    from local_operator.tools.spill import SPILL_ENTRY_LIMIT_BYTES

    assert builtin._REDACT_STREAM_LIMIT_CHARS == SPILL_ENTRY_LIMIT_BYTES
    assert builtin._REDACT_STREAM_LIMIT_CHARS > builtin.TOOL_OUTPUT_LIMIT_CHARS
