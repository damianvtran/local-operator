"""The shape scan's ONE pass, proved equal to the TWO passes it replaced.

**What this file guards.** ``redaction_shapes._run_shapes`` used to scan each
unguarded rule TWICE: ``finditer`` to collect that rule's hits, then ``sub`` to
apply the same rule to the same text — two full scans per rule, 78 scans for an
anchored line. It now runs one ``sub`` whose replacement is a callback that
records the hit and renders the mask
(:func:`local_operator.redaction_shapes._mask_with_recorded_hit`). The
``finditer`` half is gone, so nothing asserts its results any more unless this
file does.

**Why the merge is not a cosmetic change and needs a differential test.** Hit
ORDER is load-bearing, not display order:

* ``VariableStore._register_shape_hits`` walks the hit list in order and stops
  registering distinct values at ``MAX_DETECTED_REGISTRATIONS``, so a reordered
  list registers a DIFFERENT set of secrets for the rest of the session;
* ``shape_report`` keeps FIRST appearance for the notice's label order;
* the survival index grades the values in the order they were collected.

So a merge that found the same hits in a different order would change which
credentials are contained while every "does this mask?" assertion still passed.
The equivalence is meant to hold *by construction* — ``re.sub`` hands its
callback the non-overlapping matches of ONE scan, left to right, which is the
sequence ``finditer`` yields over that same string, and ``_hit_value(shape,
match)`` reads only the shape and the match — and the tests below are the
executable form of that argument, with the OLD implementation kept verbatim as
the oracle.

**The construction's premises, each checked rather than assumed** (a future
pattern can invalidate any of them, and each has a test here):

1. No shipped pattern uses a backreference. Backreferences are read from the
   ORIGINAL string during ``sub``, which is also the string ``finditer`` scans,
   so non-overlapping matches resolve identically either way — but with no
   such pattern in the table the point is moot, and the test says so.
2. No shipped pattern can match the empty string. Empty matches are the one
   class where a scanner's advance rule is visible at all; the test pins that
   the table has none, and separately runs ``finditer`` and a ``sub`` callback
   side by side on synthetic empty-capable and backreference patterns.
3. Every string replacement is a VALID template. ``pattern.sub(template, text)``
   compiles the template BEFORE it scans, so the old call raised ``re.error`` on
   a bad template even when nothing matched; the merged path only calls
   ``match.expand`` on a match. ``test_no_shipped_template_can_diverge...``
   proves that divergence class is empty for this table (and shows the control
   that proves the class exists in principle).
4. ``sub(template)`` and ``sub(lambda m: m.expand(template))`` are the same
   substitution — asserted directly, template by template, over the corpus.

**Where the corpus comes from.** ``tests/unit/secrets/credential_shape_corpus``
is the shipped specification of what MUST be masked and what MUST survive (303
positives, 194 negatives, and the type-annotation halves), and it is the same
corpus the surface-parametrised suite in ``test_credential_shapes.py`` runs. The
generated half below joins and mutates those texts, which is where cross-rule
interaction shows up: a later rule sees the earlier rule's MASK, so a merge that
changed when a hit was recorded relative to the substitution would diverge
there and nowhere else.
"""

from __future__ import annotations

import random
import re
from typing import Any, Iterator, cast

import pytest

from local_operator import redaction_shapes
from local_operator.redaction_shapes import (
    _LINE_SHAPES,
    _MULTILINE_SHAPES,
    _SHAPE_ANCHORS,
    CREDENTIAL_SHAPES,
    REDACTION_MARKER,
    Shape,
    ShapeHit,
)
from tests.unit.secrets.credential_shape_corpus import (
    NEGATIVE_CASES,
    POSITIVE_CASES,
    TYPE_ANNOTATION_NEGATIVES,
    TYPE_ANNOTATION_POSITIVES,
)

# =============================================================================
# The oracle: the two-pass implementation, verbatim from before the merge.
# =============================================================================

#: The rule bodies below are copied from ``redaction_shapes`` as it stood at
#: ``baeb8089d`` (the commit this lane is stacked on), minus only the type
#: annotations that pyright can infer. They call the module's UNCHANGED helpers
#: (``_hit_value``, ``_make_hit``, ``_apply_guarded``, ``_close_partial_masks``)
#: so that the differential isolates the one thing that changed: the scan.


def _run_shapes_two_pass(shapes: tuple[Shape, ...], text: str, hits: list[ShapeHit]) -> str:
    """The OLD ``_run_shapes``: ``finditer`` to record, then ``sub`` to mask."""
    for shape in shapes:
        if shape.guard is not None:
            text = redaction_shapes._apply_guarded(shape, text, hits)
            continue
        assert shape.replacement is not None
        matches = list(shape.pattern.finditer(text))
        if matches:
            for match in matches:
                value = redaction_shapes._hit_value(shape, match)
                if value:
                    hits.append(redaction_shapes._make_hit(shape, match, value))
        text = shape.pattern.sub(shape.replacement, text)
    return redaction_shapes._close_partial_masks(text)


def scrub_shapes_with_hits_two_pass(text: str) -> tuple[str, list[ShapeHit]]:
    """The OLD ``scrub_shapes_with_hits`` — the driver was not part of the merge.

    Mirrored here rather than imported because the shipped driver now calls the
    merged ``_run_shapes``; a differential that used it on both sides would
    compare the new code with itself.
    """
    hits: list[ShapeHit] = []
    if not text or not redaction_shapes.has_shape_anchor(text):
        return text, hits
    text = _run_shapes_two_pass(_MULTILINE_SHAPES, text, hits)
    if len(_LINE_SHAPES) == 0:  # pragma: no cover - the table would be empty
        return text, hits
    scrubbed_lines: list[str] = []
    for line in text.splitlines(keepends=True):
        if not redaction_shapes.has_shape_anchor(line):
            scrubbed_lines.append(line)
            continue
        line_hits: list[ShapeHit] = []
        scrubbed_lines.append(_run_shapes_two_pass(_LINE_SHAPES, line, line_hits))
        hits.extend(line_hits)
    text = "".join(scrubbed_lines)
    return text, redaction_shapes._only_fully_masked(hits, text)


def _signature(hits: list[ShapeHit]) -> list[tuple[str, str, str, bool, bool]]:
    """Every field of every hit, IN ORDER — the thing the hit list decides.

    A tuple rather than the dataclasses themselves so a failure prints the
    differing field instead of two 300-element reprs.
    """
    return [(h.label, h.value, h.window, h.complete, h.exposed) for h in hits]


def _two_pass(text: str) -> tuple[str, list[ShapeHit]]:
    """The differential, both sides, for one text."""
    new_text, new_hits = redaction_shapes.scrub_shapes_with_hits(text)
    old_text, old_hits = scrub_shapes_with_hits_two_pass(text)
    assert new_text == old_text, "the merged pass rendered different text"
    assert _signature(new_hits) == _signature(old_hits), "hits differ in content or order"
    return new_text, new_hits


# =============================================================================
# The corpus, as the differential's inputs.
# =============================================================================

CORPUS_TEXTS: tuple[str, ...] = (
    tuple(case.text for case in POSITIVE_CASES)
    + tuple(case.text for case in NEGATIVE_CASES)
    + tuple(case.text for case in TYPE_ANNOTATION_POSITIVES)
    + tuple(case.text for case in TYPE_ANNOTATION_NEGATIVES)
)

CORPUS_IDS: tuple[str, ...] = tuple(
    [f"positive-{i}" for i in range(len(POSITIVE_CASES))]
    + [f"negative-{i}" for i in range(len(NEGATIVE_CASES))]
    + [f"type-positive-{i}" for i in range(len(TYPE_ANNOTATION_POSITIVES))]
    + [f"type-negative-{i}" for i in range(len(TYPE_ANNOTATION_NEGATIVES))]
)

#: Ordinary text, for the gate's clean half and as the filler a generated case
#: is joined with. None of these carries an anchor, which is asserted below.
CLEAN_LINES: tuple[str, ...] = (
    "",
    "   ",
    "GET /v1/items 200 12ms",
    "total 12",
    "drwxr-xr-x  4 user staff 128 Sep 18 09:12 src",
    "SELECT id, name FROM users WHERE active = true;",
    '{"port": 8080, "status": "ok", "items": [1, 2, 3]}',
    "Traceback (most recent call last):",
    '  File "main.py", line 42, in <module>',
    "PASSED tests/unit/secrets/test_credential_shapes.py::test_x",
    "λ → 日本語テキスト",
    "100%|##########| 42/42 [00:01<00:00, 31.2it/s]",
)


def _generated_corpus() -> tuple[str, ...]:
    """Joins and mutations of corpus lines, deterministically seeded.

    Why joins: a rule sees the text the PREVIOUS rule left behind, so an
    interaction between rules is only visible once two rules fire in one text.
    Why mutations: the patterns are case-insensitive and delimiter-tolerant, and
    an uppercased or respaced case is a cheap way to reach branches the corpus's
    own spellings do not.
    """
    rng = random.Random(0x6B)
    source = (
        [case.text for case in POSITIVE_CASES[:150]]
        + [case.text for case in NEGATIVE_CASES[:60]]
        + list(CLEAN_LINES)
    )
    texts: list[str] = []
    for _ in range(240):
        picked = [rng.choice(source) for _ in range(rng.randint(2, 4))]
        sep = rng.choice(["\n", "\r\n", " ", "\n\n", "\t"])
        text = sep.join(picked)
        mutation = rng.randint(0, 3)
        if mutation == 1:
            text = text.upper()
        elif mutation == 2:
            text = "  " + text + "\t"
        elif mutation == 3:
            text = text.replace("=", " = ")
        texts.append(text)
    return tuple(texts)


GENERATED_TEXTS: tuple[str, ...] = _generated_corpus()


# =============================================================================
# Item 2: the merged scan is the two-pass scan.
# =============================================================================


@pytest.mark.parametrize("text", CORPUS_TEXTS, ids=CORPUS_IDS)
def test_the_merged_scan_matches_the_two_pass_oracle_over_the_shipped_corpus(
    text: str,
) -> None:
    """Text AND every hit (all five fields), in order, over the shipped corpus."""
    _two_pass(text)


@pytest.mark.parametrize("text", GENERATED_TEXTS)
def test_the_merged_scan_matches_the_two_pass_oracle_on_joined_and_mutated_text(
    text: str,
) -> None:
    """The cross-rule half: joins where a later rule sees an earlier rule's mask."""
    _two_pass(text)


def test_the_differential_is_not_vacuous() -> None:
    """A corpus that produced no hits would make every assertion above empty.

    These floors are structural (counts of hits and of multi-hit texts), not
    timings: they say the differential actually exercised ordered hit lists, and
    they are floors rather than equalities so growing the corpus cannot fail
    them.
    """
    cases_with_hits = 0
    multi_hit = 0
    total_hits = 0
    multi_label = 0
    for text in CORPUS_TEXTS + GENERATED_TEXTS:
        _, hits = scrub_shapes_with_hits_two_pass(text)
        if hits:
            cases_with_hits += 1
            total_hits += len(hits)
            if len(hits) > 1:
                multi_hit += 1
            if len({hit.label for hit in hits}) > 1:
                multi_label += 1
    assert cases_with_hits >= 200, cases_with_hits
    assert total_hits >= 300, total_hits
    assert multi_hit >= 20, multi_hit
    assert multi_label >= 10, multi_label


def test_the_recorded_order_is_the_scan_order_not_a_set() -> None:
    """The order itself, asserted against the text — not merely preserved from a
    run of the same code. A merge that collected hits into a set or re-sorted
    them would keep ``test_the_merged_scan...`` green if the oracle were ever
    rewritten, and this test would not.

    Four rules fire here, two of them twice, so the assertions are about a real
    sequence: same-rule hits ascend through the text, and the first-appearance
    label order is the order the rules and the positions put them in.
    """
    text = (
        "MONGO_DSN=mongodb://svc:firstsecret@a.invalid/db\n"
        "AWS_SECRET_ACCESS_KEY=abcdefghijklmnop1234\n"
        "REDIS_URL=redis://:secondsecret@b.invalid/0\n"
        "PASSWORD=thirdsecretvalue\n"
    )
    _, hits = redaction_shapes.scrub_shapes_with_hits(text)
    assert len(hits) >= 4, [hit.label for hit in hits]

    # Same-rule hits are in ascending text order: `sub` walks the string left to
    # right, and a reordering merge would break exactly this.
    for label in {hit.label for hit in hits}:
        offsets = [text.index(hit.value) for hit in hits if hit.label == label]
        assert offsets == sorted(offsets), (label, offsets)

    labels = [hit.label for hit in hits]
    assert labels[:4] == [
        "dsn-password",
        "credential-assignment",
        "dsn-password",
        "credential-assignment",
    ], labels
    # Non-vacuous: a palindromic list would satisfy an order assertion for the
    # wrong reason.
    assert hits != list(reversed(hits))


class _CountingPattern:
    """A ``re.Pattern`` stand-in that records the SCANS a rule asks for.

    AGENTS.md asks for a structural invariant in preference to a numeric one,
    and "work done" rather than wall time. This is that invariant for the merge:
    an unguarded rule must be applied to the text exactly once (``sub``) and
    must not be scanned separately to collect hits (``finditer``).
    """

    def __init__(self, pattern: re.Pattern[str]) -> None:
        self._pattern = pattern
        self.sub_calls = 0
        self.finditer_calls = 0
        self.search_calls = 0

    def sub(self, repl: Any, string: str, count: int = 0) -> str:
        self.sub_calls += 1
        return self._pattern.sub(repl, string, count)

    def finditer(self, string: str, pos: int = 0, endpos: int = 2**63 - 1) -> Iterator[Any]:
        self.finditer_calls += 1
        return self._pattern.finditer(string, pos, endpos)

    def search(self, string: str, pos: int = 0, endpos: int = 2**63 - 1) -> Any:
        self.search_calls += 1
        return self._pattern.search(string, pos, endpos)


def _wrapped(shape: Shape, counter: _CountingPattern) -> Shape:
    """The same rule, with its pattern swapped for the counting stand-in."""
    return Shape(
        label=shape.label,
        pattern=cast("Any", counter),
        replacement=shape.replacement,
        secret_group=shape.secret_group,
        guard=shape.guard,
    )


def test_each_unguarded_rule_scans_the_text_exactly_once() -> None:
    """One ``sub`` per unguarded rule, and NO ``finditer`` — the merge's whole claim.

    The text anchors, so the rules that can fire do: ``hits`` being non-empty is
    what keeps "one scan" from being satisfied by a rule that never ran.
    """
    hits: list[ShapeHit] = []
    text = "MONGO_DSN=mongodb://svc:hunter2pw@db.invalid/x\n"
    unguarded = [shape for shape in CREDENTIAL_SHAPES if shape.guard is None]
    assert unguarded, "the table would be empty"

    for shape in unguarded:
        counter = _CountingPattern(shape.pattern)
        redaction_shapes._run_shapes((_wrapped(shape, counter),), text, hits)
        assert (counter.sub_calls, counter.finditer_calls) == (1, 0), shape.label

    assert hits, "no rule fired, so the scan counts above prove nothing"


def test_a_guarded_rule_still_searches_and_never_subs() -> None:
    """The guarded path was not merged into the callback path.

    ``_apply_guarded`` is a hand-written scan-and-splice loop (it must be: a
    rejected span is re-scanned from one character in, see its docstring). The
    merge touched only the unguarded branch, and this pins that the two did not
    get crossed.
    """
    guarded = [shape for shape in CREDENTIAL_SHAPES if shape.guard is not None]
    assert guarded, "the table would have no guarded rule"
    shape = guarded[0]
    counter = _CountingPattern(shape.pattern)
    hits: list[ShapeHit] = []
    redaction_shapes._run_shapes((_wrapped(shape, counter),), "PASSWORD=hunter2hunter2", hits)
    assert counter.sub_calls == 0
    assert counter.search_calls >= 1


def test_a_template_sub_equals_the_callback_that_expands_it() -> None:
    """The substitution the merge relies on, at the regex level.

    The OLD call was ``pattern.sub(template, text)``; the merged one renders
    each match with ``match.expand(template)``. Those are the same substitution
    only if ``sub``'s own template expansion is ``match.expand`` applied per
    match — which this asserts over every shipped template and the whole corpus
    (templates with ``\\g<name>``, escaped backslashes and repeated groups are
    where the two could have differed).
    """
    checked = 0
    for shape in CREDENTIAL_SHAPES:
        replacement = shape.replacement
        if shape.guard is not None or callable(replacement):
            continue
        assert isinstance(replacement, str)
        for text in CORPUS_TEXTS + GENERATED_TEXTS[:60]:
            by_template = shape.pattern.sub(replacement, text)
            by_callback = shape.pattern.sub(lambda m, t=replacement: m.expand(t), text)
            assert by_template == by_callback, (shape.label, text)
            checked += 1
    assert checked >= 1000, checked


def test_no_shipped_template_can_diverge_when_a_rule_matches_nothing() -> None:
    """The one asymmetry between ``sub(template)`` and a per-match ``expand``.

    ``pattern.sub(template, text)`` compiles the template BEFORE scanning, so a
    template with an out-of-range group reference raises ``re.error`` even when
    the pattern matches NOTHING; the merged path only expands on a match, so
    such a template would sit silent on non-matching text and raise only there
    it matched. The control below proves that class exists; the loop proves this
    table is outside it, so the two paths agree on every input, matching or not.
    """
    with pytest.raises(re.error):
        re.compile("a").sub(r"\9", "b")

    for shape in CREDENTIAL_SHAPES:
        replacement = shape.replacement
        if shape.guard is not None or callable(replacement):
            continue
        assert isinstance(replacement, str)
        # No match, but the template is still compiled and validated.
        shape.pattern.sub(replacement, "")


def test_no_shipped_pattern_is_empty_capable_or_backreferencing() -> None:
    """Premises 1 and 2 of the construction argument, pinned against the table.

    Both classes are places where a scanner's behaviour is subtle enough that
    the equivalence deserves more than a reading of the docs. They are asserted
    absent here so a future rule that introduces one fails THIS test (and can
    then be argued about deliberately), and the next test runs the two scanners
    side by side on synthetic members of both classes so the reasoning is
    checked rather than assumed.
    """
    backreference = re.compile(r"\\[1-9]|\\g<")
    for shape in CREDENTIAL_SHAPES:
        source = shape.pattern.pattern
        assert not backreference.search(source), shape.label
        assert shape.pattern.match("") is None, shape.label


@pytest.mark.parametrize(
    ("pattern", "text"),
    [
        (r"(ab)\1", "abab abab xababx"),
        (r"a*", "aaa b aa c"),
        (r"a*?", "aaa b aa c"),
        (r"(?i)key=(\w+)", "KEY=one key=two KeY=three"),
        (r"^key=(\w+)", "key=one\nnot\nkey=two"),
        (r"(?m)^key=(\w+)", "key=one\nnot\nkey=two"),
        (r"(?<=:)([a-z]+)(?=@)", "x:alpha@y z:beta@w"),
        (r"(?P<name>\w+)=(\w+)", "a=1 bb=22"),
        (r"x|xy", "xy xxy xy"),
    ],
)
def test_finditer_and_a_sub_callback_walk_the_same_matches_in_the_same_order(
    pattern: str, text: str
) -> None:
    """The construction claim, executed on the pattern classes that could break it.

    ``re.sub`` with a callable and ``finditer`` are two views of one scanner, so
    the callback must see exactly the matches ``finditer`` yields — same spans,
    same order — for backreferences, empty-capable patterns (greedy and lazy),
    inline flags, ``^`` with and without MULTILINE, lookarounds, named groups and
    alternation.
    """
    compiled = re.compile(pattern)
    via_finditer = [(m.start(), m.end(), m.group(0)) for m in compiled.finditer(text)]
    seen: list[tuple[int, int, str]] = []

    def record(match: re.Match[str]) -> str:
        seen.append((match.start(), match.end(), match.group(0)))
        return REDACTION_MARKER

    rendered = compiled.sub(record, text)
    assert seen == via_finditer
    # And the callback's return value is what the substitution actually used.
    assert rendered == compiled.sub(REDACTION_MARKER, text)
    assert seen, "the pattern matched nothing, so the comparison is empty"


# =============================================================================
# Item 1: the anchor gate is the same predicate, and it stops at a hit.
# =============================================================================


def _any_oracle(text: str) -> bool:
    """``has_shape_anchor`` as it stood: ``any`` over a generator of anchors."""
    lowered = text.lower()
    return any(anchor in lowered for anchor in _SHAPE_ANCHORS)


def _anchor_corpus() -> tuple[str, ...]:
    """Clean lines and anchoring lines, one spelling per anchor.

    Each anchor is embedded UPPERCASED, which is both how a log line is likely
    to carry it and a check that the single ``text.lower()`` is what makes the
    comparison case-insensitive.
    """
    anchoring = [f"prefix {anchor.upper()} suffix" for anchor in _SHAPE_ANCHORS]
    anchoring += [case.text for case in POSITIVE_CASES[:40]]
    anchoring += ["KEY", "KeY=v", "MONGODB://x", "-----BEGIN", "Bearer x", "AUTH=1"]
    return CLEAN_LINES + tuple(anchoring)


ANCHOR_CORPUS: tuple[str, ...] = _anchor_corpus()
ANCHOR_IDS: tuple[str, ...] = tuple(
    [f"clean-{i}" for i in range(len(CLEAN_LINES))]
    + [f"anchoring-{i}" for i in range(len(ANCHOR_CORPUS) - len(CLEAN_LINES))]
)


class _CountingStr(str):
    """A ``str`` that counts how many times the predicate lowers it."""

    def __new__(cls, value: str) -> _CountingStr:
        instance = super().__new__(cls, value)
        instance.lower_calls = 0
        return instance

    def lower(self) -> str:
        self.lower_calls += 1
        return str.lower(self)


class _CountingAnchors(tuple):  # type: ignore[type-arg]
    """The anchor table, recording WHICH entries the predicate actually tested.

    A subclass rather than a monkeypatched function so the assertion is about
    the loop's own control flow: the tested sequence must be a PREFIX of the
    table, which is what an early return does and what a generator's ``any``
    does too — the difference between them is frames, not answers.
    """

    def __init__(self, anchors: tuple[str, ...]) -> None:
        super().__init__()
        self.tested: list[str] = []

    def __iter__(self) -> Iterator[str]:
        for anchor in tuple.__iter__(self):
            self.tested.append(anchor)
            yield anchor


@pytest.mark.parametrize("text", ANCHOR_CORPUS, ids=ANCHOR_IDS)
def test_has_shape_anchor_agrees_with_the_any_oracle(text: str) -> None:
    """Clean and anchoring lines, including every anchor's own spelling."""
    assert redaction_shapes.has_shape_anchor(text) == _any_oracle(text)


def test_a_clean_line_tests_every_anchor_and_lowers_the_text_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """74 membership tests, ONE ``lower()`` — the shape of the fix, as work done.

    The clean case is the one that pays the gate on every line of every tool
    result, so "it tested all of them and gave up" is the point. The ``lower``
    count is the other half: lowering per anchor would be 74 lowerings but the
    same answer, which is why the answer alone cannot pin it.
    """
    counting = _CountingAnchors(_SHAPE_ANCHORS)
    monkeypatch.setattr(redaction_shapes, "_SHAPE_ANCHORS", counting)
    text = _CountingStr("GET /v1/items 200 12ms from 10.0.0.4")
    assert redaction_shapes.has_shape_anchor(text) is False
    assert counting.tested == list(_SHAPE_ANCHORS)
    assert len(counting.tested) >= 70, len(counting.tested)
    assert text.lower_calls == 1


def test_an_anchoring_line_stops_at_the_first_anchor_that_hits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The early return, measured as a PREFIX of the table — never a full pass.

    The expected stop index comes from the oracle over the same text, so the
    assertion holds whatever substring relations the anchors grow; what it pins
    is that the loop returns at the first hit rather than testing the rest.
    """
    for anchor in _SHAPE_ANCHORS:
        plain = f"prefix {anchor.upper()} suffix"
        expected_stop = next(
            index for index, candidate in enumerate(_SHAPE_ANCHORS) if candidate in plain.lower()
        )
        text = _CountingStr(plain)
        counting = _CountingAnchors(_SHAPE_ANCHORS)
        monkeypatch.setattr(redaction_shapes, "_SHAPE_ANCHORS", counting)
        assert redaction_shapes.has_shape_anchor(text) is True
        assert counting.tested == list(_SHAPE_ANCHORS[: expected_stop + 1]), anchor
        assert text.lower_calls == 1


def test_the_gate_still_answers_true_for_every_positive_case() -> None:
    """The conjunction with the corpus: the fast loop cannot hide a rule.

    ``test_credential_shapes.test_every_positive_case_trips_the_gate`` asserts
    this too; it is repeated here because it is the property item 1 could have
    broken, and a report that cites the other module must be able to point at a
    run of it in this one as well.
    """
    misses = [
        case.text for case in POSITIVE_CASES if not redaction_shapes.has_shape_anchor(case.text)
    ]
    assert not misses, f"these shapes would be skipped by the gate: {misses[:5]}"
