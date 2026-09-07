"""Job ids survive the argument-coercion boundary as OPAQUE TEXT.

Regression cover for a latent defect: ``_coerce_job_targets`` unwraps the
bracketed-string shapes models emit for ``job_id``, and it did so with a plain
``json.loads``. Job ids are ``uuid4().hex[:12]``, and a measured 0.65% of those
are *also* well-formed JSON numbers -- ~0.31% all digits (``920883861377``) and
~0.33% exponent-shaped (``7019316393e2``). Those parsed to ``int``/``float``,
the ``isinstance(item, str)`` filter dropped them, and the caller reported
``unknown job [920883861377]`` -- an error whose id looks perfectly correct,
which is exactly what made it expensive to diagnose in the field.

The property test below is the point of this file: a single hand-picked digit
string proves too little when the failing input is 0.65% of RANDOM ids, so the
id shapes are GENERATED. It runs over real ``uuid4`` output plus a seeded
adversarial generator that oversamples the numeric-looking shapes, since an
unseeded 200k-draw would be needed before a plain uuid4 stream reliably covers
the exponent form.

No ``hypothesis`` dependency is added for this: the generator is a seeded
``random.Random``, so the corpus is large and adversarial but the test stays
deterministic and reproducible from the seed printed in any failure message.
"""

from __future__ import annotations

import json
import random
import uuid
from typing import Any

import pytest
from pydantic import ValidationError

from local_operator.tools.builtin import (
    HubParams,
    JobsParams,
    _coerce_hub_to,
    _coerce_job_targets,
    _coerce_single_job_id,
)

# Fixed seed: the corpus must be identical on every run and on CI, so a failure
# is reproducible from the message alone rather than "sometimes red".
_SEED = 20260907
_HEX = "0123456789abcdef"


def _adversarial_ids(rng: random.Random, count: int) -> list[str]:
    """12-char ids oversampling the shapes JSON would claim as numbers.

    A uniform uuid4 stream hits the all-digit case ~1 in 325 and the exponent
    case ~1 in 300, so a fast test needs the distribution skewed towards them.
    Every id here is still a legal ``uuid4().hex[:12]`` value -- the generator
    only changes how OFTEN each shape appears, never what a valid id may be.
    """
    out: list[str] = []
    for _ in range(count):
        kind = rng.randrange(5)
        if kind == 0:  # all digits, the reported case
            out.append("".join(rng.choice("0123456789") for _ in range(12)))
        elif kind == 1:  # all digits with leading zeros
            out.append("0" + "".join(rng.choice("0123456789") for _ in range(11)))
        elif kind == 2:  # exponent shaped: digits 'e' digits
            head = rng.randrange(1, 11)
            out.append(
                "".join(rng.choice("0123456789") for _ in range(head))
                + "e"
                + "".join(rng.choice("0123456789") for _ in range(11 - head))
            )
        elif kind == 3:  # ordinary mixed hex
            out.append("".join(rng.choice(_HEX) for _ in range(12)))
        else:  # a real uuid4 id
            out.append(uuid.uuid4().hex[:12])
    return out


@pytest.fixture(scope="module")
def id_corpus() -> list[str]:
    rng = random.Random(_SEED)
    corpus = _adversarial_ids(rng, 2000) + [uuid.uuid4().hex[:12] for _ in range(2000)]
    # Guard the generator itself: if a refactor stopped producing the numeric
    # shapes, every assertion below would pass while covering nothing.
    assert any(i.isdigit() for i in corpus), "corpus lost its all-digit ids"
    assert any(
        "e" in i and i.replace("e", "").isdigit() for i in corpus
    ), "corpus lost its exponent-shaped ids"
    return corpus


def test_single_id_round_trips_unchanged_for_every_generated_shape(id_corpus):
    """PROPERTY: for any id, each wire shape it can arrive in yields that id.

    This is the assertion that would have caught the bug: on the old code it
    fails on the first all-digit or exponent-shaped id in the corpus.
    """
    for job_id in id_corpus:
        assert _coerce_job_targets(job_id) == job_id, f"bare {job_id!r}"
        assert _coerce_job_targets(f"[{job_id}]") == job_id, f"bracketed {job_id!r}"
        assert _coerce_job_targets(json.dumps([job_id])) == job_id, f"json list {job_id!r}"
        assert _coerce_job_targets([job_id]) == job_id, f"real list {job_id!r}"
        assert _coerce_single_job_id(f"[{job_id}]") == job_id, f"single-id field {job_id!r}"


def test_result_is_always_str_never_a_number(id_corpus):
    """An id must never leave this boundary as an int/float/bool.

    Type is the invariant that matters downstream: ``_resolve_job_target``
    calls ``.strip()`` on its target, and the tool fields are typed
    ``str | list[str]``, so a numeric leak is a hard failure rather than a
    lookup miss.
    """
    for job_id in id_corpus:
        for shape in (job_id, f"[{job_id}]", [job_id]):
            got = _coerce_job_targets(shape)
            assert isinstance(got, str), f"{shape!r} -> {got!r} ({type(got).__name__})"


def test_multi_id_lists_still_work_for_every_generated_pair(id_corpus):
    """PROPERTY: the list case survives -- fixing the string case must not
    break the tool's documented ability to take several ids at once."""
    rng = random.Random(_SEED + 1)
    for _ in range(500):
        a, b = rng.choice(id_corpus), rng.choice(id_corpus)
        assert _coerce_job_targets(f"[{a}, {b}]") == [a, b]
        assert _coerce_job_targets(json.dumps([a, b])) == [a, b]
        assert _coerce_job_targets([a, b]) == [a, b]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # The exact reported failure, pinned by hand so the regression is
        # readable in the diff and not only implied by the generator.
        ("[920883861377]", "920883861377"),
        ("920883861377", "920883861377"),
        ('["920883861377"]', "920883861377"),
        # Other numeric literal shapes the JSON grammar would claim.
        ("[12e345678901]", "12e345678901"),
        ("[177650473e52]", "177650473e52"),
        ("[000123456789]", "000123456789"),
        ("[00420]", "00420"),
        ("[1e5]", "1e5"),
        ("[0]", "0"),
        ("[-1]", "-1"),
        # Value-preservation cases that a str(parsed) fix would silently
        # corrupt: 007 -> "7" and 1e5 -> "100000.0".
        ("[007]", "007"),
        ("[1.50]", "1.50"),
        # JSON's bare literals: an id spelled 'true'/'null' is still text.
        ("[true]", "true"),
        ("[null]", "null"),
        # Bare non-string scalars from the outer tool-argument decode.
        (920883861377, "920883861377"),
        ([920883861377], "920883861377"),
        # Ordinary ids and label targets are untouched.
        ("a1b2c3d4e5f6", "a1b2c3d4e5f6"),
        ("[a1b2c3d4e5f6]", "a1b2c3d4e5f6"),
        ("reviewer", "reviewer"),
        ("all", "all"),
        # Multi-id shapes, including the numeric ids that also broke.
        ("[920883861377, 468698086935]", ["920883861377", "468698086935"]),
        ("[920883861377, a1b2c3d4e5f6]", ["920883861377", "a1b2c3d4e5f6"]),
        (["920883861377", "a1b2c3d4e5f6"], ["920883861377", "a1b2c3d4e5f6"]),
        ('["a1b2c3d4e5f6", "0f1e2d3c4b5a"]', ["a1b2c3d4e5f6", "0f1e2d3c4b5a"]),
        # A genuinely nested list still flattens.
        ('[["920883861377"], "a1b2c3d4e5f6"]', ["920883861377", "a1b2c3d4e5f6"]),
    ],
)
def test_known_id_shapes(value, expected):
    assert _coerce_job_targets(value) == expected


def test_bare_int_survives_because_str_int_is_lossless():
    """A model that emits ``{"job_id": 920883861377}`` unquoted is still served.

    JSON forbids leading zeros, ``+`` and underscores in an integer literal, so
    the decoded ``int`` round-trips to exactly the characters that were written
    and recovering it invents nothing.
    """
    assert _coerce_job_targets(920883861377) == "920883861377"
    # The int is deliberately off-annotation: `job_id` is declared `str | None`
    # and the before-validator is precisely what widens the accepted input, so
    # the type checker is right about the signature and wrong about the intent.
    bare_int: Any = 920883861377
    assert JobsParams(op="peek", job_id=bare_int).job_id == "920883861377"


@pytest.mark.parametrize(
    ("literal", "decoded"),
    [
        ("7019316393e2", 701931639300.0),  # retyped: a different, valid-looking id
        ("177650473e52", 1.77650473e60),
        ("13190e419943", float("inf")),  # overflow: the id is destroyed outright
        ("1e5", 100000.0),
    ],
)
def test_bare_float_is_refused_rather_than_turned_into_a_different_id(literal, decoded):
    """MAJOR-1 (review round 1): never fabricate an id from a parsed float.

    Unlike the string path, a BARE unquoted scalar has already been through the
    outer tool-argument decode, which destroyed the source text before this
    code runs. ``13190e419943`` is a legal ``uuid4().hex[:12]``, so this is
    reachable in production at the same ~0.65% rate as the rest of the defect.

    Formatting the survivor would emit ``'701931639300.0'`` or ``'inf'`` -- a
    syntactically plausible id that was never minted, which is precisely the
    trap this module exists to close, and *worse* than the pre-fix behaviour
    that at least failed loudly. Where the literal is unrecoverable the value
    must pass through so the field's own validation reports it.
    """
    assert json.loads(literal) == decoded  # the decode really is lossy
    assert _coerce_job_targets(decoded) is decoded  # passed through, not formatted
    with pytest.raises(ValidationError, match="valid string"):
        JobsParams(op="peek", job_id=decoded)


@pytest.mark.parametrize("depth", [5, 30, 999, 1001, 5000, 20000])
def test_deep_nesting_is_bounded_and_never_raises(depth):
    """MINOR-1 (review round 1): no ``RecursionError`` escapes at any depth.

    Two distinct stack consumers had to be bounded: this module's own recursion
    (capped by ``_MAX_TARGET_NEST_DEPTH``) and ``json.loads``, which does its
    own recursive descent and blows up *before* that cap is consulted. Since
    ``RecursionError`` is not a ``ValueError``, it would otherwise sail past
    the parse guard and past the callers' ``ValidationError`` handlers.
    """
    payload = "[" * depth + '"920883861377"' + "]" * depth
    result = _coerce_job_targets(payload)  # must not raise
    assert isinstance(result, (str, list))
    assert _coerce_hub_to(payload) is not None


def test_dict_policy_matches_the_documented_asymmetry():
    """MINOR-2 (review round 1): code and comment must agree.

    A real Python list PRESERVES a non-id item so the field's own validation
    reports it; a bracketed STRING drops it, because there the surrounding
    value is a single string with no per-item validation left to speak for it,
    and preserving the dict would fail the whole payload instead of resolving
    the real ids beside it. Both behaviours are now stated where they happen.
    """
    assert _coerce_job_targets([{"a": 1}, "920883861377"]) == [{"a": 1}, "920883861377"]
    assert _coerce_job_targets('[{"a":1}, "920883861377"]') == "920883861377"


@pytest.mark.parametrize("value", [None, True, False])
def test_bare_non_id_scalars_pass_through_untouched(value):
    """A bare ``None`` means the argument was OMITTED, not an id spelled 'null'.

    ``JobsParams.job_id`` is ``str | None`` and ``op='list'`` legitimately
    carries no id, so coercing a top-level ``None`` to ``"null"`` would turn
    every plain ``jobs`` listing into a lookup for a job that cannot exist.
    ``True``/``False`` are likewise not ids in any shape a model emits.

    The in-list case is the opposite and is covered above: inside a bracketed
    payload the model literally typed those characters, so ``'[null]'`` keeps
    round-tripping to ``"null"``. Passing these through unchanged also leaves
    the field's own validation as the thing that reports a bad argument.
    """
    assert _coerce_job_targets(value) is value
    assert _coerce_single_job_id(value) is value


def test_measured_mangle_rate_over_real_uuid4_ids_is_zero():
    """The fleet-scale claim, asserted rather than described.

    10k real ids is ~65 expected failures on the old code (0.65%), so this
    fails on origin/main with near-certainty while staying fast.
    """
    ids = [uuid.uuid4().hex[:12] for _ in range(10_000)]
    mangled = [i for i in ids if _coerce_job_targets(f"[{i}]") != i]
    assert not mangled, f"{len(mangled)}/{len(ids)} ids mangled, e.g. {mangled[:5]}"


# ---------------------------------------------------------------------------
# The same defect in the sibling coercion: hub's ``to`` field.
#
# Found by QA on round 1 while this PR was fixing ``_coerce_job_targets``.
# ``_coerce_hub_to`` parsed with a plain ``json.loads`` and kept only
# ``isinstance(item, str)``, so a numeric-looking subagent id was parsed to a
# number and then dropped: ``'[920883861377]'`` coerced to ``[]``. The failure
# mode is worse than a bad lookup -- ``to`` became EMPTY, so ``hub op='ask'``
# addressed nobody and a parent blocking on the answer simply never got one,
# with no error to explain it.
# ---------------------------------------------------------------------------


def test_hub_to_round_trips_every_generated_id_shape(id_corpus):
    """PROPERTY: any id, in any wire shape, reaches ``to`` intact.

    Fails on the pre-fix code at the first all-digit or exponent-shaped id.
    """
    for job_id in id_corpus:
        assert _coerce_hub_to(job_id) == [job_id], f"bare {job_id!r}"
        assert _coerce_hub_to(f"[{job_id}]") == [job_id], f"bracketed {job_id!r}"
        assert _coerce_hub_to(json.dumps([job_id])) == [job_id], f"json {job_id!r}"


def test_hub_to_never_silently_empties(id_corpus):
    """The invariant that matters: a named target must never vanish.

    An empty ``to`` is the dangerous outcome -- ``ask`` reports no recipient
    rather than an error, so the caller waits for an answer nobody was asked
    for.
    """
    for job_id in id_corpus:
        for shape in (job_id, f"[{job_id}]", json.dumps([job_id])):
            assert _coerce_hub_to(shape), f"{shape!r} coerced to an empty target list"


def test_hub_to_multi_target_fan_out_survives(id_corpus):
    """``resume``/``send`` fan one message out to a batch; numeric ids included."""
    rng = random.Random(_SEED + 2)
    for _ in range(300):
        a, b = rng.choice(id_corpus), rng.choice(id_corpus)
        assert _coerce_hub_to(f"[{a}, {b}]") == [a, b]
        assert _coerce_hub_to(json.dumps([a, b])) == [a, b]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("[920883861377]", ["920883861377"]),
        ("920883861377", ["920883861377"]),
        ('["920883861377"]', ["920883861377"]),
        ("[13190e419943]", ["13190e419943"]),  # overflows to inf if parsed
        ("[12e345678901]", ["12e345678901"]),
        ("[000123456789]", ["000123456789"]),
        ("[920883861377, 468698086935]", ["920883861377", "468698086935"]),
        ("[a1b2c3d4e5f6]", ["a1b2c3d4e5f6"]),
        ("all", ["all"]),
        ("reviewer", ["reviewer"]),
    ],
)
def test_hub_to_known_shapes(value, expected):
    assert _coerce_hub_to(value) == expected
    assert HubParams(op="ask", to=value, message="ping").to == expected


@pytest.mark.parametrize("payload", ["[null]", "[true]", "[false]"])
def test_hub_to_still_drops_bare_json_literals(payload):
    """Deliberate divergence from ``_coerce_job_targets``, and a pre-existing
    contract: a ``to`` target is resolved against live peers rather than looked
    up as an id, so a bare JSON literal must not become a peer literally named
    ``null``.

    The empty list is the intended carrier of that refusal. ``HubParams``
    accepts it -- ``op='list'`` legitimately has no target -- and
    ``execute_hub`` is what reports ``needs a 'to' target``, so the assertion
    is on the empty ``to``, not on a ``ValidationError`` the model never
    raises.
    """
    assert _coerce_hub_to(payload) == []
    assert HubParams(op="ask", to=payload, message="ping").to == []


def test_hub_ask_reaches_a_child_whose_id_is_all_digits():
    """End-to-end shape: an id of the kind a real ``AsyncJobManager`` mints at
    this rate survives coercion and lands in ``to`` as one addressable target.

    Contrast the pre-fix behaviour, where ``to`` came back ``[]`` and the ask
    was answered by nobody.
    """
    minted = "920883861377"
    # `to` is declared `list[str]`; passing the bracketed STRING is the whole
    # point of the before-validator under test (a model emitted it that way).
    bracketed: Any = f"[{minted}]"
    params = HubParams(op="ask", to=bracketed, message="are you there?")
    assert params.to == [minted]
    assert params.op == "ask"
