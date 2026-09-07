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

import pytest

from local_operator.tools.builtin import _coerce_job_targets, _coerce_single_job_id

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
