"""The one birth-effort resolver, and the spec shape it refuses to guess from.

``resolve_birth_effort`` is the single owner of "the level a conversation born on
this model will RUN at", shared by the draft pane, the draft preview and the
resumed-conversation resolver. Its third case is DEFINED on the spec's seed, so a
caller holding a seed-cleared spec gets a silently wrong answer rather than an
error — which is exactly how the preview and the cold frame disagreed (review
round 2, R6). These tests pin the guard that turns that into a loud failure
(review round 3, R12 / Q-R3-1).
"""

import pytest

from local_operator.harness.types import ModelSpec
from local_operator.session.cold_model import resolve_birth_effort


def _seeded_spec() -> ModelSpec:
    """The shape ``build_model_spec`` returns: the default IS the seed it computed."""
    return ModelSpec(
        provider="deepseek",
        model_id="deepseek-flash",
        reasoning_efforts=("none", "low", "high", "max"),
        reasoning_effort="high",
        reasoning_default_effort="high",
    )


def test_a_seeded_spec_answers_its_seed_when_nothing_else_has_an_opinion(tmp_path) -> None:
    """No choice and no configured level: the spec's own rung is the answer.

    The config dir is a path that does not exist on purpose — the resolver reads a
    configured ``model_effort`` without materialising the directory it is pointed
    at, and a test must not read (or create) the developer's real config.
    """
    spec = _seeded_spec()
    assert resolve_birth_effort(spec, None, tmp_path / "absent") == "high"


def test_a_seed_cleared_spec_is_refused_instead_of_answered_no_level(tmp_path) -> None:
    """Clearing the seed is detectable, so it fails loudly rather than silently.

    ``reasoning_effort is None`` beside a non-``None`` default is the shape only a
    seed-cleared spec has, and the third case would otherwise answer ``None`` —
    "no level" — for a birth that runs the seed. This is the assertion that makes
    the guard's promise real: before the guard, this call returned ``None``.
    """
    cleared = _seeded_spec().model_copy(update={"reasoning_effort": None})
    with pytest.raises(ValueError, match="still carries its seed"):
        resolve_birth_effort(cleared, None, tmp_path / "absent")
