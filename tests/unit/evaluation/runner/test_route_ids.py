"""``fold_model_id`` is a bijection into ``StrictIdentifier``.

A sealed route must mean exactly one model (comparability, M1), so the fold
that lets an OpenRouter id fit the identifier alphabet has to be reversible
and collision-free -- two ids that differ anywhere must fold differently.
"""

from __future__ import annotations

import random
import re
import string

import pytest

from local_operator.evaluation.evidence.models import RouteIdentity
from local_operator.evaluation.runner.route_ids import fold_model_id, unfold_model_id

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")

EXAMPLES = {
    "deepseek/deepseek-v4-flash-vision-exp": "deepseek_sdeepseek-v4-flash-vision-exp",
    "moonshotai/kimi-k2:free": "moonshotai_skimi-k2:free",
    "claude-opus-5": "claude-opus-5",
    "a_b/c": "a__b_sc",
    "gpt 4": "gpt_x204",
    "x/é": "x_s_xc3_xa9",
}


@pytest.mark.parametrize(("raw", "folded"), sorted(EXAMPLES.items()))
def test_known_shapes_fold_readably_and_round_trip(raw: str, folded: str) -> None:
    assert fold_model_id(raw) == folded
    assert unfold_model_id(folded) == raw
    RouteIdentity(provider_id="openrouter", route_id=f"openrouter:{folded}", model_id=folded)


def test_distinct_ids_never_collide() -> None:
    """The escape is unambiguous: the pairs an unescaped fold would merge stay apart."""

    pairs = [("a/b", "a_sb"), ("a_b", "a__b"), ("a_sb", "a/b"), ("a:b", "a/b")]
    for left, right in pairs:
        assert fold_model_id(left) != fold_model_id(right)


def test_fold_is_a_strict_identifier_and_unfold_inverts_it() -> None:
    """Seeded rather than hypothesis-driven: the repo carries no hypothesis dependency."""

    alphabet = string.ascii_letters + string.digits + "_/.:- +@#é漢"
    rng = random.Random(20260902)
    for _ in range(2000):
        model_id = rng.choice(string.ascii_letters + string.digits) + "".join(
            rng.choice(alphabet) for _ in range(rng.randint(0, 30))
        )
        folded = fold_model_id(model_id)
        assert _IDENTIFIER.fullmatch(folded), (model_id, folded)
        assert unfold_model_id(folded) == model_id


@pytest.mark.parametrize("bad", ["", "/x", "_x", ".x"])
def test_ids_that_cannot_lead_an_identifier_are_refused(bad: str) -> None:
    with pytest.raises(ValueError):
        fold_model_id(bad)


@pytest.mark.parametrize("bad", ["a_", "a_q", "a_xzz", "a_x4"])
def test_malformed_escapes_are_refused_on_unfold(bad: str) -> None:
    with pytest.raises(ValueError, match="malformed escape"):
        unfold_model_id(bad)
