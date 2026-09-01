"""Score mapping: scored-or-raise, never a fabricated 0.0.

The C3 guarantee stated as tests: exact 0/1 map to binary, a genuine fraction
maps to ``partial_ppm``, and every failure mode — NaN, out-of-range,
non-numeric, a dict without a score — RAISES rather than returning an unscored
or zero artifact.
"""

from __future__ import annotations

import pytest
from lop_osworld_v2_adapter import scoring
from lop_osworld_v2_adapter.scoring import ScoringProtocolError


def test_zero_maps_to_binary_zero() -> None:
    artifact = scoring.score_to_artifact(0.0)
    assert artifact.status == "scored"
    assert artifact.binary == 0


def test_one_maps_to_binary_one() -> None:
    artifact = scoring.score_to_artifact(1.0)
    assert artifact.status == "scored"
    assert artifact.binary == 1


def test_a_fraction_maps_to_partial_ppm() -> None:
    artifact = scoring.score_to_artifact(0.5)
    assert artifact.status == "scored"
    assert artifact.partial_ppm == 500_000
    assert artifact.binary is None


def test_partial_ppm_is_an_exact_integer() -> None:
    artifact = scoring.score_to_artifact(1 / 3)
    assert artifact.partial_ppm == round((1 / 3) * 1_000_000)


def test_a_v2_dict_result_is_unwrapped() -> None:
    artifact = scoring.score_to_artifact({"score": 1.0})
    assert artifact.binary == 1


def test_nan_raises() -> None:
    with pytest.raises(ScoringProtocolError):
        scoring.score_to_artifact(float("nan"))


def test_out_of_range_raises() -> None:
    with pytest.raises(ScoringProtocolError):
        scoring.score_to_artifact(2.0)
    with pytest.raises(ScoringProtocolError):
        scoring.score_to_artifact(-0.1)


def test_non_numeric_raises() -> None:
    with pytest.raises(ScoringProtocolError):
        scoring.score_to_artifact("1.0")


def test_a_dict_without_a_score_raises() -> None:
    with pytest.raises(ScoringProtocolError):
        scoring.score_to_artifact({"result": 1.0})


def test_none_raises() -> None:
    with pytest.raises(ScoringProtocolError):
        scoring.score_to_artifact(None)
