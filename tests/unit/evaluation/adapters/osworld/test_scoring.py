"""Score mapping: scored-or-raise, never a fabricated 0.0.

The C3 guarantee stated as tests: exact 0/1 map to binary, a genuine fraction
maps to ``partial_ppm``, and every failure mode — NaN, out-of-range,
non-numeric, a dict without a score — RAISES rather than returning an unscored
or zero artifact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from lop_osworld_v2_adapter import scoring
from lop_osworld_v2_adapter.scoring import ScoringProtocolError


def test_zero_maps_to_binary_zero(tmp_path: Path) -> None:
    artifact = scoring.score_to_artifact(0.0, artifact_root=tmp_path)
    assert artifact.status == "scored"
    assert artifact.binary == 0


def test_one_maps_to_binary_one(tmp_path: Path) -> None:
    artifact = scoring.score_to_artifact(1.0, artifact_root=tmp_path)
    assert artifact.status == "scored"
    assert artifact.binary == 1
    assert artifact.partial_ppm == 1_000_000


def test_a_fraction_maps_to_partial_ppm(tmp_path: Path) -> None:
    artifact = scoring.score_to_artifact(0.5, artifact_root=tmp_path)
    assert artifact.status == "scored"
    assert artifact.partial_ppm == 500_000
    assert artifact.binary == 0


def test_partial_ppm_is_an_exact_integer(tmp_path: Path) -> None:
    artifact = scoring.score_to_artifact(1 / 3, artifact_root=tmp_path)
    assert artifact.partial_ppm == round((1 / 3) * 1_000_000)


def test_a_v2_dict_result_is_unwrapped(tmp_path: Path) -> None:
    artifact = scoring.score_to_artifact({"score": 1.0}, artifact_root=tmp_path)
    assert artifact.binary == 1


def test_nan_raises(tmp_path: Path) -> None:
    with pytest.raises(ScoringProtocolError):
        scoring.score_to_artifact(float("nan"), artifact_root=tmp_path)


def test_out_of_range_raises(tmp_path: Path) -> None:
    with pytest.raises(ScoringProtocolError):
        scoring.score_to_artifact(2.0, artifact_root=tmp_path)
    with pytest.raises(ScoringProtocolError):
        scoring.score_to_artifact(-0.1, artifact_root=tmp_path)


def test_non_numeric_raises(tmp_path: Path) -> None:
    with pytest.raises(ScoringProtocolError):
        scoring.score_to_artifact("1.0", artifact_root=tmp_path)


def test_a_dict_without_a_score_raises(tmp_path: Path) -> None:
    with pytest.raises(ScoringProtocolError):
        scoring.score_to_artifact({"result": 1.0}, artifact_root=tmp_path)


def test_none_raises(tmp_path: Path) -> None:
    with pytest.raises(ScoringProtocolError):
        scoring.score_to_artifact(None, artifact_root=tmp_path)


@pytest.mark.parametrize(
    "case", ["nodes", "depth", "cycle", "utf8", "unicode-size", "keys", "tuple"]
)
def test_details_fail_closed_without_partial_files(tmp_path: Path, case: str) -> None:
    from typing import Any

    detail: Any = None
    if case == "nodes":
        detail = [0] * scoring.MAX_SCORE_DETAIL_NODES
    elif case in ("depth", "cycle"):
        detail = []
        if case == "cycle":
            detail.append(detail)
        else:
            for _ in range(scoring.MAX_SCORE_DETAIL_DEPTH + 1):
                detail = [detail]
    elif case == "utf8":
        detail = "\ud800"
    elif case == "unicode-size":
        detail = "é" * (scoring.MAX_SCORE_DETAIL_BYTES // 2)
    elif case == "keys":
        detail = {1: "coercion would lose type"}
    else:
        detail = (1, 2)
    with pytest.raises(ScoringProtocolError):
        scoring.score_to_artifact({"score": 1, "detail": detail}, artifact_root=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_near_completion_is_not_promoted_by_ppm_rounding(tmp_path: Path) -> None:
    artifact = scoring.score_to_artifact(0.9999999, artifact_root=tmp_path)
    assert artifact.binary == 0
    assert artifact.partial_ppm == 1_000_000


def test_detail_byte_limit_is_exact_not_truncation(tmp_path: Path) -> None:
    overhead = len(scoring._detail_bytes({"score": 1, "note": ""}))
    raw = {"score": 1, "note": "x" * (scoring.MAX_SCORE_DETAIL_BYTES - overhead)}
    artifact = scoring.score_to_artifact(raw, artifact_root=tmp_path)
    assert artifact.details is not None
    assert artifact.details.byte_count == scoring.MAX_SCORE_DETAIL_BYTES
    raw["note"] += "x"
    with pytest.raises(ScoringProtocolError):
        scoring.score_to_artifact(raw, artifact_root=tmp_path)
    assert len(list(tmp_path.iterdir())) == 1


# ----------------------------------------------------------------------
# Guest-environment disclosure (``guest_xcursor``): it rides the sealed detail
# ----------------------------------------------------------------------

_PATCH = {"xcursor_patch": {"schema": "lop-guest-xcursor-patch-v1", "applied": True}}


def _detail(tmp_path: Path, artifact: Any) -> Any:
    import json

    return json.loads((tmp_path / artifact.details.sha256).read_bytes())


def test_no_environment_keeps_the_detail_bytes_unchanged(tmp_path: Path) -> None:
    """The fake provider and pre-change records stage exactly the old bytes."""

    plain = scoring.score_to_artifact(0.5, artifact_root=tmp_path)
    explicit = scoring.score_to_artifact(0.5, artifact_root=tmp_path, environment=None)
    assert plain.details == explicit.details
    assert _detail(tmp_path, plain) == 0.5


def test_the_environment_disclosure_is_sealed_beside_the_score(tmp_path: Path) -> None:
    artifact = scoring.score_to_artifact(1.0, artifact_root=tmp_path, environment=_PATCH)
    detail = _detail(tmp_path, artifact)
    assert detail[scoring.EVALUATOR_RESULT_KEY] == 1.0
    assert detail[scoring.GUEST_ENVIRONMENT_KEY] == _PATCH
    assert artifact.binary == 1  # the number is untouched


def test_an_over_budget_diagnostics_block_never_drops_the_disclosure(tmp_path: Path) -> None:
    huge = {"schema": "lop-evaluator-diagnostics-v1", "stdout": "x" * (2 * 1024 * 1024)}
    artifact = scoring.score_to_artifact(
        0.0, artifact_root=tmp_path, diagnostics=huge, environment=_PATCH
    )
    detail = _detail(tmp_path, artifact)
    assert detail[scoring.EVALUATOR_DIAGNOSTICS_KEY] == scoring.DIAGNOSTICS_REFUSED
    assert detail[scoring.GUEST_ENVIRONMENT_KEY] == _PATCH


def test_a_result_too_large_for_the_disclosure_keeps_a_fixed_marker(tmp_path: Path) -> None:
    """m1: the disclosure is refused with a marker, never silently dropped.

    The result alone fits under the 1 MiB limit, but not with the disclosure
    beside it. An absent key would read exactly like a stock/fake-provider
    score, hiding a possibly-deviated guest.
    """

    big_env = {"xcursor_patch": {"applied": True, "pad": "p" * 4096}}
    # Leave room for the wrapper and the ~40-byte marker, not the 4 KiB env.
    raw = {"score": 0.0, "blob": "x" * (scoring.MAX_SCORE_DETAIL_BYTES - 2048)}
    artifact = scoring.score_to_artifact(raw, artifact_root=tmp_path, environment=big_env)
    detail = _detail(tmp_path, artifact)
    assert detail[scoring.EVALUATOR_RESULT_KEY] == raw
    assert detail[scoring.GUEST_ENVIRONMENT_KEY] == scoring.GUEST_ENVIRONMENT_REFUSED
    assert artifact.binary == 0
