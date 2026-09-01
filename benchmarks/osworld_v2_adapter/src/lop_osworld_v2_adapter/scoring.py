"""Score mapping: OSWorld's float evaluator -> a SCORED ScoreArtifact, or raise.

The contract is scored-or-raise (runner/episode.py:766-768). "Unscored" is a
HARNESS decision expressed via finalization intent, never an adapter return
value. OSWorld's own ``evaluate()`` does the opposite of what we need: it
swallows metric exceptions into ``0.0`` and logs a missing evaluator rather
than raising (desktop_env.py:594-700). Mapping "could not evaluate" to 0.0
would report a failure the agent did not commit — the exact score-deflation
error this module exists to prevent.

Mapping rules:
- exact ``0.0`` -> ``binary=0``; exact ``1.0`` -> ``binary=1`` (the shape a
  leaderboard expects; ScoreArtifact permits either).
- any other in-range value -> ``partial_ppm=round(v * 1_000_000)``. The
  protocol's metadata subset excludes floats, and ``partial_ppm`` is the
  integer expression of a fractional score. V2's ``conj: "avg"``/``"sum"``
  evaluators produce genuine fractions, so this path is real.
- NaN, out-of-range, non-numeric, or a missing evaluator -> raise.
"""

from __future__ import annotations

import math
from typing import Any

from local_operator.evaluation.evidence.models import ScoreArtifact


class ScoringUnavailable(RuntimeError):
    """The episode cannot be honestly scored. Raised, never returned as 0.0."""


class ScoringProtocolError(RuntimeError):
    """OSWorld returned a value outside its own [0.0, 1.0] contract."""


def score_to_artifact(raw: Any) -> ScoreArtifact:
    """Map OSWorld's raw ``evaluate()`` return to a SCORED artifact, or raise.

    V2 may return a dict carrying ``{"score": float}`` (task_base.py:79-88)
    when a task overrides ``evaluate``; unwrap exactly that one key. Anything
    else that is not a real number in [0.0, 1.0] is a protocol violation, not
    a zero.
    """

    if isinstance(raw, dict):
        if "score" not in raw:
            raise ScoringProtocolError("evaluator returned a dict without a 'score' key")
        raw = raw["score"]

    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ScoringProtocolError(f"evaluator returned a non-numeric score: {type(raw).__name__}")
    value = float(raw)
    if math.isnan(value) or math.isinf(value):
        raise ScoringProtocolError("evaluator returned NaN or infinite score")
    if not (0.0 <= value <= 1.0):
        raise ScoringProtocolError(f"evaluator score {value} is outside [0.0, 1.0]")

    if value == 0.0:
        return ScoreArtifact(status="scored", binary=0)
    if value == 1.0:
        return ScoreArtifact(status="scored", binary=1)
    return ScoreArtifact(status="scored", partial_ppm=round(value * 1_000_000))
