"""``run_episode``'s exit code answers "is there a score to read".

A scored ``truncated`` / ``agent_stop`` episode used to exit 1 (only
``completed`` exited 0), so every step-capped OSWorld run sat under launchd's
"last exit code 1" beside genuine infrastructure deaths. The exit code now
follows the outcome's score; the printed status still says how it ended.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from local_operator.evaluation.evidence.models import ScoreArtifact
from scripts import run_episode


def _owner_status(outcome_path: Path) -> SimpleNamespace:
    # The owner frontend's read of the sealed outcome (the launchd path).
    return SimpleNamespace(phase="ended", receipt={}, outcome_path=str(outcome_path), detail=None)


@pytest.mark.parametrize("status", ["truncated", "agent_stop", "completed"])
def test_a_scored_outcome_exits_ok_whatever_its_status(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], status: str
) -> None:
    score = ScoreArtifact(status="scored", binary=0)
    outcome = tmp_path / "outcome.json"
    outcome.write_text(
        json.dumps(
            {
                "status": status,
                "terminal_reason": "max-steps",
                "score": score.model_dump(mode="json"),
            }
        )
    )

    assert run_episode._report_owner_outcome(None, _owner_status(outcome)) == run_episode.EXIT_OK
    assert json.loads(capsys.readouterr().out)["status"] == status
    # The in-process arms pass the typed artifact itself.
    assert run_episode._episode_exit_code(score) == run_episode.EXIT_OK


@pytest.mark.parametrize(
    "score",
    [None, ScoreArtifact(status="unscored", reason="infrastructure_failure")],
    ids=["no-score", "unscored"],
)
def test_an_unscored_failed_outcome_exits_episode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], score: ScoreArtifact | None
) -> None:
    outcome = tmp_path / "outcome.json"
    outcome.write_text(
        json.dumps(
            {
                "status": "failed",
                "terminal_reason": "session-arm",
                "score": score.model_dump(mode="json") if score else None,
            }
        )
    )

    assert (
        run_episode._report_owner_outcome(None, _owner_status(outcome)) == run_episode.EXIT_EPISODE
    )
    assert json.loads(capsys.readouterr().out)["status"] == "failed"
    assert run_episode._episode_exit_code(score) == run_episode.EXIT_EPISODE
