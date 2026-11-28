"""``run_episode``'s exit code: 0 only for a scored episode that ended cleanly.

A scored ``truncated`` / ``agent_stop`` episode used to exit 1 (only
``completed`` exited 0), so every step-capped OSWorld run sat under launchd's
"last exit code 1" beside genuine infrastructure deaths. The exit code now
needs BOTH a ``scored`` score and a ``completed``/``truncated``/``agent_stop``
outcome status: a ``failed`` (leaked cleanup, wedged bridge) or ``cancelled``
outcome can still carry a score and must stay a process failure (review r1 R1).
The three return sites -- the reply arm in ``run``, ``_run_session_arm`` and the
owner frontend's sealed-outcome read -- are each pinned below (review r1 R2).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.evaluation.evidence.models import ScoreArtifact
from local_operator.evaluation.runner.episode import EpisodeOutcome
from local_operator.evaluation.runner.secrets import StaticSecretResolver
from scripts import run_episode
from tests.unit.evaluation.runner.test_completion_gate import (  # noqa: F401 - fixture
    _driver_args,
    durable_run_root,
)
from tests.unit.evaluation.runner.test_run_episode_effort import (  # noqa: F401 - fixture
    isolated_session_env,
)

# (outcome status, score, expected exit). The ``failed``/``cancelled`` rows
# carry a ``scored`` score on purpose: that is the shape the reply arm seals for
# a rescue_required (leaked) run and the session arm for a bridge-wedged run
# whose scorer still succeeded.
_SCORED = ScoreArtifact(status="scored", binary=1)
_ROWS = [
    ("completed", _SCORED, run_episode.EXIT_OK),
    ("truncated", _SCORED, run_episode.EXIT_OK),
    ("agent_stop", _SCORED, run_episode.EXIT_OK),
    ("failed", _SCORED, run_episode.EXIT_EPISODE),
    ("cancelled", _SCORED, run_episode.EXIT_EPISODE),
    ("completed", ScoreArtifact(status="unscored", reason="infrastructure_failure"), 1),
    ("completed", None, run_episode.EXIT_EPISODE),
]
_IDS = [
    "completed-scored",
    "truncated-scored",
    "agent_stop-scored",
    "failed-scored",
    "cancelled-scored",
    "completed-unscored",
    "completed-no-score",
]


def _owner_status(outcome_path: Path) -> SimpleNamespace:
    # The owner frontend's read of the sealed outcome (the launchd path).
    return SimpleNamespace(phase="ended", receipt={}, outcome_path=str(outcome_path), detail=None)


@pytest.mark.parametrize("status", ["truncated", "agent_stop", "completed"])
def test_a_scored_clean_ending_exits_ok(
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
    assert run_episode._episode_exit_code(status, score) == run_episode.EXIT_OK


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
    assert run_episode._episode_exit_code("failed", score) == run_episode.EXIT_EPISODE


@pytest.mark.parametrize(("status", "score", "expected"), _ROWS, ids=_IDS)
def test_the_sealed_outcome_exit_needs_a_score_and_a_clean_status(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    status: str,
    score: ScoreArtifact | None,
    expected: int,
) -> None:
    """Owner frontend site: ``status`` and ``score`` are read from outcome.json."""

    outcome = tmp_path / "outcome.json"
    outcome.write_text(
        json.dumps({"status": status, "score": score.model_dump(mode="json") if score else None})
    )

    assert run_episode._report_owner_outcome(None, _owner_status(outcome)) == expected
    capsys.readouterr()
    assert run_episode._episode_exit_code(status, score) == expected


def test_a_rescue_required_failed_outcome_exits_episode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The reply arm's leaked-cleanup seal: scored, but ``failed`` + rescue_required."""

    outcome = EpisodeOutcome(
        status="failed",
        episode_id="ep-leak",
        bundle_root=tmp_path,
        score=_SCORED,
        rescue_required=True,
    )
    sealed = tmp_path / "outcome.json"
    sealed.write_text(json.dumps(run_episode._outcome_json(outcome)))

    assert run_episode._episode_exit_code(outcome.status, outcome.score) == 1
    assert run_episode._report_owner_outcome(None, _owner_status(sealed)) == 1
    capsys.readouterr()


@pytest.mark.asyncio
@pytest.mark.parametrize(("status", "score", "expected"), _ROWS, ids=_IDS)
async def test_the_reply_arm_return_site_pins_the_exit_code(
    tmp_path: Path,
    durable_run_root: Path,  # noqa: F811 - imported fixture
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    status: Any,
    score: ScoreArtifact | None,
    expected: int,
) -> None:
    """``run``'s reply-arm return, with only the ``EpisodeRunner`` stubbed.

    Both terminal gates are off so the scripted client passes the gate
    preflight (``test_completion_gate`` pins that refusal); everything up to
    the runner -- parsing, config, selector, secrets -- is the real path.
    """

    class _Runner:
        def __init__(self, spec: Any, *_args: Any, **_kwargs: Any) -> None:
            self._episode_id = spec.episode_id

        async def run(self) -> EpisodeOutcome:
            return EpisodeOutcome(
                status=status, episode_id=self._episode_id, bundle_root=None, score=score
            )

    monkeypatch.setattr(run_episode, "EpisodeRunner", _Runner)
    args = run_episode.build_parser().parse_args(
        _driver_args(
            tmp_path,
            durable_run_root,
            "--model-client",
            "scripted-finish",
            "--no-completion-gate",
            "--no-premature-terminal-gate",
        )
    )

    assert await run_episode.run(args) == expected
    assert json.loads(capsys.readouterr().out)["status"] == status


@pytest.mark.asyncio
@pytest.mark.parametrize(("status", "score", "expected"), _ROWS, ids=_IDS)
async def test_the_session_arm_return_site_pins_the_exit_code(
    tmp_path: Path,
    isolated_session_env: Path,  # noqa: F811 - imported fixture
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    status: str,
    score: ScoreArtifact | None,
    expected: int,
) -> None:
    """``_run_session_arm``'s return, with only ``run_session_episode`` stubbed."""

    from local_operator.evaluation import session_arm, source_pin

    monkeypatch.setattr(source_pin, "verify_core_source", lambda _: None)

    async def ended(**_kwargs: Any) -> Any:
        return session_arm.SessionArmOutcome(status, "exit-code", record_root=None, score=score)

    monkeypatch.setattr(session_arm, "run_session_episode", ended)
    argv = ["--selector", "s.json", "--task-id", "t", "--route", "anthropic/claude-opus-5"]
    argv += ["--run-root", str(tmp_path), "--engagement", "session"]

    result = await run_episode._run_session_arm(
        args=run_episode.build_parser().parse_args(argv),
        spec=SimpleNamespace(secret_refs=(), episode_id="exit-code"),
        config=None,
        selector=SimpleNamespace(core_source=object()),
        resolver=StaticSecretResolver({}),
        route=("anthropic", "claude-opus-5"),
    )

    assert result == expected
    assert json.loads(capsys.readouterr().out)["status"] == status
