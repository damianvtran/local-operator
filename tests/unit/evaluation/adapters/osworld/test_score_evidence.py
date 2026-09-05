"""Installed-wheel score boundary, real runner finalization and bundle verifier.

The fake substitutes only evaluate()'s upstream source. In a copied interpreter
we import the installed wheel (not the source conftest's path), run the real
adapter through its normal prepare/reset/score surface, and inspect sealed bytes.
No task answer or checkpoint fixture is tied to a benchmark task.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tests.unit.evaluation.adapters.osworld import fixtures, spawn_helpers


@pytest.fixture(scope="module")
def installed_python(tmp_path_factory: pytest.TempPathFactory) -> str:
    root = tmp_path_factory.mktemp("score-evidence-install")
    wheel = spawn_helpers.build_adapter_wheel(root / "wheel")
    selector = spawn_helpers.build_spawnable_adapter(root, wheel, {"task_plain": fixtures.PLAIN})
    return selector.python_executable


def _raw(case: str) -> Any:
    if case == "dict":
        return {
            "score": 0.5,
            "safety": {"passed": False, "notes": ["synthetic audit"]},
            "partial_scores": {"first": 1, "second": 0},
            "evaluation_error": None,
        }
    if case == "missing-submission":
        return {"score": 0, "evaluation_error": "no_submission"}
    if case == "secret":
        return {"score": 1, "safety": "score-detail-secret-canary-9827"}
    if case == "nonfinite-detail":
        return {"score": 1, "safety": float("nan")}
    if case == "nonjson":
        return {"score": 1, "safety": object()}
    if case == "oversized":
        return {"score": 1, "notes": "x" * (1024 * 1024 + 1)}
    if case in {"missing-detail", "invalid-media", "publication-io", "receipt-io"}:
        return 0.5
    return {
        "zero": 0.0,
        "one": 1.0,
        "fraction": 1 / 3,
        "near-one": 0.9999999,
        "invalid": "1.0",
        "nonfinite": float("inf"),
    }[case]


async def _exercise(root: Path, case: str) -> dict[str, Any]:
    # Deliberately local: pytest's source adapter conftest must never be loaded
    # in this child. The installed package owns every adapter call below.
    import lop_osworld_v2_adapter
    from lop_osworld_v2_adapter.providers.fake import FakeProvider

    from local_operator.evaluation.adapters.api import ScoreParams, ScoreResult
    from local_operator.evaluation.evidence.models import (
        EvidenceArtifactRef,
        ScoreArtifact,
        ScoringResultPayload,
    )
    from local_operator.evaluation.evidence.verify import verify_bundle
    from local_operator.evaluation.receipts import RedactionSet
    from local_operator.evaluation.runner.episode import EpisodeRunner
    from tests.unit.evaluation.adapters.osworld.test_fake_end_to_end import (
        _adapter,
        _AdapterSupervisorShim,
        _selector,
        _spec_with_task,
    )
    from tests.unit.evaluation.runner.conftest import ScriptedModel, build_config

    assert "site-packages" in str(lop_osworld_v2_adapter.__file__)
    raw = _raw(case)
    provider = FakeProvider(scripted_score=raw)
    adapter = _adapter(root, provider)
    selector = _selector(root, adapter._workspace_root, adapter)
    shim = _AdapterSupervisorShim(adapter, selector)
    rescued: list[bool] = []

    async def rescue(*args: Any, **kwargs: Any) -> Any:
        # The production failure path must invoke resource recovery even if
        # details fail after evaluate(). No cloud resource is created here.
        rescued.append(True)
        from types import SimpleNamespace

        return SimpleNamespace(complete=True)

    model = ScriptedModel(["finish"])
    runner = EpisodeRunner(
        _spec_with_task("ep-score-evidence"),
        build_config(root),
        selector=selector,
        model=model,
        launch=lambda _: shim,
        rescue=rescue,
        redactions=RedactionSet.from_resolved_values(("score-detail-secret-canary-9827",)),
    )
    original_score = adapter.score

    async def damaged_score(params: ScoreParams) -> ScoreResult:
        result = await original_score(params)
        ref = result.score.details
        assert ref is not None and adapter._artifact_root is not None
        path = adapter._artifact_root / ref.sha256
        if case == "missing-detail":
            path.unlink()
        elif case == "invalid-media":
            data = b"not-json"
            path = adapter._artifact_root / hashlib.sha256(data).hexdigest()
            path.write_bytes(data)
            result = ScoreResult(
                score=ScoreArtifact(
                    status="scored",
                    binary=0,
                    details=EvidenceArtifactRef(
                        sha256=path.name, media_type="application/json", byte_count=len(data)
                    ),
                )
            )
        elif case in {"publication-io", "receipt-io"}:
            from tests.unit.evaluation.evidence.test_store import _OSCallsForTest

            class FailingCalls(_OSCallsForTest):
                calls = 0

                def fsync(self, fd: int) -> None:
                    self.calls += 1
                    # Temp file, artifact directory, then the scoring receipt.
                    if self.calls == (1 if case == "publication-io" else 3):
                        raise OSError("synthetic-score-detail-io")
                    super().fsync(fd)

            assert runner._writer is not None
            runner._writer._calls = FailingCalls()
        return result

    if case in {"missing-detail", "invalid-media", "publication-io", "receipt-io"}:
        adapter.score = damaged_score
    outcome = await runner.run()
    assert provider.evaluate_calls == 1
    assert model.calls == 1
    assert "synthetic audit" not in repr(model.histories)
    assert "score-detail-secret-canary-9827" not in repr(model.histories)
    assert shim.terminated
    assert outcome.bundle_root is not None
    report = verify_bundle(outcome.bundle_root)
    accepted = case in {"zero", "one", "fraction", "near-one", "dict", "missing-submission"}
    if accepted:
        assert outcome.status == "completed", outcome.diagnostic
        assert report.valid, [issue.code for issue in report.issues]
        score = outcome.score
        assert score is not None and score.details is not None
        data = (outcome.bundle_root / "artifacts" / score.details.sha256).read_bytes()
        assert json.loads(data) == raw
        assert hashlib.sha256(data).hexdigest() == score.details.sha256
        assert len(data) == score.details.byte_count
        assert report.outcome is not None and report.outcome.result == score
        receipts = [e.payload for e in report.events if isinstance(e.payload, ScoringResultPayload)]
        assert len(receipts) == 1 and receipts[0].score == score
        value = raw["score"] if isinstance(raw, dict) else raw
        assert score.binary == int(value == 1)
        assert score.partial_ppm == round(value * 1_000_000)
        assert provider.terminated_refs == ["lop-ep-ep-score-evidence"]
        return {
            "case": case,
            "status": outcome.status,
            "binary": score.binary,
            "partial_ppm": score.partial_ppm,
            "details_preserved": True,
            "verified": report.valid,
            "evaluate_calls": provider.evaluate_calls,
        }
    assert outcome.score is None and outcome.status != "completed"
    assert rescued, "detail rejection bypassed resource rescue"
    if case != "receipt-io":
        assert not any(isinstance(e.payload, ScoringResultPayload) for e in report.events)
    assert report.outcome is None
    # Staging is untrusted adapter output, not a sealed artifact. The writer
    # must not copy a secret into its bundle or expose it in outcome diagnostics.
    canary = b"score-detail-secret-canary-9827"
    assert canary not in (outcome.diagnostic or "").encode()
    for path in outcome.bundle_root.rglob("*"):
        if path.is_file():
            assert canary not in path.read_bytes()
    return {
        "case": case,
        "status": outcome.status,
        "score": None,
        "rescued": True,
        "evaluate_calls": provider.evaluate_calls,
    }


@pytest.mark.parametrize(
    "case",
    [
        "zero",
        "one",
        "fraction",
        "near-one",
        "dict",
        "missing-submission",
        "invalid",
        "nonfinite",
        "nonfinite-detail",
        "nonjson",
        "oversized",
        "secret",
        "missing-detail",
        "invalid-media",
        "publication-io",
        "receipt-io",
    ],
)
def test_installed_score_evidence(installed_python: str, tmp_path: Path, case: str) -> None:
    result = subprocess.run(
        [
            installed_python,
            "-c",
            "import asyncio,json,sys; from pathlib import Path; "
            "from tests.unit.evaluation.adapters.osworld.test_score_evidence import _exercise; "
            "print(json.dumps(asyncio.run(_exercise(Path(sys.argv[1]),sys.argv[2]))))",
            str(tmp_path),
            case,
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads(result.stdout.splitlines()[-1])
    assert summary["evaluate_calls"] == 1
    print(json.dumps(summary, sort_keys=True))
