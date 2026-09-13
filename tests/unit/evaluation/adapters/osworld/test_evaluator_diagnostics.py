"""The evaluator's own scoring-path output is retained as score evidence.

WHY this file exists. The archived paid runs could not answer the one question
that decides "apparatus fix" versus "the agent really failed": did a ``0.00%``
row come from an evaluator that bailed out before checking anything, or from
checkpoints that were genuinely unsatisfied? The per-checkpoint values exist at
scoring time (``task_002``'s four booleans, ``task_016``'s ``email_avg``,
``task_098``'s normalised results written into the task's cache directory) and
were discarded with the worker's streams. These tests pin the retention, its
bounds, and -- just as load-bearing -- what it must NOT change: the score, the
task files, the workspace digest, the worker's own stderr, and the byte-exact
detail artifact of an episode whose evaluator says nothing.

The bundles below are REAL: the real adapter, the real ``EpisodeRunner``, the
real evidence writer and the real verifier seal and read them. The fake
substitutes only ``evaluate()``'s upstream source.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from lop_osworld_v2_adapter import diagnostics, scoring
from lop_osworld_v2_adapter.providers.fake import FakeProvider

from local_operator.evaluation.adapters.discovery import workspace_digest
from local_operator.evaluation.evidence.models import ScoringResultPayload
from local_operator.evaluation.evidence.verify import verify_bundle
from local_operator.evaluation.receipts import RedactionSet
from local_operator.evaluation.runner.episode import EpisodeRunner
from tests.unit.evaluation.adapters.osworld import fixtures, spawn_helpers
from tests.unit.evaluation.adapters.osworld.test_build_and_scripts import (  # noqa: F401
    durable_path,
)
from tests.unit.evaluation.adapters.osworld.test_fake_end_to_end import (
    _adapter,
    _AdapterSupervisorShim,
    _selector,
    _spec_with_task,
)
from tests.unit.evaluation.runner.conftest import ScriptedModel, build_config

REPO = Path(__file__).resolve().parents[5]
SCRIPT = REPO / "scripts" / "run_episode.py"
# Canary credentials: the script requires the AWS secrets to exist, and the
# spawned run must not carry them into the retained diagnostics either.
CANARY_KEY = "AKIACANARY0000000001"
CANARY_SECRET = "canary-secret-value-9f8e7d6c5b4a"
# The non-secret account facts the adapter's own inspect_requirements names.
INFRA = [
    item
    for name in (
        "AWS_REGION=us-east-1",
        "AWS_SUBNET_ID=subnet-test",
        "AWS_SECURITY_GROUP_ID=sg-test",
        "AWS_SCHEDULER_ROLE_ARN=arn:aws:iam::0:role/test",
        "OSWORLD_CLIENT_PASSWORD=pw",
        "OSWORLD_FILE_BASE_URL=http://assets.test",
    )
    for item in ("--infra", name)
]

# The three shapes the archived runs lost, in the words the evaluators use.
PARTIALS_LINE = (
    "Task002 partials: course_load=False required_courses=False "
    "common_core=False nine_am=False score=0.00"
)
EMAIL_LINE = "email_avg: 0.25, linkedin_avg: 0.0, final_score: 0.125"
NORMALIZED = '{"expected_changed_fields": ["advisor", "courses"], "changed": []}'


async def _run(
    tmp_path: Path,
    episode_id: str,
    provider: FakeProvider,
    *,
    redactions: RedactionSet | None = None,
    rescued: list[bool] | None = None,
) -> Any:
    """Drive one real episode to a sealed bundle through the real runner."""

    adapter = _adapter(tmp_path, provider)
    selector = _selector(tmp_path, adapter._workspace_root, adapter)
    shim = _AdapterSupervisorShim(adapter, selector)
    if rescued is not None:

        async def rescue(*args: Any, **kwargs: Any) -> Any:
            from types import SimpleNamespace

            rescued.append(True)
            return SimpleNamespace(complete=True)

        rescue_hook: Any = rescue
    else:
        rescue_hook = None
    return await EpisodeRunner(
        _spec_with_task(episode_id),
        build_config(tmp_path),
        selector=selector,
        model=ScriptedModel(["finish"]),
        launch=lambda _selector: shim,
        redactions=redactions,
        **({"rescue": rescue_hook} if rescue_hook is not None else {}),
    ).run()


def _root(path: Path) -> Path:
    """An artifact root as the worker's own staging directory: it exists."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def _sealed_detail(outcome: Any) -> tuple[bytes, dict[str, Any], Any]:
    """The score-detail artifact as the bundle exposes it, plus its JSON."""

    assert outcome.bundle_root is not None, outcome.diagnostic
    report = verify_bundle(outcome.bundle_root)
    assert report.valid, [issue.code for issue in report.issues]
    score = outcome.score
    assert score is not None and score.details is not None
    receipts = [e.payload for e in report.events if isinstance(e.payload, ScoringResultPayload)]
    assert len(receipts) == 1 and receipts[0].score == score
    # Reachable from the SEALED bundle: the receipt asserts this ref and the
    # verifier independently accepted the bytes it names.
    assert receipts[0].score.details == score.details
    data = (outcome.bundle_root / "artifacts" / score.details.sha256).read_bytes()
    assert hashlib.sha256(data).hexdigest() == score.details.sha256
    assert len(data) == score.details.byte_count
    return data, json.loads(data), score


@pytest.mark.asyncio
async def test_evaluator_diagnostics_reach_the_sealed_bundle(
    tmp_path: Path, episode_id: str
) -> None:
    provider = FakeProvider(
        scripted_score=0.0,
        evaluator_prints=(EMAIL_LINE,),
        evaluator_logs=(PARTIALS_LINE,),
        evaluator_cache_files={"normalized_results.json": NORMALIZED},
    )

    outcome = await _run(tmp_path, episode_id, provider)

    assert outcome.status == "completed", outcome.diagnostic
    _, data, score = _sealed_detail(outcome)
    # The raw return is preserved verbatim under its own key, so the score half
    # of the artifact reads exactly as it did before diagnostics existed.
    assert data[scoring.EVALUATOR_RESULT_KEY] == 0.0
    block = data[scoring.EVALUATOR_DIAGNOSTICS_KEY]
    assert block["schema"] == diagnostics.DIAGNOSTICS_SCHEMA
    assert EMAIL_LINE in block["stdout"]["text"]
    assert PARTIALS_LINE in block["stderr"]["text"]
    assert "truncated" not in block["stdout"]
    # The state the evaluator fetched, from the task's own cache directory --
    # the path upstream derives (``cache_dir_base/<task_id>``) and the one the
    # provider was handed.
    entries = {entry["path"]: entry for entry in block["fetched_state"]["entries"]}
    assert entries["normalized_results.json"]["text"] == NORMALIZED
    assert block["fetched_state"]["entries_cut"] is False
    # Score-neutral: a 0.0 raw still seals 0 binary / 0 ppm, exactly as before.
    assert score.status == "scored" and score.binary == 0 and score.partial_ppm == 0


@pytest.mark.asyncio
async def test_a_silent_evaluator_keeps_the_pre_change_detail_bytes(
    tmp_path: Path, episode_id: str
) -> None:
    """The additive contract: nothing emitted, nothing fetched, no byte moved."""

    provider = FakeProvider(scripted_score=0.5)

    outcome = await _run(tmp_path, episode_id, provider)

    assert outcome.status == "completed", outcome.diagnostic
    sealed, data, score = _sealed_detail(outcome)
    # Independent of the implementation: today's canonical JSON for the raw 0.5.
    expected = json.dumps(
        0.5, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    assert sealed == expected
    assert data == 0.5
    # The pre-change code path for the same raw return: identical ref and
    # identical score identity, so the sealed artifact is bit-for-bit what it
    # would have been had the capture never run.
    control = scoring.score_to_artifact(0.5, artifact_root=_root(tmp_path / "control"))
    assert control.details == score.details
    assert control.score_id == score.score_id
    assert control.binary == score.binary and control.partial_ppm == score.partial_ppm


@pytest.mark.asyncio
async def test_oversized_stream_output_is_bounded_and_marks_the_cut(
    tmp_path: Path, episode_id: str
) -> None:
    """Over budget is CUT, reported as a count, and still scores."""

    filler = "noise " * 8_000  # 48_000 characters, well past the 32k ring
    provider = FakeProvider(scripted_score=0.0, evaluator_prints=(filler, EMAIL_LINE))

    outcome = await _run(tmp_path, episode_id, provider)

    assert outcome.status == "completed", outcome.diagnostic
    sealed, data, score = _sealed_detail(outcome)
    block = data[scoring.EVALUATOR_DIAGNOSTICS_KEY]
    text = block["stdout"]["text"]
    assert len(text) <= diagnostics.MAX_STREAM_CHARS
    # The TAIL is what a scoring path explains itself with, so the decisive
    # final line survives the bound.
    assert text.endswith(EMAIL_LINE + "\n")
    total = len(filler) + 1 + len(EMAIL_LINE) + 1
    assert block["stdout"]["truncated"] == total - len(text)
    assert len(sealed) < scoring.MAX_SCORE_DETAIL_BYTES
    assert score.status == "scored" and score.binary == 0 and score.partial_ppm == 0


@pytest.mark.parametrize("case", ["over-bytes", "surrogate", "nonjson"])
def test_over_budget_or_malformed_diagnostics_never_cost_the_score(
    tmp_path: Path, case: str
) -> None:
    """A rejected diagnostics block is reported, never charged to the score.

    The alternative -- failing the score because a diagnostic block could not
    be attached -- would let the explanation for a number destroy the number.
    """

    over_budget: dict[str, Any] = {
        "schema": diagnostics.DIAGNOSTICS_SCHEMA,
        "stdout": {"text": "x" * (scoring.MAX_SCORE_DETAIL_BYTES + 1)},
    }
    diagnostics_block: dict[str, Any] = {
        "over-bytes": over_budget,
        "surrogate": {"schema": diagnostics.DIAGNOSTICS_SCHEMA, "stdout": {"text": "\ud800"}},
        "nonjson": {"schema": diagnostics.DIAGNOSTICS_SCHEMA, "stdout": {"text": object()}},
    }[case]

    refused = scoring.score_to_artifact(
        1.0, artifact_root=_root(tmp_path / "a"), diagnostics=diagnostics_block
    )
    without = scoring.score_to_artifact(1.0, artifact_root=_root(tmp_path / "b"))

    # The number is untouched...
    assert refused.status == "scored" and refused.binary == 1
    assert refused.partial_ppm == without.partial_ppm
    assert refused.details is not None and without.details is not None
    staged = (tmp_path / "a" / refused.details.sha256).read_bytes()
    # ...and the refusal is recorded, so an absent block keeps meaning
    # "nothing was emitted" rather than "something was dropped".
    data = json.loads(staged)
    assert data[scoring.EVALUATOR_DIAGNOSTICS_KEY] == scoring.DIAGNOSTICS_REFUSED
    assert data[scoring.EVALUATOR_RESULT_KEY] == 1.0
    assert json.loads((tmp_path / "b" / without.details.sha256).read_bytes()) == 1.0


@pytest.mark.asyncio
async def test_a_canary_in_evaluator_output_is_withheld_whole(
    tmp_path: Path, episode_id: str
) -> None:
    """Retained diagnostics obey the score-detail redaction discipline.

    The canary travels the identical path as a score detail that carries one:
    the publication scan refuses, the episode is rescued rather than sealed,
    and the bytes never enter the bundle.
    """

    canary = "evaluator-output-canary-9827"
    provider = FakeProvider(scripted_score=1.0, evaluator_prints=(f"final_score: 1.0 {canary}",))
    rescued: list[bool] = []

    outcome = await _run(
        tmp_path,
        episode_id,
        provider,
        redactions=RedactionSet.from_resolved_values((canary,)),
        rescued=rescued,
    )

    assert rescued, "detail rejection bypassed resource rescue"
    assert outcome.score is None and outcome.status != "completed"
    assert outcome.bundle_root is not None
    assert canary not in (outcome.diagnostic or "")
    for path in outcome.bundle_root.rglob("*"):
        if path.is_file():
            assert canary.encode() not in path.read_bytes()


@pytest.mark.asyncio
async def test_the_capture_touches_neither_the_task_file_nor_the_workspace_digest(
    tmp_path: Path, episode_id: str
) -> None:
    """The apparatus digest and the corpus files are untouched by this change.

    No task file may be rewritten and no byte may appear inside the
    digest-pinned workspace: either would change what the harness graded and
    break comparability with the published rows and our own earlier arms.
    """

    provider = FakeProvider(
        scripted_score=1.0,
        evaluator_prints=(EMAIL_LINE,),
        evaluator_logs=(PARTIALS_LINE,),
        evaluator_cache_files={"normalized_results.json": NORMALIZED},
    )
    adapter = _adapter(tmp_path, provider)
    workspace = adapter._workspace_root
    task_file = workspace / "tasks" / "task_plain.py"
    before_bytes = task_file.read_bytes()
    before_mtime = task_file.stat().st_mtime_ns
    before_digest = workspace_digest(str(workspace))
    before_tree = sorted(p.relative_to(workspace).as_posix() for p in workspace.rglob("*"))

    selector = _selector(tmp_path, workspace, adapter)
    shim = _AdapterSupervisorShim(adapter, selector)
    outcome = await EpisodeRunner(
        _spec_with_task(episode_id),
        build_config(tmp_path),
        selector=selector,
        model=ScriptedModel(["finish"]),
        launch=lambda _selector: shim,
    ).run()

    assert outcome.status == "completed", outcome.diagnostic
    assert task_file.read_bytes() == before_bytes
    assert task_file.stat().st_mtime_ns == before_mtime
    assert workspace_digest(str(workspace)) == before_digest
    assert sorted(p.relative_to(workspace).as_posix() for p in workspace.rglob("*")) == before_tree
    # The episode's own writes went to the episode cache, outside the workspace.
    assert (tmp_path / "osworld-cache" / episode_id / "task_plain").is_dir()


def test_the_capture_keeps_the_streams_and_the_log_levels_it_found() -> None:
    """Retention must not divert the worker's output or leak a level change.

    The worker's stderr tail is a real failure-path input (``EpisodeRunner.
    ``_adapter_stderr``), so a capture that swallowed the evaluator's records
    would trade one blind spot for another. Records the running configuration
    would NOT have emitted (an INFO line under the default WARNING gate) are
    retained without being invented onto stderr either.
    """

    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    task_logger = logging.getLogger("desktopenv.task002")
    saved_level = task_logger.level
    out, err = io.StringIO(), io.StringIO()
    try:
        # The worker's own configuration: no handlers anywhere, logging's
        # lastResort doing the emitting (OSWorld never calls basicConfig).
        root.handlers.clear()
        sys.stdout, sys.stderr = out, err
        with diagnostics.capture_evaluator_diagnostics() as capture:
            print(EMAIL_LINE)
            task_logger.warning("checkpoint lookup failed")
            task_logger.info(PARTIALS_LINE)
        block = capture.payload()
        assert block is not None
        # Tee'd, not diverted: what the streams carried before, they still carry.
        assert out.getvalue() == EMAIL_LINE + "\n"
        assert err.getvalue() == "checkpoint lookup failed\n"
        # Retained: both records, including the INFO line the WARNING gate drops.
        assert "checkpoint lookup failed" in block["stderr"]["text"]
        assert PARTIALS_LINE in block["stderr"]["text"]
        assert EMAIL_LINE in block["stdout"]["text"]
        assert task_logger.level == saved_level

        # With a foreign handler already emitting them, nothing is duplicated
        # onto the worker's stderr.
        foreign = logging.StreamHandler(err)
        root.addHandler(foreign)
        err.seek(0)
        err.truncate()
        with diagnostics.capture_evaluator_diagnostics() as second:
            task_logger.warning("only once")
        assert err.getvalue() == "only once\n"
        second_block = second.payload()
        assert second_block is not None and "only once" in second_block["stderr"]["text"]
        root.removeHandler(foreign)
    finally:
        root.handlers[:] = saved_handlers
        sys.stdout, sys.stderr = saved_stdout, saved_stderr
        task_logger.setLevel(saved_level)


def test_fetched_state_is_bounded_and_omits_binary_content(tmp_path: Path) -> None:
    """The manifest is bounded, never follows a symlink, and says why it cut."""

    cache = tmp_path / "002"
    cache.mkdir()
    (cache / "aa-results.json").write_text(NORMALIZED)
    (cache / "bb-blob.bin").write_bytes(b"\x00\x01\x02")
    (cache / "cc-latin.txt").write_bytes("caf\xe9".encode("latin-1"))
    (cache / "dd-large.txt").write_text("y" * (diagnostics.MAX_FETCHED_TEXT_CHARS + 1))
    (cache / "ee-link.txt").symlink_to("/etc/hosts")
    (cache / "ff-link-dir").symlink_to(tmp_path)
    for index in range(diagnostics.MAX_FETCHED_ENTRIES + 4):
        (cache / f"zz-{index:04d}.txt").write_text("entry")

    with diagnostics.capture_evaluator_diagnostics(cache_dir=cache) as capture:
        pass
    block = capture.payload()
    assert block is not None
    fetched = block["fetched_state"]
    assert fetched["entries_cut"] is True
    assert len(fetched["entries"]) == diagnostics.MAX_FETCHED_ENTRIES
    entries = {entry["path"]: entry for entry in fetched["entries"]}
    assert entries["aa-results.json"]["text"] == NORMALIZED
    assert entries["bb-blob.bin"]["text_omitted"] == "binary"
    assert entries["cc-latin.txt"]["text_omitted"] == "not_utf8"
    assert entries["dd-large.txt"]["text_omitted"] == "too_large"
    assert entries["ee-link.txt"] == {"path": "ee-link.txt", "text_omitted": "symlink"}
    # A symlinked directory is neither followed nor dropped silently.
    assert entries["ff-link-dir"] == {"path": "ff-link-dir", "text_omitted": "symlink"}

    # An existing but EMPTY task cache is not fetched state: upstream creates
    # that directory during setup for every task, so an empty one must leave the
    # detail bytes of a silent evaluator unchanged.
    empty = tmp_path / "003"
    empty.mkdir()
    with diagnostics.capture_evaluator_diagnostics(cache_dir=empty) as quiet:
        pass
    assert quiet.payload() is None
    with diagnostics.capture_evaluator_diagnostics(cache_dir=tmp_path / "absent") as absent:
        pass
    assert absent.payload() is None


# ----------------------------------------------------------------------------
# The spawned-worker proof
# ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def adapter_wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return spawn_helpers.build_adapter_wheel(tmp_path_factory.mktemp("diagnostics-wheel"))


@pytest.mark.slow
def test_a_spawned_episode_retains_the_diagnostics_it_used_to_discard(
    durable_path: Path, adapter_wheel: Path  # noqa: F811
) -> None:
    """The same retention, proved out of process through the real script.

    In the spawned worker there is no pytest logging machinery and stdout is a
    pipe to the supervisor, which is exactly the configuration the archived paid
    runs scored in -- and the one where the evaluator's output previously ended
    up nowhere.
    """

    selector = spawn_helpers.build_spawnable_adapter(
        durable_path / "adapter",
        adapter_wheel,
        {"task_plain": fixtures.PLAIN},
        provider={
            "provider": "fake",
            "scripted_score": 0.0,
            "evaluator_prints": [EMAIL_LINE],
            "evaluator_logs": [PARTIALS_LINE],
            "evaluator_cache_files": {"normalized_results.json": NORMALIZED},
        },
    )
    selector_path = durable_path / "adapter" / "selector.json"
    selector_path.write_text(selector.model_dump_json())
    run_root = durable_path / "run"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--selector",
            str(selector_path),
            "--task-id",
            "task_plain",
            "--route",
            "test/fake/model:free",
            "--run-root",
            str(run_root),
            "--secret-env",
            "AWS_ACCESS_KEY_ID",
            "--secret-env",
            "AWS_SECRET_ACCESS_KEY",
            "--no-store",
            "--model-client",
            "scripted-finish",
            "--max-steps",
            "3",
            "--max-usd",
            "0.01",
            *INFRA,
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        env={
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
            "PYTHONPATH": str(REPO),
            "AWS_ACCESS_KEY_ID": CANARY_KEY,
            "AWS_SECRET_ACCESS_KEY": CANARY_SECRET,
        },
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    outcome: dict[str, Any] = json.loads(completed.stdout)
    assert outcome["status"] == "completed", outcome
    score = outcome["score"]
    assert score["status"] == "scored" and score["binary"] == 0 and score["partial_ppm"] == 0

    bundle = Path(outcome["bundle_root"])
    report = verify_bundle(bundle)
    assert report.valid, [issue.code for issue in report.issues]
    data = json.loads((bundle / "artifacts" / score["details"]["sha256"]).read_bytes())
    block = data[scoring.EVALUATOR_DIAGNOSTICS_KEY]
    assert data[scoring.EVALUATOR_RESULT_KEY] == 0.0
    assert EMAIL_LINE in block["stdout"]["text"]
    assert PARTIALS_LINE in block["stderr"]["text"]
    entries = {entry["path"]: entry for entry in block["fetched_state"]["entries"]}
    assert entries["normalized_results.json"]["text"] == NORMALIZED
    # Retention is a new place bytes can land, so the secret discipline has to
    # hold there too: the canary credentials reach the worker over the private
    # pipe and appear in no file of the run, retained diagnostics included.
    for path in sorted(run_root.rglob("*")):
        if path.is_file():
            assert CANARY_SECRET.encode() not in path.read_bytes()
            assert CANARY_KEY.encode() not in path.read_bytes()
