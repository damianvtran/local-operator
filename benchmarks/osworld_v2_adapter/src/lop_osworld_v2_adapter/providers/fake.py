"""FakeProvider: an in-process backend that drives a real EpisodeRunner.

The point of this provider is not to simulate OSWorld faithfully — it cannot.
It is to exercise the ENTIRE adapter (RPC, state machine, observation
identity, cleanup receipts, rescue) through the real ``EpisodeRunner`` with
zero cloud spend. Everything except "boto3 calls are shaped correctly and the
OSWorld guest behaves" is proved here, which is why this is the highest-value
test in PR 1.

The provider keeps an in-memory instance registry keyed by the tag ref, so
cleanup's ``describe``/``terminate`` exercise the exact resolution path the
AWS provider will: teardown resolves the instance from the tag, never from a
stored ID, which is what a rescue worker with only the descriptor must do.

Frames are generated in-process with the stdlib PNG encoder, so the harness's
``validate_media`` and ``verify_artifact`` accept them exactly as they would a
real guest screenshot. The frame content changes with the sequence number so
two steps never hash to the same artifact.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

from lop_osworld_v2_adapter.cleanup import (
    EVIDENCE_INSTANCE_ABSENT,
    EVIDENCE_INSTANCE_TERMINATED,
    EVIDENCE_SCHEDULE_ABSENT,
    EVIDENCE_SCHEDULE_DELETED,
)
from lop_osworld_v2_adapter.observation import (
    NATIVE_SCREEN,
    SCREENSHOT_CAUSE_KEY,
    write_png_rgb,
)
from lop_osworld_v2_adapter.providers.base import bounded_observation_cause
from lop_osworld_v2_adapter.provisioning import ProvisioningPlan
from lop_osworld_v2_adapter.taskfile import TaskDescriptor


class FakeProvider:
    """Satisfies EnvironmentProvider with no cloud, no network, no spend."""

    def __init__(
        self,
        *,
        scripted_score: float = 1.0,
        fail_evaluate: bool = False,
        blind_observations: int = 0,
        blind_after_observe_calls: int = 0,
        blind_cause: str | None = None,
        has_user_simulator: bool = False,
        simulator_answer: str = "simulated user answer",
        evaluator_prints: tuple[str, ...] = (),
        evaluator_logs: tuple[str, ...] = (),
        evaluator_cache_files: Mapping[str, str] | None = None,
    ) -> None:
        self._scripted_score = scripted_score
        self._fail_evaluate = fail_evaluate
        # The evaluator-diagnostics seams. Real evaluators compute per-checkpoint
        # values and emit them as prints, log records on the ``desktopenv``
        # namespaces, or files in the task's cache directory; the fake has to do
        # the same to drive ``diagnostics`` at all. Empty by default, so every
        # existing test (and every episode whose evaluator says nothing) behaves
        # exactly as it did before these knobs existed.
        self._evaluator_prints = tuple(evaluator_prints)
        self._evaluator_logs = tuple(evaluator_logs)
        self._evaluator_cache_files = dict(evaluator_cache_files or {})
        # How many upcoming observe() calls return NO frame, reproducing a
        # guest whose screenshot server is starved (the burstable-instance
        # failure that destroyed five paid episodes). OSWorld's own
        # ``_get_obs`` returns None for the screenshot after exhausting its
        # internal retries rather than raising, so the fake does the same.
        self._blind_observations = blind_observations
        # Which observe() call the blindness STARTS at. The real outage began
        # mid-episode, after reset_start had already produced a good frame, so
        # a test that blinded the very first read would exercise a different
        # (and easier) failure than the one that cost five paid episodes.
        self._blind_after_observe_calls = blind_after_observe_calls
        # WHY the blinded read has no frame, as the provider's bounded account.
        # The real provider derives this from upstream's own failed-attempt
        # logging (``providers.aws``); the fake takes it from the test so a
        # bundle assertion can prove the cause reaches the sealed error detail.
        # ``None`` is the provider that knows nothing: it reproduces upstream's
        # silent ``None`` exactly, so the absent-cause path is unchanged.
        self._blind_cause = blind_cause
        self._has_user_simulator = has_user_simulator
        self._simulator_answer = simulator_answer
        # The in-memory registry stands in for EC2: ref -> state. Teardown
        # looks the instance up BY REF, exactly as the AWS provider will.
        self._instances: dict[str, dict[str, Any]] = {}
        self._schedules: dict[str, dict[str, Any]] = {}
        self._sequence = 0
        self.allocated = False
        self.executed_statements: list[str] = []
        self.settle_flags: list[bool] = []
        self.terminated_refs: list[str] = []
        self.deleted_schedules: list[str] = []
        self.evaluate_calls = 0
        # On the paid path each observe() is a live HTTP round-trip to the
        # guest (screenshot + a11y tree), so a duplicated call is real cost,
        # not a cosmetic issue. Counted here so a test can pin it.
        self.observe_calls = 0
        # Where the adapter told us to cache. Recorded (never written to unless
        # a diagnostics seam asks for it) so tests assert the cache root
        # actually crossed the adapter -> provider boundary. None until allocate.
        self.cache_root: Path | None = None
        # The task id upstream would name its own cache subdirectory with
        # (``DesktopEnv`` derives ``cache_dir_base/<task_id>``), so a seam that
        # writes cache files lands exactly where a real evaluator writes.
        self._task_id: str | None = None

    def _frame(self) -> bytes:
        """A deterministic but sequence-varying 1920x1080 PNG frame."""
        # Vary one channel with the sequence so consecutive frames hash
        # differently; a static frame would make two steps collide on the same
        # artifact, which is fine for the verifier but useless as evidence of
        # progression.
        width, height = NATIVE_SCREEN.width, NATIVE_SCREEN.height
        shade = self._sequence % 256
        pixel = bytes((shade, (shade * 3) % 256, (shade * 7) % 256))
        return write_png_rgb(width, height, pixel * (width * height))

    async def allocate(
        self, plan: ProvisioningPlan, task: TaskDescriptor, *, cache_root: Path
    ) -> None:
        # The ref is the tag; allocation registers the instance under it, so
        # teardown-by-ref is the same operation a rescue worker performs.
        self.cache_root = cache_root
        self._task_id = task.task_id
        self._instances[plan.tag_dict()["Name"]] = {
            "state": "running",
            "task_id": task.task_id,
            "client_token": plan.client_token,
        }
        self._schedules[f"lop-ttl-{plan.tag_dict()['lop:episode']}"] = {"state": "active"}
        self.allocated = True
        self._sequence = 0

    async def observe(self) -> dict[str, Any]:
        self.observe_calls += 1
        if self._blind_observations > 0 and self.observe_calls > self._blind_after_observe_calls:
            self._blind_observations -= 1
            blind: dict[str, Any] = {
                "screenshot": None,
                "accessibility_tree": None,
                "terminal": None,
                "instruction": "fake instruction",
            }
            if self._blind_cause is not None:
                blind[SCREENSHOT_CAUSE_KEY] = bounded_observation_cause(self._blind_cause)
            return blind
        return {
            "screenshot": self._frame(),
            "accessibility_tree": None,
            "terminal": None,
            "instruction": "fake instruction",
        }

    async def execute(self, statements: list[str], *, settle: bool = True) -> None:
        # ``settle`` is accepted to satisfy the provider protocol; the fake has
        # no desktop to repaint, so there is nothing to pause for. Recorded so
        # a test can still assert that a split batch settles exactly once.
        self.executed_statements.extend(statements)
        self.settle_flags.append(settle)
        self._sequence += 1

    async def evaluate(self) -> Any:
        self.evaluate_calls += 1
        if self._fail_evaluate:
            raise RuntimeError("scripted evaluator failure")
        self._emit_evaluator_diagnostics()
        return self._scripted_score

    def _emit_evaluator_diagnostics(self) -> None:
        """Reproduce what a real evaluator emits while it scores.

        The three shapes are the ones the archived runs could not recover:
        prints (task_016's ``email_avg``), log records on a ``desktopenv``
        namespace (task_002's INFO-level ``Task002 partials``, which the
        WARNING gate drops before any handler sees it), and files written into
        the task's own cache directory (task_098).
        """

        for line in self._evaluator_prints:
            print(line)
        for line in self._evaluator_logs:
            logging.getLogger("desktopenv.fake_evaluator").info(line)
        if self._evaluator_cache_files:
            assert self.cache_root is not None
            task_id = self._task_id or "task"
            directory = Path(self.cache_root) / task_id
            directory.mkdir(parents=True, exist_ok=True)
            for name, content in self._evaluator_cache_files.items():
                (directory / name).write_text(content)

    async def terminate(self, instance_ref: str) -> str:
        instance = self._instances.get(instance_ref)
        if instance is None:
            return EVIDENCE_INSTANCE_ABSENT
        instance["state"] = "terminated"
        self.terminated_refs.append(instance_ref)
        return EVIDENCE_INSTANCE_TERMINATED

    async def delete_schedule(self, lease_ref: str) -> str:
        if lease_ref not in self._schedules:
            return EVIDENCE_SCHEDULE_ABSENT
        del self._schedules[lease_ref]
        self.deleted_schedules.append(lease_ref)
        return EVIDENCE_SCHEDULE_DELETED

    async def describe(self, instance_ref: str) -> dict[str, Any] | None:
        return self._instances.get(instance_ref)

    async def respond(self, prompt: str) -> str | None:
        if not self._has_user_simulator:
            return None
        return self._simulator_answer
