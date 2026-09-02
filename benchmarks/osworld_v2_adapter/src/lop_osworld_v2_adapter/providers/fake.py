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

from pathlib import Path
from typing import Any

from lop_osworld_v2_adapter.cleanup import (
    EVIDENCE_INSTANCE_ABSENT,
    EVIDENCE_INSTANCE_TERMINATED,
    EVIDENCE_SCHEDULE_ABSENT,
    EVIDENCE_SCHEDULE_DELETED,
)
from lop_osworld_v2_adapter.observation import NATIVE_SCREEN, write_png_rgb
from lop_osworld_v2_adapter.provisioning import ProvisioningPlan
from lop_osworld_v2_adapter.taskfile import TaskDescriptor


class FakeProvider:
    """Satisfies EnvironmentProvider with no cloud, no network, no spend."""

    def __init__(
        self,
        *,
        scripted_score: float = 1.0,
        fail_evaluate: bool = False,
        has_user_simulator: bool = False,
        simulator_answer: str = "simulated user answer",
    ) -> None:
        self._scripted_score = scripted_score
        self._fail_evaluate = fail_evaluate
        self._has_user_simulator = has_user_simulator
        self._simulator_answer = simulator_answer
        # The in-memory registry stands in for EC2: ref -> state. Teardown
        # looks the instance up BY REF, exactly as the AWS provider will.
        self._instances: dict[str, dict[str, Any]] = {}
        self._schedules: dict[str, dict[str, Any]] = {}
        self._sequence = 0
        self.allocated = False
        self.executed_statements: list[str] = []
        self.terminated_refs: list[str] = []
        self.deleted_schedules: list[str] = []
        self.evaluate_calls = 0
        # On the paid path each observe() is a live HTTP round-trip to the
        # guest (screenshot + a11y tree), so a duplicated call is real cost,
        # not a cosmetic issue. Counted here so a test can pin it.
        self.observe_calls = 0
        # Where the adapter told us to cache. Recorded (never used to write
        # anything: the fake downloads nothing) so tests assert the cache
        # root actually crossed the adapter -> provider boundary.
        self.cache_root: Path | None = None

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
        self, plan: ProvisioningPlan, task: TaskDescriptor, *, cache_root: Path | None = None
    ) -> None:
        # The ref is the tag; allocation registers the instance under it, so
        # teardown-by-ref is the same operation a rescue worker performs.
        self.cache_root = cache_root
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
        return {
            "screenshot": self._frame(),
            "accessibility_tree": None,
            "terminal": None,
            "instruction": "fake instruction",
        }

    async def execute(self, statements: list[str]) -> None:
        self.executed_statements.extend(statements)
        self._sequence += 1

    async def evaluate(self) -> Any:
        self.evaluate_calls += 1
        if self._fail_evaluate:
            raise RuntimeError("scripted evaluator failure")
        return self._scripted_score

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
