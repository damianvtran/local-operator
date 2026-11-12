"""The evaluation arm hands off only published observation content."""

from __future__ import annotations

import base64
from types import SimpleNamespace

import pytest

from local_operator.evaluation.completion_verifier import CompletionVerifierConfig, SANDBOX_TOKEN
from local_operator.evaluation.protocol import FinishAction
from local_operator.evaluation.session_arm import _arm_completion_verifier
from local_operator.harness.verifier import VerifierVerdict
from tests.unit.evaluation.runner.test_provider_client import _framed_observation, _png
from tests.unit.evaluation.test_session_arm import _bridge, _command_result


@pytest.mark.asyncio
@pytest.mark.parametrize("unavailable", [False, True])
async def test_arm_transfers_verified_frames_or_records_an_inconclusive_refusal(
    tmp_path, unavailable
):
    root = tmp_path / "artifacts"
    pending = _framed_observation(root, 0)
    if unavailable:
        # Corrupt this test's own published artifact without introducing a path
        # argument: the frame digest/size authority must still reject its bytes.
        (root / pending.frames[0].artifact.sha256).write_bytes(b"corrupt")
    received = []

    async def verify_completion(**kwargs):
        received.append(kwargs)
        if not kwargs["typed_evidence"].images:
            return VerifierVerdict(
                status="INCONCLUSIVE", findings="required media missing", ended_by="refused"
            )
        return VerifierVerdict(status="PASS", findings="test adapter handoff", ended_by="verdict")

    async def run_command(*_args):
        return _command_result(stdout=SANDBOX_TOKEN)

    bridge = _bridge(tmp_path)
    bridge.run_command = run_command
    written = []
    state = _arm_completion_verifier(
        bridge=bridge,
        handle=SimpleNamespace(
            session=SimpleNamespace(verify_completion=verify_completion), verifier_job_ids=set()
        ),
        record=SimpleNamespace(write=lambda kind, payload: written.append((kind, payload))),
        config=CompletionVerifierConfig(rounds=1),
        artifact_root=root,
        wall_remaining=None,
    )
    verdict = await bridge.verifier(
        FinishAction(observation_id=pending.observation_id, status="done", reason="claim"),
        pending,
        1,
    )
    [kwargs] = received
    typed = kwargs["typed_evidence"]
    assert typed.required_modalities == frozenset({"image"})
    if unavailable:
        assert verdict.status == "INCONCLUSIVE"
        assert not typed.images
        assert any(kind == "verifier_evidence_unavailable" for kind, _ in written)
    else:
        assert verdict.status == "PASS"
        assert base64.b64decode(typed.images[0].data) == _png()
        assert typed.items[0].evidence_id == "frame:frame-0"
    assert state["verdicts"][0]["status"] == verdict.status
    assert set(kwargs) == {
        "request",
        "claim",
        "observation",
        "typed_evidence",
        "host_tools",
        "limits",
        "on_started",
    }
