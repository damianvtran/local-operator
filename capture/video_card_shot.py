"""Before/after capture rig for the video surfaces lane (TUI side).

One script, two runs: point LOP_TREE at the base tree for BEFORE frames, then
at the branch tree for AFTER — the state matrix is identical, so the pair
reads as the change and nothing else.

    cd <tree>
    LOP_TREE=<tree> LOP_SHIMMER=on|off \
      <branch-venv>/bin/python "$LOCAL_OPERATOR_SCRATCHPAD/video_card_shot.py" \
      STATE OUT.svg [COLSxROWS]

STATE: queued | queued-body | running | running-body | cancelling | settled | band | held

The tree is inserted at the FRONT of sys.path before any local import, and the
script asserts what it actually imported, so a capture can never come from the
wrong tree (the AGENTS.md before-frame trap). The caller also sets
PYTHONPATH=<tree> for the throwaway base checkout, where the venv's editable
finder would otherwise win over sys.path.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

TREE = Path(os.environ.get("LOP_TREE") or os.getcwd()).resolve()
sys.path.insert(0, str(TREE))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()
# Spec §7 wants the running canvas BOTH as its normal shimmer phase and as the
# reduced/flat frame; isolate_capture() pins flat, so a shimmer run unsets the
# kill switch before the app import.
if os.environ.get("LOP_SHIMMER", "on") != "off":
    os.environ.pop("LOCAL_OPERATOR_NO_SHIMMER", None)

from local_operator.harness.types import (  # noqa: E402
    AgentToolUpdate,
    TextContent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    ToolResult,
)
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.events import ToolEnded, ToolStarted, ToolUpdated  # noqa: E402
from local_operator.tui.widgets.transcript import UserBlock  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

import local_operator  # noqa: E402

assert Path(local_operator.__file__).resolve().is_relative_to(TREE), (
    f"wrong tree: imported {local_operator.__file__}, expected under {TREE}"
)

VIDEO = "generate_video"
PROMPT = "a cat surfing, golden hour"
#: The §2.4 SUBMIT receipt — the text the card actually settles with (the call
#: is detached; the tool returns this instantly and the minutes-long job runs
#: elsewhere). Remediation round 1, D1: the earlier frames staged the completion
#: sentence here, which belongs to the delivered `job_result` row (the `held`
#: state below shows it where it lives).
RECEIPT = (
    "Video generation started — job `abc123` (mode i2v, 8 s, ~$1.60 est., Radient). "
    "You'll get a completion notice when it settles; cancel with "
    "`jobs op='cancel', job_id='abc123'`."
)
#: The §5.1 completion sentence, as the `job_result` delivery carries it.
COMPLETION = (
    "background job 'Video generation (i2v, 8s, Radient)' completed:\n"
    "Video ready — job abc123 · model wan-3.0-prime · 8.0s · $1.60 (reported) · "
    "attached: digest 0123456789abcdef0123456789abcdef"
)


class _Job:
    """The band-counter fixture shape (tests/unit/tui/test_band_panels.py)."""

    def __init__(self, job_id: str, kind: str, label: str) -> None:
        self.id = job_id
        self.type = kind
        self.status = "running"
        self.queued = False
        self.label = label
        self.start_time = 1_700_000_000.0


class _Ledger:
    def __init__(self, *jobs: _Job) -> None:
        self._jobs = list(jobs)

    def list(self, *, registrant_id: str | None = None) -> list[_Job]:
        return list(self._jobs)

    def get(self, job_id: str, *, registrant_id: str | None = None) -> _Job | None:
        return next((j for j in self._jobs if j.id == job_id), None)


def _start(app, tool: str = VIDEO) -> None:
    app.post_message(
        ToolStarted(
            ToolExecutionStartEvent(tool_call_id="v0", tool_name=tool, args={"prompt": PROMPT})
        )
    )


def _update(app, tool: str, details: dict) -> None:
    app.post_message(
        ToolUpdated(
            ToolExecutionUpdateEvent(
                tool_call_id="v0",
                tool_name=tool,
                partial_result=AgentToolUpdate(details=details),
            )
        )
    )


async def main() -> None:
    state = sys.argv[1]
    out = sys.argv[2]
    size = (100, 30)
    if len(sys.argv) > 3:
        cols, rows = sys.argv[3].split("x")
        size = (int(cols), int(rows))
    # The CONTROL frame: the same running state for the image tool, whose row
    # carries the interrupt hint — the pair against `running` proves the
    # exclusion is video-only rather than a hint that silently died.
    tool = "generate_image" if state == "running-image" else VIDEO

    session = FakeSession()
    if state == "band":
        session.jobs = _Ledger(
            _Job("vid-1", "video", "Video generation (i2v, 8s, Radient)"),
            _Job("bash-1", "bash", "pytest -q"),
        )
    if state == "held":
        # The §7 frame for R6's delivery row: a held video completion, built
        # exactly as tests/unit/tui/test_harness_injection_replay.py builds
        # its rows — a job_result custom message carrying the held flag — and
        # replayed by the same boot seam every resumed session uses.
        from local_operator.harness.jobs import JOB_RESULT_MESSAGE_TYPE
        from local_operator.harness.types import CustomMessage

        session._history = [
            CustomMessage(
                custom_type=JOB_RESULT_MESSAGE_TYPE,
                attribution="user",
                details={"job_id": "abc123", "text": COMPLETION, "held": True},
            )
        ]
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        # Wait for the session to boot (the gate the app tests use), or
        # `_poll_subagents` reads a None session and the band stays empty.
        for _ in range(80):
            await pilot.pause()
            if app._session is not None:
                break
        if state == "band":
            app._append_block(UserBlock("Make me a video of a cat surfing."))
            await pilot.pause()
            app._poll_subagents()
            for _ in range(8):
                await pilot.pause()
            save_capture(app, out)
            return

        if state == "held":
            for _ in range(8):
                await pilot.pause()
            save_capture(app, out)
            return

        app._append_block(UserBlock(f"Generate a video: {PROMPT}"))
        await pilot.pause()
        _start(app, tool)
        await pilot.pause()
        card = app._tool_cards["v0"]

        if state.startswith("queued"):
            card.mark_queued()
            _update(app, tool, {"stage": "queued", "queue_position": 2})
        elif state.startswith("running"):
            _update(
                app,
                tool,
                {
                    "stage": "in_progress",
                    "queue_position": None,
                    "progress_fraction": None,
                    "log_lines": [{"message": "submitted to fal-ai/wan/v2.7 (i2v, 8s)"}],
                },
            )
        elif state == "cancelling":
            _update(
                app,
                tool,
                {
                    "stage": "cancelling",
                    "queue_position": None,
                    "progress_fraction": None,
                    "log_lines": None,
                },
            )
        elif state == "settled":
            app.post_message(
                ToolEnded(
                    ToolExecutionEndEvent(
                        tool_call_id="v0",
                        tool_name=tool,
                        result=ToolResult(
                            tool_call_id="v0",
                            tool_name=tool,
                            content=[TextContent(text=RECEIPT)],
                            # The §2.4 details as the lane's tool returns them
                            # for the submit (stage queued — honest at submit
                            # time; the completion rides the later delivery).
                            # No `AttachmentContent` here on purpose: a kind-video
                            # attachment rides the COMPLETION, and "mounts
                            # nothing broken" is already pinned by its own test
                            # (test_tool_result_image_blocks.py; D1).
                            details={
                                "job_id": "abc123",
                                "stage": "queued",
                                "provider": "radient",
                                "route_label": "Radient",
                                "cancel_support": "queued_only",
                                "cancel_hint": "jobs op='cancel', job_id='abc123'",
                            },
                        ),
                        # The instant submit's measured interval — the everyday
                        # settled spine (the bare ✓ without it is the reduced
                        # case, which the r1 frames showed).
                        duration_s=0.4,
                    )
                )
            )
        else:
            raise SystemExit(f"unknown state {state!r}")

        for _ in range(6):
            await pilot.pause()
        if state.endswith("-body") or state in ("cancelling", "settled"):
            assert card.can_expand(), "card cannot expand — rig state wrong"
            card.toggle_expanded()
            for _ in range(6):
                await pilot.pause()
        save_capture(app, out)


asyncio.run(main())
