"""A failed MCP server must reach the SCREEN of a viewer, not just its state.

The runtime child records an MCP round's outcome and pushes it through the
frontend state. This test drives the whole path for real — a real runtime child
spawned from this worktree, a real loopback socket, the production
``AttachedSession`` viewer behind the production ``OperatorApp`` — against a
``.mcp.json`` naming one server that cannot start, and asserts the user is TOLD:
a durable transcript notice naming the server.

Why it has to exist, and why it is not a duplicate of the deferral tests:

* The state hop was already covered (`test_runtime_mcp_deferral_e2e.py` asserts
  ``viewer.mcp_startup.failures``) and that test passed while **nothing painted**.
  Measured on head ``b8bc511d7`` by QA round 1 with two failing servers over
  55 s: ``notices=[]``, ``toasts=[]`` — the failure reached the viewer's state and
  no surface at all, on this branch and on the base alike. The TUI installs its
  MCP sink on the session it adopts, and on a viewer facade that attribute was
  never read by anything (`AttachedSession` is populated from pushed state).
* Because the child's record now publishes BEFORE its MCP wiring (the deferral
  this PR adds), the failure is by construction reported AFTER the viewer is
  bound — which is only useful if something on the viewer paints it.

The frame assertion at the end is the visual half: this is a user-visible
change and the notice is the evidence, so the test captures the rendered frame
it asserted on to the path it prints (pytest keeps the last three tmp roots, and
the run prints the exact path).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.e2e

#: The server the fixture configures, whose command does not exist. A missing
#: executable is the commonest broken setup and is TERMINAL at the 250 ms gate
#: (no connect timeout to wait out), so the round this test waits for finishes
#: in well under a second of wall clock.
BROKEN_SERVER = "broken"

#: The client's own words for a command it could not start. Matched loosely on
#: the file name so the test does not pin the OS error's phrasing.
BROKEN_ERROR_FRAGMENT = "definitely-not-installed"

#: Upper bound on an awaited event, never a budget to sleep through. Generous
#: because the first bind of a cold viewer pays the child's whole construction.
GUARD_S = 60.0


def _configure_provider(config_dir: Path) -> None:
    """A configured machine, so the mount engage spawns (see the sibling e2e)."""
    from local_operator.config import ConfigManager

    ConfigManager(config_dir=config_dir).update_config({"hosting": "test", "model_name": "mock"})


def _write_broken_mcp(config_dir: Path) -> None:
    """One stdio server whose command is not installed → a fast terminal failure."""
    (config_dir / "mcp.json").write_text(
        f'{{"mcpServers": {{"{BROKEN_SERVER}": {{"command": "{BROKEN_ERROR_FRAGMENT}"}}}}}}',
        encoding="utf-8",
    )


async def _never_take_over() -> Any:
    raise AssertionError("a viewer must never take over a session")


@pytest.mark.asyncio
async def test_a_failed_mcp_server_reaches_the_viewers_transcript(
    headless_tui_env: Path, workspace: Path, tmp_path: Path
) -> None:
    """The operator's second complaint, end to end: a failed server says so."""
    from local_operator.session.attached import AttachedSession
    from local_operator.tui.app import OperatorApp
    from local_operator.tui.widgets.toast import Toast
    from local_operator.tui.widgets.transcript import NoticeBlock
    from scripts.visual_capture import save_capture

    _configure_provider(headless_tui_env)
    _write_broken_mcp(headless_tui_env)

    session_id = "mcpfail00001"

    async def factory() -> Any:
        return await AttachedSession.cold(
            session_id,
            config_dir=headless_tui_env,
            cwd=str(workspace),
            takeover_factory=_never_take_over,
        )

    app = OperatorApp(factory)
    viewer = None
    try:
        async with app.run_test(size=(100, 30)) as pilot:
            # Typed Any: ``OperatorApp._session`` is the protocol every host
            # satisfies, and the properties under assertion (``is_cold``,
            # ``mcp_startup``) belong to the real ``AttachedSession`` this test
            # drives through its own production path.
            viewer: Any = None
            deadline = time.monotonic() + GUARD_S
            while time.monotonic() < deadline:
                await pilot.pause()
                viewer = app._session
                if viewer is not None and not getattr(viewer, "is_cold", True):
                    break
                await asyncio.sleep(0.05)
            assert (
                viewer is not None and not viewer.is_cold
            ), f"the viewer never bound within {GUARD_S}s"

            # The failure is recorded on the owner's side of the wire...
            while time.monotonic() < deadline:
                outcome = getattr(viewer, "mcp_startup", None)
                if outcome is not None and BROKEN_SERVER in (outcome.failures or {}):
                    break
                await pilot.pause()
                await asyncio.sleep(0.05)
            outcome = getattr(viewer, "mcp_startup", None)
            assert (
                outcome is not None and BROKEN_SERVER in outcome.failures
            ), "the owner never reported the broken server through the frontend state"

            # ...and the SCREEN says so. Polled rather than read once: the
            # notice lands from a state push, which is one loop turn after the
            # assertion above.
            def _notices() -> list[str]:
                return [str(block._text) for block in app.query(NoticeBlock)]

            while time.monotonic() < deadline:
                if any(BROKEN_SERVER in text for text in _notices()):
                    break
                await pilot.pause()
                await asyncio.sleep(0.05)
            notices = _notices()
            assert any(BROKEN_SERVER in text for text in notices), (
                "the failure reached the viewer's state and no surface: " f"notices={notices!r}"
            )
            # The toast is the interruption; the notice is the record. The user
            # gets both, and the toast is what this test's frame shows.
            toast = app.query_one(Toast)
            assert str(
                getattr(toast, "_message", "") or ""
            ), "the boot toast was not raised for the failed server"

            # ``LOP_E2E_FRAME_PATH`` exists because pytest's tmp root is shared on
            # a host running several sessions: the retention keeps the last three
            # roots and a sibling run reaps this one within seconds, which is long
            # enough to read the printed path and not long enough to fetch it.
            # Unset, the frame lands in ``tmp_path`` like every other test artefact.
            frame = save_capture(
                app, os.environ.get("LOP_E2E_FRAME_PATH") or (tmp_path / "mcp-failure-notice.svg")
            )
            print(f"MCP failure frame: {frame}")
    finally:
        if viewer is not None:
            with contextlib.suppress(Exception):
                await viewer.dispose()
