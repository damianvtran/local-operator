"""``/mcp list`` and ``/mcp logout`` on the ASSEMBLED viewer.

Both defects were reported by the operator against an installed build, and both
live on the seam between the viewer and its owner runtime — which is why the
unit suite and the sibling e2e stages were green while the TUI was wrong:

* ``/mcp list`` answered "no MCP servers configured." on a session whose own
  transcript was listing two configured servers failing to start by name. The
  routed handler tested ``manager.servers``, an attribute ``McpManager`` has
  never had, so its emptiness branch was taken on every session whose slash
  command routes to the owner — every fresh viewer, and the phone projection,
  which shares that handler.
* Typing ``/mcp logout `` killed the app as the argument list refreshed: the
  URL mapping (then ``_mcp_server_url``, now ``_mcp_configured_urls``) asked
  the follower's read-only ``SnapshotMcpManager`` for ``get_server_config``,
  which that facade does not have. The sibling grant tests drive
  ``_run_slash_command`` — which ROUTES the verb to the owner — and never the
  picker, which is built locally either way.

This stage drives app + production ``AttachedSession`` viewer + a real runtime
child + the config layer + the credential store, so it fails if either defect
returns in any of them. The typed form is used deliberately: the picker fill
that crashed is reached by TYPING the space after the verb, not by dispatching
an assembled command string.

The rendered frames for both surfaces live on the PR (evidence stays out of the
repository, per the visual-validation rules); what this file pins is the
behaviour, minus any dependence on a human having looked at a screenshot.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.e2e

#: Two servers, one of each transport shape, and neither able to start: the
#: stdio one fails at the 250 ms gate (a missing executable) and the HTTP one
#: cannot resolve. Both failures are FAST, so the boot these tests wait out
#: does not depend on a network timeout.
SERVERS = {
    "alpha-stdio": {"command": "definitely-not-installed-xyz"},
    "beta-oauth": {"url": "https://mcp.invalid.example/mcp", "auth": {"type": "oauth"}},
}

#: Upper bound on an awaited event, never a budget to sleep through. Generous
#: because the first bind of a cold viewer pays the child's whole construction.
GUARD_S = 60.0


def _configure_provider(config_dir: Path) -> None:
    """A configured machine, so the mount engage spawns (see the sibling e2e)."""
    from local_operator.config import ConfigManager

    ConfigManager(config_dir=config_dir).update_config({"hosting": "test", "model_name": "mock"})


def _write_mcp(config_dir: Path) -> None:
    (config_dir / "mcp.json").write_text(json.dumps({"mcpServers": SERVERS}), encoding="utf-8")


async def _never_take_over() -> Any:
    raise AssertionError("a viewer must never take over a session")


async def _bind_viewer(pilot: Any, app: Any) -> Any:
    """Wait for the app to adopt its session, and return that session."""
    deadline = time.monotonic() + GUARD_S
    viewer: Any = None
    while time.monotonic() < deadline:
        await pilot.pause()
        viewer = app._session
        if viewer is not None and not getattr(viewer, "is_cold", True):
            return viewer
        await asyncio.sleep(0.05)
    raise AssertionError(f"the viewer never bound within {GUARD_S}s")


async def _await_mcp_wiring(pilot: Any, viewer: Any) -> None:
    """Wait until the OWNER has published its MCP outcome.

    The viewer binds when the record is published, which is BEFORE the owner's
    MCP wiring runs. Measured on an idle box: at bind ``viewer.mcp_startup`` is
    ``None`` and the facade holds no names, and the wiring lands ~1.3 s later.
    A listing asked inside that window is a DIFFERENT state, not this defect:
    the session has no manager at all yet, so the refusal is the same one
    ``_mcp_status`` reports as ``discovery_failed``, and the transcript is not
    yet contradicting it (the boot notices naming the servers arrive with the
    same wiring). Asserting without this wait races the two, which is how this
    test first failed in the full e2e stage while passing on an idle box.
    """
    deadline = time.monotonic() + GUARD_S
    while time.monotonic() < deadline:
        outcome = getattr(viewer, "mcp_startup", None)
        if outcome is not None and set(SERVERS) <= set(outcome.configured):
            return
        await pilot.pause()
        await asyncio.sleep(0.05)
    raise AssertionError("the owner never published its MCP startup outcome")


async def _type(pilot: Any, app: Any, command: str) -> None:
    """Type ``/command`` one REAL keystroke at a time, then press Enter.

    Not a text assignment: the ``RefreshArgumentChoices`` message that refills
    ``/mcp``'s second argument slot only fires for a user at the keyboard, and
    it is the handler for that message that crashed.
    """
    from local_operator.tui.widgets.editor import Editor

    app.query_one(Editor).focus()
    await pilot.press("slash")
    for ch in command:
        await pilot.press("space" if ch == " " else ch)
    await pilot.press("enter")
    for _ in range(20):
        await pilot.pause()


def _transcript_text(app: Any) -> str:
    from tests.e2e.harness import transcript_text

    return transcript_text(app)


def _listing_text(app: Any) -> str:
    """Only the MCP LISTING block's text.

    Scoped deliberately: the boot notices name both servers too, so asserting
    their names against the whole transcript would pass on a transcript that
    never rendered a listing at all — the exact shape of the bug.
    """
    from local_operator.tui.widgets.transcript import RichBlock
    from tests.unit.tui.test_app_pilot import _renderable_plain

    for block in app.query(RichBlock):
        text = _renderable_plain(getattr(block, "renderable", ""))
        if "MCP servers" in text:
            return text
    return ""


async def _boot(config_dir: Path, workspace: Path) -> Any:
    from local_operator.session.attached import AttachedSession
    from local_operator.tui.app import OperatorApp

    async def factory() -> Any:
        return await AttachedSession.cold(
            "mcpslash0001",
            config_dir=config_dir,
            cwd=str(workspace),
            takeover_factory=_never_take_over,
        )

    return OperatorApp(factory)


@pytest.mark.asyncio
async def test_mcp_list_renders_the_servers_the_session_has_configured(
    headless_tui_env: Path, workspace: Path
) -> None:
    """The operator's first complaint: a listing that denied its own servers."""
    _configure_provider(headless_tui_env)
    _write_mcp(headless_tui_env)

    app = await _boot(headless_tui_env, workspace)
    try:
        async with app.run_test(size=(110, 34)) as pilot:
            viewer = await _bind_viewer(pilot, app)
            # The listing is answered by the OWNER, so the owner has to have
            # wired its manager before this asserts anything about its content.
            await _await_mcp_wiring(pilot, viewer)
            await _type(pilot, app, "mcp list")

            text = _transcript_text(app)
            assert "no MCP servers configured." not in text, (
                "the routed listing answered for an empty roster while the same "
                f"transcript knew the servers; transcript:\n{text}"
            )
            listing = _listing_text(app)
            assert "MCP servers" in listing, f"no MCP listing block rendered; got {listing!r}"
            for name in SERVERS:
                assert name in listing, f"{name} missing from the rendered listing: {listing!r}"
    finally:
        with contextlib.suppress(Exception):
            await app._dispose_session()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_typing_mcp_logout_does_not_crash_and_reports_an_outcome(
    headless_tui_env: Path, workspace: Path
) -> None:
    """The operator's second complaint: typing the verb killed the app.

    No credential is seeded, so the outcome this run can honestly reach is the
    "nothing to log out of" refusal — which is the point: the command REACHED
    its handler and answered, instead of the app dying while the argument list
    refreshed. On the base, the ``AttributeError`` surfaced out of
    ``app.run_test`` and failed this test before any assertion ran.
    """
    _configure_provider(headless_tui_env)
    _write_mcp(headless_tui_env)

    app = await _boot(headless_tui_env, workspace)
    try:
        async with app.run_test(size=(110, 34)) as pilot:
            await _bind_viewer(pilot, app)
            # The space after `logout` is the keystroke that refreshes the
            # argument list — and the one that used to take the app down.
            await _type(pilot, app, "mcp logout beta-oauth")

            text = _transcript_text(app)
            assert (
                "no stored credential for MCP server 'beta-oauth'" in text
            ), f"the typed logout produced no receipt; transcript:\n{text}"
    finally:
        with contextlib.suppress(Exception):
            await app._dispose_session()  # type: ignore[attr-defined]
