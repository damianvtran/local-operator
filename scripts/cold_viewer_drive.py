"""The operator's exact command on a GENUINELY cold viewer, pasted at t=0.

The attached-viewer driver beside this (``attached_viewer_drive.py``) attaches
to a running runtime with ``RemoteSession.connect`` — the live-local case. Review round 1 (R5, Q5)
pointed out that the PR's claim was about the COLD case: a fresh ``lop`` is a
``RemoteSession.cold(<minted id>)`` bound to nothing, and the four commands
this PR fixes were still lost there whenever Enter beat the 1–3 s warm engage.

This drives that case with nothing stubbed:

- the viewer is built byte-for-byte as ``cli.py`` builds it (id minted first,
  then ``RemoteSession.cold``), in an isolated ``HOME`` + config dir;
- the runtime is a REAL spawned ``python -m local_operator.session.runtime.process``
  child, talking to the ``test`` mock provider (the same one
  ``tests/e2e/test_cold_wake_e2e.py`` boots against);
- every command is submitted as one ``events.Paste`` followed immediately by
  Enter — the t=0 race QA measured — with NO keystrokes in between, so no
  speculative warm engage has a head start.

Reports per cell: whether the viewer was cold at submit, whether it was bound
afterwards, what the transcript says, and the durable side effect (team on the
runtime's transcript, credential name in the store, model in config.yml).
The credential value is a placeholder and is reported by LENGTH only.

Usage:
    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \\
        scripts/cold_viewer_drive.py [out.svg]
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

# The sandbox HOME + config dir the other shot scripts use, set BEFORE any
# local_operator import so nothing consults the real ~/.local-operator.
isolate_capture()
CONFIG = Path(os.environ["LOCAL_OPERATOR_CONFIG_DIR"])
CONFIG.mkdir(parents=True, exist_ok=True)
ISO = Path(os.environ["HOME"])
os.environ["LOCAL_OPERATOR_NO_NOTIFICATIONS"] = "1"
os.environ["LOCAL_OPERATOR_NO_TERMINAL_TITLE"] = "1"

from textual import events  # noqa: E402

from local_operator.session.remote import RemoteSession  # noqa: E402
from local_operator.session.runtime import registry  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.editor import Editor  # noqa: E402


async def _never_take_over():
    raise AssertionError("a viewer must never take over")


def _transcript(app: OperatorApp) -> str:
    return " | ".join(
        str(getattr(block, "_text", "") or "") for block in app._transcript_view().children
    )


async def _settle(pilot, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        await pilot.pause()
        await asyncio.sleep(0.05)


async def _until(pilot, predicate, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await pilot.pause()
        if predicate():
            return True
        await asyncio.sleep(0.05)
    return False


async def _paste_enter(app: OperatorApp, pilot, line: str) -> None:
    editor = app.query_one(Editor)
    editor.focus()
    await pilot.pause()
    app.post_message(events.Paste(line))
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()


def _viewer_factory(config_dir: Path):
    """Byte-for-byte ``cli.py``'s cold branch: the id is minted FIRST."""

    async def factory():
        session_id = uuid.uuid4().hex[:12]
        return await RemoteSession.cold(
            session_id, config_dir=config_dir, cwd=str(ISO), takeover_factory=_never_take_over
        )

    return factory


async def _stop_runtime(app: OperatorApp, pilot) -> None:
    session = app._session
    session_id = getattr(session, "session_id", "")
    app._run_slash_command("/stop")
    await _until(
        pilot,
        lambda: not any(r.session_id == session_id for r, _s in registry.scan(CONFIG)),
        timeout=30,
    )


async def main() -> None:
    from local_operator.config import ConfigManager
    from local_operator.teams import TeamEditFields, TeamMember, TeamRegistry

    manager = ConfigManager(CONFIG)
    manager.set_config_value("hosting", "test")
    manager.set_config_value("model_name", "mock")
    manager.set_config_value("tool_approval_mode", "auto")
    TeamRegistry(CONFIG).create_team(
        TeamEditFields(
            name="lopdev",
            description="the dev team",
            manager="manager",
            members=[TeamMember(role="coder"), TeamMember(role="reviewer")],
            instructions="collaborate",
            project="local-operator",
        )
    )
    OperatorApp._check_for_update = lambda self: None  # type: ignore[method-assign]
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else None

    def _app() -> OperatorApp:
        # The real provider controller, as `cli.py` wires it: `/model default`
        # resolves its selector through it before writing config.
        from local_operator.credentials import CredentialManager
        from local_operator.providers.auth_store import AuthStore
        from local_operator.providers.controller import ProviderController

        credentials = CredentialManager(CONFIG)
        controller = ProviderController(AuthStore(credential_manager=credentials), credentials)
        return OperatorApp(_viewer_factory(CONFIG), provider_controller=controller)

    # ---- cell 1: /team lopdev <request>, pasted at t=0 on a cold viewer ----
    app = _app()
    async with app.run_test(size=(110, 30)) as pilot:
        await _until(pilot, lambda: app._session is not None)
        session = app._session
        assert isinstance(session, RemoteSession)
        sid = session.session_id
        print(f"viewer: RemoteSession.cold  id={sid!r}  is_cold={session.is_cold}")
        print(f"        capabilities advertised: {len(session.frontend_state.slash_capabilities)}")
        print(f"        runtime records: {len(list(registry.scan(CONFIG)))}")
        print(f"        _session_runs_elsewhere(): {app._session_runs_elsewhere()}")

        t0 = time.monotonic()
        cold_at_submit = session.is_cold
        await _paste_enter(app, pilot, "/team lopdev ship the team fix")
        transcript_path = CONFIG / "sessions" / sid / "transcript.jsonl"
        done = await _until(
            pilot,
            lambda: transcript_path.exists()
            and "Hello from the mock provider" in transcript_path.read_text(),
            timeout=60,
        )
        print("\n--- /team lopdev ship the team fix (paste + Enter at t=0) ---")
        print(f"  cold at submit:           {cold_at_submit}")
        print(f"  bound afterwards:         {not session.is_cold}")
        print(f"  turn ran on a real child: {done} ({time.monotonic() - t0:.1f}s)")
        body = transcript_path.read_text() if transcript_path.exists() else ""
        print(f"  request in transcript:    {'ship the team fix' in body}")
        print(f"  team stamped on runtime:  {session.active_team_name!r}")
        rendered = _transcript(app)
        print(f"  refused with old copy:    {'but not run one' in rendered}")
        print(f"  transcript: {rendered[:220]}")
        if out is not None:
            save_capture(app, out)
        await _stop_runtime(app, pilot)

    # ---- cell 2: /credential KEY, pasted at t=0 on a cold viewer ----
    app = _app()
    async with app.run_test(size=(110, 30)) as pilot:
        await _until(pilot, lambda: app._session is not None)
        session = app._session
        assert isinstance(session, RemoteSession)
        cold_at_submit = session.is_cold
        await _paste_enter(app, pilot, "/credential DEMO_TOKEN")
        prompt_before_bind = app._key_prompt is not None
        opened = await _until(pilot, lambda: app._key_prompt is not None, timeout=60)
        bound_when_opened = not session.is_cold
        placeholder = "x" * 20
        app.post_message(events.Paste(placeholder))
        await pilot.pause()
        await pilot.press("enter")
        stored = await _until(pilot, lambda: "Stored DEMO_TOKEN" in _transcript(app), timeout=30)
        names = await session.credential_op("names")
        print("\n--- /credential DEMO_TOKEN (paste + Enter at t=0) ---")
        print(f"  cold at submit:               {cold_at_submit}")
        print(f"  prompt opened before binding: {prompt_before_bind}")
        print(f"  prompt opened, viewer bound:  {opened} / {bound_when_opened}")
        print(f"  stored on the runtime:        {stored}  names={names.get('names')}")
        print(f"  value length pasted:          {len(placeholder)}")
        rendered = _transcript(app)
        print(f"  value ever painted:           {placeholder in rendered}")
        print(f"  'not reachable' shown:        {'not reachable' in rendered}")
        await _stop_runtime(app, pilot)

    # ---- cell 3: /model default, pasted at t=0 on a cold viewer ----
    app = _app()
    async with app.run_test(size=(110, 30)) as pilot:
        await _until(pilot, lambda: app._session is not None)
        session = app._session
        assert isinstance(session, RemoteSession)
        cold_at_submit = session.is_cold
        manager.set_config_value("model_name", "mock")
        await _paste_enter(app, pilot, "/model default test/mock-model")
        await _settle(pilot, 1.0)
        rendered = _transcript(app)
        written = ConfigManager(CONFIG)
        print("\n--- /model default test/mock-model (paste + Enter at t=0) ---")
        print(f"  cold at submit:      {cold_at_submit}")
        print(f"  refused:             {'whose launches it should govern' in rendered}")
        print(
            f"  config now:          hosting={written.get_config_value('hosting')!r} "
            f"model_name={written.get_config_value('model_name')!r}"
        )
        print(f"  transcript: {rendered[:200]}")
        if not session.is_cold:
            await _stop_runtime(app, pilot)

    # ---- cell 4: /agent <name> <message>, pasted at t=0 on a cold viewer ----
    app = _app()
    async with app.run_test(size=(110, 30)) as pilot:
        await _until(pilot, lambda: app._session is not None)
        session = app._session
        assert isinstance(session, RemoteSession)
        cold_at_submit = session.is_cold
        await _paste_enter(app, pilot, "/agent reviewer look at this")
        routed = await _until(pilot, lambda: "reviewer" in _transcript(app), timeout=60)
        await _settle(pilot, 0.5)
        rendered = _transcript(app)
        print("\n--- /agent reviewer look at this (paste + Enter at t=0) ---")
        print(f"  cold at submit:        {cold_at_submit}")
        print(f"  bound afterwards:      {not session.is_cold}")
        print(f"  receipt rendered:      {routed}")
        print(f"  refused with old copy: {'but not attach one' in rendered}")
        print(f"  transcript: {rendered[:220]}")
        await _stop_runtime(app, pilot)


if __name__ == "__main__":
    asyncio.run(main())
