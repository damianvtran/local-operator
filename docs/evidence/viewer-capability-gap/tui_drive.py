"""The operator's exact command, on a cold-booted viewer, in the real TUI.

Drives ``OperatorApp`` over a production ``RemoteSession`` attached to a
production runtime, types ``/team lopdev <request>`` the way a user does, and
reports what reached the transcript and whether a turn actually started.

Usage:
    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        docs/evidence/viewer-capability-gap/tui_drive.py [out.svg]
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

CONFIG = Path(tempfile.mkdtemp(prefix="teamfix-tui-"))
os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = str(CONFIG)
os.environ["TERM"] = "xterm-256color"
os.environ.pop("NO_COLOR", None)
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from local_operator.session.runtime import registry  # noqa: E402
from local_operator.session.runtime.owned import OwnedSessionHandle  # noqa: E402
from local_operator.session.runtime.server import RuntimeServer  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from tests.e2e.harness import ScriptedStream, build_session, text_turn  # noqa: E402


async def _never_take_over():
    raise AssertionError("a viewer must never take over")


async def _wait_for_record(config_dir: Path, session_id: str, timeout: float = 10.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        for record, _state in registry.scan(config_dir):
            if getattr(record, "session_id", "") == session_id:
                return record
        await asyncio.sleep(0.05)
    raise AssertionError("no record published")


async def main() -> None:
    from local_operator.session.remote import RemoteSession
    from local_operator.teams import TeamEditFields, TeamMember, TeamRegistry

    reg = TeamRegistry(CONFIG)
    reg.create_team(
        TeamEditFields(
            name="lopdev",
            description="the dev team",
            manager="manager",
            members=[TeamMember(role="coder"), TeamMember(role="reviewer")],
            instructions="collaborate",
            project="local-operator",
        )
    )

    directory = CONFIG / "sessions" / "tuidrive001"
    directory.mkdir(parents=True)
    # Several scripted turns: the attach itself may trigger a state refresh turn,
    # and a script exhausted mid-run raises IndexError by design (harness.py).
    stream = ScriptedStream([text_turn("On it — coder is picking this up.")] * 4)
    session = build_session(directory, stream)
    session.team_registry = reg
    handle = OwnedSessionHandle(session, asyncio.get_running_loop(), cwd=str(directory))
    server = RuntimeServer(handle, kind="daemon")
    await server.start_in_process()

    viewer = None
    try:
        record = await _wait_for_record(CONFIG, session.session_id)
        viewer = await RemoteSession.connect(
            record,
            session.session_id,
            config_dir=CONFIG,
            takeover_factory=_never_take_over,
        )

        async def _factory():
            return viewer

        app = OperatorApp(_factory)
        async with app.run_test(size=(100, 30)) as pilot:
            for _ in range(60):
                await pilot.pause()
                if app._session is not None:
                    break
            print(f"session in app: {type(app._session).__name__}")

            # The operator's exact command, through the real dispatcher.
            app._run_slash_command("/team lopdev ship the team fix")
            for _ in range(80):
                await pilot.pause()
                await asyncio.sleep(0.02)

            rendered = " | ".join(
                str(getattr(block, "_text", "") or "") for block in app._transcript_view().children
            )
            print("\n--- transcript after /team lopdev ship the team fix ---")
            print(rendered[:600] or "(EMPTY - the silent failure)")

            transcript_path = directory / "transcript.jsonl"
            body = transcript_path.read_text() if transcript_path.exists() else ""
            print("\n--- runtime transcript.jsonl ---")
            print(f"  user turn reached the runtime: {'ship the team fix' in body}")
            print(f"  model answered:                {'coder is picking this up' in body}")
            print(f"  team attached on the session:  {session.active_team_name!r}")

            # /credential on the same viewer: the store lives on the owner.
            app._cmd_credential("", app._notice)
            for _ in range(40):
                await pilot.pause()
                await asyncio.sleep(0.02)
            cred = " | ".join(
                str(getattr(b, "_text", "") or "") for b in app._transcript_view().children
            )
            print("\n--- /credential (listing) on the viewer ---")
            print("  " + (cred.split("|")[-1].strip() or "(EMPTY)"))
            print(f"  says 'still starting': {'still starting' in cred}")

            # The round trip that matters: a secret pasted at the VIEWER must
            # land in the OWNER's store, because that is where the agent's bash
            # commands read it from the environment.
            answer = await viewer.credential_op("store", "DEMO_TOKEN", "s3cret-value")
            print("\n--- store a credential from the viewer ---")
            print(f"  viewer round trip: {answer}")
            owner_store = session.variables
            print(f"  owner holds it:    {owner_store.credential_names()}")
            print(f"  value never returned: {'s3cret-value' not in str(answer)}")
            listed = await viewer.credential_op("list")
            print(f"  viewer sees it:    {[r['key'] for r in listed.get('credentials') or []]}")

            # THE PROOF THAT MATTERS: the secret typed at the viewer must be
            # readable by the `bash` tool, which runs in the RUNTIME's process
            # and reads the RUNTIME's store via credential_env(). A viewer-local
            # store would pass every check above and still fail here.
            #
            # The value is never printed: we assert on presence and length only.
            from local_operator.tools.builtin import execute_bash

            class _Ctx:
                variables = session.variables
                cwd = str(directory)

            result = await execute_bash(
                "probe-1",
                {"command": 'test -n "$DEMO_TOKEN" && echo LEN=${#DEMO_TOKEN}'},
                context=_Ctx(),
            )
            out = str(getattr(result, "content", None) or getattr(result, "output", result))
            print("\n--- bash tool in the RUNTIME reads the viewer's credential ---")
            print(f"  child env saw the key: {'LEN=' in out}")
            tail = out.strip().splitlines()[-1] if "LEN=" in out else out[:80]
            print(f"  reported length:       {tail!r}")
            print(f"  value never printed:   {'s3cret-value' not in out}")

            # /model default and the picker's `d` key must AGREE: both write
            # this machine's config, so on a local viewer both must succeed.
            from local_operator.config import ConfigManager
            from local_operator.paths import config_dir as _cfg

            print("\n--- default-model persistence on a local viewer ---")
            print(f"  _session_runs_elsewhere(): {app._session_runs_elsewhere()}")
            app._run_slash_command("/model default openai/gpt-4o")
            for _ in range(40):
                await pilot.pause()
                await asyncio.sleep(0.02)
            saved = ConfigManager(_cfg()).get_config_value("model_name")
            rendered_m = " | ".join(
                str(getattr(b, "_text", "") or "") for b in app._transcript_view().children
            )
            print(f"  /model default refused:    {'whose launches it should govern' in rendered_m}")
            print(f"  last notice: {rendered_m.split('|')[-1].strip()[:110]!r}")
            print(f"  config model_name now:     {saved!r}")

            # /team chart must stay LOCAL (it opens a view this terminal paints).
            app._run_slash_command("/team chart lopdev")
            for _ in range(40):
                await pilot.pause()
                await asyncio.sleep(0.02)
            chart = " | ".join(
                str(getattr(b, "_text", "") or "") for b in app._transcript_view().children
            )
            print("\n--- /team chart lopdev ---")
            print(f"  no-such-team error: {'no team named' in chart}")
            print(f"  org chart opened:   {app.screen.__class__.__name__}")

            if len(sys.argv) > 1:
                app.save_screenshot(sys.argv[1])
                print(f"\nframe: {sys.argv[1]}")
    finally:
        if viewer is not None:
            await viewer.dispose()
        server.close()
        await session.dispose()


asyncio.run(main())
