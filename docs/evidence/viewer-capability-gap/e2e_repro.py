"""Reproduction: /team and /credential on a real viewer over a real socket.

Drives the production RemoteSession against the production OwnedSessionHandle
and RuntimeServer, then runs the operator's exact commands through the real
OperatorApp slash dispatcher. No stubs on the objects under test.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

CONFIG = Path(tempfile.mkdtemp(prefix="teamfix-repro-"))
os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = str(CONFIG)
os.environ["TERM"] = "xterm-256color"
os.environ.pop("NO_COLOR", None)
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from local_operator.session.runtime import registry  # noqa: E402
from local_operator.session.runtime.owned import OwnedSessionHandle  # noqa: E402
from local_operator.session.runtime.server import RuntimeServer  # noqa: E402
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

    # A real team on disk, so the listing has something to find.
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
    print(f"teams on disk: {[t.name for t in reg.list_teams()]}")

    directory = CONFIG / "sessions" / "reprosess01"
    directory.mkdir(parents=True)
    stream = ScriptedStream([text_turn("ack from the runtime.")])
    session = build_session(directory, stream)
    # Production wires the owner's Session with a real team registry; the e2e
    # harness does not, and without this the owner refuses at the REGISTRY
    # guard and never reaches the attach guard under test.
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
        print(f"viewer attached: {type(viewer).__name__}")

        print("\n--- capability probe on the live viewer instance ---")
        for attr in ("team_registry", "attach_team", "active_team", "variables"):
            print(f"  {attr:15} present={hasattr(viewer, attr)}")

        # The routed path: what a BOUND viewer gets back for the operator's
        # exact command. This is the shape that reaches _render_authoritative_slash.
        print("\n--- routed /team lopdev <request> (owner-side result) ---")
        outcome = await viewer.route_shared_slash("team", "lopdev build the thing", [])
        print(f"  outcome = {outcome}")
        kind = outcome.get("kind") if isinstance(outcome, dict) else None
        data = outcome.get("data") if isinstance(outcome, dict) else {}
        print(f"  kind={kind!r} data.type={ (data or {}).get('type')!r}")
        if kind == "noop":
            print("  >>> RENDERER DROPS THIS: _render_authoritative_slash returns on noop")
            print("  >>> user sees NOTHING. This is the silent failure.")

        print("\n--- routed bare /team (listing works, for contrast) ---")
        listing = await viewer.route_shared_slash("team", "", [])
        lk = listing.get("kind") if isinstance(listing, dict) else None
        print(f"  kind={lk!r} -> listing renders fine")

        print("\n--- /credential test on the viewer ---")
        store = getattr(viewer, "variables", None)
        print(f"  viewer.variables = {store!r}")
        if store is None:
            print('  >>> _cmd_credential emits "session is still starting…"')
            print("  >>> but the session is FULLY ATTACHED. The wait never ends.")
    finally:
        if viewer is not None:
            await viewer.dispose()
        server.close()
        await session.dispose()
        print(f"\nscratch config dir: {CONFIG}")


asyncio.run(main())
