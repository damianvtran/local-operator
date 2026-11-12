#!/usr/bin/env python3
"""Drive the two instruments that are supposed to be able to FAIL, and show them fail.

Review round 2 left two assertions that could not fail, and both were replaced:

* the refusal matrix's completeness claim — its five-row list WAS the mechanism, so
  a route missing from the list was silently ungated (``/mcp``, ``/credentials``,
  ``/fork``, ``/asides``, ``/adopt``: MAJOR-1). It is now a walk whose red state is
  demonstrated here;
* the "no runtime was started" cell — it argued from ``run/mobile`` being empty, and
  an isolated config root produces that whether the request was refused or never
  tried at all (the runtime cannot even be constructed without a configured hosting
  platform), so the cell proved nothing (MINOR-1). It is now a spy on the spawn seam,
  and here is the spy firing on exactly the request the reviewer measured.

Two demonstrations, no daemon needed (a script cannot be given a latched daemon's
internals, which is the same reason ``inject.py`` exists):

1. the WALK, over a scratch copy of a real router module with one ungated route
   appended — the shape round 2 measured on a live daemon;
2. the SPY, over the ASSEMBLED app with the door's refusal removed — a latched
   ``POST /v1/desktop/sessions/{id}/mcp`` reaching ``_ensure_bound``.

Run by ``run.sh`` with the worktree's interpreter, because it deliberately imports
the SAME walker and the same app the suite drives rather than a copy of either.
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import sys
import tempfile
from types import SimpleNamespace

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from tests.unit.server.test_serve_retire import (  # noqa: E402
    _DESKTOP_ROUTER_MODULES,
    _door_bypasses,
    _Module,
    _router_modules,
)

#: The route review round 2 measured reaching the spawn seam on a latched daemon.
MEASURED = "/v1/desktop/sessions/{session_id}/mcp"


def walk(scratch: pathlib.Path) -> list[str]:
    """Append one ungated route to a scratch copy of a real router module."""
    original = next(m for m in _router_modules() if m.name.endswith("desktop_lifecycle"))
    scratch.write_text(
        original.path.read_text(encoding="utf-8")
        + "\n\n@router.post('/v1/desktop/sessions/{session_id}/frobnicate')\n"
        "async def frobnicate(session_id: str, request: Request):\n"
        "    bridge = host(request).bridges[session_id]\n"
        "    await bridge.remote.bind_runtime()\n"
        "    return reply({'data': {}})\n",
        encoding="utf-8",
    )
    return _door_bypasses([_Module("scratch.desktop_lifecycle", scratch)])


async def spy() -> str:
    """A latched ``/mcp`` on the assembled app, with the DOOR and nothing else removed."""
    os.environ["LOCAL_OPERATOR_DESKTOP_TOKEN"] = "ungated-token"
    from httpx import ASGITransport, AsyncClient

    from local_operator.credentials import CredentialManager
    from local_operator.server.app import app
    from local_operator.server.utils.desktop_sessions import DesktopSessions
    from local_operator.session.attached import AttachedSession

    root = pathlib.Path(tempfile.mkdtemp(prefix="pr1102-ungated-"))
    app.state.config_manager = SimpleNamespace(config_dir=root)
    app.state.credential_manager = CredentialManager(root)
    app.state.serve_retiring = False
    pool = DesktopSessions(root, retiring=lambda: bool(getattr(app.state, "serve_retiring", False)))
    app.state.desktop_sessions = pool
    session_id = await pool.create(str(root))
    app.state.serve_retiring = True  # the daemon LATCHES

    reached: list[str] = []

    async def _seam(*_args: object, **_kwargs: object) -> None:
        reached.append("spawn")
        raise AssertionError("the runtime spawn seam was entered")

    AttachedSession._ensure_bound = _seam  # type: ignore[method-assign]
    # ONLY the door: the probe, the latch, the route table and the seam are the real
    # ones, and the request below is the one round 2 sent.
    DesktopSessions.assert_admitting = lambda self: None  # type: ignore[method-assign,assignment]

    answer = "the request never returned"
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://localhost",
            headers={"Authorization": "Bearer ungated-token"},
        ) as client:
            await client.post(MEASURED.format(session_id=session_id), json={"action": "list"})
    except AssertionError as error:
        answer = f"AssertionError: {error}"
    finally:
        await pool.close()

    print(f"  POST {MEASURED.format(session_id=session_id)} -> {answer}")
    print(f"  spawn-seam entrances recorded by the spy: {len(reached)}")
    if not reached:
        raise SystemExit("the spy did not fire: the instrument would be vacuous")
    return answer


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="pr1102-walk-") as directory:
        scratch = pathlib.Path(directory) / "scratch_desktop_lifecycle.py"
        print("1. the completeness walk, over a scratch copy with ONE ungated route:")
        offenders = walk(scratch)
        if not offenders:
            raise SystemExit("the walk stayed green on an ungated route: it proves nothing")
        for line in offenders:
            print(f"  {line}")
        print(
            f"  -> the walk named {len(offenders)} bypass(es) in "
            f"{pathlib.Path(_DESKTOP_ROUTER_MODULES[1]).name}'s scratch copy"
        )

    print("2. the spawn-seam spy, with the door's refusal removed on a latched daemon:")
    asyncio.run(spy())
    print("\nboth instruments can fail, so a green run of either is a measurement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
