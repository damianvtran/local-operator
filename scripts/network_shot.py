"""Capture the ``/network`` panel in BOTH of its phases, from one boot.

Usage: python scripts/network_shot.py OUTDIR [100x30]

Writes ``network-loading.svg`` (+ ``.geometry.json``) and ``network-loaded.svg``
(+ ``.geometry.json``) into ``OUTDIR``.

WHY BOTH, AND WHY ONE PROCESS. The panel is two-phase by design: its first frame
is this device's own records (a disk read, useful with the relay stopped) and the
second is the relay's answer, which dials every member. Those are two DIFFERENT
frames of one surface, so the visual evidence has to show both — and the pair is
only comparable if nothing else differs between them, which is why they come from
one boot with one fixture rather than from two runs that could disagree about a
fixture detail nobody looked at.

The gate is what makes the first frame honest rather than a race the script
happens to win: ``run_network`` is stubbed to park until the capture has been
taken, so the LOADING frame is the real first paint (its ``checking…`` rows) and
never a fast worker's result. Nothing sleeps, and the parked call is released in
the same process before the script exits — a stub that never returned would leave
a thread for the interpreter to wait on, which is the shape that turns a capture
script into a hang.

EVERY NETWORK VALUE HERE IS A FIXTURE. The frames go on a PR, and redaction rules
apply to an SVG exactly as they do to a clipboard: a capture of a live machine
would publish real device ids, real peer names and real member counts. The
isolation import must stay FIRST for the same reason it does in ``info_shot.py``
(it re-homes ``HOME`` and ``LOCAL_OPERATOR_CONFIG_DIR`` before anything under
``local_operator`` can resolve the operator's real config), and every ``CMUX_*``
variable is cleared before any application import: a headless pilot must not
rename the operator's real workspaces through an inherited id.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

for _key in tuple(os.environ):
    if _key.startswith("CMUX_"):
        os.environ.pop(_key)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio  # noqa: E402
import json  # noqa: E402
import threading  # noqa: E402

import scripts.probe_isolation  # noqa: E402, F401
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.network_cli import NetworkRun  # noqa: E402
from local_operator.tui.widgets.network_panel import (  # noqa: E402
    NetworkEntry,
    NetworkLocal,
    NetworkScreen,
    PeerEntry,
)
from scripts.visual_capture import (  # noqa: E402
    refuse_flag_shaped_argument,
    save_capture,
)
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

#: Which device the fixture is. A REAL-SHAPED value: the harness derives ids as
#: ``d_`` + 32 hex, so a shorter demo string would manufacture a column the panel
#: never has to lay out.
DEVICE_ID = "d_9f2c1a4b7e30a5d613f4c2b8e0a91d7f"
DEVICE_NAME = "damian-mbp"


def _local() -> NetworkLocal:
    """The disk-only half — what the first frame paints."""
    return NetworkLocal(
        device_id=DEVICE_ID,
        device_name=DEVICE_NAME,
        identity_present=True,
        relay_state="",
        networks=[
            NetworkEntry(
                network_id="n_3985570272c803eeb85a3e23",
                name="devmesh",
                epoch=7,
                role="admin",
                members=4,
                trust="active",
            ),
            NetworkEntry(
                network_id="n_4a1c9e02dd1b7f5c0a83b6e4912d776f",
                name="homelab",
                epoch=2,
                role="drive",
                members=2,
                trust="active",
            ),
        ],
        peers=[
            PeerEntry(
                device_id="d_1a2b3c4d5e6f708192a3b4c5d6e7f809",
                name="radiant-m4",
                role="admin",
                network_id="n_3985570272c803eeb85a3e23",
            ),
            PeerEntry(
                device_id="d_0f1e2d3c4b5a69788796a5b4c3d2e1f0",
                name="",
                role="drive",
                network_id="n_4a1c9e02dd1b7f5c0a83b6e4912d776f",
            ),
            PeerEntry(
                device_id="d_77aa88bb99cc00dd11ee22ff33445566",
                name="pixel-8",
                role="read",
                network_id="n_4a1c9e02dd1b7f5c0a83b6e4912d776f",
            ),
        ],
    )


def _ls_payload() -> str:
    """``lop network ls --json`` as the relay answers it, with its provenance.

    The rows are the VERIFIED kind (``stale`` empty, a membership sentence naming
    the refresh), because the loaded frame's whole point is the second phase: the
    device's own record and the relay's answer must be tellable apart in the
    pixels, and the copy that does that is the CLI's.
    """
    return json.dumps(
        {
            "ok": True,
            "networks": [
                {
                    "name": "devmesh",
                    "network_id": "n_3985570272c803eeb85a3e23",
                    "epoch": 7,
                    "role": "admin",
                    "members": 4,
                    "trust": "active",
                    "stale": "",
                },
                {
                    "name": "homelab",
                    "network_id": "n_4a1c9e02dd1b7f5c0a83b6e4912d776f",
                    "epoch": 2,
                    "role": "drive",
                    "members": 2,
                    "trust": "active",
                    "stale": "",
                },
            ],
        },
        indent=2,
        sort_keys=True,
    )


def _peers_payload() -> str:
    return json.dumps(
        {
            "ok": True,
            "peers": [
                {
                    "device_id": "d_1a2b3c4d5e6f708192a3b4c5d6e7f809",
                    "name": "radiant-m4",
                    "reachable": True,
                    "reason": "",
                },
                {
                    "device_id": "d_0f1e2d3c4b5a69788796a5b4c3d2e1f0",
                    "name": "",
                    "reachable": True,
                    "reason": "",
                },
                {
                    "device_id": "d_77aa88bb99cc00dd11ee22ff33445566",
                    "name": "pixel-8",
                    "reachable": False,
                    "reason": "connect_failed:ConnectionRefusedError",
                },
            ],
        },
        indent=2,
        sort_keys=True,
    )


def _status_payload() -> str:
    return json.dumps(
        {
            "ok": True,
            "installed": True,
            "identity_present": True,
            "relay_running": True,
            "relay_answering": True,
            "relay_state": "live",
            "relay": {"pid": 4711},
            "log": "~/Library/Logs/local-operator/network.log",
        },
        indent=2,
        sort_keys=True,
    )


async def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: network_shot.py OUTDIR [COLSxROWS]")
    # Before it is used as a path: a mistyped flag here mkdirs ``--out`` (see the
    # helper's docstring), and it happened in a shared checkout during the round
    # that filed this.
    refuse_flag_shaped_argument(sys.argv[1], what="OUTDIR")
    out = Path(sys.argv[1])
    size = (100, 30)
    for arg in sys.argv[2:]:
        refuse_flag_shaped_argument(arg, what="SIZE")
        if "x" in arg:
            cols, rows = arg.split("x")
            size = (int(cols), int(rows))
    out.mkdir(parents=True, exist_ok=True)

    # The gate the stub parks on. Set before the second capture, so the
    # interpreter has nothing left to wait for when the script returns.
    released = threading.Event()

    def stubbed(args: list[str], **kwargs: object) -> NetworkRun:
        released.wait(timeout=30)
        verb = args[0] if args else "ls"
        body = {"ls": _ls_payload, "peers": _peers_payload, "status": _status_payload}.get(verb)
        return NetworkRun(tuple(args), 0, stdout=body() if body else "")

    import local_operator.tui.widgets.network_panel as panel

    panel.run_network = stubbed  # type: ignore[assignment]

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        screen = NetworkScreen(_local())
        app.push_screen(screen)
        await pilot.pause()
        await pilot.pause()
        # LOADING: the worker is parked, so this is the real first paint.
        save_capture(app, out / "network-loading.svg")
        # Released, then settled by EVENT rather than by a clock: the panel
        # repaints when the worker's answer lands, and waiting for the worker
        # ends is waiting for the thing that changes the frame.
        released.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
        await pilot.pause()
        save_capture(app, out / "network-loaded.svg")
    print(f"wrote {out}/network-loading.svg and {out}/network-loaded.svg")


if __name__ == "__main__":
    asyncio.run(main())
