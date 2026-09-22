"""Capture the ``/new`` device picker, in both states a user meets.

Usage: python scripts/new_remote_shot.py OUT.svg {with-peers|no-peers} [100x30]

WHY THESE TWO FRAMES (design round 2, D15/D16). The picker this branch added —
one row per paired peer, each row carrying the whole ``remote <peer>`` argument —
had no artifact in the round that added it, and design round 1's own rule is that
a knob the gallery cannot reach is not evidence: this surface is reachable from
the composer with ``/new `` alone. The two frames are the two states:

* ``with-peers`` — a device that knows peers, where the rows have to read as a
  DEVICE SELECTION: the painted name is the device (``unnamed device`` for one
  without a name), the network it is in is beside it, and the repeated ``remote``
  keyword the value has to carry is not spent on the name column.
* ``no-peers`` — the state EVERY new user starts in, where the list used to
  pre-select a row whose only outcome was a red refusal and the notice naming the
  remedy was suppressed by it. Here the frame must show the notice and no row
  under the cursor.

EVERY PEER HERE IS A FIXTURE: ``network.peers.known_peers`` is replaced before the
app reads it, so no real device id or peer name leaves this machine — the rules
that apply to an SVG on a PR are the rules that apply to a clipboard.

The isolation import must stay FIRST (it re-homes ``HOME`` /
``LOCAL_OPERATOR_CONFIG_DIR`` and drops every inherited ``CMUX_*`` before any
application code reads them), for the reason ``scripts/probe_isolation.py``
documents at length.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.probe_isolation  # noqa: E402, F401
from local_operator.network import peers as peers_mod  # noqa: E402
from local_operator.network.peers import KnownPeer  # noqa: E402
from scripts.visual_capture import (  # noqa: E402
    refuse_flag_shaped_argument,
    save_capture,
)

#: The fixture's peers, as ``known_peers`` returns them: one named, one WITHOUT a
#: name (the row that used to ellipsize a hex id in two columns at once), and a
#: second named one so the column is proved to be a column rather than one row's
#: accident. Real-shaped ids (``d_`` + 32 hex) so the panel lays out the width it
#: really has.
PEERS = (
    KnownPeer(
        device_id="d_1a2b3c4d5e6f708192a3b4c5d6e7f809",
        name="radiant-m4",
        role="admin",
        network_id="n_3985570272c803eeb85a3e23",
        network_name="devmesh",
    ),
    KnownPeer(
        device_id="d_0f1e2d3c4b5a69788796a5b4c3d2e1f0",
        name="",
        role="drive",
        network_id="n_3985570272c803eeb85a3e23",
        network_name="devmesh",
    ),
    KnownPeer(
        device_id="d_77aa88bb99cc00dd11ee22ff33445566",
        name="pixel-8",
        role="read",
        network_id="n_4a1c9e02dd1b7f5c0a83b6e4912d776f",
        network_name="homelab",
    ),
)

VARIANTS = ("with-peers", "no-peers")


async def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(f"usage: new_remote_shot.py OUT.svg {{{'|'.join(VARIANTS)}}} [100x30]")
    # Before either is used: a mistyped flag here would be written to as a path.
    refuse_flag_shaped_argument(sys.argv[1], what="OUT")
    out = Path(sys.argv[1])
    variant = sys.argv[2]
    if variant not in VARIANTS:
        raise SystemExit(f"unknown variant {variant!r}; expected one of {'|'.join(VARIANTS)}")
    size = (100, 30)
    if len(sys.argv) > 3:
        refuse_flag_shaped_argument(sys.argv[3], what="SIZE")
        cols, rows = sys.argv[3].split("x")
        size = (int(cols), int(rows))

    # The app reads `known_peers` through a function-local import on every
    # keystroke, so replacing the module attribute is what the production call
    # actually sees — the same stub `tests/unit/tui/test_new_remote.py` uses.
    peers_mod.known_peers = (  # type: ignore[assignment]
        (lambda root=None: list(PEERS)) if variant == "with-peers" else (lambda root=None: [])
    )

    from local_operator.tui.app import OperatorApp
    from local_operator.tui.widgets.editor import Editor
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        for _ in range(40):
            await pilot.pause()
            if app._session is not None:
                break
        else:
            raise SystemExit("the app never adopted a session, so no picker was opened")
        editor = app.query_one(Editor)
        await pilot.press(*("space" if char == " " else char for char in "/new "))
        await pilot.pause()
        await pilot.pause()
        if not (editor._picker._matches or editor._picker._notice):
            raise SystemExit("the picker is in neither state: nothing was written")
        if variant == "no-peers" and editor._picker.is_open():
            # The defect this frame exists to show the ABSENCE of: a row under the
            # cursor in the no-peers state is a row Enter would run into a refusal.
            raise SystemExit("the no-peers picker listed a row; the frame would show the U4 state")
        if variant == "with-peers" and not editor._picker.is_open():
            raise SystemExit("the with-peers picker listed nothing; the frame would show no picker")
        save_capture(app, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
