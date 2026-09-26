"""Capture the ``/new`` device picker, in both states a user meets.

Usage: python scripts/new_remote_shot.py OUT.svg {with-peers|no-peers} [110x34]

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

BOTH ARE CAPTURED IN THE SAME APP STATE, and the script refuses rather than
hoping (QA round 12, Q-12-2): at 110x30 the pair differed by the whole boot
composition — 77 glyphs of the welcome splash's mark in one frame and none in
the other, because the picker's own rows take the vertical room the mark's height
ladder needs — so what a reviewer compared was not just the picker. See
``DEFAULT_SIZE`` for the measurement and for the grid both are taken at.

EVERY PEER HERE IS A FIXTURE: ``network.peers.known_peers`` is replaced before the
app reads it, so no real device id or peer name leaves this machine — the rules
that apply to an SVG on a PR are the rules that apply to a clipboard.

THE SPLASH'S UPDATE ROW IS PINNED ABSENT FOR THE SAME REASON, and it needed to be
(design round 5, D32). The welcome splash draws `! latest is v<X> — /update` when
the background probe finds a newer release, and that row SHIFTS everything below
it down one row. This probe re-homes ``HOME``, so the version cache is always
cold and the probe really does reach PyPI: the frame therefore used to be a
function of the capturing machine's resolver. Measured 2026-09-22, both captures
rendered at the committed artifact's own 1.8x zoom and diffed against
``static/tui-mesh-picker.png``:

* with DNS: PyPI answered (`latest='0.62.2', behind=True`), the banner row took
the place of the blank one at y=319.9, every row below moved down 17 px, and the
diff was that whole shifted block (bbox x408..1179 y558..765) — the exact bbox
D32 reported.
* with ``XPC_FLAGS=0x2`` (no resolver): AE=0, i.e. the committed PNG IS this
  state.

So the committed copy documents the BROKEN-RESOLVER machine while nothing said
so, and a re-capture on a healthy one silently produced a different frame. The
pin below is what ``known_peers`` is: the frame stops depending on what this
machine can reach.

TWO MORE ROWS OF THIS FRAME CAME FROM THE CAPTURING ENVIRONMENT RATHER THAN FROM
THE CHECKOUT, and both are pinned here for the same reason (design round 6, D35
and D36). Measured 2026-09-22 at head ``2d2ebcd0``:

* THE VERSION ROW. The committed copy reads ``v0.59.11`` and a fresh capture of
the SAME commit reads ``v0.62.1`` — one row, bbox ``x408..1179 y558..765``, the
exact bbox D32 reported — and the cause was not a commit: ``pyproject.toml`` is
0.62.1 at both heads, while this worktree's ``.venv`` dist-info was rewritten
when the fleet's shared interpreter was rebuilt. The row is an
``importlib.metadata`` read of the INSTALLED distribution, so it describes the
capturing install, not the tree. THE CLAIM THIS DOCSTRING USED TO MAKE — that "the
frame is captured from a checkout whose version is a property of the commit" — was
refuted by that measurement; ``_pin_version`` is what makes it true again.
* THE CWD ROW, and the composer band beside it. The splash prints ``os.getcwd()``,
so the committed copy carries ``/Users/damian/local-operator-worktrees/mesh-network``
and the band carries that path's tail — which is why the byte-exact comparison
above had to be run from the repository root. ``_pin_cwd`` does what the page
fixture already does (``scripts/pages_shot.py``): the capture runs in a directory
under this probe's own re-homed ``HOME``, which the splash renders as
``~/workspace``. Nothing machine-specific reaches a shipped frame, which is the rule
rather than the one-off — the version row is the same class of value and had already
drifted.

Each pin is a fixture, and each is pinned at the seam the production call actually
reads (a module attribute, or the process's own cwd), so what a reviewer compares is
the frame's layout rather than the machine that took it.

The isolation import must stay FIRST (it re-homes ``HOME`` /
``LOCAL_OPERATOR_CONFIG_DIR`` and drops every inherited ``CMUX_*`` before any
application code reads them), for the reason ``scripts/probe_isolation.py``
documents at length.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.probe_isolation  # noqa: E402, F401
from local_operator import update as update_mod  # noqa: E402
from local_operator.network import peers as peers_mod  # noqa: E402
from local_operator.network.peers import KnownPeer  # noqa: E402
from local_operator.tui.widgets import welcome as welcome_mod  # noqa: E402
from scripts.visual_capture import (  # noqa: E402
    refuse_flag_shaped_argument,
    save_capture,
    svg_text_runs_by_row,
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

#: The grid both variants are captured at, and WHY NOT 110x30 (QA round 12,
#: Q-12-2). At 30 rows the two frames were NOT the same state: the peerless one
#: painted 77 glyphs of the welcome splash's mark and the with-peers one painted
#: NONE, because the picker's own rows take the vertical room the mark's height
#: ladder needs — measured at 110x30 as 0 vs 77, and at 110x34 and 110x40 as 77
#: vs 77, stable across repeated runs rather than a race. A pair meant to differ
#: by the picker's content differed by the whole boot composition as well, so the
#: comparison a reviewer makes off it was not like for like. 34 rows is the
#: shortest grid at which BOTH states paint the same splash, and it is the height
#: UX round 3 walked this flow at.
DEFAULT_SIZE = (110, 34)

#: The one splash row this frame must never carry. The pinned state below is the
#: absent one, and the census asserts it on the exported bytes.
_UPDATE_ROW = "latest is v"

#: The version the splash's version row shows in this frame — PINNED, not read
#: (design round 6, D35; the measurements are in the module docstring). Set to the
#: version THIS COMMIT declares, so the figure cannot document a version the tree
#: never was; move it only with a re-shoot, because a pin left behind turns the
#: frame into a picture of a release that never happened.
PINNED_VERSION = "0.62.1"


#: The directory name the frame's cwd rows show, under this probe's re-homed HOME
#: (design round 6, D36). A child of HOME rather than HOME itself: at HOME the splash
#: reads "~" but the composer band renders "~/" + ".", i.e. "~/." — and both rows are
#: in the frame. A name a user's own frame would show, and one nothing else creates.
PINNED_CWD = "workspace"


def _pin_version() -> None:
    """Make the splash's version row a fixture, like the peers and the probe.

    The row resolves through ``welcome.app_version``, an ``importlib.metadata`` read
    of the INSTALLED distribution, and that is a property of the environment the
    capture runs in: rebuilding this worktree's ``.venv`` moved the committed frame's
    version row by one value with `pyproject.toml` unchanged at both heads. Replacing
    the module attribute is what the production call sees — ``session_welcome_info``
    in that module calls it as a module global.
    """

    def app_version() -> str:
        return PINNED_VERSION

    welcome_mod.app_version = app_version  # type: ignore[assignment]


def _pin_cwd() -> None:
    """Home the capture under its own ``HOME``, so no personal path is in the frame.

    The splash prints ``os.getcwd()`` and the composer band prints the same path
    shortened, so a capture launched from the repository writes the worktree root
    into the artifact — ``static/tui-mesh-picker.png`` carried
    ``/Users/damian/local-operator-worktrees/mesh-network``. This is the page
    fixture's own answer to that (``scripts/pages_shot.py``), applied at the seam the
    app actually reads: the process's cwd.

    ``HOME`` IS RESOLVED AND WRITTEN BACK FIRST, because the two halves of the app
    spell this path differently without it: ``os.getcwd()`` answers with symlinks
    resolved (``/private/var/...``) while ``Path.home()`` reads the environment
    verbatim (``/var/...``), and ``_shorten_home`` collapses the second prefix out of
    the first. Unresolved, the frame carries a temporary path instead of ``~/``.
    """

    home = Path(os.environ["HOME"]).resolve()
    os.environ["HOME"] = str(home)
    cwd = home / PINNED_CWD
    cwd.mkdir(parents=True, exist_ok=True)
    os.chdir(cwd)


def _pin_update_check() -> None:
    """Make the splash's update row absent, whatever this machine can reach.

    The app imports ``check_latest`` FUNCTION-LOCALLY at both call sites
    (``OperatorApp._check_for_update`` and ``_cmd_update``), so replacing the
    module attribute is what the production call actually sees — the same
    mechanism ``known_peers`` above uses, and the same one
    ``tests/unit/tui/test_new_remote.py`` uses for the peer fixture.

    The stub answers the way an unreachable PyPI does (``latest is None``,
    ``behind is False``), which is the state the committed artifact already
    documents — see the module docstring for the two measurements. It does NOT
    fake an install: ``installed`` is read from this install, so the value the
    splash would show on a ``/update`` stays honest.
    """

    def check_latest(*_args: object, **_kwargs: object) -> update_mod.VersionCheck:
        return update_mod.VersionCheck(
            installed=update_mod.installed_version(), latest=None, behind=False
        )

    update_mod.check_latest = check_latest  # type: ignore[assignment]


#: The mark glyphs the welcome splash draws. Used only to check the frame is in
#: the settled state below; `local_operator.tui.widgets.welcome` owns the art.
_MARK_GLYPHS = frozenset("█▀▄")


async def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(f"usage: new_remote_shot.py OUT.svg {{{'|'.join(VARIANTS)}}} [100x30]")
    # Before either is used: a mistyped flag here would be written to as a path.
    refuse_flag_shaped_argument(sys.argv[1], what="OUT")
    # Resolved HERE, before `_pin_cwd` moves the process: a caller-relative OUT would
    # otherwise be written relative to the fixture's own directory.
    out = Path(sys.argv[1]).resolve()
    variant = sys.argv[2]
    if variant not in VARIANTS:
        raise SystemExit(f"unknown variant {variant!r}; expected one of {'|'.join(VARIANTS)}")
    size = DEFAULT_SIZE
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
    # Before the app boots: the splash's background probe runs on its own thread
    # and can land between any two `pilot.pause()` calls below.
    _pin_update_check()
    # ...and the two rows that would otherwise be this machine's: the installed
    # version the splash prints, and the directory it is launched from.
    _pin_version()
    _pin_cwd()

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
        # BOTH FRAMES IN THE SAME STATE (QA round 12, Q-12-2), asserted on the
        # artifact rather than trusted to the size: the welcome splash must be up
        # in BOTH, or the pair differs by the boot composition as well as by the
        # picker — which is exactly what shipped. The census reads the exported
        # SVG, the same bytes the reviewer reads, and the mark is the part of that
        # splash a short terminal sheds first.
        exported = app.export_screenshot()
        mark = sum(exported.count(glyph) for glyph in _MARK_GLYPHS)
        # THE SPLASH STATE IS PART OF THE FRAME, so it is censused rather than
        # trusted to the pin above (design round 5, D32): the pin is one import
        # away from being bypassed by a second path to the same row, and a frame
        # whose composition moved is exactly what this file exists to make
        # impossible to ship quietly. Absent is the only passing count.
        #
        # READ THE ROW, NOT THE FILE. The check cannot be
        # `exported.count(_UPDATE_ROW)`: the capture helper gives every grapheme
        # cluster its own `<tspan>` origin, so that count is 0 even on a frame that
        # paints the banner (measured on the pre-pin capture, whose `y=319.9` row
        # reads `! latest is v0.62.2 — /update`). A census written that way is a
        # check that can never fail — worse than no census, because it looks like
        # one. `svg_text_runs_by_row` is the parse the mesh rig's own census uses.
        banner = sum(1 for runs in svg_text_runs_by_row(exported) if _UPDATE_ROW in "".join(runs))
        if banner:
            raise SystemExit(
                f"the splash is carrying the update row {banner} time(s), so this "
                "frame is the DNS-dependent one: the pin in _pin_update_check did not "
                "take effect, or the splash grew a second writer for that row."
            )
        print(
            f"census[{variant}]: update row absent, "
            f"rows {len(svg_text_runs_by_row(exported))}, mark glyphs {mark}"
        )
        if mark == 0:
            raise SystemExit(
                "the welcome splash's mark is not in this frame, so the two states are not "
                "comparably captured: the with-peers frame loses it when the picker's rows "
                "take the room it needs (measured at 110x30), which made a pair that "
                f"differed by the whole boot composition. Capture at {DEFAULT_SIZE[0]}x"
                f"{DEFAULT_SIZE[1]} or taller — see DEFAULT_SIZE."
            )
        save_capture(app, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
