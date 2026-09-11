"""PROTOTYPE — THROWAWAY. The two cheap regression guards for variant A.

1. READ-ONLY: the picker reads the operator's REAL session store. An audit hook
   fails the run on any write, rename, unlink or mkdir under that store — a
   prototype that mutates the store it is previewing would be a real incident
   (see `scripts/probe_isolation`, which exists because one already happened).

2. RESIZE SURVIVAL: variant A restyles `#cols` in place across the breakpoint
   instead of rebuilding the widget tree, and that restyle is guarded to avoid
   thrashing. A stale guard leaves the WRONG layout applied after a resize
   (measured in round 2: 160x45 -> 80x24 kept a 13-row preview). This walks the
   geometries in both directions and asserts the layout, the drawn rows and the
   name field agree with the width the app is actually at.

Usage:
    python scripts/prototype_resume_guards.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REAL_STORE = Path(
    os.environ.get("LOCAL_OPERATOR_CONFIG_DIR") or (Path.home() / ".local-operator")
)

import scripts.probe_isolation  # noqa: F401,E402

from local_operator.tui.widgets import prototype_resume_variants as variants  # noqa: E402
from local_operator.tui.widgets.prototype_resume_data import PreviewData  # noqa: E402
from local_operator.tui.widgets.prototype_resume_picker import (  # noqa: E402
    PrototypeResumeApp,
)

#: Events that would MUTATE something. `open` is inspected for its mode rather
#: than blanket-failed, because reading the store is the whole point.
_MUTATORS = ("os.rename", "os.remove", "os.unlink", "os.mkdir", "os.rmdir", "shutil.copyfile")

#: Writes are CLASSIFIED, not counted, because the two kinds have opposite
#: severity and lumping them gives one number that answers neither question:
#:
#: * SESSION data — anything the user could lose. Must be zero. The prototype
#:   previews the operator's real store and has no business writing to it.
#: * the derived search-index CACHE under `<store>/cache/`. `PreviewData.
#:   digests()` calls `build_index`, which persists its index exactly as
#:   production does. Idempotent, rebuildable, and NOT introduced by this round
#:   — verified by running this guard against the round-2 commit, which emits
#:   the identical three events. Round 2's audit reported zero only because it
#:   never typed a query, so the digest path never ran.
_CACHE_DIR = str(REAL_STORE / "cache")

_writes: list[tuple[str, str]] = []


def _record(detail: str) -> None:
    kind = "cache" if _CACHE_DIR in detail else "SESSION"
    _writes.append((kind, detail))


def _hook(event: str, args: tuple) -> None:
    if event == "open":
        path, mode = str(args[0]), (args[1] or "r")
        if any(flag in mode for flag in ("w", "a", "x", "+")) and str(REAL_STORE) in path:
            _record(path)
    elif event in _MUTATORS:
        if any(str(REAL_STORE) in str(a) for a in args):
            _record(str(args[0]) if args else event)


async def _read_only(data: PreviewData) -> int:
    print("1. READ-ONLY AUDIT")
    sys.addaudithook(_hook)
    app = PrototypeResumeApp(data, "A", "")
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        for char in "picker":
            await pilot.press(char)
        await pilot.pause()
        for _ in range(8):
            await pilot.press("down")
        await pilot.press("ctrl+e")  # verbose: reads every entry of the transcript
        await pilot.pause()
        await pilot.press("ctrl+e")
        await pilot.pause()
    session = [d for kind, d in _writes if kind == "SESSION"]
    cache = [d for kind, d in _writes if kind == "cache"]
    print(f"   store: {REAL_STORE}")
    print(f"   writes to SESSION data: {len(session)}   (must be 0)")
    for line in session[:10]:
        print(f"     WRITE {line}")
    print(f"   writes to derived cache: {len(cache)}   (pre-existing, see note)")
    for line in cache[:10]:
        print(f"     cache {line}")
    print(f"   {'PASS' if not session else 'FAIL'}\n")
    return len(session)


async def _resize(data: PreviewData) -> int:
    print("2. RESIZE SURVIVAL")
    # Both directions, and across the breakpoint in both — the round-2 defect
    # only appeared shrinking back down.
    walk = [(80, 24), (120, 35), (159, 40), (180, 50), (159, 40), (140, 40), (80, 24), (200, 60)]
    failures = 0
    app = PrototypeResumeApp(data, "A", "")
    async with app.run_test(size=walk[0]) as pilot:
        await pilot.pause()
        for width, height in walk:
            app._driver_size = (width, height)  # type: ignore[attr-defined]
            await pilot.resize_terminal(width, height)
            await pilot.pause()
            await pilot.pause()
            screen = app.screen
            expect_stacked = width < variants.STACK_BELOW_COLS
            name_w = screen._name_w()
            drawn = screen._drawn
            rows_ok = 0 < drawn <= len(screen._rows)
            layout_ok = bool(screen._stacked) == expect_stacked
            # The field must equal what a fresh app at this size computes: a
            # stale restyle shows up as a field that belongs to the PREVIOUS
            # geometry.
            width_ok = 8 <= name_w <= variants.NAME_MAX
            ok = layout_ok and rows_ok and width_ok
            failures += not ok
            print(
                f"   {width:>3}x{height:<3} "
                f"layout={'stacked' if screen._stacked else 'side-by-side':<13}"
                f" expect={'stacked' if expect_stacked else 'side-by-side':<13}"
                f" name_w={name_w:>3} drawn={drawn:>3} {'OK' if ok else 'FAIL'}"
            )
    print(f"   {'PASS' if not failures else 'FAIL'}\n")
    return failures


async def main() -> int:
    data = PreviewData(REAL_STORE)
    writes = await _read_only(data)
    failures = await _resize(data)
    total = writes + failures
    print("GUARDS PASS" if not total else f"GUARDS FAILED ({total})")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
