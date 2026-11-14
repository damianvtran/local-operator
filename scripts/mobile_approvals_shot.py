"""Rendered evidence for ``/mobile approvals on`` (design v2 §6, C3).

One running ``OperatorApp`` (the real app, stylesheet loaded) produces every frame.
Each frame is reached by TYPING the command into the composer and pressing Enter,
so the real slash dispatch, ``_cmd_mobile`` and ``setup_authority`` run:

    01-usage          `/mobile approvals` with no `on`: the usage line
    02-no-terminal    `/mobile approvals on` where the TUI cannot hand the terminal
                      to a password prompt (the headless test driver cannot suspend):
                      this terminal app's refusal, nothing staged or installed, and
                      no progress line before it
    03-cancelled      the person cancelled the prompt: progress, gesture, then the
                      cancelled outcome as a warning
    04-on             the person answered the prompt: the N5 success sentence, then
                      the file-only caveat (U18) as its own warning notice
    05-already-on     run again: idempotent, no second prompt, no progress line
    06-on-80col       a fresh host at 80 columns: how the success pair wraps

ISOLATION. ``probe_isolation`` re-homes HOME/config before any import. The anchor
root is the in-process test seam pointed at a temp dir (this script must never touch
a real /etc or /Library anchor), and root ownership is faked exactly as the unit
tests do. ``elevate.run_elevated`` is replaced BEFORE the first frame by a scripted
gesture (cancel, then approve: land the staged bytes), so no frame can launch a real
prompt; pkexec is pinned through the test-launcher-only seam
(``elevate._PKEXEC_CANDIDATES_OVERRIDE``) to a logging shim, because since R1-5 a
``PATH`` shim no longer shadows the fixed system path. The real elevation path is
covered by the desktop route's real-path evidence.

    LOP_REPO=<tree> env -u NO_COLOR TERM=xterm-256color \\
        .venv/bin/python scripts/mobile_approvals_shot.py OUTDIR
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(os.environ.get("LOP_REPO") or Path(__file__).resolve().parents[1]).resolve()
sys.path.insert(0, str(REPO))

import asyncio  # noqa: E402
import tempfile  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from typing import Any, Iterator  # noqa: E402

import scripts.probe_isolation  # noqa: E402,F401  -- MUST be the first import
from local_operator.operator import elevate, trust  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from scripts.visual_capture import save_capture  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

CONFIG_DIR = Path(os.environ["LOCAL_OPERATOR_CONFIG_DIR"])  # from probe_isolation


def _isolate_anchor() -> None:
    trust._ANCHOR_ROOT_OVERRIDE = Path(tempfile.mkdtemp(prefix="lop-shot-anchors-"))
    real_open = trust._open_no_follow

    def as_root(path: Path) -> Any:
        opened = real_open(path)
        if opened is None:
            return None
        descriptor, _ = opened
        return descriptor, SimpleNamespace(st_uid=0, st_mode=0o100644)

    trust._open_no_follow = as_root  # type: ignore[assignment]


async def _type(pilot: Any, app: OperatorApp, text: str) -> None:
    await pilot.press(*list(text))
    await pilot.press("enter")
    for _ in range(60):
        await pilot.pause(0.05)
        if not getattr(app, "_operator_setup_busy", False):
            break
    await pilot.pause(0.3)


async def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "mobile-approvals-shots").resolve()
    out.mkdir(parents=True, exist_ok=True)
    _isolate_anchor()
    # No real pkexec, ever: the seam names a shim that only logs (R1-5).
    shim_log = out / "launcher-shim.log"
    shim = out / "pkexec-shim"
    shim.write_text(f'#!/bin/sh\necho "pkexec $*" >> {shim_log}\nexit 1\n')
    shim.chmod(0o755)
    elevate._PKEXEC_CANDIDATES_OVERRIDE = (str(shim),)
    gestures: list[str] = []

    def gesture(command: list[list[str]], **_: Any) -> elevate.ElevationResult:
        """The person at the prompt, scripted: cancel once, then approve."""
        if not gestures:
            gestures.append("cancelled")
            return elevate.ElevationResult(1, cancelled=True)
        gestures.append("approved")
        trust.anchor_path().write_bytes(trust.staging_path(CONFIG_DIR).read_bytes())
        return elevate.ElevationResult(0)

    elevate.run_elevated = gesture  # type: ignore[assignment]

    @contextmanager
    def no_suspend() -> Iterator[None]:
        yield

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(110, 26)) as pilot:
        await pilot.pause(0.5)
        await _type(pilot, app, "/mobile approvals")
        save_capture(app, out / "01-usage.svg")

        await _type(pilot, app, "/mobile approvals on")
        save_capture(app, out / "02-no-terminal.svg")
        assert not trust.staging_path(CONFIG_DIR).exists()
        assert not trust.anchor_path().exists()
        assert gestures == []

        # From here the driver can hand over the terminal, as a real one does, and
        # suspend is a no-op around the scripted gesture.
        app.suspend = no_suspend  # type: ignore[method-assign]
        type(app._driver).can_suspend = property(lambda self: True)  # type: ignore[assignment]
        await _type(pilot, app, "/mobile approvals on")
        save_capture(app, out / "03-cancelled.svg")
        assert gestures == ["cancelled"] and not trust.anchor_path().exists()

        await _type(pilot, app, "/mobile approvals on")
        save_capture(app, out / "04-on.svg")
        assert trust.load_anchor().usable

        await _type(pilot, app, "/mobile approvals on")
        save_capture(app, out / "05-already-on.svg")
        assert gestures == ["cancelled", "approved"]

    # A fresh host for the narrow frame: the temp anchor this script wrote goes.
    trust.anchor_path().unlink()
    narrow = OperatorApp(lambda: _factory(FakeSession()))
    async with narrow.run_test(size=(80, 26)) as pilot:
        narrow.suspend = no_suspend  # type: ignore[method-assign]
        await pilot.pause(0.5)
        await _type(pilot, narrow, "/mobile approvals on")
        save_capture(narrow, out / "06-on-80col.svg")
        assert trust.load_anchor().usable
    assert not shim_log.exists(), "nothing reached a launcher"
    print(out)


if __name__ == "__main__":
    asyncio.run(main())
