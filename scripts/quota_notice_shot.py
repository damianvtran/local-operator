"""Capture the splash's pre-emptive no-quota line, at any terminal width.

Usage::

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/quota_notice_shot.py out.svg [WIDTHxHEIGHT] [STATE]

States — the fixtures live in ``tests/unit/tui/test_quota_notice.py``
(``quota_state`` / ``frame_app``), so the frames and the tests render the SAME
data and cannot drift:

``depleted``           a spent DeepSeek balance: the sentence and its top-up URL.
``plan-window``        a spent Anthropic plan window: the reset sentence.
``radient-unverified`` a Radient account whose free credits wait behind email
                       verification — the verify-to-claim sentence is the real
                       ``radient_recovery.recovery_line`` output, fed from the
                       process cache a failed-402 turn would have warmed.
``free-model``         a spent account on a stated-free model: NO row — the
                       model can still send, so a warning would be false.
``unknown``            a stale row: NO row — nothing definite to say.
``connected``          a healthy plan window: NO row.
``coexist``            the quota row and the credential warning, together.
``narrow``             ``depleted`` at a width the row cannot hold the URL in
                       (default 60x24): the URL is dropped whole, never clipped.

Isolates HOME and the config dir before importing the app
(``isolate_capture``) so a capture never reads the operator's providers, caches
or sessions and cannot touch a live one.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()  # BEFORE the app imports: they read HOME/config at import time.

import local_operator.update as _update  # noqa: E402


class _NoUpdateBehind:
    """Stand-in for :func:`local_operator.update.check_latest`.

    The splash's update line comes from a live PyPI probe on a background
    worker, so whether it lands before the shot is a race — and it adds a row
    to the splash, which moves the composition against every frame captured
    before it landed (the suite pins the same worker in
    ``tests/unit/tui/conftest.py``). Pinned to "not behind", the honest common
    case; bound onto the module, which ``OperatorApp._check_for_update``
    imports at call time, so the worker cannot race the capture.
    """

    behind = False
    latest: str | None = None


def _no_update_behind(*args: object, **kwargs: object) -> _NoUpdateBehind:
    return _NoUpdateBehind()


_update.check_latest = _no_update_behind  # type: ignore[assignment]

from local_operator.tui.widgets.welcome import build_welcome_lines  # noqa: E402
from tests.unit.tui.test_quota_notice import frame_app  # noqa: E402
from tests.unit.tui.test_welcome import _settled_welcome  # noqa: E402


async def main() -> None:
    out = Path(sys.argv[1]).resolve()
    size = sys.argv[2] if len(sys.argv) > 2 else ""
    state = sys.argv[3] if len(sys.argv) > 3 else "depleted"
    fixture = "depleted" if state == "narrow" else state
    if not size:
        size = "60x24" if state == "narrow" else "100x30"
    width, height = (int(part) for part in size.split("x"))

    # A neutral cwd before painting: the splash's cwd row and the composer
    # footer both name the process's working directory, and an evidence frame
    # must not carry the capturing worktree's path (the convention
    # ``docs/evidence/model-receipt`` states). Nothing is written here.
    os.chdir("/private/tmp")

    app = frame_app(fixture)
    async with app.run_test(size=(width, height)) as pilot:
        await pilot.pause()
        welcome = await _settled_welcome(pilot)
        await pilot.pause()
        save_capture(app, out)
        info = welcome._info
        lines = build_welcome_lines(info, welcome.size.width, welcome.size.height)
        # The two numbers the still cannot show: the block is content-sized
        # (its height IS the rows it draws), and the screen has not gone
        # scrollable under the extra row (virtual == actual, no scrollbar).
        screen = pilot.app.screen
        print(
            f"state={state} size={width}x{height} rows={len(lines)}"
            f" welcome_h={welcome.size.height} virtual_h={screen.virtual_size.height}"
            f" scrollbar={bool(screen.show_vertical_scrollbar)}"
            f" quota={'yes' if info.quota_notice is not None else 'no'} -> {out}"
        )
        assert len(lines) == welcome.size.height, "the block is content-sized"
        assert screen.virtual_size.height == screen.size.height, "no scroll overflow"


if __name__ == "__main__":
    asyncio.run(main())
