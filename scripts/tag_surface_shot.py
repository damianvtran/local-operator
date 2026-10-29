"""Capture the styled ``/tag`` picker surface for visual validation.

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \\
        scripts/tag_surface_shot.py OUTDIR [COLSxROWS]

Modelled on ``scripts/settings_shot.py``: it drives the REAL
:class:`OperatorApp`, which is the only host that loads ``local_operator.tcss``.
A scratch config dir is used for every capture (``isolate_capture``), so the
frames show a KNOWN registry — four categories, two applied — and can never
write to the real store.

ONE SETTLED PASS, not one launch per state. The frames exist as evidence that
SVG and PNG are the same frame, and the first shipped set was captured across
runs whose boot chrome differed (an update banner in one, a `connecting…` band
in another), so three pairs could not be reproduced from each other. Every
frame here is taken after the band has settled to the model line, and the
static (non-picker) chrome is identical across them by construction.

States, in the order the operator reaches them:

* ``tag-surface-no-arg``   bare ``/tag `` — the current conversation's list
* ``tag-surface-partial``  ``/tag dee`` — filtered rows plus the create action
* ``tag-surface-create``   ``/tag New Label`` — exactly one create action
* ``tag-surface-created``  after Enter — the receipt, no second confirmation
* ``tag-surface-after-create`` reopened — the count moved, the new tag applied

PNGs are produced separately: ``rsvg-convert -o X.png X.svg`` rasterizes the
committed SVG at its native size, which is what makes the pair checkable.
"""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

from local_operator.paths import config_dir  # noqa: E402
from local_operator.session.session_tags import (  # noqa: E402
    set_entity_tags,
    write_categories,
)
from local_operator.tui.app import NoticeFn, OperatorApp  # noqa: E402
from tests.unit.tui.test_app_pilot import (  # noqa: E402
    FakeProviderController,
    FakeSession,
)

#: A real 12-hex id: the store validates the shape, and the app paints the
#: same conversation in the sidebar frames.
SESSION_ID = "abcdef123456"


class _TagSession(FakeSession):
    @property
    def session_id(self) -> str:
        return SESSION_ID


def _frame_text(app: OperatorApp) -> str:
    """The visible text of the composited frame, for settlement checks.

    ``export_screenshot`` splits runs across ``tspan`` elements, so matching on
    the raw SVG would miss the sentence it is looking for. Joining each
    ``<text>`` element's children is enough to ask "is the band still saying
    connecting".
    """
    svg = app.export_screenshot()
    return " ".join(
        re.sub(r"<[^>]+>", "", body) for body in re.findall(r"<text[^>]*>(.*?)</text>", svg, re.S)
    )


async def _settle(app: OperatorApp, pilot) -> None:
    """Wait for the boot chrome to stop moving.

    A capture taken mid-boot is a different screen from the one the operator
    ever reads (the band says ``connecting…``), and it is what made three of
    the first frames unreproducible from their own SVGs.
    """
    for _ in range(200):
        await pilot.pause()
        text = _frame_text(app)
        if getattr(app, "_session", None) is not None and "connecting…" not in text:
            return
    raise SystemExit("the app never settled: the band kept saying connecting…")


def _notice_collector() -> tuple[list[str], NoticeFn]:
    """The command handler's notice seam, as a collector.

    The receipt reaches the transcript and the status band through the real
    handler; nothing here inspects the strings, so the notices are collected
    rather than printed.
    """
    seen: list[str] = []

    def notice(body: str, kind: str = "info") -> None:
        seen.append(body)

    return seen, notice


def _seed() -> None:
    """The committed fixture: four categories, ``Deep Work`` primary, ``Review``."""
    root = config_dir()
    review = write_categories(root, [(None, "Review")])[0]
    ops = write_categories(root, [(review["id"], "Review"), (None, "Ops")])[1]
    deep_work = write_categories(
        root, [(review["id"], "Review"), (ops["id"], "Ops"), (None, "Deep Work")]
    )[2]
    write_categories(
        root,
        [
            (review["id"], "Review"),
            (ops["id"], "Ops"),
            (deep_work["id"], "Deep Work"),
            (None, "Deep Dive"),
        ],
    )
    set_entity_tags(root, "session", SESSION_ID, [deep_work["id"], review["id"]], deep_work["id"])


async def main() -> None:
    out = Path(sys.argv[1])
    size = (120, 34)
    if len(sys.argv) > 2:
        cols, rows = sys.argv[2].split("x")
        size = (int(cols), int(rows))
    out.mkdir(parents=True, exist_ok=True)

    _seed()
    # The update check hits the network and paints a banner whose presence is
    # not part of what these frames claim; pinning it off is what makes the
    # capture repeatable on a machine that happens to be behind.
    OperatorApp._check_for_update = lambda self: None  # type: ignore[method-assign]

    session = _TagSession()

    async def factory() -> FakeSession:
        return session

    app = OperatorApp(factory, provider_controller=FakeProviderController())
    async with app.run_test(size=size) as pilot:
        await _settle(app, pilot)
        _seen, notice = _notice_collector()

        app._cmd_tag("", notice)
        await pilot.pause()
        await pilot.pause()
        save_capture(app, out / "tag-surface-no-arg.svg")

        editor = app._editor()
        editor._set_text_and_caret("/tag dee", len("/tag dee"))
        await pilot.pause()
        await pilot.pause()
        save_capture(app, out / "tag-surface-partial.svg")

        editor._set_text_and_caret("/tag New Label", len("/tag New Label"))
        await pilot.pause()
        await pilot.pause()
        save_capture(app, out / "tag-surface-create.svg")

        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        save_capture(app, out / "tag-surface-created.svg")

        app._cmd_tag("", notice)
        await pilot.pause()
        await pilot.pause()
        save_capture(app, out / "tag-surface-after-create.svg")


if __name__ == "__main__":
    asyncio.run(main())
