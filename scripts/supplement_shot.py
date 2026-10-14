"""Capture the turn-supplements block under a final answer, for visual validation.

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/supplement_shot.py OUT.svg [COLSxROWS] [SHAPE] [THEME]

``SHAPE`` selects the state the block is in, because each state is a different
picture and the amendment's rules are about how each one sits under the answer:

    plain       (default)  the answer alone — the BEFORE frame: no supplement
                           events at all, i.e. what the same conversation shows
                           on `main`.
    populated              files + the "N more" row, with HTML components on the
                           row that must NOT appear (skipped, no placeholder).
    files-only             ≤ 2 files and no components: the one-line arm.
    loading                the `decided` beat: the known files (one here, so
                           the one-liner arm) and the preparing line with its
                           shimmering lead.
    failed                 the quiet failure line: nothing landed, so no
                           callouts ride it.
    empty                  a `done` row with nothing on it: no block, no
                           reserved space.
    expanded               the populated row AFTER a pilot click on its "N more"
                           row: the stored paths revealed in place. The click
                           is driven through the real pilot hit test, so the
                           frame is the disclosure a user gets, not a
                           hand-expanded build.
    narrow-long            the overflow case (design round 1, D1): a long-path
                           file row captured at a NARROW size (pass `60x20` —
                           lane 56, where the row overflows and the fit must
                           keep its indent and file NAME while ellipsizing the
                           path); at the default size the same row fits whole.

``THEME`` is `dark` (default) or `light`; the block must re-ink with the ramp,
which is only checkable as a pair of frames.

NO IMAGE SHAPE, deliberately: the contract declares a row's ``images`` but no
engine path writes one yet (the generator emits HTML components), so a frame
would show a state users cannot reach. Image entries mount through the existing
``ImageBlock`` transport (``app._mount_supplement_images``) and are pinned by
the unit suite until the lane that starts emitting them lands.

The conversation is seeded the same way in every shape, so two frames differ by
the block and nothing else. Events are posted exactly as the runtime posts them
(``SupplementProgress`` messages), not painted by hand.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.events import SupplementProgress  # noqa: E402
from local_operator.tui.widgets.assistant import AssistantBlock  # noqa: E402
from local_operator.tui.widgets.transcript import UserBlock  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession  # noqa: E402

ANSWER = (
    "The overnight batch finished. `reports/latency.md` holds the percentile "
    "table and `data/bench.csv` the raw rows; the long-form notes are in "
    "`reports/notes.md`."
)
ANCHOR = "msg-final"


def _answer(text: str) -> AssistantBlock:
    block = AssistantBlock()
    block.update_text(text)
    block.finalize_text()
    block.completion_anchor_id = ANCHOR
    return block


def _file(path: str, size: int = 4120) -> dict[str, object]:
    return {
        "path": path,
        "name": path.rsplit("/", 1)[-1],
        "size_bytes": size,
        "mtime": 1.0,
        "why": "written by write",
    }


def _components() -> list[dict[str, object]]:
    # An HTML component: on the TUI it is skipped SILENTLY, which is exactly
    # what this frame is for — the reader must see the callouts and the "more"
    # row with no frame, stub or note where this document would have been.
    return [
        {
            "attachment": "f" * 32,
            "title": "Latency by region",
            "source": "generated",
            "mime": "text/html",
            "height_hint": 220.0,
        }
    ]


#: The FULL journaled row, as ``supplements_for`` serves it: the featured files
#: plus the disclosure keys the live event deliberately does not carry.
_FULL_ROW = {
    "anchor": ANCHOR,
    "job": "3f9c1a5e5b20",
    "version": 1,
    "state": "done",
    "files": [
        _file("reports/latency.md"),
        _file("data/bench.csv", 913),
        _file("reports/q3.md", 2048),
        _file("reports/q4.md", 2048),
    ],
    "files_more": 3,
    "more": ["data/raw.csv", "reports/appendix.md", "data/summary.json"],
    "components": _components(),
    "error": "",
}

#: The files-only finish's journaled row: the two files and nothing else — the
#: disclosure keys exist only when the featured list (≤ ``maxFeatured``, 4)
#: did not carry the whole set.
_FILES_ONLY_ROW = {
    "anchor": ANCHOR,
    "job": "3f9c1a5e5b20",
    "version": 1,
    "state": "done",
    "files": [_file("reports/latency.md"), _file("data/bench.csv", 913)],
    "components": [],
    "error": "",
}

#: A failed finish whose generated blocks never landed (a refusal, not a
#: bound): the failure row alone — no files, so no callouts ride it.
_FAILED_ROW = {
    "anchor": ANCHOR,
    "job": "3f9c1a5e5b20",
    "version": 1,
    "state": "failed",
    "files": [],
    "components": [],
    "error": "model refused",
}

#: The overflow case's journaled row (design round 1, D1): a 52-char path whose
#: first row is 83 cells wide, so a 60-column terminal's lane (56) cannot hold
#: it and the frame shows exactly where the fit truncated.
_NARROW_ROW = {
    "anchor": ANCHOR,
    "job": "3f9c1a5e5b20",
    "version": 1,
    "state": "done",
    "files": [
        _file("reports/experiments/2026-10-09/summary-of-results.md"),
        _file("data/bench.csv", 913),
        _file("reports/q3.md", 2048),
    ],
    "files_more": 3,
    "more": ["data/raw.csv", "reports/appendix.md", "data/summary.json"],
    "components": [],
    "error": "",
}

#: What each shape's read serves: the SHAPE'S OWN journaled row, never another
#: shape's. ``plain``/``loading``/``empty`` have no settled row to serve
#: (nothing ran, the job is still in its running beat, or the finish journaled
#: nothing), so their read answer is the empty mapping — the truth a real owner
#: would answer for such an anchor.
_SHAPE_ROWS: dict[str, dict[str, object]] = {
    "populated": _FULL_ROW,
    "expanded": _FULL_ROW,
    "files-only": _FILES_ONLY_ROW,
    "failed": _FAILED_ROW,
    "narrow-long": _NARROW_ROW,
}


class Owner(FakeSession):
    """A fake owner that answers the supplement read with the SHAPE's own row.

    The read is a REAL path the beats drive (the terminal beat's upgrade read
    in ``_apply_supplement``), so it has to answer like the journal would for
    the state being captured: an owner that answered the populated row for
    every anchor repainted ``files-only`` and ``failed`` into the populated
    picture before capture — one picture standing in for three claimed states
    (round-1 review R1).
    """

    def __init__(self, shape: str) -> None:
        super().__init__()
        self._shape = shape

    async def supplements_for(self, anchors) -> dict:  # type: ignore[no-untyped-def]
        row = _SHAPE_ROWS.get(self._shape)
        if row is None or ANCHOR not in {str(anchor) for anchor in anchors}:
            return {}
        return {ANCHOR: dict(row)}


async def _factory_owner(owner: Owner):
    return owner


def _seed(app: OperatorApp, shape: str) -> None:
    app._append_block(UserBlock("collect the overnight numbers", fold_width=100))
    app._append_block(_answer(ANSWER))


def _event(shape: str, **extra: Any) -> SupplementProgress:
    job = "3f9c1a5e5b20"
    if shape == "loading":
        return SupplementProgress(
            ANCHOR, job, 1, "decided", files=[_file("reports/latency.md")], **extra
        )  # type: ignore[arg-type]
    if shape == "failed":
        return SupplementProgress(  # type: ignore[return-value]
            ANCHOR, job, 1, "failed", error="model refused", error_type="provider"
        )
    if shape == "empty":
        return SupplementProgress(ANCHOR, job, 1, "done")
    if shape == "files-only":
        return SupplementProgress(
            ANCHOR,
            job,
            1,
            "done",
            files=[_file("reports/latency.md"), _file("data/bench.csv", 913)],
        )
    if shape == "narrow-long":
        return SupplementProgress(
            ANCHOR,
            job,
            1,
            "done",
            files=[
                _file("reports/experiments/2026-10-09/summary-of-results.md"),
                _file("data/bench.csv", 913),
                _file("reports/q3.md", 2048),
            ],
        )
    if shape == "image":
        # Kept as a refusal rather than a silent fallback: this shape became
        # unproducible (no engine path writes ``images`` yet — see the module
        # docstring) and a frame that quietly captured something else would be
        # worse than none.
        raise SystemExit("the image shape is not producible: no engine path writes ``images``")
    return SupplementProgress(  # type: ignore[return-value]
        ANCHOR,
        job,
        1,
        "done",
        # The wire's own shape: ``files`` is the FEATURED list (≤ maxFeatured,
        # 4). ``files_more``/``more`` ride the ROW and are served by the read
        # op — which is what the stub session below returns, so this frame
        # exercises the real beat-then-upgrade-read path end to end.
        files=[
            _file("reports/latency.md"),
            _file("data/bench.csv", 913),
            _file("reports/q3.md", 2048),
            _file("reports/q4.md", 2048),
        ],
        components=_components(),
    )


async def main() -> None:
    out = sys.argv[1]
    size = (100, 30)
    if len(sys.argv) > 2:
        cols, rows = sys.argv[2].split("x")
        size = (int(cols), int(rows))
    shape = sys.argv[3] if len(sys.argv) > 3 else "plain"
    theme = sys.argv[4] if len(sys.argv) > 4 else "dark"

    owner = Owner(shape)
    app = OperatorApp(lambda: _factory_owner(owner))
    async with app.run_test(size=size) as pilot:
        # BEFORE seeding, wait for the session's boot to land. `_adopt_session`
        # replays through `_project_settled_rows`, which ends in the lazy
        # supplement read; seeding first let that read see the seeded answer
        # and mount a journaled row before the shape's own beat — a race that
        # painted `plain` as a populated block and `loading` with the read's
        # disclosure rows on some runs (round-1 review R1). Adoption is
        # synchronous end to end, so once the session is adopted the boot's
        # read request has already run.
        for _ in range(200):
            await pilot.pause()
            if app._session is owner:
                break
        else:
            raise SystemExit("the session was never adopted")
        _seed(app, shape)
        await pilot.pause()
        if theme != "dark":
            app._apply_theme(theme)
            await pilot.pause()
        if shape != "plain":
            app.post_message(_event(shape))
            # Wait for the beat's own read round-trip instead of racing it: a
            # terminal beat asks one and the shape's owner answers it, and a
            # frame captured mid-flight would show the pre-upgrade block while
            # claiming the upgraded state.
            for _ in range(200):
                await pilot.pause()
                if ANCHOR in app._supplement_live and not app._supplement_read_inflight:
                    break
            else:
                raise SystemExit(f"{shape}: the beat was never processed")
            await pilot.pause()
            if _SHAPE_ROWS.get(shape) is not None and ANCHOR not in app._supplement_read:
                raise SystemExit(
                    f"{shape}: the read did not land — the frame would show a "
                    "state its own journaled row never reached"
                )
            if shape == "expanded":
                # The block's one disclosure, driven the way a user drives it:
                # a click on the painted "N more" line, through the pilot's hit
                # test rather than calling `expand_more()` directly. The guard
                # keeps the frame honest — a capture that did not expand is an
                # error, not a picture of the collapsed state under this name.
                block = app._supplement_blocks.get(ANCHOR)
                if block is None or block._more_line is None:
                    raise SystemExit("expanded: the disclosure row never painted")
                padding = block.styles.padding.top if block.styles.padding else 0
                delivered = await pilot.click(block, offset=(4, padding + block._more_line))
                await pilot.pause()
                if not delivered or not block._expanded:
                    raise SystemExit("expanded: the pilot click never revealed the rest")
        save_capture(app, out)
        blocks = app._transcript_view().blocks()
        print(
            f"{Path(out).name}: size={tuple(app.screen.size)} "
            f"blocks={[type(b).__name__ for b in blocks]} "
            f"supplement_blocks={len(app._supplement_blocks)} "
            f"read={sorted(app._supplement_read)} "
            f"live={sorted(app._supplement_live)} "
            f"inflight={sorted(app._supplement_read_inflight)}"
        )


asyncio.run(main())
