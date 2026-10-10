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
    loading                the `decided` beat: the callouts and the preparing
                           line with its shimmering lead.
    failed                 the failure line, with callouts above it.
    empty                  a `done` row with nothing on it: no block, no
                           reserved space.

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


class Owner(FakeSession):
    """A fake owner that HONOURS the supplement read op, so the shot drives the
    same beat-then-upgrade-read path a real runtime answers."""

    async def supplements_for(self, anchors) -> dict:  # type: ignore[no-untyped-def]
        if ANCHOR in {str(anchor) for anchor in anchors}:
            return {ANCHOR: dict(_FULL_ROW)}
        return {}


async def _factory_owner():
    return Owner()


def _seed(app: OperatorApp, shape: str) -> None:
    app._append_block(UserBlock("collect the overnight numbers", fold_width=100))
    app._append_block(_answer(ANSWER))


def _event(shape: str, **extra: object) -> SupplementProgress:
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

    app = OperatorApp(lambda: _factory_owner())
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        _seed(app, shape)
        await pilot.pause()
        if theme != "dark":
            app._apply_theme(theme)
            await pilot.pause()
        if shape != "plain":
            app.post_message(_event(shape))
            await pilot.pause()
            await pilot.pause()
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
