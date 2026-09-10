"""Frames for the oversized-frame refusal and the wire-refit caption (PR #896).

Drives the REAL ``OperatorApp`` through the real submit path, so the captures
carry ``local_operator.tcss`` and the app's own block spacing rather than a
hand-built approximation of them. Four states, each the subject of a design
round 1 finding:

``refusal``    the refused send: echo withdrawn, notice carrying both numbers.
``resent``     the user follows the advice and resends — the state D3 filed,
               where the transcript used to show the message twice.
``recovery``   the same refusal with a NON-EMPTY composer, which is the branch
               that parks the draft and offers a ``DraftRecoveryNotice`` (D5).
``refit``      a delivered message whose attachment lost pixels on the wire, so
               the ``↓`` caption is shown at the rung that earns one (D2).

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/oversized_frame_shot.py OUT_DIR [COLSxROWS]
"""

from __future__ import annotations

import sys
from pathlib import Path

# The repo root, so `scripts.` and `tests.` resolve when run from the worktree.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio  # noqa: E402

import scripts.probe_isolation  # noqa: F401,E402  -- must precede app imports
from local_operator.harness.types import ImageContent  # noqa: E402
from local_operator.mobile.attach_client import OversizedRequest  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from scripts.visual_capture import save_capture  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402


def _delivered(session: FakeSession):
    """A ``prompt`` that SUCCEEDS, for the half of a shot that must be sent."""

    async def prompt(text, images=None, **kwargs):
        session.prompts.append(text)

    return prompt


def _png(width: int, height: int) -> str:
    """A base64 PNG the transcript can actually decode and paint.

    A real raster rather than a stub: ``ImageBlock`` sniffs the header for its
    aspect fit and paints a receipt instead of pixels when the bytes do not
    decode, which would capture the wrong frame.
    """
    import base64
    import io
    import random

    from PIL import Image, ImageDraw

    image = Image.new("RGB", (width, height), (24, 26, 32))
    draw = ImageDraw.Draw(image)
    random.seed(11)
    for row in range(8, height - 10, 16):
        x = 10
        while x < width - 40:
            run = random.randint(20, 90)
            draw.rectangle([x, row, x + run, row + 8], fill=(196, 202, 214))
            x += run + 8
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


REFUSAL = (
    "image 3 is 2.4 MB and will not fit in this message's 0.2 MB per-image "
    "budget (4 attachments) even at its smallest size; remove it and send again"
)


async def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/frames-896")
    out.mkdir(parents=True, exist_ok=True)
    cols, rows = (sys.argv[2] if len(sys.argv) > 2 else "100x30").split("x")
    size = (int(cols), int(rows))

    image = ImageContent(data=_png(1672, 941), mime_type="image/png")

    for state in ("refusal", "resent", "recovery", "refit"):
        session = FakeSession()
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            source = app._interaction

            # THE REAL PATH: the refusal is raised by `prompt`, so the app's own
            # worker runs its `except OversizedRequest` branch — the withdrawal,
            # the notice and the restore all fire in the order the product uses
            # them. Calling those helpers by hand would capture a frame the
            # product cannot actually reach.
            if state != "refit":

                async def refuse(text, images=None, **kwargs):
                    session.prompts.append(text)
                    raise OversizedRequest(REFUSAL)

                session.prompt = refuse  # type: ignore[method-assign]

            if state == "refit":
                # A DELIVERED message: the row and its picture stay, and the
                # caption is the only thing the refit adds.
                app._submit_prompt("what does this screenshot show?", [image])
                await pilot.pause()
                source.turn.submitted_blocks = None
                app._notice_for(
                    source,
                    "image resized to fit this message: #1 1672x941 → 768x432 ↓",
                    "note",
                )
                await pilot.pause()
            else:
                if state == "recovery":
                    # The branch that parks the draft: the user started typing
                    # during the ~315 ms refit, so the composer is not empty.
                    app._editor().load_text("and another thing")
                    await pilot.pause()

                app._submit_prompt("what does this screenshot show?", [image])
                # The worker is a Textual worker: let it run to its `except`.
                for _ in range(12):
                    await pilot.pause()
                    await asyncio.sleep(0.05)

                if state == "resent":
                    # The user does what the copy says: drops the attachment and
                    # sends again. Exactly one prompt must be on screen.
                    session.prompt = _delivered(session)  # type: ignore[method-assign]
                    app._editor().load_text("")
                    app._submit_prompt("what does this screenshot show?", [])
                    for _ in range(12):
                        await pilot.pause()
                        await asyncio.sleep(0.05)

            await pilot.pause()
            path = out / f"{state}-{size[0]}x{size[1]}.svg"
            save_capture(app, path)
            # Consecutive frames: a first frame differing from the settled one
            # is motion the user sees.
            await pilot.pause()
            settled = out / f"{state}-{size[0]}x{size[1]}-settled.svg"
            save_capture(app, settled)
            same = path.read_bytes() == settled.read_bytes()

            blocks = app._transcript_view().blocks()
            users = sum(1 for block in blocks if type(block).__name__ == "UserBlock")
            images_shown = sum(1 for block in blocks if type(block).__name__ == "ImageBlock")
            recovery = sum(1 for block in blocks if type(block).__name__ == "DraftRecoveryNotice")
            print(
                f"{state:9s} UserBlocks={users} ImageBlocks={images_shown} "
                f"recovery_rows={recovery} prompts_sent={len(session.prompts)} "
                f"stable={same} -> {path.name}"
            )
            for block in blocks:
                name = type(block).__name__
                if name in ("NoticeBlock", "DraftRecoveryNotice"):
                    # `text()` is NoticeBlock's, not the base block's; read it
                    # off the instance so the print survives a block kind that
                    # does not carry one.
                    printable = getattr(block, "text", None)
                    if callable(printable):
                        print(f"            {name}: {printable()!r}")


if __name__ == "__main__":
    asyncio.run(main())
