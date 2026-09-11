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
from local_operator.mobile.attach_client import (  # noqa: E402
    OversizedRequest,
    fit_request_frame,
)
from local_operator.tui.app import OperatorApp  # noqa: E402
from scripts.visual_capture import save_capture  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402


def _delivered(session: FakeSession):
    """A ``prompt`` that SUCCEEDS, for the half of a shot that must be sent."""

    async def prompt(text, images=None, **kwargs):
        session.prompts.append(text)

    return prompt


def _photo(width: int, height: int, seed: int = 0) -> str:
    """A base64 PNG of CONTINUOUS-TONE content, for the states about the REFIT.

    The content decides whether this script tests anything. Line art and flat
    fills compress to almost nothing, so a fixture built from them sails under
    the limit and the refit never runs — a 64-attachment fixture of the drawn
    raster below still refitted successfully, which would have captured a
    delivered message while claiming to show a refusal. Blurred noise behaves
    the way a photograph or screenshot does, which is the same reason
    ``tests/unit/session/test_attach_frame_size.py`` builds its fixtures this
    way.

    ``seed`` makes attachments genuinely distinct: identical bytes compress to
    identical sizes and the refit walks smallest first, so a fixture of eight
    copies exercises a tie rather than the ordering the caption renders
    (design round 2, D9).
    """
    import base64
    import io
    import random

    from PIL import Image, ImageFilter

    rng = random.Random(seed or width * height)
    coarse = Image.frombytes(
        "RGB",
        (width // 4, height // 4),
        bytes(rng.getrandbits(8) for _ in range((width // 4) * (height // 4) * 3)),
    )
    smooth = coarse.resize((width, height), Image.Resampling.BILINEAR)
    image = smooth.filter(ImageFilter.GaussianBlur(1.2))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _png(width: int, height: int, seed: int = 11) -> str:
    """A base64 PNG the transcript can actually decode and paint.

    A real raster rather than a stub: ``ImageBlock`` sniffs the header for its
    aspect fit and paints a receipt instead of pixels when the bytes do not
    decode, which would capture the wrong frame. Cheap and legible on screen,
    which is what the NON-refit states need; anything asserting on the refit
    uses :func:`_photo` instead.
    """
    import base64
    import io
    import random

    from PIL import Image, ImageDraw

    image = Image.new("RGB", (width, height), (24, 26, 32))
    draw = ImageDraw.Draw(image)
    random.seed(seed)
    for row in range(8, height - 10, 16):
        x = 10
        while x < width - 40:
            run = random.randint(20, 90)
            draw.rectangle([x, row, x + run, row + 8], fill=(196, 202, 214))
            x += run + 8
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


#: How many attachments it takes to make the refit genuinely give up.
#:
#: NOT A DECORATIVE NUMBER, and it moved once already: at round 1 twenty
#: attachments could not fit, and the smallest-first reclaim (review round 1,
#: MINOR-2) raised capacity enough that the same fixture now REFITS and sends.
#: A capture harness reusing the old figure would photograph a delivered
#: message while claiming to show a refusal, so `_real_refusal` asserts the
#: refusal actually happened rather than trusting this constant (QA round 2).
REFUSAL_ATTACHMENTS = 32


async def _real_refusal(images: list[ImageContent]) -> str:
    """The refusal sentence the PRODUCT composes for ``images``, or raise.

    Why this exists at all: the script used to carry the sentence as a string
    literal, so the frames proved the layout of a row and nothing about the
    words the code generates. Three design round-2 findings (D8, D9, D10) were
    invisible from those captures precisely because the product's own text
    never appeared in them (design round 2). A capture harness that cannot see
    what the product says is a dead instrument.

    ASSERTS THE REFUSAL HAPPENED. The reachable refusal moved from 20
    attachments to 32 when the reclaim landed, so a fixture built on the old
    figure refits successfully and returns no sentence at all — which would
    read as a passing capture of a frame the product cannot reach.
    """
    from local_operator.session.attached import _image_to_wire

    frame = {
        "op": "prompt",
        "req": 1,
        "command_id": "00000000-0000-4000-8000-000000000000",
        "text": "what does this screenshot show?",
        "images": [_image_to_wire(image) for image in images],
    }
    try:
        await fit_request_frame(frame)
    except OversizedRequest as refusal:
        return str(refusal)
    raise AssertionError(
        f"{len(images)} attachments were REFITTED, not refused — this fixture no "
        "longer reaches the refusal path and the capture would prove nothing"
    )


async def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/frames-896")
    out.mkdir(parents=True, exist_ok=True)
    cols, rows = (sys.argv[2] if len(sys.argv) > 2 else "100x30").split("x")
    size = (int(cols), int(rows))

    # MARKERS ON THE IMAGES, exactly as `resolve_markers` stamps them for a
    # real composer draft: the refusal names a chip, and a fixture without
    # markers captures the positional fallback instead of the product's answer
    # (design round 2, D8).
    refused_images = [
        ImageContent(data=_photo(1024, 1024, seed=index), mime_type="image/png", marker=index + 1)
        for index in range(REFUSAL_ATTACHMENTS)
    ]
    refusal = await _real_refusal(refused_images)
    print(f"REFUSAL (product-generated, {REFUSAL_ATTACHMENTS} attachments): {refusal}\n")

    image = ImageContent(data=_png(1672, 941), mime_type="image/png", marker=1)

    for state in ("refusal", "resent", "recovery", "refit"):
        session = FakeSession()
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=size) as pilot:
            await pilot.pause()

            # THE REAL PATH: the refusal is raised by `prompt`, so the app's own
            # worker runs its `except OversizedRequest` branch — the withdrawal,
            # the notice and the restore all fire in the order the product uses
            # them. Calling those helpers by hand would capture a frame the
            # product cannot actually reach.
            if state != "refit":

                async def refuse(text, images=None, **kwargs):
                    session.prompts.append(text)
                    raise OversizedRequest(refusal)

                session.prompt = refuse  # type: ignore[method-assign]

            if state == "refit":
                # A DELIVERED message whose attachments really went through the
                # refit. The caption is composed by `_report_wire_refit_for`
                # from the report the REAL `fit_request_frame` published, so
                # this frame proves the WORDS as well as the layout — the
                # previous version hand-called `_notice_for` with a literal,
                # which is exactly what this script's own comment above warns
                # against (review round 2, NIT-1; design round 2, D9/D10).
                captioned = [
                    ImageContent(
                        data=_photo(1400, 1400, seed=index),
                        mime_type="image/png",
                        marker=index + 1,
                    )
                    for index in range(8)
                ]

                async def deliver_through_the_refit(text, images=None, **kwargs):
                    from local_operator.session.attached import _image_to_wire

                    session.prompts.append(text)
                    await fit_request_frame(
                        {
                            "op": "prompt",
                            "req": 1,
                            "command_id": "00000000-0000-4000-8000-000000000000",
                            "text": text,
                            "images": [_image_to_wire(block) for block in (images or [])],
                        }
                    )

                session.prompt = deliver_through_the_refit  # type: ignore[method-assign]
                app._submit_prompt("what do these show?", captioned)
                for _ in range(30):
                    await pilot.pause()
                    await asyncio.sleep(0.05)
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
