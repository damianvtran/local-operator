"""Bounding an image so a provider will accept it.

This module exists because of a specific failure mode, and the failure mode is
worth stating before the code: an image block that a provider refuses does not
fail once. It is in the conversation HISTORY, so every subsequent request
carries it again and earns the same 400 — including compaction, which has to
send the history in order to summarise it. The session is then unrecoverable
from the inside: reload replays the same block, ``/compact`` cannot run, and
the user sees the identical error answering every prompt forever. That is the
shape of anthropics/claude-code#19031, #24387, #47391 and #50708, and it is why
bounding happens on the way IN rather than being left to the provider.

The dimension rule is the one that is easy to miss, because it is not a
constant. Anthropic caps a single image at 8000 pixels on its long edge, but a
request carrying **more than 20 images** switches to a much stricter per-image
limit of 2000 pixels, and refuses the whole request with
``At least one of the image dimensions exceed max allowed size for
many-image requests: 2000 pixels``. So an image that was accepted for hours can
start being refused purely because the conversation grew past twenty frames —
no bytes changed, and nothing about that image was ever wrong on its own. A
long agent session with screenshot evidence crosses twenty images routinely.

:data:`IMAGE_MAX_EDGE` is therefore not a token optimisation that happens to be
safe; it is BELOW the strict 2000-pixel limit on purpose, so the many-image
threshold can never be reached no matter how many frames a session accumulates.

Centralised here rather than living next to one caller because "an image is
about to become an ``ImageContent``" is the invariant, and it has more than one
site: the ``read`` tool returning an image block, and the composer attaching a
pasted screenshot. The composer used to forward pasted bytes verbatim at
whatever size the screen produced, which is exactly how a 2206x266 paste wedged
a session permanently — the read tool had been bounded since it was written, so
the hole was only visible from the path that had no bound at all.

Bounding on the way in has one structural limit, and it is the reason
:func:`rebound_oversize_image` exists below. It can only protect blocks created
AFTER it shipped. A transcript is replayed verbatim on every resume, so history
written by an older build stays oversized on disk and goes on being refused by a
build whose composer can no longer produce such a block — observed as a 2206x266
paste that answered every prompt with the many-image refusal long after the
paste path was fixed. The repair therefore runs on the rendered history, where
every request converges, rather than at the two creation sites.

TWO KNOWING EXCEPTIONS to the creation-time bound, so a reader does not conclude
the invariant is total:

- ``compaction/snapcompact.py`` renders its own frames and does not come
  through :func:`bound_image_for_model`. Their geometry is a per-provider
  billing decision rather than a paste: 1568px, 1932px for high-res Claude
  lines, and 2048x2046 under the Google shape. Only the last is over the
  many-image ceiling, and it is reachable on another provider because
  ``session.set_model`` can swap mid-session and a Gemini-shaped archive then
  replays to whatever is current (review round 1, F4).

  The billing trade is genuinely not re-decided here, and that is why the
  render-time repair triggers at :data:`IMAGE_REFUSAL_MAX_EDGE` rather than at
  :data:`IMAGE_MAX_EDGE`: the 1932px frame is under the ceiling, so it is left
  exactly as compaction rendered it, and only the 2048px frame — which a
  provider would actually refuse — is shrunk. Gating on 1568 instead rewrote
  every high-res frame on every render, which is the trade this exception
  exists to leave alone.

  These frames are also the reason the ladder cares what an image is MADE of.
  They are bilevel renderings of a 5x7 pixel font, so a smooth resampler turns
  every one-pixel stroke into a grey ramp: it destroys the legibility the font
  was chosen for AND replaces a 2-value image with a 256-value one that PNG can
  no longer compress, making the "shrunk" frame 21x larger than the source. A
  bilevel source is therefore resampled with NEAREST and shrunk only as far as
  the refusal ceiling demands.
- ``_forward_undecoded`` cannot resize at all, because on that host there is no
  decoder. See its own docstring for what it does enforce.

TWO BOUNDS, and the split is deliberate. :data:`IMAGE_MAX_EDGE` is the
CORRECTNESS ceiling: it is what the many-image refusal argument above is
measured against, and what a repair is judged by. :data:`IMAGE_INGEST_MAX_EDGE`
is what images ENTERING the context are actually resized to, and it is lower
because the argument there is billing rather than refusal — a provider bills an
image by pixel area, so pixels that buy no legibility are pure cost. Ingest is
the only site free to move: a REPAIR must not chase it, because repairing a
block that the provider would have accepted rewrites history that a prompt
cache is holding, turning a saving into a full-prefix rewrite. That is why
:func:`rebound_oversize_image` still triggers at :data:`IMAGE_REFUSAL_MAX_EDGE`
and why lowering the ingest bound leaves every existing 1568px block untouched.
"""

from __future__ import annotations

import base64
import hashlib
import io
from collections import OrderedDict
from typing import Any

from local_operator.helpers import heif_image_module, pillow_image_module
from local_operator.media import ImageInfo, sniff_image
from local_operator.optional import missing_extra_error

#: Long-edge ceiling in pixels for an image handed to a provider.
#:
#: Two independent reasons land on a number this low, and BOTH have to hold.
#:
#: Correctness: a request carrying more than 20 images is held to a 2000-pixel
#: per-image limit (see the module docstring). Anything above that line turns
#: into a permanently wedged session once the twenty-first frame arrives, so
#: the cap must sit below 2000 with room to spare — not at it.
#:
#: Cost: Anthropic downsizes anything above 1568 server-side and bills the
#: resized token count either way, so pixels past this line are pure upload
#: with zero fidelity reaching the model. omp uses the same number and this
#: repo's snapcompact already renders its frames at 1568. Measured on real
#: files: a 2560x1600 UI screenshot costs 5,461 image tokens untouched and
#: 2,049 at 1568 (2.7x), and a 4032x3024 phone photo — 1.5 MB as JPEG, so it
#: passes any byte cap easily — costs 16,257 tokens untouched against 2,459
#: resized (6.6x).
#:
#: Images ENTERING the context take the tighter :data:`IMAGE_INGEST_MAX_EDGE`
#: instead; this constant stays the correctness ceiling and a repair is still
#: measured against it.
IMAGE_MAX_EDGE = 1568
#: Long-edge ceiling for images ENTERING the context (``read`` tool, pasted
#: screenshots), and the bound the cost argument below is measured on.
#:
#: Tighter than :data:`IMAGE_MAX_EDGE`, which stays the CORRECTNESS ceiling —
#: the many-image refusal line and the snapcompact billing geometry both answer
#: to it and are deliberately not re-decided here. Ingest is the one site free
#: to move: 1024 sits below every provider refusal line (2000px many-image,
#: 5 MB base64 wall) with wide margin.
#:
#: Providers bill an image by pixel AREA (~w*h/750 tokens), so after the 1568
#: server-side downsample there is still a billed region between 1024 and 1568
#: where pixels cost tokens without buying legibility. Measured on a real 4-up
#: document collage from this machine's own traffic: 3,184 visual tokens at
#: 1568 against 1,359 at 1024 (57% cheaper), with a vision model reading the
#: 1024 render recovering appendix titles, a 5-column score matrix, page
#: numbers and 7pt header prose. Below 1024 the margin thins — dense 6-7pt
#: table body text is at the edge at 768 — so 1024 is the floor that keeps
#: document work legible.
#:
#: ONE MEASURED LOSS, so nobody re-derives it: on a 4-up COLLAGE the footer's
#: 24-character hex report id degrades from a 4-row ink span to 1 and stops
#: being transcribable (design round 1, D1). Prose survives a downscale because
#: a reader reconstructs it from context; a long opaque identifier has no
#: context to reconstruct from. It is specific to multi-page collages — a page
#: captured alone measures ~784x1006 and is not resized at all — and the escape
#: hatch is to crop the region and re-read it, which passes through unresized.
#:
#: LINE ART IS EXEMPT and keeps :data:`IMAGE_MAX_EDGE`. The measurement above is
#: photographic content, which has no one-pixel strokes to lose; bilevel
#: renderings of a small pixel font do, and shrinking them to 1024 destroys the
#: legibility the font was chosen for — a snapcompact archive frame is exactly
#: such a rendering, and at 1024 it renders "compacted" as "conpacted" (PR #603
#: review round 1, F1; QA round 1). The exemption uses the same
#: ``_is_line_art`` predicate the repair path does.
IMAGE_INGEST_MAX_EDGE = 1024
#: The provider ceiling a REPAIR is measured against, which is deliberately not
#: :data:`IMAGE_MAX_EDGE`. Anything this module CREATES is bounded to 1568 for
#: the cost reasons above; but a block that already exists is only worth
#: rewriting when it would actually be refused, and the refusal line is the
#: many-image limit of 2000 (see the module docstring).
#:
#: Conflating the two silently re-decided a billing trade that is not this
#: module's to make: snapcompact renders Anthropic high-res archive frames at
#: 1932px — under 2000, never refused — and repairing at 1568 rewrote every one
#: of them on every render (review round 1, F1). The ingest bound and the repair
#: bound answer different questions and only coincide by accident.
IMAGE_REFUSAL_MAX_EDGE = 2000
#: What a repaired line-art block is actually resized TO, deliberately a little
#: under :data:`IMAGE_REFUSAL_MAX_EDGE` rather than exactly at it.
#:
#: The provider's wording is "exceed max allowed size ... 2000 pixels", so 2000
#: itself reads as legal and landing on it would probably work. Probably is the
#: wrong standard for the one number whose failure mode is a permanently wedged
#: session, and this module already holds that position for the ingest cap —
#: "the cap must sit below 2000 with room to spare — not at it". A repair that
#: lands exactly on a boundary inferred from an error message would be trusting
#: a strict inequality nobody has tested, on the code path that exists because
#: the provider changed its mind about what it accepts.
#:
#: 1960 keeps a 2% margin, which costs a pixel font nothing measurable (the
#: Google-shape frame goes to 0.957 scale instead of 0.977) and leaves room for
#: a provider that reads its own limit as inclusive.
IMAGE_REPAIR_TARGET_EDGE = 1960
#: Refuse to DECODE above this pixel count (~200 MB of RGBA at 4 bytes/pixel).
#: Checked against the header dimensions BEFORE the decode allocates, because a
#: decompression bomb is small on disk by construction: a byte cap cannot see
#: it coming. Measured: a flat 7000x7000 PNG is 0.15 MB on disk and takes 577
#: ms to decode, resize and re-encode. 50M pixels is ~7000x7000, comfortably
#: above any camera or display this reads from and comfortably below Pillow's
#: own 89M bomb threshold, so the refusal is ours and is an error rather than a
#: warning.
IMAGE_MAX_PIXELS = 50_000_000
#: Encoded-byte threshold for the image block, before base64 inflates it by
#: 4/3. Two jobs. It decides when a small in-bounds image is forwarded VERBATIM
#: (cheapest and lossless — no re-encode can improve an image the model will
#: see at its original size), and it decides when the resized PNG is too fat to
#: keep, which is the only reason lossy JPEG is ever reached.
#:
#: A TRIGGER, not a guarantee: the ladder stops after JPEG rather than chasing
#: quality down, so pathological input still lands above it (uniform noise at
#: 1568x1176 measured 1.19 MiB of quality-85 JPEG). The guarantee is the wall
#: this is set against — providers reject images over 5 MB of base64, and the
#: long-edge cap means even that pathological case encodes to 1.59 MiB, 32% of
#: the wall. 1 MiB was picked to leave 3.7x headroom on the ordinary path, and
#: the fat cases are real: a photographic 1568x1176 frame re-encodes to a
#: 3.3 MiB PNG (4.46 MiB base64, inside 5 MB by only 12%) and to a 804 KiB JPEG.
IMAGE_MAX_BYTES = 1024 * 1024
#: JPEG quality used for that fallback. 85 is the standard visually-lossless
#: point; on the sampled files it turned 1.9 MB of re-encoded PNG into 271 KB.
IMAGE_JPEG_QUALITY = 85
#: The base64 wall a REPAIR is measured against, the byte counterpart to
#: :data:`IMAGE_REFUSAL_MAX_EDGE`. Providers reject an image block over 5 MB of
#: base64, and dimensions alone cannot see that coming: a 1900x1900 noise PNG is
#: comfortably under every pixel ceiling and still encodes to 14.5 MB of base64
#: (review round 2, F9). Set below the wall so a block that would be refused is
#: repaired before it is sent, not after.
IMAGE_REFUSAL_MAX_B64_BYTES = 4 * 1024 * 1024


def _forward_undecoded(data: bytes, info: ImageInfo) -> tuple[bytes, str, str]:
    """Ship image bytes VERBATIM on a host with no usable Pillow.

    Pillow reaches a default install as a pillow-heif dependency rather than a
    direct one, and pillow-heif is the most platform-fragile wheel this project
    pulls. When it is missing or broken there is no decoder, so there is also
    no resize and no validation beyond the header — the two things
    :func:`bound_image_for_model` normally provides. Refusing every image on
    such a host would be the worse trade: a screenshot the model can look at
    beats a paragraph explaining why it cannot, and the format is one the
    provider clients already serialize.

    What remains enforceable from the header alone is the BYTE cap, so that is
    the line. Above it the answer is an error, because forwarding an unbounded
    blob is how a session ends up wedged behind a provider that refuses it.

    The DIMENSION cap cannot be enforced here — enforcing it means resizing,
    and resizing is exactly what is unavailable — so an oversized image on such
    a host is named in the returned SUMMARY and forwarded anyway. That is a
    deliberate residual, and the summary is the only warning of it, so a caller
    that discards the summary also discards the warning: ``read`` puts it in the
    caption the model sees, while the composer currently drops it (review round
    1, F2). The session-level degrade
    (``providers.failover.is_image_rejection``) is the net underneath both.
    """
    if not info.sendable:
        # Not a degrade: no provider accepts HEIC, so forwarding it verbatim
        # would GUARANTEE the refusal rather than risk it. Transcoding is the
        # only way to send one, and transcoding is what is unavailable.
        raise ValueError(missing_extra_error("images", "HEIC/HEIF decoding"))
    if len(data) > IMAGE_MAX_BYTES:
        raise ValueError(
            f"it is {len(data)} bytes, over the {IMAGE_MAX_BYTES}-byte cap for an "
            f"unresized image, and {missing_extra_error('images', 'resizing it')}"
        )
    summary = f"{info.mime_type}, {info.dimensions or 'dimensions unknown'}, {len(data)} bytes"
    if info.width and info.height and max(info.width, info.height) > IMAGE_MAX_EDGE:
        # Worth saying rather than swallowing: this is the case the resize
        # exists for, and the model is about to be billed several times the
        # tokens for it. Naming it is also the only hint anyone gets that the
        # host is missing the extra.
        summary += ", too large to send efficiently and no decoder to resize it"
    else:
        summary += ", forwarded without resizing"
    return data, info.mime_type, summary


#: EXIF tag number for ``Orientation``. Spelled as the integer rather than
#: imported from ``PIL.ExifTags`` so that reading it costs no Pillow import on
#: a path that may never decode anything.
_EXIF_ORIENTATION_TAG = 274


def _needs_exif_rotation(image: Any) -> bool:
    """Does this image carry an ``Orientation`` tag that actually turns it?

    Asked BEFORE transposing, because ``ImageOps.exif_transpose`` returns a
    ``copy()`` when there is nothing to do — so "is this a different object?"
    cannot answer it, and using that as the test silently destroyed the verbatim
    rung (every tagless image re-encoded for nothing).

    ``1`` means "already upright" and is treated as no rotation, so a camera
    that writes the tag explicitly still gets the cheap path.

    Best-effort: a corrupt EXIF block raises from deep inside Pillow, and the
    honest answer to "is this rotated?" when the metadata is unreadable is no.
    """
    try:
        return image.getexif().get(_EXIF_ORIENTATION_TAG, 1) not in (1, None)
    except Exception:  # noqa: BLE001 — unreadable metadata is not a reason to fail an image
        return False


def _exif_transposed(image: Any) -> Any:
    """``image`` rotated per its EXIF ``Orientation`` tag.

    ``ImageOps`` is imported here rather than at module scope for the reason
    given in :func:`~local_operator.helpers.pillow_image_module`: nothing in
    this package may pay Pillow's import on a path that never decodes an image.

    Best-effort by design. A corrupt or unreadable EXIF block raises from deep
    inside Pillow, and an unrotated image is a far better outcome than a failed
    attachment — orientation is a refinement, not the payload.
    """
    try:
        from PIL import ImageOps

        return ImageOps.exif_transpose(image) or image
    except Exception:  # noqa: BLE001 — see the docstring; never fail an image for this
        return image


#: Fraction of horizontally adjacent pixel pairs that may differ before a
#: two-valued image is judged DITHERED rather than line art. Line art is made of
#: solid runs — a pixel font measures 0.010 and flat rules 0.000 — while a
#: halftone encodes tone as alternating pixels and measures ~0.491, so the two
#: populations are separated by roughly 50x and the exact threshold is not
#: delicate. 0.15 sits in the empty middle.
_LINE_ART_MAX_TRANSITION_DENSITY = 0.15

#: Rows per strip when measuring that density. The measurement reads EVERY
#: pixel — see :func:`_is_line_art` for why sampling was abandoned — so this
#: exists only to bound peak memory, not to bound the work. A band allocates
#: four buffers of roughly ``width * band`` bytes (the strip crop, its two
#: shifted copies and their difference), which at 512 rows is 14.3 MB on a
#: 7000px-wide image against 196 MB for the same walk done whole. Smaller bands
#: cut that further but add per-crop overhead without changing the result.
_LINE_ART_BAND_ROWS = 512


def _is_line_art(image: Any) -> bool:
    """Is this image hard-edged black-and-white line art?

    Decides which resampler a downscale may use, so it has to answer a narrower
    question than "is it two-valued". Two distinct values covers BOTH the pixel
    font this protects and a dithered halftone, and the correct filter is
    opposite for the two: NEAREST keeps a glyph's one-pixel strokes crisp, but
    on a halftone — where tone is encoded as the RATIO of alternating black and
    white pixels — dropping every other pixel destroys the tone it was encoding.
    Measured on a dithered gradient, NEAREST scores a mean tone error of 84.5
    against 11.4 for LANCZOS (review round 3, F11).

    So two tests, in cost order. The histogram is bounded by ``maxcolors`` and
    bails immediately on anything photographic — which is the gate that keeps
    this cheap, because only an image already known to be two-valued reaches the
    second test.

    The density is then measured over EVERY pixel, not over sampled rows. An
    earlier version read 64 rows at a stride, and every stride is alignable:
    content whose own vertical period shares a factor with the stride is read at
    the same phase every time. A round ``height // rows`` stride misreads an
    image that is solid on exactly the rows it lands on, and switching to a
    prime stride only moves the collision — at a stride of 31 any height that is
    a multiple of 31 collapses the walk onto ``height / 31`` distinct rows (a
    930px image samples 30 rows, a 155px image just 5). Since the two
    populations are separated by ~50x, the failure is not a near miss: a
    misread halftone is downscaled with NEAREST, which destroys the tone it
    encodes.

    Reading everything removes the premise instead of retuning it, and reads
    MORE pixels in LESS time. The comparison is vectorised in Pillow — the image
    against itself shifted one column, differenced, then histogrammed — so it
    replaces a Python-level per-pixel loop with a C-level pass. Measured against
    the sampled version it supersedes, on the same fixtures: 1200x2048 costs
    5.3 ms against 15.4 ms, and at the 50M-pixel decode ceiling ~88 ms against
    ~264 ms. Roughly 2.9x faster while being exact, because the sample's cost
    was never in the number of pixels it read but in reading them one Python
    object at a time.

    Cost is bounded twice over regardless: the histogram gate above means
    photographic content never reaches this walk, and the repair it gates is
    memoized, so a given image pays once. It runs banded so peak memory stays
    bounded on a large image.

    Asked of the PIXELS rather than the mode: snapcompact renders ``L``, a
    scanner produces ``1``, and an ordinary grayscale photograph is also ``L``.
    Multi-channel images are excluded outright — two distinct RGB tuples is a
    two-colour graphic, not a glyph rendering, and NEAREST on it would be a
    guess.
    """
    if image.mode not in ("1", "L"):
        return False
    try:
        # Imported inside the function, like every other Pillow use here:
        # `imaging` is reachable from the core import path and a module-level
        # `from PIL import ...` costs ~23 ms and ~7.6 MB RSS on runs that never
        # touch an image. `tests/unit/test_import_graph.py` pins that.
        from PIL import ImageChops

        if image.getcolors(maxcolors=2) is None:
            return False
        gray = image if image.mode == "L" else image.convert("L")
        width, height = gray.size
        if width < 2 or height < 1:
            return False
        # Count horizontally adjacent pairs that differ, one horizontal band at
        # a time. Within a band the count is the number of non-zero pixels in
        # |strip - strip shifted left one column|, which Pillow computes in C.
        transitions = 0
        compared = (width - 1) * height
        for top in range(0, height, _LINE_ART_BAND_ROWS):
            bottom = min(top + _LINE_ART_BAND_ROWS, height)
            strip = gray.crop((0, top, width, bottom))
            rows = bottom - top
            delta = ImageChops.difference(
                strip.crop((0, 0, width - 1, rows)),
                strip.crop((1, 0, width, rows)),
            )
            # histogram()[0] is the count of identical pairs; everything above
            # it is a transition, whatever the magnitude of the change.
            transitions += sum(delta.histogram()[1:])
        # ``compared`` cannot be zero: the guard above establishes width >= 2
        # and height >= 1, so there is always at least one adjacent pair.
        return transitions / compared <= _LINE_ART_MAX_TRANSITION_DENSITY
    except Exception:  # noqa: BLE001 — a heuristic must never fail an image
        return False


def _is_line_art_bytes(data: bytes, info: ImageInfo) -> bool:
    """:func:`_is_line_art` for bytes that have not been decoded yet.

    Separate from the decoded check because the repair path has to make the
    decision BEFORE it calls into the ladder, and must not fail an image just
    because it could not answer the question — an unreadable histogram simply
    means "treat it as photographic", which is the conservative default.
    """
    module = pillow_image_module() if info.sendable else heif_image_module()
    if module is None:
        return False
    try:
        with module.open(io.BytesIO(data)) as image:
            image.load()
            return _is_line_art(image)
    except Exception:  # noqa: BLE001 — the ladder reports decode failures, not this
        return False


def _guard_pixel_budget(width: int, height: int) -> None:
    """Refuse an image whose pixel count would dominate the process.

    A decompression bomb is small on disk by construction, so a byte cap cannot
    see it coming and only the dimensions can.
    """
    if width * height > IMAGE_MAX_PIXELS:
        raise ValueError(
            f"it is {width}x{height} ({width * height:,} pixels) and the decode limit is "
            f"{IMAGE_MAX_PIXELS:,} pixels"
        )


def bound_image_for_model(
    data: bytes, info: ImageInfo, *, max_edge: int | None = None
) -> tuple[bytes, str, str]:
    """Decode, bound and re-encode image bytes for a provider.

    Returns ``(payload, wire_mime, summary)``; raises ``ValueError`` with a
    human-readable message when the bytes will not decode. The raise is
    load-bearing: a corrupt or truncated image forwarded as an image block
    earns a ``Could not process image`` 400, and the block is in the history by
    then. The session layer recovers from that, but a backstop is not a licence
    — the bad block is still a wasted round trip and a degraded session, and
    decoding here is where it is cheap to avoid. So the decode is never
    skipped, not even on the verbatim path.

    CPU-bound and unbounded in duration by the caller's standards: a 20 MP
    screenshot measures ~315 ms and the 50 MP ceiling ~577 ms on an M3 Max. Any
    caller on an event loop must run this in a thread.

    The ladder, cheapest first:

    1. Verbatim, when the image is already inside both bounds and in a format
       the clients serialize. No re-encode can improve an image the model sees
       at its original size, and PNG round-tripping routinely makes files
       BIGGER (a 2560x1600 UI screenshot measured 550 KB on disk against 335 KB
       re-encoded only because the resize came with it).
    2. Resize to the ingest bound (:data:`IMAGE_INGEST_MAX_EDGE` by default;
       :data:`IMAGE_MAX_EDGE` when a caller passes it explicitly) and re-encode
       as PNG. Lossless, which is what a screenshot of small text needs.
    3. JPEG when that PNG blows :data:`IMAGE_MAX_BYTES`, and only when it
       actually comes out smaller. PNG is a bad photographic codec and that is
       the usual case here — the sampled 1672x941 photographic PNG re-encoded
       to 1.9 MB of PNG against 271 KB of quality-85 JPEG — but it is not the
       only one, so the choice is measured rather than assumed.

    ``max_edge`` overrides :data:`IMAGE_INGEST_MAX_EDGE` for callers that are
    REPAIRING rather than ingesting. The default is the ingest bound, and it
    is the right default for anything arriving from a paste or a file read;
    but a caller shrinking an image that already exists is trading fidelity it
    did not choose, and for line art the smallest possible reduction is worth
    real money. See :func:`rebound_oversize_image`, which passes the refusal
    ceiling for bilevel sources so a pixel font is shrunk by 2% instead of 23%.

    With no decoder available the whole ladder collapses to
    :func:`_forward_undecoded`.
    """
    image_module = pillow_image_module() if info.sendable else heif_image_module()
    if image_module is None:
        return _forward_undecoded(data, info)

    # The pixel cap wants to fire BEFORE the decode allocates, and for every
    # format except HEIF the header already answers it. HEIF keeps its size in
    # a meta-nested ispe box that media.sniff_image deliberately does not walk,
    # so those are capped below on the decoded size instead — later than ideal,
    # but the only point at which the number exists.
    if info.width and info.height:
        _guard_pixel_budget(info.width, info.height)

    try:
        image = image_module.open(io.BytesIO(data))
        _guard_pixel_budget(*image.size)
        # Multi-frame sources never pass through: providers read frame 0 and
        # ignore the rest, so an animation's other frames are bytes uploaded to
        # be discarded.
        frames = getattr(image, "n_frames", 1)
        image.load()

        # Bake EXIF orientation into the PIXELS before anything else looks at
        # the size. A phone camera stores the sensor's raw frame plus an
        # ``Orientation`` tag saying how to turn it, and re-encoding drops the
        # tag while keeping the pixels — so a resized photo arrives sideways
        # unless it is transposed first.
        #
        # Applied on EVERY rung rather than only before a resize, because the
        # bug this closes is an INCONSISTENCY: the verbatim rung keeps the tag
        # and the resize rung silently discarded it, so two identical photos
        # differing only in resolution were delivered in different orientations
        # (review round 1, F1). Transposing unconditionally means the delivered
        # pixels are upright whichever rung runs, and the model never has to
        # know a tag existed.
        #
        # It also settles the question the tag itself raises: a provider that
        # ignores EXIF would render the untransposed image wrongly, and one that
        # honours it would double-rotate a transposed image that still carried
        # the tag. Transposing and re-encoding without the tag is correct under
        # both, which is why this cannot be left to the provider.
        #
        # The tag is READ first and the transpose only runs when it says the
        # image actually turns. That ordering is load-bearing: a tagless image
        # must still reach the verbatim rung below with its ORIGINAL bytes, and
        # ``exif_transpose`` returns a ``copy()`` rather than the same object
        # when there is nothing to do — so testing identity afterwards would
        # re-encode every ordinary screenshot for nothing.
        rotated = _needs_exif_rotation(image)
        # The size ON DISK, captured BEFORE the transpose can swap the axes.
        # The summary's "source WxH" clause means "what a model would see from
        # ``ls`` or a re-read", so reading it after a rotation reported
        # 3000x4000 for a file every other tool calls 4000x3000 (review round
        # 2, F7).
        source_width, source_height = image.size
        if rotated:
            image = _exif_transposed(image)
        width, height = image.size

        long_edge = max(width, height)
        # LINE ART KEEPS THE CORRECTNESS BOUND, and this carve-out is the same
        # argument the repair path already makes (review round 2, F8): every
        # pixel removed from a one-pixel stroke is a stroke that may vanish.
        #
        # The ingest bound below it is a BILLING argument measured on
        # photographic content, which has no strokes to lose. Applying it to
        # bilevel content took the sharper reduction with none of the
        # protection, and it is reachable in ordinary use rather than
        # hypothetical: a snapcompact archive frame is a 1568px bilevel
        # rendering of a 5x7 pixel font, it is passed through
        # ``bound_image_for_model`` whenever such a frame is re-read from disk,
        # and at 1024 its glyphs lose their strokes — measured on a real
        # ``render_frame`` output, the text "The session was compacted" is
        # legible at 1568 and is not at 1024 (PR #603 review round 1, F1).
        #
        # ``_is_line_art`` is deliberately narrower than "two distinct values"
        # (a dithered halftone is bilevel and must NOT take this path), so the
        # predicate here is the same one the repair uses rather than a second
        # opinion about what line art is.
        edge_cap = IMAGE_INGEST_MAX_EDGE if max_edge is None else max_edge
        if max_edge is None and _is_line_art(image):
            edge_cap = IMAGE_MAX_EDGE
        if (
            info.sendable
            and frames == 1
            and not rotated
            and long_edge <= edge_cap
            and len(data) <= IMAGE_MAX_BYTES
        ):
            return data, info.mime_type, f"{info.mime_type}, {width}x{height}, {len(data)} bytes"

        if long_edge > edge_cap:
            scale = edge_cap / long_edge
            size = (max(1, round(width * scale)), max(1, round(height * scale)))
            # LANCZOS everywhere EXCEPT hard-edged line art, where it is
            # actively destructive. Line art's strokes are one pixel wide, so a
            # smooth filter turns each into a grey ramp — destroying the thing
            # that made it legible AND replacing a 2-value image with a 256-value
            # one that PNG can no longer compress. Measured on a snapcompact
            # archive frame: 31 KB of pixel-font text became 665 KB after a
            # LANCZOS downscale, 21x LARGER than the source it was shrinking,
            # with the glyphs smeared (review round 2, F8).
            #
            # NEAREST keeps the two values and the compression, and for pixel
            # fonts it is the more faithful resampler: dropping whole rows and
            # columns leaves surviving strokes crisp instead of making every
            # stroke uniformly soft. It is the WRONG choice for photographic
            # content and, less obviously, for a dithered halftone — which is
            # also two-valued but encodes tone as alternating pixels. See
            # :func:`_is_line_art` for why the predicate is narrower than
            # "two distinct values" (review round 3, F11).
            resample = image_module.NEAREST if _is_line_art(image) else image_module.LANCZOS
            image = image.resize(size, resample)

        # Palette and high-bit-depth modes are legal PNG but not legal JPEG,
        # and rung 3 must not be the first place a mode problem shows up.
        #
        # ``L`` is deliberately NOT widened. It is legal in both PNG and JPEG,
        # and promoting it to RGB triples the PNG for no visible gain — which is
        # not a rounding error on the images that are actually grayscale here.
        # Snapcompact renders its archive frames as ``L`` pixel-font text, and
        # widening one measured 19,073 B -> 1,055,335 B (55x) because the
        # inflated PNG blew IMAGE_MAX_BYTES and fell through to the lossy JPEG
        # rung, putting ringing artifacts on a 5x7 bitmap font chosen precisely
        # for being crisp and deterministic (review round 1, F2). Keeping the
        # mode keeps that frame on the lossless rung at a twentieth of the size.
        if image.mode not in ("L", "LA", "RGB", "RGBA"):
            image = image.convert("RGBA" if image.mode in ("PA", "P") else "RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        payload, wire_mime = buffer.getvalue(), "image/png"

        if len(payload) > IMAGE_MAX_BYTES:
            if image.mode in ("RGBA", "LA"):
                # JPEG has no alpha channel. Compositing onto white rather than
                # dropping the channel keeps a transparent-background diagram
                # legible instead of rendering it onto black. ``LA`` is included
                # because grayscale is no longer widened to RGBA above, so it can
                # now reach this rung still carrying alpha; the flat image keeps
                # the source's own channel count so a gray frame stays gray.
                flat_mode = "RGB" if image.mode == "RGBA" else "L"
                fill = (255, 255, 255) if flat_mode == "RGB" else 255
                flat = image_module.new(flat_mode, image.size, fill)
                flat.paste(image, mask=image.getchannel("A"))
                image = flat
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=IMAGE_JPEG_QUALITY)
            jpeg = buffer.getvalue()
            # Take the smaller of the two, so the lossy rung can never make the
            # result WORSE on both axes at once. PNG beats JPEG on flat
            # synthetic images, and one of those clearing the budget is
            # possible even though nothing sampled here did it.
            if len(jpeg) < len(payload):
                payload, wire_mime = jpeg, "image/jpeg"
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001 — Pillow raises OSError, SyntaxError and its own
        raise ValueError(f"could not decode the image data ({type(exc).__name__}: {exc})") from exc

    summary = f"{wire_mime}, {image.width}x{image.height}, {len(payload)} bytes"
    if image.size != (source_width, source_height) or wire_mime != info.mime_type or rotated:
        # State the source whenever what the model sees is not what is on disk.
        # Otherwise a model comparing this against `ls -l` output, or against a
        # later re-read, has no way to reconcile the two.
        #
        # ``rotated`` is its own trigger, not a redundancy. A square-ish photo
        # can come back the same SIZE and the same MIME after a transpose, so
        # the size/mime tests both go false and the clause would vanish for the
        # one case where the pixels were most rearranged (review round 2, F7).
        summary += f"; source {source_width}x{source_height} {info.mime_type}"
        if rotated:
            # Name the rotation rather than leaving a bare size disagreement the
            # model has to explain to itself — it is the only clue that the axes
            # it might derive coordinates from have been swapped.
            summary += " (EXIF-rotated)"
    return payload, wire_mime, summary


#: Memo for :func:`rebound_oversize_image`, keyed by a digest of the block's
#: base64 text. The repair runs on the RENDERED history, which is rebuilt from
#: the transcript on every single turn (and again for every token count), so an
#: oversized block would otherwise be decoded, resized and re-encoded from
#: scratch several times a turn for the rest of the session — ~315 ms of CPU per
#: 20 MP frame, on the loop, forever. The digest is the key rather than the
#: payload so the cache holds hashes and results instead of a second copy of
#: every image in the conversation.
#:
#: In-bounds blocks never reach the cache at all — they are settled by a header
#: read — so only decisions that actually cost a decode are stored, and an
#: ordinary session of legal screenshots cannot evict the one entry that is
#: expensive to recompute.
#:
#: NOT synchronized, which is safe only because rendering is loop-bound: the
#: agent loop calls ``convert_to_llm`` on the event loop, and the token-count
#: path renders on the loop before it crosses to a thread. Anything that moves
#: this walk into a worker must revisit that, though the failure mode under the
#: GIL would be duplicated work rather than a corrupt entry (dict get/set are
#: atomic).
_REBOUND_CACHE: OrderedDict[str, tuple[str, str]] = OrderedDict()

#: Cap on :data:`_REBOUND_CACHE` measured in PAYLOAD BYTES, not entries. The
#: values are full base64 images and their sizes differ by orders of magnitude,
#: so an entry count bounds the wrong quantity: 64 resized phone photos at
#: ~1.3 MB each retain ~81 MB for the life of the PROCESS, shared across every
#: session and subagent in it (review round 1, F5). 32 MB is chosen to hold a
#: realistic session's worth of repaired frames — the observed 1568x189 repair
#: is 46 KB, and even a 1.3 MB worst case leaves room for two dozen — while
#: staying an order of magnitude under what the images themselves cost in
#: context.
#:
#: Evicted OLDEST-FIRST, not cleared wholesale. Clearing looks simpler and is
#: catastrophic here, because the access pattern is a full walk of the same
#: history on every render rather than random lookups: once the working set
#: exceeds the cap, a clear-on-overflow cache is emptied part-way through each
#: walk and every later frame in that same walk misses, so the memo never warms
#: at all. Measured on Gemini-shape archive frames, whose working set is exactly
#: the "fifty-odd frames" a snapcompact archive replays — 38 frames warm-walked
#: in 3,858 ms under clear-on-overflow against 5 ms when they fit (review round
#: 2, F7). Dropping the oldest entries instead keeps the cache hot for the
#: majority of the walk and degrades smoothly.
#:
#: Python dicts preserve insertion order, so "oldest" is the head of the map and
#: no separate bookkeeping is needed. Insertion order rather than true LRU is
#: deliberate: a walk touches every entry once per render, so recency carries no
#: information a full pass does not already erase.
_REBOUND_CACHE_MAX_BYTES = 32 * 1024 * 1024


def rebound_oversize_image(data_b64: str) -> tuple[str, str] | None:
    """Re-bound an ALREADY-ENCODED image block whose long edge is too big.

    ``None`` means "leave this block exactly as it is", which is the answer for
    the overwhelming majority of blocks and must stay cheap: the dimensions come
    from :func:`~local_operator.media.sniff_image`, a header read, so an
    in-bounds block costs one base64 decode and no Pillow decode at all.

    This is the REPAIR path, and it exists because bounding on the way in
    (:func:`bound_image_for_model`) can only ever protect blocks created after
    that code shipped. History written by an older build is already on disk and
    already oversized, and a transcript is replayed verbatim on every resume —
    so a session poisoned before the fix stays poisoned after it, which is
    precisely what was observed: a 2206x266 paste from an unbounded build kept
    earning ``At least one of the image dimensions exceed max allowed size for
    many-image requests: 2000 pixels`` on every prompt, on a build whose
    composer could no longer produce such a block.

    Deliberately keyed on the DIMENSION alone, not on bytes and not on the
    provider's complaint. The many-image ceiling is the only limit that can be
    breached by a block that was legal when it was written — the conversation
    grows past twenty frames and a block that was fine for a hundred turns
    starts being refused — so the repair has to be able to run BEFORE any
    refusal has happened, which rules out driving it from the error text.

    The trigger is :data:`IMAGE_REFUSAL_MAX_EDGE` and NOT
    :data:`IMAGE_MAX_EDGE`. Repairing is a rewrite of somebody else's decision,
    so it earns its keep only on a block that would otherwise be refused; a
    block between the two numbers is one this module would not have created but
    which the provider accepts, and rewriting it would silently overrule the
    caller that chose that size (review round 1, F1).

    Size is not the only way to be refused, so :data:`IMAGE_REFUSAL_MAX_B64_BYTES`
    is checked too: a 1900x1900 noise PNG clears every pixel ceiling and still
    encodes to 14.5 MB of base64 against a 5 MB wall (review round 2, F9).

    How FAR a repaired block is shrunk then depends on what it is. Photographic
    content goes to ``IMAGE_MAX_EDGE``, where the cost argument for 1568 was
    measured. Line art goes only to the refusal ceiling, because its strokes are
    one pixel wide and every pixel removed is a stroke that may disappear — on a
    snapcompact frame that is the difference between a 4% and a 23% reduction,
    and between legible glyphs and glyphs missing strokes (review round 2, F8).
    The target is :data:`IMAGE_REPAIR_TARGET_EDGE`, which sits just under the
    refusal line rather than on it — see that constant for why landing exactly
    on a boundary inferred from an error message is not good enough here.

    A block this cannot decode is returned unchanged rather than dropped. It may
    be perfectly acceptable to the provider (a HEIF whose dimensions this cannot
    read, a host with no Pillow), and the ``is_image_rejection`` degrade in the
    session is still underneath as the net for the genuinely bad ones. Repairing
    history must never be able to destroy context on its own.
    """
    try:
        raw = base64.b64decode(data_b64, validate=True)
    except Exception:  # noqa: BLE001 — a malformed block is the degrade's problem, not ours
        return None
    info = sniff_image(raw)
    if info is None or info.width is None or info.height is None:
        # Unreadable header: cannot prove it is oversized, so do not touch it.
        return None
    oversized = max(info.width, info.height) > IMAGE_REFUSAL_MAX_EDGE
    # The byte wall is a SEPARATE refusal that dimensions cannot predict. A
    # 1900x1900 noise PNG clears every pixel ceiling and still encodes to 14.5 MB
    # of base64, which the provider refuses exactly as flatly as an oversized
    # edge (review round 2, F9). Measured on the ENCODED text, because that is
    # what is actually sent and what the wall is expressed in.
    too_heavy = len(data_b64) > IMAGE_REFUSAL_MAX_B64_BYTES
    if not oversized and not too_heavy:
        return None
    # Only oversized blocks reach the cache, so the common path never pays for
    # it. The digest covers the bytes alone, which is the whole key: the result
    # is derived from the DECODED image, and identical bytes cannot decode two
    # ways. The block's declared mime type is deliberately not consulted
    # anywhere here — ``sniff_image`` decides format by CONTENT, because a block
    # labelled ``image/png`` that is really a JPEG must be resized as what it
    # actually is.
    key = hashlib.sha256(raw).hexdigest()
    cached = _REBOUND_CACHE.get(key)
    if cached is not None:
        return cached
    # Line art is shrunk to the REFUSAL ceiling rather than to IMAGE_MAX_EDGE:
    # every pixel removed from a one-pixel stroke is a stroke that may vanish,
    # so a repair takes the smallest reduction that clears the wall instead of
    # the cheapest one. On a snapcompact frame that is a 2% downscale rather
    # than 23%, which is the difference between legible glyphs and glyphs
    # missing strokes (review round 2, F8). It is also the cheaper choice here,
    # because a bilevel image compresses to almost nothing either way.
    #
    # Photographic content keeps the 1568 default: it has no strokes to lose,
    # and it is the case the cost argument for 1568 was measured on.
    edge = IMAGE_REPAIR_TARGET_EDGE if _is_line_art_bytes(raw, info) else IMAGE_MAX_EDGE
    try:
        payload, wire_mime, _ = bound_image_for_model(raw, info, max_edge=edge)
    except ValueError:
        # Corrupt or bomb-sized. Leaving it is strictly better than dropping it:
        # the block may still be one the provider accepts, and if it is not, the
        # session's image-rejection degrade removes it on the refusal.
        return None
    result = (base64.b64encode(payload).decode("ascii"), wire_mime)
    retained = sum(len(data) for data, _ in _REBOUND_CACHE.values()) + len(result[0])
    while retained > _REBOUND_CACHE_MAX_BYTES and _REBOUND_CACHE:
        _, (evicted, _mime) = _REBOUND_CACHE.popitem(last=False)
        retained -= len(evicted)
    _REBOUND_CACHE[key] = result
    return result
