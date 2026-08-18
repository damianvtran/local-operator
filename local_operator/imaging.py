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
  billing decision rather than a paste, and under the Google shape they are
  2048x2046 — over the many-image ceiling. That is reachable on another
  provider, since ``session.set_model`` can swap mid-session and a
  Gemini-shaped archive then replays to whatever is current (review round 1,
  F4). The billing trade is still not re-decided here; those frames are now
  caught by the render-time repair like any other oversized block, so the
  archive is rendered for the provider it was measured against and shrunk only
  when it would actually be refused.
- ``_forward_undecoded`` cannot resize at all, because on that host there is no
  decoder. See its own docstring for what it does enforce.
"""

from __future__ import annotations

import base64
import hashlib
import io
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
IMAGE_MAX_EDGE = 1568
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


def bound_image_for_model(data: bytes, info: ImageInfo) -> tuple[bytes, str, str]:
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
    2. Resize to :data:`IMAGE_MAX_EDGE` and re-encode as PNG. Lossless, which
       is what a screenshot of 9-pixel text needs.
    3. JPEG when that PNG blows :data:`IMAGE_MAX_BYTES`, and only when it
       actually comes out smaller. PNG is a bad photographic codec and that is
       the usual case here — the sampled 1672x941 photographic PNG re-encoded
       to 1.9 MB of PNG against 271 KB of quality-85 JPEG — but it is not the
       only one, so the choice is measured rather than assumed.

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
        if (
            info.sendable
            and frames == 1
            and not rotated
            and long_edge <= IMAGE_MAX_EDGE
            and len(data) <= IMAGE_MAX_BYTES
        ):
            return data, info.mime_type, f"{info.mime_type}, {width}x{height}, {len(data)} bytes"

        if long_edge > IMAGE_MAX_EDGE:
            scale = IMAGE_MAX_EDGE / long_edge
            size = (max(1, round(width * scale)), max(1, round(height * scale)))
            image = image.resize(size, image_module.LANCZOS)

        # Palette and high-bit-depth modes are legal PNG but not legal JPEG,
        # and rung 3 must not be the first place a mode problem shows up.
        image = image.convert("RGBA" if image.mode in ("RGBA", "LA", "PA", "P") else "RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        payload, wire_mime = buffer.getvalue(), "image/png"

        if len(payload) > IMAGE_MAX_BYTES:
            if image.mode == "RGBA":
                # JPEG has no alpha channel. Compositing onto white rather than
                # dropping the channel keeps a transparent-background diagram
                # legible instead of rendering it onto black.
                flat = image_module.new("RGB", image.size, (255, 255, 255))
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
#: ``None`` results are memoized too, and that is the case that matters most for
#: cost: it is the answer for every in-bounds block, so this also buys the
#: ordinary session out of re-hashing... nothing. In-bounds blocks are settled by
#: a header read before they reach here, so only decisions that cost a decode
#: are stored.
_REBOUND_CACHE: dict[str, tuple[str, str]] = {}

#: Cap on :data:`_REBOUND_CACHE` entries. A session that legitimately holds this
#: many DISTINCT oversized images has bigger problems than a cache miss, and an
#: unbounded dict keyed by content would otherwise grow for the life of the
#: process. Cleared wholesale rather than evicted by age: the entries are
#: equally valuable, the map is tiny, and a correct-but-cold cache costs one
#: resize while a subtle LRU here would cost a reader's afternoon.
_REBOUND_CACHE_MAX = 64


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
    if max(info.width, info.height) <= IMAGE_MAX_EDGE:
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
    try:
        payload, wire_mime, _ = bound_image_for_model(raw, info)
    except ValueError:
        # Corrupt or bomb-sized. Leaving it is strictly better than dropping it:
        # the block may still be one the provider accepts, and if it is not, the
        # session's image-rejection degrade removes it on the refusal.
        return None
    result = (base64.b64encode(payload).decode("ascii"), wire_mime)
    if len(_REBOUND_CACHE) >= _REBOUND_CACHE_MAX:
        _REBOUND_CACHE.clear()
    _REBOUND_CACHE[key] = result
    return result
