"""ObservationBuilder: OSWorld observation -> protocol Observation.

Coordinate space is stated exactly: ``native`` is ALWAYS the VM's real screen
(1920x1080 on the V2 AMI), never inferred from the PNG header. We assert the
RAW guest PNG's dimensions agree with that native size and raise if they do
not, because a mismatch means the guest resized and every pointer coordinate
afterwards is silently wrong. That check runs on the bytes the guest handed
us, before any bounding of our own, so it keeps guarding the guest rather than
our resizer.

The frame is then bounded to the model's screen-driving edge and
``model_visible`` is the size of THOSE bytes. Two consequences worth stating
plainly:

* The published artifact IS what the model saw. The evidence bundle holds the
  model-visible frame, not the native capture, because a bundle is judged
  against what the model was actually looking at; ``native`` is the adapter's
  private fact and reaches nobody but the coordinate converter.
* Every coordinate the model emits is in ``model_visible`` space, and
  ``actions.py`` (``_native_point`` -> ``FrameGeometry.model_to_native``) is
  the single converter back to guest pixels. Nothing else in the harness
  rescales a frame or a coordinate; the host verifier enforces that by
  checking decoded pixels against ``model_visible``.

The a11y tree is deliberately NOT shipped as a frame. ``FrameRef.geometry`` is
mandatory and a geometry for an XML document is a fiction, and
``ActionBatch.validate_for`` only ever resolves ``frame_id`` for pointer
actions, so a non-visual frame is protocol noise. Screenshot-only is also
OSWorld's own headline configuration (``--observation_type screenshot``). Its
presence is recorded in metadata; shipping it is a protocol addition, not a
fake frame.

Observation identity is content-derived: ``observation_id`` is
``observation_content_id`` over the full observation with the id excluded.
Because ``sequence`` is part of that hashed content, two visually identical
screens still yield distinct observation IDs — which is exactly what
``ExecutionReceipt``'s "must advance to a distinct observation" requires.
"""

from __future__ import annotations

import hashlib
import struct
import zlib
from collections import OrderedDict
from pathlib import Path
from typing import Any

from local_operator.evaluation.adapters.api import (
    BoundFrame,
    bound_screen_frame,
    observation_content_id,
)
from local_operator.evaluation.protocol import (
    ArtifactRef,
    FrameGeometry,
    FrameRef,
    FrameSize,
    Observation,
)

# The V2 AMI's real screen. Used as native geometry for every frame and
# asserted against the decoded PNG header; a guest that resized mid-episode
# must fail loudly rather than hand the model a miscalibrated frame.
NATIVE_SCREEN = FrameSize(width=1920, height=1080)

#: How many distinct native frames keep their bounded form in the builder.
#:
#: A ``wait`` or a no-op click returns a byte-identical screenshot, and
#: re-encoding it would cost ~70ms and — far worse — risk a different artifact
#: sha for the same screen, which is what ``_frames_identical`` reads to tell
#: the model "the screen did not change". Memoising on the native sha makes
#: identical native bytes yield an identical artifact by construction.
#:
#: Small because the value is only ever hit by an immediate repeat: an episode
#: revisiting a screen from twenty steps ago is not the case this serves, and
#: each entry holds a frame's worth of bytes.
_BOUND_CACHE_SIZE = 8


class ObservationError(ValueError):
    """A frame could not be produced that the verifier will accept."""


#: The fixed text every frameless observation fails with.
#:
#: Load-bearing as an EXACT string, not as a log line: it is what the harness
#: journals (``observation-phase-retry``), what the docs name, and what every
#: test that predates a provider-supplied cause matches on. A provider's cause
#: therefore never rewrites it -- see ``ObservationCauseError``.
NO_FRAME_MESSAGE = "environment returned no screenshot frame"

#: Optional key in the raw observation dict: the PROVIDER's own bounded account
#: of why it could not capture a frame.
#:
#: It rides the raw dict, and stops there, for two reasons. It must not reach
#: ``Observation.metadata``: that mapping feeds ``observation_content_id``, and
#: a content-addressed observation id has to be a function of what the model
#: saw rather than of the guest's health (``docs/benchmarks/osworld_2/
#: README.md``). And it is not a second exception type the harness would have to
#: learn: the builder turns it into the CAUSE of the error it already raises.
SCREENSHOT_CAUSE_KEY = "screenshot_unavailable_cause"


class ObservationCauseError(RuntimeError):
    """The provider's bounded account of a failed capture, as an exception CAUSE.

    Chained as ``raise ObservationError(NO_FRAME_MESSAGE) from
    ObservationCauseError(cause)`` rather than folded into the message, because
    the two travel by different routes and only one of them is bounded:

    * the MESSAGE is the fixed string above, and it is the ``message`` field of
      the ``RpcErrorDetail`` the worker builds. Renaming it would break the
      harness's own evidence vocabulary and every reader grepping a bundle.
    * the CAUSE is variable text, so it rides the cause chain into
      ``RpcErrorDetail.causes`` -- a field the worker bounds AND canary-checks
      per entry (``worker._error_detail``), which is exactly the discipline a
      variable string needs before it crosses the boundary.

    The provider is the honest source of this fact (it is the code that spoke
    to the guest), and it states it in a TEXT rather than a structured shape
    because the harness has no field for benchmark-specific capacity facts and
    must not grow one: ``docs/benchmarks/osworld_2/README.md`` explains why
    environment capacity must not become part of an observation's identity.
    """


def _png_dimensions(data: bytes) -> tuple[int, int]:
    """Read IHDR width/height from a PNG without decoding pixels.

    IHDR is mandated to be the first chunk, so width/height sit at fixed
    offsets. We read them to ASSERT they equal the native screen — not to
    derive geometry from the image, which would invert the trust direction.
    """

    if len(data) < 24 or not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ObservationError("screenshot is not a PNG")
    if data[12:16] != b"IHDR":
        raise ObservationError("PNG IHDR is not the first chunk")
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def write_png_rgb(width: int, height: int, rgb: bytes) -> bytes:
    """Encode a raw RGB buffer to a minimal valid PNG, stdlib only.

    Used by ``FakeProvider`` to synthesise frames and by tests to build
    fixtures, so the harness's ``validate_media`` accepts them without any
    third-party imaging dependency in the adapter wheel. Each scanline carries
    a filter-type-0 byte; the whole pixel array is zlib-compressed in one
    IDAT. This is the encoder the harness's dev suite verifies against Pillow.
    """

    if len(rgb) != width * height * 3:
        raise ValueError("rgb buffer size does not match dimensions")

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    raw = b"".join(b"\x00" + rgb[row * width * 3 : (row + 1) * width * 3] for row in range(height))
    idat = zlib.compress(raw, level=6)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


class ObservationBuilder:
    """Builds content-addressed observations against one episode's artifact root."""

    def __init__(self, artifact_root: Path) -> None:
        self._artifact_root = artifact_root
        self._bound_cache: OrderedDict[str, BoundFrame] = OrderedDict()

    def _bound_for(self, native_sha: str, png: bytes) -> BoundFrame:
        """The bounded form of ``png``, reusing the last few results.

        Keyed on the NATIVE sha so identical guest bytes always produce the
        identical artifact address — see ``_BOUND_CACHE_SIZE``.
        """

        cached = self._bound_cache.get(native_sha)
        if cached is not None:
            self._bound_cache.move_to_end(native_sha)
            return cached
        bound = bound_screen_frame(png)
        self._bound_cache[native_sha] = bound
        while len(self._bound_cache) > _BOUND_CACHE_SIZE:
            self._bound_cache.popitem(last=False)
        return bound

    def build(
        self,
        raw: dict[str, Any],
        *,
        task_id: str,
        episode_id: str,
        sequence: int,
    ) -> Observation:
        """Build one Observation from OSWorld's raw ``_get_obs()`` dict.

        ``raw`` is OSWorld's shape: ``{"screenshot": bytes|None,
        "accessibility_tree": str|None, "terminal": str|None,
        "instruction": str}``, plus the optional provider cause under
        ``SCREENSHOT_CAUSE_KEY``. A frameless observation is useless to a
        computer-use model, so a missing screenshot raises rather than
        producing an observation with no frames.
        """

        png = raw.get("screenshot")
        if png is None:
            cause = raw.get(SCREENSHOT_CAUSE_KEY)
            if cause:
                # A provider that knows WHY the frame is missing says so as the
                # error's cause, leaving the message -- and so the whole
                # downstream diagnostic -- byte-identical to the pre-cause case.
                raise ObservationError(NO_FRAME_MESSAGE) from ObservationCauseError(str(cause))
            raise ObservationError(NO_FRAME_MESSAGE)
        width, height = _png_dimensions(png)
        if (width, height) != (NATIVE_SCREEN.width, NATIVE_SCREEN.height):
            raise ObservationError(
                f"guest frame is {width}x{height}, expected the native "
                f"{NATIVE_SCREEN.width}x{NATIVE_SCREEN.height}: the guest "
                "resized, and every pointer coordinate would be wrong"
            )

        # Bound to the model's screen edge BEFORE addressing the artifact: the
        # published frame is the one the model sees, so the content address,
        # the byte count and the media type all describe those bytes. A
        # ValueError from the bound (undecodable frame, or an adapter
        # environment with no decoder) becomes an ObservationError, which is
        # the failure the episode already knows how to report.
        native_sha = hashlib.sha256(png).hexdigest()
        try:
            bound = self._bound_for(native_sha, png)
        except ValueError as error:
            raise ObservationError(f"screenshot could not be bounded for the model: {error}")

        sha = hashlib.sha256(bound.payload).hexdigest()
        # The parent reads the artifact back by opening <root>/<sha256> with
        # O_NOFOLLOW and re-hashing, so the file name IS the content address
        # and there is no extension to disagree over.
        artifact_path = self._artifact_root / sha
        if not artifact_path.exists():
            artifact_path.write_bytes(bound.payload)

        frame = FrameRef(
            frame_id="screen",
            artifact=ArtifactRef(
                sha256=sha,
                media_type=bound.media_type,
                byte_count=len(bound.payload),
            ),
            geometry=FrameGeometry(
                native=NATIVE_SCREEN,
                # Sniffed from the bounded bytes, never computed from the edge:
                # the host verifier checks decoded pixels against this field,
                # so it has to be what the file actually is.
                model_visible=bound.model_visible,
            ),
        )

        # The instruction is the observation text on sequence 0 (the reset
        # frame); later frames leave text None because the instruction does not
        # change and repeating it would only pad every transcript entry.
        text = raw.get("instruction") if sequence == 0 else None
        if text is not None and not str(text).strip():
            text = None

        # Metadata is a portable JSON subset that EXCLUDES floats. Record only
        # booleans and ints about the observation's provenance; never a score,
        # a timing float, or a coordinate.
        metadata = {
            "a11y_available": raw.get("accessibility_tree") is not None,
            "terminal_available": raw.get("terminal") is not None,
            "step": sequence,
            "vm_platform": "linux",
            "screen_width": width,
            "screen_height": height,
        }

        provisional = Observation(
            task_id=task_id,
            episode_id=episode_id,
            sequence=sequence,
            observation_id="provisional",
            text=text,
            frames=(frame,),
            metadata=metadata,
        )
        return provisional.model_copy(
            update={"observation_id": observation_content_id(provisional)}
        )
