"""Side-effect-free projection of published frames into typed model content.

Never accepts a model-selected path or URL: FrameRefs come from the adapter's
validated observation. The supervisor's digest/size/no-follow reader remains
the single authority for bytes, including on completion-verifier handoffs.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Sequence

from local_operator.evaluation.adapters.supervisor import verify_artifact
from local_operator.evaluation.protocol import FrameRef
from local_operator.harness.types import ImageContent


def frame_images(root: Path, frames: Sequence[FrameRef]) -> list[ImageContent]:
    return [
        ImageContent(
            data=base64.b64encode(verify_artifact(root, frame.artifact)).decode("ascii"),
            mime_type=frame.artifact.media_type,
        )
        for frame in frames
    ]
