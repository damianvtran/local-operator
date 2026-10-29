"""Lossless image projection for session-arm JSONL evidence.

Wire events still use ImageContent; only the durable record uses
``image_reference``. Old typed readers therefore refuse the new discriminator
instead of treating an absent inline payload as a blank image. Readers needing
the original wire shape must call :func:`resolve_record_images` with the record
path. Paths are relative to the record directory, so moving an entire run
(including its adapter artifacts) preserves references.
"""

from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path
from typing import Any

from local_operator.evaluation.adapters.supervisor import verify_artifact
from local_operator.evaluation.evidence.media import validate_media
from local_operator.evaluation.protocol import ArtifactRef
from local_operator.media import SUPPORTED_IMAGE_MIME_TYPES

# Projection must stay comfortably below the JSON encoder's own recursion
# ceiling. At this depth we preserve the original subtree rather than risk
# losing a serializable event just to optimize its image storage.
MAX_FRAME_PROJECTION_DEPTH = 64


class RecordFrames:
    """Reuse published adapter bytes, or publish previously inline bytes once.

    No reference is returned until its bytes have been verified. Projection is
    only an optimization: a block we cannot publish stays byte-for-byte inline,
    with a separate fallback note, so unusual media cannot erase event text.
    """

    def __init__(self, record_path: Path, artifact_root: Path | None = None) -> None:
        self.root = record_path.parent
        self.artifact_root = artifact_root
        self.fallback = self.root / "frames"
        self.fallbacks: list[dict[str, str]] = []

    def project(self, value: Any, *, path: str = "$", _depth: int = 0) -> Any:
        if not isinstance(value, (dict, list)):
            return value
        if _depth >= MAX_FRAME_PROJECTION_DEPTH:
            self.fallbacks.append(
                {
                    "path": path,
                    "reason": (
                        f"projection depth {_depth} reached limit {MAX_FRAME_PROJECTION_DEPTH}; "
                        "subtree kept inline"
                    ),
                }
            )
            return value
        if isinstance(value, list):
            return [
                self.project(item, path=f"{path}[{index}]", _depth=_depth + 1)
                for index, item in enumerate(value)
            ]
        if value.get("type") == "image":
            try:
                return self._project_image(value)
            except Exception as error:  # noqa: BLE001 - optimization must not discard evidence
                # Keep the original block EXACTLY, including odd schemas and
                # invalid bytes. The note lives outside content so text, tool
                # details and inline-only readers retain their original data.
                self.fallbacks.append({"path": path, "reason": f"{type(error).__name__}: {error}"})
                return value
        # Both spellings are tool arguments, not content blocks. Their values
        # can legitimately describe images without being captured frames.
        return {
            key: (
                item
                if key in {"arguments", "args"}
                else self.project(item, path=f"{path}.{key}", _depth=_depth + 1)
            )
            for key, item in value.items()
        }

    def _project_image(self, value: dict[str, Any]) -> dict[str, Any]:
        mime_type = value.get("mime_type")
        if not isinstance(mime_type, str) or not isinstance(value.get("data"), str):
            raise ValueError("image projection requires string mime_type and base64 data")
        if mime_type not in SUPPORTED_IMAGE_MIME_TYPES:
            raise ValueError(f"image projection does not support mime_type {mime_type[:127]!r}")
        data = base64.b64decode(value["data"], validate=True)
        # Use the same strict media contract as reference resolution, but
        # before publication so invalid frames do not create useless copies.
        validate_media(data, mime_type)
        reference = ArtifactRef(
            sha256=hashlib.sha256(data).hexdigest(),
            media_type=value["mime_type"],
            byte_count=len(data),
        )
        root = self.artifact_root
        if root is None or not (root / reference.sha256).exists():
            root = self.fallback
            root.mkdir(mode=0o700, parents=True, exist_ok=True)
            target = root / reference.sha256
            try:
                # Exclusive creation avoids rewriting the same frame on every
                # turn; verification below also catches interrupted writes.
                with target.open("xb") as output:
                    output.write(data)
            except FileExistsError:
                pass
        verify_artifact(root, reference)
        return {
            **{key: item for key, item in value.items() if key not in {"type", "data"}},
            "type": "image_reference",
            "artifact": {
                **reference.model_dump(mode="json"),
                "path": os.path.relpath(root / reference.sha256, self.root),
            },
        }


def resolve_record_images(value: Any, record_path: Path) -> Any:
    """Restore old inline shapes, loudly refusing missing or corrupted bytes.

    This also accepts legacy inline events unchanged. An event is not a model
    prompt until references have been resolved: never feed image_reference
    blocks directly to an inline-only consumer.
    """
    if isinstance(value, list):
        return [resolve_record_images(item, record_path) for item in value]
    if not isinstance(value, dict):
        return value
    if value.get("type") == "image_reference":
        artifact = value["artifact"]
        reference = ArtifactRef.model_validate(
            {key: artifact[key] for key in ("sha256", "media_type", "byte_count")}
        )
        path = record_path.parent / artifact["path"]
        if path.name != reference.sha256:
            raise ValueError("image_reference path must end in its sha256 digest")
        data = verify_artifact(path.parent, reference)
        return {
            **{key: item for key, item in value.items() if key not in {"type", "artifact"}},
            "type": "image",
            "data": base64.b64encode(data).decode("ascii"),
        }
    return {
        key: item if key in {"arguments", "args"} else resolve_record_images(item, record_path)
        for key, item in value.items()
    }
