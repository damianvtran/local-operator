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
from local_operator.evaluation.protocol import ArtifactRef


class RecordFrames:
    """Reuse published adapter bytes, or publish previously inline bytes once.

    No reference is returned until its bytes have been verified. Storage errors
    propagate to RecordSink's existing incomplete-record diagnostic; a paid run
    must not die because evidence storage filled up.
    """

    def __init__(self, record_path: Path, artifact_root: Path | None = None) -> None:
        self.root = record_path.parent
        self.artifact_root = artifact_root
        self.fallback = self.root / "frames"

    def project(self, value: Any) -> Any:
        if isinstance(value, list):
            return [self.project(item) for item in value]
        if not isinstance(value, dict):
            return value
        if value.get("type") == "image" and isinstance(value.get("data"), str):
            data = base64.b64decode(value["data"], validate=True)
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
        # Tool arguments are user data, not content blocks. An argument may
        # legitimately describe an image-shaped object without being a frame.
        return {
            key: item if key in {"arguments", "args"} else self.project(item)
            for key, item in value.items()
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
