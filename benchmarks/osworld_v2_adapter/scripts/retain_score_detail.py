"""Copy a finalized scorer's scrubbed JSON detail into a POSIX seal staging directory."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path

# Match the OSWorld scorer's MAX_SCORE_DETAIL_BYTES without importing the adapter
# (the operator's post-exit sealer can run even if the episode venv is broken).
MAX_DETAIL_BYTES = 1024 * 1024
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


@contextmanager
def _artifact_directory(root: Path, *, create: bool = False) -> Iterator[int]:
    # Open the fixed child relative to the root fd so no symlink can redirect it.
    root_fd = os.open(root, _DIRECTORY_FLAGS)
    try:
        if create:
            try:
                os.mkdir("artifacts", mode=0o700, dir_fd=root_fd)
            except FileExistsError:
                pass
        fd = os.open("artifacts", _DIRECTORY_FLAGS, dir_fd=root_fd)
        try:
            yield fd
        finally:
            os.close(fd)
    finally:
        os.close(root_fd)


def _refused(reason: str) -> dict[str, str]:
    return {"status": "refused", "reason": reason}


def retain_score_detail(outcome: object, raw_root: Path, seal_staging: Path) -> dict[str, str]:
    """Project only outcome.score.details; return a receipt without content or paths.

    The caller owns both root paths and their ancestors, supplies a finalized
    outcome from a scorer known to emit scrubbed JSON, and publishes the private
    staging directory only after this call. The raw root owns ``artifacts/``;
    it need not be the directory containing the outcome. This is not full bundle verification.
    """
    if outcome is not None and not isinstance(outcome, dict):
        return _refused("invalid_reference")
    score = outcome.get("score") if outcome is not None else None
    if score is not None and not isinstance(score, dict):
        return _refused("invalid_reference")
    ref = score.get("details") if score is not None else None
    if ref is None:
        return {"status": "missing", "reason": "reference_absent"}
    if not isinstance(ref, dict):
        return _refused("invalid_reference")
    digest, size = ref.get("sha256"), ref.get("byte_count")
    if (
        ref.get("media_type") != "application/json"
        or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        or not isinstance(size, int)
        or isinstance(size, bool)
        or not 1 <= size <= MAX_DETAIL_BYTES
    ):
        return _refused("invalid_reference")

    try:
        with _artifact_directory(raw_root) as source:
            if not stat.S_ISREG(os.stat(digest, dir_fd=source, follow_symlinks=False).st_mode):
                return _refused("unsafe_source")
            # NONBLOCK prevents a raced FIFO from trapping the sealer before fstat.
            fd = os.open(digest, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=source)
            with os.fdopen(fd, "rb") as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode):
                    return _refused("unsafe_source")
                if info.st_size != size:
                    return _refused("size_mismatch")
                data = stream.read(size + 1)
    except FileNotFoundError:
        return {"status": "missing", "reason": "artifact_absent"}
    except OSError:
        return _refused("unsafe_source")

    if len(data) != size:
        return _refused("size_mismatch")
    if hashlib.sha256(data).hexdigest() != digest:
        return _refused("digest_mismatch")

    with ExitStack() as stack:
        try:
            dest = stack.enter_context(_artifact_directory(seal_staging, create=True))
        except OSError:
            return _refused("unsafe_destination")
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
            fd = os.open(digest, flags, 0o400, dir_fd=dest)
        except FileExistsError:
            return _refused("already_exists")
        except OSError:
            return _refused("unsafe_destination")
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
        except OSError:
            # If cleanup itself fails, discard the private stage rather than publish partial bytes.
            os.unlink(digest, dir_fd=dest)
            return _refused("write_failed")
    return {"status": "retained"}


def publish_score_detail(outcome: object, raw_root: Path, sealed_dir: Path) -> None:
    """Publish a private diagnostic sidecar only after the primary seal and ledger append.

    The caller owns the sealed directory; only this publisher creates the
    ``score-detail`` child. Do not call this to backfill an existing seal.
    """
    sidecar = sealed_dir / "score-detail"
    try:
        try:
            sidecar.lstat()
        except FileNotFoundError:
            pass
        else:
            return  # Preserve any existing entry, even an empty directory or dangling link.

        # Keep the receipt and verified bytes invisible at their final path until
        # both are closed. Racing publishers cannot replace a nonempty sidecar.
        with tempfile.TemporaryDirectory(prefix=".score-detail-", dir=sealed_dir) as temporary:
            staging = Path(temporary)
            receipt = retain_score_detail(outcome, raw_root, staging)
            receipt_path = staging / "score-detail-receipt.json"
            with receipt_path.open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(receipt, sort_keys=True) + "\n")
            receipt_path.chmod(0o400)
            staging.rename(sidecar)
    except OSError:
        # Don't log exception messages: they can expose private paths.
        print("   sealed: WARNING: optional score detail sidecar I/O failed", file=sys.stderr)
