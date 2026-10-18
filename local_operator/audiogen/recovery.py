"""Bounded, server-side repair for audio cached before a decoder was installed.

This is not a renderer codec: the waveform route calls it in a worker thread,
only after its session gate and only when the durable sidecar is absent. A
single child decoder bounds native-code work (including malformed inputs) and
successful repairs become ordinary sidecar reads on all later requests.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

from local_operator.session.attachments import AttachmentStore

logger = logging.getLogger(__name__)

# Legacy recovery must not turn repeated HTTP misses into unbounded codec work.
# Serialize child decoders across stores; cache failed attempts as well as success.
# Failures may retry after eviction/restart, allowing a repaired installation to
# recover without writing permanent negative metadata alongside user artifacts.
_RECOVERY_LOCK = threading.Lock()
MAX_RECOVERY_BYTES = 8 * 1024 * 1024
RECOVERY_TIMEOUT_SECONDS = 10


@lru_cache(maxsize=256)
def _recover(root: str, digest: str) -> None:
    store = AttachmentStore(Path(root))
    # Check size before get_bytes(): reading a legacy movie just to refuse
    # its MIME type would bypass the recovery input bound.
    try:
        if store._content_path(digest).stat().st_size > MAX_RECOVERY_BYTES:
            return
        entry = store.get_bytes(digest)
        if entry is None or not entry[1].startswith("audio/"):
            return
        raw, _mime = entry
        if len(raw) > MAX_RECOVERY_BYTES:
            return
        result = subprocess.run(
            [sys.executable, "-m", "local_operator.audiogen.recovery"],
            input=raw,
            capture_output=True,
            timeout=RECOVERY_TIMEOUT_SECONDS,
            check=True,
        )
        peaks = json.loads(result.stdout)
        if isinstance(peaks, list):
            store.put_peaks(digest, peaks)
    except (OSError, ValueError, subprocess.SubprocessError):
        # Ordinary 404/no-waveform remains honest for damaged/oversized audio,
        # missing codecs, timeout, and read-only stores. Do not fake flat peaks.
        logger.debug("Legacy waveform recovery unavailable", exc_info=True)


def waveform_payload(store: AttachmentStore, digest: str) -> dict[str, Any] | None:
    """Read peaks, repairing a small legacy audio artifact at most once per cache.

    Must run off the event loop. Reads and successful recovery stay keyed to the
    owning store; no global/default config lookup may cross session roots.
    """
    payload = store.get_peaks(digest)
    if payload is not None:
        return payload
    # A queue behind many distinct misses must not hold HTTP workers forever.
    # A busy decoder degrades this request rather than accumulating unbounded
    # lock waits; a subsequent request may find the repaired durable sidecar.
    if not _RECOVERY_LOCK.acquire(timeout=RECOVERY_TIMEOUT_SECONDS):
        return None
    try:
        # A concurrent request or the producer may have written while we waited.
        payload = store.get_peaks(digest)
        if payload is None:
            _recover(str(store.root.resolve()), digest)
            payload = store.get_peaks(digest)
        return payload
    finally:
        _RECOVERY_LOCK.release()


if __name__ == "__main__":
    from local_operator.audiogen.peaks import compute_peaks

    # Independently enforce the input bound in the child. The timeout in the
    # parent reaps it even if a native codec hangs on corrupt provider bytes.
    data = sys.stdin.buffer.read(MAX_RECOVERY_BYTES + 1)
    print(json.dumps(compute_peaks(data) if len(data) <= MAX_RECOVERY_BYTES else None))
