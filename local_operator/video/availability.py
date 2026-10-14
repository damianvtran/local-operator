"""Sync availability probes for the video rungs (the ``createIf`` gate's substrate).

D9's rule, inherited verbatim from the image lane: the gate answers "can this
machine reach ANY video provider", every probe is SYNC and LOCAL (sqlite rows,
the encrypted store, the process environment — no sockets), and a probe that
cannot answer reads as "not available" rather than raising, because a broken
probe must never take a session's tool surface down.

The credential classes are the image lane's own (this module delegates to
``imagegen.availability`` rather than re-deriving them, so "signed in" cannot
mean two things in one process):

- **Radient** — persisted login rows only (never ``RADIENT_API_KEY`` from the
  environment; the STT/mobile rule: a device must not offer a rung the user
  never signed in to). The video hub routes spend the same bearer the image
  route does.
- **FAL** — a stored row **or** an exported ``FAL_API_KEY`` (an exported key
  genuinely runs the call), and the same rule for the tool's call-time
  resolver.

The video endpoints on both providers are the same credentialed surface the
image rungs use, so no video-specific credential class exists to probe.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["video_provider_reachable"]


def video_provider_reachable(config_dir: Path | None = None) -> bool:
    """Whether ANY video rung has a credential on this machine.

    The ``generate_video`` builder's gate. Reads the SAME probes the image
    gate reads (Radient persisted login; FAL stored row or exported key);
    anything else — including a probe that throws — reads "not reachable".
    """
    try:
        from local_operator.imagegen import availability as image_availability

        if image_availability.radient_available(config_dir):
            return True
        if image_availability.fal_key(config_dir):
            return True
    except Exception:  # noqa: BLE001 - a probe must not take a session build down
        logger.warning("video reachability probe failed; reporting no provider")
        return False
    return False
