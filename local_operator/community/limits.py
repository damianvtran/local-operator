"""The contract's submission limits, importable without the HTTP stack.

Split out of ``client`` for the httpx boot rule (``desktop_team_bridge``:
``lop serve`` imports every router at start-up, so a router module must not pull
httpx in; ``tests/unit/server/test_desktop_community.py`` pins it for this
family). ``member_ops`` builds its input models from these, and the client
re-checks them, so the two can never disagree.

Values: ``SubmissionForm`` (text ≤ 4000, ≤ 4 images), ``PayloadTooLarge``
(images ≤ 5 MiB), ``Reaction`` (emoji ≤ 32), ``Cursor``/``After`` (≤ 512) in
``docs/community-api.yaml``.
"""

from __future__ import annotations

MAX_TEXT_CHARS = 4000
MAX_IMAGES = 4
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_EMOJI_CHARS = 32
MAX_CURSOR_CHARS = 512

__all__ = [
    "MAX_CURSOR_CHARS",
    "MAX_EMOJI_CHARS",
    "MAX_IMAGES",
    "MAX_IMAGE_BYTES",
    "MAX_TEXT_CHARS",
]
