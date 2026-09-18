"""Server-side import path for the shared store-failure classification.

The classifier itself lives in ``local_operator/session/store_failures.py``: the
TUI's receipt store and the desktop ladder meet the same three conditions, and
the TUI must not import ``local_operator.server`` to name them, so the decision
was hoisted to the module both layers can reach without a layering inversion
(agent review round 1, R1; UX round 1, U2).

This module stays as a delegation rather than being deleted, and deliberately:
``routes/desktop_sessions.py`` and ``tests/unit/server/test_desktop_store_failures.py``
import these names from here, and a rename that touched every server caller would
be churn in the one file whose diff a reviewer reads for the wire contract. The
re-exports are explicit (not ``import *``) so a name added to the classification
is a deliberate addition here too.
"""

from __future__ import annotations

from local_operator.session.store_failures import (
    BUSY_MESSAGE,
    FULL_VOLUME_FLOOR_BYTES,
    OUT_OF_SPACE_MESSAGE,
    STORE_BUSY,
    STORE_OUT_OF_SPACE,
    STORE_UNAVAILABLE,
    UNAVAILABLE_MESSAGE,
    StoreFailure,
    display_root,
    out_of_space_message,
    sqlite_store_failure,
    store_failure,
    unavailable_message,
    volume_is_full,
)

__all__ = [
    "BUSY_MESSAGE",
    "FULL_VOLUME_FLOOR_BYTES",
    "OUT_OF_SPACE_MESSAGE",
    "STORE_BUSY",
    "STORE_OUT_OF_SPACE",
    "STORE_UNAVAILABLE",
    "UNAVAILABLE_MESSAGE",
    "StoreFailure",
    "display_root",
    "out_of_space_message",
    "sqlite_store_failure",
    "store_failure",
    "unavailable_message",
    "volume_is_full",
]
