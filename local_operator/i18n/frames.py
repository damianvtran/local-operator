"""The ``config`` frame's ONE shape, shared by every surface that carries it.

Two surfaces publish this frame today — the desktop feed
(``server/utils/desktop_feed.py``, machine-wide SSE for the app) and the relay
daemon (``mobile/daemon.py``, ``GET /api/config/events`` for the phone leg) —
and they must stay one contract: a field renamed for one must be renamed for
the other in the same edit, and a client that parses one parses both. Hence a
builder rather than two dicts that happen to agree today, and hence
``tests/unit/test_i18n_config_frame_contract.py``, which pins the shape this
module emits against the wire.

Wire contract (RFC §2.5 wave-4 extension; additions to the payload are
contract changes):

* ``changed_keys`` — the sorted flat registry-key spellings that actually
  moved (``language``, ``display.dock``, …). ``[]`` means "nothing changed",
  which is exactly the connect-time baseline frame's value: the current
  resolved language with no historical replay.
* ``language`` — the RESOLVED locale tag, the same value
  ``GET /v1/capabilities`` answers (never the raw setting: ``auto`` is never
  returned; an unshipped configured tag resolves to ``en`` today).

Deliberately ABSENT: ``source``. The relay-web client asked whether a write's
origin (local vs disk) could ride the frame so a client could skip toasting
its own writes; it cannot soundly — the frame is a FILE watch, and the
writer's process is not knowable from it. Clients that need own-write
suppression track their own write window.
"""

from __future__ import annotations

from typing import Any, Iterable

__all__ = ["config_frame_payload"]


def config_frame_payload(changed_keys: Iterable[str], language: str) -> dict[str, Any]:
    """One ``config`` frame's payload (see the module docstring).

    ``changed_keys`` is sorted here, not by callers, so the two surfaces cannot
    disagree about ordering either. ``language`` is taken as given — callers
    pass the resolver's output (``i18n.resolve.resolve_language``), which is
    the one place that decides what a locale tag means.
    """
    return {"changed_keys": sorted(changed_keys), "language": language}
