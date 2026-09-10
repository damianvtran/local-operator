"""Turn an exception into text a human can act on, never into nothing.

An exception whose ``str()`` is empty is not rare: ``TimeoutError()`` raised by
``asyncio.wait_for`` is the case that reached users, rendering as the dangling
``owner connection lost:``. Any bare ``raise SomeError`` does the same. Every
place that paints ``str(error)`` into the UI goes through here, so the failure
mode is fixed once rather than at each call site.

See docs/design-aside-deadline.md §6, slice 2.
"""

from __future__ import annotations


def error_text(error: BaseException) -> str:
    """``str(error)``, or the exception's class name when that is blank.

    The class name is chosen over a generic "something went wrong" because it
    is the only fact available that still distinguishes one failure from
    another — ``TimeoutError`` and ``PermissionError`` must not read alike.
    """
    return str(error).strip() or type(error).__name__
