"""Child → parent browser-surface handoff, and the fate lines that report it.

WHY THIS EXISTS (2026-11-11, session 363d303f3e48): a subagent opened a desktop
browser tab for the operator to sign in to developer.apple.com, was briefed to
keep it open, ended — and its teardown closed the tab. Its final message, and
its parent's relay of it, still said "the tab is open". It happened twice
(ui:9, ui:25). Two separate faults:

1. A child's lifetime is one task, but a tab waiting on a HUMAN is not bounded
   by it. `retain` kept the tab host-side but handed it to nobody: the record
   stayed owned by a dead child, the parent could neither drive nor close it,
   and `lop browser cleanup` refused it as "retained — the owning session must
   release it". So a retained (or approval-pending) child tab is now HANDED to
   the parent session: the parent lists it as its own, closes it, and its own
   teardown closes it (or hands it up again if the parent is a child too).
2. Nothing authoritative told the parent what happened to the tab, so the
   child's prose was all it had. The runner now appends one harness-derived
   line per surface to the child's result — ``Browser: closed ui:N`` or
   ``Browser: handed ui:N to parent <sid>`` — computed from the settle result,
   never from what the child said.

NO HOST PROTOCOL CHANGE, deliberately. Both hosts (the desktop app's
``ownership.ts`` and the extension's) fence owner calls on the scope's params
(``requester``/``owner_proof``/``owner_generation``), not on which connection
sent them — which is what lets ``lop browser cleanup`` settle a dead session's
record from another process. The parent therefore keeps the CHILD's
``BrowserResource`` and speaks with it; the host sees the same scope it always
did. The handoff skips ``owner_finish`` entirely, so the host scope never turns
terminal and stays fully drivable by its new owner.

What is handed: a surface whose record carries a ``retention`` — an explicit
``retain`` or the "pending site approval" retention ``request_access`` writes —
i.e. exactly the tabs that exist for a human. An unretained tab is closed as
before: nothing says anyone is waiting on it, and leaving it would leak a tab
per child.
"""

from __future__ import annotations

import logging
from typing import Any

from local_operator.harness.types import HandedBrowserSurface

logger = logging.getLogger(__name__)


def public_surface_label(surface_id: str) -> str:
    """``ui:9`` for ``ui:9:<capability>``: the part safe to print anywhere.

    The third segment is the drive capability the host hands only to the tab's
    owner; a fate line travels into job results, transcripts and the parent's
    model context, so it names the host and tab id and nothing more.
    """
    parts = surface_id.split(":")
    return ":".join(parts[:2]) if len(parts) >= 2 else surface_id


def surface_matches(surface_id: str, handle: str) -> bool:
    """Whether ``handle`` (full token, redacted listing token, or ``ui:N``) names it.

    A redacted token is accepted only in the exact shape the listing prints,
    ``<host>:<id>:<capability prefix>…`` with a non-empty prefix (review R3):
    treating ANY trailing ``…`` as a prefix let ``tab='…'`` (an empty prefix)
    or ``tab='u…'`` match whichever handed tab came first, so a careless close
    silently closed an arbitrary tab a human might still be using.
    """
    handle = handle.strip()
    if not handle or not surface_id:
        return False
    if handle.endswith("…"):
        stem = handle[:-1]
        parts = stem.split(":")
        if len(parts) < 3 or not all(parts[:3]):
            return False
        return surface_id.startswith(stem)
    return handle in (surface_id, public_surface_label(surface_id))


def _retention(resource: Any) -> str:
    record = getattr(resource, "record", None) or {}
    return str(record.get("retention", "") or "")


def hand_off(
    child_browser: Any, parent_browser: Any, *, child_session_id: str, parent_session_id: str
) -> list[str]:
    """Move every retained surface the child holds onto the parent. Fate lines.

    Covers the child's OWN surface and anything ITS children handed to it, so a
    tab opened two levels down climbs to the first session that outlives it.
    Returns one fate line per handed surface; the caller closes the rest.
    """
    fates: list[str] = []
    parent_sid = parent_session_id
    target = parent_sid or "the parent session"
    # Grandchildren's tabs first: they are retained by construction (only
    # retained surfaces are ever handed), so they always continue upward.
    for handed in list(getattr(child_browser, "handed", ())):
        parent_browser.handed.append(handed)
        child_browser.handed.remove(handed)
        fates.append(
            f"Browser: handed {public_surface_label(handed.surface_id)} to parent {target}"
        )
    surface = str(getattr(child_browser, "surface_id", "") or "")
    resource = getattr(child_browser, "resource", None)
    reason = _retention(resource)
    if surface and resource is not None and reason:
        try:
            # Durable note on the child's record: `lop browser tabs`/`cleanup`
            # read it (``resources.cleanup_disposition``) to say WHY a finished
            # child's record still holds a live tab and which session owns it.
            resource.record["handed_to"] = parent_sid
            resource._save()
        except (OSError, ValueError, TypeError):
            logger.warning("could not record browser handoff on the child record", exc_info=True)
        parent_browser.handed.append(
            HandedBrowserSurface(
                surface_id=surface, resource=resource, from_session=child_session_id, reason=reason
            )
        )
        child_browser.handed_to = parent_sid or "parent"
        child_browser.surface_id = ""
        fates.append(f"Browser: handed {public_surface_label(surface)} to parent {target}")
    return fates


async def close_handed(parent_browser: Any, handed: HandedBrowserSurface, *, outcome: str) -> str:
    """Close one handed surface for its new owner. Returns its fate line."""
    label = public_surface_label(handed.surface_id)
    try:
        result = await handed.resource.release_and_finish(outcome)
    except Exception as exc:  # noqa: BLE001 - teardown reports, never raises
        logger.warning("closing a handed browser surface failed", exc_info=True)
        return f"Browser: could not close {label} ({type(exc).__name__}); it may still be open"
    if result.state == "closed":
        if handed in parent_browser.handed:
            parent_browser.handed.remove(handed)
        return f"Browser: closed {label}"
    detail = f": {result.detail}" if result.detail else ""
    return f"Browser: could not close {label} ({result.state}{detail}); it may still be open"


async def close_handed_browser_surfaces(browser: Any, *, outcome: str) -> list[str]:
    """Close every surface children handed to ``browser``'s session."""
    fates: list[str] = []
    for handed in list(getattr(browser, "handed", ()) or ()):
        fates.append(await close_handed(browser, handed, outcome=outcome))
    return fates
