"""The desktop's sidebar layout preference: read it, replace it whole.

- ``GET /v1/desktop/sidebar-layout`` — the stored arrangement, or the defaults
- ``PUT /v1/desktop/sidebar-layout`` — store one arrangement (last writer wins)

**Why this is desktop-scoped, and why the backend anyway.** The terminal
app's sidebar is a session list with no destination rows to order, so there is
no second surface to share the preference with today (design §2, "The TUI
decision"). Storing it here rather than in renderer ``localStorage`` is still
right: it survives app updates, every window reads ONE value, and a future
phone or terminal surface can read the same resource without a second
preference. Nothing but the desktop calls these routes today.

**Why a whole-record PUT rather than per-destination PATCHes.** A drag changes
both regions at once — the row leaves ``visible`` and lands in ``more`` (or
the reverse) at an index the gesture chose — so a partial update would let two
requests describe a layout neither of them showed. One body, one write, one
atomic replace; the app is effectively single-window, so last writer wins is
an honest rule (design §4). There is no request journal: a retry after a
failed save is the same whole-record write, and repeating it is idempotent.

**Reads fail open, writes fail closed.** The read degrades to the defaults in
``sidebar_layout.read_layout`` (a corrupt preference must not cost the
sidebar); a failed write is a 500, because telling the user the save succeeded
and losing it is worse than an error. The refusal travels in the same
``wire.errors.desktop`` envelope every sibling route uses, so a client
classifies it by status and code rather than by prose.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from local_operator.config import ConfigManager
from local_operator.i18n.keys import wire_errors_desktop as _wd
from local_operator.server.dependencies import get_config_manager
from local_operator.server.desktop import require_desktop
from local_operator.server.models.schemas import CRUDResponse
from local_operator.server.routes.desktop_sessions import Input, reply
from local_operator.server.sidebar_layout import (
    SidebarLayout,
    read_layout,
    write_layout,
)
from local_operator.session.store_failures import display_root

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Desktop preferences"], dependencies=[Depends(require_desktop)])


def write_refusal(config_dir: Path) -> str:
    """The client's sentence for a save that did not happen.

    THE LOG RECORD IS THE DELIVERABLE, not this sentence: the route logs the
    real ``OSError`` with its traceback (``_store_refusal``'s rule in
    ``desktop_sessions``), because a store error names absolute file paths and
    a temp-file name - facts about this machine, not a user-facing message.
    What the client paints is vetted and stable, and names the ONE place the
    operator can act on, ``~``-relative like every sibling refusal
    (``display_root``, the rule ``receipts_refusal`` states).

    BOTH likely causes are named and retrying is NOT ruled out: a failed
    replace is most often a full volume (free space, then retrying IS the fix)
    or a read-only/unwritable config root (make it writable) - the sibling
    ladder splits those into two arms because it can classify the errno; this
    store does not, so one honest sentence covers both instead of borrowing
    the out-of-space arm's instruction or asserting that a retry cannot help
    when for ENOSPC it does (agent review round 2, R2-2).
    """
    where = display_root(config_dir)
    return (
        "The sidebar layout could not be saved. Free some space on the volume "
        f"holding {where} if it is full, and check that {where} is writable, "
        "then try again."
    )


class SidebarLayoutBody(Input):
    """``PUT``'s body. ``Input`` forbids extra fields, as every sibling body does.

    ``layout`` is the whole preference (design §4) and is validated by
    :class:`~local_operator.server.sidebar_layout.SidebarLayout`, so an unknown
    or duplicated destination is a 422 before any file is touched.
    """

    layout: SidebarLayout


@router.get("/v1/desktop/sidebar-layout", response_model=CRUDResponse[dict[str, Any]])
async def get_sidebar_layout(
    config_manager: ConfigManager = Depends(get_config_manager),
) -> CRUDResponse[dict[str, Any]]:
    """The stored arrangement, or the shipped one when nothing is stored.

    A read never creates the file: the defaults are not a preference anyone
    expressed, and writing them on first boot would make a later default change
    silently invisible on this machine.
    """
    return reply({"layout": read_layout(config_manager.config_dir)})


@router.put("/v1/desktop/sidebar-layout", response_model=CRUDResponse[dict[str, Any]])
async def put_sidebar_layout(
    body: SidebarLayoutBody,
    config_manager: ConfigManager = Depends(get_config_manager),
) -> CRUDResponse[dict[str, Any]]:
    """Store a layout and answer with exactly what was stored.

    The answer is the STORED record (never the request echoed back), so a
    client that sent a valid subset sees the same normalised shape on the next
    read. A ``GET`` answer may be sent back unchanged: the body accepts the
    optional ``version`` it carries and the store writes its own (agent review
    round 1, R1-3).

    An ``OSError`` is a 500: the preference was not saved, and the client must
    say so rather than pretend. The route logs the real error where it is
    still live and answers with :func:`write_refusal`'s sentence - never the
    exception text, which names this machine's absolute paths (agent review
    round 1, R1-2).
    """
    config_dir = config_manager.config_dir
    try:
        stored = write_layout(config_dir, body.layout.model_dump())
    except OSError:
        logger.exception("sidebar layout: could not write under %s", config_dir)
        raise HTTPException(
            500,
            _wd.desktop_refusal(error=write_refusal(config_dir)).envelope(
                code="sidebar_layout_write_failed"
            ),
        ) from None
    return reply({"layout": stored})
