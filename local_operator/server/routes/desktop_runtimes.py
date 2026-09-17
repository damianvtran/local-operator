"""``GET /v1/desktop/runtimes`` — the machine's runtime roster.

The one route in this module, and it answers a single question: **which session
runtimes are alive on this machine, and can each be reached?** The composition
itself lives in :mod:`local_operator.session.runtime.roster`; this module is the
thin HTTP half — authentication, the query surface, and the response model.

**Why it is not derived from ``/v1/desktop/sessions``.** That listing is ranked,
paginated and session-shaped: it walks the session store, and it can only name a
runtime through the discovery record the runtime publishes. On the reference host
34 of 57 live runtimes had no record at all, so a client asking that endpoint could
not learn they existed — the exact gap this route closes (see the architecture
report's §6, and ``roster``'s module docstring for the four sources it composes).

**Why the work happens in a worker thread.** The composition forks ``ps`` (~140 ms
measured), forks ``lsof`` (~200 ms) and opens up to one socket per runtime. None of
that may run on the event loop: this daemon serves every other desktop request on
the same loop, and a wedged runtime that makes a connect block would turn one
roster poll into a stall for every session in the product.

**Why it is bounded, and what the bound returns.** The budget is enforced as a
deadline, not as a hope: a row whose probe did not complete inside it reports
``reachability: "unknown"``. The failure this rules out is the one an operator
would actually hit — one runtime that is alive and not answering, holding the
whole response open. A dead pid is never dialled: ``pid_alive`` (signal-0) answers
first and the row reads ``gone``.

**It is a READER, and it changes nothing.** Records and boot records are read with
``reap=False``, the process table and the socket table are external read-only
calls, and the only network traffic is one loopback connect per row. Nothing in
the harness depends on this route answering.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request

from local_operator.server.desktop import require_desktop
from local_operator.server.models.desktop_runtimes import RuntimeEntry, RuntimeRoster
from local_operator.server.models.schemas import CRUDResponse
from local_operator.server.routes.desktop_sessions import errors
from local_operator.session.runtime.roster import ROSTER_BUDGET_S

logger = logging.getLogger("local_operator.server.routes.desktop_runtimes")

router = APIRouter(tags=["Desktop runtimes"], dependencies=[Depends(require_desktop)])

#: THE CEILING ON WHAT A CALLER MAY ASK FOR. The DEFAULT is the composition's own
#: (``roster.ROSTER_BUDGET_S``), imported rather than repeated: two copies of one
#: budget would let a client that omits the parameter and a client that asks for the
#: documented default get different bounds. A caller cannot remove the bound at all
#: (the endpoint would stop being a bounded answer) and cannot usefully ask for very
#: much more: every connect is a loopback dial with a 0.25 s timeout, so a budget
#: above this only waits on a machine that is not answering at all.
RUNTIME_BUDGET_MAX_S = 10.0


def _collect(root: Path, *, probe: bool, budget_s: float) -> RuntimeRoster:
    """Compose the roster. Runs on a worker thread; see the module docstring."""
    # EVERY seam is taken off the ``roster`` module rather than imported from a
    # sibling: the sources it composes (the process table, the socket table, the file
    # namespaces) are the things a test has to be able to replace, and a name imported
    # straight into this module would be a second binding that a caller replacing the
    # composition's own would not affect.
    from local_operator.session.runtime import roster as composition

    # The fleet is read HERE and handed to the composition rather than being read
    # inside it, for one reason: whether the socket table could be read at all is a
    # property of the fleet, and a caller that has to say so in the response must not
    # read the table a second time to find out.
    fleet = composition.read_fleet(root, sockets=composition.socket_evidence())
    rows = composition.build_roster(root, probe=probe, budget_s=budget_s, fleet=fleet)
    sources: list[str] = []
    if any(row.has_record for row in rows):
        sources.append("run/mobile")
    if any(row.has_boot_record for row in rows):
        sources.append("run/host")
    if any(row.age_s is not None for row in rows):
        sources.append("process-table")
    if fleet.sockets.available:
        sources.append("socket-table")
    entries = [RuntimeEntry(**row.to_json()) for row in rows]
    return RuntimeRoster(
        runtimes=entries,
        count=len(entries),
        budget_s=budget_s,
        socket_table=fleet.sockets.available,
        sources=sources,
    )


@router.get("/v1/desktop/runtimes", response_model=CRUDResponse[RuntimeRoster])
async def list_runtimes(
    request: Request,
    probe: Annotated[bool, Query()] = True,
    budget_s: Annotated[float, Query(ge=0.1, le=RUNTIME_BUDGET_MAX_S)] = ROSTER_BUDGET_S,
):
    """Every live session runtime this machine knows about, and whether it answers.

    ``probe=false`` skips the TCP connects and reports every row that names a port
    as ``unknown`` rather than ``live``/``unreachable``. It exists because the
    connect is the only part of this route that costs a socket: a client that wants
    the INVENTORY (the ids, pids, ports and build versions) on a machine with
    dozens of runtimes can ask for it without dialling any of them.
    """
    from local_operator.server.routes.desktop_sessions import reply

    root = request.app.state.config_manager.config_dir
    async with errors():
        started = time.monotonic()
        roster = await asyncio.to_thread(_collect, root, probe=probe, budget_s=budget_s)
    logger.debug(
        "runtime roster: %d rows in %.0f ms (probe=%s, budget=%.1fs)",
        roster.count,
        (time.monotonic() - started) * 1000.0,
        probe,
        budget_s,
    )
    return reply(roster)
