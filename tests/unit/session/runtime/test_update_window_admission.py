"""The red-then-green CELL for the queued admission (the 2026-09-19 incident).

This file is deliberately the smallest thing that is **red on a tree without the
update window and green on one with it**, and it uses nothing a tree without the
window does not already have: the production ``prompt``, the production
``begin_retire`` latch, and the drain's own spool receipt, which shipped with the
drain. The only thing it sets that an older tree ignores is the window string
``_updating`` — on that tree the assignment is a plain attribute, nothing reads
it, and the admission refuses, which is exactly the incident.

WHY IT IS SEPARATE from ``test_update_window.py``: that file imports the new
vocabulary, so on the older tree it fails at COLLECTION — an ImportError proves
nothing about behaviour (agent review round 1, R4 on the sibling drain bound).
Everything asserted here has to be expressible on both trees.

The incident, verbatim from the live fleet: a session whose runtime was still on
0.59.9 said "it will switch to the new version when it is next idle" and then
answered the operator's next message with "This session is leaving; it will not
start a new turn. Your message is back in the composer — send it again once the
session is running again." The runtime was IDLE, so the refusal protected
nothing, and the operator had to retype the message.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.session.errors import RuntimeRetiring
from local_operator.session.runtime.inbox import (
    SOURCE_USER,
    SPOOL_RECEIPT_PROMPT,
    peek_inbox,
)
from tests.unit.session.runtime.test_serving_drain import _prompt_host


@pytest.mark.asyncio
async def test_an_idle_handover_queues_the_owners_message_instead_of_refusing_it(
    tmp_path: Path,
) -> None:
    """The window is what the idle rung opens; the latch alone is the old refusal."""
    host, session = _prompt_host(tmp_path, busy=False)
    assert host.begin_retire("runtime-retired") is True, "the idle handover takes the latch"
    # What the update window publishes before that latch, and what the admission
    # keys on. An older tree has no reader for it, so this line is inert there.
    host._updating = "0.59.9 → 0.59.11@ead71b673a9a"

    try:
        receipt = await host.prompt("now summarise the build staleness fix", command_id="p" * 8)
    except RuntimeRetiring as refusal:  # the shipped behaviour, and the incident
        raise AssertionError(
            "an idle handover refused the owner's message instead of queueing it: " f"{refusal}"
        ) from refusal

    assert receipt == SPOOL_RECEIPT_PROMPT, receipt
    assert session.prompt_calls == [], "a turn was started on a runtime that is leaving"
    rows = peek_inbox(session.transcript.directory)
    assert len(rows) == 1, rows
    assert rows[0].source == SOURCE_USER, "the successor must run it as the owner's own prompt"
    assert rows[0].wake is True, "a user prompt asks for a turn"
