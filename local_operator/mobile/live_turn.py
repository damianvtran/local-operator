"""Owner-side live-turn tracking for the v4 event relay.

The durable transcript persists at turn boundaries and the event relay only
covers "from now on", so a follower attaching mid-turn has a gap: the
in-flight assistant message and any running tool cards. :class:`LiveTurnTracker`
is the owner's bounded record of exactly that gap.

It folds the SERIALIZED event dicts the relay fans out — not the pydantic
objects — and it folds them on the registrant loop, at the same point in the
relay pipeline where frames are sent. That ordering is the correctness story
for a mid-turn join: when a new event client's welcome is processed, the
tracker has folded exactly the frames every already-connected client has been
sent and nothing more, so the seed it snapshots covers the gap with no
duplicate and no hole.

Deliberately dumb and bounded: one accumulated assistant message, the open
tool calls of the current batch, and the streaming flag. It is NOT a second
transcript — everything durable is the follower's history load, and
everything after the seed is the live relay. That is why it clears wholesale
on ``agent_start``/``agent_end`` rather than being clever about partial state.

Risk pinned by test: the tracker is a second consumer of the event stream, so
if its idea of "open" drifts from the TUI's ``EventController`` the join
renders wrong. tests/unit/mobile/test_live_turn.py replays a recorded stream
and asserts a join at every index equals the continuously-fed view.
"""

from __future__ import annotations

import logging
from typing import Any

from local_operator.mobile.types import LiveTurnSeed

logger = logging.getLogger(__name__)


class LiveTurnTracker:
    """Fold serialized AgentEvent frames into a joinable turn snapshot."""

    def __init__(self) -> None:
        self._streaming = False
        self._generation = 0
        self._assistant_text = ""
        self._assistant_open = False
        self._assistant_message_id = ""
        # tool_call_id -> the ONE serialized event that seeds this call's card
        # (a start supersedes its compose: replaying the start alone creates
        # the running card with its args, which is what the owner's screen
        # shows once execution begins). Insertion order is emission order.
        self._open_tools: dict[str, dict[str, Any]] = {}

    def fold(self, data: dict[str, Any]) -> None:
        """Consume one serialized event; never raises (a fold bug must not
        take down the relay, same contract as the projection fold)."""
        try:
            self._fold(data)
        except Exception:  # noqa: BLE001
            logger.debug("live-turn fold failed", exc_info=True)

    def _fold(self, data: dict[str, Any]) -> None:
        kind = str(data.get("type", ""))
        if kind == "agent_start":
            self._streaming = True
            self._generation = int(data.get("generation") or 0)
            self._assistant_text = ""
            self._assistant_open = False
            self._assistant_message_id = ""
            self._open_tools.clear()
            return
        if kind == "agent_end":
            self._streaming = False
            self._assistant_text = ""
            self._assistant_open = False
            self._assistant_message_id = ""
            self._open_tools.clear()
            return
        if kind == "message_start":
            message = data.get("message") or {}
            if message.get("role") == "user":
                return  # user rows are relayed live and persisted; not seed state
            self._assistant_open = True
            self._assistant_text = ""
            self._assistant_message_id = str(message.get("id") or "")
            return
        if kind == "message_update":
            # The event carries the ACCUMULATED message (harness contract), so
            # reading the message text — not appending deltas — keeps a seed
            # correct even if the tracker were fed a partial stream.
            message = data.get("message") or {}
            text = message.get("text")
            if isinstance(text, str):
                self._assistant_text = text
            else:
                self._assistant_text += str(data.get("delta") or "")
            self._assistant_open = True
            if message.get("id"):
                self._assistant_message_id = str(message["id"])
            return
        if kind == "message_end":
            self._assistant_open = False
            self._assistant_text = ""
            self._assistant_message_id = ""
            return
        if kind == "tool_call_compose":
            call_id = str(data.get("tool_call_id") or "")
            if call_id and call_id not in self._open_tools:
                # First sight of the call; a later compose for the same id only
                # updates argument_bytes, which a joiner does not need history
                # of — keep the newest.
                self._open_tools[call_id] = data
            elif call_id and self._open_tools[call_id].get("type") == "tool_call_compose":
                self._open_tools[call_id] = data
            return
        if kind == "tool_execution_start":
            call_id = str(data.get("tool_call_id") or "")
            if call_id:
                self._open_tools[call_id] = data
            return
        if kind == "tool_execution_end":
            self._open_tools.pop(str(data.get("tool_call_id") or ""), None)
            return

    def seed(self) -> LiveTurnSeed:
        """Snapshot the in-flight turn for one joining event client."""
        return LiveTurnSeed(
            streaming=self._streaming,
            generation=self._generation,
            assistant_text=self._assistant_text,
            assistant_open=self._assistant_open,
            assistant_message_id=self._assistant_message_id,
            open_tools=list(self._open_tools.values()),
        )
