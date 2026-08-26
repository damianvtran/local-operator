"""Peer-to-peer session messaging primitives.

`lop send` lets one local lop session hand a message to another on the same
machine (no cmux required), riding the existing mobile control-socket +
registry substrate. This module holds the single shared constant naming the
transcript custom-message type so the pieces that must agree on it — the
session (which persists and renders it into LLM history), the mobile
projection (which folds it for the phone), the TUI (which paints the
cross-session indicator), and the peek renderer — import it from one place
rather than duplicating a magic string.

Kept in its own tiny module (not inside ``session.py``) so importers like
``mobile/projection.py`` and ``tui/app.py`` do not pull the whole heavyweight
``Session`` import graph just to compare a ``custom_type`` string.
"""

from __future__ import annotations

#: ``CustomMessage.custom_type`` of a peer (cross-session) message. It MUST be
#: added to the LLM-visible custom-type allow-list in ``session.py`` (beside
#: ``HUB_MESSAGE_TYPE``) or the human sees the transcript row but the model
#: never does. Rendered as a distinct inbound card in every surface — never as
#: the user's own turn, and never as a hub-parent message (a peer is not a
#: parent).
PEER_MESSAGE_MESSAGE_TYPE = "peer_message"
