"""Wire a TUI Team Bridge card to the shared card-body resolver.

WHY a separate module: ``TeamBridgeBlock`` must stay a pure widget (it is built
in tests and replay with no session at all), and the replay presenter and the
live receipt path both need the SAME rule for "can this card fetch its body".
The rule is: only a local conversation with an on-disk journal can, because the
resolver is keyed by a transcript row and reads that conversation's own
directory. A remote/peer view gets ``None`` and keeps today's card copy.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

#: The loader a card calls (lazily, on first expand) for its resolver answer.
CardBodyLoader = Callable[[], Awaitable[dict[str, Any]]]


def card_body_loader(session: Any, entry_id: str) -> CardBodyLoader | None:
    """A zero-argument coroutine factory for ``entry_id`` in ``session``, or None."""

    if not entry_id:
        return None
    transcript = getattr(session, "_transcript", None)
    directory = getattr(transcript, "directory", None)
    if directory is None:
        return None
    try:
        directory = Path(directory)
    except TypeError:
        return None
    sessions_root = directory.parent
    # ``<config>/sessions/<id>``: the config dir is the sessions root's parent,
    # the same derivation the desktop host uses (``root / "sessions"``).
    config_dir = sessions_root.parent
    session_id = directory.name

    async def load() -> dict[str, Any]:
        from local_operator.team_bridge.card_body import resolve_card_body

        return await resolve_card_body(
            config_dir, session_id, entry_id, sessions_root=sessions_root
        )

    return load
