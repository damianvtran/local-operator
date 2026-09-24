"""Session sync (R22): replicas of sessions this device does not own.

THIS IS THE P0 SEAM (mesh build plan §5). The sync slice fills it in: the
manifest/cursor/plan/fetch primitive, the replica store under
``<config>/network/replicas/<id>/`` and the owner's watcher thread. Fixed here:

* the two cadence settings and their defaults, declared in the ``/settings``
  registry (``settings_io``) so the registry and this reader cannot disagree;
* :func:`request_sync`'s signature;
* :func:`install`, which registers ``net_sync`` SLOW (a ``fetch`` can read a
  100 MB transcript) with the by-name refusal until the slice lands.

Stdlib only at import: the relay imports this module at construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from local_operator.network.relay import RelayServer

#: ``network.sync.debounce_s``: seconds of quiet after a transcript change before
#: the owner tells replica holders a new cut is available. Long enough that a turn
#: streaming tokens is one push rather than hundreds; short enough that a holder is
#: at most a turn behind. The final message before idle is not subject to it: the
#: runtime's record disappearing pushes immediately (build plan §1.1).
SYNC_DEBOUNCE_S = 30.0

#: ``network.sync.tick_s``: how often the owner's watcher stats the transcripts of
#: sessions that have replicas. A stat per replicated session per tick is the whole
#: cost, which is why this can be short without a filesystem watcher.
SYNC_TICK_S = 15.0

#: The owner-side deadline for one ``net_sync`` request (seconds). A ``fetch`` is
#: one ≤512 KiB chunk, so this bounds a slow disk, not a whole transcript.
SYNC_OP_DEADLINE_S = 60.0


@dataclass(frozen=True)
class SyncSettings:
    debounce_s: float = SYNC_DEBOUNCE_S
    tick_s: float = SYNC_TICK_S

    @classmethod
    def from_config(cls, root: Path | None = None) -> SyncSettings:
        """Read ``network.sync.*`` through the package's ONE config reader."""
        from local_operator.network import store

        return cls(
            debounce_s=float(
                store.read_config(("network", "sync", "debounce_s"), SYNC_DEBOUNCE_S, root)
            ),
            tick_s=float(store.read_config(("network", "sync", "tick_s"), SYNC_TICK_S, root)),
        )


def request_sync(
    session_id: str,
    *,
    owner: str | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Pull the latest cut of ``session_id`` from its owner into the local replica.

    ``owner`` names the peer (device id or name); ``None`` resolves it from the
    session's placement. Returns ``{"ok", "session_id", "bytes", "cursor", ...}``
    or the family's ``{"ok": False, "code", "message"}``. Implemented by the sync
    slice.
    """
    raise NotImplementedError("session sync lands with the mobility slice (build plan §5, M)")


def install(server: RelayServer) -> None:
    """Register ``net_sync`` on ``server``: the by-name refusal, on the slow path."""
    from local_operator.network.relay import not_implemented_peer_op

    server.register_ops(
        {"net_sync": not_implemented_peer_op("net_sync")},
        slow={"net_sync": SYNC_OP_DEADLINE_S},
    )
