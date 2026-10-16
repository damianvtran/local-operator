"""TUI summary adapter over the neutral, body-free peer-cycle certificate.

The mobile OWNER fold imports harness.peer_cycles directly: a headless owner
must not import terminal setup merely to certify its raw quiet pair. Re-exports
keep the TUI's one admission vocabulary; the default quiet scanner is untouched.
"""
from __future__ import annotations
from collections import Counter
from collections.abc import Sequence
from local_operator.harness.peer_cycles import (
    PeerCycleRow as PeerCycleRow, PeerCycleCertificate as PeerCycleCertificate,
    PeerCycleReplay as PeerCycleReplay, PeerCycleStart as PeerCycleStart,
    MAX_PEER_CYCLE_ROWS as MAX_PEER_CYCLE_ROWS,
    MAX_PENDING_PEER_CYCLES as MAX_PENDING_PEER_CYCLES,
    begin_peer_cycle as begin_peer_cycle, close_peer_cycle as close_peer_cycle,
    peer_cycle_sources_match,
)
from local_operator.tui.quiet_groups import QuietGroup, QuietGroupSender, quiet_sender_label

# Existing callers import the neutral API through this adapter; make that
# compatibility surface explicit instead of dropping apparently unused names.
__all__ = [
    "MAX_PEER_CYCLE_ROWS", "MAX_PENDING_PEER_CYCLES",
    "PeerCycleCertificate", "PeerCycleReplay", "PeerCycleRow", "PeerCycleStart",
    "begin_peer_cycle", "close_peer_cycle", "closed_peer_cycle_candidate",
    "peer_cycle_sources_match",
]


def closed_peer_cycle_candidate(records: Sequence[PeerCycleRow], close_certificate: PeerCycleCertificate | None) -> QuietGroup | None:
    """Reuse the existing summary shape, with truthful one-receipt arithmetic."""
    if not isinstance(close_certificate, PeerCycleCertificate) or not peer_cycle_sources_match(records, close_certificate):
        return None
    counts = Counter(quiet_sender_label(record.sender) for record in records if record.kind == "peer")
    ordered = sorted(counts.items(), key=lambda item: -item[1])
    senders = [QuietGroupSender(label, count) for label, count in ordered[:2]]
    if len(ordered) > 2:
        senders.append(QuietGroupSender(f"{len(ordered) - 2} more", sum(count for _, count in ordered[2:])))
    return QuietGroup(key=f"qg:{records[0].source_id}", family="peer", count=sum(record.kind == "peer" for record in records),
        first_ts=None, last_ts=None, senders=tuple(senders), actions=sum(record.kind == "send" for record in records),
        failed=0, open=False, row_ids=close_certificate.member_ids, mixed_actions=True)
