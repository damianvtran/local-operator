"""Rescue every episode whose descriptor is still in a rescue root.

A ``rescue.json`` under ``<root>/<episode>/`` means an episode's parent died
(SIGKILL, a sleeping laptop, a purged run directory) before its cleanup was
confirmed, and cloud resources it owns may still be billing. This module is
the one explicit caller that globs ``*/rescue.json`` --
``supervisor.load_pending_rescue`` deliberately never scans, so importing or
starting anything can never trigger a teardown by accident; only an operator
invoking the sweep (``scripts/osworld_rescue_sweep.py``) does.

For each descriptor found the sweep re-resolves the descriptor's
``secret_refs`` through the caller's ``SecretResolver`` (the descriptor on
disk never carries values), spawns the EXACT worker the descriptor pins via
``run_rescue``, reconciles every cleanup action, and unlinks the descriptor
ONLY when the aggregate reports ``complete``. A rescue that could not confirm
termination leaves the inbox entry in place for the next sweep; the inbox is
never cleared on hope.

Every failure mode is reported per descriptor rather than raised: one
unreadable or unresolvable descriptor must not stop the sweep from reclaiming
the others, and an operator reading the output needs to see every entry that
still needs attention in a single run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from local_operator.evaluation.adapters.supervisor import (
    discard_rescue,
    load_pending_rescue,
    run_rescue,
)
from local_operator.evaluation.runner.secrets import MissingSecret, SecretResolver


@dataclass(frozen=True)
class SweepEntry:
    """One descriptor's fate under a sweep.

    ``complete`` is the aggregate's own verdict (and therefore whether the
    descriptor was unlinked). ``codes`` are the receipts' evidence codes in
    plan order, which is what an operator pastes as proof of teardown.
    ``error`` names why the rescue did not run or did not finish; it carries
    a secret NAME at most, never a value.
    """

    episode_id: str
    complete: bool
    codes: tuple[str, ...]
    error: str | None = None


async def sweep_rescue_root(
    root: Path,
    resolver: SecretResolver,
    *,
    launch: Any = None,
    rescue: Any = run_rescue,
) -> tuple[SweepEntry, ...]:
    """Rescue each ``<root>/*/rescue.json``; unlink only on ``complete``.

    ``launch`` is forwarded to ``run_rescue`` when given (the supervisor's
    ``AdapterSupervisor.launch`` otherwise); ``rescue`` replaces the whole
    rescue call for tests that want to script an aggregate without a worker.
    Entries come back in sorted path order so a run is reproducible.
    """

    entries: list[SweepEntry] = []
    for path in sorted(root.glob("*/rescue.json")):
        episode_dir = path.parent
        try:
            descriptor = load_pending_rescue(episode_dir)
        except Exception as error:
            entries.append(SweepEntry(episode_dir.name, False, (), f"unreadable: {error}"))
            continue
        if descriptor is None:
            continue
        try:
            secrets = resolver.resolve([ref.name for ref in descriptor.secret_refs])
        except MissingSecret as error:
            entries.append(SweepEntry(descriptor.episode_id, False, (), str(error)))
            continue
        try:
            if launch is None:
                aggregate = await rescue(descriptor, secrets=secrets)
            else:
                aggregate = await rescue(descriptor, secrets=secrets, launch=launch)
        except Exception as error:
            entries.append(SweepEntry(descriptor.episode_id, False, (), f"rescue failed: {error}"))
            continue
        codes = tuple(receipt.evidence_code for receipt in aggregate.receipts)
        if aggregate.complete:
            discard_rescue(episode_dir)
        entries.append(SweepEntry(descriptor.episode_id, bool(aggregate.complete), codes))
    return tuple(entries)
