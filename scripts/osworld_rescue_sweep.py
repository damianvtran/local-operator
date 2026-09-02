#!/usr/bin/env python3
"""Rescue every episode whose descriptor is still in the rescue root.

A ``rescue.json`` under ``<rescue_root>/<episode>/`` means an episode's parent
died (SIGKILL, a sleeping laptop, a purged run directory) before its cleanup
was confirmed, and cloud resources it owns may still be billing. For each
descriptor found, this script spawns the EXACT worker the descriptor pins
(``run_rescue``), hands it the descriptor's ``secret_refs`` re-resolved from
the harness credential store, reconciles every cleanup action, and unlinks
the descriptor ONLY when the aggregate reports ``complete`` -- so a rescue
that could not confirm termination leaves the inbox entry in place for the
next sweep, never clears it on hope.

``load_pending_rescue`` deliberately never scans (supervisor.py); this script
is the one explicit caller that globs ``*/rescue.json``. It is an operator
command run on the controller host, never from inside a worker.

Resolution of secrets happens HERE, in the parent, from the credential store
(``~/.local-operator/config.env`` by default, or ``--config-dir``). The
values travel only over the private RPC pipe on ``begin_rescue`` and are never
written to disk, the environment, or stdout.

Usage:
    python scripts/osworld_rescue_sweep.py --rescue-root <root> [--config-dir <dir>]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from local_operator.evaluation.adapters.api import ResolvedSecret
from local_operator.evaluation.adapters.supervisor import (
    discard_rescue,
    load_pending_rescue,
    run_rescue,
)


class MissingSecret(KeyError):
    """A descriptor names a secret the store cannot resolve."""


def credential_store_resolver(config_dir: Path | None) -> Callable[[Sequence[str]], Any]:
    """Resolve secret names from the harness credential store.

    Imported lazily so the sweep's own import graph stays free of the
    application's configuration until a resolution is actually needed.
    """

    from local_operator.credentials import CredentialManager
    from local_operator.paths import config_dir as default_config_dir

    manager = CredentialManager(config_dir or default_config_dir())

    def resolve(names: Sequence[str]) -> tuple[ResolvedSecret, ...]:
        out: list[ResolvedSecret] = []
        for name in names:
            try:
                value = manager.get_credential(name).get_secret_value()
            except Exception as error:
                raise MissingSecret(name) from error
            if not value:
                raise MissingSecret(name)
            out.append(ResolvedSecret(name=name, value=value))
        return tuple(out)

    return resolve


@dataclass(frozen=True)
class SweepEntry:
    episode_id: str
    complete: bool
    codes: tuple[str, ...]
    error: str | None = None


async def sweep_rescue_root(
    root: Path,
    resolve: Callable[[Sequence[str]], Any],
    *,
    rescue: Any = run_rescue,
) -> tuple[SweepEntry, ...]:
    """Rescue each ``<root>/*/rescue.json``; unlink only on ``complete``."""

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
            secrets = resolve([ref.name for ref in descriptor.secret_refs])
        except MissingSecret as error:
            entries.append(
                SweepEntry(descriptor.episode_id, False, (), f"missing secret {error.args[0]}")
            )
            continue
        try:
            aggregate = await rescue(descriptor, secrets=secrets)
        except Exception as error:
            entries.append(SweepEntry(descriptor.episode_id, False, (), f"rescue failed: {error}"))
            continue
        codes = tuple(receipt.evidence_code for receipt in aggregate.receipts)
        if aggregate.complete:
            discard_rescue(episode_dir)
        entries.append(SweepEntry(descriptor.episode_id, aggregate.complete, codes))
    return tuple(entries)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rescue-root", required=True, type=Path)
    parser.add_argument("--config-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    entries = asyncio.run(
        sweep_rescue_root(args.rescue_root, credential_store_resolver(args.config_dir))
    )
    print(
        json.dumps(
            [
                {
                    "episode_id": e.episode_id,
                    "complete": e.complete,
                    "codes": list(e.codes),
                    "error": e.error,
                }
                for e in entries
            ],
            indent=2,
        )
    )
    return 0 if all(e.complete for e in entries) else 1


if __name__ == "__main__":
    sys.exit(main())
