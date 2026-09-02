#!/usr/bin/env python3
"""Rescue every episode whose descriptor is still in the rescue root.

A ``rescue.json`` under ``<rescue_root>/<episode>/`` means an episode's parent
died (SIGKILL, a sleeping laptop, a purged run directory) before its cleanup
was confirmed, and cloud resources it owns may still be billing. This is the
operator command that runs ``runner.rescue_sweep.sweep_rescue_root`` on the
controller host: it spawns the EXACT worker each descriptor pins, hands it the
descriptor's ``secret_refs`` re-resolved from the harness credential store,
reconciles every cleanup action, and unlinks the descriptor ONLY when the
aggregate reports ``complete``.

The sweep logic itself lives in ``local_operator.evaluation.runner.rescue_sweep``
so the in-process runner and this script share ONE implementation; this file
only parses arguments, opens the credential store, and prints the result.

Resolution of secrets happens HERE, in the parent, from the credential store
(``~/.local-operator/credentials.env`` by default, or ``--config-dir``). The
values travel only over the private RPC pipe on ``begin_rescue`` and are never
written to disk, the environment, or stdout.

Usage:
    python scripts/osworld_rescue_sweep.py --rescue-root <root> [--config-dir <dir>]

Exit 0 when every descriptor found was rescued to completion (including an
empty root), 1 when any entry is incomplete or errored.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from local_operator.evaluation.runner.rescue_sweep import SweepEntry, sweep_rescue_root


def credential_store_resolver(config_dir: Path | None) -> Any:
    """The credential-store-backed resolver over the operator's config dir.

    Imported lazily so the sweep's own import graph stays free of the
    application's configuration until a resolution is actually needed.
    """

    from local_operator.credentials import CredentialManager
    from local_operator.evaluation.runner.host_secrets import CredentialStoreResolver
    from local_operator.paths import config_dir as default_config_dir

    return CredentialStoreResolver(CredentialManager(config_dir or default_config_dir()))


def entry_json(entry: SweepEntry) -> dict[str, Any]:
    return {
        "episode_id": entry.episode_id,
        "complete": entry.complete,
        "codes": list(entry.codes),
        "error": entry.error,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rescue-root", required=True, type=Path)
    parser.add_argument("--config-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    entries = asyncio.run(
        sweep_rescue_root(args.rescue_root, credential_store_resolver(args.config_dir))
    )
    print(json.dumps([entry_json(entry) for entry in entries], indent=2))
    return 0 if all(entry.complete for entry in entries) else 1


if __name__ == "__main__":
    sys.exit(main())
