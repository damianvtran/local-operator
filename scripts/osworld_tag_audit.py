#!/usr/bin/env python3
"""List every live AWS resource carrying the OSWorld adapter tag. Read-only.

This is the LEAK DETECTOR the operator relies on around a paid run. Every
instance and volume the adapter creates carries ``lop:adapter=osworld-v2``
(applied atomically in ``run_instances`` TagSpecifications), and every TTL
lease is an EventBridge schedule named ``lop-ttl-<episode_id>``. So one
tag-filtered query per resource kind is a complete inventory, and the
assertion after every episode is that it prints ``[]``.

It deliberately terminates nothing. Teardown happens only through a
descriptor-driven rescue (``scripts/osworld_rescue_sweep.py``) so every
termination has a receipt; if this audit is non-empty and no descriptor
exists, that is an operator decision, not something a script should make.

Credentials come from the operator's own AWS profile (this runs on the
controller host, not in the stripped worker). ``--region`` is REQUIRED and
passed to every client explicitly: the default profile region on the
operator's machine is not us-east-1.

Exit 0 when nothing is found, 1 otherwise, so it composes in a shell gate.

Usage:
    python scripts/osworld_tag_audit.py --region us-east-1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# The adapter is a separate distribution that the harness does not install;
# on the controller host it is imported from the source tree exactly as the
# unit tests do.
_ADAPTER_SRC = Path(__file__).resolve().parents[1] / "benchmarks" / "osworld_v2_adapter" / "src"
if str(_ADAPTER_SRC) not in sys.path:
    sys.path.insert(0, str(_ADAPTER_SRC))


def profile_clients(region: str) -> Any:
    """Clients from the operator's own profile, region pinned explicitly."""

    from boto3.session import Session  # type: ignore[import-not-found]
    from lop_osworld_v2_adapter.providers.aws import _Clients

    session = Session(region_name=region)
    return _Clients(
        ec2=session.client("ec2", region_name=region),
        scheduler=session.client("scheduler", region_name=region),
        http_get=lambda _url, _timeout: 0,
    )


def audit(region: str, *, clients: Any | None = None) -> list[dict[str, Any]]:
    from lop_osworld_v2_adapter.providers.aws import AwsProvider

    return AwsProvider.audit(clients if clients is not None else profile_clients(region))


def main(argv: list[str] | None = None, *, clients: Any | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", required=True)
    args = parser.parse_args(argv)
    found = audit(args.region, clients=clients)
    print(json.dumps(found, indent=2, sort_keys=True, default=str))
    return 0 if not found else 1


if __name__ == "__main__":
    raise SystemExit(main())
