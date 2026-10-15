"""``python -m local_operator.team_bridge.connector`` — the always-on process.

START-UP ORDER (each step exists because getting it wrong is a real failure):

1. Parse ``--config-dir``. No defaults that could silently target the
   operator's live config from a test: the caller names the root.
2. Read ``RADIENT_API_BASE_URL`` ONCE here — the value is restart-scoped per
   §13.3; nothing below re-reads the environment.
3. Resolve the committed owner (full namespace) and take the machine-wide
   connector lock BEFORE any network call. Two connectors on one account is
   the defect the lock exists to refuse; the loser exits quietly with its
   own message.
4. Run the loop. The access provider is the H1 client's own ``current_access``
   (the owner-pinned OAuth row — the ordinary sign-in path; never a
   model-key cascade).

This entry point does NOT install services and does NOT touch the operator's
launchd/systemd/schtasks (see connector/install.py's contract).
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="local_operator.team_bridge.connector")
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--credential-id", type=int)
    parser.add_argument("--account-id")
    parser.add_argument("--tenant-id")
    parser.add_argument("--endpoint-id")
    parser.add_argument("--base-url")
    return parser


async def _main_async(args: argparse.Namespace) -> int:
    from local_operator.team_bridge.client import (
        DEFAULT_RADIENT_API_BASE_URL,
        TeamBridgeClient,
    )
    from local_operator.team_bridge.connector.lock import (
        ConnectorLock,
        ConnectorLockBusy,
    )
    from local_operator.team_bridge.connector.loop import run_forever
    from local_operator.team_bridge.store import Store

    config_dir = Path(args.config_dir)
    # Restart-scoped transport root: read HERE, once.
    base_url = args.base_url or DEFAULT_RADIENT_API_BASE_URL

    store = Store(config_dir)
    if args.credential_id and args.account_id and args.tenant_id and args.endpoint_id:
        owner = store.resolve_owner(
            credential_id=args.credential_id,
            account_id=args.account_id,
            tenant_id=args.tenant_id,
            endpoint_id=args.endpoint_id,
            canonical_base_url=base_url,
        )
    else:
        owner = store.active_owner()
    if owner is None:
        print("no committed bridge enrollment for this config dir; nothing to poll")
        return 0

    lock = ConnectorLock(config_dir)
    try:
        lock.acquire()
    except ConnectorLockBusy as busy:
        print(str(busy))
        return 0

    try:
        client = TeamBridgeClient(
            config_dir,
            base_url=base_url,
            credential_id=owner.credential_id,
            store=None,
        )

        async def access_provider():
            return await client.current_access()

        await run_forever(
            owner,
            config_dir=config_dir,
            auth_store=None,
            access_provider=access_provider,
        )
    finally:
        lock.release()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return asyncio.run(_main_async(args))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
