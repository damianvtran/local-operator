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
5. Hand the loop the config-rooted credential store. Screening runs a model,
   and the model runner resolves its provider key from this store (rows,
   then process env). Starting with none held every message as
   ``provider_error`` before any screen (live E2E F4).

This entry point does NOT install services and does NOT touch the operator's
launchd/systemd/schtasks (see connector/install.py's contract).
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import logging
from pathlib import Path
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="local_operator.team_bridge.connector")
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--credential-id", type=int)
    parser.add_argument("--account-id")
    parser.add_argument("--tenant-id")
    parser.add_argument("--endpoint-id")
    parser.add_argument("--base-url")
    return parser


#: One rotating file is bounded at (BACKUP_COUNT + 1) x MAX_BYTES = 1.5 MiB.
_LOG_MAX_BYTES = 256 * 1024
_LOG_BACKUP_COUNT = 5


class _ClassNameOnlyFormatter(logging.Formatter):
    """Persist an exception as its CLASS name, never its message or traceback.

    Several connector sites log ``exc_info=True``; a traceback carries
    ``str(exception)`` (which can echo a body, a name or a token) and frame
    paths under the user's home. The records are therefore only content-free
    if this file's formatter enforces it. The record is cloned so other
    handlers (and the cached ``exc_text``) still see the full exception.
    """

    def format(self, record: logging.LogRecord) -> str:
        clone = copy.copy(record)
        clone.exc_text = None
        clone.stack_info = None
        return super().format(clone)

    def formatException(self, ei: Any) -> str:  # noqa: N802 - logging.Formatter API
        return f"exc={ei[0].__name__}" if ei and ei[0] is not None else ""


def _configure_logging(config_dir: Path) -> logging.Handler | None:
    """Send the ``local_operator.team_bridge`` loggers to a bounded private file.

    WHY: this process runs under launchd with stdout/stderr on ``/dev/null`` and
    configured no handler, so every ``logger.warning`` in the receive path went
    to Python's last-resort stderr handler and vanished. A receive outage ran
    for hours with nothing to read. The records are already content-free by
    construction at most call sites (closed codes, stage names, exception CLASS
    names), and the formatter ENFORCES it for the ones that log ``exc_info``;
    this gives them somewhere to land. Attached to the package logger rather than
    the root so no third-party record (httpx request lines carrying URLs) is
    captured. Returns the handler (for teardown/tests), or ``None`` when the
    file cannot be opened: no log is a degraded connector, a traceback at start
    is a dead one.
    """

    from local_operator.logger import DEFAULT_LOG_FORMAT, _PrivateRotatingFileHandler
    from local_operator.team_bridge.connector.install import log_path

    target = log_path(config_dir)
    try:
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        handler = _PrivateRotatingFileHandler(
            target, maxBytes=_LOG_MAX_BYTES, backupCount=_LOG_BACKUP_COUNT, encoding="utf-8"
        )
    except OSError:
        return None
    handler.setFormatter(_ClassNameOnlyFormatter(DEFAULT_LOG_FORMAT))
    package_logger = logging.getLogger("local_operator.team_bridge")
    package_logger.addHandler(handler)
    if package_logger.level == logging.NOTSET or package_logger.level > logging.INFO:
        package_logger.setLevel(logging.INFO)
    return handler


async def _main_async(args: argparse.Namespace) -> int:
    from local_operator.providers.auth_store import AuthStore
    from local_operator.team_bridge.client import (
        DEFAULT_RADIENT_API_BASE_URL,
        TeamBridgeClient,
    )
    from local_operator.team_bridge.connector import loop as _loop
    from local_operator.team_bridge.connector.lock import (
        ConnectorLock,
        ConnectorLockBusy,
    )
    from local_operator.team_bridge.store import Store

    config_dir = Path(args.config_dir)
    _configure_logging(config_dir)
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

    # F4 (live E2E): the store the screener reads its provider key from. The
    # explicit ``db_path`` is load-bearing — ``config_dir`` alone feeds only
    # the env-override tier and would leave the DATABASE at the ambient home,
    # so a desktop login under this config root would be invisible. A PLAIN
    # store, not the mesh-aware ``build_auth_store``: the same instance backs
    # the screening cycle's live-identity check, and the bridge client is
    # deliberately never served a bearer lent by a paired device (see
    # ``TeamBridgeClient._auth_store``). Values are never read or logged here.
    auth_store = AuthStore(db_path=config_dir / "auth.db", config_dir=config_dir)
    try:
        client = TeamBridgeClient(
            config_dir,
            base_url=base_url,
            credential_id=owner.credential_id,
            # The connector's own store: a private one per client leaked an
            # auth.db connection per message (see TeamBridgeClient._auth_store).
            store=auth_store,
        )

        async def access_provider():
            return await client.current_access()

        await _loop.run_forever(
            owner,
            config_dir=config_dir,
            auth_store=auth_store,
            access_provider=access_provider,
        )
    finally:
        auth_store.close()
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
