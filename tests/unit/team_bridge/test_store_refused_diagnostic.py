"""A ``StoreRefused`` outside the diagnostic closed set must name its store code.

The delivery loop used to log only the exception class for such a
failure (``stage=iteration code=StoreRefused``), which is why the
connector could spend hours re-sending one unrecordable batch with no
visible cause. The store code (a closed, machine-readable tag, never
the message) now rides along as ``store_code=<code>`` when it matches
the safe shape; the existing line shape is otherwise unchanged.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any

from local_operator.team_bridge.connector import loop as loop_mod
from local_operator.team_bridge.store import StoreRefused
from local_operator.team_bridge.types import OwnerSnapshot

OWNER = OwnerSnapshot(
    credential_id=1,
    account_id="acc_1",
    tenant_id="tnt_1",
    endpoint_id="bep_1",
    canonical_base_url="https://api.radienthq.com/v1",
    enrollment_generation=1,
)


def _run_forever_with_failure(tmp_path: Path, failure: Exception) -> tuple[Any, str]:
    """Drive ``run_forever`` through one failing iteration; return (result, log)."""
    stop = asyncio.Event()

    async def access_provider() -> Any:
        stop.set()
        raise failure

    async def drive() -> Any:
        return await loop_mod.run_forever(
            OWNER,
            config_dir=tmp_path,
            auth_store=None,
            access_provider=access_provider,
            owner_source=lambda: OWNER,
            stop=stop,
        )

    logger = logging.getLogger(loop_mod.__name__)
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Capture()
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    try:
        result = asyncio.run(drive())
    finally:
        logger.removeHandler(handler)
    return result, "\n".join(str(r.getMessage()) for r in records)


def test_storerefused_outside_the_closed_set_logs_its_store_code(tmp_path):
    failure = StoreRefused("ref_migration_malformed", "secret body text")
    _result, log = _run_forever_with_failure(tmp_path, failure)
    assert "stage=iteration code=StoreRefused" in log
    assert "store_code=ref_migration_malformed" in log
    assert "secret body text" not in log


def test_non_storerefused_failure_logs_no_store_code(tmp_path):
    _result, log = _run_forever_with_failure(tmp_path, RuntimeError("secret body text"))
    assert "stage=iteration code=RuntimeError" in log
    assert "store_code=" not in log
    assert "secret body text" not in log


def test_storerefused_with_unsafe_code_logs_no_store_code(tmp_path):
    failure = StoreRefused("PRIVATE CODE CANARY", "secret body text")
    _result, log = _run_forever_with_failure(tmp_path, failure)
    assert "stage=iteration code=StoreRefused" in log
    assert "store_code=" not in log
    assert "PRIVATE" not in log
    assert "secret body text" not in log


def test_storerefused_in_the_closed_set_keeps_the_closed_code(tmp_path):
    failure = StoreRefused("policy_stale", "secret body text")
    _result, log = _run_forever_with_failure(tmp_path, failure)
    assert "stage=iteration code=policy_stale" in log
    assert "store_code=" not in log
    assert "secret body text" not in log


def test_delivery_failure_logs_store_code_without_the_message(caplog):
    failure = StoreRefused("ref_migration_malformed", "secret body text")
    with caplog.at_level("WARNING"):
        loop_mod._log_delivery_failure("feed_handling", failure)
    assert "stage=feed_handling code=unexpected_failure" in caplog.text
    assert "store_code=ref_migration_malformed" in caplog.text
    assert "secret body text" not in caplog.text
    assert not any(record.exc_info for record in caplog.records)
