"""Regression rig for the 2026-10-26 Team Bridge RECEIVE outage.

Three independent defects stacked into one silent failure, and each has a case
here that fails on the pre-fix tree:

* a legacy Radient OAuth row (no ``api_endpoint``) failed the owner check, so
  every screen raised ``bridge_owner_mismatch``;
* every refused attempt leaked an ``auth.db`` connection until the launchd
  descriptor limit stopped the connector;
* none of it was logged.

Everything runs against a REAL ``AuthStore`` on ``tmp_path`` and the real
client/scheduler code; only the hub transport is absent because the refusal
happens before any request would be made.
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.providers.auth_store import AuthStore
from local_operator.team_bridge import scheduler as scheduler_mod
from local_operator.team_bridge.client import BridgeRefusal, TeamBridgeClient
from local_operator.team_bridge.connector import __main__ as connector_main
from local_operator.team_bridge.connector import loop as loop_mod
from local_operator.team_bridge.connector import wake as wake_mod
from local_operator.team_bridge.connector.install import log_path, render_plist
from local_operator.team_bridge.store import (
    OwnerReceipt,
    OwnerSnapshot,
    Store,
    StoreRefused,
)

DEFAULT_HUB = "https://api.radienthq.com/v1"
ACCOUNT = "acc_legacy"


def _seed_row(config_dir: Path, **overrides: Any) -> int:
    """A Radient OAuth row shaped like one signed in BEFORE 131fc1afd81."""

    store = AuthStore(db_path=config_dir / "auth.db", config_dir=config_dir)
    try:
        data: dict[str, Any] = {
            "type": "oauth",
            "access": "synthetic-access",
            "refresh": "synthetic-refresh",
            "expires": int((time.time() + 3600) * 1000),
            "account_id": ACCOUNT,
            "org_id": "tnt_1",
            "email": "q@example.test",
        }
        data.update(overrides)
        row = store.upsert_credential("radient", data)
        if "api_endpoint" not in overrides:
            # upsert may stamp defaults; the legacy shape is "key absent".
            with sqlite3.connect(config_dir / "auth.db") as conn:
                stored = json.loads(
                    conn.execute(
                        "SELECT data FROM auth_credentials WHERE id=?", (row.id,)
                    ).fetchone()[0]
                )
                stored.pop("api_endpoint", None)
                conn.execute(
                    "UPDATE auth_credentials SET data=? WHERE id=?", (json.dumps(stored), row.id)
                )
        return row.id
    finally:
        store.close()


def _owner(credential_id: int, base: str = DEFAULT_HUB, account: str = ACCOUNT) -> OwnerSnapshot:
    return OwnerSnapshot(
        credential_id=credential_id,
        account_id=account,
        tenant_id="tnt_1",
        endpoint_id="bep_1",
        canonical_base_url=base,
        enrollment_generation=1,
    ).validated()


def _enroll(config_dir: Path, owner: OwnerSnapshot) -> None:
    Store(config_dir).commit_enrollment(
        owner, OwnerReceipt("bridge_consent_v1", "ref-1"), cos_session_id="sess_cos"
    )


async def _screen_fetch(config_dir: Path, owner: OwnerSnapshot, auth: Any) -> Any:
    """One fetch, failing the way ``run_iteration`` does.

    The loop keeps the exception in a ``failures`` list and re-raises it, so the
    frame holds an exception that holds its own traceback: a reference cycle
    that refcounting alone never frees. That cycle, not the refusal itself, is
    what kept each client (and its private store) alive in the live outage.
    """

    failures: list[Exception] = []
    try:
        return await scheduler_mod.default_status_fetcher(
            config_dir=config_dir, owner=owner, auth_store=auth
        )()
    except Exception as failure:
        failures.append(failure)
    raise failures[0]


def _open_fds() -> int:
    return len(os.listdir("/dev/fd"))


# -- 1. the owner check -------------------------------------------------------


def test_legacy_row_passes_owner_check_on_default_hub(tmp_path) -> None:
    cid = _seed_row(tmp_path)
    owner = _owner(cid)
    auth = AuthStore(db_path=tmp_path / "auth.db", config_dir=tmp_path)
    try:
        # Live-identity read (the reuse-time check of the conversation/pair cycle).
        asyncio.run(scheduler_mod._verify_live_identity(owner, auth, tmp_path))
        # Status fetcher: gets past the owner check, so it reaches the (absent)
        # hub rather than raising bridge_owner_mismatch. A transport that
        # refuses proves the GET was attempted after the check.
        reached: list[str] = []

        import httpx

        def handler(request: httpx.Request) -> httpx.Response:
            reached.append(request.url.path)
            return httpx.Response(200, json={"api_version": "1"})

        async def run() -> Any:
            return await scheduler_mod.default_status_fetcher(
                config_dir=tmp_path,
                owner=owner,
                transport=httpx.MockTransport(handler),
                auth_store=auth,
            )()

        _enroll(tmp_path, owner)
        try:
            asyncio.run(run())
        except BridgeRefusal as failure:
            assert failure.code != "bridge_owner_mismatch"
        except StoreRefused:
            pass
        assert reached, "the owner check must pass so the status GET is made"
    finally:
        auth.close()


def test_legacy_row_is_refused_for_an_owner_on_another_hub(tmp_path) -> None:
    """The unstamped row is the DEFAULT hub: it never serves a different hub's owner."""

    cid = _seed_row(tmp_path)
    owner = _owner(cid, base="https://other.example.test/v1")
    auth = AuthStore(db_path=tmp_path / "auth.db", config_dir=tmp_path)
    try:
        with pytest.raises(StoreRefused):
            asyncio.run(scheduler_mod._verify_live_identity(owner, auth, tmp_path))
        access = _access(cid, api_endpoint=None)
        with pytest.raises(BridgeRefusal) as refused:
            scheduler_mod._require_access_matches_owner(access, owner)
        assert refused.value.code == "bridge_owner_mismatch"
    finally:
        auth.close()


def test_stamped_row_for_a_different_hub_is_refused() -> None:
    owner = _owner(5)
    other = _access(5, api_endpoint="https://other.example.test/v1")
    with pytest.raises(BridgeRefusal):
        scheduler_mod._require_access_matches_owner(other, owner)
    # ... and the stamped default hub, with a trailing slash, still matches.
    scheduler_mod._require_access_matches_owner(_access(5, api_endpoint=DEFAULT_HUB + "/"), owner)


def _access(cid: int, *, api_endpoint: str | None, account_id: str = ACCOUNT) -> Any:
    return SimpleNamespace(credential_id=cid, account_id=account_id, api_endpoint=api_endpoint)


def test_wrong_credential_or_account_is_still_refused() -> None:
    owner = _owner(5)
    for access in (
        _access(6, api_endpoint=None),
        _access(5, api_endpoint=None, account_id="acc_other"),
    ):
        with pytest.raises(BridgeRefusal):
            scheduler_mod._require_access_matches_owner(access, owner)


# -- 2. descriptor leak -------------------------------------------------------


def test_refused_iterations_keep_the_fd_count_flat_with_gc_disabled(tmp_path) -> None:
    """E15 repro: 10 refusals were +21 descriptors before the fix.

    The owner's account differs from the row's, so every fetch is refused by
    the owner check AFTER the client has opened its credential store — exactly
    the outage's shape, through a fresh client per call and WITHOUT the
    connector's store handed in (the worst case: the client builds its own). A
    non-default hub would not do: that is refused before any store exists.
    """

    cid = _seed_row(tmp_path)
    owner = _owner(cid, account="acc_someone_else")
    gc.collect()
    gc.disable()
    try:
        # Warm-up: the first call may legitimately open the one shared store.
        with pytest.raises(BridgeRefusal):
            asyncio.run(_screen_fetch(tmp_path, owner, None))
        before = _open_fds()
        for _ in range(10):
            with pytest.raises(BridgeRefusal):
                asyncio.run(_screen_fetch(tmp_path, owner, None))
        after = _open_fds()
    finally:
        gc.enable()
    assert after - before <= 1, f"descriptors grew by {after - before} over 10 refusals"


def test_client_without_a_store_shares_one_connection_per_root(tmp_path) -> None:
    _seed_row(tmp_path)
    first = TeamBridgeClient(tmp_path)._auth_store()
    second = TeamBridgeClient(tmp_path)._auth_store()
    assert first is second


# -- 3. observability ---------------------------------------------------------


@pytest.mark.parametrize(
    "code", ["bridge_owner_mismatch", "bridge_sign_in_required", "bridge_owner_moved"]
)
def test_local_identity_refusals_are_named_in_the_diagnostic(code, caplog) -> None:
    with caplog.at_level("WARNING"):
        loop_mod._log_delivery_failure("pair_handling", BridgeRefusal(code, "PRIVATE_CANARY"))
    assert f"stage=pair_handling code={code}" in caplog.text
    assert "PRIVATE_CANARY" not in caplog.text


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (
            BridgeRefusal("bridge_owner_mismatch", "PRIVATE_CANARY token=PRIVATE_TOKEN"),
            "stage=iteration code=bridge_owner_mismatch",
        ),
        (
            sqlite3.OperationalError("unable to open database file PRIVATE_CANARY"),
            "stage=iteration code=OperationalError",
        ),
    ],
)
def test_run_forever_logs_a_content_free_iteration_line(
    tmp_path, caplog, failure, expected
) -> None:
    async def scenario() -> None:
        stop = asyncio.Event()

        async def provider() -> Any:
            stop.set()  # one failing iteration, then the loop exits
            raise failure

        await asyncio.wait_for(
            loop_mod.run_forever(
                _owner(1),
                config_dir=tmp_path,
                auth_store=None,
                access_provider=provider,
                stop=stop,
            ),
            timeout=30,
        )

    with caplog.at_level("WARNING"):
        asyncio.run(scenario())
    assert expected in caplog.text
    assert "PRIVATE_" not in caplog.text
    assert not any(record.exc_info for record in caplog.records)


def test_connector_log_handler_writes_a_content_free_line(tmp_path) -> None:
    package_logger = logging.getLogger("local_operator.team_bridge")
    level_before = package_logger.level
    handler = connector_main._configure_logging(tmp_path)
    assert handler is not None
    try:
        loop_mod._log_delivery_failure(
            "pair_handling", BridgeRefusal("bridge_owner_mismatch", "PRIVATE_BODY_CANARY")
        )
        handler.flush()
    finally:
        package_logger.removeHandler(handler)
        package_logger.setLevel(level_before)  # _configure_logging raises it to INFO
        handler.close()
    target = log_path(tmp_path)
    text = target.read_text()
    assert "stage=pair_handling code=bridge_owner_mismatch" in text
    assert "PRIVATE_" not in text
    assert (target.stat().st_mode & 0o777) == 0o600
    # Bounded: a rotating handler with a backup count, not an unbounded file.
    assert handler.maxBytes > 0 and handler.backupCount > 0  # type: ignore[attr-defined]


def test_exc_info_records_persist_only_the_exception_class(tmp_path) -> None:
    """The ``exc_info=True`` sites must not put ``str(exc)`` or frames in the file."""

    package_logger = logging.getLogger("local_operator.team_bridge")
    level_before = package_logger.level
    handler = connector_main._configure_logging(tmp_path)
    assert handler is not None
    other = logging.getLogger("local_operator.team_bridge.connector.wake")
    try:
        try:
            raise ValueError("PRIVATE_BODY_CANARY token=PRIVATE_TOKEN_CANARY")
        except ValueError:
            other.warning("bridge: delivery failed after the read ACK", exc_info=True)
        handler.flush()
    finally:
        package_logger.removeHandler(handler)
        package_logger.setLevel(level_before)
        handler.close()
    text = log_path(tmp_path).read_text()
    assert "delivery failed after the read ACK" in text
    assert "exc=ValueError" in text
    assert "PRIVATE_" not in text
    assert "Traceback" not in text and str(Path(__file__).parent) not in text


def test_plist_gives_launchd_an_error_path_distinct_from_the_rotating_log(tmp_path) -> None:
    plan = render_plist(tmp_path)
    assert plan["StandardErrorPath"] and plan["StandardOutPath"]
    assert plan["StandardErrorPath"] != str(log_path(tmp_path))


# -- 4. typed hold ------------------------------------------------------------


def test_screening_identity_refusal_becomes_a_typed_policy_stale_hold(
    tmp_path, monkeypatch
) -> None:
    owner = _owner(1)
    _enroll(tmp_path, owner)

    async def refused(**kwargs: Any) -> Any:
        raise BridgeRefusal("bridge_owner_mismatch", "x")

    monkeypatch.setattr(scheduler_mod, "run_screening_cycle", refused)
    result = asyncio.run(
        wake_mod.screen_item(owner, {"message": {}}, config_dir=tmp_path, auth_store=None)
    )
    assert result.disposition == "hold" and result.hold_code == "policy_stale"

    async def unreachable(**kwargs: Any) -> Any:
        raise BridgeRefusal("rate_limited", "x", http_status=429)

    monkeypatch.setattr(scheduler_mod, "run_screening_cycle", unreachable)
    with pytest.raises(BridgeRefusal):
        asyncio.run(
            wake_mod.screen_item(owner, {"message": {}}, config_dir=tmp_path, auth_store=None)
        )


# -- 5. take_up ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_take_up_with_a_conversation_message_id_explains_where_to_read(
    tmp_path, monkeypatch
) -> None:
    from typing import cast

    from local_operator.team_bridge import host as host_mod
    from local_operator.team_bridge.tool import BridgeRuntime, execute_bridge

    # The feature/admission gates are not under test; open them like the sibling tool rigs.
    monkeypatch.setattr(host_mod, "owner_fence_refusal", lambda runtime: None)
    monkeypatch.setattr(host_mod, "admission_refusal", lambda runtime, *, is_read=False: None)

    class _NoHub:
        async def message(self, *args: Any, **kwargs: Any) -> Any:
            pytest.fail("a bcmsg_ id must not be read through the pair endpoint")

    runtime = BridgeRuntime(
        client=cast(TeamBridgeClient, _NoHub()),
        config_dir=tmp_path,
        tenant_id="tnt_1",
        owner=_owner(1),
        session_id="s1",
    )
    out = await execute_bridge(
        runtime, {"op": "take_up", "message_id": "bcmsg_0123456789ab", "body_sha256": "0" * 64}
    )
    assert out["ok"] is False and out["code"] == "conversation_message"
    assert "channel_history" in out["error"] and "channel_thread" in out["error"]


# -- 4b. the real spine: hold row + backoff ------------------------------------


def test_legacy_row_mismatch_writes_a_policy_stale_hold_and_backs_off(
    tmp_path, monkeypatch
) -> None:
    """Drive the REAL ``handle_message_available`` -> ``screen_item`` -> status
    fetcher -> ``_persist_hold_backoff`` chain with a legacy row whose account is
    not the enrolled owner's. Only the hub reads and the model selection are
    stubbed; the refusal happens before any status GET would be made.
    """

    from local_operator.team_bridge import pull as pull_mod
    from local_operator.team_bridge.model_boundary import DefaultSelection
    from local_operator.team_bridge.types import feed_seq

    cid = _seed_row(tmp_path, account_id="acc_not_the_owner")
    owner = _owner(cid)  # enrolled account differs from the row's
    Store(tmp_path).commit_enrollment(
        owner, OwnerReceipt("bridge_consent_v1", "ref-1"), cos_session_id="sess_cos"
    )
    (tmp_path / "config.yml").write_text(
        json.dumps({"values": {"team_bridge": {"enabled": True}}}), encoding="utf-8"
    )
    store = Store(tmp_path)
    mid = "bmsg_legacy_hold"
    fetched: list[str] = []

    async def events(owner_: Any, **kwargs: Any) -> Any:
        ref = f"message_available:message_id={mid}"
        store.record_feed_batch(owner, feed_seq(1), [ref], resync_required=False)
        return {"cursor": 1, "refs": [ref], "resync": False}

    async def message(owner_: Any, message_id: str, **kwargs: Any) -> Any:
        return SimpleNamespace(message_id=message_id)

    async def fetch_item(owner_: Any, message_id: str, **kwargs: Any) -> Any:
        fetched.append(message_id)
        return {"message": {"id": message_id, "body": {"text": "x"}}}

    monkeypatch.setattr(pull_mod, "events", events)
    monkeypatch.setattr(pull_mod, "message", message)
    monkeypatch.setattr(wake_mod, "fetch_item", fetch_item)
    monkeypatch.setattr(wake_mod, "engagement_hold", lambda designation, config_dir: "")
    monkeypatch.setattr(
        "local_operator.team_bridge.model_boundary.resolve_default_selection",
        lambda config_dir: DefaultSelection(
            provider="p", model_id="m", spec=object(), approved_identities=frozenset({"m"})
        ),
    )
    auth = AuthStore(db_path=tmp_path / "auth.db", config_dir=tmp_path)
    try:
        started = time.time()
        # Not raised: the refusal is a typed hold, not an exception for run_forever.
        summary = asyncio.run(
            loop_mod.run_iteration(
                owner,
                config_dir=tmp_path,
                access=SimpleNamespace(credential_id=cid),
                auth_store=auth,
            )
        )
    finally:
        auth.close()
    assert summary["handled"][0]["action"] == "hold"
    assert summary["handled"][0]["code"] == "policy_stale"
    hold = store.screening_hold(owner, mid)
    assert hold is not None and hold["hold_code"] == "policy_stale"
    # Backed off by the schedule's first gap, not due again on the next turn.
    assert hold["retry_at"] is not None
    assert (
        started + loop_mod.RETRY_BASE_S - 1
        <= hold["retry_at"]
        <= time.time() + loop_mod.RETRY_BASE_S
    )
    assert store.due_screening_holds(owner, now=time.time()) == []
    assert fetched == [mid]  # one fetch, then quiet until retry_at
