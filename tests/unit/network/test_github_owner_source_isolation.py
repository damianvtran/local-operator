"""The gh source must not queue behind unrelated default-executor work.

Only synthetic login material and the existing loopback owner fixture are used.
A deliberately occupied executor proves the failure domain, not the historical
occupant of the operator's keychain. All workers are released before teardown.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

from local_operator.network.credentials import github as github_mod
from local_operator.network.credentials import owner as owner_mod
from tests.unit.network import test_github_designation_refresh as relay_fixtures
from tests.unit.network import test_github_source_ladder as source_fixtures
from tests.unit.network.test_github_designation_refresh import _lop_network_json
from tests.unit.network.test_github_source_ladder import _ask, _owner_for
from tests.unit.network.test_relay_e2e import devices  # noqa: F401 — pytest fixture

# Reuse the existing pytest fixture objects, not a second owner/relay rig.
env = source_fixtures.env
github_api = source_fixtures.github_api
app_keypair = source_fixtures.app_keypair
github_mesh = relay_fixtures.github_mesh


def test_occupied_default_workers_do_not_starve_a_github_grant(
    env: Any, github_api: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _owner_for(env, github_api, github_mod.SOURCE_GH, monkeypatch=monkeypatch) as rig:
        release = threading.Event()
        entered = [threading.Event(), threading.Event()]

        def unrelated(index: int) -> None:
            entered[index].set()
            assert release.wait(5), "the test must release each owned worker"

        async def occupy() -> list[Any]:
            loop = asyncio.get_running_loop()
            return [loop.run_in_executor(None, unrelated, i) for i in range(2)]

        futures = rig.broker._loop.submit(occupy(), timeout=2)
        assert all(event.wait(2) for event in entered)
        # Scale only the handler budget; the gh stub and actual owner grant path
        # stay real. Main waits behind these workers and answers rate_limited.
        rig.broker._op_deadline_s = owner_mod.HANDLER_WAIT_MARGIN_S + 1
        try:
            detail = _ask(rig)
            print("SOURCE_ISOLATION_RESULT", detail.get("kind"), detail.get("code"))
            assert detail.get("kind") == "grant", detail.get("code")
        finally:
            release.set()

            async def settled() -> None:
                await asyncio.gather(*futures)

            rig.broker._loop.submit(settled(), timeout=2)


def test_owner_reads_once_per_short_cache_window_and_after_invalidation(
    env: Any, github_api: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.unit.network.test_credentials_github import BORROWER_DEVICE
    from tests.unit.network.test_github_source_ladder import GHO

    with _owner_for(env, github_api, github_mod.SOURCE_GH, monkeypatch=monkeypatch) as rig:
        clock = [100.0]
        rig.broker._github_source._clock = lambda: clock[0]
        calls = []
        monkeypatch.setattr(github_mod, "read_gh_token", lambda: calls.append(1) or GHO)
        for _ in range(3):
            assert _ask(rig)["kind"] == "grant"
        assert len(calls) == 1
        clock[0] += 31
        assert _ask(rig)["kind"] == "grant"
        assert len(calls) == 2
        rig.broker.revoke_outstanding("github", BORROWER_DEVICE)
        assert _ask(rig)["kind"] == "grant"
        assert len(calls) == 3
        rig.broker.revoke_all_outstanding(cause="panic")
        assert _ask(rig)["kind"] == "grant"
        assert len(calls) == 4


def test_whole_read_timeout_is_single_flight_and_never_starts_replacement_workers(
    env: Any, github_api: Any, monkeypatch: pytest.MonkeyPatch, caplog: Any
) -> None:
    from local_operator.network.credentials import github_source
    from tests.unit.network.test_github_source_ladder import GHO

    with _owner_for(env, github_api, github_mod.SOURCE_GH, monkeypatch=monkeypatch) as rig:
        release = threading.Event()
        calls = []

        def stuck() -> str:
            calls.append(threading.get_ident())
            assert release.wait(3)
            return GHO

        monkeypatch.setattr(github_mod, "read_gh_token", stuck)
        monkeypatch.setattr(github_source, "SOURCE_TIMEOUT_S", 0.04)
        try:
            for _ in range(3):
                detail = _ask(rig)
                assert detail["kind"] == "error" and detail["code"] == "owner_busy", detail.get(
                    "code"
                )
            assert len(calls) == 1
            # Revoke while an abandoned read is still running. Its result must
            # not warm the next epoch's cache or be returned to a new caller.
            rig.broker.revoke_all_outstanding(cause="panic")
        finally:
            release.set()
            rig.broker._github_source._thread.join(2)
        assert rig.broker._github_source._token == ""
        assert not rig.broker._github_source._thread.is_alive()


def test_concurrent_different_sessions_join_the_same_source_read(
    env: Any, github_api: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from tests.unit.network.test_github_source_ladder import GHO

    with _owner_for(env, github_api, github_mod.SOURCE_GH, monkeypatch=monkeypatch) as rig:
        started = threading.Event()
        release = threading.Event()
        calls = []

        def read() -> str:
            calls.append(1)
            started.set()
            assert release.wait(3)
            return GHO

        monkeypatch.setattr(github_mod, "read_gh_token", read)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(_ask, rig, for_session=f"synthetic-{i}") for i in range(4)]
            try:
                assert started.wait(2)
            finally:
                release.set()
            assert all(f.result(3)["kind"] == "grant" for f in futures)
        assert len(calls) == 1


def test_a_real_gh_subprocess_timeout_keeps_its_existing_unusable_code(
    env: Any, github_api: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    from tests.unit.network.test_github_source_ladder import _install_stub_gh

    with _owner_for(env, github_api, github_mod.SOURCE_GH, monkeypatch=monkeypatch) as rig:
        stub = _install_stub_gh(env.home)
        stub.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(1)\n")
        monkeypatch.setattr(github_mod, "GH_CLI_TIMEOUT_S", 0.03)
        detail = _ask(rig)
        assert detail["code"] == github_mod.CODE_GH_UNUSABLE, detail.get("code")


@pytest.mark.parametrize("sleeping", [False, True])
def test_logout_cannot_serve_a_cached_token_after_its_ttl(
    env: Any, github_api: Any, monkeypatch: pytest.MonkeyPatch, sleeping: bool
) -> None:
    from tests.unit.network.test_github_source_ladder import GHO

    with _owner_for(env, github_api, github_mod.SOURCE_GH, monkeypatch=monkeypatch) as rig:
        clock = [0.0]
        rig.broker._github_source._clock = lambda: 0.0 if sleeping else clock[0]
        if sleeping:
            rig.broker._github_source._wall_clock = lambda: clock[0]
        monkeypatch.setattr(github_mod, "read_gh_token", lambda: GHO)
        assert _ask(rig)["kind"] == "grant"

        def logged_out() -> str:
            raise github_mod.GithubGhError("absent", "synthetic logout")

        monkeypatch.setattr(github_mod, "read_gh_token", logged_out)
        clock[0] = 31
        assert _ask(rig)["code"] == "no_local_credential"
        # Also check normal gh logout's hosts-file mutation inside the TTL.
        rig.broker._github_source.invalidate()
        monkeypatch.setattr(github_mod, "read_gh_token", lambda: GHO)
        assert _ask(rig)["kind"] == "grant"
        github_mod.gh_hosts_path().unlink()
        monkeypatch.setattr(github_mod, "read_gh_token", logged_out)
        assert _ask(rig)["kind"] != "grant"


@pytest.fixture()
def borrower(env: Any, github_api: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    from types import SimpleNamespace

    from local_operator.network import relay, store
    from local_operator.network.credentials.client import MeshCredentialClient
    from tests.unit.network.test_credentials_github import (
        BORROWER_DEVICE,
        _seed_borrower_network,
    )

    with _owner_for(env, github_api, github_mod.SOURCE_GH, monkeypatch=monkeypatch) as rig:
        _seed_borrower_network(env.root, "n_gh")
        client = MeshCredentialClient(
            root=env.root, self_device=BORROWER_DEVICE, network_id="n_gh", placement=rig.document
        )
        calls = []
        response = [None]

        def ask(_record: Any, op: str, **fields: Any) -> Any:
            assert op == "credential_grant"
            calls.append(fields)
            detail = response[0] or _ask(rig, for_session=fields.get("session_id", ""))
            return {"op": "ack", "detail": detail}

        monkeypatch.setattr(store, "find_own_relay", lambda _root: object())
        monkeypatch.setattr(relay, "control_request", ask)
        yield SimpleNamespace(rig=rig, client=client, calls=calls, response=response)


def test_current_permission_lease_survives_a_proactive_lookup_timeout(borrower: Any) -> None:
    import time

    from local_operator.network.credentials.types import Grant

    b = borrower
    original = b.client.request_grant_sync("github", session_id="synthetic")
    assert isinstance(original, Grant)
    original.grant_expires_at_ms = int(time.time() * 1000) + 60_000
    original_expiry = original.grant_expires_at_ms
    b.response[0] = {"kind": "error", "code": "owner_busy", "key": "github"}
    served = b.client.request_grant_sync("github", session_id="synthetic")
    assert served is original
    assert served.grant_expires_at_ms == original_expiry
    from local_operator.network.credentials.effective import effective_access

    assert (
        effective_access("github", root=b.rig.env.root, mesh=b.client).state == "effective_borrow"
    )
    assert len(b.calls) == 2, "one ordinary borrow and one proactive ask, not a storm"
    print("CURRENT_LEASE_RESULT", {"served": True, "extended": False, "asks": len(b.calls)})


@pytest.mark.parametrize("expired,forced", [(True, False), (False, True)])
def test_expired_or_force_refreshed_grant_is_never_reused(
    borrower: Any, expired: bool, forced: bool
) -> None:
    import time

    from local_operator.network.credentials.types import BrokerError, Grant

    b = borrower
    original = b.client.request_grant_sync("github", session_id="synthetic")
    assert isinstance(original, Grant)
    original.grant_expires_at_ms = int(time.time() * 1000) + (-1 if expired else 60_000)
    b.response[0] = {"kind": "error", "code": "owner_busy", "key": "github"}
    refused = b.client.request_grant_sync("github", session_id="synthetic", force_refresh=forced)
    assert isinstance(refused, BrokerError) and refused.code == "owner_busy"
    assert len(b.calls) == 3, "only one bounded retry"


@pytest.mark.parametrize("ending", ["revoke", "panic", "answered-deny"])
def test_revocation_and_panic_never_reuse_a_current_github_lease(
    borrower: Any, ending: str
) -> None:
    import time

    from local_operator.network import store
    from local_operator.network.credentials.types import BrokerError, Grant
    from tests.unit.network.test_credentials_github import BORROWER_DEVICE, OWNER_DEVICE

    b = borrower
    original = b.client.request_grant_sync("github", session_id="synthetic")
    assert isinstance(original, Grant)
    original.grant_expires_at_ms = int(time.time() * 1000) + 60_000
    b.response[0] = {"kind": "error", "code": "owner_busy", "key": "github"}
    if ending == "revoke":
        b.rig.document.revoke("github", BORROWER_DEVICE, by=OWNER_DEVICE)
        b.rig.document.save()
    elif ending == "panic":
        with store.mutate("n_gh", b.rig.env.root) as record:
            record.trust = "untrusted"
            store.save(record, b.rig.env.root)
    else:
        b.response[0] = {"kind": "error", "code": "not_a_holder", "key": "github"}
    answer = b.client.request_grant_sync("github", session_id="synthetic")
    assert isinstance(answer, BrokerError)
    assert answer.code == "not_a_holder"
    assert b.client.grants.peek("github", "synthetic") is None
    assert len(b.calls) <= 2, "a policy answer must not be retried"


def test_borrower_retries_only_the_new_code_and_stops_after_one_retry(
    borrower: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.network.credentials.types import BrokerError, Grant

    b = borrower
    responses = [
        BrokerError(code="owner_busy", key="github"),
        b.client.request_grant_sync("github", session_id="fixture-only"),
    ]
    assert isinstance(responses[1], Grant)
    calls = []

    def once(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return responses[min(len(calls) - 1, 1)]

    monkeypatch.setattr(b.client, "_request_grant_once", once)
    assert isinstance(b.client.request_grant_sync("github", session_id="new"), Grant)
    assert len(calls) == 2
    calls.clear()
    responses[:] = [BrokerError(code="github_gh_unusable", key="github")] * 2
    assert isinstance(b.client.request_grant_sync("github", session_id="new"), BrokerError)
    assert len(calls) == 1


# Actual argparse -> local control -> two paired TCP relays -> owner gh source.
@pytest.mark.parametrize("recover", [True, False])
def test_real_two_relay_refresh_recovers_one_source_deadline(
    github_mesh: Any, monkeypatch: pytest.MonkeyPatch, recover: bool
) -> None:
    from local_operator.network import cli
    from local_operator.network.credentials import github_source

    release = threading.Event()
    reads = []

    def read() -> str:
        reads.append(1)
        assert release.wait(3)
        return "synthetic-keychain-login"

    monkeypatch.setattr(github_mod, "resolve_source", lambda *a, **k: github_mod.SOURCE_GH)
    monkeypatch.setattr(github_mod, "read_gh_token", read)
    monkeypatch.setattr(github_source, "SOURCE_TIMEOUT_S", 0.04)
    original = cli._relay_answer
    codes = []

    def answer(*args: Any, **kwargs: Any) -> Any:
        detail = original(*args, **kwargs)
        codes.append(detail.get("code", detail.get("kind")))
        if detail.get("code") == "owner_busy" and recover:
            release.set()
        return detail

    monkeypatch.setattr(cli, "_relay_answer", answer)
    try:
        if not recover:
            import argparse
            import io
            import json
            from contextlib import redirect_stdout

            parser = argparse.ArgumentParser(prog="lop")
            cli.add_parser(parser.add_subparsers(dest="command"))
            output = io.StringIO()
            with redirect_stdout(output):
                rc = cli.main(
                    parser.parse_args(["network", "credential", "refresh", "github", "--json"])
                )
            refusal = json.loads(output.getvalue())
            assert rc == 1 and refusal["code"] == "owner_busy"
            assert "could not finish reading" in refusal["message"]
            assert "too many credential requests" not in refusal["message"]
            assert codes == ["owner_busy", "owner_busy"]
            print("REAL_REFRESH_REFUSAL", refusal)
            return
        receipt = _lop_network_json("credential", "refresh", "github")
        assert receipt["ok"] is True and "access_token" not in receipt
        assert codes == ["owner_busy", "grant"]
        assert len(reads) == 1
        print(
            "REAL_REFRESH_RESULT",
            {
                "ok": receipt["ok"],
                "codes": codes,
                "source_reads": len(reads),
                "verified_narrowing": bool(receipt.get("narrowing")),
            },
        )
    finally:
        release.set()


def test_owner_budget_timeout_names_only_default_executor_occupants(
    env: Any, github_api: Any, monkeypatch: pytest.MonkeyPatch, caplog: Any
) -> None:
    with _owner_for(env, github_api, github_mod.SOURCE_GH, monkeypatch=monkeypatch) as rig:
        release = threading.Event()
        entered = [threading.Event(), threading.Event()]

        def held_worker(index: int, private_argument: str) -> None:
            entered[index].set()
            assert release.wait(4)

        async def occupy() -> list[Any]:
            return [
                asyncio.create_task(
                    asyncio.to_thread(held_worker, i, "NEVER_LOG_THIS_SYNTHETIC_VALUE")
                )
                for i in range(2)
            ]

        def queued_resolve() -> None:
            pass

        async def resolver(**_kwargs: Any) -> Any:
            return await asyncio.get_running_loop().run_in_executor(None, queued_resolve)

        futures = rig.broker._loop.submit(occupy(), timeout=2)
        assert all(event.wait(2) for event in entered)
        monkeypatch.setattr(rig.broker, "_resolve_once", resolver)
        rig.broker._op_deadline_s = owner_mod.HANDLER_WAIT_MARGIN_S + 1
        try:
            assert _ask(rig)["code"] == "owner_busy"
            records = [
                r.message for r in caplog.records if "credential owner budget expired" in r.message
            ]
            assert len(records) == 1
            assert "held_worker" in records[0] and "queued_resolve" in records[0]
            assert (
                "queue_wait_ms" in records[0] and "NEVER_LOG_THIS_SYNTHETIC_VALUE" not in records[0]
            )
            assert len(records[0]) < 2000
        finally:
            release.set()

            async def settled() -> None:
                await asyncio.gather(*futures)

            rig.broker._loop.submit(settled(), timeout=2)


def test_owner_busy_does_not_erase_a_known_consent_denial(borrower: Any) -> None:
    import time

    from local_operator.network.credentials.types import BrokerError, Grant
    from tests.unit.network.test_credentials_github import OWNER_DEVICE

    b = borrower
    grant = b.client.request_grant_sync("github", session_id="synthetic")
    assert isinstance(grant, Grant)
    grant.grant_expires_at_ms = int(time.time() * 1000) + 60_000
    b.client.state.observe("github", "consent_withdrawn", owner_device=OWNER_DEVICE)
    b.client.state.save()
    b.response[0] = {"kind": "error", "code": "owner_busy", "key": "github"}
    assert isinstance(b.client.request_grant_sync("github", session_id="synthetic"), BrokerError)
    assert b.client.grants.peek("github", "synthetic") is None
    assert b.client.state.held_policy_refusal("github", OWNER_DEVICE) == "consent_withdrawn"


def test_a_revoke_during_the_source_read_is_not_reported_as_retryable_busy(
    env: Any, github_api: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from tests.unit.network.test_credentials_github import BORROWER_DEVICE, OWNER_DEVICE
    from tests.unit.network.test_github_source_ladder import GHO

    with _owner_for(env, github_api, github_mod.SOURCE_GH, monkeypatch=monkeypatch) as rig:
        entered = threading.Event()
        release = threading.Event()

        def read() -> str:
            entered.set()
            assert release.wait(3)
            return GHO

        monkeypatch.setattr(github_mod, "read_gh_token", read)
        with ThreadPoolExecutor(max_workers=1) as pool:
            answer = pool.submit(_ask, rig)
            try:
                assert entered.wait(2)
                rig.document.revoke("github", BORROWER_DEVICE, by=OWNER_DEVICE)
                rig.document.save()
                rig.broker.revoke_outstanding("github", BORROWER_DEVICE)
            finally:
                release.set()
            assert answer.result(3)["code"] == "not_a_holder"
        assert rig.broker._github_source._token == ""


def test_a_panic_during_backoff_prevents_the_second_borrower_ask(
    borrower: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.network import store
    from local_operator.network.credentials import client as client_mod
    from local_operator.network.credentials.types import BrokerError

    b = borrower
    b.response[0] = {"kind": "error", "code": "owner_busy", "key": "github"}
    sleeps = []

    def panic(delay: float) -> None:
        sleeps.append(delay)
        with store.mutate("n_gh", b.rig.env.root) as record:
            record.trust = "untrusted"
            store.save(record, b.rig.env.root)

    monkeypatch.setattr(client_mod.time, "sleep", panic)
    result = b.client.request_grant_sync("github", session_id="synthetic")
    assert isinstance(result, BrokerError) and result.code == "not_a_holder"
    assert len(b.calls) == 1 and sleeps == [0.25]


def test_isolated_source_preserves_the_owners_context(
    env: Any, github_api: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import contextvars

    marker = contextvars.ContextVar("synthetic_owner_scope", default=False)
    with _owner_for(env, github_api, github_mod.SOURCE_GH, monkeypatch=monkeypatch) as rig:
        original = github_mod.resolve_source
        observed = []

        def source(root: Any) -> str:
            observed.append(marker.get())
            return original(root)

        monkeypatch.setattr(github_mod, "resolve_source", source)
        token = marker.set(True)
        try:
            assert _ask(rig)["kind"] == "grant"
        finally:
            marker.reset(token)
        assert observed and all(observed), "off-loop probing must preserve the owner's context"
