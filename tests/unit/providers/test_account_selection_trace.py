"""The observer carries facts, not another selector or an account identity."""

from __future__ import annotations

import asyncio
import dataclasses
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from local_operator.providers.account_selection_trace import (
    SelectionExplanation,
    capture_selection,
    record_demotion,
    record_order,
    selected_reason,
)
from local_operator.providers.auth_store import AuthStore, OAuthAccess, StoredCredential


def row(cid: int, created: int = 10) -> StoredCredential:
    return StoredCredential(cid, "openai", "api_key", {"key": "synthetic"}, created_at=created)


def test_the_broker_vocabulary_still_imports_without_the_auth_store() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from local_operator.network.credentials.types import Grant; "
                "assert 'local_operator.providers.auth_store' not in sys.modules; "
                "print('no_auth_store_import')"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "no_auth_store_import"


def test_only_a_matching_returned_row_can_read_the_recorded_ordinal() -> None:
    a, b = row(1), row(2)
    assert selected_reason(a) is None
    with capture_selection():
        supplied = [a, b]
        assert record_order(supplied, "reset_rank") is supplied
        assert selected_reason(b) == "reset_rank:2"
        assert selected_reason(dataclasses.replace(b)) == "reset_rank:2"
        assert selected_reason(row(2, created=11)) == "unranked"
        assert selected_reason(None) == "unranked"
        assert selected_reason(row(3)) == "unranked"
    assert selected_reason(a) is None


def test_constructing_an_explanation_never_opens_a_store_or_reads_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("diagnostics must not query a credential store")

    monkeypatch.setattr(AuthStore, "__init__", forbidden)
    monkeypatch.setattr(AuthStore, "get_credential", forbidden)
    monkeypatch.setattr(AuthStore, "list_credentials", forbidden)
    a = row(1)
    with capture_selection():
        record_order([a], "reset_rank")
        assert (
            SelectionExplanation(
                "owner@example.test", selected_reason(a) or "unranked"
            ).audit_reason()
            == "reset_rank:1"
        )


def test_nested_observation_restores_the_parent_even_on_failure() -> None:
    a = row(1)
    with capture_selection():
        record_order([a], "affinity")
        with pytest.raises(RuntimeError, match="synthetic abort"):
            with capture_selection():
                assert selected_reason(a) == "unranked"
                record_order([a], "fair_hash")
                raise RuntimeError("synthetic abort")
        assert selected_reason(a) == "affinity:1"
    assert selected_reason(a) is None


@pytest.mark.asyncio
async def test_overlapping_tasks_and_cancelled_tasks_do_not_share_explanations() -> None:
    a = row(1)
    entered_a, entered_b, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def observe(basis: str, entered: asyncio.Event) -> str | None:
        with capture_selection():
            record_order([a], basis)
            entered.set()
            await release.wait()
            return selected_reason(a)

    first = asyncio.create_task(observe("affinity", entered_a))
    second = asyncio.create_task(observe("headroom_rank", entered_b))
    await entered_a.wait()
    await entered_b.wait()
    second.cancel()
    with pytest.raises(asyncio.CancelledError):
        await second
    release.set()
    assert await first == "affinity:1"
    assert selected_reason(a) is None


def test_demotion_observes_the_existing_final_partition_without_reordering() -> None:
    a, b, c = row(1), row(2), row(3)
    with capture_selection():
        record_order([a, b, c], "reset_rank")
        partition = [b, a, c]
        assert record_demotion(partition) is partition
        assert selected_reason(b) == "demotion_preference:1"
        assert selected_reason(a) == "demotion_preference:2"
        assert selected_reason(c) == "reset_rank:3"


@pytest.mark.parametrize(
    "bad",
    [
        "secret-material",
        "unknown:1",
        "affinity:0",
        "affinity:-1",
        "fair_hash:1\nsecret",
        "reset_rank:" + "9" * 30,
        f"reset_rank:{sys.maxsize + 1}",
    ],
)
def test_a_reason_cannot_be_an_arbitrary_payload(bad: str) -> None:
    assert SelectionExplanation(reason=bad).audit_reason() == "unranked"
    assert SelectionExplanation(reason="reset_rank:2").audit_reason() == "reset_rank:2"


@pytest.mark.asyncio
async def test_non_session_round_robin_is_observed_without_an_extra_cursor_step(
    tmp_path: Path,
) -> None:
    auth = AuthStore(config_dir=tmp_path)
    try:
        a = auth.upsert_credential("openai", {"type": "api_key", "key": "synthetic-a"})
        b = auth.upsert_credential("openai", {"type": "api_key", "key": "synthetic-b"})
        with capture_selection():
            first = await auth.get_oauth_access("openai", read_only=True)
        with capture_selection():
            second = await auth.get_oauth_access("openai", read_only=True)
        assert first is not None and second is not None
        assert (first.credential_id, second.credential_id) == (a.id, b.id)
        assert first.selection == second.selection == SelectionExplanation(None, "round_robin:1")
        assert auth._round_robin["openai:any"] == 2
    finally:
        auth.close()


@pytest.mark.asyncio
async def test_a_nested_getter_cannot_overwrite_the_outer_rows_explanation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import time

    from local_operator.providers.account_affinity import lend_selection_context

    auth = AuthStore(config_dir=tmp_path)
    try:
        a, b = [
            auth.upsert_credential(
                "openai",
                {
                    "type": "oauth",
                    "access": "synthetic-" + name,
                    "expires": int(time.time() * 1000) + 3_600_000,
                    "email": name + "@example.test",
                },
            )
            for name in ("a", "b")
        ]
        original = auth._ensure_oauth_fresh
        inner_access: list[OAuthAccess | None] = []

        async def nested(row: StoredCredential, **kwargs: Any) -> dict[str, Any]:
            if row.id == a.id:
                with lend_selection_context("reset-soonest", (b.id, b.created_at), frozenset()):
                    inner_access.append(
                        await auth.get_oauth_access("openai", "inner", read_only=True)
                    )
            return await original(row, **kwargs)

        monkeypatch.setattr(auth, "_ensure_oauth_fresh", nested)
        with (
            capture_selection(),
            lend_selection_context("reset-soonest", (a.id, a.created_at), frozenset()),
        ):
            outer = await auth.get_oauth_access("openai", "outer", read_only=True)
        assert outer is not None and outer.credential_id == a.id
        assert outer.selection == SelectionExplanation("a@example.test", "affinity:1")
        assert len(inner_access) == 1 and inner_access[0] is not None
        assert inner_access[0].credential_id == b.id
        assert inner_access[0].selection == SelectionExplanation("b@example.test", "affinity:1")
    finally:
        auth.close()


@pytest.mark.asyncio
async def test_a_request_local_pick_is_observed_without_changing_it(tmp_path: Path) -> None:
    from local_operator.harness.types import ChatRequest, ModelSpec

    auth = AuthStore(config_dir=tmp_path)
    try:
        auth.upsert_credential("openai", {"type": "api_key", "key": "synthetic-a"})
        b = auth.upsert_credential("openai", {"type": "api_key", "key": "synthetic-b"})
        boundary = auth.begin_account_request(
            "request-pick",
            ChatRequest(model=ModelSpec(provider="openai", model_id="synthetic"), messages=[]),
        )
        assert boundary is not None
        boundary.picks["openai"] = b.id
        with capture_selection():
            access = await auth.get_oauth_access("openai", "request-pick", read_only=True)
        assert access is not None and access.credential_id == b.id
        assert access.selection == SelectionExplanation(None, "request_pick:1")
        assert boundary.picks == {"openai": b.id}
        auth.end_account_request("request-pick", boundary)
    finally:
        auth.close()


@pytest.mark.asyncio
async def test_unranked_environment_access_has_no_fabricated_account_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-env")
    auth = AuthStore(config_dir=tmp_path)
    try:
        with capture_selection():
            access = await auth.get_oauth_access("openai", "synthetic-session", read_only=True)
        assert access is not None and access.credential_id == 0
        assert access.selection == SelectionExplanation(None, "unranked")
    finally:
        auth.close()
