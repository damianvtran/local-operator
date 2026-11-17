"""Owner-local grant explanations describe the selected row, not a re-run pick.

Synthetic credentials only. The first fixture drives the real broker, selector
and audit writer; its in-process relay hop is complemented by real-link cells.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from tests.unit.network.test_credentials_owner import STUB_PROVIDER
from tests.unit.network.test_credentials_real_link import (
    _audit,
    _borrower_client,
    _lop_network,
    _pull,
    _share,
)
from tests.unit.network.test_lend_selection_parity import (
    EMAIL_A,
    EMAIL_B,
    HOUR_MS,
    _ended,
    _grants,
    _lend,
    _Stub,
    _turn,
    _usage,
    borrowed,
    devices,
    idp,
    mesh,
    owner,
    stub_provider,
)

_FIXTURES = (borrowed, devices, idp, mesh, owner, stub_provider)


@pytest.mark.asyncio
async def test_the_existing_grant_audit_explains_the_actual_ranked_account(
    borrowed: Any, idp: Any
) -> None:
    rig = borrowed
    _usage(rig.owner.root, {EMAIL_A: HOUR_MS, EMAIL_B: 4 * HOUR_MS})
    stub = _Stub()
    assert _ended(await _turn(rig.store, stub, "ranked-audit"))
    [grant] = _grants(rig)
    assert grant["account_label"] == EMAIL_A
    assert grant["ranked_reason"] == "reset_rank:1"
    assert grant["policy"] == "reset-soonest"
    assert stub.seen == ["bearer-a"]
    assert idp.posts == []


def test_a_skipped_first_candidate_explains_the_actual_second_winner(
    borrowed: Any, monkeypatch: pytest.MonkeyPatch, idp: Any
) -> None:
    from local_operator.providers.auth_store import AuthStoreError

    rig = borrowed
    _usage(rig.owner.root, {EMAIL_A: HOUR_MS, EMAIL_B: 4 * HOUR_MS})
    original = rig.owner.auth._ensure_oauth_fresh
    seen: list[int] = []

    async def fresh(row: Any, **kwargs: Any) -> Any:
        seen.append(row.id)
        if row.id == rig.row_a.id:
            raise AuthStoreError("synthetic unusable first account")
        # A later usage update is not permission to rewrite the pick's reason.
        _usage(rig.owner.root, {EMAIL_A: 4 * HOUR_MS, EMAIL_B: HOUR_MS})
        return await original(row, **kwargs)

    monkeypatch.setattr(rig.owner.auth, "_ensure_oauth_fresh", fresh)
    detail = _lend(rig, "skipped-first")
    assert detail["credential_ref"]["credential_id"] == rig.row_b.id
    [grant] = _grants(rig)
    assert (grant["account_label"], grant["ranked_reason"]) == (EMAIL_B, "reset_rank:2")
    assert seen == [rig.row_a.id, rig.row_b.id]
    assert idp.posts == []
    assert not rig.owner.auth._sticky and not rig.owner.auth._deprioritized


def test_an_unusable_anchor_does_not_claim_its_sibling_was_anchored(
    borrowed: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.providers.auth_store import AuthStoreError

    rig = borrowed
    _usage(rig.owner.root, {EMAIL_A: HOUR_MS, EMAIL_B: 4 * HOUR_MS})
    original = rig.owner.auth._ensure_oauth_fresh

    async def fresh(row: Any, **kwargs: Any) -> Any:
        if row.id == rig.row_b.id:
            raise AuthStoreError("synthetic unavailable anchor")
        return await original(row, **kwargs)

    monkeypatch.setattr(rig.owner.auth, "_ensure_oauth_fresh", fresh)
    reply = _lend(rig, "anchor-skipped", affinity=(rig.row_b.id, rig.row_b.created_at))
    assert reply["credential_ref"]["credential_id"] == rig.row_a.id
    [grant] = _grants(rig)
    assert (grant["account_label"], grant["ranked_reason"], grant["anchored"]) == (
        EMAIL_A,
        "affinity_fallback:2",
        False,
    )


def test_demotion_and_affinity_keep_their_existing_precedence(borrowed: Any) -> None:
    rig = borrowed
    _usage(rig.owner.root, {EMAIL_A: HOUR_MS, EMAIL_B: 4 * HOUR_MS})
    first = _lend(rig, "demoted", demote_credential_ids=[rig.row_a.id])
    assert first["credential_ref"]["credential_id"] == rig.row_b.id
    # Demotion eligibility hides A before ranking, so B is rank ONE among the
    # remaining eligible rows, not 'globally best' and not the original rank two.
    assert _grants(rig)[-1]["ranked_reason"] == "reset_rank:1"
    second = _lend(
        rig,
        "anchored",
        affinity=(rig.row_a.id, rig.row_a.created_at),
        demote_credential_ids=[rig.row_a.id],
    )
    assert second["credential_ref"]["credential_id"] == rig.row_a.id
    grant = _grants(rig)[-1]
    assert (grant["account_label"], grant["ranked_reason"], grant["anchored"]) == (
        EMAIL_A,
        "affinity:1",
        True,
    )
    assert not rig.owner.auth._sticky and not rig.owner.auth._deprioritized


@pytest.mark.parametrize("policy,basis", [("headroom", "headroom_rank"), ("fair", "fair_hash")])
def test_the_reason_identifies_the_policy_branch_without_repeating_policy(
    borrowed: Any, policy: str, basis: str
) -> None:
    rig = borrowed
    (rig.owner.root / "config.yml").write_text(
        f"values:\n  retry:\n    accountSelectionPolicy: {policy}\n"
    )
    _usage(rig.owner.root, {EMAIL_A: HOUR_MS, EMAIL_B: 4 * HOUR_MS})
    detail = _lend(rig, "policy-reason")
    [grant] = _grants(rig)
    expected_label = (
        EMAIL_A if detail["credential_ref"]["credential_id"] == rig.row_a.id else EMAIL_B
    )
    assert grant["account_label"] == expected_label
    assert grant["ranked_reason"] == f"{basis}:1" and grant["policy"] == policy


def test_an_observer_does_not_add_rank_reads_or_refresh_calls(
    borrowed: Any, monkeypatch: pytest.MonkeyPatch, idp: Any
) -> None:
    from local_operator.providers import account_selection_trace as trace

    rig = borrowed
    _usage(rig.owner.root, {EMAIL_A: HOUR_MS, EMAIL_B: 4 * HOUR_MS})
    calls = {"rank": 0, "refresh": 0}
    rank = rig.owner.auth._reset_candidates
    refresh = rig.owner.auth._ensure_oauth_fresh

    def counted_rank(*args: Any, **kwargs: Any) -> Any:
        calls["rank"] += 1
        return rank(*args, **kwargs)

    async def counted_refresh(*args: Any, **kwargs: Any) -> Any:
        calls["refresh"] += 1
        return await refresh(*args, **kwargs)

    monkeypatch.setattr(rig.owner.auth, "_reset_candidates", counted_rank)
    monkeypatch.setattr(rig.owner.auth, "_ensure_oauth_fresh", counted_refresh)
    first = _lend(rig, "observed")
    enabled = dict(calls)
    assert _grants(rig)[-1]["ranked_reason"] == "reset_rank:1"
    monkeypatch.setattr(trace, "capture_selection", contextlib.nullcontext)
    calls.update(rank=0, refresh=0)
    second = _lend(rig, "unobserved")
    assert calls == enabled == {"rank": 1, "refresh": 1}
    assert first["credential_ref"] == second["credential_ref"]
    assert _grants(rig)[-1]["ranked_reason"] == "unranked"
    assert idp.posts == []


def test_overlapping_distinct_anchors_never_share_the_other_tasks_explanation(
    borrowed: Any, monkeypatch: pytest.MonkeyPatch, idp: Any
) -> None:
    rig = borrowed
    _usage(rig.owner.root, {EMAIL_A: HOUR_MS, EMAIL_B: 4 * HOUR_MS})
    entered = {row.id: threading.Event() for row in (rig.row_a, rig.row_b)}
    release = {row.id: threading.Event() for row in (rig.row_a, rig.row_b)}
    original = rig.owner.auth._ensure_oauth_fresh

    async def fresh(row: Any, **kwargs: Any) -> Any:
        entered[row.id].set()
        assert await asyncio.to_thread(release[row.id].wait, 5), "release fence timed out"
        return await original(row, **kwargs)

    monkeypatch.setattr(rig.owner.auth, "_ensure_oauth_fresh", fresh)
    with ThreadPoolExecutor(max_workers=2) as pool:
        try:
            a = pool.submit(_lend, rig, "parallel-a", affinity=(rig.row_a.id, rig.row_a.created_at))
            assert entered[rig.row_a.id].wait(5)
            b = pool.submit(_lend, rig, "parallel-b", affinity=(rig.row_b.id, rig.row_b.created_at))
            assert entered[rig.row_b.id].wait(5)
            release[rig.row_b.id].set()
            assert b.result(5)["credential_ref"]["credential_id"] == rig.row_b.id
            release[rig.row_a.id].set()
            assert a.result(5)["credential_ref"]["credential_id"] == rig.row_a.id
        finally:
            for gate in release.values():
                gate.set()
    rows = {
        row["session_id"]: row["detail"]
        for row in rig.owner.audit.tail(200)
        if row["event"] == "credential.grant"
    }
    assert (rows["parallel-a"]["account_label"], rows["parallel-b"]["account_label"]) == (
        EMAIL_A,
        EMAIL_B,
    )
    assert {r["ranked_reason"] for r in rows.values()} == {"affinity:1"}
    assert idp.posts == []


def test_coalesced_callers_keep_the_same_immutable_winner_explanation(
    borrowed: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = borrowed
    _usage(rig.owner.root, {EMAIL_A: HOUR_MS, EMAIL_B: 4 * HOUR_MS})
    entered, joined, release = threading.Event(), threading.Event(), threading.Event()
    refresh = rig.owner.auth._ensure_oauth_fresh
    coalesced = rig.owner.broker._coalesced_resolve
    calls = 0
    fresh_calls = 0

    async def fresh(row: Any, **kwargs: Any) -> Any:
        nonlocal fresh_calls
        fresh_calls += 1
        entered.set()
        assert await asyncio.to_thread(release.wait, 5)
        return await refresh(row, **kwargs)

    async def resolve(**fields: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            joined.set()
        return await coalesced(**fields)

    monkeypatch.setattr(rig.owner.auth, "_ensure_oauth_fresh", fresh)
    monkeypatch.setattr(rig.owner.broker, "_coalesced_resolve", resolve)
    with ThreadPoolExecutor(max_workers=2) as pool:
        try:
            a = pool.submit(_lend, rig, "coalesced")
            assert entered.wait(5)
            b = pool.submit(_lend, rig, "coalesced")
            assert joined.wait(5)
            release.set()
            first, second = a.result(5), b.result(5)
        finally:
            release.set()
    assert fresh_calls == 1 and first["grant_id"] == second["grant_id"]
    grants = _grants(rig)
    assert len(grants) == 2
    assert {(g["account_label"], g["ranked_reason"]) for g in grants} == {(EMAIL_A, "reset_rank:1")}


def test_metadata_is_owner_local_and_cannot_change_wire_or_equality(borrowed: Any) -> None:
    from local_operator.network.credentials.types import Grant
    from local_operator.providers.account_selection_trace import SelectionExplanation
    from local_operator.providers.auth_store import OAuthAccess

    detail = _lend(borrowed, "wire-compat")
    original = Grant.from_detail(detail)
    explained = dataclasses.replace(
        original, selection=SelectionExplanation(EMAIL_B, "fair_hash:2")
    )
    assert original == explained and original.to_detail() == explained.to_detail() == detail
    forged = Grant.from_detail({**detail, "selection": {"label": "forged", "reason": "affinity:1"}})
    assert forged.selection is None and forged.to_detail() == detail
    access = OAuthAccess("synthetic", 1)
    assert access == dataclasses.replace(
        access, selection=SelectionExplanation(EMAIL_A, "reset_rank:1")
    )
    # Existing positional construction remains unchanged as well as the wire.
    import inspect

    assert inspect.signature(Grant).parameters["selection"].kind is inspect.Parameter.KEYWORD_ONLY
    assert (
        inspect.signature(OAuthAccess).parameters["selection"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )


def test_only_the_two_diagnostic_fields_extend_the_existing_audit_boundary() -> None:
    from local_operator.network.audit import _sanitize_detail
    from local_operator.providers.account_selection_trace import SelectionExplanation

    detail = _sanitize_detail(
        "credential.grant",
        {
            "account_label": "owner@example.test\n",
            "ranked_reason": SelectionExplanation(reason="untrusted-payload").audit_reason(),
            "identity": {"email": "not-to-be-journaled"},
            "candidates": ["not-to-be-journaled"],
            "account_ref": "not-a-new-identity-field",
            "access_token": "synthetic-material",
            "refresh_token": "synthetic-material",
        },
    )
    assert detail == {"account_label": EMAIL_A, "ranked_reason": "unranked"}


def test_real_tcp_borrow_persists_explanation_and_the_existing_log_cli_displays_it(
    mesh: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    from local_operator.network.credentials.types import Grant

    _share(mesh)
    _pull(mesh)
    capsys.readouterr()
    grant = _borrower_client(mesh).request_grant_sync(STUB_PROVIDER, session_id="real-ranked-grant")
    assert isinstance(grant, Grant)
    assert grant.selection is None, "the owner's display metadata must not cross the wire"
    rows = _audit(mesh.a, "credential.grant")
    [row] = [r for r in rows if r["detail"]["grant_id"] == grant.grant_id]
    assert row["detail"]["account_label"] == EMAIL_A
    assert row["detail"]["ranked_reason"] == "reset_rank:1"
    assert (
        len(mesh.idp.posts) == 1
    ), "the existing expired-row refresh, not a second diagnostic resolve"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(mesh.a.root))
    assert _lop_network("log", "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    [cli_row] = [r for r in payload["records"] if r["detail"].get("grant_id") == grant.grant_id]
    assert cli_row["detail"]["account_label"] == EMAIL_A
    assert cli_row["detail"]["ranked_reason"] == "reset_rank:1"
    assert _lop_network("log") == 0
    text = capsys.readouterr().out
    assert '"ranked_reason": "reset_rank:1"' in text and EMAIL_A in text
    export = tmp_path / "synthetic-audit-export.jsonl"
    assert _lop_network("log", "--export", str(export)) == 0
    capsys.readouterr()
    assert any(
        json.loads(line).get("detail", {}).get("ranked_reason") == "reset_rank:1"
        for line in export.read_text().splitlines()
    )
    with capsys.disabled():
        print("REAL_TCP_GRANT_AUDIT " + json.dumps(row, sort_keys=True))
        print("LOG_CLI_JSON_ROW " + json.dumps(cli_row, sort_keys=True))
        print("LOG_CLI_TEXT " + next(line for line in text.splitlines() if grant.grant_id in line))
        print("IDP_POST_COUNT " + str(len(mesh.idp.posts)))
