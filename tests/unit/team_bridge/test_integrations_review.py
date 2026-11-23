"""Regressions for round-one frozen Team Integration consent findings.

These use the existing fake HTTP transport and real consent queue/store; they
are not live agent-server evidence and synthetic answers are not operator consent.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.unit.team_bridge.test_integrations_tool import APP, CHANNEL, Rig, _draft

TOKEN = "itk_" + "7" * 32


@pytest.fixture
def rig(tmp_path, monkeypatch):
    return Rig(tmp_path, monkeypatch)


def _effective_detail(rig):
    original = rig.server.handle

    def handle(method, path, body):
        if path.endswith(f"/tokens/{TOKEN}/rotate") and (body or {}).get("prepare"):
            return 200, {"prepared": True, "commitment": rig.server._commit("rotate_token")}
        status, result = original(method, path, body)
        if path.endswith(f"/apps/{APP}") and method == "GET":
            result["installations"] = [{"conversation_id": CHANNEL}]
            result["tokens"] = [{"id": TOKEN, "scopes": ["alerts:ingest", "chat:write.thread"]}]
        return status, result

    rig.server.handle = handle


@pytest.mark.parametrize("op", ["create_hook", "mint_token", "rotate_token"])
def test_every_credential_card_discloses_effective_destinations_and_scopes(rig, op):
    _effective_detail(rig)
    args: dict[str, Any] = dict(op=op, app_id=APP, secret_name="SYNTHETIC_EFFECTIVE_GRANT")
    if op == "create_hook":
        args.update(vendor="generic", name="Build alerts", channel_ids=[CHANNEL])
    elif op == "mint_token":
        args["scopes"] = ["chat:write"]
    else:
        args["token_id"] = TOKEN
        rig.store.set("SYNTHETIC_EFFECTIVE_GRANT", b"old-synthetic-credential")
    result = rig.call(**args)
    assert result["state"] == "queued", result
    record = rig.queue.find(result["ask_id"])
    assert record is not None
    question = record["questions"][0]
    assert "#alerts-prod" in question["question"]
    assert "SYNTHETIC_EFFECTIVE_GRANT" in question["question"]
    snapshot = record["consent"]["review_snapshot"]
    assert snapshot["channel_ids"] == [CHANNEL]
    assert snapshot["revision"] == 1
    if op == "mint_token":
        assert snapshot["scopes"] == ["chat:write"]
        assert "chat:write" in question["question"]
    elif op == "rotate_token":
        assert snapshot["scopes"] == ["alerts:ingest", "chat:write.thread"]
        assert all(scope in question["question"] for scope in snapshot["scopes"])


@pytest.mark.parametrize("side", ["app", "commitment"])
@pytest.mark.parametrize("revision", [None, 0, True, "1", 1.0, 2])
def test_reprepared_publish_refuses_missing_invalid_or_changed_revision_before_filing(
    rig, side, revision
):
    first = _draft(rig)
    rig.answer(first["ask_id"], "Cancel")
    before = len(rig.queue.records())
    original = rig.server.handle

    def handle(method, path, body):
        status, result = original(method, path, body)
        if side == "app" and path.endswith(f"/apps/{APP}") and method == "GET":
            result["app"]["revision"] = revision
        if side == "commitment" and path.endswith("/publish") and (body or {}).get("prepare"):
            result["commitment"]["revision"] = revision
        return status, result

    rig.server.handle = handle
    result = rig.call(op="publish_app", app_id=APP, reask=True)
    assert result["state"] == "refused", result
    assert "revision" in result["message"]
    assert len(rig.queue.records()) == before
    assert not rig.server.minted
    assert rig.server.app_status == "draft"


def test_initial_draft_revision_must_match_its_commitment(rig):
    original = rig.server.handle

    def handle(method, path, body):
        status, result = original(method, path, body)
        if method == "POST" and path.endswith("/apps"):
            result["commitment"]["revision"] = 2
        return status, result

    rig.server.handle = handle
    result = _draft(rig)
    assert result["state"] == "refused"
    assert not rig.queue.records()
    assert not rig.server.minted


@pytest.mark.parametrize("legacy", [True, False])
def test_unsafe_or_inconsistent_approved_cards_cannot_mint(rig, legacy):
    _effective_detail(rig)
    args = dict(op="mint_token", app_id=APP, secret_name="SYNTHETIC_OLD_CARD")
    first = rig.call(**args)
    frozen = rig.consent(first["ask_id"])
    rig.answer(first["ask_id"], "Cancel")
    if legacy:
        frozen.pop("review_version", None)
        frozen.pop("review_snapshot", None)
        frozen.pop("commitment_revision", None)
    else:
        frozen["commitment_revision"] = frozen["review_snapshot"]["revision"] + 1
    # Recreate the old host-authored card through the real queue; do not edit
    # its journal or replace the human-answer seam with an approval boolean.
    rig.now[0] += 1
    old = rig.seam.file(
        {
            "id": "team_integration",
            "question": "Create an app token?",
            "options": [{"label": "Approve"}, {"label": "Cancel"}],
            "recommended": 0,
        },
        frozen,
        tool_call_id="synthetic-old-host-card",
    )
    assert old.state == "queued"
    rig.answer(old.ask_id, "Approve")
    result = rig.call(**args)
    assert result["state"] == "refused", result
    assert not rig.server.minted
    # The unsafe answer is spent, not reused. A repeat must obtain a new,
    # fully disclosed ask rather than silently turning refusal into a mint.
    fresh = rig.call(**args, reask=True)
    assert fresh["state"] == "queued" and fresh["ask_id"] != old.ask_id
    assert not rig.server.minted


def test_rotation_without_inherited_grant_never_files_approval(rig):
    _effective_detail(rig)
    original = rig.server.handle

    def handle(method, path, body):
        status, result = original(method, path, body)
        if method == "GET" and path.endswith(f"/apps/{APP}"):
            result["tokens"] = []
        return status, result

    rig.server.handle = handle
    rig.store.set("SYNTHETIC_ROTATE", b"old-synthetic-credential")
    result = rig.call(op="rotate_token", app_id=APP, token_id=TOKEN, secret_name="SYNTHETIC_ROTATE")
    assert result["state"] == "refused"
    assert not rig.queue.records()
    assert not rig.server.minted
