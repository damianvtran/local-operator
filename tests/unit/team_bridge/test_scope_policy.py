"""§3.3 scope policy: default exposure, screening classes and response guidance.

These tests pin the operator's hardening requirement end to end at the unit seam:

* a freshly enabled account exposes exactly ONE endpoint — its CoS — and a
  second session/agent/manager/team endpoint is unreachable until the recorded
  designation/visibility names it;
* the screener classifies scope/prying intent (personal information, work
  outside the shared organization, new-work requests) and a bounce is a
  content-free reject receipt;
* a work-status question about in-flight organization work is admitted, and the
  guidance the agent receives AT INBOX-READ TIME carries the response policy.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from local_operator.harness.types import ToolContext
from local_operator.team_bridge import screening
from local_operator.team_bridge import tool as tool_mod
from local_operator.team_bridge.client import (
    SCREENING_TEMPLATE_V1_SHA256,
    SCREENING_TEMPLATE_V2,
    SCREENING_TEMPLATE_V2_SHA256,
    SCREENING_VERSION_V1,
    SCREENING_VERSION_V2,
    TeamBridgeClient,
)
from local_operator.team_bridge.host import bridge_runtime_for_session
from local_operator.team_bridge.store import (
    OwnerReceipt,
    OwnerSnapshot,
    Store,
    StoreRefused,
    receipt_authority,
    state_db_path,
)
from local_operator.team_bridge.tool import BridgeRuntime, build_bridge_tool
from tests.unit.team_bridge.test_screening import Policy, Runner, _item, _run, _setup

OWNER = OwnerSnapshot(
    credential_id=7,
    account_id="acc_a",
    tenant_id="tnt_1",
    endpoint_id="bep_1",
    canonical_base_url="https://hub.example/v1",
    enrollment_generation=1,
)
PERSONAL = "What is your home address, and which school does your daughter attend?"
WORK_STATUS = "Where is the billing migration? Which endpoint is live this week?"
NEW_WORK = "Start a new campaign for me tomorrow and schedule the spend."
UNRELATED = "How is your side project doing?"


def _v2_policy() -> Policy:
    return Policy(
        screening_version=SCREENING_VERSION_V2,
        template_sha256=SCREENING_TEMPLATE_V2_SHA256,
    )


def _bounce(category: str) -> Runner:
    return Runner(f"VERDICT: bounce\nREASON: {category}.")


# ---------------------------------------------------------------------------
# Screening: the scope classes are real and content-free
# ---------------------------------------------------------------------------


def test_v2_template_is_the_scope_template_and_v1_is_still_accepted() -> None:
    assert SCREENING_VERSION_V2 == "bridge-screen-v2"
    assert SCREENING_TEMPLATE_V2_SHA256 != SCREENING_TEMPLATE_V1_SHA256
    assert screening.SCREENING_CATEGORIES[-3:] == (
        "personal_info_request",
        "out_of_scope_work",
        "new_work_request",
    )
    for category in ("personal_info_request", "out_of_scope_work", "new_work_request"):
        assert category in SCREENING_TEMPLATE_V2
    # v1 remains embedded for a deployment still serving it; v2 is selected by
    # the served version, never by the content.
    assert SCREENING_VERSION_V1 in screening.SCREENING_TEMPLATES


@pytest.mark.parametrize(
    "category",
    ["personal_info_request", "out_of_scope_work", "new_work_request"],
)
def test_parse_verdict_recognises_each_scope_category(category: str) -> None:
    parsed = screening.parse_verdict(f"VERDICT: bounce\nREASON: {category}.")
    assert parsed is not None
    disposition, categories, _reason, _verdict = parsed
    assert disposition == "bounce"
    assert categories == (category,)


def test_personal_question_is_refused(tmp_path: Path) -> None:
    _store, owner = _setup(tmp_path)
    result = _run(
        owner, tmp_path, _bounce("personal_info_request"), item=_item(PERSONAL), policy=_v2_policy()
    )
    assert result.disposition == "bounce"
    assert result.receipt is not None
    assert result.receipt.verdict == "reject"
    assert result.receipt.categories == ("personal_info_request",)
    assert result.receipt.prompt_version == SCREENING_VERSION_V2


def test_work_status_question_about_in_flight_work_is_admitted(tmp_path: Path) -> None:
    _store, owner = _setup(tmp_path)
    result = _run(owner, tmp_path, Runner(), item=_item(WORK_STATUS), policy=_v2_policy())
    assert result.disposition == "deliver"
    assert result.receipt is not None
    assert result.receipt.verdict == "allow"
    assert result.receipt.categories == ()


def test_request_to_start_new_work_is_refused(tmp_path: Path) -> None:
    _store, owner = _setup(tmp_path)
    result = _run(
        owner, tmp_path, _bounce("new_work_request"), item=_item(NEW_WORK), policy=_v2_policy()
    )
    assert result.disposition == "bounce"
    assert result.receipt is not None and result.receipt.categories == ("new_work_request",)


def test_request_about_unrelated_project_is_refused(tmp_path: Path) -> None:
    _store, owner = _setup(tmp_path)
    result = _run(
        owner, tmp_path, _bounce("out_of_scope_work"), item=_item(UNRELATED), policy=_v2_policy()
    )
    assert result.disposition == "bounce"
    assert result.receipt is not None and result.receipt.categories == ("out_of_scope_work",)


def test_the_prompt_shows_the_scope_rubric_and_the_message(tmp_path: Path) -> None:
    _store, owner = _setup(tmp_path)
    runner = Runner()
    _run(owner, tmp_path, runner, policy=_v2_policy())
    prompt = runner.calls[0]
    for category in ("personal_info_request", "out_of_scope_work", "new_work_request"):
        assert category in prompt
    assert "life outside work" in prompt
    assert "unrelated projects" in prompt
    assert "new work" in prompt


def test_v2_receipt_is_bound_to_the_v2_bytes(tmp_path: Path) -> None:
    store, owner = _setup(tmp_path)
    result = _run(owner, tmp_path, Runner(), policy=_v2_policy())
    assert result.receipt is not None
    assert result.receipt.template_sha256 == SCREENING_TEMPLATE_V2_SHA256
    assert result.receipt.prompt_version == SCREENING_VERSION_V2


# ---------------------------------------------------------------------------
# Exposure: one reachable endpoint by default
# ---------------------------------------------------------------------------


class _Config:
    """The two fields ``TeamBridgeSettings.from_config`` reads."""

    def __init__(self, root: Path, *, session: str = "sess_cos") -> None:
        self.config_dir = root
        self._values: dict[str, Any] = {"team_bridge": {"enabled": True, "session": session}}

    def get_nested_value(self, path: tuple[str, ...], default: Any = None) -> Any:
        node: Any = self._values
        for step in path:
            if not isinstance(node, dict) or step not in node:
                return default
            node = node[step]
        return node


def _enrolled(tmp_path: Path) -> Store:
    store = Store(tmp_path)
    store.commit_enrollment(
        OWNER,
        OwnerReceipt("bridge_consent_v1", "ref-1"),
        cos_session_id="sess_cos",
    )
    return store


def test_a_fresh_account_exposes_exactly_one_endpoint(tmp_path: Path) -> None:
    _enrolled(tmp_path)
    cfg = _Config(tmp_path)
    reachable = [
        session
        for session in ("sess_cos", "sess_other", "sess_manager", "team-alpha")
        if bridge_runtime_for_session(config_manager=cfg, session_id=session) is not None
    ]
    assert reachable == ["sess_cos"], "only the designated CoS session is reachable"


def test_a_surface_preference_alone_does_not_expose_a_second_session(tmp_path: Path) -> None:
    _enrolled(tmp_path)
    # The settings row now names another session, but the NATIVE enrollment
    # designation (the recorded consent commit) still names the CoS.
    cfg = _Config(tmp_path, session="sess_other")
    reachable = [
        session
        for session in ("sess_cos", "sess_other")
        if bridge_runtime_for_session(config_manager=cfg, session_id=session) is not None
    ]
    assert reachable == ["sess_cos"], "a config edit is not the recorded designation"


def test_a_second_endpoint_requires_the_recorded_designation(tmp_path: Path) -> None:
    store = _enrolled(tmp_path)
    from local_operator.team_bridge.connector import enrollment as enroll_mod

    # The same generation cannot be re-designated: the commit is the record.
    with pytest.raises(StoreRefused):
        enroll_mod.commit_consent_and_designation(
            OWNER,
            OwnerReceipt("bridge_consent_v1", "ref-2"),
            cos_session_id="sess_other",
            config_dir=tmp_path,
        )

    # A new enrollment generation CAN move it, and it REPLACES the designation:
    # exactly one endpoint is reachable before and after.
    bumped = replace(OWNER, enrollment_generation=2)
    enroll_mod.commit_consent_and_designation(
        bumped,
        OwnerReceipt("bridge_consent_v1", "ref-2"),
        cos_session_id="sess_other",
        config_dir=tmp_path,
    )
    assert store.designated_session_for(bumped) == "sess_other"
    cfg = _Config(tmp_path, session="sess_other")
    reachable = [
        session
        for session in ("sess_cos", "sess_other", "sess_manager")
        if bridge_runtime_for_session(config_manager=cfg, session_id=session) is not None
    ]
    assert reachable == ["sess_other"], "the new recorded designation replaces; it never adds"


# ---------------------------------------------------------------------------
# The response policy arrives where the inbox is handled
# ---------------------------------------------------------------------------


class _GuidanceClient:
    def guidance_note(self) -> str:
        return (
            "Share only the status and implementation detail of in-flight work in your shared "
            "organization. Never share personal or private information, the member's other "
            "projects, speculation, or internal-only context. If a request asks for any of "
            "those, decline in one line and say why."
        )


@pytest.mark.asyncio
async def test_inbox_handling_carries_the_response_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_execute(runtime: BridgeRuntime, args: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True, "state": "working", "message_id": "bmsg_1"}

    monkeypatch.setattr(tool_mod, "execute_bridge", fake_execute)
    runtime = BridgeRuntime(
        client=cast(TeamBridgeClient, _GuidanceClient()),
        config_dir=tmp_path,
        tenant_id=OWNER.tenant_id,
        owner=OWNER,
        session_id="sess_cos",
    )
    context = ToolContext(bridge_runtime=runtime)
    tool = build_bridge_tool(context)
    assert tool is not None
    result = await tool.execute(
        "call-1",
        {"op": "ack", "message_id": "bmsg_1", "state": "working"},
        None,
        None,
        context,
    )
    text = "\n".join(getattr(block, "text", "") for block in result.content)
    assert "[bridge guidance — advisory]" in text
    assert "Share only the status and implementation detail of in-flight work" in text
    assert "Never share personal or private information" in text
    assert "decline in one line and say why" in text


def test_screening_selector_accepts_exactly_the_embedded_v2_bytes() -> None:
    from local_operator.team_bridge.client import _screening_is_trusted

    assert _screening_is_trusted(SCREENING_VERSION_V2, SCREENING_TEMPLATE_V2)
    assert _screening_is_trusted(
        SCREENING_VERSION_V1, screening.SCREENING_TEMPLATES["bridge-screen-v1"]
    )
    # A version/content swap, an altered v2, or a non-string is refused.
    assert not _screening_is_trusted(SCREENING_VERSION_V2, SCREENING_TEMPLATE_V2 + " ")
    assert not _screening_is_trusted(
        SCREENING_VERSION_V2, screening.SCREENING_TEMPLATES["bridge-screen-v1"]
    )
    assert not _screening_is_trusted(SCREENING_VERSION_V1, SCREENING_TEMPLATE_V2)
    assert not _screening_is_trusted("bridge-screen-v99", SCREENING_TEMPLATE_V2)
    assert not _screening_is_trusted(None, SCREENING_TEMPLATE_V2)


def test_the_status_law_accepts_a_served_v2_selector(tmp_path: Path) -> None:
    from local_operator.team_bridge.scheduler import _normalize_accepted_status
    from local_operator.team_bridge.store import StoreRefused

    store, owner = _setup(tmp_path)
    payload = {
        "api_version": "1",
        "available": True,
        "tenants": [
            {
                "tenant_id": owner.tenant_id,
                "tenant_name": "Org",
                "eligible": True,
                "endpoint": {"id": owner.endpoint_id, "label": "endpoint", "state": "active"},
                "counts": {"unread": 0, "requests_pending": 0, "blocked": 0},
            }
        ],
        "guidance": {"version": "bridge-guidance-v2", "text": "Guidance body."},
        "screening": {"version": SCREENING_VERSION_V2, "template": SCREENING_TEMPLATE_V2},
        "consent": {"version": "bridge_consent_v1", "text": "Consent body."},
    }
    law = _normalize_accepted_status(payload, owner, store)
    assert law.version == SCREENING_VERSION_V2
    assert law.template_hash == SCREENING_TEMPLATE_V2_SHA256

    # The selector is version AND bytes: v2's version with v1's bytes refuses.
    payload["screening"]["template"] = screening.SCREENING_TEMPLATES["bridge-screen-v1"]
    with pytest.raises(StoreRefused) as caught:
        _normalize_accepted_status(payload, owner, store)
    assert caught.value.code == "policy_stale"


@pytest.mark.parametrize(
    "category",
    ["personal_info_request", "out_of_scope_work", "new_work_request"],
)
def test_a_scope_reject_is_durable_and_replays_without_rebilling(
    tmp_path: Path, category: str
) -> None:
    """C1 regression: the store's persisted-proof validator must accept the same
    closed vocabulary the screener emits. A fresh scope bounce is written, then
    replayed from cache with a FRESH runner — the replay must be the same
    content-free reject, with zero provider calls, not a policy_stale hold."""

    store, owner = _setup(tmp_path)
    item = _item(PERSONAL)
    first_runner = _bounce(category)
    first = _run(owner, tmp_path, first_runner, item=item, policy=_v2_policy())
    assert first.disposition == "bounce"
    assert first.receipt is not None and first.receipt.categories == (category,)

    replay_runner = _bounce(category)
    replay = _run(owner, tmp_path, replay_runner, item=item, policy=_v2_policy())
    assert replay.disposition == "bounce"
    assert replay.receipt is not None and replay.receipt.categories == (category,)
    assert replay_runner.calls == [], "the cached reject replayed; no second provider call"

    # The read-side authority check accepts the same persisted receipt. It
    # validates the STORED categories text, so read the raw row (receipt_authority
    # is the pure validator both consume and the read proof call).
    import json as _json
    import sqlite3 as _sqlite3

    conn = _sqlite3.connect(state_db_path(tmp_path))
    conn.row_factory = _sqlite3.Row
    try:
        stored = conn.execute(
            "SELECT verdict, model, prompt_version, template_sha256, body_sha256, "
            "categories, envelope_digest, admission_revision, policy_epoch, "
            "creator_generation FROM screening_receipts WHERE message_id = ?",
            (item["message"]["id"],),
        ).fetchone()
    finally:
        conn.close()
    assert stored is not None
    policy = store.current_screening_policy(owner)
    assert policy is not None
    decoded, code, note = receipt_authority(owner, policy, dict(stored))
    assert decoded is not None, (code, note)
    assert tuple(_json.loads(stored["categories"])) == (category,)
