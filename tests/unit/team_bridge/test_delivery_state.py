"""Per-recipient delivery state (contract v1): the recipient half.

Three layers, each at its own seam:

* the ONE pure mapping (``store.delivery_hold_class`` / ``store.refusal_class``)
  — a table over every closed screening category and every hold code the
  codebase emits, so a new category or code cannot silently leak or guess;
* the transport (``TeamBridgeClient``) — the new admit field and PUT route, and
  their degradation against an older hub;
* the connector — publish on hold and on reject, nothing raw on the wire, and
  the local owner notification when the hub reports a suspected probe.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from local_operator.team_bridge import client as tb_client
from local_operator.team_bridge import store as store_mod
from local_operator.team_bridge.client import BridgeRefusal
from local_operator.team_bridge.connector import loop as loop_mod
from local_operator.team_bridge.screening import HOLD_CODES, ScreenResult
from local_operator.team_bridge.store import (
    DELIVERY_HOLD_CLASSES,
    REFUSAL_CLASSES,
    Store,
    delivery_hold_class,
    refusal_class,
)
from local_operator.team_bridge.types import SCREENING_CATEGORIES
from tests.unit.team_bridge.test_client import _err, _make_client, _ok
from tests.unit.team_bridge.test_conversation_delivery import (
    FakeServer,
    FakeWire,
    _enroll,
    _owner,
    _persist_fake_receipt,
    _receipt,
    _ref,
    _wire_rig,
)

CID = "bcv_abcdefgh"
MID = "bcmsg_abcdef01"


@pytest.fixture(autouse=True)
def _destination_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tb_client, "org_oauth_destination_allowed", lambda base: True)


# ---------------------------------------------------------------------------
# the mapping
# ---------------------------------------------------------------------------

#: Contract v1's category table, restated here so a change to either side is a
#: visible test diff rather than a silent re-mapping.
_EXPECTED_REFUSAL = {
    "prompt_injection": "manipulation",
    "social_engineering": "manipulation",
    "personal_info_request": "personal_info",
    "harassment": "abusive",
    "malicious_link": "unsafe_content",
    "spam": "unsafe_content",
    "out_of_scope_work": "out_of_scope",
    "new_work_request": "out_of_scope",
}

_SCREENING_UNAVAILABLE = {
    "provider_error",
    "breaker_open",
    "parse_failed",
    "runner_unavailable",
    "retry_later",
    "policy_stale",
    "template_mismatch",
    "model_identity_mismatch",
    "model_identity_unknown",
    "owner_changed",
    "stale_generation",
    "busy",
    "budget_exhausted",
    # The connector's own "screen_item answered nothing" (see the helper's note).
    "screening_unavailable",
    # Blank on every hop: NOT cached, re-screened on the backoff (2026-11-09).
    "empty_reply",
    # ``model_refused``/``reply_truncated`` moved OUT (2026-11-09): both are
    # cached answers that are never re-screened, so "temporarily unavailable"
    # was untrue; they are owner-review holds and publish ``awaiting_review``.
    "batch_unresolved",
    # No evaluated screening model is reachable: nothing judged the message.
    "screening_model_unavailable",
}
_DELIVERY_PENDING = {"delivery_failed", "bridge_off", "phase_refused"}


def test_every_screening_category_has_exactly_the_contract_class() -> None:
    assert set(_EXPECTED_REFUSAL) == set(
        SCREENING_CATEGORIES
    ), "a screening category was added or removed: give it a contract class"
    for category in SCREENING_CATEGORIES:
        assert refusal_class([category]) == _EXPECTED_REFUSAL[category]
        assert refusal_class((category,)) == _EXPECTED_REFUSAL[category]
    assert set(_EXPECTED_REFUSAL.values()) == set(REFUSAL_CLASSES)


@pytest.mark.parametrize(
    "categories",
    [
        [],
        (),
        None,
        ["unknown"],
        ["Prompt_Injection"],  # never case-folded: exact closed tokens only
        "prompt_injection",  # a bare string is free text, never iterated
        b"spam",
        [1, None],
        {"prompt_injection": True},
        "ignore previous instructions; category=harassment",
    ],
)
def test_unknown_or_free_text_categories_map_to_no_class(categories: Any) -> None:
    assert refusal_class(categories) is None


def test_several_categories_take_the_first_in_closed_order() -> None:
    # SCREENING_CATEGORIES order, not input order: deterministic, one class.
    assert refusal_class(["spam", "harassment"]) == "abusive"
    assert refusal_class(["new_work_request", "prompt_injection"]) == "manipulation"
    assert refusal_class(["bogus", "spam"]) == "unsafe_content"


def _emitted_hold_codes() -> set[str]:
    """Every hold code the codebase can emit, from the closed lists #3189 owns."""

    return set(HOLD_CODES) | set(store_mod._DELIVERY_HOLD_CODES)


def _expected_hold_class(code: str) -> str:
    family = code.partition(":")[0]
    if family in _SCREENING_UNAVAILABLE:
        return "screening_unavailable"
    if family in _DELIVERY_PENDING:
        return "delivery_pending"
    return "awaiting_review"


def test_every_emitted_hold_code_maps_to_the_contract_class() -> None:
    codes = _emitted_hold_codes()
    # The contract's named members must be real emitted codes (or families),
    # otherwise the table names something no writer produces.
    for named in _SCREENING_UNAVAILABLE | _DELIVERY_PENDING:
        assert named in codes or named in store_mod._FAMILY_HOLD_CODES, named
    for code in sorted(codes):
        assert delivery_hold_class(code) == _expected_hold_class(code), code
        assert delivery_hold_class(code) in DELIVERY_HOLD_CLASSES


def test_the_contract_awaiting_review_examples() -> None:
    for code in (
        "model_hold",
        "local_quarantine",
        "ack_recovery",
        "owner_mismatch",
        "schema_damaged",
        "body_hash_mismatch",
        "bounce_without_category",
        "screening_hold",
    ):
        assert delivery_hold_class(code) == "awaiting_review", code


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("phase_refused:anything_the_hub_said", "delivery_pending"),
        ("policy_stale:policy_epoch", "screening_unavailable"),
        ("ack_recovery:whatever", "awaiting_review"),
        ("breaker_open:extra", "awaiting_review"),  # not a family: unknown
        ("", "awaiting_review"),
        (None, "awaiting_review"),
        (42, "awaiting_review"),
        ("totally_new_code", "awaiting_review"),
        ("Provider_Error", "awaiting_review"),
    ],
)
def test_families_collapse_and_unknowns_are_conservative(code: Any, expected: str) -> None:
    assert delivery_hold_class(code) == expected


def test_codes_written_as_literals_in_the_connector_are_covered() -> None:
    """A literal ``"code": "<x>"`` hold in the connector must be a known code.

    Otherwise it would fall to ``awaiting_review`` by accident rather than by
    decision. The whitelist is #3189's; this pins that the delivery-state
    mapping sees the same universe.
    """

    root = Path(store_mod.__file__).parent / "connector"
    literal = re.compile(r'"code":\s*"([a-z_]+)"')
    found: set[str] = set()
    for path in root.glob("*.py"):
        found |= set(literal.findall(path.read_text(encoding="utf-8")))
    unknown = found - _emitted_hold_codes() - {"admission_unverified", "screening_bounce"}
    assert not unknown, f"connector hold codes outside the whitelist: {sorted(unknown)}"


# ---------------------------------------------------------------------------
# the transport
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reject_admit_carries_the_class_and_allow_never_does() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return _ok({"verdict": "reject"})

    client = _make_client(handler)
    await client.conversation_admit(
        CID, MID, verdict="reject", body_hash="h", refusal_class="manipulation"
    )
    await client.conversation_admit(CID, MID, verdict="reject", body_hash="h")
    assert seen == [
        {"verdict": "reject", "body_hash": "h", "refusal_class": "manipulation"},
        {"verdict": "reject", "body_hash": "h"},
    ]
    with pytest.raises(ValueError):
        await client.conversation_admit(
            CID, MID, verdict="allow", body_hash="h", refusal_class="manipulation"
        )
    with pytest.raises(ValueError):  # a raw category is never a class
        await client.conversation_admit(
            CID, MID, verdict="reject", body_hash="h", refusal_class="prompt_injection"
        )
    assert len(seen) == 2


@pytest.mark.asyncio
async def test_an_old_hub_400_on_the_field_retries_the_reject_without_it() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if "refusal_class" in body:
            return _err(400, "invalid_field")
        return _ok({"verdict": "reject"})

    client = _make_client(handler)
    out = await client.conversation_admit(
        CID, MID, verdict="reject", body_hash="h", refusal_class="abusive"
    )
    assert out == {"verdict": "reject"}
    assert [("refusal_class" in body) for body in seen] == [True, False]


@pytest.mark.asyncio
async def test_a_non_400_refusal_with_the_field_is_not_retried() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return _err(409, "verdict_conflict")

    client = _make_client(handler)
    with pytest.raises(BridgeRefusal) as caught:
        await client.conversation_admit(
            CID, MID, verdict="reject", body_hash="h", refusal_class="abusive"
        )
    assert caught.value.http_status == 409
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_put_delivery_state_shape_and_old_hub_no_op() -> None:
    seen: list[tuple[str, str, dict[str, Any]]] = []
    answer: dict[str, httpx.Response] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path, json.loads(request.content)))
        return answer["next"]

    client = _make_client(handler)
    answer["next"] = _ok({"state": "held"})
    out = await client.put_delivery_state(
        CID, MID, tenant_id="tnt_1", endpoint_id="bep_a", hold_class="awaiting_review"
    )
    assert out == {"state": "held"}
    assert seen[0] == (
        "PUT",
        f"/v1/messaging/conversations/{CID}/messages/{MID}/delivery-state",
        {
            "tenant_id": "tnt_1",
            "endpoint_id": "bep_a",
            "state": "held",
            "hold_class": "awaiting_review",
        },
    )
    for status in (400, 404, 405):
        answer["next"] = _err(status, "not_found")
        assert (
            await client.put_delivery_state(
                CID, MID, tenant_id="tnt_1", endpoint_id="bep_a", hold_class="delivery_pending"
            )
            is None
        )
    answer["next"] = _err(403, "forbidden")
    with pytest.raises(BridgeRefusal):
        await client.put_delivery_state(
            CID, MID, tenant_id="tnt_1", endpoint_id="bep_a", hold_class="delivery_pending"
        )
    calls = len(seen)
    with pytest.raises(ValueError):  # a raw hold code never reaches the wire
        await client.put_delivery_state(
            CID, MID, tenant_id="tnt_1", endpoint_id="bep_a", hold_class="model_hold"
        )
    assert len(seen) == calls


# ---------------------------------------------------------------------------
# the connector
# ---------------------------------------------------------------------------


def _rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **wire: Any):
    a = _owner("a", 1)
    config_dir = tmp_path / "a"
    config_dir.mkdir()
    _enroll(config_dir, a, "sess_a")
    server = FakeServer(notify={"bep_a": "mute"}, active={"bep_a": True}, mentioned={"bep_a"})
    published: list[dict[str, Any]] = []
    admit_kwargs: list[dict[str, Any]] = []

    async def put_delivery_state(self: FakeWire, cid: str, mid: str, **kwargs: Any):
        published.append({"conversation_id": cid, "message_id": mid, **kwargs})
        if wire.get("put_raises"):
            raise wire["put_raises"]
        return {"state": "held"}

    original_admit = FakeWire.conversation_admit

    async def conversation_admit(self: FakeWire, cid: str, mid: str, **kwargs: Any):
        admit_kwargs.append(dict(kwargs))
        await original_admit(self, cid, mid, **kwargs)
        return wire.get("admit_answer", {"ok": True})

    monkeypatch.setattr(FakeWire, "put_delivery_state", put_delivery_state, raising=False)
    monkeypatch.setattr(FakeWire, "conversation_admit", conversation_admit)
    return a, config_dir, server, published, admit_kwargs


def _retry(owner: Any, config_dir: Path) -> dict[str, Any]:
    loop_mod._RETRY_ATTEMPTS.clear()
    return asyncio.run(
        loop_mod.retry_due_holds(
            owner,
            config_dir=config_dir,
            access=SimpleNamespace(credential_id=owner.credential_id),
            auth_store=None,
        )
    )


def test_a_screening_hold_publishes_held_with_the_bare_id_on_every_drain(
    tmp_path, monkeypatch
) -> None:
    a, config_dir, server, published, _admits = _rig(tmp_path, monkeypatch)
    _wire_rig(monkeypatch, server, {"a": (a, config_dir)}, disposition="hold")
    ref = _ref(server)
    Store(config_dir).persist_hold(a, ref.encode(), "screening_hold", retry_at=0)

    first = _retry(a, config_dir)
    Store(config_dir).persist_hold(a, ref.encode(), "screening_hold", retry_at=0)
    second = _retry(a, config_dir)

    assert first["outcomes"][0]["action"] == "hold"
    assert second["outcomes"][0]["action"] == "hold"
    # Re-published on each drain while the hold persists; keyed by the BARE id.
    assert (
        published
        == [
            {
                "conversation_id": CID,
                "message_id": MID,
                "tenant_id": "tnt_1",
                "endpoint_id": "bep_a",
                "hold_class": "awaiting_review",
                "access": published[0]["access"],
            }
        ]
        * 2
    )
    assert ":" not in published[0]["message_id"]


def test_a_provider_hold_publishes_only_the_coarse_class(tmp_path, monkeypatch) -> None:
    a, config_dir, server, published, _admits = _rig(tmp_path, monkeypatch)
    _wire_rig(monkeypatch, server, {"a": (a, config_dir)}, disposition="hold")

    async def provider_hold(owner: Any, item: Any, **kwargs: Any):
        return ScreenResult(disposition="hold", hold_code="breaker_open", retry_at=1.0)

    from local_operator.team_bridge.connector import conversation as conversation_mod

    monkeypatch.setattr(conversation_mod, "screen_item", provider_hold)
    ref = _ref(server)
    Store(config_dir).persist_hold(a, ref.encode(), "screening_hold", retry_at=0)
    _retry(a, config_dir)
    assert [entry["hold_class"] for entry in published] == ["screening_unavailable"]
    assert "breaker_open" not in json.dumps(
        {k: v for k, v in published[0].items() if k != "access"}
    )


def test_a_failed_held_write_never_changes_the_hold(tmp_path, monkeypatch) -> None:
    a, config_dir, server, published, _admits = _rig(
        tmp_path, monkeypatch, put_raises=BridgeRefusal("boom", "x", http_status=503)
    )
    _wire_rig(monkeypatch, server, {"a": (a, config_dir)}, disposition="hold")
    ref = _ref(server)
    Store(config_dir).persist_hold(a, ref.encode(), "screening_hold", retry_at=0)
    out = _retry(a, config_dir)
    assert out["outcomes"][0]["action"] == "hold"
    assert len(published) == 1
    assert Store(config_dir).screening_hold(a, ref.encode()) is not None


def _bounce_with(categories: tuple[str, ...], server: FakeServer) -> Any:
    from dataclasses import replace

    return ScreenResult(
        disposition="bounce",
        receipt=replace(_receipt("reject", server.body), categories=categories),
    )


def test_a_reject_admits_the_coarse_class_never_the_category(tmp_path, monkeypatch) -> None:
    a, config_dir, server, published, admits = _rig(tmp_path, monkeypatch)
    _wire_rig(monkeypatch, server, {"a": (a, config_dir)}, disposition="bounce")

    async def bounce(owner: Any, item: Any, **kwargs: Any):
        return _bounce_with(("social_engineering", "spam"), server)

    from local_operator.team_bridge.connector import conversation as conversation_mod

    monkeypatch.setattr(conversation_mod, "screen_item", bounce)
    ref = _ref(server)
    Store(config_dir).persist_hold(a, ref.encode(), "screening_hold", retry_at=0)
    out = _retry(a, config_dir)

    assert out["outcomes"][0]["action"] == "rejected"
    assert admits[0]["verdict"] == "reject"
    assert admits[0]["refusal_class"] == "manipulation"
    wire = json.dumps({k: v for k, v in admits[0].items() if k != "access"})
    for raw in SCREENING_CATEGORIES:
        assert raw not in wire
    assert published == []  # a verdict is never ALSO published as held


def _inbox(config_dir: Path) -> list[dict[str, Any]]:
    from local_operator.notifications.store import NotificationStore

    rows, _ = NotificationStore(config_dir).list_items(limit=50)
    return rows


@pytest.mark.parametrize("suspected", [True, False, "true", None])
def test_probe_suspected_raises_one_contentless_owner_notification(
    tmp_path, monkeypatch, suspected
) -> None:
    answer = {"verdict": "reject", "probe_suspected": suspected}
    a, config_dir, server, _published, _admits = _rig(tmp_path, monkeypatch, admit_answer=answer)
    _wire_rig(monkeypatch, server, {"a": (a, config_dir)}, disposition="bounce")
    server.body = "SECRET-BODY-CANARY"

    async def bounce(owner: Any, item: Any, **kwargs: Any):
        return _bounce_with(("harassment",), server)

    from local_operator.team_bridge.connector import conversation as conversation_mod

    monkeypatch.setattr(conversation_mod, "screen_item", bounce)
    ref = _ref(server)
    for _ in range(2):
        Store(config_dir).persist_hold(a, ref.encode(), "screening_hold", retry_at=0)
        _retry(a, config_dir)

    rows = _inbox(config_dir)
    if suspected is not True:
        assert rows == []
        return
    # One item per recipient endpoint (the hub's clamp spans conversations):
    # the second probe updates it, never adds one.
    assert len(rows) == 1
    row = rows[0]
    assert row["session_id"] == "sess_a"
    assert row["key"] == "team-bridge-probe:bep_a"
    assert row["title"] == "Agent messages: a sender looks like probing"
    rendered = json.dumps(row)
    for leak in ("SECRET-BODY-CANARY", "harassment", "abusive", "Ada", "acc_human", MID, CID):
        assert leak not in rendered, leak


@pytest.mark.parametrize(
    ("setup", "code"),
    [
        (lambda server: setattr(server, "body_route_available", False), "body_unavailable"),
        (lambda server: setattr(server, "body_override", "tampered bytes"), "body_hash_mismatch"),
    ],
)
def test_a_pre_screen_hold_publishes_held_too(tmp_path, monkeypatch, setup, code) -> None:
    """PR #3198 review R2: a hold before screening is as invisible to the sender."""

    a, config_dir, server, published, admits = _rig(tmp_path, monkeypatch)
    screened: list[str] = []
    _wire_rig(monkeypatch, server, {"a": (a, config_dir)}, screen_calls=screened)
    setup(server)
    ref = _ref(server)
    Store(config_dir).persist_hold(a, ref.encode(), "screening_hold", retry_at=0)
    out = _retry(a, config_dir)

    assert out["outcomes"][0]["action"] == "hold"
    hold = Store(config_dir).screening_hold(a, ref.encode())
    assert hold is not None and hold["hold_code"] == code
    assert screened == [] and admits == []
    assert [(p["message_id"], p["hold_class"]) for p in published] == [(MID, "awaiting_review")]
    assert code not in json.dumps({k: v for k, v in published[0].items() if k != "access"})


def test_the_engagement_gate_keeps_its_code_and_publishes_nothing(tmp_path, monkeypatch) -> None:
    """A hub write before the engagement gate is the spend that gate prevents.

    PR #3198 CI (`test_v1_unengaged_conversation_message_holds_visibly_and_never_wakes`,
    which forbids even building the wire here): the publish is skipped, the hold
    reports ``open_chief_of_staff`` — never ``conversation_failed``.
    """

    from local_operator.team_bridge.connector import conversation as conversation_mod

    a, config_dir, server, published, _admits = _rig(tmp_path, monkeypatch)
    _wire_rig(monkeypatch, server, {"a": (a, config_dir)})
    ref = _ref(server)
    wired: list[str] = []

    async def must_not_run(*args: Any, **kwargs: Any):
        wired.append("called")
        raise AssertionError("no hub client before the engagement gate")

    monkeypatch.setattr(conversation_mod, "engagement_hold", lambda d, c: "open_chief_of_staff")
    monkeypatch.setattr(conversation_mod, "_wire_for", must_not_run)
    Store(config_dir).persist_hold(a, ref.encode(), "screening_hold", retry_at=0)
    out = _retry(a, config_dir)

    assert out["outcomes"][0]["action"] == "hold"
    assert published == [] and wired == []
    hold = Store(config_dir).screening_hold(a, ref.encode())
    assert hold is not None and hold["hold_code"] == "open_chief_of_staff"


def _call_handler(owner: Any, ref: Any, config_dir: Path) -> dict[str, Any]:
    """Drive the handler itself: CI asserted ITS return value, so does this."""

    from local_operator.team_bridge.connector import conversation as conversation_mod

    return asyncio.run(
        conversation_mod.handle_conversation_available(
            owner,
            ref,
            config_dir=config_dir,
            auth_store=None,
            access=SimpleNamespace(credential_id=owner.credential_id),
        )
    )


def test_a_producer_fence_never_publishes(tmp_path, monkeypatch) -> None:
    # The connector may not act for this owner at all (bridge off, consent
    # revoked, owner re-enrolled): no hub write, not even a best-effort one.
    from local_operator.team_bridge.connector import conversation as conversation_mod

    a, config_dir, server, published, _admits = _rig(tmp_path, monkeypatch)
    _wire_rig(monkeypatch, server, {"a": (a, config_dir)})
    monkeypatch.setattr(conversation_mod, "_producer_fence", lambda *a, **k: "bridge_off")
    out = _call_handler(a, _ref(server), config_dir)

    assert out == {"action": "hold", "message_id": MID, "code": "bridge_off"}
    assert published == []


@pytest.mark.parametrize(
    ("setup", "screen", "code"),
    [
        (lambda s: setattr(s, "body_route_available", False), None, "body_unavailable"),
        (lambda s: setattr(s, "body_override", "tampered bytes"), None, "body_hash_mismatch"),
        (lambda s: None, "none", "screening_unavailable"),
        (lambda s: None, "hold", "screening_hold"),
    ],
)
def test_every_held_path_returns_its_own_code(tmp_path, monkeypatch, setup, screen, code) -> None:
    """PR #3198 CI: the publish is optional, the CODE never is.

    Only the hub write may be skipped. A hold that reports its own code must
    never come back as ``conversation_failed`` because a publish was skipped,
    unavailable or refused.
    """

    from local_operator.team_bridge.connector import conversation as conversation_mod

    a, config_dir, server, published, admits = _rig(tmp_path, monkeypatch)
    _wire_rig(monkeypatch, server, {"a": (a, config_dir)}, disposition=screen or "deliver")
    setup(server)
    if screen == "none":

        async def no_result(*args: Any, **kwargs: Any):
            return None

        monkeypatch.setattr(conversation_mod, "screen_item", no_result)
    out = _call_handler(a, _ref(server), config_dir)

    # A model's own hold also carries its budget tier for the retry order (#3400).
    out.pop("tier", None)
    assert out == {"action": "hold", "message_id": MID, "code": code}
    assert admits == []
    # And the sender still learns it is held, under a coarse class only.
    assert published and published[0]["message_id"] == MID
    if code not in DELIVERY_HOLD_CLASSES:  # screening_unavailable is both
        assert code not in json.dumps({k: v for k, v in published[0].items() if k != "access"})


def test_a_failing_publish_never_changes_the_hold_code(tmp_path, monkeypatch) -> None:
    a, config_dir, server, _published, _admits = _rig(tmp_path, monkeypatch)
    _wire_rig(monkeypatch, server, {"a": (a, config_dir)}, disposition="hold")

    async def exploding(*args: Any, **kwargs: Any):
        raise RuntimeError("hub gone")

    monkeypatch.setattr(FakeWire, "put_delivery_state", exploding, raising=False)
    out = _call_handler(a, _ref(server), config_dir)

    # A screening hold also carries its budget tier for the retry order (#3400).
    assert isinstance(out.pop("tier"), int)
    assert out == {"action": "hold", "message_id": MID, "code": "screening_hold"}


def test_a_slow_hub_costs_the_drain_at_most_the_short_cap(tmp_path, monkeypatch) -> None:
    """PR #3198 review R3: the cap is paid per held message, so it stays short."""

    from local_operator.team_bridge.connector import conversation as conversation_mod

    assert conversation_mod._DELIVERY_STATE_TIMEOUT_S == 3.0
    monkeypatch.setattr(conversation_mod, "_DELIVERY_STATE_TIMEOUT_S", 0.05)
    a, config_dir, server, published, _admits = _rig(tmp_path, monkeypatch)
    _wire_rig(monkeypatch, server, {"a": (a, config_dir)}, disposition="hold")

    async def stalled(self: FakeWire, cid: str, mid: str, **kwargs: Any):
        published.append({"message_id": mid})
        await asyncio.sleep(30)

    monkeypatch.setattr(FakeWire, "put_delivery_state", stalled, raising=False)
    ref = _ref(server)
    Store(config_dir).persist_hold(a, ref.encode(), "screening_hold", retry_at=0)
    import time as _time

    started = _time.monotonic()
    out = _retry(a, config_dir)
    assert _time.monotonic() - started < 5
    assert out["outcomes"][0]["action"] == "hold"
    assert len(published) == 1


def test_a_coalesced_card_publishes_no_state_because_delivered_rides_the_admit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#3201 x #3198: the coalescer must not re-publish a state it does not own.

    Contract v1 as #3198 implements it: the connector writes ``held`` (PUT
    delivery-state) for a message it is HOLDING, while ``delivered`` and
    ``refused`` ride the ADMIT, write-once with the verdict. A card that the
    wake coalescer buffered and flushed is therefore already covered by its
    admit — and the ``already_posted`` group even more so, because its admit
    happened on an earlier pass. Publishing anything at flush time would either
    duplicate the admit's verdict or claim a delivery that this process did not
    make.

    Structural half of the same fact: ``flush_due`` receives no hub client at
    all — only a ``deliver`` callback — so the coalescer CANNOT publish. This
    cell pins the observable side: one admit, no state write, and the admit
    carrying no refusal class.
    """

    a, config_dir, server, published, admit_kwargs = _rig(tmp_path, monkeypatch)
    _wire_rig(monkeypatch, server, {"a": (a, config_dir)})

    summary = asyncio.run(
        loop_mod.run_iteration(
            a,
            config_dir=config_dir,
            access=SimpleNamespace(credential_id=a.credential_id),
            auth_store=None,
        )
    )
    # The card went out in this iteration (the coalescer flushes a quiet
    # conversation at the end of it), and it is a WAVE, not a hold.
    assert [wake["action"] for wake in summary["coalesced_wakes"]] == ["delivered"]
    assert published == [], "a delivered wake writes no delivery state; its admit did"
    assert len(admit_kwargs) == 1, "exactly one admit carries this message's verdict"
    assert "refusal_class" not in admit_kwargs[0]


# ---------------------------------------------------------------------------
# fresh work before owner-review re-drives
# ---------------------------------------------------------------------------
#
# The receiver runs only while the host is awake, and a review hold waits on a
# PERSON (or on a delivery window up to seven days). The retry pass used to run
# as one block BEFORE the feed, so five review holds spent a whole wake window
# first and a message that had just arrived was never pulled. The iteration now
# runs the non-review half before the feed and the review half after it.

REVIEW_HOLD_CODE = "model_hold"  # in hold_release.REVIEW_HOLD_CODES
FRESH_BODY = "the message that just arrived"


class _OrderingWire:
    """One fake hub in both roles: the connector's wire and the tool's client.

    ``calls`` records only the two calls this test orders (a body read and an
    admit), and ``cancel_after`` models a wake window closing during one of
    them: the call that would be the (``cancel_after`` + 1)th raises
    ``CancelledError``, exactly as a stop does mid-spine.
    """

    def __init__(self, bodies: dict[str, str], *, cancel_after: int | None = None) -> None:
        self.bodies = bodies
        self.order = list(bodies)
        self.admitted: set[str] = set()
        self.calls: list[tuple[str, str]] = []
        self.cancel_after = cancel_after

    def _call(self, kind: str, message_id: str) -> None:
        if self.cancel_after is not None and len(self.calls) >= self.cancel_after:
            raise asyncio.CancelledError
        self.calls.append((kind, message_id))

    def _sha(self, message_id: str) -> str:
        return hashlib.sha256(self.bodies[message_id].encode("utf-8")).hexdigest()

    # -- the connector's wire ------------------------------------------------

    async def conversation_body(self, conversation_id: str, message_id: str, **_: Any):
        self._call("body", message_id)
        return {
            "message_id": message_id,
            "body": self.bodies[message_id],
            "body_hash": self._sha(message_id),
            "conversation_name": "#qa-coordination",
            "sender_label": "Ada",
            "sender_kind": "agent",
            "sender_account_id": "acc_other",
            "conversation_kind": "channel",
        }

    async def conversation_admit(
        self, conversation_id: str, message_id: str, *, verdict: str, body_hash: str, **_: Any
    ) -> dict[str, Any]:
        self._call("admit", message_id)
        self.admitted.add(message_id)
        return {"ok": True}

    # -- the bridge tool's read client ---------------------------------------

    async def conversation(self, conversation_id: str, **_: Any) -> dict[str, Any]:
        return {"id": CID, "kind": "channel", "name": "#qa-coordination", "members": []}

    async def conversation_messages(self, conversation_id: str, **_: Any) -> dict[str, Any]:
        rows = [
            {
                "id": message_id,
                "conversation_id": CID,
                "seq": index + 1,
                "sender": {"account_id": "acc_other", "kind": "agent"},
                # The hub's own rule: a body exists only for an ADMITTED row.
                "body": self.bodies[message_id] if message_id in self.admitted else None,
                "body_hash": self._sha(message_id),
                "origin": "agent",
                "created_at": "2026-11-12T12:00:00Z",
            }
            for index, message_id in enumerate(self.order)
        ]
        return {"messages": list(reversed(rows)), "has_more": False}


def _ordering_rig(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wire: _OrderingWire
) -> tuple[Any, Path, Any, list[str]]:
    """Five DUE review holds + one fresh ref in the feed, over the REAL store."""

    from local_operator.team_bridge import host as host_mod
    from local_operator.team_bridge import pull as pull_mod
    from local_operator.team_bridge.connector import coalesce as coalesce_mod
    from local_operator.team_bridge.connector import conversation as conversation_mod
    from local_operator.team_bridge.events import plan_feed_refs
    from local_operator.team_bridge.types import project_events_batch

    a = _owner("a", 1)
    config_dir = tmp_path / "a"
    config_dir.mkdir()
    store = _enroll(config_dir, a, "sess_a")
    events = [
        {
            "type": "message_available",
            "ids": {
                "conversation_id": CID,
                "message_id": message_id,
                "conversation_seq": index + 1,
            },
            "at": "2026-11-12T12:00:00Z",
        }
        for index, message_id in enumerate(wire.order)
    ]
    _cursor, refs, _resync = project_events_batch({"cursor": "1", "events": events})
    deliveries = plan_feed_refs(refs).conversation_deliveries
    review_refs = [ref for ref in deliveries if ref.message_id != wire.order[-1]]
    for ref in review_refs:
        store.persist_hold(a, ref.encode(), REVIEW_HOLD_CODE, retry_at=0.0)
    fresh_ref = deliveries[-1].encode()

    async def fake_events(
        owner: Any, *, config_dir: Path, access: Any, after: Any = None, **kw: Any
    ):
        return {"cursor": 1, "refs": [fresh_ref], "resync": False}

    def fake_drain(owner: Any, *, config_dir: Path, upto_seq: int) -> list[str]:
        return [fresh_ref]

    async def fake_wire_for(owner: Any, config_dir: Path, auth_store: Any) -> Any:
        return wire

    async def fake_screen(owner: Any, item: Any, *, config_dir: Path, **kwargs: Any):
        message_id = item["message"]["id"]
        receipt = _receipt("allow", wire.bodies[message_id])
        _persist_fake_receipt(config_dir, receipt, message_id=message_id)
        return ScreenResult(disposition="deliver", receipt=receipt)

    async def fake_deliver(owner: Any, payload: str, **kwargs: Any) -> str:
        return "delivered and woke the session"

    monkeypatch.setattr(pull_mod, "events", fake_events)
    monkeypatch.setattr(pull_mod, "drain_feed_refs", fake_drain)
    monkeypatch.setattr(conversation_mod, "_wire_for", fake_wire_for)
    monkeypatch.setattr(conversation_mod, "screen_item", fake_screen)
    monkeypatch.setattr(conversation_mod, "deliver_wake", fake_deliver)
    monkeypatch.setattr(conversation_mod, "engagement_hold", lambda designation, config_dir: "")
    monkeypatch.setattr(coalesce_mod, "CONNECTOR_COALESCER", coalesce_mod.WakeCoalescer())
    # The tool's fences have their own suites; open them for the read-back.
    monkeypatch.setattr(host_mod, "owner_fence_refusal", lambda runtime: None)
    monkeypatch.setattr(host_mod, "outbound_tool_fence", lambda runtime: None)
    monkeypatch.setattr(host_mod, "admission_refusal", lambda runtime, *, is_read=False: None)
    return a, config_dir, store, [ref.message_id for ref in review_refs]


def _iterate(a: Any, config_dir: Path) -> dict[str, Any]:
    return asyncio.run(
        loop_mod._run_iteration_inner(
            a,
            config_dir=config_dir,
            access=SimpleNamespace(credential_id=a.credential_id),
            auth_store=None,
        )
    )


def test_fresh_work_is_admitted_before_any_review_held_body_is_re_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bodies = {f"bcmsg_review{index:02d}": f"held {index}" for index in range(5)}
    bodies["bcmsg_fresh01"] = FRESH_BODY
    wire = _OrderingWire(bodies)
    a, config_dir, _store, review_ids = _ordering_rig(tmp_path, monkeypatch, wire)

    outcome = _iterate(a, config_dir)

    fresh = wire.order[-1]
    assert wire.admitted == {fresh, *review_ids}, "fresh work AND every due review hold moved"
    fresh_admit = wire.calls.index(("admit", fresh))
    first_hold_read = min(
        index
        for index, (kind, message_id) in enumerate(wire.calls)
        if kind == "body" and message_id in review_ids
    )
    assert (
        fresh_admit < first_hold_read
    ), f"a review hold was re-read before the fresh message was admitted: {wire.calls}"
    assert outcome["retries"]["due"] == 5


def test_a_window_closing_mid_review_redrive_has_already_admitted_fresh_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bodies = {f"bcmsg_review{index:02d}": f"held {index}" for index in range(5)}
    bodies["bcmsg_fresh01"] = FRESH_BODY
    fresh = "bcmsg_fresh01"
    # Two calls is exactly the fresh message's body read and its admit; the
    # third (a review-held body re-read) is where the window closes.
    wire = _OrderingWire(bodies, cancel_after=2)
    a, config_dir, _store, _review_ids = _ordering_rig(tmp_path, monkeypatch, wire)

    with pytest.raises(asyncio.CancelledError):
        _iterate(a, config_dir)

    assert wire.admitted == {fresh}, "the fresh message was admitted before the window closed"
    # ... and the reader can SEE it: history serves the body the hub admitted.
    from local_operator.team_bridge.tool import BridgeRuntime, execute_bridge

    runtime = BridgeRuntime(
        client=wire,  # type: ignore[arg-type]
        config_dir=config_dir,
        tenant_id="tnt_1",
        owner=a,
        session_id="sess_a",
    )
    answer = asyncio.run(
        execute_bridge(runtime, {"op": "channel_history", "channel_id": CID, "limit": 20})
    )
    assert answer["ok"] is True, answer
    rows = {row["id"]: row for row in json.loads(answer["content"].split("\n\n", 1)[1])["messages"]}
    assert rows[fresh]["body"] == FRESH_BODY
    assert rows[fresh].get("body_state") is None
