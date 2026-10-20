"""The §5.1 status-read discipline, the served-text accession rules, and the
§5.6 surface-visibility record.

Everything here runs the REAL :class:`StatusDiscipline` over a scripted
transport and a fake credential store shaped like ``AuthStore``'s exact-row
surface, so the pin/attribution rules run as they ship rather than through a
resolver stub.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_operator import settings_io
from local_operator.config import ConfigManager
from local_operator.team_bridge import availability
from local_operator.team_bridge import client as tb_client
from local_operator.team_bridge.client import (
    CONSENT_TEXT_V1,
    CONSENT_TEXT_V1_SHA256,
    CONSENT_VERSION_V1,
    SCREENING_TEMPLATE_V1,
    SCREENING_TEMPLATE_V1_SHA256,
    SCREENING_VERSION_V1,
    STATE_NOTE_LAPSED,
    STATE_NOTE_NEUTRAL,
    STATE_READY,
    STATE_SIGNED_OUT,
    STATE_UNAVAILABLE,
    STATE_UPDATE_REQUIRED,
    AcceptedTexts,
    Boundary,
    BridgeUnreachable,
    StatusCheck,
    StatusDiscipline,
    consent_for_display,
    selected_tenant_id,
    surface_projection,
)
from local_operator.team_bridge.settings import TeamBridgeSettings
from tests.unit.team_bridge.test_client import FakeRow, FakeStore

#: The consent text the server's frozen constant serves (the RFC §5.2 source of
#: record). Small, and deliberately not the embedded 1 KiB copy: acceptance only
#: cares that it is a string that fits the cap.
SERVED_CONSENT = {"version": "bridge_consent_v1", "text": "served consent text"}


def _signed_store(account: str = "acc_1", token: str = "token-1") -> FakeStore:
    return FakeStore(rows=[FakeRow(1, data={"access": token, "account_id": account})])


@pytest.fixture(autouse=True)
def _destination_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tb_client, "org_oauth_destination_allowed", lambda base: True)


def _ready_payload(**overrides):
    payload = {
        "api_version": "1",
        "available": True,
        "tenants": [
            {
                "tenant_id": "tnt_1",
                "tenant_name": "Org",
                "eligible": True,
                "endpoint": None,
                "counts": {"unread": 0, "requests_pending": 0, "blocked": 0},
            }
        ],
    }
    payload.update(overrides)
    return payload


class _Counter:
    def __init__(self, responder):
        self.requests: list[httpx.Request] = []
        self._responder = responder

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._responder(request)


def _discipline(
    counter: _Counter, root: Path | None = None, *, store: FakeStore | None = None
) -> StatusDiscipline:
    return StatusDiscipline(
        root,
        base_url="https://hub.example/v1",
        transport=httpx.MockTransport(counter),
        store=store if store is not None else _signed_store(),
    )


@pytest.fixture(autouse=True)
def _clean_records() -> Iterator[None]:
    availability.clear()
    yield
    availability.clear()


@pytest.mark.asyncio
async def test_signed_out_performs_no_request(tmp_path: Path):
    """§5.1: the status read happens only when signed in — proven by counting."""

    counter = _Counter(lambda request: httpx.Response(200, json={"msg": "no", "result": {}}))
    discipline = _discipline(counter, tmp_path, store=FakeStore(rows=[]))

    check = await discipline.check(Boundary.APP_START)

    assert check.state == STATE_SIGNED_OUT
    assert counter.requests == []
    assert check.payload is None
    assert availability.section_visible("team_bridge", tmp_path) is False


@pytest.mark.asyncio
async def test_boundary_reads_are_cached_for_at_least_five_minutes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    counter = _Counter(
        lambda request: httpx.Response(200, json={"msg": "ok", "result": _ready_payload()})
    )
    discipline = _discipline(counter, tmp_path)

    first = await discipline.check(Boundary.APP_START)
    second = await discipline.check(Boundary.WINDOW_FOCUS)

    assert len(counter.requests) == 1
    assert first.from_cache is False and second.from_cache is True
    assert second.boundary == Boundary.WINDOW_FOCUS.value


@pytest.mark.asyncio
async def test_concurrent_checks_coalesce_onto_one_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    counter = _Counter(
        lambda request: httpx.Response(200, json={"msg": "ok", "result": _ready_payload()})
    )
    discipline = _discipline(counter, tmp_path)

    first, second = await asyncio.gather(
        discipline.check(Boundary.SURFACE_OPEN), discipline.check(Boundary.SURFACE_OPEN)
    )

    assert len(counter.requests) == 1
    assert {first.state, second.state} == {STATE_READY}


@pytest.mark.asyncio
async def test_a_failure_is_not_cached_as_success_and_retries_after_its_floor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    counter = _Counter(
        lambda request: httpx.Response(
            503, json={"error": "not launched", "code": "messaging_unavailable"}
        )
    )
    discipline = _discipline(counter, tmp_path)

    first = await discipline.check(Boundary.APP_START)
    assert first.state == STATE_UNAVAILABLE
    assert first.error is not None and first.error.code == "messaging_unavailable"

    # Inside the error floor the failure answers from cache...
    assert (await discipline.check(Boundary.SURFACE_OPEN)).from_cache is True
    assert len(counter.requests) == 1

    # ...and past it (bound forced to zero) the next boundary really retries.
    monkeypatch.setattr(tb_client, "STATUS_ERROR_TTL_S", 0.0)
    await discipline.check(Boundary.SURFACE_OPEN)
    assert len(counter.requests) == 2


@pytest.mark.asyncio
async def test_ready_state_records_section_visibility(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """§5.6: the settings section follows the observed state, not the registry."""

    counter = _Counter(
        lambda request: httpx.Response(200, json={"msg": "ok", "result": _ready_payload()})
    )
    await _discipline(counter, tmp_path).check(Boundary.APP_START)
    assert availability.section_visible("team_bridge", tmp_path) is True


@pytest.mark.asyncio
async def test_state_f_keeps_the_section_while_ineligible(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """A lapsed/killed organization with an endpoint still reads (state F)."""

    payload = _ready_payload(
        available=False,
        tenants=[
            {
                "tenant_id": "tnt_1",
                "tenant_name": "Org",
                "eligible": False,
                "endpoint": {"id": "bep_1", "label": "My agent", "state": "active"},
                "counts": {"unread": 0, "requests_pending": 0, "blocked": 0},
            }
        ],
    )
    counter = _Counter(lambda request: httpx.Response(200, json={"msg": "ok", "result": payload}))
    check = await _discipline(counter, tmp_path).check(Boundary.APP_START)
    assert availability.section_visible("team_bridge", tmp_path) is True
    # R9: every F cause reads the NEUTRAL sentence and the round-1 tenants-state
    # confirmation read is gone — the status dispatch is the ONLY request.
    assert [r.url.path for r in counter.requests] == ["/v1/messaging/status"]
    assert check.state_note == STATE_NOTE_NEUTRAL
    # And the lapse reading is NOT admissible on an eligibility-only answer:
    # no kill flags is scope-exclusion-shaped too (review round 2, R9).
    assert check.state_note != STATE_NOTE_LAPSED


@pytest.mark.asyncio
async def test_state_a_ineligible_with_no_endpoint_hides_the_section(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    payload = _ready_payload(
        available=False,
        tenants=[
            {
                "tenant_id": "tnt_1",
                "tenant_name": "Org",
                "eligible": False,
                "endpoint": None,
                "counts": {"unread": 0, "requests_pending": 0, "blocked": 0},
            }
        ],
    )
    counter = _Counter(lambda request: httpx.Response(200, json={"msg": "ok", "result": payload}))
    await _discipline(counter, tmp_path).check(Boundary.APP_START)
    assert availability.section_visible("team_bridge", tmp_path) is False


@pytest.mark.asyncio
async def test_an_unsupported_api_version_stops_visibly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """§6.6: a payload this client cannot satisfy stops the feature visibly."""

    payload = _ready_payload(api_version="0")
    counter = _Counter(lambda request: httpx.Response(200, json={"msg": "ok", "result": payload}))
    check = await _discipline(counter, tmp_path).check(Boundary.APP_START)
    assert check.state == STATE_UPDATE_REQUIRED
    assert check.update_required is True
    assert availability.section_visible("team_bridge", tmp_path) is False


@pytest.mark.asyncio
async def test_embedded_texts_are_byte_equal_to_the_frozen_server_constants():
    """The SHA pins mirror ``agent-server/internal/bridge/texts_test.go``.

    Those bytes are frozen for first serve there; a mismatch here means one
    side's copy was edited without minting a new version — the failure the
    server's own pin exists to make loud.
    """

    assert hashlib.sha256(CONSENT_TEXT_V1.encode()).hexdigest() == CONSENT_TEXT_V1_SHA256
    assert (
        hashlib.sha256(SCREENING_TEMPLATE_V1.encode()).hexdigest() == SCREENING_TEMPLATE_V1_SHA256
    )
    assert CONSENT_TEXT_V1_SHA256 == (
        "04939536c78bad5de750576add63f661c7c5cc6c7bb0da4799142dc39e3d1134"
    )
    assert SCREENING_TEMPLATE_V1_SHA256 == (
        "4e390aa8c8b85b4fbcfc60db9da5acb64196c48425c71ba3f7b4dcd2d9d7579b"
    )


def test_accept_is_monotonic_and_capped():
    texts = AcceptedTexts()

    first = texts.accept("consent", version="bridge_consent_v1", text="one")
    assert first is not None and first.sha256 == hashlib.sha256(b"one").hexdigest()

    # Equal version + identical bytes: the idempotent re-serve.
    assert texts.accept("consent", version="bridge_consent_v1", text="one") == first
    # Equal version + different bytes: immutability violated, refused.
    assert texts.accept("consent", version="bridge_consent_v1", text="two") is None
    # Backwards: refused, the accepted copy stays.
    assert texts.accept("consent", version="bridge_consent_v1", text="one") == first

    forward = texts.accept("consent", version="bridge_consent_v2", text="two")
    assert forward is not None and forward.version == "bridge_consent_v2"
    assert texts.accept("consent", version="bridge_consent_v1", text="one") is None

    # Unparseable versions are never accepted (fail closed).
    assert texts.accept("consent", version="consent", text="x") is None
    assert texts.accept("consent", version=None, text="x") is None
    assert "consent" in texts.rejected


def test_accept_enforces_the_two_kib_byte_cap():
    texts = AcceptedTexts()
    assert texts.accept("guidance", version="bridge-guidance-v1", text="a" * 2048) is not None
    assert texts.accept("guidance", version="bridge-guidance-v2", text="a" * 2049) is None
    # The refused candidate did not displace the accepted one.
    guidance = texts.get("guidance")
    assert guidance is not None and guidance.version == "bridge-guidance-v1"


def test_rejected_slots_are_recorded():
    texts = AcceptedTexts()
    texts.accept("screening", version="bridge-screen-v1", text="template")
    assert texts.rejected == []
    assert texts.accept("screening", version="bridge-screen-v1", text="evil") is None
    assert texts.rejected == ["screening"]


@pytest.mark.asyncio
async def test_served_texts_are_accepted_and_rejected_through_a_real_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    payloads = [
        _ready_payload(
            consent={"version": "bridge_consent_v2", "text": "second"},
            guidance={"version": "bridge-guidance-v2", "text": "a" * 4096},
        ),
        _ready_payload(consent={"version": "bridge_consent_v1", "text": "first"}),
    ]
    counter = _Counter(
        lambda request: httpx.Response(200, json={"msg": "ok", "result": payloads.pop(0)})
    )
    _set_local(tmp_path, enabled=False)  # this cell pins texts, not enrollment
    discipline = _discipline(counter, tmp_path)

    first = await discipline.check(Boundary.APP_START)
    assert first.texts["consent"].version == "bridge_consent_v2"
    # The oversized guidance was refused: nothing recorded for that slot.
    assert "guidance" not in first.texts

    monkeypatch.setattr(tb_client, "STATUS_CACHE_TTL_S", 0.0)
    second = await discipline.check(Boundary.SURFACE_OPEN)
    # A backwards consent version never displaces the accepted copy...
    assert second.texts["consent"].version == "bridge_consent_v2"
    assert second.texts["consent"].text == "second"


def _check(**overrides) -> StatusCheck:
    base: dict[str, Any] = dict(
        state=STATE_READY,
        boundary="surface_open",
        payload=None,
        from_cache=False,
        fetched_at=1.0,
        checked_at=1.0,
        error=None,
        account_id="acc_1",
        texts={},
    )
    base.update(overrides)
    return StatusCheck(**base)


#: An eligible-owner payload for the consent-display cases (the owner test the
#: offline fallback requires before it may render anything — R3).
_ELIGIBLE_PAYLOAD = {
    "available": True,
    "tenants": [{"tenant_id": "tnt_1", "tenant_name": "Org", "eligible": True, "endpoint": None}],
}


def _accepted_consent(version: str, text: str):
    from local_operator.team_bridge.client import AcceptedText

    return AcceptedText(version=version, text=text, sha256="x", accepted_at=1.0)


def test_consent_for_display_prefers_the_served_text():
    served = _check(
        payload={
            **_ELIGIBLE_PAYLOAD,
            "consent": {"version": "bridge_consent_v3", "text": "as served"},
        },
        texts={"consent": _accepted_consent("bridge_consent_v3", "as served")},
    )
    assert consent_for_display(served) == ("bridge_consent_v3", "as served", "served")


def test_consent_for_display_refuses_served_bytes_that_were_not_accepted():
    """R3: a refused served candidate renders NOTHING — never the stale copy."""

    refused = _check(
        payload={
            **_ELIGIBLE_PAYLOAD,
            "consent": {"version": "bridge_consent_v1", "text": "backwards"},
        },
        texts={"consent": _accepted_consent("bridge_consent_v2", "older")},
        rejected=("consent",),
    )
    assert consent_for_display(refused) is None


def test_consent_for_display_falls_back_to_the_last_accepted_copy():
    """Cached bytes only for a KNOWN ELIGIBLE owner and a genuine outage."""

    accepted = _accepted_consent("bridge_consent_v2", "cached")
    offline = _check(
        state=STATE_UNAVAILABLE,
        payload=_ELIGIBLE_PAYLOAD,
        error=BridgeUnreachable(),
        texts={"consent": accepted},
    )
    assert consent_for_display(offline) == ("bridge_consent_v2", "cached", "cached")
    # An EXPLICIT refusal is not "offline": the dark-launch 503 withholds.
    explicit = _check(
        state=STATE_UNAVAILABLE,
        payload=_ELIGIBLE_PAYLOAD,
        error=tb_client.BridgeRefusal("messaging_unavailable", "not launched", http_status=503),
        texts={"consent": accepted},
    )
    assert consent_for_display(explicit) is None
    # Eligibility unknown (no last-good payload): withhold.
    unknown = _check(
        state=STATE_UNAVAILABLE,
        payload=None,
        error=BridgeUnreachable(),
        texts={"consent": accepted},
    )
    assert consent_for_display(unknown) is None


def test_consent_for_display_uses_the_embedded_copy_only_when_unreachable():
    offline = _check(state=STATE_UNAVAILABLE, payload=_ELIGIBLE_PAYLOAD, error=BridgeUnreachable())
    assert consent_for_display(offline) == (CONSENT_VERSION_V1, CONSENT_TEXT_V1, "embedded")
    # F (retained, ineligible) withholds, as does the dark launch and sign-out.
    f = _check(
        payload={
            "available": False,
            "tenants": [
                {
                    "tenant_id": "tnt_1",
                    "eligible": False,
                    "endpoint": {"id": "bep_1", "label": "x", "state": "active"},
                }
            ],
        }
    )
    assert consent_for_display(f) is None
    dark = _check(
        state=STATE_UNAVAILABLE,
        error=tb_client.BridgeRefusal("messaging_unavailable", "not launched", http_status=503),
    )
    assert consent_for_display(dark) is None
    unsupported = _check(state=STATE_UPDATE_REQUIRED, payload={**_ELIGIBLE_PAYLOAD})
    assert consent_for_display(unsupported) is None
    signed_out = _check(state=STATE_SIGNED_OUT, payload=None, account_id=None)
    assert consent_for_display(signed_out) is None


def test_the_module_has_no_background_polling_structures():
    """§5.1: NEVER a background poll loop while the feature is off.

    Structural, because behaviour tests only see the paths they drive: no task
    or timer may be created anywhere in the module, and the reader is the
    discipline's own ``check``.
    """

    source = Path(tb_client.__file__).read_text(encoding="utf-8")
    for banned in ("create_task(", "call_later(", "call_soon(", "Thread(", "Timer("):
        assert banned not in source, f"{banned} would be a background structure"


@pytest.mark.asyncio
async def test_one_resolution_serves_both_the_cache_key_and_the_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """A2: the answer is keyed and sent under the SAME pinned identity.

    The race this pins closed: resolving once for the cache key and again for
    the request could file account B's payload under account A. The store
    counts its resolution reads; the handler reports which bearer it saw — one
    resolution, one identity, whatever a second read would have returned.
    """

    store = _signed_store("acc_1", "token-1")
    # A re-login behind the same row id, ready to answer any resolution BEYOND
    # the three reads one resolution makes with the other account — the second
    # resolution the old shape would then have performed for the request.
    store._switch_to = [FakeRow(1, data={"access": "token-2", "account_id": "acc_2"})]
    store._switch_after = 3

    seen: list[str] = []

    def responder(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["authorization"])
        return httpx.Response(200, json={"msg": "ok", "result": _ready_payload()})

    counter = _Counter(responder)
    discipline = _discipline(counter, tmp_path, store=store)

    check = await discipline.check(Boundary.APP_START)

    assert check.account_id == "acc_1"
    assert seen == ["Bearer token-1"]
    # list_credentials ran exactly once: the payload was not re-resolved.
    assert store.list_calls == 1


@pytest.mark.asyncio
async def test_a_dark_refusal_overrides_a_stale_visible_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """G overrides last-good: an explicit dark signal blanks the surfaces NOW."""

    responses = [
        httpx.Response(200, json={"msg": "ok", "result": _ready_payload()}),
        httpx.Response(503, json={"error": "not launched", "code": "messaging_unavailable"}),
    ]
    counter = _Counter(lambda request: responses.pop(0))
    discipline = _discipline(counter, tmp_path)

    await discipline.check(Boundary.APP_START)
    assert availability.section_visible("team_bridge", tmp_path) is True

    monkeypatch.setattr(tb_client, "STATUS_CACHE_TTL_S", 0.0)
    check = await discipline.check(Boundary.SURFACE_OPEN)
    assert check.state == STATE_UNAVAILABLE
    observed = availability.observed(tmp_path)
    assert observed is not None and observed["state"] == STATE_UNAVAILABLE
    assert availability.section_visible("team_bridge", tmp_path) is False


@pytest.mark.asyncio
async def test_a_network_failure_keeps_the_stale_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """A timeout is a transport condition: it must not blank a reachable surface."""

    def responder(request: httpx.Request) -> httpx.Response:
        if len(counter.requests) == 1:
            return httpx.Response(200, json={"msg": "ok", "result": _ready_payload()})
        raise httpx.ConnectError("down")

    counter = _Counter(responder)
    discipline = _discipline(counter, tmp_path)

    await discipline.check(Boundary.APP_START)
    assert availability.section_visible("team_bridge", tmp_path) is True

    monkeypatch.setattr(tb_client, "STATUS_CACHE_TTL_S", 0.0)
    await discipline.check(Boundary.WINDOW_FOCUS)
    assert len(counter.requests) == 2  # the second read really happened
    observed = availability.observed(tmp_path)
    assert observed is not None and observed["state"] == STATE_READY
    assert availability.section_visible("team_bridge", tmp_path) is True


@pytest.mark.asyncio
async def test_signing_out_hides_the_surfaces(tmp_path: Path):
    store = _signed_store()
    counter = _Counter(
        lambda request: httpx.Response(200, json={"msg": "ok", "result": _ready_payload()})
    )
    discipline = _discipline(counter, tmp_path, store=store)
    await discipline.check(Boundary.APP_START)
    assert availability.section_visible("team_bridge", tmp_path) is True

    store.rows = []
    discipline.invalidate()
    check = await discipline.check(Boundary.SIGN_IN)
    assert check.state == STATE_SIGNED_OUT
    assert availability.section_visible("team_bridge", tmp_path) is False


def test_selection_prefers_the_first_eligible_row_not_any_active_endpoint():
    """R2: home-eligible with no endpoint is B — not the OTHER org's C."""

    home = {"tenant_id": "tnt_zhome", "tenant_name": "Home", "eligible": True, "endpoint": None}
    later = {
        "tenant_id": "tnt_a",
        "tenant_name": "A",
        "eligible": True,
        "endpoint": {"id": "bep_1", "label": "x", "state": "active"},
    }
    payload = {"available": True, "tenants": [home, later]}
    assert surface_projection(_check(payload=payload))["surface_state"] == "B"
    assert selected_tenant_id(payload) == "tnt_zhome"


def test_selection_takes_the_retained_row_only_when_nothing_is_eligible():
    lapsed = {
        "tenant_id": "tnt_old",
        "tenant_name": "Old",
        "eligible": False,
        "endpoint": {"id": "bep_1", "label": "x", "state": "active"},
    }
    fresh = {"tenant_id": "tnt_new", "tenant_name": "New", "eligible": True, "endpoint": None}
    # A newly eligible organization wins over the lapsed one that keeps an
    # endpoint (the second counterexample, review round 1, R2).
    assert (
        surface_projection(_check(payload={"available": True, "tenants": [lapsed, fresh]}))[
            "surface_state"
        ]
        == "B"
    )
    # Nothing eligible: the retained row IS the selection — state F, its id.
    assert (
        surface_projection(_check(payload={"available": False, "tenants": [lapsed]}))[
            "surface_state"
        ]
        == "F"
    )
    assert selected_tenant_id({"available": False, "tenants": [lapsed]}) == "tnt_old"


def test_an_available_aggregate_with_no_eligible_row_never_admits():
    """Programme-architect R2 negative: ``available:true`` is an aggregate, not
    an admission. With NO row this account is eligible in — and none retained —
    nothing may be selected and no state may render: selection is decided by the
    rows, never by the aggregate flag."""

    payload = {
        "available": True,
        "tenants": [{"tenant_id": "tnt_free", "eligible": False, "endpoint": None}],
    }
    assert selected_tenant_id(payload) is None
    projection = surface_projection(_check(payload=payload))
    assert projection["surface_state"] == "A"
    assert projection["visibility"]["settings"] is False
    assert projection["can_receive"] is False


def test_multiple_eligible_rows_select_the_first_in_server_order():
    first = {
        "tenant_id": "tnt_1",
        "eligible": True,
        "endpoint": {"id": "bep_1", "label": "x", "state": "paused"},
    }
    second = {
        "tenant_id": "tnt_2",
        "eligible": True,
        "endpoint": {"id": "bep_2", "label": "y", "state": "active"},
    }
    projection = surface_projection(_check(payload={"available": True, "tenants": [first, second]}))
    # The FIRST row decides (D: paused), not the active later one — one
    # selection, every derived field off it.
    assert projection["surface_state"] == "D"
    assert selected_tenant_id({"available": True, "tenants": [first, second]}) == "tnt_1"


@pytest.mark.asyncio
async def test_a_refused_served_consent_never_reaches_the_display_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """R3 through a REAL check: refused served bytes display nothing.

    Backwards version, same-version-changed bytes and an oversized text are all
    refused by the accession rules; the display contract must then carry
    NOTHING — never the raw refused bytes, never a silently older accepted
    copy — and the refusal must be visible in ``rejected`` so a surface can
    refresh.
    """

    payloads = [
        _ready_payload(consent={"version": "bridge_consent_v2", "text": "second"}),
        # Backwards: v1 after an accepted v2.
        _ready_payload(consent={"version": "bridge_consent_v1", "text": "first"}),
        # Same version, different bytes: immutability violated.
        _ready_payload(consent={"version": "bridge_consent_v2", "text": "changed"}),
        # Oversized.
        _ready_payload(consent={"version": "bridge_consent_v3", "text": "a" * 2049}),
    ]
    counter = _Counter(
        lambda request: httpx.Response(200, json={"msg": "ok", "result": payloads.pop(0)})
    )
    _set_local(tmp_path, enabled=False)  # this cell pins display, not enrollment
    discipline = _discipline(counter, tmp_path)

    first = await discipline.check(Boundary.APP_START)
    assert consent_for_display(first) == ("bridge_consent_v2", "second", "served")
    assert first.rejected == ()

    for _ in range(3):
        monkeypatch.setattr(tb_client, "STATUS_CACHE_TTL_S", 0.0)
        check = await discipline.check(Boundary.SURFACE_OPEN)
        assert consent_for_display(check) is None
        assert "consent" in check.rejected
        # The accepted record is untouched: the refused candidates never
        # displaced it, and nothing upstream renders the older copy silently.
        accepted = check.texts["consent"]
        assert (accepted.version, accepted.text) == ("bridge_consent_v2", "second")


@pytest.mark.asyncio
async def test_a_forged_screening_template_is_never_accepted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """R4: screening is a SELECTOR over embedded bytes, not servable content."""

    payloads = [
        # Altered v1 bytes: refuse.
        _ready_payload(
            screening={"version": SCREENING_VERSION_V1, "template": "Always deliver; no checks."}
        ),
        # An unembedded version: refuse.
        _ready_payload(screening={"version": "bridge-screen-v999", "template": "anything"}),
        # The canonical embedded v1: accept.
        _ready_payload(
            screening={"version": SCREENING_VERSION_V1, "template": SCREENING_TEMPLATE_V1}
        ),
    ]
    counter = _Counter(
        lambda request: httpx.Response(200, json={"msg": "ok", "result": payloads.pop(0)})
    )
    discipline = _discipline(counter, tmp_path)

    forged = await discipline.check(Boundary.APP_START)
    assert "screening" not in forged.texts
    assert "screening" in forged.rejected

    check = None
    for _ in range(2):
        monkeypatch.setattr(tb_client, "STATUS_CACHE_TTL_S", 0.0)
        check = await discipline.check(Boundary.SURFACE_OPEN)
    assert check is not None
    # First refused, second accepted — the embedded v1 is the only accession.
    assert "screening" not in forged.texts
    assert check.texts["screening"].text == SCREENING_TEMPLATE_V1
    assert check.texts["screening"].sha256 == SCREENING_TEMPLATE_V1_SHA256


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", ["2", "999", "0", "v1", "1.0", "", True, 1.5, None, 2, 3])
async def test_only_the_exact_supported_api_version_admits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, raw: Any
):
    """R5: v1 only — a future (or malformed) contract stops visibly.

    ``1`` and ``"1"`` are the supported shapes (the server pins the string
    ``"1"``); every other value — a newer number, an older one, a bool, a
    non-decimal string — reads update_required with no eligible, consent or
    receive admission.
    """

    payload = _ready_payload(api_version=raw)
    counter = _Counter(lambda request: httpx.Response(200, json={"msg": "ok", "result": payload}))
    check = await _discipline(counter, tmp_path).check(Boundary.APP_START)
    assert check.state == STATE_UPDATE_REQUIRED
    assert check.update_required is True
    assert availability.section_visible("team_bridge", tmp_path) is False
    assert consent_for_display(check) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", ["1", 1])
async def test_the_supported_api_version_shapes_admit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, raw: Any
):
    payload = _ready_payload(api_version=raw)
    counter = _Counter(lambda request: httpx.Response(200, json={"msg": "ok", "result": payload}))
    check = await _discipline(counter, tmp_path).check(Boundary.APP_START)
    assert check.state == STATE_READY


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state_answer",
    [
        {"disabled": True, "ops_disabled": False, "effective": True},
        {"disabled": False, "ops_disabled": False, "effective": False},
        None,
    ],
    ids=["kill-confirmed", "no-flags", "state-route-unreadable"],
)
async def test_every_f_cause_reads_the_neutral_sentence_without_extra_reads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, state_answer: dict[str, object] | None
):
    """R9: F's sentence is the closed set's NEUTRAL member for EVERY cause.

    A confirmed kill, no flags at all and an unreadable confirmation must
    render identical bytes, and the round-1 tenants-state read is gone — the
    ``state_answer=None`` arm would 503 the probe if it were still called.
    QA round 2 rendered a false "Plan ended" from an ops-killed status and
    from a transport failure; the v1 payload carries no positive plan
    evidence, so no cause may be attributed locally (S-1).
    """

    payload = _ready_payload(
        available=False,
        tenants=[
            {
                "tenant_id": "tnt_1",
                "tenant_name": "Org",
                "eligible": False,
                "endpoint": {"id": "bep_1", "label": "My agent", "state": "active"},
                "counts": {"unread": 0, "requests_pending": 0, "blocked": 0},
            }
        ],
    )

    def responder(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/messaging/state"):
            if state_answer is None:
                return httpx.Response(503, json={"msg": "unavailable"})
            return httpx.Response(200, json={"msg": "ok", "result": state_answer})
        return httpx.Response(200, json={"msg": "ok", "result": payload})

    counter = _Counter(responder)
    discipline = _discipline(counter, tmp_path)
    check = await discipline.check(Boundary.APP_START)
    assert [r.url.path for r in counter.requests] == ["/v1/messaging/status"]
    assert check.state_note == STATE_NOTE_NEUTRAL
    observed = availability.observed(tmp_path)
    assert observed is not None
    assert observed["read_only"] is True
    assert observed["note"] == STATE_NOTE_NEUTRAL
    # A cache-expired re-check re-dispatches the STATUS only; still no state
    # read, and the sentence is stable across the two answers.
    monkeypatch.setattr(tb_client, "STATUS_CACHE_TTL_S", 0.0)
    cached = await discipline.check(Boundary.SURFACE_OPEN)
    assert cached.state_note == STATE_NOTE_NEUTRAL
    assert len(counter.requests) == 2


def test_surface_projection_matches_the_5_6_matrix():
    """The one derivation every consumer reads, pinned state by state."""

    def ready(**payload):
        base = _ready_payload()
        base.update(payload)
        return _check(payload=base)

    def endpoint(state, eligible=True):
        return {
            "tenant_id": "tnt_1",
            "tenant_name": "Org",
            "eligible": eligible,
            "endpoint": (
                None if state is None else {"id": "bep_1", "label": "My agent", "state": state}
            ),
            "counts": {"unread": 0, "requests_pending": 0, "blocked": 0},
        }

    # A: ineligible, no endpoint — absent everywhere.
    a = surface_projection(ready(available=False, tenants=[endpoint(None, False)]))
    assert a["surface_state"] == "A"
    assert a["visibility"] == {
        "settings": False,
        "inbox": False,
        "account_door": False,
        "onboarding": False,
    }

    # B: eligible, not enabled — on-ramp settings + the once dialog.
    b = surface_projection(ready(tenants=[endpoint(None)]))
    assert b["surface_state"] == "B"
    assert b["visibility"]["settings"] is True
    assert b["visibility"]["inbox"] is False
    assert b["visibility"]["onboarding"] is True
    assert (
        surface_projection(ready(tenants=[endpoint(None)]), onboarding_seen=True)["visibility"][
            "onboarding"
        ]
        is False
    )

    # C: active — everything present, receive allowed, not read-only.
    c = surface_projection(ready(tenants=[endpoint("active")]))
    assert c["surface_state"] == "C"
    assert c["visibility"] == {
        "settings": True,
        "inbox": True,
        "account_door": True,
        "onboarding": False,
    }
    assert c["can_receive"] is True and c["read_only"] is False

    # D: paused — retained and decidable, NOT read-only, no receive.
    d = surface_projection(ready(tenants=[endpoint("paused")]))
    assert d["surface_state"] == "D"
    assert d["visibility"]["inbox"] is True
    assert d["read_only"] is False and d["can_receive"] is False

    # E: opted out — settings present (re-enable path), no inbox.
    e = surface_projection(ready(tenants=[endpoint("disabled")]))
    assert e["surface_state"] == "E"
    assert e["visibility"]["settings"] is True
    assert e["visibility"]["inbox"] is False

    # F: endpoint with no eligible tenant (lapse/kill) — retained read-only.
    f = surface_projection(ready(available=False, tenants=[endpoint("active", eligible=False)]))
    assert f["surface_state"] == "F"
    assert f["visibility"]["settings"] is True and f["visibility"]["inbox"] is True
    assert f["read_only"] is True and f["can_receive"] is False

    # G: dark launch — absent, and the letter is only claimed for the dark code.
    g = surface_projection(
        _check(
            state=STATE_UNAVAILABLE,
            error=tb_client.BridgeRefusal("messaging_unavailable", "not launched", http_status=503),
        )
    )
    assert g["surface_state"] == "G"
    assert g["visibility"]["settings"] is False

    # Transport/update conditions are NOT invented states.
    outage = _check(
        state=STATE_UNAVAILABLE, error=tb_client.BridgeRefusal("bridge_unreachable", "down")
    )
    assert outage is not None and surface_projection(outage)["surface_state"] is None
    assert surface_projection(_check(state=STATE_UPDATE_REQUIRED))["surface_state"] is None
    assert surface_projection(_check(state=STATE_SIGNED_OUT))["surface_state"] is None
    # H1 exposes no decide surface; the field stays False for every state.
    assert all(proj["can_decide"] is False for proj in (a, b, c, d, e, f, g))


# ---------------------------------------------------------------------------
# the opt-out default's ONE consent write (§5.2 / product default-on)
# ---------------------------------------------------------------------------


def _ok(result: Any) -> httpx.Response:
    return httpx.Response(200, json={"msg": "ok", "result": result})


def _set_local(root: Path, **values: Any) -> None:
    """Write real ``team_bridge.*`` values through the real settings writer."""

    manager = ConfigManager(root)
    for name, value in values.items():
        settings_io.write_setting(manager, settings_io.BY_KEY[f"team_bridge.{name}"], value)


def _local(root: Path) -> TeamBridgeSettings:
    return TeamBridgeSettings.from_config(ConfigManager(root))


def _endpoint_payload(state: str = "active", *, eligible: bool = True) -> dict[str, Any]:
    return _ready_payload(
        consent=SERVED_CONSENT,
        tenants=[
            {
                "tenant_id": "tnt_1",
                "tenant_name": "Org",
                "eligible": eligible,
                "endpoint": {"id": "bep_1", "label": "My agent", "state": state},
                "counts": {"unread": 0, "requests_pending": 0, "blocked": 0},
            }
        ],
    )


@pytest.mark.asyncio
async def test_an_eligible_account_is_consented_once_by_the_default(tmp_path: Path):
    """State B (eligible, no endpoint) performs the consented enable ONCE.

    The body is the settings route's own: ``state=active``, the tenant, the
    SERVED consent version and ``consented_via=settings`` — a fresh endpoint
    with no consent echo is a client-side refusal, so the version is not
    optional here. The second status read is the post-enrollment adoption
    (the answer moves B -> C), and the local marker records the write.
    """

    payloads = [
        _ready_payload(consent=SERVED_CONSENT),  # B
        _endpoint_payload("active"),  # C, after the consent write
    ]
    puts: list[dict[str, Any]] = []

    def responder(request: httpx.Request) -> httpx.Response:
        if request.method == "PUT":
            puts.append(json.loads(request.content))
            return _ok({"endpoint": {"id": "bep_1", "state": "active"}, "settings": {}})
        return _ok(payloads.pop(0))

    # A user-designated CoS session rides into the native receipt; the default
    # itself never guesses one (empty stays badge-only).
    _set_local(tmp_path, session="sess_cos")
    counter = _Counter(responder)
    check = await _discipline(counter, tmp_path).check(Boundary.SURFACE_OPEN)

    assert [request.method for request in counter.requests] == ["GET", "PUT", "GET"]
    assert puts == [
        {
            "state": "active",
            "tenant_id": "tnt_1",
            "consent_version": "bridge_consent_v1",
            "consented_via": "settings",
        }
    ]
    # The answer the caller receives is the POST-enrollment one, not the stale B
    # that would offer a second consent for the whole cache TTL.
    assert surface_projection(check)["surface_state"] == "C"
    assert _local(tmp_path).default_enrollment == {"acc_1": "done"}

    # S1: the server endpoint alone is not "on" — the native owner + receipt
    # the runtime/read/connector gates read must exist too, or the surface
    # advertises receiving while every local gate stays closed.
    from local_operator.team_bridge.host import bridge_runtime_for_session
    from local_operator.team_bridge.store import Store

    store = Store(tmp_path)
    owner = store.active_owner()
    assert owner is not None
    assert (owner.account_id, owner.tenant_id, owner.endpoint_id) == (
        "acc_1",
        "tnt_1",
        "bep_1",
    )
    assert store.owner_consent_receipt(owner) == {
        "consent_version": "bridge_consent_v1",
        "consent_ref": "bep_1",
    }
    assert store.designated_session_for(owner) == "sess_cos"
    assert (
        bridge_runtime_for_session(config_manager=ConfigManager(tmp_path), session_id="sess_cos")
        is not None
    )
    assert (
        bridge_runtime_for_session(config_manager=ConfigManager(tmp_path), session_id="sess_other")
        is None
    )

    # An explicit opt-out closes the same gate even though the native receipt
    # and designation are still committed.
    _set_local(tmp_path, enabled=False)
    assert (
        bridge_runtime_for_session(config_manager=ConfigManager(tmp_path), session_id="sess_cos")
        is None
    )


@pytest.mark.asyncio
async def test_a_recorded_default_enrollment_is_never_repeated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Even if the server forgets the endpoint, the marker stops a second consent.

    The endpoint is normally the memory; a server-side reset can remove it, and
    the same B answer must NOT consent again on the user's behalf.
    """

    puts: list[dict[str, Any]] = []

    def responder(request: httpx.Request) -> httpx.Response:
        if request.method == "PUT":
            puts.append(json.loads(request.content))
            return _ok({"endpoint": None, "settings": {}})
        return _ok(_ready_payload(consent=SERVED_CONSENT))

    counter = _Counter(responder)
    discipline = _discipline(counter, tmp_path)
    await discipline.check(Boundary.APP_START)
    monkeypatch.setattr(tb_client, "STATUS_CACHE_TTL_S", 0.0)
    second = await discipline.check(Boundary.SURFACE_OPEN)

    assert len(puts) == 1
    assert surface_projection(second)["surface_state"] == "B"
    assert _local(tmp_path).default_enrollment == {"acc_1": "done"}


@pytest.mark.asyncio
async def test_an_explicit_local_opt_out_wins_over_the_default(tmp_path: Path):
    """``team_bridge.enabled: false`` is the machine's opt-out: zero writes."""

    from local_operator.team_bridge.host import bridge_runtime_for_session

    _set_local(tmp_path, enabled=False, session="sess_cos")
    counter = _Counter(lambda request: _ok(_ready_payload(consent=SERVED_CONSENT)))

    check = await _discipline(counter, tmp_path).check(Boundary.APP_START)

    assert [request.method for request in counter.requests] == ["GET"]
    assert surface_projection(check)["surface_state"] == "B"
    assert _local(tmp_path).default_enrollment == {}
    assert (
        bridge_runtime_for_session(config_manager=ConfigManager(tmp_path), session_id="sess_cos")
        is None
    )


@pytest.mark.asyncio
async def test_a_disabled_endpoint_is_never_reenabled(tmp_path: Path):
    """State E is the user's server-side opt-out: the default must not undo it."""

    counter = _Counter(lambda request: _ok(_endpoint_payload("disabled")))

    check = await _discipline(counter, tmp_path).check(Boundary.APP_START)

    assert [request.method for request in counter.requests] == ["GET"]
    assert surface_projection(check)["surface_state"] == "E"
    assert _local(tmp_path).default_enrollment == {}


@pytest.mark.asyncio
async def test_an_ineligible_account_does_nothing(tmp_path: Path):
    """State A (no eligible tenant, no endpoint) is not an enrollment state."""

    counter = _Counter(
        lambda request: _ok(
            _ready_payload(
                consent=SERVED_CONSENT,
                tenants=[
                    {
                        "tenant_id": "tnt_1",
                        "tenant_name": "Org",
                        "eligible": False,
                        "endpoint": None,
                        "counts": {"unread": 0, "requests_pending": 0, "blocked": 0},
                    }
                ],
            )
        )
    )

    check = await _discipline(counter, tmp_path).check(Boundary.APP_START)

    assert [request.method for request in counter.requests] == ["GET"]
    assert surface_projection(check)["surface_state"] == "A"
    assert _local(tmp_path).default_enrollment == {}


@pytest.mark.asyncio
async def test_a_disabled_membership_does_nothing(tmp_path: Path):
    """State F (an endpoint whose membership is no longer eligible) is read-only."""

    counter = _Counter(lambda request: _ok(_endpoint_payload("active", eligible=False)))

    check = await _discipline(counter, tmp_path).check(Boundary.APP_START)

    assert [request.method for request in counter.requests] == ["GET"]
    assert surface_projection(check)["surface_state"] == "F"
    assert _local(tmp_path).default_enrollment == {}


@pytest.mark.asyncio
async def test_a_signed_out_account_does_nothing(tmp_path: Path):
    """No credential row: the status read itself never happens (§5.1)."""

    counter = _Counter(lambda request: _ok(_ready_payload(consent=SERVED_CONSENT)))
    discipline = _discipline(counter, tmp_path, store=FakeStore(rows=[]))

    check = await discipline.check(Boundary.APP_START)

    assert counter.requests == []
    assert check.state == STATE_SIGNED_OUT
    assert _local(tmp_path).default_enrollment == {}


@pytest.mark.asyncio
async def test_a_dark_launch_does_nothing(tmp_path: Path):
    """G: the server's ``messaging_unavailable`` is a refusal, never an enable."""

    counter = _Counter(
        lambda request: httpx.Response(
            503, json={"error": "refused", "code": "messaging_unavailable"}
        )
    )

    check = await _discipline(counter, tmp_path).check(Boundary.APP_START)

    assert [request.method for request in counter.requests] == ["GET"]
    assert surface_projection(check)["surface_state"] == "G"
    assert _local(tmp_path).default_enrollment == {}


@pytest.mark.asyncio
async def test_the_default_enrollment_keeps_the_captured_account_guard(tmp_path: Path):
    """A sign-in switch between the status read and the consent write refuses.

    The guard is the RFC §5.2 receipt that already protects every desktop
    settings write: the account the answer was filed under must still be the
    pinned snapshot's account, and a mismatch is a local refusal with ZERO
    writes — the consent must never be sent as somebody else.
    """

    store = FakeStore(rows=[FakeRow(1, data={"access": "token-1", "account_id": "acc_1"})])

    def responder(request: httpx.Request) -> httpx.Response:
        # The sign-in switch lands AFTER the status read resolved its account
        # and BEFORE the consent write resolves its guard: the window the
        # captured-account receipt exists to close.
        store.rows = [FakeRow(9, data={"access": "token-9", "account_id": "acc_9"})]
        return _ok(_ready_payload(consent=SERVED_CONSENT))

    counter = _Counter(responder)

    check = await _discipline(counter, tmp_path, store=store).check(Boundary.APP_START)

    assert [request.method for request in counter.requests] == ["GET"]
    assert surface_projection(check)["surface_state"] == "B"
    assert _local(tmp_path).default_enrollment == {}


@pytest.mark.asyncio
async def test_a_failed_consent_write_is_retried_and_only_then_recorded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """A refusal records NOTHING; the next fresh read performs the one write."""

    puts = 0

    def responder(request: httpx.Request) -> httpx.Response:
        nonlocal puts
        if request.method == "PUT":
            puts += 1
            if puts == 1:
                return httpx.Response(
                    503, json={"error": "refused", "code": "messaging_unavailable"}
                )
            return _ok({"endpoint": None, "settings": {}})
        return _ok(_ready_payload(consent=SERVED_CONSENT))

    counter = _Counter(responder)
    discipline = _discipline(counter, tmp_path)
    await discipline.check(Boundary.APP_START)
    assert puts == 1
    assert _local(tmp_path).default_enrollment == {}

    monkeypatch.setattr(tb_client, "STATUS_CACHE_TTL_S", 0.0)
    await discipline.check(Boundary.SURFACE_OPEN)

    assert puts == 2
    assert _local(tmp_path).default_enrollment == {"acc_1": "done"}


@pytest.mark.asyncio
async def test_the_default_enrollment_marker_is_per_account(tmp_path: Path):
    """A second signed-in account still gets its own default consent write."""

    puts: list[dict[str, Any]] = []

    def responder(request: httpx.Request) -> httpx.Response:
        if request.method == "PUT":
            puts.append(json.loads(request.content))
            return _ok({"endpoint": {"id": "bep_1", "state": "active"}, "settings": {}})
        return _ok(_ready_payload(consent=SERVED_CONSENT))

    first = await _discipline(_Counter(responder), tmp_path).check(Boundary.APP_START)
    assert surface_projection(first)["surface_state"] == "B"
    second = await _discipline(
        _Counter(responder),
        tmp_path,
        store=FakeStore(rows=[FakeRow(2, data={"access": "token-2", "account_id": "acc_2"})]),
    ).check(Boundary.APP_START)

    assert surface_projection(second)["surface_state"] == "B"
    assert len(puts) == 2
    assert _local(tmp_path).default_enrollment == {"acc_1": "done", "acc_2": "done"}


@pytest.mark.asyncio
async def test_an_active_endpoint_without_a_local_receipt_is_repaired(tmp_path: Path):
    """C with no native enrollment (enabled elsewhere) commits it, no PUT.

    The repair is driven by server truth on a FRESH answer, which is what makes
    a failed local commit recoverable: the next answer is C, where a B-only
    auto-enroll would never run again.
    """

    _set_local(tmp_path, session="sess_cos")
    counter = _Counter(lambda request: _ok(_endpoint_payload("active")))

    check = await _discipline(counter, tmp_path).check(Boundary.APP_START)

    assert [request.method for request in counter.requests] == ["GET"]
    assert surface_projection(check)["surface_state"] == "C"
    from local_operator.team_bridge.host import bridge_runtime_for_session
    from local_operator.team_bridge.store import Store

    owner = Store(tmp_path).active_owner()
    assert owner is not None
    assert owner.endpoint_id == "bep_1"
    assert (
        bridge_runtime_for_session(config_manager=ConfigManager(tmp_path), session_id="sess_cos")
        is not None
    )
    # The marker records only the B-time PUT, not the repair.
    assert _local(tmp_path).default_enrollment == {}


@pytest.mark.asyncio
async def test_choosing_a_session_after_auto_enrollment_makes_it_wakeable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """M1 direction 1: enrol silently, choose the CoS session later.

    The default path must not freeze the designation at first commit: a later
    ``team_bridge.session`` write reaches the native record on the next fresh
    status read, and the host then builds the runtime for THAT session.
    """

    from local_operator.team_bridge.host import bridge_runtime_for_session
    from local_operator.team_bridge.store import Store

    payloads = [_ready_payload(consent=SERVED_CONSENT)]

    def responder(request: httpx.Request) -> httpx.Response:
        if request.method == "PUT":
            return _ok({"endpoint": {"id": "bep_1", "state": "active"}, "settings": {}})
        return _ok(payloads.pop(0) if payloads else _endpoint_payload("active"))

    discipline = _discipline(_Counter(responder), tmp_path)
    await discipline.check(Boundary.APP_START)

    store = Store(tmp_path)
    owner = store.active_owner()
    assert owner is not None
    assert store.designated_session_for(owner) is None
    assert (
        bridge_runtime_for_session(config_manager=ConfigManager(tmp_path), session_id="sess_x")
        is None
    )

    _set_local(tmp_path, session="sess_x")
    monkeypatch.setattr(tb_client, "STATUS_CACHE_TTL_S", 0.0)
    await discipline.check(Boundary.SURFACE_OPEN)

    assert store.designated_session_for(owner) == "sess_x"
    assert (
        bridge_runtime_for_session(config_manager=ConfigManager(tmp_path), session_id="sess_x")
        is not None
    )


@pytest.mark.asyncio
async def test_clearing_the_session_withdraws_wake_consent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """M1 direction 2: clearing the row is a WITHdrawal, not a no-op.

    The native designation must go back to NULL, the host runtime must close,
    and the wake producer fence must answer ``no_designation`` (badge-only).
    """

    from local_operator.providers.auth_store import OAuthAccess
    from local_operator.team_bridge.connector.wake import _producer_fence
    from local_operator.team_bridge.host import bridge_runtime_for_session
    from local_operator.team_bridge.store import Store

    _set_local(tmp_path, session="sess_cos")
    payloads = [_ready_payload(consent=SERVED_CONSENT)]

    def responder(request: httpx.Request) -> httpx.Response:
        if request.method == "PUT":
            return _ok({"endpoint": {"id": "bep_1", "state": "active"}, "settings": {}})
        return _ok(payloads.pop(0) if payloads else _endpoint_payload("active"))

    discipline = _discipline(_Counter(responder), tmp_path)
    await discipline.check(Boundary.APP_START)

    store = Store(tmp_path)
    owner = store.active_owner()
    assert owner is not None
    assert store.designated_session_for(owner) == "sess_cos"
    assert (
        bridge_runtime_for_session(config_manager=ConfigManager(tmp_path), session_id="sess_cos")
        is not None
    )

    _set_local(tmp_path, session="")
    monkeypatch.setattr(tb_client, "STATUS_CACHE_TTL_S", 0.0)
    await discipline.check(Boundary.SURFACE_OPEN)

    assert store.designated_session_for(owner) is None
    assert (
        bridge_runtime_for_session(config_manager=ConfigManager(tmp_path), session_id="sess_cos")
        is None
    )
    access = OAuthAccess(
        access_token="token-1", credential_id=owner.credential_id, account_id="acc_1"
    )
    assert _producer_fence(store, owner, access, None) == "no_designation"
