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
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_operator.team_bridge import availability
from local_operator.team_bridge import client as tb_client
from local_operator.team_bridge.client import (
    CONSENT_TEXT_V1,
    CONSENT_TEXT_V1_SHA256,
    CONSENT_VERSION_V1,
    SCREENING_TEMPLATE_V1,
    SCREENING_TEMPLATE_V1_SHA256,
    STATE_READY,
    STATE_SIGNED_OUT,
    STATE_UNAVAILABLE,
    STATE_UPDATE_REQUIRED,
    AcceptedTexts,
    Boundary,
    StatusCheck,
    StatusDiscipline,
    consent_for_display,
    surface_projection,
)
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
    await _discipline(counter, tmp_path).check(Boundary.APP_START)
    assert availability.section_visible("team_bridge", tmp_path) is True


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


def test_consent_for_display_prefers_the_served_text():
    served = _check(payload={"consent": {"version": "bridge_consent_v3", "text": "as served"}})
    assert consent_for_display(served) == ("bridge_consent_v3", "as served", "served")


def test_consent_for_display_falls_back_to_the_last_accepted_copy():
    from local_operator.team_bridge.client import AcceptedText

    accepted = AcceptedText(version="bridge_consent_v2", text="cached", sha256="x", accepted_at=1.0)
    cached = _check(payload={}, texts={"consent": accepted})
    assert consent_for_display(cached) == ("bridge_consent_v2", "cached", "cached")


def test_consent_for_display_uses_the_embedded_copy_only_when_unreachable():
    offline = _check(state=STATE_UNAVAILABLE, payload=None)
    assert consent_for_display(offline) == (CONSENT_VERSION_V1, CONSENT_TEXT_V1, "embedded")
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
