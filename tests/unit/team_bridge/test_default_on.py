"""Default-on (opt-out) Team Bridge enrolment under ``bridge_consent_v2``.

Everything runs the REAL :class:`StatusDiscipline` / :class:`TeamBridgeClient`
over a stateful ``httpx.MockTransport`` hub, with the REAL config document and
native store on ``tmp_path``. The paired served-v1 / served-v2 fixtures differ
ONLY in the consent version the hub serves, which is the whole point: v1 must
stay byte-identical to the consent hold (#2712), v2 enrols exactly once.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import httpx
import pytest

from local_operator.team_bridge import availability
from local_operator.team_bridge import client as tb_client
from local_operator.team_bridge.client import (
    CONSENT_VERSION_V1,
    CONSENT_VERSION_V2,
    Boundary,
    StatusDiscipline,
    default_on_in_force,
    surface_projection,
)
from local_operator.team_bridge.settings import TeamBridgeSettings
from local_operator.team_bridge.store import Store
from tests.unit.team_bridge.test_client import FakeRow, FakeStore
from tests.unit.team_bridge.test_discipline import _local, _ok, _set_local

pytestmark = pytest.mark.asyncio

V2_TEXT = "served v2 consent text"
V1_TEXT = "served v1 consent text"


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(tb_client, "org_oauth_destination_allowed", lambda base: True)
    availability.clear()
    yield
    availability.clear()


class Hub:
    """A scripted hub: serves a consent version and one enrolment route."""

    def __init__(
        self,
        *,
        version: str = CONSENT_VERSION_V2,
        accepted: Any = "default",
        endpoint: dict[str, Any] | None = None,
        eligible: bool = True,
        enroll: Callable[[httpx.Request], httpx.Response] | None = None,
    ) -> None:
        self.version = version
        # ``None`` means "the hub sends no accepted set" (a hub predating it).
        self.accepted = (
            [CONSENT_VERSION_V1, CONSENT_VERSION_V2] if accepted == "default" else accepted
        )
        self.endpoint = endpoint
        self.eligible = eligible
        self.enroll_override = enroll
        self.requests: list[httpx.Request] = []

    @property
    def writes(self) -> list[httpx.Request]:
        return [r for r in self.requests if r.method != "GET"]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if request.method == "GET" and path.endswith("/messaging/status"):
            consent: dict[str, Any] = {
                "version": self.version,
                "text": V2_TEXT if self.version == CONSENT_VERSION_V2 else V1_TEXT,
            }
            if self.accepted is not None:
                consent["accepted_versions"] = self.accepted
            return _ok(
                {
                    "api_version": "1",
                    "available": True,
                    "tenants": [
                        {
                            "tenant_id": "tnt_1",
                            "tenant_name": "Org",
                            "eligible": self.eligible,
                            "endpoint": self.endpoint,
                            "counts": {"unread": 0, "requests_pending": 0, "blocked": 0},
                        }
                    ],
                    "consent": consent,
                }
            )
        if request.method == "POST" and path.endswith("/messaging/enrollment"):
            if self.enroll_override is not None:
                return self.enroll_override(request)
            self.endpoint = _default_on_endpoint()
            return _ok({"endpoint": self.endpoint, "settings": {"tenant_id": "tnt_1"}})
        if request.method == "PUT" and path.endswith("/messaging/settings"):
            body = json.loads(request.content)
            assert self.endpoint is not None
            self.endpoint = {**self.endpoint, "state": body["state"]}
            return _ok({"endpoint": self.endpoint, "settings": {"tenant_id": "tnt_1"}})
        return httpx.Response(404, json={"error": "nope", "code": "not_found"})


def _default_on_endpoint(state: str = "active", via: str = "default_on") -> dict[str, Any]:
    return {
        "id": "bep_1",
        "state": state,
        "label": "My agent",
        "consent": {"version": CONSENT_VERSION_V2, "consented_via": via, "current": True},
    }


def _discipline(hub: Hub, root: Path, *, account: str = "acc_1") -> StatusDiscipline:
    return StatusDiscipline(
        root,
        base_url="https://hub.example/v1",
        transport=httpx.MockTransport(hub),
        store=FakeStore(rows=[FakeRow(1, data={"access": "tok", "account_id": account})]),
    )


def _enrolment_writes(hub: Hub) -> list[httpx.Request]:
    return [r for r in hub.writes if r.url.path.endswith("/messaging/enrollment")]


# -- the happy path ----------------------------------------------------------


async def test_v2_enrols_once_commits_the_v2_receipt_and_sends_no_consented_via(tmp_path: Path):
    hub = Hub()
    check = await _discipline(hub, tmp_path).check(Boundary.APP_START)

    posts = _enrolment_writes(hub)
    assert len(posts) == 1
    body = json.loads(posts[0].content)
    # The client NEVER asserts provenance: this body is the whole request.
    assert body == {"consent_version": CONSENT_VERSION_V2, "tenant_id": "tnt_1"}
    assert "consented_via" not in posts[0].content.decode()

    store = Store(tmp_path)
    owner = store.active_owner()
    assert owner is not None and (owner.account_id, owner.endpoint_id) == ("acc_1", "bep_1")
    receipt = store.owner_consent_receipt(owner)
    assert receipt is not None
    assert receipt == {"consent_version": CONSENT_VERSION_V2, "consent_ref": "bep_1"}
    # The receipt satisfies the accepted-set currency rule for the served v2
    # (scheduler._consent_receipt_is_current, #2778): with and without a set.
    from local_operator.team_bridge.scheduler import _consent_receipt_is_current as current

    assert current(receipt["consent_version"], CONSENT_VERSION_V2, ())
    assert current(
        receipt["consent_version"], CONSENT_VERSION_V2, (CONSENT_VERSION_V1, CONSENT_VERSION_V2)
    )

    local = _local(tmp_path)
    assert local.enabled is True and local.enabled_is_set
    assert local.default_enrollment == {"acc_1": "done"}
    # The adopted post-enrolment answer shows the endpoint, not B.
    assert surface_projection(check)["surface_state"] == "C"
    # Nothing but status reads + the one POST (and the bounded head read) fired.
    assert {(r.method, r.url.path.rsplit("/", 1)[-1]) for r in hub.writes} == {
        ("POST", "enrollment")
    }


async def test_a_second_fresh_read_does_not_enrol_again(tmp_path: Path):
    hub = Hub()
    discipline = _discipline(hub, tmp_path)
    await discipline.check(Boundary.APP_START)
    discipline.invalidate()
    await discipline.check(Boundary.SURFACE_OPEN)
    assert len(_enrolment_writes(hub)) == 1


# -- every gate, failing individually => ZERO writes --------------------------

GATES = [
    pytest.param(
        dict(version=CONSENT_VERSION_V1, accepted=[CONSENT_VERSION_V1]), {}, id="served-v1"
    ),
    pytest.param(dict(accepted=None), {}, id="no-accepted-set"),
    pytest.param(dict(accepted=[CONSENT_VERSION_V1]), {}, id="v2-not-in-accepted"),
    pytest.param(dict(accepted=[]), {}, id="empty-accepted-set"),
    pytest.param(dict(version="bridge_consent_v3", accepted=["bridge_consent_v3"]), {}, id="v3"),
    pytest.param(dict(eligible=False), {}, id="not-eligible-F-or-A"),
    pytest.param(
        dict(endpoint=_default_on_endpoint("active", "settings")), {}, id="B-with-endpoint-active"
    ),
    pytest.param(dict(endpoint=_default_on_endpoint("paused")), {}, id="endpoint-paused"),
    pytest.param(dict(endpoint=_default_on_endpoint("disabled")), {}, id="endpoint-disabled"),
    pytest.param({}, dict(enabled=False), id="explicit-local-off"),
    pytest.param({}, dict(marker=True), id="account-marker-present"),
]


@pytest.mark.parametrize(("hub_kwargs", "local_kwargs"), GATES)
async def test_each_gate_failing_alone_performs_zero_writes(
    tmp_path: Path, hub_kwargs: dict[str, Any], local_kwargs: dict[str, Any]
):
    if "enabled" in local_kwargs:
        _set_local(tmp_path, enabled=local_kwargs["enabled"])
    if local_kwargs.get("marker"):
        from local_operator.team_bridge.settings import record_default_enrollment

        record_default_enrollment(tmp_path, "acc_1")
    hub = Hub(**hub_kwargs)

    await _discipline(hub, tmp_path).check(Boundary.APP_START)

    assert hub.writes == []
    assert Store(tmp_path).committed_owner_count() == 0


async def test_a_machine_that_already_holds_an_owner_does_not_ask_the_hub(tmp_path: Path):
    """No local commit is possible, so no hub endpoint is created either."""
    from local_operator.team_bridge.store import OwnerReceipt
    from local_operator.team_bridge.types import OwnerSnapshot

    Store(tmp_path).commit_enrollment(
        OwnerSnapshot(
            credential_id=9,
            account_id="acc_other",
            tenant_id="tnt_9",
            endpoint_id="bep_9",
            canonical_base_url="https://hub.example/v1",
            enrollment_generation=1,
        ).validated(),
        OwnerReceipt(consent_version=CONSENT_VERSION_V1, consent_ref="bep_9"),
    )
    hub = Hub()
    await _discipline(hub, tmp_path).check(Boundary.APP_START)
    assert hub.writes == []


async def test_a_signed_out_machine_makes_no_requests(tmp_path: Path):
    hub = Hub()
    discipline = StatusDiscipline(
        tmp_path,
        base_url="https://hub.example/v1",
        transport=httpx.MockTransport(hub),
        store=FakeStore(rows=[]),
    )
    await discipline.check(Boundary.APP_START)
    assert hub.requests == []


async def test_a_login_switch_between_read_and_write_is_refused_locally(tmp_path: Path):
    """Gate 5: the write is pinned to the account the answer was read for."""
    hub = Hub()
    fake = FakeStore(
        rows=[FakeRow(1, data={"access": "tok", "account_id": "acc_1"})],
        switch_to=[FakeRow(2, data={"access": "tok2", "account_id": "acc_2"})],
        switch_after=2,  # the read resolves acc_1; the pinned write sees acc_2
    )
    discipline = StatusDiscipline(
        tmp_path,
        base_url="https://hub.example/v1",
        transport=httpx.MockTransport(hub),
        store=fake,
    )
    await discipline.check(Boundary.APP_START)
    assert _enrolment_writes(hub) == []
    assert Store(tmp_path).committed_owner_count() == 0


# -- silence: older hub, 409, transport ---------------------------------------


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param(
            lambda r: httpx.Response(404, json={"error": "x", "code": "not_found"}), id="404"
        ),
        pytest.param(
            lambda r: httpx.Response(405, json={"error": "x", "code": "bad_method"}), id="405"
        ),
        pytest.param(
            lambda r: httpx.Response(
                409,
                json={
                    "error": "x",
                    "code": "consent_version_unknown",
                    "details": {"current_version": "bridge_consent_v1"},
                },
            ),
            id="409",
        ),
        pytest.param(
            lambda r: httpx.Response(503, json={"error": "x", "code": "messaging_unavailable"}),
            id="503",
        ),
    ],
)
async def test_hub_refusals_leave_no_state_and_no_retry_loop(tmp_path: Path, answer):
    hub = Hub(enroll=answer)
    discipline = _discipline(hub, tmp_path)

    first = await discipline.check(Boundary.APP_START)
    discipline.invalidate()
    second = await discipline.check(Boundary.SURFACE_OPEN)

    assert len(_enrolment_writes(hub)) == 1  # debounced: never a loop
    for check in (first, second):
        assert check.error is None and check.state == "ready"
        assert surface_projection(check)["surface_state"] == "B"
    assert Store(tmp_path).committed_owner_count() == 0
    assert _local(tmp_path) == TeamBridgeSettings()


async def test_a_transport_failure_is_silent(tmp_path: Path):
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    hub = Hub(enroll=boom)
    check = await _discipline(hub, tmp_path).check(Boundary.APP_START)
    assert check.error is None
    assert surface_projection(check)["surface_state"] == "B"
    assert Store(tmp_path).committed_owner_count() == 0


async def test_a_hub_noop_commits_nothing(tmp_path: Path):
    """The hub answers 200 with the EXISTING (dialog) endpoint: nothing created here."""
    existing = _default_on_endpoint("active", "dialog")
    hub = Hub(enroll=lambda r: _ok({"endpoint": existing, "settings": {}}))
    await _discipline(hub, tmp_path).check(Boundary.APP_START)
    assert Store(tmp_path).committed_owner_count() == 0
    assert _local(tmp_path).default_enrollment == {}


# -- v1 is byte-identical to the hold -----------------------------------------


async def test_served_v1_and_served_v2_pair(tmp_path: Path):
    v1_root, v2_root = tmp_path / "v1", tmp_path / "v2"
    v1_root.mkdir()
    v2_root.mkdir()
    v1 = Hub(version=CONSENT_VERSION_V1, accepted=[CONSENT_VERSION_V1])
    v2 = Hub()
    await _discipline(v1, v1_root).check(Boundary.APP_START)
    await _discipline(v2, v2_root).check(Boundary.APP_START)

    assert [r.method for r in v1.requests] == ["GET"]  # the hold, unchanged
    assert not (v1_root / "config.yml").exists() or _local(v1_root) == TeamBridgeSettings()
    assert Store(v1_root).committed_owner_count() == 0
    assert len(_enrolment_writes(v2)) == 1
    assert Store(v2_root).committed_owner_count() == 1


def test_the_composed_default_follows_the_served_consent_only_when_absent():
    absent = TeamBridgeSettings.from_values({})
    assert absent.enabled is False  # DEFAULT_TEAM_BRIDGE_ENABLED is unchanged
    assert absent.enabled_given(False) is False
    assert absent.enabled_given(True) is True
    off = TeamBridgeSettings.from_values({"team_bridge": {"enabled": False}})
    on = TeamBridgeSettings.from_values({"team_bridge": {"enabled": True}})
    assert off.enabled_given(True) is False
    assert on.enabled_given(False) is True
    # A present-but-unreadable value fails closed even under a served v2.
    junk = TeamBridgeSettings.from_values({"team_bridge": {"enabled": None}})
    assert junk.enabled_given(True) is False


async def test_enabled_by_default_alone_starts_no_work_for_an_unenrolled_account(tmp_path: Path):
    """Absent key + v2 served + NO committed owner: every offline consumer is still off."""
    from local_operator.config import ConfigManager
    from local_operator.team_bridge.authority import authority_fence
    from local_operator.team_bridge.host import bridge_runtime_for_session

    hub = Hub(enroll=lambda r: httpx.Response(404, json={"error": "x", "code": "not_found"}))
    await _discipline(hub, tmp_path).check(Boundary.APP_START)
    runtime = bridge_runtime_for_session(config_manager=ConfigManager(tmp_path), session_id="s")
    assert runtime is None

    class _Runtime:
        config_dir = tmp_path
        owner = None

    assert authority_fence(_Runtime()) is not None


def test_default_on_in_force_requires_the_exact_accepted_pair():
    def check_for(version: str, accepted: Any):
        from local_operator.team_bridge.client import AcceptedTexts, StatusCheck

        texts = AcceptedTexts()
        consent = {"version": version, "text": "t"}
        if accepted is not None:
            consent["accepted_versions"] = accepted
        payload = {
            "api_version": "1",
            "tenants": [{"tenant_id": "t", "eligible": True, "endpoint": None}],
            "consent": consent,
        }
        texts.accept("consent", version=version, text="t")
        return StatusCheck(
            state="ready",
            boundary="x",
            payload=payload,
            from_cache=False,
            fetched_at=1.0,
            checked_at=1.0,
            error=None,
            account_id="acc_1",
            texts=texts.snapshot(),
        )

    assert default_on_in_force(check_for(CONSENT_VERSION_V2, [CONSENT_VERSION_V2])) is True
    assert default_on_in_force(check_for(CONSENT_VERSION_V2, None)) is False
    assert default_on_in_force(check_for(CONSENT_VERSION_V2, "bridge_consent_v2")) is False
    assert default_on_in_force(check_for(CONSENT_VERSION_V2, [CONSENT_VERSION_V1])) is False
    served_v1_both = check_for(CONSENT_VERSION_V1, [CONSENT_VERSION_V1, CONSENT_VERSION_V2])
    assert default_on_in_force(served_v1_both) is False


# -- Turn off: sticky across restart, refresh and re-sign-in ------------------


async def test_turn_off_is_sticky_across_restart_refresh_and_resignin(tmp_path: Path):
    hub = Hub()
    discipline = _discipline(hub, tmp_path)
    await discipline.check(Boundary.APP_START)
    assert len(_enrolment_writes(hub)) == 1

    client = tb_client.TeamBridgeClient(
        tmp_path,
        base_url="https://hub.example/v1",
        transport=httpx.MockTransport(hub),
        store=FakeStore(rows=[FakeRow(1, data={"access": "tok", "account_id": "acc_1"})]),
    )
    result = await client.put_settings(
        state="disabled", tenant_id="tnt_1", expected_account_id="acc_1"
    )
    assert result["endpoint"]["state"] == "disabled"
    puts = [r for r in hub.writes if r.method == "PUT"]
    assert puts
    assert json.loads(puts[-1].content) == {"state": "disabled", "tenant_id": "tnt_1"}

    local = _local(tmp_path)
    assert local.enabled is False and local.enabled_is_set
    assert local.default_enrollment == {"acc_1": "done"}

    # Restart-like: a FRESH discipline, status refresh, and a re-sign-in as the
    # same account; and even a hub that (wrongly) reports no endpoint again.
    hub.endpoint = None
    before = len(hub.writes)
    for _ in range(2):
        fresh = _discipline(hub, tmp_path)
        await fresh.check(Boundary.APP_START)
        fresh.invalidate()
        await fresh.check(Boundary.SURFACE_OPEN)
    assert len(hub.writes) == before
    assert len(_enrolment_writes(hub)) == 1


async def test_removing_only_the_explicit_key_still_does_not_re_enrol(tmp_path: Path):
    """The per-account marker is the second layer of stickiness."""
    from local_operator.team_bridge.settings import record_default_enrollment

    record_default_enrollment(tmp_path, "acc_1")
    hub = Hub()
    await _discipline(hub, tmp_path).check(Boundary.APP_START)
    assert hub.writes == []


async def test_enroll_default_on_body_never_carries_consented_via(tmp_path: Path):
    seen: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            seen.append(request.content)
            return _ok({"endpoint": _default_on_endpoint(), "settings": {}})
        return httpx.Response(404, json={"error": "x", "code": "not_found"})

    client = tb_client.TeamBridgeClient(
        tmp_path,
        base_url="https://hub.example/v1",
        transport=httpx.MockTransport(handler),
        store=FakeStore(rows=[FakeRow(1, data={"access": "tok", "account_id": "acc_1"})]),
    )
    await client.enroll_default_on(tenant_id="tnt_1", expected_account_id="acc_1")
    assert seen and b"consented_via" not in seen[0]
    import inspect

    assert "consented_via" not in inspect.signature(client.enroll_default_on).parameters
