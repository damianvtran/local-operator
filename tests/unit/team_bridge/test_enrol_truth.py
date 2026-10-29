"""D1 "silent enrolment": every enable says whether THIS machine is enrolled.

WHY THIS FILE EXISTS. Before this fix a ``state=active`` settings write (and
the default-on ``POST /messaging/enrollment``) could answer 200 with an ACTIVE
hub endpoint while committing no local owner, and say nothing: the device then
showed "on / can receive" and never polled. The diagnosis
(``tb-postfix-defects.md`` § D1) found five such paths:

(a) a consented enable with no captured-account guard (Python API only);
(b) a consented enable with no tenant (desktop route only);
(c) a resume of an existing endpoint with no consent echo (the shipped UI);
(d) a commit that raised and was swallowed;
(e) another owner already committed on this machine.

Each case below pins the honest answer: a local refusal BEFORE any I/O for the
caller bugs (a, b), a commit when the hub already holds this member's consent
(c), and a coded ``local_enrollment`` outcome for everything else (c without a
record, d, e). Default-on adds the recovery: a failed local commit is retried
commit-only at a later boundary, seeded at the PRE-POST head.

Everything runs on ``tmp_path`` with ``httpx.MockTransport``: no hub, no
operator store, no network.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_operator.team_bridge import client as tb_client
from local_operator.team_bridge.client import TeamBridgeClient
from local_operator.team_bridge.store import OwnerReceipt, OwnerSnapshot, Store
from tests.unit.team_bridge.test_client import FakeStore, _ok

pytestmark = pytest.mark.asyncio

HUB = "https://hub.example/v1"


@pytest.fixture(autouse=True)
def _destination_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tb_client, "org_oauth_destination_allowed", lambda base: True)


def _client(handler, config_dir: Path | None) -> TeamBridgeClient:
    return TeamBridgeClient(
        config_dir, base_url=HUB, store=FakeStore(), transport=httpx.MockTransport(handler)
    )


def _answer(consent: dict[str, Any] | None = None, *, tenant: str = "tnt_1") -> dict[str, Any]:
    endpoint: dict[str, Any] = {"id": "bep_1", "state": "active", "tenant_id": tenant}
    if consent is not None:
        endpoint["consent"] = consent
    return {"endpoint": endpoint, "settings": {"tenant_id": tenant}}


def _hub(answer: dict[str, Any], requests: list[httpx.Request], *, head: Any = None):
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/messaging/events"):
            if head is None:
                return httpx.Response(403, json={"error": "no", "code": "bridge_not_available"})
            return _ok({"events": [], "cursor": head})
        return _ok(answer)

    return handler


CONSENT_ON_RECORD = {"version": "bridge_consent_v1", "consented_via": "dialog", "current": True}


def _settings_puts(requests: list[httpx.Request]) -> list[httpx.Request]:
    return [r for r in requests if r.method == "PUT" and r.url.path.endswith("/settings")]


# -- (a) and (b): caller bugs refused BEFORE any I/O ---------------------------


async def test_a_guardless_consented_enable_is_refused_before_any_io(tmp_path: Path):
    requests: list[httpx.Request] = []
    client = _client(_hub(_answer(), requests), tmp_path)
    with pytest.raises(ValueError, match="expected_account_id"):
        await client.put_settings(
            state="active",
            tenant_id="tnt_1",
            consent_version="bridge_consent_v1",
            consented_via="settings",
            endpoint=None,
        )
    assert requests == [] and Store(tmp_path).committed_owner_count() == 0


async def test_b_tenantless_consented_enable_is_refused_before_any_io(tmp_path: Path):
    requests: list[httpx.Request] = []
    client = _client(_hub(_answer(), requests), tmp_path)
    with pytest.raises(ValueError, match="tenant_id"):
        await client.put_settings(
            state="active",
            consent_version="bridge_consent_v1",
            consented_via="settings",
            endpoint=None,
            expected_account_id="acc_1",
        )
    assert requests == [] and Store(tmp_path).committed_owner_count() == 0


async def test_b_the_route_maps_the_tenant_refusal_to_a_422(tmp_path: Path, monkeypatch):
    """The real route over the real client: no stub between them."""

    from fastapi import HTTPException

    from local_operator.server.routes import desktop_team_bridge as route

    requests: list[httpx.Request] = []
    real = TeamBridgeClient

    def _real_over_mock(config_dir, *a, **k):
        return real(
            config_dir,
            base_url=HUB,
            store=FakeStore(),
            transport=httpx.MockTransport(_hub(_answer(), requests)),
        )

    monkeypatch.setattr(tb_client, "TeamBridgeClient", _real_over_mock)

    class _CM:
        config_dir = tmp_path

    body = route.TeamBridgeSettingsPut(
        state="active", consent_version="bridge_consent_v1", expected_account_id="acc_1"
    )
    monkeypatch.setattr(route, "_refuse_write_scope", lambda request: None)
    with pytest.raises(HTTPException) as info:
        await route.put_settings(
            body=body, request=None, config_manager=_CM()  # type: ignore[arg-type]
        )
    assert info.value.status_code == 422
    detail: Any = info.value.detail
    assert detail["code"] == "invalid_request"
    assert "tenant_id" in detail["message"]
    assert requests == []


# -- (c): resume of an existing endpoint, no consent echo -----------------------


async def test_c_resume_commits_when_the_hub_holds_this_members_consent(tmp_path: Path):
    """The shipped desktop shape: the UI omits the echo because the hub says the
    consent still binds. The hub's own record is the consent, so the device is
    bound under the version that record cites, and the answer says so."""

    requests: list[httpx.Request] = []
    client = _client(_hub(_answer(CONSENT_ON_RECORD), requests, head=412), tmp_path)
    result = await client.put_settings(
        state="active",
        tenant_id="tnt_1",
        endpoint={"state": "paused"},
        expected_account_id="acc_1",
    )
    assert result["local_enrollment"] == {"committed": True, "code": "committed"}
    store = Store(tmp_path)
    owner = store.active_owner()
    assert owner is not None and (owner.account_id, owner.endpoint_id) == ("acc_1", "bep_1")
    assert store.owner_consent_receipt(owner) == {
        "consent_version": "bridge_consent_v1",
        "consent_ref": "bep_1",
    }
    # A reused endpoint: seeded at the head read BEFORE the write, never 0
    # (which would replay up to 7 days of feed as fresh wakes).
    assert int(store.sync_state(owner)["feed_seq"]) == 412
    events = [r for r in requests if r.url.path.endswith("/messaging/events")]
    assert requests.index(events[0]) < requests.index(_settings_puts(requests)[0])


@pytest.mark.parametrize(
    "consent",
    [
        pytest.param(None, id="no-consent-block"),
        pytest.param({**CONSENT_ON_RECORD, "current": False}, id="consent-no-longer-current"),
        pytest.param({**CONSENT_ON_RECORD, "version": "bridge_consent_v9"}, id="unknown-version"),
    ],
)
async def test_c_resume_without_consent_on_record_says_so(tmp_path: Path, consent):
    requests: list[httpx.Request] = []
    client = _client(_hub(_answer(consent), requests), tmp_path)
    result = await client.put_settings(
        state="active",
        tenant_id="tnt_1",
        endpoint={"state": "paused"},
        expected_account_id="acc_1",
    )
    assert result["endpoint"]["state"] == "active"
    assert result["local_enrollment"] == {"committed": False, "code": "no_consent_echo"}
    assert Store(tmp_path).committed_owner_count() == 0


async def test_c_resume_without_a_tenant_says_so(tmp_path: Path):
    requests: list[httpx.Request] = []
    client = _client(_hub(_answer(CONSENT_ON_RECORD), requests), tmp_path)
    result = await client.put_settings(
        state="active", endpoint={"state": "paused"}, expected_account_id="acc_1"
    )
    assert result["local_enrollment"] == {"committed": False, "code": "no_tenant"}
    assert Store(tmp_path).committed_owner_count() == 0


async def test_an_answer_for_another_tenant_is_refused_as_a_mismatch(tmp_path: Path):
    requests: list[httpx.Request] = []
    client = _client(_hub(_answer(tenant="tnt_other"), requests), tmp_path)
    result = await client.put_settings(
        state="active",
        tenant_id="tnt_1",
        consent_version="bridge_consent_v1",
        consented_via="settings",
        endpoint=None,
        expected_account_id="acc_1",
    )
    assert result["local_enrollment"] == {"committed": False, "code": "tenant_mismatch"}
    assert Store(tmp_path).committed_owner_count() == 0


# -- (d): a commit that raised is a coded outcome, not a swallowed 200 ----------


async def test_d_a_failed_commit_is_reported_and_logged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    from local_operator.team_bridge.connector import enrollment as enrollment_mod

    def boom(**kwargs):
        raise OSError("database is locked: /secret/path")

    monkeypatch.setattr(enrollment_mod, "commit_settings_enrollment", boom)
    requests: list[httpx.Request] = []
    client = _client(_hub(_answer(), requests), tmp_path)
    with caplog.at_level("WARNING", logger="local_operator.team_bridge.client"):
        result = await client.put_settings(
            state="active",
            tenant_id="tnt_1",
            consent_version="bridge_consent_v1",
            consented_via="settings",
            endpoint=None,
            expected_account_id="acc_1",
        )
    assert result["local_enrollment"] == {"committed": False, "code": "store_failed"}
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert "store_failed" in logged
    # Content-free: the exception's prose (paths, ids) never reaches the log.
    assert "/secret/path" not in logged


# -- (e): another owner already holds this machine ------------------------------


def _commit(tmp_path: Path, *, account: str, endpoint: str) -> None:
    Store(tmp_path).commit_enrollment(
        OwnerSnapshot(
            credential_id=1,
            account_id=account,
            tenant_id="tnt_1",
            endpoint_id=endpoint,
            canonical_base_url=HUB,
            enrollment_generation=1,
        ),
        OwnerReceipt("bridge_consent_v1", endpoint),
        cos_session_id=None,
    )


async def test_e_another_owner_is_reported_not_silently_kept(tmp_path: Path):
    _commit(tmp_path, account="acc_other", endpoint="bep_other")
    requests: list[httpx.Request] = []
    client = _client(_hub(_answer(), requests), tmp_path)
    result = await client.put_settings(
        state="active",
        tenant_id="tnt_1",
        consent_version="bridge_consent_v1",
        consented_via="settings",
        endpoint=None,
        expected_account_id="acc_1",
    )
    assert result["local_enrollment"] == {"committed": False, "code": "other_owner"}
    owner = Store(tmp_path).active_owner()
    assert owner is not None and owner.account_id == "acc_other"


async def test_e_this_members_own_owner_reads_as_already_enrolled(tmp_path: Path):
    _commit(tmp_path, account="acc_1", endpoint="bep_1")
    requests: list[httpx.Request] = []
    client = _client(_hub(_answer(), requests), tmp_path)
    result = await client.put_settings(
        state="active", tenant_id="tnt_1", endpoint={"state": "active"}, expected_account_id="acc_1"
    )
    assert result["local_enrollment"] == {"committed": True, "code": "already_owned"}


async def test_a_pause_carries_no_enrolment_claim(tmp_path: Path):
    requests: list[httpx.Request] = []
    answer = {"endpoint": {"id": "bep_1", "state": "paused"}, "settings": {"tenant_id": "tnt_1"}}
    client = _client(_hub(answer, requests), tmp_path)
    result = await client.put_settings(
        state="paused", tenant_id="tnt_1", endpoint={"state": "active"}, expected_account_id="acc_1"
    )
    assert "local_enrollment" not in result


# -- the status route: can_receive needs a local owner --------------------------


async def _route_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    from local_operator.server.routes import desktop_team_bridge as route
    from tests.unit.server.test_desktop_team_bridge import _status_check

    check = _status_check()
    payload = json.loads(json.dumps(check.payload))
    payload["tenants"][0]["endpoint"] = {"id": "bep_1", "state": "active"}
    check = replace(check, payload=payload)

    class _Discipline:
        async def check(self, boundary):
            return check

    monkeypatch.setattr(tb_client, "get_status_discipline", lambda *a, **k: _Discipline())

    class _CM:
        config_dir = tmp_path

    response = await route.status(
        boundary="app_start", config_manager=_CM()  # type: ignore[arg-type]
    )
    return response.result  # type: ignore[attr-defined]


async def test_status_does_not_claim_can_receive_without_a_local_owner(tmp_path, monkeypatch):
    result = await _route_status(tmp_path, monkeypatch)
    assert result["surface_state"] == "C"
    assert result["local"]["enrolled"] is False
    assert result["can_receive"] is False, "can_receive=True with zero local enrollments"


async def test_status_can_receive_with_this_members_owner(tmp_path, monkeypatch):
    from tests.unit.team_bridge.test_discipline import _set_local

    _commit(tmp_path, account="acc_1", endpoint="bep_1")
    # The machine an enable or resume leaves behind: the commit writes the switch
    # too (R1-1/R2-1), so a receiving device has it on.
    _set_local(tmp_path, enabled=True)
    result = await _route_status(tmp_path, monkeypatch)
    assert result["local"]["enrolled"] is True
    assert result["can_receive"] is True


async def test_status_another_endpoints_owner_is_not_this_enrolment(tmp_path, monkeypatch):
    _commit(tmp_path, account="acc_1", endpoint="bep_elsewhere")
    result = await _route_status(tmp_path, monkeypatch)
    assert result["local"]["enrolled"] is False
    assert result["can_receive"] is False


# -- default-on: a failed local commit is recovered, commit-only ----------------


def _head_advancing_hub(**kwargs: Any):
    """The default-on ``Hub`` with two extra routes: ``/messaging/events`` is
    refused before the endpoint exists (brand-new endpoint, F2: head ``None``
    -> cursor 0) and answers a NON-ZERO head afterwards, i.e. messages arrived
    while the device could not receive them (R2-1); ``GET /messaging/settings``
    answers the full endpoint (with its consent record), as the real hub does."""

    from tests.unit.team_bridge.test_default_on import Hub

    class _HeadAdvancingHub(Hub):
        def __call__(self, request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path.endswith("/messaging/events"):
                self.requests.append(request)
                if self.endpoint is None:
                    return httpx.Response(403, json={"error": "no", "code": "bridge_not_available"})
                return _ok({"events": [], "cursor": 57})
            if request.method == "GET" and path.endswith("/messaging/settings"):
                self.requests.append(request)
                return _ok({"endpoint": self.endpoint, "settings": {"tenant_id": "tnt_1"}})
            return super().__call__(request)

    return _HeadAdvancingHub(**kwargs)


def _flaky_commit(monkeypatch: pytest.MonkeyPatch, *, failures: int = 1) -> dict[str, int]:
    from local_operator.team_bridge.connector import enrollment as enrollment_mod

    real = enrollment_mod.commit_settings_enrollment
    calls = {"n": 0}

    def flaky(**kwargs):
        calls["n"] += 1
        if calls["n"] <= failures:
            raise OSError("transient: database is locked")
        return real(**kwargs)

    monkeypatch.setattr(enrollment_mod, "commit_settings_enrollment", flaky)
    return calls


async def test_default_on_recovers_a_failed_commit_without_a_second_post(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from local_operator.team_bridge.client import Boundary
    from tests.unit.team_bridge.test_default_on import _discipline, _enrolment_writes
    from tests.unit.team_bridge.test_discipline import _local

    _flaky_commit(monkeypatch)
    hub = _head_advancing_hub()
    discipline = _discipline(hub, tmp_path)
    await discipline.check(Boundary.APP_START)
    assert hub.endpoint is not None and hub.endpoint["state"] == "active"
    assert Store(tmp_path).committed_owner_count() == 0  # the transient failure
    # NOT "done": a done marker would lock the account out of every retry.
    assert _local(tmp_path).default_enrollment == {"acc_1": "pending:bep_1:0"}
    discipline.invalidate()
    await discipline.check(Boundary.SURFACE_OPEN)
    assert len(_enrolment_writes(hub)) == 1, "a recovery must never POST again"
    store = Store(tmp_path)
    owner = store.active_owner()
    assert owner is not None, "active endpoint, still no owner"
    # Seeded at the PRE-POST head (none: a brand-new endpoint -> 0), never at
    # the head the hub answers at retry time (57), which would skip messages.
    assert int(store.sync_state(owner)["feed_seq"]) == 0, "retry seeded past unwoken messages"
    assert store.owner_consent_receipt(owner) == {
        "consent_version": "bridge_consent_v2",
        "consent_ref": "bep_1",
    }
    assert _local(tmp_path).default_enrollment == {"acc_1": "done"}


async def test_default_on_retry_never_binds_an_endpoint_it_did_not_create(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The marker names the endpoint THIS machine's POST created. A different
    endpoint at the retry (the member turned off and another machine enabled)
    is never bound by the retry."""

    from local_operator.team_bridge.client import Boundary
    from tests.unit.team_bridge.test_default_on import _discipline

    _flaky_commit(monkeypatch)
    hub = _head_advancing_hub()
    discipline = _discipline(hub, tmp_path)
    await discipline.check(Boundary.APP_START)
    assert hub.endpoint is not None
    hub.endpoint = {**hub.endpoint, "id": "bep_2"}
    discipline.invalidate()
    await discipline.check(Boundary.SURFACE_OPEN)
    assert Store(tmp_path).committed_owner_count() == 0


async def test_default_on_retry_respects_an_explicit_local_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from local_operator.team_bridge.client import Boundary
    from tests.unit.team_bridge.test_default_on import _discipline
    from tests.unit.team_bridge.test_discipline import _set_local

    _flaky_commit(monkeypatch)
    hub = _head_advancing_hub()
    discipline = _discipline(hub, tmp_path)
    await discipline.check(Boundary.APP_START)
    _set_local(tmp_path, enabled=False)
    discipline.invalidate()
    await discipline.check(Boundary.SURFACE_OPEN)
    assert Store(tmp_path).committed_owner_count() == 0
    assert [r for r in hub.writes] == [
        r for r in hub.writes if r.url.path.endswith("/messaging/enrollment")
    ]


async def test_default_on_retry_never_trusts_a_non_default_on_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Same endpoint id, but its consent was re-collected elsewhere (dialog):
    the default-on retry is not the consent for it, so nothing commits."""

    from local_operator.team_bridge.client import Boundary
    from tests.unit.team_bridge.test_default_on import _default_on_endpoint, _discipline

    _flaky_commit(monkeypatch)
    hub = _head_advancing_hub()
    discipline = _discipline(hub, tmp_path)
    await discipline.check(Boundary.APP_START)
    hub.endpoint = _default_on_endpoint("active", "dialog")
    discipline.invalidate()
    await discipline.check(Boundary.SURFACE_OPEN)
    assert Store(tmp_path).committed_owner_count() == 0


async def test_guard_default_on_is_never_widened_into_state_c(tmp_path: Path):
    """SAFETY ARM (R2-6): an ACTIVE ``default_on``/v2 endpoint with NO pending
    marker from this machine (another machine's default-on created it) commits
    nothing and writes nothing here."""

    from local_operator.team_bridge.client import Boundary
    from tests.unit.team_bridge.test_default_on import (
        Hub,
        _default_on_endpoint,
        _discipline,
    )

    hub = Hub(endpoint=_default_on_endpoint("active", "default_on"))
    await _discipline(hub, tmp_path).check(Boundary.APP_START)
    assert Store(tmp_path).committed_owner_count() == 0
    assert hub.writes == []


# -- round 1 remediation ---------------------------------------------------------


async def test_c_resume_turns_the_local_switch_on_and_a_message_is_admitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """R1-1 / QA Q1: every receive fence reads ``team_bridge.enabled`` from DISK
    and an absent key reads OFF. A path-(c) resume on a reinstalled machine (key
    absent) committed the owner but left the switch unset, so the next message
    was held ``bridge_off`` while status said the device receives."""

    from local_operator.config import read_config_values
    from local_operator.team_bridge.connector import wake as wake_mod
    from tests.unit.team_bridge import test_connector as connector_tests
    from tests.unit.team_bridge.test_discipline import _set_local

    _set_local(tmp_path, session="sess_cos")  # a designation, switch still absent
    requests: list[httpx.Request] = []
    client = _client(_hub(_answer(CONSENT_ON_RECORD), requests, head=412), tmp_path)
    result = await client.put_settings(
        state="active", tenant_id="tnt_1", endpoint={"state": "paused"}, expected_account_id="acc_1"
    )
    assert result["local_enrollment"] == {"committed": True, "code": "committed"}

    # The real wake spine (heavy seams faked, the switch NOT pre-written) admits
    # the next message; before the fix it answered hold/bridge_off.
    owner = Store(tmp_path).active_owner()
    assert owner is not None
    monkeypatch.setattr(connector_tests, "_enable_bridge", lambda path: None)
    record, _ = connector_tests._wire_spine(monkeypatch, tmp_path)
    out = await wake_mod.handle_message_available(
        owner, "bmsg_1", config_dir=tmp_path, auth_store=None, access=connector_tests._access(owner)
    )
    assert out["action"] == "delivered", out
    assert record["screened"] == 1
    values = read_config_values(tmp_path) or {}
    assert values["team_bridge"]["enabled"] is True


async def test_c_resume_never_overrides_an_explicit_local_off(tmp_path: Path):
    from local_operator.config import read_config_values
    from tests.unit.team_bridge.test_discipline import _set_local

    _set_local(tmp_path, enabled=False)
    requests: list[httpx.Request] = []
    client = _client(_hub(_answer(CONSENT_ON_RECORD), requests), tmp_path)
    await client.put_settings(
        state="active", tenant_id="tnt_1", endpoint={"state": "paused"}, expected_account_id="acc_1"
    )
    values = read_config_values(tmp_path) or {}
    assert values["team_bridge"]["enabled"] is False


async def test_default_on_retry_requires_the_record_to_say_current(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """R1-3: an ABSENT ``current`` is not "still binds"; fail closed like path (c)."""

    from local_operator.team_bridge.client import Boundary
    from tests.unit.team_bridge.test_default_on import _discipline

    _flaky_commit(monkeypatch)
    hub = _head_advancing_hub()
    discipline = _discipline(hub, tmp_path)
    await discipline.check(Boundary.APP_START)
    assert hub.endpoint is not None
    consent = {k: v for k, v in hub.endpoint["consent"].items() if k != "current"}
    hub.endpoint = {**hub.endpoint, "consent": consent}
    discipline.invalidate()
    await discipline.check(Boundary.SURFACE_OPEN)
    assert Store(tmp_path).committed_owner_count() == 0


async def test_default_on_retry_runs_on_the_next_cached_check_and_is_paced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """QA Q2: the retry ran on FRESH reads only, so inside one process a failed
    commit waited out the 300 s status cache. It now runs on the next check even
    when that check is served from cache, paced to one settings GET per window."""

    from local_operator.team_bridge.client import Boundary
    from tests.unit.team_bridge.test_default_on import _discipline, _enrolment_writes

    _flaky_commit(monkeypatch, failures=2)
    hub = _head_advancing_hub()
    discipline = _discipline(hub, tmp_path)
    await discipline.check(Boundary.APP_START)  # POST; commit #1 fails

    def settings_gets() -> int:
        return sum(
            1
            for r in hub.requests
            if r.method == "GET" and r.url.path.endswith("/messaging/settings")
        )

    statuses = sum(1 for r in hub.requests if r.url.path.endswith("/messaging/status"))
    check = await discipline.check(Boundary.WINDOW_FOCUS)  # cached; retry #1 fails
    assert check.from_cache is True
    assert settings_gets() == 1
    await discipline.check(Boundary.WINDOW_FOCUS)  # cached, inside the pacing window
    assert settings_gets() == 1, "a cached check must not GET at click rate"
    assert Store(tmp_path).committed_owner_count() == 0

    monkeypatch.setitem(discipline._pending_retry_at, "acc_1", 0.0)  # window elapsed
    check = await discipline.check(Boundary.WINDOW_FOCUS)  # cached; retry #2 commits
    assert check.from_cache is True
    owner = Store(tmp_path).active_owner()
    assert owner is not None, "a pending commit waited out the status cache"
    assert int(Store(tmp_path).sync_state(owner)["feed_seq"]) == 0
    assert len(_enrolment_writes(hub)) == 1
    assert sum(1 for r in hub.requests if r.url.path.endswith("/messaging/status")) == statuses


async def test_c_resume_on_an_already_owned_machine_turns_the_switch_on(tmp_path: Path):
    """R2-1: ``already_owned`` is a COMMITTED outcome too (``_local_enrollment``
    marks both codes committed, ``_LOCAL_ENROLLMENT_COMMITTED``). A config reset,
    or a restore that keeps the owner store, reaches this with the key absent;
    gating the switch write on the ``committed`` code alone left that machine
    deaf while status said it receives."""

    from local_operator.config import read_config_values

    _commit(tmp_path, account="acc_1", endpoint="bep_1")
    requests: list[httpx.Request] = []
    client = _client(_hub(_answer(CONSENT_ON_RECORD), requests), tmp_path)
    result = await client.put_settings(
        state="active", tenant_id="tnt_1", endpoint={"state": "paused"}, expected_account_id="acc_1"
    )
    assert result["local_enrollment"] == {"committed": True, "code": "already_owned"}
    values = read_config_values(tmp_path) or {}
    assert values.get("team_bridge", {}).get("enabled") is True


async def test_status_does_not_claim_can_receive_with_an_explicit_local_off(tmp_path, monkeypatch):
    """R2-2: an owner exists, but the local switch is explicitly OFF and every
    wake fence holds ``bridge_off``. The field named for the claim must not say
    otherwise, and ``local.enabled`` beside it must agree."""

    from tests.unit.team_bridge.test_discipline import _set_local

    _commit(tmp_path, account="acc_1", endpoint="bep_1")
    _set_local(tmp_path, enabled=False)
    result = await _route_status(tmp_path, monkeypatch)
    assert result["surface_state"] == "C"
    assert result["local"]["enrolled"] is True
    assert result["local"]["enabled"] is False
    assert result["can_receive"] is False
