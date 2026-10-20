"""The bridge transport: owner pin, auth discipline, refusal mapping, PUT rules.

The real client is driven end to end against ``httpx.MockTransport``: every
assertion below is about bytes that would cross the wire (headers, bodies,
call counts), not about mocks standing in for the client's own logic. The
credential store is a small fake that mirrors ``AuthStore``'s surface
(``list_credentials`` / ``get_credential`` / ``ensure_oauth_fresh_or_raise``),
so the owner-pin rules run against rows, not against a resolver stub.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest

from local_operator.providers.auth_store import AuthStoreError
from local_operator.team_bridge import client as tb_client
from local_operator.team_bridge.client import (
    CLOSED_VALIDATION_RULES,
    VALIDATION_SENTENCES,
    BridgeRefusal,
    BridgeSignInRequired,
    BridgeUnreachable,
    TeamBridgeClient,
)

pytestmark = pytest.mark.asyncio


@dataclass
class FakeRow:
    id: int
    data: dict[str, Any] = field(default_factory=dict)
    provider: str = "radient"
    credential_type: str = "oauth"


class FakeStore:
    """The credential seam, shaped like the exact surface the client uses."""

    def __init__(
        self,
        rows: list[FakeRow] | None = None,
        *,
        refresh: dict[str, Any] | None = None,
        refresh_error: Exception | None = None,
        switch_to: list[FakeRow] | None = None,
        switch_after: int | None = None,
    ) -> None:
        self.rows = list(
            rows
            if rows is not None
            else [FakeRow(1, data={"access": "token-1", "account_id": "acc_1"})]
        )
        self.refresh_calls: list[tuple[int, bool]] = []
        self.resolve_requests = 0
        self.list_calls = 0
        self._refresh = refresh
        self._refresh_error = refresh_error
        self._switch_to = switch_to
        self._switch_after = switch_after

    def _maybe_switch(self) -> None:
        self.resolve_requests += 1
        if (
            self._switch_to is not None
            and self._switch_after is not None
            and self.resolve_requests > self._switch_after
        ):
            self.rows = list(self._switch_to)

    def list_credentials(self, provider: str | None = None, include_disabled: bool = False):
        self.list_calls += 1
        self._maybe_switch()
        return [r for r in self.rows if provider is None or r.provider == provider]

    def get_credential(self, credential_id: int) -> FakeRow | None:
        self._maybe_switch()
        for row in self.rows:
            if row.id == credential_id:
                return row
        return None

    async def ensure_oauth_fresh_or_raise(self, credential_id: int, *, force: bool = False):
        self.refresh_calls.append((credential_id, force))
        if force and self._refresh_error is not None:
            raise self._refresh_error
        row = self.get_credential(credential_id)
        if row is None:
            return None
        data = dict(row.data)
        # Mirrors the real store's shape for these tests: the ordinary read
        # serves the stored row; the FORCED (post-401) path applies the
        # scripted refresh and rewrites the row's data in place.
        if force and self._refresh is not None:
            data.update(self._refresh)
            row.data.update(self._refresh)
        return data


def _ok(result: Any, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json={"msg": "ok", "result": result})


def _err(status: int, code: str, *, details: dict[str, Any] | None = None) -> httpx.Response:
    body: dict[str, Any] = {"error": "refused", "code": code}
    if details:
        body["details"] = details
    return httpx.Response(status, json=body)


def _make_client(
    handler, *, store: Any = None, credential_id: int | None = None
) -> TeamBridgeClient:
    return TeamBridgeClient(
        None,
        base_url="https://hub.example/v1",
        store=store or FakeStore(),
        transport=httpx.MockTransport(handler),
        credential_id=credential_id,
    )


@pytest.fixture(autouse=True)
def _destination_open(monkeypatch: pytest.MonkeyPatch) -> None:
    # resolve_radient_oauth_access is deliberately no longer imported here: the
    # client resolves its own exact row (see the module docstring), so its
    # tests provision a store rather than patching the cascade resolver.
    monkeypatch.setattr(tb_client, "org_oauth_destination_allowed", lambda base: True)


# ---------------------------------------------------------------------------
# the owner pin (A1)
# ---------------------------------------------------------------------------


async def test_no_login_surfaces_without_any_request():
    """No OAuth row: §13.2's sign-in state, and ZERO traffic."""

    requests: list[httpx.Request] = []
    client = _make_client(lambda request: requests.append(request), store=FakeStore(rows=[]))

    with pytest.raises(BridgeSignInRequired):
        await client.status()
    assert requests == []


async def test_an_application_key_is_not_a_bridge_credential():
    """A key-only login proves a tenant, not the person — never a fallback."""

    key_row = FakeRow(1, data={"api_key": "rad-x"}, credential_type="api_key")
    requests: list[httpx.Request] = []
    client = _make_client(lambda request: requests.append(request), store=FakeStore(rows=[key_row]))

    with pytest.raises(BridgeSignInRequired):
        await client.status()
    assert requests == []


async def test_several_oauth_rows_refuse_rather_than_pick_a_mailbox():
    rows = [
        FakeRow(1, data={"access": "token-1", "account_id": "acc_1"}),
        FakeRow(2, data={"access": "token-2", "account_id": "acc_2"}),
    ]
    requests: list[httpx.Request] = []
    client = _make_client(lambda request: requests.append(request), store=FakeStore(rows=rows))

    with pytest.raises(BridgeSignInRequired):
        await client.status()
    assert requests == []


async def test_an_explicit_pin_selects_its_exact_row_among_several():
    rows = [
        FakeRow(1, data={"access": "token-1", "account_id": "acc_1"}),
        FakeRow(2, data={"access": "token-2", "account_id": "acc_2"}),
    ]
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["authorization"])
        return _ok({"available": True})

    client = _make_client(handler, store=FakeStore(rows=rows), credential_id=2)
    await client.status()
    assert seen == ["Bearer token-2"]


async def test_a_pin_pointing_at_a_removed_row_is_the_sign_in_state():
    requests: list[httpx.Request] = []
    client = _make_client(
        lambda request: requests.append(request), store=FakeStore(rows=[]), credential_id=7
    )
    with pytest.raises(BridgeSignInRequired):
        await client.status()
    assert requests == []


async def test_a_401_refreshes_the_exact_row_once_then_retries():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["authorization"])
        if len(seen) == 1:
            return httpx.Response(401, json={"error": "unauthorized"})
        return _ok({"available": True})

    store = FakeStore(refresh={"access": "token-2"})
    client = _make_client(handler, store=store)

    result = await client.status()

    assert result == {"available": True}
    assert seen == ["Bearer token-1", "Bearer token-2"]
    # ONE FORCED refresh, of the EXACT pinned row (id 1) — the expiry clock may
    # still read fresh for a bearer the server already rejected. (The ordinary
    # resolution read is not a refresh: only ``force`` entries count.)
    assert [call for call in store.refresh_calls if call[1]] == [(1, True)]


async def test_a_second_401_stops_and_surfaces_the_sign_in_state():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "no"})

    store = FakeStore(refresh={"access": "token-2"})
    client = _make_client(handler, store=store)

    with pytest.raises(BridgeSignInRequired):
        await client.status()
    # Never a second refresh / loop: exactly one FORCED refresh, then stop.
    assert [call for call in store.refresh_calls if call[1]] == [(1, True)]


async def test_a_dead_grant_during_refresh_is_terminal():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "no"})

    store = FakeStore(refresh_error=AuthStoreError("grant refused"))
    client = _make_client(handler, store=store)

    with pytest.raises(BridgeSignInRequired):
        await client.status()


async def test_an_owner_mismatch_refuses_before_the_retry():
    """A re-login replacing the account behind the same row must not re-send."""

    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["authorization"])
        if len(seen) == 1:
            return httpx.Response(401, json={"error": "unauthorized"})
        return _ok({"available": True})

    store = FakeStore(refresh={"access": "token-2", "account_id": "acc_2"})
    client = _make_client(handler, store=store)

    with pytest.raises(BridgeSignInRequired):
        await client.status()
    assert seen == ["Bearer token-1"]  # the retry never left


async def test_a_refused_destination_names_the_configuration_remedy(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(tb_client, "org_oauth_destination_allowed", lambda base: False)
    requests: list[httpx.Request] = []
    client = _make_client(lambda request: requests.append(request))

    with pytest.raises(BridgeRefusal) as info:
        await client.status()

    assert info.value.code == "bridge_destination_refused"
    assert "RADIENT_ORG_ALLOW_NONCANONICAL_BASE" in info.value.message
    assert requests == []


# ---------------------------------------------------------------------------
# refusals (§6.5)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status,code,retry",
    [
        (400, "invalid_request", None),
        (403, "bridge_not_available", None),
        (403, "bridge_disabled", None),
        (403, "recipient_unavailable", None),
        (409, "screening_required", None),
        (409, "invalid_transition", None),
        (409, "consent_version_unknown", None),
        (413, "payload_too_large", None),
        (422, "invalid_tenant", None),
        (429, "rate_limited", 7),
        (503, "messaging_unavailable", None),
    ],
)
async def test_the_server_code_maps_through_verbatim(status: int, code: str, retry: int | None):
    details = {"retry_after_seconds": retry} if retry is not None else None

    def handler(request: httpx.Request) -> httpx.Response:
        return _err(status, code, details=details)

    client = _make_client(handler)
    with pytest.raises(BridgeRefusal) as info:
        await client.status()

    refusal = info.value
    assert (refusal.code, refusal.http_status) == (code, status)
    if retry is not None:
        assert refusal.retry_after_seconds == retry
    else:
        assert refusal.retry_after_seconds is None


async def test_retry_after_falls_back_to_the_header():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429, json={"error": "slow down", "code": "rate_limited"}, headers={"Retry-After": "5"}
        )

    client = _make_client(handler)
    with pytest.raises(BridgeRefusal) as info:
        await client.status()
    assert info.value.retry_after_seconds == 5


async def test_consent_version_unknown_is_a_refresh_status_signal():
    def handler(request: httpx.Request) -> httpx.Response:
        return _err(409, "consent_version_unknown")

    client = _make_client(handler)
    with pytest.raises(BridgeRefusal) as info:
        await client.put_settings(
            state="active", consent_version="bridge_consent_v1", endpoint=None
        )
    assert info.value.refreshes_status is True
    assert info.value.advice is None


async def test_transport_failure_is_its_own_class():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    client = _make_client(handler)
    with pytest.raises(BridgeUnreachable) as info:
        await client.status()
    assert info.value.code == "bridge_unreachable"
    assert info.value.http_status is None


async def test_a_2xx_without_a_result_is_not_a_success():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"msg": "ok"})

    client = _make_client(handler)
    with pytest.raises(BridgeRefusal) as info:
        await client.status()
    assert info.value.code == "bridge_invalid_response"


# The closed ``(field, rule)`` validation table, pinned against RFC §6.5. The
# rule set is CLOSED: a rule that ships undocumented, or a documented rule that
# silently disappears, fails here rather than drifting.
_EXPECTED_RULES: dict[str, set[str]] = {
    "label": {
        "empty",
        "too_long",
        "taken",
        "control_character",
        "invisible_character",
        "unsupported_space",
        "invalid_utf8",
    },
    "state": {"unknown_value", "no_endpoint", "resume_via_active"},
    "consented_via": {"unknown_value"},
    "tenant_id": {"required"},
    "body": {"malformed"},
}


def test_the_closed_validation_table_matches_the_rfc():
    assert {
        field: set(rules) for field, rules in CLOSED_VALIDATION_RULES.items()
    } == _EXPECTED_RULES


@pytest.mark.parametrize(
    "field,rule", sorted((f, r) for f, rules in _EXPECTED_RULES.items() for r in rules)
)
async def test_every_field_rule_case_renders_its_client_sentence_or_none(field: str, rule: str):
    """Every closed-table case: a user-fixable rule carries its sentence, a
    client bug carries none — never a sentence dressed up as advice."""

    def handler(request: httpx.Request) -> httpx.Response:
        return _err(400, "invalid_request", details={"field": field, "rule": rule})

    client = _make_client(handler)
    with pytest.raises(BridgeRefusal) as info:
        await client.put_settings(
            state="active", consent_version="bridge_consent_v1", endpoint=None
        )

    expected = VALIDATION_SENTENCES.get((field, rule))
    assert info.value.advice == expected

    if field == "label" and rule in {
        "control_character",
        "invisible_character",
        "unsupported_space",
        "invalid_utf8",
    }:
        # The four character-class refusals share one sentence (the RFC's table
        # repeats it): one string, four rows, no drift.
        assert expected == (
            "That name contains characters that can't be used. "
            "Use plain letters, numbers and spaces."
        )


def test_the_label_sentences_are_the_rfcs_own():
    assert VALIDATION_SENTENCES[("label", "empty")] == "Give your agent a name."
    assert VALIDATION_SENTENCES[("label", "too_long")] == "Use 64 characters or fewer."
    assert VALIDATION_SENTENCES[("label", "taken")] == (
        "Another agent in your organization already uses that name. Choose a different one."
    )


# ---------------------------------------------------------------------------
# PUT rules (§5.2)
# ---------------------------------------------------------------------------


async def test_state_is_required_on_every_request():
    client = _make_client(lambda request: _ok({}))
    with pytest.raises(TypeError):
        await client.put_settings(consent_version="v1")  # type: ignore[call-arg]


@pytest.mark.parametrize("bundled", ["paused", "disabled"])
async def test_a_pause_or_opt_out_is_sent_alone(bundled: str):
    sent: list[httpx.Request] = []
    client = _make_client(lambda request: sent.append(request) or _ok({}))

    with pytest.raises(ValueError, match="sent alone"):
        await client.put_settings(state=bundled, label="My agent")
    assert sent == []

    # The name change as its own request is fine (state=active).
    await client.put_settings(state="active", consent_version="v1", label="My agent", endpoint=None)
    assert len(sent) == 1


async def test_consent_echo_is_required_from_a_fresh_endpoint_or_from_disabled():
    sent: list[httpx.Request] = []
    client = _make_client(lambda request: sent.append(request) or _ok({}))

    with pytest.raises(ValueError, match="consent_version"):
        await client.put_settings(state="active", endpoint=None)
    with pytest.raises(ValueError, match="consent_version"):
        await client.put_settings(state="active", endpoint={"state": "disabled"})
    assert sent == []

    # A resume of a paused endpoint needs no consent...
    await client.put_settings(state="active", endpoint={"state": "paused"})
    # ...and an unknown endpoint lets the SERVER drive the 409 path.
    await client.put_settings(state="active")
    assert len(sent) == 2


async def test_consented_via_is_the_closed_set():
    sent: list[dict[str, Any]] = []
    client = _make_client(lambda request: sent.append(json.loads(request.content)) or _ok({}))

    with pytest.raises(ValueError, match="consented_via"):
        await client.put_settings(
            state="active", consent_version="v1", consented_via="email", endpoint=None
        )

    await client.put_settings(
        state="active", consent_version="v1", consented_via="dialog", endpoint=None
    )
    assert sent[-1]["consented_via"] == "dialog"


async def test_the_body_bound_is_checked_before_the_wire():
    client = _make_client(lambda request: _ok({}))
    with pytest.raises(BridgeRefusal) as info:
        await client.put_settings(
            state="active", consent_version="v1", label="x" * 5000, endpoint=None
        )
    assert info.value.code == "payload_too_large"
    assert info.value.http_status == 413


async def test_tenant_id_rides_the_body_when_given():
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return _ok({"endpoint": None, "settings": {}})

    client = _make_client(handler)
    await client.put_settings(
        state="active", consent_version="v1", tenant_id="tnt_1", endpoint=None
    )
    assert sent == [{"state": "active", "tenant_id": "tnt_1", "consent_version": "v1"}]


# ---------------------------------------------------------------------------
# allowlist (§6.1, §9)
# ---------------------------------------------------------------------------


def _entry(i: int) -> dict[str, Any]:
    return {"direction": "send", "effect": "allow", "target": {"endpoint_id": f"bep_{i}"}}


async def test_replace_set_bounds():
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return _ok({"allowlist": {"tenant_id": "tnt_1", "entries": []}})

    client = _make_client(handler)

    with pytest.raises(ValueError, match="at most 200"):
        await client.put_allowlist([_entry(i) for i in range(201)])

    await client.put_allowlist([_entry(i) for i in range(200)])
    assert len(sent[0]["entries"]) == 200

    # The clear is the empty REPLACE-SET: the key is present, deliberately.
    await client.put_allowlist([])
    assert sent[1] == {"entries": []}


@pytest.mark.parametrize(
    "entry",
    [
        {"direction": "out", "effect": "allow", "target": {"endpoint_id": "bep_1"}},
        {"direction": "send", "effect": "maybe", "target": {"endpoint_id": "bep_1"}},
        {"direction": "send", "effect": "allow", "target": {}},
        {
            "direction": "send",
            "effect": "allow",
            "target": {"endpoint_id": "bep_1", "account_id": "acc_1"},
        },
        {"direction": "send", "effect": "allow", "target": {"endpoint_id": 5}},
    ],
)
async def test_entry_shape_violations_are_local_caller_bugs(entry: dict[str, Any]):
    client = _make_client(lambda request: _ok({}))
    with pytest.raises(ValueError):
        await client.put_allowlist([entry])


def test_allowlist_view_reads_modes_and_the_s_7_cues():
    from local_operator.team_bridge.client import allowlist_view

    payload = {
        "allowlist": {
            "tenant_id": "tnt_1",
            "entries": [
                {"direction": "send", "effect": "allow", "target": {"endpoint_id": "bep_1"}},
                {"direction": "receive", "effect": "deny", "target": {"account_id": "acc_9"}},
            ],
        }
    }
    view = allowlist_view(payload)

    assert view["tenant_id"] == "tnt_1"
    # Any allow entry switches its direction to listed-only; a deny alone does
    # not (it narrows a pair, the class default still applies to others).
    assert view["directions"]["send"] == {"mode": "listed_only", "allow": 1, "deny": 0}
    assert view["directions"]["receive"] == {"mode": "class_default", "allow": 0, "deny": 1}
    assert view["class_default"] == "cos_cos"
    # The S-7 cues: adding allow restricts NEW threads; removing one does NOT
    # stop an open thread — only a deny does.
    assert view["cues"] == {
        "allow_entry_switches_direction_to_listed_only": True,
        "listed_only_affects": "new_threads",
        "removing_allow_does_not_stop_open_thread": True,
        "deny_stops_pair_both_directions": True,
    }


def test_allowlist_view_empty_is_the_class_default():
    from local_operator.team_bridge.client import allowlist_view

    view = allowlist_view({"allowlist": {"tenant_id": "tnt_1", "entries": []}})
    assert view["directions"]["send"]["mode"] == "class_default"
    assert view["directions"]["receive"]["mode"] == "class_default"


# ---------------------------------------------------------------------------
# the ONE-snapshot owner binding and the captured-account guard (round 1)
# ---------------------------------------------------------------------------


class _LateSwitchStore(FakeStore):
    """The reviewer's interleaving: a re-login lands AFTER the ensure snapshot.

    ``ensure_oauth_fresh_or_raise`` returns the row's own data mapping — token
    and identity from the SAME read. The switch below then replaces the stored
    row, so any LATER metadata read would see account B. A client that pairs
    the returned token with a later read (the pre-fix shape) would attribute
    A's bearer to B; the one-snapshot client never re-reads.
    """

    async def ensure_oauth_fresh_or_raise(self, credential_id: int, *, force: bool = False):
        data = await super().ensure_oauth_fresh_or_raise(credential_id, force=force)
        self.rows = [FakeRow(credential_id, data={"access": "token-2", "account_id": "acc_b"})]
        return data


async def test_identity_and_token_come_from_one_snapshot():
    """R1: the token and the identity are the SAME returned read."""

    store = _LateSwitchStore()
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["authorization"])
        return _ok({"available": True, "tenants": []})

    client = _make_client(handler, store=store)
    access = await client.current_access()

    assert access.account_id == "acc_1"
    assert access.access_token == "token-1"
    # The status request carried the coherent snapshot's bearer.
    await client.status(access=access)
    assert seen == ["Bearer token-1"]


async def test_a_row_without_an_owner_identity_fails_closed():
    """R1: no readable account id — sign-in state, ZERO traffic."""

    requests: list[httpx.Request] = []
    store = FakeStore(rows=[FakeRow(1, data={"access": "token-1"})])
    client = _make_client(lambda request: requests.append(request), store=store)

    with pytest.raises(BridgeSignInRequired):
        await client.status()
    assert requests == []


async def test_a_refresh_that_lost_the_expected_identity_fails_closed():
    """R1: the refresh comparison is unconditional; a changed account stops."""

    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(401, json={"error": "expired", "code": "unauthorized"})

    # The forced refresh rewrites the row to a DIFFERENT account: the retry
    # must never leave — the terminal sign-in state is surfaced instead.
    store = FakeStore(refresh={"access": "token-2", "account_id": "acc_2"})
    client = _make_client(handler, store=store)

    with pytest.raises(BridgeSignInRequired):
        await client.status()
    assert len(requests) == 1
    # The initial resolve, then ONE forced refresh of the same row — and no
    # third anything: the retry never left after the identity changed.
    assert store.refresh_calls == [(1, False), (1, True)]


async def test_a_captured_account_mismatch_refuses_with_zero_writes():
    """R6: the guard compares against the pinned snapshot before any HTTP."""

    requests: list[httpx.Request] = []
    client = _make_client(lambda request: requests.append(request) or _ok({}))

    with pytest.raises(BridgeRefusal) as info:
        await client.put_settings(state="paused", expected_account_id="acc_b")

    assert info.value.code == "account_mismatch"
    assert info.value.http_status == 409
    assert info.value.refreshes_status is True
    assert requests == []


async def test_the_captured_account_rides_the_same_snapshot_and_stays_local():
    """R6: a matching guard dispatches under the pinned snapshot, guard stripped."""

    sent: list[httpx.Request] = []
    client = _make_client(lambda request: sent.append(request) or _ok({"endpoint": None}))

    await client.put_settings(state="paused", expected_account_id="acc_1")

    assert len(sent) == 1
    assert sent[0].headers["authorization"] == "Bearer token-1"
    assert json.loads(sent[0].content) == {"state": "paused"}


async def test_the_allowlist_write_takes_the_same_guard():
    """R6: a policy surface must not be the one mutation that skips the receipt."""

    requests: list[httpx.Request] = []
    client = _make_client(lambda request: requests.append(request) or _ok({"allowlist": {}}))

    with pytest.raises(BridgeRefusal) as info:
        await client.put_allowlist([], expected_account_id="acc_b")
    assert info.value.code == "account_mismatch"
    assert requests == []

    await client.put_allowlist([], expected_account_id="acc_1")
    assert len(requests) == 1
    assert json.loads(requests[0].content) == {"entries": []}


async def test_tenant_state_reads_the_frozen_route_with_the_pinned_bearer():
    """The members-readable kill confirmation (§5.5, S-22) — one GET, no body."""

    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, request.headers["authorization"]))
        return _ok({"disabled": True, "ops_disabled": False, "effective": True})

    client = _make_client(handler)
    result = await client.tenant_state("tnt_1")

    assert result["effective"] is True
    assert seen == [("/v1/tenants/tnt_1/messaging/state", "Bearer token-1")]


# ---------------------------------------------------------------------------
# the account-scope read surface (desktop multi-org forwarding)
# ---------------------------------------------------------------------------


async def test_conversations_sends_the_scope_and_cursor_wire_names():
    """The merged page's names belong to the other repo; pin the seam here."""

    seen: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url)
        return _ok({"conversations": [], "cursor": None})

    client = _make_client(handler)
    await client.conversations(scope="account", cursor="bcur_x", limit=20)

    assert len(seen) == 1
    assert seen[0].path == "/v1/messaging/conversations"
    assert dict(seen[0].params) == {
        "limit": "20",
        "cursor": "bcur_x",
        "scope": "account",
    }


async def test_unread_sends_the_scope_wire_name():
    """The badge's aggregate is one GET; a dropped name would read one org."""

    seen: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url)
        return _ok({"total": 0, "organizations": []})

    client = _make_client(handler)
    await client.unread(scope="account")

    assert len(seen) == 1
    assert seen[0].path == "/v1/messaging/unread"
    assert dict(seen[0].params) == {"scope": "account"}


# ---------------------------------------------------------------------------
# agent actors, discovery and notification policy (contract §10.1)
# ---------------------------------------------------------------------------


async def test_create_conversation_sends_the_agent_actor_wire_names():
    """The actor fields are the server PR's names; a rename would act human."""

    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _ok({"id": "conv_1", "kind": "channel", "members": []})

    client = _make_client(handler)
    await client.create_conversation(
        {
            "tenant_id": "tnt_1",
            "kind": "channel",
            "name": "build",
            "visibility": "tenant",
            "members": [
                {"account_id": "acc_1", "kind": "agent", "endpoint_id": "bep_1"},
            ],
        },
        actor_endpoint_id="bep_1",
        origin_session_id="sess_1",
    )

    assert seen[0].method == "POST"
    assert seen[0].url.path == "/v1/messaging/conversations"
    assert json.loads(seen[0].content) == {
        "tenant_id": "tnt_1",
        "kind": "channel",
        "name": "build",
        "visibility": "tenant",
        "members": [{"account_id": "acc_1", "kind": "agent", "endpoint_id": "bep_1"}],
        "actor_endpoint_id": "bep_1",
        "origin_session_id": "sess_1",
    }


async def test_conversation_send_sends_the_agent_actor_and_never_agent_label():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _ok({"message": {"id": "cmsg_1", "seq": 4}})

    client = _make_client(handler)
    await client.conversation_send(
        "conv_1",
        client_ref="ref-1",
        text="hello",
        tenant_id="tnt_1",
        actor_endpoint_id="bep_1",
        origin_session_id="sess_1",
    )

    assert seen[0].url.path == "/v1/messaging/conversations/conv_1/messages"
    payload = json.loads(seen[0].content)
    assert payload == {
        "client_ref": "ref-1",
        "body": "hello",
        "tenant_id": "tnt_1",
        "actor_endpoint_id": "bep_1",
        "origin_session_id": "sess_1",
    }
    assert "agent_label" not in payload


async def test_the_human_conversation_calls_are_unchanged_without_an_actor():
    """No actor means no actor field — the pre-agent wire shape, exactly."""

    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _ok({"message": {"id": "cmsg_1"}})

    client = _make_client(handler)
    await client.conversation_send("conv_1", client_ref="ref-1", text="hello")
    assert json.loads(seen[0].content) == {"client_ref": "ref-1", "body": "hello"}

    await client.create_conversation({"kind": "dm", "members": []})
    assert json.loads(seen[1].content) == {"kind": "dm", "members": []}


async def test_conversation_notify_posts_the_policy_on_the_read_route():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _ok({"muted": False})

    client = _make_client(handler)
    await client.conversation_notify("conv_1", tenant_id="tnt_1", notify="mention")

    assert seen[0].url.path == "/v1/messaging/conversations/conv_1/read"
    assert json.loads(seen[0].content) == {"notify": "mention", "tenant_id": "tnt_1"}

    with pytest.raises(ValueError):
        await client.conversation_notify("conv_1", notify="loud")  # type: ignore[arg-type]


async def test_conversations_passes_discoverable_rows_through_untouched():
    """Discovery is the server's authorization answer, not the client's to edit."""

    row = {
        "id": "conv_2",
        "kind": "channel",
        "visibility": "tenant",
        "discoverable": True,
        "member": False,
        "member_count": 4,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return _ok({"conversations": [row], "next_cursor": "bcur_2"})

    client = _make_client(handler)
    result = await client.conversations(tenant_id="tnt_1", limit=10)

    assert result["conversations"] == [row]
    assert result["next_cursor"] == "bcur_2"
