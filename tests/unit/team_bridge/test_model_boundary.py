"""H3 model boundary: runner isolation, wire guard charging + pre-log sanitization."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from local_operator.harness.types import (
    StreamEndEvent,
    StreamModelEvent,
    StreamTextDelta,
    Usage,
)
from local_operator.model.configure import build_model_spec
from local_operator.providers.failover import ProviderError
from local_operator.team_bridge.model_boundary import (
    DefaultSelection,
    ScreeningWireGuard,
    ScreenModelRunner,
    _ScreeningStreamRefused,
)
from local_operator.team_bridge.store import OwnerReceipt, Store
from local_operator.team_bridge.types import OwnerSnapshot

CANARY_PROSE = "VENDOR-PROSE-CANARY: your request echoed BODY"


def _owner() -> OwnerSnapshot:
    return OwnerSnapshot(
        credential_id=7,
        account_id="acc_a",
        tenant_id="tnt_1",
        endpoint_id="bep_1",
        canonical_base_url="https://qa.invalid/v1",
        enrollment_generation=1,
    ).validated()


def _setup(tmp_path: Path) -> tuple[Store, OwnerSnapshot]:
    store = Store(tmp_path)
    owner = _owner()
    store.commit_enrollment(owner, OwnerReceipt("bridge_consent_v1", "r"))
    return store, owner


class FakeStreamFn:
    def __init__(self, events):
        self.events = events
        self.requests = []

    def __call__(self, request, signal):
        self.requests.append(request)
        events = self.events

        async def gen():
            for event in events:
                yield event

        return gen()


def test_runner_builds_one_fresh_isolated_user_request(tmp_path) -> None:
    stamped = Usage(input_tokens=1, output_tokens=1).model_copy(
        update={"provider": "openai", "model_id": "qa-default"}
    )
    events = [
        StreamModelEvent(model=build_model_spec("openai", "qa-default")),
        SimpleNamespace(type="reasoning_delta", text="thinking"),
        StreamTextDelta(delta="VERDICT: deliver\nREASON: ok."),
        StreamEndEvent(stop_reason="stop", usage=stamped),
    ]
    fn = FakeStreamFn(events)
    selection = DefaultSelection(
        provider="qa",
        model_id="qa/default",
        spec=build_model_spec("openai", "qa-default"),
        approved_identities=frozenset({"qa-default"}),
    )
    runner = ScreenModelRunner(fn, selection)
    reply, actual = runner("PROMPT", default_model="qa/default")
    assert reply == "VERDICT: deliver\nREASON: ok."
    assert actual == "qa-default"  # the spec's own model id is the serving evidence
    request = fn.requests[0]
    assert request.isolated is True
    assert request.tools == [] and request.tool_choice == "none"
    assert len(request.messages) == 1 and request.messages[0].role == "user"
    assert request.system_blocks == []
    assert request.prompt_cache_key is None


def test_runner_refuses_tools_and_unevidenced_identity(tmp_path) -> None:
    tool_events = [SimpleNamespace(type="tool_call_delta", tool_call=None)]
    end_only = [StreamEndEvent(stop_reason="stop")]
    selection = DefaultSelection(
        provider="qa",
        model_id="qa/default",
        spec=build_model_spec("openai", "qa-default"),
        approved_identities=frozenset({"qa/default"}),
    )
    with pytest.raises(_ScreeningStreamRefused):
        ScreenModelRunner(FakeStreamFn(tool_events), selection)("p", default_model="qa/default")
    # A default the runner was not built for is unevidenced -> hold signal.
    assert ScreenModelRunner(FakeStreamFn(end_only), selection)("p", default_model="qa/other") == (
        "",
        None,
    )
    # Missing end (exhaustion is not completion) refuses even with clean text.
    text_only = [StreamTextDelta(delta="VERDICT: deliver\nREASON: ok.")]
    with pytest.raises(_ScreeningStreamRefused):
        ScreenModelRunner(FakeStreamFn(text_only), selection)("p", default_model="qa/default")


def test_wire_guard_charges_each_attempt_and_sanitizes(tmp_path) -> None:
    store, owner = _setup(tmp_path)
    calls: list[int] = []

    class Inner:
        def __init__(self) -> None:
            import httpx as _httpx

            self._http = _httpx.AsyncClient(transport=_httpx.MockTransport(self._send))

        def _send(self, request):
            import httpx as _httpx

            calls.append(1)
            if len(calls) == 2:
                from local_operator.providers.failover import ProviderError

                raise ProviderError(
                    500, CANARY_PROSE, retryable=True, kind="transient", request_sent=True
                )
            return _httpx.Response(200, content=b"x")

        async def _drive(self):
            response = await self._http.get("https://api.openai.com/v1/responses")
            return response

        def stream(self, request, api_key, oauth_access=None):
            async def ok():
                await self._drive()
                yield "event"

            return ok()

    guard = ScreeningWireGuard(Inner(), store=store, owner=owner, message_id="bmsg_1")

    async def drive_once():
        got = []
        async for event in guard.stream(None, "k"):
            got.append(event)
        return got

    assert asyncio.new_event_loop().run_until_complete(drive_once()) == ["event"]
    # Second attempt (a resend) is charged separately and its error is
    # scrubbed at the boundary: no prose, no cause, non-retryable typed.
    with pytest.raises(Exception) as refusal:
        asyncio.new_event_loop().run_until_complete(drive_once())
    err = refusal.value
    assert isinstance(err, ProviderError)
    assert CANARY_PROSE not in str(err)
    assert err.__cause__ is None
    assert err.retryable is False and err.request_sent is True
    conn_attempts = [
        r[0]
        for r in __import__("sqlite3")
        .connect(
            __import__(
                "local_operator.team_bridge.store", fromlist=["state_db_path"]
            ).state_db_path(tmp_path)
        )
        .execute("SELECT kind FROM screening_attempts")
    ]
    assert conn_attempts == ["http", "http"]


def test_wire_guard_budget_refusal_is_non_retryable(tmp_path) -> None:
    store, owner = _setup(tmp_path)
    for _ in range(40):
        store.reserve_http_attempt(owner, message_id="bmsg_1")

    class Inner:
        def stream(self, request, api_key, oauth_access=None):
            raise AssertionError("no request may leave after budget exhaustion")

    guard = ScreeningWireGuard(Inner(), store=store, owner=owner, message_id="bmsg_1")

    async def drive():
        async for _ in guard.stream(None, "k"):
            pass

    with pytest.raises(Exception) as refusal:
        asyncio.new_event_loop().run_until_complete(drive())
    budget_err = refusal.value
    assert isinstance(budget_err, ProviderError)
    assert budget_err.retryable is False and budget_err.request_sent is False


def test_opaque_route_refuses_before_io(tmp_path) -> None:
    store, owner = _setup(tmp_path)

    class Bare:
        def stream(self, request, api_key, oauth_access=None):
            raise AssertionError("must not be called")

    guard = ScreeningWireGuard(Bare(), store=store, owner=owner, message_id="bmsg_op")
    from local_operator.providers.failover import ProviderError

    async def drive():
        async for _ in guard.stream(None, "k"):
            pass

    with pytest.raises(ProviderError):
        import asyncio as _asyncio

        _asyncio.run(drive())
