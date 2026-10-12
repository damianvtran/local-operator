"""H3 model boundary: runner isolation, wire guard charging + pre-log sanitization."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from local_operator.harness.types import (
    StreamEndEvent,
    StreamModelEvent,
    StreamTextDelta,
)
from local_operator.model.configure import build_model_spec
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
    events = [
        StreamModelEvent(model=build_model_spec("openai", "qa-default")),
        SimpleNamespace(type="reasoning_delta", text="thinking"),
        StreamTextDelta(delta="VERDICT: deliver\nREASON: ok."),
        StreamEndEvent(stop_reason="stop"),
    ]
    fn = FakeStreamFn(events)
    selection = DefaultSelection(
        provider="openai",
        model_id="qa-default",
        spec=build_model_spec("openai", "qa-default"),
        approved_identities=frozenset({"qa-default"}),
    )
    runner = ScreenModelRunner(fn, selection)
    reply, actual = runner("PROMPT", default_model="qa-default")
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


def test_wire_guard_requires_job_and_charges_bound_dispatch(tmp_path) -> None:
    store, owner = _setup(tmp_path)

    class Inner:
        def __init__(self) -> None:
            import httpx as _httpx

            self._http = _httpx.AsyncClient(transport=_httpx.MockTransport(self._send))

        def _send(self, request):
            import httpx as _httpx

            return _httpx.Response(200, content=b"x")

        def stream(self, request, api_key, oauth_access=None):
            async def ok():
                await self._http.get("https://api.openai.com/v1/responses")
                yield "event"

            return ok()

    # No bound job: the guard refuses BEFORE any I/O (R5 job-required law).
    bare = ScreeningWireGuard(Inner(), store=store, owner=owner, message_id="bmsg_1")
    from local_operator.providers.failover import ProviderError

    async def drive_bare():
        async for _ in bare.stream(None, "k"):
            pass

    import asyncio as _asyncio

    with pytest.raises(ProviderError):
        _asyncio.run(drive_bare())

    # A fully bound job: published policy + logical attempt + held turn are
    # validated inside the native reservation before each send.
    store.publish_screening_policy(
        owner,
        version="bridge-screen-v1",
        template_sha256="t" * 64,
        default_model="qa-default",
        revision="rev-1",
    )
    # R9/Q3-5: a reservation carries the complete registered input; the bind
    # must compare a REAL job against it (no legacy no-proof bypass).
    proof = {
        "owner": (
            f"{owner.credential_id}|{owner.account_id}|{owner.tenant_id}"
            f"|{owner.endpoint_id}|{owner.canonical_base_url}|{owner.enrollment_generation}"
        ),
        "body": "b" * 64,
        "template": "t" * 64,
        "version": "bridge-screen-v1",
        "provider": "openai",
        "model": "qa-default",
        "approved": ["qa-default"],
        "envelope": "e" * 64,
        "request": "r" * 64,
        "display": "d" * 64,
        "revision": "rev-1",
        "epoch": 1,
        "message": "bmsg_1",
        "tenant": owner.tenant_id,
        "logical": "rev-1|b|e",
    }
    attempt = store.reserve_screening_attempt(
        owner, message_id="bmsg_1", fingerprint="rev-1|b|e", input_proof=proof
    )
    turn = store.claim_screen_turn(owner, holder_prefix="qa", now=__import__("time").time(), ttl=90)
    assert turn is not None
    from local_operator.team_bridge.model_boundary import AdmissionJob

    job = AdmissionJob(
        owner_generation=owner.enrollment_generation,
        credential_id=owner.credential_id,
        account_id=owner.account_id,
        tenant_id=owner.tenant_id,
        endpoint_id=owner.endpoint_id,
        hub_url=owner.canonical_base_url,
        message_id="bmsg_1",
        attempt_id=attempt,
        body_sha256="b" * 64,
        template_sha256="t" * 64,
        template_version="bridge-screen-v1",
        provider="openai",
        model="qa-default",
        approved=("qa-default",),
        envelope_digest="e" * 64,
        request_digest="r" * 64,
        display_digest="d" * 64,
        revision="rev-1",
        epoch=int((store.current_screening_policy(owner) or {"epoch": 0})["epoch"]),
        turn=turn,
    )
    store.bind_attempt_fingerprint(owner, attempt_id=attempt, fingerprint=job.digest, job=job)
    inner = Inner()
    guard = ScreeningWireGuard(
        inner, store=store, owner=owner, message_id="bmsg_1", job_ref={"job": job}
    )

    async def drive_bound():
        got = []
        async for event in guard.stream(None, "k"):
            got.append(event)
        return got

    assert _asyncio.run(drive_bound()) == ["event"]
    import sqlite3

    from local_operator.team_bridge.store import state_db_path

    with sqlite3.connect(state_db_path(tmp_path)) as db:
        rows = db.execute(
            "SELECT kind, fingerprint FROM screening_attempts WHERE kind='http'"
        ).fetchall()
    assert rows == [("http", job.digest)]


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


def test_route_replacement_after_install_refuses_before_io(tmp_path) -> None:
    """R6/Q14: a primary transport REPLACED after installation must refuse
    BEFORE any I/O — the historical install flag is not evidence."""

    store, owner = _setup(tmp_path)
    hits: list[str] = []

    class Inner:
        def __init__(self) -> None:
            import httpx as _httpx

            self._http = _httpx.AsyncClient(transport=_httpx.MockTransport(self._send))

        def _send(self, request):
            import httpx as _httpx

            hits.append("hit")
            return _httpx.Response(200, content=b"x")

        def stream(self, request, api_key, oauth_access=None):
            async def ok():
                await self._http.get("https://api.openai.com/v1/responses")
                yield "event"

            return ok()

    inner = Inner()
    guard = ScreeningWireGuard(inner, store=store, owner=owner, message_id="bmsg_rep")
    # Replace the SELECTED route AFTER the guard installed its proxy.
    import httpx as _httpx

    from local_operator.providers.failover import ProviderError

    inner._http._transport = _httpx.MockTransport(inner._send)

    async def drive():
        async for _ in guard.stream(None, "k"):
            pass

    import asyncio as _asyncio

    with pytest.raises(ProviderError):
        _asyncio.run(drive())
    assert hits == []  # ZERO handler calls after replacement


def test_successor_owner_snapshot_cannot_release_new_turn(tmp_path) -> None:
    """R6/Q15: a stale owner SNAPSHOT presenting a successor's capability
    token must not release/renew/match it — the fence is the live
    generation, not the hub/account key alone."""

    import time as _time

    store, owner = _setup(tmp_path)
    turn = store.claim_screen_turn(owner, holder_prefix="qa", now=_time.time(), ttl=90)
    assert turn is not None
    # The store moves to the successor generation: the OLD snapshot is stale.
    Store(tmp_path).invalidate(owner)
    assert store.release_screen_turn(owner, turn=turn) is False
    assert store.renew_screen_turn(owner, turn=turn, now=_time.time(), ttl=90) is False
    assert store.screen_turn_matches(owner, turn=turn, now=_time.time()) is False


def test_selected_send_boundary_refuses_foreign_replacement(tmp_path) -> None:
    """R7/Q14: the DEDICATED client's transport SELECTION is validated — a
    primary replaced after installation is refused at the selection point
    itself (before any I/O), even though the outer stream check already ran.
    """

    store, owner = _setup(tmp_path)
    from local_operator.team_bridge.model_boundary import (
        _ChargingAsyncTransport,
        _guarded_screening_client,
    )

    hits: list[str] = []

    def _send(request):
        import httpx as _httpx

        hits.append("send")
        return _httpx.Response(200, content=b"x")

    class _StubGuard:
        def __init__(self) -> None:
            self.proxies: list[object] = []

        def route_ok(self, route) -> bool:
            return any(route is proxy for proxy in self.proxies)

        def route_error(self):
            from local_operator.providers.failover import ProviderError

            return ProviderError(
                409,
                "screening selected route is not this job's guard",
                retryable=False,
                kind="request",
                request_sent=False,
                fail_closed=True,
            )

    import httpx as _httpx

    client = _guarded_screening_client(transport=_httpx.MockTransport(_send))
    guard = _StubGuard()
    client._tb_guard = guard

    # The foreign transport is not OUR route: refused AT SELECTION, no send.
    with pytest.raises(Exception):
        client._transport_for_url(_httpx.URL("https://api.openai.com/v1/responses"))
    assert hits == []

    # An owned proxy routes normally.
    proxy = _ChargingAsyncTransport(_httpx.MockTransport(_send), lambda: None, owner_token=object())
    guard.proxies = [proxy]
    client._transport = proxy
    assert client._transport_for_url(_httpx.URL("https://api.openai.com/v1/responses")) is proxy


def test_one_bind_replays_exactly_and_refuses_a_change(tmp_path) -> None:
    """R7/Q3: the packed logical reservation binds to the job digest ONCE;
    an identical replay is a no-op and a CHANGED pending bind refuses while
    the original binding stays on record."""

    import types as _types

    store, owner = _setup(tmp_path)
    proof = {
        "owner": (
            f"{owner.credential_id}|{owner.account_id}|{owner.tenant_id}"
            f"|{owner.endpoint_id}|{owner.canonical_base_url}|{owner.enrollment_generation}"
        ),
        "body": "b" * 64,
        "template": "t" * 64,
        "version": "bridge-screen-v1",
        "provider": "openai",
        "model": "qa-default",
        "approved": ["qa-default"],
        "envelope": "e" * 64,
        "request": "r" * 64,
        "display": "d" * 64,
        "revision": "rev-1",
        "epoch": 0,
        "message": "bmsg_1",
        "tenant": owner.tenant_id,
        "logical": "rev-1|b|e",
    }
    attempt = store.reserve_screening_attempt(
        owner, message_id="bmsg_1", fingerprint="rev-1|b|e", input_proof=proof
    )
    stub = _types.SimpleNamespace(
        credential_id=owner.credential_id,
        account_id=owner.account_id,
        tenant_id=owner.tenant_id,
        endpoint_id=owner.endpoint_id,
        hub_url=owner.canonical_base_url,
        owner_generation=owner.enrollment_generation,
        body_sha256="b" * 64,
        envelope_digest="e" * 64,
        request_digest="r" * 64,
        display_digest="d" * 64,
        provider="openai",
        model="qa-default",
        template_sha256="t" * 64,
        template_version="bridge-screen-v1",
        revision="rev-1",
        message_id="bmsg_1",
        approved=("qa-default",),
        epoch=0,
    )
    from local_operator.team_bridge.store import StoreRefused, _canonical_job_digest

    # R10/B: the bind digest is the CANONICAL derivation from the job fields
    # (a literal string or forged attribute is refused), so the test derives
    # both arms the same way the production binder now verifies them.
    stub.digest = _canonical_job_digest(stub)
    job_a = stub.digest
    store.bind_attempt_fingerprint(owner, attempt_id=attempt, fingerprint=job_a, job=stub)
    store.bind_attempt_fingerprint(owner, attempt_id=attempt, fingerprint=job_a, job=stub)
    changed = _types.SimpleNamespace(**{**vars(stub), "model": "qa-other"})
    changed.digest = _canonical_job_digest(changed)
    job_b = changed.digest

    with pytest.raises(StoreRefused) as err:
        store.bind_attempt_fingerprint(owner, attempt_id=attempt, fingerprint=job_b, job=changed)
    assert err.value.code == "write_once_conflict"


def test_live_owner_with_token_from_another_frame_cannot_release(tmp_path) -> None:
    """R7/Q15-6: a DIFFERENT but LIVE owner snapshot on the same account,
    handed the CURRENT token string, cannot release, renew or match it —
    the token carries the initiating frame digest."""

    import dataclasses as _dataclasses
    import time as _time

    store, owner = _setup(tmp_path)
    turn = store.claim_screen_turn(owner, holder_prefix="qa", now=_time.time(), ttl=90)
    assert turn is not None
    other = _dataclasses.replace(owner, tenant_id="tnt_other")
    assert store.release_screen_turn(other, turn=turn) is False
    assert store.renew_screen_turn(other, turn=turn, now=_time.time(), ttl=90) is False
    assert store.screen_turn_matches(other, turn=turn, now=_time.time()) is False
    assert store.renew_screen_turn(owner, turn=turn, now=_time.time(), ttl=90) is True
