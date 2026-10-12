"""Request-final observation: carrier, builder, fast path, refusals.

These pin the seams whose failure is silent or remote: a subscription that
covers the wrong tree (or a literal None copied at fork), a fast path that
allocates or prices on the ordinary request, usage semantics that price a
missing receipt as zero (or double-count a duplicate snapshot), a handler bug
that masks a provider outcome, and a viewer that installs an inert subscriber
instead of refusing. The REAL-SDK acceptance matrix (real Session, task tool,
forks) lives beside this file in ``test_request_observation_sdk.py``.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from local_operator.harness.types import (
    ChatRequest,
    StreamEndEvent,
    StreamUsageEvent,
    Usage,
)
from local_operator.model import configure as configure_module
from local_operator.model.configure import SessionStreamFn, build_model_spec
from local_operator.session import request_observation as ro


def _spec(provider: str = "test", model: str = "mock") -> Any:
    return build_model_spec(provider, model)


def _request(purpose: str = "turn") -> ChatRequest:
    return ChatRequest(model=_spec(), purpose=purpose)


def _stream(session_id: str = "synthetic-root") -> SessionStreamFn:
    from unittest.mock import MagicMock

    # A MagicMock store, like the sibling configure tests: the wrapper only
    # touches the recorder path when a request runs, and these rigs drive
    # ``_record_stream`` with a stubbed ``_record_usage``.
    return SessionStreamFn(MagicMock(), {}, session_id)


def _wire(usage: Usage | None = None, *, duplicate: bool = False, host: str | None = None):
    async def gen():
        if usage is not None:
            yield StreamUsageEvent(usage=usage)
            if duplicate:
                yield StreamUsageEvent(usage=usage)
        yield StreamEndEvent(stop_reason="stop", usage=usage, served_provider=host)

    return gen()


async def _drain(owner: SessionStreamFn, request: ChatRequest, wire) -> None:
    async for _event in owner._record_stream(request, wire):
        pass


@pytest.mark.asyncio
async def test_none_fast_path_allocates_nothing_and_notifies_nobody(monkeypatch) -> None:
    """Ordinary unset: no holder, no builder call, no allocation on the path."""
    stream = _stream()
    calls: list[Any] = []
    monkeypatch.setattr(
        configure_module,
        "build_request_final_observation",
        lambda **kwargs: calls.append(kwargs),
    )
    assert stream._request_observers is None
    usage = Usage(input_tokens=10, output_tokens=5)
    await _drain(stream, _request(), _wire(usage, duplicate=True, host="h"))
    assert calls == []
    assert stream._request_observers is None


@pytest.mark.asyncio
async def test_subscription_covers_preexisting_forks_and_grandchildren() -> None:
    """The root reference, not a copied holder value: forks made BEFORE the
    install still deliver, and grandchildren inherit the same root."""
    root = _stream()
    child = root.fork("synthetic-child")
    grand = child.fork("synthetic-grandchild")
    child.bind_request_actor(job_id="job-1", actor_role="task")
    seen: list[ro.RequestFinalObservation] = []
    unsubscribe = root.subscribe_requests(seen.append)

    for owner in (root, child, grand):
        await _drain(owner, _request(), _wire(Usage(usd_cost=0.12, provider="p", model_id="m")))
    assert [row.session_id for row in seen] == [
        "synthetic-root",
        "synthetic-child",
        "synthetic-grandchild",
    ]
    assert [row.parent_session_id for row in seen] == [
        None,
        "synthetic-root",
        "synthetic-child",
    ]
    assert all(row.tree_id == "synthetic-root" for row in seen)
    assert [row.job_id for row in seen] == [None, "job-1", None]
    assert len({row.request_id for row in seen}) == 3

    unsubscribe()
    unsubscribe()  # idempotent by contract
    await _drain(root, _request(), _wire(Usage(usd_cost=1.0, provider="p", model_id="m")))
    assert len(seen) == 3


@pytest.mark.asyncio
async def test_request_start_snapshot_governs_mid_flight_unsubscribe() -> None:
    """A request begun while subscribed delivers even if the caller
    unsubscribes before it finishes; the snapshot is taken at request start."""
    stream = _stream()
    seen: list[ro.RequestFinalObservation] = []
    unsubscribe = stream.subscribe_requests(seen.append)

    usage = Usage(input_tokens=1, output_tokens=1, provider="p", model_id="m")

    async def wire():
        yield StreamUsageEvent(usage=usage)
        unsubscribe()  # mid-flight: must not retroactively drop this request
        yield StreamEndEvent(stop_reason="stop", usage=usage)

    await _drain(stream, _request(), wire())
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_duplicate_usage_snapshots_replace_and_never_add() -> None:
    """A provisional usage frame followed by a final one is ONE final record;
    duplicate end+usage snapshots do not double-count the money.

    ``provider``/``model_id`` on the usage are the failover layer's serving
    stamp (``_stamp_serving_spec`` writes them from the on-wire spec after a
    real fallback), so this is the requested-A/served-B shape at the seam the
    stamp exists on: the stamp must survive the wrapper exactly as written.
    """
    stream = _stream()
    seen: list[ro.RequestFinalObservation] = []
    stream.subscribe_requests(seen.append)
    usage = Usage(
        input_tokens=100,
        output_tokens=20,
        cache_read_tokens=80,
        reasoning_tokens=5,
        usd_cost=0.12,
        provider="served-b",
        model_id="model-b",
    )
    await _drain(stream, _request(), _wire(usage, duplicate=True, host="upstream-x"))
    assert len(seen) == 1
    row = seen[0]
    assert row.usage is not None
    assert (row.usage.input_tokens, row.usage.cache_read_tokens, row.usage.reasoning_tokens) == (
        100,
        80,
        5,
    )
    assert row.usage.usd_cost == 0.12
    assert row.upstream_host == "upstream-x"
    assert row.served_identity_source == "usage_stamp"
    assert row.served_provider == "served-b" and row.served_model_id == "model-b"
    assert row.usage_coverage == "reported_final"


@pytest.mark.asyncio
async def test_missing_usage_is_unknown_and_recorded_zero_is_known() -> None:
    """``usage=None`` is unknown (never ``Usage()`` priced as zero); a reported
    ``usd_cost=0.0`` is a KNOWN zero and stays one."""
    stream = _stream()
    seen: list[ro.RequestFinalObservation] = []
    stream.subscribe_requests(seen.append)

    async def failing():
        raise RuntimeError("synthetic failure before usage")
        yield

    try:
        await _drain(stream, _request(), failing())
    except RuntimeError:
        pass
    assert len(seen) == 1
    row = seen[0]
    assert row.usage is None and row.usage_reported is False
    assert row.usage_coverage == "missing"
    assert row.served_provider is None and row.served_model_id is None
    assert row.ok is False

    known_zero = Usage(usd_cost=0.0, provider="p", model_id="m")
    await _drain(stream, _request(), _wire(known_zero, host="h"))
    zero = seen[1]
    assert zero.usage is not None and zero.usage.usd_cost == 0.0
    assert zero.usage_reported is True and zero.usage_coverage == "reported_final"


def test_builder_served_identity_three_cases() -> None:
    """usage_stamp / request_assumed / no-usage: exactly the three documented
    cases, with no phantom served identity on a missing receipt."""
    stamped = Usage(provider="served-b", model_id="model-b")
    row = ro.build_request_final_observation(
        request_id="r1",
        tree_id="t",
        session_id="s",
        parent_session_id=None,
        job_id=None,
        actor_role=None,
        request=_request(),
        usage=stamped,
        upstream_host=None,
        ok=True,
        outcome="stop",
        duration_ms=1.0,
    )
    assert row.served_identity_source == "usage_stamp"
    assert (row.served_provider, row.served_model_id) == ("served-b", "model-b")

    unstamped = Usage(input_tokens=1)
    row = ro.build_request_final_observation(
        request_id="r2",
        tree_id="t",
        session_id="s",
        parent_session_id=None,
        job_id=None,
        actor_role=None,
        request=_request(),
        usage=unstamped,
        upstream_host=None,
        ok=True,
        outcome="stop",
        duration_ms=1.0,
    )
    assert row.served_identity_source == "request_assumed"
    assert (row.served_provider, row.served_model_id) == ("test", "mock")

    row = ro.build_request_final_observation(
        request_id="r3",
        tree_id="t",
        session_id="s",
        parent_session_id=None,
        job_id=None,
        actor_role=None,
        request=_request(),
        usage=None,
        upstream_host=None,
        ok=False,
        outcome="RuntimeError",
        duration_ms=1.0,
    )
    assert row.served_identity_source == "request_assumed"
    assert row.served_provider is None and row.served_model_id is None


@pytest.mark.asyncio
async def test_usage_is_detached_and_handlers_cannot_cross_mutate() -> None:
    """The observation owns a detached copy; with two handlers each gets its
    own, so a mutating consumer cannot rewrite a sibling's view."""
    stream = _stream()
    usage = Usage(input_tokens=1, output_tokens=1, provider="p", model_id="m")
    first: list[ro.RequestFinalObservation] = []
    second: list[ro.RequestFinalObservation] = []
    stream.subscribe_requests(first.append)
    stream.subscribe_requests(second.append)

    await _drain(stream, _request(), _wire(usage))
    usage.input_tokens = 999  # the LIVE object mutating later must not reach rows
    assert first[0].usage is not None and first[0].usage.input_tokens == 1
    assert second[0].usage is not None
    first[0].usage.input_tokens = 777  # handler mutation must not cross
    assert second[0].usage.input_tokens == 1


@pytest.mark.asyncio
async def test_handler_failure_is_counted_surfaced_and_never_raised(caplog) -> None:
    """A raising handler is logged + counted on the carrier (surfaced later on
    the shutdown receipt); the healthy handler still receives; the request's
    outcome is untouched."""
    stream = _stream()
    good: list[ro.RequestFinalObservation] = []

    def bad(_row: ro.RequestFinalObservation) -> None:
        raise ValueError("observer bug")

    stream.subscribe_requests(bad)
    stream.subscribe_requests(good.append)

    await _drain(stream, _request(), _wire(Usage(usd_cost=0.1, provider="p", model_id="m")))
    assert len(good) == 1
    holder = stream._request_observers
    assert holder is not None and holder.callback_errors == 1


def test_protocol_split_keeps_viewers_non_owning() -> None:
    """The owning surface lives on its own protocol extension (review round 1,
    R1): the shared ``SessionProtocol`` and every structural conformer built
    against it stay unchanged, a real ``Session`` carries the methods, and a
    viewer facade simply is NOT an owner -- no stub is grown anywhere."""
    from typing import _get_protocol_attrs  # type: ignore[attr-defined]

    from local_operator.session.attached import AttachedSession
    from local_operator.session.protocol import (
        OwningSessionProtocol,
        SessionProtocol,
        ViewerSessionProtocol,
    )
    from local_operator.session.session import Session

    own_attrs = set(_get_protocol_attrs(OwningSessionProtocol))
    assert {"subscribe_requests", "shutdown_owned_work"} <= own_attrs
    shared_attrs = set(_get_protocol_attrs(SessionProtocol))
    assert not (
        {"subscribe_requests", "shutdown_owned_work"} & shared_attrs
    ), "the shared protocol must not grow owning members: every FakeSession would need stubs"
    viewer_attrs = set(_get_protocol_attrs(ViewerSessionProtocol))
    assert not ({"subscribe_requests", "shutdown_owned_work"} & viewer_attrs)
    # Real Session carries them; the viewer never pretends to.
    assert callable(Session.subscribe_requests)
    assert callable(Session.shutdown_owned_work)
    assert not hasattr(AttachedSession, "subscribe_requests")
    assert not hasattr(AttachedSession, "shutdown_owned_work")


@pytest.mark.asyncio
async def test_unsubscribe_is_registration_exact_and_idempotent() -> None:
    """A stale unsubscribe cannot remove a LATER registration of the same
    callable, and a repeated call removes nothing: the handle matches its own
    registration identity once (review round 1, R4)."""
    stream = _stream()
    seen: list[ro.RequestFinalObservation] = []
    old = stream.subscribe_requests(seen.append)
    old()  # remove the first registration
    new = stream.subscribe_requests(seen.append)
    old()  # MUST NOT remove the new registration
    holder = stream._request_observers
    assert holder is not None and holder.subscriber_count == 1
    await _drain(stream, _request(), _wire(Usage(usd_cost=0.1, provider="p", model_id="m")))
    assert len(seen) == 1, "the surviving registration must still be delivered to"
    new()  # now remove it for real
    assert holder.subscriber_count == 0


@pytest.mark.asyncio
async def test_cancelled_error_from_handler_is_isolated_and_counted() -> None:
    """An observer-raised CancelledError is a callback failure: counted, the
    remaining start-time subscriber still receives, and the request path
    completes normally -- a handler must never cancel a COMPLETED request
    (review round 1, R5)."""
    stream = _stream()
    good: list[ro.RequestFinalObservation] = []

    def cancelling(_row: ro.RequestFinalObservation) -> None:
        raise asyncio.CancelledError("observer tried to cancel a finished request")

    stream.subscribe_requests(cancelling)
    stream.subscribe_requests(good.append)

    await _drain(stream, _request(), _wire(Usage(usd_cost=0.1, provider="p", model_id="m")))
    assert len(good) == 1, "the healthy subscriber must still be delivered to"
    holder = stream._request_observers
    assert holder is not None and holder.callback_errors == 1


@pytest.mark.asyncio
async def test_session_refuses_streams_without_observation_support() -> None:
    """A session whose stream cannot observe refuses clearly instead of
    silently installing nothing."""
    from local_operator.session.session import Session

    class _BareStream:
        async def __call__(self, request):  # pragma: no cover - never invoked
            raise AssertionError

    class _BareSession:
        _stream_fn = _BareStream()

    delegate = Session.subscribe_requests.__get__(_BareSession(), _BareSession)
    with pytest.raises(RuntimeError, match="does not support request observation"):
        delegate(lambda _row: None)


def test_result_types_are_frozen_values() -> None:
    """Both public records are frozen dataclasses -- mutation raises, so a
    consumer cannot rewrite what a sibling handler already saw."""
    row = ro.build_request_final_observation(
        request_id="r",
        tree_id=None,
        session_id=None,
        parent_session_id=None,
        job_id=None,
        actor_role=None,
        request=_request(),
        usage=None,
        upstream_host=None,
        ok=True,
        outcome="stop",
        duration_ms=0.0,
    )
    with pytest.raises(Exception):
        row.request_id = "other"  # type: ignore[misc]
    receipt = ro.OwnedShutdownResult(complete=True, reason="done", elapsed_ms=1.0)
    with pytest.raises(Exception):
        receipt.complete = False  # type: ignore[misc]
    assert receipt.reason == "done"
