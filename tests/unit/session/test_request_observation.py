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


def test_viewer_facade_refuses_owning_methods_clearly(tmp_path) -> None:
    """AttachedSession refuses with the owning-runtime remedy, never an inert
    subscriber or a fake shutdown receipt."""
    from local_operator.session.attached import AttachedSession

    viewer = AttachedSession(
        config_dir=tmp_path / "cfg",
        session_id="viewer-session",
        takeover_factory=lambda: None,
    )
    with pytest.raises(RuntimeError, match="OWNING runtime"):
        viewer.subscribe_requests(lambda _row: None)

    async def drive() -> None:
        with pytest.raises(RuntimeError, match="OWNING runtime"):
            await viewer.shutdown_owned_work("test")

    asyncio.run(drive())


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
