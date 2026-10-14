"""Pure negative controls for shutdown accounting: no SDK session or native child."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest

from local_operator.harness.types import AbortSignal, StreamEndEvent
from local_operator.model.configure import SessionStreamFn
from local_operator.providers import failover
from local_operator.providers.failover import ProviderError
from local_operator.session.request_observation import RequestSubscriberHolder
from tests.unit.providers.test_failover import FakeAuth, _request
from tests.unit.session.shutdown_accounting import ShutdownAccounting
from tests.unit.session.test_request_observation_sdk import Wire
from tests.unit.session.test_request_observation_sdk import (
    shutdown_accounting as _accounting_fixture,
)
from tests.unit.session.test_request_observation_sdk import wire as _wire_fixture

# Reuse the exact opt-in fixture without shadowing an unused imported name in
# every control's parameter list (the suite treats that as flake8 F811).
wire = _wire_fixture
shutdown_accounting = _accounting_fixture


def _recorder(rows: list[Any]) -> SessionStreamFn:
    # The canonical stream/final builder needs only recorder state. Building a
    # real SDK session would add native/lifecycle surfaces to these controls.
    record = object.__new__(SessionStreamFn)
    record._session_id = "synthetic-accounting"
    record._parent_session_id = None
    record._request_job_id = None
    record._request_actor_role = None
    record._observation_root = None
    record._request_observers = RequestSubscriberHolder()
    record._request_observers.subscribe(rows.append)
    record._counts_as_child_request = False
    record._affinity_enabled = False
    record._record_usage = lambda *args, **kwargs: None
    return record


async def _exercise(case: str, wire: Wire, trace: ShutdownAccounting) -> list[Any]:
    rows: list[Any] = []
    record = _recorder(rows)
    request = _request()
    signal = AbortSignal()
    trace.begin_shutdown()
    if case == "aborted":
        signal.abort("synthetic-shutdown")
    wire.script("helper", [StreamEndEvent(stop_reason="stop")])

    async def client_for(_spec: Any) -> Any:
        if case == "unknown":
            raise ProviderError(None, "synthetic refusal", kind="auth", retryable=False)
        return wire

    async def cancelled() -> Any:
        raise asyncio.CancelledError()
        yield  # async-generator shape; no transport is reached

    upstream = (
        cancelled()
        if case == "cancelled"
        else failover.stream_with_failover(
            request,
            FakeAuth({"openai": ["synthetic-not-a-credential"]}),
            {"retry": {"enabled": False}},
            client_for,
            signal=signal,
        )
    )
    try:
        async for _event in record._record_stream(request, upstream):
            pass
    except ProviderError as exc:
        assert case in {"aborted", "unknown"}
        assert exc.kind == ("aborted" if case == "aborted" else "auth")
    except asyncio.CancelledError:
        assert case == "cancelled"
    assert len(rows) == 1
    assert trace.active.get() is None, "request context leaked across consumer yield"
    return rows


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["success", "aborted", "cancelled"])
async def test_shutdown_finals_match_witnessed_transport_or_pre_wire_exit(
    case: str, wire: Wire, shutdown_accounting: ShutdownAccounting
) -> None:
    rows = await _exercise(case, wire, shutdown_accounting)
    assert len(wire.requests) == (1 if case == "success" else 0)
    if case != "success":
        # The OLD scalar-count assertion is red on legitimate pre-wire exits.
        # This is not a native shutdown reproduction or a guessed historical kind.
        with pytest.raises(AssertionError):
            assert len(wire.requests) == len(rows)
    shutdown_accounting.validate(rows, len(wire.requests))


@pytest.mark.asyncio
async def test_unknown_provider_error_is_not_a_shutdown_abort_allowance(
    wire: Wire, shutdown_accounting: ShutdownAccounting
) -> None:
    rows = await _exercise("unknown", wire, shutdown_accounting)
    assert len(wire.requests) == 0
    with pytest.raises(AssertionError, match="unwitnessed shutdown failure"):
        shutdown_accounting.validate(rows, 0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation", ["success_without_wire", "duplicate", "wrong_actor", "outside_shutdown"]
)
async def test_unmatched_or_misattributed_rows_are_still_rejected(
    mutation: str, wire: Wire, shutdown_accounting: ShutdownAccounting
) -> None:
    rows = await _exercise("aborted", wire, shutdown_accounting)
    if mutation == "success_without_wire":
        rows[0] = replace(rows[0], ok=True, outcome="stop")
    elif mutation == "duplicate":
        rows.append(rows[0])
    elif mutation == "wrong_actor":
        rows[0] = replace(rows[0], session_id="wrong-synthetic-actor")
    else:
        shutdown_accounting.logical[0].during_shutdown = False
    with pytest.raises(AssertionError):
        shutdown_accounting.validate(rows, 0)
