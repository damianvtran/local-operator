"""Test-only accounting of logical requests across shutdown, including pre-wire aborts.

The canonical recorder emits a final on every exit, while failover can refuse an
already-aborted request before calling the transport. Count equality loses that
distinction. Trace the actual boundary instead: a per-advance ContextVar binds
Wire entries and final UUIDs to one logical request, without comparing prompts or
assuming copied ChatRequests retain object identity. No exception is suppressed.

Scope is reset BEFORE yielding to the consumer: successive async-generator pulls
may run in different tasks, where resetting a token held across a yield is unsafe.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from local_operator.providers.failover import ProviderError
from local_operator.session.request_observation import RequestFinalObservation


@dataclass
class _Logical:
    purpose: str
    actor: tuple[Any, ...]
    scope: str
    transports: list[str] = field(default_factory=list)
    final_ids: list[str] = field(default_factory=list)
    witness: str | None = None
    aborted: bool = False
    during_shutdown: bool = False
    advancing_task: Any = None


class ShutdownAccounting:
    """Observe real boundaries, then reject anything not explained by their metadata."""

    def __init__(self, scope_of: Callable[[Any], str]) -> None:
        self.scope_of = scope_of
        self.active: ContextVar[_Logical | None] = ContextVar("sdk-shutdown-request", default=None)
        self.logical: list[_Logical] = []
        self.finals: dict[str, _Logical] = {}
        self.errors: list[str] = []
        self.shutting_down = False

    def begin_shutdown(self) -> None:
        self.shutting_down = True

    def transport_enter(self, request: Any) -> None:
        slot = self.active.get()
        if slot is None or slot.advancing_task is not asyncio.current_task():
            self.errors.append("transport outside the advancing logical request task")
            return
        slot.transports.append(self.scope_of(request))

    def wrap_failover(self, original: Any) -> Any:
        async def traced(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
            try:
                async for event in original(*args, **kwargs):
                    yield event
            except ProviderError as exc:
                slot = self.active.get()
                signal = kwargs.get("signal")
                if slot is not None:
                    # Only the structured kind and signal state: exception messages
                    # and abort reasons can contain prompt/credential diagnostics.
                    slot.witness = exc.kind
                    slot.aborted = signal is not None and signal.aborted
                    slot.during_shutdown = self.shutting_down
                raise

        return traced

    def wrap_record(self, original: Any) -> Any:
        async def traced(owner: Any, request: Any, upstream: Any) -> AsyncIterator[Any]:
            slot = _Logical(
                purpose=request.purpose,
                actor=(owner._session_id, owner._request_job_id, owner._request_actor_role),
                scope=self.scope_of(request),
            )
            self.logical.append(slot)
            iterator = original(owner, request, upstream)
            try:
                while True:
                    reset = self.active.set(slot)
                    slot.advancing_task = asyncio.current_task()
                    try:
                        event = await anext(iterator)
                    except StopAsyncIteration:
                        return
                    except asyncio.CancelledError:
                        slot.witness = "CancelledError"
                        slot.during_shutdown = self.shutting_down
                        raise
                    finally:
                        self.active.reset(reset)
                    yield event
            finally:
                reset = self.active.set(slot)
                slot.advancing_task = asyncio.current_task()
                try:
                    await iterator.aclose()
                finally:
                    self.active.reset(reset)

        return traced

    def wrap_final(self, original: Any) -> Any:
        def traced(owner: Any, target: Any, **kwargs: Any) -> None:
            slot = self.active.get()
            request_id = kwargs["request_id"]
            if slot is None:
                self.errors.append("final notification outside its logical request")
            elif request_id in self.finals:
                self.errors.append("duplicate final notification UUID")
            else:
                slot.final_ids.append(request_id)
                self.finals[request_id] = slot
            original(owner, target, **kwargs)

        return traced

    def validate(self, rows: list[RequestFinalObservation], transport_count: int) -> None:
        """No unmatched success, unknown failure, lost transport or invented final."""
        assert not self.errors, self.errors
        assert sum(len(slot.transports) for slot in self.logical) == transport_count
        assert len(rows) == len(self.logical) == len(self.finals), "logical final lost/duplicated"
        seen: set[str] = set()
        for row in rows:
            assert row.request_id not in seen, "duplicate observed final"
            seen.add(row.request_id)
            slot = self.finals[row.request_id]
            assert slot.final_ids == [row.request_id], "multiple finals for one logical request"
            assert row.purpose == slot.purpose
            assert (row.session_id, row.job_id, row.actor_role) == slot.actor
            assert all(scope == slot.scope for scope in slot.transports), "wire scope misrouted"
            assert len(slot.transports) <= 1, "unexpected repeated transport in scripted SDK cell"
            if row.ok:
                assert len(slot.transports) == 1, "success without transport"
                continue
            cancelled = slot.witness == "CancelledError" and row.outcome == "CancelledError"
            aborted = slot.witness == "aborted" and slot.aborted and row.outcome == "ProviderError"
            assert slot.during_shutdown and (cancelled or aborted), "unwitnessed shutdown failure"
            assert row.purpose == "turn", "unexpected failed helper request"
            if not slot.transports:
                assert (
                    row.usage is None and not row.usage_reported
                ), "pre-wire request claimed usage"
        assert seen == set(self.finals), "unobserved logical final"
