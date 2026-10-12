"""Zero-cost REAL-SDK acceptance matrix: request-tree observation + owned shutdown.

Each case drives the REAL chain the contract names: SDK ``open_session`` (the
real ``session_factory.create_session``) -> real ``Session``/AgentLoop -> the
shipped ``task`` tool launching a REAL child ``Session`` through the real
subagent fork of the parent's ``SessionStreamFn`` -> real ``read``/``hub``/
``bash`` tools running in-process. ONLY the transport/client seam is scripted:
``MockClient.stream`` (the shipped ``hosting="test"`` wire) is replaced by a
FIFO turn script, exactly the seam the mock provider already exists for. No
Session, AgentLoop, launcher or fork semantics are stubbed, no network request
is made and no inference is billed -- the mock wire never leaves the process.

Isolation: each case runs under its own scratch root (HOME / config / cwd via
``SessionRoots`` with ``allow_volatile``), ambient ``LOCAL_OPERATOR_*`` decoys
are installed, and every inherited ``CMUX_*``/``LOP_*`` variable is stripped so
a rig can never reach the operator's real session state.

Two things these tests deliberately DO NOT claim, matching the implementation
docs: (a) retry/fallback attempts BELOW the stream wrapper may carry bills a
request-final row cannot see -- the partial-then-fail case pins the UNKNOWN
disposition such attempts must produce, and invoice-total accuracy needs the
separate transport-attempt seam; (b) a cancellation-resistant cleanup is
returned as an honest ``complete=False`` timeout with the task retained -- the
outer process-group bound belongs to the supervising host, not this wrapper.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, cast

import pytest

from local_operator.harness.types import (
    ChatRequest,
    StreamEndEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    StreamUsageEvent,
    Usage,
)
from local_operator.session.request_observation import RequestFinalObservation

_PARENT_MARKER = "PSTART-parent-turn"
_CHILD_MARKER = "CCHILD-child-turn"
_GRAND_MARKER = "GGGG-grandchild-turn"

#: A wire item that parks the stream forever (until the consumer cancels it) --
#: the shape a real provider takes when it stops answering.
_PARK = object()


class Wire:
    """The scripted transport seam: FIFO turns per conversation scope.

    Scope is derived from the LAST user message of each request (parent /
    child / grandchild / helper), so interleaving between concurrently running
    trees cannot misroute a scripted turn: each scope consumes its own queue in
    its own order. An unscripted request for a scope fails loudly rather than
    inventing a reply.
    """

    def __init__(self) -> None:
        self._turns: dict[str, list[list[Any]]] = {}
        self.requests: list[ChatRequest] = []

    def script(self, scope: str, *turns: Any) -> None:
        self._turns.setdefault(scope, []).extend(turns)

    @staticmethod
    def scope_of(request: ChatRequest) -> str:
        last_user = next(
            (message for message in reversed(request.messages) if message.role == "user"), None
        )
        text = (last_user.text or "") if last_user is not None else ""
        if _GRAND_MARKER in text:
            return "grand"
        if _CHILD_MARKER in text:
            return "child"
        if _PARENT_MARKER in text:
            return "parent"
        # The completion-attention delivery: a settled model-owned job is
        # auto-delivered back into the session that owns it as a new turn
        # (existing harness behaviour this matrix must not fight). Routed per
        # OWNER by the job's label so concurrent deliveries cannot swap turns.
        if "background job 'kid'" in text:
            return "delivery-root"
        if "background job 'grand'" in text:
            return "delivery-child"
        return "helper"

    def scoped(self, scope: str) -> list[ChatRequest]:
        return [request for request in self.requests if self.scope_of(request) == scope]

    async def stream(
        self, request: ChatRequest, api_key: str | None, oauth_access: Any = None
    ) -> AsyncIterator[Any]:
        self.requests.append(request)
        scope = self.scope_of(request)
        turns = self._turns.get(scope)
        assert turns, (
            f"no scripted turn for scope {scope!r} "
            f"(purpose={request.purpose!r}, requests so far={len(self.requests)})"
        )
        self._turns[scope] = turns[1:]
        turn = turns[0]
        if callable(turn):
            # A dynamic turn: built from the concrete request, for tool calls
            # whose arguments (a just-launched job id) only exist at run time.
            turn = turn(request)
        for item in turn:
            if isinstance(item, BaseException):
                raise item
            if item is _PARK:
                await asyncio.Event().wait()
                continue
            yield item


def text_turn(text: str = "done", *, usage: Usage | None = None, host: str | None = None):
    usage = usage if usage is not None else Usage(input_tokens=10, output_tokens=5)
    return [
        StreamTextDelta(delta=text),
        StreamEndEvent(stop_reason="stop", usage=usage, served_provider=host),
    ]


def tool_turn(name: str, args: dict[str, Any], call_id: str, *, usage: Usage | None = None):
    usage = usage if usage is not None else Usage(input_tokens=10, output_tokens=5)
    return [
        StreamToolCallDelta(index=0, id=call_id, name=name),
        StreamToolCallDelta(index=0, argument_delta=json.dumps(args)),
        StreamEndEvent(stop_reason="toolUse", usage=usage),
    ]


def wait_for_launched_job(request: ChatRequest) -> list[Any]:
    """Build a ``wait`` call for the job id named in the previous tool result.

    The id only exists at run time, so the turn is a callable. Waiting (rather
    than ending the turn) is what keeps a launched grandchild ALIVE: a child
    session that finishes while its own job runs disposes it -- cancellation
    before first request -- which is a shape this matrix records separately
    rather than fights.
    """
    import re

    last = request.messages[-1] if request.messages else None
    text = str(getattr(last, "text", "") or "")
    match = re.search(r"job ([0-9a-f]+)", text)
    assert match, f"no launched job id in the previous tool result: {text[:200]!r}"
    return tool_turn("wait", {"job_id": match.group(1), "wait_ms": 30000}, "call-wait")


@pytest.fixture
def scratch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """A scratch root in the shape the SDK isolation contract expects."""
    from local_operator import agent_shell
    from local_operator.sdk import SessionRoots

    home = tmp_path / "home"
    root = home / ".local-operator"
    agent_home = home / "local-operator-home"
    work = home / "work"
    for directory in (root, agent_home, work):
        directory.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "decoy-config"))
    monkeypatch.setenv("LOCAL_OPERATOR_HOME", str(tmp_path / "decoy-home"))
    monkeypatch.delenv(agent_shell.AGENT_SHELL_ENV, raising=False)
    monkeypatch.delenv(agent_shell.MAY_DELEGATE_ENV, raising=False)
    monkeypatch.delenv(agent_shell.ALLOW_NESTED_SESSION_ENV, raising=False)
    for key in [k for k in os.environ if k.startswith(("CMUX_", "LOP_"))]:
        monkeypatch.delenv(key, raising=False)
    return SessionRoots(config_dir=root, agent_home=agent_home, cwd=work, allow_volatile=True)


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch) -> Wire:
    from local_operator.providers.clients import MockClient

    scripted = Wire()
    monkeypatch.setattr(MockClient, "stream", scripted.stream)
    return scripted


def _spec(**overrides: Any) -> Any:
    from local_operator.session.spec import ApprovalPolicy, SessionSpec

    # ``auto`` because the matrix drives REAL tools (task/bash) unattended:
    # the SDK's default ``refuse`` policy would turn every launch into an
    # approval refusal instead of exercising the launcher under observation.
    values: dict[str, Any] = {
        "hosting": "test",
        "model": "mock",
        "approvals": ApprovalPolicy.auto(),
    }
    values.update(overrides)
    return SessionSpec(**values)


@asynccontextmanager
async def _opened(scratch: Any) -> AsyncIterator[Any]:
    from local_operator import sdk

    async with sdk.open_session(_spec(), roots=scratch) as session:
        yield cast(Any, session)


async def _settle(session: Any, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not [job for job in session.jobs.list() if job.status == "running"]:
            return
        await asyncio.sleep(0.02)
    raise AssertionError("owned jobs did not settle inside the test budget")


async def _quiesce(session: Any, wire: Wire, seen: list[Any], timeout: float = 20.0) -> None:
    """Wait until every request the wire saw has its row and nothing is in flight.

    ``_settle`` covers JOBS only, but a settled job triggers a completion
    DELIVERY turn on its owner -- itself a request with a row of its own.
    Asserting before that turn finishes races a request that is already on the
    wire: seen lags requests by exactly one.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        idle = not [job for job in session.jobs.list() if job.status == "running"]
        if idle and len(seen) == len(wire.requests):
            # One settle tick so a delivery that was mid-spawn is visible.
            await asyncio.sleep(0.05)
            if len(seen) == len(wire.requests) and not [
                job for job in session.jobs.list() if job.status == "running"
            ]:
                return
        await asyncio.sleep(0.02)
    raise AssertionError(f"tree did not quiesce: {len(seen)} rows vs {len(wire.requests)} requests")


def _rows(seen: list[RequestFinalObservation], **match: Any) -> list[RequestFinalObservation]:
    return [row for row in seen if all(getattr(row, key) == value for key, value in match.items())]


def _transcript_text(session: Any) -> str:
    """Every transcript entry as JSON text -- the honest read of what landed."""
    return "\n".join(
        json.dumps(getattr(entry, "payload", entry), default=str)
        for entry in session._transcript.entries()
    )


# ---------------------------------------------------------------------------
# Ordinary unset -- the fast path, through the real session
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ordinary_unset_is_allocation_free_and_turn_output_unchanged(
    scratch: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.model import configure as configure_module

    calls: list[Any] = []
    real_builder = configure_module.build_request_final_observation
    monkeypatch.setattr(
        configure_module,
        "build_request_final_observation",
        lambda **kwargs: calls.append(kwargs) or real_builder(**kwargs),
    )
    async with _opened(scratch) as session:
        reply = await session.prompt(f"{_PARENT_MARKER}: say hello")
        assert reply is None, "prompt's return is the admission ack, not the text"
        assert "mock provider" in _transcript_text(session), "the real mock reply landed"
        assert session._stream_fn._request_observers is None, "nobody subscribed; holder is None"
        assert calls == [], "an unsubscribed request must not build a record"


# ---------------------------------------------------------------------------
# Single native tool loop -- one row per request, duplicate frames once
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_native_tool_loop_one_row_per_request_with_native_results(
    scratch: Any, wire: Wire
) -> None:
    target = scratch.cwd / "note.txt"
    target.write_text("native-file-content-marker")
    usage = Usage(input_tokens=100, output_tokens=20, provider="p", model_id="m")
    wire.script("parent", tool_turn("read", {"path": str(target)}, "call-read"))
    # Duplicate usage + end in ONE request: the final snapshot replaces, never adds.
    wire.script(
        "parent",
        [
            StreamTextDelta(delta="read it"),
            StreamEndEvent(stop_reason="stop", usage=usage, served_provider=None),
            StreamEndEvent(stop_reason="stop", usage=usage, served_provider=None),
        ],
    )

    async with _opened(scratch) as session:
        seen: list[RequestFinalObservation] = []
        session.subscribe_requests(seen.append)
        await session.prompt(f"{_PARENT_MARKER}: read the note")

    assert [row.purpose for row in seen] == ["turn", "turn"]
    assert len({row.request_id for row in seen}) == 2
    assert all(row.ok for row in seen)
    # The native tool ran for real: the second request carried the tool result.
    second = wire.scoped("parent")[1]
    assert any(
        "native-file-content-marker" in (message.text or "")
        for message in second.messages
        if message.role != "assistant"
    ), "the read tool result must ride the second request"


# ---------------------------------------------------------------------------
# Native task/hub -- real child and grandchild sessions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_native_task_tree_records_parent_child_and_grandchild(
    scratch: Any, wire: Wire
) -> None:
    target = scratch.cwd / "child-note.txt"
    target.write_text("child-file-marker")
    wire.script(
        "parent",
        tool_turn(
            "task", {"label": "kid", "prompt": f"{_CHILD_MARKER}: read and delegate"}, "call-task"
        ),
    )
    wire.script("parent", text_turn("parent done"))
    wire.script("child", tool_turn("read", {"path": str(target)}, "call-child-read"))
    wire.script("child", tool_turn("hub", {"op": "list"}, "call-child-hub"))
    wire.script(
        "child",
        tool_turn(
            "task", {"label": "grand", "prompt": f"{_GRAND_MARKER}: just answer"}, "call-grand-task"
        ),
    )
    wire.script("child", wait_for_launched_job)
    wire.script("child", text_turn("child done"))
    wire.script("grand", tool_turn("read", {"path": str(target)}, "call-grand-read"))
    wire.script("grand", text_turn("grand done"))
    wire.script("delivery-root", text_turn("delivery ack"))
    wire.script("delivery-child", text_turn("child delivery ack"))

    async with _opened(scratch) as session:
        root_id = session.session_id
        seen: list[RequestFinalObservation] = []
        session.subscribe_requests(seen.append)
        await session.prompt(f"{_PARENT_MARKER}: delegate the work")
        await _settle(session)
        await _quiesce(session, wire, seen)

    # EXACTLY ONCE, for every request the wire actually saw -- including the
    # completion deliveries the harness auto-runs for settled jobs.
    assert len(seen) == len(wire.requests), (len(seen), len(wire.requests))
    assert len({row.request_id for row in seen}) == len(wire.requests)
    root_rows = [row for row in seen if row.session_id == root_id]
    child_rows = [row for row in seen if row.parent_session_id == root_id]
    child_id = child_rows[0].session_id if child_rows else ""
    grand_rows = [row for row in seen if row.parent_session_id == child_id and child_id]
    assert len(root_rows) == len(wire.scoped("parent")) + len(wire.scoped("delivery-root"))
    assert len(child_rows) == len(wire.scoped("child")) + len(wire.scoped("delivery-child"))
    assert len(grand_rows) == len(wire.scoped("grand"))
    assert all(row.tree_id == root_id for row in seen)
    assert all(row.job_id is None and row.actor_role is None for row in root_rows)
    assert all(row.job_id is not None and row.actor_role == "task" for row in child_rows)
    assert all(row.job_id is not None and row.actor_role == "task" for row in grand_rows)
    assert child_rows[0].job_id != grand_rows[0].job_id, "each delegation is its own job"
    # Cache identity: one key per session, distinct across the tree, carried on
    # every request of that session -- what a consumer may group by (never by
    # provider/model).
    parent_keys = {row.requested_prompt_cache_key for row in root_rows}
    child_keys = {row.requested_prompt_cache_key for row in child_rows}
    grand_keys = {row.requested_prompt_cache_key for row in grand_rows}
    assert len(parent_keys) == len(child_keys) == len(grand_keys) == 1
    assert len(parent_keys | child_keys | grand_keys) == 3
    # read AND hub genuinely ran: the launch request carries both results.
    child_requests = wire.scoped("child")
    launch_carrier = "\n".join(
        (message.text or "")
        for message in child_requests[2].messages
        if message.role != "assistant"
    )
    assert "child-file-marker" in launch_carrier, "the read result rides the launch request"
    assert "no subagents launched" in launch_carrier, "the hub result rides it too"


# ---------------------------------------------------------------------------
# Preexisting fork + unsubscribe coverage through the SDK
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subscription_covers_preforked_children_and_unsubscribes(
    scratch: Any, wire: Wire
) -> None:
    wire.script("parent", text_turn("first"))
    wire.script("parent", text_turn("second"))

    async with _opened(scratch) as session:
        stream = session._stream_fn
        preexisting = stream.fork("pre-fork-child")
        # The shared carrier is a ROOT REFERENCE (not a copied holder value),
        # so the fork made before any subscription resolves the later install.
        assert preexisting._observation_root is stream
        # ...while ordinary routing state stays child-local.
        assert preexisting._provider_affinity is not stream._provider_affinity

        seen: list[RequestFinalObservation] = []
        unsubscribe = session.subscribe_requests(seen.append)
        await session.prompt(f"{_PARENT_MARKER}: one")
        unsubscribe()
        unsubscribe()  # idempotent
        await session.prompt(f"{_PARENT_MARKER}: two")

    assert len(seen) == 1 and seen[0].session_id == session.session_id


# ---------------------------------------------------------------------------
# Cache/cost -- exact receipts, duplicate snapshots never double
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_and_cost_receipts_sum_once_subtract_nothing(scratch: Any, wire: Wire) -> None:
    """Three actors, three stamped receipts (.12/.03/0), every receipt
    DUPLICATED on the wire: the sum must be .15 -- a duplicate snapshot
    replaces the provisional one upstream and can never add up to .30."""

    def receipt(cost: float) -> Usage:
        return Usage(
            input_tokens=100,
            output_tokens=20,
            cache_read_tokens=80,
            reasoning_tokens=5,
            usd_cost=cost,
            provider="p",
            model_id="m",
        )

    zero, paid, small = receipt(0.0), receipt(0.12), receipt(0.03)

    def text_dup(text: str, usage: Usage) -> list[Any]:
        return [
            StreamTextDelta(delta=text),
            StreamUsageEvent(usage=usage),
            StreamUsageEvent(usage=usage),
            StreamEndEvent(stop_reason="stop", usage=usage),
        ]

    wire.script(
        "parent",
        [
            StreamToolCallDelta(index=0, id="call-task", name="task"),
            StreamToolCallDelta(
                index=0,
                argument_delta=json.dumps({"label": "kid", "prompt": f"{_CHILD_MARKER}: delegate"}),
            ),
            StreamUsageEvent(usage=zero),
            StreamUsageEvent(usage=zero),
            StreamEndEvent(stop_reason="toolUse", usage=zero),
        ],
    )
    wire.script("parent", text_dup("parent done", paid))
    wire.script("child", text_dup("child done", small))
    wire.script("delivery-root", text_dup("delivery ack", zero))

    async with _opened(scratch) as session:
        seen: list[RequestFinalObservation] = []
        session.subscribe_requests(seen.append)
        await session.prompt(f"{_PARENT_MARKER}: fan out")
        await _settle(session)
        await _quiesce(session, wire, seen)

    from local_operator.model.costs import cost_summary

    with_usage = [row for row in seen if row.usage is not None]
    assert len(with_usage) == len(seen), "every request here carried a receipt"
    components = [row.usage for row in with_usage]
    total, unknown = cost_summary(components, recorded_only=True)
    assert unknown is False
    assert total is not None and abs(total - 0.15) < 1e-12, "duplicate snapshots must not make 0.30"
    assert sum(1 for row in with_usage if row.usage is not None and row.usage.usd_cost == 0.12) == 1
    for row in with_usage:
        assert row.usage is not None
        # Raw buckets, exactly as delivered: cache stays its own bucket,
        # reasoning stays a subset of output, input is never re-folded.
        assert row.usage.cache_read_tokens == 80, "cache stays its own bucket"
        assert row.usage.reasoning_tokens == 5, "reasoning is never added to output"
        assert row.usage.input_tokens == 100, "input is never re-folded with cache"
        assert row.usage.output_tokens == 20


# ---------------------------------------------------------------------------
# Unknown/error -- missing usage is unknown, zero is known, cancellation rows
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_usage_is_unknown_reported_zero_remains_known(
    scratch: Any, wire: Wire
) -> None:
    # The failure carries PARTIAL OUTPUT before dying: the failover layer never
    # replays an attempt whose content reached the caller, so this is one
    # request that failed before ANY usage frame -- the receipt-less shape.
    wire.script(
        "parent",
        [StreamTextDelta(delta="partial"), RuntimeError("synthetic mid-stream death before usage")],
    )
    wire.script(
        "parent",
        [
            StreamTextDelta(delta="ok"),
            StreamEndEvent(
                stop_reason="stop", usage=Usage(usd_cost=0.15, provider="p", model_id="m")
            ),
        ],
    )
    wire.script(
        "parent",
        [
            StreamTextDelta(delta="free"),
            StreamEndEvent(
                stop_reason="stop", usage=Usage(usd_cost=0.0, provider="p", model_id="m")
            ),
        ],
    )

    async with _opened(scratch) as session:
        seen: list[RequestFinalObservation] = []
        session.subscribe_requests(seen.append)
        try:
            await session.prompt(f"{_PARENT_MARKER}: fail")
        except (
            Exception
        ):  # noqa: BLE001 -- a failed turn may surface or record; the ROW is the contract
            pass
        await session.prompt(f"{_PARENT_MARKER}: paid")
        await session.prompt(f"{_PARENT_MARKER}: zero")

    failed, paid, zero = seen
    assert failed.usage is None and failed.usage_reported is False
    assert failed.usage_coverage == "missing" and failed.ok is False
    assert failed.outcome != "stop", "a failed attempt must not read as success"
    assert paid.usage is not None and paid.usage.usd_cost == 0.15
    assert zero.usage is not None and zero.usage.usd_cost == 0.0 and zero.usage_reported
    # The caller-side rules the receipt shape exists for:
    assert (
        sum(
            row.usage.usd_cost
            for row in seen
            if row.usage is not None and row.usage.usd_cost is not None
        )
        == 0.15
    ), "known lower bound retained"
    assert (
        sum(1 for row in seen if not row.usage_reported) == 1
    ), "the missing row contributes uncertainty"
    from local_operator.model.costs import cost_summary

    known_zero, unknown_zero = cost_summary([zero.usage], recorded_only=True)
    assert known_zero == 0.0 and unknown_zero is False, "a reported zero stays a known zero"


@pytest.mark.asyncio
async def test_cancelled_request_records_unknown_usage(scratch: Any, wire: Wire) -> None:
    wire.script("parent", [StreamTextDelta(delta="partial"), _PARK])

    async with _opened(scratch) as session:
        seen: list[RequestFinalObservation] = []
        session.subscribe_requests(seen.append)
        turn = asyncio.create_task(session.prompt(f"{_PARENT_MARKER}: park"))
        deadline = time.monotonic() + 10
        while not wire.requests and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
        assert wire.requests, "the parked request must have started"
        session.abort("test-cancel")
        await turn  # the turn unwinds; the observation lands in the wrapper's finally
        await session.shutdown_owned_work("test-cleanup", timeout_s=15)

    assert len(seen) == 1
    assert seen[0].usage is None and seen[0].usage_reported is False
    assert seen[0].usage_coverage == "missing" and seen[0].ok is False


# ---------------------------------------------------------------------------
# Served route -- requested vs served, host separate; partial-then-fail unknown
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_served_route_and_upstream_host_are_distinct_facts(scratch: Any, wire: Wire) -> None:
    """The serving ROUTE and the aggregator upstream HOST are different facts.

    On a primary success the failover layer stamps the on-the-wire spec onto
    usage -- that stamp IS the served-identity channel (source
    ``usage_stamp``), which is why a synthetic provider string on the frame is
    overwritten with the real route. The end frame's ``served_provider`` is
    the UPSTREAM HOST and rides separately; neither replaces the other.

    A requested!=served stamp under a REAL cross-route fallback needs
    configured fallback chains; that stamp survival is pinned at the stream
    seam in ``test_request_observation.py``, and failover-attempt cost
    coverage is disclosed incomplete in the module docstring."""
    host = "upstream-host-X"
    wire.script(
        "parent",
        [
            StreamToolCallDelta(index=0, id="call-task", name="task"),
            StreamToolCallDelta(
                index=0,
                argument_delta=json.dumps(
                    {"label": "kid", "prompt": f"{_CHILD_MARKER}: route check"}
                ),
            ),
            StreamEndEvent(
                stop_reason="toolUse",
                usage=Usage(input_tokens=10, output_tokens=5),
                served_provider=host,
            ),
        ],
    )
    wire.script(
        "parent",
        [
            StreamTextDelta(delta="parent done"),
            StreamEndEvent(
                stop_reason="stop",
                usage=Usage(
                    input_tokens=10, output_tokens=5, provider="will-be-overwritten", model_id="x"
                ),
                served_provider=host,
            ),
        ],
    )
    wire.script(
        "child",
        [
            StreamTextDelta(delta="child done"),
            StreamEndEvent(
                stop_reason="stop",
                usage=Usage(input_tokens=10, output_tokens=5),
                served_provider=host,
            ),
        ],
    )
    wire.script(
        "delivery-root",
        [
            StreamTextDelta(delta="delivery ack"),
            StreamEndEvent(
                stop_reason="stop",
                usage=Usage(input_tokens=10, output_tokens=5),
                served_provider=host,
            ),
        ],
    )

    async with _opened(scratch) as session:
        root_id = session.session_id
        seen: list[RequestFinalObservation] = []
        session.subscribe_requests(seen.append)
        await session.prompt(f"{_PARENT_MARKER}: route check")
        await _settle(session)
        await _quiesce(session, wire, seen)

    assert {row.session_id for row in seen} == {root_id} | {
        row.session_id for row in seen if row.parent_session_id == root_id
    }
    assert len({row.session_id for row in seen}) == 2, "parent and its task child both checked"
    for row in seen:
        assert row.requested_provider == "test" and row.requested_model_id == "mock"
        assert (
            row.served_provider == "test" and row.served_model_id == "mock"
        ), "the served stamp is the on-wire spec, not the frame's host"
        assert row.served_identity_source == "usage_stamp"
        assert row.upstream_host == host, "the upstream host is recorded as its own fact"


@pytest.mark.asyncio
async def test_partial_usage_then_fail_keeps_reported_receipt_never_fabricated(
    scratch: Any, wire: Wire
) -> None:
    """A failed attempt that DID report usage keeps that reported receipt --
    faithful capture, never fabricated, never re-priced.

    What this row CANNOT prove is disclosed in the module docstring: earlier
    retry/fallback attempts below the stream wrapper may carry bills it never
    saw, so invoice totals must not be claimed from request-final rows. The
    no-receipt-at-all shape (usage=None, unknown) is pinned by the
    ``missing_usage`` case."""
    wire.script(
        "parent",
        [
            StreamUsageEvent(
                usage=Usage(input_tokens=7, usd_cost=0.12, provider="p", model_id="m")
            ),
            StreamTextDelta(delta="x"),
            RuntimeError("die after partial usage"),
        ],
    )
    wire.script(
        "parent",
        [
            StreamTextDelta(delta="second"),
            StreamEndEvent(
                stop_reason="stop", usage=Usage(usd_cost=0.03, provider="p", model_id="m")
            ),
        ],
    )

    async with _opened(scratch) as session:
        seen: list[RequestFinalObservation] = []
        session.subscribe_requests(seen.append)
        try:
            await session.prompt(f"{_PARENT_MARKER}: partial")
        except Exception:  # noqa: BLE001 -- the ROW is the contract, not the raise
            pass
        await session.prompt(f"{_PARENT_MARKER}: follow-up")

    first, second = seen
    assert first.ok is False and first.usage_reported is True
    assert first.usage is not None and first.usage.usd_cost == 0.12, "the reported receipt is kept"
    assert first.usage_coverage == "reported_final"
    assert first.outcome != "stop"
    assert second.usage is not None and second.usage.usd_cost == 0.03
    from local_operator.model.costs import cost_summary

    known, unknown = cost_summary([first.usage, second.usage], recorded_only=True)
    assert known is not None and abs(known - 0.15) < 1e-12 and unknown is False


# ---------------------------------------------------------------------------
# Helper requests -- compaction/naming through the same canonical wrapper
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_helper_requests_record_their_purposes(scratch: Any, wire: Wire) -> None:
    wire.script("helper", text_turn("SUMMARY"))
    wire.script("helper", text_turn("A title"))
    wire.script("parent", text_turn("turn done"))

    async with _opened(scratch) as session:
        seen: list[RequestFinalObservation] = []
        session.subscribe_requests(seen.append)
        summary = await session._one_shot_complete("system", "summarize the transcript")
        assert summary == "SUMMARY"
        errand = session._errand_request(session._lowest_effort(session._model), "sys", "name this")
        title = await session._drain_errand(errand)
        assert title == "A title"
        await session.prompt(f"{_PARENT_MARKER}: and a turn")

    assert [row.purpose for row in seen] == ["compaction", "naming", "turn"]
    assert len({row.request_id for row in seen}) == 3, "helpers are not folded into the turn"


# ---------------------------------------------------------------------------
# Cleanup tree -- abort control keeps the child; shutdown joins it (and its tree)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_abort_control_leaves_child_alive_and_shutdown_joins_it(
    scratch: Any, wire: Wire
) -> None:
    wire.script(
        "parent",
        tool_turn(
            "task", {"label": "kid", "prompt": f"{_CHILD_MARKER}: sleep a while"}, "call-task"
        ),
    )
    wire.script("parent", text_turn("parent done"))
    wire.script("child", tool_turn("bash", {"command": "sleep 3", "timeout": 30.0}, "call-bash"))
    wire.script("child", text_turn("child done"))

    async with _opened(scratch) as session:
        session.subscribe_requests(lambda _row: None)
        await session.prompt(f"{_PARENT_MARKER}: delegate a sleeper")
        # The child is mid-bash when we abort the PARENT: child continuation is
        # the existing, unchanged contract for a plain abort.
        await asyncio.sleep(0.3)
        running_before = [job for job in session.jobs.list() if job.status == "running"]
        assert running_before, "the child must still be running before the abort"
        session.abort("operator-cancel")
        await asyncio.sleep(0.2)
        still_running = [job for job in session.jobs.list() if job.status == "running"]
        assert still_running, "abort must NOT stop the child -- first-Esc semantics stay"

        receipt = await session.shutdown_owned_work("test-teardown", timeout_s=20)
        assert receipt.complete is True, receipt
        assert receipt.pending_job_ids == ()
        assert not [job for job in session.jobs.list() if job.status == "running"]
        await _settle(session)


@pytest.mark.asyncio
async def test_shutdown_joins_grandchildren_and_background_bash_jobs(
    scratch: Any, wire: Wire
) -> None:
    wire.script(
        "parent",
        tool_turn(
            "task", {"label": "kid", "prompt": f"{_CHILD_MARKER}: hold the tree"}, "call-task"
        ),
    )
    wire.script("parent", text_turn("parent done"))
    wire.script(
        "child",
        tool_turn("bash", {"command": "sleep 30", "timeout": 60.0, "background": True}, "call-bg"),
    )
    wire.script(
        "child",
        tool_turn("task", {"label": "grand", "prompt": f"{_GRAND_MARKER}: hold"}, "call-grand"),
    )
    wire.script("child", text_turn("child done"))
    wire.script(
        "grand",
        tool_turn("bash", {"command": "sleep 30", "timeout": 60.0, "background": True}, "call-bg2"),
    )
    wire.script("grand", text_turn("grand done"))

    async with _opened(scratch) as session:
        observed: list[RequestFinalObservation] = []
        session.subscribe_requests(observed.append)
        await session.prompt(f"{_PARENT_MARKER}: hold the tree")
        await asyncio.sleep(0.4)  # let the children register their sleeping jobs
        receipt = await session.shutdown_owned_work("test-teardown", timeout_s=30)
        assert receipt.complete is True, receipt
        assert receipt.pending_job_ids == ()
        assert not [job for job in session.jobs.list() if job.status == "running"]
        # No request reaches the wire after the tree settled.
        assert len(wire.requests) == len(observed) + 0, "every row is a real request"


# ---------------------------------------------------------------------------
# Shutdown race/timeout -- late admission refused; timeout is honest
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_late_admission_is_refused_and_the_gate_is_the_discriminator(
    scratch: Any, wire: Wire, monkeypatch: pytest.MonkeyPatch
) -> None:
    launched: list[str] = []

    def _record_launch(label: str, prompt: str, **kwargs: Any) -> str:
        launched.append(label)
        return "fake-job-id"

    monkeypatch.setattr("local_operator.session.session.run_subagent", _record_launch)

    async with _opened(scratch) as session:
        # Normal admission: the launch path is reachable.
        assert session._launch_subagent("ok", "prompt") == "fake-job-id"
        assert launched == ["ok"]

        # Mid-shutdown: the latch refuses BEFORE any work is done.
        session._owned_work_closing = True
        with pytest.raises(RuntimeError, match="shutting down"):
            session._launch_subagent("late", "prompt")
        assert launched == ["ok"], "a refused launch must not reach run_subagent"

        # Discriminator: clear BOTH gates and the same call escapes -- proving
        # the refusal above came from the latch, not from some other check.
        session._owned_work_closing = False
        session._disposed = False
        assert session._launch_subagent("escape", "prompt") == "fake-job-id"
        assert launched == ["ok", "escape"]

        # After the real wrapper, admission is closed for good.
        await session.shutdown_owned_work("test-done", timeout_s=15)
        with pytest.raises(RuntimeError, match="shutting down"):
            session._launch_subagent("terminal", "prompt")


@pytest.mark.asyncio
async def test_cancellation_resistant_cleanup_times_out_honestly(
    scratch: Any, wire: Wire, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with _opened(scratch) as session:

        async def stubborn() -> None:
            await asyncio.sleep(0.5)

        monkeypatch.setattr(session, "_dispose_impl", stubborn)
        receipt = await session.shutdown_owned_work("test-stubborn", timeout_s=0.05)
        assert receipt.complete is False
        assert any("timeout" in error for error in receipt.errors), receipt.errors
        task = session._owned_shutdown_task
        assert (
            task is not None and not task.done() and not task.cancelled()
        ), "retained, never pretend-killed"
        assert session._owned_shutdown_result is None, "a timeout is not cached as the receipt"

        # The retained cleanup still lands; a second call reports the truth.
        await task
        again = await session.shutdown_owned_work("test-stubborn", timeout_s=5)
        assert again.complete is True
        assert session._owned_shutdown_result is not None


# ---------------------------------------------------------------------------
# Callback failure/unsubscribe -- outcome intact, count surfaced on the receipt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_observer_failure_does_not_mask_outcome_and_surfaces_on_receipt(
    scratch: Any, wire: Wire
) -> None:
    wire.script("parent", text_turn("still fine"))
    wire.script("parent", text_turn("second"))
    wire.script("parent", text_turn("third"))

    async with _opened(scratch) as session:
        good: list[RequestFinalObservation] = []

        def bad(_row: RequestFinalObservation) -> None:
            raise ValueError("observer bug")

        session.subscribe_requests(bad)
        unsubscribe = session.subscribe_requests(good.append)
        await session.prompt(f"{_PARENT_MARKER}: one")
        assert "still fine" in _transcript_text(session), "the provider outcome is untouched"
        unsubscribe()
        unsubscribe()  # double unsubscribe is safe
        await session.prompt(f"{_PARENT_MARKER}: two")
        await session.prompt(f"{_PARENT_MARKER}: three")

        receipt = await session.shutdown_owned_work("test-done", timeout_s=15)

    assert len(good) == 1, "unsubscribed handlers stop receiving, the live one keeps receiving"
    assert (
        receipt.observer_callback_errors == 3
    ), "every failed delivery is counted on the receipt (bad raised once per request)"
    assert receipt.complete is True, receipt


# ---------------------------------------------------------------------------
# Canonical disposal join, cancellation truth, hook-failure truth (R2/R3/R7)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shutdown_joins_an_already_running_disposal_and_reports_truthfully(
    scratch: Any, wire: Wire
) -> None:
    """Ordinary ``dispose`` and ``shutdown_owned_work`` share ONE cleanup
    (R2): a shutdown racing a disposal-in-progress waits for THAT cleanup -- it
    must not start a second (early-returning) one, must NOT cache an incomplete
    receipt, and returns the completed receipt once the shared cleanup settles."""
    async with _opened(scratch) as session:
        entered = asyncio.Event()
        release = asyncio.Event()
        runs: list[str] = []

        async def teardown() -> None:
            runs.append("hook")
            entered.set()
            await release.wait()

        session.add_dispose_hook(teardown, last=True)
        ordinary = asyncio.create_task(session.dispose())
        await asyncio.wait_for(entered.wait(), 10)

        receipt = await session.shutdown_owned_work("join-existing-dispose", timeout_s=0.03)
        assert receipt.complete is False, "the shared cleanup is still pending"
        assert any("timeout" in error for error in receipt.errors), receipt.errors
        assert session._owned_shutdown_result is None, "an incomplete receipt is never cached"
        assert not ordinary.done(), "the ordinary disposal still owns the pending cleanup"

        release.set()
        await ordinary
        again = await session.shutdown_owned_work("join-existing-dispose", timeout_s=10)
        assert again.complete is True, again.errors
        third = await session.shutdown_owned_work("anything", timeout_s=10)
        assert third is again, "a complete receipt is the same object on repeat calls"
        assert runs == ["hook"], "simultaneous entrants run the cleanup ONCE"


@pytest.mark.asyncio
async def test_cancelled_cleanup_is_incomplete_unverified_and_never_cached(
    scratch: Any, wire: Wire, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R3: a cancelled cleanup is UNVERIFIED -- never success, never cached --
    whether the cancellation came from a raising hook or from outside, and the
    ordinary dispose entry stays non-raising for it (historical behaviour)."""
    # A: the cleanup task ends cancelled because a hook raised CancelledError.
    async with _opened(scratch) as session:

        async def cancelled_teardown() -> None:
            raise asyncio.CancelledError("scripted host cleanup cancellation")

        session.add_dispose_hook(cancelled_teardown, last=True)
        receipt = await session.shutdown_owned_work("cancelled-cleanup", timeout_s=5)
        assert receipt.complete is False, "cancellation proves nothing about resources"
        assert any("cancelled" in error for error in receipt.errors), receipt.errors
        task = session._owned_shutdown_task
        assert task is not None and task.cancelled()
        assert session._owned_shutdown_result is None, "a cancelled cleanup must never cache"

        again = await session.shutdown_owned_work("cancelled-cleanup", timeout_s=5)
        assert again.complete is False
        assert session._owned_shutdown_result is None
        await session.dispose()  # joining a cancelled canonical cleanup is quiet

    # B: an EXTERNAL cancel of the retained canonical cleanups reports the same truth.
    async with _opened(scratch) as session:

        async def stubborn() -> None:
            await asyncio.sleep(30)

        monkeypatch.setattr(session, "_dispose_impl", stubborn)
        first = await session.shutdown_owned_work("external-cancel", timeout_s=0.05)
        assert first.complete is False
        task = session._owned_shutdown_task
        assert task is not None
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        second = await session.shutdown_owned_work("external-cancel", timeout_s=5)
        assert second.complete is False
        assert any("cancelled" in error for error in second.errors), second.errors
        assert session._owned_shutdown_result is None
        await session.dispose()


@pytest.mark.asyncio
async def test_failed_cleanup_hook_is_reported_never_read_as_success(
    scratch: Any, wire: Wire
) -> None:
    """R7: a disposal hook that raises is logged-and-continued by ordinary
    disposal (pre-existing policy) but the owning receipt must NOT read that
    as success -- incomplete, the hook failure named, and never cached."""
    async with _opened(scratch) as session:

        async def failed_teardown() -> None:
            raise RuntimeError("scripted resource close failed before release")

        session.add_dispose_hook(failed_teardown, last=True)
        receipt = await session.shutdown_owned_work("failed-hook", timeout_s=5)
        assert receipt.complete is False
        assert any("failed_teardown" in error for error in receipt.errors), receipt.errors
        assert any("RuntimeError" in error for error in receipt.errors), receipt.errors
        assert session._owned_shutdown_result is None
        # Ordinary disposal itself remains non-raising, exactly as before.
        await session.dispose()
        again = await session.shutdown_owned_work("failed-hook", timeout_s=5)
        assert again.complete is False and session._owned_shutdown_result is None


# ---------------------------------------------------------------------------
# Observer-originated CancelledError on the REAL session (R5)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_observer_cancelled_error_is_isolated_on_the_real_session(
    scratch: Any, wire: Wire
) -> None:
    """A subscriber raising CancelledError must not surface as a cancellation
    of a COMPLETED request: the turn returns its text, the helper returns its
    text, the healthy subscriber still receives, and the count lands on the
    receipt (R5)."""
    wire.script("parent", text_turn("still fine"))
    wire.script("helper", text_turn("HELP"))
    async with _opened(scratch) as session:
        good: list[RequestFinalObservation] = []

        def cancelling(_row: RequestFinalObservation) -> None:
            raise asyncio.CancelledError("observer tried to cancel a finished request")

        session.subscribe_requests(cancelling)
        session.subscribe_requests(good.append)
        await session.prompt(f"{_PARENT_MARKER}: cancellation isolation")
        helper = await session._one_shot_complete("system", "helper after observer cancellation")
        assert helper == "HELP", "the helper call must return its text, not raise CancelledError"
        assert "still fine" in _transcript_text(session)
        assert len(good) == 2
        receipt = await session.shutdown_owned_work("test-done", timeout_s=15)
    assert receipt.complete is True, receipt
    assert receipt.observer_callback_errors == 2


# ---------------------------------------------------------------------------
# Root-tree scope of a child-installed subscription + child receipt (R6/R8)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_child_subscription_is_root_scoped_and_child_receipt_reports_tree_errors(
    scratch: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R8/R6 on a REAL native tree: a subscription installed through a CHILD
    observes the whole original root tree -- ancestors and siblings, every
    ``tree_id`` naming the root -- and the child's shutdown receipt reports the
    SHARED root holder's callback-error count, not a local zero."""
    from types import MethodType

    from local_operator.harness import subagent as subagent_module
    from local_operator.providers.clients import MockClient

    release = asyncio.Event()
    both_parked = asyncio.Event()
    state: dict[str, Any] = {"children": [], "parked": 0, "parent_calls": 0}
    original_construct = subagent_module._construct_child_session

    async def capture(*args: Any, **kwargs: Any) -> Any:
        child = await original_construct(*args, **kwargs)
        state["children"].append(child)
        return child

    monkeypatch.setattr(subagent_module, "_construct_child_session", capture)

    async def stream(
        _bound: Any, request: ChatRequest, api_key: Any = None, oauth_access: Any = None
    ) -> AsyncIterator[Any]:
        last_user = next((m for m in reversed(request.messages) if m.role == "user"), None)
        text = str(getattr(last_user, "text", "") or "")
        if "CHILD-HOLD" in text:
            state["parked"] += 1
            if state["parked"] >= 2:
                both_parked.set()
            yield StreamTextDelta(delta="child-started")
            await release.wait()
            yield StreamEndEvent(stop_reason="stop", usage=Usage(input_tokens=10, output_tokens=2))
            return
        if "HELPER-" in text:
            yield StreamTextDelta(delta="helper-ok")
            yield StreamEndEvent(stop_reason="stop", usage=Usage(input_tokens=10, output_tokens=2))
            return
        if "background job" in text:
            yield StreamTextDelta(delta="delivery noted")
            yield StreamEndEvent(stop_reason="stop", usage=Usage(input_tokens=10, output_tokens=2))
            return
        # The parent's launch turn, then its follow-up after the tool results.
        state["parent_calls"] += 1
        if state["parent_calls"] == 1:
            for index, child_id in enumerate(("a", "b")):
                yield StreamToolCallDelta(index=index, id=f"call-{child_id}", name="task")
                yield StreamToolCallDelta(
                    index=index,
                    argument_delta=json.dumps(
                        {"label": f"kid-{child_id}", "prompt": "CHILD-HOLD: wait for the test"}
                    ),
                )
            yield StreamEndEvent(
                stop_reason="toolUse", usage=Usage(input_tokens=10, output_tokens=2)
            )
        else:
            yield StreamTextDelta(delta="parent done")
            yield StreamEndEvent(stop_reason="stop", usage=Usage(input_tokens=10, output_tokens=2))

    monkeypatch.setattr(MockClient, "stream", MethodType(stream, object()))

    async with _opened(scratch) as session:
        await session.prompt("PARENT-LAUNCH: start two children")
        await asyncio.wait_for(both_parked.wait(), 20)
        assert len(state["children"]) == 2
        child, sibling = state["children"]
        assert child.session_id != sibling.session_id

        seen: list[RequestFinalObservation] = []

        def failing(_row: RequestFinalObservation) -> None:
            raise ValueError("native child observer callback failed")

        child.subscribe_requests(seen.append)
        child.subscribe_requests(failing)

        # Helpers through the parent, the subscribing child, and its SIBLING:
        # all three rows must land on the child-installed subscription (R8),
        # and every row must name the ROOT tree.
        helpers = (
            await session._one_shot_complete("system", "HELPER-PARENT"),
            await child._one_shot_complete("system", "HELPER-CHILD"),
            await sibling._one_shot_complete("system", "HELPER-SIBLING"),
        )
        assert helpers == ("helper-ok",) * 3
        observed_sessions = {row.session_id for row in seen}
        assert {
            session.session_id,
            child.session_id,
            sibling.session_id,
        } <= observed_sessions, "a child-installed subscription covers ancestors and siblings"
        assert all(row.tree_id == session.session_id for row in seen), "tree_id names the ROOT"

        root_holder = session._stream_fn._request_observers
        assert root_holder is not None and root_holder.callback_errors >= 3

        release.set()
        child_receipt = await child.shutdown_owned_work("child-local-cleanup", timeout_s=20)
        # R6: the child receipt reads the SHARED root holder, not a local zero.
        assert child_receipt.observer_callback_errors == root_holder.callback_errors
        assert child_receipt.observer_callback_errors >= 3
        assert child_receipt.complete is True, child_receipt.errors

        root_receipt = await session.shutdown_owned_work("tree-cleanup", timeout_s=20)
        assert root_receipt.complete is True, root_receipt.errors
        await _settle(session)


@pytest.mark.asyncio
async def test_cancelling_an_ordinary_dispose_waiter_keeps_the_cleanup_joinable(
    scratch: Any, wire: Wire
) -> None:
    """R9/Q6: cancelling an ordinary dispose WAITER must not cancel the shared
    canonical cleanup. The waiter's CancelledError still propagates (waiter
    cancellation stays honest), a PEER disposer keeps joining, the resource
    close still lands on release, and a later owning receipt completes."""
    async with _opened(scratch) as session:
        entered = asyncio.Event()
        release = asyncio.Event()
        released: list[bool] = []

        async def cleanup() -> None:
            entered.set()
            await release.wait()
            released.append(True)

        session.add_dispose_hook(cleanup, last=True)
        first = await session.shutdown_owned_work("retain-on-timeout", timeout_s=0.01)
        assert first.complete is False
        canonical = session._owned_shutdown_task
        assert canonical is not None

        waiter = asyncio.create_task(session.dispose())
        peer = asyncio.create_task(session.dispose())
        await asyncio.sleep(0)
        assert waiter.cancel(), "the waiter cancellation must be accepted, not swallowed"
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert (
            not canonical.done() and not canonical.cancelled()
        ), "the shared cleanup must survive a waiter's cancellation"

        release.set()
        await peer
        assert released == [True], "the resource close must still land after release"
        final = await session.shutdown_owned_work("rejoin-original", timeout_s=5)
        assert final.complete is True, final.errors
        assert canonical.cancelled() is False


@pytest.mark.asyncio
async def test_root_receipt_includes_disposed_child_cleanup_failures(
    scratch: Any, wire: Wire, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R7-descendant/Q5: a native child whose own disposal is known-unverified
    must make the owning ancestor's receipt incomplete, with the descendant
    NAMED -- a settled child job is not proof its resources joined cleanly.
    (The healthy counterpart -- a cancelled child job with a VERIFIED cleanup
    stays root-complete -- is pinned by the abort-control tests above.)"""
    from local_operator.harness import subagent as subagent_module

    children: list[Any] = []
    original = subagent_module._construct_child_session

    async def capture(*args: Any, **kwargs: Any) -> Any:
        child = await original(*args, **kwargs)

        async def child_resource_close() -> None:
            raise RuntimeError("native child resource cleanup failure")

        child.add_dispose_hook(child_resource_close, last=True)
        children.append(child)
        return child

    monkeypatch.setattr(subagent_module, "_construct_child_session", capture)
    wire.script(
        "parent",
        tool_turn("task", {"label": "kid", "prompt": f"{_CHILD_MARKER}: finish"}, "call-task"),
    )
    wire.script("parent", text_turn("parent done"))
    wire.script("child", text_turn("child done"))
    wire.script("delivery-root", text_turn("delivery ack"))

    async with _opened(scratch) as session:
        seen: list[RequestFinalObservation] = []
        session.subscribe_requests(seen.append)
        await session.prompt(f"{_PARENT_MARKER}: launch a child")
        await _quiesce(session, wire, seen)
        assert len(children) == 1

        parent_receipt = await session.shutdown_owned_work("tree-cleanup", timeout_s=20)
        child_receipt = await children[0].shutdown_owned_work("child-local", timeout_s=10)

    assert child_receipt.complete is False
    assert any("child_resource_close" in error for error in child_receipt.errors)
    assert parent_receipt.complete is False, parent_receipt.errors
    assert any(
        "descendant" in error and children[0].session_id in error for error in parent_receipt.errors
    ), parent_receipt.errors
    assert any("child_resource_close" in error for error in parent_receipt.errors)
