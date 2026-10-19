"""Channel-ledger accounting for the classification layer: one row per vendor request.

THE CLAIM UNDER TEST: every HTTP request the layer places is countable in the
session's ``classification`` channel, with an UNKNOWN amount (never 0) when the
vendor sent no cost, and a request the layer did not place records nothing.

What is real and what is not, because a green test here is only worth what it
exercises: the SERVICE, the real ``cascade`` and the three REAL leg classes
(``RadientVendor`` / ``TypeSafeVendor`` / ``OpenRouterVendor``, their credential
resolution, their 401 ladder and their body parsing) all run; only the
``httpx.AsyncClient``'s TRANSPORT is a ``MockTransport``, so no socket is opened
and nothing is billed. The tests that need a stub leg (``install_legs``) are in
``test_service.py``/``test_decide.py`` and are unaffected: a stub never calls
:func:`~local_operator.classification.accounting.note_post`.

The sink here is a plain ``list.append``; ``test_accounting_session.py`` binds a
real ``Session`` and ``test_accounting_sdk.py`` drives a real ``sdk.open_session``.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

import httpx
import pytest

from local_operator.classification.accounting import (
    answered_record,
    failed_record,
    interrupted_record,
)
from local_operator.classification.recommend import RecommendationRequest
from local_operator.classification.service import ClassificationService
from local_operator.classification.types import (
    DecisionResponse,
    DecisionVendorError,
    Question,
)
from local_operator.session.channel_spend import (
    STATUS_CANCELLED,
    STATUS_FAILED,
    STATUS_OK,
    ChannelSpendRecord,
    combine,
    fold_records,
)
from tests.unit.classification.support import candidate

pytestmark = pytest.mark.asyncio

ROSTER = [
    candidate("minerva-deploy", kind="skill", description="Deploy a Minerva service"),
    candidate("tunnel", kind="guide", description="Phone access over a tunnel"),
]

QUESTION = Question(
    id="monitor_materiality",
    kind="choice",
    instructions="Is this change material?",
    criteria={"material": "yes", "ignorable": "no"},
)


def request(message: str = "deploy core to qa") -> RecommendationRequest:
    return RecommendationRequest(user_message=message, context=None, candidates=ROSTER)


def answer_body(
    *, cost: float | None = 0.000027, question: str = "recommend_skill", choice: str = "none"
) -> dict[str, Any]:
    """A 200 body the real parser accepts; ``cost=None`` omits ``usage`` entirely.

    ``choice`` must be an option the question offered: the real parser refuses an id it
    never sent (that refusal is itself a case below), so a mismatched default would
    test the refusal instead of the thing named.
    """
    body: dict[str, Any] = {"answers": {question: {"choice": choice, "confidence": 1.0}}}
    if cost is not None:
        body["usage"] = {"cost": cost, "input_tokens": 100, "output_tokens": 0}
    return body


class Wire:
    """The mock transport, plus the record of what reached it (the ground truth)."""

    def __init__(self, handler: Callable[[httpx.Request], Any]) -> None:
        self.hosts: list[str] = []
        self._handler = handler

    async def __call__(self, req: httpx.Request) -> httpx.Response:
        self.hosts.append(req.url.host)
        outcome = self._handler(req)
        if asyncio.iscoroutine(outcome):
            outcome = await outcome
        return outcome


def build(
    manager: Any, wire: Wire, *, settings: dict[str, Any] | None = None
) -> tuple[ClassificationService, list[ChannelSpendRecord]]:
    """A real service whose client rides ``wire`` and whose sink is a list."""
    effective = settings if settings is not None else {"classification": {"auto": True}}
    subject = ClassificationService(config_dir=manager, settings=effective)
    rows: list[ChannelSpendRecord] = []
    subject.bind_spend_sink(rows.append)
    # The service's own keep-alive slot is where the production client lives; filling
    # it with a mock-transport client exercises every real code path above the socket.
    subject._http = httpx.AsyncClient(transport=httpx.MockTransport(wire))
    return subject, rows


def requests_counted(rows: list[ChannelSpendRecord]) -> int:
    return int(sum(row.units for row in rows))


# -- answered ----------------------------------------------------------------


async def test_an_answer_records_one_billed_row_from_the_vendors_own_cost(manager) -> None:
    wire = Wire(lambda req: httpx.Response(200, json=answer_body(cost=0.000027)))
    subject, rows = build(manager, wire)

    recommendation = await subject.recommend_resources(request())

    assert recommendation.skipped is None and recommendation.vendor == "radient"
    assert wire.hosts == ["api.radienthq.com"]
    (row,) = rows
    assert row.channel == "classification"
    assert (row.provider, row.model) == ("radient", "jev-1.13")
    assert (row.units, row.unit) == (1, "calls")
    assert row.amount_micro == 27
    assert (row.billing_basis, row.cost_source, row.status) == (
        "billed",
        "server_reported",
        STATUS_OK,
    )
    assert row.record_id.startswith("classification:")


async def test_a_response_with_no_cost_is_an_unknown_amount_never_zero(manager) -> None:
    wire = Wire(lambda req: httpx.Response(200, json=answer_body(cost=None)))
    subject, rows = build(manager, wire)

    await subject.recommend_resources(request())

    (row,) = rows
    assert row.amount_micro is None, "no usage.cost is NOT a free call"
    assert row.billing_basis == "not_tracked", "the record normalises a basis with no amount"
    assert row.cost_source == "none"
    assert row.status == STATUS_OK and row.units == 1
    # ...and the published object says so instead of painting a number.
    published = combine(rows, tracked=True)
    assert published["knowledge"] == "unknown"
    assert published["by_basis"]["not_tracked_calls"] == 1
    assert published["total_micro"] == 0


async def test_a_radient_top_level_cost_object_records_a_server_reported_row(manager) -> None:
    """The measured Radient shape (no ``usage.cost``, a top-level ``cost``) is a receipt.

    Before the cascade existed this row read ``amount_micro=None`` /
    ``cost_source="none"``, so the session's classification spend was invisible
    and the operator's cost line showed ``$—`` beside a priced inference row.
    """

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                **answer_body(cost=None),
                "cost": {
                    "v": 1,
                    "amount_usd": 0.000013692,
                    "basis": "billed",
                    "units": 326,
                    "unit": "token",
                    "provider": "typesafe",
                    "model": "jev-1.13",
                    "usage_record_id": "a1b2c3d4e5f60718293a4b5c",
                    "price_version": "decisions-rate@2026-09-19",
                },
            },
        )

    subject, rows = build(manager, Wire(handler))
    await subject.recommend_resources(request())

    (row,) = rows
    assert row.amount_micro == 14, "13.692 micro-USD: the server's own exact figure"
    assert (row.billing_basis, row.cost_source, row.status) == (
        "billed",
        "server_reported",
        STATUS_OK,
    )
    assert row.cost_source != "none"
    # The amount lands in the billed bucket, so the published object says exact.
    published = combine(rows, tracked=True)
    assert published["total_micro"] == 14
    assert published["by_basis"]["billed"] == 14
    assert published["knowledge"] == "exact"


async def test_a_quoted_radient_figure_is_estimated_on_the_row(manager) -> None:
    """A quote must reach the row as ``estimated`` — never printed as a charge."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                **answer_body(cost=None),
                "cost": {"v": 1, "amount_usd": 0.00042, "basis": "quote"},
            },
        )

    subject, rows = build(manager, Wire(handler))
    await subject.recommend_resources(request())

    (row,) = rows
    assert row.amount_micro == 420
    assert row.billing_basis == "estimated"
    assert combine(rows, tracked=True)["by_basis"]["estimated"] == 420


async def test_a_server_amount_with_no_basis_is_billed_not_not_tracked(manager) -> None:
    """An object that states an amount but no basis lands in the BILLED bucket.

    The wire's absent basis maps to the adapter's ``not_tracked`` sentinel, which is
    truthy: without the parser mapping it to ``None`` the row kept it, so a genuinely
    billed amount fell into ``not_tracked_micro`` and the row contradicted the
    convention this module documents for a stated figure.
    """

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={**answer_body(cost=None), "cost": {"v": 1, "amount_usd": 0.000027}},
        )

    subject, rows = build(manager, Wire(handler))
    await subject.recommend_resources(request())

    (row,) = rows
    assert row.amount_micro == 27
    assert row.billing_basis == "billed", "no stated basis defaults to today's billed"
    assert row.cost_source == "server_reported"
    published = combine(rows, tracked=True)
    assert published["by_basis"]["billed"] == 27
    assert published["by_basis"]["not_tracked_micro"] == 0


async def test_a_negative_server_amount_records_an_unknown_row(manager) -> None:
    """A negative wire amount is not a credit: the row records unknown, never -2_500_000."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={**answer_body(cost=None), "cost": {"v": 1, "amount_usd": -2.5}},
        )

    subject, rows = build(manager, Wire(handler))
    await subject.recommend_resources(request())

    (row,) = rows
    assert row.amount_micro is None
    assert row.cost_source == "none"
    assert combine(rows, tracked=True)["knowledge"] == "unknown"


async def test_a_vendor_stated_zero_is_a_known_zero(manager) -> None:
    wire = Wire(lambda req: httpx.Response(200, json=answer_body(cost=0.0)))
    subject, rows = build(manager, wire)
    await subject.recommend_resources(request())
    assert rows[0].amount_micro == 0
    assert combine(rows, tracked=True)["knowledge"] == "exact"


async def test_the_third_party_legs_cost_is_provider_reported(manager) -> None:
    # Pin the OpenRouter leg: radient and typesafe 500, so the third leg answers.
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.host == "openrouter.ai":
            return httpx.Response(200, json=answer_body(cost=0.0001))
        return httpx.Response(500, json={"error": "down"})

    subject, rows = build(manager, Wire(handler))
    await subject.recommend_resources(request())
    answered = [row for row in rows if row.status == STATUS_OK]
    assert [(row.provider, row.cost_source, row.amount_micro) for row in answered] == [
        ("openrouter", "provider_reported", 100)
    ]


# -- failed / fallback / retry walk ------------------------------------------


async def test_a_leg_fallback_records_the_failed_leg_and_the_answering_leg(manager) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.host == "api.radienthq.com":
            return httpx.Response(503, json={"error": "overloaded"})
        return httpx.Response(200, json=answer_body(cost=0.00002))

    wire = Wire(handler)
    subject, rows = build(manager, wire)
    recommendation = await subject.recommend_resources(request())

    assert recommendation.vendor == "typesafe"
    assert wire.hosts == ["api.radienthq.com", "api.typesafe.ai"]
    assert [(row.provider, row.status) for row in rows] == [
        ("radient", STATUS_FAILED),
        ("typesafe", STATUS_OK),
    ]
    assert rows[0].amount_micro is None and rows[0].detail == "server"
    assert requests_counted(rows) == len(wire.hosts), "rows == requests that left the process"


async def test_every_request_of_a_failed_two_walk_pass_is_countable(
    manager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The headline: 3 legs x 2 walks of 503s is SIX requests, and six rows count them."""
    monkeypatch.setattr("local_operator.classification.service.DEFAULT_RETRY_BACKOFF_S", 0.0)
    wire = Wire(lambda req: httpx.Response(503, json={"error": "down"}))
    subject, rows = build(manager, wire)

    recommendation = await subject.recommend_resources(request())

    assert recommendation.skipped == "error"
    assert len(wire.hosts) == 6
    assert len(rows) == 6 and requests_counted(rows) == 6
    assert {row.status for row in rows} == {STATUS_FAILED}
    # A refused request is presumed unbilled: attempts are counted, knowledge is not
    # degraded by them (the ledger's documented presumption).
    assert combine(rows, tracked=True)["by_basis"]["not_tracked_calls"] == 0


async def test_a_401_re_send_inside_one_attempt_counts_both_requests(manager) -> None:
    """The 401 ladder re-sends inside ONE ``decide``; a per-attempt constant 1 would undercount."""
    wire = Wire(lambda req: httpx.Response(401, json={"error": "no"}))
    subject, rows = build(manager, wire)

    await subject.recommend_resources(request())

    assert requests_counted(rows) == len(wire.hosts)
    assert any(row.units > 1 for row in rows), "the re-send is inside one attempt's row"


async def test_a_schema_rejection_is_a_failed_row_and_is_not_retried(manager) -> None:
    message = '[{"path": ["questions", "recommend_skill", "criteria"], "message": "bad"}]'
    wire = Wire(lambda req: httpx.Response(400, json={"error": {"message": message}}))
    subject, rows = build(manager, wire)

    recommendation = await subject.recommend_resources(request())

    assert recommendation.skipped == "error"
    assert len(wire.hosts) == 1
    (row,) = rows
    assert (row.status, row.detail, row.units) == (STATUS_FAILED, "schema-rejected", 1)


async def test_a_200_the_parser_refuses_is_served_so_possibly_billed(manager) -> None:
    wire = Wire(lambda req: httpx.Response(200, json={"no": "answers block"}))
    subject, rows = build(manager, wire)

    await subject.recommend_resources(request())

    unusable = [row for row in rows if row.detail == "unusable-response"]
    assert unusable, "a 200 we could not use is not a refusal"
    assert {(row.status, row.amount_micro) for row in unusable} == {(STATUS_OK, None)}
    # ok + unknown => the session total reads as a lower bound, not as exact.
    assert combine(unusable, tracked=True)["knowledge"] != "exact"


async def test_a_refused_connection_is_unbilled_but_a_lost_response_is_not(
    manager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Never-connected is presumed unbilled; a request SENT and then lost is not.

    The two are indistinguishable to the cascade (both are ``transport``) and
    opposite to the ledger: a refused connection cannot have been charged, a read
    timeout may have been.
    """
    monkeypatch.setattr("local_operator.classification.service.DEFAULT_RETRY_BACKOFF_S", 0.0)

    def refused(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=req)

    def lost(req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("no reply", request=req)

    refused_subject, refused_rows = build(manager, Wire(refused))
    lost_subject, lost_rows = build(manager, Wire(lost))
    await refused_subject.recommend_resources(request())
    await lost_subject.recommend_resources(request())

    assert {row.status for row in refused_rows} == {STATUS_FAILED}
    assert {row.status for row in lost_rows} == {STATUS_CANCELLED}
    assert {row.amount_micro for row in lost_rows} == {None}
    assert {row.detail for row in lost_rows} == {"response-lost"}
    assert combine(lost_rows, tracked=True)["knowledge"] != "exact"


# -- timeout / cancel ---------------------------------------------------------


async def test_a_timeout_records_a_not_tracked_cancelled_row_so_the_attempt_is_countable(
    manager,
) -> None:
    started = asyncio.Event()

    async def hang(req: httpx.Request) -> httpx.Response:
        started.set()
        await asyncio.sleep(30)
        return httpx.Response(200, json=answer_body())

    wire = Wire(hang)
    subject, rows = build(
        manager, wire, settings={"classification": {"auto": True, "timeoutMs": 80}}
    )

    recommendation = await subject.recommend_resources(request())

    assert recommendation.skipped == "timeout"
    assert started.is_set() and len(wire.hosts) == 1
    (row,) = rows
    assert (row.status, row.amount_micro, row.billing_basis) == (
        STATUS_CANCELLED,
        None,
        "not_tracked",
    )
    assert (row.units, row.detail, row.provider) == (1, "interrupted-in-flight", "radient")
    assert combine(rows, tracked=True)["by_basis"]["not_tracked_calls"] == 1


async def test_a_dispose_mid_request_records_the_abandoned_attempt(manager) -> None:
    started = asyncio.Event()

    async def hang(req: httpx.Request) -> httpx.Response:
        started.set()
        await asyncio.sleep(30)
        return httpx.Response(200, json=answer_body())

    subject, rows = build(manager, Wire(hang))
    call = asyncio.create_task(subject.recommend_resources(request()))
    await asyncio.wait_for(started.wait(), timeout=5)

    await subject.aclose()  # the dispose hook's call: cancels what is in flight
    await asyncio.gather(call, return_exceptions=True)

    (row,) = rows
    assert (row.status, row.amount_micro, row.detail) == (
        STATUS_CANCELLED,
        None,
        "interrupted-in-flight",
    )


async def test_a_cancelled_joiner_does_not_cancel_the_shared_call_or_double_record(manager) -> None:
    """``recommend_resources`` shields the attempt: a caller's cancel is not the attempt's."""
    release = asyncio.Event()
    started = asyncio.Event()

    async def slow(req: httpx.Request) -> httpx.Response:
        started.set()
        await release.wait()
        return httpx.Response(200, json=answer_body())

    wire = Wire(slow)
    subject, rows = build(manager, wire)
    first = asyncio.create_task(subject.recommend_resources(request()))
    await asyncio.wait_for(started.wait(), timeout=5)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert rows == [], "nothing recorded until the shared request actually settles"

    release.set()
    for _ in range(200):
        if rows:
            break
        await asyncio.sleep(0.01)
    assert len(wire.hosts) == 1 and len(rows) == 1 and rows[0].status == STATUS_OK


# -- records nothing ----------------------------------------------------------


async def test_a_cache_hit_records_nothing(manager) -> None:
    wire = Wire(lambda req: httpx.Response(200, json=answer_body()))
    subject, rows = build(manager, wire)

    first = await subject.recommend_resources(request())
    assert len(rows) == 1 and len(wire.hosts) == 1
    again = await subject.recommend_resources(request())

    assert again.latency_s == 0.0 and again.vendor == first.vendor
    assert len(rows) == 1 and len(wire.hosts) == 1, "a hit places no request and writes no row"


async def test_a_decide_cache_hit_records_nothing(manager) -> None:
    wire = Wire(
        lambda req: httpx.Response(
            200, json=answer_body(question="monitor_materiality", choice="material")
        )
    )
    subject, rows = build(manager, wire)

    assert await subject.decide(state="delta", question=QUESTION) is not None
    assert await subject.decide(state="delta", question=QUESTION) is not None

    assert len(rows) == 1 and len(wire.hosts) == 1


async def test_decide_is_accounted_by_the_same_single_point(manager) -> None:
    """Monitor gate, title-fit and supplements all call ``decide``: one point covers them."""
    wire = Wire(
        lambda req: httpx.Response(
            200, json=answer_body(question="monitor_materiality", choice="material")
        )
    )
    subject, rows = build(manager, wire)

    answer = await subject.decide(state="a changed monitor", question=QUESTION)

    assert answer is not None and answer.value == "material"
    (row,) = rows
    assert (row.channel, row.provider, row.status, row.amount_micro) == (
        "classification",
        "radient",
        STATUS_OK,
        27,
    )


async def test_concurrent_identical_callers_share_one_request_and_one_row(manager) -> None:
    """The double-count guard: a joiner shares the in-flight attempt, so one POST, one row."""
    gate = asyncio.Event()

    async def held(req: httpx.Request) -> httpx.Response:
        await gate.wait()
        return httpx.Response(200, json=answer_body())

    wire = Wire(held)
    subject, rows = build(manager, wire)
    calls = [asyncio.create_task(subject.recommend_resources(request())) for _ in range(4)]
    await asyncio.sleep(0.05)
    gate.set()
    results = await asyncio.gather(*calls)

    assert all(result.skipped is None for result in results)
    assert len(wire.hosts) == 1
    assert len(rows) == 1
    assert len({row.record_id for row in rows}) == len(rows)


async def test_a_row_replayed_into_the_fold_changes_nothing(manager) -> None:
    """Idempotence by ``record_id``: the ledger's own guard against any double delivery."""
    wire = Wire(lambda req: httpx.Response(200, json=answer_body()))
    subject, rows = build(manager, wire)
    await subject.recommend_resources(request())

    fold = fold_records(rows + rows + rows)
    assert len(fold) == 1 and fold.total_known_micro() == 27


async def test_auto_false_opens_no_socket_and_records_nothing(manager) -> None:
    wire = Wire(lambda req: httpx.Response(200, json=answer_body()))
    subject = ClassificationService(
        config_dir=manager, settings={"classification": {"auto": False}}
    )
    rows: list[ChannelSpendRecord] = []
    subject.bind_spend_sink(rows.append)

    subject.warm_up()  # the factory calls this; disabled must not even build the client
    recommendation = await subject.recommend_resources(request())
    assert await subject.decide(state="delta", question=QUESTION) is None

    assert recommendation.skipped == "disabled"
    assert subject._http is None, "no client object, so no socket"
    assert wire.hosts == [] and rows == []


async def test_no_sink_bound_is_a_quiet_no_op(manager) -> None:
    wire = Wire(lambda req: httpx.Response(200, json=answer_body()))
    subject, rows = build(manager, wire)
    subject.bind_spend_sink(None)

    recommendation = await subject.recommend_resources(request())

    assert recommendation.skipped is None and len(wire.hosts) == 1
    assert rows == [], "an unbound service must not emit a session record"


async def test_a_raising_sink_cannot_fail_the_classification(manager) -> None:
    wire = Wire(lambda req: httpx.Response(200, json=answer_body()))
    subject, _ = build(manager, wire)

    def broken(record: ChannelSpendRecord) -> None:
        raise RuntimeError("ledger exploded")

    subject.bind_spend_sink(broken)
    recommendation = await subject.recommend_resources(request())

    assert recommendation.skipped is None and recommendation.vendor == "radient"


async def test_a_leg_with_no_credential_places_no_request_and_records_no_row(bare_manager) -> None:
    wire = Wire(lambda req: httpx.Response(200, json=answer_body()))
    subject, rows = build(bare_manager, wire)

    recommendation = await subject.recommend_resources(request())

    assert recommendation.skipped == "no-vendor"
    assert wire.hosts == [] and rows == []


async def test_an_open_breaker_places_no_request_and_records_no_row(
    manager, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("local_operator.classification.service.DEFAULT_RETRY_BACKOFF_S", 0.0)
    wire = Wire(lambda req: httpx.Response(503, json={"error": "down"}))
    subject, rows = build(manager, wire)
    for index in range(3):
        await subject.recommend_resources(request(f"message {index}"))
    placed, recorded = len(wire.hosts), len(rows)
    assert subject._circuit_open

    opened = await subject.recommend_resources(request("after the breaker opened"))

    assert opened.skipped == "circuit-open"
    assert (len(wire.hosts), len(rows)) == (placed, recorded)
    assert requests_counted(rows) == placed


async def test_concurrent_distinct_questions_do_not_share_post_counts(manager) -> None:
    """Two callers share a leg, but each attempt owns its task-local POST count.

    A meter on the leg would read two for one or both calls; both requests must
    enter the mock transport before either can finish to make that race structural.
    """
    arrived = 0
    both_entered = asyncio.Event()

    async def held(req: httpx.Request) -> httpx.Response:
        nonlocal arrived
        arrived += 1
        if arrived == 2:
            both_entered.set()
        await asyncio.wait_for(both_entered.wait(), timeout=5)
        return httpx.Response(
            200, json=answer_body(question="monitor_materiality", choice="material")
        )

    wire = Wire(held)
    subject, rows = build(manager, wire)
    answers = await asyncio.gather(
        subject.decide(state="change A", question=QUESTION),
        subject.decide(state="change B", question=QUESTION),
    )
    assert all(answer is not None for answer in answers)
    assert len(wire.hosts) == 2
    assert len(rows) == 2
    assert [row.units for row in rows] == [1, 1]
    assert len({row.record_id for row in rows}) == 2
    await subject.aclose()


@pytest.mark.parametrize(
    "cost", [True, False, float("nan"), float("inf"), -float("inf"), -0.01, 1e308, 10**400]
)
async def test_malformed_or_unstorable_cost_keeps_answer_and_records_unknown(manager, cost) -> None:
    """The vendor served a valid answer: its bad receipt must not replace that answer."""
    import json

    wire = Wire(lambda req: httpx.Response(200, content=json.dumps(answer_body(cost=cost))))
    subject, rows = build(manager, wire)
    result = await subject.recommend_resources(request())
    assert result.skipped is None and result.vendor == "radient"
    (row,) = rows
    assert (row.status, row.amount_micro, row.billing_basis, row.cost_source) == (
        "ok",
        None,
        "not_tracked",
        "none",
    )
    assert row.units == 1
    await subject.aclose()


@pytest.mark.parametrize("outcome", ["answer", "error", "cancel"])
@pytest.mark.parametrize("bound", [True, False])
async def test_record_factory_fault_never_changes_vendor_outcome(
    manager, monkeypatch, outcome, bound
) -> None:
    """Success object and original exception identity survive the ENTIRE bookkeeping seam."""
    from local_operator.classification import accounting
    from local_operator.classification.types import (
        DecisionRequest,
        DecisionResponse,
        DecisionVendorError,
    )

    answer = DecisionResponse(vendor="radient", model="mock", answers={}, cost_usd=0.000027)
    original = (
        asyncio.CancelledError()
        if outcome == "cancel"
        else DecisionVendorError("original", kind="server")
    )

    class Vendor:
        name = "radient"
        model_id = "mock"

        async def credential(self, config_dir):
            from pydantic import SecretStr

            return SecretStr("synthetic")

        async def decide(self, request, *, timeout_s):
            accounting.note_post()
            if outcome != "answer":
                raise original
            return answer

    subject = ClassificationService(config_dir=manager, settings={"classification": {"auto": True}})
    rows = []
    if bound:
        subject.bind_spend_sink(rows.append)
    built = []

    def broken(*args, **kwargs):
        built.append(True)
        raise RuntimeError("record construction failed")

    for name in ("answered_record", "failed_record", "interrupted_record"):
        monkeypatch.setattr(accounting, name, broken)
    if outcome == "answer":
        assert (
            await subject._metered_decide(Vendor(), DecisionRequest(state="x", questions=()), 1)
            is answer
        )
    else:
        with pytest.raises(type(original)) as caught:
            await subject._metered_decide(Vendor(), DecisionRequest(state="x", questions=()), 1)
        assert caught.value is original
    assert rows == []
    assert len(built) == int(bound), "an unbound service must not invoke a record factory"


# -- billing identity: the priced-from id, not the vendor echo ---------------


@pytest.mark.parametrize(
    "server_id",
    [
        None,
        "jev-1.13",
        "jev-1.13.0",
        "typesafe/jev-1.13",
        "jev-1.13-20261019",
        "typesafe/jev-1.13-20261019",
    ],
)
async def test_every_server_emittable_decision_id_lands_on_the_canonical_row(
    server_id: str | None,
) -> None:
    """One model, one ledger identity: the aliases the server can state collapse."""
    response = DecisionResponse(
        vendor="radient",
        # The vendor's own echo, deliberately a different spelling from the
        # canonical id. It stays on ``response.model`` for evidence.
        model="jev-1.13.0",
        answers={},
        cost_usd=1.2e-05,
        cost_model=server_id,
    )
    row = answered_record(response, posts=1, requested_model="jev-1.13")

    assert row.model == "jev-1.13"
    assert row.detail == "vendor-model=jev-1.13.0"


async def test_without_a_cost_object_the_requested_leg_model_is_the_fallback() -> None:
    response = DecisionResponse(
        vendor="radient", model="jev-1.13.0", answers={}, cost_usd=None, cost_model=None
    )
    row = answered_record(response, posts=1, requested_model="typesafe/jev-1.13")

    assert row.model == "jev-1.13"
    assert row.amount_micro is None, "unknown stays unknown"


async def test_a_failed_or_interrupted_row_canonicalises_its_model_argument() -> None:
    error = DecisionVendorError("down", kind="server")

    failed = failed_record("radient", "jev-1.13.0", error, posts=1)
    interrupted = interrupted_record("radient", "typesafe/jev-1.13-20261019", posts=1)

    assert failed.model == "jev-1.13"
    assert interrupted.model == "jev-1.13"
    assert (
        failed_record("typesafe", "another-model", error, posts=1).model == "another-model"
    ), "an unknown id is not rewritten into the JEV family"


async def test_the_service_records_the_priced_id_while_the_echo_rides_detail(manager) -> None:
    """The measured wire: top-level ``model`` echoes the pinned id, cost.model prices."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                **answer_body(cost=None),
                "model": "jev-1.13.0",
                "cost": {
                    "v": 1,
                    "amount_usd": 1.1424e-05,
                    "basis": "billed",
                    "units": 272,
                    "unit": "token",
                    "provider": "typesafe",
                    "model": "jev-1.13",
                    "usage_record_id": "a1b2c3d4e5f60718293a4b5c",
                    "price_version": "decisions-rate@2026-09-19",
                },
            },
        )

    subject, rows = build(manager, Wire(handler))
    await subject.recommend_resources(request())

    (row,) = rows
    assert row.model == "jev-1.13", "the row is the priced-from id, not the echo"
    assert row.detail == "vendor-model=jev-1.13.0", "the echo survives as evidence"
    assert row.amount_micro == 11, "1.1424e-05 USD rounds to 11 micro-USD"
