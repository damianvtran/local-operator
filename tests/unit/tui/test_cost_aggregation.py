"""The parent's status-band total includes what its children spent.

Driven through the REAL app: the ledger, the 1 Hz poll and the ``agent_end``
handler are the wiring that was reported broken, and a test that called the
accessors directly would pass while the band stayed frozen.

The owner's report: "The cost calculation in the parent should be the aggregate
of cost from subagents."
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from local_operator.harness.types import Usage
from local_operator.model.registry import ModelInfo
from local_operator.tui.app import OperatorApp
from local_operator.tui.events import TurnEnded
from tests.unit.tui.test_band_panels import FakeSession, _async_factory, _fake_jobs

#: $10 in / $100 out per MTok — a million tokens is a round dollar figure.
_PARENT_MODEL = ModelInfo(
    id="opus", name="opus", description="", input_price=10.0, output_price=100.0
)
#: A tenth of the parent's rate, so a mispriced child is unmistakable.
_CHILD_MODEL = ModelInfo(
    id="haiku", name="haiku", description="", input_price=1.0, output_price=10.0
)
_MODELS = {"opus": _PARENT_MODEL, "haiku": _CHILD_MODEL}


def _resolving():
    return patch(
        "local_operator.model.configure.resolve_model_info",
        side_effect=lambda provider, model_id: _MODELS.get(
            model_id, ModelInfo(id=model_id, name=model_id, description="")
        ),
    )


class _TaskJob:
    """An ``AsyncJob``-shaped row carrying the two fields the accessor reads."""

    def __init__(
        self,
        job_id: str,
        *,
        usage: Usage | None = None,
        model_label: str | None = None,
        status: str = "running",
    ) -> None:
        self.id = job_id
        self.type = "task"
        self.status = status
        self.label = job_id
        self.start_time = 1_700_000_000.0
        self.result_text: str | None = None
        self.error_text: str | None = None
        self.settled_at: float | None = None
        self.trajectory: list[dict[str, Any]] | None = None
        self.usage = usage
        self.model_label = model_label
        self.child_jobs: Any | None = None


class _Session(FakeSession):
    """``FakeSession`` with a settable model label; the base one is read-only."""

    def __init__(self, model_label: str) -> None:
        super().__init__()
        self._model_label = model_label

    @property
    def model_label(self) -> str:
        return self._model_label

    @property
    def effective_model_label(self) -> str:
        return self._model_label


def _band_cost(app: OperatorApp) -> str:
    """What the status line is currently holding in its cost segment."""
    assert app._status is not None
    return app._status._cost


def _harvest(session: _Session) -> OperatorApp:
    """Exercise the production tree walk without mounting an unrelated frame."""
    app = OperatorApp(_async_factory(session))
    app._session = session
    app._harvest_subagent_costs()
    return app


@pytest.mark.asyncio
async def test_band_total_is_own_spend_plus_children() -> None:
    """A delegated turn's money shows up in the parent's figure.

    The child runs on a DIFFERENT model here on purpose: the whole reason the
    label is recorded per job is that pricing a cheap child at the parent's
    flagship rate — or the reverse — is the failure this replaces.
    """
    session = _Session("anthropic/opus")
    session.jobs = _fake_jobs(
        _TaskJob("kid", usage=Usage(input_tokens=1_000_000), model_label="anthropic/haiku")
    )
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 28)) as pilot:
        await pilot.pause()
        with _resolving():
            app.post_message(
                TurnEnded(False, None, context_tokens=0, usage=Usage(input_tokens=1_000_000))
            )
            await pilot.pause()
    # Parent burned $10 of its own; the child burned $1 at its own cheaper rate.
    assert app._total_cost == pytest.approx(10.0)
    assert app._subagent_costs == {"kid": pytest.approx(1.0)}
    assert app._spend_total() == pytest.approx(11.0)
    assert _band_cost(app) == "$11.00"


def test_harvest_counts_a_hundred_direct_children_once_each() -> None:
    session = _Session("anthropic/haiku")
    session.jobs = _fake_jobs(
        *(
            _TaskJob(
                f"kid-{index}",
                usage=Usage(input_tokens=1_000_000),
                model_label="anthropic/haiku",
            )
            for index in range(100)
        )
    )
    with _resolving():
        app = _harvest(session)
        app._harvest_subagent_costs()
    assert len(app._subagent_costs) == 100
    assert app._spend_total() == pytest.approx(100.0)


def test_harvest_walks_every_nesting_depth_without_counting_ancestors_twice() -> None:
    leaf = _TaskJob(
        "great-grandchild",
        usage=Usage(input_tokens=4_000_000),
        model_label="anthropic/haiku",
    )
    grandchild = _TaskJob(
        "grandchild",
        usage=Usage(input_tokens=3_000_000),
        model_label="anthropic/haiku",
    )
    child = _TaskJob("child", usage=Usage(input_tokens=2_000_000), model_label="anthropic/haiku")
    grandchild.child_jobs = _fake_jobs(leaf)
    child.child_jobs = _fake_jobs(grandchild)
    session = _Session("anthropic/opus")
    session.jobs = _fake_jobs(child)
    with _resolving():
        app = _harvest(session)
        for _ in range(10):
            app._harvest_subagent_costs()
    assert app._subagent_costs == {
        "child": pytest.approx(2.0),
        "grandchild": pytest.approx(3.0),
        "great-grandchild": pytest.approx(4.0),
    }
    assert app._spend_total() == pytest.approx(9.0)


def test_nested_harvest_preserves_provider_receipts_and_table_estimates() -> None:
    receipt = _TaskJob(
        "receipt",
        usage=Usage(input_tokens=9_000_000, usd_cost=0.25),
        model_label="openrouter/routed",
    )
    estimate = _TaskJob(
        "estimate",
        usage=Usage(input_tokens=2_000_000),
        model_label="anthropic/haiku",
    )
    parent = _TaskJob("parent", usage=Usage(input_tokens=1_000_000), model_label="anthropic/haiku")
    parent.child_jobs = _fake_jobs(receipt, estimate)
    session = _Session("anthropic/opus")
    session.jobs = _fake_jobs(parent)
    with _resolving():
        app = _harvest(session)
    assert app._subagent_costs == {
        "parent": pytest.approx(1.0),
        "receipt": pytest.approx(0.25),
        "estimate": pytest.approx(2.0),
    }
    assert app._spend_total() == pytest.approx(3.25)


def test_nested_cost_survives_branch_eviction_after_it_was_harvested() -> None:
    grandchild = _TaskJob(
        "grandchild",
        usage=Usage(input_tokens=2_000_000),
        model_label="anthropic/haiku",
        status="completed",
    )
    child = _TaskJob("child", usage=Usage(input_tokens=1_000_000), model_label="anthropic/haiku")
    child.child_jobs = _fake_jobs(grandchild)
    session = _Session("anthropic/opus")
    session.jobs = _fake_jobs(child)
    with _resolving():
        app = _harvest(session)
        child.child_jobs = _fake_jobs()
        app._harvest_subagent_costs()
    assert app._spend_total() == pytest.approx(3.0)


@pytest.mark.asyncio
async def test_poll_moves_the_total_while_a_child_keeps_spending() -> None:
    """The band must not freeze between parent turns.

    A delegated child outlives the turn that launched it: the parent finishes,
    the band goes idle, and the child keeps burning tokens for minutes. Harvest
    only at turn end and the number sits still through exactly the window in
    which it is moving.
    """
    session = _Session("anthropic/opus")
    kid = _TaskJob("kid", usage=Usage(input_tokens=1_000_000), model_label="anthropic/haiku")
    session.jobs = _fake_jobs(kid)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 28)) as pilot:
        await pilot.pause()
        with _resolving():
            app._poll_subagents()
            await pilot.pause()
            assert _band_cost(app) == "$1.00"

            # The same child, further along. Its entry is REPLACED, not added
            # to — the figure is cumulative on the job, not a delta.
            kid.usage = Usage(input_tokens=3_000_000)
            app._poll_subagents()
            await pilot.pause()
    assert app._subagent_costs == {"kid": pytest.approx(3.0)}
    assert _band_cost(app) == "$3.00"


@pytest.mark.asyncio
async def test_total_survives_a_settled_child_leaving_the_ledger() -> None:
    """Spend never goes DOWN.

    ``AsyncJobManager`` sweeps settled jobs out after a retention window. A
    total computed live from the ledger would drop when a finished child is
    evicted, which reads as money being refunded.
    """
    session = _Session("anthropic/opus")
    session.jobs = _fake_jobs(
        _TaskJob("kid", usage=Usage(input_tokens=1_000_000), status="completed")
    )
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 28)) as pilot:
        await pilot.pause()
        with _resolving():
            app._poll_subagents()
            await pilot.pause()
            assert app._spend_total() == pytest.approx(10.0)

            session.jobs = _fake_jobs()  # retention sweep
            app._poll_subagents()
            await pilot.pause()
    assert app._spend_total() == pytest.approx(10.0)


@pytest.mark.asyncio
async def test_a_turn_that_only_delegated_still_reports_a_total() -> None:
    """A parent whose entire turn was one `task` call reports no usage of its own.

    ``$—`` beside a working subagent says the session is free when it is not.
    """
    session = _Session("anthropic/opus")
    session.jobs = _fake_jobs(_TaskJob("kid", usage=Usage(input_tokens=1_000_000)))
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 28)) as pilot:
        await pilot.pause()
        with _resolving():
            app.post_message(TurnEnded(False, None, context_tokens=0, usage=None))
            await pilot.pause()
    assert app._total_cost == 0.0
    assert _band_cost(app) == "$10.00"


@pytest.mark.asyncio
async def test_unpriced_model_with_no_children_still_reports_unavailable() -> None:
    """The D20 behaviour is intact: tokens billed, price unknown, say so."""
    session = _Session("anthropic/mystery")
    session.jobs = _fake_jobs()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 28)) as pilot:
        await pilot.pause()
        with _resolving():
            app.post_message(
                TurnEnded(False, None, context_tokens=0, usage=Usage(input_tokens=1_000_000))
            )
            await pilot.pause()
    assert _band_cost(app) == "$—"
