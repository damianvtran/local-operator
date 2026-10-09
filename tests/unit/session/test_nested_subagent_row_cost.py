"""A running subagent's row prices its WHOLE subtree, live children included.

The incident (session 3463fc25dade): six running subagents showed $0.02-$0.08
each while the footer read ~$3.09 and later ~$5.5. The difference was five
depth-2 Sonnet scouts under one depth-1 row whose own figure stayed at $0.02.
The footer comes from the job ledger (``accounting_components``), which reads a
running task's LIVE child manager; the rows priced ``usage`` plus
``descendant_usage`` only, and that list is filled when the child manager is
DETACHED at completion. These tests drive the real ``AsyncJobManager`` pair
(parent ledger, attached child ledger) and every surface that prices a row, and
pin the invariant the incident broke: the root rows add up to the ledger total.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest

from local_operator.harness.jobs import AsyncJob, AsyncJobManager
from local_operator.harness.types import Usage
from local_operator.model.registry import ModelInfo
from local_operator.session.frontend_state import (
    CostKnowledge,
    JobState,
    _carry_child_cost,
    _ledger_cost,
    _released_row,
)
from local_operator.tui.app import OperatorApp
from local_operator.tui.costs import cost_summary, job_cost, job_subtree_cost
from local_operator.tui.widgets.subagent_panel import job_stats
from tests.unit.tui.test_band_panels import FakeSession, _async_factory

#: Per-million-token rates. Deliberately different per model so a component
#: priced at the PARENT's rate (the failure this guards) cannot hide.
_RATES = {
    # Anthropic's convention: cache buckets are disjoint from ``input_tokens``.
    "claude-sonnet-5-5": ModelInfo(
        id="claude-sonnet-5-5",
        name="sonnet",
        description="",
        input_price=2.0,
        output_price=10.0,
        cache_reads_price=0.20,
        cache_writes_price=2.50,
    ),
    "auto": ModelInfo(id="auto", name="auto", description="", input_price=1.0, output_price=5.0),
    "haiku": ModelInfo(
        id="haiku", name="haiku", description="", input_price=1.0, output_price=10.0
    ),
}


def _priced():
    """Both resolvers patched: ``turn_cost`` prices through the paint-safe one."""
    return patch.multiple(
        "local_operator.model.configure",
        resolve_model_info=lambda provider, model_id: _RATES.get(
            model_id, ModelInfo(id=model_id, name=model_id, description="")
        ),
        resolve_model_info_paint=lambda provider, model_id: (
            _RATES.get(model_id, ModelInfo(id=model_id, name=model_id, description="")),
            True,
        ),
    )


def _row(
    job_id: str,
    *,
    model: str,
    usage: Usage | None = None,
    status: str = "running",
    descendants: list[Usage] | None = None,
) -> AsyncJob:
    return AsyncJob(
        id=job_id,
        type="task",
        status=status,  # type: ignore[arg-type]
        start_time=1.0,
        label=job_id,
        model_label=model,
        usage=usage,
        descendant_usage=descendants or [],
    )


def _scouts(count: int = 5) -> AsyncJobManager:
    """The incident's five depth-2 Sonnet scouts.

    Its 6.97M cache-read, 1.14M cache-write and 126k output tokens, split five ways.
    """
    child = AsyncJobManager()
    for index in range(count):
        child._jobs[f"scout-{index}"] = _row(
            f"scout-{index}",
            model="anthropic/claude-sonnet-5-5",
            usage=Usage(
                input_tokens=40,
                output_tokens=25_000,
                cache_read_tokens=1_393_792,
                cache_write_tokens=227_754,
            ),
        )
    return child


#: One scout, priced by hand from ``_RATES``' Sonnet row (2 / 10 / 0.20 / 2.50 per MTok).
_SCOUT = (40 * 2.0 + 25_000 * 10.0 + 1_393_792 * 0.20 + 227_754 * 2.50) / 1e6


def test_the_hand_priced_scout_matches_the_shape_of_the_incident() -> None:
    # ~$5.5 for five scouts is what the incident's roster measured at table
    # prices; if the fixture drifts from that shape the other tests prove less.
    assert 5 * _SCOUT == pytest.approx(5.5, abs=0.2)


def test_a_running_parent_row_includes_its_live_children() -> None:
    """The reproduction: this is $0.02 on origin/main and ~$5.5 here."""
    scouts = _scouts()
    parent = _row(
        "grc",
        model="radient/auto",
        usage=Usage(input_tokens=20_000, provider="radient", model_id="auto", usd_cost=0.02),
    )
    manager = AsyncJobManager()
    manager._jobs[parent.id] = parent
    manager.attach_child_manager(parent.id, scouts)

    with _priced():
        own = job_cost(parent, default_model_label="radient/auto")
        cost, lower_bound = job_subtree_cost(parent, default_model_label="radient/auto")
        stats = job_stats(parent, default_model_label="radient/auto")
        row = JobState.from_job(parent)

    assert own == pytest.approx(0.02)  # what the row used to say, in full
    assert cost == pytest.approx(0.02 + 5 * _SCOUT)
    assert lower_bound is False
    # Every surface that shows the row agrees with the same number.
    assert stats.cost == pytest.approx(cost)
    assert not stats.cost_partial
    assert row.direct_cost == pytest.approx(cost)
    assert row.direct_cost_knowledge == CostKnowledge.EXACT


def test_children_are_priced_at_their_own_identity_not_the_parents() -> None:
    parent = _row("grc", model="radient/auto", usage=Usage(input_tokens=1_000_000))
    manager = AsyncJobManager()
    manager._jobs[parent.id] = parent
    child = AsyncJobManager()
    child._jobs["kid"] = _row(
        "kid", model="anthropic/haiku", usage=Usage(input_tokens=1_000_000, output_tokens=0)
    )
    manager.attach_child_manager(parent.id, child)
    with _priced():
        cost, _ = job_subtree_cost(parent, default_model_label="radient/auto")
    # parent 1M @ $1 (radient/auto) + child 1M @ $1 (haiku): equal here, so make
    # the rates differ through output instead.
    child._jobs["kid"].usage = Usage(output_tokens=1_000_000)
    child.note_usage_changed()
    with _priced():
        cost, _ = job_subtree_cost(parent, default_model_label="radient/auto")
    # 1M input at auto's $1 + 1M output at haiku's $10. Pricing the child at the
    # parent's output rate would give $6.
    assert cost == pytest.approx(11.0)


def test_a_completed_child_is_counted_once_across_the_detach() -> None:
    """Live manager before detach, ``descendant_usage`` after: same money."""
    parent = _row("grc", model="radient/auto", usage=Usage(input_tokens=1_000_000))
    manager = AsyncJobManager()
    manager._jobs[parent.id] = parent
    scouts = _scouts(2)
    manager.attach_child_manager(parent.id, scouts)

    with _priced():
        live, _ = job_subtree_cost(parent, default_model_label="radient/auto")
        snapshot = scouts.accounting_components()
        manager.detach_child_manager(parent.id, snapshot)
        assert parent.child_jobs is None
        assert parent.descendant_usage
        detached, _ = job_subtree_cost(parent, default_model_label="radient/auto")

    assert live == pytest.approx(1.0 + 2 * _SCOUT)
    assert detached == pytest.approx(live)


def test_a_settled_child_in_a_live_manager_is_not_added_twice() -> None:
    """A child that finished inside a still-attached manager lives in its
    settled accumulator; the row must read it there and not also off the row."""
    child = AsyncJobManager()
    done = _row(
        "done",
        model="anthropic/claude-sonnet-5-5",
        usage=Usage(output_tokens=1_000_000),
        status="running",
    )
    child._jobs[done.id] = done
    parent = _row("grc", model="radient/auto")
    manager = AsyncJobManager()
    manager._jobs[parent.id] = parent
    manager.attach_child_manager(parent.id, child)
    done.status = "completed"
    child._settle(done)

    with _priced():
        cost, _ = job_subtree_cost(parent, default_model_label="radient/auto")
        ledger = cost_summary(manager.accounting_components())[0]
    assert cost == pytest.approx(10.0)
    assert ledger == pytest.approx(10.0)


def test_nested_two_levels_deep() -> None:
    leaf = AsyncJobManager()
    leaf._jobs["leaf"] = _row("leaf", model="anthropic/haiku", usage=Usage(output_tokens=1_000_000))
    mid = AsyncJobManager()
    mid._jobs["mid"] = _row("mid", model="anthropic/haiku", usage=Usage(input_tokens=1_000_000))
    mid.attach_child_manager("mid", leaf)
    top = AsyncJobManager()
    root = _row("root", model="radient/auto", usage=Usage(input_tokens=1_000_000))
    top._jobs["root"] = root
    top.attach_child_manager("root", mid)

    with _priced():
        cost, lower = job_subtree_cost(root, default_model_label="radient/auto")
        ledger = cost_summary(top.accounting_components())[0]
    # 1M auto input ($1) + 1M haiku input ($1) + 1M haiku output ($10)
    assert cost == pytest.approx(12.0)
    assert ledger == pytest.approx(cost)
    assert lower is False


def test_receipt_beats_estimate_per_component() -> None:
    parent = _row(
        "p",
        model="radient/auto",
        usage=Usage(input_tokens=1_000_000, provider="radient", model_id="auto"),
    )
    child = AsyncJobManager()
    # A provider receipt for a model whose table price would say $10.
    child._jobs["a"] = _row(
        "a",
        model="anthropic/claude-sonnet-5-5",
        usage=Usage(output_tokens=1_000_000, usd_cost=0.25),
    )
    # A sibling with no receipt, priced from the table.
    child._jobs["b"] = _row(
        "b", model="anthropic/claude-sonnet-5-5", usage=Usage(output_tokens=1_000_000)
    )
    manager = AsyncJobManager()
    manager._jobs["p"] = parent
    manager.attach_child_manager("p", child)
    with _priced():
        cost, lower = job_subtree_cost(parent, default_model_label="radient/auto")
    assert cost == pytest.approx(1.0 + 0.25 + 10.0)
    assert lower is False


def test_unpriced_children_leave_a_lower_bound_and_all_unpriced_stays_unknown() -> None:
    parent = _row("p", model="radient/auto", usage=Usage(input_tokens=1_000_000))
    child = AsyncJobManager()
    child._jobs["known"] = _row(
        "known", model="anthropic/haiku", usage=Usage(input_tokens=1_000_000)
    )
    child._jobs["mystery"] = _row(
        "mystery", model="nobody/never-heard-of-it", usage=Usage(input_tokens=1_000_000)
    )
    manager = AsyncJobManager()
    manager._jobs["p"] = parent
    manager.attach_child_manager("p", child)

    with _priced():
        cost, lower = job_subtree_cost(parent, default_model_label="radient/auto")
        row = JobState.from_job(parent)
        stats = job_stats(parent, default_model_label="radient/auto")
    # The two priced components survive the unpriced one, marked as a floor.
    assert cost == pytest.approx(2.0)
    assert lower is True
    assert row.direct_cost_knowledge == CostKnowledge.PARTIAL
    assert stats.cost == pytest.approx(2.0)
    assert stats.cost_partial is True

    # Nothing priceable anywhere: unknown, never a confident $0.
    nowhere = _row("q", model="nobody/never-heard-of-it", usage=Usage(input_tokens=5))
    with _priced():
        cost, lower = job_subtree_cost(nowhere, default_model_label="nobody/never-heard-of-it")
    assert cost is None


def test_a_parent_with_no_usage_of_its_own_is_billed_by_its_children() -> None:
    """The depth-1 row is 'Waiting for research scouts' and has reported nothing."""
    parent = _row("p", model="radient/auto", usage=None)
    manager = AsyncJobManager()
    manager._jobs["p"] = parent
    manager.attach_child_manager("p", _scouts(1))
    with _priced():
        stats = job_stats(parent, default_model_label="radient/auto")
    assert stats.billed is True
    assert stats.cost == pytest.approx(_SCOUT)
    # And a row that truly has nothing still reads as nothing.
    with _priced():
        assert job_subtree_cost(_row("e", model="radient/auto"))[0] is None


def test_the_released_row_prices_the_same_subtree() -> None:
    parent = _row(
        "p",
        model="radient/auto",
        usage=Usage(input_tokens=1_000_000),
        status="completed",
        descendants=[Usage(output_tokens=1_000_000, provider="anthropic", model_id="haiku")],
    )
    with _priced():
        released = _released_row(parent)
        full = JobState.from_job(parent)
    assert released.direct_cost == pytest.approx(11.0)
    assert released.direct_cost == full.direct_cost


def test_a_frozen_follower_row_is_not_repriced_in_the_viewer() -> None:
    """The runtime priced the subtree; a viewer reads that figure verbatim."""
    parent = _row("p", model="radient/auto", usage=Usage(input_tokens=1_000_000))
    manager = AsyncJobManager()
    manager._jobs["p"] = parent
    manager.attach_child_manager("p", _scouts(1))
    with _priced():
        wire = JobState.model_validate_json(JobState.from_job(parent).model_dump_json())
    costs: dict[str, float] = {}
    # No pricing patch: a viewer with an empty memo must still carry the figure.
    _carry_child_cost(costs, wire, default_model_label="openai/gpt")
    assert costs["p"] == pytest.approx(1.0 + _SCOUT)
    assert job_stats(wire).cost == pytest.approx(1.0 + _SCOUT)


def test_a_partly_unpriced_tick_never_lowers_a_figure_already_shown() -> None:
    parent = _row("p", model="radient/auto", usage=Usage(input_tokens=1_000_000))
    manager = AsyncJobManager()
    manager._jobs["p"] = parent
    child = AsyncJobManager()
    child._jobs["k"] = _row("k", model="anthropic/haiku", usage=Usage(input_tokens=1_000_000))
    manager.attach_child_manager("p", child)
    costs: dict[str, float] = {}
    with _priced():
        _carry_child_cost(costs, parent, default_model_label="radient/auto")
    assert costs["p"] == pytest.approx(2.0)

    # A new component on a model nothing can price this tick: the priced part is
    # unchanged, the row is a floor, and the stored figure must not dip.
    child._jobs["m"] = _row("m", model="nobody/new", usage=Usage(input_tokens=9))
    child.note_usage_changed()
    with _priced():
        _carry_child_cost(costs, parent, default_model_label="radient/auto")
    assert costs["p"] == pytest.approx(2.0)


def test_a_live_branch_that_cannot_be_read_is_a_lower_bound_not_a_crash() -> None:
    class _Exploding:
        def accounting_components(self) -> list[Usage]:
            raise RuntimeError("child ledger torn down mid-read")

    parent = SimpleNamespace(
        id="p",
        usage=Usage(input_tokens=1_000_000),
        model_label="radient/auto",
        descendant_usage=[],
        child_jobs=_Exploding(),
    )
    with _priced():
        cost, lower = job_subtree_cost(parent, default_model_label="radient/auto")
    assert cost == pytest.approx(1.0)
    assert lower is True


# -- the incident-shaped fixture: rows and footer must agree ------------------------


def _incident() -> tuple[AsyncJobManager, dict[str, AsyncJob]]:
    """Six root jobs; one running with five live nested Sonnet scouts."""
    manager = AsyncJobManager()
    roots: dict[str, AsyncJob] = {}
    for name, spent in {
        "tam-market-sizing": 0.08,
        "grc-incumbents": 0.02,
        "ai-native-challengers": 0.03,
        "fintech-risk-vendors": 0.05,
        "buyer-priority-budget": 0.04,
        "prismlayer-novelty": 0.06,
    }.items():
        row = _row(
            name,
            model="radient/auto",
            usage=Usage(input_tokens=1_000, provider="radient", model_id="auto", usd_cost=spent),
        )
        manager._jobs[name] = row
        roots[name] = row
    manager.attach_child_manager("grc-incumbents", _scouts())
    return manager, roots


def test_root_rows_add_up_to_the_footer_subagent_total() -> None:
    manager, roots = _incident()
    session = SimpleNamespace(jobs=manager)
    with _priced():
        ledger = _ledger_cost(session)["subagent_cost"]
        rows = {name: JobState.from_job(row).direct_cost or 0.0 for name, row in roots.items()}
        costs: dict[str, float] = {}
        for row in roots.values():
            _carry_child_cost(costs, row, default_model_label="radient/auto")
    assert rows["grc-incumbents"] == pytest.approx(0.02 + 5 * _SCOUT)
    assert sum(rows.values()) == pytest.approx(ledger)
    assert sum(costs.values()) == pytest.approx(ledger)
    # The other five rows are untouched by the fix.
    assert rows["tam-market-sizing"] == pytest.approx(0.08)


def test_the_tui_harvest_feeds_the_band_the_same_rollup() -> None:
    manager, roots = _incident()

    class _Session(FakeSession):
        @property
        def model_label(self) -> str:
            return "radient/auto"

        @property
        def effective_model_label(self) -> str:
            return "radient/auto"

    session = _Session()
    session.jobs = manager  # type: ignore[assignment]
    app = OperatorApp(_async_factory(session))
    app._session = session
    with _priced():
        app._harvest_subagent_costs()
        ledger = cost_summary(manager.accounting_components())[0]
    assert app._subagent_costs["grc-incumbents"] == pytest.approx(0.02 + 5 * _SCOUT)
    assert sum(app._subagent_costs.values()) == pytest.approx(ledger)


def test_the_row_moves_as_the_children_keep_spending() -> None:
    manager, roots = _incident()
    parent = roots["grc-incumbents"]
    child = parent.child_jobs
    assert isinstance(child, AsyncJobManager)
    with _priced():
        before = JobState.from_job(parent).direct_cost
        child._jobs["scout-0"].usage.output_tokens += 1_000_000  # type: ignore[union-attr]
        child.note_usage_changed()
        after = JobState.from_job(parent).direct_cost
    assert after == pytest.approx((before or 0.0) + 10.0)


def _unused(*_: Any) -> None:  # keeps ``Any`` imported for the annotations above
    return None


def test_a_cold_viewer_row_includes_settled_descendants_without_a_manager() -> None:
    """The daemonless sidecar overlay (``AttachedSession._durable_roster``) prices
    recorded money only; its row must still cover the settled descendants."""
    from local_operator.session.attached import AttachedSession
    from local_operator.session.frontend_state import FrontendSessionState
    from local_operator.session.session import _subagent_job_row

    parent = _row(
        "p",
        model="radient/auto",
        status="completed",
        usage=Usage(provider="radient", model_id="auto", usd_cost=0.02),
        descendants=[
            Usage(provider="anthropic", model_id="claude-sonnet-5-5", estimated_usd_cost=5.5),
        ],
    )
    durable = FrontendSessionState(session_id="s", epoch="e")
    rows = AttachedSession._durable_roster(
        cast(Any, SimpleNamespace()), durable, payload={"jobs": [_subagent_job_row(parent)]}
    )
    (cold,) = rows
    assert cold.direct_cost == pytest.approx(5.52)
    assert cold.direct_cost_knowledge == CostKnowledge.EXACT


def test_the_store_publishes_rows_and_child_costs_that_sum_to_the_ledger() -> None:
    """The real desktop/mobile path: ``FrontendStateStore.refresh_jobs`` on the live manager.

    ``state.jobs[*].direct_cost`` is what Run details renders, ``child_costs`` is the
    compatibility map the mobile glance and ``cumulative_cost`` read, and
    ``subagent_cost`` is the ledger. All three must tell one story.
    """
    from local_operator.session.frontend_state import (
        FrontendSessionState,
        FrontendStateStore,
    )

    manager, _ = _incident()
    session = SimpleNamespace(jobs=manager, model=None, session_id="s1", queued_steering=lambda: [])
    store = FrontendStateStore(FrontendSessionState(session_id="s1", epoch="e"))
    with _priced():
        store.refresh_jobs(session)
    state = store.state
    by_label = {job.label: job for job in state.jobs}
    assert by_label["grc-incumbents"].direct_cost == pytest.approx(0.02 + 5 * _SCOUT)
    assert by_label["grc-incumbents"].direct_cost_knowledge == CostKnowledge.EXACT
    assert sum(job.direct_cost or 0.0 for job in state.jobs) == pytest.approx(
        state.subagent_cost or 0.0
    )
    assert sum(state.child_costs.values()) == pytest.approx(state.subagent_cost or 0.0)
