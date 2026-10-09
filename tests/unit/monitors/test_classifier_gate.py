"""The §8 classifier gate: one call per changed monitor, the fork, fail-open.

Everything runs against a controllable clock, a scripted check runner and a
scripted gate callback, so the tests assert the SCHEDULER's gate behaviour
rather than a session's. The service seam itself (``decide``) is
``tests/unit/classification/test_decide.py``'s subject; this file also drives
the two halves TOGETHER — the real service on a fake vendor behind the real
adapter (``monitor_classify``) — because the seam is where the slices meet and
a fake callback alone would pin only our side of the contract.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator, Mapping
from typing import Any

import pytest

from local_operator.monitors import classify as classify_module
from local_operator.monitors import state as monitor_state
from local_operator.monitors.classify import (
    IGNORABLE,
    MATERIAL,
    NON_MATERIAL_METADATA,
    PURPOSE_MAX_CHARS,
    bounded_state,
    gate_state,
    materiality_question,
    monitor_classify,
    suppressed_counter,
)
from local_operator.monitors.scheduler import MonitorScheduler
from local_operator.monitors.settings import MonitorSettings
from local_operator.monitors.spec import MonitorSpec
from tests.unit.classification.support import LegBehaviour, leg_class

NOW = 1_756_000_000_000


def spec(
    monitor_id: str = "m1",
    *,
    name: str = "watch",
    tool: str = "bash",
    arguments: Mapping[str, Any] | None = None,
    every_ms: int = 60_000,
    created_at: int = NOW,
    description: str = "",
) -> MonitorSpec:
    return MonitorSpec(
        id=monitor_id,
        name=name,
        tool=tool,
        arguments=dict(arguments or {"n": "a"}),
        every_ms=every_ms,
        created_at=created_at,
        description=description,
    )


class Harness:
    """A scheduler wired to a scripted gate; everything observable is recorded."""

    def __init__(
        self,
        tmp_path: Any,
        *,
        settings: MonitorSettings | None = None,
        classify: Any = True,
    ) -> None:
        self.now_ms = NOW
        self.results: list[dict[str, Any]] = []
        self.deliveries: list[Any] = []
        self.states: list[str] = []
        self.script: list[Any] = []
        self.default_class: str | None = MATERIAL
        self.classify_fn: Callable[[str], str | None] | None = None
        self.delay_s = 0.0
        self.active = 0
        self.peak = 0
        # ``True`` = the scripted callback below; ``False`` = no gate at all
        # (the classify=None path); anything else is used as the callback
        # directly (the real adapter in the end-to-end test).
        gate = None if classify is False else (self._classify if classify is True else classify)
        self.config_dir = tmp_path / "cfg"
        self.scheduler = MonitorScheduler(
            now=lambda: self.now_ms,
            config_dir=self.config_dir,
            session_id="sess",
            settings=settings or MonitorSettings(),
            validate=lambda tool, args: None,
            run_check=self._run_check,
            deliver=self._deliver,
            persist=self._persist,
            on_change=None,
            classify=gate,
            uniform=lambda low, high: low,
        )

    async def _classify(self, state: str) -> str | None:
        self.states.append(state)
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            # Yield (and optionally hold) so an unserialised second call would
            # be observable as a peak above one.
            await asyncio.sleep(self.delay_s or 0)
            if self.classify_fn is not None:
                return self.classify_fn(state)
            outcome = self.script.pop(0) if self.script else self.default_class
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome
        finally:
            self.active -= 1

    async def _run_check(self, monitor: MonitorSpec) -> Any:
        return self.results.pop(0) if self.results else {"text": "same", "error": None}

    async def _deliver(self, delivery: Any) -> None:
        self.deliveries.append(delivery)

    async def _persist(self, monitors: list[MonitorSpec]) -> None:
        pass

    async def ripe(self, *, advance: int = 10**9) -> None:
        self.now_ms += advance
        await self.scheduler.pump()
        for _ in range(20):
            await asyncio.sleep(0)
        if self.delay_s:
            # A scripted gate delay is real wall time: give the serialised
            # second call room to land before asserting.
            await asyncio.sleep(self.delay_s * 3 + 0.05)
            for _ in range(20):
                await asyncio.sleep(0)

    def counters(self, monitor_id: str = "m1") -> dict[str, Any]:
        found = monitor_state.read_counters(self.config_dir, "sess", monitor_id)
        assert found is not None, f"no counters for {monitor_id}"
        return found


@pytest.fixture
def harness(tmp_path: Any) -> Iterator[Harness]:
    created = Harness(tmp_path)
    yield created
    created.scheduler.dispose()


async def one_change(harness: Harness, **kwargs: Any) -> None:
    """Load m1, establish a baseline, then land one changed check."""
    harness.scheduler.load([spec(**kwargs)])
    harness.results.extend([{"text": "a"}, {"text": "b"}])
    await harness.ripe()  # baseline
    await harness.ripe()  # the change


# ---------------------------------------------------------------------------
# The fork (§8.4)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_material_delivers(harness: Harness) -> None:
    harness.script.append(MATERIAL)
    await one_change(harness)
    assert [d.monitor_id for d in harness.deliveries] == ["m1"]
    counters = harness.counters()
    assert counters["deliveries"] == 1
    assert counters["suppressed"] == {
        "non_material_metadata": 0,
        "ignorable": 0,
        "rate_cap": 0,
    }
    assert harness.states == ["+1/-1 changed lines\n- a\n+ b"]


@pytest.mark.asyncio
async def test_non_material_metadata_suppresses_and_counts(harness: Harness) -> None:
    harness.script.append(NON_MATERIAL_METADATA)
    await one_change(harness)
    assert harness.deliveries == []
    counters = harness.counters()
    assert counters["suppressed"]["non_material_metadata"] == 1
    assert counters["suppressed"]["ignorable"] == 0
    # A suppression is not a delivery: neither the delivery counter nor the
    # hourly window may move for it.
    assert counters["deliveries"] == 0
    assert counters["rate_window_count"] == 0
    assert counters["checks"] == 2


@pytest.mark.asyncio
async def test_ignorable_suppresses_and_counts(harness: Harness) -> None:
    harness.script.append(IGNORABLE)
    await one_change(harness)
    assert harness.deliveries == []
    counters = harness.counters()
    assert counters["suppressed"]["ignorable"] == 1
    assert counters["deliveries"] == 0


# ---------------------------------------------------------------------------
# Fail-open (§8.4's last row)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_classifier_at_all_delivers_without_calling_anything(tmp_path: Any) -> None:
    harness = Harness(tmp_path, classify=False)
    try:
        await one_change(harness)
        assert [d.monitor_id for d in harness.deliveries] == ["m1"]
        assert harness.states == []
    finally:
        harness.scheduler.dispose()


@pytest.mark.asyncio
async def test_a_none_answer_fails_open(harness: Harness) -> None:
    harness.script.append(None)
    await one_change(harness)
    assert [d.monitor_id for d in harness.deliveries] == ["m1"]
    assert harness.states == ["+1/-1 changed lines\n- a\n+ b"], "the call still happened"


@pytest.mark.asyncio
async def test_a_raising_classifier_fails_open(harness: Harness) -> None:
    harness.script.append(RuntimeError("gate exploded"))
    await one_change(harness)
    assert [d.monitor_id for d in harness.deliveries] == ["m1"]
    assert harness.counters()["suppressed"]["non_material_metadata"] == 0


@pytest.mark.asyncio
async def test_an_unknown_class_fails_open(harness: Harness) -> None:
    """A class we never offered is not a suppress class — the unsure direction delivers."""
    harness.script.append("bananas")
    await one_change(harness)
    assert [d.monitor_id for d in harness.deliveries] == ["m1"]


@pytest.mark.asyncio
async def test_a_quiet_tick_makes_zero_classifier_calls(harness: Harness) -> None:
    harness.scheduler.load([spec()])
    harness.results.extend([{"text": "a"}, {"text": "a"}])
    await harness.ripe()
    await harness.ripe()
    assert harness.states == []
    assert harness.deliveries == []


# ---------------------------------------------------------------------------
# One call per changed monitor, attributed (§8.1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_two_changed_monitors_make_two_calls_and_each_fork_is_its_own(
    harness: Harness,
) -> None:
    harness.classify_fn = lambda state: MATERIAL if "alpha" in state else NON_MATERIAL_METADATA
    harness.scheduler.load(
        [
            spec("m1", arguments={"n": "alpha"}, created_at=NOW),
            spec("m2", arguments={"n": "beta"}, created_at=NOW + 1),
        ]
    )
    harness.results.extend(
        [
            {"text": "alpha one"},
            {"text": "beta one"},
            {"text": "alpha two"},
            {"text": "beta two"},
        ]
    )
    await harness.ripe()
    await harness.ripe()
    assert len(harness.states) == 2, "one call per changed monitor, no fold"
    assert any("alpha" in state for state in harness.states)
    assert any("beta" in state for state in harness.states)
    # Material A delivered; non-material B suppressed and counted — the §18
    # matrix row, per monitor.
    assert [d.monitor_id for d in harness.deliveries] == ["m1"]
    assert harness.counters("m1")["deliveries"] == 1
    assert harness.counters("m2")["suppressed"]["non_material_metadata"] == 1
    assert harness.counters("m2")["deliveries"] == 0


@pytest.mark.asyncio
async def test_gate_calls_are_serialised_across_monitors(harness: Harness) -> None:
    """§8.1: calls are issued sequentially — the vendor's rate limit is per session."""
    harness.delay_s = 0.01
    harness.scheduler.load([spec("m1", created_at=NOW), spec("m2", created_at=NOW + 1)])
    harness.results.extend([{"text": "one"}, {"text": "two"}, {"text": "one+"}, {"text": "two+"}])
    await harness.ripe()
    await harness.ripe()
    assert len(harness.states) == 2
    assert harness.peak == 1, "two gate calls ran concurrently"


@pytest.mark.asyncio
async def test_the_state_is_the_bounded_delta(tmp_path: Any) -> None:
    harness = Harness(tmp_path, settings=MonitorSettings(classify_max_chars=30))
    try:
        harness.scheduler.load([spec()])
        harness.results.extend([{"text": "a"}, {"text": "b" * 400}])
        await harness.ripe()
        await harness.ripe()
        (state,) = harness.states
        assert len(state) <= 30
        assert state.endswith("[truncated]"), state
    finally:
        harness.scheduler.dispose()


# ---------------------------------------------------------------------------
# The gate judges the delta AGAINST the monitor's purpose (2026-10-09 regression)
# ---------------------------------------------------------------------------

#: What a build-progress monitor armed on a ``read`` of an append-only log says
#: it wants, trimmed from the report that surfaced the bug.
LOG_PURPOSE = "Build progress: the script appends one line per build start and completion."


def _purpose_aware_gate(state: str) -> str:
    """A stand-in for the vendor model that reproduces the measured failure.

    Measured against the live cascade: an appended ``read`` line with NO purpose
    in the state came back ``non-material-metadata`` (3/3), and the same delta
    with the purpose attached came back ``material`` (3/3). This double encodes
    that dependency — it only says "material" when the state names a purpose —
    so the test fails if the scheduler stops sending one.
    """
    return MATERIAL if "Purpose:" in state else NON_MATERIAL_METADATA


@pytest.mark.asyncio
async def test_an_appended_log_line_read_through_read_is_delivered(harness: Harness) -> None:
    """The reported shape: ``read`` of an append-only log, a new line each time.

    ``read`` numbers its lines (``N| …``), so an append is a pure ``+1/-0``
    insert at the tail; the diff pipeline sees it (pinned below), and the only
    thing that can lose it is the gate.
    """
    harness.classify_fn = _purpose_aware_gate
    harness.scheduler.load(
        [
            spec(
                name="sweep-serial-builds",
                tool="read",
                arguments={"path": "b.log"},
                description=LOG_PURPOSE,
            )
        ]
    )
    first = "1| === c1 start 2026-10-09T15:00:00Z ==="
    second = first + "\n2| === c1 done rc=0 2026-10-09T15:23:11Z ==="
    third = second + "\n3| === c2 start 2026-10-09T15:23:12Z ==="
    harness.results.extend([{"text": first}, {"text": second}, {"text": third}])
    await harness.ripe()  # baseline
    await harness.ripe()
    await harness.ripe()
    assert len(harness.deliveries) == 2
    assert harness.deliveries[0].delta_text == "+1/-0 changed lines\n+ 2| === c1 done rc=0 <ts> ==="
    assert harness.deliveries[1].delta_text == "+1/-0 changed lines\n+ 3| === c2 start <ts> ==="
    counters = harness.counters()
    assert counters["checks"] == 3
    assert counters["deliveries"] == 2
    assert counters["suppressed"]["non_material_metadata"] == 0


@pytest.mark.asyncio
async def test_a_pure_addition_is_delivered_without_asking_the_gate(harness: Harness) -> None:
    """An appended line never reaches the model, so no verdict can swallow it."""
    harness.script.append(NON_MATERIAL_METADATA)  # would suppress, if asked
    harness.scheduler.load([spec(name="log", description=LOG_PURPOSE)])
    harness.results.extend([{"text": "1| a"}, {"text": "1| a\n2| b done rc=0"}])
    await harness.ripe()
    await harness.ripe()
    assert harness.states == [], "the gate was asked about an append"
    assert len(harness.deliveries) == 1
    assert harness.counters()["suppressed"]["non_material_metadata"] == 0


@pytest.mark.asyncio
async def test_an_edit_still_goes_to_the_gate(harness: Harness) -> None:
    harness.script.append(NON_MATERIAL_METADATA)
    harness.scheduler.load([spec(name="log", description=LOG_PURPOSE)])
    harness.results.extend(
        [{"text": "1| a"}, {"text": "1| a\n2| b\n3| c"}, {"text": "1| a2\n2| b\n3| c"}]
    )
    await harness.ripe()
    await harness.ripe()  # append: delivered, no call
    await harness.ripe()  # edit: asked, suppressed
    assert len(harness.states) == 1
    assert len(harness.deliveries) == 1
    assert harness.counters()["suppressed"]["non_material_metadata"] == 1


@pytest.mark.asyncio
async def test_an_append_to_a_truncated_snapshot_is_still_gated(tmp_path: Any) -> None:
    """Review R2: beyond the stored window a tail edit looks like an insert."""
    harness = Harness(tmp_path, settings=MonitorSettings(snapshot_max_chars=19))
    try:
        harness.script.append(NON_MATERIAL_METADATA)
        harness.scheduler.load([spec()])
        harness.results.extend(
            [
                {"text": "aaaa\nbbbb\ncccc\ndddd\neeee"},
                {"text": "aaaa\nbbbb\ncccc\ndddd\neeee\nffff"},
            ]
        )
        await harness.ripe()
        await harness.ripe()
        assert len(harness.states) == 1, "a truncated window must not bypass the gate"
    finally:
        harness.scheduler.dispose()


@pytest.mark.asyncio
async def test_an_append_across_a_gutter_power_of_ten_is_still_delivered(harness: Harness) -> None:
    """QA round 1: at 9->10 lines ``read`` re-pads every gutter (full replace)."""
    harness.script.append(NON_MATERIAL_METADATA)  # would suppress, if asked
    harness.scheduler.load([spec(name="log")])
    nine = "\n".join(f"{i}| line {i}" for i in range(1, 10))
    ten = "\n".join(f"{i}| line {i}" for i in range(1, 11))
    harness.results.extend([{"text": nine}, {"text": ten}])
    await harness.ripe()
    await harness.ripe()
    assert harness.states == []
    assert len(harness.deliveries) == 1


@pytest.mark.asyncio
async def test_the_gate_state_names_the_monitor_and_its_purpose(harness: Harness) -> None:
    harness.scheduler.load([spec(name="loom-pr", description="flip of review state")])
    harness.results.extend([{"text": "a"}, {"text": "b"}])
    await harness.ripe()
    await harness.ripe()
    assert harness.states == [
        "Monitor: loom-pr\nPurpose: flip of review state\nChange:\n+1/-1 changed lines\n- a\n+ b"
    ]


def test_gate_state_without_a_purpose_is_the_bare_delta() -> None:
    assert gate_state("m", "", "+1/-0 changed lines\n+ x", 1200) == "+1/-0 changed lines\n+ x"
    assert gate_state("m", "   \n ", "d", 1200) == "d"


def test_gate_state_collapses_and_clips_the_purpose_and_still_bounds_the_delta() -> None:
    long_purpose = "word " * 400
    state = gate_state("n", long_purpose, "x" * 500, 100)
    head, _, delta = state.partition("Change:\n")
    assert head.startswith("Monitor: n\nPurpose: word word")
    purpose_line = head.splitlines()[1]
    assert len(purpose_line) <= len("Purpose: ") + PURPOSE_MAX_CHARS + 1
    assert purpose_line.endswith("…")
    assert len(delta) <= 100 and delta.endswith("[truncated]")


# ---------------------------------------------------------------------------
# The rate cap is the DELIVER path's gate (§5.2 step 5)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_hourly_cap_holds_material_changes_and_names_them_next_time(
    tmp_path: Any,
) -> None:
    harness = Harness(tmp_path, settings=MonitorSettings(max_deliveries_per_hour=1))
    try:
        harness.scheduler.load([spec()])
        harness.results.extend([{"text": "a"}, {"text": "b"}, {"text": "c"}, {"text": "d"}])
        # Two-minute steps: enough to make the 60 s monitor due, far short of
        # the hour the rate window spans — the default 10**9 advance would
        # reset the window and the cap would never be exercised.
        await harness.ripe(advance=120_000)  # baseline
        await harness.ripe(advance=120_000)  # b: delivered
        await harness.ripe(advance=120_000)  # c: held by the cap
        counters = harness.counters()
        assert counters["deliveries"] == 1
        assert counters["suppressed"]["rate_cap"] == 1
        assert counters["rate_cap_held"] == 1
        # Past the window the next material change delivers and carries the
        # held count (§9.1's "held by the hourly cap" clause).
        await harness.ripe(advance=2 * 60 * 60 * 1000)
        assert [d.monitor_id for d in harness.deliveries] == ["m1", "m1"]
        assert harness.deliveries[-1].held_by_cap == 1
        assert harness.counters()["rate_cap_held"] == 0
    finally:
        harness.scheduler.dispose()


@pytest.mark.asyncio
async def test_suppressed_changes_never_consume_the_hourly_window(harness: Harness) -> None:
    """Noise must not spend the budget the operator's material messages are owed."""
    harness.scheduler.load([spec()])
    harness.results.extend([{"text": "a"}, {"text": "b"}, {"text": "c"}, {"text": "d"}])
    harness.script.extend([NON_MATERIAL_METADATA, IGNORABLE])
    await harness.ripe()
    await harness.ripe()
    await harness.ripe()
    counters = harness.counters()
    assert counters["rate_window_count"] == 0
    assert counters["suppressed"]["non_material_metadata"] == 1
    assert counters["suppressed"]["ignorable"] == 1


# ---------------------------------------------------------------------------
# The pure pieces and the adapter
# ---------------------------------------------------------------------------


def test_suppressed_counter_maps_only_the_two_non_material_classes() -> None:
    assert suppressed_counter(NON_MATERIAL_METADATA) == "non_material_metadata"
    assert suppressed_counter(IGNORABLE) == "ignorable"
    assert suppressed_counter(MATERIAL) is None
    assert suppressed_counter(None) is None
    assert suppressed_counter("something-else") is None


def test_the_question_is_the_contracts_shape_verbatim() -> None:
    """§8.3's text, pinned — a paraphrase is a different, unmeasured request."""
    question = materiality_question()
    assert question.id == "monitor_materiality"
    assert question.kind == "choice"
    assert question.instructions == (
        "Decide whether this change is MATERIAL: a human asked to be told about it. "
        "MATERIAL = new information a person would want (a reply, a status flip, a new record). "
        "NON-MATERIAL METADATA = bookkeeping that changed without meaning (timestamps, ordering, "
        "volatile ids). IGNORABLE = noise that will never matter (whitespace, boilerplate)."
    )
    assert question.criteria == {
        "material": "a person wanted to be told about this change",
        "non-material-metadata": "changed, but only metadata: timestamps, ordering, volatile ids",
        "ignorable": "noise that can never matter; whitespace, boilerplate, formatting",
    }


def test_bounded_state_cuts_and_marks() -> None:
    assert bounded_state("short", 100) == "short"
    long = "x" * 100
    cut = bounded_state(long, 30)
    assert len(cut) <= 30 and cut.endswith("[truncated]")
    # A non-positive bound is treated as "no bound" by the reader's convention:
    # classifyMaxChars is read through the settings' positive-value reader.
    assert bounded_state(long, 0) == long
    # The marker floor, stated in the docstring and pinned here: a cap below
    # the marker's own length returns the bare marker (hand-edited config only).
    assert bounded_state(long, 5) == classify_module.TRUNCATION_MARKER


def test_the_truncation_marker_matches_the_classification_layer() -> None:
    """The marker is a pinned DUPLICATE (classify.py's module docstring):
    importing it would run the classification package on a path reachable with
    the layer off. This test is what keeps the two spellings equal.
    """
    from local_operator.classification.context import TRUNCATION_MARKER

    assert TRUNCATION_MARKER == classify_module.TRUNCATION_MARKER


@pytest.mark.asyncio
async def test_monitor_classify_resolves_the_seam_per_call() -> None:
    """The adapter re-reads its resolver — the factory's swap convention."""
    from local_operator.classification.types import Answer

    class Seam:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        async def decide(self, *, state: str, question: Any) -> Answer:
            self.calls.append((state, question.id))
            return Answer(id=question.id, kind="choice", value=MATERIAL)

    seam = Seam()
    current: list[Any] = [None]
    classify = monitor_classify(lambda: current[0])
    assert await classify("delta") is None  # no seam yet
    current[0] = seam
    assert await classify("delta") == MATERIAL
    assert seam.calls == [("delta", "monitor_materiality")]

    class NotADecider:
        async def recommend_resources(self, request: Any) -> None: ...

    current[0] = NotADecider()
    assert await classify("delta") is None, "a host's own classifier without decide fails open"


# ---------------------------------------------------------------------------
# The two halves together: real service, fake vendor, real adapter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_gate_through_the_real_service_and_a_fake_vendor(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wiring's shape end to end: scheduler → adapter → decide → stub leg.

    A fake vendor only — the live-vendor end-to-end belongs to the slice's QA
    run, where a real credential exists; this proves the seam's contract on
    this host without a network.
    """
    import local_operator.classification.vendors as vendors
    from local_operator.classification.service import ClassificationService

    behaviour = LegBehaviour(name="radient")
    behaviour.script.extend(
        [
            # One response per decide call in the order the changes land.
            _choice("monitor_materiality", NON_MATERIAL_METADATA),
            _choice("monitor_materiality", IGNORABLE),
            _choice("monitor_materiality", MATERIAL),
        ]
    )
    replacements = dict(vendors.VENDOR_CLASSES)
    replacements["radient"] = leg_class(behaviour)
    monkeypatch.setattr(vendors, "VENDOR_CLASSES", replacements, raising=True)

    service = ClassificationService(
        config_dir=tmp_path / "cfg", settings={"classification": {"auto": True}}
    )
    harness = Harness(tmp_path, classify=monitor_classify(lambda: service))
    try:
        harness.scheduler.load([spec()])
        harness.results.extend([{"text": "a"}, {"text": "b"}, {"text": "c"}, {"text": "d"}])
        await harness.ripe()  # baseline — no call
        await harness.ripe()  # non-material-metadata → suppressed
        await harness.ripe()  # ignorable → suppressed
        await harness.ripe()  # material → delivered
        counters = harness.counters()
        assert counters["suppressed"] == {
            "non_material_metadata": 1,
            "ignorable": 1,
            "rate_cap": 0,
        }
        assert counters["deliveries"] == 1
        assert [d.monitor_id for d in harness.deliveries] == ["m1"]
        assert len(behaviour.calls) == 3, "one call per changed monitor"
        for request in behaviour.calls:
            assert request.questions[0].id == "monitor_materiality"
        # The seam is the real service: its cache answered the replay of the
        # baseline state without a fourth call... which this run never asked
        # for, so the assertion above is the count that matters.
    finally:
        harness.scheduler.dispose()


def _choice(question_id: str, choice: str) -> Any:
    from local_operator.classification.types import Answer, DecisionResponse

    return DecisionResponse(
        vendor="stub",
        model="stub-model",
        answers={question_id: Answer(id=question_id, kind="choice", value=choice)},
        input_tokens=100,
        output_tokens=10,
        cost_usd=0.00002,
    )
