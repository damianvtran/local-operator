"""A pinned child moved off its model must SAY SO — on the row and on the stream.

The pin exists so a delegated review cannot collapse onto the author's model
(``Session._launch_subagent``: "Independence that can silently collapse into
self-review is not independence"). Before this change a provider fallback did
exactly that collapse SILENTLY: the relay overwrote the job's ``model_label``
with the effective model, the pin label survived nowhere, and the band, the
dock and the parent's ``wait`` receipt all reported the substitute with no sign
anyone had asked for something else.

The writer under test is the relay's ``ModelChangeEvent`` branch
(``harness/subagent.py``), and each surface is pinned by its own test:

* the JOB ROW keeps the requested label forever and carries the fallback
  marker + reason while the substitution stands (and clears them on recovery);
* the PARENT STREAM gets one notice per fallback episode with all four facts —
  the two models in the product's vocabulary, joined as an atomic `A → B` pair;
* the ROSTER ROW (the sidecar projection) carries the pin label so a restored
  row still renders the badge after a restart;
* ``wait``'s ``_job_summary`` names both models;
* the band's model segment renders the badge — the full pair while the row can
  hold it, the shed `⚠ <effective>` form below that, and the same suffix on the
  irreducible rung — because the pair must never evict cwd or the context
  reading the base row kept (D1/U2);
* the DURABLE COMPLETION row states the pin — the surface the walk-away flow
  ends on, which no replay of the live notice reaches (U1);
* the mobile projection carries the badge AND its boolean, so the phone's
  roster row can paint the substitution without parsing prose.

The NEGATIVE ARM is as load-bearing as the positive one: a child with no pin
(``owns_model`` False/None — every reviewer/qa-tester/coder launched without an
effort tier) must never get the marker, the notice, or the badge. Those roles
are SUPPOSED to inherit the parent, and a fix that marked every child would
pass a test that only asserted "a marker appears".
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest
from textual.widgets import Static

from local_operator.harness.jobs import (
    AsyncJob,
    AsyncJobManager,
    is_model_fallback,
    model_fallback_badge,
)
from local_operator.harness.subagent import _make_relay
from local_operator.harness.types import ModelChangeEvent, NoticeEvent
from local_operator.tui.widgets.subagent_panel import job_stats

#: The incident's shape: a role-pinned review child (``effort:hi`` through
#: ``subagents.models.hi``) and the operator's own session model — which is
#: what their fallback chain reaches first.
PIN = "anthropic/claude-sonnet-5-5"
EFFECTIVE = "deepseek/deepseek-flash"


def _pinned_job() -> SimpleNamespace:
    """One duck-typed job row in the pinned state, exactly as registration + a
    route edge leave it: ``owns_model`` and ``requested_model_label`` stamped,
    ``model_label`` on the effective model."""
    return SimpleNamespace(
        id="designer1",
        type="task",
        label="round1-designer",
        agent_role="designer",
        trajectory=[],
        owns_model=True,
        requested_model_label=PIN,
        model_label=PIN,
        model_fallback=False,
        model_fallback_reason="",
    )


def _relay(job: Any, emitted: list[Any]) -> Any:
    async def _emit(event: Any) -> None:
        emitted.append(event)

    return _make_relay(
        str(job.id),
        str(job.label),
        job,
        AsyncJobManager(),
        _emit,
        lambda _progress: None,
        {"text": "", "error": None},
    )


def _fallback_event(reason: str = "provider failure") -> ModelChangeEvent:
    return ModelChangeEvent(
        provider="deepseek",
        model_id="deepseek-flash",
        reason=reason,
        is_fallback=True,
        context_window=160_000,
    )


def _recovery_event() -> ModelChangeEvent:
    return ModelChangeEvent(
        provider="anthropic",
        model_id="claude-sonnet-5-5",
        reason="primary model recovered",
        is_fallback=False,
    )


# ---------------------------------------------------------------------------
# the job row: marker, reason, and the pin that is never overwritten
# ---------------------------------------------------------------------------


def test_a_fallback_edge_marks_the_pinned_job_and_keeps_the_pin() -> None:
    """(a) The effective label moves; the requested label does NOT.

    Without the split, ``model_label`` following the fallback ERASES the only
    record of what was asked for — the state this whole surface exists to
    render.
    """
    job = _pinned_job()
    emitted: list[Any] = []
    asyncio.run(_relay(job, emitted)(_fallback_event()))

    assert job.model_fallback is True
    assert job.model_fallback_reason == "provider failure"
    assert job.requested_model_label == PIN
    assert job.model_label == EFFECTIVE
    assert model_fallback_badge(job.requested_model_label, job.model_label) == (
        f"{PIN} → DeepSeek Flash ⚠ fallback"
    )


def test_a_recovery_edge_clears_the_marker_and_its_reason() -> None:
    """(b) A badge that outlived the fallback would claim a substitution that
    has ended — the same class of lie as the silent one, in reverse."""
    job = _pinned_job()
    relay = _relay(job, [])
    asyncio.run(relay(_fallback_event()))
    assert job.model_fallback is True

    asyncio.run(relay(_recovery_event()))
    assert job.model_fallback is False
    assert job.model_fallback_reason == ""
    assert job.model_label == PIN
    assert model_fallback_badge(job.requested_model_label, job.model_label) == ""


def test_an_unpinned_child_is_never_marked_and_never_notices() -> None:
    """(c) The negative control: inherit-by-design children stay untouched.

    A plain ``task`` child (no effort tier, no role pin) passes
    ``model_spec=None`` and ``owns_model`` stays False; the effective label is
    still kept current — that was always the relay's job — but none of the
    pin-integrity state may fire.
    """
    job = _pinned_job()
    job.owns_model = False
    job.requested_model_label = None
    emitted: list[Any] = []
    asyncio.run(_relay(job, emitted)(_fallback_event()))

    assert job.model_fallback is False
    assert job.model_fallback_reason == ""
    assert job.model_label == EFFECTIVE  # the label tracking still works
    assert [e for e in emitted if isinstance(e, NoticeEvent)] == []


def test_a_fallback_that_lands_on_the_pin_does_not_mark_the_job() -> None:
    """A route edge reporting the PIN as the effective model is not a
    substitution — nothing to mark, nothing to say."""
    job = _pinned_job()
    emitted: list[Any] = []
    event = _fallback_event()
    event.provider, event.model_id = "anthropic", "claude-sonnet-5-5"
    asyncio.run(_relay(job, emitted)(event))

    assert job.model_fallback is False
    assert [e for e in emitted if isinstance(e, NoticeEvent)] == []


# ---------------------------------------------------------------------------
# the parent-stream notice
# ---------------------------------------------------------------------------


def test_the_notice_carries_the_four_facts() -> None:
    """(d) The operator-visible sentence: WHICH child (label + role), the pin,
    the effective model, and the route edge's own cause.

    The enriched cause ("anthropic refused (rate limit)") is what the failover
    driver will hand the event once the mechanism names the refusing provider;
    this test fakes that input because the notice's contract is pass-through —
    it must carry the phrase VERBATIM and never re-derive or invent one.
    """
    job = _pinned_job()
    emitted: list[Any] = []
    asyncio.run(_relay(job, emitted)(_fallback_event(reason="anthropic refused (rate limit)")))

    notices = [e for e in emitted if isinstance(e, NoticeEvent)]
    assert len(notices) == 1
    notice = notices[0]
    assert notice.kind == "warning"
    text = notice.text
    assert "round1-designer" in text  # the child's label
    assert "(designer)" in text  # its role, spelled as the role clause
    # The two models in the product's own vocabulary (UX U3), joined as an
    # `A → B` pair — the same relation spelling the band's badge uses (D4).
    assert "Claude Sonnet 5.5 → DeepSeek Flash" in text.replace("\u00a0", " ")
    # And the pair is ONE wrap token: `wrap_cells` splits on ASCII spaces
    # only, so it can never be broken between requested and effective.
    from local_operator.tui.widgets.transcript import wrap_cells

    assert "\u00a0→\u00a0" in text
    for width in (110, 74, 60):
        rows = wrap_cells(text, width)
        assert any(
            "Claude\u00a0Sonnet\u00a05.5\u00a0→\u00a0DeepSeek\u00a0Flash" in row for row in rows
        ), (width, rows)
    assert "anthropic" in text and "rate limit" in text  # the refusing cause
    # The toast glance names the child and the target rather than slicing prose.
    assert "round1-designer" in notice.headline
    assert EFFECTIVE in notice.headline

    # And today's un-enriched reason is carried just as faithfully — the
    # sentence loses detail, never truth.
    plain_job = _pinned_job()
    plain_emitted: list[Any] = []
    asyncio.run(_relay(plain_job, plain_emitted)(_fallback_event()))
    [plain] = [e for e in plain_emitted if isinstance(e, NoticeEvent)]
    assert "provider failure" in plain.text


def test_one_notice_per_fallback_episode() -> None:
    """A later hop to another fallback updates the row without re-announcing
    the descent; a genuine recovery clears the marker so the NEXT episode
    speaks again — dedup on the transition, not on the event count."""
    job = _pinned_job()
    emitted: list[Any] = []
    relay = _relay(job, emitted)

    asyncio.run(relay(_fallback_event()))
    asyncio.run(relay(_fallback_event()))
    assert len([e for e in emitted if isinstance(e, NoticeEvent)]) == 1

    asyncio.run(relay(_recovery_event()))
    asyncio.run(relay(_fallback_event()))
    notices = [e for e in emitted if isinstance(e, NoticeEvent)]
    assert len(notices) == 2, "a new episode after a recovery is news again"
    assert job.model_fallback is True


def test_the_notice_builder_states_the_role_only_when_recorded() -> None:
    """A duck row without a role still gets a grammatical sentence."""
    from local_operator.harness.subagent import _pinned_fallback_notice

    text = _pinned_fallback_notice("a", "", PIN, EFFECTIVE, "cause")
    assert text == (
        "subagent 'a' pinned Claude\u00a0Sonnet\u00a05.5\u00a0→\u00a0DeepSeek\u00a0Flash — cause."
    )


# ---------------------------------------------------------------------------
# the roster round trip (the row that survives a restart)
# ---------------------------------------------------------------------------


def test_the_pin_label_survives_the_roster_round_trip() -> None:
    """(f) A restored row has labels but no runtime flag; the badge must still
    render from the comparison, which is why ``requested_model_label`` is a
    persisted field — and why the flag/reason pair deliberately is not (a
    strict reader that meets an unknown key drops the whole row)."""
    from local_operator.session.session import _subagent_job_row

    job = AsyncJob(
        start_time=0,
        id="designer1",
        type="task",
        label="round1-designer",
        model_label=EFFECTIVE,
        requested_model_label=PIN,
        owns_model=True,
        model_fallback=True,
        model_fallback_reason="provider failure",
    )
    row = _subagent_job_row(job)
    assert row["requested_model_label"] == PIN
    assert "model_fallback" not in row
    assert "model_fallback_reason" not in row

    restored = AsyncJob.model_validate(row)
    assert restored.requested_model_label == PIN
    assert restored.model_label == EFFECTIVE
    assert restored.model_fallback is False  # the reason is what a restart loses
    assert model_fallback_badge(restored.requested_model_label, restored.model_label) != ""


# ---------------------------------------------------------------------------
# the badge renderer and the surfaces that paint it
# ---------------------------------------------------------------------------


def test_the_badge_renders_only_while_the_pin_is_off() -> None:
    """One rule, three spellings: the full pair, the effective half only when
    the row cannot hold the pair, and nothing at all while the pin serves."""
    assert model_fallback_badge(PIN, EFFECTIVE) == f"{PIN} → DeepSeek Flash ⚠ fallback"
    # Serving the pin, no pin, or nothing to compare: no badge.
    assert model_fallback_badge(PIN, PIN) == ""
    assert model_fallback_badge(None, EFFECTIVE) == ""
    assert model_fallback_badge(PIN, None) == ""
    assert model_fallback_badge("", "") == ""
    # The SAME predicate every pin-integrity surface keys on (the dock's
    # marker, the projection's boolean, the completion row's clause):
    # divergence of the two labels, never the runtime-only flag.
    assert is_model_fallback(PIN, EFFECTIVE) is True
    assert is_model_fallback(PIN, PIN) is False
    assert is_model_fallback(None, EFFECTIVE) is False
    assert is_model_fallback("", "") is False
    # Naming's honesty rule bounds the effective half: an unresolvable model
    # keeps its selector rather than acquiring a fictional name.
    assert model_fallback_badge("a/x", "unknown/vendor-model") == (
        "a/x → unknown/vendor-model ⚠ fallback"
    )


def test_the_band_paints_the_badge_for_a_pinned_child_off_its_model() -> None:
    """The TUI surface ``job_stats``' model reaches: while the child's page is
    open the band must show the substitution instead of the bare effective
    name — in the form the width can afford, never at the expense of the
    ordinary readings (design D1 / UX U2: at 80x24 the 64-cell pair used to
    take cwd and the context reading with it, leaving the band LESS
    informative than the base row it replaced).
    """
    from local_operator.tui.widgets.status_line import StatusLine, SubagentBand
    from tests.unit.tui.test_subagent_stats import _Dock

    status = StatusLine(cast(Static, _Dock(150)))
    status.update(
        model_label=EFFECTIVE, context_tokens=1, context_window=10, cwd="/Users/tester/wt"
    )
    status.set_subagent(
        SubagentBand(
            model_label=EFFECTIVE,
            requested_model_label=PIN,
            label="round1-designer",
            context_tokens=44_000,
            context_window=160_000,
            cost="$0.041",
            effort="hi",
        )
    )

    # Wide: the full pair, with the EFFECTIVE half resolved to its display
    # name (D5) while the requested half stays the pin's recorded selector.
    wide = status.render_text(150).plain
    assert f"{PIN} → DeepSeek Flash ⚠ fallback" in wide, wide

    # A 80-column terminal's band (75-76 cells): the pair sheds to the marker
    # plus the effective half's short display form, and EVERYTHING the base
    # row carried at this width survives.
    narrow = status.render_text(76).plain
    assert "⚠ DeepSeek Flash" in narrow, narrow
    assert "→" not in narrow, narrow
    assert "⌂ wt" in narrow, narrow
    assert "27.5%/160k" in narrow, narrow

    # Below the ladder the irreducible rung carries the same marker under the
    # child's name — the design round's proposed shape.
    tiny = status.render_text(30).plain
    assert "⚠ DeepSeek Flash" in tiny, tiny
    assert "round1-des" in tiny, tiny

    # The ⚠ takes the app's `warning` semantic as an ADDITIVE cue (D3): the
    # text cues (→, ⚠, fallback) do not move, and NO_COLOR loses none of them.
    from local_operator.tui import theme as theme_mod

    ink = {
        wide[span.start : span.end].strip(): getattr(span.style, "color", None)
        for span in status.render_text(150).spans
    }
    marker_ink = ink.get("⚠")
    assert marker_ink is not None
    assert marker_ink.name == theme_mod.semantic_color("warning")

    # The negative arm: a child on its pin paints the resolved display name as
    # before, and a consumer that carries no pin (every band built before this
    # field existed) is unaffected.
    status.set_subagent(SubagentBand(model_label=PIN, requested_model_label=PIN))
    on_pin = status.render_text(150).plain
    assert "⚠" not in on_pin, on_pin


def test_the_completion_row_states_the_pin_the_walk_away_reader_needs() -> None:
    """UX round 1, U1: the walk-away flow ends on the completion delivery.

    The transcript notice is a LIVE row — it scrolls and no replay carries
    it — and the roster distinguishes a completed pinned-fallback child only
    via the dock marker, so the durable row itself must say the pin was
    abandoned. Selectors, because this row is handed to the MODEL as well as
    shown, and the ``wait`` receipt beside it speaks the same spelling; the
    route edge's cause rides along when the live row still carries one.
    """
    from local_operator.session.session import Session

    job = _pinned_job()
    job.model_label = EFFECTIVE
    job.model_fallback_reason = "provider failure"
    text = Session._job_result_message("designer1", "the review findings", job).details["text"]
    assert text.startswith("background job 'round1-designer' completed")
    assert f"(pinned {PIN}, ran on {EFFECTIVE} — provider failure)" in text

    # No cause recorded (a restart, or an un-enriched edge): the clause keeps
    # the two labels and drops only the phrase it does not have.
    job.model_fallback_reason = ""
    bare = Session._job_result_message("designer1", "x", job).details["text"]
    assert f"(pinned {PIN}, ran on {EFFECTIVE})" in bare

    # Recovered or unpinned: the row is byte-identical to the one it always
    # read.
    job.model_label = PIN
    recovered = Session._job_result_message("designer1", "x", job).details["text"]
    assert "(pinned" not in recovered
    unpinned = _pinned_job()
    unpinned.requested_model_label = None
    plain = Session._job_result_message("designer1", "x", unpinned).details["text"]
    assert "(pinned" not in plain


def test_the_notice_and_row_carry_the_cross_vendor_clause() -> None:
    """The walk's enriched settle reason — these exact words are what the
    SHIPPED default emits on a cross-vendor descent (the end-to-end string is
    asserted in tests/unit/providers/test_failover.py) — must reach the
    parent notice and the durable completion row VERBATIM.

    This is the disclosure half of the default policy: the child keeps
    running on the substitute, so the reason is the surface that tells the
    operator the substitution was a cross-vendor descent rather than an
    ordinary same-family hop.
    """
    from local_operator.session.session import Session

    reason = (
        "provider failure: quota HTTP 429 "
        "(cross-vendor descent for pinned anthropic/claude-sonnet-5-5)"
    )
    job = _pinned_job()
    emitted: list[Any] = []
    asyncio.run(_relay(job, emitted)(_fallback_event(reason=reason)))

    [notice] = [e for e in emitted if isinstance(e, NoticeEvent)]
    assert reason in notice.text
    assert "cross-vendor descent" in notice.text

    text = Session._job_result_message("designer1", "x", job).details["text"]
    assert f"(pinned {PIN}, ran on {EFFECTIVE} — {reason})" in text


def test_job_stats_reads_the_registration_stamp_off_the_job() -> None:
    """The stats are how the band (and every duck-typed consumer) learns both
    labels; a host whose job predates the fields degrades to ""."""
    stats = job_stats(_pinned_job())
    assert stats.model_label == PIN
    assert stats.requested_model_label == PIN

    legacy = SimpleNamespace(model_label=EFFECTIVE)
    degraded = job_stats(legacy)
    assert degraded.model_label == EFFECTIVE
    assert degraded.requested_model_label == ""
    assert model_fallback_badge(degraded.requested_model_label, degraded.model_label) == ""


# ---------------------------------------------------------------------------
# the parent's wait receipt
# ---------------------------------------------------------------------------


def test_job_summary_names_the_pin_beside_the_effective_model() -> None:
    """(e) The delegating model's own view: ``model=<effective>`` alone let a
    fallback read as the pin it asked for."""
    from local_operator.tools.builtin import _job_summary

    job = SimpleNamespace(
        id="designer1",
        type="task",
        label="round1-designer",
        status="completed",
        model_label=EFFECTIVE,
        requested_model_label=PIN,
        result_text="done",
    )
    text, _details = _job_summary(job, None)
    assert f"model={EFFECTIVE} (pinned {PIN})" in text

    # No divergence (or no pin): today's single spelling, unchanged.
    job.model_label = PIN
    plain, _details = _job_summary(job, None)
    assert f"model={PIN}" in plain and "(pinned" not in plain

    job.requested_model_label = None
    unpinned, _details = _job_summary(job, None)
    assert f"model={PIN}" in unpinned and "(pinned" not in unpinned


# ---------------------------------------------------------------------------
# the registration stamp (the writer that makes all of the above possible)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_launch_stamps_the_requested_model_at_registration(tmp_path, monkeypatch) -> None:
    """The pin label is recorded where ``owns_model`` is: at REGISTRATION.

    A queued job that never starts, and the runner's later effective-label
    overwrite alike, must not erase what the launch asked for — without this
    stamp there is nothing for any of the surfaces above to compare against.
    """
    from tests.unit.session.test_pinned_subagent_model import (
        make_session,
        wait_for,
        write_tiers,
    )

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    write_tiers(tmp_path / "config", hi="openrouter/moonshotai/kimi-k3")
    session = make_session(tmp_path)
    job_id = session._launch_subagent(
        label="designer", prompt="review it", agent="designer", effort="hi"
    )
    job = session.jobs.get(job_id)
    assert job is not None
    assert job.owns_model is True
    assert job.requested_model_label == "openrouter/moonshotai/kimi-k3"
    # The effective label tracks the pin while the child runs on it, and the
    # requested stamp is unmoved by the runner's overwrite.
    await wait_for(lambda: job.status == "completed")
    assert job.model_label == "openrouter/moonshotai/kimi-k3"
    assert job.requested_model_label == "openrouter/moonshotai/kimi-k3"
    await session.dispose()


@pytest.mark.asyncio
async def test_an_unpinned_launch_records_no_requested_model(tmp_path, monkeypatch) -> None:
    """The inherit path writes neither stamp: ``owns_model`` stays False and
    ``requested_model_label`` stays ``None`` — the convention every reader
    above keys on."""
    from tests.unit.session.test_pinned_subagent_model import make_session, wait_for

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    session = make_session(tmp_path)
    job_id = session._launch_subagent(label="coder", prompt="code it")
    job = session.jobs.get(job_id)
    assert job is not None
    assert job.owns_model is False
    assert job.requested_model_label is None
    await wait_for(lambda: job.status == "completed")
    assert job.requested_model_label is None
    await session.dispose()


def test_the_wire_row_carries_the_pin_to_an_attached_viewer() -> None:
    """An attached viewer builds its job rows via ``JobState.from_job``; the
    band there must be able to render the same badge, and an unpinned row's
    ``None`` must survive the strict validation (the sidecar shape)."""
    from local_operator.session.frontend_state import JobState

    job = _pinned_job()
    job.model_label = EFFECTIVE  # the route has moved it off the pin
    wire = JobState.from_job(job)
    assert wire.requested_model_label == PIN
    assert wire.model_label == EFFECTIVE
    assert model_fallback_badge(wire.requested_model_label, wire.model_label) != ""

    unpinned = JobState.from_job(SimpleNamespace(model_label="test/m"))
    assert unpinned.requested_model_label is None
