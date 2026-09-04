"""The render-seam byte guard and the graduated 413 recovery, end to end.

Companion to ``tests/unit/compaction/test_wire_bytes.py``, which covers the
pure ruler/trigger/shed. What is proved HERE is the part that was broken for
the user: a session whose history is already 34 MB on disk becomes sendable on
its next launch, without anyone knowing to type ``/compact``, and without the
transcript being rewritten.

The failure being regressed: 42 screenshots totalling 33.9 MB against
Anthropic's 32 MB cap wedged a session on ``invalid request (HTTP 413):
Request exceeds the maximum size``. Every later turn failed identically —
including ``/compact``, which has to SEND the history to summarise it — so the
session could not recover from the inside.
"""

from __future__ import annotations

import types

import pytest

from local_operator.compaction.api import CompactionSettings
from local_operator.harness.types import (
    AbortSignal,
    ChatRequest,
    ImageContent,
    Message,
    ModelSpec,
    NoticeEvent,
    StreamEndEvent,
    StreamTextDelta,
    TextContent,
    ToolCall,
)
from local_operator.providers.failover import ProviderError
from local_operator.session.session import FRAMES_SHED_NOTICE, Session
from local_operator.session.transcript import Transcript

MODEL = ModelSpec(provider="test", model_id="m", context_window=1_000_000)

#: Median base64 length of the 42 frames in the session that wedged.
FRAME_B64 = 803_888


def make_session(tmp_path, stream, **kwargs) -> Session:
    return Session(
        model=kwargs.pop("model", MODEL),
        stream_fn=stream,
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["stable"],
        **kwargs,
    )


def _frames(count: int, size: int = FRAME_B64) -> list[Message]:
    """A screenshot-driving history at the measured frame size."""
    out: list[Message] = []
    for index in range(count):
        out.append(
            Message(
                role="user",
                content=[TextContent(text=f"shot {index}"), ImageContent(data="A" * size)],
            )
        )
        out.append(Message.assistant(f"ok {index}"))
    return out


def _image_blocks(request: ChatRequest) -> int:
    return sum(
        isinstance(block, ImageContent) for message in request.messages for block in message.content
    )


def _request_bytes(request: ChatRequest) -> int:
    from local_operator.compaction.api import estimate_wire_bytes

    return estimate_wire_bytes(request.messages)


class ScriptedOk:
    """Records every request and always succeeds."""

    def __init__(self) -> None:
        self.requests: list[ChatRequest] = []

    def __call__(self, request: ChatRequest, signal: AbortSignal | None):
        self.requests.append(request)

        async def gen():
            yield StreamTextDelta(delta="ok")
            yield StreamEndEvent(stop_reason="endTurn")

        return gen()


class RefusesOversizeRequests:
    """A provider with a HARD byte cap, like the real one.

    Refuses with Anthropic's literal 413 wording whenever the request it is
    handed exceeds ``cap``, which is what makes this a faithful reproduction:
    the refusal is a property of the request's SIZE, so it recurs on every
    turn until something actually sends fewer bytes.
    """

    def __init__(self, cap: int) -> None:
        self.cap = cap
        self.requests: list[ChatRequest] = []
        self.refusals = 0

    def __call__(self, request: ChatRequest, signal: AbortSignal | None):
        self.requests.append(request)
        too_big = _request_bytes(request) > self.cap
        if too_big:
            self.refusals += 1

        async def gen():
            if too_big:
                raise ProviderError(413, "Request exceeds the maximum size")
            yield StreamTextDelta(delta="ok")
            yield StreamEndEvent(stop_reason="endTurn")

        return gen()


# ---------------------------------------------------------------------------
# The render seam
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_oversize_history_is_shed_at_the_render_seam(tmp_path):
    """THE regression: a 34 MB history is sendable on the very next turn.

    No ``/compact``, no user action, no provider round trip — the shed is a
    local byte scan at the one seam every wire-history path converges on.
    """
    stream = ScriptedOk()
    session = make_session(tmp_path, stream)
    await session.seed_history(_frames(42))

    await session.prompt("continue")

    sent = _request_bytes(stream.requests[0])
    budget = session._wire_bytes_budget()
    assert budget == 24_000_000
    assert sent <= budget, f"the request went out at {sent} bytes, over the {budget} budget"
    # The RECENT frames — the ones the model is actually working with — stayed.
    assert 25 <= _image_blocks(stream.requests[0]) <= 30
    await session.dispose()


@pytest.mark.asyncio
async def test_the_shed_never_touches_the_transcript(tmp_path):
    """The contract that makes this safe: the stored frames survive, so
    ``/export``, forks, and a later session on a provider with a larger cap
    all still see every screenshot."""
    stream = ScriptedOk()
    session = make_session(tmp_path, stream)
    await session.seed_history(_frames(42))

    path = session._transcript.path
    before = path.read_bytes()

    await session.prompt("continue")

    assert _image_blocks(stream.requests[0]) < 42, "nothing was shed; the test proves nothing"

    # Counted over the RAW entries, not over ``build_llm_history``: the byte
    # trigger also fires a real compaction pass on a history this size, and
    # replay honours that pass's cut. What this test owns is the stronger and
    # narrower claim — the stored rows are untouched, so ``/export``, a fork,
    # and a later session on a larger-cap provider still have every frame.
    # Image blocks are externalized to content-addressed ATTACHMENT refs on
    # append (which is why a 34 MB history is under 1 MB on disk), so the
    # stored form is ``{"attachment": <digest>, ...}``, not ``type: image``.
    stored = sum(
        1
        for entry in session._transcript.entries()
        for block in entry.payload.get("content", []) or []
        if isinstance(block, dict) and ("attachment" in block or block.get("type") == "image")
    )
    assert stored == 42, f"the shed reached the transcript ({stored} of 42 frames left)"
    # Append-only rather than byte-identical: a history this size ALSO trips
    # the soft byte trigger, and that compaction pass appends its marker entry
    # legitimately. What must never happen is an existing row being rewritten,
    # which is what a prefix check proves and a length check would not.
    assert path.read_bytes().startswith(before), "an existing transcript row was rewritten"
    await session.dispose()


@pytest.mark.asyncio
async def test_a_session_under_budget_is_completely_unaffected(tmp_path):
    """The guarantee that this changes nothing for ordinary sessions: images
    a user pasted are distinct evidence and must never be shed for size of
    CONTEXT, only to keep a request legal."""
    stream = ScriptedOk()
    session = make_session(tmp_path, stream)
    await session.seed_history(_frames(4))

    notices: list[str] = []
    session.subscribe(lambda e: notices.append(e.text) if isinstance(e, NoticeEvent) else None)

    await session.prompt("continue")

    assert _image_blocks(stream.requests[0]) == 4, "an under-budget session lost a frame"
    assert FRAMES_SHED_NOTICE not in notices
    await session.dispose()


@pytest.mark.asyncio
async def test_the_user_is_told_once_that_old_screenshots_were_dropped(tmp_path):
    """Silent context loss is the main risk of shedding, so it is announced —
    and announced ONCE, because ``_render_history`` runs several times per
    turn and a per-render notice would repeat itself."""
    stream = ScriptedOk()
    session = make_session(tmp_path, stream)
    await session.seed_history(_frames(42))

    notices: list[str] = []
    session.subscribe(lambda e: notices.append(e.text) if isinstance(e, NoticeEvent) else None)

    await session.prompt("one")
    await session.prompt("two")
    await asyncio_yield()

    assert notices.count(FRAMES_SHED_NOTICE) == 1
    await session.dispose()


@pytest.mark.asyncio
async def test_the_shed_can_be_turned_off(tmp_path):
    """An operator on a provider with a larger cap keeps every frame."""
    stream = ScriptedOk()
    session = make_session(
        tmp_path, stream, compaction_settings=CompactionSettings(wire_bytes_budget=0)
    )
    await session.seed_history(_frames(42))

    await session.prompt("continue")

    assert _image_blocks(stream.requests[0]) == 42
    await session.dispose()


# ---------------------------------------------------------------------------
# The graduated 413 recovery
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_413_recovers_without_the_sticky_image_degrade(tmp_path):
    """The over-reaction guard, and the reason 413 got its own predicate.

    ``_images_rejected`` drops EVERY image for the whole session, forever. For
    a size problem where shedding a few frames restores the session that is
    catastrophically heavy-handed — and its own justification ("not a
    preventable condition on our side") does not apply, because a 413 is
    preventable by sending fewer bytes.
    """
    # A cap below the default budget, so the render seam's shed is not enough
    # on its own and the reactive ladder has to engage.
    stream = RefusesOversizeRequests(cap=12_000_000)
    session = make_session(tmp_path, stream)
    await session.seed_history(_frames(42))

    await session.prompt("first")
    assert stream.refusals == 1, "the fixture did not reproduce the refusal"
    assert not session._images_rejected, "a 413 tripped the sticky image degrade"
    assert session._wire_budget_override is not None, "the budget did not ratchet down"
    assert session._wire_budget_override < 24_000_000

    # The NEXT turn is correct without any re-entry into the failed turn.
    await session.prompt("second")
    assert _request_bytes(stream.requests[-1]) < _request_bytes(stream.requests[0])
    assert _image_blocks(stream.requests[-1]) > 0, "images survived; only the oldest went"
    await session.dispose()


@pytest.mark.asyncio
async def test_the_ratchet_converges_until_the_request_is_accepted(tmp_path):
    """Each refusal is a measurement of the real cap — a proxy in front of the
    API can refuse below the documented limit — so the budget tightens until
    the provider accepts, rather than guessing from a per-provider table."""
    stream = RefusesOversizeRequests(cap=8_000_000)
    session = make_session(tmp_path, stream)
    await session.seed_history(_frames(42))

    for attempt in range(6):
        await session.prompt(f"turn {attempt}")
        if _request_bytes(stream.requests[-1]) <= stream.cap:
            break

    assert _request_bytes(stream.requests[-1]) <= stream.cap, "never converged"
    assert not session._images_rejected
    await session.dispose()


@pytest.mark.asyncio
async def test_a_413_is_classified_with_an_actionable_hint(tmp_path):
    """Defect 4: the model was told ``unknown`` with an empty hint and retried
    the identical request. It is journaled as an incident the model can read."""
    from local_operator.incidents import classify_incident

    rendered = str(ProviderError(413, "Request exceeds the maximum size"))
    incident = classify_incident(rendered)

    assert incident.category == "context-length"
    assert incident.hint, "the model was given nothing actionable"
    assert "retry" in incident.hint.lower()


@pytest.mark.asyncio
async def test_an_ordinary_image_refusal_still_takes_the_sticky_degrade(tmp_path):
    """The size ladder must not swallow the failure mode it sits in front of:
    a genuine per-block refusal is still a poisoned image."""

    class RefusesImages:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, request: ChatRequest, signal: AbortSignal | None):
            self.calls += 1
            first = self.calls == 1

            async def gen():
                if first:
                    raise ProviderError(400, "Could not process image")
                yield StreamTextDelta(delta="ok")
                yield StreamEndEvent(stop_reason="endTurn")

            return gen()

    session = make_session(tmp_path, RefusesImages())
    await session.seed_history(_frames(2))

    await session.prompt("go")

    assert session._images_rejected, "the sticky degrade stopped firing for real refusals"
    assert session._wire_budget_override is None, "a block refusal ratcheted the byte budget"
    await session.dispose()


# ---------------------------------------------------------------------------
# The byte-side anti-thrash band (the architect's top risk)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_byte_triggered_pass_still_over_budget_schedules_no_continuation(tmp_path):
    """Risk 4, the dead-loop guard.

    ``RECOVERY_BAND`` is defined on TOKENS. A byte-triggered pass leaves the
    token residual far inside that band — 154,690 tokens is 15% of a 1M
    window — so the token side alone would say "headroom created" and queue an
    auto-continue, whose next turn re-fires the byte trigger on a context
    nothing shrank. That is exactly the live dead loop ``RECOVERY_BAND`` was
    added to prevent, reached through the new trigger.
    """
    from local_operator.compaction import api as compaction_api
    from local_operator.session.session import _CompactionPlan

    stream = ScriptedOk()
    session = make_session(
        tmp_path,
        stream,
        # No shed, so the render used by the band check stays over budget.
        compaction_settings=CompactionSettings(wire_bytes_budget=0),
    )
    await session.seed_history(_frames(42))

    settings = CompactionSettings(wire_bytes_budget=0)
    plan = _CompactionPlan(
        compaction_api=compaction_api,
        settings=settings,
        strategy="snapcompact",
        llm_history=[],
        cut=1,
        context_tokens=154_690,
        tokens_before=154_690,
    )

    # The token band passes trivially: 154,690 is well under 0.8 * 600,000.
    threshold = compaction_api.resolve_threshold_tokens(1_000_000, settings)
    assert 154_690 <= compaction_api.RECOVERY_BAND * threshold

    # The byte band does not, because the render is still ~34 MB.
    assert session._cleared_wire_headroom(plan) is False
    await session.dispose()


@pytest.mark.asyncio
async def test_the_byte_band_allows_a_continuation_once_the_payload_is_small(tmp_path):
    """The band withholds only while the pass has genuinely not recovered."""
    from local_operator.compaction import api as compaction_api
    from local_operator.session.session import _CompactionPlan

    session = make_session(tmp_path, ScriptedOk())
    await session.seed_history(_frames(2))

    settings = CompactionSettings()
    plan = _CompactionPlan(
        compaction_api=compaction_api,
        settings=settings,
        strategy="snapcompact",
        llm_history=[],
        cut=1,
        context_tokens=1_000,
        tokens_before=1_000,
    )

    assert session._cleared_wire_headroom(plan) is True
    await session.dispose()


async def asyncio_yield() -> None:
    """Let background notice emissions land before asserting on them."""
    import asyncio

    await asyncio.sleep(0)
    await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# The ladder's terminal rung (agent review round 1 R1/R2, QA round 1 Q1/Q2)
# ---------------------------------------------------------------------------
#
# Both round 1 gates independently reproduced a permanent wedge here, which is
# the exact failure class this whole change exists to delete. The two traps:
#
#   R1 — the handover was gated on ``at_floor AND frames_left == 0``, so at the
#        floor any surviving frame short-circuited it and the ladder returned
#        "handled" forever while changing nothing.
#   R2/Q1 — the terminal rung DELEGATED to ``_degrade_if_image_rejected``,
#        whose ``is_image_rejection`` guard is False for a 413 by construction,
#        so it was structurally unreachable.
#
# The previous tests missed both because their caps sat ABOVE
# ``_WIRE_BUDGET_FLOOR`` and their histories were all frames. These cover the
# cap BELOW the floor and the text-only history specifically.


@pytest.mark.asyncio
async def test_a_cap_below_the_floor_reaches_the_terminal_rung(tmp_path):
    """R1: at the floor the budget cannot move, so the ladder must hand over.

    A proxy capping below ``_WIRE_BUDGET_FLOOR`` is the ratchet docstring's own
    stated reason to exist. Before the fix this ratcheted to 4 MB and then
    returned "handled" on every subsequent turn with nothing changing —
    measured at 15 consecutive 413s.
    """
    from local_operator.session.session import _WIRE_BUDGET_FLOOR

    stream = RefusesOversizeRequests(cap=2_000_000)
    assert stream.cap < _WIRE_BUDGET_FLOOR, "the trap needs a cap below the floor"

    session = make_session(tmp_path, stream)
    await session.seed_history(_frames(42))

    for turn in range(12):
        await session.prompt(f"t{turn}")
        if stream.requests and _request_bytes(stream.requests[-1]) <= stream.cap:
            break

    assert session._images_rejected, "the terminal rung was never reached"
    assert _request_bytes(stream.requests[-1]) <= stream.cap, "never became sendable"
    assert _image_blocks(stream.requests[-1]) == 0
    await session.dispose()


@pytest.mark.asyncio
async def test_a_text_only_oversize_history_reaches_the_terminal_rung(tmp_path):
    """R2/Q2: with no frames to shed, tightening the budget buys nothing.

    The payload never moves while the budget ratchets away beneath it, and the
    user is told "send your message again" — advice that cannot work. The
    ladder must recognise it has no lever and stop claiming success.
    """
    stream = RefusesOversizeRequests(cap=1_000_000)
    session = make_session(tmp_path, stream)
    await session.seed_history([Message.user("T" * 5_000_000), Message.assistant("ack")])

    for turn in range(6):
        await session.prompt(f"t{turn}")

    # Nothing to shed means the FIRST refusal is already terminal: there is no
    # sequence of budget cuts that makes a text payload smaller.
    assert session._images_rejected, "the ladder kept ratchetting a budget it could not use"
    assert session._wire_budget_override is None, "the budget was tightened for no gain"
    await session.dispose()


@pytest.mark.asyncio
async def test_the_terminal_rung_is_reached_without_loosening_is_image_rejection(tmp_path):
    """The invariant, stated as a test: the ladder sets the sticky flag ITSELF.

    Both gates agreed that teaching ``is_image_rejection`` about 413 is the
    WRONG fix — it would give every size refusal the sticky whole-session
    degrade this ladder exists to avoid. So the predicate must still decline
    the very error that just drove the terminal rung.
    """
    from local_operator.providers.failover import is_image_rejection

    stream = RefusesOversizeRequests(cap=1_000_000)
    session = make_session(tmp_path, stream)
    await session.seed_history([Message.user("T" * 5_000_000)])

    await session.prompt("go")

    assert session._images_rejected, "the terminal rung did not fire"
    assert not is_image_rejection(ProviderError(413, "Request exceeds the maximum size"))
    await session.dispose()


@pytest.mark.asyncio
async def test_the_terminal_rung_does_not_repeat_its_notice(tmp_path):
    """Once terminal, later refusals must not re-announce the same degrade."""
    stream = RefusesOversizeRequests(cap=1_000)
    session = make_session(tmp_path, stream)
    await session.seed_history([Message.user("T" * 5_000_000)])

    notices: list[str] = []
    session.subscribe(lambda e: notices.append(e.text) if isinstance(e, NoticeEvent) else None)

    for turn in range(3):
        await session.prompt(f"t{turn}")

    # Matched on the terminal advice rather than a phrase, so the assertion is
    # about the notice firing once and not about its exact wording (which
    # branches on whether the history had frames at all — round 2, R9).
    degrade_notices = [n for n in notices if "/compact" in n]
    assert len(degrade_notices) == 1, f"the degrade announced itself {len(degrade_notices)}x"
    await session.dispose()


# ---------------------------------------------------------------------------
# Override lifetime (agent review round 1, R4)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_provider_switch_clears_the_measured_budget(tmp_path):
    """R4a: the cap belonged to the provider that demonstrated it.

    Keeping it pins the session to a departed connection's limit, shedding
    screenshots the new provider would have accepted.
    """
    stream = RefusesOversizeRequests(cap=12_000_000)
    session = make_session(tmp_path, stream)
    await session.seed_history(_frames(42))

    await session.prompt("first")
    assert session._wire_budget_override is not None, "no ratchet to clear"

    session.set_model(ModelSpec(provider="other", model_id="lax", context_window=1_000_000))
    assert session._wire_budget_override is None, "the departed provider's cap survived"
    assert session._wire_bytes_budget() == 24_000_000
    await session.dispose()


@pytest.mark.asyncio
async def test_a_same_model_knob_change_keeps_the_measured_budget(tmp_path):
    """The other half of R4a: an `/effort` keystroke is not a provider change,
    so it must not discard a limit that is still in force."""
    stream = RefusesOversizeRequests(cap=12_000_000)
    session = make_session(tmp_path, stream)
    await session.seed_history(_frames(42))

    await session.prompt("first")
    measured = session._wire_budget_override
    assert measured is not None

    session.set_model(MODEL.model_copy(update={"temperature": 0.5}))
    assert session._wire_budget_override == measured, "a knob change dropped a real limit"
    await session.dispose()


@pytest.mark.asyncio
async def test_raising_the_live_setting_clears_the_measured_budget(tmp_path):
    """R4b: the key is registered LIVE precisely so an operator watching a
    session shed can raise it without restarting. The ``min()`` against the
    ratchet made that a painted lie."""
    stream = RefusesOversizeRequests(cap=12_000_000)
    session = make_session(tmp_path, stream)
    await session.seed_history(_frames(42))

    await session.prompt("first")
    assert session._wire_budget_override is not None

    session._apply_config_change(
        types.SimpleNamespace(
            changed_keys=frozenset({"compaction.wire_bytes_budget"}),
            values={"compaction": {"wire_bytes_budget": 30_000_000}},
        )
    )

    assert session._wire_budget_override is None
    assert session._wire_bytes_budget() == 30_000_000, "the explicit edit was swallowed"
    await session.dispose()


# ---------------------------------------------------------------------------
# The size-driven strip is provider-scoped (agent review round 2, R8/R9)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_provider_switch_lifts_a_size_driven_image_strip(tmp_path):
    """R8: the strip was evidence about the DEPARTED provider's cap.

    R4 already established the principle for the budget. The sticky flag is
    set from the same evidence, so it needs the same lifetime — otherwise the
    session stays permanently blind on a provider that would have accepted
    every frame, which is the exact harm R4's comment argues against.
    """
    stream = RefusesOversizeRequests(cap=1_000)
    session = make_session(tmp_path, stream)
    await session.seed_history(_frames(6))

    # The ladder ratchets before it strips, so drive it to the terminal rung
    # rather than assuming one refusal reaches it.
    for turn in range(12):
        await session.prompt(f"t{turn}")
        if session._images_rejected:
            break
    assert session._images_rejected, "the size strip never fired"
    assert session._images_rejected_for_size

    session.set_model(ModelSpec(provider="other", model_id="lax", context_window=1_000_000))

    assert not session._images_rejected, "the departed provider's strip survived"
    assert not session._images_rejected_for_size
    assert session._wire_budget_override is None
    # The next render carries images again, which is the point of lifting it.
    rendered = session._render_history(list(session._context.messages))
    assert any(
        isinstance(block, ImageContent) for message in rendered for block in message.content
    ), "images did not come back for the new provider"
    await session.dispose()


@pytest.mark.asyncio
async def test_a_provider_switch_does_not_lift_a_refusal_driven_strip(tmp_path):
    """The other half of R8, and the more important one: a block the provider
    REFUSED is poisoned on any provider. Only the size-caused strip is
    provider-scoped; lifting the refusal-caused one would re-send the block
    that bricked the session in the first place."""

    class RefusesImages:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, request: ChatRequest, signal: AbortSignal | None):
            self.calls += 1
            first = self.calls == 1

            async def gen():
                if first:
                    raise ProviderError(400, "Could not process image")
                yield StreamTextDelta(delta="ok")
                yield StreamEndEvent(stop_reason="endTurn")

            return gen()

    session = make_session(tmp_path, RefusesImages())
    await session.seed_history(_frames(2))

    await session.prompt("go")
    assert session._images_rejected
    assert not session._images_rejected_for_size, "a block refusal was marked as size-caused"

    session.set_model(ModelSpec(provider="other", model_id="lax", context_window=1_000_000))

    assert session._images_rejected, "a poisoned-block strip was lifted by a model switch"
    await session.dispose()


@pytest.mark.asyncio
async def test_the_terminal_notice_does_not_claim_screenshots_on_a_text_history(tmp_path):
    """R9: this rung is reached on the FIRST refusal for a text-only history,
    where "dropping screenshots" describes something that never happened."""
    stream = RefusesOversizeRequests(cap=1_000)
    session = make_session(tmp_path, stream)
    await session.seed_history([Message.user("T" * 5_000_000)])

    notices: list[str] = []
    session.subscribe(lambda e: notices.append(e.text) if isinstance(e, NoticeEvent) else None)

    await session.prompt("go")

    assert session._images_rejected
    terminal = [n for n in notices if "/compact" in n]
    assert terminal, "no terminal notice was emitted"
    assert "screenshot" not in terminal[0].lower(), terminal[0]
    assert "images have been removed" not in terminal[0].lower(), terminal[0]
    await session.dispose()


# ---------------------------------------------------------------------------
# A surrogate on disk must not wedge the session (agent review round 3, R10)
# ---------------------------------------------------------------------------

#: Legal in a Python ``str`` and legal JSON, so it round-trips into the
#: transcript and is replayed verbatim on every resume.
LONE_SURROGATE = "bad \ud800 here"


@pytest.mark.asyncio
async def test_a_surrogate_in_history_does_not_kill_the_render(tmp_path):
    """R10: the sizer runs inside ``_render_history``, so raising there takes
    out every wire path including ``/compact`` — the escape hatch.

    Fails on 2b15c340 with UnicodeEncodeError.
    """
    stream = ScriptedOk()
    session = make_session(tmp_path, stream)
    call = Message.assistant("x")
    call.tool_calls = [ToolCall(id="1", name="write", arguments={"t": LONE_SURROGATE})]
    await session.seed_history([Message.user(f"look {LONE_SURROGATE}"), call])

    rendered = session._render_history(list(session._context.messages))

    assert len(rendered) == 2
    await session.prompt("hello")
    assert stream.requests, "the turn never reached the provider"
    await session.dispose()


@pytest.mark.asyncio
async def test_a_surrogate_persisted_to_disk_survives_a_resume(tmp_path):
    """The wedge shape specifically: the codepoint is IN the transcript, so a
    resumed session replays it on every turn forever.

    This is the case that made R10 a blocker rather than a crash — recovery
    from inside was impossible, exactly like the 413 this PR exists to fix.
    """
    directory = tmp_path / "sess"
    first = ScriptedOk()
    session = make_session(tmp_path, first)
    call = Message.assistant("x")
    call.tool_calls = [ToolCall(id="1", name="write", arguments={"t": LONE_SURROGATE})]
    await session.seed_history([Message.user("hi"), call])
    await session.prompt("one")
    await session.dispose()

    # The surrogate really is on disk, unescaped by the JSON round trip.
    assert "ud800" in (directory / "transcript.jsonl").read_text(encoding="utf-8")

    # A cold relaunch from that directory — what `--resume` does.
    second = ScriptedOk()
    resumed = Session(
        model=MODEL,
        stream_fn=second,
        tools=[],
        transcript=Transcript(directory),
        system_blocks_provider=lambda: ["stable"],
    )
    assert resumed._render_history(list(resumed._context.messages))
    await resumed.prompt("after resume")
    assert second.requests, "the resumed session could not send"
    await resumed.dispose()


# ---------------------------------------------------------------------------
# A genuine refusal downgrades the strip's scope (QA round 3, Q6)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_refusal_after_a_size_strip_makes_the_strip_permanent(tmp_path):
    """Q6: the size strip is provider-scoped, a refused BLOCK is not.

    When a real refusal arrives while a size strip is already in force, the
    strip stops being provider-scoped — otherwise the next ``/model`` switch
    lifts it and re-admits a block the provider actually refused, which is the
    one thing R8's own comment says must never happen.
    """
    stream = RefusesOversizeRequests(cap=1_000)
    session = make_session(tmp_path, stream)
    await session.seed_history(_frames(6))

    for turn in range(12):
        await session.prompt(f"t{turn}")
        if session._images_rejected:
            break
    assert session._images_rejected_for_size, "the size strip never fired"

    # A genuine image refusal now supersedes it.
    await session._degrade_if_image_rejected(ProviderError(400, "Could not process image"))
    assert session._images_rejected
    assert not session._images_rejected_for_size, "the strip is still marked provider-scoped"

    session.set_model(ModelSpec(provider="other", model_id="lax", context_window=1_000_000))

    assert session._images_rejected, "a refused block was re-admitted by a model switch"
    rendered = session._render_history(list(session._context.messages))
    assert not any(
        isinstance(block, ImageContent) for message in rendered for block in message.content
    )
    await session.dispose()
