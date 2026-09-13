"""A live-context-only injection must not be BAKED into the transcript.

WHY THIS FILE EXISTS
--------------------
A TRANSIENT (failover) model-switch notice is deliberate live-context-only
state: ``journal_model_switch(..., transient=True)`` appends a ``CustomMessage``
so the running model knows it is on a fallback, and ``_is_persistable_message``
refuses to persist it because a fallback that outlives the process is stale by
definition.

But ``_run_compaction`` rebuilds the live context from the RENDERED history,
and the default renderer turns a ``session_model_switch`` into a plain
``Message(role="user")`` stamped ``harness_injected``. Once rendered, the row is
indistinguishable from an operator prompt, and the turn-end persist pass writes
every plain ``Message`` — so the notice the user never typed landed in the
transcript as a genuine user row (four consecutive ones in session
``835fbcafdc27``) and every front end painted it as the user's own words.

``_render_for_compaction`` already excluded two hand-listed types (a todo
reminder and a credential record) for exactly this reason. The fix replaces the
enumeration with the structural rule — a ``CustomMessage`` the transcript does
not hold is one a resume cannot replay — and these tests pin both halves of
that rule:

* the leak is gone at the seam (render → rebuild → persist) AND through a real
  compaction pass, and
* the trap is closed: every record whose PRODUCER persisted it is still in the
  rebuild, because dropping one would make the live context diverge from the
  resume that replays it. ``Transcript.has_entry`` is the test;
  ``_is_persistable_message`` is NOT, and using it would take hub/peer/wake
  deliveries out of the rebuild.
"""

from __future__ import annotations

import pytest

from local_operator.compaction.api import CompactionSettings
from local_operator.compaction.cutpoint import RENDERED_INJECTION_KEY
from local_operator.compaction.marker import build_compaction_marker
from local_operator.harness.comms import SubagentComms
from local_operator.harness.rows import is_harness_notice_row
from local_operator.harness.types import (
    AgentMessage,
    CustomMessage,
    Message,
    ModelSpec,
    StreamEndEvent,
    TextContent,
)
from local_operator.mobile.projection import fold_messages_to_entries
from local_operator.session.peer import PEER_MESSAGE_MESSAGE_TYPE
from local_operator.session.session import Session, _is_persistable_message
from local_operator.session.transcript import Transcript

MODEL = ModelSpec(provider="test", model_id="m", context_window=100_000)

#: The legacy shape QA measured on the operator's own session: a switch notice
#: written before the ``harness_injected`` stamp existed — a plain user row with
#: NO ``provider_payload`` at all, which a later pass harvested as a user turn.
LEGACY_NOTICE = (
    "[model switch] You are now running as zai/glm-5.3 (was anthropic/claude-opus-5).\n"
    "Reason: provider failure\n"
    "This is a temporary fallback for the current request; the session may return to its "
    "primary model at a later turn. Capabilities and context window may differ from the "
    "primary."
)

#: Small enough that a few short turns leave history outside the kept window,
#: so a real pass has something to summarize and the notice — appended last —
#: lands INSIDE the kept window, which is the only way it could be baked in.
KEEP_RECENT = 40


class ScriptedStream:
    """Replays one script per call and records the requests it saw."""

    def __init__(self, scripts: list[list[object]] | None = None) -> None:
        self.scripts = list(scripts or [])
        self.requests: list[object] = []

    def __call__(self, request, signal):
        self.requests.append(request)
        script = self.scripts.pop(0) if self.scripts else [StreamEndEvent(stop_reason="stop")]

        async def gen():
            for event in script:
                yield event

        return gen()


def make_session(
    tmp_path, stream, *, session_id: str = "sess", keep_recent: int = KEEP_RECENT
) -> Session:
    return Session(
        model=MODEL,
        stream_fn=stream,
        tools=[],
        transcript=Transcript(tmp_path / session_id),
        session_id=session_id,
        system_blocks_provider=lambda: ["stable"],
        compaction_settings=CompactionSettings(keep_recent_tokens=keep_recent),
    )


def rendered_texts(messages: list[Message]) -> list[str]:
    return [getattr(message, "text", "") or "" for message in messages]


def leaked_user_rows(entries_or_messages) -> list[Message]:
    """Every ``role="user"`` row the harness minted, wherever it was read from."""
    return [
        message
        for message in entries_or_messages
        if isinstance(message, Message)
        and message.role == "user"
        and (message.provider_payload or {}).get(RENDERED_INJECTION_KEY)
    ]


@pytest.mark.asyncio
async def test_a_transient_switch_notice_never_reaches_the_transcript(tmp_path) -> None:
    """The regression, at the seam ``_run_compaction`` drives.

    Fails on the pre-fix tree with one leaked ``role="user"`` row carrying the
    switch notice's text — the rendered copy of a message the transcript was
    deliberately never given.
    """
    session = make_session(tmp_path, ScriptedStream())
    await session.journal_model_switch(
        "zai/glm-5.3",
        "anthropic/claude-opus-5",
        reason="anthropic quota exhausted (0% remaining)",
        transient=True,
    )
    # Live-only BY DESIGN: nothing was journalled, and the live context holds
    # the record as a CustomMessage.
    assert session._transcript.entries() == []
    assert any(isinstance(m, CustomMessage) for m in session._context.messages)

    # The REQUEST render still carries it — the notice is not inert, it is how
    # the running model learns it is on a fallback (the half that must survive).
    request_render = "\n".join(rendered_texts(session._render_history(session._context.messages)))
    assert "[model switch] You are now running as zai/glm-5.3" in request_render

    # The COMPACTION render does not: this is the fix, and it is what makes the
    # rebuilt context equal what a resume can replay.
    assert not any(
        "[model switch]" in text for text in rendered_texts(session._render_for_compaction())
    )

    # What ``_run_compaction`` then does: rebuild from that render, and let the
    # turn-end pass persist whatever is in the context.
    session._context.messages = list(session._render_for_compaction())
    await session._persist_new_messages(session._context.messages)

    assert session._transcript.entries() == []
    # A session constructed over that transcript (what ``--resume`` does)
    # renders no injected row either — the other half of the parity claim.
    resumed = make_session(tmp_path, ScriptedStream())
    assert leaked_user_rows(resumed._render_history(resumed._context.messages)) == []
    await resumed.dispose()
    await session.dispose()


@pytest.mark.asyncio
async def test_a_real_compaction_pass_leaves_no_injected_user_row(tmp_path) -> None:
    """The same claim through ``compact_now`` — the assembled pass, not the seam.

    A failover lands between turns, the context fills past the threshold, and
    the pass that reclaims it must not carry the notice forward into the kept
    window it persists.
    """
    stream = ScriptedStream()
    session = make_session(tmp_path, stream)
    for index in range(3):
        await session.prompt(f"question {index} " + "detail " * 30)

    await session.journal_model_switch(
        "kimi/k3", "anthropic/claude-opus-5", reason="provider failure", transient=True
    )
    outcome = await session.compact_now()
    assert outcome.ran is True, outcome.reason
    # The pass rebuilds the context; the ROW reaches the transcript at the next
    # turn end, when the persist pass writes every plain Message it finds. Both
    # steps are needed to reproduce the report, so both are exercised here.
    await session.prompt("after the pass " + "detail " * 30)

    # The kept window was rebuilt from the render, so the notice is gone from
    # the context AND from the transcript, and a resume agrees.
    assert not any(
        isinstance(m, CustomMessage) and m.custom_type == "session_model_switch"
        for m in session._context.messages
    )
    # The rendered copy is what leaked: a plain user Message carrying the
    # notice's text, which is what the persist pass wrote pre-fix.
    assert not any(
        not isinstance(m, CustomMessage) and "[model switch]" in (getattr(m, "text", "") or "")
        for m in session._context.messages
    )
    assert leaked_user_rows(session._transcript.build_llm_history()) == []
    resumed = make_session(tmp_path, ScriptedStream())
    assert leaked_user_rows(resumed._render_history(resumed._context.messages)) == []
    await resumed.dispose()
    await session.dispose()


@pytest.mark.asyncio
async def test_the_rebuild_keeps_every_record_the_transcript_holds(tmp_path) -> None:
    """The trap: a wrong predicate breaks live/resume equivalence.

    ``_is_persistable_message`` is the tempting test and the wrong one — it
    answers "may the turn-end flush write this?" and returns False for
    ``hub_message``/``peer_message``/``wake_prompt``, whose PRODUCERS persist
    them. Dropping those from the rebuild would leave the live context holding
    less than the resume that replays the same transcript, so this test asserts
    BOTH halves at once: the deliberate switch, the incident, the hub delivery
    and the peer delivery are all still rendered, while a transient notice
    appended beside them is not.
    """
    session = make_session(tmp_path, ScriptedStream())
    await session.journal_model_switch("zai/glm-5.3", "anthropic/claude-opus-5")
    await session.journal_incident("429 rate limit from provider")
    # Built by the real producers, then persisted and appended in the order
    # every custom producer uses (durable first, then live context).
    hub = SubagentComms(session)._to_child_message("focus on the parser", expects_reply=False)
    peer = session._peer_custom_message(
        "rebasing now", {"pid": 4242, "conversation_name": "peer", "model_label": "test/m"}
    )
    for message in (hub, peer):
        await session._transcript.append_message(message)
        session._context.messages.append(message)
    await session.journal_model_switch("kimi/k3", "zai/glm-5.3", transient=True)

    # The premise of the trap: the tempting predicate would drop these two.
    assert _is_persistable_message(hub) is False
    assert _is_persistable_message(peer) is False

    rendered = "\n".join(rendered_texts(session._render_for_compaction()))
    assert "[model switch] You are now running as zai/glm-5.3" in rendered
    assert "429 rate limit from provider" in rendered
    assert "focus on the parser" in rendered
    assert "rebasing now" in rendered
    assert "[model switch] You are now running as kimi/k3" not in rendered

    # …and the rebuilt context is what a resume replays, row for row. The
    # resumed session renders the SAME transcript, so its render is the other
    # side of the equivalence — comparing the raw replay would compare
    # ``CustomMessage``s against rendered ``Message``s.
    session._context.messages = list(session._render_for_compaction())
    live = rendered_texts(session._render_for_compaction())
    resumed = make_session(tmp_path, ScriptedStream())
    assert rendered_texts(resumed._render_for_compaction()) == live
    await resumed.dispose()
    await session.dispose()


@pytest.mark.asyncio
async def test_the_compaction_marker_is_not_an_ephemeral_injection(tmp_path) -> None:
    """The marker's content IS persisted — as a compaction entry, not a message.

    ``append_compaction`` stores it under its own entry type, so the fresh id
    ``build_compaction_marker`` mints is in no entry set while a resume still
    replays the summary from that payload. Without the type exemption the
    structural rule would drop it, leaving every later pass planning and
    pricing a context that has lost its own summary.
    """
    session = make_session(tmp_path, ScriptedStream())
    marker = build_compaction_marker("summary of the older history")
    assert session._transcript.has_entry(marker.id) is False
    session._context.messages.append(marker)

    rendered = "\n".join(rendered_texts(session._render_for_compaction()))
    assert "<previous-context-summary>" in rendered
    assert "summary of the older history" in rendered
    await session.dispose()


@pytest.mark.asyncio
async def test_the_predicate_is_id_based_not_type_based(tmp_path) -> None:
    """One live CustomMessage of a PERSISTED type is kept; an unpersisted one is not.

    The discriminator is the transcript's id set, so a record of a
    ``_PERSISTABLE_CUSTOM_TYPES`` member that reached the live context without
    its durable append (an aside still parked, a producer that failed to write)
    is filtered like any other live-only injection — the rule does not trust a
    type name to prove a write happened.
    """
    session = make_session(tmp_path, ScriptedStream())
    unpersisted: list[AgentMessage] = [
        CustomMessage(
            custom_type="session_state",
            attribution="system",
            details={"text": "[session-state]\n## Environment\nnever written"},
        )
    ]
    session._context.messages.extend(unpersisted)

    assert rendered_texts(session._render_for_compaction()) == []
    await session.dispose()


@pytest.mark.asyncio
async def test_a_delivery_in_the_kept_window_keeps_its_receipt_across_a_pass(tmp_path) -> None:
    """M1: the commit must re-seat the SOURCE, not an anonymous stamped copy.

    The rebuild commits the RENDERED history, and a rendered delivery is a plain
    stamped user message. Without ``_restore_custom_sources`` a peer delivery in
    the kept window would come back as an injection-shaped row — which the folds
    hide, because a stamped row is never the operator's words — while a RESUME of
    the same session replays the persisted custom entry and paints the peer
    receipt. The receipt would vanish from the live session and reappear on the
    next restart, which is the opposite of live/replay parity.
    """
    session = make_session(tmp_path, ScriptedStream())
    for index in range(3):
        await session.prompt(f"question {index} " + "detail " * 30)
    # The real inbound path: persisted durably AND appended to the live context.
    await session.receive_peer_message(
        "rebasing now — expect a redirect",
        mode="mailbox",
        wake=False,
        sender={"pid": 4242, "conversation_name": "peer", "model_label": "test/m"},
    )
    delivered = [
        message
        for message in session._context.messages
        if isinstance(message, CustomMessage) and message.custom_type == PEER_MESSAGE_MESSAGE_TYPE
    ]
    assert delivered, "the peer delivery never reached the live context"
    assert session._transcript.has_entry(delivered[-1].id)

    outcome = await session.compact_now()
    assert outcome.ran is True, outcome.reason

    # Identity survived the pass…
    survivors = [
        message
        for message in session._context.messages
        if isinstance(message, CustomMessage) and message.custom_type == PEER_MESSAGE_MESSAGE_TYPE
    ]
    assert [message.id for message in survivors] == [delivered[-1].id]
    # …so nothing in the kept window is an anonymous stamped row, and the row
    # the fold sees is therefore one it PAINTS (a receipt, not a suppressed
    # injection).
    assert leaked_user_rows(session._context.messages) == []
    assert not any(is_harness_notice_row(message) for message in session._context.messages)

    # Live equals resume, on the request render AND row for row through the
    # phone fold — the two claims a receipt depends on.
    live_texts = rendered_texts(session._render_for_compaction())
    resumed = make_session(tmp_path, ScriptedStream())
    assert rendered_texts(resumed._render_for_compaction()) == live_texts
    live_rows = [
        (row.kind, row.text) for row in fold_messages_to_entries(list(session._context.messages))
    ]
    resume_rows = [
        (row.kind, row.text) for row in fold_messages_to_entries(list(resumed._context.messages))
    ]
    assert live_rows == resume_rows
    assert any("rebasing now — expect a redirect" in text for _, text in live_rows)
    await resumed.dispose()
    await session.dispose()


@pytest.mark.asyncio
async def test_a_legacy_notice_is_not_harvested_into_the_marker(tmp_path) -> None:
    """Q1: an unstamped notice row must not be lifted as a user turn.

    The harvest refuses stamped rows already; this is the legacy shape it could
    not see (the operator's session carries eight switch notices with no
    ``provider_payload`` at all), and lifting one re-seats the harness's words
    as a user row on every replay of the marker.
    """
    session = make_session(tmp_path, ScriptedStream())
    legacy = Message(role="user", content=[TextContent(text=LEGACY_NOTICE)])
    session._context.messages.append(legacy)
    await session._transcript.append_message(legacy)
    for index in range(3):
        await session.prompt(f"question {index} " + "detail " * 30)

    outcome = await session.compact_now()
    assert outcome.ran is True, outcome.reason

    marker = session._transcript.latest_entry("compaction")
    assert marker is not None
    stored = marker.payload.get("preserved_user_turns") or []
    assert not [
        turn for turn in stored if str(turn.get("text", "")).startswith("[model switch]")
    ], stored
    await session.dispose()


@pytest.mark.asyncio
async def test_a_marker_carrying_a_legacy_notice_replays_without_it(tmp_path) -> None:
    """Q1: the heal for a block an older build already wrote, end to end.

    A real session's latest marker holds eight notice copies from before the
    stamp existed. Replay must not re-seat them (they are the harness's words,
    not the operator's), must still re-seat the operator's own preserved turn
    verbatim, and must account the shed copies as INJECTIONS — the elision
    notice's "not authored by the user" bucket — rather than as turns the
    operator wrote and lost.
    """
    directory = tmp_path / "sess"
    transcript = Transcript(directory)
    rows = [Message.user(f"row {index}") for index in range(4)]
    await transcript.append_messages(rows)
    await transcript.append_compaction(
        "summary",
        rows[-1].id,
        500,
        preserved_user_turns=[
            {"id": "legacy-notice-1", "text": LEGACY_NOTICE},
            {"id": "legacy-notice-2", "text": LEGACY_NOTICE},
            {"id": rows[0].id, "text": "the operator's own constraint"},
        ],
        preserved_turns_cap=100_000,
    )

    replayed = [
        message
        for message in Transcript(directory).build_llm_history()
        if isinstance(message, Message)
    ]
    texts = [message.text for message in replayed]
    assert "[model switch]" not in "\n".join(texts)
    assert "the operator's own constraint" in texts
    assert not leaked_user_rows(replayed)
    # The shed copies are reported as injections, and the genuine turn was not
    # miscounted as one.
    elision = "\n".join(texts)
    assert "2 harness-injected message(s)" in elision
    assert "older user message(s) you wrote were dropped" not in elision


def test_a_plain_stored_notice_is_a_notice_row_whatever_its_provenance() -> None:
    """QA round 2 Q1: the notice rule is NOT scoped to carried copies.

    The audit phase of the attached viewer replays STORED rows, and a stored row
    from before the stamp existed has no ``provider_payload`` at all — so a rule
    that needed a carried marker let the four plain switch notices on the
    operator's session come back into view the moment the carried copies were
    shed. Text is the test for any user row now; the cost (a pasted notice loses
    its DISPLAY row, and nothing else) is pinned by the fold tests.
    """
    plain = Message(role="user", content=[TextContent(text=LEGACY_NOTICE)])

    assert is_harness_notice_row(plain)
    assert plain.provider_payload is None, "the fixture must be the unprovenanced shape"
    assert not is_harness_notice_row(Message.user("why did the model change?"))
    # A DELIVERY envelope is not a notice: it has its own parser, and a person
    # quoting one keeps their words (pinned in the fold tests).
    assert not is_harness_notice_row(Message.user("<parent-message>\nwhy does my log show this?"))
