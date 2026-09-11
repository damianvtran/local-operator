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
from local_operator.compaction.marker import build_compaction_marker
from local_operator.harness.comms import SubagentComms
from local_operator.harness.types import (
    AgentMessage,
    CustomMessage,
    Message,
    ModelSpec,
    StreamEndEvent,
)
from local_operator.session.session import Session, _is_persistable_message
from local_operator.session.transcript import Transcript

MODEL = ModelSpec(provider="test", model_id="m", context_window=100_000)

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
        and (message.provider_payload or {}).get("harness_injected")
    ]


async def wait_for(predicate, attempts: int = 200) -> None:
    """Let the session's journalling settle — those producers are fire-and-forget."""
    import asyncio

    for _ in range(attempts):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition never became true")


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
