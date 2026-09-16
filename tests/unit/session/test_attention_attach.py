"""What an acknowledgement ANSWERS on the transport a follower runs on.

Agent review round 1 (R4) read the attached path as one on which the TUI's
post-acknowledgement verification could only ever see a pre-ack snapshot: the
owner publishes its projection asynchronously (``ServingSessionHandle.
acknowledge_attention`` sets the projection, ``Session.refresh_frontend_state``
pushes it, the runtime writes the frame), while ``AttachedSession.
acknowledge_attention`` answers out of the state the follower had already
applied. If that were true, an honest receipt would read as a lost one and the
verification would accuse the wrong party.

So this module MEASURES the transport rather than arguing about it: a real
``Session``, the real ``ServingSessionHandle``, a real ``RuntimeServer`` and a
real ``AttachedSession`` over a loopback socket, with the completion published
into an ISOLATED attention store. What it pins is the property the poll's
inconclusive branch rests on. If the answer ever IS stale here, this test says
so in bytes -- and the poll's job becomes to stay inconclusive rather than to
claim a verdict, which is the other half of R4 and is pinned in
``tests/unit/tui/test_attention.py``.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import pytest

from local_operator.harness.types import Message, TextContent
from local_operator.session.attached import AttachedSession
from local_operator.session.attention import AttentionStore, conversation_identity
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.runtime.serving import ServingSessionHandle
from tests.e2e.harness import ScriptedStream, build_session, seed_transcript, text_turn
from tests.unit.session.test_remote import _never_take_over


@pytest.mark.asyncio
async def test_a_followers_acknowledgement_answers_with_the_state_its_owner_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The receipt handshake over a real socket, in the shape the TUI reads it.

    The follower must be able to tell "the owner read it" from "the answer says
    nothing yet", and the only way to know which one this transport delivers is
    to run it: hence a real owner runtime and a real attach, not a fake whose
    ``refresh_attention`` returns the store state synchronously (which is
    structurally unable to see a push lag).
    """
    config = tmp_path / "config"
    config.mkdir()
    (config / "config.yml").write_text(
        "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n"
    )
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    monkeypatch.setenv("HOME", str(tmp_path))
    directory = config / "sessions" / "synthetic-ack"
    await seed_transcript(directory, [])
    session = build_session(directory, ScriptedStream([text_turn("unused")]), cwd=tmp_path)

    identity = conversation_identity(directory)
    result = Message(role="assistant", content=[TextContent(text="The finished result.")])
    store = AttentionStore(config / "attention.db")
    token = str(uuid.uuid4())
    store.publish(identity, token, result.id, "complete")
    # The owner has to KNOW about it before anyone attaches: the follower's seed
    # is the owner's projection, and a seed that never carried the completion
    # would make the assertion below pass for the wrong reason.
    assert (await session.refresh_attention())["completion_token"] == token

    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    server = RuntimeServer(handle, kind="daemon")
    await server.start_in_process()
    follower = await AttachedSession.connect(
        server._record,
        directory.name,
        config_dir=config,
        takeover_factory=_never_take_over,
        display_window=True,
    )
    try:
        seeded = follower.frontend_state.attention
        assert seeded.get("completion_token") == token, "the follower never saw the completion"
        assert seeded.get("unseen") is True, "the seed already read; nothing left to prove"

        answered = await follower.acknowledge_attention(token)

        assert answered.get("unseen") is False, (
            "the follower's own answer still reads unseen: on this transport the poll's "
            "verification can only be inconclusive, and must not report a lost receipt"
        )
        assert store.state(identity)["unseen"] is False, "the receipt did not advance"
        # The projection itself catches up ASYNCHRONOUSLY -- that lag is the whole
        # reason the answer above has to carry the state -- and it is what the
        # next poll reads, so it must converge rather than stay behind forever.
        for _ in range(100):
            if follower.frontend_state.attention.get("unseen") is False:
                break
            await asyncio.sleep(0.02)
        assert (
            follower.frontend_state.attention.get("unseen") is False
        ), "the follower's projection never caught up; the next poll would re-send the receipt"
    finally:
        await follower.dispose()
        server.close()
        await handle.dispose()
