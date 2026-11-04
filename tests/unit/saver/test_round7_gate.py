"""QA round 7: no ruler means no upload, and opt_in says why it is off.

Q-R7-2: with tiktoken installed but cl100k unable to load, a `find_spec` gate
said yes, so every due prefetch uploaded and was charged before the per-epoch
ruler refused it — and the user was told the response "failed its safety
checks". Q-R7-4: `opt_in` while signed out said nothing at all.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.harness.types import ChatRequest, Message, ModelSpec, TextContent


def test_the_gate_is_the_loaded_encoding_not_the_package(monkeypatch: pytest.MonkeyPatch) -> None:
    from local_operator.compaction import tokens
    from local_operator.saver.provenance import estimator_available

    monkeypatch.setattr(tokens, "_get_encoding", lambda: None)
    assert estimator_available() is False


@pytest.mark.asyncio
async def test_no_encoding_uploads_nothing_and_says_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_operator.compaction import tokens
    from local_operator.saver import chain as chain_mod

    monkeypatch.setattr(tokens, "_get_encoding", lambda: None)
    notices: list[Any] = []

    async def notice(message: Any) -> None:
        notices.append(message)

    class Client:
        calls = 0

        async def prepare(self, *args: Any, **kwargs: Any) -> Any:
            Client.calls += 1
            raise AssertionError("uploaded without a ruler")

    # The Client stands in for a SaverClient: this test is about the gate ahead
    # of the upload, so it only has to make an upload observable.
    chain = chain_mod.SaverChain(
        client=Client(),  # type: ignore[arg-type]
        session_id="s",
        notice=notice,
    )
    request = ChatRequest(
        model=ModelSpec(provider="anthropic", model_id="m"),
        messages=[Message(role="user", content=[TextContent(text="hello")])],
    )
    assert await chain.apply(request) is request
    assert Client.calls == 0
    assert [m.code for m in notices] == [chain_mod.notice_message("unavailable").code]
