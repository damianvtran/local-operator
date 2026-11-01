"""Ephemeral first-value routing must not cross ownership or durable boundaries."""

from local_operator.harness.types import ChatRequest, ModelSpec
from local_operator.providers.turn_state import CodexTurnState


def test_first_value_wins_and_stale_response_cannot_repopulate() -> None:
    state = CodexTurnState()
    state.observe_attempt("openai", "account-a")
    generation, token = state.bind_endpoint("oauth-endpoint")
    assert token is None
    state.receive(generation, "first")
    state.receive(generation, "replacement")
    assert state.bind_endpoint("oauth-endpoint")[1] == "first"
    state.observe_attempt("", "")
    state.observe_attempt("openai", "account-a")
    next_generation, token = state.bind_endpoint("oauth-endpoint")
    assert token is None
    state.receive(generation, "late-old-account")
    assert state.bind_endpoint("oauth-endpoint")[1] is None
    state.receive(next_generation, "new-first")
    assert state.bind_endpoint("oauth-endpoint")[1] == "new-first"


def test_owner_endpoint_model_and_turn_resets() -> None:
    state = CodexTurnState()
    for owner, endpoint in [
        ("a:model-1", "endpoint-1"), ("b:model-1", "endpoint-1"),
        ("a:model-1", "endpoint-1"), ("a:model-1", "endpoint-2"),
        ("a:model-2", "endpoint-2"),
    ]:
        state.observe_attempt("openai", owner)
        generation, token = state.bind_endpoint(endpoint)
        assert token is None
        state.receive(generation, "bounded-synthetic")
    assert CodexTurnState().bind_endpoint("endpoint-2")[1] is None


def test_invalid_header_is_rejected_not_truncated() -> None:
    for value in (None, "", "has space", "a\rb", "a\nb", "a\x00b", "é", "a" * 4097):
        state = CodexTurnState()
        generation, _ = state.bind_endpoint("endpoint")
        state.receive(generation, value)
        assert state.bind_endpoint("endpoint")[1] is None
    state = CodexTurnState()
    generation, _ = state.bind_endpoint("endpoint")
    state.receive(generation, "a" * 4096)
    token = state.bind_endpoint("endpoint")[1]
    assert token is not None
    assert len(token) == 4096


def test_private_attachment_clone_and_public_serialization() -> None:
    request = ChatRequest(model=ModelSpec(provider="openai", model_id="synthetic"))
    before = request.model_dump_json()
    clone = request.model_copy()
    clone._codex_turn_state = CodexTurnState()
    generation, _ = clone._codex_turn_state.bind_endpoint("endpoint")
    clone._codex_turn_state.receive(generation, "never-persist-this-header")
    assert request._codex_turn_state is None
    assert clone.model_copy()._codex_turn_state is clone._codex_turn_state
    assert clone.with_model(clone.model)._codex_turn_state is clone._codex_turn_state
    assert clone.model_dump_json() == before
    assert "never-persist-this-header" not in repr(clone)
    assert "never-persist-this-header" not in repr(clone._codex_turn_state)
