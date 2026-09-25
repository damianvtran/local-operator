"""Receive side of ``peer_set_model``: the shared core and the serving/exec handle.

The TUI handle's half (``/model`` on the Textual thread, then the read-back) is
pinned against a real ``OperatorApp`` in ``tests/unit/tui/test_peer_model_tui.py``.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.mobile import peer_model
from local_operator.model.configure import ModelSelectionRefused


class _Store:
    """An AuthStore stand-in that records whether it was closed."""

    instances: list["_Store"] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.closed = False
        _Store.instances.append(self)

    def list_credentials(self, provider: Any = None) -> list[Any]:
        return []

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_store(monkeypatch):
    import local_operator.providers.auth_store as auth_store

    _Store.instances = []
    monkeypatch.setattr(auth_store, "AuthStore", _Store)
    return _Store


def test_the_credential_store_is_closed_when_the_provider_is_usable(fake_store, monkeypatch):
    """D3's resolution: a short-lived store per call, closed on every path."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-not-real")
    assert peer_model.provider_usable_here("deepseek") is True
    assert [store.closed for store in fake_store.instances] == [True]


def test_the_credential_store_is_closed_when_the_switch_is_refused(fake_store, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ModelSelectionRefused) as caught:
        peer_model.validate_peer_selection("deepseek", "deepseek-flash")
    assert caught.value.code == "provider_unusable"
    assert [store.closed for store in fake_store.instances] == [True]


def test_the_credential_store_is_closed_when_the_read_raises(fake_store, monkeypatch):
    def locked(self, provider=None):  # noqa: ANN001
        raise RuntimeError("database is locked")

    monkeypatch.setattr(_Store, "list_credentials", locked)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ModelSelectionRefused) as caught:
        peer_model.validate_peer_selection("deepseek", "deepseek-flash")
    # "cannot confirm" is its own refusal, never "no credential".
    assert caught.value.code == "credentials_unreadable"
    assert [store.closed for store in fake_store.instances] == [True]


def test_an_unknown_pair_never_opens_the_store(fake_store):
    with pytest.raises(ModelSelectionRefused):
        peer_model.validate_peer_selection("deepseek", "not-a-model")
    assert fake_store.instances == []


def test_a_keyless_provider_is_usable_with_no_credential(fake_store):
    assert peer_model.provider_usable_here("test") is True


@pytest.mark.parametrize(
    "busy,calling,children,expected",
    [
        (False, True, 0, ["switched to b/y (was a/x)", "its next turn runs on it"]),
        (
            True,
            True,
            0,
            [
                "switched to b/y (was a/x)",
                "mid-turn: the call in flight finishes on the old model; later calls use the "
                "new one",
            ],
        ),
        (
            True,
            False,
            0,
            [
                "switched to b/y (was a/x)",
                "mid-turn: the current step finishes on the old model; later calls use the "
                "new one",
            ],
        ),
        (
            False,
            True,
            1,
            [
                "switched to b/y (was a/x)",
                "its next turn runs on it",
                "1 running subagent keeps its model; new and resumed ones use the new one",
            ],
        ),
        (
            False,
            True,
            3,
            [
                "switched to b/y (was a/x)",
                "its next turn runs on it",
                "3 running subagents keep their model; new and resumed ones use the new one",
            ],
        ),
    ],
)
def test_the_switch_receipts_are_outcome_first_short_lines(busy, calling, children, expected):
    detail = peer_model.switched_detail(
        "a/x", "b/y", busy=busy, calling=calling, running_subagents=children
    )
    assert detail.splitlines() == expected


def test_every_receipt_line_fits_an_expanded_card_at_100_columns() -> None:
    """D1: the card clips each body line; with realistic model ids every line of
    the busiest receipt stays inside the ~94-cell body at 100 columns."""
    old, new = "anthropic/claude-opus-5", "deepseek/deepseek-flash"
    detail = peer_model.switched_detail(old, new, busy=True, calling=True, running_subagents=12)
    assert max(len(line) for line in detail.splitlines()) <= 94, detail


def test_the_other_receipts_lead_with_a_distinct_outcome() -> None:
    assert peer_model.already_on_detail("a/x").splitlines() == ["already on a/x", "nothing changed"]
    assert peer_model.accepted_detail("a/x").startswith("pending: switch to a/x accepted\n")
    assert (
        peer_model.refusal_detail("'q' is not a known provider.", "a/x")
        == "refused: 'q' is not a known provider; still on a/x"
    )


def test_the_audit_card_leads_with_the_new_model_and_names_the_sender() -> None:
    """D3/U3: on resume the card is the only trace, and it is clipped from the right."""
    assert peer_model.audit_body("a/x", "b/y", {"conversation_name": "fleet boss"}) == (
        "[remote model switch] now on b/y (was a/x) — switched by fleet boss"
    )
    assert peer_model.audit_body("a/x", "b/y", {"pid": 7}) == (
        "[remote model switch] now on b/y (was a/x) — switched by pid 7"
    )
    assert peer_model.audit_body("a/x", "b/y") == "[remote model switch] now on b/y (was a/x)"


def test_a_call_in_flight_is_told_apart_from_an_open_tool_batch() -> None:
    """U7: an assistant tail with unanswered tool calls is a tool step, not a call."""
    from types import SimpleNamespace

    from local_operator.harness.types import Message, TextContent, ToolCall

    user = Message(role="user", content=[TextContent(text="go")])
    call = Message(role="assistant", tool_calls=[ToolCall(id="c1", name="bash")])
    waiting_on_tool = SimpleNamespace(_context=SimpleNamespace(messages=[user, call]))
    assert peer_model.provider_call_in_flight(waiting_on_tool) is False
    # A turn whose tail is the user's own message is waiting on the provider.
    waiting_on_model = SimpleNamespace(_context=SimpleNamespace(messages=[user]))
    assert peer_model.provider_call_in_flight(waiting_on_model) is True
    assert peer_model.provider_call_in_flight(SimpleNamespace()) is True


# ---------------------------------------------------------------------------
# ServingSessionHandle.receive_peer_model
# ---------------------------------------------------------------------------


def _serving(monkeypatch):
    from tests.unit.session.runtime.test_serving import make_handle

    handle, session = make_handle()
    session.model_label = session.effective_model_label = "test/mock"
    applied: list[Any] = []

    def set_model(spec, explicit=False):  # noqa: ANN001
        applied.append((spec, explicit))
        session.model_label = session.effective_model_label = f"{spec.provider}/{spec.model_id}"

    monkeypatch.setattr(session, "set_model", set_model, raising=False)
    monkeypatch.setattr(handle, "_refresh_state", lambda: None)
    monkeypatch.setattr(peer_model, "provider_usable_here", lambda _p: True)
    cards: list[tuple[str, dict[str, Any]]] = []

    async def receive_peer_message(
        text, *, mode="mailbox", wake=False, sender=None
    ):  # noqa: ANN001
        assert (mode, wake) == ("mailbox", False), "the audit card must never open a turn"
        cards.append((text, sender or {}))
        return "delivered to the mailbox (will be read on the next turn)"

    monkeypatch.setattr(handle, "receive_peer_message", receive_peer_message)
    return handle, session, applied, cards


@pytest.mark.asyncio
async def test_serving_switches_reads_back_and_records_the_card(monkeypatch) -> None:
    handle, _session, applied, cards = _serving(monkeypatch)
    sender = {"pid": 4242, "conversation_name": "fleet boss"}

    detail = await handle.receive_peer_model("DeepSeek", " deepseek-flash ", sender=sender)

    assert detail.splitlines() == [
        "switched to deepseek/deepseek-flash (was test/mock)",
        "its next turn runs on it",
    ]
    assert [(spec.provider, spec.model_id, explicit) for spec, explicit in applied] == [
        ("deepseek", "deepseek-flash", True)
    ]
    assert cards == [
        (
            "[remote model switch] now on deepseek/deepseek-flash (was test/mock) — switched by "
            "fleet boss",
            sender,
        )
    ]


@pytest.mark.asyncio
async def test_serving_refusal_mutates_nothing(monkeypatch) -> None:
    handle, _session, applied, cards = _serving(monkeypatch)
    with pytest.raises(ValueError) as caught:
        await handle.receive_peer_model("nosuchprov", "x", sender={})
    assert str(caught.value) == "refused: 'nosuchprov' is not a known provider; still on test/mock"
    assert applied == [] and cards == []


@pytest.mark.asyncio
async def test_serving_same_pair_is_a_no_op(monkeypatch) -> None:
    handle, session, applied, cards = _serving(monkeypatch)
    session.model_label = session.effective_model_label = "deepseek/deepseek-flash"
    detail = await handle.receive_peer_model("deepseek", "deepseek-flash", sender={})
    assert detail == "already on deepseek/deepseek-flash\nnothing changed"
    assert applied == [] and cards == []


@pytest.mark.asyncio
async def test_serving_switching_onto_the_active_fallback_is_a_real_switch(monkeypatch) -> None:
    """M1: while a fallback serves ``deepseek/deepseek-flash`` for a Claude
    selection, asking for ``deepseek/deepseek-flash`` must make it the SELECTION
    (the explicit re-selection withdraws the pin, as ``/model`` does) — not
    answer "already on" and leave the session to return to Claude."""
    from types import SimpleNamespace

    handle, session, applied, cards = _serving(monkeypatch)
    session.model_label = "anthropic/claude-opus-4"
    session.effective_model_label = "deepseek/deepseek-flash"
    # FakeSession declares no fallback slot; the handle reads it with getattr.
    setattr(session, "active_fallback", SimpleNamespace(provider="deepseek", model_id="x"))

    def set_model(spec, explicit=False):  # noqa: ANN001
        applied.append((spec, explicit))
        setattr(session, "active_fallback", None)  # the explicit re-selection drops the pin
        session.model_label = session.effective_model_label = f"{spec.provider}/{spec.model_id}"

    monkeypatch.setattr(session, "set_model", set_model, raising=False)
    detail = await handle.receive_peer_model("deepseek", "deepseek-flash", sender={})
    assert detail.splitlines()[0] == (
        "switched to deepseek/deepseek-flash (was anthropic/claude-opus-4)"
    )
    assert [(s.model_id, explicit) for s, explicit in applied] == [("deepseek-flash", True)]
    assert session.model_label == "deepseek/deepseek-flash"
    assert len(cards) == 1


@pytest.mark.asyncio
async def test_serving_busy_switch_names_the_call_in_flight(monkeypatch) -> None:
    handle, session, _applied, _cards = _serving(monkeypatch)
    session.is_streaming = True
    session.running_children = 2
    detail = await handle.receive_peer_model("deepseek", "deepseek-flash", sender={})
    assert "mid-turn: the call in flight finishes on the old model; later calls use" in detail
    assert "2 running subagents keep their model" in detail


@pytest.mark.asyncio
async def test_serving_a_switch_that_did_not_take_is_a_refusal(monkeypatch) -> None:
    """The answer comes from the read-back, never from the switch's receipt."""
    handle, session, _applied, cards = _serving(monkeypatch)
    monkeypatch.setattr(session, "set_model", lambda spec, explicit=False: None, raising=False)
    with pytest.raises(ValueError) as caught:
        await handle.receive_peer_model("deepseek", "deepseek-flash", sender={})
    assert "did not take effect" in str(caught.value)
    assert "still on test/mock" in str(caught.value)
    assert cards == []


@pytest.mark.asyncio
async def test_serving_a_raise_after_the_switch_took_is_reported_as_switched(monkeypatch) -> None:
    """N1: ``Session.set_model`` assigns before its journal and notify steps, so a
    later raise can leave the new model in force. The answer is read back, never
    "nothing changed"; the card is still written."""
    handle, session, _applied, cards = _serving(monkeypatch)

    def set_model(spec, explicit=False):  # noqa: ANN001
        session.model_label = session.effective_model_label = f"{spec.provider}/{spec.model_id}"
        raise RuntimeError("journal write failed")

    monkeypatch.setattr(session, "set_model", set_model, raising=False)
    detail = await handle.receive_peer_model("deepseek", "deepseek-flash", sender={})
    assert detail.splitlines() == [
        "switched to deepseek/deepseek-flash (was test/mock)",
        "with an error after the switch: RuntimeError: journal write failed",
    ]
    assert len(cards) == 1


@pytest.mark.asyncio
async def test_serving_a_raise_before_the_switch_took_is_a_refusal_on_the_real_model(
    monkeypatch,
) -> None:
    handle, session, _applied, cards = _serving(monkeypatch)

    def set_model(spec, explicit=False):  # noqa: ANN001
        raise RuntimeError("store locked")

    monkeypatch.setattr(session, "set_model", set_model, raising=False)
    with pytest.raises(ValueError) as caught:
        await handle.receive_peer_model("deepseek", "deepseek-flash", sender={})
    assert str(caught.value) == (
        "refused: the switch to deepseek/deepseek-flash did not take effect; still on test/mock"
    )
    assert cards == []
