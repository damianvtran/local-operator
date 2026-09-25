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
    "busy,children,expected",
    [
        (False, 0, "switched to b/y (was a/x); its next turn runs on it"),
        (
            True,
            0,
            "switched to b/y (was a/x) mid-turn; the call in flight finishes on a/x, "
            "every later call uses b/y",
        ),
        (
            False,
            1,
            "switched to b/y (was a/x); its next turn runs on it; 1 running subagent keeps "
            "its current model — new and resumed ones use b/y",
        ),
        (
            False,
            3,
            "switched to b/y (was a/x); its next turn runs on it; 3 running subagents keep "
            "their current model — new and resumed ones use b/y",
        ),
    ],
)
def test_the_switch_receipts_are_the_designed_strings(busy, children, expected) -> None:
    assert (
        peer_model.switched_detail("a/x", "b/y", busy=busy, running_subagents=children) == expected
    )


def test_the_other_receipts() -> None:
    assert peer_model.already_on_detail("a/x") == "already on a/x; nothing changed"
    assert (
        peer_model.refusal_detail("'q' is not a known provider.", "a/x")
        == "refused: 'q' is not a known provider; still on a/x"
    )
    assert peer_model.audit_body("a/x", "b/y") == (
        "[remote model switch] switched this session from a/x to b/y"
    )


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

    assert detail == (
        "switched to deepseek/deepseek-flash (was test/mock); its next turn runs on it"
    )
    assert [(spec.provider, spec.model_id, explicit) for spec, explicit in applied] == [
        ("deepseek", "deepseek-flash", True)
    ]
    assert cards == [
        (
            "[remote model switch] switched this session from test/mock to deepseek/deepseek-flash",
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
    assert detail == "already on deepseek/deepseek-flash; nothing changed"
    assert applied == [] and cards == []


@pytest.mark.asyncio
async def test_serving_busy_switch_names_the_call_in_flight(monkeypatch) -> None:
    handle, session, _applied, _cards = _serving(monkeypatch)
    session.is_streaming = True
    session.running_children = 2
    detail = await handle.receive_peer_model("deepseek", "deepseek-flash", sender={})
    assert "mid-turn; the call in flight finishes on test/mock" in detail
    assert "2 running subagents keep their current model" in detail


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
