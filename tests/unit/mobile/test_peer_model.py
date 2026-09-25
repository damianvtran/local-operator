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
                "mid-turn: the call in flight finishes on the old model",
            ],
        ),
        (
            True,
            False,
            0,
            [
                "switched to b/y (was a/x)",
                "mid-turn: the current step finishes on the old model",
            ],
        ),
        (
            False,
            True,
            1,
            [
                "switched to b/y (was a/x)",
                "its next turn runs on it",
                "1 running subagent stays on the old model; new and resumed ones switch",
            ],
        ),
        (
            False,
            True,
            3,
            [
                "switched to b/y (was a/x)",
                "its next turn runs on it",
                "3 running subagents stay on the old model; new and resumed ones switch",
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


def test_the_audit_card_leads_with_the_new_model_and_never_repeats_the_header() -> None:
    """D3/U3, then D7/U9/Q5: on resume the card is the only trace, clipped from the
    right. The header names the sender, so the body does not repeat it; a
    terminal sender's directory is the one fact the header cannot hold."""
    body = "[remote model switch] now on b/y (was a/x)"
    assert peer_model.audit_body("a/x", "b/y", {"conversation_name": "fleet boss"}) == body
    assert peer_model.audit_body("a/x", "b/y", {"pid": 7}) == body
    assert peer_model.audit_body("a/x", "b/y") == body
    terminal = {"conversation_name": "terminal", "via": "terminal", "cwd": "/srv/work/"}
    assert peer_model.audit_body("a/x", "b/y", terminal) == body + " — from a terminal in work"


@pytest.mark.parametrize("cwd", ["/", "", None])
def test_a_terminal_sender_with_no_usable_directory_names_none(cwd) -> None:
    """NIT-4: `/` has no basename and a deleted cwd reports none."""
    sender = {"conversation_name": "terminal", "via": "terminal", "cwd": cwd}
    assert peer_model.audit_body("a/x", "b/y", sender) == (
        "[remote model switch] now on b/y (was a/x)"
    )


def test_reclaiming_the_displaced_selection_reads_back_on_it() -> None:
    """N5: re-selecting the model a pinned fallback displaced is not
    `switched to X (was X)`; the fallback was dropped."""
    detail = peer_model.switched_detail(
        "a/x", "a/x", busy=False, running_subagents=0, dropped_fallback="b/y"
    )
    assert detail.splitlines()[:2] == ["back on a/x", "was on fallback b/y"]
    assert peer_model.audit_body("a/x", "a/x", dropped_fallback="b/y") == (
        "[remote model switch] back on a/x (was on fallback b/y)"
    )


def test_the_back_on_receipt_fits_an_expanded_card_at_80_columns() -> None:
    """D11: as one line the back-on receipt carried two full ids and clipped a
    long fallback id out of the ~72-cell body at 80 columns. Every line,
    INCLUDING the first, now fits — the busy and partial forms too."""
    new, fallback = "anthropic/claude-opus-5", "openrouter/qwen3-coder-plus"
    for detail in (
        peer_model.switched_detail(
            new, new, busy=False, running_subagents=0, dropped_fallback=fallback
        ),
        peer_model.switched_detail(
            new,
            new,
            busy=True,
            calling=True,
            running_subagents=12,
            dropped_fallback=fallback,
        ),
        peer_model.partial_switch_detail(new, new, OSError("disk full"), dropped_fallback=fallback),
    ):
        assert max(len(line) for line in detail.splitlines()) <= 72, detail
        assert detail.splitlines()[1] == f"was on fallback {fallback}"


def test_a_pending_reclaim_card_names_the_fallback_it_is_on() -> None:
    """N6: a pending re-selection of the displaced model is on the FALLBACK until
    it applies, not on the model it asked for."""
    assert peer_model.pending_audit_body("a/x", "a/x", {}, dropped_fallback="b/y") == (
        "[remote model switch] switch back to a/x requested (on fallback b/y until it applies)"
    )
    assert peer_model.pending_audit_body("a/x", "c/z", {}) == (
        "[remote model switch] switch to c/z requested (on a/x until it applies)"
    )


def test_a_refusal_names_whose_fallback_it_is_still_on() -> None:
    """NIT-5: `did not take effect; still on X` for a switch TO X read as a
    contradiction while X was a fallback. The displaced selection is named."""
    assert peer_model.refusal_detail(
        "the switch to b/y did not take effect", "b/y", displaced="a/x"
    ) == ("refused: the switch to b/y did not take effect; still on b/y (fallback for a/x)")
    assert peer_model.refusal_detail("no.", "a/x") == "refused: no; still on a/x"


def test_every_line_fits_an_expanded_card_at_80_columns() -> None:
    """D8: the busiest receipt's lines stay inside the ~72-cell body at 80 columns
    (the first line carries the ids, which are whatever length they are)."""
    detail = peer_model.switched_detail(
        "anthropic/claude-opus-5",
        "deepseek/deepseek-flash",
        busy=True,
        calling=True,
        running_subagents=12,
    )
    assert max(len(line) for line in detail.splitlines()[1:]) <= 72, detail


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
            "[remote model switch] now on deepseek/deepseek-flash (was test/mock)",
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
    assert "mid-turn: the call in flight finishes on the old model" in detail
    assert "2 running subagents stay on the old model" in detail


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


def _pinned(monkeypatch, *, selected: str, fallback: str):
    """A serving handle whose session SELECTS ``selected`` while a pinned fallback
    serves ``fallback`` (the real ``Session`` shape: effective = the fallback)."""
    from types import SimpleNamespace

    handle, session, applied, cards = _serving(monkeypatch)
    session.model_label = selected
    session.effective_model_label = fallback
    provider, _, model_id = fallback.partition("/")
    setattr(session, "active_fallback", SimpleNamespace(provider=provider, model_id=model_id))
    return handle, session, applied, cards


@pytest.mark.asyncio
async def test_serving_pinned_on_the_request_but_the_apply_never_ran_is_a_refusal(
    monkeypatch,
) -> None:
    """M2: the effective label ALREADY equals the request while the fallback
    serves it, so it cannot say whether the switch took. The selection can."""
    handle, session, applied, cards = _pinned(
        monkeypatch, selected="anthropic/claude-opus-5", fallback="deepseek/deepseek-flash"
    )

    async def never_applied(*_args, **_kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("metadata fetch timed out")  # before Session.set_model

    monkeypatch.setattr(handle, "set_model_effort", never_applied)
    with pytest.raises(ValueError) as caught:
        await handle.receive_peer_model("deepseek", "deepseek-flash", sender={})
    assert str(caught.value) == (
        "refused: the switch to deepseek/deepseek-flash did not take effect; "
        "still on deepseek/deepseek-flash (fallback for anthropic/claude-opus-5)"
    )
    assert cards == [], "a switch that did not happen must not write a card"
    assert session.model_label == "anthropic/claude-opus-5"


@pytest.mark.asyncio
async def test_serving_reclaiming_the_displaced_selection_says_back_on(monkeypatch) -> None:
    """N5 on the serving host."""
    handle, session, applied, cards = _pinned(
        monkeypatch, selected="anthropic/claude-opus-5", fallback="deepseek/deepseek-flash"
    )

    def set_model(spec, explicit=False):  # noqa: ANN001
        applied.append((spec, explicit))
        setattr(session, "active_fallback", None)
        session.effective_model_label = session.model_label

    monkeypatch.setattr(session, "set_model", set_model, raising=False)
    detail = await handle.receive_peer_model("anthropic", "claude-opus-5", sender={})
    assert detail.splitlines()[:2] == [
        "back on anthropic/claude-opus-5",
        "was on fallback deepseek/deepseek-flash",
    ]
    assert [text for text, _ in cards] == [
        "[remote model switch] back on anthropic/claude-opus-5 "
        "(was on fallback deepseek/deepseek-flash)"
    ]
