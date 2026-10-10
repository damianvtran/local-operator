"""The /session "Spend by channel" section: published object only, no local sums.

The section is the panel half of the one rule this feature exists to establish:
every surface renders the object the backend publishes, and ``None`` (an old
backend) draws the legacy search block instead of an invented channel view.
"""

from __future__ import annotations

from dataclasses import replace

from local_operator.session.channel_spend import (
    ChannelSpendRecord,
    ChildrenSnapshot,
    InferenceSnapshot,
    combine,
    fold_records,
)
from local_operator.session.frontend_state import FrontendSpendChannels
from local_operator.tui.widgets.session_panel import (
    SessionDiagnostics,
    _Body,
    _draw_spend_channels,
)
from tests.unit.tui.test_session_panel import runtime


def payload(*, tracked: bool = True) -> FrontendSpendChannels:
    records = [
        ChannelSpendRecord(
            record_id="image:a",
            rev=1,
            channel="image",
            provider="radient",
            model="gpt-image-2",
            units=1,
            unit="images",
            amount_micro=61000,
            billing_basis="billed",
            cost_source="server_reported",
            status="ok",
        ),
        ChannelSpendRecord(
            record_id="tts:a",
            channel="tts",
            provider="radient",
            model="elevenlabs",
            units=420,
            unit="chars",
            amount_micro=None,
            status="ok",
        ),
    ]
    snapshot = combine(
        fold_records(records).rows(),
        inference=InferenceSnapshot(
            micro=900000,
            calls=10,
            priced_calls=10,
            knowledge="exact",
            by_identity={
                "anthropic/claude-sonnet-5-5": {
                    "provider": "anthropic",
                    "model_id": "claude-sonnet-5-5",
                    "micro": 900000,
                    "calls": 10,
                    "unpriced": 0,
                }
            },
        ),
        children=ChildrenSnapshot(),
        tracked=tracked,
        lost=False,
    )
    return FrontendSpendChannels.model_validate(snapshot)


def rendered(runtime_diag: SessionDiagnostics) -> str:
    body = _Body(width=100)
    assert _draw_spend_channels(body, runtime_diag) is True
    return body.to_text().plain


def test_section_renders_the_published_rows_total_and_basis() -> None:
    text = rendered(replace(runtime(), spend_channels=payload()))
    assert "Spend by channel" in text
    # The total is the published one (inference + channels), marked partial by
    # the TTS row's unknown amount — never a locally re-summed figure.
    assert "$0.961 +" in text
    assert "knowledge: partial" in text
    assert "inference · anthropic/claude-sonnet-5-5" in text
    assert "$0.900" in text
    assert "image · radient/gpt-image-2" in text and "$0.061" in text
    assert "tts · radient/elevenlabs" in text and "$—" in text
    assert "By basis: billed $0.061 · 2 not tracked" in text, (
        # Two units: the TTS row's unstated amount, and inference itself — its
        # route→basis mapping (billed vs subscription) is a PR-3 item, so its
        # money is in the total and the row, but it cannot be attributed to a
        # basis bucket yet and the count SAYS so rather than hiding it.
        "the basis line must account for the money it cannot bucket"
    )


def test_section_sheds_without_a_published_object() -> None:
    body = _Body(width=100)
    assert _draw_spend_channels(body, runtime()) is False
    assert body.to_text().plain.strip() == "", "no published object means no section"


def test_untracked_session_says_so_and_keeps_its_rows() -> None:
    text = rendered(replace(runtime(), spend_channels=payload(tracked=False)))
    assert "Channels not tracked for this conversation" in text
    assert "image · radient/gpt-image-2" in text, "recovered rows still render"
