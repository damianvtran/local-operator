"""The shared deferred duration seam preserves legacy arithmetic and string types."""

from __future__ import annotations

import pytest

from local_operator.i18n.duration_message import format_duration
from local_operator.i18n.messages import Msg


@pytest.mark.parametrize(
    ("seconds", "suffix", "params"),
    [
        (-1, "seconds", {"count": 0}),
        (0, "seconds", {"count": 0}),
        (0.9, "seconds", {"count": 0}),
        (59.9, "seconds", {"count": 59}),
        (60, "minutes", {"count": 1}),
        (3599.9, "minutes", {"count": 59}),
        (3600, "hours_minutes", {"hours": 1, "minutes": 0}),
        (86399.9, "hours_minutes", {"hours": 23, "minutes": 59}),
        (86400, "days_hours", {"days": "1", "hours": 0}),
        (86400000, "days_hours", {"days": "1000", "hours": 0}),
    ],
)
def test_thresholds_preserve_callback_string_subclass(seconds, suffix, params) -> None:
    class Provenance(str):
        pass

    result = Provenance("rendered")
    received: list[Msg] = []

    def render(message: Msg) -> Provenance:
        received.append(message)
        return result

    assert format_duration(seconds, render) is result
    assert len(received) == 1
    assert received[0].code.endswith(".duration_" + suffix)
    assert received[0].params == params


@pytest.mark.parametrize("seconds", [0, 60, 3600, 86400])
def test_miswired_callback_cannot_return_an_envelope(seconds) -> None:
    # Deliberately violate typing: runtime boundaries must also reject envelopes.
    with pytest.raises(TypeError, match="duration renderer must return str"):
        format_duration(seconds, lambda message: message)  # type: ignore[type-var]


@pytest.mark.parametrize("locale", ["en", "fr", "es", "zh-CN", "ru", "vi", "ur", "hi"])
@pytest.mark.parametrize("seconds", [0, 60, 3660, 86400000])
def test_info_callback_preserves_actual_source(locale, seconds) -> None:
    from local_operator.i18n import messages, text
    from local_operator.info.copy import InfoText, info_duration

    original = text.active_locale()
    try:
        text.set_active_locale(locale)
        expected = format_duration(
            seconds,
            lambda message: messages.render_record(
                message.code, message.params, locale=locale
            ).text,
        )
        value = info_duration(seconds)
        assert isinstance(value, InfoText)
        assert str(value) == expected
        assert len(value.parts) == 1
        assert value.parts[0].source == locale
        assert not value.parts[0].literals
        assert not any(control in value for control in "\u2066\u2067\u2068\u2069")
    finally:
        text.set_active_locale(original)


@pytest.mark.parametrize("source", ["en", None])
def test_info_callback_does_not_relabel_fallback_or_unknown(monkeypatch, source) -> None:
    from local_operator.i18n import messages, text
    from local_operator.info.copy import info_duration

    original = text.active_locale()
    try:
        text.set_active_locale("ur")
        monkeypatch.setattr(
            messages, "render_record",
            lambda code, params, *, locale: messages.RenderedMessage("fallback", source),
        )
        value = info_duration(60)
        assert str(value) == "fallback"
        assert value.parts[0].source is source
    finally:
        text.set_active_locale(original)


def test_info_duration_obeys_operational_kill_switch(monkeypatch) -> None:
    from local_operator.i18n import text
    from local_operator.info.copy import info_duration

    original = text.active_locale()
    try:
        text.set_active_locale("fr")
        monkeypatch.setenv("LOP_I18N", "0")
        value = info_duration(3660)
        assert value == "1h 1m"
        assert value.parts[0].source == "en"
    finally:
        text.set_active_locale(original)
