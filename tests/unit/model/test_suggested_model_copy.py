"""The router suggestion's display name is catalogue copy; product names are not.

Desktop fr census row 11: the connect-provider list read "Suggère Automatic".
``Automatic`` describes Radient's router rather than naming a product, so
``suggested_model_for`` renders it from ``wire.providers.connect`` at read time;
vendor model names ("Claude Opus 5.5") stay as written, and the id ``auto`` is
a machine token that never changes.
"""

from __future__ import annotations

import pytest

from local_operator.i18n import messages
from local_operator.model import defaults


def test_the_router_name_is_rendered_from_the_catalogue(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(messages, "render_resolved", lambda message: f"<{message.code}>")

    radient = defaults.suggested_model_for("radient")

    assert radient == defaults.SuggestedModel(
        "auto", "<wire.providers.connect.suggested_automatic>"
    )
    # The table entry itself keeps the English source and fallback.
    assert defaults.SUGGESTED_MODELS["radient"].name == "Automatic"


def test_vendor_product_names_are_not_keyed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(messages, "render_resolved", lambda message: f"<{message.code}>")

    assert defaults.suggested_model_for("anthropic") == defaults.SUGGESTED_MODELS["anthropic"]
    assert defaults.suggested_model_for("kimi", oauth=True) == defaults.SuggestedModel(
        "k3", "Kimi K3"
    )


def test_english_is_unchanged_and_a_render_failure_keeps_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert defaults.suggested_model_for("radient") == defaults.SuggestedModel("auto", "Automatic")

    def boom(message: object) -> str:
        raise RuntimeError("catalogue unreadable")

    monkeypatch.setattr(messages, "render_resolved", boom)
    assert defaults.suggested_model_for("radient") == defaults.SuggestedModel("auto", "Automatic")
