"""P3b team authoring copy is complete in every requested locale, not fallback."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_operator.i18n import catalogues, runtime
from local_operator.i18n.keys import tui_team_authoring


@pytest.mark.parametrize("locale", ["en", "es", "fr", "hi", "ru", "vi"])
def test_authoring_catalogue_is_complete_and_renderable(locale: str) -> None:
    source = catalogues.load_catalogue("en", "tui.team_authoring")
    translated = catalogues.load_catalogue(locale, "tui.team_authoring")
    assert len(source) == 35
    assert set(translated) == set(source)
    for key, value in translated.items():
        assert value
        rendered = runtime.render_message(
            value, {"name": "crew", "reason": "test", "detail": "test"}, locale
        )
        assert rendered
        assert "{name}" not in rendered
        assert "{reason}" not in rendered
        function = getattr(tui_team_authoring, key.removeprefix("tui.team_authoring."))
        params = {key: "test" for key in ("name", "reason", "detail") if "{" + key + "}" in value}
        assert function(**params).code == key


def test_authoring_context_and_zero_literal_activation() -> None:
    context = json.loads(
        (catalogues.catalogue_root() / "en/tui.team_authoring.context.json").read_text()
    )
    assert set(context) == set(catalogues.load_catalogue("en", "tui.team_authoring"))
    assert all(row["note"] and row["role"] and row["tone"] for row in context.values())
    root = Path(__file__).resolve().parents[3]
    baseline = json.loads((root / "i18n/baseline.json").read_text())
    for path in (
        "local_operator/tui/team_authoring.py",
        "local_operator/tui/widgets/team_form.py",
    ):
        assert path in baseline["enforced"]
        assert baseline["files"].get(path, 0) == 0
