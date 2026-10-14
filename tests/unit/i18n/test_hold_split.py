"""Hindi-only split: held Chinese bytes must be absent, not merely unlisted.

No runtime policy is changed here. The current main's public wire and TUI
entrypoints enforce eligibility and fall back to English for held locales.
The complete candidate remains on a remote-held ref for future audit work.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rich.cells import cell_len

from local_operator.i18n import catalogues, resolve, runtime, text
from local_operator.i18n.messages import Msg, envelope


def test_active_ledger_excludes_the_entire_held_chinese_candidate() -> None:
    ledger = Path(__file__).resolve().parents[3] / "i18n" / "ledger.json"
    entries = json.loads(ledger.read_bytes())["entries"]
    assert not any(entry["locale"] == "zh-CN" for entry in entries)
    hindi = [entry for entry in entries if entry["locale"] == "hi"]
    # ELEVEN, not ten: the Wave-3 ten still carry the audited old set, and the
    # 17-key ``tui.transcript`` union adds ONE more row for the same namespace
    # under the CURRENT English source hash. The count is the pin that keeps a
    # held candidate from being smuggled in as a re-hash of an existing
    # namespace, so it moves only with a reviewed union row like this one.
    assert len(hindi) == 11
    assert all(entry["status"] == "passed" for entry in hindi)


@pytest.mark.parametrize("namespace", catalogues.namespaces("en"))
def test_no_held_chinese_namespace_can_be_loaded(namespace: str) -> None:
    with pytest.raises(catalogues.CatalogueNotFound):
        catalogues.load_catalogue("zh-CN", namespace)
    with pytest.raises(catalogues.CatalogueNotFound):
        catalogues.catalogue_sha256("zh-CN", namespace)


@pytest.mark.parametrize("key", ("wire.slash.delete", "wire.exec.receipt_suffix"))
def test_explicit_held_locale_uses_existing_wire_english_fallback(key: str) -> None:
    message = Msg(key)
    english = message.envelope(locale="en")
    assert english["text"] != key
    assert message.envelope(locale="zh-CN") == english
    assert envelope(key, locale="zh-CN") == english


@pytest.mark.parametrize("key", ("wire.slash.delete", "wire.exec.receipt_suffix"))
def test_direct_tr_never_exposes_held_copy_fresh_or_cached(key: str) -> None:
    namespace = key.rsplit(".", 1)[0]
    message = Msg(key)
    render = text.tr(namespace)
    previous = text.active_locale()
    try:
        # The upstream eligibility/fallback fix normalizes held tags at the
        # direct entrypoint too. Preserve that landed policy during convergence,
        # rather than restoring this branch's older message-code degradation.
        text.set_active_locale("zh-CN")
        assert text.active_locale() == "en"
        english = message.envelope(locale="en")["text"]
        assert english != key
        assert render(message) == english
        assert render(message) == english
        # Normal product selection also resolves the held tag before using tr.
        text.set_active_locale(resolve.resolve_language("zh-CN", environ={}, system="Linux"))
        assert render(message) == message.envelope(locale="en")["text"]
    finally:
        text.set_active_locale(previous)


@pytest.mark.parametrize(
    "suffix",
    (
        "team_bridge.description",
        "team_bridge.title",
        "team_bridge_enabled.help",
        "team_bridge_enabled.label",
        "team_bridge_session.help",
        "team_bridge_session.label",
    ),
)
def test_hindi_team_bridge_delta_uses_audited_copy_not_english_fallback(suffix: str) -> None:
    # The landed English source gained these six keys after the old 755-key
    # audit. Check the public consumer, not only JSON parity: a missing Hindi
    # value would otherwise silently render English while locale selection works.
    key = f"wire.settings.{suffix}"
    hindi = catalogues.load_catalogue("hi", "wire.settings")[key]
    english = catalogues.load_catalogue("en", "wire.settings")[key]
    rendered = Msg(key).envelope(locale="hi")["text"]
    assert rendered == hindi
    assert rendered != english
    assert runtime.message_arguments(hindi) == ()
    if suffix.endswith(".label"):
        # Use the real row budget; help and description have no authored cap.
        from local_operator.tui.widgets.settings_view import _LABEL_BUDGET

        assert cell_len(rendered) <= _LABEL_BUDGET
    if suffix == "team_bridge_enabled.help":
        assert "Radient" in rendered
    if suffix == "team_bridge_session.label":
        assert "CoS" in rendered
