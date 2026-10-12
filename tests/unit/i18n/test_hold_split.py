"""Hindi-only split: held Chinese bytes must be absent, not merely unlisted.

No runtime policy is changed here. Public wire messages already fall back to
English; direct TUI ``tr`` intentionally yields the code for missing groups.
The complete candidate remains on a remote-held ref for future audit work.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_operator.i18n import catalogues, resolve, text
from local_operator.i18n.messages import Msg, envelope


def test_active_ledger_excludes_the_entire_held_chinese_candidate() -> None:
    ledger = Path(__file__).resolve().parents[3] / "i18n" / "ledger.json"
    entries = json.loads(ledger.read_bytes())["entries"]
    assert not any(entry["locale"] == "zh-CN" for entry in entries)
    hindi = [entry for entry in entries if entry["locale"] == "hi"]
    assert len(hindi) == 10
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
        # Do not invent an English fallback in direct-tr as part of a data
        # split. Its documented missing-group degradation is the key itself.
        text.set_active_locale("zh-CN")
        assert render(message) == key
        assert render(message) == key
        # Normal product selection resolves the held tag before using tr.
        text.set_active_locale(resolve.resolve_language("zh-CN", environ={}, system="Linux"))
        assert render(message) == message.envelope(locale="en")["text"]
    finally:
        text.set_active_locale(previous)
