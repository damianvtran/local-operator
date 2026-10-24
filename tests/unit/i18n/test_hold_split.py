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
    # TWENTY-FOUR: the union plus independently audited P3a/P3b namespaces,
    # the scratchpad-hygiene slice's hi wire.settings row, P4a's audited
    # tui.viewer row (including its keyboard-hint audit refresh), and the
    # i18n lane's audited `wire.memory` row for the namespace translated in
    # five active locales.
    # Base is twelve (the Wave-3 ten retain their audited source pins). Main
    # added a wire.settings row for the hub-setting deletion and the LOH-19
    # review slice added a tui.review row; the earlier branch added a
    # tui.transcript row for the independent re-attestation of the carried
    # content; LOH-5 slice 4 adds two wire.settings rows for the
    # transcription-cleanup setting and its D1 copy remediation. P1 adds the
    # audited `tui.schedule` row on top — SEVENTEEN from main's history plus
    # that one. LOH-19 P2 adds the audited `tui.projects_edit` row — EIGHTEEN
    # plus that one. P3a adds `tui.authoring` after its independent 40-key
    # audit — NINETEEN plus one. Keep the cardinality pin: adding an audited union must not
    # silently admit an unrelated held namespace or discard the previous audit
    # history.
    # P3b adds its 35-key tui.team_authoring namespace after independent audit.
    # The audited `hi/wire.memory` row this branch lands makes it TWENTY-FOUR;
    # the /tag slice's independently re-audited 27-key `tui.tags` row makes it
    # TWENTY-FIVE. LOH-11's independently audited `relay.session.tags` row makes
    # it TWENTY-SIX. The remote_compute six-key wire.settings delta's
    # independently audited row makes it TWENTY-SEVEN. The S3e remote-compute
    # wave adds one row per carried namespace (`cli.exec_remote_help`,
    # `wire.remote_compute`) -- TWENTY-NINE. The pin stays cardinal so an
    # unrelated held namespace cannot slip in. The session-transcript-
    # compression slice's independently audited 8-key `wire.settings` delta
    # (0 blocker, 0 major; comments carried) makes it THIRTY. The R5 stale-ask
    # wave's two independently audited namespaces (`tui.asks`, `relay.ask`)
    # make it THIRTY-TWO, and the P5 command-palette slice's independently
    # audited 7-key `tui.palette` row makes it THIRTY-THREE. The fast-mode
    # fix's refusal notices add one audited row each for `tui.notices` and
    # `wire.errors.desktop` -- THIRTY-FIVE (both newly carried, 30 and 84 keys).
    assert len(hindi) == 35
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
