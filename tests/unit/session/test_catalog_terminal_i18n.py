"""Whole terminal labels are localized; stored diagnostics and state are not.

Synthetic catalogue markers prove dispatch without inventing target translations.
The actual locale/audit contribution is intentionally separate from these tests.
"""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from local_operator.i18n import catalogues
from local_operator.incidents import (
    CUT_OFF_CAUSES,
    render_cut_off_reason,
    render_stop_attribution,
    stop_rung_from_reason,
    stop_rung_phrase,
)
from local_operator.resume import SessionRow
from local_operator.session import catalog


@pytest.fixture(autouse=True)
def english(monkeypatch):
    monkeypatch.setattr(catalog, "_label_locale", lambda: "en")


def entry(kind="", *, unseen=False, reason="", **row_fields):
    row = SessionRow("synthetic001", 1.0, "Synthetic session", **row_fields)
    return catalog.entry_for(row, {
        "kind": kind, "unseen": unseen, "reason": reason,
        "completion_token": "synthetic-token" if kind else "",
    })


@pytest.mark.parametrize(("kind", "code", "label"), [
    ("interrupted", "interrupted", "Interrupted"),
    ("retired", "interrupted", "Interrupted — retired for an update"),
    ("error", "error", "Error"),
    ("complete", "complete", "Complete"),
    ("", "recent", "Recent"),
])
def test_acknowledged_terminal_state(kind, code, label):
    item = entry(kind)
    assert (item.status_code, item.status) == (code, label)


@pytest.mark.parametrize(("kind", "label"), [
    ("interrupted", "Unseen interruption"),
    ("retired", "Unseen interruption — retired for an update"),
    ("error", "Unseen error"),
    ("complete", "Unseen completion"),
])
def test_unseen_terminal_state(kind, label):
    item = entry(kind, unseen=True)
    assert item.status == label
    assert catalog.status_of(item.row, {
        "kind": kind, "unseen": True, "completion_token": "synthetic-token",
    }) == (item.status_code, label)


@pytest.mark.parametrize("unseen", [False, True])
@pytest.mark.parametrize(("rung", "phrase"), [
    ("socket", ""), ("sigterm", " — stopped with a signal by /stop --all"),
    ("sigkill", " — killed by /stop --all"), ("future", ""),
])
def test_stop_escalation_uses_semantic_decoder(unseen, rung, phrase):
    reason = render_cut_off_reason("user-stop") + render_stop_attribution(
        rung=rung, command="/stop --all", killer_pid=123,
    )
    item = entry("interrupted", unseen=unseen, reason=reason)
    assert item.status == ("Unseen interruption" if unseen else "Interrupted") + phrase
    assert item.completion_reason == reason
    if rung in {"sigterm", "sigkill"}:
        assert stop_rung_from_reason(reason) == (rung, "/stop --all")
        assert stop_rung_phrase(reason) == phrase.removeprefix(" — ")
    else:
        assert stop_rung_from_reason(reason) == ("", "")


@pytest.mark.parametrize("rung", ["sigterm", "sigkill"])
@pytest.mark.parametrize("unseen", [False, True])
def test_pid_only_attribution_is_unknown_not_a_command(monkeypatch, rung, unseen):
    monkeypatch.setattr(catalog, "_label_locale", lambda: "fr")
    reason = render_cut_off_reason("user-stop") + render_stop_attribution(
        rung=rung, command="", killer_pid=123,
    )
    # Historical partial markers lost the command/pid distinction in their
    # sentence. Identical bytes cannot truthfully recover different actors.
    ambiguous = render_cut_off_reason("user-stop") + render_stop_attribution(
        rung=rung, command="killer pid 123",
    )
    assert reason == ambiguous
    assert stop_rung_from_reason(reason) == (rung, "")
    item = entry("interrupted", unseen=unseen, reason=reason)
    # The existing no-known-command ICU arm must retain the escalation rather
    # than leaking diagnostic English or manufacturing a technical command.
    from local_operator.i18n.keys import wire_session_status as keys
    from local_operator.i18n.messages import render

    key = (
        keys.unseen_signal_stop(command="none") if unseen else keys.signal_stop(command="none")
    ) if rung == "sigterm" else (
        keys.unseen_forced_kill(command="none") if unseen else keys.forced_kill(command="none")
    )
    assert item.status == render(key.code, dict(key.params), locale="fr")
    assert "killer pid" not in item.status and " via " not in item.status
    assert item.status_code == "interrupted"
    assert item.completion_reason == reason
    assert entry("interrupted", unseen=unseen, reason=ambiguous).status == item.status


@pytest.mark.parametrize("rung", ["sigterm", "sigkill"])
@pytest.mark.parametrize("command", ["killer pid 123", "/stop --all opaque-東京", "killer pid 123 extra"])
@pytest.mark.parametrize("unseen", [False, True])
def test_separate_pid_preserves_real_opaque_command(monkeypatch, rung, command, unseen):
    monkeypatch.setattr(catalog, "_label_locale", lambda: "fr")
    reason = render_cut_off_reason("user-stop") + render_stop_attribution(
        rung=rung, command=command, killer_pid=456,
    )
    assert stop_rung_from_reason(reason) == (rung, command)
    item = entry("interrupted", unseen=unseen, reason=reason)
    assert command in item.status
    assert "456" not in item.status
    assert item.status_code == "interrupted"
    assert item.completion_reason == reason


@pytest.mark.parametrize("command", ["killer pid 123 extra", "killer pid 0123", "killer pid nope", "prefix killer pid 123"])
def test_noncanonical_actor_is_not_stripped(command):
    reason = render_cut_off_reason("user-stop") + render_stop_attribution(
        rung="sigkill", command=command,
    )
    assert stop_rung_from_reason(reason) == ("sigkill", command)
    assert stop_rung_from_reason("external killer pid 123") == ("", "")


@pytest.mark.parametrize("cause", list(CUT_OFF_CAUSES))
@pytest.mark.parametrize("unseen", [False, True])
def test_known_error_causes_are_message_selectors(cause, unseen):
    reason = render_cut_off_reason(cause, detail="synthetic forensic detail")
    item = entry("error", unseen=unseen, reason=reason)
    assert item.status.startswith("Unseen error — " if unseen else "Error — ")
    assert "synthetic forensic detail" not in item.status
    assert item.completion_reason == reason
    assert item.status_code == "error"


@pytest.mark.parametrize("reason", ["Provider refusal\nraw second line", "legacy unknown attribution"])
def test_unknown_diagnostic_stays_raw_not_a_translated_parameter(reason):
    attention = {"kind": "error", "unseen": True, "completion_token": "t", "reason": reason}
    before = json.dumps(attention)
    row = SessionRow("synthetic001", 1., "Synthetic")
    item = catalog.entry_for(row, attention)
    assert catalog.status_of(row, attention) == ("error", "Unseen error — " + reason.splitlines()[0])
    assert item.completion_reason == reason
    assert json.dumps(attention) == before
    assert entry("interrupted", reason=reason).status == "Interrupted"


@pytest.mark.parametrize("count", [1, 2, 0, 21])
def test_dormant_plural_count_and_code(count):
    # The dormant arm requires at least one wake; zero remains the recent arm.
    item = entry(wakes=count, wakes_dormant=True)
    assert item.status == (f"Stopped ({count} {'wake' if count == 1 else 'wakes'} dormant)" if count else "Recent")
    assert item.status_code == ("dormant" if count else "recent")


def test_catalogue_not_english_base_and_locale_does_not_change_order(tmp_path, monkeypatch):
    source = catalogues._CATALOGUES / "en" / "wire.session_status.json"
    messages = json.loads(source.read_text())
    for locale in ("en", "fr"):
        (tmp_path / locale).mkdir()
        # Opaque fixture markers, NOT a translation contribution.
        payload = messages if locale == "en" else {
            key: f"fixture:{key.rsplit('.', 1)[1]}" for key in messages
        }
        (tmp_path / locale / "wire.session_status.json").write_text(json.dumps(payload))
    monkeypatch.setattr(catalogues, "_CATALOGUES", tmp_path)
    from local_operator.i18n import resolve

    # Eligibility belongs to this isolated fixture too. Never alter the real
    # generated shipped set to make an incomplete translation contribution pass.
    monkeypatch.setattr(resolve, "shipped_locales", lambda: ("en", "fr"))
    items = [entry("retired", unseen=True), entry("error"), entry("complete"), entry()]
    old = [(item.status_code, item.rank, item.active, item.shows_completion_mark) for item in items]
    monkeypatch.setattr(catalog, "_label_locale", lambda: "fr")
    assert [item.status for item in items] == [
        "fixture:unseen_retirement", "fixture:error", "fixture:complete", "fixture:recent",
    ]
    assert [(item.status_code, item.rank, item.active, item.shows_completion_mark) for item in items] == old


@pytest.mark.parametrize("live_state", ["busy", "attached", "idle"])
def test_terminal_extraction_preserves_live_precedence(live_state):
    original = entry("error", unseen=True, live_state=live_state)
    changed = replace(original, completion_reason="external diagnostic")
    assert (changed.status_code, changed.rank) == (original.status_code, original.rank)
    # Busy outranks the unseen receipt; attached/idle do not. Their useful
    # external reason remains visible, exactly as before this extraction.
    assert changed.status == (
        "Working" if live_state == "busy" else "Unseen error — external diagnostic"
    )


@pytest.mark.parametrize("unseen", [False, True])
@pytest.mark.parametrize("reason", ["  external (detail)\nsecond line", "x" * 170])
def test_external_diagnostic_retains_legacy_label_bytes(unseen, reason):
    base = "Unseen error" if unseen else "Error"
    sentence = reason.strip().splitlines()[0].strip().split(" (", 1)[0].rstrip()
    if len(sentence) > 160:
        sentence = sentence[:157].rstrip() + "…"
    item = entry("error", unseen=unseen, reason=reason)
    assert item.status == f"{base} — {sentence}"
    assert item.completion_reason == reason
