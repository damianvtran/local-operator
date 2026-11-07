"""The ``closed`` / ``retired`` outcome sentences follow the ACTIVE language.

The defect these pin: ``tui/notify.py`` imported the two sentences from
``harness/rows.py`` BY VALUE into the banner vocabulary, so any non-English
language showed a localized subtitle over an English body (found independently
by four auditors and a QA run), and the phone push appended an English
``N conversations need you``. The sentences now live in ``wire.outcome_notices``
and the count clause in ``wire.push_alert``.

LOCALE-AGNOSTIC ON PURPOSE. Expectations are derived from the generated key
functions through the same resolver the product uses, never typed as French (or
any other) prose, so they stay true when a real translation lands. ``fr`` is a
FIXTURE locale here — a temp catalogue root of marker values plus a widened
``shipped_locales`` — because the seven target locales do not carry these
namespaces until they are translated and audited, and a real-locale test would
silently compare English with English.

THE SWITCH HAPPENS AFTER IMPORT. A constant resolved at import (or imported by
value) reads English forever; only a live ``set_active_locale`` AFTER every
module is loaded can prove the banner vocabulary re-materialises.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from local_operator.harness import notice_copy
from local_operator.harness.rows import completion_notice
from local_operator.i18n import catalogues
from local_operator.i18n import resolve as resolve_mod
from local_operator.i18n import text
from local_operator.i18n.keys import wire_outcome_notices as _k
from local_operator.i18n.keys import wire_push_alert as _kp
from local_operator.mobile import push_payload
from local_operator.notifications.compose import BODIES as COMPOSE_BODIES
from local_operator.tui import notify

NAMESPACES = ("wire.outcome_notices", "wire.push_alert")


@pytest.fixture(autouse=True)
def _restore_locale() -> Iterator[None]:
    """Move the process back to ``en`` against the REAL root (see test_language_live)."""
    real_root = catalogues.catalogue_root()
    yield
    patched = catalogues._CATALOGUES
    catalogues._CATALOGUES = real_root
    try:
        text.set_active_locale("en")
    finally:
        catalogues._CATALOGUES = patched


@pytest.fixture
def fr_fixture(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A shipped-looking ``fr`` whose values are the English source behind a marker."""
    real = catalogues.catalogue_root()
    for locale in ("en", "fr"):
        (tmp_path / locale).mkdir()
    for namespace in NAMESPACES:
        source = json.loads((real / "en" / f"{namespace}.json").read_text(encoding="utf-8"))
        (tmp_path / "en" / f"{namespace}.json").write_text(json.dumps(source), encoding="utf-8")
        (tmp_path / "fr" / f"{namespace}.json").write_text(
            json.dumps({key: "FR:" + value for key, value in source.items()}), encoding="utf-8"
        )
    monkeypatch.setattr(catalogues, "_CATALOGUES", tmp_path)
    monkeypatch.setattr(resolve_mod, "shipped_locales", lambda: ("en", "fr"))


def test_english_is_byte_identical_to_the_pre_keyed_literals() -> None:
    """English parity: keying the copy must not move one byte of the English output."""
    assert completion_notice("closed", "ignored") == (
        "Completed — runtime retired/disposed",
        "info",
    )
    assert completion_notice("retired", "ignored") == (
        "Retired for an update — a turn was in flight and was cut; its earlier output is kept",
        "warning",
    )
    assert notify.BODIES["closed"] == "Completed — runtime retired/disposed"
    assert notify.BODIES["retired"] == (
        "Retired for an update — a turn was in flight and was cut; its earlier output is kept"
    )
    assert push_payload.count_phrase(1) == "1 conversation needs you"
    assert push_payload.count_phrase(2) == "2 conversations need you"
    # Digits stay raw (no ICU grouping) — the phrase has always printed them so.
    assert push_payload.count_phrase(9999) == "9999 conversations need you"
    assert push_payload.count_phrase(0) == "0 conversations need you"


def test_a_live_language_switch_moves_every_surface_off_english(fr_fixture: None) -> None:
    """The reported defect: no surface may keep the English sentence after a move."""
    english = {
        "closed": notice_copy.closed_notice_text(),
        "retired": notice_copy.retired_notice_text(),
        "count": push_payload.count_phrase(5),
    }
    # The dict the composer and the push builder hold is the SAME object the
    # refresher must update in place; rebinding it would strand them in English.
    assert COMPOSE_BODIES is notify.BODIES
    assert push_payload.BODIES is notify.BODIES

    text.set_active_locale("fr")
    resolver = text.tr("wire.outcome_notices")
    want_closed = resolver(_k.closed())
    want_retired = resolver(_k.retired())
    assert want_closed.startswith("FR:") and want_retired.startswith("FR:")

    # Row surfaces (TUI poller, phone projection frame, desktop notice).
    assert completion_notice("closed") == (want_closed, "info")
    assert completion_notice("retired") == (want_retired, "warning")
    # Banner vocabulary, including the by-reference consumers.
    assert notify.BODIES["closed"] == notify.BODY_CLOSED == want_closed
    assert notify.BODIES["retired"] == notify.BODY_RETIRED == want_retired
    assert COMPOSE_BODIES["retired"] == want_retired
    # Phone push: outcome phrase AND count clause both follow.
    want_count = text.tr("wire.push_alert")(_kp.attention_count(total=5, count="5"))
    assert want_count.startswith("FR:") and "5" in want_count
    alert = push_payload.completion_alert("retired", 5)[push_payload.ALERT_BODY_FIELD]
    assert alert == f"{want_retired} · {want_count}"
    assert push_payload.count_phrase(5) == want_count
    assert want_closed != english["closed"]
    assert want_retired != english["retired"]
    assert want_count != english["count"]

    # And back: the move home restores English everywhere (no stale constant).
    text.set_active_locale("en")
    assert notify.BODIES["retired"] == english["retired"]
    assert notify.BODIES["closed"] == english["closed"]
    assert push_payload.count_phrase(5) == english["count"]
    assert completion_notice("retired")[0] == english["retired"]


def test_missing_locale_keys_fall_back_to_english_not_codes() -> None:
    """A shipped locale that does not carry the namespace yet reads English source.

    This is the state between this change and the translation/audit wave: the
    real ``fr`` (shipped) has no ``wire.outcome_notices`` file at all.
    """
    english = notice_copy.retired_notice_text()
    text.set_active_locale("fr")
    assert notice_copy.retired_notice_text() == english
    assert notify.BODIES["retired"] == english
    assert "wire.outcome_notices" not in push_payload.count_phrase(3)


def test_context_sidecars_state_the_constraints_translators_need() -> None:
    """The retired sentence must keep the cut-by-update and kept-output facts."""
    root = catalogues.catalogue_root() / "en"
    context = json.loads((root / "wire.outcome_notices.context.json").read_text(encoding="utf-8"))
    source = json.loads((root / "wire.outcome_notices.json").read_text(encoding="utf-8"))
    retired = context["wire.outcome_notices.retired"]
    note = retired["note"].lower()
    assert "update" in note and "cut" in note and "kept" in note
    for key, entry in context.items():
        assert entry["role"] and entry["tone"]
        assert entry["maxCells"] >= len(source[key]), key
    push = json.loads((root / "wire.push_alert.context.json").read_text(encoding="utf-8"))
    entry = push["wire.push_alert.attention_count"]
    assert "{total}" in entry["note"] and "{count}" in entry["note"]
