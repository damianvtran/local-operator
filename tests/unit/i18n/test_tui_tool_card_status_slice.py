"""The tool card's status words, keyed under ``tui.tool_card.status``.

WHY THIS FILE EXISTS. The operator's goal is that switching language leaves no
English in the interface. After the card's chrome moved to ``tui.tool_card``
(``test_tui_tool_card_slice.py``), the words that say what a row IS still
painted English: the status column (``running``/``queued``/``waiting``), the
reason slot (``Partial``, ``session reset``, ``interrupted``/``cut off``, the
``error``/``not run`` fallbacks, ``not delivered: …``) and the send row's
outcome words (``wake unconfirmed``, ``delivery unconfirmed``, ``receipt
reused · no new wake``, the model-switch words). They live in a CHILD
namespace so ``tui.tool_card``'s audited bytes do not move.

Three claims, each driven through the REAL paint paths (``_status_runs`` and
``_build_row``), never the catalogue:

1. English is byte-identical to the pre-extraction literals, spelled out here
   on purpose — reading them back from the catalogue would let a wording change
   slip through as "still equal".
2. With a synthetic ``fr`` carrying only this namespace, every status surface
   paints the fr word after a LIVE language change, settled rows included.
3. The status-column caps measure terminal CELLS: a wide (CJK) word is shed
   when it does not fit, a long word appears exactly when the cap holds its
   cell width, and a shorter running word is padded to the settled spine. A
   mutation control proves the cell check is what decides (``len`` would not).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from rich.cells import cell_len

from local_operator.i18n import catalogues, resolve, text
from local_operator.tui.widgets import tool_card
from local_operator.tui.widgets.tool_card import DURATION_COL, ICON_SUCCESS, ToolCard

NS = "tui.tool_card.status"
EN_ROOT = Path(catalogues.__file__).resolve().parent / "catalogues"
BASE: dict[str, str] = json.loads((EN_ROOT / "en" / f"{NS}.json").read_text("utf-8"))


def _status(card: ToolCard, cap: int = 0) -> list[str]:
    return [run for run, _style in card._status_runs(cap)]


def _surfaces() -> dict[str, object]:
    """Every keyed status surface, read from what the card actually paints."""
    out: dict[str, object] = {}

    running = ToolCard("r", "wait", {"job_id": "7a73c97ffc54"})
    running.restore(state="running")  # no start known -> the word, not a clock
    out["running"] = _status(running)

    queued = ToolCard("q", "wake", {"text": "30m"})
    queued.set_composing(14, "wake")
    queued.mark_queued()
    out["queued"] = _status(queued)

    waiting = ToolCard("w", "bash", {"command": "rm -rf build"})
    waiting.mark_waiting()
    out["waiting"] = _status(waiting)

    partial = ToolCard("p", "send", {"pid": 1, "message": "hi"})
    partial.mark_done("ok", {"partial_result": True})
    out["partial"] = partial._build_row(80).plain

    reset = ToolCard("k", "eval", {"code": "1 + 1"})
    reset.mark_done("result: 2", {"kernel_reset": True})
    out["reset"] = reset._build_row(100).plain

    stopped = ToolCard("i", "bash", {"command": "sleep 9"})
    stopped.mark_interrupted()
    out["interrupted"] = stopped._build_row(80).plain
    cut = ToolCard("c", "bash", {"command": "sleep 9"})
    cut.mark_interrupted(cut_off=True)
    out["cut_off"] = cut._build_row(80).plain

    failed = ToolCard("e", "bash", {"command": "false"})
    failed.mark_failed("")
    out["error"] = failed._build_row(80).plain
    restored = ToolCard("e2", "bash", {"command": "false"})
    restored.restore(state="error")
    out["error_restored"] = restored._build_row(80).plain
    never = ToolCard("n", "bash", {"command": "x"})
    never.mark_not_run("")
    out["not_run"] = never._build_row(80).plain

    lost = ToolCard("f", "send", {"target": "ghost-session", "message": "hi"})
    lost.mark_failed(
        "could not deliver to ghost-session: refused",
        details={"delivery": {"state": "failed", "reason": "this session cannot receive"}},
    )
    out["not_delivered"] = lost._build_row(200).plain

    for state in ("mailbox", "unconfirmed"):
        send = ToolCard("s", "send", {"target": "release-owner", "message": "gates are green"})
        send.mark_done("receipt", {"delivery": {"state": state}, "partial_result": True})
        out[state] = send._build_row(100).plain

    reused = ToolCard("u", "send", {"target": "release-owner", "message": "same note"})
    reused.mark_done(
        "duplicate suppressed",
        {
            "duplicate_suppressed": True,
            "original_receipt": {"details": {"wake": True, "delivery": {"state": "delivered"}}},
        },
    )
    out["reused"] = reused._build_row(100).plain

    for outcome in ("switched", "unchanged", "pending", "partial"):
        switch = ToolCard("m", "send", {"pid": 48213, "model": "deepseek/deepseek-flash"})
        details: dict[str, object] = {"outcome": outcome}
        if outcome == "partial":
            details["partial_result"] = True
        switch.mark_done("receipt", details)
        out[f"switch_{outcome}"] = switch._build_row(100).plain
    return out


def test_english_bytes_match_the_pre_extraction_literals() -> None:
    got = _surfaces()
    assert got["running"] == ["running"]
    assert got["queued"] == ["queued"]
    assert got["waiting"] == ["waiting"]
    assert "Partial" in got["partial"]
    assert "session reset" in got["reset"]
    assert "interrupted" in got["interrupted"]
    assert "cut off" in got["cut_off"] and "interrupted" not in got["cut_off"]
    assert " error " in got["error"]
    assert " error " in got["error_restored"]
    assert " not run " in got["not_run"]
    assert "not delivered: this session cannot receive" in got["not_delivered"]
    assert "wake unconfirmed · release-owner" in got["mailbox"]
    assert "delivery unconfirmed · release-owner" in got["unconfirmed"]
    assert "receipt reused · no new wake · release-owner" in got["reused"]
    assert "switched · pid 48213" in got["switch_switched"]
    assert "no change · pid 48213" in got["switch_unchanged"]
    assert "pending · pid 48213" in got["switch_pending"]
    assert "switched (error) · pid 48213" in got["switch_partial"]
    # The legacy constant names other modules import still give the English.
    assert tool_card.RUNNING_LABEL == "running"
    assert tool_card.QUEUED_LABEL == "queued"
    assert tool_card._PARTIAL_LABEL == "Partial"
    assert tool_card._RESET_LABEL == "session reset"
    assert tool_card.SEND_NOT_DELIVERED_WORD == "not delivered"
    assert tool_card.DELIVERY_OUTCOME_WORDS == {
        "mailbox": "wake unconfirmed",
        "unconfirmed": "delivery unconfirmed",
    }
    assert tool_card.SWITCH_OUTCOME_WORDS == {
        "switched": "switched",
        "unchanged": "no change",
        "pending": "pending",
        "partial": "switched (error)",
    }
    # The spine equality the running word was sized to still holds in English.
    assert cell_len(tool_card.RUNNING_LABEL) == cell_len(f"{ICON_SUCCESS} ") + DURATION_COL


def _install_fr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, values: dict[str, str]) -> None:
    # A COPY, not a symlink: `catalogues._file` resolves paths and refuses one
    # that leaves the root, so a symlinked `en` silently loses every English
    # catalogue and the fallback paints raw key IDs instead of English.
    shutil.copytree(
        EN_ROOT / "en", tmp_path / "en", ignore=shutil.ignore_patterns("*.context.json")
    )
    (tmp_path / "fr").mkdir()
    (tmp_path / "fr" / f"{NS}.json").write_text(json.dumps(values, ensure_ascii=False), "utf-8")
    monkeypatch.setattr(catalogues, "_CATALOGUES", tmp_path)
    monkeypatch.setattr(text, "_active_locale", "en")
    monkeypatch.setattr(text, "_refreshers", [])
    monkeypatch.setattr(resolve, "shipped_locales", lambda: ("en", "fr"))
    monkeypatch.delenv("LOP_I18N", raising=False)
    text._sources.cache_clear()


@pytest.fixture
def fr_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Only this fixture may make a synthetic locale eligible, never production."""
    _install_fr(tmp_path, monkeypatch, {k: f"FR {v}" for k, v in BASE.items()})
    yield
    text.set_active_locale("en")
    text._sources.cache_clear()


#: Synthetic words chosen for their geometry, not their meaning: a 3-char
#: 6-cell CJK running word (len and cell width disagree) and a 10-cell waiting
#: word longer than the 7-cell spine.
_WIDE_RUNNING = "実行中"
_LONG_WAITING = "en attente"


@pytest.fixture
def fr_geometry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    values = dict(BASE)
    values[f"{NS}.running"] = _WIDE_RUNNING
    values[f"{NS}.queued"] = _WIDE_RUNNING
    values[f"{NS}.waiting"] = _LONG_WAITING
    _install_fr(tmp_path, monkeypatch, values)
    text.set_active_locale("fr")
    yield
    text.set_active_locale("en")
    text._sources.cache_clear()


def test_every_status_surface_follows_a_live_language_change(fr_status) -> None:
    english = _surfaces()  # built AND painted in English first
    text.set_active_locale("fr")
    got = _surfaces()
    assert got["running"] == ["FR running"]
    assert got["queued"] == ["FR queued"]
    assert got["waiting"] == ["FR waiting"]
    assert "FR Partial" in got["partial"]
    assert "FR session reset" in got["reset"]
    assert "FR interrupted" in got["interrupted"]
    assert "FR cut off" in got["cut_off"]
    assert " FR error " in got["error"]
    assert " FR error " in got["error_restored"]
    assert " FR not run " in got["not_run"]
    assert "FR not delivered: this session cannot receive" in got["not_delivered"]
    assert "FR wake unconfirmed · release-owner" in got["mailbox"]
    assert "FR delivery unconfirmed · release-owner" in got["unconfirmed"]
    assert "FR receipt reused · FR no new wake · release-owner" in got["reused"]
    assert "FR switched · pid 48213" in got["switch_switched"]
    assert "FR no change · pid 48213" in got["switch_unchanged"]
    assert "FR pending · pid 48213" in got["switch_pending"]
    assert "FR switched (error) · pid 48213" in got["switch_partial"]
    # The partial arm's no-double-word guard still recognises the keyed switch
    # word: a translated `switched (error)` row must not grow a `Partial` too.
    assert "FR Partial" not in got["switch_partial"]
    assert got != english
    assert tool_card.DELIVERY_OUTCOME_WORDS["mailbox"] == "FR wake unconfirmed"


def test_a_settled_row_repaints_on_a_language_change(fr_status) -> None:
    """The word is resolved at paint time, not frozen when the row settled."""
    card = ToolCard("c", "bash", {"command": "sleep 9"})
    card.mark_interrupted(cut_off=True)
    failed = ToolCard("e", "bash", {"command": "false"})
    failed.mark_failed("")
    assert "cut off" in card._build_row(80).plain
    text.set_active_locale("fr")
    assert "FR cut off" in card._build_row(80).plain
    assert " FR error " in failed._build_row(80).plain


def test_status_caps_measure_cells_not_code_points(fr_geometry) -> None:
    running = ToolCard("r", "wait", {"job_id": "x"})
    running.restore(state="running")
    # Padded to the settled spine so the row does not reflow on settling.
    spine = cell_len(f"{ICON_SUCCESS} ") + DURATION_COL
    padded = " " * (spine - cell_len(_WIDE_RUNNING)) + _WIDE_RUNNING
    assert _status(running) == [padded]
    assert cell_len(_status(running)[0]) == spine
    assert _status(running, cap=spine) == [padded]
    # One cell short of the padded word: shed whole, never clipped.
    assert _status(running, cap=spine - 1) == [""]

    queued = ToolCard("q", "wake", {"text": "30m"})
    queued.set_composing(14, "wake")
    queued.mark_queued()
    # 3 code points but 6 cells: a 5-cell cap must shed it (len would keep it).
    assert _status(queued, cap=5) == [""]
    assert _status(queued, cap=6) == [_WIDE_RUNNING]

    waiting = ToolCard("w", "bash", {"command": "x"})
    waiting.mark_waiting()
    width = cell_len(_LONG_WAITING)
    assert _status(waiting, cap=width) == [_LONG_WAITING]
    assert _status(waiting, cap=width - 1) == [""]


def test_mutation_control_len_would_let_a_wide_word_overrun(
    fr_geometry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prove the cell assertion above can fail: with the measurement reverted to
    code points, the 6-cell word survives a 5-cell cap — the overrun the cell
    check exists to prevent."""
    queued = ToolCard("q", "wake", {"text": "30m"})
    queued.set_composing(14, "wake")
    queued.mark_queued()
    monkeypatch.setattr(tool_card, "cell_len", len)
    assert _status(queued, cap=5) == [_WIDE_RUNNING]
