"""Ratchet slice 3 (`tui.small`): ask_queue / copy_picker / link_picker copy.

The fixture beside this file is a SNAPSHOT of the nine literals this slice
extracted, transcribed from the pre-slice source on `origin/main` @ `1bca880ac`
— the committed form of the byte-identical proof review and QA can re-run.
Each test drives the REAL call path (the module functions and the widgets'
own card builders) or the generated key function and requires the rendered
string to match the golden EXACTLY.

Byte fidelity is the contract, and it is non-obvious in three places:

* ``ask_queue`` compact clauses and the two long ``question_*`` clauses print the
  count as a RAW digit string (``{shown}``), never through the number formatter:
  ``1200`` stays ``1200`` and never becomes ``1,200``. The ``question_*`` keys
  add an ICU plural on a SEPARATE ``count`` argument that selects the branch
  only — the rendered number is still ``{shown}``.
* ``copy_picker.more_lines`` is the same shape, and writes the ellipsis and the
  space as part of the value; ``from_line`` keeps its LEADING ``" · "`` because
  it is appended to the same Rich Text row as the line count.
* ``picker.showing_prefix``/``picker.of_separator`` are REUSED from the merged
  `tui.small` slice (the ask and move pickers' window counter), not re-created:
  the link picker's counter is the same shape, so one vocabulary serves it.

``link_picker._card_width`` reads the title through the SAME key it paints (a
one-source-of-truth invariant the file's docstring records), so the measurement
stays in lockstep with the copy instead of measuring a second literal.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_operator.i18n import catalogues, runtime
from local_operator.i18n.keys import tui_small as small_copy
from local_operator.i18n.text import tr
from local_operator.tui.copy_targets import CopyTarget
from local_operator.tui.link_targets import LinkTarget
from local_operator.tui.widgets import ask_queue
from local_operator.tui.widgets.copy_picker import CopyPickerScreen
from local_operator.tui.widgets.link_picker import LinkPickerScreen

FIXTURE = Path(__file__).parent / "fixtures" / "tui_next_small_ratchet3_en_golden.json"
GOLDEN: dict[str, str] = json.loads(FIXTURE.read_text(encoding="utf-8"))
REPO = Path(__file__).resolve().parents[3]

#: The three files this slice activated, exact paths (never a directory prefix:
#: sibling files in `tui/widgets/` still carry literals, so a prefix would
#: red-CI the whole tree for every session).
ACTIVATED = (
    "local_operator/tui/widgets/ask_queue.py",
    "local_operator/tui/widgets/copy_picker.py",
    "local_operator/tui/widgets/link_picker.py",
)

#: The three files #2491 activated (still open at this slice's base, merged
#: before it) — pinned so this slice's baseline edit cannot walk them back.
RETAINED = (
    "local_operator/tui/widgets/projects_form.py",
    "local_operator/tui/widgets/subagent_panel.py",
    "local_operator/tui/widgets/aside_panel.py",
)

_small = tr("tui.small")


def _en(key: str, **params: object) -> str:
    """The message rendered from the FIXTURE source, never from the live file."""
    return runtime.render_message(GOLDEN[key], params, "en")


def test_the_new_keys_match_the_pre_slice_golden() -> None:
    assert len(GOLDEN) == 9
    live = catalogues.load_catalogue("en", "tui.small")
    for key, value in GOLDEN.items():
        assert live[key] == value, key


def test_the_sidecar_names_role_tone_and_note_for_every_new_key() -> None:
    sidecar = catalogues.context("en", "tui.small")
    for key in GOLDEN:
        assert sidecar[key]["role"] and sidecar[key]["tone"] and sidecar[key]["note"], key


def test_ask_queue_counts_keep_raw_digits_and_no_grouping() -> None:
    # 1 matters (no "1 questions"), 1200 matters (no "1,200"): the pre-slice
    # f-strings printed the number via str() at every count.
    for count in (1, 2, 11, 999, 1000, 1200, 100000):
        shown = str(count)
        assert _small(small_copy.ask_queue_waiting_clause(shown=shown)) == f"{count} waiting"
        assert _small(small_copy.ask_queue_timed_out_clause(shown=shown)) == f"{count} timed out"
        assert _small(small_copy.ask_queue_urgent_clause(shown=shown)) == f"{count} urgent"
        assert _small(small_copy.ask_queue_waiting_clause(shown=shown)) == _en(
            "tui.small.ask_queue.waiting_clause", shown=shown
        )


def test_ask_queue_headline_plural_matches_the_pre_slice_noun_choice() -> None:
    # The pre-slice code wrote `"question" if n == 1 else "questions"`; the ICU
    # plural must reproduce that branch exactly, with `{shown}` still raw.
    for count in (1, 2, 11, 1000, 1200, 100000):
        noun = "question" if count == 1 else "questions"
        shown = str(count)
        assert (
            _small(small_copy.ask_queue_question_waiting(count=count, shown=shown))
            == f"{count} {noun} waiting"
        )
        assert (
            _small(small_copy.ask_queue_question_timed_out(count=count, shown=shown))
            == f"{count} {noun} timed out"
        )


def test_the_real_headlines_assemble_to_the_pre_slice_bytes() -> None:
    assert ask_queue.compact_headline(2, 1) == "2 waiting · 1 timed out"
    assert ask_queue.compact_headline(1, 0) == "1 waiting"
    assert ask_queue.queue_headline(2, 3) == "2 questions waiting · 3 questions timed out"
    assert ask_queue.queue_headline(1, 0) == "1 question waiting"
    assert ask_queue.queue_headline(12, 0, 5) == "12 questions waiting · 5 urgent"
    # A large count renders ungrouped, the pre-slice bytes.
    assert ask_queue.queue_headline(1000, 1200) == (
        "1000 questions waiting · 1200 questions timed out"
    )


def test_the_picker_titles_and_counter_words_match_the_pre_slice_bytes() -> None:
    assert _small(small_copy.copy_picker_title()) == "Copy to clipboard"
    assert _small(small_copy.link_picker_title()) == "Open a link"
    # REUSED from the merged slice: the link picker's counter is the same
    # shape as the ask/move pickers'.
    assert _small(small_copy.picker_showing_prefix()) == "showing "
    assert _small(small_copy.picker_of_separator()) == " of "


def test_copy_picker_markers_match_the_pre_slice_bytes() -> None:
    # `from_line` keeps its leading " · " (appended to the count's Rich Text
    # row) and its raw digit line number.
    for offset in (1, 2, 3, 1199, 1200):
        assert _small(small_copy.copy_picker_from_line(line=str(offset))) == (
            f" \u00b7 from line {offset}"
        )
    for count in (1, 2, 11, 1000, 1200, 100000):
        noun = "line" if count == 1 else "lines"
        assert _small(small_copy.copy_picker_more_lines(count=count, shown=str(count))) == (
            f"\u2026 {count} more {noun}"
        )


def test_link_picker_card_paints_the_title_and_counter() -> None:
    one = LinkPickerScreen([LinkTarget(url="https://a.test/1", sender="you", rank=0)])
    assert one._card_text().plain.splitlines()[0] == "Open a link"

    many = LinkPickerScreen(
        [LinkTarget(url=f"https://a.test/{index}", sender="you", rank=index) for index in range(40)]
    )
    counter = [line for line in many._card_text().plain.splitlines() if "showing" in line]
    assert counter == ["showing 1\u201311 of 40"]
    # The width measurement reads the same key the title paints, so a title
    # change cannot leave the card sized against a stale literal.
    assert many._card_width() >= len("Open a link")


def test_copy_picker_card_paints_the_title_and_markers() -> None:
    preview = "\n".join(f"line {index}" for index in range(120))
    target = CopyTarget(
        id="msg:1:code:0",
        label="Block 1",
        hint="3 code",
        preview=preview,
        content="x",
        copy_message="",
        language=None,
    )
    screen = CopyPickerScreen([target])
    # The card is built through `_card_text`, the same real render path the
    # widget paints and `render_lines_for_test` reads back.
    lines = screen._card_text().plain.splitlines()
    assert lines[0] == "Copy to clipboard"
    assert any(line.startswith("\u2026 ") and line.endswith(" more lines") for line in lines)
    # `from line` only appears once the preview is scrolled off the top: the
    # header claims no position at offset 0, so drive the state under test.
    screen._preview_offset = 2
    header = next(line for line in screen._card_text().plain.splitlines() if "Preview" in line)
    assert header == "Preview \u00b7 3 code \u00b7 from line 3"


def test_every_activated_file_lost_its_ceiling() -> None:
    """The toothlessness guard: enforced WITHOUT a ceiling, exact path only.

    A file left in ``files`` while also in ``enforced`` still absorbs a
    planted literal into its ceiling, so the gate silently does nothing.
    """
    baseline = json.loads((REPO / "i18n" / "baseline.json").read_text(encoding="utf-8"))
    for path in ACTIVATED:
        assert path in baseline["enforced"], f"{path} is not activated"
        assert path not in baseline["files"], f"{path} kept its ceiling; the gate is toothless"
    # #2491's activated files are review state this slice must not walk back.
    for path in RETAINED:
        assert path in baseline["enforced"], f"{path} lost its activation"
        assert path not in baseline["files"], f"{path} regained a ceiling"
    assert not [entry for entry in baseline["enforced"] if entry.endswith("/") and "tui" in entry]


@pytest.mark.parametrize("path", ACTIVATED)
def test_a_planted_literal_fails_the_activated_checker(
    path: str, tmp_path, monkeypatch, i18n_check
) -> None:
    """Prove the activation BITES: the real checker entrypoint, isolated copy."""
    source = (REPO / path).read_text(encoding="utf-8")
    target = tmp_path / path
    target.parent.mkdir(parents=True)
    target.write_text(source, encoding="utf-8")
    baseline_dir = tmp_path / "i18n"
    baseline_dir.mkdir()
    baseline = baseline_dir / "baseline.json"
    baseline.write_text(json.dumps({"note": "slice fixture", "enforced": [path], "files": {}}))
    allowlist = baseline_dir / "allowlist.toml"
    allowlist.write_text("entries = []\n")
    monkeypatch.setattr(i18n_check, "REPO", tmp_path)
    monkeypatch.setattr(i18n_check, "BASELINE", baseline)
    monkeypatch.setattr(i18n_check, "ALLOWLIST", allowlist)
    monkeypatch.setattr(i18n_check, "check_catalogues", lambda: [])
    assert i18n_check.main([]) == 0
    target.write_text(source + '\nprint("planted slice prose")\n', encoding="utf-8")
    assert i18n_check.main([]) == 1
    target.write_text(source, encoding="utf-8")
    assert i18n_check.main([]) == 0
