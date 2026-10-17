"""Ratchet slice 10: three widget files, two shared namespaces.

`todo_panel` (14) and `command_picker` (2) extend **`tui.small`** — the shared
widget-chrome namespace several panels already read — while `projects_start`
(4) extends **`tui.projects`**, the namespace the projects_* family owns. None
of the three owned a namespace before, so this is the "extend the namespace the
feature owns" rule; neither file needed a new one.

`session_sidebar` is deliberately NOT in this slice: five of its six findings
are the rung ladder's composition templates (`f"{pin}{key}{{t}}"` and friends),
which carry no words at all — the words live in assignments beside them
(`key = " · ctrl+f asks"`). Whether that family is display vocabulary worth an
`# i18n: ignore` reason or a whole-set extraction is a CLASS decision, and it
gets its own slice rather than a rushed call at the end of this one. Its
word-carrying finding (`opened by {role}`) rides with it.

Two renders here are worth naming, because they are the file's own comment
materialised as keys:

* **The padded scroll cue.** `f"↓ {remaining:>{len(str(total))}} more · "` pads
  the digit so the hotkey's column does not jump at every power-of-ten
  boundary. The key takes the ALREADY-PADDED string (`count` is a string, never
  a number), so the padding survives translation instead of being re-derived;
  its end-state sibling passes a run of spaces the same width.
* **The two-valued switches.** collapse/expand, more/done and blocked/dropped
  are pairs; each half is its own key so no locale has to reconstruct a
  conditional from a fragment.
"""

from __future__ import annotations

import json
from pathlib import Path

from local_operator.i18n import catalogues
from local_operator.i18n.keys import tui_projects as _p
from local_operator.i18n.keys import tui_small as _x
from local_operator.i18n.text import tr

REPO = Path(__file__).resolve().parents[3]
_small = tr("tui.small")
_projects = tr("tui.projects")

ACTIVATED = (
    "local_operator/tui/widgets/todo_panel.py",
    "local_operator/tui/widgets/command_picker.py",
    "local_operator/tui/widgets/projects_start.py",
)

RETAINED = (
    "local_operator/tui/widgets/settings_view.py",
    "local_operator/tui/widgets/usage_panel.py",
    "local_operator/tui/projects_render.py",
)

#: Activated by slice 9, which was still OPEN when this slice was cut, so on this
#: base it is expected to still carry its ceiling. Only the contradictory state
#: is forbidden: enforced AND ceilinged at once. Once slice 9 merges this reads
#: enforced-and-ceiling-free, which is also fine.
SLICE_NINE = "local_operator/tui/app.py"

SMALL_BASE = {
    "tui.small.todo_panel.header": "Todos",
    "tui.small.todo_panel.resolved": "{resolved}/{total} resolved",
    "tui.small.todo_panel.dropped": "{count} dropped",
    "tui.small.todo_panel.more_todos": "… {count} more todos",
    "tui.small.todo_panel.stage": " · stage {stage}/{total}",
    "tui.small.todo_panel.more_hidden": "+{count} more · ",
    "tui.small.todo_panel.done_hidden": "+{count} done · ",
    "tui.small.todo_panel.collapse": "ctrl+t to collapse",
    "tui.small.todo_panel.expand": "ctrl+t to expand",
    "tui.small.todo_panel.more_below": "↓ {count} more · ",
    "tui.small.todo_panel.more_below_padded": "↓ {pad} more · ",
    "tui.small.todo_panel.blocked": " — blocked",
    "tui.small.todo_panel.dropped_row": " — dropped",
    "tui.small.command_picker.projects": (
        "{count, plural, one {{shown} project} other {{shown} projects}}"
    ),
    "tui.small.command_picker.entries": (
        "{count, plural, one {{shown} entry} other {{shown} entries}}"
    ),
}

PROJECTS_BASE = {
    "tui.projects.start.title": "start session",
    "tui.projects.start.filter_placeholder": "type to filter",
    "tui.projects.start.legend": "type to filter · ↑↓ move · ↵ select · esc close",
    "tui.projects.start.more": "+{count} more",
}


def test_the_new_keys_are_exactly_their_base_bytes() -> None:
    live_small = catalogues.load_catalogue("en", "tui.small")
    for key, value in SMALL_BASE.items():
        assert live_small[key] == value, key
    live_projects = catalogues.load_catalogue("en", "tui.projects")
    for key, value in PROJECTS_BASE.items():
        assert live_projects[key] == value, key
    for namespace, mapping in (("tui.small", SMALL_BASE), ("tui.projects", PROJECTS_BASE)):
        sidecar = catalogues.context("en", namespace)
        for key in mapping:
            entry = sidecar[key]
            assert all(entry.get(field) for field in ("role", "tone", "note")), key


def test_the_padded_scroll_cue_keeps_the_digit_column_pinned() -> None:
    # Was: f"↓ {total} more · " (plain) and f"↓ {remaining:>{len(str(total))}} more · "
    # (right-aligned) and f"↓ {'':>{len(str(total))}} more · " (end state).
    assert _small(_x.todo_panel_more_below(count="10")) == "↓ 10 more · "
    assert _small(_x.todo_panel_more_below(count=f"{9:>{len(str(10))}}")) == "↓  9 more · "
    assert _small(_x.todo_panel_more_below_padded(pad="".rjust(len(str(10))))) == "↓    more · "
    # The column is pinned: both digits end at the same cell.
    assert _small(_x.todo_panel_more_below(count="10")).index(" m") == _small(
        _x.todo_panel_more_below(count=f"{9:>{len(str(10))}}")
    ).index(" m")


def test_the_count_keys_render_their_plurals_with_raw_digits() -> None:
    # Was: f"{projects} project" if projects == 1 else f"{projects} projects"
    assert _small(_x.command_picker_projects(count=1, shown="1")) == "1 project"
    assert _small(_x.command_picker_projects(count=4, shown="4")) == "4 projects"
    assert _small(_x.command_picker_entries(count=1, shown="1")) == "1 entry"
    assert _small(_x.command_picker_entries(count=4, shown="4")) == "4 entries"
    for count in (1200, 10000):
        rendered = _small(_x.command_picker_projects(count=count, shown=str(count)))
        assert rendered == f"{count} projects"
        assert "," not in rendered


def test_the_two_valued_pairs_are_whole_strings() -> None:
    assert _small(_x.todo_panel_collapse()) == "ctrl+t to collapse"
    assert _small(_x.todo_panel_expand()) == "ctrl+t to expand"
    assert _small(_x.todo_panel_more_hidden(count="3")) == "+3 more · "
    assert _small(_x.todo_panel_done_hidden(count="3")) == "+3 done · "
    assert _small(_x.todo_panel_blocked()) == " — blocked"
    assert _small(_x.todo_panel_dropped_row()) == " — dropped"
    # Header and counters, including the shared "Todos" key (two sites).
    assert _small(_x.todo_panel_header()) == "Todos"
    assert _small(_x.todo_panel_resolved(resolved="2", total="5")) == "2/5 resolved"
    assert _small(_x.todo_panel_dropped(count="1")) == "1 dropped"
    assert _small(_x.todo_panel_more_todos(count="7")) == "… 7 more todos"
    assert _small(_x.todo_panel_stage(stage="2", total="4")) == " · stage 2/4"
    # The projects card.
    assert _projects(_p.start_title()) == "start session"
    assert _projects(_p.start_filter_placeholder()) == "type to filter"
    assert _projects(_p.start_legend()) == "type to filter · ↑↓ move · ↵ select · esc close"
    assert _projects(_p.start_more(count="5")) == "+5 more"


def test_the_activated_files_lost_their_ceilings() -> None:
    baseline = json.loads((REPO / "i18n" / "baseline.json").read_text(encoding="utf-8"))
    for path in ACTIVATED:
        assert path in baseline["enforced"], f"{path} is not activated"
        assert path not in baseline["files"], f"{path} kept its ceiling"
    for path in RETAINED:
        assert path in baseline["enforced"], f"{path} lost its activation"
        assert path not in baseline["files"], f"{path} regained a ceiling"
    assert not (
        SLICE_NINE in baseline["enforced"] and SLICE_NINE in baseline["files"]
    ), f"{SLICE_NINE} is both enforced and ceilinged"
    for path in ("local_operator/tui/widgets/session_sidebar.py",):
        assert path not in baseline["enforced"], f"{path} was activated without its class decision"


def test_a_planted_literal_fails_the_activated_checker(tmp_path, monkeypatch, i18n_check) -> None:
    source = (REPO / ACTIVATED[0]).read_text(encoding="utf-8")
    target = tmp_path / ACTIVATED[0]
    target.parent.mkdir(parents=True)
    target.write_text(source, encoding="utf-8")
    baseline_dir = tmp_path / "i18n"
    baseline_dir.mkdir()
    baseline = baseline_dir / "baseline.json"
    baseline.write_text(
        json.dumps({"note": "slice fixture", "enforced": [ACTIVATED[0]], "files": {}})
    )
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
