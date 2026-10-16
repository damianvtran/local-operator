"""Ratchet slice 8: `projects_render` (tui.projects).

Extends `tui.projects` — the namespace the four `projects_*` widgets already
share (67 keys, now 91) — because this file renders the same project records
those widgets show; a fourth namespace or a jump to `tui.small` would have
put one feature's copy in two catalogues.

Every assertion is DIFFERENTIAL: the generated key through the real resolver
against the expression the site carried before, written out literally.

Two shapes worth naming, because they are what this file is made of:

* **Computed fragments travel as parameters.** `reported {age} ago{suffix}`,
  `progress reported {age} ago{suffix}: `, `filed by {ids}{more}` and
  `[{count} more {noun} — ↵ {verb}]` are assembled in code around their
  catalysed parts, so the fragments (` (stale)`, ` · stale`, ` by X`,
  ` +N more`, `line`/`lines`, `collapse`/`expand`) are English words built in
  ASSIGNMENTS — invisible to the scanner and NOT extracted here. They are
  recorded as this file's next tranche, in the same class as the `window(s)`
  plural slice 7 extracted by hand and the `_RESET_PREFIX` note QA raised.
* **Raw digits everywhere.** Every count is passed as a string so `1200`
  never groups to `1,200`.
"""

from __future__ import annotations

import json
from pathlib import Path

from local_operator.i18n import catalogues
from local_operator.i18n.keys import tui_projects as _p
from local_operator.i18n.text import tr

REPO = Path(__file__).resolve().parents[3]
_projects = tr("tui.projects")

#: The file this slice activated, exact path (never a directory prefix:
#: sibling files under `tui/widgets/` still carry literals).
ACTIVATED = "local_operator/tui/projects_render.py"

#: Slice 7's activated files, pinned so this slice's baseline edit cannot walk
#: them back.
RETAINED = (
    "local_operator/tui/widgets/settings_view.py",
    "local_operator/tui/widgets/usage_panel.py",
)

NEW_KEYS = (
    "tui.projects.render.more_not_shown_indent",
    "tui.projects.render.more_not_shown",
    "tui.projects.render.no_estimate_or_dates",
    "tui.projects.render.no_progress",
    "tui.projects.render.reported_ago",
    "tui.projects.render.live_count",
    "tui.projects.render.no_dates_count",
    "tui.projects.render.no_dates_label",
    "tui.projects.render.progress_none",
    "tui.projects.render.progress_reported",
    "tui.projects.render.no_linked_sessions",
    "tui.projects.render.working_prefix",
    "tui.projects.render.session_missing",
    "tui.projects.render.more_working",
    "tui.projects.render.filed_by",
    "tui.projects.render.no_progress_recorded",
    "tui.projects.render.running_settled",
    "tui.projects.render.todos_counts",
    "tui.projects.render.own_session",
    "tui.projects.render.open_over_total",
    "tui.projects.render.next_clause",
    "tui.projects.render.todo_snapshots_none",
    "tui.projects.render.more_lines_hint",
    "tui.projects.render.older_updates",
)


def test_every_new_key_exists_with_a_full_sidecar() -> None:
    live = catalogues.load_catalogue("en", "tui.projects")
    sidecar = catalogues.context("en", "tui.projects")
    assert len(NEW_KEYS) == 24
    for key in NEW_KEYS:
        assert key in live, key
        entry = sidecar[key]
        assert all(entry.get(field) for field in ("role", "tone", "note")), key


def test_overflow_rows_are_byte_identical_and_keep_their_own_indent() -> None:
    # Was: f"  … +{len(views) - PROJECTS_MAX} more not shown"  (board, indented)
    # and f"… +{len(views) - PROJECTS_MAX} more not shown"      (Text, not)
    assert _projects(_p.render_more_not_shown_indent(count="3")) == "  … +3 more not shown"
    assert _projects(_p.render_more_not_shown(count="3")) == "… +3 more not shown"


def test_the_shared_empty_state_is_one_key() -> None:
    # Was: "no estimate or dates" at BOTH call sites (the card's fact row and
    # the detail page's Text). One key, so the two cannot drift apart.
    assert _projects(_p.render_no_estimate_or_dates()) == "no estimate or dates"
    live = catalogues.load_catalogue("en", "tui.projects")
    assert [k for k in live if k.endswith("no_estimate_or_dates")] == [
        "tui.projects.render.no_estimate_or_dates"
    ]


def test_progress_states_are_byte_identical() -> None:
    # Was: "no progress" (card) vs "no progress recorded" (detail) — two keys,
    # because a value word and a sentence are not interchangeable.
    assert _projects(_p.render_no_progress()) == "no progress"
    assert _projects(_p.render_no_progress_recorded()) == "no progress recorded"
    assert _projects(_p.render_progress_none()) == "progress: none recorded"


def test_age_and_suffix_fragments_reassemble_to_the_original_rows() -> None:
    # Was: f"reported {age} ago{stale}"
    assert (
        _projects(_p.render_reported_ago(age="5h", suffix=" (stale)")) == "reported 5h ago (stale)"
    )
    assert _projects(_p.render_reported_ago(age="5h", suffix="")) == "reported 5h ago"
    # Was: f"progress reported {age} ago{reporter}{stale}{checked}: "
    row = _projects(
        _p.render_progress_reported(age="5h", suffix=" by A · stale · refreshed 1h ago by B")
    )
    assert row == "progress reported 5h ago by A · stale · refreshed 1h ago by B: "


def test_session_rows_are_byte_identical() -> None:
    assert _projects(_p.render_live_count(count="2")) == "2 live"
    assert _projects(_p.render_no_linked_sessions()) == "no linked sessions"
    assert _projects(_p.render_working_prefix()) == "working: "
    assert _projects(_p.render_own_session()) == "own session"
    # Was: f"{session_id} [missing]" — the identifier stays untranslated.
    assert _projects(_p.render_session_missing(session="s-42")) == "s-42 [missing]"
    assert _projects(_p.render_more_working(count="3")) == " · +3 more"
    # Was: f"filed by {filed_ids}{more}"
    assert _projects(_p.render_filed_by(ids="a, b", more=" +2 more")) == "filed by a, b +2 more"
    assert _projects(_p.render_filed_by(ids="a, b", more="")) == "filed by a, b"


def test_count_rows_keep_raw_digits_at_every_magnitude() -> None:
    for n in (1, 2, 1200, 10000):
        shown = str(n)
        assert _projects(_p.render_live_count(count=shown)) == f"{n} live"
        assert _projects(_p.render_older_updates(count=shown)) == f"… {n} older updates"
        assert _projects(_p.render_running_settled(running=shown, settled=shown)) == (
            f" · {n} running/{n} settled"
        )
        assert _projects(_p.render_todos_counts(open=shown, total=shown)) == f" · todos {n}/{n}"
        assert _projects(_p.render_open_over_total(open=shown, total=shown)) == f"{n} open / {n}"
        for rendered in (
            _projects(_p.render_live_count(count=shown)),
            _projects(_p.render_more_not_shown(count=shown)),
            _projects(_p.render_next_clause(joined=shown)),
        ):
            assert "," not in rendered


def test_list_tails_and_affordances_are_byte_identical() -> None:
    # Was: f"no dates ({len(undated)}): " and f"no dates ({label}, {n}): "
    assert _projects(_p.render_no_dates_count(count="4")) == "no dates (4): "
    assert (
        _projects(_p.render_no_dates_label(label="Design", count="4")) == "no dates (Design, 4): "
    )
    assert _projects(_p.render_todo_snapshots_none()) == "todo snapshots: none yet"
    # Was: f" — next: {joined}"
    assert _projects(_p.render_next_clause(joined="ship 1.0")) == " — next: ship 1.0"
    # Was: f"[{count} more {noun} — ↵ {verb}]"
    assert _projects(_p.render_more_lines_hint(count="5", noun="lines", verb="collapse")) == (
        "[5 more lines — ↵ collapse]"
    )


def test_the_untouched_projects_namespace_still_carries_its_older_keys() -> None:
    # The four projects_* widgets own this namespace; extending it must not
    # disturb them. Pin a sample from each, including this file's siblings.
    live = catalogues.load_catalogue("en", "tui.projects")
    assert live["tui.projects.send.no_match"] == "no matching targets — backspace to widen"
    for key in (
        "tui.projects.send.empty",
        "tui.projects.send.compose",
        "tui.projects.send.sent",
    ):
        assert key in live, key


def test_the_activated_file_lost_its_ceiling() -> None:
    """The toothlessness guard: enforced WITHOUT a ceiling, exact path only."""
    baseline = json.loads((REPO / "i18n" / "baseline.json").read_text(encoding="utf-8"))
    assert ACTIVATED in baseline["enforced"], f"{ACTIVATED} is not activated"
    assert ACTIVATED not in baseline["files"], f"{ACTIVATED} kept its ceiling"
    for path in RETAINED:
        assert path in baseline["enforced"], f"{path} lost its activation"
        assert path not in baseline["files"], f"{path} regained a ceiling"
    assert not [e for e in baseline["enforced"] if e.endswith("/") and "/tui/" in e]


def test_a_planted_literal_fails_the_activated_checker(tmp_path, monkeypatch, i18n_check) -> None:
    """Prove the activation BITES: the real checker entrypoint, isolated copy."""
    source = (REPO / ACTIVATED).read_text(encoding="utf-8")
    target = tmp_path / ACTIVATED
    target.parent.mkdir(parents=True)
    target.write_text(source, encoding="utf-8")
    baseline_dir = tmp_path / "i18n"
    baseline_dir.mkdir()
    baseline = baseline_dir / "baseline.json"
    baseline.write_text(json.dumps({"note": "slice fixture", "enforced": [ACTIVATED], "files": {}}))
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
