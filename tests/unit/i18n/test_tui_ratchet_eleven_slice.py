"""Ratchet slice 11: `session_sidebar` — including the CLASS DECISION.

**The decision, all six findings, one line each** (root's acceptance on #2547:
explicit, whole-set, and falsifiable by an independent stream):

| finding | literal content | class |
|---|---|---|
| `f"opened by {role}"` | words | **copy -> extracted** |
| `f"{position} · {pin}{key}{layer}{{t}}"` | separators + `{t}` | **ignored, reason on site** |
| `f"{position} · {pin}{key}{{t}}"` | same | **scaffolding -> ignored** |
| `f"{position} · {pin}{layer}{{t}}"` | same | **scaffolding -> ignored** |
| `f"{position} · {pin}{{t}}"` | same | **scaffolding -> ignored** |
| `f"{pin}{key}{layer}{{t}}"` | same | **scaffolding -> ignored** |

The five templates are not copy because every character of them is either a
separator or the `{t}` **format token**, which `expand()` substitutes
(`template.format(t=variant)`, session_sidebar.py:2752) — and ICU would parse a
bare `{t}` as an *argument*, so the family cannot be rushed into message keys at
all. The ignore reason names the documented category: keycap display vocabulary.

**But an ignore must not mask live English**, so the words the ladder carries in
ASSIGNMENTS — invisible to the collector, which is exactly why they were at risk
of being forgotten — are extracted in the same slice: `lead` (esc return / f9
focus), `pin_suffix`, `asks_suffix`, `hide_suffix`. `layer` (` · ctrl+a ⌥`) and
`key_short` (` · ctrl+f`) are word-free and stay as they are.

**What this slice still does NOT extract — and the list is NOT exhaustive**
(the review's sweep found more than my first draft named, which is the honest
correction): the `_tail(...)` composition's `ask_note`/`hub_note`/`chip`
contents; `AGENT_OPENED_MARK` ("agent-opened"), owned by
`local_operator/session/preview.py`; `"Refresh failed"` and `"Opening…"` in the
same `_footer_line` builder the ladder lives in; and file-wide residuals the
collector does not see at all — section headings, the `"+n more pinned — scroll"`
affordance, empty/loading/error states, `"on {device}"` / `" (last known state)"`
/ `"unreachable · …"`, and `WEDGED_REMEDY`. So this file activated at zero
findings while carrying English beyond its perimeter: it is not "done", and the
paragraph is a map rather than a certificate.
"""

from __future__ import annotations

import json
from pathlib import Path

from local_operator.i18n import catalogues
from local_operator.i18n.keys import tui_small as _x
from local_operator.i18n.text import tr

REPO = Path(__file__).resolve().parents[3]
_small = tr("tui.small")

ACTIVATED = "local_operator/tui/widgets/session_sidebar.py"

RETAINED = (
    "local_operator/tui/widgets/settings_view.py",
    "local_operator/tui/widgets/usage_panel.py",
    "local_operator/tui/projects_render.py",
    "local_operator/tui/app.py",
    "local_operator/tui/widgets/todo_panel.py",
    "local_operator/tui/widgets/command_picker.py",
    "local_operator/tui/widgets/projects_start.py",
)

BASE = {
    "tui.small.session_sidebar.opened_by": "opened by {role}",
    "tui.small.session_sidebar.lead_return": "esc return",
    "tui.small.session_sidebar.lead_focus": "f9 focus",
    "tui.small.session_sidebar.pin_suffix": " · ctrl+k pin",
    "tui.small.session_sidebar.asks_suffix": " · ctrl+f asks",
    "tui.small.session_sidebar.hide_suffix": " · ctrl+b hide",
}


def test_the_six_keys_are_exactly_their_base_bytes() -> None:
    live = catalogues.load_catalogue("en", "tui.small")
    golden = json.loads(
        (REPO / "tests/unit/i18n/fixtures/tui_small_en_golden.json").read_text(encoding="utf-8")
    )
    # A REAL equality: the review caught that the first draft could not fail
    # (F2/Q1) — a comparison that passes when the dicts differ is decoration.
    assert live == golden, "the shipped catalogue and the fixture golden drifted"
    # 137 = the previous 121 plus the sixteen classification-spend note keys
    # (11 wide/narrow rungs + 3 mixed-attempt rungs + 2 shortest rungs) added
    # with the JEV spend fix. Their exact bytes are pinned by the spend-notes
    # test, so this count is a second, independent stop against a silent drop.
    assert len(golden) == 137, len(golden)
    for key, value in BASE.items():
        assert live[key] == value, key
    sidecar = catalogues.context("en", "tui.small")
    for key in BASE:
        entry = sidecar[key]
        assert all(entry.get(field) for field in ("role", "tone", "note")), key


def test_the_fragments_render_the_bytes_the_assignments_carried() -> None:
    # Was: lead = "esc return" if self.has_focus else "f9 focus"
    assert _small(_x.session_sidebar_lead_return()) == "esc return"
    assert _small(_x.session_sidebar_lead_focus()) == "f9 focus"
    # Was: pin = f"{lead} · ctrl+k pin" / key = " · ctrl+f asks" / the hide suffix
    assert f"f9 focus{_small(_x.session_sidebar_pin_suffix())}" == "f9 focus · ctrl+k pin"
    assert _small(_x.session_sidebar_asks_suffix()) == " · ctrl+f asks"
    assert f"esc return{_small(_x.session_sidebar_hide_suffix())}" == "esc return · ctrl+b hide"
    # Was: lines.append(f"opened by {role}" if role else AGENT_OPENED_MARK)
    assert _small(_x.session_sidebar_opened_by(role="user")) == "opened by user"
    assert _small(_x.session_sidebar_opened_by(role="subagent")) == "opened by subagent"


def test_the_scaffold_ignores_are_present_and_reasoned() -> None:
    """The decision's other half: five sites ignored, each carrying the category."""
    source = (REPO / ACTIVATED).read_text(encoding="utf-8")
    pragmas = [line for line in source.splitlines() if "# i18n: ignore" in line]
    assert len(pragmas) == 5, pragmas
    for line in pragmas:
        assert "keycap display vocabulary" in line, line
        # the reason must say WHY, not merely "no": name the format token
        assert "{t}" in line, line
    # The format-token claim itself: `{t}` is substituted, never rendered raw.
    assert "template.format(t=" in source


def test_the_activated_file_lost_its_ceiling() -> None:
    baseline = json.loads((REPO / "i18n" / "baseline.json").read_text(encoding="utf-8"))
    assert ACTIVATED in baseline["enforced"], f"{ACTIVATED} is not activated"
    assert ACTIVATED not in baseline["files"], f"{ACTIVATED} kept its ceiling"
    for path in RETAINED:
        assert path in baseline["enforced"], f"{path} lost its activation"
        assert path not in baseline["files"], f"{path} regained a ceiling"


def test_a_planted_literal_fails_the_activated_checker(tmp_path, monkeypatch, i18n_check) -> None:
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
