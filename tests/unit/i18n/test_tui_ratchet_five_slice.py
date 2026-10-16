"""Ratchet slice 5 (`tui.small`): org_chart_view / tool_card / session_picker.

The fixture beside this file is a SNAPSHOT of the literals this slice
extracted, transcribed from the pre-slice source on `origin/main` @
`7ebf68bf40529176152ce7d088a587f17aabb104` — the committed form of the
byte-identical proof review and QA can re-run. Each test drives the REAL
call path (the org chart's own chrome painter, the tool card's own row
builders, the session picker's own panes) or the generated key function
and requires the rendered string to match the golden EXACTLY.

Byte fidelity is the contract, and it is non-obvious in four places:

* The scanner re-derivation matched the recorded ceilings exactly
  (5 / 5 / 7 literals); no stale under-count this slice.
* `session_picker`'s counter reuses the EXISTING
  `tui.small.picker.of_separator` (" of ") rather than minting a second
  key for the same bytes — a duplicate key would be a second spelling
  of one separator that could drift. Its `showing_prefix` is a NEW key
  because its three leading spaces differ from the existing
  `picker.showing_prefix` ("showing ").
* `all_archived` keeps the pre-slice `conversation(s)` spelling at every
  count: the original f-string never pluralised, so adding an ICU plural
  branch would change bytes. `{total}` is a raw digit string for the
  same reason (`1200`, never `1,200`).
* The HTTP rows keep `status`/`reason` as raw interpolated strings and
  the trailing period INSIDE the catalogue value: the pre-slice rows
  ended `... {clause}.` / `... content.` and any trim changes bytes.

Scanner-blind copy noticed and deliberately NOT extracted here (the
checker cannot see it, so activating these files does not gate it; it
needs its own slice): the org-chart legend's tuple glyphs
(`? unresolved`, `↩ cycle`, `⋯ depth-limit`, `·N members`, `×N copies`),
tool_card's `Provider:`/`Sources:`/`Fetched:` assemblies and
session_picker's `ctrl+u/ctrl+d` preview candidates and
`{count} {word}` tally (count noun arrives via a variable).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import PropertyMock, patch

import pytest
from textual.geometry import Size

from local_operator.i18n import catalogues, runtime
from local_operator.i18n.keys import tui_small as small_copy
from local_operator.i18n.text import tr
from local_operator.resume import SessionRow
from local_operator.tui.widgets.org_chart_view import OrgChartView
from local_operator.tui.widgets.session_picker import SessionPickerScreen
from local_operator.tui.widgets.tool_card import (
    _fetch_result_output,
    _search_result_output,
)
from local_operator.web_fetch.failure import block_lead

FIXTURE = Path(__file__).parent / "fixtures" / "tui_ratchet_five_en_golden.json"
GOLDEN: dict[str, str] = json.loads(FIXTURE.read_text(encoding="utf-8"))
REPO = Path(__file__).resolve().parents[3]
NOW = 1_000_000.0

#: The three files this slice activated, exact paths (never a directory
#: prefix: sibling files in `tui/widgets/` still carry literals, so a
#: prefix would red-CI the whole tree for every session).
ACTIVATED = (
    "local_operator/tui/widgets/org_chart_view.py",
    "local_operator/tui/widgets/tool_card.py",
    "local_operator/tui/widgets/session_picker.py",
)

#: Slice 4's activated files, pinned so this slice's baseline edit cannot
#: walk them back.
RETAINED = (
    "local_operator/tui/widgets/info_panel.py",
    "local_operator/tui/widgets/subagent_view.py",
    "local_operator/tui/widgets/wake_panel.py",
)

_small = tr("tui.small")


def _en(key: str, **params: object) -> str:
    """The message rendered from the FIXTURE source, never from the live file."""
    return runtime.render_message(GOLDEN[key], params, "en")


def test_the_new_keys_match_the_pre_slice_golden() -> None:
    # 16 NEW keys; the 17th extracted literal (` of `) reuses the existing
    # `picker.of_separator` key and so adds no catalogue entry.
    assert len(GOLDEN) == 16
    live = catalogues.load_catalogue("en", "tui.small")
    for key, value in GOLDEN.items():
        assert live[key] == value, key
    assert live["tui.small.picker.of_separator"] == " of "


class _Capture:
    """A stand-in Static: records the Text it was asked to paint."""

    def __init__(self) -> None:
        self.text = None
        self.display = False

    def update(self, text) -> None:  # noqa: ANN001, ANN202 - test double
        self.text = text


def _chrome_view(*, expand_all: bool, legend_open: bool) -> OrgChartView:
    view = OrgChartView.__new__(OrgChartView)
    view._root = None
    view._team_name = "org"
    view._tier = 1
    view._expand_all = expand_all
    view._title = _Capture()
    view._rule = _Capture()
    view._legend = _Capture()
    view._legend_open = legend_open
    view._paint_hints = lambda: None
    return view


def test_org_chart_title_and_legend_paint_the_extracted_copy() -> None:
    # The painter reads the widget's geometry; an unmounted view has none,
    # so the size is stubbed while every copy fragment runs for real.
    with patch.object(OrgChartView, "size", new_callable=PropertyMock, return_value=Size(120, 1)):
        view = _chrome_view(expand_all=True, legend_open=True)
        view._paint_chrome()
        assert view._title_text.plain == "team org · org chart · zoom: standard · expanded"
        assert view._legend.text.plain.startswith("◆ manager")

        collapsed = _chrome_view(expand_all=False, legend_open=False)
        collapsed._paint_chrome()
        assert collapsed._title_text.plain == "team org · org chart · zoom: standard"


def test_tool_card_search_and_fetch_rows_paint_the_extracted_copy() -> None:
    search = _search_result_output(
        {
            "provider": "brave",
            "sources": [{"title": "T", "url": "https://e.example", "snippet": "s"}],
        }
    )
    assert search[-1] == GOLDEN["tui.small.tool_card.search_fetch_hint"]

    generic = _fetch_result_output({"url": "https://e.example", "status": 503, "http_error": True})
    assert generic[0] == "⚠ HTTP 503 Service Unavailable — error/block page, not page content."
    assert generic[0] == _en(
        "tui.small.tool_card.http_error_page", status="503", reason="Service Unavailable"
    )

    blocked = _fetch_result_output(
        {
            "url": "https://e.example",
            "status": 403,
            "http_error": True,
            "failure_kind": "blocked",
            "block_vendor": "cloudflare",
        }
    )
    assert blocked[0] == _en(
        "tui.small.tool_card.http_blocked",
        status="403",
        reason="Forbidden",
        clause=block_lead("cloudflare"),
    )

    rendered = _fetch_result_output(
        {
            "url": "https://e.example",
            "status": 200,
            "render_method": "markdownify",
            "lines": 800,
            "bytes": 1024,
            "low_quality": True,
        }
    )
    assert any(row.startswith("Rendered: markdownify") for row in rendered)
    assert rendered[-1] == GOLDEN["tui.small.tool_card.low_quality_hint"]


def _row(session_id: str, name: str, *, archived: bool = False) -> SessionRow:
    return SessionRow(id=session_id, mtime=NOW - 60.0, name=name, archived=archived)


def test_session_picker_panes_paint_the_extracted_copy() -> None:
    empty = SessionPickerScreen([], NOW)
    assert empty.render_preview_for_test() == ["no session"]
    assert "type to filter" in empty.render_footer_for_test()

    filtered = SessionPickerScreen([_row("aaa111", "asteroids")], NOW)
    filtered.set_query("zzz")
    assert "no session matches that filter" in "\n".join(filtered.render_lines_for_test())

    archived = SessionPickerScreen([_row("aaa111", "old", archived=True)], NOW)
    assert "all 1 conversation(s) are archived" in "\n".join(archived.render_lines_for_test())

    many = SessionPickerScreen(
        [_row(f"{index:012x}", f"session {index}") for index in range(80)], NOW
    )
    footer = many.render_footer_for_test()
    assert "   showing " in footer
    assert " of " in footer
    assert "ctrl+e" in footer


def test_session_picker_catalogue_bytes_keep_spaces_and_raw_digits() -> None:
    assert (
        _small(small_copy.session_picker_showing_prefix())
        == _en("tui.small.session_picker.showing_prefix")
        == "   showing "
    )
    assert _small(small_copy.picker_of_separator()) == " of "
    # 1200 must stay 1200, and the `(s)` spelling holds at every count.
    for total, expected in (
        ("1", "all 1 conversation(s) are archived"),
        ("1200", "all 1200 conversation(s) are archived"),
    ):
        rendered = _small(small_copy.session_picker_all_archived(total=total))
        assert rendered == expected
        assert rendered == _en("tui.small.session_picker.all_archived", total=total)


def test_every_activated_file_lost_its_ceiling() -> None:
    """The toothlessness guard: enforced WITHOUT a ceiling, exact path only.

    A file left in ``files`` while also in ``enforced`` still absorbs a
    planted literal into its ceiling, so the gate silently does nothing.
    """
    baseline = json.loads((REPO / "i18n" / "baseline.json").read_text(encoding="utf-8"))
    for path in ACTIVATED:
        assert path in baseline["enforced"], f"{path} is not activated"
        assert path not in baseline["files"], f"{path} kept its ceiling; the gate is toothless"
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
