"""Scanner-coverage slice: the assembled copy the improved detector found.

The scanner now folds `+` and the ternary in argument position (see
`scripts/i18n/check.py::_literal_worthy`). Running it over the tree surfaced
fifteen literals in files that were ALREADY activated — copy the old scanner
could not see, which is exactly why the operator's requirement (a gate that
fails on hardcoded user-facing strings) needed the fold before more
extraction.

Every assertion below is DIFFERENTIAL: the left side is the generated key
rendering through the real resolver, the right side is the expression the
site carried before this change, written out literally. Equal means the
extraction changed no bytes.

The sixteenth finding was not copy at all: the two `exec_mode.py` hits are
SGR colour codes wrapping already-extracted copy, and they carry an
`# i18n: ignore` pragma with that reason rather than a catalogue key.
"""

from __future__ import annotations

import pytest

from local_operator.i18n.keys import tui_small as _x
from local_operator.i18n.text import tr

_small = tr("tui.small")


def test_approval_decision_prefixes_are_byte_identical() -> None:
    # Was: "allowed " if allowed else "denied "  (trailing space = layout)
    assert _small(_x.approval_decision_allowed()) == "allowed "
    assert _small(_x.approval_decision_denied()) == "denied "


def test_info_panel_fragments_are_byte_identical() -> None:
    # Was: "not answering" if line.state == "wedged" else line.state
    assert _small(_x.info_panel_not_answering()) == "not answering"
    # Was: "Install info\n" + "─" * max(1, self._card_width())
    assert _small(_x.info_panel_install_info_heading()) == "Install info\n"
    assert _small(_x.panels_scroll_hint()) == " · ↑↓ scroll"


def test_model_picker_fields_are_byte_identical() -> None:
    # Was: "effort " + ("/".join(row.reasoning_efforts) or "none")
    value = "/".join(["low", "high"])
    assert _small(_x.model_picker_effort(value=value)) == "effort low/high"
    assert (
        _small(_x.model_picker_effort(value=_small(_x.model_picker_none_word()))) == "effort none"
    )
    # Was: "default " + row.reasoning_default_effort
    assert _small(_x.model_picker_default(value="low")) == "default low"
    # Was: "released " + row.release_date
    assert _small(_x.model_picker_released(value="2026-01-02")) == "released 2026-01-02"


def test_model_picker_empty_states_are_byte_identical() -> None:
    # Was: "no matching models" if query else "no models available"
    assert _small(_x.model_picker_no_matching_models()) == "no matching models"
    assert _small(_x.model_picker_no_models_available()) == "no models available"


def test_card_headings_are_byte_identical() -> None:
    # Was: "Mesh networks\n" + "─" * width  (the rule is drawn in code)
    assert _small(_x.network_panel_mesh_networks_heading()) == "Mesh networks\n"
    # Was: "Session diagnostics\n" + "─" * max(1, self._card_width())
    assert _small(_x.session_panel_diagnostics_heading()) == "Session diagnostics\n"


def test_subagent_panel_keybinding_hints_are_byte_identical() -> None:
    # Was: "ctrl+g to shrink" if self._expanded else "ctrl+g to expand"
    assert _small(_x.subagent_panel_ctrl_g_shrink()) == "ctrl+g to shrink"
    assert _small(_x.subagent_panel_ctrl_g_expand()) == "ctrl+g to expand"


@pytest.mark.parametrize("count", [0, 1, 2, 1200])
def test_scheduled_count_matches_the_expression_it_replaced(count: int) -> None:
    # Was: f"{len(rows)} scheduled" if len(rows) != 1 else "1 scheduled"
    # Raw digits at every magnitude — 1200 must never group to 1,200.
    expected = "1 scheduled" if count == 1 else f"{count} scheduled"
    rendered = _small(_x.wake_panel_scheduled_count(count=count, shown=str(count)))
    assert rendered == expected
    if count == 1200:
        assert "," not in rendered


def test_the_two_exec_mode_hits_are_pragma_exempt_not_extracted() -> None:
    """The ANSI concatenations are exempt by pragma, and the pragma has a reason.

    Pinned because it is the one place this slice silences rather than
    extracts: if the pragma loses its reason the checker refuses it, and if
    someone extracts the escape codes into a catalogue key this test should
    be replaced with a rendering assertion instead.
    """
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    source = (repo / "local_operator" / "exec_mode.py").read_text(encoding="utf-8")
    # This file carries other, pre-existing pragmas (the stable-prefix sites);
    # the two added here are the SGR ones, and each must state its reason.
    pragmas = [line for line in source.splitlines() if "i18n: ignore" in line and "SGR" in line]
    assert len(pragmas) == 2, pragmas
    for line in pragmas:
        reason = line.split("ignore", 1)[1].strip()
        assert reason.startswith("SGR") and len(reason) > 20, line
