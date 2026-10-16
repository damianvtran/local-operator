"""Ratchet slice 6 (`tui.small`): goal_panel / network_panel.

The fixture beside this file is a SNAPSHOT of the literals this slice
extracted, transcribed from the pre-slice source on `origin/main` @
`6cf0c17fbe138ff855110c15dc40d5a57c2a7065` — the committed form of the
byte-identical proof review and QA can re-run. Each test drives the
generated key function through the real `tr()` resolver and requires the
rendered string to match the golden EXACTLY.

Byte fidelity is the contract, and it is non-obvious in four places:

* The scanner re-derivation matched the recorded ceilings exactly
  (8 / 11 literals); no stale under-count this slice.
* `goal_panel.more_settled` and `goal_panel.more_settled_indent` are TWO
  keys for one sentence because the two call sites indent differently
  (two leading spaces vs none). A key is not interchangeable just
  because the visible text looks the same — whitespace is layout.
* `network_panel.member_count` uses the plural convention this programme
  settled on: `{count}` selects the branch and `{shown}` carries the
  digit as a RAW string, so `1200` stays `1200` and never groups to
  `1,200`. The pre-slice code chose singular/plural by hand
  (`" member" + ("" if n == 1 else "s")`), which is the same output.
* The detail rows keep protocol vocabulary (`installed:`, `identity:`,
  `relay:`, `pid`) inside the catalogue VALUE and keep the column
  padding and the trailing newline as part of the row: trimming either
  would change bytes on screen.

Scanner-blind copy noticed and deliberately NOT extracted here (the
checker cannot see it, so activating these files does not gate it; it
needs its own slice): `goal_panel`'s `_HISTORY_CLIPPED_NOTE` module
constant and its judge `state` word table, and `network_panel`'s
`yes`/`no` and `present`/`missing` value pairs passed as f-string
ternaries (their LABEL rows are extracted; the value words travel as
parameters and are not catalogue keys yet).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_operator.i18n import catalogues, runtime
from local_operator.i18n.keys import tui_small as small_copy
from local_operator.i18n.text import tr

FIXTURE = Path(__file__).parent / "fixtures" / "tui_ratchet_six_en_golden.json"
GOLDEN: dict[str, str] = json.loads(FIXTURE.read_text(encoding="utf-8"))
REPO = Path(__file__).resolve().parents[3]

#: The two files this slice activated, exact paths (never a directory
#: prefix: sibling files in `tui/widgets/` still carry literals, so a
#: prefix would red-CI the whole tree for every session).
ACTIVATED = (
    "local_operator/tui/widgets/goal_panel.py",
    "local_operator/tui/widgets/network_panel.py",
)

#: Slice 5's activated files, pinned so this slice's baseline edit cannot
#: walk them back.
RETAINED = (
    "local_operator/tui/widgets/org_chart_view.py",
    "local_operator/tui/widgets/tool_card.py",
    "local_operator/tui/widgets/session_picker.py",
)

_small = tr("tui.small")


def _en(key: str, **params: object) -> str:
    """The message rendered from the FIXTURE source, never from the live file."""
    return runtime.render_message(GOLDEN[key], params, "en")


def test_the_new_keys_match_the_pre_slice_golden() -> None:
    assert len(GOLDEN) == 20
    live = catalogues.load_catalogue("en", "tui.small")
    for key, value in GOLDEN.items():
        assert live[key] == value, key


def test_the_sidecar_names_role_tone_and_note_for_every_new_key() -> None:
    sidecar = catalogues.context("en", "tui.small")
    for key in GOLDEN:
        entry = sidecar[key]
        assert all(entry.get(field) for field in ("role", "tone", "note")), key


def test_goal_panel_rows_are_byte_identical_to_the_pre_slice_copy() -> None:
    assert _small(small_copy.goal_panel_settled_heading()) == "settled"
    assert _small(small_copy.goal_panel_no_goal_hint()) == "no goal set — /goal <text> to set one"
    # The two leading spaces ARE the indent; `/goal <text>` is a command a user
    # types, so it stays verbatim in every locale.
    assert _small(small_copy.goal_panel_none_yet()) == "  none yet"
    assert _small(small_copy.goal_panel_done_suffix()) == "  — done"
    assert _small(small_copy.goal_panel_judge_absent()) == "judge: —"
    assert _small(small_copy.goal_panel_judge_line(words="working")) == "judge: working"
    assert _small(small_copy.goal_panel_judge_run(run="2", cap="5")) == "run 2/5"


def test_the_two_more_settled_keys_keep_their_own_indentation() -> None:
    # Two keys for one sentence: the call sites indent differently, so a single
    # key would shift one of the two rows by two cells.
    indented = _small(small_copy.goal_panel_more_settled_indent(hidden="3"))
    plain = _small(small_copy.goal_panel_more_settled(hidden="3"))
    assert indented == "  … 3 more settled"
    assert plain == "… 3 more settled"
    assert indented == "  " + plain


def test_network_panel_rows_are_byte_identical_to_the_pre_slice_copy() -> None:
    assert _small(small_copy.network_panel_relay_checking_indent()) == "  relay: checking…\n"
    assert _small(small_copy.network_panel_checking_indent()) == "  checking…\n"
    assert _small(small_copy.network_panel_no_members()) == "  no members reported\n"
    # The run of spaces after `relay:` aligns this row's value with the
    # `running, pid …` sibling row, so the padding is layout, not decoration.
    assert _small(small_copy.network_panel_relay_not_running()) == "  relay:      not running\n"
    assert _small(small_copy.network_panel_stale_tag()) == "[stale]"
    assert _small(small_copy.network_panel_suspect_tag()) == "  [suspect: key may be copied]"
    assert _small(small_copy.network_panel_trust_field(trust="revoked")) == "trust revoked"
    assert _small(small_copy.network_panel_installed_row(value="yes")) == "  installed:  yes\n"
    assert (
        _small(small_copy.network_panel_identity_row(identity="present"))
        == "  identity:   present\n"
    )
    assert (
        _small(small_copy.network_panel_relay_running(pid="4321", state="answering"))
        == "  relay:      running, pid 4321 — answering\n"
    )


@pytest.mark.parametrize(("members", "expected"), [(1, "1 member"), (2, "2 members")])
def test_member_count_follows_the_number(members: int, expected: str) -> None:
    assert (
        _small(small_copy.network_panel_member_count(count=members, shown=str(members))) == expected
    )


@pytest.mark.parametrize("members", [1200, 10000])
def test_member_count_keeps_raw_digits_at_any_magnitude(members: int) -> None:
    # The grouping boundary: a number slot would render 1,200 where the
    # pre-slice f-string printed 1200, so the digit travels as a string.
    rendered = _small(small_copy.network_panel_member_count(count=members, shown=str(members)))
    assert rendered == f"{members} members"
    assert "," not in rendered


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
