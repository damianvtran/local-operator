"""`tui.small`/`tui.projects` next-smallest slice: byte-identical en copy.

The fixture beside this file is a SNAPSHOT of the six literals this slice
extracted, transcribed from the pre-slice source on `origin/main` @
`3edbd3ae3` — the committed form of the byte-identical proof review and QA
can re-run. Each test drives the REAL call path or the generated key
function and requires the rendered string to match the golden EXACTLY.

Byte fidelity is the contract, and it is non-obvious in two places:

* ``copy.code_count``/``copy.quote_count`` use a ``str`` slot and NO plural
  branch: the pre-slice f-strings printed the bare word at every count
  (``1 code``, ``1200 code``), so pluralising or number-grouping would be a
  visible change.
* ``picker.showing_prefix``/``picker.of_separator`` keep their spaces: they
  are separate Rich ``Text`` fragments (the numerals are styled brighter),
  so the spaces are the only separators in the assembled counter.

``local_operator/tui/images.py`` is deliberately NOT in this slice: its two
scanner literals are kitty graphics protocol tokens (``a=t,i=…,f=100,q=2``),
machine syntax a terminal parses, never user-facing copy.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_operator.i18n import catalogues, runtime
from local_operator.i18n.keys import tui_projects as projects_copy
from local_operator.i18n.keys import tui_small as small_copy
from local_operator.i18n.text import tr
from local_operator.tui import copy_targets

FIXTURE = Path(__file__).parent / "fixtures" / "tui_next_small_en_golden.json"
GOLDEN: dict[str, str] = json.loads(FIXTURE.read_text(encoding="utf-8"))
REPO = Path(__file__).resolve().parents[3]

#: The four files this slice activated, exact paths (never a directory
#: prefix: sibling files in `tui/` and `tui/widgets/` still carry literals,
#: so a prefix would red-CI the whole tree for every session).
ACTIVATED = (
    "local_operator/tui/copy_targets.py",
    "local_operator/tui/widgets/ask_picker.py",
    "local_operator/tui/widgets/move_picker.py",
    "local_operator/tui/widgets/projects_detail.py",
)

_small = tr("tui.small")
_projects = tr("tui.projects")


def _en(key: str, **params: object) -> str:
    """The message rendered from the FIXTURE source, never from the live file."""
    return runtime.render_message(GOLDEN[key], params, "en")


def test_the_new_keys_match_the_pre_slice_golden() -> None:
    assert len(GOLDEN) == 6
    small = catalogues.load_catalogue("en", "tui.small")
    projects = catalogues.load_catalogue("en", "tui.projects")
    for key, value in GOLDEN.items():
        live = (small if key.startswith("tui.small.") else projects)[key]
        assert live == value, key


def test_the_sidecars_name_role_tone_and_note_for_every_new_key() -> None:
    for namespace in ("tui.small", "tui.projects"):
        sidecar = catalogues.context("en", namespace)
        for key in GOLDEN:
            if key.startswith(f"{namespace}."):
                assert sidecar[key]["role"] and sidecar[key]["tone"] and sidecar[key]["note"]


def test_copy_counts_keep_raw_digits_and_never_pluralise() -> None:
    # 1 matters (no "1 codes"), 1200 matters (no "1,200"): the pre-slice
    # f-strings printed f"{n} code" / f"{n} quote" at every count.
    for count in (0, 1, 2, 11, 999, 1000, 1200, 100000):
        assert _small(small_copy.copy_code_count(shown=str(count))) == f"{count} code"
        assert _small(small_copy.copy_quote_count(shown=str(count))) == f"{count} quote"
        assert _small(small_copy.copy_code_count(shown=str(count))) == _en(
            "tui.small.copy.code_count", shown=str(count)
        )


def test_the_real_message_hint_matches_the_pre_slice_bytes() -> None:
    assert copy_targets._message_hint("one\ntwo", 1, 0, False) == "2 lines · 1 code"
    assert copy_targets._message_hint("one", 0, 1200, True) == "truncated · 1 line · 1200 quote"
    assert copy_targets._message_hint("one", 3, 2, False) == "1 line · 3 code · 2 quote"


def test_picker_counter_fragments_assemble_to_the_pre_slice_bytes() -> None:
    # The ask picker drops the prefix at narrow widths; both pickers keep
    # the separator, so the assembled phrase must equal the old literals.
    for span, total in (("1–1", "4"), ("1–9", "42"), ("10–18", "1200")):
        assembled = (
            f"{_small(small_copy.picker_showing_prefix())}{span}"
            f"{_small(small_copy.picker_of_separator())}{total}"
        )
        assert assembled == f"showing {span} of {total}"
    assert _small(small_copy.picker_showing_prefix()) == "showing "
    assert _small(small_copy.picker_of_separator()) == " of "


def test_detail_placeholders_match_the_pre_slice_bytes() -> None:
    assert _projects(projects_copy.detail_no_text()) == "(no text in this entry)"
    assert _projects(projects_copy.detail_no_description()) == "no description yet"


def test_every_activated_file_lost_its_ceiling() -> None:
    """The toothlessness guard: enforced WITHOUT a ceiling, exact path only.

    A file left in ``files`` while also in ``enforced`` still absorbs a
    planted literal into its ceiling, so the gate silently does nothing.
    """
    baseline = json.loads((REPO / "i18n" / "baseline.json").read_text(encoding="utf-8"))
    for path in ACTIVATED:
        assert path in baseline["enforced"], f"{path} is not activated"
        assert path not in baseline["files"], f"{path} kept its ceiling; the gate is toothless"
    assert not [entry for entry in baseline["enforced"] if entry.endswith("/") and "tui" in entry]
    # images.py was excluded on purpose (protocol tokens, not copy): it must
    # NOT have been activated, and keeps its scanner ceiling.
    assert "local_operator/tui/images.py" not in baseline["enforced"]
    assert baseline["files"]["local_operator/tui/images.py"] == 2


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
