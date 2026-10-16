"""`tui.projects` form / `tui.small` aside+subagents slice: byte-identical en.

The fixture beside this file is a SNAPSHOT of the fourteen literals this
slice extracted, transcribed from the pre-slice source on `origin/main` @
`57b6c5303` — the committed form of the byte-identical proof review and QA
can re-run. Each test drives the REAL call path or the generated key
function and requires the rendered string to match the golden EXACTLY.

Byte fidelity is the contract, and it is non-obvious in three places:

* ``form.date_placeholder`` is a FORMAT TOKEN (``YYYY-MM-DD``), not prose:
  it stays exactly that shape in every locale because it is the one
  spelling the store parses, and both date fields share the one key.
* ``aside.off_the_record`` keeps its TWO LEADING SPACES and
  ``subagents.of_scope`` its leading space: they are Rich ``Text``
  fragments appended to a row that already carries the title, so the
  spaces are the only separators in the assembled line.
* ``form.readback_hint``/``form.readback_error`` keep the
  ``{label} hint: …`` shape verbatim: the form's existing test suite
  prefix-matches readback rows on it, so the layout is a contract, not
  decoration.

``local_operator/tui/session_move.py`` is deliberately NOT in this slice:
its three scanner literals are ``--keep``/``--queue``/``--json`` CLI argv
flags appended to a subprocess command — machine syntax a parser reads,
never user-facing copy (the `images.py` precedent), so it keeps its
ceiling unenforced.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_operator.i18n import catalogues, runtime
from local_operator.i18n.keys import tui_projects as projects_copy
from local_operator.i18n.keys import tui_small as small_copy
from local_operator.i18n.text import tr
from local_operator.tui.widgets.projects_form import ProjectsFormPage

FIXTURE = Path(__file__).parent / "fixtures" / "tui_ratchet_two_en_golden.json"
GOLDEN: dict[str, str] = json.loads(FIXTURE.read_text(encoding="utf-8"))
REPO = Path(__file__).resolve().parents[3]

#: The three files this slice activated, exact paths (never a directory
#: prefix: sibling files in `tui/widgets/` still carry literals, so a
#: prefix would red-CI the whole tree for every session).
ACTIVATED = (
    "local_operator/tui/widgets/projects_form.py",
    "local_operator/tui/widgets/subagent_panel.py",
    "local_operator/tui/widgets/aside_panel.py",
)

_projects = tr("tui.projects")
_small = tr("tui.small")


def _en(key: str, **params: object) -> str:
    """The message rendered from the FIXTURE source, never from the live file."""
    return runtime.render_message(GOLDEN[key], params, "en")


def test_the_new_keys_match_the_pre_slice_golden() -> None:
    assert len(GOLDEN) == 14
    projects = catalogues.load_catalogue("en", "tui.projects")
    small = catalogues.load_catalogue("en", "tui.small")
    for key, value in GOLDEN.items():
        live = (projects if key.startswith("tui.projects.") else small)[key]
        assert live == value, key


def test_the_sidecars_name_role_tone_and_note_for_every_new_key() -> None:
    for namespace in ("tui.projects", "tui.small"):
        sidecar = catalogues.context("en", namespace)
        for key in GOLDEN:
            if key.startswith(f"{namespace}."):
                assert sidecar[key]["role"] and sidecar[key]["tone"] and sidecar[key]["note"]


def test_form_placeholders_match_the_pre_slice_bytes() -> None:
    assert _projects(projects_copy.form_title_placeholder()) == "a human-readable name"
    assert _projects(projects_copy.form_key_placeholder()) == "the reference handle"
    assert _projects(projects_copy.form_tags_placeholder()) == "comma or space separated"
    assert _projects(projects_copy.form_team_placeholder()) == "team"
    assert _projects(projects_copy.form_owner_placeholder()) == "owner"
    # The date placeholder is a raw format token: no grouping, no
    # localisation of its shape, and both date fields share it.
    assert _projects(projects_copy.form_date_placeholder()) == "YYYY-MM-DD"
    assert _projects(projects_copy.form_estimate_placeholder()) == "points"


def test_the_real_form_carries_the_golden_placeholders_and_readback() -> None:
    # The real widget path: constructing the page wires every Input's
    # placeholder, and readback() restates hint/error rows byte-identically.
    page = ProjectsFormPage(lambda edit: None, lambda: None)
    assert page._title_input.placeholder == "a human-readable name"
    assert page._key_input.placeholder == "the reference handle"
    assert page._tags_input.placeholder == "comma or space separated"
    assert page._team_input.placeholder == "team"
    assert page._owner_input.placeholder == "owner"
    assert page._target_input.placeholder == "YYYY-MM-DD"
    assert page._start_input.placeholder == "YYYY-MM-DD"
    assert page._estimate_input.placeholder == "points"
    rows = page.readback()
    assert "key hint: follows the title until you edit it" in rows
    # ``set_error`` also repaints the widget, which needs a mounted app; the
    # readback reads the stored refusal text, so store it directly here.
    page._key_block._error_text = "the key is required"
    assert "key error: the key is required" in page.readback()


def test_readback_messages_render_from_the_golden_source() -> None:
    for label, hint in (("key", "follows the title until you edit it"), ("estimate", "x")):
        assert _projects(projects_copy.form_readback_hint(label=label, hint=hint)) == _en(
            "tui.projects.form.readback_hint", label=label, hint=hint
        )
    assert (
        _projects(projects_copy.form_readback_error(label="title", error="too long"))
        == "title error: too long"
    )


def test_aside_and_subagent_fragments_assemble_to_the_pre_slice_bytes() -> None:
    # Title + contract line are one Rich Text row: the assembled line must
    # equal the old literals, leading spaces included.
    assembled = f"{_small(small_copy.aside_title())}{_small(small_copy.aside_off_the_record())}"
    assert assembled == "Aside  off the record — nothing here joins the chat"
    assert _small(small_copy.aside_superseded()) == "(superseded)"
    # The subagent header drops the scope suffix at the top level and
    # appends it verbatim when drilled into one child.
    assert _small(small_copy.subagents_title()) == "Subagents"
    for scope in ("worker-1", "a very long truncated label…"):
        suffix = _small(small_copy.subagents_of_scope(scope=scope))
        header = f"{_small(small_copy.subagents_title())}{suffix}"
        assert header == f"Subagents of {scope}"
        assert _small(small_copy.subagents_of_scope(scope=scope)) == _en(
            "tui.small.subagents.of_scope", scope=scope
        )


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
    # session_move.py was excluded on purpose (CLI argv flags, not copy):
    # it must NOT have been activated, and keeps its scanner ceiling.
    assert "local_operator/tui/session_move.py" not in baseline["enforced"]
    assert baseline["files"]["local_operator/tui/session_move.py"] == 3


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
