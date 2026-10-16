"""`tui.small` slice: the five smallest TUI files' en copy is byte-identical.

The fixture beside this file is a SNAPSHOT of the six literals this slice
extracted, transcribed from the pre-slice source on `origin/main` @
`3eb91bcf4d` (the slice's worktree base) — the committed form of the
byte-identical proof review and QA can re-run. Each test drives the REAL call
path or the generated key function and requires the rendered string to match
the golden EXACTLY.

Byte fidelity is the contract, and two keys carry it in a non-obvious way:

* ``draft.attachment_count`` is a PLURAL whose digit comes from a ``str`` slot,
  not from the plural argument: interpolating the count would group 1000 as
  "1,000" and break the pre-slice ``f"{images} image..."`` at high magnitudes,
  so the magnitude battery below covers the grouping boundary.
* ``model_picker.window_position`` is a raw-digit pair for the same reason —
  ``str(end - start)`` / ``str(total)``, matching the pre-slice f-string and the
  ``tui.analytics.totals.calls_meta`` raw-count convention.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_operator.i18n import catalogues, runtime
from local_operator.i18n.keys import tui_small as copy
from local_operator.i18n.messages import Msg
from local_operator.i18n.text import tr

FIXTURE = Path(__file__).parent / "fixtures" / "tui_small_en_golden.json"
GOLDEN: dict[str, str] = json.loads(FIXTURE.read_text(encoding="utf-8"))
REPO = Path(__file__).resolve().parents[3]

#: The five files this slice activated, exact paths (never a directory prefix:
#: sibling files in `tui/` and `tui/widgets/` still carry literals, so a prefix
#: would red-CI the whole tree for every session).
ACTIVATED = (
    "local_operator/tui/__init__.py",
    "local_operator/tui/session_presentation.py",
    "local_operator/tui/widgets/approval.py",
    "local_operator/tui/widgets/model_picker.py",
    "local_operator/tui/widgets/session_panel.py",
)

_t = tr("tui.small")


def _en(key: str, **params: object) -> str:
    """The message rendered from the FIXTURE source, never from the live file."""
    return runtime.render_message(GOLDEN[key], params, "en")


def test_catalogue_matches_the_pre_slice_golden() -> None:
    assert catalogues.load_catalogue("en", "tui.small") == GOLDEN
    # Grown slice by slice: later ratchet slices extend the same namespace, so
    # this count tracks the catalogue, not one slice's key set.
    assert len(GOLDEN) == 34


def test_the_sidecar_names_role_tone_and_note_for_every_key() -> None:
    sidecar = catalogues.context("en", "tui.small")
    assert set(sidecar) == set(GOLDEN)
    assert all(all(entry.get(k) for k in ("role", "tone", "note")) for entry in sidecar.values())


def test_the_zero_argument_literals_are_their_golden_values() -> None:
    assert _t(copy.approval_allow_verb()) == _en("tui.small.approval.allow_verb") == "allow "
    # The trailing space is load-bearing: the caller appends the target after it.
    assert _t(copy.approval_allow_verb()).endswith(" ")
    assert _t(copy.timing_no_samples()) == _en("tui.small.timing.no_samples")
    assert _t(copy.spend_channel_not_itemised()) == _en("tui.small.spend.channel_not_itemised")


def test_the_resume_hint_keeps_its_leading_and_trailing_newlines() -> None:
    for hint in ("lop --resume abc123", "lop --resume 0f9c1e", "lop --resume a"):
        rendered = _t(copy.session_resume_hint(hint=hint))
        assert rendered == _en("tui.small.session.resume_hint", hint=hint)
        # The pre-slice f-string: newline, sentence, indented hint, newline.
        assert rendered == f"\nsession ended — resume with:\n  {hint}\n"


def test_the_attachment_plural_is_byte_identical_across_the_grouping_boundary() -> None:
    # 1000/1200/100000 matter: a {images, number} or `#` slot would print
    # "1,200", where the pre-slice f-string printed the bare integer.
    for images in (0, 1, 2, 3, 5, 11, 999, 1000, 1200, 1234, 100000):
        rendered = _t(copy.draft_attachment_count(images=images, shown=str(images)))
        assert rendered == _en("tui.small.draft.attachment_count", images=images, shown=str(images))
        assert rendered == f"({images} image{'s' if images != 1 else ''})"


def test_the_model_picker_counter_keeps_raw_digits() -> None:
    for shown, total in ((14, 300), (0, 0), (1, 1), (9, 12), (1193, 1200), (20, 3000)):
        rendered = _t(copy.model_picker_window_position(shown=str(shown), total=str(total)))
        assert rendered == _en(
            "tui.small.model_picker.window_position", shown=str(shown), total=str(total)
        )
        assert rendered == f"{shown} of {total}"
        # Raw digits, never grouped.
        assert "," not in rendered


def test_unknown_messages_degrade_to_their_code() -> None:
    resolve = tr("tui.small")
    assert resolve(Msg("tui.small.nope", {})) == "tui.small.nope"


def test_every_activated_file_lost_its_ceiling() -> None:
    """The toothlessness guard: enforced WITHOUT a ceiling, exact path only.

    A file left in ``files`` while also in ``enforced`` still absorbs a
    planted literal into its ceiling, so the gate silently does nothing —
    the defect this programme shipped once. Read from the committed baseline,
    not a working copy.
    """
    baseline = json.loads((REPO / "i18n" / "baseline.json").read_text(encoding="utf-8"))
    for path in ACTIVATED:
        assert path in baseline["enforced"], f"{path} is not activated"
        assert path not in baseline["files"], f"{path} kept its ceiling; the gate is toothless"
    # No directory prefix was used to activate them.
    assert not [entry for entry in baseline["enforced"] if entry.endswith("/") and "tui" in entry]


@pytest.mark.parametrize("path", ACTIVATED)
def test_a_planted_literal_fails_the_activated_checker(
    path: str, tmp_path, monkeypatch, i18n_check
) -> None:
    """Prove the activation BITES: the real checker entrypoint, isolated copy.

    Mirrors the headless-print slice's control — a single planted sink literal
    must turn rc 0 into rc 1 naming the file, and restoring it must return rc 0.
    The checker module is session-scoped, so every override goes through
    ``monkeypatch`` rather than assignment: a bare write would leak the fixture
    tree into every later test in the run.
    """
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
