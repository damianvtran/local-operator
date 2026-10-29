"""The `session.py` enforcement slice, pinned against the real baseline.

`session/session.py` keeps English for its model-facing todo/project reminder
blocks, the open-asks reminder (SA-D3), the missed-wake prompt and its internal
cleanup diagnostics; its user-facing notices live on the `tui.notices`
catalogue. After the slice the file carries scoped pragmas, no `files` ceiling
and an `enforced` entry, and a reintroduced unflagged literal is blocking.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PATH = "local_operator/session/session.py"
PRAGMAS = 18


def test_session_is_pragma_clean_and_activated(i18n_check) -> None:
    baseline = i18n_check.load_baseline(i18n_check.BASELINE)
    source = (REPO / PATH).read_text(encoding="utf-8")
    assert i18n_check.scan_source(PATH, source) == []
    assert source.count("# i18n: ignore") == PRAGMAS
    assert PATH in baseline["enforced"]
    assert PATH not in baseline["files"]


def test_a_reintroduced_literal_in_session_is_blocking(i18n_check) -> None:
    baseline = i18n_check.load_baseline(i18n_check.BASELINE)
    source = (REPO / PATH).read_text(encoding="utf-8")
    planted = source + '\n\ndef _planted():\n    print("Plant a hardcoded user-facing label")\n'
    violations = i18n_check.scan_source(PATH, planted)
    assert len(violations) == 1
    assert violations[0].kind == "print"
    assert i18n_check.in_enforced_scope(PATH, baseline["enforced"])
