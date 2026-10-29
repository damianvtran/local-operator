"""Activated non-translatable harness sinks, pinned against the real baseline.

`harness/loop.py` keeps English for its model-facing tool-argument errors;
`incidents.py` builds incident blocks that are injected into the session and
addressed to the model. Both files now carry scoped pragmas, no `files`
ceiling, and an `enforced` entry. A reintroduced unflagged literal is a
blocking finding.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

PATHS = {
    "local_operator/harness/loop.py": 2,
    # 14 -> 15: the failover sentence in `Incident.render` is the same
    # model-facing incident block as its siblings, so it carries the same
    # pragma rather than an extracted string.
    "local_operator/incidents.py": 15,
}


def test_each_non_translatable_sink_is_pragma_clean_and_activated(i18n_check) -> None:
    baseline = i18n_check.load_baseline(i18n_check.BASELINE)
    for path, pragmas in PATHS.items():
        source = (REPO / path).read_text(encoding="utf-8")
        assert i18n_check.scan_source(path, source) == [], path
        assert source.count("# i18n: ignore") == pragmas, path
        assert path in baseline["enforced"], path
        assert path not in baseline["files"], path


def test_a_reintroduced_literal_in_any_activated_file_is_blocking(i18n_check) -> None:
    baseline = i18n_check.load_baseline(i18n_check.BASELINE)
    for path in PATHS:
        source = (REPO / path).read_text(encoding="utf-8")
        planted = source + '\n\ndef _planted():\n    print("Plant a hardcoded user-facing label")\n'
        violations = i18n_check.scan_source(path, planted)
        assert len(violations) == 1, path
        assert violations[0].kind == "print"
        assert i18n_check.in_enforced_scope(path, baseline["enforced"]), path
