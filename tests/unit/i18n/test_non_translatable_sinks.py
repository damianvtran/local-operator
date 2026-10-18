"""Non-translatable sinks activated by pragma, not by translation.

`team_tool.py` emits a model-facing tool result (the roster and override
description a model reads); `supplements/generator.py` appends model-facing
evidence-envelope tags and an internal generation journal;
`evaluation/runner/episode.py` builds failure-artifact diagnostics. RFC §2.7
keeps model-facing text and dev logs English, so each of these calls carries a
scoped ``# i18n: ignore <reason>`` pragma, the file has no remaining ceiling,
and its path is in the enforced scope. A reintroduced unflagged literal is a
blocking finding — that is what the activation buys.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

#: Path -> the exact number of scoped pragmas the file carries. The count is
#: pinned so deleting code cannot turn a finding-free scan into a "pass"
#: without also deleting the pragma record.
PATHS = {
    "local_operator/tools/team_tool.py": 4,
    "local_operator/supplements/generator.py": 13,
    "local_operator/evaluation/runner/episode.py": 8,
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
    """The negative control, against the REAL baseline and production scanner."""
    baseline = i18n_check.load_baseline(i18n_check.BASELINE)
    for path in PATHS:
        source = (REPO / path).read_text(encoding="utf-8")
        planted = source + '\n\ndef _planted():\n    print("Plant a hardcoded user-facing label")\n'
        violations = i18n_check.scan_source(path, planted)
        assert len(violations) == 1, path
        assert violations[0].kind == "print"
        assert i18n_check.in_enforced_scope(path, baseline["enforced"]), path
