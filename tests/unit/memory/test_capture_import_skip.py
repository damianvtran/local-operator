"""Automatic memory capture skips sessions an import marked ``capture_memory: false``."""

from __future__ import annotations

import json

from local_operator.memory_capture import import_capture_disabled


def test_absent_manifest_keeps_capture_enabled(tmp_path):
    assert import_capture_disabled(tmp_path) is False
    assert import_capture_disabled(None) is False


def test_import_manifest_disables_capture(tmp_path):
    (tmp_path / "harness-import.json").write_text(json.dumps({"capture_memory": False}))
    assert import_capture_disabled(tmp_path) is True


def test_malformed_or_true_manifest_keeps_capture_enabled(tmp_path):
    (tmp_path / "harness-import.json").write_text("{not json")
    assert import_capture_disabled(tmp_path) is False
    (tmp_path / "harness-import.json").write_text(json.dumps({"capture_memory": True}))
    assert import_capture_disabled(tmp_path) is False
