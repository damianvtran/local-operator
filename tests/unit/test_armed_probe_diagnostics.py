"""Contract for the armed-probe cell's failure-only diagnostics.

``tests/e2e/armed_probe_diagnostics.py`` exists because the CI failure of the
armed-probe e2e cell reported two ``complete`` completion rows and nothing else,
while the turn's ``bash`` result was a synthetic abort and the child's stderr
went to DEVNULL. These tests pin, on FABRICATED rows and files (no runtime, no
child, no e2e), that the evidence the report used to drop is now kept — bounded
and credential-scrubbed — and that the ordinary rows still read as before.
Remove together with that module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from local_operator.redaction_shapes import REDACTION_MARKER
from tests.e2e import armed_probe_diagnostics as diag

# A shape the product scrubber is guaranteed to recognise (vendor-prefixed
# GitHub token), built at runtime so no scanner flags this file.
FAKE_TOKEN = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"


def _complete_row(anchor: str, token: str = "tok-opaque-123") -> dict[str, Any]:
    return {
        "id": f"c-{anchor}",
        "ts": 1.0,
        "type": "custom",
        "payload": {
            "custom_type": "completion_attention",
            "details": {
                "conversation_id": "session/refresharmed1",
                "token": token,
                "anchor": anchor,
                "kind": "complete",
                "cause": "",
                "reason": "",
                "notify": True,
            },
        },
    }


def _aborted_bash_row(text: str = "aborted") -> dict[str, Any]:
    """The row shape measured on the diagnostic run (.rig evidence, attempt 3)."""
    return {
        "id": "m-tool-1",
        "ts": 2.5,
        "type": "message",
        "payload": {
            "kind": "message",
            "role": "tool",
            "content": [{"text": text}],
            "tool_call_id": "call_mock_bash",
            "tool_name": "bash",
            "is_error": True,
            "provider_payload": {
                "details": {"__fault": "aborted", "__synthetic": True},
                "useless": False,
                "duration_s": 0.010868915996979922,
            },
        },
    }


def _ok_tool_row() -> dict[str, Any]:
    return {
        "id": "m-tool-ok",
        "ts": 2.0,
        "type": "message",
        "payload": {
            "kind": "message",
            "role": "tool",
            "content": [{"text": "fine"}],
            "tool_call_id": "call_ok",
            "tool_name": "read",
            "is_error": False,
            "provider_payload": {"details": {}, "duration_s": 0.2},
        },
    }


def _write(path: Path, rows: list[dict[str, Any]], torn_tail: bool = False) -> Path:
    text = "\n".join(json.dumps(row) for row in rows) + "\n"
    if torn_tail:
        text += '{"type": "message", "payl'
    path.write_text(text, encoding="utf-8")
    return path


def test_synthetic_aborted_tool_row_is_reported_with_its_metadata(tmp_path: Path) -> None:
    transcript = _write(
        tmp_path / "transcript.jsonl",
        [_complete_row("a"), _ok_tool_row(), _aborted_bash_row(), _complete_row("b")],
        torn_tail=True,
    )

    rows = diag.tool_fault_rows(transcript)

    assert rows == [
        {
            "id": "m-tool-1",
            "ts": 2.5,
            "tool_name": "bash",
            "tool_call_id": "call_mock_bash",
            "is_error": True,
            "details": {"__fault": "aborted", "__synthetic": True},
            "duration_s": 0.010868915996979922,
            "content": '[{"text": "aborted"}]',
        }
    ]
    report = diag.evidence(transcript, None)
    assert "errored/synthetic tool-result rows (1):" in report
    assert '"__fault": "aborted"' in report and '"__synthetic": true' in report
    assert "does not name the abort's caller" in report


def test_completion_rows_stay_visible_with_their_token_masked() -> None:
    rows = [_complete_row("a")["payload"]["details"], _complete_row("b")["payload"]["details"]]

    shown = diag.redacted_rows(rows)

    assert [row["kind"] for row in shown] == ["complete", "complete"]
    assert [row["anchor"] for row in shown] == ["a", "b"]
    assert all(row["token"] == REDACTION_MARKER for row in shown)
    assert shown[0]["conversation_id"] == "session/refresharmed1"
    # The ASSERTION still reads the raw rows; only the report is masked.
    assert rows[0]["token"] == "tok-opaque-123"


def test_tool_content_is_bounded_and_scrubbed(tmp_path: Path) -> None:
    transcript = _write(
        tmp_path / "transcript.jsonl",
        [_aborted_bash_row(f"export GITHUB_TOKEN={FAKE_TOKEN} " + "x" * 5000)],
    )

    (row,) = diag.tool_fault_rows(transcript)

    assert FAKE_TOKEN not in row["content"]
    assert len(row["content"]) <= diag.MAX_EXCERPT_CHARS + 40
    assert row["content"].endswith("chars cut]")


def test_fault_rows_are_capped(tmp_path: Path) -> None:
    transcript = _write(
        tmp_path / "transcript.jsonl", [_aborted_bash_row() for _ in range(diag.MAX_FAULT_ROWS + 5)]
    )

    report = diag.evidence(transcript, None)

    assert f"({diag.MAX_FAULT_ROWS + 5}, oldest 5 omitted)" in report
    assert report.count('"tool_name": "bash"') == diag.MAX_FAULT_ROWS


def test_stderr_tail_keeps_the_end_bounded_and_scrubbed(tmp_path: Path) -> None:
    stderr = tmp_path / "armed-child-stderr.log"
    stderr.write_bytes(
        b"boot noise\n" * 2000 + f"Traceback ...\nAuthorization: Bearer {FAKE_TOKEN}\n".encode()
    )

    tail = diag.stderr_tail(stderr)

    assert tail.startswith("[first ")
    assert "Traceback ..." in tail
    assert FAKE_TOKEN not in tail
    assert len(tail) <= diag.MAX_STDERR_BYTES + 60


def test_missing_or_empty_inputs_never_raise(tmp_path: Path) -> None:
    empty = tmp_path / "empty.log"
    empty.write_bytes(b"")

    assert diag.stderr_tail(empty) == "(empty)"
    assert diag.stderr_tail(None) == "(no stderr file)"
    assert diag.stderr_tail(tmp_path / "absent.log").startswith("(stderr unreadable:")
    assert diag.tool_fault_rows(tmp_path / "absent.jsonl") == []
    assert "rows (0):" in diag.evidence(tmp_path / "absent.jsonl", empty)
