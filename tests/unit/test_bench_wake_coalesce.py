"""R3: overlapping note spans attribute one parent call only once."""

import json
import sqlite3
import time

from scripts import bench_wake_coalesce


def test_overlapping_spans_deduplicate_calls(tmp_path, monkeypatch, capsys):
    now = time.time()
    db = sqlite3.connect(tmp_path / "analytics.db")
    db.execute(
        "create table calls (session_id text, parent_session_id text, purpose text, "
        "ts_ms integer, context_tokens integer)"
    )
    db.execute(
        "insert into calls values ('parent', '', 'turn', ?, 123456)", (int((now - 5) * 1000),)
    )
    db.commit()
    db.close()
    directory = tmp_path / "sessions" / "parent"
    directory.mkdir(parents=True)
    payloads = [
        (
            -10,
            {"role": "tool", "tool_name": "wait", "content": [{"text": "a hub message arrived"}]},
        ),
        (-9, {"custom_type": "hub_message", "details": {"direction": "to_parent", "job_id": "a"}}),
        (-8, {"custom_type": "hub_message", "details": {"direction": "to_parent", "job_id": "b"}}),
        (-2, {"custom_type": "job_result", "details": {"job_id": "a"}}),
        (-1, {"custom_type": "job_result", "details": {"job_id": "b"}}),
    ]
    (directory / "transcript.jsonl").write_text(
        "\n".join(
            json.dumps({"ts": now + delta, "payload": p}, separators=(",", ":"))
            for delta, p in payloads
        )
    )
    monkeypatch.setattr("sys.argv", ["bench", "--home", str(tmp_path)])
    assert bench_wake_coalesce.main() == 0
    output = capsys.readouterr().out
    row = next(line.split() for line in output.splitlines() if line.split()[:1] == ["20"])
    assert row[:5] == ["20", "2", "2", "1", "123,456"]
    assert "NOT demonstrated eliminated calls" in output
