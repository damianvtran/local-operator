from __future__ import annotations

import json
from pathlib import Path

from local_operator.session_files.scan import FileScan


def feed(scan: FileScan, rows: list[dict]) -> None:
    for ordinal, row in enumerate(rows):
        scan.feed(ordinal, (json.dumps(row, separators=(",", ":")) + "\n").encode())


def record(scan: FileScan, path: Path):
    return scan.records[f"p:{path}"]


def test_operator_reference_file_url_and_prose_paths(tmp_path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "work"
    cwd.mkdir()
    ref = tmp_path / "brief.pdf"
    ref.write_bytes(b"%PDF")
    url = tmp_path / "chart.png"
    url.write_bytes(b"png")
    prose = tmp_path / "notes.md"
    prose.write_text("# n")
    scan = FileScan(cwd=str(cwd), home=str(home))
    feed(
        scan,
        [
            {
                "id": "u1",
                "ts": 1,
                "type": "message",
                "payload": {
                    "kind": "message",
                    "role": "user",
                    "content": [
                        {
                            "text": (
                                f'<reference path="{ref}">\n'
                                f"see file://{url}\n"
                                f"and {prose}, plus relative.png\n"
                                "and /missing/path.pdf"
                            )
                        }
                    ],
                },
            }
        ],
    )
    assert record(scan, ref).origin == "mentioned"
    assert record(scan, url).origin == "mentioned"
    assert record(scan, prose).origin == "mentioned"
    assert f"p:{cwd / 'relative.png'}" not in scan.records
    assert "p:/missing/path.pdf" not in scan.records


def test_tool_argument_mentions_and_user_attachment_digests(tmp_path) -> None:
    seen = tmp_path / "input.csv"
    seen.write_text("a,b")
    digest = "a" * 32
    scan = FileScan(cwd=str(tmp_path), home=str(tmp_path))
    feed(
        scan,
        [
            {
                "id": "a1",
                "ts": 1,
                "type": "message",
                "payload": {
                    "kind": "message",
                    "role": "assistant",
                    "content": [],
                    "tool_calls": [
                        {"id": "c1", "name": "read", "arguments": {"path": str(seen), "i": "read"}}
                    ],
                },
            },
            {
                "id": "u1",
                "ts": 2,
                "type": "message",
                "payload": {
                    "kind": "message",
                    "role": "user",
                    "content": [
                        {"text": "look"},
                        {"kind": "image", "attachment": digest, "content_type": "image/png"},
                    ],
                },
            },
        ],
    )
    assert record(scan, seen).origin == "mentioned"
    artifact = scan.records[f"a:{digest}"]
    assert artifact.origin == "mentioned" and artifact.kind == "image"


def test_a_mentioned_file_is_upgraded_by_a_write(tmp_path) -> None:
    path = tmp_path / "out.txt"
    scan = FileScan(cwd=str(tmp_path), home=str(tmp_path))
    feed(
        scan,
        [
            {
                "id": "u1",
                "ts": 1,
                "type": "message",
                "payload": {
                    "kind": "message",
                    "role": "user",
                    "content": [{"text": f"{path}"}],
                },
            }
        ],
    )
    path.write_text("written later")
    feed(
        scan,
        [
            {
                "id": "a1",
                "ts": 2,
                "type": "message",
                "payload": {
                    "kind": "message",
                    "role": "assistant",
                    "content": [],
                    "tool_calls": [
                        {"id": "c1", "name": "write", "arguments": {"path": str(path)}}
                    ],
                },
            },
            {
                "id": "t1",
                "ts": 3,
                "type": "message",
                "payload": {
                    "kind": "message",
                    "role": "tool",
                    "content": [{"text": "ok"}],
                    "tool_call_id": "c1",
                    "tool_name": "write",
                    "provider_payload": {"details": {"path": str(path)}},
                },
            },
        ],
    )
    assert record(scan, path).origin == "written"


def test_directories_and_denied_paths_are_not_mentioned(tmp_path) -> None:
    directory = tmp_path / "folder.png"
    directory.mkdir()
    secret = tmp_path / "secret.png"
    secret.write_bytes(b"x")
    scan = FileScan(cwd=str(tmp_path), home=str(tmp_path), deny=lambda path: path == str(secret))
    feed(
        scan,
        [
            {
                "id": "u1",
                "ts": 1,
                "type": "message",
                "payload": {
                    "kind": "message",
                    "role": "user",
                    "content": [{"text": f"{directory} {secret}"}],
                },
            }
        ],
    )
    assert scan.records == {}
