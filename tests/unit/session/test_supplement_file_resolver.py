"""The ONE featured-file resolver (``session/media.resolve_supplement_file``).

Extracted from the relay's local file route so a mesh owner answering a remote
viewer runs the identical policy. These cases pin the extracted contract directly;
``tests/unit/supplements/test_surfaces.py`` pins the local route's HTTP behaviour on
top of it, unchanged.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.session.media import (
    SUPPLEMENT_FILE_MAX_BYTES,
    SUPPLEMENT_MISSING,
    SUPPLEMENT_NO_CWD,
    SUPPLEMENT_TOO_LARGE,
    SessionMedia,
    resolve_supplement_file,
)

JOB = "3f9c1a7e5b20"


def _session(cfg: Path, files: list[str], *, version_rows: int = 1) -> Path:
    directory = cfg / "sessions" / "ab12cd34ef56"
    directory.mkdir(parents=True)
    rows = []
    for version in range(1, version_rows + 1):
        details = {
            "anchor": "a" * 32,
            "job": JOB,
            "version": version,
            "state": "done",
            # Older versions name a decoy, so "newest version wins" is observable.
            "files": [{"path": p} for p in files] if version == version_rows else [],
        }
        rows.append(
            {
                "id": f"{version:032d}",
                "ts": 1791000000.0,
                "type": "custom",
                "payload": {"custom_type": "supplement_v1", "details": details},
            }
        )
    (directory / "transcript.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return directory


@pytest.fixture()
def cfg(tmp_path: Path, monkeypatch: Any) -> Path:
    # The config root sits BESIDE the work dir: the denylist refuses everything
    # under the live config dir, and that rule is not the one under test.
    root = tmp_path / "cfg"
    root.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    # ``~`` is a root too; pin it away from the work dir so "escapes every root"
    # cannot pass by accident on a host whose temp dir sits under home.
    (tmp_path / "home").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return root


def test_serves_bytes_identity_and_allowlisted_type(cfg: Path, tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    (work / "notes.md").write_bytes(b"# hi\n")
    directory = _session(cfg, ["notes.md"], version_rows=2)
    resolved = resolve_supplement_file(directory, JOB, 0, cwd=str(work))
    assert isinstance(resolved, SessionMedia)
    assert resolved.data == b"# hi\n" and resolved.mime == "text/markdown"
    assert resolved.media_id == hashlib.sha256(b"# hi\n").hexdigest()
    assert resolved.total == 5 and resolved.read(2, 2) == b"hi"


def test_relative_path_without_live_cwd_is_no_cwd(cfg: Path) -> None:
    directory = _session(cfg, ["notes.md"])
    assert resolve_supplement_file(directory, JOB, 0, cwd="") == SUPPLEMENT_NO_CWD


def test_scratchpad_file_serves_without_cwd(cfg: Path) -> None:
    """The session's own pad is a root of its own: no live cwd is needed."""
    pad_file = cfg / "sessions" / "ab12cd34ef56" / "scratchpad" / "out.csv"
    directory = _session(cfg, [str(pad_file)])
    pad_file.parent.mkdir()
    pad_file.write_text("a,b\n")
    resolved = resolve_supplement_file(directory, JOB, 0, cwd="")
    assert isinstance(resolved, SessionMedia) and resolved.mime == "text/csv"


@pytest.mark.parametrize(
    "name, body, expected",
    [
        (".env", b"TOKEN=x\n", SUPPLEMENT_MISSING),  # denylist
        ("page.html", b"<script>", SUPPLEMENT_MISSING),  # not on the allowlist
        ("art.svg", b"<svg/>", SUPPLEMENT_MISSING),
        ("big.txt", b"x" * (SUPPLEMENT_FILE_MAX_BYTES + 1), SUPPLEMENT_TOO_LARGE),
    ],
)
def test_refusals(cfg: Path, tmp_path: Path, name: str, body: bytes, expected: str) -> None:
    work = tmp_path / "work"
    work.mkdir()
    (work / name).write_bytes(body)
    directory = _session(cfg, [name])
    assert resolve_supplement_file(directory, JOB, 0, cwd=str(work)) == expected


@pytest.mark.parametrize("job, index", [("../etc/x", 0), ("A" * 12, 0), (JOB, 1), (JOB, -1)])
def test_address_misses(cfg: Path, tmp_path: Path, job: str, index: int) -> None:
    work = tmp_path / "work"
    work.mkdir()
    (work / "notes.md").write_text("x")
    directory = _session(cfg, ["notes.md"])
    assert resolve_supplement_file(directory, job, index, cwd=str(work)) == SUPPLEMENT_MISSING


def test_path_escaping_every_root_is_missing(cfg: Path, tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("not under cwd, home or the pad")
    directory = _session(cfg, ["../outside.txt"])
    assert resolve_supplement_file(directory, JOB, 0, cwd=str(work)) == SUPPLEMENT_MISSING
