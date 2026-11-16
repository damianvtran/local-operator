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


@pytest.fixture(autouse=True)
def _fresh_memo() -> Any:
    """The resolver memo is process-wide; no case may see another's entries."""
    from local_operator.session import media

    media._resolved_file_memo.clear()
    yield
    media._resolved_file_memo.clear()


def test_unlink_after_read_still_serves_the_bytes_read(
    cfg: Path, tmp_path: Path, monkeypatch: Any
) -> None:
    """R2: no post-read stat turns bytes already held into a 404 (local route parity)."""
    work = tmp_path / "work"
    work.mkdir()
    target = work / "notes.md"
    target.write_bytes(b"# held\n")
    directory = _session(cfg, ["notes.md"])
    original = Path.read_bytes

    def read_then_unlink(self: Path) -> bytes:
        data = original(self)
        if self == target.resolve():
            self.unlink()
        return data

    monkeypatch.setattr(Path, "read_bytes", read_then_unlink)
    resolved = resolve_supplement_file(directory, JOB, 0, cwd=str(work))
    assert isinstance(resolved, SessionMedia) and resolved.data == b"# held\n"
    assert not target.exists()


def test_unchanged_file_is_not_reread_and_a_rewrite_is(
    cfg: Path, tmp_path: Path, monkeypatch: Any
) -> None:
    """S-3: per-chunk re-resolution reuses unchanged bytes; a rewrite re-hashes."""
    import os

    work = tmp_path / "work"
    work.mkdir()
    target = work / "notes.md"
    target.write_bytes(b"first version\n")
    directory = _session(cfg, ["notes.md"])
    reads: list[Path] = []
    original = Path.read_bytes

    def counting(self: Path) -> bytes:
        reads.append(self)
        return original(self)

    monkeypatch.setattr(Path, "read_bytes", counting)
    first = resolve_supplement_file(directory, JOB, 0, cwd=str(work))
    again = resolve_supplement_file(directory, JOB, 0, cwd=str(work))
    assert isinstance(first, SessionMedia) and isinstance(again, SessionMedia)
    assert again.media_id == first.media_id and len(reads) == 1
    # Same size, explicitly later mtime: the stat identity alone must miss the memo.
    target.write_bytes(b"other version\n")
    stat = target.stat()
    os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    changed = resolve_supplement_file(directory, JOB, 0, cwd=str(work))
    assert isinstance(changed, SessionMedia) and len(reads) == 2
    assert changed.media_id != first.media_id and changed.data == b"other version\n"


def test_memo_does_not_bypass_policy(cfg: Path, tmp_path: Path) -> None:
    """A memoised file is still refused once the row no longer names it."""
    work = tmp_path / "work"
    work.mkdir()
    (work / "notes.md").write_text("x")
    directory = _session(cfg, ["notes.md"])
    assert isinstance(resolve_supplement_file(directory, JOB, 0, cwd=str(work)), SessionMedia)
    assert resolve_supplement_file(directory, JOB, 0, cwd="") == SUPPLEMENT_NO_CWD
    assert resolve_supplement_file(directory, JOB, 1, cwd=str(work)) == SUPPLEMENT_MISSING


def test_memo_is_bounded(cfg: Path, tmp_path: Path) -> None:
    from local_operator.session import media

    work = tmp_path / "work"
    work.mkdir()
    names = [f"n{i}.md" for i in range(media._RESOLVED_FILE_MEMO_ENTRIES + 3)]
    for name in names:
        (work / name).write_text(name)
    directory = _session(cfg, names)
    for index in range(len(names)):
        assert isinstance(
            resolve_supplement_file(directory, JOB, index, cwd=str(work)), SessionMedia
        )
    assert len(media._resolved_file_memo) == media._RESOLVED_FILE_MEMO_ENTRIES


# --- S-2: the digest families' authorization facts (supplement_digest_media) -------


def _digest_session(cfg: Path, session_id: str, details: dict[str, Any]) -> None:
    import asyncio

    from local_operator.harness.types import Message
    from local_operator.session.transcript import Transcript

    directory = cfg / "sessions" / session_id
    directory.mkdir(parents=True)

    async def write() -> None:
        transcript = Transcript(directory)
        try:
            await transcript.append_message(Message.assistant("a finished answer"))
            await transcript.append_custom("supplement_v1", details)
        finally:
            transcript.close()

    asyncio.run(write())


@pytest.fixture()
def digests(cfg: Path) -> dict[str, str]:
    """Session A references a component and a sound; session B references neither."""
    from local_operator.session.attachments import AttachmentStore

    store = AttachmentStore(cfg / "attachments")
    component = store.put_bytes(b"<p>component</p>", "text/html")
    sound = store.put_bytes(b"ID3\x04" + bytes(range(64)), "audio/mpeg")
    stray = store.put_bytes(b"never referenced", "audio/mpeg")
    assert component and sound and stray
    base = {"anchor": "a" * 32, "job": JOB, "version": 1, "state": "done", "files": []}
    _digest_session(
        cfg,
        "session-a",
        {
            **base,
            "components": [{"attachment": component.digest, "title": "t", "source": "s"}],
            "sounds": [{"attachment": sound.digest, "mime": "audio/mpeg", "title": "clip"}],
        },
    )
    _digest_session(cfg, "session-b", {**base, "components": [], "sounds": []})
    return {"component": component.digest, "sound": sound.digest, "stray": stray.digest}


def test_digest_controls_are_served(cfg: Path, digests: dict[str, str]) -> None:
    """Positive control, so the three refusals below are refusals of policy, not setup."""
    from local_operator.session.media import supplement_digest_media

    document = supplement_digest_media(cfg, "session-a", digests["component"], "document")
    sound = supplement_digest_media(cfg, "session-a", digests["sound"], "sound")
    assert isinstance(document, SessionMedia) and isinstance(sound, SessionMedia)
    assert sound.mime == "audio/mpeg"


@pytest.mark.parametrize(
    "session_id, which, kind",
    [
        ("session-a", "stray", "sound"),  # unreferenced digest
        ("session-a", "stray", "document"),
        ("session-b", "component", "document"),  # A's digest read as session B
        ("session-b", "sound", "sound"),
        ("session-a", "component", "sound"),  # component digest via the audio family
    ],
)
def test_digest_authorization_refusals(
    cfg: Path, digests: dict[str, str], session_id: str, which: str, kind: str
) -> None:
    from local_operator.session.media import supplement_digest_media

    assert supplement_digest_media(cfg, session_id, digests[which], kind) == SUPPLEMENT_MISSING
