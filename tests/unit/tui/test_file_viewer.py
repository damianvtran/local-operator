"""The viewer bounds bytes before decoding and never follows an untrusted path."""
from pathlib import Path

import pytest

from local_operator.tui.file_viewer import (
    MAX_BYTES, MAX_LINES, ViewerError, read_document, viewer_kind,
)


@pytest.mark.parametrize(("name", "kind"), [
    ("a.py", "code"), ("a.md", "markdown"), ("a.txt", "text"), ("a.html", "code"),
    ("a.csv", "text"), ("a.pdf", None), ("a.png", None),
])
def test_kind(name: str, kind: str | None) -> None:
    assert viewer_kind(Path(name)) == kind


def test_bounded_bytes(tmp_path: Path) -> None:
    path = tmp_path / "large.txt"
    path.write_bytes(b"x" * (MAX_BYTES + 100))
    doc = read_document("large.txt", cwd=tmp_path)
    assert doc.truncated and len(doc.text.encode()) <= MAX_BYTES


def test_bounded_lines_and_replacement(tmp_path: Path) -> None:
    path = tmp_path / "large.txt"
    path.write_bytes(b"\xff\n" * (MAX_LINES + 1))
    doc = read_document("large.txt", cwd=tmp_path)
    assert doc.truncated and len(doc.text.splitlines()) == MAX_LINES
    assert "\ufffd" in doc.text


def test_binary_and_missing(tmp_path: Path) -> None:
    (tmp_path / "binary.txt").write_bytes(b"hello\x00there")
    for name in ("binary.txt", "missing.txt"):
        with pytest.raises(ViewerError):
            read_document(name, cwd=tmp_path)


def test_escape_and_changed_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "safe.txt"
    path.write_text("before")
    doc = read_document("safe.txt", cwd=tmp_path)
    path.write_text("changed")
    with pytest.raises(ViewerError, match="changed"):
        read_document("safe.txt", cwd=tmp_path, expected=doc.identity)
    with pytest.raises(ViewerError):
        read_document("../escape.txt", cwd=tmp_path)


def test_refuses_non_regular_files_before_open(tmp_path: Path, monkeypatch) -> None:
    import os
    from local_operator.tui import file_viewer

    if not hasattr(os, "mkfifo"):
        pytest.skip("named FIFO creation is unavailable on this platform")
    os.mkfifo(tmp_path / "pipe.txt")
    (tmp_path / "dir.txt").mkdir()
    opened = []
    with monkeypatch.context() as patch:
        patch.setattr(file_viewer.os, "open", lambda *args: opened.append(args))
        for name in ("pipe.txt", "dir.txt"):
            with pytest.raises(ViewerError):
                read_document(name, cwd=tmp_path)
        assert opened == []


def test_refuses_symlink_escape_and_executable(tmp_path: Path, monkeypatch) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: allowed))
    (tmp_path / "outside.txt").write_text("outside")
    (allowed / "link.txt").symlink_to(tmp_path / "outside.txt")
    (allowed / "exec.py").write_text("print('no')")
    (allowed / "exec.py").chmod(0o700)
    for name in ("link.txt", "exec.py", "../outside.txt"):
        with pytest.raises(ViewerError):
            read_document(name, cwd=allowed)


@pytest.mark.asyncio
async def test_actual_reader_cancellation_closes_descriptor(tmp_path: Path, monkeypatch) -> None:
    import asyncio
    from local_operator.tui import file_viewer

    (tmp_path / "large.txt").write_bytes(b"x" * MAX_BYTES)
    handles = []
    original = file_viewer.os.fdopen

    def capture(*args, **kwargs):
        handle = original(*args, **kwargs)
        handles.append(handle)
        return handle
    monkeypatch.setattr(file_viewer.os, "fdopen", capture)
    task = asyncio.create_task(file_viewer.read_document_async("large.txt", cwd=tmp_path))
    await asyncio.sleep(0)
    assert len(handles) == 1 and not handles[0].closed
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert handles[0].closed


def test_view_is_tui_only_and_path_completion_has_no_reference_sigil() -> None:
    from local_operator.slash_commands import SLASH_COMMANDS
    from local_operator.tui.autocomplete import file_completion_token

    command = next(c for c in SLASH_COMMANDS if c.name == "view")
    assert not command.desktop_destination
    token = file_completion_token("/view nested/a")
    assert token and token.query == "nested/a"
    assert file_completion_token("/view nested/a", 0) is None
    from local_operator.session.frontend_state import _FRONTEND_LOCAL_SLASHES
    assert "view" in _FRONTEND_LOCAL_SLASHES


@pytest.mark.parametrize("data", [b"x" * MAX_BYTES, b"x\n" * MAX_LINES, b""])
def test_exact_bounds_are_not_truncated(tmp_path: Path, data: bytes) -> None:
    (tmp_path / "exact.txt").write_bytes(data)
    document = read_document("exact.txt", cwd=tmp_path)
    assert not document.truncated
    assert document.text == data.decode()


def test_unreadable_and_unsupported_refusals(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "report.pdf").write_bytes(b"%PDF")
    with pytest.raises(ViewerError, match="unsupported"):
        read_document("report.pdf", cwd=tmp_path)
    (tmp_path / "private.txt").write_text("private")
    import os
    original = os.open

    def denied(path, *args, **kwargs):
        if Path(path) == tmp_path / "private.txt":
            raise PermissionError("denied")
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(os, "open", denied)
        with pytest.raises(ViewerError, match="unreadable"):
            read_document("private.txt", cwd=tmp_path)


@pytest.mark.parametrize("filename", [
    r"C:\Users\Test\report with spaces # 20%.txt",
    r"\\server\share\folder name\report # %20.txt",
    r"\\localhost\share\report.txt",
])
def test_windows_uri_roundtrip_uses_drive_and_unc_authority(filename: str) -> None:
    from pathlib import PureWindowsPath
    from local_operator.tui.link_targets import file_uri_to_path

    original = PureWindowsPath(filename)
    decoded = PureWindowsPath(file_uri_to_path(original.as_uri(), windows=True))
    assert decoded == original and decoded.is_absolute()
    assert decoded.drive == original.drive


def test_native_uri_roundtrip_decodes_once(tmp_path: Path) -> None:
    from local_operator.tui.link_targets import file_uri_to_path

    original = tmp_path / "source # literal %20.txt"
    assert Path(file_uri_to_path(original.as_uri())) == original
    original.write_text("the percent sequence is a filename, not another escape")
    assert read_document(str(original), cwd=tmp_path).path == original


def test_absolute_home_path_has_a_cross_drive_safe_spelling(tmp_path, monkeypatch) -> None:
    import os
    from local_operator.tui.link_targets import file_affordance_spelling

    home = tmp_path / "home"
    home.mkdir()
    original = home / "allowed.txt"
    original.write_text("allowed on another drive")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    with monkeypatch.context() as patch:
        def cross_drive(*args):
            raise ValueError("different drives")
        patch.setattr(os.path, "relpath", cross_drive)
        assert file_affordance_spelling(str(original), cwd=tmp_path) == "~/allowed.txt"
        assert read_document(str(original), cwd=tmp_path).text == "allowed on another drive"
        assert file_affordance_spelling(str(tmp_path / "outside.txt"), cwd=home) == ""


def test_reader_works_without_optional_posix_open_flags(tmp_path: Path, monkeypatch) -> None:
    import os

    (tmp_path / "portable.txt").write_text("portable\r\nUTF-8: café")
    with monkeypatch.context() as patch:
        patch.delattr(os, "O_NOFOLLOW", raising=False)
        patch.delattr(os, "O_NONBLOCK", raising=False)
        document = read_document("portable.txt", cwd=tmp_path)
        assert document.text == "portable\r\nUTF-8: café"


@pytest.mark.asyncio
async def test_external_opener_is_argv_safe_and_reaps_on_cancel(tmp_path, monkeypatch) -> None:
    import asyncio
    import sys
    from local_operator.tui import attachments

    children = []
    original = asyncio.create_subprocess_exec

    async def capture(*args, **kwargs):
        process = await original(*args, **kwargs)
        children.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", capture)
    monkeypatch.setattr(attachments, "opener_argv", lambda path: [
        sys.executable, "-c", "import sys;sys.exit(0 if sys.argv[1].endswith('.txt') else 1)",
        path,
    ])
    dangerous_name = str(tmp_path / "file; touch NEVER_CREATED.txt")
    assert await attachments.open_path_quietly(dangerous_name)
    assert not (tmp_path / "NEVER_CREATED.txt").exists()
    assert children[0].returncode == 0
    monkeypatch.setattr(attachments, "opener_argv", lambda path: [
        sys.executable, "-c", "import time;time.sleep(60)", path,
    ])
    task = asyncio.create_task(attachments.open_path_quietly(dangerous_name))
    for _ in range(40):
        if len(children) == 2:
            break
        await asyncio.sleep(0.01)
    assert len(children) == 2
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert children[1].returncode is not None
