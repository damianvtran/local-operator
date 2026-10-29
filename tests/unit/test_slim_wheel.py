"""Regressions for ``scripts/slim_wheel.py`` and ``scripts/check_dist_size.py``.

Both are release build tools run by path in CI (``publish.yml``, ``ci.yml``,
``release-gate.yml``), so they are loaded with ``runpy.run_path`` like
``test_smoke_wheel.py`` does. Every wheel here is a tiny synthetic one built in
``tmp_path``: the properties under test — comments gone, pragmas/shebang/coding kept,
line numbers unchanged, a RECORD that validates, fail-closed on an unverifiable module,
the size budget and the wheels-only upload directory — do not need the real 20 MB wheel.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import runpy
import traceback
import zipfile
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SLIM: dict[str, Any] = runpy.run_path(str(REPO_ROOT / "scripts" / "slim_wheel.py"))
SIZE: dict[str, Any] = runpy.run_path(str(REPO_ROOT / "scripts" / "check_dist_size.py"))

DIST_INFO = "demo-1.0.dist-info"

MODULE = (
    "#!/usr/bin/env python3\n"
    "# -*- coding: utf-8 -*-\n"
    '"""Docstring # with a hash, which is a string and stays."""\n'
    "# a plain comment line\n"
    "import os  # an inline comment\n"
    "x = os.sep  # type: ignore[attr-defined]\n"
    "y = 1  # noqa: E501\n"
    "\n"
    "    # an indented comment\n"
    "def boom():\n"
    "    # why this raises\n"
    "    raise RuntimeError('# not a comment')  # inline\n"
)


def _build_wheel(directory: Path, members: dict[str, bytes], *, mode: int = 0o644) -> Path:
    """A minimal ``py3-none-any`` wheel whose RECORD matches its members."""
    wheel = directory / "demo-1.0-py3-none-any.whl"
    record_name = f"{DIST_INFO}/RECORD"
    members = {**members, f"{DIST_INFO}/WHEEL": b"Wheel-Version: 1.0\nTag: py3-none-any\n"}
    with zipfile.ZipFile(wheel, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            info = zipfile.ZipInfo(name, date_time=(2020, 2, 3, 4, 5, 6))
            info.external_attr = (0o755 if name.endswith(".sh") else mode) << 16
            zf.writestr(info, data)
        rows = [",".join(SLIM["_record_row"](n, d)) for n, d in members.items()]
        zf.writestr(record_name, "\n".join(rows + [f"{record_name},,"]) + "\n")
    return wheel


def _read(wheel: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(wheel) as zf:
        return {name: zf.read(name) for name in zf.namelist()}


def test_strips_comments_and_keeps_shebang_coding_pragmas_and_docstrings() -> None:
    slimmed = SLIM["strip_comments"](MODULE)
    lines = slimmed.split("\n")
    assert lines[0] == "#!/usr/bin/env python3"
    assert lines[1] == "# -*- coding: utf-8 -*-"
    assert lines[2] == '"""Docstring # with a hash, which is a string and stays."""'
    assert lines[3] == ""
    assert lines[4] == "import os"
    assert lines[5] == "x = os.sep  # type: ignore[attr-defined]"
    assert lines[6] == "y = 1  # noqa: E501"
    assert lines[8] == ""
    assert lines[10] == ""
    assert lines[11] == "    raise RuntimeError('# not a comment')"
    assert "plain comment" not in slimmed and "why this raises" not in slimmed


def test_kept_pragma_spellings() -> None:
    for text in ("# type: ignore", "# noqa", "#noqa: F401", "# pragma: no cover", "# pyright: x"):
        assert SLIM["_kept"](text, 10), text
    assert SLIM["_kept"]("# fmt: off", 10)
    assert not SLIM["_kept"]("# a typed note", 10)
    # A shebang or coding line counts only where the interpreter reads it.
    assert not SLIM["_kept"]("#!/bin/sh", 2)
    assert not SLIM["_kept"]("# coding: latin-1", 3)


def test_line_numbers_survive_so_tracebacks_point_at_the_same_line(tmp_path: Path) -> None:
    """A traceback through the stripped module names the original line number."""

    def raise_line(source: str, name: str) -> int:
        path = tmp_path / name
        path.write_text(source)
        namespace: dict[str, Any] = {}
        exec(compile(source, str(path), "exec"), namespace)
        with pytest.raises(RuntimeError) as caught:
            namespace["boom"]()
        lineno = traceback.extract_tb(caught.value.__traceback__)[-1].lineno
        assert lineno is not None
        return lineno

    original = raise_line(MODULE, "orig.py")
    assert original == 12
    assert raise_line(SLIM["strip_comments"](MODULE), "slim.py") == original


def test_crlf_and_comment_only_tail_keep_line_endings() -> None:
    source = "a = 1  # one\r\n# two\r\nb = 2\r\n# tail without newline"
    slimmed = SLIM["strip_comments"](source)
    assert slimmed == "a = 1\r\n\r\nb = 2\r\n"
    assert slimmed.count("\n") == source.count("\n")


def test_slims_the_wheel_in_place_with_a_valid_record(tmp_path: Path) -> None:
    payload = b"\x00binary # not python\n"
    wheel = _build_wheel(
        tmp_path,
        {"demo/__init__.py": MODULE.encode(), "demo/data.bin": payload, "demo/run.sh": b"#!x\n"},
    )
    with zipfile.ZipFile(wheel) as zf:
        original_infos = {i.filename: (i.date_time, i.external_attr) for i in zf.infolist()}

    before, after = SLIM["slim_wheel"](wheel)

    assert 0 < after < before
    members = _read(wheel)
    assert b"plain comment" not in members["demo/__init__.py"]
    assert members["demo/data.bin"] == payload
    with zipfile.ZipFile(wheel) as zf:
        for info in zf.infolist():
            assert (info.date_time, info.external_attr) == original_infos[info.filename]
    record_name = f"{DIST_INFO}/RECORD"
    rows = list(csv.reader(io.StringIO(members[record_name].decode())))
    assert {row[0] for row in rows} == set(members)
    for path, digest, size in rows:
        if path == record_name:
            assert digest == size == ""
            continue
        data = members[path]
        expected = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
        assert digest == f"sha256={expected.decode()}" and int(size) == len(data)

    # Idempotent: a second pass rewrites byte-identical members.
    SLIM["slim_wheel"](wheel)
    assert _read(wheel) == members
    assert not list(tmp_path.glob(".slim-*")), "the temporary wheel was left behind"


def test_the_slimmed_wheel_keeps_the_source_wheels_permissions(tmp_path: Path) -> None:
    """``mkstemp`` creates 0600 and ``os.replace`` keeps it; the slimmer must not."""
    wheel = _build_wheel(tmp_path, {"demo/__init__.py": MODULE.encode()})
    wheel.chmod(0o644)
    SLIM["slim_wheel"](wheel)
    assert wheel.stat().st_mode & 0o777 == 0o644


def test_verification_parses_the_written_bytes_so_a_lost_cookie_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The check covers the shipped bytes: dropping a PEP 263 cookie is caught.

    The text-level AST is unchanged when a ``# coding: latin-1`` line is blanked, but the
    written latin-1 bytes are no longer valid UTF-8 — which only parsing the bytes sees.
    """
    data = "# coding: latin-1\ns = 'caf\xe9'\n".encode("latin-1")
    assert SLIM["slim_module"]("demo/m.py", data) == data
    monkeypatch.setitem(
        SLIM["slim_module"].__globals__,
        "strip_comments",
        lambda source: source.replace("# coding: latin-1", ""),
    )
    with pytest.raises(SLIM["SlimError"], match="demo/m.py"):
        SLIM["slim_module"]("demo/m.py", data)


def test_unparseable_module_fails_closed_and_leaves_the_wheel_untouched(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    wheel = _build_wheel(tmp_path, {"demo/bad.py": b"def broken(:\n    pass  # c\n"})
    original = wheel.read_bytes()
    assert SLIM["main"]([str(wheel)]) == 1
    assert "demo/bad.py" in capsys.readouterr().err
    assert wheel.read_bytes() == original
    assert not list(tmp_path.glob(".slim-*"))


def test_an_ast_mismatch_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stripper bug that changes code is caught by the AST comparison, not shipped."""
    # run_path returns a COPY of the module namespace; patch the dict the function
    # actually resolves its globals through.
    monkeypatch.setitem(
        SLIM["slim_module"].__globals__, "strip_comments", lambda source: source.replace("1", "2")
    )
    with pytest.raises(SLIM["SlimError"], match="AST differs"):
        SLIM["slim_module"]("demo/m.py", b"x = 1\n")


def test_a_line_count_mismatch_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        SLIM["slim_module"].__globals__, "strip_comments", lambda source: source + "\n"
    )
    with pytest.raises(SLIM["SlimError"], match="line count"):
        SLIM["slim_module"]("demo/m.py", b"x = 1\n")


def test_refuses_a_platform_wheel(tmp_path: Path) -> None:
    wheel = tmp_path / "demo-1.0-py3-none-macosx_11_0_universal2.whl"
    wheel.write_bytes(b"")
    assert SLIM["main"]([str(wheel)]) == 1


def _sized(path: Path, size: int) -> Path:
    with path.open("wb") as handle:
        handle.truncate(size)
    return path


def test_size_guard_passes_within_budget_and_warns_near_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _sized(tmp_path / "a-1.0-py3-none-any.whl", 14 * SIZE["MIB"])
    assert SIZE["main"]([str(tmp_path)]) == 0
    assert "::warning::" not in capsys.readouterr().out
    _sized(tmp_path / "a-1.0-py3-none-any.whl", 17 * SIZE["MIB"])
    assert SIZE["main"]([str(tmp_path)]) == 0
    assert "::warning::" in capsys.readouterr().out


def test_size_guard_fails_over_budget(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _sized(tmp_path / "a-1.0-py3-none-any.whl", SIZE["BUDGET"] + 1)
    assert SIZE["main"]([str(tmp_path)]) == 1
    assert "over the 18 MiB budget" in capsys.readouterr().err


def test_a_tighter_budget_fails_what_the_release_budget_passes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """PR CI's ``--budget-mib 17.5`` goes red before the release's 18 MiB would."""
    _sized(tmp_path / "a-1.0-py3-none-any.whl", int(17.75 * SIZE["MIB"]))
    assert SIZE["main"]([str(tmp_path)]) == 0
    assert SIZE["main"](["--budget-mib", "17.5", str(tmp_path)]) == 1
    assert "over the 17.5 MiB budget" in capsys.readouterr().err


def test_the_budget_flag_cannot_raise_the_line(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        SIZE["main"](["--budget-mib", "19", str(tmp_path)])


def test_size_guard_fails_on_an_empty_directory(tmp_path: Path) -> None:
    assert SIZE["main"]([str(tmp_path)]) == 1


def test_wheels_only_rejects_an_sdist_in_the_upload_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _sized(tmp_path / "a-1.0-py3-none-any.whl", 1024)
    (tmp_path / ".gitignore").write_text("*\n")  # what `uv build` drops in its out-dir
    assert SIZE["main"](["--wheels-only", str(tmp_path)]) == 0
    _sized(tmp_path / "a-1.0.tar.gz", 1024)
    assert SIZE["main"]([str(tmp_path)]) == 0, "without the flag an sdist is not checked"
    assert SIZE["main"](["--wheels-only", str(tmp_path)]) == 1
    assert "a-1.0.tar.gz is not a wheel" in capsys.readouterr().err
