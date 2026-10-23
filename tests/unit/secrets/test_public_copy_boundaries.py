"""Boundary tests for the sanctioned public-copy reader.

The reader is a result-redaction exemption, so its security boundary is the
FILE it opens, not the masking pass.  Review round 1 reproduced the writable-root
bypass — an agent writes a credential into a catalogue and reads it back
verbatim — so these tests pin the answers: serve the committed blob at
``HEAD``, refuse a working-tree difference, a symlink escape and an untracked
file, and record every read before returning it.  A hard link is another
working-tree name and does not change ``HEAD``, so it is served.

The fixture is a real miniature git checkout: ``HEAD`` is the whole control, so
a test that never commits is not testing the route.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from local_operator.harness.types import ToolContext
from local_operator.tools.public_copy import execute_read_public_copy

FAKE_GITHUB = "ghp_" + "A" * 36


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _public_repo(tmp_path: Path) -> Path:
    """A committed miniature checkout with a catalogue and a key module."""
    root = tmp_path / "repo"
    catalogue = root / "local_operator/i18n/catalogues/en"
    catalogue.mkdir(parents=True)
    (catalogue / "wire.slash.json").write_text(
        json.dumps(
            {"wire.slash.credential": "Type or paste a secret after a space; masked"},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    keys = root / "local_operator/i18n/keys"
    keys.mkdir(parents=True)
    (keys / "wire_slash.py").write_text(
        'return Msg("wire.slash.credential", {})\n', encoding="utf-8"
    )
    (root / "README.md").write_text("not public copy\n", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    _git(root, "add", "-A")
    _git(root, "-c", "commit.gpgsign=false", "commit", "-qm", "fixture")
    return root


def _context(root: Path, tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=str(root), session_dir=str(tmp_path / "sessions" / "abc123"))


@pytest.mark.asyncio
async def test_a_working_tree_change_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reproduced bypass: write a credential into a catalogue, read it back."""
    root = _public_repo(tmp_path)
    catalogue = root / "local_operator/i18n/catalogues/en/wire.slash.json"
    catalogue.write_text(
        json.dumps({"wire.slash.credential": "GITHUB_TOKEN=" + FAKE_GITHUB}),
        encoding="utf-8",
    )
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    result = await execute_read_public_copy(
        "call-1",
        {
            "path": "local_operator/i18n/catalogues/en/wire.slash.json",
            "key": "wire.slash.credential",
        },
        None,
        None,
        _context(root, tmp_path),
    )

    assert result.is_error
    assert "differs from HEAD" in result.text
    assert FAKE_GITHUB not in result.text


@pytest.mark.asyncio
async def test_an_untracked_public_copy_file_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _public_repo(tmp_path)
    new_file = root / "local_operator/i18n/catalogues/en/new.json"
    new_file.write_text(json.dumps({"wire.slash.new": "public"}), encoding="utf-8")
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    result = await execute_read_public_copy(
        "call-2",
        {"path": "local_operator/i18n/catalogues/en/new.json", "key": "wire.slash.new"},
        None,
        None,
        _context(root, tmp_path),
    )

    assert result.is_error
    assert "not committed at HEAD" in result.text


@pytest.mark.asyncio
async def test_a_clean_crlf_checkout_is_served(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Git for Windows' default: CRLF on disk, LF in the blob, clean index.

    Round 2 measured the raw byte-equality version refusing every read here. The
    route now compares with ``git diff --quiet`` and serves the blob, so the
    platform's line endings cannot kill the audit.
    """
    root = _public_repo(tmp_path)
    _git(root, "config", "core.autocrlf", "true")
    catalogue = root / "local_operator/i18n/catalogues/en/wire.slash.json"
    body = json.dumps(
        {"wire.slash.credential": "Type or paste a secret after a space; masked"},
        ensure_ascii=False,
        indent=2,
    )
    catalogue.write_bytes(body.replace("\n", "\r\n").encode("utf-8") + b"\r\n")
    _git(root, "add", "local_operator/i18n/catalogues/en/wire.slash.json")
    _git(root, "-c", "commit.gpgsign=false", "commit", "-qm", "crlf")
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    result = await execute_read_public_copy(
        "call-crlf",
        {
            "path": "local_operator/i18n/catalogues/en/wire.slash.json",
            "key": "wire.slash.credential",
        },
        None,
        None,
        _context(root, tmp_path),
    )

    assert not result.is_error, result.text
    assert "Type or paste a secret after a space; masked" in result.text


@pytest.mark.asyncio
async def test_a_hard_link_does_not_change_the_served_blob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hard link is another name, not different committed bytes.

    Round 2 measured that the old nlink refusal was a false refuse once the
    route serves the HEAD blob: the link cannot change what HEAD holds. The
    test keeps the old fixture (a second link to a clean catalogue) and asserts
    the read still succeeds.
    """
    root = _public_repo(tmp_path)
    catalogue = root / "local_operator/i18n/catalogues/en/wire.slash.json"
    os.link(catalogue, catalogue.with_name("hard.json"))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    result = await execute_read_public_copy(
        "call-3",
        {
            "path": "local_operator/i18n/catalogues/en/wire.slash.json",
            "key": "wire.slash.credential",
        },
        None,
        None,
        _context(root, tmp_path),
    )

    assert not result.is_error, result.text
    assert "Type or paste a secret after a space; masked" in result.text


@pytest.mark.asyncio
async def test_a_non_git_directory_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The route requires a git checkout; the refusal names that, not 'untracked'."""
    root = tmp_path / "plain"
    catalogue = root / "local_operator/i18n/catalogues/en"
    catalogue.mkdir(parents=True)
    (catalogue / "wire.slash.json").write_text(
        json.dumps({"wire.slash.credential": "public"}), encoding="utf-8"
    )
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    result = await execute_read_public_copy(
        "call-3b",
        {
            "path": "local_operator/i18n/catalogues/en/wire.slash.json",
            "key": "wire.slash.credential",
        },
        None,
        None,
        _context(root, tmp_path),
    )

    assert result.is_error
    assert "not a git checkout" in result.text


@pytest.mark.asyncio
async def test_a_symlink_to_a_non_public_file_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _public_repo(tmp_path)
    catalogue = root / "local_operator/i18n/catalogues/en/wire.slash.json"
    catalogue.with_name("link.json").symlink_to(root / "README.md")
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    result = await execute_read_public_copy(
        "call-4",
        {"path": "local_operator/i18n/catalogues/en/link.json", "key": "x"},
        None,
        None,
        _context(root, tmp_path),
    )

    assert result.is_error
    assert "not a public-copy path" in result.text


@pytest.mark.asyncio
async def test_an_unkeyed_oversized_catalogue_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _public_repo(tmp_path)
    catalogue = root / "local_operator/i18n/catalogues/en/big.json"
    catalogue.write_text(
        json.dumps({f"wire.big.{index}": "x" * 100 for index in range(2100)}),
        encoding="utf-8",
    )
    _git(root, "add", "local_operator/i18n/catalogues/en/big.json")
    _git(root, "-c", "commit.gpgsign=false", "commit", "-qm", "big")
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    result = await execute_read_public_copy(
        "call-5",
        {"path": "local_operator/i18n/catalogues/en/big.json"},
        None,
        None,
        _context(root, tmp_path),
    )

    assert result.is_error
    assert "larger than" in result.text


@pytest.mark.asyncio
async def test_an_empty_key_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _public_repo(tmp_path)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    result = await execute_read_public_copy(
        "call-6",
        {"path": "local_operator/i18n/keys/wire_slash.py", "key": "   "},
        None,
        None,
        _context(root, tmp_path),
    )

    assert result.is_error
    assert "key must not be empty" in result.text


@pytest.mark.asyncio
async def test_every_read_appends_and_the_audit_is_not_truncated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _public_repo(tmp_path)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    context = _context(root, tmp_path)
    request = {
        "path": "local_operator/i18n/catalogues/en/wire.slash.json",
        "key": "wire.slash.credential",
    }

    first = await execute_read_public_copy("call-7", request, None, None, context)
    second = await execute_read_public_copy("call-8", request, None, None, context)

    assert not first.is_error, first.text
    assert not second.is_error, second.text
    log = tmp_path / "config" / "audit" / "public-copy-reads.jsonl"
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(records) == 2
    assert records[0]["sha256"] == records[1]["sha256"]
    assert os.stat(log).st_mode & 0o777 == 0o600
