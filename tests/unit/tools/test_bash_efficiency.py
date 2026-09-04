"""Structural regression guards for waiting, bounded capture and redaction."""

from __future__ import annotations

import asyncio
import shlex
import sys

import pytest

from local_operator.harness.types import AbortSignal, ToolContext
from local_operator.tools import builtin


@pytest.mark.asyncio
async def test_bash_races_exit_and_never_rewaits_completed_eof(monkeypatch, tmp_path):
    """Closed pipes are not exit; a done EOF must not become a busy-loop wake.

    The spy proves the scheduling invariant, independent of host load. Waiting
    for a never-fired abort with ALL_COMPLETED used to spend 250 ms per poll.
    Restoring that return_when default fails deterministically here.
    """
    original = asyncio.wait
    calls = []

    async def wait(tasks, **kwargs):
        if any("_pump" in task.get_coro().__qualname__ for task in tasks):
            calls.append(True)
            assert kwargs.get("return_when") == asyncio.FIRST_COMPLETED
            assert all(not task.done() for task in tasks)
        return await original(tasks, **kwargs)

    monkeypatch.setattr(builtin.asyncio, "wait", wait)
    command = f"{shlex.quote(sys.executable)} -c 'import os; os.close(1); os.close(2)'"
    result = await builtin.execute_bash(
        "bash-race", {"command": command}, AbortSignal(), None, ToolContext(cwd=str(tmp_path))
    )
    assert not result.is_error, result.text
    assert "exit code: 0" in result.text
    assert calls


def test_capture_retention_is_independent_of_total_output():
    capture = builtin._BashOutput(limit=1024)
    capture.append(b"HEADER\n" + b"a" * 505)
    for _ in range(1024):
        capture.append(b"b" * 1024)
        assert capture.retained_bytes <= 1024
    capture.append(b"\nFINAL DIAGNOSTIC")
    assert capture.total_bytes > 1024 * 1024
    assert capture.omitted_bytes > 1024 * 1000
    assert capture.decode().startswith("HEADER\n")
    assert capture.decode().endswith("FINAL DIAGNOSTIC")
    assert "bytes omitted" in capture.decode()


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 7, 16, 65536])
def test_pipe_redaction_preserves_unicode_and_secret_boundaries(chunk_size):
    secret = "secret-😸-credential"
    raw = ("before 😺 " + secret + " after " + secret + " done").encode()
    redactor = builtin._PipeRedactor({"TOKEN": secret})
    chunks = [redactor.feed(raw[i : i + chunk_size]) for i in range(0, len(raw), chunk_size)]
    chunks.append(redactor.feed(b"", final=True))
    text = b"".join(chunks).decode()
    assert text == "before 😺 [redacted] after [redacted] done"


def test_redactor_holds_overlapping_known_secret_at_cut():
    redactor = builtin._PipeRedactor({"A": "prefix-secret", "B": "secret"})
    chunks = [
        redactor.feed(b"prefix-"),
        redactor.feed(b"secret after"),
        redactor.feed(b"", final=True),
    ]
    assert b"".join(chunks) == b"[redacted] after"


@pytest.mark.asyncio
async def test_bash_spill_reports_capture_omissions(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    real = builtin._BashOutput
    monkeypatch.setattr(real.__init__, "__defaults__", (16384,))
    command = (
        f'{shlex.quote(sys.executable)} -c \'print("start\\n" + "middle\\n" * 10000 + "FINAL")\''
    )
    result = await builtin.execute_bash(
        "bash-cap", {"command": command}, AbortSignal(), None, ToolContext(cwd=str(tmp_path))
    )
    assert not result.is_error, result.text
    assert "start" in result.text and "FINAL" in result.text
    meta = (result.details or {})["spill"]
    assert meta["complete"] is False
    assert "stored copy is itself head+tail" in result.text


def test_mutation_resources_match_hardlinks_and_new_paths(tmp_path):
    first = tmp_path / "first"
    first.write_text("x")
    alias = tmp_path / "alias"
    alias.hardlink_to(first)
    left = set(builtin._file_resource_keys({"path": str(first)}, str(tmp_path)))
    right = set(builtin._file_resource_keys({"path": str(alias)}, str(tmp_path)))
    assert left & right
    assert not left & set(builtin._file_resource_keys({"path": "new"}, str(tmp_path)))


@pytest.mark.parametrize("names", [("café.txt", "cafe\u0301.txt"), ("CAFÉ.txt", "cafe\u0301.txt")])
def test_new_unicode_aliases_share_scheduler_and_transaction_identity(tmp_path, monkeypatch, names):
    """Pre-creation aliases must conflict even before stat can identify them.

    A macOS probe demonstrated both real transactions entering before either
    created this file. Assert the two independent coordination surfaces: a
    scheduler key alone cannot protect separate parent/child loops. Observe
    transaction hash inputs so a random stripe collision cannot hide a broken
    identity. This remains conservative on normalization-sensitive filesystems.
    """
    paths = [tmp_path / name for name in names]
    assert not any(path.exists() for path in paths)
    resources = [
        set(builtin._file_resource_keys({"path": str(path)}, str(tmp_path))) for path in paths
    ]
    assert resources[0] & resources[1]
    transaction_keys = []

    def hash_spy(key):
        transaction_keys.append(key)
        return hash(key)

    monkeypatch.setattr(builtin, "hash", hash_spy, raising=False)
    for path in paths:
        with builtin._file_transaction(path):
            pass
    assert len(transaction_keys) == 2  # No inode exists yet.
    assert transaction_keys[0] == transaction_keys[1]
