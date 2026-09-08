"""Conversation birth cannot follow activity, process generations or inherited bytes."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from local_operator.harness.types import Message
from local_operator.session.creation import (
    CREATED_AT_NAME,
    ensure_session_created_at,
    session_created_at,
)
from local_operator.session.transcript import Transcript


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    for key in tuple(os.environ):
        if key.startswith("CMUX_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))


def test_birth_survives_append_reopen_compaction_and_directory_loss(tmp_path):
    directory = tmp_path / "conversation"
    transcript = Transcript(directory)
    birth = session_created_at(directory)
    assert birth > 0
    asyncio.run(transcript.append_message(Message.user("hello")))
    os.utime(transcript.path, (1, 1))
    reopened = Transcript(directory)
    assert session_created_at(directory) == birth
    asyncio.run(reopened.compact_file())
    assert session_created_at(directory) == birth
    shutil.rmtree(directory)
    asyncio.run(transcript.append_message(Message.assistant("restored")))
    assert session_created_at(directory) == birth


def test_deferred_materialization_and_precreated_claim(tmp_path):
    directory = tmp_path / "claimed"
    directory.mkdir()
    transcript = Transcript(directory, defer_materialise=True)
    assert not (directory / CREATED_AT_NAME).exists()
    asyncio.run(transcript.append_message(Message.user("materialize")))
    birth = session_created_at(directory)
    assert birth > 0
    assert json.loads((directory / CREATED_AT_NAME).read_text()) == birth


def test_deferred_owner_adopts_competing_materialization(tmp_path):
    directory = tmp_path / "race"
    deferred = Transcript(directory, defer_materialise=True)
    first = Transcript(directory)
    birth = session_created_at(directory)
    asyncio.run(first.append_message(Message.user("first")))
    asyncio.run(deferred.append_message(Message.assistant("second")))
    shutil.rmtree(directory)
    asyncio.run(deferred.append_message(Message.user("self-heal")))
    assert session_created_at(directory) == birth


def test_legacy_birthtime_and_read_only_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(
        type(tmp_path), "stat", lambda *args, **kwargs: SimpleNamespace(st_birthtime=42)
    )
    assert session_created_at(tmp_path) == 42

    def readonly(*args):
        raise OSError("read-only")

    monkeypatch.setattr("local_operator.session.creation.os.link", readonly)
    assert ensure_session_created_at(tmp_path, 999) == 42
    assert list(tmp_path.iterdir()) == []


def test_creation_metadata_does_not_consume_retention_budget(tmp_path):
    from local_operator.session.cleanup import _dir_bytes

    (tmp_path / "transcript.jsonl").write_text("row\n")
    before = _dir_bytes(tmp_path)
    ensure_session_created_at(tmp_path, 1788841600.123)
    assert before == 4
    assert _dir_bytes(tmp_path) == before


def test_publish_is_atomic_no_clobber(tmp_path):
    with ThreadPoolExecutor(max_workers=8) as pool:
        dates = list(pool.map(lambda n: ensure_session_created_at(tmp_path, n), range(10, 30)))
    assert len(set(dates)) == 1
    assert session_created_at(tmp_path) == dates[0]
    assert ensure_session_created_at(tmp_path, 999) == dates[0]
    assert [p.name for p in tmp_path.iterdir()] == [CREATED_AT_NAME]


@pytest.mark.parametrize("bad", ["null", "true", '"123"', "-1", "NaN", "Infinity", "{}"])
def test_corrupt_metadata_uses_immutable_legacy_fallback(tmp_path, monkeypatch, bad):
    (tmp_path / CREATED_AT_NAME).write_text(bad)
    (tmp_path / "origin.json").write_text('{"origin":"fork","forked_at":123}')
    assert session_created_at(tmp_path) == 123
    (tmp_path / "origin.json").unlink()
    monkeypatch.setattr(type(tmp_path), "stat", lambda *args, **kwargs: SimpleNamespace())
    assert session_created_at(tmp_path) == 0
    assert ensure_session_created_at(tmp_path, 999) == 0


def test_legacy_load_does_not_mint_current_date(tmp_path, monkeypatch):
    (tmp_path / "transcript.jsonl").write_text("")
    (tmp_path / "origin.json").write_text('{"origin":"fork","forked_at":123}')
    Transcript(tmp_path)
    assert session_created_at(tmp_path) == 123


def test_fork_has_own_birth_not_parent_or_inherited_journal(tmp_path):
    from local_operator.fork import fork_session

    parent = tmp_path / "sessions" / "parent"
    transcript = Transcript(parent)
    asyncio.run(transcript.append_message(Message.user("parent")))
    (parent / CREATED_AT_NAME).write_text("1")
    fork_id = fork_session(tmp_path, "parent")
    fork = parent.parent / fork_id
    assert session_created_at(parent) == 1
    assert session_created_at(fork) > 1
    assert (fork / "transcript.jsonl").read_bytes() == transcript.path.read_bytes()
