"""Transcript compression: the safety protocol, the codecs, the guards, the reads.

The ordering this file is really testing is the module's promise: a compressed
archive is published only after it has been decoded back and hashed to the
original's sha256, the source is removed only after a second re-stat proves it
did not move, and any doubt leaves the original byte-identical with no ``.tmp``
behind. Every failure path below is a NEGATIVE control for that promise, because
the asymmetry is total: a skipped session costs disk, a lost one costs somebody
their history.

The suite runs on the repository's 3.12 venv, so the gzip path is the one that
always executes; the zstd tests are ``importorskip``-ed and run in CI's 3.14 job
(and can be run locally against the production runtime — see the PR).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from local_operator.session import cleanup, compression, retention, search_index

DAY = 86400.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _transcript(turns: int = 40, *, word: str = "retention") -> bytes:
    """A small but realistic transcript: JSONL messages plus bookkeeping rows."""
    rows = [{"type": "meta", "payload": {"note": "not a message"}}]
    for index in range(turns):
        role = "user" if index % 2 == 0 else "assistant"
        rows.append(
            {
                "type": "message",
                "payload": {"role": role, "content": f"{word} turn {index} " + "x" * 40},
            }
        )
    return b"".join(json.dumps(row).encode() + b"\n" for row in rows)


def _store(tmp_path: Path, name: str = "store") -> Path:
    """A marked session store: ``<config>/sessions`` plus the marker file."""
    config = tmp_path / name
    sessions = config / "sessions"
    sessions.mkdir(parents=True)
    (sessions / cleanup.STORE_MARKER_NAME).write_text("{}", encoding="utf-8")
    return config


def _session(
    config: Path,
    sid: str,
    *,
    body: bytes | None = None,
    age_days: float = 40.0,
    now: float | None = None,
) -> Path:
    directory = config / "sessions" / sid
    directory.mkdir(parents=True)
    payload = _transcript() if body is None else body
    transcript = directory / compression.PLAIN_FILENAME
    transcript.write_bytes(payload)
    stamp = (now if now is not None else time.time()) - age_days * DAY
    os.utime(transcript, (stamp, stamp))
    os.utime(directory, (stamp, stamp))
    return directory


def _policy(**overrides) -> compression.CompressionPolicy:
    return compression.CompressionPolicy(**{"enabled": True, "after_days": 30, **overrides})


class _StubConfig:
    """The duck-typed config reader ``policy_from_config`` documents."""

    def __init__(self, values: dict[tuple[str, ...], object] | None = None) -> None:
        self.values = values or {}

    def get_nested_value(self, path: tuple[str, ...], default: object = None) -> object:
        return self.values.get(tuple(path), default)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


def test_round_trip_readers_cache_and_restore(tmp_path: Path) -> None:
    config = _store(tmp_path)
    directory = _session(config, "abc123")
    original = (directory / compression.PLAIN_FILENAME).read_bytes()
    digest = hashlib.sha256(original).hexdigest()

    outcome = compression.compress_session(directory)

    assert outcome.status == "compressed", outcome
    assert outcome.bytes_before == len(original)
    assert outcome.bytes_after < outcome.bytes_before
    assert not (directory / compression.PLAIN_FILENAME).exists()
    archive = compression.archive_path(directory)
    assert archive is not None
    assert archive.name.endswith(compression.ARCHIVE_SUFFIXES[outcome.codec])
    assert compression.has_transcript(directory)
    assert compression.transcript_source(directory) == (archive, outcome.codec)

    # The readers, against the archived form.
    assert compression.read_head(directory, 40) == original[:40]
    assert compression.read_tail(directory, 40) == original[-40:]
    # `iter_lines` drops the terminator (it yields TEXT lines); rebuilding the
    # file is the round-trip assertion.
    assert "".join(f"{line}\n" for line in compression.iter_lines(directory)).encode() == original

    # The materialised read path is a real file with the original bytes.
    read_path = compression.transcript_for_read(directory)
    assert read_path is not None
    assert _sha(read_path) == digest
    cache_entry = config / "cache" / "decompressed" / "abc123"
    assert read_path == cache_entry / compression.PLAIN_FILENAME
    meta = json.loads((cache_entry / ".meta.json").read_text())
    assert meta["sha256"] == digest
    assert meta["codec"] == outcome.codec

    # And a verified in-place restore, which is what a writer needs.
    assert compression.restore_for_write(directory) is True
    assert (directory / compression.PLAIN_FILENAME).read_bytes() == original
    assert compression.archive_path(directory) is None
    assert not cache_entry.exists() or not (cache_entry / compression.PLAIN_FILENAME).exists()


def test_restore_preserves_the_original_clock(tmp_path: Path) -> None:
    config = _store(tmp_path)
    directory = _session(config, "clock01", age_days=45)
    before = (directory / compression.PLAIN_FILENAME).stat().st_mtime_ns

    compression.compress_session(directory)
    archive = compression.archive_path(directory)
    assert archive is not None
    assert archive.stat().st_mtime_ns == before

    compression.restore_for_write(directory)
    assert (directory / compression.PLAIN_FILENAME).stat().st_mtime_ns == before


# ---------------------------------------------------------------------------
# Failure paths: the original survives, and nothing is left behind
# ---------------------------------------------------------------------------


def test_a_preexisting_tmp_looking_file_is_never_adopted(tmp_path: Path) -> None:
    config = _store(tmp_path)
    directory = _session(config, "tmpsquat")
    decoy = directory / ".transcript.jsonl.999.aaaaaaaa.tmp"
    decoy.write_bytes(b"do not touch me")

    outcome = compression.compress_session(directory)

    assert outcome.status == "compressed", outcome
    assert decoy.read_bytes() == b"do not touch me"
    assert compression.transcript_for_read(directory) is not None


def test_verification_mismatch_keeps_the_original_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _store(tmp_path)
    directory = _session(config, "verify01")
    original = (directory / compression.PLAIN_FILENAME).read_bytes()

    monkeypatch.setattr(compression, "_decoded_digest", lambda _archive, _codec: ("0" * 64, 1))

    outcome = compression.compress_session(directory)

    assert outcome.status == "failed"
    assert outcome.reason == "verification mismatch"
    assert (directory / compression.PLAIN_FILENAME).read_bytes() == original
    assert compression.archive_path(directory) is None
    assert [p.name for p in directory.iterdir() if p.name.endswith(".tmp")] == []


def test_a_decode_failure_during_verification_keeps_the_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _store(tmp_path)
    directory = _session(config, "verify02")
    original = (directory / compression.PLAIN_FILENAME).read_bytes()

    def explode(_archive: Path, _codec: str) -> tuple[str, int]:
        raise compression.ArchiveCorrupt("cannot decode")

    monkeypatch.setattr(compression, "_decoded_digest", explode)

    outcome = compression.compress_session(directory)

    assert outcome.status == "failed"
    assert outcome.reason == "verification failed"
    assert (directory / compression.PLAIN_FILENAME).read_bytes() == original
    assert compression.archive_path(directory) is None
    assert not list(directory.glob("*.tmp"))


def test_a_source_that_grows_after_publish_is_kept_and_the_archive_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A writer appended between the publish and the confirming re-stat.

    The archive is not the current content any more, so it must not stay as a
    stale second form: the pass removes it and keeps the source. This is the one
    rollback in the protocol that touches a VERIFIED archive on purpose.
    """
    config = _store(tmp_path)
    directory = _session(config, "grown01")
    source = directory / compression.PLAIN_FILENAME
    real_publish = compression._publish

    def publish_then_append(tmp: Path, destination: Path) -> None:
        real_publish(tmp, destination)
        with source.open("ab") as handle:
            handle.write(b'{"type": "message", "payload": {"role": "user", "content": "late"}}\n')
            handle.flush()
            os.fsync(handle.fileno())

    monkeypatch.setattr(compression, "_publish", publish_then_append)

    outcome = compression.compress_session(directory)

    assert outcome.status == "skipped"
    assert "changed" in outcome.reason
    assert source.exists()
    assert source.read_bytes().endswith(b'"late"}}\n')
    assert compression.archive_path(directory) is None
    assert not list(directory.glob("*.gz")) and not list(directory.glob("*.zst"))


def test_a_source_that_changes_before_publish_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _store(tmp_path)
    directory = _session(config, "growth2")
    source = directory / compression.PLAIN_FILENAME
    real_fsync = compression._fsync_file

    def fsync_then_touch(path: Path) -> None:
        real_fsync(path)
        os.utime(source, (time.time(), time.time()))

    monkeypatch.setattr(compression, "_fsync_file", fsync_then_touch)

    outcome = compression.compress_session(directory)

    assert outcome.status == "skipped"
    assert "changed" in outcome.reason
    assert source.exists()
    assert compression.archive_path(directory) is None
    assert not list(directory.glob("*.tmp"))


# ---------------------------------------------------------------------------
# Guards: each one skips with its own reason and touches nothing
# ---------------------------------------------------------------------------


def _arm_wake(config: Path, sid: str, directory: Path) -> None:
    from local_operator.wakes.store import entry_path

    path = entry_path(config, sid)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"id": sid, "kind": "once"}), encoding="utf-8")


def _arm_monitor(config: Path, sid: str, directory: Path) -> None:
    from local_operator.monitors.store import entry_path

    path = entry_path(config, sid)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"monitors": [{"pattern": "boom"}]}), encoding="utf-8")


def _claim(config: Path, sid: str, directory: Path) -> None:
    retention.claim_session(directory, os.getpid())


def _hold(config: Path, sid: str, directory: Path) -> None:
    assert cleanup.set_hold(config / "sessions", sid, reason="evidence", by="tests")


def _open_project(config: Path, sid: str, directory: Path) -> None:
    projects = config / "projects"
    projects.mkdir(parents=True, exist_ok=True)
    (projects / "proj1.json").write_text(
        json.dumps({"id": "proj1", "status": "active", "sessions": [sid]}), encoding="utf-8"
    )


def _active_parent(config: Path, sid: str, directory: Path) -> None:
    parent = config / "sessions" / "parent1"
    parent.mkdir(parents=True, exist_ok=True)
    (parent / compression.PLAIN_FILENAME).write_bytes(_transcript(turns=2))
    (parent / "subagent-roster.v1.json").write_text(
        json.dumps({"records": [{"session_dir": str(config / "sessions" / sid)}]}),
        encoding="utf-8",
    )


def _scratchpad_hazard(config: Path, sid: str, directory: Path) -> None:
    repo = directory / "scratchpad" / "repo"
    repo.mkdir(parents=True)
    env = {
        **os.environ,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",
    }
    subprocess.run(["git", "init", "-q", str(repo)], check=True, env=env)
    (repo / "uncommitted.txt").write_text("work in progress", encoding="utf-8")


def _lock(config: Path, sid: str, directory: Path) -> None:
    assert compression.acquire_writer(directory)


CASES = {
    "live_dir": (_active_parent, "the current session"),
    "live_claim": (_claim, "claimed by a live process"),
    "armed_wake": (_arm_wake, "has an armed wake"),
    "armed_monitor": (_arm_monitor, "has an armed monitor"),
    "operator_hold": (_hold, "held by the operator"),
    "open_project": (_open_project, "linked to an open project"),
    "active_parent": (_active_parent, "parent session still active"),
    "scratchpad_git": (_scratchpad_hazard, "uncommitted/unpushed work"),
    "lock_busy": (_lock, "session in use"),
}


@pytest.mark.parametrize("case", sorted(CASES), ids=sorted(CASES))
def test_every_guard_skips_with_its_own_reason(
    case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup, expected = CASES[case]
    config = _store(tmp_path, name=case)
    directory = _session(config, "guarded1")
    original = (directory / compression.PLAIN_FILENAME).read_bytes()
    kwargs: dict[str, object] = {}
    setup(config, "guarded1", directory)
    if case == "live_dir":
        # The session currently being BUILT is kept by identity, not by a probe
        # on disk, so this case only supplies the path.
        kwargs["live_dir"] = directory
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    monkeypatch.setenv("GIT_TERMINAL_PROMPT", "0")
    try:
        result = compression.run_compression(config, _policy(), **kwargs)  # type: ignore[arg-type]
    finally:
        if case == "lock_busy":
            compression.release_writer(directory)

    assert [row.status for row in result.compressed] == []
    reasons = {row.reason for row in result.skipped_rows}
    assert any(expected in reason for reason in reasons), (case, reasons)
    assert (directory / compression.PLAIN_FILENAME).read_bytes() == original
    assert compression.archive_path(directory) is None


def test_the_pass_fails_closed_when_a_store_wide_guard_cannot_be_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _store(tmp_path)
    directory = _session(config, "unknow01")
    original = (directory / compression.PLAIN_FILENAME).read_bytes()

    def unreadable(_config_dir: Path, _now: float, **_kwargs: object) -> set[str]:
        from local_operator.session.delegated_retention import _Unknowable

        raise _Unknowable("projects/: permission denied")

    monkeypatch.setattr(
        "local_operator.session.delegated_retention.active_parent_sessions", unreadable
    )

    result = compression.run_compression(config, _policy())

    assert result.compressed == []
    assert result.skipped is not None and "cannot read" in result.skipped
    assert result.errors == 1
    assert (directory / compression.PLAIN_FILENAME).read_bytes() == original
    assert compression.archive_path(directory) is None


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def test_dry_run_lists_candidates_and_skip_reasons_and_changes_nothing(tmp_path: Path) -> None:
    config = _store(tmp_path)
    eligible = _session(config, "eligible")
    guarded = _session(config, "guarded2")
    _arm_wake(config, "guarded2", guarded)
    (eligible / compression.PLAIN_FILENAME).read_bytes()

    before = {
        path.name: (
            path.stat().st_size,
            path.stat().st_mtime_ns,
            path.read_bytes() if path.is_file() else b"",
        )
        for path in sorted(eligible.iterdir())
    }
    before_guarded = {
        path.name: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(guarded.iterdir())
    }

    result = compression.run_compression(config, _policy(), dry_run=True)

    assert result.dry_run
    assert [row.session for row in result.candidates] == ["eligible"]
    assert result.compressed == []
    assert any("armed wake" in reason for reason in result.skipped_by_reason)
    after = {
        path.name: (
            path.stat().st_size,
            path.stat().st_mtime_ns,
            path.read_bytes() if path.is_file() else b"",
        )
        for path in sorted(eligible.iterdir())
    }
    assert after == before
    assert {
        path.name: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(guarded.iterdir())
    } == before_guarded
    # A rehearsal creates no lock file, no cache and no log.
    assert (eligible / compression.LOCK_NAME).exists() is False
    assert (config / "cache").exists() is False
    assert not (config / "sessions" / compression.COMPRESSION_LOG_NAME).exists()


# ---------------------------------------------------------------------------
# Codecs
# ---------------------------------------------------------------------------


def test_the_suffix_drives_the_reader(tmp_path: Path) -> None:
    config = _store(tmp_path)
    directory = _session(config, "suffix01")
    body = _transcript(turns=5)

    outcome = compression.compress_session(directory, codec="gzip")
    assert outcome.codec == "gzip"
    gzip_archive = compression.archive_path(directory)
    assert gzip_archive is not None and gzip_archive.name == "transcript.jsonl.gz"
    assert compression.read_head(directory, 10) == body[:10]

    # The plain file wins whenever both exist (an interrupted restore), and an
    # unknown suffix is not an archive this build claims to read.
    (directory / compression.PLAIN_FILENAME).write_bytes(b"plain wins")
    assert compression.transcript_source(directory)[1] is None
    assert compression.archive_path(directory) is not None
    (directory / compression.PLAIN_FILENAME).unlink()
    assert compression.transcript_source(directory)[1] == "gzip"

    # An archive suffix this build does not know is NOT an archive it reads: the
    # session reads as having no transcript rather than as garbage.
    lonely = _session(config, "suffix02")
    (lonely / compression.PLAIN_FILENAME).unlink()
    (lonely / "transcript.jsonl.xz").write_bytes(b"whatever")
    assert compression.archive_path(lonely) is None
    assert compression.has_transcript(lonely) is False


def test_an_unknown_or_unavailable_codec_is_reported_never_absent(tmp_path: Path) -> None:
    config = _store(tmp_path)
    directory = _session(config, "nothere")
    source = directory / compression.PLAIN_FILENAME
    source.unlink()
    # A .zst archive and a build that cannot decode it is the cross-version case
    # the module documents; on 3.14 the same bytes are simply a corrupt archive.
    (directory / "transcript.jsonl.zst").write_bytes(b"\x28\xb5\x2f\xfd not really zstd")
    if compression.available_codec() == "gzip":
        with pytest.raises(compression.ArchiveUnavailable):
            compression.read_head(directory, 64)
    else:
        with pytest.raises(compression.ArchiveCorrupt):
            compression.read_head(directory, 64)
    # Either way the session is NOT absent: listing and search still find it.
    assert compression.has_transcript(directory) is True
    assert compression.transcript_source(directory)[1] == "zstd"


def test_a_corrupt_archive_never_reads_as_an_empty_transcript(tmp_path: Path) -> None:
    config = _store(tmp_path)
    directory = _session(config, "corrupt1")
    compression.compress_session(directory, codec="gzip")
    archive = compression.archive_path(directory)
    assert archive is not None
    archive.write_bytes(archive.read_bytes()[:20])

    with pytest.raises(compression.ArchiveCorrupt):
        compression.read_head(directory, 64)
    with pytest.raises(compression.ArchiveCorrupt):
        compression.read_tail(directory, 64)
    with pytest.raises(compression.ArchiveCorrupt):
        list(compression.iter_lines(directory))
    assert compression.has_transcript(directory) is True
    # A corrupt archive is not restored over the truth: nothing is written.
    assert compression.restore_for_write(directory) is False
    assert compression.archive_path(directory) is not None
    assert not (directory / compression.PLAIN_FILENAME).exists()


def test_zstd_round_trip_when_this_interpreter_has_it(tmp_path: Path) -> None:
    pytest.importorskip("compression.zstd")
    config = _store(tmp_path)
    directory = _session(config, "zstd0001")
    body = (directory / compression.PLAIN_FILENAME).read_bytes()

    outcome = compression.compress_session(directory, codec="zstd")

    assert outcome.codec == "zstd"
    zstd_archive = compression.archive_path(directory)
    assert zstd_archive is not None and zstd_archive.name == "transcript.jsonl.zst"
    assert compression.read_head(directory, 100) == body[:100]
    assert compression.read_tail(directory, 100) == body[-100:]
    materialised = compression.transcript_for_read(directory)
    assert materialised is not None and materialised.read_bytes() == body


def test_gzip_and_zstd_archives_coexist_and_each_reads_by_suffix(tmp_path: Path) -> None:
    zstd = pytest.importorskip("compression.zstd")
    config = _store(tmp_path)
    first = _session(config, "mixed001")
    body = (first / compression.PLAIN_FILENAME).read_bytes()
    compression.compress_session(first, codec="gzip")
    assert compression.read_head(first, 32) == body[:32]

    # A .zst written alongside (simulating another runtime's archive) is read by
    # its suffix, not by whatever this build would write today.
    second = _session(config, "mixed002")
    second_body = (second / compression.PLAIN_FILENAME).read_bytes()
    (second / "transcript.jsonl.zst").write_bytes(zstd.compress(second_body))
    (second / compression.PLAIN_FILENAME).unlink()
    assert compression.read_head(second, 32) == second_body[:32]
    assert compression.transcript_source(second)[1] == "zstd"


# ---------------------------------------------------------------------------
# Listing: the clock must not move
# ---------------------------------------------------------------------------


def test_compression_does_not_move_the_activity_clock(tmp_path: Path) -> None:
    config = _store(tmp_path)
    fresh = _session(config, "recent01", age_days=1)
    _session(config, "old00001", age_days=60)
    ids = ("recent01", "old00001")
    before = {sid: retention.session_activity(config / "sessions" / sid) for sid in ids}
    assert before["old00001"] is not None

    result = compression.run_compression(config, _policy())

    assert [row.session for row in result.compressed] == ["old00001"]
    after = {sid: retention.session_activity(config / "sessions" / sid) for sid in ids}
    assert after == before
    assert (fresh / compression.PLAIN_FILENAME).exists()


def test_the_activity_file_set_names_the_archives() -> None:
    """``retention`` spells the archive names as literals; this pins them equal.

    The two must never drift: a missing name reads as "no clock" for every
    compressed session and silently reorders the picker and the cleanup policy's
    recent-N guard.
    """
    assert set(compression.ARCHIVE_FILENAMES) == {
        compression.PLAIN_FILENAME + suffix for suffix in compression.ARCHIVE_SUFFIXES.values()
    }
    assert retention.TRANSCRIPT_FILENAME == compression.PLAIN_FILENAME
    assert set(compression.ARCHIVE_FILENAMES) <= set(retention._ACTIVITY_FILES)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def test_the_cache_is_reused_and_revalidated_against_the_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _store(tmp_path)
    directory = _session(config, "cache001")
    compression.compress_session(directory)
    entry = config / "cache" / "decompressed" / "cache001"

    first = compression.transcript_for_read(directory)
    assert first is not None and first.exists()
    writes = (entry / compression.PLAIN_FILENAME).stat().st_mtime_ns
    second = compression.transcript_for_read(directory)
    assert second == first
    assert (entry / compression.PLAIN_FILENAME).stat().st_mtime_ns == writes

    # A different archive invalidates the cache: the meta records the archive's
    # stat, and a mismatch must decode again rather than serve stale bytes.
    archive = compression.archive_path(directory)
    assert archive is not None
    archive.write_bytes(archive.read_bytes() + b"")  # same bytes, new mtime below
    os.utime(archive, (time.time() + 5, time.time() + 5))
    third = compression.transcript_for_read(directory)
    assert third is not None and third.exists()


def test_the_cache_evicts_down_to_its_bounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _store(tmp_path)
    monkeypatch.setattr(compression, "CACHE_MAX_ENTRIES", 1)
    first = _session(config, "cacheA01")
    second = _session(config, "cacheB01")
    for directory in (first, second):
        compression.compress_session(directory)
        assert compression.transcript_for_read(directory) is not None

    root = config / "cache" / "decompressed"
    present = sorted(
        child.name for child in root.iterdir() if (child / compression.PLAIN_FILENAME).exists()
    )
    assert present == ["cacheB01"]
    # Eviction removes FILES only: this module may not remove a directory (the
    # deletion guard reserves rmdir for the cleanup remover), so the empty entry
    # directory is expected to stay.
    assert (root / "cacheA01").is_dir()


def test_a_session_outside_a_sessions_directory_uses_the_temp_cache_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(compression.tempfile, "gettempdir", lambda: str(tmp_path / "tmp"))
    loose = tmp_path / "loose" / "sess-1"
    loose.mkdir(parents=True)
    (loose / compression.PLAIN_FILENAME).write_bytes(b"x")
    root = compression._cache_root(loose)
    assert root == tmp_path / "tmp" / "local-operator-decompressed" / "cache" / "decompressed"


# ---------------------------------------------------------------------------
# Policy, settings and the pass entry point
# ---------------------------------------------------------------------------


def test_policy_defaults_and_nested_reads(tmp_path: Path) -> None:
    assert compression.DEFAULT_ENABLED is True
    assert compression.DEFAULT_AFTER_DAYS == 30
    assert compression.policy_from_config(_StubConfig()) == _policy()

    manager = compression.policy_from_config(
        _StubConfig(
            {
                ("session", "compression", "enabled"): False,
                ("session", "compression", "after_days"): 7,
            }
        )
    )
    assert manager == compression.CompressionPolicy(enabled=False, after_days=7)

    # A manager that cannot be read at all does not guess: the documented default
    # is ON, so nothing may run from a stub that cannot answer the switch.
    class Broken:
        def get_nested_value(self, path, default=None):  # noqa: ANN001, ANN201
            raise RuntimeError("no config")

    assert compression.policy_from_config(Broken()) == compression.CompressionPolicy()


def test_the_registry_defaults_match_the_consumers() -> None:
    from local_operator import settings_io

    enabled = settings_io.BY_KEY["session.compression.enabled"]
    after = settings_io.BY_KEY["session.compression.after_days"]
    assert enabled.kind is settings_io.Kind.BOOL
    assert enabled.default is compression.DEFAULT_ENABLED
    assert after.default == compression.DEFAULT_AFTER_DAYS
    assert after.minimum == compression.MIN_AFTER_DAYS
    assert after.maximum == compression.MAX_AFTER_DAYS
    assert after.unit == "days"
    assert after.gated_by == "session.compression.enabled"
    section = next(s for s in settings_io.SECTIONS if s.name == "session_compression")
    assert section.scope is settings_io.Scope.NEW_LAUNCH
    assert compression.COMPRESSION_PATH == ("session", "compression")


def test_switching_off_stops_new_compressions_but_never_hides_old_ones(tmp_path: Path) -> None:
    config = _store(tmp_path)
    first = _session(config, "before01")
    body = (first / compression.PLAIN_FILENAME).read_bytes()
    assert compression.compress_from_config(_StubConfig(), config).compressed

    second = _session(config, "after001")
    off = _StubConfig({("session", "compression", "enabled"): False})
    result = compression.compress_from_config(off, config)

    assert result.skipped == "disabled"
    assert result.compressed == []
    assert (second / compression.PLAIN_FILENAME).exists()
    # The switch is a writer policy only: reads never consult it.
    assert compression.has_transcript(first)
    assert compression.read_head(first, 16) == body[:16]
    assert compression.transcript_for_read(first) is not None


def test_the_pass_never_raises_into_the_maintenance_sequence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _store(tmp_path)
    _session(config, "boom0001")

    def explode(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("unexpected")

    monkeypatch.setattr(compression, "run_compression", explode)

    result = compression.compress_from_config(_StubConfig(), config)

    assert result.errors == 1
    assert result.skipped is not None and "pass failed" in result.skipped


def test_the_log_and_summary_record_only_real_work(tmp_path: Path) -> None:
    config = _store(tmp_path)
    sessions = config / "sessions"
    recent = _session(config, "recent02", age_days=1)

    quiet = compression.compress_from_config(_StubConfig(), config)
    assert quiet.compressed == []
    assert not (sessions / compression.COMPRESSION_LOG_NAME).exists()
    assert not (sessions / compression.LAST_COMPRESSION_NAME).exists()

    _session(config, "old00002", age_days=90)
    worked = compression.compress_from_config(_StubConfig(), config)
    assert [row.session for row in worked.compressed] == ["old00002"]
    assert worked.bytes_after < worked.bytes_before

    rows = [
        json.loads(line)
        for line in (sessions / compression.COMPRESSION_LOG_NAME).read_text().splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    assert rows[0]["session"] == "old00002"
    assert rows[0]["actor"] == "startup"
    assert rows[0]["codec"] == worked.compressed[0].codec

    summary = json.loads((sessions / compression.LAST_COMPRESSION_NAME).read_text())
    assert summary["compressed"] == 1
    assert summary["bytes_before"] == worked.bytes_before
    assert (recent / compression.PLAIN_FILENAME).exists()


def test_the_budget_stops_cleanly_between_sessions(tmp_path: Path) -> None:
    config = _store(tmp_path)
    for index in range(3):
        _session(config, f"budget{index:02d}", age_days=90)

    calls = {"n": 0}

    def stop_after_first() -> bool:
        calls["n"] += 1
        return calls["n"] > 1

    result = compression.run_compression(config, _policy(), should_stop=stop_after_first)

    assert result.stopped is True
    assert len(result.compressed) == 1
    assert result.scanned == 3


# ---------------------------------------------------------------------------
# The search index must survive compression WITHOUT a re-digest
# ---------------------------------------------------------------------------


def test_the_index_entry_is_rewritten_in_place_and_never_redigested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _store(tmp_path)
    directory = _session(config, "search01", body=_transcript(turns=6, word="kangaroo"))
    index_file = search_index.index_path(config)
    digests = search_index.build_index(config, ["search01"])
    original_digest = digests["search01"]
    assert "kangaroo" in original_digest

    calls = {"n": 0}
    real = search_index.digest_transcript_read

    def counting(path: Path) -> tuple[str, bool]:
        calls["n"] += 1
        return real(path)

    monkeypatch.setattr(search_index, "digest_transcript_read", counting)

    outcome = compression.compress_session(directory)
    assert outcome.status == "compressed"
    assert calls["n"] == 0, "compression itself must not re-digest"

    archive = compression.archive_path(directory)
    assert archive is not None
    entry = json.loads(index_file.read_text())["entries"]["search01"]
    assert entry["codec"] == outcome.codec
    assert entry["signature"][0] == archive.stat().st_size
    assert entry["signature"][1] == archive.stat().st_mtime
    assert entry["digest"] == original_digest

    # A picker open over the compressed store reuses the entry: no decompression,
    # no re-digest, same digest string.
    again = search_index.build_index(config, ["search01"])
    assert calls["n"] == 0
    assert again["search01"] == original_digest
    assert search_index.search_digests(again, "kangaroo") == {"search01"}


def test_an_index_without_an_entry_for_the_session_is_left_alone(tmp_path: Path) -> None:
    config = _store(tmp_path)
    directory = _session(config, "search02")
    assert (
        search_index.record_archive(
            config, "search02", source=directory / compression.PLAIN_FILENAME, codec="gzip"
        )
        is False
    )
    assert search_index.index_path(config).exists() is False
