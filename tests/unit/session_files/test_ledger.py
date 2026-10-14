"""The resumable cache: the signature ladder, partial scans, and everything that must rescan.

The ladder's three rungs (exact match / grown-and-verified / anything else) each get a
positive and a NEGATIVE control. The dangerous one is a rewrite that keeps the size and the
tail ROW'S ID but changes its content: an id-only probe (``code_requests``' rule) would resume
onto stale rows; the content-hash tail must refuse it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from local_operator.session_files import ledger

from ._fixtures import (
    SESSION,
    append_journal,
    assistant,
    call,
    generate_image_rows,
    line,
    tool,
    write_journal,
)

D1, D2, D3 = "1" * 32, "2" * 32, "3" * 32
CWD, HOME = "/work/proj", "/home/u"


def _refresh(config: Path, session_dir: Path, **kwargs: Any) -> ledger.Snapshot:
    snapshot = ledger.refresh(config, SESSION, session_dir, cwd=CWD, home=HOME, **kwargs)
    assert snapshot is not None
    return snapshot


def _setup(tmp_path: Path) -> tuple[Path, Path]:
    config = tmp_path / "cfg"
    session_dir = config / "sessions" / SESSION
    write_journal(session_dir, generate_image_rows(1, 1.0, D1, with_artifacts=True))
    return config, session_dir


def test_no_journal_is_none(tmp_path: Path) -> None:
    assert ledger.refresh(tmp_path, SESSION, tmp_path / "nope", cwd=CWD, home=HOME) is None


def test_cold_scan_then_exact_hit_reads_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, session_dir = _setup(tmp_path)
    first = _refresh(config, session_dir)
    assert first.complete and list(first.scan.records) == [f"a:{D1}"]
    assert ledger.cache_path(config, SESSION).is_file()

    reads: list[int] = []
    real = ledger._read_into
    monkeypatch.setattr(ledger, "_read_into", lambda *a, **k: reads.append(1) or real(*a, **k))
    again = _refresh(config, session_dir)
    assert reads == [] and again.revision == first.revision, "an exact signature must not read"


def test_cache_survives_a_process_restart(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, session_dir = _setup(tmp_path)
    first = _refresh(config, session_dir)
    ledger.forget_resident()  # a new process has no resident scans, only the file
    reads: list[int] = []
    real = ledger._read_into
    monkeypatch.setattr(ledger, "_read_into", lambda *a, **k: reads.append(1) or real(*a, **k))
    again = _refresh(config, session_dir)
    assert reads == [] and list(again.scan.records) == list(first.scan.records)


def test_append_is_incremental_from_the_saved_offset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, session_dir = _setup(tmp_path)
    first = _refresh(config, session_dir)
    appended = generate_image_rows(2, 10.0, D2, with_artifacts=True)
    append_journal(session_dir, appended)

    starts: list[int] = []
    real = ledger._read_into
    monkeypatch.setattr(
        ledger,
        "_read_into",
        lambda scan, path, offset, deadline: starts.append(offset)
        or real(scan, path, offset, deadline),
    )
    second = _refresh(config, session_dir)
    assert starts == [first.sig["offset"]], "only the appended region is read"
    assert set(second.scan.records) == {f"a:{D1}", f"a:{D2}"}
    assert second.revision > first.revision, "revision is monotone across a publish"


def test_torn_tail_is_not_consumed_and_completes_next_time(tmp_path: Path) -> None:
    config, session_dir = _setup(tmp_path)
    _refresh(config, session_dir)
    rows = generate_image_rows(2, 10.0, D2, with_artifacts=True)
    whole = b"".join(line(r) for r in rows)
    cut = len(whole) - 20
    with (session_dir / "transcript.jsonl").open("ab") as handle:
        handle.write(whole[:cut])  # the writer is mid-append: the last row has no newline
    torn = _refresh(config, session_dir)
    assert f"a:{D2}" not in torn.scan.records, "a row without its newline is left for the next pass"
    assert torn.sig["offset"] < torn.sig["size"]
    with (session_dir / "transcript.jsonl").open("ab") as handle:
        handle.write(whole[cut:])
    done = _refresh(config, session_dir)
    assert f"a:{D2}" in done.scan.records


def test_compaction_rewrite_new_inode_rescans(tmp_path: Path) -> None:
    config, session_dir = _setup(tmp_path)
    first = _refresh(config, session_dir)
    path = ledger.transcript_path(session_dir)
    # compact_file: write a sibling and os.replace it -> a NEW inode with different content.
    replacement = path.with_suffix(".new")
    replacement.write_bytes(
        b"".join(line(r) for r in generate_image_rows(5, 1.0, D3, with_artifacts=True))
    )
    os.replace(replacement, path)
    second = _refresh(config, session_dir)
    assert set(second.scan.records) == {f"a:{D3}"}, "the old arrangement must not survive a rewrite"
    assert second.sig["inode"] != first.sig["inode"]


def test_same_size_same_inode_in_place_rewrite_of_the_tail_rescans(tmp_path: Path) -> None:
    """The case an id-only tail probe cannot see: same row id, same length, different bytes."""
    config, session_dir = _setup(tmp_path)
    first = _refresh(config, session_dir)
    path = ledger.transcript_path(session_dir)
    data = path.read_bytes()
    assert D1.encode() in data
    path.write_bytes(data.replace(D1.encode(), D2.encode()))  # identical length, same ids/rows
    os.utime(path, (first.sig["mtime"] + 5, first.sig["mtime"] + 5))
    # Same inode (write_bytes truncates in place), same size, newer mtime.
    assert path.stat().st_size == first.sig["size"] and path.stat().st_ino == first.sig["inode"]
    # Grow it too, so rung 2 (resume) is what gets tried - and must refuse.
    append_journal(session_dir, generate_image_rows(3, 20.0, D3, with_artifacts=True))
    second = _refresh(config, session_dir)
    assert set(second.scan.records) == {f"a:{D2}", f"a:{D3}"}, second.scan.records.keys()


def test_truncation_rescans(tmp_path: Path) -> None:
    config, session_dir = _setup(tmp_path)
    append_journal(session_dir, generate_image_rows(2, 10.0, D2, with_artifacts=True))
    _refresh(config, session_dir)
    write_journal(session_dir, generate_image_rows(1, 1.0, D1, with_artifacts=True))  # shorter
    again = _refresh(config, session_dir)
    assert set(again.scan.records) == {f"a:{D1}"}


def test_budget_publishes_a_partial_scan_that_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "cfg"
    session_dir = config / "sessions" / SESSION
    rows: list[dict[str, Any]] = []
    for n in range(300):
        rows += generate_image_rows(n, float(n), f"{n:032x}", with_artifacts=True)
    write_journal(session_dir, rows)
    partial = _refresh(config, session_dir, budget_s=0.0)
    assert not partial.complete and 0 < partial.sig["offset"] < partial.sig["size"]
    assert 0 < len(partial.scan.records) < 300
    assert (
        ledger.peek(config, SESSION) is partial
    ), "a `building` answer serves the resident partial"

    # The partial survives a restart and resumes from its saved offset, not from byte 0.
    ledger.forget_resident()
    starts: list[int] = []
    real = ledger._read_into

    def spy(scan: Any, path: Path, offset: int, deadline: float | None) -> Any:
        starts.append(offset)
        return real(scan, path, offset, deadline)

    monkeypatch.setattr(ledger, "_read_into", spy)
    resumed: ledger.Snapshot | None = None
    while resumed is None or not resumed.complete:
        resumed = _refresh(config, session_dir, budget_s=0.0)
    assert starts[0] == partial.sig["offset"] and starts == sorted(starts)
    assert len(resumed.scan.records) == 300


def test_changed_cwd_rescans(tmp_path: Path) -> None:
    config, session_dir = _setup(tmp_path)
    first = _refresh(config, session_dir)
    other = ledger.refresh(config, SESSION, session_dir, cwd="/elsewhere", home=HOME)
    assert other is not None and other.cwd == "/elsewhere" and other is not first


@pytest.mark.parametrize(
    "damage",
    [
        b"",
        b"not json",
        b'{"schema": 99}',
        b'{"schema": 1, "scan_schema": 1, "sig": {}, "scan": {"rows": "x"}}',
    ],
)
def test_corrupt_or_foreign_cache_is_a_rescan_not_an_error(tmp_path: Path, damage: bytes) -> None:
    config, session_dir = _setup(tmp_path)
    _refresh(config, session_dir)
    ledger.cache_path(config, SESSION).write_bytes(damage)
    ledger.forget_resident()
    again = _refresh(config, session_dir)
    assert again.complete and list(again.scan.records) == [f"a:{D1}"]


def test_cache_path_refuses_a_traversal_id(tmp_path: Path) -> None:
    for bad in ("../x", "a/b", "", "x" * 200, "a b"):
        with pytest.raises(ValueError):
            ledger.cache_path(tmp_path, bad)


def test_cache_lives_outside_the_session_directory(tmp_path: Path) -> None:
    """Writing a sidecar into the session dir would move its mtime and reorder the session list."""
    config, session_dir = _setup(tmp_path)
    before = sorted(p.name for p in session_dir.iterdir())
    mtime = session_dir.stat().st_mtime_ns
    _refresh(config, session_dir)
    assert sorted(p.name for p in session_dir.iterdir()) == before
    assert session_dir.stat().st_mtime_ns == mtime
    assert ledger.cache_path(config, SESSION).is_relative_to(config / "cache" / "session_files")


def test_denied_path_is_never_cached(tmp_path: Path) -> None:
    config = tmp_path / "cfg"
    session_dir = config / "sessions" / SESSION
    write_journal(
        session_dir,
        [
            assistant("a", 1, [call("c1", "write", path="/work/proj/.env")]),
            tool("t1", 2, "c1", "write", "ok", {"path": "/work/proj/.env"}),
            assistant("a2", 3, [call("c2", "write", path="/work/proj/ok.md")]),
            tool("t2", 4, "c2", "write", "ok", {"path": "/work/proj/ok.md"}),
        ],
    )
    from local_operator.session_files import service

    snapshot = _refresh(config, session_dir, deny=service.make_deny(config))
    assert list(snapshot.scan.records) == ["p:/work/proj/ok.md"]
    assert ".env" not in ledger.cache_path(config, SESSION).read_text()


def test_append_between_eof_and_publication_is_resumed(tmp_path: Path, monkeypatch) -> None:
    config, directory = _setup(tmp_path)
    real = ledger._read_into
    injected = False

    def append_at_eof(*args):
        nonlocal injected
        result = real(*args)
        if not injected:
            injected = True
            append_journal(directory, generate_image_rows(2, 10.0, D2, with_artifacts=True))
        return result

    monkeypatch.setattr(ledger, "_read_into", append_at_eof)
    first = _refresh(config, directory)
    assert first.covered_bytes < first.total_bytes and not first.complete
    second = _refresh(config, directory)
    assert second.complete and set(second.scan.records) == {f"a:{D1}", f"a:{D2}"}


def test_replacement_between_eof_and_publication_is_not_certified(
    tmp_path: Path, monkeypatch
) -> None:
    config, directory = _setup(tmp_path)
    real = ledger._read_into
    injected = False

    def replace_at_eof(*args):
        nonlocal injected
        result = real(*args)
        if not injected:
            injected = True
            replacement = tmp_path / "replacement"
            replacement.write_bytes(b"".join(line(r) for r in generate_image_rows(2, 10.0, D2)))
            os.replace(replacement, directory / "transcript.jsonl")
        return result

    monkeypatch.setattr(ledger, "_read_into", replace_at_eof)
    first = _refresh(config, directory)
    assert not first.complete and not first.scan.records
    second = _refresh(config, directory)
    assert second.complete and set(second.scan.records) == {f"a:{D2}"}


def test_same_size_changed_prefix_unchanged_tail_rescans(tmp_path: Path) -> None:
    config, directory = _setup(tmp_path)
    append_journal(directory, [assistant("unchanged-tail", 100.0, [])])
    first = _refresh(config, directory)
    path = directory / "transcript.jsonl"
    raw = path.read_bytes()
    path.write_bytes(raw.replace(D1.encode(), D2.encode()))
    os.utime(path, (first.sig["mtime"] + 2, first.sig["mtime"] + 2))
    assert path.stat().st_ino == first.sig["inode"] and path.stat().st_size == first.sig["size"]
    second = _refresh(config, directory)
    assert set(second.scan.records) == {f"a:{D2}"}
