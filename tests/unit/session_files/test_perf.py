"""A 15,000-record journal: the cost model holds, asserted structurally.

AGENTS.md ("Timing") rules out wall-clock bounds as the assertion, so the invariants here are
COUNTS: how many rows reach ``json.loads``, how many bytes a warm hit reads, how many files a
default page ``stat``s, how big the cache is. A single loose CPU ceiling (``thread_time``,
~30x what this machine needs) is a backstop against an accidental quadratic, not the claim.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from local_operator.session_files import ledger
from local_operator.session_files import scan as scan_module
from local_operator.session_files import service

from ._fixtures import SESSION, assistant, call, line, tool

RECORDS = 15_000
CWD, HOME = "/work/proj", "/home/u"


def _journal(path: Path, records: int) -> int:
    """``records`` file-producing write pairs, each followed by ~3 prose/state rows, plus a
    few 400 KB result rows (rows reach ~900 KB in real stores: the prefilter must skip them)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fat = "x" * 400_000
    file_rows = 0
    with path.open("wb") as handle:
        for i in range(records):
            p = f"{CWD}/out/f{i}.md"
            handle.write(line(assistant(f"a{i}", float(i), [call(f"c{i}", "write", path=p)])))
            handle.write(
                line(
                    tool(
                        f"t{i}",
                        i + 0.5,
                        f"c{i}",
                        "write",
                        "ok",
                        {"path": p, "added": 1, "removed": 0},
                    )
                )
            )
            file_rows += 2
            for j in range(3):
                handle.write(
                    line(
                        {
                            "id": f"p{i}_{j}",
                            "ts": float(i),
                            "type": "message",
                            "payload": {
                                "kind": "message",
                                "role": "assistant",
                                "content": [{"text": "prose " * 20}],
                            },
                        }
                    )
                )
            if i % 5000 == 0:
                handle.write(
                    line(
                        tool(
                            f"fat{i}", float(i), f"r{i}", "read", fat, {"path": f"{CWD}/big{i}.txt"}
                        )
                    )
                )
    return file_rows


@pytest.fixture
def big(tmp_path: Path) -> tuple[Path, Path, int]:
    config = tmp_path / "cfg"
    session_dir = config / "sessions" / SESSION
    file_rows = _journal(session_dir / "transcript.jsonl", RECORDS)
    return config, session_dir, file_rows


def test_cold_scan_parses_only_file_producing_rows(big, monkeypatch: pytest.MonkeyPatch) -> None:
    config, session_dir, file_rows = big
    parses: list[int] = []
    real = scan_module._row
    monkeypatch.setattr(scan_module, "_row", lambda raw: parses.append(1) or real(raw))
    started = time.thread_time()
    snapshot = ledger.refresh(config, SESSION, session_dir, cwd=CWD, home=HOME)
    cpu = time.thread_time() - started
    assert snapshot is not None and snapshot.complete
    assert len(snapshot.scan.records) == RECORDS
    # Every assistant call row + every tool result row, and NOTHING else (prose, fat reads).
    assert len(parses) == file_rows
    assert snapshot.scan.rows > 4 * RECORDS
    assert cpu < 30.0, f"cold scan burned {cpu:.1f}s CPU: look for a quadratic"


def test_warm_paths_read_nothing_and_cache_is_bounded(big, monkeypatch: pytest.MonkeyPatch) -> None:
    config, session_dir, _ = big
    ledger.refresh(config, SESSION, session_dir, cwd=CWD, home=HOME)
    cache = ledger.cache_path(config, SESSION)
    # Measured 168 B/record here (19 positional fields + a ~25-byte path); the bound is the point
    # that a record never carries a row's TEXT (a 400 KB result row would blow it by 1000x).
    assert cache.stat().st_size < 250 * RECORDS, cache.stat().st_size
    reads: list[int] = []
    real = ledger._read_into
    monkeypatch.setattr(ledger, "_read_into", lambda *a, **k: reads.append(1) or real(*a, **k))
    ledger.forget_resident()  # a restart: only the cache file
    again = ledger.refresh(config, SESSION, session_dir, cwd=CWD, home=HOME)
    assert again is not None and reads == [] and len(again.scan.records) == RECORDS


def test_default_page_stats_only_the_page(big, monkeypatch: pytest.MonkeyPatch) -> None:
    config, session_dir, _ = big
    stats: list[str] = []
    real = service.stat_path
    monkeypatch.setattr(
        service, "stat_path", lambda path, now=None: stats.append(path) or real(path, now)
    )
    family = service.Family(SESSION, session_dir)
    answer = service.list_files(config, family, service.Query(limit=200), budget_s=None, home=HOME)
    assert answer["total"] == RECORDS and len(answer["rows"]) == 200 and answer["next_cursor"]
    assert len(stats) == 200, "recent/name/kind sorts never stat the candidate set"
    stats.clear()
    service.forget_stats()
    # This is a structural count assertion, not a native filesystem speed requirement.
    # Freeze only the service's clock: CI contention or a slow mount may legitimately
    # exhaust the 0.75 s production budget. The separate expiry test covers that branch.
    monkeypatch.setattr(
        service, "time", SimpleNamespace(monotonic=lambda: 0.0, time=time.time, sleep=time.sleep)
    )
    answer = service.list_files(
        config, family, service.Query(sort="modified", limit=200), budget_s=None, home=HOME
    )
    assert answer["stat_complete"] is True and len(stats) >= 200
    assert len(stats) <= RECORDS, "a stat-derived sort is bounded by the candidate set, once each"


def test_incremental_append_reads_only_the_appended_bytes(big) -> None:
    config, session_dir, _ = big
    first = ledger.refresh(config, SESSION, session_dir, cwd=CWD, home=HOME)
    assert first is not None
    with (session_dir / "transcript.jsonl").open("ab") as handle:
        handle.write(line(assistant("z", 9e6, [call("cz", "write", path=f"{CWD}/new.md")])))
        handle.write(line(tool("zt", 9e6 + 1, "cz", "write", "ok", {"path": f"{CWD}/new.md"})))
    second = ledger.refresh(config, SESSION, session_dir, cwd=CWD, home=HOME)
    assert second is not None and len(second.scan.records) == RECORDS + 1
    assert second.sig["offset"] - first.sig["offset"] < 2_000, "two small rows, not a rescan"


def test_stat_budget_expiry_reports_partial_sort_and_still_stats_the_page(big, monkeypatch):
    """Deadline exhaustion is a supported result, independently of native I/O/host load.

    Advance a module-local clock per real stat: precisely four candidates fit the default
    0.75 s budget. A false stat_complete is honest, not a test failure; page rows must still
    receive their stats and no further candidate-wide work may be hidden behind fallback.
    """
    config, session_dir, _ = big
    clock = [0.0]
    stats: list[str] = []
    sort_stats: list[int] = []
    real_stat, real_sort = service.stat_path, service._sort
    monkeypatch.setattr(
        service,
        "time",
        SimpleNamespace(monotonic=lambda: clock[0], time=time.time, sleep=time.sleep),
    )

    def controlled_stat(path, now=None):
        stats.append(path)
        clock[0] += 0.25
        return real_stat(path, now)

    def counted_sort(entries, query):
        result = real_sort(entries, query)
        sort_stats.append(len(stats))
        return result

    monkeypatch.setattr(service, "stat_path", controlled_stat)
    monkeypatch.setattr(service, "_sort", counted_sort)
    answer = service.list_files(
        config,
        service.Family(SESSION, session_dir),
        service.Query(sort="modified", limit=200),
        budget_s=None,
        home=HOME,
    )
    assert answer["stat_complete"] is False
    assert sort_stats == [4], "deadline must stop candidate stat work at the controlled boundary"
    assert 200 <= len(stats) <= 204, "only the four candidates plus page rows are statted"
    assert answer["total"] == RECORDS and len(answer["rows"]) == 200 and answer["next_cursor"]
    assert all(
        row["exists"] is False for row in answer["rows"]
    ), "page stats still refresh missing files"
