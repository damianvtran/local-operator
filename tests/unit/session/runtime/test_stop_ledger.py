"""``stop_ledger``: append, rotation, the target cap, swallowed failures, modes."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any

import pytest

from local_operator.session.runtime import stop_ledger


def _targets(n: int) -> list[dict[str, Any]]:
    return [
        {"session_id": f"s{i}", "pid": 100 + i, "rec_kind": "exec", "name": f"n{i}"}
        for i in range(n)
    ]


def test_begin_then_end_are_two_rows_sharing_one_id(tmp_path: Path) -> None:
    sweep_id = stop_ledger.begin_sweep(
        mechanism="stop-all", command="lop stop --all", targets=_targets(3), root=tmp_path
    )
    stop_ledger.end_sweep(sweep_id, [{"pid": 100, "method": "socket"}], root=tmp_path)
    rows = stop_ledger.read_sweeps(tmp_path)
    assert [r["phase"] for r in rows] == ["begin", "end"]
    assert {r["sweep_id"] for r in rows} == {sweep_id} and len(sweep_id) == 12
    begin = rows[0]
    assert begin["counts"] == {"targets": 3, "exec": 3}
    assert begin["command"] == "lop stop --all" and begin["pid"] == os.getpid()
    path = stop_ledger.sweeps_path(tmp_path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_targets_are_capped_and_flagged(tmp_path: Path) -> None:
    stop_ledger.begin_sweep(
        mechanism="stop-all",
        command="c",
        targets=_targets(stop_ledger.MAX_TARGETS + 5),
        root=tmp_path,
    )
    begin = stop_ledger.read_sweeps(tmp_path)[0]
    assert len(begin["targets"]) == stop_ledger.MAX_TARGETS and begin["truncated"] is True
    assert begin["counts"]["targets"] == stop_ledger.MAX_TARGETS + 5


def test_the_file_rotates_once_past_the_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(stop_ledger, "ROTATE_BYTES", 400)
    for _ in range(6):
        stop_ledger.begin_sweep(mechanism="m", command="c", targets=_targets(2), root=tmp_path)
    path = stop_ledger.sweeps_path(tmp_path)
    assert path.with_name(path.name + ".1").exists()
    assert path.stat().st_size <= 400 + 1500
    assert len(stop_ledger.read_sweeps(tmp_path)) >= 2


def test_a_failing_write_is_swallowed_and_the_id_is_still_returned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_a: Any, **_k: Any) -> int:
        raise OSError("no")

    monkeypatch.setattr(stop_ledger.os, "open", boom)
    sweep_id = stop_ledger.begin_sweep(mechanism="m", command="c", targets=[], root=tmp_path)
    assert sweep_id
    stop_ledger.end_sweep(sweep_id, [], root=tmp_path)  # must not raise


def test_a_torn_line_is_skipped_by_the_reader(tmp_path: Path) -> None:
    stop_ledger.begin_sweep(mechanism="m", command="c", targets=[], root=tmp_path)
    with open(stop_ledger.sweeps_path(tmp_path), "a") as handle:
        handle.write('{"torn": ')
    assert len(stop_ledger.read_sweeps(tmp_path)) == 1
