"""The off-process index build (POC, ``LOP_INDEX_SCAN_PROCESS``).

WHY THESE TESTS. The worker runs the SAME ``refresh_index`` in a spawned
process, so the one property that matters is that moving the build changes
WHERE it runs and nothing it produces: the published document must equal the
thread build's, byte for byte once the per-build identity (generation, inode,
mtime) is set aside, and the read-model companion must be on disk when the
task settles (the thread path defers it to an idle timer). A broken pool must
not wedge later refreshes. See docs/design/transcript-index-off-process.md.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest

from local_operator.session import transcript_index as ti

SID = "abcdef000001"


@pytest.fixture(autouse=True)
def _clean_module_state():
    ti._reset_for_tests()
    yield
    ti._reset_for_tests()


def _seed(root: Path, turns: int = 40) -> None:
    path = root / "sessions" / SID / "transcript.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for turn in range(turns):
        rows.append(
            {
                "id": f"u{turn:04d}",
                "ts": float(turn),
                "type": "message",
                "payload": {"kind": "message", "role": "user", "content": [{"text": f"q{turn}"}]},
            }
        )
        rows.append(
            {
                "id": f"a{turn:04d}",
                "ts": turn + 0.5,
                "type": "message",
                "payload": {
                    "kind": "message",
                    "role": "assistant",
                    "content": [{"text": f"answer {turn} " + "x" * 400}],
                    "stop_reason": "stop",
                },
            }
        )
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def _comparable(root: Path) -> str:
    ti._reset_for_tests()
    index = ti.read_index(root, SID)
    assert index is not None
    doc = ti._document(index)
    doc.pop("journal_generation", None)
    for key in ("inode", "mtime"):
        doc.get("sig", {}).pop(key, None)
    return json.dumps(doc, sort_keys=True, default=str)


async def _refresh(root: Path) -> Any:
    task, started = ti.start_refresh(root, SID)
    assert started
    return await task


def test_the_worker_publishes_the_thread_builds_document_and_its_companion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    thread_root, worker_root = tmp_path / "thread", tmp_path / "worker"
    _seed(thread_root)
    _seed(worker_root)

    monkeypatch.delenv(ti._SCAN_PROCESS_ENV, raising=False)
    built = asyncio.run(_refresh(thread_root))
    assert built is not None and ti._SCAN_POOL is None, "the default path must not start a pool"

    monkeypatch.setenv(ti._SCAN_PROCESS_ENV, "1")
    hydrated = asyncio.run(_refresh(worker_root))
    assert hydrated is not None, "mode 1 hydrates the publication for resident consumers"
    assert ti._SCAN_POOL is not None
    companions = list((ti._index_dir(worker_root) / "read-models" / SID).glob("*.sqlite"))
    assert companions, "the worker must publish the read-model companion before settling"

    assert _comparable(worker_root) == _comparable(thread_root)


def test_a_broken_pool_fails_one_refresh_and_the_next_builds_a_fresh_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(tmp_path)
    monkeypatch.setenv(ti._SCAN_PROCESS_ENV, "2")

    async def scenario() -> None:
        pool = ti._scan_pool()
        # Kill the pool's workers the way the memory guard would.
        pool.submit(os.getpid).result(timeout=60)
        for process in list(pool._processes.values()):  # noqa: SLF001 - test-only reach-in
            process.kill()
        with pytest.raises(Exception):
            await _refresh(tmp_path)
        assert ti._SCAN_POOL is None, "a broken pool must be dropped, not reused"
        assert await _refresh(tmp_path) is None  # mode 2: no serve-side hydration
        assert ti.index_path(tmp_path, SID).is_file()

    asyncio.run(scenario())
