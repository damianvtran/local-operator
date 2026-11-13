"""C5 of remote files parity: precache the media of active remote sessions.

docs/design/remote-files-parity.md §4 "Precache" and the C5 acceptance: the active
set (2 h window, the opened LRU, the cap of 16) on synthetic catalogue rows; a real
loopback pass that lands ONLY image/audio/video at most 32 MiB, respects the budget
and yields to an on-screen request; and the RAM check (10 x 16 MiB with an RSS
delta under 64 MiB).
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import files
from local_operator.remote_files import cache as remote_cache
from local_operator.remote_files import fetch as remote_fetch
from local_operator.remote_files import precache
from local_operator.session.attachments import AttachmentStore
from tests.unit.network.test_mesh_file_pull import mesh  # noqa: F401 — fixture
from tests.unit.network.test_mobility import SESSION, devices, pair  # noqa: F401
from tests.unit.network.test_remote_files_fetch import viewer  # noqa: F401 — fixture
from tests.unit.session_files._fixtures import append_journal, assistant, call, tool

NOW = 1_800_000_000.0


def _row(sid: str, *, age_s: float, **overrides: Any) -> dict[str, Any]:
    return {
        "id": sid,
        "mtime": NOW - age_s,
        "locality": "remote",
        "reachable": True,
        "archived": False,
        "owner_device": "d_owner",
        **overrides,
    }


def test_the_active_set_is_recent_reachable_unarchived_plus_opened_capped_at_16() -> None:
    rows = [_row(f"{i:012x}", age_s=60.0 * i) for i in range(30)]
    rows += [
        _row("a" * 12, age_s=60, archived=True),
        _row("b" * 12, age_s=60, reachable=False),
        _row("c" * 12, age_s=60, locality="local"),
        _row("d" * 12, age_s=10 * 3600),  # old, but opened here
        _row("e" * 12, age_s=10 * 3600),  # old and never opened
    ]
    chosen = [row["id"] for row in precache.active_set(rows, ["d" * 12], now=NOW)]
    assert chosen[0] == "d" * 12, "an opened session comes first whatever its age"
    assert len(chosen) == precache.MAX_SESSIONS == 16
    assert not {"a" * 12, "b" * 12, "c" * 12, "e" * 12} & set(chosen)
    assert chosen[1:] == [f"{i:012x}" for i in range(15)], "newest first"
    within = precache.active_set(rows[:3] + [_row("f" * 12, age_s=2 * 3600 + 1)], [], now=NOW)
    assert "f" * 12 not in [row["id"] for row in within], "the window is 2 h"


def test_a_session_is_re_listed_only_when_it_moved_and_at_most_once_a_minute(
    tmp_path: Path,
) -> None:
    worker = precache.Precacher(tmp_path)
    queued: list[str] = []
    worker._enqueue = lambda row: queued.append(row["id"])  # type: ignore[method-assign]
    row = _row("1" * 12, age_s=0)
    row["mtime"] = time.time()
    worker.observe([row])
    worker.passes[row["id"]] = (time.monotonic(), row["mtime"])
    worker.observe([row])  # unchanged mtime
    worker.observe([{**row, "mtime": row["mtime"] + 1}])  # moved, but within 60 s
    worker.passes[row["id"]] = (time.monotonic() - 61, row["mtime"])
    worker.observe([{**row, "mtime": row["mtime"] + 1}])  # moved, past 60 s
    assert queued == ["1" * 12, "1" * 12]
    (tmp_path / "config.yml").write_text("desktop:\n  remote_files_cache_bytes: 0\n")
    worker.observe([{**row, "mtime": row["mtime"] + 5}])
    assert len(queued) == 2, "a 0 budget turns precache off"


def _pad_media(server: Any, sizes: dict[str, int]) -> dict[str, bytes]:
    """Pad files written by the session (path rows), with their bytes."""
    directory = Path(server.root) / "sessions" / SESSION
    pad = directory / "scratchpad"
    pad.mkdir(parents=True, exist_ok=True)
    (directory / "desktop.json").write_text(json.dumps({"version": 1, "cwd": str(pad)}))
    written: dict[str, bytes] = {}
    rows: list[dict[str, Any]] = []
    for index, (name, size) in enumerate(sizes.items()):
        payload = os.urandom(size)
        (pad / name).write_bytes(payload)
        written[name] = payload
        rows += [
            assistant(
                f"pa{index}", 10.0 + index, [call(f"pc{index}", "write", path=str(pad / name))]
            ),
            tool(
                f"pt{index}", 10.5 + index, f"pc{index}", "write", "ok", {"path": str(pad / name)}
            ),
        ]
    append_journal(directory, rows)
    return written


def _catalogue_row(server_a: Any) -> dict[str, Any]:
    return _row(SESSION, age_s=0, mtime=time.time(), owner_device=server_a.identity.device_id)


def _cached(server_b: Any, payload: bytes) -> bool:
    import hashlib

    digest = hashlib.sha256(payload).hexdigest()[:32]
    return (remote_cache.cache_root(Path(server_b.root)) / f"{digest}.bin").exists()


@pytest.mark.asyncio
async def test_a_pass_lands_only_media_within_the_per_file_ceiling(viewer) -> None:  # noqa: F811
    _client, (server_a, server_b, *_rest) = viewer
    written = _pad_media(
        server_a,
        {
            "photo.png": 200_000,
            "clip.wav": 300_000,
            "movie.mp4": 1_000_000,
            "notes.txt": 50_000,
            "huge.mp4": precache.PER_FILE_MAX_BYTES + 1,
        },
    )
    worker = precache.Precacher(Path(server_b.root))
    landed = await worker.prepare(_catalogue_row(server_a))
    for name in ("photo.png", "clip.wav", "movie.mp4"):
        assert _cached(server_b, written[name]), name
    assert not _cached(server_b, written["notes.txt"]), "a document is not media"
    assert not _cached(server_b, written["huge.mp4"]), "over the per-file ceiling"
    assert landed == 1_500_000
    assert not list((Path(server_b.root) / "attachments").glob("*.bin"))


@pytest.mark.asyncio
async def test_a_pass_stops_at_the_budget_and_evicts_nothing_it_landed(
    viewer,  # noqa: F811
) -> None:
    _client, (server_a, server_b, *_rest) = viewer
    written = _pad_media(server_a, {f"v{i}.mp4": 400_000 for i in range(4)})
    (Path(server_b.root) / "config.yml").write_text(
        "desktop:\n  remote_files_cache_bytes: 1000000\n"
    )
    worker = precache.Precacher(Path(server_b.root))
    await worker.prepare(_catalogue_row(server_a))
    present = [name for name, payload in written.items() if _cached(server_b, payload)]
    assert len(present) == 2, present
    assert remote_cache.total_bytes(Path(server_b.root)) <= 1_000_000
    assert len(worker.landed) == 2


@pytest.mark.asyncio
async def test_an_on_screen_request_completes_while_the_precache_queue_is_full(
    viewer, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    client, (server_a, server_b, *_rest) = viewer
    _pad_media(server_a, {f"slow{i}.mp4": 300_000 for i in range(5)})
    on_screen = os.urandom(150_000)
    ref = AttachmentStore(Path(server_a.root) / "attachments").put_bytes(on_screen, "image/png")
    assert ref is not None
    append_journal(
        Path(server_a.root) / "sessions" / SESSION,
        [{"id": "os1", "content": [{"attachment": ref.digest}]}],
    )
    transfer = files._transfer

    def slow(asker: Any, fd: int, meta: dict[str, Any], start: int) -> None:
        if str(meta.get("name", "")).startswith("slow"):
            time.sleep(1.5)
        transfer(asker, fd, meta, start)

    monkeypatch.setattr(files, "_transfer", slow)
    worker = precache.Precacher(Path(server_b.root))
    running = asyncio.ensure_future(worker.prepare(_catalogue_row(server_a)))
    # Ask only once a precache pull is IN FLIGHT (holding the owner's semaphore):
    # asking between two pulls would prove nothing.
    for _ in range(200):
        if remote_fetch.pending_count() >= 1:
            break
        await asyncio.sleep(0.02)
    assert remote_fetch.pending_count() >= 1
    started = time.monotonic()
    response = await client.get(f"/v1/desktop/sessions/{SESSION}/attachments/{ref.digest}")
    elapsed = time.monotonic() - started
    assert response.status_code == 200 and response.content == on_screen
    assert not running.done(), "the precache finished first; the cell proved nothing"
    assert elapsed < 1.0, f"the on-screen request waited {elapsed:.2f}s behind precache"
    await running
    assert len(worker.landed) == 5


def _rss_bytes() -> int:
    with open("/proc/self/statm", encoding="ascii") as handle:
        return int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


@pytest.mark.skipif(not Path("/proc/self/statm").exists(), reason="needs /proc for RSS")
@pytest.mark.asyncio
async def test_precaching_160_mib_keeps_the_daemon_rss_flat(viewer) -> None:  # noqa: F811
    """§4's RAM check: 10 x 16 MiB precached with an RSS delta under 64 MiB. Both relays
    run in THIS process, so the delta also covers the owner's serving side."""
    _client, (server_a, server_b, *_rest) = viewer
    _pad_media(server_a, {f"big{i}.mp4": 16 * 1024 * 1024 for i in range(10)})
    worker = precache.Precacher(Path(server_b.root))
    before = _rss_bytes()
    landed = await worker.prepare(_catalogue_row(server_a))
    after = _rss_bytes()
    assert landed == 160 * 1024 * 1024
    assert after - before < 64 * 1024 * 1024, f"RSS grew {(after - before) / 2**20:.1f} MiB"
