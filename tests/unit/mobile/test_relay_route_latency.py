"""Regression tests for the relay's three slow routes (operator report, 2026-11-14).

The phone's resume burst -- ``GET /api/sessions?include_peers=true`` beside its
list stream, ``GET /api/models`` twice and ``GET /api/directories`` twice --
paid, on the request path:

* the federated peer listing, a blocking fan-out to every peer device
  (1.1-2.8 s measured) whenever its 20 s cache had lapsed, i.e. every resume;
* a full model-catalogue build per request (140-500 ms of GIL-holding work,
  plus provider round trips once a listing passed its 15-minute TTL);
* two full ``agent.yml`` scans per directories request (~290-430 ms each).

These pin the STRUCTURAL fix rather than a wall-clock number (AGENTS.md
"Prefer a structural invariant to a numeric one"): how many builds/reads a
sequence of requests causes, and that a stale cache answers without waiting.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest
from starlette.testclient import TestClient

import local_operator.mobile.daemon as daemon_module
from local_operator.mobile.auth import COOKIE_NAME, sign_cookie
from local_operator.mobile.daemon import (
    MobileDaemon,
    _ModelsMemo,
    _PeerRowsRefresher,
    build_app,
)
from local_operator.resume import SessionRow
from local_operator.session import peer_rows

# -- /api/models ------------------------------------------------------------------


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _memo(builds: list[int], key: list[object], clock: _Clock) -> _ModelsMemo:
    def build() -> dict[str, object]:
        builds.append(1)
        return {"models": [len(builds)], "default": None}

    return _ModelsMemo(build, lambda: key[0], fresh_s=60.0, stale_s=3600.0, clock=clock)


def test_models_concurrent_requests_share_one_build_and_reuse_it_while_fresh() -> None:
    builds: list[int] = []
    clock = _Clock()
    memo = _memo(builds, ["k1"], clock)

    async def run() -> list[dict[str, object]]:
        # The app's duplicate pair, logged 10-30 ms apart on the operator's tunnel.
        first = await asyncio.gather(memo.get(), memo.get())
        clock.now += 59.0
        return [*first, await memo.get()]

    answers = asyncio.run(run())
    assert len(builds) == 1, "the pair and a re-open inside FRESH must cost ONE build"
    assert all(answer == {"models": [1], "default": None} for answer in answers)


def test_models_fingerprint_change_rebuilds_before_answering() -> None:
    """A login, logout or default change must never be answered from the memo."""
    builds: list[int] = []
    key: list[object] = ["k1"]
    memo = _memo(builds, key, _Clock())

    async def run() -> tuple[dict[str, object], dict[str, object]]:
        before = await memo.get()
        key[0] = "k2"
        return before, await memo.get()

    before, after = asyncio.run(run())
    assert len(builds) == 2
    assert before["models"] == [1] and after["models"] == [2]


def test_models_stale_body_answers_at_once_while_one_background_build_runs() -> None:
    builds: list[int] = []
    clock = _Clock()
    memo = _memo(builds, ["k1"], clock)

    async def run() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        await memo.get()
        clock.now += 61.0  # past FRESH, inside STALE
        stale_a, stale_b = await asyncio.gather(memo.get(), memo.get())
        # Let the single background rebuild land.
        for _ in range(50):
            if len(builds) == 2:
                break
            await asyncio.sleep(0.01)
        return stale_a, stale_b, await memo.get()

    stale_a, stale_b, refreshed = asyncio.run(run())
    assert stale_a["models"] == [1] and stale_b["models"] == [1], "stale answers must not wait"
    assert len(builds) == 2, "two stale requests must start ONE background rebuild"
    assert refreshed["models"] == [2]


def test_models_past_stale_bound_waits_for_a_fresh_build() -> None:
    builds: list[int] = []
    clock = _Clock()
    memo = _memo(builds, ["k1"], clock)

    async def run() -> dict[str, object]:
        await memo.get()
        clock.now += 3601.0
        return await memo.get()

    assert asyncio.run(run())["models"] == [2]


def test_models_failed_build_is_not_memoised() -> None:
    calls: list[int] = []

    def build() -> dict[str, object]:
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("catalogue unavailable")
        return {"models": ["ok"], "default": None}

    memo = _ModelsMemo(build, lambda: "k", clock=_Clock())

    async def run() -> dict[str, object]:
        with pytest.raises(RuntimeError):
            await memo.get()
        return await memo.get()

    assert asyncio.run(run())["models"] == ["ok"]
    assert len(calls) == 2


def _authed_client(daemon: MobileDaemon) -> TestClient:
    client = TestClient(build_app(daemon), follow_redirects=False)
    client.cookies.set(COOKIE_NAME, sign_cookie("pw"))
    return client


def test_models_route_builds_once_for_the_apps_duplicate_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def list_models() -> list[dict[str, object]]:
        calls.append(1)
        return [{"selector": "p/m"}]

    monkeypatch.setattr(daemon_module, "_list_models", list_models)
    monkeypatch.setattr(daemon_module, "_configured_default", lambda: None)
    with _authed_client(MobileDaemon(password="pw", dial_registrants=False)) as client:
        first = client.get("/api/models")
        second = client.get("/api/models")
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json() == {"models": [{"selector": "p/m"}], "default": None}
    assert len(calls) == 1


def test_models_route_still_answers_a_failed_build_with_502(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken() -> list[dict[str, object]]:
        raise RuntimeError("Model catalogue unavailable for OpenAI; retry or log in again")

    monkeypatch.setattr(daemon_module, "_list_models", broken)
    with _authed_client(MobileDaemon(password="pw", dial_registrants=False)) as client:
        response = client.get("/api/models")
    assert response.status_code == 502
    assert "Model catalogue unavailable" in response.json()["error"]


def test_models_fingerprint_moves_with_credentials_and_config() -> None:
    from local_operator.paths import config_dir

    root = config_dir()  # the suite's isolated root, never the operator's
    root.mkdir(parents=True, exist_ok=True)
    empty = daemon_module._models_fingerprint()
    (root / "config.yml").write_text("values: {}\n")
    with_config = daemon_module._models_fingerprint()
    assert with_config != empty
    (root / "auth.db-wal").write_bytes(b"x")  # a committed login not yet checkpointed
    assert daemon_module._models_fingerprint() != with_config


# -- /api/directories ---------------------------------------------------------------


def _write_agent(root: Path, agent_id: str, cwd: Path, when: str) -> None:
    folder = root / "agents" / agent_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "agent.yml").write_text(
        f"id: {agent_id}\nname: {agent_id}\ncreated_date: '2026-01-01T00:00:00Z'\n"
        f"version: '1'\nlast_message: ''\nlast_message_datetime: '{when}'\n"
        f"current_working_directory: {cwd}\n"
    )


@pytest.fixture
def counted_registry(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    from local_operator import agents

    constructed: list[int] = []
    real = agents.AgentRegistry

    class Counting(real):  # type: ignore[misc, valid-type]
        def __init__(self, *args: object, **kwargs: object) -> None:
            constructed.append(1)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(agents, "AgentRegistry", Counting)
    monkeypatch.setattr(daemon_module, "_AGENT_CWDS_MEMO", None)
    return constructed


def test_directories_route_scans_the_registry_once_not_per_reader(
    counted_registry: list[int],
) -> None:
    """``recent`` and ``default`` used to construct the registry once EACH."""
    from local_operator.paths import config_dir

    root = config_dir()
    work = Path.home() / "work-a"
    work.mkdir(parents=True, exist_ok=True)
    _write_agent(root, "agent-a", work, "2026-11-14T10:00:00Z")
    with _authed_client(MobileDaemon(password="pw", dial_registrants=False)) as client:
        first = client.get("/api/directories").json()
        second = client.get("/api/directories").json()
    assert first == second
    assert first["recent"] == [str(work)]
    assert first["default"] == str(work.resolve())
    assert len(counted_registry) == 1, "two requests, two readers each: ONE scan"


def test_directories_memo_sees_a_manifest_write_and_a_deleted_directory(
    counted_registry: list[int],
) -> None:
    from local_operator.paths import config_dir

    root = config_dir()
    older, newer = Path.home() / "older", Path.home() / "newer"
    older.mkdir(parents=True, exist_ok=True)
    newer.mkdir(parents=True, exist_ok=True)
    _write_agent(root, "agent-a", older, "2026-11-14T10:00:00Z")
    assert daemon_module._recent_directories() == [str(older)]

    # A new agent (a write the fingerprint sees) re-ranks at once.
    _write_agent(root, "agent-b", newer, "2026-11-14T11:00:00Z")
    assert daemon_module._recent_directories() == [str(newer), str(older)]
    assert len(counted_registry) == 2

    # A directory removed with NO registry change stops being offered at once:
    # the gate/is_dir filter is per call, never memoised.
    newer.rmdir()
    assert daemon_module._recent_directories() == [str(older)]
    assert len(counted_registry) == 2


# -- /api/sessions?include_peers=true ---------------------------------------------


@pytest.fixture
def fake_fanout(monkeypatch: pytest.MonkeyPatch):
    """Replace the mesh fan-out with a gated, counted fake."""

    class Fanout:
        def __init__(self) -> None:
            self.reads = 0
            self.name = "first"
            self.gate = threading.Event()
            self.gate.set()

        def __call__(self, root: object, catalog: object) -> tuple[tuple[SessionRow, ...], tuple]:
            self.reads += 1
            assert self.gate.wait(10), "test fan-out was never released"
            return (SessionRow(id="remote-1", mtime=1.0, name=self.name),), ()

    fanout = Fanout()
    monkeypatch.setattr(peer_rows, "_read", fanout)
    monkeypatch.setattr(peer_rows, "_CACHE", {})
    return fanout


def _names(rows: list[dict[str, object]]) -> list[object]:
    return [row["conversation_name"] for row in rows]


def test_peers_cold_cache_waits_once_for_concurrent_frames(fake_fanout, tmp_path: Path) -> None:
    rings: list[int] = []
    refresher = _PeerRowsRefresher(lambda: rings.append(1))

    async def run() -> list[list[dict[str, object]]]:
        # The phone's GET and its stream's first frame arrive together.
        return list(
            await asyncio.gather(refresher.rows(tmp_path, set()), refresher.rows(tmp_path, set()))
        )

    answers = asyncio.run(run())
    assert [_names(rows) for rows in answers] == [["first"], ["first"]]
    assert fake_fanout.reads == 1


def test_peers_stale_cache_answers_without_waiting_and_rings_when_fresh(
    fake_fanout, tmp_path: Path
) -> None:
    rings: list[int] = []
    refresher = _PeerRowsRefresher(lambda: rings.append(1))

    async def run() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        await refresher.rows(tmp_path, set())  # cold: the one read that waits
        rings.clear()
        # Age the entry past the listing TTL, then hold the next fan-out open.
        moment, rows, unanswered, at_ms = peer_rows._CACHE[str(tmp_path)]
        peer_rows._CACHE[str(tmp_path)] = (
            moment - peer_rows.LISTING_TTL_S - 1,
            rows,
            unanswered,
            at_ms,
        )
        fake_fanout.gate.clear()
        fake_fanout.name = "second"
        # Must return while the fan-out is still blocked: the old rows, now.
        stale = await asyncio.wait_for(refresher.rows(tmp_path, set()), timeout=5)
        assert fake_fanout.reads == 2 or await _eventually(lambda: fake_fanout.reads == 2)
        assert rings == [], "no ring before the refresh lands"
        fake_fanout.gate.set()
        assert await _eventually(lambda: rings == [1]), "the landed refresh rings the list"
        return stale, await refresher.rows(tmp_path, set())

    stale, fresh = asyncio.run(run())
    assert _names(stale) == ["first"]
    assert _names(fresh) == ["second"]
    assert fake_fanout.reads == 2


async def _eventually(predicate, timeout: float = 5.0) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return predicate()


def test_remote_session_rows_with_infinite_ttl_never_dials(fake_fanout, tmp_path: Path) -> None:
    from local_operator.mobile import mesh

    peer_rows.read_listing(tmp_path, ttl_s=0)
    moment, rows, unanswered, at_ms = peer_rows._CACHE[str(tmp_path)]
    peer_rows._CACHE[str(tmp_path)] = (moment - 10_000, rows, unanswered, at_ms)
    assert _names(mesh.remote_session_rows(tmp_path, pinned=set(), ttl_s=float("inf"))) == ["first"]
    assert fake_fanout.reads == 1
    # The default keeps the listing's own TTL: a stale cache still dials.
    mesh.remote_session_rows(tmp_path, pinned=set())
    assert fake_fanout.reads == 2


def test_listing_age_treats_a_seeded_only_entry_as_no_listing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(peer_rows, "_CACHE", {})
    assert peer_rows.listing_age_s(tmp_path) is None
    peer_rows.seed_peer_row(tmp_path, SessionRow(id="x", mtime=1.0, name="x"))
    assert peer_rows.listing_age_s(tmp_path) == float("inf")
