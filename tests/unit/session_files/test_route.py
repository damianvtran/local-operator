"""``GET``/``POST /v1/desktop/sessions/{id}/files`` over HTTP, against the real app wiring.

Driven through the REAL router, pool, ``require_desktop`` dependency and shared failure ladder
(``errors``), because the things that go wrong live at the boundary: who may call, which ids
exist, what a draft or a peer answers, and what a bad cursor looks like on the wire. The
roster test builds a real ``subagent-roster.v1.json`` and proves BOTH directions of the
containment rule (a rostered child is listed; an unrostered directory that merely exists is not).
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.server.routes import (
    capabilities,
    desktop_session_files,
    desktop_sessions,
)
from local_operator.server.utils.desktop_sessions import DesktopSessions
from local_operator.session.attachments import cache_media

from ._fixtures import (
    CHILD,
    SESSION,
    assistant,
    call,
    generate_image_rows,
    png,
    tool,
    write_journal,
)

pytestmark = pytest.mark.asyncio

TOKEN = "session-files-test-token"
OTHER = "aaaaaaaaaaaa"


@pytest_asyncio.fixture
async def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for name in list(os.environ):
        if name.startswith(("CMUX_", "LOP_")):
            monkeypatch.delenv(name)
    root = tmp_path / "cfg"
    root.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    app = FastAPI()
    app.state.config_manager = ConfigManager(root)
    app.state.desktop_sessions = DesktopSessions(root)
    app.include_router(desktop_session_files.router)
    app.include_router(desktop_sessions.router)
    app.include_router(capabilities.router)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        yield client, app, root
    await app.state.desktop_sessions.close()


def _session(root: Path, session_id: str = SESSION, *, origin: str | None = None) -> Path:
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "desktop.json").write_text(json.dumps({"version": 1, "cwd": str(root.parent)}))
    if origin:
        (directory / "origin.json").write_text(json.dumps({"origin": origin}))
    return directory


def _artifact(seed: int) -> str:
    block = cache_media(png(seed=seed), "image/png", kind="image", name=f"img-{seed}.png")
    assert block is not None and block.attachment
    return block.attachment


def _seed_images(root: Path, n: int, session_id: str = SESSION) -> list[str]:
    directory = _session(root, session_id)
    digests = [_artifact(seed) for seed in range(1, n + 1)]
    rows: list[dict[str, Any]] = []
    for i, digest in enumerate(digests, start=1):
        rows += generate_image_rows(
            i, float(i * 10), digest, name=f"img-{i}.png", with_artifacts=True
        )
    write_journal(directory, rows)
    return digests


URL = f"/v1/desktop/sessions/{SESSION}/files"


async def test_list_returns_artifact_rows_and_requires_the_bearer(api) -> None:
    client, app, root = api
    digests = _seed_images(root, 3)
    response = await client.get(URL)
    assert response.status_code == 200, response.text
    data = response.json()["result"]
    assert data["state"] == "ready" and data["total"] == 3 and data["scope"] == "family"
    assert [r["ref"]["digest"] for r in data["rows"]] == digests[::-1], "newest first by default"
    assert set(data["rows"][0]) >= {
        "id",
        "ref",
        "name",
        "kind",
        "size_bytes",
        "created_at",
        "modified_at",
        "origin",
        "tool",
        "highlighted",
        "session",
        "exists",
        "first_seq",
        "last_ts",
    }
    assert data["counts"] == {"image": 3}

    for headers in ({"Authorization": "Bearer wrong"}, {"Authorization": ""}):
        denied = await client.get(URL, headers=headers)
        assert denied.status_code == 401, denied.text
    post = await client.post(
        f"{URL}/{'0' * 16}/materialise", headers={"Authorization": "Bearer wrong"}
    )
    assert post.status_code == 401


async def test_bad_session_id_shapes_are_refused_before_any_handler(api) -> None:
    client, _, _ = api
    for bad in ("short", "ABCDEF123456", "abcdef12345g", "../../etc", "abcdef1234567"):
        response = await client.get(f"/v1/desktop/sessions/{bad}/files")
        assert response.status_code in (404, 422), (bad, response.status_code)


async def test_unknown_session_is_404_with_the_shared_envelope(api) -> None:
    client, _, _ = api
    response = await client.get(URL)
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "wire.errors.desktop.requested_not_found"


async def test_wrong_session_sees_none_of_another_sessions_files(api) -> None:
    client, _, root = api
    _seed_images(root, 2)
    _session(root, OTHER)
    write_journal(root / "sessions" / OTHER, [])
    other = await client.get(f"/v1/desktop/sessions/{OTHER}/files")
    assert other.status_code == 200 and other.json()["result"]["rows"] == []


async def test_draft_answers_an_empty_ready_page_not_a_404(api) -> None:
    client, app, root = api
    draft_id = await app.state.desktop_sessions.mint_draft(str(root.parent))
    response = await client.get(f"/v1/desktop/sessions/{draft_id}/files")
    assert response.status_code == 200
    data = response.json()["result"]
    assert data["state"] == "ready" and data["rows"] == [] and data["total"] == 0


async def test_peer_session_answers_unsupported_with_no_rows(api, monkeypatch) -> None:
    client, _, root = api
    monkeypatch.setattr(
        "local_operator.session.remote_open.remote_row_and_silence",
        lambda session_id, root=None: (object() if session_id == OTHER else None, ()),
    )
    peer = await client.get(f"/v1/desktop/sessions/{OTHER}/files")
    assert peer.status_code == 200
    assert peer.json()["result"]["state"] == "unsupported" and peer.json()["result"]["rows"] == []
    assert (
        await client.get(URL)
    ).status_code == 404, "an id nobody holds is still the ordinary 404"


async def test_a_silent_peer_makes_an_unknown_id_a_409_not_a_404(api, monkeypatch) -> None:
    client, _, _ = api
    from local_operator.session.peer_rows import UnansweredPeer

    silent = (UnansweredPeer("dev-1", "laptop", "timeout"),)
    monkeypatch.setattr(
        "local_operator.session.remote_open.remote_row_and_silence",
        lambda session_id, root=None: (None, silent),
    )
    response = await client.get(URL)
    assert response.status_code == 409 and response.json()["detail"]["code"] == "session_unresolved"


async def test_a_subagent_directory_cannot_be_listed_by_naming_it(api) -> None:
    client, _, root = api
    _session(root, CHILD, origin="subagent")
    write_journal(root / "sessions" / CHILD, [])
    assert (await client.get(f"/v1/desktop/sessions/{CHILD}/files")).status_code == 404


@pytest.mark.parametrize(
    "query",
    [
        "scope=galaxy",
        "sort=color",
        "order=sideways",
        "origin=teleported",
        "kind=Bad%20Kind!",
        "highlighted=maybe",
    ],
)
async def test_bad_filters_are_422(api, query: str) -> None:
    client, _, root = api
    _seed_images(root, 1)
    response = await client.get(f"{URL}?{query}")
    assert response.status_code == 422, (query, response.text)


async def test_limit_and_q_bounds(api) -> None:
    client, _, root = api
    _seed_images(root, 1)
    for query in ("limit=0", "limit=201", f"q={'x' * 257}"):
        assert (await client.get(f"{URL}?{query}")).status_code == 422, query
    assert (await client.get(f"{URL}?limit=200")).status_code == 200


async def test_paging_walks_every_row_once_and_bad_cursors_are_422(api) -> None:
    client, _, root = api
    digests = _seed_images(root, 5)
    seen: list[str] = []
    cursor = None
    while True:
        url = f"{URL}?limit=2" + (f"&cursor={cursor}" if cursor else "")
        data = (await client.get(url)).json()["result"]
        seen += [r["ref"]["digest"] for r in data["rows"]]
        cursor = data["next_cursor"]
        if not cursor:
            break
    assert seen == digests[::-1]

    garbage = await client.get(f"{URL}?cursor=not-a-cursor")
    assert (
        garbage.status_code == 422
        and garbage.json()["detail"]["code"] == "session_files_bad_cursor"
    )
    first = (await client.get(f"{URL}?limit=2&sort=name")).json()["result"]["next_cursor"]
    cross = await client.get(f"{URL}?limit=2&sort=size&cursor={first}")
    assert cross.status_code == 422, "a cursor minted for another sort must not be honoured"


async def test_filter_sort_search_over_http(api) -> None:
    client, _, root = api
    directory = _session(root)
    work = root.parent
    (work / "report.md").write_text("# hi")
    (work / "data.csv").write_text("a,b")
    digest = _artifact(1)
    rows = generate_image_rows(1, 10.0, digest, name="sunset.png", with_artifacts=True)
    for i, name in enumerate(("report.md", "data.csv")):
        path = str(work / name)
        rows += [
            assistant(f"w{i}", 20.0 + i, [call(f"cw{i}", "write", path=path)]),
            tool(f"wt{i}", 21.0 + i, f"cw{i}", "write", "ok", {"path": path}),
        ]
    write_journal(directory, rows)

    kinds = (await client.get(f"{URL}?kind=image")).json()["result"]
    assert [r["name"] for r in kinds["rows"]] == ["sunset.png"] and kinds["counts"] == {
        "image": 1,
        "markdown": 1,
        "csv": 1,
    }
    origin = (await client.get(f"{URL}?origin=written")).json()["result"]
    assert sorted(r["name"] for r in origin["rows"]) == ["data.csv", "report.md"]
    found = (await client.get(f"{URL}?q=SUNS")).json()["result"]
    assert [r["name"] for r in found["rows"]] == ["sunset.png"]
    by_name = (await client.get(f"{URL}?sort=name&order=asc")).json()["result"]
    assert [r["name"] for r in by_name["rows"]] == ["data.csv", "report.md", "sunset.png"]
    path_row = next(r for r in by_name["rows"] if r["name"] == "report.md")
    assert path_row["ref"] == {
        "type": "path",
        "path": str(work / "report.md"),
        "digest": None,
        "content_type": None,
    }
    assert (
        path_row["exists"] is True
        and path_row["size_bytes"] == 4
        and path_row["created_at"]
        and path_row["modified_at"]
    )


async def test_roster_containment_lists_children_and_refuses_strangers(api) -> None:
    client, _, root = api
    parent = _session(root)
    digest = _artifact(4)
    write_journal(parent, [])
    child = _session(root, CHILD, origin="subagent")
    write_journal(child, generate_image_rows(1, 5.0, digest, name="kid.png", with_artifacts=True))
    stranger_id = "111111111111"
    stranger = _session(root, stranger_id, origin="subagent")
    write_journal(
        stranger,
        generate_image_rows(1, 6.0, _artifact(5), name="stranger.png", with_artifacts=True),
    )
    escape = {
        "job_id": "j3",
        "label": "evil",
        "depth": 1,
        "session_dir": str(root.parent / "elsewhere" / "222222222222"),
    }
    (parent / "subagent-roster.v1.json").write_text(
        json.dumps(
            {
                "version": 1,
                "generation": 1,
                "jobs": [],
                "accounting": {},
                "records": [
                    {
                        "job_id": "j1",
                        "label": "scout",
                        "agent_role": "scout",
                        "depth": 1,
                        "session_dir": str(child),
                    },
                    escape,
                ],
            }
        )
    )
    family = (await client.get(URL)).json()["result"]
    assert [r["name"] for r in family["rows"]] == [
        "kid.png"
    ], "the unrostered subagent dir stays invisible"
    row = family["rows"][0]
    assert row["session"] == {"id": CHILD, "role": "subagent", "child_id": CHILD, "label": "scout"}
    assert family["scan"]["children"] == 1 and family["scan"]["children_skipped"] == 1
    own = (await client.get(f"{URL}?scope=self")).json()["result"]
    assert own["rows"] == [] and own["scan"]["children"] == 0


async def test_materialise_ok_traversal_and_refusals_over_http(api) -> None:
    client, _, root = api
    _seed_images(root, 1)
    row_id = (await client.get(URL)).json()["result"]["rows"][0]["id"]

    ok = await client.post(f"{URL}/{row_id}/materialise")
    assert ok.status_code == 200, ok.text
    data = ok.json()["result"]
    saved = Path(data["path"])
    assert data["created"] is True and saved.is_file() and saved.read_bytes() == png(seed=1)
    assert saved.parent == (root / "sessions" / SESSION / "scratchpad" / "generated").resolve()
    again = await client.post(f"{URL}/{row_id}/materialise")
    assert (
        again.json()["result"]["created"] is False
        and again.json()["result"]["path"] == data["path"]
    )

    for bad in ("..", "%2e%2e%2f%2e%2e", "ZZZZZZZZZZZZZZZZ", "0" * 15):
        refused = await client.post(f"{URL}/{bad}/materialise")
        assert refused.status_code in (404, 422), (bad, refused.status_code)
    unknown = await client.post(f"{URL}/{'0' * 16}/materialise")
    assert (
        unknown.status_code == 404
        and unknown.json()["detail"]["code"] == "session_files_file_not_found"
    )
    assert sorted(p.name for p in saved.parent.iterdir()) == [
        saved.name
    ], "nothing else was written"


async def test_materialise_a_path_row_is_409(api) -> None:
    client, _, root = api
    directory = _session(root)
    path = str(root.parent / "a.md")
    (root.parent / "a.md").write_text("x")
    write_journal(
        directory,
        [
            assistant("a", 1, [call("c", "write", path=path)]),
            tool("t", 2, "c", "write", "ok", {"path": path}),
        ],
    )
    row_id = (await client.get(URL)).json()["result"]["rows"][0]["id"]
    refused = await client.post(f"{URL}/{row_id}/materialise")
    assert (
        refused.status_code == 409
        and refused.json()["detail"]["code"] == "session_files_not_an_artifact"
    )


async def test_capability_key_is_advertised(api) -> None:
    client, _, _ = api
    response = await client.get("/v1/capabilities")
    assert response.status_code == 200
    assert response.json()["result"]["features"]["session_files"] == 1


async def test_a_cold_scan_over_budget_answers_building_then_completes_in_the_background(
    api, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Lazy, resumable backfill, end to end over HTTP.

    A zero budget forces the cold scan to publish a PARTIAL answer, so the route must say
    ``building`` with real (partial) rows, start ONE single-flight background task that drives
    the scan to completion in slices, and the next poll must answer ``ready`` with everything.
    The event is awaited on the task itself, never on a clock (AGENTS.md "Timing").
    """
    client, _, root = api
    directory = _session(root)
    rows: list[dict[str, Any]] = []
    for n in range(300):
        rows += generate_image_rows(n, float(n), f"{n:032x}", with_artifacts=True)
    write_journal(directory, rows)
    monkeypatch.setattr(desktop_session_files.service, "DEFAULT_BUDGET_S", 0.0)

    # Hold the background completion on an EVENT, so "a second poll joins the same flight" is a
    # fact about ordering rather than about how fast this machine finishes a 300-row scan.
    release = threading.Event()
    started: list[int] = []
    real = desktop_session_files.service.complete_in_background

    def gated(*args: Any, **kwargs: Any) -> Any:
        started.append(1)
        assert release.wait(timeout=30), "the test never released the background scan"
        return real(*args, **kwargs)

    monkeypatch.setattr(desktop_session_files.service, "complete_in_background", gated)

    first = (await client.get(URL + "?limit=200")).json()["result"]
    assert first["state"] == "building" and first["scan"]["complete"] is False
    assert 0 < first["total"] < 300, "partial rows are real rows"
    assert 0 < first["scan"]["covered_bytes"] < first["scan"]["total_bytes"]

    key = (str(root), SESSION, "family")
    task = desktop_session_files._IN_FLIGHT[key]
    second = (await client.get(URL)).json()["result"]
    assert second["state"] == "building"
    assert desktop_session_files._IN_FLIGHT[key] is task, "a second poll joins the same flight"
    release.set()
    await task
    assert key not in desktop_session_files._IN_FLIGHT and len(started) == 1, "single-flight"

    done = (await client.get(URL + "?limit=200")).json()["result"]
    assert done["state"] == "ready" and done["total"] == 300 and done["scan"]["complete"] is True
    assert done["revision"] > first["revision"]


async def test_materialise_of_a_pruned_store_entry_is_410_over_http(api) -> None:
    client, _, root = api
    digest = _seed_images(root, 1)[0]
    row_id = (await client.get(URL)).json()["result"]["rows"][0]["id"]
    (root / "attachments" / f"{digest}.bin").unlink()
    gone = await client.post(f"{URL}/{row_id}/materialise")
    assert (
        gone.status_code == 410
        and gone.json()["detail"]["code"] == "session_files_artifact_missing"
    )
    assert not (root / "sessions" / SESSION / "scratchpad" / "generated").exists()


async def test_sensitive_scratchpad_symlink_target_is_not_listed(api) -> None:
    client, _, root = api
    directory = _session(root)
    scratch = directory / "scratchpad"
    scratch.mkdir()
    target = root / "master.key"
    target.write_text("synthetic fixture, not a secret")
    report = scratch / "report.txt"
    report.symlink_to(target)
    harmless = scratch / "notes.txt"
    harmless.write_text("public fixture")
    write_journal(
        directory,
        [
            tool("bad", 1, "cb", "write", "ok", {"path": str(report)}),
            tool("good", 2, "cg", "write", "ok", {"path": str(harmless)}),
        ],
    )
    answer = (await client.get(URL)).json()["result"]
    assert [row["name"] for row in answer["rows"]] == ["notes.txt"]


@pytest.mark.parametrize("swapped_component", ["generated", "scratchpad"])
async def test_parent_swap_at_exclusive_create_never_writes_outside(
    api, monkeypatch, swapped_component
) -> None:
    client, _, root = api
    _seed_images(root, 1)
    row_id = (await client.get(URL)).json()["result"]["rows"][0]["id"]
    generated = root / "sessions" / SESSION / "scratchpad" / "generated"
    outside = root.parent / "outside"
    outside.mkdir()
    victim_parent = generated if swapped_component == "generated" else generated.parent
    parked = victim_parent.with_name("parked")
    real_open = os.open
    swapped = False

    def swap_before_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if flags & os.O_CREAT and not swapped:
            swapped = True
            victim_parent.rename(parked)
            victim_parent.symlink_to(outside, target_is_directory=True)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    # The wrappers still implement the same platform operation; retain capability detection.
    monkeypatch.setattr(os, "supports_dir_fd", os.supports_dir_fd | {swap_before_open})
    monkeypatch.setattr(os, "open", swap_before_open)
    response = await client.post(f"{URL}/{row_id}/materialise")
    assert swapped and response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "session_files_unsafe_destination"
    assert list(outside.iterdir()) == []
    parked_generated = parked if swapped_component == "generated" else parked / "generated"
    assert list(parked_generated.iterdir()) == []


async def test_symlinked_scratchpad_refusal_is_409_not_500(api) -> None:
    client, _, root = api
    _seed_images(root, 1)
    row_id = (await client.get(URL)).json()["result"]["rows"][0]["id"]
    outside = root.parent / "outside"
    outside.mkdir()
    (root / "sessions" / SESSION / "scratchpad").symlink_to(outside)
    response = await client.post(f"{URL}/{row_id}/materialise")
    assert (
        response.status_code == 409
        and response.json()["detail"]["code"] == "session_files_unsafe_destination"
    )
    assert list(outside.iterdir()) == []


async def test_roster_membership_add_remove_invalidates_cursor(api) -> None:
    client, _, root = api
    parent = _session(root)
    child = _session(root, CHILD, origin="subagent")
    digest = _artifact(8)
    write_journal(child, generate_image_rows(1, 1, digest, name="a.png", with_artifacts=True))
    # Warm child FIRST: adding/removing it cannot be detected by max(parent, child) clock.
    from local_operator.session_files import ledger

    ledger.refresh(root, CHILD, child, cwd=str(root.parent), home=str(root.parent))
    write_journal(
        parent, generate_image_rows(2, 2, _artifact(9), name="b.png", with_artifacts=True)
    )
    sidecar = parent / "subagent-roster.v1.json"

    def roster(on):
        sidecar.write_text(
            json.dumps(
                {
                    "version": 1,
                    "generation": 1,
                    "jobs": [],
                    "records": (
                        [{"job_id": "j", "session_dir": str(child), "depth": 1}] if on else []
                    ),
                }
            )
        )

    roster(True)
    page = (await client.get(URL + "?sort=name&limit=1")).json()["result"]
    assert page["rows"][0]["name"] == "a.png"
    roster(False)
    removed = (await client.get(URL + "?sort=name&limit=1&cursor=" + page["next_cursor"])).json()[
        "result"
    ]
    assert removed["cursor_reset"] and removed["rows"][0]["name"] == "b.png"
    assert removed["revision"] > page["revision"]
    # A previously-issued cursor remains invalid when the old cached member returns.
    roster(True)
    added = (await client.get(URL + "?sort=name&limit=1&cursor=" + page["next_cursor"])).json()[
        "result"
    ]
    assert added["cursor_reset"] and added["revision"] > removed["revision"]


async def test_warm_no_roster_route_does_not_parse_journal_and_self_skips_discovery(
    api, monkeypatch
) -> None:
    client, _, root = api
    directory = _session(root)
    rows = [assistant(f"prose-{n}", float(n), []) for n in range(15_000)]
    write_journal(directory, rows)
    await client.get(URL)  # complete the authoritative pass, including roster absence
    from local_operator.session_files import scan

    parses = []
    real = scan._row
    monkeypatch.setattr(scan, "_row", lambda raw: parses.append(1) or real(raw))
    monkeypatch.setattr(
        "local_operator.server.utils.desktop_sessions.read_latest_custom_entry",
        lambda *a, **kw: pytest.fail("unbounded journal fallback"),
    )
    for _ in range(3):
        response = await client.get(URL)
        assert response.status_code == 200 and response.json()["result"]["total"] == 0
    assert parses == [], "warm no-roster requests do no journal JSON parsing"
    monkeypatch.setattr(
        desktop_session_files,
        "contained_descendants",
        lambda *a, **kw: pytest.fail("scope=self discovered family"),
    )
    assert (await client.get(URL + "?scope=self")).status_code == 200


async def test_concurrent_cold_route_returns_building_without_waiting_for_scanner(
    api, monkeypatch
) -> None:
    import asyncio

    from local_operator.session_files import ledger

    client, _, root = api
    _seed_images(root, 1)
    entered, release = threading.Event(), threading.Event()
    real = ledger._read_into

    def held(*args):
        entered.set()
        assert release.wait(10), "test did not release the scanner"
        return real(*args)

    monkeypatch.setattr(ledger, "_read_into", held)
    monkeypatch.setattr(desktop_session_files, "_kick", lambda *a: None)
    owner = asyncio.create_task(client.get(URL))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        # Deterministic ordering, not a latency SLA: every loser MUST finish while the owner
        # is still gated. The old blocking lock deadlocks until this negative-control timeout.
        burst = await asyncio.wait_for(asyncio.gather(*[client.get(URL) for _ in range(8)]), 3)
        assert not owner.done()
        assert all(
            r.status_code == 200 and r.json()["result"]["state"] == "building" for r in burst
        )
    finally:
        release.set()
        await owner


async def test_legacy_roster_is_indexed_once_and_refreshes_on_append(api, monkeypatch) -> None:
    client, _, root = api
    parent = _session(root)
    child = _session(root, CHILD, origin="subagent")
    write_journal(child, generate_image_rows(1, 1, _artifact(5), name="child.png"))
    write_journal(parent, [assistant("first", 1, [])])
    assert (await client.get(URL)).json()["result"]["total"] == 0
    from ._fixtures import append_journal

    append_journal(
        parent,
        [
            {
                "id": "legacy",
                "ts": 2,
                "type": "custom",
                "payload": {
                    "custom_type": "subagent_roster",
                    "details": {
                        "records": [{"session_dir": str(child), "job_id": "legacy-job", "depth": 1}]
                    },
                },
            }
        ],
    )
    monkeypatch.setattr(
        "local_operator.server.utils.desktop_sessions.read_latest_custom_entry",
        lambda *a, **kw: pytest.fail("unbounded roster fallback"),
    )
    for _ in range(2):
        answer = (await client.get(URL)).json()["result"]
        assert [r["name"] for r in answer["rows"]] == ["child.png"]
    # A new process reloads the bounded scan's roster, rather than losing legacy children.
    from local_operator.session_files import ledger

    ledger.forget_resident()
    answer = (await client.get(URL)).json()["result"]
    assert answer["scan"]["children"] == 1


async def test_contended_refresh_serves_published_rows_as_building(api, monkeypatch) -> None:
    import asyncio

    from local_operator.session_files import ledger

    from ._fixtures import append_journal

    client, _, root = api
    _seed_images(root, 1)
    first = (await client.get(URL)).json()["result"]
    directory = root / "sessions" / SESSION
    append_journal(directory, generate_image_rows(2, 50, _artifact(2), name="new.png"))
    entered, release = threading.Event(), threading.Event()
    real = ledger._read_into

    def held(*args):
        entered.set()
        assert release.wait(10)
        return real(*args)

    monkeypatch.setattr(ledger, "_read_into", held)
    monkeypatch.setattr(desktop_session_files, "_kick", lambda *a: None)
    owner = asyncio.create_task(client.get(URL))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        loser = (await asyncio.wait_for(client.get(URL), 3)).json()["result"]
        assert loser["state"] == "building" and loser["total"] == first["total"]
        assert loser["rows"][0]["id"] == first["rows"][0]["id"]
    finally:
        release.set()
        await owner
    final = (await client.get(URL)).json()["result"]
    assert final["state"] == "ready" and final["total"] == 2
