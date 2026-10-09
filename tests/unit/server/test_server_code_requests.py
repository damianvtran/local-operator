"""``GET``/``POST /v1/desktop/sessions/{id}/code-requests`` over HTTP.

Driven against the real route and the real scanner, because the things that can go wrong
are at the boundary: the index is a file another process writes (so absence, a stale
index and a journal that moved underneath all have to degrade to a ROW rather than to a
500), and the first read of a session that has never been scanned must not block on a
124 MB journal.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.code_requests import ledger
from local_operator.code_requests.refs import HostContext, Remote
from local_operator.config import ConfigManager
from local_operator.server.routes import desktop_code_requests

TOKEN = "desktop-code-requests-token"
SESSION = "abcdef123456"
CWD = HostContext(remotes=(Remote("origin", "github.com", "damianvtran/local-operator"),))
PR = "https://github.com/damianvtran/local-operator/pull/1904"


@pytest_asyncio.fixture
async def desktop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    app = FastAPI()
    app.include_router(desktop_code_requests.router)
    app.state.config_manager = ConfigManager(tmp_path)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        yield client, tmp_path


def _seed_session(root: Path, session_id: str = SESSION, *, created: bool = True) -> Path:
    directory = root / "sessions" / session_id
    if not created:
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "id": "a1",
            "ts": time.time() - 60,
            "type": "message",
            "payload": {
                "kind": "message",
                "role": "assistant",
                "content": [{"text": "opening the PR"}],
                "tool_calls": [
                    {"id": "c1", "name": "bash", "arguments": {"command": "gh pr create --title t"}}
                ],
            },
        },
        {
            "id": "t1",
            "ts": time.time() - 59,
            "type": "message",
            "payload": {
                "kind": "message",
                "role": "tool",
                "tool_name": "bash",
                "tool_call_id": "c1",
                "content": [
                    {"text": f"exit code: 0\n--- stdout ---\n{PR}\n\n--- stderr ---\n(empty)"}
                ],
            },
        },
    ]
    with (directory / "transcript.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    return directory


@pytest.mark.asyncio
async def test_a_known_session_with_no_index_answers_an_empty_list(desktop):
    client, root = desktop
    # The session EXISTS but has nothing to scan: a directory and no journal. That is a
    # real state (a conversation that has not run a tool yet) and it must not 404.
    (root / "sessions" / SESSION).mkdir(parents=True)
    response = await client.get(f"/v1/desktop/sessions/{SESSION}/code-requests")
    assert response.status_code == 200
    body = response.json()["result"]
    assert body["rows"] == [] and body["session_id"] == SESSION


@pytest.mark.asyncio
async def test_a_scanned_session_answers_its_rows(desktop):
    client, root = desktop
    session_dir = _seed_session(root)
    ledger.refresh(root, SESSION, session_dir, context=CWD)
    response = await client.get(f"/v1/desktop/sessions/{SESSION}/code-requests")
    assert response.status_code == 200
    body = response.json()["result"]
    assert [row["key"] for row in body["rows"]] == ["github.com/damianvtran/local-operator#1904"]
    row = body["rows"][0]
    assert row["relation"] == "opened" and row["link_only"] is True
    assert row["summary"] is None and row["lanes"] is None
    assert row["evidence"][-1]["rule"] == "gh-pr-create-stdout"
    assert body["revision"] > 0


@pytest.mark.asyncio
async def test_the_first_read_of_an_unscanned_session_does_not_block(desktop, monkeypatch):
    """A cold session answers immediately and kicks a scan; the next read is current."""
    client, root = desktop
    _seed_session(root)
    first = await client.get(f"/v1/desktop/sessions/{SESSION}/code-requests")
    assert first.status_code == 200
    body = first.json()["result"]
    assert body["scan_state"] in ("refreshing", "ready")
    # The scan ran in the background (the route returns without waiting for it): give it a
    # moment, then the rows are there.
    for _ in range(50):
        if ledger.read_index(root, SESSION) is not None:
            break
        await _tick()
    second = await client.get(f"/v1/desktop/sessions/{SESSION}/code-requests")
    assert second.json()["result"]["rows"][0]["number"] == 1904
    assert second.json()["result"]["scan_state"] == "ready"


async def _tick() -> None:
    import asyncio

    await asyncio.sleep(0.02)


@pytest.mark.asyncio
async def test_an_unknown_session_is_a_404(desktop):
    client, _root = desktop
    response = await client.get("/v1/desktop/sessions/ffffffffffff/code-requests")
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "session_not_found"


@pytest.mark.asyncio
async def test_a_malformed_session_id_is_refused_before_the_handler(desktop):
    client, _root = desktop
    assert (await client.get("/v1/desktop/sessions/nope/code-requests")).status_code == 422


@pytest.mark.asyncio
async def test_tool_output_mentions_expand_on_request(desktop):
    client, root = desktop
    session_dir = _seed_session(root)
    with (session_dir / "transcript.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "id": "t2",
                    "ts": time.time() - 30,
                    "type": "message",
                    "payload": {
                        "kind": "message",
                        "role": "tool",
                        "tool_name": "bash",
                        "tool_call_id": "c9",
                        "content": [
                            {
                                "text": (
                                    "exit code: 0\n--- stdout ---\n"
                                    "https://github.com/o/r/pull/5\n"
                                )
                            }
                        ],
                    },
                },
                separators=(",", ":"),
            )
            + "\n"
        )
    ledger.refresh(root, SESSION, session_dir, context=CWD, force=True)
    plain = await client.get(f"/v1/desktop/sessions/{SESSION}/code-requests")
    assert plain.json()["result"]["tool_output_only_count"] == 1
    assert len(plain.json()["result"]["rows"]) == 1
    expanded = await client.get(
        f"/v1/desktop/sessions/{SESSION}/code-requests", params={"include": "mentions_tool"}
    )
    assert len(expanded.json()["result"]["rows"]) == 2


@pytest.mark.asyncio
async def test_refresh_rescans_and_says_what_it_did_not_do(desktop):
    client, root = desktop
    session_dir = _seed_session(root)
    ledger.refresh(root, SESSION, session_dir, context=CWD)
    with (session_dir / "transcript.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "id": "a2",
                    "ts": time.time(),
                    "type": "message",
                    "payload": {
                        "kind": "message",
                        "role": "assistant",
                        "content": [{"text": "another"}],
                        "tool_calls": [
                            {
                                "id": "c2",
                                "name": "bash",
                                "arguments": {"command": "gh pr create --title two"},
                            }
                        ],
                    },
                },
                separators=(",", ":"),
            )
            + "\n"
        )
        handle.write(
            json.dumps(
                {
                    "id": "t3",
                    "ts": time.time(),
                    "type": "message",
                    "payload": {
                        "kind": "message",
                        "role": "tool",
                        "tool_name": "bash",
                        "tool_call_id": "c2",
                        "content": [
                            {
                                "text": (
                                    "exit code: 0\n--- stdout ---\n"
                                    "https://github.com/damianvtran/local-operator/pull/2001\n"
                                )
                            }
                        ],
                    },
                },
                separators=(",", ":"),
            )
            + "\n"
        )
    response = await client.post(
        f"/v1/desktop/sessions/{SESSION}/code-requests/refresh", json={"force": False}
    )
    assert response.status_code == 202
    receipt = response.json()["result"]
    assert receipt["accepted"] is True
    assert "link-only" in receipt["note"] and "later change" in receipt["note"]
    listing = await client.get(f"/v1/desktop/sessions/{SESSION}/code-requests")
    assert {row["number"] for row in listing.json()["result"]["rows"]} == {1904, 2001}


@pytest.mark.asyncio
async def test_refresh_for_an_unknown_session_is_a_404(desktop):
    client, _root = desktop
    response = await client.post("/v1/desktop/sessions/ffffffffffff/code-requests/refresh", json={})
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_refresh_refuses_an_unknown_body_field(desktop):
    client, root = desktop
    _seed_session(root)
    response = await client.post(
        f"/v1/desktop/sessions/{SESSION}/code-requests/refresh", json={"keis": ["x"]}
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_the_scan_uses_the_sessions_recorded_cwd(desktop):
    """F6: a self-hosted URL is confirmed by the CWD's remote, not by the session dir.

    The transcript holds one GitHub-Enterprise-shaped link. With no way to know the host
    the row is detect-and-link and carries a reason; once the session's recorded working
    directory names that host as a remote, the same scan confirms it. The scan must read
    the session's cwd from the wake index — scanning with ``<config>/sessions/<id>`` as
    the cwd made the remote half of host identification impossible to satisfy.
    """
    client, root = desktop
    session_dir = _seed_session(root)
    url = "https://ghe.internal.acme.io/acme/api/pull/9"
    with (session_dir / "transcript.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "id": "a9",
                    "ts": time.time() - 20,
                    "type": "message",
                    "payload": {
                        "kind": "message",
                        "role": "assistant",
                        "content": [{"text": f"tracked in {url}"}],
                    },
                },
                separators=(",", ":"),
            )
            + "\n"
        )

    from local_operator.code_requests.hook import load_context_async
    from local_operator.server.routes.desktop_code_requests import _context_cwd
    from local_operator.wakes import store as wake_store

    async def rows(*, force: bool = False) -> dict[str, Any]:
        # The ROUTE'S OWN cwd choice, which is what F6 is about. ``force`` on the second
        # leg because the scan cache is keyed on the JOURNAL's signature — the cwd is not
        # part of it, so a session that only gained a wake entry in the meantime is
        # otherwise answered from the last scan (which is the state a browser refresh
        # passes ``force`` to escape).
        await ledger.refresh_async(
            root,
            SESSION,
            session_dir,
            context=await load_context_async(_context_cwd(root, SESSION, session_dir)),
            force=force,
        )
        response = await client.get(f"/v1/desktop/sessions/{SESSION}/code-requests")
        # By NUMBER, not by position: the seeded create sorts before a mention.
        return next(row for row in response.json()["result"]["rows"] if row["number"] == 9)

    # 1. No recorded cwd: the host is unknown, so the row is link-only WITH a reason.
    blind = await rows()
    assert blind["number"] == 9 and blind["reason"]

    # 2. The session's own working directory names the host as a remote.
    # A REAL checkout: ``git -C <dir> config`` refuses a hand-written ``.git/config``
    # ("not a git repository"), which is exactly the read host identification makes.
    # ``GIT_CONFIG_SYSTEM=/dev/null`` is hygiene for the whole fleet: the system gitconfig
    # installs the osxkeychain credential helper, and no test on this machine should give
    # a helper a reason to run.
    project = root / "checkouts" / "acme-api"
    project.mkdir(parents=True)
    git_env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(root),
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_TERMINAL_PROMPT": "0",
    }
    for command in (
        ["git", "init", "-q", str(project)],
        [
            "git",
            "-C",
            str(project),
            "remote",
            "add",
            "origin",
            "git@ghe.internal.acme.io:acme/api.git",
        ],
    ):
        subprocess.run(command, check=True, capture_output=True, text=True, env=git_env)
    wake_store.write_entry(
        root,
        SESSION,
        cwd=str(project),
        schedules=[{"kind": "at", "at": int(time.time() * 1000) + 600_000}],
    )
    confirmed = await rows(force=True)
    assert confirmed["number"] == 9
    assert confirmed["reason"] is None
    assert confirmed["host"] == "ghe.internal.acme.io"
