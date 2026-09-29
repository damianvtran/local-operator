"""``POST``/``DELETE /v1/desktop/monitors``: arming and cancelling watches.

The whole point of these routes is WHICH WRITER they reach, so that is what
most of this file pins: a session with no live runtime is mutated through
``monitors/arm.py`` (transcript first, index second), and the one case where
that would be wrong — a runtime that exists and answers (or that is wedged) —
is refused with a 503 rather than written around. A monitor appended behind a
live session's back is deleted by that session's next persist *and* skipped by
nothing until then, so "the request succeeded" and "the watch will tick" come
apart silently. That is the failure these tests exist to prevent.

The retry question is pinned here too, because these routes solve it
structurally: the dedupe identity (tool + canonical arguments) answers a
repeated arm with ``already_armed`` and never appends a second row, so there is
no request journal to replay.

The routes are driven over HTTP against the real app: the wires under test are
the status code, the body shape the desktop client unwraps, and the bytes that
land on disk.
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.monitors.store import read_entry
from local_operator.server.routes import desktop_monitors

TOKEN = "desktop-monitors-write-token"


@pytest_asyncio.fixture
async def desktop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    app = FastAPI()
    app.include_router(desktop_monitors.router)
    app.state.config_manager = ConfigManager(tmp_path)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        yield client, tmp_path


def _session(root: Path, session_id: str, *, transcript: bool = True) -> Path:
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    if transcript:
        (directory / "transcript.jsonl").write_text(
            json.dumps(
                {
                    "id": "m1",
                    "ts": 1.0,
                    "type": "message",
                    "payload": {
                        "kind": "message",
                        "role": "user",
                        "content": [{"type": "text", "text": "hello"}],
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
    return directory


def _plant(root: Path, session_id: str, rows: list[dict[str, Any]], *, next_seq: int) -> Path:
    """A session with monitors already in its transcript, written directly.

    The write under test must treat the TRANSCRIPT as the base, so the fixture
    writes the transcript (never the index) and lets the writer derive the rest.
    """
    directory = _session(root, session_id)
    (directory / "transcript.jsonl").write_text(
        (directory / "transcript.jsonl").read_text(encoding="utf-8")
        + json.dumps(
            {
                "id": "monitor-entry-1",
                "ts": 2.0,
                "type": "custom",
                "payload": {
                    "custom_type": "monitor_schedules",
                    "details": {"monitors": rows, "next_seq": next_seq},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return directory


def _row(monitor_id: str, **extra: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": monitor_id,
        "name": f"watch {monitor_id}",
        "tool": "bash",
        "arguments": {"command": f"ls {monitor_id}"},
        "every_ms": 60_000,
        "created_at": 1_700_000_000_000,
    }
    row.update(extra)
    return row


def _rows_on_disk(directory: Path) -> list[dict[str, Any]]:
    latest: list[dict[str, Any]] = []
    for line in (directory / "transcript.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        payload = entry.get("payload") or {}
        if payload.get("custom_type") == "monitor_schedules":
            latest = list((payload.get("details") or {}).get("monitors") or [])
    return latest


def _snapshot_count(directory: Path) -> int:
    """How many ``monitor_schedules`` entries the transcript holds — the write log.

    Needed where the QUESTION is "did the request write?", which the rows alone
    cannot answer: re-running the same mutation against the same base writes an
    identical list under a new entry.
    """
    return sum(
        1
        for line in (directory / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
        and (json.loads(line).get("payload") or {}).get("custom_type") == "monitor_schedules"
    )


def _arm_body(session_id: str, **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "session_id": session_id,
        "name": "watch the build",
        "tool": "bash",
        "arguments": {"command": "ls"},
        "every": "60s",
    }
    body.update(extra)
    return body


@pytest.mark.asyncio
async def test_arming_a_cold_session_writes_the_transcript_and_the_index(desktop) -> None:
    """The arm path for an existing conversation with nobody home."""
    client, root = desktop
    directory = _session(root, "aaaaaaaabbbb")

    response = await client.post("/v1/desktop/monitors", json=_arm_body("aaaaaaaabbbb"))

    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["monitor_id"] == "m1"
    assert result["name"] == "watch the build"
    assert result["receipt"] == "applied"
    assert result["index_written"] is True
    assert result["already_armed"] is False
    assert result["reactivated"] is False
    assert result["remaining"] == 1
    assert result["next_due_at"] is not None and result["next_due_at"] > 0
    rows = _rows_on_disk(directory)
    assert [(row["id"], row["tool"], row["arguments"]) for row in rows] == [
        ("m1", "bash", {"command": "ls"})
    ]
    entry = read_entry(root, "aaaaaaaabbbb")
    assert entry is not None
    assert entry["monitors"][0]["id"] == "m1"
    # The derived index carries the counters/health the receipt reports, so a
    # cold reader (the CLI, the sidebar) sees the due instant the arm just set.
    assert entry["monitors"][0]["next_due_at"] == result["next_due_at"]
    assert entry["monitors"][0]["disabled"] is False


@pytest.mark.asyncio
async def test_a_retried_arm_is_answered_by_the_dedupe_not_a_second_row(desktop) -> None:
    """The identity IS the idempotency key: the second identical request finds
    the spec in force and writes nothing at all."""
    client, root = desktop
    directory = _session(root, "aaaaaaaabbbb")

    first = await client.post("/v1/desktop/monitors", json=_arm_body("aaaaaaaabbbb"))
    assert first.status_code == 200, first.text
    writes = _snapshot_count(directory)

    second = await client.post("/v1/desktop/monitors", json=_arm_body("aaaaaaaabbbb"))

    assert second.status_code == 200, second.text
    result = second.json()["result"]
    assert result["monitor_id"] == "m1"
    assert result["already_armed"] is True
    assert result["reactivated"] is False
    assert result["remaining"] == 1
    assert _snapshot_count(directory) == writes, "the retry appended a second row"
    assert [row["id"] for row in _rows_on_disk(directory)] == ["m1"]


@pytest.mark.asyncio
async def test_a_disabled_spec_is_reactivated_rather_than_duplicated(desktop) -> None:
    """§11.3's "re-arm to reactivate": the same spec resets the failure state,
    keeps the row's id and name, and does not move the transcript — the list did
    not change; the counters file is what did."""
    from local_operator.monitors import state as monitor_state

    client, root = desktop
    directory = _plant(
        root,
        "aaaaaaaabbbb",
        [_row("m1", tool="bash", arguments={"command": "ls"}, name="watch the build")],
        next_seq=2,
    )
    monitor_state.write_counters(
        root,
        "aaaaaaaabbbb",
        "m1",
        {"schema": 1, "monitor_id": "m1", "disabled": True, "disabled_reason": "5 failed"},
    )
    writes = _snapshot_count(directory)

    response = await client.post("/v1/desktop/monitors", json=_arm_body("aaaaaaaabbbb"))

    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["reactivated"] is True
    assert result["already_armed"] is False
    assert result["monitor_id"] == "m1"
    counters = monitor_state.read_counters(root, "aaaaaaaabbbb", "m1")
    assert counters is not None
    assert counters["disabled"] is False
    assert counters["next_due_at"] == result["next_due_at"]
    assert _snapshot_count(directory) == writes, "reactivation moved the transcript"
    entry = read_entry(root, "aaaaaaaabbbb")
    assert entry is not None
    assert entry["monitors"][0]["disabled"] is False
    assert entry["monitors"][0]["next_due_at"] == result["next_due_at"]


@pytest.mark.asyncio
async def test_the_ninth_arm_is_refused_with_the_cap_s_own_sentence(desktop) -> None:
    """The cap is a rejection with a sentence naming it, never a silent drop —
    and a refusal leaves the write log standing still."""
    client, root = desktop
    directory = _plant(root, "aaaaaaaabbbb", [_row(f"m{i}") for i in range(1, 9)], next_seq=9)
    writes = _snapshot_count(directory)

    response = await client.post(
        "/v1/desktop/monitors", json=_arm_body("aaaaaaaabbbb", arguments={"command": "ls new"})
    )

    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "monitor_refused"
    assert "monitor limit reached (8 per session)" in detail["message"]
    assert _snapshot_count(directory) == writes


@pytest.mark.asyncio
async def test_a_malformed_body_and_unsupportable_targets_are_refused_by_kind(desktop) -> None:
    """422 for input this server cannot read, 409 for a call the feature
    declines to watch, 404 for a session that is not there — the split the
    desktop client keys on, and the reason they are not all "the request was
    bad"."""
    client, root = desktop
    _session(root, "aaaaaaaaaaaa")

    bad_duration = await client.post(
        "/v1/desktop/monitors", json=_arm_body("aaaaaaaaaaaa", every="banana")
    )
    bad_regex = await client.post(
        "/v1/desktop/monitors", json=_arm_body("aaaaaaaaaaaa", ignore=["("])
    )
    write_target = await client.post(
        "/v1/desktop/monitors",
        json=_arm_body("aaaaaaaaaaaa", tool="write", arguments={"path": "x"}),
    )
    past_until = await client.post(
        "/v1/desktop/monitors", json=_arm_body("aaaaaaaaaaaa", until="2020-01-01T00:00:00")
    )
    unknown = await client.post("/v1/desktop/monitors", json=_arm_body("nosuchsession"))

    assert bad_duration.status_code == 422, bad_duration.text
    assert bad_duration.json()["detail"]["code"] == "monitor_invalid"
    assert bad_regex.status_code == 422, bad_regex.text
    assert bad_regex.json()["detail"]["code"] == "monitor_invalid"
    # The read-only gate is NOT malformed input: it is a well-formed request for
    # something monitors must not re-run unattended (§6.7's fail-loudly rule).
    assert write_target.status_code == 409, write_target.text
    assert write_target.json()["detail"]["code"] == "monitor_refused"
    assert '"write"' in write_target.json()["detail"]["message"]
    assert past_until.status_code == 409, past_until.text
    assert past_until.json()["detail"]["code"] == "monitor_refused"
    # The host's own lookup answers first for a session nobody can locate (a
    # string detail, the shape every desktop route's unknown-id refusal has),
    # so this asserts the status the client keys on and not an envelope shape
    # this path never produces — the writer's own ``session_not_found`` envelope
    # is for the race where a directory vanishes between the two.
    assert unknown.status_code == 404, unknown.text


@pytest.mark.asyncio
async def test_a_wedged_owner_is_refused_rather_than_written_around(
    desktop, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wedged runtime holds the transcript lease, so a monitor written here
    could neither tick nor survive — and its process is alive, which is exactly
    why "no owner" is the wrong reading of it. Both ops refuse."""
    import local_operator.wakes.supervisor as supervisor_module

    client, root = desktop
    directory = _session(root, "aaaaaaaaaaaa")
    monkeypatch.setattr(supervisor_module, "wedged_runtime", lambda *a, **k: (4321, 99.5))

    arm = await client.post("/v1/desktop/monitors", json=_arm_body("aaaaaaaaaaaa"))
    cancel = await client.delete("/v1/desktop/monitors/aaaaaaaaaaaa/m1")

    assert arm.status_code == 503, arm.text
    assert arm.json()["detail"]["code"] == "monitor_owner_wedged"
    assert "not responding" in json.dumps(arm.json())
    assert cancel.status_code == 503, cancel.text
    assert read_entry(root, "aaaaaaaaaaaa") is None
    assert _rows_on_disk(directory) == []


class _StubRemote:
    owner_reachable = True


class _StubBridge:
    remote = _StubRemote()


class _StubPool:
    """A host whose session owner is reachable — the state a live desktop
    conversation is in. Monitors have no routed command ladder, so both ops
    must refuse rather than fall through to the file writer."""

    @contextlib.asynccontextmanager
    async def session(self, session_id: str):
        yield _StubBridge()


@pytest.mark.asyncio
async def test_an_answering_owner_is_refused_because_no_monitor_ladder_exists(
    desktop, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The deliberate gap of this slice, pinned as behaviour: ``monitors`` has
    no ``_wake_slash`` sibling, so a conversation whose runtime answers gets the
    writer's own retryable sentence — never a file write that its next persist
    would delete."""
    client, root = desktop
    directory = _session(root, "aaaaaaaaaaaa")
    monkeypatch.setattr(desktop_monitors, "host", lambda request: _StubPool())

    arm = await client.post("/v1/desktop/monitors", json=_arm_body("aaaaaaaaaaaa"))
    cancel = await client.delete("/v1/desktop/monitors/aaaaaaaaaaaa/m1")

    assert arm.status_code == 503, arm.text
    assert arm.json()["detail"]["code"] == "monitor_owner_present"
    assert "open in a running session" in arm.json()["detail"]["message"]
    assert "Nothing was written" in arm.json()["detail"]["message"]
    assert cancel.status_code == 503, cancel.text
    assert read_entry(root, "aaaaaaaaaaaa") is None
    assert _rows_on_disk(directory) == []


@pytest.mark.asyncio
async def test_a_live_pid_with_no_record_is_refused_by_the_writer_s_guard(desktop) -> None:
    """The review-round-2 state the wake writer had to be fixed for, pinned for
    monitors: a live ``.session.pid`` with NO discovery record used to answer
    200 and write behind a live process. The route still takes its cold path
    here (no dialable owner), so this refusal is the WRITER's guard firing."""
    client, root = desktop
    directory = _session(root, "aaaaaaaaaaaa")
    (directory / ".session.pid").write_text(str(os.getpid()), encoding="utf-8")

    response = await client.post("/v1/desktop/monitors", json=_arm_body("aaaaaaaaaaaa"))

    assert response.status_code == 503, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "monitor_owner_present"
    assert "does not answer as a runtime" in detail["message"]
    assert f"process {os.getpid()}" in detail["message"]
    assert "Nothing was written" in detail["message"]
    assert _rows_on_disk(directory) == []
    assert read_entry(root, "aaaaaaaaaaaa") is None


@pytest.mark.asyncio
async def test_cancelling_one_monitor_rewrites_the_transcript_and_the_index(desktop) -> None:
    """The cancel path, end to end: transcript first, the index entry removed
    with the last row (which is also what releases the cleanup reap guard)."""
    client, root = desktop
    directory = _session(root, "aaaaaaaabbbb")
    arm = await client.post("/v1/desktop/monitors", json=_arm_body("aaaaaaaabbbb"))
    assert arm.status_code == 200, arm.text

    response = await client.delete("/v1/desktop/monitors/aaaaaaaabbbb/m1")

    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["monitor_id"] == "m1"
    assert result["remaining"] == 0
    assert result["next_due_at"] is None
    assert result["index_written"] is True
    assert _rows_on_disk(directory) == []
    assert read_entry(root, "aaaaaaaabbbb") is None

    # And the repeated cancel is the honest "already gone" refusal.
    again = await client.delete("/v1/desktop/monitors/aaaaaaaabbbb/m1")
    assert again.status_code == 404, again.text
    assert again.json()["detail"]["code"] == "monitor_not_found"


@pytest.mark.asyncio
async def test_cancelling_one_of_two_keeps_the_other(desktop) -> None:
    """The remainder is the NEW list, and the sibling's row survives with it —
    the transcript and the index both move off the same rows."""
    client, root = desktop
    directory = _plant(root, "aaaaaaaabbbb", [_row("m1"), _row("m2")], next_seq=3)

    response = await client.delete("/v1/desktop/monitors/aaaaaaaabbbb/m1")

    assert response.status_code == 200, response.text
    assert response.json()["result"]["remaining"] == 1
    assert [row["id"] for row in _rows_on_disk(directory)] == ["m2"]
    entry = read_entry(root, "aaaaaaaabbbb")
    assert entry is not None and [row["id"] for row in entry["monitors"]] == ["m2"]


# ---------------------------------------------------------------------------
# The desk payload
# ---------------------------------------------------------------------------


def test_the_frontend_state_monitors_field_rides_the_wire_payload() -> None:
    """Close the design §12 row's loop on THIS repo's side: the field is not
    only on the store — the snapshot a desktop attaches with is a full dump of
    it (``sync_wire_payload``), so nothing server-side filters ``monitors`` out
    between the runtime and the renderer."""
    from local_operator.session.frontend_state import (
        FrontendSessionState,
        FrontendStateStore,
        MonitorState,
        sync_wire_payload,
    )

    store = FrontendStateStore(FrontendSessionState(session_id="s1", epoch="owner-a", cwd="/repo"))
    store.mutate(monitors=[MonitorState(id="m1", name="watch the build", next_due_at=123)])

    payload = sync_wire_payload(store.subscribe(lambda _u: None).sync)

    rows = payload["snapshot"]["monitors"]
    assert [row["id"] for row in rows] == ["m1"]
    assert rows[0]["name"] == "watch the build"
    assert rows[0]["next_due_at"] == 123
