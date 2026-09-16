"""The three write routes: ``POST``/``PATCH``/``DELETE /v1/desktop/wakes``.

The whole point of these routes is WHICH WRITER they reach, so that is what
most of this file pins: a session with no live runtime is mutated through
``wakes/arm.py`` (transcript first, index second), and the one case where that
would be wrong — a runtime that exists but cannot answer — is refused with a
503 rather than written around. A wake appended behind a live session's back is
deleted by that session's next persist *and* skipped by the supervisor in the
meantime, so "the request succeeded" and "the reminder will fire" come apart
silently. That is the failure these tests exist to prevent.

The routes are driven over HTTP against the real app: the wires under test are
the status code, the body shape the desktop client unwraps, and the bytes that
land on disk.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.server.routes import desktop_wakes
from local_operator.wakes.store import read_entry

TOKEN = "desktop-wakes-write-token"


@pytest_asyncio.fixture
async def desktop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    app = FastAPI()
    app.include_router(desktop_wakes.router)
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


def _rows_on_disk(directory: Path) -> list[dict[str, Any]]:
    latest: list[dict[str, Any]] = []
    for line in (directory / "transcript.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        payload = entry.get("payload") or {}
        if payload.get("custom_type") == "wake_schedules":
            latest = list((payload.get("details") or {}).get("schedules") or [])
    return latest


def _request_id(n: int) -> str:
    return f"00000000-0000-4000-8000-{n:012d}"


@pytest.mark.asyncio
async def test_arming_a_cold_session_writes_the_transcript_and_the_index(desktop) -> None:
    """The arm path for an existing conversation with nobody home."""
    client, root = desktop
    directory = _session(root, "aaaaaaaabbbb")

    response = await client.post(
        "/v1/desktop/wakes",
        json={
            "request_id": _request_id(1),
            "session_id": "aaaaaaaabbbb",
            "message": "check the build",
            "in": "30m",
        },
    )

    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["wake_id"] == "w1"
    assert result["created_session"] is False
    assert result["receipt"] == "applied"
    assert result["index_written"] is True
    assert [row["message"] for row in _rows_on_disk(directory)] == ["check the build"]
    entry = read_entry(root, "aaaaaaaabbbb")
    assert entry is not None and entry["schedules"][0]["id"] == "w1"


@pytest.mark.asyncio
async def test_a_new_scheduled_task_is_one_request_and_one_conversation(desktop) -> None:
    """Create+arm: the session exists because the wake needs somewhere to fire,
    and it is NAMED after the prompt so the row is identifiable.

    The name is the harness's honest substitute for seeding a fake opening user
    turn — a synthetic message would replay into the model's context as
    something the user said.
    """
    from local_operator.resume import session_name

    client, root = desktop

    response = await client.post(
        "/v1/desktop/wakes",
        json={
            "request_id": _request_id(2),
            "cwd": str(root),
            "message": "read my unread email and group it by whether it needs a reply",
            "in": "2h",
        },
    )

    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["created_session"] is True
    session_id = result["session_id"]
    directory = root / "sessions" / session_id
    assert directory.is_dir()
    assert (directory / "transcript.jsonl").exists()
    assert session_name(directory).startswith("read my unread email")

    listing = (await client.get("/v1/desktop/wakes")).json()["result"]
    assert [entry["session_id"] for entry in listing["entries"]] == [session_id]


@pytest.mark.asyncio
async def test_a_retried_create_cannot_make_a_second_conversation(desktop) -> None:
    """The receipt, not the client, is what makes create+arm at-most-once: a
    retry after a lost response must answer the first attempt's result."""
    client, root = desktop
    body = {
        "request_id": _request_id(3),
        "cwd": str(root),
        "message": "nightly backup check",
        "in": "1h",
    }

    first = await client.post("/v1/desktop/wakes", json=body)
    second = await client.post("/v1/desktop/wakes", json=body)

    assert first.status_code == 200 and second.status_code == 200
    assert first.json()["result"]["session_id"] == second.json()["result"]["session_id"]
    assert second.json()["result"]["receipt"] == "replayed"
    created = list((root / "sessions").iterdir())
    assert len(created) == 1


@pytest.mark.asyncio
async def test_a_bad_working_directory_is_refused_before_a_session_is_made(desktop) -> None:
    """The same admission ``sessions.create`` applies, and it must leave no
    directory behind — a draft with no wake in it is a phantom row."""
    client, root = desktop

    response = await client.post(
        "/v1/desktop/wakes",
        json={
            "request_id": _request_id(4),
            "cwd": str(root / "does-not-exist"),
            "message": "hi",
            "in": "5m",
        },
    )

    assert response.status_code == 409, response.text
    assert not (root / "sessions").exists() or list((root / "sessions").iterdir()) == []


@pytest.mark.asyncio
async def test_the_sixteenth_arm_is_the_last_one_that_lands(desktop) -> None:
    """The cap sentence is the validator's, so this refusal reads exactly like
    the agent's tool's."""
    client, root = desktop
    _session(root, "aaaaaaaaaaaa")
    for n in range(1, 17):
        response = await client.post(
            "/v1/desktop/wakes",
            json={
                "request_id": _request_id(100 + n),
                "session_id": "aaaaaaaaaaaa",
                "message": f"wake {n}",
                "in": "30m",
            },
        )
        assert response.status_code == 200, response.text

    refused = await client.post(
        "/v1/desktop/wakes",
        json={
            "request_id": _request_id(200),
            "session_id": "aaaaaaaaaaaa",
            "message": "one too many",
            "in": "30m",
        },
    )

    assert refused.status_code == 409
    assert "16" in json.dumps(refused.json())
    entry = read_entry(root, "aaaaaaaaaaaa")
    assert entry is not None and len(entry["schedules"]) == 16


@pytest.mark.asyncio
async def test_a_malformed_body_and_an_unknown_session_are_refused_by_kind(desktop) -> None:
    """422 for input this server cannot read, 404 for a session that is not
    there — the split the desktop client keys on, and the reason the two are
    not both "the request was bad"."""
    client, _root = desktop
    _session(_root, "aaaaaaaaaaaa")

    bad_duration = await client.post(
        "/v1/desktop/wakes",
        json={
            "request_id": _request_id(5),
            "session_id": "aaaaaaaaaaaa",
            "message": "hi",
            "in": "banana",
        },
    )
    bad_bound = await client.post(
        "/v1/desktop/wakes",
        json={
            "request_id": _request_id(6),
            "session_id": "aaaaaaaaaaaa",
            "message": "hi",
            "in": "5m",
            "limit": 3,
        },
    )
    unknown = await client.post(
        "/v1/desktop/wakes",
        json={
            "request_id": _request_id(7),
            "session_id": "nosuchsession",
            "message": "hi",
            "in": "5m",
        },
    )
    mixed = await client.post(
        "/v1/desktop/wakes",
        json={
            "request_id": _request_id(8),
            "session_id": "aaaaaaaaaaaa",
            "cwd": str(_root),
            "message": "hi",
            "in": "5m",
        },
    )

    assert bad_duration.status_code == 422, bad_duration.text
    assert bad_bound.status_code == 422, bad_bound.text
    assert unknown.status_code == 404, unknown.text
    assert mixed.status_code == 422, mixed.text


@pytest.mark.asyncio
async def test_a_patch_rewords_a_wake_and_keeps_its_handle(desktop) -> None:
    client, root = desktop
    directory = _session(root, "aaaaaaaaaaaa")
    await client.post(
        "/v1/desktop/wakes",
        json={
            "request_id": _request_id(9),
            "session_id": "aaaaaaaaaaaa",
            "message": "first wording",
            "in": "30m",
        },
    )

    response = await client.patch(
        "/v1/desktop/wakes/aaaaaaaaaaaa/w1",
        json={"message": "second wording"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["result"]["wake_id"] == "w1"
    rows = _rows_on_disk(directory)
    assert rows[0]["message"] == "second wording"


@pytest.mark.asyncio
async def test_cancelling_the_last_wake_removes_the_row_from_the_listing(desktop) -> None:
    client, root = desktop
    _session(root, "aaaaaaaaaaaa")
    await client.post(
        "/v1/desktop/wakes",
        json={
            "request_id": _request_id(10),
            "session_id": "aaaaaaaaaaaa",
            "message": "only one",
            "in": "30m",
        },
    )
    assert (await client.get("/v1/desktop/wakes")).json()["result"]["total"] == 1

    response = await client.delete("/v1/desktop/wakes/aaaaaaaaaaaa/w1")

    assert response.status_code == 200, response.text
    assert response.json()["result"]["next_due_at"] is None
    assert not (root / "wakes" / "aaaaaaaaaaaa.json").exists()
    assert (await client.get("/v1/desktop/wakes")).json()["result"]["total"] == 0


@pytest.mark.asyncio
async def test_the_two_write_routes_refuse_an_unknown_handle(desktop) -> None:
    client, root = desktop
    _session(root, "aaaaaaaaaaaa")
    await client.post(
        "/v1/desktop/wakes",
        json={
            "request_id": _request_id(11),
            "session_id": "aaaaaaaaaaaa",
            "message": "only one",
            "in": "30m",
        },
    )

    patched = await client.patch("/v1/desktop/wakes/aaaaaaaaaaaa/w9", json={"message": "nope"})
    deleted = await client.delete("/v1/desktop/wakes/aaaaaaaaaaaa/w9")
    malformed_id = await client.delete("/v1/desktop/wakes/aaaaaaaaaaaa/banana")

    assert patched.status_code == 404, patched.text
    assert deleted.status_code == 404, deleted.text
    # The handle's shape is refused by the path pattern, before any handler runs.
    assert malformed_id.status_code == 422, malformed_id.text


@pytest.mark.asyncio
async def test_a_wedged_owner_is_refused_rather_than_written_around(
    desktop, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The last silent dead-end. A wedged runtime holds the transcript lease,
    so a wake written here could neither fire nor survive — and its process is
    alive, which is exactly why "no owner" is the wrong reading of it."""
    import local_operator.wakes.supervisor as supervisor_module

    client, root = desktop
    _session(root, "aaaaaaaaaaaa")
    monkeypatch.setattr(supervisor_module, "wedged_runtime", lambda *a, **k: (4321, 99.5))

    response = await client.post(
        "/v1/desktop/wakes",
        json={
            "request_id": _request_id(12),
            "session_id": "aaaaaaaaaaaa",
            "message": "will not land",
            "in": "30m",
        },
    )

    assert response.status_code == 503, response.text
    assert "not responding" in json.dumps(response.json())
    assert read_entry(root, "aaaaaaaaaaaa") is None
    assert _rows_on_disk(root / "sessions" / "aaaaaaaaaaaa") == []


# ---------------------------------------------------------------------------
# The owner path
# ---------------------------------------------------------------------------
#
# Driven through ``_via_owner`` with a stub bridge rather than over HTTP: a
# genuinely reachable owner needs a live runtime on a control socket, which is
# what QA drives end to end against the assembled application. What is worth
# pinning here is the CONTRACT this module owes the runtime — the command word,
# the payload it lands in the ladder with, and the mapping from the runtime's
# typed refusal back to an HTTP status.


class _StubRemote:
    def __init__(self, outcome: dict[str, Any]) -> None:
        self.outcome = outcome
        self.calls: list[tuple[str, str]] = []

    async def route_shared_slash(self, command: str, args: str) -> dict[str, Any]:
        self.calls.append((command, args))
        return self.outcome


class _StubBridge:
    def __init__(self, outcome: dict[str, Any]) -> None:
        self.remote = _StubRemote(outcome)


@pytest.mark.asyncio
async def test_the_owner_path_sends_the_ladder_word_and_reports_the_file(tmp_path: Path) -> None:
    """The route never writes: it asks the owner, and then VERIFIES the derived
    index rather than reporting the owner's intent (the session's own index
    write is best-effort and swallows its failure)."""
    root = tmp_path / "cfg"
    (root / "wakes").mkdir(parents=True)
    (root / "wakes" / "aaaaaaaaaaaa.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "session_id": "aaaaaaaaaaaa",
                "cwd": "/work/here",
                "updated_at": 1,
                "schedules": [{"id": "w1", "message": "m", "next_due_at": 42}],
            }
        ),
        encoding="utf-8",
    )
    bridge = _StubBridge(
        {
            "kind": "notice",
            "text": "Scheduled wake 'w1'",
            "data": {"wake_id": "w1", "next_due_at": 42},
        }
    )

    receipt = await desktop_wakes._via_owner(
        root,
        bridge,
        "aaaaaaaaaaaa",
        "create",
        "",
        {"message": "check the build", "in": "30m"},
    )

    command, args = bridge.remote.calls[0]
    assert command == "wake"
    assert json.loads(args) == {
        "op": "create",
        "wake_id": "",
        "request": {"message": "check the build", "in": "30m"},
    }
    assert receipt["wake_id"] == "w1"
    assert receipt["next_due_at"] == 42
    assert receipt["index_written"] is True
    assert receipt["created_session"] is False


@pytest.mark.asyncio
async def test_an_owner_refusal_keeps_the_runtime_s_own_sentence(tmp_path: Path) -> None:
    """One wording for one mistake: the sentence the agent's tool gets is the
    sentence the desktop client is shown, and the code — not the prose — is what
    decides the status."""
    root = tmp_path / "cfg"
    root.mkdir(parents=True)
    bridge = _StubBridge(
        {
            "kind": "error",
            "text": "at most 16 wake schedules are allowed.",
            "data": {"code": "wake_invalid"},
        }
    )

    with pytest.raises(HTTPException) as refused:
        await desktop_wakes._via_owner(
            root, bridge, "aaaaaaaaaaaa", "create", "", {"message": "x", "in": "5m"}
        )

    assert refused.value.status_code == 422
    assert refused.value.detail == {
        "code": "wake_invalid",
        "message": "at most 16 wake schedules are allowed.",
    }


@pytest.mark.asyncio
async def test_the_body_projection_keeps_only_the_scheduling_fields() -> None:
    """What the route hands the SHARED validator, which is closed.

    The create body is a subclass carrying ``request_id``/``session_id``/``cwd``/
    ``target``/``title``, and ``exclude_unset=True`` keeps every one of those the
    request actually set — so the projection is what stops this route's own
    fields reaching a validator that refuses them. Driven through a real runtime
    it was a 422 on a perfectly good schedule; the cold path hid it, because
    ``build_wake_schedule`` reads only the keys it knows.
    """
    body = desktop_wakes.WakeCreate.model_validate(
        {
            "request_id": _request_id(4242),
            "session_id": "aaaaaaaaaaaa",
            "message": "check the build",
            "in": "30m",
            "every": "1h",
            "limit": 3,
        }
    )

    assert desktop_wakes._wake_request(body) == {
        "message": "check the build",
        "in": "30m",
        "every": "1h",
        "limit": 3,
    }

    # A CREATE with the fields omitted reports them omitted, not as nulls: the
    # runtime's validator treats "absent" and "present and null" differently.
    minimal = desktop_wakes.WakeCreate.model_validate(
        {
            "request_id": _request_id(4243),
            "cwd": "/tmp",
            "message": "hi",
            "at": "+45m",
        }
    )
    assert desktop_wakes._wake_request(minimal) == {"message": "hi", "at": "+45m"}


@pytest.mark.asyncio
async def test_an_explicit_title_names_the_conversation_it_arms(desktop) -> None:
    """``title`` has a meaning on BOTH shapes of the body, which is why it is
    not silently ignored on the arm-an-existing one: a field the wire accepts
    and the server drops is a field the client will believe took effect.
    """
    from local_operator.resume import session_name

    client, root = desktop
    directory = _session(root, "aaaaaaaaaaaa")

    await client.post(
        "/v1/desktop/wakes",
        json={
            "request_id": _request_id(13),
            "session_id": "aaaaaaaaaaaa",
            "message": "check the backup",
            "in": "30m",
            "title": "Nightly backup watch",
        },
    )
    named = await client.post(
        "/v1/desktop/wakes",
        json={
            "request_id": _request_id(14),
            "session_id": "aaaaaaaaaaaa",
            "message": "this prompt must NOT become the name",
            "in": "45m",
        },
    )

    assert named.status_code == 200, named.text
    assert session_name(directory) == "Nightly backup watch"


# ---------------------------------------------------------------------------
# The create+arm shape's refusals, the recorded outcome, and concurrency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("index", "wake_fields", "status", "code"),
    [
        (40, {"message": "boom", "in": "banana"}, 422, "wake_invalid"),
        (41, {"message": "boom", "in": "10m", "every": "30s"}, 422, "wake_invalid"),
        (42, {"message": "boom", "at": "2001-01-01T00:00:00+00:00"}, 409, "wake_refused"),
        (43, {"message": "boom", "in": "10m", "limit": 3}, 422, "wake_invalid"),
        (44, {"message": "x" * 2100, "in": "10m"}, 422, "wake_invalid"),
    ],
)
async def test_a_refused_schedule_on_the_create_shape_is_a_typed_refusal(
    desktop, index: int, wake_fields: dict[str, Any], status: int, code: str
) -> None:
    """R1/Q1: the shape the create dialog uses must answer the SAME refusals the
    named-session shape does.

    It used to call the writer directly and re-raise, so this branch answered
    500 for every one of these bodies while the identical schedule with
    ``session_id`` answered 422/409 carrying the validator's sentence — and it
    is the "New scheduled task" flow, i.e. the call the whole feature exists
    for. Five refusal kinds, one per row, because the defect was the branch and
    not one input.
    """
    client, root = desktop

    response = await client.post(
        "/v1/desktop/wakes",
        json={"request_id": _request_id(index), "cwd": str(root), **wake_fields},
    )

    assert response.status_code == status, response.text
    detail = response.json()["detail"]
    assert detail["code"] == code
    assert detail["message"], "the validator's sentence travels with the code"
    # ROLLED BACK, not left behind: a draft with no wake in it is a phantom
    # conversation in the sidebar.
    assert not (root / "sessions").exists() or list((root / "sessions").iterdir()) == []


@pytest.mark.asyncio
async def test_a_refused_create_replays_its_refusal_instead_of_indeterminate(desktop) -> None:
    """The receipt records the REFUSAL, so a retry is actionable.

    ``receipts().run`` stores what the operation returns; a raised refusal left
    the row NULL, and the retry of the same ``request_id`` met the journal's
    "outcome is indeterminate" — a dead end for a user who mistyped a duration.
    Same id, same body, same answer.
    """
    client, root = desktop
    body = {
        "request_id": _request_id(45),
        "cwd": str(root),
        "message": "boom",
        "in": "banana",
    }

    first = await client.post("/v1/desktop/wakes", json=body)
    second = await client.post("/v1/desktop/wakes", json=body)

    assert first.status_code == 422, first.text
    assert second.status_code == 422, second.text
    assert second.json()["detail"] == first.json()["detail"]
    assert "indeterminate" not in json.dumps(second.json())


@pytest.mark.asyncio
async def test_the_created_id_comes_back_when_the_rollback_declines(
    desktop, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one case the design's "retry the arm against it" is about.

    The rollback removes the draft only while it is provably untouched; when
    something has adopted it, the directory stays and the refusal has to name
    it, or the caller is left holding an error and no way to reach the
    conversation the request created.
    """
    import local_operator.server.routes.desktop_wakes as route

    client, root = desktop

    async def arm_and_adopt(config_dir, session_id, request, *, cwd=None, now_ms=None):
        # Something wrote into the draft between the create and the arm, so the
        # rollback's identity proof fails by design — then the arm refuses.
        (Path(config_dir) / "sessions" / session_id / "transcript.jsonl").write_text(
            '{"id":"x","ts":1.0,"type":"message","payload":{}}\n', encoding="utf-8"
        )
        raise route.WakeWriteError(
            "at most 16 wake schedules are allowed.", status=409, code="wake_refused"
        )

    monkeypatch.setattr(route, "arm_wake", arm_and_adopt)

    response = await client.post(
        "/v1/desktop/wakes",
        json={
            "request_id": _request_id(46),
            "cwd": str(root),
            "message": "boom",
            "in": "10m",
        },
    )

    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "wake_refused"
    assert detail["session_id"]
    assert (root / "sessions" / detail["session_id"]).is_dir()


@pytest.mark.asyncio
async def test_an_owner_appearing_at_the_append_is_refused_through_the_route(
    desktop, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R2/Q3 end to end through the route: the owner check that matters is the
    one the WRITER makes, because the route's is a moment old by then."""
    import local_operator.wakes.supervisor as supervisor

    client, root = desktop
    session_dir = _session(root, "aaaaaaaaaaaa")

    async def live(config_dir, session_id):
        return True

    monkeypatch.setattr(supervisor, "_has_live_runtime", live)

    response = await client.post(
        "/v1/desktop/wakes",
        json={
            "request_id": _request_id(47),
            "session_id": "aaaaaaaaaaaa",
            "message": "raced wake",
            "in": "30m",
        },
    )

    assert response.status_code == 503, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "wake_owner_present"
    assert "Retry in a moment" in detail["message"]
    assert _rows_on_disk(session_dir) == []
    assert read_entry(root, "aaaaaaaaaaaa") is None


@pytest.mark.asyncio
async def test_concurrent_arms_through_the_route_leave_one_row_each(
    desktop, monkeypatch: pytest.MonkeyPatch
) -> None:
    """QA's Q2 cell at the HTTP level: five simultaneous arms of one cold
    session, five 200s, five rows, five distinct handles — never six rows and
    never a 409 for a write that landed.

    The read is slowed so the race is a deterministic catch rather than a
    probabilistic one (the same magnifier a large transcript provided when this
    was found): without the per-session lock all five requests merge onto one
    base and the assertions below collapse to a single row.
    """
    import local_operator.wakes.arm as arm_module
    from local_operator.harness.wake import WakeSchedule

    client, root = desktop
    session_dir = _session(root, "aaaaaaaaaaaa")
    original_read = arm_module._read_rows

    def slow_read(directory: Path) -> list[WakeSchedule]:
        time.sleep(0.02)
        return original_read(directory)

    monkeypatch.setattr(arm_module, "_read_rows", slow_read)

    responses = await asyncio.gather(
        *[
            client.post(
                "/v1/desktop/wakes",
                json={
                    "request_id": _request_id(50 + index),
                    "session_id": "aaaaaaaaaaaa",
                    "message": f"w{index}",
                    "in": f"{index + 1}0m",
                },
            )
            for index in range(5)
        ]
    )

    assert [response.status_code for response in responses] == [200] * 5, [
        response.text for response in responses
    ]
    handles = [response.json()["result"]["wake_id"] for response in responses]
    rows = _rows_on_disk(session_dir)
    assert len(rows) == 5, "one row per request, and none dropped"
    assert len(set(handles)) == 5, "no handle handed out twice"
    assert sorted(row["id"] for row in rows) == sorted(handles)
    entry = read_entry(root, "aaaaaaaaaaaa")
    assert entry is not None and len(entry["schedules"]) == 5
