"""The desktop child reader's LIVE half: the opt-in, and what it may show.

Three properties carry this change, and each has cells here that fail on the
obvious wrong implementation:

* an UNWATCHED connection's `frontend.update` frame is what it always was — the
  trajectory pair empty — so opting in cannot become an implicit broadcast to
  every window of a session;
* the subscription is REFCOUNTED per job on the session's shared bridge, so one
  window closing the reader cannot freeze another window's page (a failure that
  is invisible on screen: the frozen reader's existing rows all stay correct);
* the op is contained — a job id is proved to be one of THIS conversation's own
  child jobs before it can be subscribed — and its "nothing to follow" answers
  are distinct, because a reader must retry on one and stop on the other.
"""

import json
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.harness.jobs import TRAJECTORY_SEQ_KEY
from local_operator.resume import ORIGIN_FORK, ORIGIN_SUBAGENT, mark_session_origin
from local_operator.server.routes import capabilities, desktop_sessions
from local_operator.server.utils.desktop_sessions import (
    DesktopSessionBridge,
    DesktopSessions,
    SubagentChildUnavailable,
)
from local_operator.session.frontend_state import (
    FrontendSessionState,
    FrontendUpdate,
    JobState,
)

PARENT_ID = "0123456789ab"
CHILD_ID = "abcdef012345"
JOB_ID = "a1b2c3d4e5f6"
OTHER_JOB_ID = "0f0e0d0c0b0a"
TOKEN = "child-trajectory-token"


def a_parent_and_child(
    root: Path, *, child_origin: str | None = ORIGIN_SUBAGENT
) -> tuple[Path, Path]:
    """A parent conversation directory plus one child directory on disk.

    The child's origin marker is written by the PRODUCTION writer
    (`mark_session_origin`), because that marker is a containment fact this
    route reads: a hand-rolled copy in the fixture could drift from the one the
    launcher writes and every refusal cell would pass vacuously.
    """
    sessions = root / "sessions"
    parent_dir = sessions / PARENT_ID
    parent_dir.mkdir(parents=True, exist_ok=True)
    (parent_dir / "desktop.json").write_text(json.dumps({"version": 1, "cwd": str(root)}))
    child_dir = sessions / CHILD_ID
    child_dir.mkdir(parents=True, exist_ok=True)
    if child_origin is not None:
        mark_session_origin(child_dir, child_origin, label="reviewer")
    return parent_dir, child_dir


def name_child(parent_dir: Path, session_dir: Path, *, job_id: str = JOB_ID) -> None:
    """Record one child on the parent's PERSISTED roster, through the real writer.

    `_write_roster_sidecar` is what `Session._persist_subagent_roster` calls on
    every roster move, so the proof is exercised against the store shape a
    runtime actually writes. The record carries `session_dir` and `job_id` and
    no `session_id`, which is the shape the job-keyed fallback must handle.
    """
    from local_operator.session.session import (
        SUBAGENT_ROSTER_SIDECAR,
        _write_roster_sidecar,
    )

    _write_roster_sidecar(
        parent_dir / SUBAGENT_ROSTER_SIDECAR,
        {
            "version": 1,
            "generation": 1,
            "jobs": [],
            "accounting": [],
            "records": [{"job_id": job_id, "label": "reviewer", "session_dir": str(session_dir)}],
        },
    )


class TrajectoryOwner:
    """A stand-in owner connection: the three ops the trajectory path uses.

    Only the wire calls are replaced. The real `AttachedSession` above it does
    the subscribing, the paging and the seeding, which is the code under test.
    """

    def __init__(self, rows: list[dict[str, Any]] | None = None, *, fail_watch: bool = False):
        self.connected = True
        self.rows = rows if rows is not None else []
        self.fail_watch = fail_watch
        self.watched: list[str] = []
        self.unwatched: list[str] = []
        self.fetches = 0

    async def watch_job(self, job_id: str) -> str:
        if self.fail_watch:
            raise ConnectionError("owner is gone")
        self.watched.append(job_id)
        return "ok"

    async def unwatch_job(self, job_id: str) -> str:
        self.unwatched.append(job_id)
        return "ok"

    async def close(self) -> None:
        # The bridge's detach disposes its facade, which closes the connection.
        self.connected = False

    async def job_trajectory(self, job_id: str, offset: int = 0, limit: int = 120) -> Any:
        self.fetches += 1
        return {"rows": list(self.rows), "total": len(self.rows), "base_seq": None}


def install_roster(bridge: Any, **fields: Any) -> None:
    """Install canonical state on a bound bridge, the way an owner does."""
    bridge.remote._install_frontend(
        FrontendSessionState(session_id=bridge.session_id, epoch="e1", **fields)
    )


def a_task_job(session_id: str | None = CHILD_ID) -> JobState:
    return JobState(id=JOB_ID, type="task", status="running", session_id=session_id)


def an_update(appends: dict[str, list[dict[str, Any]]], replacements: list[str]) -> FrontendUpdate:
    """One jobs delta carrying the trajectory pair, as the runtime publishes it."""
    return FrontendUpdate(
        epoch="e1",
        sequence=1,
        changes={"streaming": False},
        job_trajectory_appends=appends,
        job_trajectory_replacements=replacements,
    )


def a_bridge(root: Path, frames: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch) -> Any:
    """A bare bridge for `PARENT_ID`, publishing into `frames` instead of a queue.

    Built directly rather than through the pool: these cells are about the
    frame algebra and the job count, and neither needs an owner dial or a
    subscription. The `publish` capture keeps the assertions on the PAYLOAD,
    which is what a client receives.
    """
    bridge = DesktopSessionBridge(root, PARENT_ID, cwd=str(root), retiring=lambda: False)
    monkeypatch.setattr(bridge, "publish", lambda kind, payload, **kw: frames.append(payload))
    return bridge


# -- the frame's opt-in -------------------------------------------------------


def test_an_unwatched_bridge_publishes_exactly_todays_frame(tmp_path, monkeypatch):
    """The blanking survives as a FILTER: a connection that opted in to nothing
    sees nothing new.

    Proven by COMPARISON rather than by inspection: the frame a bridge publishes
    when the wire carried appends for an unwatched job must be indistinguishable
    from the one it publishes when those fields arrived empty — which is the
    frame every existing client and every existing test already reads.
    """

    def publish_once(appends: dict[str, list[dict[str, Any]]], replacements: list[str]):
        frames: list[dict[str, Any]] = []
        bridge = a_bridge(tmp_path / uuid.uuid4().hex, frames, monkeypatch)
        bridge._frontend(an_update(appends, replacements))
        assert len(frames) == 1
        return frames[-1]

    rows = [{"type": "message_update", "delta": "hello", TRAJECTORY_SEQ_KEY: 4}]
    with_appends = publish_once({JOB_ID: rows}, [JOB_ID])
    with_none = publish_once({}, [])
    assert with_appends == with_none
    assert with_appends["job_trajectory_appends"] == {}
    assert with_appends["job_trajectory_replacements"] == []


def test_only_the_loaded_jobs_rows_pass_through(tmp_path, monkeypatch):
    """A watched job's rows ride; a sibling job's do not, in the SAME frame."""
    frames: list[dict[str, Any]] = []
    bridge = a_bridge(tmp_path, frames, monkeypatch)
    bridge._trajectory_watches[JOB_ID] = 1
    mine = [{"type": "message_update", "delta": "mine", TRAJECTORY_SEQ_KEY: 9}]
    theirs = [{"type": "message_update", "delta": "theirs", TRAJECTORY_SEQ_KEY: 3}]
    bridge._frontend(an_update({JOB_ID: mine, OTHER_JOB_ID: theirs}, [JOB_ID, OTHER_JOB_ID]))

    payload = frames[-1]
    assert payload["job_trajectory_appends"] == {JOB_ID: mine}
    # The marker must ride WITH the rows: dropping it would leave a hole in the
    # follower's list permanently, because it is what says "replacement".
    assert payload["job_trajectory_replacements"] == [JOB_ID]


def test_a_frozen_window_thaws_to_plain_json_rows(tmp_path):
    """The seed's rows are ordinary containers, stamps and order untouched.

    Canonical state holds rows in its frozen shapes, which are `tuple`
    subclasses and serialize as ARRAYS OF PAIRS rather than objects. A reply
    model would either reject them or ship a row the client cannot read, and
    every row carries whole tool results, so the mangling would be expensive to
    find. `_lo_seq` is asserted verbatim and in order: it is the reader's
    identity for a row, and a wire value that re-ordered or re-stamped it would
    break the one rule that makes the live stream safe to merge with the seed.
    """
    from local_operator.session.frontend_state import (
        FrontendSessionState,
        FrontendStateStore,
        job_trajectory_wire_value,
    )

    store = FrontendStateStore(FrontendSessionState(session_id=PARENT_ID, epoch="e1"))
    rows = [
        {
            "type": "tool_call_compose",
            "args": {"nested": {"deep": [1, {"x": True}]}},
            TRAJECTORY_SEQ_KEY: 4,
        },
        {"type": "message_update", "delta": "hi", TRAJECTORY_SEQ_KEY: 5},
    ]
    store.mutate(jobs=[JobState(id=JOB_ID, type="task", status="running")])
    assert store.seed_job_trajectory(JOB_ID, rows) is True

    window = next(job for job in store.state.jobs if job.id == JOB_ID)
    frozen = list(window.trajectory)
    # The frozen shapes are tuples, so the raw values would NOT round-trip
    # through JSON as objects — which is the whole reason the helper exists.
    assert isinstance(frozen[0], tuple)
    assert job_trajectory_wire_value(frozen) == rows
    assert job_trajectory_wire_value(None) == []


def test_the_cold_and_attention_rewrites_still_ride_the_frame(tmp_path, monkeypatch):
    """The pass-through is ADDITIVE: nothing else about the frame moved.

    The two fields this change touches sit in a method that also rewrites
    `changes.attention` and merges the cold pair. A pass-through that replaced
    the whole payload rather than those two keys would take both with it.
    """
    frames: list[dict[str, Any]] = []
    bridge = a_bridge(tmp_path, frames, monkeypatch)
    bridge._frontend(an_update({}, []))
    payload = frames[-1]
    assert "attention" not in payload["changes"]
    assert set(bridge._cold_fields()) <= set(payload)


# -- the refcount -------------------------------------------------------------


@pytest.mark.asyncio
async def test_two_readers_of_one_child_share_one_subscription(tmp_path, monkeypatch):
    """The bridge's count, not the caller's, decides when the watch is real.

    Two windows on one child are two HTTP requests against ONE bridge, so the
    owner must be asked to watch exactly once — and a close on either window must
    not release the other's stream. Without the count the second window's page
    silently stops growing, which is the regression this feature is most likely
    to ship.
    """
    parent_dir, child_dir = a_parent_and_child(tmp_path)
    name_child(parent_dir, child_dir)
    pool = DesktopSessions(tmp_path)
    owner = TrajectoryOwner([{"type": "message_end", TRAJECTORY_SEQ_KEY: 1}])
    frames: list[dict[str, Any]] = []
    async with pool.session(PARENT_ID, read=True) as bridge:
        bridge.remote._client = owner  # type: ignore[assignment]
        install_roster(bridge, jobs=[a_task_job()])
        first = await bridge.load_child_trajectory(JOB_ID)
        second = await bridge.load_child_trajectory(JOB_ID)
        assert first["available"] is True and second["available"] is True
        assert owner.watched == [JOB_ID]
        assert bridge.watched_trajectory_jobs == frozenset({JOB_ID})

        released = await pool.unload_child_trajectory(PARENT_ID, JOB_ID)
        assert released == {"watching": True, "watchers": 1}
        assert owner.unwatched == []
        # ...and the frame still carries the job's rows, for the window whose
        # reader is still open.
        monkeypatch.setattr(bridge, "publish", lambda k, payload, **kw: frames.append(payload))
        bridge._frontend(an_update({JOB_ID: [{"type": "message_end"}]}, []))
        assert set(frames[-1]["job_trajectory_appends"]) == {JOB_ID}

        assert await pool.unload_child_trajectory(PARENT_ID, JOB_ID) == {
            "watching": False,
            "watchers": 0,
        }
        assert owner.unwatched == [JOB_ID]
        # Idempotent: the last release reached the owner once, and a further
        # release is an answer rather than a second unsubscribe.
        assert await pool.unload_child_trajectory(PARENT_ID, JOB_ID) == {
            "watching": False,
            "watchers": 0,
        }
        assert owner.unwatched == [JOB_ID]


@pytest.mark.asyncio
async def test_a_failed_load_does_not_leave_a_count_behind(tmp_path):
    """A watch that never armed must not make the reader unretryable."""
    parent_dir, child_dir = a_parent_and_child(tmp_path)
    name_child(parent_dir, child_dir)
    pool = DesktopSessions(tmp_path)
    async with pool.session(PARENT_ID, read=True) as bridge:
        bridge.remote._client = TrajectoryOwner(fail_watch=True)  # type: ignore[assignment]
        install_roster(bridge, jobs=[a_task_job()])
        assert await bridge.load_child_trajectory(JOB_ID) == {
            "rows": [],
            "base_seq": None,
            "total": 0,
            "trajectory_length": 0,
            "available": False,
            "reason": "no-owner",
        }
        assert bridge.watched_trajectory_jobs == frozenset()
        # The retry is not refused by a count the failure left behind.
        bridge.remote._client = TrajectoryOwner()  # type: ignore[assignment]
        assert (await bridge.load_child_trajectory(JOB_ID))["available"] is True


@pytest.mark.asyncio
async def test_a_release_never_builds_a_bridge(tmp_path):
    """Cleanup must not be the request that attaches (or spawns) a session."""
    pool = DesktopSessions(tmp_path)
    assert await pool.unload_child_trajectory(PARENT_ID, JOB_ID) == {
        "watching": False,
        "watchers": 0,
    }
    assert pool.bridges == {}


@pytest.mark.asyncio
async def test_a_release_refuses_a_value_that_could_not_name_a_child(tmp_path):
    """Idempotent is not the same as unvalidated: a malformed id is not a target."""
    pool = DesktopSessions(tmp_path)
    with pytest.raises(SubagentChildUnavailable):
        await pool.unload_child_trajectory(PARENT_ID, "not-an-id")


# -- the seed -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_seed_is_the_window_the_stream_will_extend(tmp_path):
    """The reply carries the loaded rows, their identity stamp, and the counts."""
    parent_dir, child_dir = a_parent_and_child(tmp_path)
    name_child(parent_dir, child_dir)
    rows = [
        {"type": "message_start", "message": {"id": "m1"}, TRAJECTORY_SEQ_KEY: 7},
        {"type": "message_update", "delta": "hi", TRAJECTORY_SEQ_KEY: 8},
    ]
    pool = DesktopSessions(tmp_path)
    async with pool.session(PARENT_ID, read=True) as bridge:
        bridge.remote._client = TrajectoryOwner(rows)  # type: ignore[assignment]
        install_roster(bridge, jobs=[a_task_job()])
        seed = await bridge.load_child_trajectory(JOB_ID)
        assert seed["available"] is True and seed["reason"] is None
        assert seed["rows"] == rows
        assert seed["base_seq"] == 7
        assert seed["total"] == 2 and seed["trajectory_length"] == 2
        # Ordinary JSON containers, not the store's frozen shapes: a row that
        # reached a client as a list of pairs would be worse than useless.
        assert all(isinstance(row, dict) for row in seed["rows"])
        assert seed["rows"][0][TRAJECTORY_SEQ_KEY] == 7


# -- containment and the absences ---------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["job-id-is-not-an-id", "job-id-is-a-path", "job-unknown"])
async def test_the_route_refuses_a_job_this_conversation_does_not_own(tmp_path, case):
    parent_dir, child_dir = a_parent_and_child(tmp_path)
    name_child(parent_dir, child_dir)
    pool = DesktopSessions(tmp_path)
    job_id = {
        "job-id-is-not-an-id": "not-an-id",
        "job-id-is-a-path": str(child_dir),
        "job-unknown": OTHER_JOB_ID,
    }[case]
    async with pool.session(PARENT_ID, read=True) as bridge:
        bridge.remote._client = TrajectoryOwner()  # type: ignore[assignment]
        install_roster(bridge, jobs=[a_task_job()])
        with pytest.raises(SubagentChildUnavailable):
            await bridge.load_child_trajectory(job_id)
        assert bridge.watched_trajectory_jobs == frozenset()


@pytest.mark.asyncio
async def test_a_child_that_is_not_a_subagent_is_refused(tmp_path):
    """The origin clause still runs for a job row that names a real directory."""
    parent_dir, child_dir = a_parent_and_child(tmp_path, child_origin=ORIGIN_FORK)
    name_child(parent_dir, child_dir)
    pool = DesktopSessions(tmp_path)
    async with pool.session(PARENT_ID, read=True) as bridge:
        bridge.remote._client = TrajectoryOwner()  # type: ignore[assignment]
        install_roster(bridge, jobs=[a_task_job()])
        with pytest.raises(SubagentChildUnavailable):
            await bridge.load_child_trajectory(JOB_ID)


@pytest.mark.asyncio
async def test_a_job_with_no_live_lineage_is_answered_by_the_persisted_roster(tmp_path):
    """A swept comms registry must not make a readable child unreadable.

    The roster row's `session_id` comes from the comms registry, so a child that
    settled long enough ago may be gone from it while its directory — and the
    parent's own record of it — are still on disk.
    """
    parent_dir, child_dir = a_parent_and_child(tmp_path)
    name_child(parent_dir, child_dir)
    pool = DesktopSessions(tmp_path)
    async with pool.session(PARENT_ID, read=True) as bridge:
        bridge.remote._client = TrajectoryOwner()  # type: ignore[assignment]
        install_roster(bridge, jobs=[a_task_job(session_id=None)])
        seed = await bridge.load_child_trajectory(JOB_ID)
        assert seed["available"] is True


@pytest.mark.asyncio
async def test_a_bash_job_answers_unsupported_rather_than_not_yours(tmp_path):
    """A background job of this conversation is not a containment refusal.

    It has no child directory to contain, so a proof-first order could only
    refuse it — turning "this job type has nothing to follow" into "this job is
    not yours", which is the one answer the reader must not act on.
    """
    a_parent_and_child(tmp_path)
    pool = DesktopSessions(tmp_path)
    owner = TrajectoryOwner()
    async with pool.session(PARENT_ID, read=True) as bridge:
        bridge.remote._client = owner  # type: ignore[assignment]
        install_roster(bridge, jobs=[JobState(id=JOB_ID, type="bash", status="running")])
        answer = await bridge.load_child_trajectory(JOB_ID)
        assert answer["available"] is False and answer["reason"] == "unsupported"
        assert owner.watched == []


@pytest.mark.asyncio
async def test_a_bridge_with_no_canonical_state_answers_no_owner(tmp_path):
    """The unsynchronized case: retryable, and never a false "not yours"."""
    a_parent_and_child(tmp_path)
    pool = DesktopSessions(tmp_path)
    async with pool.session(PARENT_ID, read=True) as bridge:
        assert bridge.roster_jobs() == ()
        answer = await bridge.load_child_trajectory(JOB_ID)
        assert answer["available"] is False and answer["reason"] == "no-owner"
        # The id SHAPES are still enforced here, so an empty roster cannot be
        # used to walk values the surface refuses everywhere else.
        with pytest.raises(SubagentChildUnavailable):
            await bridge.load_child_trajectory("not-an-id")


# -- the route and the capability ---------------------------------------------


@pytest.fixture
def desktop_app(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    app = FastAPI()
    app.state.config_manager = SimpleNamespace(config_dir=tmp_path)
    pool = DesktopSessions(tmp_path)
    app.state.desktop_sessions = pool
    app.include_router(desktop_sessions.router)
    app.include_router(capabilities.router)
    return app, pool


@pytest.mark.asyncio
async def test_the_route_pair_is_wired_and_the_capability_is_advertised(desktop_app, tmp_path):
    """The op's shape on the wire, and the key that gates it on the client."""
    app, pool = desktop_app
    parent_dir, child_dir = a_parent_and_child(tmp_path)
    name_child(parent_dir, child_dir)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        features = (await client.get("/v1/capabilities")).json()["result"]["features"]
        assert features["subagent_trajectory"] == 1
        # A NEW key beside the reader's existing gate, never a bump of it.
        assert features["subagent_transcript"] == 1

        async with pool.session(PARENT_ID, read=True) as bridge:
            rows = [{"type": "message_end", "text": "done", TRAJECTORY_SEQ_KEY: 3}]
            bridge.remote._client = TrajectoryOwner(rows)  # type: ignore[assignment]
            install_roster(bridge, jobs=[a_task_job()])
            url = f"/v1/desktop/sessions/{PARENT_ID}/children/{JOB_ID}/trajectory"
            opened = await client.post(url)
            assert opened.status_code == 200, opened.text
            assert opened.json()["result"] == {
                "rows": rows,
                "base_seq": 3,
                "total": 1,
                "trajectory_length": 1,
                "available": True,
                "reason": None,
            }

            released = await client.delete(url)
            assert released.status_code == 200, released.text
            assert released.json()["result"] == {"watching": False, "watchers": 0}

            # The containment refusal, asked with the roster installed: an id
            # that is not one of this conversation's own jobs is never a
            # retryable "no owner" — the reader must not keep re-probing
            # something that is not its child.
            refused = await client.post(
                f"/v1/desktop/sessions/{PARENT_ID}/children/{OTHER_JOB_ID}/trajectory"
            )
            assert refused.status_code == 404, refused.text
            assert refused.json()["detail"]["code"] == "child_not_found"

        unauthenticated = await client.post(
            f"/v1/desktop/sessions/{PARENT_ID}/children/{JOB_ID}/trajectory",
            headers={"Authorization": "Bearer wrong"},
        )
        assert unauthenticated.status_code in (401, 403)
