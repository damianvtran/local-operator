"""The desktop child reader's LIVE path over real HTTP and a running child.

Everything here is production: uvicorn, the bearer gate, the new route pair, the
bridge and its per-job refcount, the real `Session` and `Session._launch_subagent`
(a real child with real tools), the runtime's own SSE relay and its
per-connection trajectory filter. The only double is the provider stream, as
everywhere in this stage.

Four properties, in the order a reader experiences them:

* an UNWATCHED connection receives the trajectory pair EMPTY while the child is
  demonstrably producing events (its roster row's `trajectory_length` grows) —
  which is the frame a client that never opts in receives;
* after the watch, the child's rows arrive as `job_trajectory_appends` on the
  session's EXISTING stream, stamped so a reader can merge them with the seed it
  was handed;
* ONE release while a second reader still holds a reference does not stop the
  stream (the refcount — without it, one window closing freezes another's page),
  and the LAST release does;
* the degenerate answers: an id that is not this conversation's child is a 404, a
  job type that records no trajectory is `unsupported`, and a SETTLED child still
  hands over its retained window.
"""

import asyncio
import contextlib
import json
import os
import secrets
import socket
import uuid
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_operator.harness.types import (
    StreamEndEvent,
    StreamTextDelta,
    StreamToolCallDelta,
)
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.runtime.serving import ServingSessionHandle
from tests.e2e.harness import build_session
from tests.e2e.test_desktop_sessions import serve_app

pytestmark = pytest.mark.e2e


#: Seconds between the child's text deltas. Under ``SUBAGENT_TEXT_FLUSH_S``
#: (0.25 s), so several deltas coalesce into ONE `message_update` row rather than
#: one row per token — the producer's own cadence. A test that streamed faster
#: would be measuring a rate it invented.
TEXT_S = 0.18


class TickingChildStream:
    """A provider stream that keeps a child WORKING until the test releases it.

    Each call emits a few text deltas and then a `bash` call, so the child
    produces both halves of what the live reader exists for — streaming text and
    an IN-FLIGHT tool call — repeatedly for as long as the cell needs. The parent
    makes no model call here (the child is launched directly), so every call this
    answers belongs to the child.

    The `release` latch is what turns each observation window into an assertion
    rather than a race: until it is set the child cannot finish, so a window that
    saw no appends saw them SUPPRESSED rather than never produced.
    """

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.calls = 0

    def __call__(self, request: Any, signal: Any = None) -> Any:
        self.calls += 1
        index = self.calls

        async def gen():
            for step in range(3):
                yield StreamTextDelta(delta=f"working {index}.{step} ")
                await asyncio.sleep(TEXT_S)
            if self.release.is_set():
                yield StreamTextDelta(delta="finished.")
                yield StreamEndEvent(stop_reason="stop")
                return
            yield StreamToolCallDelta(
                index=0,
                id=f"call-{index}",
                name="bash",
                argument_delta=json.dumps({"command": f"printf tick-{index}"}),
            )
            yield StreamEndEvent(stop_reason="toolUse")

        return gen()


class FrameLog:
    """Every frame the session's SSE stream delivered, filled by a background task.

    A background reader rather than a `wait_for` around the next line: cancelling
    an in-flight read on an async generator can poison the stream, and every
    assertion below is about what arrived in a WINDOW of time rather than about
    the next frame. Frames are appended in arrival order and never dropped, so a
    window is a slice.
    """

    def __init__(self, lines: Any) -> None:
        self.frames: list[dict[str, Any]] = []
        self._lines = lines
        self._task = asyncio.create_task(self._read())

    async def _read(self) -> None:
        try:
            async for line in self._lines:
                if line.startswith("data: "):
                    self.frames.append(json.loads(line[6:]))
        except Exception:
            # The response closing under us is how this test ENDS, not a result:
            # every assertion here is against the frames already collected.
            return

    async def stop(self) -> None:
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await self._task

    @property
    def mark(self) -> int:
        """Where the log ends now: the start of the NEXT window."""
        return len(self.frames)

    def since(self, mark: int) -> list[dict[str, Any]]:
        return self.frames[mark:]

    def updates(self, mark: int) -> list[dict[str, Any]]:
        return [frame for frame in self.since(mark) if frame.get("type") == "frontend.update"]

    def appends(self, mark: int, job_id: str) -> list[list[dict[str, Any]]]:
        """The batches of trajectory rows this job received since `mark`."""
        batches: list[list[dict[str, Any]]] = []
        for frame in self.updates(mark):
            payload = frame.get("payload") or {}
            rows = (payload.get("job_trajectory_appends") or {}).get(job_id)
            if rows:
                batches.append(rows)
        return batches

    def replacements(self, mark: int, job_id: str) -> list[dict[str, Any]]:
        return [
            frame
            for frame in self.updates(mark)
            if job_id in ((frame.get("payload") or {}).get("job_trajectory_replacements") or [])
        ]

    def retained_marks(self, mark: int, job_id: str) -> list[int]:
        """The retained-length figures the roster published for this job since `mark`."""
        marks: list[int] = []
        for frame in self.updates(mark):
            jobs = ((frame.get("payload") or {}).get("changes") or {}).get("jobs") or []
            for row in jobs:
                if isinstance(row, dict) and row.get("id") == job_id:
                    marks.append(int(row.get("trajectory_length") or 0))
        return marks

    async def wait_until(self, predicate: Any, *, timeout: float) -> int | None:
        """The mark from which `predicate(log, mark)` holds, or ``None`` on timeout.

        ``None`` rather than a falsy mark, because the mark is an INDEX: a
        predicate that holds on the very first frame returns 0, and a caller
        asserting truthiness would read the strongest possible result as a
        failure.
        """
        mark = self.mark
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            if predicate(self, mark):
                return mark
            await asyncio.sleep(0.05)
        return None


def rows_of(batches: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [row for batch in batches for row in batch]


@pytest.mark.asyncio
async def test_the_desktop_stream_carries_a_watched_childs_trajectory(
    headless_tui_env: Path, workspace: Path, monkeypatch
):
    """The opt-in, the appends, the refcount and the degenerate answers, live."""
    root = headless_tui_env
    token = secrets.token_hex(32)
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", token)
    monkeypatch.delenv("LOCAL_OPERATOR_DESKTOP_ORIGINS", raising=False)
    (root / "config.yml").write_text(
        "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n"
    )
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    server, serving = await serve_app(listener, token)
    stream = TickingChildStream()
    runtime: RuntimeServer | None = None
    handle: ServingSessionHandle | None = None
    parent: Any = None
    try:
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{listener.getsockname()[1]}", timeout=30
        ) as client:
            client.headers["Authorization"] = f"Bearer {token}"
            path = "/v1/desktop/sessions"
            created = await client.post(
                path, json={"request_id": str(uuid.uuid4()), "cwd": str(workspace)}
            )
            assert created.status_code == 200, created.text
            session_id = created.json()["result"]["session_id"]
            target = f"{path}/{session_id}"

            parent = build_session(root / "sessions" / session_id, stream, cwd=workspace)
            # Named up front: an unnamed session fires the runtime's one-shot
            # auto-naming errand, which is a real provider call and would be
            # answered from the CHILD's stream.
            parent.set_conversation_name("Live child", user_set=True)
            handle = ServingSessionHandle(
                parent,
                asyncio.get_running_loop(),
                cwd=str(workspace),
                # THE CHILD'S TOOL GATE IS THIS HANDLE'S. `harness/subagent`
                # builds a child `yolo=False` holding the PARENT's own
                # approval handler (deliberately: a child must not be able
                # to skip the gate object), and this handle's gate asks the
                # attached VIEWER — which this cell has none of. Left
                # asking, the child parks on its first tool call, its
                # trajectory stops growing, and every "no appends" window
                # below would prove nothing. Full-auto is how the rest of
                # this suite runs turns (the parent is built `yolo=True`),
                # and it is the same value production reads from
                # `tool_approval_mode: auto`.
                auto_approve=True,
            )
            runtime = RuntimeServer(handle, kind="daemon")
            await runtime.start_in_process()
            (root / "sessions" / session_id / ".session.pid").write_text(str(os.getpid()))
            await parent.async_init()

            async with client.stream("GET", target + "/events") as response:
                assert response.status_code == 200
                log = FrameLog(response.aiter_lines())
                assert (
                    await log.wait_until(
                        lambda log, mark: any(f["type"] == "snapshot" for f in log.frames),
                        timeout=30,
                    )
                    is not None
                ), "no snapshot on the session stream"

                job_id = parent._launch_subagent(label="audit", prompt="audit the config")
                assert job_id, "the parent did not launch a child"

                # -- 1. UNWATCHED: the child works, the frame stays what it was --
                baseline = log.mark
                await asyncio.sleep(6.0)
                updates = log.updates(baseline)
                assert updates, "the session published no frontend.update at all"
                for frame in updates:
                    payload = frame["payload"]
                    assert payload["job_trajectory_appends"] == {}, payload
                    assert payload["job_trajectory_replacements"] == [], payload
                unwatched_marks = log.retained_marks(baseline, job_id)
                assert (
                    unwatched_marks and max(unwatched_marks) >= 1
                ), "the child retained no events, so the empty pair proves nothing"
                print(
                    f"unwatched: {len(updates)} frontend.update frames in 6.0 s, every one with "
                    f"job_trajectory_appends={{}} and job_trajectory_replacements=[], while the "
                    f"child's roster row reached trajectory_length={max(unwatched_marks)}"
                )

                # -- 2. THE DEGENERATE REFUSALS, against the live roster ----------
                # A job of a type that records no trajectory: a real background
                # `bash` job registered on this session's own job manager.
                async def background(job_id: str, signal: Any, report: Any) -> str:
                    await asyncio.sleep(60)
                    return "done"

                bash_job = parent.jobs.register("bash", "background sleep", background)
                # Waited for the ROSTER, deliberately: the type question is
                # answered from the session's own published job rows, so a POST
                # that outran the 50 ms roster coalescer would be refused as an
                # unknown id and prove nothing about the type branch.
                assert (
                    await log.wait_until(
                        lambda log, mark: any(
                            frame["type"] == "frontend.update"
                            and any(
                                row.get("id") == bash_job
                                for row in (frame["payload"].get("changes", {}).get("jobs") or [])
                            )
                            for frame in log.since(mark)
                        ),
                        timeout=15,
                    )
                    is not None
                ), "the background job never reached the follower's roster"
                unsupported = await client.post(f"{target}/children/{bash_job}/trajectory")
                assert unsupported.status_code == 200, unsupported.text
                assert unsupported.json()["result"] == {
                    "rows": [],
                    "base_seq": None,
                    "total": 0,
                    "trajectory_length": 0,
                    "watchers": 0,
                    "joined": False,
                    "available": False,
                    "reason": "unsupported",
                }
                # Reaped here rather than left to the session's disposal: it is a
                # real job on a real manager and the cell no longer needs it.
                await parent.jobs.cancel(bash_job)
                # An id that is not one of this conversation's child jobs.
                refused = await client.post(f"{target}/children/{'fedcba987654'}/trajectory")
                assert refused.status_code == 404, refused.text
                assert refused.json()["detail"]["code"] == "child_not_found"
                print(
                    "degenerate: a background bash job answers available=False "
                    "reason=unsupported, and an id this conversation never launched answers "
                    "404 child_not_found"
                )

                # -- 3. WATCHED: the seed, then appends on the SAME stream --------
                url = f"{target}/children/{job_id}/trajectory"
                opened = await client.post(url)
                assert opened.status_code == 200, opened.text
                seed = opened.json()["result"]
                assert seed["available"] is True, seed
                assert seed["rows"], f"the seed was empty for a running child: {seed}"
                stamps = [row["_lo_seq"] for row in seed["rows"]]
                assert stamps == sorted(stamps), stamps
                assert seed["base_seq"] == stamps[0], seed
                assert seed["total"] == len(seed["rows"])
                assert seed["trajectory_length"] >= seed["total"], seed

                watched = log.mark
                assert (
                    await log.wait_until(lambda log, mark: log.appends(mark, job_id), timeout=20)
                    is not None
                ), "no trajectory appends arrived on the watched stream"
                await asyncio.sleep(3.0)
                batches = log.appends(watched, job_id)
                rows = rows_of(batches)
                assert rows, "no rows in the appended batches"
                assert all("_lo_seq" in row for row in rows), rows
                # The identity rule the reader merges on. Rows at or below the
                # seed's watermark may legitimately ride again (the owner's delta
                # is computed against its own last published window, not against
                # this connection's seed), which is exactly why the rule is a
                # watermark and not an append: what must hold is that the stream
                # ADVANCES past the seed.
                assert max(row["_lo_seq"] for row in rows) > max(stamps), (
                    stamps[-1],
                    max(row["_lo_seq"] for row in rows),
                )
                kinds = sorted({str(row.get("type")) for row in rows})
                assert log.replacements(watched, job_id) == [], "rows were dropped mid-window"
                print(
                    f"watched: seed {seed['total']} rows (base_seq={seed['base_seq']}, "
                    f"trajectory_length={seed['trajectory_length']}); {len(batches)} append "
                    f"frames carried {len(rows)} rows, stamps "
                    f"{min(r['_lo_seq'] for r in rows)}-{max(r['_lo_seq'] for r in rows)} "
                    f"(seed ended at {stamps[-1]}), no replacement marker, row types {kinds}"
                )

                # -- 3b. THE SEED UNDER A RACING STREAM --------------------------
                # The seed is read back out of the window the SAME connection is
                # growing, and the owner computes each delta against its own last
                # published window — so a run of rows emitted between that publish
                # and the seed's install can arrive twice: once inside the seed and
                # once as an append. Sixteen re-opens on a live child reproduced it
                # (17 rows for 15 distinct stamps), and the row that must never
                # repeat is the TEXT delta, whose reducer has no content-based
                # dedupe. So this cell re-opens until a text delta has been inside
                # a seed, and asserts the reply is the documented window: every
                # stamp once, in order, with both counts agreeing.
                deadline = asyncio.get_running_loop().time() + 30.0
                racing: list[tuple[int, int, int]] = []
                text_seeds = 0
                while (len(racing) < 30 or text_seeds == 0) and (
                    asyncio.get_running_loop().time() < deadline
                ):
                    again = await client.post(url)
                    assert again.status_code == 200, again.text
                    again_seed = again.json()["result"]
                    again_stamps = [row["_lo_seq"] for row in again_seed["rows"]]
                    duplicates = sorted(
                        {stamp for stamp in again_stamps if again_stamps.count(stamp) > 1}
                    )
                    assert not duplicates, (len(racing), duplicates, again_seed["rows"])
                    assert again_stamps == sorted(again_stamps), (len(racing), again_stamps)
                    assert again_seed["total"] == len(again_seed["rows"]), again_seed
                    assert again_seed["trajectory_length"] == again_seed["total"], again_seed
                    # The reference the section-3 open still holds: this POST
                    # joined a live window rather than opening one, and says so.
                    assert again_seed["watchers"] == 2, again_seed
                    assert again_seed["joined"] is True, again_seed
                    if any(row.get("type") == "message_update" for row in again_seed["rows"]):
                        text_seeds += 1
                    racing.append((len(again_seed["rows"]), again_stamps[0], again_stamps[-1]))
                    assert (await client.delete(url)).status_code == 200
                    await asyncio.sleep(0.02)
                assert text_seeds > 0, (
                    "no seed carried a text delta, so the prose case was not measured",
                    racing,
                )
                print(
                    f"racing seed: {len(racing)} re-opens on a live child, {text_seeds} of them "
                    f"carrying a message_update TEXT delta inside the window; every reply held "
                    f"each stamp once, in order, with total==trajectory_length; windows "
                    f"{min(rows for rows, _, _ in racing)}-{max(rows for rows, _, _ in racing)} "
                    f"rows, base stamps {min(b for _, b, _ in racing)}-"
                    f"{max(b for _, b, _ in racing)}"
                )

                # -- 4. THE REFCOUNT: one release, another window still reading ---
                second = await client.post(url)
                assert second.status_code == 200, second.text
                held = log.mark
                released = await client.delete(url)
                assert released.status_code == 200, released.text
                assert released.json()["result"] == {"watching": True, "watchers": 1}
                await asyncio.sleep(4.0)
                still = log.appends(held, job_id)
                assert still, "the stream stopped although another reader still held a reference"
                print(
                    "refcount: two watches on one child; ONE DELETE answered "
                    "{'watching': True, 'watchers': 1} and the stream kept delivering "
                    f"({len(rows_of(still))} rows in the next 4.0 s)"
                )

                # -- 5. THE LAST RELEASE: the child keeps working, nothing arrives -
                before_release = log.retained_marks(held, job_id)
                stopped = await client.delete(url)
                assert stopped.status_code == 200, stopped.text
                assert stopped.json()["result"] == {"watching": False, "watchers": 0}
                quiet = log.mark
                await asyncio.sleep(4.0)
                assert log.appends(quiet, job_id) == [], "appends survived the last release"
                assert log.replacements(quiet, job_id) == [], "a marker survived the last release"
                after_release = log.retained_marks(quiet, job_id)
                assert max(after_release or [0]) > max(before_release or [0]), (
                    "the child stopped producing, so 'no appends' would prove nothing",
                    before_release,
                    after_release,
                )
                print(
                    "last release: 0 append frames and 0 markers in the next 4.0 s while the "
                    f"child's roster row still grew ({max(before_release)} -> "
                    f"{max(after_release)} retained events)"
                )

                # -- 6. A SETTLED child still hands over its retained window ------
                stream.release.set()
                async with asyncio.timeout(60):
                    while str(getattr(parent.jobs.get(job_id), "status", "")) != "completed":
                        await asyncio.sleep(0.1)
                settled = await client.post(url)
                assert settled.status_code == 200, settled.text
                settled_seed = settled.json()["result"]
                assert settled_seed["available"] is True, settled_seed
                assert settled_seed["rows"], settled_seed
                print(
                    "settled child: POST still answers available=True with "
                    f"{settled_seed['total']} retained rows "
                    f"(trajectory_length={settled_seed['trajectory_length']})"
                )

                # A release for a job nobody is watching is an answer, not an error.
                idle = await client.delete(url)
                assert idle.status_code == 200, idle.text
                assert idle.json()["result"] == {"watching": False, "watchers": 0}
                print("post-settle release: answered 200 {'watching': False, 'watchers': 0}")

                await log.stop()
    finally:
        stream.release.set()
        server.should_exit = True
        await asyncio.wait_for(serving, 30)
        listener.close()
        if runtime is not None:
            await runtime.aclose()
        if handle is not None:
            await handle.dispose()
        elif parent is not None:
            await parent.dispose()
