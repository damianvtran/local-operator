"""Drive `/fork` through the REAL app and record what actually happened.

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python scripts/fork_evidence.py OUTDIR

This is evidence-gathering, not a test: it exercises the real ``OperatorApp``
against a real session store in a scratch config dir, and prints what each
scenario produced (the receipt text, the fork's directory contents, whether the
parent moved, timings). A unit test asserts a property; this shows the behaviour
a reviewer would otherwise have to take on trust.

The spawn backend is stubbed to RECORD the argv rather than open windows —
opening six terminal windows during evidence collection is not useful, and the
real cmux calls are exercised separately by hand (see the PR). Everything up to
and including the constructed launch command is genuinely executed.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_SCRATCH = tempfile.mkdtemp(prefix="lo-fork-evidence-")
os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = _SCRATCH

from local_operator.harness.types import Message  # noqa: E402
from local_operator.session.transcript import Transcript  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.assistant import AssistantBlock  # noqa: E402
from local_operator.tui.widgets.transcript import NoticeBlock, UserBlock  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

PARENT_ID = "aaaaaaaaaaaa"
SPAWNED: list[dict[str, object]] = []


class ForkableSession(FakeSession):
    """FakeSession pinned to the parent id this evidence run seeds on disk.

    ``FakeSession.session_id`` is a fixed property, so the id is overridden here
    rather than by mutating the shared fixture every other test depends on.
    Also records deferred fork requests so the mid-turn scenario can show the
    request landing on the session rather than being dropped.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fork_requests: list[dict[str, object]] = []
        self._fork_pending = None

    @property
    def session_id(self) -> str:
        return PARENT_ID

    def request_fork(self, config_dir, *, message="", on_complete):  # noqa: ANN001
        self.fork_requests.append({"config_dir": str(config_dir), "message": message})
        self._fork_pending = on_complete

    def has_pending_fork(self) -> bool:
        return self._fork_pending is not None

    async def drain_fork(self) -> None:
        """What the real session does at its next safe boundary."""
        from local_operator.fork import fork_session
        from local_operator.paths import config_dir

        callback, self._fork_pending = self._fork_pending, None
        if callback is None:
            return
        request = self.fork_requests[-1]
        fork_id = await asyncio.to_thread(
            fork_session, config_dir(), PARENT_ID, message=str(request["message"])
        )
        callback(fork_id, "")


class _RecordingBackend:
    """Stands in for a real emulator: records the launch, opens nothing."""

    name = "cmux"

    def __init__(self, *, succeed: bool = True) -> None:
        self.succeed = succeed

    def detect(self, env) -> bool:  # noqa: ANN001
        return True

    def spawn(self, launch, env) -> bool:  # noqa: ANN001
        from local_operator.spawn.cmux import workspace_argv

        SPAWNED.append(
            {
                "argv": workspace_argv("/opt/homebrew/bin/cmux", launch),
                "cwd": launch.cwd,
                "session_id": launch.session_id,
                "succeeded": self.succeed,
            }
        )
        return self.succeed


async def _seed_parent(config_dir: Path) -> None:
    """A parent conversation with real history, named the way a real one is."""
    from local_operator.session.session import CONVERSATION_NAME_CUSTOM_TYPE

    parent = config_dir / "sessions" / PARENT_ID
    parent.mkdir(parents=True, exist_ok=True)
    transcript = Transcript(parent)
    await transcript.append_message(Message.user("refactor the YAML loader"))
    await transcript.append_message(Message.assistant("It parses lazily; here is the plan."))
    await transcript.append_custom(
        CONVERSATION_NAME_CUSTOM_TYPE, {"text": "Refactor the loader", "user_set": False}
    )
    from local_operator.resume import write_session_title

    write_session_title(parent, "Refactor the loader", user_set=False, past_names=[])
    # The fork's stamp must be strictly after the parent's title entry; real
    # forks are minutes later, evidence runs are microseconds.
    time.sleep(0.01)


def _notices(app: OperatorApp) -> list[str]:
    return [(block.text() or "") for block in app.query(NoticeBlock)]


async def _run_scenario(
    name: str, *, message: str, live: bool, mode: str = "window", spawn_ok: bool = True
) -> dict[str, object]:
    """One `/fork` through the real app; returns what it produced."""
    import local_operator.spawn.registry as registry
    from local_operator.paths import config_dir

    SPAWNED.clear()
    config = config_dir()
    shutil.rmtree(config / "sessions", ignore_errors=True)
    await _seed_parent(config)

    before = sorted(os.listdir(config / "sessions"))
    parent_transcript = config / "sessions" / PARENT_ID / "transcript.jsonl"
    parent_before = (parent_transcript.read_bytes(), parent_transcript.stat().st_mtime)

    session = ForkableSession()
    app = OperatorApp(lambda: _factory(session))

    original = registry.active_backend
    registry.active_backend = lambda env=None, **kw: _RecordingBackend(succeed=spawn_ok)
    if mode == "switch":
        registry.active_backend = original

    try:
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app._session = session
            app._append_block(UserBlock("refactor the YAML loader"))
            prose = AssistantBlock()
            prose.update_text("It parses lazily; here is the plan.")
            app._append_block(prose)

            if mode == "switch":
                from local_operator.config import ConfigManager

                manager = ConfigManager(config)
                manager.set_config_value("fork", {"mode": "switch"})

            # The live-turn case: the app defers to the session's boundary.
            app._loop_running = live

            started = time.monotonic()
            app._cmd_fork(
                message, lambda body, kind="info": app._append_block(NoticeBlock(body, kind))
            )
            for _ in range(40):
                await pilot.pause()
                await asyncio.sleep(0.02)
                if not live and _fork_dirs(config, before):
                    break
            if live:
                # The parent's turn reaches its boundary and drains, which is
                # what the session does in production.
                assert session.fork_requests, "the deferred fork never reached the session"
                await session.drain_fork()
                await pilot.pause()
            elapsed = time.monotonic() - started
            notices = _notices(app)
    finally:
        registry.active_backend = original

    forks = _fork_dirs(config, before)
    parent_after = (parent_transcript.read_bytes(), parent_transcript.stat().st_mtime)

    result: dict[str, object] = {
        "scenario": name,
        "notices": notices,
        "elapsed_s": round(elapsed, 3),
        "fork_created": bool(forks),
        "parent_unchanged": parent_before == parent_after,
        "spawned": list(SPAWNED),
        "deferred_request": list(session.fork_requests),
    }
    if forks:
        fork_dir = config / "sessions" / forks[0]
        result["fork_id"] = forks[0]
        result["fork_files"] = sorted(os.listdir(fork_dir))
        result["origin"] = json.loads((fork_dir / "origin.json").read_text())
        boot = fork_dir / "boot-prompt.json"
        result["boot_prompt"] = json.loads(boot.read_text())["text"] if boot.exists() else None
        result["transcript_identical"] = (
            fork_dir / "transcript.jsonl"
        ).read_bytes() == parent_before[0]
    return result


def _fork_dirs(config: Path, before: list[str]) -> list[str]:
    return sorted(set(os.listdir(config / "sessions")) - set(before))


async def main() -> None:
    outdir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/fork-evidence")
    outdir.mkdir(parents=True, exist_ok=True)

    results = []
    results.append(
        await _run_scenario("idle-with-message", message="try the streaming parser", live=False)
    )
    results.append(await _run_scenario("idle-bare", message="", live=False))
    results.append(await _run_scenario("turn-running", message="try it another way", live=True))
    results.append(await _run_scenario("spawn-failure", message="", live=False, spawn_ok=False))

    text = json.dumps(results, indent=2)
    (outdir / "fork-scenarios.json").write_text(text)
    print(text)


asyncio.run(main())
