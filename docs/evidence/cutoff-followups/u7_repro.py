"""U7 reproduction: the message typed after a watched cut-off is never served.

Standalone, tree-agnostic: run it with PYTHONPATH pointed at the checkout under
test. It exercises the REAL path — a real runtime subprocess parked in the real
`bash` tool, SIGKILLed while a real viewer watches — and then types again into
the session that just took the cut-off verdict.

Prints a sample line per second so the SPIN and the elapsed time are visible,
and exits 0 having printed a verdict line: WEDGED (never served) or SERVED.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Every family that can pin a spawned runtime to the caller's provider, plus
# the cmux workspace identity that let a headless test rename real workspaces.
for _name in list(os.environ):
    if _name.startswith(("CMUX_", "LOP_MOBILE_CHILD_", "LOP_RUNTIME_")):
        del os.environ[_name]

CONFIG = Path(tempfile.mkdtemp(prefix="u7-repro-"))
os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = str(CONFIG)
os.environ["LOCAL_OPERATOR_NO_SHIMMER"] = "1"
os.environ.pop("NO_COLOR", None)
os.environ["TERM"] = "xterm-256color"

SESSION = "u7repro0001"
PARK_S = 30
SAMPLE_S = 60.0

from local_operator.harness.types import AgentEndEvent  # noqa: E402
from local_operator.session.attached import AttachedSession  # noqa: E402
from local_operator.session.runtime import registry  # noqa: E402


def seed() -> Path:
    directory = CONFIG / "sessions" / SESSION
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_text(
        '{"id": "seed", "ts": 1, "type": "message", "payload": {"kind": "message", '
        '"role": "user", "content": [{"type": "text", "text": "seed"}]}}\n',
        encoding="utf-8",
    )
    (CONFIG / "config.yml").write_text(
        "values:\n  hosting: test\n  model_name: mock\n  tool_approval_mode: auto\n",
        encoding="utf-8",
    )
    return directory


LOG = Path("/tmp/u7_repro_child.log")


def spawn() -> subprocess.Popen[bytes]:
    env = dict(os.environ)
    env.update(
        {
            "LOCAL_OPERATOR_CONFIG_DIR": str(CONFIG),
            "LOP_MOBILE_CHILD_CWD": str(CONFIG),
            "LOP_MOBILE_CHILD_RESUME": SESSION,
            "LOP_SESSION_GRACE_S": "600",
        }
    )
    handle = LOG.open("wb")
    return subprocess.Popen(
        [sys.executable, "-m", "local_operator.session.runtime.process"],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


async def wait_record(timeout: float = 30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for record, _state in registry.scan(CONFIG):
            if record.session_id == SESSION:
                return record
        await asyncio.sleep(0.05)
    raise AssertionError("no runtime record appeared")


def rows(directory: Path) -> int:
    path = directory / "transcript.jsonl"
    return len(path.read_text(encoding="utf-8").splitlines()) if path.exists() else 0


async def main() -> int:
    directory = seed()
    child = spawn()
    viewer = None
    try:
        record = await wait_record()
        assert record.pid == child.pid, (record.pid, child.pid)

        async def never_takes_over():
            raise RuntimeError("a viewer never takes over a session")

        viewer = await AttachedSession.connect(
            record, SESSION, config_dir=CONFIG, takeover_factory=never_takes_over
        )
        await viewer.prompt(f"please [bash:{PARK_S}]")
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            state = getattr(viewer, "frontend_state", None)
            if state is not None and getattr(state, "streaming", False):
                break
            await asyncio.sleep(0.2)
        else:
            raise AssertionError("the parked turn never started")
        await asyncio.sleep(1.0)

        ends: list[AgentEndEvent] = []
        viewer.subscribe(
            lambda event: ends.append(event) if isinstance(event, AgentEndEvent) else None
        )
        child.kill()
        child.wait(timeout=10)
        killed_at = time.monotonic()
        print(f"[kill] runtime {child.pid} SIGKILLed mid-turn (t=0.0s)", flush=True)

        # The cut-off verdict — U1's fix, and the last thing the user hears.
        while time.monotonic() - killed_at < 20:
            if ends:
                break
            await asyncio.sleep(0.1)
        if ends:
            print(
                f"[verdict] t+{time.monotonic() - killed_at:.1f}s  "
                f"aborted={ends[0].aborted} "
                f"cut_off_cause={getattr(ends[0], 'cut_off_cause', '<absent>')!r}\n"
                f"          {ends[0].error}",
                flush=True,
            )
        else:
            print("[verdict] NONE within 20s", flush=True)

        # ...and now type again, exactly as the operator did.
        before_rows = rows(directory)
        typed_at = time.monotonic()
        task = asyncio.create_task(viewer.prompt("are you there?"))
        print(f"[type] message accepted at t+{time.monotonic() - killed_at:.1f}s", flush=True)

        served = False
        while time.monotonic() - typed_at < SAMPLE_S:
            if task.done():
                break
            await asyncio.sleep(1.0)
            print(
                f"  t+{time.monotonic() - killed_at:6.1f}s  "
                f"streaming={viewer.is_streaming} recovering={getattr(viewer, '_recovering', '?')} "
                f"can_go_cold={getattr(viewer, '_can_go_cold', '?')} "
                f"records={len([r for r, _ in registry.scan(CONFIG) if r.session_id == SESSION])} "
                f"transcript_rows={rows(directory)}",
                flush=True,
            )

        if task.done():
            error = task.exception()
            if error is not None:
                print(
                    f"[result] the next message was REFUSED after "
                    f"{time.monotonic() - typed_at:.1f}s: {type(error).__name__}: {error}",
                    flush=True,
                )
                # A refusal that names the state and leaves the session usable is
                # the bounded outcome; a SECOND message has to be served then.
                print("[retry] typing once more into the released session", flush=True)
                second = asyncio.create_task(viewer.prompt("second try [bash:2]"))
                waited = 0.0
                while waited < 90 and not second.done():
                    await asyncio.sleep(1.0)
                    waited += 1
                if second.done() and second.exception() is None:
                    print(f"[retry] SERVED on the second message after {waited:.0f}s", flush=True)
                    served = True
                else:
                    print(f"[retry] still not served after {waited:.0f}s", flush=True)
            else:
                print(
                    f"[result] the next message was SERVED after "
                    f"{time.monotonic() - typed_at:.1f}s "
                    f"(transcript rows {before_rows} -> {rows(directory)})",
                    flush=True,
                )
                served = True
        else:
            print(
                f"[result] WEDGED: accepted at t+{time.monotonic() - killed_at:.1f}s and never "
                f"served after {time.monotonic() - typed_at:.0f}s "
                f"(recovering={getattr(viewer, '_recovering', '?')}, "
                f"transcript_rows={rows(directory)})",
                flush=True,
            )

        print(f"[verdict-line] {'SERVED' if served else 'WEDGED'}", flush=True)
        task.cancel()
        return 0 if served else 1
    finally:
        if viewer is not None:
            try:
                await viewer.dispose()
            except Exception:  # noqa: BLE001
                pass
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
        for record, _state in registry.scan(CONFIG):
            try:
                os.kill(record.pid, 9)
            except ProcessLookupError:
                pass
        shutil.rmtree(CONFIG, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
