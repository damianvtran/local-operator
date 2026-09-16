"""The record arm on the REAL app: an owner that is discoverable but silent.

WHAT THIS DRIVES, AND WHY IT HAS TO BE A SCRIPT. The shape the review rounds
argue about is not reproducible from a unit test: the viewer must be a real
``OperatorApp`` under the production stylesheet, bound to a real spawned
runtime, and the owner must die in the one way the registry models as LIVE — a
record on disk whose pid is alive with nothing listening on its control port
(the recycled-pid shape). Every arm then depends on real timing: the give-up's
bound, whether a released writer binds again, whether a queued steer is
delivered, and which of the app's rows is on screen when it is not.

Round 2 of PR #998 measured this shape in a scratch cell and the reviewer could
only verify the loop half, so the driver is committed here (review round 2,
NIT-3). Nothing at the product boundary is stubbed:

  * a REAL ``python -m local_operator.session.runtime.process`` child publishing
    its own discovery record, on the ``test``/``mock`` provider;
  * the REAL ``OperatorApp`` driven with ``run_test``, typing through the real
    ``Editor`` (Paste + Enter, the t=0 race);
  * the viewer constructed the way ``cli.py``'s ``viewer_factory`` constructs it
    — attach when a usable record exists, cold otherwise, with
    ``takeover_factory`` raising by construction;
  * the silent owner MANUFACTURED the way the registry itself admits one: the
    killed owner's record re-published at a live pid (a plain ``sleep``) with the
    legacy ``.session.pid`` mirror, so ``scan`` reports ``live`` on every pass
    and every dial raises.

Modes:
  silent    -- nothing typed; the in-flight turn must reach a verdict and settle.
  steer     -- a message typed while the turn is still believed live (a steer).
  ride      -- the same steer, then a second message once the give-up has
               released the viewer: whether the queued steer still rides along
               with it is the Q-1 measurement.
  aftergive -- nothing typed until after the give-up: the prompt path's own
               hand-back, and the row it prints.
  retry     -- the prompt path's hand-back followed by a SECOND Enter on the
               text it restored, which is what a user following the row's advice
               does. The shape U6 measured: each refusal must not leave another
               row for the same message or another copy of the same warning.
  late      -- no planted record; a REAL successor spawned ``--successor-after``
               seconds in, so a slow-but-real re-engage is measured rather than
               assumed.

Usage:
    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/silent_owner_drive.py --mode ride --ride-at 55 --marks 10,30,45,60 \
        --out /tmp/arm --label new

Run it from the worktree under test, with that worktree's OWN venv: a worktree
whose ``.venv`` is a symlink to another checkout makes every spawned runtime
child import the OTHER tree (``python_argv()`` prepends ``-P``, which drops cwd
from ``sys.path``), and the children then run code that has nothing to do with
the commit being measured.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# The tree under test is the one this script was invoked from, not the one this
# file happens to live in: the driver is committed, but the measurements are
# taken per worktree.
TREE = Path(os.environ.get("SILENT_OWNER_TREE", os.getcwd())).resolve()
sys.path.insert(0, str(TREE))

# Every child this script spawns must be unable to reach the operator's own
# session: the families the harness injects into an agent-run shell are stripped
# before anything reads the environment, and the sandbox HOME below is the only
# config root any of them can see.
for _name in list(os.environ):
    if _name.startswith(("CMUX_", "LOP_MOBILE_CHILD_", "LOP_RUNTIME_")):
        del os.environ[_name]

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()
CONFIG = Path(os.environ["LOCAL_OPERATOR_CONFIG_DIR"])
CONFIG.mkdir(parents=True, exist_ok=True)
HOME = Path(os.environ["HOME"])
os.environ["LOCAL_OPERATOR_NO_NOTIFICATIONS"] = "1"
os.environ["LOCAL_OPERATOR_NO_TERMINAL_TITLE"] = "1"

from textual import events  # noqa: E402

from local_operator.session.attached import (  # noqa: E402
    COLD_FALLBACK_S,
    AttachedSession,
)
from local_operator.session.runtime import registry  # noqa: E402
from local_operator.session.runtime.types import (  # noqa: E402
    PROTOCOL_VERSION,
    SessionRecord,
)
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.editor import Editor  # noqa: E402

SESSION = "silent-owner-drive"
#: The mock provider's ``[bash:N]`` marker is the only way to make an assembled
#: runtime genuinely busy for a known duration, and its pattern takes ONE OR TWO
#: digits (``\d{1,2}``, bounded to 60 s) — so ``[bash:120]`` silently matches
#: NOTHING and the turn completes with a text answer instead of parking. The
#: first draft of this driver used 120 and measured a turn that had already
#: finished, which made every "steer" a prompt. It has to be 60.
PARK_S = 60
LOG = HOME / "child.log"


async def _never_take_over():
    raise RuntimeError("a viewer never takes over a session")


def seed() -> None:
    """A user conversation with one row, and a config the mock provider reads."""
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


def spawn_runtime() -> subprocess.Popen[bytes]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CMUX_", "LOP_"))}
    env.update(
        {
            "HOME": str(HOME),
            "LOCAL_OPERATOR_CONFIG_DIR": str(CONFIG),
            "LOP_MOBILE_CHILD_CWD": str(CONFIG),
            "LOP_MOBILE_CHILD_RESUME": SESSION,
            "LOP_SESSION_GRACE_S": "600",
        }
    )
    handle = LOG.open("ab")
    return subprocess.Popen(
        [sys.executable, "-m", "local_operator.session.runtime.process"],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


async def wait_record(timeout: float = 40.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for record, state in registry.scan(CONFIG):
            if record.session_id == SESSION and state == "live":
                return record
        await asyncio.sleep(0.05)
    raise AssertionError("no live record appeared")


def plant_silent_owner(pid: int) -> None:
    """Re-publish the record at a LIVE pid nothing listens on.

    The record advertises the frontend capability exactly as a real runtime's
    does: without it the viewer can see the record and cannot USE it, which is a
    different arm (the unusable-record one) bounded on every tree.
    """
    from local_operator.session.frontend_state import FRONTEND_CAPABILITY
    from local_operator.session.retention import LIVE_MARKER_NAME

    record = SessionRecord(
        pid=pid,
        kind="tui",
        session_id=SESSION,
        conversation_name="silent owner",
        cwd=str(CONFIG),
        model_label="mock",
        control_port=1,  # nothing listens here
        control_key="0" * 64,
        protocol=PROTOCOL_VERSION,
        capabilities=[FRONTEND_CAPABILITY],
    )
    registry.publish(record, CONFIG)
    (CONFIG / "sessions" / SESSION / LIVE_MARKER_NAME).write_text(str(pid), encoding="utf-8")


def remove_planted_owner(pid: int) -> None:
    from local_operator.session.retention import LIVE_MARKER_NAME

    registry.unpublish(pid, CONFIG)
    (CONFIG / "sessions" / SESSION / LIVE_MARKER_NAME).unlink(missing_ok=True)


def transcript_text(app: OperatorApp) -> str:
    out = []
    for block in app._transcript_view().blocks():
        text = getattr(block, "text", None)
        if callable(text):
            text = text()
        if not text:
            text = getattr(block, "_text", "") or ""
        if text:
            out.append(f"<{type(block).__name__}> {str(text).strip()}")
    return "\n".join(out)


def editor_text(app: OperatorApp) -> str:
    return app.query_one(Editor).text


def session_record_pids() -> list[int]:
    """Every record for this session, read RAW.

    ``registry.scan`` unlinks stale records, so using it here would perturb the
    recovery under observation.
    """
    run = registry.run_dir(CONFIG)
    pids = []
    for path in sorted(run.glob("*.json")):
        try:
            record = SessionRecord.from_json(json.loads(path.read_text()))
        except Exception:  # noqa: BLE001 — a listing must survive a torn record
            continue
        if record.session_id == SESSION:
            pids.append(record.pid)
    return pids


async def _settle(pilot, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        await pilot.pause()
        await asyncio.sleep(0.05)


async def _until(pilot, predicate, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await pilot.pause()
        if predicate():
            return True
        await asyncio.sleep(0.05)
    return False


async def _paste_enter(app: OperatorApp, pilot, line: str) -> None:
    """Paste + Enter with nothing in between, which is the t=0 submit race."""
    editor = app.query_one(Editor)
    editor.focus()
    await pilot.pause()
    app.post_message(events.Paste(line))
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()


def snapshot(app: OperatorApp, session: Any, label: str) -> dict[str, Any]:
    composer = editor_text(app)
    return {
        "label": label,
        "composer": composer,
        "recovering": getattr(session, "_recovering", None),
        "can_go_cold": getattr(session, "_can_go_cold", None),
        "is_cold": getattr(session, "is_cold", None),
        "streaming": getattr(session, "is_streaming", None),
        "runtime_pid": getattr(session, "runtime_pid", None),
        "transcript": transcript_text(app),
        "session_records": session_record_pids(),
    }


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="steer",
        choices=["silent", "steer", "ride", "aftergive", "retry", "late"],
    )
    parser.add_argument("--label", default="head")
    parser.add_argument("--successor-after", type=float, default=4.0)
    parser.add_argument("--ride-at", type=float, default=12.0)
    parser.add_argument("--marks", default="")
    parser.add_argument("--out", default=".")
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "mode": args.mode,
        "tree": str(TREE),
        "cold_fallback_s": COLD_FALLBACK_S,
    }

    OperatorApp._check_for_update = lambda self: None  # type: ignore[method-assign]
    seed()
    child = spawn_runtime()
    sleeper: subprocess.Popen[bytes] | None = None
    successor: subprocess.Popen[bytes] | None = None
    planted: int | None = None
    try:
        record = await wait_record()
        assert record.pid == child.pid, (record.pid, child.pid)
        summary["owner_pid"] = record.pid
        print(f"[{args.label}] runtime pid={child.pid} record published", flush=True)

        factory_order: list[str] = []

        async def factory():
            """``cli.py``'s ``viewer_factory``, verbatim in effect."""
            from local_operator.mobile.attach_client import find_runtime_record

            found, _owner = await asyncio.to_thread(find_runtime_record, CONFIG, SESSION)
            if found is not None:
                try:
                    session = await AttachedSession.connect(
                        found, SESSION, config_dir=CONFIG, takeover_factory=_never_take_over
                    )
                    factory_order.append("attached")
                    return session
                except (ConnectionError, OSError, TimeoutError) as error:
                    print(f"[{args.label}]   attach failed, going cold: {error!r}", flush=True)
            factory_order.append("cold")
            return await AttachedSession.cold(
                SESSION, config_dir=CONFIG, cwd=str(CONFIG), takeover_factory=_never_take_over
            )

        app = OperatorApp(factory)
        app._check_for_update = lambda: None  # type: ignore[method-assign]

        async with app.run_test(size=(100, 30)) as pilot:
            await _until(pilot, lambda: app._session is not None, timeout=60)
            session = app._session
            assert session is not None, "the viewer never built a session"
            summary["factory"] = factory_order[:]
            await _settle(pilot, 2.0)

            # A real turn, parked in a real tool on the real child.
            await _paste_enter(app, pilot, f"please [bash:{PARK_S}]")
            parked = await _until(pilot, lambda: session.is_streaming, timeout=30)
            summary["parked"] = parked
            print(f"[{args.label}] parked a turn: streaming={session.is_streaming}", flush=True)
            await _settle(pilot, 1.0)

            # The owner dies. SIGKILL is the shape with no owner-side cleanup.
            killed_at = time.monotonic()
            child.kill()
            child.wait(timeout=10)
            if args.mode != "late":
                sleeper = subprocess.Popen(
                    ["/bin/sleep", "600"],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                )
                planted = sleeper.pid
                plant_silent_owner(sleeper.pid)
                print(
                    f"[{args.label}] owner SIGKILLed at t=0.0s; record re-planted at LIVE pid "
                    f"{sleeper.pid} (nothing listening)",
                    flush=True,
                )
            else:
                print(
                    f"[{args.label}] owner SIGKILLed at t=0.0s; real successor at "
                    f"t+{args.successor_after}s",
                    flush=True,
                )

                async def spawn_late() -> None:
                    nonlocal successor
                    await asyncio.sleep(args.successor_after)
                    registry.scan(CONFIG)  # reap the killed owner's stale record
                    successor = spawn_runtime()
                    print(
                        f"[{args.label}] real successor {successor.pid} spawned at "
                        f"t+{time.monotonic() - killed_at:.1f}s",
                        flush=True,
                    )

                asyncio.create_task(spawn_late())

            checkpoints = {
                "silent": [5, 10, 30, 50],
                "steer": [5, 10, 30, 50],
                "late": [5, 10, 30, 50],
                "aftergive": [13, 20, 30, 50],
                "retry": [20, 30],
                "ride": [5, 10, 20, 35, 50],
            }
            marks = (
                [float(x) for x in args.marks.split(",")] if args.marks else checkpoints[args.mode]
            )
            events: list[tuple[float, str, object]] = [(m, "check", m) for m in marks]
            if args.mode in {"steer", "ride", "late"}:
                events.append((2.0, "type", "are you there?"))
            if args.mode == "aftergive":
                events.append((12.0, "type", "are you there?"))
            if args.mode == "retry":
                # The FIRST press types; the second is a bare Enter, because that
                # is what the restored composer asks for — the text is already
                # there and pasting it again would measure the driver, not the
                # app.
                events.append((12.0, "type", "are you there?"))
                events.append((14.0, "enter", None))
            if args.mode == "ride":
                events.append((args.ride_at, "ride", None))
            events.sort(key=lambda item: (item[0], 0 if item[1] == "type" else 1))

            async def checkpoint(mark: float) -> None:
                snap = snapshot(app, session, f"t+{int(mark)}")
                summary.setdefault("checkpoints", []).append(snap)
                print(
                    f"[{args.label}] t+{int(mark):>2}s recovering={snap['recovering']} "
                    f"is_cold={snap['is_cold']} can_go_cold={snap['can_go_cold']} "
                    f"streaming={snap['streaming']} runtime_pid={snap['runtime_pid']} "
                    f"composer={snap['composer']!r}",
                    flush=True,
                )
                print(
                    f"[{args.label}]        transcript: {snap['transcript'][-500:]!r}", flush=True
                )
                path = out / f"{args.label}-{args.mode}-t{int(mark)}.svg"
                save_capture(app, path)
                drawn = path.read_text(encoding="utf-8")
                print(
                    f"[{args.label}]        in-frame: queued={'queued' in drawn} "
                    f"runtime_stopped={'runtime stopped' in drawn} "
                    f"rode_along={'rode along' in drawn}",
                    flush=True,
                )

            for offset, kind, payload in events:
                await _until(
                    pilot,
                    lambda offset=offset: time.monotonic() - killed_at >= offset,
                    timeout=offset + 40,
                )
                if kind == "check":
                    await checkpoint(payload)  # type: ignore[arg-type]
                elif kind == "type":
                    await _paste_enter(app, pilot, str(payload))
                    print(
                        f"[{args.label}] typed {payload!r} at t+"
                        f"{time.monotonic() - killed_at:.1f}s "
                        f"(composer now {editor_text(app)!r})",
                        flush=True,
                    )
                elif kind == "enter":
                    app.query_one(Editor).focus()
                    await pilot.pause()
                    await pilot.press("enter")
                    await pilot.pause()
                    print(
                        f"[{args.label}] pressed enter on the restored text at t+"
                        f"{time.monotonic() - killed_at:.1f}s "
                        f"(composer now {editor_text(app)!r})",
                        flush=True,
                    )
                elif kind == "ride":
                    if planted is not None:
                        remove_planted_owner(planted)
                    print(
                        f"[{args.label}] planted record removed at t+"
                        f"{time.monotonic() - killed_at:.1f}s; sending the next message",
                        flush=True,
                    )
                    await _paste_enter(app, pilot, "second message [bash:2]")
                    await _until(
                        pilot,
                        lambda: "second message"
                        in (CONFIG / "sessions" / SESSION / "transcript.jsonl").read_text(
                            encoding="utf-8"
                        ),
                        timeout=60,
                    )
                    await _settle(pilot, 3.0)
                    print(
                        f"[{args.label}] second message served at t+"
                        f"{time.monotonic() - killed_at:.1f}s",
                        flush=True,
                    )

            body = (CONFIG / "sessions" / SESSION / "transcript.jsonl").read_text(encoding="utf-8")
            summary.update(
                {
                    "transcript_rows": len(body.splitlines()),
                    "steer_on_runtime_transcript": body.count("are you there"),
                    "second_on_runtime_transcript": body.count("second message"),
                    "successor_pid": successor.pid if successor else None,
                    "successor_alive": (
                        successor.poll() is None if successor is not None else None
                    ),
                    "records_at_end": session_record_pids(),
                    "final_transcript": transcript_text(app),
                    "final_composer": editor_text(app),
                }
            )
            print(
                f"[{args.label}] runtime transcript rows={summary['transcript_rows']} "
                f"steer_rows={summary['steer_on_runtime_transcript']} "
                f"second_rows={summary['second_on_runtime_transcript']}",
                flush=True,
            )
            print(f"[{args.label}] final screen: {summary['final_transcript']!r}", flush=True)
            await _settle(pilot, 0.5)
            try:
                await session.dispose()
            except Exception:  # noqa: BLE001 — teardown only
                pass
    finally:
        print("SUMMARY " + json.dumps(summary), flush=True)
        for proc in (child, sleeper, successor):
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)
        # Nothing this run spawned may outlive it: the sandbox HOME goes with it.
        for record, _state in registry.scan(CONFIG):
            try:
                os.kill(record.pid, signal.SIGKILL)
            except OSError:
                pass
        shutil.rmtree(HOME, ignore_errors=not args.keep)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
