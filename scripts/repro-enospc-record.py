#!/usr/bin/env python3
"""Reproduce "a full volume voids the episode record" -- and show it is fixed.

WHAT THIS REPRODUCES, AND WHY IT NEEDS A DISK IMAGE. On 2026-09-28 two
session-arm episodes (``a1696-p1/task_013``, ``a1696-p2/task_004``) completed
their whole interaction -- one through the completion gate, one scored
``partial_ppm 359700`` -- and sealed ``status: failed, steps: 0`` because a
single record write met ENOSPC while a shared host's volume filled mid-run.
The work and the spend were committed; the product was destroyed. Two
properties have to be demonstrated together on a REAL full volume for that to
be a finding rather than a guess:

* the record stops at a line boundary and the seal still lands (the fix), and
* on the pre-fix tree the same run voids (the defect).

So everything here lives on a BOUNDED APFS image (``hdiutil create -size Nm``)
that is detached and deleted on the way out. Nothing outside the image is
written except a small scratch directory.

HOW IT RUNS TWO TREES. ``--legacy-tree`` names a checkout of the pre-fix code
(``origin/main`` worktree); the script re-executes itself with ``PYTHONPATH``
pointed at it and loads ``local_operator`` from there, then prints both
outcomes. The child refuses to run if the import did not resolve to the named
tree, so a fallback to the installed editable package cannot pass as evidence.

WHAT IS FAKED, AND WHAT IS NOT. The adapter (the supervisor's RPC surface) and
the session (its event stream) are in-process fakes -- this is the RECORD path
under test, and the defect lives entirely in it. Everything else is the real
code from the tree under test: the real ``run_session_episode``, the real
``RecordSink``/``_RecordWriter``, the real seal sequence, real files on a real
volume that really fills.

Run it under a worktree venv, from the repository root:

    .venv/bin/python scripts/repro-enospc-record.py --legacy-tree <origin/main worktree>

It exits non-zero when the CURRENT tree does not survive the scenario, so a
silent pass is not possible: scenario A must refuse before ``launch`` (the
volume cannot hold the record), and scenario B must keep the run's real status
and a line-complete record. The legacy result is reported, never asserted --
the pre-fix tree is EXPECTED to void.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

#: The repository this script lives in, put on the path first, so the run
#: exercises the TREE THE SCRIPT LIVES IN (and, in the child, the tree named by
#: ``--code-tree``, which ``PYTHONPATH`` puts ahead of any editable install).
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: The image's filesystem and name. APFS is the host's own filesystem, so the
#: ENOSPC arrives from the same allocator the incident met.
IMAGE_FS = "APFS"

#: The two scenarios: A refuses before launch on a volume that cannot hold a
#: record; B runs on a volume that can, and lets the record fill it.
SCENARIO_REFUSAL_IMAGE_MB = 64
SCENARIO_FULL_IMAGE_MB = 160

#: One emitted event line is a reasoning delta plus JSON framing. The deltas
#: are LARGE on purpose (200 KB against the 0.84 MB largest line observed): the
#: volume has to fill in hundreds of writes, not tens of thousands, and the
#: write path under test is identical for every line size.
EVENT_DELTA_CHARS = 200_000
EVENT_OVERSHOOT_BYTES = 4 * 1024 * 1024

#: Where the episode's scratch lives, and where the image is created. Off the
#: image on purpose: only the RECORD volume fills in this repro.
SCRATCH_HINT = "LOCAL_OPERATOR_SCRATCHPAD"


# ---------------------------------------------------------------------------
# the bounded volume
# ---------------------------------------------------------------------------


def _hdiutil(*argv: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["hdiutil", *argv], check=True, capture_output=True)


def create_and_mount(root: Path, megabytes: int) -> tuple[Path, Path]:
    """Create a bounded APFS image and mount it; returns (image, mountpoint)."""

    root.mkdir(parents=True, exist_ok=True)
    volume = f"lo-enospc-record-{uuid.uuid4().hex[:6]}"
    image = root / f"{volume}.dmg"
    _hdiutil(
        "create",
        "-size",
        f"{megabytes}m",
        "-fs",
        IMAGE_FS,
        "-volname",
        volume,
        "-quiet",
        str(image),
    )
    _hdiutil("attach", str(image), "-nobrowse", "-quiet")
    return image, Path("/Volumes") / volume


def detach_and_delete(image: Path, mountpoint: Path) -> None:
    subprocess.run(
        ["hdiutil", "detach", str(mountpoint), "-force", "-quiet"],
        check=False,
        capture_output=True,
    )
    image.unlink(missing_ok=True)


def free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


# ---------------------------------------------------------------------------
# the fakes: an adapter that answers the protocol, a session that emits events
# ---------------------------------------------------------------------------


def _build_fakes(tmp_path: Path) -> Any:
    """The adapter: ``tests/unit/evaluation/runner/conftest.FakeAdapter``.

    The adapter follows the maintained construction of the pinned protocol
    models; the subclass adds the one capability the session arm requires
    (adapter-owned ask answers). Importing the conftest rather than re-building
    these models here is how the script and the tests stay one definition.
    """

    from local_operator.evaluation.evidence.models import ScoreArtifact
    from tests.unit.evaluation.runner.conftest import FakeAdapter

    class _SessionFakeAdapter(FakeAdapter):
        async def handshake(self, *, timeout: float = 10.0) -> Any:
            base = await super().handshake(timeout=timeout)
            capabilities = base.metadata.capabilities.model_copy(
                update={"ask_user_answer_owner": "adapter"}
            )
            return base.model_copy(
                update={"metadata": base.metadata.model_copy(update={"capabilities": capabilities})}
            )

    return _SessionFakeAdapter(
        tmp_path, "ep-enospc-record", score=ScoreArtifact(status="scored", binary=1)
    )


def _make_session(events: list[Any]) -> Any:
    """The recording session: emits the scripted events during one prompt."""

    from local_operator.evaluation.action_server import ACTION_TOOL_NAME, SERVER_NAME
    from local_operator.mcp.tool_bridge import create_mcp_tool_name

    class _Session:
        def __init__(self) -> None:
            # The action tool is in the LIVE inventory so the settle gate
            # passes without an MCP runtime; nothing in this repro calls it.
            self._tools = [
                type("T", (), {"name": create_mcp_tool_name(SERVER_NAME, ACTION_TOOL_NAME)})()
            ]
            self.mcp_manager = None
            self.mcp_startup = None
            self._sink: Any = None
            self.emitted = 0

        def subscribe(self, sink: Any) -> Any:
            self._sink = sink
            return lambda: None

        def set_tool_confinement(self, root: Any) -> None:  # noqa: ARG002
            return None

        async def prompt(self, text: str, images: Any = None) -> None:  # noqa: ARG002
            for event in events:
                self.emitted += 1
                self._sink(event)

        async def abort(self, reason: Any = None) -> None:  # noqa: ARG002
            return None

    return _Session()


class _Context:
    def __init__(self, session: Any) -> None:
        self._session = session

    async def __aenter__(self) -> Any:
        return self._session

    async def __aexit__(self, *exc: Any) -> bool:
        del exc
        return False


# ---------------------------------------------------------------------------
# one scenario run
# ---------------------------------------------------------------------------


def _build_episode(scratch: Path, record_root_parent: Path) -> tuple[Any, Any, Any, Any]:
    """(spec, config, roots, session_spec) -- the episode's pinned inputs."""

    from local_operator.evaluation.runner.episode import EpisodeConfig
    from local_operator.session.spec import ApprovalPolicy, SessionRoots, SessionSpec
    from tests.unit.evaluation.runner.conftest import build_spec

    spec = build_spec("ep-enospc-record")
    run_root = record_root_parent / "run"
    evidence = run_root / "evidence"
    artifacts = run_root / "artifacts"
    rescue = run_root / "rescue" / spec.episode_id
    config = EpisodeConfig(
        evidence_root=evidence,
        artifact_root=artifacts,
        rescue_root=rescue,
        max_steps=4,
        prepare_timeout=5.0,
        reset_timeout=5.0,
        step_timeout=5.0,
        score_timeout=5.0,
        cleanup_timeout=5.0,
        ask_deadline_ms=1000,
        handshake_timeout=5.0,
    )
    home = scratch / "home"
    roots = SessionRoots(
        config_dir=home / ".local-operator",
        agent_home=home / "local-operator-home",
        cwd=home / "work",
        allow_volatile=True,
    )
    for path in (roots.config_dir, roots.agent_home, roots.cwd):
        Path(path).mkdir(parents=True, exist_ok=True)
    session_spec = SessionSpec(
        hosting="test", model="mock", approvals=ApprovalPolicy.auto(), name="arm-enospc"
    )
    return spec, config, roots, session_spec


async def _drive(
    *,
    scratch: Path,
    record_parent: Path,
    emit_lines: int,
    launched: list[Any],
) -> dict[str, Any]:
    from local_operator.evaluation.session_arm import run_session_episode
    from local_operator.harness.types import ReasoningDeltaEvent
    from tests.unit.evaluation.runner.conftest import selector

    # ISOLATION FIRST, exactly as the tests and the launchd launcher do it: the
    # MCP declaration assert refuses any source outside the scratch, and without
    # a redirected HOME it would find the OPERATOR's own config -- which is how
    # the first draft of this repro "reproduced" the wrong failure. Both
    # prefixes a ``lop`` parent exports are stripped for the same reason the
    # launcher strips them (``CMUX_*`` renames real workspaces; ``LOP_*`` feeds
    # the child product itself).
    home = scratch / "home"
    home.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(home)
    os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = str(home / ".local-operator")
    os.environ["LOCAL_OPERATOR_HOME"] = str(home / "local-operator-home")
    for name in [key for key in os.environ if key.startswith(("CMUX_", "LOP_"))]:
        del os.environ[name]

    spec, config, roots, session_spec = _build_episode(scratch, record_parent)
    adapter = _build_fakes(scratch)
    events = [
        ReasoningDeltaEvent(message_id=f"m{index}", delta="r" * EVENT_DELTA_CHARS)
        for index in range(emit_lines)
    ]
    session = _make_session(events)

    # Monkeypatched at ``sdk.open_session`` (NOT passed as ``session_opener``):
    # the pre-fix tree has no such parameter, so this harness has to drive both
    # trees through one call shape. Both trees resolve ``sdk.open_session`` at
    # call time inside the REAL ``open_episode_session``, so the subscription
    # and the ``EpisodeSession`` wrapping stay in play.
    import local_operator.sdk as sdk

    original = sdk.open_session

    def opener(_spec: Any, roots: Any = None, mode: Any = None) -> Any:  # noqa: ARG001
        del roots, mode
        return _Context(session)

    sdk.open_session = opener

    async def rescue(descriptor: Any, **kwargs: Any) -> Any:
        del descriptor, kwargs
        return type("A", (), {"complete": True, "receipts": (), "rescue_required": False})()

    def launch(selected: Any) -> Any:
        launched.append(selected)
        return adapter

    report: dict[str, Any] = {"raised": None}
    try:
        outcome = await run_session_episode(
            spec=spec,
            config=config,
            selector=selector(scratch),
            roots=roots,
            scratch_root=scratch / "home",
            session_spec=session_spec,
            secrets=(),
            launch=launch,
            rescue=rescue,
        )
        from local_operator.evaluation.session_arm import _outcome_json

        report["outcome"] = _outcome_json(outcome)
    except BaseException as error:  # noqa: BLE001 - the pre-fix tree voids by raising
        report["raised"] = f"{type(error).__name__}: {error}"
    finally:
        sdk.open_session = original

    record_root = config.evidence_root / f"{spec.episode_id}-session"
    report["record_root"] = str(record_root)
    report["lines"] = _line_report(record_root / "events.jsonl")
    report["artifacts"] = (
        sorted(path.name for path in record_root.iterdir() if path.is_file())
        if record_root.exists()
        else []
    )
    report["emitted_lines"] = session.emitted
    report["launched"] = len(launched)
    report["free_after"] = free_bytes(config.evidence_root.parent)
    return report


def _line_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"present": False}
    raw = path.read_bytes()
    lines = raw.split(b"\n")
    complete = lines[:-1] if raw.endswith(b"\n") else lines
    torn = 0
    for line in complete:
        try:
            json.loads(line)
        except json.JSONDecodeError:
            torn += 1
    return {
        "present": True,
        "bytes": len(raw),
        "lines": len(complete),
        "torn_lines": torn,
        "terminated": raw.endswith(b"\n"),
    }


def _scenario(
    scratch: Path, images: Path, megabytes: int, emit_lines: int, *, refusal: bool
) -> dict[str, Any]:
    image, mountpoint = create_and_mount(images, megabytes)
    try:
        launched: list[Any] = []
        report = asyncio.run(
            _drive(
                scratch=scratch,
                record_parent=mountpoint,
                emit_lines=emit_lines,
                launched=launched,
            )
        )
        report["scenario"] = "refusal" if refusal else "full"
        return report
    finally:
        detach_and_delete(image, mountpoint)


def _short_scratch() -> Path:
    """A session-unique scratch DIRECTORY short enough for a UNIX socket.

    ``sun_path`` is bounded (~104 bytes on macOS) and the action bridge binds
    inside this directory, so the session scratchpad -- which is deep -- cannot
    host it. ``mkdtemp`` under the system temporary directory is the shortest
    path a session can name uniquely, and it is deleted on the way out; the
    IMAGES stay in the session scratchpad beside it.
    """

    return Path(tempfile.mkdtemp(prefix="lore-"))


# ---------------------------------------------------------------------------
# entry point: one scenario per process, two trees per invocation
# ---------------------------------------------------------------------------


def _emit_lines_for(megabytes: int) -> int:
    """Enough lines to overrun an image of ``megabytes`` -- with overshoot."""

    per_line = EVENT_DELTA_CHARS + 200
    target = megabytes * 1024 * 1024 + EVENT_OVERSHOOT_BYTES
    return math.ceil(target / per_line)


def _run_child(
    tree: str | None, megabytes: int, emit_lines: int, *, refusal: bool
) -> dict[str, Any]:
    """Run one scenario in a fresh process, optionally loading ``tree``."""

    argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--image-mb",
        str(megabytes),
        "--emit-lines",
        str(emit_lines),
    ]
    if refusal:
        argv.append("--refusal")
    env = dict(os.environ)
    # The notification gate, spelled explicitly rather than inherited: this rig
    # spawns children that drive a real session, and a rig must never put a
    # banner on the operator's screen. The same sweep that enforces this across
    # ``scripts/`` (tests/unit/test_notification_isolation.py) is why it is
    # named here and not left to the ambient environment.
    env["LOCAL_OPERATOR_NO_NOTIFICATIONS"] = "1"
    if tree is not None:
        env["PYTHONPATH"] = tree + os.pathsep + env.get("PYTHONPATH", "")
        argv.extend(["--code-tree", tree])
    completed = subprocess.run(argv, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    if completed.returncode != 0:
        raise SystemExit(
            f"child scenario failed (rc={completed.returncode}):\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    marker = "SCENARIO-RESULT "
    for line in completed.stdout.splitlines():
        if line.startswith(marker):
            return json.loads(line[len(marker) :])
    raise SystemExit(f"child printed no result:\n{completed.stdout}\n{completed.stderr}")


def _child_main(args: argparse.Namespace) -> int:
    """One scenario, one process. Refuses to run unless the load is the tree asked for."""

    import local_operator

    resolved = Path(local_operator.__file__).resolve()
    if args.code_tree is not None and not resolved.is_relative_to(Path(args.code_tree).resolve()):
        print(
            "REFUSING: local_operator resolved to "
            f"{resolved}, outside --code-tree {args.code_tree}; PYTHONPATH did not win "
            "over the editable install and the comparison would be a lie",
            file=sys.stderr,
        )
        return 2
    scratch = _short_scratch()
    images = Path(os.environ.get(SCRATCH_HINT) or tempfile.gettempdir()) / (
        f"enospc-images-{uuid.uuid4().hex[:8]}"
    )
    try:
        report = _scenario(scratch, images, args.image_mb, args.emit_lines, refusal=args.refusal)
        report["tree"] = str(Path(args.code_tree).resolve()) if args.code_tree else str(REPO_ROOT)
        report["resolved_module"] = str(resolved)
        report["version"] = getattr(local_operator, "__version__", "?")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        shutil.rmtree(images, ignore_errors=True)
    print("SCENARIO-RESULT " + json.dumps(report, sort_keys=True))
    return 0


def _parent_main(args: argparse.Namespace) -> int:
    scratch = (
        Path(os.environ.get(SCRATCH_HINT) or tempfile.mkdtemp(prefix="lop-enospc-record-"))
        / f"scratch-{uuid.uuid4().hex[:8]}"
    )
    try:
        return _parent_scenarios(args, scratch)
    finally:
        # The scratch holds only sockets, config and the report; the images are
        # the child's own and it deletes them with its scratch. Because this
        # host is at ~97% disk, both halves clean up even on the failure path.
        shutil.rmtree(scratch, ignore_errors=True)


def _parent_scenarios(args: argparse.Namespace, scratch: Path) -> int:
    (scratch / "home").mkdir(parents=True, exist_ok=True)

    print("== scenario A: a volume that cannot hold the record must refuse before launch")
    refusal = _run_child(
        None, SCENARIO_REFUSAL_IMAGE_MB, _emit_lines_for(SCENARIO_REFUSAL_IMAGE_MB), refusal=True
    )
    print(json.dumps(refusal, indent=2, sort_keys=True))

    print("== scenario B: a volume that fills mid-run keeps the run and lands the seal")
    lines_b = _emit_lines_for(SCENARIO_FULL_IMAGE_MB)
    current = _run_child(None, SCENARIO_FULL_IMAGE_MB, lines_b, refusal=False)
    print(json.dumps(current, indent=2, sort_keys=True))
    legacy = None
    if args.legacy_tree:
        print(f"== scenario B on the pre-fix tree: {args.legacy_tree}")
        legacy = _run_child(args.legacy_tree, SCENARIO_FULL_IMAGE_MB, lines_b, refusal=False)
        print(json.dumps(legacy, indent=2, sort_keys=True))

    ok = _verdict(refusal, current)
    print("=== verdict")
    print(f"  A refusal before launch : {'PASS' if ok['refusal'] else 'FAIL'}")
    print(f"  B record survives run   : {'PASS' if ok['survives'] else 'FAIL'}")
    if legacy is not None:
        void = _legacy_voided(legacy)
        print(f"  pre-fix tree voids      : {'CONFIRMED' if void else 'NOT REPRODUCED'}")
    return 0 if (ok["refusal"] and ok["survives"]) else 1


def _verdict(refusal: dict[str, Any], current: dict[str, Any]) -> dict[str, bool]:
    outcome = refusal.get("outcome") or {}
    refusal_ok = (
        outcome.get("status") == "failed_pre_bundle"
        and (outcome.get("diagnostic") or "").startswith("refusing to start")
        and refusal.get("launched") == 0
    )
    current_outcome = current.get("outcome") or {}
    lines = current.get("lines") or {}
    survives = (
        current_outcome.get("status") == "agent_stop"
        and current_outcome.get("record_incomplete") is True
        and lines.get("torn_lines") == 0
        and "score.json" in (current.get("artifacts") or [])
        and "outcome.json" in (current.get("artifacts") or [])
        # The outcome-side facts this repro exists to preserve: a real score,
        # the tool inventory the run actually used, and a diagnostic that names
        # the VOLUME rather than the run (review round 1, R1-F2). Without these
        # the demo could pass while the sealed summary lied about what the
        # spend bought.
        and current_outcome.get("score") is not None
        and bool(current_outcome.get("tool_names"))
        and "ran out of room" in (current_outcome.get("record_diagnostic") or "")
    )
    return {"refusal": refusal_ok, "survives": survives}


def _legacy_voided(legacy: dict[str, Any]) -> bool:
    """Whether the pre-fix tree voided WITH THE ENOSPC SIGNATURE.

    Deliberately not just ``failed``: the first draft of this repro counted an
    isolation refusal (a real failure, but the wrong one) as the defect, which
    is exactly the "green number that measures nothing" this campaign keeps
    finding. The void must name the record sink or the full volume.
    """

    outcome = legacy.get("outcome") or {}
    raised = legacy.get("raised") or ""
    diagnostic = f"{outcome.get('diagnostic') or ''} {raised}"
    voided = outcome.get("status") == "failed" and outcome.get("steps") == 0
    signature = "record sink" in diagnostic or "No space left on device" in diagnostic
    # The signature gates BOTH arms (review round 1, R1-F1). An exception on
    # its own is not this defect -- the first draft of this check counted an
    # isolation refusal as the void, the "green number that measures nothing"
    # the docstring above warns about -- so a raise only counts when its message
    # names the sink or the full volume.
    return signature and (bool(raised) or voided)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-tree", default=None, help="a pre-fix checkout to compare")
    parser.add_argument("--code-tree", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--refusal", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--image-mb", type=int, default=SCENARIO_FULL_IMAGE_MB)
    parser.add_argument("--emit-lines", type=int, default=0)
    args = parser.parse_args()
    if args.emit_lines == 0:
        args.emit_lines = _emit_lines_for(args.image_mb)
    if args.child and args.code_tree is not None:
        # The tree asked for goes AHEAD of the module-level REPO_ROOT insert, so
        # ``import local_operator`` resolves to it; ``_child_main`` re-checks
        # the resolution and refuses otherwise.
        sys.path.insert(0, str(Path(args.code_tree).resolve()))
    if args.child:
        return _child_main(args)
    return _parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
