"""A cold, read-only desktop session enter stays off the live tool stack.

WHY THIS FILE EXISTS. Profiling a cache-hit desktop open of a 32k-row journal
(2026-10-25) found two costs that a read-only open never needed:

1. The cold facade restored its subagent roster through
   ``session.session._read_roster_sidecar``. Importing ``session.session`` runs
   ``harness.guard_area`` -> ``tools.builtin`` and ~100 further modules, so the
   first cold open in every fresh process paid for the whole live tool layer.
   The reader now lives in the stdlib-only ``session.roster_sidecar`` leaf.
2. ``locate`` answered a marker-less session's cwd with a backward walk for the
   frontend checkpoint row; on a journal that has none, that walk parses the
   whole file. It now consults the validated ``tail_anchor`` absence proof first.

The import guard runs in a FRESH SUBPROCESS, for the reason
``tests/unit/test_import_graph.py`` gives: pytest has already imported half the
tree, so an in-process ``sys.modules`` check would pass on a real regression.
It drives the REAL enter path (pool -> ``session()`` -> subscribe -> open and
snapshot frames) with a roster sidecar present, so the roster restore that used
to trigger the import actually executes — the guard is structural (it fails if
any cold-path module reaches ``session.session``), not incidental to a fixture
that skips the code.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from local_operator.harness.types import Message
from local_operator.server.utils.desktop_sessions import DesktopSessions
from local_operator.session import roster_sidecar, tail_anchor
from local_operator.session import transcript as transcript_module
from local_operator.session.frontend_state import FRONTEND_CHECKPOINT_CUSTOM_TYPE
from local_operator.session.transcript import Transcript

REPO = Path(__file__).resolve().parents[3]
SESSION_ID = "c01dfa57e0e1"

_HEAVY = ("local_operator.session.session", "local_operator.tools")

# One fresh interpreter: seed a marker-less session with a roster sidecar, open it
# through the real pool, consume the open + snapshot frames, report sys.modules.
_PROBE = r"""
import asyncio, json, sys
from pathlib import Path
from local_operator.harness.types import Message
from local_operator.server.utils.desktop_sessions import DesktopSessions
from local_operator.session.roster_sidecar import SUBAGENT_ROSTER_SIDECAR
from local_operator.session.transcript import Transcript

root = Path(sys.argv[1]); sid = sys.argv[2]

async def main():
    directory = root / "sessions" / sid
    directory.mkdir(parents=True)
    t = Transcript(directory)
    await t.append_message(Message.user("question"))
    await t.append_message(Message.assistant("answer"))
    (directory / SUBAGENT_ROSTER_SIDECAR).write_text(json.dumps(
        {"version": 1, "generation": 1, "jobs": [], "records": [], "accounting": []}))
    loaded_before = sorted(sys.modules)
    pool = DesktopSessions(root)
    cm = pool.session(sid)
    bridge = await cm.__aenter__()
    sub = bridge.subscribe()
    stream = bridge.events(sub, epoch=bridge.epoch, after_seq=bridge.sequence)
    opened = await anext(stream)
    snapshot = await anext(stream)
    await stream.aclose()
    await cm.__aexit__(None, None, None)
    print(json.dumps({
        "before": loaded_before,
        "after": sorted(sys.modules),
        "kinds": [opened.get("type") or opened.get("kind"),
                  snapshot.get("type") or snapshot.get("kind")],
    }))

asyncio.run(main())
"""


def _offenders(modules: list[str]) -> list[str]:
    return sorted(m for m in modules if any(m == h or m.startswith(h + ".") for h in _HEAVY))


def test_cold_desktop_enter_never_imports_the_live_session_or_tool_stack(tmp_path: Path) -> None:
    root = tmp_path / "store"
    home = tmp_path / "home"
    root.mkdir()
    home.mkdir()
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(home),
        "LOCAL_OPERATOR_CONFIG_DIR": str(root),
        "LOCAL_OPERATOR_HOME": str(root / "home"),
        "LOCAL_OPERATOR_LOG_DIR": str(root / "logs"),
    }
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE, str(root), SESSION_ID],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env=env,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    # The probe must have reached the frames, or "nothing imported" proves nothing.
    assert all(result["kinds"]), result["kinds"]
    # Neither importing the server module nor the cold enter may load them.
    assert _offenders(result["before"]) == []
    assert _offenders(result["after"]) == [], (
        "a cold read-only desktop enter imported the live session/tool stack: "
        f"{_offenders(result['after'])[:5]}"
    )


def test_the_live_session_module_reexports_the_one_roster_contract() -> None:
    """The owner path keeps its historical names, bound to the leaf's objects.

    Identity, not equality: a restated copy could drift from the reader the cold
    path uses, which is the drift the leaf exists to make impossible.
    """
    from local_operator.session import session as live

    assert live._read_roster_sidecar is roster_sidecar.read_roster_sidecar
    assert live.SUBAGENT_ROSTER_SIDECAR == roster_sidecar.SUBAGENT_ROSTER_SIDECAR
    assert live.SUBAGENT_ROSTER_CUSTOM_TYPE == roster_sidecar.SUBAGENT_ROSTER_CUSTOM_TYPE
    assert live._SUBAGENT_ROSTER_VERSION == roster_sidecar.SUBAGENT_ROSTER_VERSION


def test_the_roster_leaf_imports_nothing_from_the_package(tmp_path: Path) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, sys; import local_operator.session.roster_sidecar; "
            "print(json.dumps(sorted(sys.modules)))",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    modules = json.loads(proc.stdout.strip().splitlines()[-1])
    package = sorted(m for m in modules if m.startswith("local_operator."))
    assert package == ["local_operator.session", "local_operator.session.roster_sidecar"]


@pytest.mark.parametrize(
    ("payload", "accepted"),
    [
        ({"version": 1, "generation": 0, "jobs": [], "records": []}, True),
        ({"version": 2, "generation": 0, "jobs": [], "records": []}, False),
        ({"version": 1, "generation": True, "jobs": [], "records": []}, False),
        ({"version": 1, "generation": -1, "jobs": [], "records": []}, False),
        ({"version": 1, "generation": 0, "jobs": {}, "records": []}, False),
        ([], False),
    ],
)
def test_roster_reader_contract_is_unchanged_by_the_move(
    tmp_path: Path, payload: object, accepted: bool
) -> None:
    path = tmp_path / roster_sidecar.SUBAGENT_ROSTER_SIDECAR
    path.write_text(json.dumps(payload))
    assert (roster_sidecar.read_roster_sidecar(path) is not None) is accepted
    assert roster_sidecar.read_roster_sidecar(tmp_path / "absent.json") is None


async def _journal(directory: Path, *, checkpoint_cwd: str | None = None) -> None:
    transcript = Transcript(directory, defer_materialise=False)
    for index in range(20):
        await transcript.append_message(Message.user(f"question {index}"))
        await transcript.append_message(Message.assistant(f"answer {index}"))
    if checkpoint_cwd is not None:
        await transcript.append_custom(
            FRONTEND_CHECKPOINT_CUSTOM_TYPE, {"state": {"cwd": checkpoint_cwd}}
        )


def _count_walks(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []
    real = transcript_module.read_latest_custom_entry

    def counted(directory, custom_type):  # type: ignore[no-untyped-def]
        calls.append(custom_type)
        return real(directory, custom_type)

    monkeypatch.setattr(transcript_module, "read_latest_custom_entry", counted)
    return calls


async def _bridge_cwd(root: Path) -> str:
    pool = DesktopSessions(root)
    async with pool.session(SESSION_ID) as bridge:
        return str(bridge.cwd)


@pytest.mark.asyncio
async def test_a_proven_absent_checkpoint_skips_the_whole_journal_walk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "store"
    directory = root / "sessions" / SESSION_ID
    directory.mkdir(parents=True)
    await _journal(directory)
    # Control first: without a proof the walk runs and answers the fallback.
    calls = _count_walks(monkeypatch)
    control = await _bridge_cwd(root)
    assert calls == [FRONTEND_CHECKPOINT_CUSTOM_TYPE]
    assert control == str(root.parent)

    anchor = tail_anchor.build_anchor(directory)
    assert anchor is not None
    assert tail_anchor.write_anchor(directory, anchor)
    assert tail_anchor.proves_absent(directory)
    calls.clear()
    assert await _bridge_cwd(root) == control
    assert calls == []


@pytest.mark.asyncio
async def test_a_checkpoint_appended_after_the_proof_is_still_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The proof is a validated hint: a row written since it falls back to the walk."""
    root = tmp_path / "store"
    directory = root / "sessions" / SESSION_ID
    directory.mkdir(parents=True)
    await _journal(directory)
    anchor = tail_anchor.build_anchor(directory)
    assert anchor is not None
    assert tail_anchor.write_anchor(directory, anchor)
    checkpoint_cwd = str(tmp_path / "launched-here")
    await Transcript(directory).append_custom(
        FRONTEND_CHECKPOINT_CUSTOM_TYPE, {"state": {"cwd": checkpoint_cwd}}
    )
    assert not tail_anchor.proves_absent(directory)
    calls = _count_walks(monkeypatch)
    assert await _bridge_cwd(root) == checkpoint_cwd
    assert calls == [FRONTEND_CHECKPOINT_CUSTOM_TYPE]
