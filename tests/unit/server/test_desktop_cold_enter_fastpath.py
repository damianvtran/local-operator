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
from local_operator.server.utils import desktop_sessions
from local_operator.server.utils.desktop_sessions import DesktopSessions
from local_operator.session import index_prewarm, roster_sidecar, tail_anchor
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
    # Release the writer lock now rather than at finalisation: a test that then
    # compresses this journal would otherwise be refused as "session in use"
    # whenever the GC has not yet run ``__del__``.
    transcript.close()


def _without_the_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin ``_locate`` to the SIDECAR arm alone.

    Since L10 a checkpointless journal with no valid sidecar is proven absent on
    the spot by ``custom_type_proven_absent``, so a "no proof yet → walk" control
    would never reach the walk. The sidecar tests below are about the record's own
    validation (append, rewrite, tick ordering); declining the scan keeps their
    walk counts measuring that, while the scan arm has its own tests further down.
    """
    monkeypatch.setattr(desktop_sessions, "custom_type_proven_absent", lambda *_a, **_k: False)


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
    _without_the_scan(monkeypatch)
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


@pytest.mark.asyncio
async def test_a_same_size_middle_rewrite_falls_back_to_the_walk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review round 1 (MINOR 2), the reviewer's exact repro.

    A checkpoint row written IN PLACE (``r+b``) over a middle row keeps the inode,
    the size and the final row, so ``validate_anchor`` alone still says absent —
    asserted below so this test notices if that gate ever learns to catch it. The
    ``locate`` guard must refuse on the journal's ctime and answer what the walk
    answers: the planted row's cwd, not the fallback.
    """
    root = tmp_path / "store"
    directory = root / "sessions" / SESSION_ID
    directory.mkdir(parents=True)
    await _journal(directory)
    anchor = tail_anchor.build_anchor(directory)
    assert anchor is not None
    assert tail_anchor.write_anchor(directory, anchor)
    journal = directory / "transcript.jsonl"
    before = journal.stat()
    lines = journal.read_bytes().splitlines(keepends=True)
    target_index = len(lines) // 2
    original = lines[target_index]
    planted_cwd = "/tmp/relaunched-elsewhere"
    row = json.dumps(
        {
            "id": "planted",
            "ts": 1,
            "type": "custom",
            "payload": {
                "custom_type": FRONTEND_CHECKPOINT_CUSTOM_TYPE,
                "details": {"state": {"cwd": planted_cwd}},
            },
        },
        separators=(",", ":"),
    ).encode()
    assert len(row) + 1 <= len(original), "fixture row too short to host the plant"
    # Pad inside the JSON string so the line stays exactly the original length.
    pad = len(original) - 1 - len(row)
    row = row.replace(b'"id":"planted"', b'"id":"planted' + b"x" * pad + b'"')
    assert len(row) + 1 == len(original)
    offset = sum(len(line) for line in lines[:target_index])
    with journal.open("r+b") as handle:
        handle.seek(offset)
        handle.write(row + b"\n")
    after = journal.stat()
    assert (after.st_ino, after.st_size) == (before.st_ino, before.st_size)
    assert lines[-1] == journal.read_bytes().splitlines(keepends=True)[-1]
    assert tail_anchor.proves_absent(directory), "validate_anchor now catches this itself"

    # COARSE-TICK SIMULATION (#2941 CI, shard test(3.12,2)). On a filesystem whose
    # ctime/mtime granularity is coarse, a same-tick rewrite leaves the journal's
    # ctime EQUAL to the record's mtime. Pin that state here so this reproduction
    # fails on a fine-grained host exactly the way CI failed, rather than passing
    # by ten microseconds of luck.
    record = directory / tail_anchor.ANCHOR_FILENAME
    journal_ctime_ns = journal.stat().st_ctime_ns
    os.utime(record, ns=(journal_ctime_ns, journal_ctime_ns))
    assert journal.stat().st_ctime_ns == record.stat().st_mtime_ns  # the CI tick
    assert not desktop_sessions._checkpoint_proven_absent(directory)
    assert index_prewarm._rewritten_since_record(directory, journal.stat())

    calls = _count_walks(monkeypatch)
    assert await _bridge_cwd(root) == planted_cwd
    assert calls == [FRONTEND_CHECKPOINT_CUSTOM_TYPE]


@pytest.mark.asyncio
async def test_an_ordinary_append_after_the_proof_keeps_the_fast_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rewrite guard must not cost the append path its skip: a journal that grew
    with ordinary rows is still proven by the suffix scan, whatever its ctime."""
    root = tmp_path / "store"
    directory = root / "sessions" / SESSION_ID
    directory.mkdir(parents=True)
    await _journal(directory)
    anchor = tail_anchor.build_anchor(directory)
    assert anchor is not None
    assert tail_anchor.write_anchor(directory, anchor)
    transcript = Transcript(directory)
    await transcript.append_message(Message.user("appended after the proof"))
    journal = directory / "transcript.jsonl"
    assert journal.stat().st_size > anchor.size
    # Derived from the guard itself, not from a timestamp ordering the fixture
    # cannot guarantee on a coarse-tick filesystem: a grown journal is the append
    # path, so the same-size rewrite rule does not apply to it at all.
    assert desktop_sessions._checkpoint_proven_absent(directory)
    assert not index_prewarm._rewritten_since_record(directory, journal.stat())
    calls = _count_walks(monkeypatch)
    assert await _bridge_cwd(root) == str(root.parent)
    assert calls == []


@pytest.mark.asyncio
async def test_a_metadata_only_change_is_rebuilt_once_by_prewarm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review round 2 (MINOR): a ``chmod`` bumps ctime without touching a byte, so
    the desktop guard refuses the record. Prewarm must then call the record stale
    and republish it ONCE, after which the cold open skips the walk again and the
    selector stops re-queueing the journal."""
    _without_the_scan(monkeypatch)
    root = tmp_path / "store"
    directory = root / "sessions" / SESSION_ID
    directory.mkdir(parents=True)
    await _journal(directory)
    assert index_prewarm.write_tail_anchor(root, SESSION_ID)
    assert index_prewarm._anchor_current(directory)
    journal = directory / "transcript.jsonl"
    journal.chmod(0o600)
    record = directory / tail_anchor.ANCHOR_FILENAME
    # The question is the guard's, not this host's clock: a metadata-only change
    # leaves the journal at or past the record, and either way it must not be
    # trusted. Both predicates are asserted so they cannot drift apart.
    assert journal.stat().st_ctime_ns >= record.stat().st_mtime_ns
    assert index_prewarm._rewritten_since_record(directory, journal.stat())
    assert not desktop_sessions._checkpoint_proven_absent(directory)

    calls = _count_walks(monkeypatch)
    assert await _bridge_cwd(root) == str(root.parent)
    assert calls == [FRONTEND_CHECKPOINT_CUSTOM_TYPE]  # refused: falls back to the walk

    builds: list[object] = []
    real_build = index_prewarm.build_anchor

    def counted_build(target):  # type: ignore[no-untyped-def]
        builds.append(target)
        return real_build(target)

    monkeypatch.setattr(index_prewarm, "build_anchor", counted_build)
    assert not index_prewarm._anchor_current(directory)
    assert index_prewarm.write_tail_anchor(root, SESSION_ID)
    assert len(builds) == 1
    # SETTLE THE TICK, do not assume it. The rebuilt record's mtime is "now"; on a
    # coarse-tick filesystem that is EQUAL to the journal's ctime, and equality must
    # keep refusing (see `_rewritten_since_record`'s residual note). Stamp the
    # ordering a fine-grained filesystem produces naturally, so the assertions below
    # test the settlement rule rather than this host's clock granularity.
    if record.stat().st_mtime_ns <= journal.stat().st_ctime_ns:
        later = journal.stat().st_ctime_ns + 1
        os.utime(record, ns=(later, later))
    assert index_prewarm._anchor_current(directory)
    assert not index_prewarm.write_tail_anchor(root, SESSION_ID)
    assert len(builds) == 1

    calls.clear()
    assert await _bridge_cwd(root) == str(root.parent)
    assert calls == []


@pytest.mark.asyncio
async def test_an_equal_tick_record_refuses_and_a_later_one_hits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CI tick, driven by hand: equal timestamps must fail SAFE, and only a
    strictly later record may skip the walk.

    ``os.utime`` puts the record's mtime exactly on the journal's ctime, the state a
    coarse-granularity filesystem produces with no rewrite at all. "Cannot tell" has
    to mean the walk, so a same-tick record is refused — and pushing the record one
    nanosecond later (what a fine-grained filesystem gives naturally) restores the
    fast path, which proves the refusal is about ordering and not a lost record.
    """
    _without_the_scan(monkeypatch)
    root = tmp_path / "store"
    directory = root / "sessions" / SESSION_ID
    directory.mkdir(parents=True)
    await _journal(directory)
    anchor = tail_anchor.build_anchor(directory)
    assert anchor is not None
    assert tail_anchor.write_anchor(directory, anchor)
    journal = directory / "transcript.jsonl"
    record = directory / tail_anchor.ANCHOR_FILENAME
    ctime_ns = journal.stat().st_ctime_ns

    os.utime(record, ns=(ctime_ns, ctime_ns))
    assert journal.stat().st_ctime_ns == record.stat().st_mtime_ns
    assert tail_anchor.proves_absent(directory), "validate_anchor alone still says absent"
    assert not desktop_sessions._checkpoint_proven_absent(directory)
    assert index_prewarm._rewritten_since_record(directory, journal.stat())
    calls = _count_walks(monkeypatch)
    assert await _bridge_cwd(root) == str(root.parent)
    assert calls == [FRONTEND_CHECKPOINT_CUSTOM_TYPE]

    os.utime(record, ns=(ctime_ns + 1, ctime_ns + 1))
    assert desktop_sessions._checkpoint_proven_absent(directory)
    assert not index_prewarm._rewritten_since_record(directory, journal.stat())
    calls.clear()
    assert await _bridge_cwd(root) == str(root.parent)
    assert calls == []


def test_the_chrome_prompts_are_one_literal_behind_two_spellings() -> None:
    """The prompt strings the renderers recognise live in the import-free leaf.

    ``harness.rows`` asks "is this row the harness's own words" on every replayed
    user row, in both renderers. Reading the strings straight out of ``harness.loop``
    and ``session.session`` made that question import the tool stack: 231
    ``local_operator`` modules against 5 read from the leaf — module counts, not
    timings, because the graph is what decides and the clock moves with host load.
    The producers re-export the leaf's objects, so identity, not equality, is what
    keeps one literal behind both spellings.
    """
    from local_operator.harness import chrome_prompts, loop
    from local_operator.harness.rows import harness_chrome_prompts
    from local_operator.session import session as live

    assert loop.CONNECTIVITY_CONTINUATION_PROMPT is chrome_prompts.CONNECTIVITY_CONTINUATION_PROMPT
    assert (
        loop.CONNECTIVITY_TOOL_CALL_CONTINUATION_PROMPT
        is chrome_prompts.CONNECTIVITY_TOOL_CALL_CONTINUATION_PROMPT
    )
    assert loop.CONTINUE_TRUNCATED_ANSWER_PROMPT is chrome_prompts.CONTINUE_TRUNCATED_ANSWER_PROMPT
    assert loop.LENGTH_STOP_RECOVERY_PROMPT is chrome_prompts.LENGTH_STOP_RECOVERY_PROMPT
    assert (
        loop.DEGENERATE_STREAM_RECOVERY_PROMPT is chrome_prompts.DEGENERATE_STREAM_RECOVERY_PROMPT
    )
    assert loop.NAMELESS_CALL_RECOVERY_PROMPT is chrome_prompts.NAMELESS_CALL_RECOVERY_PROMPT
    assert loop.WORDLESS_STOP_RECOVERY_PROMPT is chrome_prompts.WORDLESS_STOP_RECOVERY_PROMPT
    assert loop.is_connectivity_continuation_instruction is (
        chrome_prompts.is_connectivity_continuation_instruction
    )
    assert live._CONTINUATION_PROMPT is chrome_prompts.CONTINUATION_AFTER_COMPACTION
    # And the renderer still recognises the session's own continuation prompt.
    assert chrome_prompts.CONTINUATION_AFTER_COMPACTION in harness_chrome_prompts()


def test_recognising_harness_chrome_imports_no_session_or_tool_stack(tmp_path: Path) -> None:
    """The replayed-row decision must not pay for the live stack — in its own process.

    Fresh subprocess for the reason ``tests/unit/test_import_graph.py`` gives: pytest
    has already imported the tree here, so an in-process ``sys.modules`` check proves
    nothing. ``harness.rows`` is the module both renderers call per row.
    """
    probe = (
        "import json, sys;"
        "from local_operator.harness.rows import harness_chrome_prompts, is_harness_chrome;"
        "ok = is_harness_chrome(harness_chrome_prompts()[1]);"
        "print(json.dumps({'ok': ok, 'modules': sorted(sys.modules)}))"
    )
    home = tmp_path / "home"
    home.mkdir()
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(home),
        "LOCAL_OPERATOR_CONFIG_DIR": str(tmp_path / "config"),
    }
    proc = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, cwd=str(REPO), env=env
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["ok"] is True, "the probe must really recognise a chrome prompt"
    offenders = _offenders(result["modules"])
    assert offenders == [], (
        "recognising harness chrome imported the live session/tool stack: " f"{offenders[:5]}"
    )


# --- L10: absence proven by scan when no sidecar exists ----------------------


def _checkpoint_row(cwd: str, *, separators: tuple[str, str] = (",", ":")) -> bytes:
    """A checkpoint row as a writer with ``separators`` would persist it."""
    row = {
        "id": "row-checkpoint",
        "ts": 1,
        "type": "custom",
        "payload": {
            "custom_type": FRONTEND_CHECKPOINT_CUSTOM_TYPE,
            "details": {"state": {"cwd": cwd}},
        },
    }
    return json.dumps(row, separators=separators).encode("utf-8") + b"\n"


def _tree_bytes(directory: Path) -> dict[str, bytes]:
    """Every file under ``directory``, recursively, by relative path."""
    return {
        str(p.relative_to(directory)): p.read_bytes() for p in directory.rglob("*") if p.is_file()
    }


@pytest.mark.asyncio
async def test_a_checkpointless_journal_without_a_sidecar_skips_the_walk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The L10 path: no sidecar, no checkpoint row → the scan answers, not the walk.

    The control is the same open with the scan declined: the walk runs and the
    answer is identical. Nothing is written beside the journal — the sidecar
    stays prewarm's to publish.
    """
    root = tmp_path / "store"
    directory = root / "sessions" / SESSION_ID
    directory.mkdir(parents=True)
    await _journal(directory)
    assert not desktop_sessions._checkpoint_proven_absent(directory)
    before = _tree_bytes(directory)
    calls = _count_walks(monkeypatch)
    answer = await _bridge_cwd(root)
    assert calls == []
    assert answer == str(root.parent)

    with monkeypatch.context() as scoped:
        scoped.setattr(desktop_sessions, "custom_type_proven_absent", lambda *_a, **_k: False)
        assert await _bridge_cwd(root) == answer
    assert calls == [FRONTEND_CHECKPOINT_CUSTOM_TYPE]
    assert _tree_bytes(directory) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("separators", [(",", ":"), (", ", ": ")], ids=["compact", "spaced"])
async def test_a_checkpoint_row_under_either_separator_is_still_walked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, separators: tuple[str, str]
) -> None:
    """A real checkpoint row near BOF, compact or ``json.dumps``-default spaced:
    the scan must see the spelling and leave the walk to answer the row's cwd."""
    root = tmp_path / "store"
    directory = root / "sessions" / SESSION_ID
    directory.mkdir(parents=True)
    launched = str(tmp_path / "launched-here")
    journal = directory / "transcript.jsonl"
    journal.write_bytes(_checkpoint_row(launched, separators=separators))
    await _journal(directory)
    assert not transcript_module.custom_type_proven_absent(
        directory, FRONTEND_CHECKPOINT_CUSTOM_TYPE
    )
    calls = _count_walks(monkeypatch)
    assert await _bridge_cwd(root) == launched
    assert calls == [FRONTEND_CHECKPOINT_CUSTOM_TYPE]


@pytest.mark.asyncio
async def test_a_spelling_only_inside_message_text_still_walks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The scan proves absence, never presence: a user message QUOTING the
    spelling is not a checkpoint row, so the walk runs and answers the fallback."""
    root = tmp_path / "store"
    directory = root / "sessions" / SESSION_ID
    directory.mkdir(parents=True)
    await _journal(directory)
    quoted = '{"custom_type":"%s"}' % FRONTEND_CHECKPOINT_CUSTOM_TYPE
    await Transcript(directory, defer_materialise=False).append_message(Message.user(quoted))
    # The row text is JSON-escaped inside the message, so plant the raw spelling
    # the way a tool result carrying unescaped bytes would land it in a string.
    with (directory / "transcript.jsonl").open("ab") as handle:
        handle.write(
            json.dumps(
                {"id": "text", "ts": 2, "type": "note", "payload": {"text": "x"}},
                separators=(",", ":"),
            ).encode()[:-1]
            + b',"raw":{"custom_type":"'
            + FRONTEND_CHECKPOINT_CUSTOM_TYPE.encode()
            + b'"}}\n'
        )
    assert not transcript_module.custom_type_proven_absent(
        directory, FRONTEND_CHECKPOINT_CUSTOM_TYPE
    )
    calls = _count_walks(monkeypatch)
    assert await _bridge_cwd(root) == str(root.parent)
    assert calls == [FRONTEND_CHECKPOINT_CUSTOM_TYPE]


@pytest.mark.asyncio
async def test_an_archived_journal_declines_the_scan_and_walks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An archive has no plain bytes to scan: decline, and the walk (which
    materialises the archive) answers exactly as before."""
    from local_operator.session import compression

    root = tmp_path / "store"
    directory = root / "sessions" / SESSION_ID
    directory.mkdir(parents=True)
    launched = str(tmp_path / "launched-here")
    await _journal(directory, checkpoint_cwd=launched)
    outcome = compression.compress_session(directory)
    assert outcome.status == "compressed", outcome
    assert not transcript_module.custom_type_proven_absent(
        directory, FRONTEND_CHECKPOINT_CUSTOM_TYPE
    )
    calls = _count_walks(monkeypatch)
    assert await _bridge_cwd(root) == launched
    assert calls == [FRONTEND_CHECKPOINT_CUSTOM_TYPE]


def test_the_scan_declines_past_its_cap_and_on_a_missing_journal(tmp_path: Path) -> None:
    directory = tmp_path / "session"
    directory.mkdir()
    assert not transcript_module.custom_type_proven_absent(
        directory, FRONTEND_CHECKPOINT_CUSTOM_TYPE
    )
    (directory / "transcript.jsonl").write_bytes(b'{"id":"a","ts":1,"type":"note"}\n' * 4)
    assert transcript_module.custom_type_proven_absent(directory, FRONTEND_CHECKPOINT_CUSTOM_TYPE)
    assert not transcript_module.custom_type_proven_absent(
        directory, FRONTEND_CHECKPOINT_CUSTOM_TYPE, max_scan_bytes=16
    )
    # A type with no fixed byte spelling cannot be proven absent by bytes.
    assert not transcript_module.custom_type_proven_absent(directory, 'quote"type')


@pytest.mark.asyncio
async def test_a_desktop_marker_never_reaches_the_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "store"
    directory = root / "sessions" / SESSION_ID
    directory.mkdir(parents=True)
    await _journal(directory)
    marked = str(tmp_path / "marked")
    (directory / "desktop.json").write_text(json.dumps({"version": 1, "cwd": marked}))
    scans: list[object] = []
    monkeypatch.setattr(
        desktop_sessions, "custom_type_proven_absent", lambda *a, **_k: scans.append(a) or False
    )
    calls = _count_walks(monkeypatch)
    assert await _bridge_cwd(root) == marked
    assert scans == [] and calls == []
