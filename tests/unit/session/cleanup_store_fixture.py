"""A synthetic session store with the real on-disk shapes, for the cleanup tests.

Shared by ``test_cleanup_roster_ledger.py`` (the evidence for the in-flight roster
rule and the pin guard) and by the measurement script quoted in the PR, so the
store the tests prove things about is the store the timings were taken on.

Every class the delegated pass has an opinion about is present, once or many
times, with the EXPECTED outcome stated beside it. Nothing here touches a real
store: callers hand in a ``tmp_path`` root.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from local_operator.session.cleanup import mark_store
from local_operator.session.delegated_retention import ROSTER_NAME

HOUR = 3600.0
#: Past the 48 h window by a comfortable margin, so no test depends on the clock edge.
OLD_H = 100.0

_GIT_ENV = {
    "GIT_CONFIG_SYSTEM": "/dev/null",
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@t",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@t",
}


@dataclass
class Store:
    """The built store plus what each class of directory must come to."""

    root: Path
    now: float
    #: Session ids the delegated RECORD pass must remove.
    removable: set[str] = field(default_factory=set)
    #: Of ``removable``, the ones only the in-flight rule frees (the old rule kept
    #: every one of them as "parent session still active").
    freed_by_the_roster_fix: set[str] = field(default_factory=set)
    #: ``name -> substring of the keep reason`` for every record the pass must keep.
    kept: dict[str, str] = field(default_factory=dict)
    #: Directories outside the delegated class that must come through untouched.
    survives: set[str] = field(default_factory=set)
    #: Pads whose CONTENT the pass reclaims although their record is kept.
    content: set[str] = field(default_factory=set)
    #: Directories whose permissions the test must restore before cleanup.
    locked: list[Path] = field(default_factory=list)

    @property
    def sessions(self) -> Path:
        return self.root / "sessions"

    def restore(self) -> None:
        for path in self.locked:
            os.chmod(path, 0o755)


def _session(
    store: Store, name: str, origin: str | None = "subagent", age_h: float | None = OLD_H
) -> Path:
    directory = store.sessions / name
    directory.mkdir(parents=True, exist_ok=True)
    if origin is not None:
        (directory / "origin.json").write_text(json.dumps({"origin": origin}))
    if age_h is not None:
        transcript = directory / "transcript.jsonl"
        transcript.write_text('{"type":"message"}\n')
        stamp = store.now - age_h * HOUR
        os.utime(transcript, (stamp, stamp))
    return directory


def _record(store: Store, child: str, **fields: object) -> dict[str, object]:
    """One ``records[]`` row in the shape ``session.py`` writes."""
    row: dict[str, object] = {
        "job_id": child[:12],
        "label": "x",
        "session_dir": str(store.sessions / child),
        "outcome": "completed",
        "result_text": "done",
        "paused": False,
        "settled_at": store.now - 90 * HOUR,
    }
    row.update(fields)
    return row


def _roster(directory: Path, records: Sequence[object]) -> None:
    (directory / ROSTER_NAME).write_text(json.dumps({"version": 1, "records": records}))


#: How old an IDLE pad's content is in these fixtures: past every grace and
#: window the content layer uses, so an unadjusted pad is the only thing that
#: looks active.
IDLE_PAD_AGE_S = 100 * HOUR


def idle_pads(root: Path, *, now: float, age_s: float = IDLE_PAD_AGE_S) -> None:
    """Backdate every scratchpad entry under ``root`` to ``now - age_s``.

    A fixture that writes pad content stamps it with the wall clock, and the
    content layer's whole-pad recency floor reads exactly that — correctly: a
    file written a second ago IS active work. A test that means "an idle pad"
    therefore has to say so, and this is where it says it. FILES AND
    DIRECTORIES, deepest first, because creating a file bumps its parent
    directory's mtime and a fresh directory keeps the pad just as effectively as
    a fresh file.

    Deliberately a function the fixture calls, not an autouse side effect: a
    test that wants a genuinely fresh pad simply does not call it, and the
    difference is visible in the test.
    """
    stamp = now - age_s
    for pad in sorted((root / "sessions").glob("*/scratchpad")):
        for path in sorted(pad.rglob("*"), key=lambda entry: len(entry.parts), reverse=True):
            try:
                os.utime(path, (stamp, stamp), follow_symlinks=False)
            except OSError:
                continue
        try:
            os.utime(pad, (stamp, stamp), follow_symlinks=False)
        except OSError:
            continue


def _stale_file(path: Path, size_mb: int = 11, age_h: float = 24.0, now: float = 0.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.seek(size_mb * 1024 * 1024 - 1)
        handle.write(b"\0")
    stamp = now - age_h * HOUR
    os.utime(path, (stamp, stamp))
    return path


def _git(directory: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(directory), *args],
        check=True,
        capture_output=True,
        env={**os.environ, **_GIT_ENV},
    )


def build_store(
    root: Path,
    *,
    settled: int = 120,
    orphans: int = 60,
    filler: int = 100,
    now: float | None = None,
) -> Store:
    """Build ~``settled + orphans + filler + 30`` directories under ``root/sessions``."""
    store = Store(root=root, now=time.time() if now is None else now)
    mark_store(store.sessions)

    # -- A long-lived PARENT (unmarked: the user's own conversation, active an hour
    # ago) whose roster is its lifetime ledger. Every row below is one record.
    parent = _session(store, "parent", origin=None, age_h=1)
    records: list[object] = []

    for index in range(settled):
        name = f"settled{index:03d}"
        _session(store, name)
        records.append(_record(store, name))
        store.removable.add(name)
        store.freed_by_the_roster_fix.add(name)

    # Cancelled and failed children are history too.
    for name, outcome in (("cancelled0", "cancelled"), ("failed0", "failed")):
        _session(store, name)
        records.append(_record(store, name, outcome=outcome))
        store.removable.add(name)
        store.freed_by_the_roster_fix.add(name)

    # Several records for ONE child, one of them settled: any in-flight record wins.
    _session(store, "twice")
    records.append(_record(store, "twice"))
    records.append(_record(store, "twice", settled_at=None, outcome=""))
    store.kept["twice"] = "parent session still active"

    # In flight (no settled_at), field missing entirely, paused-yet-settled, and
    # fields that cannot be read: all protected, fail closed.
    _session(store, "inflight")
    records.append(_record(store, "inflight", settled_at=None, outcome=""))
    store.kept["inflight"] = "parent session still active"

    _session(store, "nofield")
    row = _record(store, "nofield")
    del row["settled_at"]
    records.append(row)
    store.kept["nofield"] = "parent session still active"

    # THE CASE THE CORRECTION NAMES: 44 records on the operator's store are
    # settled AND paused, and `hub` resumes a paused child.
    _session(store, "paused")
    records.append(_record(store, "paused", paused=True))
    store.kept["paused"] = "parent session still active"

    for name, bad in (
        ("settled-str", "yesterday"),
        ("settled-bool", True),
        ("settled-nan", float("nan")),
    ):
        _session(store, name)
        records.append(_record(store, name, settled_at=bad))
        store.kept[name] = "parent session still active"

    _session(store, "paused-str")
    records.append(_record(store, "paused-str", paused="false"))  # not a bool: not evidence
    store.kept["paused-str"] = "parent session still active"

    # A record naming a child with no directory at all protects nothing and breaks nothing.
    records.append(_record(store, "ghost-never-existed"))

    # TRANSITIVE: an in-flight child is a parent in its own right.
    child = _session(store, "inflight-parent", age_h=OLD_H)
    records.append(_record(store, "inflight-parent", settled_at=None, outcome=""))
    store.kept["inflight-parent"] = "parent session still active"
    _session(store, "gc-settled")
    _session(store, "gc-inflight")
    _roster(
        child,
        [
            _record(store, "gc-settled"),
            _record(store, "gc-inflight", settled_at=None, outcome=""),
        ],
    )
    store.removable.add("gc-settled")
    store.freed_by_the_roster_fix.add("gc-settled")
    store.kept["gc-inflight"] = "parent session still active"

    # A pad-heavy child under an in-flight record: the RECORD stays, the stale
    # build output inside its pad is what the content layer reclaims.
    heavy = _session(store, "padheavy")
    records.append(_record(store, "padheavy", settled_at=None, outcome=""))
    store.kept["padheavy"] = "parent session still active"
    _stale_file(heavy / "scratchpad" / "desktop-suite.log", now=store.now)
    store.content.add("padheavy")

    _roster(parent, records)

    # -- Delegated sessions NO roster names.
    for index in range(orphans):
        name = f"orphan{index:03d}"
        _session(store, name)
        store.removable.add(name)

    # Pinned: named by no roster, so only the pin keeps it.
    _session(store, "pinned")
    store.kept["pinned"] = "pinned by the operator"
    (store.root / "sidebar-pins.json").write_text(json.dumps(["pinned"]))

    # Live claim (this process is alive), held, armed wake, armed monitor.
    live = _session(store, "claimed")
    (live / ".session.pid").write_text(str(os.getpid()))
    store.kept["claimed"] = "claimed by a live process"

    _session(store, "held")
    from local_operator.session.cleanup import set_hold

    assert set_hold(store.sessions, "held", reason="evidence", by="operator", kind="evidence")
    store.kept["held"] = "held by the operator"

    _session(store, "wake")
    (store.root / "wakes").mkdir(exist_ok=True)
    (store.root / "wakes" / "wake.json").write_text(json.dumps({"schedules": [{"id": "w1"}]}))
    store.kept["wake"] = "armed wake"

    _session(store, "monitor")
    (store.root / "monitors").mkdir(exist_ok=True)
    (store.root / "monitors" / "monitor.json").write_text(
        json.dumps({"monitors": [{"id": "m1", "every_ms": 60000}]})
    )
    store.kept["monitor"] = "armed monitor"

    # A scratchpad git repo with unpushed work, and one nothing can inspect.
    _session(store, "githazard")
    repo = store.sessions / "githazard" / "scratchpad" / "repo"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q")
    (repo / "f").write_text("x")
    store.kept["githazard"] = "scratchpad git repo has uncommitted/unpushed work"

    _session(store, "unreadablepad")
    pad = store.sessions / "unreadablepad" / "scratchpad"
    pad.mkdir()
    (pad / "f").write_text("x")
    os.chmod(pad, 0o000)
    store.locked.append(pad)
    store.kept["unreadablepad"] = "scratchpad git repo has uncommitted/unpushed work"

    # Origins retention never reaps (and `fork` / user sessions are not delegated).
    _session(store, "workstream", origin="agent-workstream")
    _session(store, "agentconfig", origin="agent-config")
    # agent-config is hidden-but-never-reapable (reported as such); agent-workstream
    # is a LISTED origin, so it is the user class and the delegated pass never
    # looks at it. Both must simply survive.
    store.kept["agentconfig"] = "origin not reapable"
    store.survives.add("workstream")

    # Filler: the user's own old conversations (never the delegated class) and
    # delegated sessions still inside the window.
    for index in range(filler):
        _session(store, f"user{index:03d}", origin=None, age_h=OLD_H + index)
        store.survives.add(f"user{index:03d}")
    store.survives.add("parent")
    for index in range(20):
        _session(store, f"fresh{index:03d}", age_h=2)
        store.kept[f"fresh{index:03d}"] = "inside the 48h window"
    return store
