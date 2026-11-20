"""What already travels with a move — pinned END TO END (design §2, operator 2026-11-18).

The operator asked that "projects, worktrees, working files, scratch, session memory,
etc." all carry over. The design's §2 inventory found that, of those, only the
working directories did not; this cell is the evidence for the rest, in ONE real move
over the in-process two-relay rig (the CLI's own hop, ``mobility.request_move``):

* the scratchpad tree (``sync.COPY_SET_TREES``),
* session memory (``memory-session.jsonl`` in ``sync.COPY_SET_NAMES``),
* todos (``todo_snapshot`` custom entries ride the transcript),
* linked projects (``carry.linked_project_rows`` → ``carry.adopt_projects``).

Each was pinned somewhere at unit level before; nothing pinned them TOGETHER on the
move path, so a copy-set change could drop one while every unit test stayed green.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from local_operator.network import mobility
from local_operator.projects import ProjectEdit, ProjectRegistry
from tests.unit.network.test_mobility import (  # noqa: F401 — `devices` is a fixture
    SESSION,
    Devices,
    _owned_session,
    devices,
)
from tests.unit.network.test_relay_e2e import _pair_settled


@pytest.fixture()
def pair(request: pytest.FixtureRequest) -> Devices:
    value: Devices = request.getfixturevalue("devices")
    _server_a, server_b, _host, _port = value
    server_b.bind_control()
    server_b.start()
    return value


def test_scratch_memory_todos_and_projects_arrive_with_the_moved_session(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    server_a, server_b, _host, _port = pair
    _pair_settled(pair, monkeypatch, role="admin")
    source = _owned_session(server_a)
    (source / "scratchpad" / "notes").mkdir(parents=True)
    (source / "scratchpad" / "notes" / "plan.md").write_text("# plan\n", encoding="utf-8")
    memory_row = {"id": "m1", "text": "the operator prefers patch bumps", "scope": "session"}
    (source / "memory-session.jsonl").write_text(json.dumps(memory_row) + "\n", encoding="utf-8")
    todo_row: dict[str, Any] = {
        "id": "t1",
        "type": "custom",
        "customType": "todo_snapshot",
        "details": {"phases": [{"phase": "Todos", "items": [{"text": "ship A", "done": False}]}]},
    }
    with (source / "transcript.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(todo_row) + "\n")
    registry = ProjectRegistry(server_a.root)
    registry.create_project(ProjectEdit(name="move-working-set"), sessions=[SESSION])

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_b.root))
    moved = dict(mobility.request_move(SESSION, to="local", root=server_b.root))

    assert moved["ok"] is True, moved
    arrived = server_b.root / "sessions" / SESSION
    assert (arrived / "scratchpad" / "notes" / "plan.md").read_text(encoding="utf-8") == "# plan\n"
    assert json.loads((arrived / "memory-session.jsonl").read_text(encoding="utf-8")) == memory_row
    rows = [
        json.loads(line)
        for line in (arrived / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert todo_row in rows
    adopted = ProjectRegistry(server_b.root).get_project_by_name("move-working-set")
    assert adopted is not None and adopted.sessions == [SESSION]
    # No worktree here, so the receipt carries no working-set block at all.
    assert "working_set" not in moved
