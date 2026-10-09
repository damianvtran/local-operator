"""The live seam: what a tool result writes, and how a child's open reaches its parent.

The hook is the only writer that runs INSIDE a turn, so every test here is about what it
must not do as much as what it must: never raise into the turn, never touch a session
without a transcript, and always label a propagated event with the child that produced it.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from local_operator.code_requests import hook
from local_operator.code_requests.ledger import EVENT_CUSTOM_TYPE
from local_operator.code_requests.refs import HostContext, Remote

CWD = HostContext(remotes=(Remote("origin", "github.com", "damianvtran/local-operator"),))
CALL = {"command": "gh pr create --repo damianvtran/local-operator --title t"}
RESULT = (
    "exit code: 0\n--- stdout ---\n"
    "https://github.com/damianvtran/local-operator/pull/1904\n\n--- stderr ---\n(empty)"
)


class FakeTranscript:
    def __init__(self, directory: Path | None) -> None:
        self.directory = directory
        self.rows: list[tuple[str, dict[str, Any]]] = []

    async def append_custom(self, custom_type: str, details: dict[str, Any]) -> None:
        self.rows.append((custom_type, details))
        if self.directory is not None:
            with (self.directory / "transcript.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"custom_type": custom_type, "details": details}) + "\n")


class FakeSession:
    def __init__(self, directory: Path | None, **attrs: Any) -> None:
        self._transcript = FakeTranscript(directory)
        self.session_id = attrs.pop("session_id", "abcdef123456")
        self._job_id = attrs.pop("_job_id", None)
        self._job_label = attrs.pop("_job_label", "")
        self._agent_type = attrs.pop("_agent_type", "")
        for key, value in attrs.items():
            setattr(self, key, value)


def _detections():
    return hook.classify(FakeSession(None), "bash", CALL, RESULT, is_error=False, context=CWD)


def _record(session, detections, **kwargs):
    """``record_detections`` with the tool/call the harness always has to hand."""
    return asyncio.run(
        hook.record_detections(session, detections, tool="bash", call_id="call_1", **kwargs)
    )


def test_classify_gates_before_working():
    assert hook.classify(FakeSession(None), "bash", {"command": "ls"}, "ok", is_error=False) == []
    assert hook.classify(FakeSession(None), "read", {"path": "x"}, RESULT, is_error=False) == []
    assert len(_detections()) == 1


def test_classify_never_raises_on_hostile_input():
    class Exploding(dict[str, Any]):
        def get(self, *_args, **_kwargs):
            raise RuntimeError("boom")

    assert hook.classify(FakeSession(None), "bash", Exploding(), "x", is_error=False) == []


def test_a_detection_writes_one_event_row(tmp_path):
    directory = tmp_path / "sess"
    directory.mkdir()
    session = FakeSession(directory)
    written = _record(session, _detections())
    assert written == 1
    custom_type, details = session._transcript.rows[0]
    assert custom_type == EVENT_CUSTOM_TYPE
    assert details["kind"] == "opened" and details["v"] == 1
    assert details["ref"]["number"] == 1904
    assert details["evidence"]["rule"] == "gh-pr-create-stdout"
    # A top-level session's own row carries no ``via``: nothing arrived from elsewhere.
    assert "via" not in details


def test_a_failing_writer_costs_the_row_not_the_turn():
    class Broken(FakeTranscript):
        async def append_custom(self, *_args, **_kwargs):
            raise OSError("disk full")

    session = FakeSession(None)
    session._transcript = Broken(None)
    assert _record(session, _detections()) == 0


def test_a_session_without_a_transcript_directory_writes_nothing():
    session = FakeSession(None)
    assert _record(session, _detections()) == 0
    assert session._transcript.rows == []


def test_a_child_open_propagates_to_the_parent_with_its_label(tmp_path):
    parent_dir = tmp_path / "parent"
    child_dir = tmp_path / "child"
    for directory in (parent_dir, child_dir):
        directory.mkdir()
    parent = FakeSession(parent_dir, session_id="parent123456")
    child = FakeSession(
        child_dir,
        session_id="child1234567",
        _job_id="job-1",
        _job_label="coder",
        _agent_type="coder",
    )
    hook.attach_parent(child, parent)
    _record(child, _detections())

    assert len(child._transcript.rows) == 1
    assert "via" not in child._transcript.rows[0][1]
    assert len(parent._transcript.rows) == 1
    _, propagated = parent._transcript.rows[0]
    assert propagated["propagated"] is True
    assert propagated["via"]["label"] == "coder"
    assert propagated["via"]["agent_role"] == "coder"
    assert propagated["via"]["child_session_id"] == "child1234567"
    assert propagated["via"]["path"] == ["coder"]


def test_a_deep_child_labelling_does_not_lose_the_outer_path(tmp_path):
    root = FakeSession(
        (tmp_path / "root").mkdir() or (tmp_path / "root"), session_id="root12345678"
    )
    mid_dir = tmp_path / "mid"
    mid_dir.mkdir()
    leaf_dir = tmp_path / "leaf"
    leaf_dir.mkdir()
    mid = FakeSession(
        mid_dir, session_id="mid123456789", _job_id="j1", _job_label="coder", _agent_type="coder"
    )
    hook.attach_parent(mid, root)
    leaf = FakeSession(
        leaf_dir,
        session_id="leaf12345678",
        _job_id="j2",
        _job_label="reviewer",
        _agent_type="reviewer",
    )
    hook.attach_parent(leaf, mid)

    _record(leaf, _detections())
    assert leaf._transcript.rows == [] or len(leaf._transcript.rows) == 1
    _, at_mid = mid._transcript.rows[0]
    assert at_mid["via"]["label"] == "reviewer" and at_mid["via"]["path"] == ["reviewer"]
    # The mid level's own row carries no via of its own, so its propagation starts fresh.
    _, at_root = root._transcript.rows[0]
    assert at_root["via"]["label"] == "coder"


def test_attach_parent_tolerates_an_object_that_refuses_attributes():
    class Frozen:
        __slots__ = ()

    hook.attach_parent(Frozen(), FakeSession(None))


def test_stamp_origin_parent_keeps_what_was_already_recorded(tmp_path):
    directory = tmp_path / "child"
    directory.mkdir()
    (directory / "origin.json").write_text(
        json.dumps({"origin": "subagent", "label": "coder", "agent": "coder"}), encoding="utf-8"
    )
    hook.stamp_origin_parent(directory, "parent123456")
    payload = json.loads((directory / "origin.json").read_text(encoding="utf-8"))
    assert payload == {
        "origin": "subagent",
        "label": "coder",
        "agent": "coder",
        "parent": "parent123456",
    }
    # Idempotent, and it never invents a file for a session with no marker.
    hook.stamp_origin_parent(directory, "parent123456")
    assert (
        json.loads((directory / "origin.json").read_text(encoding="utf-8"))["parent"]
        == "parent123456"
    )
    hook.stamp_origin_parent(tmp_path / "absent", "")
    assert not (tmp_path / "absent" / "origin.json").exists()


def test_an_acted_detection_is_recorded_with_its_act(tmp_path):
    directory = tmp_path / "sess"
    directory.mkdir()
    session = FakeSession(directory)
    detections = hook.classify(
        session,
        "bash",
        {"command": "gh pr merge 1904 --admin --squash"},
        RESULT,
        is_error=False,
        context=CWD,
    )
    assert _record(session, detections) == 1
    _, details = session._transcript.rows[0]
    assert details["kind"] == "acted" and details["act"] == "merge"
