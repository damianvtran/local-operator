"""A moved session keeps its monitors, projects, team brief and queued move.

Regression tests for the 2026-11-13 Mac → lop-remote-node-2 move of
ce6fcb93c28a, which reported four parity bugs:

(a) carried monitors kept ``cwd=/Users/damian`` and failed their first check;
(b) the linked project did not exist on the destination's per-device store;
(c) the team brief was missing on the destination and nothing said so (and the
    queued path never pushed definitions at all);
(d) a ``--queue`` move stayed ``queued`` after its session went idle.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import carry
from local_operator.network import move_queue as mq
from local_operator.projects import ProjectEdit, ProjectRegistry

SID = "ce6fcb93c28a"


# -- (a) monitor cwds --------------------------------------------------------


class TestMonitorCwds:
    def test_the_source_home_maps_to_this_home(self, tmp_path: Path) -> None:
        home = tmp_path / "home" / "ec2-user"
        home.mkdir(parents=True)
        assert carry.translate_host_path("/Users/nobody-here", home=str(home)) == str(home)

    def test_a_path_under_the_source_home_maps_by_tail(self, tmp_path: Path) -> None:
        home = tmp_path / "h"
        (home / "local-operator" / "sub").mkdir(parents=True)
        mapped = carry.translate_host_path("/Users/nobody-here/local-operator/sub", home=str(home))
        assert mapped == str(home / "local-operator" / "sub")

    def test_rows_are_translated_or_fall_back_and_counted(self, tmp_path: Path) -> None:
        home = tmp_path / "h"
        home.mkdir()
        session_cwd = str(tmp_path / "session-cwd")
        existing = str(tmp_path)
        rows, counts = carry.translate_monitor_cwds(
            [
                {"id": "m15", "cwd": "/Users/nobody-here"},
                {"id": "m16", "cwd": "/opt/never/exists/anywhere-xyz"},
                {"id": "m17", "cwd": existing},
                {"id": "m18", "cwd": ""},
            ],
            session_cwd=session_cwd,
            home=str(home),
        )
        assert [row["cwd"] for row in rows] == [str(home), session_cwd, existing, ""]
        assert counts == {"translated": 1, "fallback": 1}
        # Idempotent: a second pass over the translated rows changes nothing.
        again, counts_again = carry.translate_monitor_cwds(
            [r for r in rows if r["cwd"] != session_cwd],
            session_cwd=session_cwd,
            home=str(home),
        )
        assert counts_again == {"translated": 0, "fallback": 0}


# -- (b) projects ------------------------------------------------------------


def _make_project(root: Path, name: str, *, sessions: list[str], progress: str) -> None:
    registry = ProjectRegistry(root)
    project = registry.create_project(ProjectEdit(name=name), sessions=sessions)
    registry.update_project(project.id, ProjectEdit(progress=progress), reporter=SID)


class TestProjectCarry:
    def test_a_linked_project_is_created_on_the_destination(self, tmp_path: Path) -> None:
        source, dest = tmp_path / "src", tmp_path / "dst"
        _make_project(
            source,
            "secret-guardrails-relax",
            sessions=[SID, "aaaaaaaaaaaa"],
            progress="p1",
        )
        rows = carry.linked_project_rows(source, SID)
        assert [row["name"] for row in rows] == ["secret-guardrails-relax"]
        # The wire carries JSON: prove the rows survive a round trip.
        rows = json.loads(json.dumps(rows))

        block = carry.adopt_projects(dest, SID, rows)

        assert block == {"created": ["secret-guardrails-relax"]}
        adopted = ProjectRegistry(dest).get_project_by_name("secret-guardrails-relax")
        assert adopted is not None
        # Only the moved session is linked here: the other id lives on the source.
        assert adopted.sessions == [SID]
        assert adopted.progress == "p1"
        assert [entry.text for entry in adopted.updates] == ["p1"]
        # The source row is never touched (other sessions may link it).
        kept = ProjectRegistry(source).get_project_by_name("secret-guardrails-relax")
        assert kept is not None and SID in kept.sessions

    def test_a_same_named_row_is_merged_not_duplicated(self, tmp_path: Path) -> None:
        source, dest = tmp_path / "src", tmp_path / "dst"
        _make_project(dest, "secret-guardrails-relax", sessions=[], progress="dest-old")
        _make_project(source, "secret-guardrails-relax", sessions=[SID], progress="src-new")
        rows = json.loads(json.dumps(carry.linked_project_rows(source, SID)))

        block = carry.adopt_projects(dest, SID, rows)

        assert block == {"merged": ["secret-guardrails-relax"]}
        registry = ProjectRegistry(dest)
        assert len(registry.list_projects()) == 1
        merged = registry.get_project_by_name("secret-guardrails-relax")
        assert merged is not None and merged.sessions == [SID]
        assert [entry.text for entry in merged.updates] == ["dest-old", "src-new"]
        assert merged.progress == "src-new"
        # A second adopt is a no-op merge, never a duplicated history.
        carry.adopt_projects(dest, SID, rows)
        again = ProjectRegistry(dest).get_project_by_name("secret-guardrails-relax")
        assert again is not None and len(again.updates) == 2

    def test_nothing_carried_reports_nothing(self, tmp_path: Path) -> None:
        assert carry.linked_project_rows(tmp_path, SID) == []
        assert carry.adopt_projects(tmp_path, SID, []) == {}


# -- (c) team brief ----------------------------------------------------------


class TestTeamResolution:
    def _session(self, root: Path, team: str) -> Path:
        directory = root / "sessions" / SID
        directory.mkdir(parents=True)
        (directory / "attachment.json").write_text(
            json.dumps({"team": team, "agent": "", "goal": ""}), encoding="utf-8"
        )
        return directory

    def test_an_unresolvable_team_is_named_on_the_receipt(self, tmp_path: Path) -> None:
        block = carry.attachment_resolution(tmp_path, self._session(tmp_path, "lopdev"))
        assert block["name"] == "lopdev" and block["resolved"] is False
        assert "team 'lopdev' is not defined on this device" in block["notice"]

    def test_a_resolvable_team_reports_resolved(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from local_operator import teams

        monkeypatch.setattr(
            teams.TeamRegistry, "get_team_by_name", lambda self, name, **kw: object()
        )
        block = carry.attachment_resolution(tmp_path, self._session(tmp_path, "lopdev"))
        assert block == {"name": "lopdev", "resolved": True}

    def test_no_attachment_reports_nothing(self, tmp_path: Path) -> None:
        assert carry.attachment_resolution(tmp_path, tmp_path) == {}

    def test_the_queued_copy_pushes_definitions_before_the_invite(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from local_operator.network import mobility

        mq.enqueue(
            tmp_path,
            SID,
            to_device="d_b",
            to_name="node-2",
            request_id="r",
            requested_by="d_a",
        )
        mq.transition(tmp_path, SID, mq.QUEUE_PHASE_COPYING, writer="relay")
        order: list[str] = []
        monkeypatch.setattr(mobility, "_resolve_typed_peer", lambda server, name: ("d_b", "node-2"))
        monkeypatch.setattr(mobility, "_link_for_move", lambda server, dev, name: object())
        monkeypatch.setattr(
            mobility,
            "_push_move_definitions",
            lambda server, sid, dev, label: order.append(f"push:{sid}:{dev}") or "",
        )

        class _Transport:
            def __init__(self, *args: Any) -> None:
                pass

            def ask(self, frame: dict[str, Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
                order.append(f"invite:{frame.get('phase')}")
                raise RuntimeError("stop after the invite")

        monkeypatch.setattr(mobility, "LinkTransport", _Transport)

        class _Server:
            root = tmp_path

        mq._drive_copy(_Server(), SID)  # noqa: SLF001
        assert order[:2] == [f"push:{SID}:d_b", "invite:invite"]


# -- (d) the queued move of an idle session ----------------------------------


class _Runtime:
    pid = 4242

    def __init__(self) -> None:
        from local_operator.session.runtime.types import QUEUED_MOVE_CAPABILITY

        self.capabilities = (QUEUED_MOVE_CAPABILITY,)


class TestIdleQueuedMove:
    def test_an_idle_runtime_that_cannot_take_the_intent_still_moves(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from local_operator.network import mobility

        mq.enqueue(
            tmp_path,
            SID,
            to_device="d_b",
            to_name="node-2",
            request_id="r",
            requested_by="d_a",
        )
        # The incident's shape: a runtime is there (its window is attached, so it
        # never disappears) and every delivery comes back unreachable.
        monkeypatch.setattr(mq, "_find_runtime", lambda root, sid: _Runtime())
        monkeypatch.setattr(mq, "_deliver_intent", lambda server, sid: "unreachable")
        answers = iter(["busy", "busy", "retired"])
        probes: list[str] = []

        def _retire(root: Path, sid: str, **kw: Any) -> dict[str, Any]:
            result = next(answers, "retired")
            probes.append(result)
            return {"result": result, "sentence": ""}

        monkeypatch.setattr(mobility, "_retire_local_runtime", _retire)
        copied: list[str] = []
        monkeypatch.setattr(mq, "_drive_copy", lambda server, sid: copied.append(sid) or "resumed")
        monkeypatch.setattr(mq, "DRIVER_POLL_S", 0.01)
        monkeypatch.setattr(mq, "DRIVER_WAIT_S", 0.01)

        class _Server:
            root = tmp_path

        worker = threading.Thread(target=mq.drive, args=(_Server(), SID), daemon=True)
        worker.start()
        worker.join(timeout=10)
        assert not worker.is_alive()
        # A busy runtime was never retired (no turn cut); the idle one was.
        assert probes == ["busy", "busy", "retired"]
        assert copied == [SID]
        record = mq.read_record(tmp_path, SID) or {}
        assert record["phase"] == mq.QUEUE_PHASE_COPYING

    def test_a_cancel_that_won_is_never_overridden_by_the_idle_retire(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from local_operator.network import mobility

        mq.enqueue(
            tmp_path,
            SID,
            to_device="d_b",
            to_name="node-2",
            request_id="r",
            requested_by="d_a",
        )
        mq.cancel(tmp_path, SID)
        called: list[str] = []
        monkeypatch.setattr(
            mobility, "_retire_local_runtime", lambda *a, **k: called.append("x") or {}
        )
        assert mq._retire_if_idle(tmp_path, SID) is False  # noqa: SLF001
        assert called == []


class TestSessionLoadTranslatesMonitorCwds:
    def test_a_carried_monitor_runs_in_a_directory_this_host_has(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The session's own load applies the promote's rule: the transcript keeps
        the source host's ``/Users/damian`` and the armed spec gets this home."""
        from types import SimpleNamespace

        from local_operator.monitors.spec import MIN_MONITOR_INTERVAL_MS
        from local_operator.session.session import Session

        home = tmp_path / "home" / "ec2-user"
        home.mkdir(parents=True)
        monkeypatch.setenv("HOME", str(home))
        row = {
            "id": "m15",
            "name": "pr",
            "tool": "bash",
            "arguments": {"command": "gh pr view 1"},
            "every_ms": MIN_MONITOR_INTERVAL_MS,
            "cwd": "/Users/nobody-here",
        }
        entry = SimpleNamespace(ts=1.0, payload={"details": {"monitors": [row], "next_seq": 18}})
        loaded: list[Any] = []
        fake = SimpleNamespace(
            _transcript=SimpleNamespace(
                latest_custom_entry=lambda kind: entry, directory=tmp_path / "s"
            ),
            _cwd=str(tmp_path),
            _monitors=SimpleNamespace(load=lambda specs, next_seq=None: loaded.extend(specs)),
        )
        Session._load_monitor_schedules(fake)  # type: ignore[arg-type]
        assert [spec.cwd for spec in loaded] == [str(home)]
        # The transcript's own row is untouched (the move verified those bytes).
        assert row["cwd"] == "/Users/nobody-here"


class TestRebuiltIndexTranslatesMonitorCwds:
    def test_the_promote_index_names_a_cwd_this_host_has_and_reports_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from local_operator.monitors import store as monitor_store
        from tests.unit.network.test_carry import MONITOR_ROW, _custom_entry

        home = tmp_path / "home" / "ec2-user"
        home.mkdir(parents=True)
        monkeypatch.setenv("HOME", str(home))
        directory = tmp_path / "sessions" / SID
        directory.mkdir(parents=True)
        rows = [
            dict(MONITOR_ROW, id="m15", cwd="/Users/nobody-here"),
            dict(MONITOR_ROW, id="m16", cwd="/opt/never/exists/anywhere-xyz"),
        ]
        (directory / "transcript.jsonl").write_text(
            _custom_entry("monitor_schedules", {"monitors": rows}) + "\n", encoding="utf-8"
        )
        (directory / "desktop.json").write_text(json.dumps({"cwd": str(home)}), encoding="utf-8")

        report = carry.rebuild_indexes(tmp_path, SID, session_dir=directory)

        assert report["monitor_cwds"] == {"translated": 1, "fallback": 1}
        index = monitor_store.read_entry(tmp_path, SID) or {}
        assert [row["cwd"] for row in index["monitors"]] == [str(home), str(home)]
