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
from typing import Any, cast

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

    def test_a_missing_path_under_this_home_is_left_alone(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Review r1 m2: a non-moved session's removed worktree keeps failing loudly
        instead of silently re-arming at the session cwd."""
        home = tmp_path / "home" / "damian"
        (home / "local-operator").mkdir(parents=True)
        monkeypatch.setenv("HOME", str(home))
        gone = str(home / "local-operator" / ".worktrees" / "gone-xyz")
        rows, counts = carry.translate_monitor_cwds(
            [{"cwd": gone}], session_cwd=str(tmp_path), home=str(home)
        )
        assert rows == [{"cwd": gone}]
        assert counts == {"translated": 0, "fallback": 0}

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

    def test_an_empty_home_never_falls_back_to_the_process_home(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cwd omission census: ``home`` is the caller's explicit destination home.

        An empty one must not be replaced by the receiving process's own home —
        here a HOME holding the very tail the foreign path names, so a silent
        default WOULD map it — and must not re-arm the monitor at ``session_cwd``.
        """
        process_home = tmp_path / "process-home"
        (process_home / "proj").mkdir(parents=True)
        monkeypatch.setenv("HOME", str(process_home))
        foreign = "/Users/nobody-here/proj"
        assert carry.translate_host_path(foreign, home="") == ""
        assert carry.translate_host_path("/Users/nobody-here", home="") == ""
        rows, counts = carry.translate_monitor_cwds(
            [{"cwd": foreign}], session_cwd=str(tmp_path), home="", moved=True
        )
        assert rows == [{"cwd": foreign}]
        assert counts == {"translated": 0, "fallback": 0}
        # Control: the same inputs WITH that home named do map, so the empty-home
        # result above is the guard, not a path that could never map.
        assert carry.translate_host_path(foreign, home=str(process_home)) == str(
            process_home / "proj"
        )

    def test_home_is_a_required_argument(self) -> None:
        with pytest.raises(TypeError):
            carry.translate_host_path("/Users/nobody-here")  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            carry.translate_monitor_cwds([], session_cwd="/")  # type: ignore[call-arg]


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

    def test_the_receiver_bounds_an_oversized_history(self, tmp_path: Path) -> None:
        """Review r1 M1, the reproduced 14 MB row: 700 × 20 KB entries from a peer."""
        from local_operator.projects import PROGRESS_MAX

        row = {
            "id": "a" * 32,
            "name": "p1",
            "updates": [
                {
                    "at": f"2026-11-13T00:{i // 60:02d}:{i % 60:02d}Z",
                    "text": "x" * 20_000,
                }
                for i in range(700)
            ],
        }
        block = carry.adopt_projects(tmp_path, SID, [row])

        assert block["created"] == ["p1"] and block["truncated"] == ["p1"]
        assert f"newest {carry.CARRIED_UPDATES_MAX} entries" in block["limits"]
        stored = ProjectRegistry(tmp_path).get_project_by_name("p1")
        assert stored is not None
        assert len(stored.updates) == carry.CARRIED_UPDATES_MAX
        assert {len(entry.text) for entry in stored.updates} == {PROGRESS_MAX}
        # The NEWEST entries were kept.
        assert stored.updates[-1].at.startswith("2026-11-13T00:11:39")
        size = sum(p.stat().st_size for p in (tmp_path / "projects").rglob("*.json"))
        assert size < carry.CARRIED_PROJECT_BYTES_MAX

    def test_a_row_over_the_byte_cap_is_refused_and_said(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(carry, "CARRIED_PROJECT_BYTES_MAX", 2_000)
        rows = [
            {
                "id": "b" * 32,
                "name": "big",
                "updates": [{"at": "2026-11-13T00:00:00Z", "text": "y" * 900}] * 3,
            },
            {"id": "c" * 32, "name": "small", "updates": []},
        ]
        block = carry.adopt_projects(tmp_path, SID, rows)
        assert block["refused"] == ["big"] and block["created"] == ["small"]
        assert "big not adopted" in block["limits"]
        assert ProjectRegistry(tmp_path).get_project_by_name("big") is None

    def test_rows_past_the_count_cap_are_counted_on_the_receipt(self, tmp_path: Path) -> None:
        rows = [{"id": f"{i:032x}", "name": f"p{i}", "updates": []} for i in range(10)]
        block = carry.adopt_projects(tmp_path, SID, rows)
        assert len(block["created"]) == carry.CARRIED_PROJECTS_MAX
        assert "2 more linked projects not carried" in block["limits"]

    def test_a_merge_keeps_the_history_cap(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The merge path applies the append's text-only UPDATES_MAX trim."""
        from local_operator import projects

        _make_project(tmp_path, "p1", sessions=[], progress="dest-old")
        monkeypatch.setattr(projects, "UPDATES_MAX", 3)
        row = {
            "id": "d" * 32,
            "name": "p1",
            # Dated in the future so the carried entries sort AFTER today's "dest-old".
            "updates": [{"at": f"2999-01-01T00:00:0{i}Z", "text": f"t{i}"} for i in range(5)],
        }
        assert carry.adopt_projects(tmp_path, SID, [row])["merged"] == ["p1"]
        stored = ProjectRegistry(tmp_path).get_project_by_name("p1")
        assert stored is not None and [e.text for e in stored.updates] == [
            "t2",
            "t3",
            "t4",
        ]
        # The oldest (text-only) entry, dest-old, was the one trimmed.

    def test_a_full_project_is_named_unlinked_not_merged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Review r1 N2: no silent "merged" when the session could not be linked."""
        from local_operator import projects

        _make_project(tmp_path, "p1", sessions=["aaaaaaaaaaaa"], progress="dest-old")
        monkeypatch.setattr(projects, "SESSIONS_MAX", 1)
        block = carry.adopt_projects(tmp_path, SID, [{"id": "e" * 32, "name": "p1", "updates": []}])
        assert block["unlinked"] == ["p1"] and "merged" not in block
        assert "could not be linked to p1" in block["limits"]


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
        assert block == {"name": "lopdev", "resolved": True, "applied": True}

    def test_an_incomplete_registry_reports_not_applied(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Review r1 m4: the restore attaches nothing while the profile registry is
        incomplete, so a resolving name must not read as an applied brief."""
        from local_operator import agents, teams
        from local_operator.session.errors import ProfileRegistryUnavailable

        monkeypatch.setattr(
            teams.TeamRegistry, "get_team_by_name", lambda self, name, **kw: object()
        )

        # The exception takes a COUNT, never a message: its text crosses the
        # transport, so it names how many definitions are unreadable, not which.
        def incomplete(self: Any) -> None:
            raise ProfileRegistryUnavailable(count=2)

        monkeypatch.setattr(agents.AgentRegistry, "require_complete_metadata", incomplete)
        block = carry.attachment_resolution(tmp_path, self._session(tmp_path, "lopdev"))
        assert block["resolved"] is True and block["applied"] is False
        assert "team brief is not applied yet" in block["notice"]
        assert "2 agent definitions could not be read" in block["notice"]

    def test_a_stored_agent_beside_a_team_is_not_named_missing(self, tmp_path: Path) -> None:
        """Review r1 N1: a team owns the agent slot, so the stored agent is superseded."""
        directory = tmp_path / "sessions" / SID
        directory.mkdir(parents=True)
        (directory / "attachment.json").write_text(
            json.dumps({"team": "lopdev", "agent": "ghost", "goal": ""}),
            encoding="utf-8",
        )
        block = carry.attachment_resolution(tmp_path, directory)
        assert "agent" not in block and "ghost" not in block["notice"]
        assert "team 'lopdev'" in block["notice"]

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

        # Only ``root`` is read before the stubbed transport stops the drive.
        mq._drive_copy(cast(Any, _Server()), SID)  # noqa: SLF001
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


class TestMovedSessionReappliesItsTeamBrief:
    """(c) root cause, from the moved session on lop-remote-node-2 (ce6fcb93c28a).

    ``attachment.json`` travelled (``{team: "lopdev"}``) and the node listed
    ``lopdev`` as a MIRRORED team, yet the conversation's prompt had no brief: the
    restore runs only at construction, and a moved (answered) conversation never
    re-adopts. A runtime that opened before the mirror resolved must pick the brief
    up on its next turn.
    """

    @pytest.mark.asyncio
    async def test_a_mirrored_team_that_lands_after_the_runtime_briefs_the_next_turn(
        self, tmp_path: Path
    ) -> None:
        from datetime import datetime, timezone

        from local_operator.agents import AgentRegistry
        from local_operator.harness.types import (
            StreamEndEvent,
            StreamEvent,
            StreamTextDelta,
        )
        from local_operator.network import definitions
        from local_operator.prompts_api import build_system_blocks
        from local_operator.resume import write_session_attachment
        from local_operator.session.goal import GoalState
        from local_operator.session.session import Session
        from local_operator.session.transcript import Transcript
        from local_operator.teams import Team, TeamRegistry
        from tests.unit.session.test_session import MODEL, ScriptedStream

        source, dest = tmp_path / "src", tmp_path / "dst"
        session_dir = dest / "sessions" / SID
        session_dir.mkdir(parents=True)
        write_session_attachment(session_dir, team="lopdev", agent="", goal="")

        goal_state = GoalState()

        def provider() -> list[str]:
            return build_system_blocks(
                [],
                "<skills/>",
                "test environment",
                "2026-11-13",
                team_brief=goal_state.team_brief,
                agent_brief=goal_state.agent_brief,
            )

        setattr(provider, "append_only_state", True)
        reply: list[StreamEvent] = [StreamTextDelta(delta="ok"), StreamEndEvent(stop_reason="stop")]
        stream = ScriptedStream([reply, reply])
        teams = TeamRegistry(dest)
        session = Session(
            model=MODEL,
            stream_fn=stream,
            tools=[],
            transcript=Transcript(session_dir),
            system_blocks_provider=provider,
            goal_state=goal_state,
            agent_registry=AgentRegistry(dest),
            team_registry=teams,
        )
        try:
            # The runtime opened before the team resolved here: recorded as a miss.
            assert session.active_team is None
            assert session._unresolved_team == "lopdev"
            await session.prompt("answered before the definitions landed")
            assert "[team: lopdev]" not in "\n".join(stream.requests[0].system_blocks)

            # The definitions push lands: lopdev is installed as a MIRROR of the
            # source's row (provenance index names the origin device).
            TeamRegistry(source).save_team(
                Team(
                    id="t-lopdev",
                    name="lopdev",
                    created_date=datetime.now(timezone.utc),
                    manager="manager",
                    instructions="Ship reviewed work.",
                    project="local-operator",
                )
            )
            applied = definitions.apply_bundle(
                dest, definitions.local_bundle(source), origin_device="dev-source"
            )
            assert "lopdev" in json.dumps(applied)
            assert "lopdev" in json.dumps(definitions.read_index(dest).get("teams") or {})

            await session.prompt("the next turn after the move")
            assert session.active_team_name == "lopdev"
            assert session._unresolved_team == ""
            # The prefix froze on the first turn, so the brief rides that request's
            # authoritative state delta (``[session-state]``) — what the model reads.
            sent = stream.requests[1].model_dump_json()
            assert "[team: lopdev]" in sent
            assert "[team: lopdev]" not in stream.requests[0].model_dump_json()
        finally:
            await session.dispose()


def _role(registry: Any, name: str, prompt: str) -> None:
    """A ``role``-tagged agent row (attachable by name) with ``prompt`` as its voice."""
    from local_operator.agents import AgentEditFields

    fields: dict[str, Any] = dict.fromkeys(
        (
            "description",
            "categories",
            "security_prompt",
            "hosting",
            "model",
            "last_message",
            "temperature",
            "top_p",
            "top_k",
            "max_tokens",
            "stop",
            "frequency_penalty",
            "presence_penalty",
            "seed",
            "current_working_directory",
        )
    )
    agent = registry.create_agent(AgentEditFields(name=name, tags=["role"], **fields))
    registry.set_agent_system_prompt(agent.id, prompt)


async def _answered_session(dest: Path, *, team: str = "", agent: str = "") -> tuple[Any, Any]:
    """A session on ``dest`` whose attachment names definitions this device lacks,
    answered once (so the late adoption is closed, as for every moved session)."""
    from local_operator.agents import AgentRegistry
    from local_operator.harness.types import (
        StreamEndEvent,
        StreamEvent,
        StreamTextDelta,
    )
    from local_operator.prompts_api import build_system_blocks
    from local_operator.resume import write_session_attachment
    from local_operator.session.goal import GoalState
    from local_operator.session.session import Session
    from local_operator.session.transcript import Transcript
    from local_operator.teams import TeamRegistry
    from tests.unit.session.test_session import MODEL, ScriptedStream

    session_dir = dest / "sessions" / SID
    session_dir.mkdir(parents=True)
    write_session_attachment(session_dir, team=team, agent=agent, goal="")
    goal_state = GoalState()

    def provider() -> list[str]:
        return build_system_blocks(
            [],
            "<skills/>",
            "test environment",
            "2026-11-13",
            team_brief=goal_state.team_brief,
            agent_brief=goal_state.agent_brief,
        )

    provider.append_only_state = True  # type: ignore[attr-defined]
    reply: list[StreamEvent] = [StreamTextDelta(delta="ok"), StreamEndEvent(stop_reason="stop")]
    stream = ScriptedStream([reply, reply, reply])
    session = Session(
        model=MODEL,
        stream_fn=stream,
        tools=[],
        transcript=Transcript(session_dir),
        system_blocks_provider=provider,
        goal_state=goal_state,
        agent_registry=AgentRegistry(dest),
        team_registry=TeamRegistry(dest),
    )
    await session.prompt("answered before the definitions landed")
    return session, stream


class TestReresolveRefreshesBothRegistries:
    """Review r2 R2-M1: the agent registry was never force-refreshed (it spells
    the method ``refresh_now``), so agent lookups inside the 5 s snapshot missed
    and the spent fingerprint meant no retry."""

    @pytest.mark.asyncio
    async def test_a_team_and_its_mirrored_manager_both_reach_the_next_turn(
        self, tmp_path: Path
    ) -> None:
        from datetime import datetime, timezone

        from local_operator.agents import AgentRegistry
        from local_operator.network import definitions
        from local_operator.teams import Team, TeamRegistry

        source, dest = tmp_path / "src", tmp_path / "dst"
        session, stream = await _answered_session(dest, team="lopdev")
        try:
            assert session._unresolved_team == "lopdev"
            # Within the agent registry's 5 s snapshot of turn 1: the push lands.
            _role(AgentRegistry(source), "qa-mgr", "MGR-VOICE-MARKER")
            TeamRegistry(source).save_team(
                Team(
                    id="t-lopdev",
                    name="lopdev",
                    created_date=datetime.now(timezone.utc),
                    manager="qa-mgr",
                    instructions="Ship reviewed work.",
                    project="local-operator",
                )
            )
            definitions.apply_bundle(
                dest, definitions.local_bundle(source), origin_device="dev-source"
            )

            await session.prompt("the next turn")
            assert session.active_team_name == "lopdev"
            assert session._goal_state.agent_name == "qa-mgr"
            assert "MGR-VOICE-MARKER" in session._goal_state.team_brief
            sent = stream.requests[1].model_dump_json()
            assert "[team: lopdev]" in sent and "MGR-VOICE-MARKER" in sent
        finally:
            await session.dispose()

    @pytest.mark.asyncio
    async def test_an_agent_only_attachment_resolves_on_the_next_turn(self, tmp_path: Path) -> None:
        from local_operator.agents import AgentRegistry

        dest = tmp_path / "dst"
        session, stream = await _answered_session(dest, agent="qa-helper")
        try:
            assert session._unresolved_agent == "qa-helper"
            _role(AgentRegistry(dest), "qa-helper", "HELPER-VOICE-MARKER")

            await session.prompt("the next turn")
            assert session._goal_state.agent_name == "qa-helper"
            assert session._unresolved_agent == ""
            assert "HELPER-VOICE-MARKER" in stream.requests[1].model_dump_json()
        finally:
            await session.dispose()

    @pytest.mark.asyncio
    async def test_an_unresolved_slot_keeps_its_marker_and_retries(self, tmp_path: Path) -> None:
        from local_operator.agents import AgentRegistry

        dest = tmp_path / "dst"
        session, stream = await _answered_session(dest, agent="qa-helper")
        try:
            # An unrelated row lands: the attempt runs, misses, and keeps the name.
            _role(AgentRegistry(dest), "someone-else", "x")
            await session.prompt("still missing")
            assert session._unresolved_agent == "qa-helper"
            _role(AgentRegistry(dest), "qa-helper", "HELPER-VOICE-MARKER")
            await session.prompt("now it lands")
            assert session._goal_state.agent_name == "qa-helper"
            assert "HELPER-VOICE-MARKER" in stream.requests[2].model_dump_json()
        finally:
            await session.dispose()


class TestMovedInLiftsTheLocalHomeRule:
    """Review r2 R2-m1: a move between devices sharing a home path still maps."""

    def test_a_moved_session_maps_a_missing_cwd_under_a_shared_home(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home" / "damian"
        (home / "proj").mkdir(parents=True)
        monkeypatch.setenv("HOME", str(home))
        source_only = str(home / "src-only" / "proj")
        rows, counts = carry.translate_monitor_cwds(
            [{"cwd": source_only}], session_cwd=str(tmp_path), home=str(home), moved=True
        )
        assert rows == [{"cwd": str(home / "proj")}] and counts["translated"] == 1
        # Not moved: the same path is this host's own and is left to fail loudly.
        rows, counts = carry.translate_monitor_cwds(
            [{"cwd": source_only}], session_cwd=str(tmp_path), home=str(home)
        )
        assert rows == [{"cwd": source_only}] and counts == {"translated": 0, "fallback": 0}

    def test_the_promote_flag_is_read_back_and_never_creates_a_marker(self, tmp_path: Path) -> None:
        assert carry.mark_moved_in(tmp_path) is False
        assert not (tmp_path / carry.DESKTOP_MARKER_NAME).exists()
        (tmp_path / carry.DESKTOP_MARKER_NAME).write_text('{"cwd": "/x"}', encoding="utf-8")
        assert carry.moved_in(tmp_path) is False
        assert carry.mark_moved_in(tmp_path) is True
        assert carry.moved_in(tmp_path) is True
        marker = json.loads((tmp_path / carry.DESKTOP_MARKER_NAME).read_text(encoding="utf-8"))
        assert marker == {"cwd": "/x", carry.MOVED_IN_KEY: True}
