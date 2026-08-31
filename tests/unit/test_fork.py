"""Forking a session — the properties whose failure is silent or remote.

What is pinned here is deliberately not "the copy copied a file". It is the set
of facts whose violation shows up somewhere else entirely:

- **The parent is not touched.** A fork that truncated, re-stamped or even
  re-dated the conversation it branched from would damage the work the user is
  still doing, in the window they are still looking at.
- **The sidecar copy is an ALLOW-LIST.** Copying ``.session.pid`` makes the fork
  look owned by the parent's pid, so the fork's own boot takes the *attach* path
  instead of opening — a spectacular bug that is one ``copytree`` away. The set
  assertion below is what catches that rewrite.
- **The clone is legal on the wire.** A transcript whose tail is an assistant
  ``tool_use`` with no ``tool_result`` makes the fork's FIRST request a provider
  400 — in a different window, minutes after the command.
- **The boot prompt fires exactly once.** A fork the user later resumes must not
  re-run the instruction that opened it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

import pytest
from textual.widgets import Static

from local_operator.fork import (
    BOOT_PROMPT_NAME,
    COPIED_SIDECARS,
    EXCLUDED_SIDECARS,
    FORK_BOUNDARY_INSTRUCTION,
    FORK_BOUNDARY_NAME,
    ForkError,
    consume_boot_prompt,
    consume_fork_boundary,
    fork_parent,
    fork_session,
    write_boot_prompt,
)
from local_operator.resume import (
    ATTACHMENT_SIDECAR_NAME,
    ORIGIN_FORK,
    ORIGIN_SUBAGENT,
    TITLE_SIDECAR_NAME,
    TRANSCRIPT_NAME,
    is_user_session,
    live_session_owner,
    mark_session_origin,
    recent_sessions,
    session_origin,
)

PARENT_ID = "aaaaaaaaaaaa"

#: A transcript ending on a COMPLETE exchange — the shape a fork taken at a safe
#: boundary produces.
TRANSCRIPT_LINES = [
    {"id": "m1", "role": "user", "content": "read the loader"},
    {"id": "m2", "role": "assistant", "content": "It parses YAML."},
]


def _seed_parent(config_dir: Path, session_id: str = PARENT_ID, **sidecars: str) -> Path:
    """A session directory with a transcript and whichever sidecars are named."""
    parent = config_dir / "sessions" / session_id
    parent.mkdir(parents=True)
    (parent / TRANSCRIPT_NAME).write_text(
        "\n".join(json.dumps(line) for line in TRANSCRIPT_LINES) + "\n", encoding="utf-8"
    )
    for name, body in sidecars.items():
        (parent / name.replace("__", ".")).write_text(body, encoding="utf-8")
    return parent


class TestTheCloneItself:
    def test_the_fork_is_byte_identical_and_the_parent_is_untouched(self, tmp_path: Path) -> None:
        """T1. The parent's bytes, size AND mtime all survive the fork."""
        parent = _seed_parent(tmp_path)
        transcript = parent / TRANSCRIPT_NAME
        before_bytes = transcript.read_bytes()
        before_stat = transcript.stat()

        fork_id = fork_session(tmp_path, PARENT_ID)

        assert fork_id != PARENT_ID
        assert len(fork_id) == 12
        fork_transcript = tmp_path / "sessions" / fork_id / TRANSCRIPT_NAME
        # Byte-identical, not merely equivalent: the fork's first request has to
        # reproduce the parent's cached prefix exactly, and a read-parse-write
        # would reorder JSON keys while changing nothing semantically.
        assert fork_transcript.read_bytes() == before_bytes
        after_stat = transcript.stat()
        assert after_stat.st_size == before_stat.st_size
        assert after_stat.st_mtime == before_stat.st_mtime

    def test_only_the_allow_listed_sidecars_are_copied(self, tmp_path: Path) -> None:
        """T2. THE guard against a future ``copytree``.

        Asserted as an exact SET rather than as "the pid file is absent",
        because the failure this catches is someone replacing the loop with a
        directory copy — which would bring across every file added to a session
        directory from now on, not just the ones known today.
        """
        parent = _seed_parent(tmp_path)
        (parent / ATTACHMENT_SIDECAR_NAME).write_text('{"team": "lopdev"}', encoding="utf-8")
        (parent / TITLE_SIDECAR_NAME).write_text('{"title": "Refactor"}', encoding="utf-8")
        # Every file that must NOT travel, present in the parent.
        for name in EXCLUDED_SIDECARS:
            (parent / name).write_text("do not copy me", encoding="utf-8")

        fork_id = fork_session(tmp_path, PARENT_ID)
        fork_dir = tmp_path / "sessions" / fork_id

        assert set(os.listdir(fork_dir)) == {
            TRANSCRIPT_NAME,
            ATTACHMENT_SIDECAR_NAME,
            TITLE_SIDECAR_NAME,
            FORK_BOUNDARY_NAME,  # written by the fork, not inherited
            "origin.json",  # written by the fork itself, not copied
        }

    def test_the_fork_is_left_unowned_so_its_own_boot_can_claim_it(self, tmp_path: Path) -> None:
        """R1. The fork must have NO live owner when the clone returns.

        ``fork_session`` runs inside the PARENT's process, so the transient
        ``claim_session`` it takes to close the retention window stamps the
        parent's pid. Left in place, the fork's own boot reads that marker and
        either refuses ("session <id> is open in an older Local Operator
        process") or attaches the new window as a FOLLOWER of its parent
        instead of opening the branched conversation.

        Asserted through ``live_session_owner`` rather than by checking that
        the file is absent, deliberately: ownership is the property that
        actually matters, and a future claim written by some other path would
        slip past a filename assertion while breaking the boot exactly the same
        way.
        """
        _seed_parent(tmp_path)
        fork_id = fork_session(tmp_path, PARENT_ID)

        assert live_session_owner(tmp_path, fork_id) is None
        assert not (tmp_path / "sessions" / fork_id / ".session.pid").exists()

    def test_the_parents_own_claim_is_untouched_by_a_fork(self, tmp_path: Path) -> None:
        """Releasing the fork's claim must not release the PARENT's.

        The parent is a live session that legitimately owns itself; a fork
        taken from it must leave that ownership exactly as it found it.
        """
        parent = _seed_parent(tmp_path)
        (parent / ".session.pid").write_text(str(os.getpid()), encoding="utf-8")

        fork_session(tmp_path, PARENT_ID)

        assert live_session_owner(tmp_path, PARENT_ID) == os.getpid()

    def test_a_missing_optional_sidecar_is_not_an_error(self, tmp_path: Path) -> None:
        """A young conversation has no title and no persona; forking still works."""
        _seed_parent(tmp_path)
        fork_id = fork_session(tmp_path, PARENT_ID)
        fork_dir = tmp_path / "sessions" / fork_id
        assert (fork_dir / TRANSCRIPT_NAME).is_file()
        assert not (fork_dir / TITLE_SIDECAR_NAME).exists()

    def test_forking_a_session_with_no_transcript_raises(self, tmp_path: Path) -> None:
        """The one hard failure: no fork was created, and the caller must know.

        A ForkError is distinguishable from a failed SPAWN, which is what lets
        the TUI say "there is no fork" rather than "your fork is waiting".
        """
        (tmp_path / "sessions" / PARENT_ID).mkdir(parents=True)
        with pytest.raises(ForkError):
            fork_session(tmp_path, PARENT_ID)

    def test_the_copied_sidecar_list_is_what_the_module_documents(self) -> None:
        """The allow-list and the deny-list must not overlap — a file cannot be
        both inherited and excluded, and a rename that broke that would be
        invisible until a fork misbehaved."""
        assert not set(COPIED_SIDECARS) & set(EXCLUDED_SIDECARS)


class TestProvenance:
    def test_the_fork_records_its_parent_and_stays_a_user_session(self, tmp_path: Path) -> None:
        """T4. Provenance is recorded; visibility is NOT sacrificed for it."""
        _seed_parent(tmp_path)
        fork_id = fork_session(tmp_path, PARENT_ID)
        fork_dir = tmp_path / "sessions" / fork_id

        assert session_origin(fork_dir) == ORIGIN_FORK
        payload = json.loads((fork_dir / "origin.json").read_text(encoding="utf-8"))
        assert payload["parent"] == PARENT_ID
        assert "forked_at" in payload
        assert fork_parent(fork_dir) == PARENT_ID
        # The whole point of the USER_ORIGINS allow-list.
        assert is_user_session(fork_dir) is True

    def test_fork_parent_is_empty_for_an_ordinary_session(self, tmp_path: Path) -> None:
        parent = _seed_parent(tmp_path)
        assert fork_parent(parent) == ""

    def test_the_picker_lists_forks_and_still_hides_subagents(self, tmp_path: Path) -> None:
        """T5. The allow-list must not widen into "any origin is fine"."""
        _seed_parent(tmp_path)
        fork_id = fork_session(tmp_path, PARENT_ID)

        child = _seed_parent(tmp_path, session_id="cccccccccccc")
        mark_session_origin(child, ORIGIN_SUBAGENT)

        listed = {session_id for session_id, _mtime in recent_sessions(tmp_path)}
        assert fork_id in listed
        assert PARENT_ID in listed
        assert "cccccccccccc" not in listed


class TestTheForkBoundary:
    def test_every_fork_gets_exactly_one_nonpersistent_context_boundary(
        self, tmp_path: Path
    ) -> None:
        parent = _seed_parent(tmp_path)
        before = (parent / TRANSCRIPT_NAME).read_bytes()
        fork_id = fork_session(tmp_path, PARENT_ID)
        fork_dir = tmp_path / "sessions" / fork_id

        assert (fork_dir / TRANSCRIPT_NAME).read_bytes() == before
        assert consume_fork_boundary(fork_dir) == FORK_BOUNDARY_INSTRUCTION
        assert consume_fork_boundary(fork_dir) == ""
        assert (fork_dir / TRANSCRIPT_NAME).read_bytes() == before

    @pytest.mark.asyncio
    async def test_session_places_the_boundary_once_at_the_inherited_context_tail(
        self, tmp_path: Path
    ) -> None:
        from local_operator.session.transcript import Transcript

        parent = tmp_path / "sessions" / PARENT_ID
        parent.mkdir(parents=True)
        transcript = Transcript(parent)
        await transcript.append_message(_user("read the loader"))
        await transcript.append_message(_assistant("It parses YAML."))
        fork_id = fork_session(tmp_path, PARENT_ID)
        fork_dir = tmp_path / "sessions" / fork_id

        first = _build_session(fork_dir)
        inherited = first.history()
        assert [getattr(message, "role", "") for message in inherited[:2]] == [
            "user",
            "assistant",
        ]
        boundary = inherited[-1]
        assert getattr(boundary, "custom_type", "") == "fork_boundary"
        assert getattr(boundary, "details", {})["text"] == FORK_BOUNDARY_INSTRUCTION
        assert (fork_dir / TRANSCRIPT_NAME).read_text(encoding="utf-8").count("fork_boundary") == 0

        resumed = _build_session(fork_dir)
        assert all(
            getattr(message, "custom_type", "") != "fork_boundary" for message in resumed.history()
        )


class TestTheBootPrompt:
    def test_the_message_round_trips_and_fires_exactly_once(self, tmp_path: Path) -> None:
        """T6. The delete is the load-bearing half.

        A fork the user later resumes must replay its transcript and idle, like
        every other session — not re-run the instruction that opened it.
        """
        _seed_parent(tmp_path)
        fork_id = fork_session(tmp_path, PARENT_ID, message="try the other loader")
        fork_dir = tmp_path / "sessions" / fork_id

        assert consume_boot_prompt(fork_dir) == "try the other loader"
        assert not (fork_dir / BOOT_PROMPT_NAME).exists()
        # The second boot of the same session submits nothing.
        assert consume_boot_prompt(fork_dir) == ""

    def test_no_sidecar_is_written_without_a_message(self, tmp_path: Path) -> None:
        _seed_parent(tmp_path)
        fork_id = fork_session(tmp_path, PARENT_ID)
        assert not (tmp_path / "sessions" / fork_id / BOOT_PROMPT_NAME).exists()

    def test_a_whitespace_only_message_is_not_a_message(self, tmp_path: Path) -> None:
        _seed_parent(tmp_path)
        fork_id = fork_session(tmp_path, PARENT_ID, message="   \n ")
        assert not (tmp_path / "sessions" / fork_id / BOOT_PROMPT_NAME).exists()

    @pytest.mark.parametrize(
        "body",
        [
            "{not json at all",
            '{"version": 1, "text"',  # truncated mid-write
            '["a", "list"]',
            '{"version": 1}',  # no text key
            '{"version": 1, "text": 12}',  # wrong type
        ],
        ids=["garbage", "truncated", "not-a-mapping", "no-text", "wrong-type"],
    )
    def test_a_corrupt_sidecar_is_ignored_and_cleared(self, tmp_path: Path, body: str) -> None:
        """T9. Losing the injected message is a notice; failing the boot is not
        survivable — the same tolerance rule the title and attachment readers
        document. The file is cleared either way, or every later boot retries it.
        """
        session_dir = tmp_path / "sessions" / PARENT_ID
        session_dir.mkdir(parents=True)
        (session_dir / BOOT_PROMPT_NAME).write_text(body, encoding="utf-8")

        assert consume_boot_prompt(session_dir) == ""
        assert not (session_dir / BOOT_PROMPT_NAME).exists()

    def test_consuming_a_missing_sidecar_is_silent(self, tmp_path: Path) -> None:
        session_dir = tmp_path / "sessions" / PARENT_ID
        session_dir.mkdir(parents=True)
        assert consume_boot_prompt(session_dir) == ""

    def test_the_sidecar_survives_unicode(self, tmp_path: Path) -> None:
        """The message is arbitrary user text and reaches disk as JSON."""
        session_dir = tmp_path / "sessions" / PARENT_ID
        session_dir.mkdir(parents=True)
        write_boot_prompt(session_dir, "refactor the café loader — twice")
        assert consume_boot_prompt(session_dir) == "refactor the café loader — twice"


class TestForkRuntimeOwnership:
    @pytest.mark.asyncio
    async def test_parent_runtime_ownership_stays_parent_only(self, tmp_path: Path) -> None:
        from local_operator.harness.wake import WakeSchedule
        from local_operator.session.session import WAKE_SCHEDULES_CUSTOM_TYPE
        from local_operator.session.transcript import Transcript

        parent = tmp_path / "sessions" / PARENT_ID
        parent.mkdir(parents=True)
        transcript = Transcript(parent)
        await transcript.append_message(_user("schedule the audit and delegate research"))
        await transcript.append_custom(
            WAKE_SCHEDULES_CUSTOM_TYPE,
            {
                "schedules": [
                    {
                        "id": "w-parent",
                        "message": "audit now",
                        "next_due_at": 9_999_999_999_999,
                        "created_at": 1,
                    }
                ]
            },
        )
        await transcript.append_custom(
            "subagent_roster",
            {
                "generation": 4,
                "records": [{"job_id": "parent-job", "session_dir": "child-parent"}],
                "jobs": [],
            },
        )
        parent_session = _build_session(parent)
        assert [wake.id for wake in parent_session.wake_scheduler.schedules] == ["w-parent"]
        assert parent_session.subagent_comms.session_dir_of("parent-job") == Path("child-parent")

        fork_id = fork_session(tmp_path, PARENT_ID)
        fork = _build_session(tmp_path / "sessions" / fork_id)
        assert fork.wake_scheduler.schedules == ()
        assert fork.jobs.get("parent-job") is None
        assert fork.subagent_comms.session_dir_of("parent-job") is None
        # Parent ownership was read, never moved or cancelled.
        assert [wake.id for wake in parent_session.wake_scheduler.schedules] == ["w-parent"]
        assert parent_session.subagent_comms.session_dir_of("parent-job") == Path("child-parent")

        own = WakeSchedule(
            id="w-fork", message="divergent audit", next_due_at=9_999_999_999_999, created_at=2
        )
        await fork.set_wake_schedules([own])
        assert [wake.id for wake in fork.wake_scheduler.schedules] == ["w-fork"]
        assert [wake.id for wake in parent_session.wake_scheduler.schedules] == ["w-parent"]

        # Isolation must not disable the capability: the fork owns children it
        # launches after boot, while the inherited parent's child stays absent.
        fork.subagent_comms.restore(
            [{"job_id": "fork-job", "session_dir": "child-fork", "label": "new research"}]
        )
        assert fork.subagent_comms.session_dir_of("fork-job") == Path("child-fork")
        assert fork.subagent_comms.session_dir_of("parent-job") is None
        assert parent_session.subagent_comms.session_dir_of("fork-job") is None


class TestForkCmuxOwnership:
    def test_parent_never_schedules_a_cmux_rename(self, tmp_path: Path, monkeypatch) -> None:
        from local_operator.tui.app import OperatorApp

        parent = _seed_parent(tmp_path)
        scheduled: list[Any] = []
        app = cast(
            OperatorApp,
            type(
                "ForkNameHarness",
                (),
                {
                    "_session": type("SessionStub", (), {"session_id": parent.name})(),
                    "_fork_cmux_name": "",
                    "run_worker": lambda self, awaitable, **kwargs: scheduled.append(awaitable),
                },
            )(),
        )
        monkeypatch.setattr("local_operator.paths.config_dir", lambda: tmp_path)

        OperatorApp._sync_fork_cmux_name(app, "Parent work")

        assert scheduled == []
        assert app._fork_cmux_name == ""

    def test_fork_schedules_only_its_owned_target_rename(self, tmp_path: Path, monkeypatch) -> None:
        from local_operator.tui.app import OperatorApp

        _seed_parent(tmp_path)
        fork_id = fork_session(tmp_path, PARENT_ID)
        scheduled: list[Any] = []
        app = cast(
            OperatorApp,
            type(
                "ForkNameHarness",
                (),
                {
                    "_session": type("SessionStub", (), {"session_id": fork_id})(),
                    "_fork_cmux_name": "",
                    "run_worker": lambda self, awaitable, **kwargs: scheduled.append(awaitable),
                },
            )(),
        )
        monkeypatch.setattr("local_operator.paths.config_dir", lambda: tmp_path)

        OperatorApp._sync_fork_cmux_name(app, "  Divergent   work ")

        assert app._fork_cmux_name == "Divergent work"
        assert len(scheduled) == 1
        # The harness deliberately does not run the worker: argv construction is
        # covered separately, and closing avoids a false un-awaited warning.
        scheduled[0].close()


class TestTheClonedTranscriptIsLegalOnTheWire:
    @pytest.mark.asyncio
    async def test_a_clone_of_a_complete_exchange_replays_legally(self, tmp_path: Path) -> None:
        """T3. The failure this guards has the worst diagnostic distance in the
        feature: a fork taken mid-batch carries a persisted assistant
        ``tool_use`` with no ``tool_result``, and the fork's first request 400s
        in a different window, minutes later.

        Asserted over the CLONED file through the real replay, which is what the
        fork's boot will do.
        """
        from local_operator.session.transcript import Transcript

        parent = tmp_path / "sessions" / PARENT_ID
        parent.mkdir(parents=True)
        transcript = Transcript(parent)
        await transcript.append_message(_user("read the loader"))
        await transcript.append_message(_assistant_with_tool_call("call-1", "read"))
        await transcript.append_message(_tool_result("call-1", "it parses YAML"))
        await transcript.append_message(_assistant("The loader parses YAML."))

        fork_id = fork_session(tmp_path, PARENT_ID)
        replayed = Transcript(tmp_path / "sessions" / fork_id).build_llm_history()

        assert _unanswered_tool_calls(replayed) == []

    @pytest.mark.asyncio
    async def test_the_guard_would_catch_an_unanswered_tool_call(self, tmp_path: Path) -> None:
        """The negative control. Without it, the assertion above could be
        vacuous — passing because the helper never finds anything at all.
        """
        from local_operator.session.transcript import Transcript

        parent = tmp_path / "sessions" / PARENT_ID
        parent.mkdir(parents=True)
        transcript = Transcript(parent)
        await transcript.append_message(_user("read the loader"))
        await transcript.append_message(_assistant_with_tool_call("call-1", "read"))

        fork_id = fork_session(tmp_path, PARENT_ID)
        replayed = Transcript(tmp_path / "sessions" / fork_id).build_llm_history()

        assert _unanswered_tool_calls(replayed) == ["call-1"]


# --- helpers ---------------------------------------------------------------


def _user(text: str):
    from local_operator.harness.types import Message

    return Message.user(text)


def _assistant(text: str):
    from local_operator.harness.types import Message

    return Message.assistant(text)


def _assistant_with_tool_call(call_id: str, name: str):
    from local_operator.harness.types import Message, ToolCall

    message = Message.assistant("")
    message.tool_calls = [ToolCall(id=call_id, name=name, arguments={})]
    return message


def _tool_result(call_id: str, text: str):
    from local_operator.harness.types import Message, TextContent, ToolResult

    return Message.tool_result(
        ToolResult(tool_call_id=call_id, tool_name="read", content=[TextContent(text=text)])
    )


def _unanswered_tool_calls(messages) -> list[str]:
    """Ids of tool calls with no matching result — what a provider 400s on."""
    answered = {
        getattr(message, "tool_call_id", None)
        for message in messages
        if getattr(message, "role", "") == "tool"
    }
    pending: list[str] = []
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            if call.id not in answered:
                pending.append(call.id)
    return pending


class TestTheForkIsNamedForItsOwnWork:
    """The fork must be named for what it was forked to DO.

    A fork carries a byte-identical copy of its parent's transcript, so the
    parent's journalled title comes across inside it. Adopting that title would
    name the branch after the work it LEFT and then latch ``requested``, so no
    naming call ever fires and the picker fills with rows all called the same
    thing. These pin the suppression and — just as important — its release.
    """

    def test_a_fresh_fork_declines_the_inherited_title(self, tmp_path: Path) -> None:
        """The parent's name is in the copied transcript; the fork ignores it."""
        parent = _seed_named_parent(tmp_path, "Refactor the loader")
        assert parent.exists()
        fork_id = fork_session(tmp_path, PARENT_ID, message="try the streaming parser")

        session = _build_session(tmp_path / "sessions" / fork_id)
        assert session.conversation_name == ""

    def test_a_fork_that_has_named_itself_keeps_that_name(self, tmp_path: Path) -> None:
        """The release. Suppression is one-shot: it is decided by TIME, so the
        first title this fork writes lands after ``forked_at`` and every later
        resume restores it normally. Without this the fork would boot nameless
        forever and re-name itself on every single resume.
        """
        import asyncio

        from local_operator.session.session import CONVERSATION_NAME_CUSTOM_TYPE
        from local_operator.session.transcript import Transcript

        _seed_named_parent(tmp_path, "Refactor the loader")
        fork_id = fork_session(tmp_path, PARENT_ID, message="try the streaming parser")
        fork_dir = tmp_path / "sessions" / fork_id

        async def name_itself() -> None:
            await Transcript(fork_dir).append_custom(
                CONVERSATION_NAME_CUSTOM_TYPE,
                {"text": "Streaming parser attempt", "user_set": False},
            )

        asyncio.run(name_itself())

        session = _build_session(fork_dir)
        assert session.conversation_name == "Streaming parser attempt"

    def test_an_ordinary_resume_still_inherits_its_own_title(self, tmp_path: Path) -> None:
        """The control that stops this becoming "titles never restore".

        A plain session is not a fork and must restore its name exactly as
        before — this is the behaviour the suppression must not have widened.
        """
        parent = _seed_named_parent(tmp_path, "Refactor the loader")
        session = _build_session(parent)
        assert session.conversation_name == "Refactor the loader"

    def test_an_unnamed_fork_still_has_a_picker_row(self, tmp_path: Path) -> None:
        """Declining the title must not produce a BLANK row in /resume.

        This is why ``title.json`` is still copied even though the session
        declines to wear it at boot: the picker reads the sidecar, so the fork
        is labelled with the parent's name for the seconds before its own title
        lands, rather than rendering nameless. The fork's opening message is the
        parent's, so the opener fallback would not save it either.
        """
        from local_operator.resume import session_name

        _seed_named_parent(tmp_path, "Refactor the loader")
        fork_id = fork_session(tmp_path, PARENT_ID, message="try the streaming parser")

        assert session_name(tmp_path / "sessions" / fork_id).strip() != ""

    def test_the_naming_path_is_open_for_both_branches(self, tmp_path: Path) -> None:
        """Both ``/fork <message>`` and bare ``/fork`` leave naming ARMED.

        The TUI's ``_maybe_name_conversation`` fires only while
        ``session.conversation_name`` is empty, so an empty name is exactly the
        precondition both branches need: the injected message names the fork
        immediately, and a bare fork is named by the first request the user
        types into it.
        """
        _seed_named_parent(tmp_path, "Refactor the loader")

        with_message = fork_session(tmp_path, PARENT_ID, message="try the streaming parser")
        bare = fork_session(tmp_path, PARENT_ID)

        for fork_id in (with_message, bare):
            session = _build_session(tmp_path / "sessions" / fork_id)
            assert session.conversation_name == ""


def _seed_named_parent(config_dir: Path, title: str) -> Path:
    """A parent whose title is journalled the way a real session's is."""
    import asyncio

    from local_operator.resume import write_session_title
    from local_operator.session.session import CONVERSATION_NAME_CUSTOM_TYPE
    from local_operator.session.transcript import Transcript

    parent = config_dir / "sessions" / PARENT_ID
    parent.mkdir(parents=True)

    async def seed() -> None:
        transcript = Transcript(parent)
        await transcript.append_message(_user("refactor the loader"))
        await transcript.append_custom(
            CONVERSATION_NAME_CUSTOM_TYPE, {"text": title, "user_set": False}
        )

    asyncio.run(seed())
    write_session_title(parent, title, user_set=False, past_names=[])
    # The clone's stamp must be strictly later than the parent's title entry,
    # which is true in practice (a fork happens after the turn that named the
    # parent) but needs help at test speed, where both land in the same
    # microsecond.
    import time

    time.sleep(0.01)
    return parent


def _build_session(session_dir: Path):
    """A Session over ``session_dir``, built the way construction does.

    Only the transcript-derived state matters here (the title restore), so this
    stays a direct construction rather than standing up the whole factory —
    which would need a model, an auth store and a provider.
    """
    from local_operator.session.session import Session
    from local_operator.session.transcript import Transcript

    return Session(
        transcript=Transcript(session_dir),
        model=_model_spec(),
        tools=[],
        stream_fn=_never_called_stream,
        system_blocks_provider=lambda: ["stable", "env"],
    )


def _model_spec():
    from local_operator.harness.types import ModelSpec

    return ModelSpec(provider="anthropic", model_id="claude-opus-5")


async def _never_called_stream(*args, **kwargs):
    raise AssertionError("these tests must not reach a provider")
    yield  # pragma: no cover - generator shape only


class TestTheBootPromptFiresInBothModes:
    """R2. The injected message must be consumed exactly once, in EITHER mode.

    `fork.mode = window` opens a new process, which reads the sidecar on its
    cold boot. `fork.mode = switch` swaps this terminal onto the fork in
    process, and that path has its own adoption site — so a sidecar consumed
    only on the cold path is dropped in switch mode, and then fires on the
    fork's next resume, which is the "restore-and-idle, never auto-continue"
    rule broken in the other direction.
    """

    def test_the_sidecar_is_consumed_once_and_only_once(self, tmp_path: Path) -> None:
        _seed_parent(tmp_path)
        fork_id = fork_session(tmp_path, PARENT_ID, message="try the streaming parser")
        fork_dir = tmp_path / "sessions" / fork_id

        # Whichever mode ran, the first adoption reads it...
        assert consume_boot_prompt(fork_dir) == "try the streaming parser"
        # ...and a later resume of the same fork must find nothing to run.
        assert consume_boot_prompt(fork_dir) == ""
        assert not (fork_dir / BOOT_PROMPT_NAME).exists()

    def test_both_adoption_sites_call_the_same_consumer(self) -> None:
        """The property that makes the two modes agree, asserted structurally.

        Both the cold-boot path (`_boot_session`) and the in-process swap path
        (`_reload_session`) must call `_submit_boot_prompt`. A source assertion
        rather than a pilot run because the swap path needs a real session
        factory and a real transition; what can go wrong here is one of the two
        sites being forgotten, which this catches directly.
        """
        import inspect

        from local_operator.tui.app import OperatorApp

        for method in ("_boot_session", "_reload_session"):
            source = inspect.getsource(getattr(OperatorApp, method))
            assert "_submit_boot_prompt" in source, (
                f"{method} does not consume the fork boot prompt, so a "
                f"/fork <message> in that mode silently drops the message"
            )


class TestTheForkRowIsDistinguishableWhileItBorrowsATitle:
    """D2/U2. A fresh fork and its parent were byte-identical picker rows.

    Same name, same "just now", separable only by a 12-hex id — in exactly the
    window where a user is most likely hunting for one of the two. The marker
    is about the AMBIGUOUS STATE, not ancestry, so it must clear the moment the
    fork names itself or it becomes permanent noise.
    """

    def test_a_fork_wearing_its_parents_title_is_marked(self, tmp_path: Path) -> None:
        from local_operator.resume import recent_session_rows

        _seed_named_parent(tmp_path, "Refactor the loader")
        fork_id = fork_session(tmp_path, PARENT_ID)

        rows = {row.id: row for row in recent_session_rows(tmp_path)}
        assert rows[fork_id].forked is True
        assert rows[PARENT_ID].forked is False
        # Both still READ the same, which is why the flag has to exist.
        assert rows[fork_id].name == rows[PARENT_ID].name

    def test_the_mark_clears_once_the_fork_names_itself(self, tmp_path: Path) -> None:
        """The half that stops this becoming a permanent tag."""
        import time as _time

        from local_operator.resume import recent_session_rows, write_session_title

        _seed_named_parent(tmp_path, "Refactor the loader")
        fork_id = fork_session(tmp_path, PARENT_ID)
        _time.sleep(0.02)
        write_session_title(
            tmp_path / "sessions" / fork_id,
            "Streaming parser attempt",
            user_set=False,
            past_names=[],
        )

        rows = {row.id: row for row in recent_session_rows(tmp_path)}
        assert rows[fork_id].name == "Streaming parser attempt"
        assert rows[fork_id].forked is False, "the fork stayed tagged after naming itself"

    def test_an_ordinary_session_is_never_marked(self, tmp_path: Path) -> None:
        from local_operator.resume import recent_session_rows

        _seed_named_parent(tmp_path, "Refactor the loader")
        rows = recent_session_rows(tmp_path)
        assert all(row.forked is False for row in rows)


class TestASecondForkRequestIsNotSilent:
    """U3. Two `/fork`s produced two acknowledgements and one fork."""

    def test_replacing_a_pending_request_is_reported(self, tmp_path: Path) -> None:
        session = _bare_session(tmp_path)

        first = session.request_fork(tmp_path, message="a", on_complete=lambda i, e: None)
        second = session.request_fork(tmp_path, message="b", on_complete=lambda i, e: None)

        assert first is False, "the first request replaced nothing"
        assert second is True, "the second request silently discarded the first"


class TestAPendingForkIsCancellable:
    """U4. Esc stopped the turn and the fork arrived anyway, minutes later."""

    def test_abort_withdraws_a_pending_fork(self, tmp_path: Path) -> None:
        session = _bare_session(tmp_path)
        session.request_fork(tmp_path, message="x", on_complete=lambda i, e: None)
        assert session.has_pending_fork() is True

        session.abort("interrupted")

        assert session.has_pending_fork() is False

    def test_cancel_reports_whether_anything_was_pending(self, tmp_path: Path) -> None:
        session = _bare_session(tmp_path)
        assert session.cancel_fork() is False
        session.request_fork(tmp_path, message="x", on_complete=lambda i, e: None)
        assert session.cancel_fork() is True


def _bare_session(tmp_path: Path):
    """A Session over an empty transcript, for the request/cancel bookkeeping."""
    _seed_parent(tmp_path, session_id="reqreqreqreq")
    return _build_session(tmp_path / "sessions" / "reqreqreqreq")


class TestThePickerPaysNothingForSessionsThatAreNotForks:
    """R6. The fork mark must not cost the picker's UI-thread path.

    An earlier revision asked ``wears_inherited_title`` per row
    unconditionally, which attempted two reads per row on a store containing
    ZERO forks — absence was discovered from the OSError — and measured +52% on
    a 3,000-session store. ``recent_sessions`` documents at length that
    "unmarked is the cheap path"; this pins that the mark did not give it back.
    """

    def test_an_unmarked_store_costs_one_read_per_row(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        from local_operator.resume import recent_session_rows

        sessions = tmp_path / "sessions"
        sessions.mkdir(parents=True)
        for index in range(12):
            directory = sessions / f"{index:012x}"
            directory.mkdir()
            (directory / TRANSCRIPT_NAME).write_text(
                json.dumps(TRANSCRIPT_LINES[0]) + "\n", encoding="utf-8"
            )

        original = Path.read_text
        calls: list[str] = []

        def counting(self: Path, *args: object, **kwargs: object) -> str:
            calls.append(self.name)
            return original(self, *args, **kwargs)  # type: ignore[arg-type]

        with patch.object(Path, "read_text", counting):
            rows = recent_session_rows(tmp_path)

        assert len(rows) == 12
        # PER-ROW reads only: the store-wide origin-verdict cache is read once
        # for the whole scan and is not a per-row cost, so it is excluded.
        per_row = [name for name in calls if name != "origin-verdicts.json"]
        # One bounded head read per row for the NAME, and nothing else. The
        # origin verdict rides out of the scan that already parsed it, so no
        # `origin.json` is opened here at all — that is the regression this
        # pins, and the failing shape was 2.00 reads/row with `origin.json`
        # in the set.
        assert len(per_row) <= len(rows), f"{len(per_row) / len(rows):.2f} reads/row: {set(calls)}"
        assert "origin.json" not in set(per_row)
        assert not any(row.forked for row in rows)

    def test_the_scan_threads_the_origin_it_already_parsed(self, tmp_path: Path) -> None:
        """The mechanism the cost fix relies on, asserted directly."""
        from local_operator.resume import _recent_sessions_with_origin

        _seed_parent(tmp_path)
        fork_id = fork_session(tmp_path, PARENT_ID)

        origins = {row[0]: row[2] for row in _recent_sessions_with_origin(tmp_path)}
        assert origins[fork_id] == ORIGIN_FORK
        assert origins[PARENT_ID] == ""


class TestAForkOfANeverNamedParentIsStillMarked:
    """R8. A never-named parent still DISPLAYS a name — the transcript opener —
    and the clone copies the transcript, so the fork shows the identical opener
    beside it. That is the duplicate-row confusion the mark exists to resolve.
    """

    def test_the_opener_case_is_marked(self, tmp_path: Path) -> None:
        import asyncio
        import time as _time

        from local_operator.resume import recent_session_rows
        from local_operator.session.transcript import Transcript

        parent = tmp_path / "sessions" / PARENT_ID
        parent.mkdir(parents=True)

        async def seed() -> None:
            await Transcript(parent).append_message(_user("debug the flaky retention test"))

        asyncio.run(seed())
        _time.sleep(0.02)
        fork_id = fork_session(tmp_path, PARENT_ID)

        rows = {row.id: row for row in recent_session_rows(tmp_path)}
        # Both display the same opener, which is exactly why the mark matters.
        assert rows[fork_id].name == rows[PARENT_ID].name
        assert rows[fork_id].forked is True
        assert rows[PARENT_ID].forked is False


class TestTheSwitchReceiptCannotNarrateTheWrongSession:
    """R7. The stash survived a FAILED transition and fired on a later,
    unrelated one — telling the user they had switched to a fork they never
    reached, from a session that was neither.
    """

    def test_it_does_not_fire_for_a_different_session(self) -> None:
        from local_operator.tui.app import OperatorApp

        app = OperatorApp.__new__(OperatorApp)
        app._pending_fork_receipt = ("forkaaaaaaaa", "switched to fork forkaaaaaaaa — …")
        emitted: list[str] = []
        app._notice = lambda body, kind="info": emitted.append(body)  # type: ignore[method-assign]

        OperatorApp._flush_fork_receipt(app, "unrelatedbbb")

        assert emitted == [], "the receipt narrated a session it was not about"
        assert app._pending_fork_receipt is None, "a mismatched stash must not linger"

    def test_it_fires_for_the_fork_it_was_written_about(self) -> None:
        from local_operator.tui.app import OperatorApp

        app = OperatorApp.__new__(OperatorApp)
        app._pending_fork_receipt = ("forkaaaaaaaa", "switched to fork forkaaaaaaaa — …")
        emitted: list[str] = []
        app._notice = lambda body, kind="info": emitted.append(body)  # type: ignore[method-assign]

        OperatorApp._flush_fork_receipt(app, "forkaaaaaaaa")

        assert len(emitted) == 1
        assert app._pending_fork_receipt is None, "the receipt must fire exactly once"

    def test_a_failed_transition_drops_the_stash(self) -> None:
        """The other half: the id gate stops a WRONG narration, and this stops
        a stash from lingering until some later boot happens to match."""
        import inspect

        from local_operator.tui.app import OperatorApp

        source = inspect.getsource(OperatorApp._reload_session)
        assert (
            "self._pending_fork_receipt = None" in source
        ), "the failed-transition branch does not drop the fork receipt"


class TestTheForkInheritsItsParentsCacheKey:
    """R9. The fork's warm-prefix inheritance, pinned at the two places it can
    silently break.

    A fork replays a byte-identical transcript, so its first request reproduces
    the parent's cached prefix exactly and must be ROUTED to it. On the
    OpenAI-shaped wire that routing is a ``prompt_cache_key``, and it is
    inherited from the parent rather than defaulted to the fork's own id.

    Nothing in the suite covered this before, and its failure is entirely
    silent: a fork that loses the key still works, still answers, and merely
    pays full price for a prefix that was already warm. Measured live on
    ``openai/gpt-5.4`` over a real Responses endpoint: the inherited key read
    the parent's prefix in every trial of every run (10/10 in the canonical
    N=10, and never a miss across four runs and a dozen probes), so losing it
    is a real exposure. What that endpoint could NOT establish is that the key
    is NECESSARY — the control arms are unstable there (a wrong-key repeat
    ranged 3/8 to 10/10 across runs), because the underlying cache is
    content-addressed and the key is a routing hint, not a lock. The honest
    claim is sufficiency-with-never-a-cost, which is what shipping the
    inheritance needs; the evidence is in ``docs/evidence/fork-cache/``.

    Two distinct failure modes are pinned:

    1. **The boot path.** ``session_factory`` is what turns ``origin.json`` into
       a ``cache_lineage_id``. A synthetic ``SessionStreamFn`` built by hand in
       a test can pass while the REAL boot never plumbs the id through, so the
       wiring is asserted against the factory's own source rather than against a
       stand-in.
    2. **The scope separation.** The cache key and the sticky-credential scope
       must remain SEPARATE values. Collapsing them (an inviting "simplification"
       — they are equal for every non-fork session) would make two sessions
       sharing a cache key also share a PINNED CREDENTIAL ROW, which is a
       genuine bug rather than a lost optimisation.
    """

    def _stream_fn(self, session_dir: Path):
        """A stream fn wired EXACTLY as ``session_factory`` wires one."""
        from local_operator.model.configure import create_stream_fn

        class _Auth:  # only the resolved ids are inspected; nothing is streamed
            pass

        return create_stream_fn(
            _Auth(),  # type: ignore[arg-type]
            settings={},
            session_id=session_dir.name,
            cache_lineage_id=fork_parent(session_dir) or None,
        )

    def test_a_fork_takes_the_parents_cache_key_and_its_own_credential_scope(
        self, tmp_path: Path
    ) -> None:
        _seed_parent(tmp_path)
        fork_id = fork_session(tmp_path, PARENT_ID)
        fork_dir = tmp_path / "sessions" / fork_id

        stream = self._stream_fn(fork_dir)

        assert stream._cache_lineage_id == PARENT_ID, "the fork did not inherit the cache key"
        assert stream._session_id == fork_id
        # The safety property, not a restatement of the line above: sticky
        # credential selection keys on `_session_id` alone, so the fork must not
        # share the parent's pinned credential row.
        assert stream._cache_lineage_id != stream._session_id

    def test_an_ordinary_session_keys_its_cache_on_its_own_id(self, tmp_path: Path) -> None:
        """The default must survive: only a fork diverges the two values."""
        parent = _seed_parent(tmp_path)
        stream = self._stream_fn(parent)

        assert stream._cache_lineage_id == PARENT_ID
        assert stream._session_id == PARENT_ID

    def test_the_real_boot_path_plumbs_the_lineage_id(self) -> None:
        """The highest-risk assumption in the feature, asserted on the REAL
        factory: a hand-built stream fn proves nothing about what boot does."""
        import inspect

        from local_operator import session_factory

        source = inspect.getsource(session_factory)
        assert (
            "cache_lineage_id=fork_parent(transcript_dir) or None" in source
        ), "session_factory no longer derives the fork's cache key from origin.json"

    def test_the_key_reaches_the_openai_wire_body(self, tmp_path: Path) -> None:
        """The inherited id is only worth anything if it is actually SENT.

        ``_build_responses_body`` is the sole place the key reaches the wire,
        gated on ``supports_prompt_cache``; the chat-completions body never
        carries it. This drives the real builder rather than asserting on
        source, so a change to the gate or the field name fails here.
        """
        from local_operator.harness.types import ChatRequest, Message
        from local_operator.model.configure import build_model_spec
        from local_operator.providers.clients import OpenAICompatClient

        _seed_parent(tmp_path)
        fork_id = fork_session(tmp_path, PARENT_ID)
        fork_dir = tmp_path / "sessions" / fork_id
        lineage = self._stream_fn(fork_dir)._cache_lineage_id

        spec = build_model_spec("openai", "gpt-5.4")
        assert spec.supports_prompt_cache, "the gate this key rides on is off for gpt-5.4"

        client = OpenAICompatClient(base_url="https://api.openai.com/v1", openai_api="responses")
        body = client._build_responses_body(
            ChatRequest(
                model=spec,
                system_blocks=["stable prefix"],
                messages=[Message.user("go")],
                prompt_cache_key=lineage,
            )
        )

        assert body["prompt_cache_key"] == PARENT_ID
        assert body["prompt_cache_retention"] == "24h"


class TestTheForkMarkSurvivesTruncation:
    """U1. The shipped ``(fork)`` suffix was truncated away on ordinary titles.

    The suffix lived INSIDE the name field, which is the first thing an
    ellipsis eats. The name is condensed to ``resume.NAME_MAX_CHARS`` (64)
    before the picker sees it and the card's name column measures 48 cells at
    100 columns, so any title over ~40 characters lost the mark at EVERY
    terminal width — 17% of the titles in the operator's real store. That
    returns the twin-row confusion the mark exists to resolve, on exactly the
    long descriptive conversations a user is most likely to fork from.

    The tag now rides in reserved chrome ahead of the name, so what truncates
    is the tail of the title.
    """

    #: 56 characters — past the ~40 at which the old suffix disappeared.
    LONG = "Refactor the YAML loader to stream anchors instead of buf"

    def _rows(self):
        from local_operator.resume import SessionRow

        return [
            SessionRow("a1b2c3d4e5f6", 1_700_000_000.0, self.LONG, forked=True),
            SessionRow("9f8e7d6c5b4a", 1_700_000_000.0, self.LONG),
        ]

    @pytest.mark.parametrize("width", [100, 80, 70, 60])
    def test_the_tag_survives_at_every_width(self, width: int) -> None:
        from local_operator.tui.widgets.session_picker import render_rows

        lines = [line.plain for line in render_rows(self._rows(), 0, width, 1_700_000_000.0)]

        assert "[fork]" in lines[0], f"the mark was truncated away at {width} columns"
        # And the row is no longer identical to its parent's, which is the
        # property the mark exists for.
        assert lines[0].strip() != lines[1].strip()

    def test_unforked_rows_pad_the_column_so_names_stay_aligned(self) -> None:
        """A ragged left edge on the one field read down the list is D2's
        defect moved onto the fork column."""
        from local_operator.tui.widgets.session_picker import render_rows

        lines = [line.plain for line in render_rows(self._rows(), 0, 100, 1_700_000_000.0)]

        assert lines[0].index("Refactor") == lines[1].index("Refactor")

    def test_a_list_with_no_fork_reserves_nothing(self) -> None:
        """The column is chrome only when the RESULT SET has a fork in it, so
        an ordinary store's picker is exactly as wide as it was."""
        from local_operator.resume import SessionRow
        from local_operator.tui.widgets.session_picker import render_rows

        plain = [SessionRow("9f8e7d6c5b4a", 1_700_000_000.0, "Refactor the loader")]
        line = render_rows(plain, 0, 100, 1_700_000_000.0)[0].plain

        assert "[fork]" not in line
        assert line.index("Refactor") == 2, "the name moved for a column nothing uses"

    def test_the_column_is_reserved_from_the_result_set_not_the_page(self) -> None:
        """A fork off-page must still hold the column, or every name jumps
        sideways as it scrolls past — D2's ragged edge on the time axis."""
        from local_operator.resume import SessionRow
        from local_operator.tui.widgets.session_picker import render_rows

        page = [
            SessionRow("9f8e7d6c5b4a", 1_700_000_000.0, self.LONG),
            SessionRow("77c1aa02bd31", 1_700_000_000.0, "Wire the retention sweep"),
        ]
        without = render_rows(page, 0, 80, 1_700_000_000.0, forked=False)[0].plain
        with_col = render_rows(page, 0, 80, 1_700_000_000.0, forked=True)[0].plain

        assert "[fork]" not in with_col, "the page itself has no fork to paint"
        # The reserved blanks push the name right of where it sits without the
        # column, which is the scroll-stability property: names stay put as a
        # fork scrolls in and out of view.
        assert with_col.index("Refactor") > without.index("Refactor")

    def test_the_tag_is_dim_not_name_ink(self) -> None:
        """D7. At name weight it read as part of the title — as though the
        conversation were called \"Refactor the loader (fork)\". It is metadata
        about the row, so it takes the ink the age and the id already use."""
        from rich.style import Style

        from local_operator.tui import theme as theme_mod
        from local_operator.tui.widgets.session_picker import render_rows

        line = render_rows(self._rows(), 0, 100, 1_700_000_000.0)[0]
        tag_style = next(
            span.style
            for span in line.spans
            if line.plain[span.start : span.end].strip() == "[fork]"
        )

        assert isinstance(tag_style, Style)
        assert tag_style.color is not None
        assert tag_style.color.name == theme_mod.semantic_color("dim")
        # And a step quieter than the body-match marker, which is `muted`
        # because it is the only thing explaining an otherwise unmatched row.
        assert tag_style.color.name != theme_mod.semantic_color("muted")


class TestAForkIsFindableByTypingFork:
    """U3. The mark was spliced in at render time, so ``filter_rows`` could not
    see it: a user who read ``[fork]`` on screen and typed it got ZERO rows —
    a picker saying "no session matches" about a store full of visibly tagged
    forks, which reads as a broken filter rather than an unsupported query.
    """

    def _rows(self):
        from local_operator.resume import SessionRow

        return [
            SessionRow("a1b2c3d4e5f6", 2.0, "Refactor the loader", forked=True),
            SessionRow("9f8e7d6c5b4a", 1.0, "Refactor the loader"),
        ]

    def test_the_filter_admits_a_tagged_fork(self) -> None:
        from local_operator.tui.widgets.session_picker import filter_rows

        assert [row.id for row in filter_rows(self._rows(), "fork")] == ["a1b2c3d4e5f6"]

    def test_the_bracket_form_matches_too(self) -> None:
        """What is on screen is what is typed."""
        from local_operator.tui.widgets.session_picker import filter_rows

        assert [row.id for row in filter_rows(self._rows(), "[fork]")] == ["a1b2c3d4e5f6"]

    def test_a_fork_admitted_on_its_tag_ranks_in_the_name_tier(self) -> None:
        """Ranked through the same composition it was admitted by, or it would
        fall to the soft tier and sort below every incidental body hit."""
        from local_operator.resume import SessionRow
        from local_operator.tui.widgets.session_picker import rank_rows

        fork = SessionRow("a1b2c3d4e5f6", 1.0, "Refactor the loader", forked=True)
        body_only = SessionRow("ccc3ccc3ccc3", 9.0, "Unrelated", forked=False)

        ranked = rank_rows([body_only, fork], "fork", {"ccc3ccc3ccc3"})

        assert ranked[0].id == "a1b2c3d4e5f6"

    def test_the_tag_is_not_also_explained_as_a_body_match(self) -> None:
        """The row already carries a visible tag; a second mark saying "found
        in the conversation" would contradict it."""
        from local_operator.tui.widgets.session_picker import matched_in_body

        rows = self._rows()
        assert matched_in_body(rows[0], "fork", {"a1b2c3d4e5f6"}) is False

    def test_an_ordinary_row_is_unaffected(self) -> None:
        from local_operator.tui.widgets.session_picker import filter_rows

        assert filter_rows(self._rows(), "loader") == self._rows()


class TestTheMobileSurfaceCarriesTheForkMark:
    """U4. ``_past_sessions``/``_search_sessions`` built ``{id, name, mtime}``
    and DISCARDED ``forked``, so the phone's history list kept showing the twin
    identical rows the TUI had stopped showing.
    """

    def test_the_history_payload_carries_it(self, tmp_path: Path, monkeypatch) -> None:
        import local_operator.mobile.daemon as daemon_mod

        _seed_named_parent(tmp_path, "Refactor the loader")
        fork_id = fork_session(tmp_path, PARENT_ID)
        monkeypatch.setattr("local_operator.paths.config_dir", lambda: tmp_path)

        rows = {row["id"]: row for row in daemon_mod._past_sessions()}

        assert rows[fork_id]["forked"] is True
        assert rows[PARENT_ID]["forked"] is False

    def test_the_search_payload_carries_it_and_matches_on_it(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        import local_operator.mobile.daemon as daemon_mod

        _seed_named_parent(tmp_path, "Refactor the loader")
        fork_id = fork_session(tmp_path, PARENT_ID)
        monkeypatch.setattr("local_operator.paths.config_dir", lambda: tmp_path)

        empty = {row["id"]: row for row in daemon_mod._search_sessions("")}
        assert empty[fork_id]["forked"] is True

        hits = daemon_mod._search_sessions("fork")
        assert [row["id"] for row in hits] == [fork_id]


class TestTheWindowTitleNamesTheFork:
    """U2. A running fork's ``conversation_name`` is EMPTY by design, so every
    host fell back to a label derived from the replayed history — which is the
    PARENT's opening message — or to the cwd. Two cmux sidebar rows then read
    identically, on the one surface a user scans to find the window that just
    opened.
    """

    def test_the_session_exposes_the_inherited_title_state(self, tmp_path: Path) -> None:
        _seed_named_parent(tmp_path, "Refactor the loader")
        fork_id = fork_session(tmp_path, PARENT_ID)

        fork = _build_session(tmp_path / "sessions" / fork_id)
        parent = _build_session(tmp_path / "sessions" / PARENT_ID)

        assert fork.wears_inherited_title is True
        assert fork.conversation_name == "", "the empty name is what makes the tag necessary"
        assert parent.wears_inherited_title is False

    def test_naming_the_fork_clears_it(self, tmp_path: Path) -> None:
        """The tag marks the ambiguous STATE, not ancestry."""
        _seed_named_parent(tmp_path, "Refactor the loader")
        fork_id = fork_session(tmp_path, PARENT_ID)
        fork = _build_session(tmp_path / "sessions" / fork_id)

        fork.set_conversation_name("Streaming parser attempt", user_set=False)

        assert fork.wears_inherited_title is False

    def test_the_tab_of_an_unnamed_fork_carries_the_mark(self) -> None:
        """Through the REAL band and the REAL title writer, so the fallback
        chain and the rendered OSC string are the live ones.

        The fork's own name is empty, so the label resolves to the cwd — the
        same string the parent's tab would show. That is the collision."""
        forked = _title_written(_titled_band(forked=True))
        ordinary = _title_written(_titled_band(forked=False))

        assert forked == "lo \u203a [fork] lop-forkux"
        assert ordinary == "lo \u203a lop-forkux"
        assert forked != ordinary, "the fork's tab still reads as the parent's"

    def test_a_named_session_tab_is_untouched(self) -> None:
        """The tag rides on the unnamed-fork state, so a session with a name of
        its own — including a fork that has named itself — is unaffected."""
        assert _title_written(_titled_band(name="Refactor the loader")) == (
            "lo \u203a Refactor the loader"
        )

    def test_a_live_rename_clears_the_tab_mark(self, tmp_path: Path) -> None:
        """R1. `set_conversation_name` clears the session flag, but the TAB is
        painted by `StatusLine`, whose `update` treats a missing `forked=` as
        leave-alone — so a `_cmd_rename` that pushed only the name left the
        band's stale `True` in force and the tab kept prefixing `[fork]` onto
        the name the user just typed. The disk picker clears from
        `title.json`'s mtime, so the live window and `/resume` disagreed about
        the same session.

        Driven through the REAL `_cmd_rename` — not through
        `set_conversation_name` alone, which is the gap the prior test left —
        with a REAL `Session` over a REAL fork directory and a REAL
        `TerminalTitle` writer, so the assertion is the rendered OSC string.
        """
        from local_operator.tui.app import OperatorApp

        _seed_named_parent(tmp_path, "Refactor the loader")
        fork_id = fork_session(tmp_path, PARENT_ID)
        fork = _build_session(tmp_path / "sessions" / fork_id)
        assert fork.wears_inherited_title is True

        app = OperatorApp.__new__(OperatorApp)
        app._session = fork
        app._provisional_name = ""
        app._status = _titled_band(forked=True)
        # `_cmd_rename` ends with a best-effort phone push; a `None` handle is
        # the no-phone state, which is what this double wants.
        app._mobile_handle = None
        notices: list[str] = []

        def _notice(body: str, kind: str = "info") -> None:
            # Matches the ``NoticeFn`` protocol (``kind`` is part of its shape);
            # only the body is asserted here.
            notices.append(body)

        OperatorApp._cmd_rename(app, "Streaming parser attempt", _notice)

        written = _title_written(app._status)
        assert fork.wears_inherited_title is False
        assert (
            "[fork]" not in written
        ), f"the tab still announces a borrowed title after /rename: {written!r}"
        assert written == "lo \u203a Streaming parser attempt"


class TestAPendingForkIsVisibleAndRevocable:
    """U6. The deferred acknowledgement scrolls away behind a long tool run,
    leaving nothing on screen saying a fork is armed — and then a window opens
    minutes later with no visible cause. Esc has always withdrawn it, and
    nothing ever said so.
    """

    def test_the_deferred_receipt_names_the_escape(self) -> None:
        import inspect

        from local_operator.tui.app import OperatorApp

        source = inspect.getsource(OperatorApp._cmd_fork)

        assert "esc to cancel" in source

    def test_the_band_shows_a_pending_fork(self) -> None:
        from local_operator.tui.widgets.status_line import FORK_PENDING_TEXT, StatusLine

        band = _band()
        band._fork_pending = True

        row = StatusLine._render(band, 120).plain

        assert FORK_PENDING_TEXT in row

    def test_an_idle_band_says_nothing_about_forking(self) -> None:
        from local_operator.tui.widgets.status_line import FORK_PENDING_TEXT, StatusLine

        assert FORK_PENDING_TEXT not in StatusLine._render(_band(), 120).plain


class TestTheOpenedReceiptSaysHowToReachTheFork:
    """U5. The success receipt named the place and stopped there, which made it
    strictly LESS actionable than the failure receipt beside it — that one
    hands the user a ``lop --resume`` command. The fork is deliberately never
    focused, so "a workspace exists somewhere" leaves the user asking why
    nothing happened.
    """

    def test_it_says_not_focused_and_how_to_reach_it(self) -> None:
        import inspect

        from local_operator.tui.app import OperatorApp

        source = inspect.getsource(OperatorApp._on_fork_complete)

        assert "(not focused)" in source
        assert "lop --resume {fork_id}" in source


def _band():
    """A ``StatusLine`` with the fields ``_render`` reads, and no widget."""
    from local_operator.tui.widgets.status_line import McpStatus, StatusLine

    band = StatusLine.__new__(StatusLine)
    band._model_label = "test/model"
    band._model_name = ""
    band._effort = ""
    band._agent_profile = ""
    band._team = ""
    band._cwd = "/tmp"
    band._context_tokens = 0
    band._context_is_estimate = False
    band._context_window = 0
    band._subagents = 0
    band._jobs = 0
    # `_render` reads this even when the deferred-history segment is empty
    # (0 = absent). The helper predates that rung, so it has to seed the
    # field the same way it seeds `_jobs` — `__new__` skips `__init__`.
    band._deferred = 0
    band._streaming = False
    band._cost = ""
    band._conversation_name = ""
    band._forked = False
    band._fork_pending = False
    band._mcp = McpStatus()
    band._dropped = frozenset()
    band._approvals_auto = False
    band._approvals_always = False
    band._attention = False
    band._active_seconds = 0.0
    band._turn_started_at = None
    band._spinner_index = 0
    band._subagent = None
    band._title = None
    return band


def _titled_band(*, forked: bool = False, name: str = ""):
    """A ``StatusLine`` wired to a REAL ``TerminalTitle`` over a capture sink."""
    from local_operator.tui.terminal_title import TerminalTitle
    from local_operator.tui.widgets.status_line import StatusLine
    from tests.unit.tui.test_status_line import FakeDock

    band = _band()
    band._conversation_name = name
    band._cwd = "/tmp/lop-forkux"
    band._forked = forked
    # `update()` repaints through the dock (``FakeDock`` mirrors the three
    # things StatusLine asks of its widget; `cast` because the double stands
    # in for a `Static` exactly as test_status_line's own `_dock` does), so a
    # caller driving the REAL command path — not just `_sync_terminal_title`
    # by hand — works too.
    band._dock = cast(Static, FakeDock())
    # A real writer over a sink that goes nowhere: what is asserted is the
    # rendered title (``TerminalTitle.current``), and the sink only keeps the
    # OSC escape out of the test's stdout.
    band._title = TerminalTitle(lambda _escape: None)
    assert isinstance(band, StatusLine)
    return band


def _title_written(band) -> str:
    """The label the band pushes at the title writer, as ``build_title`` renders
    it — read back off the writer rather than off a stub, so the assertion is
    about the string that reaches the terminal."""
    from local_operator.tui.widgets.status_line import StatusLine

    StatusLine._sync_terminal_title(band)
    return band._title.current
