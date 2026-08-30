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

import pytest

from local_operator.fork import (
    BOOT_PROMPT_NAME,
    COPIED_SIDECARS,
    EXCLUDED_SIDECARS,
    ForkError,
    consume_boot_prompt,
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
