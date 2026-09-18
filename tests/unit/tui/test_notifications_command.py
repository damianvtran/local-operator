"""``/notifications``: the unread completions this terminal is painting.

The command exists because the pile of checkmarks a sidebar can show is cleared
one conversation at a time today, and the report it answers is "clear the pile I
can SEE". Every test below is therefore about ONE set: the rows the listing
prints are the rows the clearing form acknowledges, and anything the store says
about that set is named rather than rounded into a clean sweep.

Driven through the real ``OperatorApp`` against a real ``AttentionStore`` and a
real session catalogue, because the parts that can silently rot are the ones
between them: which entries ``load_catalog`` reports as ``unseen``, which tokens
they carry, and whether the receipt is written after the call rather than before
it. A test that called the handler's helpers directly would pass with the
command wired to nothing.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

import pytest

from local_operator.session.attention import AttentionStore
from local_operator.tui import session_catalog
from local_operator.tui.app import OperatorApp
from local_operator.tui.session_interaction import SessionInteraction
from tests.unit.tui.test_app_pilot import FakeSession, _factory

#: The worker group ``OperatorApp._cmd_notifications`` dispatches under. Named
#: once so the wait below cannot drift onto another group's workers, and scoped
#: to that group rather than taking `app.workers.wait_for_complete()`, which an
#: unrelated long-lived worker would turn into a hang.
_WORKER_GROUP = "notifications"


@pytest.fixture
def config_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An isolated config root, with every ``CMUX_*`` variable scrubbed.

    The scrub is the house rule for anything that boots the TUI: this suite runs
    inside cmux on the maintainer's machine, and an inherited workspace id is how
    a test once renamed the operator's real workspaces.
    """
    root = tmp_path / "config"
    (root / "sessions").mkdir(parents=True)
    for key in ("CMUX_SURFACE_ID", "CMUX_WORKSPACE_ID", "CMUX_SOCKET_PATH", "CMUX_PANE_ID"):
        monkeypatch.delenv(key, raising=False)
    # Patched on the MODULE the handler imports from, which is the only spelling
    # that reaches a call-time `from local_operator.paths import config_dir`.
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: root)
    return root


#: Monotonic birth clock. Module-level so the type checker can see it, and
#: stamped on every fixture session because the catalogue ranks by creation:
#: without it these directories fall through to the filesystem birthtime, which
#: macOS has and Linux does not, so the row ORDER would differ between the two.
_birth_clock: float = 1_700_000_000.0


def _make_session(root: Path, session_id: str, name: str) -> None:
    """A session directory the catalogue can scan, name and rank."""
    from local_operator.resume import write_session_title

    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    global _birth_clock
    _birth_clock += 1.0
    (directory / "created_at.json").write_text(str(_birth_clock))
    (directory / "transcript.jsonl").write_text(
        '{"id":"e1","ts":1,"type":"message",'
        '"payload":{"kind":"message","role":"user","content":[{"text":"go"}]}}\n'
    )
    write_session_title(directory, name, user_set=False, past_names=[])


def _publish(root: Path, session_id: str, *, kind: str = "complete") -> str:
    token = str(uuid.uuid4())
    AttentionStore(root / "attention.db").publish(
        f"session/{session_id}", token, f"anchor-{session_id}", kind
    )
    return token


def _unread_session(root: Path, session_id: str, name: str) -> str:
    _make_session(root, session_id, name)
    return _publish(root, session_id)


def _store(root: Path) -> AttentionStore:
    return AttentionStore(root / "attention.db")


async def _settle(pilot: Any, rounds: int = 4) -> None:
    for _ in range(rounds):
        await pilot.pause()


async def _await_notifications(app: OperatorApp) -> None:
    """Wait for the receipt's worker to reach its end.

    The handler's scan runs on a worker thread, so the notice it appends lands
    after the dispatching call has already returned. Waiting on the worker is
    waiting on the app's own completion signal; a fixed number of ``pause()``
    rounds is measured in LOOP TURNS while the scan costs WALL TIME, so on a
    contended runner the pauses run out first and the assertion reads the
    transcript before its receipt exists.
    """
    workers = [worker for worker in app.workers if worker.group == _WORKER_GROUP]
    if workers:
        await asyncio.gather(*(worker.wait() for worker in workers))


def _notices(app: OperatorApp) -> list[str]:
    from local_operator.tui.widgets.transcript import NoticeBlock

    return [
        block._text for block in app._transcript_view().blocks() if isinstance(block, NoticeBlock)
    ]


def _painted(app: OperatorApp) -> str:
    return "\n".join(strip.text for strip in app.screen._compositor.render_strips())


async def _run(pilot: Any, app: OperatorApp, command: str) -> None:
    app._run_slash_command(command)
    await _await_notifications(app)
    await _settle(pilot)


@pytest.mark.asyncio
async def test_the_listing_names_the_count_and_stops_at_ten_rows(config_root: Path) -> None:
    """A receipt, bounded: the header counts them all, the rows do not.

    Thirteen unread completions, and the listing spends ten rows on them. The
    bound is not tidiness — the notice is one block in the transcript, and the
    last line is the form that CLEARS them, so an unbounded listing would push
    the answer off the frame.
    """
    for index in range(13):
        _unread_session(config_root, f"{index + 1:012x}", f"conversation {index}")

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _run(pilot, app, "/notifications")
        text = _notices(app)[-1]

    lines = text.splitlines()
    assert lines[0] == "13 unread completions:"
    assert len([line for line in lines if "✓" in line]) == 10
    assert lines[-2] == "  …3 more"
    assert lines[-1] == "/notifications read clears these"
    # A row carries the row's own name, kind and age — the same three facts the
    # picker's row shows, in the order a reader scans them.
    assert any("conversation 12" in line and "— complete ·" in line for line in lines)


@pytest.mark.asyncio
async def test_a_zero_state_says_so_and_names_no_gesture(config_root: Path) -> None:
    _make_session(config_root, "00000000000a", "a read conversation")

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _run(pilot, app, "/notifications")
        assert _notices(app)[-1] == "No unread completions."


@pytest.mark.asyncio
async def test_read_clears_exactly_the_listed_set(config_root: Path) -> None:
    """The listing and the clearing are ONE set, over the real store."""
    ids = ["00000000000a", "00000000000b", "00000000000c"]
    for index, session_id in enumerate(ids):
        _unread_session(config_root, session_id, f"conversation {index}")
    # A delivered-but-unread completion rides along: notifying is not reading,
    # so a clear must not be allowed to claim it changed anything about delivery.
    store = _store(config_root)
    store.claim_delivery(
        f"session/{ids[0]}", store.state(f"session/{ids[0]}")["completion_token"], "tui"
    )

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _run(pilot, app, "/notifications read")
        assert _notices(app)[-1] == "Marked 3 completions read."

    assert [store.state(f"session/{session_id}")["unseen"] for session_id in ids] == [
        False,
        False,
        False,
    ]
    # The delivery watermark is a different fact and stays where it was.
    with __import__("sqlite3").connect(config_root / "attention.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM deliveries").fetchone()[0] == 1


@pytest.mark.asyncio
async def test_the_clearing_receipt_names_what_stayed_unread(
    config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1 through the real command: a completion published after the render.

    The race is injected at the one place it happens — between the catalogue read
    the listing is computed from and the write — so this is the hazard the whole
    design is about, reached by the gesture that could sweep it away.
    """
    settled, racing = "00000000000a", "00000000000b"
    _unread_session(config_root, settled, "finished once")
    _unread_session(config_root, racing, "finished again")
    real = session_catalog.load_catalog

    def load_then_publish(root, *args, **kwargs):
        entries = real(root, *args, **kwargs)
        _publish(config_root, racing)
        return entries

    monkeypatch.setattr(session_catalog, "load_catalog", load_then_publish)

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _run(pilot, app, "/notifications read")
        assert _notices(app)[-1] == (
            "Marked 1 completion read. 1 has a newer result and stays unread."
        )

    store = _store(config_root)
    assert store.state(f"session/{settled}")["unseen"] is False
    assert store.state(f"session/{racing}")["unseen"] is True


@pytest.mark.asyncio
async def test_read_with_nothing_unread_writes_nothing_at_all(config_root: Path) -> None:
    """``Nothing unread.`` is read off the catalogue, not off a zero-item write.

    The store exists here (so this is not the missing-file case): the assertion
    is that its revision does not move, which is what makes the empty form a
    no-op rather than a batch that opens a write transaction to clear nothing.
    """
    session_id = "00000000000a"
    token = _unread_session(config_root, session_id, "already read")
    store = _store(config_root)
    store.acknowledge(f"session/{session_id}", token)
    before = store.revision()

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _run(pilot, app, "/notifications read")
        assert _notices(app)[-1] == "Nothing unread."

    assert store.revision() == before


@pytest.mark.asyncio
async def test_an_unknown_argument_is_refused_by_name(config_root: Path) -> None:
    """One word is the vocabulary, and a second word is refused rather than run.

    ``all`` is the tempting one, and it is exactly the sweep this command does
    not have: the refusal has to say what was typed so the user can see the word
    was read and rejected, not mistyped into the clearing form.
    """
    _unread_session(config_root, "00000000000a", "still unread")
    store = _store(config_root)
    before = store.revision()

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _run(pilot, app, "/notifications all")
        refusal = _notices(app)[-1]

    assert "no such argument" in refusal and "'all'" in refusal
    assert "/notifications read" in refusal, "the refusal must name the form that works"
    assert store.revision() == before, "a refused argument must not clear anything"


@pytest.mark.asyncio
async def test_one_enter_on_the_offered_row_fills_the_command_instead_of_clearing(
    config_root: Path,
) -> None:
    """The irreversible row is one keystroke away from the gesture that lists.

    Pressing space after the command opens the list with ``read`` as its only
    row. Without the row's ``alert`` flag the editor treats a single unambiguous
    match as chosen and RUNS it, so ``/notifications `` + Enter — what a user
    presses when they want to SEE the pile — would clear it, permanently. The
    first Enter fills the buffer; the second runs the word the user can now read.
    """
    _unread_session(config_root, "00000000000a", "still unread")
    store = _store(config_root)

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        from local_operator.tui.widgets.editor import Editor

        editor = app.query_one(Editor)
        editor.focus()
        # The seed idiom the ghost-text suite uses: set the buffer, park the
        # caret at the end and re-sync, THEN type the character under test, so
        # the argument list opens from a real keystroke rather than from a
        # hand-posted message.
        editor.text = "/notifications"
        editor.move_cursor(editor._end_of_buffer())
        editor._sync_picker()
        await _settle(pilot, 6)
        await pilot.press("space")
        await _settle(pilot, 8)

        assert "Mark every unread completion read" in _painted(app)
        assert [choice.name for choice in app.query_one(Editor).picker._choices] == ["read"]

        await pilot.press("enter")
        await _settle(pilot)
        assert editor.text.strip() == "/notifications read"
        assert (
            store.state("session/00000000000a")["unseen"] is True
        ), "one Enter on the offered row cleared the pile"
        # ...and the second Enter, on the word the user can now read, clears it.
        await _run(pilot, app, "/notifications read")
        assert store.state("session/00000000000a")["unseen"] is False


@pytest.mark.asyncio
async def test_the_command_answers_where_a_source_has_no_authority(
    config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Why it is a SAVED LOCAL command: both halves are this machine's.

    ``display_only`` is the state a follower sits in while it binds — the saved
    excerpt is on screen and the socket work has not happened — and it is the
    state every refusal arm above lands in. The marks this command lists are on
    disk in front of the user the whole time, so asking a source for permission
    first would refuse a command that cannot fail for the reason the gate
    exists. Measured rather than assumed: the dispatch below reaches the handler
    while ``_source_commands_ready()`` answers False.
    """
    _unread_session(config_root, "00000000000a", "still unread")
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _settle(pilot, 60)
        source = SessionInteraction(session)
        source.display_only = True
        app._interaction = source
        app._interactions[id(session)] = source
        assert app._source_commands_ready() is False, "the state under test did not arrive"

        reached: list[str] = []
        monkeypatch.setattr(app, "_cmd_notifications", lambda arg, notice: reached.append(arg))
        # The gate and the dispatch both consult the set, and both used to refuse
        # a command whose whole answer is local.
        assert app.composer_submission_blocked("/notifications") is False
        app._run_slash_command("/notifications")
        assert reached == [""]
        # Still refused where a refusal is the truth: a command that needs an
        # owner is not a saved-local one.
        assert app.composer_submission_blocked("/model gpt-4o") is True
