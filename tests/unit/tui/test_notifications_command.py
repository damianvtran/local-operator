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
import time
import uuid
from pathlib import Path
from typing import Any

import pytest

from local_operator.session.attention import AttentionStore
from local_operator.tui import session_catalog
from local_operator.tui.app import (
    OperatorApp,
    _notifications_listing,
    _notifications_store_failure,
)
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
    assert lines[-2] == "  …3 more — ctrl+b shows or hides the sidebar"
    # "all 13" rather than "these": at the bound the short form named the ten
    # rows on screen while the write covered every one of them (design round 1,
    # D2). The short form is what the unbounded case prints, where "these" is
    # literally the whole set.
    assert lines[-1] == "/notifications read marks all 13 read"
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


# -- round 1 remediation: the write is bounded by a render, and a failed store is
# -- never reported as an empty pile --------------------------------------------


def _entry(
    session_id: str,
    name: str,
    mtime: float,
    *,
    unseen: bool = False,
    kind: str = "complete",
    token: str = "",
):
    """One catalogue entry, built by hand so a test can pin a row's age or token.

    The real construction site is ``catalog.entry_for``; this mirrors its field
    order and nothing else, which is what a test of the LISTING's own logic wants
    — the catalogue's derivation is pinned where it lives.
    """
    from local_operator.resume import SessionRow
    from local_operator.session.catalog import CatalogEntry

    return CatalogEntry(SessionRow(session_id, mtime, name), unseen, kind, token, "anchor", "")


def _sqlite_error(message: str, errorname: str):
    """A real ``sqlite3.Error`` wearing the errorname a condition raises.

    ``sqlite_errorname`` is the shared classifier's input on both surfaces, so a
    test that wants "contention" has to supply that attribute rather than
    whichever sentence it hopes to see.
    """
    import sqlite3

    error = sqlite3.OperationalError(message)
    error.sqlite_errorname = errorname
    return error


@pytest.mark.asyncio
async def test_the_clearing_form_paints_the_set_it_writes(
    config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """U1 and U4: the batch is a render the user can see, in the SAME call.

    The clearing form used to be a second, independent catalogue scan. A fourth
    completion published between ``/notifications`` and ``/notifications read``
    was therefore acknowledged having never been painted, and a user who typed
    the space and two Enters cleared a whole pile that was never on screen. Both
    are one defect seen from two sides, and the property that closes them is one
    fact: at the instant of the write, the transcript already carries the rows,
    and the batch is exactly those rows.

    Captured from INSIDE ``acknowledge_many``, because a receipt checked
    afterwards cannot tell a render that preceded the write from one that
    followed it — and the order is the point. The extra completion is injected
    inside the catalogue read itself, which is the window U1 named.
    """
    settled = ["00000000000a", "00000000000b", "00000000000c"]
    latecomer = "00000000000d"
    for index, session_id in enumerate(settled):
        _unread_session(config_root, session_id, f"conversation {index}")
    _make_session(config_root, latecomer, "finished while you read")

    real_catalog = session_catalog.load_catalog

    def load_then_publish(root, *args, **kwargs):
        entries = real_catalog(root, *args, **kwargs)
        _publish(config_root, latecomer)
        return entries

    monkeypatch.setattr(session_catalog, "load_catalog", load_then_publish)

    holder: dict[str, Any] = {}
    at_write: list[tuple[list[str], list[tuple[str, str]]]] = []
    real_ack = AttentionStore.acknowledge_many

    def observed(self, items):  # noqa: ANN001 — mirrors the bound method's shape
        at_write.append((_notices(holder["app"]), list(items)))
        return real_ack(self, items)

    monkeypatch.setattr(AttentionStore, "acknowledge_many", observed)

    app = OperatorApp(lambda: _factory(FakeSession()))
    holder["app"] = app
    async with app.run_test(size=(120, 40)) as pilot:
        await _run(pilot, app, "/notifications read")
        receipt = _notices(app)[-1]

    assert len(at_write) == 1, "the clearing form must write exactly once"
    painted, items = at_write[0]
    block = painted[-1]
    assert block.splitlines()[0] == "3 unread completions:", block
    assert {identity.split("/", 1)[1] for identity, _token in items} == set(settled)
    # The latecomer is not in the batch and is not on the rows either: the write
    # covered what was rendered, and the completion that arrived after that render
    # stays unread for the next listing.
    assert latecomer not in {identity for identity, _token in items}
    assert "finished while you read" not in block
    assert _store(config_root).state(f"session/{latecomer}")["unseen"] is True
    assert receipt == "Marked 3 completions read."


@pytest.mark.asyncio
async def test_an_unreadable_store_is_not_an_empty_pile(config_root: Path) -> None:
    """U2: ``No unread completions.`` is a finding, and a failed read is not one.

    Three genuinely unread completions behind a store this process cannot read
    (the file is there and is not a database — a truncated write). Both forms used
    to answer from an empty attention state the catalogue had swallowed, so the
    operator was told their pile was empty; the clearing form must additionally
    write NOTHING, because a write it cannot verify is not a receipt.
    """
    for index in range(3):
        _unread_session(config_root, f"{index + 1:012x}", f"conversation {index}")
    (config_root / "attention.db").write_bytes(b"not a database")

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _run(pilot, app, "/notifications")
        listing = _notices(app)[-1]
        await _run(pilot, app, "/notifications read")
        clearing = _notices(app)[-1]

    for text, did in ((listing, "listed"), (clearing, "cleared")):
        assert text != "No unread completions." and text != "Nothing unread."
        assert "The read receipts could not be read" in text, text
        # The clearing form says "or written" because it tried to; the listing
        # form never attempted a write and must not claim one (agent review
        # round 2, F1).
        assert (" or written" in text) is (did == "cleared"), text
        assert "Retrying will not help" in text, text
        assert f"nothing was {did}" in text, text
        assert "not a verdict about what is unread" in text, text
    # The store is exactly as unreadable as it was: no schema repair, no write,
    # not even a reread that could have recreated it.
    assert (config_root / "attention.db").read_bytes() == b"not a database"


@pytest.mark.asyncio
async def test_the_write_names_the_condition_it_met(
    config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1: three conditions, three answers, and the copy is the shared module's.

    The handler used to flatten every ``sqlite3.Error`` into the contention
    sentence, so a full volume or an unopenable store told the operator to send
    it again — the misreport the desktop ladder was split to end. Both arms below
    drive the same call path the real store takes; only the errorname differs,
    which is what the shared classifier reads. The pile is on screen BEFORE the
    write either way, so a failure leaves the user looking at what did not clear.
    """
    _unread_session(config_root, "00000000000a", "still unread")
    monkeypatch.setattr(session_catalog, "load_catalog", session_catalog.load_catalog)

    # Busy: retryable, warning ink, the sentence the desktop shows for the same
    # condition.
    def busy(self, items):  # noqa: ANN001
        raise _sqlite_error("database is locked", "SQLITE_BUSY")

    monkeypatch.setattr(AttentionStore, "acknowledge_many", busy)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _run(pilot, app, "/notifications read")
        notices = _notices(app)

    assert "Read state is busy right now" in notices[-1]
    # The remedy, in the sentence, because this surface has no code to carry it
    # (UX round 2, U8).
    assert "Try again in a moment — run /notifications read again." in notices[-1]
    assert "catch up on its own" not in notices[-1], "a user-initiated clear is not self-healing"
    assert "nothing was cleared" in notices[-1]
    assert "not a verdict about what is unread" in notices[-1]
    assert notices[-2].splitlines()[0] == "1 unread completion:", notices[-2]
    assert _store(config_root).state("session/00000000000a")["unseen"] is True

    # Unopenable: NOT retryable, and the sentence says so.
    def unopenable(self, items):  # noqa: ANN001
        raise _sqlite_error("unable to open database file", "SQLITE_CANTOPEN")

    monkeypatch.setattr(AttentionStore, "acknowledge_many", unopenable)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _run(pilot, app, "/notifications read")
        text = _notices(app)[-1]

    assert "Retrying will not help" in text, text
    assert "busy right now" not in text, "the contention sentence leaked into the other arm"
    assert "not a verdict about what is unread" in text, text
    # …and it is the TUI's own sentence, not the send path's: no message is in
    # play in a receipt clear, and nothing is sent (agent review round 2, F1).
    assert "message" not in text and "send it again" not in text, text
    assert _store(config_root).state("session/00000000000a")["unseen"] is True


@pytest.mark.asyncio
async def test_a_store_failure_never_echoes_the_exception_text(
    config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R2: the arm the rule is written for is the one that used to break it.

    The handler states three lines above its catch-all that the store's own
    wording is never echoed because it can name file paths — and then
    interpolated the exception into the transcript. An unexpected failure gets a
    vetted sentence; the diagnostic survives in the log, which is where the
    route's own ladder puts it.
    """
    _unread_session(config_root, "00000000000a", "still unread")
    secret_path = str(config_root / "attention.db")

    def explode(self, items):  # noqa: ANN001
        raise RuntimeError(f"[Errno 28] No space left: {secret_path}")

    monkeypatch.setattr(AttentionStore, "acknowledge_many", explode)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _run(pilot, app, "/notifications read")
        text = _notices(app)[-1]

    assert "attention.db" not in text, text
    assert "Errno" not in text, text
    assert "nothing was cleared" in text, text


@pytest.mark.asyncio
async def test_a_receipt_that_is_not_here_is_named_without_store_vocabulary(
    config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D3: the bucket is a fact about THIS machine, not a past tense.

    A listed row whose token the store does not hold (another root's receipt, a
    store that was replaced) used to be described as "no longer in the receipt
    store" — internal vocabulary, and a claim the bucket does not establish.
    """
    from dataclasses import replace

    _make_session(config_root, "00000000000a", "foreign receipt")
    real = session_catalog.load_catalog

    def load_foreign(root, *args, **kwargs):
        return [
            replace(
                entry, unseen=True, completion_kind="complete", completion_token=str(uuid.uuid4())
            )
            for entry in real(root, *args, **kwargs)
        ]

    monkeypatch.setattr(session_catalog, "load_catalog", load_foreign)

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _run(pilot, app, "/notifications read")
        text = _notices(app)[-1]

    assert "could not be found on this machine and stays unread." in text, text
    assert "receipt store" not in text, text


@pytest.mark.asyncio
async def test_the_rows_are_ordered_by_the_age_they_print(
    config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D5: the printed column IS the sort key, so the bound hides the oldest.

    The catalogue ranks by conversation — newest first — while the row prints the
    completion's age. At the bound that meant the ten newest CONVERSATIONS, so the
    rows "…N more" hid were not the oldest receipts. Ordered by the age the row
    shows, the listing and its bound say the same thing.
    """
    now = time.time()
    entries = [
        _entry("00000000000a", "oldest conversation", now - 7200, unseen=True),
        _entry("00000000000b", "middle conversation", now - 600, unseen=True),
        _entry("00000000000c", "newest conversation", now - 5, unseen=True),
    ]
    monkeypatch.setattr(session_catalog, "load_catalog", lambda *a, **k: list(entries))

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _run(pilot, app, "/notifications")
        text = _notices(app)[-1]

    rows = [line for line in text.splitlines() if line.startswith("  ✓")]
    assert [line.split(" — ")[0].strip("  ✓") for line in rows] == [
        "newest conversation",
        "middle conversation",
        "oldest conversation",
    ], text


def test_a_name_is_truncated_rather_than_wrapped_at_a_narrow_budget() -> None:
    """R4: the fallback was a superset of its neighbour's, and it wrapped.

    ``room`` goes negative in a narrow split, and the guard ``room > 0`` then
    emitted the UNTRUNCATED name — the one outcome the docstring above it rules
    out, in the one case where the budget cannot hold the tail either. Matched to
    ``/stop all``'s condition for the unknown budget, and clamped above it.
    """
    entry = _entry("00000000000a", "reconcile the desktop attention receipts", 1_700_000_000.0)

    narrow = _notifications_listing([entry], 8)
    assert "reconcile the desktop attention receipts" not in narrow, narrow
    assert "…" in narrow, narrow
    # A budget of zero is still "no opinion" (the caller could not measure), and
    # that arm keeps the whole name rather than truncating to a single cell.
    assert "reconcile the desktop attention receipts" in _notifications_listing([entry], 0)


@pytest.mark.asyncio
async def test_a_pending_line_precedes_the_scan(config_root: Path) -> None:
    """U3: the slow leg is the refusal one, and a silent command looks wedged.

    Measured at 4.1 s from Enter to the refusal while another writer held the
    store, with nothing on screen in between. One line, before the scan, the way
    ``/update`` prints "checking for updates…".
    """
    _unread_session(config_root, "00000000000a", "still unread")

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _run(pilot, app, "/notifications")
        notices = _notices(app)

    assert notices[0] == "reading receipts…", notices
    assert notices[-1].startswith("1 unread completion:"), notices[-1]


# -- round 2: share the classification, compose the sentence --------------------


def test_the_copy_is_composed_per_surface_not_borrowed_from_the_send_path() -> None:
    """Ruling: share the codes and the ink, compose the sentence here (F1, U8).

    Round 1's fix shared the desktop's PROSE as well as its codes, and two of
    those strings were written for the send path — "the message could not be
    written", "send it again" — in an operation with no message in it, false
    about the listing form's operation in particular (it never writes). This pins
    the composed sentence for every condition and both forms, including the two
    properties the reviews asked for by name: contention keeps a remedy, and the
    two non-retryable conditions say retrying will not help.
    """
    import logging

    from local_operator.session.store_failures import (
        STORE_BUSY,
        STORE_OUT_OF_SPACE,
        STORE_UNAVAILABLE,
        StoreFailure,
    )

    root = Path("/tmp/lo-copy-pin")
    cases = {
        STORE_BUSY: (logging.WARNING, "warning"),
        STORE_OUT_OF_SPACE: (logging.ERROR, "error"),
        STORE_UNAVAILABLE: (logging.ERROR, "error"),
    }
    for code, (level, ink) in cases.items():
        for clearing in (False, True):
            failure = StoreFailure(500, code, "SENTINEL-PROSE-FROM-THE-SHARED-MODULE", level, False)
            text, kind = _notifications_store_failure(failure, root, clearing=clearing)
            assert kind == ink, (code, clearing, kind)
            # The classifier's own message is NOT the notice: the codes are
            # shared, the prose is this surface's.
            assert "SENTINEL-PROSE-FROM-THE-SHARED-MODULE" not in text, (code, text)
            assert "message" not in text and "send it again" not in text, (code, text)
            assert "not a verdict about what is unread" in text, (code, text)
            assert ("nothing was cleared" if clearing else "nothing was listed") in text, (
                code,
                text,
            )
            if code == STORE_BUSY:
                assert "Try again in a moment" in text, text
                assert "catch up on its own" not in text, text
                assert f"run /notifications{' read' if clearing else ''} again" in text, text
            else:
                assert "Retrying will not help" in text or "Free some space" in text, (code, text)
                assert "Try again in a moment" not in text, (code, text)
    # …and an unclassified failure invents no cause at all.
    for clearing in (False, True):
        text, kind = _notifications_store_failure(None, root, clearing=clearing)
        assert kind == "error"
        assert "Retrying will not help" not in text and "disk space" not in text, text
        assert ("nothing was cleared" if clearing else "nothing was listed") in text, text


@pytest.mark.asyncio
async def test_a_full_volume_names_the_disk_and_the_remedy(
    config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F2: the condition this whole classification exists for, through the app.

    ``SQLITE_FULL`` is the incident the classifier was written for, and this arm
    had no test — which is why the send path's prose reached a receipts surface
    unnamed. The sentence must name the disk, the remedy that is the user's to
    take, and the form to run again, and must never borrow "the message could not
    be written" from a surface that has a message.
    """
    _unread_session(config_root, "00000000000a", "still unread")

    def full(self, items):  # noqa: ANN001
        raise _sqlite_error("database or disk is full", "SQLITE_FULL")

    monkeypatch.setattr(AttentionStore, "acknowledge_many", full)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _run(pilot, app, "/notifications read")
        text = _notices(app)[-1]

    assert "out of disk space" in text, text
    assert "nothing was cleared" in text, text
    assert "run /notifications read again" in text, text
    assert "message" not in text and "send it again" not in text, text
    assert _store(config_root).state("session/00000000000a")["unseen"] is True


@pytest.mark.asyncio
async def test_a_catalogue_that_cannot_be_walked_is_not_an_empty_pile(
    config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F2: the one place ``SessionStoreUnavailable`` is classified on this surface.

    It carries no ``errno``, so the shared classifier answers ``None`` — and the
    notice must then invent no cause: no retry claim, no disk story, just the fact
    that the receipts were not read and that this is not a verdict. Pinned because
    the arm was untested and the answer rests on ``store_failure``'s ``None``
    contract.
    """
    from local_operator.session.errors import SessionStoreUnavailable

    def unavailable(*args, **kwargs):  # noqa: ANN002, ANN003
        raise SessionStoreUnavailable("the store could not be walked")

    monkeypatch.setattr(session_catalog, "load_catalog", unavailable)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _run(pilot, app, "/notifications read")
        clearing = _notices(app)[-1]
        await _run(pilot, app, "/notifications")
        listing = _notices(app)[-1]

    for text in (clearing, listing):
        assert "could not be read" in text, text
        assert (
            "Retrying will not help" not in text
        ), "a cause was invented for an unclassified failure"
        assert "not a verdict about what is unread" in text, text
    assert "nothing was cleared" in clearing, clearing
    assert "nothing was listed" in listing, listing


@pytest.mark.asyncio
async def test_the_clearing_form_does_not_instruct_the_user_who_just_ran_it(
    config_root: Path,
) -> None:
    """U7 and F4: the block paints the rows, not the next instruction.

    Sharing one composition is what gives "these" a referent, but the clearing
    form was also reprinting the listing's call-to-action — the command telling
    the user to run the command that is running — and its pointer to rows that
    are about to stop being unread. The rows stay; the instruction and the
    pointer go.
    """
    for index in range(13):
        _unread_session(config_root, f"{index + 1:012x}", f"conversation {index}")

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        # The LISTING form keeps the instruction and the pointer — that is what
        # the flag switches off, so both arms are read from one app.
        await _run(pilot, app, "/notifications")
        listed = _notices(app)[-1]
        await _run(pilot, app, "/notifications read")
        block = _notices(app)[-2]
        receipt = _notices(app)[-1]

    assert "ctrl+b shows or hides the sidebar" in listed, listed
    assert "/notifications read marks all 13 read" in listed, listed
    assert block.splitlines()[0] == "13 unread completions:", block
    assert "  …3 more" in block, block
    assert "ctrl+b" not in block, block
    assert "/notifications read marks" not in block, block
    assert receipt == "Marked 13 completions read."
    assert set(block.splitlines()[1:-1]) == set(
        listed.splitlines()[1:-2]
    ), "the clearing form must paint the same rows the listing did"


def test_no_painted_row_exceeds_the_budget() -> None:
    """F3: the claim is about the ROW, and the fixed cells are cells too.

    At a budget below ``lead + tail`` no amount of name truncation can fit the
    row, so the clamp that bounded only the name cell left the "cannot wrap" claim
    false exactly where a narrow split lands. The composed row is bounded now.
    """
    from rich.cells import cell_len

    entries = [
        _entry(f"{index + 1:012x}", "reconcile the desktop attention receipts", 1_700_000_000.0)
        for index in range(3)
    ]
    for budget in (8, 20, 60):
        text = _notifications_listing(entries, budget)
        rows = [
            line
            for line in text.splitlines()
            if line.startswith("  ") and not line.lstrip().startswith("…")
        ]
        assert len(rows) == 3, (budget, text)
        for line in rows:
            assert cell_len(line) <= budget, (budget, line)
    # Zero is the caller's "could not measure": the row is emitted whole rather
    # than truncated to a cell, exactly as ``/stop all``'s listing does.
    assert "reconcile the desktop attention receipts" in _notifications_listing(entries, 0)
