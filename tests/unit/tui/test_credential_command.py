"""``/credential``: list, store, forget — and never echo the value.

The command is the operator-facing half of session credentials. The value is
captured through the same masked paste the login flow already uses, so these
tests pin the verbs that do not need a paste and the one path that does: the
receipt names the KEY, the store holds the value, and the painted frame never
contains the secret.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from local_operator.harness.types import CustomMessage, StreamEndEvent
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.key_prompt import MASK_CHAR, KeyPromptBlock
from tests.unit.tui.test_app_pilot import FakeSession, _factory
from tests.unit.tui.test_slash_echo import _boot, _notice_texts, _submit


def _painted(app: OperatorApp) -> str:
    return "\n".join(strip.text for strip in app.screen._compositor.render_strips())


@pytest.mark.asyncio
async def test_bare_credential_lists_nothing_on_a_fresh_session() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/credential")
        notices = _notice_texts(app)
    assert any("No credentials stored" in n for n in notices), notices


@pytest.mark.asyncio
async def test_forget_on_an_unknown_key_says_so() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/credential --forget GITHUB_TOKEN")
        notices = _notice_texts(app)
    assert any("No credential named GITHUB_TOKEN" in n for n in notices), notices


@pytest.mark.asyncio
async def test_storing_a_credential_masks_the_value_and_names_the_key() -> None:
    """The reported path: ``/credential GITHUB_TOKEN``, paste, enter.

    The secret must not appear in the transcript, the painted frame, or any
    notice. The store must hold it under the normalised key.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    secret = "ghp_this_must_never_paint"
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/credential github token")
        await pilot.pause()
        prompts = list(app.query(KeyPromptBlock))
        assert prompts, "the store verb must open the masked paste"
        for char in secret:
            await pilot.press(char)
        await pilot.pause()
        painted = _painted(app)
        assert secret not in painted, painted
        assert MASK_CHAR * len(secret) in painted, painted
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        notices = _notice_texts(app)
        painted_after = _painted(app)
    assert any("Stored GITHUB_TOKEN" in n for n in notices), notices
    assert secret not in painted_after
    assert secret not in "\n".join(notices)
    assert session.variables.credential_names() == ["GITHUB_TOKEN"]
    assert session.variables.credential_env()["GITHUB_TOKEN"] == secret


@pytest.mark.asyncio
async def test_storing_a_credential_announces_it_to_the_session_journal() -> None:
    """The model-visible half of /credential: the store write alone changes
    only the system-prompt tail, which the model has no reason to re-read.
    A real session journals a ``session_credential`` record naming the KEY
    (never the value) so the next turn — and a resume — see it."""
    from local_operator.incidents import SESSION_CREDENTIAL_MESSAGE_TYPE
    from tests.unit.session.test_session import ScriptedStream
    from tests.unit.session.test_session import make_session as make_real_session

    # A REAL session, not the FakeSession the pilot tests boot with: the
    # journal lives on Session (``journal_credential_change``), and the fake
    # has no journal to call.
    real = make_real_session(
        Path(tempfile.mkdtemp()), ScriptedStream([[StreamEndEvent(stop_reason="stop")]])
    )
    secret = "ghp_never_in_the_journal"

    real.journal_credential_change("GITHUB_TOKEN")

    records = [
        m
        for m in real._context.messages
        if isinstance(m, CustomMessage) and m.custom_type == SESSION_CREDENTIAL_MESSAGE_TYPE
    ]
    assert len(records) == 1
    assert "GITHUB_TOKEN" in records[0].details["text"]
    assert secret not in records[0].details["text"]
    assert records[0].details["action"] == "stored"
    await real.dispose()


@pytest.mark.asyncio
async def test_forget_removes_a_stored_credential() -> None:
    session = FakeSession()
    session.variables.store_credential("GITHUB_TOKEN", "ghp_keep_me_out", "command")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/credential")
        listed = _notice_texts(app)
        await _submit(pilot, app, "/credential --forget GITHUB_TOKEN")
        forgotten = _notice_texts(app)
    assert any("GITHUB_TOKEN" in n and "from /credential" in n for n in listed), listed
    assert any("Forgot GITHUB_TOKEN" in n for n in forgotten), forgotten
    assert session.variables.credential_names() == []


@pytest.mark.asyncio
async def test_forget_all_announces_every_removal_to_the_journal() -> None:
    """R1: ``--forget-all`` must announce EACH removed key, not zero of
    them. The handler used to iterate ``credential_names()`` AFTER
    ``clear_credentials()`` emptied the store, so the loop was dead code and
    the model kept believing every key still existed — on the one verb whose
    whole job is to say they are gone. A journalling fake pins the TUI seam
    (``_cmd_credential`` → ``_journal_credential_change`` → the session hook)
    rather than calling the session method directly."""
    from local_operator.variables import VariableStore

    class JournalingSession(FakeSession):
        """FakeSession with the journal hook the real Session exposes, so the
        test drives the same getattr seam the TUI handler uses."""

        def __init__(self) -> None:
            super().__init__()
            self.credential_journal: list[tuple[str, str]] = []

        def journal_credential_change(
            self, key: str, *, action: str = "stored", **_: object
        ) -> None:
            self.credential_journal.append((key, action))

    session = JournalingSession()
    # The fake builds a bare VariableStore on first property access; stage the
    # two keys before the command runs so the clear has something to remove.
    store = session.variables
    assert isinstance(store, VariableStore)
    store.store_credential("GITHUB_TOKEN", "ghp_one", "command")
    store.store_credential("ANTHROPIC_API_KEY", "sk-ant-two", "command")

    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/credential --forget-all")
        notices = _notice_texts(app)

    assert any("Forgot 2 credentials" in n for n in notices), notices
    assert session.credential_journal == [
        ("GITHUB_TOKEN", "forgot"),
        ("ANTHROPIC_API_KEY", "forgot"),
    ], session.credential_journal
    assert store.credential_names() == []


@pytest.mark.asyncio
async def test_cred_alias_runs_the_same_handler() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        app._run_slash_command("/cred")
        await pilot.pause()
        notices = _notice_texts(app)
    assert any("No credentials stored" in n for n in notices), notices
    assert not any("unknown command" in n for n in notices)


@pytest.mark.asyncio
async def test_the_credential_prompt_speaks_credential_not_login(tmp_path) -> None:
    """UX round 1, U5: the masked card is reused from ``/login``, and it said so.

    ``Paste your DB_PASSWORD API key`` names a key no provider issued, and on
    Esc it painted ``DB_PASSWORD login cancelled`` AND a second ``not stored``
    notice — two rows, one about a login the user never started. The card
    now knows it is holding a credential: it asks for the VALUE, and its own
    settled receipt is the single cancel row.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/credential DB_PASSWORD")
        await pilot.pause()
        painted = _painted(app)
        assert "Paste the value for DB_PASSWORD" in painted, painted
        assert "API key" not in painted, painted
        assert "paste or type the value" in painted, painted
        await pilot.press("escape")
        for _ in range(3):
            await pilot.pause()
        painted = _painted(app)
        notices = _notice_texts(app)
    assert "login cancelled" not in painted, painted
    assert "DB_PASSWORD not stored" in painted, painted
    # Exactly one cancel row: the card's receipt, and no doubled notice.
    assert painted.count("not stored") == 1, painted
    assert not any("Cancelled;" in n for n in notices), notices
    assert session.variables.credential_names() == []


@pytest.mark.asyncio
async def test_the_credential_receipt_counts_a_value_not_a_key() -> None:
    """The success receipt keeps the length-only shape, in credential words."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    placeholder = "x" * 12
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/credential DEPLOY_TOKEN")
        await pilot.pause()
        for char in placeholder:
            await pilot.press(char)
        await pilot.press("enter")
        for _ in range(3):
            await pilot.pause()
        painted = _painted(app)
    assert "DEPLOY_TOKEN value received (12 chars)" in painted, painted
    assert "key received" not in painted, painted
