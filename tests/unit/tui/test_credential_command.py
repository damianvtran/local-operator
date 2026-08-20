"""``/credential``: list, store, forget — and never echo the value.

The command is the operator-facing half of session credentials. The value is
captured through the same masked paste the login flow already uses, so these
tests pin the verbs that do not need a paste and the one path that does: the
receipt names the KEY, the store holds the value, and the painted frame never
contains the secret.
"""

from __future__ import annotations

import pytest

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
async def test_cred_alias_runs_the_same_handler() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        app._run_slash_command("/cred")
        await pilot.pause()
        notices = _notice_texts(app)
    assert any("No credentials stored" in n for n in notices), notices
    assert not any("unknown command" in n for n in notices)
