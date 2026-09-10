"""Ctrl+C cancels a pending `/login`, and takes nothing else with it.

The reported scenario: `/login anthropic` opens the browser, the browser lands
in the provider's own portal instead of coming back to the loopback redirect,
the user closes the tab — and the app is parked with no way out but the 300 s
`DEFAULT_TIMEOUT_SECONDS`.

Half of this file is the fix and half is the LADDER REGRESSION GUARD, which is
the part worth reading twice. `action_interrupt` is a ladder of rungs that each
claim Ctrl+C under some condition, and this area's entire history is rungs
stealing each other's presses — a draft cleared when the user meant to
interrupt, an exit ladder made unreachable, an exit ladder armed when the press
was absorbed. So the new rung is pinned on three axes:

* it takes the press when a login is pending,
* it does NOT arm the double-tap exit ladder when it does (a user
  reflex-double-tapping to kill a stuck login must not quit the app — measured
  on the pre-change tree, where one press rendered "ctrl+c again to exit" while
  the login stayed pending), and
* with no login pending, every rung below it behaves byte-identically to
  before.

The tests drive the REAL `OperatorApp` with the real `ProviderController` over
a throwaway store, because the defect lived in the seam between the app's key
handling and the provider layer and a stub standing in for either end would be
a test of the stub.
"""

from __future__ import annotations

import asyncio
import socket
from pathlib import Path
from typing import Any

import pytest

from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.key_prompt import KeyPromptBlock
from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView
from tests.unit.tui.test_app_pilot import FakeSession, _factory

pytestmark = pytest.mark.asyncio


def _controller(tmp_path: Path) -> Any:
    from local_operator.providers.auth_store import AuthStore
    from local_operator.providers.controller import ProviderController

    return ProviderController(AuthStore(tmp_path / "auth.db"))


def _notices(app: OperatorApp) -> list[str]:
    return [
        block._text
        for block in app.query_one(TranscriptView).blocks()
        if isinstance(block, NoticeBlock)
    ]


async def _boot(pilot: Any, app: OperatorApp) -> None:
    for _ in range(60):
        await pilot.pause()
        if app._session is not None:
            return


async def _start_login(pilot: Any, app: OperatorApp, provider: str) -> None:
    editor = app.query_one(Editor)
    editor.text = f"/login {provider}"
    await pilot.pause()
    if editor.picker.is_open():
        await pilot.press("escape")
        await pilot.pause()
    await pilot.press("enter")


async def _settle(predicate: Any, limit: int = 250) -> bool:
    """Pump until the predicate holds, bounded in TURNS rather than seconds."""
    for _ in range(limit):
        await asyncio.sleep(0.01)
        if predicate():
            return True
    return False


def _port_held(port: int) -> bool:
    probe = socket.socket()
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        probe.bind(("127.0.0.1", port))
        return False
    except OSError:
        return True
    finally:
        probe.close()


@pytest.fixture
def no_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("webbrowser.open", lambda *a, **k: True)


async def test_ctrl_c_cancels_a_pending_login(tmp_path: Path, no_browser: None) -> None:
    """The reported bug: a parked login now ends on one press.

    `alibaba` is the provider here because its login is a pure paste prompt,
    which parks deterministically with no port to bind — the loopback half is
    covered separately below, where the port is the assertion.
    """
    controller = _controller(tmp_path)
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=controller)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _start_login(pilot, app, "alibaba")
        assert await _settle(lambda: bool(app.query(KeyPromptBlock))), _notices(app)
        assert app._login_signal is not None, "a pending login must publish its signal"

        await pilot.press("ctrl+c")

        assert await _settle(
            lambda: any("login cancelled" in note for note in _notices(app))
        ), _notices(app)
        notices = _notices(app)
        assert not any("failed" in note for note in notices), notices
        assert app._login_signal is None, "the signal must be cleared when the flow ends"
        lock = app._login_lock
        assert (
            lock is not None and not lock.locked()
        ), "a held lock refuses the retry the message invites"

    assert controller.auth_store.list_credentials(provider=None) == []


async def test_the_cancel_message_says_how_to_retry(tmp_path: Path, no_browser: None) -> None:
    """A user who just watched their browser dump them somewhere unexpected
    needs to know the local listener is gone and how to try again — "cancelled"
    on its own answers neither."""
    controller = _controller(tmp_path)
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=controller)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _start_login(pilot, app, "anthropic")
        assert await _settle(lambda: app._login_signal is not None)
        await pilot.press("ctrl+c")
        assert await _settle(lambda: any("login cancelled" in n for n in _notices(app)))

        message = next(n for n in _notices(app) if "login cancelled" in n)
        assert "/login anthropic" in message, message
        assert "listener stopped" in message, message


async def test_a_paste_only_provider_is_not_told_a_listener_stopped(
    tmp_path: Path, no_browser: None
) -> None:
    """The clause is conditional so that it stays TRUE: `alibaba` never binds a
    port, and a confident wrong detail costs trust in the rest of the line."""
    controller = _controller(tmp_path)
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=controller)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _start_login(pilot, app, "alibaba")
        assert await _settle(lambda: app._login_signal is not None)
        await pilot.press("ctrl+c")
        assert await _settle(lambda: any("login cancelled" in n for n in _notices(app)))

        message = next(n for n in _notices(app) if "login cancelled" in n)
        assert "/login alibaba" in message
        assert "listener stopped" not in message


async def test_the_cancel_releases_the_port_and_a_retry_works(
    tmp_path: Path, no_browser: None
) -> None:
    """The property the whole change is for: cancelled means GONE.

    A cancel that raised but left the listener up would be worse than the
    timeout it replaces — the retry would fail with "port is required for this
    login flow but is already in use", and Anthropic pins 54545 so nothing
    falls back to hide it. The port is probed with a real bind rather than read
    off a flag: "the flow says it stopped" and "the OS will let the next login
    have the port" are different claims.
    """
    port = 54545
    if _port_held(port):
        pytest.skip("port 54545 is already in use on this machine")

    controller = _controller(tmp_path)
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=controller)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _start_login(pilot, app, "anthropic")
        # THE PRECONDITION, ASSERTED BEFORE THE CANCEL. A fast "cancelled" only
        # means something if a login was genuinely pending and genuinely held
        # the port: a setup that quietly did not take looks exactly like a
        # feature that works. Both facts are checked — the flow published its
        # signal, and the OS says the socket is taken.
        assert await _settle(lambda: app._login_signal is not None), "no login was ever pending"
        assert await _settle(lambda: _port_held(port)), "the listener never came up"

        await pilot.press("ctrl+c")
        assert await _settle(lambda: not _port_held(port)), "the port was never released"

        # The retry the message invites, for real.
        await _start_login(pilot, app, "anthropic")
        assert await _settle(lambda: _port_held(port)), "the retry could not bind the port"
        assert not any("already in progress" in n for n in _notices(app)), _notices(app)
        assert not any("already in use" in n for n in _notices(app)), _notices(app)

        # Leave nothing listening behind for the next test.
        await pilot.press("ctrl+c")
        await _settle(lambda: not _port_held(port))


# -- the ladder regression guard --------------------------------------------


async def test_cancelling_a_login_does_not_arm_the_exit_ladder(
    tmp_path: Path, no_browser: None
) -> None:
    """THE hazard of adding this rung.

    A user whose login is stuck reaches for Ctrl+C and, when the first press
    appears to do nothing, presses it again — that is the reflex the whole
    feature is responding to. If the cancelling press also armed the double-tap
    exit ladder, the second press would QUIT THE APP.

    Measured on the pre-change tree: one Ctrl+C during a pending login rendered
    "ctrl+c again to exit" while the login stayed pending, so the trap was
    already loaded before this rung existed.

    The ladder is ARMED FIRST, deliberately. A press on an idle app leaves
    `_last_interrupt_at` at 0.0 anyway, so a version of this test that started
    from rest would pass against a rung that never disarms anything —
    confirmed by mutation, where dropping the disarm lines left the whole file
    green. Arming it first is what makes the disarm the thing under test, and
    it is also the real sequence: the user presses Ctrl+C once (arming the
    ladder and showing "ctrl+c again to exit"), THEN starts a login, then
    presses again to kill it.
    """
    controller = _controller(tmp_path)
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=controller)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)

        # Arm the exit ladder, exactly as an ordinary interrupt does.
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert app._last_interrupt_at != 0.0, "precondition: the ladder is armed"
        armed_at = app._last_interrupt_at

        await _start_login(pilot, app, "alibaba")
        assert await _settle(lambda: app._login_signal is not None)

        await pilot.press("ctrl+c")
        assert await _settle(lambda: any("login cancelled" in n for n in _notices(app)))

        assert app._last_interrupt_at == 0.0, (
            "the cancelling press left the exit ladder armed: the reflex second "
            f"press would quit the app (was {armed_at}, still {app._last_interrupt_at})"
        )
        assert app._exit_hint is None, "a stale 'ctrl+c again to exit' hint is still on screen"

        # The reflex second press, for real: the app must still be running.
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert app.is_running, "a double-tap to kill a stuck login quit the app"


async def test_with_no_login_pending_the_ladder_is_unchanged(tmp_path: Path) -> None:
    """The other half of the guard: the rung must not claim a press it has no
    business claiming.

    With no login in flight, `_cancel_pending_login` returns False and the
    press falls through to exactly the rungs it reached before — here, the
    interrupt rung, which arms the double-tap exit ladder.
    """
    controller = _controller(tmp_path)
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=controller)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        assert app._login_signal is None

        app._last_interrupt_at = 0.0
        await pilot.press("ctrl+c")
        await pilot.pause()

        assert app._last_interrupt_at != 0.0, "the exit ladder should arm as it always did"


async def test_the_draft_rung_still_wins_when_no_login_is_pending(tmp_path: Path) -> None:
    """The rung sits AHEAD of the draft rung, so this pins that the draft rung
    is still reachable — a new rung that shadowed it would silently cost users
    the composer text this repo has repeatedly paid to protect."""
    controller = _controller(tmp_path)
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=controller)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        editor = app.query_one(Editor)
        editor.text = "a sentence the user is still writing"
        await pilot.pause()

        await pilot.press("ctrl+c")
        await pilot.pause()

        assert editor.text.strip() == "", "the draft rung no longer clears the composer"
        assert any("draft cleared" in note for note in _notices(app)), _notices(app)


async def test_a_pending_login_takes_the_press_ahead_of_the_draft(
    tmp_path: Path, no_browser: None
) -> None:
    """The ordering decision, pinned.

    A login is pending AND the composer holds a draft. The login wins, and the
    draft SURVIVES — which is the asymmetry the ordering rests on: the draft is
    still there to clear with the next press, whereas a login cancelled late is
    a login that was never cancelled at all.
    """
    controller = _controller(tmp_path)
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=controller)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _start_login(pilot, app, "alibaba")
        assert await _settle(lambda: app._login_signal is not None)

        editor = app.query_one(Editor)
        editor.text = "a draft the user does not want to lose"
        await pilot.pause()

        await pilot.press("ctrl+c")
        assert await _settle(lambda: any("login cancelled" in n for n in _notices(app)))

        assert editor.text == "a draft the user does not want to lose", "the draft was destroyed"
        assert not any("draft cleared" in note for note in _notices(app)), _notices(app)


async def test_a_second_press_after_the_cancel_reaches_the_draft_rung(
    tmp_path: Path, no_browser: None
) -> None:
    """The ladder stays MONOTONIC: the rung claims exactly one press.

    A rung that could be reached twice in a row is how the exit ladder became
    unreachable in an earlier round, so the press after a completed cancel must
    fall through to the rung it would have reached anyway.
    """
    controller = _controller(tmp_path)
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=controller)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _start_login(pilot, app, "alibaba")
        assert await _settle(lambda: app._login_signal is not None)
        editor = app.query_one(Editor)
        editor.text = "still here"
        await pilot.pause()

        await pilot.press("ctrl+c")  # takes the login
        assert await _settle(lambda: any("login cancelled" in n for n in _notices(app)))
        await pilot.press("ctrl+c")  # must now reach the draft rung
        await pilot.pause()

        assert editor.text.strip() == "", "the second press did not reach the draft rung"


async def test_the_key_prompt_does_not_swallow_the_press(tmp_path: Path, no_browser: None) -> None:
    """The specific trap named in the brief.

    A paste-code provider mounts a `KeyPromptBlock` and FOCUSES it, and that
    block binds escape and consumes printable keys. If it also swallowed
    Ctrl+C, the press would never reach the ladder and the login would stay
    pending — the exact bug being fixed, reintroduced one layer down.

    Ctrl+C is not printable and the block does not bind it, so it bubbles to
    the app; this pins that rather than leaving it to the block's key handling
    staying the way it is today.
    """
    controller = _controller(tmp_path)
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=controller)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _start_login(pilot, app, "alibaba")
        assert await _settle(lambda: bool(app.query(KeyPromptBlock)))
        prompt = next(iter(app.query(KeyPromptBlock)))
        assert prompt.has_focus, "precondition: the prompt owns the keyboard"

        await pilot.press("ctrl+c")

        assert await _settle(lambda: any("login cancelled" in n for n in _notices(app)))
        assert prompt.answered, "the prompt must settle with the login it belonged to"


async def test_escape_still_cancels_from_the_prompt(tmp_path: Path, no_browser: None) -> None:
    """Escape keeps its existing meaning on the block that binds it.

    The two keys now both end the login, by different routes — escape through
    the prompt's own `action_cancel`, Ctrl+C through the app's ladder — and
    this pins that adding the second did not disturb the first.
    """
    controller = _controller(tmp_path)
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=controller)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _start_login(pilot, app, "alibaba")
        assert await _settle(lambda: bool(app.query(KeyPromptBlock)))

        await pilot.press("escape")

        assert await _settle(lambda: any("cancelled" in n for n in _notices(app))), _notices(app)
        assert not any("failed" in note for note in _notices(app)), _notices(app)


# -- loopback-only providers: the state every other test here cannot reach ----
#
# Every ordering test above uses a provider that mounts a `KeyPromptBlock`, and
# that block closes the aside and the floating views on mount. So none of them
# can observe what a LOOPBACK-ONLY login does — no prompt is ever mounted for
# one — and both majors of agent review round 1 lived in exactly that gap.


async def test_a_loopback_login_cancels_with_no_prompt_mounted(
    tmp_path: Path, no_browser: None
) -> None:
    """The baseline for the two tests below: openai mounts NO prompt.

    Asserted rather than assumed, because it is the precondition that makes
    them meaningful — if a prompt did mount, they would be re-testing the
    paste-provider path under a different name.
    """
    controller = _controller(tmp_path)
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=controller)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _start_login(pilot, app, "openai")
        assert await _settle(lambda: app._login_signal is not None), "no login was pending"
        assert not list(app.query(KeyPromptBlock)), "openai must not mount a paste prompt"

        await pilot.press("ctrl+c")

        assert await _settle(lambda: any("login cancelled" in n for n in _notices(app))), _notices(
            app
        )
        assert app._login_signal is None


async def test_cancelling_a_loopback_login_returns_to_the_conversation(
    tmp_path: Path, no_browser: None
) -> None:
    """Agent review round 1, major-2: the receipt must not be drawn out of sight.

    The aside floats over the transcript at one elevation step, so a notice
    appended behind it is drawn where it cannot be read — the rule the exit
    ladder's own tail states and honours. The cancel appends exactly such a
    notice, and on a loopback-only provider nothing else closes the card, so
    the user pressed ctrl+C, the login really cancelled, and the only evidence
    landed behind the aside: "nothing appeared to happen".

    The CONTROL is the point of the test: an ordinary ctrl+C closes the aside,
    so a login-cancelling one that did not would be a behavioural split with no
    reason a user could infer.
    """
    controller = _controller(tmp_path)
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=controller)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _start_login(pilot, app, "openai")
        assert await _settle(lambda: app._login_signal is not None), "no login was pending"
        assert not list(app.query(KeyPromptBlock)), "precondition: no prompt closes the aside"

        app._open_aside()
        await pilot.pause()
        assert app._aside_is_open(), "precondition: the aside is open over the transcript"

        await pilot.press("ctrl+c")
        assert await _settle(lambda: any("login cancelled" in n for n in _notices(app)))

        assert not app._aside_is_open(), (
            "the cancel receipt was appended behind the floating aside, where the "
            "user cannot read it"
        )

        # Control: the ordinary interrupt rung closes it too, so the two presses
        # agree.
        app._open_aside()
        await pilot.pause()
        assert app._aside_is_open() and app._login_signal is None
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert not app._aside_is_open(), "control: ordinary ctrl+C should close the aside"


async def test_starting_a_login_retires_an_armed_exit_hint(
    tmp_path: Path, no_browser: None
) -> None:
    """UX round 1, U3: the screen must not say ctrl+C exits while a login is up.

    Interrupt something first and the transcript carries "ctrl+c again to
    exit". Start a login and that line is now FALSE — the next press cancels
    the login and the app keeps running. It is false in the direction that
    costs the user the feature: they read "again to exit", believe the rescue
    key will quit their session, and sit out the 300 s wait instead.
    """
    controller = _controller(tmp_path)
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=controller)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)

        await pilot.press("ctrl+c")
        await pilot.pause()
        assert app._last_interrupt_at != 0.0, "precondition: the exit ladder is armed"
        assert app._exit_hint is not None, "precondition: the hint is on screen"
        assert any("again to exit" in note for note in _notices(app)), _notices(app)

        await _start_login(pilot, app, "openai")
        assert await _settle(lambda: app._login_signal is not None)

        assert app._exit_hint is None, "the stale exit hint survived into the pending login"
        assert app._last_interrupt_at == 0.0
        assert not any(
            "again to exit" in note for note in _notices(app)
        ), "the screen still tells the user ctrl+c exits, which is now false"

        # And the press it was contradicting does the right thing.
        await pilot.press("ctrl+c")
        assert await _settle(lambda: any("login cancelled" in n for n in _notices(app)))
        assert app.is_running, "the login-cancelling press must not quit the app"


async def test_the_browser_wait_names_the_key_that_cancels_it(
    tmp_path: Path, no_browser: None
) -> None:
    """UX round 1, U4: a rescue key nobody knows about rescues nobody.

    For a loopback-only login this block is the ONLY surface the pending state
    puts on screen, so without this line the frame advertises no way out at
    all — and the user it is for is watching a browser that landed somewhere
    unexpected.
    """
    controller = _controller(tmp_path)
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=controller)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _start_login(pilot, app, "openai")
        assert await _settle(lambda: app._login_signal is not None)

        from tests.unit.tui.test_app_pilot import _transcript_text

        rendered = _transcript_text(app)
        assert "ctrl+c" in rendered, f"the pending frame names no escape key:\n{rendered}"

        await pilot.press("ctrl+c")
        await _settle(lambda: app._login_signal is None)
