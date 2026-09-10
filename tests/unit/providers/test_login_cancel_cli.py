"""`local-operator login` must be escapable with Ctrl+C.

Two defects met on this path and the tests are split by which one they can see.

The IN-PROCESS tests drive `run_login` with a login that parks, and assert the
contract: an abort ends it, the message tells the user how to retry, and the
exit code is 130 (the shell's SIGINT convention, already used by
`credential_update_command`).

The SUBPROCESS test exists because the worse defect was invisible from inside
the process. A real Ctrl+C was delivered, `run_login`'s `except
KeyboardInterrupt` was reached and printed its line — and then the process sat
for FIVE MINUTES before exiting. `asyncio.run` shuts down the loop's default
executor by joining its workers with `THREAD_JOIN_TIMEOUT` (300 s), and the
paste prompt's worker was blocked in a terminal read that never returns. From
the user's side the cancel did not work at all. Nothing in-process can observe
that: the assertion is about whether the INTERPRETER exits, so the test has to
own a real child.

It runs under a pty because that is the only way to make Ctrl+C mean what it
means for a user: a process started in the background from a non-interactive
shell inherits SIGINT as SIG_IGN, and writing \\x03 to a pty master goes through
the line discipline exactly as a keypress does.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import socket
import sys
from typing import Any

import pytest

from local_operator.harness.types import AbortSignal
from local_operator.providers import auth_cli
from local_operator.providers.oauth.callback_server import LoginCallbacks
from local_operator.providers.registry import get_provider_definition


class _Store:
    """The AuthStore surface `run_login` touches, and nothing else."""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}

    def upsert_credential(self, provider: str, credential: dict[str, Any]) -> Any:
        self.rows[provider] = dict(credential)
        return dataclasses.make_dataclass("Row", ["provider"])(provider)


def _swap_login(monkeypatch: pytest.MonkeyPatch, provider_id: str, fn: Any) -> None:
    definition = get_provider_definition(provider_id)
    assert definition is not None
    monkeypatch.setattr(
        auth_cli,
        "get_provider_definition",
        lambda pid: (
            dataclasses.replace(definition, login=fn) if pid == provider_id else definition
        ),
    )


def test_a_cancelled_login_exits_130_and_says_how_to_retry(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The message has to answer "is it still running, and how do I try again?".

    A bare "cancelled" leaves the user of the reported scenario — a browser
    that never came back — unsure whether something is still listening on the
    port they were told about.
    """
    from local_operator.providers.oauth.callback_server import LoginCancelledError

    async def cancelling_login(callbacks: Any, **kwargs: Any) -> dict[str, Any]:
        raise LoginCancelledError("Login cancelled")

    monkeypatch.setattr(auth_cli, "_apply_login_defaults", lambda provider_id: None)
    _swap_login(monkeypatch, "anthropic", cancelling_login)

    code = auth_cli.run_login("anthropic", None, _Store())  # type: ignore[arg-type]

    out = capsys.readouterr().out
    assert code == 130, "130 is the shell's SIGINT convention"
    assert "Login cancelled" in out
    assert "local-operator login anthropic" in out, "the retry command must be named"
    assert "listener stopped" in out, "anthropic pins a callback port, so one was running"
    assert "Login failed" not in out, "a cancel is an outcome, not a failure"


def test_a_paste_only_provider_does_not_claim_a_listener_was_stopped(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The clause is conditional so it stays TRUE.

    `alibaba` reads a pasted key and never starts a callback server; telling
    the user a local listener was stopped would be a confident, wrong detail.
    """
    from local_operator.providers.oauth.callback_server import LoginCancelledError

    async def cancelling_login(callbacks: Any, **kwargs: Any) -> str:
        raise LoginCancelledError("Login cancelled")

    monkeypatch.setattr(auth_cli, "_apply_login_defaults", lambda provider_id: None)
    _swap_login(monkeypatch, "alibaba", cancelling_login)

    code = auth_cli.run_login("alibaba", None, _Store())  # type: ignore[arg-type]

    out = capsys.readouterr().out
    assert code == 130
    assert "local-operator login alibaba" in out
    assert "listener stopped" not in out


@pytest.mark.asyncio
async def test_an_abort_ends_a_login_that_ignores_the_signal() -> None:
    """The paste-a-key providers accept `signal` and ignore it.

    Their whole flow is one `on_manual_code_input` await, so an abort alone
    would leave the command blocked on a prompt read forever. `_login_or_cancel`
    races the flow against the signal for exactly this shape; verified under a
    pty before the race existed, where Ctrl+C printed nothing and the process
    had to be SIGKILLed.

    Driven at `_login_or_cancel` rather than through `run_login`, because the
    signal is delivered by a REAL SIGINT handler there — and a test that
    installs one fights pytest for the interpreter's signal disposition. The
    end-to-end claim is covered by the pty test below, which uses a real
    keypress instead of simulating one.
    """
    from local_operator.providers.oauth.callback_server import LoginCancelledError

    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def never_returns(callbacks: Any, **kwargs: Any) -> str:
        entered.set()
        try:
            await asyncio.Future()  # a prompt nobody answers
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "unreachable"  # pragma: no cover

    aborted = AbortSignal()
    task = asyncio.ensure_future(
        auth_cli._login_or_cancel(never_returns, LoginCallbacks(), aborted)
    )
    await asyncio.wait_for(entered.wait(), timeout=10)

    aborted.abort("Login cancelled")
    with pytest.raises(LoginCancelledError, match="Login cancelled"):
        await asyncio.wait_for(task, timeout=10)

    # The losing task is reaped rather than left running detached from the
    # command that started it.
    await asyncio.wait_for(cancelled.wait(), timeout=10)


@pytest.mark.asyncio
async def test_a_login_that_finishes_first_wins_the_race() -> None:
    """The race must not cost the ordinary path its result."""
    aborted = AbortSignal()

    async def prompt_login(callbacks: Any, **kwargs: Any) -> str:
        return "sk-real"

    result = await auth_cli._login_or_cancel(prompt_login, LoginCallbacks(), aborted)
    assert result == "sk-real"


def test_the_signal_reaches_the_provider_login(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI builds a signal and hands it down, like the TUI does."""
    seen: dict[str, Any] = {}

    async def recording_login(callbacks: Any, **kwargs: Any) -> str:
        seen.update(kwargs)
        return "sk-key"

    monkeypatch.setattr(auth_cli, "_apply_login_defaults", lambda provider_id: None)
    monkeypatch.setattr(auth_cli, "_invalidate_cached_listing", lambda pid: None)
    monkeypatch.setattr(auth_cli, "_invalidate_cached_usage", lambda pid, store: None)
    _swap_login(monkeypatch, "alibaba", recording_login)

    assert auth_cli.run_login("alibaba", None, _Store()) == 0  # type: ignore[arg-type]
    assert isinstance(seen.get("signal"), AbortSignal)


def test_a_successful_login_is_unaffected_by_the_cancel_plumbing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The regression guard: the ordinary path still stores and reports."""
    store = _Store()

    async def good_login(callbacks: Any, **kwargs: Any) -> str:
        return "sk-real"

    monkeypatch.setattr(auth_cli, "_apply_login_defaults", lambda provider_id: None)
    monkeypatch.setattr(auth_cli, "_invalidate_cached_listing", lambda pid: None)
    monkeypatch.setattr(auth_cli, "_invalidate_cached_usage", lambda pid, s: None)
    _swap_login(monkeypatch, "alibaba", good_login)

    assert auth_cli.run_login("alibaba", None, store) == 0  # type: ignore[arg-type]
    assert store.rows["alibaba"]["key"] == "sk-real"
    assert "Stored API key" in capsys.readouterr().out


# -- the hang only a real process can show ----------------------------------


_CHILD = """
import os, sys, tempfile
cfg = tempfile.mkdtemp(prefix="lo-cancel-test-")
os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = cfg
os.environ["HOME"] = cfg
sys.path.insert(0, {root!r})
import webbrowser
webbrowser.open = lambda *a, **k: True
from local_operator.providers.auth_cli import run_login
from local_operator.providers.auth_store import AuthStore
print("READY", flush=True)
code = run_login({provider!r}, None, AuthStore(os.path.join(cfg, "auth.db")))
print("EXIT_CODE=%d" % code, flush=True)
raise SystemExit(code)
"""


def _listener_is_up(port: int) -> bool:
    """True when something already holds ``port`` on loopback."""
    probe = socket.socket()
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        probe.bind(("127.0.0.1", port))
        return False
    except OSError:
        return True
    finally:
        probe.close()


@pytest.mark.skipif(sys.platform == "win32", reason="pty is POSIX-only")
@pytest.mark.parametrize("provider", ["anthropic", "alibaba"])
def test_ctrl_c_really_exits_the_command(provider: str, tmp_path: Any) -> None:
    """A REAL Ctrl+C on a REAL terminal ends the process, promptly.

    Both provider shapes, because they hang for different reasons: `anthropic`
    runs a loopback listener AND a paste prompt (the executor-join hang), while
    `alibaba` is a bare prompt read (the flow that ignores the signal).

    The 30 s bound is not a performance assertion — the measured cancel is
    30-60 ms. It is a backstop that fails loudly against the 300 s wait this
    replaces, chosen wide enough that a loaded CI runner cannot trip it.
    """
    import pty
    import select
    import signal as signal_module
    import time

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    script = _CHILD.format(root=root, provider=provider)

    env = {k: v for k, v in os.environ.items() if not k.startswith("CMUX_")}
    env["TERM"] = "xterm-256color"
    env.pop("NO_COLOR", None)

    pid, fd = pty.fork()
    if pid == 0:  # child
        # A process started from a non-interactive shell inherits SIGINT as
        # SIG_IGN and exec preserves it, so without this the test measures its
        # own harness rather than the product.
        signal_module.signal(signal_module.SIGINT, signal_module.SIG_DFL)
        os.execve(sys.executable, [sys.executable, "-c", script], env)
        os._exit(127)  # pragma: no cover

    output = bytearray()

    def drain(seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            ready, _, _ = select.select([fd], [], [], 0.1)
            if not ready:
                continue
            try:
                chunk = os.read(fd, 65536)
            except OSError:
                return
            if not chunk:
                return
            output.extend(chunk)

    try:
        # THE PRECONDITION, ASSERTED. "Cancelled in 40 ms" is worthless if the
        # login never started — a setup that silently did not take is
        # indistinguishable from a feature that works. So this waits for proof
        # the flow is genuinely PARKED (it has printed its prompt) and, for a
        # loopback provider, that the listener is really BOUND. Without both,
        # the cancel below could be measuring an empty process exiting.
        deadline = time.monotonic() + 60
        while b"empty to cancel" not in bytes(output) and time.monotonic() < deadline:
            drain(0.2)
        assert b"empty to cancel" in bytes(output), (
            "the login never reached its prompt, so there was nothing pending to "
            f"cancel:\n{bytes(output).decode(errors='replace')}"
        )
        if provider == "anthropic":
            assert _listener_is_up(54545), (
                "the loopback listener was never bound, so a fast 'cancel' would "
                "prove nothing about releasing it"
            )

        started = time.monotonic()
        os.write(fd, b"\x03")  # a real Ctrl+C keypress

        # Wait for the CHILD to report its own exit rather than for waitpid to
        # win a race. Under xdist the worker process has other machinery that
        # reaps children, so `waitpid` can return ECHILD even though the
        # command exited perfectly — observed on CI, where the captured output
        # contained "EXIT_CODE=130" while this loop concluded the process was
        # still running. The pty EOF and the child's own printed exit code are
        # facts about the product; `waitpid` succeeding is a fact about the
        # test runner.
        status: int | None = None
        while time.monotonic() - started < 30.0:
            drain(0.1)
            if b"EXIT_CODE=" in bytes(output):
                break
            try:
                done, waited = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                break  # already reaped by the runner; the output is the record
            if done:
                status = waited
                break
        elapsed = time.monotonic() - started
        text = bytes(output).decode(errors="replace")

        assert "EXIT_CODE=130" in text or (
            status is not None and os.WIFEXITED(status) and os.WEXITSTATUS(status) == 130
        ), (
            f"the command did not exit 130 within {elapsed:.0f}s of Ctrl+C — "
            f"it used to hang for 300s in the executor join. Output:\n{text}"
        )
        assert elapsed < 30.0, f"the cancel took {elapsed:.0f}s"
        assert "Login cancelled" in text
        assert f"local-operator login {provider}" in text
        if provider == "anthropic":
            # The listener the cancel was supposed to tear down is really gone,
            # which is what makes an immediate retry possible.
            released_by = time.monotonic() + 10
            while _listener_is_up(54545) and time.monotonic() < released_by:
                time.sleep(0.1)
            assert not _listener_is_up(54545), "the loopback listener outlived the cancel"
    finally:
        try:
            os.kill(pid, signal_module.SIGKILL)
            os.waitpid(pid, 0)
        except (ProcessLookupError, ChildProcessError):
            pass
        os.close(fd)


def test_the_terminal_guard_restores_attributes_it_snapshotted(monkeypatch) -> None:
    """Agent review round 1, major-1: a cancel must not leave ECHO disabled.

    The paste prompt reads through `getpass`, which turns ECHO off and restores
    it in its own `finally` — on the daemon reader thread, which is never
    joined (that is what fixes the 300 s executor join). So when a cancel exits
    the process with the reader still parked, the restore never runs and the
    user is returned to a shell where their typing is invisible.

    Driven at the guard rather than through a pty here: the pty form is real
    evidence but needs a controlling terminal, which the suite does not have.
    The under-test property is "whatever the terminal looked like on the way in
    is what it looks like on the way out", which is exactly what this asserts,
    including that it survives an exception.
    """
    calls: list[tuple[str, Any]] = []
    sentinel = ["saved-attrs"]

    class _Termios:
        TCSADRAIN = 2

        @staticmethod
        def tcgetattr(fd):
            calls.append(("get", fd))
            return sentinel

        @staticmethod
        def tcsetattr(fd, when, attrs):
            calls.append(("set", (fd, when, attrs)))

    class _Tty:
        def isatty(self):
            return True

        def fileno(self):
            return 7

    monkeypatch.setitem(sys.modules, "termios", _Termios)
    monkeypatch.setattr(sys, "stdin", _Tty())

    with auth_cli._terminal_state_restored():
        pass
    assert calls[0] == ("get", 7)
    assert calls[-1] == ("set", (7, _Termios.TCSADRAIN, sentinel)), calls

    # And on the path that matters: the cancel raises through the guard.
    calls.clear()
    with pytest.raises(RuntimeError):
        with auth_cli._terminal_state_restored():
            raise RuntimeError("cancelled")
    assert calls[-1] == ("set", (7, _Termios.TCSADRAIN, sentinel)), calls


def test_the_terminal_guard_is_inert_without_a_tty(monkeypatch) -> None:
    """Piped stdin and CI have no terminal to restore, and a guard that tried
    would raise on every non-interactive login."""

    class _Pipe:
        def isatty(self):
            return False

        def fileno(self):  # pragma: no cover - must not be reached
            raise AssertionError("fileno must not be consulted for a non-tty")

    monkeypatch.setattr(sys, "stdin", _Pipe())
    with auth_cli._terminal_state_restored():
        pass  # no exception is the assertion
