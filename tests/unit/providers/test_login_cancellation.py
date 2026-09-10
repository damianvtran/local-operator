"""Cancelling a PENDING login, end to end through the layers that carry it.

The reported problem: a `/login` opens the browser, the browser lands somewhere
other than the loopback redirect (Anthropic's own portal, an expired consent
page, a tab the user closes), and the flow stays parked. The only way out was
`DEFAULT_TIMEOUT_SECONDS`, which is 300 s.

Every mechanism the cancel needs already existed and none of it was reachable:
`OAuthCallbackFlow._await_code` races an abort watcher against its capture
futures, the device flow polls the signal, and the registry's lazy thunks
forward it — but `ProviderController.login` never constructed a signal, so no
caller could ever abort a flow. These tests cover the seam that was missing and
the properties a cancel has to leave behind: the listener stopped, the port
free, and a retry that works immediately.

The port assertions bind a real socket rather than reading a flag, because
"the login said it stopped" and "the OS will let the next login have the port"
are different claims and only the second one is the user's problem.
"""

from __future__ import annotations

import asyncio
import socket
import time
from typing import Any

import pytest

from local_operator.harness.types import AbortSignal
from local_operator.providers.oauth.callback_server import (
    DEFAULT_TIMEOUT_SECONDS,
    CallbackFlowOptions,
    LoginCallbacks,
    LoginCancelledError,
    OAuthCallbackFlow,
)

pytestmark = pytest.mark.asyncio


class _Flow(OAuthCallbackFlow):
    """A real callback flow whose only fake is the token exchange."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.exchanged = False

    async def generate_auth_url(self, state: str, redirect_uri: str) -> str:
        return f"https://provider.invalid/authorize?state={state}&redirect_uri={redirect_uri}"

    async def exchange_token(self, code: str, state: str, redirect_uri: str) -> dict[str, Any]:
        self.exchanged = True
        return {"access": "token"}


def _free_port() -> int:
    """A port that is free RIGHT NOW, so the test never fights the real 54545.

    The suite runs under xdist with sibling worktrees on the same machine; a
    hardcoded port here would make two unrelated runs collide and report a
    product bug.
    """
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = int(probe.getsockname()[1])
    probe.close()
    return port


def _port_is_free(port: int) -> bool:
    probe = socket.socket()
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        probe.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        probe.close()


async def _await_bound(flow: OAuthCallbackFlow, limit: int = 250) -> None:
    """Wait for the listener to come up, bounded in TURNS not seconds."""
    for _ in range(limit):
        await asyncio.sleep(0)
        if flow.bound_port is not None:
            return
    raise AssertionError("the callback server never bound")


async def test_aborting_a_pending_flow_raises_at_once_instead_of_waiting_out_the_timeout() -> None:
    """The whole point of the change, as a number.

    The bound is deliberately enormous relative to the work (the abort is an
    event set on an already-parked waiter) and tiny relative to what it
    replaces. It is a BACKSTOP against a regression that reintroduces a wait,
    not a calibrated performance assertion — anything in the same order of
    magnitude as 300 s fails it, and ordinary CI contention cannot.
    """
    port = _free_port()
    signal = AbortSignal()
    flow = _Flow(
        options=CallbackFlowOptions(preferred_port=port, allow_port_fallback=False),
        open_browser=lambda url: None,
        signal=signal,
    )
    task = asyncio.create_task(flow.run())
    await _await_bound(flow)

    started = time.monotonic()
    signal.abort("Login cancelled")
    with pytest.raises(LoginCancelledError, match="Login cancelled"):
        await task
    elapsed = time.monotonic() - started

    assert elapsed < 5.0, f"cancel took {elapsed:.1f}s"
    assert elapsed < DEFAULT_TIMEOUT_SECONDS / 10
    assert not flow.exchanged, "a cancelled login must not exchange a token"


async def test_a_cancelled_login_releases_the_port_for_an_immediate_retry() -> None:
    """The property the user actually experiences after a cancel.

    A cancel that raised but left the listener up would be worse than the
    timeout it replaces: the retry the message invites would then fail with
    "port is required for this login flow but is already in use", and the
    provider pins these ports (54545 for Anthropic, 1455 for OpenAI) so there
    is no fallback to hide it.
    """
    port = _free_port()
    signal = AbortSignal()
    flow = _Flow(
        options=CallbackFlowOptions(preferred_port=port, allow_port_fallback=False),
        open_browser=lambda url: None,
        signal=signal,
    )
    task = asyncio.create_task(flow.run())
    await _await_bound(flow)
    assert not _port_is_free(port), "precondition: the listener holds the port"

    signal.abort("Login cancelled")
    with pytest.raises(LoginCancelledError):
        await task

    assert _port_is_free(port), "the loopback listener outlived the cancelled login"

    # The retry, for real: a second flow on the SAME pinned port, which is what
    # the cancel message tells the user to do.
    retry = _Flow(
        options=CallbackFlowOptions(preferred_port=port, allow_port_fallback=False),
        open_browser=lambda url: None,
        signal=AbortSignal(),
    )
    retry_task = asyncio.create_task(retry.run())
    await _await_bound(retry)
    assert retry.bound_port == port
    retry_task.cancel()
    await asyncio.gather(retry_task, return_exceptions=True)


async def test_an_abort_beats_a_paste_prompt_that_never_answers() -> None:
    """Anthropic's shape: a paste prompt races the loopback callback.

    The prompt parks forever when the user declines, which is exactly the state
    the reported bug leaves the app in — so the abort has to win against a
    waiter that is never going to resolve.
    """
    port = _free_port()
    signal = AbortSignal()
    parked = asyncio.Event()

    async def never_answers() -> str | None:
        parked.set()
        await asyncio.Future()  # the user is looking at a dead browser tab
        return None

    flow = _Flow(
        options=CallbackFlowOptions(preferred_port=port, allow_port_fallback=False),
        callbacks=LoginCallbacks(on_manual_code_input=never_answers),
        open_browser=lambda url: None,
        signal=signal,
    )
    task = asyncio.create_task(flow.run())
    await asyncio.wait_for(parked.wait(), timeout=10)

    signal.abort("Login cancelled")
    with pytest.raises(LoginCancelledError):
        await task
    assert _port_is_free(port)


async def test_a_signal_aborted_before_the_flow_starts_still_cancels() -> None:
    """The race a user creates by pressing ctrl+C while the browser is opening.

    `_await_code` builds its abort watcher from an already-set event here, so
    the flow must not sail past it and park for the full timeout.
    """
    port = _free_port()
    signal = AbortSignal()
    signal.abort("Login cancelled")
    flow = _Flow(
        options=CallbackFlowOptions(preferred_port=port, allow_port_fallback=False),
        open_browser=lambda url: None,
        signal=signal,
    )
    with pytest.raises(LoginCancelledError):
        await asyncio.wait_for(flow.run(), timeout=10)
    assert _port_is_free(port)


async def test_no_signal_means_the_flow_behaves_exactly_as_before() -> None:
    """The regression guard for every caller that passes nothing.

    `_await_code` only appends the abort watcher when a signal is present, so
    the no-signal path must keep waiting on its capture futures rather than
    acquiring a new way to end early.
    """
    port = _free_port()
    flow = _Flow(
        options=CallbackFlowOptions(
            preferred_port=port, allow_port_fallback=False, timeout_seconds=0.2
        ),
        open_browser=lambda url: None,
    )
    from local_operator.providers.oauth.callback_server import LoginTimeoutError

    with pytest.raises(LoginTimeoutError):
        await flow.run()
    assert _port_is_free(port), "the timeout path still stops the server"


# -- the seam that was missing: ProviderController.login ---------------------


def _controller(tmp_path: Any) -> Any:
    from local_operator.providers.auth_store import AuthStore
    from local_operator.providers.controller import ProviderController

    return ProviderController(AuthStore(tmp_path / "auth.db"))


async def test_the_controller_forwards_its_signal_to_the_provider(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The defect this change fixes, stated directly.

    Before it, `ProviderController.login` called `definition.login(callbacks)`
    with no signal, so every abort mechanism beneath it was unreachable code.
    Driven through the REAL controller over a real store, with only the
    provider's own login coroutine swapped — the seam under test is between
    the controller and the registry, and a stub standing in for the controller
    would be a test of the stub.
    """
    import dataclasses

    import local_operator.providers.controller as controller_module
    from local_operator.providers.registry import get_provider_definition

    definition = get_provider_definition("anthropic")
    assert definition is not None
    seen: dict[str, Any] = {}

    async def fake_login(callbacks: Any, **kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return {"access": "t", "refresh": "r"}

    patched = dataclasses.replace(definition, login=fake_login)
    monkeypatch.setattr(
        controller_module,
        "get_provider_definition",
        lambda pid: patched if pid == "anthropic" else get_provider_definition(pid),
    )

    controller = _controller(tmp_path)
    signal = AbortSignal()
    await controller.login("anthropic", signal=signal)

    assert seen.get("signal") is signal, f"the signal never reached the provider: {seen}"


async def test_the_controller_still_works_when_no_signal_is_passed(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`signal` is optional, and every existing caller omits it.

    The keyword is forwarded even when it is None (one call shape for all
    providers), so this pins that a None reaches the flow harmlessly rather
    than the parameter being dropped for the default case.
    """
    import dataclasses

    import local_operator.providers.controller as controller_module
    from local_operator.providers.registry import get_provider_definition

    definition = get_provider_definition("anthropic")
    assert definition is not None
    seen: dict[str, Any] = {}

    async def fake_login(callbacks: Any, **kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        seen["called"] = True
        return {"access": "t", "refresh": "r"}

    patched = dataclasses.replace(definition, login=fake_login)
    monkeypatch.setattr(
        controller_module,
        "get_provider_definition",
        lambda pid: patched if pid == "anthropic" else get_provider_definition(pid),
    )

    controller = _controller(tmp_path)
    await controller.login("anthropic")

    assert seen.get("called") is True
    assert seen.get("signal") is None


async def test_the_controller_passes_a_signal_to_every_login_in_the_registry() -> None:
    """Enumerated over the shipped registry, not checked for one provider.

    The controller passes `signal=` UNCONDITIONALLY, so a login callable that
    did not accept the keyword would raise `TypeError` for that provider only —
    the kind of break that ships because the test named the two OAuth providers
    someone happened to think of. The paste-a-key logins take it through
    `**_kwargs` and ignore it; that is fine and is exactly what this asserts.
    """
    import inspect

    from local_operator.providers.registry import PROVIDER_REGISTRY

    rejects = []
    for provider in PROVIDER_REGISTRY:
        if provider.login is None:
            continue
        parameters = inspect.signature(provider.login).parameters
        accepts = "signal" in parameters or any(
            value.kind is inspect.Parameter.VAR_KEYWORD for value in parameters.values()
        )
        if not accepts:
            rejects.append(provider.id)
    assert rejects == [], f"these logins would raise TypeError on signal=: {rejects}"
