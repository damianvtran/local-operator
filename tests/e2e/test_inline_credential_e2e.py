"""Inline ``/credential`` over the session class the shipping product uses.

**This file exists because a green four-stream review shipped a half-feature.**
The inline-credential submit seam was built primarily against
:class:`~local_operator.session.session.Session` — and `lop`'s TUI never builds
one. ``cli.py``'s ``viewer_factory`` returns a ``AttachedSession`` on every path
and its takeover closure raises outright, so the branch the tests certified was
unreachable in the product and the branch that actually ran was written as the
"degradation". The operator hit it on his first use: he typed a secret and was
told no store was available, with advice to paste ``/credential <KEY>``
instead.

So the rule here is the one ``test_viewer_attach_e2e`` states and for the same
reason: **the production handle class, the production server, a real loopback
socket, the production ``AttachedSession`` client, and the real ``OperatorApp``
driven by keystrokes.** A test that constructs a ``Session`` and hands it to
the app proves nothing about `lop` — that substitution IS the defect this file
guards. ``test_the_app_under_test_holds_the_class_the_product_builds`` asserts
the harness has not quietly drifted back to the reachable-looking-but-wrong
shape.

The value asserted here is synthetic and its bytes are never printed: length is
proof enough, and a canary that reached a log would be a leak of the same shape
the feature exists to prevent.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path
from typing import Any, cast

import pytest
from textual import events

from local_operator.session.runtime import registry
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.runtime.serving import ServingSessionHandle
from tests.e2e.harness import ScriptedStream, build_session, text_turn

pytestmark = pytest.mark.e2e

#: A synthetic 52-character secret. Never printed, never asserted by value into
#: a message a human reads — only its LENGTH crosses into an assertion, which
#: is the same discipline ``test_viewer_attach_e2e`` follows.
CANARY = "lopcanary" + "Z" * 39 + "END"


async def _never_take_over() -> Any:
    raise AssertionError("a viewer never takes over a session")


async def _wait_for_record(config_dir: Path, session_id: str, timeout: float = 15.0) -> Any:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        for record, _state in registry.scan(config_dir):
            if getattr(record, "session_id", "") == session_id:
                return record
        await asyncio.sleep(0.05)
    raise AssertionError(f"no record published for {session_id} within {timeout}s")


async def _pump(pilot: Any, predicate: Any, tries: int = 400) -> bool:
    for _ in range(tries):
        await pilot.pause()
        if predicate():
            return True
    return False


class _Runtime:
    """A real ``Session`` behind the production handle behind the real server."""

    def __init__(self, config_dir: Path) -> None:
        self.config_dir = config_dir
        # The id is DERIVED from the directory name, exactly as a real
        # runtime's is — `Session.session_id` is read-only for that reason.
        self.session_id = uuid.uuid4().hex[:12]
        self.directory = config_dir / "sessions" / self.session_id
        self.directory.mkdir(parents=True, exist_ok=True)
        self.session: Any = None
        self.server: RuntimeServer | None = None
        #: The scripted provider itself, held directly: the recorded requests
        #: are the evidence for "the model was told the NAME", and reaching
        #: them through the session's private provider plumbing would couple
        #: the test to it.
        self.stream = ScriptedStream([text_turn("ack"), text_turn("ack")])

    async def start(self) -> None:
        stream = self.stream
        self.session = build_session(self.directory, stream)
        assert self.session.session_id == self.session_id, (
            "the session must publish under the directory's id, which is what " "the app dials"
        )
        handle = ServingSessionHandle(
            self.session, asyncio.get_running_loop(), cwd=str(self.directory)
        )
        self.server = RuntimeServer(handle, kind="daemon")
        await self.server.start_in_process()
        (self.directory / ".session.pid").write_text(str(os.getpid()), encoding="utf-8")
        await _wait_for_record(self.config_dir, self.session_id)

    async def stop(self) -> None:
        if self.server is not None:
            self.server.close()
        if self.session is not None:
            await self.session.dispose()


async def _viewer_app(runtime: _Runtime) -> Any:
    """The real ``OperatorApp`` over a real attached ``AttachedSession``.

    ``cli.viewer_factory`` in miniature and deliberately no smaller: a live
    record exists, so the production ``AttachedSession.connect`` is what the app
    adopts — the exact object `lop` gives it.
    """
    from local_operator.session.attached import AttachedSession
    from local_operator.tui.app import OperatorApp

    record = await _wait_for_record(runtime.config_dir, runtime.session_id)

    async def factory() -> AttachedSession:
        return await AttachedSession.connect(
            record,
            runtime.session_id,
            config_dir=runtime.config_dir,
            takeover_factory=_never_take_over,
        )

    OperatorApp._check_for_update = lambda self: None  # type: ignore[method-assign]
    return OperatorApp(factory)


async def _arm_and_paste(pilot: Any, app: Any, prefix: str, secret: str = CANARY) -> None:
    """The reported gesture: type ``/credential ``, paste, describe, Enter.

    Keystrokes rather than a direct call into the handler, because the whole
    defect lived between the composer and the submit seam.
    """
    from local_operator.tui.widgets.editor import Editor

    editor = app.query_one(Editor)
    editor.focus()
    await pilot.pause()
    for char in f"{prefix}/credential ":
        await pilot.press(char)
    await pilot.pause()
    app.post_message(events.Paste(secret))
    await pilot.pause()
    await pilot.pause()


def _credential_keys(app: Any) -> list[str]:
    from local_operator.tui.widgets.editor import Editor, PastedCredential

    editor = app.query_one(Editor)
    return [
        payload.key
        for payload in editor._attachments.values()
        if isinstance(payload, PastedCredential)
    ]


@pytest.mark.asyncio
async def test_the_app_under_test_holds_the_class_the_product_builds(
    headless_tui_env: Path, workspace: Path
) -> None:
    """The harness canary: assert the SHAPE before trusting any result from it.

    Every other test in this file is only evidence about `lop` if the app it
    drives holds what `lop`'s factory hands it. This asserts that directly —
    and it is not ceremony: the substitution it forbids (an in-process
    ``Session`` injected into ``OperatorApp``) is exactly what let the inline
    ``/credential`` half-feature pass four review streams.
    """
    from local_operator.session.attached import AttachedSession
    from local_operator.session.session import Session

    runtime = _Runtime(headless_tui_env)
    await runtime.start()
    app = await _viewer_app(runtime)
    try:
        async with app.run_test(size=(120, 40)) as pilot:
            assert await _pump(pilot, lambda: app._session is not None), "the app never adopted"
            assert isinstance(app._session, AttachedSession), (
                "this harness must drive the class cli.viewer_factory returns; "
                f"it holds {type(app._session).__name__}"
            )
            assert not isinstance(app._session, Session)
            # The store the value must reach is the RUNTIME's, and the viewer
            # must not have grown one of its own — a local store would satisfy
            # a naive round-trip and leave every bash command unable to read it.
            assert getattr(app._session, "variables", None) is None
    finally:
        await runtime.stop()


@pytest.mark.asyncio
async def test_a_typed_credential_reaches_the_runtime_store_and_the_agents_tools(
    headless_tui_env: Path, workspace: Path
) -> None:
    """RED before this change, GREEN after: the gesture on the REACHABLE path.

    The four assertions are the feature, in order: the value lands in the
    store that ``credential_env()`` reads, a real child process can use it,
    the model is told the NAME (never the bytes), and the operator's own words
    survive around the citation.
    """
    from local_operator.tools.builtin import execute_bash

    runtime = _Runtime(headless_tui_env)
    await runtime.start()
    app = await _viewer_app(runtime)
    try:
        async with app.run_test(size=(120, 40)) as pilot:
            assert await _pump(pilot, lambda: app._session is not None)
            await _arm_and_paste(pilot, app, "deploy with ")
            keys = _credential_keys(app)
            assert len(keys) == 1, f"the chip must be minted before submit: {keys}"
            key = keys[0]
            for char in " for the staging deploy":
                await pilot.press(char)
            await pilot.press("enter")

            store = runtime.session.variables
            landed = await _pump(pilot, lambda: key in store.credential_names())
            assert landed, (
                "the typed credential never reached the runtime's store; "
                f"it holds {store.credential_names()}"
            )
            # ASSERT THE MUTATION LANDED, by the value's LENGTH rather than by
            # its bytes: a store that held a truncated or empty value would
            # otherwise be indistinguishable from a correct one.
            assert len(store.credential_env()[key]) == len(CANARY)

            # USABLE, not merely present: the whole point of the store is the
            # environment of the commands the agent runs.
            class _Ctx:
                variables = store
                cwd = str(runtime.directory)

            result = await execute_bash(
                "e2e-inline-cred",
                {"command": f'test -n "${key}" && echo LEN=${{#{key}}}'},
                None,
                None,
                cast("Any", _Ctx()),
            )
            rendered = str(getattr(result, "content", result))
            assert f"LEN={len(CANARY)}" in rendered, rendered
            assert CANARY not in rendered, "the value must not appear in tool output"

            # The model is handed the NAME and the operator's description.
            sent = " ".join(
                str(getattr(request, "messages", "")) for request in runtime.stream.requests
            )
            assert CANARY not in sent
            assert key in sent, "the model must be told the key it can use"
            assert "NOT stored" not in sent, "the degradation must not fire on a live runtime"
            assert "for the staging deploy" in sent
    finally:
        await runtime.stop()


@pytest.mark.asyncio
async def test_the_secret_reaches_no_transcript_history_or_draft(
    headless_tui_env: Path, workspace: Path
) -> None:
    """Containment, over the same real submit — with the grep canaried.

    The value now lives across an await it did not cross before, so this is
    the invariant most at risk from the change and least visible when broken.

    **The grep is canaried in both directions**: a file known to contain the
    canary must be found, or an "absent everywhere" result would be
    indistinguishable from a scanner that reads nothing. Both the literal form
    and the normalised (``-``→``_``, upper-cased) form are searched, since a
    key-shaped rewrite of the value would evade a literal-only scan.
    """
    normalised = CANARY.replace("-", "_").upper()

    runtime = _Runtime(headless_tui_env)
    await runtime.start()
    app = await _viewer_app(runtime)
    try:
        async with app.run_test(size=(120, 40)) as pilot:
            assert await _pump(pilot, lambda: app._session is not None)
            await _arm_and_paste(pilot, app, "token ")
            key = _credential_keys(app)[0]
            for char in " is the deploy key":
                await pilot.press(char)
            await pilot.press("enter")
            store = runtime.session.variables
            assert await _pump(pilot, lambda: key in store.credential_names())
            await pilot.pause()

            # POSITIVE CONTROL for the scanner. Without it, "no hits anywhere"
            # is exactly what a scanner pointed at an empty tree reports.
            planted = headless_tui_env / "planted-canary.txt"
            planted.write_text(f"prefix {CANARY} suffix\n{normalised}\n", encoding="utf-8")

            scanned: list[Path] = []
            hits: list[str] = []
            for path in headless_tui_env.rglob("*"):
                if not path.is_file():
                    continue
                try:
                    blob = path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                scanned.append(path)
                if CANARY in blob or normalised in blob:
                    hits.append(str(path.relative_to(headless_tui_env)))

            assert "planted-canary.txt" in hits, (
                "the scanner never found the planted canary, so its silence "
                f"about every other file means nothing (scanned {len(scanned)})"
            )
            assert hits == ["planted-canary.txt"], f"the secret reached disk: {hits}"

            # In-memory holders the scanner cannot see.
            from local_operator.tui.widgets.editor import Editor

            editor = app.query_one(Editor)
            assert CANARY not in editor.text
            assert CANARY not in "\n".join(editor._history)
            painted = "\n".join(strip.text for strip in app.screen._compositor.render_strips())
            assert CANARY not in painted and normalised not in painted
    finally:
        await runtime.stop()


@pytest.mark.asyncio
async def test_a_lost_runtime_still_refuses_loudly_instead_of_promising_a_key(
    headless_tui_env: Path, workspace: Path
) -> None:
    """The refusal must SURVIVE the fix — otherwise only a check was removed.

    Round-tripping to the runtime replaces a wrong degradation, not the
    ability to degrade. With the runtime gone between the chip and Enter,
    ``credential_op`` answers ``disconnected`` and the model must be told the
    credential did NOT land — never handed a key nothing holds.
    """
    runtime = _Runtime(headless_tui_env)
    await runtime.start()
    app = await _viewer_app(runtime)
    try:
        async with app.run_test(size=(120, 40)) as pilot:
            assert await _pump(pilot, lambda: app._session is not None)
            await _arm_and_paste(pilot, app, "deploy with ")
            key = _credential_keys(app)[0]

            # The runtime goes away AFTER the chip is minted and BEFORE Enter:
            # the exact window the honest-failure invariant is about.
            await runtime.stop()
            await pilot.pause()

            await pilot.press("enter")
            await _pump(
                pilot,
                lambda: "NOT stored"
                in "\n".join(strip.text for strip in app.screen._compositor.render_strips()),
            )
            painted = "\n".join(strip.text for strip in app.screen._compositor.render_strips())
            assert CANARY not in painted
            assert "NOT stored" in painted, (
                "a failed round-trip must say so; a silent drop is the one "
                f"belief this feature must never create. screen: {painted[-800:]!r}"
            )
            # And the copy must not send the operator after a privileged
            # process that does not exist.
            for forbidden in ("owner", "viewer"):
                assert forbidden not in painted.lower(), (
                    f"the failure copy names {forbidden!r}, which is the model "
                    "that produced this defect"
                )
            assert key not in painted or "NOT stored" in painted
    finally:
        await runtime.stop()


@pytest.mark.asyncio
async def test_the_paste_route_still_stores_on_the_runtime(
    headless_tui_env: Path, workspace: Path
) -> None:
    """``/credential <KEY>`` pasted whole — the route that already worked.

    It now shares ``credential_op`` with the typed route, so it is exercised
    against the same live runtime to prove the shared verb did not regress the
    one path the operator has been relying on.
    """
    runtime = _Runtime(headless_tui_env)
    await runtime.start()
    app = await _viewer_app(runtime)
    try:
        async with app.run_test(size=(120, 40)) as pilot:
            assert await _pump(pilot, lambda: app._session is not None)
            from local_operator.tui.widgets.editor import Editor

            editor = app.query_one(Editor)
            editor.focus()
            await pilot.pause()
            app.post_message(events.Paste("/credential PASTED_TOKEN"))
            await pilot.pause()
            await pilot.press("enter")

            # The command opens a masked prompt card; the value is TYPED into
            # it, which is the paste route's own gesture (the card is the
            # masked-paste surface, not the composer).
            def _card_up() -> bool:
                try:
                    return bool(app.query("KeyPromptBlock"))
                except Exception:
                    return False

            assert await _pump(pilot, _card_up), "the store verb must open the masked paste"
            for char in CANARY:
                await pilot.press(char)
            await pilot.pause()
            await pilot.press("enter")

            store = runtime.session.variables
            landed = await _pump(pilot, lambda: "PASTED_TOKEN" in store.credential_names())
            assert landed, f"the paste route regressed: {store.credential_names()}"
            assert len(store.credential_env()["PASTED_TOKEN"]) == len(CANARY)
    finally:
        await runtime.stop()
