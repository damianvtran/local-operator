"""The `console` tool: its gate, its schema, its wire params, and its refusals.

Three things are worth pinning here rather than anywhere else:

* **The gate is one file-only predicate** over a record the app publishes, so the
  matrix (no record / record-but-console-off / record-and-console-on) is asserted
  through the REAL predicate with a real record on disk in the state tests, and
  here through the builder's own decision;
* **the results are the design's §15 copy**, typed and never substring-matched —
  including the two absences, which are answers rather than errors;
* **the secret path never puts a value where a model can read it**, which is a
  claim only a negative test can hold.

Nothing in this module touches the default state path: `_ui_console_liveness` and
`ui_console_advertisable` are patched per test, so a run on a machine with a real
app record behaves exactly like a run without one.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import pytest

from local_operator.browser_bridge.backend import BridgeError, BridgeUnreachable
from local_operator.browser_bridge.protocol import ErrorCode
from local_operator.harness.types import ToolContext
from local_operator.tools import builtin
from local_operator.ui_console import backend as console_backend
from local_operator.ui_console.state import ConsoleHostState, Liveness

PNG = b"\x89PNG\r\n\x1a\n" + b"payload"


def _record(**updates: Any) -> ConsoleHostState:
    values: dict[str, Any] = {
        "pid": 4242,
        "port": 52133,
        "session_key": "k" * 32,
        "proto": 1,
        "console": True,
    }
    values.update(updates)
    return ConsoleHostState.model_validate(values)


@pytest.fixture
def live_app(monkeypatch: pytest.MonkeyPatch) -> None:
    """A live, console-capable app, without touching the real discovery path."""
    monkeypatch.setattr(builtin, "ui_console_advertisable", lambda: True)
    monkeypatch.setattr(builtin, "_ui_console_liveness", lambda: (Liveness.FRESH, _record()))


class _FakeClient:
    """Stands in for the loopback client, recording what the tool sent."""

    def __init__(self, result: dict[str, Any] | None = None, error: Exception | None = None):
        self.result = result or {}
        self.error = error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((method, params))
        if self.error is not None:
            raise self.error
        return self.result


def _install(monkeypatch: pytest.MonkeyPatch, client: _FakeClient) -> _FakeClient:
    monkeypatch.setattr(console_backend, "ConsoleHostClient", lambda *a, **k: client)
    return client


def _context(session_id: str = "s-1", variables: Any = None) -> ToolContext:
    return ToolContext(cwd=".", session_id=session_id, variables=variables)


async def _call(  # type: ignore[no-untyped-def]
    args: dict[str, Any], context: ToolContext | None = None
):
    return await builtin.execute_console("c-1", args, None, None, context or _context())


# --- the gate ---------------------------------------------------------------


def test_the_gate_returns_no_tool_without_the_app(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(builtin, "ui_console_advertisable", lambda: False)
    assert builtin.build_console_tool(None) is None


def test_the_gate_offers_the_tool_when_the_app_publishes_a_console(
    live_app: None,
) -> None:
    tool = builtin.build_console_tool(None)
    assert tool is not None
    assert tool.name == "console"
    assert tool.label == "Console"
    # NOT hidden: a concealed tool can only be absent, and R7 needs an agent
    # asked to test a TUI to know the capability exists (design §14.2).
    assert tool.hidden is False


def test_the_description_does_the_five_jobs_the_design_asks_of_it(live_app: None) -> None:
    """The description is code, and §14.3 enumerates what it must carry.

    Each assertion below is one job: differentiate from `bash` on CONSEQUENCES,
    name the handle's host (the fact that stops the playwright-class mistake of
    going looking for another terminal), name `list`'s user-opened surfaces
    (R18's whole case), and route the per-method detail to the guide rather than
    paying for it in the schema.
    """
    tool = builtin.build_console_tool(None)
    assert tool is not None
    description = tool.description
    assert "`bash` returns output directly" in description
    assert "NOT for ordinary commands" in description
    assert "con:" in description
    assert "`list` shows surfaces the USER opened too" in description
    assert "guide://console" in description
    # And it does NOT overclaim the approval gate as the discouragement: §14.6
    # forbids counting on a tier that `auto` mode does not even install.
    assert "approval" not in description.lower()
    # The whole string stays in the short-by-policy range the budget guard
    # measures; the per-method detail lives in the parameter descriptions.
    assert len(description) < 900


def test_the_approval_tier_is_read_for_the_four_read_methods(live_app: None) -> None:
    """§11.1: `read` for list/status/read/screenshot, `exec` for everything else."""
    tool = builtin.build_console_tool(None)
    assert tool is not None
    assert tool.approval_tier == "exec"  # the highest op sets the static tier
    assert tool.call_approval_tier is not None
    for method in ("list", "status", "read", "screenshot"):
        assert tool.call_approval_tier({"method": method}) == "read", method
    for method in ("create", "input", "keys", "resize", "secure", "close"):
        assert tool.call_approval_tier({"method": method}) == "exec", method


def test_read_methods_and_writes_share_one_resource_rule(live_app: None) -> None:
    """Two surfaces may run in one batch; one surface may not race itself."""
    tool = builtin.build_console_tool(None)
    assert tool is not None
    assert tool.resource_keys is not None
    surface = tool.resource_keys({"method": "read", "surface": "con:1:ab"}, ".")
    assert surface == ("console:surface:con:1:ab",)
    # list/create name no surface, so they declare nothing and therefore do not
    # serialise a fleet of independent surfaces behind one barrier.
    assert tool.resource_keys({"method": "list"}, ".") == ()
    assert tool.resource_keys({"method": "create"}, ".") == ()


# --- argument validation ----------------------------------------------------


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ({"method": "nope"}, "unknown console method"),
        ({"method": "read"}, "needs the surface handle"),
        ({"method": "create"}, "OK"),
        ({"method": "read", "surface": "surface:7"}, "is not a Local Operator console surface"),
        (
            {"method": "read", "surface": "con:1:ab", "mode": "sideways"},
            "must be 'viewport' or 'scrollback'",
        ),
        (
            {"method": "read", "surface": "con:1:ab", "count": 5},
            "start/count only with mode='scrollback'",
        ),
        ({"method": "input", "surface": "con:1:ab"}, "needs 'text' or a 'secret_ref'"),
        (
            {"method": "input", "surface": "con:1:ab", "text": "x", "secret_ref": "S"},
            "'text' or 'secret_ref', not both",
        ),
        ({"method": "keys", "surface": "con:1:ab"}, "at least one named key"),
        ({"method": "resize", "surface": "con:1:ab", "cols": 80}, "needs both 'cols' and 'rows'"),
        ({"method": "secure", "surface": "con:1:ab"}, "needs 'on'"),
    ],
)
@pytest.mark.asyncio
async def test_argument_refusals_name_what_to_do(
    args: dict[str, Any], expected: str, live_app: None
) -> None:
    result = await _call(args)
    if expected == "OK":
        # `create` with nothing but a method is legal: the app starts the user's
        # own shell. (It fails later, on the absent client, which is a different
        # test — this one is about validation.)
        assert result.is_error is False or "unknown console method" not in result.text
        return
    assert result.is_error is True
    assert expected in result.text


@pytest.mark.asyncio
async def test_the_handle_refusal_says_another_terminal_is_not_this_one(live_app: None) -> None:
    """R18's failure mode: an agent asked about "the terminal" must be told which
    terminal this tool can read rather than guessing at another one's contents."""
    result = await _call({"method": "status", "surface": "iterm-7"})
    assert result.is_error is True
    assert "NOT readable by this tool" in result.text
    assert "con:" in result.text


@pytest.mark.asyncio
async def test_a_create_without_a_session_id_is_refused_rather_than_guessed(
    live_app: None,
) -> None:
    result = await _call({"method": "create"}, _context(session_id=""))
    assert result.is_error is True
    assert "no id" in result.text


@pytest.mark.asyncio
async def test_unknown_arguments_are_rejected_by_the_schema(live_app: None) -> None:
    result = await _call({"method": "list", "colour": "red"})
    assert result.is_error is True
    assert "invalid arguments" in result.text


# --- absences ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_app_is_answered_as_absence_not_as_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The honest answer, without a socket and without a hang."""
    monkeypatch.setattr(builtin, "_ui_console_liveness", lambda: (Liveness.ABSENT, _record()))
    client = _install(monkeypatch, _FakeClient())
    result = await _call({"method": "list"})
    assert result.is_error is True
    assert "No Local Operator desktop app is running" in result.text
    assert client.calls == [], "an absent app must not be dialled"


@pytest.mark.asyncio
async def test_an_app_with_its_console_off_says_which_remedy_to_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        builtin,
        "_ui_console_liveness",
        lambda: (Liveness.FRESH, _record(console=False)),
    )
    client = _install(monkeypatch, _FakeClient())
    result = await _call({"method": "list"})
    assert result.is_error is True
    assert "reports its console feature as unavailable" in result.text
    assert "LOCAL_OPERATOR_UI_CONSOLE_HOST=0" in result.text
    assert client.calls == []


# --- the wire ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_passes_the_sessions_own_id_and_renders_provenance(
    monkeypatch: pytest.MonkeyPatch, live_app: None
) -> None:
    """R18: a surface the USER opened must be visible AS such in the listing."""
    client = _install(
        monkeypatch,
        _FakeClient(
            {
                "surfaces": [
                    {
                        "surface": "con:1:aaaa",
                        "session_id": "s-1",
                        "origin": "user",
                        "command": "zsh",
                        "cwd": "/tmp",
                        "cols": 100,
                        "rows": 30,
                        "running": True,
                        "live": True,
                        "last_activity": "4s ago",
                    }
                ]
            }
        ),
    )
    result = await _call({"method": "list"})
    assert result.is_error is False, result.text
    assert client.calls == [("console_list", {"session_id": "s-1"})]
    assert "con:1:aaaa" in result.text
    assert "user" in result.text
    assert "zsh" in result.text


@pytest.mark.asyncio
async def test_create_sends_only_what_the_method_defines(
    monkeypatch: pytest.MonkeyPatch, live_app: None
) -> None:
    client = _install(
        monkeypatch,
        _FakeClient(
            {
                "surface": "con:2:bbbb",
                "cols": 120,
                "rows": 40,
                "pid": 999,
                "live": True,
                "revealed": False,
            }
        ),
    )
    result = await _call(
        {
            "method": "create",
            "command": "htop",
            "args": ["-d", "2"],
            "cwd": "/tmp",
            "cols": 120,
            "rows": 40,
        }
    )
    method, params = client.calls[0]
    assert method == "console_create"
    # Deliberately absent: no `text`, no `secret_ref`, no `paste`, no `retain`,
    # no `reveal` — a pass-through of the whole model would put every unrelated
    # argument on the wire (and 20 keys is a lot of noise for the app to sift).
    assert params == {
        "session_id": "s-1",
        "cols": 120,
        "rows": 40,
        "command": "htop",
        "cwd": "/tmp",
        "args": ["-d", "2"],
    }
    assert "con:2:bbbb" in result.text
    # The downgrade is reported, not hidden: `revealed: false` means the pane
    # was left alone because opening it would have raised the app's window.
    assert "revealed" in result.text or "left alone" in result.text


@pytest.mark.asyncio
async def test_read_renders_the_text_and_the_footer_facts(
    monkeypatch: pytest.MonkeyPatch, live_app: None
) -> None:
    client = _install(
        monkeypatch,
        _FakeClient(
            {
                "text": "hello from the pty\n",
                "cols": 100,
                "rows": 30,
                "cursor": {"row": 1, "col": 4},
                "truncated": False,
                "live": True,
                "mode": "viewport",
            }
        ),
    )
    result = await _call({"method": "read", "surface": "con:1:a"})
    assert client.calls == [("console_read", {"surface": "con:1:a", "mode": "viewport"})]
    assert "hello from the pty" in result.text
    assert "stdout and stderr are one pty stream" in result.text
    assert "100x30" in result.text


@pytest.mark.asyncio
async def test_status_does_not_invent_a_waiting_for_input_verdict(
    monkeypatch: pytest.MonkeyPatch, live_app: None
) -> None:
    """§11.1 rejected the heuristic; the renderer must not smuggle it back."""
    _install(
        monkeypatch,
        _FakeClient({"running": True, "cols": 100, "rows": 30, "idle_ms": 2500, "live": True}),
    )
    result = await _call({"method": "status", "surface": "con:1:a"})
    assert "2.5s since the last output" in result.text
    assert "waiting for input" in result.text  # as the honest caveat sentence...
    assert "running: True" in result.text
    # ...and never as a field the app did not send.
    assert "waiting_for_input" not in result.text


@pytest.mark.asyncio
async def test_typed_refusals_are_rendered_from_the_code_not_the_message(
    monkeypatch: pytest.MonkeyPatch, live_app: None
) -> None:
    for code, expected in (
        (ErrorCode.UNSUPPORTED_METHOD, "has no console. Update Local Operator"),
        (ErrorCode.SURFACE_NOT_OWNED, "belongs to another session"),
        (ErrorCode.PROCESS_EXITED, "has exited"),
        (ErrorCode.SECURE_INPUT_ACTIVE, "refuses to read"),
        (ErrorCode.CONSOLE_CAPTURE_FULL, "one at a time"),
    ):
        _install(
            monkeypatch,
            _FakeClient(error=BridgeError(code, "a host sentence that must not be used")),
        )
        result = await _call({"method": "status", "surface": "con:1:a"})
        assert result.is_error is True
        assert expected in result.text, (code, result.text)
        assert "a host sentence that must not be used" not in result.text


@pytest.mark.asyncio
async def test_a_dead_app_mid_call_gets_the_transport_sentence(
    monkeypatch: pytest.MonkeyPatch, live_app: None
) -> None:
    _install(
        monkeypatch,
        _FakeClient(error=BridgeUnreachable(console_backend.CONSOLE_COPY.no_state)),
    )
    result = await _call({"method": "list"})
    assert result.is_error is True
    assert "not running" in result.text
    assert "cannot outlive it" in result.text


@pytest.mark.asyncio
async def test_the_console_client_never_waits_like_a_browser_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The console sizes its OWN calls: no method waits on a human.

    Inheriting the browser's `base + ORIGIN_PROMPT_WINDOW_S + margin` arithmetic
    would give an unknown console method a 190-second ceiling and turn a wedged
    app into a three-minute hang — the outcome §15's absence copy exists to
    replace with an honest answer.
    """
    assert console_backend.console_timeout("console_list") == 20.0
    assert console_backend.console_timeout("console_screenshot") == 65.0
    assert console_backend.console_timeout("console_invented") == 65.0
    assert console_backend.console_timeout("console_list") < 190.0


# --- screenshots ------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_screenshot_is_written_only_when_it_is_really_a_png(
    monkeypatch: pytest.MonkeyPatch, live_app: None, tmp_path: Path
) -> None:
    _install(
        monkeypatch,
        _FakeClient(
            {
                "image_base64": base64.b64encode(PNG).decode(),
                "rendered": "offscreen",
                "cols": 100,
                "rows": 30,
                "theme": "dark",
                "live": True,
            }
        ),
    )
    result = await _call({"method": "screenshot", "surface": "con:1:a"})
    assert result.is_error is False, result.text
    details = result.details or {}
    assert "path" in details
    path = Path(str(details["path"]))
    assert path.read_bytes() == PNG
    # `rendered` is the honest half of the answer and rides in the text: an
    # offscreen frame is a reconstruction, not a photograph of a live screen.
    assert "offscreen" in result.text
    assert "reconstruction from the surface's record" in result.text


@pytest.mark.asyncio
async def test_a_non_png_capture_is_refused_rather_than_written(
    monkeypatch: pytest.MonkeyPatch, live_app: None
) -> None:
    _install(
        monkeypatch,
        _FakeClient({"image_base64": base64.b64encode(b"<html>not a png</html>").decode()}),
    )
    result = await _call({"method": "screenshot", "surface": "con:1:a"})
    assert result.is_error is True
    assert "is not a PNG" in result.text


# --- the secret path --------------------------------------------------------
#
# The store is the REAL `VariableStore`, not a double: the property under test is
# that the session's own redaction seam contains the value afterwards, and a fake
# that merely recorded the call would assert only that the fake was called.

SECRET = "correct-horse-battery-staple"


def _store() -> Any:
    from local_operator.variables import VariableStore

    return VariableStore(cwd=".")


@pytest.fixture
def stored_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real resolution path, minus the encrypted store on disk."""
    from local_operator.secrets import access

    monkeypatch.setattr(access, "retrieve_secret", lambda name, base=None: SECRET.encode())


@pytest.mark.asyncio
async def test_a_secret_ref_reaches_the_pty_without_reaching_the_model(
    monkeypatch: pytest.MonkeyPatch, live_app: None, stored_secret: None
) -> None:
    """The negative test the brief asks for, asserted rather than argued.

    Three surfaces of the model's view must be clean and one must not be: its tool
    RESULT, the arguments it emitted (what a trace renders), and the session's
    output afterwards — while the value does travel to the app in this one call's
    params, because the app is the only process that can write to a pty and §11.3
    states that disclosure instead of glossing it.
    """
    store = _store()
    client = _install(monkeypatch, _FakeClient({"accepted": True, "bytes": len(SECRET)}))
    args = {"method": "input", "surface": "con:1:a", "secret_ref": "SUDO_PASSWORD"}
    result = await _call(args, _context(variables=store))

    assert result.is_error is False, result.text
    method, params = client.calls[0]
    assert method == "console_input"
    # The app receives the value: the accepted v1 disclosure (§11.3), asserted here
    # so the claim stays honest in BOTH directions rather than being quietly
    # upgraded to "the value never leaves the runtime".
    assert params["text"] == SECRET
    # The model's RESULT does not.
    assert SECRET not in result.text
    assert "SUDO_PASSWORD" in result.text
    # The ARGUMENTS the model emitted name the ref, which is the whole point of the
    # indirection: a transcript rendering of this call says `secret_ref:
    # "SUDO_PASSWORD"` and nothing else.
    assert SECRET not in repr(args)
    assert SECRET not in repr(result.details)
    # And the value is contained for the rest of the session through the store's
    # real redaction seam — the same one `session/session.py` hands the loop as
    # `redact_tool_result`.
    assert SECRET in store.redaction_values()
    assert store.redact(f"env SUDO_PASSWORD={SECRET}") == "env SUDO_PASSWORD=[redacted]"


@pytest.mark.asyncio
async def test_a_literal_text_input_registers_nothing(
    monkeypatch: pytest.MonkeyPatch, live_app: None
) -> None:
    """Only the resolved value is registered: `text` is the model's own string,
    already in its context by definition, and registering it would scrub an
    ordinary word out of every later result."""
    store = _store()
    _install(monkeypatch, _FakeClient({"accepted": True, "bytes": 3}))
    await _call(
        {"method": "input", "surface": "con:1:a", "text": "ls\n"}, _context(variables=store)
    )
    assert store.redaction_values() == []


@pytest.mark.asyncio
async def test_no_redaction_sink_fails_closed(
    monkeypatch: pytest.MonkeyPatch, live_app: None, stored_secret: None
) -> None:
    """A store that cannot contain the value must not receive it.

    This is the security-relevant branch: without the registration, a program that
    echoes its input would put the plaintext into a tool result with nothing to
    mask it. So the refusal happens BEFORE the value is fetched and before the app
    is dialled, and it names a path that does contain the value.
    """
    client = _install(monkeypatch, _FakeClient())
    result = await _call(
        {"method": "input", "surface": "con:1:a", "secret_ref": "SUDO_PASSWORD"},
        _context(variables=None),
    )
    assert result.is_error is True
    assert "no redaction sink" in result.text
    assert "lop secret get" in result.text
    assert client.calls == []


@pytest.mark.asyncio
async def test_an_unresolvable_secret_ref_is_refused_without_dialling(
    monkeypatch: pytest.MonkeyPatch, live_app: None, stored_secret: None
) -> None:
    from local_operator.secrets import access

    def boom(name: str, base: Path | None = None) -> bytes:
        raise KeyError(name)

    monkeypatch.setattr(access, "retrieve_secret", boom)
    client = _install(monkeypatch, _FakeClient())
    result = await _call(
        {"method": "input", "surface": "con:1:a", "secret_ref": "NOPE"},
        _context(variables=_store()),
    )
    assert result.is_error is True
    assert "could not resolve secret_ref" in result.text
    # The remedy is named, and it is not "put the value in text".
    assert "do not put the value itself in `text`" in result.text
    assert client.calls == []


# --- the approval prompt ----------------------------------------------------


def test_the_approval_prompt_names_the_surface_the_session_and_the_payload() -> None:
    """§11.1: the prompt is the only place a person learns what they are
    authorising, and the JSON fallback buries all three of these."""
    describe = builtin._describe_console_approval
    context = _context(session_id="s-9")
    typed = describe(
        {"method": "input", "surface": "con:1:a", "text": "sudo rm -rf /tmp/x\n"},
        "/w",
        context=context,
    )
    assert "con:1:a" in typed and "s-9" in typed
    assert "sudo rm -rf /tmp/x" in typed
    by_ref = describe(
        {"method": "input", "surface": "con:1:a", "secret_ref": "SUDO_PASSWORD"},
        "/w",
        context=context,
    )
    assert "SUDO_PASSWORD" in by_ref and SECRET not in by_ref
    keys = describe(
        {"method": "keys", "surface": "con:1:a", "keys": ["ctrl-c", "up"]}, "/w", context=context
    )
    assert "ctrl-c" in keys and "up" in keys
    created = describe(
        {"method": "create", "surface": "", "command": "python3", "args": ["-i"], "cwd": "/app"},
        "/w",
        context=context,
    )
    assert "python3 -i" in created and "/app" in created and "(a new surface)" in created
    # A long payload is bounded: the prompt names the decision rather than
    # reproducing it.
    long = describe(
        {"method": "input", "surface": "con:1:a", "text": "x" * 5000}, "/w", context=context
    )
    assert len(long) < 300
