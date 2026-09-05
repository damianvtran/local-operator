"""The runtime's capability surface must be COMPLETE, not merely present.

Four consecutive review rounds each found another capability that did not
survive the owner-path deletion, one instance at a time:

- round 1: `subscribe_frontend` / `subscribe_events` — no session was usable;
- round 3: `run_slash_authoritative` / `cancel_subagents_count` — eleven slash
  commands failed, and Esc silently stopped cancelling subagents;
- round 3 (audit): `complete_aside`, `adopt_aside`, `recall_steer`,
  `slash_images` — four more nobody had reported;
- round 4: five of the eleven commands INSIDE `run_slash_authoritative` were
  never dispatched, so the defect class recurred one level down, inside the
  very method re-homed to end it.

The audit that found the third batch existed only as a table in a pull-request
description, which is not something anyone consults when the protocol next
changes (round 4, Q4). These tests are that audit, executable, and they check
BOTH layers the class has appeared at:

1. every attribute the server asks a handle for is answered by the handle;
2. every slash command the runtime ADVERTISES is dispatched by it.

Derived from the source and from a live class rather than from a hand-written
list: a list would need updating by the same person who forgot to implement
the capability.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

RUNTIME_DIR = Path(__file__).resolve().parents[4] / "local_operator" / "session" / "runtime"

#: Handle attributes the server reads that are genuinely OPTIONAL — each is
#: guarded by a `callable(...)` check whose negative branch is a designed
#: degradation, not a defect. Kept explicit and justified so that adding a
#: capability to this set is a deliberate act rather than a way to silence
#: the test.
OPTIONAL_CAPABILITIES = {
    # Answered by the TUI-owner handle only: a runtime IS the process being
    # stopped, so it has no separate "ask the host to stop" seam.
    "request_stop",
    # v4/v5 protocol additions. An older runtime answering `unknown op` is the
    # documented mixed-version path the viewer degrades against.
    "subscribe_events",
    "job_trajectory",
    # Set by the registrant on itself, not implemented by the handle.
    "_frontend",
    "_install_interactivity_probe",
}


def _handle_attributes_read_by_the_server() -> set[str]:
    """Every handle attribute `server.py` reaches for, by AST rather than grep.

    Covers both shapes the server uses: `getattr(h, "name", None)` capability
    probes and direct `self._handle.name(...)` calls. A regex over one of them
    is how the round-3 audit missed four capabilities.
    """
    return _attributes_read_in((RUNTIME_DIR / "server.py").read_text())


def _attributes_read_in(source: str) -> set[str]:
    """The extractor itself, over arbitrary source.

    Split from its input so the mutation test below can feed it a synthetic
    module and prove the extractor sees a capability HOWEVER it is spelled.
    Testing the real extractor is the point: a second copy written for the
    test would pass while the production one stayed blind.
    """
    tree = ast.parse(source)
    names: set[str] = set()

    # LOCAL ALIASES OF THE HANDLE, discovered rather than hardcoded. The
    # server's own dispatchers open with `h = self._handle` and then call
    # `h.prompt(...)`, `h.slash(...)`, `h.abort()` — a shape that is neither a
    # `getattr` probe nor a `self._handle.x` access, so the audit could not see
    # TEN capabilities including the two most central ones. Proved by mutation:
    # the same missing capability was caught when spelled `self._handle.x` and
    # missed when spelled `h.x` (round 5, Q5).
    #
    # Collected from the assignments themselves so a future `handle = self._handle`
    # in a new dispatcher is covered without editing this list — the hardcoded
    # `{"h", "handle"}` set below is only the receiver names for `getattr`,
    # which is a separate and much older shape.
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
            if (
                isinstance(target, ast.Name)
                and isinstance(value, ast.Attribute)
                and value.attr == "_handle"
            ):
                aliases.add(target.id)

    for node in ast.walk(tree):
        # getattr(h, "name", ...) / getattr(handle, "name", ...)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            target = node.args[0]
            receiver = ""
            if isinstance(target, ast.Name):
                receiver = target.id
            elif isinstance(target, ast.Attribute):
                receiver = target.attr
            if receiver in {"h", "handle", "_handle"} | aliases:
                names.add(node.args[1].value)
        if isinstance(node, ast.Attribute):
            # self._handle.name(...)
            if isinstance(node.value, ast.Attribute) and node.value.attr == "_handle":
                names.add(node.attr)
            # h.name(...) — the aliased read, via a name assigned from the handle
            elif isinstance(node.value, ast.Name) and node.value.id in aliases:
                names.add(node.attr)

    return names


def test_the_owned_handle_answers_every_capability_the_server_asks_for() -> None:
    """The round-1/round-3 defect class, as an enforced property.

    `OwnedSessionHandle` is the handle EVERY session on this release uses, so
    a capability the server asks for and it does not answer is a feature that
    fails for every user — which is exactly how eleven slash commands, both
    aside paths and Esc-cancels-subagents shipped broken behind a green suite.
    """
    from local_operator.session.runtime.owned import OwnedSessionHandle

    asked = _handle_attributes_read_by_the_server()
    assert asked, "the AST walk found no handle attributes — the extractor has drifted"

    missing = sorted(
        name
        for name in asked
        if name not in OPTIONAL_CAPABILITIES and not hasattr(OwnedSessionHandle, name)
    )
    assert not missing, (
        "the runtime's handle does not answer capabilities the server asks it for: "
        f"{missing}. Implement them on OwnedSessionHandle, or add each to "
        "OPTIONAL_CAPABILITIES with the reason its absence is a designed degradation."
    )


def test_every_optional_capability_is_actually_optional() -> None:
    """The escape hatch must not become a place to hide real gaps.

    A name in `OPTIONAL_CAPABILITIES` that the server does NOT guard with a
    `callable(...)` check is a hard requirement wearing an exemption, so the
    exemption list is itself checked against the source.
    """
    source = (RUNTIME_DIR / "server.py").read_text()
    unguarded = []
    for name in sorted(OPTIONAL_CAPABILITIES):
        if name.startswith("_"):
            continue  # registrant-owned attributes, not handle capabilities
        # Anchor on the GETATTR site specifically: a bare `"name"` also
        # appears in op-name sets like `_PAYLOAD_OPS`, and splitting on the
        # first occurrence reads the wrong window.
        anchor = next(
            (
                probe
                for probe in (f'getattr(h, "{name}"', f'getattr(handle, "{name}"')
                if probe in source
            ),
            "",
        )
        if not anchor:
            continue
        window = source.split(anchor, 1)[1][:400]
        if "callable(" not in window and "is None" not in window:
            unguarded.append(name)
    assert (
        not unguarded
    ), f"these are listed optional but the server does not guard them: {unguarded}"


def _advertised_authoritative_commands() -> set[str]:
    from local_operator.session.frontend_state import _slash_capabilities

    return {
        capability.command
        for capability in _slash_capabilities()
        if capability.scope == "authoritative_session"
    }


def _commands_dispatched_by_the_runtime() -> set[str]:
    """The commands `OwnedSessionHandle._slash_result` actually handles.

    Read from the dispatcher body rather than from a list, so a command added
    to the advertisement without a branch here is caught by construction.
    """
    source = (RUNTIME_DIR / "owned.py").read_text()
    body = source.split("async def _slash_result", 1)[1].split("\n    def ", 1)[0]
    return set(re.findall(r'command == "([a-z_]+)"', body))


def test_the_runtime_dispatches_every_slash_command_it_advertises() -> None:
    """The round-4 defect class: advertised, then refused.

    The viewer routes a command BECAUSE the runtime advertises it as
    `authoritative_session`. Advertising eleven and dispatching six meant five
    commands were unreachable on every session — and the fallback told an
    attached user to "reattach", advice that routes straight back here.

    A command that genuinely cannot run in a session process should not be
    advertised as one the session process answers.
    """
    advertised = _advertised_authoritative_commands()
    dispatched = _commands_dispatched_by_the_runtime()
    assert advertised, "no authoritative capabilities found — the extractor has drifted"

    unhandled = sorted(advertised - dispatched)
    assert not unhandled, (
        f"advertised as authoritative_session but never dispatched: {unhandled}. "
        "Implement them in OwnedSessionHandle._slash_result, or stop advertising them."
    )


@pytest.mark.parametrize(
    "command",
    sorted(_advertised_authoritative_commands()),
)
def test_no_advertised_command_tells_an_attached_user_to_reattach(command: str) -> None:
    """The refusal copy must never name an action the user has already taken.

    Every session on this release is detached and the viewer routes before its
    own local handling, so there is no terminal on which "reattach to run it"
    is followable. A command that cannot run here has to say what it actually
    reads and where it does work.
    """
    from local_operator.session.frontend_state import SlashResult
    from local_operator.session.runtime.owned import OwnedSessionHandle

    # Read the STRING CONSTANTS the dispatcher can return, not its comments —
    # the comment explaining this rule contains the word by necessity.
    source = (RUNTIME_DIR / "owned.py").read_text()
    tree = ast.parse(source)
    literals: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in {
            "_slash_result",
            "_effort_slash",
            "_fast_slash",
            "_model_slash",
            "_context_slash",
            "_team_slash",
            "_agent_slash",
            "_mcp_slash",
        }:
            for inner in ast.walk(node):
                if isinstance(inner, ast.Constant) and isinstance(inner.value, str):
                    literals.append(inner.value)

    offenders = [text for text in literals if "reattach" in text.lower()]
    assert not offenders, (
        f"the routed-slash dispatcher tells the user to reattach: {offenders}. "
        "Every session is detached and the viewer routes before its own local "
        "handling, so that instruction cannot be followed."
    )
    assert command in _advertised_authoritative_commands()
    assert SlashResult is not None and OwnedSessionHandle is not None


def test_the_runtime_knows_every_mcp_verb_the_terminal_offers() -> None:
    """`/mcp add` and `/mcp remove` used to fall through to a server LISTING.

    That was worse than an error: a table of servers is a plausible answer to
    `add`, so the user read it as "done, here is the current state" and
    discovered days later that nothing had been written — and `/mcp remove
    github` had the same shape pointed at deletion (round 5, U15).

    The verb list is canonical in `session.frontend_state` precisely so the
    two surfaces cannot know different sets; this pins that they don't.
    """
    from local_operator.session.frontend_state import MCP_SUBCOMMANDS
    from local_operator.tui.app import OperatorApp

    assert OperatorApp.MCP_SUBCOMMANDS == MCP_SUBCOMMANDS

    source = (RUNTIME_DIR / "owned.py").read_text()
    body = source.split("def _mcp_slash", 1)[1].split("\n    def ", 1)[0]
    # Every mutating verb must be dispatched by name somewhere in the handler.
    for verb in ("add", "remove"):
        assert f'"{verb}"' in body, (
            f"/mcp {verb} is not dispatched by the runtime, so it falls through "
            "to a listing that looks like success"
        )


@pytest.mark.parametrize(
    "spelling",
    [
        "self._handle.{name}()",
        # The alias, which the extractor was blind to. `_dispatch` opens with
        # `h = self._handle` and calls `h.prompt(...)` / `h.slash(...)`, so
        # this is the shape the server ACTUALLY uses for its most central
        # capabilities — and the audit could not see ten of them (round 5, Q5).
        "h = self._handle\n    h.{name}()",
        'getattr(self._handle, "{name}", None)',
        'getattr(h, "{name}", None)',
    ],
)
def test_the_extractor_sees_a_capability_however_it_is_spelled(spelling: str) -> None:
    """QA's mutation, committed so the blind spot cannot silently return.

    The guard this file provides is only worth what its extractor can see. QA
    proved the same missing capability was CAUGHT when written
    `self._handle.x` and MISSED when written `h.x` — so whether the audit
    protects a capability depended on the author's spelling, not on the code.

    Runs the real extractor against a synthetic module rather than asserting
    against a name list: a list would be updated by the same person who
    forgot to teach the extractor a new shape.
    """
    import ast as _ast

    marker = "qa5_probe_capability"
    statements = spelling.format(name=marker).split("\n")
    module = "class RuntimeServer:\n    def _dispatch(self):\n" + "".join(
        f"        {line.strip()}\n" for line in statements
    )
    # Parse the fixture first: a malformed synthetic module would otherwise
    # yield an empty name set and read as a blind spot that is not there.
    _ast.parse(module)

    seen = _attributes_read_in(module)
    assert marker in seen, (
        f"the extractor cannot see a capability spelled {spelling!r}; a guard "
        "that depends on how a call is written does not guard anything"
    )


def test_the_runtime_applies_fast_mode_to_the_spec_it_builds_requests_from() -> None:
    """`/fast` from a phone reaches the DETACHED runtime's spec.

    The dial only matters on the spec the next provider call is built from,
    and in a headless process that is the runtime's own — the terminal's copy
    of the rule is not loaded there. A producer that reported success without
    writing the spec would be the round-2 MAJOR-1 shape: a receipt for an
    operation that never ran.
    """
    from local_operator.model.configure import build_model_spec
    from local_operator.session.frontend_state import SlashResult
    from local_operator.session.runtime.owned import OwnedSessionHandle

    class _Session:
        model_label = "anthropic/claude-opus-5"

        def __init__(self) -> None:
            self.model = build_model_spec("anthropic", "claude-opus-5")

        def set_model(self, spec, *, explicit: bool = False) -> None:  # noqa: ANN001
            self.model = spec

    handle = OwnedSessionHandle.__new__(OwnedSessionHandle)
    handle._notify = lambda: None  # type: ignore[method-assign]
    session = _Session()
    assert session.model.supports_fast_mode and not session.model.fast_mode

    on = handle._fast_slash(session, "", SlashResult)
    assert session.model.fast_mode is True
    assert "premium" in on.text.lower()

    again = handle._fast_slash(session, "on", SlashResult)
    assert session.model.fast_mode is True, "`on` names a state, it does not flip"
    assert "on" in again.text.lower()

    off = handle._fast_slash(session, "off", SlashResult)
    assert session.model.fast_mode is False
    assert "standard" in off.text.lower()

    bad = handle._fast_slash(session, "maybe", SlashResult)
    assert bad.style == "warning" and session.model.fast_mode is False

    # `on` clears the driver's refusal latch through the stream fn hook.
    class _Stream:
        forgotten = 0

        def forget_fast_refusal(self) -> None:
            self.forgotten += 1

    session._stream_fn = _Stream()
    handle._fast_slash(session, "on", SlashResult)
    assert session._stream_fn.forgotten == 1

    # A route with no fast tier says so rather than taking a state the wire
    # would silently drop.
    session.model = build_model_spec("google", "gemini-3-pro")
    refused = handle._fast_slash(session, "on", SlashResult)
    assert "not available" in refused.text.lower()
    assert session.model.fast_mode is False
