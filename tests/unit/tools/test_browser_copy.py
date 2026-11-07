"""The `copy` action: routing, the receipt, and the fact that no text travels.

The action exists because a value shown exactly once is lost when the page's own
Copy button fails (measured 2026-11-06 in the desktop app's browser host: the
page's ``navigator.clipboard.writeText`` is refused, because the browser session
is default-deny for clipboard-write). The fix has two halves: the UI repo grants
the page's write, and this half gives the harness a route that needs no page
affordance at all — the APP's main process reads one element and writes the OS
clipboard itself.

Everything asserted here is a property the HARNESS owns, so none of it trusts the
host:

* `copy` is desktop-only and gated on the host's advertised capabilities, so an
  extension host and a pre-`copy` app are refused in words that name the remedy
  instead of sending a method the peer answers with a bare `internal`.
* the result is a RECEIPT (``chars``, a ``sha256`` prefix, ``verified``) and never
  the copied bytes — including when the host is generous, wrong-typed or hostile
  about what it returns. The element is named by the caller's own selector, so no
  page-supplied string reaches the receipt at all.
* a failure names the failure, without relaying a host message that could quote
  what the page held.
* the guide still carries the recipe that makes the copied value usable without
  ever entering a transcript.

Every string in this file is synthetic. A real value must never appear in a test
fixture either: fixtures get printed, quoted in review comments and pasted into
bug reports.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

import local_operator.tools.builtin as builtin
from local_operator.browser_bridge.backend import (
    HOST_EXTENSION,
    HOST_UI,
    BridgeError,
    HostCapabilities,
)
from local_operator.browser_bridge.protocol import (
    CAPABILITY_GATED_METHODS,
    COMMAND_TIMEOUTS,
    DESKTOP_ONLY_METHODS,
    METHODS,
    PROTO_VERSION,
    ErrorCode,
)
from local_operator.harness.types import BrowserSurface, ToolContext
from local_operator.ui_browser import backend as backend_mod
from local_operator.ui_browser import state as state_store

#: A synthetic "one-time key" the fake host reports copying. Shaped like the
#: real thing only in being long and non-alphabetic; it identifies no account.
POISON = "SYNTH-copy-token-9f3a2b-do-not-print"

#: The same idea where an element IDENTITY would go. Deliberately not
#: identifier-shaped (whitespace, a newline, a quote), because the receipt does
#: accept an element's own id/name — a page's markup writes those — so the
#: assertion that matters for those fields is that free text does not fit the
#: grammar, not that a name-shaped value is impossible (see
#: `_COPY_ELEMENT_PATTERNS`).
POISON_FREE = 'SYNTH copy "token"\n9f3a2b'

GUIDE = Path(__file__).resolve().parents[3] / "local_operator" / "guides" / "browser" / "GUIDE.md"


def _params(**kwargs: Any) -> builtin.BrowserParams:
    return builtin.BrowserParams(action="copy", **kwargs)


def _surface(surface: str = "ui:7:nonce") -> BrowserSurface:
    holder = BrowserSurface()
    holder.surface_id = surface
    return holder


class _Host:
    """A fake host client: a record to answer the capability gate, a wire to answer.

    ``methods`` is what the discovery record advertises — the gate reads it, never
    the socket, so a host that does not serve `copy` must never see the call at
    all (``calls`` stays empty, which is the assertion that matters).
    """

    def __init__(self, *, host: str = HOST_UI, methods: tuple[str, ...] = (), reply: Any = None):
        self.host = host
        self._methods = methods
        self._reply = reply
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def capabilities(self) -> HostCapabilities:
        return HostCapabilities(methods=self._methods, version="0.101.3", capabilities_known=True)

    async def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((method, dict(params)))
        if isinstance(self._reply, Exception):
            raise self._reply
        return dict(self._reply or {})


def _run(client: Any, *, selector: str = "#key") -> Any:
    """One `copy` through `_bridge_action`, over the real capability gate."""
    return asyncio.run(
        builtin._bridge_action(
            "call",
            _surface(),
            "copy",
            _params(selector=selector),
            ToolContext(cwd="."),
            client=client,
        )
    )


# ---------------------------------------------------------------------------
# The protocol contract: additive, desktop-only, gated
# ---------------------------------------------------------------------------


def test_copy_is_an_additive_desktop_only_gated_action() -> None:
    assert "copy" in builtin.BROWSER_ACTIONS
    assert "copy" in builtin.CMUX_UNSUPPORTED_BROWSER_ACTIONS
    assert "copy" in METHODS
    assert "copy" in DESKTOP_ONLY_METHODS
    assert "copy" in CAPABILITY_GATED_METHODS
    assert "copy" in COMMAND_TIMEOUTS
    # Additive: the capability advertisement is the gate, so the wire version
    # must not move. A bump here would strand every already-installed peer.
    assert PROTO_VERSION == 1


def test_copy_requires_a_selector() -> None:
    # No "copy whatever is focused": the caller names the element, and the
    # receipt names it back.
    problem = builtin._validate_browser_args("copy", _params(selector="  "))
    assert "requires a selector" in problem
    assert builtin._validate_browser_args("copy", _params(selector="#key")) == ""
    # The argv-shaped refusal the other page-script actions share.
    assert "flag-shaped" in builtin._validate_browser_args("copy", _params(selector="-x"))


# ---------------------------------------------------------------------------
# Routing: the desktop app's host only, refused by name everywhere else
# ---------------------------------------------------------------------------


def test_extension_host_refuses_copy_before_the_wire() -> None:
    client = _Host(host=HOST_EXTENSION, methods=("open", "goto"), reply={})
    result = _run(client)
    assert result.details["error_code"] == "capability_unsupported"
    assert "desktop app" in result.text
    # The one thing that must be true of a desktop-only action: the peer that
    # cannot serve it never sees the request.
    assert client.calls == []


def test_pre_copy_desktop_app_refuses_copy_and_names_the_update() -> None:
    # A record from an app build that predates the action. Sending it would earn
    # a bare `internal` from a dispatch with no handler, which reads to a model
    # as "the bridge broke".
    client = _Host(host=HOST_UI, methods=("open", "read", "query"), reply={})
    result = _run(client)
    assert result.details["error_code"] == "capability_unsupported"
    assert "built before this action existed" in result.text
    # The measured 2026-11-06 incident: this copy read as a dead end, so its
    # reader photographed a tenant-wide admin API key off the screen. Pin what
    # it has to carry now — the METHOD, the HOST, and the fact that only that
    # host serves it — because dropping any of the three brings the dead end
    # back. `copy` deliberately has no recorded app-version floor, so the copy
    # must SAY that rather than invent one from the harness release.
    assert "does not provide 'copy'" in result.text
    assert "desktop app's browser host" in result.text
    assert "only that host serves it" in result.text
    assert "no desktop-app version floor is recorded for it" in result.text
    assert "update the desktop app, then retry" in result.text
    # The exit that breaks the update-retry loop (review R1-2): while no released
    # app serves `copy`, "update" alone sends the reader round in a circle, and
    # the circle ended at a screenshot. Conditional, so it stays true later.
    assert "if the latest desktop app still refuses it" in result.text
    assert "the service's API or a download instead, never from the screen" in result.text
    assert client.calls == []


def test_the_desktop_only_scope_and_copy_exit_stay_off_shared_methods() -> None:
    # `download` is served by BOTH hosts, so "only that host serves it" would
    # send a caller away from an extension that works (review R1-3), and the
    # copy-specific API/download exit means nothing for a download.
    from local_operator.browser_bridge.backend import capability_refusal, format_error

    capabilities = HostCapabilities(methods=("open",), version="0.101.3", capabilities_known=True)
    error = capability_refusal("download", host=HOST_UI, capabilities=capabilities)
    message = format_error(error, action="download", host=HOST_UI)
    assert "does not provide 'download'" in message
    assert "only that host serves it" not in message
    assert "never from the screen" not in message
    assert message.endswith("every other browser action still works.")


def test_a_recorded_app_floor_is_named_in_the_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The floor, when one IS recorded, is named without a second edit.

    No released desktop build's floor is measured today, so the table is empty
    and the refusal says so. This drives the other arm: a floor recorded here
    reaches the copy as a concrete version, which is what makes "record it when
    you measure it" a data change rather than a copy rewrite.
    """
    from local_operator.browser_bridge import backend as bridge_backend

    monkeypatch.setattr(bridge_backend, "_CAPABILITY_MIN_APP_VERSION", {"copy": "0.102.0"})
    client = _Host(host=HOST_UI, methods=("open", "read"), reply={})
    result = _run(client)
    assert result.details["error_code"] == "capability_unsupported"
    # `_Host` reports 0.101.3, which IS older than the floor: name both.
    assert "this app (0.101.3) predates 0.102.0" in result.text
    assert "update the desktop app, then retry" in result.text
    assert "no desktop-app version floor is recorded" not in result.text
    assert client.calls == []


def test_an_app_at_the_recorded_floor_is_not_told_to_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Review R1-4: the extension arm says "update" only when `extension_older`
    # proves the peer is older; an app already past the floor that still does
    # not advertise the method would loop on an update that cannot help.
    from local_operator.browser_bridge import backend as bridge_backend

    monkeypatch.setattr(bridge_backend, "_CAPABILITY_MIN_APP_VERSION", {"copy": "0.101.0"})
    client = _Host(host=HOST_UI, methods=("open", "read"), reply={})
    result = _run(client)
    assert result.details["error_code"] == "capability_unsupported"
    assert "updating is not the fix" in result.text
    assert "update the desktop app" not in result.text
    assert "never from the screen" in result.text
    assert client.calls == []


@pytest.mark.asyncio
async def test_cmux_refuses_copy_naming_only_the_desktop_app(monkeypatch) -> None:
    """One host, not "either non-cmux host".

    The generic cmux sentence ends with `_non_cmux_host_hint`, which offers the
    extension as an alternative — and the extension refuses `copy` too, because a
    clipboard write needs a main process it does not have. Offering it sends the
    agent to a second dead end (review R1-2).
    """
    monkeypatch.setattr(builtin, "ui_browser_available", lambda: True)
    monkeypatch.setattr(builtin, "ui_browser_reachable", AsyncMock(return_value=True))
    holder = BrowserSurface()
    holder.surface_id = "surface:cmux-open"
    result = await builtin.execute_browser(
        "t", {"action": "copy", "selector": "#key"}, None, None, ToolContext(browser=holder)
    )
    assert result.is_error
    assert "not supported on the cmux backend" in result.text
    assert "desktop app's browser host" in result.text
    # The wording that would send it to a host which cannot serve it either.
    assert "browser extension" not in result.text
    assert "either" not in result.text


def test_current_desktop_app_receives_the_selector_and_nothing_else() -> None:
    client = _Host(
        host=HOST_UI,
        methods=("open", "read", "copy"),
        reply={"chars": 4, "sha256": "0" * 16, "verified": True, "element": {"tag": "input"}},
    )
    result = _run(client, selector="  #key  ")
    assert len(client.calls) == 1
    method, wire = client.calls[0]
    assert method == "copy"
    assert wire["selector"] == "#key"
    # Identity metadata rides every owned-tab command; nothing else may. In
    # particular there is no text/value field: the host reads the element itself,
    # so a caller cannot ask for an arbitrary string to be put on the clipboard.
    assert not set(wire) - {"tab", "selector", "requester", "session_label"}
    assert not result.is_error


# ---------------------------------------------------------------------------
# The receipt: no text, whatever the host returns
# ---------------------------------------------------------------------------


def _poisoned_reply() -> dict[str, Any]:
    """A host being generous: the copied text under every name it could pick."""
    return {
        "chars": len(POISON),
        "sha256": "0a1b2c3d4e5f6071",
        "verified": True,
        "element": {"tag": "input", "id": "key", "name": "key", "value": POISON},
        "text": POISON,
        "value": POISON,
        "copied": POISON,
        "clipboard": {"text": POISON},
    }


def test_the_receipt_carries_no_copied_text() -> None:
    client = _Host(host=HOST_UI, methods=("copy",), reply=_poisoned_reply())
    result = _run(client)
    assert not result.is_error
    # The whole contract, in one assertion each: what the model reads, and what
    # a renderer/compactor sees, must both be free of the value.
    assert POISON not in result.text
    assert POISON not in json.dumps(result.details, default=str)
    assert result.details["chars"] == len(POISON)
    assert result.details["sha256"] == "0a1b2c3d4e5f6071"
    assert result.details["verified"] is True
    # No `element` at all: the receipt names the element from the CALLER's
    # selector, so there is no page-supplied string in it (review R1-1).
    assert "element" not in result.details
    assert result.details["selector"] == "#key"
    assert "'#key'" in result.text
    # And it must say where the value IS, because it is nowhere in this result.
    assert "pbpaste" in result.text and "lop secret set" in result.text


def test_wrongly_typed_fields_are_dropped_not_coerced() -> None:
    # `chars` holding a string is the copied text under the wrong name; a truthy
    # `verified` string is a readback that never happened; an element field that
    # is not an identifier is free text pasted where a name belongs. None may
    # survive.
    client = _Host(
        host=HOST_UI,
        methods=("copy",),
        reply={
            "chars": POISON,
            "sha256": POISON,
            "verified": "yes",
            "element": {"tag": POISON_FREE, "id": POISON_FREE, "name": 42},
        },
    )
    result = _run(client)
    assert POISON not in result.text
    assert POISON not in json.dumps(result.details, default=str)
    assert "SYNTH copy" not in result.text
    assert "SYNTH copy" not in json.dumps(result.details, default=str)
    assert result.details["chars"] == 0
    assert result.details["sha256"] == ""
    assert result.details["verified"] is False
    assert "element" not in result.details
    # Unverified and unmeasured must read as unconfirmed, not as a copy.
    assert "unconfirmed" in result.text or "could not confirm" in result.text


def test_the_element_is_named_by_the_callers_selector() -> None:
    # The label is the selector the caller typed, which is the only element name
    # the receipt is allowed to carry: a real page id (`secret-key`, `api_key`)
    # is indistinguishable from a token-shaped secret, so no host-supplied string
    # is rendered at all (review R1-1).
    client = _Host(
        host=HOST_UI,
        methods=("copy",),
        reply={
            "chars": 12,
            "sha256": "ab12cd34ef56ab78",
            "verified": True,
            "element": {"tag": "input", "id": "secret-key"},
        },
    )
    result = _run(client, selector="#api-key-field")
    assert "'#api-key-field'" in result.text
    assert "secret-key" not in result.text
    assert "secret-key" not in json.dumps(result.details, default=str)


def test_an_uppercase_digest_is_normalised_not_silently_dropped() -> None:
    # The prefix contract is a sha256 spelling, not a field case: an uppercase
    # producer spelled the same digest differently, and losing the field would
    # read to the caller as "the host reported no digest" (review R1-4).
    client = _Host(
        host=HOST_UI,
        methods=("copy",),
        reply={"chars": 4, "sha256": "AB12CD34EF56AB78", "verified": True},
    )
    result = _run(client)
    assert result.details["sha256"] == "ab12cd34ef56ab78"
    assert "ab12cd34ef56ab78" in result.text


def test_a_digest_that_is_not_a_prefix_is_not_reported_as_one() -> None:
    # A truncated copy of the value would look like a fresh digest if the field
    # were only length-capped, so the shape is matched, not just its size.
    client = _Host(
        host=HOST_UI,
        methods=("copy",),
        reply={"chars": 4, "sha256": "KEY!KEY!", "verified": True, "element": {}},
    )
    result = _run(client)
    assert result.details["sha256"] == ""
    assert "KEY!KEY!" not in result.text


# ---------------------------------------------------------------------------
# Failures: typed, actionable, and never a relay for page text
# ---------------------------------------------------------------------------


def test_empty_element_text_is_typed_and_relays_no_page_text() -> None:
    # A host message is untrusted input: it could quote what the page held. The
    # rendering for this code is fixed copy for exactly that reason.
    error = BridgeError(ErrorCode.EMPTY_ELEMENT_TEXT, f"element held nothing: {POISON}")
    client = _Host(host=HOST_UI, methods=("copy",), reply=error)
    result = _run(client)
    assert result.is_error
    assert result.details["error_code"] == "empty_element_text"
    assert POISON not in result.text
    assert "no text to copy" in result.text
    # And the remedy names the likelier mistake rather than "retry".
    assert "CARRIES the value" in result.text


def test_clipboard_write_failure_is_typed_and_relays_no_page_text() -> None:
    error = BridgeError(ErrorCode.CLIPBOARD_WRITE_FAILED, f"clipboard refused {POISON}")
    client = _Host(host=HOST_UI, methods=("copy",), reply=error)
    result = _run(client)
    assert result.is_error
    assert result.details["error_code"] == "clipboard_write_failed"
    assert POISON not in result.text
    assert "nothing landed" in result.text


def test_element_not_found_stays_the_shared_typed_miss() -> None:
    # The existing code is reused rather than reinvented, and its rendering still
    # carries the guidance that made it worth typing in the first place.
    error = BridgeError(ErrorCode.ELEMENT_NOT_FOUND, "element not found: #key")
    client = _Host(host=HOST_UI, methods=("copy",), reply=error)
    result = _run(client)
    assert result.details["error_code"] == "element_not_found"
    assert "element not found" in result.text


# ---------------------------------------------------------------------------
# The guide: requirement 3 cannot silently regress
# ---------------------------------------------------------------------------


def test_guide_carries_the_secret_store_recipe() -> None:
    text = GUIDE.read_text(encoding="utf-8")
    # Prose assertions run against the guide with its line wrapping collapsed:
    # an edit that reflows a paragraph is not a regression, and a test that
    # breaks on wrapping is one nobody keeps.
    flat = " ".join(text.split())
    # The line an operator or an agent runs after a copy. Command substitution is
    # load-bearing (it strips the trailing newline a page may have written), so
    # the exact form is asserted rather than "pbpaste appears somewhere".
    assert "printf '%s' \"$(pbpaste)\" | lop secret set " in text
    assert "--length --fingerprint" in text
    # The two properties a future edit must not quietly drop: the value never
    # enters the transcript, and the page cannot read the clipboard.
    assert "never enters the transcript" in flat
    assert "cannot read the clipboard" in flat
    # The correction that must not regress: pages are granted NO clipboard
    # permission, and the host performs the write itself (there is no
    # write-only grant to describe).
    assert "granted no clipboard permission at all" in flat
    assert "never grants — or implements — a read" in flat
    assert "printf '' | pbcopy" in flat
    # `copy` is named as the fallback that needs no page affordance, and as
    # desktop-host-only.
    assert "`copy` is the fallback" in flat
    assert "desktop app's browser host only" in flat
    # The two limits the recipe has to carry, because both were learned the hard
    # way (2026-11-06): the main-frame restriction on the page's own Copy button
    # with its measured reason, and the interim policy for a host that predates
    # `copy` — prefer an API or a download route, never screenshot/OCR as the
    # routine fallback.
    assert "out of reach of both clipboard routes" in flat
    assert "navigator.userActivation.isActive" in flat
    assert "isolated world the page cannot patch" in flat
    assert "borrow the top frame's activation" in flat
    # Review R1-1 / QA Q-1: `copy` resolves its selector in the top document, so
    # the guide must never offer it as the iframe workaround again.
    assert "selectors cannot cross frames" in flat
    assert "open it top-level and copy there" in flat
    assert "`copy` ACTION against that element" not in flat
    # R1-5: conditional on an app that serves the page write at all.
    assert "On a desktop app that serves the page's clipboard write" in flat
    # The interim paragraph is worded to stay true AFTER the fix ships: it is
    # conditional on the host refusing, never on the feature being absent.
    assert "If the desktop app refuses `copy`, the page's own Copy button is dead" in flat
    assert "prefer an API or a download route" in flat
    assert "compromise" in flat
    assert "screenshot read with OCR as the routine fallback" in flat


# ---------------------------------------------------------------------------
# The real transport: the actual UI client, envelope and Response parsing
# ---------------------------------------------------------------------------
#
# Everything above swaps `_bridge_call`, which is honest for the rendering but
# proves nothing about the wire. These two drive the REAL `UiHostClient` — its
# discovery record, its request envelope, its key header, the RPC response parse
# — with only the app's HTTP listener faked by `httpx.MockTransport`. The app
# half of the feature (`clipboard.writeText` in main) is the UI repo's PR; what
# is provable here is that this side sends exactly `copy` + `selector` and turns
# whatever comes back into a receipt with no text in it.


def _publish_ui_host(tmp_path: Path, *, capabilities: list[str]) -> Any:
    """Write the app's discovery record, as the app would (0600, atomic)."""
    record = state_store.UiHostState.model_validate(
        {
            "pid": os.getpid(),
            "port": 52133,
            "session_key": secrets.token_hex(24),
            "proto": PROTO_VERSION,
            "app_version": "0.101.3",
            "capabilities": capabilities,
        }
    )
    state_store.publish(record, tmp_path)
    return record


def _real_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, handler: Any) -> Any:
    """A `UiHostClient` whose sockets are a mock transport (the repo's own pattern)."""
    real_async_client = httpx.AsyncClient

    def fake_async_client(**kwargs: Any) -> Any:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_async_client(**kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", fake_async_client)
    return backend_mod.UiHostClient(tmp_path)


def test_the_real_client_sends_copy_and_the_receipt_crosses_the_wire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _publish_ui_host(tmp_path, capabilities=["open", "read", "copy"])
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        # The app's own answer, generous on purpose: the receipt the caller reads
        # must be built here, not forwarded.
        return httpx.Response(
            200,
            json={
                "id": body["id"],
                "ok": True,
                "result": {
                    "chars": len(POISON),
                    "sha256": "ab12cd34ef56ab78",
                    "verified": True,
                    "element": {"tag": "input", "id": "key"},
                    "text": POISON,
                },
            },
        )

    client = _real_client(tmp_path, monkeypatch, handler)
    assert isinstance(record.session_key, str)
    result = asyncio.run(
        builtin._bridge_action(
            "call",
            _surface(),
            "copy",
            _params(selector="  #key  "),
            ToolContext(cwd="."),
            client=client,
        )
    )
    assert not result.is_error
    # Exactly one `copy` request, carrying the trimmed selector and the tab,
    # with no text/value slot a caller could use to put anything on the clipboard.
    assert len(seen) == 1
    assert seen[0]["method"] == "copy"
    assert seen[0]["params"]["selector"] == "#key"
    assert not set(seen[0]["params"]) - {"tab", "selector", "requester", "session_label"}
    details = result.details or {}
    assert details["chars"] == len(POISON)
    assert details["sha256"] == "ab12cd34ef56ab78"
    assert "element" not in details
    assert POISON not in result.text
    assert POISON not in json.dumps(details, default=str)


def test_a_real_record_without_copy_never_reaches_the_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The record is the gate, and it is read BEFORE any socket work — that is the
    # whole point of the advertisement. A host that answers a pre-`copy` dispatch
    # with a bare `internal` would read as a broken bridge.
    _publish_ui_host(tmp_path, capabilities=["open", "read"])

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - asserted
        raise AssertionError(f"copy must not be sent to a host that does not serve it: {request}")

    client = _real_client(tmp_path, monkeypatch, handler)
    result = asyncio.run(
        builtin._bridge_action(
            "call",
            _surface(),
            "copy",
            _params(selector="#key"),
            ToolContext(cwd="."),
            client=client,
        )
    )
    assert result.is_error
    assert (result.details or {})["error_code"] == "capability_unsupported"
    assert "built before this action existed" in result.text
