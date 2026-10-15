"""Which host owns a session's surface, and the gate that decides whether the
ownership lane runs at all.

The defect these tests pin is a closed loop: the lane's gate read
``resource.generation``, which only the lane's own ``initialize()`` ever sets, so
on a host with no reachable bridge the lane was skipped on the first action and
therefore on every action. `recover`, `retain` and `release` then fell through the
dispatcher to its last branch — a screenshot — and reported success.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.browser_bridge import resources
from local_operator.browser_bridge.resources import (
    BrowserOwnershipError,
    BrowserResource,
)
from local_operator.harness.types import BrowserSurface, ToolContext
from local_operator.tools import builtin


class FakeUiClient:
    """A UI host that answers the ownership verbs and records what it was sent."""

    host = "ui"

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        #: How this host answers `owner_recover`. The default is a session that
        #: ALREADY holds a tab, which is what most ownership tests describe, and
        #: `recover()` adopts whatever `tab` it returns into the record — so a test
        #: about a FRESH session (adoption of a tab the user handed over, where
        #: nothing is held yet) must say `unresolved` explicitly rather than
        #: inheriting a held tab from the fixture.
        self.recover_state = "owned"

    async def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((method, params))
        if method == "open":
            return {"tab": "ui:100:aaaaaaaabbbbccccddddeeeeffff0000", "url": params["url"]}
        if method == "read":
            return {"url": "https://example.com/", "title": "Example Domain", "text": "body"}
        if method == "tabs":
            return {"tabs": [], "limit": 8}
        if method == "owner_recover":
            if self.recover_state == "unresolved":
                return {"ownership_version": 1, "state": "unresolved", "tab": ""}
            if self.recover_state == "closed":
                # The app's real answer for a settled scope with no live tab
                # (local-operator-ui src/main/browser/ownership.ts): a SUCCESSFUL
                # acknowledgement, which is what collapses the replay chain.
                return {"ownership_version": 1, "state": "closed", "tab": ""}
            return {"ownership_version": 1, "state": "owned", "tab": "ui:100:capability"}
        if method == "owner_retain":
            return {"state": "retained"}
        if method == "owner_release":
            return {"state": "closed"}
        if method == "owner_finish":
            return {"state": "closed"}
        return {}


class FakeBridgeClient:
    """The daemon's transport, faking the wire the same way `FakeUiClient` does.

    Both hosts are faked rather than one, because the defect these tests pin is
    the SELECTION between them: a test with only one fake cannot tell "the lane
    went to the right host" from "there was only ever one host".
    """

    host = "extension"

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((method, params))
        if method == "open":
            return {"tab": "bridge:5:ccccccccddddeeeeffff0000111111", "url": params["url"]}
        if method == "read":
            return {"url": "https://example.com/", "title": "Example Domain", "text": "body"}
        if method == "tabs":
            return {"tabs": [], "limit": 8}
        if method == "owner_recover":
            return {"ownership_version": 1, "state": "owned", "tab": "bridge:5:capability"}
        if method == "owner_retain":
            return {"state": "retained"}
        if method == "owner_release":
            return {"state": "closed"}
        if method == "owner_finish":
            return {"state": "closed"}
        return {}


@pytest.fixture
def ui_client(monkeypatch: pytest.MonkeyPatch) -> FakeUiClient:
    fake = FakeUiClient()
    from local_operator.ui_browser import backend as ui_backend

    monkeypatch.setattr(ui_backend, "UiHostClient", lambda root=None: fake)
    return fake


@pytest.fixture
def bridge_client(monkeypatch: pytest.MonkeyPatch) -> FakeBridgeClient:
    """Both the lane's and the dispatcher's daemon client, faked as one object.

    They are resolved independently — the lane through `resources.BridgeClient`
    (the module global the browser suite has always monkeypatched) and the
    dispatcher through `builtin`'s lazy import of the same name — so a test that
    patched only one would see an ownership verb on the fake and a real dial on
    the action.
    """
    fake = FakeBridgeClient()
    from local_operator.browser_bridge import backend as bridge_backend

    monkeypatch.setattr(resources, "BridgeClient", lambda *args, **kwargs: fake)
    monkeypatch.setattr(bridge_backend, "BridgeClient", lambda *args, **kwargs: fake)
    return fake


def _ui_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """A host with the desktop app reachable and nothing else."""
    _hosts(monkeypatch, ui=True, bridge=False)

    async def ui_reachable(classified: tuple[Any, Any] | None = None) -> bool:
        return True

    async def bridge_unreachable(classified: tuple[Any, Any] | None = None) -> bool:
        return False

    monkeypatch.setattr(builtin, "ui_browser_reachable", ui_reachable)
    monkeypatch.setattr(builtin, "ui_browser_available", lambda: True)
    monkeypatch.setattr(builtin, "bridge_browser_reachable", bridge_unreachable)
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: bool(False))


def _hosts(monkeypatch: pytest.MonkeyPatch, *, ui: bool, bridge: bool) -> None:
    """Availability, stated per host, for the two gates that consult it.

    `_ownership_lane_host`'s probes and the dispatcher's own availability reads
    are separate code paths over the same two questions, so both are supplied:
    a test that set one and not the other would leave the decision half-faked.
    """

    from local_operator.browser_bridge import state as bridge_state
    from local_operator.ui_browser import state as ui_state

    # A reachable successful transport has a record. Always-unknown fake peer
    # identity hid the pre-action refresh and could resurrect closed fake tabs.
    ui_host = (
        ui_state.UiHostState(pid=2, port=52133, session_key="k" * 32, app_version="0.21.0", proto=1)
        if ui
        else None
    )
    bridge_host = (
        bridge_state.BridgeState(
            pid=2,
            port=52134,
            session_key="k" * 32,
            extension_version="1.31.0",
            proto=1,
            extension_connected=True,
        )
        if bridge
        else None
    )
    monkeypatch.setattr(ui_state, "read", lambda root=None: ui_host)
    monkeypatch.setattr(bridge_state, "read", lambda root=None: bridge_host)

    def reachable(answer: bool) -> Any:
        async def probe(classified: tuple[Any, Any] | None = None) -> bool:
            return answer

        return probe

    monkeypatch.setattr(builtin, "ui_browser_reachable", reachable(ui))
    monkeypatch.setattr(builtin, "bridge_browser_reachable", reachable(bridge))
    monkeypatch.setattr(builtin, "ui_browser_available", lambda: ui)
    monkeypatch.setattr(builtin, "bridge_browser_available", lambda: bridge)
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: False)


def _cmux_only(monkeypatch: pytest.MonkeyPatch) -> None:
    async def bridge_unreachable(classified: tuple[Any, Any] | None = None) -> bool:
        return False

    monkeypatch.setattr(builtin, "ui_browser_reachable", bridge_unreachable)
    monkeypatch.setattr(builtin, "bridge_browser_reachable", bridge_unreachable)
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: True)


def _context(tmp_path: Path, *, surface_id: str = "") -> tuple[ToolContext, BrowserResource]:
    resource = BrowserResource(tmp_path, tmp_path.name)
    surface = BrowserSurface()
    surface.surface_id = surface_id
    surface.resource = resource  # type: ignore[assignment]
    return ToolContext(browser=surface), resource


def _resumed_context(
    tmp_path: Path, *, surface_id: str, host: str
) -> tuple[ToolContext, BrowserResource]:
    """A session directory holding what a PREVIOUS run left behind, plus a fresh
    resource — which is the state a resumed session is in when the gate runs:
    `generation` empty, `record` unloaded, `state.surface_id` not yet adopted.

    Written through the writer's own API (`initialize` + `remember`) rather than
    as hand-written JSON, so the fixture cannot drift from the record the
    product actually produces.
    """
    previous = BrowserResource(tmp_path, tmp_path.name)
    previous.initialize()
    previous.remember(surface_id, host=host)
    context, resource = _context(tmp_path)
    assert resource.generation == "" and resource.record == {}, "not a resumed session"
    return context, resource


@pytest.mark.asyncio
async def test_the_lane_runs_on_a_ui_only_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ui_client: FakeUiClient
) -> None:
    """Asserted through `initialize()` having run, not through the result text.

    The gate's whole failure mode is that the lane never ran, so a test that
    checks the rendered sentence can pass while the lane is still skipped (the
    old code answered a `retain` with a screenshot SUCCESS message). The durable
    evidence that the lane ran is the generation it mints.
    """
    _ui_only(monkeypatch)
    context, resource = _context(tmp_path)
    assert resource.generation == ""

    result = await builtin.execute_browser(
        "t", {"action": "open", "url": "https://example.com"}, None, None, context
    )

    assert result.is_error is False, result.text
    assert resource.generation != "", "the ownership lane never ran"
    assert [method for method, _params in ui_client.calls][:2] == ["owner_recover", "open"]
    # The lane is entered through the UI host's transport, and nothing in this
    # session went looking for a daemon.
    assert any(method.startswith("owner_") for method, _ in ui_client.calls)


@pytest.mark.asyncio
async def test_a_ui_only_session_never_constructs_a_bridge_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ui_client: FakeUiClient
) -> None:
    """The lane must not require a daemon the surface does not use.

    `resources.BridgeClient` is monkeypatched to raise, so any attempt to dial
    the daemon from a UI-only session is a hard failure rather than a silently
    unreachable one.
    """
    _ui_only(monkeypatch)

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("a UI-only session constructed a BridgeClient")

    monkeypatch.setattr(resources, "BridgeClient", forbidden)
    context, resource = _context(tmp_path)
    result = await builtin.execute_browser(
        "t", {"action": "open", "url": "https://example.com"}, None, None, context
    )
    assert result.is_error is False, result.text
    assert resource.generation != ""


@pytest.mark.asyncio
async def test_no_ownership_host_still_skips_the_lane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ui_client: FakeUiClient
) -> None:
    """A cmux-only host has no ownership lane, and must not invent one."""
    _cmux_only(monkeypatch)
    context, resource = _context(tmp_path)
    monkeypatch.setattr(
        builtin,
        "_browser_open",
        lambda *_a, **_k: asyncio.sleep(0, result=builtin._text("t", "b", "cmux")),
    )
    result = await builtin.execute_browser(
        "t", {"action": "open", "url": "https://example.com"}, None, None, context
    )
    assert result.text == "cmux"
    assert resource.generation == ""
    assert ui_client.calls == []


@pytest.mark.asyncio
async def test_an_ownership_action_with_no_ownership_host_is_a_typed_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cmux fall-through repro: `retain` must never become a screenshot.

    Before the fix, a cmux-pinned surface took the lane-skipping exit, `retain`
    matched none of the dispatcher's branches, and the request fell through to
    `_browser_screenshot`: the model asked to hold a tab open and read
    "Screenshot of … saved to …" with a PNG on disk.
    """
    _cmux_only(monkeypatch)
    context, _resource = _context(tmp_path, surface_id="surface:7")

    for action in ("recover", "retain", "release"):
        result = await builtin.execute_browser(
            "t", {"action": action, "text": "pending login"}, None, None, context
        )
        assert result.is_error, f"{action} was not refused: {result.text}"
        assert (result.details or {}).get("error_code") == "ownership_unavailable"
        assert "Screenshot" not in result.text
        assert "cmux owns" in result.text


@pytest.mark.asyncio
async def test_an_ownership_action_with_no_surface_at_all_is_also_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _cmux_only(monkeypatch)
    context, _resource = _context(tmp_path)
    result = await builtin.execute_browser("t", {"action": "retain"}, None, None, context)
    assert result.is_error
    assert (result.details or {}).get("error_code") == "ownership_unavailable"
    # Not the generic "no browser surface open" answer: the agent needs to know
    # that opening a surface is what makes this verb available, and on WHICH host.
    assert "no ownership host is reachable" in result.text


@pytest.mark.asyncio
async def test_a_ui_record_selects_the_ui_client_and_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ui_client: FakeUiClient
) -> None:
    """The record's `host` is the whole selection: client AND discovery read."""
    from local_operator.browser_bridge import state as bridge_state
    from local_operator.ui_browser import state as ui_state

    resource = BrowserResource(tmp_path, tmp_path.name)
    resource.initialize()
    resource.remember("ui:100:capability", host="ui")
    assert resource.record["host"] == "ui"

    # The discovery read must come from the UI host's record. Given a bridge
    # record that claims an old extension, the UI host must NOT be read as a
    # pre-ownership peer: the floor is an extension version, and comparing an app
    # version to it is a category error.
    bridge_state.publish(
        bridge_state.BridgeState(
            pid=1,
            port=4099,
            session_key="k" * 32,
            proto=1,
            extension_connected=True,
            extension_version="0.1.4",
        ),
        tmp_path,
    )
    monkeypatch.setattr(
        ui_state,
        "read",
        lambda root=None: ui_state.UiHostState(
            pid=2, port=52133, session_key="k" * 32, proto=1, app_version="0.21.0"
        ),
    )

    resumed = BrowserResource(tmp_path, tmp_path.name)
    resumed.initialize()
    assert resumed.host == ""  # the prefix/record decides, not the constructor
    assert resumed._lane().name == resources.HOST_UI
    assert resumed._peer_identity() == ("0.21.0", 1)
    assert resumed._peer_is_pre_ownership() is False
    # None here means "must ask", never "unavailable": the verdict is learned by
    # the probe, and on the UI host the probe goes to the UI host.
    assert resumed.ownership_mode() is None
    await resumed.recover()
    assert resumed.ownership_mode() is True
    assert [method for method, _ in ui_client.calls] == ["owner_recover"]


@pytest.mark.asyncio
async def test_same_version_ui_instance_change_invalidates_verdict_not_durable_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ui_client: FakeUiClient
) -> None:
    from local_operator.ui_browser import state as ui_state

    current = ui_state.UiHostState(
        pid=2,
        port=52133,
        session_key="k" * 32,
        proto=1,
        app_version="0.21.0",
        instance_id="instance-a",
    )
    monkeypatch.setattr(ui_state, "read", lambda root=None: current)
    resource = BrowserResource(tmp_path, tmp_path.name)
    resource.initialize()
    resource.remember("ui:100:capability", host="ui")
    await resource.recover()
    assert resource.ownership_mode() is True
    assert resource._peer_identity() == ("0.21.0", 1, "instance-a")
    durable = resource._load()
    current = current.model_copy(update={"instance_id": "instance-b"})
    assert resource.ownership_mode() is None, "same-version restarts must re-probe ownership"
    assert (
        resource._load() == durable
    ), "discovery identity never rewrites durable session ownership"
    await resource.recover()
    assert [method for method, _ in ui_client.calls] == ["owner_recover", "owner_recover"]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["ordinary-action", "finish"])
async def test_public_action_and_finish_refresh_recovery_without_manual_recover(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ui_client: FakeUiClient, operation: str
) -> None:
    from local_operator.ui_browser import state as ui_state

    _ui_only(monkeypatch)
    current = ui_state.UiHostState(
        pid=2,
        port=52133,
        session_key="k" * 32,
        app_version="0.21.0",
        proto=1,
        instance_id="instance-a",
    )
    monkeypatch.setattr(ui_state, "read", lambda root=None: current)
    context, resource = _context(tmp_path)
    opened = await builtin.execute_browser(
        "open", {"action": "open", "url": "https://example.com"}, None, None, context
    )
    assert not opened.is_error
    assert resource.recovered
    proof = resource.params()["owner_proof"]
    ui_client.calls.clear()
    current = current.model_copy(update={"instance_id": "instance-b"})
    ui_client.recover_state = "unresolved"
    if operation == "ordinary-action":
        result = await builtin.execute_browser("read", {"action": "read"}, None, None, context)
        assert result.is_error, "ordinary read must learn the new ledger has no adopted tab"
        assert context.browser is not None and context.browser.surface_id == ""
        assert [method for method, _ in ui_client.calls] == ["owner_recover"]
    else:
        finished = await resource.finish(resource.generation, "completed")
        assert finished.state == "unresolved"
        assert [method for method, _ in ui_client.calls] == [
            "owner_recover"
        ], "unproven replacement must never receive destructive finish"
    assert resource.params()["owner_proof"] == proof
    assert ui_client.calls[0][1]["owner_proof"] == proof


@pytest.mark.asyncio
async def test_unresolvable_discovery_does_not_churn_the_learned_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ui_client: FakeUiClient
) -> None:
    """A→None→A must NOT invalidate: an unresolvable record is not a new peer.

    `_peer_identity()` returns ``None`` for "cannot tell" (record gone or
    unreadable). Treating that as a change would clear the learned verdict and
    force a needless `owner_recover` on every flap. The guard is
    `peer is not None and peer != self._ownership_peer`.
    """
    from local_operator.ui_browser import state as ui_state

    _ui_only(monkeypatch)
    a = ui_state.UiHostState(
        pid=2, port=52133, session_key="k" * 32, app_version="0.21.0", proto=1, instance_id="a"
    )
    monkeypatch.setattr(ui_state, "read", lambda root=None: a)
    resource = BrowserResource(tmp_path, tmp_path.name)
    resource.initialize()
    resource.remember("ui:100:capability", host="ui")
    await resource.recover()
    assert resource.ownership_mode() is True
    ui_client.calls.clear()
    # The record becomes unreadable: discovery answers None.
    monkeypatch.setattr(ui_state, "read", lambda root=None: None)
    assert resource.ownership_mode() is True, "an unknown discovery is not a changed peer"
    await resource.ensure_recovered()
    assert ui_client.calls == [], "A→None must not force a re-recovery"
    # And it comes back: still the learned peer, still no churn.
    monkeypatch.setattr(ui_state, "read", lambda root=None: a)
    assert resource.ownership_mode() is True
    await resource.ensure_recovered()
    assert ui_client.calls == [], "A→None→A must not churn"


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["ordinary-action", "finish"])
async def test_changed_ledger_success_recovery_runs_before_action_and_keeps_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ui_client: FakeUiClient, operation: str
) -> None:
    """CR1 success half: a NEW incarnation that DOES hold our tab is adopted by
    the automatic recovery, before the ordinary action or the finalizer runs, and
    the durable proof is unchanged across it.

    The companion test above pins the UNPROVEN replacement; this one pins the
    proven one, so a fix that only reconciles the failure case cannot pass both.
    The recovery must be owed by `ensure_recovered` on the public entry/finish
    path — never a manual `owner_recover` the caller has to remember.
    """
    from local_operator.ui_browser import state as ui_state

    _ui_only(monkeypatch)
    current = ui_state.UiHostState(
        pid=2,
        port=52133,
        session_key="k" * 32,
        app_version="0.21.0",
        proto=1,
        instance_id="instance-a",
    )
    monkeypatch.setattr(ui_state, "read", lambda root=None: current)
    context, resource = _context(tmp_path)
    opened = await builtin.execute_browser(
        "open", {"action": "open", "url": "https://example.com"}, None, None, context
    )
    assert not opened.is_error
    proof = resource.params()["owner_proof"]
    ui_client.calls.clear()
    # A new incarnation of the ledger, still holding the tab (the default
    # `recover_state="owned"`), so recovery SUCCEEDS.
    current = current.model_copy(update={"instance_id": "instance-b"})
    if operation == "ordinary-action":
        result = await builtin.execute_browser("read", {"action": "read"}, None, None, context)
        assert not result.is_error, "a proven replacement must be adopted automatically"
    else:
        finished = await resource.finish(resource.generation, "completed")
        assert finished.state == "closed", "a proven replacement may receive the finalizer"
    # The recovery is owed FIRST and by the public entry — never a manual
    # `owner_recover` the caller has to remember. For an ordinary read the
    # action's own verb follows, so this asserts the ORDER, not the count.
    assert [method for method, _ in ui_client.calls][0] == "owner_recover"
    assert resource.params()["owner_proof"] == proof
    assert ui_client.calls[0][1]["owner_proof"] == proof


@pytest.mark.asyncio
@pytest.mark.parametrize("resumed", [False, True])
async def test_failed_explicit_extension_open_keeps_public_approvals_on_extension(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ui_client: FakeUiClient,
    bridge_client: FakeBridgeClient,
    resumed: bool,
) -> None:
    """No seeded pin or patched recovery: the public denied navigation must
    itself save the transport used by all three surface-free approval actions.
    Reconstruction tests that the intent survives a new context of the same SID.
    Only the transport is fake; selection, resource, recovery and persistence are
    production paths. No user/foreign tab is adopted or authority invented.
    """
    from local_operator.browser_bridge.backend import BridgeError
    from local_operator.browser_bridge.protocol import ErrorCode

    _hosts(monkeypatch, ui=True, bridge=True)
    permitted = False

    async def transport(method: str, params: dict[str, Any]) -> dict[str, Any]:
        bridge_client.calls.append((method, params))
        if method == "owner_recover":
            return {"ownership_version": 1, "state": "closed", "tab": ""}
        if method == "open":
            assert "tab" not in params
            if permitted:
                return {"tab": "bridge:5:permitted-capability", "url": params["url"]}
            raise BridgeError(
                ErrorCode.ORIGIN_NOT_ALLOWED,
                "site not allowed",
                {
                    "origin": "https://example.test",
                    "url": params["url"],
                },
            )
        if method in ("request_access", "await_access", "cancel_access"):
            answer = "cancelled" if method == "cancel_access" else "allowed"
            return {"origin": "https://example.test", "state": answer}
        return {"state": "closed"}

    monkeypatch.setattr(bridge_client, "call", transport)
    context, resource = _context(tmp_path)
    denied = await builtin.execute_browser(
        "deny",
        {
            "action": "open",
            "backend": "extension",
            "url": "https://example.test",
        },
        None,
        None,
        context,
    )
    assert denied.is_error
    assert (denied.details or {}).get("error_code") == "origin_not_allowed"
    assert context.browser is not None and context.browser.surface_id == ""
    assert ui_client.calls == [], "explicit extension must also select the ownership lane"
    proof = resource.params()["owner_proof"]
    if resumed:
        context, resource = _context(tmp_path)
    for action in ("request_access", "await_access", "cancel_access"):
        result = await builtin.execute_browser(
            action,
            {
                "action": action,
                "url": "https://example.test",
                "timeout_s": 1,
            },
            None,
            None,
            context,
        )
        assert not result.is_error, result.text
        expected = "cancelled" if action == "cancel_access" else "allowed"
        assert (result.details or {}).get("state") == expected
        assert context.browser is not None and context.browser.surface_id == ""
    assert ui_client.calls == []
    assert [
        method
        for method, _ in bridge_client.calls
        if method
        in (
            "request_access",
            "await_access",
            "cancel_access",
        )
    ] == ["request_access", "await_access", "cancel_access"]
    assert resource.params()["owner_proof"] == proof
    assert "pending_access_host" not in resource.record, "cancel discharges routing intent"
    assert "pending_access_host" not in resource._load(), "cancel also persists the removal"

    # Recreate the intent through another actual denied navigation, not by
    # seeding a record. A later permitted open must consume the routing marker:
    # its minted capability now names the host, so stale approval intent cannot
    # govern a future surface-free flow.
    denied_again = await builtin.execute_browser(
        "deny-again",
        {
            "action": "open",
            "backend": "extension",
            "url": "https://example.test",
        },
        None,
        None,
        context,
    )
    assert denied_again.is_error
    assert (denied_again.details or {}).get("error_code") == "origin_not_allowed"
    assert resource.record.get("pending_access_host") == "bridge"
    permitted = True
    opened = await builtin.execute_browser(
        "permitted-open",
        {
            "action": "open",
            "backend": "extension",
            "url": "https://example.test",
        },
        None,
        None,
        context,
    )
    assert not opened.is_error, opened.text
    assert context.browser is not None
    assert context.browser.surface_id == "bridge:5:permitted-capability"
    assert "pending_access_host" not in resource.record
    assert "pending_access_host" not in resource._load()
    assert resource.params()["owner_proof"] == proof
    assert ui_client.calls == []


@pytest.mark.asyncio
async def test_same_sid_new_lease_registers_via_predecessor_and_fences_the_old(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ui_client: FakeUiClient
) -> None:
    """M2 (audit-corrected): SAME durable SID + a NEW valid execution lease.

    The production shape is `SessionFactory` acquiring the lease BEFORE it
    constructs `Session`, which reconstructs `BrowserResource` over the same
    directory — so the resource that resumes reads the NEW lease and must
    register as a legitimate successor, not a foreigner. This builds exactly
    that at the documented construction boundary (no whole `Session` fixture):

      * run 1 — acquire lease G1, open a tab, finalize (record carries terminal).
      * run 2 — acquire a NEW lease G2 over the same SID, reconstruct the
        resource. `initialize()`'s fence compares the SAME new lease, so it does
        NOT fire; `generation` becomes G2 while G1 survives as
        `previous_generation`/`bridge_generations`, and `params()` ships both.
    The proof/allocation are preserved, and the OLD owner is fenced by the lease.
    """
    from local_operator.session_lease import acquire_session_lease

    directory = tmp_path / "sessions" / "synthetic-sid"
    directory.mkdir(parents=True)

    lease1 = acquire_session_lease(directory)
    first = BrowserResource(directory, "synthetic-sid")
    first.initialize()
    assert first.generation == lease1.generation
    proof = first.params()["owner_proof"]
    allocation = first.params()["allocation_id"]
    first.remember("ui:100:capability", host="ui")
    # Finalize so the record carries a durable obligation the successor inherits.
    first.record["terminal"] = "completed"
    first._save()
    # Run 1 ends. A resume acquires a genuinely NEW lease (in production the
    # predecessor's claim is either released on a clean exit or proven dead and
    # recovered); the SID stays the same, which is the whole point.
    lease1.release()
    assert not (directory / ".execution-lease").exists()

    lease2 = acquire_session_lease(directory)
    assert lease2.generation != lease1.generation
    second = BrowserResource(directory, "synthetic-sid")
    second.initialize()  # fence compares the SAME new lease: must NOT raise
    assert second.generation == lease2.generation
    assert second.previous_generation == lease1.generation
    assert lease1.generation in second.params()["previous_generations"]
    assert second.params()["owner_proof"] == proof
    assert second.params()["allocation_id"] == allocation

    # The successor RECONCILES over the wire, and the request it sends proves it:
    # current generation G2, predecessor G1 declared, and the SAME proof and
    # allocation preserved. `second.recover()` is what the ordinary entry calls
    # (`ensure_recovered`), so this exercises the real reconciliation path rather
    # than only the record's fields.
    ui_client.calls.clear()
    recovered = await second.recover()
    assert recovered.get("state") == "owned", "the new lease is accepted as a successor"
    sent = [params for method, params in ui_client.calls if method == "owner_recover"]
    assert len(sent) == 1
    assert sent[0]["owner_generation"] == lease2.generation
    assert sent[0]["previous_generation"] == lease1.generation
    assert lease1.generation in sent[0]["previous_generations"]
    assert sent[0]["owner_proof"] == proof
    assert sent[0]["allocation_id"] == allocation

    # The OLD owner is fenced: its identity came from the old lease, so a
    # generation check against the live claim now refuses it.
    with pytest.raises(BrowserOwnershipError):
        first.assert_current()
    lease2.release()


@pytest.mark.asyncio
async def test_owner_refused_stale_and_foreign_are_told_apart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The typed refusal discriminator the GCP neighbours asked for.

    Both a stale-generation refusal and a foreign-proof refusal arrive as
    `owner_refused`; the additive `data.refusal` key is what lets the session
    tell them apart WITHOUT reading the message. A peer that predates the key
    omits it and keeps the prior bare-refusal behaviour.
    """
    from local_operator.browser_bridge.backend import BridgeError, ErrorCode
    from local_operator.browser_bridge.resources import BrowserOwnershipError

    class _Refusing:
        host = "ui"

        def __init__(self, refusal: str | None) -> None:
            self.refusal = refusal

        async def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            data = {} if self.refusal is None else {"refusal": self.refusal}
            raise BridgeError(ErrorCode.OWNER_REFUSED, "browser owner generation is stale", data)

    for refusal, expect in (("stale", "stale"), ("foreign", "different owner")):
        resource = BrowserResource(tmp_path / refusal, f"session:{refusal}")
        resource.initialize()
        monkeypatch.setattr(resource, "client", lambda r=_Refusing(refusal): r)
        with pytest.raises(BrowserOwnershipError, match=expect):
            await resource.recover()
    # An older peer that omits the key keeps the bare refusal (still an error).
    resource = BrowserResource(tmp_path / "bare", "session:bare")
    resource.initialize()
    monkeypatch.setattr(resource, "client", lambda: _Refusing(None))
    with pytest.raises(BridgeError):
        await resource.recover()
    # An UNKNOWN discriminator (a future producer, or a mis-shaped payload) must
    # NOT be swallowed as "handled": it stays the bare refusal so an unrecognised
    # value can never silently change which recovery branch runs.
    resource = BrowserResource(tmp_path / "unknown", "session:unknown")
    resource.initialize()
    monkeypatch.setattr(resource, "client", lambda: _Refusing("someday-key"))
    with pytest.raises(BridgeError):
        await resource.recover()


def test_a_legacy_record_without_a_host_falls_back_to_the_bridge(tmp_path: Path) -> None:
    """Fail-safe, not fail-open: a record written before `host` existed names the
    bridge, which is where that session was talking."""
    resource = BrowserResource(tmp_path, tmp_path.name)
    resource.path.write_text('{"session_id": "%s", "generation": "g"}' % tmp_path.name)
    resource.initialize()
    assert resource.host == ""
    assert resource._lane().name == resources.HOST_BRIDGE


@pytest.mark.asyncio
@pytest.mark.parametrize(("ui_up", "bridge_up"), [(True, False), (True, True)])
async def test_a_resumed_bridge_record_keeps_the_bridge_lane_with_the_app_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ui_client: FakeUiClient,
    bridge_client: FakeBridgeClient,
    ui_up: bool,
    bridge_up: bool,
) -> None:
    """R1, direction 1: the record wins over the availability order.

    Asserted on the WIRE, not on the decision: the record names the bridge, the
    app is the host a probe would pick, and the `owner_recover` that proves the
    lane ran must arrive at the daemon. The old gate ran the probes first (UI
    first), `select_host` set `self.host`, `_lane()` prefers `self.host` over
    `record["host"]`, and a resumed `bridge` session was moved onto the app —
    which does not own its tab — for the rest of its life.

    Both availability shapes are covered because they fail for the same reason
    and are tempting in different ways: with only the app up the daemon looks
    unreachable, and with both up it looks like a free choice.
    """
    _hosts(monkeypatch, ui=ui_up, bridge=bridge_up)
    context, resource = _resumed_context(tmp_path, surface_id="bridge:5:nonce", host="bridge")

    result = await builtin.execute_browser("t", {"action": "read"}, None, None, context)

    assert result.is_error is False, result.text
    assert resource.host == "bridge", "the lane moved off the record's host"
    assert [method for method, _p in bridge_client.calls][:1] == ["owner_recover"]
    assert ui_client.calls == []


@pytest.mark.asyncio
async def test_a_resumed_ui_record_keeps_the_apps_lane_with_the_app_down(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ui_client: FakeUiClient,
    bridge_client: FakeBridgeClient,
) -> None:
    """R1, direction 2 — §10.5's consequence 2, in mirror image.

    The record names the app and the app is down, so this is the case where
    "pick the host that is up" looks harmless and is not: the tab belongs to the
    app, the daemon cannot reconcile it, and moving the session there is exactly
    the defect the design's consequence 2 names (`owner_recover` going to the
    bridge for a `ui:` surface). The honest answer is the app's own refusal,
    which is what the pinned-handle path already does with the app down.
    """
    _hosts(monkeypatch, ui=False, bridge=True)
    context, resource = _resumed_context(tmp_path, surface_id="ui:100:nonce", host="ui")

    result = await builtin.execute_browser("t", {"action": "read"}, None, None, context)

    assert result.is_error is False, result.text
    assert resource.host == "ui", "the lane moved off the record's host"
    assert [method for method, _p in ui_client.calls][:1] == ["owner_recover"]
    assert bridge_client.calls == []


@pytest.mark.asyncio
async def test_a_resumed_legacy_record_keeps_the_bridge_lane_with_the_app_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ui_client: FakeUiClient,
    bridge_client: FakeBridgeClient,
) -> None:
    """The same gate, for the record that names no host at all.

    §10.5's fail-safe clause is about exactly this record ("an old record
    behaves exactly as it does today"), and its host is spelled nowhere but its
    handle. `_lane()` reads that as the bridge, so the gate has to as well —
    otherwise the two disagree and the probe silently settles it in favour of
    the app.
    """
    _hosts(monkeypatch, ui=True, bridge=False)
    # `host=""` writes a handle and NO host field, which is byte-for-byte the
    # record shape a pre-`host` run left behind.
    context, resource = _resumed_context(tmp_path, surface_id="bridge:5:nonce", host="")
    assert "host" not in json.loads(resource.path.read_text())

    result = await builtin.execute_browser("t", {"action": "read"}, None, None, context)

    assert result.is_error is False, result.text
    assert resource.host == "bridge"
    assert [method for method, _p in bridge_client.calls][:1] == ["owner_recover"]
    assert ui_client.calls == []


def _disagreeing_context(tmp_path: Path) -> tuple[ToolContext, BrowserResource]:
    """A record whose field names one host and whose handle names another.

    THIS SHAPE IS NOW LEGACY, and that is the point: the writer used to produce
    it (`select_host` kept an ESTABLISHED lane's host, so `remember("ui:…",
    host="ui")` recorded `host: "bridge"` beside a `ui:` handle), and R6/R7
    makes the writer keep the two consistent instead. The disagreement is still
    reachable — a record written by an earlier client, or one left by a lane
    move that never completed — so the READER's precedence stays under test
    rather than the case being deleted along with the bug that produced it.

    The app is DOWN here on purpose: the field, the availability order and the
    probe all point at the daemon, and only the handle points at the tab.
    """
    previous = BrowserResource(tmp_path, tmp_path.name)
    previous.initialize()
    previous.select_host("bridge")
    previous.remember("ui:100:capability", host="ui")
    # Force the legacy shape through the same writer, then assert it landed.
    previous.record["host"] = "bridge"
    previous._save()
    record = json.loads(previous.path.read_text())
    assert (record["host"], record["surface_id"]) == ("bridge", "ui:100:capability")
    return _context(tmp_path)


async def _obligation_without_a_handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ToolContext, BrowserResource]:
    """A record left behind by a recovery that could not prove its handle.

    `recover()` moves a handle the host would not confirm into
    `unresolved_surface_id` and clears `surface_id`, so the surviving evidence
    of which host holds the tab is the `host` field alone. Driven through the
    product's own writer — `remember` plus a recovery whose host answers
    `unresolved` — rather than by writing JSON, so the shape cannot drift from
    what the product produces.
    """
    from local_operator.ui_browser import backend as ui_backend

    class UnresolvedUiClient(FakeUiClient):
        async def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            if method == "owner_recover":
                self.calls.append((method, params))
                return {"ownership_version": 1, "state": "unresolved", "tab": ""}
            return await super().call(method, params)

    previous = BrowserResource(tmp_path, tmp_path.name)
    previous.initialize()
    previous.remember("ui:100:capability", host="ui")

    monkeypatch.setattr(ui_backend, "UiHostClient", lambda root=None: UnresolvedUiClient())
    resumed = BrowserResource(tmp_path, tmp_path.name)
    resumed.initialize()
    await resumed.recover()

    record = json.loads(resumed.path.read_text())
    assert record["host"] == "ui" and record["unresolved_surface_id"]
    assert not record["surface_id"]
    return _context(tmp_path)


@pytest.mark.asyncio
async def test_the_handle_decides_when_it_disagrees_with_the_record_host(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ui_client: FakeUiClient,
    bridge_client: FakeBridgeClient,
) -> None:
    """R6: the field and the handle can name different hosts, and the HANDLE wins.

    Read field-first, the lane went to the daemon for a tab that lives in the
    app: the daemon answers `unresolved` with no tab, `recover()` moves the
    handle into `unresolved_surface_id`, and the action then fails "no browser
    surface open" with the app's live tab stranded — where the pre-fix code,
    which had no field to read, self-healed on the handle. The app is DOWN here
    on purpose: the field, the availability order and the probe all point at the
    daemon, and only the handle points at the tab.
    """
    _hosts(monkeypatch, ui=False, bridge=True)
    context, resource = _disagreeing_context(tmp_path)

    result = await builtin.execute_browser("t", {"action": "read"}, None, None, context)

    assert result.is_error is False, result.text
    assert resource.host == "ui", "the lane followed the field, not the handle"
    assert [method for method, _p in ui_client.calls][:1] == ["owner_recover"]
    assert bridge_client.calls == []


@pytest.mark.asyncio
async def test_a_handle_less_record_that_owes_a_reconciliation_keeps_its_host(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ui_client: FakeUiClient,
    bridge_client: FakeBridgeClient,
) -> None:
    """R7, direction 1: with no handle, an obligation in flight is what pins.

    This is the shape `recover()` leaves when it cannot prove a handle, and the
    session still owes `owner_*` an answer about the tab it lost track of. The
    field is the only thing naming the host holding it, so it must outrank the
    probes even with the app down — the honest refusal is the point, and a
    probe-decided lane would move the obligation onto a host that never held
    the tab.
    """
    _hosts(monkeypatch, ui=False, bridge=True)
    context, resource = await _obligation_without_a_handle(tmp_path, monkeypatch)
    # Back to the fixture's host for the gate call: the recovery above replaced
    # the client factory, and the assertion is about the WIRE this session's
    # lane speaks on, not about a fresh fake's own call list.
    from local_operator.ui_browser import backend as ui_backend

    monkeypatch.setattr(ui_backend, "UiHostClient", lambda root=None: ui_client)

    await builtin.execute_browser("t", {"action": "read"}, None, None, context)

    assert resource.host == "ui", "the obligation was moved off its host"
    assert [method for method, _p in ui_client.calls][:1] == ["owner_recover"]
    assert bridge_client.calls == []


@pytest.mark.asyncio
async def test_a_handle_less_record_without_an_obligation_lets_availability_decide(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ui_client: FakeUiClient,
    bridge_client: FakeBridgeClient,
) -> None:
    """R7, direction 2: a settled record must not govern the next `open`.

    `close` clears `surface_id` and deliberately leaves `host` behind, so this
    is what every session that has closed a tab carries. A fresh `open` has no
    transport to keep stable, and the documented cascade is availability-based:
    pinning to the field here makes a session whose app is down refuse to open
    at all while the extension that would serve it sits there running. The
    rewrite of the record is asserted too — that is what stops the field and the
    surface going on to contradict each other.
    """
    _hosts(monkeypatch, ui=False, bridge=True)
    previous = BrowserResource(tmp_path, tmp_path.name)
    previous.initialize()
    previous.remember("ui:100:capability", host="ui")
    previous.remember("")  # the close path: handle cleared, host kept
    assert json.loads(previous.path.read_text())["host"] == "ui"
    context, resource = _context(tmp_path)

    result = await builtin.execute_browser(
        "t", {"action": "open", "url": "https://example.com"}, None, None, context
    )

    assert result.is_error is False, result.text
    assert resource.host == "bridge", "a settled record pinned the fresh open"
    assert "open" in [method for method, _p in bridge_client.calls]
    assert ui_client.calls == []
    # The record follows the surface, so the contradiction R7 names cannot
    # survive the open that would have created it.
    record = json.loads(resource.path.read_text())
    assert record["host"] == "bridge" and record["surface_id"].startswith("bridge:")


@pytest.mark.asyncio
async def test_a_ui_host_serves_the_whole_flow_on_its_own_wire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ui_client: FakeUiClient
) -> None:
    """open -> read -> tabs -> retain -> release -> close, all on the UI wire.

    The capability matrix's claim is that the UI host is a superset of the
    bridge's wire, so the flow that matters is the ordinary one: nothing here may
    reach for a daemon, and the surface must stay `ui:`-prefixed throughout.
    """
    _ui_only(monkeypatch)
    context, resource = _context(tmp_path)
    seen: list[str] = []

    for args in (
        {"action": "open", "url": "https://example.com"},
        {"action": "read"},
        {"action": "tabs"},
        {"action": "retain", "text": "waiting on the login form"},
        {"action": "release"},
        {"action": "close"},
    ):
        result = await builtin.execute_browser("t", args, None, None, context)
        assert result.is_error is False, f"{args['action']}: {result.text}"
        seen.append(args["action"])
        holder = context.browser
        assert holder is not None
        if args["action"] != "close":
            assert holder.surface_id.startswith("ui:"), result.text

    assert seen == ["open", "read", "tabs", "retain", "release", "close"]
    assert [method for method, _params in ui_client.calls] == [
        "owner_recover",
        "open",
        "read",
        "tabs",
        "owner_retain",
        "owner_release",
        "close",
    ]
    # The surface's host is written down on the first successful open, so a
    # resumed session selects the same lane.
    assert json.loads(resource.path.read_text())["host"] == "ui"


def test_the_handle_prefixes_agree_with_the_record_spelling() -> None:
    """Two modules name the same hosts, deliberately, and must not drift.

    `builtin` cannot import these at module scope (the bridge package pulls in
    httpx, and `builtin` is imported on the CLI path for every session), so the
    literals are repeated. This test is what keeps the repetition honest: a
    handle prefix that disagreed with the record spelling would select the wrong
    lane for a resumed session.
    """
    from local_operator.browser_bridge.backend import HOST_EXTENSION, HOST_UI

    assert builtin.HOST_UI_PREFIX == resources.HOST_UI == HOST_UI == "ui"
    assert builtin.HOST_BRIDGE_PREFIX == resources.HOST_BRIDGE == "bridge"
    # The copy spelling is the one that differs for the bridge (prose says "the
    # extension"), and `_copy_host` is the single translation.
    assert builtin._copy_host("ui") == HOST_UI
    assert builtin._copy_host("bridge") == builtin._copy_host("") == HOST_EXTENSION


@pytest.mark.asyncio
async def test_a_same_object_lifecycle_moves_the_lane_to_the_host_that_allocated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ui_client: FakeUiClient,
    bridge_client: FakeBridgeClient,
) -> None:
    """R6/R7 + QA Q-5: the host that ALLOCATED the tab is the host that cleans it up.

    The lifecycle is the ordinary one, not a résumé: ONE `BrowserResource` object
    opens on the reachable host, closes that tab, and then opens again after the
    other host comes up. The defect this pins was in the second half — the fresh
    allocation was recorded (`surface_id` became a `ui:` handle and `pinned_host()`
    answered `ui`) while the LANE still said `bridge`, so `finish` sent
    `owner_finish` to the daemon, the app's tab stayed live, and the record lost
    the handle it was claiming to have cleaned up.

    Asserted on the wire rather than on the record alone: the exact methods each
    host received, the handle that came back, and the cleanup side effects
    (which host was told to finish, and that the OTHER host was told nothing).
    """
    # 1. Only the daemon is up: the first open lands on the extension.
    _hosts(monkeypatch, ui=False, bridge=True)
    context, resource = _context(tmp_path)
    first = await builtin.execute_browser(
        "t", {"action": "open", "url": "https://example.com"}, None, None, context
    )
    assert first.is_error is False, first.text
    assert resource.record["surface_id"].startswith("bridge:")
    assert resource.host == "bridge"
    # The handle the host RETURNED is the handle the call reports and the one the
    # record holds: a result that named a tab the record does not hold would send
    # the model to a surface it cannot address.
    assert first.details is not None
    assert first.details["surface_id"] == resource.record["surface_id"]
    assert [method for method, _p in bridge_client.calls] == ["owner_recover", "open"]

    # 2. Close it. The tab is gone, so nothing is in flight any more.
    closed = await builtin.execute_browser("t", {"action": "close"}, None, None, context)
    assert closed.is_error is False, closed.text
    assert resource.record["surface_id"] == ""
    assert resource.record["state"] == "closed"

    # 3. The app comes up. A FRESH open must use it, and the lane must follow the
    #    allocation it actually got.
    _hosts(monkeypatch, ui=True, bridge=True)
    second = await builtin.execute_browser(
        "t", {"action": "open", "url": "https://example.com"}, None, None, context
    )
    assert second.is_error is False, second.text
    assert resource.record["surface_id"].startswith("ui:")
    assert resource.record["host"] == "ui"
    assert resource.pinned_host() == "ui"
    assert resource._lane().name == "ui", "the lane still names the closed tab's host"
    assert second.details is not None
    assert second.details["surface_id"] == resource.record["surface_id"]

    # 4. Finalization reaches the host HOLDING the tab, and only that host.
    bridge_before = list(bridge_client.calls)
    ui_before = list(ui_client.calls)
    result = await resource.finish(resource.generation, "completed")

    assert result.state == "closed"
    assert [method for method, _p in ui_client.calls[len(ui_before) :]] == ["owner_finish"]
    assert (
        bridge_client.calls == bridge_before
    ), "the previous host must not receive the new allocation's cleanup"
    # The handle is cleared only by the host that owned it acknowledging the finish.
    assert resource.record["surface_id"] == ""
    assert resource.record["state"] == "closed"


@pytest.mark.asyncio
async def test_the_lane_follows_the_allocation_in_the_other_direction_too(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ui_client: FakeUiClient,
    bridge_client: FakeBridgeClient,
) -> None:
    """The mirror of the lifecycle above: the app first, then the extension.

    Same one-object lifecycle in the OTHER transition direction, because the rule
    is symmetric only where something proves it is: a `ui:` allocation closes, the
    app goes away, the extension comes up, and the fresh allocation has to move a
    lane that was left naming the app. Every assertion the forward test makes is
    made here in mirror — the field and the handle both end up naming the
    extension, `finish` reaches the host holding the tab, and the host that no
    longer holds anything hears NOTHING.

    The two directions are not each other's proof: a rule that pinned the lane to
    the record's `host` field would pass one of them and fail the other, which is
    exactly the half-fix review R6/R7 sent back.
    """
    # 1. Only the app is up: the first open lands there.
    _hosts(monkeypatch, ui=True, bridge=False)
    context, resource = _context(tmp_path)
    first = await builtin.execute_browser(
        "t", {"action": "open", "url": "https://example.com"}, None, None, context
    )
    assert first.is_error is False, first.text
    assert resource.record["surface_id"].startswith("ui:")
    assert resource.host == "ui"
    assert first.details is not None
    assert first.details["surface_id"] == resource.record["surface_id"]
    assert [method for method, _p in ui_client.calls] == ["owner_recover", "open"]

    # 2. Close it: nothing is in flight, so nothing pins the lane any more.
    closed = await builtin.execute_browser("t", {"action": "close"}, None, None, context)
    assert closed.is_error is False, closed.text
    assert resource.record["surface_id"] == ""
    assert resource.record["state"] == "closed"

    # 3. The app goes away and the daemon is up. The fresh open lands on the
    #    extension, and the lane must follow the allocation it actually got.
    _hosts(monkeypatch, ui=False, bridge=True)
    second = await builtin.execute_browser(
        "t", {"action": "open", "url": "https://example.com"}, None, None, context
    )
    assert second.is_error is False, second.text
    assert resource.record["surface_id"].startswith("bridge:")
    assert resource.record["host"] == "bridge"
    assert resource.pinned_host() == "bridge"
    assert resource._lane().name == "bridge", "the lane still names the app's closed tab"
    assert second.details is not None
    assert second.details["surface_id"] == resource.record["surface_id"]

    # 4. Finalization reaches the host HOLDING the tab, and only that host.
    ui_before = list(ui_client.calls)
    bridge_before = list(bridge_client.calls)
    result = await resource.finish(resource.generation, "completed")

    assert result.state == "closed"
    assert [method for method, _p in bridge_client.calls[len(bridge_before) :]] == ["owner_finish"]
    assert (
        ui_client.calls == ui_before
    ), "the host with nothing in flight must not receive the new allocation's cleanup"
    assert resource.record["surface_id"] == ""


@pytest.mark.asyncio
async def test_adoption_forwards_the_user_handed_handle_and_refuses_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ui_client: FakeUiClient
) -> None:
    """The open-tab adoption field: create(url) and adopt(handle) stay distinct.

    A tab the user handed over is neither "resume our own handle" nor "create a
    new tab": the capability exists already and already belongs to this session.
    These assertions are the whole contract — the handle is forwarded UNCHANGED
    (Python has no authority to mint or rewrite one), an adoption carries no URL
    and must not invent one, and every other spelling is refused before it can
    reach a host.
    """
    _ui_only(monkeypatch)
    handle = "ui:7:aaaaaaaabbbbccccddddeeeeffff0000"
    context, resource = _context(tmp_path)
    # A FRESH session: nothing is held yet, which is the state a handed-over tab
    # arrives in (the host answers `unresolved` for a recovery of no tab).
    ui_client.recover_state = "unresolved"

    adopted = await builtin.execute_browser(
        "t", {"action": "open", "tab": handle}, None, None, context
    )
    assert adopted.is_error is False, adopted.text
    opened = [params for method, params in ui_client.calls if method == "open"]
    assert len(opened) == 1
    assert opened[0]["tab"] == handle, "the handle must travel through unchanged"
    assert opened[0]["url"] == "", "adoption must not invent a URL"
    assert opened[0]["owner_proof"], "adoption still carries this session's identity"
    # The record holds the capability the HOST returned, not the one that was
    # offered: the host is the authority on what this session now holds, and a
    # record keyed on the request instead of the answer is how a stale handle
    # gets remembered.
    assert resource.record["surface_id"] == "ui:100:aaaaaaaabbbbccccddddeeeeffff0000"

    # The same handle on any other action is refused, not silently ignored.
    other_action_context, _ = _context(tmp_path / "other")
    for action in ("read", "close", "goto", "tabs"):
        refused = await builtin.execute_browser(
            "t", {"action": action, "tab": handle}, None, None, other_action_context
        )
        assert refused.is_error is True, action
        assert "'tab' is only valid for 'open'" in refused.text, refused.text

    # A handle that is not a handle is refused before any host sees it.
    bad_context, _ = _context(tmp_path / "bad")
    for bad in ("--help", "../../etc/passwd", "ui:7:not a capability", "surface:1", "7"):
        refused = await builtin.execute_browser(
            "t", {"action": "open", "tab": bad}, None, None, bad_context
        )
        assert refused.is_error is True, bad
        assert "refusing 'tab' handle" in refused.text, refused.text


@pytest.mark.asyncio
async def test_adoption_is_refused_while_another_surface_is_held_and_on_cmux(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ui_client: FakeUiClient
) -> None:
    """Two ends of the same rule: one surface at a time, and a host that can adopt.

    Holding a surface and adopting a second one would leave the first tab's
    cleanup addressed to a lane that no longer names it, so it is refused rather
    than half-performed. And because cmux keeps no multi-surface registry, a
    handed-over tab cannot be adopted there at all — falling through to a new cmux
    surface would report a successful adoption of a tab nobody is driving.
    """
    _ui_only(monkeypatch)
    context, resource = _context(tmp_path)
    held = await builtin.execute_browser(
        "t", {"action": "open", "url": "https://example.com"}, None, None, context
    )
    assert held.is_error is False, held.text
    assert resource.record["surface_id"]

    refused = await builtin.execute_browser(
        "t",
        {"action": "open", "tab": "ui:9:bbbbbbbbccccddddeeeeffff00001111"},
        None,
        None,
        context,
    )
    assert refused.is_error is True
    assert "already driving a browser tab" in refused.text, refused.text

    # With no non-cmux host reachable, adoption is refused instead of becoming a
    # brand-new cmux surface.
    _cmux_only(monkeypatch)
    cmux_context, _ = _context(tmp_path / "cmux")
    unavailable = await builtin.execute_browser(
        "t",
        {"action": "open", "tab": "ui:9:bbbbbbbbccccddddeeeeffff00001111"},
        None,
        None,
        cmux_context,
    )
    assert unavailable.is_error is True
    assert (
        "needs the desktop app's browser tab or the browser extension" in unavailable.text
    ), unavailable.text


def _opens(client: Any) -> list[dict[str, Any]]:
    """The `open` params a fake host received, in order."""
    return [params for method, params in client.calls if method == "open"]


@pytest.mark.asyncio
async def test_adoption_routes_by_the_handles_host_not_by_availability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ui_client: FakeUiClient,
    bridge_client: FakeBridgeClient,
) -> None:
    """Both hosts up: an adoption goes to the host that MINTED the handle.

    Availability order is the right rule for a fresh `open` — no handle names a
    host there — and the wrong one for an adoption: the handle IS the host, and
    an adoption resolves with nothing held, so the probe order used to choose.
    A `bridge:` hand-over while the app was up therefore went to the app, which
    never issued the capability and can only answer `owner_refused` "not yours"
    — a refusal that blames the model for the routing (review R10).
    """
    _hosts(monkeypatch, ui=True, bridge=True)
    # A handed-over tab arrives at a session holding nothing, which the fakes
    # spell as an `unresolved` recovery.
    ui_client.recover_state = "unresolved"
    daemon_handle = "bridge:31:aaaaaaaabbbbccccddddeeeeffff0000"
    app_handle = "ui:31:aaaaaaaabbbbccccddddeeeeffff0000"

    # 1. A `bridge:` hand-over reaches the daemon even with the app up.
    bridge_context, _ = _context(tmp_path / "bridge")
    adopted = await builtin.execute_browser(
        "t", {"action": "open", "tab": daemon_handle}, None, None, bridge_context
    )
    assert adopted.is_error is False, adopted.text
    assert [params.get("tab") for params in _opens(bridge_client)] == [
        daemon_handle
    ], "the daemon that minted the handle must receive the adoption"
    assert _opens(ui_client) == [], (
        "the app was asked to adopt a capability it never minted, which is the "
        "`owner_refused` misattribution this routing removes"
    )

    # 2. A `ui:` hand-over reaches the app's host.
    ui_context, _ = _context(tmp_path / "ui")
    adopted = await builtin.execute_browser(
        "t", {"action": "open", "tab": app_handle}, None, None, ui_context
    )
    assert adopted.is_error is False, adopted.text
    assert [params.get("tab") for params in _opens(ui_client)] == [app_handle]
    assert [params.get("tab") for params in _opens(bridge_client)] == [
        daemon_handle
    ], "the daemon must not hear about an app adoption"

    # 3. With no handle, availability still decides: the app first, then the
    #    extension. Nothing about the adoption rule may leak into a fresh open.
    fresh_context, _ = _context(tmp_path / "fresh")
    fresh = await builtin.execute_browser(
        "t", {"action": "open", "url": "https://example.com"}, None, None, fresh_context
    )
    assert fresh.is_error is False, fresh.text
    assert [params.get("tab", "") for params in _opens(ui_client)] == [app_handle, ""]
    assert [params.get("tab") for params in _opens(bridge_client)] == [daemon_handle]


class FencedBridgeClient(FakeBridgeClient):
    """The extension's generation fence, mirrored from `extension/src/ownership.ts`.

    `FakeBridgeClient` accepts every generation, which is exactly why it could not
    see the per-lane recovery defect: the real extension refuses a plain `open`
    whose `owner_generation` differs from its stored scope's (strict equality,
    `mutate()`), and ONLY `owner_recover` consults `previous_generation(s)` to
    advance that scope (`withOwnership`). A fake that skips the fence cannot tell
    "the extension was reconciled" from "the extension was never asked".
    """

    def __init__(self) -> None:
        super().__init__()
        #: proof -> {"session", "generation", "tab"}: the extension's ownerScopes.
        self.scopes: dict[str, dict[str, str]] = {}
        #: Refuse a fresh `open` with origin_not_allowed (a denied explicit open).
        self.deny_open = False

    async def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        from local_operator.browser_bridge.backend import BridgeError
        from local_operator.browser_bridge.protocol import ErrorCode

        proof = str(params.get("owner_proof", ""))
        generation = str(params.get("owner_generation", ""))
        session = str(params.get("requester", ""))
        if method == "owner_recover":
            self.calls.append((method, params))
            scope = self.scopes.get(proof)
            if scope is None:
                return {"ownership_version": 1, "state": "unresolved", "tab": ""}
            if scope["session"] != session:
                raise BridgeError(ErrorCode.OWNER_REFUSED, "foreign", {"refusal": "foreign"})
            predecessors = params.get("previous_generations") or []
            if scope["generation"] != generation and (
                scope["generation"] != params.get("previous_generation")
                and scope["generation"] not in predecessors
            ):
                raise BridgeError(ErrorCode.OWNER_REFUSED, "stale", {"refusal": "stale"})
            scope["generation"] = generation
            return {
                "ownership_version": 1,
                "state": "owned" if scope["tab"] else "closed",
                "tab": scope["tab"],
            }
        if method == "open":
            self.calls.append((method, params))
            scope = self.scopes.setdefault(
                proof, {"session": session, "generation": generation, "tab": ""}
            )
            if scope["session"] != session:
                raise BridgeError(ErrorCode.OWNER_REFUSED, "foreign", {"refusal": "foreign"})
            if scope["generation"] != generation:
                raise BridgeError(
                    ErrorCode.OWNER_REFUSED,
                    "browser owner generation is stale or unresolved",
                    {"refusal": "stale"},
                )
            if self.deny_open:
                raise BridgeError(
                    ErrorCode.ORIGIN_NOT_ALLOWED,
                    "site not allowed",
                    {"origin": "https://example.test", "url": params["url"]},
                )
            scope["tab"] = "bridge:5:ccccccccddddeeeeffff0000111111"
            return {"tab": scope["tab"], "url": params["url"]}
        if method in ("request_access", "await_access", "cancel_access"):
            self.calls.append((method, params))
            answer = "cancelled" if method == "cancel_access" else "allowed"
            return {"origin": "https://example.test", "state": answer}
        if method in ("close", "owner_finish", "owner_release"):
            scope = self.scopes.get(proof)
            if scope is not None:
                scope["tab"] = ""
        return await super().call(method, params)


@pytest.fixture
def fenced_bridge(monkeypatch: pytest.MonkeyPatch) -> FencedBridgeClient:
    """`bridge_client`'s wiring, with the extension's generation fence."""
    fake = FencedBridgeClient()
    from local_operator.browser_bridge import backend as bridge_backend

    monkeypatch.setattr(resources, "BridgeClient", lambda *args, **kwargs: fake)
    monkeypatch.setattr(bridge_backend, "BridgeClient", lambda *args, **kwargs: fake)
    return fake


async def _resume_on_ui_lane(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ui_client: FakeUiClient,
    bridge: FencedBridgeClient,
) -> tuple[Any, ToolContext, BrowserResource]:
    """Run 1 registers an extension scope under lease G1; run 2 is the SAME durable
    SID under a NEW lease G2 whose first work lands on the app's lane.

    This is the live shape: the production `SessionFactory` acquires a new lease
    on resume and rebuilds the resource, the session works on the app (ui first in
    the availability order), and only then asks for the extension explicitly.
    """
    from local_operator.session_lease import acquire_session_lease

    directory = tmp_path / "sessions" / "sid"
    directory.mkdir(parents=True)
    lease1 = acquire_session_lease(directory)
    _hosts(monkeypatch, ui=False, bridge=True)
    first_context, first = _context(directory)
    opened = await builtin.execute_browser(
        "t", {"action": "open", "url": "https://example.com"}, None, None, first_context
    )
    assert not opened.is_error, opened.text
    closed = await builtin.execute_browser("t", {"action": "close"}, None, None, first_context)
    assert not closed.is_error, closed.text
    assert bridge.scopes[str(first.params()["owner_proof"])]["generation"] == lease1.generation
    lease1.release()

    lease2 = acquire_session_lease(directory)
    assert lease2.generation != lease1.generation
    _hosts(monkeypatch, ui=True, bridge=True)
    ui_client.recover_state = "unresolved"
    context, resource = _context(directory)
    worked = await builtin.execute_browser("t", {"action": "tabs"}, None, None, context)
    assert not worked.is_error, worked.text
    assert resource._lane().name == "ui", "the session's first work must land on the app"
    return lease2, context, resource


@pytest.mark.asyncio
async def test_explicit_extension_open_after_ui_work_reconciles_the_extension_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ui_client: FakeUiClient,
    fenced_bridge: FencedBridgeClient,
) -> None:
    """Same SID, NEW lease, app lane first, then an explicit `backend=extension` open.

    `recovered` was ONE flag for the whole resource while reconciliation is per
    host: the app's acknowledgement stood in for the extension's, so the extension
    was never sent `owner_recover`, its older-generation scope was never advanced,
    and the plain `open` met its strict-equality fence — a typed `owner_refused`
    ("generation is stale or unresolved") on a perfectly legitimate resume.

    Driven through `execute_browser`, because the lane has to MOVE on the real
    entry: with a generation already loaded the gate is skipped, so before the fix
    nothing moved the lane to the hinted host at all.
    """
    lease2, context, resource = await _resume_on_ui_lane(
        tmp_path, monkeypatch, ui_client, fenced_bridge
    )
    ui_before = list(ui_client.calls)
    fenced_bridge.calls.clear()

    opened = await builtin.execute_browser(
        "t",
        {"action": "open", "backend": "extension", "url": "https://example.com"},
        None,
        None,
        context,
    )

    assert not opened.is_error, opened.text
    assert [method for method, _p in fenced_bridge.calls] == ["owner_recover", "open"]
    recover = fenced_bridge.calls[0][1]
    assert recover["owner_generation"] == lease2.generation
    assert recover["previous_generation"] != "", "the predecessor is how the scope advances"
    assert resource.record["surface_id"].startswith("bridge:")
    assert ui_client.calls == ui_before, "the app must hear nothing of the extension's open"
    lease2.release()


@pytest.mark.asyncio
async def test_recover_after_a_lane_move_goes_to_the_new_lane_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ui_client: FakeUiClient,
    fenced_bridge: FencedBridgeClient,
) -> None:
    """`recover` takes no `backend`, so it can only follow the lane the open set.

    With the lane on the extension, the action is sent there, once, and never to
    the app. The destination was already reconciled by the open, so the entry's
    own refresh must not add a second `owner_recover` in front of it.
    """
    lease2, context, _resource = await _resume_on_ui_lane(
        tmp_path, monkeypatch, ui_client, fenced_bridge
    )
    opened = await builtin.execute_browser(
        "t",
        {"action": "open", "backend": "extension", "url": "https://example.com"},
        None,
        None,
        context,
    )
    assert not opened.is_error, opened.text
    ui_before = list(ui_client.calls)
    fenced_bridge.calls.clear()

    recovered = await builtin.execute_browser("t", {"action": "recover"}, None, None, context)

    assert not recovered.is_error, recovered.text
    assert [method for method, _p in fenced_bridge.calls] == ["owner_recover"]
    assert ui_client.calls == ui_before, "recover must never be sent to the app"
    lease2.release()


@pytest.mark.asyncio
async def test_a_same_lane_repeat_owes_no_second_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ui_client: FakeUiClient,
    fenced_bridge: FencedBridgeClient,
) -> None:
    """Per-lane state must not turn into per-call state: a lane reconciled once in
    this run stays reconciled, on the app and on the extension alike."""
    lease2, context, _resource = await _resume_on_ui_lane(
        tmp_path, monkeypatch, ui_client, fenced_bridge
    )
    again = await builtin.execute_browser("t", {"action": "tabs"}, None, None, context)
    assert not again.is_error, again.text
    assert [m for m, _p in ui_client.calls if m == "owner_recover"] == ["owner_recover"]

    fenced_bridge.calls.clear()
    for _ in range(2):
        opened = await builtin.execute_browser(
            "t",
            {"action": "open", "backend": "extension", "url": "https://example.com"},
            None,
            None,
            context,
        )
        assert not opened.is_error, opened.text
        closed = await builtin.execute_browser("t", {"action": "close"}, None, None, context)
        assert not closed.is_error, closed.text
    assert [m for m, _p in fenced_bridge.calls if m == "owner_recover"] == ["owner_recover"]
    lease2.release()


@pytest.mark.asyncio
async def test_approval_after_a_denied_explicit_extension_open_stays_on_the_extension(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ui_client: FakeUiClient,
    fenced_bridge: FencedBridgeClient,
) -> None:
    """Resumed same SID, app lane first, then a DENIED explicit extension open.

    No tab is minted, yet the approval the operator owes belongs to the extension
    that refused the navigation. `request_access`/`await_access`/`cancel_access`
    take no `backend`, so they follow the saved transport — and the app must not
    receive any of them, nor any ownership verb for the extension's scope.
    """
    lease2, context, resource = await _resume_on_ui_lane(
        tmp_path, monkeypatch, ui_client, fenced_bridge
    )
    ui_before = list(ui_client.calls)
    fenced_bridge.deny_open = True
    denied = await builtin.execute_browser(
        "t",
        {"action": "open", "backend": "extension", "url": "https://example.test"},
        None,
        None,
        context,
    )
    assert denied.is_error
    assert (denied.details or {}).get("error_code") == "origin_not_allowed", denied.text
    assert context.browser is not None and context.browser.surface_id == ""

    fenced_bridge.calls.clear()
    for action in ("request_access", "await_access", "cancel_access"):
        result = await builtin.execute_browser(
            action,
            {"action": action, "url": "https://example.test", "timeout_s": 1},
            None,
            None,
            context,
        )
        assert not result.is_error, result.text
        assert context.browser is not None and context.browser.surface_id == ""
    assert [m for m, _p in fenced_bridge.calls if m.endswith("_access")] == [
        "request_access",
        "await_access",
        "cancel_access",
    ]
    assert ui_client.calls == ui_before, "approvals must never be raised on the app"
    assert resource.record["surface_id"] == ""
    assert "pending_access_host" not in resource.record
    lease2.release()


async def _crash_after_initialize(directory: Path) -> None:
    """A run that publishes its generation durably and dies before any host ACKs.

    `initialize()` appends the run's generation to the replay chain and saves it;
    no peer has heard of it. The lease is released as a crashed owner's would be
    once proven dead, so the next resume acquires a fresh one.
    """
    from local_operator.session_lease import acquire_session_lease

    lease = acquire_session_lease(directory)
    _context(directory)[1].initialize()
    lease.release()


@pytest.mark.asyncio
async def test_one_lanes_ack_keeps_the_other_lanes_replay_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ui_client: FakeUiClient,
    fenced_bridge: FencedBridgeClient,
) -> None:
    """Review R1 / QA Q1: the app's ACK must not erase what the extension still needs.

    The extension's scope is stamped G1 (its last acknowledged generation). G2 is
    published and the run dies; G3 resumes, works on the APP (whose successful
    `owner_recover` ACK used to replace the whole chain with `[G3]`), and only then
    opens explicitly on the extension. The extension's fence accepts G1 only as
    the current generation, `previous_generation` (G2 here, not G1) or a member of
    `previous_generations` — so the chain must still carry G1 or the open is
    refused as stale on a legitimate resume.
    """
    from local_operator.session_lease import acquire_session_lease

    directory = tmp_path / "sessions" / "sid"
    directory.mkdir(parents=True)
    lease1 = acquire_session_lease(directory)
    _hosts(monkeypatch, ui=False, bridge=True)
    first_context, first = _context(directory)
    for action in ("open", "close"):
        done = await builtin.execute_browser(
            "t", {"action": action, "url": "https://example.com"}, None, None, first_context
        )
        assert not done.is_error, done.text
    g1 = lease1.generation
    assert fenced_bridge.scopes[str(first.params()["owner_proof"])]["generation"] == g1
    lease1.release()
    await _crash_after_initialize(directory)

    lease3 = acquire_session_lease(directory)
    _hosts(monkeypatch, ui=True, bridge=True)
    ui_client.recover_state = "closed"
    context, resource = _context(directory)
    worked = await builtin.execute_browser("t", {"action": "tabs"}, None, None, context)
    assert not worked.is_error, worked.text
    assert resource._lane().name == "ui"
    assert resource.previous_generation != g1, "G1 must not be the immediate predecessor"

    fenced_bridge.calls.clear()
    opened = await builtin.execute_browser(
        "t",
        {"action": "open", "backend": "extension", "url": "https://example.com"},
        None,
        None,
        context,
    )

    assert not opened.is_error, opened.text
    assert [m for m, _p in fenced_bridge.calls] == ["owner_recover", "open"]
    assert g1 in fenced_bridge.calls[0][1]["previous_generations"]
    lease3.release()


@pytest.mark.asyncio
async def test_the_replay_chain_collapses_once_every_lane_acked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ui_client: FakeUiClient,
    fenced_bridge: FencedBridgeClient,
) -> None:
    """The converse of retention: it must END. One lane's ACK trims only that
    lane's view; the shared chain (what a never-acknowledged lane is owed)
    collapses only when EVERY lane has acknowledged."""
    from local_operator.session_lease import acquire_session_lease

    directory = tmp_path / "sessions" / "sid"
    directory.mkdir(parents=True)
    lease1 = acquire_session_lease(directory)
    _hosts(monkeypatch, ui=False, bridge=True)
    first_context, _first = _context(directory)
    for action in ("open", "close"):
        done = await builtin.execute_browser(
            "t", {"action": action, "url": "https://example.com"}, None, None, first_context
        )
        assert not done.is_error, done.text
    lease1.release()

    lease2 = acquire_session_lease(directory)
    _hosts(monkeypatch, ui=True, bridge=True)
    ui_client.recover_state = "closed"
    context, resource = _context(directory)
    worked = await builtin.execute_browser("t", {"action": "tabs"}, None, None, context)
    assert not worked.is_error, worked.text
    # Only the app has acknowledged: its own view collapsed, the extension's did not.
    assert resource.lane_generations("ui") == [resource.generation]
    assert len(resource.lane_generations("bridge")) > 1, "the extension still owes an ACK"
    assert len(resource.record["bridge_generations"]) > 1

    opened = await builtin.execute_browser(
        "t",
        {"action": "open", "backend": "extension", "url": "https://example.com"},
        None,
        None,
        context,
    )
    assert not opened.is_error, opened.text
    # Both lanes acknowledged: nothing older can be held by any host any more.
    assert resource.record["bridge_generations"] == [resource.generation]
    assert resource.lane_generations("ui") == [resource.generation]
    assert resource.lane_generations("bridge") == [resource.generation]
    lease2.release()


@pytest.mark.asyncio
async def test_a_lane_that_never_acknowledges_cannot_grow_the_chain_unbounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ui_client: FakeUiClient,
    fenced_bridge: FencedBridgeClient,
) -> None:
    """Retention for an absent lane is bounded: across many resumes where only the
    app works, the durable chain and what is sent never exceed the cap."""
    from local_operator.browser_bridge.resources import MAX_REPLAY_GENERATIONS
    from local_operator.session_lease import acquire_session_lease

    directory = tmp_path / "sessions" / "sid"
    directory.mkdir(parents=True)
    _hosts(monkeypatch, ui=True, bridge=True)
    ui_client.recover_state = "closed"
    # Bound before the loop so the closing assertion reads a value on EVERY path:
    # a loop that never ran leaves it empty and fails the assertion, rather than
    # raising UnboundLocalError (and rather than leaving the name unbound for the
    # type checker, which cannot prove the loop body executed).
    final_chain: list[str] = []
    for _ in range(MAX_REPLAY_GENERATIONS + 8):
        lease = acquire_session_lease(directory)
        context, resource = _context(directory)
        worked = await builtin.execute_browser("t", {"action": "tabs"}, None, None, context)
        assert not worked.is_error, worked.text
        assert len(resource.record["bridge_generations"]) <= MAX_REPLAY_GENERATIONS
        assert len(resource.lane_generations("bridge")) <= MAX_REPLAY_GENERATIONS
        final_chain = list(resource.record["bridge_generations"])
        lease.release()
    assert len(final_chain) == MAX_REPLAY_GENERATIONS


def test_a_record_without_lane_chains_behaves_as_the_shared_chain(tmp_path: Path) -> None:
    """Compatibility: a record written before per-lane chains existed has no
    `lane_generations`, and every lane is then owed the one shared chain, exactly
    as before. The first acknowledgement is what introduces the new field."""
    directory = tmp_path / "sessions" / "sid"
    directory.mkdir(parents=True)
    old = BrowserResource(directory, "sid")
    old.initialize()
    assert "lane_generations" not in old.record, "initialize must not introduce the field"
    old.record["bridge_generations"] = ["g-old", "g-mid", old.generation]
    old._save()

    resumed = BrowserResource(directory, "sid")
    resumed.initialize()
    for lane in ("ui", "bridge"):
        assert resumed.lane_generations(lane) == resumed.record["bridge_generations"]
        assert "g-old" in resumed.lane_generations(lane)
    assert "lane_generations" not in resumed.record


# --- A finished scope's obligation on one host, then an explicit open on the other ---------
#
# Reproduced against the installed 0.71.3 (not only against these fakes): run B works on
# one host and ends with `finish("disposed")`, so the record is `host=<B>, terminal=disposed,
# surface_id=""`. Run C is the SAME durable session under a NEW lease and opens with
# `backend=<other host>`. The record's obligation-only `host` pinned the lane to B, the
# explicit hint was therefore ignored by the lane gate (but honoured by the dispatcher), B was
# reconciled, and the other host got a bare `open` stamped with a generation it had never been
# told about -> `owner_refused: browser owner generation is stale or unresolved`.


def _trace_rpcs(ui: FakeUiClient, bridge: FakeBridgeClient) -> list[str]:
    """One ORDERED log across both hosts, e.g. `ui.owner_recover`, `ext.open`.

    The defect is about which host heard what, and in which order; two separate call
    lists cannot show that `owner_recover` reached a host BEFORE its `open`.
    """
    order: list[str] = []

    def wrap(client: Any, label: str) -> None:
        original = client.call

        async def call(method: str, params: dict[str, Any]) -> dict[str, Any]:
            order.append(f"{label}.{method}")
            return await original(method, params)

        client.call = call

    wrap(ui, "ui")
    wrap(bridge, "ext")
    return order


async def _finished_scope_on(
    directory: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    first: str,
    second: str,
    legacy: bool = False,
) -> Any:
    """Run A on `first`, run B on `second` ending in `finish("disposed")`; return lease B.

    Written by the product's own `BrowserResource` and `execute_browser`, so the record
    (host, terminal, chain) is what the code really leaves behind. `legacy` strips the
    per-lane chains afterwards, which is the shape a record written before they existed has.
    """
    from local_operator.session_lease import acquire_session_lease

    for name in (first, second):
        lease = acquire_session_lease(directory)
        _hosts(monkeypatch, ui=name == "ui", bridge=name == "bridge")
        context, resource = _context(directory)
        opened = await builtin.execute_browser(
            "t", {"action": "open", "url": "https://example.com"}, None, None, context
        )
        assert not opened.is_error, opened.text
        closed = await builtin.execute_browser("t", {"action": "close"}, None, None, context)
        assert not closed.is_error, closed.text
        if name == second:
            ended = await resource.finish(resource.generation, "disposed")
            assert ended.state == "closed", ended
        lease.release()
    path = directory / ".browser-resource.json"
    record = json.loads(path.read_text())
    assert record["terminal"] == "disposed" and record["host"] == (
        "ui" if second == "ui" else "bridge"
    )
    if legacy:
        record.pop("lane_generations", None)
        path.write_text(json.dumps(record))
    return record


@pytest.mark.asyncio
@pytest.mark.parametrize("ui_answer", ["closed", "unresolved"])
@pytest.mark.parametrize("legacy", [False, True], ids=["current-record", "legacy-no-lane-chains"])
async def test_explicit_extension_open_after_a_finished_app_scope_reconciles_app_then_extension(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ui_client: FakeUiClient,
    fenced_bridge: FencedBridgeClient,
    ui_answer: str,
    legacy: bool,
) -> None:
    """The consumer shape: `host=ui, terminal=disposed`, same SID, new lease, explicit extension.

    The old app obligation is discharged FIRST (it is never dropped), the fresh open is
    then re-decided from the caller's `backend`, the destination is reconciled BEFORE it
    allocates, and each host is asked to recover exactly once.
    """
    from local_operator.session_lease import acquire_session_lease

    directory = tmp_path / "sessions" / "sid"
    directory.mkdir(parents=True)
    await _finished_scope_on(directory, monkeypatch, first="bridge", second="ui", legacy=legacy)
    assert (
        "lane_generations" not in json.loads((directory / ".browser-resource.json").read_text())
        or not legacy
    )

    order = _trace_rpcs(ui_client, fenced_bridge)
    ui_client.recover_state = ui_answer
    lease = acquire_session_lease(directory)
    _hosts(monkeypatch, ui=True, bridge=True)
    context, resource = _context(directory)
    result = await builtin.execute_browser(
        "t",
        {"action": "open", "backend": "extension", "url": "https://console.cloud.google.com"},
        None,
        None,
        context,
    )
    assert not result.is_error, result.text
    assert order == ["ui.owner_recover", "ext.owner_recover", "ext.open"], order
    assert resource.record["surface_id"].startswith("bridge:")
    assert resource.record["host"] == "bridge"
    assert "terminal" not in resource.record
    lease.release()


@pytest.mark.asyncio
async def test_explicit_app_open_after_a_finished_extension_scope_reconciles_extension_then_app(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ui_client: FakeUiClient,
    fenced_bridge: FencedBridgeClient,
) -> None:
    """Mirror direction: `host=bridge, terminal=disposed`, then an explicit `backend=ui`."""
    from local_operator.session_lease import acquire_session_lease

    directory = tmp_path / "sessions" / "sid"
    directory.mkdir(parents=True)
    await _finished_scope_on(directory, monkeypatch, first="ui", second="bridge")

    order = _trace_rpcs(ui_client, fenced_bridge)
    ui_client.recover_state = "closed"
    lease = acquire_session_lease(directory)
    _hosts(monkeypatch, ui=True, bridge=True)
    context, resource = _context(directory)
    result = await builtin.execute_browser(
        "t", {"action": "open", "backend": "ui", "url": "https://example.com"}, None, None, context
    )
    assert not result.is_error, result.text
    assert order == ["ext.owner_recover", "ui.owner_recover", "ui.open"], order
    assert resource.record["surface_id"].startswith("ui:")
    lease.release()


def _inject(directory: Path, **fields: Any) -> None:
    """Hand-write obligation fields into a record (labelled: NOT written by product code)."""
    path = directory / ".browser-resource.json"
    record = json.loads(path.read_text())
    record.update(fields)
    path.write_text(json.dumps(record))


@pytest.mark.asyncio
async def test_a_resolved_release_pause_is_released_before_the_explicit_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ui_client: FakeUiClient,
    fenced_bridge: FencedBridgeClient,
) -> None:
    """A paused-scope retention on the old host: released there, then the open proceeds."""
    from local_operator.session_lease import acquire_session_lease

    directory = tmp_path / "sessions" / "sid"
    directory.mkdir(parents=True)
    await _finished_scope_on(directory, monkeypatch, first="bridge", second="ui")
    _inject(directory, retention="paused scope", release_pause=True)

    order = _trace_rpcs(ui_client, fenced_bridge)
    ui_client.recover_state = "closed"
    lease = acquire_session_lease(directory)
    _hosts(monkeypatch, ui=True, bridge=True)
    context, resource = _context(directory)
    result = await builtin.execute_browser(
        "t",
        {"action": "open", "backend": "extension", "url": "https://example.com"},
        None,
        None,
        context,
    )
    assert not result.is_error, result.text
    assert order == ["ui.owner_recover", "ui.owner_release", "ext.owner_recover", "ext.open"], order
    assert "release_pause" not in resource.record
    lease.release()


@pytest.mark.asyncio
async def test_an_unresolved_old_host_obligation_refuses_instead_of_allocating_elsewhere(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ui_client: FakeUiClient,
    fenced_bridge: FencedBridgeClient,
) -> None:
    """The app answers `unresolved`, so its release-pause is NOT discharged.

    Dropping the obligation to honour the hint would strand the app's paused scope, and
    dispatching to the extension anyway is the mismatched-host defect. The call is refused
    with the typed conflict code and the extension hears NOTHING (no recover, no open).
    """
    from local_operator.session_lease import acquire_session_lease

    directory = tmp_path / "sessions" / "sid"
    directory.mkdir(parents=True)
    await _finished_scope_on(directory, monkeypatch, first="bridge", second="ui")
    _inject(directory, retention="paused scope", release_pause=True)

    order = _trace_rpcs(ui_client, fenced_bridge)
    scopes_before = json.dumps(fenced_bridge.scopes, sort_keys=True)
    ui_client.recover_state = "unresolved"
    lease = acquire_session_lease(directory)
    _hosts(monkeypatch, ui=True, bridge=True)
    context, resource = _context(directory)
    result = await builtin.execute_browser(
        "t",
        {"action": "open", "backend": "extension", "url": "https://example.com"},
        None,
        None,
        context,
    )
    assert result.is_error
    assert (result.details or {}).get("error_code") == "backend_conflict", result.text
    assert order == ["ui.owner_recover"], order
    assert resource.record.get("release_pause"), "the obligation must still be owed"
    assert (
        json.dumps(fenced_bridge.scopes, sort_keys=True) == scopes_before
    ), "destination untouched"
    lease.release()


@pytest.mark.asyncio
async def test_a_failed_old_host_recovery_does_not_reach_the_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ui_client: FakeUiClient,
    fenced_bridge: FencedBridgeClient,
) -> None:
    """If the old host cannot even answer `owner_recover`, the destination hears nothing."""
    from local_operator.browser_bridge.backend import BridgeError
    from local_operator.browser_bridge.protocol import ErrorCode
    from local_operator.session_lease import acquire_session_lease

    directory = tmp_path / "sessions" / "sid"
    directory.mkdir(parents=True)
    await _finished_scope_on(directory, monkeypatch, first="bridge", second="ui")

    order = _trace_rpcs(ui_client, fenced_bridge)
    scopes_before = json.dumps(fenced_bridge.scopes, sort_keys=True)
    real = ui_client.call

    async def failing(method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "owner_recover":
            raise BridgeError(ErrorCode.OWNER_REFUSED, "stale", {"refusal": "stale"})
        return await real(method, params)

    ui_client.call = failing  # type: ignore[method-assign]
    lease = acquire_session_lease(directory)
    _hosts(monkeypatch, ui=True, bridge=True)
    context, _resource = _context(directory)
    result = await builtin.execute_browser(
        "t",
        {"action": "open", "backend": "extension", "url": "https://example.com"},
        None,
        None,
        context,
    )
    assert result.is_error
    assert not any(step.startswith("ext.") for step in order), order
    assert (
        json.dumps(fenced_bridge.scopes, sort_keys=True) == scopes_before
    ), "destination untouched"
    lease.release()


@pytest.mark.asyncio
async def test_a_held_handle_still_outranks_an_explicit_hint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ui_client: FakeUiClient,
    fenced_bridge: FencedBridgeClient,
) -> None:
    """Precedence is unchanged: a HELD app tab refuses `backend=extension`, extension untouched."""
    _hosts(monkeypatch, ui=True, bridge=True)
    context, _resource = _context(tmp_path)
    first = await builtin.execute_browser(
        "t", {"action": "open", "url": "https://example.com"}, None, None, context
    )
    assert not first.is_error, first.text
    held = context.browser.surface_id if context.browser else ""
    assert held.startswith("ui:")
    order = _trace_rpcs(ui_client, fenced_bridge)
    refused = await builtin.execute_browser(
        "t",
        {"action": "open", "backend": "extension", "url": "https://example.com"},
        None,
        None,
        context,
    )
    assert refused.is_error
    assert (refused.details or {}).get("error_code") == "backend_conflict"
    assert not any(step.startswith("ext.") for step in order), order
    assert context.browser and context.browser.surface_id == held
