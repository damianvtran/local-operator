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
from local_operator.browser_bridge.resources import BrowserResource
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
