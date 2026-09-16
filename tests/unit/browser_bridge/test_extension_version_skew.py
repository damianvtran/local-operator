"""The extension/runtime version skew: a window, a degradation, an advisory.

Three separate defects live in one change and each has its own section below:

1. the handshake required proto EQUALITY, so a runtime release refused every
   installed extension until Google approved the matching one (and the `owner_*`
   call sites then told the user to update an extension the store would not
   serve). The window is now `MIN_SUPPORTED_PROTO..PROTO_VERSION`.
2. a pre-ownership extension was misdiagnosed as "update the extension" on a
   path that could just keep working — and, worse, told the same thing about a
   WEDGED current extension whose real remedy is the OFF/ON toggle. The two are
   separated by the peer's reported version.
3. the update advice had to become a note that blocks nothing, which means a
   predicate with exactly one spelling (protocol) published on three surfaces.

The version-pin test at the end is what stops `EXPECTED_EXTENSION_VERSION` from
drifting away from the extension tree it describes; it skips when the extension
is not present (the installed-wheel case).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from local_operator.browser_bridge import daemon as daemon_module
from local_operator.browser_bridge import resources
from local_operator.browser_bridge import state as state_store
from local_operator.browser_bridge.backend import BridgeError
from local_operator.browser_bridge.daemon import create_app
from local_operator.browser_bridge.protocol import (
    EXPECTED_EXTENSION_VERSION,
    EXTENSION_UPDATE_NOTE,
    MIN_SUPPORTED_PROTO,
    OWNERSHIP_MIN_EXTENSION_VERSION,
    PROTO_VERSION,
    ErrorCode,
    extension_older,
    extension_update_note,
    parse_extension_version,
    proto_supported,
)
from local_operator.browser_bridge.resources import (
    BrowserOwnershipError,
    BrowserResource,
    cleanup_disposition,
    read_inventory,
)

EXTENSION_ID = "a" * 32
ORIGIN = f"chrome-extension://{EXTENSION_ID}"


def _hello(proto: int, version: str = EXPECTED_EXTENSION_VERSION) -> dict[str, Any]:
    return {
        "event": "hello",
        "proto": proto,
        "token": "",
        "extension_version": version,
        "browser": "Chrome/153",
    }


# --- 1. the window --------------------------------------------------------


@pytest.mark.parametrize("proto", list(range(MIN_SUPPORTED_PROTO, PROTO_VERSION + 1)))
def test_every_proto_in_the_window_handshakes(tmp_path: Path, proto: int) -> None:
    """The equality test this replaces had NO test; this is the regression.

    Every supported proto must be ACKED (not closed), and the peer's own proto
    must be recorded on the link — it is what a future daemon->extension frame
    has to be gated on, so losing it silently would re-create the skew one layer
    down.
    """
    app = create_app(root=tmp_path)
    with TestClient(app) as client:
        with client.websocket_connect("/extension", headers={"origin": ORIGIN}) as socket:
            socket.send_json(_hello(proto, "0.1.8"))
            ack = socket.receive_json()
            assert ack["event"] == "hello_ack"
            assert app.state.bridge.link.peer_proto == proto
            assert app.state.bridge.link.extension_version == "0.1.8"


def test_the_acceptance_rule_is_a_window_not_an_equality() -> None:
    """R1-6: at today's width (`MIN_SUPPORTED_PROTO == PROTO_VERSION == 1`) the
    window and the equality test it replaced accept EXACTLY the same set, so no
    handshake test can tell them apart — the test above fails under a revert on
    the absent `peer_proto` attribute, not on window semantics.

    So the rule is pinned where it is decidable: on a synthetic
    ``(low, proto, high)`` triple, where an equality reintroduced as
    ``proto != high`` accepts only 3 and fails on 1 and 2. The bounds are passed
    explicitly rather than read from this module, which is also how `daemon.py`
    calls it — so a monkeypatched window (as the negotiation test above opens)
    moves what the daemon accepts.
    """
    low, high = 1, 3
    accepted = [proto for proto in range(-1, 6) if proto_supported(proto, low=low, high=high)]
    assert accepted == [1, 2, 3], "the window is inclusive on BOTH edges"
    # The live window, and the shape of a revert.
    assert proto_supported(PROTO_VERSION, low=MIN_SUPPORTED_PROTO, high=PROTO_VERSION)
    assert not proto_supported(PROTO_VERSION + 1, low=MIN_SUPPORTED_PROTO, high=PROTO_VERSION)
    assert not proto_supported(MIN_SUPPORTED_PROTO - 1, low=MIN_SUPPORTED_PROTO, high=PROTO_VERSION)


@pytest.mark.parametrize(
    ("proto", "reason"),
    [
        (MIN_SUPPORTED_PROTO - 1, "proto_too_old"),
        (PROTO_VERSION + 1, "proto_too_new"),
    ],
)
def test_proto_outside_the_window_closes_4001_with_its_own_reason(
    tmp_path: Path, proto: int, reason: str
) -> None:
    """4001 is KEPT (worker.ts maps it to the popup's "incompatible" card, and
    an unknown code renders as a plain disconnect), and the two directions carry
    distinct reasons so a debug log says which side of the window the peer
    missed.
    """
    app = create_app(root=tmp_path)
    with TestClient(app) as client:
        with pytest.raises(Exception) as caught:  # Starlette surfaces the close.
            with client.websocket_connect("/extension", headers={"origin": ORIGIN}) as socket:
                socket.send_json(_hello(proto))
                socket.receive_json()
        # Starlette surfaces the close as a WebSocketDisconnect carrying the
        # code and the reason; `str()` of it is empty, so read the attributes.
        assert getattr(caught.value, "code", None) == 4001
        assert getattr(caught.value, "reason", None) == reason


def test_hello_ack_reports_the_negotiated_proto(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`HelloAck(paired=…)` reports PROTO_VERSION by default, which is a LIE once
    the window is wider than one. Open the window artificially so the assertion
    is about negotiation rather than about today's single-value range.
    """
    monkeypatch.setattr(daemon_module, "PROTO_VERSION", 3)
    monkeypatch.setattr(daemon_module, "MIN_SUPPORTED_PROTO", 1)
    app = create_app(root=tmp_path)
    with TestClient(app) as client:
        with client.websocket_connect("/extension", headers={"origin": ORIGIN}) as socket:
            socket.send_json(_hello(2))
            ack = socket.receive_json()
            assert ack["proto"] == 2, "the pair's proto is the LOWER of the two, not the ceiling"
            assert app.state.bridge.link.peer_proto == 2


def test_forget_link_state_drops_the_peer_identity(tmp_path: Path) -> None:
    """Both stamps are link-scoped: a version that outlived its socket would
    drive the advisory and the ownership split against a peer that is gone."""
    app = create_app(root=tmp_path)
    with TestClient(app) as client:
        with client.websocket_connect("/extension", headers={"origin": ORIGIN}) as socket:
            socket.send_json(_hello(PROTO_VERSION, "0.1.8"))
            socket.receive_json()
            assert app.state.bridge.link.extension_version == "0.1.8"
        app.state.bridge.link.forget_link_state()
        assert app.state.bridge.link.extension_version == ""
        assert app.state.bridge.link.peer_proto == PROTO_VERSION


# --- 3. the advisory ------------------------------------------------------


@pytest.mark.parametrize(
    ("have", "want"),
    [
        ("0.1.10", EXPECTED_EXTENSION_VERSION),  # the live-store case
        ("0.1.9", "0.1.10"),  # and the numeric, not lexical, comparison
        ("0.1", EXPECTED_EXTENSION_VERSION),  # a prefix is older, not unparseable
    ],
)
def test_known_older_versions_are_older(have: str, want: str) -> None:
    assert extension_older(have, want) is True


@pytest.mark.parametrize(
    ("have", "want"),
    [
        (EXPECTED_EXTENSION_VERSION, EXPECTED_EXTENSION_VERSION),  # equal is not older
        ("0.1.16", "0.1.15"),  # fixed pair: an extension bump cannot invert this fixture
        ("", EXPECTED_EXTENSION_VERSION),  # nothing reported: unknown
        ("0.1.10-beta", EXPECTED_EXTENSION_VERSION),  # not a dotted numeric version
        ("Chrome/153", EXPECTED_EXTENSION_VERSION),
    ],
)
def test_unparseable_ahead_and_equal_are_never_older(have: str, want: str) -> None:
    """The direction that matters: "unknown" must never collapse into "older",
    or an unreadable version would nag (and, in resources, silently demote a
    current extension to the capability-only path)."""
    assert extension_older(have, want) is False


def test_parse_is_strict_about_the_parts() -> None:
    assert parse_extension_version("0.1.10") == (0, 1, 10)
    assert parse_extension_version("") is None
    assert parse_extension_version("0.1.x") is None
    assert parse_extension_version(" 0.1.10 ") == (0, 1, 10)


def test_the_note_is_generated_and_never_demands_anything() -> None:
    note = extension_update_note("0.1.10")
    assert note == EXTENSION_UPDATE_NOTE.format(have="0.1.10", want=EXPECTED_EXTENSION_VERSION)
    assert "0.1.10" in note and EXPECTED_EXTENSION_VERSION in note
    assert "nothing is blocked" in note
    # The register of the copy this change removes. A future edit that
    # reintroduces it must fail here rather than at a user's desk.
    assert "requires" not in note and "must" not in note.lower()
    assert "update it in Chrome when a newer version is offered" in note


def test_health_reports_the_extension_identity_and_the_advisory(tmp_path: Path) -> None:
    app = create_app(root=tmp_path)
    with TestClient(app) as client:
        empty = client.get("/health").json()
        # Nothing attached: the fields exist (additive contract) and say
        # "unknown", so no stale stamp can drive the advisory.
        assert empty["extension_version"] == ""
        assert empty["extension_proto"] == 0
        assert empty["extension_expected_version"] == EXPECTED_EXTENSION_VERSION
        assert empty["extension_update_available"] is False
        with client.websocket_connect("/extension", headers={"origin": ORIGIN}) as socket:
            socket.send_json(_hello(PROTO_VERSION, "0.1.10"))
            socket.receive_json()
            health = client.get("/health").json()
            assert health["extension_version"] == "0.1.10"
            assert health["extension_proto"] == PROTO_VERSION
            assert health["extension_update_available"] is True
            # The daemon's OWN protocol version stays a separate field: that
            # conflation is what made a skew undiagnosable.
            assert health["proto"] == PROTO_VERSION
        # Disconnected again: the stamp must not outlive the socket.
        after = client.get("/health").json()
        assert after["extension_version"] == ""
        assert after["extension_update_available"] is False


def test_state_file_carries_the_identity_and_defaults_quiet(tmp_path: Path) -> None:
    """The advisory must be readable WITHOUT a socket (a session-side consumer
    may not open one), and an older file must never nag."""
    app = create_app(root=tmp_path)
    with TestClient(app) as client:
        with client.websocket_connect("/extension", headers={"origin": ORIGIN}) as socket:
            socket.send_json(_hello(PROTO_VERSION, "0.1.10"))
            socket.receive_json()
            app.state.bridge.publish()
            published = state_store.read(tmp_path)
            assert published is not None
            assert published.extension_version == "0.1.10"
            assert published.extension_update_available is True
        app.state.bridge.publish()
        detached = state_store.read(tmp_path)
        assert detached is not None
        assert detached.extension_version == ""
        assert detached.extension_update_available is False
    # A file written by an older daemon: absent keys, not a nag.
    legacy = json.loads(json.dumps({"pid": 1, "port": 4099, "session_key": "k" * 32, "proto": 1}))
    (tmp_path / state_store.RUN_DIRNAME).mkdir(parents=True, exist_ok=True)
    state_store.state_path(tmp_path).write_text(json.dumps(legacy))
    parsed = state_store.read(tmp_path)
    assert parsed is not None
    assert parsed.extension_version == ""
    assert parsed.extension_update_available is False


# --- 2. degradation -------------------------------------------------------


class _Peer:
    """A disposable protocol peer for the ownership split.

    ``owner_recover`` raises whatever the test needs; everything else is an
    AssertionError so an unexpected verb is loud rather than silently ignored.
    """

    def __init__(self, recover: dict[str, Any] | BaseException | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.recover = (
            recover
            if recover is not None
            else {
                "ownership_version": 1,
                "state": "owned",
                "tab": "bridge:100:private",
            }
        )

    async def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((method, params))
        if method == "owner_recover":
            if isinstance(self.recover, BaseException):
                raise self.recover
            assert isinstance(self.recover, dict)
            return self.recover
        if method == "close":
            return {}
        raise AssertionError(method)


def _install_peer(monkeypatch: pytest.MonkeyPatch, peer: _Peer) -> None:
    monkeypatch.setattr(resources, "BridgeClient", lambda: peer)


def _install_peer_identity(
    monkeypatch: pytest.MonkeyPatch, version: str, proto: int = PROTO_VERSION
) -> None:
    """Publish a live link with the given extension version into the file seam."""

    def read(root: Path | None = None) -> state_store.BridgeState:
        return state_store.BridgeState(
            pid=1,
            port=4099,
            session_key="k" * 32,
            proto=PROTO_VERSION,
            extension_connected=True,
            extension_version=version,
            extension_proto=proto,
            extension_update_available=extension_older(version, EXPECTED_EXTENSION_VERSION),
        )

    monkeypatch.setattr(resources.state_store, "read", read)


BARE_INTERNAL = BridgeError(
    ErrorCode.INTERNAL,
    "an internal error occurred",
    {},
)


def test_the_ownership_floor_is_the_first_tree_that_shipped_the_lifecycle() -> None:
    """R1-1: the floor named 0.1.10, one release ABOVE the first tree carrying
    `owner_*`.

    `git show ee146fb73:extension/manifest.json` reads `0.1.9`, and that same
    commit already declares `owner_recover|owner_finish|owner_retain|
    owner_release` in `extension/src/protocol.gen.ts`. 0.1.9 was never submitted
    to the store, but an unpacked build of it is a peer this runtime meets, and
    classifying it pre-ownership gives it the SILENT degrade instead of the
    OFF/ON remedy — the misdiagnosis this change removes, mirrored.

    The boundary is asserted from BOTH sides because the arms around it jump
    0.1.8 -> 0.1.10 and would pass with either value.
    """
    assert OWNERSHIP_MIN_EXTENSION_VERSION == "0.1.9", (
        "the floor must name the first TREE carrying owner_* — see the provenance "
        "in protocol.py; a value one release high silently degrades a capable peer"
    )
    assert extension_older("0.1.8", OWNERSHIP_MIN_EXTENSION_VERSION) is True
    assert extension_older("0.1.9", OWNERSHIP_MIN_EXTENSION_VERSION) is False
    assert (
        extension_older(OWNERSHIP_MIN_EXTENSION_VERSION, OWNERSHIP_MIN_EXTENSION_VERSION) is False
    )
    assert extension_older(OWNERSHIP_MIN_EXTENSION_VERSION, EXPECTED_EXTENSION_VERSION)


@pytest.mark.asyncio
async def test_pre_ownership_peer_degrades_instead_of_demanding_an_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live-store case. <= 0.1.8 cannot answer `owner_recover` at all, and
    the old copy told that user to update — an instruction the store may be
    unable to satisfy while the extension is otherwise perfectly usable."""
    peer = _Peer(BARE_INTERNAL)
    _install_peer(monkeypatch, peer)
    _install_peer_identity(monkeypatch, "0.1.8")
    directory = tmp_path / "synthetic"
    resource = BrowserResource(directory, directory.name)
    resource.initialize()
    result = await resource.recover()
    assert result["tab"] == ""
    assert result["state"] == "closed"
    assert resource.ownership is False
    assert resource.record["ownership"] == "unavailable"
    assert resource.recovered is True
    # No obligation was invented: a fresh scope stays clean.
    assert resource.has_durable_obligation() is False


@pytest.mark.asyncio
async def test_a_durable_obligation_keeps_the_honest_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pre-ownership peer WITH something to reconcile must not be quietly
    downgraded: the session owes the extension a reconciliation it cannot
    perform, and pretending otherwise would strand the tab."""
    peer = _Peer(BARE_INTERNAL)
    _install_peer(monkeypatch, peer)
    _install_peer_identity(monkeypatch, "0.1.8")
    directory = tmp_path / "synthetic"
    resource = BrowserResource(directory, directory.name)
    resource.initialize()
    resource.remember("bridge:9:capability")
    with pytest.raises(BrowserOwnershipError) as caught:
        await resource.recover()
    assert "updated Local Operator extension" in str(caught.value)
    assert "stopped answering" not in str(caught.value), "a pre-ownership peer is not a wedge"
    assert resource.ownership is not False, "no degradation while an obligation is live"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "version",
    ["0.1.9", "0.1.10", "0.1.12", EXPECTED_EXTENSION_VERSION],
    ids=["first-ownership-tree", "live-store", "pending", "head"],
)
async def test_a_current_extension_is_reported_as_wedged_not_old(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    """The same bare-`internal` shape, opposite remedy.

    Every version here IS ownership-capable, so telling its owner to update
    named an action that could not help; the toggle does. 0.1.9 is included on
    purpose: it is the FLOOR (R1-1), and the one member of this list that the
    store never served — a build of that tree must reach the wedge arm, not the
    silent degrade."""
    peer = _Peer(BARE_INTERNAL)
    _install_peer(monkeypatch, peer)
    _install_peer_identity(monkeypatch, version)
    directory = tmp_path / "synthetic"
    resource = BrowserResource(directory, directory.name)
    resource.initialize()
    with pytest.raises(BrowserOwnershipError) as caught:
        await resource.recover()
    message = str(caught.value)
    assert "stopped answering" in message
    assert "OFF then ON" in message and "chrome://extensions" in message
    assert "Update the extension" not in message, "the update demand is the defect"
    assert resource.ownership is not False


@pytest.mark.asyncio
async def test_an_unknown_version_is_not_read_as_old(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ "Cannot tell" must not demote a current extension to the capability-only
    path — that would hide a real wedge behind a silent downgrade."""
    peer = _Peer(BARE_INTERNAL)
    _install_peer(monkeypatch, peer)
    _install_peer_identity(monkeypatch, "")
    directory = tmp_path / "synthetic"
    resource = BrowserResource(directory, directory.name)
    resource.initialize()
    with pytest.raises(BrowserOwnershipError) as caught:
        await resource.recover()
    assert "stopped answering" in str(caught.value)
    assert resource.ownership is not False


@pytest.mark.asyncio
async def test_a_wedge_with_data_discriminators_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`timeout_s` / `stalled` are the other, already-honest producers. They must
    keep passing straight through to `format_error`'s copy rather than being
    swept into the new branch."""
    for data in ({"timeout_s": 20.0}, {"stalled": "open"}):
        peer = _Peer(BridgeError(ErrorCode.INTERNAL, "stalled", data))
        _install_peer(monkeypatch, peer)
        _install_peer_identity(monkeypatch, "0.1.8")
        directory = tmp_path / f"synthetic-{sorted(data)[0]}"
        resource = BrowserResource(directory, directory.name)
        resource.initialize()
        with pytest.raises(BridgeError):
            await resource.recover()
        assert resource.ownership is not False


@pytest.mark.asyncio
async def test_a_proto_refusal_still_demands_the_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A genuine incompatibility is not the legacy-skew case: the popup's
    `#incompatible` card and this refusal are the same verdict."""
    peer = _Peer(BridgeError(ErrorCode.PROTO_MISMATCH, "proto mismatch", {}))
    _install_peer(monkeypatch, peer)
    _install_peer_identity(monkeypatch, "0.1.8")
    directory = tmp_path / "synthetic"
    resource = BrowserResource(directory, directory.name)
    resource.initialize()
    with pytest.raises(BrowserOwnershipError) as caught:
        await resource.recover()
    assert "Update the extension" in str(caught.value)


@pytest.mark.asyncio
async def test_an_extension_update_mid_session_restores_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sticky but REFRESHABLE: the cached verdict is keyed on the peer identity,
    so a user who updates the extension gets ownership back mid-session."""
    peer = _Peer(BARE_INTERNAL)
    _install_peer(monkeypatch, peer)
    _install_peer_identity(monkeypatch, "0.1.8")
    directory = tmp_path / "synthetic"
    resource = BrowserResource(directory, directory.name)
    resource.initialize()
    await resource.recover()
    assert resource.ownership is False
    # The user updates the extension; the daemon reports the new pair.
    _install_peer_identity(monkeypatch, EXPECTED_EXTENSION_VERSION)
    peer.recover = {"ownership_version": 1, "state": "owned", "tab": "bridge:5:capability"}
    result = await resource.recover()
    assert result["ownership_version"] == 1
    assert resource.ownership is True
    assert "ownership" not in resource.record, "the legacy marker is retired by a real reconcile"


@pytest.mark.asyncio
async def test_degraded_finish_closes_the_recorded_surface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without this the fallback trades a dead tool for a permanently stranded
    tab: `owner_finish` does not exist on a legacy peer, so the scope must settle
    through the plain `close`."""
    peer = _Peer(BARE_INTERNAL)
    _install_peer(monkeypatch, peer)
    _install_peer_identity(monkeypatch, "0.1.8")
    directory = tmp_path / "synthetic"
    resource = BrowserResource(directory, directory.name)
    resource.initialize()
    await resource.recover()
    resource.remember("bridge:42:capability")
    closed = await resource.finish(resource.generation, "completed")
    assert closed.state == "closed"
    assert resource.record["surface_id"] == ""
    close_params = next(params for method, params in peer.calls if method == "close")
    assert close_params["tab"] == "bridge:42:capability"
    # R1-2: the tab capability ALONE is refused by a released peer on any
    # surface carrying an allocation, so the close must carry the identity
    # params the docstring promises are never forked away.
    assert close_params["owner_proof"], "the identity params are not forked by the fallback"
    assert close_params["allocation_id"]
    owner_verbs = [method for method, _ in peer.calls if method.startswith("owner_")]
    assert owner_verbs == ["owner_recover"], "only the probe may run: no lifecycle verb exists"


@pytest.mark.asyncio
async def test_degraded_finish_survives_a_peer_that_enforces_the_owner_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1-2, as a regression rather than a shape assertion.

    The peer below models the RELEASED extension verbatim:
    `extension/src/commands/nav.ts` throws `owner_refused` / "owner-aware client
    required" when a `close` names a surface with an `allocationId` and no
    `owner_proof` — and every surface this flow opens carries one, because
    `_browser_identity_params` folds `allocation_id` into every command from the
    same `params()` this method must pass.

    So the one-argument form did not merely look wrong: it left the tab OPEN and
    the record `pending` on the very peer the fallback exists to keep working
    (reproduced as `finish='pending'` for 0.1.9 before R1-1, and reachable by any
    future misclassification after it).
    """

    class _Guarded(_Peer):
        #: The peer's OWN surface record carries the allocation — the extension
        #: opened this tab for an owner, so it wants that owner's proof back.
        #: The guard is on the SURFACE, not on the close's params: a close that
        #: repeats neither `allocation_id` nor `owner_proof` is refused all the
        #: same, which is exactly why dropping the params is fatal.
        allocated = True

        async def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            self.calls.append((method, params))
            if method == "owner_recover":
                raise BARE_INTERNAL
            if method == "close":
                if self.allocated and not params.get("owner_proof"):
                    raise BridgeError(ErrorCode.OWNER_REFUSED, "owner-aware client required", {})
                return {}
            raise AssertionError(method)

    peer = _Guarded()
    _install_peer(monkeypatch, peer)
    _install_peer_identity(monkeypatch, "0.1.8")
    directory = tmp_path / "synthetic"
    resource = BrowserResource(directory, directory.name)
    resource.initialize()
    await resource.recover()
    resource.remember("bridge:42:capability")
    closed = await resource.finish(resource.generation, "completed")
    assert closed.state == "closed", closed.detail
    assert resource.record["surface_id"] == ""
    # …and the guard really does bite, so this test cannot pass by accident on a
    # peer that accepts anything.
    with pytest.raises(BridgeError):
        await peer.call("close", {"tab": "bridge:42:capability"})


@pytest.mark.asyncio
async def test_degraded_finish_clears_the_unenforceable_retention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1-4: a retention the extension never accepted must not outlive the tab.

    A degraded `retain` is a LOCAL statement of intent, and `finish_degraded`
    closes the tab — but the record kept `retention` set, so `cleanup_disposition`
    read a SETTLED row as "retained: … the owning session must release it": a row
    that is neither cleanable nor true, and the one place the operator looks to
    find out whether anything was stranded.
    """
    peer = _Peer(BARE_INTERNAL)
    _install_peer(monkeypatch, peer)
    _install_peer_identity(monkeypatch, "0.1.8")
    directory = tmp_path / "synthetic"
    resource = BrowserResource(directory, directory.name)
    resource.initialize()
    await resource.recover()
    resource.remember("bridge:42:capability")
    resource.record["retention"] = "pending login"
    resource._save()
    assert cleanup_disposition(resource.record)[0] is False
    assert cleanup_disposition(resource.record)[1].startswith("retained:")
    closed = await resource.finish(resource.generation, "completed")
    assert closed.state == "closed"
    assert resource.record["retention"] == ""
    settled, why = cleanup_disposition(resource.record)
    assert not why.startswith(
        "retained:"
    ), "a settled row must not report a live retention for a tab that is closed"
    assert settled is False


@pytest.mark.asyncio
async def test_degraded_finish_never_adopts_unresolved_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The existing "never adopt by numeric tab id" stance still holds in
    degraded mode: an unproven capability from a previous run is not closed."""
    peer = _Peer(BARE_INTERNAL)
    _install_peer(monkeypatch, peer)
    _install_peer_identity(monkeypatch, "0.1.8")
    directory = tmp_path / "synthetic"
    resource = BrowserResource(directory, directory.name)
    resource.initialize()
    await resource.recover()
    resource.record["unresolved_surface_id"] = "bridge:old:capability"
    resource.record["surface_id"] = ""
    closed = await resource.finish(resource.generation, "completed")
    assert closed.state == "unresolved"
    assert "no tab adopted" in closed.detail
    assert not [call for call in peer.calls if call[0] == "close"]


@pytest.mark.asyncio
async def test_degraded_finish_reports_a_failed_close_as_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed close must stay visible as a stranded tab rather than reporting
    a settled scope."""

    class _Refusing(_Peer):
        async def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            self.calls.append((method, params))
            if method == "owner_recover":
                raise BARE_INTERNAL
            if method == "close":
                raise BridgeError(ErrorCode.EXTENSION_DISCONNECTED, "no browser", {})
            raise AssertionError(method)

    peer = _Refusing()
    _install_peer(monkeypatch, peer)
    _install_peer_identity(monkeypatch, "0.1.8")
    directory = tmp_path / "synthetic"
    resource = BrowserResource(directory, directory.name)
    resource.initialize()
    await resource.recover()
    resource.remember("bridge:42:capability")
    closed = await resource.finish(resource.generation, "completed")
    assert closed.state == "pending"
    assert resource.record["surface_id"] == "bridge:42:capability"


def test_inventory_surfaces_the_redacted_legacy_marker(tmp_path: Path) -> None:
    """`lop browser tabs` has to be able to say "legacy mode": "it worked" and
    "ownership is proven" need opposite next steps and are otherwise identical
    on that screen. The marker names the MODE and no capability."""
    directory = tmp_path / "synthetic"
    resource = BrowserResource(directory, directory.name)
    resource.initialize()
    resource.remember("bridge:42:secret-capability")
    resource.record["ownership"] = "unavailable"
    resource._save()
    rows = read_inventory(tmp_path)
    assert rows[0]["ownership"] == "unavailable"
    assert "secret-capability" not in json.dumps(rows)
    assert resource.record["proof"] not in json.dumps(rows)


# --- 4. the version pin ---------------------------------------------------


def _extension_root() -> Path | None:
    root = Path(__file__).resolve().parents[3] / "extension"
    return root if (root / "manifest.json").is_file() else None


@pytest.mark.skipif(_extension_root() is None, reason="extension tree absent (installed wheel)")
def test_expected_extension_version_pins_the_extension_tree() -> None:
    """`EXPECTED_EXTENSION_VERSION` is hand-written, so this is what keeps it
    honest: it must equal the version the extension tree actually ships, and the
    two manifests must agree with each other (a bump applied to one of them ships
    a zip whose version does not match what was promoted)."""
    root = _extension_root()
    assert root is not None
    manifest = json.loads((root / "manifest.json").read_text())
    package = json.loads((root / "package.json").read_text())
    assert manifest["version"] == EXPECTED_EXTENSION_VERSION
    assert package["version"] == EXPECTED_EXTENSION_VERSION
    assert MIN_SUPPORTED_PROTO <= PROTO_VERSION
    # The ownership floor must be a version this runtime would not refuse, and
    # it must be behind the expected head — otherwise the degrade branch could
    # never be reached by a real peer.
    assert extension_older(OWNERSHIP_MIN_EXTENSION_VERSION, EXPECTED_EXTENSION_VERSION)


@pytest.mark.skipif(_extension_root() is None, reason="extension tree absent (installed wheel)")
def test_generated_typescript_carries_the_same_note() -> None:
    """One spelling: the popup renders the template the Python constant defines,
    so the generated file is the contract between them."""
    root = _extension_root()
    assert root is not None
    generated = (root / "src" / "protocol.gen.ts").read_text()
    assert f"export const EXPECTED_EXTENSION_VERSION = '{EXPECTED_EXTENSION_VERSION}'" in generated
    assert f"export const EXTENSION_UPDATE_NOTE = '{EXTENSION_UPDATE_NOTE}'" in generated


# --- the advisory through the tool ----------------------------------------


def _live_state(version: str) -> state_store.BridgeState:
    return state_store.BridgeState(
        pid=1,
        port=4099,
        session_key="k" * 32,
        proto=PROTO_VERSION,
        extension_connected=True,
        extension_version=version,
        extension_proto=PROTO_VERSION,
        extension_update_available=extension_older(version, EXPECTED_EXTENSION_VERSION),
    )


def _note(monkeypatch: pytest.MonkeyPatch, version: str):
    """Drive the advisory helper with the discovery file stubbed.

    The lookup goes through the `state` MODULE (the tool imports it lazily), so
    patching the module attribute is what the call site actually reads.
    """
    from local_operator.browser_bridge import state as store
    from local_operator.harness.types import BrowserSurface
    from local_operator.tools import builtin

    monkeypatch.setattr(store, "read", lambda root=None: _live_state(version))
    surface = BrowserSurface()
    result = builtin._text("t", "browser", "Opened a tab.")
    return builtin._browser_update_note(surface, result), surface


def test_the_tool_appends_the_advisory_exactly_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tool results are re-billed every turn, so this is the one surface that
    must never repeat. The flag lives on the host-owned surface, which outlives
    the per-turn ToolContext — a flag on the context would reset every turn."""
    from local_operator.tools import builtin

    noted, surface = _note(monkeypatch, "0.1.10")
    assert extension_update_note("0.1.10") in noted.text
    assert noted.text.startswith("Opened a tab.")
    assert surface.extension_update_notified is True

    # The SAME surface, a second result: no second sentence.
    again = builtin._browser_update_note(surface, builtin._text("t", "browser", "Read a page."))
    assert again.text == "Read a page."


def test_the_tool_says_nothing_without_an_update(monkeypatch: pytest.MonkeyPatch) -> None:
    noted, surface = _note(monkeypatch, EXPECTED_EXTENSION_VERSION)
    assert noted.text == "Opened a tab."
    assert surface.extension_update_notified is False, "nothing was spent"


def test_the_tool_never_nags_over_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """ "nothing is blocked" beside an error reads as a contradiction, and the
    failure copy is already what the agent must act on."""
    from local_operator.browser_bridge import state as store
    from local_operator.harness.types import BrowserSurface
    from local_operator.tools import builtin

    monkeypatch.setattr(store, "read", lambda root=None: _live_state("0.1.10"))
    surface = BrowserSurface()
    result = builtin._browser_update_note(
        surface, builtin._error("t", "browser", "the site was refused")
    )
    assert result.text == "the site was refused"


class _DegradedResource:
    """The seams `execute_browser` drives for the retention verbs.

    ``params()`` asserts because the degraded path must FORK the owner lifecycle
    verbs and never the identity parameters: a fallback implemented as "stop
    sending owner params" would break the mid-version cases it exists for.
    """

    def __init__(self) -> None:
        self.generation = "g-test"
        self.record: dict[str, Any] = {
            "surface_id": "bridge:42:capability",
            "proof": "proof",
            "allocation_id": "allocation",
            "bridge_generations": ["g-test"],
            "state": "owned",
        }
        self.recovered = True
        self.lock = asyncio.Lock()
        self.finished = 0

    def initialize(self) -> None:
        return None

    async def recover(self) -> dict[str, Any]:
        raise AssertionError("a known-legacy link must not re-probe owner_recover")

    def ownership_mode(self) -> bool | None:
        return False

    def assert_current(self) -> None:
        return None

    def _save(self) -> None:
        return None

    def params(self) -> dict[str, Any]:
        raise AssertionError("the identity params are not forked by the fallback")

    async def finish_degraded(self):
        self.finished += 1
        self.record["surface_id"] = ""
        self.record["state"] = "closed"
        return resources.BrowserCleanupResult("closed")


def _degraded_context(resource: _DegradedResource):
    from local_operator.harness.types import BrowserSurface, ToolContext

    surface = BrowserSurface()
    surface.surface_id = "bridge:42:capability"
    surface.resource = resource  # type: ignore[assignment]
    return ToolContext(browser=surface)


@pytest.mark.asyncio
async def test_retain_degrades_to_a_local_record_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The extension cannot be ASKED to hold a tab, so the copy must say the
    retention is local — and the record must still carry it, because that is
    what fences the row in `lop browser tabs`."""
    from local_operator.tools import builtin

    resource = _DegradedResource()
    result = await builtin.execute_browser(
        "t",
        {"action": "retain", "text": "pending login"},
        None,
        None,
        _degraded_context(resource),
    )
    assert result.is_error is False
    assert "locally only" in result.text
    assert "cannot enforce" in result.text
    assert resource.record["retention"] == "pending login"


@pytest.mark.asyncio
async def test_release_in_degraded_mode_still_closes_a_terminal_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_operator.tools import builtin

    resource = _DegradedResource()
    resource.record["retention"] = "pending login"
    resource.record["terminal"] = "completed"
    context = _degraded_context(resource)
    result = await builtin.execute_browser("t", {"action": "release"}, None, None, context)
    assert result.is_error is False
    assert resource.finished == 1, "the tab must actually be closed, not stranded"
    assert resource.record["retention"] == ""
    assert resource.record["state"] == "closed"
    surface = context.browser
    assert surface is not None
    assert surface.surface_id == ""
