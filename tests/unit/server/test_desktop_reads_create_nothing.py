"""The desktop's read plane answers on a fresh install without creating one.

MEASURED, on an isolated config root, before the fix this file pins: ``GET
/v1/desktop/commands`` returned 200 and left ``<config>/network`` and
``<config>/network/networks`` on disk, so ``utils.desktop_mesh.has_any_network`` —
an ``is_dir`` probe and honest on its own — answered that this device was in a
mesh it had never joined. A renderer reading that answer draws a mesh surface;
the machine has no relay, no identity and no record.

``tests/unit/network/test_reads_create_nothing.py`` pins the resolvers that made
the directory. THIS FILE DRIVES THE REAL APP AND THE REAL ROUTES, because the
resolvers can be clean while a route still creates something else on the way, and
because the answer ``has_any_network`` gives is the thing the desktop acts on.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from local_operator.network import store, types
from local_operator.server.app import app
from local_operator.server.utils.desktop_mesh import has_any_network

#: A literal, as the sibling catalogue tests use: this is a test token the fixture
#: installs, never a real one (see ``test_desktop_command_catalogue.py``).
TOKEN = "[redacted]"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

NETWORK = "n_0123456789abcdef01234567"
SELF_DEVICE = "d_" + "a" * 32
PEER_DEVICE = "d_" + "b" * 32
PEER_NAME = "devon"


def _isolate(tmp_path: Path, monkeypatch) -> Path:
    """Point the app at a private config root, and return it.

    The config root is read from the environment on every call rather than resolved
    once at import (``paths.config_dir``), so this is enough for a route that builds
    its own root from ``config_dir()`` — which is exactly what the catalogue does.
    """
    root = tmp_path / ".local-operator"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    monkeypatch.delenv("LOCAL_OPERATOR_DESKTOP_ORIGINS", raising=False)
    return root


def _record(*, with_peer: bool) -> types.NetworkRecord:
    record = types.NetworkRecord(
        network_id=NETWORK,
        name="home-net",
        epoch=1,
        created_by=SELF_DEVICE,
        self_device_id=SELF_DEVICE,
        self_role="admin",
        self_capabilities=sorted(types.capabilities_for_role("admin")),
    )
    if with_peer:
        record.members.append(
            types.MemberRecord(device_id=PEER_DEVICE, name=PEER_NAME, role="drive")
        )
    return record


def test_the_command_catalogue_does_not_create_the_network_plane(tmp_path, monkeypatch) -> None:
    """The reproduction, as a test: this GET created the directory it reported on.

    ``/v1/desktop/commands`` is also the mesh's zero-peer cost gate in reverse — the
    desktop asks for the command list on every launch, so a read that created the
    plane made every install look meshed from its first frame.
    """
    root = _isolate(tmp_path, monkeypatch)

    with TestClient(app) as client:
        assert not (root / "network").exists(), "the app's own startup created the plane"
        response = client.get("/v1/desktop/commands", headers=HEADERS)
        assert response.status_code == 200
        assert (root / "network").exists() is False, "the catalogue GET created the network plane"

    # The two spellings of the same lie: the directory, and the answer the desktop's
    # mesh surface keys on.
    assert has_any_network(root) is False


def test_the_mesh_reads_answer_empty_and_create_nothing_on_a_fresh_install(
    tmp_path, monkeypatch
) -> None:
    """The same surface's other reads, pinned against a regression in either direction:
    they must keep answering empty, and they must keep creating nothing.

    ``has_any_network`` is what makes them cheap (no socket, no file); a read that
    created the plane would answer its next poll from a mesh it invented.
    """
    root = _isolate(tmp_path, monkeypatch)

    with TestClient(app) as client:
        peers = client.get("/v1/desktop/peers", headers=HEADERS)
        networks = client.get("/v1/desktop/networks", headers=HEADERS)
        assert (peers.status_code, networks.status_code) == (200, 200)
        assert peers.json()["result"]["peers"] == []
        assert networks.json()["result"]["networks"] == []

    assert (root / "network").exists() is False
    assert has_any_network(root) is False


def test_the_other_read_routes_do_not_create_the_run_plane_either(tmp_path, monkeypatch) -> None:
    """THE SWEEP, AS A TEST — and the reason it exists is that the first version of
    this file proved ``0 of N`` for ``network/`` while ``run/`` still grew.

    Measured on a fresh isolated root (review round 1, R1-1): ``GET
    /v1/desktop/info``, ``/networks``, ``/runtimes``, ``/sessions`` and
    ``/sessions/{id}`` created ``run/mobile``, ``run/peers`` and ``run/viewers``,
    because each reached a scan and ``registry.scan`` opened with a mkdir. A fix
    scoped to ``network/`` would have left all five, so the plane this file guards
    is BOTH namespaces.

    Status is deliberately not asserted: the point is the filesystem, and these
    routes answer differently on a machine with no relay, no runtime and no
    session (200 with an empty read, or a refusal sentence). What must hold for
    every one of them is that the tree afterwards is the tree before it.
    """
    root = _isolate(tmp_path, monkeypatch)
    routes = (
        "/v1/desktop/commands",
        "/v1/desktop/info",
        "/v1/desktop/networks",
        "/v1/desktop/peers",
        "/v1/desktop/runtimes",
        "/v1/desktop/sessions",
        "/v1/desktop/sessions/aaaaaaaaaaaa",
    )

    with TestClient(app) as client:
        for route in routes:
            before = sorted(str(path.relative_to(root)) for path in root.rglob("*"))
            client.get(route, headers=HEADERS)
            after = sorted(str(path.relative_to(root)) for path in root.rglob("*"))
            assert after == before, f"{route} created {sorted(set(after) - set(before))}"

    assert (root / "network").exists() is False
    assert (root / "run").exists() is False


def test_the_probe_and_the_vocabulary_still_read_a_real_network(tmp_path, monkeypatch) -> None:
    """The fix must not turn an honest probe into a permanently false one.

    A record written by the real writer answers ``True``, and the peer vocabulary the
    catalogue publishes (``/new remote <name>``) comes from that record — the read
    path was always correct; what was wrong was that it could not be reached without
    writing. Asserted THROUGH THE ENDPOINT, because that field is what a renderer
    completes from, and an empty vocabulary is the one answer it cannot guess at.

    Deliberately not asserted here: what the mesh reads answer once a record exists
    but no relay does. That is the refusal ladder's business (a 503 naming the
    remedy), it needs no network plane created or not created to be right, and
    pinning it here would make this file fail for a reason that is not its subject.
    """
    root = _isolate(tmp_path, monkeypatch)
    store.save(_record(with_peer=True), root)

    assert has_any_network(root) is True

    with TestClient(app) as client:
        rows = client.get("/v1/desktop/commands", headers=HEADERS).json()["result"]["commands"]
    remote = next(row for row in rows if row["name"] == "new")
    # Both halves of the vocabulary: the name a person types, then the id as the
    # fallback for a device that was never named (``known_peer_names``).
    assert remote["argument_words"] == [PEER_NAME, PEER_DEVICE]
