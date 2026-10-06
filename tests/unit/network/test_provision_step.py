"""``step_provision`` — the provisioning transaction (S1), owner side.

WHAT IT PINS, cell by cell, against ``mesh-consent-provisioning.md`` §1 and §9.2's
S1 row:

* receipts/order: the step settles between ``grants`` and ``verify``, folds to
  ``connected`` with the run, and its structured ``data`` carries every
  sub-outcome a reviewer needs (grants, capability, pushes, placement read,
  git, the MCP needs-list);
* the writes: the per-class default HELD rows come from the OFFER's own
  projection (not a second classifier), each through ``grant``, and the
  ``broker_credential`` capability lands on the node's member row;
* idempotent re-run: a retry re-runs every write without duplicating a holder
  and without changing what the step reports;
* refusal arms: device missing, network unknown, member row missing, placement
  write refuses, and the read-role gate (§1.3: no credential rows);
* the delivery: the node's placement pull is driven IN-RUN and its own listing
  is read back — the note's "the node's first ``lop network credentials`` is
  already true";
* the seed: the node's git identity is filled from this device's only where it
  is MISSING (a set identity on the node is left alone);
* the pushes: one forced definitions + MCP push through this device's own CLI,
  recorded rather than fatal when the node cannot answer.

THE HARNESS is ``test_onboard.py``'s (scripted ``FakeTransport``, the fake
approvals module, the ``isolated`` config root): the steps' own machinery is not
re-derived here. The two-config-roots cell at the bottom is the one that swaps
the script for real relays and real node commands.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import pytest

from local_operator.network import onboard, onboard_approvals, relay
from local_operator.network import store as network_store
from local_operator.network import types
from local_operator.network.credentials import placement as placement_mod
from local_operator.network.credentials.types import BROKER_CAPABILITY
from local_operator.network.types import MeshRefusal
from tests.unit.network.test_onboard import (  # noqa: F401 — the harness, by import
    _REAL_LIST_NETWORKS,
    HAPPY_OUTPUTS,
    FakeApprovals,
    FakeTransport,
    _happy_run_local,
    _install_fakes,
    _record,
    _result,
    isolated,
)
from tests.unit.network.test_pair_offer import _seed, _seed_mcp_json
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixtures by import
    _pair,
    devices,
)


@pytest.fixture()
def provision_root(isolated: Path) -> Path:  # noqa: F811 — test_onboard's rig, by import
    """That file's isolated config root, under a local name so cells do not shadow it."""
    return isolated


OPERATOR = "d_mac"
NODE = "d_node"
NETWORK_ID = "n_1"


# ---------------------------------------------------------------------------
# Rig: a REAL operator-side record, so the capability write is executed
# ---------------------------------------------------------------------------


def _operator_record(root: Path) -> Any:
    """A real owner record on disk with the node already a member row (§1.3).

    The ``isolated`` fixture fakes ``list_networks`` for the older cells; the
    provisioning step MUTATES the record (the capability pair write), so cells
    that assert it re-patch to the real list and save a real record here.
    """
    record = types.NetworkRecord(
        network_id=NETWORK_ID,
        name="damian-mesh",
        created_by=OPERATOR,
        self_device_id=OPERATOR,
        self_role="admin",
        self_capabilities=sorted(types.capabilities_for_role("admin")),
        epoch=1,
    )
    relay.admit(
        record,
        device_id=OPERATOR,
        public_key="k-mac",
        name="this-mac",
        role="admin",
        added_by=OPERATOR,
        added_via="self",
        capabilities=sorted(types.capabilities_for_role("admin")),
        persist=False,
    )
    relay.admit(
        record,
        device_id=NODE,
        public_key="k-node",
        name="cloud-node-1",
        role="drive",
        added_by=OPERATOR,
        capabilities=sorted(types.capabilities_for_role("drive")),
        persist=False,
    )
    network_store.save(record, root)
    return record


def _real_networks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Read this device's REAL records (undoing the fixture's pinned fake)."""
    monkeypatch.setattr(network_store, "list_networks", _REAL_LIST_NETWORKS)


def _view(**overrides: Any) -> onboard_approvals.ApprovalView:
    device = {"device_id": NODE, "name": "cloud-node-1"}
    what = {"network_id": NETWORK_ID}
    device.update(overrides.pop("device", {}) or {})
    what.update(overrides.pop("what", {}) or {})
    return onboard_approvals.ApprovalView(
        approval_id="ap_prov",
        kind="device_onboard",
        state="approved",
        device=device,
        what=what,
        run_id="run_prov",
    )


def _run(
    view: onboard_approvals.ApprovalView, transport: Any, run_local: Any
) -> onboard.OnboardRun:
    return onboard.OnboardRun(
        "ap_prov",
        view=view,
        transport=transport,
        resolve=lambda ref: onboard.ResolvedCredential(kind="file", label="x"),
        local_cli=["lop"],
        run_local=run_local,
    )


def _pushes_ok(recorded: list[list[str]]):
    """A ``run_local`` that answers the two pushes with the CLI's own payload."""

    def run_local(argv: list[str], *, timeout: float) -> onboard.CommandResult:
        recorded.append(list(argv))
        if "definitions" in argv or "mcp" in argv:
            return _result((), stdout=json.dumps({"ok": True, "peers": [{"device_id": NODE}]}))
        return _result(())

    return run_local


# ---------------------------------------------------------------------------
# The writes, the receipt, idempotency
# ---------------------------------------------------------------------------


def test_the_transaction_writes_the_offer_sets_rows_and_the_capability(
    provision_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One run: the offer's default set gets holder rows through ``grant``, the
    capability lands on the node's row, and the receipt reports both."""
    _seed(provision_root, "openai", {"refresh": "r", "access": "a", "email": "d@example.com"})
    _operator_record(provision_root)
    _real_networks(monkeypatch)
    recorded: list[list[str]] = []
    transport = FakeTransport(
        outputs=[
            ("network credentials", {"stdout": json.dumps(_listing(held=["openai"]))}),
            ("mcp state", {"stdout": json.dumps({"ok": True, "servers": []})}),
        ]
    )
    outcome = _run(_view(), transport, _pushes_ok(recorded)).step_provision()

    assert outcome.ok, outcome.detail
    assert outcome.data["grants"]["granted"] == ["openai"]
    assert outcome.data["grants"]["skipped"] == []
    assert outcome.data["capability"] == {"set": True, "changed": True}
    assert outcome.data["placement"]["refreshed"] is True
    assert outcome.data["placement"]["missing"] == []
    # The pushes went through THIS device's own CLI, one forced push each.
    assert any("definitions" in argv and "push" in argv for argv in recorded)
    assert any("mcp" in argv and "push" in argv for argv in recorded)

    document = placement_mod.PlacementDocument.load(
        NETWORK_ID, provision_root, self_device=OPERATOR
    )
    entry = document.entry("openai")
    assert entry is not None
    assert entry.owner_device == OPERATOR
    assert entry.holders[0].device == OPERATOR
    assert entry.holders[-1].device == NODE
    # §1.2's pair write, on the record: the node may now DIAL the broker.
    record = types.NetworkRecord.from_json(network_store.load(NETWORK_ID, provision_root).to_json())
    node_member = record.member(NODE)
    assert node_member is not None
    assert BROKER_CAPABILITY in node_member.capabilities


def test_a_re_run_is_idempotent(provision_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every write re-runs clean: one holder row, the capability already set,
    and the same granted list — the resumable step's contract."""
    _seed(provision_root, "openai", {"refresh": "r", "access": "a", "email": "d@example.com"})
    _operator_record(provision_root)
    _real_networks(monkeypatch)
    recorded: list[list[str]] = []
    transport = FakeTransport(
        outputs=[
            ("network credentials", {"stdout": json.dumps(_listing(held=["openai"]))}),
            ("mcp state", {"stdout": json.dumps({"ok": True, "servers": []})}),
        ]
    )
    run = _run(_view(), transport, _pushes_ok(recorded))

    first = run.step_provision()
    second = run.step_provision()

    assert first.ok and second.ok, second.detail
    assert first.data["grants"]["granted"] == second.data["grants"]["granted"] == ["openai"]
    assert second.data["capability"] == {"set": True, "changed": False}
    document = placement_mod.PlacementDocument.load(
        NETWORK_ID, provision_root, self_device=OPERATOR
    )
    entry = document.entry("openai")
    assert entry is not None
    holders = [holder.device for holder in entry.holders]
    assert holders.count(NODE) == 1, holders


def test_the_receipt_settles_in_order_and_folds_to_connected(
    provision_root: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The full machine: ``provision`` sits between ``grants`` and ``verify``,
    carries its data on the run payload, and the record folds to connected."""
    _install_fakes(monkeypatch)
    _seed(provision_root, "openai", {"refresh": "r", "access": "a", "email": "d@example.com"})
    _operator_record(provision_root)
    _real_networks(monkeypatch)
    token = tmp_path / "invite.token"
    token.write_text("token-bytes", encoding="utf-8")
    record = _record()
    fake = FakeApprovals(record)
    monkeypatch.setattr(onboard_approvals, "_module", lambda: fake)
    transport = FakeTransport(
        outputs=HAPPY_OUTPUTS
        + [
            ("network credentials", {"stdout": json.dumps(_listing(held=["openai"]))}),
            ("mcp state", {"stdout": json.dumps({"ok": True, "servers": []})}),
        ]
    )

    payload = onboard.execute_approval(
        "ap_aaaa1111",
        transport=transport,
        resolve=lambda ref: onboard.ResolvedCredential(kind="file", label=ref["ref"]),
        local_cli=["lop"],
        run_local=_happy_run_local(token),
    )

    assert payload["ok"] is True, payload
    assert [row["step"] for row in payload["steps"]] == list(onboard.STEP_NAMES)
    provision = next(row for row in payload["steps"] if row["step"] == "provision")
    assert provision["ok"] is True
    assert provision["data"]["grants"]["granted"] == ["openai"]
    assert provision["data"]["device_id"] == NODE
    assert provision["detail"].startswith("provisioned cloud-node-1:")


# ---------------------------------------------------------------------------
# Refusal arms (§9.2: "device missing, network unknown, placement write refuses")
# ---------------------------------------------------------------------------


def test_no_device_refuses_before_any_write(
    provision_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _real_networks(monkeypatch)
    outcome = _run(
        _view(device={"device_id": "", "name": ""}),
        FakeTransport(outputs=[("identity show", {"stdout": json.dumps({"ok": True})})]),
        _pushes_ok([]),
    ).step_provision()
    assert outcome.ok is False
    assert "device" in outcome.detail
    # Nothing was written: no placement doc for the network exists.
    assert not placement_mod.placement_path(NETWORK_ID, provision_root).exists()


def test_an_unknown_network_refuses(provision_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _real_networks(monkeypatch)
    outcome = _run(
        _view(what={"network_id": "n_elsewhere"}),
        FakeTransport(),
        _pushes_ok([]),
    ).step_provision()
    assert outcome.ok is False
    assert "does not know the network" in outcome.detail


def test_a_missing_member_row_refuses(
    provision_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _operator_record(provision_root)
    record.members = [row for row in record.members if row.device_id != NODE]
    network_store.save(record, provision_root)
    _real_networks(monkeypatch)
    outcome = _run(_view(), FakeTransport(), _pushes_ok([])).step_provision()
    assert outcome.ok is False
    assert "active member" in outcome.detail


def test_a_placement_write_refusal_fails_the_step(
    provision_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(provision_root, "openai", {"refresh": "r", "access": "a", "email": "d@example.com"})
    _operator_record(provision_root)
    _real_networks(monkeypatch)

    def _refuse(*args: Any, **kwargs: Any) -> Any:
        raise MeshRefusal("busy", "another writer held the sharing list")

    monkeypatch.setattr(placement_mod, "mutate", _refuse)
    outcome = _run(_view(), FakeTransport(), _pushes_ok([])).step_provision()
    assert outcome.ok is False
    assert "refused the write" in outcome.detail


def test_a_read_role_member_gets_no_credential_rows(
    provision_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§1.3: definitions, MCP rows and git identity only — a `read` member
    holds no prompt/borrow path, so credential rows would widen authority for
    no work."""
    _seed(provision_root, "openai", {"refresh": "r", "access": "a", "email": "d@example.com"})
    record = _operator_record(provision_root)
    record.member(NODE).role = "read"
    network_store.save(record, provision_root)
    _real_networks(monkeypatch)
    transport = FakeTransport(
        outputs=[("mcp state", {"stdout": json.dumps({"ok": True, "servers": []})})]
    )
    outcome = _run(_view(), transport, _pushes_ok([])).step_provision()

    assert outcome.ok, outcome.detail
    assert outcome.data["grants"] == {"granted": [], "skipped": []}
    assert "read" in outcome.data["grants_note"]
    assert outcome.data["capability"] == {"set": False, "changed": False}
    document = placement_mod.PlacementDocument.load(
        NETWORK_ID, provision_root, self_device=OPERATOR
    )
    assert document.entry("openai") is None
    # The pushes still ran: definitions and MCP rows are role-independent.
    assert outcome.data["definitions"]["ok"] is True


# ---------------------------------------------------------------------------
# The delivery read-back, the pushes' failure shape, git, and the needs list
# ---------------------------------------------------------------------------


def test_the_listing_read_back_reports_shares_the_node_does_not_have_yet(
    provision_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The confirmation is the node's OWN listing: a key missing from it is
    named, and a relay that did not answer is its own sentence — never silent."""
    _seed(provision_root, "openai", {"refresh": "r", "access": "a", "email": "d@example.com"})
    _operator_record(provision_root)
    _real_networks(monkeypatch)
    transport = FakeTransport(
        outputs=[
            ("network credentials", {"stdout": json.dumps(_listing(held=[]))}),
            ("mcp state", {"stdout": json.dumps({"ok": True, "servers": []})}),
        ]
    )
    outcome = _run(_view(), transport, _pushes_ok([])).step_provision()
    assert outcome.ok, outcome.detail
    assert outcome.data["placement"]["missing"] == ["openai"]
    assert "caveat:" in outcome.detail and "openai" in outcome.detail

    # A relay that never answered is a different sentence, not the same one.
    transport2 = FakeTransport(
        outputs=[
            ("network credentials", {"stdout": json.dumps({"ok": True, "refreshed": False})}),
            ("mcp state", {"stdout": json.dumps({"ok": True, "servers": []})}),
        ]
    )
    outcome2 = _run(_view(), transport2, _pushes_ok([])).step_provision()
    assert outcome2.ok and outcome2.data["placement"]["refreshed"] is False
    assert "did not answer" in outcome2.detail


def test_a_push_that_cannot_reach_the_node_is_recorded_not_fatal(
    provision_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(provision_root, "openai", {"refresh": "r", "access": "a", "email": "d@example.com"})
    _operator_record(provision_root)
    _real_networks(monkeypatch)

    def run_local(argv: list[str], *, timeout: float) -> onboard.CommandResult:
        return _result(
            (),
            rc=1,
            stdout=json.dumps(
                {"ok": False, "code": "relay_unavailable", "message": "no relay is running"}
            ),
        )

    transport = FakeTransport(
        outputs=[
            ("network credentials", {"stdout": json.dumps(_listing(held=["openai"]))}),
            ("mcp state", {"stdout": json.dumps({"ok": True, "servers": []})}),
        ]
    )
    outcome = _run(_view(), transport, run_local).step_provision()
    assert outcome.ok, outcome.detail
    assert outcome.data["definitions"]["ok"] is False
    assert outcome.data["definitions"]["code"] == "relay_unavailable"
    assert "definitions not pushed" in outcome.detail


def test_the_git_identity_is_seeded_only_where_the_node_has_none(
    provision_root: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The seed fills the GAP the readiness row dead-ends on; an identity the
    node already commits under is not the transaction's to overwrite."""
    _seed(provision_root, "openai", {"refresh": "r", "access": "a", "email": "d@example.com"})
    _operator_record(provision_root)
    _real_networks(monkeypatch)
    owner_home = tmp_path / "owner-home"
    owner_home.mkdir()
    (owner_home / ".gitconfig").write_text(
        "[user]\n\tname = Damian Tran\n\temail = damian@example.com\n", encoding="utf-8"
    )
    monkeypatch.setenv("HOME", str(owner_home))
    transport = FakeTransport(
        outputs=[
            ("git=", {"stdout": "git=yes\nname=\nemail=\n"}),
            ("network credentials", {"stdout": json.dumps(_listing(held=["openai"]))}),
            ("mcp state", {"stdout": json.dumps({"ok": True, "servers": []})}),
        ]
    )
    outcome = _run(_view(), transport, _pushes_ok([])).step_provision()
    assert outcome.ok, outcome.detail
    assert outcome.data["git"]["seeded"] == ["user.name", "user.email"]
    writes = [
        " ".join(str(part) for part in call[1]) for call in transport.calls if call[0] == "run"
    ]
    assert any("git config --global user.name" in line and "Damian Tran" in line for line in writes)
    assert any("user.email" in line and "damian@example.com" in line for line in writes)

    # The node already has its own identity: nothing is written over it.
    transport2 = FakeTransport(
        outputs=[
            ("git=", {"stdout": "git=yes\nname = Node N. Node\nemail = node@example.com\n"}),
            ("network credentials", {"stdout": json.dumps(_listing(held=["openai"]))}),
            ("mcp state", {"stdout": json.dumps({"ok": True, "servers": []})}),
        ]
    )
    outcome2 = _run(_view(), transport2, _pushes_ok([])).step_provision()
    assert outcome2.ok
    assert outcome2.data["git"]["seeded"] == []
    assert outcome2.data["git"]["already_set"] == ["user.name", "user.email"]
    writes2 = [
        " ".join(str(part) for part in call[1]) for call in transport2.calls if call[0] == "run"
    ]
    assert not any("git config --global user.name" in line for line in writes2)


def test_the_mcp_needs_list_is_recorded_for_the_later_copy_decision(
    provision_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one verification read of §1.2 row 4: "the keys to set" on the node,
    recorded — the needs list S4's copy-set default reads."""
    _seed(provision_root, "openai", {"refresh": "r", "access": "a", "email": "d@example.com"})
    _operator_record(provision_root)
    _real_networks(monkeypatch)
    transport = FakeTransport(
        outputs=[
            ("network credentials", {"stdout": json.dumps(_listing(held=["openai"]))}),
            (
                "mcp state",
                {
                    "stdout": json.dumps(
                        {
                            "ok": True,
                            "servers": [
                                {
                                    "name": "gitlab",
                                    "refs": [
                                        {"id": "GITLAB_TOKEN", "set": False},
                                        {"id": "OTHER", "set": True},
                                    ],
                                }
                            ],
                        }
                    )
                },
            ),
        ]
    )
    outcome = _run(_view(), transport, _pushes_ok([])).step_provision()
    assert outcome.ok, outcome.detail
    assert outcome.data["mcp_state"]["keys_needed"] == ["GITLAB_TOKEN"]
    assert "GITLAB_TOKEN" in outcome.detail


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _listing(held: list[str], *, refreshed: bool = True) -> dict[str, Any]:
    """A ``lop network credentials --json`` payload as the node would render it."""
    rows = [
        {
            "credential_name": key,
            "kind": "oauth-rotating",
            "owner_device": OPERATOR,
            "owner_device_name": "this-mac",
            "identity_label": "",
            "owned_here": False,
            "holders": [
                {"device": OPERATOR, "name": "this-mac", "scope": "device"},
                {"device": NODE, "name": "cloud-node-1", "scope": "session"},
            ],
        }
        for key in held
    ]
    return {
        "ok": True,
        "self_device": NODE,
        "refreshed": refreshed,
        "newly_borrowable": [],
        "networks": [{"network_id": NETWORK_ID, "network": "damian-mesh", "credentials": rows}],
    }


# ---------------------------------------------------------------------------
# Two config roots, two real relays, real node commands
# ---------------------------------------------------------------------------


#: Commands the transaction issues to the NODE, run for real in this cell; every
#: other remote command answers from the scripted happy path (its own suite owns
#: it). The probes are the cheap system facts the scripted steps consume.
_NODE_SIDE_TOKENS = ("network credentials", "network mcp state", "git config", "command -v git")


class _SpawningTransport(FakeTransport):
    """The scripted transport, with the provisioning step's commands REAL.

    Anything matching ``_NODE_SIDE_TOKENS`` is executed as a real subprocess with
    the NODE's own HOME and config root (``env -i``-style: only the four keys the
    node needs), so the two-config-roots claim is held by the node's own CLI,
    store and files — never by this process's ambient state.
    """

    def __init__(self, *, node_env: dict[str, str], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.node_env = dict(node_env)

    def run(
        self, argv: Sequence[str], *, timeout: float, stdin: bytes | None = None
    ) -> onboard.CommandResult:
        command = " ".join(str(part) for part in argv)
        if any(token in command for token in _NODE_SIDE_TOKENS):
            done = subprocess.run(
                ["sh", "-c", command],
                env=self.node_env,
                capture_output=True,
                timeout=timeout,
            )
            self.calls.append(("run", tuple(str(part) for part in argv)))
            return onboard.CommandResult(
                tuple(str(part) for part in argv),
                done.returncode,
                done.stdout.decode("utf-8", "replace"),
                done.stderr.decode("utf-8", "replace"),
                at=0.0,
            )
        return super().run(argv, timeout=timeout, stdin=stdin)


def _e2e_run_local(scripted: Any, a_env: dict[str, str]) -> Any:
    """The runner's LOCAL calls: the two pushes run for real (this device's own
    root and relay), everything else answers from the scripted happy path."""

    def run_local(argv: list[str], *, timeout: float) -> onboard.CommandResult:
        joined = " ".join(str(part) for part in argv)
        if "push" in joined:
            done = subprocess.run(
                [str(part) for part in argv], env=a_env, capture_output=True, timeout=timeout
            )
            return onboard.CommandResult(
                tuple(str(part) for part in argv),
                done.returncode,
                done.stdout.decode("utf-8", "replace"),
                done.stderr.decode("utf-8", "replace"),
                at=0.0,
            )
        return scripted(argv, timeout=timeout)

    return run_local


@pytest.fixture()
def two_roots(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Owner A and node B: two real relays on loopback, both dialable.

    ``devices`` gives the two serve-shaped relays and the join machinery; this
    fixture adds what the transaction needs on top of it: B LISTENING (the
    pushes dial the node — the pairing only dials A), both copies carrying the
    other's endpoint (the pairing leaves neither), an owner login for the share,
    an owner MCP server for the push, an owner git identity for the seed, and
    the ambient root pinned to the OWNER once the ceremony is over.
    """
    server_a, server_b, host_a, port_a = request.getfixturevalue("devices")
    record, _host, _port = _pair((server_a, server_b, host_a, port_a), monkeypatch, role="drive")
    host_b, port_b = server_b.bind()
    server_b.bind_control()
    server_b.start()
    with network_store.mutate(record.network_id, server_a.root) as copy:
        node_row = copy.member(server_b.identity.device_id)
        assert node_row is not None
        node_row.endpoints = [f"{host_b}:{port_b}"]
        network_store.save(copy, server_a.root)
    with network_store.mutate(record.network_id, server_b.root) as copy:
        owner_row = copy.member(server_a.identity.device_id)
        assert owner_row is not None
        owner_row.endpoints = [f"{host_a}:{port_a}"]
        network_store.save(copy, server_b.root)
    _seed(server_a.root, "openai", {"refresh": "r", "access": "a", "email": "d@example.com"})
    _seed_mcp_json(server_a.root, {"tools": {"type": "sse", "url": "https://mcp.example/sse"}})
    a_home = server_a.root.parent / "a-home"
    b_home = server_b.root.parent / "b-home"
    a_home.mkdir(parents=True)
    b_home.mkdir(parents=True)
    (a_home / ".gitconfig").write_text(
        "[user]\n\tname = Damian T\n\temail = damian@example.test\n", encoding="utf-8"
    )
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_a.root))
    monkeypatch.setenv("HOME", str(a_home))
    venv_bin = str(Path(sys.executable).parent)
    common = {"PATH": f"{venv_bin}:{os.environ.get('PATH', '')}", "TERM": "xterm-256color"}
    return SimpleNamespace(
        a=server_a,
        b=server_b,
        network_id=record.network_id,
        owner_id=server_a.identity.device_id,
        node_id=server_b.identity.device_id,
        a_root=server_a.root,
        b_root=server_b.root,
        a_home=a_home,
        b_home=b_home,
        a_env={**common, "HOME": str(a_home), "LOCAL_OPERATOR_CONFIG_DIR": str(server_a.root)},
        b_env={**common, "HOME": str(b_home), "LOCAL_OPERATOR_CONFIG_DIR": str(server_b.root)},
    )


def test_two_config_roots_the_node_holds_the_share_before_the_run_returns(
    two_roots: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE S1 GATE (the note's two-config-roots e2e), with the script removed
    from everything the transaction itself does:

    the earlier steps answer from the scripted harness (their own suites own
    them); the transaction's pushes run through THIS device's real CLI and
    relay (leg A → B), and the node's pull, git seed and MCP report run through
    the NODE's real CLI and relay (leg B → A). The assertions are the note's
    own: the placement document is on the node before the run returns, the
    capability pair write is on the owner's record, and the node's identity is
    seeded from the owner's — the Radient observation rides the PR text.
    """
    _install_fakes(monkeypatch)
    token = two_roots.a_root.parent / "invite.token"
    token.write_text("token-bytes", encoding="utf-8")
    record = _record(network_id=two_roots.network_id, role="drive")
    record["device"]["device_id"] = two_roots.node_id
    record["device"]["name"] = "cloud-node-1"
    fake = FakeApprovals(record)
    monkeypatch.setattr(onboard_approvals, "_module", lambda: fake)
    transport = _SpawningTransport(
        node_env=two_roots.b_env,
        outputs=HAPPY_OUTPUTS
        + [
            (
                "member grant",
                {
                    "stdout": json.dumps(
                        {
                            "ok": True,
                            "capabilities": ["approve", "unattended"],
                            "applied": True,
                        }
                    )
                },
            )
        ],
    )

    payload = onboard.execute_approval(
        "ap_aaaa1111",
        transport=transport,
        resolve=lambda ref: onboard.ResolvedCredential(kind="file", label=ref["ref"]),
        local_cli=["lop"],
        run_local=_e2e_run_local(
            _happy_run_local(
                token,
                network_id=two_roots.network_id,
                network_name="damian-mesh",
            ),
            two_roots.a_env,
        ),
    )

    assert payload["ok"] is True, payload.get("error")
    provision = next(row for row in payload["steps"] if row["step"] == "provision")
    assert provision["ok"] is True, provision["detail"]
    data = provision["data"]
    assert data["grants"]["granted"] == ["openai"]
    assert data["placement"]["refreshed"] is True
    assert data["placement"]["missing"] == []
    assert data["definitions"]["ok"] is True
    assert data["mcp"]["ok"] is True
    assert data["git"]["seeded"] == ["user.name", "user.email"]

    # THE CLAIM: the document is ON the node before the run returns — written by
    # the node's own relay, from the node's own pull, within the run.
    node_doc = placement_mod.PlacementDocument.load(
        two_roots.network_id, two_roots.b_root, self_device=two_roots.node_id
    )
    entry = node_doc.entry("openai")
    assert entry is not None, node_doc.entries
    assert entry.owner_device == two_roots.owner_id
    assert two_roots.node_id in [holder.device for holder in entry.holders]

    # The capability pair write, on the owner's record, read back from disk.
    owner_record = network_store.load(two_roots.network_id, two_roots.a_root)
    node_member = owner_record.member(two_roots.node_id)
    assert node_member is not None
    assert BROKER_CAPABILITY in node_member.capabilities

    # The seed landed in the node's OWN git config files.
    seeded = (two_roots.b_home / ".gitconfig").read_text(encoding="utf-8")
    assert "Damian T" in seeded and "damian@example.test" in seeded

    # The MCP server row is on the node's disk, pushed over the real link.
    assert "tools" in (two_roots.b_root / "mcp.json").read_text(encoding="utf-8")
