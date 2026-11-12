"""A CHANGED GitHub designation reaches the borrower, in order, with no restart.

THE FIELD FINDING (remote-session parity, 2026-11-12): the head Mac widened
``network.credentials.github.repositories`` to two repositories; on the node
``lop network credential refresh github`` answered ``ok`` and the node's list
stayed at one, and its audit trail FLAPPED between the old and new list. Two
causes, both pinned here:

* ``refresh`` re-minted at the owner and then DROPPED the grant, so its
  narrowing never reached the node's config (``cli._cmd_credential_refresh``);
* every session process keeps its own in-memory grant cache, and a grant
  cached before the change kept re-applying the OLD list on every command,
  racing processes holding a fresh grant. Deliveries carried no order.

The fix stamps every delivery with ``{digest, generation}`` (owner side,
``github.grant_narrowing``) and makes the borrower apply only the newest per
owner (``github.remember_delivered_repositories``). The second half of this
file drives the whole thing over TWO REAL RELAYS through the shipped CLI.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from local_operator.network import store
from local_operator.network.credentials import github as github_mod
from local_operator.network.credentials import placement as placement_mod
from local_operator.network.credentials.types import Grant
from tests.unit.network.test_credentials_github import (  # noqa: F401 — fixtures by import
    BORROWER_DEVICE,
    OWNER_DEVICE,
    SCRATCH,
    _ask_grant,
    _Env,
    _FakeGithub,
    _no_secret_broker_daemon,
    _write_config,
    app_keypair,
    env,
    github_api,
    owner,
)
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixtures by import
    _pair,
    devices,
)

SECOND = "damianvtran/scratch-two"
NET = "n_gh"


def _owner_rig(request: pytest.FixtureRequest) -> tuple[Any, _FakeGithub, _Env]:
    """The App-arm owner fixtures, resolved BY NAME (a parameter of the same name
    would shadow the import that makes each fixture reachable)."""
    return (
        request.getfixturevalue("owner"),
        request.getfixturevalue("github_api"),
        request.getfixturevalue("env"),
    )


def _node(tmp_path: Path, repositories: list[str] | None) -> Path:
    root = tmp_path / "node"
    root.mkdir()
    if repositories is not None:
        _write_config(root, repositories)
    return root


def _deliver(root: Path, narrowing: dict[str, Any]) -> dict[str, Any]:
    return github_mod.remember_delivered_repositories(
        narrowing.get("repositories"),
        root=root,
        self_device=BORROWER_DEVICE,
        owner_device=OWNER_DEVICE,
        network_id=NET,
        digest=str(narrowing.get("digest") or ""),
        generation=narrowing.get("generation") or 0,
    )


def _applied_rows(root: Path) -> list[dict[str, Any]]:
    from local_operator.network.audit import AuditLog

    return [
        row
        for row in AuditLog(root).tail(200, network_id=NET)
        if row.get("event") == "credential.narrowing_applied"
    ]


def _helper_serves(root: Path, repository: str) -> bool:
    """The REAL helper core against the list this device holds now."""
    reply = github_mod.git_helper_reply(
        "get",
        ["protocol=https", "host=github.com", f"path={repository}.git", ""],
        token="tok-under-test",
        repositories=github_mod.repositories_for(root),
    )
    return "password=tok-under-test" in reply


# ---------------------------------------------------------------------------
# Owner (App arm, real minter against a fake GitHub) -> borrower config
# ---------------------------------------------------------------------------


def test_widen_then_narrow_propagates_and_the_app_arm_re_mints(
    request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    """Borrow with [A]; rig widens to [A,B]; a refresh re-mints for both and the
    borrower allows B; rig narrows back to [A]; the borrower refuses B again."""
    rig, api, home = _owner_rig(request)
    node = _node(tmp_path, None)
    first = _ask_grant(rig, BORROWER_DEVICE)
    assert _deliver(node, first["narrowing"])["changed"] is True
    assert _helper_serves(node, SCRATCH) and not _helper_serves(node, SECOND)

    _write_config(home.root, [SCRATCH, SECOND])
    widened = _ask_grant(rig, BORROWER_DEVICE)
    # THE APP ARM NARROWS SERVER-SIDE: the re-mint asked GitHub for both names.
    assert api.mint_requests[-1]["repositories"] == ["scratch", "scratch-two"]
    outcome = _deliver(node, widened["narrowing"])
    assert outcome["changed"] is True and outcome["stale"] is False, outcome
    assert outcome["previous"] == [SCRATCH]
    assert _helper_serves(node, SECOND), "the widened designation did not reach the helper"

    # Narrowing is the security-relevant direction: removal must land too.
    _write_config(home.root, [SCRATCH])
    narrowed = _ask_grant(rig, BORROWER_DEVICE)
    assert api.mint_requests[-1]["repositories"] == ["scratch"]
    assert _deliver(node, narrowed["narrowing"])["changed"] is True
    assert not _helper_serves(node, SECOND), "a removed repository is still served"
    rows = _applied_rows(node)
    assert [row["detail"]["repositories"] for row in rows] == [
        [SCRATCH],
        [SCRATCH, SECOND],
        [SCRATCH],
    ]
    assert rows[-1]["detail"]["digest"] == github_mod.designation_digest([SCRATCH])


def test_an_unchanged_designation_is_a_receipted_no_op(
    request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    rig, _api, _env = _owner_rig(request)
    node = _node(tmp_path, None)
    assert _deliver(node, _ask_grant(rig, BORROWER_DEVICE)["narrowing"])["changed"] is True
    again = _deliver(node, _ask_grant(rig, BORROWER_DEVICE)["narrowing"])
    assert again["changed"] is False and again["error"] == "" and again["stale"] is False
    assert len(_applied_rows(node)) == 1, "an unchanged delivery wrote a second receipt"


# ---------------------------------------------------------------------------
# Ordering: an older delivery never overwrites a newer one
# ---------------------------------------------------------------------------


def test_an_out_of_order_delivery_never_puts_the_old_list_back(tmp_path: Path) -> None:
    """The flapping, reproduced: a grant cached before the change re-delivers [A]
    AFTER a fresh grant delivered [A,B]. The newer designation must win."""
    node = _node(tmp_path, None)
    old = github_mod.grant_narrowing([SCRATCH], generation_ms=1_000)
    new = github_mod.grant_narrowing([SCRATCH, SECOND], generation_ms=2_000)
    assert _deliver(node, old)["changed"] is True
    assert _deliver(node, new)["changed"] is True
    for _ in range(3):  # a stale cache re-delivers on every command
        late = _deliver(node, old)
        assert late["stale"] is True and late["changed"] is False, late
        assert late["repositories"] == [SCRATCH, SECOND]
    assert github_mod.repositories_for(node) == (SCRATCH, SECOND)
    assert len(_applied_rows(node)) == 2, "a stale delivery wrote a receipt"


def test_an_older_grant_for_the_same_designation_is_not_stale(tmp_path: Path) -> None:
    """Two processes holding grants for ONE designation must not force re-mints at
    each other: equal digest from an older generation is a quiet no-op."""
    node = _node(tmp_path, None)
    _deliver(node, github_mod.grant_narrowing([SCRATCH], generation_ms=2_000))
    older = _deliver(node, github_mod.grant_narrowing([SCRATCH], generation_ms=1_000))
    assert older["stale"] is False and older["changed"] is False, older


# ---------------------------------------------------------------------------
# ABSENT IS NOT EMPTY
# ---------------------------------------------------------------------------


def test_an_explicit_clear_refuses_every_repository(
    request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    """Owner config set to ``[]`` -> an explicit empty designation -> refuse-all."""
    home: _Env = request.getfixturevalue("env")
    _write_config(home.root, [])
    repositories, explicit = github_mod.designation_for(home.root)
    assert repositories == () and explicit is True
    narrowing = github_mod.grant_narrowing(repositories, explicit=explicit)
    assert narrowing["repositories"] == [] and narrowing["digest"], narrowing
    # Survives the peer boundary as an explicit empty list, not as absence.
    carried = Grant.from_detail({"access_token": "t", "narrowing": narrowing}).narrowing
    assert carried == narrowing

    node = _node(tmp_path, [SCRATCH])
    outcome = _deliver(node, carried)
    assert outcome["changed"] is True and outcome["repositories"] == [], outcome
    assert github_mod.repositories_for(node) == ()
    assert not _helper_serves(node, SCRATCH), "an explicit clear must refuse, never allow"
    written = yaml.safe_load((node / "config.yml").read_text(encoding="utf-8"))
    assert written["values"]["network"]["credentials"]["github"]["repositories"] == []


def test_a_never_set_owner_list_and_a_missing_field_leave_the_backstop(
    request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    """Never set -> nothing on the wire; an older owner's grant (no digest, no
    field) -> the borrower's standing list is untouched."""
    home: _Env = request.getfixturevalue("env")
    (home.root / "config.yml").write_text(yaml.safe_dump({"values": {}}), encoding="utf-8")
    repositories, explicit = github_mod.designation_for(home.root)
    assert explicit is False
    assert github_mod.grant_narrowing(repositories, explicit=explicit) == {}

    node = _node(tmp_path, [SCRATCH])
    client = SimpleNamespace(self_device=BORROWER_DEVICE, network_id=NET, owner_of=lambda k: "")
    assert github_mod.apply_grant_narrowing(Grant.from_detail({}), client=client, root=node) == {}
    legacy_empty = github_mod.remember_delivered_repositories([], root=node)
    assert legacy_empty["changed"] is False and legacy_empty["error"] == "no_usable_entries"
    assert github_mod.repositories_for(node) == (SCRATCH,)


def test_a_digest_that_does_not_match_its_list_is_refused(tmp_path: Path) -> None:
    node = _node(tmp_path, [SCRATCH])
    forged = {"repositories": [SECOND], "digest": "0" * 32, "generation": "5"}
    outcome = _deliver(node, forged)
    assert outcome["error"] == "digest_mismatch" and outcome["changed"] is False
    assert github_mod.repositories_for(node) == (SCRATCH,)


def test_the_delivery_never_raises_and_records_a_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    node = _node(tmp_path, [SCRATCH])

    def explode(root: Path | None, repositories: list[str]) -> None:
        raise OSError("disk on fire")

    monkeypatch.setattr(github_mod, "_write_repositories", explode)
    outcome = _deliver(node, github_mod.grant_narrowing([SECOND], generation_ms=9))
    assert outcome["error"] == "OSError" and outcome["changed"] is False
    assert github_mod.designation_delivered(node, OWNER_DEVICE) == {}, "order advanced on failure"


# ---------------------------------------------------------------------------
# The bash spawn path: a stale cached grant is replaced, not used
# ---------------------------------------------------------------------------


def test_borrowed_git_env_re_asks_when_its_cached_grant_is_stale(tmp_path: Path) -> None:
    """A session's cached grant predates a designation another process applied:
    its list is not re-applied and its (App: server-side narrowed) token is
    replaced by a forced re-ask."""
    node = _node(tmp_path, None)
    _deliver(node, github_mod.grant_narrowing([SCRATCH, SECOND], generation_ms=2_000))
    asks: list[bool] = []

    def grant(token: str, narrowing: dict[str, Any]) -> Grant:
        return Grant.from_detail(
            {"access_token": token, "grant_expires_at_ms": 0, "narrowing": narrowing}
        )

    dropped: list[str] = []

    class Grants:
        def drop(self, key: str) -> None:
            dropped.append(key)

    class Client:
        self_device = BORROWER_DEVICE
        network_id = NET
        grants = Grants()

        def owner_of(self, key: str) -> str:
            return OWNER_DEVICE

        def should_borrow(self, key: str) -> bool:
            return True

        def request_grant_sync(self, key: str, **kwargs: Any) -> Grant:
            # The re-ask must NOT be ``force_refresh`` (admin-only at the owner):
            # it is a plain ask after the cached grant was dropped.
            asks.append(bool(kwargs.get("force_refresh")))
            if dropped:
                return grant(
                    "fresh", github_mod.grant_narrowing([SCRATCH, SECOND], generation_ms=3_000)
                )
            return grant("stale", github_mod.grant_narrowing([SCRATCH], generation_ms=1_000))

    env_vars, token = github_mod.borrowed_git_env(root=node, client=Client())
    assert asks == [False, False] and dropped == [github_mod.GITHUB_KEY], (asks, dropped)
    assert token == "fresh" and env_vars
    assert github_mod.repositories_for(node) == (SCRATCH, SECOND)
    assert github_mod.designation_delivered(node, OWNER_DEVICE)["generation"] == 3_000


# ---------------------------------------------------------------------------
# TWO REAL RELAYS: `lop network credential refresh github --json` on the node
# ---------------------------------------------------------------------------


def _lop_network_json(*argv: str) -> dict[str, Any]:
    import argparse

    from local_operator.network import cli as network_cli

    parser = argparse.ArgumentParser(prog="lop")
    network_cli.add_parser(parser.add_subparsers(dest="command"))
    out = io.StringIO()
    with redirect_stdout(out):
        rc = network_cli.main(parser.parse_args(["network", *argv, "--json"]))
    assert rc == 0, out.getvalue()
    return json.loads(out.getvalue())


@pytest.fixture()
def github_mesh(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Owner A (PAT arm, synthetic token) shares github with node B; both relays up.

    The PAT arm keeps GitHub off the wire (re-delivery is enough for it, design
    §3.4); the App arm's re-mint is pinned above against the fake GitHub.
    """
    pair_devices = request.getfixturevalue("devices")
    server_a, server_b, host, port = pair_devices
    record, _h, _p = _pair(pair_devices, monkeypatch, role="admin")
    with store.mutate(record.network_id, server_b.root) as copy:
        owner_row = copy.member(server_a.identity.device_id)
        assert owner_row is not None
        owner_row.endpoints = [f"{host}:{port}"]
        store.save(copy, server_b.root)
    monkeypatch.setattr(github_mod, "resolve_source", lambda *a, **k: github_mod.SOURCE_TOKEN)
    monkeypatch.setattr(github_mod, "read_token_secret", lambda *a, **k: "synthetic-pat")
    _write_config(server_a.root, [SCRATCH])
    owner_id = server_a.identity.device_id
    document = placement_mod.PlacementDocument.load(
        record.network_id, server_a.root, self_device=owner_id
    )
    document.declare(
        github_mod.GITHUB_KEY,
        owner_device=owner_id,
        owner_device_name=server_a.identity.name,
        provider=github_mod.GITHUB_KEY,
        kind=github_mod.GITHUB_KIND,
        by=owner_id,
    )
    document.grant(github_mod.GITHUB_KEY, server_b.identity.device_id, scope="device", by=owner_id)
    document.save()
    server_b.start()
    # The node pulls the share through its own relay (leg 1 -> leg 2).
    found = store.find_own_relay(server_b.root)
    assert found is not None
    from local_operator.network import relay

    relay.control_request(found, "credential_placement", timeout=30.0)
    # The CLI below runs AS THE NODE: its config dir is B's root.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_b.root))
    try:
        yield SimpleNamespace(a=server_a, b=server_b, network_id=record.network_id)
    finally:
        server_b.stop()


def test_two_relays_refresh_delivers_widen_narrow_and_no_op(github_mesh: Any) -> None:
    node = github_mesh.b.root
    first = _lop_network_json("credential", "refresh", "github")
    assert first["ok"] is True and "access_token" not in json.dumps(first)
    assert first["narrowing"]["repositories"] == [SCRATCH], first
    assert first["narrowing"]["changed"] is True and first["narrowing"]["applies"] == "now"

    _write_config(github_mesh.a.root, [SCRATCH, SECOND])
    widened = _lop_network_json("credential", "refresh", "github")["narrowing"]
    assert widened["repositories"] == [SCRATCH, SECOND] and widened["changed"] is True, widened
    assert widened["previous"] == [SCRATCH]
    assert _helper_serves(node, SECOND)

    same = _lop_network_json("credential", "refresh", "github")["narrowing"]
    assert same["changed"] is False and same["repositories"] == [SCRATCH, SECOND], same

    _write_config(github_mesh.a.root, [SCRATCH])
    narrowed = _lop_network_json("credential", "refresh", "github")["narrowing"]
    assert narrowed["repositories"] == [SCRATCH] and narrowed["changed"] is True, narrowed
    assert not _helper_serves(node, SECOND)

    from local_operator.network.audit import AuditLog

    rows = [
        row
        for row in AuditLog(node).tail(200)
        if row.get("event") == "credential.narrowing_applied"
    ]
    assert [row["detail"]["repositories"] for row in rows] == [
        [SCRATCH],
        [SCRATCH, SECOND],
        [SCRATCH],
    ], rows
    assert {row["subject"] for row in rows} == {github_mesh.a.identity.device_id}


def test_the_human_receipt_names_the_list_and_whether_it_changed() -> None:
    from local_operator.network import cli as network_cli

    lines = network_cli._narrowing_lines
    assert lines({"delivered": True, "repositories": [SCRATCH], "changed": True}) == [
        f"repositories: {SCRATCH} (changed)"
    ]
    assert lines({"delivered": True, "repositories": [], "changed": True}) == [
        "repositories: none (refuses every repository) (changed)"
    ]
    assert "sent no designation" in lines({"delivered": False, "repositories": [SCRATCH]})[0]


# ---------------------------------------------------------------------------
# Round-1 remediation (PR #3841): cap, stale storm, owner key, receipt text
# ---------------------------------------------------------------------------


def _repos(count: int) -> list[str]:
    return [f"acme/repo-{index:03d}" for index in range(count)]


@pytest.mark.parametrize("count", [65, 500])
def test_a_long_designation_crosses_the_peer_boundary_whole(tmp_path: Path, count: int) -> None:
    """Mesh m2: a list over 64 was cut at the boundary and then failed its digest
    forever. Up to the App mint maximum (500) it now arrives whole and applies,
    and the settings validator accepts exactly that many."""
    from local_operator.settings_io import _validate_github_repositories

    repositories = _repos(count)
    assert _validate_github_repositories(repositories) == repositories
    narrowing = github_mod.grant_narrowing(repositories, generation_ms=7)
    carried = Grant.from_detail({"access_token": "t", "narrowing": narrowing}).narrowing
    assert carried["repositories"] == repositories
    node = _node(tmp_path, [SCRATCH])
    outcome = _deliver(node, carried)
    assert outcome["error"] == "" and outcome["changed"] is True, outcome["error"]
    assert github_mod.repositories_for(node) == tuple(repositories)


def test_a_designation_over_the_cap_is_refused_never_truncated(tmp_path: Path) -> None:
    """Over 500: the validator refuses it where a person types it, and a
    hand-edited owner list that the boundary cuts is REFUSED on the borrower with
    a named reason — never applied as a partial list."""
    from local_operator.settings_io import _validate_github_repositories

    too_many = _repos(501)
    with pytest.raises(ValueError, match="at most 500"):
        _validate_github_repositories(too_many)
    narrowing = github_mod.grant_narrowing(too_many, generation_ms=7)
    carried = Grant.from_detail({"access_token": "t", "narrowing": narrowing}).narrowing
    assert len(carried["repositories"]) == 500
    node = _node(tmp_path, [SCRATCH])
    outcome = _deliver(node, carried)
    assert outcome["error"] == "too_many_entries" and outcome["changed"] is False
    assert github_mod.repositories_for(node) == (SCRATCH,)


def _grant(
    token: str, narrowing: dict[str, Any], *, grant_id: str = "", served_by: str = ""
) -> Grant:
    detail: dict[str, Any] = {
        "access_token": token,
        "grant_expires_at_ms": 0,
        "narrowing": narrowing,
        "grant_id": grant_id,
    }
    if served_by:
        detail["credential_ref"] = {"owner_device": served_by}
    return Grant.from_detail(detail)


def test_a_re_ask_that_is_stale_too_is_not_re_asked_again(tmp_path: Path) -> None:
    """R1-2: an owner whose generation went backwards answers every re-ask with a
    grant that is stale too. One re-ask, one ``generation_regressed`` receipt, and
    later spawns use that grant instead of dropping and re-minting per command."""
    from local_operator.network.audit import AuditLog

    node = _node(tmp_path, None)
    _deliver(node, github_mod.grant_narrowing([SCRATCH, SECOND], generation_ms=5_000))
    regressed = _grant(
        "regressed",
        github_mod.grant_narrowing([SCRATCH], generation_ms=1_000),
        grant_id="g_r1",
    )
    asks: list[int] = []
    cached: list[Grant] = [
        _grant(
            "old",
            github_mod.grant_narrowing([SCRATCH], generation_ms=900),
            grant_id="g_o",
        )
    ]

    class Grants:
        def drop(self, key: str) -> None:
            cached.clear()

    class Client:
        self_device = BORROWER_DEVICE
        network_id = NET
        grants = Grants()

        def owner_of(self, key: str) -> str:
            return OWNER_DEVICE

        def should_borrow(self, key: str) -> bool:
            return True

        def request_grant_sync(self, key: str, **kwargs: Any) -> Grant:
            asks.append(1)
            if not cached:
                cached.append(regressed)  # the owner's answer, cached like the client does
            return cached[0]

    client = Client()
    _, first = github_mod.borrowed_git_env(root=node, client=client)
    assert first == "regressed" and len(asks) == 2
    for _ in range(3):
        _, token = github_mod.borrowed_git_env(root=node, client=client)
        assert token == "regressed"
    assert len(asks) == 5, "each later spawn is ONE ask (the cache), never a drop + re-ask"
    refused = [
        row
        for row in AuditLog(node).tail(200, network_id=NET)
        if row.get("event") == "credential.narrowing_refused"
    ]
    assert [row["detail"]["reason"] for row in refused] == ["generation_regressed"]
    assert github_mod.repositories_for(node) == (SCRATCH, SECOND), "never the older list"


def test_an_unknown_owner_never_records_under_an_empty_key(tmp_path: Path) -> None:
    """R1-3: no owner anywhere -> the verified list applies, the order record is
    neither read nor advanced, and the result says so."""
    node = _node(tmp_path, [SCRATCH])
    client = SimpleNamespace(self_device=BORROWER_DEVICE, network_id=NET, owner_of=lambda k: "")
    outcome = github_mod.apply_grant_narrowing(
        _grant("t", github_mod.grant_narrowing([SECOND], generation_ms=9)),
        client=client,
        root=node,
    )
    assert outcome["owner_unknown"] is True and outcome["changed"] is True, outcome
    assert github_mod.repositories_for(node) == (SECOND,)
    assert github_mod._read_narrowing_state(node) == {}  # noqa: SLF001


def test_the_order_record_is_keyed_by_the_device_that_served_the_grant(
    tmp_path: Path,
) -> None:
    """R1-4: a grant cached from the PREVIOUS owner, applied after ownership moved,
    is ordered against that owner's deliveries, not the current owner's."""
    node = _node(tmp_path, None)
    client = SimpleNamespace(
        self_device=BORROWER_DEVICE, network_id=NET, owner_of=lambda k: "d_new_owner"
    )
    github_mod.apply_grant_narrowing(
        _grant(
            "t",
            github_mod.grant_narrowing([SCRATCH], generation_ms=10),
            served_by="d_old_owner",
        ),
        client=client,
        root=node,
    )
    assert github_mod.designation_delivered(node, "d_old_owner")["generation"] == 10
    assert github_mod.designation_delivered(node, "d_new_owner") == {}


def test_the_refresh_receipt_matches_its_json_on_stale_and_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1-5 + QA Q-1: ``after_next_mint`` says so in words, and a REFUSED delivery
    reports the list this device actually holds, with ``applies: not_applied``."""
    from local_operator.network import cli as network_cli
    from local_operator.network.credentials import client as client_mod

    node = _node(tmp_path, [SCRATCH])
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(node))
    monkeypatch.setattr(
        client_mod.MeshCredentialClient,
        "for_this_device",
        classmethod(lambda cls, r: None),
    )
    forged = {"repositories": [SCRATCH, SECOND], "digest": "0" * 32, "generation": "5"}
    receipt = network_cli._apply_refreshed_narrowing(  # noqa: SLF001
        {"narrowing": forged, "credential_ref": {"owner_device": OWNER_DEVICE}}
    )
    assert receipt["applies"] == "not_applied" and receipt["error"] == "digest_mismatch"
    assert receipt["repositories"] == [SCRATCH], "the held list, not the rejected one"
    assert network_cli._narrowing_lines(receipt) == [  # noqa: SLF001
        f"repositories: not applied (digest_mismatch); still {SCRATCH}"
    ]
    stale_line = network_cli._narrowing_lines(  # noqa: SLF001
        {"delivered": True, "repositories": [SCRATCH], "applies": "after_next_mint"}
    )[0]
    assert "not applied" in stale_line and "applies after the next mint" in stale_line


def test_the_narrowing_lock_file_is_private(tmp_path: Path) -> None:
    """QA NIT: the lock beside the order record is 0600 like its neighbours."""
    node = _node(tmp_path, None)
    _deliver(node, github_mod.grant_narrowing([SCRATCH], generation_ms=3))
    lock = github_mod._narrowing_state_path(node).with_suffix(".lock")  # noqa: SLF001
    assert lock.stat().st_mode & 0o777 == 0o600
