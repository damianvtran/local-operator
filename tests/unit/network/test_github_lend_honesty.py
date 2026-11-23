"""GitHub lend honesty (2026-11-19): name the arm that actually serves.

The operator kept the full gh login on node2, so a GitHub lend from there is the
account's broad user token behind a git-only repository fence. Before this, the
credentials listing printed ``github-app`` for it and ``ready`` said nothing.
These cells pin the honest label and the WARN row for the gh arm, and that the
App and token arms keep exactly their old words with no row.

Each arm is seeded the way the ladder reads it — a secret-store ENTRY for the App
and token arms, a tokenless ``hosts.yml`` (gh's keyring shape) under a scratch
HOME for the gh arm — so ``resolve_source`` runs unpatched and nothing ever asks
gh for a token: source resolution reads presence only.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest
import yaml

from local_operator.network import cli as net_cli
from local_operator.network import readiness
from local_operator.network.credentials import github as github_mod
from local_operator.paths import CONFIG_DIR_ENV

GH_LABEL = "gh CLI login (full account)"
WARNING = "a broad user token (gh CLI login) sits behind a git-only repo fence"


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A scratch HOME and config root; no secret broker, no relay, no gh binary."""
    from local_operator.secrets import client as secrets_client

    home = tmp_path / "home"
    root = tmp_path / "config"
    home.mkdir()
    root.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv(CONFIG_DIR_ENV, str(root))
    for name in ("GH_TOKEN", "GITHUB_TOKEN", "GH_CONFIG_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(secrets_client, "ensure_broker", lambda *a, **k: False)
    # The listing asks the relay first; there is none here, and a stub keeps the
    # cell from dialling a socket a developer's own relay might be holding.
    monkeypatch.setattr(net_cli, "_relay_call", lambda *a, **k: None)
    # ``gh auth token`` must never run: a resolution that tried would fail loudly.
    monkeypatch.setattr(
        github_mod,
        "find_gh",
        lambda *a, **k: pytest.fail("source resolution must not look for gh"),
    )


def _root() -> Path:
    import os

    return Path(os.environ[CONFIG_DIR_ENV])


def _seed_secret(name: str) -> None:
    from local_operator.secrets.keys import load_master_key
    from local_operator.secrets.store import SecretStore

    master_key = load_master_key(_root(), create=True)
    store = SecretStore(master_key, base=_root())
    store.initialize()
    store.set(name, b"placeholder-not-a-credential", description=f"test {name}")


def _seed_gh_login() -> None:
    import os

    path = Path(os.environ["HOME"]) / ".config" / "gh" / "hosts.yml"
    path.parent.mkdir(parents=True)
    # gh's keyring shape: the entry names the account, the token lives elsewhere.
    path.write_text(
        yaml.safe_dump({"github.com": {"user": "octo", "git_protocol": "https"}}),
        encoding="utf-8",
    )


def _arm(source: str) -> None:
    if source == github_mod.SOURCE_APP:
        _seed_secret(github_mod.APP_SECRET_NAME)
    elif source == github_mod.SOURCE_TOKEN:
        _seed_secret(github_mod.TOKEN_SECRET_NAME)
    else:
        _seed_gh_login()
    assert github_mod.resolve_source(_root()) == source


def _github_row(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    assert net_cli._cmd_credentials(Namespace(json=True, network="")) == 0  # noqa: SLF001
    payload = json.loads(capsys.readouterr().out)
    rows = [row for row in payload["shareable"] if row.get("provider") == "github"]
    assert len(rows) == 1, payload["shareable"]
    return rows[0]


def _listing_text(capsys: pytest.CaptureFixture[str]) -> str:
    assert net_cli._cmd_credentials(Namespace(json=False, network="")) == 0  # noqa: SLF001
    return capsys.readouterr().out


def test_selection_order_prefers_app_then_token_then_gh() -> None:
    """The ladder the PR documents: app > token > gh, first match wins."""
    _seed_gh_login()
    assert github_mod.resolve_source(_root()) == github_mod.SOURCE_GH
    _seed_secret(github_mod.TOKEN_SECRET_NAME)
    assert github_mod.resolve_source(_root()) == github_mod.SOURCE_TOKEN
    _seed_secret(github_mod.APP_SECRET_NAME)
    assert github_mod.resolve_source(_root()) == github_mod.SOURCE_APP


def test_gh_source_lists_the_full_account_label_and_warns(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _arm(github_mod.SOURCE_GH)
    row = _github_row(capsys)
    # ``kind`` is the stable machine field (offers, relay grant filter, placement
    # wire type); the honest name rides ``label`` beside the arm it came from.
    assert row["kind"] == "github-app"
    assert row["source"] == "gh"
    assert row["label"] == GH_LABEL
    out = _listing_text(capsys)
    assert f"  github  {GH_LABEL}  login held — share:" in out, out
    assert f"      {WARNING}" in out, out
    assert "github-app" not in out, out

    rows = readiness.local_github_source_rows(_root(), device_id="d_self")
    assert len(rows) == 1
    (warn,) = rows
    assert warn["capability"] == "github_broad_token"
    assert warn["ok"] is False
    assert readiness.row_state(warn) == readiness.ROW_STATE_WARN
    # Non-gating: it is named, never an onboarding failure.
    assert readiness.onboarding_failures(rows) == []
    assert WARNING in warn["detail"]
    assert GH_LABEL in warn["detail"]
    line, *remedies = readiness.render_check_lines(rows)
    assert line.startswith("warn readiness github_broad_token this device: "), line
    assert WARNING in line
    assert any("GITHUB_TOKEN" in remedy for remedy in remedies), remedies


@pytest.mark.parametrize("source", [github_mod.SOURCE_APP, github_mod.SOURCE_TOKEN])
def test_app_and_token_sources_keep_the_old_label_and_no_row(
    source: str, capsys: pytest.CaptureFixture[str]
) -> None:
    _arm(source)
    row = _github_row(capsys)
    assert row["kind"] == "github-app"
    assert row["source"] == source
    assert row["label"] == "github-app"
    out = _listing_text(capsys)
    assert "  github  github-app  login held — share:" in out, out
    assert WARNING not in out
    assert GH_LABEL not in out
    assert readiness.local_github_source_rows(_root(), device_id="d_self") == []


def test_fallback_ready_carries_the_gh_warning_row() -> None:
    """``lop network ready`` with no relay answers from ``_ready_locally``; the
    row must ride that path too, not only the relay composer."""
    _arm(github_mod.SOURCE_GH)
    payload = net_cli._ready_locally(Namespace(peer="", model="", session=""))  # noqa: SLF001
    caps = [check.get("capability") for check in payload["checks"]]
    assert "github_broad_token" in caps, caps


def test_readiness_mirror_matches_the_ladder_spelling() -> None:
    assert readiness.GITHUB_SOURCE_GH == github_mod.SOURCE_GH
    assert github_mod.GH_SOURCE_LABEL == GH_LABEL


def _declare_github_lend() -> str:
    """This device owns a ``github`` placement shared with a peer: node2's shape."""
    from local_operator.network import identity as identity_mod
    from local_operator.network import store, types
    from local_operator.network.credentials import placement as placement_mod

    network = "n_0123456789abcdef01234567"
    self_device = identity_mod.load_or_mint(name="node2").device_id
    peer = "d_" + "b" * 32
    record = types.NetworkRecord(
        network_id=network, name="home", self_device_id=self_device, self_role="admin"
    )
    record.members.append(types.MemberRecord(device_id=self_device, name="node2"))
    record.members.append(types.MemberRecord(device_id=peer, name="laptop"))
    store.save(record, _root())
    document = placement_mod.PlacementDocument(network, root=_root())
    document.declare(
        github_mod.GITHUB_KEY,
        owner_device=self_device,
        owner_device_name="node2",
        kind=github_mod.GITHUB_KIND,
        by=self_device,
    )
    document.grant(github_mod.GITHUB_KEY, peer, scope="device", by=self_device)
    document.save()
    return self_device


@pytest.mark.parametrize(
    ("source", "label"),
    [
        (github_mod.SOURCE_GH, GH_LABEL),
        (github_mod.SOURCE_APP, "github-app"),
        (github_mod.SOURCE_TOKEN, "github-app"),
    ],
)
def test_owned_github_lend_row_names_its_serving_arm(
    source: str, label: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """The network block's owner row — the lend itself — reads the same label."""
    _arm(source)
    _declare_github_lend()
    assert net_cli._cmd_credentials(Namespace(json=True, network="")) == 0  # noqa: SLF001
    payload = json.loads(capsys.readouterr().out)
    (network,) = payload["networks"]
    (row,) = [r for r in network["credentials"] if r["credential_name"] == "github"]
    assert row["owned_here"] is True
    assert row["kind"] == "github-app"
    assert (row["source"], row["label"]) == (source, label)
    out = _listing_text(capsys)
    # The type column widens to the longest label, so ``owner:`` stays aligned
    # (review n1 / QA Q3) and the gh lend carries the same caution as ``shareable``.
    assert f"  github         {label:<{max(15, len(label))}} owner: this device" in out, out
    assert (
        f"  github         {label:<{max(15, len(label))}} owner: this device\n      {WARNING}"
        in out
    ) == (source == github_mod.SOURCE_GH), out


def test_listing_type_column_aligns_owner_across_rows(capsys: pytest.CaptureFixture[str]) -> None:
    """A wide label sizes the column for every row of that network (review n1)."""
    from local_operator.network.credentials import placement as placement_mod

    _arm(github_mod.SOURCE_GH)
    self_device = _declare_github_lend()
    document = placement_mod.PlacementDocument.load(
        "n_0123456789abcdef01234567", _root(), self_device=self_device
    )
    document.declare(
        "OPENAI_API_KEY",
        owner_device=self_device,
        owner_device_name="node2",
        kind="api_key",
        by=self_device,
    )
    document.save()
    out = _listing_text(capsys)
    rows = [line for line in out.splitlines() if "owner: this device" in line]
    assert len(rows) == 2, out
    assert len({line.index("owner:") for line in rows}) == 1, out


def test_agent_tool_digest_names_the_serving_arm() -> None:
    """The agent's ``credentials`` digest reads ``label``, not ``kind`` (review M1/S1)."""
    from local_operator.network import tool as net_tool

    def payload(source: str) -> dict[str, object]:
        return {
            "ok": True,
            "networks": [
                {
                    "network": "n1",
                    "credentials": [
                        {
                            "credential_name": "github",
                            "kind": github_mod.GITHUB_KIND,
                            "source": source,
                            "label": github_mod.source_label(source),
                            "owned_here": True,
                        }
                    ],
                }
            ],
            "shareable": [],
        }

    # Through ``_scrub`` like the production path, so a field it drops would show.
    gh = net_tool._render(
        "credentials", net_tool._scrub(payload(github_mod.SOURCE_GH))
    )  # noqa: SLF001
    assert gh == ["n1:", f"  github  {GH_LABEL}  owner: this device", f"      {WARNING}"]
    app = net_tool._render(
        "credentials", net_tool._scrub(payload(github_mod.SOURCE_APP))
    )  # noqa: SLF001
    assert app == ["n1:", "  github  github-app  owner: this device"]
