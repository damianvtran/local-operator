"""The forge source ladder (S2 of ``mesh-consent-provisioning.md`` §3).

The note's gates for this slice, each a cell below: the SOURCE LADDER (order,
presence semantics, the broken-arm refusals that must not silently downgrade),
the REFUSAL ARMS (per-source codes and their borrower sentences), the PER-SOURCE
RECEIPTS (revoke copy and share disclosure), the GitLab stub's pinned names —
and, in the loopback section, a REAL ``git push`` served through the helper per
token source, because "delivery unchanged" is only proven by driving it.

The heavy harness (the fake GitHub, the CONNECT+TLS proxy, the git http-backend,
the env/marker plumbing) lives in ``test_credentials_github.py``; this module
imports it rather than forking it, the same way the sibling provisioning tests
import their fixtures. What is deliberately NOT here: the App arm's own
behaviour (its file owns that) and anything about copies/sync (S3/S4).
"""

from __future__ import annotations

import contextlib
import time
from pathlib import Path
from typing import Any, Iterator

import pytest
import yaml

from local_operator.network.credentials import github as github_mod
from local_operator.network.credentials import owner as owner_mod
from local_operator.network.credentials import placement as placement_mod
from local_operator.paths import CONFIG_DIR_ENV
from tests.unit.network.test_credentials_github import (
    BORROWER_DEVICE,
    OWNER_DEVICE,
    SCRATCH,
    _detail,
    _Env,
    _FakeGithub,
    _grant_frame,
    _HttpsGithubProxy,
    _Link,
    _make_cert,
    _marker,
    _node_env,
    _node_gitconfig,
    _run,
    _seed_app_key,
    _tree_bytes,
    _work_repo,
    _write_config,
)

#: The two token-arm values. Fixed strings on purpose: the loopback backend
#: authenticates by EXACT value against the fake forge's liveness set, so a cell
#: asserting "this token pushed" cannot drift onto a minted value.
PAT = "github_pat_11TEST0000000000000000000000000000000000000000000"
GHO = "gho_test_login_token_0000000000000000000001"


@pytest.fixture(autouse=True)
def _no_daemon(monkeypatch: pytest.MonkeyPatch) -> None:
    """The sibling module's guard, inlined (fixtures cannot wrap other fixtures).

    ``access.retrieve_secret`` reaches the secret broker daemon first when one
    is reachable; pinning the fallback keeps every cell on the keyfile tier and
    launches no process.
    """
    from local_operator.secrets import client as secrets_client

    monkeypatch.setattr(secrets_client, "ensure_broker", lambda *a, **k: False)


@pytest.fixture()
def app_keypair() -> tuple[str, str]:
    """A real RSA keypair (public half for the fake GitHub's JWT verifier)."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    public = (
        key.public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        .decode()
    )
    return pem, public


@pytest.fixture()
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Env:
    holder = _Env(tmp_path)
    monkeypatch.setenv("HOME", str(holder.home))
    monkeypatch.setenv(CONFIG_DIR_ENV, str(holder.root))
    return holder


@pytest.fixture()
def github_api(app_keypair: tuple[str, str]) -> Any:
    fake = _FakeGithub(app_keypair[1])
    try:
        yield fake
    finally:
        fake.close()


# ---------------------------------------------------------------------------
# Seeding the arms and building the owner rig
# ---------------------------------------------------------------------------


def _seed_secret(root: Path, name: str, value: str) -> None:
    """One secret-store entry through the keyfile tier (``_seed_app_key``'s path).

    Re-seeding an existing name UPDATES it (``SecretStore.set`` refuses overwrites
    by design); cells that mutate an arm's value in sequence rely on this.
    """
    from local_operator.secrets.errors import SecretExists
    from local_operator.secrets.keys import load_master_key
    from local_operator.secrets.store import SecretStore

    master_key = load_master_key(root, create=True)
    store = SecretStore(master_key, base=root)
    store.initialize()
    try:
        store.set(name, value.encode(), description=f"test {name}")
    except SecretExists:
        store.update(name, value.encode(), description=f"test {name}")


def _seed_gh_login(home: Path, token: str, *, host: str = "github.com") -> None:
    path = home / ".config" / "gh" / "hosts.yml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump({host: {"user": "operator", "oauth_token": token}}), encoding="utf-8"
    )


class _Rig:
    def __init__(self, env: _Env, broker: Any, audit: Any, document: Any) -> None:
        self.env = env
        self.broker = broker
        self.audit = audit
        self.document = document


@contextlib.contextmanager
def _owner_for(env: _Env, github_api: _FakeGithub, source: str, *, pem: str = "") -> Iterator[_Rig]:
    """The broker rig for one ladder arm (mirrors the sibling's ``owner`` fixture).

    Same placement document, same share, same audit — only the ARM differs, so a
    difference between cells is the ladder's, never the rig's.
    """
    if source == github_mod.SOURCE_APP:
        _seed_app_key(env.root, pem)
    elif source == github_mod.SOURCE_TOKEN:
        _seed_secret(env.root, github_mod.TOKEN_SECRET_NAME, PAT)
        github_api.tokens.add(PAT)  # live at the fake forge, for the loopback cells
    elif source == github_mod.SOURCE_GH:
        _seed_gh_login(env.home, GHO)
        github_api.tokens.add(GHO)
    _write_config(env.root, [SCRATCH])
    document = placement_mod.PlacementDocument("n_gh", root=env.root, written_by=OWNER_DEVICE)
    document.declare(
        github_mod.GITHUB_KEY,
        owner_device=OWNER_DEVICE,
        owner_device_name="owner-laptop",
        provider=github_mod.GITHUB_KEY,
        identity_label="",
        by=OWNER_DEVICE,
    )
    document.grant(github_mod.GITHUB_KEY, BORROWER_DEVICE, scope="device", by=OWNER_DEVICE)
    document.save()
    from local_operator.network.audit import AuditLog

    audit = AuditLog(root=env.root)
    broker = owner_mod.MeshCredentialBroker(
        root=env.root,
        self_device=OWNER_DEVICE,
        self_device_name="owner-laptop",
        network_id="n_gh",
        audit=audit,
        github_minter=github_mod.GithubMinter(base_url=github_api.url),
    )
    try:
        yield _Rig(env=env, broker=broker, audit=audit, document=document)
    finally:
        broker.close()
        audit.close()


def _ask(rig: _Rig, device: str = BORROWER_DEVICE, **kwargs: Any) -> dict[str, Any]:
    return _detail(rig.broker.on_broker(_Link(device), _grant_frame(device, **kwargs)))


# ---------------------------------------------------------------------------
# The one ladder order: presence semantics
# ---------------------------------------------------------------------------


def test_the_ladder_prefers_strength_and_is_the_one_order_serving_uses(
    env: _Env, app_keypair: tuple[str, str]
) -> None:
    """gh < token < App; each rung only decides when the ones above are absent."""
    assert github_mod.resolve_source(env.root) == ""
    _seed_gh_login(env.home, GHO)
    assert github_mod.resolve_source(env.root) == github_mod.SOURCE_GH
    _seed_secret(env.root, github_mod.TOKEN_SECRET_NAME, PAT)
    assert github_mod.resolve_source(env.root) == github_mod.SOURCE_TOKEN
    _seed_app_key(env.root, app_keypair[0])
    assert github_mod.resolve_source(env.root) == github_mod.SOURCE_APP


def test_a_logged_out_hosts_file_is_not_an_arm(env: _Env) -> None:
    """``gh auth logout`` leaves a file with no github.com entry: no arm, not a refusal."""
    path = env.home / ".config" / "gh" / "hosts.yml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"github.example.com": {"user": "x"}}), encoding="utf-8")
    assert github_mod.resolve_source(env.root) == ""


def test_a_broken_hosts_file_still_resolves_so_the_refusal_can_name_it(env: _Env) -> None:
    """Present-but-broken is NOT silently skipped: falling through would lend wider."""
    path = env.home / ".config" / "gh" / "hosts.yml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("github.com: [unclosed", encoding="utf-8")
    assert github_mod.resolve_source(env.root) == github_mod.SOURCE_GH


def test_read_token_secret_arms(env: _Env) -> None:
    with pytest.raises(github_mod.GithubTokenError) as absent:
        github_mod.read_token_secret(env.root)
    assert absent.value.kind == "absent"

    _seed_secret(env.root, github_mod.TOKEN_SECRET_NAME, "   \n ")
    with pytest.raises(github_mod.GithubTokenError) as blank:
        github_mod.read_token_secret(env.root)
    assert blank.value.kind == "unusable"

    _seed_secret(env.root, github_mod.TOKEN_SECRET_NAME, "two\nlines")
    with pytest.raises(github_mod.GithubTokenError) as multi:
        github_mod.read_token_secret(env.root)
    assert multi.value.kind == "unusable"

    _seed_secret(env.root, github_mod.TOKEN_SECRET_NAME, "x" * 513)
    with pytest.raises(github_mod.GithubTokenError) as huge:
        github_mod.read_token_secret(env.root)
    assert huge.value.kind == "unusable"

    _seed_secret(env.root, github_mod.TOKEN_SECRET_NAME, PAT)
    assert github_mod.read_token_secret(env.root) == PAT


def test_read_gh_token_arms(env: _Env) -> None:
    with pytest.raises(github_mod.GithubGhError) as absent:
        github_mod.read_gh_token(env.home)
    assert absent.value.kind == "absent"

    path = env.home / ".config" / "gh" / "hosts.yml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"github.com": {"user": "operator"}}), encoding="utf-8")
    with pytest.raises(github_mod.GithubGhError) as tokenless:
        github_mod.read_gh_token(env.home)
    assert tokenless.value.kind == "unusable"

    path.write_text("not yaml: [", encoding="utf-8")
    with pytest.raises(github_mod.GithubGhError) as malformed:
        github_mod.read_gh_token(env.home)
    assert malformed.value.kind == "unusable"

    _seed_gh_login(env.home, GHO)
    assert github_mod.read_gh_token(env.home) == GHO


def test_no_source_message_names_all_three_arms_and_the_guide() -> None:
    message = github_mod.no_source_message()
    assert "gh CLI" in message
    assert github_mod.TOKEN_SECRET_NAME in message
    assert "GitHub App" in message
    assert "the ladder" in message
    assert "unaffected" in message


# ---------------------------------------------------------------------------
# Serving from the token arms
# ---------------------------------------------------------------------------


def test_the_token_arm_serves_the_stored_token_without_minting(
    env: _Env, github_api: _FakeGithub
) -> None:
    """A grant from arm 2: the stored value, ``refreshed: false``, nothing registered.

    The registry being empty afterwards is the §3.3 decision made observable:
    there is no server-side revoke handle for a token lop did not mint, so the
    lender has nothing to track and the receipt (below) says why.
    """
    with _owner_for(env, github_api, github_mod.SOURCE_TOKEN) as rig:
        detail = _ask(rig)
        assert detail["kind"] == "grant", detail
        assert detail["access_token"] == PAT
        assert detail["token_kind"] == "bearer"
        assert detail["refreshed"] is False
        assert detail["token_expires_at_ms"] == 0
        assert detail["credential_ref"]["kind"] == github_mod.GITHUB_KIND
        assert detail["credential_ref"]["provider"] == github_mod.GITHUB_KEY
        assert detail["scope"] == {"kind": "device", "session_id": ""}
        now_ms = time.time() * 1000
        assert detail["grant_expires_at_ms"] <= now_ms + 900_000 + 5_000
        # No mint was attempted against the forge, and the lender tracks nothing.
        assert github_api.mint_requests == []
        assert rig.broker._lender().outstanding() == 0  # noqa: SLF001 — the registry's own count
        # A second serve re-reads the source (§3.3's refresh path) and serves it again.
        again = _ask(rig)
        assert again["access_token"] == PAT
        assert github_api.mint_requests == []


def test_the_gh_arm_serves_the_login_the_cli_already_holds(
    env: _Env, github_api: _FakeGithub
) -> None:
    with _owner_for(env, github_api, github_mod.SOURCE_GH) as rig:
        detail = _ask(rig)
        assert detail["access_token"] == GHO
        assert detail["refreshed"] is False
        assert detail["token_expires_at_ms"] == 0
        assert github_api.mint_requests == []
        assert rig.broker._lender().outstanding() == 0  # noqa: SLF001


def test_a_configured_but_broken_app_refuses_and_never_downgrades(
    env: _Env, github_api: _FakeGithub
) -> None:
    """The App is the narrower arm; a broken one refuses BY NAME (no silent fall-through).

    Seeded alongside a perfectly good token and gh login — the cell asserts the
    refusal, not a downgrade, which is the whole point of the strongest-first
    resolution treating presence as the decision.
    """
    _seed_secret(env.root, github_mod.TOKEN_SECRET_NAME, PAT)
    _seed_gh_login(env.home, GHO)
    _seed_secret(env.root, github_mod.APP_SECRET_NAME, "not json")
    with _owner_for(env, github_api, source="") as rig:
        detail = _ask(rig)
        assert detail["code"] == github_mod.CODE_APP_UNUSABLE, detail


def test_a_broken_token_secret_refuses_and_never_downgrades_to_gh(
    env: _Env, github_api: _FakeGithub
) -> None:
    _seed_secret(env.root, github_mod.TOKEN_SECRET_NAME, "   ")
    _seed_gh_login(env.home, GHO)
    with _owner_for(env, github_api, source="") as rig:
        detail = _ask(rig)
        assert detail["code"] == github_mod.CODE_TOKEN_UNUSABLE, detail
        assert github_mod.TOKEN_SECRET_NAME in detail["message"]


def test_a_tokenless_gh_login_refuses_by_name(env: _Env, github_api: _FakeGithub) -> None:
    path = env.home / ".config" / "gh" / "hosts.yml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"github.com": {"user": "operator"}}), encoding="utf-8")
    with _owner_for(env, github_api, source="") as rig:
        detail = _ask(rig)
        assert detail["code"] == github_mod.CODE_GH_UNUSABLE, detail


def test_an_empty_ladder_refuses_with_the_one_ladder_sentence(
    env: _Env, github_api: _FakeGithub
) -> None:
    with _owner_for(env, github_api, source="") as rig:
        detail = _ask(rig)
        assert detail["code"] == "no_local_credential", detail
        assert detail["message"] == github_mod.no_source_message()


def test_the_new_refusal_codes_render_borrower_sentences(
    env: _Env, github_api: _FakeGithub
) -> None:
    """Each arm's code has a sentence naming the owner and a repair that is not a retry."""
    from local_operator.network.credentials.messages import render_broker_error
    from local_operator.network.credentials.types import BrokerError

    for code, needle in (
        (github_mod.CODE_TOKEN_UNUSABLE, "GITHUB_TOKEN"),
        (github_mod.CODE_GH_UNUSABLE, "gh CLI"),
        ("no_local_credential", "the ladder"),
    ):
        sentence = render_broker_error(
            BrokerError(code=code, key=github_mod.GITHUB_KEY),
            key=github_mod.GITHUB_KEY,
            owner_name="owner-laptop",
        )
        assert "owner-laptop" in sentence, (code, sentence)
        assert needle in sentence, (code, sentence)


# ---------------------------------------------------------------------------
# Receipts: revoke copy and the share disclosure, per source
# ---------------------------------------------------------------------------


def test_the_revoke_receipt_is_per_source() -> None:
    from local_operator.network import cli as network_cli

    app_lines = network_cli._github_revoke_lines("peer-b", 1, github_mod.SOURCE_APP)  # noqa: SLF001
    assert "DELETE /installation/token" in app_lines[0]
    app_payload = network_cli._github_revocation_payload(  # noqa: SLF001
        1, 900, github_mod.SOURCE_APP
    )
    assert app_payload["minted_tokens_revoked"] == 1

    token_lines = network_cli._github_revoke_lines(
        "peer-b", 0, github_mod.SOURCE_TOKEN
    )  # noqa: SLF001
    joined = "\n".join(token_lines)
    assert "was not minted here" in joined
    assert "revoke the PAT at GitHub" in joined
    assert "DELETE /installation/token" not in joined
    token_payload = network_cli._github_revocation_payload(  # noqa: SLF001
        0, 900, github_mod.SOURCE_TOKEN
    )
    assert token_payload["source"] == github_mod.SOURCE_TOKEN
    assert "not minted by this device" in token_payload["copied_bearer"]

    gh_lines = network_cli._github_revoke_lines(
        "peer-b", None, github_mod.SOURCE_GH
    )  # noqa: SLF001
    assert "sign out of the gh CLI" in "\n".join(gh_lines)

    none_lines = network_cli._github_revoke_lines("peer-b", None, "")  # noqa: SLF001
    assert "no GitHub source is configured" in "\n".join(none_lines)


def test_the_share_disclosure_carries_the_narrowing_honesty_clause_for_token_arms() -> None:
    from local_operator.network import cli as network_cli

    app_line = network_cli._github_share_disclosure("peer-b", github_mod.SOURCE_APP)  # noqa: SLF001
    assert "authorised by the DEVICE" in app_line
    assert "copy of your login" not in app_line

    for source in (github_mod.SOURCE_TOKEN, github_mod.SOURCE_GH):
        line = network_cli._github_share_disclosure("peer-b", source)  # noqa: SLF001
        assert "authorised by the DEVICE" in line
        assert "nothing else through lop's own paths" in line
        assert "treat a share like a copy of your login" in line


def test_a_share_is_refused_with_the_ladder_sentence_when_no_arm_resolves(
    env: _Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.network import cli as network_cli
    from local_operator.network.types import MeshRefusal

    monkeypatch.setattr(network_cli, "_config_dir", lambda: env.root)
    with pytest.raises(MeshRefusal) as refusal:
        network_cli._require_local_credential(
            github_mod.GITHUB_KEY, github_mod.GITHUB_KEY
        )  # noqa: SLF001
    assert refusal.value.code == "no_local_credential"
    assert refusal.value.sentence == github_mod.no_source_message()


def test_a_gh_login_alone_is_shareable(env: _Env, monkeypatch: pytest.MonkeyPatch) -> None:
    from local_operator.network import cli as network_cli

    _seed_gh_login(env.home, GHO)
    monkeypatch.setattr(network_cli, "_config_dir", lambda: env.root)
    network_cli._require_local_credential(
        github_mod.GITHUB_KEY, github_mod.GITHUB_KEY
    )  # noqa: SLF001


def test_the_offer_row_exists_and_the_ledger_row_renders_for_any_arm(env: _Env) -> None:
    """The candidate gate is ``resolve_source`` now, not App-only (§3.2, S2)."""
    from local_operator.network.credentials import offers

    assert offers.credential_here(github_mod.GITHUB_KEY, env.root) is False
    _seed_gh_login(env.home, GHO)
    assert offers.credential_here(github_mod.GITHUB_KEY, env.root) is True
    rows = offers.enumerate_candidates(env.root)
    assert {"key": github_mod.GITHUB_KEY, "kind": github_mod.GITHUB_KIND, "label": ""} in rows
    assert offers.kind_label(github_mod.GITHUB_KIND) == "GitHub"


# ---------------------------------------------------------------------------
# The GitLab stub: the slice's decided names, pinned before any code lands
# ---------------------------------------------------------------------------


def test_the_gitlab_stub_pins_the_decided_names() -> None:
    from local_operator.network.credentials import gitlab as gitlab_mod

    assert gitlab_mod.GITLAB_KEY == "gitlab"
    assert gitlab_mod.TOKEN_SECRET_NAME == "GITLAB_TOKEN"
    assert gitlab_mod.is_gitlab_key("gitlab") is True
    assert gitlab_mod.is_gitlab_key("github") is False


# ---------------------------------------------------------------------------
# The loopback cells: a REAL git push, served per token source
# ---------------------------------------------------------------------------


def _push_through_the_ladder(
    env: _Env, github_api: _FakeGithub, tmp_path: Path, source: str, token: str
) -> None:
    """One real push to (loopback) github.com, exactly the sibling cell's shape.

    The proxy, the git http-backend and the helper invocation are the sibling
    module's real ones; the only difference from the App cell is which arm
    produced ``token`` — which is the point: delivery is per-key, and the gate
    "helper unchanged" means these cells exercise the same delivery object.
    """
    marker = env.home / "marker.sh"
    marker_log = env.home / "marker.log"
    _marker(marker)
    _node_gitconfig(env.home, marker, marker_log)
    before = _tree_bytes(env.home)

    project_root = tmp_path / "gitroot"
    (project_root / "damianvtran").mkdir(parents=True)
    bare = project_root / "damianvtran" / "scratch.git"
    bare.mkdir()
    git_env = _node_env(env)
    for cmd in (["git", "init", "--bare", "-q"], ["git", "config", "http.receivepack", "true"]):
        assert _run(cmd, env=git_env, cwd=bare).returncode == 0, cmd
    from tests.unit.network.test_credentials_github import _GitBackend

    backend = _GitBackend(project_root, github_api)
    certfile, keyfile = _make_cert("github.com", tmp_path)
    proxy = _HttpsGithubProxy(backend, certfile, keyfile)
    try:
        work = _work_repo(env, tmp_path)
        push_env = _node_env(env)
        push_env.update(github_mod.git_env_for_token(token))
        push_env.update(
            {
                "https_proxy": f"http://127.0.0.1:{proxy.port}",
                "GIT_SSL_NO_VERIFY": "true",
                "MARKER_LOG": str(marker_log),
            }
        )
        pushed = _run(
            ["git", "push", "https://github.com/damianvtran/scratch.git", "main:main"],
            env=push_env,
            cwd=work,
        )
        assert pushed.returncode == 0, (pushed.stdout, pushed.stderr[-2000:])
        assert backend.saw_receive_pack.is_set(), "no receive-pack ever arrived"
        assert any(a["authorised"] and a["token"] == token for a in backend.auth_attempts)
    finally:
        proxy.close()

    assert not (env.home / ".git-credentials").exists()
    marker_text = marker_log.read_text(encoding="utf-8") if marker_log.exists() else ""
    assert "host=github.com" not in marker_text
    # The gh arm's token legitimately LIVES in gh's own hosts file — it is the
    # SOURCE, not a trace of delivery — so that one path is compared
    # before/after instead: delivery must not write anything, anywhere.
    hosts = env.home / ".config" / "gh" / "hosts.yml"
    before_by_name = {name: content for name, content in before}
    for root in (env.home, env.root):
        for relative, content in _tree_bytes(root):
            if Path(root) / relative == hosts:
                continue
            assert token.encode() not in content, f"the token reached {root}/{relative}"
    if hosts.exists():
        assert (
            before_by_name.get(str(hosts.relative_to(env.home))) == hosts.read_bytes()
        ), "delivery rewrote gh's own hosts file"


@pytest.mark.parametrize(
    ("source", "token"),
    [(github_mod.SOURCE_TOKEN, PAT), (github_mod.SOURCE_GH, GHO)],
)
def test_a_push_through_each_token_source_succeeds_and_leaves_no_trace(
    env: _Env,
    github_api: _FakeGithub,
    tmp_path: Path,
    source: str,
    token: str,
) -> None:
    """The note's gate: the real-git loopback cell, extended per source.

    The grant comes off the wire exactly as a borrower gets it, and the push
    uses ONLY the delivered env — so "the token arms deliver like the App arm"
    is proven by git succeeding, not by comparing field-by-field.
    """
    with _owner_for(env, github_api, source) as rig:
        detail = _ask(rig)
        assert detail["access_token"] == token
        assert detail["refreshed"] is False
    _push_through_the_ladder(env, github_api, tmp_path, source, token)
    # The arms never minted anything: no POST ever reached the fake forge.
    assert github_api.mint_requests == []
    assert github_api.revoke_requests == []
