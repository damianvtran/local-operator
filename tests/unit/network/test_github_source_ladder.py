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
import json
import shlex
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
PAT2 = "github_pat_ROTATED_000000000000000000001"
GHO = "gho_test_login_token_0000000000000000000001"
GHO2 = "gho_rotated_login_token_0000000000000000001"


@pytest.fixture(autouse=True)
def _no_daemon(monkeypatch: pytest.MonkeyPatch) -> None:
    """The sibling module's guard, inlined (fixtures cannot wrap other fixtures).

    ``access.retrieve_secret`` reaches the secret broker daemon first when one
    is reachable; pinning the fallback keeps every cell on the keyfile tier and
    launches no process.
    """
    from local_operator.secrets import client as secrets_client

    monkeypatch.setattr(secrets_client, "ensure_broker", lambda *a, **k: False)


# The suite-wide ``_no_real_gh`` guard lives in ``tests/unit/network/conftest.py``
# (widened there in review round 2, n3); cells that exercise the ask-gh path
# override it with ``_pin_gh``.


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
    """The FILE shape (``--insecure-storage`` installs): hosts.yml carries the token."""
    path = home / ".config" / "gh" / "hosts.yml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump({host: {"user": "operator", "oauth_token": token}}), encoding="utf-8"
    )


def _install_stub_gh(
    home: Path, token: str = GHO, *, exit_code: int = 0, log: Path | None = None
) -> Path:
    """A stand-in for the real gh, installed at ``~/.local/bin/gh``.

    Written by the cells (never shipped): prints its token, optionally appends a
    line per invocation, and can fail structurally (exit code + stderr noise that
    the product must NOT reproduce).
    """
    path = home / ".local" / "bin" / "gh"
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "#!/bin/sh\n"
    if log is not None:
        body += f"printf '%s\\n' fired >> {shlex.quote(str(log))}\n"
    if exit_code:
        body += "echo 'gh: not logged in to any hosts' >&2\n"
        body += f"exit {exit_code}\n"
    else:
        body += f"printf '%s\\n' {shlex.quote(token)}\n"
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)
    return path


def _pin_gh(monkeypatch: pytest.MonkeyPatch, stub: Path, *, on_path: bool = True) -> None:
    """Pin ``find_gh``'s discovery: the stub (on PATH, or only via ``~/.local/bin``)."""
    if on_path:
        monkeypatch.setattr(
            github_mod,
            "_resolve_program",
            lambda name, path=None: str(stub) if name == "gh" else None,
        )
    else:
        monkeypatch.setattr(
            github_mod,
            "_resolve_program",
            lambda name, path=None: (str(stub) if (name == "gh" and path is not None) else None),
        )


def _seed_gh_keyring(
    env: _Env,
    monkeypatch: pytest.MonkeyPatch,
    *,
    token: str = GHO,
    log: Path | None = None,
    exit_code: int = 0,
    on_path: bool = True,
) -> Path:
    """The DEFAULT macOS shape: the login is in the OS keychain, hosts.yml tokenless."""
    path = env.home / ".config" / "gh" / "hosts.yml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump({"github.com": {"user": "operator", "git_protocol": "https"}}),
        encoding="utf-8",
    )
    stub = _install_stub_gh(env.home, token, exit_code=exit_code, log=log)
    _pin_gh(monkeypatch, stub, on_path=on_path)
    return stub


class _Rig:
    def __init__(self, env: _Env, broker: Any, audit: Any, document: Any) -> None:
        self.env = env
        self.broker = broker
        self.audit = audit
        self.document = document


@contextlib.contextmanager
def _owner_for(
    env: _Env,
    github_api: _FakeGithub,
    source: str,
    *,
    pem: str = "",
    monkeypatch: pytest.MonkeyPatch | None = None,
    gh_token: str = GHO,
    gh_log: Path | None = None,
) -> Iterator[_Rig]:
    """The broker rig for one ladder arm (mirrors the sibling's ``owner`` fixture).

    Same placement document, same share, same audit — only the ARM differs, so a
    difference between cells is the ladder's, never the rig's. For ``SOURCE_GH``
    the default seed is the KEYRING shape (hosts.yml without a token + a pinned
    stub gh), because that is a default macOS install; cells pass
    ``monkeypatch=None`` to seed the file shape instead.
    """
    if source == github_mod.SOURCE_APP:
        _seed_app_key(env.root, pem)
    elif source == github_mod.SOURCE_TOKEN:
        _seed_secret(env.root, github_mod.TOKEN_SECRET_NAME, PAT)
        github_api.tokens.add(PAT)  # live at the fake forge, for the loopback cells
    elif source == github_mod.SOURCE_GH:
        if monkeypatch is not None:
            _seed_gh_keyring(env, monkeypatch, token=gh_token, log=gh_log)
        else:
            _seed_gh_login(env.home, gh_token)
        github_api.tokens.add(gh_token)
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


def test_read_gh_token_arms(env: _Env, monkeypatch: pytest.MonkeyPatch) -> None:
    """Both shapes: gh itself for a keyring login (the DEFAULT), the file when it has one."""
    with pytest.raises(github_mod.GithubGhError) as absent:
        github_mod.read_gh_token(env.home)
    assert absent.value.kind == "absent"

    path = env.home / ".config" / "gh" / "hosts.yml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"github.com": {"user": "operator"}}), encoding="utf-8")
    # The default macOS shape, with NO gh to ask: unusable, and the message says why.
    with pytest.raises(github_mod.GithubGhError) as tokenless:
        github_mod.read_gh_token(env.home)
    assert tokenless.value.kind == "unusable"
    assert "no gh executable was found to ask" in str(tokenless.value)
    assert "probed PATH" in str(tokenless.value), "the message must say where it looked"

    # The default macOS shape WITH gh: asked directly, the answer is served.
    log = env.home / "gh-invocations.log"
    _seed_gh_keyring(env, monkeypatch, token=GHO, log=log)
    assert github_mod.read_gh_token(env.home) == GHO
    assert log.read_text(encoding="utf-8").splitlines() == ["fired"]

    # The FILE shape wins without spawning gh when it holds a token.
    _pin_gh(monkeypatch, _install_stub_gh(env.home, "stub-should-not-run", log=log))
    _seed_gh_login(env.home, GHO)
    assert github_mod.read_gh_token(env.home) == GHO
    assert log.read_text(encoding="utf-8").splitlines() == [
        "fired"
    ], "a file-carried token must not spawn gh"

    path.write_text("not yaml: [", encoding="utf-8")
    with pytest.raises(github_mod.GithubGhError) as malformed:
        github_mod.read_gh_token(env.home)
    assert malformed.value.kind == "unusable"


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
        # §3.4: for the token arms the helper's allow-list is the ONLY bound, so
        # the owner's designation must travel with the bearer (an App-only
        # delivery would leave these arms unbounded on the borrower).
        assert detail["narrowing"] == {"repositories": [SCRATCH]}
        now_ms = time.time() * 1000
        assert detail["grant_expires_at_ms"] <= now_ms + 900_000 + 5_000
        # No mint was attempted against the forge, and the lender tracks nothing.
        assert github_api.mint_requests == []
        assert rig.broker._lender().outstanding() == 0  # noqa: SLF001 — the registry's own count
        # A second serve re-reads the source (§3.3's refresh path). ROTATE the value
        # in between, or a regression to caching would still pass this cell (n2).
        _seed_secret(env.root, github_mod.TOKEN_SECRET_NAME, PAT2)
        again = _ask(rig)
        assert again["access_token"] == PAT2
        assert again["access_token"] != detail["access_token"]
        assert github_api.mint_requests == []


def test_the_gh_arm_serves_the_login_the_cli_already_holds(
    env: _Env, github_api: _FakeGithub, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The DEFAULT macOS shape: hosts.yml has no token; the broker asks gh itself."""
    log = env.home / "gh-invocations.log"
    with _owner_for(
        env, github_api, github_mod.SOURCE_GH, monkeypatch=monkeypatch, gh_log=log
    ) as rig:
        detail = _ask(rig)
        assert detail["access_token"] == GHO
        assert detail["refreshed"] is False
        assert detail["token_expires_at_ms"] == 0
        # The narrowing rides this arm too (see the token-arm cell's note).
        assert detail["narrowing"] == {"repositories": [SCRATCH]}
        assert github_api.mint_requests == []
        assert rig.broker._lender().outstanding() == 0  # noqa: SLF001
    assert log.read_text(encoding="utf-8").splitlines() == ["fired"]


def test_the_file_shape_still_serves_and_never_spawns_gh(
    env: _Env, github_api: _FakeGithub, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--insecure-storage`` installs keep working: the file token is used as-is."""
    log = env.home / "gh-invocations.log"
    _seed_gh_login(env.home, GHO)
    _pin_gh(monkeypatch, _install_stub_gh(env.home, "stub-should-not-run", log=log))
    with _owner_for(env, github_api, source="") as rig:
        detail = _ask(rig)
    assert detail["access_token"] == GHO
    assert not log.exists(), "a file-carried token must not spawn gh"


def test_gh_found_off_path_is_still_asked(
    env: _Env, github_api: _FakeGithub, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The launchd lesson: gh not on PATH but in ``~/.local/bin`` is still found and asked."""
    log = env.home / "gh-invocations.log"
    _seed_gh_keyring(env, monkeypatch, token=GHO, log=log, on_path=False)
    with _owner_for(env, github_api, source="") as rig:
        detail = _ask(rig)
    assert detail["access_token"] == GHO
    assert log.read_text(encoding="utf-8").splitlines() == ["fired"]


def test_find_gh_probes_the_standard_prefixes_under_the_launchd_env(
    env: _Env, github_api: _FakeGithub, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B2's repro shape, isolated: launchd's PATH, gh ONLY in a standard prefix.

    launchd hands the relay PATH=/usr/bin:/bin:/usr/sbin:/sbin (measured on the
    operator's machine), and a Homebrew gh lives at /opt/homebrew/bin/gh —
    outside every probe that existed before this fix. The prefix fallback must
    find it anyway and the arm must serve. The prefix is a temp directory
    monkeypatched over ``GH_FALLBACK_BIN_DIRS`` (a test must not write into the
    real /opt/homebrew); the REAL constant is pinned first, and the discovery
    seam is put back to the real ``shutil.which`` for this cell so the fallback
    probe itself is exercised rather than a stub pin.
    """
    import shutil

    assert github_mod.GH_FALLBACK_BIN_DIRS == ("/opt/homebrew/bin", "/usr/local/bin")
    monkeypatch.setenv("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
    if shutil.which("gh") is not None:
        # The cell's premise is that the ONLY gh lives in the fallback prefix.
        # A distro package (e.g. the ubuntu runner image's gh deb at /usr/bin/gh)
        # makes that premise false — state it instead of assuming it; the
        # fallback-dirs behaviour itself is pinned platform-independently by
        # ``test_find_gh_probes_fallback_dirs_in_order_platform_independently``.
        pytest.skip(
            "a gh already sits on the launchd-shaped PATH (a distro package): "
            "this cell exercises the fallback-prefix discovery on hosts where "
            "the launchd PATH carries none"
        )

    prefix = env.tmp / "opt-homebrew-bin"
    prefix.mkdir()
    stub = _install_stub_gh(env.home, GHO)
    (prefix / "gh").write_bytes(stub.read_bytes())
    (prefix / "gh").chmod(0o755)
    stub.unlink()  # gh exists ONLY in the prefix, as on the relay's machine
    monkeypatch.setattr(github_mod, "GH_FALLBACK_BIN_DIRS", (str(prefix),))
    # Undo the suite guard for this cell: exercise the REAL discovery seam.
    monkeypatch.setattr(
        github_mod, "_resolve_program", lambda name, path=None: shutil.which(name, path=path)
    )

    assert github_mod.find_gh(env.home) == str(prefix / "gh")
    # And the arm serves through the normal door under this environment.
    hosts = env.home / ".config" / "gh" / "hosts.yml"
    hosts.parent.mkdir(parents=True, exist_ok=True)
    hosts.write_text(
        yaml.safe_dump({"github.com": {"user": "operator", "git_protocol": "https"}}),
        encoding="utf-8",
    )
    with _owner_for(env, github_api, source="") as rig:
        detail = _ask(rig)
    assert detail["access_token"] == GHO


def test_find_gh_probes_fallback_dirs_in_order_platform_independently(
    env: _Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fallback LOOP and its order, seam-only: runs on every platform.

    ``find_gh`` probes this process's PATH, then ``~/.local/bin``, then
    ``GH_FALLBACK_BIN_DIRS`` in order. This cell drives the real loop against a
    fake file tree (the seam mirrors ``shutil.which`` truthfully against the
    fixture), so it neither needs a gh-installed host nor the absence of one —
    the launchd-env cell above owns the restricted-PATH story, and this one
    owns the order, which must hold everywhere.
    """
    from pathlib import Path as _Path

    real_dirs = github_mod.GH_FALLBACK_BIN_DIRS
    assert real_dirs == ("/opt/homebrew/bin", "/usr/local/bin"), "the standard prefixes"

    prefix = env.tmp / "prefix-bin"
    prefix.mkdir()
    stub = _install_stub_gh(env.home, GHO)  # lands in ~/.local/bin
    (prefix / "gh").write_bytes(stub.read_bytes())
    (prefix / "gh").chmod(0o755)

    calls: list[str | None] = []

    def fake(name: str, path: str | None = None) -> str | None:
        calls.append(path)
        if path is None:
            return None  # the process-PATH probe misses
        candidate = _Path(path) / name
        return str(candidate) if candidate.exists() else None

    monkeypatch.setattr(github_mod, "_resolve_program", fake)
    monkeypatch.setattr(github_mod, "GH_FALLBACK_BIN_DIRS", (str(prefix),))

    # User-local bin wins over the prefixes when both exist.
    assert github_mod.find_gh(env.home) == str(stub)
    stub.unlink()
    # With it gone, the prefix answers — and NOT before the user-local probe ran.
    assert github_mod.find_gh(env.home) == str(prefix / "gh")
    local_bin = str(env.home / ".local" / "bin")
    assert calls == [None, local_bin, None, local_bin, str(prefix)], calls


def test_gh_that_cannot_answer_refuses_with_a_structural_message(
    env: _Env, github_api: _FakeGithub, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing gh (logged out / broken) refuses by name; its stderr is never reproduced."""
    _seed_gh_keyring(env, monkeypatch, token="", exit_code=7)
    with _owner_for(env, github_api, source="") as rig:
        detail = _ask(rig)
    assert detail["code"] == github_mod.CODE_GH_UNUSABLE, detail
    assert "exit 7" in detail["message"]
    assert (
        "not logged in to any hosts" not in detail["message"]
    ), "gh's stderr is account material and must not travel"


def test_a_gh_re_login_reaches_the_next_serve(
    env: _Env, github_api: _FakeGithub, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Per-serve, no cache (§3.3): a re-login changes what the NEXT serve carries."""
    log = env.home / "gh-invocations.log"
    with _owner_for(
        env, github_api, github_mod.SOURCE_GH, monkeypatch=monkeypatch, gh_log=log
    ) as rig:
        first = _ask(rig)
        assert first["access_token"] == GHO
        _install_stub_gh(env.home, GHO2, log=log)  # the "re-login"
        second = _ask(rig)
        assert second["access_token"] == GHO2
        assert log.read_text(encoding="utf-8").splitlines() == ["fired", "fired"]


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


@pytest.mark.parametrize("layer", ["master_key", "store_db"])
def test_a_corrupt_store_stops_the_ladder_and_never_serves_the_gh_login(
    env: _Env, github_api: _FakeGithub, monkeypatch: pytest.MonkeyPatch, layer: str
) -> None:
    """M1 / QA-Q2, reproduced at BOTH corruption layers (m2).

    The QA door's shape: a ``GITHUB_TOKEN`` in the store, a full gh login
    present, one store layer corrupted-but-present. Before the fix, resolution
    fell through and the broker SERVED the gh login (a wider credential than the
    one sitting unreadable on disk); now the ladder stops and refuses by name.

    Both layers are pinned because they exercise DIFFERENT except arms: a zeroed
    ``master.key`` makes ``open_store`` itself raise (the first arm), while a
    zeroed ``store.db`` lets ``open_store`` succeed and the metadata read raise
    (the second) — a regression in either classification would otherwise pass CI.
    """
    from local_operator.secrets.keys import key_path, store_path

    target = key_path if layer == "master_key" else store_path
    log = env.home / "gh-invocations.log"
    _seed_secret(env.root, github_mod.TOKEN_SECRET_NAME, PAT)
    _seed_gh_keyring(env, monkeypatch, token=GHO, log=log)
    assert github_mod.resolve_source(env.root) == github_mod.SOURCE_TOKEN  # control
    original = target(env.root).read_bytes()
    target(env.root).write_bytes(b"\x00" * 32)
    try:
        assert github_mod.resolve_source(env.root) == github_mod.SOURCE_UNREADABLE
        assert github_mod.source_present(env.root) is False
        # The closed bool view stays closed (a promise surface must not claim it).
        assert github_mod.app_secret_present(env.root) is False
        with _owner_for(env, github_api, source="") as rig:
            detail = _ask(rig)
        assert detail["code"] == github_mod.CODE_STORE_UNREADABLE, detail
        assert GHO not in json.dumps(detail), "the gh login must never be served"
        assert github_api.mint_requests == []
        assert not log.exists(), "gh must not even be asked while the store is unreadable"
        with pytest.raises(github_mod.GithubTokenError) as broken:
            github_mod.read_token_secret(env.root)
        assert broken.value.kind == "unusable", "present-but-unreadable is not 'absent'"
    finally:
        target(env.root).write_bytes(original)
    assert github_mod.resolve_source(env.root) == github_mod.SOURCE_TOKEN  # restored


def test_the_share_verb_refuses_by_name_on_a_corrupt_store(
    env: _Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The door's other half: a share must not promise a device that cannot resolve."""
    from local_operator.network import cli as network_cli
    from local_operator.network.credentials import offers
    from local_operator.network.types import MeshRefusal
    from local_operator.secrets.keys import store_path

    _seed_secret(env.root, github_mod.TOKEN_SECRET_NAME, PAT)
    _seed_gh_keyring(env, monkeypatch, token=GHO)
    original = store_path(env.root).read_bytes()
    store_path(env.root).write_bytes(b"\x00" * 32)
    try:
        monkeypatch.setattr(network_cli, "_config_dir", lambda: env.root)
        with pytest.raises(MeshRefusal) as refusal:
            network_cli._require_local_credential(
                github_mod.GITHUB_KEY, github_mod.GITHUB_KEY
            )  # noqa: SLF001
        assert refusal.value.code == github_mod.CODE_STORE_UNREADABLE
        assert offers.credential_here(github_mod.GITHUB_KEY, env.root) is False
    finally:
        store_path(env.root).write_bytes(original)


def test_a_tokenless_gh_login_refuses_by_name(env: _Env, github_api: _FakeGithub) -> None:
    path = env.home / ".config" / "gh" / "hosts.yml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"github.com": {"user": "operator"}}), encoding="utf-8")
    with _owner_for(env, github_api, source="") as rig:
        detail = _ask(rig)
        assert detail["code"] == github_mod.CODE_GH_UNUSABLE, detail
        assert "no gh executable was found to ask" in detail["message"]


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
        (github_mod.CODE_STORE_UNREADABLE, "secret store"),
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
    # The gh arm's token legitimately LIVES in its SOURCES — gh's hosts file in
    # the file shape, the pinned stub script in the keyring shape — so those
    # paths are compared before/after instead: delivery must not write anything,
    # anywhere, and a stray write to a source is still a landing.
    allowed = {env.home / ".config" / "gh" / "hosts.yml", env.home / ".local" / "bin" / "gh"}
    before_by_name = {name: content for name, content in before}
    for root in (env.home, env.root):
        for relative, content in _tree_bytes(root):
            if Path(root) / relative in allowed:
                continue
            assert token.encode() not in content, f"the token reached {root}/{relative}"
    for path in sorted(allowed):
        if path.exists():
            assert (
                before_by_name.get(str(path.relative_to(env.home))) == path.read_bytes()
            ), f"delivery rewrote {path}"


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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The note's gate: the real-git loopback cell, extended per source.

    The grant comes off the wire exactly as a borrower gets it, and the push
    uses ONLY the delivered env — so "the token arms deliver like the App arm"
    is proven by git succeeding, not by comparing field-by-field. The gh arm
    runs the KEYRING shape (the default install): a tokenless hosts file plus
    the pinned stub gh.
    """
    with _owner_for(env, github_api, source, monkeypatch=monkeypatch) as rig:
        detail = _ask(rig)
        assert detail["access_token"] == token
        assert detail["refreshed"] is False
    _push_through_the_ladder(env, github_api, tmp_path, source, token)
    # The arms never minted anything: no POST ever reached the fake forge.
    assert github_api.mint_requests == []
    assert github_api.revoke_requests == []
