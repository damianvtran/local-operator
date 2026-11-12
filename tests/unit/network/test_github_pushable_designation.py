"""The default GitHub designation: every repo the owner's login can push to, minus opt-outs.

Operator decision (2026-11-12): with no explicit
``network.credentials.github.repositories``, a share lends for the repositories the
owner's serving arm can PUSH to, minus ``network.credentials.github.excluded`` —
always an enumerated list (never a wildcard, never an installation-wide grant),
shown on the receipt and in readiness, narrowed on the App mint and delivered on
the grant so the borrower's helper enforces it.

Every cell drives the REAL owner broker (``MeshCredentialBroker.on_broker``)
against a loopback GitHub that answers the documented shapes: the App mint, the
installation listing, ``/user/repos`` with ``permissions``, and the revoke. No
cell reaches api.github.com (the network conftest also pins the default client
offline).
"""

from __future__ import annotations

import contextlib
import datetime
import json
import os
import re
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

import pytest
import yaml

from local_operator.network import cli as net_cli
from local_operator.network import readiness
from local_operator.network.credentials import github as github_mod
from local_operator.network.credentials import owner as owner_mod
from local_operator.network.credentials import placement as placement_mod
from local_operator.network.credentials import types as cred_types
from local_operator.paths import CONFIG_DIR_ENV
from tests.unit.network.test_credentials_github import (
    APP_ID,
    BORROWER_DEVICE,
    OWNER_DEVICE,
    _detail,
    _Env,
    _grant_frame,
    _Link,
    _seed_app_key,
)
from tests.unit.network.test_credentials_real_link import _lop_network
from tests.unit.network.test_github_source_ladder import _seed_secret
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixtures by import
    _pair,
    devices,
)


@pytest.fixture(scope="module")
def app_keypair() -> tuple[str, str]:
    """A real RSA keypair (the sibling file's shape): PEM in the store, public verifies."""
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
    """An isolated HOME and config dir (the sibling file's ``env``)."""
    holder = _Env(tmp_path)
    monkeypatch.setenv("HOME", str(holder.home))
    monkeypatch.setenv(CONFIG_DIR_ENV, str(holder.root))
    return holder


#: A token-arm value. Synthetic; the fake forge authenticates by exact value.
TOKEN = "ghp_designation_test_token_000000000000001"
OTHER_TOKEN = "ghp_designation_other_account_00000000001"


def _repo(full_name: str, *, push: bool = True) -> dict[str, Any]:
    return {"full_name": full_name, "permissions": {"admin": False, "push": push, "pull": True}}


@dataclass
class _Forge:
    """Mutable answers the loopback forge serves; every request is recorded."""

    user_repos: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    installation_repos: list[dict[str, Any]] = field(default_factory=list)
    fail_status: int = 0
    #: Fail only this page of a listing (the walk breaks partway through).
    fail_page: int = 0
    #: Answer this many revokes with 500 before honouring one.
    fail_revokes: int = 0
    gets: list[str] = field(default_factory=list)
    mints: list[dict[str, Any]] = field(default_factory=list)
    revoked: list[str] = field(default_factory=list)
    tokens: set[str] = field(default_factory=set)


class _FakeGithubApi:
    """``api.github.com`` over loopback: mint, revoke, and the two listings."""

    def __init__(self, public_pem: str) -> None:
        import jwt

        self.forge = _Forge()
        forge = self.forge
        lock = threading.Lock()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                pass

            def _json(self, status: int, payload: Any) -> None:
                data = b"" if payload is None else json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                if data:
                    self.wfile.write(data)

            def _page(self, rows: list[Any]) -> list[Any]:
                query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
                per_page = int(query.get("per_page", ["30"])[0])
                page = int(query.get("page", ["1"])[0])
                return rows[(page - 1) * per_page : page * per_page]

            def do_GET(self) -> None:  # noqa: N802
                with lock:
                    forge.gets.append(self.path)
                if forge.fail_status:
                    self._json(forge.fail_status, {"message": "forced failure"})
                    return
                query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
                if forge.fail_page and query.get("page") == [str(forge.fail_page)]:
                    self._json(500, {"message": "forced page failure"})
                    return
                token = self.headers.get("Authorization", "").removeprefix("Bearer ")
                path = urllib.parse.urlsplit(self.path).path
                if path == "/user/repos":
                    if token not in forge.user_repos:
                        self._json(401, {"message": "Bad credentials"})
                        return
                    self._json(200, self._page(forge.user_repos[token]))
                    return
                if path == "/installation/repositories":
                    if token not in forge.tokens:
                        self._json(401, {"message": "Bad credentials"})
                        return
                    rows = self._page(forge.installation_repos)
                    self._json(
                        200, {"total_count": len(forge.installation_repos), "repositories": rows}
                    )
                    return
                self._json(404, {"message": "not found"})

            def do_POST(self) -> None:  # noqa: N802
                if not re.fullmatch(r"/app/installations/\d+/access_tokens", self.path):
                    self._json(404, {"message": "not found"})
                    return
                try:
                    jwt.decode(
                        self.headers.get("Authorization", "").removeprefix("Bearer "),
                        public_pem,
                        algorithms=["RS256"],
                        issuer=APP_ID,
                        options={"verify_aud": False},
                    )
                except Exception:  # noqa: BLE001 — a forged JWT is a 401
                    self._json(401, {"message": "Bad credentials"})
                    return
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
                token = f"ghs_designation_{os.urandom(4).hex()}"
                with lock:
                    forge.mints.append(body)
                    forge.tokens.add(token)
                expires = (
                    datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1)
                ).strftime("%Y-%m-%dT%H:%M:%SZ")
                self._json(201, {"token": token, "expires_at": expires})

            def do_DELETE(self) -> None:  # noqa: N802
                token = self.headers.get("Authorization", "").removeprefix("Bearer ")
                with lock:
                    failing = forge.fail_revokes > 0
                    if failing:
                        forge.fail_revokes -= 1
                    else:
                        forge.revoked.append(token)
                        forge.tokens.discard(token)
                if failing:
                    self._json(500, {"message": "forced revoke failure"})
                    return
                self._json(204, None)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def minter(self) -> github_mod.GithubMinter:
        return github_mod.GithubMinter(base_url=self.url)

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture()
def api(app_keypair: tuple[str, str]) -> Iterator[_FakeGithubApi]:
    fake = _FakeGithubApi(app_keypair[1])
    try:
        yield fake
    finally:
        fake.close()


def _config(root: Path, **github: Any) -> None:
    """Write ``network.credentials.github`` (only the keys given; absent = unset)."""
    (root / "config.yml").write_text(
        yaml.safe_dump({"values": {"network": {"credentials": {"github": dict(github)}}}}),
        encoding="utf-8",
    )


@contextlib.contextmanager
def _broker(env: _Env, api: _FakeGithubApi) -> Iterator[Any]:
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
        github_minter=api.minter(),
    )
    try:
        yield broker
    finally:
        broker.close()
        audit.close()


def _ask(broker: Any) -> dict[str, Any]:
    reply = broker.on_broker(_Link(BORROWER_DEVICE), _grant_frame(BORROWER_DEVICE))
    return reply


def _granted(broker: Any) -> dict[str, Any]:
    """A served grant; a refusal fails here with its code and sentence visible."""
    detail = _detail(_ask(broker))
    assert detail.get("kind") == "grant", (detail.get("code"), detail.get("message"))
    return detail


def _refused(broker: Any) -> dict[str, Any]:
    """The broker's refusal detail: an ``ack`` whose detail is ``kind: error``."""
    reply = _ask(broker)
    detail = reply.get("detail")
    assert reply.get("op") == "ack" and isinstance(detail, dict), reply
    assert detail.get("kind") == "error", reply
    assert "access_token" not in detail and "narrowing" not in detail, reply
    return detail


def _cli(*argv: str) -> int:
    """``lop network …`` through the REAL parser and dispatcher."""
    import argparse

    parser = argparse.ArgumentParser(prog="lop")
    net_cli.add_parser(parser.add_subparsers(dest="command"))
    return net_cli.main(parser.parse_args(["network", *argv]))


def _token_arm(env: _Env, api: _FakeGithubApi, rows: list[dict[str, Any]]) -> None:
    _seed_secret(env.root, github_mod.TOKEN_SECRET_NAME, TOKEN)
    api.forge.user_repos[TOKEN] = rows


def _age_cache(root: Path, seconds: float) -> None:
    record = github_mod.read_enumeration_cache(root)
    record["fetched_at"] = float(record["fetched_at"]) - seconds
    github_mod._write_enumeration_cache(root, record)  # noqa: SLF001


# ---------------------------------------------------------------------------
# RED on main: an unset designation refused; now it is enumerated
# ---------------------------------------------------------------------------


def test_an_unset_designation_mints_the_app_narrowed_to_the_installation_list(
    env: _Env, api: _FakeGithubApi, app_keypair: tuple[str, str]
) -> None:
    """Main refused ``github_repositories_unset``. Now: the installation's repos,
    minus opt-outs, as the mint's ``repositories`` and the grant's narrowing; the
    listing token is metadata-read only and revoked before the serve returns."""
    _seed_app_key(env.root, app_keypair[0])
    api.forge.installation_repos = [_repo(f"acme/repo-{n:03d}") for n in range(150)]
    _config(env.root, excluded=["acme/repo-007", "ACME/Repo-100"])
    with _broker(env, api) as broker:
        detail = _granted(broker)
    expected = [f"acme/repo-{n:03d}" for n in range(150) if n not in (7, 100)]
    assert detail["narrowing"]["repositories"] == expected
    listing, lending = api.forge.mints
    assert listing == {"permissions": {"metadata": "read"}}
    assert lending["permissions"] == github_mod.GITHUB_PERMISSIONS
    assert lending["repositories"] == [name.split("/")[1] for name in expected]
    # The paginated walk: 150 rows at per_page=100 is two pages.
    assert sum("/installation/repositories" in path for path in api.forge.gets) == 2
    # The one installation-wide (metadata-only) token did not outlive the walk.
    assert len(api.forge.revoked) == 1 and api.forge.revoked[0] not in api.forge.tokens


def test_the_token_arm_lends_only_what_the_login_can_push_to_across_pages(
    env: _Env, api: _FakeGithubApi
) -> None:
    rows = [_repo(f"me/r{n:03d}", push=n % 3 != 0) for n in range(230)]
    _token_arm(env, api, rows)
    with _broker(env, api) as broker:
        detail = _granted(broker)
    pushable = [f"me/r{n:03d}" for n in range(230) if n % 3 != 0]
    assert detail["narrowing"]["repositories"] == pushable
    user_gets = [path for path in api.forge.gets if path.startswith("/user/repos")]
    assert len(user_gets) == 3
    assert "affiliation=owner,collaborator,organization_member" in user_gets[0]
    assert "per_page=100" in user_gets[0]
    # Nothing was minted: the token arm serves the owner's own token.
    assert api.forge.mints == []


# ---------------------------------------------------------------------------
# Opt-outs, explicit override, and the cache bound
# ---------------------------------------------------------------------------


def test_an_explicit_list_overrides_and_never_enumerates(env: _Env, api: _FakeGithubApi) -> None:
    _token_arm(env, api, [_repo("me/a"), _repo("me/b")])
    _config(env.root, repositories=["me/hand-picked"], excluded=[])
    with _broker(env, api) as broker:
        detail = _granted(broker)
    assert detail["narrowing"]["repositories"] == ["me/hand-picked"]
    assert api.forge.gets == []


def test_the_cache_is_reused_inside_its_bound_and_refreshed_after(
    env: _Env, api: _FakeGithubApi
) -> None:
    _token_arm(env, api, [_repo("me/a")])
    with _broker(env, api) as broker:
        _granted(broker)
        _granted(broker)
        assert len(api.forge.gets) == 1, "a fresh list was re-walked"
        api.forge.user_repos[TOKEN] = [_repo("me/a"), _repo("me/new")]
        _age_cache(env.root, github_mod.ENUMERATION_MAX_AGE_S + 1)
        detail = _granted(broker)
    assert len(api.forge.gets) == 2
    assert detail["narrowing"]["repositories"] == ["me/a", "me/new"]


def test_a_failed_refresh_keeps_the_last_good_list_and_says_so(
    env: _Env, api: _FakeGithubApi
) -> None:
    _token_arm(env, api, [_repo("me/a"), _repo("me/b")])
    with _broker(env, api) as broker:
        _granted(broker)
        _age_cache(env.root, github_mod.ENUMERATION_MAX_AGE_S + 60)
        api.forge.fail_status = 503
        detail = _granted(broker)
    assert detail["narrowing"]["repositories"] == ["me/a", "me/b"]
    designated = github_mod.designation(env.root, source=github_mod.SOURCE_TOKEN, allow_fetch=False)
    assert designated.origin == github_mod.ORIGIN_LAST_GOOD
    assert "503" in designated.error or "forced failure" in designated.error
    fact = readiness.github_fact(env.root, env.home)
    assert fact["repositories_origin"] == "last_good"
    clause = readiness._github_scope_clause(fact, "owner-laptop")  # noqa: SLF001
    assert "the list fetched 6 h ago" in clause, clause
    assert "could not reach GitHub at its last check" in clause, clause
    assert "full list: lop network credential github repos on owner-laptop" in clause, clause


@pytest.mark.parametrize("status", [401, 503])
def test_no_last_good_list_refuses_by_name_and_never_widens(
    env: _Env, api: _FakeGithubApi, status: int
) -> None:
    _token_arm(env, api, [_repo("me/a")])
    api.forge.fail_status = status
    with _broker(env, api) as broker:
        reply = _refused(broker)
    assert reply["code"] == github_mod.CODE_REPOSITORIES_UNSET, reply
    # A device-neutral detail: the borrower's catalogue names the owner (QA F1).
    assert "could not be fetched" in reply["message"], reply
    assert "this device" not in reply["message"], reply
    assert "lop network credential github repos --refresh" in reply["message"], reply


def test_another_accounts_list_is_never_served_as_last_good(env: _Env, api: _FakeGithubApi) -> None:
    """A re-login as another user must not inherit the previous login's list."""
    _token_arm(env, api, [_repo("me/a")])
    with _broker(env, api) as broker:
        _granted(broker)
    _seed_secret(env.root, github_mod.TOKEN_SECRET_NAME, OTHER_TOKEN)
    api.forge.fail_status = 503
    with _broker(env, api) as broker:
        reply = _refused(broker)
    assert reply["code"] == github_mod.CODE_REPOSITORIES_UNSET


def test_opting_everything_out_refuses_with_that_reason(env: _Env, api: _FakeGithubApi) -> None:
    _token_arm(env, api, [_repo("me/a")])
    _config(env.root, excluded=["me/a"])
    with _broker(env, api) as broker:
        reply = _refused(broker)
    assert "opted out" in reply["message"], reply


# ---------------------------------------------------------------------------
# Size: past the old 64, and the honest refusals above the real limits
# ---------------------------------------------------------------------------


def test_a_list_past_64_travels_whole_across_the_peer_boundary_and_lands(
    env: _Env, api: _FakeGithubApi, tmp_path: Path
) -> None:
    rows = [_repo(f"org/service-{n:04d}") for n in range(300)]
    _token_arm(env, api, rows)
    with _broker(env, api) as broker:
        detail = _granted(broker)
    # The borrower's parse of the wire detail: the boundary used to cut at 64.
    grant = cred_types.Grant.from_detail(detail)
    assert len(grant.narrowing["repositories"]) == 300
    borrower = tmp_path / "borrower"
    borrower.mkdir()
    result = github_mod.remember_delivered_repositories(
        grant.narrowing["repositories"], root=borrower, owner_device=OWNER_DEVICE
    )
    assert result["changed"] and not result["error"], result
    assert len(github_mod.repositories_for(borrower)) == 300


def test_the_boundary_drops_an_oversized_list_whole_never_a_prefix() -> None:
    too_many = [f"o/r{n}" for n in range(cred_types.MAX_NARROWING_LIST_ENTRIES + 1)]
    assert cred_types._bounded_narrowing({"repositories": too_many}) == {}  # noqa: SLF001
    assert github_mod.MAX_DESIGNATED_REPOSITORIES == cred_types.MAX_NARROWING_LIST_ENTRIES


def test_an_app_designation_over_githubs_mint_limit_is_refused_not_cut(
    env: _Env, api: _FakeGithubApi, app_keypair: tuple[str, str]
) -> None:
    _seed_app_key(env.root, app_keypair[0])
    api.forge.installation_repos = [
        _repo(f"acme/r{n:04d}") for n in range(github_mod.APP_MINT_MAX_REPOSITORIES + 1)
    ]
    with _broker(env, api) as broker:
        reply = _refused(broker)
    assert reply["code"] == github_mod.CODE_REPOSITORIES_OVERFLOW, reply
    assert reply["message"] == "501 repositories; one share can carry at most 500", reply
    # Only the metadata-read listing token was ever minted, and it was revoked.
    assert [mint.get("permissions") for mint in api.forge.mints] == [{"metadata": "read"}]


# ---------------------------------------------------------------------------
# Removal converges on the next borrow, through the CLI the operator uses
# ---------------------------------------------------------------------------


def test_an_opt_out_stops_the_borrower_pushing_at_the_next_borrow(
    env: _Env, api: _FakeGithubApi, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:

    _token_arm(env, api, [_repo("me/keep"), _repo("me/secret-ops")])
    borrower = tmp_path / "borrower"
    borrower.mkdir()

    def borrow_and_land() -> tuple[str, ...]:
        with _broker(env, api) as broker:
            detail = _granted(broker)
        github_mod.remember_delivered_repositories(
            detail["narrowing"]["repositories"], root=borrower, owner_device=OWNER_DEVICE
        )
        return github_mod.repositories_for(borrower)

    def helper_serves(path: str, allowed: tuple[str, ...]) -> bool:
        lines = ["protocol=https", "host=github.com", f"path={path}.git"]
        return bool(github_mod.git_helper_reply("get", lines, token="t", repositories=allowed))

    allowed = borrow_and_land()
    assert helper_serves("me/secret-ops", allowed)

    assert _cli("credential", "github", "exclude", "me/secret-ops", "--json") == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["changed"] is True and receipt["excluded"] == ["me/secret-ops"]

    allowed = borrow_and_land()
    assert allowed == ("me/keep",)
    assert not helper_serves("me/secret-ops", allowed)
    assert helper_serves("me/keep", allowed)

    assert _cli("credential", "github", "include", "me/secret-ops", "--json") == 0
    capsys.readouterr()
    assert borrow_and_land() == ("me/keep", "me/secret-ops")


def test_the_repos_command_lists_origin_count_and_the_opt_out_step(
    env: _Env,
    api: _FakeGithubApi,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:

    _token_arm(env, api, [_repo("me/a"), _repo("me/b"), _repo("me/ro", push=False)])
    _config(env.root, excluded=["me/b"])
    monkeypatch.setattr(github_mod, "default_minter", api.minter)
    assert _cli("credential", "github", "repos", "--refresh") == 0
    text = capsys.readouterr().out
    assert "GitHub repositories lent for: 1" in text, text
    assert "minus 1 opted out" in text, text
    assert "  me/a" in text and "me/ro" not in text, text
    assert "lop network credential github exclude <owner/repo>" in text, text
    assert _cli("credential", "github", "repos", "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["origin"] == "enumerated" and payload["repositories"] == ["me/a"]
    assert payload["excluded"] == ["me/b"] and payload["excluded_count"] == 1
    assert _cli("credential", "github", "exclude", "not-a-repo", "--json") == 1
    refusal = json.loads(capsys.readouterr().out)
    assert refusal["ok"] is False and refusal["code"] == "bad_request"


def test_readiness_never_calls_github(env: _Env, api: _FakeGithubApi) -> None:
    _token_arm(env, api, [_repo("me/a")])
    fact = readiness.github_fact(env.root, env.home)
    assert api.forge.gets == []
    assert fact["repositories_origin"] == "unavailable"
    assert fact["repositories"] == [] and fact["repositories_count"] == 0
    clause = readiness._github_scope_clause(fact, "owner-laptop")  # noqa: SLF001
    assert "no repository listed yet" in clause, clause
    assert "refused rather than widened" in clause, clause


# ---------------------------------------------------------------------------
# The real path: `credential share github` over two loopback relays, then the
# viewer's readiness row — what the operator actually reads
# ---------------------------------------------------------------------------


def test_the_share_receipt_and_readiness_show_the_list_its_origin_and_the_opt_out(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    api: _FakeGithubApi,
    tmp_path: Path,
) -> None:
    from local_operator.network import store

    # The git row carries the GitHub clause once a git author resolves. Both
    # in-process relays read ``~/.gitconfig`` (``readiness.git_identity_fact``),
    # so HOME is a synthetic one holding a synthetic author.
    home = tmp_path / "home"
    home.mkdir()
    (home / ".gitconfig").write_text("[user]\n\tname = Probe\n\temail = probe@example.test\n")
    monkeypatch.setenv("HOME", str(home))

    server_a, server_b, host, port = request.getfixturevalue("devices")
    record, _host, _port = _pair((server_a, server_b, host, port), monkeypatch, role="drive")
    # B's relay listens and knows A's address, so the viewer's `ready` can dial
    # the owner (the `drive_mesh` rig's wiring, test_credentials_share_push.py).
    server_b.bind()
    server_b.bind_control()
    server_b.start()
    with store.mutate(record.network_id, server_b.root) as copy:
        owner_row = copy.member(server_a.identity.device_id)
        assert owner_row is not None
        owner_row.endpoints = [f"{host}:{port}"]
        store.save(copy, server_b.root)
    _seed_secret(server_a.root, github_mod.TOKEN_SECRET_NAME, TOKEN)
    api.forge.user_repos[TOKEN] = [
        _repo("me/app"),
        _repo("me/site"),
        _repo("me/secrets"),
        _repo("org/readonly", push=False),
    ]
    _config(server_a.root, excluded=["me/secrets"])
    # The CLI's own client (``default_minter``) is pointed at the loopback forge.
    monkeypatch.setattr(github_mod, "default_minter", api.minter)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_a.root))
    capsys.readouterr()
    assert _lop_network("credential", "share", "github", "--with", server_b.identity.name) == 0
    receipt = capsys.readouterr().out
    assert "GitHub repositories lent for: 2" in receipt, receipt
    assert "every repository this device's GitHub login can push to" in receipt, receipt
    assert "minus 1 opted out" in receipt, receipt
    assert "  me/app" in receipt and "  me/site" in receipt, receipt
    assert "  me/secrets" not in receipt and "org/readonly" not in receipt, receipt
    assert "opted out (1): me/secrets — bring one back" in receipt, receipt
    assert "lop network credential github exclude <owner/repo>" in receipt, receipt
    # The share refreshed the list: exactly one walk of /user/repos.
    assert sum(path.startswith("/user/repos") for path in api.forge.gets) == 1

    # The viewer (B) asks the owner's readiness: the list, its origin, the opt-out
    # count — from the owner's cache, with no further call to GitHub.
    gets_before = len(api.forge.gets)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_b.root))
    assert _lop_network("ready", "--peer", server_a.identity.name) in (0, 1)
    ready = capsys.readouterr().out
    assert len(api.forge.gets) == gets_before
    assert "me/app and me/site (2 — every repository its login can push to" in ready, ready
    assert "full list: lop network credential github repos on device-a" in ready, ready
    assert "1 opted out" in ready, ready
    # Evidence for the PR (shown with -rP): the receipt and the readiness row.
    print("=== share receipt ===")
    print(receipt)
    print("=== readiness (viewer B about owner A) ===")
    print("\n".join(line for line in ready.splitlines() if "github" in line.lower()))
    server_b.stop()


# ---------------------------------------------------------------------------
# Round-1 remediation (UX U1–U4, N2–N5; review M1, M3, M4, N1; QA F1, F2)
# ---------------------------------------------------------------------------


def _cli_text(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str]:
    capsys.readouterr()
    code = _cli(*argv)
    return code, capsys.readouterr().out


def test_u1_opted_out_names_are_shown_and_a_typo_is_warned_then_stored(
    env: _Env,
    api: _FakeGithubApi,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """U1: names, not a count; ``exclude`` of a name not in the list says so and
    stores it anyway (a repo the login gains later arrives opted out)."""
    _token_arm(env, api, [_repo("me/a"), _repo("me/b")])
    monkeypatch.setattr(github_mod, "default_minter", api.minter)
    code, text = _cli_text(capsys, "credential", "github", "exclude", "me/b")
    assert code == 0
    assert "me/b is opted out — devices you share with lose push to it the next time" in text
    assert "within 15 min" in text, text
    assert "opted out (1): me/b — bring one back: lop network credential github include" in text
    code, text = _cli_text(capsys, "credential", "github", "exclude", "me/bb")
    with capsys.disabled():
        print("=== exclude me/bb (a typo of me/b) ===")
        print(text)
    assert code == 0
    assert (
        "me/bb isn't in the list this device's GitHub login can push to — check the "
        "spelling; it's opted out anyway, in case it's added later"
    ) in text, text
    assert "opted out (2): me/b, me/bb" in text, text
    assert github_mod.excluded_repositories(env.root) == ("me/b", "me/bb")
    code, text = _cli_text(capsys, "credential", "github", "repos")
    assert "opted out (2): me/b, me/bb — bring one back" in text, text
    code, text = _cli_text(capsys, "credential", "github", "include", "me/b")
    with capsys.disabled():
        print("=== include me/b ===")
        print(text)
    assert "me/b is back in — devices you share with can push to it again" in text, text
    assert "opted out (1): me/bb" in text, text
    code, text = _cli_text(capsys, "credential", "github", "include", "me/zz")
    assert "me/zz was not opted out" in text, text
    code, text = _cli_text(capsys, "credential", "github", "include", "me/bb", "--json")
    payload = json.loads(text)
    assert payload["in_list"] is False and payload["changed"] is True
    assert payload["excluded"] == []


def test_u2_repos_lists_every_name_and_the_receipt_keeps_a_short_head(
    env: _Env,
    api: _FakeGithubApi,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rows = [_repo(f"octo-org/repo-{n:03d}") for n in range(200)]
    _token_arm(env, api, rows)
    _config(env.root, excluded=["octo-org/repo-007", "octo-org/repo-100", "octo-org/repo-199"])
    monkeypatch.setattr(github_mod, "default_minter", api.minter)
    code, text = _cli_text(capsys, "credential", "github", "repos", "--refresh")
    assert code == 0
    assert "GitHub repositories lent for: 197" in text, text
    assert sum(line.startswith("  octo-org/repo-") for line in text.splitlines()) == 197
    assert "more — see them all" not in text, text
    assert "opted out (3): octo-org/repo-007, octo-org/repo-100, octo-org/repo-199" in text
    print("=== repos (200 pushable, 3 opted out) ===")
    print(text)
    designated, _source = net_cli._github_designation_now(refresh=False)  # noqa: SLF001
    receipt = net_cli._designation_lines(  # noqa: SLF001
        designated, github_mod.SOURCE_TOKEN, receipt=True
    )
    assert sum(line.startswith("  octo-org/") for line in receipt) == 10
    assert "  … and 187 more — see them all: lop network credential github repos" in receipt


def test_u3_one_age_wording_and_no_config_keys_or_capitals_in_human_text(
    env: _Env, api: _FakeGithubApi
) -> None:
    assert [
        github_mod.listed_ago(value) for value in (5, 300, 5 * 3600, 30 * 3600, 3 * 86400, None)
    ] == ["just now", "5 min ago", "5 h ago", "30 h ago", "3 days ago", "at an unknown time"]
    last_good = github_mod.Designation(
        repositories=("me/a",),
        origin=github_mod.ORIGIN_LAST_GOOD,
        fetched_at=time.time() - 30 * 3600,
        error="server: forced failure",
        candidates=1,
        listed=("me/a",),
    )
    explicit_empty = github_mod.Designation(repositories=(), origin=github_mod.ORIGIN_EXPLICIT)
    all_out = github_mod.Designation(
        repositories=(),
        origin=github_mod.ORIGIN_ENUMERATED,
        excluded=("me/a",),
        excluded_count=1,
        candidates=1,
    )
    lines = net_cli._designation_lines(last_good, "token", receipt=False)  # noqa: SLF001
    assert lines[0].startswith("GitHub repositories lent for: 1 — the list fetched 30 h ago"), lines
    assert "GitHub couldn't be reached just now (server: forced failure)" in lines[0]
    texts = [
        *lines,
        *net_cli._designation_lines(explicit_empty, "token", receipt=False),  # noqa: SLF001
        *net_cli._designation_lines(all_out, "token", receipt=True),  # noqa: SLF001
        explicit_empty.refusal(),
        all_out.refusal(),
    ]
    for text in texts:
        assert "network.credentials" not in text, text
        assert "LAST GOOD" not in text, text
        for jargon in ("designat", "minted", "grant"):
            assert jargon not in text, (jargon, text)


def test_u4_refusal_copy_points_at_the_real_fix_and_a_recorded_share_is_not_failed() -> None:
    all_out = github_mod.Designation(
        repositories=(),
        origin=github_mod.ORIGIN_ENUMERATED,
        excluded=("me/a", "me/b"),
        excluded_count=2,
        candidates=2,
    )
    assert all_out.refusal() == (
        "every repository its GitHub login can push to (2) is opted out — bring one back "
        "with lop network credential github include <owner/repo>"
    )
    receipt = net_cli._designation_lines(all_out, "token", receipt=True)  # noqa: SLF001
    assert receipt[0].startswith(
        "GitHub repositories: none yet — the share is in place, but every borrow is refused"
    ), receipt
    assert "nothing was lent" not in " ".join(receipt)


@pytest.mark.parametrize(
    ("code", "detail", "expected"),
    [
        (
            github_mod.CODE_REPOSITORIES_UNSET,
            "every repository its GitHub login can push to (2) is opted out — bring one back "
            "with lop network credential github include <owner/repo>",
            "owner-laptop has no GitHub repository it can lend access for: every repository "
            "its GitHub login can push to (2) is opted out — bring one back with lop network "
            "credential github include <owner/repo>. Fix it on owner-laptop; nothing was lent.",
        ),
        (
            github_mod.CODE_REPOSITORIES_OVERFLOW,
            "2001 repositories; one share can carry at most 2000",
            "owner-laptop has more GitHub repositories than one share can carry (2001 "
            "repositories; one share can carry at most 2000), so nothing was lent. On "
            "owner-laptop, pick the ones to lend with the GitHub repositories setting (search "
            "it in /settings), or opt some out with lop network credential github exclude "
            "<owner/repo>.",
        ),
    ],
)
def test_f1_the_borrower_sentence_is_rendered_once(code: str, detail: str, expected: str) -> None:
    """QA F1: the relay leg and the runtime client both render; the second pass
    must not nest the sentence inside itself."""
    from local_operator.network.credentials.messages import render_broker_error

    error = cred_types.BrokerError(code=code, message=detail)
    once = render_broker_error(error, key="github", provider="github", owner_name="owner-laptop")
    assert once == expected
    error.message = once
    twice = render_broker_error(error, key="github", provider="github", owner_name="owner-laptop")
    assert twice == once


def test_n2_n3_usage_shows_json_and_the_example_is_not_a_real_handle(
    env: _Env, capsys: pytest.CaptureFixture[str]
) -> None:
    import argparse

    net_cli._guard_credential_subcommand(argparse.Namespace())  # noqa: SLF001
    usage = capsys.readouterr().err
    assert (
        "lop network credential github repos [--refresh] [--json] | exclude|include "
        "<owner/repo> [--json]"
    ) in usage, usage
    code, text = _cli_text(capsys, "credential", "github", "exclude", "nope", "--json")
    assert code == 1
    message = json.loads(text)["message"]
    assert "octo-org/website" in message and "damianvtran" not in message, message


def test_n5_a_slow_listing_prints_one_progress_line_and_a_fast_one_none(
    env: _Env,
    api: _FakeGithubApi,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _token_arm(env, api, [_repo("me/a")])
    monkeypatch.setattr(github_mod, "default_minter", api.minter)
    capsys.readouterr()
    net_cli._github_designation_now(refresh=True, progress=True)  # noqa: SLF001
    assert "listing the repositories" not in capsys.readouterr().err
    real = github_mod.designation

    def slow(*args: Any, **kwargs: Any) -> Any:
        time.sleep(0.3)
        return real(*args, **kwargs)

    monkeypatch.setattr(github_mod, "designation", slow)
    monkeypatch.setattr(net_cli, "_LISTING_PROGRESS_AFTER_S", 0.05)
    net_cli._github_designation_now(refresh=True, progress=True)  # noqa: SLF001
    err = capsys.readouterr().err
    assert err.count("listing the repositories your GitHub login can push to…") == 1, err


def test_m1_a_failed_listing_token_revoke_is_retried_once_then_logged(
    env: _Env, api: _FakeGithubApi, app_keypair: tuple[str, str], caplog: pytest.LogCaptureFixture
) -> None:
    app = github_mod.GithubAppKey(app_id=APP_ID, installation_id="4242", private_key=app_keypair[0])
    api.forge.installation_repos = [_repo("acme/one")]
    api.forge.fail_revokes = 1
    assert api.minter().installation_repositories(app) == ["acme/one"]
    assert len(api.forge.revoked) == 1  # the retry succeeded
    api.forge.fail_revokes = 2
    with caplog.at_level("WARNING", logger="local_operator.network.credentials.github"):
        assert api.minter().installation_repositories(app) == ["acme/one"]
    assert len(api.forge.revoked) == 1
    assert "could not be revoked after one retry" in caplog.text, caplog.text


def test_m3_a_listing_that_fails_partway_still_revokes_its_token(
    env: _Env, api: _FakeGithubApi, app_keypair: tuple[str, str]
) -> None:
    _seed_app_key(env.root, app_keypair[0])
    api.forge.installation_repos = [_repo(f"acme/r{n:03d}") for n in range(150)]
    api.forge.fail_page = 2
    with _broker(env, api) as broker:
        reply = _refused(broker)
    assert reply["code"] == github_mod.CODE_REPOSITORIES_UNSET, reply
    assert sum(path.endswith("&page=1") for path in api.forge.gets) == 1
    assert sum(path.endswith("&page=2") for path in api.forge.gets) == 1
    assert [mint.get("permissions") for mint in api.forge.mints] == [{"metadata": "read"}]
    assert len(api.forge.revoked) == 1 and not api.forge.tokens & set(api.forge.revoked)


def test_m4_f2_readiness_never_shows_the_previous_accounts_list(
    env: _Env, api: _FakeGithubApi
) -> None:
    _token_arm(env, api, [_repo("acct-one/x1"), _repo("acct-one/x2")])
    with _broker(env, api) as broker:
        _granted(broker)
    fact = readiness.github_fact(env.root, env.home)
    assert fact["repositories"] == ["acct-one/x1", "acct-one/x2"], fact
    # Re-login as another account; GitHub then fails (QA's repro).
    time.sleep(0.01)
    _seed_secret(env.root, github_mod.TOKEN_SECRET_NAME, OTHER_TOKEN)
    api.forge.fail_status = 500
    fact = readiness.github_fact(env.root, env.home)
    assert fact["repositories"] == [] and fact["repositories_origin"] == "unavailable", fact
    assert "earlier GitHub login" in fact.get("repositories_error", ""), fact
    clause = readiness._github_scope_clause(fact, "owner-laptop")  # noqa: SLF001
    assert "acct-one" not in clause, clause
    print("=== readiness after an account switch ===")
    print(clause)


def test_review_n1_exactly_the_page_bound_is_listed_and_one_more_is_refused(
    api: _FakeGithubApi,
) -> None:
    bound = github_mod.ENUMERATION_PAGE_SIZE * github_mod.ENUMERATION_MAX_PAGES
    api.forge.user_repos[TOKEN] = [_repo(f"me/r{n:05d}") for n in range(bound)]
    assert len(api.minter().pushable_user_repositories(TOKEN)) == bound
    api.forge.user_repos[TOKEN].append(_repo("me/one-more"))
    with pytest.raises(github_mod.GithubApiError) as refused:
        api.minter().pushable_user_repositories(TOKEN)
    assert f"more than {bound} repositories" in str(refused.value)
    assert "network.credentials" not in str(refused.value)
