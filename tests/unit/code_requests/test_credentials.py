"""The credential resolver: ladders, remedies, caching — and never a leak.

Every case here asserts on SHAPE and refusal TEXT, never on a value reaching a
string a caller could log. The live cross-check (a real ``gh``/``glab`` login
resolving, then the token staying unreadable in reprs and scrubbed in output)
was run against the operator's own machine and is recorded in the PR body; the
store-secret arm reuses the mesh adapters' tested primitives and is covered
there.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.code_requests import credentials
from local_operator.network.credentials import github as gh_credentials


@pytest.fixture(autouse=True)
def _clean_cache(monkeypatch):
    credentials._reset_for_tests()
    # ISOLATION FROM THE HOST: the github.com ladder now asks the mesh broker and
    # the env BEFORE gh's own login, so an unpatched test would read this
    # device's real placement (and a session's injected GH_TOKEN). Tests that
    # exercise either arm opt back in explicitly.
    monkeypatch.setattr(gh_credentials, "_client_for", lambda root: None)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    yield
    credentials._reset_for_tests()


class _Proc:
    def __init__(self, stdout: str, returncode: int = 0) -> None:
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = ""


def test_github_com_uses_the_mesh_probe_and_never_prints(monkeypatch) -> None:
    monkeypatch.setattr(gh_credentials, "read_gh_token", lambda home: "gho_test_value_123456")
    token = credentials.resolve("github.com", "github")
    assert token.source == "gh" and token.host == "github.com"
    assert token.value not in repr(token)
    assert "gho_test_value_123456" != str(token)


def test_github_absent_maps_to_the_sign_in_remedy(monkeypatch) -> None:
    def refuse(home):
        raise gh_credentials.GithubGhError("absent", "no gh login for this host")

    monkeypatch.setattr(gh_credentials, "read_gh_token", refuse)
    with pytest.raises(credentials.CredentialError) as caught:
        credentials.resolve("github.com", "github")
    assert caught.value.kind == "absent"
    assert "gh auth login" in caught.value.message


def test_github_enterprise_without_the_cli_is_absent(monkeypatch) -> None:
    monkeypatch.setattr(gh_credentials, "find_gh", lambda home: None)
    with pytest.raises(credentials.CredentialError) as caught:
        credentials.resolve("ghe.example.com", "github")
    assert caught.value.kind == "absent"


def test_github_enterprise_queries_the_cli_with_the_host(monkeypatch) -> None:
    seen: list[list[str]] = []

    def fake_run(argv, **kwargs):
        seen.append(list(argv))
        return _Proc("gho_ghes_token_value\n")

    monkeypatch.setattr(gh_credentials, "find_gh", lambda home: "/usr/bin/gh")
    monkeypatch.setattr(credentials.subprocess, "run", fake_run)
    token = credentials.resolve("ghe.example.com", "github")
    assert token.source == "gh"
    assert seen and seen[0][1:] == ["auth", "token", "--hostname", "ghe.example.com"]


def test_gitlab_uses_glab_config_get_token_for_the_host(monkeypatch) -> None:
    seen: list[list[str]] = []

    def fake_run(argv, **kwargs):
        seen.append(list(argv))
        return _Proc("glpat_test_value_123\n")

    monkeypatch.setattr(credentials, "_find_glab", lambda home: "/usr/bin/glab")
    monkeypatch.setattr(credentials.subprocess, "run", fake_run)
    token = credentials.resolve("gitlab.com", "gitlab")
    assert token.source == "glab"
    assert seen[0][1:] == ["config", "get", "token", "--host", "gitlab.com"]


def test_gitlab_unknown_host_is_absent_not_a_wrong_token(monkeypatch) -> None:
    # glab exits 0 with EMPTY output for a host it does not know: that is the
    # absent signal, and it must never fall through to another host's token.
    monkeypatch.setattr(credentials, "_find_glab", lambda home: "/usr/bin/glab")
    monkeypatch.setattr(credentials.subprocess, "run", lambda argv, **kw: _Proc(""))
    with pytest.raises(credentials.CredentialError) as caught:
        credentials.resolve("gitlab.example.com", "gitlab")
    assert caught.value.kind == "absent"
    assert "glab auth login" in caught.value.message


def test_gitlab_refusal_is_unusable_and_quotes_no_output(monkeypatch) -> None:
    monkeypatch.setattr(credentials, "_find_glab", lambda home: "/usr/bin/glab")
    monkeypatch.setattr(
        credentials.subprocess, "run", lambda argv, **kw: _Proc("secret-ish stderr", returncode=1)
    )
    with pytest.raises(credentials.CredentialError) as caught:
        credentials.resolve("gitlab.com", "gitlab")
    assert caught.value.kind == "unusable"
    assert "secret-ish" not in caught.value.message


def test_cache_serves_second_call_and_fresh_skips_it(monkeypatch) -> None:
    calls = {"n": 0}

    def fake_run(argv, **kwargs):
        calls["n"] += 1
        return _Proc("glpat_cached_value\n")

    monkeypatch.setattr(credentials, "_find_glab", lambda home: "/usr/bin/glab")
    monkeypatch.setattr(credentials.subprocess, "run", fake_run)
    credentials.resolve("gitlab.com", "gitlab")
    credentials.resolve("gitlab.com", "gitlab")
    assert calls["n"] == 1
    credentials.resolve("gitlab.com", "gitlab", fresh=True)
    assert calls["n"] == 2


def test_unknown_forge_is_detect_and_link(monkeypatch) -> None:
    with pytest.raises(credentials.CredentialError) as caught:
        credentials.resolve("bitbucket.org", "bitbucket")
    assert "detect-and-link" in caught.value.message


def test_resolved_values_are_registered_for_scrubbing(monkeypatch) -> None:
    monkeypatch.setattr(gh_credentials, "read_gh_token", lambda home: "gho_scrubme_value_9")
    credentials.resolve("github.com", "github")
    from local_operator.mcp import redaction

    scrubbed = redaction.scrub("before gho_scrubme_value_9 after")
    assert "gho_scrubme_value_9" not in scrubbed
    assert "before" in scrubbed and "after" in scrubbed


# ---------------------------------------------------------------------------
# F1 — a token only travels to an AUTHENTICATED host
# ---------------------------------------------------------------------------


def test_cli_env_strips_every_token_valued_variable(monkeypatch) -> None:
    """The strip is the membership check: gh/glab echo an env token for ANY host.

    The rule is EVERY name ending in TOKEN, not a family list — the family list
    missed ``OAUTH_TOKEN``, one of glab's documented env precedence names, and
    leaked a token to any ``--host`` a URL named (review round 2, N1).
    """
    sample = (
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "GH_ENTERPRISE_TOKEN",
        "GITLAB_TOKEN",
        "GITLAB_ACCESS_TOKEN",
        "GITLAB_ANYTHING_TOKEN",
        "GLAB_TOKEN",
        "OAUTH_TOKEN",
    )
    for name in sample:
        monkeypatch.setenv(name, "tok-value")
    env = credentials._cli_env()
    for name in sample:
        assert name not in env, name
    # Non-token variables survive: PATH is what makes the child runnable.
    assert env.get("PATH")
    # ...and the rule is name-ending, so a token-ish name that does not end in
    # TOKEN still survives (the member is not over-stripped).
    monkeypatch.setenv("MY_TOKEN_NOTE", "keep")
    assert credentials._cli_env().get("MY_TOKEN_NOTE") == "keep"


def test_an_oauth_token_echoed_by_a_fake_glab_is_stripped_out(tmp_path: Path, monkeypatch) -> None:
    """The N1 vector, end to end: glab's OAUTH_TOKEN arm cannot serve an unknown host."""
    fake = tmp_path / "fake-glab"
    fake.write_text("#!/bin/sh\nprintf '%s' \"${OAUTH_TOKEN:-}\"\n")
    fake.chmod(0o755)
    monkeypatch.setenv("OAUTH_TOKEN", "oauth-token-must-not-leak")
    monkeypatch.setattr(credentials, "_find_glab", lambda home: str(fake))
    monkeypatch.setattr(credentials, "_store_secret", lambda config_dir: "")
    with pytest.raises(credentials.CredentialError) as caught:
        credentials.resolve("evil.example.invalid", "gitlab")
    assert caught.value.kind == "absent"


def test_store_secret_only_serves_gitlab_com(monkeypatch) -> None:
    monkeypatch.setattr(credentials, "_find_glab", lambda home: None)
    monkeypatch.setattr(credentials, "_store_secret", lambda config_dir: "store-secret-value")
    monkeypatch.delenv("GITLAB_TOKEN", raising=False)
    token = credentials.resolve("gitlab.com", "gitlab")
    assert token.source == "secret-store" and token.value == "store-secret-value"
    credentials._reset_for_tests()
    with pytest.raises(credentials.CredentialError) as caught:
        credentials.resolve("evil.example.invalid", "gitlab")
    assert caught.value.kind == "absent"
    assert "glab auth login" in caught.value.message


def test_env_gitlab_token_only_speaks_for_gitlab_com(monkeypatch) -> None:
    monkeypatch.setattr(credentials, "_find_glab", lambda home: None)
    monkeypatch.setattr(credentials, "_store_secret", lambda config_dir: "")
    monkeypatch.setenv("GITLAB_TOKEN", "env-value")
    token = credentials.resolve("gitlab.com", "gitlab")
    assert token.source == "env" and token.value == "env-value"
    credentials._reset_for_tests()
    with pytest.raises(credentials.CredentialError):
        credentials.resolve("evil.example.invalid", "gitlab")


def test_an_env_token_echoed_by_a_fake_cli_is_stripped_out(tmp_path: Path, monkeypatch) -> None:
    """End-to-end: a CLI that prints the ENV token for any host resolves nothing.

    This is the reviewer's measured vector (``GITLAB_TOKEN=… glab config get
    token --host evil`` prints the token). The fake below behaves exactly like
    the real CLI, and the child env has the variable stripped, so the answer is
    empty and the host stays unauthenticated.
    """
    fake = tmp_path / "fake-glab"
    fake.write_text("#!/bin/sh\nprintf '%s' \"${GITLAB_TOKEN:-}\"\n")
    fake.chmod(0o755)
    monkeypatch.setenv("GITLAB_TOKEN", "env-token-must-not-leak")
    monkeypatch.setattr(credentials, "_find_glab", lambda home: str(fake))
    monkeypatch.setattr(credentials, "_store_secret", lambda config_dir: "")
    with pytest.raises(credentials.CredentialError) as caught:
        credentials.resolve("evil.example.invalid", "gitlab")
    assert caught.value.kind == "absent"


def test_an_env_token_echoed_by_a_fake_gh_is_stripped_out_for_ghes(
    tmp_path: Path, monkeypatch
) -> None:
    fake = tmp_path / "fake-gh"
    fake.write_text("#!/bin/sh\nprintf '%s' \"${GH_ENTERPRISE_TOKEN:-}\"\n")
    fake.chmod(0o755)
    monkeypatch.setenv("GH_ENTERPRISE_TOKEN", "ghes-token-must-not-leak")
    monkeypatch.setattr(gh_credentials, "find_gh", lambda home: str(fake))
    with pytest.raises(credentials.CredentialError) as caught:
        credentials.resolve("ghe.example.com", "github")
    assert caught.value.kind == "absent"


def test_env_gh_token_counts_for_github_com_but_never_for_ghes(monkeypatch) -> None:
    """QA round 1, Q5: a headless device authenticated by environment IS signed in."""

    def refuse(home):
        raise gh_credentials.GithubGhError("absent", "no hosts.yml login")

    monkeypatch.setattr(gh_credentials, "read_gh_token", refuse)
    monkeypatch.setenv("GH_TOKEN", "env-gh-token")
    token = credentials.resolve("github.com", "github")
    assert token.source == "env" and token.value == "env-gh-token"
    credentials._reset_for_tests()
    monkeypatch.setattr(gh_credentials, "find_gh", lambda home: None)
    with pytest.raises(credentials.CredentialError):
        credentials.resolve("ghe.example.com", "github")


# ---------------------------------------------------------------------------
# X2 — the gh lookup honours GH_CONFIG_DIR and the keyring path
# ---------------------------------------------------------------------------


def _script(tmp_path: Path, name: str, body: str) -> str:
    path = tmp_path / name
    path.write_text(body)
    path.chmod(0o755)
    return str(path)


def test_gh_is_asked_when_the_default_hosts_file_is_absent(tmp_path: Path, monkeypatch) -> None:
    """X2: a login stored under GH_CONFIG_DIR (or the keyring) still resolves —
    by asking gh itself, the CLI's own resolution."""

    def refuse(home):
        raise gh_credentials.GithubGhError("absent", "no gh CLI login is stored here")

    monkeypatch.setattr(gh_credentials, "read_gh_token", refuse)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    cfg = tmp_path / "gh-config"
    cfg.mkdir()
    (cfg / "token").write_text("relocated-token-value\n")
    exe = _script(tmp_path, "fake-gh", '#!/bin/sh\ncat "$GH_CONFIG_DIR/token" 2>/dev/null\n')
    monkeypatch.setattr(gh_credentials, "find_gh", lambda home: exe)
    monkeypatch.setenv("GH_CONFIG_DIR", str(cfg))
    token = credentials.resolve("github.com", "github")
    assert token.source == "gh" and token.value == "relocated-token-value"


def test_the_probe_cannot_echo_an_env_token_back(tmp_path: Path, monkeypatch) -> None:
    """The X2 probe keeps the F1 rule: the child env is stripped, so a gh that
    would echo GH_TOKEN for any request stays silent here."""

    def refuse(home):
        raise gh_credentials.GithubGhError("absent", "no gh CLI login is stored here")

    monkeypatch.setattr(gh_credentials, "read_gh_token", refuse)
    monkeypatch.setenv("GH_TOKEN", "env-token")
    exe = _script(tmp_path, "fake-gh", '#!/bin/sh\nprintf "%s" "${GH_TOKEN:-}"\n')
    monkeypatch.setattr(gh_credentials, "find_gh", lambda home: exe)
    # The explicit env arm answers first (github.com only, by design)...
    assert credentials.resolve("github.com", "github").value == "env-token"
    # ...and with the env arm gone the stripped probe answers nothing.
    credentials._reset_for_tests()
    monkeypatch.delenv("GH_TOKEN", raising=False)
    with pytest.raises(credentials.CredentialError) as caught:
        credentials.resolve("github.com", "github")
    assert caught.value.kind == "absent"


def test_no_login_anywhere_is_still_absent(tmp_path: Path, monkeypatch) -> None:
    def refuse(home):
        raise gh_credentials.GithubGhError("absent", "no gh CLI login is stored here")

    monkeypatch.setattr(gh_credentials, "read_gh_token", refuse)
    monkeypatch.setattr(gh_credentials, "find_gh", lambda home: None)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with pytest.raises(credentials.CredentialError) as caught:
        credentials.resolve("github.com", "github")
    assert caught.value.kind == "absent"
    assert "gh auth login" in caught.value.message


# ---------------------------------------------------------------------------
# Mesh-brokered github.com login (lop-remote-node-2: bash's `gh pr view` worked
# on the borrowed GH_TOKEN while code_requests refused "could not be read")
# ---------------------------------------------------------------------------


class _Grant:
    def __init__(self, token: str) -> None:
        self.access_token = token
        self.grant_expires_at_ms = 0


class _FakeBorrowClient:
    """The ``MeshCredentialClient`` surface ``borrowed_git_env`` and the
    resolver touch: a holder of a github credential owned by ``owner``."""

    self_device = "node-device"
    network_id = "net-synthetic"

    def __init__(self, answer, *, holder: bool = True, owner: str = "studio-mac") -> None:
        self.answer = answer
        self.holder = holder
        self.owner = owner
        self.asked = 0
        self.session_ids: list[str | None] = []

    def should_borrow(self, key: str) -> bool:
        return self.holder

    def owner_label(self, key: str) -> str:
        return self.owner

    def request_grant_sync(self, key: str, **kwargs):
        self.asked += 1
        self.session_ids.append(kwargs.get("session_id"))
        return self.answer

    def cached_refusal(self, key: str):
        from local_operator.network.credentials.types import BrokerError

        return self.answer if isinstance(self.answer, BrokerError) else None


def _node_without_gh_login(tmp_path: Path, monkeypatch) -> Path:
    """The node as measured: hosts.yml is ``{}`` (gh logged out) and the gh on
    PATH exits 1 for ``auth token`` \u2014 ``read_gh_token`` calls that
    "unusable"."""
    home = tmp_path / "home"
    (home / ".config" / "gh").mkdir(parents=True)
    (home / ".config" / "gh" / "hosts.yml").write_text("{}\n")
    exe = _script(tmp_path, "fake-gh", "#!/bin/sh\necho 'no oauth token' >&2\nexit 1\n")
    monkeypatch.setattr(gh_credentials, "find_gh", lambda home: exe)
    # The self-revoke timer would DELETE the synthetic token at GitHub.
    monkeypatch.setattr(gh_credentials, "_schedule_self_revoke", lambda token, exp: None)
    return home


def test_brokered_login_resolves_when_the_node_has_no_gh_login(tmp_path: Path, monkeypatch) -> None:
    home = _node_without_gh_login(tmp_path, monkeypatch)
    client = _FakeBorrowClient(_Grant("ghs_synthetic_borrowed_1"))
    monkeypatch.setattr(gh_credentials, "_client_for", lambda root: client)
    token = credentials.resolve("github.com", "github", home=home)
    assert token.source == "brokered" and token.value == "ghs_synthetic_borrowed_1"
    assert token.lender == "studio-mac"
    assert "ghs_synthetic_borrowed_1" not in repr(token)


def test_a_borrowed_login_never_vouches_for_ghes(tmp_path: Path, monkeypatch) -> None:
    _node_without_gh_login(tmp_path, monkeypatch)
    client = _FakeBorrowClient(_Grant("ghs_synthetic_borrowed_2"))
    monkeypatch.setattr(gh_credentials, "_client_for", lambda root: client)
    with pytest.raises(credentials.CredentialError):
        credentials.resolve("ghe.example.com", "github", home=tmp_path / "home")
    assert client.asked == 0


def test_env_gh_token_beats_an_unreadable_gh_login(tmp_path: Path, monkeypatch) -> None:
    """gh's own precedence: GH_TOKEN before hosts.yml \u2014 a failing gh must not
    short-circuit the env arm."""
    home = _node_without_gh_login(tmp_path, monkeypatch)
    monkeypatch.setenv("GH_TOKEN", "env-synthetic-token")
    token = credentials.resolve("github.com", "github", home=home)
    assert token.source == "env" and token.value == "env-synthetic-token"


def test_a_logged_out_hosts_file_is_absent_not_unreadable(tmp_path: Path, monkeypatch) -> None:
    home = _node_without_gh_login(tmp_path, monkeypatch)
    with pytest.raises(credentials.CredentialError) as caught:
        credentials.resolve("github.com", "github", home=home)
    assert caught.value.kind == "absent"
    assert "could not be read" not in caught.value.message
    assert "`gh auth login`" in caught.value.message


def test_a_stored_but_unreadable_gh_login_still_says_so(tmp_path: Path, monkeypatch) -> None:
    """The "could not be read" sentence stays for a login that IS stored."""
    home = _node_without_gh_login(tmp_path, monkeypatch)
    (home / ".config" / "gh" / "hosts.yml").write_text("github.com:\n  user: someone\n")
    with pytest.raises(credentials.CredentialError) as caught:
        credentials.resolve("github.com", "github", home=home)
    assert caught.value.kind == "unusable"
    assert "could not be read" in caught.value.message


def test_a_lapsed_borrow_names_the_lender_and_expiry(tmp_path: Path, monkeypatch) -> None:
    from local_operator.network.credentials.types import BrokerError

    home = _node_without_gh_login(tmp_path, monkeypatch)
    refusal = BrokerError(code="no_local_credential", key="github", owner_device_name="studio-mac")
    monkeypatch.setattr(gh_credentials, "_client_for", lambda root: _FakeBorrowClient(refusal))
    with pytest.raises(credentials.CredentialError) as caught:
        credentials.resolve("github.com", "github", home=home)
    message = caught.value.message
    assert "borrowed from studio-mac expired" in message
    assert "sign in again on studio-mac" in message
    assert "could not be read on this device" not in message


def test_an_unreachable_lender_says_reconnect(tmp_path: Path, monkeypatch) -> None:
    from local_operator.network.credentials.types import BrokerError

    home = _node_without_gh_login(tmp_path, monkeypatch)
    refusal = BrokerError(code="owner_offline", key="github", owner_device_name="studio-mac")
    monkeypatch.setattr(gh_credentials, "_client_for", lambda root: _FakeBorrowClient(refusal))
    with pytest.raises(credentials.CredentialError) as caught:
        credentials.resolve("github.com", "github", home=home)
    assert "could not be reached; reconnect studio-mac" in caught.value.message


def test_a_refused_borrow_still_falls_back_to_an_env_token(tmp_path: Path, monkeypatch) -> None:
    from local_operator.network.credentials.types import BrokerError

    home = _node_without_gh_login(tmp_path, monkeypatch)
    refusal = BrokerError(code="owner_offline", key="github")
    monkeypatch.setattr(gh_credentials, "_client_for", lambda root: _FakeBorrowClient(refusal))
    monkeypatch.setenv("GITHUB_TOKEN", "env-synthetic-fallback")
    assert credentials.resolve("github.com", "github", home=home).source == "env"


def test_a_revoked_lend_is_not_served_from_memory(tmp_path: Path, monkeypatch) -> None:
    """Review round 1, R1-1: the owner revokes the share (or a panic fences the
    mesh) AFTER a borrow; the next resolve must re-ask, exactly as bash does per
    command, not re-serve the lent token from the process cache."""
    home = _node_without_gh_login(tmp_path, monkeypatch)
    client = _FakeBorrowClient(_Grant("ghs_synthetic_revoked_1"))
    monkeypatch.setattr(gh_credentials, "_client_for", lambda root: client)
    assert credentials.resolve("github.com", "github", home=home).source == "brokered"
    client.holder = False  # should_borrow -> False: the share was revoked
    with pytest.raises(credentials.CredentialError) as caught:
        credentials.resolve("github.com", "github", home=home)
    assert caught.value.kind == "absent"
    assert client.asked == 1, "a revoked lend must not be asked again"


def test_a_lent_token_is_still_registered_for_scrubbing(tmp_path: Path, monkeypatch) -> None:
    home = _node_without_gh_login(tmp_path, monkeypatch)
    client = _FakeBorrowClient(_Grant("ghs_synthetic_scrub_1"))
    monkeypatch.setattr(gh_credentials, "_client_for", lambda root: client)
    registered: list[str] = []
    monkeypatch.setattr(credentials, "_register_for_scrubbing", registered.append)
    credentials.resolve("github.com", "github", home=home)
    assert registered == ["ghs_synthetic_scrub_1"]


@pytest.mark.parametrize(
    "code", ["internal", "unsupported", "malformed_frame", "quota_blocked", None]
)
def test_an_unclassified_borrow_failure_does_not_blame_the_owner_login(
    tmp_path: Path, monkeypatch, code
) -> None:
    """Review round 1, R1-3: a broker/version fault, or a borrow with no
    recorded refusal, proves nothing about the owner's login."""
    from local_operator.network.credentials.types import BrokerError

    home = _node_without_gh_login(tmp_path, monkeypatch)
    answer = BrokerError(code=code, key="github") if code else None
    monkeypatch.setattr(gh_credentials, "_client_for", lambda root: _FakeBorrowClient(answer))
    with pytest.raises(credentials.CredentialError) as caught:
        credentials.resolve("github.com", "github", home=home)
    message = caught.value.message
    assert "borrowed from studio-mac is not available right now; try again shortly" in message
    assert "expired" not in message and "sign in again" not in message
    assert caught.value.lender == "studio-mac"


def test_borrows_are_attributed_to_code_requests(tmp_path: Path, monkeypatch) -> None:
    """Review round 1, R1-4: the owner's audit row names the surface."""
    home = _node_without_gh_login(tmp_path, monkeypatch)
    client = _FakeBorrowClient(_Grant("ghs_synthetic_attr_1"))
    monkeypatch.setattr(gh_credentials, "_client_for", lambda root: client)
    credentials.resolve("github.com", "github", home=home)
    assert client.session_ids == ["code_requests"]
