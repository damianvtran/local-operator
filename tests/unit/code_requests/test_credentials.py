"""The credential resolver: ladders, remedies, caching — and never a leak.

Every case here asserts on SHAPE and refusal TEXT, never on a value reaching a
string a caller could log. The live cross-check (a real ``gh``/``glab`` login
resolving, then the token staying unreadable in reprs and scrubbed in output)
was run against the operator's own machine and is recorded in the PR body; the
store-secret arm reuses the mesh adapters' tested primitives and is covered
there.
"""

from __future__ import annotations

import pytest

from local_operator.code_requests import credentials
from local_operator.network.credentials import github as gh_credentials


@pytest.fixture(autouse=True)
def _clean_cache():
    credentials._reset_for_tests()
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
