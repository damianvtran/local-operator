"""Repo and HEAD detection against REAL git repositories.

WHY REAL GIT. The three questions this module answers — what is ``origin``, what
is HEAD, and is HEAD on an ``origin`` branch as this clone last saw it — are
answered by git's own output shapes, and a stubbed ``subprocess`` would encode my
guess about those shapes rather than git's. So every case here builds a throwaway
repository (and a bare "remote") under ``tmp_path`` and runs the module against
it.

THE REFUSALS ARE THE POINT. An SSH origin and an unpushed commit both produced
runs that failed minutes later at the far end, so they are refused here with a
reason a surface has copy for.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from local_operator.remote_compute import repo as repo_module
from local_operator.remote_compute.errors import UnusableRepo
from local_operator.remote_compute.repo import RepoRef, detect_repo

pytestmark = pytest.mark.skipif(
    subprocess.run(["git", "--version"], capture_output=True, check=False).returncode != 0,
    reason="git is not installed",
)

#: An https origin: the only form a keyless container can clone.
HTTPS_ORIGIN = "https://example.invalid/team/repo.git"
SSH_ORIGIN = "git@example.invalid:team/repo.git"


def _git(*arguments: str, cwd: Path) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=cwd, capture_output=True, text=True, check=True
    )
    return completed.stdout.strip()


@pytest.fixture()
def checkout(tmp_path: Path) -> Path:
    """A repository whose HEAD is on ``origin``'s main, as a fresh push leaves it.

    The tracking ref is created with ``update-ref`` rather than by cloning the
    bare remote, so ``origin`` can carry the https URL a real clone would have:
    a local clone would write a ``file://`` origin, which is exactly the form the
    module refuses.
    """
    work = tmp_path / "work"
    work.mkdir()
    _git("init", "--quiet", "--initial-branch=main", cwd=work)
    _git("config", "user.email", "test@example.invalid", cwd=work)
    _git("config", "user.name", "Test", cwd=work)
    (work / "file.txt").write_text("one\n", encoding="utf-8")
    _git("add", "file.txt", cwd=work)
    _git("commit", "--quiet", "-m", "one", cwd=work)
    _git("remote", "add", "origin", HTTPS_ORIGIN, cwd=work)
    _git("update-ref", "refs/remotes/origin/main", "HEAD", cwd=work)
    return work


def _head(checkout: Path) -> str:
    return _git("rev-parse", "HEAD", cwd=checkout)


def _commit(checkout: Path, name: str = "two") -> str:
    (checkout / f"{name}.txt").write_text(f"{name}\n", encoding="utf-8")
    _git("add", f"{name}.txt", cwd=checkout)
    _git("commit", "--quiet", "-m", name, cwd=checkout)
    return _head(checkout)


def test_a_pushed_head_answers_the_url_and_the_commit(checkout: Path) -> None:
    detected = detect_repo(checkout)
    assert isinstance(detected, RepoRef)
    assert detected.url == HTTPS_ORIGIN
    assert detected.sha == _head(checkout)
    assert detected.refs == ("origin/main",)


def test_the_refusal_names_the_reason_a_directory_is_unusable(checkout: Path) -> None:
    _commit(checkout)
    with pytest.raises(UnusableRepo) as caught:
        detect_repo(checkout)
    assert caught.value.reason == repo_module.REASON_UNPUSHED
    assert caught.value.detail.startswith(_head(checkout)[:12])


def test_a_subdirectory_is_the_repository_it_sits_in(checkout: Path) -> None:
    nested = checkout / "src" / "deeper"
    nested.mkdir(parents=True)
    assert detect_repo(nested).sha == _head(checkout)


def test_an_ssh_origin_is_refused_before_a_request_is_composed(checkout: Path) -> None:
    _git("remote", "set-url", "origin", SSH_ORIGIN, cwd=checkout)
    with pytest.raises(UnusableRepo) as caught:
        detect_repo(checkout)
    assert caught.value.reason == repo_module.REASON_NOT_HTTPS
    # The refusal names the remote it is about, without the ssh user segment.
    assert "example.invalid" in caught.value.detail
    assert "git@" not in caught.value.detail


def test_a_scp_style_origin_is_refused_the_same_way(checkout: Path) -> None:
    _git("remote", "set-url", "origin", "team@example.invalid:team/repo.git", cwd=checkout)
    with pytest.raises(UnusableRepo) as caught:
        detect_repo(checkout)
    assert caught.value.reason == repo_module.REASON_NOT_HTTPS


def test_an_https_origin_carrying_credentials_is_refused_and_not_echoed(
    checkout: Path,
) -> None:
    _git(
        "remote",
        "set-url",
        "origin",
        "https://user:zq-token-9@example.invalid/team/repo.git",
        cwd=checkout,
    )
    with pytest.raises(UnusableRepo) as caught:
        detect_repo(checkout)
    assert caught.value.reason == repo_module.REASON_NOT_HTTPS
    assert "zq-token-9" not in caught.value.detail
    assert "user" not in caught.value.detail
    assert caught.value.detail == "https://example.invalid/team/repo.git"


def test_no_origin_is_a_repository_refusal(checkout: Path) -> None:
    _git("remote", "remove", "origin", cwd=checkout)
    with pytest.raises(UnusableRepo) as caught:
        detect_repo(checkout)
    assert caught.value.reason == repo_module.REASON_NO_REMOTE


def test_another_remote_does_not_stand_in_for_origin(checkout: Path) -> None:
    _git("remote", "rename", "origin", "upstream", cwd=checkout)
    with pytest.raises(UnusableRepo) as caught:
        detect_repo(checkout)
    assert caught.value.reason == repo_module.REASON_NO_REMOTE


def test_a_directory_that_is_not_a_repository_is_refused(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(UnusableRepo) as caught:
        detect_repo(plain)
    assert caught.value.reason == repo_module.REASON_NO_REMOTE


def test_a_path_that_does_not_exist_is_refused(tmp_path: Path) -> None:
    with pytest.raises(UnusableRepo) as caught:
        detect_repo(tmp_path / "missing")
    assert caught.value.reason == repo_module.REASON_NO_REMOTE


def test_a_repository_with_no_commit_is_refused(checkout: Path) -> None:
    fresh = checkout.parent / "fresh"
    fresh.mkdir()
    _git("init", "--quiet", "--initial-branch=main", cwd=fresh)
    _git("remote", "add", "origin", HTTPS_ORIGIN, cwd=fresh)
    with pytest.raises(UnusableRepo) as caught:
        detect_repo(fresh)
    assert caught.value.reason == repo_module.REASON_NO_REMOTE


def test_a_failed_refresh_falls_back_to_the_local_view(checkout: Path) -> None:
    """``refresh=True`` asks the network; a refusal there is not a refusal here.

    The origin is an https URL that does not resolve in this test, so the fetch
    fails. The answer must still be the local one — an offline laptop can say
    what it knows, which is what the panel needs.
    """
    detected = detect_repo(checkout, refresh=True)
    assert detected.url == HTTPS_ORIGIN


def test_git_missing_is_a_refusal_rather_than_a_crash(
    checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing(*arguments: object, **keywords: object) -> None:
        raise FileNotFoundError(2, "No such file or directory", "git")

    monkeypatch.setattr(repo_module.subprocess, "run", missing)
    with pytest.raises(UnusableRepo) as caught:
        detect_repo(checkout)
    assert caught.value.reason == repo_module.REASON_NO_REMOTE


def test_git_is_never_asked_to_prompt_for_credentials(checkout: Path) -> None:
    """A fetch that wants a password must fail; a panel has no terminal to answer."""
    seen: list[dict[str, str]] = []
    real = subprocess.run

    def spy(*arguments: object, **keywords: object) -> subprocess.CompletedProcess[str]:
        environment = keywords.get("env")
        if isinstance(environment, dict):
            seen.append(environment)
        return real(*arguments, **keywords)  # type: ignore[arg-type]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(repo_module.subprocess, "run", spy)
        detect_repo(checkout, refresh=True)
    assert seen and all(env.get("GIT_TERMINAL_PROMPT") == "0" for env in seen)
