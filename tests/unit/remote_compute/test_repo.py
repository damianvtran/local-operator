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


# ---------------------------------------------------------------------------
# The environment git is given, and the scrubber under a malformed URL
# ---------------------------------------------------------------------------


def test_an_inherited_git_dir_does_not_answer_for_another_repository(
    checkout: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A shell's exported ``GIT_DIR`` must not redirect what detection reads.

    Reproduced before the fix: with ``GIT_DIR`` pointing at another repository,
    ``git -C <checkout>`` answered THAT repository's origin URL and its pushed
    state, so a run could be submitted — under the operator's identity — as code
    from a different checkout. Plausible in practice: an exported ``GIT_DIR`` in
    the shell that then started the panel.
    """
    other = tmp_path / "other"
    other.mkdir()
    _git("init", "--quiet", "--initial-branch=main", cwd=other)
    _git("remote", "add", "origin", "https://example.invalid/other/other.git", cwd=other)
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(other))
    detected = detect_repo(checkout)
    assert detected.url == HTTPS_ORIGIN
    assert detected.directory == checkout


def test_only_the_two_read_only_git_variables_reach_git(
    checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every inherited ``GIT_*`` is dropped, not just the repository pointers.

    The named-variable alternative leaves the next such variable (a config
    override, a credential helper, an index path) to be discovered the same way
    ``GIT_DIR`` was, so the rule is "none of them".
    """
    monkeypatch.setenv("GIT_DIR", "/nowhere/.git")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    seen: list[dict[str, str]] = []
    real = subprocess.run

    def spy(*arguments: object, **keywords: object) -> subprocess.CompletedProcess[str]:
        environment = keywords.get("env")
        if isinstance(environment, dict):
            seen.append(environment)
        return real(*arguments, **keywords)  # type: ignore[arg-type]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(repo_module.subprocess, "run", spy)
        detect_repo(checkout)
    assert seen
    for environment in seen:
        assert sorted(key for key in environment if key.startswith("GIT_")) == [
            "GIT_OPTIONAL_LOCKS",
            "GIT_TERMINAL_PROMPT",
        ]
        assert environment["GIT_OPTIONAL_LOCKS"] == "0"


def test_the_scrubber_survives_a_port_urllib_refuses() -> None:
    """An out-of-range port is a typed refusal's detail, never a ``ValueError``.

    ``urlsplit().port`` raises for it, and the refusal path reads the port — so a
    ``git remote set-url`` typo escaped ``detect_repo``, which promises
    ``RepoRef | UnusableRepo``. The fallback keeps the netloc (userinfo removed),
    which is what shows the operator the value they typed.
    """
    assert (
        repo_module.scrub_url("https://user:hunter2-dummy@example.invalid:99999999/r.git")
        == "https://example.invalid:99999999/r.git"
    )
    assert repo_module.scrub_url("https://example.invalid/r.git") == "https://example.invalid/r.git"
    assert (
        repo_module.scrub_url("https://user:hunter2-dummy@example.invalid/r.git")
        == "https://example.invalid/r.git"
    )


def test_an_origin_with_a_bad_port_and_credentials_is_refused_without_raising(
    checkout: Path,
) -> None:
    _git(
        "remote",
        "set-url",
        "origin",
        "ssh://user:hunter2-dummy@example.invalid:99999999/team/repo.git",
        cwd=checkout,
    )
    with pytest.raises(UnusableRepo) as caught:
        detect_repo(checkout)
    assert caught.value.reason == repo_module.REASON_NOT_HTTPS
    assert "hunter2-dummy" not in caught.value.detail
    assert caught.value.detail == "ssh://example.invalid:99999999/team/repo.git"


def test_the_scrubber_survives_a_url_urllib_cannot_parse() -> None:
    """``urlsplit`` itself raises for these, before ``.port`` is ever consulted.

    Two reachable shapes: an unbalanced bracket, one keystroke away from a real
    IPv6 origin, and a bracket ANYWHERE in the netloc — which is how a password
    such as a bracketed token presents, because urllib's bracketed-host check then
    runs on the host that follows. The second one is the credential shape this
    function exists for, and it escaped as ``ValueError`` into a path that calls
    the scrubber AFTER the far end has created the run.

    The answer is the placeholder, not a substring of the input: the parse
    failed because the userinfo boundary is unknowable, so nothing taken from
    the string can be proven credential-free.
    """
    # Fragment-built so the password text is not written whole: it exists
    # only to carry the bracket that makes urllib refuse the netloc.
    password = "ab" + "[cd]" + "ef"
    bracketed_password = "https://user:" + password + "@example.invalid/r.git"
    for refused in ("https://[::1/r.git", bracketed_password):
        scrubbed = repo_module.scrub_url(refused)
        assert scrubbed == repo_module.UNPARSEABLE_URL
        assert password not in scrubbed
        assert "@" not in scrubbed


def test_an_origin_urllib_cannot_parse_is_refused_without_raising(checkout: Path) -> None:
    """``_https_origin`` promises ``RepoRef | UnusableRepo`` as well.

    A credential-bearing origin is exactly what it refuses in order to sanitise
    it, so the parse failure has to answer the typed refusal the same way the
    out-of-range port above does.
    """
    _git("remote", "set-url", "origin", "https://[::1/r.git", cwd=checkout)
    with pytest.raises(UnusableRepo) as caught:
        detect_repo(checkout)
    assert caught.value.reason == repo_module.REASON_NOT_HTTPS
    assert caught.value.detail == repo_module.UNPARSEABLE_URL


def test_the_scrubber_keeps_the_brackets_that_make_ipv6_a_host() -> None:
    """Dropping the brackets turns a valid origin into one that no longer parses.

    The scrubbed value is what a refusal's detail and the record carry, so it
    has to stay a URL rather than merely lack userinfo.
    """
    assert (
        repo_module.scrub_url("https://user:pass@[::1]:8443/o/r.git")
        == "https://[::1]:8443/o/r.git"
    )


def _origin_with_password(password: str, *, host: str = "github.com", path: str = "o/r.git") -> str:
    """One https origin carrying ``password`` as its userinfo.

    ASSEMBLED FROM PIECES so no credential-shaped literal is written in this
    file: the password is never a real one, and it exists only to carry a URL
    delimiter — which is what moved the host out of the netloc.
    """
    userinfo = "user" + ":" + password
    return "https://" + userinfo + "@" + host + "/" + path


def test_the_scrubber_removes_a_password_that_contains_a_url_delimiter() -> None:
    """M1: a netloc-only rule returns this URL byte-identical, host and all.

    ``urlsplit`` ends the authority at the first ``/``, ``?`` or ``#``, so the
    userinfo lands in the path and the netloc looks clean. The parse succeeds,
    the rebuilt URL was the input, and ``record_submission`` wrote the whole
    credential to ``runs.jsonl``. The rule the scrubber applies now: an ``@`` at
    or after the authority boundary is userinfo unless the authority can be a
    ``host[:port]``, and everything up to the LAST ``@`` goes — last, because a
    password may itself contain one.
    """
    passwords = ("pa/ss", "pa?ss", "pa#ss", "pa/ss@word", "pa?ss#w/ord@x")
    for password in passwords:
        origin = _origin_with_password(password)
        scrubbed = repo_module.scrub_url(origin)
        assert scrubbed == "https://github.com/o/r.git", (origin, scrubbed)
        assert password not in scrubbed
        assert "@" not in scrubbed


def test_the_scrubber_keeps_an_at_sign_that_belongs_to_the_path() -> None:
    """The strictness has a floor: a host-shaped authority means the ``@`` is a path.

    ``https://github.com/o/r@v1.git`` is a legitimate origin, and a rule that
    refused every ``@`` after the authority would strip it to ``v1.git``. The
    discriminator is the authority's shape, not the presence of an ``@``.
    """
    for origin, expected in (
        ("https://github.com/o/r@v1.git", "https://github.com/o/r@v1.git"),
        ("https://github.com:8443/o/r@v1.git", "https://github.com:8443/o/r@v1.git"),
        (
            _origin_with_password("pass", path="o/r@v1.git"),
            "https://github.com/o/r@v1.git",
        ),
        (
            _origin_with_password("pass", host="example.invalid:99999999", path="o/r@v1.git"),
            "https://example.invalid:99999999/o/r@v1.git",
        ),
    ):
        assert repo_module.scrub_url(origin) == expected, origin


def test_an_origin_whose_password_contains_a_delimiter_is_refused(checkout: Path) -> None:
    """The detection serves the REFUSAL too: this URL is sent, not only recorded.

    ``_https_origin`` refuses an ``@`` in the netloc, and the escaped shape
    leaves the netloc clean — so accepting it would put a credential in the run
    body, which the far end and every proxy in between would see.
    """
    _git("remote", "set-url", "origin", _origin_with_password("pa/ss"), cwd=checkout)
    with pytest.raises(UnusableRepo) as caught:
        detect_repo(checkout)
    assert caught.value.reason == repo_module.REASON_NOT_HTTPS
    assert caught.value.detail == "https://github.com/o/r.git"
    assert "pa/ss" not in caught.value.detail
