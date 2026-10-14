"""Host identification and ref parsing: the rules that decide what a URL IS.

The cases are the ones that would produce a WRONG forge family if the hostname were
trusted on its own, plus the explicit negatives (an issue URL, a create-PR link, a
compare page) that must never become a row.
"""

from __future__ import annotations

import pytest

from local_operator.code_requests.refs import (
    EMPTY_CONTEXT,
    HostContext,
    Remote,
    iter_refs,
    load_host_context,
    parse_any,
    parse_qualified,
    parse_remote_url,
    parse_url,
)

GHE = HostContext(
    github_hosts=frozenset({"ghe.example.com"}),
    gitlab_hosts=frozenset({"gl.example.com"}),
    gitea_hosts=frozenset({"git.example.com"}),
    remotes=(Remote(name="origin", host="ghe.example.com", project="acme/api"),),
)


@pytest.mark.parametrize(
    "url,forge,project,number",
    [
        (
            "https://github.com/damianvtran/local-operator/pull/1904",
            "github",
            "damianvtran/local-operator",
            1904,
        ),
        (
            "https://github.com/damianvtran/local-operator/pull/1904/files#diff-1",
            "github",
            "damianvtran/local-operator",
            1904,
        ),
        (
            "https://github.com/damianvtran/local-operator/pull/1904#issuecomment-6081512558",
            "github",
            "damianvtran/local-operator",
            1904,
        ),
        (
            "https://gitlab.com/minervaai/minerva-skills/-/merge_requests/53",
            "gitlab",
            "minervaai/minerva-skills",
            53,
        ),
        (
            "https://gitlab.com/group/sub/project/-/merge_requests/7#note_1",
            "gitlab",
            "group/sub/project",
            7,
        ),
        ("https://codeberg.org/forgejo/forgejo/pulls/42", "gitea", "forgejo/forgejo", 42),
        ("https://bitbucket.org/team/repo/pull-requests/9/overview", "bitbucket", "team/repo", 9),
        (
            "https://dev.azure.com/org/proj/_git/repo/pullrequest/77",
            "azure",
            "org/proj/_git/repo",
            77,
        ),
        (
            "https://review.opendev.org/c/openstack/nova/+/123456",
            "gerrit",
            "openstack/nova",
            123456,
        ),
        ("https://git.example.com/acme/api/pulls/12", "gitea", "acme/api", 12),
        ("https://gl.example.com/acme/api/-/merge_requests/3", "gitlab", "acme/api", 3),
    ],
)
def test_a_recognised_link_becomes_a_ref(url, forge, project, number):
    ref = parse_url(url, GHE)
    assert ref is not None, url
    assert (ref.forge, ref.project, ref.number) == (forge, project, number)
    assert ref.key.endswith(f"{project}{'!' if forge == 'gitlab' else '#'}{number}")


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/o/r/issues/12",
        "https://github.com/o/r/pull/new/feat-x",
        "https://gitlab.com/o/r/-/merge_requests/new?merge_request%5Bsource_branch%5D=x",
        "https://github.com/o/r/compare/main...feat",
        "https://github.com/o/r/pulls",
        "https://github.com/o/r/pull/0",
    ],
)
def test_the_explicit_negatives_never_become_a_ref(url):
    assert parse_url(url, GHE) is None


def test_a_known_host_contradicting_the_shape_is_refused():
    """``gitlab.com/o/r/pull/3`` is not a PR, whatever the path looks like."""
    assert parse_url("https://gitlab.com/o/r/pull/3", GHE) is None
    assert parse_url("https://github.com/o/r/merge_requests/3", GHE) is None


def test_a_github_shaped_host_nobody_knows_is_detect_and_link():
    ref = parse_url("https://random.example.net/o/r/pull/5", EMPTY_CONTEXT)
    assert ref is not None and ref.full is False
    assert "gh CLI is logged into" in (ref.reason or "")


def test_a_confirmed_enterprise_host_is_full():
    from_login = parse_url("https://ghe.example.com/acme/api/pull/5", GHE)
    via_remote = HostContext(remotes=(Remote("origin", "ghe.example.com", "acme/api"),))
    from_remote = parse_url("https://ghe.example.com/acme/api/pull/5", via_remote)
    assert from_login is not None and from_login.full is True
    assert from_remote is not None and from_remote.full is True


def test_gitlab_is_confirmed_by_the_shape_alone():
    """``/-/merge_requests/N`` is unique to GitLab, so a self-hosted host needs no login."""
    ref = parse_url("https://gl.internal.acme.io/team/svc/-/merge_requests/9", EMPTY_CONTEXT)
    assert ref is not None and ref.full is True and ref.host == "gl.internal.acme.io"


def test_detect_and_link_forges_say_why():
    for url in (
        "https://codeberg.org/o/r/pulls/1",
        "https://bitbucket.org/o/r/pull-requests/1",
        "https://dev.azure.com/o/p/_git/r/pullrequest/1",
        "https://g.example.com/c/p/+/1",
    ):
        ref = parse_url(url, EMPTY_CONTEXT)
        assert ref is not None and ref.full is False and ref.reason


def test_qualified_refs_split_by_notation():
    issue_or_pr = parse_qualified("damianvtran/local-operator", "#", 2090, EMPTY_CONTEXT)
    assert issue_or_pr is not None and issue_or_pr.forge == "github" and issue_or_pr.full is False
    assert "could be an issue or a pull request" in (issue_or_pr.reason or "")

    mr = parse_qualified("minervaai/minerva-skills", "!", 57, GHE)
    assert mr is not None and mr.forge == "gitlab" and mr.number == 57

    # A GitLab project path with more than two segments is not GitHub's notation.
    assert parse_qualified("group/sub/project", "#", 3, EMPTY_CONTEXT) is None


def test_two_gitlab_hosts_leave_a_bare_ref_nowhere_to_land():
    two = HostContext(gitlab_hosts=frozenset({"gl.a.io", "gl.b.io"}))
    assert parse_qualified("group/project", "!", 5, two) is None
    one = HostContext(gitlab_hosts=frozenset({"gl.a.io"}))
    single = parse_qualified("group/project", "!", 5, one)
    assert single is not None and single.host == "gl.a.io"


def test_mentions_in_text_miss_prose_and_url_tails():
    text = (
        "see damianvtran/local-operator#2090 and minervaai/minerva-skills!57; "
        "#1 priority; README.md#3; "
        "https://github.com/o/r/pull/3#issuecomment-1"
    )
    keys = [ref.key for ref in iter_refs(text, EMPTY_CONTEXT)]
    assert "github.com/damianvtran/local-operator#2090" in keys
    assert "gitlab.com/minervaai/minerva-skills!57" in keys
    # The URL yields ONE ref: its own fragment must not also be read as a qualified ref
    # for the same number, and ``README.md#3`` is a file anchor, not a repository.
    assert keys.count("github.com/o/r#3") == 1
    assert not any(key.endswith("#1") for key in keys)
    assert not any(key.startswith("github.com/README.md") for key in keys)


def test_a_bare_gitlab_ref_prefers_the_public_instance():
    """A logged-in self-hosted GitLab must not capture a bare ``group/project!N``."""
    context = HostContext(gitlab_hosts=frozenset({"gitlab.com", "gl.internal.acme.io"}))
    public = parse_qualified("group/project", "!", 5, context)
    assert public is not None and public.host == "gitlab.com"
    self_hosted = HostContext(gitlab_hosts=frozenset({"gl.internal.acme.io"}))
    only = parse_qualified("group/project", "!", 5, self_hosted)
    assert only is not None and only.host == "gl.internal.acme.io"
    remote = HostContext(
        gitlab_hosts=frozenset({"gitlab.com", "gl.internal.acme.io"}),
        remotes=(Remote("origin", "gl.internal.acme.io", "group/project"),),
    )
    from_remote = parse_qualified("group/project", "!", 5, remote)
    assert from_remote is not None and from_remote.host == "gl.internal.acme.io"


def test_parse_any_takes_a_whole_string_only():
    from_url = parse_any("https://github.com/o/r/pull/4", GHE)
    from_ref = parse_any("o/r#4", GHE)
    assert from_url is not None and from_url.number == 4
    assert from_ref is not None and from_ref.number == 4
    assert parse_any("please look at https://github.com/o/r/pull/4", GHE) is None


@pytest.mark.parametrize(
    "remote,expected",
    [
        (
            "git@github.com:damianvtran/local-operator.git",
            ("github.com", "damianvtran/local-operator"),
        ),
        ("https://user@gitlab.com/g/s/p.git", ("gitlab.com", "g/s/p")),
        ("ssh://git@git.example.com:2222/acme/api.git", ("git.example.com", "acme/api")),
        ("not a remote", None),
    ],
)
def test_remote_urls_reduce_to_host_and_project(remote, expected):
    assert parse_remote_url(remote) == expected


def test_host_context_reads_this_machines_logins(tmp_path):
    (tmp_path / ".config" / "gh").mkdir(parents=True)
    (tmp_path / ".config" / "gh" / "hosts.yml").write_text(
        "github.com:\n    oauth_token: secret-do-not-read\n", encoding="utf-8"
    )
    glab = tmp_path / "Library" / "Application Support" / "glab-cli"
    glab.mkdir(parents=True)
    (glab / "config.yml").write_text(
        "hosts:\n    gitlab.com:\n        token: secret-do-not-read\n        api_protocol: https\n",
        encoding="utf-8",
    )
    context = load_host_context(None, home=tmp_path)
    assert context.github_hosts == frozenset({"github.com"})
    assert context.gitlab_hosts == frozenset({"gitlab.com"})
    # The token VALUES are never materialised: only names are read.
    assert "secret-do-not-read" not in repr(context)
