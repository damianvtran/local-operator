"""Detection: what counts as proof that a session opened or acted on a code request.

The load-bearing cases are the FALSE POSITIVES the design measured in the wild:

* a ``python3 - <<'EOF'`` heredoc whose STRING LITERALS contain "gh pr create" (session
  ``439818272d84``) — a regex over the command text counted it as a create;
* a real failed ``gh pr create`` (heredoc syntax error, ``exit code: 2``) immediately
  before the retry that worked, in the same session;
* a script that merely PRINTS a URL, which is ``unknown`` rather than ``opened``.

Every rule below is therefore stated as "the command word, exit 0, and the URL in the
CLI's own output shape", and the negatives are tested beside the positives.
"""

from __future__ import annotations

import pytest

from local_operator.code_requests.detect import (
    McpServer,
    could_matter,
    detect_bash,
    detect_mcp,
    detect_tool_result,
    load_mcp_servers,
    parse_bash_result,
    split_stages,
)
from local_operator.code_requests.refs import HostContext, Remote

GITHUB_CWD = HostContext(
    remotes=(Remote("origin", "github.com", "damianvtran/local-operator"),),
)
GITLAB_CWD = HostContext(remotes=(Remote("origin", "gitlab.com", "minervaai/minerva-skills"),))
GITLAB_MCP = (McpServer(prefix="mcp__gitlab_", host="gitlab.com", launch=""),)


def _bash(command: str, stdout: str, *, code: int = 0, stderr: str = "", timed_out: bool = False):
    body = f"exit code: {code}\n--- stdout ---\n{stdout}\n\n--- stderr ---\n{stderr or '(empty)'}"
    if timed_out:
        body = f"TIMEOUT after 120s (process killed)\n{body}"
    return detect_bash(command, body, GITHUB_CWD)


def _kinds(detections):
    return [
        (item.kind, item.rule, item.ref.key if item.ref else None, item.act) for item in detections
    ]


# -- the create rules -------------------------------------------------------


def test_gh_pr_create_on_the_cli_stdout_shape_is_opened():
    detections = _bash(
        "cd ~/wt && gh pr create --repo damianvtran/local-operator --title t --body-file b.md",
        "https://github.com/damianvtran/local-operator/pull/1904",
    )
    assert _kinds(detections) == [
        ("opened", "gh-pr-create-stdout", "github.com/damianvtran/local-operator#1904", None)
    ]
    assert detections[0].exit == 0 and detections[0].verb == "gh pr create"


def test_glab_mr_create_is_opened_and_the_creating_line_is_ignored():
    body = (
        "exit code: 0\n--- stdout ---\n\nCreating merge request for docs/x into main in "
        "minervaai/minerva-skills\n\n"
        "https://gitlab.com/minervaai/minerva-skills/-/merge_requests/53\n\n--- stderr ---\n(empty)"
    )
    detections = detect_bash("glab mr create --source-branch docs/x", body, GITLAB_CWD)
    assert _kinds(detections) == [
        ("opened", "glab-mr-create-stdout", "gitlab.com/minervaai/minerva-skills!53", None)
    ]


def test_a_quoted_final_line_is_still_the_prose_case():
    """A create verb whose stdout ENDS in prose is not a create: the URL must be the line."""
    detections = _bash("gh pr create --title t", "Created.\nSee the docs.")
    assert detections == []


def test_the_heredoc_false_positive_is_not_a_create():
    """The real one from 439818272d84: string literals, a python heredoc, exit 0."""
    command = (
        "python3 - <<'EOF'\n"
        "import json\n"
        'TEXT = "run gh pr create --title x to open it"\n'
        "print(json.dumps({'note': TEXT}))\n"
        "EOF"
    )
    assert _bash(command, '{"note": "run gh pr create --title x to open it"}') == []
    # A non-zero exit is never a create either — the real failed attempt before the retry.
    failed = _bash('gh pr create --title t --body "$(cat x)"', "", code=2, stderr="syntax error")
    assert failed == []


def test_a_script_printing_a_url_is_unknown_not_opened():
    detections = _bash("./scripts/open-pr.sh --title x", "done\nhttps://github.com/o/r/pull/77")
    assert _kinds(detections) == [("unknown", "script-stdout", "github.com/o/r#77", None)]
    assert "possibly opened by this call" in (detections[0].reason or "")


def test_an_interpreter_with_an_inline_program_is_not_a_script():
    """``python3 -c``/heredoc is the false-positive class; an interpreter with a SCRIPT
    PATH is the design's script case with a different suffix."""
    inline = _bash("python3 -c 'print(\"gh pr create\")'", "gh pr create --title x")
    assert inline == []
    heredoc = _bash("python3 - <<'EOF'\nprint('gh pr create')\nEOF", "gh pr create")
    assert heredoc == []
    script = _bash("python3 open-pr.py", "https://github.com/o/r/pull/77")
    assert [item.kind for item in script] == ["unknown"]
    # An interpreter with a flag is never a script path, whatever the URL looks like.
    assert _bash("bash -lc 'echo x'", "https://github.com/o/r/pull/77") == []


def test_a_compound_stage_still_counts():
    detections = _bash(
        "cd ~/wt && git push -u origin feat/x && gh pr create --fill",
        "https://github.com/damianvtran/local-operator/pull/1905",
    )
    assert [item.kind for item in detections] == ["opened"]


def test_git_push_create_link_is_a_hint_never_a_row():
    detections = _bash(
        "git push -u origin feat/x",
        "remote: \nremote: Create a pull request for 'feat/x' on GitHub by visiting:\n"
        "remote:      https://github.com/o/r/pull/new/feat/x\n",
    )
    assert [item.kind for item in detections] == ["hint"]
    assert detections[0].ref is None and detections[0].hint


# -- the act rules ----------------------------------------------------------


@pytest.mark.parametrize(
    "command,act",
    [
        ("gh pr comment 1904 --body hi", "comment"),
        ("gh pr merge 1904 --admin --squash", "merge"),
        ("gh pr review 1904 --approve", "review"),
        ("gh pr edit 1904 --title x", "edit"),
        ("gh pr close 1904", "close"),
        # ``gh pr ready`` marks a draft ready: an EDIT of the request, not a review.
        ("gh pr ready 1904", "edit"),
    ],
)
def test_gh_pr_verbs_are_acts_on_the_repos_ref(command, act):
    detections = _bash(command, "https://github.com/damianvtran/local-operator/pull/1904")
    assert _kinds(detections) == [
        ("acted", "gh-pr-act", "github.com/damianvtran/local-operator#1904", act)
    ]


@pytest.mark.parametrize(
    "command,act",
    [
        ("glab mr note 53 --message hi", "comment"),
        ("glab mr merge 53", "merge"),
        ("glab mr approve 53", "review"),
        ("glab mr update 53 --title x", "edit"),
        ("glab mr close 53", "close"),
    ],
)
def test_glab_mr_verbs_are_acts(command, act):
    body = (
        "exit code: 0\n--- stdout ---\n"
        "https://gitlab.com/minervaai/minerva-skills/-/merge_requests/53\n\n--- stderr ---\n(empty)"
    )
    assert _kinds(detect_bash(command, body, GITLAB_CWD)) == [
        ("acted", "glab-mr-act", "gitlab.com/minervaai/minerva-skills!53", act)
    ]


def test_a_read_verb_is_never_an_act():
    for command in (
        "gh pr view 1904 --json url",
        "gh pr list",
        "glab mr show 53",
        "gh pr diff 1904",
    ):
        assert _bash(command, "https://github.com/damianvtran/local-operator/pull/1904") == []


# -- the API rules ----------------------------------------------------------


def test_gh_api_post_that_returns_a_created_pull_alt():
    detections = _bash(
        "gh api repos/o/r/pulls -X POST -f title=x",
        '{"number": 31, "html_url": "https://github.com/o/r/pull/31"}',
    )
    assert _kinds(detections) == [("opened", "gh-api-create-json", "github.com/o/r#31", None)]


def test_gh_api_with_field_implies_post_and_a_get_does_not():
    implied = _bash(
        "gh api repos/o/r/pulls -f title=x",
        '{"number": 8, "html_url": "https://github.com/o/r/pull/8"}',
    )
    assert [item.kind for item in implied] == ["opened"]
    listed = _bash(
        "gh api repos/o/r/pulls", '[{"number": 8, "html_url": "https://github.com/o/r/pull/8"}]'
    )
    assert listed == []


def test_glab_api_post_for_merge_requests():
    body = (
        '{"iid": 54, "web_url": "https://gitlab.com/minervaai/minerva-skills/-/merge_requests/54"}'
    )
    detections = detect_bash(
        "glab api projects/minervaai%2Fminerva-skills/merge_requests -X POST -f title=x",
        f"exit code: 0\n--- stdout ---\n{body}\n\n--- stderr ---\n(empty)",
        GITLAB_CWD,
    )
    assert _kinds(detections) == [
        ("opened", "glab-api-create-json", "gitlab.com/minervaai/minerva-skills!54", None)
    ]


def test_curl_post_creating_a_pull_request():
    body = '{"number": 9, "html_url": "https://github.com/o/r/pull/9"}'
    detections = _bash(
        'curl -sS -X POST -d \'{"title": "x"}\' https://api.github.com/repos/o/r/pulls',
        body,
    )
    assert _kinds(detections) == [("opened", "curl-create-json", "github.com/o/r#9", None)]


def test_an_issue_comment_is_an_ambiguous_act():
    detections = _bash(
        "gh api repos/o/r/issues/12/comments -X POST -f body=x",
        '{"html_url": "https://github.com/o/r/issues/12#issuecomment-1"}',
    )
    assert _kinds(detections) == [("acted", "gh-api-issue-comment", "github.com/o/r#12", "comment")]
    assert (
        detections[0].rule
        in __import__(
            "local_operator.code_requests.detect", fromlist=["AMBIGUOUS_ACT_RULES"]
        ).AMBIGUOUS_ACT_RULES
    )


# -- MCP --------------------------------------------------------------------


def test_gitlab_mcp_save_merge_request_without_an_iid_creates():
    result = (
        '{"iid": 61, "web_url": "https://gitlab.com/minervaai/minerva-skills/-/merge_requests/61"}'
    )
    detections = detect_mcp(
        "mcp__gitlab_save_merge_request",
        {"project_id": "minervaai/minerva-skills", "title": "x"},
        result,
        GITLAB_CWD,
        GITLAB_MCP,
    )
    assert _kinds(detections) == [
        ("opened", "gitlab-mcp-create", "gitlab.com/minervaai/minerva-skills!61", None)
    ]


def test_gitlab_mcp_with_an_iid_edits_the_mr_it_names():
    result = (
        '{"iid": 99, "web_url": "https://gitlab.com/minervaai/minerva-skills/-/merge_requests/99"}'
    )
    detections = detect_mcp(
        "mcp__gitlab_save_merge_request",
        {"project_id": "minervaai/minerva-skills", "merge_request_iid": 53, "title": "x"},
        result,
        GITLAB_CWD,
        GITLAB_MCP,
    )
    assert _kinds(detections) == [
        ("acted", "gitlab-mcp-act", "gitlab.com/minervaai/minerva-skills!53", "edit")
    ]


def test_gitlab_mcp_notes_and_reviews_are_acts_on_the_argument_ref():
    for tool, method, act in (
        ("mcp__gitlab_save_note", {"body": "x"}, "comment"),
        (
            "mcp__gitlab_save_merge_request_review",
            {"method": "create_note", "body": "lgtm"},
            "review",
        ),
        ("mcp__gitlab_accept_merge_request", {}, "merge"),
    ):
        detections = detect_mcp(
            tool,
            {"url": "https://gitlab.com/g/p/-/merge_requests/9", **method},
            '{"ok": true}',
            GITLAB_CWD,
            GITLAB_MCP,
        )
        assert _kinds(detections) == [("acted", "gitlab-mcp-act", "gitlab.com/g/p!9", act)], tool


def test_a_gitlab_mcp_work_item_note_is_not_a_merge_request():
    detections = detect_mcp(
        "mcp__gitlab_save_note",
        {"project_id": "g/p", "work_item_iid": 4, "body": "x"},
        '{"ok": true}',
        GITLAB_CWD,
        GITLAB_MCP,
    )
    assert detections == []


def test_the_server_is_matched_by_its_configured_host_not_its_key():
    """The operator's MCP key is theirs to choose, so only the URL host identifies it."""
    keyed = McpServer(prefix="mcp__my_forge_", host="gitlab.com", launch="")
    detections = detect_mcp(
        "mcp__my_forge_save_merge_request",
        {"project_id": "g/p", "title": "x"},
        '{"iid": 2, "web_url": "https://gitlab.com/g/p/-/merge_requests/2"}',
        GITLAB_CWD,
        (keyed,),
    )
    assert [item.kind for item in detections] == ["opened"]
    # An MCP server whose host is NOT a forge is not classified at all.
    other = detect_mcp(
        "mcp__my_forge_save_merge_request",
        {"project_id": "g/p", "title": "x"},
        '{"iid": 2, "web_url": "https://gitlab.com/g/p/-/merge_requests/2"}',
        GITLAB_CWD,
        (McpServer(prefix="mcp__my_forge_", host="mcp.linear.app", launch=""),),
    )
    assert other == []


def test_github_mcp_names_are_detected_but_marked_unverified():
    servers = (McpServer(prefix="mcp__github_", host="api.githubcopilot.com", launch=""),)
    detections = detect_mcp(
        "mcp__github_create_pull_request",
        {"owner": "o", "repo": "r", "title": "x"},
        '{"number": 12, "html_url": "https://github.com/o/r/pull/12"}',
        GITHUB_CWD,
        servers,
    )
    assert [item.kind for item in detections] == ["opened"]
    assert detections[0].unverified is True


def test_an_errored_mcp_call_is_never_an_open():
    servers = (McpServer(prefix="mcp__gitlab_", host="gitlab.com", launch=""),)
    detections = detect_tool_result(
        "mcp__gitlab_save_merge_request",
        {"project_id": "g/p", "title": "x"},
        "Error: 403",
        is_error=True,
        context=GITLAB_CWD,
        mcp_servers=servers,
    )
    assert detections == []


# -- the gate, the splitter and the result parser ---------------------------


def test_could_matter_gates_on_tool_arguments_and_result_shape():
    assert could_matter("bash", {"command": "gh pr create --title x"}, "") is False
    assert could_matter("bash", {"command": "ls"}, "exit code: 0\n--- stdout ---\nok") is False
    assert could_matter("bash", {"command": "echo hi"}, "") is False
    # A create attempt with an EMPTY result still passes the gate: the caller decides, and a
    # pruned result is exactly the case the live seam exists for.
    # A successful result rides the command path: the exit line is part of the bash tool's
    # own result text, so a real call always carries it while an empty one cannot be
    # classified (nothing says the command succeeded).
    assert (
        could_matter("bash", {"command": "gh pr merge 3"}, "exit code: 0\n--- stdout ---\n") is True
    )
    assert could_matter("bash", {"command": "gh pr merge 3"}, "") is False
    # A non-bash, non-MCP tool is never classified live: a URL inside a file someone read
    # is a tool-output MENTION, which the scanner records, not a detection.
    assert could_matter("read", {"path": "x"}, "https://github.com/o/r/pull/3") is False
    assert (
        could_matter(
            "mcp__gitlab_save_note", {"url": "https://gitlab.com/g/p/-/merge_requests/1"}, ""
        )
        is True
    )


def test_split_stages_ignores_heredocs_and_quotes():
    stages = split_stages(
        "cat > /tmp/x <<'EOF'\ngh pr create --title x\nEOF\necho 'a && b' && gh pr merge 3"
    )
    assert stages == ["cat > /tmp/x <<'EOF'", "echo 'a && b'", "gh pr merge 3"]


def test_parse_bash_result_reads_the_harness_shapes():
    parsed = parse_bash_result(
        "exit code: 0\n--- stdout ---\nline\n--- stderr ---\nwarning\n\nmore"
    )
    assert parsed.exit == 0 and "line" in parsed.stdout and "warning" in parsed.stderr
    timed = parse_bash_result("TIMEOUT after 5s (process killed)\nexit code: 124\n--- stdout ---\n")
    assert timed.exit == 124
    empty = parse_bash_result("nothing familiar")
    assert empty.exit is None and empty.stdout == "" and empty.stderr == ""


def test_load_mcp_servers_never_raises_on_this_machine(tmp_path):
    assert isinstance(load_mcp_servers(str(tmp_path)), tuple)
    assert isinstance(load_mcp_servers(None), tuple)
