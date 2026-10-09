"""Decide, deterministically, whether a tool call opened, acted on, or hinted at a PR/MR.

WHY THIS IS STRICT. "This session opened PR #1904" is a claim the UI makes on the
session's behalf, and it is the one claim a reader cannot easily check. So
**"opened" needs proof from the tool's own output, never from text matching**
(design §A.4):

* The command is TOKENISED. The forge verb must be the COMMAND WORD of a top-level
  stage. Heredoc bodies, quoted strings and command substitutions never count.
  This rule comes from a real false positive found in session 439818272d84: a
  ``python3 - <<'EOF'`` audit script whose *string literals* contained
  "gh pr create". A regex over the command text counted it as a create. The
  tokeniser sees ``python3`` as the command word and drops the heredoc body.
* The exit code must be 0. The same session has a real failed ``gh pr create``
  (a heredoc syntax error, ``exit code: 2``) right before the retry that worked.
* The URL must be on stdout in the CLI's own output shape. For ``gh pr create``
  that is the last non-empty stdout line. For ``glab mr create`` it is a stdout
  line that is exactly the MR URL. For the API paths it is a response JSON object
  with ``number``+``html_url`` (GitHub/Gitea) or ``iid``+``web_url`` (GitLab).

Anything short of that is never "opened":

* a create verb ran but its output is not in shape → ``unknown`` ("possibly
  opened"), keeping the URL it printed;
* a script with no recognised verb printed a PR URL as its last line →
  ``unknown``. The scanner drops this when the ref was seen earlier, because a
  script that merely PRINTS a known PR is the common case;
* ``git push``'s "create a pull request" link → a ``hint``, never a row;
* anything a user pastes → mentioned. That is the scanner's job, not this module's.

MCP. A bridged tool is named ``mcp__<server>_<tool>`` (``mcp/tool_bridge.py``), and
``<server>`` is the operator's own key, so it proves nothing. The server is
identified by its CONFIGURED URL HOST (``gitlab → https://gitlab.com/api/v4/mcp``)
or, for a stdio server, by the package it runs. GitLab's ``save_merge_request``
creates when ``merge_request_iid`` is OMITTED (its own description: "Omitting it
always creates"). The GitHub MCP tool names here come from the server's public
docs and have NOT been checked against a live server on this machine, so those
detections carry ``unverified: True`` in their evidence.

PURE AND CHEAP. :func:`detect_tool_result` does no I/O. The caller supplies a
:class:`~local_operator.code_requests.refs.HostContext` and the MCP server map,
both loaded off the event loop. :func:`could_matter` is the allocation-free
pre-filter that keeps the post-tool hook at zero cost for the ~all tool calls that
have nothing to do with a forge.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping
from urllib.parse import unquote, urlsplit

from local_operator.code_requests.refs import (
    KNOWN_PUBLIC_HOSTS,
    HostContext,
    Ref,
    iter_urls,
    parse_url,
)

logger = logging.getLogger(__name__)

KIND_OPENED = "opened"
KIND_ACTED = "acted"
KIND_HINT = "hint"
KIND_UNKNOWN = "unknown"

#: Rules whose target number could be an issue OR a pull request: GitHub numbers
#: both from one sequence, so ``issues/N/comments`` comments on either. The
#: scanner applies these acts only to a ref it already knows from elsewhere, so a
#: comment on an issue never invents a PR row.
AMBIGUOUS_ACT_RULES = frozenset({"gh-api-issue-comment", "github-mcp-issue-comment"})


@dataclass(frozen=True)
class Detection:
    """One classified fact about a tool call.

    ``ref`` is None only for ``hint`` (a ``/pull/new/<branch>`` link names no PR).
    ``rule`` names the exact rule that fired, so every row can be audited back
    to its evidence (risk 1 in the design: a false "opened" must be traceable).
    """

    kind: str
    ref: Ref | None
    rule: str
    verb: str | None = None
    exit: int | None = None
    act: str | None = None
    reason: str | None = None
    #: A rule's own payload: the ``git push`` hint's url/project, or the candidate refs an
    #: unattributable create printed (``{"candidates": [payload, …]}``). ``Any`` because it
    #: is a bag a reader inspects, never a value any rule branches on.
    hint: Mapping[str, Any] | None = None
    unverified: bool = False


@dataclass(frozen=True)
class McpServer:
    """What the MCP config says about one server, reduced to identity facts."""

    #: The sanitized prefix the bridge mints tool names under: ``mcp__<s>_``.
    prefix: str
    #: The configured URL's host for an http/sse server, else ``""``.
    host: str
    #: ``command + args`` of a stdio server, joined, else ``""``.
    launch: str


# ---------------------------------------------------------------------------
# Pre-filter
# ---------------------------------------------------------------------------

#: Substrings without which a bash call cannot match any rule. The command carries
#: the verb, and the result carries the URL (the script rule has no verb, so the
#: result alone must be enough to keep it).
_COMMAND_MARKERS = ("gh", "glab", "tea", "az ", "curl", "git")
_RESULT_MARKERS = ("/pull", "merge_request", "pullrequest", "pull-requests", "/+/")


def could_matter(tool_name: str, args: Mapping[str, Any], result_text: str) -> bool:
    """Cheap gate: can this call produce any detection at all?

    Called on EVERY tool result by the live hook. It must stay a handful of
    substring checks: no regex, no tokenising, no I/O.
    """
    if tool_name.startswith("mcp__"):
        return True
    if tool_name != "bash":
        return False
    command = args.get("command")
    if not isinstance(command, str):
        return False
    if any(marker in result_text for marker in _RESULT_MARKERS):
        return True
    return any(marker in command for marker in _COMMAND_MARKERS) and "exit code: 0" in result_text


# ---------------------------------------------------------------------------
# Shell tokenising: top-level stages only
# ---------------------------------------------------------------------------

_HEREDOC = re.compile(r"<<(-?)[ \t]*(['\"]?)([^\s'\"<>;&|()]+)\2")
_SEPARATORS = ";&|()\n"


def split_stages(command: str) -> list[str]:
    """The TOP-LEVEL simple commands of ``command``, in order.

    Splits at unquoted ``; & && || | ( )`` and newlines. Each of the following is
    dropped and can never be read as a command: heredoc bodies, comments, the
    contents of ``$( … )`` and backticks, and quoted text (which stays inside its
    stage's words). That one property is what makes "the verb is the command word"
    mean something. The scanner stops at ``len(command)`` whatever the input, and
    an unterminated construct just ends the last stage.
    """
    stages, _ = _scan(command, 0, nested=False)
    return [stage for stage in (s.strip() for s in stages) if stage]


def _scan(command: str, start: int, *, nested: bool) -> tuple[list[str], int]:
    """The shared scanner. ``nested`` stops at the ``)`` that closes a ``$(``."""
    stages: list[str] = []
    current: list[str] = []
    heredocs: list[tuple[str, bool]] = []
    quote: str | None = None
    depth = 0
    word_start = True
    i = start
    length = len(command)
    while i < length:
        ch = command[i]
        if quote == "'":
            current.append(ch)
            if ch == "'":
                quote = None
            i += 1
            continue
        if quote == '"':
            if ch == "\\" and i + 1 < length:
                current.append(command[i : i + 2])
                i += 2
                continue
            if ch == "$" and command.startswith("$(", i):
                end = _skip_substitution(command, i)
                current.append(command[i:end])
                i = end
                continue
            if ch == "`":
                end = _skip_backticks(command, i)
                current.append(command[i:end])
                i = end
                continue
            if ch == '"':
                quote = None
            current.append(ch)
            i += 1
            continue
        # -- unquoted --------------------------------------------------------
        if ch == "\\" and i + 1 < length:
            if command[i + 1] == "\n":
                i += 2  # a line continuation joins the two lines
                continue
            current.append(command[i : i + 2])
            i += 2
            word_start = False
            continue
        if ch in ("'", '"'):
            quote = ch
            current.append(ch)
            i += 1
            word_start = False
            continue
        if ch == "$" and command.startswith("$(", i):
            end = _skip_substitution(command, i)
            current.append(command[i:end])
            i = end
            word_start = False
            continue
        if ch == "`":
            end = _skip_backticks(command, i)
            current.append(command[i:end])
            i = end
            word_start = False
            continue
        if ch == "#" and word_start:
            newline = command.find("\n", i)
            i = length if newline < 0 else newline
            continue
        if ch == "<" and command.startswith("<<", i) and not command.startswith("<<<", i):
            match = _HEREDOC.match(command, i)
            if match is not None:
                heredocs.append((match.group(3), match.group(1) == "-"))
                current.append(match.group(0))
                i = match.end()
                word_start = False
                continue
        if (
            ch == ">"
            or (ch == "<" and not command.startswith("<<", i))
            or (ch == "&" and i + 1 < length and command[i + 1] == ">")
        ):
            # A REDIRECTION is where output goes, never an argument and never a stage
            # boundary. Reading the ``&`` of ``2>&1`` as a separator split
            # ``gh pr create -f 2>&1 | tail -1`` into a phantom stage ``1`` — a stage that
            # could "print anything", which made every URL in the command unattributable
            # and turned a real create into ``unknown``.
            _strip_trailing_fd(current)
            i = _skip_redirect(command, i)
            word_start = False
            continue
        if nested and ch == ")" and depth == 0:
            stages.append("".join(current))
            return stages, i + 1
        if ch in _SEPARATORS:
            if ch == "(":
                depth += 1
            elif ch == ")" and depth > 0:
                depth -= 1
            stages.append("".join(current))
            current = []
            word_start = True
            i += 1
            if ch == "\n" and heredocs:
                i = _skip_heredoc_bodies(command, i, heredocs)
                heredocs = []
            continue
        current.append(ch)
        word_start = ch in " \t"
        i += 1
    stages.append("".join(current))
    return stages, length


def _strip_trailing_fd(current: list[str]) -> None:
    """Drop a file-descriptor number that the redirect operator belongs to (``2`` in ``2>&1``)."""
    text = "".join(current)
    if re.search(r"(?:^|\s)\d+$", text):
        current[:] = [re.sub(r"\d+$", "", text)]


def _redirect_operator_length(command: str, i: int) -> int:
    """How many characters the redirect operator at ``i`` spans (``>``, ``>>``, ``>&``, ``&>>``)."""
    if command.startswith("&>>", i):
        return 3
    if command.startswith("&>", i):
        return 2
    if command.startswith(">>", i):
        return 2
    if command.startswith((">&", "<&", "<>"), i):
        return 2
    return 1


def _skip_redirect(command: str, i: int) -> int:
    """Index just past a redirection and its target, so neither becomes an argument.

    ``&1``/``&2`` are part of the operator; ``/dev/null`` or a filename after it is the
    target. Quotes are respected: ``> "my file"`` consumes the quoted name as one target.
    """
    length = len(command)
    i += _redirect_operator_length(command, i)
    if i < length and command[i] == "&":
        i += 1
    # ``> file`` puts the target on the far side of a space: skip the gap before deciding
    # whether anything is left to consume, or the target stays behind as an argument.
    while i < length and command[i] in " \t":
        i += 1
    if i >= length or command[i] in _SEPARATORS:
        return i
    while i < length and not command[i].isspace() and command[i] not in _SEPARATORS:
        if command[i] in ("'", '"'):
            quote = command[i]
            i += 1
            while i < length and command[i] != quote:
                i += 1
        i += 1
    return i


def _skip_heredoc_bodies(command: str, i: int, heredocs: list[tuple[str, bool]]) -> int:
    """Skip each pending heredoc body, in order. Return the index after the last delimiter."""
    length = len(command)
    for delimiter, strip_tabs in heredocs:
        while i < length:
            newline = command.find("\n", i)
            end = length if newline < 0 else newline
            line = command[i:end]
            i = end + 1 if newline >= 0 else length
            if (line.lstrip("\t") if strip_tabs else line) == delimiter:
                break
    return min(i, length)


def _skip_substitution(command: str, i: int) -> int:
    """Index just past the ``)`` closing the ``$(`` at ``i``, heredoc- and quote-aware."""
    _, end = _scan(command, i + 2, nested=True)
    return end


def _skip_backticks(command: str, i: int) -> int:
    j = i + 1
    while j < len(command):
        if command[j] == "\\":
            j += 2
            continue
        if command[j] == "`":
            return j + 1
        j += 1
    return len(command)


_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
#: Wrappers that run their arguments as the command. ``env`` takes its own
#: options and ``NAME=value`` pairs first.
_WRAPPERS = frozenset({"command", "builtin", "exec", "nohup", "time", "noglob"})


def stage_words(stage: str) -> list[str]:
    """``stage`` as resolved words with the command word first, or ``[]`` when it does not parse.

    Leading assignments (``GH_PAGER= gh …``), ``env [opts] [NAME=v]`` and plain
    wrappers (``command``, ``time``, …) are stripped, so ``env -u XPC_FLAGS gh pr
    create`` still has ``gh`` as its command word. That is exactly how the
    operator's team spells it.
    """
    try:
        words = shlex.split(stage, posix=True)
    except ValueError:
        return []
    while words:
        head = words[0]
        if head in ("{", "}", "!") or _ASSIGNMENT.match(head):
            words = words[1:]
            continue
        if head in _WRAPPERS:
            words = words[1:]
            continue
        if os.path.basename(head) == "env":
            words = words[1:]
            while words and (words[0].startswith("-") or _ASSIGNMENT.match(words[0])):
                flag = words[0]
                words = words[1:]
                if flag in ("-u", "--unset", "-C", "--chdir", "-S") and words:
                    words = words[1:]
            continue
        break
    return words


def command_word(words: list[str]) -> str:
    return os.path.basename(words[0]) if words else ""


def _commands(command: str, *, depth: int = 0) -> list[list[str]]:
    """Every top-level stage's words. One level into ``bash|sh|zsh -c '<cmd>'``.

    A ``-c`` string IS executed as commands, unlike a heredoc fed to python, so its
    stages are real stages. One level is enough for every shape seen in practice,
    and the bound keeps a pathological input linear.
    """
    out: list[list[str]] = []
    for stage in split_stages(command):
        words = stage_words(stage)
        if not words:
            continue
        name = command_word(words)
        if depth == 0 and name in ("bash", "sh", "zsh") and len(words) >= 3:
            flags = words[1]
            if flags.startswith("-") and "c" in flags[1:] and not flags.startswith("--"):
                out.extend(_commands(words[2], depth=1))
                continue
        out.append(words)
    return out


# ---------------------------------------------------------------------------
# Bash result text
# ---------------------------------------------------------------------------

_EXIT_LINE = re.compile(r"^exit code: (-?\d+)\s*$", re.MULTILINE)
_STDOUT_MARK = "--- stdout ---\n"
_STDERR_MARK = "\n--- stderr ---\n"


@dataclass(frozen=True)
class BashResult:
    exit: int | None
    stdout: str
    stderr: str


def parse_bash_result(text: str) -> BashResult:
    """Split the bash tool's result text (``tools/builtin.py``: exit line + two sections).

    Lines before ``exit code:`` (``TIMEOUT …``, the memory line) are ignored. An
    ``(empty)`` section is the empty string. Anything after stderr (a spill footer,
    hook notes) stays in stderr, where no rule reads it.
    """
    match = _EXIT_LINE.search(text)
    exit_code = int(match.group(1)) if match else None
    start = text.find(_STDOUT_MARK)
    if start < 0:
        return BashResult(exit_code, "", "")
    start += len(_STDOUT_MARK)
    end = text.find(_STDERR_MARK, start)
    stdout = text[start:] if end < 0 else text[start:end]
    stderr = "" if end < 0 else text[end + len(_STDERR_MARK) :]
    if stdout.strip() == "(empty)":
        stdout = ""
    if stderr.strip() == "(empty)":
        stderr = ""
    return BashResult(exit_code, stdout, stderr)


def _last_line(text: str) -> str:
    for line in reversed(text.splitlines()):
        if line.strip():
            return line.strip()
    return ""


def _line_ref(line: str, context: HostContext, forges: Iterable[str]) -> Ref | None:
    """``line`` is EXACTLY one code-request URL of an allowed forge, else None."""
    stripped = line.strip()
    if not stripped or any(c.isspace() for c in stripped):
        return None
    ref = parse_url(stripped, context)
    if ref is None or ref.forge not in forges:
        return None
    return ref


def _last_url_ref(text: str, context: HostContext, forges: Iterable[str]) -> Ref | None:
    allowed = set(forges)
    last: Ref | None = None
    for ref in iter_urls(text, context):
        if ref.forge in allowed:
            last = ref
    return last


def _json_object(text: str) -> Any:
    """The JSON value in ``text``, tolerating a header block (``gh api -i``). None if absent."""
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except ValueError:
        pass
    brace = stripped.find("{")
    if brace < 0:
        return None
    try:
        return json.loads(stripped[brace:])
    except ValueError:
        return None


def _shaped_ref(value: Any, context: HostContext, *, depth: int = 0) -> Ref | None:
    """A create response: ``number``+``html_url`` or ``iid``+``web_url`` that AGREE.

    Agreement is the check that matters: the URL must parse to a ref whose number
    is the response's own number. A response that merely contains some URL (a
    linked issue, the base repo) is not a create response.
    """
    if depth > 3:
        return None
    if isinstance(value, dict):
        for number_key, url_key in (("number", "html_url"), ("iid", "web_url")):
            number = value.get(number_key)
            url = value.get(url_key)
            if isinstance(number, int) and not isinstance(number, bool) and isinstance(url, str):
                ref = parse_url(url, context)
                if ref is not None and ref.number == number:
                    return ref
        for key in ("merge_request", "mergeRequest", "pull_request", "pullRequest", "data"):
            nested = value.get(key)
            if nested is not None:
                found = _shaped_ref(nested, context, depth=depth + 1)
                if found is not None:
                    return found
    return None


# ---------------------------------------------------------------------------
# Verb tables
# ---------------------------------------------------------------------------

_GH_CREATE = frozenset({"create", "new"})
_GLAB_CREATE = frozenset({"create", "new"})
_TEA_PULLS = frozenset({"pulls", "pull", "pr"})
_TEA_CREATE = frozenset({"create", "c", "new"})

_GH_ACTS = {
    "comment": "comment",
    "merge": "merge",
    "review": "review",
    "edit": "edit",
    "close": "close",
    "ready": "edit",
    "reopen": "edit",
}
_GLAB_ACTS = {
    "note": "comment",
    "comment": "comment",
    "merge": "merge",
    "approve": "review",
    "revoke": "review",
    "update": "edit",
    "close": "close",
    "reopen": "edit",
}

#: Flags that CONSUME the next word, across the act verbs. A flag missing here
#: costs at most a wrongly skipped target: an unknown value is never taken as a
#: number unless it is all digits, and a body text rarely is.
_VALUE_FLAGS = frozenset(
    {
        "-R",
        "--repo",
        "-b",
        "--body",
        "-F",
        "--body-file",
        "-t",
        "--title",
        "-B",
        "--base",
        "-H",
        "--head",
        "-a",
        "--assignee",
        "-l",
        "--label",
        "--milestone",
        "-p",
        "--project",
        "-T",
        "--template",
        "--add-assignee",
        "--remove-assignee",
        "--add-label",
        "--remove-label",
        "--add-reviewer",
        "--remove-reviewer",
        "--add-project",
        "--remove-project",
        "--subject",
        "--match-head-commit",
        "-A",
        "--author-email",
        "--message",
        "--sha",
    }
)


def _flag_value(words: list[str], names: Iterable[str]) -> str | None:
    wanted = set(names)
    for index, word in enumerate(words):
        if word in wanted and index + 1 < len(words):
            return words[index + 1]
        for name in wanted:
            if name.startswith("--") and word.startswith(name + "="):
                return word[len(name) + 1 :]
    return None


def _positionals(words: list[str], value_flags: frozenset[str]) -> list[str]:
    out: list[str] = []
    skip = False
    for word in words:
        if skip:
            skip = False
            continue
        if word == "--":
            continue
        if word.startswith("-"):
            if word in value_flags:
                skip = True
            continue
        out.append(word)
    return out


def _github_target(
    rest: list[str], context: HostContext, *, value_flags: frozenset[str] = _VALUE_FLAGS
) -> Ref | None:
    """The PR an act names: a URL, or a number resolved against ``-R`` or the one remote."""
    for word in _positionals(rest, value_flags):
        if "://" in word:
            ref = parse_url(word, context)
            return ref if ref is not None and ref.forge == "github" else None
        number = word.lstrip("#")
        if number.isdigit() and int(number) > 0:
            repo = _flag_value(rest, ("-R", "--repo"))
            return _github_ref(repo, int(number), context)
        return None  # a branch name: the PR it maps to is not knowable here
    return None


def _github_ref(repo: str | None, number: int, context: HostContext) -> Ref | None:
    if repo:
        parts = repo.strip("/").split("/")
        if len(parts) == 3:
            host, owner, name = parts
        elif len(parts) == 2:
            host, owner, name = "github.com", parts[0], parts[1]
        else:
            return None
        return parse_url(f"https://{host}/{owner}/{name}/pull/{number}", context)
    projects = {
        (r.host, r.project)
        for r in context.remotes
        if r.project.count("/") == 1 and context.confirms(r.host, "github")
    }
    if len(projects) != 1:
        return None
    host, project = next(iter(projects))
    return parse_url(f"https://{host}/{project}/pull/{number}", context)


def _gitlab_target(rest: list[str], context: HostContext) -> Ref | None:
    for word in _positionals(rest, _VALUE_FLAGS | {"-m"}):
        if "://" in word:
            ref = parse_url(word, context)
            return ref if ref is not None and ref.forge == "gitlab" else None
        number = word.lstrip("!")
        if number.isdigit() and int(number) > 0:
            return _gitlab_ref(_flag_value(rest, ("-R", "--repo")), int(number), context)
        return None
    return None


def _is_gitlab_host(host: str, context: HostContext) -> bool:
    return KNOWN_PUBLIC_HOSTS.get(host) == "gitlab" or host in context.gitlab_hosts


def _gitlab_ref(repo: str | None, number: int, context: HostContext) -> Ref | None:
    if repo:
        if "://" in repo:
            parts = urlsplit(repo)
            host, project = parts.netloc.lower(), parts.path.strip("/")
        else:
            segments = repo.strip("/").split("/")
            if len(segments) >= 3 and "." in segments[0]:
                host, project = segments[0].lower(), "/".join(segments[1:])
            else:
                host, project = "", repo.strip("/")
        if not host:
            hosts = {r.host for r in context.remotes if _is_gitlab_host(r.host, context)}
            hosts |= set(context.gitlab_hosts)
            host = next(iter(hosts)) if len(hosts) == 1 else "gitlab.com"
        return parse_url(f"https://{host}/{project}/-/merge_requests/{number}", context)
    projects = {(r.host, r.project) for r in context.remotes if _is_gitlab_host(r.host, context)}
    if len(projects) != 1:
        return None
    host, project = next(iter(projects))
    return parse_url(f"https://{host}/{project}/-/merge_requests/{number}", context)


# ---------------------------------------------------------------------------
# API calls (gh api, glab api, curl)
# ---------------------------------------------------------------------------

_GH_API_VALUE_FLAGS = frozenset(
    {
        "-X",
        "--method",
        "-f",
        "--raw-field",
        "-F",
        "--field",
        "--input",
        "-H",
        "--header",
        "-q",
        "--jq",
        "-t",
        "--template",
        "-p",
        "--preview",
        "--hostname",
        "--cache",
    }
)
_IMPLIES_POST = frozenset({"-f", "--raw-field", "-F", "--field", "--input"})


@dataclass(frozen=True)
class _ApiCall:
    method: str
    endpoint: str  # path only, no leading slash, no query
    host: str | None


def _cli_api_call(rest: list[str]) -> _ApiCall | None:
    """``gh api``/``glab api`` arguments → method and endpoint path."""
    method: str | None = None
    implied = False
    endpoint: str | None = None
    hostname: str | None = None
    skip_next = False
    for index, word in enumerate(rest):
        if skip_next:
            skip_next = False
            continue
        flag, eq, inline = word.partition("=")
        if word.startswith("-"):
            name = flag if word.startswith("--") and eq else word
            value = inline if word.startswith("--") and eq else None
            if name in _GH_API_VALUE_FLAGS:
                if value is None:
                    value = rest[index + 1] if index + 1 < len(rest) else ""
                    skip_next = True
                if name in ("-X", "--method"):
                    method = value.upper()
                elif name in _IMPLIES_POST:
                    implied = True
                elif name == "--hostname":
                    hostname = value
            elif word.startswith("-X") and len(word) > 2:
                method = word[2:].upper()
            continue
        if endpoint is None:
            endpoint = word
    if endpoint is None:
        return None
    resolved = method or ("POST" if implied else "GET")
    return _ApiCall(resolved, _endpoint_path(endpoint), hostname)


_CURL_VALUE_FLAGS = frozenset(
    {
        "-H",
        "--header",
        "-o",
        "--output",
        "-u",
        "--user",
        "-A",
        "--user-agent",
        "-b",
        "--cookie",
        "-c",
        "--cookie-jar",
        "-e",
        "--referer",
        "-w",
        "--write-out",
        "-K",
        "--config",
        "--connect-timeout",
        "-m",
        "--max-time",
        "--retry",
        "--oauth2-bearer",
        "--url",
    }
)
_CURL_DATA = frozenset(
    {
        "-d",
        "--data",
        "--data-raw",
        "--data-binary",
        "--data-urlencode",
        "--data-ascii",
        "--json",
        "-F",
        "--form",
        "-T",
        "--upload-file",
    }
)


def _curl_call(rest: list[str]) -> _ApiCall | None:
    method: str | None = None
    implied = False
    get = False
    url: str | None = None
    skip_next = False
    for index, word in enumerate(rest):
        if skip_next:
            skip_next = False
            continue
        name, eq, inline = word.partition("=")
        if word.startswith("--") and eq:
            if name in ("--request",):
                method = inline.upper()
            elif name in _CURL_DATA:
                implied = True
            elif name == "--url":
                url = inline
            continue
        if word in ("-X", "--request"):
            method = (rest[index + 1] if index + 1 < len(rest) else "").upper()
            skip_next = True
            continue
        if word.startswith("-X") and len(word) > 2 and not word.startswith("--"):
            method = word[2:].upper()
            continue
        if word in _CURL_DATA:
            implied = True
            skip_next = True
            continue
        if word in ("-G", "--get"):
            get = True
            continue
        if word == "--url":
            url = rest[index + 1] if index + 1 < len(rest) else None
            skip_next = True
            continue
        if word in _CURL_VALUE_FLAGS:
            skip_next = True
            continue
        if word.startswith("-"):
            continue
        if url is None and re.match(r"^https?://", word):
            url = word
    if url is None:
        return None
    parts = urlsplit(url)
    resolved = method or ("GET" if get else ("POST" if implied else "GET"))
    return _ApiCall(resolved, _endpoint_path(parts.path), parts.netloc.lower() or None)


def _endpoint_path(endpoint: str) -> str:
    path = endpoint.split("?", 1)[0].split("#", 1)[0]
    if "://" in path:
        path = urlsplit(path).path
    return unquote(path).strip("/")


_GH_CREATE_ENDPOINT = re.compile(r"(?:^|/)repos/[^/]+/[^/]+/pulls$")
_GL_CREATE_ENDPOINT = re.compile(r"(?:^|/)projects/.+/merge_requests$")
_GH_ACT_ENDPOINT = re.compile(
    r"(?:^|/)repos/(?P<owner>[^/{}]+)/(?P<repo>[^/{}]+)/"
    r"(?P<kind>issues|pulls)/(?P<n>[1-9]\d*)/(?P<what>comments|reviews|merge)$"
)
_GL_ACT_ENDPOINT = re.compile(
    r"(?:^|/)projects/(?P<project>.+?)/merge_requests/(?P<n>[1-9]\d*)/"
    r"(?P<what>notes|discussions|merge|approve)$"
)


def _api_detections(
    call: _ApiCall, verb: str, result: BashResult, context: HostContext
) -> list[Detection]:
    is_create = call.method == "POST" and (
        _GH_CREATE_ENDPOINT.search(call.endpoint) or _GL_CREATE_ENDPOINT.search(call.endpoint)
    )
    if is_create:
        ref = _shaped_ref(_json_object(result.stdout), context)
        if ref is not None:
            return [Detection(KIND_OPENED, ref, f"{_rule_stem(verb)}-create-json", verb, 0)]
        loose = _last_url_ref(result.stdout, context, ("github", "gitlab", "gitea"))
        if loose is not None:
            return [
                Detection(
                    KIND_UNKNOWN,
                    loose,
                    f"{_rule_stem(verb)}-create-unshaped",
                    verb,
                    0,
                    reason=(
                        "a create request ran, but its response is not the create JSON "
                        "shape (number + html_url, or iid + web_url)"
                    ),
                )
            ]
        return []
    gh = _GH_ACT_ENDPOINT.search(call.endpoint)
    if gh is not None:
        what, kind = gh.group("what"), gh.group("kind")
        act = {
            ("comments", "POST"): "comment",
            ("reviews", "POST"): "review",
            ("merge", "PUT"): "merge",
        }.get((what, call.method))
        if act is None or (kind == "issues" and what != "comments"):
            return []
        host = call.host if call.host and call.host not in ("api.github.com",) else "github.com"
        if host.startswith("api."):
            host = host[4:]
        ref = parse_url(
            f"https://{host}/{gh.group('owner')}/{gh.group('repo')}/pull/{gh.group('n')}",
            context,
        )
        if ref is None:
            return []
        rule = "gh-api-issue-comment" if kind == "issues" else f"{_rule_stem(verb)}-act"
        return [Detection(KIND_ACTED, ref, rule, verb, 0, act=act)]
    gl = _GL_ACT_ENDPOINT.search(call.endpoint)
    if gl is not None:
        act = {
            ("notes", "POST"): "comment",
            ("discussions", "POST"): "comment",
            ("merge", "PUT"): "merge",
            ("approve", "POST"): "review",
        }.get((gl.group("what"), call.method))
        project = gl.group("project")
        if act is None or project.isdigit() or project.startswith(":"):
            return []
        host = call.host or _single_gitlab_host(context)
        ref = parse_url(f"https://{host}/{project}/-/merge_requests/{gl.group('n')}", context)
        return [] if ref is None else [Detection(KIND_ACTED, ref, "glab-api-act", verb, 0, act=act)]
    return []


def _single_gitlab_host(context: HostContext) -> str:
    hosts = set(context.gitlab_hosts)
    return next(iter(hosts)) if len(hosts) == 1 else "gitlab.com"


def _rule_stem(verb: str) -> str:
    return verb.replace(" ", "-")


# ---------------------------------------------------------------------------
# Bash rules
# ---------------------------------------------------------------------------

_FORGE_CLIS = frozenset({"gh", "glab", "tea", "az"})

#: Forge verbs whose output url ALWAYS carries a fragment (``#issuecomment-…``,
#: ``#note_…``), so the stage can never own a bare url line.
_FRAGMENT_ONLY_VERBS = frozenset(
    {("pr", "comment"), ("issue", "comment"), ("mr", "note"), ("issue", "note")}
)

#: Stages that cannot print a code-request url of their own: shell builtins, file
#: housekeeping, and ``git`` (whose url-shaped output is a push hint, on stderr, prefixed).
_BLIND_COMMANDS = frozenset(
    {
        "cd",
        "pushd",
        "popd",
        "export",
        "unset",
        "set",
        "source",
        ".",
        "wait",
        "sleep",
        "true",
        "false",
        ":",
        "mkdir",
        "rmdir",
        "rm",
        "mv",
        "cp",
        "touch",
        "chmod",
        "chown",
        "ln",
        "test",
        "[",
        "[[",
        "declare",
        "local",
        "shift",
        "trap",
        "umask",
        "git",
        "kill",
        "wait",
    }
)

#: Filters that print only their stdin, and therefore add no line of their own — but
#: only while they have no file operand (``cat summary.txt`` prints a file: see
#: ``_has_file_operand``; ``tee report.txt`` never reads one).
_PASS_THROUGH_COMMANDS = frozenset({"cat", "tail", "head", "sort", "uniq", "cut", "tr", "nl"})
_PASS_THROUGH_ALWAYS = frozenset({"tee", "wc"})
_INTERPRETERS = frozenset(
    {"bash", "sh", "zsh", "python", "python3", "node", "ruby", "perl", "deno", "bun"}
)
_SCRIPT_SUFFIXES = (".sh", ".bash", ".zsh", ".py", ".js", ".mjs", ".ts", ".rb", ".pl")

#: Shell constructs whose text IS executed but which ``split_stages`` cannot see through
#: as a stage of its own: a command substitution and a loop body. A create inside one of
#: these is a MISS if it is not recognised (QA round 2, row 3f: ``URL=$(gh pr create -f)``
#: and ``for b in …; do gh pr create …; done`` recorded nothing at all).
_SUBSTITUTION = re.compile(r"\$\((?P<sub>[^()]*(?:\([^()]*\)[^()]*)*)\)|`(?P<tick>[^`]*)`")
_LOOP_BODY = re.compile(r"\b(?:for|while)\b(?P<loop>.*?)\bdone\b", re.DOTALL)
_CREATE_INVOCATION = re.compile(
    r"\b(?:gh\s+pr\s+create|glab\s+mr\s+create|tea\s+pulls\s+create|az\s+repos\s+pr\s+create)\b"
)


def _is_script_stage(words: list[str]) -> bool:
    """A stage that runs a SCRIPT FILE: a path command word, or an interpreter plus a file.

    ``python3 -`` (stdin/heredoc) and ``python3 -c`` are not script files. Neither
    is a bare binary on PATH: ``ls`` printing a URL is not a script that might
    have opened a PR.
    """
    head = words[0]
    name = os.path.basename(head)
    if name in _FORGE_CLIS:
        return False
    if "/" in head or head.endswith(_SCRIPT_SUFFIXES):
        return True
    if name in _INTERPRETERS or re.fullmatch(r"python3\.\d+", name):
        for word in words[1:]:
            if word in ("-c", "-", "-e", "-m"):
                return False
            if word.startswith("-"):
                continue
            return word.endswith(_SCRIPT_SUFFIXES) or "/" in word
    return False


def _git_push_hints(result: BashResult) -> list[Detection]:
    """``git push``'s "create a pull/merge request" links: hints, never rows."""
    hints: list[Detection] = []
    seen: set[str] = set()
    for match in re.finditer(r"https?://\S+", result.stderr + "\n" + result.stdout):
        url = match.group(0).rstrip(".,;")
        parts = urlsplit(url)
        path = unquote(parts.path)
        branch: str | None = None
        project: str | None = None
        gh = re.match(r"^/([^/]+/[^/]+)/pull/new/(.+)$", path)
        if gh is not None:
            project, branch = gh.group(1), gh.group(2)
        gl = re.match(r"^/(.+?)/-/merge_requests/new$", path)
        if gl is not None:
            project = gl.group(1)
            found = re.search(r"source_branch\]=([^&]+)", unquote(parts.query))
            branch = found.group(1) if found else None
        if project is None or url in seen:
            continue
        seen.add(url)
        hint = {"url": url, "host": parts.netloc.lower(), "project": project}
        if branch:
            hint["branch"] = branch
        hints.append(Detection(KIND_HINT, None, "git-push-hint", "git push", 0, hint=hint))
    return hints


def detect_bash(command: str, result_text: str, context: HostContext) -> list[Detection]:
    """Classify one bash call. See the module docstring for the rules."""
    result = parse_bash_result(result_text)
    if result.exit != 0:
        # A failed create proves nothing, and neither does a failed act. The
        # failed ``gh pr create`` in 439818272d84 printed nothing to stdout, but
        # a non-zero exit is refused before stdout is even read.
        return []
    stages = _commands(command)
    if not stages:
        return []
    detections: list[Detection] = []
    names = [command_word(words) for words in stages]
    for index, words in enumerate(stages):
        name = command_word(words)
        rest = words[1:]
        stage = {"stages": stages, "index": index}
        if name == "gh" and len(rest) >= 2 and rest[0] == "pr" and rest[1] in _GH_CREATE:
            detections.extend(_create_by_cli(result, context, "gh pr create", "github", **stage))
        elif name == "glab" and len(rest) >= 2 and rest[0] == "mr" and rest[1] in _GLAB_CREATE:
            detections.extend(_create_by_cli(result, context, "glab mr create", "gitlab", **stage))
        elif name == "tea" and len(rest) >= 2 and rest[0] in _TEA_PULLS and rest[1] in _TEA_CREATE:
            detections.extend(_create_by_cli(result, context, "tea pulls create", "gitea", **stage))
        elif name == "az" and rest[:3] == ["repos", "pr", "create"]:
            detections.extend(_create_by_az(result, context, stages=stages, index=index))
        elif name == "gh" and len(rest) >= 2 and rest[0] == "pr" and rest[1] in _GH_ACTS:
            ref = _github_target(rest[2:], context)
            if ref is not None:
                verb = f"gh pr {rest[1]}"
                detections.append(
                    Detection(KIND_ACTED, ref, "gh-pr-act", verb, 0, act=_GH_ACTS[rest[1]])
                )
        elif name == "glab" and len(rest) >= 2 and rest[0] == "mr" and rest[1] in _GLAB_ACTS:
            ref = _gitlab_target(rest[2:], context)
            if ref is not None:
                verb = f"glab mr {rest[1]}"
                detections.append(
                    Detection(KIND_ACTED, ref, "glab-mr-act", verb, 0, act=_GLAB_ACTS[rest[1]])
                )
        elif name in ("gh", "glab") and rest[:1] == ["api"]:
            call = _cli_api_call(rest[1:])
            if call is not None:
                detections.extend(_api_detections(call, f"{name} api", result, context))
        elif name == "curl":
            call = _curl_call(rest)
            if call is not None:
                detections.extend(_api_detections(call, "curl", result, context))
        elif name == "git" and "push" in rest[:3]:
            detections.extend(_git_push_hints(result))
    if detections:
        return _dedupe(detections)
    shape = _unreachable_create_shape(command)
    if shape is not None and not any(_is_create_stage(words) for words in stages):
        # A create ran where the tokeniser cannot see it. Any bare URL on stdout may be
        # its own, and which one it is cannot be established — so the answer is the
        # design's "possibly opened", with every candidate carried as evidence.
        candidates = [
            ref
            for line in result.stdout.splitlines()
            for ref in [_line_ref(line, context, ("github", "gitlab", "gitea", "azure"))]
            if ref is not None and not _has_fragment_or_query(line)
        ]
        if candidates:
            return [
                Detection(
                    KIND_UNKNOWN,
                    candidates[0],
                    "create-unreachable",
                    "gh pr create",
                    0,
                    reason=(
                        f"a create ran inside {shape}, so this call's output cannot be "
                        "attributed to it: possibly opened by this call"
                    ),
                    hint={"candidates": [ref.to_payload() for ref in candidates]},
                )
            ]
    # The script rule needs an ABSENCE: no forge CLI ran at all. ``gh pr view
    # --json url`` and its friends print PR URLs as a READ.
    if not any(name in _FORGE_CLIS for name in names) and any(
        _is_script_stage(words) for words in stages
    ):
        ref = _line_ref(_last_line(result.stdout), context, ("github", "gitlab", "gitea"))
        if ref is not None:
            return [
                Detection(
                    KIND_UNKNOWN,
                    ref,
                    "script-stdout",
                    None,
                    0,
                    reason=(
                        "a script printed this URL as its last line: possibly opened by "
                        "this call"
                    ),
                )
            ]
    return []


def _create_by_cli(
    result: BashResult,
    context: HostContext,
    verb: str,
    forge: str,
    *,
    stages: list[list[str]],
    index: int,
) -> list[Detection]:
    """The URL a create CLI's own stage printed, or the honest `unknown` instead.

    WHY NOT "THE LAST LINE OF STDOUT", twice over. The bash tool reports ONE stdout for
    the whole command, so every stage's output lands in the same buffer. Review round 1
    (F1) caught the last line being read as the creation when a LATER stage printed a
    different URL; round 2 (Q5) caught the same class from the other side — a later
    script, a file read, or an EARLIER forge act whose line was taken instead. The rule
    is now fail-closed and positional:

    1. **A create prints a BARE url**: no fragment and no query. ``#issuecomment-…``,
       ``#note_…`` and ``#diff-…`` are other commands' shapes.
    2. **Only stages that can print a bare url of their own are candidates for the
       blame.** Filters with no file operand (``| tail -1``, ``| tee x``), shell
       builtins and ``git`` cannot print one; a script, an interpreter with a script, a
       file reader (``cat summary.txt``), a stage whose ARGUMENTS name a url, and every
       forge-CLI stage that is not a comment/note verb can. A ``gh pr comment`` /
       ``glab mr note`` stage prints its url WITH a fragment, so it is not a bare-url
       printer at all — which is what keeps ``gh pr create -f && gh pr comment 5``
       attributable to the create (QA round 2, row 3a).
    3. **A stage whose own arguments name the ref it printed is eliminated**, and the
       candidate line carrying that ref is explained by it. ``gh pr create -f && gh pr
       view 5 --json url`` therefore attributes the FIRST line to the create, because
       the second line is named by the view's arguments.
    4. If exactly one printer is left and exactly one candidate is unexplained, that
       candidate is the create's → ``opened``. If every remaining printer is a CREATE
       and the candidates number the same, they map in stage order (three creates in
       one command, QA round 2 Q6 — dropping the middle rows silently was the finding).
       Anything else is `unknown` ("possibly opened"), with the candidate lines carried
       as evidence, never a chosen first or last line.
    """
    stem = _rule_stem(verb)
    candidates = [
        ref
        for line in result.stdout.splitlines()
        for ref in [_line_ref(line, context, (forge,))]
        if ref is not None and not _has_fragment_or_query(line)
    ]
    if not candidates:
        # The CLI ran and printed a url of its own, but not in the shape a create prints
        # (a fragment, a comment link, a table). Same answer as before: unshaped.
        loose = _last_url_ref(result.stdout, context, (forge,))
        if loose is None:
            return []
        return [
            Detection(
                KIND_UNKNOWN,
                loose,
                f"{stem}-unshaped",
                verb,
                0,
                reason=f"{verb} ran, but its URL is not in the CLI's own stdout shape",
            )
        ]
    printers = [
        item
        for item, words in enumerate(stages)
        if _url_printer_reason(words, context, forge) is not None
    ]
    explained: set[str] = set()
    unexplained: list[int] = []
    for item in printers:
        named = _named_keys(stages[item], context)
        keys = {ref.key for ref in candidates}
        if item != index and named and named <= keys:
            # Its own arguments name the refs it printed, so this stage accounts for those
            # lines and cannot be the source of the others.
            explained |= named
            continue
        unexplained.append(item)
    remaining = [ref for ref in candidates if ref.key not in explained]
    if unexplained == [index] and len(remaining) == 1:
        return [Detection(KIND_OPENED, remaining[0], f"{stem}-stdout", verb, 0)]
    if unexplained and all(_is_create_stage(stages[item]) for item in unexplained):
        if len(remaining) == len(unexplained):
            return [Detection(KIND_OPENED, ref, f"{stem}-stdout", verb, 0) for ref in remaining]
    reason = _unattributed_reason(stages, unexplained, index, verb)
    # ``remaining`` can be empty when every candidate line is named by another stage's
    # arguments. The row is still recorded — the call did run a create — so it reports the
    # first candidate it saw rather than dropping the fact.
    subject = remaining[0] if remaining else candidates[0]
    return [
        Detection(
            KIND_UNKNOWN,
            subject,
            f"{stem}-unattributed",
            verb,
            0,
            reason=reason,
            hint={"candidates": [ref.to_payload() for ref in candidates]},
        )
    ]


def _unreachable_create_shape(command: str) -> str | None:
    """A create the tokeniser cannot see: inside ``$( )``/backticks or a loop body.

    Both are EXECUTED shell text, unlike a heredoc body fed to python — which is why the
    heredoc false positive this module already tests stays refused. The answer for one of
    these is never ``opened`` (nobody can tell which iteration's URL landed on stdout), and
    the design's rule for a create whose evidence is out of reach is ``unknown``,
    "possibly opened", rather than silence.
    """
    for match in _SUBSTITUTION.finditer(command):
        body = match.group("sub") or match.group("tick") or ""
        if _CREATE_INVOCATION.search(body):
            return "a command substitution"
    for match in _LOOP_BODY.finditer(command):
        if _CREATE_INVOCATION.search(match.group("loop") or ""):
            return "a loop body"
    return None


def _unattributed_reason(
    stages: list[list[str]], unexplained: list[int], index: int, verb: str
) -> str:
    """Why this call's output could not be attributed, naming the stage that could fake it."""
    others = [item for item in unexplained if item != index]
    if others:
        names = ", ".join(_stage_label(stages[item]) for item in others[:3])
        return (
            f"{verb} ran, but other stages of the same command could print a URL of "
            f"their own ({names}), so this call's output cannot be attributed to the create"
        )
    return f"{verb} ran, and its stdout carries more than one URL that could be its own"


def _stage_label(words: list[str]) -> str:
    """A short human label for a stage: ``gh pr view``, ``cat summary.txt``."""
    return " ".join(words[:3]) if words else "?"


def _is_create_stage(words: list[str]) -> bool:
    name = command_word(words)
    rest = words[1:]
    if name == "gh":
        return len(rest) >= 2 and rest[0] == "pr" and rest[1] in _GH_CREATE
    if name == "glab":
        return len(rest) >= 2 and rest[0] == "mr" and rest[1] in _GLAB_CREATE
    if name == "tea":
        return len(rest) >= 2 and rest[0] in _TEA_PULLS and rest[1] in _TEA_CREATE
    if name == "az":
        return rest[:3] == ["repos", "pr", "create"]
    return False


#: Commands that print their arguments, and the metacharacters that make an argument
#: something other than literal text.
_ECHO_COMMANDS = frozenset({"echo", "printf"})
_EXPANSION = re.compile(r"[$`*?\[]")


def _is_literal_echo(words: list[str]) -> bool:
    """``echo done`` prints ``done``; ``echo $URL`` prints whatever the variable holds."""
    if command_word(words) not in _ECHO_COMMANDS:
        return False
    return not any(_EXPANSION.search(word) for word in words[1:])


def _named_keys(words: list[str], context: HostContext) -> set[str]:
    """The refs a forge stage's own ARGUMENTS name, through the same resolvers its
    detection uses — so "the ref this stage is about" has one definition in the module."""
    name = command_word(words)
    rest = words[1:]
    ref: Ref | None = None
    if name == "gh" and len(rest) >= 2 and rest[0] == "pr":
        ref = _github_target(rest[2:], context)
    elif name == "glab" and len(rest) >= 2 and rest[0] == "mr":
        ref = _gitlab_target(rest[2:], context)
    return {ref.key} if ref is not None else set()


def _url_printer_reason(words: list[str], context: HostContext, forge: str) -> str | None:
    """Why this stage could print a BARE url of its own, or ``None`` when it cannot.

    The ``None`` answers are the load-bearing ones, because they are what keeps a real
    create attributable: a shell builtin, a filter with no file operand, and ``git``
    cannot conjure a url line out of nothing. Everything else — a script, an
    interpreter, a file reader, an unrecognised command, a forge CLI, a stage whose
    arguments name a url — can, and therefore makes the command ambiguous.
    """
    name = command_word(words)
    if not name:
        return None
    if name in _FORGE_CLIS:
        rest = words[1:]
        if name in ("gh", "glab") and len(rest) >= 2:
            verb = tuple(rest[:2])
            if verb in _FRAGMENT_ONLY_VERBS:
                # A comment/note prints its own url WITH a fragment, so it can never be
                # mistaken for the create's bare line.
                return None
        return f"the forge CLI '{name}'"
    if _is_script_stage(words):
        return f"the script '{os.path.basename(words[0])}'"
    for word in words[1:]:
        if any(ref.forge == forge for ref in iter_urls(word, context)):
            return f"'{name}' names a URL in its arguments"
    if _is_literal_echo(words):
        # An echo of LITERAL text prints exactly that text, so it can only add a url line
        # if one of its own arguments is one — which the rule above already caught. This
        # is what keeps the everyday ``gh pr create -f && echo done`` attributable.
        return None
    if name in _PASS_THROUGH_ALWAYS:
        return None
    if name in _PASS_THROUGH_COMMANDS and not _has_file_operand(words):
        # A filter with no file operand prints only what it received on stdin, which is
        # this same stdout — it cannot add a line the create did not produce.
        return None
    if name in _BLIND_COMMANDS:
        return None
    return f"the command '{name}'"


#: A shell redirection as a token: ``2>&1``, ``>/dev/null``, ``&>log``. It names where
#: output goes, never an input file, so it must not read as a filter's operand.
_REDIRECT = re.compile(r"^(?:\d*[<>]|&>)")


def _has_file_operand(words: list[str]) -> bool:
    """Does a filter stage read a FILE (as opposed to its stdin)?

    ``2>&1`` is a redirection, not an operand: ``gh pr create -f 2>&1 | tail -1`` is the
    create's own stdout through a filter, and reading the redirect as a filename made the
    tail look like a reader of arbitrary files (and so made the whole command ambiguous).
    """
    for word in words[1:]:
        if word.startswith("-") or _REDIRECT.match(word) or word.isdigit():
            # ``tail -n 1`` / ``tail 1``: the number is the COUNT, not a file.
            continue
        return True
    return False


def _has_fragment_or_query(line: str) -> bool:
    """Does this stdout line carry a URL FRAGMENT or query? Then it is another's shape."""
    return "#" in line or "?" in line


def _later_stage_names_a_url(
    stages: list[list[str]], index: int, context: HostContext, forge: str
) -> bool:
    """Does any stage AFTER ``index`` name a same-forge URL in its arguments?"""
    for words in stages[index + 1 :]:
        for word in words[1:]:
            for ref in iter_urls(word, context):
                if ref.forge == forge:
                    return True
    return False


def _create_by_az(
    result: BashResult,
    context: HostContext,
    *,
    stages: list[list[str]],
    index: int,
) -> list[Detection]:
    """``az repos pr create``: its own JSON document, which aggregation cannot fake.

    Unlike the CLI rules there is no line-attribution problem to solve: the rule parses
    the WHOLE stdout as one JSON object, so a compound command whose later stage prints
    anything at all fails the parse and yields nothing. The stage list is taken only so
    that a later stage naming a URL is refused in the same way the CLI rules refuse it —
    ``azure`` hosts are detect-and-link, so a wrong `opened` here would be unrecoverable
    by a fetch (there is none).
    """
    if _later_stage_names_a_url(stages, index, context, "azure"):
        return []
    value = _json_object(result.stdout)
    if not isinstance(value, dict):
        return []
    number = value.get("pullRequestId")
    repository = value.get("repository")
    web_url = repository.get("webUrl") if isinstance(repository, dict) else None
    if not isinstance(number, int) or not isinstance(web_url, str):
        return []
    ref = parse_url(f"{web_url.rstrip('/')}/pullrequest/{number}", context)
    if ref is None:
        return []
    return [Detection(KIND_OPENED, ref, "az-repos-pr-create-json", "az repos pr create", 0)]


def _dedupe(detections: list[Detection]) -> list[Detection]:
    """One detection per ``(kind, ref, act)``. A real create also wins over its own unknown."""
    opened = {d.ref.key for d in detections if d.kind == KIND_OPENED and d.ref is not None}
    out: list[Detection] = []
    seen: set[tuple[str, str, str | None, str]] = set()
    for item in detections:
        key = item.ref.key if item.ref is not None else str(item.hint)
        if item.kind == KIND_UNKNOWN and key in opened:
            continue
        identity = (item.kind, key, item.act, item.rule)
        if identity in seen:
            continue
        seen.add(identity)
        out.append(item)
    return out


# ---------------------------------------------------------------------------
# MCP rules
# ---------------------------------------------------------------------------

_GITHUB_MCP_HOSTS = frozenset({"api.githubcopilot.com", "github.com", "api.github.com"})
_GITHUB_MCP_PACKAGES = ("github-mcp-server", "server-github")


def _server_for(tool_name: str, servers: Iterable[McpServer]) -> tuple[McpServer, str] | None:
    # Longest prefix first: server ``git`` must not claim ``mcp__gitlab_*``.
    for server in sorted(servers, key=lambda s: len(s.prefix), reverse=True):
        if tool_name.startswith(server.prefix):
            return server, tool_name[len(server.prefix) :]
    return None


def detect_mcp(
    tool_name: str,
    args: Mapping[str, Any],
    result_text: str,
    context: HostContext,
    servers: Iterable[McpServer],
) -> list[Detection]:
    found = _server_for(tool_name, servers)
    if found is None:
        return []
    server, tool = found
    if server.host and (
        KNOWN_PUBLIC_HOSTS.get(server.host) == "gitlab" or server.host in context.gitlab_hosts
    ):
        return _gitlab_mcp(tool, args, result_text, context, server.host)
    if server.host in _GITHUB_MCP_HOSTS or any(p in server.launch for p in _GITHUB_MCP_PACKAGES):
        return _github_mcp(tool, args, result_text, context)
    return []


def _gitlab_mcp(
    tool: str, args: Mapping[str, Any], text: str, context: HostContext, host: str
) -> list[Detection]:
    verb = f"mcp {tool}"
    iid = args.get("merge_request_iid")
    if tool == "save_merge_request" and iid in (None, ""):
        ref = _shaped_ref(_json_object(text), context)
        if ref is not None and ref.forge == "gitlab":
            return [Detection(KIND_OPENED, ref, "gitlab-mcp-create", verb)]
        loose = _last_url_ref(text, context, ("gitlab",))
        if loose is None:
            return []
        return [
            Detection(
                KIND_UNKNOWN,
                loose,
                "gitlab-mcp-create-unshaped",
                verb,
                reason="save_merge_request created, but its result lacks a web_url + iid pair",
            )
        ]
    act = {
        "save_merge_request": "edit",
        "save_note": "comment",
        "save_merge_request_review": "review",
        "accept_merge_request": "merge",
    }.get(tool)
    if act is None:
        return []
    ref = _gitlab_mcp_target(args, text, context, host)
    if ref is None:
        return []
    return [Detection(KIND_ACTED, ref, "gitlab-mcp-act", verb, act=act)]


def _gitlab_mcp_target(
    args: Mapping[str, Any], text: str, context: HostContext, host: str
) -> Ref | None:
    """The MR an ACTED call targeted, from the ARGUMENTS first.

    The arguments come first because they name the target the operator chose, while
    a result's ``web_url`` is whatever the server happened to return — and for
    ``save_merge_request`` those differ in exactly the case that matters: an UPDATE
    carries the iid it edited, and the result may echo a different merge request.
    A numeric ``project_id`` names no path, so no URL can be built from it; that
    case falls through to the result and then to nothing, never to a guess.
    """
    url = args.get("url")
    if isinstance(url, str):
        ref = parse_url(url, context)
        if ref is not None and ref.forge == "gitlab":
            return ref
    project = args.get("project_id")
    iid = args.get("merge_request_iid")
    if isinstance(project, str) and "/" in project and iid not in (None, ""):
        try:
            number = int(str(iid))
        except ValueError:
            number = 0
        if number > 0:
            built = parse_url(
                f"https://{host}/{project.strip('/')}/-/merge_requests/{number}", context
            )
            if built is not None:
                return built
    shaped = _shaped_ref(_json_object(text), context)
    if shaped is not None and shaped.forge == "gitlab":
        return shaped
    return None


def _github_mcp(
    tool: str, args: Mapping[str, Any], text: str, context: HostContext
) -> list[Detection]:
    verb = f"mcp {tool}"
    if tool == "create_pull_request":
        ref = _shaped_ref(_json_object(text), context)
        if ref is not None and ref.forge == "github":
            return [Detection(KIND_OPENED, ref, "github-mcp-create", verb, unverified=True)]
        return []
    act = {
        "merge_pull_request": "merge",
        "create_pull_request_review": "review",
        "submit_pending_pull_request_review": "review",
        "pull_request_review_write": "review",
        "update_pull_request": "edit",
        "add_issue_comment": "comment",
    }.get(tool)
    if act is None:
        return []
    owner, repo = args.get("owner"), args.get("repo")
    number = args.get("pullNumber", args.get("pull_number", args.get("issue_number")))
    if not isinstance(owner, str) or not isinstance(repo, str):
        return []
    try:
        n = int(str(number))
    except ValueError:
        return []
    ref = parse_url(f"https://github.com/{owner}/{repo}/pull/{n}", context)
    if ref is None:
        return []
    rule = "github-mcp-issue-comment" if tool == "add_issue_comment" else "github-mcp-act"
    return [Detection(KIND_ACTED, ref, rule, verb, act=act, unverified=True)]


# ---------------------------------------------------------------------------
# Entry point and MCP server loading
# ---------------------------------------------------------------------------


def detect_tool_result(
    tool_name: str,
    args: Mapping[str, Any],
    result_text: str,
    *,
    is_error: bool,
    context: HostContext,
    mcp_servers: Iterable[McpServer] = (),
) -> list[Detection]:
    """Every detection for one tool call. Pure. Never raises on malformed input."""
    try:
        if tool_name == "bash":
            command = args.get("command")
            if not isinstance(command, str):
                return []
            # A bash call with a non-zero exit is ``is_error`` here. The exit
            # code is re-read from the text anyway, which is the same answer.
            return detect_bash(command, result_text, context)
        if is_error or not tool_name.startswith("mcp__"):
            return []
        return detect_mcp(tool_name, args, result_text, context, mcp_servers)
    except Exception:  # noqa: BLE001 - classification must never break a caller
        logger.debug("code-request detection failed for %s", tool_name, exc_info=True)
        return []


_MCP_CACHE: dict[str, tuple[float, tuple[McpServer, ...]]] = {}
_MCP_LOCK = threading.Lock()
_MCP_TTL_S = 60.0


def load_mcp_servers(cwd: str | None) -> tuple[McpServer, ...]:
    """Every configured MCP server's identity facts. Blocking, cached, and never raises.

    Reads the same config files the MCP manager does (``load_all_mcp_configs``).
    The prefix comes from the bridge's own name minting, so the two cannot drift.
    """
    key = cwd or ""
    now = time.monotonic()
    with _MCP_LOCK:
        cached = _MCP_CACHE.get(key)
        if cached is not None and now - cached[0] < _MCP_TTL_S:
            return cached[1]
    servers: list[McpServer] = []
    try:
        from local_operator.mcp.config import load_all_mcp_configs
        from local_operator.mcp.tool_bridge import create_mcp_tool_name

        configs, _sources = load_all_mcp_configs(cwd or os.getcwd())
        for name, config in configs.items():
            probe = create_mcp_tool_name(name, "probe")
            prefix = probe[: -len("probe")]
            url = getattr(config, "url", "") or ""
            host = urlsplit(url).netloc.lower().split("@")[-1].split(":")[0] if url else ""
            launch = " ".join(
                [str(getattr(config, "command", "") or "")]
                + [str(a) for a in (getattr(config, "args", None) or [])]
            ).strip()
            servers.append(McpServer(prefix=prefix, host=host, launch=launch))
    except Exception:  # noqa: BLE001 - an unreadable MCP config detects nothing
        logger.debug("could not load MCP configs for code-request detection", exc_info=True)
    result = tuple(servers)
    with _MCP_LOCK:
        _MCP_CACHE[key] = (now, result)
        if len(_MCP_CACHE) > 64:
            _MCP_CACHE.clear()
            _MCP_CACHE[key] = (now, result)
    return result


def iter_detected_refs(detections: Iterable[Detection]) -> Iterator[Ref]:
    for item in detections:
        if item.ref is not None:
            yield item.ref


def _reset_for_tests() -> None:
    with _MCP_LOCK:
        _MCP_CACHE.clear()


__all__ = [
    "AMBIGUOUS_ACT_RULES",
    "BashResult",
    "Detection",
    "KIND_ACTED",
    "KIND_HINT",
    "KIND_OPENED",
    "KIND_UNKNOWN",
    "McpServer",
    "command_word",
    "could_matter",
    "detect_bash",
    "detect_mcp",
    "detect_tool_result",
    "load_mcp_servers",
    "parse_bash_result",
    "split_stages",
    "stage_words",
]
