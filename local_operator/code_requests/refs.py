"""Parse a forge code-request reference (PR/MR URL or qualified ref) and say which host it is.

WHY THIS EXISTS. A session's code requests are tracked from the URLs and refs it
sees and creates. Before anything else can happen, a string such as
``https://git.example.com/acme/api/pull/12`` has to become a typed :class:`Ref`, and
the parser must say how sure it is. A wrong forge family is worse than "unknown":
the later adapters (PR1b) would send a GitHub API call to a Gitea host, and the UI
would draw the wrong glyph and state.

HOST IDENTIFICATION NEVER TRUSTS THE HOSTNAME ALONE (design §B.3). Evidence is
weighed in this order:

1. **Path shape → candidate family.** ``/-/merge_requests/N`` is GitLab and only
   GitLab. ``/pulls/N`` is the Gitea family (Gitea, Forgejo, Codeberg).
   ``/pull-requests/N`` is Bitbucket, ``/_git/<repo>/pullrequest/N`` is Azure
   DevOps, and ``/c/<project>/+/N`` is Gerrit. ``/pull/N`` is GitHub or GitHub
   Enterprise.
2. **Known public hosts confirm** the family (``github.com``, ``gitlab.com``,
   ``codeberg.org``, ``bitbucket.org``, ``dev.azure.com``). A known host whose
   family CONTRADICTS the path shape (``gitlab.com/o/r/pull/3``) is not a ref.
3. **Operator-configured hosts confirm** self-hosted instances: the hosts gh is
   logged into (``~/.config/gh/hosts.yml``), glab's host list, and tea's logins.
4. **The session cwd's git remotes** confirm that a host is a forge the operator
   actually works with.
5. Anything still unconfirmed becomes **detect-and-link**. The row exists and
   opens in the browser, but ``full`` is False and ``reason`` says why.

A ``/pull/N`` link on an unconfirmed host is the main ambiguous case. A GHES
instance the operator never logged into looks exactly like one.

QUALIFIED REFS. ``owner/repo#N`` is ambiguous between an issue and a PR, because
GitHub numbers both from one sequence. It is therefore always detect-and-link with
that reason. ``group/project!N`` is GitLab's MR-only notation and is unambiguous
about its kind. Its host comes from a cwd remote that carries that project path,
else from the operator's single configured GitLab host. Bare ``#N`` is NEVER
parsed: it is ambiguous, and ``#1`` in a list is prose.

CONSTRAINTS.

* **No network, ever.** Everything here reads local files: gh/glab/tea config,
  plus ``git config`` for remotes. That is why it can run inside the post-tool
  hook and the transcript scanner.
* **Context loading is cached and bounded.** :func:`load_host_context` reads a few
  small files and runs one ``git config`` per cwd under a 2 s timeout. Its result
  is cached for :data:`_CONTEXT_TTL_S`, so a burst of tool results costs one read.
* **Tokens are never read.** Only host NAMES are taken from the CLI configs. The
  gh/glab files also hold tokens, and this module parses keys line by line so a
  token value is never even materialised into a parsed mapping.
"""

from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator
from urllib.parse import unquote, urlsplit

#: Forges whose adapters PR1 builds fully (summary, comments, CI). Every other
#: family is detect-and-link: the row exists and opens in the browser, with no
#: state fetched.
FULL_FORGES = frozenset({"github", "gitlab"})

FORGES = ("github", "gitlab", "gitea", "bitbucket", "azure", "gerrit")

#: Public hosts whose family is a FACT. A path shape that contradicts one of these
#: is rejected outright rather than guessed: ``gitlab.com/o/r/pull/3`` is not a PR.
KNOWN_PUBLIC_HOSTS: dict[str, str] = {
    "github.com": "github",
    "www.github.com": "github",
    "gitlab.com": "gitlab",
    "codeberg.org": "gitea",
    "gitea.com": "gitea",
    "bitbucket.org": "bitbucket",
    "dev.azure.com": "azure",
}

#: How long a loaded :class:`HostContext` is reused for one cwd. Long enough that a
#: turn's burst of tool results reads the configs once. Short enough that a fresh
#: ``gh auth login`` or a new remote is seen within a minute.
_CONTEXT_TTL_S = 60.0

#: Bound on the one subprocess this module runs (``git config`` for remotes). It is
#: local and normally takes milliseconds. The bound is for a wedged filesystem,
#: never for the common case.
_GIT_TIMEOUT_S = 2.0


@dataclass(frozen=True)
class Ref:
    """One code request: a GitHub PR, GitLab MR, or a detect-and-link sibling.

    ``project`` is the forge's own path for the repository, such as
    ``owner/repo``, ``group/sub/project``, Azure's ``org/project/_git/repo``, or a
    Gerrit project. ``number`` is the PR number or MR iid. ``url`` is the
    CANONICAL link, with trailing ``/files``, ``#issuecomment-…`` and queries
    removed, so two spellings of one PR collapse into one row through :attr:`key`.

    ``full`` says whether a PR1 adapter exists for this forge AND the host was
    confirmed. ``reason`` is set whenever ``full`` is False and explains why, in a
    sentence a UI can show.
    """

    forge: str
    host: str
    project: str
    number: int
    url: str
    full: bool
    reason: str | None = None

    @property
    def key(self) -> str:
        """Stable identity across spellings: ``host/project#N`` or ``host/project!N``.

        GitLab keeps its ``!`` because a GitLab project can have issue #3 and MR !3
        at the same time. The key must never merge the two.
        """
        sep = "!" if self.forge == "gitlab" else "#"
        return f"{self.host}/{self.project}{sep}{self.number}"

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "key": self.key,
            "forge": self.forge,
            "host": self.host,
            "project": self.project,
            "number": self.number,
            "url": self.url,
            "full": self.full,
        }
        if self.reason:
            payload["reason"] = self.reason
        return payload

    @staticmethod
    def from_payload(raw: object) -> "Ref | None":
        """Rebuild a :class:`Ref` from :meth:`to_payload` output, or ``None`` if malformed."""
        if not isinstance(raw, dict):
            return None
        try:
            forge = str(raw["forge"])
            host = str(raw["host"])
            project = str(raw["project"])
            number = int(raw["number"])
            url = str(raw["url"])
        except (KeyError, TypeError, ValueError):
            return None
        if forge not in FORGES or not host or not project or number <= 0:
            return None
        reason = raw.get("reason")
        return Ref(
            forge=forge,
            host=host,
            project=project,
            number=number,
            url=url,
            full=bool(raw.get("full")),
            reason=str(reason) if isinstance(reason, str) and reason else None,
        )


@dataclass(frozen=True)
class Remote:
    """One git remote of the session's cwd, reduced to ``(host, project path)``."""

    name: str
    host: str
    project: str


@dataclass(frozen=True)
class HostContext:
    """What this machine says about forge hosts. Every field is local data.

    The empty default is a valid context: it confirms only the public hosts. Tests
    and the pure transcript scan use it when no machine state should leak in.
    """

    github_hosts: frozenset[str] = frozenset()
    gitlab_hosts: frozenset[str] = frozenset()
    gitea_hosts: frozenset[str] = frozenset()
    remotes: tuple[Remote, ...] = ()

    def confirms(self, host: str, forge: str) -> bool:
        """Whether local configuration confirms ``host`` as an instance of ``forge``."""
        if KNOWN_PUBLIC_HOSTS.get(host) == forge:
            return True
        configured = {
            "github": self.github_hosts,
            "gitlab": self.gitlab_hosts,
            "gitea": self.gitea_hosts,
        }.get(forge, frozenset())
        if host in configured:
            return True
        # A cwd remote on this host proves it is a forge the operator pushes to.
        # The path shape already chose the family, so the remote only needs to
        # confirm the host is real. It cannot contradict the family.
        return any(remote.host == host for remote in self.remotes)


EMPTY_CONTEXT = HostContext()

# ---------------------------------------------------------------------------
# URL parsing
# ---------------------------------------------------------------------------

#: Candidate URL spans inside free text. Deliberately loose. The structured parse
#: below is the judge, and this only finds things worth judging. It stops at
#: whitespace, quotes, angle brackets, backticks and closing brackets, which is
#: what ends a URL in markdown, JSON and terminal output.
_URL_IN_TEXT = re.compile(r"https?://[^\s\"'<>`)\]}|\\]+", re.IGNORECASE)

_NUM = r"(?P<n>[1-9]\d{0,9})"
#: What may follow the number: nothing, a sub-page (``/files``, ``/diffs``), a
#: query, or a fragment. ``/pull/12abc`` is not PR 12.
_TAIL = r"(?:[/?#].*)?$"

_GITLAB_PATH = re.compile(rf"^/(?P<project>[^/].*?)/-/merge_requests/{_NUM}{_TAIL}")
_GITEA_PATH = re.compile(rf"^/(?P<owner>[^/]+)/(?P<repo>[^/]+)/pulls/{_NUM}{_TAIL}")
_GITHUB_PATH = re.compile(rf"^/(?P<owner>[^/]+)/(?P<repo>[^/]+)/pull/{_NUM}{_TAIL}")
_BITBUCKET_CLOUD_PATH = re.compile(
    rf"^/(?P<owner>[^/]+)/(?P<repo>[^/]+)/pull-requests/{_NUM}{_TAIL}"
)
_BITBUCKET_SERVER_PATH = re.compile(
    rf"^(?P<prefix>/[^?#]*?)?/projects/(?P<owner>[^/]+)/repos/(?P<repo>[^/]+)"
    rf"/pull-requests/{_NUM}{_TAIL}",
    re.IGNORECASE,
)
_AZURE_PATH = re.compile(
    rf"^/(?P<project>(?:[^/]+/)*?[^/]+/_git/[^/]+)/pullrequest/{_NUM}{_TAIL}", re.IGNORECASE
)
_GERRIT_PATH = re.compile(rf"^(?:/#)?/c/(?P<project>.+?)/\+/{_NUM}{_TAIL}")

#: Owner/repo segment as the forges allow it. Also used to keep a qualified ref
#: from matching a file path such as ``src/app.py#12``.
_SEGMENT = re.compile(r"^[A-Za-z0-9_.][A-Za-z0-9_.-]*$")
_FILE_SUFFIXES = (
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".md",
    ".json",
    ".yml",
    ".yaml",
    ".toml",
    ".txt",
    ".html",
    ".css",
    ".rs",
    ".go",
    ".sh",
    ".rb",
    ".java",
    ".kt",
    ".swift",
    ".c",
    ".h",
    ".cpp",
)


def _canonical_host(netloc: str) -> str:
    host = netloc.rsplit("@", 1)[-1].lower()
    # A port is part of a self-hosted host's identity (``git.corp:8443``), so it
    # stays. The default ports are dropped so two spellings collapse.
    for default in (":443", ":80"):
        if host.endswith(default):
            host = host[: -len(default)]
    return host


def _family_by_shape(host: str, path: str) -> tuple[str, str, int] | None:
    """``(forge, project, number)`` from the path shape alone, or ``None``.

    The ORDER is load-bearing. GitLab's ``/-/`` separator is checked first because
    a GitLab project path may itself contain ``pull`` or ``pulls`` segments.
    Bitbucket Server's ``/projects/…/repos/`` comes before the generic Bitbucket
    Cloud shape, which it would otherwise match with the wrong owner. Azure
    is matched by its ``_git`` marker, and Gerrit by ``/c/…/+/``.
    """
    match = _GITLAB_PATH.match(path)
    if match:
        return "gitlab", match.group("project"), int(match.group("n"))
    match = _AZURE_PATH.match(path)
    if match:
        return "azure", match.group("project"), int(match.group("n"))
    match = _GERRIT_PATH.match(path)
    if match:
        return "gerrit", match.group("project"), int(match.group("n"))
    match = _BITBUCKET_SERVER_PATH.match(path)
    if match:
        project = f"{match.group('owner')}/{match.group('repo')}"
        return "bitbucket", project, int(match.group("n"))
    match = _BITBUCKET_CLOUD_PATH.match(path)
    if match:
        return "bitbucket", f"{match.group('owner')}/{match.group('repo')}", int(match.group("n"))
    match = _GITEA_PATH.match(path)
    if match:
        return "gitea", f"{match.group('owner')}/{match.group('repo')}", int(match.group("n"))
    match = _GITHUB_PATH.match(path)
    if match:
        return "github", f"{match.group('owner')}/{match.group('repo')}", int(match.group("n"))
    return None


def _host_family(host: str) -> str | None:
    """The family a host's NAME pins, for the few names that pin one."""
    if host in KNOWN_PUBLIC_HOSTS:
        return KNOWN_PUBLIC_HOSTS[host]
    if host.endswith(".visualstudio.com"):
        return "azure"
    return None


def _canonical_url(forge: str, scheme: str, host: str, prefix: str, project: str, n: int) -> str:
    base = f"{scheme}://{host}{prefix}"
    if forge == "gitlab":
        return f"{base}/{project}/-/merge_requests/{n}"
    if forge == "gitea":
        return f"{base}/{project}/pulls/{n}"
    if forge == "bitbucket":
        owner, _, repo = project.partition("/")
        if host == "bitbucket.org":
            return f"{base}/{project}/pull-requests/{n}"
        return f"{base}/projects/{owner}/repos/{repo}/pull-requests/{n}"
    if forge == "azure":
        return f"{base}/{project}/pullrequest/{n}"
    if forge == "gerrit":
        return f"{base}/c/{project}/+/{n}"
    return f"{base}/{project}/pull/{n}"


def parse_url(url: str, context: HostContext = EMPTY_CONTEXT) -> Ref | None:
    """Parse ONE URL into a :class:`Ref`, or ``None`` when it is not a code request.

    Explicit negatives return ``None``: issue URLs, ``/pull/new/<branch>``,
    ``/-/merge_requests/new?…`` and compare pages all fail the numeric shape.
    So does a path shape that a known public host contradicts.
    """
    text = url.strip().rstrip(".,;:!?")
    try:
        parts = urlsplit(text)
    except ValueError:
        return None
    if parts.scheme.lower() not in ("http", "https") or not parts.netloc:
        return None
    host = _canonical_host(parts.netloc)
    path = unquote(parts.path)
    # Azure's legacy host and some Gerrit installs put the shape in the fragment
    # (``/#/c/…``), so the fragment is appended for the shape match only.
    shaped = path + (f"#{parts.fragment}" if path in ("", "/") and parts.fragment else "")
    if shaped.startswith("/#/"):
        shaped = shaped[2:]
    found = _family_by_shape(host, shaped)
    if found is None:
        return None
    forge, project, number = found
    project = project.strip("/")
    if forge == "bitbucket":
        # Bitbucket Server keys repositories by project KEY, which is
        # case-insensitive. Lower-casing keeps the two spellings one row.
        project = project if host == "bitbucket.org" else project.lower()
    pinned = _host_family(host)
    if pinned is not None and pinned != forge:
        # A host whose family is a fact contradicts the shape. Recognising it
        # would invent a code request: gitlab.com has no ``/pull/N`` pages.
        return None
    prefix = ""
    if forge == "bitbucket" and host != "bitbucket.org":
        server = _BITBUCKET_SERVER_PATH.match(shaped)
        prefix = (server.group("prefix") or "") if server else ""
    scheme = parts.scheme.lower()
    canonical = _canonical_url(forge, scheme, host, prefix, project, number)
    if forge not in FULL_FORGES:
        return Ref(
            forge,
            host,
            project,
            number,
            canonical,
            full=False,
            reason=f"{_FORGE_NAMES[forge]} is detect-and-link: the link opens, no state is fetched",
        )
    if forge == "gitlab" or context.confirms(host, forge):
        # GitLab's ``/-/merge_requests/N`` shape is unique to GitLab, so the
        # shape alone confirms the family on any host. The adapter (PR1b) still
        # needs a token for a self-hosted instance. That is a link-only
        # question, not an identification one.
        return Ref(forge, host, project, number, canonical, full=True)
    return Ref(
        forge,
        host,
        project,
        number,
        canonical,
        full=False,
        reason=(
            f"{host} has a GitHub-shaped link, but it is not github.com, a host the gh CLI "
            "is logged into, or a git remote of this session's directory"
        ),
    )


_FORGE_NAMES = {
    "github": "GitHub",
    "gitlab": "GitLab",
    "gitea": "Gitea/Forgejo",
    "bitbucket": "Bitbucket",
    "azure": "Azure DevOps",
    "gerrit": "Gerrit",
}


def iter_urls(text: str, context: HostContext = EMPTY_CONTEXT) -> Iterator[Ref]:
    """Every code-request URL in ``text``, in order. Repeats are kept: callers count them."""
    for match in _URL_IN_TEXT.finditer(text):
        ref = parse_url(match.group(0), context)
        if ref is not None:
            yield ref


# ---------------------------------------------------------------------------
# Qualified refs
# ---------------------------------------------------------------------------

#: ``owner/repo#N`` or ``group/sub/project!N``. The lookbehind refuses a match that
#: continues a URL, a path or a word (``https://x/o/r#3``, ``a/b/o/r#3`` inside a
#: longer path, ``foo.o/r#3``). Any ``://`` URL is handled by :func:`iter_urls`, never
#: here, and the scanner removes URL spans before it looks for qualified refs.
_QUALIFIED = re.compile(
    r"(?<![\w./:@-])(?P<project>[A-Za-z0-9_.][A-Za-z0-9_.-]*(?:/[A-Za-z0-9_.][A-Za-z0-9_.-]*)+)"
    r"(?P<sep>[#!])(?P<n>[1-9]\d{0,9})(?![\w])"
)


def _looks_like_file(project: str) -> bool:
    return project.lower().endswith(_FILE_SUFFIXES)


def parse_qualified(
    project: str, sep: str, number: int, context: HostContext = EMPTY_CONTEXT
) -> Ref | None:
    """Resolve one qualified ref to a :class:`Ref`, with ``full`` False and a reason when unsure."""
    if any(not _SEGMENT.match(part) for part in project.split("/")) or _looks_like_file(project):
        return None
    if sep == "!":
        host = _gitlab_host_for(project, context)
        if host is None:
            return None
        url = f"https://{host}/{project}/-/merge_requests/{number}"
        return Ref("gitlab", host, project, number, url, full=True)
    if project.count("/") != 1:
        # ``owner/repo#N`` is GitHub's (and Gitea's) notation, and both have
        # exactly two path segments. A deeper path is a GitLab project with an
        # issue number, or a file anchor. Neither is a PR.
        return None
    host = _github_host_for(project, context)
    url = f"https://{host}/{project}/pull/{number}"
    return Ref(
        "github",
        host,
        project,
        number,
        url,
        full=False,
        reason=(
            f"{project}#{number} could be an issue or a pull request: GitHub numbers both "
            "from one sequence, so it is linked, not tracked"
        ),
    )


def _remote_host_for(project: str, context: HostContext) -> str | None:
    for remote in context.remotes:
        if remote.project.lower() == project.lower():
            return remote.host
    return None


def _gitlab_host_for(project: str, context: HostContext) -> str | None:
    host = _remote_host_for(project, context)
    if host is not None:
        return host
    # No remote names this project, so the ref's home is decided in a fixed order:
    # gitlab.com first — the public instance is the default for a bare
    # ``group/project!N``, and an operator who is ALSO logged into a self-hosted
    # instance must not have their public refs attributed to it — then a single
    # configured self-hosted instance. Two or more self-hosted instances and no
    # remote means the ref names nothing: drop it rather than guess.
    if "gitlab.com" in context.gitlab_hosts or not context.gitlab_hosts:
        # No login recorded at all means the operator has only the public instance to
        # mean, so a bare ``group/project!N`` lands there rather than nowhere.
        return "gitlab.com"
    self_hosted = context.gitlab_hosts
    return next(iter(self_hosted)) if len(self_hosted) == 1 else None


def _github_host_for(project: str, context: HostContext) -> str:
    return _remote_host_for(project, context) or "github.com"


def iter_qualified(text: str, context: HostContext = EMPTY_CONTEXT) -> Iterator[Ref]:
    """Every qualified ref in ``text``, in order. URL spans must be removed by the caller."""
    for match in _QUALIFIED.finditer(text):
        ref = parse_qualified(
            match.group("project"), match.group("sep"), int(match.group("n")), context
        )
        if ref is not None:
            yield ref


def iter_refs(text: str, context: HostContext = EMPTY_CONTEXT) -> Iterator[Ref]:
    """URLs first, then qualified refs found outside those URL spans.

    Masking the URL spans is what stops ``https://github.com/o/r/pull/3#discussion``
    from also yielding a qualified ref for its tail.
    """
    if not text:
        return
    yield from iter_urls(text, context)
    if "#" not in text and "!" not in text:
        return
    masked = _URL_IN_TEXT.sub(lambda m: " " * len(m.group(0)), text)
    yield from iter_qualified(masked, context)


def parse_any(value: str, context: HostContext = EMPTY_CONTEXT) -> Ref | None:
    """One URL or qualified ref as the WHOLE string, such as a tool argument. ``None`` otherwise."""
    value = value.strip()
    if "://" in value:
        return parse_url(value, context)
    match = _QUALIFIED.fullmatch(value)
    if match is None:
        return None
    return parse_qualified(
        match.group("project"), match.group("sep"), int(match.group("n")), context
    )


# ---------------------------------------------------------------------------
# Host context loading (local files only)
# ---------------------------------------------------------------------------


def _top_level_yaml_keys(path: Path) -> list[str]:
    """Top-level mapping keys of a small YAML file, read line by line.

    gh's ``hosts.yml`` keys are host names, and the values beneath them hold
    tokens. A full YAML parse would bring those tokens into this process's
    objects for no reason, so only the unindented ``key:`` lines are read.
    """
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            lines = handle.read(256 * 1024).splitlines()
    except OSError:
        return []
    keys: list[str] = []
    for line in lines:
        if not line or line[0] in " \t#-":
            continue
        name, sep, _rest = line.partition(":")
        if sep and name.strip():
            keys.append(name.strip().strip("'\"").lower())
    return keys


def _glab_hosts(path: Path) -> list[str]:
    """Host keys under glab's ``hosts:`` block (one indentation level), never values."""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            lines = handle.read(256 * 1024).splitlines()
    except OSError:
        return []
    hosts: list[str] = []
    inside = False
    indent: int | None = None
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not line[0].isspace():
            inside = line.split(":", 1)[0].strip() == "hosts"
            indent = None
            continue
        if not inside:
            continue
        width = len(line) - len(line.lstrip())
        if indent is None:
            indent = width
        if width == indent and stripped.endswith(":"):
            hosts.append(stripped[:-1].strip().strip("'\"").lower())
    return hosts


_TEA_URL = re.compile(r"^\s*-?\s*url:\s*['\"]?(?P<url>https?://[^\s'\"]+)", re.MULTILINE)


def _tea_hosts(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")[: 256 * 1024]
    except OSError:
        return []
    return [_canonical_host(urlsplit(m.group("url")).netloc) for m in _TEA_URL.finditer(text)]


def _glab_config_paths(home: Path) -> Iterable[Path]:
    # glab moved its config across versions: macOS Application Support (1.5x+,
    # and what ``glab auth status`` reports on this platform), then XDG, then
    # the legacy dot-directory. All three are read, because only host NAMES are
    # wanted and a stale file can only add a confirmation, never a token.
    yield home / "Library" / "Application Support" / "glab-cli" / "config.yml"
    yield home / ".config" / "glab-cli" / "config.yml"


_SCP_REMOTE = re.compile(r"^(?:[^@/]+@)?(?P<host>[^:/]+):(?P<path>[^/].*)$")


def parse_remote_url(url: str) -> tuple[str, str] | None:
    """``(host, project)`` from a git remote URL: https, ssh:// or scp-like.

    The project keeps every path segment (GitLab subgroups) and drops ``.git``.
    Azure and Bitbucket Server remotes carry extra path furniture (``/scm/``,
    ``/_git/``). They still return a host, which is all :class:`HostContext`
    needs from them.
    """
    url = url.strip()
    if not url:
        return None
    if "://" in url:
        parts = urlsplit(url)
        if not parts.netloc:
            return None
        host = _canonical_host(parts.netloc)
        path = parts.path
        if parts.scheme in ("ssh", "git+ssh") and ":" in host:
            # ssh://git@host:22/owner/repo, where the port is not part of the web host.
            host = host.split(":", 1)[0]
    else:
        match = _SCP_REMOTE.match(url)
        if match is None:
            return None
        host, path = match.group("host").lower(), match.group("path")
    project = path.strip("/")
    if project.endswith(".git"):
        project = project[:-4]
    if not host or not project:
        return None
    return host, project


def _git_remotes(cwd: str) -> tuple[Remote, ...]:
    """The cwd's remotes via ``git config``. READ-ONLY, bounded, and silent on any failure.

    ``git config --get-regexp`` reads config files and nothing else. It never
    contacts a remote and never consults a credential helper, which is why it is
    safe here. The environment strips the prompt and askpass paths anyway, so a
    misconfigured system can never put a dialog on screen from a hook.
    """
    if not cwd or not os.path.isdir(cwd):
        return ()
    env = dict(os.environ)
    env.update({"GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": "/bin/false", "LC_ALL": "C"})
    try:
        proc = subprocess.run(
            ["git", "-C", cwd, "config", "--get-regexp", r"^remote\..*\.(push)?url$"],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            env=env,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    if proc.returncode != 0:
        return ()
    remotes: list[Remote] = []
    seen: set[tuple[str, str, str]] = set()
    for line in proc.stdout.splitlines():
        key, _, value = line.partition(" ")
        name = key[len("remote.") :].rsplit(".", 1)[0]
        parsed = parse_remote_url(value)
        if parsed is None:
            continue
        item = (name, parsed[0], parsed[1])
        if item in seen:
            continue
        seen.add(item)
        remotes.append(Remote(name=name, host=parsed[0], project=parsed[1]))
    return tuple(remotes)


@dataclass
class _CacheEntry:
    loaded_at: float
    context: HostContext


_CACHE: dict[tuple[str, str], _CacheEntry] = {}
_CACHE_LOCK = threading.Lock()


def load_host_context(cwd: str | None, *, home: Path | None = None) -> HostContext:
    """The machine's host context for ``cwd``. Blocking (file reads plus one git call).

    Call it from a worker thread when on an event loop. Results are cached per
    ``(home, cwd)`` for :data:`_CONTEXT_TTL_S`.
    """
    root = Path.home() if home is None else home
    key = (str(root), cwd or "")
    now = time.monotonic()
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached is not None and now - cached.loaded_at < _CONTEXT_TTL_S:
            return cached.context
    gh = {h for h in _top_level_yaml_keys(root / ".config" / "gh" / "hosts.yml")}
    glab: set[str] = set()
    for path in _glab_config_paths(root):
        glab.update(_glab_hosts(path))
    tea = set(_tea_hosts(root / ".config" / "tea" / "config.yml"))
    context = HostContext(
        github_hosts=frozenset(gh),
        gitlab_hosts=frozenset(glab),
        gitea_hosts=frozenset(tea),
        remotes=_git_remotes(cwd) if cwd else (),
    )
    with _CACHE_LOCK:
        _CACHE[key] = _CacheEntry(now, context)
        # Bounded by construction: a long-lived daemon serving many cwds must
        # not grow this without limit. A full cache is simply cleared, because
        # each entry is cheap to rebuild.
        if len(_CACHE) > 64:
            _CACHE.clear()
            _CACHE[key] = _CacheEntry(now, context)
    return context


def _reset_for_tests() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()


__all__ = [
    "EMPTY_CONTEXT",
    "FORGES",
    "FULL_FORGES",
    "HostContext",
    "KNOWN_PUBLIC_HOSTS",
    "Ref",
    "Remote",
    "iter_qualified",
    "iter_refs",
    "iter_urls",
    "load_host_context",
    "parse_any",
    "parse_qualified",
    "parse_remote_url",
    "parse_url",
]
