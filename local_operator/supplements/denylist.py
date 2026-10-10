"""The server-side sensitive-path denylist for turn supplements (memo §4.3).

ONE predicate, :func:`is_sensitive`: ``path -> rule id`` (``""`` = allowed). It runs in the
pre-filter BEFORE a path becomes a candidate, so nothing it denies is ever sent to a
decision vendor, journaled, listed, previewed or served by the file route (which re-checks
with this same function). A denylist gap is only visible at the data, so every rule has a
stable id and ``tests/unit/supplements/test_denylist.py`` pins the memo's S14 table as
``name -> expected rule``.

COMPOSITION, NOT A THIRD LIST. The credential vocabularies that already exist are imported
and unioned in, so a class added to ``references.py`` (the ``@``-reference gate) or
``browser_files.py`` (the upload gate) is denied here without an edit; a parity test fails
if this module ever stops being a superset of either. They are imported LAZILY because
``references`` pulls the whole tool layer (~0.9 s cold); the pre-filter runs on a worker
thread (``asyncio.to_thread``), so the first call pays that import off the event loop.

WHAT IS COMPUTED, NEVER LITERAL. The config directory (it holds ``secrets/store.db``,
``master.key`` and ``audit.log``) and the scratchpad root are read from their LIVE accessors
on every call (``paths.config_dir()``, ``LOCAL_OPERATOR_SCRATCHPAD``): a relocated
``LOCAL_OPERATOR_CONFIG_DIR`` must be as safe as the default, and a hard-coded
``~/.local-operator`` passes every non-relocated fixture while still leaking the secret
store under a relocation (memo round-2 S-R2-4).

BOTH FORMS. The rules are applied to the path AS WRITTEN and to its RESOLVED form (symlinks
followed): ``report.md -> ~/.ssh/id_rsa`` must be denied by its target, and a benign-named
target behind a sensitive-named link by its link. Either matching denies.

CASE-FOLDED, because APFS and NTFS are case-insensitive by default: ``PROD.ENV`` and ``.ENV``
are the same inode as the lowercase names (``references._sensitive_name`` measured this).
"""

import fnmatch
import os
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Final

#: The rule ids this module can return. Public so a test can assert the table is exhaustive
#: and a surface can log which class refused a path without the path.
RULE_CONFIG_DIR: Final = "config-dir"
RULE_SCRATCHPAD: Final = "scratchpad"
RULE_NAME: Final = "name"
RULE_SUFFIX: Final = "suffix"
RULE_NAME_PREFIX: Final = "name-prefix"
RULE_COMPONENT: Final = "component"
RULE_CREDENTIAL_PATTERN: Final = "credential-pattern"
RULE_DATABASE: Final = "database"
RULE_TOKEN_SECRET: Final = "token-or-secret-name"
RULE_COMPOSE: Final = "compose-file"
RULE_GH_HOSTS: Final = "gh-hosts"
RULE_DOCKER_CONFIG: Final = "docker-config"
RULE_GCLOUD: Final = "gcloud"
RULE_TERRAFORM: Final = "terraform"
RULE_TOOL_CONFIG: Final = "tool-credential-config"
RULE_BROWSER_STORE: Final = "browser-store"

#: Basenames (casefolded) of credential files the two composed lists do not name. Each is a
#: file whose whole purpose is to hold a secret, so a basename match needs no directory.
_EXTRA_NAMES: Final[dict[str, str]] = {
    ".terraformrc": RULE_TERRAFORM,
    "terraform.rc": RULE_TERRAFORM,
    ".s3cfg": RULE_TOOL_CONFIG,
    ".htpasswd": RULE_TOOL_CONFIG,
    "rclone.conf": RULE_TOOL_CONFIG,
    "application_default_credentials.json": RULE_GCLOUD,
    # Browser cookie / credential stores: the platform names for Chromium (``Cookies``,
    # ``Login Data``, ``Web Data``), Firefox (``cookies.sqlite``, ``key4.db``,
    # ``logins.json``) and Safari (``Cookies.binarycookies``). The ``.sqlite``/``.db``
    # suffix rule covers most of them already; these are the extension-less ones.
    "cookies": RULE_BROWSER_STORE,
    "login data": RULE_BROWSER_STORE,
    "web data": RULE_BROWSER_STORE,
    "logins.json": RULE_BROWSER_STORE,
    "cookies.binarycookies": RULE_BROWSER_STORE,
}

#: ``fnmatch`` patterns (casefolded basename) beyond the two composed lists.
_EXTRA_PATTERNS: Final[tuple[tuple[str, str], ...]] = (
    ("*_credentials.json", RULE_GCLOUD),
    ("*.tfrc.json", RULE_TERRAFORM),
    ("*.sqlite", RULE_DATABASE),
    ("*.sqlite3", RULE_DATABASE),
    ("*.db", RULE_DATABASE),
    ("*token*", RULE_TOKEN_SECRET),
    ("*secret*", RULE_TOKEN_SECRET),
    # The operator's standing rule: some repos keep env vars in docker-compose.
    ("docker-compose*.y*ml", RULE_COMPOSE),
    ("compose*.y*ml", RULE_COMPOSE),
    # A kubeconfig is a cluster credential store wherever it lives. The ``.kube``
    # component already covers the conventional location, but a copy rendered into a
    # project tree (``deploy/kubeconfig``) was a live candidate until agent review round 1
    # (R5) reproduced the slip. Both the prefix and the suffix spelling, because trees
    # write the variants ``kubeconfig-prod`` and ``prod.kubeconfig``.
    ("kubeconfig*", RULE_TOOL_CONFIG),
    ("*.kubeconfig", RULE_TOOL_CONFIG),
)

#: Directory components (casefolded) that deny everything beneath them, beyond the
#: composed ``SENSITIVE_DIR_PARTS`` / ``CREDENTIAL_COMPONENTS``.
_EXTRA_COMPONENTS: Final[frozenset[str]] = frozenset(
    {
        ".terraform.d",
        ".credentials",
        "secrets",
        "keychains",
        ".azure",
        ".gnupg",
    }
)

#: (``~/.aws/sso/`` and ``~/.azure/`` need no row of their own: ``.aws`` and ``.azure`` are
#: whole-directory components above, which is wider than the memo's sso-only wording.)
#:
#: Component SEQUENCES (casefolded) that deny everything beneath them. A sequence, not a bare
#: component, where the leading part is too common to deny on its own (``.config`` holds
#: harmless files; only ``gh``/``gcloud``/``rclone`` under it are credentials).
_COMPONENT_SEQUENCES: Final[tuple[tuple[tuple[str, ...], str], ...]] = (
    ((".config", "gh"), RULE_GH_HOSTS),
    ((".config", "gcloud"), RULE_GCLOUD),
    ((".config", "rclone"), RULE_TOOL_CONFIG),
)

#: ``<dir>/<file>`` pairs (casefolded) where the FILE is the secret but its directory is
#: otherwise ordinary: ``.docker/config.json`` carries registry auth, an arbitrary
#: ``config.json`` does not.
_DIR_FILE_PAIRS: Final[tuple[tuple[str, str, str], ...]] = (
    (".docker", "config.json", RULE_DOCKER_CONFIG),
    ("gh", "hosts.yml", RULE_GH_HOSTS),
)


@lru_cache(maxsize=1)
def _composed() -> (
    tuple[frozenset[str], frozenset[str], frozenset[str], frozenset[str], tuple[str, ...]]
):
    """``(names, suffixes, prefixes, dir parts, credential patterns)`` from the two existing
    gates, casefolded. Cached: the sets are module constants upstream and never change."""
    from local_operator import browser_files, references

    names = frozenset(item.casefold() for item in references.SENSITIVE_NAMES)
    suffixes = frozenset(item.casefold() for item in references.SENSITIVE_SUFFIXES)
    prefixes = frozenset(item.casefold() for item in references.SENSITIVE_NAME_PREFIXES)
    parts = frozenset(
        item.casefold()
        for item in (*references.SENSITIVE_DIR_PARTS, *browser_files.CREDENTIAL_COMPONENTS)
    )
    patterns = tuple(item.casefold() for item in browser_files.CREDENTIAL_NAME_PATTERNS)
    return names, suffixes, prefixes, parts, patterns


def _under(child: str, root: str) -> bool:
    """Whether ``child`` is ``root`` or beneath it (both already absolute and normalised)."""
    if not root:
        return False
    root = root.rstrip(os.sep) or os.sep
    return child == root or child.startswith(root + os.sep)


def _live_roots() -> list[tuple[str, str]]:
    """``(rule, root)`` pairs computed from live accessors, each in raw AND resolved form.

    Both forms because the denied tree may itself be reached through a symlink (macOS
    ``/var`` -> ``/private/var``): comparing a resolved candidate against an unresolved root
    would miss it, and the reverse likewise.
    """
    from local_operator.paths import config_dir
    from local_operator.tools.search_guard import SCRATCHPAD_ENV

    roots: list[tuple[str, str]] = []
    candidates = [(RULE_CONFIG_DIR, str(config_dir()))]
    scratch = os.environ.get(SCRATCHPAD_ENV, "").strip()
    if scratch:
        candidates.append((RULE_SCRATCHPAD, scratch))
    for rule, raw in candidates:
        expanded = os.path.abspath(os.path.expanduser(raw))
        roots.append((rule, expanded))
        resolved = os.path.realpath(expanded)
        if resolved != expanded:
            roots.append((rule, resolved))
    return roots


def _rule_for(path: str, roots: list[tuple[str, str]]) -> str:
    """The first rule that denies one absolute, normalised ``path`` string, or ``""``."""
    for rule, root in roots:
        if _under(path, root):
            return rule
    names, suffixes, prefixes, parts, patterns = _composed()
    pure = PurePosixPath(path.replace(os.sep, "/"))
    components = [part.casefold() for part in pure.parts if part not in ("/", "")]
    if not components:
        return ""
    name = components[-1]
    dirs = components[:-1]

    if name in names:
        return RULE_NAME
    if any(name.startswith(prefix) for prefix in prefixes):
        return RULE_NAME_PREFIX
    if os.path.splitext(name)[1] in suffixes:
        return RULE_SUFFIX
    if name in _EXTRA_NAMES:
        return _EXTRA_NAMES[name]
    # Components: ANY ancestor directory. The leaf is not a component (a file called
    # ``secrets`` is caught by the ``*secret*`` pattern below, and a plain file called
    # ``.azure`` is not a directory of credentials).
    for part in dirs:
        if part in parts or part in _EXTRA_COMPONENTS:
            return RULE_COMPONENT
    for sequence, rule in _COMPONENT_SEQUENCES:
        width = len(sequence)
        if any(tuple(dirs[i : i + width]) == sequence for i in range(len(dirs) - width + 1)):
            return rule
    for directory, filename, rule in _DIR_FILE_PAIRS:
        if name == filename and dirs and dirs[-1] == directory:
            return rule
    for pattern in patterns:
        if fnmatch.fnmatchcase(name, pattern):
            return RULE_CREDENTIAL_PATTERN
    for pattern, rule in _EXTRA_PATTERNS:
        if fnmatch.fnmatchcase(name, pattern):
            return rule
    return ""


def is_sensitive(path: str | os.PathLike[str], *, cwd: str | None = None) -> str:
    """The id of the rule that denies ``path``, or ``""`` when it is allowed.

    ``cwd`` anchors a relative path (a tool call's argument); ``~`` is expanded. The check
    is made on the path AS WRITTEN (made absolute and normalised, symlinks NOT followed) and
    on its RESOLVED form; the first denial wins, as-written first so a sensitive link name is
    reported as itself. A path that cannot be resolved (permission, loop) is judged on its
    written form alone -- failing closed here would hide every file behind an unreadable
    parent, and the caller's own ``stat`` already refuses what it cannot read.
    """
    raw = os.fspath(path)
    expanded = os.path.expanduser(raw)
    if not os.path.isabs(expanded):
        expanded = os.path.join(cwd or os.getcwd(), expanded)
    written = os.path.normpath(expanded)
    roots = _live_roots()
    verdict = _rule_for(written, roots)
    if verdict:
        return verdict
    try:
        resolved = os.path.realpath(written)
    except (OSError, ValueError):
        return ""
    if resolved == written:
        return ""
    return _rule_for(resolved, roots)


def is_allowed(path: str | os.PathLike[str] | Path, *, cwd: str | None = None) -> bool:
    """Convenience inverse of :func:`is_sensitive`."""
    return not is_sensitive(path, cwd=cwd)
