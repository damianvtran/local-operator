"""The sanctioned read path for redacted-but-public copy.

WHY THIS TOOL EXISTS
--------------------
The credential-shape pass is deliberately broad: a public catalogue key such as
``wire.slash.credential`` can sit in a credential-shaped position, and the value
beside it can read like a passphrase.  The pass therefore masks it, and a
translation auditor cannot read the very string they are auditing.  The
operator's reports on this class (2026-10-16 and the LOH-28 identifier report)
were each answered with a narrow rule patch; the i18n programme then withheld
``wire.slash`` at 51/52 and recorded that it refused to use a redaction bypass
(commit ``ebb70df3a8``).  A policy with no owner cannot be amended, only worked
around.

This module is the amendment: a narrow, allowlisted reader for content that is
PUBLIC BY DEFINITION — the i18n catalogues and the generated key maps.  It is
not a switch that turns redaction off for a tool call.  The caller cannot name
an arbitrary file, cannot read outside the allowlist, and every successful read
is appended to an audit log before the content is returned.  A failure to write
that record refuses the read (fail closed).

WHAT IS *NOT* EXEMPT
--------------------
Every other tool result and every tool-call argument still goes through the
normal redaction pass.  A genuine credential read with ``bash``, ``read`` or
``grep`` is masked exactly as before.  The exemption is keyed to this tool's
IDENTITY in :data:`local_operator.harness.redaction.PUBLIC_COPY_TOOLS`, never to
an argument the caller supplies, so another tool cannot borrow it.

WHICH CHECKOUTS
---------------
Two separate checkouts publish copy through this route: this repository
(``local_operator/i18n/...`` plus the mobile web key maps) and the companion
UI repository (``src/i18n/catalogues/*/*.json`` and ``src/i18n/keys.gen.ts``).
The UI checkout is a different repository, but its committed product copy is
equally public — an auditor has to be able to read a catalogue key where it
lives, and the alternative is the workaround this route exists to replace. The
roots stay an EXACT GLOB LIST in both checkouts: no directory rule
(``src/i18n/*``) and no extension rule (``*.json`` under any root), because
either would admit files nobody reviewed as public copy. The commit-clean /
``HEAD``-blob / audit-log requirements below apply equally to both.

The ``src/`` roots are additionally bound to checkout identity: they are
exposed only when the workspace's ``origin`` remote names the UI repository and
the workspace is the root of its checkout. A path is not an identity — any
repository, including one created on the spot, can hold a ``src/i18n/`` tree —
so ``git remote get-url origin`` must resolve to :data:`UI_REPOSITORY`. The gate
lives in :func:`_committed_public_copy`; ``docs/security/redaction-policy.md``
states the exact rule and the same-UID residual.

THE TRUST BOUNDARY
------------------
The allowlisted roots are public copy by repository policy
(``docs/security/redaction-policy.md``).  A secret committed into a catalogue is
a policy violation, not something this reader can detect: the content check is
structure (a JSON object of string keys to string values), not a credential
scan, because a credential scan is exactly the false positive this route exists
to bypass.  Review of changes under these roots is the control.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import stat
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field

from local_operator.harness.types import AgentTool, TextContent, ToolContext, ToolResult
from local_operator.paths import config_dir
from local_operator.tools.builtin import _resolve_workspace_path
from local_operator.tools.confinement import ToolConfinement, confinement_of

#: The tool name the loop exempts from result redaction.  Kept here and in
#: ``harness/redaction.py`` so the two cannot drift silently: a test asserts the
#: set is exactly this name.
TOOL_NAME = "read_public_copy"

#: Public-copy roots, relative to the selected checkout (session cwd by default).
#: These are the only paths this reader will open.  No directory rule and no
#: extension rule: an exact glob list, so adding a root is a reviewed policy
#: change rather than an accident.  The list spans two checkouts — this
#: repository and the companion UI repository — because both publish committed
#: product copy. Explicit selectors additionally bind the checkout to the
#: reviewed GitHub origin and corresponding core/UI path family.
#:
#: The `src/` roots additionally require checkout IDENTITY (:data:`UI_REPOSITORY`):
#: a path shape is not an identity, and any repository — including one the agent
#: creates itself — can contain a `src/i18n/` tree (review S-1).
PUBLIC_COPY_GLOBS: tuple[str, ...] = (
    # This repository: TUI/server catalogues, key modules and mobile web maps.
    "local_operator/i18n/catalogues/*/*.json",
    "local_operator/i18n/keys/*.py",
    "local_operator/mobile/web/src/i18n/keys.gen.ts",
    "local_operator/mobile/web/src/i18n/bundled.gen.ts",
    # The companion UI checkout: a separate repository whose committed product
    # copy is equally public.  Its roots are spelled from that checkout's own
    # root, so they cannot alias a path inside this repository.
    "src/i18n/catalogues/*/*.json",
    "src/i18n/keys.gen.ts",
)

#: The companion UI checkout's repository identity.  The `src/` roots are served
#: only when the workspace's `origin` remote names this repository: any checkout
#: can contain a `src/i18n/` tree, so the path alone must not lift redaction
#: (review S-1).  A fork that wants the same treatment belongs in this list, in
#: review — not in a path rule.
UI_REPOSITORY = "damianvtran/local-operator-ui"
CORE_REPOSITORY = "damianvtran/local-operator"

#: The same list as anchored patterns. ``fnmatch``'s ``*`` crosses ``/``, which
#: would admit ``catalogues/a/b/c.json``; these do not. Kept beside the globs so
#: the human-readable list and the enforcement list are edited together.  The
#: UI patterns are a separate tuple because they carry the identity requirement
#: above; :data:`_PUBLIC_COPY_PATTERNS` is their concatenation and no call site
#: matches against the UI tuple alone except the identity gate itself.
_CORE_PUBLIC_COPY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"local_operator/i18n/catalogues/[^/]+/[^/]+\.json"),
    re.compile(r"local_operator/i18n/keys/[^/]+\.py"),
    re.compile(r"local_operator/mobile/web/src/i18n/keys\.gen\.ts"),
    re.compile(r"local_operator/mobile/web/src/i18n/bundled\.gen\.ts"),
)

_UI_PUBLIC_COPY_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Anchored at ``src/``, so a lookalike deeper in a tree
    # (``vendor/src/i18n/keys.gen.ts``) is still refused.
    re.compile(r"src/i18n/catalogues/[^/]+/[^/]+\.json"),
    re.compile(r"src/i18n/keys\.gen\.ts"),
)

_PUBLIC_COPY_PATTERNS: tuple[re.Pattern[str], ...] = (
    _CORE_PUBLIC_COPY_PATTERNS + _UI_PUBLIC_COPY_PATTERNS
)

#: The policy document this route cites.  The tool's details carry it so a
#: transcript reader can find the rule that permitted the read.
POLICY_DOCUMENT = "docs/security/redaction-policy.md"

#: How much unkeyed catalogue text one call may return.  Keyed reads are the
#: audit path; a whole-file read is allowed for a small catalogue but bounded so
#: this route cannot be used as a bulk file reader.
_MAX_UNKEYED_CHARS = 200_000

#: How many matching lines a keyed read of a key module / generated map returns.
_MAX_KEY_LINES = 20

#: The audit log lives beside the session store, not in the checkout: it records
#: USE of the route, and a repository file would be committed or copied.
_AUDIT_DIRNAME = "audit"
_AUDIT_FILENAME = "public-copy-reads.jsonl"


# NOT a docstring: pydantic copies a class docstring into the published schema's
# ``description`` and every request pays for it, so the contract lives here.
#
# ``repo_root`` CONTRACT (the schema states it tersely; this is the full rule):
#   * it only SELECTS which checkout is read -- read-only, never moves the session;
#   * it must be the exact checkout/worktree root, not a subdirectory of it;
#   * that checkout's GitHub origin must be one of ``ACCEPTED_REPOSITORIES``;
#   * the public-copy allowlist is NOT widened -- ``path`` is still matched against
#     the same patterns, relative to the selected root;
#   * a confinement refusal answers ``refused: <code>`` plus one JSON line of repair
#     fields (``_next_step``), so the schema carries no refusal prose.
class ReadPublicCopyParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="Repo-relative public copy path.")
    repo_root: str | None = Field(
        default=None,
        description=(
            "Checkout/worktree root, not a subdir; origin damianvtran/local-operator or "
            "damianvtran/local-operator-ui. Read-only."
        ),
    )
    key: str | None = Field(
        default=None,
        description="Message key; required for key modules/maps.",
    )


#: The two reviewed repositories and the exact path family each one may serve.
#: Spelled once so a refusal can tell the caller which checkout a path needs
#: without a second list drifting from :data:`PUBLIC_COPY_GLOBS`.
ACCEPTED_REPOSITORIES: tuple[str, ...] = (CORE_REPOSITORY, UI_REPOSITORY)
_FAMILY_GLOBS: dict[str, list[str]] = {
    CORE_REPOSITORY: [glob for glob in PUBLIC_COPY_GLOBS if not glob.startswith("src/")],
    UI_REPOSITORY: [glob for glob in PUBLIC_COPY_GLOBS if glob.startswith("src/")],
}

#: Line one of the legacy "run it from a checkout" refusal; the structured hint
#: below is appended to it only when the caller did not pass ``repo_root``.
_NOT_A_CHECKOUT = "the workspace is not a git checkout"


class _ConfinementRefusal(Exception):
    """A confinement refusal that must not be re-read as an unreadable copy.

    WHY ITS OWN TYPE. ``_git`` raises this from inside a ``try`` whose
    ``except (OSError, subprocess.SubprocessError)`` exists to report an
    unreadable copy, and ``PermissionError`` IS an ``OSError``: the structured
    confinement refusal used to be swallowed there and re-emitted as
    "could not read the committed copy of <path> (<refusal>)" -- a doubled
    message with no code and no repair fields (review R3-1). Carrying the body
    on a type outside that handler keeps the fail-closed behaviour identical
    (still refuse, still never run Git unwrapped) while letting the caller
    answer with the code the other refusals use.
    """

    def __init__(self, body: dict[str, Any]) -> None:
        self.body = body
        super().__init__(str(body.get("code", "public_copy_confinement_unavailable")))


def _next_step(**fields: Any) -> dict[str, Any]:
    """Technical fields every structured refusal shares.

    NO PROSE HERE. This route's refusals reach the model (and the card
    headline), and AGENTS.md "Internationalization is mandatory" keeps new
    human-facing English out of code. A code plus the accepted repositories,
    path families and the one requirement that failed is enough for the caller
    to repair its own call; the ``repo_root`` CONTRACT comment above
    :class:`ReadPublicCopyParams` explains what those fields mean once.
    """
    return {
        "accepted_repositories": list(ACCEPTED_REPOSITORIES),
        "path_families": _FAMILY_GLOBS,
        **fields,
    }


def _refusal(
    tool_call_id: str, code: str, *, sentence: str | None = None, **fields: Any
) -> ToolResult:
    """A refusal as ``refused: <code>`` plus one JSON line of repair fields.

    The first line keeps the shape every other refusal here has (and the card
    headline); the second is what makes it actionable. ``details`` carries the
    same object for the transcript, but providers never see ``details`` (see
    ``Message.tool_result``), so the text line is the only copy the model gets.
    """
    body = {"code": code, **fields}
    return _text(
        tool_call_id,
        f"refused: {sentence or code}\n{json.dumps(body, sort_keys=True, ensure_ascii=False)}",
        details={"refusal": body},
        is_error=True,
    )


def _text(
    tool_call_id: str,
    text: str,
    *,
    details: dict[str, Any] | None = None,
    is_error: bool = False,
) -> ToolResult:
    """A plain-text result in this tool's own name."""
    return ToolResult(
        tool_call_id=tool_call_id,
        tool_name=TOOL_NAME,
        content=[TextContent(text=text)],
        details=details,
        is_error=is_error,
    )


def _matches_public_copy(relative: str) -> bool:
    """Whether ``relative`` (POSIX spelling) is exactly an allowlisted file."""
    return any(pattern.fullmatch(relative) for pattern in _PUBLIC_COPY_PATTERNS)


def _requires_ui_checkout(relative: str) -> bool:
    """Whether ``relative`` belongs to the companion UI checkout's roots."""
    return any(pattern.fullmatch(relative) for pattern in _UI_PUBLIC_COPY_PATTERNS)


def _origin_identity(url: str) -> str:
    """Reduce an origin URL to ``owner/repo``, lower case; ``""`` if unreadable.

    Git hands back three shapes for a GitHub remote: the scp form
    (``git@github.com:owner/repo.git``), an ssh/https URL, or a local path.  Only
    the last two path components matter, and only :data:`UI_REPOSITORY` matches
    them, so a lookalike repository in another namespace does not qualify.  The
    raw URL is never echoed: it can embed a credential, and this string reaches
    the model-visible refusal sentence.
    """
    value = url.strip().rstrip("/")
    if value.endswith(".git"):
        value = value[:-4]
    if "://" in value:
        after_scheme = value.split("://", 1)[1]
        path = after_scheme.split("/", 1)[1] if "/" in after_scheme else ""
    elif re.match(r"^[^/@]+@[^/:]+:", value):
        path = value.split(":", 1)[1]
    else:
        path = value
    parts = [part for part in path.split("/") if part]
    if len(parts) < 2:
        return ""
    return "/".join(parts[-2:]).lower()


def _trusted_origin_identity(url: str) -> str:
    """Known-host origin for explicit selectors; never return credentials/URLs.

    Origin is a repository-policy assertion, not cryptographic attestation. The
    same-UID residual remains, but a lookalike path on an arbitrary host or on
    disk must not borrow the reviewed GitHub repository's identity.
    """
    value = url.strip()
    if "://" in value:
        try:
            parsed = urlsplit(value)
        except ValueError:
            return ""
        if parsed.scheme not in ("ssh", "https") or parsed.hostname != "github.com":
            return ""
        path = parsed.path.strip("/")
    else:
        match = re.fullmatch(r"git@github\.com:([^\s]+)", value, flags=re.IGNORECASE)
        if match is None:
            return ""
        path = match.group(1).rstrip("/")
    if path.endswith(".git"):
        path = path[:-4]
    return path.lower() if re.fullmatch(r"[^/]+/[^/]+", path) else ""


#: The audit log's open flags.  ``O_NONBLOCK`` is for the FIFO case: with
#: ``O_WRONLY`` and no reader the open returns ``ENXIO`` instead of blocking, so
#: a FIFO planted at the path is a refusal rather than a hung tool call (review
#: S-8 measured the blocking open past 6 s).  Regular files ignore the flag.
#: Windows has no FIFO at a filesystem path and no ``O_NONBLOCK``, so it
#: resolves to 0 there.
_AUDIT_OPEN_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NONBLOCK", 0)


def _audit_path() -> Path:
    """The audit log this process appends to.

    ``paths.config_dir()`` reads the override on every call, so a test or an
    isolated run lands under its own home rather than the operator's real
    ``~/.local-operator`` — the same invariant the secret store keeps.
    """
    return config_dir() / _AUDIT_DIRNAME / _AUDIT_FILENAME


def _open_audit_log(audit_dir: Path) -> int:
    """Open the audit log without following links; return a private regular fd.

    The record is a control, so the log is not opened like an ordinary file.
    Review S-3 measured the bypasses this closes: a symlink at
    ``public-copy-reads.jsonl`` (to ``/dev/null`` or anywhere else) suppressed
    the record while the read served, a symlinked ``audit`` directory redirected
    records, and an existing permissive file stayed permissive.

    POSIX opens the directory itself with ``O_DIRECTORY | O_NOFOLLOW``, pins the
    file to that directory with ``dir_fd``, ``O_NOFOLLOW`` and ``O_NONBLOCK``,
    requires a regular file and enforces mode ``0600`` on the open inode
    (``fchmod`` on the pinned object, not a path lookup).  ``O_NONBLOCK`` makes
    a FIFO at the log path fail the open (``ENXIO``) instead of blocking it
    before the regular-file check (review S-8).  Windows has no such flags, so
    it refuses the two link cases by ``lstat`` and checks the opened handle;
    ACLs replace the mode check there.  Any failure raises and the caller
    refuses the read.
    """
    audit_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = audit_dir / _AUDIT_FILENAME
    if os.name == "nt":
        if audit_dir.is_symlink() or path.is_symlink():
            raise OSError(f"{path} is reached through a symlink")
        fd = os.open(path, _AUDIT_OPEN_FLAGS, 0o600)
    else:
        dir_fd = os.open(audit_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fchmod(dir_fd, 0o700)
            fd = os.open(
                _AUDIT_FILENAME,
                _AUDIT_OPEN_FLAGS | os.O_NOFOLLOW,
                0o600,
                dir_fd=dir_fd,
            )
        finally:
            os.close(dir_fd)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise OSError(f"{path} is not a regular file")
        if os.name != "nt":
            if info.st_mode & 0o777 != 0o600:
                os.fchmod(fd, 0o600)
                info = os.fstat(fd)
            if info.st_mode & 0o777 != 0o600:
                raise OSError(f"{path} is not mode 0600")
    except OSError:
        os.close(fd)
        raise
    return fd


def _record_read(
    *,
    relative: str,
    key: str | None,
    chars: int,
    digest: str,
    context: ToolContext,
    provenance: dict[str, str],
) -> Path:
    """Append one audit record; raise when it cannot be written.

    The caller turns the exception into a refused read.  Returning content
    without its record is the one outcome this route may not have: it would be
    a silent bypass with a policy name on it.
    """
    path = _audit_path()
    session = Path(context.session_dir).name if context.session_dir else "unknown"
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tool": TOOL_NAME,
        "session": session,
        "path": relative,
        "key": key,
        "chars": chars,
        "sha256": digest,
        "policy": POLICY_DOCUMENT,
        **provenance,
    }
    line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    # O_APPEND keeps concurrent readers' records whole without a lock file; a
    # single write of a short line is atomic on the local filesystem.
    fd = _open_audit_log(path.parent)
    try:
        os.write(fd, line.encode("utf-8"))
        # The record is the control, not a log best-effort: flush it to stable
        # storage before the caller gets the content.
        os.fsync(fd)
    finally:
        os.close(fd)
    return path


def _read_catalogue(text: str, key: str | None) -> str:
    """Validate a catalogue and return the requested key (or the whole object)."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"not a JSON catalogue: {exc.msg} at line {exc.lineno}") from exc
    if not isinstance(payload, dict):
        raise ValueError("catalogue must be a JSON object")
    for name, value in payload.items():
        if not isinstance(name, str) or not isinstance(value, str):
            raise ValueError("catalogue keys and values must be strings")
    if key is None:
        return json.dumps(payload, ensure_ascii=False, indent=1)
    if key not in payload:
        raise KeyError(f"key not found in this catalogue: {key}")
    return json.dumps({key: payload[key]}, ensure_ascii=False, indent=1)


def _read_key_source(text: str, key: str) -> str:
    """Return the lines of a key module / generated map that name ``key``."""
    matches = [line for line in text.splitlines() if key in line]
    if not matches:
        raise KeyError(f"key not found in this key source: {key}")
    return "\n".join(matches[:_MAX_KEY_LINES])


def _metadata_outside(
    root: Path,
    confinement: ToolConfinement,
    git: Any,
) -> dict[str, Any] | None:
    """A refusal when Git would read repository metadata outside the confinement root.

    WHY THIS EXISTS. The kernel sandbox allows reads of ``/private/var/folders``
    (``confinement.py``, "WHAT THIS DOES NOT COVER"), which is where macOS puts
    the default temp directory. A linked worktree confined to its own directory
    but whose common ``.git`` lives in per-user temp therefore let Git read it
    inside the sandbox, so the reader was no stricter than ``bash`` there. Code
    review called that clean on a rig whose ``TMPDIR`` hid it (the sandbox
    denied ``/tmp`` and the home directory); QA and design re-ran it at the
    macOS default and it failed. The sandbox allowlist is shared with every
    other tool and is deliberately left alone: the reader refuses instead.

    CHECKED: the git dir, the common dir, and every ``objects/info/alternates``
    entry (the same class: Git follows them to read objects). Fails closed:
    anything unreadable or unresolvable refuses.
    """

    def _resolved(args: tuple[str, ...]) -> Path | None:
        result = git(*args)
        text = result.stdout.decode("utf-8", "replace").strip()
        if result.returncode != 0 or not text:
            return None
        candidate = Path(text)
        try:
            return (candidate if candidate.is_absolute() else root / candidate).resolve()
        except (OSError, ValueError):
            return None

    def _refused(what: str) -> dict[str, Any]:
        return {
            "code": "public_copy_git_metadata_outside_confinement",
            "reason": what,
            "confinement_root": str(confinement.root),
            "repo_root": str(root),
        }

    git_dir = _resolved(("rev-parse", "--git-dir"))
    common_dir = _resolved(("rev-parse", "--git-common-dir"))
    if git_dir is None or common_dir is None:
        return _refused("git_metadata_unresolvable")
    for label, location in (("git_dir", git_dir), ("git_common_dir", common_dir)):
        if not confinement.contains(location):
            return _refused(f"{label}_outside_root")
    alternates = common_dir / "objects" / "info" / "alternates"
    try:
        entries = alternates.read_text(encoding="utf-8").splitlines() if alternates.exists() else []
    except (OSError, UnicodeDecodeError):
        return _refused("alternates_unreadable")
    for entry in entries:
        if not entry.strip() or entry.lstrip().startswith("#"):
            continue
        target = Path(entry.strip())
        try:
            target = (target if target.is_absolute() else alternates.parent / target).resolve()
        except (OSError, ValueError):
            return _refused("alternates_unresolvable")
        if not confinement.contains(target):
            return _refused("alternates_outside_root")
    return None


def _committed_public_copy(
    root: Path,
    relative: str,
    *,
    requires_ui_identity: bool = False,
    explicit_root: bool = False,
    confinement: ToolConfinement | None = None,
) -> tuple[bytes, dict[str, Any], str | dict[str, Any] | None]:
    """The committed blob for ``relative``, or a refusal.

    A refusal is a legacy sentence (``str``, the pre-existing wording, kept
    verbatim) or a ``dict`` with a ``code`` plus repair fields, which the caller
    renders through :func:`_refusal`.

    ``HEAD`` is the committed copy, and this reader serves THAT blob rather than
    the working-tree file. Five consequences, all measured in review:

    * a working-tree secret cannot be read back (the write-a-secret bypass);
    * a clean checkout with ``core.autocrlf=true`` (Git for Windows' default)
      still serves, because the comparison is Git's own index/worktree
      comparison and the bytes are the blob's LF bytes. Raw byte equality
      against the working tree failed every read on that platform;
    * ``--no-replace-objects`` keeps a replace ref from substituting the served
      blob while ``status``/``diff`` still look clean (review S-4);
    * the workspace root must be the checkout root (review S-5): ``ls-files``
      and ``diff`` are cwd-relative while ``show HEAD:<path>`` is
      toplevel-relative, so a workspace below the toplevel could validate one
      file and serve another;
    * the UI roots need checkout identity, not path shape (review S-1);
      ``requires_ui_identity`` gates that check here so all git probing shares
      one wrapper.

    The helper is synchronous and runs off the event loop
    (:func:`execute_read_public_copy` awaits it in a thread).
    """

    def _git(*args: str) -> subprocess.CompletedProcess[bytes]:
        argv = ["git", "-C", str(root), "--no-replace-objects", "-c", "core.fsmonitor=false", *args]
        if confinement is not None:
            wrapped = confinement.wrap(argv)
            if wrapped is None:
                # Fields only: ``execute_read_public_copy`` wraps a dict refusal
                # through ``_next_step``, so prose here would bypass the
                # fields-only rule the other refusals follow (the ``repo_root``
                # CONTRACT comment above ``ReadPublicCopyParams``).
                raise _ConfinementRefusal(
                    {
                        "code": "public_copy_confinement_unavailable",
                        "reason": "host_cannot_confine_git",
                        "confinement_root": str(confinement.root),
                        "fix": "read from an unconfined session or a host with a kernel sandbox",
                    }
                )
            argv = wrapped
        # A caller's Git environment must not redirect -C to another index,
        # object store or worktree. Reads also must not invoke configured diff
        # helpers, credential helpers, fsmonitor hooks or optional index writes.
        env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
        env.update(
            GIT_CONFIG_NOSYSTEM="1",
            GIT_CONFIG_GLOBAL=os.devnull,
            GIT_TERMINAL_PROMPT="0",
            GIT_OPTIONAL_LOCKS="0",
            # A missing promisor blob must fail, never turn a copy read into
            # network/credential work. The empty transport allowlist also
            # closes lazy-fetch fallback on Git builds predating its opt-out.
            GIT_NO_LAZY_FETCH="1",
            GIT_ALLOW_PROTOCOL="",
        )
        return subprocess.run(argv, capture_output=True, check=False, timeout=10, env=env)

    try:
        inside = _git("rev-parse", "--is-inside-work-tree")
        if inside.returncode != 0 or inside.stdout.strip() != b"true":
            return b"", {}, "the workspace is not a git checkout"
        top = _git("rev-parse", "--show-toplevel")
        if top.returncode != 0 or not top.stdout.strip():
            return b"", {}, "the workspace is not a git checkout"
        checkout_root = Path(top.stdout.decode("utf-8", "replace").strip()).resolve()
        if checkout_root != root:
            return (
                b"",
                {},
                (
                    f"the workspace root {root} is not the checkout root {checkout_root}; "
                    "run read_public_copy from the checkout root"
                ),
            )
        if confinement is not None:
            escaped = _metadata_outside(root, confinement, _git)
            if escaped is not None:
                return b"", {}, escaped
        origin = _git("remote", "get-url", "origin")
        origin_url = origin.stdout.decode("utf-8", "replace") if origin.returncode == 0 else ""
        identity = _origin_identity(origin_url)
        # RECORD ONLY WHAT THE STRICT CHECK PROVED. The default-root gate below
        # keeps its historical loose match (last two path components, any host)
        # so that path reads exactly the files it always did; but provenance is
        # a trust claim, so ``repository`` comes from the GitHub-only check and
        # is ``None`` when that check did not pass (design D1/D4).
        verified = _trusted_origin_identity(origin_url) or None
        if explicit_root:
            expected = UI_REPOSITORY if requires_ui_identity else CORE_REPOSITORY
            if verified != expected:
                return (
                    b"",
                    {},
                    {
                        "code": "public_copy_repository_not_allowed",
                        # Only a reviewed constant is ever named, never the raw
                        # origin (it can embed a credential).
                        "reason": (
                            "wrong_checkout_for_path"
                            if verified in ACCEPTED_REPOSITORIES
                            else "origin_not_accepted"
                        ),
                        "checkout_is": verified,
                        "path_needs": expected,
                    },
                )
        elif requires_ui_identity and identity != UI_REPOSITORY:
            return (
                b"",
                {},
                (
                    f"{relative} is the companion UI checkout's copy; this workspace"
                    f" is not {UI_REPOSITORY} (origin: {identity or 'unusable'})"
                ),
            )
        # Capture once: a concurrent branch switch may make the cleanliness
        # comparison refuse, but can never relabel bytes from another commit.
        head = _git("rev-parse", "--verify", "HEAD^{commit}")
        commit = head.stdout.decode("ascii", "replace").strip()
        if head.returncode != 0 or not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit):
            return (
                b"",
                {},
                {"code": "public_copy_commit_unavailable", "reason": "head_has_no_commit"},
            )
        provenance = {
            "repository_root": str(root),
            "repository": verified,
            "repository_verified": verified is not None,
            "commit": commit,
        }
        tracked = _git("ls-files", "--error-unmatch", "--", relative)
        if tracked.returncode != 0:
            return b"", {}, f"{relative} is not committed at HEAD; commit the public copy first"
        # --no-ext-diff/--no-textconv do not disable clean/process filters:
        # Git can execute one while refreshing its worktree hash. A read-tier
        # tool must refuse such paths before diff, not run repository commands
        # with the caller's write authority. check-attr itself runs no filter.
        attributes = _git("check-attr", "-z", "--all", "--", relative)
        fields = attributes.stdout.split(b"\0")
        # --all distinguishes absence from an actual driver literally named
        # "unspecified" or "unset". A named-attribute query renders those
        # strings identically to its absence/disabled sentinels.
        if attributes.returncode != 0 or len(fields) % 3 != 1:
            return (
                b"",
                {},
                {
                    "code": "public_copy_filter_not_supported",
                    "reason": "attributes_unreadable",
                    "path": relative,
                },
            )
        if b"filter" in fields[1::3]:
            declared = fields[2::3][fields[1::3].index(b"filter")]
            return (
                b"",
                {},
                {
                    "code": "public_copy_filter_not_supported",
                    "reason": "path_declares_git_filter",
                    "path": relative,
                    "filter": declared.decode("utf-8", "replace")[:80],
                },
            )
        clean = _git("diff", "--no-ext-diff", "--no-textconv", "--quiet", commit, "--", relative)
        if clean.returncode == 1:
            return (
                b"",
                {},
                (f"{relative} differs from HEAD; commit the public copy before auditing it"),
            )
        if clean.returncode != 0:
            return b"", {}, f"could not compare {relative} with HEAD"
        blob = _git("show", f"{commit}:{relative}")
        if blob.returncode != 0:
            # Reaching here means the path is tracked AND the worktree equals the
            # pinned tree (the diff above returned 0), so the path IS committed
            # and only its object is missing: a partial clone whose fetch is
            # disabled on purpose. Saying "not committed" sent auditors to commit
            # a file that already is (QA Q1).
            return (
                b"",
                {},
                {
                    "code": "public_copy_blob_unavailable",
                    "reason": "object_missing_locally",
                    "path": relative,
                    "commit": commit,
                },
            )
        return blob.stdout, provenance, None
    except _ConfinementRefusal as exc:
        return b"", {}, exc.body
    except (OSError, subprocess.SubprocessError) as exc:
        return b"", {}, f"could not read the committed copy of {relative} ({exc})"


async def execute_read_public_copy(
    tool_call_id: str,
    arguments: dict[str, Any],
    signal: Any = None,
    on_update: Any = None,
    context: ToolContext | None = None,
) -> ToolResult:
    """Read one allowlisted public-copy file (or key) and record the read."""
    del signal, on_update
    if context is None:
        return _text(
            tool_call_id,
            "read_public_copy needs a session context to resolve and audit the path.",
            is_error=True,
        )
    path_arg = str(arguments.get("path") or "")
    key = arguments.get("key")
    if key is not None:
        key = str(key).strip()
        if not key:
            return _text(tool_call_id, "key must not be empty", is_error=True)
    if not path_arg or "\x00" in path_arg:
        return _text(
            tool_call_id,
            "path must be a non-empty repository-relative path",
            is_error=True,
        )

    selected = arguments.get("repo_root")
    explicit_root = selected is not None
    if explicit_root and (not isinstance(selected, str) or not selected.strip()):
        return _refusal(
            tool_call_id,
            "public_copy_invalid_repo_root",
            **_next_step(
                reason="repo_root_blank_or_not_a_string", fix="pass a checkout path or omit"
            ),
        )
    if explicit_root and Path(path_arg).is_absolute():
        return _refusal(
            tool_call_id,
            "public_copy_path_must_be_relative",
            **_next_step(reason="absolute_path_with_repo_root", path=path_arg),
        )
    try:
        candidate = Path(str(selected)).expanduser() if explicit_root else Path(context.cwd)
        if explicit_root and not candidate.is_absolute():
            candidate = Path(context.cwd) / candidate
        root = candidate.resolve(strict=explicit_root)
    except (OSError, ValueError):
        return _refusal(
            tool_call_id,
            "public_copy_invalid_repo_root",
            **_next_step(reason="repo_root_does_not_exist", repo_root=str(selected)),
        )
    confinement = confinement_of(context)
    if confinement is not None:
        denial = confinement.path_denial(root)
        if denial is not None:
            return _text(tool_call_id, denial, is_error=True)
    resolved, inside, resolvable = _resolve_workspace_path(path_arg, str(root))
    if confinement is not None:
        denial = confinement.path_denial(resolved, resolvable=resolvable)
        if denial is not None:
            return _text(tool_call_id, denial, is_error=True)
    if not resolvable or not inside:
        return _text(
            tool_call_id,
            "refused: the path is not inside this session's workspace",
            is_error=True,
        )
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError:
        return _text(tool_call_id, "refused: the path escapes the workspace", is_error=True)
    if not _matches_public_copy(relative):
        if explicit_root:
            return _refusal(
                tool_call_id,
                "public_copy_path_not_allowed",
                **_next_step(
                    # ``.git`` (a directory, or the file a linked worktree has) is
                    # only present at a checkout root: no Git process is needed
                    # to tell "you gave me a subdirectory" from "wrong path".
                    reason=(
                        "path_outside_allowlist_relative_to_repo_root"
                        if (root / ".git").exists()
                        else "repo_root_is_not_the_checkout_root"
                    ),
                    path=relative,
                    repo_root=str(root),
                    hint="repo_root must be the exact checkout root; path is relative to it",
                ),
            )
        return _text(
            tool_call_id,
            (
                "refused: not a public-copy path. This tool reads only "
                + ", ".join(PUBLIC_COPY_GLOBS)
            ),
            is_error=True,
        )

    # THE BYTES MUST BE THE COMMITTED BYTES. The allowlisted roots are writable
    # by the same agent that calls this tool, so trusting whatever is on disk is
    # a two-call bypass: write a credential into a catalogue, read it back
    # verbatim. ``_committed_public_copy`` serves the HEAD blob and refuses a
    # working-tree difference or an untracked file; it is also why a CRLF
    # checkout works (Git's comparison, not raw byte equality).
    blob, provenance, refusal = await asyncio.to_thread(
        _committed_public_copy,
        root,
        relative,
        requires_ui_identity=_requires_ui_checkout(relative),
        explicit_root=explicit_root,
        confinement=confinement,
    )
    if isinstance(refusal, dict):
        return _refusal(tool_call_id, refusal.pop("code"), **_next_step(**refusal))
    if refusal is not None:
        if not explicit_root and refusal == _NOT_A_CHECKOUT:
            # THE FIRST PLACE A HOME-ROOT SESSION FAILS (UX U2): name the option
            # that exists for exactly this, as fields, with no new prose.
            return _refusal(
                tool_call_id,
                "public_copy_workspace_not_a_checkout",
                sentence=refusal,
                **_next_step(reason="session_cwd_is_not_a_checkout", fix="pass repo_root"),
            )
        return _text(tool_call_id, f"refused: {refusal}", is_error=True)
    try:
        text = blob.decode("utf-8")
    except UnicodeDecodeError as exc:
        return _text(tool_call_id, f"could not decode {relative} as UTF-8: {exc}", is_error=True)

    try:
        if relative.endswith(".json"):
            if key is None and len(text) > _MAX_UNKEYED_CHARS:
                return _text(
                    tool_call_id,
                    f"refused: catalogue is larger than {_MAX_UNKEYED_CHARS} characters; pass key",
                    is_error=True,
                )
            body = _read_catalogue(text, key)
        else:
            if key is None:
                return _text(
                    tool_call_id,
                    "key is required for a key module or generated map",
                    is_error=True,
                )
            body = _read_key_source(text, key)
    except (KeyError, ValueError) as exc:
        return _text(tool_call_id, f"refused: {exc}", is_error=True)

    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    try:
        audit_file = _record_read(
            relative=relative,
            key=key,
            chars=len(body),
            digest=digest,
            context=context,
            provenance=provenance,
        )
    except OSError as exc:
        # FAIL CLOSED. A route whose use is not recorded is not this route.
        return _text(
            tool_call_id,
            f"refused: the audit record could not be written ({exc})",
            is_error=True,
        )

    details = {
        "path": relative,
        "key": key,
        "public_copy": True,
        "policy": POLICY_DOCUMENT,
        "audit_log": str(audit_file),
        "sha256": digest,
        **provenance,
    }
    return _text(tool_call_id, body, details=details)


def build_read_public_copy_tool() -> AgentTool:
    """The tool definition advertised to the model."""
    return AgentTool(
        name=TOOL_NAME,
        label="Read public copy",
        description=("Read allowlisted public i18n copy: key, key-module line or map; audited."),
        parameters=ReadPublicCopyParams.model_json_schema(),
        approval_tier="read",
        concurrency="shared",
        interruptible=False,
        execute=execute_read_public_copy,
    )
