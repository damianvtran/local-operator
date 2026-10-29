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
Three separate checkouts publish copy through this route: this repository
(``local_operator/i18n/...`` plus the mobile web key maps) and the two consumer
apps that ship their own translated copy — the companion UI repository
(``src/i18n/catalogues/*/*.json`` and ``src/i18n/keys.gen.ts``) and the companion
mobile repository (``src/i18n/catalogues/*/*.json`` and
``src/i18n/catalogue.gen.ts``). A consumer checkout is a different repository,
but its committed product copy is equally public — an auditor has to be able to
read a catalogue key where it lives, and the alternative is the workaround this
route exists to replace. The roots stay an EXACT GLOB LIST in all three
checkouts: no directory rule (``src/i18n/*``) and no extension rule (``*.json``
under any root), because either would admit files nobody reviewed as public
copy. The commit-clean / ``HEAD``-blob / audit-log requirements below apply
equally to all three.

The ``src/`` roots are additionally bound to checkout identity: they are
exposed only when the workspace's ``origin`` remote names the consumer
repository that OWNS that root, and the workspace is the root of its checkout.
A path is not an identity — any repository, including one created on the spot,
can hold a ``src/i18n/`` tree — so ``git remote get-url origin`` must parse to
the identity :data:`_CONSUMER_CHECKOUTS` binds to the path. The binding is per
checkout, not one shared list, because both apps generated compiled manifests
under ``src/i18n/`` and the UI's map must not be served from a mobile checkout
nor the reverse; the gate lives in :func:`_committed_public_copy`;
``docs/security/redaction-policy.md`` states the exact rule and the same-UID
residual.

NAMING A CHECKOUT
-----------------
A session's cwd is not always the tree under audit: an independent translation
audit runs from wherever the agent is and reads a specific branch, and the
session has no way to move its own workspace there.  So ``repo_root`` may name a
DIRECTORY.  It is admitted only when its ``origin`` reduces to one of
:data:`PUBLISHING_REPOSITORIES` (:func:`_admitted_checkout`) — the same
identities the roots above are spelled for.  Inside an admitted root the
allowlist, the per-checkout identity binding, the committed-``HEAD`` rule and
the audit record are the ones above, unchanged, and the record gains the
resolved identity and path.  The file argument stays repository-relative:
naming a checkout changes WHERE the allowlist is applied, never WHAT may be
read, and a directory with no origin, an unrecognised origin or a symlink is
refused by name.

That identity check separates an UNRELATED tree from a publishing one — it is a
guard against auditing the wrong repository, not proof of provenance.  It reads
the checkout's own ``origin`` configuration, so a caller who can run three git
commands can manufacture a directory that passes it; the controls that remain are
the allowlist, the committed-blob rule and the audit record, and
``docs/security/redaction-policy.md`` §6 owns the same-UID residual.

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

from pydantic import BaseModel, ConfigDict, Field

from local_operator.harness.types import AgentTool, TextContent, ToolContext, ToolResult
from local_operator.paths import config_dir
from local_operator.tools.builtin import _resolve_workspace_path

#: The tool name the loop exempts from result redaction.  Kept here and in
#: ``harness/redaction.py`` so the two cannot drift silently: a test asserts the
#: set is exactly this name.
TOOL_NAME = "read_public_copy"

#: Public-copy roots, relative to the session working directory (the checkout).
#: These are the only paths this reader will open.  No directory rule and no
#: extension rule: an exact glob list, so adding a root is a reviewed policy
#: change rather than an accident.  The list spans three checkouts — this
#: repository and the two consumer apps — because all publish committed product
#: copy; a session resolves the roots against its own working directory, so each
#: checkout only ever exposes its own copy.
#:
#: This tuple is the reviewed INVENTORY, not the binding: which checkout may
#: serve which `src/` root lives in :data:`_CONSUMER_CHECKOUTS`.
#:
#: The `src/` roots additionally require checkout IDENTITY
#: (:data:`_CONSUMER_CHECKOUTS`): a path shape is not an identity, and any
#: repository — including one the agent creates itself — can contain a
#: `src/i18n/` tree (review S-1).
PUBLIC_COPY_GLOBS: tuple[str, ...] = (
    # This repository: TUI/server catalogues, key modules and mobile web maps.
    "local_operator/i18n/catalogues/*/*.json",
    "local_operator/i18n/keys/*.py",
    "local_operator/mobile/web/src/i18n/keys.gen.ts",
    "local_operator/mobile/web/src/i18n/bundled.gen.ts",
    # The consumer checkouts: separate repositories whose committed product copy
    # is equally public.  Their roots are spelled from each checkout's own root,
    # so they cannot alias a path inside this repository.  The catalogue root is
    # shared — both apps keep `catalogues/<locale>/<key>.json` — while the
    # generated map differs per app and is bound per app below.
    "src/i18n/catalogues/*/*.json",
    "src/i18n/keys.gen.ts",
    "src/i18n/catalogue.gen.ts",
)

#: This repository's own identity.  It decides nothing for the session's own
#: workspace — that is trusted by being the workspace — but a directory NAMED
#: through ``repo_root`` has to show it is a published tree, and this is the
#: identity the core roots belong to.
CORE_REPOSITORY = "damianvtran/local-operator"

#: The consumer checkouts' repository identities.  A `src/` root is served only
#: when the workspace's `origin` remote parses to the identity that owns it: any
#: checkout can contain a `src/i18n/` tree, so the path alone must not lift
#: redaction (review S-1).  A fork that wants the same treatment belongs in this
#: list, in review — not in a path rule.
UI_REPOSITORY = "damianvtran/local-operator-ui"
MOBILE_REPOSITORY = "damianvtran/local-operator-mobile"

#: The same list as anchored patterns. ``fnmatch``'s ``*`` crosses ``/``, which
#: would admit ``catalogues/a/b/c.json``; these do not. Kept beside the globs so
#: the human-readable list and the enforcement list are edited together.
#:
#: The `src/` patterns are grouped per checkout because the identity gate is per
#: checkout: a mobile checkout must not serve the UI's generated map, and the
#: reverse.  The shapes are defined once and referenced from both groups, so the
#: catalogue root cannot drift between the two apps.
_CORE_PUBLIC_COPY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"local_operator/i18n/catalogues/[^/]+/[^/]+\.json"),
    re.compile(r"local_operator/i18n/keys/[^/]+\.py"),
    re.compile(r"local_operator/mobile/web/src/i18n/keys\.gen\.ts"),
    re.compile(r"local_operator/mobile/web/src/i18n/bundled\.gen\.ts"),
)

#: Anchored at ``src/``, so a lookalike deeper in a tree
#: (``vendor/src/i18n/keys.gen.ts``) is still refused.
_SRC_CATALOGUE_PATTERN = re.compile(r"src/i18n/catalogues/[^/]+/[^/]+\.json")
_UI_MAP_PATTERN = re.compile(r"src/i18n/keys\.gen\.ts")
_MOBILE_MAP_PATTERN = re.compile(r"src/i18n/catalogue\.gen\.ts")

_UI_PUBLIC_COPY_PATTERNS: tuple[re.Pattern[str], ...] = (
    _SRC_CATALOGUE_PATTERN,
    _UI_MAP_PATTERN,
)

_MOBILE_PUBLIC_COPY_PATTERNS: tuple[re.Pattern[str], ...] = (
    _SRC_CATALOGUE_PATTERN,
    _MOBILE_MAP_PATTERN,
)

#: Which identity may publish which `src/` root.  ``(identity, description,
#: patterns)``: the description is what the refusal sentence names, so a caller
#: who used the wrong checkout is told which checkout the root belongs to without
#: the raw URL (which can embed a credential).  Ordered, so a refusal for a root
#: two checkouts share names them in a stable order.
_CONSUMER_CHECKOUTS: tuple[tuple[str, str, tuple[re.Pattern[str], ...]], ...] = (
    (UI_REPOSITORY, "companion UI checkout", _UI_PUBLIC_COPY_PATTERNS),
    (MOBILE_REPOSITORY, "mobile app checkout", _MOBILE_PUBLIC_COPY_PATTERNS),
)

_PUBLIC_COPY_PATTERNS: tuple[re.Pattern[str], ...] = _CORE_PUBLIC_COPY_PATTERNS + (
    _SRC_CATALOGUE_PATTERN,
    _UI_MAP_PATTERN,
    _MOBILE_MAP_PATTERN,
)

#: Every checkout whose committed copy this route may serve, derived from the
#: two tables above rather than restated: the identities a named ``repo_root`` is
#: admitted by are exactly the identities whose roots the allowlist spells.  This
#: is a guard against admitting an unrelated tree, NOT an authenticity proof —
#: the identity is read from the checkout's own ``origin`` configuration, so a
#: caller who can run three git commands can claim one (disclosed in
#: :func:`_admitted_checkout`).  Adding a fork here is a reviewed policy change
#: like adding a glob.
PUBLISHING_REPOSITORIES: tuple[str, ...] = (CORE_REPOSITORY,) + tuple(
    identity for identity, _, _ in _CONSUMER_CHECKOUTS
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


class ReadPublicCopyParams(BaseModel):
    """Arguments for the sanctioned public-copy reader."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="Repository-relative public i18n copy path.")
    key: str | None = Field(
        default=None,
        description="Catalogue/message key; required for key modules and maps.",
    )
    repo_root: str | None = Field(
        default=None,
        description=(
            "Absolute path of a publishing-repository checkout to read from, when the"
            " session cwd is not that checkout; `~` is expanded, a relative spelling is"
            " refused."
        ),
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


def _publishing_checkouts(relative: str) -> tuple[tuple[str, str], ...]:
    """The ``(identity, description)`` pairs whose checkout publishes ``relative``.

    Empty for this repository's own roots: they carry no identity gate.  A root
    shaped like a consumer app's is bound to the identity that owns it, so the
    UI's generated map is never served from a mobile checkout and the reverse.
    """
    return tuple(
        (identity, description)
        for identity, description, patterns in _CONSUMER_CHECKOUTS
        if any(pattern.fullmatch(relative) for pattern in patterns)
    )


def _checkout_refusal(relative: str, required: tuple[tuple[str, str], ...], identity: str) -> str:
    """The refusal sentence for a ``src/`` root served from the wrong checkout.

    Names the expected identity (or both, when two checkouts publish the same
    relative root) and the parsed identity of this workspace — never the raw
    URL, which can embed a credential and whose text reaches the model.
    """
    sources = " or ".join(f"the {description}'s copy" for _, description in required)
    identities = [name for name, _ in required]
    if len(identities) == 1:
        requirement = f"this workspace is not {identities[0]}"
    else:
        requirement = f"this workspace is neither {' nor '.join(identities)}"
    return f"{relative} is {sources}; {requirement} (origin: {identity or 'unusable'})"


def _origin_identity(url: str) -> str:
    """Reduce an origin URL to ``owner/repo``, lower case; ``""`` if unreadable.

    Git hands back three shapes for a GitHub remote: the scp form
    (``git@github.com:owner/repo.git``), an ssh/https URL, or a local path.  Only
    the last two path components matter, and only the identities in
    :data:`_CONSUMER_CHECKOUTS` match them, so a lookalike repository in another
    namespace does not qualify.  The raw URL is never echoed: it can embed a
    credential, and this string reaches the model-visible refusal sentence.
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


def _admitted_checkout(raw: str) -> tuple[Path, str, str | None]:
    """The checkout ``repo_root`` names, with its identity, or a refusal sentence.

    WHY A DIRECTORY ARGUMENT IS NOT A FILE ARGUMENT
    -----------------------------------------------
    An independent translation audit reads copy from a checkout the session is
    not standing in: the session cwd is wherever the agent happens to be (often
    the home directory), while the branch under audit lives in a named clone.
    With no audited way to say which tree, that audit has only a redaction
    bypass — the workaround this module exists to replace.

    So the caller may name a DIRECTORY, admitted only when its ``origin`` remote
    reduces (through the same :func:`_origin_identity` the ``src/`` gate uses) to
    one of :data:`PUBLISHING_REPOSITORIES`.  Naming a checkout changes WHERE the
    untouched allowlist is applied, never WHAT may be read: the file argument
    stays repository-relative, :func:`_committed_public_copy` still serves the
    ``HEAD`` blob and still re-derives the identity of the root it reads, and
    the audit record still precedes the content.

    The spelling must be ABSOLUTE once ``~`` is expanded.  A relative one is
    resolved by ``Path.resolve()`` against the harness PROCESS's working
    directory — a directory the caller cannot see or predict — so ``repo_root:
    "branch-a"`` would audit a tree the caller never named and the receipt would
    record it as if they had.  Refusing is the only answer that keeps the receipt
    truthful (review m-2).

    Refusals name the reason, because the caller's next move differs: a NUL or a
    symlink is a malformed argument, a relative spelling is ambiguous, a
    non-directory is not a checkout, a directory outside a git working tree is
    not a clone at all, a missing origin cannot be shown to be a publishing
    checkout, and an unrecognised origin is a DIFFERENT repository.  The raw
    origin URL is never echoed (it can embed a credential); only the reduced
    identity reaches the message.

    WHAT THIS IDENTITY CHECK IS NOT
    -------------------------------
    It is a guard against ADMITTING AN UNRELATED TREE, not proof of provenance.
    It reads the repository's own ``origin`` configuration, so three commands
    (``git init``, ``git remote add origin <allowlisted>``, commit) manufacture a
    directory that passes it — a disclosure, not a bug to paper over: the route's
    real controls are the allowlist, the committed-blob rule and the audit
    record, and the same-UID residual in ``docs/security/redaction-policy.md`` §6
    already owns the rest.  What it does stop is the accident — auditing the
    wrong repository, or a ``src/i18n/`` tree that merely looks like copy.
    """
    if "\x00" in raw:
        return Path(), "", "repo_root must not contain NUL; it names a directory"
    try:
        spelled = Path(raw).expanduser()
    except RuntimeError:
        # ``~user`` naming nobody: ``expanduser`` raises rather than returning.
        return Path(), "", f"repo_root {raw} names no user this platform can resolve"
    if not spelled.is_absolute():
        return (
            Path(),
            "",
            (
                f"repo_root must be an absolute path; {raw!r} would resolve against the"
                " harness process's working directory, not this session's"
            ),
        )
    if spelled.is_symlink():
        # The identity check would describe the TARGET while the refusal the
        # caller reads would name the link; ask for the directory itself.
        return Path(), "", f"repo_root {raw} is a symlink; name the checkout directory itself"
    try:
        root = spelled.resolve()
    except (OSError, ValueError) as exc:
        return Path(), "", f"repo_root {raw} could not be resolved ({exc})"
    if not root.is_dir():
        return Path(), "", f"repo_root {root} is not a directory"

    def _git(*args: str) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            ["git", "-C", str(root), "--no-replace-objects", *args],
            capture_output=True,
            check=False,
            timeout=10,
        )

    try:
        inside = _git("rev-parse", "--is-inside-work-tree")
        if inside.returncode != 0 or inside.stdout.strip() != b"true":
            return Path(), "", f"repo_root {root} is not inside a git working tree"
        origin = _git("remote", "get-url", "origin")
    except (OSError, subprocess.SubprocessError) as exc:
        return Path(), "", f"repo_root {root} could not be inspected ({exc})"
    identity = (
        _origin_identity(origin.stdout.decode("utf-8", "replace")) if origin.returncode == 0 else ""
    )
    if not identity:
        return (
            Path(),
            "",
            (
                f"repo_root {root} has no usable git origin; a named checkout must be a"
                " clone of one of: " + ", ".join(PUBLISHING_REPOSITORIES)
            ),
        )
    if identity not in PUBLISHING_REPOSITORIES:
        return (
            Path(),
            "",
            (
                f"repo_root {root} is a checkout of {identity}, not one of the publishing"
                " repositories: " + ", ".join(PUBLISHING_REPOSITORIES)
            ),
        )
    return root, identity, None


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
    checkout: str | None = None,
    checkout_root: Path | None = None,
    checkout_head: str | None = None,
) -> Path:
    """Append one audit record; raise when it cannot be written.

    The caller turns the exception into a refused read.  Returning content
    without its record is the one outcome this route may not have: it would be
    a silent bypass with a policy name on it.

    ``checkout`` is present only for a read from a NAMED checkout.  Then the
    record also carries that root's path — two clones of the same repository
    share an identity and only the path says which tree served the bytes — and
    the ``checkout_head`` commit the blob came from, because the same path in the
    same worktree on two commits differs only by digest, and an audit record that
    cannot name the revision cannot be reconciled with the ledger row it was
    taken to support (review m-3).  A session-workspace read keeps exactly the
    fields it always had, so existing consumers of this log parse it unchanged.
    """
    path = _audit_path()
    session = Path(context.session_dir).name if context.session_dir else "unknown"
    record: dict[str, Any] = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tool": TOOL_NAME,
        "session": session,
        "path": relative,
        "key": key,
        "chars": chars,
        "sha256": digest,
        "policy": POLICY_DOCUMENT,
    }
    if checkout is not None:
        record["checkout"] = checkout
        record["checkout_root"] = str(checkout_root) if checkout_root is not None else None
        record["checkout_head"] = checkout_head
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


def _committed_public_copy(
    root: Path,
    relative: str,
    *,
    allowed_checkouts: tuple[tuple[str, str], ...] = (),
    required_identity: str | None = None,
    named_root: bool = False,
) -> tuple[bytes, str | None, str | None]:
    """The committed blob for ``relative``, its refusal, and the served ``HEAD``.

    ``HEAD`` is the committed copy, and this reader serves THAT blob rather than
    the working-tree file. Six consequences, all measured in review:

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
    * a consumer checkout's roots need checkout identity, not path shape (review
      S-1); ``allowed_checkouts`` gates that check here so all git probing
      shares one wrapper;
    * ``required_identity`` re-derives the identity HERE, in the same wrapper
      call that serves the bytes, so the identity a receipt records is the one
      the serving repository had — not the one an earlier admission probe saw
      (review m-4). The alternative, documenting the window instead, leaves a
      receipt that can be false in the log an audit is reconciled against, and
      the check costs one already-shared git call.

    ``named_root`` changes only the SENTENCES: a caller who named a checkout
    cannot "run read_public_copy from the checkout root", so that arm is told to
    name the root ``git rev-parse --show-toplevel`` reports (review n-1).

    The third element is the commit the blob was requested from — read in this
    same wrapper, immediately BEFORE the ``show``, so a commit landing mid-call
    makes the record understate the revision rather than overstate it; the body
    digest remains the exact answer for the bytes. It is ``None`` when git did
    not report one.

    The helper is synchronous and runs off the event loop
    (:func:`execute_read_public_copy` awaits it in a thread).
    """

    def _git(*args: str) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            ["git", "-C", str(root), "--no-replace-objects", *args],
            capture_output=True,
            check=False,
            timeout=10,
        )

    subject = "the named checkout" if named_root else "the workspace"
    not_a_checkout = f"{subject} is not a git checkout"
    try:
        inside = _git("rev-parse", "--is-inside-work-tree")
        if inside.returncode != 0 or inside.stdout.strip() != b"true":
            return b"", not_a_checkout, None
        top = _git("rev-parse", "--show-toplevel")
        if top.returncode != 0 or not top.stdout.strip():
            return b"", not_a_checkout, None
        checkout_root = Path(top.stdout.decode("utf-8", "replace").strip()).resolve()
        if checkout_root != root:
            if named_root:
                return (
                    b"",
                    (
                        f"the named checkout root {root} is not the checkout root"
                        f" {checkout_root}; name the directory that"
                        " `git rev-parse --show-toplevel` reports, not one below it"
                    ),
                    None,
                )
            return (
                b"",
                (
                    f"the workspace root {root} is not the checkout root {checkout_root}; "
                    "run read_public_copy from the checkout root"
                ),
                None,
            )
        if allowed_checkouts or required_identity is not None:
            origin = _git("remote", "get-url", "origin")
            identity = (
                _origin_identity(origin.stdout.decode("utf-8", "replace"))
                if origin.returncode == 0
                else ""
            )
            if required_identity is not None and identity != required_identity:
                return (
                    b"",
                    (
                        f"the origin of {root} is now {identity or 'unusable'}, not the"
                        f" {required_identity} this call admitted; name the checkout again"
                    ),
                    None,
                )
            if allowed_checkouts and identity not in {
                allowable for allowable, _ in allowed_checkouts
            }:
                return b"", _checkout_refusal(relative, allowed_checkouts, identity), None
        tracked = _git("ls-files", "--error-unmatch", "--", relative)
        if tracked.returncode != 0:
            return b"", f"{relative} is not committed at HEAD; commit the public copy first", None
        clean = _git("diff", "--quiet", "HEAD", "--", relative)
        if clean.returncode == 1:
            return (
                b"",
                f"{relative} differs from HEAD; commit the public copy before auditing it",
                None,
            )
        if clean.returncode != 0:
            return b"", f"could not compare {relative} with HEAD", None
        head = _git("rev-parse", "HEAD")
        served = head.stdout.decode("utf-8", "replace").strip() if head.returncode == 0 else None
        blob = _git("show", f"HEAD:{relative}")
        if blob.returncode != 0:
            return b"", f"{relative} is not committed at HEAD; commit the public copy first", None
        return blob.stdout, None, served or None
    except (OSError, subprocess.SubprocessError) as exc:
        return b"", f"could not read the committed copy of {relative} ({exc})", None


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
    repo_root_arg: str | None = None
    if arguments.get("repo_root") is not None:
        repo_root_arg = str(arguments["repo_root"])
        if not repo_root_arg:
            # An empty string is a caller that meant a checkout and did not name
            # one; falling back to the session workspace would read a tree they
            # did not ask for.
            return _text(tool_call_id, "repo_root must name a directory", is_error=True)

    root = Path(context.cwd).expanduser().resolve()
    checkout: str | None = None
    if repo_root_arg is not None:
        # OFF THE EVENT LOOP. Admission shells out to `git` up to three times;
        # a checkout on a stalled mount would otherwise freeze streaming, the UI
        # and every concurrent tool call for the length of its timeouts (review
        # M-1 measured a 4.0 s heartbeat gap on a 2 s-per-call git stub). The
        # cwd arm never probed git here, so this is the only call site.
        admitted, checkout, refusal = await asyncio.to_thread(_admitted_checkout, repo_root_arg)
        if refusal is not None:
            return _text(tool_call_id, f"refused: {refusal}", is_error=True)
        root = admitted
    # One resolver for both arms: ``_resolve_workspace_path`` is the single
    # normalisation (``~``, ``..``, symlinks, NUL) the rest of the tool surface
    # uses, and the admitted checkout replaces the session cwd as its root.
    resolved, inside, resolvable = _resolve_workspace_path(path_arg, str(root))
    if not resolvable or not inside:
        return _text(
            tool_call_id,
            (
                "refused: the path is not inside this session's workspace"
                if checkout is None
                else f"refused: the path is not inside the named checkout {root}"
            ),
            is_error=True,
        )
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError:
        return _text(
            tool_call_id,
            (
                "refused: the path escapes the workspace"
                if checkout is None
                else f"refused: the path escapes the named checkout {root}"
            ),
            is_error=True,
        )
    if not _matches_public_copy(relative):
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
    blob, refusal, checkout_head = await asyncio.to_thread(
        _committed_public_copy,
        root,
        relative,
        allowed_checkouts=_publishing_checkouts(relative),
        # For a named root the identity is re-derived HERE, in the call that
        # serves the bytes, so a directory swapped between admission and the read
        # is refused instead of served under the admitted identity's name (m-4).
        required_identity=checkout,
        named_root=checkout is not None,
    )
    if refusal is not None:
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
            checkout=checkout,
            checkout_root=root if checkout is not None else None,
            checkout_head=checkout_head,
        )
    except OSError as exc:
        # FAIL CLOSED. A route whose use is not recorded is not this route.
        return _text(
            tool_call_id,
            f"refused: the audit record could not be written ({exc})",
            is_error=True,
        )

    details: dict[str, Any] = {
        "path": relative,
        "key": key,
        "public_copy": True,
        "policy": POLICY_DOCUMENT,
        "audit_log": str(audit_file),
        "sha256": digest,
    }
    if checkout is not None:
        # The receipt says WHICH tree served the bytes and WHICH revision of it,
        # not merely that a tree did.
        details["checkout"] = checkout
        details["checkout_root"] = str(root)
        details["checkout_head"] = checkout_head
    return _text(tool_call_id, body, details=details)


def build_read_public_copy_tool() -> AgentTool:
    """The tool definition advertised to the model."""
    return AgentTool(
        name=TOOL_NAME,
        label="Read public copy",
        description=(
            "Read committed public i18n copy through the audited route: a catalogue "
            "key, a key-module line, or a generated key map. Allowlisted; every read "
            "is recorded. repo_root names a publishing repository's checkout to read "
            "when the session cwd is not one (origin must be "
            + ", ".join(PUBLISHING_REPOSITORIES)
            + "); the path stays relative and allowlisted."
        ),
        parameters=ReadPublicCopyParams.model_json_schema(),
        approval_tier="read",
        concurrency="shared",
        interruptible=False,
        execute=execute_read_public_copy,
    )
