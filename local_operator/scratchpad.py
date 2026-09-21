"""``scratchpad://`` — this session's own scratch area, addressed as a URL.

Why this module exists
----------------------
An agent needs somewhere to put its OWN intermediate work: a benchmark's raw
numbers, a scratch list, a throwaway script, a long tool result it wants to keep
across turns. Without a designated place it litters the user's working
directory, and the litter is indistinguishable from an output the user asked
for. ``scratchpad://`` names that place: one scheme served by the EXISTING
``read``/``write``/``edit`` tools (footprint rung 1 — no new tool, no new
schema), rooted in the session directory every host already creates.

The name is the AREA, not an artefact type. ``notes`` could not cover a ``.csv``
or a one-off ``.py``, and it reads as the deliverable the user asked for;
``scratch`` alone is already this codebase's word for the OS temp directory.
``scratchpad`` is the ecosystem's word for a session-scoped, user-browsable
folder the agent works in and keeps off the user's disk.

Scratch belongs to the SESSION, not the user and not the process: it lives at
``<session dir>/scratchpad`` so the existing session cleanup and retention
passes reclaim it with zero new deletion code. This module therefore contains
no removal of any kind; see ``tests/unit/session/test_no_session_deletion.py``.

Grammar and its two layers of safety
------------------------------------
The URL is validated twice, and the second layer is the authority:

1. Segment validation — absolute paths, ``..`` segments and dotfiles are
   refused by name, so the common mistake gets a sentence that says what is
   wrong. The shape mirrors ``local_operator/skills/protocol.py`` so the
   harness keeps ONE grammar for URL-addressed trees.
2. ``resolve()`` + containment — the joined path is fully resolved and must
   stay under the resolved scratchpad root. This is what actually stops a
   SYMLINK inside the folder pointing at ``~/.ssh``, and what keeps the Windows
   backslash case safe. It deliberately does NOT go through
   ``_resolve_workspace_path``, which is where the ``[outside workspace]``
   approval escalation lives: a scratch file is outside the working directory
   by construction and must never raise that escalation.

The name lives in ONE place
---------------------------
:data:`SCRATCHPAD_NAMESPACE` is the only occurrence of the name in this module's
logic: the URL prefix and the on-disk directory name are both derived from it,
and the parser compares the URL's scheme against it rather than against a
hardcoded literal. Renaming the protocol is that token plus the prose in the
guide and the system prompt.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import NamedTuple
from urllib.parse import unquote, urlsplit

from local_operator.session.retention import SESSIONS_DIRNAME

#: The name of the scheme, and the ONLY place it appears in code. Everything
#: else — the URL prefix, the directory name, the messages below — is derived
#: from it, so a rename cannot leave the tools reading one address while
#: writing another.
SCRATCHPAD_NAMESPACE = "scratchpad"

#: The URL prefix ``read``/``write``/``edit`` recognise. Compared with
#: ``str.startswith`` on the raw argument, which is what keeps the branch cheap
#: enough to sit above the generic internal-URL catch-all.
SCRATCHPAD_SCHEME = f"{SCRATCHPAD_NAMESPACE}://"

#: Directory name under a session directory. Derived from the same token.
SCRATCHPAD_DIRNAME = SCRATCHPAD_NAMESPACE

#: The environment variable that carries this session's scratchpad ROOT into a
#: child process — the ``bash`` tool's shell and the ``eval`` worker — as an
#: absolute path.
#:
#: WHY A PATH AND NOT JUST THE SCHEME. An agent needs an absolute path BEFORE it
#: can create anything: ``mktemp -d``, ``nohup … > log``, ``sys.path.insert(0,
#: dir)``. With ``scratchpad://`` alone the only way to learn that path is to
#: write a file first and read the receipt — one extra round trip that ``/tmp``
#: does not cost, and the reason the scheme lost the shell channel by ~9:1
#: (measured 2026-09-21 over 400 transcripts: 8,766 shell calls created scratch
#: under a temp root, against 44 that reached the scratchpad).
#:
#: THREE ARMS, the same rule ``agent_shell.MAY_DELEGATE_ENV`` is signed with,
#: and signed from the same spawn sites. Set to this session's root when it has
#: one; CLEARED (the empty string) when the name is inherited from the launcher
#: and this session has none; NOT WRITTEN AT ALL otherwise. The clear is what
#: keeps a nested session from writing into its PARENT's scratchpad: a child's
#: environment starts as a copy of the harness's own in ``shell_env``'s default
#: ``inherit`` mode, so an inherited path would survive a child that has no
#: scratchpad of its own and hand it somewhere it must not write. Unlike the
#: delegation allowance the NAME is not a mechanism — knowing it grants nothing —
#: so the omit arm exists only so a session without a scratchpad is not handed a
#: variable it would then read as empty and have to interpret.
SCRATCHPAD_PATH_ENV = "LOCAL_OPERATOR_SCRATCHPAD"

#: The one message every scratchpad operation returns on a host that has no
#: session directory. The fallback it names is the guide's: a real temporary
#: directory, NEVER the working directory — "just put it in the workspace" is
#: the littering this whole scheme exists to prevent, so the refusal must not
#: recommend it.
SCRATCHPAD_UNAVAILABLE = (
    f"{SCRATCHPAD_SCHEME} is unavailable here: this session has no scratchpad "
    "directory (a host with no session, or an --train agent directory). Keep scratch "
    "out of the user's working directory: use a temporary directory instead "
    "(`bash mktemp -d`), or ask where the user wants the file."
)


class ScratchpadPathError(ValueError):
    """A ``scratchpad://`` URL that is malformed or escapes the scratchpad root.

    A ``ValueError`` so callers that only guard broad shapes still catch it,
    and a named class so the tools can turn it into an ``invalid arguments``
    result (the model's fault) instead of a failure (the machine's).
    """


class ScratchpadTarget(NamedTuple):
    """One parsed scratchpad URL. ``path`` is absolute and inside the root."""

    path: Path
    #: The URL named a directory: the bare root, ``scratchpad://.``, or a
    #: trailing ``/``. ``read`` turns this into a listing; the mutating tools
    #: refuse it, because a file needs a name.
    directory: bool


def scratchpad_root(session_dir: Path | str | None) -> Path | None:
    """``<session dir>/scratchpad`` for a session-store directory, else ``None``.

    ``None`` for a directory that is not directly under ``sessions/``. An
    ``--train`` run keeps its transcript in ``<config_dir>/agents/<id>/``, and
    ``AgentRegistry.export_agent`` zips an agent directory whole and publishes
    it to the Agent Hub — a scratch folder there would ship to strangers. The
    predicate is the one retention already uses (``session/retention.py``), so
    "is this a session store directory" has a single answer on this machine.
    """
    if session_dir is None:
        return None
    try:
        directory = Path(session_dir)
    except TypeError:
        # A host that passed something path-like-but-not: treat it as absent
        # rather than failing the turn, matching how the session's id
        # derivation tolerates a transcript with no directory.
        return None
    if directory.parent.name != SESSIONS_DIRNAME:
        return None
    return directory / SCRATCHPAD_DIRNAME


def scratchpad_dir_of(context: object | None) -> str | None:
    """The scratchpad root a live tool context carries, or ``None``.

    ``""`` is treated as absent rather than as a path: ``Path("")`` is the cwd,
    which would silently make the whole working directory the session's scratch
    area. Duck-typed rather than annotated as ``ToolContext`` because the
    ``tests/e2e`` doubles are not one, and a bare attribute access would make
    them raise.
    """
    raw = getattr(context, "scratchpad_dir", None)
    if not isinstance(raw, str) or not raw:
        return None
    return raw


def scratchpad_env_injection(scratchpad_dir: str | None) -> dict[str, str]:
    """The three-arm :data:`SCRATCHPAD_PATH_ENV` write for a child environment.

    See :data:`SCRATCHPAD_PATH_ENV` for why the arms are what they are. The root
    is a PARAMETER rather than a read of this process's environment because the
    caller is the code that knows which session it is spawning for — one process
    holds several child sessions, and the ``eval`` worker in particular is spawned
    per ``session_key``. The only environment read here is the presence test that
    chooses between the clear and the omit, and that test is exactly the right
    one: it asks whether THIS process inherited the name from whatever launched
    it, which is the fact that decides whether a child would carry a stale path
    with nothing to clear it. An injection survives ``shell_env``'s strict mode
    by construction, so the arm that is written is the arm the child sees.
    """
    if scratchpad_dir:
        return {SCRATCHPAD_PATH_ENV: str(scratchpad_dir)}
    if SCRATCHPAD_PATH_ENV in os.environ:
        return {SCRATCHPAD_PATH_ENV: ""}
    return {}


def ensure_scratchpad_dir(scratchpad_dir: str | None) -> str | None:
    """``scratchpad_dir`` with its directory created, or ``None`` when there is none.

    Called by the two places that HAND A PAD TO A CHILD — the ``bash`` tool and the
    ``eval`` worker — so the path those processes are told is one they can write to
    on the first try. It has to be created somewhere, and this is the only moment
    that both needs it and can be sure the session directory exists: the tools that
    take the scheme (``write``/``edit``) make their own parents, while a SHELL
    cannot, so before this a fresh session's first ``> "$LOCAL_OPERATOR_SCRATCHPAD/
    x.log"`` was ``No such file or directory`` and ``mktemp -d
    "$LOCAL_OPERATOR_SCRATCHPAD/rig.XXXXXX"`` was ``mkdtemp failed`` — at exactly
    the moment the export exists to keep the work out of ``/tmp``, so the recovery a
    model reaches for under that error was the behaviour the export prevents.
    Measured when this was found: 420 of this machine's 8,109 session directories
    had no ``scratchpad/`` at all, and ``read scratchpad://`` answers ``(0 entries)``
    for a missing root rather than an error, so nothing surfaced it.

    Deliberately NOT done where the session derives the path
    (``Session._scratchpad_dir``): that runs during construction, and
    ``Transcript(defer_materialise=True)`` exists so a speculative runtime leaves
    nothing on disk — a mkdir there breaks a pinned invariant
    (``test_birth_selection_is_durable_only_when_work_is_admitted``).

    Idempotent (``exist_ok=True`` is one ``mkdir`` syscall beside a spawn that is
    orders of magnitude dearer), and a failure is SWALLOWED: a session whose
    directory cannot be written to must still get its command run — the tool that
    uses the path reports its own error — and a mkdir is not a reason to refuse.
    """
    if not scratchpad_dir:
        return None
    with contextlib.suppress(OSError):
        Path(scratchpad_dir).mkdir(parents=True, exist_ok=True)
    return scratchpad_dir


def parse_scratchpad_url(url: str, root: Path) -> ScratchpadTarget:
    """Resolve one ``scratchpad://`` URL; raise :class:`ScratchpadPathError`.

    ``root`` is the scratchpad root itself (what :func:`scratchpad_root`
    returns), not the session directory.
    """
    parts = urlsplit(url)
    # Compared against the CONSTANT, never a literal: a hardcoded spelling here
    # is what makes a rename half-apply, leaving every URL rejected with an
    # error that reads as nonsense.
    if parts.scheme != SCRATCHPAD_NAMESPACE:
        raise ScratchpadPathError(f"Invalid scratchpad URL '{url}': not a {SCRATCHPAD_SCHEME} URL")
    if not url.startswith(SCRATCHPAD_SCHEME):
        # ONE case rule, and it is the caller's: this harness dispatches on the
        # typed prefix (`str.startswith` in the tools), so a URL that only
        # *parses* as the right scheme — ``SCRATCHPAD://x``, which urlsplit
        # lower-cases — is refused HERE naming the spelling it should have used,
        # rather than being accepted here and then rejected elsewhere as a
        # stranger scheme, which read as a contradiction (review round 1, R7).
        raise ScratchpadPathError(
            f"Invalid scratchpad URL '{url}': the scheme must be written exactly "
            f"'{SCRATCHPAD_SCHEME}' — lower-case, with the '//'"
        )
    if parts.query or parts.fragment:
        # A '?' or '#' would silently truncate the name, so the file the agent
        # asked for and the file it got would differ with no signal.
        raise ScratchpadPathError(
            f"Invalid scratchpad URL '{url}': '?' and '#' open a query or fragment; "
            "percent-encode them ('%3F', '%23') to name a file that contains one"
        )
    # Unquote BEFORE splitting, so an encoded separator ('%2F') becomes a real
    # segment boundary and is validated as one rather than riding through as
    # part of a file name.
    raw = unquote(parts.netloc + parts.path)
    if raw.startswith(("/", "\\")):
        raise ScratchpadPathError(f"Invalid scratchpad URL '{url}': absolute paths are not allowed")
    if "://" in raw:
        # A scheme in the REMAINDER is a URL inside a URL: ``scratchpad://notes://x``
        # would otherwise be accepted and materialise ``<root>/notes:/x`` — a
        # directory named after a URL that means something other than it spells,
        # inside the one folder this module promises holds the agent's own files
        # (round 2, Q6). The unquote above already treats ``%2F`` as a real
        # segment boundary; this is that rule for the scheme separator, and it is
        # why the test is on ``://`` and not on ``:`` — a single colon is a legal
        # POSIX filename character (``a:b.txt`` writes, and a test pins that).
        nested = raw.split("://", 1)[0].lstrip("/") or raw
        raise ScratchpadPathError(
            f"Invalid scratchpad URL '{url}': '{nested}://' is a URL, not a file name; "
            f"a {SCRATCHPAD_SCHEME} URL takes a plain name or path after the scheme"
        )
    segments = [segment for segment in raw.split("/") if segment not in ("", ".")]
    if any(segment == ".." for segment in segments):
        raise ScratchpadPathError(f"Invalid scratchpad URL '{url}': '..' segments are not allowed")
    if any(segment.startswith(".") for segment in segments):
        # Dotfiles are neither listed nor read: the listing skips them, so
        # accepting one here would let the agent read a name it can never see.
        raise ScratchpadPathError(
            f"Invalid scratchpad URL '{url}': dotfiles are not listed and cannot be read"
        )

    base = _resolve_scratchpad(root, url)
    target = _resolve_scratchpad(base.joinpath(*segments), url) if segments else base
    if not target.is_relative_to(base):
        # The authority for traversal: reached when a component of the path (a
        # symlink, on POSIX) resolves outside the root even though every
        # segment was legal.
        raise ScratchpadPathError(
            f"Invalid scratchpad URL '{url}': escapes the scratchpad directory"
        )
    # The directory marker is read off the UNQUOTED string, not off
    # ``parts.path``: ``scratchpad://logs%2F`` is a directory URL whose last
    # character is not a slash until it is unquoted, so testing the quoted form
    # silently answered "that file does not exist" for a folder (R6).
    return ScratchpadTarget(target, directory=not segments or raw.endswith("/"))


def _resolve_scratchpad(path: Path, url: str) -> Path:
    """``Path.resolve()`` with every failure mode folded into the parse error.

    A symlink loop raises ``RuntimeError`` on 3.12/3.13 and returns an
    unresolved path on 3.14, a missing parent raises nothing at all, and a NUL
    byte in a segment raises ``ValueError`` — so the failure the containment
    check must not miss is normalised HERE rather than allowed to escape as a
    bare ``OSError`` (or, for the NUL, a traceback and an execution fault) from
    inside ``read``. ``_resolve_workspace_path`` catches the same triple for the
    same reason (review round 1, R2/Q2).
    """
    try:
        return path.resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ScratchpadPathError(
            f"Invalid scratchpad URL '{url}': cannot be resolved ({exc})"
        ) from exc
