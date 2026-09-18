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
    return ScratchpadTarget(target, directory=not segments or parts.path.endswith("/"))


def _resolve_scratchpad(path: Path, url: str) -> Path:
    """``Path.resolve()`` with both failure modes folded into the parse error.

    A symlink loop raises ``RuntimeError`` on 3.12/3.13 and returns an
    unresolved path on 3.14, and a missing parent raises nothing at all — so the
    failure the containment check must not miss has to be normalised here rather
    than allowed to escape as a bare ``OSError`` from inside ``read``.
    """
    try:
        return path.resolve()
    except (OSError, RuntimeError) as exc:
        raise ScratchpadPathError(
            f"Invalid scratchpad URL '{url}': cannot be resolved ({exc})"
        ) from exc
