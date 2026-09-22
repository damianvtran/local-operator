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


class ScratchpadContentError(ValueError):
    """A well-formed URL pointed at material the pad does not keep.

    A DIFFERENT fault from :class:`ScratchpadPathError`, and the distinction is
    the reason for a second class. That one is about the ADDRESS: a URL that is
    malformed or escapes the root, refused whatever happens to be at the end of
    it. This one is about the CONTENT: the address is valid and resolves inside
    the root, and what the caller wants to put there is build output, a
    dependency tree or an archive rather than scratch. There is always an
    alternative to name here — the material has a home, it simply is not this
    one — so every message raised with this class must say WHERE the material
    belongs. A path error has no such alternative to offer, which is why the
    two must not be merged into one refusal.

    A ``ValueError``, like the path error and for the same two reasons: a
    caller that only guards broad shapes still catches it, and it is the
    model's fault, so the tools turn it into an ``invalid arguments`` result
    rather than a failure.
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


# ---------------------------------------------------------------------------
# Content policy: the pad keeps scratch, not build output
# ---------------------------------------------------------------------------
# Why a policy is here at all, measured on this machine 2026-09-22: the session
# store's scratchpads held 34.8 GB, and 34.6 GB of that belonged to sessions
# which had already ENDED. The bytes were not the notes, scripts and rendered
# frames the scheme exists for. They were 7.6 GB of compiled output
# (.o/.map/.a), 6.5 GB of node_modules, 5.7 GB of extensionless binaries and
# 2.6 GB of git packs, against 3.4 GB of intended material. Nothing refused any
# of it, so a build tree written through the pad cost exactly what a note cost —
# and because a pad is reclaimed only when its session goes away, that cost was
# paid in full and then thrown away: the material was unreachable the moment the
# session ended. Hence a REFUSAL at the write, naming where the material belongs
# instead. Not a size report afterwards, and never a deletion — this module
# removes nothing (see its docstring), and a policy that deleted would be free to
# reap a pad whose owner is still working in it.
#
# The test is a SHAPE, in the name and in the size, and deliberately nothing
# else, because that is all a refusal can rest on without reading the payload: a
# segment shaped like a build or dependency tree, a basename shaped like a
# compiled artefact, an archive, a model or a versioned library, a payload too
# large for one write, or a pad already holding too much for any. The shapes are
# MATCHED rather than listed because a list is a guess about shape, and the 34.8
# GB was made of the guesses nobody made (``cmake-build-debug/``, ``bazel-out/``,
# ``_build/``, ``*-cache/``, a versioned ``libfoo.so.1.2``); the pad-total ceiling
# below is the backstop for the shapes no name rule can reach at all, the 4.94 GB
# ``node_modules`` included. Everything the pad is FOR survives — a 20 MB shaped
# CSV, a rendered frame, notes, a one-off script, a benchmark table.

#: Path SEGMENTS refused wherever they appear BELOW the pad root: the build
#: trees, dependency stores and caches of the toolchains this fleet runs. These
#: are the COMMON CASES of the shape rule below rather than the rule itself —
#: kept as names because they are what a caller actually types, so the refusal
#: can name the directory it found.
#:
#: Ambiguous names are left OUT on purpose (``objects``, ``bin``, ``lib``): a
#: refusal that fires on a name an agent legitimately meant as scratch teaches
#: it to route around the pad, which is how the user's own working tree ends up
#: holding the litter instead.
#:
#: The dot-prefixed entries are already unreachable through a URL — the parser
#: refuses every dotfile segment — and are listed anyway so the policy is stated
#: in one place for a path that arrives from anywhere else.
#:
#: Membership is CASE-INSENSITIVE, because the volume it runs on is not: APFS is
#: case-insensitive, so ``NODE_MODULES`` IS ``node_modules`` on disk, the same
#: directory one keystroke away, and a spelling-exact rule would refuse only the
#: spelling the author happened to type. No legitimate scratch name differs from
#: a refused one by case alone, so the fold costs no false refusal — and both
#: arms of the policy fold, so neither can drift from the other.
SCRATCHPAD_REFUSED_SEGMENTS: frozenset[str] = frozenset(
    {
        "node_modules",
        "bower_components",
        "vendor",
        "target",
        "build",
        "dist",
        "out",
        "obj",
        ".next",
        ".nuxt",
        ".svelte-kit",
        ".turbo",
        ".parcel-cache",
        ".pnpm-store",
        ".venv",
        "venv",
        "site-packages",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".gradle",
        ".m2",
        "Pods",
        "DerivedData",
        ".terraform",
        ".lake",
        ".git",
    }
)

#: :data:`SCRATCHPAD_REFUSED_SEGMENTS` folded, which is what membership is
#: actually tested against. The set above keeps each toolchain's own spelling
#: because that is how it is read; folding here rather than spelling the names
#: twice in lower case is what keeps the two from drifting apart.
_REFUSED_SEGMENTS_FOLDED: frozenset[str] = frozenset(
    name.lower() for name in SCRATCHPAD_REFUSED_SEGMENTS
)

#: The same tokens as BOUNDARY prefixes: a segment that IS a token, or that
#: BEGINS with one up to a ``.`` (``build.old``, ``node_modules.bak``).
#:
#: WHY A BOUNDARY, AND WHY NOT THE HYPHEN. The shapes a name-exact list misses
#: are the QUALIFIED trees, and a dot is the one separator a hyphenated word does
#: not continue through. A hyphen does, in both directions: ``build-report.csv``
#: and ``node_modules-notes.md`` are a report and a note that merely begin with a
#: token and must stay scratch, while the hyphenated BUILD trees are caught as the
#: qualifiers they end in (below) or as one of the two lead-ins (below that).
#: ``build-debug`` therefore stays allowed, and that is the known cost of the
#: boundary rather than an oversight.
_REFUSED_SEGMENT_BOUNDARIES: tuple[str, ...] = tuple(
    f"{name}." for name in sorted(_REFUSED_SEGMENTS_FOLDED)
)

#: A segment that ENDS WITH one of these: a build tree or a cache QUALIFIED by
#: what it is (``cmake-build-debug``'s siblings — ``foo-build``, ``_build``,
#: ``repo-cache``), the two packaging metadata directories, and a dSYM.
#:
#: ``.dsym`` belongs here rather than among the file suffixes below because a dSYM
#: is a DIRECTORY: the suffix arm judges the basename alone, so ``Foo.app.dSYM/…``
#: would be allowed by its leaf, while the segment rule sees the bundle itself at
#: whatever depth it appears.
SCRATCHPAD_REFUSED_SEGMENT_SUFFIXES: tuple[str, ...] = (
    "-build",
    "_build",
    "-cache",
    "-out",
    "-dist",
    ".egg-info",
    ".dist-info",
    ".dsym",
)

#: A segment that STARTS WITH one of these, exactly: the two build systems whose
#: trees are qualified with a HYPHEN — ``cmake-build-debug``, ``bazel-out``,
#: ``bazel-bin`` — which is the one continuation the boundary rule above allows
#: on purpose for an ordinary name.
SCRATCHPAD_REFUSED_SEGMENT_LEADINS: tuple[str, ...] = ("cmake-build-", "bazel-")


def _is_refused_segment(segment: str) -> bool:
    """Whether one path segment is shaped like a build tree or a dependency store."""
    folded = segment.lower()
    return (
        folded in _REFUSED_SEGMENTS_FOLDED
        or folded.startswith(SCRATCHPAD_REFUSED_SEGMENT_LEADINS)
        or folded.endswith(SCRATCHPAD_REFUSED_SEGMENT_SUFFIXES)
        or folded.startswith(_REFUSED_SEGMENT_BOUNDARIES)
    )


#: File-suffix TOKENS refused below the pad root: compiled artefacts, build
#: intermediates, archives, disk images, executables and model weights. None of
#: these has a reading as scratch, and the non-image ones have no text to return
#: through the scheme either (``read`` refuses their bytes as text), so a pad
#: copy is unreadable as well as heavy.
#:
#: Compared case-insensitively, like the segments above: an extension is a
#: well-known token whose canonical spelling is lower case, and the same archive
#: arrives spelled ``ZIP`` from whatever tool upper-cased it.
SCRATCHPAD_REFUSED_SUFFIXES: frozenset[str] = frozenset(
    {
        ".o",
        ".obj",
        ".a",
        ".lib",
        ".so",
        ".dylib",
        ".dll",
        ".rlib",
        ".rmeta",
        ".class",
        ".jar",
        ".pyc",
        ".pyo",
        ".whl",
        ".egg",
        ".zip",
        ".tar",
        ".tgz",
        ".gz",
        ".bz2",
        ".xz",
        ".zst",
        ".7z",
        ".rar",
        ".dmg",
        ".iso",
        ".img",
        ".bin",
        ".exe",
        ".wasm",
        ".onnx",
        ".pt",
        ".pth",
        ".safetensors",
        ".gguf",
        ".model",
        ".ckpt",
    }
)

#: The COMPOUND suffixes a single token cannot express, kept apart so the refusal
#: can name the archive the caller actually typed — ``foo.tar.gz``, not ``foo.gz``,
#: which the plain ``.gz`` would refuse too while naming the wrong half of it.
SCRATCHPAD_REFUSED_MULTIPART_SUFFIXES: tuple[str, ...] = (
    ".tar.gz",
    ".tar.bz2",
    ".tar.xz",
    ".tar.zst",
)

#: :data:`SCRATCHPAD_REFUSED_SUFFIXES` in a FIXED order and after the compound
#: ones, because the arm returns the first token that matches and the message
#: names that token — and a ``frozenset`` has no order to depend on.
_REFUSED_SUFFIXES_ORDERED: tuple[str, ...] = SCRATCHPAD_REFUSED_MULTIPART_SUFFIXES + tuple(
    sorted(SCRATCHPAD_REFUSED_SUFFIXES)
)


def _refused_suffix(basename: str) -> str | None:
    """The refused suffix token ``basename`` ends with, or ``None``.

    Judged on the case-folded BASENAME rather than on ``Path.suffix``, which two
    of the shapes this policy exists for both defeat: ``Path('foo.tar.gz').suffix``
    is ``.gz`` alone, and ``Path('libfoo.so.1.2').suffix`` is ``.2``, so a
    suffix-exact rule never sees the ``.tar`` or the ``.so`` at all. The versioned
    spelling is reached by stripping trailing dot-separated digit groups first
    (``libfoo.so.1.2`` -> ``libfoo.so``), and the strip is also why ``notes.2`` and
    ``rows.csv.1`` stay scratch: what is left after it is what gets judged, and it
    is not refused. A versioned DYLD needs no help from the strip —
    ``libbar.1.dylib`` still ends in the token.
    """
    folded = basename.lower()
    for name in (folded, _without_version_groups(folded)):
        for token in _REFUSED_SUFFIXES_ORDERED:
            if name.endswith(token):
                return token
    return None


def _without_version_groups(name: str) -> str:
    """``name`` without trailing ``.digits`` groups: ``libfoo.so.1.2`` -> ``libfoo.so``."""
    while (dot := name.rfind(".")) > 0 and name[dot + 1 :].isdigit():
        name = name[:dot]
    return name


#: The most a single ``write`` may put in a pad, in bytes. Set far above real
#: scratch on purpose — the largest artefact the audit found that the scheme
#: exists FOR was a shaped CSV orders of magnitude below it, and a rendered frame
#: is smaller still — so this catches a dump rather than shaping legitimate work.
#: A per-write ceiling; the pad's own total (below) is the other half of the size
#: policy.
SCRATCHPAD_MAX_WRITE_BYTES = 32 * 1024 * 1024

#: The most a pad may HOLD, in bytes, before every further write is refused.
#:
#: This is the BACKSTOP, and it is the arm that does not guess: every name rule
#: above is an inference from a shape, and the 34.8 GB was mostly shapes nobody
#: had inferred. A total is a fact about the pad itself, so it catches whatever
#: filled it — including the 4.94 GB ``node_modules`` even had its name never
#: been listed. Set far above any pad this scheme exists for (the intended
#: material across the WHOLE fleet's pads measured 3.4 GB before this policy, so
#: a quarter of a gigabyte in ONE pad is still generous), because a backstop that
#: refuses real work is one the fleet routes around.
SCRATCHPAD_TOTAL_BUDGET_BYTES = 256 * 1024 * 1024

#: How many directory entries the budget walk visits before it stops and refuses
#: anyway. The walk runs on EVERY write and its input is a tree the caller may have
#: filled by accident, so it must be BOUNDED — and the bound is an entry count
#: rather than a timeout because a count is the same on every machine.
#:
#: 20,000 is two orders of magnitude past the tens of entries a pad holds, so a
#: walk that reaches it has found a tree, which is the shape being refused; the
#: WORK is bounded with it, measured at 11.1 us per entry here (2026-09-22, 20,000
#: entries in one warm APFS directory: 222.9 ms), i.e. 0.22 s at the cap and
#: microseconds for a pad that keeps working.
SCRATCHPAD_BUDGET_SCAN_ENTRIES = 20_000

#: The sentence EVERY content refusal ends with: where the material DOES belong.
#: A refusal that only says no sends the agent to the next-worst place — the
#: user's working directory — which is the littering the scheme exists to
#: prevent, so the alternative is part of the refusal rather than a note beside
#: it. It is built to read after every arm: the segments, the extensions, the
#: ceiling and the unplaceable path each end with it and add no tail of their own,
#: because a second copy of the alternatives inline is a second copy that drifts.
#: Hence it names each home by the material rather than by the rule that refused
#: it — a large shaped extract is not build output, and a sentence that called it
#: that would send the caller to the wrong place. ``mktemp -d`` with no template
#: lands in ``$TMPDIR``, NOT ``/tmp``: macOS's ``com.apple.tmp_cleaner`` prunes
#: ``/tmp`` entries older than three days, so a template carrying that directory
#: is the one way to put a live session's scratch under a reaper.
SCRATCHPAD_ELSEWHERE = (
    "The pad is for scratch you will read back in this session: a dependency tree or build "
    "output belongs in a git worktree (`git worktree add <path>`), and a throwaway binary, "
    "an archive or an oversized extract in `bash mktemp -d`, which lands in $TMPDIR rather "
    "than /tmp."
)


def _relative_parts(path: Path, root: Path) -> tuple[str, ...] | None:
    """The segments of ``path`` BELOW ``root``, or ``None`` when they cannot be related.

    The comparison must judge only what is below the root, because ``path`` is
    absolute and a pad routinely sits under a directory that is itself named
    like a refused one — a session store inside a ``build/`` tree, a worktree
    under ``target/`` — and judging the whole path would refuse every write in
    that pad.

    The resolved second attempt is for the other direction, and it is not
    hypothetical: ``_scratchpad_root`` returns the context's string as a plain
    ``Path`` while the parsed target is ``Path.resolve()``d, so a session
    directory reached through a symlink (``/tmp`` is ``/private/tmp`` on macOS)
    shares no textual prefix with it.

    When neither attempt relates the two the answer is ``None``, and the caller
    REFUSES — never the bare file name. Returning ``(path.name,)`` there fails
    OPEN for the segment arm: ``node_modules/x.js`` would be judged on ``x.js``
    alone and allowed, which is exactly the outcome this function exists to
    prevent. ``parse_scratchpad_url`` has already proven containment before the
    check runs, so ``None`` is unreachable through the tools and is here as
    defence in depth for a direct caller.
    """
    try:
        return path.relative_to(root).parts
    except ValueError:
        pass
    try:
        return path.resolve().relative_to(root.resolve()).parts
    except (OSError, RuntimeError, ValueError):
        return None


def _pad_bytes_below(root: Path, replaced: Path, budget: int, cap: int) -> tuple[int, bool]:
    """``(bytes under ``root`` excluding a file at ``replaced``, whether the walk hit ``cap``)``.

    Stops as soon as ``budget`` is exceeded — the rest of the tree cannot change
    the answer — and at ``cap`` entries otherwise, so the cost of measuring is a
    bound and not a property of whatever a shell left in the pad.

    Iterative rather than recursive: the depth is bounded only by the entry cap,
    and a stack of one directory per entry is what keeps a pad shaped like a
    single deep chain from meeting the interpreter's recursion limit first. A
    symlink is never followed (a pad may hold one pointing out of it; it is
    ``resolve`` that refuses a write through one) and an entry that cannot be
    read contributes 0 rather than failing a write over a directory listing.
    """
    total = 0
    seen = 0
    stack = [root]
    while stack:
        try:
            entries = os.scandir(stack.pop())
        except OSError:
            continue
        with entries:
            for entry in entries:
                seen += 1
                if seen > cap:
                    return total, True
                try:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(Path(entry.path))
                        continue
                    if entry.name == replaced.name and entry.path == str(replaced):
                        # An overwrite REPLACES these bytes rather than adding to
                        # them, so counting them and then adding the payload would
                        # refuse a write that leaves the pad SMALLER. The caller
                        # spells ``replaced`` from the scan root for exactly this
                        # comparison, and the name is tested first so the walk
                        # pays one string compare per entry rather than a join.
                        continue
                    total += entry.stat(follow_symlinks=False).st_size
                except OSError:
                    continue
                if total > budget:
                    return total, False
    return total, False


def check_scratchpad_write(path: Path, root: Path, url: str, size: int | None = None) -> None:
    """Refuse a write whose NAME or SIZE is build output rather than scratch.

    Raises :class:`ScratchpadContentError` on a refused segment, a refused
    suffix, a ``size`` over :data:`SCRATCHPAD_MAX_WRITE_BYTES`, a write that would
    take the pad past :data:`SCRATCHPAD_TOTAL_BUDGET_BYTES`, a pad too wide to
    finish measuring (:data:`SCRATCHPAD_BUDGET_SCAN_ENTRIES`), or a path that
    cannot be placed inside the pad; returns ``None`` otherwise. ``url`` is
    echoed so the message names the address the caller actually typed, and every
    message ends with :data:`SCRATCHPAD_ELSEWHERE`, which is where the material
    belongs instead.

    ``size`` is the payload's length in BYTES, and it is optional because
    ``edit`` has none to give: an edit sees only its hunks, never the whole
    file, so an edit is judged on the name alone — and, for the pad total, on
    what the pad already holds rather than on what the edit would add to it.
    ``write`` passes the encoded length — bytes and not characters, because every
    ceiling here is about what lands on the shared disk.

    READS ARE DELIBERATELY NOT GATED. Pads written before this policy are full of
    exactly these names, and their owner is the agent cleaning them up: a gate on
    ``read`` would strand the very files it is meant to help retire, leaving only
    a shell (given the absolute path) as the way in.
    """
    below = _relative_parts(path, root)
    if below is None:
        raise ScratchpadContentError(
            f"{url}: this write could not be placed inside the pad, so it is refused "
            f"rather than judged. {SCRATCHPAD_ELSEWHERE}"
        )
    for segment in below:
        if _is_refused_segment(segment):
            raise ScratchpadContentError(
                f"{url}: '{segment}' is a build or dependency directory, not scratch. "
                f"{SCRATCHPAD_ELSEWHERE}"
            )
    suffix = _refused_suffix(below[-1]) if below else None
    if suffix is not None:
        raise ScratchpadContentError(
            f"{url}: '{suffix}' is a compiled, archived or model artefact, not scratch. "
            f"{SCRATCHPAD_ELSEWHERE}"
        )
    if size is not None and size > SCRATCHPAD_MAX_WRITE_BYTES:
        raise ScratchpadContentError(
            f"{url}: {size} bytes is over the {SCRATCHPAD_MAX_WRITE_BYTES}-byte ceiling for a "
            f"single write. {SCRATCHPAD_ELSEWHERE}"
        )
    # Spelled from ``root`` rather than used as ``path``: the walk enumerates
    # ``root``, so the file it must recognise as the one being REPLACED has to be
    # spelled from the same string. ``path`` is resolved and the root need not be
    # — on macOS that is ``/private/var`` against ``/var``, which silently drops
    # the exclusion and then refuses an overwrite that SHRINKS the pad. That
    # matters beyond tidiness: a pad over its total would otherwise be writable
    # only from a shell, which is the route around the pad this policy exists to
    # avoid.
    replaced = root.joinpath(*below)
    held, truncated = _pad_bytes_below(
        root, replaced, SCRATCHPAD_TOTAL_BUDGET_BYTES, SCRATCHPAD_BUDGET_SCAN_ENTRIES
    )
    if truncated:
        # The walk stopped counting, so the pad is refused on the SHAPE of what
        # it holds rather than on its size: 20,000 entries is a tree.
        raise ScratchpadContentError(
            f"{url}: this pad holds more than {SCRATCHPAD_BUDGET_SCAN_ENTRIES:,} entries, which "
            f"is a tree rather than a pad. {SCRATCHPAD_ELSEWHERE}"
        )
    would_hold = held + (size or 0)
    if would_hold > SCRATCHPAD_TOTAL_BUDGET_BYTES:
        raise ScratchpadContentError(
            f"{url}: the pad holds {held:,} bytes, so this write would leave it holding "
            f"{would_hold:,} — over the {SCRATCHPAD_TOTAL_BUDGET_BYTES:,}-byte ceiling for a "
            f"pad. {SCRATCHPAD_ELSEWHERE}"
        )
