"""Directory suggestions and validation behind ``/move``.

Kept free of widget imports for the reason
:mod:`local_operator.tui.copy_targets` is: the assembly rules are pure
functions over a directory and a config root, so they are testable — and
reusable by a non-Textual frontend — without importing Textual to learn what
``/move`` would offer.

WHY A PICKER NEEDS SUGGESTIONS AT ALL. The reported problem was a session
started in the wrong folder, which means the user does not want to *type* a
path: they want to recognise the one they meant. So the list leads with places
the user has actually been (this session's directory, then other sessions'),
and only then with the structural fallbacks — home, ``/tmp``, the parent, and
the children that make the picker a navigator rather than a bookmark list.

WHERE "RECENT" COMES FROM, and why there is a small file for it. Three sources,
cheapest first, deduplicated in this order:

* the LIVE runtime records (``session/runtime/registry.scan``), which carry
  ``cwd`` and are the other ``lop`` sessions running right now. This is the
  most valuable source and it costs one directory listing.
* the wake index, whose entries also carry ``cwd`` — sessions that are not
  running but have work armed.
* :data:`RECENTS_FILE`, this module's own tiny list of directories ``/move``
  has been used to reach.

The third exists because the first two are both LIVE state: quit every session
and the recents list goes empty, so the second use of the feature on a fresh
morning would offer nothing the first use taught it. It is deliberately the
smallest durable thing that fixes that — a capped JSON array of strings,
atomically replaced, best-effort at both ends (an unreadable or malformed file
is treated as empty and a failed write costs the user nothing but a missing
row). Notably it is NOT a general "directory history": nothing writes it but a
completed move, so it cannot grow without the user having chosen each entry.

The session TRANSCRIPT also records a cwd (the durable frontend checkpoint),
and it is deliberately not read here: the picker opens synchronously and those
files reach 100 MB, so mining them per open would trade a directory listing for
a multi-hundred-megabyte parse to learn something the two live sources already
answer for every session that matters.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

#: Where the durable half of "recent directories" lives, under the config root
#: beside the other small derived indexes (``wakes/``, ``mobile-seen.json``).
RECENTS_FILE = "move-recents.json"

#: How many directories the durable list keeps. A picker is scanned, not paged
#: through, and the live sources already contribute the sessions that matter
#: right now — so this is a memory of where the user has deliberately gone,
#: not a shell history. Small enough that the file stays a single short read.
RECENTS_LIMIT = 20

#: Immediate subdirectories of the current directory offered as rows. Bounded
#: because a node_modules-shaped directory has thousands and the card can draw
#: about ten: past this the rows stop being suggestions and become noise the
#: user has to filter anyway, which is what TYPING is for.
CHILD_LIMIT = 12

#: Suggestion rows in total. The card scrolls, so this is not a display cap —
#: it bounds the WORK an open costs, and it sits well past what fits on screen
#: so the filter still has material to narrow.
SUGGESTION_LIMIT = 60


@dataclass(frozen=True)
class MoveTarget:
    """One directory ``/move`` can go to, with why it is being offered.

    ``path`` is absolute and normalised, because it is what the session's cwd
    is set to; ``label`` is the home-relative rendering the user reads. Keeping
    both means the row a user recognises (``~/workspace/repos/x``) and the
    value the runtime is spawned with can never drift apart.
    """

    path: str
    label: str
    #: ``current`` | ``recent`` | ``home`` | ``tmp`` | ``parent`` | ``child``.
    #: Drives the row's note and nothing else; the picker does not branch on it.
    kind: str
    #: The short right-hand note ("current", "session", "home", …). Carried on
    #: the target rather than derived in the widget so the ordering rationale
    #: and the label the user reads live in one place.
    detail: str = ""


class MoveError(ValueError):
    """A target that cannot be moved to, carrying the sentence to show.

    A distinct type rather than a returned string so a caller cannot forget to
    check: every rejection reaches the user as one vetted line, and the three
    reasons (absent, not a directory, unreadable) are told apart because they
    call for different actions.
    """


def format_label(path: str | Path, *, home: Path | None = None) -> str:
    """``~/rel/path`` inside the home tree, the absolute path outside it.

    The same rendering :func:`local_operator.tui.widgets.status_line.format_cwd`
    gives the band, and deliberately so: the row the user picks and the segment
    they then read on the band must be the same string, or the move looks like
    it landed somewhere else.
    """
    text = str(path)
    if not text:
        return ""
    root = home if home is not None else Path.home()
    try:
        relative = Path(text).relative_to(root)
    except ValueError:
        return text
    return "~" if str(relative) == "." else f"~/{relative}"


def expand_path(text: str, *, cwd: str | Path) -> Path:
    """A typed path as an absolute one: ``~`` expanded, relative resolved.

    Resolved against the SESSION's directory rather than the process's, because
    those differ routinely — a resumed conversation carries its own cwd while
    the terminal was launched somewhere else — and ``/move ../sibling`` has to
    mean the sibling of the directory the band is showing.

    Symlinks are deliberately NOT resolved (``absolute()``, not ``resolve()``):
    a user who moves into ``~/current-project`` means the symlink, and printing
    its target back at them on the band would read as the move having gone
    somewhere else.
    """
    raw = os.path.expanduser(text.strip())
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = Path(cwd) / candidate
    return Path(os.path.normpath(candidate))


def validate_target(path: str | Path) -> Path:
    """``path`` as a usable working directory, or raise :class:`MoveError`.

    Three distinct rejections because they call for three different next
    moves, and a single "invalid directory" told the user none of them. The
    executable bit is checked as well as existence: a directory that cannot be
    entered would let the move "succeed" and then fail every tool call made
    inside it, which is the half-applied state this exists to prevent.
    """
    candidate = Path(path)
    if not candidate.exists():
        raise MoveError(f"no such directory: {format_label(candidate)}")
    if not candidate.is_dir():
        raise MoveError(f"not a directory: {format_label(candidate)}")
    if not os.access(candidate, os.R_OK | os.X_OK):
        raise MoveError(f"cannot enter {format_label(candidate)}: permission denied")
    return candidate


def _readable_dir(path: str | Path) -> bool:
    """Whether ``path`` is a directory this process could actually work in.

    The suggestion assembler's filter. Never raises: a source can name a
    directory on a volume that has since been unmounted, and a picker that
    cannot open because one stale row's ``stat`` failed is worse than a row
    that is quietly not offered.
    """
    try:
        return validate_target(path) is not None
    except (MoveError, OSError):
        return False


def read_recents(config_dir: Path) -> list[str]:
    """The durable recent-directory list, newest first; ``[]`` when unreadable.

    Best-effort by contract. This file is an enhancement to a list that has two
    live sources already, so every failure mode — absent, truncated, holding a
    JSON object instead of an array, holding non-strings — degrades to "no
    remembered directories" rather than costing the user the picker.
    """
    path = Path(config_dir) / RECENTS_FILE
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError:
        return []
    except (OSError, ValueError):
        logger.debug("move recents unreadable; continuing without them", exc_info=True)
        return []
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, str) and item]


def remember_recent(config_dir: Path, path: str | Path) -> None:
    """Record ``path`` as the newest entry, deduplicated and capped.

    Written to a temporary file in the SAME directory and ``os.replace``d over
    the target, the discipline every small index here uses
    (``config.py``, ``multiplexer/markers.py``): a torn read of this file would
    silently empty a user's remembered directories, and same-directory replace
    is the only form that is atomic.

    Never raises. This runs immediately after a move the user has already been
    told succeeded, so a read-only config directory must cost them the memory
    of the move and not the move itself.
    """
    directory = Path(config_dir)
    text = str(path)
    entries = [item for item in read_recents(directory) if item != text]
    entries.insert(0, text)
    del entries[RECENTS_LIMIT:]
    try:
        directory.mkdir(parents=True, exist_ok=True)
        handle_fd, temporary = tempfile.mkstemp(dir=directory, prefix=".move-recents-")
        try:
            with os.fdopen(handle_fd, "w") as handle:
                json.dump(entries, handle)
            os.replace(temporary, directory / RECENTS_FILE)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
    except OSError:
        logger.debug("could not record the move in recents", exc_info=True)


def live_session_dirs(config_dir: Path) -> list[str]:
    """Directories other ``lop`` sessions are working in, newest session first.

    The most valuable recents source and the cheapest: one records listing
    answers it for every running session, which is exactly the "take me to
    where my other work is" case ``/move`` was asked for. Sessions are ordered
    by start time so the one the user most recently opened leads.

    Best-effort, like every other consumer of the registry here
    (``session_catalog.decorate_rows``): an unreadable run directory costs the
    picker its liveness rows, never its ability to open.
    """
    out: list[str] = []
    try:
        from local_operator.session.runtime import registry

        scanned = registry.scan(Path(config_dir))
    except Exception:  # noqa: BLE001 — suggestions are an enhancement, never a gate
        logger.debug("could not scan session records for directories", exc_info=True)
        scanned = []
    rows = [record for record, state in scanned if state != "stale"]
    rows.sort(key=lambda record: float(getattr(record, "started_at", 0.0) or 0.0), reverse=True)
    for record in rows:
        cwd = str(getattr(record, "cwd", "") or "")
        if cwd:
            out.append(cwd)
    return out


def wake_session_dirs(config_dir: Path) -> list[str]:
    """Directories of sessions that are not running but have wakes armed.

    The second live source. A session with a wake is one the user expects to
    come back to, so its directory belongs in the same tier as a running
    session's — and the index already carries ``cwd`` for the supervisor's own
    use, so reading it costs one small file.
    """
    try:
        from local_operator.wakes.store import read_index

        index = read_index(Path(config_dir))
    except Exception:  # noqa: BLE001
        logger.debug("could not read the wake index for directories", exc_info=True)
        return []
    out: list[str] = []
    for entry in (index or {}).values():
        if not isinstance(entry, dict):
            continue
        cwd = str(entry.get("cwd") or "")
        if cwd:
            out.append(cwd)
    return out


def child_dirs(cwd: str | Path, *, limit: int = CHILD_LIMIT) -> list[str]:
    """Immediate subdirectories of ``cwd``, alphabetical, hidden ones omitted.

    What makes the picker a navigator instead of a bookmark list. Hidden
    directories are skipped because ``.git``/``.venv``/``.mypy_cache`` would
    otherwise crowd out every real child on any checkout — a user who wants one
    can type it, and typing is the path that reaches ANY directory.
    """
    try:
        entries = sorted(Path(cwd).iterdir(), key=lambda item: item.name.lower())
    except OSError:
        return []
    out: list[str] = []
    for entry in entries:
        if entry.name.startswith("."):
            continue
        try:
            if not entry.is_dir():
                continue
        except OSError:
            continue
        out.append(str(entry))
        if len(out) >= limit:
            break
    return out


def suggest_targets(
    cwd: str | Path,
    *,
    config_dir: Path,
    home: Path | None = None,
    limit: int = SUGGESTION_LIMIT,
) -> list[MoveTarget]:
    """The rows ``/move`` opens with, in order of usefulness and deduplicated.

    The ORDER is the feature, so it is stated once here rather than being an
    accident of how the sources are concatenated:

    1. the current directory, always first and always present — a picker that
       does not show where you are makes "did that work?" unanswerable, and it
       is also the row that makes Esc-equivalent (pick where you already are) a
       visible option rather than a thing you have to know;
    2. recent directories — other sessions', then remembered ones. These are
       places the user has demonstrably worked, which beats anything structural;
    3. home and ``/tmp``, the two destinations that need no memory at all;
    4. the parent, then the children — the navigator tier, last because they
       are one step each where the tiers above are whole destinations.

    Deduplicated by resolved path with the FIRST occurrence winning, so a
    directory that is both the home directory and a running session's cwd is
    offered once, in the higher tier, with that tier's note. Only directories
    that exist and can be entered are offered: a suggestion that fails
    validation the moment it is chosen is worse than no suggestion.
    """
    root = home if home is not None else Path.home()
    current = Path(os.path.normpath(Path(cwd)))
    config_root = Path(config_dir)

    # (path, kind, detail), in the tiers documented above. Built as one flat
    # list so the dedup below sees them in exactly the offered order.
    candidates: list[tuple[str, str, str]] = [(str(current), "current", "current")]
    for path in live_session_dirs(config_root):
        candidates.append((path, "recent", "session"))
    for path in wake_session_dirs(config_root):
        candidates.append((path, "recent", "wake armed"))
    for path in read_recents(config_root):
        candidates.append((path, "recent", "recent"))
    candidates.append((str(root), "home", "home"))
    # Noted like every other row. A blank note read as a missing value in the
    # rendered frame — every neighbour carries one, so the gap looked like the
    # row had failed to resolve rather than like a row that needed no
    # explanation. "scratch" is what the directory is FOR, which is the same
    # kind of statement "home" and "parent" make.
    candidates.append((os.path.normpath("/tmp"), "tmp", "scratch"))
    parent = current.parent
    if parent != current:
        candidates.append((str(parent), "parent", "parent"))
    for path in child_dirs(current):
        candidates.append((path, "child", "in this folder"))

    out: list[MoveTarget] = []
    seen: set[str] = set()
    for raw, kind, detail in candidates:
        path = os.path.normpath(os.path.expanduser(raw))
        # Deduplicated on the RESOLVED path, offered as the un-resolved one.
        # On macOS `/tmp` is a symlink to `/private/tmp`, and a running
        # session reports the resolved form while this module's own `/tmp`
        # row does not — so a plain string dedup offered the same directory
        # twice, in two spellings, in the same list (observed on the real
        # store). Resolution decides identity; the label still shows the path
        # the user asked for, because `expand_path` deliberately keeps
        # symlinks unresolved.
        try:
            identity = os.path.realpath(path)
        except OSError:
            identity = path
        if identity in seen:
            continue
        seen.add(identity)
        # The CURRENT directory is offered even when it fails validation: a
        # session whose directory was deleted underneath it is exactly when a
        # user reaches for `/move`, and hiding the row would leave the picker
        # silently disagreeing with the band about where the session is.
        if kind != "current" and not _readable_dir(path):
            continue
        out.append(
            MoveTarget(path=path, label=format_label(path, home=root), kind=kind, detail=detail)
        )
        if len(out) >= limit:
            break
    return out


def complete_path(
    text: str, *, cwd: str | Path, home: Path | None = None, limit: int = 40
) -> list[MoveTarget]:
    """Directories matching a typed PATH prefix, for the completion tier.

    The half of the picker that reaches an arbitrary directory. A list of
    suggestions can only ever offer what it guessed; a filter that also
    completes paths means no directory on the machine is unreachable, which is
    the difference between a picker people use and one they abandon.

    The prefix is split the way a shell splits it — everything up to the last
    separator is the directory to list, the remainder is the fragment to match
    — so ``~/work`` lists ``~`` for names starting ``work`` while ``~/work/``
    lists inside the directory itself. Matching is case-insensitive because the
    common filesystem here is, and a case-sensitive filter over a
    case-insensitive filesystem reports "no matches" for a directory the user
    can see.

    Hidden directories are offered ONLY when the fragment itself starts with a
    dot: the same rule :func:`child_dirs` uses, with the escape hatch that
    makes a typed ``~/.config`` reachable.
    """
    raw = text.strip()
    if not raw:
        return []
    expanded = os.path.expanduser(raw)
    # A trailing separator means "inside this directory", so the fragment is
    # empty and the whole of it is the base.
    if expanded.endswith(os.sep):
        base_text, fragment = expanded, ""
    else:
        base_text, _, fragment = expanded.rpartition(os.sep)
        if not base_text:
            # A bare relative word ("src"): complete inside the session's own
            # directory rather than treating it as a filesystem root.
            base_text = str(cwd) if not expanded.startswith(os.sep) else os.sep
    base = Path(base_text) if base_text else Path(cwd)
    if not base.is_absolute():
        base = Path(cwd) / base
    root = home if home is not None else Path.home()
    needle = fragment.lower()
    try:
        entries = sorted(base.iterdir(), key=lambda item: item.name.lower())
    except OSError:
        return []
    out: list[MoveTarget] = []
    for entry in entries:
        name = entry.name
        if name.startswith(".") and not fragment.startswith("."):
            continue
        if needle and not name.lower().startswith(needle):
            continue
        path = os.path.normpath(str(entry))
        if not _readable_dir(path):
            continue
        out.append(
            MoveTarget(path=path, label=format_label(path, home=root), kind="typed", detail="")
        )
        if len(out) >= limit:
            break
    return out


def looks_like_path(text: str) -> bool:
    """Whether ``text`` should be COMPLETED as a path rather than used to filter.

    The picker has one input and two jobs, so the rule that splits them has to
    be one a user can predict without being told: anything with a separator, or
    anchored at ``~``, ``/`` or ``.``, is a path. Everything else is a filter
    over the suggestions — which is what a user typing ``repos`` means, and
    completing that as a relative directory would answer an empty list because
    no such child exists.
    """
    stripped = text.strip()
    if not stripped:
        return False
    return stripped.startswith(("~", "/", ".")) or os.sep in stripped


def filter_targets(targets: list[MoveTarget], query: str) -> list[MoveTarget]:
    """``targets`` whose label or path contains ``query``, order preserved.

    Narrowing NEVER reorders, matching ``session_picker.filter_rows``' contract
    for a fixed query: the tier order above is the picker's whole argument for
    what to offer first, and re-ranking on each keystroke would move a row out
    from under a cursor that is aiming at it.
    """
    needle = query.strip().lower()
    if not needle:
        return list(targets)
    return [
        target
        for target in targets
        if needle in target.label.lower() or needle in target.path.lower()
    ]
