"""Teams: a named roster of reusable agents under one manager.

WHY THIS EXISTS
---------------

Roles and specialist agents are reusable building blocks — a ``coder`` or a
"User Dashboard Agent" should be writable once and usable in many groupings.
A Team is the grouping: a manager, a roster of members (role or specialist,
with counts), plus TWO instruction layers that do not belong on any one
agent because they describe the GROUP rather than the person:

1. **Collaboration** (``instructions.md``) — how this team works together:
   review order, who blocks a release, how the manager delegates.
2. **Project** (``project.md``) — the product or domain this instance of the
   team is responsible for. The same Feature Release roster can staff two
   products by swapping only this file.

The three layers a member actually sees, outermost last:

- the agent's own ``system_prompt.md`` (base behaviour, reusable)
- the team's collaboration brief (how we work)
- the team's project brief (what we are responsible for)

A manager session also gets a roster so it can ``task(agent=...)`` the right
people without the operator restating the org chart every turn.

STORAGE
-------

``<config_dir>/teams/<id>/``:

- ``team.yml`` — id, name, description, manager, members
- ``instructions.md`` — collaboration brief
- ``project.md`` — project / product brief

Members are referenced by NAME (a role or a specialist agent), never by
registry id, so a team survives an agent being deleted and recreated and so
the same ``coder`` row can sit on many teams. Counts let a team ask for two
coders without inventing a second profile.

The registry is NEVER enumerated into the prompt. ``list`` is the explicit
action that reveals names; the ``teams`` guide is what the model reads to
learn the concept.
"""

from __future__ import annotations

import errno
import logging
import os
import re
import shutil
import stat
import tempfile
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

# Shared identity with ``local_operator.types`` (see its docstring): the CLI
# catches this at zero startup cost while ``teams`` raises it where the lock
# times out. Importing the name (not redefining it) keeps the two identical.
from local_operator.types import TeamRegistryLockTimeout, TeamRegistryRecoveryError

logger = logging.getLogger(__name__)

#: Cap on each team instruction file. Same bound as a role body: these ride
#: in front of a manager session and every member launch, so an unbounded
#: paste is an unbounded per-turn bill.
MAX_TEAM_INSTRUCTIONS_CHARS = 8_000

#: A team name is also a slash-command argument, so it cannot contain spaces
#: or slashes — ``/team feature-release ship it`` has to parse unambiguously.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

#: IDs are filesystem row addresses, including on transported ``Team`` models.
#: Keep the historical safe-segment shape (fixtures use short IDs) while
#: excluding every separator, absolute path, and dot segment.
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

#: Windows has no descriptor-pinned row reads. Immediate bounded retries cover
#: the directory-swap gap without combining files from different revisions.
_ROW_SNAPSHOT_ATTEMPTS = 3
_ROW_SNAPSHOT_RETRY_S = 0.005

#: Persistence is normally a few local-file writes. A bounded retry keeps a
#: wedged peer from parking a synchronous tool call forever while still making
#: ordinary concurrent registry mutations serialize rather than fail spuriously.
_TEAM_LOCK_TIMEOUT_S = 10.0
_TEAM_LOCK_RETRY_S = 0.01

#: How long a READ may wait for the writer lock before serving the current
#: on-disk view instead (R7-1).
#:
#: Reads run SYNCHRONOUSLY on the TUI event loop — ``_team_choices`` calls
#: ``list_teams`` from a keystroke handler — and a `.<id>.backup.*` artifact is
#: visible for the whole ``target -> backup -> staged -> target`` window of
#: every healthy publish (measured at ~35% of wall time under continuous peer
#: writes). Letting a reader take the bounded 10 s writer wait therefore turned
#: one `/team ` keystroke, which fans out into several registry reads, into a
#: 60 s frozen app. A reader never NEEDS the lock: recovery is an optimisation
#: that heals a crashed peer's tree, and the current snapshot is a legitimate
#: answer while another process is mid-publish.
#:
#: So the read path try-acquires. Zero means "one non-blocking attempt", which
#: is what every UI-reachable read uses. A CLI/tool read — which owns its
#: process and blocks no event loop — may pass a short bound so a one-shot
#: `teams list` immediately after a crash still heals the tree rather than
#: losing the race to a peer that happens to hold the lock (U6-1).
_READ_RECOVERY_UI_WAIT_S = 0.0
_READ_RECOVERY_CLI_WAIT_S = 2.0

#: Minimum seconds between read-path recovery ATTEMPTS for an unchanged artifact
#: set (R7-1). Without it, a failing or contended attempt is retried on every
#: read, so a burst of keystrokes pays the probe repeatedly for one artifact
#: that is not going anywhere. Recovery is still attempted IMMEDIATELY whenever
#: the artifact set changes, so a genuinely new crash is never delayed by it.
_READ_RECOVERY_COOLDOWN_S = 1.0


#: How deep an org (a team whose members are themselves teams) may nest before
#: the resolver stops descending. Eight levels is far past any real human org
#: and keeps the resolver, the tidy-tree layout, and the render all bounded: a
#: reference deeper than this is truncated with a visible "depth limit" node
#: rather than followed. It also backstops the cycle guard — even a
#: mis-detected cycle cannot run away past this. Lives here, in the MODEL layer,
#: because depth is a property of the data, not of any one widget that draws it.
MAX_ORG_DEPTH = 8


class TeamMember(BaseModel):
    """One roster slot: a named agent/role, or a nested TEAM, possibly repeated."""

    role: str = Field(
        ...,
        description="Role/specialist name, or team name when kind='team'.",
    )
    count: int = Field(
        default=1,
        ge=1,
        le=16,
        description="How many of this role to run.",
    )
    # NEW discriminator. Absent ("agent") in every existing ``team.yml``, so old
    # files load unchanged — the field defaults to "agent" and validation is a
    # no-op on a roster that never mentions it, which is the whole point of a
    # default rather than a required key (a required field would silently drop
    # every stored team through ``_load``'s except-and-skip).
    #
    # "team" marks this slot as a reference to ANOTHER team by name, turning a
    # flat roster into an "org" (a team of teams). The referenced name still
    # lives in ``role`` rather than a new field so that ``member_names()``,
    # ``member_count()``, and every existing reader keep working without
    # knowing nesting exists — a team slot simply reads as its team name. One
    # roster, one authored order: a separate ``subteams`` list would split the
    # roster into two lists the author has to keep mentally merged and would
    # force every reader to concatenate them.
    kind: Literal["agent", "team"] = Field(
        default="agent",
        description="'agent' (a role/specialist) or 'team' (a nested sub-team).",
    )

    @field_validator("role")
    @classmethod
    def _role_name(cls, value: str) -> str:
        name = (value or "").strip()
        if not name:
            raise ValueError("member role is required")
        return name


class TeamEditFields(BaseModel):
    """Partial update. ``None`` means leave the stored value alone."""

    name: str | None = None
    description: str | None = None
    manager: str | None = None
    members: list[TeamMember] | None = None
    instructions: str | None = None
    project: str | None = None


class Team(BaseModel):
    """A durable team: manager + members + layered instruction briefs.

    ``instructions`` and ``project`` are plain strings with NO marker for
    whether they have been read: an unloaded brief and an intentionally
    empty one are indistinguishable ON THE MODEL, by design. The loaded/
    unloaded distinction is REGISTRY-owned (``TeamRegistry._briefs_loaded``),
    because it is a fact about which files THIS registry has read — not a
    property of the value, and not something that can survive a
    ``model_dump``/``model_validate`` round trip through a tool or transport
    (review round 2, R2-1: a str-subclass sentinel serialized as ``""`` and
    revalidated as a plain ``""``, so a transported metadata-only team read
    as a deliberate clear and ``save_team`` truncated the briefs). The
    consequence is a documented constraint on ``save_team``: it cannot trust
    a transported object's empty briefs and preserves the on-disk files for
    any team id this registry has not loaded. Explicit brief writes go
    through :meth:`TeamRegistry.update_team` (or ``create_team``), which
    hydrate first and apply the caller's fields — including ``""`` clears —
    onto a known-loaded object.
    """

    id: str
    name: str
    created_date: datetime
    description: str = ""
    manager: str = "manager"
    members: list[TeamMember] = Field(default_factory=list)
    # Both briefs default to "" and never carry a marker: see the class
    # docstring and ``save_team`` for why the loaded state lives in the
    # registry instead of the value. ``save_team``'s dump excludes both
    # fields from ``team.yml`` either way.
    instructions: str = ""
    project: str = ""

    @field_validator("id")
    @classmethod
    def _id(cls, value: str) -> str:
        return validate_team_id(value)

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        name = (value or "").strip()
        if not _NAME_RE.match(name):
            raise ValueError(
                "team name must be 1-64 characters of letters, digits, "
                "dot, underscore or hyphen, and cannot start with a hyphen"
            )
        return name

    @field_validator("manager")
    @classmethod
    def _manager(cls, value: str) -> str:
        name = (value or "").strip()
        if not name:
            raise ValueError("manager is required")
        return name

    def roster_lines(self) -> list[str]:
        """One scannable line per slot, manager first.

        A nested-team slot is badged ``(team)`` so a reader (and ``team show``)
        can tell an org apart from a flat roster — a member named ``pod`` and a
        sub-team named ``pod`` would otherwise render identically.
        """
        lines = [f"- manager: {self.manager} (you, when this team is invoked)"]
        for member in self.members:
            suffix = f" x{member.count}" if member.count > 1 else ""
            badge = " (team)" if member.kind == "team" else ""
            lines.append(f"- {member.role}{badge}{suffix}")
        return lines

    def member_count(self) -> int:
        """Total member copies on the roster (counts summed, manager excluded).

        Distinct from ``len(members)``: a ``reviewer x2`` slot is two members
        in one slot, and a summary that reports it as one understates the
        team the user assembled.
        """

        return sum(member.count for member in self.members)

    def member_names(self) -> list[str]:
        """Role names on the roster, manager included, first occurrence winning."""
        names: list[str] = []
        for name in (self.manager, *(member.role for member in self.members)):
            if name not in names:
                names.append(name)
        return names

    def manager_preamble(self) -> str:
        """Standing brief stamped into a manager session's instructions.

        Empty briefs cost nothing: a team that has only a roster still names
        the roster, and a team with nothing at all yields an empty string so
        it does not tax a session that has not been briefed yet.
        """
        parts: list[str] = [f"[team: {self.name}]"]
        if self.description.strip():
            parts.append(self.description.strip())
        parts.append("You are the manager of this team. You coordinate; you do not implement.")
        parts.append(
            "Delegate with task(agent='<role>') using the roster below. "
            "Each member already carries this team's collaboration and project "
            "briefs — give them the TASK, not a restatement of the team. "
            "Spin up the counts the roster names; do not invent extra copies."
        )
        parts.append("Roster:\n" + "\n".join(self.roster_lines()))
        collab = self.instructions.strip()
        if collab:
            parts.append("Collaboration:\n" + collab)
        project = self.project.strip()
        if project:
            parts.append("Project:\n" + project)
        return "\n\n".join(parts) + "\n"

    def member_preamble(self, role: str) -> str:
        """Brief stamped in front of a member's one-shot prompt.

        Shorter than the manager brief: a member does not need to be told how
        to delegate, and the role's own preamble already said how to do the
        job. This is the GROUP context the role file must not carry, because
        the same role sits on many teams.
        """
        parts: list[str] = [
            f"[team: {self.name}]",
            f"You are {role} on this team. The manager is {self.manager}.",
            "Teammates:\n" + "\n".join(self.roster_lines()),
        ]
        collab = self.instructions.strip()
        if collab:
            parts.append("Collaboration:\n" + collab)
        project = self.project.strip()
        if project:
            parts.append("Project:\n" + project)
        return "\n\n".join(parts) + "\n\n"


def validate_team_id(team_id: str) -> str:
    """Return an ID that is exactly one safe filesystem path segment."""
    candidate = team_id or ""
    if not _ID_RE.fullmatch(candidate) or candidate in {".", ".."}:
        raise ValueError(
            "team id must be 1-128 characters of letters, digits, dot, "
            "underscore or hyphen, start with a letter or digit, and contain no path separators"
        )
    return candidate


def validate_team_name(name: str) -> str:
    """Return a stripped, legal team name or raise ``ValueError``."""
    return Team.model_validate(
        {
            "id": "x",
            "name": name,
            "created_date": datetime.now(timezone.utc),
            "manager": "manager",
        }
    ).name


def _try_lock_exclusive(fd: int) -> bool:
    """Take one non-blocking exclusive lock attempt on ``fd``."""
    if os.name == "nt":  # pragma: no cover - platform specific
        import msvcrt

        try:
            if os.fstat(fd).st_size == 0:
                os.write(fd, b"\0")
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except OSError as exc:
            if exc.errno in (errno.EDEADLOCK, errno.EACCES, errno.EAGAIN):
                return False
            raise

    import fcntl

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError as exc:
        if exc.errno in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
            return False
        raise


def _unlock(fd: int) -> None:
    """Release a lock acquired by :func:`_try_lock_exclusive`."""
    if os.name == "nt":  # pragma: no cover - platform specific
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(fd, fcntl.LOCK_UN)


def _atomic_write_text(path: Path, text: str) -> None:
    """Publish one complete text file without exposing a truncated target."""
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _fsync_dir(path: Path) -> None:
    """Flush a directory entry, suppressing only unsupported implementations.

    Directory fsync is unavailable on Windows and a few filesystems reject it
    with a documented unsupported-operation errno. Real durability failures —
    including EIO, ENOSPC, EDQUOT, and permission errors — must reach the caller.
    """
    if os.name == "nt":  # pragma: no cover - exercised by Windows CI
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    except OSError as exc:
        unsupported = {
            errno.EINVAL,
            errno.EBADF,
            getattr(errno, "ENOTSUP", errno.EINVAL),
            getattr(errno, "EOPNOTSUPP", errno.EINVAL),
        }
        if exc.errno not in unsupported:
            raise
    finally:
        os.close(fd)


def _write_row_files(directory: Path, metadata: str, team: Team) -> None:
    """Write one complete team row into a staging directory.

    The three files are written with explicit flush+fsync (the directory is
    fsynced by the caller's publish step) so a crash after the publish rename
    can never expose a row whose contents are still in the page cache. Order
    is metadata FIRST — the same order the create path has always used — so a
    partially written staging directory loads (if it were ever seen) as a
    metadata row with empty briefs, never as briefs without identity.
    """
    _write_row_files_after(directory, metadata, team, stop_after=None)


def _write_row_files_after(
    directory: Path,
    metadata: str,
    team: Team,
    *,
    stop_after: str | None,
) -> None:
    """``_write_row_files`` with an optional early stop AFTER a named file.

    Exists for the R5-2 failure-injection tests: they need a staged row that
    is INCOMPLETE at a chosen point (the mid-row crash state) rather than a
    row that was never started, because the cleanup contract being verified
    is "whatever partial staging exists is removed and never published".
    Production callers use the complete form above.
    """
    contents = (
        ("team.yml", metadata),
        ("instructions.md", team.instructions),
        ("project.md", team.project),
    )
    for filename, text in contents:
        path = directory / filename
        with path.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if filename == stop_after:
            return


class TeamRegistry:
    """On-disk registry of teams under ``<config_dir>/teams``."""

    def __init__(self, config_dir: Path, refresh_interval: float = 5.0) -> None:
        self.config_dir = Path(config_dir)
        self.teams_dir = self.config_dir / "teams"
        # No mkdir here: every interactive session constructs a registry, and
        # an unused feature must not litter the config dir. ``save_team``
        # creates the tree on first write, and ``_load`` treats a missing
        # directory as "no teams".
        self._teams: dict[str, Team] = {}
        # Picker/listing callers need only ``team.yml`` metadata. Briefs can be
        # 8k each and sit on the session's five-second refresh path, so remember
        # which teams paid that I/O only when an attach/show/preamble lookup asks.
        self._briefs_loaded: set[str] = set()
        self._last_refresh_time = 0.0
        self._refresh_interval = refresh_interval
        # Which artifact set the last read-path recovery attempt was made
        # against, and when — the bound that keeps a burst of keystrokes from
        # re-probing one stranded backup on every read (R7-1).
        self._recovery_attempted_for: frozenset[str] | None = None
        self._recovery_attempted_at = 0.0
        # The artifact set a restore was attempted against and FAILED. Distinct
        # from the attempt latch above because the two get opposite treatment
        # INSIDE the cooldown: a contended attempt serves the current view, a
        # failed restore keeps raising while the artifact is on disk (R6-4).
        # It does NOT suppress the re-attempt past the cooldown — a repaired
        # tree must heal without a restart (R8-1). The exception is kept so the
        # repeat carries the ORIGINAL cause and wording rather than a re-derived
        # approximation of it.
        self._recovery_failed_for: frozenset[str] | None = None
        self._recovery_failure: TeamRegistryRecoveryError | None = None
        # Recovery is the sole startup read that may create a lock sidecar, and
        # only when a crash artifact proves work is required. Unused registries
        # remain side-effect free.
        #
        # CONSTRUCTION MUST NOT RAISE (R7-2). Every interactive session builds a
        # registry during boot, and `session_factory` builds this one before the
        # model, the tools and the transcript exist. An unrecoverable `teams/`
        # — a feature the user may never have touched — therefore used to take
        # down the entire session. The recovery failure is REMEMBERED instead
        # and re-raised by the first real read/mutation, so the teams feature
        # degrades on its own while the session starts normally, and every
        # caller that does touch teams still gets the concise guidance.
        self.recovery_error: TeamRegistryRecoveryError | None = None
        try:
            self._recover_for_read_if_needed()
        except TeamRegistryRecoveryError as exc:
            self.recovery_error = exc
        self._load()

    def _load(self) -> None:
        loaded: dict[str, Team] = {}
        try:
            children = list(self.teams_dir.iterdir())
        except OSError:
            self._teams = {}
            self._briefs_loaded = set()
            self._last_refresh_time = time.time()
            return
        for child in children:
            # Crafted symlink rows must never turn metadata reads or deletion
            # into operations outside the registry root.
            if child.is_symlink():
                continue
            # R5-1: dot-prefixed entries are NEVER team rows. Create staging
            # directories (``.<id>.<rand>``), update staging/backup swap
            # directories, per-file ``.team.yml.*`` temporaries and any crash
            # artifact left by an interrupted write must stay invisible to
            # readers: a staged ``team.yml`` exists only because a writer put
            # it there FIRST, before the publish rename, so trusting it here
            # exposed a half-written row (empty briefs hydrated as canonical,
            # then saved back over the authored ones). Names are validated to
            # ``[A-Za-z0-9._-]`` and ids are uuid4, so no legitimate row can
            # start with a dot.
            if child.name.startswith("."):
                continue
            try:
                validate_team_id(child.name)
            except ValueError:
                continue
            if not child.is_dir():
                continue
            # R5-1: the directory name IS the row's address — briefs are read
            # from ``teams/<dir>/`` and deletion removes ``teams/<dir>/``. A
            # YAML whose internal id points elsewhere (a stale staging copy,
            # a hand-moved directory) would make the cache key disagree with
            # where the files actually live, so it is skipped as invalid
            # rather than trusted.
            path = child / "team.yml"
            if path.is_symlink() or not path.is_file():
                continue
            try:
                with path.open("r", encoding="utf-8") as handle:
                    data = yaml.safe_load(handle) or {}
                if not isinstance(data, dict):
                    continue
                team = Team.model_validate(data)
                if team.id != child.name:
                    logger.warning(
                        "ignoring team directory %s: metadata id %s does not match",
                        child.name,
                        team.id,
                    )
                    continue
                loaded[team.id] = team
            except FileNotFoundError:
                # R5-2: the row directory was swapped out between the scan and
                # this open — the documented publish gap, not a corrupt row.
                # Skip silently; the next refresh sees the new revision.
                continue
            except Exception as exc:  # noqa: BLE001 — one bad file must not hide the rest
                logger.warning("invalid team metadata in %s: %s", child.name, exc)
        self._teams = loaded
        # A refresh replaces every model with metadata-only instances. Keeping
        # an old loaded marker would return blank briefs from the replacement.
        self._briefs_loaded = set()
        self._last_refresh_time = time.time()

    def _refresh_if_needed(self, *, wait: float | None = None) -> None:
        """Heal, then refresh the snapshot if the interval has elapsed.

        ``wait`` is threaded to :meth:`_recover_for_read_if_needed` so a
        process-owning caller (CLI, tool) can afford a short bounded acquisition
        while a UI caller stays strictly non-blocking (R7-1).
        """
        self._raise_if_recovery_failed()
        self._recover_for_read_if_needed(wait=wait)
        if time.time() - self._last_refresh_time > self._refresh_interval:
            self._load()

    def _raise_if_recovery_failed(self) -> None:
        """Re-raise a construction-time recovery failure at the first real use.

        Construction swallows it so a session can still start (R7-2), but a
        caller that actually reads or mutates teams must not be told the
        registry is merely empty — a hidden authoritative row invalidates
        absence and uniqueness answers exactly as it does mid-session (R6-4).
        Re-raised rather than latched forever: the next call re-attempts
        recovery through the normal path, so fixing the permissions and
        retrying works without restarting the session.

        :meth:`_recover_for_read_if_needed` holds the same property by a
        different mechanism (a cooldown-gated re-attempt rather than a one-shot
        memo pop) because it runs on every read and must also bound its attempt
        RATE. The two are deliberately not merged: this one is a one-shot hand-
        off of a construction-time error, and folding it into the rate limiter
        would make a boot failure wait out a cooldown before the first read
        could report it. What they must share — and now do — is that no failure
        state survives the repair that fixes it (R8-1).
        """
        error, self.recovery_error = self.recovery_error, None
        if error is not None:
            raise error

    def _recovery_failure_repeat(self) -> TeamRegistryRecoveryError:
        """A FRESH exception carrying the latched failure's wording and cause.

        Re-``raise``-ing the stored instance appends the new raise site to that
        one object's ``__traceback__`` on every read, and ``/team`` fans out
        several reads per keystroke — measured at ~4 frames per read, 805 frames
        by read 200, each retaining its locals (R8-2). A new object per raise
        keeps the traceback the size of one stack while preserving what the user
        and the diagnostics need: the original guidance text and the chained
        storage error that caused it.
        """
        stored = self._recovery_failure
        assert stored is not None  # only called under a set latch
        repeat = TeamRegistryRecoveryError(*stored.args)
        repeat.__cause__ = stored.__cause__
        return repeat

    def _recover_for_read_if_needed(self, *, wait: float | None = None) -> None:
        """Recover hidden backups before a read can report a row as missing.

        NEVER blocks on the writer lock by default (R7-1). ``wait`` is the
        acquisition budget: the default :data:`_READ_RECOVERY_UI_WAIT_S` is 0,
        i.e. ONE non-blocking attempt, because this runs synchronously on the
        TUI event loop through ``_team_choices`` -> ``list_teams``. A crash
        artifact is present for the whole publish window of every healthy peer
        write, so a reader that waited turned an ordinary keystroke into a
        multi-second freeze.

        Failing to ACQUIRE and failing to RESTORE are deliberately different
        outcomes (R6-4 must survive R7-1's fix):

        * Lock unavailable — transient, and it means a compliant peer is
          holding it, so the artifact is very likely that peer's own in-flight
          publish rather than a crash. Serve the current on-disk view and try
          again later. Raising here would make a healthy concurrent write look
          like corruption.
        * Restore attempted and FAILED — authoritative. ``_recover_interrupted
          _swap_locked`` raises :class:`TeamRegistryRecoveryError` and it
          propagates, so no read can present a durable row as missing.

        A proven failure latches the ATTEMPT RATE, never the OUTCOME (R8-1).
        The error text instructs the user to fix registry permissions and retry,
        so a latch that never re-attempts makes that instruction false: the
        registry is session-lifetime, and the TUI's ``_team_choices`` swallows
        the exception, so ``/team`` silently offered NO teams for the rest of
        the session while the row sat healthy on disk — the same "durable row
        presented as absent" mode R6-4/U6-1 exist to prevent. So a failed set is
        re-attempted once the cooldown elapses (and immediately when the set
        CHANGES), and every read inside the cooldown still fails closed on the
        latched error rather than softening to "no teams".
        """
        try:
            artifacts = frozenset(
                child.name for child in self.teams_dir.iterdir() if _backup_row_id(child)
            )
        except FileNotFoundError:
            return
        except OSError as exc:
            raise TeamRegistryRecoveryError(
                "Could not inspect interrupted team saves; fix registry access and retry"
            ) from exc
        if not artifacts:
            # The tree is clean, so a later artifact is genuinely new and must
            # not be held off by state this pass would otherwise leave set.
            self._recovery_attempted_for = None
            self._recovery_failed_for = None
            self._recovery_failure = None
            return
        # A set we have already PROVEN unrecoverable (R6-4). Held as state, not
        # as a verdict: it decides what a read INSIDE the cooldown answers, and
        # never whether a read past the cooldown re-attempts.
        proven_failed = (
            artifacts == self._recovery_failed_for and self._recovery_failure is not None
        )
        # Bound the ATTEMPTS. An unchanged artifact set we tried to heal
        # recently and could not even get the lock for is not worth re-probing
        # on every keystroke; a CHANGED set is a new crash and is attempted
        # immediately.
        now = time.monotonic()
        if (
            artifacts == self._recovery_attempted_for
            and now - self._recovery_attempted_at < _READ_RECOVERY_COOLDOWN_S
        ):
            if proven_failed:
                # Not re-probing is a rate decision; it cannot downgrade a
                # proven failure into "serve the current view", which would be
                # the U6-1 silent-empty answer with the row still on disk.
                raise self._recovery_failure_repeat()
            return
        self._recovery_attempted_for = artifacts
        self._recovery_attempted_at = now
        budget = _READ_RECOVERY_UI_WAIT_S if wait is None else wait
        try:
            with self._persistence_lock(wait=budget):
                # The lock performs recovery before yielding. Reload while still
                # serialized so the read adopts exactly the healed tree.
                self._load()
        except TeamRegistryLockTimeout:
            # Transient contention only — see the docstring. The current
            # snapshot stands and the next read past the cooldown retries.
            if proven_failed:
                # ...unless this set is already proven unrecoverable. Losing a
                # lock race does not un-prove that, and answering from the
                # snapshot here would hide the row exactly as R6-4 forbids.
                raise self._recovery_failure_repeat() from None
            return
        except TeamRegistryRecoveryError as exc:
            # Restore was attempted and genuinely failed. Latch the exact
            # artifact set so reads inside the cooldown cannot turn into a
            # silent empty answer, and let the caller see the guidance. `exc` is
            # a fresh object per attempt, so replacing the stored one here also
            # keeps the retained traceback bounded (R8-2).
            self._recovery_failed_for = artifacts
            self._recovery_failure = exc
            raise
        # Recovery succeeded, so the artifacts it healed are gone. Clear the
        # latches rather than leaving them pointing at names that no longer
        # exist, which would suppress the immediate attempt a genuinely new
        # crash carrying the SAME row id deserves.
        self._recovery_attempted_for = None
        self._recovery_failed_for = None
        self._recovery_failure = None

    def list_teams(self, *, recovery_wait: float | None = None) -> list[Team]:
        """Every team, metadata only, sorted by name.

        ``recovery_wait`` is the read-path lock budget (R7-1). It defaults to a
        strictly NON-BLOCKING attempt, which is what EVERY caller reachable from
        an event loop must use: the TUI's `/team` picker calls this from a
        keystroke handler, and the ``team`` tool is awaited on the session's
        loop, so a bounded wait in either freezes the app.

        Only the ``teams`` CLI passes :data:`_READ_RECOVERY_CLI_WAIT_S`. It owns
        its process, blocks no UI, and is a ONE-SHOT: losing a lock race there
        means the user's single command silently fails to heal the tree, with no
        later read to retry. U6-1 itself does not depend on the wait — an
        interrupted save with no peer contending is recovered by the
        non-blocking attempt on the first try, on every path.
        """
        self._refresh_if_needed(wait=recovery_wait)
        return sorted(self._teams.values(), key=lambda team: team.name.lower())

    def _load_briefs(self, team: Team) -> Team:
        """Populate a metadata-only team the first time a full lookup needs it.

        R5-2: a revision publish is a whole-DIRECTORY rename, so a path-based
        read here can straddle it — metadata already refreshed from the new
        revision while the brief files resolve into the old one (or into a
        missing path, reading as ""). The briefs are therefore read through a
        directory descriptor PINNED to the row, and the metadata is re-read
        through that same descriptor and compared: if the directory was
        replaced between the cache refresh and this hydration, the pinned
        metadata disagrees with the cached row and the registry re-resolves
        the row instead of marrying old metadata to new briefs. Either way the
        caller receives one consistent revision.
        """
        if team.id in self._briefs_loaded:
            return team
        team_id = validate_team_id(team.id)
        team_dir = self.teams_dir / team_id
        if _DIR_FD_READS:
            # A revision publish has a tiny window where the target name does
            # not exist (between ``target -> backup`` and ``staged -> target``).
            # Hydrating into that window must NOT record empty briefs as
            # authoritative: retry briefly, and if the row is still absent
            # leave it UNLOADED so the next lookup re-hydrates against the
            # published revision.
            for attempt in range(3):
                try:
                    directory_fd = _open_row_directory(team_dir)
                except FileNotFoundError:
                    if attempt < 2:
                        time.sleep(0.005)
                        continue
                    return team
                except OSError:
                    break
                try:
                    metadata_text, instructions, project = _read_row_through_fd(directory_fd)
                    pinned = _parse_metadata_text(metadata_text)
                    if pinned is not None and pinned.id == team_id:
                        # Adopt the complete pinned model. Selected-field
                        # comparisons let manager/member/created changes mix
                        # with briefs from another revision.
                        pinned.instructions = instructions
                        pinned.project = project
                        self._teams[team_id] = pinned
                        self._briefs_loaded.add(team_id)
                        return pinned
                finally:
                    os.close(directory_fd)
                if attempt < 2:
                    time.sleep(0.005)
                    continue
                break
        # Windows lacks openat pinning. Verify metadata bytes and directory
        # identity before/after all reads; any swap seam invalidates the sample.
        for attempt in range(_ROW_SNAPSHOT_ATTEMPTS):
            snapshot = _read_row_snapshot(team_dir, team_id)
            if snapshot is not None:
                self._teams[team_id] = snapshot
                self._briefs_loaded.add(team_id)
                return snapshot
            if attempt + 1 < _ROW_SNAPSHOT_ATTEMPTS:
                time.sleep(_ROW_SNAPSHOT_RETRY_S)
        return team

    def _hydrate_briefs_for_save(self, team: Team) -> Team:
        """Make ``team`` safe to persist: its briefs must be REAL values.

        R2-1: the loaded/unloaded distinction is registry-owned
        (``_briefs_loaded``), so a ``Team`` that reached this registry from
        ANYWHERE but this registry's own ``get_team*`` — a ``list_teams``
        metadata row, or a model dumped to JSON/Python and revalidated by a
        tool or transport — carries ``""`` briefs that mean "never read",
        not "deliberately emptied". Writing those verbatim truncated both
        brief files. This method is the single choke point that resolves the
        ambiguity the ONLY way that cannot lose data: it reads the brief
        files this registry has on disk and merges them into the object,
        then records the id as loaded so the write persists exactly what was
        on disk.

        A team whose briefs this registry HAS loaded (``get_team*`` result,
        ``create_team``/``update_team`` output, or a previous save) keeps its
        in-memory values untouched: the caller is editing briefs it actually
        read, so its strings are authoritative and are written as-is. An
        explicit clear is authored through :meth:`update_team`, which hydrates
        first and then applies ``instructions=""`` onto the loaded object —
        reaching this method already loaded, so the empty string persists.

        Reads are best-effort like ``_load_briefs``: an unreadable file logs
        and saves empty rather than failing the whole metadata save, matching
        how ``_load`` treats bad YAML.

        A team id this registry has NEVER SEEN (not in ``_teams``, no brief
        files on disk) is a CREATE: the caller-supplied briefs are the only
        briefs that will ever exist, so they are kept verbatim and the id is
        simply marked loaded. Without this guard the fresh, empty directory
        the save is about to make would be read back as two absent files and
        the caller's briefs would be replaced with ``""`` on the very first
        write — the exact truncation this method exists to prevent, reached
        from the other side.
        """
        canonical = self._teams.get(team.id)
        if team.id in self._briefs_loaded and team is canonical:
            # Loadedness authorizes only the exact canonical object this
            # registry hydrated. A model-dumped/revalidated COPY with the same
            # id is not evidence that its "" briefs are intentional; trusting
            # the id alone would reintroduce R2-1 whenever a registry hydrated
            # the original before receiving the transported copy.
            return team
        team_id = validate_team_id(team.id)
        team_dir = self.teams_dir / team_id
        # R5-2: this runs UNDER the persistence lock, so no compliant writer can
        # swap the row mid-read; the pinned-fd read is still used because it
        # costs nothing here and keeps the read consistent even if a peer from
        # a pre-lock release is mid-swap.
        on_disk_instructions, on_disk_project = self._read_briefs_pinned(team_dir)
        if (
            canonical is not None
            or (team_dir / "instructions.md").is_file()
            or (team_dir / "project.md").is_file()
        ):
            # An existing entry: the disk files are the authoritative briefs
            # for an object that never read them.
            team.instructions = on_disk_instructions
            team.project = on_disk_project
        # Loadedness is adopted only after publication succeeds. Marking the
        # canonical ID here would poison a failed save: the cache still carries
        # metadata-only blank briefs but future reads would trust them as loaded.
        return team

    def _read_briefs_pinned(self, team_dir: Path) -> tuple[str, str]:
        """Read both briefs through one pinned row-directory descriptor."""
        if _DIR_FD_READS:
            try:
                directory_fd = _open_row_directory(team_dir)
            except OSError:
                return _read_optional(team_dir / "instructions.md"), _read_optional(
                    team_dir / "project.md"
                )
            try:
                return (
                    _read_optional_at(directory_fd, "instructions.md"),
                    _read_optional_at(directory_fd, "project.md"),
                )
            finally:
                os.close(directory_fd)
        return _read_optional(team_dir / "instructions.md"), _read_optional(team_dir / "project.md")

    def get_team(self, team_id: str, *, recovery_wait: float | None = None) -> Team:
        team_id = validate_team_id(team_id)
        self._refresh_if_needed(wait=recovery_wait)
        team = self._teams.get(team_id)
        if team is None:
            raise KeyError(f"Team with id {team_id} not found")
        return self._load_briefs(team)

    def _find_cached_team_by_name(self, name: str) -> Team | None:
        """Find metadata in the current snapshot without refresh or hydration.

        Mutation paths must validate collisions against the SAME snapshot as
        the object they mutate. Calling a public getter mid-mutation can refresh
        ``_teams`` and orphan an already-hydrated canonical object.
        """
        key = (name or "").strip().casefold()
        if not key:
            return None
        return next((team for team in self._teams.values() if team.name.casefold() == key), None)

    def get_team_by_name(self, name: str, *, recovery_wait: float | None = None) -> Team | None:
        self._refresh_if_needed(wait=recovery_wait)
        team = self._find_cached_team_by_name(name)
        return self._load_briefs(team) if team is not None else None

    @contextmanager
    def _persistence_lock(self, *, wait: float | None = None) -> Iterator[None]:
        """Serialize one registry mutation across processes.

        The sidecar lives at config level rather than inside ``teams/`` so it
        can never be mistaken for a team row. Construction remains read-only:
        the config directory and lock file appear only on the first mutation.

        Registry methods are synchronous local-filesystem calls already, so a
        short synchronous wait matches their API. As with the project's OAuth
        and process-ledger locks, each kernel attempt is NON-BLOCKING and the
        retry is bounded; a dead peer can never park the caller indefinitely.
        A timeout raises :class:`TeamRegistryLockTimeout` so the CLI/tool
        boundaries can present contention as a recoverable state instead of a
        crash traceback (U5-1).

        ``wait`` overrides how long acquisition may take, for the READ path
        only (R7-1). Mutations keep the full :data:`_TEAM_LOCK_TIMEOUT_S`
        because a create/update/delete that gives up has failed to do the thing
        the user asked for; a read that gives up simply serves the snapshot it
        already has, which is always a truthful answer. ``wait=0`` makes exactly
        one non-blocking attempt, which is what a caller on the UI event loop
        must use — see :data:`_READ_RECOVERY_UI_WAIT_S`.
        """
        self.config_dir.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.config_dir / ".teams.lock", os.O_CREAT | os.O_RDWR, 0o600)
        acquired = False
        budget = _TEAM_LOCK_TIMEOUT_S if wait is None else max(0.0, wait)
        deadline = time.monotonic() + budget
        try:
            while not acquired:
                acquired = _try_lock_exclusive(fd)
                if acquired:
                    break
                if time.monotonic() >= deadline:
                    raise TeamRegistryLockTimeout(
                        "Timed out waiting for the teams registry lock; "
                        "retry after the other lop process finishes"
                    )
                time.sleep(_TEAM_LOCK_RETRY_S)
            # R5-2 recovery: a writer that died between ``target -> backup`` and
            # ``staged -> target`` leaves the row only in the hidden backup.
            # Every locked mutation/load passes here first, so the next
            # compliant process restores it before touching the registry.
            self._recover_interrupted_swap_locked()
            yield
        finally:
            if acquired:
                _unlock(fd)
            os.close(fd)

    def create_team(self, fields: TeamEditFields) -> Team:
        name = validate_team_name(fields.name or "")
        with self._persistence_lock():
            # The refresh is unconditional and occurs inside the same lock as
            # validation and publication. Interval-gated snapshots cannot prove
            # uniqueness when another process has written since our last read.
            self._load()
            team = Team(
                id=str(uuid.uuid4()),
                name=name,
                created_date=datetime.now(timezone.utc),
                description=(fields.description or "").strip(),
                manager=(fields.manager or "manager").strip() or "manager",
                members=list(fields.members or []),
                instructions=(
                    _bounded(fields.instructions) if fields.instructions is not None else ""
                ),
                project=_bounded(fields.project) if fields.project is not None else "",
            )
            return self._save_team_locked(team, briefs_authoritative=True)

    def update_team(self, team_id: str, fields: TeamEditFields) -> Team:
        team_id = validate_team_id(team_id)
        with self._persistence_lock():
            # Refresh exactly once under the writer lock, then mutate and save
            # that canonical object without re-locking or replacing the cache.
            self._load()
            current = self._teams.get(team_id)
            if current is None:
                raise KeyError(f"Team with id {team_id} not found")
            current = self._load_briefs(current)
            # Keep the canonical cache on the last acknowledged durable row.
            # Stage all edits on a detached candidate and adopt only after the
            # directory transaction completes successfully.
            candidate = current.model_copy(deep=True)

            updates = fields.model_dump(exclude_unset=True)
            if "name" in updates and updates["name"] is not None:
                new_name = validate_team_name(updates["name"])
                occupant = self._find_cached_team_by_name(new_name)
                if occupant is not None and occupant.id != team_id:
                    raise ValueError(f"Team with name {new_name} already exists")
                candidate.name = new_name
            if "description" in updates and updates["description"] is not None:
                candidate.description = updates["description"].strip()
            if "manager" in updates and updates["manager"] is not None:
                manager = updates["manager"].strip()
                if not manager:
                    raise ValueError("manager is required")
                candidate.manager = manager
            if "members" in fields.model_fields_set and fields.members is not None:
                # ``model_dump`` recursively turns Pydantic children into dicts.
                # Keep the validated TeamMember objects: roster rendering and
                # orchestration call ``member.role`` / ``member.count`` immediately
                # after an update, before a reload can rehydrate them from YAML.
                candidate.members = list(fields.members)
            if "instructions" in updates and updates["instructions"] is not None:
                # An explicit "" is a DELIBERATE clear. This path hydrated first,
                # so the empty string is authoritative rather than transported.
                candidate.instructions = _bounded(updates["instructions"])
            if "project" in updates and updates["project"] is not None:
                candidate.project = _bounded(updates["project"])
            return self._save_team_locked(candidate, briefs_authoritative=True)

    def save_team(self, team: Team) -> Team:
        """Write ``team`` to disk and adopt it as this registry's current row.

        CONSTRAINT (R2-1): the loaded/unloaded brief state is registry-owned,
        so this method cannot tell from the OBJECT whether its ``""`` briefs
        are "never read" or "deliberately emptied" — a model-dumped,
        revalidated team has lost the distinction by construction. It
        therefore resolves the ambiguity the safe way: for an id this
        registry has not loaded, the on-disk brief files are merged into the
        object before anything is written (``_hydrate_briefs_for_save``), so
        a metadata-only ``list_teams`` row — or a round-tripped copy of one —
        preserves both briefs while its metadata edits persist. A team this
        registry HAS loaded (a ``get_team*`` result the caller edited, or the
        output of ``create_team``/``update_team``) saves its briefs verbatim.
        To CLEAR a brief, use :meth:`update_team` with an explicit empty
        string: that path hydrates first, so the clear lands on a loaded
        object and persists.
        """
        # Loadedness authorizes only the canonical object this registry actually
        # hydrated. Preserve that fact across the mandatory disk refresh; a
        # transported copy with the same id remains untrusted by construction.
        try:
            # Assignment validation is intentionally not enabled on this shared
            # model, so validate a detached transport copy before any filesystem
            # operation. On failure, restore this registry's canonical snapshot
            # from disk because the caller may have mutated that exact object.
            candidate = Team.model_validate(team.model_dump())
            team_id = validate_team_id(candidate.id)
        except ValueError:
            self._load()
            raise
        briefs_authoritative = team_id in self._briefs_loaded and team is self._teams.get(team_id)
        # Direct callers may edit the canonical result before save. Reloading
        # restores the cache from disk, while this detached candidate prevents a
        # failed publication from re-adopting the rejected in-memory revision.
        with self._persistence_lock():
            self._load()
            saved = self._save_team_locked(candidate, briefs_authoritative=briefs_authoritative)
        # Existing callers reasonably keep the object they passed. Reflect the
        # now-durable hydrated briefs only after publication succeeds; failure
        # still leaves both the canonical cache and caller object untouched.
        team.instructions = saved.instructions
        team.project = saved.project
        return saved

    def _save_team_locked(self, team: Team, *, briefs_authoritative: bool) -> Team:
        """Validate and publish ``team`` while ``_persistence_lock`` is held."""
        # Defense in depth for ``model_construct`` and validation-bypassing
        # transports: reject before mkdir, temp creation, rename, or cleanup.
        team_id = validate_team_id(team.id)
        name_key = team.name.casefold()
        occupant = next(
            (
                stored
                for stored in self._teams.values()
                if stored.id != team.id and stored.name.casefold() == name_key
            ),
            None,
        )
        if occupant is not None:
            raise ValueError(f"Team with name {team.name} already exists")

        # Hydrate BEFORE creating the final directory. An untrusted transported
        # row carries empty brief strings that mean "not loaded", while create
        # and update explicitly mark their already-known values authoritative.
        team_dir = self.teams_dir / team_id
        if team_dir.is_symlink():
            raise ValueError("refusing to save through a symlinked team row")
        if not briefs_authoritative:
            self._hydrate_briefs_for_save(team)
        payload = team.model_dump(mode="json", exclude={"instructions", "project"})
        metadata = yaml.safe_dump(payload, default_flow_style=False, sort_keys=False)
        try:
            self.teams_dir.mkdir(parents=True, exist_ok=True)
            # R5-2: BOTH the create and the update path publish the row as one
            # directory rename. Before this, an existing row was rewritten
            # file-by-file, so a failure (or a crash) between ``team.yml`` and
            # the briefs left readers with a MIXED revision — new name, old
            # briefs — that no retry could classify. Staging the complete new
            # directory first and swapping it in with a recoverable backup
            # means an unlocked reader sees the old complete row or the new
            # complete row, never a mixture.
            staging = Path(tempfile.mkdtemp(prefix=f".{team_id}.", dir=self.teams_dir))
            try:
                _write_row_files(staging, metadata, team)
                if team_dir.exists():
                    self._swap_row_directory_locked(staging, team_dir)
                else:
                    # Publish a new id as one directory rename so a failed
                    # create cannot leave a half-written team row visible on
                    # disk (and, since R5-1, staging is invisible to readers
                    # even before the rename).
                    os.replace(staging, team_dir)
                    try:
                        _fsync_dir(self.teams_dir)
                    except BaseException:
                        # A create is not acknowledged unless its directory entry
                        # is durable. Remove the unacknowledged row so a retry can
                        # use the same name without discovering phantom success.
                        shutil.rmtree(team_dir)
                        _fsync_dir(self.teams_dir)
                        raise
            except BaseException:
                shutil.rmtree(staging, ignore_errors=True)
                raise
        except Exception as exc:
            raise Exception(f"Failed to save team metadata: {exc}") from exc
        self._teams[team_id] = team
        self._briefs_loaded.add(team_id)
        return team

    def _swap_row_directory_locked(self, staging: Path, target: Path) -> None:
        """Replace the existing row directory ``target`` with ``staging``.

        Must be called with ``_persistence_lock`` held (the caller's
        ``_save_team_locked`` guarantees it). Sequence:

        1. ``target -> .<id>.backup.<rand>`` — the live row moves aside under
           a hidden name, so ``_load`` keeps ignoring it (R5-1) and ordinary
           unlocked readers simply miss the row for the gap rather than read
           a mixed one. "Hidden" is what makes the gap safe: a reader that
           cannot see the backup cannot hydrate its files as canonical.
        2. ``staging -> target`` — the new complete row appears atomically.
        3. remove the backup.

        If step 2 fails, the backup is renamed back before raising, so the
        durable row is the old complete one and the caller sees the error. If
        the process dies between 1 and 2, the next lock holder runs
        ``_recover_interrupted_swap_locked`` and restores the backup.

        POSIX offers no portable atomic directory EXCHANGE (``renameat2``
        ``RENAME_EXCHANGE`` is Linux-only and has no stdlib binding), so a
        reader hitting the exact gap between the two renames observes NO row
        rather than a mixed one; its next refresh recovers. That gap is the
        documented, accepted trade for cross-platform robustness — the lock
        plus hidden names mean no compliant reader can persist anything
        derived from the gap state.
        """
        backup = Path(tempfile.mkdtemp(prefix=f".{target.name}.backup.", dir=target.parent))
        # mkdtemp created the backup directory itself; an empty directory in
        # the way would make the target->backup rename fail, so remove it and
        # let the rename recreate the name.
        backup.rmdir()
        renamed_aside = False
        try:
            os.replace(target, backup)
            renamed_aside = True
            _fsync_dir(target.parent)
            os.replace(staging, target)
            _fsync_dir(target.parent)
        except BaseException:
            if renamed_aside:
                # A post-publish fsync failure is still a failed save. Remove
                # only our staged revision and restore the authoritative backup
                # so disk and cache both remain on the old row.
                if target.exists():
                    shutil.rmtree(target)
                if backup.exists():
                    os.replace(backup, target)
                    _fsync_dir(target.parent)
            raise
        shutil.rmtree(backup, ignore_errors=True)

    def _recover_interrupted_swap_locked(self) -> None:
        """Restore a row stranded by a crash between the swap's two renames.

        Runs under ``_persistence_lock`` before any mutation refreshes, so at
        most one process performs the recovery and no writer interleaves with
        it. A crash after ``target -> backup`` but before ``staged -> target``
        leaves the live row ONLY in the hidden backup; here ``target`` is
        missing and the backup is restored. If both exist (crash between the
        second rename and the backup cleanup), ``target`` is authoritative —
        it holds the newer complete revision — and the stale backup is
        removed. Scope is strictly ``.<id>.backup.*`` siblings of ``teams/``;
        nothing else is touched.
        """
        try:
            children = list(self.teams_dir.iterdir())
        except FileNotFoundError:
            return  # no teams tree yet: nothing to recover
        except OSError as exc:
            raise TeamRegistryRecoveryError(
                "Could not inspect interrupted team saves; fix access to "
                f"{self.teams_dir} and retry"
            ) from exc
        for child in children:
            row_id = _backup_row_id(child)
            if row_id is None:
                continue
            target = self.teams_dir / row_id
            try:
                if target.is_symlink():
                    raise OSError(errno.ELOOP, "published team row is a symlink")
                if target.exists():
                    # A target is authoritative only when it is a complete real
                    # row for this ID. A file/corrupt directory beside a backup
                    # cannot justify deleting the only known durable revision.
                    if not _published_row_matches(target, row_id):
                        raise OSError(errno.EINVAL, "published team row is incomplete or invalid")
                    # The valid target is the newer complete revision; the
                    # backup is an interrupted-cleanup leftover.
                    shutil.rmtree(child)
                    _fsync_dir(self.teams_dir)
                else:
                    if not _published_row_matches(child, row_id):
                        raise OSError(errno.EINVAL, "backup team row is incomplete or invalid")
                    logger.warning("recovering team row %s from interrupted save", row_id)
                    os.replace(child, target)
                    _fsync_dir(self.teams_dir)
            except OSError as exc:
                # A hidden authoritative row invalidates absence and uniqueness
                # answers. Abort every reader/writer until recovery succeeds.
                raise TeamRegistryRecoveryError(
                    f"Could not recover team {row_id!r} from an interrupted save; "
                    "fix registry permissions and retry"
                ) from exc

    def delete_team(self, team_id: str) -> None:
        team_id = validate_team_id(team_id)
        with self._persistence_lock():
            # Delete participates in the same ordering as create/save so a
            # delete-recreate-stale-save sequence has one unambiguous winner.
            self._load()
            if team_id not in self._teams:
                raise KeyError(f"Team with id {team_id} not found")
            # Remove the on-disk copy FIRST: if rmtree fails, cache still agrees
            # with disk and the row remains visible after the exception.
            team_dir = self.teams_dir / team_id
            if team_dir.is_symlink():
                raise ValueError("refusing to delete a symlinked team row")
            if team_dir.exists():
                shutil.rmtree(team_dir)
            self._teams.pop(team_id)
            self._briefs_loaded.discard(team_id)


def parse_members(raw: Iterable[str] | None) -> list[TeamMember]:
    """Parse ``coder`` / ``coder:2`` / ``team:pod`` / ``team:pod:2`` tokens.

    Two members of the same role collapse into one slot with a summed count
    so a caller can pass ``--member coder --member coder`` or ``coder:2``.

    A leading ``team:`` prefix marks the slot as a nested TEAM (an org): the
    name after the prefix is a team name, and ``team:pod:2`` is two independent
    copies of the ``pod`` sub-org. This is the ONE place the tool authors
    nesting; a bare token (no ``team:`` prefix) stays ``kind='agent'`` so the
    existing ``coder`` / ``coder:2`` grammar is untouched. Agent and team slots
    live in one keyed namespace here (``(kind, role)``) so a member ``pod`` and
    a sub-team ``pod`` do not collapse into each other.
    """
    slots: dict[tuple[Literal["agent", "team"], str], int] = {}
    order: list[tuple[Literal["agent", "team"], str]] = []
    for token in raw or ():
        text = (token or "").strip()
        if not text:
            continue
        kind: Literal["agent", "team"] = "agent"
        # ``team:`` is a case-insensitive prefix on the WHOLE token, stripped
        # before the ``role:count`` split so ``team:pod:2`` still parses its
        # count. A bare ``pod:2`` is unaffected — no prefix, stays an agent.
        if text[:5].lower() == "team:":
            kind = "team"
            text = text[5:].strip()
            if not text:
                raise ValueError(f"invalid team member {token!r}: no team name")
        role, sep, count_text = text.partition(":")
        role = role.strip()
        if not role:
            raise ValueError(f"invalid member {token!r}")
        count = 1
        if sep:
            try:
                count = int(count_text.strip())
            except ValueError as exc:
                raise ValueError(f"invalid member count in {token!r}") from exc
        if count < 1:
            raise ValueError(f"member count must be >= 1 in {token!r}")
        key = (kind, role)
        if key not in slots:
            order.append(key)
            slots[key] = 0
        slots[key] += count
    return [TeamMember(role=role, count=slots[(kind, role)], kind=kind) for (kind, role) in order]


def _bounded(text: str) -> str:
    body = text or ""
    if len(body) > MAX_TEAM_INSTRUCTIONS_CHARS:
        raise ValueError(
            f"team instructions exceed {MAX_TEAM_INSTRUCTIONS_CHARS} characters; "
            "they ride in front of every run of this team, so they must stay short."
        )
    return body


def _read_optional(path: Path) -> str:
    try:
        if path.is_symlink():
            logger.warning("refusing to follow symlinked team file %s", path)
            return ""
        if path.is_file():
            return path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        logger.warning("could not read %s", path)
    return ""


def _parse_metadata_text(text: str | None) -> Team | None:
    """Parse one captured metadata revision, returning None when invalid."""
    if not text:
        return None
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError:
        return None
    if not isinstance(data, dict):
        return None
    try:
        return Team.model_validate(data)
    except Exception:  # noqa: BLE001 - invalid metadata is an unavailable snapshot
        return None


def _read_row_through_fd(directory_fd: int) -> tuple[str | None, str, str]:
    """Pin every row file before reading so backup cleanup cannot mix a row.

    A directory fd survives rename, but that alone is not enough: after the new
    row publishes, backup cleanup can unlink the OLD directory's child files
    while a reader opens them sequentially. Open all three first, then reopen
    metadata as a liveness check. If cleanup raced any open, verification fails
    and hydration retries the current target instead of treating a vanished
    brief as an authored empty string. Once opened, file fds survive unlinking.
    """
    fds: list[int] = []
    try:
        metadata_fd = _open_optional_at(directory_fd, "team.yml")
        if metadata_fd is None:
            return None, "", ""
        fds.append(metadata_fd)
        instructions_fd = _open_optional_at(directory_fd, "instructions.md")
        project_fd = _open_optional_at(directory_fd, "project.md")
        # Every row this registry publishes contains all three files, including
        # explicit-empty briefs. A missing child on a pinned old directory is
        # therefore the signature of concurrent backup cleanup, not a valid
        # revision. Legacy rows missing optional briefs fall through after the
        # bounded retries to the stable path snapshot, where absence means "".
        if instructions_fd is None or project_fd is None:
            return None, "", ""
        fds.extend((instructions_fd, project_fd))
        verification_fd = _open_optional_at(directory_fd, "team.yml")
        if verification_fd is None:
            return None, "", ""
        fds.append(verification_fd)

        metadata_text = _read_text_fd(metadata_fd)
        if not metadata_text or metadata_text != _read_text_fd(verification_fd):
            return None, "", ""
        return (
            metadata_text,
            _read_text_fd(instructions_fd),
            _read_text_fd(project_fd),
        )
    finally:
        for fd in fds:
            os.close(fd)


def _open_optional_at(directory_fd: int, filename: str) -> int | None:
    """Open one no-follow row file relative to a pinned directory."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        return os.open(filename, flags, dir_fd=directory_fd)
    except OSError:
        return None


def _read_text_fd(fd: int) -> str:
    """Read a pinned text fd without taking ownership of the descriptor."""
    os.lseek(fd, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(fd, 64 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks).decode("utf-8-sig", errors="replace")


#: ``dir_fd`` reads need POSIX ``openat`` semantics. Windows lacks them, so
#: readers there use path reads (see ``_read_optional_at``).
_DIR_FD_READS = hasattr(os, "open") and os.name == "posix"


def _open_row_directory(team_dir: Path) -> int:
    """Open one real row directory without following a crafted symlink."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    return os.open(team_dir, flags)


def _backup_row_id(path: Path) -> str | None:
    """Return the safe row ID encoded by a real hidden backup directory."""
    match = re.fullmatch(r"\.([A-Za-z0-9][A-Za-z0-9._-]{0,127})\.backup\..+", path.name)
    if match is None:
        return None
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        return None
    try:
        return validate_team_id(match.group(1))
    except ValueError:
        return None


def _published_row_matches(team_dir: Path, expected_id: str) -> bool:
    """Return whether a recovery target is a real complete row for its ID."""
    if team_dir.is_symlink() or not team_dir.is_dir():
        return False
    metadata = team_dir / "team.yml"
    if metadata.is_symlink():
        return False
    try:
        team = _parse_metadata_text(metadata.read_text(encoding="utf-8-sig", errors="replace"))
    except OSError:
        return False
    return team is not None and team.id == expected_id


def _read_row_snapshot(team_dir: Path, expected_id: str) -> Team | None:
    """Read one complete path-based row revision or reject a racing sample."""
    try:
        before_stat = team_dir.stat(follow_symlinks=False)
        if not stat.S_ISDIR(before_stat.st_mode):
            return None
        metadata_path = team_dir / "team.yml"
        if metadata_path.is_symlink():
            return None
        metadata_before = metadata_path.read_bytes()
        instructions = _read_optional_strict(team_dir / "instructions.md")
        project = _read_optional_strict(team_dir / "project.md")
        metadata_after = metadata_path.read_bytes()
        after_stat = team_dir.stat(follow_symlinks=False)
    except OSError:
        return None
    identity_before = (before_stat.st_dev, before_stat.st_ino, before_stat.st_mtime_ns)
    identity_after = (after_stat.st_dev, after_stat.st_ino, after_stat.st_mtime_ns)
    if metadata_before != metadata_after or identity_before != identity_after:
        return None
    metadata_text = metadata_before.decode("utf-8-sig", errors="replace")
    team = _parse_metadata_text(metadata_text)
    if team is None or team.id != expected_id:
        return None
    team.instructions = instructions
    team.project = project
    return team


def _read_optional_strict(path: Path) -> str:
    """Read an optional brief while preserving non-absence failures for retry."""
    try:
        if path.is_symlink():
            raise OSError(errno.ELOOP, "team brief is a symlink")
        return path.read_text(encoding="utf-8-sig", errors="replace")
    except FileNotFoundError:
        return ""


def _read_optional_at(directory_fd: int, filename: str) -> str:
    """Read one optional file RELATIVE to a pinned directory descriptor.

    WHY (R5-2): a row revision is published by renaming a whole directory
    into place. A reader that resolves ``teams/<id>/instructions.md`` by PATH
    twice — once for metadata, once for a brief — can straddle the rename and
    read metadata from the OLD revision and briefs from the NEW one (or from
    a missing path, yielding ""). Reading through a directory fd opened ONCE
    pins the inode: after the rename the fd still addresses the same directory
    the metadata came from, so the three files are one consistent revision.
    """
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(filename, flags, dir_fd=directory_fd)
    except OSError:
        return ""
    try:
        with os.fdopen(fd, "r", encoding="utf-8-sig", errors="replace") as handle:
            return handle.read()
    except OSError:
        return ""
