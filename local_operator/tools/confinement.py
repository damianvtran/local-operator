"""How a confined session's local tools are kept inside their root.

WHY THIS EXISTS. A session's local tools -- the shell, the path readers and
writers -- run on the HOST, in the operator's own account. That is exactly
right for a session the operator opened. It is wrong for a session the
operator did not open and is not watching, and the benchmark's session
engagement put a measured edge on "wrong": an episode's model ran

    find /Users/damian/worktrees/osworld ... -name prize-items.pdf
    pdftotext -layout /Users/damian/worktrees/osworld/gated/assets/task_013/...

and read a task's input out of the campaign apparatus's gated assets -- a
tree that also holds the adapter build and other tasks' material. Nothing
about the task needed it; the model simply had a shell on a host whose
filesystem contained the answer key. An arm whose results are meant to be
comparable with a published benchmark cannot let its model leave the episode.

THE MECHANISM, and why it is a kernel boundary rather than a path check. A
tool that takes one path (``read``) can be checked exactly: resolve it, refuse
anything outside the root, and symlinks are covered because the check runs on
the resolved path. A SHELL cannot be checked that way -- ``cd``, ``$VAR``,
command substitution and a built string all construct paths the checker never
saw, and a restriction that "holds against absolute paths" by parsing command
text is a restriction that leaks. So shell children are confined by the
kernel: on macOS every command runs under a profile-based seatbelt sandbox
(``/usr/bin/sandbox-exec``) whose read allowlist is the root plus the system's
own executable machinery, whose write allowlist is the root plus the null
devices, and whose default is DENY. The enforcement point is the syscall, so
absolute paths, ``cd``, symlinks (the vnode is what the kernel checks) and
constructed strings are all the same case. Where no such mechanism exists the
confinement FAILS CLOSED: the command is refused with the reason named,
because a restriction that looks like a jail but leaks is worse than a clear
refusal -- it would read as a solved problem.

WHAT THIS DOES NOT COVER, stated rather than implied:

* Reads of SYSTEM paths stay allowed -- ``/usr``, ``/System``, ``/Library``,
  ``/opt/homebrew``, ``/private/etc`` and the per-user temp and dyld caches.
  Running an interpreter at all requires reading the interpreter; these are
  the host's machinery, not the operator's data. The operator's own home
  (outside the root), other volumes, ``/tmp`` and every other user's files
  are NOT in the allowlist and the kernel denies them.
* Network is deliberately unchanged (``network-outbound`` is allowed): this
  boundary is about the filesystem, and an episode's shell needing to reach a
  package index is not the leak this closes.
* The confinement binds a session's OWN tools and the children it spawns
  (``task`` children inherit it). A same-account process that is not part of
  the session -- the operator's own terminal, a daemon -- is untouched; this
  is a boundary around the session, not around the machine.
* Tool surfaces that are host CAPABILITIES by design -- the browser tool (a
  real browser on the operator's machine), the console tool, peer messaging
  -- are not confined by this module and are named as out of scope for it.
  Where they write a file the path IS checked (see ``builtin``'s uses of
  :meth:`ToolConfinement.path_denial`).
* ``/dev`` reads are allowed wholesale (devices a child legitimately opens:
  null, zero, random, tty, fd). Raw disk devices are root-only by their own
  permissions, and this module does not attempt to re-den them.
* Names, not content: ``file-read-metadata`` is allowed on the WHOLE tree,
  because path resolution (``realpath``, ``namei`` walks, DNS) stats
  ancestors, and without it ordinary commands break (see :meth:`profile`). A
  confined child can therefore learn that a path EXISTS and its size/mtime;
  it cannot read file CONTENT or list directories outside the allowlist.
* Mach services stay reachable (``mach-lookup`` is allowed: the runtime needs
  it to function at all), so credential stores and daemons behind Mach --
  securityd included -- answer requests under their OWN access controls.
  Ordering Mach service registration is a follow-up; this module makes no
  claim about it.

USING IT. The enforcement points all read one answer -- ``ToolContext.
confinement_root`` -- so a session that carries a confinement can never have
half its tools agree and half not:

* ``Session.set_tool_confinement(root)`` installs one (and ``task`` children
  copy it at construction);
* the ``bash`` tool wraps its spawn with :meth:`ToolConfinement.wrap` and
  refuses when that returns ``None``;
* the path tools call :meth:`ToolConfinement.path_denial` on the path their
  own resolver produced.

The profile's read allowlist is built ONCE per confinement and passed to
sandbox-exec inline (``-p``), never through a scratch file: a file inside the
very tree being jailed would let a command rewrite the profile its successors
run under.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only; avoids an import cycle at runtime
    from local_operator.harness.types import ToolContext

#: The platform's profile-based kernel sandbox. macOS-only by construction:
#: the Linux equivalent (Landlock/``bwrap``) is a named follow-up, and a host
#: with neither refuses shell children instead of pretending (see
#: :meth:`ToolConfinement.spawn_refusal`).
SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")

#: Host machinery a child needs to run at all, granted READ-ONLY. Every entry
#: is the implementation of "the system", not user data: the system and
#: Homebrew trees hold the executables, ``/private/etc`` holds system config,
#: and the two ``/private/var`` subtrees are dyld's caches and the per-user
#: temp/cache area ordinary commands write scratch state into (the session's
#: OWN temp is redirected into the confinement root -- see the ``bash`` tool).
_SYSTEM_READ_SUBPATHS: tuple[str, ...] = (
    "/usr",
    "/bin",
    "/sbin",
    "/System",
    "/Library",
    "/opt/homebrew",
    "/usr/local",
    "/private/etc",
    "/private/var/db",
    "/private/var/folders",
    # The system shell SELECTOR (/bin/sh reads /private/var/select/sh at
    # startup): without it every command prints "Error opening
    # /private/var/select/sh: Operation not permitted" before doing its real
    # work -- noise, not a failure, but noise every reader would chase.
    "/private/var/select",
    "/dev",
)

#: Devices a shell child legitimately writes. ``/dev/fd`` is covered by the
#: read allowlist and by the process's own descriptors; writes go to the null
#: device and the terminal.
_SYSTEM_WRITE_LITERALS: tuple[str, ...] = (
    "/dev/null",
    "/dev/tty",
    "/dev/stdout",
    "/dev/stderr",
)


def _sandbox_reason_unavailable() -> str | None:
    """Why this host cannot sandbox a child, or ``None`` when it can."""

    if sys.platform != "darwin":
        return f"no kernel sandbox is implemented for {sys.platform} yet"
    if not SANDBOX_EXEC.exists():
        return f"{SANDBOX_EXEC} is not present on this host"
    return None


@dataclass(frozen=True)
class ToolConfinement:
    """One session's confinement root and the decisions that follow from it."""

    root: Path

    @classmethod
    def at(cls, root: str | Path) -> ToolConfinement:
        """Build a confinement for ``root``, fully resolved.

        Resolution matters: seatbelt compares vnode paths, and the file-tool
        checks compare against this root, so both must be the REAL path --
        a ``/tmp``-style symlink spelling must not make the jail and the
        check disagree about the same directory.
        """

        return cls(root=Path(root).expanduser().resolve())

    @classmethod
    def from_context(cls, context: ToolContext | None) -> ToolConfinement | None:
        """The confinement a tool call carries, or ``None`` for a free session.

        ``getattr`` rather than a bare attribute read for the reason the bash
        tool's delegation marker uses one: the ``tests/e2e`` doubles and other
        duck-typed contexts are not required to have the field, and "no
        answer" must read as "not confined" here -- the confinement comes from
        a session that installed one, never from a default.
        """

        raw = getattr(context, "confinement_root", None)
        if not raw:
            return None
        return cls.at(raw)

    def contains(self, path: Path) -> bool:
        """Whether a RESOLVED path lies at or under the confinement root."""

        try:
            path.relative_to(self.root)
        except ValueError:
            return False
        return True

    def path_denial(self, path: Path, *, resolvable: bool = True) -> str | None:
        """The refusal for a path outside the root, or ``None`` when inside.

        ``path`` is the RESOLVED path the calling tool's own resolver already
        computed -- this module deliberately does not resolve again at a
        second moment (a symlink moved between the two would be a second
        answer), and it deliberately does not guess the raw spelling's meaning.
        An unresolvable path is denied: it cannot be shown to be inside.
        """

        if not resolvable:
            return (
                f"refused: this session is confined to {self.root}, and this path "
                "could not be resolved inside it"
            )
        if self.contains(path):
            return None
        return f"refused: this session is confined to {self.root}; {path} is outside it"

    def cwd_denial(self, cwd: str | None) -> str | None:
        """The refusal for a working directory outside the root, or ``None``."""

        if not cwd:
            return None
        try:
            resolved = Path(cwd).expanduser().resolve()
        except (OSError, ValueError):
            return (
                f"refused: this session is confined to {self.root} and its working "
                f"directory ({cwd}) cannot be resolved inside it"
            )
        if self.contains(resolved):
            return None
        return (
            f"refused: this session is confined to {self.root} and its working "
            f"directory ({cwd}) is outside it"
        )

    def wrap(self, argv: Sequence[str]) -> list[str] | None:
        """``argv`` wrapped in the host's kernel sandbox, or ``None``.

        ``None`` means "this host has no mechanism this module trusts"; the
        caller must refuse the command rather than run it unwrapped.
        """

        if _sandbox_reason_unavailable() is not None:
            return None
        return [str(SANDBOX_EXEC), "-p", self.profile(), *argv]

    def spawn_refusal(self) -> str:
        """The failure-closed sentence for a host that cannot enforce the root."""

        reason = _sandbox_reason_unavailable() or "an unknown reason"
        return (
            f"refused: this session is confined to {self.root} and this host "
            f"cannot enforce that boundary for shell children ({reason}); a "
            "confined session refuses to run commands it cannot confine"
        )

    def temp_dir(self) -> Path:
        """The session's temp directory INSIDE the root, created on demand."""

        temp = self.root / "tmp"
        temp.mkdir(parents=True, exist_ok=True)
        return temp

    def _machinery_read_subpaths(self) -> tuple[str, ...]:
        """The running interpreter's own tree, read-only.

        The frozen ``_SYSTEM_READ_SUBPATHS`` cannot know where THIS process's
        interpreter lives: a venv keeps it under the worktree, a uv-managed
        Python under the user's data dir. A shell child that re-enters Python
        (``python3 -c``, the eval worker, a script's shebang) must be able to
        read the interpreter it runs, and that is machinery -- not operator
        data -- wherever it happens to sit on disk. Best-effort by design: an
        interpreter path that cannot be resolved simply contributes nothing,
        and the allowlist stays correct for the binaries the system paths
        already cover.
        """

        try:
            executable = Path(sys.executable).resolve()
        except (OSError, ValueError):  # pragma: no cover - defensive
            return ()
        # ``<venv>/bin/python`` -> ``<venv>``; a bare system interpreter ->
        # its prefix, which the system paths already cover.
        return (str(executable.parent.parent),)

    def profile(self) -> str:
        """The seatbelt profile for this root, as sandbox-exec's ``-p`` text.

        ``deny default`` plus an explicit allowlist, with THREE structural
        pieces worth stating because each was measured while building this.
        The first: reading the ROOT DIRECTORY itself is allowed
        (``literal "/"``). Without it every command aborts before exec;
        measured while building this, the missing piece was the directory
        entry read of ``/``, not any executable.

        The second: ``file-read-metadata`` is allowed on the whole tree. PATH
        RESOLUTION stats ancestors -- ``realpath`` (homebrew python does it at
        startup), ``namei`` walks, and DNS all pass through ancestor metadata,
        and a content-only allowlist broke ordinary commands (``python3 -V``
        died with ``realpath: /opt/homebrew/bin/: Operation not permitted``
        and DNS resolution failed) until this rule was added. Metadata is
        existence/size/mtime of a NAME; CONTENT reads and DIRECTORY LISTINGS
        are still allowlisted, which is the line this boundary draws (see the
        module docstring's residuals).

        The third: the write allowlist is the root plus the null devices, and
        nothing else -- writes are the direction that matters most and they
        are the strictest list in the file.
        """

        read_subpaths = (
            str(self.root),
            *self._machinery_read_subpaths(),
            *_SYSTEM_READ_SUBPATHS,
        )
        reads = " ".join(f'(subpath "{path}")' for path in read_subpaths)
        writes = " ".join(f'(literal "{path}")' for path in _SYSTEM_WRITE_LITERALS)
        return (
            "(version 1)\n"
            "(deny default)\n"
            "(allow process-fork)\n"
            "(allow process-exec)\n"
            "(allow sysctl-read)\n"
            "(allow mach-lookup)\n"
            "(allow network-outbound)\n"
            '(allow file-read-metadata (subpath "/"))\n'
            f'(allow file-read* (literal "/") {reads})\n'
            f'(allow file-write* (subpath "{self.root}") {writes})\n'
        )


def confinement_of(context: ToolContext | None) -> ToolConfinement | None:
    """The one call site-facing spelling of :meth:`ToolConfinement.from_context`."""

    return ToolConfinement.from_context(context)
