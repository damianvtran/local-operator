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

THE NETWORK SIDE (landed 2026-09-28 -- the completion of the same
argument). An episode that can query the host's network is not comparable to
the published harnesses, none of which can: their numbers come from driving
the GUEST, never the host, and the session engagement measured its model doing
exactly the opposite -- ``web_search``/``web_fetch`` calls against
NeurIPS/ICML/ICLR award pages, alongside 27 ``bash`` calls. So the bench-side
boundary was extended the same way the filesystem one was: shell children
get NO network allowance (the profile's old ``(allow network-outbound)`` is
gone; ``(deny network*)`` rides the default deny), and the denial was
MEASURED, not assumed. A jailed child's TCP connects fail with ``EPERM``
against the open internet AND against the host's own loopback; UDP
``sendto`` fails the same way; ``getaddrinfo`` fails FAST with ``EAI_NONAME``
(no resolver round trip, no hang); even ``AF_UNIX`` connects are denied, so
the loopback services this machine runs -- the desktop app's bridge, cmux, a
local proxy, another session's socket -- are unreachable from the jail too.
The in-process half is refused rather than sandboxed: ``web_search``,
``web_fetch``, the ``read <url>`` sugar, ``web_read`` and ``browser`` all
refuse under confinement via :meth:`ToolConfinement.network_refusal`, because
a kernel sandbox cannot wrap a call the session's own process makes, and a
text-level check a tool could circumvent (a built URL, a redirect, a second
backend) is the leak-shaped restriction this module exists to avoid. The
fidelity the refusals trade -- shipped tools a real session has -- is a
DEFERRED FINDING that must ride the session-arm write-up: the arm's reduced
surface has to be stated next to its numbers, not discovered by a reader
comparing tool inventories. The GUEST's own network access is untouched by
all of this: an episode drives its VM through the apparatus (the action
bridge, the adapter worker), and the only caller of :meth:`ToolConfinement.
wrap` is the ``bash`` tool, so no apparatus process is inside the jail.

MACH SERVICES (landed 2026-10-26 -- the same boundary, closed on the IPC
side). The profile used to carry an unrestricted ``(allow mach-lookup)``, and
that one line reached around the filesystem and network denials into the
operator's GUI session: measured from inside the real profile,
``pbpaste`` exited 0 READING the operator's clipboard, ``pbcopy`` exited 0
OVERWRITING it, and ``security list-keychains`` reached securityd. It was
not hypothetical: across 180 recorded benchmark episodes, 13 ``pbpaste``
calls in 4 episodes read the operator's clipboard (none wrote it). The
allowance is now an allowlist of Mach GLOBAL NAMES
(``_MACH_LOOKUP_GLOBAL_NAMES``), DERIVED by measurement rather than copied
from Apple's ``system.sb``:

1. a representative confined workload (``python3`` incl. ``venv``/``ssl``/
   ``sqlite3``/locale/tz, ``git`` init/commit/clone/status, ``uv``,
   ``clang``/``cc``/``make``/``ld``, ``xcrun --find``/``--show-sdk-path``,
   ``node``, ``perl``, ``ruby``, ``swift``, ``pdftotext``, coreutils,
   ``tar``/``gzip``, ``curl`` and ``getaddrinfo`` for DNS) was run under the
   old allow-all profile and under a profile with NO mach allowance, and the
   outputs were diffed;
2. every lookup the no-mach run was refused was read back from launchd's
   ``denied lookup ... Sandbox restriction`` log lines (the kernel logs
   nothing for these; launchd does);
3. each candidate was then ABLATED one at a time against the allow-all
   output.

Only TWO names changed observable behaviour, so only two are allowed:
``com.apple.system.opendirectoryd.libinfo`` (user/group NAME resolution --
without it ``getpwuid`` raises ``KeyError`` and ``id``/``ls -l``/``whoami``
print bare numbers) and ``com.apple.system.opendirectoryd.membership``
(without it ``confstr(_CS_DARWIN_USER_TEMP_DIR)`` stops resolving, so
``clang`` fails "unable to make temporary file" and the ``/usr/bin`` xcrun
shims print "couldn't create cache file" on every ``python3``/``git`` when
``TMPDIR`` is unset; the ``bash`` tool sets it, a script may not). With those two the
workload's output is identical to the allow-all run's. Everything else the
machinery asked for is denied and changed nothing measurable: ``logd``,
``system.logger``, ``notification_center`` (notifyd), ``bsd.dirhelper``,
``analyticsd``, ``SystemConfiguration.configd`` (curl's proxy probe; DNS
still fails FAST), ``coreservicesd``/``diskarbitrationd`` (swift-driver).
The denied set the brief named is pinned by tests with a ctypes
``bootstrap_look_up`` that only LOOKS UP a name and never messages it:
``com.apple.pasteboard.1``, ``com.apple.SecurityServer``,
``com.apple.securityd.xpc``, ``com.apple.coreservices.launchservicesd``,
``com.apple.lsd.open``/``.mapdb``, ``com.apple.coreservices.appleevents``,
``com.apple.windowserver.active``, ``com.apple.tccd`` all return
``BOOTSTRAP_NOT_PRIVILEGED`` (1100) from inside the jail.

Residuals of the Mach side, stated rather than implied: the opendirectoryd
libinfo service answers DIRECTORY records (local user and group names, home
paths, memberships) -- account metadata, not file content, and the price of
``getpwuid`` working; the confined child's own ``os_log`` output is dropped
(logd is denied); ``mdls`` (Spotlight via ``metadata.mds``) and
``defaults`` (cfprefsd) no longer answer, the latter already failing under
the old profile; and the allowlist was measured on the build host's macOS
(27.0), so a future release that moves libinfo behind another name would show
up as "uid not found" in a confined child, not as a leak.

WHAT THIS DOES NOT COVER, stated rather than implied:

* Reads of SYSTEM paths stay allowed -- ``/usr``, ``/System``, ``/Library``,
  ``/opt/homebrew``, ``/private/etc``, the dyld caches, and
  ``/private/var/folders`` (the per-user temp: READS of it are permitted,
  WRITES are not -- the session's own temp is redirected INSIDE the root,
  see the ``bash`` tool). Running an interpreter at all requires reading the
  interpreter; these are the host's machinery, not the operator's data. The
  operator's own home (outside the root), other volumes, ``/tmp`` and every
  other user's files are NOT in the allowlist and the kernel denies them.
* An ALREADY-OPEN connection handed into the jail keeps working: measured (a
  connected fd passed to a sandboxed child sent and received normally --
  seatbelt checks the connect, not the send). Nothing hands one in -- children
  are spawned with only stdio pipes, every other fd closed -- so this is a
  property of the mechanism, not of this deployment.
* The confinement binds a session's OWN tools and the children it spawns
  (``task`` children inherit it). A same-account process that is not part of
  the session -- the operator's own terminal, a daemon -- is untouched; this
  is a boundary around the session, not around the machine.
* Tool surfaces that are host CAPABILITIES by design -- the console tool
  (a pty inside the desktop app) and peer messaging -- are not confined by
  this module and are named as out of scope for it. Where they write a file
  the path IS checked (see ``builtin``'s uses of
  :meth:`ToolConfinement.path_denial`). The BROWSER tool is refused under
  confinement rather than left out of scope: it is a per-session tool that
  can exist in an episode (its cmux arm is a PATH lookup, measured true even
  under ``env -i`` on this fleet) and every one of its actions reaches the
  host's network on the episode's behalf. The console and peer surfaces are
  effectively absent where it matters -- both resolve through the SESSION'S
  config dir, which for an episode is the run scratch, and no console host
  or peer registry is published there -- but that is scoping, not a
  boundary; treat them as UNVERIFIED-in-episode and name them in the
  write-up alongside the network refusals.
* ``/dev`` reads are allowed wholesale (devices a child legitimately opens:
  null, zero, random, tty, fd). Raw disk devices are root-only by their own
  permissions, and this module does not attempt to re-den them.
* ``mktemp``'s macOS defaults: the system ``mktemp`` consults the per-user
  confstr temp, NOT ``$TMPDIR``, so bare ``mktemp`` and ``mktemp -t`` fail
  inside the jail (the write deny is by design). Working forms a confined
  script should use: ``mktemp -p "$TMPDIR" <template>``, a relative template
  (``mktemp ./x.XXXXXX``), or Python's ``tempfile`` (it honours ``TMPDIR``,
  which the ``bash`` tool redirects into the root). Stated because a model
  that reaches for plain ``mktemp`` burns a step discovering this.
* Names, not content: ``file-read-metadata`` is allowed on the WHOLE tree,
  because path resolution (``realpath``, ``namei`` walks, DNS) stats
  ancestors, and without it ordinary commands break (see :meth:`profile`). A
  confined child can therefore learn that a path EXISTS and its size/mtime;
  it cannot read file CONTENT or list directories outside the allowlist.
* Mach services beyond the two-name allowlist are denied (see MACH SERVICES
  above); the directory records the allowed libinfo service answers are the
  residual.
* DAEMON-MEDIATED SPAWN AND AUTOMATION -- the family where a request leaves the
  jail through a host service rather than a syscall. **Tested:** ``launchctl
  submit`` is refused inside the jail (rc=1, no job registered); ``launchctl
  print gui/<uid>`` answers metadata; ``open -a <missing app>`` prints the
  ordinary "Unable to find application named" (not a sandbox denial) -- and
  the SAME text appears under a blanket ``(deny mach-lookup)`` profile, so the
  missing-app path is not evidence either way. **NOT tested, deliberately:**
  whether ``open -a <existing app>`` is actually honoured from inside the jail,
  and whether Apple Events (``appleevent-send``) are reachable -- a real launch
  or an Apple Event can raise an app window or a TCC consent prompt on the
  operator's screen, which the campaign's own noise rules forbid. Treat app
  launch / URL open / automation from a confined child as UNVERIFIED until a
  bench-side run answers it (start from the blanket-mach probe above); named
  follow-up.

USING IT. The enforcement points all read one answer -- ``ToolContext.
confinement_root`` -- so a session that carries a confinement can never have
half its tools agree and half not:

* ``Session.set_tool_confinement(root)`` installs one (and ``task`` children
  copy it at construction);
* the ``bash`` tool wraps its spawn with :meth:`ToolConfinement.wrap` and
  refuses when that returns ``None``;
* the path tools call :meth:`ToolConfinement.path_denial` on the path their
  own resolver produced;
* the in-process network tools call :meth:`ToolConfinement.network_refusal`
  before any service is constructed, so the refusal costs no connection.

The profile's read allowlist is built ONCE per confinement and passed to
sandbox-exec inline (``-p``), never through a scratch file: a file inside the
very tree being jailed would let a command rewrite the profile its successors
run under.
"""

from __future__ import annotations

import asyncio
import functools
import subprocess
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

#: The ONLY Mach services a confined child may look up. Derived by measurement
#: (allow-all vs no-mach diff, launchd's ``denied lookup`` log, then one-at-a-
#: time ablation -- see the module docstring's MACH SERVICES section): these
#: two are the whole set whose denial changed an ordinary command's output.
#: Exact names, never a regex or prefix: a ``com.apple.system.*`` pattern would
#: quietly readmit notifyd and logd today and whatever Apple registers there
#: tomorrow. Adding a name here widens the jail into the operator's session;
#: it needs its own measurement, and must never be a pasteboard, security/
#: keychain, window server/display, launchservices/lsd, AppleEvents or TCC
#: service.
_MACH_LOOKUP_GLOBAL_NAMES: tuple[str, ...] = (
    "com.apple.system.opendirectoryd.libinfo",
    "com.apple.system.opendirectoryd.membership",
)


def _interpreter_subtree() -> Path | None:
    """The running interpreter's own tree, or ``None`` when unresolvable.

    The frozen ``_SYSTEM_READ_SUBPATHS`` cannot know where THIS process's
    interpreter lives: a venv keeps it under the worktree, a uv-managed
    Python under the user's data dir. A shell child that re-enters Python
    (``python3 -c``, the eval worker, a script's shebang) must be able to
    read the interpreter it runs, and that is machinery -- not operator
    data -- wherever it happens to sit on disk. Best-effort by design: an
    interpreter path that cannot be resolved simply contributes nothing, and
    the allowlist stays correct for the binaries the system paths already
    cover.
    """

    try:
        executable = Path(sys.executable).resolve()
    except (OSError, ValueError):  # pragma: no cover - defensive
        return None
    # ``<venv>/bin/python`` -> ``<venv>``; a bare system interpreter -> its
    # prefix, which the system paths already cover.
    return executable.parent.parent


def sandbox_unavailable_reason() -> str | None:
    """Why this host cannot sandbox a child, or ``None`` when it can."""

    if sys.platform != "darwin":
        return f"no kernel sandbox is implemented for {sys.platform} yet"
    if not SANDBOX_EXEC.exists():
        return f"{SANDBOX_EXEC} is not present on this host"
    return None


def kernel_exposed_read_subtrees() -> tuple[Path, ...]:
    """Trees a confined child's kernel profile reads BESIDES its root.

    The seatbelt profile allows reads under the confinement root plus
    ``_SYSTEM_READ_SUBPATHS`` and this interpreter's tree (see ``profile``).
    A validation pass that accepts a layout MUST consult this list: a
    controller store INSIDE one of these trees stays readable to the child
    even though it sits outside the root -- and the profile cannot carve it
    back out. Measured on the build host: a profile that allows a broad
    subtree and then a specific DENY of a child path still reached the
    denied file, in BOTH rule orders; only an allow-free profile refused.
    Layouts that place controller state in these trees are refused at
    validation before anything is constructed instead (see
    ``sdk._resolve_world_root``).

    CORRECTED 2026-10-26 (the scout-shell work re-measured it on macOS
    27.0.1): an earlier version of this note said a broad allow followed by a
    specific deny "still reached the denied file, in BOTH rule orders". That
    is not what this host does. With a FILTERED allow (``(allow file-read*
    (subpath ...))``, the shape :meth:`ToolConfinement.profile` emits) the
    LATER rule wins: allow-then-deny refuses the file, deny-then-allow reads
    it. With an UNFILTERED ``(allow file-read*)`` a filtered deny wins in
    either order. Both are pinned by ``tests/unit/tools/
    test_shell_write_confinement.py``. Refusing such layouts at validation
    stays the right call regardless: it does not depend on seatbelt's
    precedence rules, which Apple documents nowhere and could change.
    """

    subtrees = [Path(path) for path in _SYSTEM_READ_SUBPATHS]
    interpreter = _interpreter_subtree()
    if interpreter is not None:
        subtrees.append(interpreter)
    return tuple(subtrees)


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

        if sandbox_unavailable_reason() is not None:
            return None
        return [str(SANDBOX_EXEC), "-p", self.profile(), *argv]

    def spawn_refusal(self) -> str:
        """The failure-closed sentence for a host that cannot enforce the root."""

        reason = sandbox_unavailable_reason() or "an unknown reason"
        return (
            f"refused: this session is confined to {self.root} and this host "
            f"cannot enforce that boundary for shell children ({reason}); a "
            "confined session refuses to run commands it cannot confine"
        )

    def network_refusal(self, tool_name: str) -> str:
        """The refusal for a tool that reaches the host's network in-process.

        ONE spelling for the whole family (``web_search``, ``web_fetch``, the
        ``read <url>`` sugar, ``web_read``, ``browser``): the same reach must
        not answer with two sentences depending on which spelling the model
        chose, and the refusal a trajectory shows is also the deferred finding
        the session-arm write-up carries, so its text is pinned by tests.

        Why refuse instead of allow: the shell's boundary is a kernel
        sandbox, and an in-process HTTP client is not a child process the
        kernel can wrap -- confining it would need a per-tool check the tool
        itself can bypass (a URL built after the check, a redirect, a second
        backend), which is the leak-shaped restriction this module exists to
        avoid. A clear refusal is the honest answer (see the module docstring
        for the comparability reason and the fidelity trade).
        """

        return (
            f"refused: this session is confined to {self.root}, and `{tool_name}` "
            "reaches the host's network -- a reach this boundary does not cover"
        )

    def temp_dir(self) -> Path:
        """The session's temp directory INSIDE the root, created on demand."""

        temp = self.root / "tmp"
        temp.mkdir(parents=True, exist_ok=True)
        return temp

    def _machinery_read_subpaths(self) -> tuple[str, ...]:
        """The running interpreter's own tree, read-only (module helper).

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

        subtree = _interpreter_subtree()
        return () if subtree is None else (str(subtree),)

    def profile(self) -> str:
        """The seatbelt profile for this root, as sandbox-exec's ``-p`` text.

        ``deny default`` plus an explicit allowlist, with FIVE structural
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

        The fourth: the network line. ``(deny network*)`` REPLACED the
        ``(allow network-outbound)`` this profile used to carry. The deny line
        is documentation for the reader; the correctness is the ABSENT allow,
        because seatbelt lets a network allow defeat a network deny wherever
        either line sits -- measured on the build host in BOTH orders
        (``allow;deny`` and ``deny;allow`` each reached a live loopback
        listener; only the allow-free profile refused, and the same is true
        of the internet leg). What the allow-free profile does was measured
        too: TCP (internet and loopback), UDP and AF_UNIX connects all fail
        with ``EPERM``, and DNS fails fast, while in-scratch file work is
        untouched (see the module docstring's network section).

        The fifth: the Mach line. ``mach-lookup`` is allowed for the exact
        ``_MACH_LOOKUP_GLOBAL_NAMES`` only, REPLACING the bare
        ``(allow mach-lookup)`` that let a confined child read and overwrite
        the operator's clipboard and reach securityd. The allowlist is the
        whole grant -- like the network line, no broader mach allow may ride
        beside it, or the narrow one stops meaning anything.
        """

        read_subpaths = (
            str(self.root),
            *self._machinery_read_subpaths(),
            *_SYSTEM_READ_SUBPATHS,
        )
        reads = " ".join(f'(subpath "{path}")' for path in read_subpaths)
        writes = " ".join(f'(literal "{path}")' for path in _SYSTEM_WRITE_LITERALS)
        mach = " ".join(f'(global-name "{name}")' for name in _MACH_LOOKUP_GLOBAL_NAMES)
        return (
            "(version 1)\n"
            "(deny default)\n"
            "(allow process-fork)\n"
            "(allow process-exec)\n"
            "(allow sysctl-read)\n"
            f"(allow mach-lookup {mach})\n"
            "(deny network*)\n"
            '(allow file-read-metadata (subpath "/"))\n'
            f'(allow file-read* (literal "/") {reads})\n'
            f'(allow file-write* (subpath "{self.root}") {writes})\n'
        )


def confinement_of(context: ToolContext | None) -> ToolConfinement | None:
    """The one call site-facing spelling of :meth:`ToolConfinement.from_context`."""

    return ToolConfinement.from_context(context)


# ---------------------------------------------------------------------------
# Write-only confinement: the read-only research shell (``scout``).
# ---------------------------------------------------------------------------
#
# WHY A SECOND SHAPE. ``ToolConfinement`` jails an unattended episode: reads
# AND writes stay inside one root and the in-process network tools refuse. A
# ``scout`` is the opposite trade. Its job is reading -- the whole workspace
# and the web -- and two real jobs failed because it could not run
# ``strings`` on a binary or unpack an ``.asar`` into scratch, so it fell back
# to an (edited) copy of the source on GitHub. What it must not do is CHANGE
# anything. So this shape keeps reads open, confines WRITES to the scout's
# own scratchpad, and gives the shell no network at all (``web_search``/
# ``web_fetch`` stay the scout's network, unchanged and outside this module).
#
# The enforcement is the same kernel boundary as above, for the same reason:
# a mutating command cannot be recognised from its text (``python3 -c``,
# ``tar -x``, ``find -delete`` and ``git`` all write), so the write check runs
# at the syscall. Measured on macOS 27.0.1 with this profile: writes to the
# root, ``strings``/``otool``/``file``, ``python3`` with ``TMPDIR`` in the
# root and ``tar -x`` into the root all work; writing, ``rm``, ``mv``,
# ``chmod``, ``touch``, a symlink or hard link escape, ``git add``/``commit``,
# loopback and AF_UNIX connects, ``kill`` of a process outside the sandbox and
# a nested permissive ``sandbox-exec`` are all refused (see the darwin tests).
#
# THE RESIDUAL, stated (corrected in agent review round 1 of PR #2987): the
# shell reads whatever the user can read, minus the credential stores denied
# below. That is NOT "no wider than ``read``": ``read`` asks before opening a
# path outside the workspace, while this shell is ``read`` tier and never
# asks, so a file the deny list misses is readable here with no prompt -- and
# the scout's ``web_fetch`` could carry it out. The deny list is therefore
# DERIVED from the sensitive-path vocabulary lop already maintains
# (``supplements.denylist``, itself the union of ``references`` and
# ``browser_files``) plus lop's own config-dir stores, so a class added there is
# denied here without an edit. It is still a list, not a proof: a credential
# in a file no rule names (a project's ``config.yaml`` with a token in it) is
# readable, and command output then passes only the bash tool's credential
# masking, which recognises known shapes, not every secret.

#: Credential stores in the operator's HOME that a write-confined shell may
#: not READ, relative to the resolved home directory, that the DERIVED rules
#: (:func:`_derived_denial_filters`) do not already reach by name. SSH keys,
#: the login keychain files (measured: readable as raw files without a deny),
#: ``.aws``, ``.config/gh`` and ``.netrc`` are derived too and stay listed
#: here as belt and braces for the stores this module was written for.
#: ``.config/glab-cli`` is the GitLab twin of ``.config/gh``;
#: ``.codex/auth.json`` (an OpenAI token) and ``.config/op`` (1Password CLI)
#: were measured readable in agent review / QA round 1 of PR #2987.
_HOME_SECRET_SUBPATHS: tuple[str, ...] = (
    ".ssh",
    "Library/Keychains",
    ".aws",
    ".config/gh",
    ".config/glab-cli",
    ".netrc",
    ".codex/auth.json",
    ".config/op",
)

#: lop's OWN credential stores, relative to a config root (the active one and
#: the default ``~/.local-operator``). ``secrets/`` holds the encrypted store
#: and its master key; ``network/identity`` the mesh device PRIVATE key
#: (``device.json``); ``network/credentials`` mesh-borrowed credentials;
#: ``operator/`` the operator signing key; ``tunnel/cloudflared.token`` the
#: tunnel token; ``push-handle.key`` the push handle; ``mcp.json`` MCP server
#: definitions, which carry API keys in ``env``/headers. Each was measured
#: readable from this shell in round 1 of PR #2987. These sit under one root
#: with harmless siblings (``config.yml``), so they are listed rather than
#: derived: the generic vocabulary deliberately does not deny ``*.json``.
_CONFIG_SECRET_SUBPATHS: tuple[str, ...] = (
    "secrets",
    "network/identity",
    "network/credentials",
    "operator",
    "tunnel/cloudflared.token",
    "push-handle.key",
    "mcp.json",
)

#: Per-network secret files: ``network/networks/<id>.secrets.json``. A regex
#: because ``<id>`` varies and the sibling ``<id>.json`` is not a secret.
_NETWORK_SECRETS_SUFFIX = ".secrets.json"

#: Patterns from ``supplements.denylist._EXTRA_PATTERNS`` that are NOT carried
#: into the kernel deny, each because it matches the source and data a research
#: shell exists to read: ``*token*`` is ``tokenizer.py``, ``*secret*`` is
#: ``secret_store.py``, and ``*.db``/``*.sqlite*`` is every project database
#: (and the sqlite files a scout makes in its own scratchpad). The supplement
#: pre-filter can afford them because it only decides what to OFFER a model;
#: here they would refuse a ``grep -r`` over a source tree. Every name here
#: must still exist upstream -- a parity test fails if one is renamed away, so
#: this list cannot silently start excluding nothing.
_KERNEL_EXCLUDED_PATTERNS: frozenset[str] = frozenset(
    {"*token*", "*secret*", "*.db", "*.sqlite", "*.sqlite3"}
)

#: Components likewise NOT carried into the kernel deny. ``secrets`` is lop's
#: own SOURCE package (``local_operator/secrets/``, ``tests/unit/secrets``)
#: and a common module name elsewhere; a whole-tree deny would refuse a scout
#: auditing exactly that code. lop's real ``secrets/`` store is denied by
#: path (:data:`_CONFIG_SECRET_SUBPATHS`), and ``Library/Keychains`` by path.
#: The residual, stated: a project-local ``secrets/`` directory of real
#: secrets is readable from this shell.
_KERNEL_EXCLUDED_COMPONENTS: frozenset[str] = frozenset({"secrets"})

#: Dotenv files with a trailing qualifier (``prod.env.local``), which the
#: derived name/suffix/prefix rules do not reach; ``envy.txt`` stays readable.
_DOTENV_REGEX = r"/[^/]*\.env(\.[^/]*)?$"

#: Characters a path may not contain to be written into a seatbelt profile.
#: Measured on macOS 27.0.1: inside ``"..."`` a ``\\"`` escape works, but inside a
#: ``#"..."`` regex literal a ``"`` cannot be expressed at all (``\\"`` ends the
#: literal: "unbound variable"), and the same root is interpolated into both.
#: So rather than escape one and not the other, a root (or HOME/config dir)
#: containing ``"``, ``\\`` or a control character is REFUSED, deliberately:
#: a lop-generated scratchpad never contains one, and a profile built from one
#: would at best fail to parse (and show the model a raw parser error) and at
#: worst parse as rules nobody wrote -- the ``"x) (allow default) (y`` root.
_PROFILE_UNSAFE_CHARS = frozenset('"\\') | frozenset(chr(code) for code in range(0x20)) | {"\x7f"}

#: The no-op profile the availability probe applies. ``sandbox-exec`` can be
#: PRESENT and still unable to apply a profile -- most commonly when lop is
#: itself running inside a sandbox, where ``sandbox_apply`` answers EPERM --
#: and a host in that state would refuse every command at execute time. The
#: probe finds that once, before a scout is ever handed a shell it cannot use.
_PROBE_PROFILE = "(version 1)(allow default)"


def _seatbelt_regex_literal(path: str) -> str:
    """``path`` with every regex metacharacter escaped, for a ``#"..."`` rule.

    ``"`` and ``\\`` are not handled here: they cannot reach this function, because
    :func:`profile_path_refusal` refuses any path containing them first.
    """

    return "".join(f"\\{ch}" if ch in r".^$*+?()[]{}|" else ch for ch in path)


def profile_path_refusal(path: str | Path) -> str | None:
    """Why ``path`` cannot be written into a seatbelt profile, or ``None``.

    See :data:`_PROFILE_UNSAFE_CHARS`: refusal, never escaping, so a path that
    could change the profile's meaning never reaches ``sandbox-exec``.
    """

    text = str(path)
    bad = sorted({repr(ch) for ch in text if ch in _PROFILE_UNSAFE_CHARS})
    if bad:
        return f"the path {text!r} contains {', '.join(bad)}, which a sandbox profile cannot carry"
    return None


def _glob_to_seatbelt_regex(pattern: str) -> str:
    """A basename ``fnmatch`` glob as a seatbelt regex anchored to a path's leaf.

    Only ``*`` and ``?`` occur in the vocabulary (a parity test pins that), so
    this is the whole translation: ``*`` never crosses a ``/``.
    """

    body = "".join(
        "[^/]*" if ch == "*" else "[^/]" if ch == "?" else _seatbelt_regex_literal(ch)
        for ch in pattern
    )
    return f"/{body}$"


def _derived_denial_filters() -> tuple[str, ...]:
    """Kernel deny FILTERS derived from the sensitive-path vocabulary.

    The source is ``supplements.denylist``: the ONE place that already unions
    ``references`` (the ``@``-reference gate) and ``browser_files`` (the upload
    gate) and adds its own rows, with parity tests keeping it a superset of
    both. Deriving from it rather than copying is what review round 1 of PR
    #2987 asked for: a class added there is denied here without an edit.

    Each rule kind maps to the regex that means the same thing on a path:

    * names / credential globs / extra names / extra patterns -> the LEAF;
    * suffixes and name prefixes -> the leaf's end / start;
    * components and component sequences -> that directory and all beneath;
    * dir+file pairs -> exactly that file under that directory.

    LEAF rules exclude DIRECTORIES (``require-not (vnode-type DIRECTORY)``).
    The vocabulary names credential FILES, and a leaf regex alone also matched
    a directory of the same name -- measured: ``(regex #"/credentials$")``
    refused ``ls local_operator/network/credentials``, which is lop SOURCE,
    and so silently blinded ``grep -r`` over the tree. With the vnode filter
    the directory lists and its ``.py`` files read, while a FILE called
    ``credentials`` (and a symlink to one) is still refused (measured).

    Case: seatbelt's path match is case-insensitive on this (case-insensitive)
    volume -- measured: ``\\.pem$`` refuses ``UPPER.PEM`` and ``/\\.ssh(/|$)``
    refuses ``X/.SSH/id`` -- and the vocabulary is lower-case, so no folding.
    """

    from local_operator.supplements import (
        denylist,  # local: lazy like the module itself
    )

    names, suffixes, prefixes, parts, patterns = denylist._composed()
    leaf_globs = {
        *names,
        *denylist._EXTRA_NAMES,
        *patterns,
        *(pattern for pattern, _rule in denylist._EXTRA_PATTERNS),
    } - _KERNEL_EXCLUDED_PATTERNS
    leaves = [_glob_to_seatbelt_regex(glob) for glob in sorted(leaf_globs)]
    leaves += [f"/[^/]*{_seatbelt_regex_literal(suffix)}$" for suffix in sorted(suffixes)]
    leaves += [f"/{_seatbelt_regex_literal(prefix)}[^/]*$" for prefix in sorted(prefixes)]
    # ``prod.env.local``: a dotenv with a trailing qualifier, which no name,
    # suffix or prefix rule above reaches (kept from the original list).
    leaves.append(_DOTENV_REGEX)
    # De-duplicated in order: the composed sources overlap (``*.pem`` is both a
    # suffix and a credential glob), and one rule per meaning keeps the
    # profile -- compiled on every command -- no larger than it must be.
    filters = [
        f'(require-all (regex #"{regex}") (require-not (vnode-type DIRECTORY)))'
        for regex in dict.fromkeys(leaves)
    ]
    components = {*parts, *denylist._EXTRA_COMPONENTS} - _KERNEL_EXCLUDED_COMPONENTS
    trees = [f"/{_seatbelt_regex_literal(component)}(/|$)" for component in sorted(components)]
    for sequence, _rule in denylist._COMPONENT_SEQUENCES:
        trees.append("/" + "/".join(_seatbelt_regex_literal(part) for part in sequence) + "(/|$)")
    for directory, filename, _rule in denylist._DIR_FILE_PAIRS:
        trees.append(f"/{_seatbelt_regex_literal(directory)}/{_seatbelt_regex_literal(filename)}$")
    filters += [f'(regex #"{regex}")' for regex in dict.fromkeys(trees)]
    return tuple(filters)


@functools.lru_cache(maxsize=1)
def _sandbox_apply_probe_reason() -> str | None:
    """Why ``sandbox-exec`` cannot apply a profile here, or ``None`` if it can.

    Cached for the life of the process: the answer is a property of the host
    and of how lop was launched, neither of which changes mid-run, and the
    probe is a ~16 ms spawn that would otherwise run per command.
    """

    try:
        completed = subprocess.run(
            [str(SANDBOX_EXEC), "-p", _PROBE_PROFILE, "/usr/bin/true"],
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"{SANDBOX_EXEC} could not be run ({exc})"
    if completed.returncode != 0:
        detail = completed.stderr.decode(errors="replace").strip() or "no detail"
        return f"{SANDBOX_EXEC} cannot apply a profile on this host ({detail})"
    return None


def shell_sandbox_unavailable_reason() -> str | None:
    """Why a write-confined shell cannot run here, or ``None`` when it can.

    The static check first (platform, binary present), then the cached apply
    probe. Every caller that decides whether a scout HAS a shell, and every
    spawn that decides whether to RUN one, reads this one answer, so the two
    cannot disagree. SYNCHRONOUS and possibly a spawn on first use: from a
    coroutine call :func:`shell_sandbox_unavailable_reason_async` instead.
    """

    return sandbox_unavailable_reason() or _sandbox_apply_probe_reason()


def _probe_is_cached() -> bool:
    """Whether the apply probe has already answered in this process.

    ``getattr`` because tests replace the probe with a plain callable, which
    has no cache and answers instantly -- treated as cached.
    """

    info = getattr(_sandbox_apply_probe_reason, "cache_info", None)
    return info is None or info().currsize > 0


async def shell_sandbox_unavailable_reason_async() -> str | None:
    """The same answer as :func:`shell_sandbox_unavailable_reason`, off the loop.

    The first probe is a ``subprocess.run`` with a 10 s timeout; on a wedged
    host that would stall every session on the event loop for up to 10 s
    (review round 1 R4 on PR #2987). So the FIRST probe runs on a worker
    thread and populates the same ``lru_cache`` every later read hits; the
    answer is then read through the synchronous function, so there is one
    definition of it. Still fail closed: a probe that raises or exits non-zero
    is a reason.
    """

    if sandbox_unavailable_reason() is None and not _probe_is_cached():
        await asyncio.to_thread(_sandbox_apply_probe_reason)
    return shell_sandbox_unavailable_reason()


def shell_sandbox_known_unavailable_reason() -> str | None:
    """The unavailability reason known WITHOUT spawning anything, or ``None``.

    For synchronous callers on the event loop (a profile attached to a live
    session): the full answer once the probe has run, else the static check
    alone. ``None`` here does NOT mean the sandbox works -- only that nothing
    yet says it does not -- which is safe because the shell's executor awaits
    the full answer before every spawn and refuses rather than run unwrapped.
    """

    if _probe_is_cached():
        return shell_sandbox_unavailable_reason()
    return sandbox_unavailable_reason()


def _secret_read_denials() -> tuple[str, ...]:
    """The ``(deny file-read* ...)`` filters for credential stores.

    Paths are RESOLVED because seatbelt matches the vnode's real path: a
    ``/tmp``-style symlinked home or config dir spelled unresolved would match
    nothing. Both the active config dir and the default ``~/.local-operator``
    are covered, so an isolated ``LOCAL_OPERATOR_CONFIG_DIR`` does not leave
    the operator's real store readable. The caller has already refused a
    HOME or config root that :func:`profile_path_refusal` rejects
    (:func:`_profile_roots_refusal`), so every interpolation here is safe.
    """

    home, config_roots = _profile_roots()
    filters: list[str] = []
    for root in config_roots:
        filters.extend(f'(subpath "{root / rel}")' for rel in _CONFIG_SECRET_SUBPATHS)
        # The ``credentials`` dir, ``credentials.env`` (+ backups) and
        # ``auth.db`` (+ WAL/SHM) are matched by prefix so a sibling
        # spelling the store writes later is covered too.
        prefix = _seatbelt_regex_literal(str(root))
        filters.append(f'(regex #"^{prefix}/(credentials|auth\\.db)")')
        suffix = _seatbelt_regex_literal(_NETWORK_SECRETS_SUFFIX)
        filters.append(f'(regex #"^{prefix}/network/networks/[^/]*{suffix}$")')
    filters.extend(f'(subpath "{home / rel}")' for rel in _HOME_SECRET_SUBPATHS)
    filters.extend(_derived_denial_filters())
    return tuple(filters)


def _profile_roots() -> tuple[Path, tuple[Path, ...]]:
    """``(home, config roots)`` as the profile names them: resolved, sorted."""

    from local_operator.paths import config_dir  # local: paths imports widely

    home = Path.home().expanduser().resolve()
    roots = {config_dir().expanduser().resolve(), home / ".local-operator"}
    return home, tuple(sorted(roots))


def _profile_roots_refusal(root: Path) -> str | None:
    """Why a profile for ``root`` cannot be built safely, or ``None``.

    Every path the profile interpolates: the write root, HOME and the config
    roots. A HOME that cannot be written into the profile means the credential
    denials cannot be either, so that refuses too -- fail closed.
    """

    home, config_roots = _profile_roots()
    for path in (root, home, *config_roots):
        refusal = profile_path_refusal(path)
        if refusal is not None:
            return refusal
    return None


@dataclass(frozen=True)
class ShellWriteConfinement:
    """A shell that reads the host, writes only under ``root``, and has no network.

    The ``scout`` role's shell (see ``harness.subagent.READ_ONLY_SHELL_ROLES``).
    Mirrors :class:`ToolConfinement`'s call-site surface (``from_context``,
    ``wrap``, ``spawn_refusal``, ``temp_dir``) so ``execute_bash`` treats the
    two the same way: wrap, or refuse -- never run unwrapped.
    """

    root: Path

    @classmethod
    def at(cls, root: str | Path) -> ShellWriteConfinement:
        """A confinement for ``root``, resolved once (see :meth:`ToolConfinement.at`)."""

        return cls(root=Path(root).expanduser().resolve())

    @classmethod
    def from_context(cls, context: ToolContext | None) -> ShellWriteConfinement | None:
        """The write confinement a tool call carries, or ``None``.

        ``getattr`` for the duck-typed contexts :meth:`ToolConfinement.
        from_context` already tolerates: no field means not confined.
        """

        raw = getattr(context, "shell_write_root", None)
        if not raw:
            return None
        return cls.at(raw)

    def wrap(self, argv: Sequence[str]) -> list[str] | None:
        """``argv`` under the write-confining sandbox, or ``None`` to refuse.

        From a coroutine, await :func:`shell_sandbox_unavailable_reason_async`
        first so the probe this reads is already cached (``execute_bash`` does).
        """

        if self.unavailable_reason() is not None:
            return None
        return [str(SANDBOX_EXEC), "-p", self.profile(), *argv]

    def unavailable_reason(self) -> str | None:
        """Why this confinement cannot be enforced here, or ``None``.

        The host's answer, or a root/HOME the profile cannot carry safely
        (:func:`profile_path_refusal`). Both refuse the command.
        """

        return _profile_roots_refusal(self.root) or shell_sandbox_unavailable_reason()

    def spawn_refusal(self) -> str:
        """The failure-closed sentence for a host that cannot enforce the boundary."""

        reason = self.unavailable_reason() or "an unknown reason"
        return (
            f"refused: this shell may write only under {self.root} and this host "
            f"cannot enforce that for shell children ({reason}); a read-only "
            "session refuses to run commands it cannot confine"
        )

    def temp_dir(self) -> Path:
        """The shell's temp directory INSIDE the root, created on demand."""

        temp = self.root / "tmp"
        temp.mkdir(parents=True, exist_ok=True)
        return temp

    def profile(self) -> str:
        """The seatbelt profile for this root, as sandbox-exec's ``-p`` text.

        RULE ORDER IS LOAD-BEARING, and the credential denials come LAST. On
        this host a later conflicting rule wins over an earlier one when both
        carry a filter, and a filtered deny beats the unfiltered
        ``(allow file-read*)`` in either order (measured; see
        :func:`kernel_exposed_read_subtrees`). Putting the denies after every
        allow is correct under both behaviours, so a future change to how the
        unfiltered allow ranks cannot silently reopen them.

        Network: there is NO network allow anywhere, which is the actual
        guarantee -- the deny line is documentation (see
        :meth:`ToolConfinement.profile`'s fourth point for why a network allow
        defeats the deny wherever it sits). Mach: the same two-name allowlist
        as :class:`ToolConfinement`. Signals: only within the same sandbox,
        so the shell cannot ``kill`` the operator's processes.
        """

        refusal = _profile_roots_refusal(self.root)
        if refusal is not None:
            # Never build the string: ``wrap`` refuses first, and a direct
            # caller must not get a profile whose meaning a path could change.
            raise ValueError(refusal)
        writes = " ".join(f'(literal "{path}")' for path in _SYSTEM_WRITE_LITERALS)
        mach = " ".join(f'(global-name "{name}")' for name in _MACH_LOOKUP_GLOBAL_NAMES)
        denials = " ".join(_secret_read_denials())
        return (
            "(version 1)\n"
            "(deny default)\n"
            "(allow process-fork)\n"
            "(allow process-exec)\n"
            "(allow signal (target same-sandbox))\n"
            "(allow sysctl-read)\n"
            f"(allow mach-lookup {mach})\n"
            "(deny network*)\n"
            "(allow file-read*)\n"
            f'(allow file-write* (subpath "{self.root}") {writes})\n'
            '(allow file-ioctl (literal "/dev/null") (literal "/dev/tty"))\n'
            f"(deny file-read* {denials})\n"
        )


def shell_write_confinement_of(context: ToolContext | None) -> ShellWriteConfinement | None:
    """The call-site spelling of :meth:`ShellWriteConfinement.from_context`."""

    return ShellWriteConfinement.from_context(context)
