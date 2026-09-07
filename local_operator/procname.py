"""Make this project's processes identifiable in the OS process listing.

THE PROBLEM
-----------
A machine running Local Operator shows ~20 rows of ``python3.14`` in Activity
Monitor: TUI sessions, subagent runtimes, eval workers, the mobile daemon, the
browser bridge, ``lop serve``. They are indistinguishable from each other and
from unrelated Python, so nobody can tell which one is eating 85% CPU. Worse,
starting a LaunchAgent makes macOS notify "python3 is running in the
background", which reads like malware right after an install.

TWO INDEPENDENT NAME AXES (this is the thing to understand first)
-----------------------------------------------------------------
1. ``p_comm`` — the basename of the **executed binary image**. This is what
   **Activity Monitor** shows, what ``proc_name()`` returns, and what
   ``ps -o ucomm`` prints. The kernel takes it at ``execve`` time from the path
   of the image it loaded. Nothing a running process does can change it on
   macOS.
2. ``argv[0]`` — what ``ps -o args``/``top -o command``/``pgrep -f`` show. A
   process may pass anything here at spawn time.

They are separate, and fixing only one leaves the complaint half-answered:
Activity Monitor ignores argv entirely.

WHY NOT ``setproctitle``
------------------------
Measured on this machine: ``setproctitle`` rewrites **argv only**;
``proc_name()`` still returned ``b'python3.12'`` afterwards, so Activity
Monitor was unchanged. It would add a compiled dependency and still not solve
complaint (1). The zero-dependency approach below covers both axes. Do not
re-add it on the assumption that it does more than it does.

HOW THE BINARY-IMAGE NAME IS CHANGED WITHOUT A DEPENDENCY
---------------------------------------------------------
Execute the interpreter through a path whose basename is the name we want.
Measured constraints, all of them the hard way:

- A **symlink** does NOT work: the kernel resolves it and takes ``p_comm``
  from the target, so the process is still ``python3.12``. A **hardlink**
  does work — it is a second name for the same inode, and the kernel has no
  target to resolve.
- ``cp`` of the interpreter does NOT work: the copy dies with
  ``dyld: Library not loaded: @rpath/libpython3.X.dylib``. The interpreter's
  only ``LC_RPATH`` is ``@executable_path/../lib``, so a copy placed anywhere
  else cannot find its runtime library. (The hardlink has the same problem;
  the symlink below is the fix.)
- **A ``<venv>/lib/libpython3.X.dylib`` symlink beside the hardlink is
  MANDATORY.** The hardlink lives in ``<venv>/bin``, so ``@executable_path/..``
  resolves to the venv, and dyld looks for ``<venv>/lib/libpython3.X.dylib``,
  which a venv does not have. Measured without it: **0 successes, 40/40 aborts**
  (``Abort trap: 6``). With it: 60/60 successes.

  **THE WARM-CACHE TRAP.** A *first* run without the symlink can spuriously
  SUCCEED, because dyld reuses a warm launch closure from the identical inode's
  earlier legitimate launch. Reproduced in this module's own bring-up: run 1
  printed ``ok`` and runs 2-5 aborted. So a single manual smoke test "passes"
  and production then fails 100%. If you change this shape, loop your smoke
  test at least 20 times.

WHY THE ``lop`` SHEBANG IS NOT REWRITTEN
----------------------------------------
Pointing the console script's shebang at the branded link would be free (no
re-exec at all), and it was proposed. It is rejected on a measurement: with the
link removed, ``./lop`` exits **126 — ``bad interpreter: No such file or
directory``**. The command is bricked, and the self-healing staleness check in
this module can never run to repair it, because the process never starts. A
cosmetic feature must not be able to kill the user's CLI. The shebang keeps
pointing at the real interpreter, and ``cli.main`` re-execs through the link
instead: measured +30-36 ms against a ~1100 ms ``lop --version`` baseline (~3%),
and it fails safe — link missing or broken means no re-exec, the process is
alive, and it repairs the link for next time.

NEVER ``chmod``/``chown`` THE PLANTED LINK
------------------------------------------
A hardlink is not a copy: it is a second NAME for the interpreter's inode, and
permissions live on the inode. ``chmod`` on ``<venv>/bin/Local Operator``
therefore changes the mode of the **shared uv interpreter that every venv on
this machine resolves to** — reproduced during bring-up, where a ``chmod 000``
on the link broke an unrelated worktree's ``local-operator`` console script with
``bad interpreter: Permission denied``. Nothing in this module writes modes, and
nothing added to it should. Unlinking is safe (it only drops one name).

STALENESS
---------
The hardlink pins an **inode**, not a version. When ``uv tool install --force``
replaces the interpreter, the planted link keeps pointing at the OLD inode —
uv preserves unknown files in ``bin/`` but does not refresh them — and the
process would silently run a stale interpreter with a fresh site-packages.
:func:`ensure_branded_interpreter` therefore re-checks three facts at every
startup (a handful of stats plus one symlink resolution, tens of microseconds
— measured ~180 µs here against a startup already north of a second) and
re-plants when any fails. The
inode is ground truth; there is deliberately **no version-stamp file**, because
a stamp is a second source of truth that can itself go stale.

EVERY FUNCTION HERE IS NO-RAISE BY CONTRACT, in the style of ``proc.py``. This
is decoration on a process listing: no failure of it may ever stop a session
from starting. The fallback ladder is hardlink+symlink → argv-only labelling →
exactly today's behaviour, and every rung is a silent no-op on failure.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

#: The display name users see in Activity Monitor and ``ps -o ucomm``.
#:
#: Spaces are legal in ``execve`` argv and in a launchd ``ProgramArguments``
#: entry (both take a real string vector, not a shell line), which is what lets
#: this be the product's actual name rather than a squashed identifier. They
#: are FATAL in a shebang (``#!/path/Local Operator`` fails with
#: ``bad interpreter``) — but nothing here writes a shebang, by the deliberate
#: decision recorded in the module docstring, so no space-free twin link is
#: created. A second inode nothing consumes is a maintenance liability and an
#: extra staleness surface; add one only if a shebang consumer ever appears.
BRAND = "Local Operator"

#: ``p_comm`` holds 31 characters. ``ps -o ucomm`` truncates at 16, so a label
#: must already be distinguishable within its first 16 characters. Both limits
#: only bind the BINARY NAME (``BRAND``, 14 chars — fine); the argv labels below
#: are not truncated in ``ps -o args``.
_PROC_NAME_MAX = 31

#: THERE IS NO ENVIRONMENT GUARD, on purpose — this note is here so nobody
#: reintroduces one.
#:
#: A re-exec'd process reports ``sys.executable`` as the branded link it was
#: executed through, so :func:`should_reexec` answers "already branded?" from
#: the process's own identity. An earlier revision used a
#: ``LOCAL_OPERATOR_BRANDED`` marker instead and it was wrong three ways:
#:
#:  - written into ``os.environ``, it was copied by ``reexec.replace_self``
#:    (``env = os.environ.copy()``) into ``/reload`` and ``/update``, so the
#:    relaunched session inherited "already branded" and de-branded itself
#:    PERMANENTLY — reproduced: post-reload image ``python``;
#:  - it leaked into every child spawned with an inherited environment
#:    (``launch.py`` does ``dict(os.environ)``), so genuine launches beneath us
#:    silently refused to brand;
#:  - being presence-tested, ``…=0`` DISABLED branding, the opposite of every
#:    other flag here.
#:
#: An identity check has none of those failure modes: there is no marker to
#: set, inherit, leak, scrub, or misread as configuration.

#: Fixed argv[0] labels. ``argv`` is a PUBLIC channel — every user on the host
#: can read it out of ``ps`` — so these are FIXED format strings with only
#: machine-generated, non-sensitive substitutions: hex ids, integer ports, and
#: agent names that are already slugged upstream. **Never interpolate prompt,
#: file, or model-produced text here.**
LABEL_SESSION = "{brand} [session] agent={agent} id={id}"
#: Same row without an agent: a detached runtime is spawned before any agent is
#: bound, and `agent=-` would be noise in a 16-character `ucomm` window.
LABEL_SESSION_ANON = "{brand} [session] id={id}"
#: And without an id: the interactive TUI is branded before a session id is
#: minted, so the agent name is the only identity available at that point.
LABEL_SESSION_AGENT = "{brand} [session] agent={agent}"
LABEL_EVAL = "{brand} [eval] session={id}"
#: ``job`` and not a slug of the prompt: the prompt is user text, and the rule
#: above forbids it on a world-readable channel. The job id is what ``exec
#: --background`` already prints to the user ("Started background job <id>"), so
#: it is both safe and the better correlation handle.
LABEL_EXEC = "{brand} [exec] job={job}"
LABEL_MOBILE = "{brand} [mobile daemon] port={port}"
LABEL_BROWSER = "{brand} [browser bridge] port={port}"
LABEL_WAKES = "{brand} [wakes]"
LABEL_TUNNEL = "{brand} [tunnel]"
LABEL_SERVE = "{brand} [serve] port={port}"


def safe_field(value: object, limit: int = 24) -> str:
    """Reduce ``value`` to ``[A-Za-z0-9._-]`` for use inside a label.

    The label vocabulary is a set of FIXED templates precisely because argv is
    world-readable, and the one field that can carry human text is the agent
    name from ``--agent``. That text is already in the operator's own argv, so
    this is not about disclosure; it is about keeping the row PARSEABLE and
    bounded — an agent name with spaces or a newline in it would otherwise
    split the ``[session] agent=…`` row into something no operator can read and
    no ``pgrep -f`` pattern can match.
    """
    text = str(value)
    cleaned = "".join(ch if (ch.isalnum() or ch in "._-") else "-" for ch in text)
    return cleaned[:limit] or "-"


def branded_argv0(label: str, **fields: object) -> str:
    """Render one of the ``LABEL_*`` templates into the argv[0] a child is given.

    ``branded_argv0(LABEL_EVAL, id="deadbeef")``. The brand is substituted here
    so the product name lives in exactly one place, and the caller supplies only
    its own machine-generated fields. A template that does not render (a missing
    field after a refactor) degrades to the bare brand rather than raising:
    a spawn must never fail over its own decoration.
    """
    try:
        return label.format(brand=BRAND, **fields)
    except Exception:  # noqa: BLE001 — a label is decoration, never a failure
        return BRAND


def _is_venv() -> bool:
    """True when this interpreter runs inside a venv/virtualenv.

    The plant target is ``<sys.prefix>/bin``, which must be a venv the project
    owns. Planting into a system or Homebrew prefix would write into a shared,
    possibly root-owned installation that other software depends on — out of
    the question for a cosmetic feature.
    """
    return sys.prefix != sys.base_prefix


def _libpython_name() -> str | None:
    """Basename of the dylib the interpreter loads via ``@rpath``, or None.

    Read from ``sysconfig`` rather than guessed, because the name carries the
    ABI flags on some builds (``libpython3.13t.dylib`` on a free-threaded
    build) and a guessed ``libpython{major}.{minor}.dylib`` would be wrong
    there — which is exactly the 100%-abort failure mode.

    ``sysconfig`` is stdlib and already imported by the interpreter's own
    startup on most paths; measured cold cost here is ~0.4 ms, paid once.
    """
    try:
        import sysconfig

        name = sysconfig.get_config_var("LDLIBRARY") or sysconfig.get_config_var("INSTSONAME")
    except Exception:  # noqa: BLE001 — probe only
        return None
    if not isinstance(name, str) or not name.endswith(".dylib"):
        # A static build (``libpython3.12.a``) or a framework build (whose
        # LDLIBRARY is ``Python.framework/...``) has nothing we can symlink
        # into a venv lib dir. Both are handled by refusing to plant.
        return None
    return os.path.basename(name)


def _real_interpreter() -> Path | None:
    """``sys.executable`` with symlinks resolved, or None when unusable.

    The hardlink must be made against the REAL file: ``os.link`` on a symlink
    follows it anyway, but resolving here is what lets the staleness check
    compare inodes against a stable target.
    """
    executable = sys.executable
    if not executable:
        return None
    try:
        return Path(os.path.realpath(executable))
    except OSError:
        return None


def _is_framework_build(real: Path) -> bool:
    """True for a macOS framework interpreter, which CANNOT be branded.

    Measured on Homebrew's ``python@3.14``: hardlinking that binary and running
    it reports ``proc_name = b'Python'``, not the link's name — the shipped
    ``bin/python3.14`` is a stub that re-executes the real binary inside
    ``Resources/Python.app/Contents/MacOS/Python``, so the kernel takes
    ``p_comm`` from *that* second exec and the link name is discarded. Its
    ``LDLIBRARY`` is a framework path with no plain dylib to symlink either.

    Detected by path rather than by ``PYTHONFRAMEWORK`` because a venv built on
    a framework Python reports an empty ``PYTHONFRAMEWORK`` while still
    resolving to the framework binary. uv-managed interpreters — what ``lop``
    actually ships on — are ordinary Mach-O executables and brand correctly.
    """
    return ".framework" in str(real)


def branded_link_path() -> Path | None:
    """Where the branded hardlink lives, or None when this environment is out.

    ``<sys.prefix>/bin/Local Operator`` — beside the venv's own ``python``, so
    dyld's ``@executable_path/../lib`` resolves into the venv where the
    companion symlink is planted. Returns None (no plant, no re-exec, today's
    behaviour) when the platform is not macOS or the interpreter is not a
    brandable venv interpreter.
    """
    if sys.platform != "darwin":
        return None
    if not _is_venv():
        return None
    real = _real_interpreter()
    if real is None or _is_framework_build(real):
        return None
    if len(BRAND) > _PROC_NAME_MAX:
        return None
    try:
        return Path(sys.prefix) / "bin" / BRAND
    except Exception:  # noqa: BLE001 — a path build never fails a startup
        return None


def _needs_replant(link: Path, real: Path, libpython: Path | None) -> bool:
    """Whether the planted shape is missing or stale. Stats plus one resolve.

    Cost, measured rather than asserted: two ``stat`` calls on the link and the
    interpreter, one ``exists`` on the symlink, and one ``resolve()`` pair for
    the dylib-identity check below. ``resolve()`` ``lstat``s every path
    component, so the exact syscall count scales with how deep the venv sits
    and is not worth pinning to a number here — the total is tens of
    microseconds either way (~180 µs for the whole steady-state
    ``ensure_branded_interpreter``, against a >1 s startup).

    Refresh when ANY of:

    (a) the link's inode differs from the real interpreter's — ``uv tool
        install --force`` replaced the interpreter and PRESERVED our planted
        file without refreshing it, so the link now names an old inode: a
        stale interpreter under fresh site-packages;
    (b) ``st_nlink < 2`` — something replaced the hardlink with a copy (an
        archive restore, an rsync without ``-H``), which is the ``@rpath``
        failure again and is no longer the same inode as the interpreter;
    (c) the companion libpython symlink is missing, dangling, or points at
        something other than THIS interpreter's dylib — the measured
        100%-abort mode, plus the subtler case where an interpreter upgrade
        leaves a live symlink aimed at the previous install's library.

    The inode is ground truth, so no version-stamp file is written; a stamp
    would be a second source of truth that can itself go stale while the inode
    it describes has already been replaced.
    """
    try:
        link_stat = link.stat()  # follows nothing meaningful: a hardlink IS the file
    except OSError:
        return True
    try:
        real_stat = real.stat()
    except OSError:
        # The interpreter we are running vanished mid-check. Nothing sane to
        # plant against; leave what is there rather than deleting a link that
        # may still work.
        return False
    if (link_stat.st_dev, link_stat.st_ino) != (real_stat.st_dev, real_stat.st_ino):
        return True  # (a)
    if link_stat.st_nlink < 2:
        return True  # (b)
    if libpython is not None:
        # `exists` follows the symlink, so a dangling one is already False.
        # Inside the try because it is NOT total: an unreadable parent
        # directory makes it raise `PermissionError` (reproduced), and this
        # function's contract is that it never raises. Treated as "replant",
        # which is the safe direction — a libpython we cannot even stat is not
        # one we should assume is correct.
        try:
            live = libpython.exists()
        except OSError:
            return True
        if not live:
            return True  # (c) missing or dangling
        # A LIVE symlink is not automatically a CORRECT one. After an
        # interpreter upgrade the old link still resolves — to the previous
        # install's dylib — and loading a mismatched libpython beside a fresh
        # interpreter is exactly the crash this whole shape exists to avoid.
        # Compared by resolved path against the running interpreter's own lib.
        try:
            if libpython.resolve() != (real.parent.parent / "lib" / libpython.name).resolve():
                return True  # (c) live but pointing at the wrong library
        except OSError:
            return True
    return False


def _sweep_orphan_temps(directory: Path, prefix: str) -> None:
    """Remove ``.<prefix>.<pid>.tmp`` entries left by a plant that was killed.

    The plant is link-to-temp + ``os.replace``, which is atomic and cleans up
    after itself on an ``OSError`` — but NOT when the process is killed between
    the two steps (reproduced with ``kill -9``: the temp entry survives). Each
    orphan is a harmless zero-byte-of-data directory entry, but they accumulate
    once per crashed startup and turn ``ls <venv>/bin`` into a junkyard.

    Only entries whose embedded pid is no longer alive are removed, so a plant
    running concurrently in another worktree is never disturbed.
    """
    try:
        for entry in directory.glob(f".{prefix}.*.tmp"):
            pid_text = entry.name[len(prefix) + 2 : -4]
            if not pid_text.isdigit():
                continue
            try:
                os.kill(int(pid_text), 0)
                continue  # still running: not ours to clean
            except ProcessLookupError:
                pass
            except OSError:
                continue  # EPERM: alive but another user's — leave it
            try:
                entry.unlink()
            except OSError:
                pass
    except Exception:  # noqa: BLE001 — housekeeping never fails a startup
        logger.debug("orphan temp sweep skipped", exc_info=True)


def _plant_hardlink(link: Path, real: Path) -> bool:
    """Create/refresh ``link`` as a hardlink to ``real``. Atomic, never raises.

    Link-to-temp then ``os.replace`` so a concurrent process (many worktrees on
    this machine run this simultaneously) never observes a missing or partially
    created link: ``os.replace`` is atomic within a directory, and a process
    that opened the old inode keeps running it.

    ``EXDEV`` (cross-device) is an EXPECTED outcome, not an error: an
    interpreter on a different filesystem from the venv simply cannot be
    hardlinked, and the caller falls back to argv-only labelling.
    """
    tmp = link.with_name(f".{BRAND}.{os.getpid()}.tmp")
    try:
        link.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        os.link(real, tmp)
        os.replace(tmp, link)
        return True
    except OSError as exc:
        # EXDEV, EPERM (a read-only or foreign-owned prefix), ENOSPC — all mean
        # "no branded image here", which is a supported outcome.
        logger.debug("branded hardlink not planted: %s", exc)
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return False


def _plant_libpython(link: Path, real: Path, name: str) -> bool:
    """Symlink ``<venv>/lib/<name>`` at the interpreter's own dylib.

    Without this the hardlink aborts on launch 100% of the time once dyld's
    warm launch closure expires — see the module docstring. A symlink is
    correct here (and a hardlink would be wrong): only the EXECUTED IMAGE's
    name is taken by the kernel, and dyld happily follows a symlink for a
    library, so this costs one inode-free directory entry.
    """
    source = real.parent.parent / "lib" / name
    target = link.parent.parent / "lib" / name
    try:
        if not source.is_file():
            return False
        target.parent.mkdir(parents=True, exist_ok=True)
        if (target.is_symlink() or target.exists()) and target.resolve() == source.resolve():
            return True
        # NO unlink of a wrong-but-live symlink before the replace. `os.replace`
        # overwrites a symlink atomically, so the unlink bought nothing and
        # opened a window in which the venv has NO libpython at all — and any
        # concurrent start landing in that window hits the measured
        # 100%-abort mode (`Abort trap: 6`). The whole point of the
        # temp-then-replace shape is that the target is never absent; removing
        # it first defeated that.
        #
        # Symlink via a temp name + replace, for the same concurrency reason as
        # the hardlink: sibling worktrees plant this at the same moment.
        tmp = target.with_name(f".{name}.{os.getpid()}.tmp")
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        os.symlink(source, tmp)
        os.replace(tmp, target)
        return True
    except OSError as exc:
        logger.debug("libpython symlink not planted: %s", exc)
        return False


def ensure_branded_interpreter() -> Path | None:
    """Plant/refresh the branded interpreter image; return its path or None.

    Cheap enough for every startup: the common case is the staleness probe
    above (~180 µs measured, against a startup already over a second) and no
    writes. Only a first run in a fresh venv, or a genuine staleness trigger,
    does any filesystem work — and a hardlink plus a symlink cost zero bytes of
    data, which is what makes this safe for the many concurrent worktrees on a
    development machine (each plants into its OWN venv; they never contend for
    one path).

    None means "no branded image" — an unsupported platform, a non-venv
    interpreter, a framework build, a cross-device prefix, a read-only install.
    Every caller must treat None as "carry on exactly as before".
    """
    try:
        link = branded_link_path()
        if link is None:
            return None
        real = _real_interpreter()
        if real is None:
            return None
        name = _libpython_name()
        libpython = (link.parent.parent / "lib" / name) if name else None

        if not _needs_replant(link, real, libpython):
            return link

        if name is None:
            # No dylib to pin means the hardlink would abort at launch. Refuse
            # rather than plant a landmine.
            return None
        # BOTH plant directories. A killed plant can leak a temp in `bin/`
        # (named for the brand) or in `lib/` (named for the dylib), and
        # sweeping only the first left lib temps accumulating forever. Only on
        # the replant path, so the common startup stays a stat probe with no
        # directory scan.
        _sweep_orphan_temps(link.parent, BRAND)
        _sweep_orphan_temps(link.parent.parent / "lib", name)
        if not _plant_libpython(link, real, name):
            return None
        if not _plant_hardlink(link, real):
            return None
        return link
    except Exception:  # noqa: BLE001 — decoration must never break a startup
        logger.debug("branded interpreter setup skipped", exc_info=True)
        return None


def should_reexec() -> Path | None:
    """The branded image to re-exec through, or None to stay as we are.

    None whenever: this process is ALREADY running through the branded image
    (which is what makes a re-exec loop impossible), it is not a real ``lop``
    launch, or no branded image could be planted.
    """
    try:
        # THE LOOP GUARD, and it is deliberately not an environment variable.
        # A re-exec'd process reports `sys.executable` as the LINK it was
        # executed through, so "am I already branded?" is answered by the
        # process's own identity — no marker to set, inherit, leak or scrub.
        # This also covers a launchd job pointed straight at the link.
        if os.path.basename(sys.executable) == BRAND:
            return None
        # Only a real `lop` launch may replace itself; see `is_own_launch`.
        if not is_own_launch():
            return None
        link = ensure_branded_interpreter()
        if link is None:
            return None
        if not os.access(link, os.X_OK):
            return None
        return link
    except Exception:  # noqa: BLE001
        return None


#: Console-script basenames that mean "this process IS a Local Operator launch".
#: `[project.scripts]` defines both; `lop` is the launcher the operator uses.
_LAUNCHER_NAMES = frozenset({"lop", "lo", "local-operator"})


def is_own_launch() -> bool:
    """True only when this process was started as the ``lop`` command itself.

    THE GUARD THAT KEEPS A RE-EXEC FROM EATING ITS CALLER. ``cli.main()`` is not
    only an entry point: the test suite calls it in-process (``assert main()
    == 7``), and so can any embedder. An unguarded re-exec there does not
    "rename the process" — it REPLACES the running pytest with a fresh
    interpreter, which silently truncated a 98-test file at 57% with exit 0.
    Reproduced before this guard existed.

    ``sys.orig_argv[1]`` is the script the interpreter was pointed at, which for
    a real launch is the generated console script (``…/bin/lop``). Under
    pytest, an embedder, or ``python -c`` it is something else entirely, so the
    process correctly declines to replace itself. ``-m local_operator.cli`` is
    also accepted: that is a documented way to launch the app.
    """
    try:
        argv = sys.orig_argv
        if len(argv) < 2:
            return False  # a bare REPL
        first = argv[1]
        if first == "-m":
            return len(argv) > 2 and argv[2] in {"local_operator", "local_operator.cli"}
        if first.startswith("-"):
            return False  # -c, -X, an inline flag: never a launcher invocation
        return os.path.basename(first) in _LAUNCHER_NAMES
    except Exception:  # noqa: BLE001
        return False


def reexec_branded(label: str | None = None) -> None:
    """Replace this process with the same launch under the branded image.

    Never returns on success; returns normally (a no-op) whenever branding is
    unavailable, so the caller carries on with today's behaviour and — crucially
    — is still ALIVE to have repaired the link for next time. This is the whole
    reason the ``lop`` shebang is left alone; see the module docstring.

    cwd and the controlling tty are preserved exactly: ``exec`` keeps cwd and
    every file descriptor (so the tty, and Textual's view of it, is untouched),
    and ``sys.orig_argv`` is the interpreter's own launch line including the
    ``-X``/``-u`` style flags a plain ``sys.argv`` would drop.

    One thing is deliberately NOT preserved, and it is the point: ``argv[0]``
    becomes the label. The ENVIRONMENT is passed through untouched — the loop
    guard is the process's own ``sys.executable``, not a marker, for the
    reasons recorded at :data:`BRAND`.

    WHY THIS DOES NOT BREAK ``resume_executable`` OR ``/reload``. Both read
    ``sys.argv[0]``, and CPython sets that from the SCRIPT path, not from the
    process's argv[0] — verified: after this re-exec, ``orig_argv[0]`` is
    ``'Local Operator [serve] port=1'`` while ``sys.argv[0]`` is still the
    ``lop`` launcher path, so ``broadcast.resume_executable()`` and
    ``reexec.plan_argv()`` both keep returning the launcher. That distinction is
    load-bearing: a label reaching ``os.execvpe`` as a path fails with
    ``FileNotFoundError``, which would break every crash-restore.

    It is deliberately independent of :mod:`local_operator.reexec`: that module
    replaces the process AFTER the TUI tears down, to pick up a new wheel, and
    exits ``REEXEC_CODE`` (75) to get there. This runs before anything starts,
    changes no argument, and leaves no marker behind — so a later ``/reload``
    execs the ``lop`` launcher normally and that new process brands itself
    again from scratch, which is exactly what a reload should do.
    """
    try:
        link = should_reexec()
        if link is None:
            return
        argv = list(sys.orig_argv)
        if not argv:
            return
        # argv[0] is the interpreter path; replacing it with the label is what
        # puts a readable line in ``ps -o args``. The IMAGE is `link`, which is
        # what Activity Monitor reads.
        argv[0] = branded_argv0(label) if label else BRAND
        # Plain `execv`: the environment is inherited unchanged and NOTHING is
        # written into `os.environ`. The replacement process recognises itself
        # as branded from `sys.executable`; see the note at `BRAND` for the
        # three ways the marker this replaced went wrong.
        os.execv(str(link), argv)
    except Exception:  # noqa: BLE001 — a failed exec must leave us running
        logger.debug("branded re-exec skipped", exc_info=True)


def launchd_program(module: str, *args: str, label: str | None = None) -> list[str]:
    """``ProgramArguments`` for a LaunchAgent that runs ``python -m <module>``.

    THE SECOND HALF OF THE OPERATOR'S COMPLAINT. macOS's Background Task
    Management names a non-bundle login item by the **basename of
    ``ProgramArguments[0]``** — not ``Label``, not the plist filename, not
    ``CFBundleName``. The documented anti-pattern is exactly what these plists
    used to do, ``[sys.executable, "-m", module]``, which is why installing a
    daemon raised "python3 is running in the background" and why System
    Settings > Login Items listed a bare ``python3``. Pointing element 0 at the
    branded hardlink makes both read "Local Operator".

    ``label`` is not passed as argv[0] here the way a ``Popen`` label is:
    launchd uses ``ProgramArguments[0]`` as BOTH the image to execute and
    argv[0] (there is no ``Program`` key set), so the two axes collapse into
    one string and it must remain a real executable path.

    Falls back to ``sys.executable`` when no branded image exists, which is
    byte-for-byte the plist these installers wrote before.
    """
    del label  # see docstring: launchd cannot separate the image from argv[0]
    try:
        link = ensure_branded_interpreter()
    except Exception:  # noqa: BLE001
        link = None
    return [str(link) if link is not None else sys.executable, "-m", module, *args]


def set_process_name(name: str = BRAND) -> bool:
    """Linux: set this thread's ``comm`` via ``prctl(PR_SET_NAME)``.

    Zero-dependency (``ctypes`` against libc) and what ``/proc/<pid>/comm``,
    ``ps -o comm``, ``htop`` and most Linux monitors read. Two properties that
    shape every caller:

    - it truncates at **15** bytes plus NUL, so ``BRAND`` ("Local Operator",
      14) fits exactly and a longer label would be silently cut;
    - it is **NOT inherited across ``fork``/``exec``** — an exec resets ``comm``
      to the new image's basename — so every child process must call this
      itself rather than relying on the parent having done it.

    Capability-probed rather than branched on ``sys.platform``: the constant is
    absent on macOS, and probing is what keeps this correct on a platform that
    is neither (a BSD, a musl container) without a growing platform list.
    Returns True only when the name was actually set.
    """
    try:
        if sys.platform != "linux":
            return False
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        prctl = getattr(libc, "prctl", None)
        if prctl is None:
            return False
        # 15 == PR_SET_NAME. Encoded to bytes explicitly so a non-ASCII name
        # cannot raise inside ctypes' implicit conversion.
        buf = ctypes.create_string_buffer(name.encode("utf-8", "replace")[:15])
        return prctl(15, ctypes.byref(buf), 0, 0, 0) == 0
    except Exception:  # noqa: BLE001 — decoration never raises
        return False


def brand_this_process(label: str | None = None) -> None:
    """The one call a child process makes to name itself, on any platform.

    macOS gets its name from the image it was EXEC'd through, so there is
    nothing to do in-process — the parent supplied it via ``executable=``.
    Linux must set ``comm`` itself because ``prctl`` does not survive the exec.
    One helper so a spawn site does not have to know which axis its platform
    uses.
    """
    del label  # reserved: Linux comm is capped at 15 bytes, so only BRAND fits
    set_process_name()
