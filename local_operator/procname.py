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

A THIRD CONSUMER MAKES THIS MANDATORY RATHER THAN MERELY SAFE. The desktop app
(``~/local-operator-ui``, ``src/main/backend/owned-serve-launch.ts``) locates
the ``lop`` console script by requiring its SHEBANG to end in ``python``
(``consoleInterpreter()``) and refuses a launcher it cannot resolve that way.
A shebang pointing at the branded link would therefore break that app's managed
backend on every machine, including ones where the link is perfectly healthy.

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
startup (a handful of stats plus one symlink resolution — measured ~180 µs
here, against a startup already north of a second) and re-plants when any
fails. The
inode is ground truth; there is deliberately **no version-stamp file**, because
a stamp is a second source of truth that can itself go stale.

EVERY FUNCTION HERE IS NO-RAISE BY CONTRACT, in the style of ``proc.py``. This
is decoration on a process listing: no failure of it may ever stop a session
from starting. The fallback ladder is hardlink+symlink → the unbranded
interpreter → exactly today's behaviour, and every rung is a silent no-op on
failure.

THE LADDER, AND WHERE EACH RUNG IS IMPLEMENTED
----------------------------------------------
- **Rung 1 — the image AND the argv row.** ``ensure_branded_interpreter()``
  plants the hardlink; :func:`spawn_identity` hands a spawn site the label for
  ``argv[0]`` and the link for ``executable=``.
- **Rung 2 — the interpreter, unlabelled.** With no link,
  :func:`spawn_identity` returns ``(sys.executable, None)``: ``argv[0]`` stays
  the interpreter path and NO label is applied, so the row is ``python3.x``,
  exactly as it was before this module existed.
- **Rung 3 — nothing.** :func:`launchd_job` falls back to the plist shape every
  installer wrote before branding (no ``Program`` key), and
  :func:`brand_this_process` is a no-op off Linux.

**THE ARGV-ONLY RUNG THIS MODULE USED TO PROMISE IS DELIBERATELY NOT
IMPLEMENTED, and the reason is measured.** A label is not free: on Linux
CPython derives ``sys.executable`` from ``argv[0]``, so a labelled ``argv[0]``
leaves the child with ``sys.executable == ""``. CI reproduced it on
ubuntu/py3.12 — the eval worker's own broker spawn, ``secrets/client.py:298``,
dies with ``PermissionError: [Errno 13] Permission denied: ''``, four tests on
one cause — and the same breakage reaches any user cell doing
``subprocess.run([sys.executable, …])``. macOS hides it completely: it resolves
the interpreter from the EXECUTED IMAGE, so the identical child reports a real
``sys.executable`` there, which is why this shipped once and only failed on
Linux. A label that costs the child its interpreter identity is not a naming
improvement, so the label rides with the image or not at all.

Linux therefore names this product on the ``comm`` axis — set in-process by
:func:`brand_this_process`, 15 bytes, brand only — and the argv axis belongs to
rung 1, where the image already carries the interpreter. No half-renamed rows:
where the product cannot be named on either axis, the row says ``python3.x``.
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
#: The secret broker, and the ONE field is a digest rather than the config dir
#: itself. Which store a broker holds is exactly what an operator reading `ps`
#: needs in order to tell a test's broker from their live one, but the config
#: dir is an absolute path that can name a user, a project, or a customer — and
#: argv is world-readable per the rule above. The digest is the SAME one
#: `protocol._runtime_fallback_dir` already derives from the secrets directory,
#: so a row here can be matched against the socket directory on disk without
#: disclosing the path it came from.
#:
#: ORDER MATTERS HERE, because the readers this exists for truncate. `ps` with
#: stdout not a tty falls back to an 80-column screen width on Linux and cuts the
#: row (measured on CI: `... -m local_operator.secrets.brok` at exactly 80). The
#: identity is therefore at the FRONT, so a cut row still says "secret broker"
#: and still names its store; only the module, which a reader can infer, is lost.
#: A reader that needs the whole line uses `ps -ww` or `/proc/<pid>/cmdline`.
#:
#: This label exists because of issue #958: every pre-#954 CI teardown listed a
#: bare, unexplained `Local Operator` child. `_spawn_broker` launches
#: `[sys.executable, "-m", ...]`, and once a parent has been through
#: `reexec_branded`, `sys.executable` IS the branded hardlink — so the broker
#: inherited the product name with nothing to say it was a broker, and on Linux
#: `comm` truncates at 15 bytes, which made every branded child identical in
#: that listing. A daemon that can outlive its starter has to say what it is.
#:
#: WHERE THAT LABEL ACTUALLY LANDS, because it is not every platform: a label is
#: applied only alongside a planted image (`spawn_identity`), and no image can be
#: planted off macOS — so on Linux a broker's argv is the interpreter plus the
#: module, and `comm` is what names it (`brokerd.main` calls
#: `brand_this_process`; 15 bytes, brand only). Nothing reads a broker's argv
#: text programmatically — `lop secret broker stop` asks the socket for the pid —
#: so the store digest simply not appearing in a Linux `ps` is the accepted cost
#: of never handing a child a label without the image that makes it affordable.
LABEL_BROKER = "{brand} [secret broker] store={digest}"
#: The four remaining self-spawns that are not a running service, added when the
#: acceptance bar became "every process this product spawns is named": an EDR
#: quarantines an UNNAMED interpreter doing something sensitive, and each of
#: these does something sensitive from a process the user never sees.
#:
#: ``[install] pip`` installs over the network; ``[mobile restart]`` runs the
#: post-upgrade bounce; ``[daemons] refresh`` rewrites LaunchAgents and restarts
#: the daemons; ``[open] browser`` opens an OAuth URL in a browser. All four are
#: short-lived, which is exactly why they were easy to miss — and why they are
#: also the rows an operator reading `ps` at the wrong moment would find
#: unexplained rather than harmless.
LABEL_INSTALL = "{brand} [install] pip"
LABEL_MOBILE_RESTART = "{brand} [mobile restart]"
LABEL_DAEMONS_REFRESH = "{brand} [daemons] refresh"
LABEL_OPEN_BROWSER = "{brand} [open] browser"


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
    and is not worth pinning to a number here — the total is a couple of
    hundred microseconds either way (~180 µs for the whole steady-state
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
    hardlinked, and the caller falls back to rung 2 — the interpreter, with no
    label on either axis (see the ladder in the module docstring).
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

    **THE ``-c`` LAUNCHER SHAPE IS A DELIBERATE NON-GOAL — do not "fix" it by
    accepting it.** The desktop app (``~/local-operator-ui``,
    ``src/main/backend/owned-serve-launch.ts``) starts its managed backend as
    ``python3 -c "from local_operator.cli import main; main()" serve --port <n>``
    and then verifies that the process it spawned is the one serving by asking
    the SAME ``-c`` string to report its own ``sys.executable``; a mismatch
    raises "its base interpreter … is not the serving process" and the app
    refuses to start its backend. A re-exec through the branded hardlink makes
    ``sys.executable`` become ``<prefix>/bin/Local Operator``, so every one of
    those probes would fail and the desktop app would be dead on every machine.
    The measurement on the operator's machine is a live unbranded row
    (``python3.14 -c …``, parent ``/Applications/Local Operator.app``), and
    that row is the CORRECT outcome of this guard, not a gap in it.
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


def spawn_identity(label: str, **fields: object) -> tuple[str, str | None]:
    """``(argv[0], executable)`` for a child this product spawns.

    The two independent name axes of the module docstring, in one call, so no
    spawn site can implement half of them:

    - ``argv[0]`` is the rendered label — what ``ps -o args``, ``top -o
      command`` and ``pgrep -f`` read — and it is a label ONLY alongside a
      branded image (below);
    - ``executable`` is the branded hardlink when one is planted (what Activity
      Monitor reads, via ``p_comm``), and ``None`` otherwise.

    WHY THE LABEL RIDES WITH THE IMAGE, measured on CI rather than reasoned:
    on Linux CPython derives ``sys.executable`` from ``argv[0]``, so a labelled
    ``argv[0]`` leaves the child with an EMPTY ``sys.executable``. That is not
    cosmetic — ``secrets/client.py`` spawns the broker with
    ``executable=sys.executable`` and dies with ``PermissionError: [Errno 13]
    Permission denied: ''``, and so does any eval cell doing
    ``subprocess.run([sys.executable, …])``. Rung 2 is precisely the rung
    WITHOUT an image, so there is nothing to buy that cost back. macOS cannot
    show this (it resolves the interpreter from the executed image), which is
    why it must be pinned by a Linux-executed child test rather than by a
    parent-side assertion: see ``tests/unit/test_procname_linux.py``.

    ``None`` rather than ``sys.executable`` for the image is deliberate: it
    says "no branded image" to the caller, which keeps ``executable=`` set only
    when it names the link — the invariant ``tests/unit/test_exec_mode.py``
    pins — and ``subprocess`` treats ``None`` exactly as an unset keyword.

    On POSIX the pairing is also load-bearing for rung 1: ``Popen(argv=[…])``
    with ``executable`` unset EXECUTES ``argv[0]``, so a label returned without
    the image would look for a file literally named ``Local Operator [session]
    id=…``. Callers must pass both halves straight through; no call site should
    decorate ``argv[0]`` itself.

    Never raises: a name is decoration, and no spawn may fail for one.
    """
    try:
        link = ensure_branded_interpreter()
    except Exception:  # noqa: BLE001 — a name is decoration, never a failure
        link = None
    if link is None:
        # Rung 2 — see the module ladder: no label, because a labelled argv[0]
        # costs the child its sys.executable on Linux and there is no image
        # here to restore it.
        return sys.executable, None
    return branded_argv0(label, **fields), str(link)


def supervised_image() -> Path | None:
    """The STABLE interpreter a supervised unit should name, or ``None``.

    WHY SUPERVISED UNITS NEED A DIFFERENT ANSWER from the one
    :func:`spawn_identity` and :func:`launchd_job` gave before. A unit is
    executed again on every restart, hours or weeks after it was installed, so
    the image it names has to survive things a running process does not care
    about: a generation flip, and the PRUNE that reclaims the generation it left.
    Naming the branded hardlink — which lives inside one venv, with a libpython
    dylib pin beside it — is what killed 113 processes on 2026-09-15: launchd
    respawned them out of a tree the installer had already emptied, and dyld
    aborted at load. The stable path is a shim that resolves the pointer at exec
    (``update._DAEMON_SHIM``), so the only path a unit names is one that never
    moves.

    ``None`` means "this machine has no generation layout", and every caller
    keeps exactly the shape it shipped before: the branded image, else
    ``sys.executable``. A pip/pipx install and a source checkout answer ``None``
    by design — pointing the operator's plists at a pointer a worktree does not
    own would be worse than the naming it buys.

    Function-local import: ``update`` reaches ``urllib`` and ``subprocess``, and
    this module is on every process's startup path.
    """
    try:
        from local_operator import update

        return update.daemon_image()
    except Exception:  # noqa: BLE001 — a name is decoration, and never a failure
        logger.debug("stable supervised image unavailable", exc_info=True)
        return None


def launchd_job(module: str, *args: str, label: str | None = None) -> dict[str, object]:
    """The ``Program``/``ProgramArguments`` pair for a LaunchAgent.

    THE PLIST HALF OF THE BRANDING, and the reason every launchd daemon used to
    collapse to one indistinguishable ``Local Operator`` row: with only
    ``ProgramArguments`` set, launchd uses element 0 as BOTH the image and
    argv[0], so the two name axes of this module cannot be separated and the
    installer had to choose the image (``launchd_program``).

    Setting ``Program`` to the branded link and ``ProgramArguments[0]`` to the
    role label separates them, and launchd then executes ``Program`` while
    passing the whole array as argv. Measured on the operator's machine
    (macOS, launchd, scratch label, ``Program`` = the planted hardlink,
    ``ProgramArguments[0]`` = a label):

    .. code-block:: text

        launchctl list              -> pid present
        ps -o comm=                 -> Local Operator [        (16-char cut)
        ps -o ucomm=                -> Local Operator
        ps -o args=                 -> Local Operator [label probe] role=test -c …

    TRADE-OFF, recorded because it cannot be measured on a developer machine
    (``sfltool dumpbtm`` needs admin) and was decided rather than discovered:
    macOS Background Task Management names a login item by the basename of
    ``ProgramArguments[0]``. With this shape the notification and the Login
    Items row therefore read the ROLE label — "Local Operator [mobile daemon]
    port=4098" — instead of a bare "Local Operator". That is strictly more
    informative and still unmistakably ours; what it is not is verified against
    BTM's own output. The alternative (keep element 0 an image path) is the
    status quo that cannot tell two daemons apart.

    ``label`` is a RENDERED argv[0] (``branded_argv0(LABEL_MOBILE, port=…)``),
    not a template; ``None`` degrades to the bare brand. Falls back to exactly
    the pre-branding plist — ``launchd_program``'s argv, no ``Program`` key —
    when no branded image can be planted, so rung 3 of the ladder is the shape
    these installers already shipped.

    THE GENERATION LAYOUT RE-POINTS THE IMAGE, to a STABLE path rather than to
    this venv's branded link, and leaves the label where it is. See
    :func:`supervised_image`: a unit is re-executed on every restart, including
    after the tree it was installed from has been pruned, so naming a path
    inside one venv is what launchd respawned processes into on 2026-09-15.
    ``ProgramArguments[0]`` is unchanged, so the name macOS shows for the login
    item is the same in both shapes.
    """
    try:
        link = ensure_branded_interpreter()
    except Exception:  # noqa: BLE001
        link = None
    # The generation layout WINS over the branded link when this machine has it:
    # see :func:`supervised_image` for the crash that makes a per-venv image path
    # unsafe for a unit launchd may restart after the tree is gone. The label
    # still rides at ``ProgramArguments[0]``, which is what macOS's Background
    # Task Management names the login item by — so the row a person reads is
    # unchanged either way.
    stable = supervised_image()
    if stable is not None:
        return {
            "Program": str(stable),
            "ProgramArguments": [branded_argv0(label) if label else BRAND, "-m", module, *args],
        }
    if link is None:
        return {"ProgramArguments": launchd_program(module, *args)}
    return {
        "Program": str(link),
        "ProgramArguments": [branded_argv0(label) if label else BRAND, "-m", module, *args],
    }


def launchd_program(module: str, *args: str, label: str | None = None) -> list[str]:
    """``ProgramArguments`` for a LaunchAgent that runs ``python -m <module>``.

    THE FALLBACK RUNG of :func:`launchd_job`, and the shape every installer in
    this project used before it: element 0 is a real image path, because
    without a ``Program`` key launchd uses it as both the image and argv[0].

    macOS's Background Task Management names a non-bundle login item by the
    **basename of ``ProgramArguments[0]``** — not ``Label``, not the plist
    filename, not ``CFBundleName``. The documented anti-pattern is exactly what
    these plists used to do, ``[sys.executable, "-m", module]``, which is why
    installing a daemon raised "python3 is running in the background" and why
    System Settings > Login Items listed a bare ``python3``. Pointing element 0
    at the branded hardlink makes both read "Local Operator".

    The generation layout moves element 0 to the STABLE shim
    (:func:`supervised_image`) and leaves the rest of the argv alone, for the
    reason recorded there: this shape re-executes element 0 on every restart, so
    a path inside the tree an install (or a prune) replaced is a process that
    dies at load.

    ``label`` is not passed as argv[0] here the way a ``Popen`` label is:
    the two axes collapse into one string and it must remain a real executable
    path. Callers that want the label want :func:`launchd_job`.

    Falls back to ``sys.executable`` when no branded image exists, which is
    byte-for-byte the plist these installers wrote before.
    """
    del label  # see docstring: launchd cannot separate the image from argv[0]
    stable = supervised_image()
    if stable is not None:
        # Element 0 is BOTH the image and argv[0] in this shape, so the stable
        # shim is named here and the argument list is unchanged — see
        # :func:`supervised_image` for why a per-venv path cannot be named.
        return [str(stable), "-m", module, *args]
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

    ``label`` is accepted and ignored on purpose: Linux ``comm`` is capped at
    15 bytes, so only ``BRAND`` (14) fits and a role label would be truncated
    into an unreadable prefix. On Linux this is the ONLY naming axis the
    product has, because the argv label belongs to rung 1 and a branded image
    cannot be planted there (see the module ladder); ``/proc/<pid>/cmdline``
    therefore shows the plain interpreter argv, and ``comm``/``ps -o comm``
    shows the product name.
    """
    del label  # Linux comm is capped at 15 bytes, so only BRAND fits
    set_process_name()
