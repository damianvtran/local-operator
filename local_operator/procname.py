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

- A **symlink at ``<venv>/bin/Local Operator``** works, provided it points at an
  image whose OWN basename is the brand. The kernel resolves the symlink and
  takes ``p_comm`` from the image it finally loads, so the image — not the link
  — has to be named ``Local Operator``. A link straight at ``python3.12`` still
  reports ``python3.12``.
- **A per-venv hardlink also works** — a second name for the interpreter's
  inode, with no target for the kernel to resolve — and it is the FALLBACK rung
  (see the ladder below). It is a second path per venv and per generation, which
  is exactly what macOS's TCC rows key on, and it shares the interpreter's
  inode, which is the blast radius this module now avoids by default.
- **A copy of the interpreter needs its ``lib`` beside it.** The interpreter's
  only ``LC_RPATH`` is ``@executable_path/../lib``, so an image placed in
  ``<dir>/bin`` cannot find ``libpython3.X.dylib`` or the standard library
  unless ``<dir>/lib`` resolves to the interpreter's own ``lib``. Measured
  without it: ``dyld: Library not loaded`` standing in for execution, and — as
  CPython looks for its stdlib the same way — ``No module named 'encodings'``.

  **THE WARM-CACHE TRAP.** A *first* run without the library beside it can
  spuriously SUCCEED, because dyld reuses a warm launch closure from the
  identical inode's earlier legitimate launch. Reproduced in this module's own
  bring-up: run 1 printed ``ok`` and runs 2-5 aborted. So a single manual smoke
  test "passes" and production then fails 100%. If you change this shape, loop
  your smoke test at least 20 times.

ONE STABLE IMAGE PER PYTHON BUILD (the macOS privacy identity)
--------------------------------------------------------------
macOS gate-keeps some resources per EXECUTABLE IMAGE PATH: an ad-hoc-signed
binary is remembered by the path it was executed from plus its cdhash, so a
second path is a second client and a second prompt. Measured on the operator's
machine after a handful of updates: nine rows in App Management and Files &
Folders — one per generation, plus the worktree venvs — because the branded
hardlink lived at ``<venv>/bin/Local Operator`` and every new venv was a new
path.

The plant therefore builds ONE image per Python build and every venv links to
it::

    ~/.local/share/lop/runtime/<build-key>/bin/Local Operator   # APFS clone
    ~/.local/share/lop/runtime/<build-key>/lib      -> <real>/../lib
    ~/.local/share/lop/runtime/<build-key>/include  -> <real>/../include
    <venv>/bin/Local Operator                       -> that image

``<build-key>`` is ``<real.parent.parent.name>-<sha256(realpath, size,
mtime_ns)[:10]>``: a new Python build is a new key, and a replaced interpreter
cannot be mistaken for the one already published. Published directories are
IMMUTABLE UNLESS CORRUPT and are never garbage-collected — a running process
may still be executing one, and a key that is still on disk is one prompt that
will not happen again. The one exception is a published directory whose image
fails the integrity check (:func:`_healthy_image`): nothing can be executing an
image that cannot start, so it is replaced under the same key.

The image is a CLONE (``clonefile``, falling back to ``shutil.copy2``), not a
hardlink: its own inode, ``nlink == 1``, same bytes and mtime. That removes the
shared-inode blast radius below and makes it the first file of ours that could
be re-signed without modifying uv's interpreter.

Measured on this machine, 25 of 25 launches through the venv symlink
(re-measured 2026-10-19, uv 3.12.14): ``proc_pidpath`` reports the runtime image
path (and ``SecCodeCopyPath`` agreed in the original 25-launch run),
``sys.prefix`` is still the venv, and ``ucomm`` is the brand. That triple is the
whole point — macOS sees one stable client, Python still sees its own venv.

The ``lib`` symlink is REQUIRED, not decoration: CPython locates its stdlib
relative to the real image directory, and without it the clone dies with
``No module named 'encodings'`` before any of our code runs.

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
On the runtime rung the venv link is a SYMLINK to a clone we own, so a mode
write on it would land on the clone — this module still never does one, because
that image is shared by every venv on the machine and a mode is the one thing
that would make a healthy image unexecutable everywhere.

The hardlink rung is worse, and it is why the rule is absolute: a hardlink is
not a copy, it is a second NAME for the interpreter's inode, and permissions
live on the inode. ``chmod`` on ``<venv>/bin/Local Operator`` therefore changes
the mode of the **shared uv interpreter that every venv on this machine
resolves to** — reproduced during bring-up, where a ``chmod 000`` on the link
broke an unrelated worktree's ``local-operator`` console script with ``bad
interpreter: Permission denied``. Unlinking is safe (it only drops one name).

A TEMP NAME IS THE LINK'S INODE TOO, so the same rule covers WRITING THROUGH
one: the hardlink plant's ``.<brand>.<pid>.<thread>.tmp`` is a second name for
whatever the ``os.replace`` is about to install, and before that replace it is
a second name for the interpreter the link already resolves to. Measured
2026-09-18, the hard way: a write through a LEAKED temp truncated a uv-managed,
machine-wide interpreter —
``~/.local/share/uv/python/cpython-3.12.13-macos-aarch64-none/bin/python3.12``
— to a 12-byte text file reachable under 23 different names, breaking every 3.12
venv on the host until ``uv`` re-materialised it. "Unlinking is safe" holds for
a temp precisely because it only drops one name. The runtime plant's temp is a
DIRECTORY it created itself, beside the image, so a write into it cannot reach
the interpreter at all — which is one more reason the default rung moved.

STALENESS
---------
On the runtime rung the link is checked by IDENTITY against the build it should
name: a symlink to the runtime image for THIS interpreter, that image
executable, and its stdlib resolvable through the ``lib`` beside it. Anything
else is replanted. The interpreter's ``(realpath, size, mtime_ns)`` are what
the build key is made of, so a ``uv tool install --force`` that replaces the
interpreter names a DIFFERENT key, the link is repointed at that image, and the
published one is left alone. The source is ground truth; there is deliberately
**no version-stamp file**, because a stamp is a second source of truth that can
itself go stale.

The hardlink rung keeps the older, inode-shaped rule because its failures are
inode-shaped: when ``uv tool install --force`` replaces the interpreter, that
link keeps pointing at the OLD inode — uv preserves unknown files in ``bin/``
but does not refresh them — and the process would silently run a stale
interpreter with a fresh site-packages.

AND A FOURTH ROT IS NOT OURS BUT IS MEASURED: the link is not always the
image we planted. On the operator's machine an EDR replaced it in
five worktree venvs with a 49968-byte, mode-``-rw-------`` file owned by its
own agent user (``_sentinel``), so the path named a file we can neither exec
nor read. "Present but not executable by us" is therefore stale as well, and
a spawn site that would have been handed such an image takes the unbranded rung
instead of failing at ``execve``.

EVERY FUNCTION HERE IS NO-RAISE BY CONTRACT, in the style of ``proc.py``. This
is decoration on a process listing: no failure of it may ever stop a session
from starting. The ladder is runtime image → hardlink+pin → the unbranded
interpreter → exactly today's behaviour, and every rung is a silent no-op on
failure.

THE LADDER, AND WHERE EACH RUNG IS IMPLEMENTED
----------------------------------------------
- **Rung 1 — the image AND the argv row.** ``ensure_branded_interpreter()``
  plants the runtime image and links this venv to it;
  :func:`spawn_identity` hands a spawn site the label for ``argv[0]`` and the
  link for ``executable=``.
- **Rung 1b — the per-venv hardlink and its dylib pin.** Taken when the
  runtime image cannot be planted (a read-only or full stable root) or when the
  interpreter's own ``lib`` cannot back an image. A non-APFS volume is NOT one
  of those: the clone falls back to a ``shutil.copy2`` byte copy and the
  runtime rung still holds. It is one release of insurance, kept because a
  cosmetic failure must not cost the name everywhere. It MUST NOT replace a
  healthy runtime symlink: a sandboxed child with a read-only runtime root must
  not undo what the parent planted.
- **Rung 2 — the interpreter, unlabelled.** With no link,
  :func:`spawn_identity` returns ``(sys.executable, None)``: ``argv[0]`` stays
  the interpreter path and NO label is applied, so the row is ``python3.x``,
  exactly as it was before this module existed.
- **Rung 3 — nothing.** :func:`launchd_job` falls back to the plist shape every
  installer wrote before branding (no ``Program`` key), and
  :func:`brand_this_process` is a no-op off Linux.

**AND THE PLANT MUST GO WHERE THE EXEC GOES, which is not always this venv.**
Two spawn sites run a DIFFERENT tree's interpreter: the post-upgrade repair
child (``update._post_upgrade_invocation``, which must run the build just
installed) and a session runtime engaged onto the next generation
(``session/runtime/launch._spawn_runtime``). ``ensure_branded_interpreter`` can
only brand the venv THIS process runs from, so both sites used to keep the
label while replacing the image with a bare interpreter — a labelled
``argv[0]`` on a ``python3.x`` image, which both breaks the "rides with the
image" rule above and is the exact row an EDR killed 1079 times on a
colleague's Mac on 2026-09-19 (threat name ``python3.14``, the flagged child's
``argv`` = ``Local Operator [daemons] refresh …``). So the link is planted
BESIDE the interpreter that will actually be executed
(:func:`spawn_identity_for_interpreter`), and no spawn site patches
``executable`` itself.

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

# The install layout's stable directory, spelled once for both this module and
# ``update`` — which reaches it through ``~`` and this module through the
# passwd home (see :func:`runtime_root`). ``install_defaults`` is deliberately
# stdlib-only so importing it costs this startup path nothing.
from local_operator.install_defaults import STABLE_ROOT_RELATIVE

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


def _libpython_name_for(real: Path) -> str | None:
    """Basename of the dylib an EXPLICIT interpreter loads, or None.

    :func:`_libpython_name` answers this for the interpreter RUNNING this code,
    out of ``sysconfig``. That answer is wrong for the target of
    :func:`spawn_identity_for_interpreter`, which may be a different CPython
    minor (or a free-threaded build) than ours, and a mismatched libpython
    beside a fresh interpreter is the measured 100%-abort mode.

    Read from the interpreter's OWN FILE NAME rather than by running it: a
    ``python3.12 -> libpython3.12.dylib`` mapping is exact for every
    interpreter uv installs (including the ``python3.13t`` free-threaded
    spelling), and a subprocess here would put an interpreter startup on every
    spawn. Existence is checked, so a name that does not resolve to a real
    dylib is refused rather than planted — a link to nothing is the abort.

    The fallback matters for a tree whose interpreter is not version-named
    (``python3``): exactly one ``libpython*.dylib`` beside it is unambiguous,
    more than one is not, and an ambiguous ABI is refused.
    """
    try:
        lib = real.parent.parent / "lib"
        name = real.name
        if name.startswith("python") and len(name) > len("python"):
            candidate = f"lib{name}.dylib"
            if (lib / candidate).is_file():
                return candidate
        matches = sorted(entry.name for entry in lib.glob("libpython*.dylib"))
        if len(matches) == 1:
            return matches[0]
    except OSError:
        return None
    return None


def _venv_home(prefix: Path) -> Path | None:
    """The base ``bin`` a venv's ``pyvenv.cfg`` names, or ``None``.

    ``executable =`` is honoured first because it is the exact binary the venv
    was created from (uv writes it on some platforms); ``home =`` is what every
    uv/macOS venv carries and is the directory the interpreter name is resolved
    in. Neither is trusted blindly — the caller stats the result.
    """
    try:
        text = (prefix / "pyvenv.cfg").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    home: str | None = None
    executable: str | None = None
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        if not separator:
            continue
        name = key.strip().lower()
        if name == "home":
            home = value.strip()
        elif name == "executable":
            executable = value.strip()
    if executable:
        return Path(executable).parent
    return Path(home) if home else None


def real_interpreter() -> Path | None:
    """The base interpreter this venv was built from, symlinks resolved.

    NOT ``realpath(sys.executable)``, and that is not a detail: once a process
    runs through the branded symlink, ``realpath(sys.executable)`` IS the
    runtime image, and planting from it would clone the clone forever. The
    venv's own ``pyvenv.cfg`` names the build it was created from, and that is
    the build whose image every venv of it should share.

    Falls back to the executable path when there is no venv or no readable
    ``pyvenv.cfg`` (a source checkout's interpreter, a bare ``python -c``).
    ``None`` when there is nothing usable to answer with, and when the answer
    would be a published image that no valid ``source`` marker maps back (see
    :func:`_build_or_none`).
    """
    try:
        if _is_venv():
            home = _venv_home(Path(sys.prefix))
            if home is not None:
                executing = os.path.basename(sys.executable or "")
                names: list[str] = []
                if executing and executing != BRAND:
                    names.append(executing)
                names.append(f"python{sys.version_info.major}.{sys.version_info.minor}")
                names.extend(("python3", "python"))
                for name in names:
                    candidate = home / name
                    if not candidate.is_file():
                        continue
                    resolved = Path(os.path.realpath(candidate))
                    if resolved.is_file():
                        return _build_or_none(resolved)
        executable = sys.executable
        if not executable:
            return None
        return _build_or_none(Path(os.path.realpath(executable)))
    except Exception:  # noqa: BLE001 — a probe, and the caller has a fallback
        return None


def _build_or_none(resolved: Path) -> Path | None:
    """The build ``resolved`` stands for, or ``None`` when it is an unmapped clone.

    A published image maps back through its ``source`` marker. One that does
    not — marker missing, unreadable or refused by :func:`_source_for_image` —
    is still a CLONE, and answering it would make the next plant publish a
    clone of it under a second key: a second macOS privacy row for one Python
    build. ``None`` there means "no plant", which costs this venv the brand and
    nothing else.
    """
    source = _source_for_image(resolved)
    if source is not None:
        return source
    root = runtime_root()
    if root is not None:
        try:
            if _under(Path(os.path.realpath(root)), resolved):
                return None
        except OSError:
            return None
    return resolved


def _real_interpreter() -> Path | None:
    """The private seam :func:`real_interpreter` used to be; kept for callers."""
    return real_interpreter()


def _real_interpreter_for(interpreter: str | os.PathLike[str]) -> Path | None:
    """The base interpreter a DIFFERENT venv's ``bin/python3`` was built from.

    Same reasoning as :func:`real_interpreter`, one tree over: the path the
    caller holds is a symlink into that venv's base install, and resolving it
    with ``realpath`` is right there because the tree is not executing through
    a branded symlink.
    """
    try:
        target = Path(interpreter)
        home = _venv_home(target.parent.parent)
        if home is not None:
            name = os.path.basename(target)
            if name != BRAND:
                candidate = home / name
                if candidate.is_file():
                    resolved = Path(os.path.realpath(candidate))
                    if resolved.is_file():
                        return _build_or_none(resolved)
        return _build_or_none(Path(os.path.realpath(target)))
    except Exception:  # noqa: BLE001 — a probe, and the caller has a fallback
        return None


#: File inside a published runtime directory naming the interpreter it was
#: cloned from. See :func:`_source_for_image`.
_SOURCE_MARKER = "source"


def _under(root: Path, path: Path) -> bool:
    """True when ``path`` is ``root`` or lies beneath it; both already resolved."""
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _source_for_image(path: Path) -> Path | None:
    """The interpreter a published runtime image was cloned from, or ``None``.

    Asked only about a path that is already a real file. A published image is
    not an installation: it has no ``pyvenv.cfg`` of its own and its ``lib`` is
    a symlink into the build, so a venv created ON TOP of it (``<venv>/bin/Local
    Operator -m venv X``) names the IMAGE as both ``home`` and ``executable``.
    Resolving that venv therefore answers the clone, and planting from that
    answer publishes a clone OF a clone under a second build key — a second
    macOS privacy row for the same Python build. Mapping the answer back
    through the marker is what keeps :func:`real_interpreter` answering about a
    build.

    THE MARKER IS ONLY TRUSTED ON A PUBLISHED IMAGE, and the target is only
    trusted when it is an executable Python OUTSIDE the publish tree. Read
    unconditionally, ``<path>/../../source`` exists in trees nothing here wrote
    (a pyenv or framework install, a checkout), and a stray file there was taken
    as the answer — a marker naming ``/bin/ls`` made this function return
    ``/bin/ls``, and the next plant would have published a clone of it. The
    marker is a sidecar of THIS layout, so it is honoured only where that layout
    put it (``<runtime_root>/<key>/bin/<BRAND>``), only when it points back out
    of the tree (a target inside it is another clone, the loop this exists to
    stop), and only when the target names a Python build — the family
    :func:`real_interpreter` draws its answer from, so a target that does not is
    not something a plant here could have written.

    An image published before markers existed, or one whose recorded source has
    since been removed, answers ``None``; the caller then keeps what it
    resolved, exactly as before.
    """
    root = runtime_root()
    if root is None:
        return None
    try:
        image = Path(os.path.realpath(path))
        published = Path(os.path.realpath(root))
    except OSError:
        return None
    if image.name != BRAND or not _under(published, image.parent.parent):
        return None
    try:
        recorded = Path((image.parent.parent / _SOURCE_MARKER).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    try:
        if not recorded.is_file() or not os.access(recorded, os.X_OK):
            return None
        recorded = Path(os.path.realpath(recorded))
    except OSError:
        return None
    if _under(published, recorded):
        return None
    if not (recorded.name.startswith("python") and len(recorded.name) > len("python")):
        return None
    return recorded


def _write_source_marker(published: Path, real: Path) -> None:
    """Record, beside a staging image, which interpreter it was cloned from.

    Written into the STAGING tree so the publish stays one ``rename``. A
    failure here is not fatal — the image is still a working interpreter, it
    just cannot vouch for the build it came from.
    """
    try:
        (published / _SOURCE_MARKER).write_text(f"{os.path.realpath(real)}\n", encoding="utf-8")
    except OSError:
        logger.debug("could not record an image's source", exc_info=True)


def runtime_root() -> Path | None:
    """Where the per-build runtime images live: ``<stable root>/runtime``, or None.

    RESOLVED AGAINST THE PASSWD HOME, NOT ``$HOME``. ``agent/sdk.py`` sets
    ``HOME`` to an agent home for SDK children, so a root derived from the
    environment would plant a second image — a second macOS privacy client —
    for every such child, which is the exact duplication this layout exists to
    remove. The passwd entry is the home that does not move under a process's
    feet, and it is where ``lop`` itself was installed.

    ``LOCAL_OPERATOR_RUNTIME_ROOT`` relocates it outright, for the same reason
    ``LOCAL_OPERATOR_CONFIG_DIR`` exists: a test or a sandbox must keep a run
    out of the operator's real ``~/.local/share/lop``, and the value has to
    cross a ``subprocess`` boundary — the plant happens in the parent, but a
    child re-checks the link it was executed through.
    """
    override = os.environ.get("LOCAL_OPERATOR_RUNTIME_ROOT")
    if override:
        return Path(override)
    try:
        import pwd

        home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    except Exception:  # noqa: BLE001 — no passwd entry is not a startup failure
        # None, NOT a ``~``-shaped placeholder. ``Path("~/")`` is never expanded,
        # so the next ``mkdir(parents=True)`` created a literal ``./~/`` directory
        # in the CURRENT working directory (reproduced) — a plant into the
        # operator's checkout under a name nothing would ever clean up. With no
        # absolute home there is nowhere legitimate to publish, so this says so
        # and the caller takes rung 2.
        return None
    if not home.is_absolute():
        return None
    return home / STABLE_ROOT_RELATIVE / "runtime"


def _build_key(real: Path) -> str:
    """``<python-build-name>-<digest>``: the identity of ONE interpreter build.

    Both halves are load-bearing. The name is what a reader sees in ``ls``
    (``cpython-3.12.14-macos-aarch64-none``), and the digest is over the
    RESOLVED path, size and mtime, so an interpreter re-installed at the same
    path under the same version publishes a NEW image instead of reusing one
    whose bytes no longer match. ``[:10]`` is 40 bits: a collision would need
    two builds colliding in a directory that already identifies the build.
    """
    import hashlib

    stat = real.stat()
    material = f"{os.path.realpath(real)}\0{stat.st_size}\0{stat.st_mtime_ns}"
    return f"{real.parent.parent.name}-{hashlib.sha256(material.encode()).hexdigest()[:10]}"


def runtime_image(real: Path) -> Path | None:
    """The stable image path for ``real``, whether or not it is planted yet."""
    root = runtime_root()
    if root is None:
        return None
    return root / _build_key(real) / "bin" / BRAND


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
    """Where this venv's branded link lives, or None when this environment is out.

    ``<sys.prefix>/bin/Local Operator`` — beside the venv's own ``python``, so a
    process executed through it still finds this venv's ``pyvenv.cfg``. It is
    normally a symlink to the runtime image for this build; on the fallback
    rung it is a per-venv hardlink with its libpython pin in ``<venv>/lib``.
    Returns None (no plant, no re-exec, today's behaviour) when the platform is
    not macOS or the interpreter is not a brandable venv interpreter.
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


def _stdlib_through(image: Path) -> bool:
    """Does the stdlib resolve through the ``lib`` beside ``image``?

    CPython locates ``encodings`` — the first module it imports — relative to
    the DIRECTORY of the image it runs. The runtime image therefore needs
    ``<image dir>/../lib/python3.X/encodings/__init__.py`` to exist, and the
    ``lib`` symlink the plant publishes is what makes that true. Measured
    without it: ``No module named 'encodings'`` on every launch, before any of
    our code runs.
    """
    try:
        lib = image.parent.parent / "lib"
        return any(
            (entry / "encodings" / "__init__.py").is_file() for entry in lib.glob("python3*")
        )
    except OSError:
        return False


#: The first four bytes of every image this module can exec: thin Mach-O
#: (32/64-bit, either byte order) and fat/universal (32/64-bit offsets).
_MACHO_MAGICS = frozenset(
    bytes.fromhex(magic)
    for magic in (
        "feedface",
        "cefaedfe",
        "feedfacf",
        "cffaedfe",
        "cafebabe",
        "cafebabf",
    )
)

#: How far an image's mtime may sit from its source's and still be the same
#: copy. ``clonefile`` keeps the nanoseconds, but the ``shutil.copy2`` fallback
#: lands on whatever the target volume stores: measured on an HFS+ volume, the
#: copy's mtime came back truncated to the second (114,922,487 ns short). An
#: exact comparison would call every copy-fallback image corrupt and republish
#: it on every start. An in-place overwrite stamps "now", which is days from
#: the interpreter's install time, so the slack costs no detection.
_MTIME_SLACK_NS = 2_000_000_000


def _intact_copy(resolved: Path, real: Path | None) -> bool:
    """``resolved`` still holds an executable image, and ``real``'s when known.

    Two stats and a 4-byte read, because ``X_OK`` plus a resolvable stdlib is
    not the whole question an image has to answer. A TRUNCATED clone passes
    both and dies at ``execve`` with ``[Errno 8] Exec format error``
    (reproduced by ``truncate(12)``); so does one overwritten IN PLACE at the
    same size (reproduced by zeroing the first 64 bytes). One image is shared
    by every venv of its build, so either state disables the brand for all of
    them. Size and mtime catch an overwrite (a clone keeps its source's mtime,
    and a write stamps a new one); the Mach-O magic catches one whose size and
    mtime were put back, such as an equal-length script.
    """
    if real is not None:
        image_stat = resolved.stat()
        real_stat = real.stat()
        if image_stat.st_size != real_stat.st_size:
            return False
        if abs(image_stat.st_mtime_ns - real_stat.st_mtime_ns) >= _MTIME_SLACK_NS:
            return False
    with open(resolved, "rb") as handle:
        return handle.read(4) in _MACHO_MAGICS


def _healthy_image(image: Path, real: Path | None = None) -> bool:
    """``image`` is a file we can exec whose bytes and stdlib are intact.

    Follows symlinks. ``real`` is the interpreter the image must be a copy of;
    when it is known, its size and mtime are compared as well as the image's
    magic — see :func:`_intact_copy` for the failures each one catches.
    """
    try:
        resolved = Path(os.path.realpath(image))
        if not resolved.is_file() or not os.access(resolved, os.X_OK):
            return False
        if not _intact_copy(resolved, real):
            return False
        return _stdlib_through(resolved)
    except OSError:
        return False


def _needs_replant(link: Path, real: Path, image: Path | None) -> bool:
    """Whether the planted shape is missing or stale. Stats plus one resolve.

    Identity, not freshness. The expected image path is a FUNCTION of the
    interpreter's own (path, size, mtime), so a symlink naming it is correct by
    construction and anything else is stale: a legacy hardlink, a link aimed at
    a previous build, an EDR's replacement file. The steady state costs a
    couple of ``lstat``s and one ``readlink`` — the whole
    ``ensure_branded_interpreter`` steady state measured ~180 µs against a
    startup north of a second.

    Refresh when ANY of:

    (a) the path is not a symlink at all — a missing link, or the per-venv
        hardlink this layout replaces;
    (b) it names a different image — ``uv tool install --force`` replaced the
        interpreter, so the build key moved;
    (c) the image is not executable by us, its stdlib does not resolve through
        the ``lib`` beside it, or its bytes are no longer the interpreter it
        was cloned from (size, mtime or Mach-O magic, see
        :func:`_intact_copy`). Each shape dies at ``execve`` or at ``No module
        named 'encodings'``, and the last one is a truncated or overwritten
        clone: ``X_OK`` and a resolvable stdlib both pass, then the kernel
        answers ``[Errno 8] Exec format error``;
    (d) the file at the link path is present but WE CANNOT EXECUTE IT. Not
        hypothetical, and not necessarily ours: an EDR on the operator's
        machine replaced the planted link in five worktree venvs with a
        49968-byte mode-0600 file owned by its own agent user, so the path
        named an image that would fail at ``execve`` while every other trigger
        was satisfied by whatever it replaced (measured 2026-09-19).

    The interpreter's bytes are ground truth, through the key, so no
    version-stamp file is written; a stamp is a second source of truth that can
    itself go stale while the image it describes has already been replaced.
    ``real`` is the interpreter the image must be a copy of; it supplies the
    size and mtime in (c), and the key that names the expected image is derived
    from it upstream.
    """
    try:
        if not link.is_symlink():
            return True  # (a)
        if image is None or os.readlink(link) != str(image):
            return True  # (b)
        return not _healthy_image(link, real)  # (c) and (d)
    except OSError:
        return True


def _executable_image(link: Path | None, real: Path | None = None) -> Path | None:
    """``link`` when it is an image we can actually EXEC, else ``None``.

    The last gate before a path is handed to ``subprocess`` as ``executable=``,
    and it exists because the plant can be defeated after it succeeds: an EDR
    on the operator's machine rewrote the link between two of our starts
    (measured 2026-09-19), and a spawn that passed such a path to ``execve``
    dies with ``Permission denied`` at the child's expense. ``None`` is the
    honest answer — it means rung 2, which still runs the requested
    interpreter, just without a name.

    A path INSIDE A VENV gets one extra question — does the stdlib resolve
    through the ``lib`` beside the image — because CPython derives the stdlib
    from the directory of the executable it ends up running, and a broken
    ``lib`` fails with ``No module named 'encodings'`` before any of our code
    runs. Which shape is asked is decided by what the path IS, not by whether
    the venv has a ``pyvenv.cfg``:

    * a SYMLINK always asks. Every ``<venv>/bin/Local Operator`` this module
      plants is a symlink to the runtime image, whose directory (``<key>/bin``)
      is where CPython looks — measured: a venv link to an image whose ``lib``
      dangles returns from here and then dies with ``No module named
      'encodings'``, rc=1;
    * an in-venv HARDLINK does not. It IS the venv's interpreter, so CPython
      finds the venv's own ``pyvenv.cfg`` beside it and takes the stdlib from
      its base install; requiring a second ``lib`` under ``<venv>`` would
      refuse a shape that starts.

    Every shape also gets the integrity question (:func:`_intact_copy`): the
    Mach-O magic always, and size and mtime against ``real`` when it is known.
    """
    if link is None:
        return None
    try:
        if not os.access(link, os.X_OK):
            return None
        target = Path(os.path.realpath(link))
        if link.is_symlink() or not (link.parent.parent / "pyvenv.cfg").is_file():
            if not _stdlib_through(target):
                return None
        if not _intact_copy(target, real):
            return None
        return link
    except OSError:  # pragma: no cover — access answers False for a bad path
        return None


def _sweep_orphan_temps(directory: Path, prefix: str) -> None:
    """Remove ``.<prefix>.<pid>[.<thread>].tmp`` entries left by a plant that was killed.

    The plant is link-to-temp + ``os.replace``, which is atomic and cleans up
    after itself on an ``OSError`` — but NOT when the process is killed between
    the two steps (reproduced with ``kill -9``: the temp entry survives). Each
    orphan is a harmless zero-byte-of-data directory entry, but they accumulate
    once per crashed startup and turn ``ls <venv>/bin`` into a junkyard.

    Only entries whose embedded pid is no longer alive are removed, so a plant
    running concurrently in another worktree is never disturbed. That question
    is answered by TWO probes, and the second one is the fix: existence, then
    zombiehood.

    Existence goes through :func:`procstate.pid_alive` and NEVER through
    ``os.kill(pid, 0)``. That call is a liveness PROBE on POSIX and TERMINATE on
    Windows — CPython's ``os_kill_impl`` ends in ``TerminateProcess`` for any
    signal that is not ``CTRL_C_EVENT`` / ``CTRL_BREAK_EVENT`` — so a sweep
    written on it would kill whichever live process happened to match a stale
    entry's embedded pid, and then read the successful kill as "still running"
    and keep the entry. Housekeeping must not be able to do that, and this is
    the same shared probe the rest of the package was moved onto.

    A ZOMBIE IS NOT A LIVE PLANT, THOUGH, and existence is not the whole
    question because a zombie exists. ``pid_alive`` is
    ``pid_liveness(pid) is not False``, and on POSIX ``pid_liveness`` is itself
    signal 0 — so it answers for an exited-but-unreaped process exactly as it
    does for a running one, which is the same answer the raw call it replaced
    gave. Measured here: a ``kill -9``'d child reads as alive through signal 0
    while ``ps`` reports it as ``Z``. That is precisely the fleet-wide case this
    sweep exists for, so the error is worst exactly when it matters: when a
    whole fleet dies at once (the incident shape this repo has now seen three
    times) every leaked temp belongs to a zombie and NONE is reclaimed, because
    the parent that would have reaped it is the long-lived frontend that just
    lost all its children.

    The pids that read alive are therefore asked the ZOMBIE question as well,
    through :func:`procstate.zombie_states` — the module that owns it, in ONE
    ``ps`` fork for the whole surviving set, rather than through a second probe
    written here that could answer it differently.

    A pid the zombie probe does not answer for reads as NOT a zombie, the
    fail-closed direction ``procstate`` documents: an unprobeable set may cost a
    leftover file, and must never cost a live plant its temp. Another account's
    pid lands on that side by construction rather than through the branch this
    function used to carry — ``pid_alive`` answers True for ``EPERM`` (the
    process exists and cannot be a corpse of ours), and ``ps`` still reports
    such a pid's state, which is not a zombie — so it is passed over, exactly as
    the old ``EPERM`` arm left it.
    """
    try:
        # Function-local import: this sweep runs only on the REPLANT path, and
        # ``procstate`` is a module body nothing above this function reads — the
        # house rule here is that the common startup (a stat probe and no writes)
        # pays for nothing it does not use (see the ``update`` and ``ctypes``
        # imports below).
        #
        # THE SAVING IS ONE MODULE, MEASURED (review round 4, MINOR 2):
        # ``import local_operator.procname`` leaves BOTH ``procstate`` and
        # ``subprocess`` out of ``sys.modules``, but ``subprocess`` is not what this
        # deferral is buying. ``cli.py`` imports it at module scope, so on the ``lop``
        # path the module-level form would add only ``procstate`` — a stdlib-only
        # body. The deferral is kept because that body is still the whole saving and
        # because the sweep is not on the common path at all; the locality is pinned
        # by ``test_importing_procname_defers_procstate``, so collapsing it reads as
        # a no-op cleanup only until that test reddens.
        from local_operator import procstate

        gone: list[Path] = []
        unproven: dict[int, Path] = {}
        for entry in directory.glob(f".{prefix}.*.tmp"):
            # THE PID, NOT THE WHOLE TAIL: a temp is ``.<brand>.<pid>.<thread>.tmp``
            # (see ``_plant_hardlink`` on why the thread is there), and the thread
            # component is what this sweep does NOT need to answer its question —
            # the pid is the planter either way.
            pid_text = entry.name[len(prefix) + 2 : -4].split(".", 1)[0]
            if not pid_text.isdigit():
                continue
            # POSIX semantics preserved exactly: a gone pid is False and is
            # swept, while EPERM (alive, another user's) and an unanswerable
            # probe both fail CLOSED to "alive" — ``pid_alive`` is
            # ``pid_liveness(pid) is not False`` — which leaves the entry alone
            # rather than racing somebody else's plant. "Alive" is not the end
            # of it: it is the input to the zombie probe below.
            if not procstate.pid_alive(int(pid_text)):
                gone.append(entry)
                continue
            unproven[int(pid_text)] = entry
        zombies = procstate.zombie_states(unproven)
        # A pid absent from the probe's answer reads as NOT a zombie, the same
        # fail-closed direction ``procstate`` documents: an unprobeable set may
        # cost a leftover file, and must never cost a live plant its temp.
        gone.extend(entry for pid, entry in unproven.items() if zombies.get(pid, False))
        for entry in gone:
            try:
                entry.unlink()
            except OSError:
                pass
    except Exception:  # noqa: BLE001 — housekeeping never fails a startup
        logger.debug("orphan temp sweep skipped", exc_info=True)


def _temp_thread_token() -> str:
    """The thread component of a plant's temp name (see :func:`_plant_hardlink`).

    WHY THE NAME NEEDS A SECOND COMPONENT AT ALL: the plant unlinks its own
    ``.<brand>.<pid>.tmp`` before linking, which is right for one thread and wrong
    for two — a second plant in the same process would drop the first's freshly
    written temp, its ``os.link`` would then fail ``ENOENT``, and one plant would
    have cost a sibling its rung 1 (review round 1, NIT 2). All four
    ``ensure_branded_interpreter()`` call sites are single-threaded startup paths
    today, so that collision is unreachable rather than impossible, and a name that
    cannot collide is cheaper than the reasoning that keeps re-proving it cannot.

    Isolated at this seam to keep ``threading`` out of the common startup, which is
    this module's standing house rule: the plant runs only on the REPLANT path, and
    everything above it is a stat probe and no writes.
    """
    import threading

    return str(threading.get_ident())


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
    tmp = link.with_name(f".{BRAND}.{os.getpid()}.{_temp_thread_token()}.tmp")
    try:
        link.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        os.link(real, tmp)
        os.replace(tmp, link)
        # THE TEMP IS CONSUMED ONLY WHEN THE TWO NAMES ARE DIFFERENT INODES, and
        # the ordinary replant is the case where they are the SAME one: trigger
        # (c) (a missing or wrong companion libpython) fires while the link
        # already names the right interpreter, so ``link`` and ``tmp`` are two
        # hardlinks to one inode — and POSIX says ``os.replace`` of two
        # hardlinks to one file "shall return successfully and perform no other
        # action". The temp therefore survived EVERY such replant, one per
        # stale-fleet startup: measured on this host as 573 leftovers across 17
        # generations, each a real directory entry in ``<venv>/bin`` (and
        # reproduced directly: one ``_plant_hardlink`` call onto the link's own
        # inode leaves ``.<BRAND>.<pid>.tmp`` behind).
        #
        # Unlinking the NAME is safe in both shapes and touches nothing else:
        # when the replace did consume it this raises ``ENOENT``; when it did
        # not, the temp is a second name for ``link``'s own inode, so removing
        # it leaves ``link`` — the file the caller asked for — exactly as it
        # was, and every process already running that inode keeps running it.
        # ``link`` itself is never unlinked here, and no mode or owner is
        # changed: see the module docstring on why mutating the shared inode's
        # metadata is forbidden.
        try:
            os.unlink(tmp)
        except OSError:
            pass  # already consumed by the replace — the expected case
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


def _clone_or_copy(source: Path, target: Path) -> None:
    """Make ``target`` an APFS clone of ``source``, else a byte copy.

    ``clonefile`` shares blocks instead of copying them and preserves mode and
    times, which is what makes one image per Python build cost no extra blocks
    on APFS: the image is logically the interpreter's whole size (18,041,888
    bytes for the uv 3.12.14 build this was measured on) and costs no measured
    extra allocation while the source still exists, because every block is
    shared. On a volume without clones ``shutil.copy2`` below pays that size
    once per build key — still one copy per BUILD rather than one per venv,
    which is the property that matters. It is called through libc
    rather than ``os.clonefile`` because this ships on interpreters from 3.12,
    where that binding does not exist. Any failure — a non-APFS volume, a full
    disk, an unusual filesystem — falls through to ``shutil.copy2``, which
    preserves the same attributes.
    """
    import shutil

    if sys.platform == "darwin":
        try:
            import ctypes

            libc = ctypes.CDLL(None, use_errno=True)
            clonefile = libc.clonefile
            clonefile.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
            clonefile.restype = ctypes.c_int
            if clonefile(os.fsencode(source), os.fsencode(target), 0) == 0:
                return
        except Exception:  # noqa: BLE001 — any failure falls through to the copy
            pass
    shutil.copy2(source, target)


#: The two private names a runtime plant puts beside a build key: its staging
#: tree, and a corrupt published tree it moved aside before replacing it.
_PLANT_SCRATCH_INFIXES = (".tmp.", ".dead.")


def _has_plant_scratch(root: Path) -> bool:
    return any(any(root.glob(f"*{infix}*")) for infix in _PLANT_SCRATCH_INFIXES)


def _sweep_orphan_plants(root: Path) -> None:
    """Remove ``<key>.tmp|dead.<pid>.<thread>`` trees a killed plant left behind.

    The runtime plant's staging name is a DIRECTORY beside the image, so a
    planter killed between ``mkdir`` and ``rename`` leaks a tree; one killed
    between moving a corrupt tree aside and removing it leaks that. The liveness
    question is the one :func:`_sweep_orphan_temps` asks — ``procstate``, never
    ``os.kill``, and a zombie counts as dead — and the direction is fail-closed:
    a pid the probe cannot answer for keeps its tree, and a tree that cannot be
    removed is left alone. Housekeeping, so every failure here is silent.
    """
    import shutil

    try:
        from local_operator import procstate

        for infix in _PLANT_SCRATCH_INFIXES:
            for entry in root.glob(f"*{infix}*"):
                if entry.is_symlink() or not entry.is_dir():
                    continue
                _, _, tail = entry.name.rpartition(infix)
                pid_text = tail.split(".", 1)[0]
                if not pid_text.isdigit():
                    continue
                pid = int(pid_text)
                if procstate.pid_alive(pid) and not procstate.zombie_states([pid]).get(pid, False):
                    continue
                shutil.rmtree(entry, ignore_errors=True)
    except Exception:  # noqa: BLE001 — housekeeping must never cost a startup
        return


def _plant_runtime(real: Path, root: Path | None) -> Path | None:
    """Publish the one runtime image for ``real`` under ``root``, or ``None``.

    Built under a PRIVATE staging directory and published with a single
    ``os.rename``, so a concurrent planter (several worktrees starting at once)
    sees either no directory or a complete one. Published directories are
    IMMUTABLE UNLESS CORRUPT and are never collected: a running process may
    still be executing one, and one that stays on disk is one macOS privacy
    prompt that will not happen again. A published directory whose image fails
    the integrity check (:func:`_healthy_image`) is replaced under the same
    key. A loser of the rename race removes its own staging tree and uses the
    winner's image.

    The two refusals are about the SOURCE, not this process: an interpreter
    whose own tree has no stdlib cannot back an image (the clone would die with
    ``No module named 'encodings'``), and one whose dylib cannot be found would
    die in dyld. Either way the hardlink rung is the honest answer, so this
    returns ``None`` rather than publishing an image that aborts.
    """
    import shutil

    try:
        if root is None:
            return None
        key = _build_key(real)
        image = root / key / "bin" / BRAND
        if _healthy_image(image, real):
            return image
        if not _stdlib_through(real):
            return None
        base = real.parent.parent
        name = _libpython_name_for(real)
        if name is not None and not (base / "lib" / name).is_file():
            return None
        root.mkdir(parents=True, exist_ok=True)
        # Ask before sweeping: a sweep forks a process (``procstate`` runs
        # ``ps`` for its zombie probe) and this runs inside a spawn path. With
        # no staging directory there is nothing to reclaim, and the common
        # fresh plant must not pay for a probe of the machine.
        if _has_plant_scratch(root):
            _sweep_orphan_plants(root)
        staging = root / f"{key}.tmp.{os.getpid()}.{_temp_thread_token()}"
        try:
            (staging / "bin").mkdir(parents=True)
            _clone_or_copy(real, staging / "bin" / BRAND)
            _write_source_marker(staging, real)
            # The stdlib and the dylib are both resolved relative to the IMAGE
            # directory, so both must resolve back into the interpreter's own
            # tree. ``include`` is only needed by a build that ships headers.
            os.symlink(base / "lib", staging / "lib")
            if (base / "include").is_dir():
                os.symlink(base / "include", staging / "include")
        except OSError:
            # A FAILED BUILD IS NOT A PUBLISHED IMAGE. This block used to fall
            # through to the corrupt-directory branch below, which ``rename``d a
            # staging tree holding an empty ``bin/`` (or a half-written clone,
            # on ENOSPC) onto the build key. Nothing could ever call it healthy,
            # but it published a key a reader can see and sent the next planter
            # down the corrupt-repair path for a directory that never held an
            # image. Remove it and answer rung 2.
            shutil.rmtree(staging, ignore_errors=True)
            return image if _healthy_image(image, real) else None
        try:
            os.rename(staging, root / key)
        except OSError:
            if _healthy_image(image, real):
                # A concurrent planter published it first, and it is good.
                shutil.rmtree(staging, ignore_errors=True)
                return image
            # The published directory is CORRUPT (a truncated or replaced
            # image), not a race winner. Published directories are immutable
            # unless corrupt: an image that does not execute cannot have a live
            # process behind it, so it is ours to replace instead of downgrading
            # every venv of this build to the fallback for the life of the
            # machine.
            return _replace_corrupt(root, key, staging, image, real)
        return _settled_image(image, real)
    except Exception:  # noqa: BLE001 — rung 2 exists for exactly this
        logger.debug("runtime image unavailable", exc_info=True)
        return None


def _replace_corrupt(root: Path, key: str, staging: Path, image: Path, real: Path) -> Path | None:
    """Swap a corrupt published ``<root>/<key>`` for ``staging``; the image or ``None``.

    ``os.rename`` cannot replace a non-empty directory, so the dead tree has to
    leave first — and it leaves by RENAME, to a private ``<key>.dead.<pid>.
    <thread>``, never by an in-place ``rmtree``. Deleting in place left the key
    missing for as long as the delete ran, and concurrent planters that looked
    in that window answered ``None``: measured, 139 of 240 planters (8 at a time
    on a truncated image), each one a venv dropped to the hardlink rung. The
    aside rename is atomic, so the key is missing only between two renames.

    The health question is asked again right before the move, because a
    concurrent planter may have published a good tree since this one looked. A
    planter that still moves aside a tree which turns out HEALTHY puts it back
    rather than deleting a fresh image, and in that short gap another planter
    can look at an empty key — so every answer here goes through
    :func:`_settled_image`, which yields briefly before calling the key dead.
    """
    import shutil

    published = root / key
    if _healthy_image(image, real):
        shutil.rmtree(staging, ignore_errors=True)
        return image
    aside = root / f"{key}.dead.{os.getpid()}.{_temp_thread_token()}"
    try:
        os.rename(published, aside)
    except OSError:
        pass  # another planter moved it first, or it is already gone
    else:
        if _healthy_image(aside / "bin" / BRAND, real):
            try:
                os.rename(aside, published)
            except OSError:
                pass  # someone published in the gap; theirs stands
        shutil.rmtree(aside, ignore_errors=True)
    try:
        os.rename(staging, published)
    except OSError:
        shutil.rmtree(staging, ignore_errors=True)
    return _settled_image(image, real)


#: How long a planter on the repair path waits for a concurrent swap to land
#: before it calls the key dead: ten yields of 20 ms. Only the corrupt-image
#: path pays it; the steady state never reaches this function.
_SETTLE_TRIES = 10
_SETTLE_PAUSE_S = 0.02


def _settled_image(image: Path, real: Path) -> Path | None:
    """``image`` once it is healthy, or ``None`` if it stays unhealthy briefly.

    A concurrent repair holds the key empty (or a healthy tree aside) for the
    span of two renames. Reading the key once in that span answered ``None`` —
    one venv dropped to the hardlink rung for nothing — so the answer is
    re-read a bounded number of times first.
    """
    import time

    for _ in range(_SETTLE_TRIES):
        if _healthy_image(image, real):
            return image
        time.sleep(_SETTLE_PAUSE_S)
    return image if _healthy_image(image, real) else None


def _link_to(link: Path, image: Path) -> bool:
    """Point ``link`` at ``image`` (temp symlink + ``os.replace``), or False.

    ``os.replace`` is atomic, so a concurrent reader sees the old link or the
    new one and never nothing. A LEGACY HARDLINK at the same path is replaced
    here, which is how a machine that already has the per-venv shape migrates
    without an install step. A link that already names the image is left
    untouched: this runs at every startup, and churning it for nothing is a
    write an editor or a backup notices.
    """
    try:
        if link.is_symlink() and os.readlink(link) == str(image):
            return True
        # A killed planter's temp is the same shape this is about to create, and
        # the sweep only forks its probe when one exists.
        _sweep_orphan_temps(link.parent, BRAND)
        staging = link.parent / f".{BRAND}.{os.getpid()}.{_temp_thread_token()}.tmp"
        try:
            os.symlink(image, staging)
            os.replace(staging, link)
        finally:
            try:
                staging.unlink()
            except OSError:
                pass
        return True
    except OSError:
        return False


def _is_runtime_link(link: Path, real: Path | None = None) -> bool:
    """Is ``link`` already a symlink to a healthy image under the runtime root?

    The question the hardlink rung asks before touching anything, and it is
    what stops a sandboxed child from undoing its parent: a child whose runtime
    root is read-only cannot plant, and it must not downgrade a shape that
    already works to one that costs a new path per venv.

    "Healthy" includes "the same bytes as ``real``" when the caller knows the
    interpreter (see :func:`_healthy_image`).
    """
    try:
        if not link.is_symlink():
            return False
        target = Path(os.path.realpath(link))
        if not target.is_file():
            return False
        root = runtime_root()
        if root is None or Path(os.path.realpath(root)) not in target.parents:
            return False
        return _healthy_image(target, real)
    except OSError:
        return False


def _stale_hardlink(link: Path, real: Path, libpython: Path | None) -> bool:
    """Whether the FALLBACK hardlink shape is missing or stale.

    The inode rule the runtime rung replaced, kept because the fallback's
    failure modes are inode-shaped: (a) the link names a different inode —
    ``uv tool install --force`` replaced the interpreter and preserved the
    planted file; (b) ``st_nlink < 2`` — something turned it into a copy; (c)
    the companion libpython symlink is missing, dangling, or aimed at another
    install's library (the measured 100%-abort mode); (d) the file is present
    but not executable by us, which is the EDR shape measured 2026-09-19.
    Never raises.
    """
    try:
        link_stat = link.stat()  # follows nothing meaningful: a hardlink IS the file
    except OSError:
        return True
    try:
        real_stat = real.stat()
    except OSError:
        # The interpreter vanished mid-check. Nothing sane to plant against;
        # leave what is there rather than deleting a link that may still work.
        return False
    if (link_stat.st_dev, link_stat.st_ino) != (real_stat.st_dev, real_stat.st_ino):
        return True
    if link_stat.st_nlink < 2:
        return True
    try:
        if not os.access(link, os.X_OK):
            return True
    except OSError:
        return True
    if libpython is not None:
        # ``exists`` follows the symlink, so a dangling one is already False.
        # Inside the try because it is NOT total: an unreadable parent
        # directory makes it raise ``PermissionError`` (reproduced).
        try:
            if not libpython.exists():
                return True
            if libpython.resolve() != (real.parent.parent / "lib" / libpython.name).resolve():
                return True
        except OSError:
            return True
    return False


def _plant_branded_image(link: Path, real: Path, name: str | None) -> Path | None:
    """Refresh the brand beside ``real``, preferring the stable runtime image.

    Factored out of :func:`ensure_branded_interpreter` so the SAME logic can
    plant beside an interpreter that is not this process's — the two spawn
    sites that must run another tree's build (see
    :func:`spawn_identity_for_interpreter`). Everything it needs is an explicit
    argument: ``link`` is where the branded name must appear, ``real`` is the
    interpreter it must name, and ``name`` is the dylib (from ``sysconfig`` for
    this venv, from the target's own ``lib`` otherwise).

    THE LADDER, and the order is the contract:

    1. the runtime image, and a symlink to it — one path per Python build, so
       macOS sees one privacy client instead of one per venv and per update;
    2. the per-venv hardlink and its dylib pin, ONLY when the runtime rung
       could not be taken and the path is not ALREADY a healthy runtime link.
       That second condition is what keeps a sandboxed child, whose runtime
       root is read-only, from undoing the link its parent planted;
    3. ``None`` — rung 2 for the caller, which still runs the requested
       interpreter, just without a name.

    Returns the usable link or ``None``. Never raises.
    """
    try:
        # The steady state: a symlink naming the image for THIS build, and an
        # image that can start. No writes, no directory scan.
        if not _needs_replant(link, real, runtime_image(real)):
            return _executable_image(link, real)
        image = _plant_runtime(real, runtime_root())
        if image is not None:
            if _link_to(link, image):
                return _executable_image(link, real)
        if _is_runtime_link(link, real):
            return _executable_image(link, real)
        if name is None:
            # A hardlink with no companion dylib aborts at load (the measured
            # 100%-abort mode), and "no dylib" is indistinguishable from "a
            # dylib we could not identify", so the fallback is refused rather
            # than planted blind.
            return None
        libpython = link.parent.parent / "lib" / name
        if not _stale_hardlink(link, real, libpython):
            return _executable_image(link, real)
        # BOTH plant directories. A killed plant can leak a temp in `bin/`
        # (named for the brand) or in `lib/` (named for the dylib), and sweeping
        # only the first left lib temps accumulating forever. Only on the
        # replant path, so the common startup stays a stat probe with no
        # directory scan.
        _sweep_orphan_temps(link.parent, BRAND)
        _sweep_orphan_temps(link.parent.parent / "lib", name)
        if not _plant_libpython(link, real, name):
            return None
        if not _plant_hardlink(link, real):
            return None
        return _executable_image(link, real)
    except Exception:  # noqa: BLE001 — a name is decoration, never a failure
        logger.debug("branded image unavailable", exc_info=True)
        return None


def ensure_branded_interpreter() -> Path | None:
    """Plant/refresh the branded interpreter image; return its path or None.

    Cheap enough for every startup: the common case is the staleness probe
    above (~180 µs measured, against a startup already over a second) and no
    writes. Only a first run in a fresh venv, or a genuine staleness trigger,
    does any filesystem work — and publishing the runtime image costs one
    clone plus two symlinks per PYTHON BUILD, not per venv, which is what
    makes this safe for the many concurrent worktrees on a development machine
    (they converge on one image and never contend for one path beyond the
    rename that publishes it).

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
        return _plant_branded_image(link, real, _libpython_name())
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

#: Interpreter options that may legitimately precede the program on the
#: command line. Deliberately EXACTLY these: each is either measured on a real
#: launch of this product (``-P``, what every spawn site of ours passes) or is
#: an option an operator or an app can plausibly add to a ``lop`` line. An
#: option not listed here answers "not our launch", which only costs a name —
#: see :func:`is_own_launch` for why that direction is the safe one.
_LEADING_OPTION_FLAGS = frozenset({"-P", "-u", "-E", "-s", "-I", "-B"})

#: The same, for options that take a SEPARATE value (``-X dev``, ``-W ignore``):
#: the token after one of these is its value, not the program.
_OPTION_WITH_VALUE_FLAGS = frozenset({"-X", "-W"})

#: Long options that can precede the program, always with an attached ``=``.
_LONG_OPTION_PREFIXES = ("--check-hash-based-pycs",)


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

    **LEADING INTERPRETER OPTIONS ARE SKIPPED, because they are not the
    program.** ``python3 -P -m local_operator.cli serve …`` is a real launch —
    it is the shape ``server/reload.py`` execs as a reload successor, and the
    shape a ``lop serve`` daemon has after one (measured live on the operator's
    machine: pid 1343, ``python3.14``, ppid 1, ``--listener-fd``). Reading
    ``orig_argv[1]`` directly answered False for every one of those, so the
    successor never branded itself and the daemon stayed a bare ``python3.x``
    row for its whole life. The options that may legitimately precede the
    program are skipped one by one — the ones that take a SEPARATE value
    (``-X dev``, ``-W ignore``) skip their value too — and an UNKNOWN option
    still answers False, which is the safe direction: it can delay a name, but
    it can never re-exec something that is not a launch of this product.
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
        index = 1
        while index < len(argv):
            first = argv[index]
            if first.startswith("-c"):
                # The ``-c`` shape, and it stays a non-goal even when options
                # precede it: see the paragraph above. ``startswith`` rather
                # than equality because CPython accepts ``-ccode`` too, and a
                # missed ``-c`` is the app-compat break this refuses.
                return False
            if first == "-m":
                if index + 1 >= len(argv):
                    return False
                module = argv[index + 1]
                # The package, or ANY submodule of it. `-m local_operator.x`
                # is a launch of this product by construction, and the modules
                # the product spawns itself are all in that set
                # (`local_operator.cli`, `local_operator.session.runtime
                # .process`, `local_operator.tools.eval_worker`,
                # `local_operator.secrets.brokerd`, the wake supervisor, the
                # tunnel service, the browser bridge). The prefix test is
                # bounded at a dot, so a similarly-named distribution
                # (`local_operators`) cannot slip in; a sibling module that is
                # NOT ours would have to be literally inside this package.
                return module == "local_operator" or module.startswith("local_operator.")
            if first in _OPTION_WITH_VALUE_FLAGS:
                index += 2  # the option AND its separate value
                continue
            if first in _LEADING_OPTION_FLAGS:
                index += 1
                continue
            if first.startswith(("-X", "-W")) or first.startswith(_LONG_OPTION_PREFIXES):
                # Attached spelling (``-Xdev``, ``-Wignore``,
                # ``--check-hash-based-pycs=always``). The detached spellings
                # are handled above; a long option given AS ``name value`` is
                # not something CPython accepts, so it is not skipped here
                # either.
                index += 1
                continue
            if first.startswith("-"):
                return False  # an option we do not know: never a launcher
            return os.path.basename(first) in _LAUNCHER_NAMES
        return False  # options and nothing else: a bare REPL, not a launch
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


def _branded_link_beside(interpreter: str | os.PathLike[str]) -> Path | None:
    """Where the branded link for an EXPLICIT interpreter belongs, or ``None``.

    The target-side twin of :func:`branded_link_path`, and the reason that one
    cannot be reused: it answers with ``<sys.prefix>/bin/<BRAND>``, the venv
    THIS process runs from, while a cross-tree spawn site needs the venv of the
    interpreter it is about to execute.

    Venv-ness is answered by the target's OWN ``pyvenv.cfg`` rather than by
    :func:`_is_venv` (which compares ``sys.prefix`` against ``sys.base_prefix``
    and would therefore describe us). A target with no ``pyvenv.cfg`` beside it
    is a system/Homebrew/managed interpreter, and planting into one is out of
    the question for a cosmetic feature — same refusal as the own-venv path,
    reached from the other direction.
    """
    try:
        if sys.platform != "darwin":
            return None
        if len(BRAND) > _PROC_NAME_MAX:
            return None
        target = Path(interpreter)
        root = target.parent.parent
        if not (root / "pyvenv.cfg").is_file():
            return None
        if _is_framework_build(Path(os.path.realpath(target))):
            return None
        return root / "bin" / BRAND
    except OSError:
        return None


def spawn_identity_for_interpreter(
    label: str, interpreter: str | os.PathLike[str], **fields: object
) -> tuple[str, str | None]:
    """``(argv[0], executable)`` for a child that runs a DIFFERENT interpreter.

    :func:`spawn_identity` brands THIS process's venv, so a site that must run
    another tree's build — the post-upgrade repair child (``update``) and a
    runtime engaged onto the next generation (``session/runtime/launch``) —
    could only ever keep the label and replace the image, which produced the
    ``python3.14`` row an EDR killed 1079 times on 2026-09-19. This plants the
    link BESIDE ``interpreter`` and hands back the pair, so both axes describe
    the tree that will actually be executed.

    THE TARGET'S VENV IS WHAT MAKES THE NAME STICK. The link lives in
    ``<target venv>/bin``, so the kernel takes ``p_comm`` from it AND the child
    still resolves ``sys.prefix`` from the ``pyvenv.cfg`` beside it — the
    generation and build the caller asked for are unchanged. That is not a
    side effect to preserve carefully; it is what the plant is for.

    Returns ``(str(interpreter), None)`` — rung 2, the bare interpreter with
    the label deliberately WITHHELD — when no image can be planted (not
    darwin, not a venv, a framework build, cross-device, a read-only prefix, no
    dylib to pin) or when the link that is there cannot be executed by us. The
    caller MUST pass both halves straight through, and must not treat ``None``
    as "use my own executable": the whole point is that this child runs the
    OTHER build.

    Cheap in the steady state: one ``pyvenv.cfg`` probe and one ``realpath``,
    then the plant path's own staleness probe; a plant happens only on a miss
    (a fresh venv, a replaced interpreter, or an EDR-neutered link).

    Never raises — a name is decoration, and a spawn may not fail for one.
    """
    target = str(interpreter)
    try:
        link = _branded_link_beside(target)
        if link is None:
            return target, None
        real = _real_interpreter_for(target)
        if real is None:
            return target, None
        image = _plant_branded_image(link, real, _libpython_name_for(real))
        if image is None:
            return target, None
        return branded_argv0(label, **fields), str(image)
    except Exception:  # noqa: BLE001 — a name is decoration, never a failure
        logger.debug("cross-tree spawn identity unavailable", exc_info=True)
        return target, None


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
