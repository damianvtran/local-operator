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

ONE MACOS PRIVACY IDENTITY (bundle-wrapped runtime images)
----------------------------------------------------------
macOS's privacy database (TCC) keys a BARE Mach-O by the PATH it was executed
from (plus its cdhash), so every new path is a new System Settings row and a
new prompt. History: per-venv hardlinks (2026-10-09 → 10-18) made one row per
generation and per worktree; one image per Python build (since 10-18) still
made one row per Python build. ~13 "Local Operator" rows accumulated on the
operator's host.

Measured 2026-11-09 with tccd's ``AUTHREQ_SUBJECT`` (preflight only, child
spawned disclaimed):

- the bare image, ad-hoc → subject is the PATH; signing a bare binary with
  ``--identifier`` changes nothing (still the path);
- the same bytes in ``X.app/Contents/MacOS/`` with a bound Info.plist → subject
  is the BUNDLE ID, and an identical bundle at a second path is the SAME
  subject; a non-``.app`` extension (``.lopruntime``) keys the same way;
- a venv ``bin/Local Operator`` symlink to the bundle executable → subject is
  the bundle id, ``sys.prefix`` is the venv, ``proc_pidpath`` is the bundle
  executable.

So every runtime image is now a bundle, and the row follows its id::

    ~/.local/share/lop/runtime/<build-key>-<flavor>/
        lib      -> <real>/../lib          # BESIDE the bundle, never inside
        include  -> <real>/../include
        source                             # the interpreter it was cloned from
        image                              # "<size> <mtime_ns> <flavor> <id>"
        Local Operator.lopruntime/Contents/
            Info.plist                     # CFBundleIdentifier, LSBackgroundOnly
            MacOS/Local Operator           # clone; LC_RPATH -> @loader_path/../../../lib
            _CodeSignature/
    <venv>/bin/Local Operator              -> …/MacOS/Local Operator

Two flavors. ``adhoc`` (``com.local-operator.runtime``, "Local Operator
Runtime") is what every spawn site plants: ad-hoc signing never reads a
keychain, so it is safe on the session path and under an isolated ``HOME``.
``team`` (``com.local-operator``, the desktop app's own id) is signed with the
operator's Developer ID by :func:`publish_team_runtime` — only from ``lop
update`` and ``lop permissions setup``, behind a recorded opt-in — and shares
the APP's grant: one row for the app and every runtime (measured on the
operator's host, see that function). A healthy team image is preferred; an
ad-hoc image NEVER carries the app's id (a csreq mismatch could overwrite the
app's grant). The disclaim and launchd ownership — what keeps runtimes alive
when the UI dies — are untouched: they depend on process responsibility, not
on the image's code identity, and no termination path in either repo selects a
process by bundle id (``NSRunningApplication`` never lists a runtime; the
bundle is never LaunchServices-registered).

``lib`` sits beside the bundle and the interpreter's only ``LC_RPATH`` is
rewritten in place (:func:`_rewrite_rpath`) because an out-of-bundle symlink in
``Contents`` fails ``codesign --strict``; if a build's rpath does not fit, the
ad-hoc flavor falls back to a ``Contents/lib`` link (still keyed by bundle id,
not strict-clean) and the team flavor is refused. If the bundle cannot be
built at all, the legacy ``<build-key>/bin/Local Operator`` image is published
exactly as before — a cosmetic identity never fails a launch.

``<build-key>`` is ``<real.parent.parent.name>-<sha256(realpath, size,
mtime_ns)[:10]>``: a new Python build is a new key, and a replaced interpreter
cannot be mistaken for the one already published. Published directories are
IMMUTABLE UNLESS CORRUPT and are never garbage-collected — a running process
may still be executing one. A venv link naming the legacy image is stale and
repoints at the bundle on its next start; the legacy tree itself stays on
disk. The one exception to immutability is a published directory whose image
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

#: The runtime image's bundle wrapper. TCC keys a bare Mach-O by PATH and a
#: bundled one by ``CFBundleIdentifier`` (tccd ``AUTHREQ_SUBJECT``, measured
#: 2026-11-09 — see "ONE MACOS PRIVACY IDENTITY" in the module docstring). The
#: extension is deliberately NOT ``.app``: tccd still keys a ``.lopruntime`` by
#: its bundle id, while LaunchServices never treats it as an application, so it
#: is never registered, never offered by ``open -b``, and never an
#: ``NSRunningApplication`` a bundle-id selector could pick (30+ execs measured,
#: zero registrations).
RUNTIME_BUNDLE_DIRNAME = "Local Operator.lopruntime"

#: The desktop app's bundle id. Only a TEAM-SIGNED runtime may carry it: an
#: image whose signature satisfies the app's designated requirement shares the
#: app's privacy grant (one row). An AD-HOC image carrying it would collide
#: with the app's row and a csreq mismatch can re-prompt and overwrite the
#: APP's grant — that pairing is forbidden, and :func:`_bundle_identifier`
#: is the one place that maps a flavor to an id so it cannot happen by typo.
APP_BUNDLE_ID = "com.local-operator"

#: The ad-hoc runtime's bundle id: "Local Operator Runtime", one row per host
#: (re-toggled after a Python-build change, because an ad-hoc requirement is a
#: cdhash). Never equal to :data:`APP_BUNDLE_ID`.
RUNTIME_BUNDLE_ID = "com.local-operator.runtime"

#: Display name of the ad-hoc runtime row in System Settings.
RUNTIME_DISPLAY_NAME = "Local Operator Runtime"

#: The two signing flavors of a runtime bundle, in PREFERENCE order. ``team``
#: is published only by :func:`publish_team_runtime` (``lop update`` and
#: ``lop permissions setup``, behind a recorded opt-in); ``adhoc`` is what the
#: session path plants and never touches a keychain.
FLAVOR_TEAM = "team"
FLAVOR_ADHOC = "adhoc"
_FLAVORS = (FLAVOR_TEAM, FLAVOR_ADHOC)

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
#: The session host (``session/runtime/host``): the launchd-owned parent every
#: host-spawned session runtime descends from. Its own row, not ``[wakes]``'s,
#: because an operator reading ``ps`` after a quit has to be able to tell "the
#: runtimes' parent" from "the wake supervisor" at a glance.
LABEL_SESSION_HOST = "{brand} [session host]"
LABEL_TUNNEL = "{brand} [tunnel]"
LABEL_SERVE = "{brand} [serve] port={port}"
#: The Team Bridge connector LaunchAgent: one per user, no per-instance field.
LABEL_BRIDGE_CONNECTOR = "{brand} [bridge connector]"
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

#: Sidecar beside a bundle-flavor image: ``<size> <mtime_ns> <flavor>
#: <identifier>`` of the SIGNED executable. The interpreter's size is no longer
#: an integrity oracle for a bundle image — signing rewrites the signature
#: blob (measured: an ad-hoc re-sign took a 49968-byte image to 49840) — so
#: the bundle records its own final stat at publish time and health compares
#: against that.
_IMAGE_SIDECAR = "image"

#: Beside a LEGACY ``<key>/`` tree: "the bundle flavor could not be built for
#: this key". Without it a host whose bundle plant fails (no ``codesign``, a
#: refusing filesystem) would retry the clone + sign on every startup, because
#: the expected image would always be the unpublished bundle. Retried after
#: :data:`_BUNDLE_RETRY_S`, so a transient failure (a full disk) heals.
_BUNDLE_UNAVAILABLE = "bundle-unavailable"
_BUNDLE_RETRY_S = 24 * 3600

#: Absolute, never PATH-resolved: a ``codesign`` earlier on a session's PATH
#: must not be what signs the shared runtime. Tests replace this attribute
#: with a fake that records its argv.
_CODESIGN = "/usr/bin/codesign"

#: The interpreter's only ``LC_RPATH`` and its in-bundle replacement. From
#: ``<key>/<bundle>/Contents/MacOS/`` three levels up is ``<key>/``, whose
#: ``lib`` links the build's own ``lib``. 25 bytes + NUL fits the slot the
#: 23-byte original occupies (load commands are 8-byte aligned, so that slot is
#: 32 bytes); ``install_name_tool -add_rpath`` fails for lack of header pad,
#: an in-place rewrite does not (measured). The rewrite is done here in Python
#: rather than with ``install_name_tool`` because that tool is an Xcode
#: Command Line Tools shim on a stock Mac — invoking it without the CLT
#: installed raises the "install developer tools" dialog, which a startup path
#: must never do.
_RPATH_OLD = b"@executable_path/../lib"
_RPATH_NEW = b"@loader_path/../../../lib"
_LC_RPATH = 0x8000001C


def _bundle_identifier(flavor: str) -> str:
    """The ``CFBundleIdentifier`` for ``flavor``; ad-hoc NEVER gets the app's id."""
    return APP_BUNDLE_ID if flavor == FLAVOR_TEAM else RUNTIME_BUNDLE_ID


def _flavor_dir(root: Path, key: str, flavor: str) -> Path:
    return root / f"{key}-{flavor}"


def _bundle_image(root: Path, key: str, flavor: str) -> Path:
    """``<root>/<key>-<flavor>/Local Operator.lopruntime/Contents/MacOS/Local Operator``."""
    return _flavor_dir(root, key, flavor) / RUNTIME_BUNDLE_DIRNAME / "Contents" / "MacOS" / BRAND


def _legacy_image(root: Path, key: str) -> Path:
    """``<root>/<key>/bin/Local Operator`` — the pre-bundle, PATH-keyed shape."""
    return root / key / "bin" / BRAND


def _is_bundle_image(image: Path) -> bool:
    """Does ``image`` sit at ``<home>/<bundle>/Contents/MacOS/<BRAND>``?"""
    parents = image.parents
    return (
        len(parents) > 3
        and image.name == BRAND
        and parents[0].name == "MacOS"
        and parents[1].name == "Contents"
        and parents[2].name == RUNTIME_BUNDLE_DIRNAME
    )


def _image_home(image: Path) -> Path:
    """The directory holding an image's ``lib``/``source``/``image`` sidecars.

    Depth 4 for a bundle image (``<home>/<bundle>/Contents/MacOS/<BRAND>``),
    depth 2 for every other shape (``<home>/bin/<BRAND>``, and a venv's own
    ``bin/`` for the hardlink rung) — the rule this module always had.
    """
    return image.parents[3] if _is_bundle_image(image) else image.parent.parent


def _bundle_flavor(image: Path) -> str | None:
    """The flavor recorded in a bundle image's sidecar, or ``None``."""
    try:
        fields = (_image_home(image) / _IMAGE_SIDECAR).read_text(encoding="utf-8").split()
    except (OSError, ValueError):
        return None
    return fields[2] if len(fields) >= 4 and fields[2] in _FLAVORS else None


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
    # Depth 4 for a bundle image, 2 for the legacy ``<key>/bin`` shape: the
    # marker lives in the build-key directory either way.
    home = _image_home(image)
    if image.name != BRAND or not _under(published, home):
        return None
    try:
        recorded = Path((home / _SOURCE_MARKER).read_text(encoding="utf-8").strip())
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
    """The image a venv link for ``real`` SHOULD name, whether or not it is planted.

    The selection, in order:

    1. a healthy ``team`` bundle (``com.local-operator``, the app's own row) —
       published only by :func:`publish_team_runtime`, so its presence IS the
       recorded opt-in having taken effect;
    2. the ``adhoc`` bundle (``com.local-operator.runtime``) — healthy or
       not yet planted, it is what the session-path plant builds;
    3. the legacy ``<key>/bin`` image, ONLY when a recent bundle attempt for
       this key recorded that the bundle cannot be built here. Without that
       marker a legacy link is stale, which is how an existing venv link
       repoints at the bundle on its next start.

    Steady-state cost on top of the key: one ``lexists`` for the team tree
    (absent on almost every host) and the adhoc health check.
    """
    root = runtime_root()
    if root is None:
        return None
    key = _build_key(real)
    team = _bundle_image(root, key, FLAVOR_TEAM)
    if os.path.lexists(_flavor_dir(root, key, FLAVOR_TEAM)) and _healthy_image(team, real):
        return team
    adhoc = _bundle_image(root, key, FLAVOR_ADHOC)
    if _healthy_image(adhoc, real):
        return adhoc
    legacy = _legacy_image(root, key)
    if _bundle_recently_unavailable(root, key) and _healthy_image(legacy, real):
        return legacy
    return adhoc


def _bundle_recently_unavailable(root: Path, key: str) -> bool:
    """Did a bundle plant for ``key`` fail within :data:`_BUNDLE_RETRY_S`?"""
    import time

    try:
        stamp = (root / key / _BUNDLE_UNAVAILABLE).stat().st_mtime
    except OSError:
        return False
    return time.time() - stamp < _BUNDLE_RETRY_S


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
        # A bundle image's ``lib`` sits beside the BUNDLE, in the build-key
        # directory (``@loader_path/../../../lib``), never inside it: an
        # out-of-bundle symlink in ``Contents`` fails ``codesign --strict``
        # ("invalid destination for symbolic link in bundle", measured).
        lib = _image_home(image) / "lib"
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

    A BUNDLE image answers to its own sidecar instead of ``real``: signing
    rewrote its bytes, so the interpreter's size and mtime are no longer what
    an intact image has. A bundle with no readable sidecar is unhealthy — the
    sidecar is the last thing a plant writes before publishing, so its absence
    means a tree this module did not finish.
    """
    if _is_bundle_image(resolved):
        try:
            fields = (_image_home(resolved) / _IMAGE_SIDECAR).read_text(encoding="utf-8").split()
            size, mtime_ns = int(fields[0]), int(fields[1])
        except (IndexError, ValueError):
            return False
        image_stat = resolved.stat()
        if image_stat.st_size != size:
            return False
        if abs(image_stat.st_mtime_ns - mtime_ns) >= _MTIME_SLACK_NS:
            return False
    elif real is not None:
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
    try:
        if root is None:
            return None
        key = _build_key(real)
        team = _bundle_image(root, key, FLAVOR_TEAM)
        if os.path.lexists(_flavor_dir(root, key, FLAVOR_TEAM)) and _healthy_image(team, real):
            # Published by ``publish_team_runtime`` after the operator opted in;
            # it shares the app's row, so it outranks anything planted here.
            return team
        image = _publish_tree(
            real,
            root,
            _flavor_dir(root, key, FLAVOR_ADHOC).name,
            _bundle_image(root, key, FLAVOR_ADHOC),
            lambda staging, base: _populate_bundle(staging, real, base, FLAVOR_ADHOC, _sign_adhoc),
        )
        if image is not None:
            return image
        # TODAY'S SHAPE IS THE FALLBACK, never a failed launch: the bundle could
        # not be built (no ``codesign``, an unsigned-able filesystem, a Mach-O
        # whose rpath we cannot place), so the PATH-keyed image is published
        # exactly as before and the key is marked so the next start does not
        # pay for the same failed attempt.
        legacy = _publish_tree(
            real,
            root,
            key,
            _legacy_image(root, key),
            lambda staging, base: _populate_legacy(staging, real, base),
        )
        if legacy is not None:
            _mark_bundle_unavailable(root, key)
        return legacy
    except Exception:  # noqa: BLE001 — rung 2 exists for exactly this
        logger.debug("runtime image unavailable", exc_info=True)
        return None


def _mark_bundle_unavailable(root: Path, key: str) -> None:
    try:
        (root / key / _BUNDLE_UNAVAILABLE).write_text("", encoding="utf-8")
    except OSError:
        logger.debug("could not record a bundle failure", exc_info=True)


def _populate_legacy(staging: Path, real: Path, base: Path) -> None:
    """The pre-bundle tree: ``bin/<BRAND>`` clone plus ``lib``/``include`` links."""
    # Darwin-only by construction (every caller is behind a darwin check), but
    # restated here so the static cross-platform scan sees the guard at the
    # ``os.symlink`` calls below: raising lands in ``_publish_tree``'s failure
    # path, which never publishes a partial tree.
    if sys.platform != "darwin":
        raise OSError("runtime trees are built on macOS only")
    (staging / "bin").mkdir(parents=True)
    _clone_or_copy(real, staging / "bin" / BRAND)
    _write_source_marker(staging, real)
    # The stdlib and the dylib are both resolved relative to the IMAGE
    # directory, so both must resolve back into the interpreter's own
    # tree. ``include`` is only needed by a build that ships headers.
    os.symlink(base / "lib", staging / "lib")
    if (base / "include").is_dir():
        os.symlink(base / "include", staging / "include")


def _publish_tree(
    real: Path,
    root: Path,
    dirname: str,
    image: Path,
    populate,  # Callable[[Path, Path], None]; raises on any failure
) -> Path | None:
    """Build ``root/dirname`` in staging with ``populate`` and publish it by rename.

    The one publish protocol for every runtime tree shape (legacy and both
    bundle flavors): a private staging directory, one ``os.rename``, a loser of
    the race adopts the winner, a CORRUPT published tree is swapped aside.
    ``image`` is the executable inside the published tree.
    """
    import shutil

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
    staging = root / f"{dirname}.tmp.{os.getpid()}.{_temp_thread_token()}"
    try:
        staging.mkdir(parents=True)
        populate(staging, base)
    except Exception:  # noqa: BLE001 — a failed build (OSError, codesign) is never published
        # A FAILED BUILD IS NOT A PUBLISHED IMAGE. This block used to fall
        # through to the corrupt-directory branch below, which ``rename``d a
        # staging tree holding an empty ``bin/`` (or a half-written clone,
        # on ENOSPC) onto the build key. Nothing could ever call it healthy,
        # but it published a key a reader can see and sent the next planter
        # down the corrupt-repair path for a directory that never held an
        # image. Remove it and answer rung 2.
        logger.debug("runtime tree %s not built", dirname, exc_info=True)
        shutil.rmtree(staging, ignore_errors=True)
        return image if _healthy_image(image, real) else None
    try:
        os.rename(staging, root / dirname)
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
        return _replace_corrupt(root, dirname, staging, image, real)
    return _settled_image(image, real)


def _rewrite_rpath(exe: Path) -> bool:
    """Rewrite ``exe``'s ``LC_RPATH`` in place to reach ``lib`` beside the bundle.

    True when every Mach-O slice had the expected ``@executable_path/../lib``
    and the replacement fit its load command; False (nothing written) for any
    other shape, which sends the caller to the in-bundle ``Contents/lib``
    fallback. Writes only into ``exe``, which is a CLONE this plant created
    (its own inode, checked), never the interpreter — the 2026-09-18 incident
    (a write through a shared name truncated a machine-wide interpreter) is why
    the link count is asserted before opening it for writing.
    """
    import struct

    if os.stat(exe).st_nlink != 1:
        return False
    data = bytearray(exe.read_bytes())
    slices: list[int] = []
    magic = bytes(data[:4])
    if magic in (bytes.fromhex("cafebabe"), bytes.fromhex("cafebabf")):
        wide = magic == bytes.fromhex("cafebabf")
        (count,) = struct.unpack_from(">I", data, 4)
        entry = 32 if wide else 20
        for index in range(count):
            at = 8 + index * entry
            offset = struct.unpack_from(">Q" if wide else ">I", data, at + 8)[0]
            slices.append(int(offset))
    else:
        slices.append(0)
    edits: list[tuple[int, bytes]] = []
    for base in slices:
        head = bytes(data[base : base + 4])
        if head == bytes.fromhex("cffaedfe"):
            endian, header = "<", 32
        elif head == bytes.fromhex("feedfacf"):
            endian, header = ">", 32
        elif head == bytes.fromhex("cefaedfe"):
            endian, header = "<", 28
        elif head == bytes.fromhex("feedface"):
            endian, header = ">", 28
        else:
            return False
        (ncmds,) = struct.unpack_from(f"{endian}I", data, base + 16)
        cursor = base + header
        found = False
        for _ in range(ncmds):
            cmd, size = struct.unpack_from(f"{endian}II", data, cursor)
            if cmd == _LC_RPATH:
                (str_off,) = struct.unpack_from(f"{endian}I", data, cursor + 8)
                start, stop = cursor + str_off, cursor + size
                current = bytes(data[start:stop]).split(b"\0", 1)[0]
                if current != _RPATH_OLD:
                    return False  # an rpath we did not expect: do not guess
                if len(_RPATH_NEW) + 1 > stop - start:
                    return False  # does not fit the slot: the fallback shape
                edits.append((start, _RPATH_NEW.ljust(stop - start, b"\0")))
                found = True
            if size <= 0:
                return False
            cursor += size
        if not found:
            return False
    for start, payload in edits:
        data[start : start + len(payload)] = payload
    with open(exe, "r+b") as handle:
        handle.write(data)
    return True


def _write_info_plist(contents: Path, flavor: str) -> None:
    """The bundle's Info.plist; ``CFBundleIdentifier`` from :func:`_bundle_identifier`.

    ``CFBundleShortVersionString`` is ``0`` so LaunchServices would rank this
    below the real app even if an indexer ever registered it, and
    ``LSBackgroundOnly``/``LSUIElement`` keep it out of the Dock and the
    app switcher. ``CFBundleExecutable`` is the brand, so ``p_comm`` and
    Activity Monitor keep saying "Local Operator".
    """
    import plistlib

    display = BRAND if flavor == FLAVOR_TEAM else RUNTIME_DISPLAY_NAME
    info = {
        "CFBundleIdentifier": _bundle_identifier(flavor),
        "CFBundleExecutable": BRAND,
        "CFBundleName": display,
        "CFBundleDisplayName": display,
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": "0",
        "CFBundleVersion": "0",
        "CFBundleInfoDictionaryVersion": "6.0",
        "LSUIElement": True,
        "LSBackgroundOnly": True,
    }
    with open(contents / "Info.plist", "wb") as handle:
        plistlib.dump(info, handle)


def _codesign_run(*args: str) -> tuple[bool, str]:
    """Run ``codesign`` with ``args``: ``(ok, last stderr line)``. Bounded, stdin closed.

    The stderr line is kept because ``lop permissions setup`` shows it when a
    Developer ID signature fails: without it the operator only learned that
    signing "did not finish" and the cause lived in a debug log (UX round 1, U5).
    """
    import subprocess

    try:
        result = subprocess.run(
            [_CODESIGN, *args],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("codesign could not run", exc_info=True)
        return False, str(exc)
    if result.returncode != 0:
        logger.debug("codesign %s failed: %s", args[:-1], result.stderr[-400:])
        lines = [
            line.strip()
            for line in (result.stderr or b"").decode("utf-8", "replace").splitlines()
            if line.strip()
        ]
        return False, (lines[-1] if lines else f"codesign exited {result.returncode}")
    return True, ""


def _codesign(*args: str) -> bool:
    """Run ``codesign`` with ``args``; True on exit 0. Bounded, stdin closed."""
    return _codesign_run(*args)[0]


def _sign_adhoc(bundle: Path) -> bool:
    """Ad-hoc sign: ``--sign -`` NEVER reads a keychain, so it can run anywhere.

    0.11 s measured. This is the only signing the session path ever does.
    """
    return _codesign("--force", "--sign", "-", "--identifier", RUNTIME_BUNDLE_ID, str(bundle))


def _populate_bundle(staging: Path, real: Path, base: Path, flavor: str, sign) -> None:
    """Build ``<staging>/Local Operator.lopruntime`` signed as ``flavor``; raise on failure.

    Order matters: every byte change (rpath, plist) happens BEFORE signing,
    and the sidecar is written last from the signed image's own stat, so a
    tree is healthy only once it is complete.
    """
    # Darwin-only by construction (every caller is behind a darwin check), but
    # restated here so the static cross-platform scan sees the guard at the
    # ``os.symlink`` calls below: raising lands in ``_publish_tree``'s failure
    # path, which never publishes a partial tree.
    if sys.platform != "darwin":
        raise OSError("runtime trees are built on macOS only")
    _write_source_marker(staging, real)
    # ``lib`` beside the bundle, never inside it (``--strict``, see
    # ``_RPATH_NEW``); ``include`` as the legacy shape has it.
    os.symlink(base / "lib", staging / "lib")
    if (base / "include").is_dir():
        os.symlink(base / "include", staging / "include")
    bundle = staging / RUNTIME_BUNDLE_DIRNAME
    contents = bundle / "Contents"
    (contents / "MacOS").mkdir(parents=True)
    exe = contents / "MacOS" / BRAND
    _clone_or_copy(real, exe)
    if not _rewrite_rpath(exe):
        if flavor == FLAVOR_TEAM:
            # The fallback below fails ``codesign --strict``, and the team
            # image is verified strictly against the app's requirement.
            raise OSError("rpath rewrite did not fit; no team image for this build")
        # Still keyed by bundle id (measured), just not ``--strict`` clean.
        os.symlink(base / "lib", contents / "lib")
    _write_info_plist(contents, flavor)
    if not sign(bundle):
        raise OSError(f"codesign refused the {flavor} runtime bundle")
    stat = exe.stat()
    (staging / _IMAGE_SIDECAR).write_text(
        f"{stat.st_size} {stat.st_mtime_ns} {flavor} {_bundle_identifier(flavor)}\n",
        encoding="utf-8",
    )


#: The opt-in record ``lop permissions setup`` writes, under the config dir.
#: ``{"identity": <SHA-1>, "team": <TeamIdentifier>, "always_allow_confirmed":
#: bool}``. Its existence IS the consent to team signing: nothing else writes
#: it, and :func:`publish_team_runtime` is a no-op without it.
SIGNING_OPTIN_FILENAME = "runtime-signing.json"

#: Where the desktop app may be installed; the first that exists is the one
#: whose TeamIdentifier and designated requirement the team image must meet.
_APP_CANDIDATES = ("/Applications/Local Operator.app", "~/Applications/Local Operator.app")


def _passwd_home() -> Path | None:
    try:
        import pwd

        home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    except Exception:  # noqa: BLE001 — no passwd entry: nowhere legitimate to sign for
        return None
    return home if home.is_absolute() else None


def team_signing_refusal() -> str | None:
    """Why team signing may NOT run in this process, or ``None`` when it may.

    Machine codes (``not-darwin``, ``isolated-home``, ``runtime-root-override``,
    ``config-dir-override``), never prose: the CLI renders them. Every guard
    exists to keep a Developer ID signature — and the keychain dialog it can
    raise — out of tests, sandboxes and rigs: an isolated ``HOME`` has no
    login keychain, and macOS answers a keychain lookup there by asking the
    operator to CREATE one (the measured 2026-09-18 rig incident).
    """
    if sys.platform != "darwin":
        return "not-darwin"
    home = _passwd_home()
    if home is None or os.path.realpath(os.environ.get("HOME", "")) != os.path.realpath(home):
        return "isolated-home"
    if os.environ.get("LOCAL_OPERATOR_RUNTIME_ROOT"):
        return "runtime-root-override"
    config = os.environ.get("LOCAL_OPERATOR_CONFIG_DIR")
    # Some launchers export the DEFAULT config dir explicitly (``paths.py``);
    # only a relocated one is an isolated run.
    if config and os.path.realpath(config) != os.path.realpath(home / ".local-operator"):
        return "config-dir-override"
    return None


def _optin_path() -> Path | None:
    home = _passwd_home()
    return None if home is None else home / ".local-operator" / SIGNING_OPTIN_FILENAME


def read_signing_optin() -> dict[str, object] | None:
    """The recorded team-signing opt-in, validated, or ``None``."""
    import json
    import re

    path = _optin_path()
    if path is None:
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict):
        return None
    identity, team = record.get("identity"), record.get("team")
    # A SHA-1 and a 10-character team id, nothing else: the identity is
    # passed to codesign as an argument and must never be a name that could
    # match some other certificate in the keychain.
    if not isinstance(identity, str) or not re.fullmatch(r"[0-9A-F]{40}", identity):
        return None
    if not isinstance(team, str) or not re.fullmatch(r"[A-Z0-9]{10}", team):
        return None
    return record


def write_signing_optin(identity: str, team: str, *, always_allow_confirmed: bool) -> Path:
    """Record the opt-in. Called ONLY by ``lop permissions setup``; raises on failure."""
    import json

    path = _optin_path()
    if path is None:
        raise OSError("no passwd home")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "identity": identity.upper(),
        "team": team,
        "always_allow_confirmed": bool(always_allow_confirmed),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def installed_app() -> Path | None:
    """The installed desktop app bundle, or ``None``."""
    for candidate in _APP_CANDIDATES:
        path = Path(candidate).expanduser()
        if path.is_dir():
            return path
    return None


def _codesign_output(*args: str) -> tuple[int, str]:
    """``codesign`` with ``args``; ``(returncode, stdout+stderr)``. Never raises."""
    import subprocess

    try:
        result = subprocess.run(
            [_CODESIGN, *args],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return -1, ""
    return result.returncode, (result.stdout or "") + (result.stderr or "")


def code_identity(path: Path) -> dict[str, str]:
    """``Identifier``/``TeamIdentifier``/``Signature`` of signed code at ``path``.

    ``codesign -d`` reads the embedded signature only; it never touches a
    keychain, so it is safe on any path, including ``lop permissions status``.
    """
    rc, text = _codesign_output("-d", "--verbose=2", str(path))
    found: dict[str, str] = {}
    if rc != 0:
        return found
    for line in text.splitlines():
        name, sep, value = line.partition("=")
        if sep and name in {"Identifier", "TeamIdentifier", "Signature", "Authority"}:
            found.setdefault(name, value.strip())
    return found


def app_requirement(app: Path) -> str | None:
    """The app's designated requirement text (``codesign -d -r-``), or ``None``."""
    rc, text = _codesign_output("-d", "-r-", str(app))
    if rc != 0:
        return None
    for line in text.splitlines():
        if line.startswith("designated => "):
            return line[len("designated => ") :].strip() or None
    return None


class TeamPublish:
    """What :func:`publish_team_runtime_result` did, for callers that report it.

    ``image`` is the team image (or ``None``). ``signed`` is True only when
    THIS call ran a Developer ID signature — setup asks the "Always Allow"
    question only then, because an image that already existed exercised no
    keychain ACL (review round 1, R4). ``reason`` is a machine code for why no
    image came back (``None`` on success); ``detail`` carries codesign's own
    error line for ``signing-failed``.
    """

    # A plain class, not ``typing.NamedTuple``: this module sits on the
    # interpreter's startup path and imports nothing beyond the stdlib core.
    __slots__ = ("image", "signed", "reason", "detail")

    def __init__(
        self,
        image: Path | None,
        *,
        signed: bool = False,
        reason: str | None = None,
        detail: str = "",
    ) -> None:
        self.image = image
        self.signed = signed
        self.reason = reason
        self.detail = detail

    def __repr__(self) -> str:
        return (
            f"TeamPublish(image={self.image!r}, signed={self.signed!r}, "
            f"reason={self.reason!r}, detail={self.detail!r})"
        )


def publish_team_runtime(
    interpreter: str | os.PathLike[str], *, interactive: bool = False, allow_prompt: bool = False
) -> Path | None:
    """Publish the TEAM-signed bundle (``com.local-operator``) for ``interpreter``'s build.

    WHY: an image signed by the app's own team whose identifier is the app's
    satisfies the app's designated requirement, and tccd then answers the
    runtime with the APP's grant — one row shared by the desktop app and every
    runtime. Measured on the operator's host 2026-11-09 (§6 consent check): the
    disclaimed probe's tccd request read ``requesting={identifier=
    com.local-operator, binary_path=…/Local Operator.lopruntime/Contents/MacOS/
    Local Operator}`` with no responsible app, ``subject=com.local-operator``,
    ``authValue=2``. Runtimes keep their own process identity (disclaimed or
    launchd-owned), so sharing the grant adds no kill path from the UI.

    NEVER on the session path. Only ``lop update`` (before the flip) and
    ``lop permissions setup`` call this, because a Developer ID signature may
    raise the keychain dialog. ``interactive`` is True only from setup.
    Otherwise it signs only when the operator recorded that they chose
    "Always Allow", or when ``allow_prompt`` is True AND stdin is a terminal:
    ``allow_prompt`` is passed ONLY by the ``lop update`` CLI entry, because a
    TTY alone is not consent — the TUI's ``/update`` runs with a TTY in the
    middle of a live session (review round 1, R2), and a keychain dialog over
    it, with codesign blocking up to 120 s, is exactly what this gate exists
    to prevent. The signed bundle is verified with ``codesign --verify --strict
    -R=<app DR>`` BEFORE it is published, so an image that would not share the
    app's grant is never selected. Never raises; ``None`` means runtimes keep
    the ad-hoc "Local Operator Runtime" flavor.
    """
    return publish_team_runtime_result(
        interpreter, interactive=interactive, allow_prompt=allow_prompt
    ).image


def publish_team_runtime_result(
    interpreter: str | os.PathLike[str], *, interactive: bool = False, allow_prompt: bool = False
) -> TeamPublish:
    """:func:`publish_team_runtime`, reporting whether it signed and why not.

    Reason codes: ``refused:<team_signing_refusal code>``, ``not-opted-in``,
    ``needs-always-allow`` (no recorded "Always Allow" and this caller may not
    prompt), ``no-terminal``, ``unsupported-interpreter``, ``app-missing``,
    ``team-mismatch``, ``no-requirement``, ``signing-failed``, ``error``.
    Never raises.
    """
    try:
        refusal = team_signing_refusal()
        if refusal is not None:
            return TeamPublish(None, reason=f"refused:{refusal}")
        record = read_signing_optin()
        if record is None:
            return TeamPublish(None, reason="not-opted-in")
        if not interactive and not record.get("always_allow_confirmed"):
            if not allow_prompt:
                return TeamPublish(None, reason="needs-always-allow")
            try:
                if not sys.stdin.isatty():
                    return TeamPublish(None, reason="no-terminal")
            except (AttributeError, ValueError):
                return TeamPublish(None, reason="no-terminal")
        real = _real_interpreter_for(interpreter)
        root = runtime_root()
        if real is None or root is None or _is_framework_build(real):
            return TeamPublish(None, reason="unsupported-interpreter")
        key = _build_key(real)
        image = _bundle_image(root, key, FLAVOR_TEAM)
        if os.path.lexists(_flavor_dir(root, key, FLAVOR_TEAM)) and _healthy_image(image, real):
            return TeamPublish(image)
        app = installed_app()
        if app is None:
            return TeamPublish(None, reason="app-missing")
        if code_identity(app).get("TeamIdentifier") != record["team"]:
            return TeamPublish(None, reason="team-mismatch")
        requirement = app_requirement(app)
        if requirement is None:
            return TeamPublish(None, reason="no-requirement")
        identity = str(record["identity"])
        attempts: list[str] = []

        def sign(bundle: Path) -> bool:
            ok, error = _codesign_run(
                "--force", "--sign", identity, "--identifier", APP_BUNDLE_ID, str(bundle)
            )
            attempts.append(error)
            if not ok:
                return False
            ok, error = _codesign_run("--verify", "--strict", f"-R={requirement}", str(bundle))
            if not ok:
                attempts.append(error)
            return ok

        published = _publish_tree(
            real,
            root,
            _flavor_dir(root, key, FLAVOR_TEAM).name,
            image,
            lambda staging, base: _populate_bundle(staging, real, base, FLAVOR_TEAM, sign),
        )
        # ``attempts`` non-empty means this call reached the Developer ID
        # signature (a concurrent publisher that won the rename still means
        # our codesign ran and any keychain dialog was answered).
        if published is None:
            failures = [error for error in attempts if error]
            return TeamPublish(
                None,
                signed=bool(attempts),
                reason="signing-failed",
                detail=failures[-1] if failures else "",
            )
        return TeamPublish(published, signed=bool(attempts))
    except Exception:  # noqa: BLE001 — an identity never fails an update
        logger.debug("team runtime image not published", exc_info=True)
        return TeamPublish(None, reason="error")


def runtime_identity(interpreter: str | os.PathLike[str] | None = None) -> dict[str, object]:
    """Read-only: which image and identity runtimes of this build resolve to.

    For ``lop permissions status``. No plant, no signing, no keychain: stats,
    one sidecar read, and ``codesign -d`` on the selected image.
    """
    info: dict[str, object] = {"platform": sys.platform}
    try:
        real = _real_interpreter_for(interpreter or sys.executable)
        root = runtime_root()
        info["interpreter"] = str(real) if real else None
        if real is None or root is None:
            return info
        image = runtime_image(real)
        info["image"] = str(image) if image else None
        if image is None or not _healthy_image(image, real):
            info["planted"] = False
            return info
        info["planted"] = True
        info["flavor"] = _bundle_flavor(image) if _is_bundle_image(image) else "legacy"
        if _is_bundle_image(image):
            info.update({k.lower(): v for k, v in code_identity(image.parents[2]).items()})
    except Exception:  # noqa: BLE001 — a status read never fails
        logger.debug("runtime identity unreadable", exc_info=True)
    return info


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
        # The image's place inside the tree, whichever shape it is (legacy
        # ``bin/`` or a bundle's ``Contents/MacOS``).
        if _healthy_image(aside / image.relative_to(published), real):
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


def is_own_launch() -> bool:
    """True only when this process is really a launch of this product.

    THE GUARD THAT KEEPS A RE-EXEC FROM EATING ITS CALLER. ``cli.main()`` is not
    only an entry point: the test suite calls it in-process (``assert main()
    == 7``), and so can any embedder. An unguarded re-exec there does not
    "rename the process" -- it REPLACES the running pytest with a fresh
    interpreter, which silently truncated a 98-test file at 57% with exit 0.
    Reproduced before this guard existed.

    IT ASKS WHAT IS RUNNING, NOT WHAT THE COMMAND LINE LOOKS LIKE. The answer comes
    from ``__main__``: a ``-m`` launch has ``__main__.__spec__.name`` set to the
    module (``local_operator.cli``, ``session.runtime.process``, the eval worker --
    every module this product spawns is inside the package), and a console script
    has ``__main__.__file__`` naming the script (``bin/lop``). Reading the command
    line instead was a losing game: it has to re-derive CPython's flag grammar, and
    every flag it did not know (``-O``, ``-Ou``, ``-q``, ``-b``, ``-Es``, ``-sP``,
    an attached ``-mlocal_operator.cli``) answered "not our launch" -- which for
    ``server.desktop.adopt_spawn_bearer`` means no pipe re-exec, so the daemon kept
    the desktop bearer in its exec-time block for its whole life (#3022 code review
    rounds 5 and 6). Interpreter flags cannot change what ``__main__`` is, and the
    re-exec rebuilds its argv verbatim from ``sys.orig_argv``, so nothing here needs
    to understand them.

    THE ``-c`` SHAPE IS A DELIBERATE NON-GOAL. The desktop app
    (``~/local-operator-ui``, ``src/main/backend/owned-serve-launch.ts``) starts its
    managed backend as ``python3 -c "from local_operator.cli import main; main()"
    serve --port <n>`` and then verifies that the process it spawned is the one
    serving by asking the SAME ``-c`` string to report its own ``sys.executable``; a
    mismatch raises "its base interpreter ... is not the serving process" and the
    app refuses to start its backend. A re-exec through the branded hardlink makes
    ``sys.executable`` become ``<prefix>/bin/Local Operator``, so every one of those
    probes would fail and the desktop app would be dead on every machine. A ``-c``
    process has no ``__main__.__file__`` and no spec, so it answers False here --
    the bearer channel handles that app shape itself
    (``server.desktop._DESKTOP_SERVE_PROGRAM``), which is branding, not this guard.

    Anything else -- pytest, an embedder, ``-m pip``, a foreign script, a REPL --
    answers False, and the direction that matters: a false negative costs a name or
    one re-exec, a false positive replaces a caller's process.
    """
    try:
        main = sys.modules.get("__main__")
        spec = getattr(main, "__spec__", None)
        name = getattr(spec, "name", None) if spec is not None else None
        if name:
            # Bounded at the dot: a similarly-named distribution
            # (``local_operators``) cannot pass, and neither can a sibling module
            # that is not inside this package.
            return name == "local_operator" or name.startswith("local_operator.")
        filename = getattr(main, "__file__", None)
        if not filename:
            return False
        return os.path.basename(filename) in _LAUNCHER_NAMES
    except Exception:  # noqa: BLE001 - a guard that raises must answer "no"
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
