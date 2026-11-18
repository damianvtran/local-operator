"""A machine-wide lease that limits how many heavy builds or tests run at once per host.

WHY THIS EXISTS
---------------
On 2026-11-16 at 02:06Z, lop-remote-node-2 (15.7 GB RAM) hit a global OOM:
several sessions ran parallel esbuild, node-22 and chrome builds at once, and the
kernel killed seven processes, among them a node-22 at 1.47 GB RSS. The fleet
rule "one heavy job at a time, memory-capped" existed only as guidance, and
agents running in parallel sessions do not coordinate on guidance.

The per-command memory guard (:mod:`local_operator.memory_guard`) could not
prevent it: it sizes and polices ONE command's process group, so every build
stayed under its own ceiling while their SUM exhausted the machine. What was
missing is a node-wide concurrency limit, and this module is that limit.

HOW IT WORKS
------------
A cross-process counting semaphore made of N slot files,
``<config_dir>/run/heavy-jobs/slot-<i>.lock``. Taking a slot is a non-blocking
``flock(LOCK_EX)`` on one of them; the lock is held by an open file descriptor
in the harness process for the life of the command, and the spawned command
inherits a duplicate of that descriptor (``Lease.fileno`` via ``pass_fds``).
Constraints this relies on:

* ``flock`` locks belong to the OPEN FILE DESCRIPTION, not the process, so two
  sessions inside one harness process (subagents share the pid) still exclude
  each other: each ``open`` makes its own description.
* The kernel drops the lock when the LAST descriptor for that description
  closes, including when the holder dies by SIGKILL or OOM. Because the
  command holds a duplicate, a harness killed mid-build does NOT free the slot
  while its (detached, ``start_new_session``) build keeps running and keeping
  its memory: the slot frees when the build exits too (PR #4104 QA Q1, where
  the next queued build started beside the orphan). The lock, not the JSON
  record, is the truth.
* On a NORMAL end the harness unlocks explicitly (``LOCK_UN`` on its own
  descriptor, which releases the lock for every duplicate) as soon as the
  command's shell exits, so a daemon the build left behind holding an
  inherited copy (an nx/turbo daemon) cannot pin the slot. The only residual:
  such a daemon outliving a harness that died outright keeps the slot until it
  exits; ``fuser <config_dir>/run/heavy-jobs/slot-N.lock`` finds it.
* A command that self-backgrounds (``(build &)``, ``nohup build &``) keeps
  the slot only until its shell exits, because of that same ``LOCK_UN``. The
  sessions guide forbids escaping the queue that way; the lease does not try
  to chase a process that deliberately left its process group.
* ``flock`` exists on Linux and macOS. On Windows ``fcntl`` is absent and the
  lease is DISABLED (:func:`supported` is False): there is no fleet of
  Windows build nodes to protect, and a half-working lease would be worse than
  none.
* ``flock`` is not reliable on network filesystems; the config dir is local on
  every supported install.

Each holder writes ``slot-<i>.json`` beside its lock (session id, harness pid,
command pgid, command preview, start time) so a waiter can say WHO it is
queued behind. The record is advisory: a record whose ``pid`` is dead is
stale and ignored, because only the flock can prove a slot is held.

USING AND DISABLING IT
----------------------
The ``bash`` tool classifies each command with :func:`is_heavy` and, for a
heavy one, acquires a slot before spawning (see ``execute_bash``). A
non-heavy command never touches the lease. The kill switch is the config key
``bash.heavy_jobs.enabled`` (LIVE, read per heavy command; default on). Two
node tuning knobs are environment overrides: ``LOCAL_OPERATOR_HEAVY_JOB_SLOTS``
(0 = auto, see :func:`auto_slot_count`) and ``LOCAL_OPERATOR_HEAVY_JOB_WAIT_S``
(how long a heavy command queues before it is refused, default 900, separate
from and never capped by the command's own ``timeout``).
``lop services status`` prints the current holders.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import shlex
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

try:  # POSIX only; Windows has no flock and runs without the lease.
    import fcntl
except ImportError:  # pragma: no cover - exercised on Windows hosts
    fcntl = None  # type: ignore[assignment]

#: Where the one config key lives under ``values``; spelled once and pinned to
#: the ``settings_io`` row by a test, like ``memory_guard.BASH_MEMORY_*_PATH``.
HEAVY_JOBS_ENABLED_PATH: tuple[str, ...] = ("bash", "heavy_jobs", "enabled")

#: On by default: the failure it prevents (a node-wide OOM killing other
#: sessions' work) is far worse than the one it risks (a build waits its turn).
HEAVY_JOBS_ENABLED_DEFAULT = True

#: The slot count and the queue wait are ENVIRONMENT overrides, not config keys,
#: following ``memory_guard.ADMISSION_FLOOR_ENV``. Why: every config key must be a
#: localized ``/settings`` row (AGENTS.md "Adding a configuration key"), and these
#: two are node-operator tuning knobs nobody needs to discover in a settings
#: screen — the on/off switch is the one the operator does. Constraint: the
#: environment is the HARNESS process's, read per heavy command; set it where
#: lop is launched (shell profile, systemd unit) so every session on the node
#: agrees, because two processes that disagree on the slot count share only the
#: slots both of them count.
HEAVY_JOBS_SLOTS_ENV = "LOCAL_OPERATOR_HEAVY_JOB_SLOTS"
HEAVY_JOBS_WAIT_SECONDS_ENV = "LOCAL_OPERATOR_HEAVY_JOB_WAIT_S"
#: ``0`` (or unset) means "size from the host" (:func:`auto_slot_count`).
HEAVY_JOBS_SLOTS_DEFAULT = 0
#: Fifteen minutes: long enough to outlast a typical 5-10 minute build ahead in
#: the queue (PR #4104 QA K2 measured that a 120 s queue refused exactly that),
#: short enough that a wedged holder surfaces as a refusal, not a silent hang.
#: Deliberately NOT capped by the bash call's ``timeout``: that bounds the
#: command's own run, which starts at spawn, after the queue.
HEAVY_JOBS_WAIT_SECONDS_DEFAULT = 900
#: Accepted ranges for the env overrides; a value outside (or non-finite, or
#: unparseable) falls back to the default rather than stopping a command.
_WAIT_SECONDS_RANGE = (1.0, 4 * 3600.0)

#: A caller asking for this much memory (``memory_mb``) is declaring a heavy
#: job whatever the command looks like.
HEAVY_MEMORY_MB_THRESHOLD = 1024

#: Hosts at or below this much RAM get exactly one slot. node2's 15.7 GB OOM'd
#: with three builds in parallel; one is the only count proven safe there.
_SINGLE_SLOT_MAX_TOTAL_MB = 16 * 1024
#: Above that, one slot per this much AVAILABLE memory: a node-22 build peaked
#: at ~1.5 GB RSS and chrome/esbuild siblings several more, so 8 GB per slot
#: leaves headroom for everything else on the box.
_MB_PER_SLOT = 8 * 1024
_MAX_SLOTS = 4

#: How often a waiter retries; flock(LOCK_NB) is a syscall, so this is cheap.
_POLL_SECONDS = 0.25
#: Command preview length in records and messages: enough to recognise a build.
PREVIEW_CHARS = 80


@dataclass(frozen=True)
class HeavyClass:
    """Why a command counts as heavy, as a short human-readable label."""

    reason: str
    #: The line also starts a watcher/server (``pnpm build && pnpm dev``): it
    #: is still heavy, so it keeps the memory cap, but it takes NO slot, since
    #: the lease would last as long as the server (PR #4104 R2-1). The caller
    #: prints a note telling the agent to run the build on its own so it queues.
    long_lived: bool = False


@dataclass(frozen=True)
class Holder:
    """One live slot holder, as read from its JSON record."""

    slot: int
    pid: int
    session_id: str
    command: str
    started_at: float
    pgid: int | None = None
    #: False when the harness pid is dead and only the command's group keeps
    #: the record live (a harness SIGKILLed mid-build): the pid alone would
    #: point an operator at a process that no longer exists (PR #4104 QA note).
    harness_alive: bool = True

    def who(self) -> str:
        """``<session>/<pid>`` when the session is known, else ``pid N``.

        Adds ``(harness gone; command pgid N still running)`` when the
        orphaned command group, not the named harness, holds the slot.
        """
        who = f"{self.session_id}/{self.pid}" if self.session_id else f"pid {self.pid}"
        if not self.harness_alive and self.pgid is not None:
            who += f" (harness gone; command pgid {self.pgid} still running)"
        return who


def supported() -> bool:
    """Whether this platform can host the lease at all (POSIX flock)."""
    return fcntl is not None


# -- classification ---------------------------------------------------------

#: Splits a command line into simple commands. ``&`` is included so that
#: ``build & other`` is two segments. A redirection such as ``2>&1`` lexes as
#: ``2>``, ``&``, ``1``, so it splits into a stray ``1`` segment; that is
#: harmless because ``2>`` stays with its command and ``1`` names no tool.
_SEPARATORS = {"&&", "||", ";", "|", "&", "\n", "(", ")", ";;", "|&"}
_ENV_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

#: Wrappers that run "the rest of the line" as the real command: name ->
#: (positional arguments eaten before the wrapped command, options that take a
#: separate value). Per wrapper because one letter means different things:
#: ``nice -n 10`` takes a value, while ``taskset -c 0-3`` is a flag followed by
#: the positional cpu list.
_PREFIX_WRAPPERS: dict[str, tuple[int, frozenset[str]]] = {
    "timeout": (1, frozenset({"-s", "--signal", "-k", "--kill-after"})),
    "nice": (0, frozenset({"-n", "--adjustment"})),
    "nohup": (0, frozenset()),
    "setsid": (0, frozenset()),
    "time": (0, frozenset({"-f", "--format", "-o", "--output"})),
    "command": (0, frozenset()),
    "exec": (0, frozenset({"-a"})),
    "env": (0, frozenset({"-u", "--unset", "-C", "--chdir"})),
    "stdbuf": (0, frozenset({"-i", "-o", "-e"})),
    "ionice": (0, frozenset({"-c", "--class", "-n", "--classdata"})),
    "caffeinate": (0, frozenset({"-t", "-w"})),
    "taskset": (1, frozenset()),
    "chrt": (1, frozenset()),
    "prlimit": (0, frozenset({"-p", "--pid", "-o", "--output"})),
    "xvfb-run": (
        0,
        frozenset({"-n", "--server-num", "-s", "--server-args", "-e", "--error-file"})
        | frozenset({"-f", "--auth-file", "-p", "--xauth-protocol", "-w", "--wait"}),
    ),
    "corepack": (0, frozenset()),
    "npx": (0, frozenset({"-p", "--package"})),
    "bunx": (0, frozenset({"-p", "--package"})),
}

#: Package-manager scripts that run builds, test runners or type checks. These
#: are the scripts every JS repo on the fleet maps to tsc/vite/vitest/esbuild.
_PM_HEAVY_SCRIPT = re.compile(r"^(build|test|typecheck|type-check)(:.*)?$")
_PACKAGE_MANAGERS = ("pnpm", "npm", "yarn", "bun")
#: Options that take a separate value, per package manager: ``-w`` is npm's
#: ``--workspace <name>`` but pnpm's value-less ``--workspace-root``.
_PM_VALUE_OPTS = {"-C", "--dir", "--filter", "-F", "--prefix", "--cwd", "--workspace"}
#: Options that point the package manager at another package.json, so the
#: script table under the cwd no longer applies: fall back to name heuristics.
_PM_ELSEWHERE_OPTS = {"--filter", "-F", "-r", "--recursive", "--workspace", "-w", "--workspaces"}
_PM_DIR_OPTS = {"-C", "--dir", "--prefix", "--cwd"}
#: `npm <x>` runs a script only for these; any other word is an npm command.
_NPM_SCRIPT_ALIASES = {"test": "test", "t": "test", "tst": "test", "start": "start"}
#: Informational flags. The short ones count only as the SOLE argument
#: (``tsc -v``, ``vitest -h``), because ``-v`` is "verbose" for many test
#: runners (``pytest -v -n 4`` is a parallel run, not a version query).
_VERSION_FLAGS = {"--version", "-v", "-V", "--help", "-h"}
_INFO_FLAGS_ANYWHERE = {"--version", "--help"}
_SHELLS = {"bash", "sh", "zsh", "dash", "ksh"}
_PYTHON = re.compile(r"python(\d+(\.\d+)?)?")
#: JS tools that `node <path>/.bin/<tool>` (or `node …/tsc`) is really running.
_NODE_TOOLS = {"esbuild", "tsc", "vue-tsc", "vitest", "jest", "vite", "electron-vite"} | {
    "webpack",
    "rollup",
    "next",
}
#: Recursion bound for `bash -c`, package.json script bodies and wrappers that
#: re-dispatch. Cycles (`"build": "pnpm build"`) are stopped by a visited set of
#: (package dir, script), so this bound only limits honest nesting; 4 was too
#: shallow (`run_bounded.py -- bash -c "pnpm full"` over a three-script chain
#: fell off it, PR #4104 R2-3). Past it, :func:`_shallow_guess` decides from
#: tokens, never "light" by default.
_MAX_DEPTH = 8
#: package.json files larger than this are not read (a cheap read by contract).
_PACKAGE_JSON_MAX_BYTES = 1_000_000

#: Sentinel verdict: the segment is a watcher or a server that runs until it is
#: stopped. It never takes the slot (it would pin the machine's only slot for
#: hours while using little memory), and a command line containing one is not
#: leased at all, because the lease would last as long as the watcher. When the
#: same line also runs a heavy segment, see :data:`_HEAVY_LONG_LIVED`.
_LONG_LIVED = "\0long-lived"
#: Prefix verdict: heavy (the suffix is the reason) AND long-lived, as in
#: ``pnpm build && pnpm dev``. The build still needs the memory cap, yet a lease
#: would last as long as the server, so the caller takes no slot and tells the
#: agent to split the build out (PR #4104 R2-1).
_HEAVY_LONG_LIVED = "\0heavy-long-lived:"
#: Tools matched by :func:`_shallow_guess` past the recursion bound.
_SHALLOW_HEAVY_TOOLS = {"esbuild", "tsc", "vue-tsc", "vitest", "jest", "electron-vite"} | {
    "webpack",
    "rollup",
}

#: Placeholder for a ``$(…)``/backtick substitution, masked out of the line
#: before tokenising. Why (node2, 2026-11-18): ``env $(env | grep -o
#: '^CMUX_[A-Z_]*' | sed 's/^/-u /') -u NO_COLOR make test-affected`` split
#: on the substitution's own ``(``/``|``/``)``, so the segment that held
#: ``make test-affected`` began with ``-u`` and the whole line came back
#: unclassified: a heavy run with no slot, which is how node2 OOM'd before the
#: lease existed. The substitution's body is classified on its own instead.
_SUBST = "__lop_subst__"

#: pytest options that consume the NEXT argument, so it is not a test target
#: (``-k expr``, ``--deselect x``). The ``--opt=value`` form needs no entry.
_PYTEST_VALUE_OPTS = frozenset(
    {"-k", "-m", "-p", "-c", "-o", "-W", "-r", "--rootdir", "--confcutdir", "--deselect"}
    | {"--ignore", "--ignore-glob", "--durations", "--maxfail", "--tb", "--junitxml"}
    | {"--basetemp", "--timeout", "--dist", "--log-level", "--override-ini", "--import-mode"}
    | {"--cov", "--cov-report", "--max-worker-restart", "--config-file", "--capture"}
)
#: Calls that never run a test: collection listings and plugin/fixture docs.
_PYTEST_INFO_FLAGS = frozenset({"--collect-only", "--co", "--fixtures", "--markers"})

_Seen = frozenset[tuple[str | None, str]]


def _mask_substitutions(command: str) -> tuple[str, list[str]]:
    """``command`` with each top-level ``$(…)``/backtick body replaced by
    :data:`_SUBST`, plus those bodies.

    Single-quoted text is literal in the shell (``echo '$(pnpm build)'`` runs
    nothing), so it is left alone. Parentheses nest (``$(a $(b))``); an
    unbalanced one masks to the end of the line, which errs toward classifying
    the body rather than hiding what follows it. Linear, no subprocess.
    """
    out: list[str] = []
    bodies: list[str] = []
    i, n = 0, len(command)
    in_single = in_double = False
    while i < n:
        char = command[i]
        if char == "\\" and not in_single:
            out.append(command[i : i + 2])
            i += 2
            continue
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif not in_single and (char == "`" or command.startswith("$(", i)):
            if char == "`":
                end = command.find("`", i + 1)
                end = n if end < 0 else end
                bodies.append(command[i + 1 : end])
                i = end + 1
            else:
                depth, j = 1, i + 2
                while j < n and depth:
                    depth += {"(": 1, ")": -1}.get(command[j], 0)
                    j += 1
                bodies.append(command[i + 2 : j - 1 if depth == 0 else j])
                i = j
            out.append(_SUBST)
            continue
        out.append(char)
        i += 1
    return "".join(out), bodies


def _unpack(verdict: str | None) -> tuple[str | None, bool]:
    """``(heavy reason or None, long-lived?)`` for any verdict string."""
    if verdict is None:
        return None, False
    if verdict == _LONG_LIVED:
        return None, True
    if verdict.startswith(_HEAVY_LONG_LIVED):
        return verdict[len(_HEAVY_LONG_LIVED) :], True
    return verdict, False


def _shallow_guess(command: str) -> str | None:
    """Token-only verdict once the recursion bound is reached (PR #4104 R2-3).

    A false negative is what the lease exists to prevent, so a chain too deep
    to resolve is judged by its words: a known heavy tool, or a package-manager
    script whose NAME is a build/test/typecheck. No package.json is read here.
    """
    words = command.split()
    for index, word in enumerate(words):
        name = os.path.basename(word.strip("'\""))
        if name in _SHALLOW_HEAVY_TOOLS:
            return name
        is_module = name == "-m" and words[index + 1 : index + 2] == ["pytest"]
        if name == "pytest" or is_module:
            # Past the recursion bound a pytest gets the same verdict as a
            # resolved one, from its words up to the next shell operator.
            after: list[str] = []
            for later in words[index + (2 if is_module else 1) :]:
                if later in ("&&", "||", ";", "|", "&"):
                    break
                after.append(later.strip("'\""))
            if verdict := _classify_pytest(after):
                return verdict
        if name in _PACKAGE_MANAGERS:
            rest = [w.strip("'\"") for w in words[index + 1 :] if not w.startswith("-")]
            if rest[:1] in (["run"], ["run-script"]):
                rest = rest[1:]
            if rest and _PM_HEAVY_SCRIPT.match(rest[0]):
                return f"{name} {rest[0]}"
    return None


def _segments(command: str) -> list[list[str]]:
    """Split ``command`` into simple-command token lists; [] when unparseable."""
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|()")
        lexer.whitespace_split = True
        lexer.commenters = "#"
        tokens = list(lexer)
    except ValueError:
        # An unbalanced quote: fall back to a plain whitespace split rather
        # than give up, so a heavy build with a stray quote still queues.
        tokens = command.split()
    segments: list[list[str]] = [[]]
    for token in tokens:
        if token in _SEPARATORS:
            segments.append([])
        else:
            segments[-1].append(token)
    return [segment for segment in segments if segment]


def _strip_wrappers(tokens: list[str]) -> list[str]:
    """Drop env assignments and runner prefixes to reach the real program."""
    i = 0
    while i < len(tokens):
        token = tokens[i]
        base = os.path.basename(token)
        if _ENV_ASSIGNMENT.match(token):
            i += 1
            continue
        if base == "systemd-run":
            # Everything up to `--` (or the first non-option) is systemd-run's.
            rest = tokens[i + 1 :]
            if "--" in rest:
                i = i + 1 + rest.index("--") + 1
                continue
            i += 1
            while i < len(tokens) and tokens[i].startswith("-"):
                i += 1
            continue
        if base in ("uv", "poetry", "pdm", "hatch", "pipenv") and tokens[i + 1 : i + 2] == ["run"]:
            i += 2
            while i < len(tokens) and tokens[i].startswith("-"):
                # `uv run --with x pytest`: value-taking options are common here.
                if tokens[i] in ("--with", "--python", "-p", "--project", "--directory", "--env"):
                    i += 1
                i += 1
            continue
        if base in ("uvx", "pipx"):
            i += 1
            while i < len(tokens) and tokens[i].startswith("-"):
                i += 1
            if base == "pipx" and i < len(tokens) and tokens[i] == "run":
                i += 1
            continue
        if base in _PREFIX_WRAPPERS:
            i += 1
            positionals, value_opts = _PREFIX_WRAPPERS[base]
            while i < len(tokens):
                if tokens[i] == "--":
                    i += 1
                    break
                if tokens[i].startswith("-") and tokens[i] != "-":
                    i += 2 if tokens[i] in value_opts else 1
                    continue
                if base == "env" and (_ENV_ASSIGNMENT.match(tokens[i]) or tokens[i] == _SUBST):
                    # A substitution in env's argument position expands to
                    # flags/assignments (`env $(… sed 's/^/-u /') make test`).
                    i += 1
                    continue
                if positionals > 0:
                    positionals -= 1
                    i += 1
                    continue
                break
            continue
        break
    return tokens[i:]


def _watch_flag(arg: str) -> bool:
    """``--watch``/``--watchAll`` (or ``=true``-style values) on any tool."""
    if arg in ("--watch", "--watchAll", "--watch-all"):
        return True
    for flag in ("--watch=", "--watchAll="):
        if arg.startswith(flag):
            return arg[len(flag) :].lower() not in ("false", "0", "no")
    return False


def _script_is_long_lived(name: str) -> bool:
    """``dev``, ``serve``, ``start``, ``watch``, ``*:watch``, ``build:watch``…"""
    parts = [part for part in re.split(r"[:\-_.]", name.lower()) if part]
    if not parts:
        return False
    return (
        "watch" in parts
        or parts[0] in ("dev", "serve", "start", "preview")
        or parts[-1] in ("serve", "start")
    )


def _positionals(args: list[str]) -> list[str]:
    return [arg for arg in args if not arg.startswith("-")]


def _cd(cwd: str | None, args: list[str]) -> str | None:
    """The directory after ``cd <args>``; None once it is unknowable (``cd -``)."""
    if cwd is None:
        return None
    target = args[0] if args else "~"
    if target == "-" or "$" in target:
        return None
    return os.path.join(cwd, os.path.expanduser(target))


def _package_scripts(directory: str | None) -> dict[str, str] | None:
    """``scripts`` from the nearest package.json at or above ``directory``.

    Why: ``pnpm build`` is only as heavy as what its script runs, and what
    exec would run when no script has that name is the binary itself. Cheap by
    contract: one small JSON read, no package-manager subprocess; any failure
    means "unknown" and the caller falls back to name heuristics.
    """
    if not directory:
        return None
    try:
        current = Path(directory).expanduser().resolve()
    except (OSError, RuntimeError):
        return None
    for folder in (current, *current.parents):
        path = folder / "package.json"
        try:
            if not path.is_file():
                continue
            if path.stat().st_size > _PACKAGE_JSON_MAX_BYTES:
                return None
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        scripts = data.get("scripts") if isinstance(data, dict) else None
        if not isinstance(scripts, dict):
            return {}
        return {k: v for k, v in scripts.items() if isinstance(k, str) and isinstance(v, str)}
    return None


def _classify_command(
    command: str, cwd: str | None, depth: int, seen: _Seen = frozenset()
) -> str | None:
    """Heavy reason, :data:`_LONG_LIVED`, a :data:`_HEAVY_LONG_LIVED` verdict,
    or None for a whole command line.

    Every segment is classified: a long-lived segment no longer hides a heavy
    one beside it (``pnpm build && pnpm dev`` is heavy AND long-lived).
    """
    if depth > _MAX_DEPTH:
        return _shallow_guess(command)
    heavy: str | None = None
    long_lived = False
    command, bodies = _mask_substitutions(command)
    for body in bodies:
        # `$(pnpm build)` still runs the build; only its heaviness counts, since
        # a watcher's output can never be substituted (it would never end).
        heavy = heavy or _unpack(_classify_command(body, cwd, depth + 1, seen))[0]
    for segment in _segments(command):
        if segment[0] in ("cd", "pushd"):
            cwd = _cd(cwd, segment[1:])
            continue
        reason, lived = _unpack(_classify_segment(segment, cwd, depth, seen))
        long_lived = long_lived or lived
        if heavy is None:
            heavy = reason
    if long_lived:
        return _HEAVY_LONG_LIVED + heavy if heavy else _LONG_LIVED
    return heavy


def _shell_c_string(args: list[str]) -> str | None:
    """The command string of ``bash -c '…'`` / ``sh -lc '…'``, if any."""
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("-o", "+o"):
            i += 2
            continue
        if arg.startswith("--"):
            if arg == "--":
                return None
            i += 1
            continue
        if not arg.startswith("-"):
            return None  # a script file: not inspected
        if "c" in arg[1:]:
            return args[i + 1] if i + 1 < len(args) else None
        i += 1
    return None


def _classify_pm(
    program: str, args: list[str], cwd: str | None, depth: int, seen: _Seen = frozenset()
) -> str | None:
    """``pnpm|npm|yarn|bun …``: classify what it really runs."""
    pkg_dir, elsewhere = cwd, False
    i = 0
    while i < len(args) and args[i].startswith("-"):
        opt, _, attached = args[i].partition("=")
        if opt in _PM_ELSEWHERE_OPTS:
            elsewhere = True
        value = attached or (args[i + 1] if i + 1 < len(args) else "")
        if opt in _PM_DIR_OPTS:
            pkg_dir = _cd(pkg_dir, [value])
        takes_value = opt in _PM_VALUE_OPTS or (program == "npm" and opt == "-w")
        i += 2 if takes_value and not attached else 1
    if i >= len(args):
        return None
    sub, rest = args[i], args[i + 1 :]
    if sub in ("exec", "dlx", "x"):
        j = 0
        while j < len(rest) and rest[j].startswith("-"):
            if rest[j] == "--":
                j += 1
                break
            j += 2 if rest[j] in ("-p", "--package") else 1
        return _classify_segment(rest[j:], pkg_dir, depth + 1, seen) if rest[j:] else None
    explicit_run = sub in ("run", "run-script")
    if explicit_run:
        j = 0
        while j < len(rest) and rest[j].startswith("-"):
            j += 2 if rest[j] in _PM_VALUE_OPTS else 1
        if j >= len(rest):
            return None
        script, extra = rest[j], rest[j + 1 :]
    elif program == "npm":
        if sub not in _NPM_SCRIPT_ALIASES:
            return None  # `npm build`, `npm install`: npm commands, not scripts
        script, extra = _NPM_SCRIPT_ALIASES[sub], rest
    else:
        script, extra = sub, rest
    # `pnpm test -- --watch`, `pnpm test --watch`: the flag reaches the runner.
    if any(_watch_flag(arg) for arg in extra):
        return _LONG_LIVED
    # The NAME wins first for watchers: `"test:watch": "vitest"` runs vitest
    # in watch mode in a terminal, which its body alone does not show.
    if _script_is_long_lived(script):
        return _LONG_LIVED
    scripts = None if elsewhere else _package_scripts(pkg_dir)
    body = scripts.get(script) if scripts else None
    if (pkg_dir, script) in seen:
        body = None  # a script that (indirectly) calls itself: judge it by name
    if body is not None:
        verdict = _classify_command(body, pkg_dir, depth + 1, seen | {(pkg_dir, script)})
        reason, lived = _unpack(verdict)
        if lived:
            return _HEAVY_LONG_LIVED + f"{program} {script}" if reason else _LONG_LIVED
        if reason is not None:
            return f"{program} {script}"
    if _PM_HEAVY_SCRIPT.match(script):
        return f"{program} {script}"
    if body is None and not explicit_run and program != "npm":
        # No script by that name: pnpm/yarn/bun run the local binary, so
        # `pnpm vitest run` is vitest and `pnpm tsc --noEmit` is tsc.
        return _classify_segment([script, *extra], pkg_dir, depth + 1, seen)
    return None


def _task_names(args: list[str]) -> list[str]:
    """Target names in an nx/turbo argv: ``proj:build``, ``-t build,test``…"""
    names: list[str] = []
    take_next = False
    for arg in args:
        if take_next:
            names.extend(arg.split(","))
            take_next = False
            continue
        opt, eq, value = arg.partition("=")
        if opt in ("-t", "--target", "--targets", "--task", "--tasks"):
            if eq:
                names.extend(value.split(","))
            else:
                take_next = True
            continue
        if not arg.startswith("-"):
            names.extend(
                part for chunk in arg.split(",") for part in chunk.split(":")[1:2] or [chunk]
            )
    return names


def _classify_segment(
    tokens: list[str], cwd: str | None, depth: int = 0, seen: _Seen = frozenset()
) -> str | None:
    if depth > _MAX_DEPTH:
        return _shallow_guess(" ".join(tokens))
    tokens = _strip_wrappers(tokens)
    if not tokens:
        return None
    program = os.path.basename(tokens[0])
    args = tokens[1:]
    if (len(args) == 1 and args[0] in _VERSION_FLAGS) or any(
        a in _INFO_FLAGS_ANYWHERE for a in args
    ):
        # `tsc --version`, `pnpm build --help`: an informational call is not a build.
        return None
    if program in _SHELLS:
        script = _shell_c_string(args)
        return _classify_command(script, cwd, depth + 1, seen) if script is not None else None
    if program == "run_bounded.py" or (
        _PYTHON.fullmatch(program) and args[:1] and os.path.basename(args[0]) == "run_bounded.py"
    ):
        # The bounded runner AGENTS.md prescribes for heavy runs: `… -- <cmd>`.
        rest = args[args.index("--") + 1 :] if "--" in args else []
        return _classify_segment(rest, cwd, depth + 1, seen) if rest else None
    if program in _PACKAGE_MANAGERS:
        verdict = _classify_pm(program, args, cwd, depth, seen)
        if verdict is not None or program != "bun":
            return verdict
    if program in ("node", "bun", "deno"):
        if "--test" in args:
            return _LONG_LIVED if any(_watch_flag(a) for a in args) else f"{program} --test"
        if any(_watch_flag(a) for a in args):
            return _LONG_LIVED
        # `node node_modules/.bin/esbuild …`: the script is the tool.
        script_at = next((n for n, a in enumerate(args) if not a.startswith("-")), None)
        if script_at is not None:
            tool = re.sub(r"\.(c|m)?js$", "", os.path.basename(args[script_at]))
            if tool in _NODE_TOOLS:
                return _classify_segment([tool, *args[script_at + 1 :]], cwd, depth + 1, seen)
        return None
    first = _positionals(args)[:1]
    if program in ("vite", "electron-vite"):
        # Builds only. `vite`/`electron-vite dev|preview` is a long-lived server.
        if "build" in args and not any(_watch_flag(a) or a == "-w" for a in args):
            return f"{program} build"
        return _LONG_LIVED
    if program in ("tsc", "vue-tsc"):
        return _LONG_LIVED if any(_watch_flag(a) or a == "-w" for a in args) else program
    if program == "vitest":
        if first in (["watch"], ["dev"]) or any(_watch_flag(a) or a == "-w" for a in args):
            return _LONG_LIVED
        return program
    if program == "jest":
        return _LONG_LIVED if any(_watch_flag(a) for a in args) else program
    if program == "esbuild":
        if any(_watch_flag(a) or a == "--serve" or a.startswith("--serve=") for a in args):
            return _LONG_LIVED
        return program
    if program in ("webpack", "webpack-cli", "rollup"):
        if first in (["serve"], ["watch"]) or any(_watch_flag(a) or a == "-w" for a in args):
            return _LONG_LIVED
        return program
    if program == "webpack-dev-server":
        return _LONG_LIVED
    if program == "next":
        if first == ["build"]:
            return "next build"
        return _LONG_LIVED if first in (["dev"], ["start"]) else None
    if program in ("turbo", "nx"):
        if first == ["watch"] or any(_watch_flag(a) for a in args):
            return _LONG_LIVED
        names = _task_names(args)
        if any(_script_is_long_lived(name) for name in names):
            return _LONG_LIVED
        heavy = next((name for name in names if _PM_HEAVY_SCRIPT.match(name)), None)
        return f"{program} {heavy}" if heavy else None
    if program in ("cargo", "go") and args[:1] and args[0] in ("build", "test"):
        return f"{program} {args[0]}"
    if program == "make" and any(a.startswith("test") for a in args if not a.startswith("-")):
        return "make test"
    if program == "affected_tests.py" or (
        _PYTHON.fullmatch(program) and args[:1] and os.path.basename(args[0]) == "affected_tests.py"
    ):
        # `make test-affected`'s engine: `--run`/`--evidence` run the selection
        # (node2 measured 1139 s / 6720 tests with no slot); bare, it only lists.
        return "affected_tests --run" if {"--run", "--evidence"} & set(args) else None
    pytest_args: list[str] | None = None
    if program in ("pytest", "py.test"):
        pytest_args = args
    elif _PYTHON.fullmatch(program) and args[:2] == ["-m", "pytest"]:
        pytest_args = args[2:]
    if pytest_args is not None:
        return _classify_pytest(pytest_args)
    return None


def _xdist_workers(args: list[str]) -> str | None:
    """The LAST ``-n`` value in ``args`` (pytest lets the last one win), or None."""
    value: str | None = None
    for index, arg in enumerate(args):
        if arg in ("-n", "--numprocesses"):
            value = args[index + 1] if index + 1 < len(args) else ""
        elif match := re.fullmatch(r"-n(\S+)|--numprocesses=(\S+)", arg):
            value = match.group(1) or match.group(2)
        elif arg in ("-p", "no:xdist") and "no:xdist" in args[index : index + 2]:
            return "0"
    return value


def _parallel_workers(value: str | None) -> bool:
    """True when a command-line ``-n`` value forks more than one worker.

    ``auto``/``logical`` and any N > 1 fork; ``0``/``1`` do not. A value we
    cannot read (``-n "$N"``) counts as parallel: a false negative is what the
    lease exists to prevent.
    """
    if value is None:
        return False
    if value in ("auto", "logical"):
        return True
    try:
        return int(value) > 1
    except ValueError:
        return True


def _classify_pytest(args: list[str]) -> str | None:
    """Heavy reason for a pytest run, or None.

    Policy (AGENTS.md "To wait for heavy work"; manager decision 2026-11-18,
    PR #4227): "a single targeted test file or `-k` run should stay light, so
    small iterations don't queue behind a long run." Only the COMMAND LINE is
    read, never the repo's pytest ``addopts``: local-operator's addopts carry
    ``-n auto``, and honouring them would put every inner-loop run of one file
    in the queue behind a 20-minute suite. We accept that such a run forks
    xdist workers without a slot; it is short, and the run_bounded memory cap
    still bounds it. So, in order:

    * ``--collect-only``/``--fixtures``/``--markers`` never run a test: light.
    * an explicit ``-n N`` (N > 1) or ``-n auto``/``logical``: heavy. The last
      ``-n`` wins and ``-p no:xdist`` serialises, as in pytest.
    * narrowed with ``-k``: light, whatever the target (the decision above).
    * only test files or node ids (``x.py``, ``x.py::test``): light.
    * a directory target, or no target at all (the whole ``testpaths``), is
      heavy even serially: one interpreter importing the suite (node2,
      2026-11-18: ``pytest -q tests/unit`` ran with no slot).
    """
    if any(arg in _PYTEST_INFO_FLAGS for arg in args):
        return None
    if _parallel_workers(_xdist_workers(args)):
        return "pytest -n"
    targets: list[str] = []
    skip = False
    for arg in args:
        if skip:
            skip = False
        elif arg.startswith("-"):
            if arg.startswith("-k"):  # `-k expr` and `-kexpr`
                return None
            skip = arg in _PYTEST_VALUE_OPTS or arg in ("-n", "--numprocesses")
        else:
            targets.append(arg)
    if not targets:
        return "pytest (whole suite)"
    if any(not (t.split("::", 1)[0].endswith(".py")) for t in targets):
        return "pytest <directory>"
    return None


def is_heavy(
    command: str, memory_mb: float | None = None, cwd: str | None = None
) -> HeavyClass | None:
    """Classify a ``bash`` command as heavy (lease it) or not (run it now).

    Deliberately conservative: a false positive only queues a command behind a
    build, while a false negative is what the lease exists to prevent, yet a
    lease that queues `ls` would be disabled by the first operator it annoys.
    So only program families that measurably spend gigabytes are matched:
    bundlers (esbuild, vite/electron-vite build, webpack, rollup, next build),
    the TypeScript compiler, JS test runners (vitest, jest, node --test),
    package-manager build/test/typecheck scripts, nx/turbo build|test targets,
    parallel pytest (``-n`` N>1/auto on the command line; repo addopts are not
    read), a directory or whole-suite pytest not narrowed by ``-k``,
    ``affected_tests.py --run|--evidence``, ``make test*`` and
    ``cargo``/``go`` build|test. ``$(…)``/backtick bodies are classified too,
    and never hide the command around them.

    Classified by what actually runs: wrappers (``cd x &&``, env assignments,
    ``timeout N``, ``nice``, ``taskset``, ``chrt``, ``prlimit``, ``xvfb-run``,
    ``corepack``, ``systemd-run … --``, ``scripts/run_bounded.py … --``,
    ``uv run``, ``npx``, ``pnpm exec|dlx``) are seen through, ``bash -c '…'``
    strings are classified, ``node <path>/.bin/<tool>`` is the tool, and a
    package-manager script is resolved through the nearest package.json under
    ``cwd`` (when given and readable): its body decides, and with no script of
    that name pnpm/yarn/bun run the binary, as ``exec`` would. Without
    ``cwd`` (or package.json) the script NAME decides.

    Watchers and servers are never heavy (``--watch``, ``tsc -w``, ``vitest
    watch|dev``, ``jest --watchAll``, ``esbuild --serve``, ``vite`` dev, and
    scripts named ``dev``/``serve``/``start``/``watch``/``*:watch``): they run
    until stopped, so a lease would pin the machine's slot for hours. A line
    that runs a heavy segment AND one of those (``pnpm build && pnpm dev``) is
    heavy with ``long_lived=True``: capped, but never leased.
    ``--help``/``--version`` anywhere is informational, never heavy.

    A caller-declared ``memory_mb`` at or above :data:`HEAVY_MEMORY_MB_THRESHOLD`
    is heavy regardless.
    """
    if memory_mb is not None and memory_mb >= HEAVY_MEMORY_MB_THRESHOLD:
        return HeavyClass(f"memory_mb={int(memory_mb)}")
    reason, long_lived = _unpack(_classify_command(command, cwd, 0))
    if reason is None:
        return None
    return HeavyClass(reason, long_lived=long_lived)


# -- slot sizing ------------------------------------------------------------


def auto_slot_count(total_mb: int | None, available_mb: int | None) -> int:
    """Slots for ``slots: 0``: 1 on ≤16 GB hosts, else one per 8 GB available, max 4.

    Unmeasurable memory means one slot: the safe answer, never a guess upward.
    Available (not total) memory above the small-host line, because a large
    host already crowded by other work should not admit four builds.
    """
    if total_mb is None or total_mb <= _SINGLE_SLOT_MAX_TOTAL_MB or available_mb is None:
        return 1
    return max(1, min(_MAX_SLOTS, available_mb // _MB_PER_SLOT))


def resolve_slot_count(configured: Any) -> int:
    """The slot count for a configured value; non-positive or bad means auto.

    Clamped to ``_MAX_SLOTS``: more parallel builds than the largest auto size
    is the failure this module prevents, not a tuning choice.
    """
    if isinstance(configured, int) and not isinstance(configured, bool) and configured > 0:
        return min(configured, _MAX_SLOTS)
    from local_operator import memory_guard

    return auto_slot_count(
        memory_guard._total_memory_mb(),
        memory_guard._available_memory_mb(memory_guard._default_runner),
    )


@dataclass(frozen=True)
class Settings:
    enabled: bool
    slots: int
    wait_seconds: float


def _env_number(name: str, default: float, low: float, high: float) -> float:
    """A finite env number in ``[low, high]``, else ``default``.

    ``inf`` once crashed every heavy call (``int(inf)`` raised OverflowError,
    PR #4104 R4), and ``nan`` compares false to everything, so both are
    rejected explicitly instead of trusting comparisons.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if not math.isfinite(value) or not low <= value <= high:
        return default
    return value


def configured_settings() -> Settings:
    """Read the switch LIVE (fresh ConfigManager) and the two env overrides.

    A broken ``config.yml`` or a malformed override must never stop a command,
    so any read failure or wrongly typed value falls back to the default.
    """
    enabled: Any = HEAVY_JOBS_ENABLED_DEFAULT
    try:
        from local_operator.config import ConfigManager
        from local_operator.paths import config_dir

        config = ConfigManager(config_dir())
        enabled = config.get_nested_value(HEAVY_JOBS_ENABLED_PATH, enabled)
    except Exception:  # noqa: BLE001 - config trouble must never block a command
        pass
    if not isinstance(enabled, bool):
        enabled = HEAVY_JOBS_ENABLED_DEFAULT
    # Slots: 0 = auto, 1.._MAX_SLOTS as given; anything else is auto.
    slots = int(_env_number(HEAVY_JOBS_SLOTS_ENV, HEAVY_JOBS_SLOTS_DEFAULT, 0, _MAX_SLOTS))
    wait = _env_number(
        HEAVY_JOBS_WAIT_SECONDS_ENV, HEAVY_JOBS_WAIT_SECONDS_DEFAULT, *_WAIT_SECONDS_RANGE
    )
    return Settings(enabled=enabled, slots=resolve_slot_count(slots), wait_seconds=wait)


def lease_dir() -> Path:
    """``<config_dir>/run/heavy-jobs``: shared by every session of this config."""
    from local_operator.paths import config_dir

    return config_dir() / "run" / "heavy-jobs"


# -- the lease ----------------------------------------------------------------


def _pid_alive(pid: int) -> bool:
    from local_operator.procstate import pid_liveness

    # Doubt counts as alive: naming a holder that just died is harmless, while
    # hiding a live one would misreport who is using the machine.
    return pid_liveness(pid) is not False


def _group_alive(pgid: int) -> bool:
    if pgid <= 1:
        return False
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True  # EPERM: it exists, it is just not ours to signal
    return True


class Lease:
    """One held slot. Release exactly once; closing the fd drops the flock."""

    def __init__(self, slot: int, fd: int, record_path: Path, record: dict[str, Any]):
        self.slot = slot
        self._fd: int | None = fd
        self._record_path = record_path
        self._record = record

    @property
    def held(self) -> bool:
        return self._fd is not None

    def fileno(self) -> int | None:
        """The lock descriptor, for the spawn's ``pass_fds`` (None once released).

        The command inherits a duplicate so the flock outlives a harness that
        dies mid-build (see the module docstring); the harness's own
        :meth:`release` still frees the slot for every duplicate.
        """
        return self._fd

    def note_pgid(self, pgid: int | None) -> None:
        """Record the spawned group once known, so status can name the command."""
        if self._fd is None or pgid is None:
            return
        self._record["pgid"] = pgid
        _write_record(self._record_path, self._record)

    def __del__(self) -> None:
        # Safety net only: every caller releases explicitly. A Lease dropped on
        # an unexpected path must not keep the machine's slot until exit.
        self.release()

    def release(self) -> None:
        fd, self._fd = self._fd, None
        if fd is None:
            return
        # Remove the record BEFORE dropping the lock, so the next holder's
        # fresh record can never be deleted by our late unlink.
        try:
            self._record_path.unlink()
        except OSError:
            pass
        # LOCK_UN, not just close: the command (and anything it left behind)
        # holds a duplicate of this description, and closing ours alone would
        # leave the slot locked by that copy. An explicit unlock releases the
        # lock for every duplicate at once.
        if fcntl is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
        try:
            os.close(fd)
        except OSError:
            pass


def _write_record(path: Path, record: dict[str, Any]) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{id(record)}.tmp")
    try:
        tmp.write_text(json.dumps(record), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        # The record is advisory; a failed write only costs a waiter the name.
        try:
            tmp.unlink()
        except OSError:
            pass


def try_acquire(
    directory: Path,
    slots: int,
    *,
    session_id: str = "",
    command: str = "",
    pid: int | None = None,
) -> Lease | None:
    """Take a free slot without waiting, or return None when all are held."""
    if fcntl is None:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    for slot in range(max(1, slots)):
        lock_path = directory / f"slot-{slot}.lock"
        # Read-only: flock needs no write access (Linux and macOS), and the
        # leased command inherits this descriptor, so a sandboxed read-tier
        # shell must not get a writable fd outside its sandbox (PR #4104 R2-2).
        fd = os.open(lock_path, os.O_RDONLY | os.O_CREAT | getattr(os, "O_CLOEXEC", 0), 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            continue
        record = {
            "slot": slot,
            "pid": pid if pid is not None else os.getpid(),
            "session_id": session_id,
            "command": command[:PREVIEW_CHARS],
            "started_at": time.time(),
            "pgid": None,
        }
        record_path = directory / f"slot-{slot}.json"
        _write_record(record_path, record)
        return Lease(slot, fd, record_path, record)
    return None


def read_holders(directory: Path, slots: int | None = None) -> list[Holder]:
    """Live holders from the slot records; stale records are skipped.

    A record is live while its harness ``pid`` OR its command's ``pgid`` is
    alive: a harness killed mid-build leaves its record behind, and the build
    still holds the slot through its inherited descriptor, so it is still the
    holder a waiter must be told about.

    Reads JSON only and never probes the flock, so it is cheap enough for a
    status command and cannot disturb a holder.
    """
    holders: list[Holder] = []
    try:
        paths = sorted(directory.glob("slot-*.json"))
    except OSError:
        return holders
    for path in paths:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            slot = int(raw["slot"])
            pid = int(raw["pid"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if slots is not None and slot >= slots:
            continue
        pgid = raw.get("pgid")
        harness_alive = _pid_alive(pid)
        if not harness_alive and not (isinstance(pgid, int) and _group_alive(pgid)):
            continue
        holders.append(
            Holder(
                slot=slot,
                pid=pid,
                session_id=str(raw.get("session_id") or ""),
                command=str(raw.get("command") or ""),
                started_at=float(raw.get("started_at") or 0.0),
                pgid=pgid if isinstance(pgid, int) else None,
                harness_alive=harness_alive,
            )
        )
    return holders


def describe_holders(holders: Iterable[Holder]) -> str:
    """``<session or pid N> (<command>)`` for each holder, comma-joined."""
    parts = [f"{h.who()} ({h.command})" if h.command else h.who() for h in holders]
    return ", ".join(parts) if parts else "an unidentified holder"


@dataclass(frozen=True)
class Refusal:
    """The wait ran out; ``message`` is the tool error to return."""

    message: str
    holders: tuple[Holder, ...]


def refusal_message(
    holders: Iterable[Holder], waited: float, reason: str, wait_seconds: float | None = None
) -> str:
    """The refusal once the QUEUE bound ran out.

    The queue bound is the only bound that can refuse here: the command's own
    ``timeout`` starts at spawn and never caps the queue (PR #4104 Q3 found
    the old message blaming the wrong bound). So the remedy names the queue
    bound and its owner, the operator's harness environment, not a parameter
    the agent could change.
    """
    bound = f"{wait_seconds if wait_seconds is not None else waited:g}"
    return (
        f"HEAVY JOB QUEUE FULL: this command ({reason}) waited {int(waited)}s, the "
        f"queue limit ({HEAVY_JOBS_WAIT_SECONDS_ENV}={bound}s), for the machine-wide "
        "heavy-job slot and did not get it.\n"
        f"Held by: {describe_holders(holders)}.\n"
        "Heavy builds/tests share a few slots per machine (one on hosts with 16 GB "
        "or less), so parallel builds cannot OOM the host. Retry once the holder "
        "finishes (check with `lop services status`). The operator can lengthen the "
        f"queue limit with {HEAVY_JOBS_WAIT_SECONDS_ENV} in lop's environment, or turn "
        "the queue off on this machine with bash.heavy_jobs.enabled=false."
    )


def queued_line(holders: Iterable[Holder], waited: float) -> str:
    return f"queued behind {describe_holders(holders)}, waited {int(waited)}s"


async def acquire(
    directory: Path,
    slots: int,
    *,
    wait_seconds: float,
    reason: str,
    session_id: str = "",
    command: str = "",
    on_wait: Callable[[str], None] | None = None,
    aborted: Callable[[], bool] | None = None,
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
) -> Lease | Refusal | None:
    """Wait up to ``wait_seconds`` for a slot.

    Returns the :class:`Lease`, a :class:`Refusal` when the wait ran out, or
    ``None`` when ``aborted()`` turned true (the caller handles its own abort).
    ``on_wait`` receives the visible "queued behind …" line about once a second.
    """
    started = time.monotonic()
    last_note = -1.0
    while True:
        lease = try_acquire(directory, slots, session_id=session_id, command=command)
        if lease is not None:
            return lease
        waited = time.monotonic() - started
        if aborted is not None and aborted():
            return None
        if waited >= wait_seconds:
            holders = tuple(read_holders(directory))
            return Refusal(refusal_message(holders, waited, reason, wait_seconds), holders)
        if on_wait is not None and waited - last_note >= 1.0:
            last_note = waited
            on_wait(queued_line(read_holders(directory), waited))
        await sleep(min(_POLL_SECONDS, max(0.01, wait_seconds - waited)))


def status_line(directory: Path, slots: int, now: float | None = None) -> str:
    """The ``lop services status`` row: holders with age, or the free count.

    Counts EVERY live holder, not just those below the current slot count:
    auto sizing follows MemAvailable, which falls while builds run, so a
    holder taken under a larger count must not vanish from the report (PR
    #4104 R5). When they disagree the row says so: ``2 held (1 slot now)``.
    """
    holders = read_holders(directory)
    if not holders:
        return f"heavy jobs: 0/{slots} slots in use"
    now = time.time() if now is None else now
    parts = []
    for holder in holders:
        minutes = max(0, int((now - holder.started_at) // 60))
        parts.append(f"{holder.who()} {holder.command} ({minutes}m)")
    if len(holders) <= slots and all(holder.slot < slots for holder in holders):
        head = f"{len(holders)}/{slots} slots held"
    else:
        head = f"{len(holders)} held ({slots} slot{'s' if slots != 1 else ''} now)"
    return f"heavy jobs: {head}: " + "; ".join(parts)
