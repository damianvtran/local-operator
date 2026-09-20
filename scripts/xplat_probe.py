#!/usr/bin/env python3
"""Cross-platform probe battery for local-operator.

Why this exists
---------------

`local-operator` is developed and released from macOS. The unit suite is run on
Linux in CI and the interactive surface is exercised on macOS, so a mechanism
that is silently macOS-only -- or silently POSIX-only -- passes every gate we
have. The failure class this file targets is the *silent* one: not an
`ImportError` at startup, which somebody would notice, but a code path that
returns "unsupported" and leaves the user with a feature that reports success
and does nothing.

This script is the instrument, not a test. It runs the same battery on every OS
it is pointed at and prints one matrix, so the *difference* between platforms is
the output. It is deliberately:

* **stdlib-only**, so it runs in a bare container before any dependency is
  installed;
* **self-driving**, so no pytest, no fixtures and no repository test helpers are
  required to get a reading;
* **fail-soft**, so one crashing probe does not cost the other twenty their
  evidence -- every probe is isolated in its own try/except with a timeout;
* **isolated**, so it never reads or writes the operator's live
  `~/.local-operator` (see `--isolate`), because a probe battery that mutates
  the thing it measures is worse than no probe.

Usage
-----

    python scripts/xplat_probe.py                    # full battery, table
    python scripts/xplat_probe.py --json out.json    # ... and machine output
    python scripts/xplat_probe.py --only tui cli     # substring filter
    python scripts/xplat_probe.py --list             # probe names only
    python scripts/xplat_probe.py --keep             # keep the sandbox dir
    python scripts/xplat_probe.py --budget 1200      # ... and return inside 20 minutes

Exit code is 0 when no probe FAILed, 1 otherwise. `WARN` and `SKIP` do not fail
the run: `WARN` is "works, but with a gap you should read", `SKIP` is "this
probe could not be attempted here" -- either because the platform has no such
mechanism or because the surface needs something (a network, a display) the
host does not have. Both are printed with the reason, because an unexplained
skip is how a probe battery lies.
"""

from __future__ import annotations

import argparse
import ast
import bisect
import csv
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

REPO = Path(__file__).resolve().parent.parent

#: The subcommands `cli.subcommands` walks. Kept as a literal rather than
#: discovered from `--help`, because a subcommand that fails to register is
#: exactly the defect this probe should report, and discovering the list from
#: the same parser that is broken would report a clean run.
SUBCOMMANDS = (
    "credential",
    "config",
    "agents",
    "teams",
    "serve",
    "mobile",
    "tunnel",
    "secret",
    "browser",
    "send",
    "sessions",
    "stop",
    "refresh",
    "wake",
    "update",
    "install",
    "exec",
    "login",
    "logout",
    "login-status",
    "status",
    "mcp",
    "search",
    "fetch",
)

#: Modules that exist on POSIX and not on Windows. A module-level import of one
#: of these makes the importing module -- and every module that imports it --
#: unimportable on Windows. `ptyprocess` is deliberately absent: it is a
#: third-party package that would have to be a dependency to be a defect here.
POSIX_ONLY_MODULES = ("pty", "fcntl", "termios", "pwd", "grp", "resource", "tty", "crypt")


@dataclass
class Result:
    """One probe's outcome. `status` is PASS / FAIL / WARN / SKIP."""

    name: str
    status: str
    detail: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Harness plumbing
# --------------------------------------------------------------------------- #


def _sandbox_root() -> Path:
    """A scratch root outside the repository and outside the live config."""
    return Path(tempfile.mkdtemp(prefix="lop-xplat-probe-"))


def isolated_env(root: Path) -> dict[str, str]:
    """Environment with HOME and every derived root pointed at `root`.

    `LOCAL_OPERATOR_CONFIG_DIR` alone is not enough here for the same reason it
    is not enough in the test suite: cache and agent-home roots derive from HOME
    independently. `USERPROFILE` is included because that, not `HOME`, is what
    `os.path.expanduser` consults on Windows.
    """
    env = dict(os.environ)
    # Inherited cmux state is worse than noise: a booted TUI can RENAME the
    # operator's real cmux workspaces from an inherited CMUX_WORKSPACE_ID.
    for key in [k for k in env if k.startswith("CMUX_")]:
        env.pop(key, None)
    env["HOME"] = str(root)
    env["USERPROFILE"] = str(root)
    env["LOCAL_OPERATOR_CONFIG_DIR"] = str(root / "config")
    env["XDG_CONFIG_HOME"] = str(root / "config-home")
    env["XDG_CACHE_HOME"] = str(root / "cache-home")
    env["XDG_DATA_HOME"] = str(root / "data-home")
    env["XDG_STATE_HOME"] = str(root / "state-home")
    env["TMPDIR"] = str(root / "tmp")
    env["TEMP"] = str(root / "tmp")
    env["TMP"] = str(root / "tmp")
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    # NOTHING IN THIS BATTERY MAY REACH THE macOS KEYCHAIN. Under a redirected
    # HOME a keychain client carries no bundle identity, so macOS offers to
    # CREATE a login keychain and puts up a "Keychain Not Found - A keychain
    # cannot be found to store ..." dialog on the operator's screen, once per
    # launch. `git` makes every credential lookup a keychain call here (the
    # system gitconfig sets `credential.helper = osxkeychain`), and `gh` keeps
    # its token there. Redirecting their config and refusing interactive
    # prompts removes the whole class, and costs the battery nothing: none of
    # its probes needs a git or GitHub credential.
    env["GIT_CONFIG_SYSTEM"] = "/dev/null"
    env["GIT_CONFIG_GLOBAL"] = str(root / "gitconfig-none")
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_ASKPASS"] = "/bin/false"
    env["GH_CONFIG_DIR"] = str(root / "gh-config")
    env["GH_TOKEN"] = ""
    env["GLAB_CONFIG_DIR"] = str(root / "glab-config")
    env["SSH_AUTH_SOCK"] = ""
    # Every child here is a Python program writing to a PIPE, so it encodes its
    # output with the LOCALE codec -- cp1252 on the Windows runner. That cost
    # run 35405383805 two false FAILs, in both directions: `config.roundtrip`
    # failed to DECODE the child's bytes in this parent ("er maps to
    # <undefined>"), and `tui.boot` failed to ENCODE them in the child
    # (UnicodeEncodeError inside the child's own cp1252.py, writing the logo's
    # block characters). Pin BOTH ends to UTF-8 -- the child here, this parent
    # via `run()`/`_spawn_child()` -- because the battery measures platform
    # mechanisms, and a console codec is not one of them.
    env["PYTHONIOENCODING"] = "utf-8"
    # AND THE FILE-OPENING CODEC, which is a DIFFERENT knob. `PYTHONIOENCODING`
    # names the stdio codec only: `open()` with no `encoding=` still asks the
    # locale, so the `tui.boot` child's `save_screenshot` write died on Windows
    # with `UnicodeEncodeError: 'charmap' codec can't encode character '\u2584'`
    # while every `print()` in the same child was already UTF-8. UTF-8 mode is
    # what moves `open()` as well, and it is the half the comment above this
    # paragraph already claimed was handled.
    env["PYTHONUTF8"] = "1"
    # AND UNBUFFERED, so a long-lived child's log is evidence WHILE it runs. A
    # child whose stdout is a FILE gets block buffering, so `lop serve` on its
    # way to binding writes nothing until it flushes or exits -- which is why
    # the failure detail for a timing-out server carried an EMPTY child output
    # and could not separate "still importing" from "crashed" (QA round 1, Q1).
    env["PYTHONUNBUFFERED"] = "1"
    # A child whose stdout is a FILE (see `ChildRun`) is block-buffered, so
    # everything it printed on the way to a crash or a timeout would still be
    # sitting in its buffer -- which is exactly the evidence the daemon and
    # server probes exist to capture, and exactly what run 35405383805 could not
    # explain about `serve.health`, `serve.double_bind` and
    # `mobile.daemon_serve`.
    env["PYTHONUNBUFFERED"] = "1"
    # A TUI that thinks it has no colour exercises a different render path, and
    # the probe is meant to measure the render path a user actually gets.
    env.pop("NO_COLOR", None)
    env.setdefault("TERM", "xterm-256color")
    return env


def _cli_argv(*args: str) -> list[str]:
    """The CLI invoked through the interpreter, not through a console script.

    The console-script shim is a `pip` artefact (a shell script on POSIX, an
    `.exe` launcher on Windows). Invoking the entry point directly keeps this
    probe honest about *our* code on a host where the package is present but no
    script was installed, and keeps one probe battery usable from both a
    container and a CI checkout.
    """
    driver = "import sys; from local_operator.cli import main; sys.exit(main())"
    return [sys.executable, "-c", driver, *args]


def run(
    argv: list[str],
    env: dict[str, str],
    *,
    timeout: float = 60.0,
    cwd: Path | None = None,
    stdin: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a child with a hard timeout, never inheriting this process's tty.

    `encoding`/`errors` are named rather than left to `locale.getpreferredencoding()`
    on purpose. `errors="replace"` is the load-bearing half: a probe's detail
    line is EVIDENCE, and a child that emits one byte the codec of the day cannot
    decode must not turn a working surface into a FAIL whose detail is a codec
    problem (run 35405383805: `config.roundtrip` FAILed with "er maps to
    <undefined>", which is cp1252, not lop).
    """
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        cwd=str(cwd or REPO),
        input=stdin,
        timeout=BUDGET.clamp(timeout),
    )


def _tail(text: str, limit: int = 400) -> str:
    """Last non-empty lines of `text`, truncated -- evidence, not a transcript."""
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    joined = "\n".join(lines[-4:])
    return joined[-limit:] if len(joined) > limit else joined


def _first_line(text: str, limit: int = 240) -> str:
    for line in (text or "").splitlines():
        if line.strip():
            return line.strip()[:limit]
    return ""


# --------------------------------------------------------------------------- #
# The battery's own wall-clock budget
# --------------------------------------------------------------------------- #


class BudgetSpent(Exception):
    """The battery's own budget ran out while a probe was still running.

    Deliberately NOT a `subprocess.TimeoutExpired`: a probe's timeout is a
    statement about the surface ("it did not answer in 60 s"), while the budget
    is a statement about the RUN ("there was no time left to ask"). Reporting
    the second as the first turns a truncated run into a list of fabricated
    defects, which is the one way this gate could do more harm than good.
    """


class Budget:
    """The whole battery's wall-clock ceiling, and the arithmetic around it.

    The per-probe timeouts SUM to ~64 minutes (3836 s at the time of writing)
    against job ceilings of 25 and 45 minutes, so three or four probes hanging to
    their own documented bound would have the JOB killed -- and a job killed by
    `timeout-minutes` prints no matrix and uploads no artifact, which is the
    exact failure the `if: always()` upload step exists to prevent (reviewer B,
    A2). The way out is not to lengthen the legs until they fit the worst case:
    it is for the battery to RETURN before the ceiling, having marked the probes
    it never reached as not-run rather than as failures.

    0 (the default) means no aggregate bound, which is what a local run wants --
    there is no job ceiling to beat, and the per-probe timeouts already bound it.
    """

    def __init__(self, seconds: float = 0.0) -> None:
        self.seconds = 0.0
        self.deadline: float | None = None
        self.start(seconds)

    def start(self, seconds: float) -> None:
        self.seconds = float(seconds or 0.0)
        self.deadline = time.monotonic() + self.seconds if self.seconds > 0 else None

    def left(self) -> float:
        """Seconds until the battery must stop; `inf` when unbounded."""
        if self.deadline is None:
            return float("inf")
        return self.deadline - time.monotonic()

    def spent(self) -> bool:
        return self.left() <= 0

    def window(self, seconds: float) -> float:
        """`seconds` of polling, or less where the budget ends first.

        Used by the probes that poll a long-lived child instead of calling
        `run()`, so that the budget is a bound on the whole battery and not only
        on the children started through `run()`.
        """
        return max(min(seconds, self.left()), 0.0)

    def clamp(self, timeout: float) -> float:
        """`timeout`, shortened to what the budget has left.

        Raises `BudgetSpent` once there is nothing left, so a probe cannot be
        STARTED on borrowed time and report the resulting instant timeout as a
        defect in the surface it was measuring.
        """
        if self.deadline is None:
            return timeout
        left = self.left()
        if left <= 0:
            raise BudgetSpent(f"the battery's {int(self.seconds)}s budget was spent")
        return min(timeout, left)

    def note(self) -> str:
        """A suffix for a failure line when the BUDGET, not the surface, ran out."""
        if self.deadline is not None and self.spent():
            return f" (the battery's {int(self.seconds)}s budget was spent first)"
        return ""


#: Set once, from `--budget`, before the first probe runs. Module-level because
#: `run()` and the probes' poll loops read it directly; a parameter threaded
#: through twenty probe signatures would be forgotten at the twenty-first.
BUDGET = Budget(0.0)


# --------------------------------------------------------------------------- #
# Probes
# --------------------------------------------------------------------------- #


def probe_static_posix_imports(env: dict[str, str]) -> Result:
    """Every module that imports a POSIX-only module at MODULE level.

    Module level, not inside a function and not inside a `try:` -- a guarded
    import is a handled platform difference and reporting it would drown the
    real defect. The parsing is `ast`-based rather than regex-based because a
    regex cannot tell an import from a mention in a docstring, and this report
    is only useful if every line on it is real.
    """
    import ast

    offenders: list[str] = []
    guarded: list[str] = []
    for path in sorted((REPO / "local_operator").rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError) as exc:  # pragma: no cover
            offenders.append(f"{path.relative_to(REPO)}: unparseable ({exc})")
            continue
        rel = path.relative_to(REPO)
        for node in tree.body:  # top level only
            if isinstance(node, ast.Try):
                for child in ast.walk(node):
                    for name in _posix_import_names(child):
                        guarded.append(f"{rel}:{node.lineno} (guarded) {name}")
                continue
            for name in _posix_import_names(node):
                offenders.append(f"{rel}:{getattr(node, 'lineno', 0)} import {name}")
    detail = f"{len(offenders)} unguarded, {len(guarded)} guarded"
    return Result(
        "static.posix_imports",
        "WARN" if offenders else "PASS",
        detail,
        {"unguarded": offenders, "guarded_imports": len(guarded)},
    )


def _posix_import_names(node: object) -> list[str]:
    """Names from `import x` / `from x import y` nodes that are POSIX-only."""
    import ast

    found: list[str] = []
    if isinstance(node, ast.Import):
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root in POSIX_ONLY_MODULES:
                found.append(alias.name)
    elif isinstance(node, ast.ImportFrom):
        root = (node.module or "").split(".")[0]
        if root in POSIX_ONLY_MODULES:
            found.append(node.module or "")
    return found


#: Exception types a module raises when it deliberately REFUSES to be imported
#: on this platform. Kept to the types a refusal is actually written with here:
#: `RuntimeError` is the repository's spelling
#: (`evaluation/adapters/supervisor.py`), and the import/OS errors are what a
#: module that gates a dependency raises. A `TypeError` or an `AttributeError`
#: is never a refusal -- it is the module breaking -- so no marker can rescue it.
REFUSAL_EXCEPTIONS = frozenset(
    {"RuntimeError", "NotImplementedError", "ImportError", "ModuleNotFoundError", "OSError"}
)

#: Words that say the PLATFORM is the reason. A refusal without one of these
#: does not count: "it broke" is not "it declined", and an unexplained failure
#: is exactly what this probe exists to surface.
REFUSAL_PLATFORM_MARKERS = (
    "posix",
    "windows",
    "linux",
    "darwin",
    "macos",
    "mac os",
    "this platform",
    "not available on",
    "unsupported platform",
    "requires systemd",
    "requires launchd",
)


def _classify_import_failures(failed: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    """Split `{module: "Type: message"}` into (refused, unexpected).

    Both halves are reported; only the second fails the probe. The distinction
    has to be made on the message, because that is all an import can tell us --
    a module that refuses and a module that crashes are the same event from the
    outside, and the difference lives in the sentence it chose.
    """
    refused: dict[str, str] = {}
    unexpected: dict[str, str] = {}
    for module, report in failed.items():
        exc_type = report.split(":", 1)[0].strip()
        lowered = report.lower()
        if exc_type in REFUSAL_EXCEPTIONS and any(
            marker in lowered for marker in REFUSAL_PLATFORM_MARKERS
        ):
            refused[module] = report
        else:
            unexpected[module] = report
    return refused, unexpected


def probe_import_package(env: dict[str, str]) -> Result:
    """Import EVERY submodule of the package, and report what would not import.

    This is the single most informative probe in an OS-portability battery: it
    turns "does lop run on this OS" into a list of module names, and it catches
    the module that is only imported on a rarely-taken branch.

    ...but only the SECOND of its two failure kinds is a defect. A module that
    REFUSES to be imported here, with a named platform reason, is the deliberate
    shape this branch standardises -- `evaluation/adapters/supervisor.py` raises
    `RuntimeError: evaluation adapter supervision requires POSIX process groups`
    and five modules import it -- and failing on it would make this probe red on
    Windows forever for a design decision (reviewer B, A4/Q1).
    """
    driver = textwrap.dedent("""
        import importlib, json, pkgutil, sys, traceback
        import local_operator
        failed = {}
        count = 0
        for mod in pkgutil.walk_packages(local_operator.__path__, "local_operator."):
            count += 1
            try:
                importlib.import_module(mod.name)
            except BaseException as exc:  # noqa: BLE001 - report, never raise
                failed[mod.name] = f"{type(exc).__name__}: {exc}"
        print(json.dumps({"total": count, "failed": failed}))
        """)
    try:
        proc = run([sys.executable, "-c", driver], env, timeout=600.0)
    except subprocess.TimeoutExpired:
        return Result(
            "import.package",
            "FAIL",
            f"timed out importing the package{BUDGET.note()}",
        )
    if proc.returncode != 0:
        return Result(
            "import.package",
            "FAIL",
            _tail(proc.stderr) or "driver exited non-zero",
            {"returncode": proc.returncode},
        )
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    failed = payload["failed"]
    refused, unexpected = _classify_import_failures(failed)
    parts = [f"{payload['total'] - len(failed)}/{payload['total']} modules import"]
    if refused:
        parts.append(f"{len(refused)} refuse off POSIX with a named reason")
    if unexpected:
        parts.append(f"{len(unexpected)} unexpected")
    return Result(
        "import.package",
        "FAIL" if unexpected else "PASS",
        "; ".join(parts),
        {"failed": failed, "refused": refused, "unexpected": unexpected},
    )


def _source_version() -> str | None:
    """The version the SOURCE under test declares, from `pyproject.toml`.

    Read with a regex rather than `tomllib` so this stays stdlib-only on every
    Python the battery runs on, and read from `pyproject.toml` rather than
    `local_operator.__version__` because the file is what a release bumps.
    """
    try:
        root = Path(__file__).resolve().parent.parent
        text = (root / "pyproject.toml").read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return match.group(1) if match else None


def probe_cli_version(env: dict[str, str]) -> Result:
    """The CLI's own version, AND the version of the code actually under test.

    `lop --version` reports the INSTALLED distribution's metadata. On this
    harness that is a separate non-editable `uv tool` install, so on exactly the
    runs this battery exists for -- a worktree, a branch, a container -- the
    command can print a version that is NOT the source being exercised (QA round
    1, Q2 measured `lop --version` printing 0.59.2 while the battery exercised
    0.59.9). Version is the one line a reader compares across operating systems,
    so a mismatch is reported rather than printed as though it settled the
    question.
    """
    proc = run(_cli_argv("--version"), env, timeout=120.0)
    if proc.returncode != 0:
        return Result("cli.version", "FAIL", _tail(proc.stderr), {"rc": proc.returncode})
    reported = _first_line(proc.stdout + proc.stderr)
    source = _source_version()
    extra: dict[str, object] = {"reported": reported, "source": source}
    if source is not None and source not in reported:
        return Result(
            "cli.version",
            "WARN",
            f"{reported} — the CLI reports the INSTALLED distribution, and this "
            f"checkout declares {source}; the reading above is not about this source",
            extra,
        )
    return Result("cli.version", "PASS", reported, extra)


def probe_cli_help(env: dict[str, str]) -> Result:
    proc = run(_cli_argv("--help"), env, timeout=120.0)
    if proc.returncode != 0:
        return Result("cli.help", "FAIL", _tail(proc.stderr), {"rc": proc.returncode})
    return Result("cli.help", "PASS", f"{len(proc.stdout.splitlines())} lines of usage")


def probe_cli_subcommands(env: dict[str, str]) -> Result:
    """`--help` for every subcommand. A parser that raises is a hard user-block.

    `--help` is used rather than a real invocation because it reaches the
    subparser's own registration code without needing credentials, a model, or
    network -- and a subcommand whose parser cannot even render is broken for
    everybody, not just for this probe.
    """
    failed: dict[str, str] = {}
    for name in SUBCOMMANDS:
        try:
            proc = run(_cli_argv(name, "--help"), env, timeout=120.0)
        except subprocess.TimeoutExpired:
            failed[name] = "timed out"
            continue
        if proc.returncode != 0:
            failed[name] = _first_line(proc.stderr) or f"exit {proc.returncode}"
    detail = f"{len(SUBCOMMANDS) - len(failed)}/{len(SUBCOMMANDS)} render"
    return Result(
        "cli.subcommands",
        "FAIL" if failed else "PASS",
        detail,
        {"failed": failed},
    )


def probe_paths_roots(env: dict[str, str]) -> Result:
    """Where lop resolves its roots on this OS -- the map the other probes read."""
    driver = textwrap.dedent("""
        import json, os
        from local_operator import paths
        out = {}
        for name in dir(paths):
            if name.startswith("_"):
                continue
            obj = getattr(paths, name)
            if callable(obj):
                try:
                    obj = obj()
                except Exception:
                    continue
            if isinstance(obj, (str, os.PathLike, type(None))):
                out[name] = str(obj) if obj is not None else None
        # Path-typed module attributes (roots are usually constants, not calls).
        for name, obj in vars(paths).items():
            if name.startswith("_") or name in out:
                continue
            if isinstance(obj, os.PathLike):
                out[name] = str(obj)
        print(json.dumps(out, default=str))
        """)
    proc = run([sys.executable, "-c", driver], env, timeout=120.0)
    if proc.returncode != 0:
        return Result("paths.roots", "FAIL", _tail(proc.stderr))
    roots = json.loads(proc.stdout.strip().splitlines()[-1])
    return Result("paths.roots", "PASS", f"{len(roots)} roots resolved", {"roots": roots})


def probe_config_roundtrip(env: dict[str, str]) -> Result:
    """Create a config file and read one value back, through the CLI."""
    created = run(_cli_argv("config", "create"), env, timeout=120.0, stdin="")
    listed = run(_cli_argv("config", "list"), env, timeout=120.0)
    if listed.returncode != 0:
        return Result(
            "config.roundtrip",
            "FAIL",
            _tail(listed.stderr),
            {"create_rc": created.returncode},
        )
    config_file = Path(env["LOCAL_OPERATOR_CONFIG_DIR"]) / "config.yml"
    return Result(
        "config.roundtrip",
        "PASS",
        f"create rc={created.returncode}, {len(listed.stdout.splitlines())} options"
        + (f", config.yml={'yes' if config_file.exists() else 'no'}"),
        {"created": created.returncode},
    )


def probe_secret_roundtrip(env: dict[str, str]) -> Result:
    """Store a secret and read it back. Exercises the encrypted store end to end.

    The value is a throwaway in an isolated root, so nothing sensitive is
    involved; what is being measured is whether the store can seal and open a
    record at all on this OS, which is where a keyring/crypto platform arm
    shows up.
    """
    status = run(_cli_argv("secret", "status"), env, timeout=120.0)
    if status.returncode != 0:
        return Result(
            "secret.roundtrip",
            "FAIL",
            f"secret status failed: {_first_line(status.stderr)}",
        )
    setp = run(_cli_argv("secret", "set", "xplat_probe"), env, timeout=180.0, stdin="probe-value")
    if setp.returncode != 0:
        return Result(
            "secret.roundtrip",
            "FAIL",
            f"secret set failed: {_first_line(setp.stderr)}",
            {"status": _first_line(status.stdout)},
        )
    got = run(_cli_argv("secret", "get", "xplat_probe"), env, timeout=120.0)
    ok = got.returncode == 0 and got.stdout.strip() == "probe-value"
    return Result(
        "secret.roundtrip",
        "PASS" if ok else "FAIL",
        "sealed, reopened" if ok else f"get rc={got.returncode} value-mismatch",
    )


def probe_sessions_list(env: dict[str, str]) -> Result:
    proc = run(_cli_argv("sessions", "--json"), env, timeout=120.0)
    if proc.returncode != 0:
        return Result("sessions.list", "FAIL", _tail(proc.stderr))
    return Result("sessions.list", "PASS", _first_line(proc.stdout) or "empty roster")


def probe_wake_status(env: dict[str, str]) -> Result:
    """`wake status` -- and, crucially, WHICH supervisor it claims to have.

    The detail carries the whole line, because "supported: false" is the
    finding on an OS where the supervisor installer has no arm.

    THREE states, not two, and the third is the one this probe used to get
    wrong. `supervisor_available` was computed from a denylist of phrases, so
    the CLI's other answer -- "cannot be verified for this store", which is what
    a redirected HOME always produces -- contained none of them and was read as
    support being PRESENT: the "reports success while doing nothing" class this
    battery exists to catch, inside the battery (reviewer B, A5). Under this
    harness that is the NORMAL reading on macOS and Linux, so the WARN is not
    noise: the state is named, and a reader can see that no supervisor was
    confirmed rather than assuming one was found.
    """
    proc = run(_cli_argv("wake", "status"), env, timeout=120.0)
    text = (proc.stdout + proc.stderr).strip()
    if proc.returncode != 0:
        return Result("wake.status", "FAIL", _tail(proc.stderr))
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    line = next(
        (ln for ln in lines if ln.lower().startswith("supervisor")), lines[0] if lines else ""
    )
    said = line.split(":", 1)[1].strip().lower() if ":" in line else line.lower()
    if any(marker in said for marker in ("not installed", "no supervisor")):
        state = "absent"
    elif any(
        marker in said for marker in ("cannot be verified", "not addressable", "could not be")
    ):
        state = "unverified"
    else:
        state = "reported"
    return Result(
        "wake.status",
        "PASS" if state == "reported" else "WARN",
        line[:240] or "no output",
        {
            "raw": text[:1200],
            # A boolean for the readers that grep it, and False for BOTH
            # non-answers: "no supervisor" and "could not be confirmed" are
            # equally not a supervisor that can fire a wake with no TUI open.
            "supervisor_available": state == "reported",
            "supervisor_state": state,
        },
    )


def probe_wake_install(env: dict[str, str]) -> Result:
    """Install the wake supervisor. This is the scheduling mechanism under test.

    A platform with no arm here cannot fire a scheduled wake unattended: the
    supervisor is the only thing that runs when no TUI is open, so `wake
    create` succeeds and the wake then never fires. That is a FAIL, not a WARN
    -- it is the whole feature missing on that OS, reported as success.
    """
    proc = run(_cli_argv("wake", "install"), env, timeout=180.0)
    text = (proc.stdout + proc.stderr).strip()
    unavailable = _platform_unavailable(text)
    if unavailable:
        return Result(
            "wake.install",
            "FAIL",
            f"no supervisor on this OS ({unavailable!r}): {_first_line(text)}",
            {"raw": text[:1200]},
        )
    if proc.returncode != 0:
        return Result(
            "wake.install",
            "FAIL",
            _first_line(text) or f"exit {proc.returncode}",
            {"raw": text[:1200]},
        )
    return Result("wake.install", "PASS", _first_line(text) or "installed", {"raw": text[:1200]})


def probe_mobile_status(env: dict[str, str]) -> Result:
    """`mobile status` must ANSWER, whatever the answer is.

    A non-zero exit is right for "not installed", so this probe reports on
    whether the command spoke -- `mobile.install` is the probe that owns the
    platform finding, and double-reporting it here would make one defect look
    like three.
    """
    proc = run(_cli_argv("mobile", "status"), env, timeout=120.0)
    text = (proc.stdout + proc.stderr).strip()
    if "installed" in text.lower() or "daemon" in text.lower():
        return Result("mobile.status", "PASS", _first_line(text), {"rc": proc.returncode})
    return Result(
        "mobile.status",
        "FAIL",
        _first_line(text) or f"exit {proc.returncode} with no message",
        {"rc": proc.returncode, "raw": text[:800]},
    )


def _mobile_daemon_log_tail(env: dict[str, str], limit: int = 2000) -> str:
    """The supervised daemon's OWN log, for when the installer says it never came up.

    ``lop mobile install`` verifies the daemon by polling ``/healthz`` and, on
    timeout, answers ``daemon did not come up healthy; see <log>`` — a pointer,
    not the evidence. On the Windows runner that pointer was the whole of what
    the artifact carried, so a real failure arrived with no readable cause and
    the probe's own detail line degenerated into a slice of a PATH dump.
    Derived from ``LOCAL_OPERATOR_CONFIG_DIR`` rather than imported from
    ``local_operator.paths``, because this script stays stdlib-only.
    """
    config = env.get("LOCAL_OPERATOR_CONFIG_DIR")
    if not config:
        return ""
    path = Path(config) / "logs" / "mobile.log"
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"(no daemon log at {path}: {exc})"
    # Lines a reader cannot use are dropped rather than truncated: the Windows
    # daemon logs the registry PATH it retrieved, several hundred characters on
    # one line, and keeping it pushed the actual failure out of the tail.
    kept = [line for line in raw.splitlines() if len(line) <= 300]
    return "\n".join(kept)[-limit:]


#: The mobile daemon's Task Scheduler name. Spelled here rather than imported
#: because this script stays stdlib-only and must run without the package it is
#: measuring; the installer's `TASK_NAME` is the authority and the two are a
#: pair, so a rename has to touch both.
MOBILE_TASK_NAME = "Local Operator Mobile"

#: The `schtasks /V /FO CSV` columns worth carrying. Read by HEADER NAME out of
#: a positional CSV rather than by matching localized field labels, and a header
#: that matches none of them is reported rather than silently returning nothing
#: (agent review round 2, n2 -- the same defect class as the product's own
#: status parse, A6).
_TASK_STATE_FIELDS = (
    "Status",
    "Last Run Time",
    "Last Result",
    "Task To Run",
    "Run As User",
    "Scheduled Task State",
)


def _supervised_task_state(env: dict[str, str]) -> str:
    """What Task Scheduler thinks of the task, when there is one to ask about.

    The daemon log answers "what did it say"; this answers "is it even still
    running", which is the difference between a daemon that died and one that is
    merely slow on a cold runner.

    `/FO CSV` rather than `/FO LIST` because the LIST field NAMES are localized
    ("Statut:", "Estado:", ":") and matching them in English makes a
    non-English Windows read as "no fields at all" -- answering a question it
    could not read. CSV fixes the field ORDER across locales.
    """
    if sys.platform != "win32":
        return ""
    try:
        query = subprocess.run(
            ["schtasks", "/Query", "/TN", MOBILE_TASK_NAME, "/V", "/FO", "CSV"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"(schtasks query failed: {exc})"
    body = (query.stdout or "").strip()
    if not body:
        stderr = (query.stderr or "").strip()[:200]
        return f"(schtasks returned nothing; stderr: {stderr or 'empty'})"
    try:
        rows = list(csv.reader(io.StringIO(body)))
    except csv.Error as exc:
        return f"(schtasks output was not CSV: {exc})"
    if len(rows) < 2:
        return f"(schtasks output had no data row: {body[:200]})"
    headers, values = rows[0], rows[1]
    pairs = [
        f"{header.strip()}={value.strip()}"
        for header, value in zip(headers, values, strict=False)
        if header.strip() in _TASK_STATE_FIELDS
    ]
    if not pairs:
        return f"(no recognised fields; headers: {headers[:8]})"
    return "; ".join(pairs)


def probe_mobile_install(env: dict[str, str]) -> Result:
    """Install the phone-portal daemon -- the background-runner surface.

    DELIBERATELY NOT RUN ON macOS. `lop mobile install` stores the portal
    password through the `security` CLI, and under the redirected HOME this
    battery runs in, macOS has no login keychain to store it in -- so the call
    raises a "Keychain Not Found" DIALOG on the operator's screen, once per
    run. Measured, and reported to this session by a peer that traced the
    dialog to isolated-home runs. On darwin the installer is also the one arm
    that is already known to work on a real machine, so the reading it would
    give here is worth less than the dialog costs. Pass `--allow-keychain` to
    run it anyway on a machine where that is acceptable.
    """
    if sys.platform == "darwin" and not env.get("LOP_XPLAT_ALLOW_KEYCHAIN"):
        return Result(
            "mobile.install",
            "SKIP",
            "not run on macOS: the installer writes to the login keychain, which an "
            "isolated HOME does not have (it would raise a keychain dialog). "
            "Use --allow-keychain to force it.",
        )
    proc = run(_cli_argv("mobile", "install"), env, timeout=180.0)
    text = (proc.stdout + proc.stderr).strip()
    unavailable = _platform_unavailable(text)
    if unavailable:
        return Result(
            "mobile.install",
            "FAIL",
            f"no daemon installer on this OS ({unavailable!r}): {_first_line(text)}",
            {"raw": text[:1200]},
        )
    if proc.returncode != 0:
        # A macOS Keychain absent from an ISOLATED home is an artefact of the
        # sandbox, not a defect: `security` addresses the login keychain, and a
        # freshly made HOME has none. Reported as SKIP-with-reason so that a
        # genuine macOS failure is not drowned by a probe artefact -- the same
        # mistake in the other direction is the one that hides a real bug.
        if sys.platform == "darwin" and "keychain" in text.lower():
            return Result(
                "mobile.install",
                "SKIP",
                "macOS keychain is unreachable from the isolated HOME the probe uses",
                {"raw": text[:600]},
            )
        # A DOCUMENTED PREREQUISITE THAT IS ABSENT is a host state, not a
        # defect -- the distinction `tunnel.status` already draws for "not
        # configured". The portal bundle is built once, with Node >= 22, and the
        # installer now names exactly that and how to get it. Reported as SKIP
        # with the installer's own sentence so the reason travels; if the
        # prerequisite IS present (`node` on PATH) and the install still fails,
        # this stays a FAIL -- the case that must never be hidden behind a
        # friendly message.
        # AN INSTALL THAT REFUSED TO ADDRESS THE *REAL* SUPERVISOR is the same
        # class as the macOS keychain above: correct behaviour that an isolated
        # HOME makes unavoidable. `systemctl --user` has no sandbox -- it reaches
        # the calling user's live manager whatever $HOME says -- so the installer
        # deliberately declines to enable a unit the real home does not own, and
        # a redirected-home run therefore can never reach the enable step. The
        # file half is still proven (the step lines are in the detail); the
        # enable half is proven by the sibling `wake.install` probe and by the
        # real-systemd run recorded in docs/XPLATFORM.md. Reported as SKIP with
        # the installer's own sentence, so the reason travels with the reading.
        if "refusing to enable it from a redirected home" in text:
            return Result(
                "mobile.install",
                "SKIP",
                "the installer wrote its unit and declined to enable it from a "
                "redirected HOME, which cannot reach the real user manager: " + _first_line(text),
                {"raw": text[:1200]},
            )
        prerequisite_missing = "is not installed" in text and (
            "nodejs.org" in text or "install node" in text.lower()
        )
        if prerequisite_missing and shutil.which("node") is None:
            return Result(
                "mobile.install",
                "SKIP",
                "portal bundle needs Node >= 22, absent on this host: " + _tail(text),
                {"raw": text[:1200]},
            )
        # THE DAEMON'S OWN LOG, not just the pointer to it. "did not come up
        # healthy; see <path>" is a location, and the artifact is read on
        # another machine days later; without the log a supervised start that
        # failed is indistinguishable from one that was slow, and the reader has
        # nothing to act on. `_tail` of the log (the end is where a crash is),
        # plus Task Scheduler's own view of the task on Windows.
        extra: dict[str, object] = {"rc": proc.returncode, "raw": text[:2000]}
        detail = _tail(text) or f"exit {proc.returncode}"
        if "did not come up healthy" in text:
            daemon_log = _mobile_daemon_log_tail(env)
            if daemon_log:
                extra["daemon_log"] = daemon_log
            task_state = _supervised_task_state(env)
            if task_state:
                extra["task_state"] = task_state
            # The LAST meaningful daemon line is the cause; the installer's own
            # progress lines and the registry PATH dumps are not.
            cause = _tail(daemon_log, limit=400) if daemon_log else "(no daemon log)"
            detail = f"the supervised daemon never became healthy: {cause}"
        return Result(
            "mobile.install",
            "FAIL",
            detail,
            extra,
        )
    # The LAST step, not the first: a successful install prints its progress in
    # order, so `_first_line` reported "generated a new portal password" -- a
    # step that happens BEFORE the bundle is built and the task registered --
    # as the outcome (agent review round 2, n1).
    steps = [line.strip() for line in text.splitlines() if line.strip()]
    return Result(
        "mobile.install",
        "PASS",
        steps[-1] if steps else "installed",
        {"raw": text[:1200]},
    )


def probe_mobile_daemon_serve(env: dict[str, str]) -> Result:
    """Boot the phone-portal daemon in the foreground and prove it serves.

    The daemon is the mechanism that has to survive on a headless Linux box, so
    the probe starts it for real and reads its health endpoint rather than
    trusting `mobile status`, which reports on an installed unit.
    """
    import urllib.error
    import urllib.request

    port = _free_port()
    # The daemon refuses to start without a password, and `mobile install` --
    # which is what normally sets one -- has no arm on this platform. Setting
    # it in the environment keeps this probe measuring THE DAEMON rather than
    # measuring the installer a second time.
    env = {**env, "LOP_MOBILE_PASSWORD": "xplat-probe-not-a-credential"}
    window = BUDGET.window(120.0)
    with _spawn_cli(
        ["mobile", "serve", "--port", str(port)], env, log=_child_log(env, "mobile-serve")
    ) as child:
        deadline = time.monotonic() + window
        last = ""
        while time.monotonic() < deadline:
            if child.proc.poll() is not None:
                return Result(
                    "mobile.daemon_serve",
                    "FAIL",
                    f"daemon exited rc={child.proc.returncode}: {child.output() or last}",
                    child.extra(),
                )
            for path in ("/healthz", "/health"):
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}{path}", timeout=2
                    ) as response:
                        body = response.read(200).decode("utf-8", errors="replace")
                        return Result(
                            "mobile.daemon_serve",
                            "PASS",
                            f"{path} -> {response.status} {body[:80]}",
                            child.extra(),
                        )
                except urllib.error.HTTPError as exc:
                    # A 401 from the gate is a served daemon, not a failure.
                    return Result(
                        "mobile.daemon_serve",
                        "PASS",
                        f"{path} -> {exc.code} (auth gate answering)",
                        child.extra(),
                    )
                except Exception as exc:  # noqa: BLE001 - keep polling
                    last = f"{type(exc).__name__}: {exc}"
            time.sleep(0.5)
        # On run 35405383805 this probe reported only the timeout, so the
        # Windows failure it found ("mobile.install" registered the scheduled
        # task and the daemon never came up) had no child output to read. The
        # tail is what makes the NEXT run diagnostic rather than merely red.
        return Result(
            "mobile.daemon_serve",
            "FAIL",
            _no_response(child, window, last),
            child.extra(),
        )


def probe_tunnel_status(env: dict[str, str]) -> Result:
    proc = run(_cli_argv("tunnel", "status"), env, timeout=120.0)
    text = (proc.stdout + proc.stderr).strip()
    if proc.returncode != 0:
        # "No tunnel configured" is a HOST STATE, not a platform defect. Coding
        # it as FAIL would put a red cell in every column of an unconfigured
        # machine and teach a reader to ignore the column.
        if "not configured" in text.lower() or "lop tunnel create" in text:
            return Result("tunnel.status", "SKIP", "no tunnel configured on this host")
        return Result("tunnel.status", "FAIL", _first_line(text), {"rc": proc.returncode})
    return Result("tunnel.status", "PASS", _first_line(text) or "ok")


def probe_tunnel_install(env: dict[str, str]) -> Result:
    """Install the tunnel service -- which fetches the vendor binary.

    Network-dependent, so a failure here is reported as SKIP-with-reason when
    the reason is a download rather than a platform arm: conflating the two
    would make an offline runner look like a portability defect.
    """
    proc = run(_cli_argv("tunnel", "install"), env, timeout=300.0)
    text = (proc.stdout + proc.stderr).strip()
    if proc.returncode != 0:
        lowered = text.lower()
        if "not configured" in lowered or "lop tunnel create" in lowered:
            return Result("tunnel.install", "SKIP", "no tunnel configured on this host")
        offline = any(
            marker in lowered
            for marker in ("connection", "timed out", "temporary failure", "ssl", "network")
        )
        return Result(
            "tunnel.install",
            "SKIP" if offline else "FAIL",
            _first_line(text) or f"exit {proc.returncode}",
            {"raw": text[:1200]},
        )
    return Result("tunnel.install", "PASS", _first_line(text) or "installed", {"raw": text[:800]})


def probe_serve_health(env: dict[str, str]) -> Result:
    """Stand the HTTP API up for real and read a real endpoint."""
    import urllib.error
    import urllib.request

    port = _free_port()
    # 180 and not 60: a fixed 60s window produced a FALSE FAIL on the release
    # platform under host load (QA round 1, Q1). `lop serve` was measured
    # healthy at 25.7s on this hardware, and on a machine running ~25 sessions
    # the same code and the same command read "no response in 60s" on one run
    # and FAIL=0 on the next. A window that can go red under load alone is a
    # gate nobody will keep, and the budget still bounds it.
    window = BUDGET.window(180.0)
    with _spawn_cli(
        ["serve", "--port", str(port)], env, log=_child_log(env, "serve-health")
    ) as child:
        deadline = time.monotonic() + window
        last = ""
        while time.monotonic() < deadline:
            if child.proc.poll() is not None:
                return Result(
                    "serve.health",
                    "FAIL",
                    f"server exited rc={child.proc.returncode}: {child.output() or last}",
                    child.extra(),
                )
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=2
                ) as response:
                    body = response.read(120).decode("utf-8", errors="replace")
                    return Result(
                        "serve.health",
                        "PASS",
                        f"/health -> {response.status} {body}",
                        child.extra(),
                    )
            except urllib.error.HTTPError as exc:
                return Result("serve.health", "FAIL", f"/health -> HTTP {exc.code}", child.extra())
            except Exception as exc:  # noqa: BLE001 - keep polling
                last = f"{type(exc).__name__}: {exc}"
            time.sleep(0.5)
        # The child's own output is the difference between "the port never
        # answered" and WHY (a bind error, a missing dependency, a traceback):
        # the tail is what run 35405383805's artifact could not supply.
        return Result(
            "serve.health",
            "FAIL",
            _no_response(child, window, last),
            child.extra(),
        )


def probe_tui_boot(env: dict[str, str]) -> Result:
    """Boot the real Textual app in an in-memory compositor and screenshot it.

    `run_test()` is not a visual proof -- it paints into a compositor, not a
    terminal -- but it IS a proof that the widget tree assembles, the stylesheet
    loads and the app reaches a settled first frame on this OS. The frame is
    written out so a human can look at it; `--json` records that it was written.
    """
    out = Path(env.get("LOP_XPLAT_SHOT_DIR", env["HOME"])) / "tui-boot.svg"
    proc = run(
        [sys.executable, str(Path(__file__).resolve()), "--driver", "tui", str(out)],
        env,
        timeout=300.0,
    )
    text = (proc.stdout + proc.stderr).strip()
    if proc.returncode == 3 and "TUI_DRIVER_UNAVAILABLE" in text:
        return Result(
            "tui.boot",
            "SKIP",
            "the test suite (which owns the app's fake session) is not in this checkout",
        )
    if proc.returncode != 0 or not out.exists():
        return Result("tui.boot", "FAIL", _tail(text) or f"exit {proc.returncode}")
    return Result(
        "tui.boot",
        "PASS",
        f"settled frame written ({out.stat().st_size} bytes SVG)",
        {"screenshot": str(out)},
    )


def probe_tui_driver_tty(env: dict[str, str]) -> Result:
    """Boot the console script against a REAL pty and type into it.

    This is the honest TUI probe: it is the only one that exercises the
    terminal driver, the input thread and the raw-mode setup, which is where a
    platform difference in terminal handling actually lives. Windows gets its
    own console API (ConPTY) and no `termios`, so a difference here is exactly
    what we are looking for.
    """
    driver = textwrap.dedent("""
        import os, pty, select, signal, sys, time
        pid, fd = pty.fork()
        if pid == 0:
            os.execv(sys.executable, [sys.executable, "-c",
                "import sys; from local_operator.cli import main; sys.exit(main())"])
        sent = False
        buf = b""
        deadline = time.monotonic() + 75
        try:
            while time.monotonic() < deadline:
                r, _, _ = select.select([fd], [], [], 0.5)
                if r:
                    try:
                        chunk = os.read(fd, 65536)
                    except OSError:
                        break
                    if not chunk:
                        break
                    buf += chunk
                elif not sent and len(buf) > 2000:
                    os.write(fd, b"\\x03")  # Ctrl-C: reach the input path
                    sent = True
                    time.sleep(1.0)
                    break
            print("BYTES", len(buf))
            print(buf[-400:].decode("utf-8", "replace"))
        finally:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
            os.waitpid(pid, 0)
        """)
    if os.name != "posix":
        return Result(
            "tui.tty",
            "SKIP",
            "needs a POSIX pty; Windows has no pty module (ConPTY is a different API)",
        )
    try:
        proc = run([sys.executable, "-c", driver], env, timeout=180.0)
    except subprocess.TimeoutExpired:
        return Result("tui.tty", "FAIL", "pty-driven boot did not return in 180s")
    out = (proc.stdout or "").strip()
    if "BYTES" not in out:
        return Result("tui.tty", "FAIL", _tail(out or proc.stderr))
    count = int(out.split("BYTES", 1)[1].split()[0])
    if count < 500:
        return Result(
            "tui.tty",
            "FAIL",
            f"tty produced only {count} bytes before exiting",
            {"raw": out[-600:]},
        )
    return Result("tui.tty", "PASS", f"tty rendered {count} bytes", {"raw": out[-600:]})


def probe_exec_offline(env: dict[str, str]) -> Result:
    """`exec` with no provider configured: a clean refusal, never a traceback.

    A traceback here is a real user-facing defect on the platform -- it is the
    first thing a new user on that OS sees.
    """
    proc = run(_cli_argv("exec", "say hello"), env, timeout=180.0)
    text = (proc.stdout or "") + (proc.stderr or "")
    traceback = "Traceback (most recent call last)" in text
    if traceback:
        return Result("exec.offline", "FAIL", _tail(text), {"rc": proc.returncode})
    return Result(
        "exec.offline",
        "PASS",
        f"rc={proc.returncode}, no traceback",
        {"rc": proc.returncode, "out": text[-400:]},
    )


def probe_file_lock(env: dict[str, str]) -> Result:
    """Does lop's cross-process mutual exclusion work on this OS?

    Two processes contend for the same lease/lock; the second must NOT be
    granted it while the first holds it. A platform where the lock is a no-op
    reports success twice, and every "one owner" invariant in lop rests on it.
    """
    driver = textwrap.dedent("""
        import json, sys, time
        from pathlib import Path
        from local_operator.session_lease import (
            SessionLeaseHeldError, acquire_session_lease,
        )
        session_dir = Path(sys.argv[1])
        mode = sys.argv[2]
        if mode == "hold":
            lease = acquire_session_lease(session_dir)
            print(json.dumps({"acquired": True, "generation": lease.generation}), flush=True)
            time.sleep(6)
        else:
            time.sleep(1.5)
            try:
                acquire_session_lease(session_dir)
            except SessionLeaseHeldError as exc:
                print(json.dumps({"acquired": False, "error": str(exc)[:200]}))
            else:
                print(json.dumps({"acquired": True}))
        """)
    session_dir = Path(env["HOME"]) / "lease-probe"
    # `_spawn_child`, not a raw Popen: the holder's output is what a FAIL here
    # would have to be read against, and a `PIPE` this probe never drains is the
    # shape that lost the daemon probes their evidence on the Windows runner.
    holder = _spawn_child(
        [sys.executable, "-c", driver, str(session_dir), "hold"],
        env,
        log=_child_log(env, "lease-holder"),
    )
    try:
        time.sleep(1.0)
        second = run(
            [sys.executable, "-c", driver, str(session_dir), "wait"],
            env,
            timeout=60.0,
        )
        combined = (second.stdout or "") + (second.stderr or "")
        if "acquired" not in combined:
            return Result("lock.exclusive", "FAIL", _tail(combined), holder.extra())
        granted = json.loads(combined.strip().splitlines()[-1]).get("acquired")
        if granted:
            return Result(
                "lock.exclusive",
                "FAIL",
                "a second holder was granted the same lease while the first held it",
                holder.extra(),
            )
        return Result("lock.exclusive", "PASS", "second holder correctly refused", holder.extra())
    finally:
        holder.stop()


def probe_static_posix_attributes(env: dict[str, str]) -> Result:
    """Every USE of a POSIX-only attribute, and whether anything guards it.

    The module-level-import scan above cannot see the worse half of this class:
    `os.killpg`, `signal.SIGKILL`, `os.getuid`, `loop.add_signal_handler` and
    `os.kill(pid, 0)` all import cleanly on every platform and then raise (or,
    for `os.kill(pid, 0)` on Windows, silently KILL the process being probed)
    at the call. That is the difference between a crash somebody reports and a
    defect that ships.

    Guard detection is deliberately conservative and syntactic: an attribute is
    reported as guarded when some enclosing node is a `try:`, an `if` whose
    test mentions the platform (`sys.platform`, `os.name`, `platform.system`,
    a `_PLATFORM`-style constant) or a capability probe (`hasattr`). Anything
    else counts as unguarded, which over-reports rather than under-reports --
    the right direction for a battery whose job is to find these.
    """
    import ast

    fatal: list[str] = []
    warn: list[str] = []
    gated: list[str] = []
    guarded_count = 0
    for path in sorted((REPO / "local_operator").rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            source = path.read_text(encoding="utf-8")
        except (SyntaxError, UnicodeDecodeError):
            continue
        rel = path.relative_to(REPO)
        # A module that refuses to be imported off POSIX is a legitimate shape
        # (the evaluation adapter supervisor does exactly this) and every call
        # inside it is unreachable on Windows by construction. Detected on the
        # source because the guard is a module-level `raise`, which is visible
        # to neither the enclosing-`if` rule nor the early-return rule.
        module_gate = _module_posix_gate(source)
        for line, pattern, is_fatal, is_guarded in _scan_posix_uses(tree):
            entry = f"{rel}:{line}  {pattern}"
            if is_guarded:
                guarded_count += 1
            elif module_gate:
                gated.append(f"{entry}  (module refuses off POSIX: {module_gate})")
            elif is_fatal:
                fatal.append(entry)
            else:
                warn.append(entry)

    detail = (
        f"{len(fatal)} fatal, {len(warn)} lead, {guarded_count} guarded, "
        f"{len(gated)} in POSIX-gated modules"
    )
    return Result(
        "static.posix_attributes",
        "FAIL" if fatal else ("WARN" if (warn or gated) else "PASS"),
        detail,
        {
            "fatal": fatal,
            # Leads, not findings. This scan is purely syntactic, so it cannot
            # see every shape of guard. Each entry here has to be confirmed by
            # reading it; a FAIL above is the trustworthy part.
            "warn_leads": warn,
            "module_gated": gated,
            "guarded": guarded_count,
        },
    )


# --------------------------------------------------------------------------- #
# The static scan's vocabulary and its platform-guard discovery
# --------------------------------------------------------------------------- #

#: Terms whose presence in a test says "this branch is about the platform".
#: `hasattr` belongs here rather than with the capability probes below, because
#: a `hasattr(os, "getuid")` test IS the platform question.
PLATFORM_TERMS = (
    "os.name",
    "sys.platform",
    "platform.system",
    "_PLATFORM",
    "IS_WINDOWS",
    "is_windows",
    "win32",
    "on_windows",
    "hasattr",
)

#: Strings that name a platform when something is COMPARED against them. Only a
#: compared literal counts -- see `_compares_against_platform_literal` -- because
#: every module here documents the platform it targets, and an unparsed docstring
#: is indistinguishable from code.
PLATFORM_STRING_LITERALS = frozenset({"win32", "posix", "nt", "darwin", "linux", "cygwin"})

#: Attributes that are unambiguously ABSENT off POSIX and whose absence crashes
#: or corrupts: `os.kill(pid, 0)` (which on Windows TERMINATES the process it is
#: probing), `loop.add_signal_handler`, `os.killpg`, `os.getuid`/`geteuid`,
#: `os.getpgid`, `os.setsid`, `os.fork`, `os.symlink`.
FATAL_TARGETS = {
    "killpg": "os.killpg",
    "getuid": "os.getuid",
    "geteuid": "os.geteuid",
    "getgid": "os.getgid",
    "setsid": "os.setsid",
    "getpgid": "os.getpgid",
    "fork": "os.fork",
    "symlink": "os.symlink",
}

#: Absent off POSIX too, but not destructive where they are: a lead.
LEAD_TARGETS = {
    "SIGKILL": "signal.SIGKILL",
    "SIGUSR1": "signal.SIGUSR1",
    "SIGUSR2": "signal.SIGUSR2",
    "SIGWINCH": "signal.SIGWINCH",
    "SIGSTOP": "signal.SIGSTOP",
    "chmod": "os.chmod",
    "getlogin": "os.getlogin",
    "nice": "os.nice",
}

#: Every attribute name above, as a vocabulary for `_platform_guard_names`: a
#: capability probe that asks about one of THESE is a platform question, while
#: `getattr(obj, "cr_frame", None)` is not, and treating the two alike would let
#: any attribute probe in the tree silence a real POSIX hit.
POSIX_ATTRIBUTE_NAMES = frozenset(FATAL_TARGETS) | frozenset(LEAD_TARGETS)


def _atom_text(node: ast.AST) -> str:
    """The dotted name a `Name`/`Attribute` spells, or "" for anything else.

    Deliberately not `ast.unparse`: unparsing (or worse, copying a subtree to
    blank out its docstring) costs seconds on the large files where reading an
    attribute chain costs nothing. It also cannot be fooled the way a text match
    can: a docstring is not an `Attribute`, so prose about "posix" is invisible
    here by construction rather than by a rule that has to be remembered.
    """
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _compare_has_platform_literal(node: ast.Compare) -> bool:
    """Does this comparison test something against a platform literal?

    The comparison is what makes the literal a question about the platform:
    `if host == "win32":` asks one, while the sentence "posix has no such call"
    in a docstring or a log message merely contains the word.
    """
    for operand in [node.left, *node.comparators]:
        if (
            isinstance(operand, ast.Constant)
            and isinstance(operand.value, str)
            and operand.value.strip().lower() in PLATFORM_STRING_LITERALS
        ):
            return True
    return False


def _mentions_posix_attribute(node: ast.AST) -> bool:
    """Does this subtree name one of the POSIX-only attributes, as a string?"""
    return any(
        isinstance(child, ast.Constant)
        and isinstance(child.value, str)
        and child.value in POSIX_ATTRIBUTE_NAMES
        for child in ast.walk(node)
    )


def _call_is_capability_probe(node: ast.Call) -> bool:
    """Does this call ASK whether a POSIX-only attribute exists here?

    Two spellings, both of them in this tree: `hasattr(os, "getuid")` and
    `getattr(signal, "SIGUSR1", None)` -- the `None` default IS the test.
    Recognising only the first is how `session/runtime/process.py`'s guarded
    `loop.add_signal_handler` still read as an unguarded POSIX use.

    The attribute name is required, so that `getattr(obj, "cr_frame", None)` --
    which this package uses constantly for other reasons -- cannot silence a
    real POSIX hit.
    """
    if not (isinstance(node.func, ast.Name) and _mentions_posix_attribute(node)):
        return False
    if node.func.id == "hasattr":
        return True
    return node.func.id == "getattr" and any(
        isinstance(arg, ast.Constant) and arg.value is None for arg in node.args
    )


class _Span(NamedTuple):
    """A node's (start, end) positions -- see `_span`."""

    start: tuple[int, int]
    end: tuple[int, int]


#: A position that cannot be contained by anything and contains nothing. Used for
#: the synthetic nodes that carry no position, so "unknown" reads as "no guard"
#: and "not a platform test" rather than as "everywhere".
_NO_SPAN = _Span((1 << 30, 1 << 30), (1 << 30, 1 << 30))


class _Query(NamedTuple):
    """One place in the file that answers part of the platform question.

    `kind` is "code" (a name that mentions the platform), "literal" (a comparison
    against a platform word), "capability" (a `hasattr`/`getattr` probe about a
    POSIX-only attribute) or "name" (any name, for the guard-chain lookup).
    """

    span: _Span
    kind: str
    name: str = ""


class _Candidate(NamedTuple):
    """Something that can BE a platform test: an assignment or a function.

    `span` is the assignment's VALUE, or the whole function definition, so that
    containment asks the question the discovery means: "does the thing this name
    is defined as ask the platform question?"
    """

    span: _Span
    names: tuple[str, ...]
    kind: str  # "const" for an assignment, "pred" for a function


class _Hit(NamedTuple):
    """A POSIX use found by the traversal, with its guard REASONS not yet read.

    The guards are kept as spans and resolved after the traversal, because which
    name is a platform constant is only known once every node has been seen --
    and resolving it by walking the tree again is the cost this design exists to
    avoid (see `note`).
    """

    lineno: int
    pattern: str
    fatal: bool
    guards: tuple[_Span, ...]
    in_try: bool


def _span(node: ast.AST) -> _Span:
    """A node's start and end position, for the containment tests below.

    Every node of a parsed tree carries these, which is what lets the analysis
    ask "is this question inside that candidate?" without walking the candidate's
    subtree again.
    """
    start_line = getattr(node, "lineno", None)
    start_col = getattr(node, "col_offset", None)
    end_line = getattr(node, "end_lineno", None)
    end_col = getattr(node, "end_col_offset", None)
    if start_line is None or start_col is None:
        return _NO_SPAN
    return _Span(
        (start_line, start_col),
        (
            end_line if end_line is not None else start_line,
            end_col if end_col is not None else start_col,
        ),
    )


def _index_queries(queries: list[_Query]) -> tuple[list[_Query], list[tuple[int, int]]]:
    """Queries ordered by start position, plus their starts, for `bisect`."""
    ordered = sorted(
        (query for query in queries if query.span.start != _NO_SPAN.start),
        key=lambda query: query.span.start,
    )
    return ordered, [q.span.start for q in ordered]


def _candidate_facts(
    ordered: list[_Query],
    starts: list[tuple[int, int]],
    span: _Span,
    interesting: frozenset[str] | set[str],
) -> tuple[bool, frozenset[str]]:
    """What the code inside `span` says: (platform question?, guard names used).

    The ONE place containment is turned into an answer, and it is deliberately
    O(queries inside the span), not O(queries in the file): `ordered` is sorted,
    so the scan starts at `bisect.bisect_left` and stops at the first query past
    the span. `interesting` filters name references down to the names that can
    matter (the module's own bindings, or its guard names), which is what keeps
    these sets small on a 40 000-line module.
    """
    static = False
    names: set[str] = set()
    index = bisect.bisect_left(starts, span.start)
    while index < len(ordered) and ordered[index].span.start <= span.end:
        query = ordered[index]
        if query.span.end <= span.end:
            if query.kind in _QUESTION_KINDS:
                static = True
            elif query.kind == "name" and query.name in interesting:
                names.add(query.name)
        index += 1
    return static, frozenset(names)


#: Query kinds that make a candidate a platform question all by themselves.
_QUESTION_KINDS = ("code", "literal", "capability")


class _GuardAnalysis(NamedTuple):
    """The module's platform guards, with the query index they were read from.

    Kept together so that every later question ("is this test a platform
    question?") is answered against the same reading of the file, and so a caller
    that already holds the index does not pay for it twice.
    """

    constants: frozenset[str]
    predicates: frozenset[str]
    ordered: list[_Query]
    starts: list[tuple[int, int]]

    def guarded(self, span: _Span) -> bool:
        """Does whatever sits at `span` ask the platform question?"""
        guards = self.constants | self.predicates
        static, names = _candidate_facts(self.ordered, self.starts, span, guards)
        return static or bool(names & guards)


def _assign_names(node: ast.Assign | ast.AnnAssign) -> list[str]:
    """The plain names an assignment binds (attribute/subscript targets skipped)."""
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    names: list[str] = []
    for target in targets:
        if isinstance(target, ast.Name):
            names.append(target.id)
        elif isinstance(target, ast.Tuple):
            names.extend(elt.id for elt in target.elts if isinstance(elt, ast.Name))
    return names


def _platform_guards(candidates: list[_Candidate], queries: list[_Query]) -> _GuardAnalysis:
    """Which candidates ARE platform tests, discovered from the module itself.

    The scan's other rules are syntactic, and this repository's convention is to
    ask the platform question ONCE, give it a name, and then branch on the name:

        _UID_IS_MEANINGFUL = os.name == "posix"
        ...
        if not _UID_IS_MEANINGFUL:
            return 0
        return os.getuid()

    Nothing in that branch mentions a platform, so a scan that only reads terms
    reports the FIX as the defect -- measured on the Windows runner as the audit's
    own `_UID_IS_MEANINGFUL` guard coming back as one of six "fatal" hits, and as
    `static.posix_attributes` FAILing on every OS, which made both new CI legs
    fail by construction (reviewer B, A1).

    DISCOVERED from the module's own source rather than listed in this file: a
    list would be a second place for the convention to drift, and it would need
    editing every time a module asked its question differently. The chain is
    followed to a fixpoint, so `PEER_AUTHENTICATION_SUPPORTED = _IS_DARWIN or
    _IS_LINUX` is understood through the two constants it is derived from, with
    none of the three named anywhere here.

    Everything is interval arithmetic over positions the caller collected in its
    ONE traversal. An earlier version re-walked each candidate's subtree per
    round, which is quadratic in nesting and measured at minutes on one large
    module -- a battery slow enough to die on its own job ceiling.
    """
    ordered, starts = _index_queries(queries)
    # The names that can MATTER are the module's own bindings: a reference to
    # anything else cannot be part of a guard chain, and carrying every name in
    # the file through the fixpoint is what made this quadratic.
    candidate_names = frozenset(name for candidate in candidates for name in candidate.names)
    # One containment pass per candidate, ONCE, not once per round: a module-level
    # statement's span covers the whole statement (nested definitions included),
    # and re-doing that per round is where the time went.
    facts = [
        (candidate, *_candidate_facts(ordered, starts, candidate.span, candidate_names))
        for candidate in candidates
    ]

    constants: set[str] = set()
    predicates: set[str] = set()
    changed = True
    while changed:
        changed = False
        known = constants | predicates
        for candidate, static, refs in facts:
            if not (static or (refs & known)):
                continue
            target = constants if candidate.kind == "const" else predicates
            for name in candidate.names:
                if name not in target:
                    target.add(name)
                    changed = True
    return _GuardAnalysis(frozenset(constants), frozenset(predicates), ordered, starts)


def _module_posix_gate(source: str) -> str | None:
    """The module-level platform refusal, if this module has one."""
    for line in source.splitlines()[:200]:
        stripped = line.split("#", 1)[0].strip()
        if stripped.startswith("if ") and "posix" in stripped and stripped.endswith(":"):
            return stripped
    return None


def _scan_posix_uses(tree: "ast.Module") -> list[tuple[int, str, bool, bool]]:
    """Walk `tree`, yielding (lineno, pattern, is_fatal, guarded) for each hit.

    `is_fatal` is reserved for the attributes that are unambiguously ABSENT off
    POSIX and whose absence crashes or corrupts -- `os.kill(pid, 0)` (which on
    Windows TERMINATES the process it is probing), `loop.add_signal_handler`,
    `os.killpg`, `os.getuid`/`geteuid`, `os.getpgid`, `os.setsid`, `os.fork`,
    `os.symlink`. Everything else is a lead: `signal.SIGKILL` is absent off
    POSIX too, but a module that refuses to run there at all is a legitimate
    shape this scan cannot distinguish from a defect, and a FAIL that is wrong
    once stops being read.

    Three precision rules, all of them shapes this codebase actually uses, and
    all of them narrowing the scan to the unreachability question it is really
    asking:

    * an `if` TEST is scanned with that same `if`'s guard active
      (``if not hasattr(os, "geteuid") or info.st_uid != os.geteuid()``);
    * a platform question that TERMINATES its block guards the rest of it
      (``store.py:196-198``, and the ``if not _UID_IS_MEANINGFUL: return 0``
      convention above);
    * a branch on a name the module itself computed FROM the platform is a
      platform branch -- see :func:`_platform_guards`.

    ``guarded`` still means "unreachable off POSIX as far as a syntax tree can
    tell". Nothing here can see a helper that is only CALLED from the launchd
    arm, so such a helper must state its own guard to be read as guarded.
    """
    hits: list[_Hit] = []
    # Everything the guard analysis needs is collected in the SAME traversal that
    # finds the hits: a second pass over this package (600 000 nodes) measured at
    # ~40 s, which is most of what a CI leg has to spend.
    candidates: list[_Candidate] = []
    queries: list[_Query] = []

    def note(node: ast.AST) -> None:
        """Record what this node contributes to the guard analysis."""
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            if node.value is not None:
                targets = _assign_names(node)
                if targets:
                    candidates.append(_Candidate(_span(node.value), tuple(targets), "const"))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            candidates.append(_Candidate(_span(node), (node.name,), "pred"))
        elif isinstance(node, ast.Name):
            if any(term in node.id for term in PLATFORM_TERMS):
                queries.append(_Query(_span(node), "code"))
            queries.append(_Query(_span(node), "name", node.id))
        elif isinstance(node, ast.Attribute):
            text = _atom_text(node)
            if any(term in text for term in PLATFORM_TERMS):
                queries.append(_Query(_span(node), "code"))
            queries.append(_Query(_span(node), "name", node.attr))
        elif isinstance(node, ast.Compare):
            if _compare_has_platform_literal(node):
                queries.append(_Query(_span(node), "literal"))
        elif isinstance(node, ast.Call):
            if _call_is_capability_probe(node):
                queries.append(_Query(_span(node), "capability"))

    def _terminates(node: ast.AST) -> bool:
        """Does any branch of this `if` leave the enclosing block entirely?"""
        terminators = (ast.Return, ast.Raise, ast.Continue, ast.Break)
        return any(isinstance(child, terminators) for child in ast.walk(node))

    def visit(node: ast.AST, guards: tuple[_Span, ...], in_try: bool = False) -> None:
        note(node)
        if isinstance(node, ast.Try):
            for child in node.body + node.handlers + node.orelse + node.finalbody:
                visit(child, guards, True)
            return
        if isinstance(node, ast.If):
            nested = guards + (_span(node.test),)
            # `in_try` must be threaded into the TEST too: an `if` inside a
            # `try:` whose condition itself calls a POSIX-only function is
            # guarded by that `try`, and dropping the flag here is what made
            # the first run report `control.py:663` as unguarded.
            visit(node.test, nested, in_try)
            for child in node.body + node.orelse:
                visit(child, nested, in_try)
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # `kw_defaults` legitimately contains None for a keyword-only
            # argument with no default, so the Nones are filtered rather than
            # handed to the AST walker.
            extras = [
                child
                for child in (
                    list(node.decorator_list)
                    + list(node.args.defaults)
                    + list(node.args.kw_defaults)
                )
                if isinstance(child, ast.AST)
            ]
            for child in extras:
                visit(child, guards, in_try)
            # The BODY goes through visit_block, so a capability check that
            # raises early in the function guards the rest of it.
            visit_block(node.body, guards, in_try)
            return
        if isinstance(node, (ast.ClassDef, ast.With)):
            for child in ast.iter_child_nodes(node):
                visit(child, guards, in_try)
            return
        if isinstance(node, ast.Module):
            visit_block(node.body, guards, in_try)
            return
        if isinstance(node, ast.Attribute):
            owner = ast.unparse(node.value) if isinstance(node.value, ast.Name) else ""
            name = node.attr
            is_fatal = name in FATAL_TARGETS
            pattern = FATAL_TARGETS.get(name) or LEAD_TARGETS.get(name)
            if pattern and (owner in ("os", "signal") or name == "add_signal_handler"):
                hits.append(_Hit(node.lineno, pattern, is_fatal, guards, in_try))
            elif name == "add_signal_handler":
                hits.append(_Hit(node.lineno, "loop.add_signal_handler", True, guards, in_try))
        if isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "kill"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value == 0
            ):
                # A surrounding `try` is NOT a guard for this one. The whole
                # defect on Windows is that the call SUCCEEDS at killing the
                # process and raises nothing for the handler to catch, so a
                # `try: os.kill(pid, 0) except OSError:` reads as protected to
                # every other rule and is the exact shape that ships.
                hits.append(_Hit(node.lineno, "os.kill(pid, 0)", True, guards, False))
        for child in ast.iter_child_nodes(node):
            visit(child, guards, in_try)

    def visit_block(body: list[ast.stmt], guards: tuple[_Span, ...], in_try: bool = False) -> None:
        """Visit statements in order, honouring a preceding platform check."""
        blocked = guards
        for stmt in body:
            visit(stmt, blocked, in_try)
            # A platform question that TERMINATES the block guards everything
            # AFTER it in that block, which is how this repository states both
            # its Windows refusals and its platform-constant early returns:
            #     if not _UID_IS_MEANINGFUL:
            #         return 0
            #     return os.getuid()
            # This is ONE rule, not two. The `hasattr` case it used to be
            # spelled for (`if not hasattr(os, "getuid"): raise ...`, the
            # evidence store's Windows refusal) fires here as well, because
            # `hasattr` is a platform term and a `raise` terminates.
            if isinstance(stmt, ast.If) and _terminates(stmt):
                blocked = blocked + (_span(stmt.test),)

    for node in ast.iter_child_nodes(tree):
        visit(node, ())

    analysis = _platform_guards(candidates, queries)
    return sorted(
        (
            hit.lineno,
            hit.pattern,
            hit.fatal,
            hit.in_try or any(analysis.guarded(span) for span in hit.guards),
        )
        for hit in hits
    )


def probe_os_attributes(env: dict[str, str]) -> Result:
    """What this interpreter actually HAS. The map the scanners are read against."""
    driver = textwrap.dedent("""
        import json, os, signal, socket, sys
        out = {"python": sys.version.split()[0], "platform": sys.platform}
        for name in ("killpg", "getuid", "geteuid", "getpgid", "setsid", "fork",
                     "symlink", "getlogin", "startfile", "setsid", "device_encoding"):
            out[f"os.{name}"] = hasattr(os, name)
        for name in ("SIGKILL", "SIGUSR1", "SIGUSR2", "SIGWINCH", "SIGSTOP", "SIGTERM"):
            out[f"signal.{name}"] = hasattr(signal, name)
        out["socket.AF_UNIX"] = hasattr(socket, "AF_UNIX")
        out["socket.SO_EXCLUSIVEADDRUSE"] = hasattr(socket, "SO_EXCLUSIVEADDRUSE")
        out["os.name"] = os.name
        print(json.dumps(out))
        """)
    proc = run([sys.executable, "-c", driver], env, timeout=60.0)
    if proc.returncode != 0:
        return Result("os.attributes", "FAIL", _tail(proc.stderr))
    attrs = json.loads(proc.stdout.strip().splitlines()[-1])
    missing = [k for k, v in attrs.items() if v is False]
    detail = f"{len(attrs) - len(missing)} present, absent: {', '.join(missing) or 'none'}"
    return Result("os.attributes", "PASS", detail, attrs)


def probe_serve_double_bind(env: dict[str, str]) -> Result:
    """A SECOND `lop serve` on the same port must REFUSE, on every OS.

    This is the probe that catches `SO_REUSEADDR` semantics diverging by
    platform: on Windows a second bind over a listening holder SUCCEEDS, so the
    documented "daemon already running" refusal never fires and two servers
    race on one port. On macOS and Linux the same code refuses correctly, which
    is exactly why a macOS-only test suite cannot see the defect.
    """
    import urllib.error
    import urllib.request

    port = _free_port()
    # Same 180s window and the same reason as `probe_serve_health`.
    window = BUDGET.window(180.0)
    with _spawn_cli(
        ["serve", "--port", str(port)], env, log=_child_log(env, "double-bind-first")
    ) as first:
        deadline = time.monotonic() + window
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
                    break
            except Exception:  # noqa: BLE001 - keep polling
                if first.proc.poll() is not None:
                    return Result(
                        "serve.double_bind",
                        "FAIL",
                        f"the first server never came up (rc={first.proc.returncode}): "
                        f"{first.output()}",
                        first.extra(),
                    )
                time.sleep(0.5)
        else:
            return Result(
                "serve.double_bind",
                "FAIL",
                _no_response(first, window, "the first server never answered"),
                first.extra(),
            )

        second = run(_cli_argv("serve", "--port", str(port)), env, timeout=60.0)
        text = ((second.stdout or "") + (second.stderr or "")).strip()
        if second.returncode == 0:
            return Result(
                "serve.double_bind",
                "FAIL",
                "a SECOND server accepted the same port while the first held it",
                first.extra(second_rc=second.returncode, raw=text[:600]),
            )
        return Result(
            "serve.double_bind",
            "PASS",
            f"second bind refused (rc={second.returncode}): {_first_line(text)}",
            first.extra(second_rc=second.returncode),
        )


def probe_daemon_supervisors(env: dict[str, str]) -> Result:
    """One row per daemon: does its INSTALLER SURFACE render on this OS?

    Reported as four separate facts rather than one PASS, because the useful
    reading is which of them is missing -- on Linux the answer was "two of
    four", which a single aggregate verdict would have hidden.

    WHAT THIS MEASURES, exactly, because its old docstring asked the larger
    question and answered the smaller one: it runs `install --help` and reports
    whether the parser renders. That is true on every platform -- the subcommand
    exists everywhere the installer is registered -- and says nothing about
    whether THIS HOST has a supervisor binary to hand the unit to (reviewer B,
    A4). So the reading is named for what it is (`renders`), the result name is
    the surface list rather than a supervisor claim, and the claim itself is
    carried in `extra["measured"]` so a reader of the artifact is not left to
    infer it from a PASS.
    """
    daemons = ("mobile", "wake", "tunnel", "browser")
    per_daemon: dict[str, str] = {}
    for name in daemons:
        argv = {
            "mobile": ["mobile", "install", "--help"],
            "wake": ["wake", "install", "--help"],
            "tunnel": ["tunnel", "install", "--help"],
            "browser": ["browser", "install", "--help"],
        }[name]
        proc = run(_cli_argv(*argv), env, timeout=90.0)
        per_daemon[name] = "renders" if proc.returncode == 0 else _first_line(proc.stderr)
    return Result(
        "daemon.surfaces",
        "PASS" if all(v == "renders" for v in per_daemon.values()) else "FAIL",
        ", ".join(f"{k}={v}" for k, v in per_daemon.items()),
        {
            **per_daemon,
            "measured": "each installer's `install --help` renders; this is not a "
            "check that a supervisor exists on this host",
        },
    )


def probe_host_facts(env: dict[str, str]) -> Result:
    """The environment the rest of the matrix has to be read against."""
    facts = {
        "platform": sys.platform,
        "os_name": os.name,
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "shell": os.environ.get("SHELL") or os.environ.get("ComSpec"),
        "term": os.environ.get("TERM"),
        "isatty": sys.stdout.isatty(),
        "resolution": REPO.name,
    }
    return Result(
        "host.facts", "PASS", f"{facts['system']} {facts['release']} {facts['machine']}", facts
    )


# --------------------------------------------------------------------------- #
# Probe registry
# --------------------------------------------------------------------------- #

PROBES = (
    probe_host_facts,
    probe_os_attributes,
    probe_static_posix_imports,
    probe_static_posix_attributes,
    probe_import_package,
    probe_cli_version,
    probe_cli_help,
    probe_cli_subcommands,
    probe_paths_roots,
    probe_config_roundtrip,
    probe_secret_roundtrip,
    probe_sessions_list,
    probe_wake_status,
    probe_wake_install,
    probe_mobile_status,
    probe_mobile_install,
    probe_mobile_daemon_serve,
    probe_daemon_supervisors,
    probe_tunnel_status,
    probe_tunnel_install,
    probe_serve_health,
    probe_serve_double_bind,
    probe_tui_boot,
    probe_tui_driver_tty,
    probe_exec_offline,
    probe_file_lock,
)


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #


@dataclass
class ChildRun:
    """A long-lived child whose output goes to a FILE, not to a pipe.

    Three failures in run 35405383805 could not be explained from the artifact
    (`serve.health`, `serve.double_bind`, `mobile.daemon_serve` -- all "no
    response in Ns"), because a long-lived child's stdout went to a `PIPE` that
    this harness never drained. That is two defects in one: the output is LOST
    when the probe gives up, and a child that fills the 64 KiB pipe buffer BLOCKS
    on its next log line -- a plausible cause of the very timeouts the artifact
    could not explain, since `lop serve` and `lop mobile serve` both log on the
    way to binding. A file answers both: nothing blocks on a full pipe, and the
    tail travels in the artifact as evidence the next reader can act on.
    """

    proc: subprocess.Popen[Any]
    log: Path

    def output(self, limit: int = 400) -> str:
        """The tail of everything the child has written so far (never raises)."""
        try:
            return _tail(self.log.read_text(encoding="utf-8", errors="replace"), limit)
        except OSError:  # the file is created by `_spawn_child`; absent only on a bug
            return ""

    def extra(self, **more: Any) -> dict[str, Any]:
        """`Result.extra` fields that make a long-lived child diagnosable."""
        return {"child_log": str(self.log), "child_output": self.output(), **more}

    def stop(self) -> None:
        """Kill the child and everything it spawned, then reap it."""
        _terminate(self.proc)

    def __enter__(self) -> "ChildRun":
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


def _no_response(child: "ChildRun", window: float, last: str) -> str:
    """A poll that ran out of window, in the terms that separate the causes.

    "no response in 60s" cannot tell a reader whether the daemon crashed on
    startup, is still importing, or was starved by the host -- and those need
    different responses from whoever reads the artifact. The process's own state
    at the deadline answers the first two, and `extra`'s `child_output` answers
    the third (QA round 1, Q1: the same code and the same command read FAIL=2
    under load and FAIL=0 on the next run, and the artifact could not say which
    it had been).
    """
    alive = child.proc.poll() is None
    state = (
        "the process was STILL RUNNING at the deadline"
        if alive
        else f"the process had exited rc={child.proc.returncode}"
    )
    return f"no response in {window:.0f}s ({state}; last error {last or 'none'}){BUDGET.note()}"


def _spawn_child(argv: list[str], env: dict[str, str], *, log: Path) -> ChildRun:
    """Start a long-lived child in its OWN process group / session.

    `start_new_session` is load-bearing, not tidiness. A daemon child that
    shares this process's group makes `os.killpg(os.getpgid(child))` in
    :func:`_terminate` resolve to the PROBE'S OWN group, so the cleanup
    SIGTERMs the harness -- observed as the battery dying mid-run with rc=143
    the moment it reached `serve.health`, having already printed everything
    before it. Windows has no process groups in that sense; it gets a new
    console process group instead so its own children can be signalled.

    Output goes to `log` rather than a pipe -- see `ChildRun` for why -- and the
    parent's own descriptor is closed as soon as the child has its duplicate, so
    a battery that starts a child per probe does not leak one descriptor each.
    """
    kwargs: dict[str, Any] = {}
    if os.name == "posix":
        kwargs["start_new_session"] = True
    else:  # pragma: no cover - exercised on the Windows runner
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = open(log, "wb")  # noqa: SIM115 - inherited by the child, closed below
    try:
        proc = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            argv,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(REPO),
            **kwargs,
        )
    finally:
        handle.close()
    return ChildRun(proc, log)


def _spawn_cli(args: list[str], env: dict[str, str], *, log: Path) -> ChildRun:
    """`_spawn_child` for a `lop` subcommand (see `_cli_argv`)."""
    return _spawn_child(_cli_argv(*args), env, log=log)


def _child_log(env: dict[str, str], name: str) -> Path:
    """Where a long-lived child's output is written, inside the sandbox.

    Under `$HOME`, not `TMPDIR`: the sandbox root is the one directory this
    battery owns end to end, and a run that keeps its sandbox (`--keep`) then
    keeps the child logs beside everything else the probes wrote.
    """
    return Path(env["HOME"]) / "child-logs" / f"{name}.log"


def _platform_unavailable(text: str) -> str | None:
    """The phrase in `text` that says "this OS has no arm for that", if any.

    One list, used by every installer probe, because the alternative is five
    hand-rolled substring tests that drift apart: the first version of this
    file recognised "not supported" and missed "no supervisor installer for
    this platform", so a Linux host reported `wake.install` as PASS while the
    supervisor had not been installed at all. That is the exact silent-failure
    shape this battery exists to catch, reproduced in the battery itself.
    """
    lowered = text.lower()
    markers = (
        "no supervisor installer for this platform",
        "needs macos",
        "only supported on",
        "not supported on this platform",
        "unsupported on this platform",
        "is not supported",
        "foreground elsewhere",
        "foreground on this platform",
        "requires launchd",
        "requires systemd",
        "no service manager",
    )
    for marker in markers:
        if marker in lowered:
            return marker
    return None


def _free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _terminate(proc: subprocess.Popen[str]) -> None:
    """Kill a child and everything it spawned, then reap it.

    `terminate()` alone is not enough: lop's server and daemon spawn children of
    their own, and a probe battery that leaks a daemon onto the host is a probe
    battery nobody will run twice.

    The group kill is guarded against this process's OWN group. A child started
    without `start_new_session` shares it, and `killpg` on it terminates the
    harness -- which is exactly how the first version of this file died at
    `serve.health` with rc=143.
    """
    if proc.poll() is not None:
        return
    own_group: int | None = None
    child_group: int | None = None
    if os.name == "posix":
        try:
            own_group = os.getpgid(0)
            child_group = os.getpgid(proc.pid)
        except OSError:
            pass
    try:
        if os.name == "posix" and child_group is not None and child_group != own_group:
            import signal

            os.killpg(child_group, signal.SIGTERM)
        else:
            proc.terminate()
    except Exception:  # noqa: BLE001 - best effort
        proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "posix" and child_group is not None and child_group != own_group:
                import signal

                os.killpg(child_group, signal.SIGKILL)
            else:
                proc.kill()
        except Exception:  # noqa: BLE001
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass


def _tui_driver(out: str) -> int:
    """The child half of `tui.boot`: boot the real app, write a frame, exit."""
    sys.path.insert(0, str(REPO))
    # Import-time isolation, deliberately: it re-homes HOME and refuses if any
    # `local_operator` module was already imported. A function call can be
    # forgotten (and was, at cost -- see scripts/probe_isolation.py).
    import asyncio  # noqa: PLC0415

    import scripts.probe_isolation  # noqa: F401, PLC0415
    from local_operator.tui.app import OperatorApp  # noqa: PLC0415
    from scripts.visual_capture import save_capture  # noqa: PLC0415

    try:
        # The fake session lives in the test suite on purpose -- it is the
        # double the TUI's own tests are written against, so reusing it keeps
        # this probe on the same app-construction path instead of inventing a
        # second one that can pass while the tested path is broken.
        from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: PLC0415
    except ImportError as exc:
        print(f"TUI_DRIVER_UNAVAILABLE: {exc}")
        return 3

    async def main() -> None:
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(120, 34)) as pilot:
            await pilot.pause()
            save_capture(app, out)
            await pilot.pause()

    asyncio.run(main())
    return 0


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def _host_facts() -> dict[str, Any]:
    """The environment a run's numbers have to be read against."""
    return {
        "platform": sys.platform,
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
    }


def _write_matrix(path: Path, results: list[Result]) -> None:
    """Write the artifact, rebuilt from the results accumulated so far.

    Rebuilt rather than appended to because the caller writes it after EVERY
    probe: one function that describes the state, called as often as the state
    changes, cannot drift from it. That is also what makes a PARTIAL run
    evidence -- see the call site for why the upload step needs it.
    """
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    payload = {
        "host": _host_facts(),
        "counts": counts,
        "results": [
            {"name": r.name, "status": r.status, "detail": r.detail, "extra": r.extra}
            for r in results
        ],
    }
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--json", metavar="PATH", help="write the full result matrix as JSON")
    parser.add_argument(
        "--only", nargs="*", default=None, help="run only probes whose name contains one of these"
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="stop STARTING probes once this much wall clock has been spent (0 = no "
        "aggregate bound). CI sets it BELOW each job's timeout-minutes: a battery that "
        "returns with a partial matrix is evidence, while a job killed at its ceiling "
        "prints no matrix and uploads nothing",
    )
    parser.add_argument("--list", action="store_true", help="print probe names and exit")
    parser.add_argument(
        "--keep", action="store_true", help="keep the sandbox directory for inspection"
    )
    parser.add_argument(
        "--allow-keychain",
        action="store_true",
        help="allow probes that write to the macOS login keychain (off by default: "
        "an isolated HOME has none, so the call raises a keychain dialog)",
    )
    parser.add_argument(
        "--out-dir",
        metavar="PATH",
        help="write TUI frames here instead of inside the sandbox (use this to collect "
        "visual evidence out of a container)",
    )
    parser.add_argument("--driver", choices=["tui"], help=argparse.SUPPRESS)
    parser.add_argument("rest", nargs="*", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.driver == "tui":
        return _tui_driver(args.rest[0])

    # Before the first probe, and only in the parent: the `--driver tui` child
    # above is a probe's own subprocess and inherits no budget of its own.
    BUDGET.start(args.budget)

    if args.list:
        for probe in PROBES:
            # Both spellings, because the two are easy to confuse: the printed
            # name is the dotted one a reader copies, while matching happens on
            # the function name. A tool whose `--list` output does not work as
            # `--only` input wastes the reader's next command.
            dotted = probe.__name__.replace("probe_", "").replace("_", ".")
            underscored = probe.__name__.replace("probe_", "")
            print(f"{dotted:<28} ({underscored})")
        return 0

    root = _sandbox_root()
    env = isolated_env(root)
    if args.allow_keychain:
        env["LOP_XPLAT_ALLOW_KEYCHAIN"] = "1"
    if args.out_dir:
        shot_dir = Path(args.out_dir).expanduser()
        shot_dir.mkdir(parents=True, exist_ok=True)
        env["LOP_XPLAT_SHOT_DIR"] = str(shot_dir)
    # `--only` accepts either spelling, and normalises both to the underscore
    # form the probe functions actually use, so a token copied out of `--list`
    # selects what it says it selects.
    tokens = None
    if args.only is not None:
        tokens = [token.replace(".", "_") for token in args.only]
    selected = [
        probe
        for probe in PROBES
        if tokens is None
        or any(token in probe.__name__ for token in tokens)
        or any(token in probe.__name__.replace("probe_", "") for token in (args.only or []))
    ]
    if tokens is not None and not selected:
        print(
            f"no probe matched {args.only!r}; run --list for the names",
            file=sys.stderr,
        )
        return 2

    results: list[Result] = []
    for probe in selected:
        name = probe.__name__.replace("probe_", "").replace("_", ".")
        started = time.monotonic()
        try:
            # Checked BEFORE the probe runs: a probe started on borrowed time
            # reports an instant timeout, which reads as a defect in the surface
            # it was about to measure.
            if BUDGET.spent():
                raise BudgetSpent(f"the battery's {int(args.budget)}s budget was spent")
            result = probe(env)
        except BudgetSpent as exc:
            # SKIP, not FAIL: the surface was never asked anything.
            result = Result(name, "SKIP", f"not run: {exc}")
        except subprocess.TimeoutExpired as exc:
            result = Result(name, "FAIL", f"timed out: {exc}{BUDGET.note()}")
        except Exception as exc:  # noqa: BLE001 - a probe must never kill the battery
            result = Result(name, "FAIL", f"probe raised {type(exc).__name__}: {exc}")
        result.extra.setdefault("seconds", round(time.monotonic() - started, 1))
        results.append(result)
        print(f"  {result.status:4}  {result.name:22}  {result.detail}", flush=True)
        if args.json:
            # Rewritten after EVERY probe, not once after the loop. A battery
            # SIGKILLed at the job's ceiling, or killed by a crash out of the
            # loop, used to leave no artifact at all -- the opposite of what the
            # `if: always()` upload step names as its reason for existing, on
            # exactly the run that needed the artifact (reviewer B, A2).
            _write_matrix(Path(args.json), results)

    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    summary = " ".join(
        f"{key}={counts[key]}" for key in ("PASS", "WARN", "SKIP", "FAIL") if key in counts
    )
    print()
    print(f"{platform.system()} {platform.machine()} py{sys.version.split()[0]}: {summary}")

    if args.json:
        _write_matrix(Path(args.json), results)
        print(f"wrote {args.json}")

    if args.keep:
        print(f"sandbox kept at {root}")
    else:
        shutil.rmtree(root, ignore_errors=True)

    return 0 if counts.get("FAIL", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
