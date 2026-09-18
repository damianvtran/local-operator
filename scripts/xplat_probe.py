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
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    """Run a child with a hard timeout, never inheriting this process's tty."""
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(cwd or REPO),
        input=stdin,
        timeout=timeout,
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


def probe_import_package(env: dict[str, str]) -> Result:
    """Import EVERY submodule of the package, and report what would not import.

    This is the single most informative probe in an OS-portability battery: it
    turns "does lop run on this OS" into a list of module names, and it catches
    the module that is only imported on a rarely-taken branch.
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
        return Result("import.package", "FAIL", "timed out importing the package")
    if proc.returncode != 0:
        return Result(
            "import.package",
            "FAIL",
            _tail(proc.stderr) or "driver exited non-zero",
            {"returncode": proc.returncode},
        )
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    failed = payload["failed"]
    detail = f"{payload['total'] - len(failed)}/{payload['total']} modules import"
    return Result(
        "import.package",
        "FAIL" if failed else "PASS",
        detail,
        {"failed": failed},
    )


def probe_cli_version(env: dict[str, str]) -> Result:
    proc = run(_cli_argv("--version"), env, timeout=120.0)
    if proc.returncode != 0:
        return Result("cli.version", "FAIL", _tail(proc.stderr), {"rc": proc.returncode})
    return Result("cli.version", "PASS", _first_line(proc.stdout + proc.stderr))


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
    """
    proc = run(_cli_argv("wake", "status"), env, timeout=120.0)
    text = (proc.stdout + proc.stderr).strip()
    if proc.returncode != 0:
        return Result("wake.status", "FAIL", _tail(proc.stderr))
    text = (proc.stdout + proc.stderr).strip()
    lowered = text.lower()
    supported = not (
        _platform_unavailable(text)
        or any(marker in lowered for marker in ("not installed", "no supervisor"))
    )
    return Result(
        "wake.status",
        "PASS" if supported else "WARN",
        text.splitlines()[0][:240] if text else "no output",
        {"raw": text[:1200], "supervisor_available": supported},
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
        return Result(
            "mobile.install",
            "FAIL",
            _first_line(text) or f"exit {proc.returncode}",
            {"raw": text[:1200]},
        )
    return Result("mobile.install", "PASS", _first_line(text) or "installed", {"raw": text[:1200]})


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
    proc = _spawn_cli(["mobile", "serve", "--port", str(port)], env)
    try:
        deadline = time.monotonic() + 45.0
        last = ""
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                out = proc.stdout.read() if proc.stdout else ""
                return Result(
                    "mobile.daemon_serve",
                    "FAIL",
                    f"daemon exited rc={proc.returncode}: {_tail(out or last)}",
                )
            for path in ("/healthz", "/health"):
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}{path}", timeout=2
                    ) as response:
                        body = response.read(200).decode(errors="replace")
                        return Result(
                            "mobile.daemon_serve",
                            "PASS",
                            f"{path} -> {response.status} {body[:80]}",
                        )
                except urllib.error.HTTPError as exc:
                    # A 401 from the gate is a served daemon, not a failure.
                    return Result(
                        "mobile.daemon_serve",
                        "PASS",
                        f"{path} -> {exc.code} (auth gate answering)",
                    )
                except Exception as exc:  # noqa: BLE001 - keep polling
                    last = f"{type(exc).__name__}: {exc}"
            time.sleep(0.5)
        return Result("mobile.daemon_serve", "FAIL", f"no response in 45s ({last})")
    finally:
        _terminate(proc)


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
    proc = _spawn_cli(["serve", "--port", str(port)], env)
    try:
        deadline = time.monotonic() + 60.0
        last = ""
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                out = proc.stdout.read() if proc.stdout else ""
                return Result(
                    "serve.health",
                    "FAIL",
                    f"server exited rc={proc.returncode}: {_tail(out or last)}",
                )
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=2
                ) as response:
                    body = response.read(120).decode(errors="replace")
                    return Result(
                        "serve.health",
                        "PASS",
                        f"/health -> {response.status} {body}",
                    )
            except urllib.error.HTTPError as exc:
                return Result("serve.health", "FAIL", f"/health -> HTTP {exc.code}")
            except Exception as exc:  # noqa: BLE001 - keep polling
                last = f"{type(exc).__name__}: {exc}"
            time.sleep(0.5)
        return Result("serve.health", "FAIL", f"no response in 60s ({last})")
    finally:
        _terminate(proc)


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
    holder = subprocess.Popen(
        [sys.executable, "-c", driver, str(session_dir), "hold"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        cwd=str(REPO),
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
            return Result("lock.exclusive", "FAIL", _tail(combined))
        granted = json.loads(combined.strip().splitlines()[-1]).get("acquired")
        if granted:
            return Result(
                "lock.exclusive",
                "FAIL",
                "a second holder was granted the same lease while the first held it",
            )
        return Result("lock.exclusive", "PASS", "second holder correctly refused")
    finally:
        _terminate(holder)


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

    Two precision rules are implemented, both from false positives seen on the
    first run against this tree: an `if` TEST is scanned with that same `if`'s
    guard active (``if not hasattr(os, "geteuid") or info.st_uid != os.geteuid()``
    is guarded), and a capability check that RAISES on the preceding line
    guards the rest of its block (``store.py:196-198``).
    """
    import ast

    hits: list[tuple[int, str, bool, bool]] = []

    platform_terms = (
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

    fatal_targets = {
        "killpg": "os.killpg",
        "getuid": "os.getuid",
        "geteuid": "os.geteuid",
        "getgid": "os.getgid",
        "setsid": "os.setsid",
        "getpgid": "os.getpgid",
        "fork": "os.fork",
        "symlink": "os.symlink",
    }
    lead_targets = {
        "SIGKILL": "signal.SIGKILL",
        "SIGUSR1": "signal.SIGUSR1",
        "SIGUSR2": "signal.SIGUSR2",
        "SIGWINCH": "signal.SIGWINCH",
        "SIGSTOP": "signal.SIGSTOP",
        "chmod": "os.chmod",
        "getlogin": "os.getlogin",
        "nice": "os.nice",
    }

    def guarded_by_test(node: ast.AST) -> bool:
        try:
            text = ast.unparse(node)
        except Exception:  # noqa: BLE001 - an unparseable test is not a guard
            return False
        return any(term in text for term in platform_terms)

    def raises(node: ast.AST) -> bool:
        return any(isinstance(child, ast.Raise) for child in ast.walk(node))

    def _terminates(node: ast.AST) -> bool:
        """Does any branch of this `if` leave the enclosing block entirely?"""
        terminators = (ast.Return, ast.Raise, ast.Continue, ast.Break)
        return any(isinstance(child, terminators) for child in ast.walk(node))

    def visit(node: ast.AST, platform_guarded: bool, in_try: bool = False) -> None:
        guarded = platform_guarded or in_try
        if isinstance(node, ast.Try):
            for child in node.body + node.handlers + node.orelse + node.finalbody:
                visit(child, platform_guarded, True)
            return
        if isinstance(node, ast.If):
            nested = platform_guarded or guarded_by_test(node.test)
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
                visit(child, platform_guarded, in_try)
            # The BODY goes through visit_block, so a capability check that
            # raises early in the function guards the rest of it.
            visit_block(node.body, platform_guarded, in_try)
            return
        if isinstance(node, (ast.ClassDef, ast.With)):
            for child in ast.iter_child_nodes(node):
                visit(child, platform_guarded, in_try)
            return
        if isinstance(node, ast.Module):
            visit_block(node.body, platform_guarded, in_try)
            return
        if isinstance(node, ast.Attribute):
            owner = ast.unparse(node.value) if isinstance(node.value, ast.Name) else ""
            name = node.attr
            is_fatal = name in fatal_targets
            pattern = fatal_targets.get(name) or lead_targets.get(name)
            if pattern and (owner in ("os", "signal") or name == "add_signal_handler"):
                hits.append((node.lineno, pattern, is_fatal, guarded))
            elif name == "add_signal_handler":
                hits.append((node.lineno, "loop.add_signal_handler", True, guarded))
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
                hits.append((node.lineno, "os.kill(pid, 0)", True, platform_guarded))
        for child in ast.iter_child_nodes(node):
            visit(child, platform_guarded, in_try)

    def visit_block(body: list[ast.stmt], platform_guarded: bool, in_try: bool = False) -> None:
        """Visit statements in order, honouring a preceding capability check."""
        blocked = platform_guarded
        for stmt in body:
            visit(stmt, blocked, in_try)
            # `if not hasattr(x, "y"): raise ...` guards everything AFTER it in
            # this block, which is how the evidence store states its Windows
            # refusal. Without this rule the scan reports it as unguarded.
            if isinstance(stmt, ast.If) and raises(stmt):
                try:
                    test_text = ast.unparse(stmt.test)
                except Exception:  # noqa: BLE001
                    test_text = ""
                if "hasattr" in test_text:
                    blocked = True
            # A platform test that TERMINATES the block guards the rest of it,
            # which is how the repo's existing Windows arms are written:
            #     if _PLATFORM == "win32":\n    return True
            #     os.kill(pid, 0)
            # Seen on the first run as four false positives (resume.py,
            # session/retention.py, session_lease.py, procname.py).
            if isinstance(stmt, ast.If) and _terminates(stmt):
                try:
                    test_text = ast.unparse(stmt.test)
                except Exception:  # noqa: BLE001
                    test_text = ""
                if any(term in test_text for term in platform_terms):
                    blocked = True

    for node in ast.iter_child_nodes(tree):
        visit(node, False)
    return sorted(hits)


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
    first = _spawn_cli(["serve", "--port", str(port)], env)
    try:
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
                    break
            except Exception:  # noqa: BLE001 - keep polling
                if first.poll() is not None:
                    return Result("serve.double_bind", "FAIL", "the first server never came up")
                time.sleep(0.5)
        else:
            return Result("serve.double_bind", "FAIL", "the first server never came up")

        second = run(_cli_argv("serve", "--port", str(port)), env, timeout=60.0)
        text = ((second.stdout or "") + (second.stderr or "")).strip()
        if second.returncode == 0:
            return Result(
                "serve.double_bind",
                "FAIL",
                "a SECOND server accepted the same port while the first held it",
                {"second_rc": second.returncode, "raw": text[:600]},
            )
        return Result(
            "serve.double_bind",
            "PASS",
            f"second bind refused (rc={second.returncode}): {_first_line(text)}",
            {"second_rc": second.returncode},
        )
    finally:
        _terminate(first)


def probe_daemon_supervisors(env: dict[str, str]) -> Result:
    """One row per daemon: does this OS have a way to keep it running?

    Reported as four separate facts rather than one PASS, because the useful
    reading is which of them is missing -- on Linux the answer was "two of
    four", which a single aggregate verdict would have hidden.
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
        per_daemon,
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


def _spawn_cli(
    args: list[str], env: dict[str, str], *, stdout: int | None = subprocess.PIPE
) -> subprocess.Popen[str]:
    """Start a long-lived child in its OWN process group / session.

    `start_new_session` is load-bearing, not tidiness. A daemon child that
    shares this process's group makes `os.killpg(os.getpgid(child))` in
    :func:`_terminate` resolve to the PROBE'S OWN group, so the cleanup
    SIGTERMs the harness -- observed as the battery dying mid-run with rc=143
    the moment it reached `serve.health`, having already printed everything
    before it. Windows has no process groups in that sense; it gets a new
    console process group instead so its own children can be signalled.
    """
    kwargs: dict[str, Any] = {}
    if os.name == "posix":
        kwargs["start_new_session"] = True
    else:  # pragma: no cover - exercised on the Windows runner
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    return subprocess.Popen(  # noqa: S603 - fixed argv, no shell
        _cli_argv(*args),
        stdout=stdout,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        cwd=str(REPO),
        **kwargs,
    )


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--json", metavar="PATH", help="write the full result matrix as JSON")
    parser.add_argument(
        "--only", nargs="*", default=None, help="run only probes whose name contains one of these"
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
            result = probe(env)
        except subprocess.TimeoutExpired as exc:
            result = Result(name, "FAIL", f"timed out: {exc}")
        except Exception as exc:  # noqa: BLE001 - a probe must never kill the battery
            result = Result(name, "FAIL", f"probe raised {type(exc).__name__}: {exc}")
        result.extra.setdefault("seconds", round(time.monotonic() - started, 1))
        results.append(result)
        print(f"  {result.status:4}  {result.name:22}  {result.detail}", flush=True)

    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    summary = " ".join(
        f"{key}={counts[key]}" for key in ("PASS", "WARN", "SKIP", "FAIL") if key in counts
    )
    print()
    print(f"{platform.system()} {platform.machine()} py{sys.version.split()[0]}: {summary}")

    if args.json:
        payload = {
            "host": {
                "platform": sys.platform,
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "python": sys.version.split()[0],
            },
            "counts": counts,
            "results": [
                {"name": r.name, "status": r.status, "detail": r.detail, "extra": r.extra}
                for r in results
            ],
        }
        Path(args.json).write_text(
            json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.json}")

    if args.keep:
        print(f"sandbox kept at {root}")
    else:
        shutil.rmtree(root, ignore_errors=True)

    return 0 if counts.get("FAIL", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
