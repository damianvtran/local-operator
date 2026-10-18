#!/usr/bin/env python3
"""Install a local-operator wheel into a throwaway venv and start it.

WHY THIS EXISTS
---------------
v0.77.0 shipped to PyPI unstartable on Python 3.14. ``import
local_operator.server.app`` raised::

    TypeError: 'function' object is not subscriptable

``RadientTranscriptionResponseData`` annotates ``removed: list[dict[str, Any]]``
in a class body that also defines ``def dict(...)``. Under PEP 649 the deferred
annotation is evaluated lazily in that class namespace, so ``dict`` resolved to
the method rather than the builtin. The source tree was green on the CI Python
(3.12); the artifact was not, on the interpreter the project's own
``requires-python`` and classifiers admit.

Neither existing gate could see it:

* ``ci.yml``'s ``server-sanity`` does ``pip install -e ".[server]"`` on 3.12 —
  it tests the source tree, never the built artifact;
* ``publish.yml`` builds and uploads the wheel but never installs or starts it;
* ``lop --version`` exits 0 on the broken wheel, because it reads installed
  metadata and deliberately avoids importing the server stack. A version check
  is not a startability check.

So this script exercises the artifact the user will actually run, in the TWO
INSTALL SHAPES the project ships:

* **base** (``pip install local-operator``) — the shape every ``uv tool install``
  / ``lop update`` user gets. It imports ``local_operator``,
  ``local_operator.cli`` and ``local_operator.clients.radient`` and checks
  ``lop --version``. ``clients.radient`` is the discriminator: it is part of the
  base install and it is the module that raised the v0.77.0 TypeError.
* **server extra** (``pip install "local-operator[server]"``) — installed INTO
  THE SAME venv after the base phase, because that is what a user adding the
  extra does. It imports ``local_operator.server.app`` and starts ``lop serve``.

A ``[server]``-only smoke would miss the base shape entirely, and a base-only
smoke would not start the HTTP surface. The base phase costs one extra pip
resolution (~5-10s), not another CI job.

WHY THE IMPORT RUNS OUTSIDE THE REPO
------------------------------------
``import local_operator`` from the repository root resolves to the source tree,
which is exactly the thing that was green while the wheel was broken. The import
subprocess therefore runs with ``cwd`` set to a scratch directory and with
``-I`` (isolated: no ``PYTHONPATH``, no user site), and asserts the reported
``__file__`` is under the venv's ``site-packages``. Without that assertion a
green check could be a check of the checkout.

Only the standard library is used: this must run on the interpreter CI selected
(3.12/3.13/3.14) before the project is installed anywhere, and it must not add a
dependency to the smoke it is testing.

USAGE
-----
    python scripts/smoke_wheel.py --wheel dist/local_operator-0.79.0-py3-none-any.whl
    python scripts/smoke_wheel.py --index-version 0.79.0

``--expect-failure`` is the negative control: it requires the BASE-install import
(the ``clients.radient`` module, the one that actually raised) to FAIL and
exits 0, proving the harness detects the v0.77.0 failure mode rather than
passing vacuously.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import venv
from pathlib import Path

INDEX_URL = "https://pypi.org/simple"
DEFAULT_SERVER_TIMEOUT = 90
DEFAULT_INDEX_WAIT = 300
#: Installation pulls ~10 server dependencies; the bound is generous and exists
#: only so a wedged pip cannot hang the job. It is not the server-start timeout
#: (`--timeout`), which is the one a caller tunes.
INSTALL_TIMEOUT = 900
#: Lines of captured output shown on failure. Full logs live in the venv root,
#: but the root is removed by then, so the useful tail is printed eagerly.
TAIL_LINES = 40
#: A wheel filename is ``local_operator-<version>[-<build>]-<python>-<abi>-<plat>.whl``
#: and a normalized version cannot contain ``-``.
WHEEL_VERSION_RE = re.compile(r"^local_operator-(?P<version>[^-]+)-")

#: pip error fragments that mean "the index has not caught up yet", and nothing
#: else. Every other failure (bad dependency set, missing build tag, network
#: refused) is a real failure and must not be retried into a 300 second timeout.
PROPAGATION_ERRORS = (
    "no matching distribution",
    "could not find a version that satisfies",
    "404 client error",
    "not found for url",
)

#: The base install's import probe. `clients.radient` is part of the base
#: install and is the v0.77.0 discriminator (the annotation that resolved `dict`
#: to the method). `local_operator.cli` guards the console-script import path.
BASE_IMPORT = (
    "import local_operator; import local_operator.cli; "
    "import local_operator.clients.radient; print(local_operator.__file__)"
)

#: The server extra's import probe, the v0.77.0 symptom itself.
SERVER_IMPORT = (
    "import local_operator.server.app; import local_operator; print(local_operator.__file__)"
)


class SmokeError(RuntimeError):
    """A check failed; ``args[0]`` is the message shown to the user."""


def _step(name: str, passed: bool, detail: str = "") -> None:
    suffix = f" - {detail}" if detail else ""
    print(f"STEP {name}: {'PASS' if passed else 'FAIL'}{suffix}", flush=True)


def _tail(text: str | bytes | None) -> str:
    if not text:
        return "<empty>"
    if isinstance(text, bytes):
        text = text.decode("utf-8", "replace")
    lines = text.rstrip().splitlines()
    if len(lines) > TAIL_LINES:
        lines = lines[-TAIL_LINES:]
    return "\n".join(lines)


@contextlib.contextmanager
def _without_inherited_python_env():
    """Hide interpreter-selection variables while `venv` shells out.

    Only the `EnvBuilder` call needs this; every other subprocess gets the
    explicit `_clean_env` mapping. Restored in `finally` so the harness's own
    Python is unaffected.
    """
    saved = {
        name: os.environ[name]
        for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")
        if name in os.environ
    }
    try:
        for name in saved:
            del os.environ[name]
        yield
    finally:
        os.environ.update(saved)


def _clean_env(root: Path) -> dict[str, str]:
    """A deliberately minimal environment for every subprocess.

    Inherited ``PYTHONPATH``/``PYTHONHOME``/``VIRTUAL_ENV`` would point at the
    caller's interpreter or checkout; ``CMUX_*``/``LOP_*`` belong to a host
    session and must not steer the artifact. ``HOME`` and
    ``LOCAL_OPERATOR_CONFIG_DIR`` are inside the scratch root so nothing this
    smoke writes can reach the operator's real config or credentials.
    """
    home = root / "home"
    config = root / "config"
    home.mkdir()
    config.mkdir()
    return {
        "PATH": os.defpath,
        "HOME": str(home),
        "LOCAL_OPERATOR_CONFIG_DIR": str(config),
        "TERM": "xterm-256color",
        "LANG": os.environ.get("LANG", "C.UTF-8"),
    }


def _run(
    cmd: list[str],
    *,
    env: dict[str, str],
    cwd: Path,
    timeout: float,
    label: str,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            cmd,
            env=env,
            cwd=cwd,
            timeout=timeout,
            text=True,
            capture_output=True,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise SmokeError(
            f"{label} timed out after {timeout:.0f}s\n{_tail(exc.stdout)}\n{_tail(exc.stderr)}"
        ) from exc


def _python_paths(python: Path, env: dict[str, str], cwd: Path) -> Path:
    """The venv's ``purelib``, asked of the venv's own interpreter.

    Computed rather than guessed from ``lib/pythonX.Y`` so a platform layout
    difference cannot make the ``__file__`` assertion compare against the wrong
    directory.
    """
    proc = _run(
        [str(python), "-I", "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        env=env,
        cwd=cwd,
        timeout=60,
        label="venv path probe",
    )
    if proc.returncode != 0:
        raise SmokeError(f"could not read the venv's purelib path\n{_tail(proc.stderr)}")
    return Path(proc.stdout.strip())


def _wheel_path(args: argparse.Namespace) -> Path:
    # Resolve here, once: pip runs with cwd set to the scratch directory, so a
    # relative wheel path from the caller would otherwise not be found.
    return Path(args.wheel).expanduser().resolve()


def _expected_version(args: argparse.Namespace) -> str:
    if args.index_version:
        return args.index_version
    match = WHEEL_VERSION_RE.match(_wheel_path(args).name)
    if not match:
        raise SmokeError(
            f"cannot read a version from wheel filename {_wheel_path(args).name!r}; "
            "expected local_operator-<version>-...whl"
        )
    return match.group("version")


def _install(
    python: Path,
    spec: str,
    *,
    from_index: bool,
    index_wait: float,
    env: dict[str, str],
    cwd: Path,
    label: str,
) -> subprocess.CompletedProcess[str]:
    command = [str(python), "-m", "pip", "install", "--no-cache-dir", "--disable-pip-version-check"]
    if from_index:
        command += ["--index-url", INDEX_URL, spec]
        deadline = time.monotonic() + index_wait
        while True:
            # PyPI propagation is eventually consistent: a release that the
            # upload job just published can 404 on the simple index for a
            # minute or two. Retry ONLY that class of failure — retrying a
            # broken dependency resolution would just delay an honest red.
            last = _run(command, env=env, cwd=cwd, timeout=INSTALL_TIMEOUT, label=label)
            if last.returncode == 0:
                return last
            combined = f"{last.stdout}\n{last.stderr}".lower()
            retryable = any(fragment in combined for fragment in PROPAGATION_ERRORS)
            if not retryable or time.monotonic() >= deadline:
                return last
            print(
                f"  pip has not seen {spec} yet; retrying for up to "
                f"{index_wait:.0f}s (index propagation)",
                flush=True,
            )
            time.sleep(10)
    command.append(spec)
    return _run(command, env=env, cwd=cwd, timeout=INSTALL_TIMEOUT, label=label)


def _resolved_version(python: Path, env: dict[str, str], cwd: Path) -> str:
    proc = _run(
        [str(python), "-m", "pip", "show", "local-operator"],
        env=env,
        cwd=cwd,
        timeout=60,
        label="pip show",
    )
    if proc.returncode != 0:
        raise SmokeError(f"pip show local-operator failed\n{_tail(proc.stderr)}")
    for line in proc.stdout.splitlines():
        if line.lower().startswith("version:"):
            return line.split(":", 1)[1].strip()
    raise SmokeError(f"pip show printed no Version line\n{_tail(proc.stdout)}")


def _import_check(
    python: Path, code: str, label: str, env: dict[str, str], cwd: Path
) -> subprocess.CompletedProcess[str]:
    # -I plus cwd=<scratch> is the load-bearing isolation: from the repo root
    # this would import the checkout and prove nothing about the wheel.
    return _run(
        [str(python), "-I", "-c", code],
        env=env,
        cwd=cwd,
        timeout=120,
        label=label,
    )


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _reap(proc: subprocess.Popen[bytes]) -> None:
    """Kill the server's whole process group; uvicorn may spawn children.

    Deliberately does NOT return early when the leader has already exited: a
    worker it spawned can outlive it, and that is exactly the leak `--port`
    reuse would then inherit. ``start_new_session=True`` guarantees the leader's
    pid is the group id.
    """
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass


def _serve_check(
    lop: Path,
    expected_version: str,
    env: dict[str, str],
    cwd: Path,
    timeout: float,
    log_path: Path,
) -> None:
    port = _free_loopback_port()
    # A fresh process session gives the whole uvicorn tree one pgid to reap, so
    # neither a worker nor a re-parented child survives a failed assertion.
    with log_path.open("wb") as log:
        proc = subprocess.Popen(
            [str(lop), "serve", "--port", str(port)],
            env=env,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        url = f"http://127.0.0.1:{port}/health"
        deadline = time.monotonic() + timeout
        last_error = "no response"
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise SmokeError(
                    f"`lop serve` exited with {proc.returncode} before /health answered\n"
                    f"{_tail(log_path.read_text(errors='replace'))}"
                )
            try:
                with urllib.request.urlopen(url, timeout=2) as response:
                    status = response.status
                    body = json.loads(response.read().decode("utf-8"))
                break
            except (urllib.error.URLError, ConnectionError, OSError, json.JSONDecodeError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(1)
        else:
            raise SmokeError(
                f"/health did not answer within {timeout:.0f}s ({last_error})\n"
                f"{_tail(log_path.read_text(errors='replace'))}"
            )

        problems = []
        if not isinstance(body, dict):
            raise SmokeError(f"/health returned JSON that is not an object: {body!r}")
        if status != 200:
            problems.append(f"HTTP status is {status}, expected 200")
        if body.get("status") != 200:
            problems.append(f"body status is {body.get('status')!r}, expected 200")
        if body.get("message") != "ok":
            problems.append(f"body message is {body.get('message')!r}, expected 'ok'")
        result = body.get("result") or {}
        if result.get("version") != expected_version:
            problems.append(
                f"result.version is {result.get('version')!r}, expected {expected_version!r}"
            )
        # The proof that the artifact being exercised is the installed one: a
        # source checkout reports `editable`, a uv tool install `uv-tool`.
        if result.get("install_kind") != "pip":
            problems.append(
                f"result.install_kind is {result.get('install_kind')!r}, expected 'pip'"
            )
        if problems:
            raise SmokeError(
                "/health answered but the payload is wrong:\n  " + "\n  ".join(problems)
            )
        print(f"  /health: {json.dumps(body, sort_keys=True)}", flush=True)
    finally:
        _reap(proc)


def _run_checks(args: argparse.Namespace, root: Path) -> None:
    expected = _expected_version(args)
    env = _clean_env(root)
    scratch = root / "scratch"
    scratch.mkdir()
    venv_dir = root / "venv"
    bin_dir = venv_dir / "bin"
    python = bin_dir / "python"
    lop = bin_dir / "lop"

    # STEP 1 - an isolated interpreter, with its own HOME/config under the root.
    # EnvBuilder's own subprocesses inherit this process's environment, so the
    # same variables a caller could have set for an unrelated interpreter are
    # removed for the duration of creation; `env` below is what every later
    # subprocess gets.
    with _without_inherited_python_env():
        # `symlinks=True` matches the `python -m venv` CLI default on POSIX and
        # is not cosmetic: EnvBuilder defaults to COPYING the interpreter, and a
        # copy of a uv-managed CPython dies in ensurepip with SIGABRT (`python
        # -m venv` succeeds because the CLI passes symlinks=True). Measured on
        # this host, CPython 3.14.3 (uv-managed): EnvBuilder(with_pip=True) →
        # ensurepip SIGABRT; the same call with symlinks=True → venv OK.
        venv.EnvBuilder(with_pip=True, clear=True, symlinks=os.name != "nt").create(venv_dir)
    purelib = _python_paths(python, env, scratch)
    _step("venv", True, f"{venv_dir} (python {sys.version.split()[0]}, purelib {purelib})")

    # STEP 2 - the BASE install shape, which is what `uv tool install` and
    # `lop update` produce. Installed before the extra so the base shape is
    # proved on its own rather than inferred from the extra's success.
    wheel = _wheel_path(args) if args.wheel else None
    from_index = bool(args.index_version)
    base_spec = f"local-operator=={args.index_version}" if from_index else str(wheel)
    server_spec = (
        f"local-operator[server]=={args.index_version}" if from_index else f"{wheel}[server]"
    )
    install = _install(
        python,
        base_spec,
        from_index=from_index,
        index_wait=args.index_wait,
        env=env,
        cwd=scratch,
        label="pip install (base)",
    )
    if install.returncode != 0:
        _step("install-base", False)
        raise SmokeError(
            f"pip install (base) failed\n{_tail(install.stdout)}\n{_tail(install.stderr)}"
        )
    resolved = _resolved_version(python, env, scratch)
    print(f"  pip resolved local-operator {resolved}", flush=True)
    if resolved != expected:
        _step("install-base", False, f"resolved {resolved}, expected {expected}")
        raise SmokeError(
            f"pip installed local-operator {resolved}, but the artifact under test is {expected}"
        )
    _step("install-base", True, f"local-operator {resolved} (no extras)")

    # STEP 3 - the base imports, outside the repo, from the installed location.
    # `clients.radient` is deliberate: it ships in the base install and it is
    # the module whose v0.77.0 annotation raised the TypeError.
    imported = _import_check(python, BASE_IMPORT, "import base modules", env, scratch)
    if args.expect_failure:
        if imported.returncode == 0:
            _step("import-base", False, "expected the import to fail")
            raise SmokeError(
                "base import unexpectedly succeeded; the negative control proves nothing\n"
                f"reported module: {imported.stdout.strip()}"
            )
        _step("import-base", False, "failed as expected")
        print(_tail(imported.stderr), flush=True)
        print("EXPECTED FAILURE OBSERVED", flush=True)
        return
    if imported.returncode != 0:
        _step("import-base", False)
        raise SmokeError(f"base import failed\n{_tail(imported.stderr)}\n{_tail(imported.stdout)}")
    reported = imported.stdout.strip().splitlines()[-1] if imported.stdout.strip() else ""
    if not reported or purelib.resolve() not in Path(reported).resolve().parents:
        _step("import-base", False, "module was not imported from the venv")
        raise SmokeError(
            "base import succeeded but local_operator came from outside the venv's "
            f"site-packages:\n{imported.stdout.strip()}"
        )
    _step("import-base", True, imported.stdout.strip())

    # STEP 4 - the console script reports the version the artifact carries.
    version = _run(
        [str(lop), "--version"], env=env, cwd=scratch, timeout=120, label="lop --version"
    )
    if version.returncode != 0 or version.stdout.strip() != f"v{expected}":
        _step("version", False)
        raise SmokeError(
            f"`lop --version` returned rc={version.returncode}, stdout="
            f"{version.stdout.strip()!r} (expected 'v{expected}')\n{_tail(version.stderr)}"
        )
    _step("version", True, version.stdout.strip())

    # STEP 5 - add the server extra INTO THE SAME venv: that is what a user who
    # wants the HTTP surface does, and it must not require a fresh install.
    install = _install(
        python,
        server_spec,
        from_index=from_index,
        index_wait=args.index_wait,
        env=env,
        cwd=scratch,
        label="pip install (server extra)",
    )
    if install.returncode != 0:
        _step("install-server", False)
        raise SmokeError(
            f"pip install (server extra) failed\n{_tail(install.stdout)}\n{_tail(install.stderr)}"
        )
    _step("install-server", True, "local-operator[server]")

    # STEP 6 - the server import, from the installed location.
    imported = _import_check(
        python, SERVER_IMPORT, "import local_operator.server.app", env, scratch
    )
    if imported.returncode != 0:
        _step("import-server", False)
        raise SmokeError(
            f"server import failed\n{_tail(imported.stderr)}\n{_tail(imported.stdout)}"
        )
    reported = imported.stdout.strip().splitlines()[-1] if imported.stdout.strip() else ""
    if not reported or purelib.resolve() not in Path(reported).resolve().parents:
        _step("import-server", False, "module was not imported from the venv")
        raise SmokeError(
            "server import succeeded but local_operator came from outside the venv's "
            f"site-packages:\n{imported.stdout.strip()}"
        )
    _step("import-server", True, imported.stdout.strip())

    # STEP 7 - start the real server and read its own report of what is running.
    _serve_check(lop, expected, env, scratch, args.timeout, root / "serve.log")
    _step("serve", True, "/health reports status=200, message=ok, install_kind=pip")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--wheel", help="wheel file to install")
    source.add_argument("--index-version", help="version to install from https://pypi.org/simple")
    parser.add_argument(
        "--expect-failure",
        action="store_true",
        help="require the base-install import to fail (negative control); only valid with --wheel",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_SERVER_TIMEOUT,
        help=f"seconds to wait for the server's /health (default {DEFAULT_SERVER_TIMEOUT})",
    )
    parser.add_argument(
        "--index-wait",
        type=float,
        default=DEFAULT_INDEX_WAIT,
        help=f"seconds to retry a not-yet-propagated PyPI release (default {DEFAULT_INDEX_WAIT})",
    )
    args = parser.parse_args(argv)
    if args.expect_failure and not args.wheel:
        parser.error("--expect-failure requires --wheel")

    root = Path(tempfile.mkdtemp(prefix="local-operator-wheel-smoke-"))
    try:
        _run_checks(args, root)
    except SmokeError as exc:
        print(f"\nSMOKE FAILED\n{exc}", file=sys.stderr, flush=True)
        print(f"venv: {root / 'venv'}", file=sys.stderr, flush=True)
        return 1
    except Exception:
        # Unexpected exceptions still get the cleanup and the venv location;
        # re-raised so the traceback names the defect.
        print(f"venv: {root / 'venv'}", file=sys.stderr, flush=True)
        raise
    else:
        print("\nWHEEL SMOKE PASSED", flush=True)
        return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
