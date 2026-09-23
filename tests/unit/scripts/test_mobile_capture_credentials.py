"""The phone capture rigs must not carry a REUSABLE credential.

A fixed fixture password outlives the run it was written for: it lives in the repo,
every script that imports it shares one value, and any transcript, log or agent
context that merely READS those files picks it up. That happened here — one of these
constants reached a reviewer's context and had to be rotated — so the rig now mints a
value per run and every entry point FAILS CLOSED without one.

These tests assert the SHAPE (no literal, no default, no printing) and the BEHAVIOUR
(refusal without a credential, acceptance with one), and they never print a value:
the subprocess checks assert on the exit status and on a fixed word, and one asserts
the value is absent from the output rather than comparing it.
"""

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = REPO_ROOT / "scripts"

#: Every entry point that resolves one of these passwords.
PASSWORD_SOURCES = (
    "mobile_delegating_fixture.py",
    "mobile_delegating_shot.py",
    "mobile_overflow_fixture.py",
    "mobile_overflow_capture.py",
    "mobile_reachability_check.py",
)

PASSWORD_ENV = "LOP_MOBILE_FIXTURE_PASSWORD"

CREDENTIAL_NAME = re.compile(r"(pass(word)?|secret|token|credential|api_?key)", re.IGNORECASE)
#: `token_urlsafe`-ish output: mixed case, a digit, and long enough to be a secret.
SECRET_SHAPE = re.compile(r"^[A-Za-z0-9_\-\.]{12,}$")


def _tree(name: str) -> ast.Module:
    return ast.parse((SCRIPTS / name).read_text(encoding="utf-8"))


def _secret_shaped_strings(name: str) -> list[int]:
    """Line numbers of string literals that LOOK like a minted credential."""
    found = []
    for node in ast.walk(_tree(name)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
            if not SECRET_SHAPE.match(value):
                continue
            if (
                any(c.isdigit() for c in value)
                and any(c.isupper() for c in value)
                and any(c.islower() for c in value)
            ):
                found.append(node.lineno)
    return found


def _literal_credential_assignments(name: str) -> list[int]:
    """Assignments whose TARGET is credential-named and whose VALUE is a literal."""
    found = []
    for node in ast.walk(_tree(name)):
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name) or not CREDENTIAL_NAME.search(target.id):
                continue
            # The env var NAME is a name, not a value: it is checked separately.
            if target.id.endswith("_ENV"):
                continue
            found.append(node.lineno)
    return found


@pytest.mark.parametrize("name", PASSWORD_SOURCES)
def test_no_capture_script_carries_a_credential_literal(name: str) -> None:
    assert _secret_shaped_strings(name) == [], f"{name} holds a credential-shaped literal"
    assert _literal_credential_assignments(name) == [], f"{name} assigns a literal credential"


@pytest.mark.parametrize("name", PASSWORD_SOURCES)
def test_only_the_environment_variable_name_is_a_credential_named_constant(name: str) -> None:
    """A NAME may be fixed; a VALUE may not. Keep the two distinguishable."""
    for node in ast.walk(_tree(name)):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id.endswith("_ENV"):
                assert isinstance(node.value, ast.Constant), target.id
                assert node.value.value == PASSWORD_ENV, f"{target.id} names something else"


@pytest.mark.parametrize(
    "name",
    ("mobile_delegating_fixture.py", "mobile_overflow_fixture.py", "mobile_overflow_capture.py"),
)
def test_every_resolver_refuses_rather_than_defaulting(name: str) -> None:
    source = (SCRIPTS / name).read_text(encoding="utf-8")
    assert "SystemExit" in source, f"{name} has no refusal path"
    assert PASSWORD_ENV in source, f"{name} does not name the variable it reads"
    # The resolver must READ the variable rather than fall back to anything: a default
    # is the reusable credential again (its absence is pinned by the literal scan
    # above), and a silently minted value would leave a human unable to log in.
    assert "os.environ.get" in source


@pytest.mark.parametrize("name", PASSWORD_SOURCES)
def test_nothing_prints_a_credential(name: str) -> None:
    """No print/format call may reference a credential-named identifier."""
    for node in ast.walk(_tree(name)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if called != "print":
            continue
        for arg in list(node.args) + [kw.value for kw in node.keywords]:
            for inner in ast.walk(arg):
                if isinstance(inner, ast.Name) and CREDENTIAL_NAME.search(inner.id):
                    assert inner.id.endswith("_ENV"), f"{name}: print references {inner.id}"


def _run(
    home: Path, args: list[str], *, env_extra: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    """Run in a fresh interpreter, with HOME redirected.

    The capture module re-homes itself on import (`scripts.probe_isolation`), but a
    child that came up BEFORE that import must never read the operator's own config,
    and an UNSET ``HOME`` is worse than a wrong one: Python treats it as ``/``.
    """
    env = {
        "HOME": str(home),
        "LOCAL_OPERATOR_CONFIG_DIR": str(home / ".local-operator"),
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(REPO_ROOT),
        "TERM": "xterm-256color",
        **env_extra,
    }
    return subprocess.run(args, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=120)


def test_the_capture_refuses_without_a_credential_and_prints_none(tmp_path: Path) -> None:
    done = _run(
        tmp_path,
        [
            sys.executable,
            "-c",
            "import scripts.mobile_overflow_capture as m; m.fixture_password()",
        ],
        env_extra={},
    )
    assert done.returncode != 0, done.stdout
    assert PASSWORD_ENV in done.stderr, done.stderr
    # A refusal must not carry a value, not even a generated one.
    assert SECRET_SHAPE.match(done.stderr.strip()) is None


def test_the_capture_accepts_a_per_run_credential_without_echoing_it(tmp_path: Path) -> None:
    minted = subprocess.run(
        [sys.executable, "-c", "import secrets; print(secrets.token_urlsafe(16))"],
        env={"PATH": os.environ.get("PATH", "")},
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout.strip()
    assert len(minted) >= 16, "the generator must actually mint something"

    done = _run(
        tmp_path,
        [
            sys.executable,
            "-c",
            "import scripts.mobile_overflow_capture as m; "
            "print('ok' if m.fixture_password() else 'empty')",
        ],
        env_extra={PASSWORD_ENV: minted},
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "ok"
    assert minted not in done.stdout and minted not in done.stderr


@pytest.mark.parametrize("name", ("mobile_delegating_fixture.py", "mobile_overflow_fixture.py"))
def test_the_fixtures_refuse_to_start_without_a_credential(tmp_path: Path, name: str) -> None:
    done = _run(tmp_path, [sys.executable, str(SCRIPTS / name), "4199"], env_extra={})
    assert done.returncode != 0, f"{name} started without a credential"
    assert PASSWORD_ENV in (done.stdout + done.stderr)
    assert SECRET_SHAPE.match(done.stdout.strip()) is None


def test_the_delegating_shot_mints_a_fresh_credential_per_run() -> None:
    """The one place the value is BORN: `secrets`, inside the run, never a constant."""
    source = (SCRIPTS / "mobile_delegating_shot.py").read_text(encoding="utf-8")
    assert "secrets.token_urlsafe" in source
    # ...and the fixture it starts is handed that value as an argument.
    assert "mobile_delegating_fixture.py" in source
    for node in ast.walk(_tree("mobile_delegating_shot.py")):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "Popen":
            assert any(
                isinstance(arg, ast.List)
                and any(
                    isinstance(el, ast.Constant) and "mobile_delegating_fixture" in str(el.value)
                    for el in arg.elts
                )
                for arg in node.args
            )
