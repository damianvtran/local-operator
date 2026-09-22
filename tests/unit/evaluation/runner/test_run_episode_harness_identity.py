"""The sealed harness identity is read from the install that is RUNNING.

WHY THIS FILE EXISTS. The paid campaign's sealed bundles declared harness version
``0.61.9`` while the build that produced them was ``0.61.11``: the operator
script read ``<its own directory>/../pyproject.toml`` -- the working tree it
happened to be sitting in, which was the SHARED CHECKOUT, not the copy the
interpreter was running. Its ``harness_git_revision`` was wrong the same way, a
40-hex digest of that checkout's HEAD. A bundle whose declared harness version
and revision are not the build that ran cannot be compared to another bundle,
which is the only thing those two fields are for.

So both fields are asserted here to come from the executing install
(``local_operator.update.installed_build``), and neither may be answered by a
file beside the script. The decoy-checkout case is the one that discriminates:
the same tree that produced the mislabelled bundles, in miniature.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from local_operator import update
from scripts import run_episode

REPO = Path(__file__).resolve().parents[4]
SCRIPT = REPO / "scripts" / "run_episode.py"

#: A version no install in this test universe reports, so an answer that comes
#: from somewhere else (a checkout, a default) is visible rather than plausible.
SENTINEL = "9.9.9"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _checkout_version() -> str:
    """The version in the checkout beside ``scripts/run_episode.py``."""

    with (REPO / "pyproject.toml").open("rb") as handle:
        return str(tomllib.load(handle)["project"]["version"])


def test_the_harness_version_is_what_the_running_install_reports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The install's own metadata wins, and the adjacent checkout cannot answer.

    The guard assertion is what makes this discriminating rather than trivially
    true: the sentinel differs from the checkout's declared version, so an
    implementation that reads the file beside the script answers with the wrong
    number here instead of passing.
    """

    assert _checkout_version() != SENTINEL, "the sentinel must not be the checkout's version"
    monkeypatch.setattr(update, "installed_version", lambda: SENTINEL)

    assert run_episode._harness_version() == SENTINEL


def test_a_bare_checkout_with_no_install_records_an_absent_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No install means no version to claim, and a fabricated one is worse.

    The manifest field is a required identifier, so the answer is the sentinel
    ``0.0.0`` rather than a number borrowed from whatever tree is lying around.
    """

    monkeypatch.setattr(update, "installed_version", lambda: "")

    assert run_episode._harness_version() == "0.0.0"


def test_a_recorded_commit_is_what_the_revision_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """The install's own ``.lop-source`` ref, hashed to the 64 hex the field wants."""

    commit = "b" * 40
    monkeypatch.setattr(update, "installed_version", lambda: SENTINEL)
    monkeypatch.setattr(update, "source_ref", lambda prefix=None: commit)

    assert run_episode._harness_git_revision() == _digest(commit)


def test_an_install_with_no_recorded_commit_hashes_its_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wheel built elsewhere has no commit, and must not borrow one.

    The predecessor answered this case with ``git -C <script dir>/..``, which is
    a commit -- just not necessarily of the code that ran. Hashing the version
    records what IS known and is stable, and cannot name a commit that was never
    recorded.
    """

    monkeypatch.setattr(update, "installed_version", lambda: SENTINEL)
    monkeypatch.setattr(update, "source_ref", lambda prefix=None: "")

    assert run_episode._harness_git_revision() == _digest(SENTINEL)
    assert run_episode._harness_version() == SENTINEL


def test_a_checkout_sitting_beside_the_script_cannot_answer_for_the_build(
    tmp_path: Path,
) -> None:
    """THE MEASURED SHAPE, replayed: a copy of the script in a decoy checkout.

    The copy lives in a tree whose ``pyproject.toml`` declares a version no
    install of this harness reports -- exactly what the shared checkout was to
    the 0.61.11 build that ran the campaign. Run for real, in a subprocess, so
    ``Path(__file__)`` points into the decoy: a reader of that file is what the
    bundles were produced by, and what must no longer be believed.
    """

    decoy = tmp_path / "decoy-checkout"
    (decoy / "scripts").mkdir(parents=True)
    (decoy / "scripts" / "run_episode.py").write_bytes(SCRIPT.read_bytes())
    (decoy / "pyproject.toml").write_text(
        '[project]\nname = "local-operator"\nversion = "7.7.7"\n', encoding="utf-8"
    )
    driver = tmp_path / "driver.py"
    driver.write_text(
        "import importlib.util, json, sys\n"
        f"path = {str(decoy / 'scripts' / 'run_episode.py')!r}\n"
        "spec = importlib.util.spec_from_file_location('decoy_run_episode', path)\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "sys.modules['decoy_run_episode'] = module\n"
        "spec.loader.exec_module(module)\n"
        "from local_operator import update\n"
        "stamp = update.installed_build()\n"
        "print(json.dumps({\n"
        "    'reported_version': module._harness_version(),\n"
        "    'reported_revision': module._harness_git_revision(),\n"
        "    'install_version': stamp.version,\n"
        "    'install_ref': stamp.source_ref,\n"
        "}))\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [sys.executable, str(driver)],
        capture_output=True,
        text=True,
        env={
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
            # The script resolves the harness under test, not an installed one.
            "PYTHONPATH": str(REPO),
        },
        check=False,
    )

    assert completed.returncode == 0, completed.stderr[-2000:]
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    # The decoy's declaration is not evidence about the running build.
    assert payload["reported_version"] != "7.7.7"
    assert payload["reported_version"] == payload["install_version"]
    assert payload["reported_revision"] == _digest(
        payload["install_ref"] or payload["install_version"]
    )
