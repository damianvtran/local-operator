"""The loop-cd modelling may only ADD what a command is ranked against.

PR #3964 taught the resolver where a ``cd`` inside a ``for`` body lands in
every iteration (``_peel_subshell_keyword``, ``_Walker._cwd_fan_for``) so the
field case ``for r in …; do (cd $r; git worktree add …); done`` records one
worktree per repository. Three review waves in a row (R1-M1, R2-M1, R3-M1)
then found a shape where that new model dropped an iteration's real cwd and a
certain ``~/.local-operator`` delete fell from base ``3bb7023``'s CRITICAL to
LOW. ``resolve_command`` now unions the pre-PR model's ranked facts into the
result whenever the new model ran, which makes the invariant structural.

This file pins it as a property over a corpus: every command the two PR test
files (``test_family_delete_ownership.py``, ``test_classification_accuracy.py``)
hand the resolver, recorded by wrapping ``resolve_command`` during a run of
both, plus every concrete reviewer row from rounds 1-3. For each, the head
rank must be ``>=`` base's, as an unrelated session and as the session that
created the directories the commands work in. Base ranks run through the
CURRENT ``classify`` (only ``resolve`` is swapped), so the property isolates
the cwd model from the PR's intended ownership relaxations.

Base ranks are PINNED in ``tests/fixtures/guardrails/loop_cd_base_ranks.json``
because CI checks out with ``fetch-depth: 2`` and has no ``3bb7023`` object.
Where the object is present (a developer checkout), base's ``resolve.py`` is
also loaded from git in memory and the pins are checked against it, so a
layout change cannot silently stale them. Regenerate pins with
``_live_base_ranks`` after changing the layout below.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from local_operator.guardrails import classify as classify_mod
from local_operator.guardrails import provenance as prov
from local_operator.guardrails import resolve as resolve_mod
from local_operator.guardrails.classify import Risk, classify_bash

BASE = "3bb7023"
OWNER = "aaaaaaaaaaaa"  # creates the layout's directories (and PARENT in the corpus)
UNRELATED = "999999999999"
REPO = Path(__file__).resolve().parents[3]
PINS = REPO / "tests" / "fixtures" / "guardrails" / "loop_cd_base_ranks.json"
#: Every directory the corpus cds into or names, created by OWNER's ``mkdir``.
LAYOUT = (
    "a/sub",
    "a/build",
    "a/.worktrees/x",
    "b/sub",
    "b/build",
    "build",
    "w",
    "work/ui",
    "minervaai/infrastructure/services/.worktrees",
    "minervaai/infrastructure/terraform-cloud-bootstrap",
)
SESSIONS = (OWNER, "bbbbbbbbbbbb", "cccccccccccc", "ffffffffffff", UNRELATED)


def _rows() -> list[dict[str, Any]]:
    return json.loads(PINS.read_text())


@pytest.fixture
def home(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    state = home / ".local-operator"
    for sid in SESSIONS:
        (state / "sessions" / sid / "scratchpad").mkdir(parents=True)
    (state / "sessions" / "bbbbbbbbbbbb" / "scratchpad" / "rig").mkdir()
    (state / "config.yml").write_text("version: 1\n")
    return home


@pytest.fixture
def layout(home: Path) -> Iterator[Path]:
    """OWNER's ledger rows for the layout, written by the real creation path.

    Function-scoped on purpose: the suite's autouse ``isolate_environment``
    repoints HOME per test, so a module-scoped state root would be read from
    the wrong HOME and every owner rank would silently lose its ledger rows.
    """
    from local_operator.session.runtime import registry

    with pytest.MonkeyPatch.context() as mp:
        for key, value in _env(home).items():
            mp.setenv(key, value)
        for key in [k for k in os.environ if k.startswith(("CMUX_", "GIT_", "LOP_"))]:
            mp.delenv(key, raising=False)
        mp.setenv("GIT_CONFIG_SYSTEM", "/dev/null")
        mp.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
        mp.setenv("GIT_TERMINAL_PROMPT", "0")
        # Synthetic process/forge facts: nobody running, no forge reachable.
        mp.setattr(prov, "_live_cwd_inside", lambda _: False)
        mp.setattr(prov, "_cwd_inventory_cache", None)
        mp.setattr(prov, "_github_repo", lambda _: None)
        mp.setattr(prov, "_gitlab_repo", lambda _: None)
        mp.setattr(registry, "scan", lambda *a, **k: [])
        command = "mkdir -p " + " ".join(LAYOUT)
        pending = prov.plan_creations(command, cwd=str(home), env=_env(home))
        subprocess.run(["bash", "-c", command], cwd=home, check=True)
        assert pending is not None
        prov.commit_creations(pending, session_id=OWNER, env=_env(home), scratchpad=None)
        yield home


def _env(home: Path) -> dict[str, str]:
    return {"HOME": str(home), "LOCAL_OPERATOR_CONFIG_DIR": str(home / ".local-operator")}


def _cwd(home: Path, rel: str | None) -> str | None:
    """The recorded cwd under this home; ``None`` stays unknown, and a cwd
    outside the home (a test's ``../tmp``) or absent here falls back to it."""
    if rel is None:
        return None
    if rel.startswith("..") or not (home / rel).is_dir():
        return str(home)
    return str(home / rel)


def _rank(home: Path, row: dict[str, Any], sid: str) -> Risk:
    pad = home / ".local-operator" / "sessions" / sid / "scratchpad"
    env = {**_env(home), "LOCAL_OPERATOR_SCRATCHPAD": str(pad)}
    command = row["cmd"].replace("{home}", str(home))
    verdict = classify_bash(
        command, cwd=_cwd(home, row["cwd"]), env=env, scratchpad=str(pad), session_id=sid
    )
    return verdict.risk


def _load_base() -> types.ModuleType | None:
    """Base's ``resolve.py`` from git objects, in memory; ``None`` without them."""
    try:
        src = subprocess.run(
            ["git", "show", f"{BASE}:local_operator/guardrails/resolve.py"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    module = types.ModuleType("_resolve_base_3bb7023")
    module.__file__ = f"<git {BASE}>"
    sys.modules[module.__name__] = module  # dataclasses look their module up
    exec(compile(src, module.__file__, "exec"), module.__dict__)  # noqa: S102
    return module


def _converted(res: Any) -> resolve_mod.Resolution:
    def conv(obj: Any, cls: type) -> Any:
        return cls(**{f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)})

    return resolve_mod.Resolution(
        targets=[conv(t, resolve_mod.Target) for t in res.targets],
        opaque_reasons=list(res.opaque_reasons),
        created=[conv(c, resolve_mod.Creation) for c in res.created],
        touched=[conv(t, resolve_mod.Target) for t in res.touched],
        blind=list(res.blind),
        globstar=res.globstar,
    )


def _live_base_ranks(home: Path, base: types.ModuleType) -> list[dict[str, str]]:
    """Base ranks for every row: base ``resolve`` behind the current ``classify``."""
    original = classify_mod.resolve_command
    classify_mod.resolve_command = lambda *a, **k: _converted(base.resolve_command(*a, **k))
    try:
        return [
            {"unrelated": _rank(home, row, UNRELATED).name, "owner": _rank(home, row, OWNER).name}
            for row in _rows()
        ]
    finally:
        classify_mod.resolve_command = original


def test_corpus_covers_every_review_round() -> None:
    sources = {row["source"] for row in _rows()}
    assert {"test_family_delete_ownership", "test_classification_accuracy", "review"} <= sources
    # The six R3-M1 rows are in the corpus and pinned CRITICAL at base.
    pinned = {r["cmd"]: r["base"]["unrelated"] for r in _rows()}
    assert [pinned.get(c) for c in R3_M1] == ["CRITICAL"] * len(R3_M1)


def test_head_never_ranks_below_base(layout: Path) -> None:
    home = layout
    base = _load_base()
    if base is not None:
        live = _live_base_ranks(home, base)
        stale = [
            (row["cmd"][:120], row["base"], now)
            for row, now in zip(_rows(), live, strict=True)
            if row["base"] != now
        ]
        assert not stale, f"pins disagree with live base {BASE}: {stale[:5]}"
    lowered = []
    for row in _rows():
        for role, sid in (("unrelated", UNRELATED), ("owner", OWNER)):
            head = _rank(home, row, sid)
            if head < Risk[row["base"][role]]:
                lowered.append((role, row["base"][role], head.name, row["cmd"][:160]))
    assert not lowered, lowered


R3_M1 = [
    "cd ~; for r in a .; do (cd $r; cd /tmp; cd -; rm -rf .local-operator); done",
    "cd ~; for r in a .; do (cd $r; cd ~; cd -; rm -rf .local-operator); done",
    'cd ~; for r in a .; do (cd $r; cd /tmp; cd "$OLDPWD"; rm -rf .local-operator); done',
    "cd ~/a; for r in sub .; do (cd $r; cd ..; cd ~; cd -; rm -rf .local-operator); done",
    "cd ~/a; for r in sub .; do for s in ..; do (cd $r; cd $s; cd /tmp; cd -;"
    " rm -rf .local-operator); done; done",
    "cd ~; for r in a .; do for s in .; do (cd $r; cd $s; cd /tmp; cd -;"
    " rm -rf .local-operator); done; done",
]


@pytest.mark.parametrize("command", R3_M1)
def test_cd_back_to_oldpwd_restores_the_loop_fan(layout: Path, command: str) -> None:
    """Review R3-M1: bash runs ``rm`` at ``~/a`` then at ``~``. The fan the
    absolute ``cd`` left is OLDPWD's in every iteration, so ``cd -`` brings it
    back — in the current model on its own, not only through the base union."""
    home = layout
    row = {"cmd": command, "cwd": "."}
    assert _rank(home, row, UNRELATED) is Risk.CRITICAL
    walked = resolve_mod._walk_once(
        command, cwd=str(home), env=_env(home), home=str(home), model=resolve_mod._CwdModel()
    )
    paths = {t.path for t in walked.resolution.targets}
    assert str(home / ".local-operator") in paths
    assert str(home / "a" / ".local-operator") in paths


@pytest.mark.parametrize(
    "command",
    [
        "cd ~; for r in $(ls) .; do (cd $r; rm -rf .local-operator); done",
        "cd ~; for r in . $(ls); do (cd $r; rm -rf .local-operator); done",
    ],
)
def test_opaque_plus_literal_loop_keeps_base_critical(layout: Path, command: str) -> None:
    """Review R3-m1: base's model (``cd`` unseen) names ``~/.local-operator``;
    the union keeps that CRITICAL where the fan alone gave HIGH ``[None]``."""
    assert _rank(layout, {"cmd": command, "cwd": "."}, UNRELATED) is Risk.CRITICAL


def test_base_pass_runs_only_when_the_loop_model_engaged(layout: Path) -> None:
    """The second walk is skipped for commands the new model never touches."""
    home = layout
    env = _env(home)
    plain = resolve_mod._walk_once(
        "rm -rf build", cwd=str(home), env=env, home=str(home), model=resolve_mod._CwdModel()
    )
    looped = resolve_mod._walk_once(
        "for r in a; do rm -rf $r/build; done",
        cwd=str(home),
        env=env,
        home=str(home),
        model=resolve_mod._CwdModel(),
    )
    peeled = resolve_mod._walk_once(
        "if true; then (cd a; rm -rf build); fi",
        cwd=str(home),
        env=env,
        home=str(home),
        model=resolve_mod._CwdModel(),
    )
    assert (plain.engaged, looped.engaged, peeled.engaged) == (False, True, True)
