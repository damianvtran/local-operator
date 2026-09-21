"""The pnpm fetch guard's CHILD ENVIRONMENT, and the arm that builds from a seeded pin.

WHY THESE EXIST, AND WHAT THEY ARE A DELTA ON
---------------------------------------------
A release install ran ``pnpm install --frozen-lockfile`` in a tree pinning
``packageManager: pnpm@11.22.0`` while the pnpm on PATH was 10.30.3. pnpm resolves
that mismatch by installing the pinned version WITH PNPM (``switchCliVersion`` ->
``installPnpmToTools`` in its shipped bundle, i.e. ``pnpm add pnpm@11.22.0``), which
forked until the host had 0.1 GB free and was rebooted. ``tests/unit/mobile/
test_bundle_build_bound.py`` (PR #1394) ships the GROUP BOUND and the mismatch
refusal. This file pins the delta on top of those shapes, and re-tests none of them:

  * the child environment every package-manager command runs with
    (:func:`install._package_manager_env`) — the settings that make the fetch
    unreachable and make a wrong package manager FAIL rather than warn, and the
    shared homes a seeded machine resolves against;
  * the SATISFACTION ROUTES (:func:`install._seeded_pnpm` /
    :func:`install._runner_or_refusal`) — a pin already installed where pnpm keeps
    managed versions, and corepack, both of which let a machine that CAN build
    build instead of being refused;
  * the wiring that carries the environment to the children, because an armed
    environment nothing passes on is the shape an earlier review caught.

THE POLICY THIS DIFFERS FROM #1394's GUARD BY, stated because two guards for one
hazard disagreeing silently is worse than either policy: main refuses whenever the
runner's reported version is not the pin. This refuses only when NO route can
supply the pin — in order, the runner itself when it already reports the pin, then
a pin verified where pnpm keeps managed versions (local: nothing is downloaded),
then corepack (a bounded fetch that converges, and a route pnpm's own switch
cannot reach) — so on a host with a seeded 11.22.0 and a PATH 10.30.3, main
refuses and this builds. The trade-off is argued in the PR body; the boundary is
pinned here, and no route can fetch pnpm INTO the tree: the seeded arm requires the
candidate to ANSWER with the pin, through a probe that runs in an empty directory
with the fetch disarmed.

POSIX-only, and it says so rather than pretending: the stand-ins are executable
``#!/usr/bin/env python3`` files, which is how the seeded arm invokes a candidate on
this platform. The Windows spelling of the same paths (``pnpm.cmd``, the ``_shim_argv``
wrapper, ``taskkill``) is exercised where ``lop mobile install`` actually runs — CI's
probe battery, ``scripts/xplat_probe.py``.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

from local_operator.mobile import install
from tests.unit.mobile.test_bundle_build_bound import StandIn, _web

#: The group-bound file's stand-in is reused rather than re-implemented: it already
#: records every invocation (so a test can assert what the builder did NOT run),
#: answers ``--version`` from its config, and can create ``dist/`` for a build step.
#: One instrument, one owner — an edit there that breaks this import fails loudly at
#: collection rather than quietly changing what this file measures.
pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="executable stand-ins; Windows runs the probe battery"
)


def _seed(root: Path, pin: str, *, version: str, **config: object) -> StandIn:
    """Plant a pnpm stand-in where pnpm keeps managed versions, as ``pnpm`` would.

    ``<PNPM_HOME>/.tools/pnpm/<version>/bin/pnpm`` is the layout the switch looks in
    (``getToolDirPath`` in the shipped bundle), and ``root`` is expected to be the
    ``PNPM_HOME`` the test pointed the module at. The stand-in is COPIED to the name
    ``pnpm`` because that is what the switch re-executes, and it keeps reading the
    ``config.json`` beside it — from ITS directory, which is why the copy has to land
    in the same directory rather than somewhere on PATH. Both spellings log to that
    one directory, so the returned object sees the whole conversation.
    """
    bin_dir = root / ".tools" / "pnpm" / pin / "bin"
    stand_in = StandIn(bin_dir, version=version, **config)
    seeded = bin_dir / ("pnpm.cmd" if os.name == "nt" else "pnpm")
    shutil.copy(stand_in.script, seeded)
    seeded.chmod(0o755)
    return stand_in


def _corepack_stand_in(tools: Path) -> Path:
    """A ``corepack`` on PATH that records every call and can resolve the pin.

    Written with an ABSOLUTE interpreter in the shebang rather than
    ``#!/usr/bin/env python3``: the tests that use it run with ``PATH`` pointing at
    this directory ALONE, so that the host's own corepack (present on many
    machines, absent on this one) cannot decide what the reading means.

    The recorded line carries the two settings the guard is supposed to hand every
    package-manager child, because "corepack resolution *through* the bounded,
    armed runner" is only true if the child saw them.
    """
    script = tools / "corepack"
    script.write_text(
        "\n".join(
            [
                f"#!{sys.executable}",
                '"""Generated by tests/unit/mobile/test_pnpm_fetch_guard.py."""',
                "import json, os, sys",
                "from pathlib import Path",
                "HERE = Path(__file__).resolve().parent",
                'with (HERE / "corepack-calls.jsonl").open("a", encoding="utf-8") as handle:',
                "    handle.write(",
                "        json.dumps(",
                "            {",
                '                "argv": sys.argv[1:],',
                '                "manage": os.environ.get(',
                '                    "npm_config_manage_package_manager_versions"',
                "                ),",
                '                "strict": os.environ.get(',
                '                    "npm_config_package_manager_strict_version"',
                "                ),",
                "            }",
                "        )",
                '        + "\\n"',
                "    )",
                "args = sys.argv[1:]",
                'if args[:1] == ["enable"]:',
                "    raise SystemExit(0)",
                "args = args[1:]",
                'if "--version" in args:',
                '    print("11.22.0")',
                "    raise SystemExit(0)",
                'if args[:1] == ["build"]:',
                '    dist = Path(os.getcwd()) / "dist"',
                "    dist.mkdir(exist_ok=True)",
                '    (dist / "index.html").write_text("<html></html>", encoding="utf-8")',
                "raise SystemExit(0)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def _corepack_calls(tools: Path) -> list[dict[str, Any]]:
    log = tools / "corepack-calls.jsonl"
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]


def test_the_build_environment_disarms_self_install_and_shares_the_homes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The three things the build environment exists for, and what it must NOT do.

    The disarm is the load-bearing one: ``npm_config_manage_package_manager_versions
    =false`` is what pnpm's switch is gated on, so the ``pnpm add pnpm@<pin>`` fetch
    becomes unreachable rather than merely unlikely. It is defence in depth BEHIND
    the refusal (:func:`install._pin_mismatch`) and not a substitute: on its own it
    does not fail closed, which is why the refusal is not relaxed to rely on it.
    """
    home = tmp_path / "pnpm-home"
    corepack = tmp_path / "corepack-home"
    monkeypatch.setenv("PNPM_HOME", str(home))
    monkeypatch.setenv("COREPACK_HOME", str(corepack))
    web = _web(tmp_path, pin="pnpm@11.22.0")

    env = install._package_manager_env(web)

    assert env[install._MANAGE_PM_VERSIONS_ENV] == "false"
    assert env["PNPM_HOME"] == str(home)
    assert env["COREPACK_HOME"] == str(corepack)
    assert env[install._COREPACK_DOWNLOAD_PROMPT_ENV] == "0"
    # A COPY of the ambient environment plus these keys, never a replacement: the
    # child is a package manager that needs PATH, HOME and everything else the
    # operator's shell had.
    assert env["PATH"] == os.environ["PATH"]
    assert env["HOME"] == os.environ["HOME"]


@pytest.mark.parametrize(
    ("pin", "expected"),
    [
        ("pnpm@11.22.0", True),
        ("pnpm@11.22.0+sha512.deadbeef", True),
        ("pnpm@^11", False),
        ("npm@10.9.0", False),
        (None, False),
    ],
)
def test_only_an_exact_pnpm_pin_arms_the_fetch_and_the_strict_pair(
    tmp_path: Path, pin: str | None, expected: bool
) -> None:
    """A pin a fetch can act on is armed; a range is left alone — on BOTH counts.

    pnpm reaches its self-install only for a version ``semver.valid`` accepts, so a
    range is not a fetch and the disarm would turn pnpm's warn-and-continue into an
    error on a tree that works today. The strict pair is gated identically for a
    second reason: a range can never EQUAL the running version, so pnpm's strict
    version check would throw on a tree that is fine.
    """
    web = _web(tmp_path, pin=pin)
    env = install._package_manager_env(web)

    armed = install._MANAGE_PM_VERSIONS_ENV in env
    strict = install._STRICT_PM_VERSION_ENV in env and install._STRICT_PM_ENV in env

    assert armed is expected
    assert strict is expected
    if expected:
        assert env[install._STRICT_PM_VERSION_ENV] == "true"
        assert env[install._STRICT_PM_ENV] == "true"


def test_the_pin_under_dev_engines_disarms_it_too(tmp_path: Path) -> None:
    """pnpm 11's second spelling is the same hazard, so it is armed the same way.

    ``devEngines.packageManager`` with ``onFail: download`` is the spelling whose
    miss DOWNLOADS the manager; an exact pin there reaches the same switch.
    """
    web = _web(
        tmp_path,
        pin=None,
        dev_engines={"packageManager": {"name": "pnpm", "version": "11.22.0"}},
    )

    assert install._package_manager_env(web)[install._MANAGE_PM_VERSIONS_ENV] == "false"


def test_the_build_steps_carry_the_armed_child_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The WIRING, not just the helper: drop ``env=env`` and nothing else notices.

    That is what makes this worth an assertion rather than a comment (agent review
    round 1, m2): the child environment is the layer the refusal stands on, and
    without it an unwritable or misconfigured shared home silently drops back to
    whatever the operator's shell had — the state the incident happened in.

    Also pinned: the PROBE's environment does not relocate the homes. Relocating
    ``COREPACK_HOME`` for a probe would make the guard itself download a package
    manager on a host whose pnpm is a corepack shim with a cold cache.
    """
    home = tmp_path / "pnpm-home"
    monkeypatch.setenv("PNPM_HOME", str(home))
    fake = StandIn(tmp_path / "bin", version="11.22.0", exit=0, dist=True)
    web = _web(tmp_path, pin="pnpm@11.22.0")
    seen: list[Any] = []
    real = install._run_build_step

    def spy(argv: list[str], cwd: Path, **kwargs: Any) -> Any:
        seen.append(kwargs.get("env"))
        return real(argv, cwd, **kwargs)

    monkeypatch.setattr(install, "_run_build_step", spy)

    assert install._build_bundle(web, fake.runner) is None
    # The probe, then the two steps: one runner, so one place arms and one place
    # passes it on.
    assert len(seen) == 3, f"expected probe + two steps through the runner: {fake.argv_seen()}"
    for entry in seen:
        assert isinstance(entry, dict), "a child ran without an environment"
    probe, first_step, second_step = seen
    for step in (first_step, second_step):
        assert step[install._MANAGE_PM_VERSIONS_ENV] == "false"
        assert step["PNPM_HOME"] == str(home)
        assert step["COREPACK_HOME"] == str(install._corepack_home())
    assert probe[install._MANAGE_PM_VERSIONS_ENV] == "false"
    assert probe.get("COREPACK_HOME") == os.environ.get("COREPACK_HOME"), (
        "the probe moves nothing: a relocated corepack cache would make the guard "
        "itself download a package manager"
    )


def test_a_seeded_pin_is_used_only_when_it_answers_with_the_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A directory that EXISTS is not a pin — it has to answer, and that is the point.

    pnpm's switch treats ``<PNPM_HOME>/.tools/pnpm/<version>/bin`` existing as "the
    pinned version is installed" and re-executes whatever is inside. The residue this
    incident left is exactly that trap: this host carries ~14,800 ``11.22.0_tmp_<pid>``
    staging directories under ``~/Library/pnpm/.tools/pnpm/`` and no completed
    ``11.22.0`` at all, so an existence test would have "found" the pin in a
    half-written fetch and re-executed it.
    """
    home = tmp_path / "pnpm-home"
    monkeypatch.setenv("PNPM_HOME", str(home))
    bin_dir = home / ".tools" / "pnpm" / "11.22.0" / "bin"
    bin_dir.mkdir(parents=True)
    half_written = bin_dir / "pnpm"
    half_written.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    half_written.chmod(0o755)

    assert install._seeded_pnpm("11.22.0") is None, "a candidate that cannot answer is not a pin"

    _seed(home, "11.22.0", version="11.22.0")

    assert install._seeded_pnpm("11.22.0") == [str(bin_dir / "pnpm")]


def test_the_seeded_pin_builds_where_the_path_pnpm_would_be_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE DELTA, in one reading: main refuses this machine; this one builds on it.

    A tree pinning 11.22.0, a pnpm 10.30.3 on PATH, and a VERIFIED 11.22.0 already
    installed where pnpm keeps managed versions. #1394's guard refuses this —
    correctly, for the hazard it knows about — and this arm builds instead, because
    the binary that does the building IS the pin and was verified through a probe
    that cannot fetch.

    The mismatching PATH runner is asserted to have been PROBED and never asked to
    build: that is the whole safety claim, stated as a process list rather than as
    prose. With :func:`install._runner_or_refusal`'s call replaced by the plain
    ``_pin_mismatch`` judgement this test fails with the refusal in hand (measured —
    see the PR body).
    """
    home = tmp_path / "pnpm-home"
    monkeypatch.setenv("PNPM_HOME", str(home))
    path_fake = StandIn(tmp_path / "bin", version="10.30.3", exit=1)
    seeded = _seed(home, "11.22.0", version="11.22.0", exit=0, dist=True)
    web = _web(tmp_path, pin="pnpm@11.22.0")

    error = install._build_bundle(web, path_fake.runner)

    assert error is None, f"the seeded pin should have built this tree: {error}"
    assert path_fake.argv_seen() == [
        ["--version"]
    ], "the runner that cannot satisfy the pin is probed and never asked to build"
    # The candidate's verification probe, then the two steps — and no second probe:
    # the arm verified the pin itself, so the guard is not asked about the runner it
    # chose. Counted rather than described, because a boundary that spawns more than
    # it says is the default this module was written about.
    assert seeded.argv_seen() == [
        ["--version"],
        ["install", "--frozen-lockfile"],
        ["build"],
    ]


def test_a_matching_path_runner_is_not_replaced_by_a_seeded_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The seed is a fallback, not a preference for its own copy.

    PATH is what the operator chose, and nothing is fetched or even ASKED for when
    it satisfies the pin: the seeded candidate is not probed at all, because the
    routes are consulted only when the guard would refuse.
    """
    home = tmp_path / "pnpm-home"
    monkeypatch.setenv("PNPM_HOME", str(home))
    seeded = _seed(home, "11.22.0", version="11.22.0", exit=0, dist=True)
    path_fake = StandIn(tmp_path / "bin", version="11.22.0", exit=0, dist=True)
    web = _web(tmp_path, pin="pnpm@11.22.0")

    assert install._build_bundle(web, path_fake.runner) is None

    assert ["build"] in path_fake.argv_seen(), "PATH's pnpm is the one that builds"
    assert seeded.argv_seen() == [], "nothing to fix, so nothing is consulted"


def test_the_corepack_route_builds_a_tree_the_path_pnpm_cannot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RESTORED ARM: a host with corepack but no matching pnpm BUILDS (round 3, R3-1).

    #1394's guard refuses this machine, and the earlier revision of this branch
    refused it too — a regression against what was already on main, where a
    resolving corepack kept such a host building. Corepack RESOLVES the pin (that is
    what it is for), so it is one of the ways the pin can be satisfied.

    Two properties, both asserted rather than described: the route goes through the
    same bounded runner and the same ARMED environment as every other child (the
    stand-in records what it was handed), and the mismatching PATH runner is probed
    and never asked to build.
    """
    tools = tmp_path / "tools"
    tools.mkdir()
    _corepack_stand_in(tools)
    monkeypatch.setenv("PATH", str(tools))
    monkeypatch.setenv("PNPM_HOME", str(tmp_path / "pnpm-home"))
    path_fake = StandIn(tmp_path / "bin", version="10.30.3", exit=1)
    web = _web(tmp_path, pin="pnpm@11.22.0")

    error = install._build_bundle(web, path_fake.runner)

    assert error is None, f"corepack resolves the pin, so this host must build: {error}"
    calls = _corepack_calls(tools)
    assert [call["argv"] for call in calls] == [
        ["enable"],
        ["pnpm", "install", "--frozen-lockfile"],
        ["pnpm", "build"],
    ], f"enable, then the two steps, all through corepack: {[c['argv'] for c in calls]}"
    assert all(call["manage"] == "false" for call in calls), "every corepack child is disarmed"
    assert all(call["strict"] == "true" for call in calls), "and fails rather than warns"
    assert all(
        entry == ["--version"] for entry in path_fake.argv_seen()
    ), "the runner that cannot satisfy the pin is probed and never asked to build"


def test_a_corepack_shaped_runner_is_left_alone_even_with_a_seeded_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Corepack RESOLVES the pin, so its runner is never swapped for a seeded copy.

    Two reasons, and the second is the one that matters: corepack owns its own cache
    (a fresh download is its job, not this module's), and the seeded arm must not
    rewrite a runner it has no complaint about.
    """
    home = tmp_path / "pnpm-home"
    monkeypatch.setenv("PNPM_HOME", str(home))
    _seed(home, "11.22.0", version="11.22.0", exit=0)
    shims = tmp_path / "shims"
    shims.mkdir()
    shim = shims / "pnpm"
    shim.write_text('#!/bin/sh\n# corepack shim\nexec corepack pnpm "$@"\n', encoding="utf-8")
    shim.chmod(0o755)
    monkeypatch.setenv("PATH", str(shims))
    web = _web(tmp_path, pin="pnpm@11.22.0")

    assert install._corepack_shaped(["pnpm"]) is True
    assert install._runner_or_refusal(["pnpm"], web, env={}) == (["pnpm"], None)


def test_without_corepack_or_a_seeded_pin_the_mismatch_is_still_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The boundary this delta must not weaken: NEITHER route, nothing changes.

    PATH is emptied rather than left to the host, because the boundary is "no
    corepack resolves" and a developer machine (or a CI runner) with corepack
    installed would otherwise measure a different decision than the one this test is
    about. The refusal's copy, its routes and its probe count are
    ``test_bundle_build_bound.py``'s business — it owns that sentence — so what is
    asserted here is only the shape the routes could have broken.
    """
    tools = tmp_path / "tools"
    tools.mkdir()
    monkeypatch.setenv("PATH", str(tools))
    monkeypatch.setenv("PNPM_HOME", str(tmp_path / "empty-home"))
    fake = StandIn(tmp_path / "bin", version="10.30.3", exit=1)
    web = _web(tmp_path, pin="pnpm@11.22.0")

    assert install._build_bundle(web, fake.runner) is not None
    assert fake.argv_seen() == [["--version"]], "a refusal spawns no build child"
    runner_after, mismatch = install._runner_or_refusal(fake.runner, web, env={})

    assert runner_after == fake.runner, "no route to try: the runner goes on unchanged"
    assert mismatch is not None, "and the guard's refusal is what comes back"
