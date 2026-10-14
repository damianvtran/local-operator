"""Portable-rig safety: inherited CMUX_*/LOP_* never survive the prologue.

The evidence rigs boot the REAL ``OperatorApp``, the surface the team rule
covers ("any test that boots the TUI or a fork MUST unset every inherited
CMUX_* variable and use synthetic IDs — an inherited CMUX_WORKSPACE_ID let
headless tests rename the operator's real cmux workspaces"), and they are
documented to run from a normal shell. This regression executes each rig's
STDLIB SETUP PROLOGUE — everything before its first rich/local_operator/
scripts/tests import — in a subprocess carrying fake sentinels, then asserts
none survive to the import boundary. It never boots the TUI, never touches
cmux, and never mutates operator state: the prologue only pops environment
names and points a scratch HOME/config at a tmp dir (review round 3, R12/Q7).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RIGS = (
    ROOT / "docs/evidence/team-bridge-h1/capture-rig.py",
    ROOT / "docs/evidence/team-bridge-h1/proactive-rig.py",
)

#: The first harness import — where the prologue must already have stripped.
_HARNESS_IMPORT = re.compile(r"^(?:from|import)\s+(?:local_operator|rich|scripts|tests)\b", re.M)

#: Fake values under BOTH namespaces, including a FUTURE unknown suffix: the
#: strip is prefix-based, so a key nobody has invented yet must not survive.
_SENTINELS = {
    "CMUX_WORKSPACE_ID": "fake-workspace-sentinel",
    "CMUX_FUTURE_UNKNOWN_SENTINEL": "fake-future-sentinel",
    "LOP_RUNTIME_ADOPT_SESSION": "fake-adopt-sentinel",
    "LOP_FUTURE_SENTINEL": "fake-future-sentinel",
    "LOCAL_OPERATOR_KEEP_ME": "kept-sentinel",
}


def _prologue(text: str, name: str) -> str:
    match = _HARNESS_IMPORT.search(text)
    assert match, f"{name}: no harness import found to bound the prologue"
    return text[: match.start()]


def test_each_rig_strips_host_identifiers_before_its_first_harness_import(
    tmp_path: Path,
) -> None:
    for rig in RIGS:
        text = rig.read_text()
        harness = _HARNESS_IMPORT.search(text)
        assert harness is not None, f"{rig.name}: no harness import found"
        first_import = harness.start()
        strip_at = text.index('startswith(("CMUX_", "LOP_"))')
        assert (
            strip_at < first_import
        ), f"{rig.name}: the namespace strip must precede the first harness import"

        scratch = tmp_path / rig.stem
        scratch.mkdir()
        prologue = tmp_path / f"{rig.stem}-prologue.py"
        prologue.write_text(_prologue(text, rig.name))
        child = "\n".join(
            [
                "import json, os, sys",
                'namespace = {"__file__": %r, "__name__": "__main__"}' % str(rig),
                "with open(%r) as fh:" % str(prologue),
                '    exec(compile(fh.read(), %r, "exec"), namespace)' % str(rig),
                "print(json.dumps({",
                '    "cmux": sorted(k for k in os.environ if k.startswith("CMUX_")),',
                '    "lop": sorted(k for k in os.environ if k.startswith("LOP_")),',
                '    "kept": os.environ.get("LOCAL_OPERATOR_KEEP_ME"),',
                '    "harness": sorted(',
                "        m for m in sys.modules",
                '        if m.split(".")[0] in ("local_operator", "rich", "scripts", "tests")',
                "    ),",
                "}))",
            ]
        )
        env = {k: v for k, v in os.environ.items() if not k.startswith(("CMUX_", "LOP_"))}
        env.update(_SENTINELS)
        env["LOCAL_OPERATOR_SCRATCHPAD"] = str(scratch)
        result = subprocess.run(
            [sys.executable, "-c", child],
            env=env,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"{rig.name}: {result.stderr[-2000:]}"
        report = json.loads(result.stdout)
        assert report["cmux"] == [], f"{rig.name}: CMUX_* survived the prologue: {report['cmux']}"
        assert report["lop"] == [], f"{rig.name}: LOP_* survived the prologue: {report['lop']}"
        # The rig's OWN namespace is deliberately retained (scratch/config).
        assert report["kept"] == "kept-sentinel", f"{rig.name}: stripped its own namespace"
        # No harness module may have been imported by the prologue itself: the
        # markers must be gone BEFORE the harness can read them.
        assert report["harness"] == [], f"{rig.name}: harness imported inside the prologue"
