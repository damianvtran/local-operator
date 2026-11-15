"""Exercise the analyzer boundary, not a second interpretation of its globs."""

import json
import subprocess
import sys
from pathlib import Path


def test_historical_repro_boundary_keeps_shipping_source_checked(tmp_path: Path) -> None:
    """Moving old docs tools must not expand or weaken the shipping type gate.

    Give every surface the same deliberate type error. Running the real analyzer
    with the repository config proves both sides of the boundary: historical
    rigs stay outside discovery, while package, tests and ordinary scripts still
    report errors. The hidden-directory control also guards the default excludes
    that an explicit exclude list would otherwise replace.
    """
    tmp_path = tmp_path.resolve()
    root = Path(__file__).resolve().parents[2]
    (tmp_path / "pyproject.toml").write_text((root / "pyproject.toml").read_text())
    checked = {
        "local_operator/scope_control.py",
        "tests/scope_control.py",
        "scripts/scope_control.py",
    }
    excluded = {
        "scripts/repro/historical/scope_control.py",
        "docs/scope_control.py",
        ".venv/scope_control.py",
    }
    for relative in checked | excluded:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('value: int = "deliberate type error"\n')
    result = subprocess.run(
        [sys.executable, "-m", "pyright", "--outputjson", "."],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    report = json.loads(result.stdout)
    errors = {
        Path(diagnostic["file"]).relative_to(tmp_path).as_posix()
        for diagnostic in report["generalDiagnostics"]
        if diagnostic["severity"] == "error"
    }
    assert errors == checked, report
    assert report["summary"]["errorCount"] == len(checked), report
