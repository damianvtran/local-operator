"""Fail closed when another producer starts inventing a directory for omission.

The fourth handoff regression was not a resolver bug: eager HOME defaults erased
intent before that resolver ran. Empty string is the existing spawn omission
sentinel. New fallback sites require classification here instead of silently
creating a fifth producer with the same bug. This ratchet does not rewrite real
explicit directory choices or ban unrelated home-relative metadata discovery.
"""

from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path

import pytest

# Function identities, not line numbers: source formatting cannot waive the
# guard. Reasons make every retained default reviewable in the producer census.
SAFE_DEFAULTS = {
    ("hook_forwarding.py", "load_hook_commands"): "hook config discovery, not session startup",
    ("exec_session.py", "session_cwd"): "publishes the already-constructed Session's cwd",
    ("variables.py", "__init__"): "variable expansion context, not a spawn producer",
    ("cli.py", "viewer_factory"): "intentional CLI shell directory, including explicit HOME",
    ("supplements/candidates.py", "prefilter"): "home-relative candidate discovery",
    ("supplements/denylist.py", "is_sensitive"): "sensitive-path classification only",
    ("tools/builtin.py", "_sessions_move_destination_label"): "approval path rendering only",
    ("library/service.py", "list_assets"): "library index metadata roots",
    ("library/service.py", "get_asset"): "library index metadata roots",
    ("library/index.py", "build_catalog"): "library index metadata roots",
    ("library/indexer.py", "__init__"): "library index metadata roots",
    ("mobile/daemon.py", "api_resume_session"): "after canonical handoff resolution",
    ("code_requests/detect.py", "load_mcp_servers"): "MCP config discovery only",
    ("skills/api.py", "ecosystem_skill_roots"): "skill discovery roots only",
    ("session/frontend_state.py", "refresh_from_session"): "already-running state publication",
    ("session/runtime/launch.py", "_spawn_runtime"): "after canonical handoff resolution",
    ("session/runtime/process.py", "amain"): "final runtime cwd consumer, not a producer",
}
DEFAULT = re.compile(
    r"\bor\s+(?:str\(Path\.home\(\)\)|Path\.home\(\)|"
    r"(?:os\.path\.)?expanduser\([\"']~[\"']\)|os\.getcwd\(\))"
)


def _default_sites(package: Path) -> set[tuple[str, str]]:
    hits: set[tuple[str, str]] = set()
    for path in package.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        matches = list(DEFAULT.finditer(source))
        if not matches:
            continue
        functions = [
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        for match in matches:
            line = source.count("\n", 0, match.start()) + 1
            parents = [
                node
                for node in functions
                if node.lineno <= line <= (node.end_lineno or node.lineno)
            ]
            owner = (
                min(parents, key=lambda node: (node.end_lineno or node.lineno) - node.lineno).name
                if parents
                else "<module>"
            )
            hits.add((path.relative_to(package).as_posix(), owner))
    return hits


def test_eager_directory_defaults_are_all_explicitly_classified() -> None:
    package = Path(__file__).resolve().parents[4] / "local_operator"
    hits = _default_sites(package)
    assert hits == SAFE_DEFAULTS.keys(), (
        "An unclassified cwd/home fallback can erase omission before spawn "
        "and cancel a recorded move. "
        f"New: {hits - SAFE_DEFAULTS.keys()}; removed: {SAFE_DEFAULTS.keys() - hits}"
    )


@pytest.mark.parametrize("engine", ["aida", "librarian"])
def test_orphan_wake_metadata_keeps_an_unspecified_directory(tmp_path: Path, engine: str) -> None:
    from local_operator.wakes import store

    module = importlib.import_module(f"local_operator.{engine}.proactive")
    sid = "cwdorphan001"
    store.write_entry(
        tmp_path,
        sid,
        cwd="",
        schedules=[
            {"id": module.CADENCE_ID, "kind": "scheduled", "message": "engine", "next_due_at": 1},
            {"id": "user-row", "kind": "scheduled", "message": "user", "next_due_at": 1},
        ],
    )
    assert module.reap_orphan_rows(tmp_path) == (sid,)
    kept = store.read_entry(tmp_path, sid)
    assert kept is not None and kept["cwd"] == ""
    assert [row["id"] for row in kept["schedules"]] == ["user-row"]


def test_wake_and_monitor_metadata_do_not_invent_a_storage_directory(tmp_path: Path) -> None:
    from local_operator.wakes.arm import _resolved_cwd

    assert _resolved_cwd(tmp_path / "session", {}) == ""
    # A durable directory actually recorded by a caller is still a choice;
    # never reinterpret a deliberate HOME as omission.
    assert _resolved_cwd(tmp_path / "session", {"cwd": str(Path.home())}) == str(Path.home())


def test_the_census_rejects_a_new_producer_not_just_the_known_relay(tmp_path: Path) -> None:
    (tmp_path / "new_producer.py").write_text(
        "def wake(cwd):\n    return cwd or str(Path.home())\n", encoding="utf-8"
    )
    hits = _default_sites(tmp_path)
    assert hits == {("new_producer.py", "wake")}
    assert hits - SAFE_DEFAULTS.keys(), "a fifth eager default must not disappear from the census"
