#!/usr/bin/env python3
"""Prove the six serial store-walk / status-verb controls were absent, then present.

WHY THIS EXISTS. Two unit cells (and their four controls) hold real threads, a real
child process and real sockets, and they flaked under ``pytest -n auto`` on shared CI
runners. They now run ONCE, serially (``-n0``), in a separate step of shard 0 and are
``--deselect``-ed from every shard's xdist run. That arrangement is only worth having
if CI can SHOW it happened: a deselect that silently also dropped the cells, or a
serial step that was skipped, would leave a green run that exercised nothing. This
checker reads the JUnit XML each step wrote and turns "it ran" into a pass/fail fact:

* ``absent``  — the shard's ordinary XML names none of the six (the deselect worked
  and nothing runs twice);
* ``present`` — the serial XML names exactly the six, each once, with no failure,
  error or skip, and each carrying the receipt properties its test records (PIDs, the
  source-identity hashes, the sequence receipts, WHICH oracle guard spoke).

Exit status: ``0`` proven; ``1`` the XML says the contract is violated (FAIL); ``2``
the proof could not be made — the file is missing or unreadable, or holds no test
cases (BLOCKED). A missing proof is never a pass: BLOCKED fails the step on purpose.

STDLIB ONLY and no repository imports, so it runs before (and independent of) the
code under test. It is exercised by synthetic XML in ``tests/unit/test_ci_hygiene.py``
— never by running the real cases.
"""

from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path

EXIT_PROVEN = 0
EXIT_FAIL = 1
EXIT_BLOCKED = 2

#: Receipts every store-walk case records (``tests/unit/test_session_factory.py``).
_WALK_PROPERTIES = (
    "walk_mode",
    "walk_parent_pid",
    "walk_child_pid",
    "walk_child_pgid",
    "walk_reap",
    "walk_child_returncode",
    "walk_report",
    "walk_source_identity",
    "walk_oracle",
)
#: Receipts the status-verb cases record (``test_membership_convergence.py``).
_STATUS_PROPERTIES = ("status_receipts", "status_relay_source", "status_oracle")

#: Required receipts per test FUNCTION name. Keyed by name (the id the workflow
#: carries is path::name) so the table and the workflow's single list of ids cannot
#: drift apart unnoticed: the checker refuses an id it has no row for.
REQUIRED_PROPERTIES: dict[str, tuple[str, ...]] = {
    "test_a_leaving_runtime_stops_the_store_walk_at_a_directory": _WALK_PROPERTIES,
    "test_cell_a_control_the_old_observer_setup_fails_its_precondition": _WALK_PROPERTIES,
    "test_cell_a_control_a_consumer_that_ignores_a_set_stop_runs_the_walk_to_the_end": (
        _WALK_PROPERTIES
    ),
    "test_the_status_verb_asks_over_the_control_socket_for_a_fresh_pass": _STATUS_PROPERTIES,
    "test_status_verb_control_a_dropped_refresh_field_is_rejected_for_not_asking": (
        _STATUS_PROPERTIES
    ),
    "test_status_verb_control_the_old_empty_assertion_fails_after_a_legitimate_pass": (
        "status_old_empty_table",
        "status_old_empty_receipts",
        "status_oracle",
    ),
}

#: Cases whose oracle must PASS; every other case is a control whose recorded oracle
#: line must be a failure sentence (the guard it is about), never ``passed``.
_POSITIVE = frozenset(
    {
        "test_a_leaving_runtime_stops_the_store_walk_at_a_directory",
        "test_the_status_verb_asks_over_the_control_socket_for_a_fresh_pass",
    }
)

_SUMMARY_VALUE_LIMIT = 160


class Blocked(Exception):
    """The proof could not be made at all (missing/unreadable/empty XML)."""


def split_id(node_id: str) -> tuple[str, str]:
    """``tests/unit/x.py::test_y`` -> (``tests.unit.x``, ``test_y``), JUnit's pair."""
    path, sep, name = node_id.partition("::")
    if not sep or not name or "::" in name or not path.endswith(".py"):
        raise Blocked(f"not a top-level test id: {node_id!r}")
    return path[: -len(".py")].replace("/", "."), name


def parse_ids(raw: str) -> list[tuple[str, str]]:
    ids = raw.split()
    if not ids:
        raise Blocked("no serial control ids were given")
    pairs = [split_id(i) for i in ids]
    if len(set(pairs)) != len(pairs):
        raise Blocked("the serial control id list holds a duplicate")
    unknown = [name for _, name in pairs if name not in REQUIRED_PROPERTIES]
    if unknown:
        raise Blocked(f"no receipt contract for {unknown}: add it to REQUIRED_PROPERTIES")
    return pairs


def load_cases(path: Path) -> list[ET.Element]:
    try:
        root = ET.parse(path).getroot()
    except FileNotFoundError:
        raise Blocked(f"{path} does not exist") from None
    except (ET.ParseError, OSError) as error:
        raise Blocked(f"{path} is unreadable: {error}") from None
    cases = list(root.iter("testcase"))
    if not cases:
        raise Blocked(f"{path} holds no test cases")
    return cases


def _matches(case: ET.Element, pair: tuple[str, str]) -> bool:
    """Exact function match, or one of its parametrisations (``name[...]``)."""
    classname, name = pair
    got = case.get("name") or ""
    return case.get("classname") == classname and (got == name or got.startswith(name + "["))


def check_absent(cases: list[ET.Element], pairs: list[tuple[str, str]]) -> list[str]:
    return [
        f"{c.get('classname')}::{c.get('name')} ran in the ordinary shard run"
        for c in cases
        if any(_matches(c, p) for p in pairs)
    ]


def check_present(cases: list[ET.Element], pairs: list[tuple[str, str]]) -> tuple[list[str], str]:
    problems: list[str] = []
    digest: list[str] = []
    seen: dict[tuple[str, str], int] = {p: 0 for p in pairs}
    for case in cases:
        owner = next((p for p in pairs if _matches(case, p)), None)
        if owner is None:
            problems.append(f"unexpected case {case.get('classname')}::{case.get('name')}")
            continue
        seen[owner] += 1
    for pair, count in seen.items():
        if count != 1:
            problems.append(f"{pair[0]}::{pair[1]} ran {count} time(s), expected exactly 1")
    for case in cases:
        owner = next((p for p in pairs if _matches(case, p)), None)
        if owner is None:
            continue
        label = f"{owner[1]}"
        for verdict in ("failure", "error", "skipped"):
            if case.find(verdict) is not None:
                problems.append(f"{label}: recorded <{verdict}>")
        props = {
            p.get("name"): p.get("value")
            for p in case.findall("./properties/property")
            if p.get("name")
        }
        missing = [n for n in REQUIRED_PROPERTIES[owner[1]] if not props.get(n)]
        if missing:
            problems.append(f"{label}: missing receipt properties {missing}")
        oracle = props.get("walk_oracle") or props.get("status_oracle") or ""
        if owner[1] in _POSITIVE and oracle != "passed":
            problems.append(f"{label}: oracle did not pass ({oracle[:_SUMMARY_VALUE_LIMIT]!r})")
        if owner[1] not in _POSITIVE and (not oracle or oracle.startswith("passed")):
            problems.append(f"{label}: control recorded no failing guard ({oracle!r})")
        digest.append(f"| {label} | {oracle[:_SUMMARY_VALUE_LIMIT]} |")
    return problems, "\n".join(digest)


def main(argv: Sequence[str] | None = None) -> int:
    # `python -OO` strips docstrings, so the module's `__doc__` is None there; the
    # description must not be the thing that turns `--help` into a traceback.
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("mode", choices=("absent", "present"))
    parser.add_argument("--junit", required=True, type=Path)
    parser.add_argument("--ids", required=True, help="whitespace-separated pytest node ids")
    parser.add_argument("--summary", default="", help="file to append a digest to")
    args = parser.parse_args(argv)
    try:
        pairs = parse_ids(args.ids)
        cases = load_cases(args.junit)
    except Blocked as reason:
        print(f"BLOCKED ({args.mode}): {reason}")
        return EXIT_BLOCKED

    if args.mode == "absent":
        problems = check_absent(cases, pairs)
        digest = ""
        headline = f"{len(pairs)} serial control(s) absent from {args.junit.name}"
    else:
        problems, digest = check_present(cases, pairs)
        headline = f"{len(pairs)} serial control(s) present once in {args.junit.name}"

    verdict = "FAIL" if problems else "PASS"
    print(f"{verdict} ({args.mode}): {headline}")
    for problem in problems:
        print(f"  - {problem}")
    if digest:
        print(digest)
    target = args.summary or os.environ.get("GITHUB_STEP_SUMMARY", "")
    if target and args.mode == "present":
        with open(target, "a", encoding="utf-8") as out:
            out.write(f"### Serial controls — {verdict}\n\n| case | oracle |\n|---|---|\n")
            out.write(f"{digest}\n")
            out.writelines(f"- {p}\n" for p in problems)
    return EXIT_FAIL if problems else EXIT_PROVEN


if __name__ == "__main__":
    sys.exit(main())
