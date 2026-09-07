"""Deterministic repro for the all-digit / numeric-looking job-id defect.

Run it from anywhere; it inserts the repo root on ``sys.path`` itself::

    .venv/bin/python docs/evidence/job-id-coercion/repro_job_id.py

Prints provenance (module __file__, sha256 of that exact file, git HEAD) so the
A/B arms cannot silently import the same tree -- this repo's editable venv has
produced false "no difference" results before.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys
import uuid
from dataclasses import dataclass

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from local_operator.tools import builtin  # noqa: E402
from local_operator.tools.builtin import (  # noqa: E402
    JobsParams,
    _coerce_job_targets,
    _coerce_single_job_id,
    _resolve_job_target,
)


@dataclass
class FakeJob:
    id: str
    label: str = "child"
    status: str = "running"


class FakeJobs:
    """Minimal stand-in for the jobs manager: only the real id resolves."""

    def __init__(self, ids: list[str]) -> None:
        self._jobs = [FakeJob(i) for i in ids]

    def get(self, target):
        return next((j for j in self._jobs if j.id == target), None)

    def list(self):
        return list(self._jobs)


def provenance() -> None:
    path = builtin.__file__
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        cwd=str(pathlib.Path(__file__).parent),
    ).stdout.strip()
    print(f"module.__file__ = {path}")
    print(f"sha256(builtin.py)[:16] = {digest}")
    print(f"git HEAD = {head}")
    # Provenance guard: assert the import came from the worktree this script
    # was launched from, not from another checkout or the global uv tool
    # install. This repo's editable venv has produced false "no difference"
    # A/B results for reviewers who skipped this check.
    assert path.endswith("local_operator/tools/builtin.py"), path
    assert (
        _REPO_ROOT in pathlib.Path(path).resolve().parents
    ), f"imported the WRONG tree: {path} (expected under {_REPO_ROOT})"
    print()


# The shapes a job id can arrive in. Each entry: (label, input, expected).
CASES: list[tuple[str, object, object]] = [
    # --- the defect: 12-hex ids that are entirely digits ---
    ("all-digit id, bracketed", "[920883861377]", "920883861377"),
    ("all-digit id, bare", "920883861377", "920883861377"),
    ("all-digit id, JSON-quoted", '["920883861377"]', "920883861377"),
    # --- same root cause, different numeric literal shapes ---
    ("sci-notation id (12e345678901)", "[12e345678901]", "12e345678901"),
    ("sci-notation id (177650473e52)", "[177650473e52]", "177650473e52"),
    ("leading-zero id", "[000123456789]", "000123456789"),
    ("all-digit, other length", "[00420]", "00420"),
    ("hex-looking float id", "[1e5]", "1e5"),
    # --- bare non-string scalar (model emitted a JSON number) ---
    ("bare int id", 920883861377, "920883861377"),
    ("int inside a real list", [920883861377], "920883861377"),
    # --- the list case must keep working ---
    ("list of two digit ids", "[920883861377, 468698086935]", ["920883861377", "468698086935"]),
    (
        "JSON list of two hex ids",
        '["a1b2c3d4e5f6", "0f1e2d3c4b5a"]',
        ["a1b2c3d4e5f6", "0f1e2d3c4b5a"],
    ),
    ("mixed digit + hex list", "[920883861377, a1b2c3d4e5f6]", ["920883861377", "a1b2c3d4e5f6"]),
    ("real list of str ids", ["920883861377", "a1b2c3d4e5f6"], ["920883861377", "a1b2c3d4e5f6"]),
    # --- ordinary ids must be untouched ---
    ("ordinary hex id", "a1b2c3d4e5f6", "a1b2c3d4e5f6"),
    ("ordinary id bracketed", "[a1b2c3d4e5f6]", "a1b2c3d4e5f6"),
    ("label target", "reviewer", "reviewer"),
    ("the literal 'all'", "all", "all"),
]


def main() -> int:
    provenance()
    failures = 0

    print("=== _coerce_job_targets ===")
    for label, value, expected in CASES:
        got = _coerce_job_targets(value)
        ok = got == expected
        failures += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:32} {value!r:38} -> {got!r}")
        if not ok:
            print(f"         expected {expected!r}")

    # End-to-end: the shape the operator actually sees is the error string from
    # _resolve_job_target, which is where "unknown job [920883861377]" surfaced.
    print("\n=== end-to-end resolve (the reported symptom) ===")
    real_id = "920883861377"
    jobs = FakeJobs([real_id])
    target = _coerce_single_job_id(f"[{real_id}]")
    if isinstance(target, str):
        resolved, err = _resolve_job_target(target, jobs)
    else:
        resolved, err = None, f"target is not a str: {target!r}"
    ok = resolved == real_id and err is None
    failures += not ok
    print(f"  [{'PASS' if ok else 'FAIL'}] resolve('[{real_id}]') -> {resolved!r} err={err!r}")

    # MAJOR-1 (review round 1): a bare unquoted scalar has already lost its
    # source text in the outer tool-argument decode. An int survives losslessly;
    # a float must NOT be formatted into a plausible-but-never-minted id.
    print("\n=== bare unquoted scalars (source text already gone) ===")
    for literal, want in [
        ("920883861377", "920883861377"),  # int: str(int) is lossless
        ("7019316393e2", None),  # float: unrecoverable -> must be refused
        ("13190e419943", None),  # overflows to inf -> id destroyed
    ]:
        decoded = json.loads(literal)
        try:
            got = JobsParams(op="peek", job_id=decoded).job_id
        except Exception:
            got = None
        ok = got == want
        failures += not ok
        shown = repr(got) if got is not None else "refused (field validation speaks)"
        print(
            f"  [{'PASS' if ok else 'FAIL'}] model wrote {literal:14} -> {decoded!r:18} -> {shown}"
        )

    # Fleet-scale rate over real uuid4 ids, through the real function.
    print("\n=== rate over 200k real uuid4().hex[:12] ids ===")
    n = 200_000
    bad = []
    for _ in range(n):
        hid = uuid.uuid4().hex[:12]
        if _coerce_job_targets(f"[{hid}]") != hid:
            bad.append(hid)
    print(f"  mangled {len(bad)}/{n} = {len(bad) / n:.4%}   e.g. {bad[:5]}")
    failures += len(bad) > 0

    print(f"\nRESULT: {'ALL PASS' if failures == 0 else f'{failures} FAILING GROUPS'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
