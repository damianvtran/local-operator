"""Qualification-ONLY oracle adapter/verifier pair (ORACLE-ISOLATION-DESIGN.md step 5).

WHY (oracle sweep a18da995, headless-terminal): a task's reference
``solve.sh`` may copy its own gold file into the workspace
(``cp /solution/headless_terminal.py /app/``). The post-agent gold scan then
correctly refuses grading -- for a MODEL that file would be leaked gold -- so
the oracle can never qualify the task. Deleting the oracle's outputs, or an
``oracle=true`` switch in the paid path, would weaken the very check that
protects model arms. Instead the oracle gets an explicitly separate, narrow
qualification policy:

1. ``QualificationOracle.setup`` runs the COMMON preflight
   (``adapters._preinference_checks``: same gold-isolation + egress fence as
   every model arm) BEFORE Harbor's stock oracle uploads ``/solution``. A dirty
   or unparsed preflight raises, so no receipt exists and the oracle never runs.
2. It then writes a host-side receipt in the trial directory (NOT a container
   mount, so nothing in ``main`` can rewrite it) binding trial id, image
   digest, provenance manifest digest, task payload digest, the exact
   ``/solution`` tree about to be uploaded, and the ordering
   ``preflight-clean -> oracle-upload-run -> oracle-returned``.
3. ``QualificationVerifier`` requires that receipt, complete and bound to the
   current trial/environment, then runs the SAME ``BenchVerifier`` tail with
   one extra probe argument: the authorized tree. The probe then permits
   matches whose digest is a manifest SOLUTION digest supplied by that tree
   and an exact ``/solution`` tree; verifier-only (tests) digests, extra
   ``/solution`` entries, ``/tests``, verifier logs and truncation still fail.

Solution/test digest overlap is recorded, because bytes supplied in the oracle
source cannot be claimed secret from that oracle.

CONSTRAINTS: never a model policy. ``driver.py`` (paid) selects
``bench_verifier:BenchVerifier`` and the frozen model adapters only, and no
paid-closure module imports this one (tested in test_public_provenance).
Only the zero-model qualification rig selects
``-a oracle_qualification:QualificationOracle --verifier-import-path
oracle_qualification:QualificationVerifier``. Oracle reward and isolation
acceptance stay separate fields; a task qualifies only with both.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path

from bench_verifier import BenchVerifier

RECEIPT = "bench-oracle-receipt.json"
RECORD = "bench-oracle-qualification.json"
ORDER = ["preflight-clean", "oracle-upload-run", "oracle-returned"]


def _sha(path: Path) -> str:
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def solution_tree(source_dir: Path) -> dict[str, str]:
    """``relpath -> sha256`` of the directory Harbor uploads as ``/solution``."""
    tree = {}
    for path in sorted(Path(source_dir).rglob("*")):
        if path.is_symlink():
            raise ValueError("solution tree contains a symlink")
        if path.is_file():
            tree[str(path.relative_to(source_dir))] = _sha(path)
    if not tree:
        raise ValueError("empty solution tree")
    return tree


def binding(session_id: str) -> dict[str, str]:
    """Identity the receipt must carry; from the Harbor process env, never --ae."""
    return {
        "trial": session_id,
        "image_digest": os.environ.get("BENCH_IMAGE_DIGEST", ""),
        "manifest_sha256": os.environ.get("BENCH_PUBLIC_PROVENANCE_SHA256", ""),
        "task_payload_sha256": os.environ.get("BENCH_TASK_PAYLOAD_SHA256", ""),
    }


def _oracle_base() -> type:
    return importlib.import_module("harbor.agents.oracle").OracleAgent


# Harbor's own staging of the uploaded solution selection, so the receipt
# hashes exactly the tree the stock oracle uploads (file filter applied).
_FILE_STAGING = importlib.import_module("harbor.utils.file_staging")


def _constructing_trial(logs_dir):
    """The Harbor ``Trial`` whose ``_init_agent`` is constructing this agent.

    WHY (S9e, d0b2b32c sweep): Harbor 0.25 ``Trial._init_agent`` hands
    ``task_dir``, ``trial_paths`` and ``agent_timeout_sec`` ONLY to an agent
    whose ``--agent`` name is ``oracle``; an import-path subclass (how this
    module is selected) gets ``logs_dir`` alone and Harbor's stock
    ``OracleAgent.__init__`` raises TypeError before setup. A name of
    ``oracle`` cannot carry an import path (the factory then builds the stock
    oracle). So the values are read from the one object that holds them: the
    constructing Trial on the call stack, bound by ``paths.agent_dir ==
    logs_dir`` (Harbor passes ``logs_dir=self.paths.agent_dir``). These are
    the exact objects Harbor gives its own oracle; nothing is reconstructed,
    and no rig shim or extra ``--ak`` is needed. No matching Trial refuses.
    """
    trial_cls = importlib.import_module("harbor.trial.trial").Trial
    frame = inspect.currentframe()
    try:
        while frame is not None:
            owner = frame.f_locals.get("self")
            if isinstance(owner, trial_cls) and Path(owner.paths.agent_dir) == Path(logs_dir):
                return owner
            frame = frame.f_back
    finally:
        del frame
    raise TypeError("QualificationOracle needs task_dir/trial_paths: construct it from a "
                    "Harbor Trial, or pass Harbor's oracle kwargs explicitly")


class QualificationOracle(_oracle_base()):  # type: ignore[misc]
    def __init__(self, logs_dir, task_dir=None, trial_paths=None, *args,
                 agent_timeout_sec=None, **kwargs):
        # Explicit oracle kwargs (Harbor's ``oracle`` branch, or a direct
        # caller) are forwarded unchanged; only when ALL are absent are
        # Harbor's own values taken from the constructing Trial.
        if task_dir is None and trial_paths is None:
            trial = _constructing_trial(logs_dir)
            task_dir, trial_paths = trial.task.task_dir, trial.paths
            if agent_timeout_sec is None:
                agent_timeout_sec = trial.agent_timeout_sec
        elif task_dir is None or trial_paths is None:
            raise TypeError("QualificationOracle needs both task_dir and trial_paths")
        if agent_timeout_sec is not None:
            kwargs["agent_timeout_sec"] = agent_timeout_sec
        super().__init__(logs_dir, task_dir, trial_paths, *args, **kwargs)

    @staticmethod
    def name() -> str:
        return "bench-qualification-oracle"

    def _receipt_path(self) -> Path:
        return Path(self._trial_paths.trial_dir) / RECEIPT

    def _write(self, receipt: dict) -> None:
        self._receipt_path().write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")

    async def setup(self, environment) -> None:
        import adapters

        await super().setup(environment)
        if self._task.has_steps:
            raise RuntimeError("multi-step oracle qualification is not supported")
        # The receipt must never predate a clean preflight.
        self._receipt_path().unlink(missing_ok=True)
        await adapters._preinference_checks(self, environment)
        solution_dir, _ = self._resolve_solution_paths()
        selection = self._task.file_filter.select(solution_dir)
        with _FILE_STAGING.selected_directory(selection) as source:
            tree = solution_tree(Path(source))
        tests = Path(self._task.paths.tests_dir)
        test_digests = ({_sha(p) for p in tests.rglob("*") if p.is_file()}
                        if tests.is_dir() else set())
        self._write({**binding(environment.session_id), "solution_tree": tree,
                     "solution_test_overlap": sorted(set(tree.values()) & test_digests),
                     "order": ORDER[:1]})

    async def run(self, instruction, environment, context) -> None:
        path = self._receipt_path()
        receipt = json.loads(path.read_text()) if path.is_file() else {}
        if receipt.get("order") != ORDER[:1] or any(
                receipt.get(k) != v for k, v in binding(environment.session_id).items()):
            raise RuntimeError("oracle run without a clean, bound preflight receipt")
        receipt["order"].append(ORDER[1])
        self._write(receipt)
        await super().run(instruction, environment, context)
        receipt["order"].append(ORDER[2])
        self._write(receipt)


class QualificationVerifier(BenchVerifier):
    """BenchVerifier's shared tail plus the receipt-bound authorized source."""

    _authorized: dict[str, str] | None = None

    def _isolation_extra_args(self) -> list[str]:
        if not self._authorized:
            raise RuntimeError("qualification verifier has no authorized source")
        raw = json.dumps(self._authorized, sort_keys=True, separators=(",", ":"))
        return ["--authorized-solution", raw, hashlib.sha256(raw.encode()).hexdigest()]

    def check_receipt(self) -> dict:
        path = Path(self.trial_paths.trial_dir) / RECEIPT
        try:
            receipt = json.loads(path.read_text())
        except (OSError, ValueError):
            receipt = {}
        expected = binding(self.environment.session_id)
        problems = [k for k, v in expected.items() if not v or receipt.get(k) != v]
        if receipt.get("order") != ORDER:
            problems.append("order")
        tree = receipt.get("solution_tree")
        if not isinstance(tree, dict) or not tree:
            problems.append("solution_tree")
        record = {"receipt_sha256": hashlib.sha256(path.read_bytes()).hexdigest()
                  if path.is_file() else None, "problems": problems,
                  "solution_test_overlap": receipt.get("solution_test_overlap"),
                  "policy": "oracle-qualification-only"}
        out = self.trial_paths.verifier_dir / RECORD
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=1) + "\n")
        if problems:
            raise RuntimeError(f"oracle qualification receipt refused: {problems}")
        return tree

    async def verify(self):  # noqa: D102 -- see module docstring
        self._authorized = self.check_receipt()
        return await super().verify()
