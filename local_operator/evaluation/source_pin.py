"""The caller's core-source pin: module identity computed from the RUNNING tree.

WHY THIS EXISTS (blueprint §4, native session caller). The native arm's
correctness lives in the CORE tree -- SDK, session/session arm, loop,
configure, subagent, action tool -- not only in the adapter wheel. The old
arm's selector pinned the adapter (package/release digests) and the workspace
but NOTHING pinned the core: a selector built against a tree that predates
the merged caller composition would launch, allocate, and record an episode
whose "the API exists" claim the caller never actually exercised. This module
computes the pin a selector must carry and verifies it against the ACTUAL
interpreter BEFORE any allocation; ``run_episode``'s session engagement calls
it at preflight, and an old selector (no ``core_source``) is refused with a
named sentence rather than silently mis-attributed.

WHAT THE PIN IS, AND WHAT IT IS NOT. Content hashes (sha256 of each module
file, from the running interpreter's own import resolution) plus the checkout
commit WHEN a git worktree contains the package; a wheel/archive install
cannot be asked for a commit, and its file hashes remain the artifact
identity. A version string or a generation NAME is not a commit and is never
substituted for one (the same rule the release runbook states). The module
SET is exact: a pin that omits a pinned module is refused rather than read as
a partial pass, because coverage holes are how "verified" drifts from true.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from local_operator.evaluation.adapters.api import CoreSourcePin

#: The module set the pin covers, spelled ONCE: the key is the pin's stable
#: name (it rides the selector JSON and the record), the value is the dotted
#: module path resolved through the RUNNING interpreter. Keep this list in
#: lockstep with ``CoreSourcePin.module_hashes`` consumers -- verification
#: requires the exact key set, so adding a module here is a deliberate,
#: visible pin widening rather than a silent scope change.
CORE_SOURCE_MODULES: tuple[tuple[str, str], ...] = (
    ("sdk", "local_operator.sdk"),
    ("session", "local_operator.session.session"),
    ("session_arm", "local_operator.evaluation.session_arm"),
    ("loop", "local_operator.harness.loop"),
    ("configure", "local_operator.model.configure"),
    ("subagent", "local_operator.harness.subagent"),
    ("action_tool", "local_operator.evaluation.runner.action_tool"),
)

#: The canonical key set, derived rather than re-spelled, so verification and
#: computation cannot drift apart.
CORE_SOURCE_KEYS: tuple[str, ...] = tuple(key for key, _ in CORE_SOURCE_MODULES)


#: The declared commit's allowed syntax: hex, 7..40 characters -- a full SHA
#: or a reasonably abbreviated prefix of one (R6, review round 1).
_COMMIT_RE = re.compile(r"[0-9a-f]{7,40}")


class SourcePinError(RuntimeError):
    """The running core does not match the selector's declared source pin."""


def _module_origin(module_name: str) -> Path:
    """The file the RUNNING interpreter would execute for ``module_name``.

    ``find_spec`` resolves through the same import machinery a real import
    uses, so the answer cannot differ from what the arm will load -- a path
    computed any other way (checkout-relative, ``__file__`` of a different
    module) is exactly the divergence this pin exists to catch.
    """

    spec = importlib.util.find_spec(module_name)
    if spec is None or not spec.origin or spec.origin == "namespace":
        raise SourcePinError(
            f"the pin cannot locate module {module_name!r} in the running interpreter"
        )
    return Path(spec.origin)


def compute_core_source_hashes() -> dict[str, str]:
    """sha256 per pinned module, as executed by THIS interpreter."""

    return {
        key: hashlib.sha256(_module_origin(module).read_bytes()).hexdigest()
        for key, module in CORE_SOURCE_MODULES
    }


def resolve_core_commit() -> str | None:
    """The git commit of the checkout containing the running package, if any.

    Best effort and explicitly two-valued: a worktree/checkout answers
    ``git rev-parse HEAD``; an installed wheel answers ``None`` and the pin
    verification records ``commit_source="unavailable"`` -- never a version
    or generation name dressed up as a commit.
    """

    import local_operator

    root = Path(local_operator.__file__).resolve().parent.parent
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit or None


def verify_core_source(pin: "CoreSourcePin") -> dict[str, Any]:
    """Verify the running interpreter against ``pin``; return the provenance report.

    Refusals name EVERY mismatch at once (a pin is rebuilt once, not one
    module per cycle): a missing pin module, an unexpected extra module, a
    differing hash, or a commit mismatch when git can answer. The commit is
    asserted when discoverable; when it is not, the report says
    ``commit_source="unavailable"`` -- honest, and never a pass pretending the
    hashes were checked against a commit they were not.
    """

    actual_hashes = compute_core_source_hashes()
    declared = dict(pin.module_hashes) if pin.module_hashes else {}
    declared_keys, canonical_keys = set(declared), set(CORE_SOURCE_KEYS)
    missing_modules = sorted(canonical_keys - declared_keys)
    extra_modules = sorted(declared_keys - canonical_keys)
    mismatched = sorted(
        key for key in canonical_keys & declared_keys if declared[key] != actual_hashes[key]
    )
    if missing_modules or extra_modules or mismatched:
        raise SourcePinError(
            "the running core does not match the selector's core_source pin "
            f"(missing={missing_modules}, extra={extra_modules}, mismatched={mismatched}); "
            "refusing before allocation -- rebuild the selector against the tree "
            "that will run the arm"
        )
    actual_commit = resolve_core_commit()
    # COMMIT SYNTAX FIRST (R6): hex, 7..40 characters. The earlier
    # two-direction prefix check accepted a malformed extension ('<sha>
    # -corrupt') because the ACTUAL commit is a prefix of it -- an inverted
    # acceptance. The model validates this on construction; the check is
    # repeated here so a bypassed construction cannot slip past, and the
    # prefix direction below is ONE-WAY: the declared pin must be a prefix of
    # the actual commit, never the reverse.
    if not _COMMIT_RE.fullmatch(pin.commit):
        raise SourcePinError(
            "the selector's core_source commit is not a sha (hex, 7..40 characters): "
            f"{pin.commit!r}; refusing before allocation"
        )
    if actual_commit and not actual_commit.startswith(pin.commit):
        raise SourcePinError(
            "the running checkout's commit does not match the selector's core_source "
            f"commit (selector={pin.commit}, actual={actual_commit[:12]}); refusing "
            "before allocation"
        )
    return {
        "module_hashes": actual_hashes,
        "commit": actual_commit,
        "commit_source": "git" if actual_commit is not None else "unavailable",
    }


def main(argv: list[str] | None = None) -> int:
    """Print the pin for the RUNNING tree as JSON -- how a selector is built.

    Operator convenience only: the operator runs this from the frozen head
    and copies the two fields into the selector JSON. It refuses (exit 2) when
    the tree cannot be identified, so a half-empty pin never reaches a
    selector.
    """

    del argv
    try:
        report: dict[str, Any] = {
            "commit": resolve_core_commit(),
            "module_hashes": compute_core_source_hashes(),
        }
    except SourcePinError as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - operator entry point
    raise SystemExit(main())
