"""Guards for the release workflow's publish gates.

``publish.yml`` is the only workflow that can push to PyPI or attach an asset to a
GitHub Release, and until this module existed **no test read it at all** — the three
workflow parsers in ``tests/`` all load ``ci.yml`` specifically, and the only two
occurrences of the string ``publish.yml`` under ``tests/`` were prose in files that do
not read it (``tests/unit/mobile/test_web_bundle_guard.py`` and
``tests/unit/operator/test_operator_authority.py``). So both failure directions of the
manual trigger added in #1588 were silent:

- drop ``github.event_name == 'release'`` from a publisher's ``if:`` and a
  ``workflow_dispatch`` rehearsal — which exists to be run *before* a release, and
  which satisfies every other condition on that job by design — publishes for real;
- mistype the event (``workflow_dispatch``, or a stray quote) and a real release
  publishes nothing, or attaches no asset, with no signal until PyPI shows the gap.

Both are the v0.62.39 class of defect: a release path whose first honest run is the
release itself. ``ci.yml`` already gets this treatment in
``tests/unit/test_ci_hygiene.py``; this is the same shape for the one workflow that
writes outside the repository.

The invariant: **any job that can publish to PyPI or attach a Release asset must
require ``github.event_name == 'release'``** — not merely the job it depends on,
because a dispatched run satisfies every ``needs``-based condition by construction.

WHERE THE GUARD STOPS, said out loud because that sentence is broader than the check:
a job is publish-capable here only when a step's ``uses:`` names one of
:data:`PUBLISHING_ACTIONS`. A publisher expressed another way — ``run: twine upload``,
``gh release upload``, ``uv publish`` — is outside it, and supporting one means adding
its ref to that tuple. Every publisher this workflow has today goes through one of the
two, and ``test_the_workflow_has_publish_capable_jobs_to_guard`` fails if that stops
being true, so the boundary cannot go silently stale — but it is an enumeration, not
a proof over every way a job could publish.

Each assertion below is mutation-tested against the defect it claims to catch, so a
checker that silently stopped discriminating cannot pass this module.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

# Release/dispatch must inspect the same event target the build consumes, not
# mutable main or a classifier-selected PR subset.
RELEASE_TARGET_REF = "${{ github.event.release.tag_name || github.sha }}"
IMMUTABLE_SOURCE_REF = "${{ github.sha }}"
INVARIANT_JOB = "release-invariants"


def _release_invariant_problems(doc: dict[Any, Any]) -> list[str]:
    jobs = doc["jobs"]
    gate = jobs.get(INVARIANT_JOB, {})
    problems = []
    if not gate or "if" in gate or gate.get("needs") or gate.get("continue-on-error"):
        problems.append("invariant job can be skipped")
    steps = gate.get("steps", [])
    if any("if" in step or step.get("continue-on-error") for step in steps):
        problems.append("invariant step can be skipped")
    checkout = [step for step in steps if str(step.get("uses", "")).startswith("actions/checkout@")]
    if not checkout or checkout[0].get("with", {}).get("ref") != RELEASE_TARGET_REF:
        problems.append("invariant checkout is not the event target")
    node = [step for step in steps if str(step.get("uses", "")).startswith("actions/setup-node@")]
    if not node or str(node[0].get("with", {}).get("node-version")) != "24":
        problems.append("invariant Node is not 24")
    python = [
        step for step in steps if str(step.get("uses", "")).startswith("actions/setup-python@")
    ]
    if not python or str(python[0].get("with", {}).get("python-version")) != "3.12":
        problems.append("invariant Python is not 3.12")
    run = "\n".join(step.get("run", "") for step in steps)
    commands = ["python scripts/i18n/generate.py --check", "python scripts/i18n/check.py"]
    if not all(command in run.splitlines() for command in commands):
        problems.append("invariant commands missing")
    elif run.index(commands[0]) > run.index(commands[1]):
        problems.append("invariant commands reordered")
    if 'test "$(git rev-parse HEAD)" = "$GITHUB_SHA"' not in run:
        problems.append("event SHA is not verified")
    for name in ("release-build", "extension-release"):
        needs = jobs[name].get("needs", [])
        if isinstance(needs, str):
            needs = [needs]
        if INVARIANT_JOB not in needs:
            problems.append(f"{name} bypasses invariant dependency")

    def ancestors(name: str, visited: set[str]) -> set[str]:
        if name in visited:
            return set()
        visited.add(name)
        needs = jobs[name].get("needs", [])
        if isinstance(needs, str):
            needs = [needs]
        return set(needs).union(*(ancestors(parent, visited) for parent in needs))

    for name in _publishing_jobs(doc):
        if INVARIANT_JOB not in ancestors(name, set()):
            problems.append(f"{name} publisher bypasses invariants")
    for name in ("release-build", "extension-release", "keyagent-macos"):
        condition = str(jobs[name].get("if", ""))
        if "always(" in condition or "cancelled(" in condition:
            problems.append(f"{name} overrides dependency success")
        for step in jobs[name]["steps"]:
            if str(step.get("uses", "")).startswith("actions/checkout@"):
                if step.get("with", {}).get("ref") != IMMUTABLE_SOURCE_REF:
                    problems.append(f"{name} checkout is not immutable event SHA")
        checkout_index = next(
            i
            for i, step in enumerate(jobs[name]["steps"])
            if str(step.get("uses", "")).startswith("actions/checkout@")
        )
        identity = jobs[name]["steps"][checkout_index + 1]
        if (
            identity.get("run", "").strip() != 'test "$(git rev-parse HEAD)" = "$GITHUB_SHA"'
            or "if" in identity
            or identity.get("continue-on-error")
        ):
            problems.append(f"{name} source identity is not checked before consumption")
    return problems


def _tag_retarget_model(doc: dict[Any, Any], event: str, tag_moved: bool) -> dict[str, Any]:
    """Nonwriting model of actual checkout expressions/needs, not Actions execution."""
    jobs = doc["jobs"]
    event_sha, moved_sha = "a" * 40, "b" * 40

    def resolve(ref: str, tag_sha: str) -> str:
        if ref == IMMUTABLE_SOURCE_REF:
            return event_sha
        if ref == RELEASE_TARGET_REF:
            return tag_sha if event == "release" else event_sha
        raise AssertionError(f"unsupported source-ref expression: {ref}")

    def checkout(name: str) -> str:
        return next(
            step["with"]["ref"]
            for step in jobs[name]["steps"]
            if str(step.get("uses", "")).startswith("actions/checkout@")
        )

    def gate_dependency(name: str, visited: set[str]) -> bool:
        if name == INVARIANT_JOB:
            return True
        if name in visited:
            return False
        visited.add(name)
        needs = jobs[name].get("needs", [])
        if isinstance(needs, str):
            needs = [needs]
        return any(gate_dependency(parent, visited) for parent in needs)

    validated = resolve(checkout(INVARIANT_JOB), event_sha)
    assert validated == event_sha
    later_tag = moved_sha if tag_moved else event_sha
    # Extension is release-only; its dispatch ref is still checked as a source
    # configuration control, not a claim that the dispatch actually runs it.
    rows = {}
    for name in ("release-build", "extension-release", "keyagent-macos"):
        steps = jobs[name]["steps"]
        index = next(
            i
            for i, step in enumerate(steps)
            if str(step.get("uses", "")).startswith("actions/checkout@")
        )
        next_step = steps[index + 1]
        identity_assert = (
            next_step.get("run", "").strip() == 'test "$(git rev-parse HEAD)" = "$GITHUB_SHA"'
            and "if" not in next_step
            and not next_step.get("continue-on-error")
        )
        source = resolve(checkout(name), later_tag)
        dependency = gate_dependency(name, set())
        rows[name] = {
            "guard_dependency": dependency,
            "validated_commit": validated,
            "source_commit": source,
            "identity_assert": identity_assert,
            "build_reachable": dependency and (not identity_assert or source == event_sha),
        }
    return rows


def test_release_tag_retarget_cannot_change_validated_source() -> None:
    for event, moved in (("release", True), ("release", False), ("workflow_dispatch", True)):
        rows = _tag_retarget_model(_workflow(), event, moved)
        assert all(row["guard_dependency"] for row in rows.values())
        assert all(row["source_commit"] == row["validated_commit"] for row in rows.values())
        assert all(row["build_reachable"] for row in rows.values())


def test_each_mutable_source_ref_is_rejected_after_guard_success() -> None:
    for name in ("release-build", "extension-release", "keyagent-macos"):
        doc = copy.deepcopy(_workflow())
        step = next(
            step
            for step in doc["jobs"][name]["steps"]
            if str(step.get("uses", "")).startswith("actions/checkout@")
        )
        step["with"]["ref"] = RELEASE_TARGET_REF
        row = _tag_retarget_model(doc, "release", True)[name]
        assert row["guard_dependency"]
        assert row["source_commit"] != row["validated_commit"]
        assert not row["build_reachable"]
        assert f"{name} checkout is not immutable event SHA" in _release_invariant_problems(doc)


def test_old_unchecked_tag_retarget_cannot_return_silently() -> None:
    for name in ("release-build", "extension-release", "keyagent-macos"):
        doc = copy.deepcopy(_workflow())
        steps = doc["jobs"][name]["steps"]
        index = next(
            i
            for i, step in enumerate(steps)
            if str(step.get("uses", "")).startswith("actions/checkout@")
        )
        steps[index]["with"]["ref"] = RELEASE_TARGET_REF
        del steps[index + 1]
        row = _tag_retarget_model(doc, "release", True)[name]
        assert row["build_reachable"] and row["source_commit"] != row["validated_commit"]
        assert _release_invariant_problems(doc)


def test_source_identity_assertion_cannot_be_skipped() -> None:
    for name in ("release-build", "extension-release", "keyagent-macos"):
        doc = copy.deepcopy(_workflow())
        steps = doc["jobs"][name]["steps"]
        index = next(
            i
            for i, step in enumerate(steps)
            if str(step.get("uses", "")).startswith("actions/checkout@")
        )
        steps[index + 1]["if"] = "${{ false }}"
        assert (
            f"{name} source identity is not checked before consumption"
            in _release_invariant_problems(doc)
        )


def test_extension_change_detection_uses_verified_commit() -> None:
    step = next(
        step
        for step in _workflow()["jobs"]["extension-release"]["steps"]
        if step.get("id") == "ext-changed"
    )
    assert '"${GITHUB_SHA}^"' in step["run"]
    assert 'git diff --quiet "$PREV_TAG" "$GITHUB_SHA" -- extension/' in step["run"]
    assert 'git diff --quiet "$PREV_TAG" "$CURRENT_TAG"' not in step["run"]


def test_release_invariants_guard_both_event_paths() -> None:
    doc = _workflow()
    assert {"release", "workflow_dispatch"} <= set(_triggers(doc))
    assert _release_invariant_problems(doc) == []


def test_release_invariants_cannot_be_conditionally_skipped() -> None:
    doc = copy.deepcopy(_workflow())
    for condition in ("needs.changes.outputs.i18n == 'true'", "${{ false }}", False):
        doc["jobs"][INVARIANT_JOB]["if"] = condition
        assert "invariant job can be skipped" in _release_invariant_problems(doc)


def test_release_invariant_steps_cannot_be_skipped() -> None:
    doc = copy.deepcopy(_workflow())
    doc["jobs"][INVARIANT_JOB]["steps"][-1]["if"] = "${{ false }}"
    assert "invariant step can be skipped" in _release_invariant_problems(doc)


def test_release_invariant_commands_cannot_be_dropped_or_reordered() -> None:
    for mutation in ("drop", "reorder"):
        doc = copy.deepcopy(_workflow())
        step = doc["jobs"][INVARIANT_JOB]["steps"][-1]
        generation = "python scripts/i18n/generate.py --check"
        parity = "python scripts/i18n/check.py"
        if mutation == "drop":
            step["run"] = step["run"].replace(parity, "echo parity skipped")
            expected = "invariant commands missing"
        else:
            step["run"] = step["run"].replace(generation, "SWAP").replace(parity, generation)
            step["run"] = step["run"].replace("SWAP", parity)
            expected = "invariant commands reordered"
        assert expected in _release_invariant_problems(doc)


def test_release_build_and_extension_cannot_bypass_guard() -> None:
    for name in ("release-build", "extension-release"):
        doc = copy.deepcopy(_workflow())
        doc["jobs"][name]["needs"] = []
        assert f"{name} bypasses invariant dependency" in _release_invariant_problems(doc)


def test_pypi_cannot_drop_transitive_invariant_dependency() -> None:
    doc = copy.deepcopy(_workflow())
    doc["jobs"]["pypi-publish"]["needs"] = []
    assert "pypi-publish publisher bypasses invariants" in _release_invariant_problems(doc)


def test_release_guard_cannot_depend_on_pr_classifier() -> None:
    doc = copy.deepcopy(_workflow())
    doc["jobs"]["changes"] = {"runs-on": "ubuntu-latest", "steps": []}
    doc["jobs"][INVARIANT_JOB]["needs"] = ["changes"]
    assert "invariant job can be skipped" in _release_invariant_problems(doc)


def test_release_build_cannot_ignore_guard_failure() -> None:
    doc = copy.deepcopy(_workflow())
    doc["jobs"]["release-build"]["if"] = "${{ always() }}"
    assert "release-build overrides dependency success" in _release_invariant_problems(doc)


def test_release_invariant_cannot_check_mutable_main() -> None:
    doc = copy.deepcopy(_workflow())
    step = doc["jobs"][INVARIANT_JOB]["steps"][0]
    step["with"]["ref"] = "main"
    assert "invariant checkout is not the event target" in _release_invariant_problems(doc)


REPO = Path(__file__).resolve().parents[2]
PUBLISH_YML = REPO / ".github" / "workflows" / "publish.yml"

#: The action refs that write outside the repository. A step using one of these is
#: what makes a job publish-capable; the assertion below fails loudly if neither is
#: found, so renaming an action breaks this module rather than emptying it.
PUBLISHING_ACTIONS = (
    "pypa/gh-action-pypi-publish",
    "softprops/action-gh-release",
)

#: The event gate, as it must appear in a publisher's ``if:``. Compared after
#: whitespace normalisation so a reflowed expression is still recognised.
RELEASE_EVENT_GATE = "github.event_name == 'release'"


def _workflow() -> dict[Any, Any]:
    """``publish.yml`` as parsed YAML.

    PyYAML resolves the bare ``on:`` key to the boolean ``True`` (YAML 1.1), so the
    trigger block is read through the same normalisation in one place rather than
    re-derived per test.
    """
    return yaml.safe_load(PUBLISH_YML.read_text())


def _triggers(doc: dict[Any, Any]) -> dict[str, Any]:
    """The ``on:`` block, keyed by event name whatever PyYAML did to the key."""
    block = doc.get("on", doc.get(True))
    assert isinstance(block, dict), "publish.yml has no trigger block"
    return {str(k): v for k, v in block.items()}


def _normalised(text: Any) -> str:
    return " ".join(str(text or "").split())


def _publishing_jobs(doc: dict[Any, Any]) -> dict[str, str]:
    """Map job name -> the publishing action that job invokes.

    A job is publish-capable when any of its steps uses one of
    :data:`PUBLISHING_ACTIONS`. Returns ``{}`` only for a workflow that genuinely
    has none, which the caller must treat as a failure of this module's premise
    rather than as a pass.
    """
    found: dict[str, str] = {}
    for name, job in doc["jobs"].items():
        for step in job.get("steps") or []:
            uses = _normalised(step.get("uses"))
            for action in PUBLISHING_ACTIONS:
                if uses.startswith(action):
                    found[name] = action
    return found


def _ungated_publishers(doc: dict[Any, Any]) -> list[str]:
    """Publish-capable jobs whose own ``if:`` does not require the release event."""
    return sorted(
        name
        for name in _publishing_jobs(doc)
        if RELEASE_EVENT_GATE not in _normalised(doc["jobs"][name].get("if"))
    )


def test_the_workflow_has_publish_capable_jobs_to_guard() -> None:
    """The premise: a gate test over an empty set passes for the wrong reason."""
    publishers = _publishing_jobs(_workflow())
    assert publishers, (
        "no job in publish.yml invokes a publishing action — either the release path "
        "moved out of this file, or an action ref was renamed and this module stopped "
        f"looking at anything. Looked for: {PUBLISHING_ACTIONS}"
    )
    assert "pypi-publish" in publishers, (
        "the PyPI publisher is gone or renamed; the gate assertions below would no "
        "longer cover the upload that v0.62.39 never made"
    )


def test_every_publish_capable_job_requires_the_release_event() -> None:
    """A dispatched run must reach no publisher.

    The gate must be on the JOB, not inferred from what it ``needs``: a dispatched
    run satisfies every ``needs``-based condition by construction — that is the whole
    point of the manual trigger — so the event check is the only thing standing
    between a rehearsal and a real upload.
    """
    ungated = _ungated_publishers(_workflow())
    assert not ungated, (
        f"{ungated} can publish without requiring the release event; a "
        "workflow_dispatch run would reach it. Add "
        f"`if: ${{{{ {RELEASE_EVENT_GATE} && … }}}}` to the job."
    )


def test_a_manual_dispatch_reaches_the_signing_job_but_no_publisher() -> None:
    """Both halves of the trigger's contract, from the parsed workflow.

    A dispatch that cannot reach ``keyagent-macos`` is useless — that job is the only
    place the signed bundle is ever built, so a release-gated one reproduces the
    defect the trigger exists to end.
    """
    doc = _workflow()
    assert "workflow_dispatch" in _triggers(doc), (
        "publish.yml does not expose workflow_dispatch, so the signing job can only "
        "be tested by cutting a release"
    )
    assert "keyagent-macos" in doc["jobs"], "the signing job is gone or renamed"
    keyagent_if = _normalised(doc["jobs"]["keyagent-macos"].get("if"))
    assert "github.event_name" not in keyagent_if, (
        "keyagent-macos is gated on the event, so a manual dispatch can no longer "
        "build and sign the bundle — the dispatch exists to prove that path before a tag"
    )
    assert not _ungated_publishers(doc)


def test_the_pypi_publisher_still_depends_on_the_signed_macos_wheel() -> None:
    """The other half of the release contract, pinned for the same reason.

    ``pypi-publish`` needing ``keyagent-macos`` is what stops a release shipping a
    pure wheel when the signing job fails: ``supported()`` is True on macOS, so the
    key agent's presence tier is what the macOS wheel carries, and publishing without
    it is the silent downgrade the design forbids. Dropping that edge is silent too —
    the publish succeeds, minus the feature.
    """
    doc = _workflow()
    needs = doc["jobs"]["pypi-publish"].get("needs") or []
    if isinstance(needs, str):
        needs = [needs]
    assert "keyagent-macos" in needs, (
        "pypi-publish no longer needs keyagent-macos, so a release whose signing job "
        "failed would publish the pure wheel alone"
    )


# --- the checker itself, against the two defects it exists to catch -------------
# Without these, a checker that returned `[]` for every input would pass every
# assertion above — the failure mode this whole module is a response to.


def _with_gate_removed(gate: str = RELEASE_EVENT_GATE) -> dict[Any, Any]:
    """A copy of the workflow with ``gate`` stripped out of every ``if:``."""
    doc = copy.deepcopy(_workflow())
    for job in doc["jobs"].values():
        if isinstance(job.get("if"), str):
            job["if"] = job["if"].replace(gate, "").replace(" &&  && ", " && ")
    return doc


def test_the_gate_check_catches_a_dropped_gate() -> None:
    """Defect 1: the literal is deleted, and a dispatch can publish."""
    mutated = _with_gate_removed()
    assert _ungated_publishers(mutated), (
        "removing the release-event gate from every job did not make any job look "
        "ungated — this module cannot catch the defect it exists for"
    )
    assert set(_ungated_publishers(mutated)) == set(_publishing_jobs(mutated))


def test_the_gate_check_catches_a_mistyped_event() -> None:
    """Defect 2: the gate is present but names the wrong event.

    ``workflow_dispatch`` here is the plausible mis-edit, and it fails in the
    direction nobody watches: the rehearsal still works, so the mistake is only
    visible when a real release PUBLISHES NOTHING.
    """
    doc = copy.deepcopy(_workflow())
    for job in doc["jobs"].values():
        if isinstance(job.get("if"), str):
            job["if"] = job["if"].replace(
                RELEASE_EVENT_GATE, "github.event_name == 'workflow_dispatch'"
            )
    assert set(_ungated_publishers(doc)) == set(_publishing_jobs(doc))


def test_the_gate_check_would_catch_a_second_publisher_job() -> None:
    """A new publisher added without the gate must be caught too.

    The defect this guards is additive: someone adds a job that uploads an asset and
    does not carry the gate over, which today would publish from a rehearsal.
    """
    doc = copy.deepcopy(_workflow())
    doc["jobs"]["brand-new-publisher"] = {
        "runs-on": "ubuntu-latest",
        "steps": [{"uses": "softprops/action-gh-release@v3"}],
    }
    assert "brand-new-publisher" in _ungated_publishers(doc)
