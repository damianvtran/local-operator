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
