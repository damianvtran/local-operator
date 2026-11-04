"""H0 — the Team Bridge moderation evaluation harness.

WHAT THIS PACKAGE IS. The measuring instrument for the privacy/moderation work on
Team Bridge: a versioned dataset of labelled asks, a fixture world with planted
canaries, and a runner that drives the REAL inbound screen and the REAL agent
reply path against that fixture with real models. It exists because the shipped
screening tests use a stub runner that returns literal verdicts (design §1 R12),
so nothing in the tree could answer "is this deployment actually leaking?".

WHY IT LIVES IN ``evaluation/`` AND NOT IN ``evaluation/runner/``. The runner core
must never import the application (``evaluation/session_arm.py`` states the rule
and its isolation test enforces it), while this apparatus exists precisely to
drive the real session. So it sits on the application side of that boundary, the
same side as ``session_arm``, and nothing under ``runner/`` may import it.

WHAT IS MEASURED, AND WHAT IS NOT.
  * Inbound: the production template bytes, the production ``ScreenModelRunner``
    dispatch and the production ``parse_verdict`` grammar. Real.
  * The agent turn: a real session opened through ``local_operator.sdk``, the
    production ``team_bridge_message`` delivery path, the real tool loop against
    the fixture world. Real.
  * The wire: NOT real. The hub is replaced by a recording client, because the
    point is what the agent WOULD have sent - and an eval that needs a production
    tenant to run is an eval nobody runs. Every egress is recorded before any
    network call, and the report says so.
  * The outbound gate (design §3.2) and the capability profile (design §4.2) do
    not exist on ``main`` yet. ``gates.py`` declares their seams and records their
    absence rather than reimplementing them: a second screening implementation
    would be measuring itself.

THE CANARY LAW (design review r1, M2). A leak is an EXACT canary string appearing
in text that left the agent. No judge decides whether something leaked; the judge,
where configured, only classifies refusal SHAPE and is recorded as such. This is
the one signal that is sound when the gate has no ground truth of its own.
"""

from __future__ import annotations

__all__ = [
    "DATASET_SCHEMA",
    "Dataset",
    "DatasetProblem",
    "FixtureWorld",
    "ModerationCase",
    "Taxonomy",
    "load_dataset",
    "load_taxonomy",
]

#: The case-file schema this harness reads. Bumped when a required field changes
#: meaning; the reporting layer records it so two reports cannot be compared
#: across an incompatible dataset revision by accident.
DATASET_SCHEMA = "tb-moderation-case-v1"


def __getattr__(name: str) -> object:
    """Resolve the public names lazily.

    WHY lazy: ``local_operator.evaluation.team_bridge_moderation`` is imported by
    the CLI before it touches the app, and the dataset/fixture modules pull in
    only the standard library. Keeping the re-export lazy means importing this
    package (for a version string, or from a test that only lints the data) never
    drags in the session engine, the provider stack or the bridge client.
    """

    if name in ("Taxonomy", "load_taxonomy"):
        from . import dataset as _dataset

        return getattr(_dataset, name)
    if name in ("Dataset", "DatasetProblem", "ModerationCase", "load_dataset"):
        from . import dataset as _dataset

        return getattr(_dataset, name)
    if name == "FixtureWorld":
        from . import fixture as _fixture

        return getattr(_fixture, name)
    raise AttributeError(name)
