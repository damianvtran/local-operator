"""Derive adapter requirements from a TaskDescriptor — never a hardcoded list.

Every requirement is a *function of the task*: "anything that is a property of
the task is derived; anything that is a property of the AWS account or an
external account is declared." That line is exactly the ``Requirement.kind``
split between ``"secret"`` (a name the host resolves to bytes) and ``"infra"``
(a non-secret ``ScopedInfraValue``).

The adapter's job is to NAME requirements accurately. The richer
``receipts.*Requirement`` models (``ComputeRequirement``,
``ExternalServiceRequirement``, …) are the host's own ``DependencyPlan``
vocabulary, built by the harness; the adapter returns the closed
``api.Requirement`` type, and the host maps names to plan entries.

Rules trace each requirement to a task-file fact, so the unit tests are the
executable spec of this table.
"""

from __future__ import annotations

from lop_osworld_v2_adapter.taskfile import TaskDescriptor

from local_operator.evaluation.adapters.api import Requirement

# ---------------------------------------------------------------------------
# Always-on requirements. These exist because the worker environment is
# stripped to locale/temp (supervisor._ENV_ALLOW), so NOTHING — not HOME, not
# PATH, not AWS_* — is inherited. Everything the boto3 session or the OSWorld
# guest needs must arrive explicitly over RPC.
# ---------------------------------------------------------------------------

# AWS credentials: the provider's boto3 session cannot fall back to
# ~/.aws/credentials because HOME is absent in the worker.
_AWS_SECRETS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
)

# AWS account facts: OSWorld's manager raises without a subnet and security
# group (manager.py:208-211) and defaults to us-east-1 but must be told the
# account's actual region/subnet/SG. AWS_SCHEDULER_ROLE_ARN is REQUIRED, not
# optional: without it the EventBridge TTL creation degrades to a logged
# warning (manager.py:274-276), removing the last line of defence against an
# orphaned instance if this machine dies.
_AWS_INFRA = (
    "AWS_REGION",
    "AWS_SUBNET_ID",
    "AWS_SECURITY_GROUP_ID",
    "AWS_SCHEDULER_ROLE_ARN",
)

# The guest's `user` password and the asset mirror. OSWORLD_FILE_BASE_URL is
# required because OSWorld's README says not to rely on online asset
# resolution; the release pins assets at a base URL.
_ALWAYS_INFRA = (
    "OSWORLD_CLIENT_PASSWORD",
    "OSWORLD_FILE_BASE_URL",
)


def _requirement(name: str, *, kind: str, required: bool) -> Requirement:
    # requirement_id is the name itself: it is unique within an episode and
    # self-describing, which is what rescue-from-descriptor needs.
    return Requirement(
        requirement_id=name,
        kind=kind,  # type: ignore[arg-type]
        name=name,
        required=required,
    )


def _has_config_type(descriptor: TaskDescriptor, *types: str) -> bool:
    for entry in descriptor.config:
        if isinstance(entry, dict) and entry.get("type") in types:
            return True
    return False


def _evaluator_text(descriptor: TaskDescriptor) -> str:
    """Flatten the evaluator structure for substring checks.

    The evaluator is OSWorld's own nested dict; we only ever look for the
    presence of specific getter/func names, so a flattened repr is sufficient
    and avoids a recursive walk that would need to handle every shape.
    """

    return repr(descriptor.evaluator) if descriptor.evaluator is not None else ""


def derive_requirements(descriptor: TaskDescriptor) -> tuple[Requirement, ...]:
    """Return the exact requirement set for one task, in deterministic order."""

    out: list[Requirement] = []

    for name in _AWS_SECRETS:
        out.append(_requirement(name, kind="secret", required=True))
    for name in _AWS_INFRA:
        out.append(_requirement(name, kind="infra", required=True))
    for name in _ALWAYS_INFRA:
        out.append(_requirement(name, kind="infra", required=True))

    # --- Conditional on the task -------------------------------------------

    if descriptor.proxy:
        # The DataImpulse proxy needs credentials and an endpoint; both are
        # task-declared needs but account-declared values.
        out.append(_requirement("OSWORLD_PROXY_CREDENTIALS", kind="secret", required=True))
        out.append(_requirement("OSWORLD_PROXY_ENDPOINT", kind="infra", required=True))

    if _has_config_type(descriptor, "googledrive", "login"):
        out.append(_requirement("GOOGLE_ACCOUNT_CREDENTIALS", kind="secret", required=True))

    # user_simulator is a dict like {"type": "llm", "provider": ..., "model": ...}
    # Only an LLM-backed simulator needs an API key; scripted/fixed do not.
    sim = descriptor.user_simulator
    if isinstance(sim, dict) and sim.get("type") == "llm":
        out.append(_requirement("OSWORLD_USER_SIM_API_KEY", kind="secret", required=True))

    # A date-sensitive evaluator needs the host to pin the episode clock.
    # Detected from the evaluator text so the requirement follows the task.
    evaluator_text = _evaluator_text(descriptor)
    if "rule_relativeTime" in evaluator_text or "relativeTime" in evaluator_text:
        out.append(_requirement("OSWORLD_TASK_DATE", kind="infra", required=False))

    # Controllers that raise at IMPORT when their env var is unset (C4). The
    # adapter never imports them at adapter-import time; the requirement is
    # declared so the host injects the value before the first lazy import.
    # Detection reads the task's SOURCE, not its module name.
    source_text = descriptor.source_text
    if "controllers.gitlab" in source_text or "controllers import gitlab" in source_text:
        out.append(_requirement("GITLAB_PRIVATE_TOKEN", kind="secret", required=True))
        out.append(_requirement("GITLAB_URL", kind="infra", required=True))
    if "controllers.website" in source_text or "WEBSITE_HOST_SUFFIX" in source_text:
        out.append(_requirement("WEBSITE_HOST_SUFFIX", kind="infra", required=True))

    # The gated HF corpus is materialised into the workspace at build time, so
    # HF_TOKEN is NOT required at episode time. It is optional here for the
    # case where a host wants to re-fetch rather than use the pinned corpus.
    out.append(_requirement("HF_TOKEN", kind="secret", required=False))

    # Deterministic order: dedupe by name, keep the required-flag of the
    # stricter declaration, sort by name so two parses of the same task give
    # the identical tuple.
    deduped: dict[str, Requirement] = {}
    for req in out:
        existing = deduped.get(req.name)
        if existing is None or (req.required and not existing.required):
            deduped[req.name] = req
    return tuple(deduped[name] for name in sorted(deduped))
