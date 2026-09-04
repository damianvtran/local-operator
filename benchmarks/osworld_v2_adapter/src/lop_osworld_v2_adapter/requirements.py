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

import ast

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

# Optional operator knobs. OSWORLD_INPUTS_ROOT names the durable directory
# holding the gated assets and the prepared checkout (the workspace pins their
# manifests by sha but cannot hold the 4.2 GB of assets under its 4 GiB cap);
# OSWORLD_TTL_SECONDS overrides the budget-derived lease length.
# AWS_INSTANCE_TYPE replaces the EC2 instance type for the benchmark VM. It is
# the escape hatch from burstable-credit exhaustion: the default t3.xlarge is
# BURSTABLE and a starved guest stops answering its screenshot server, which
# killed five paid episodes (CPUCreditBalance 4.2, surplus 0.0, CPU pinned at
# 10.3%) while AWS status checks read "ok". It is infra, not a task field,
# precisely because task files are content-hash verified and cannot be edited
# to work around the operator's hardware. See provisioning._resolve_instance_type.
_OPTIONAL_INFRA = (
    "AWS_INSTANCE_TYPE",
    "OSWORLD_INPUTS_ROOT",
    "OSWORLD_TTL_SECONDS",
)

# The LLM judge. OSWorld's ``model_client`` resolves the key from the
# environment and ``llm_metrics`` returns 0.0 on ANY exception, so a judged
# task run without a key scores a silent zero -- the previous pilot lost ~17%
# of its suite that way. These are REQUIRED for a task whose source imports
# the judge client, and absent for every other task, so preflight refuses a
# judged episode up front rather than sealing a zero.
_JUDGE_SECRET = "OSWORLD_EVAL_MODEL_API_KEY"
_JUDGE_INFRA = (
    "OSWORLD_EVAL_MODEL_PROVIDER",
    "OSWORLD_EVAL_MODEL_NAME",
)
# The judge's CALL SURFACE, not an import spelling. OSWorld exposes the LLM
# judge three ways and the pinned corpus uses all of them: the client itself
# (``model_client.generate_text``), the ``llm_metrics`` module, and the
# metric functions RE-EXPORTED through ``desktop_env.evaluators.metrics``
# (``metrics/__init__.py:194-200``), which a task reaches as
# ``metrics.compare_text_with_llm`` with no ``llm_metrics`` substring in
# its source at all (task_007). Detection therefore walks the task's AST for
# any reference -- attribute or bare name -- to one of these symbols, or any
# import of the two judge modules. Every symbol here is a judge entry point
# by construction of the pinned upstream; the set is closed and pinned with
# it. ``_with_llm`` covers the five metric names and any sibling added
# under the same convention. ``compare_pdf_answers`` (metrics/pdf.py) calls
# ``_compare_answers_with_llm`` for ``llm_match`` rules without the suffix
# in its own name; no pinned task uses it, but it is a judge entry point.
_JUDGE_MODULES = frozenset({"model_client", "llm_metrics"})
_JUDGE_SYMBOLS = frozenset({"generate_text", "generate_json", "compare_pdf_answers"})
_JUDGE_SYMBOL_SUFFIX = "_with_llm"


def _is_judge_symbol(name: str) -> bool:
    return name in _JUDGE_SYMBOLS or name.endswith(_JUDGE_SYMBOL_SUFFIX)


def is_judged(descriptor: TaskDescriptor) -> bool:
    """Whether the task's evaluator calls the LLM judge.

    AST-based so a re-exported metric (``metrics.compare_text_with_llm``) is
    caught the same as a direct import. A module that fails to parse cannot
    be judged honestly either way and falls back to a substring scan, which
    is strictly a superset of the old behaviour.
    """

    source_text = descriptor.source_text
    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        return any(module in source_text for module in _JUDGE_MODULES)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.rsplit(".", 1)[-1] in _JUDGE_MODULES:
                return True
            # ``from desktop_env.evaluators.metrics import llm_metrics`` names
            # the judge MODULE as an imported name, not as ``node.module``.
            if any(
                _is_judge_symbol(alias.name) or alias.name in _JUDGE_MODULES for alias in node.names
            ):
                return True
        elif isinstance(node, ast.Import):
            if any(alias.name.rsplit(".", 1)[-1] in _JUDGE_MODULES for alias in node.names):
                return True
        elif isinstance(node, ast.Attribute):
            if _is_judge_symbol(node.attr) or node.attr in _JUDGE_MODULES:
                return True
        elif isinstance(node, ast.Name):
            if _is_judge_symbol(node.id):
                return True
    return False


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
    for name in _OPTIONAL_INFRA:
        out.append(_requirement(name, kind="infra", required=False))

    if is_judged(descriptor):
        out.append(_requirement(_JUDGE_SECRET, kind="secret", required=True))
        for name in _JUDGE_INFRA:
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
