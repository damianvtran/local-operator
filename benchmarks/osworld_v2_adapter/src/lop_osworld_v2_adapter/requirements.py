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
from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType

from lop_osworld_v2_adapter.provisioning import resolve_proxy_policy
from lop_osworld_v2_adapter.taskfile import TaskDescriptor
from lop_osworld_v2_adapter.vendor_bridge import SECRET_ENV_NAMES, USER_SIM_KEY_ENV

from local_operator.evaluation.adapters.api import (
    Requirement,
    ResolvedSecret,
    ScopedInfraValue,
    SecretRef,
)

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
# AWS_ROOT_VOLUME_SIZE replaces the root volume size (GiB) for the benchmark VM.
# It is the escape hatch from the OTHER failure that presents identically: the
# guest's x11grab screen recorder fills the root filesystem at ~6.8 MB/s against
# ~2.2 GB free, so the disk hits 0 bytes at ~t+383s and the next screenshot
# request fails -- which is why 7 of 8 runs died in a 424-466s window on a clock
# rather than on workload, and why changing the instance type fixed nothing.
# Infra rather than a task field for the same content-hash reason.
# See provisioning._resolve_root_volume_gb.
_OPTIONAL_INFRA = (
    "AWS_INSTANCE_TYPE",
    "AWS_ROOT_VOLUME_SIZE",
    "OSWORLD_ENABLE_PROXY",
    "OSWORLD_INPUTS_ROOT",
    "OSWORLD_TTL_SECONDS",
    # The runner always pins the effective policy; optional preserves older
    # invocation compatibility while making adapters reject silent omissions.
    "OSWORLD_ACTION_SETTLE_POLICY",
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


class MissingImagingDecoder(RuntimeError):
    """The worker has no image decoder, so it cannot publish a frame.

    Every observation is bounded to the model's screen geometry before it is
    published (``observation.bound_screen_frame``), and that resize needs a
    real decoder. Without one the adapter cannot honour the dimensions
    invariant the host verifier enforces, so an episode would die on its FIRST
    observation — after the VM is allocated and paid for.
    """


def require_imaging_decoder() -> None:
    """Refuse an adapter environment that cannot resize a frame.

    An ENVIRONMENT precondition rather than a task-derived requirement, which
    is why it is a check and not a :class:`Requirement`: the ``Requirement``
    table names values the HOST resolves and injects, and no env var can
    conjure a missing wheel. It lives in this module anyway because this is
    where the adapter's preflight refusals are collected, and it is called from
    ``prepare`` — the last boundary that allocates nothing — so a venv built
    without the imaging extra fails for free instead of mid-episode.
    """

    from local_operator.helpers import pillow_image_module

    if pillow_image_module() is None:
        raise MissingImagingDecoder(
            "this adapter environment has no image decoder (Pillow), so guest "
            "screenshots cannot be bounded to the model's screen geometry; "
            "install the harness 'images' extra in the adapter venv"
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


def _baseline_requirements() -> tuple[Requirement, ...]:
    """The ALWAYS-ON layer: what every episode needs, whatever its task is.

    These are properties of the worker environment and of the selected cloud
    provider rather than of any task, so they are declared for every episode --
    including one whose task is not yet known, which is what lets
    ``inspect_requirements`` answer before the runner names the task.
    """

    out: list[Requirement] = []

    for name in _AWS_SECRETS:
        out.append(_requirement(name, kind="secret", required=True))
    for name in _AWS_INFRA:
        out.append(_requirement(name, kind="infra", required=True))
    for name in _ALWAYS_INFRA:
        out.append(_requirement(name, kind="infra", required=True))
    for name in _OPTIONAL_INFRA:
        out.append(_requirement(name, kind="infra", required=False))

    # The gated HF corpus is materialised into the workspace at build time, so
    # HF_TOKEN is NOT required at episode time. It is optional here for the
    # case where a host wants to re-fetch rather than use the pinned corpus.
    out.append(_requirement("HF_TOKEN", kind="secret", required=False))
    return tuple(out)


def derive_task_requirements(
    descriptor: TaskDescriptor, *, infra_values: tuple[ScopedInfraValue, ...] = ()
) -> tuple[Requirement, ...]:
    """The TASK-DERIVED layer: what THIS task adds to the always-on baseline.

    Every requirement whose presence is a function of the task's own fields is
    declared here. It is the layer :func:`require_supplied` refuses over, and
    the reason is structural: this set is knowable only once a descriptor
    exists, so no earlier boundary can see it -- ``inspect_requirements`` runs
    before the task is named and therefore answers from the baseline alone.
    """

    enable_proxy = resolve_proxy_policy(infra_values, task_proxy=bool(descriptor.proxy))
    out: list[Requirement] = []

    if is_judged(descriptor):
        out.append(_requirement(_JUDGE_SECRET, kind="secret", required=True))
        for name in _JUDGE_INFRA:
            out.append(_requirement(name, kind="infra", required=True))

    # --- Conditional on the task -------------------------------------------

    if descriptor.proxy and enable_proxy:
        # PROXY_CONFIG_FILE is what upstream ACTUALLY consumes: it loads the
        # pool from this path at import of desktop_env.controllers.setup. It is
        # required here because an absent pool is not a degraded run, it is a
        # guaranteed crash at reset_start after the VM is paid for.
        out.append(_requirement("PROXY_CONFIG_FILE", kind="infra", required=True))
        # OSWORLD_PROXY_CREDENTIALS and OSWORLD_PROXY_ENDPOINT are NO LONGER
        # required. They were declared when no working proxy path existed, and
        # they are consumed by nothing: a grep of this package finds
        # OSWORLD_PROXY_CREDENTIALS at no call site at all, and
        # OSWORLD_PROXY_ENDPOINT only in the env-injection allowlist -- upstream
        # reads neither, because it takes its endpoints from the pool file.
        # Demanding a secret the apparatus cannot consume trains an operator to
        # fabricate one, and a fabricated credential that "works" is worse than
        # a missing one. They stay OPTIONAL so an existing invocation that
        # supplies them is still accepted unchanged.
        out.append(_requirement("OSWORLD_PROXY_ENDPOINT", kind="infra", required=False))

    if _has_config_type(descriptor, "googledrive", "login"):
        # Declared OPTIONAL, and the row is kept so an invocation that already
        # supplies it is accepted unchanged. A search of the pinned tree finds
        # this name at NO call site at all -- the googledrive controller signs in
        # with the credentials its own settings carry -- so declaring it binding
        # is the anti-pattern this table already condemns for
        # OSWORLD_PROXY_CREDENTIALS: a required name the apparatus cannot consume
        # trains an operator to fabricate a credential.
        out.append(_requirement("GOOGLE_ACCOUNT_CREDENTIALS", kind="secret", required=False))

    # user_simulator is a dict like {"type": "llm", "provider": ..., "model": ...}
    # Only an LLM-backed simulator needs an API key; scripted/fixed do not. The
    # name comes from vendor_bridge (the module that owns the env names) so the
    # declaration, the delivery allowlist and the substitute table cannot spell it
    # three ways.
    sim = descriptor.user_simulator
    if isinstance(sim, dict) and sim.get("type") == "llm":
        out.append(_requirement(USER_SIM_KEY_ENV, kind="secret", required=True))

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

    return tuple(out)


def _merged(*layers: tuple[Requirement, ...]) -> tuple[Requirement, ...]:
    # Deterministic order: dedupe by name, keep the required-flag of the
    # stricter declaration, sort by name so two parses of the same task give
    # the identical tuple.
    deduped: dict[str, Requirement] = {}
    for layer in layers:
        for req in layer:
            existing = deduped.get(req.name)
            if existing is None or (req.required and not existing.required):
                deduped[req.name] = req
    return tuple(deduped[name] for name in sorted(deduped))


def derive_requirements(
    descriptor: TaskDescriptor, *, infra_values: tuple[ScopedInfraValue, ...] = ()
) -> tuple[Requirement, ...]:
    """Both layers in one ordered tuple: the set a host is TOLD about.

    The layers stay separable because only the task-derived one can be enforced
    at the task seam; see :func:`require_supplied` for what that leaves to the
    always-on baseline's own consumers.
    """

    return _merged(
        _baseline_requirements(),
        derive_task_requirements(descriptor, infra_values=infra_values),
    )


# Which declared SECRETS can actually reach their consumer in this build.
#
# A requirement the host cannot deliver is not one it can satisfy, and
# ``require_supplied`` refuses such a name outright rather than asking for a value
# that changes nothing. That is QA round 1's finding, not a hypothetical: this
# check matched GITLAB_PRIVATE_TOKEN by name, the operator supplied it, and the
# episode still allocated a guest and died in vendor code -- because
# ``desktop_env/controllers/gitlab.py:19-23`` reads it through ``os.getenv`` at
# import and nothing in this build put it there.
#
# DERIVED, never restated: the AWS pair reaches the provider directly out of the
# resolved secrets, and every other secret needs the env-delivery allowlist
# (``vendor_bridge``) that exists for exactly this reason. A name that gains a
# channel therefore becomes deliverable here for free, and a declaration naming
# something that has none is refused instead of offering an inert remedy.
#
# Infra values are absent from this table because they are deliverable by
# construction: they arrive as ``infra_values`` and the adapter injects the ones
# upstream reads into the worker's environment before its first import.
_DELIVERED_SECRETS = frozenset({"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"}) | SECRET_ENV_NAMES

# Names the pinned vendor ACCEPTS IN PLACE OF another, so supplying the substitute
# satisfies the requirement. ``user_simulator.respond`` hands its options to
# ``model_client.generate_chat`` WITHOUT a key when the simulator has none
# configured, and that client then resolves one through its own
# ``default_api_key_env`` -- which for these tasks is the judge key
# (``model_client.py:162-172``). Measured on the release corpus, 6 tasks
# (task_007/024/026/034/095/098) declare the simulator key and nothing else, so
# refusing it on its own absence would be a coverage regression: those episodes
# ran before this check existed. What they risk is also CONDITIONAL -- the
# simulator is only called when the agent asks the user something -- which is why
# the substitute, not the requirement, is what has to be relaxed.
_SECRET_SUBSTITUTES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {USER_SIM_KEY_ENV: (_JUDGE_SECRET,)}
)


class MissingRequiredRequirements(RuntimeError):
    """A requirement the task declares as REQUIRED was not supplied.

    Raised from ``reset_start`` once the descriptor exists and BEFORE the
    provider is constructed, so it names the absent REF and costs nothing: no
    guest is allocated and no vendor code runs, and the episode carries a
    diagnostic instead of a traceback from inside upstream's own setup.
    """


class UndeliverableRequirement(RuntimeError):
    """The task requires a value this build has no channel to hand over.

    Deliberately NOT the same failure as :class:`MissingRequiredRequirements`,
    because the remedy is different: that one asks the operator for a value they
    can supply, while this one says the episode cannot run here at all -- a
    supplied value changes nothing, so the choices are a build whose secret path
    carries the name or an excluded task. Raised from the same seam, so it is
    equally free: nothing is allocated and the diagnostic names only the ref.
    """


def undelivered_secrets(requirements: Iterable[Requirement]) -> tuple[str, ...]:
    """The required secret names no channel in this build can carry.

    Sorted, and derived from :data:`_DELIVERED_SECRETS` rather than from a list
    of names, so a secret that gains a delivery channel stops being refused by
    that change alone.
    """

    return tuple(
        sorted(
            req.name
            for req in requirements
            if req.required and req.kind == "secret" and req.name not in _DELIVERED_SECRETS
        )
    )


def _remedy(requirement: Requirement) -> str:
    """How an operator supplies this name, in one step: the flag and the kind."""

    if requirement.kind == "secret":
        flag = f"--secret {requirement.name}"
    else:
        flag = f"--infra {requirement.name}=..."
    alternatives = _SECRET_SUBSTITUTES.get(requirement.name)
    if alternatives:
        return f"{requirement.name} (supply as {flag}, or as the substitute {list(alternatives)})"
    return f"{requirement.name} (supply as {flag})"


def missing_required(
    requirements: Iterable[Requirement],
    *,
    supplied: set[str],
    substitutes: Mapping[str, tuple[str, ...]] = MappingProxyType({}),
) -> tuple[str, ...]:
    """The names in ``requirements`` that are required and were NOT supplied.

    A name counts as supplied when one of its accepted SUBSTITUTES was: the
    pinned vendor resolves some keys through a fallback chain, and refusing a
    name it would have found by another route is a coverage regression rather
    than a safety win.

    Sorted, so the refusal is deterministic. ``required=False`` declarations
    are excluded by construction: their absence is benign by the table's own
    contract, and refusing one would train an operator to fabricate a value
    nothing consumes.
    """

    satisfied = set(supplied)
    for name, alternatives in substitutes.items():
        if satisfied.intersection(alternatives):
            satisfied.add(name)
    return tuple(
        sorted(req.name for req in requirements if req.required and req.name not in satisfied)
    )


def require_supplied(
    descriptor: TaskDescriptor,
    *,
    task_id: str,
    secret_refs: Sequence[SecretRef | ResolvedSecret] = (),
    infra_values: Sequence[ScopedInfraValue] = (),
) -> None:
    """Refuse a task whose declared-but-absent requirements would break it later.

    THE CONTRACT: declaring a requirement means refusing when it is missing.
    A requirement the table marks ``required=True`` is one the task cannot run
    without, so an episode that proceeds without it does not run degraded -- it
    dies, and the death is measured in an allocated guest. This check is where
    that contract is enforced for the requirements a TASK introduces, and it is
    name-free by construction: it reads the table in
    :func:`derive_task_requirements` and never names a value of its own.

    WHY HERE, AND NOT IN THE RUNNER. The runner's gate
    (``episode._refuse_undeclared_disclosed_infra``) refuses a value an adapter
    does not DECLARE, and deliberately cannot refuse the inverse case: it calls
    ``inspect_requirements`` before the task is named, so a task-conditional
    requirement is legitimately absent from the answer it gates on. The derived
    set exists at exactly one seam -- after the descriptor is loaded and before
    the provider is constructed -- and that seam is this one.

    THE MEASURED FAILURE. 40 of the release's 108 tasks reference a controller
    that reads its value from the environment when the guest's task object is
    instantiated, and nothing in the pipeline supplied it. The episode built a
    guest and died INSIDE vendor code:

        ValueError: WEBSITE_HOST_SUFFIX must be set in environment variables
          threads.py:25 to_thread <- aws.py:863 _start_desktop_env
          <- vendor_bridge.py:202 instantiate_task <- task_016.py:7
          <- website.py:22

    -- a traceback naming no adapter, before any frame was scored, with an EC2
    instance left to tear down. GOOGLE_ACCOUNT_CREDENTIALS,
    OSWORLD_USER_SIM_API_KEY, GITLAB_PRIVATE_TOKEN and GITLAB_URL fail the same
    way. This turns every one of them into a single refusal that names the ref.

    WHAT IT DOES NOT COVER, deliberately: the always-on baseline. Those values
    belong to the worker environment and the SELECTED provider rather than to
    the task -- the AWS credentials are already refused by name when the
    provider is built (``adapter._aws_credentials``, still before allocation),
    and a cloud-free build (``adapter-provider.json`` selecting the fake)
    legitimately omits the whole group. Demanding them here would refuse a run
    the adapter can complete.

    Only REFS are named, never values: this text crosses the RPC boundary into
    the episode diagnostic and the sealed bundle.

    TWO REFUSALS, because there are two failure modes and they need different
    words. A name that is absent but deliverable is
    :class:`MissingRequiredRequirements`, and its message names the flag that
    supplies it (``--secret``/``--infra``) so the remedy is one step. A name this
    build cannot deliver at all is :class:`UndeliverableRequirement` and is
    refused whether or not it was supplied: an episode that proceeds on a value
    the vendor can never read would allocate a guest and die in vendor code, which
    is the exact defect QA round 1 reproduced on ``GITLAB_PRIVATE_TOKEN``. If you
    want that family runnable, the change is to put the name on the env-delivery
    allowlist -- a secret-plumbing change with its own review, deliberately not
    taken here.

    ``task_id`` is the name the RUNNER gave this task (``ResetStartParams``),
    which is the spelling the episode, the manifest and the operator's command
    line carry -- the descriptor's own ``task_id`` is the task file's internal
    id and can differ (``016`` for ``task_016.py``). The TABLE still comes from
    the descriptor; the two describe one task or the run is already wrong.
    """

    derived = derive_task_requirements(descriptor, infra_values=tuple(infra_values))
    supplied = {ref.name for ref in secret_refs} | {value.name for value in infra_values}

    undeliverable = undelivered_secrets(derived)
    if undeliverable:
        raise UndeliverableRequirement(
            f"task {task_id!r} declares {list(undeliverable)} as required, but this "
            "adapter build has no channel that can hand the value to the code that "
            "reads it, so supplying it would change nothing; refusing before the "
            "provider is constructed. A build whose secret path carries the name "
            "(vendor_bridge.SECRET_ENV_NAMES) can run this task; this one cannot."
        )

    missing = missing_required(derived, supplied=supplied, substitutes=_SECRET_SUBSTITUTES)
    if missing:
        by_name = {req.name: req for req in derived}
        raise MissingRequiredRequirements(
            f"task {task_id!r} declares {[_remedy(by_name[name]) for name in missing]} as "
            "required, and the host supplied neither a secret ref nor an infra value for "
            "them; refusing before the provider is constructed, because the task's own "
            "controller reads them the moment the guest's task object is instantiated"
        )
