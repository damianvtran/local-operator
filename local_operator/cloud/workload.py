"""The container's environment, built from a request.

WHY THE ENV NAMES ARE VERBATIM, AND PINNED. ``infra/remote-agents-poc/image/
entrypoint.sh`` and ``image/lop_launch.py`` read these names, so renaming one is a
second change in the image and this slice does not need it (§4.1). The names are
therefore held in ONE tuple, and ``tests/unit/cloud/test_task_environment.py``
checks both directions: the builder emits exactly this list, and every name in it
still appears in the two image files that read it. Rename either side and the test
goes red, which is the only way a name that two programs agree on can be changed
safely.

WHAT IS NOT HERE. Defaults. ``POC_REPO_URL`` is passed through as the request
carries it, so an empty ``repo_url`` reaches the container empty — applying the
POC's fixture default is a CLI decision (§7.2), not a property of the request, and
a builder that quietly substituted one would hide the omission from every surface
that shows a request.
"""

from __future__ import annotations

from local_operator.cloud.types import PoolRequest

#: The ten names the image reads and this builder writes, in the order the POC's
#: ``build_environment`` emits them. The list is the contract between the provider
#: and the image; see the module docstring for who pins it.
TASK_ENV_NAMES: tuple[str, ...] = (
    "POC_RUN_ID",
    "POC_REPO_URL",
    "POC_SHA",
    "POC_PROMPT",
    "POC_PROBES_URL",
    "POC_RESULTS_URL",
    "POC_MODEL_SECRET_ARN",
    "POC_MOCK",
    "POC_HOSTING",
    "POC_MODEL",
)

#: The two names the POC appends only when it has a value for them.
CONDITIONAL_ENV_NAMES: frozenset[str] = frozenset({"POC_HOSTING", "POC_MODEL"})


def task_environment(
    request: PoolRequest,
    *,
    probes_url: str,
    results_url: str,
    model_secret_arn: str = "",
) -> list[dict[str, str]]:
    """The container's env overrides for ``request``.

    ``POC_RUN_ID`` is the ``request_id`` and not a fresh id: the container checks
    out the branch ``lop/$POC_RUN_ID``, and the task record's ``branch`` is
    ``lop/<request_id>``, so the two agree by construction rather than by a
    coincidence nobody would notice breaking.

    ``POC_MODEL_SECRET_ARN`` is an ARN, which is not a secret: it is here because
    the container's probe 4a must show that READING the model secret is denied, and
    naming the resource is what makes that a test rather than an assertion.

    ``POC_MOCK`` is ``"1"``/``"0"`` because the entrypoint compares it as a string,
    and the POC's key-free end-to-end run depends on that spelling.
    """
    environment = [
        {"name": "POC_RUN_ID", "value": request.request_id},
        {"name": "POC_REPO_URL", "value": request.repo_url},
        {"name": "POC_SHA", "value": request.repo_sha},
        {"name": "POC_PROMPT", "value": request.prompt},
        {"name": "POC_PROBES_URL", "value": probes_url},
        {"name": "POC_RESULTS_URL", "value": results_url},
        {"name": "POC_MODEL_SECRET_ARN", "value": model_secret_arn},
        {"name": "POC_MOCK", "value": "1" if request.mock else "0"},
    ]
    # Conditional on purpose, and the POC's own shape: the entrypoint distinguishes
    # "unset" from "set but empty" (it derives POC_HOSTING from the key's prefix
    # when the variable is absent), so an empty value is NOT the same statement.
    if request.hosting:
        environment.append({"name": "POC_HOSTING", "value": request.hosting})
    if request.model:
        environment.append({"name": "POC_MODEL", "value": request.model})
    return environment
