"""ProvisioningPlan: pure resolution of what reset_start will allocate.

The whole point of ``prepare`` being declarative (runner/episode.py:289-292)
is that this module performs NO I/O. ``resolve()`` takes the task descriptor
and the infra values the host supplied, and returns a fully-described plan.
No boto3 client is constructed; not even a read-only ``describe_images`` is
issued. The one OSWorld call that wants to happen before launch —
``resolve_aws_root_volume_size`` — is left inside ``reset_start``'s allocation
path, where OSWorld itself already puts it (manager.py:105-128).

Every rule traces to a task field or an infra value; there is no "latest"
fallback and no ambient default, because an episode that provisioned the
wrong AMI is a result nobody can reproduce.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from lop_osworld_v2_adapter.taskfile import TaskDescriptor

from local_operator.evaluation.adapters.api import ScopedInfraValue

_AMI_RE = re.compile(r"^ami-[a-f0-9]{8,17}$")
# ``<family>.<size>``, lowercase, both halves allowed internal hyphens so the
# real oddballs still parse: ``u-6tb1.metal``, ``m7i.metal-24xl``, ``c5n.18xlarge``.
# Shape borrowed from _AMI_RE on purpose -- a malformed value must fail in
# ``resolve`` (prepare time, nothing allocated) rather than as an opaque
# botocore InvalidParameterValue midway through a paid run_instances.
#
# SHAPE ONLY, deliberately: 'zz99.megahuge' is well-formed and nonexistent, and
# still reaches run_instances. Confirming a type EXISTS needs a live
# describe-instance-types call, and ``resolve`` performs no I/O by contract --
# that is the whole reason ``prepare`` can be declarative. So this narrows the
# failure window rather than closing it: a typo in the FORM of the value fails
# free, a plausible-but-fictional type still costs a launch attempt.
_INSTANCE_TYPE_RE = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*\.[a-z0-9]+(?:-[a-z0-9]+)*$")

# The V2 release manifest pins exactly one AMI for the 1920x1080 Ubuntu guest
# in us-east-1. Stated identically in benchmark_releases/osworld-v2-2026.08.08.json
# and docs/PUBLIC_EVALUATION_GUIDELINE.md. A task may override it with its own
# ``image`` field when that field is a valid AMI id.
_DEFAULT_AMI = "ami-01017272139e01feb"
_DEFAULT_INSTANCE_TYPE = "t3.xlarge"
# Public: the adapter's rescue path needs the same default when a descriptor
# predates AWS_REGION being supplied.
DEFAULT_REGION = "us-east-1"
_DEFAULT_REGION = DEFAULT_REGION
# The only screen geometry the V2 AMI map and the released IMAGE_ID_MAP carry.
_SCREEN = (1920, 1080)


class ProvisioningError(ValueError):
    """A required infra value was absent or a task field was unresolvable.

    Raised from ``resolve`` so ``prepare`` fails before persisting a plan the
    host cannot satisfy — cheap, and exactly what preflight is for.
    """


@dataclass(frozen=True)
class ProvisioningPlan:
    """What reset_start will allocate. Pure data; nothing has been created."""

    ami_id: str
    instance_type: str
    volume_gb: int | None  # None = resolve at launch from the AMI's own BDM
    screen: tuple[int, int]
    region: str
    subnet_id: str
    security_group_id: str
    scheduler_role_arn: str
    enable_proxy: bool
    client_password: str
    file_base_url: str
    tags: tuple[tuple[str, str], ...]
    client_token: str

    def tag_dict(self) -> dict[str, str]:
        return dict(self.tags)


def _infra(infra_values: tuple[ScopedInfraValue, ...], name: str) -> str:
    for value in infra_values:
        if value.name == name:
            return value.value
    raise ProvisioningError(f"required infra value {name!r} was not supplied")


def resolve(
    task: TaskDescriptor,
    *,
    episode_id: str,
    infra_values: tuple[ScopedInfraValue, ...],
) -> ProvisioningPlan:
    """Resolve the full plan from the task plus declared infra values.

    Pure: this function is the entire content of ``prepare``'s provisioning
    work, and it is total — it either returns a complete plan or raises before
    anything exists.
    """

    ami_id = (
        task.image if task.image is not None and _AMI_RE.fullmatch(task.image) else _DEFAULT_AMI
    )
    instance_type = _resolve_instance_type(task, infra_values)
    # Volume size is left None unless the task pins it: the AMI's own
    # BlockDeviceMappings carry a default that OSWorld resolves at launch,
    # and replicating that lookup here would require the very describe_images
    # call prepare must not make.
    volume_gb = task.volume_size

    region = (
        _infra(infra_values, "AWS_REGION") if _has(infra_values, "AWS_REGION") else _DEFAULT_REGION
    )
    subnet_id = _infra(infra_values, "AWS_SUBNET_ID")
    security_group_id = _infra(infra_values, "AWS_SECURITY_GROUP_ID")
    scheduler_role_arn = _infra(infra_values, "AWS_SCHEDULER_ROLE_ARN")
    client_password = _infra(infra_values, "OSWORLD_CLIENT_PASSWORD")
    file_base_url = _infra(infra_values, "OSWORLD_FILE_BASE_URL")

    # The client token is the idempotency key for run_instances. Minting it
    # from the episode ID makes a retried reset_start reuse the same launch
    # rather than double-allocate, and makes the resource_ref self-describing
    # for a rescue worker that has only the descriptor.
    client_token = hashlib.sha256(f"lop-osworld-v2|{episode_id}".encode()).hexdigest()[:32]

    tags = (
        ("Name", f"lop-ep-{episode_id}"),
        ("lop:episode", episode_id),
        ("lop:adapter", "osworld-v2"),
    )

    return ProvisioningPlan(
        ami_id=ami_id,
        instance_type=instance_type,
        volume_gb=volume_gb,
        screen=_SCREEN,
        region=region,
        subnet_id=subnet_id,
        security_group_id=security_group_id,
        scheduler_role_arn=scheduler_role_arn,
        enable_proxy=bool(task.proxy),
        client_password=client_password,
        file_base_url=file_base_url,
        tags=tags,
        client_token=client_token,
    )


def _has(infra_values: tuple[ScopedInfraValue, ...], name: str) -> bool:
    # Delegates to _get so this module has ONE lookup rule: a second scan with
    # its own matching logic is how "present" and "readable" drift apart.
    return _get(infra_values, name) is not None


def _resolve_instance_type(
    task: TaskDescriptor,
    infra_values: tuple[ScopedInfraValue, ...],
) -> str:
    """``AWS_INSTANCE_TYPE`` > the task's own pin > the release default.

    WHY THIS EXISTS. Five paid episodes died on
    ``ObservationError: environment returned no screenshot frame``. The cause
    was measured, not guessed: the default ``t3.xlarge`` is a BURSTABLE
    instance, and at the failure moment CloudWatch reported CPUCreditBalance
    4.2 with CPUSurplusCreditBalance 0.0 while CPU sat at 10.3% -- throttled
    to its baseline, not idle. A starved guest cannot answer its screenshot
    HTTP server, so the episode dies at step 9-32 having already spent
    $0.12-$0.32, and AWS status checks stay "ok" throughout because from the
    hypervisor's side nothing is wrong. Swapping to a non-burstable family
    (m5/c5) is the fix, and it has to be reachable WITHOUT editing task files:
    those are content-hash verified against the release pin, so editing one
    invalidates the very digest that makes a score reproducible.

    WHY THE OVERRIDE BEATS A TASK'S OWN PIN. A task pins ``instance_type``
    for a task-specific reason (a heavy build wants more cores), which argues
    for deferring to it. It loses anyway, because the two knobs answer
    different questions: the task author chose a SIZE against hardware they
    could reach, while the operator is working around an INFRASTRUCTURE
    failure that author never saw and cannot fix from inside a hash-pinned
    file. An override that a task pin could veto would silently fail exactly
    on the tasks most likely to be starved -- the ones heavy enough to pin a
    bigger box -- which is the opposite of the intent. The override is opt-in
    and absent by default, so this precedence is unreachable unless an
    operator deliberately supplied it; omitting the value reproduces the
    previous behaviour exactly, which is what keeps a default run comparable.

    Because the override changes the hardware a score was produced on, the
    host stamps it into the evidence manifest's metadata (see
    ``scripts/run_episode.py``), so a run on non-default hardware is
    disclosable from the bundle alone rather than only from operator memory.
    """

    override = _get(infra_values, "AWS_INSTANCE_TYPE")
    if override is not None:
        # Rejected, never ignored -- the asymmetry with ``image`` above is
        # deliberate. An invalid ``task.image`` falls back to the manifest AMI
        # because the task file is hash-pinned and the adapter cannot correct
        # it. An invalid override is something the operator typed seconds ago
        # and can fix; quietly discarding it would launch the burstable
        # default they were trying to escape and destroy more paid episodes
        # with no signal that the knob never took effect.
        if not _INSTANCE_TYPE_RE.fullmatch(override):
            raise ProvisioningError(
                f"AWS_INSTANCE_TYPE {override!r} is not a valid EC2 instance type "
                "(expected '<family>.<size>', e.g. 'm5.xlarge')"
            )
        return override
    return task.instance_type or _DEFAULT_INSTANCE_TYPE


def _get(infra_values: tuple[ScopedInfraValue, ...], name: str) -> str | None:
    for value in infra_values:
        if value.name == name:
            return value.value
    return None
