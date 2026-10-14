"""Cloud compute provisioning: the provider seam, its records, and the ledger.

A LEAF PACKAGE BY DESIGN (§3.1). Nothing in the boot closure imports it, so it
never widens the import graph a plain ``lop`` start pays for; the CLI's one
dispatch arm is the only thing that reaches it, and that arrives in a later
slice. It imports ``local_operator.paths`` and ``local_operator.network.types``
(the mesh's ``MemberLifecycle``) and nothing else from the harness.

NOTHING HERE IS BOTO3-DEPENDENT AT IMPORT TIME. The modules that need the AWS SDK
import it inside the function that needs it, behind ``optional.require_extra``, so
``import local_operator.cloud`` succeeds on a default install and the failure a
user actually meets names the extra and the exact command that fixes it
(``tests/unit/cloud/test_provider_contract.py`` proves that with ``boto3``
simulated absent).
"""

from __future__ import annotations

from local_operator.cloud.aws import (
    EXPECTED_POC_ACCOUNT,
    REGION,
    AwsCredentials,
    CloudBlobs,
    FargateClients,
    assume_role,
    build_clients,
    sdk_exceptions,
    verify_account,
)
from local_operator.cloud.fake import FakeComputeProvider
from local_operator.cloud.fargate import (
    OVERRIDES_LIMIT_BYTES,
    RUN_DEADLINE_SECONDS,
    FargateProvider,
    FargateTargets,
)
from local_operator.cloud.ledger import (
    MAX_CLOUD_TASKS,
    PROVISION_DEADLINE_S,
    TASK_TTL_S,
    CloudLedger,
    cloud_dir,
    lock_for,
    read_request,
    read_task,
    request_path,
    state_for_status,
    task_path,
    tasks_dir,
)
from local_operator.cloud.provider import (
    CloudRefused,
    ComputeProvider,
    ComputeStatus,
    DrainOutcome,
    ProviderCapacity,
    ProviderPhase,
)
from local_operator.cloud.types import (
    COMPUTE_TASK_VERSION,
    POOL_REQUEST_VERSION,
    SIZES,
    ComputeTask,
    Hosting,
    PoolRequest,
    Size,
    SizeClass,
    TaskState,
    Tier,
    member_lifecycle_for,
)
from local_operator.cloud.workload import TASK_ENV_NAMES, task_environment

__all__ = [
    "COMPUTE_TASK_VERSION",
    "EXPECTED_POC_ACCOUNT",
    "MAX_CLOUD_TASKS",
    "OVERRIDES_LIMIT_BYTES",
    "POOL_REQUEST_VERSION",
    "PROVISION_DEADLINE_S",
    "REGION",
    "RUN_DEADLINE_SECONDS",
    "SIZES",
    "TASK_ENV_NAMES",
    "TASK_TTL_S",
    "AwsCredentials",
    "CloudBlobs",
    "CloudLedger",
    "CloudRefused",
    "ComputeProvider",
    "ComputeStatus",
    "ComputeTask",
    "DrainOutcome",
    "FakeComputeProvider",
    "FargateClients",
    "FargateProvider",
    "FargateTargets",
    "Hosting",
    "PoolRequest",
    "ProviderCapacity",
    "ProviderPhase",
    "Size",
    "SizeClass",
    "TaskState",
    "Tier",
    "assume_role",
    "build_clients",
    "cloud_dir",
    "lock_for",
    "member_lifecycle_for",
    "read_request",
    "read_task",
    "request_path",
    "sdk_exceptions",
    "state_for_status",
    "task_environment",
    "task_path",
    "tasks_dir",
    "verify_account",
]
