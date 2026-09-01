"""ProvisioningPlan resolution: pure, total, no I/O.

The point of these tests is that ``resolve`` is a pure function of the task
plus the declared infra values — the entire content of ``prepare``'s
provisioning work, and total: it either returns a complete plan or raises
``ProvisioningError`` before anything exists. No boto3 client is constructed;
not even a read-only ``describe_images`` is issued.
"""

from __future__ import annotations

import pytest
from lop_osworld_v2_adapter import provisioning, taskfile
from lop_osworld_v2_adapter.provisioning import ProvisioningError

from local_operator.evaluation.adapters.api import ScopedInfraValue
from tests.unit.evaluation.adapters.osworld import fixtures

_INFRA = tuple(
    ScopedInfraValue(name=name, purpose="benchmark_compute", value=f"value-{name}")
    for name in (
        "AWS_REGION",
        "AWS_SUBNET_ID",
        "AWS_SECURITY_GROUP_ID",
        "AWS_SCHEDULER_ROLE_ARN",
        "OSWORLD_CLIENT_PASSWORD",
        "OSWORLD_FILE_BASE_URL",
    )
)


def _resolve(source: str):
    descriptor = taskfile.load_static(source.encode(), module_name="tasks/t.py")
    return provisioning.resolve(descriptor, episode_id="ep-1", infra_values=_INFRA)


def test_ami_falls_back_to_the_release_manifest_ami() -> None:
    plan = _resolve(fixtures.PLAIN)
    assert plan.ami_id == "ami-01017272139e01feb"


def test_task_image_overrides_the_default_when_valid() -> None:
    plan = _resolve(fixtures.CUSTOM_INSTANCE)
    assert plan.ami_id == "ami-0123456789abcdef0"
    assert plan.instance_type == "t3.2xlarge"
    assert plan.volume_gb == 100


def test_an_invalid_task_image_is_ignored() -> None:
    # A task image that is not a valid AMI id must not be used: the manifest
    # default is the only safe fallback, never a guess.
    descriptor = taskfile.load_static(fixtures.PLAIN.encode(), module_name="tasks/t.py")
    bad = taskfile.TaskDescriptor(**{**descriptor.__dict__, "image": "not-an-ami"})
    plan = provisioning.resolve(bad, episode_id="ep-1", infra_values=_INFRA)
    assert plan.ami_id == "ami-01017272139e01feb"


def test_volume_defaults_to_none_for_launch_time_resolution() -> None:
    # Left None so OSWorld's own resolve_aws_root_volume_size runs at launch;
    # replicating the lookup here would need the describe_images call prepare
    # must not make.
    plan = _resolve(fixtures.PLAIN)
    assert plan.volume_gb is None
    assert plan.instance_type == "t3.xlarge"


def test_screen_is_always_the_native_1920x1080() -> None:
    assert _resolve(fixtures.PLAIN).screen == (1920, 1080)


def test_a_missing_required_infra_value_raises() -> None:
    descriptor = taskfile.load_static(fixtures.PLAIN.encode(), module_name="tasks/t.py")
    short = tuple(v for v in _INFRA if v.name != "AWS_SUBNET_ID")
    with pytest.raises(ProvisioningError):
        provisioning.resolve(descriptor, episode_id="ep-1", infra_values=short)


def test_client_token_is_deterministic_per_episode() -> None:
    a = _resolve(fixtures.PLAIN)
    b = _resolve(fixtures.PLAIN)
    assert a.client_token == b.client_token
    other = provisioning.resolve(
        taskfile.load_static(fixtures.PLAIN.encode(), module_name="tasks/t.py"),
        episode_id="ep-2",
        infra_values=_INFRA,
    )
    assert other.client_token != a.client_token


def test_tags_carry_the_episode_and_adapter() -> None:
    tags = _resolve(fixtures.PLAIN).tag_dict()
    assert tags["Name"] == "lop-ep-ep-1"
    assert tags["lop:episode"] == "ep-1"
    assert tags["lop:adapter"] == "osworld-v2"


def test_proxy_flag_comes_from_the_task() -> None:
    assert _resolve(fixtures.PROXY).enable_proxy is True
    assert _resolve(fixtures.PLAIN).enable_proxy is False
