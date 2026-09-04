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


def _with_instance_type(value: str) -> tuple[ScopedInfraValue, ...]:
    return _INFRA + (
        ScopedInfraValue(name="AWS_INSTANCE_TYPE", purpose="benchmark_compute", value=value),
    )


def test_instance_type_defaults_to_the_release_default_without_an_override() -> None:
    # Omitting the knob must reproduce the previous behaviour EXACTLY: a score
    # run on default hardware stays comparable to every earlier run.
    plan = _resolve(fixtures.PLAIN)
    assert plan.instance_type == "t3.xlarge"


def test_an_instance_type_override_replaces_the_default() -> None:
    # The reason the knob exists: t3.xlarge is burstable, and credit
    # exhaustion (CPUCreditBalance 4.2, surplus 0.0) silently killed five paid
    # episodes by starving the guest's screenshot server.
    descriptor = taskfile.load_static(fixtures.PLAIN.encode(), module_name="tasks/t.py")
    plan = provisioning.resolve(
        descriptor, episode_id="ep-1", infra_values=_with_instance_type("m5.xlarge")
    )
    assert plan.instance_type == "m5.xlarge"


def test_an_instance_type_override_beats_a_task_pinned_instance_type() -> None:
    # The precedence decision, asserted rather than assumed. CUSTOM_INSTANCE
    # pins t3.2xlarge -- still burstable, so a task pin that could veto the
    # override would leave exactly the starvation case unfixable from outside
    # a hash-pinned task file.
    descriptor = taskfile.load_static(fixtures.CUSTOM_INSTANCE.encode(), module_name="tasks/t.py")
    assert descriptor.instance_type == "t3.2xlarge"
    plan = provisioning.resolve(
        descriptor, episode_id="ep-1", infra_values=_with_instance_type("m5.2xlarge")
    )
    assert plan.instance_type == "m5.2xlarge"


def test_a_task_pinned_instance_type_still_wins_without_an_override() -> None:
    # The override is opt-in: absent it, the task's own pin is untouched.
    plan = _resolve(fixtures.CUSTOM_INSTANCE)
    assert plan.instance_type == "t3.2xlarge"


@pytest.mark.parametrize(
    "bad",
    [
        "not-an-instance-type",  # no size separator at all
        "m5",  # family without a size
        "m5.",  # empty size
        ".xlarge",  # empty family
        "M5.XLarge",  # EC2 types are lowercase; a case slip must not launch
        "m5 .xlarge",  # whitespace
        "m5.xlarge; rm -rf /",  # anything that is not the closed shape
    ],
)
def test_a_malformed_instance_type_override_is_rejected_at_prepare_time(bad: str) -> None:
    # Rejected, not ignored: silently discarding it would launch the burstable
    # default the operator was escaping, with no signal the knob never applied.
    descriptor = taskfile.load_static(fixtures.PLAIN.encode(), module_name="tasks/t.py")
    with pytest.raises(ProvisioningError) as excinfo:
        provisioning.resolve(descriptor, episode_id="ep-1", infra_values=_with_instance_type(bad))
    assert "AWS_INSTANCE_TYPE" in str(excinfo.value)


@pytest.mark.parametrize(
    "good",
    [
        "m5.xlarge",
        "c5n.18xlarge",
        "u-6tb1.metal",  # hyphenated family
        "m7i.metal-24xl",  # hyphenated size
        "t3.nano",
    ],
)
def test_real_ec2_instance_type_spellings_are_accepted(good: str) -> None:
    # The validator must not reject the oddballs AWS actually sells; a regex
    # that only knows ``m5.xlarge`` would block the metal instances that are
    # the strongest answer to a starved burstable guest.
    descriptor = taskfile.load_static(fixtures.PLAIN.encode(), module_name="tasks/t.py")
    plan = provisioning.resolve(
        descriptor, episode_id="ep-1", infra_values=_with_instance_type(good)
    )
    assert plan.instance_type == good


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
