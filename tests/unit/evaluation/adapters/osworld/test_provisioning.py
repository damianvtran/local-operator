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


# ---------------------------------------------------------------------------
# AWS_ROOT_VOLUME_SIZE: the disk-exhaustion escape hatch
# ---------------------------------------------------------------------------
#
# The failure this knob answers is a CLOCK, not a workload: the root filesystem
# reaches 0 bytes at ~t+383s and the next screenshot request fails. 7 of 8 runs
# first failed in a 424-466s window regardless of agent activity (16-32 steps)
# and on both t3.xlarge and m5.xlarge -- the volume is identical either way,
# which is why the instance-type knob changed nothing.
#
# The CONSUMER is the guest's own snapd (a 9.7 GB /var/lib/snapd/cache plus a
# boot-time auto-refresh), NOT the x11grab recorder an earlier revision of this
# comment named: pgrep found no ffmpeg process on a failing guest. The adapter
# now reclaims that space itself at episode start (``guest_disk``); this knob
# remains an independent lever, and cannot fully fix it on its own because the
# root partition stays 29.5G inside whatever volume is requested.


def _with_volume_size(value: str) -> tuple[ScopedInfraValue, ...]:
    return _INFRA + (
        ScopedInfraValue(name="AWS_ROOT_VOLUME_SIZE", purpose="benchmark_compute", value=value),
    )


def test_a_root_volume_override_replaces_the_launch_time_default() -> None:
    descriptor = taskfile.load_static(fixtures.PLAIN.encode(), module_name="tasks/t.py")
    plan = provisioning.resolve(
        descriptor, episode_id="ep-1", infra_values=_with_volume_size("120")
    )
    assert plan.volume_gb == 120


def test_a_root_volume_override_beats_a_task_pinned_volume_size() -> None:
    # The precedence decision, asserted rather than assumed, and deliberately
    # the same rule AWS_INSTANCE_TYPE uses: an override a task pin could veto
    # would fail on exactly the tasks that run long enough to hit the wall.
    descriptor = taskfile.load_static(fixtures.CUSTOM_INSTANCE.encode(), module_name="tasks/t.py")
    assert descriptor.volume_size == 100
    plan = provisioning.resolve(
        descriptor, episode_id="ep-1", infra_values=_with_volume_size("250")
    )
    assert plan.volume_gb == 250


def test_a_task_pinned_volume_size_still_wins_without_an_override() -> None:
    # The override is opt-in: absent it, the task's own pin is untouched.
    assert _resolve(fixtures.CUSTOM_INSTANCE).volume_gb == 100


def test_volume_stays_none_without_an_override_or_a_task_pin() -> None:
    # Omitting the knob must reproduce today's behaviour EXACTLY -- None, so
    # OSWorld's own launch-time resolution from the AMI's BDM still runs.
    assert _resolve(fixtures.PLAIN).volume_gb is None


@pytest.mark.parametrize(
    "bad",
    [
        # No empty-string case: ``ScopedInfraValue`` rejects it upstream with a
        # min_length constraint, so it cannot reach ``resolve`` at all.
        "eighty",  # not a number at all
        "40.5",  # fractional GiB do not exist
        "1e3",  # scientific notation
        "0x28",  # hex
        " 40",  # leading whitespace int() would silently accept
        "40 ",  # trailing whitespace
        "+40",  # signed, which int() would also accept
        "4_0",  # PEP 515 underscore int() would read as 40
        "\uff14\uff10",  # full-width digits str.isdigit() alone accepts
        "40; rm -rf /",  # anything outside the closed shape
    ],
)
def test_a_malformed_root_volume_override_is_rejected_at_prepare_time(bad: str) -> None:
    # Rejected, not ignored: silently discarding it would launch the 2.2 GB-free
    # default the operator was escaping and lose another paid episode at t+424s.
    descriptor = taskfile.load_static(fixtures.PLAIN.encode(), module_name="tasks/t.py")
    with pytest.raises(ProvisioningError) as excinfo:
        provisioning.resolve(descriptor, episode_id="ep-1", infra_values=_with_volume_size(bad))
    assert "AWS_ROOT_VOLUME_SIZE" in str(excinfo.value)


@pytest.mark.parametrize("bad", ["0", "-1", "-100", "16385", "999999"])
def test_an_out_of_range_root_volume_override_is_rejected(bad: str) -> None:
    # Zero and negatives are meaningless; above 16384 GiB is refused by AWS
    # itself because the provider pins gp3. Catching them here costs nothing
    # and beats an opaque botocore error midway through a paid launch.
    descriptor = taskfile.load_static(fixtures.PLAIN.encode(), module_name="tasks/t.py")
    with pytest.raises(ProvisioningError) as excinfo:
        provisioning.resolve(descriptor, episode_id="ep-1", infra_values=_with_volume_size(bad))
    assert "AWS_ROOT_VOLUME_SIZE" in str(excinfo.value)


@pytest.mark.parametrize("good", ["1", "40", "80", "120", "16384"])
def test_root_volume_sizes_at_and_inside_the_bounds_are_accepted(good: str) -> None:
    # The bounds are INCLUSIVE; an off-by-one here would reject the largest
    # gp3 volume AWS sells, and 1 is the smallest the API accepts.
    descriptor = taskfile.load_static(fixtures.PLAIN.encode(), module_name="tasks/t.py")
    plan = provisioning.resolve(descriptor, episode_id="ep-1", infra_values=_with_volume_size(good))
    assert plan.volume_gb == int(good)


def test_the_two_infra_overrides_apply_independently() -> None:
    # Both knobs answer the same symptom (a frameless observation) from
    # different causes, so an operator hitting both applies both at once.
    descriptor = taskfile.load_static(fixtures.PLAIN.encode(), module_name="tasks/t.py")
    infra = _INFRA + (
        ScopedInfraValue(name="AWS_INSTANCE_TYPE", purpose="benchmark_compute", value="m5.2xlarge"),
        ScopedInfraValue(name="AWS_ROOT_VOLUME_SIZE", purpose="benchmark_compute", value="150"),
    )
    plan = provisioning.resolve(descriptor, episode_id="ep-1", infra_values=infra)
    assert plan.instance_type == "m5.2xlarge"
    assert plan.volume_gb == 150
