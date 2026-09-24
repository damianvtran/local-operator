"""Mixed infrastructure purposes reuse the closed adapter value contract."""

import pytest

from scripts import run_episode


def test_legacy_global_scope_and_value_bytes_are_preserved() -> None:
    values = run_episode._parse_infra(["NAME=https://example.test/a=b"], "benchmark_storage")
    assert [(v.name, v.purpose, v.value) for v in values] == [
        ("NAME", "benchmark_storage", "https://example.test/a=b")
    ]


def test_per_value_scope_overrides_only_its_own_entry() -> None:
    values = run_episode._parse_infra(
        [
            "AWS_REGION=us-east-1",
            "benchmark_user_simulator:OSWORLD_USER_SIM_MODEL=test/model",
        ],
        "benchmark_compute",
    )
    assert [(v.name, v.purpose, v.value) for v in values] == [
        ("AWS_REGION", "benchmark_compute", "us-east-1"),
        ("OSWORLD_USER_SIM_MODEL", "benchmark_user_simulator", "test/model"),
    ]


def test_identical_scoped_and_legacy_duplicates_coalesce() -> None:
    values = run_episode._parse_infra(
        ["NAME=value", "benchmark_compute:NAME=value"], "benchmark_compute"
    )
    assert len(values) == 1


@pytest.mark.parametrize(
    "items, purpose",
    [
        (["unknown:NAME=secret-canary"], "benchmark_compute"),
        (["=secret-canary"], "benchmark_compute"),
        (["benchmark_compute:=secret-canary"], "benchmark_compute"),
        ([":NAME=secret-canary"], "benchmark_compute"),
        (["secret-canary"], "benchmark_compute"),
        (["NAME="], "benchmark_compute"),
        (["NAME=secret-canary"], "unknown"),
        (["NAME=one", "benchmark_compute:NAME=secret-canary"], "benchmark_compute"),
        (
            ["NAME=secret-canary", "benchmark_storage:NAME=secret-canary"],
            "benchmark_compute",
        ),
    ],
)
def test_invalid_or_conflicting_entries_never_echo_input(items: list[str], purpose: str) -> None:
    with pytest.raises(ValueError) as error:
        run_episode._parse_infra(items, purpose)
    assert "secret-canary" not in str(error.value)


@pytest.mark.parametrize("prefix", ["", "benchmark_compute:"])
def test_scope_normalization_cannot_bypass_policy_disclosure(prefix: str) -> None:
    assert run_episode._infra_disclosure_metadata([f"{prefix}OSWORLD_ENABLE_PROXY=false"]) == {
        "osworld_enable_proxy_override": "false",
        "osworld_action_settle_policy": "throughput",
        "osworld_action_settle_seconds": 3.0,
    }
    assert run_episode._infra_disclosure_metadata(
        [
            "benchmark_compute:OSWORLD_ENABLE_PROXY=false",
            "benchmark_compute:AWS_ROOT_VOLUME_SIZE=80",
        ],
        "benchmark_user_simulator",
    ) == {
        "osworld_enable_proxy_override": "false",
        "aws_root_volume_size_override": "80",
        "osworld_action_settle_policy": "throughput",
        "osworld_action_settle_seconds": 3.0,
    }


def test_missing_settle_policy_is_pinned_for_episode_and_manifest() -> None:
    parsed = run_episode._parse_infra(["AWS_REGION=us-east-1"], "benchmark_compute")
    normalized = run_episode._ensure_action_settle_policy(parsed, "benchmark_compute")

    assert [
        (value.name, value.value) for value in normalized if value.name.endswith("SETTLE_POLICY")
    ] == [("OSWORLD_ACTION_SETTLE_POLICY", "throughput")]
    assert run_episode._infra_disclosure_metadata(normalized) == {
        "osworld_action_settle_policy": "throughput",
        "osworld_action_settle_seconds": 3.0,
    }
    assert run_episode._infra_disclosure_metadata(normalized)[
        "osworld_action_settle_policy"
    ] == next(value.value for value in normalized if value.name == "OSWORLD_ACTION_SETTLE_POLICY")


def test_invalid_settle_policy_fails_during_preflight_normalization() -> None:
    parsed = run_episode._parse_infra(["OSWORLD_ACTION_SETTLE_POLICY=fast"], "benchmark_compute")
    with pytest.raises(ValueError, match="unsupported OSWORLD_ACTION_SETTLE_POLICY"):
        run_episode._effective_action_settle_policy(parsed)


def test_invalid_settle_policy_exits_preflight_from_the_real_run_path(capsys, tmp_path) -> None:
    """The policy is resolved INSIDE the guarded preflight, not after it.

    ``_effective_action_settle_policy`` used to be called from ``build_config``'s
    argument list, outside the one handler that turns bad ``--infra`` input into
    ``EXIT_PREFLIGHT`` (that ``try`` catches ``VolatileRootError`` alone), so
    ``--infra OSWORLD_ACTION_SETTLE_POLICY=fast`` left ``run`` as an uncaught
    ``ValueError`` traceback raised by an ordinary command-line flag. Pinning the
    helper's raise cannot see that -- it passes while the CLI dies -- so this
    drives the real parser and the real ``run`` and asserts the exit status, the
    rendered message and the absence of a traceback.
    """

    import asyncio

    args = run_episode.build_parser().parse_args(
        [
            "--selector",
            str(tmp_path / "selector.json"),
            "--task-id",
            "synthetic",
            "--route",
            "test/model",
            "--run-root",
            str(tmp_path / "must-not-exist"),
            "--infra",
            "OSWORLD_ACTION_SETTLE_POLICY=fast",
            "--no-store",
        ]
    )

    assert asyncio.run(run_episode.run(args)) == run_episode.EXIT_PREFLIGHT

    output = capsys.readouterr()
    assert output.out == ""
    assert "unsupported OSWORLD_ACTION_SETTLE_POLICY" in output.err
    assert "Traceback" not in output.err
    # Refused before anything was created, like the other preflight refusals:
    # this is a bad flag, not a run that got as far as minting a run root.
    assert not (tmp_path / "must-not-exist").exists()


def test_paper_settle_delay_metadata_is_derived_not_an_override() -> None:
    parsed = run_episode._parse_infra(["OSWORLD_ACTION_SETTLE_POLICY=paper"], "benchmark_compute")
    assert run_episode._infra_disclosure_metadata(parsed) == {
        "osworld_action_settle_policy": "paper",
        "osworld_action_settle_seconds": 3.0,
    }


def test_actual_cli_rejects_invalid_infra_before_reading_selector(capsys, tmp_path) -> None:
    status = run_episode.main(
        [
            "--selector",
            str(tmp_path / "nonexistent.json"),
            "--task-id",
            "synthetic",
            "--route",
            "test/model",
            "--run-root",
            str(tmp_path / "must-not-exist"),
            "--infra",
            "unknown:NAME=secret-canary",
            "--no-store",
        ]
    )
    output = capsys.readouterr()
    assert status == run_episode.EXIT_PREFLIGHT
    assert output.out == ""
    assert output.err.startswith("--infra expects")
    assert "secret-canary" not in output.err and "Traceback" not in output.err
    assert not (tmp_path / "must-not-exist").exists()


def test_the_cloud_lease_is_derived_to_outlast_the_wall_budget() -> None:
    """A raised wall must not be reclaimed by a shorter fixed lease.

    The wall budget is not carried on the adapter wire, so the provider's
    ``ttl_seconds_for`` sees ``None`` and falls back to a fixed 7200 s. While
    the wall default was 1800 that was harmless. Raising it to 18000 -- so the
    500-step budget can actually be reached -- inverted the relationship, and
    an episode past two hours would die on a TERMINATED INSTANCE rather than
    at a budget boundary: it loses the episode instead of ending it, and reads
    as an infrastructure fault rather than a deliberate cap.
    """

    import argparse

    args = argparse.Namespace(max_wall_s=18000, infra_purpose="benchmark_compute")
    values = run_episode._ensure_lease_outlasts_wall((), args)

    lease = [v for v in values if v.name == "OSWORLD_TTL_SECONDS"]
    assert len(lease) == 1
    assert int(lease[0].value) > args.max_wall_s


def test_an_explicit_lease_override_is_never_overwritten() -> None:
    """The operator's own ``OSWORLD_TTL_SECONDS`` wins, however short."""

    import argparse

    args = argparse.Namespace(max_wall_s=18000, infra_purpose="benchmark_compute")
    given = run_episode._parse_infra(["OSWORLD_TTL_SECONDS=99"], "benchmark_compute")

    values = run_episode._ensure_lease_outlasts_wall(given, args)

    assert [v.value for v in values if v.name == "OSWORLD_TTL_SECONDS"] == ["99"]


def test_settle_policy_validation_is_bound_to_the_preflight_before_allocation() -> None:
    """An unknown policy is refused by the PREFLIGHT, before anything exists.

    The adapter validates the settle policy too, but an adapter is reached only
    once a worker has been spawned and a resource may exist, so its refusal is a
    late failure. The runner-side check is the one that has to happen for every
    adapter build -- including one that does not check the value at all -- and it
    has to happen where the other ``--infra`` refusals happen: in the guarded
    preflight, exiting as ``EXIT_PREFLIGHT`` instead of as a traceback. That exit
    is driven end to end by
    ``test_invalid_settle_policy_exits_preflight_from_the_real_run_path``; this
    test pins the value contract and the PLACEMENT of the call that enforces it.

    Deliberately runner-side only. Reaching into ``lop_osworld_v2_adapter`` from
    this file made it fail whenever it ran without the osworld package's
    conftest, which is what puts the adapter source tree on ``sys.path``: a test
    that only passes in the presence of another suite is a defect in the test,
    not evidence about the runner.
    """

    parsed = run_episode._parse_infra(["OSWORLD_ACTION_SETTLE_POLICY=unknown"], "benchmark_compute")
    with pytest.raises(ValueError, match="unsupported OSWORLD_ACTION_SETTLE_POLICY"):
        run_episode._effective_action_settle_policy(parsed)

    # Placement, because "validated somewhere" is not the contract: it must be
    # resolved BEFORE a config or a spec exists, i.e. before the selector is read
    # and long before any allocation (which happens inside a spawned worker).
    # Reading the source is the honest check here -- driving run() past this point
    # needs a selector, credentials and a cloud provider.
    import inspect

    source = inspect.getsource(run_episode.run)
    assert "_effective_action_settle_policy(" in source, (
        "run() must resolve the settle policy itself; without this call an unknown "
        "value reaches the adapter, where the refusal is late"
    )
    assert source.index("_effective_action_settle_policy(") < source.index(
        "build_config("
    ), "the policy must be resolved in the guarded preflight, before build_config"


def test_the_run_path_actually_applies_the_lease_derivation() -> None:
    """The DERIVATION is wired into ``run``, not merely defined beside it.

    A round-3 review deleted the sole call site and the entire 1311-test suite
    still passed: the helper's logic was pinned, its USE was not, so a future
    refactor could silently restore the 7200 s fixed lease this work removed --
    the bug where an episode past two hours dies on a terminated instance
    rather than at a budget boundary.

    Reading the source is the honest check here. Driving ``run`` end to end
    would need a selector, an adapter, credentials and a cloud provider, and a
    test that heavy would be skipped in exactly the environments that matter.
    """

    import inspect

    source = inspect.getsource(run_episode.run)

    assert "_ensure_lease_outlasts_wall(" in source, (
        "run() must derive the cloud lease from the wall budget; without this "
        "call the provider falls back to a fixed 7200 s lease that is SHORTER "
        "than the 18000 s wall default"
    )
    # It must happen before the spec is built, or the adapter never sees it.
    assert source.index("_ensure_lease_outlasts_wall(") < source.index(
        "build_spec("
    ), "the lease must be derived before build_spec() consumes infra_values"
