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
        "osworld_enable_proxy_override": "false"
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
