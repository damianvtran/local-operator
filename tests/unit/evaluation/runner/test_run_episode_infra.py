"""Mixed infrastructure purposes reuse the closed adapter value contract."""

import pytest

from scripts import run_episode


def test_legacy_global_scope_and_value_bytes_are_preserved() -> None:
    values = run_episode._parse_infra(
        ["NAME=https://example.test/a=b"], "benchmark_storage"
    )
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
def test_invalid_or_conflicting_entries_never_echo_input(
    items: list[str], purpose: str
) -> None:
    with pytest.raises(ValueError) as error:
        run_episode._parse_infra(items, purpose)
    assert "secret-canary" not in str(error.value)


@pytest.mark.parametrize("prefix", ["", "benchmark_compute:"])
def test_scope_normalization_cannot_bypass_policy_disclosure(prefix: str) -> None:
    assert run_episode._infra_disclosure_metadata(
        [f"{prefix}OSWORLD_ENABLE_PROXY=false"]
    ) == {"osworld_enable_proxy_override": "false"}
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


def test_actual_cli_rejects_invalid_infra_before_reading_selector(
    capsys, tmp_path
) -> None:
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
