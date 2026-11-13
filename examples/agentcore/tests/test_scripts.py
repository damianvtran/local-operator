"""deploy.sh / teardown.sh argument and error handling, against a fake ``aws``.

The scripts' AWS calls are the part a unit test cannot prove (the README
transcripts and the PR's live evidence cover those). What is testable offline is
how they REACT to the CLI: a credential failure or an unreadable stack must
stop teardown with a non-zero exit, never print "not found" and claim success,
and only CloudFormation's "does not exist" means a stack is absent. The fake
``aws`` below records every call and answers from environment variables, so
each test states the CLI behaviour it simulates.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

FAKE_AWS = r"""#!/usr/bin/env bash
# Strip the global --profile/--region pair the scripts always pass.
args=()
while [[ $# -gt 0 ]]; do
  case "$1" in --profile|--region) shift 2 ;; *) args+=("$1"); shift ;; esac
done
echo "${args[*]}" >> "$FAKE_AWS_LOG"
case "${args[0]} ${args[1]}" in
  "configure get") echo "us-east-1" ;;
  "sts get-caller-identity")
    if [[ -n "${FAKE_STS_ERROR:-}" ]]; then echo "$FAKE_STS_ERROR" >&2; exit 254; fi
    echo "arn:aws:iam::000000000000:user/fake" ;;
  "cloudformation describe-stacks")
    if [[ -n "${FAKE_DESCRIBE_ERROR:-}" ]]; then echo "$FAKE_DESCRIBE_ERROR" >&2; exit 254; fi
    case "${args[*]}" in
      *AgentRuntimeId*) echo "demo_rt-AbC123" ;;
      *BuildProjectName*) echo "demo-image" ;;
      *SourceBucketName*) echo "demo-bucket" ;;
    esac ;;
  "logs describe-log-groups") printf '/aws/bedrock-agentcore/runtimes/demo_rt-AbC123-DEFAULT\n' ;;
  "logs delete-log-group")
    if [[ "${args[*]}" == *codebuild* ]]; then
      echo "An error occurred (ResourceNotFoundException) when calling DeleteLogGroup" >&2; exit 254
    fi ;;
esac
exit 0
"""

NOT_FOUND = (
    "An error occurred (ValidationError) when calling the DescribeStacks operation: "
    "Stack with id demo-runtime does not exist"
)


@pytest.fixture
def fake_aws(tmp_path: Path) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    aws = bin_dir / "aws"
    aws.write_text(FAKE_AWS)
    aws.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("AWS_", "CMUX_"))}
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["FAKE_AWS_LOG"] = str(tmp_path / "calls.log")
    return env


def run(script: str, env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPTS / script), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def calls(env: dict[str, str]) -> list[str]:
    path = Path(env["FAKE_AWS_LOG"])
    return path.read_text().splitlines() if path.exists() else []


def test_teardown_stops_on_bad_credentials_before_touching_stacks(fake_aws: dict[str, str]) -> None:
    """Q1: an invalid key used to print "not found; skipping" and exit 0."""
    fake_aws["FAKE_STS_ERROR"] = "An error occurred (InvalidClientTokenId) ... security token"
    result = run("teardown.sh", fake_aws, "--profile", "p", "--stack-name", "demo")
    assert result.returncode != 0
    assert "InvalidClientTokenId" in result.stderr
    assert "Teardown complete" not in result.stdout
    assert not any(c.startswith("cloudformation") for c in calls(fake_aws))


def test_teardown_surfaces_a_describe_error_that_is_not_absence(fake_aws: dict[str, str]) -> None:
    fake_aws["FAKE_DESCRIBE_ERROR"] = "An error occurred (AccessDenied) when calling DescribeStacks"
    result = run("teardown.sh", fake_aws, "--profile", "p", "--stack-name", "demo")
    assert result.returncode == 1
    assert "cannot read stack demo-runtime" in result.stderr and "AccessDenied" in result.stderr
    assert "not found" not in result.stdout


def test_teardown_treats_only_does_not_exist_as_absent(fake_aws: dict[str, str]) -> None:
    fake_aws["FAKE_DESCRIBE_ERROR"] = NOT_FOUND
    result = run("teardown.sh", fake_aws, "--profile", "p", "--stack-name", "demo")
    assert result.returncode == 0, result.stderr
    assert "demo-runtime not found; skipping" in result.stdout
    assert "demo-build not found; skipping" in result.stdout
    assert not any("delete" in c for c in calls(fake_aws))


def test_teardown_deletes_stacks_then_only_this_deployments_log_groups(
    fake_aws: dict[str, str],
) -> None:
    """Q2: the service-created log groups go too, named from the stack outputs."""
    result = run("teardown.sh", fake_aws, "--profile", "p", "--stack-name", "demo")
    assert result.returncode == 0, result.stderr
    log = calls(fake_aws)
    deletes = [c for c in log if c.split()[1].startswith("delete-")]
    assert deletes == [
        "cloudformation delete-stack --stack-name demo-runtime",
        "cloudformation delete-stack --stack-name demo-build",
        "logs delete-log-group --log-group-name /aws/codebuild/demo-image",
        "logs delete-log-group --log-group-name "
        "/aws/bedrock-agentcore/runtimes/demo_rt-AbC123-DEFAULT",
    ]
    # The runtime prefix carries the runtime's unique id, never a broad pattern.
    assert (
        "logs describe-log-groups --log-group-name-prefix "
        "/aws/bedrock-agentcore/runtimes/demo_rt-AbC123- "
        "--query logGroups[].logGroupName --output text"
    ) in log
    assert "Log group /aws/codebuild/demo-image not found; skipping" in result.stdout
    assert (
        "Deleted log group /aws/bedrock-agentcore/runtimes/demo_rt-AbC123-DEFAULT" in result.stdout
    )


@pytest.mark.parametrize("script", ["deploy.sh", "teardown.sh"])
@pytest.mark.parametrize("argv", [["--profile"], ["--profile", "--region", "us-east-1"]])
def test_a_flag_without_its_value_is_a_usage_error(
    fake_aws: dict[str, str], script: str, argv: list[str]
) -> None:
    """Q4: used to exit 1 with no message from a bare `shift 2` under `set -e`."""
    result = run(script, fake_aws, *argv)
    assert result.returncode == 2
    assert "error: --profile needs a value" in result.stderr
    assert calls(fake_aws) == []
