"""Requirement derivation: the executable spec of DESIGN.md's §3.2 table.

Each test asserts the EXACT requirement set a task produces, because the whole
point is that requirements are derived from the task, never hardcoded. The
table is: always-on AWS creds + account infra + client password + file base
URL; then conditional on the task's own fields.
"""

from __future__ import annotations

from lop_osworld_v2_adapter import requirements, taskfile

from tests.unit.evaluation.adapters.osworld import fixtures

_ALWAYS = {
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_REGION",
    "AWS_SUBNET_ID",
    "AWS_SECURITY_GROUP_ID",
    "AWS_SCHEDULER_ROLE_ARN",
    "OSWORLD_CLIENT_PASSWORD",
    "OSWORLD_FILE_BASE_URL",
    "HF_TOKEN",  # optional at episode time: the corpus is pre-materialised
}


def _names(source: str) -> set[str]:
    descriptor = taskfile.load_static(source.encode(), module_name="tasks/t.py")
    return {req.name for req in requirements.derive_requirements(descriptor)}


def _by_name(source: str) -> dict[str, tuple[str, bool]]:
    descriptor = taskfile.load_static(source.encode(), module_name="tasks/t.py")
    return {
        req.name: (req.kind, req.required) for req in requirements.derive_requirements(descriptor)
    }


def test_plain_task_has_only_the_always_on_set() -> None:
    assert _names(fixtures.PLAIN) == _ALWAYS


def test_proxy_task_adds_proxy_requirements() -> None:
    names = _names(fixtures.PROXY)
    assert {"OSWORLD_PROXY_CREDENTIALS", "OSWORLD_PROXY_ENDPOINT"} <= names


def test_gitlab_task_adds_gitlab_requirements() -> None:
    names = _names(fixtures.GITLAB)
    assert {"GITLAB_PRIVATE_TOKEN", "GITLAB_URL"} <= names


def test_website_task_adds_website_host_suffix() -> None:
    assert "WEBSITE_HOST_SUFFIX" in _names(fixtures.WEBSITE)


def test_googledrive_task_adds_google_credentials() -> None:
    assert "GOOGLE_ACCOUNT_CREDENTIALS" in _names(fixtures.GOOGLEDRIVE)


def test_llm_simulator_adds_api_key() -> None:
    assert "OSWORLD_USER_SIM_API_KEY" in _names(fixtures.LLM_SIMULATOR)


def test_scripted_simulator_needs_no_api_key() -> None:
    assert "OSWORLD_USER_SIM_API_KEY" not in _names(fixtures.SCRIPTED_SIMULATOR)


def test_clock_task_adds_optional_task_date() -> None:
    by_name = _by_name(fixtures.CLOCK)
    assert "OSWORLD_TASK_DATE" in by_name
    assert by_name["OSWORLD_TASK_DATE"] == ("infra", False)  # optional


def test_aws_credentials_are_secrets_not_infra() -> None:
    by_name = _by_name(fixtures.PLAIN)
    assert by_name["AWS_ACCESS_KEY_ID"] == ("secret", True)
    assert by_name["AWS_SECRET_ACCESS_KEY"] == ("secret", True)
    # AWS_SCHEDULER_ROLE_ARN is REQUIRED: without it OSWorld's TTL degrades to
    # a logged warning, removing the last defence against an orphaned instance.
    assert by_name["AWS_SCHEDULER_ROLE_ARN"] == ("infra", True)


def test_hf_token_is_optional_at_episode_time() -> None:
    by_name = _by_name(fixtures.PLAIN)
    assert by_name["HF_TOKEN"] == ("secret", False)


def test_derivation_is_deterministic() -> None:
    descriptor = taskfile.load_static(fixtures.GITLAB.encode(), module_name="tasks/t.py")
    a = requirements.derive_requirements(descriptor)
    b = requirements.derive_requirements(descriptor)
    assert [r.name for r in a] == [r.name for r in b]
