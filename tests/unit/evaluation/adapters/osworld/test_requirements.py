"""Requirement derivation: the executable spec of DESIGN.md's §3.2 table.

Each test asserts the EXACT requirement set a task produces, because the whole
point is that requirements are derived from the task, never hardcoded. The
table is: always-on AWS creds + account infra + client password + file base
URL; then conditional on the task's own fields.
"""

from __future__ import annotations

import pytest
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
    "OSWORLD_INPUTS_ROOT",  # optional: the durable root the assets live in
    "OSWORLD_TTL_SECONDS",  # optional: lease-length override
}

_JUDGE = {
    "OSWORLD_EVAL_MODEL_API_KEY",
    "OSWORLD_EVAL_MODEL_PROVIDER",
    "OSWORLD_EVAL_MODEL_NAME",
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


def test_inputs_root_and_ttl_are_optional_infra() -> None:
    by_name = _by_name(fixtures.PLAIN)
    assert by_name["OSWORLD_INPUTS_ROOT"] == ("infra", False)
    assert by_name["OSWORLD_TTL_SECONDS"] == ("infra", False)


def test_judged_task_requires_the_judge_key_and_settings() -> None:
    """A task whose evaluator calls the LLM judge is REFUSED at preflight
    without the key: OSWorld's ``llm_metrics`` returns 0.0 on any exception,
    so running it without a key seals a silent zero."""

    by_name = _by_name(fixtures.JUDGED)
    assert by_name["OSWORLD_EVAL_MODEL_API_KEY"] == ("secret", True)
    assert by_name["OSWORLD_EVAL_MODEL_PROVIDER"] == ("infra", True)
    assert by_name["OSWORLD_EVAL_MODEL_NAME"] == ("infra", True)
    assert _names(fixtures.JUDGED) == _ALWAYS | _JUDGE


def test_judged_via_llm_metrics_import_is_also_detected() -> None:
    assert _JUDGE <= _names(fixtures.JUDGED_VIA_METRICS)


def test_judged_via_metrics_reexport_is_detected() -> None:
    """MAJOR-2: task_007 reaches the judge as ``metrics.compare_text_with_llm``
    with no judge-module substring in its source. Detection is by symbol."""

    assert _JUDGE <= _names(fixtures.JUDGED_VIA_REEXPORT)
    assert _JUDGE <= _names(fixtures.JUDGED_VIA_BARE_NAME)


def test_a_non_judge_metric_from_the_same_package_is_not_judged() -> None:
    assert not (_JUDGE & _names(fixtures.METRICS_NOT_JUDGED))


def test_judge_detection_over_the_pinned_corpus_matches_the_call_surface() -> None:
    """Every task in the real corpus that references a judge entry point is
    detected, and none that does not. The truth set is a plain text scan for
    the judge's call surface, independent of the AST walk under test."""

    import glob
    import os
    import pwd
    import re
    from pathlib import Path

    real_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    root = Path(os.environ.get("OSWORLD_INPUTS_ROOT", real_home / "worktrees" / "osworld"))
    tasks = sorted(glob.glob(str(root / "gated" / "tasks" / "task_*.py")))
    if len(tasks) != 108:  # pragma: no cover - inputs root absent on CI
        pytest.skip("pinned OSWorld corpus not present")
    truth_re = re.compile(r"generate_text|generate_json|llm_metrics|model_client|_with_llm")
    detected: set[str] = set()
    truth: set[str] = set()
    for path in tasks:
        source = Path(path).read_bytes()
        descriptor = taskfile.load_static(source, module_name=path)
        stem = Path(path).stem
        if requirements.is_judged(descriptor):
            detected.add(stem)
        if truth_re.search(source.decode()):
            truth.add(stem)
    assert detected == truth
    assert len(detected) == 19
    assert "task_007" in detected


def test_plain_task_does_not_require_the_judge() -> None:
    assert not (_JUDGE & _names(fixtures.PLAIN))
    descriptor = taskfile.load_static(fixtures.PLAIN.encode(), module_name="tasks/t.py")
    assert requirements.is_judged(descriptor) is False


def test_hf_token_is_optional_at_episode_time() -> None:
    by_name = _by_name(fixtures.PLAIN)
    assert by_name["HF_TOKEN"] == ("secret", False)


def test_derivation_is_deterministic() -> None:
    descriptor = taskfile.load_static(fixtures.GITLAB.encode(), module_name="tasks/t.py")
    a = requirements.derive_requirements(descriptor)
    b = requirements.derive_requirements(descriptor)
    assert [r.name for r in a] == [r.name for r in b]
