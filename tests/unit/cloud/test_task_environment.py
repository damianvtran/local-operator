"""The container environment: the names, the values, and the binding to the image.

The names are a contract between two programs in this repo — the provider that
writes them and the image that reads them — so both directions are checked:
every name the builder emits must still appear in the image file that reads it
(rename either side and this goes red), and the image's own extra ``POC_*`` names
must be the ones the image sets for ITSELF, never ones a caller is expected to
provide.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from local_operator.cloud.types import PoolRequest
from local_operator.cloud.workload import (
    CONDITIONAL_ENV_NAMES,
    TASK_ENV_NAMES,
    task_environment,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

#: The two files that read the container's environment.
IMAGE_FILES: tuple[str, ...] = (
    "infra/remote-agents-poc/image/entrypoint.sh",
    "infra/remote-agents-poc/image/lop_launch.py",
)

#: The ``POC_*`` names the image reads that a request does NOT drive, with the
#: reason each one is the image's business:
#:
#: * ``POC_KEY_FD`` / ``POC_KEY_LEN`` / ``POC_MODEL_KEY`` are the model key's
#:   delivery channel — the launcher exports them, reads the key off the inherited
#:   descriptor and UNSETS them, precisely so no caller and no task definition
#:   carries the secret;
#: * ``POC_ENVIRON_WATCH_SELFTEST`` / ``POC_ENVIRON_WATCH_COEXIST`` are the
#:   entrypoint's own self-test switches.
IMAGE_ONLY_ENV_NAMES: frozenset[str] = frozenset(
    {
        "POC_KEY_FD",
        "POC_KEY_LEN",
        "POC_MODEL_KEY",
        "POC_ENVIRON_WATCH_SELFTEST",
        "POC_ENVIRON_WATCH_COEXIST",
    }
)

#: The presigned URLs a submit hands the container, in the shape the fargate
#: suite's fake S3 produces. The signature is a marker: nothing is signed here.
_URL_BASE = "https://s3.ca-central-1.amazonaws.com/bucket/runs/req_1a2b3c4d"
PROBES_URL = f"{_URL_BASE}/probes.json?X-Amz-Signature=marker"
RESULTS_URL = f"{_URL_BASE}/results.tar.gz?X-Amz-Signature=marker"
SECRET_ARN = "arn:aws:secretsmanager:ca-central-1:325492156725:secret:lop-poc-model-abc"


def _request(**overrides: Any) -> PoolRequest:
    values: dict[str, Any] = {
        "request_id": "req_1a2b3c4d",
        "repo_url": "https://example.invalid/fixture.git",
        "repo_sha": "69db7e55fc14f918cccdf2fea62894fc37f1f642",
        "prompt": "make the failing test `test_add` pass",
        "hosting": "hosted",
        "model": "anthropic/claude-sonnet-4-5",
        "mock": True,
    }
    values.update(overrides)
    return PoolRequest(**values)


def _environment(request: PoolRequest) -> dict[str, str]:
    return {
        entry["name"]: entry["value"]
        for entry in task_environment(
            request,
            probes_url=PROBES_URL,
            results_url=RESULTS_URL,
            model_secret_arn=SECRET_ARN,
        )
    }


def _image_text() -> str:
    return "\n".join((REPO_ROOT / name).read_text(encoding="utf-8") for name in IMAGE_FILES)


def test_env_names_are_the_ones_the_image_reads() -> None:
    """The exact ten names, in the POC's order, and every one of them in the image."""
    environment = task_environment(
        _request(),
        probes_url=PROBES_URL,
        results_url=RESULTS_URL,
        model_secret_arn=SECRET_ARN,
    )
    names = [entry["name"] for entry in environment]
    assert names == list(TASK_ENV_NAMES)

    text = _image_text()
    unread = [name for name in names if name not in text]
    assert not unread, (
        f"the builder writes {unread}, which the image does not read: a rename on "
        "either side is a container that starts with an empty variable"
    )


def test_the_image_requires_no_name_this_builder_does_not_write() -> None:
    """The other direction: an eleventh required name must not appear unnoticed."""
    text = _image_text()
    image_names = {
        token
        for token in (
            "POC_MODEL_KEY",
            "POC_KEY_FD",
            "POC_KEY_LEN",
            "POC_ENVIRON_WATCH_SELFTEST",
            "POC_ENVIRON_WATCH_COEXIST",
            *TASK_ENV_NAMES,
        )
        if token in text
    }
    assert image_names == set(TASK_ENV_NAMES) | IMAGE_ONLY_ENV_NAMES


def test_the_values_are_the_requests_own() -> None:
    """Every value is the request's or the caller's — nothing is invented here."""
    request = _request()
    environment = _environment(request)
    assert environment["POC_RUN_ID"] == request.request_id
    assert environment["POC_REPO_URL"] == request.repo_url
    assert environment["POC_SHA"] == request.repo_sha
    assert environment["POC_PROMPT"] == request.prompt
    assert environment["POC_PROBES_URL"] == PROBES_URL
    assert environment["POC_RESULTS_URL"] == RESULTS_URL
    assert environment["POC_MODEL_SECRET_ARN"] == SECRET_ARN
    assert environment["POC_HOSTING"] == request.hosting
    assert environment["POC_MODEL"] == request.model


def test_the_run_id_is_the_request_id_so_the_branch_is_the_one_the_ledger_names() -> None:
    """``lop/<request_id>`` has to be the branch the container checks out.

    The entrypoint does ``git checkout -b "lop/$POC_RUN_ID"``; the ledger's
    ``ComputeTask.branch`` is ``lop/<request_id>``. Nothing else ties the two
    together, so this is the test that keeps them equal.
    """
    request = _request()
    assert _environment(request)["POC_RUN_ID"] == request.request_id


def test_mock_is_the_string_the_entrypoint_compares() -> None:
    """``[ "$POC_MOCK" = "1" ]``: a bool would serialise as ``True`` and never match."""
    assert _environment(_request(mock=True))["POC_MOCK"] == "1"
    assert _environment(_request(mock=False))["POC_MOCK"] == "0"


def test_the_conditional_names_are_omitted_when_the_request_has_none() -> None:
    """The entrypoint distinguishes "unset" from "set but empty"."""
    environment = _environment(_request(hosting="", model=""))
    assert CONDITIONAL_ENV_NAMES.isdisjoint(environment), (
        "an empty POC_MODEL/POC_HOSTING reads as a declared-but-empty value, which "
        "is not the same statement as an absent one"
    )
    assert set(TASK_ENV_NAMES) - CONDITIONAL_ENV_NAMES == set(environment)


def test_the_posture_leaves_an_empty_repo_url_alone() -> None:
    """No fixture default is applied here: that is the CLI's decision (§7.2)."""
    environment = _environment(_request(repo_url="", repo_sha=""))
    assert environment["POC_REPO_URL"] == ""
    assert environment["POC_SHA"] == ""
