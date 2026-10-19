"""Redaction: credential shapes are masked, ordinary prose survives."""

from __future__ import annotations

from local_operator.harness_import.redact import redact_text, redact_value

BEARER_SECRET = "syntheticBearer0123456789abcdef"
ENV_SECRET = "synthetic-value-123"
PEM_SECRET = "c3ludGhldGljLXNlY3JldC1ibG9jaw=="


def test_bearer_and_env_assignment_are_masked_without_values():
    bearer = f"Authorization: Bearer {BEARER_SECRET}"
    env = f"export SYNTHETIC_API_TOKEN={ENV_SECRET}"
    pem = (
        "-----BEGIN SYNTHETIC PRIVATE KEY-----\n"
        f"{PEM_SECRET}\n"
        "-----END SYNTHETIC PRIVATE KEY-----"
    )
    for text, secret in ((bearer, BEARER_SECRET), (env, ENV_SECRET), (pem, PEM_SECRET)):
        cleaned, count = redact_text(text)
        assert count >= 1
        assert secret not in cleaned


def test_ordinary_prose_is_not_masked():
    text = "The token budget is 100 and this sentence talks about a secret plan."
    cleaned, count = redact_text(text)
    assert cleaned == text
    assert count == 0


def test_tool_arguments_are_redacted_recursively():
    value = {"command": f"curl -H 'Authorization: Bearer {BEARER_SECRET}'"}
    cleaned, count = redact_value(value)
    assert count >= 1
    assert BEARER_SECRET not in str(cleaned)


def test_non_secret_env_names_survive():
    text = "MODE=production\nLOG_LEVEL=debug"
    cleaned, count = redact_text(text)
    assert cleaned == text
    assert count == 0


def test_credential_location_paths_survive():
    text = (
        "MINERVA_CREDENTIALS_DIR=/tmp/minerva/credentials\n"
        "GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcloud/application_default_credentials.json"
    )
    cleaned, count = redact_text(text)
    assert cleaned == text
    assert count == 0
