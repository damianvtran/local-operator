"""Profile summaries are not file diffs; redact spans, retain the proposal."""

import pytest

from local_operator.harness.loop import AgentLoop
from local_operator.redaction_shapes import REDACTION_MARKER, scrub_secrets


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", ["agent", "team", "skill", "profile_proposal"])
async def test_profile_summary_is_not_withheld_as_malformed_hunk(tool):
    rows = ["instructions (replaced; new text):", "Sell carefully.", "Retain public guidance."]
    result = await AgentLoop._redact_details({"diff": rows}, scrub_secrets, tool, {})
    assert result["diff"] == rows


@pytest.mark.asyncio
async def test_profile_summary_masks_only_secret_spans():
    # Synthetic known value, never an actual credential; using a real masking
    # callback verifies complete-text semantics without a keychain or live store.
    secret = "synthetic" + "-credential-for-this-test"
    rows = [
        "instructions (replaced; new text):",
        "Public before.",
        f"token={secret}",
        "Public after.",
    ]
    result = await AgentLoop._redact_details(
        {"diff": rows},
        lambda text: text.replace(secret, REDACTION_MARKER),
        "team",
        {},
    )
    assert secret not in "\n".join(result["diff"])
    assert "Public before." in result["diff"]
    assert "Public after." in result["diff"]
    assert not any("withheld" in row for row in result["diff"])
    assert secret in "\n".join(rows)  # original executor evidence is untouched


@pytest.mark.asyncio
async def test_profile_full_new_body_redacts_multiline_credentials_together():
    block = "BEGIN SYNTHETIC CREDENTIAL\nbody-one\nbody-two\nEND SYNTHETIC CREDENTIAL"
    rows = [
        "instructions (replaced; new text):",
        "Public before.",
        *block.splitlines(),
        "Public after.",
    ]
    result = await AgentLoop._redact_details(
        {"diff": rows},
        lambda text: text.replace(block, REDACTION_MARKER),
        "agent",
        {},
    )
    assert "body-one" not in result["diff"] and "body-two" not in result["diff"]
    assert result["diff"] == [
        "instructions (replaced; new text):",
        "Public before.",
        REDACTION_MARKER,
        "Public after.",
    ]
