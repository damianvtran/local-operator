"""No code can reach an operator as an owner's raw diagnostic.

Audit round 2 (F1) plus review round 1 (R2). The rule these cells pin is the one the
whole copy module exists for: **every** refusal a person reads is composed from THIS
build's table, and an owner's own words are detail, never the sentence. The first
version of this file pinned a hand-maintained ``SENTENCED_CODES`` set instead — a second
list that had to be kept in step with the chain in :func:`render_broker_error`, which is
exactly the drift the module's own docstring forbids. The set is gone (R4): the chain is
the single source, and these cells assert the property directly for every code.
"""

from __future__ import annotations

from local_operator.network.credentials.messages import _generic, render_broker_error
from local_operator.network.credentials.types import BROKER_ERROR_CODES, BrokerError

OWNER = "d_" + "a" * 32
DIAGNOSTIC = "a diagnostic only a log should read"


def _error(code: str, message: str = DIAGNOSTIC) -> BrokerError:
    return BrokerError(
        code=code, key="zai", owner_device=OWNER, owner_device_name="damian-mbp", message=message
    )


def test_no_code_renders_the_owner_s_words_as_the_sentence() -> None:
    """Every code in the closed set gets a sentence of this build's own."""
    for code in sorted(BROKER_ERROR_CODES):
        rendered = render_broker_error(
            _error(code), key="zai", owner_name="damian-mbp", provider="zai"
        )
        assert rendered != DIAGNOSTIC, f"{code} handed the operator the raw diagnostic"
        assert "damian-mbp" in rendered, f"{code} does not name the owner: {rendered}"


def test_the_generic_arm_names_owner_credential_and_remedy() -> None:
    """The unknown-code arm is still a sentence, not an echo (R2's contract)."""
    error = _error("a_code_from_a_newer_owner")
    rendered = render_broker_error(error, key="zai", owner_name="damian-mbp", provider="zai")
    assert rendered.startswith("damian-mbp declined to lend 'zai'"), rendered
    assert DIAGNOSTIC in rendered, rendered
    assert "run 'lop login zai' here" in rendered, rendered
    assert rendered == _generic(error, label="zai", owner="damian-mbp", login="lop login zai")


def test_an_mcp_key_is_told_the_mcp_verb() -> None:
    """``mcp:<url>`` has no provider verb, so the MCP arms must say ``/mcp login``."""
    url = "https://srv.example/mcp"
    rendered = render_broker_error(
        _error("interactive_required", ""),
        key=f"mcp:{url}",
        owner_name="damian-mbp",
    )
    assert f"/mcp login {url}" in rendered, rendered
    assert "lop login" not in rendered, rendered
