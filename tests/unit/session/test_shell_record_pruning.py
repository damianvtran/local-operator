"""A bang-mode receipt must stay inert for compaction's pruning passes.

``shell_record_messages`` builds its tool row through ``Message.tool_result``,
which carries ``details``/``useless``/``duration_s`` on ``provider_payload``.
Unlike the viewer's display-only rows, a receipt lands in the owner's
``Session._context.messages`` and in the durable transcript, so it is a real
pruning INPUT — the pass reads exactly those keys (``_is_useless`` on
``useless``, ``_supersede_key`` on ``details['path']``/``details['range']``/
``details['supersede_key']``).

It is inert today only because ``execute_bash`` sets none of them: its sole
detail key is ``spill``, which no pruning rule reads. That is a property of
another module, invisible from here, and a future ``details={"path": ...}``
there would make two runs of one command supersede each other — silently
blanking output the operator ran by hand and can no longer see. These tests
pin the guarantee where it is consumed rather than trusting that property to
hold, and drive the REAL producer so a fixture cannot pass in its place.
"""

from __future__ import annotations

import os

import pytest

from local_operator.compaction.pruning import (
    MIN_PRUNE_TOKENS,
    _is_useless,
    _supersede_key,
    prune_tool_outputs,
)
from local_operator.compaction.tokens import estimate_tokens
from local_operator.harness.types import AbortSignal, ToolContext
from local_operator.session.shell_record import shell_record_messages
from local_operator.tools import builtin

NOW = 10_000_000
ACTIVE = NOW  # not idle, so the warm-cache guard is the only one relaxed


@pytest.fixture(autouse=True)
def _no_inherited_cmux(monkeypatch: pytest.MonkeyPatch) -> None:
    """A test that runs a real tool must not inherit the operator's session."""
    for key in [k for k in os.environ if k.startswith("CMUX_")]:
        monkeypatch.delenv(key, raising=False)


async def _receipt(call_id: str, command: str, cwd: str):
    """One real bang-mode receipt: execute_bash, then the synthetic exchange."""
    result = await builtin.execute_bash(
        call_id, {"command": command}, AbortSignal(), None, ToolContext(cwd=cwd)
    )
    return result, shell_record_messages(command, result)[-1]


@pytest.mark.asyncio
async def test_bang_receipt_carries_no_pruning_key(tmp_path) -> None:
    """The payload exists, and none of its keys is one the prune pass acts on.

    Asserted against a SPILLING command, because that is the only shape whose
    ``details`` is non-empty today (``{'spill': ...}``) — a small ``echo``
    writes no payload at all and would make this vacuous.
    """
    result, tool = await _receipt(
        "spill-1",
        "for i in $(seq 1 4000); do echo line-$i-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa; done",
        str(tmp_path),
    )
    payload = tool.provider_payload or {}
    assert payload, "a spilling receipt should carry a payload; the probe is otherwise vacuous"

    details = payload.get("details") or {}
    assert details, "expected the spill detail; without it the supersede assertions are vacuous"
    # The exact keys pruning reads. `spill` is fine; `path`/`range`/
    # `supersede_key` are not, because two runs of one command would then
    # blank each other's output.
    assert not (set(details) & {"path", "range", "supersede_key", "url", "useless"}), (
        f"execute_bash grew a detail key the prune pass acts on: {sorted(details)}. "
        "A bang receipt is the operator's own output and must not be superseded "
        "or useless-blanked; see this module's docstring."
    )
    assert _supersede_key(tool) is None, "a bang receipt became supersede-eligible"
    assert _is_useless(tool) is False, "a bang receipt became useless-eligible"
    assert result.useless is False


@pytest.mark.asyncio
async def test_repeated_bang_receipts_are_never_blanked(tmp_path) -> None:
    """End of the chain: the real pass leaves two identical receipts intact.

    This is the behaviour the keys above protect, asserted on the output the
    operator would actually lose. Both rows clear ``MIN_PRUNE_TOKENS``, so a
    pass that wanted to blank them is not merely being stopped by the floor.
    """
    command = "for i in $(seq 1 4000); do echo line-$i-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa; done"
    _, first = await _receipt("spill-1", command, str(tmp_path))
    _, second = await _receipt("spill-2", command, str(tmp_path))

    for row in (first, second):
        assert estimate_tokens(row) >= MIN_PRUNE_TOKENS, (
            "receipt is below the prune floor, so this test would pass without "
            "the guarantee it claims to pin"
        )

    before = [first.text, second.text]
    out, changed = prune_tool_outputs([first, second], NOW, ACTIVE)
    after = [m.text for m in out if m.role == "tool"]

    assert changed is False, "compaction blanked a bang-mode receipt"
    assert after == before, "a bang receipt's output was rewritten by the prune pass"
