"""Session code memory (``complete_session_variables``) against the REAL worker.

The property under test is not "a dict round-trips through JSON" — it is that a
desktop panel reading a session's code memory sees exactly what the cells of
that session can see, and nothing the harness owns. So each case here spawns the
real worker the ``eval`` tool spawns (no fake kernel), mutates the real
namespace through a real cell, and reads it through the verb the desktop route
calls.

Four rules from the frozen design are easy to "simplify" away and are pinned
explicitly because losing any of them is silent:

* a READ must not move ``last_used`` (or looking at the panel keeps an idle
  interpreter alive forever);
* a timeout must NOT retire the kernel (retiring on a five-second deadline
  destroys user state that was never lost);
* an absent kernel is reported ABSENT rather than as an empty namespace (the
  panel's copy differs, and "Nothing stored yet" over an interpreter that never
  started is a lie);
* a namespace mutated mid-walk is a RETRYABLE refusal, never a half-old list.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from local_operator.harness.types import ToolContext
from local_operator.tools import eval as eval_tool

SESSION = "eval-variables"


@pytest_asyncio.fixture(autouse=True)
async def _clean_kernel_registry():
    """Isolate the module-level registry and kill every worker this file spawns."""
    eval_tool._KERNELS.clear()
    eval_tool._LOST_KERNELS.clear()
    eval_tool._ACTIVE_KERNELS.clear()
    eval_tool._CLOSE_ON_RETURN.clear()
    yield
    for kernel in list(eval_tool._KERNELS.values()):
        await eval_tool._close_kernel(kernel)
    eval_tool._KERNELS.clear()
    for task in list(eval_tool._CLOSING):
        task.cancel()


@pytest.fixture
def context(tmp_path) -> ToolContext:
    return ToolContext(cwd=str(tmp_path), session_id=SESSION)


async def _cell(context: ToolContext, code: str) -> str:
    """Run one real cell and return its model-visible text."""
    tool = eval_tool.build_eval_tool()
    result = await tool.execute("call-1", {"code": code}, None, None, context)
    assert result.is_error is False, result.text
    return result.text


async def _verb(session_id: str, action: str, key: str = "", value: str = "", value_type: str = ""):
    return await eval_tool.complete_session_variables(
        session_id, action, key=key, value=value, value_type=value_type
    )


def _keys(answer: dict) -> list[str]:
    return [entry["key"] for entry in answer["variables"]]


def _entry(answer: dict, key: str) -> dict:
    return next(entry for entry in answer["variables"] if entry["key"] == key)


# ---------------------------------------------------------------------------
# what counts as code memory
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_variable_is_listed_with_its_runtime_type(context) -> None:
    await _cell(context, "outstanding = 3\ntotal_outstanding = 3.5\ndf_like = [1, 2]")
    answer = await _verb(SESSION, "list")

    assert answer["ok"] is True
    assert answer["state"] == "observed" and answer["kernel"] == "resident"
    assert answer["truncated"] is False
    assert _entry(answer, "outstanding")["type"] == "int"
    assert _entry(answer, "outstanding")["value"] == "3"
    assert _entry(answer, "total_outstanding")["type"] == "float"
    assert _entry(answer, "df_like")["type"] == "list"


@pytest.mark.asyncio
async def test_harness_bindings_imports_and_dunders_are_not_code_memory(context) -> None:
    """A module is an import, and ``display``/``tool``/``secrets`` are ours.

    Run AFTER a cell on purpose: ``display`` and ``tool`` only exist once the
    worker has served one request, so a test that listed a fresh namespace would
    prove nothing about excluding them.
    """
    await _cell(context, "import math\nanswer = 42")
    answer = await _verb(SESSION, "list")

    assert _keys(answer) == ["answer"], _keys(answer)


@pytest.mark.asyncio
async def test_a_function_is_code_memory(context) -> None:
    """Helpers are exactly what the panel is for — it must not read as empty."""
    await _cell(context, "def double(x):\n    return x * 2")
    answer = await _verb(SESSION, "list")

    assert "double" in _keys(answer)
    assert _entry(answer, "double")["type"] == "function"


# ---------------------------------------------------------------------------
# writes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("value_type", "submitted", "rendered", "type_name"),
    [
        ("str", "hello", "'hello'", "str"),
        ("int", "42", "42", "int"),
        ("float", "4.25", "4.25", "float"),
        ("bool", "yes", "True", "bool"),
        ("list", "[1, 2, 3]", "[1, 2, 3]", "list"),
        ("dict", '{"a": 1}', "{'a': 1}", "dict"),
    ],
)
async def test_each_type_round_trips_into_the_next_cell(
    context, value_type, submitted, rendered, type_name
) -> None:
    """The coercion table is only real if a CELL sees the object, not the text."""
    await _cell(context, "seed = 0")
    answer = await _verb(SESSION, "set", "chosen", submitted, value_type)

    assert answer["ok"] is True, answer
    assert answer["variable"]["type"] == type_name
    assert answer["variable"]["value"] == rendered
    assert answer["variable"]["editable"] is True
    # The tool appends its stdout/stderr summary to every result, so the
    # trailing-expression line is what this asserts on.
    assert (await _cell(context, "chosen")).startswith(f"result: {rendered}")


@pytest.mark.asyncio
async def test_create_over_an_existing_key_is_refused_and_update_missing_too(context) -> None:
    await _cell(context, "seed = 0")
    await _verb(SESSION, "set", "kept", "1", "int")

    duplicate = await _verb(SESSION, "set", "kept", "2", "int")
    assert duplicate == {
        "ok": False,
        "code": "already_exists",
        "message": "A variable with that name already exists.",
    }
    missing = await _verb(SESSION, "update", "absent", "2", "int")
    assert missing["ok"] is False and missing["code"] == "not_found"
    # Refusals leave the stored value alone.
    assert _entry(await _verb(SESSION, "list"), "kept")["value"] == "1"


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["secrets", "display", "tool", "__builtins__", "__mine__"])
async def test_a_reserved_or_dunder_name_is_refused(context, name) -> None:
    await _cell(context, "x = 1")

    answer = await _verb(SESSION, "set", name, "1", "int")

    assert answer["ok"] is False
    assert answer["code"] == "reserved_name"
    # The interpreter keeps its own binding.
    assert name not in _keys(await _verb(SESSION, "list"))


@pytest.mark.asyncio
async def test_a_failed_coercion_names_the_type_and_never_the_value(context) -> None:
    await _cell(context, "seed = 0")
    answer = await _verb(SESSION, "set", "n", "not-a-number", "int")

    assert answer["ok"] is False and answer["code"] == "invalid_value"
    assert "not-a-number" not in answer["message"]


@pytest.mark.asyncio
async def test_an_oversized_value_is_refused_and_never_stored(context) -> None:
    await _cell(context, "seed = 0")
    answer = await _verb(SESSION, "set", "big", "z" * 4097, "str")

    assert answer["ok"] is False and answer["code"] == "too_large"
    assert "big" not in _keys(await _verb(SESSION, "list"))


@pytest.mark.asyncio
async def test_delete_removes_the_variable_from_the_namespace(context) -> None:
    await _cell(context, "seed = 0")
    await _verb(SESSION, "set", "gone", "1", "int")

    assert await _verb(SESSION, "delete", "gone") == {"ok": True, "state": "ok"}
    assert "gone" not in _keys(await _verb(SESSION, "list"))
    assert (await _cell(context, "'gone' in globals()")).startswith("result: False")
    assert (await _verb(SESSION, "delete", "gone"))["code"] == "not_found"


# ---------------------------------------------------------------------------
# the kernel-ownership rules
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_absent_kernel_reports_absent_rather_than_an_empty_namespace(context) -> None:
    """No kernel is a DIFFERENT fact from an empty one, and never a spawn."""
    assert SESSION not in eval_tool._KERNELS
    answer = await _verb(SESSION, "list")

    assert answer == {
        "ok": True,
        "state": "observed",
        "kernel": "absent",
        "variables": [],
        "truncated": False,
    }
    assert SESSION not in eval_tool._KERNELS, "a read must never spawn a kernel"
    # A write has nowhere to go, and says so instead of starting an interpreter.
    refused = await _verb(SESSION, "set", "k", "1", "int")
    assert refused["ok"] is False and refused["code"] == "no_kernel"
    assert SESSION not in eval_tool._KERNELS


@pytest.mark.asyncio
async def test_an_absent_kernel_after_a_close_reports_absent(context) -> None:
    await _cell(context, "x = 1")
    await eval_tool.close_session_kernel(SESSION)

    answer = await _verb(SESSION, "list")
    assert answer["kernel"] == "absent" and answer["variables"] == []


@pytest.mark.asyncio
async def test_a_read_does_not_extend_the_interpreter_lease_and_a_write_does(context) -> None:
    await _cell(context, "x = 1")
    kernel = eval_tool._KERNELS[SESSION]
    kernel.last_used = 0.0

    await _verb(SESSION, "list")
    assert eval_tool._KERNELS[SESSION].last_used == 0.0, (
        "reading the panel extended the kernel's lease; a session left open on "
        "the variables view would keep an idle interpreter alive forever"
    )

    await _verb(SESSION, "set", "y", "1", "int")
    assert eval_tool._KERNELS[SESSION].last_used > 0.0


@pytest.mark.asyncio
async def test_the_busy_guard_refuses_rather_than_sharing_a_live_pipe(context) -> None:
    """A cell in flight owns the interpreter: no second request may be written."""
    await _cell(context, "x = 1")
    eval_tool._ACTIVE_KERNELS.add(SESSION)
    try:
        assert await _verb(SESSION, "list") == {"ok": True, "state": "busy"}
        refused = await _verb(SESSION, "set", "y", "1", "int")
        assert refused["ok"] is False and refused["code"] == "kernel_busy"
    finally:
        eval_tool._ACTIVE_KERNELS.discard(SESSION)
    # And the kernel is untouched by the refusals.
    assert _entry(await _verb(SESSION, "list"), "x")["value"] == "1"


@pytest.mark.asyncio
async def test_a_slow_read_reports_busy_and_LEAVES_THE_KERNEL_RESIDENT(
    context, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Timing out must not destroy the namespace it was too slow to read.

    A late answer is safe by construction: ``_exchange`` skips a response whose
    id does not match, so the next verb skips the straggler and still works —
    which this asserts rather than assumes.
    """
    await _cell(
        context,
        "import time\n\nclass Slow:\n"
        "    def __repr__(self):\n        time.sleep(0.4)\n        return 'slow'\n\n"
        "sleepy = Slow()",
    )
    monkeypatch.setattr(eval_tool, "_SESSION_VARIABLES_TIMEOUT_S", 0.05)

    assert await _verb(SESSION, "list") == {"ok": True, "state": "busy"}
    assert (
        SESSION in eval_tool._KERNELS
    ), "the kernel was retired on a read timeout; that silently loses user state"

    monkeypatch.setattr(eval_tool, "_SESSION_VARIABLES_TIMEOUT_S", 5.0)
    after = await _verb(SESSION, "list")
    assert after["kernel"] == "resident"
    assert "sleepy" in _keys(after)


@pytest.mark.asyncio
async def test_a_namespace_mutated_mid_walk_is_a_retryable_refusal(context) -> None:
    """A half-old, half-new listing must never be answered as a snapshot.

    The mutation is performed BY the walk (a value's ``__repr__`` writes to its
    own globals), which is exactly the shape a background thread the cell
    started produces — and it is deterministic rather than a race to be hoped
    for.
    """
    await _cell(
        context,
        "class Sneaky:\n"
        "    def __repr__(self):\n"
        "        globals()['surprise'] = 1\n"
        "        return 'sneaky'\n\n"
        "sneaky = Sneaky()",
    )

    first = await _verb(SESSION, "list")
    assert first["ok"] is False and first["code"] == "changed_under_read"
    # Retryable, and the retry succeeds: the mutation is idempotent, so the
    # second walk sees a stable namespace.
    second = await _verb(SESSION, "list")
    assert second["ok"] is True
    assert "sneaky" in _keys(second) and "surprise" in _keys(second)


# ---------------------------------------------------------------------------
# secrets
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_value_the_cell_retrieved_from_the_secret_ledger_is_redacted(context) -> None:
    """The panel reads the namespace; it must not become a secret exfiltration path.

    ``_RedactionLedger.register`` is what ``SecretsMapping.__getitem__`` calls on
    every retrieval, so registering here is the same registration a real
    ``secrets["NAME"]`` performs — without an encrypted store, which the worker
    has no access to in a unit test.
    """
    await _cell(
        context,
        "from local_operator.secrets import runtime as _runtime\n"
        "_runtime._LEDGER.register('tok-abcdef-123456')\n"
        "leaked = 'tok-abcdef-123456'",
    )

    answer = await _verb(SESSION, "list")
    rendered = _entry(answer, "leaked")["value"]
    assert "tok-abcdef-123456" not in rendered
    assert "[redacted]" in rendered
    # The ECHO on a write is the same road out of the process.
    echo = await _verb(SESSION, "set", "again", "tok-abcdef-123456", "str")
    assert "tok-abcdef-123456" not in echo["variable"]["value"]
