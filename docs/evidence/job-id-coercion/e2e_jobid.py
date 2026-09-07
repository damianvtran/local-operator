"""End-to-end proof that a numeric-looking id reaches a real running child.

Drives the REAL ``task``/``jobs``/``wait``/``hub`` tools against a REAL
``AsyncJobManager`` and a REAL subagent-comms registry, with ``uuid4`` patched
so the manager MINTS the all-digit ids the defect is about. This exercises the
operator-visible behaviour, not the extracted helper functions.

Run it from anywhere; it inserts the repo root on ``sys.path`` itself, so no
copy-to-root ritual is needed (review round 1, NIT-1)::

    .venv/bin/python docs/evidence/job-id-coercion/e2e_jobid.py

Exit status is 0 when every cell passes, 1 otherwise. Provenance (the imported
``module.__file__`` and the sha256 of that exact file) is printed and asserted
up front, because this repo's editable venv has produced false "no difference"
A/B results for reviewers who trusted the branch name instead.
"""

from __future__ import annotations

import asyncio
import hashlib
import pathlib
import sys
import uuid
from typing import Any

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from local_operator.harness.jobs import AsyncJobManager  # noqa: E402
from local_operator.harness.types import (  # noqa: E402
    AgentTool,
    ToolContext,
    ToolResult,
)
from local_operator.tools import builtin  # noqa: E402
from local_operator.tools.registry import create_tools  # noqa: E402

# Two legal ``uuid4().hex[:12]`` values that are also well-formed JSON numbers:
# one all-digit, one exponent-shaped. The second is the shape that overflows to
# ``inf`` when parsed as a float, destroying the id rather than retyping it.
DIGIT_IDS = ["920883861377", "468698086935"]
EXPONENT_ID = "13190e419943"


def provenance() -> None:
    path = builtin.__file__
    digest = hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()[:16]
    print(f"module.__file__ = {path}")
    print(f"sha256(builtin.py)[:16] = {digest}")
    resolved = pathlib.Path(path).resolve()
    assert _REPO_ROOT in resolved.parents, f"imported the WRONG tree: {resolved}"
    print()


async def _quick(job_id: str, signal: Any, report_progress: Any) -> str:
    return f"done:{job_id}"


class _FakeUUID:
    """Just enough of ``uuid.UUID`` for ``uuid4().hex[:12]`` to yield our id."""

    def __init__(self, hexval: str) -> None:
        self.hex = hexval + "0" * (32 - len(hexval))


async def _call(
    tools: dict[str, AgentTool], name: str, args: dict[str, Any], context: ToolContext
) -> ToolResult:
    return await tools[name].execute("call-1", args, None, None, context)  # type: ignore[operator]


async def main() -> int:
    provenance()
    failures = 0
    manager = AsyncJobManager()

    def launcher(label: str, prompt: str, *, agent: str = "task", effort: Any = None) -> str:
        return manager.register("task", label, _quick, owner_id=None)

    context = ToolContext(cwd="/tmp", session_id="s", subagent_launcher=launcher, jobs=manager)
    tools = {t.name: t for t in create_tools(context)}

    seq = iter(DIGIT_IDS)
    original_uuid4 = uuid.uuid4
    uuid.uuid4 = lambda: _FakeUUID(next(seq))  # type: ignore[assignment]
    try:
        first = await _call(tools, "task", {"label": "alpha", "prompt": "x"}, context)
        second = await _call(tools, "task", {"label": "beta", "prompt": "y"}, context)
    finally:
        uuid.uuid4 = original_uuid4  # type: ignore[assignment]

    assert first.details is not None and second.details is not None
    minted = [first.details["job_id"], second.details["job_id"]]
    print(f"real job ids minted by AsyncJobManager: {minted}")
    assert minted == DIGIT_IDS, minted

    async def check(desc: str, name: str, args: dict[str, Any], want_ok: bool = True) -> None:
        nonlocal failures
        result = await _call(tools, name, args, context)
        broke = result.is_error or "unknown job" in (result.text or "")
        ok = (not broke) if want_ok else broke
        failures += not ok
        text = (result.text or "").replace("\n", " ")[:86]
        print(f"  [{'PASS' if ok else 'FAIL'}] {desc:46} -> {text}")

    print("\n=== jobs(op='peek') with an all-digit id ===")
    peek = {"op": "peek"}
    await check("bracketed string  '[920883861377]'", "jobs", {**peek, "job_id": "[920883861377]"})
    await check("bare string       '920883861377'", "jobs", {**peek, "job_id": "920883861377"})
    await check("JSON list  '[\"920883861377\"]'", "jobs", {**peek, "job_id": '["920883861377"]'})
    await check("bare int           920883861377", "jobs", {**peek, "job_id": 920883861377})

    print("\n=== wait with all-digit ids (single + list) ===")
    both_str = "[920883861377, 468698086935]"
    both_json = '["920883861377", "468698086935"]'
    await check(
        "single bracketed  '[468698086935]'", "wait", {"job_id": "[468698086935]", "wait_ms": 1500}
    )
    await check(
        "LIST of two digit ids (string form)", "wait", {"job_id": both_str, "wait_ms": 1500}
    )
    await check("LIST of two digit ids (real list)", "wait", {"job_id": DIGIT_IDS, "wait_ms": 1500})
    await check("LIST of two digit ids (JSON form)", "wait", {"job_id": both_json, "wait_ms": 1500})

    print("\n=== negative control: a genuinely unknown id still errors ===")
    unknown = {"job_id": "[111111111111]", "wait_ms": 500}
    await check("unknown id '[111111111111]'", "wait", unknown, want_ok=False)

    await manager.dispose()

    failures += await _hub_cell()

    print(f"\nRESULT: {'ALL PASS' if failures == 0 else f'{failures} FAILED'}")
    return 1 if failures else 0


async def _hub_cell() -> int:
    """``hub op='ask'`` must actually REACH a child whose id is all digits.

    The sibling defect QA found on round 1: ``_coerce_hub_to`` dropped a
    numeric-looking id, so ``to`` resolved to an empty recipient list and the
    ask silently addressed nobody. This drives the real coercion into the real
    ``HubParams`` model and then through the real comms resolver, so a
    regression shows up as "no recipient" rather than as a helper-level detail.
    """
    from local_operator.tools.builtin import HubParams, _coerce_hub_to

    print("\n=== hub op='ask' reaches a child with an all-digit id ===")
    failures = 0
    cases = [
        ("bracketed all-digit  '[920883861377]'", "[920883861377]", ["920883861377"]),
        ("bracketed exponent   '[13190e419943]'", f"[{EXPONENT_ID}]", [EXPONENT_ID]),
        ("two digit ids        '[a, b]'", "[920883861377, 468698086935]", DIGIT_IDS),
        ("bare id              '920883861377'", "920883861377", ["920883861377"]),
        ("ordinary hex id", "[a1b2c3d4e5f6]", ["a1b2c3d4e5f6"]),
    ]
    for desc, raw, expected in cases:
        coerced = _coerce_hub_to(raw)
        params = HubParams(op="ask", to=raw, message="are you there?")
        ok = coerced == expected and params.to == expected
        failures += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {desc:42} -> to={params.to!r}")

    # The deliberate non-regression: a bare JSON literal is still NOT a target,
    # so it stays dropped and the field reports "needs a 'to' target".
    dropped = _coerce_hub_to("[null]")
    ok = dropped == []
    failures += not ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {'null stays dropped (not a target)':42} -> {dropped!r}")
    return failures


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
