"""The tool-approval gate's two accepted shapes, and the arity rule.

``harness/approval.ask_approval`` is the single place that decides how to put
an approval question to a host. Both call sites (the loop's tier gate and the
builtins' self-gate) go through it, so the rule it implements has to hold for
every host shape in the tree — including the ones that predate the third
argument and will never be updated.

Both ways of getting it wrong are SILENT, which is why the cases below are
worth their length: guessing narrow drops the provenance and the host can
never scope its denial, and guessing wide raises ``TypeError`` inside the
host, which both call sites catch and turn into a refusal nobody sees.
"""

from __future__ import annotations

import functools
from typing import Any

import pytest

from local_operator.harness.approval import ask_approval


@pytest.mark.asyncio
async def test_a_two_argument_host_is_asked_with_two_arguments() -> None:
    """The original shape. Every host in the tree was written against it, and
    calling one with three arguments is a TypeError the call sites swallow into
    a silent denial — the tool refuses and nobody is ever asked."""
    seen: list[tuple[Any, ...]] = []

    async def gate(tool_name: str, description: str) -> bool:
        seen.append((tool_name, description))
        return True

    assert await ask_approval(gate, "bash", "rm -rf /tmp/x", "job-9") is True
    assert seen == [("bash", "rm -rf /tmp/x")]


@pytest.mark.asyncio
async def test_a_third_parameter_named_job_id_receives_it() -> None:
    """The point of the widening: the host learns which background job the ask
    belongs to, so it can scope a denial to that work instead of to every
    request that follows."""
    seen: list[tuple[Any, ...]] = []

    async def gate(tool_name: str, description: str, job_id: str | None) -> bool:
        seen.append((tool_name, description, job_id))
        return False

    assert await ask_approval(gate, "write", "edit foo.py", "job-9") is False
    assert seen == [("write", "edit foo.py", "job-9")]


@pytest.mark.asyncio
async def test_a_keyword_only_job_id_receives_it() -> None:
    """The natural way to add provenance to an existing handler without
    breaking its callers — and it has only TWO positional parameters, so a rule
    that counted them would call this host with two and leave its ``job_id``
    ``None`` forever, with no error to notice."""
    seen: list[str | None] = []

    async def gate(tool_name: str, description: str, *, job_id: str | None = None) -> bool:
        seen.append(job_id)
        return True

    await ask_approval(gate, "write", "edit foo.py", "job-4")
    assert seen == ["job-4"]


@pytest.mark.asyncio
async def test_a_positional_only_job_id_receives_it() -> None:
    """``(..., job_id, /)`` cannot take the keyword, so the resolver has to
    distinguish it rather than passing everything by name."""
    seen: list[str | None] = []

    async def gate(tool_name: str, description: str, job_id: str | None, /) -> bool:
        seen.append(job_id)
        return True

    await ask_approval(gate, "write", "edit foo.py", "job-5")
    assert seen == ["job-5"]


@pytest.mark.asyncio
async def test_the_foreground_asks_with_no_job_id() -> None:
    """``None`` is a real answer and not a missing one: it says the ask came
    from the session's own turn, which is what a host distinguishes against."""
    seen: list[str | None] = []

    async def gate(tool_name: str, description: str, job_id: str | None) -> bool:
        seen.append(job_id)
        return True

    await ask_approval(gate, "write", "edit foo.py")
    assert seen == [None]


@pytest.mark.asyncio
async def test_a_third_parameter_that_is_not_the_job_id_is_left_alone() -> None:
    """A pre-existing handler whose third parameter means something else. A
    rule that counted parameters would hand this host a job id as its timeout —
    no error, just a wrong value it goes on to use."""
    seen: list[tuple[Any, ...]] = []

    async def gate(tool_name: str, description: str, timeout: int = 30) -> bool:
        seen.append((tool_name, description, timeout))
        return True

    await ask_approval(gate, "bash", "ls", "job-9")
    assert seen == [("bash", "ls", 30)]


@pytest.mark.asyncio
async def test_a_star_args_passthrough_is_asked_narrow() -> None:
    """The regression this rule exists for. A hand-rolled wrapper forwarding to
    a two-argument gate is the shape this codebase writes, and treating its
    ``*args`` as wide calls the INNER gate with three arguments: TypeError,
    swallowed by both call sites into a denial with no prompt shown. Nothing
    about ``*args`` says the thing behind it wants a job id."""
    inner: list[tuple[Any, ...]] = []

    async def two_argument_host(tool_name: str, description: str) -> bool:
        inner.append((tool_name, description))
        return True

    async def passthrough(*args) -> bool:
        return await two_argument_host(*args)

    assert await ask_approval(passthrough, "bash", "ls", "job-9") is True
    assert inner == [("bash", "ls")]


@pytest.mark.asyncio
async def test_a_star_args_wrapper_that_names_job_id_still_gets_it() -> None:
    """...and a wrapper that genuinely wants provenance says so, which is the
    whole point of keying on the name: the narrow default is not a ceiling."""
    seen: list[str | None] = []

    async def wrapper(*args, job_id: str | None = None) -> bool:
        seen.append(job_id)
        return True

    await ask_approval(wrapper, "bash", "ls", "job-9")
    assert seen == ["job-9"]


@pytest.mark.asyncio
async def test_a_bound_method_host_is_measured_without_its_self() -> None:
    """The TUI installs ``self.request_tool_approval``. ``inspect.signature``
    drops the bound ``self``, and getting that wrong would misjudge the arity
    of the one host shape that matters most in production."""

    class Host:
        def __init__(self) -> None:
            self.asks: list[str | None] = []

        async def request_tool_approval(
            self, tool_name: str, description: str, job_id: str | None = None
        ) -> bool:
            self.asks.append(job_id)
            return True

    host = Host()
    await ask_approval(host.request_tool_approval, "write", "edit", "job-7")
    assert host.asks == ["job-7"]


@pytest.mark.asyncio
async def test_a_partial_bound_gate_is_measured_on_what_is_left() -> None:
    """A ``partial`` that has already supplied an argument leaves a different
    signature behind. Measuring the underlying function instead of the partial
    would misread which parameters are still free."""
    calls: list[tuple[Any, ...]] = []

    async def gate(host: str, tool_name: str, description: str) -> bool:
        calls.append((host, tool_name, description))
        return True

    bound = functools.partial(gate, "tui")
    assert await ask_approval(bound, "bash", "ls", "job-1") is True
    assert calls == [("tui", "bash", "ls")]


@pytest.mark.asyncio
async def test_an_unreadable_signature_degrades_to_the_two_argument_shape() -> None:
    """A callable whose signature cannot be read at all must take the shape
    that ALWAYS works. Guessing the wide one would raise TypeError inside the
    gate, and both call sites turn that into a denial nobody sees."""
    calls: list[tuple[Any, ...]] = []

    class Opaque:
        """Callable, and deliberately unintrospectable."""

        @property
        def __signature__(self):
            raise ValueError("no signature for you")

        async def __call__(self, tool_name: str, description: str) -> bool:
            calls.append((tool_name, description))
            return True

    assert await ask_approval(Opaque(), "bash", "ls", "job-1") is True
    assert calls == [("bash", "ls")]


@pytest.mark.asyncio
async def test_a_truthy_non_bool_answer_is_normalized() -> None:
    """The gate's contract is a bool. A host returning something merely truthy
    must not leak that object into the loop's ``if not approved`` branch."""

    async def gate(tool_name: str, description: str) -> Any:
        return "yes"

    result = await ask_approval(gate, "bash", "ls")
    assert result is True
