"""The code-memory verb table and the two session shapes that reach it.

``session/variable_ops`` exists so one implementation serves both a session that
runs its own tools and a runtime handle that runs them for a viewer — the same
doctrine ``credential_ops`` states, and the reason a verb added for one shape is
a verb added for both. These tests pin the parts that only exist at that seam:

* the refusals a caller can make WITHOUT a kernel (reserved name, unusable name,
  oversized value) never reach the kernel — a panel typing into an idle session
  must not pay a round trip to be told a name is the interpreter's;
* the answer passes the session's ``VariableStore.redact`` on the way out, which
  is the second half of the secret policy (the first, ``_safe_repr`` →
  ``_scrub_secrets``, is the worker's and is covered against a real worker in
  ``tests/unit/tools/test_eval_variables.py``);
* both shapes forward the SESSION ID THEY OWN rather than anything a caller
  passed, so a viewer cannot address another conversation's namespace.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, get_args

import pytest

from local_operator.session.session import Session
from local_operator.session.variable_ops import (
    VARIABLE_TYPES,
    VariableType,
    run_variable_verb,
)


class _Recorder:
    """A completer that records its call and answers a canned envelope."""

    def __init__(self, answer: dict[str, Any] | None = None) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.answer = answer or {
            "ok": True,
            "state": "observed",
            "kernel": "resident",
            "variables": [],
            "truncated": False,
        }

    async def __call__(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append((args, kwargs))
        return self.answer


def test_the_type_vocabulary_is_derived_from_the_literal() -> None:
    """One source, so the body model and the coercion table cannot drift."""
    assert VARIABLE_TYPES == get_args(VariableType)
    assert set(VARIABLE_TYPES) == {"str", "int", "float", "bool", "list", "dict"}


@pytest.mark.asyncio
async def test_a_refused_name_never_reaches_the_kernel() -> None:
    recorder = _Recorder()
    for key, code in (
        ("", "invalid_value"),
        ("x" * 129, "invalid_value"),
        ("bad\x01name", "invalid_value"),
        # A ``/`` decodes into extra path segments before the router matches, so
        # no route can ever address the key: refused where the name policy is,
        # with a sentence that names the rule rather than the value.
        ("a/b", "invalid_value"),
        ("secrets", "reserved_name"),
        ("display", "reserved_name"),
        ("tool", "reserved_name"),
        ("__builtins__", "reserved_name"),
        ("__dunder__", "reserved_name"),
    ):
        answer = await run_variable_verb("s1", "set", key, "1", "int", complete=recorder)
        assert answer["ok"] is False
        assert answer["code"] == code, (key, answer)
        assert answer["message"], (key, answer)
    assert (
        recorder.calls == []
    ), "a refusal decided from the shared table must not cost a kernel round trip"


@pytest.mark.asyncio
async def test_the_unaddressable_name_sentence_names_the_rule() -> None:
    """One wording, in the shared table, for parent and worker alike.

    The worker used to answer its own "1-128 characters" bound for every
    ``invalid_value`` name, and the parent's pre-validation made it unreachable
    — two sentences for one rule, only one of them ever shown.
    """
    recorder = _Recorder()
    answer = await run_variable_verb("s1", "set", "a/b", "1", "int", complete=recorder)

    assert "path segment" in answer["message"]
    assert "a/b" not in answer["message"]
    assert recorder.calls == []


@pytest.mark.asyncio
async def test_an_oversized_value_never_reaches_the_kernel() -> None:
    recorder = _Recorder()
    answer = await run_variable_verb("s1", "set", "big", "z" * 4097, "str", complete=recorder)

    assert answer["ok"] is False and answer["code"] == "too_large"
    assert recorder.calls == []


@pytest.mark.asyncio
async def test_a_read_forwards_with_no_key_and_the_worker_owns_the_rest() -> None:
    """``list`` is not key-validated: there is no key to refuse."""
    recorder = _Recorder()
    await run_variable_verb("s1", "list", "secrets", complete=recorder)

    assert recorder.calls == [(("s1", "list"), {"key": "secrets", "value": "", "value_type": ""})]


@pytest.mark.asyncio
async def test_the_answer_passes_the_sessions_redactor_on_values_only() -> None:
    answer_with_secret = {
        "ok": True,
        "state": "observed",
        "kernel": "resident",
        "variables": [
            {
                "key": "token",
                "type": "str",
                "value": "'sk-live-4417'",
                "editable": True,
                "truncated": False,
            }
        ],
        "truncated": False,
    }
    recorder = _Recorder(answer_with_secret)
    answer = await run_variable_verb(
        "s1",
        "list",
        complete=recorder,
        redact=lambda text: text.replace("sk-live-4417", "[redacted]"),
    )

    assert answer["variables"][0]["value"] == "'[redacted]'"
    # The KEY is not redacted: it is the name the panel has to print.
    assert answer["variables"][0]["key"] == "token"


@pytest.mark.asyncio
async def test_the_write_echo_is_redacted_too() -> None:
    recorder = _Recorder(
        {
            "ok": True,
            "state": "ok",
            "variable": {
                "key": "token",
                "type": "str",
                "value": "'sk-live-4417'",
                "editable": True,
                "truncated": False,
            },
        }
    )
    answer = await run_variable_verb(
        "s1",
        "set",
        "token",
        "sk-live-4417",
        "str",
        complete=recorder,
        redact=lambda text: text.replace("sk-live-4417", "[redacted]"),
    )

    assert answer["variable"]["value"] == "'[redacted]'"


@pytest.mark.asyncio
async def test_an_unknown_action_is_a_programming_error_not_a_refusal() -> None:
    """Both callers validate it first, so this can only be a caller bug."""
    with pytest.raises(ValueError, match="unknown code-memory action"):
        await run_variable_verb("s1", "explode", complete=_Recorder())


@pytest.mark.asyncio
async def test_the_in_process_shape_forwards_its_own_session_and_store(monkeypatch) -> None:
    """``Session.variables_op`` names ITS id and ITS store, nothing from a caller.

    Called unbound with a stub ``self`` on purpose: the method reads exactly two
    attributes, and building a whole ``Session`` (transcript, providers, tool
    loop) to exercise two lines would make this test's cost about everything
    else. The kernel registry is the module global the real method reaches, so
    the id it forwards is the whole claim under test.
    """
    import local_operator.tools.eval as eval_module

    recorder = _Recorder()
    monkeypatch.setattr(eval_module, "complete_session_variables", recorder)
    store = SimpleNamespace(redact=lambda text: text.replace("shh", "[redacted]"))
    # ``Any`` for the same reason the docstring gives: this is not a Session,
    # and annotating it as one would have pyright check the ~110 members the
    # two lines under test never reach.
    stub: Any = SimpleNamespace(_session_id="8fd6c6a40934", _variables=store)

    answer = await Session.variables_op(stub, "list")

    assert recorder.calls[0][0][0] == "8fd6c6a40934"
    assert answer["state"] == "observed"


@pytest.mark.asyncio
async def test_the_runtime_shape_forwards_the_session_it_serves(monkeypatch) -> None:
    """Same claim across the socket: the frame carries no session identity."""
    import local_operator.tools.eval as eval_module
    from local_operator.session.runtime.serving import ServingSessionHandle

    recorder = _Recorder()
    monkeypatch.setattr(eval_module, "complete_session_variables", recorder)
    session = SimpleNamespace(
        session_id="000471114995",
        variables=SimpleNamespace(redact=lambda text: text),
    )

    handle: Any = SimpleNamespace(_session=session)
    await ServingSessionHandle.variables_op(handle, "list")

    assert recorder.calls[0][0][0] == "000471114995"
