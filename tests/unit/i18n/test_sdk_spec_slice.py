"""SDK input guards: real paths, byte compatibility, and exact owned-site coverage.

Goldens came from the unkeyed source through these same constructors/public SDK
calls, not from the new catalogue. Extraction does not authorize translation or
change approval policy. Filesystem and runtime/isolation mechanics are excluded.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
import textwrap
from pathlib import Path
from typing import Any, cast

import pytest

from local_operator import sdk
from local_operator.i18n import catalogues, messages, resolve
from local_operator.i18n.keys import wire_errors_sdk_spec
from local_operator.session.spec import ApprovalPolicy, SessionRoots, SessionSpec, SessionSpecError

_EXPECTED = json.loads(
    (Path(__file__).parent / "fixtures" / "sdk_spec_en_golden.json").read_text(encoding="utf-8")
)
_OWNED = {
    ApprovalPolicy.__post_init__: [
        "approval_declared_tools_empty",
        "approval_tools_wrong_mode",
        "approval_callback_gate_required",
        "approval_handler_wrong_mode",
    ],
    SessionSpec.__post_init__: [
        "agent_identity_conflict",
        "resume_empty",
        "tools_empty",
        "yolo_approval_unreachable",
        "output_format_invalid",
        "output_schema_requires_format",
        "output_retries_requires_format",
        "output_retries_range",
        "world_root_empty",
    ],
    sdk.open_session: ["mode_invalid", "attach_resume_required"],
    sdk._refuse_attach_extras: ["attach_output_unsupported", "attach_fields_unsupported"],
    sdk._refuse_spawn_extras: ["spawn_output_unsupported", "spawn_fields_unsupported"],
    sdk.spawn_session: ["spawn_model_pair_required", "spawn_birth_effort_pair_required"],
}


def _cases(tmp_path):
    roots = SessionRoots(
        config_dir=tmp_path / "config",
        agent_home=tmp_path / "agent",
        cwd=tmp_path / "world",
        allow_volatile=True,
    )

    async def open_invalid(mode, spec):
        async with sdk.open_session(spec, roots=roots, mode=mode):
            pytest.fail("invalid input yielded a session")

    async def spawn_invalid(spec):
        await sdk.spawn_session(spec, roots=roots, errand=cast(Any, None))

    return {
        "agent_identity_conflict": lambda: SessionSpec(agent_name="a", agent_id="b"),
        "resume_empty": lambda: SessionSpec(resume=""),
        "tools_empty": lambda: SessionSpec(tools=()),
        "yolo_approval_unreachable": lambda: SessionSpec(yolo=True),
        "output_format_invalid": lambda: SessionSpec(output_format=cast(Any, "invalid")),
        "output_schema_requires_format": lambda: SessionSpec(output_schema={}),
        "output_retries_requires_format": lambda: SessionSpec(output_retries=1),
        "output_retries_range": lambda: SessionSpec(output_format="json", output_retries=True),
        "world_root_empty": lambda: SessionSpec(world_root=" "),
        "approval_declared_tools_empty": lambda: ApprovalPolicy.declared(()),
        "approval_tools_wrong_mode": lambda: ApprovalPolicy(mode="refuse", tools=("read",)),
        "approval_callback_gate_required": lambda: ApprovalPolicy.callback(cast(Any, None)),
        "approval_handler_wrong_mode": lambda: ApprovalPolicy(
            mode="auto", handler=cast(Any, lambda *a: True)
        ),
        "attach_output_unsupported": lambda: sdk._refuse_attach_extras(
            SessionSpec(output_format="json")
        ),
        "attach_fields_unsupported": lambda: sdk._refuse_attach_extras(
            SessionSpec(name="opaque-'Ω-{name}-%(x)s")
        ),
        "spawn_output_unsupported": lambda: sdk._refuse_spawn_extras(
            SessionSpec(output_format="json")
        ),
        "spawn_fields_unsupported": lambda: sdk._refuse_spawn_extras(
            SessionSpec(name="opaque-'Ω-{name}-%(x)s")
        ),
        "mode_invalid": lambda: open_invalid("opaque-'Ω-{mode}-%(x)s", SessionSpec()),
        "attach_resume_required": lambda: open_invalid("attach", SessionSpec()),
        "spawn_model_pair_required": lambda: spawn_invalid(SessionSpec(hosting="test")),
        "spawn_birth_effort_pair_required": lambda: spawn_invalid(SessionSpec(birth_effort="high")),
    }


def _refuse(call):
    with pytest.raises(SessionSpecError) as caught:
        result = call()
        if inspect.iscoroutine(result):
            asyncio.run(result)
    error = caught.value
    assert type(error) is SessionSpecError
    assert len(error.args) == 1 and type(error.args[0]) is str
    assert not hasattr(error, "code") and not hasattr(error, "params")
    return error


@pytest.mark.parametrize("case", sorted(_EXPECTED))
def test_real_guards_preserve_english_bytes_and_missing_namespace_fallback(
    case, tmp_path, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("source-only spec diagnostics must never resolve configured language")

    monkeypatch.setattr(resolve, "resolve_language", forbidden)
    original = messages.render
    calls = []

    def observe(code, params, *, locale=None):
        assert locale == "en"
        calls.append((code, params))
        return original(code, params, locale=locale)

    monkeypatch.setattr(messages, "render", observe)
    error = _refuse(_cases(tmp_path)[case])
    assert str(error).encode() == _EXPECTED[case].encode()
    assert error.args == (_EXPECTED[case],)
    assert len(calls) == 1
    code, params = calls[0]
    assert code == "wire.errors.sdk_spec." + case
    # A carried locale missing this child must use English source formatting,
    # not reinterpret repr quotes or opaque parameters as catalogue syntax.
    assert original(code, params, locale="fr") == _EXPECTED[case]


@pytest.mark.parametrize("case", sorted(_EXPECTED))
def test_real_guards_consume_the_keyed_render_not_a_parallel_literal(case, tmp_path, monkeypatch):
    def marker(code, params, *, locale=None):
        assert locale == "en"
        assert code == "wire.errors.sdk_spec." + case
        return "source-marker:" + code

    monkeypatch.setattr(messages, "render", marker)
    error = _refuse(_cases(tmp_path)[case])
    assert error.args == ("source-marker:wire.errors.sdk_spec." + case,)


def _owned_raise_keys(source):
    tree = ast.parse(source)
    found = []
    for node in sorted(ast.walk(tree), key=lambda node: getattr(node, "lineno", 0)):
        if not isinstance(node, ast.Raise) or not isinstance(node.exc, ast.Call):
            continue
        if not isinstance(node.exc.func, ast.Name) or node.exc.func.id != "SessionSpecError":
            continue
        keyed = [
            call.func.attr
            for call in ast.walk(node.exc)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "wire_errors_sdk_spec"
        ]
        found.append(keyed)
    return found


def test_exact_owned_scope_has_21_keyed_raises_and_rejects_planted_literal():
    observed = []
    for owner, expected in _OWNED.items():
        source = textwrap.dedent(inspect.getsource(owner))
        actual = _owned_raise_keys(source)
        assert actual == [[key] for key in expected]
        observed.extend(key for keys in actual for key in keys)
        # Negative control uses the same predicate as the exact-site guard.
        planted = source + '\nraise SessionSpecError("planted unkeyed SDK diagnostic")\n'
        assert _owned_raise_keys(planted)[-1] == []
    assert len(observed) == len(set(observed)) == 21
    assert set(observed) == set(_EXPECTED)


def test_source_catalogue_and_context_cover_only_the_owned_21_keys():
    namespace = "wire.errors.sdk_spec"
    source = catalogues.load_catalogue("en", namespace)
    context = catalogues.context("en", namespace)
    expected = {namespace + "." + case for case in _EXPECTED}
    assert set(source) == set(context) == expected
    assert all(entry["role"] and "Placeholders:" in entry["note"] for entry in context.values())
    for key in expected:
        assert callable(getattr(wire_errors_sdk_spec, key.rsplit(".", 1)[1]))
    unknown = namespace + ".unknown_key"
    assert messages.render(unknown, {}, locale="en") == unknown
    missing = "wire.errors.unknown_namespace.unknown_key"
    assert messages.render(missing, {}, locale="fr") == missing
