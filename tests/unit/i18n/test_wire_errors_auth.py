"""Auth extraction pins the entire pre-extraction sentence and constructor status.

The golden was evaluated from all 14 constructors at base 4d5849698e; it
includes both provider-verdict branches. AST evaluation deliberately enumerates
returned/helper constructors too, without opening a credential store. TCP route
execution is separate evidence: this census cannot prove route admission.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from local_operator.server.routes import auth

ROOT = Path(__file__).resolve().parents[3]
PATH = "local_operator/server/routes/auth.py"
OPAQUE = "synthetic ' {tail} العربية 日本語"


def constructors(source: str | None = None):
    return sorted(
        (
            node
            for node in ast.walk(ast.parse((ROOT / PATH).read_text() if source is None else source))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "HTTPException"
        ),
        key=lambda node: node.lineno,
    )


# These are additions after the frozen extraction, not extra legacy sentences.
# Enumerate their owners and exact constructor inventory: filtering by namespace
# alone would silently hide a new, uncatalogued constructor in an old route.
BEDROCK_CONSTRUCTORS = {"_bedrock_refusal": 1, "bedrock_sso_login": 1}


def legacy_constructors(source: str | None = None):
    source = (ROOT / PATH).read_text() if source is None else source
    tree = ast.parse(source)
    excluded = set()
    for owner, expected in BEDROCK_CONSTRUCTORS.items():
        functions = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == owner
        ]
        assert len(functions) == 1
        calls = constructors(ast.unparse(functions[0]))
        assert len(calls) == expected, owner
        excluded.update(
            node.lineno
            for node in ast.walk(functions[0])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "HTTPException"
        )
    return [call for call in constructors(source) if call.lineno not in excluded]


def test_legacy_census_does_not_hide_new_constructors():
    source = (ROOT / PATH).read_text()
    assert len(legacy_constructors(source)) == 14
    assert (
        len(
            legacy_constructors(
                source + "\nraise HTTPException(422, w.input_invalid().envelope())\n"
            )
        )
        == 15
    )
    with pytest.raises(AssertionError, match="_bedrock_refusal"):
        legacy_constructors(
            source.replace(
                "return HTTPException(status, message.envelope(code=code))",
                "HTTPException(422, message.envelope(code=code))\n"
                "    return HTTPException(status, message.envelope(code=code))",
            )
        )


@pytest.mark.parametrize(
    "status,code,message",
    [
        (422, "bedrock_input_invalid", "input_invalid"),
        (500, "bedrock_context_failed", "context_failed"),
    ],
)
def test_bedrock_helper_envelopes_are_accounted_separately(status, code, message):
    from local_operator.i18n.keys import wire_errors_bedrock as keys

    sentence = (
        keys.input_invalid(field="region")
        if message == "input_invalid"
        else keys.context_failed(detail=OPAQUE)
    )
    refusal = auth._bedrock_refusal(status, code, sentence)
    assert refusal.status_code == status
    assert refusal.detail == sentence.envelope(code=code)
    invalid = auth._invalid_input(SimpleNamespace(code="bedrock_input_invalid", field="region"))
    assert invalid.status_code == 422
    assert invalid.detail == keys.input_invalid(field="region").envelope(
        code="bedrock_input_invalid"
    )


def test_bedrock_sso_service_refusal_is_not_a_legacy_constructor():
    tree = ast.parse((ROOT / PATH).read_text())
    route = next(
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "bedrock_sso_login"
    )
    (call,) = constructors(ast.unparse(route))
    assert isinstance(call.args[0], ast.Constant) and call.args[0].value == 503
    payload = eval(
        compile(ast.Expression(call.args[1]), "<bedrock-sso-refusal>", "eval"),
        {**vars(auth), "error": ValueError(OPAQUE)},
    )
    assert payload == {
        "code": "wire.errors.auth.auth_refusal",
        "params": {"error": OPAQUE},
        "text": OPAQUE,
    }


def test_full_sentence_status_and_params_match_base():
    golden = json.loads(
        (Path(__file__).parent / "fixtures/wire_errors_auth_en_golden.json").read_text()
    )
    calls = legacy_constructors()
    assert len(calls) == 14
    samples = []
    for call in calls:
        for reason in (
            [None, OPAQUE] if any(isinstance(n, ast.IfExp) for n in ast.walk(call)) else [None]
        ):
            body = eval(
                compile(ast.Expression(call.args[1]), "<auth-constructor>", "eval"),
                {
                    **vars(auth),
                    "error": ValueError(OPAQUE),
                    "verdict": SimpleNamespace(reason=reason),
                },
            )
            assert set(body) == {"code", "params", "text"}
            assert body["code"].startswith("wire.errors.auth.")
            expected_params = {"error": OPAQUE} if body["text"] == OPAQUE else {}
            assert body["params"] == expected_params
            status = call.args[0]
            assert isinstance(status, ast.Constant)
            samples.append((status.value, body["text"], reason))
    assert samples == [(row["status"], row["text"], row["reason"]) for row in golden]


def test_ratchet_exact_path_and_counterfactual(i18n_check):
    baseline = i18n_check.load_baseline(ROOT / "i18n/baseline.json")
    assert PATH in baseline["enforced"]
    assert baseline["files"][PATH] == 0
    source = (ROOT / PATH).read_text()
    assert i18n_check.scan_source(PATH, source) == []
    findings = i18n_check.scan_source(
        PATH, source + '\nraise HTTPException(422, "A synthetic literal regression.")\n'
    )
    assert len(findings) == 1
    assert i18n_check.in_enforced_scope(PATH, baseline["enforced"])
    assert not i18n_check.in_enforced_scope(PATH + ".other", baseline["enforced"])


@pytest.mark.parametrize("value", ["", " ", "x" * 32769])
def test_secret_validation_still_refuses_without_echo(value):
    from fastapi import HTTPException
    from pydantic import SecretStr

    with pytest.raises(HTTPException) as raised:
        auth._secret(auth.SecretInput(value=SecretStr(value)))
    assert raised.value.status_code == 422
    # Starlette types detail as str even though FastAPI accepts JSON details.
    detail = cast(dict[str, Any], raised.value.detail)
    assert isinstance(detail, dict)
    assert detail["text"] == "Enter a non-empty value of at most 32768 characters."
    assert detail["params"] == {}


@pytest.mark.parametrize("opaque", [OPAQUE, "{nested} '' apostrophe's", "кириллица हिन्दी 中文"])
def test_lower_layer_reasons_are_opaque_and_fallback_is_english(opaque):
    from local_operator.i18n.keys import wire_errors_auth as w

    for factory in (w.auth_refusal, w.provider_refusal):
        for locale in ("en", "es", "fr", "ru", "vi"):
            assert factory(error=opaque).envelope(locale=locale) == {
                "code": "wire.errors.auth." + factory.__name__,
                "params": {"error": opaque},
                "text": opaque,
            }


def test_namespace_census_has_no_unused_or_missing_keys():
    from local_operator.i18n.keys import wire_errors_auth as w

    source = ast.parse((ROOT / PATH).read_text())
    used = {
        node.func.attr
        for node in ast.walk(source)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "w"
    }
    catalogue = json.loads(
        (ROOT / "local_operator/i18n/catalogues/en/wire.errors.auth.json").read_text()
    )
    assert {"wire.errors.auth." + name for name in used} == set(catalogue)
    assert all(callable(getattr(w, name)) for name in used)


def test_secret_trim_is_unchanged():
    from pydantic import SecretStr

    assert auth._secret(auth.SecretInput(value=SecretStr(" synthetic "))) == "synthetic"
