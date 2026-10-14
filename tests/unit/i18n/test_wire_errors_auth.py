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


def constructors():
    return sorted(
        (
            node
            for node in ast.walk(ast.parse((ROOT / PATH).read_text()))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "HTTPException"
        ),
        key=lambda node: node.lineno,
    )


def test_full_sentence_status_and_params_match_base():
    golden = json.loads(
        (Path(__file__).parent / "fixtures/wire_errors_auth_en_golden.json").read_text()
    )
    calls = constructors()
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
