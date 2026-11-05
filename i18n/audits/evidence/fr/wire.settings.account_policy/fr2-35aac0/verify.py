"""Namespace-only audit probes. Scratch fixtures, never the repository catalogues."""
from __future__ import annotations
import hashlib
import json
import sys
import unicodedata
from pathlib import Path

REPO = Path.home() / "local-operator"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from scripts.i18n import check
from local_operator.i18n import catalogues, runtime
from rich.cells import cell_len

NS = "wire.settings.account_policy"
catalogues._CATALOGUES = HERE / "catalogues"
target = catalogues._CATALOGUES / "fr" / f"{NS}.json"
original = target.read_bytes()

def strict_object(pairs):
    result = {}
    for key, value in pairs:
        assert key not in result, f"duplicate key: {key}"
        result[key] = value
    return result

def load(data):
    return json.loads(data, object_pairs_hook=strict_object)

en = load((catalogues._CATALOGUES / "en" / f"{NS}.json").read_bytes())
fr = load(original)
ctx = load((catalogues._CATALOGUES / "en" / f"{NS}.context.json").read_bytes())
assert len(en) == len(fr) == len(ctx) == 14
assert list(en) == list(fr)
assert set(en) == set(ctx)

# Run the repo's real namespace validator first on only this two-locale fixture.
problems = check.check_catalogues()
assert not problems, problems

# Independently assert the narrower contract and token/width rules, which the
# general checker does not enforce on every namespace.
def extra_checks(candidate):
    assert set(candidate) == set(en)
    for key, text in candidate.items():
        assert isinstance(text, str) and text.strip()
        assert runtime.message_arguments(text) == runtime.message_arguments(en[key]) == ()
        assert not any(unicodedata.category(ch) == "Cf" for ch in text), key
        assert text != en[key], key
        if "maxCells" in ctx[key]:
            assert cell_len(text) <= ctx[key]["maxCells"], key
    assert "true" in candidate[f"{NS}.legacy.help"]
    assert "false" in candidate[f"{NS}.legacy.help"]
    assert "value" in candidate[f"{NS}.error.invalid_shape"]
    assert "JSON" in candidate[f"{NS}.error.invalid_json"]

extra_checks(fr)
controls = []
# These controls prove the actual core checker rejects parity/ICU corruption.
for label, mutate in [
    ("dropped_key", lambda d: d.pop(f"{NS}.error.unavailable")),
    ("unexpected_ICU_argument", lambda d: d.__setitem__(f"{NS}.error.unavailable", "Indisponible {account}")),
]:
    bad = dict(fr)
    mutate(bad)
    try:
        target.write_text(json.dumps(bad, ensure_ascii=False))
        errors = check.check_catalogues()
        assert errors, f"negative control unexpectedly accepted: {label}"
        controls.append({"name": label, "rejected_by": "repo check_catalogues", "actual_errors": errors})
    finally:
        target.write_bytes(original)
# Additional negative controls isolate the machine-token, width and bidi checks.
for label, key, text in [
    ("machine_value_translated", f"{NS}.error.invalid_shape", "Seul le champ valeur est attendu."),
    ("label_width_overflow", f"{NS}.label", "x" * 30),
    ("bidi_control_injection", f"{NS}.error.unavailable", "Indisponible\u202e"),
]:
    bad = dict(fr)
    bad[key] = text
    try:
        extra_checks(bad)
    except AssertionError:
        controls.append({"name": label, "rejected_by": "independent extra_checks"})
    else:
        raise AssertionError(f"negative control unexpectedly accepted: {label}")
assert not check.check_catalogues()
assert target.read_bytes() == original

out = {
    "provenance_reference": "record.json:auditor",
    "execution_kind": "actual namespace-only Python validator and discriminator execution",
    "repo_validator": {"function": "scripts.i18n.check.check_catalogues", "problems": problems},
    "locale": "fr", "namespace": NS,
    "key_count": len(fr),
    "arguments": {key: list(runtime.message_arguments(fr[key])) for key in fr},
    "plural_and_select": "not applicable: all 14 source and target strings are argument-free",
    "cf_count": sum(unicodedata.category(c) == "Cf" for text in fr.values() for c in text),
    "widths": [{"key": key, "text": fr[key], "cells": cell_len(fr[key]), "maxCells": ctx[key]["maxCells"]} for key in ctx if "maxCells" in ctx[key]],
    "negative_controls": controls,
    "fixture_restored": True,
    "tool_inputs": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in [REPO / "scripts/i18n/check.py", REPO / "local_operator/i18n/runtime.py", REPO / "local_operator/i18n/catalogues.py"]],
    "pseudo": "NOT RUN", "RTL_rendering": "NOT RUN", "native_rendered_fit": "NOT RUN",
}
print(json.dumps(out, ensure_ascii=False, indent=2))
