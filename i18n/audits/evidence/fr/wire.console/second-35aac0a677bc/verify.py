"""Independent namespace-only translation probes; no scans or application launch."""
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

REPO = Path.home() / "local-operator"
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from scripts.i18n import check
from local_operator.i18n import catalogues, runtime
from rich.cells import cell_len

catalogues._CATALOGUES = ROOT / "catalogues"
target = next((ROOT / "catalogues/fr").glob("*.json"))
ns = target.stem
raw = target.read_bytes()
def strict(pairs):
    out = {}
    for key, value in pairs:
        assert key not in out, f"duplicate key {key}"
        out[key] = value
    return out

en = json.loads((ROOT / f"catalogues/en/{ns}.json").read_bytes(), object_pairs_hook=strict)
fr = json.loads(raw, object_pairs_hook=strict)
ctx = json.loads((ROOT / f"catalogues/en/{ns}.context.json").read_bytes(), object_pairs_hook=strict)
assert set(en) == set(fr) == set(ctx)
assert list(en) == list(fr)

# Existing repository validator, isolated to this single carried namespace.
problems = check.check_catalogues()
assert not problems, problems

# Independently authored technical-token inventory, not the translator's checks.
tokens = {}
if ns == "wire.errors.sdk_spec":
    tokens = {
        "agent_identity_conflict": ["agent_name", "agent_id"],
        "approval_callback_gate_required": ["ApprovalPolicy.callback()", "gate"],
        "approval_declared_tools_empty": ["ApprovalPolicy.declared()", "ApprovalPolicy.refuse()"],
        "approval_handler_wrong_mode": ["ApprovalPolicy.handler", "''callback''"],
        "approval_tools_wrong_mode": ["ApprovalPolicy.tools", "''declared''"],
        "attach_fields_unsupported": ["mode=''attach''", "spec.resume", "resume", "mode=''own''"],
        "attach_output_unsupported": ["runtime"],
        "attach_resume_required": ["mode=''attach''", "spec.resume"],
        "mode_invalid": ["mode", "''own''", "''attach''"],
        "output_format_invalid": ["output_format"],
        "output_retries_range": ["output_retries", "0", "5"],
        "output_retries_requires_format": ["output_retries", "output_format"],
        "output_schema_requires_format": ["output_schema", "output_format"],
        "resume_empty": ["resume", '"@latest"'],
        "spawn_birth_effort_pair_required": ["birth_effort", "hosting/model", "open_session"],
        "spawn_fields_unsupported": ["spawn_session()", "open_session", "dispose", "resume"],
        "spawn_model_pair_required": ["spawn_session()", "hosting/model", "runtime"],
        "spawn_output_unsupported": ["spawn_session()", "open_session(..., mode=''own'')"],
        "tools_empty": ["tools", "None"],
        "world_root_empty": ["world_root"],
        "yolo_approval_unreachable": [en[f"{ns}.yolo_approval_unreachable"].split(" makes ", 1)[0], "ApprovalPolicy.{approval_mode}()", "ApprovalPolicy.auto()", "yolo"],
    }

def extra(candidate):
    assert set(candidate) == set(en)
    for key, text in candidate.items():
        assert isinstance(text, str) and text.strip()
        assert text != en[key], key
        assert runtime.message_arguments(text) == runtime.message_arguments(en[key]), key
        assert re.findall(r"\d+", text) == re.findall(r"\d+", en[key]), key
        assert re.findall(r"`[^`]*`", text) == re.findall(r"`[^`]*`", en[key]), key
        assert not any(unicodedata.category(ch) == "Cf" for ch in text), key
        if "maxCells" in ctx[key]:
            assert cell_len(text) <= ctx[key]["maxCells"], key
        for token in tokens.get(key.removeprefix(ns + "."), []):
            assert token in text, (key, token)
extra(fr)

# Exercise ICU formatting with opaque syntax-looking parameters. They must be
# inserted as data, never reparsed or stripped; this does not exercise SDK guards.
rows = []
opaque = "{opaque, plural, one {A} other {B}}"
for key, text in fr.items():
    args = runtime.message_arguments(text)
    params = {name: opaque for name, kind in args}
    rendered = runtime.render_message(text, params, "fr")
    for name, kind in args:
        assert kind == "simple", (key, name, kind)
        assert opaque in rendered, key
    rows.append({"key": key, "arguments": list(args), "cells": cell_len(rendered), "maxCells": ctx[key].get("maxCells"), "rendered_parser_sample": rendered})

controls = []
for label, mutate in [
    ("missing_key", lambda d: d.pop(next(iter(d)))),
    ("unexpected_argument", lambda d: d.__setitem__(next(iter(d)), d[next(iter(d))] + " {unexpected}")),
]:
    bad = dict(fr)
    mutate(bad)
    try:
        target.write_text(json.dumps(bad, ensure_ascii=False))
        errors = check.check_catalogues()
        assert errors, label
        controls.append({"name": label, "rejected_by": "repo check_catalogues", "actual_errors": errors})
    finally:
        target.write_bytes(raw)
for label, key, text in [
    ("number_mutation", next((k for k in fr if re.search(r"\d", fr[k])), None), None),
    ("Cf_injection", next(iter(fr)), fr[next(iter(fr))] + "\u202e"),
]:
    if key is None:
        continue
    bad = dict(fr)
    bad[key] = re.sub(r"\d", "9", fr[key], count=1) if text is None else text
    try:
        extra(bad)
    except AssertionError:
        controls.append({"name": label, "rejected_by": "independent extra"})
    else:
        raise AssertionError(f"negative accepted: {label}")
if ns == "wire.errors.sdk_spec":
    bad = dict(fr)
    bad[f"{ns}.resume_empty"] = bad[f"{ns}.resume_empty"].replace("@latest", "@dernier")
    try:
        extra(bad)
    except AssertionError:
        controls.append({"name": "technical_token_mutation", "rejected_by": "independent extra"})
    else:
        raise AssertionError("technical token mutation accepted")
assert target.read_bytes() == raw
assert not check.check_catalogues()
out = {
    "provenance_reference": "record.json:auditor", "locale": "fr", "namespace": ns,
    "key_count": len(fr), "repo_validator_problems": problems, "messages": rows,
    "cf_count": 0, "numeric_literal_parity": True, "technical_tokens": tokens,
    "plural_select_CLDR": "not applicable: zero plural/select in source or target",
    "negative_controls": controls, "fixture_restored": True,
    "execution_scope": "core ICU parser/formatter and namespace validator only; no SDK guards, live settings, app or native rendering",
    "pseudo": "NOT RUN", "RTL_native": "NOT RUN", "native_fit": "NOT RUN",
    "tool_inputs": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in [REPO / "scripts/i18n/check.py", REPO / "local_operator/i18n/runtime.py", REPO / "local_operator/i18n/catalogues.py"]],
}
print(json.dumps(out, ensure_ascii=False, indent=2))
