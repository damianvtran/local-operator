"""A10 vectors: the shared contract fixture, run with no model.

The merge core is direction-agnostic BY CONSTRUCTION: :func:`merge_field` takes
three texts and returns a result, and never learns which side the result will be
written to. So there is exactly one run per vector here, not a pull/push pair (a
parametrisation over a value the code under test cannot see would run identical
code twice and claim a check it does not make). The push half proves the OTHER
direction by consuming this same file unchanged (a vector edit needs both sides'
sign-off).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.hub_sync.merge import (
    ConflictProposal,
    ConflictRequest,
    FieldInput,
    MergeOptions,
    MergeResult,
    merge_field,
    replace_field,
)

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "hub_merge" / "vectors.json"
VECTORS = json.loads(FIXTURE.read_text(encoding="utf-8"))["vectors"]


class ScriptedResolver:
    """Replays the vector's ``model_stub`` proposals in order."""

    def __init__(self, script: list[dict[str, Any]]) -> None:
        self._script = list(script)
        self.requests: list[ConflictRequest] = []

    def resolve(self, req: ConflictRequest) -> ConflictProposal:
        self.requests.append(req)
        item = self._script.pop(0) if len(self._script) > 1 else self._script[0]
        return ConflictProposal(
            text=item["text"],
            covers=tuple(item.get("covers", ())),
            drops=tuple(item.get("drops", ())),
        )


def _run(vector: dict[str, Any], *, use_model: bool, prefer: str, ack: bool) -> MergeResult:
    inp = FieldInput(
        field={"markdown": "instructions", "scalar": "manager", "roster": "members"}[
            vector["field_kind"]
        ],
        kind=vector["field_kind"],
        base=vector["B"],
        local=vector["L"],
        remote=vector["R"],
    )
    if vector.get("replace"):
        return replace_field(inp, take=vector["replace"])
    resolver = (
        ScriptedResolver(vector["model_stub"]) if use_model and vector["model_stub"] else None
    )
    return merge_field(
        inp,
        MergeOptions(
            prefer=prefer,  # type: ignore[arg-type]
            allow_llm=resolver is not None,
            acknowledge_unknown_baseline=ack,
            max_items=vector.get("max_items"),
            resolver=resolver,
        ),
    )


def _norm(text: Any) -> Any:
    return text.strip() if isinstance(text, str) else text


def _check(result: MergeResult, expected: dict[str, Any]) -> None:
    assert result.outcome == expected["outcome"], result.to_json()
    if expected.get("merged") is not None:
        assert _norm(result.merged) == _norm(expected["merged"])
    by_name = {r.name: r for r in result.regions}
    for name, prov in expected.get("provenance_by_region", {}).items():
        assert name in by_name, (name, sorted(by_name))
        assert by_name[name].provenance == prov, (name, by_name[name])
    for name, who in expected.get("removed_by", {}).items():
        assert by_name[name].removed_by == who
    for warning in expected.get("warnings", []):
        assert warning in result.warnings
    for needle in expected.get("contains", []):
        assert needle in result.merged
    for needle in expected.get("absent", []):
        assert needle not in str(result.merged)
    if "engine_mode" in expected:
        assert result.engine.mode == expected["engine_mode"]
    if "refusal" in expected:
        assert expected["refusal"] in result.refusal
    for name, text in expected.get("dropped", {}).items():
        assert text in str(by_name[name].dropped)
    if "dropped_contains" in expected:
        assert any(expected["dropped_contains"] in str(r.dropped) for r in result.regions)


@pytest.mark.parametrize("vector", VECTORS, ids=[v["id"].split(" ")[0] for v in VECTORS])
def test_vector(vector: dict[str, Any]) -> None:
    use_model = bool(vector["model_stub"])
    _check(
        _run(
            vector, use_model=use_model, prefer="none", ack=vector["acknowledge_unknown_baseline"]
        ),
        vector["expected"],
    )


@pytest.mark.parametrize(
    "vector", [v for v in VECTORS if v.get("prefer_variants")], ids=lambda v: v["id"][:3]
)
def test_prefer_variants(vector: dict[str, Any]) -> None:
    for side, expected in vector["prefer_variants"].items():
        _check(_run(vector, use_model=False, prefer=side, ack=False), expected)


@pytest.mark.parametrize(
    "vector", [v for v in VECTORS if v.get("no_model")], ids=lambda v: v["id"][:3]
)
def test_no_model_variant(vector: dict[str, Any]) -> None:
    _check(_run(vector, use_model=False, prefer="none", ack=False), vector["no_model"])


@pytest.mark.parametrize(
    "vector", [v for v in VECTORS if v.get("ack_variant")], ids=lambda v: v["id"][:3]
)
def test_acknowledged_unknown_baseline(vector: dict[str, Any]) -> None:
    _check(_run(vector, use_model=False, prefer="none", ack=True), vector["ack_variant"])
