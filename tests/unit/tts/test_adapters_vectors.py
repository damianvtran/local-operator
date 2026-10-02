"""The golden-vector conformance test: this repo's adapters vs the shared contract.

The SAME ``vectors.v1.json`` runs in the hub's Go suite and here. That is the
whole mechanism that keeps the two executors from drifting — a mapping change
landing in one repo and not the other turns a test red here, instead of
changing how a user's speech sounds with nothing to point at.

The vectors are read from the vendored snapshot, never re-authored (see
``voicing_map/README.md`` for the sync discipline).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from local_operator.tts import adapters
from local_operator.tts.adapters import Legacy
from local_operator.tts.descriptor import DESCRIPTOR_FIELDS, VoiceDescriptor

MAP_PATH = adapters.VOICING_MAP_DIR / "speech_voicing_map.v1.json"
VECTORS_PATH = adapters.VOICING_MAP_DIR / "vectors.v1.json"


def _load_vectors() -> dict[str, Any]:
    with VECTORS_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


VECTORS = _load_vectors()["vectors"]


def _attempt_wire(attempt: adapters.Attempt) -> dict[str, Any]:
    """One attempt in the vectors' own shape, with the omitempty rules applied."""
    params: dict[str, object] = {}
    if attempt.params.voice_id:
        params["voice_id"] = attempt.params.voice_id
    if attempt.params.model_id:
        params["model_id"] = attempt.params.model_id
    if attempt.params.language_code:
        params["language_code"] = attempt.params.language_code
    if attempt.params.instructions:
        params["instructions"] = attempt.params.instructions
    if attempt.params.stability is not None:
        params["stability"] = attempt.params.stability
    if attempt.params.speed is not None:
        params["speed"] = attempt.params.speed
    return {
        "provider": attempt.provider,
        "voice_key": attempt.voice_key,
        "params": params,
        "notes": [note.to_wire() for note in attempt.notes],
    }


def test_the_vendored_pair_matches_its_own_version_constant() -> None:
    """A half-vendored pair (new JSON, stale constant) is a red test.

    The constant is what the daemon reports as the mapping it used, so a skew
    between it and the file would make the reported version a lie.
    """
    doc = adapters.load_map()
    assert doc.map_version == adapters.VENDORED_VERSION
    assert doc.descriptor_version == 1


def test_the_vendored_pair_is_internally_version_consistent() -> None:
    """The pair names the same map version, and it is the one this code reports.

    NOT a byte-identity check -- that is verified against the source commit
    when the pair is vendored (see ``voicing_map/README.md``) and cannot be
    recomputed offline from a single repo. What this pins is the failure that
    WOULD be silent: a half-vendored pair (new JSON, stale constant) making the
    reported mapping version a lie.
    """
    with MAP_PATH.open(encoding="utf-8") as handle:
        map_version = json.load(handle)["map_version"]
    with VECTORS_PATH.open(encoding="utf-8") as handle:
        vectors_version = json.load(handle)["map_version"]
    assert map_version == vectors_version == adapters.VENDORED_VERSION


@pytest.mark.parametrize("vector", VECTORS, ids=[v["name"] for v in VECTORS])
def test_every_golden_vector_matches(vector: dict[str, Any]) -> None:
    descriptor = VoiceDescriptor.from_wire(vector["descriptor"])
    legacy = Legacy(**vector.get("legacy", {}))
    attempt = adapters.resolve(vector["provider"], vector.get("model", ""), descriptor, legacy)
    assert _attempt_wire(attempt) == vector["expect"]


@pytest.mark.parametrize("vector", VECTORS, ids=[v["name"] for v in VECTORS])
def test_every_descriptor_field_is_applied_or_noted(vector: dict[str, Any]) -> None:
    """The invariant: nothing is dropped silently.

    A field with no note would be a descriptor that asked for something and
    the mapping that quietly ignored it — the failure mode this whole feature
    exists to prevent.
    """
    descriptor = VoiceDescriptor.from_wire(vector["descriptor"])
    attempt = adapters.resolve(vector["provider"], vector.get("model", ""), descriptor)
    noted = {note.field for note in attempt.notes}
    for field in DESCRIPTOR_FIELDS:
        assert field in noted, f"{vector['name']}: {field} is neither applied nor noted"


def test_an_absent_descriptor_maps_nothing() -> None:
    """No descriptor is exactly the pre-descriptor behaviour."""
    attempt = adapters.resolve("elevenlabs", "eleven_multilingual_v2", None)
    assert list(attempt.notes) == []
    assert attempt.params.voice_id == ""
    assert attempt.params.stability is None


def test_the_degraded_header_is_capped_on_a_rune_boundary() -> None:
    """A header must never be the thing that fails a request."""
    tokens = [f"field{n}:ignored={'x' * 40}" for n in range(40)]
    rendered = adapters.format_degraded(tokens)
    assert len(rendered.encode("utf-8")) <= 512
    assert rendered.endswith("…")
    # Valid UTF-8 (the cut ran on a boundary, not through a rune).
    rendered.encode("utf-8").decode("utf-8")
