"""The caller's core-source pin: computation, verification and the selector model.

Blueprint §4: a native-caller selector must pin the CORE tree, not only the
adapter wheel, and an old selector (no pin) must fail at preflight. These
tests exercise the pin against the RUNNING tree (the same interpreter the
offline rig runs the arm with), so a match is a real match, not a fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from local_operator.evaluation.adapters.api import (
    ADAPTER_SCHEMA_VERSION,
    AdapterSelector,
    CoreSourcePin,
)
from local_operator.evaluation.source_pin import (
    CORE_SOURCE_KEYS,
    CORE_SOURCE_MODULES,
    SourcePinError,
    _module_origin,
    compute_core_source_hashes,
    main,
    resolve_core_commit,
    verify_core_source,
)


def _old_selector_payload(tmp_path: Path) -> dict[str, str]:
    """The pre-``core_source`` selector dictionary: exactly the old wire shape."""

    return {
        "schema_version": ADAPTER_SCHEMA_VERSION,
        "adapter_id": "osworld-v2",
        "distribution": "lop-osworld-v2-adapter",
        "version": "0.1.0",
        "entry_point": "lop_osworld_v2_adapter:create",
        "package_digest": "0" * 64,
        "release_digest": "1" * 64,
        "python_executable": str(tmp_path / "python"),
        "workspace": str(tmp_path / "workspace"),
        "workspace_digest": "2" * 64,
        "route_capability": "computer",
    }


class _LegacySelector(BaseModel):
    """The old consumer, simulated: same fields, ``extra="forbid"``, no pin.

    R5's round-trip needs a consumer that does NOT know the field (the
    released worker's selector, whose model forbids unknown keys) -- parsing
    the new class from an old dict only proves the new class accepts its own
    default. This model is what makes the wire-compatibility claim falsifiable
    in both directions.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    adapter_id: str
    distribution: str
    version: str
    entry_point: str
    package_digest: str
    release_digest: str
    python_executable: str
    workspace: str
    workspace_digest: str
    route_capability: str


def _valid_pin(**overrides: object) -> CoreSourcePin:
    values: dict[str, object] = {
        "commit": resolve_core_commit() or "deadbeef" * 5,
        "module_hashes": compute_core_source_hashes(),
    }
    values.update(overrides)
    return CoreSourcePin(**values)  # type: ignore[arg-type]


class TestCompute:
    def test_covers_every_pinned_module_with_hex_digests(self) -> None:
        hashes = compute_core_source_hashes()
        assert sorted(hashes) == sorted(CORE_SOURCE_KEYS)
        for digest in hashes.values():
            assert len(digest) == 64
            assert all(character in "0123456789abcdef" for character in digest)

    def test_module_origins_resolve_inside_the_running_package(self) -> None:
        import local_operator

        root = Path(local_operator.__file__).resolve().parent
        for _key, module in CORE_SOURCE_MODULES:
            origin = _module_origin(module)
            assert root in origin.parents, (module, origin)


class TestVerify:
    def test_a_matching_pin_passes_and_reports_provenance(self) -> None:
        report = verify_core_source(_valid_pin())
        assert sorted(report["module_hashes"]) == sorted(CORE_SOURCE_KEYS)
        assert report["commit_source"] in ("git", "unavailable")
        if resolve_core_commit() is not None:
            assert report["commit"] == resolve_core_commit()

    def test_a_mismatched_hash_refuses_naming_the_module(self) -> None:
        hashes = compute_core_source_hashes()
        hashes["loop"] = "0" * 64
        with pytest.raises(SourcePinError) as exc:
            verify_core_source(_valid_pin(module_hashes=hashes))
        assert "loop" in str(exc.value)

    def test_a_partial_pin_refuses(self) -> None:
        hashes = compute_core_source_hashes()
        del hashes["session_arm"]
        with pytest.raises(SourcePinError) as exc:
            verify_core_source(_valid_pin(module_hashes=hashes))
        assert "session_arm" in str(exc.value)

    def test_an_extra_module_refuses(self) -> None:
        hashes = compute_core_source_hashes()
        hashes["tool_bridge"] = hashes["loop"]
        with pytest.raises(SourcePinError) as exc:
            verify_core_source(_valid_pin(module_hashes=hashes))
        assert "tool_bridge" in str(exc.value)

    def test_a_commit_mismatch_refuses_when_git_can_answer(self) -> None:
        if resolve_core_commit() is None:
            pytest.skip("the running package sits outside a git checkout")
        with pytest.raises(SourcePinError) as exc:
            verify_core_source(_valid_pin(commit="0" * 40))
        assert "commit" in str(exc.value)

    def test_the_cli_prints_the_pin_as_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main([]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert sorted(payload["module_hashes"]) == sorted(CORE_SOURCE_KEYS)


class TestSelectorModel:
    def test_an_old_selector_still_parses_with_core_source_none(self, tmp_path: Path) -> None:
        payload = _old_selector_payload(tmp_path)
        selector = AdapterSelector.model_validate(payload)
        assert selector.core_source is None
        pinned = AdapterSelector.model_validate(
            {
                **payload,
                "core_source": {
                    "commit": "abc1234",
                    "module_hashes": compute_core_source_hashes(),
                },
            }
        )
        assert pinned.core_source is not None
        assert pinned.core_source.commit == "abc1234"

    def test_an_absent_pin_is_omitted_from_the_wire_for_old_consumers(self, tmp_path: Path) -> None:
        """R5: absent pin => canonical bytes WITHOUT the key, old shape intact.

        The old-selector direction that actually broke in review round 1 was
        the worker ECHO: the new model parsed an old selector, re-serialized
        it with ``core_source: null``, and the old parent refused the
        handshake before allocation. The round-trip asserted here is on the
        WIRE bytes: they must be byte-identical to the pre-field canonical
        form, and an old consumer (simulated by ``_LegacySelector``) must
        parse them.
        """

        payload = _old_selector_payload(tmp_path)
        canonical = AdapterSelector.model_validate(payload).to_canonical_json()
        assert b"core_source" not in canonical
        assert json.loads(canonical) == payload
        legacy = _LegacySelector.model_validate(json.loads(canonical))
        assert legacy.adapter_id == payload["adapter_id"]
        # Symmetric stability: parsing those bytes back and dumping again is
        # the same canonical form (no null key can reappear mid-flight).
        again = AdapterSelector.model_validate(json.loads(canonical))
        assert again.to_canonical_json() == canonical

    def test_a_present_pin_rides_the_wire_and_old_consumers_refuse_it(self, tmp_path: Path) -> None:
        """The one direction documented as incompatible: fail-closed at hello.

        A pinned selector is sent WITH ``core_source``; an old worker refuses
        it at the handshake (``extra="forbid"`` builds), before prepare/reset
        can allocate. That is a refusal, not data loss -- and the test pins
        the refusal so a future loosening of the model's strictness cannot
        silently accept a pin the old worker would ignore.
        """

        payload = {
            **_old_selector_payload(tmp_path),
            "core_source": {
                "commit": "abc1234",
                "module_hashes": compute_core_source_hashes(),
            },
        }
        canonical = AdapterSelector.model_validate(payload).to_canonical_json()
        assert b"core_source" in canonical
        with pytest.raises(ValidationError):
            _LegacySelector.model_validate(json.loads(canonical))


class TestCommitSyntax:
    """R6: the declared commit must be a hex SHA that PREFIXES the actual one."""

    def test_a_malformed_commit_extension_is_refused_at_construction(self) -> None:
        # '<sha>-corrupt' used to PASS verification: the actual commit is a
        # prefix of it, and the old check accepted either direction. It now
        # fails the field's syntax at construction, so a malformed pin cannot
        # even be serialized onto the wire.
        with pytest.raises(ValidationError):
            CoreSourcePin(commit="abc1234-corrupt", module_hashes={})

    def test_verify_refuses_a_malformed_commit_that_bypassed_construction(self) -> None:
        bypassed = CoreSourcePin.model_construct(
            commit="abc1234-corrupt", module_hashes=compute_core_source_hashes()
        )
        with pytest.raises(SourcePinError) as exc:
            verify_core_source(bypassed)
        assert "sha" in str(exc.value)

    def test_an_abbreviated_prefix_of_the_actual_commit_passes(self) -> None:
        actual = resolve_core_commit()
        if actual is None:
            pytest.skip("the running package sits outside a git checkout")
        report = verify_core_source(_valid_pin(commit=actual[:12]))
        assert report["commit"] == actual
