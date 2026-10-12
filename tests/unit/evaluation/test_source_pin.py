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
        payload = {
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
