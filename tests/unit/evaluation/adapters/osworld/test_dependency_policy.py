"""Keep upstream compatibility overrides explicit and aligned with the harness.

The optional SDK test exercises its real signing helper without making a model
call. CI need not install OSWorld's large optional dependency tree merely to
check policy; the paid runtime runs this same test with that tree installed.
"""

from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[5]


def _requirements(values: list[str]) -> dict[str, Requirement]:
    parsed = [Requirement(value) for value in values]
    return {canonicalize_name(value.name): value for value in parsed}


def test_overrides_match_current_harness_floors() -> None:
    harness = tomllib.loads((ROOT / "pyproject.toml").read_text())
    adapter = tomllib.loads((ROOT / "benchmarks/osworld_v2_adapter/pyproject.toml").read_text())
    required = _requirements(harness["project"]["dependencies"])
    overrides = _requirements(adapter["tool"]["uv"]["override-dependencies"])
    assert set(overrides) == {"requests", "pyjwt"}
    for name in overrides:
        assert overrides[name].specifier == required[name].specifier
        assert overrides[name].extras == required[name].extras
    assert Version("2.8.0") not in overrides["pyjwt"].specifier
    assert Version("2.13.0") in overrides["pyjwt"].specifier
    assert Version("3.0.0") not in overrides["pyjwt"].specifier
    assert Version("2.31.0") not in overrides["requests"].specifier
    assert Version("2.34.2") in overrides["requests"].specifier


def test_legacy_sdk_signing_remains_compatible() -> None:
    pytest.importorskip("jwt")
    pytest.importorskip("zhipuai.core._jwt_token")
    # Reuse the exact no-pytest entry point run in the paid environment. Do not
    # install test dependencies into that frozen interpreter merely for a probe.
    path = ROOT / "benchmarks/osworld_v2_adapter/probes/jwt_signing.py"
    spec = importlib.util.spec_from_file_location("sdk_signing_probe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.probe()
    assert result["token_sha256"] == module.EXPECTED_DIGEST
    assert result["claims_verified"] is True
    assert result["header_verified"] is True
