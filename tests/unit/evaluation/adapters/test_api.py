from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from local_operator.evaluation.adapters.api import (
    AdapterCapabilities,
    AdapterMetadata,
    AdapterSelector,
    Handshake,
    PythonRuntime,
    RescueDescriptor,
    ScopedInfraValue,
    SecretRef,
)
from local_operator.evaluation.lifecycle import CleanupAction, CleanupPlan

DIGEST = "a" * 64


def selector(tmp_path: Path) -> AdapterSelector:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return AdapterSelector(
        schema_version="1.0",
        adapter_id="tiny",
        distribution="tiny-adapter",
        version="1.2.3",
        entry_point="tiny_adapter:create",
        package_digest=DIGEST,
        release_digest="b" * 64,
        python_executable=str(Path(sys.executable).resolve()),
        workspace=str(workspace),
        route_capability="computer",
    )


def metadata() -> AdapterMetadata:
    return AdapterMetadata(
        adapter_id="tiny",
        distribution="tiny-adapter",
        version="1.2.3",
        entry_point="tiny_adapter:create",
        package_digest=DIGEST,
        release_digest="b" * 64,
        schema_version="1.0",
        capabilities=AdapterCapabilities(routes=("computer",), ask_user=True, scoring=True),
    )


def handshake(tmp_path: Path) -> Handshake:
    selected = selector(tmp_path)
    return Handshake(
        selector=selected,
        metadata=metadata(),
        python=PythonRuntime.current(),
        workspace_digest="c" * 64,
        selected_route="computer",
    )


def cleanup_plan() -> CleanupPlan:
    return CleanupPlan(
        episode_id="episode",
        actions=(
            CleanupAction(
                action_id="release",
                kind="release_instance",
                resource_ref="instance-ref",
                timeout_ms=1000,
                max_attempts=2,
            ),
        ),
    )


def test_package_root_does_not_import_discovery_or_worker() -> None:
    probe = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                f"import sys;sys.path.insert(0,{str(Path.cwd())!r});"
                "import local_operator.evaluation.adapters;"
                "assert 'local_operator.evaluation.adapters.discovery' not in sys.modules;"
                "assert 'local_operator.evaluation.adapters.worker' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stderr


def test_selector_rejects_relative_paths(tmp_path: Path) -> None:
    payload = selector(tmp_path).model_dump(mode="json")
    payload["python_executable"] = "python"
    with pytest.raises(ValidationError, match="normalized absolute"):
        AdapterSelector.model_validate(payload, strict=True)


def test_handshake_requires_every_exact_pin(tmp_path: Path) -> None:
    selected = selector(tmp_path)
    changed = metadata().model_copy(update={"version": "1.2.4"})
    with pytest.raises(ValidationError, match="exact adapter selection"):
        Handshake(
            selector=selected,
            metadata=changed,
            python=PythonRuntime.current(),
            workspace_digest="c" * 64,
            selected_route="computer",
        )


def test_scoped_infra_has_no_provider_or_model_purpose() -> None:
    schema = ScopedInfraValue.model_json_schema()
    purposes = schema["properties"]["purpose"]["enum"]
    assert not {"provider", "model", "openrouter"} & set(purposes)
    with pytest.raises(ValidationError):
        ScopedInfraValue.model_validate(
            {"name": "route", "purpose": "provider", "value": "bad"}, strict=True
        )


def test_rescue_descriptor_is_content_bound_and_has_refs_not_secrets(tmp_path: Path) -> None:
    selected = selector(tmp_path)
    descriptor = RescueDescriptor(
        schema_version="1.0",
        selector=selected,
        handshake=handshake(tmp_path),
        episode_id="episode",
        cleanup_plan=cleanup_plan(),
        secret_refs=(SecretRef(name="BENCHMARK_TOKEN"),),
        infra_values=(
            ScopedInfraValue(name="region", purpose="benchmark_compute", value="us-test-1"),
        ),
        artifact_root=str(tmp_path),
    )
    encoded = descriptor.to_canonical_json()
    assert b"BENCHMARK_TOKEN" in encoded
    assert b"secret-value" not in encoded
    changed = json.loads(encoded)
    changed["episode_id"] = "different"
    with pytest.raises(ValidationError):
        RescueDescriptor.model_validate(changed, strict=True)
