"""Environment injection into the vendored OSWorld package.

The bridge exists because upstream reads configuration from ``os.environ``, and
some of it at MODULE IMPORT time rather than at call time. Injecting an
import-time name after the first ``desktop_env`` import is a silent no-op: the
value is present, the process looks configured, and upstream has already made
its decision from the default. These tests pin the distinction.
"""

from __future__ import annotations

import os

import pytest
from lop_osworld_v2_adapter import vendor_bridge

from local_operator.evaluation.adapters.api import ScopedInfraValue


def _infra(name: str, value: str) -> ScopedInfraValue:
    return ScopedInfraValue(name=name, purpose="benchmark_compute", value=value)


@pytest.fixture(autouse=True)
def _restore_environment():
    """Injection writes to the real process environment; put it back."""
    names = ("PROXY_CONFIG_FILE", "WEBSITE_HOST_SUFFIX", "AWS_REGION", "NOT_INJECTABLE", "PATH")
    saved = {name: os.environ.get(name) for name in names}
    yield
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def test_the_proxy_config_path_reaches_the_process_environment() -> None:
    """The one property that makes a proxy task runnable at all.

    ``desktop_env.controllers.setup`` calls ``init_proxy_pool(PROXY_CONFIG_FILE)``
    at import, defaulting to a CWD-relative path that never resolves under the
    ``-I`` worker. Without this name being injected the pool loads zero proxies
    and the episode crashes at ``reset_start`` -- after the VM is billed.
    """
    os.environ.pop("PROXY_CONFIG_FILE", None)
    vendor_bridge.inject_infra_environment((_infra("PROXY_CONFIG_FILE", "/abs/proxies.json"),))
    assert os.environ["PROXY_CONFIG_FILE"] == "/abs/proxies.json"


def test_the_proxy_config_path_is_an_import_time_name() -> None:
    """Placement matters, not just presence.

    ``_OSWORLD_IMPORT_ENV`` is the set the adapter must set before importing
    ``desktop_env``. If PROXY_CONFIG_FILE ever moves to the call-time list the
    injection still "works" in a unit test and silently stops working in a real
    episode, so the membership itself is the assertion.
    """
    assert "PROXY_CONFIG_FILE" in vendor_bridge._OSWORLD_IMPORT_ENV


def test_a_name_outside_the_allowlist_is_not_injected() -> None:
    """The allowlist is a containment boundary, not a convenience."""
    os.environ.pop("NOT_INJECTABLE", None)
    vendor_bridge.inject_infra_environment((_infra("NOT_INJECTABLE", "secret-shaped-value"),))
    assert "NOT_INJECTABLE" not in os.environ


def test_ordinary_injectable_names_still_reach_the_environment() -> None:
    vendor_bridge.inject_infra_environment(
        (_infra("WEBSITE_HOST_SUFFIX", "example.test"), _infra("AWS_REGION", "us-east-1"))
    )
    assert os.environ["WEBSITE_HOST_SUFFIX"] == "example.test"
    assert os.environ["AWS_REGION"] == "us-east-1"


@pytest.fixture
def _stub_task_base(monkeypatch):
    """Make ``instantiate_task`` importable without the upstream package.

    ``instantiate_task`` imports ``desktop_env.task_base`` for the final
    ``issubclass(BaseTask)`` fallback in its class search. ``desktop_env`` is
    installed only in the evaluation venv, never in CI -- so guarding these
    tests with ``importorskip`` meant the whole ``sys.modules`` registration
    could be deleted with the suite green, which is exactly the regression
    they exist to catch. Neither probe reaches the fallback (both define
    ``get_task``), so a two-line stub buys real CI coverage of the fix.
    """
    import sys
    import types

    if "desktop_env.task_base" in sys.modules:  # the evaluation venv: use the real one
        yield
        return
    package = types.ModuleType("desktop_env")
    setattr(package, "__path__", [])  # a package, so the submodule import resolves
    task_base = types.ModuleType("desktop_env.task_base")

    class BaseTask:  # noqa: D401 - stands in for the upstream marker class
        """Stand-in for the upstream base class used only by an issubclass check."""

    # setattr, not attribute assignment: ModuleType has no declared
    # 'BaseTask' slot, so pyright rejects the direct form.
    setattr(task_base, "BaseTask", BaseTask)
    monkeypatch.setitem(sys.modules, "desktop_env", package)
    monkeypatch.setitem(sys.modules, "desktop_env.task_base", task_base)
    yield


def test_a_task_module_is_registered_before_execution(tmp_path, _stub_task_base) -> None:
    """A task combining future annotations with @dataclass must import.

    ``module_from_spec`` does NOT register in ``sys.modules``; the import
    system does that itself before executing a module, and upstream's loader
    gets it for free by going through ``import``. Skipping it breaks string
    annotations: dataclasses resolves them through ``_is_type``, which does
    ``sys.modules.get(cls.__module__)`` and reads ``.__dict__`` on the result.
    For an unregistered module that is None, and the task dies at
    ``reset_start`` -- with the VM already allocated -- with a bare
    ``AttributeError: 'NoneType' object has no attribute '__dict__'`` that
    names neither the task nor the cause.
    """
    import sys as _sys

    from lop_osworld_v2_adapter import vendor_bridge as bridge

    module_path = tmp_path / "task_probe.py"
    module_path.write_text(
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class _Result:\n"
        "    ok: bool\n"
        "    detail: str | None\n"
        "def get_task():\n"
        "    return _Result(ok=True, detail=None)\n",
        encoding="utf-8",
    )
    try:
        task = bridge.instantiate_task(str(module_path), "probe")
        assert task.ok is True
    finally:
        _sys.modules.pop("osworld_task_probe", None)


def test_a_task_module_that_fails_to_import_is_not_left_registered(
    tmp_path, _stub_task_base
) -> None:
    """A half-initialised module must not stay visible to the next episode."""
    import sys as _sys

    from lop_osworld_v2_adapter import vendor_bridge as bridge

    module_path = tmp_path / "task_broken.py"
    module_path.write_text("raise RuntimeError('boom')\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="boom"):
        bridge.instantiate_task(str(module_path), "broken")
    assert "osworld_task_broken" not in _sys.modules


def test_model_client_providers_reads_the_registry_the_judge_will_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The judge preflight must ask the REGISTRY, never a copied list.

    ``create_backend`` accepts exactly ``list_providers()``; a hand-kept copy
    in our tree would drift from the pinned upstream the day it changes,
    re-opening the silent-zero class this read exists to close. The stub
    carries marker names, not the real ones, so passing proves the list came
    from the module rather than from any literal -- and CI has no
    ``desktop_env`` (installed only in the evaluation venv).
    """
    import sys
    import types

    if "desktop_env.evaluators.backends" in sys.modules:  # the evaluation venv: the real one
        names = vendor_bridge.model_client_providers()
        assert {"openai", "openai_compatible", "anthropic", "bedrock", "gemini"} <= set(names)
        return

    package = types.ModuleType("desktop_env")
    setattr(package, "__path__", [])
    evaluators = types.ModuleType("desktop_env.evaluators")
    setattr(evaluators, "__path__", [])
    backends = types.ModuleType("desktop_env.evaluators.backends")
    setattr(backends, "list_providers", lambda: ["zeta", "alpha"])
    monkeypatch.setitem(sys.modules, "desktop_env", package)
    monkeypatch.setitem(sys.modules, "desktop_env.evaluators", evaluators)
    monkeypatch.setitem(sys.modules, "desktop_env.evaluators.backends", backends)

    assert vendor_bridge.model_client_providers() == ("zeta", "alpha")


def test_an_undeclared_tool_path_writes_nothing() -> None:
    """Unset must be today's environment byte for byte: no PATH key appears."""
    os.environ.pop("PATH", None)
    before = dict(os.environ)
    vendor_bridge.inject_evaluator_tool_path(None)
    assert dict(os.environ) == before and "PATH" not in os.environ


def test_declared_directories_follow_the_default_search_path() -> None:
    """Appended, never prepended: a declared directory cannot shadow a system tool."""
    os.environ["PATH"] = "/parent/shell/bin"
    vendor_bridge.inject_evaluator_tool_path(("/opt/tools/bin", "/srv/more"))
    assert os.environ["PATH"].split(os.pathsep) == [
        *os.defpath.split(os.pathsep),
        "/opt/tools/bin",
        "/srv/more",
    ]


_WORKER_PROBE = """
import json, os, shutil, subprocess, sys
sys.path[:0] = json.loads(sys.argv[1])
from lop_osworld_v2_adapter import provisioning, vendor_bridge
from local_operator.evaluation.adapters.api import ScopedInfraValue

def run():
    try:
        return subprocess.run(["lop-fake-evaluator-tool"], capture_output=True, text=True).stdout
    except FileNotFoundError:
        return None

report = {"path_at_start": os.environ.get("PATH"), "before": run()}
infra = tuple(
    ScopedInfraValue(name="EVALUATOR_TOOL_PATH", purpose="benchmark_compute", value=value)
    for value in sys.argv[3:]
)
vendor_bridge.inject_evaluator_tool_path(
    provisioning.resolve_evaluator_tool_path(infra, forbidden_roots=(sys.argv[2],))
)
report.update(
    after=run(),
    which=shutil.which("lop-fake-evaluator-tool"),
    path=os.environ.get("PATH"),
    parent_only=os.environ.get("LOP_PARENT_ONLY_MARKER"),
    names=sorted(os.environ),
)
print(json.dumps(report))
"""


def _probe_worker(tmp_path, *tool_path: str) -> dict:
    """Run the adapter's own resolve+inject in a child whose environment is
    built the way the supervisor builds the worker's, from a parent that has
    a PATH and a marker the child must never see."""
    import json
    import subprocess
    import sys

    from local_operator.evaluation.adapters.supervisor import minimal_environment

    parent = {
        "PATH": os.pathsep.join((str(tmp_path / "parent-bin"), "/usr/bin", "/bin")),
        "LOP_PARENT_ONLY_MARKER": "parent-only",
        "LANG": "C",
    }
    environment = minimal_environment(parent, protocol_fds={})
    # The constructed environment itself is unchanged by this feature.
    assert environment == {"LANG": "C"}
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    done = subprocess.run(
        [sys.executable, "-I", "-c", _WORKER_PROBE, json.dumps(sys.path), str(workspace)]
        + list(tool_path),
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.splitlines()[-1])


def _fake_tool(directory) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    tool = directory / "lop-fake-evaluator-tool"
    tool.write_text("#!/bin/sh\necho resolved-from-declared-dir\n")
    tool.chmod(0o755)


def test_a_worker_resolves_a_tool_only_from_a_declared_directory(tmp_path) -> None:
    """The real path: a child process built like the worker, shelling out.

    The same tool also sits in a directory that is ONLY on the parent's PATH;
    it must stay invisible, declared or not -- the declared list is the whole
    widening, and nothing else of the parent's environment arrives.
    """
    declared = tmp_path / "declared-tools"
    _fake_tool(declared)
    _fake_tool(tmp_path / "parent-bin")

    unset = _probe_worker(tmp_path)
    assert unset["path_at_start"] is None and unset["path"] is None
    assert unset["before"] is None and unset["after"] is None and unset["which"] is None
    assert "PATH" not in unset["names"] and "LOP_PARENT_ONLY_MARKER" not in unset["names"]

    declared_run = _probe_worker(tmp_path, str(declared))
    assert declared_run["before"] is None  # negative control inside the same child
    assert declared_run["after"] == "resolved-from-declared-dir\n"
    assert declared_run["which"] == str(declared / "lop-fake-evaluator-tool")
    assert declared_run["path"].split(os.pathsep) == [*os.defpath.split(os.pathsep), str(declared)]
    assert str(tmp_path / "parent-bin") not in declared_run["path"]
    assert declared_run["parent_only"] is None
    # The interpreter and the platform add names of their own at startup
    # (locale coercion, CoreFoundation), declared or not: the feature's whole
    # footprint is the difference between the two children.
    assert set(declared_run["names"]) - set(unset["names"]) == {"PATH"}
    assert set(unset["names"]) <= set(declared_run["names"])
