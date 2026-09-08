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
    names = ("PROXY_CONFIG_FILE", "WEBSITE_HOST_SUFFIX", "AWS_REGION", "NOT_INJECTABLE")
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


def test_a_task_module_is_registered_before_execution(tmp_path) -> None:
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

    # instantiate_task imports BaseTask for the class search; the upstream
    # package is only present in the evaluation venv.
    pytest.importorskip("desktop_env.task_base")
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


def test_a_task_module_that_fails_to_import_is_not_left_registered(tmp_path) -> None:
    """A half-initialised module must not stay visible to the next episode."""
    import sys as _sys

    pytest.importorskip("desktop_env.task_base")
    from lop_osworld_v2_adapter import vendor_bridge as bridge

    module_path = tmp_path / "task_broken.py"
    module_path.write_text("raise RuntimeError('boom')\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="boom"):
        bridge.instantiate_task(str(module_path), "broken")
    assert "osworld_task_broken" not in _sys.modules
