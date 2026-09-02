"""The only module permitted to import ``desktop_env``.

C4: ``desktop_env.controllers.gitlab`` and ``desktop_env.controllers.website``
raise ``ValueError`` AT IMPORT when their environment variables are unset. If
any adapter module imported ``desktop_env`` at adapter-import time, the whole
worker would fail to load on a task that never touches those controllers. So
this module is the single, lazy bridge: it is imported only inside
``reset_start``/``score``, AFTER the corresponding ``infra_values`` have been
written into ``os.environ`` by the adapter.

Writing our own process env from ``infra_values`` is legitimate: those values
are non-secret by ``ScopedInfraValue``'s definition, and OSWorld's controllers
read them from the environment. Secrets never go through ``os.environ`` — they
are handed to the boto3 session / OSWorld settings directly.

THE ONE DOCUMENTED EXCEPTION is the judge key. OSWorld's evaluator model
client (``desktop_env/evaluators/model_client.py``) resolves its API key from
the process environment and nowhere else, and it runs INSIDE upstream's
``env.evaluate()`` — the binding honesty rule is that upstream evaluators are
called unmodified, so there is no argument path to hand it the key. The
adapter therefore writes ``OSWORLD_EVAL_MODEL_API_KEY`` (and the simulator
twin) into the WORKER's own environment at ``reset_start`` and scrubs both on
``close``. That environment is the stripped one the supervisor built, it is
never inherited by a parent, and the worker never spawns children that could
inherit it. See ``adapter.OSWorldV2Adapter._install_judge_environment``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from local_operator.evaluation.adapters.api import ScopedInfraValue

# Infra values OSWorld's controllers read from the process environment. Only
# NON-SECRET values may appear here; secrets are never written to os.environ.
#
# The ``OSWORLD_EVAL_MODEL_*`` / ``OSWORLD_USER_SIM_*`` names are the judge's
# and simulator's provider/model/base-URL settings (``model_client.py``
# ``_ENV_MAP``). They are non-secret and arrive as ``benchmark_judge`` /
# ``benchmark_user_simulator`` infra; the corresponding ``*_API_KEY`` names are
# deliberately NOT here — they are secrets and take the documented exception
# path above.
_ENV_INJECTABLE = frozenset(
    {
        "WEBSITE_HOST_SUFFIX",
        "GITLAB_URL",
        "OSWORLD_FILE_BASE_URL",
        "OSWORLD_CLIENT_PASSWORD",
        "OSWORLD_PROXY_ENDPOINT",
        "OSWORLD_TASK_DATE",
        "OSWORLD_EVAL_MODEL_PROVIDER",
        "OSWORLD_EVAL_MODEL_NAME",
        "OSWORLD_EVAL_MODEL_BASE_URL",
        "OSWORLD_USER_SIM_PROVIDER",
        "OSWORLD_USER_SIM_MODEL",
        "OSWORLD_USER_SIM_BASE_URL",
    }
)

# The two secret names OSWorld reads from the environment. Written by the
# adapter only inside the worker and only for the episode's duration.
JUDGE_KEY_ENV = "OSWORLD_EVAL_MODEL_API_KEY"
USER_SIM_KEY_ENV = "OSWORLD_USER_SIM_API_KEY"
SECRET_ENV_NAMES = frozenset({JUDGE_KEY_ENV, USER_SIM_KEY_ENV})

# OSWorld's AWS manager module raises ``EnvironmentError`` AT IMPORT unless
# these three are in the environment (manager.py:18-23) — and ``DesktopEnv``
# imports it through ``create_vm_manager_and_provider`` even though our
# provider never calls ``_allocate_vm``. They are account facts, not secrets,
# and are already infra values; they are mirrored into the env purely so the
# import succeeds. ``ENABLE_TTL=false`` is set for the same reason: OSWorld's
# own TTL path is a warning-on-failure one we never rely on, and leaving it
# enabled would race our own schedule with a second, unnamed one.
_OSWORLD_IMPORT_ENV = ("AWS_REGION", "AWS_SUBNET_ID", "AWS_SECURITY_GROUP_ID")


def inject_infra_environment(infra_values: tuple[ScopedInfraValue, ...]) -> None:
    """Write non-secret infra values into the worker's own process env.

    Called at the START of ``reset_start``, before any ``desktop_env`` import.
    A value not on the closed injectable list is refused, so a secret named
    like an env var cannot leak into the process environment where a child
    process or a crash report would inherit it.
    """

    for value in infra_values:
        if value.name in _ENV_INJECTABLE or value.name in _OSWORLD_IMPORT_ENV:
            os.environ[value.name] = value.value
    os.environ["ENABLE_TTL"] = "false"
    os.environ["AWS_AUTO_CREATE_SCHEDULER_ROLE"] = "false"


def install_secret_environment(values: dict[str, str]) -> None:
    """The documented exception: hand OSWorld the judge/simulator keys.

    Only the two names in ``SECRET_ENV_NAMES`` are ever written; any other
    name is refused so the AWS credentials (which the provider consumes
    directly) can never take this path by accident.
    """

    for name, value in values.items():
        if name not in SECRET_ENV_NAMES:
            raise ValueError(f"{name!r} is not an environment-delivered secret")
        os.environ[name] = value


def scrub_secret_environment() -> None:
    """Undo ``install_secret_environment``; safe to call when nothing was set."""

    for name in SECRET_ENV_NAMES:
        os.environ.pop(name, None)


def load_desktop_env() -> Any:
    """Lazily import and return ``desktop_env.desktop_env.DesktopEnv``.

    Only called from ``reset_start`` after ``inject_infra_environment``. The
    import is deferred to here because ``desktop_env`` pulls in gymnasium,
    pyautogui and OSWorld's provider modules, some of which read the
    environment at import; none of that may happen at adapter-import time.
    """

    from desktop_env.desktop_env import DesktopEnv  # type: ignore[import-not-found]

    return DesktopEnv


def instantiate_task(module_path: str, task_id: str) -> Any:
    """Import the task module by file location and instantiate its task class.

    Mirrors OSWorld's own ``task_loader._instantiate_task_from_module`` (the
    ``get_task()`` / ``TASK_CLASS`` / ``Task`` / ``BaseTask``-subclass search)
    so the object handed to ``DesktopEnv.reset(task_config=...)`` is exactly
    what upstream's runner would build. Done by location, never via
    ``sys.path``, so a task module cannot shadow an installed package.
    """

    import importlib.util

    from desktop_env.task_base import BaseTask  # type: ignore[import-not-found]

    spec = importlib.util.spec_from_file_location(f"osworld_task_{task_id}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build an import spec for {module_path!r}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    get_task = getattr(module, "get_task", None)
    if callable(get_task):
        return get_task()
    task_class = getattr(module, "TASK_CLASS", None)
    if task_class is not None:
        return task_class()
    task = getattr(module, "Task", None)
    if callable(task):
        return task()
    for value in vars(module).values():
        if isinstance(value, type) and issubclass(value, BaseTask) and value is not BaseTask:
            return value()
    raise ValueError(f"no task class found in {Path(module_path).name}")
