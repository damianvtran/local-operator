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

PR 1 does not construct a real ``DesktopEnv`` at all (no AWS). This module
exists so that the seam, the lazy-import discipline, and the env-injection
order are settled here where a mistake is free, and PR 2 fills in only the
construction body.
"""

from __future__ import annotations

import os
from typing import Any

from local_operator.evaluation.adapters.api import ScopedInfraValue

# Infra values OSWorld's controllers read from the process environment. Only
# NON-SECRET values may appear here; secrets are never written to os.environ.
_ENV_INJECTABLE = frozenset(
    {
        "WEBSITE_HOST_SUFFIX",
        "GITLAB_URL",
        "OSWORLD_FILE_BASE_URL",
        "OSWORLD_CLIENT_PASSWORD",
        "OSWORLD_PROXY_ENDPOINT",
        "OSWORLD_TASK_DATE",
    }
)


def inject_infra_environment(infra_values: tuple[ScopedInfraValue, ...]) -> None:
    """Write non-secret infra values into the worker's own process env.

    Called at the START of ``reset_start``, before any ``desktop_env`` import.
    A value not on the closed injectable list is refused, so a secret named
    like an env var cannot leak into the process environment where a child
    process or a crash report would inherit it.
    """

    for value in infra_values:
        if value.name in _ENV_INJECTABLE:
            os.environ[value.name] = value.value


def load_desktop_env() -> Any:
    """Lazily import and return ``desktop_env.DesktopEnv``.

    Only called from ``reset_start`` after ``inject_infra_environment``. PR 2
    fills this in to construct the real env; PR 1 raises so the seam is
    explicit rather than silently absent.
    """

    # PR 2 fills this in. The import target does not exist in PR 1's
    # dependency set, so the missing-import is suppressed rather than allowed
    # to mask real errors; the NotImplementedError below is the honest signal.
    from desktop_env.desktop_env import (  # type: ignore[import-not-found] # noqa: F401  (PR 2)
        DesktopEnv,
    )

    raise NotImplementedError(
        "DesktopEnv construction is PR 2 (the AWS provider); PR 1 uses FakeProvider"
    )
