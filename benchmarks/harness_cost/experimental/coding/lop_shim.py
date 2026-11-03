"""Rebind only this benchmark child's first-party Anthropic endpoint.

WHY: the pinned provider ignores ANTHROPIC_BASE_URL. Rebinding both the client
constant and registry definition retains first-party cache/diagnostic behaviour
while measuring the same wire as other harnesses. Imported by lop_launch.py,
NOT as global sitecustomize/PYTHONPATH, which would break task Python tools.
No credentials are handled here and no installed provider source is modified.
"""

import dataclasses
import os

_url = os.environ.get("BENCH_ANTHROPIC_URL")
if _url:
    from local_operator.providers import clients, registry

    clients.ANTHROPIC_API_URL = _url
    for i, definition in enumerate(registry.PROVIDER_REGISTRY):
        if definition.id in ("anthropic", "anthropic-key"):
            replacement = dataclasses.replace(definition, base_url=_url)
            registry.PROVIDER_REGISTRY[i] = replacement
            registry._BY_ID[definition.id] = replacement
