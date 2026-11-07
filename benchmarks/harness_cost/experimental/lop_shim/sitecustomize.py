"""Benchmark-only shim: point Local Operator's Anthropic wire at the metering proxy.

WHY: lop has no base-URL override for its first-party `anthropic` provider
(registry base_url is fixed; providers/local.resolve_base_url only honours
overrides for local providers). Every other harness reads ANTHROPIC_BASE_URL,
so to measure lop at the same point we rebind, in the benchmark child only:
  * clients.ANTHROPIC_API_URL -> proxy URL (keeps the `== ANTHROPIC_API_URL`
    first-party branches -- cache-TTL wire hints, diagnostics -- behaving as
    they do against api.anthropic.com, so lop is measured as shipped);
  * the `anthropic` ProviderDefinition's base_url -> proxy URL.
Active only when BENCH_ANTHROPIC_URL is set; no effect on any other process.
"""

import dataclasses
import os

_url = os.environ.get("BENCH_ANTHROPIC_URL")
if _url:
    from local_operator.providers import clients, registry

    clients.ANTHROPIC_API_URL = _url
    for i, d in enumerate(registry.PROVIDER_REGISTRY):
        if d.id in ("anthropic", "anthropic-key"):
            nd = dataclasses.replace(d, base_url=_url)
            registry.PROVIDER_REGISTRY[i] = nd
            registry._BY_ID[d.id] = nd
