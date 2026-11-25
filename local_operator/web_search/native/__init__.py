"""The native-search family: one module per model vendor whose API has built-in search.

``NATIVE_SEARCH_VENDORS`` is the registry ``web_search.providers`` reads at import
time to add each vendor to ``PROVIDERS`` (its transport and status row) and to
``NATIVE_VENDORS`` (the band code's view). A vendor joins by adding its module
name to :data:`_VENDOR_MODULES`; each module exposes one ``VENDOR``
(:class:`~local_operator.web_search.native.base.NativeSearchVendor`).

DeepSeek's native search is NOT here: it predates the family, carries its own
balance gate and evidence pass, and stays in ``providers.py`` unchanged. It
shares this package's Messages parser (``anthropic_wire``).

Registration order is the within-class order for legs with equal fees (plan
legs keep it; paid legs sort cheapest-first, ``providers._native_order``).
"""

from __future__ import annotations

from importlib import import_module

from local_operator.web_search.native.base import NativeSearchVendor

#: Vendor modules under this package, in registration order. Kept as names (not
#: imports) so a vendor slice adds exactly one line here and nothing else in
#: shared code needs to know the module exists.
_VENDOR_MODULES: tuple[str, ...] = ("openai", "anthropic", "xai", "alibaba")


def _load() -> dict[str, NativeSearchVendor]:
    vendors: dict[str, NativeSearchVendor] = {}
    for name in _VENDOR_MODULES:
        vendor: NativeSearchVendor = import_module(f"{__name__}.{name}").VENDOR
        if vendor.id in vendors:
            raise RuntimeError(f"native search vendor {vendor.id!r} registered twice")
        vendors[vendor.id] = vendor
    return vendors


NATIVE_SEARCH_VENDORS: dict[str, NativeSearchVendor] = _load()

__all__ = ["NATIVE_SEARCH_VENDORS", "NativeSearchVendor"]
