"""RFC §3.1's render-point helper: ``tr`` — text for a generated message.

The extraction slices resolve their copy through one of two shapes, and the
choice is measured, not stylistic (S1's comment in ``settings_io`` carries the
measurement — a per-message read put hundreds of file parses on that import):

* ``messages.render`` is the correctness floor: it PROBES for the message's
  namespace and READS the catalogue on every call. Right for one-off wire
  resolution, wrong for a module that resolves dozens of strings at import
  and a handful per frame.
* ``tr(namespace)`` is this module. It loads one namespace's catalogue ONCE
  per (root, namespace, locale) and returns a resolver, so a slice
  materialises its constants and its render-point strings without a file read
  per message::

      _t = tr("tui.welcome")
      TIP_PASTE = _t(tip_paste())

Both fall back to English when the selected catalogue or key is missing,
then degrade to the message CODE if English is missing or does not render — the
same additive contract ``messages.render`` documents — so a wheel without its catalogues loses the
copy, never the app.

**Locale is the process's ACTIVE locale**, not a pin: ``active_locale()``
starts at ``en`` (the only shipped locale while the program ships dark, RFC
§9 P1) and ``set_active_locale(locale)`` moves it — the re-resolution point
the T1 docstring reserved for the live-language slice (C). Resolvers read the
active locale on EVERY call, so a per-frame resolution follows a move with no
bookkeeping; a module-level CONSTANT (materialised once at import) does not,
and re-materialises through the refresh registry::

      def _reload_copy() -> None:
          global TIP_PASTE
          TIP_PASTE = _t(tip_paste())

      register_refresh(_reload_copy)

``set_active_locale`` runs every registered refresher when the locale
actually moves (a same-locale call is a no-op, so the idempotent paths — boot,
a config watcher echo — cost nothing). Registration order is import order;
the app's ``/language`` command and its config watcher are the callers.

The cache is keyed by the catalogue ROOT (as a string) — resolved at each
call, so a test that re-points ``catalogues.catalogue_root()`` resolves from
its own tree — by namespace, so the two modules that share a namespace (a TUI
module and the registry it reads from) share the one read, and by locale, so
a move reads the new tree's file once rather than per message.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable

from . import catalogues
from . import resolve as _locale
from . import runtime as _runtime
from .messages import Msg

_logger = logging.getLogger(__name__)

#: The process's active locale. A plain string, read by every resolver call:
#: the app owns WHEN it moves (boot resolution, ``/language``, the config
#: watcher), this module owns only the bookkeeping around a move.
_active_locale: str = "en"

#: Re-materialisers for module-level constants, in registration (import)
#: order. A module whose copy lives in constants registers ONE function that
#: rebinds them all; modules that resolve per frame register nothing.
_refreshers: list[Callable[[], None]] = []


def active_locale() -> str:
    """The locale every ``tr`` resolver currently resolves under."""
    return _active_locale


def set_active_locale(locale: str) -> None:
    """Move the process to ``locale``; re-materialise registered constants.

    Eligibility is checked before the no-op comparison: disk-only catalogues
    must not activate through this direct entrypoint. An unchanged effective
    locale runs no refreshers, keeping idempotent callers free.
    """
    global _active_locale
    locale = _locale.eligible_locale(locale)
    if locale == _active_locale:
        return
    _active_locale = locale
    for refresh in tuple(_refreshers):
        refresh()


def register_refresh(refresh: Callable[[], None]) -> None:
    """Register a module's constant re-materialiser for locale moves.

    ``set_active_locale`` iterates a SNAPSHOT of the list, so a refresher
    that (re-)registers during a move cannot mutate the list being walked.
    """
    _refreshers.append(refresh)


@functools.lru_cache(maxsize=64)
def _sources(root: str, namespace: str, locale: str) -> dict[str, str]:
    """One namespace's catalogue for one locale, read once per process key.

    Empty on a missing or malformed file: the resolver retries English before
    degrading to the code. Raw reads remain available for offline tooling;
    production eligibility belongs at the public entrypoints, not here.
    """
    try:
        return catalogues.load_catalogue(locale, namespace)
    except (catalogues.CatalogueNotFound, catalogues.CatalogueInvalid):
        return {}


def tr(namespace: str) -> Callable[[Msg], str]:
    """The resolver for one namespace's copy: ``_t = tr("tui.welcome")``.

    The namespace is the caller's to state, deliberately: a slice owns exactly
    one catalogue, and discovering it from each code would re-run the probe
    ``messages._namespace_for`` keeps for codes arriving off the wire, for a
    fact the call site already knows. Root and locale are read at each call,
    not captured here, so a resolver built before a catalogue re-point or a
    locale move still resolves from the tree and language in force now.
    """

    def resolve(message: Msg) -> str:
        # Recheck the operational switch on cached resolvers too: toggling it
        # must not leave previously selected translations live until a move.
        locale = _locale.eligible_locale(_active_locale)
        root = str(catalogues.catalogue_root())
        sources = _sources(root, namespace, locale)
        source = sources.get(message.code)
        source_locale = locale
        if source is None and locale != "en":
            source = _sources(root, namespace, "en").get(message.code)
            source_locale = "en"
        if source is None:
            return message.code
        if source_locale != locale:
            _logger.debug(
                "i18n fallback code=%s requested=%s source=%s",
                message.code,
                locale,
                source_locale,
            )
        try:
            return _runtime.render_message(source, dict(message.params), source_locale)
        except _runtime.MessageError:
            return message.code

    return resolve
