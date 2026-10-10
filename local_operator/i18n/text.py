"""RFC §3.1's render-point helper: ``tr`` — text for a generated message.

The extraction slices resolve their copy through one of two shapes, and the
choice is measured, not stylistic (S1's comment in ``settings_io`` carries the
measurement — a per-message read put hundreds of file parses on that import):

* ``messages.render`` is the correctness floor: it PROBES for the message's
  namespace and READS the catalogue on every call. Right for one-off wire
  resolution, wrong for a module that resolves dozens of strings at import
  and a handful per frame.
* ``tr(namespace)`` is this module. It loads the namespace's ``en`` catalogue
  ONCE per process and returns a resolver, so a slice materialises its
  constants and its render-point strings without a file read per message::

      _t = tr("tui.welcome")
      TIP_PASTE = _t(tip_paste())

Both degrade to the message CODE when the catalogue or the message is
missing, or the message does not render — the same additive contract
``messages.render`` documents — so a wheel without its catalogues loses the
copy, never the app.

Locale is pinned to ``en``: the only shipped locale while the program ships
dark (RFC §9 P1), so a full resolution and this pin agree byte for byte. When
the translation waves land, the re-resolution point is this function — and
the clients, which render the served catalogue by key (§2.6).

The cache is keyed by the catalogue ROOT (as a string), so a test that
re-points ``catalogues.catalogue_root()`` resolves from its own tree, and by
namespace, so the two modules that share a namespace (a TUI module and the
registry it reads from) share the one read.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

from . import catalogues
from . import runtime as _runtime
from .messages import Msg


@functools.lru_cache(maxsize=64)
def _sources(root: str, namespace: str) -> dict[str, str]:
    """One namespace's ``en`` catalogue, read once per process.

    Empty on a missing or malformed file: the resolver below then degrades
    each of that namespace's messages to its own code, which is what keeps a
    broken wheel from making an import fail (the S1 failure model).
    """
    try:
        return catalogues.load_catalogue("en", namespace)
    except (catalogues.CatalogueNotFound, catalogues.CatalogueInvalid):
        return {}


def tr(namespace: str) -> Callable[[Msg], str]:
    """The resolver for one namespace's ``en`` copy: ``_t = tr("tui.welcome")``.

    The namespace is the caller's to state, deliberately: a slice owns exactly
    one catalogue, and discovering it from each code would re-run the probe
    ``messages._namespace_for`` keeps for codes arriving off the wire, for a
    fact the call site already knows.
    """
    root = str(catalogues.catalogue_root())
    sources = _sources(root, namespace)

    def resolve(message: Msg) -> str:
        source = sources.get(message.code)
        if source is None:
            return message.code
        try:
            return _runtime.render_message(source, dict(message.params), "en")
        except _runtime.MessageError:
            return message.code

    return resolve
