"""Install-layout defaults, in a module that costs nothing to import.

THE SAME PATTERN AS :mod:`local_operator.web_defaults`, FOR THE SAME REASON.
``DEFAULT_KEEP_GENERATIONS`` is an ``int`` that lives in
:mod:`local_operator.update`, a module that imports ``ssl``,
``urllib.request`` and ``http.client`` (measured at 47.5 ms of CPU here; the
comment it replaces priced the same import at ~28 ms on an older baseline). The
CLI parser needed that one integer at BUILD time — twice over, once for
``--version`` and once for ``--keep``'s ``%(default)s`` help text — so every
``lop`` invocation paid the whole update stack, including the verbs that install
nothing and never consult PyPI.

The value moves here and :mod:`local_operator.update` imports *from* this module
and re-exports it, so there is still exactly ONE definition: a caller that
imports ``DEFAULT_KEEP_GENERATIONS`` from ``update`` (the CLI dispatcher does,
inside its own lazy ``install`` import) is unaffected.

Keep this module import-cheap. It exists to be the cheap end of a value that a
heavy module also uses; a third-party import here re-introduces the cost it was
created to remove.
"""

from __future__ import annotations

#: How many unreferenced install generations ``lop install prune`` keeps by
#: default. The full retention RULE lives on
#: :func:`local_operator.update.prune_generations` — a generation is kept when
#: the pointer targets it or a live/persisted record names it, and this count is
#: only the margin for a session that has no record yet. Two rather than one
#: because the previous generation is exactly the one a just-flipped fleet is
#: still reading from.
DEFAULT_KEEP_GENERATIONS = 2
