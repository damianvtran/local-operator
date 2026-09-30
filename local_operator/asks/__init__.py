"""Queued, non-blocking, timeout-bounded ``ask`` (design ``docs/design/ask-nonblocking.md``).

The package is split three ways on purpose, and the split is the import
contract rather than a filing convention:

* :mod:`local_operator.asks.store` — the durable log, the pure fold and the
  derived index. **Stdlib-only**, because its readers include a cold aggregate
  view and the cleanup sweep, neither of which may load the harness
  (``tests/unit/test_import_graph.py`` pins it, as it does ``wakes/store.py``).
* :mod:`local_operator.asks.policy` — the bounds, the flag and the timeout
  parse. Stdlib plus ``store``; no pydantic, no session.
* :mod:`local_operator.asks.render` — the ONE place the queue's text is
  written, so the model's turn and every card agree about the same ask.
* :mod:`local_operator.asks.queue` — ``AskQueue``, the stateful object a
  ``Session`` owns. The only module here that imports session-side types.

Nothing is re-exported here: an ``__init__`` that pulled ``queue`` in would put
pydantic and the harness on the path of ``import local_operator.asks.store``,
which is exactly the closure the import-graph pin exists to keep empty.
"""
