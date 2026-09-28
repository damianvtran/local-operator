"""Monitors — a standing question to a read-only tool call.

A monitor re-runs one read-only call on an interval **with no model in the
loop**, normalizes the output, compares it against the last snapshot, and only
when something differs does it deliver a bounded delta into the conversation.
Ticks that find nothing cost no model tokens, inject no context, and write no
transcript row.

The package is split so each module stays importable where it is read:

- :mod:`local_operator.monitors.spec` — the spec model, id allocation and the
  arm-time request validator. Pydantic + stdlib only, because
  ``harness.types`` annotates the scheduler protocol with ``MonitorSpec``.
- :mod:`local_operator.monitors.settings` — the ``values.monitor`` snapshot.
- :mod:`local_operator.monitors.readonly` — the read-only evaluator (the
  safety core).
- :mod:`local_operator.monitors.diff` — normalization, hashing, bounded line
  deltas.
- :mod:`local_operator.monitors.classify` — the §8 classifier gate: the typed
  question, the bounded state, the fork mapping, and the adapter over the
  session's shared classification seam.
- :mod:`local_operator.monitors.state` — per-monitor counters + snapshot files.
- :mod:`local_operator.monitors.store` — the derived per-session index.
- :mod:`local_operator.monitors.scheduler` — the in-session scheduler.
- :mod:`local_operator.monitors.delivery` — the delivery envelope text.

The design contract is ``docs/design/monitor-tool.md``; section references in
these modules point at it.
"""
