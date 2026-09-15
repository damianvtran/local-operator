"""Scheduled wakes outside a live session: the index and its supervisor.

A wake is owned by the session that created it and persisted in that
session's transcript (``wake_schedules`` custom entry, see
:mod:`local_operator.harness.wake`). That is the right home for the source of
truth but the wrong place to *find* wakes from outside: answering "which
sessions have a wake due?" would mean opening every transcript on the
machine. This package holds the pieces that make wakes discoverable and
fireable with no session process running:

- :mod:`.store` — the derived per-session index under
  ``<config_dir>/wakes/``, rewritten by the session on every schedule change
  and on every open.
- :mod:`.deliveries` — the supervisor's OWN ledger under
  ``<config_dir>/wakes/deliveries/``: one record per fire it attempted and has
  not yet handed to a runtime. This is what makes a undeliverable fire durable
  (retried with a backoff, never dropped on a schedule's due time moving on)
  and visible (``lop wake status`` reports it). It describes delivery
  attempts, never schedules — the transcript is still the only source of truth
  for what a schedule is.
- :mod:`.install` — the install-on-demand hook for the supervisor that reads
  that index and engages a runtime when a cold session's wake comes due. A
  no-op stub until the supervisor lands.

Everything here that the supervisor reads must stay import-light: the
supervisor is a ~40 MB always-on process whose whole justification is that it
does NOT carry the harness. See the module docstrings for the exact rule.
"""
