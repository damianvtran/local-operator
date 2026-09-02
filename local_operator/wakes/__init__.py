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
- :mod:`.install` — the install-on-demand hook for the supervisor that reads
  that index and engages a runtime when a cold session's wake comes due. A
  no-op stub until the supervisor lands.

Everything here that the supervisor reads must stay import-light: the
supervisor is a ~40 MB always-on process whose whole justification is that it
does NOT carry the harness. See the module docstrings for the exact rule.
"""
