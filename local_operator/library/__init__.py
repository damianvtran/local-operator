"""The global, cross-conversation asset catalog.

The Library is deliberately separate from the per-session Files inventory:
that answers "what did this conversation produce?", this answers "where is
that thing I made months ago?". The split the design freezes is:

* ``metadata.py`` — the AUTHORITATIVE SQLite store (names, descriptions,
  tags, chats, conventions, edit log) under ``<config>/library/``;
* ``index.py`` — a DERIVED, discardable journal index under
  ``<config>/cache/library/`` rebuilt from ``session_files`` scans;
* ``service.py`` — query/projection/content over the two.

Nothing in this package writes user files. Metadata writes are local,
reversible and recorded with provenance (``user`` / ``librarian``).
"""

from __future__ import annotations
