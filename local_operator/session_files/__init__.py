"""Per-session file inventory: what files did this conversation (and its subagents) produce?

Three modules, one job each, mirroring ``local_operator.code_requests``:

* :mod:`~local_operator.session_files.scan` - PURE: journal lines in, file records out. No I/O,
  no clock, no HTTP; every rule is testable against a list of byte strings.
* :mod:`~local_operator.session_files.ledger` - the derived, signature-keyed, RESUMABLE scan
  cache under ``<config>/cache/session_files/``. It owns the journal I/O and the invalidation
  ladder (``session/transcript_index.py``'s, for the same reasons).
* :mod:`~local_operator.session_files.service` - the read side: merge a session's family
  (itself plus contained subagents), overlay Highlights, filter, sort, page, ``stat`` the
  returned page, and materialise a digest-only artifact to a real file on request.

The service has no HTTP coupling on purpose. The desktop route
(``server/routes/desktop_session_files.py``) is its first caller; the mobile daemon and the TUI
are expected to reuse it with ``display=True`` (cwd-relative / ``~/`` paths, memo S-R14) rather
than grow a second index. The package is deliberately EMPTY at import so a route module that
mounts it costs nothing at server start.
"""
