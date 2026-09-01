"""The session runtime: one live session, reachable over a socket.

The *runtime* is the process that owns a running session and makes it
addressable by something other than whatever started it: it publishes a
discovery record (:mod:`.registry`), listens on an authenticated loopback
control socket (:mod:`.server`), applies incoming requests to the session
through a :class:`~local_operator.session.runtime.server.SessionHandle`, and —
when nothing interactive owns it — is the whole process (:mod:`.process`).

The counterpart term is a **viewer**: a TUI or a phone attached to a runtime.
A viewer observes and steers; it does not own the computation. That
distinction is the point of the vocabulary, and the reaper in
:mod:`.process` already encodes it — viewer count is deliberately not an input
to whether an idle runtime exits.

**Why this package is neutral rather than mobile-scoped.** All of it grew up
under ``local_operator/mobile/`` because the phone was the first viewer that
needed to reach into a live session, but none of it is about the phone. The
record, the socket, the client-kind separation and the owned-session handle
describe *one session with zero or more attached viewers*; the phone daemon is
one viewer, an attach terminal (``lop attach``) is another, and autonomous
wakes and background automations are the next. Leaving it under ``mobile/``
forced every non-phone consumer to import a front end it does not use, and
made the phone look like the owner of a mechanism it merely borrows. Moving it
here says the real thing: there is ONE session-runtime concept, and the phone
is a viewer of it.

**Nothing here changed behaviour in the move.** Every literal — the record
directory, the heartbeat intervals, the protocol version, the spawn
environment variables, the control thread's name — is byte-identical to what
``mobile/`` published, because two binaries of different versions coexist in
running processes on the same machine and must keep seeing each other's
sessions. See :mod:`.types` for the ``run/mobile`` note specifically.

The old module paths (``local_operator.mobile.registrant``, ``.registry``,
``.owned``, ``.child``) remain as thin re-export shims, and ``Registrant``
remains an alias of
:class:`~local_operator.session.runtime.server.RuntimeServer`.

This ``__init__`` deliberately exports NOTHING and imports NOTHING. The
runtime sits on the CLI startup path, where
``import local_operator.session.runtime.types`` must not drag in :mod:`.server`
(asyncio) or :mod:`.owned` (the composition root).
``tests/unit/test_import_graph.py`` is the guard; import the submodule you
need directly.
"""
