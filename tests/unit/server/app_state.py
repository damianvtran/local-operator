"""Snapshot-and-restore of the module-level ``app`` state for tests.

WHY this exists: ``local_operator.server.app.app`` is a process-wide singleton,
so several test files snapshot ``app.state`` before a test and put it back
afterwards. A plain snapshot is not enough for state the lifespan CLOSES on its
way out. ``app.state.desktop_auth`` is created lazily by the first request that
needs a provider credential (``routes/auth.get_desktop_auth``) and the lifespan
teardown closes it (``app.py``: ``await desktop_auth.close()``). When an earlier
test in the worker had already left a live host there, a lifespan-running test
closed THAT host and the snapshot then restored the closed object, so the next
``/v1/models`` request died with ``sqlite3.ProgrammingError: Cannot operate on
a closed database`` (merge-gate run 44354652248, shard 5:
``test_server_models`` after ``test_serve_retire``).

The constraint: production never reaches this state. A server process runs one
lifespan, which closes the host it created and sets the attribute to ``None``,
and the next request builds a fresh one. Only the test-side snapshot resurrects
a closed handle, so the fix lives here, not in the product.

Usage: wrap a test in ``restored_app_state(app)`` (the fixtures in
``test_serve_lifecycle``, ``test_serve_retire`` and ``test_desktop_aside_route``
do). Hosts listed in ``LIFESPAN_CLOSED_HOSTS`` are detached for the duration, so
a lifespan inside the test only ever closes a host the test itself created, and
the caller's live host comes back untouched.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

#: Lazily created ``app.state`` hosts that the lifespan teardown closes. Keep in
#: step with the shutdown block of ``local_operator.server.app.lifespan``.
LIFESPAN_CLOSED_HOSTS = ("desktop_auth",)


@contextmanager
def restored_app_state(app: Any) -> Iterator[None]:
    """Give the process back the ``app`` state it had before the block.

    Taken through Starlette's mapping interface (``__iter__`` +
    ``__getitem__``), NOT ``vars(app.state)``: that returns ``{'_state': …}``,
    whose value is the same dict the app keeps writing into, so a snapshot taken
    that way would restore the test's own changes. Deliberately blunt and total:
    a hand-written list of keys is what goes stale as the lifespan grows.
    """
    state = app.state
    saved = {key: state[key] for key in state}
    for key in LIFESPAN_CLOSED_HOSTS:
        if key in saved:
            del state[key]
    try:
        yield
    finally:
        for key in list(state):
            del state[key]
        for key, value in saved.items():
            state[key] = value
