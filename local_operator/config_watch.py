"""Live propagation of ``config.yml`` edits to every running session.

The problem this solves
-----------------------

A setting written in one place did not reach any other running consumer. The
``/settings`` registry labelled five of nine sections ``NEW_SESSIONS`` and
the label was honest: ``compaction.*`` was coerced once at build,
``SessionStreamFn`` held the manager's ``values`` dict for the session's life,
``subagents.max_running`` was read once into ``AsyncJobManager``. A user with
three ``lop`` panes open who changed the compaction threshold in one of them
had to ``/new`` the other two — and the page had no way to tell them so beyond
the scope tag.

The mechanism
-------------

**One :class:`ConfigWatcher` per process**, stat-polled every
:attr:`ConfigWatcher.POLL_INTERVAL_S` seconds on the app's event loop, with an
in-process fast path from :mod:`local_operator.settings_io` writes and an
opportunistic kqueue wake-up on the config *directory* where the platform has
one. No daemon is involved. The numbers that shaped it (measured on the design
machine, ``/tmp/lop-settings-propagation/bench_stat.py``):

* ``os.stat`` of the file: **1 µs**. A full ``ConfigManager`` construct-and-load:
  **2 060 µs**. So the poll never parses; it stats and compares a fingerprint,
  and only a changed fingerprint pays for a parse.
* The real cost of polling is the loop WAKE (~70 µs CPU), not the stat. At 50
  processes on a 2 s cadence that is 0.18 % of one core — cheaper than the
  mobile daemon's own 2 s registry scan that already runs on this machine.
* A Unix-socket round trip to that daemon would cost 49–644 µs per poll, needs
  the daemon to be running (it is optional), and the daemon would still have to
  detect the change by stat or kqueue itself. Polling the daemon adds a hop and
  a reconnect state machine and removes nothing; rejected.
* A directory kqueue delivered **0 events in 20 s** on the real config
  directory with 74 ``lop`` processes alive — the SQLite WAL churn is inside
  file inodes, not directory entries — so the accelerator is quiet, not noisy.

Two facts about the write path shape everything here:

* **Every write is an atomic ``os.replace``** of a same-directory temp file
  (``ConfigManager._write_config``), so the file's inode changes on every
  write. That is why the fingerprint includes ``st_ino`` (a same-size write
  within one mtime tick still moves it) and why the kqueue watches the
  DIRECTORY rather than the file: a watch on the file's inode reports one
  ``KQ_NOTE_DELETE`` and then nothing, forever.
* **``ConfigManager._load_config`` degrades a malformed file to defaults and
  moves it aside** to ``config.yml.bad.<stamp>``. A live reloader that went
  through it would turn a half-typed hand edit into a destroyed config on the
  next tick in every open session. The watcher therefore parses the raw bytes
  itself under the same rules ``settings_io._require_readable_config``
  applies before a write, keeps the last good snapshot when they fail, and
  never constructs a ``ConfigManager`` at all.

Threading contract
------------------

``_tick`` and listener fan-out run on the loop thread only. :meth:`notify_local`
may be called from anywhere — the ``SettingsView`` writes on the loop, but a
tool-driven ``settings_io`` write could be on a worker — so it hops through
``call_soon_threadsafe`` when it is off-loop, and simply records-and-drops when
no loop was ever started (the CLI's ``config edit`` path, which has no watcher
to notify). :attr:`values` may be read from any thread: the snapshot swap is
one attribute assignment of a dict nobody mutates afterwards, which is atomic
under the GIL.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import os
import sys
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import yaml

logger = logging.getLogger("local_operator.config_watch")

#: ``(st_ino, st_size, st_mtime_ns)`` — what a tick compares. See the module
#: docstring for why the inode is part of it.
Fingerprint = tuple[int, int, int]

#: A subscriber. Synchronous, called on the loop thread, one call per
#: :class:`ConfigChange`. A raising listener is logged and never propagated
#: (same contract as ``Session._emit``), so one bad subscriber cannot starve
#: the others of a change.
Listener = Callable[["ConfigChange"], None]

#: The name of the file every fingerprint and parse is taken from. Restated
#: rather than imported from :mod:`local_operator.config` so this module has
#: no import-time dependency on the manager it exists to bypass.
_CONFIG_FILE_NAME = "config.yml"


@dataclass(frozen=True)
class ConfigChange:
    """One delivered change: the new values, what moved, and who moved it.

    ``values`` is the watcher's own deep copy of ``config.yml``'s ``values``
    mapping — read-only by contract. Listeners that need a typed object
    (``CompactionSettings``) build it from here; listeners that need the whole
    mapping (``SessionStreamFn``) rebind to it.

    ``changed_keys`` names REGISTRY keys (``settings_io.SETTINGS`` ``.key``
    spellings such as ``compaction.threshold_percent``), not YAML paths, and is
    computed per key rather than per file: every write also bumps
    ``metadata.last_modified`` and a no-op write must not produce a
    notification. It is what the TUI prints in its notice, so it speaks the
    same vocabulary as ``/settings`` and ``lop config edit``.

    ``source`` tells a listener whether THIS process made the write
    (``"local"``: the page or CLI here already showed its own result, so the
    TUI stays quiet) or another one did (``"disk"``: name the keys so the user
    knows why behaviour just changed under them).
    """

    values: Mapping[str, Any]
    changed_keys: frozenset[str]
    source: Literal["disk", "local"]


class ConfigWatcher:
    """Stat-poll ``config.yml`` and fan out per-key changes to subscribers.

    Construct one per config directory per process — :func:`process_watcher`
    is the canonical way to get it — and :meth:`start` it on the loop the
    sessions run on. Everything else is driven from the tick.
    """

    #: The poll cadence. 2 s is the latency a user perceives as "it just
    #: applied" when they tab between panes, and at 70 µs per wake the cost is
    #: invisible even at 50 processes (module docstring). 5 s would halve a
    #: cost that is already ~0.2 % of a core and make the cross-pane change
    #: feel laggy; sub-second would spend real wakes on an event that happens
    #: a few times a day. Not a tuning knob.
    POLL_INTERVAL_S = 2.0

    def __init__(self, config_dir: Path, *, interval: float = POLL_INTERVAL_S) -> None:
        self._config_dir = Path(config_dir)
        self._config_file = self._config_dir / _CONFIG_FILE_NAME
        self._interval = float(interval)
        self._listeners: list[Listener] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task[None] | None = None
        # The fingerprint of the file the CURRENT snapshot was parsed from. A
        # tick whose stat equals this returns without reading a byte.
        self._fingerprint: Fingerprint | None = None
        # The fingerprint of a file that FAILED to parse. Remembered so an
        # unreadable file is parsed once, not every tick: a user mid-edit in
        # vim would otherwise cost every open session a YAML parse twice a
        # second until they saved something valid. Cleared as soon as the
        # fingerprint moves again.
        self._bad_fingerprint: Fingerprint | None = None
        # kqueue accelerator state (darwin/BSD only; ``None`` elsewhere or when
        # arming failed). Both descriptors are closed by :meth:`stop`.
        self._kqueue: Any = None
        self._dir_fd: int | None = None
        # The last good snapshot: THE process's canonical live view of
        # ``values``. Seeded synchronously here so ``values`` is meaningful
        # before the first tick and so a listener subscribed before ``start``
        # sees the same baseline the first diff is taken against. A missing
        # file IS the defaults — exactly what ``ConfigManager._load_config``
        # yields for one — so the first write to a fresh install diffs against
        # the defaults it back-fills rather than announcing thirty keys.
        self._values: Mapping[str, Any] = _with_defaults({})
        self._prime()

    # -- public surface -----------------------------------------------------

    @property
    def config_dir(self) -> Path:
        return self._config_dir

    @property
    def values(self) -> Mapping[str, Any]:
        """The last good ``values`` mapping. Read-only by contract.

        Safe from any thread: the swap in :meth:`_adopt` is a single attribute
        assignment of a mapping that is never mutated afterwards.
        """
        return self._values

    def subscribe(self, listener: Listener) -> Callable[[], None]:
        """Register ``listener``; returns its unsubscribe.

        Idempotent to unsubscribe twice — a session disposed through two paths
        (its own dispose hook and a subagent's ``_dispose_child``) must not
        raise on the second call.
        """
        self._listeners.append(listener)

        def unsubscribe() -> None:
            try:
                self._listeners.remove(listener)
            except ValueError:
                pass

        return unsubscribe

    def start(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Begin polling on ``loop`` (default: the running loop). Idempotent.

        Also arms the kqueue accelerator where available. A second call while
        the task is alive is a no-op, which is what lets every
        ``create_session`` call it unconditionally: the first session in the
        process starts the watcher, the ``/new`` that follows finds it running.

        A call that gets PAST that guard — the task died with its loop, and
        nobody called :meth:`stop` — must release the previous accelerator
        before arming a new one (review round 1, M4). Without this,
        ``_arm_kqueue`` overwrites ``_kqueue``/``_dir_fd`` while the old
        descriptors are still open and unreferenced, so a later ``stop()`` can
        only close the newest pair: measured at four directory fds
        (``[6, 8, 7, 9]``) across four loops, three still open afterwards.
        Production runs one loop per process, so this is a latent leak rather
        than a live one — but ``start`` is called unconditionally from four
        sites and EMFILE is the failure mode the registry's reaping already
        exists to prevent, which is too small a margin to leave a known leak
        inside.
        """
        if self._task is not None and not self._task.done():
            return
        self._disarm_kqueue()
        if loop is None:
            loop = asyncio.get_running_loop()
        self._loop = loop
        # Re-fingerprint before the first tick so a change that landed between
        # construction and start is delivered rather than silently adopted as
        # the baseline. ``poll_now`` returns it to the caller too; nothing here
        # needs the value.
        self.poll_now()
        self._task = loop.create_task(self._run(), name="config-watch")
        self._arm_kqueue(loop)

    async def stop(self) -> None:
        """Cancel the poll task and release the kqueue descriptors."""
        task = self._release()
        if task is not None and not task.done():
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001 — teardown proceeds
                pass

    def _release(self) -> "asyncio.Task[None] | None":
        """Synchronous teardown: descriptors closed, poll task cancelled.

        The part of :meth:`stop` that needs no loop, split out so the registry
        can reap a watcher whose loop is already gone (see
        :func:`process_watcher`). Returns the cancelled task for a caller that
        can await its quietus.
        """
        self._disarm_kqueue()
        task, self._task = self._task, None
        if task is not None and not task.done():
            loop = self._loop
            if loop is not None and not loop.is_closed():
                task.cancel()
        return task

    @property
    def is_idle(self) -> bool:
        """True when nothing can ever hear from this watcher again.

        No listeners, or a loop that has closed (its subscribers died with
        it). The registry uses this to bound the fds a process holds.
        """
        loop = self._loop
        return not self._listeners or (loop is not None and loop.is_closed())

    def poll_now(self) -> ConfigChange | None:
        """One synchronous tick: stat, parse if moved, diff, fan out.

        Returns the change it delivered (or ``None``). The write fast path and
        the tests use it directly; the poll task calls it every interval.
        """
        return self._tick(source="disk")

    def notify_local(self, values: Mapping[str, Any] | None = None) -> None:
        """The in-process write fast path.

        Called by ``settings_io._store``/``_delete`` right after a write lands.
        Re-stats and re-parses the file (rather than trusting the ``values``
        the writer passes, which is the writer's manager's view and may carry
        keys ``_load_config`` back-filled that are not on disk), records the new
        fingerprint so the next poll tick is a no-op, and fans out with
        ``source="local"`` — SYNCHRONOUSLY when called on the loop thread, so a
        ``/settings`` toggle of ``compaction.enabled`` reaches the writer's own
        session on the same call stack the page runs on.

        Off the loop thread it hops through ``call_soon_threadsafe``: listeners
        touch session state and widgets that are loop-affine. On a platform
        with the kqueue accelerator the directory event can wake the loop and
        deliver the change as ``"disk"`` before that hop lands — the change
        still arrives exactly once and on the loop thread, only its ``source``
        is then the conservative one (the TUI prints a line it could have
        skipped). Accepted: no production writer runs off-loop today, and the
        alternative is a pre-write handshake in ``settings_io`` for a cosmetic
        difference. With no loop started (the CLI's ``config edit``) it just
        re-reads so ``values`` stays current for anyone who asks, and delivers
        to nobody — there is nobody.

        ``values`` is accepted for API symmetry with the design and ignored for
        the reason above; the file is the source of truth.
        """
        loop = self._loop
        if loop is None or loop.is_closed():
            self._tick(source="local")
            return
        if _on_loop(loop):
            self._tick(source="local")
            return
        try:
            loop.call_soon_threadsafe(self._tick, "local")
        except RuntimeError:
            # The loop closed between the check and the call. Nothing to
            # deliver to; keep the snapshot honest for any late reader.
            self._tick(source="local")

    # -- the tick -----------------------------------------------------------

    def _stat(self) -> Fingerprint | None:
        """The file's fingerprint, or ``None`` when it does not exist.

        A missing file is a first run, not an error: ``_require_readable_config``
        treats it the same way, and the watcher must too or a fresh install
        would log a failure on every tick until the first write.
        """
        try:
            st = os.stat(self._config_file)
        except FileNotFoundError:
            return None
        except OSError:
            # Permissions or I/O trouble. Not a config change; leave the
            # snapshot alone and try again next tick.
            return self._fingerprint
        return (st.st_ino, st.st_size, st.st_mtime_ns)

    def _prime(self) -> None:
        """Take the initial snapshot without notifying anyone."""
        fingerprint = self._stat()
        if fingerprint is None:
            self._fingerprint = None
            return
        parsed = self._parse()
        if parsed is None:
            self._bad_fingerprint = fingerprint
            return
        self._fingerprint = fingerprint
        self._values = parsed

    def _tick(self, source: Literal["disk", "local"] = "disk") -> ConfigChange | None:
        fingerprint = self._stat()
        if fingerprint == self._fingerprint:
            return None
        if fingerprint is None:
            # The file was removed. The last good snapshot stays the live
            # view — a deleted config is the defaults on the NEXT launch, not a
            # reason to reconfigure every running session — but the
            # fingerprint is cleared so the file's return is seen as a change.
            self._fingerprint = None
            self._bad_fingerprint = None
            return None
        if fingerprint == self._bad_fingerprint:
            # Already known unreadable; do not re-parse until it moves.
            return None
        parsed = self._parse()
        if parsed is None:
            self._bad_fingerprint = fingerprint
            return None
        self._bad_fingerprint = None
        previous = self._values
        self._fingerprint = fingerprint
        changed = _changed_registry_keys(previous, parsed)
        self._adopt(parsed)
        if not changed:
            return None
        change = ConfigChange(values=self._values, changed_keys=changed, source=source)
        self._fan_out(change)
        return change

    def _parse(self) -> Mapping[str, Any] | None:
        """``values`` from the raw bytes, or ``None`` when the file is unreadable.

        Mirrors ``settings_io._require_readable_config`` exactly and on purpose
        — a YAML syntax error, a non-mapping top level, an empty file, and
        non-UTF-8 bytes are the shapes ``_load_config`` would degrade on, and
        degrading is what this must never do. Unlike that guard this returns
        rather than raises, because the watcher's answer to "unreadable" is
        "keep the last good snapshot", not "abort".

        The ``values`` block is DEEP-COPIED so the snapshot cannot alias
        anything a later parse or a consumer's own mutation could touch.
        """
        try:
            raw = self._config_file.read_text(encoding="utf-8")
        except (FileNotFoundError, OSError, UnicodeDecodeError) as error:
            logger.debug("config.yml not readable: %s", error)
            return None
        if not raw.strip():
            logger.debug("config.yml is empty; keeping the last good settings")
            return None
        try:
            loaded = yaml.safe_load(raw)
        except yaml.YAMLError as error:
            logger.debug("config.yml did not parse; keeping the last good settings: %s", error)
            return None
        if loaded is None:
            return None
        if not isinstance(loaded, Mapping):
            logger.debug(
                "config.yml top level is %s, not a mapping; keeping the last good settings",
                type(loaded).__name__,
            )
            return None
        values = loaded.get("values")
        if not isinstance(values, Mapping):
            # A file with a mapping top level but no usable ``values`` reads
            # as "nothing set", which ``_load_config`` resolves to the
            # defaults; so does this.
            values = {}
        return _with_defaults(values)

    def _adopt(self, values: Mapping[str, Any]) -> None:
        # ONE assignment: this is the cross-thread publication point for
        # ``values`` (see the class docstring).
        self._values = values

    def _fan_out(self, change: ConfigChange) -> None:
        for listener in list(self._listeners):
            try:
                listener(change)
            except Exception:  # noqa: BLE001 — one listener must not starve the rest
                logger.warning("config change listener failed", exc_info=True)

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self._interval)
            try:
                self._tick(source="disk")
            except Exception:  # noqa: BLE001 — the poller must outlive any one bad tick
                logger.warning("config watch tick failed", exc_info=True)

    # -- kqueue accelerator -------------------------------------------------

    def _arm_kqueue(self, loop: asyncio.AbstractEventLoop) -> None:
        """Wake the tick on a directory write, where the platform can say so.

        BSD/macOS only (``select.kqueue``); Linux keeps the poll alone, which
        gives ≤2 s latency — inotify is not in the stdlib and ``watchdog`` was
        judged not worth a native dependency for a few-times-a-day event.
        Watches the DIRECTORY because every write is an ``os.replace`` and the
        file's inode dies with it (module docstring). Any failure — EMFILE, a
        filesystem without vnode events, a missing directory on first run — is
        silent: the poll is the primary mechanism and this only trims latency.
        """
        import select

        # ``sys.platform == "darwin"`` rather than ``hasattr(select,
        # "kqueue")``: both are correct at RUNTIME, but only the equality form
        # NARROWS for the type checker. CI type-checks on Linux, where
        # typeshed's ``select`` stub genuinely has no
        # ``kqueue``/``kevent``/``KQ_*``, and a hasattr guard therefore left
        # seven ``reportAttributeAccessIssue`` errors on a gate that is green
        # on a macOS dev box — exactly the pass-locally/fail-on-CI shape
        # AGENTS.md warns about. A tuple membership test (``not in (...)``) is
        # not a narrowing form pyright recognises either; the literal
        # comparison is.
        #
        # This costs the BSDs the accelerator even though they have kqueue.
        # That is the deliberate trade and it is cheap: the accelerator only
        # trims LATENCY, so a FreeBSD user gets the same ≤2 s poll Linux gets
        # rather than sub-second, and no behaviour differs. The ``hasattr``
        # below stays as the runtime belt for a darwin build without kqueue.
        if sys.platform != "darwin":
            return
        if not hasattr(select, "kqueue"):
            return
        try:
            flags = os.O_RDONLY | getattr(os, "O_EVTONLY", 0)
            dir_fd = os.open(self._config_dir, flags)
        except OSError:
            return
        try:
            kq = select.kqueue()
            event = select.kevent(
                dir_fd,
                filter=select.KQ_FILTER_VNODE,
                flags=select.KQ_EV_ADD | select.KQ_EV_CLEAR,
                fflags=select.KQ_NOTE_WRITE | select.KQ_NOTE_ATTRIB,
            )
            kq.control([event], 0, 0)
            loop.add_reader(kq.fileno(), self._on_kqueue_readable)
        except Exception:  # noqa: BLE001 — accelerator is optional
            os.close(dir_fd)
            return
        self._kqueue = kq
        self._dir_fd = dir_fd

    def _on_kqueue_readable(self) -> None:
        kq = self._kqueue
        if kq is None:
            return
        try:
            # Drain: EV_CLEAR resets the state after each read, but several
            # writes may have coalesced while the loop was busy.
            kq.control(None, 16, 0)
        except Exception:  # noqa: BLE001 — a failed drain still ticks
            pass
        try:
            self._tick(source="disk")
        except Exception:  # noqa: BLE001
            logger.warning("config watch tick failed", exc_info=True)

    def _disarm_kqueue(self) -> None:
        kq, self._kqueue = self._kqueue, None
        dir_fd, self._dir_fd = self._dir_fd, None
        loop = self._loop
        if kq is not None:
            if loop is not None and not loop.is_closed():
                try:
                    loop.remove_reader(kq.fileno())
                except Exception:  # noqa: BLE001
                    pass
            try:
                kq.close()
            except Exception:  # noqa: BLE001
                pass
        if dir_fd is not None:
            try:
                os.close(dir_fd)
            except OSError:
                pass


def _with_defaults(values: Mapping[str, Any]) -> dict[str, Any]:
    """``values`` deep-copied, with missing TOP-LEVEL keys back-filled.

    Mirrors the one normalisation ``ConfigManager._load_config`` applies so
    the watcher's view of a file equals a manager's view of the same file:
    a consumer switching from ``manager.get_config().values`` to
    ``watcher.values`` must not start seeing ``retry`` as absent because the
    user's file predates the key. Top level only, as in ``_load_config`` —
    nested blocks are the consumers' own coercers' business
    (``RetrySettings.from_settings``, ``CompactionSettings``).

    ``DEFAULT_CONFIG`` is the module-level constant, imported lazily so this
    module stays importable before the config module's own imports settle,
    and deep-copied because that object is shared process-wide and a
    consumer mutating a nested default through the snapshot would poison
    every later parse.
    """
    from local_operator.config import DEFAULT_CONFIG

    merged: dict[str, Any] = copy.deepcopy(dict(values))
    for key, default in DEFAULT_CONFIG.values.items():
        if key not in merged:
            merged[key] = copy.deepcopy(default)
    return merged


def _on_loop(loop: asyncio.AbstractEventLoop) -> bool:
    """Whether the caller is executing on ``loop``'s thread right now."""
    try:
        return asyncio.get_running_loop() is loop
    except RuntimeError:
        return False


def _changed_registry_keys(before: Mapping[str, Any], after: Mapping[str, Any]) -> frozenset[str]:
    """Registry keys whose stored value differs between two ``values`` mappings.

    Walks ``settings_io.SETTINGS`` with its own ``_walk`` so the comparison
    uses exactly the paths the page and the CLI use — including the flat
    ``display.*`` keys whose dot is literal. ~55 keys, microseconds. A key
    absent on both sides is unchanged; absent on one side is changed (an unset
    is a real edit the consumers must see, since it means "back to default").
    """
    from local_operator.settings_io import _MISSING, SETTINGS, _walk

    changed: set[str] = set()
    for setting in SETTINGS:
        old = _walk(before, setting.path)
        new = _walk(after, setting.path)
        if old is _MISSING and new is _MISSING:
            continue
        if old is _MISSING or new is _MISSING or old != new:
            changed.add(setting.key)
    return frozenset(changed)


# ---------------------------------------------------------------------------
# The per-process singleton
# ---------------------------------------------------------------------------

_watchers: dict[Path, ConfigWatcher] = {}
_watchers_lock = threading.Lock()


def process_watcher(config_dir: Path | None = None) -> ConfigWatcher:
    """The one watcher for ``config_dir`` (default ``paths.config_dir()``).

    Keyed on the directory rather than a bare module global because tests and
    the exec worker point ``LOCAL_OPERATOR_CONFIG_DIR`` at scratch directories,
    and a watcher on the wrong directory would silently watch nothing. Created
    lazily and never started here: ``start`` needs a loop, and the factory
    that knows the loop calls it.
    """
    if config_dir is None:
        from local_operator.paths import config_dir as _config_dir

        config_dir = _config_dir()
    key = Path(config_dir)
    with _watchers_lock:
        watcher = _watchers.get(key)
        if watcher is None:
            # Reap watchers on OTHER directories that can never deliver again
            # before adding one. A production process has exactly one config
            # directory for its whole life, so this never fires there; a
            # process that moves between directories (the test suite: one
            # ``tmp_path`` per test, thousands per worker) would otherwise
            # accumulate a directory fd and a kqueue per directory until
            # EMFILE. Reaping only the IDLE ones keeps a live session's
            # watcher on a second directory intact.
            for other_key, other in list(_watchers.items()):
                if other.is_idle:
                    other._release()
                    del _watchers[other_key]
            watcher = ConfigWatcher(key)
            _watchers[key] = watcher
        return watcher


def existing_watcher(config_dir: Path | None = None) -> ConfigWatcher | None:
    """The watcher for ``config_dir`` if one was ever created, else ``None``.

    The write fast path in ``settings_io`` uses this rather than
    :func:`process_watcher` so a CLI process that only ever writes does not
    build a watcher it will never start.
    """
    if config_dir is None:
        from local_operator.paths import config_dir as _config_dir

        config_dir = _config_dir()
    with _watchers_lock:
        return _watchers.get(Path(config_dir))


def _reset_for_tests() -> None:
    """Drop every registered watcher. Tests only."""
    with _watchers_lock:
        _watchers.clear()


__all__ = [
    "ConfigChange",
    "ConfigWatcher",
    "Fingerprint",
    "Listener",
    "existing_watcher",
    "process_watcher",
]
