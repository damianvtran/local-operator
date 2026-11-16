"""Failure-only evidence for ONE stale-handoff integration test, not recovery policy.

The historic CI cause is unknown. These wrappers observe existing calls without
retries, sleeps, tracing, new threads or journal reads. The operator requested
128 critical events / 32 KiB output (not a product timeout or runtime policy).
Only synthetic root/instance aliases, fixed event labels, exception *types* and
allowlisted source locations leave memory; never arguments, payloads or locals.
"""

from __future__ import annotations

import json
import sys
import threading
from functools import wraps
from pathlib import Path
from typing import Any, Callable

MAX_EVENTS = 128
MAX_BYTES = 32 * 1024
TARGET = "test_a_relay_start_recovers_a_stale_handoff_and_frees_the_owner_s_session"
_UNAVAILABLE = "mobility failure diagnostics unavailable (observer failure)\n"
_FRAME_FILES = frozenset(
    {
        "local_operator/network/mobility.py",
        "local_operator/network/relay.py",
        "local_operator/session/cleanup.py",
        "local_operator/session/placement.py",
    }
)


def emit_before_cleanup(observer: Any, failure: BaseException | None) -> None:
    """Even a broken observer/sink must preserve the original exception object."""
    if failure is None:
        return
    try:
        observer.before_cleanup(failure)
    except BaseException:  # noqa: BLE001 — diagnostics cannot replace the failure
        try:
            sys.stderr.write(_UNAVAILABLE)
        except BaseException:  # noqa: BLE001 — a broken output stream is not the test failure
            pass


def observe_failure(function: Callable[..., Any]) -> Callable[..., Any]:
    """Catch failures outside the inner finally too, before fixture teardown."""

    @wraps(function)
    def observed(*args: Any, **kwargs: Any) -> Any:
        try:
            return function(*args, **kwargs)
        except BaseException:
            emit_before_cleanup(kwargs.get("mobility_failure_observer"), sys.exc_info()[1])
            raise

    return observed


class MobilityFailureObserver:
    """Bounded, thread-safe state; successful tests never render or write it."""

    def __init__(self, sink: Callable[[str], Any] | None = None) -> None:
        self._lock = threading.Lock()
        self._events: list[dict[str, Any]] = []
        self._roots: dict[Path, str] = {}
        self._instances: dict[str, str] = {}
        self._workers: dict[int, tuple[str, str]] = {}
        self._dropped = 0
        self._observer_errors = 0
        self._sealed = False
        self._sink = sink or (lambda text: sys.stderr.write(text))

    def _safe(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        try:
            return operation(*args, **kwargs)
        except BaseException:  # noqa: BLE001 — observer defects are never product defects
            with self._lock:
                self._observer_errors += 1
            return None

    def _context(self, root: Path, instance: str = "") -> tuple[str, str]:
        with self._lock:
            # Alias tables are bounded too; event overflow cannot retain paths/IDs.
            if root not in self._roots and len(self._roots) < MAX_EVENTS:
                self._roots[root] = f"root-{len(self._roots) + 1}"
            if instance and instance not in self._instances and len(self._instances) < MAX_EVENTS:
                self._instances[instance] = f"instance-{len(self._instances) + 1}"
            return self._roots.get(root, "root-overflow"), self._instances.get(instance, "none")

    def record(self, event: str, context: tuple[str, str], error: str = "") -> None:
        with self._lock:
            if self._sealed:
                return
            if len(self._events) >= MAX_EVENTS:
                self._dropped += 1
                return
            self._events.append(
                {"event": event, "root": context[0], "instance": context[1], "type": error}
            )

    def _worker(self, entering: bool, context: tuple[str, str]) -> None:
        with self._lock:
            ident = threading.get_ident()
            if entering and not self._sealed and len(self._workers) < MAX_EVENTS:
                self._workers[ident] = context
            elif not entering:
                self._workers.pop(ident, None)

    def wrap(
        self,
        function: Callable[..., Any],
        label: str,
        context: Callable[..., tuple[str, str]],
        *,
        worker: bool = False,
    ) -> Callable[..., Any]:
        """Original execution/return/raise occurs exactly once, outside our lock."""

        @wraps(function)
        def observed(*args: Any, **kwargs: Any) -> Any:
            location = self._safe(context, *args, **kwargs) or ("unknown", "none")
            if worker:
                self._safe(self._worker, True, location)
            self._safe(self.record, f"{label}.begin", location)
            try:
                result = function(*args, **kwargs)
            except BaseException as exc:
                self._safe(self.record, f"{label}.exception", location, type(exc).__name__)
                raise
            else:
                self._safe(self.record, f"{label}.complete", location)
                return result
            finally:
                if worker:
                    self._safe(self.record, "worker.exit", location)
                    self._safe(self._worker, False, location)

        return observed

    def install(self, monkeypatch: Any) -> None:
        """Install before pair constructs either relay; no product files change."""
        from local_operator.network import mobility, relay
        from local_operator.session import cleanup, placement

        def server_context(server: Any, *args: Any, **kwargs: Any) -> tuple[str, str]:
            return self._context(Path(server.root), server.instance_id)

        def root_context(root: Any, *args: Any, **kwargs: Any) -> tuple[str, str]:
            # Nested operations inherit the exact worker instance, not a root guess.
            with self._lock:
                active = self._workers.get(threading.get_ident())
            return active or self._context(Path(root))

        def journal_context(path: Path, *args: Any, **kwargs: Any) -> tuple[str, str]:
            # Only the config root is aliased; the journal filename is private.
            return root_context(Path(path).parent.parent)

        original_register = relay.RelayServer.register_ops

        @wraps(original_register)
        def register(server: Any, *args: Any, **kwargs: Any) -> Any:
            if "mobility-recovery" not in (kwargs.get("on_start") or {}):
                return original_register(server, *args, **kwargs)
            return self.wrap(original_register, "hook.registration", server_context)(
                server, *args, **kwargs
            )

        monkeypatch.setattr(relay.RelayServer, "register_ops", register)
        monkeypatch.setattr(
            mobility,
            "_start_recovery",
            self.wrap(mobility._start_recovery, "worker", server_context, worker=True),
        )
        for module, name, label in (
            (cleanup, "sweep_handed_away", "cleanup"),
            (mobility, "sweep_staging", "staging"),
            (mobility, "reconcile", "reconcile"),
            (placement, "_write_journal", "journal.write"),
        ):
            get_context = journal_context if name == "_write_journal" else root_context
            monkeypatch.setattr(module, name, self.wrap(getattr(module, name), label, get_context))

    def _frames(self, workers: dict[int, tuple[str, str]]) -> tuple[list[dict[str, Any]], bool]:
        if not workers:
            return [], False
        # CPython only offers a process-wide map. Immediately select tracked hook
        # IDs; do not inspect/format any unrelated thread, locals or exception data.
        current = sys._current_frames()
        selected = [(current.get(ident), context) for ident, context in workers.items()]
        del current
        frames = []
        truncated = False
        repo = Path(__file__).resolve().parents[3]
        for frame, context in selected:
            for _ in range(MAX_EVENTS):
                if frame is None or len(frames) >= MAX_EVENTS:
                    break
                try:
                    relative = Path(frame.f_code.co_filename).relative_to(repo).as_posix()
                except ValueError:
                    relative = ""
                if relative in _FRAME_FILES:
                    frames.append(
                        {
                            "file": relative,
                            "function": frame.f_code.co_name,
                            "line": frame.f_lineno,
                            "root": context[0],
                            "instance": context[1],
                        }
                    )
                frame = frame.f_back
            truncated = truncated or frame is not None
        return frames, truncated

    def before_cleanup(self, failure: BaseException) -> None:
        """Seal once before potentially blocking stop, including finally-unwind."""
        with self._lock:
            if self._sealed:
                return
            self._sealed = True
            events = list(self._events)
            workers = dict(self._workers)
            dropped = self._dropped
            observer_errors = self._observer_errors
        frames, frames_truncated = self._frames(workers)
        document = {
            "diagnostic": "mobility-failure",
            "events": events,
            "frames": frames,
            "frames_truncated": frames_truncated,
            "dropped_events": dropped,
            "truncated": bool(dropped) or frames_truncated,
            "observer_errors": observer_errors,
        }
        # Drop whole records, never slice JSON/UTF-8 into misleading partial data.
        while True:
            text = json.dumps(document, ensure_ascii=True, separators=(",", ":")) + "\n"
            # Reserve the fixed fallback marker too: a sink may write THEN fail.
            if len(text.encode("utf-8")) + len(_UNAVAILABLE.encode()) <= MAX_BYTES:
                self._sink(text)
                return
            document["truncated"] = True
            if document["frames"]:
                document["frames"].pop()
                document["frames_truncated"] = True
            elif events:
                events.pop()
                document["dropped_events"] += 1
            else:
                return
