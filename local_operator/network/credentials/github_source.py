"""Bounded owner-side gh source reads, isolated from provider/MCP workers.

A single daemon worker is retained until its whole read finishes: timing out an
await must not spawn replacement threads around a wedged keychain or filesystem.
Only the raw owner login is briefly cached; repository policy is resolved afresh
on this same isolated lane for every completed flight. No bearer is persisted.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import functools
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable

from local_operator.network.credentials import github

# Successful owner grant latency observed in the incident: n=98, p90=.7s,
# max=3.2s. Keep the whole-read deadline ABOVE gh's existing 10s subprocess
# bound, so a real subprocess failure retains github_gh_unusable. The outer
# bound additionally covers hosts-file/program lookup and the isolated lane.
SOURCE_TIMEOUT_S = 12.0
# Shorter than the existing 900s permission lease; logout/account-file edits
# invalidate immediately, and keychain-only changes are observed within 30s.
SOURCE_CACHE_S = 30.0


class SourceBusy(TimeoutError):
    """The whole owner source read missed its bound, not an invalid gh login."""


class NamedExecutor(concurrent.futures.ThreadPoolExecutor):
    """The existing default pool, with bounded names-only timeout diagnostics."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._names_lock = threading.Lock()
        self._names: dict[object, tuple[str, float, bool]] = {}

    def submit(
        self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any
    ) -> concurrent.futures.Future[Any]:
        target = fn.func if isinstance(fn, functools.partial) else fn
        # asyncio.to_thread wraps the real callable in Context.run. Unwrap only
        # that known callable slot, never format its (possibly secret) args.
        if (
            isinstance(fn, functools.partial)
            and isinstance(getattr(target, "__self__", None), contextvars.Context)
            and fn.args
            and callable(fn.args[0])
        ):
            target = fn.args[0]
        name = re.sub(
            r"[^a-zA-Z0-9_.]", "_", getattr(target, "__qualname__", type(target).__name__)
        )[:96]
        slot = object()
        with self._names_lock:
            self._names[slot] = (name, time.monotonic(), False)

        def run() -> Any:
            with self._names_lock:
                self._names[slot] = (name, time.monotonic(), True)
            try:
                return fn(*args, **kwargs)
            finally:
                with self._names_lock:
                    self._names.pop(slot, None)

        try:
            future = super().submit(run)
        except BaseException:
            with self._names_lock:
                self._names.pop(slot, None)
            raise

        def done(_future: Any) -> None:
            if _future.cancelled():
                with self._names_lock:
                    self._names.pop(slot, None)

        future.add_done_callback(done)
        return future

    def diagnostics(self) -> dict[str, Any]:
        with self._names_lock:
            rows = list(self._names.values())
        return {
            "running": sorted({name for name, _at, running in rows if running})[:8],
            "queued": sorted({name for name, _at, running in rows if not running})[:8],
            "queue_wait_ms": int(
                1000
                * max(
                    (time.monotonic() - at for _name, at, running in rows if not running), default=0
                )
            ),
        }


class GithubSource:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._future: concurrent.futures.Future[Any] | None = None
        self._thread: threading.Thread | None = None
        self._epoch = 0
        self._deadline = 0.0
        self._wall_deadline = 0.0
        self._closed = False
        self._token = ""
        self._until = 0.0
        self._stamp: tuple[int, int, int] | None = None
        self._clock = time.monotonic
        self._wall_clock = time.time
        self._cached_wall = 0.0

    def invalidate(self) -> None:
        with self._lock:
            self._epoch += 1
            self._token = ""
            self._until = 0.0
            self._stamp = None

    async def resolve(self, root: Path | None, minter: Any) -> tuple[str, str, Any]:
        with self._lock:
            if self._closed:
                raise SourceBusy("the GitHub source reader is closed")
            future = self._future
            if future is None or future.done():
                future = concurrent.futures.Future()
                self._future = future
                self._deadline = time.monotonic() + SOURCE_TIMEOUT_S
                self._wall_deadline = self._wall_clock() + SOURCE_TIMEOUT_S
                epoch = self._epoch
                self._thread = threading.Thread(
                    # Preserve the owner's existing scope/audit context while
                    # moving its probe off-loop; do not infer it from child env.
                    target=contextvars.copy_context().run,
                    args=(self._run, future, epoch, root, minter),
                    name="mesh-github-source",
                    daemon=True,
                )
                self._thread.start()
            wall_deadline = self._wall_deadline
            remaining = min(self._deadline - time.monotonic(), wall_deadline - self._wall_clock())
        if remaining <= 0:
            raise SourceBusy("the in-flight GitHub login read already timed out")
        try:
            # Shield the flight: cancelling one caller must not cancel a source
            # another caller joined, nor free the slot while its worker runs.
            result = await asyncio.wait_for(asyncio.shield(asyncio.wrap_future(future)), remaining)
        except asyncio.TimeoutError as exc:
            raise SourceBusy("the whole GitHub login read timed out") from exc
        with self._lock:
            if self._closed or result[0] != self._epoch:
                raise SourceBusy("the GitHub source changed during its read")
            # Darwin's monotonic clock can exclude sleep. A resumed owner must
            # not deliver a flight or cache whose wall bound already ran out
            # while its lid was closed. A backwards step fails closed too.
            if self._wall_clock() >= wall_deadline:
                raise SourceBusy("the GitHub source read exceeded its wall deadline")
            if (
                result[1] == github.SOURCE_GH
                and not 0 <= self._wall_clock() - self._cached_wall < SOURCE_CACHE_S
            ):
                raise SourceBusy("the GitHub source cache expired during its read")
        return result[1], result[2], result[3]

    def _run(
        self, future: concurrent.futures.Future[Any], epoch: int, root: Path | None, minter: Any
    ) -> None:
        try:
            # The ladder's gh-presence probe also reads hosts/program metadata;
            # doing it on the loop before this flight would escape the bound.
            source = github.resolve_source(root)
            if source != github.SOURCE_GH:
                with self._lock:
                    self._token, self._until, self._stamp = "", 0.0, None
                future.set_result((epoch, source, "", None))
                return
            path = github.gh_hosts_path()
            try:
                stat = path.stat()
                stamp = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
            except FileNotFoundError:
                stamp = None
            with self._lock:
                token = (
                    self._token
                    if self._stamp == stamp
                    and self._clock() < self._until
                    and 0 <= self._wall_clock() - self._cached_wall < SOURCE_CACHE_S
                    else ""
                )
            if not token:
                token = github.read_gh_token()
                # Do not warm the cache after logout/revoke/panic. A future
                # result from the previous epoch is likewise rejected by resolve.
                with self._lock:
                    if self._epoch == epoch and not self._closed:
                        self._token, self._stamp = token, stamp
                        self._until = self._clock() + SOURCE_CACHE_S
                        self._cached_wall = self._wall_clock()
            designated = github.designation(
                root, source=github.SOURCE_GH, token=token, minter=minter
            )
            future.set_result((epoch, source, token, designated))
        except BaseException as exc:
            if isinstance(exc, github.GithubGhError):
                self.invalidate()
            future.set_exception(exc)

    def close(self, timeout: float = 5.0) -> None:
        self.invalidate()
        with self._lock:
            self._closed = True
            thread = self._thread
        if thread is not None:
            thread.join(timeout)
