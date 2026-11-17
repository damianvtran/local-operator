"""Detect a receiver that is running but can never persist what it pulls (a jam).

WHY THIS EXISTS (Team Bridge incident, 2026-11-17). A teammate's receiver
pulled the same feed batch for four days and failed to save it every time:
an unparseable threaded ref made the all-or-nothing batch save refuse with
``ref_migration_malformed``, the cursor never advanced, and 30 deliveries were
never screened. The process was alive, held its lock and polled on schedule,
so every health signal said "fine"; the only symptom an owner could see was a
peer's ``not_admitted``. The loop did log one ``stage=iteration`` line per
failure, but nothing summarised "this is the SAME local refusal, over and
over", which is what distinguishes a jam from a bad minute.

WHAT COUNTS. Only a LOCAL persistence refusal — a ``StoreRefused`` whose code
has the closed machine shape (``loop._STORE_CODE_PATTERN``) — repeated with the
SAME code on consecutive iterations. A hub outage, a DNS failure or an offline
laptop raise other errors, which reset nothing and count nothing: they are not
a jam, and a laptop on a train must not report one. Any successful iteration
clears the state, as does a different refusal code (a new code is a new
problem, counted from one).

THE THRESHOLD (:data:`JAM_THRESHOLD`) is a judgement, not a measured
distribution, and is stated as one: the loop backs off 1, 2, 4 ... 60 s
(``loop.BACKOFF_BASE_S``/``BACKOFF_MAX_S``), so ten identical refusals in a
row span roughly five minutes of continuous failure. A transient refusal (a
busy database past its bound) clears on the next pass; a deterministic one,
like the incident's, repeats forever. Five minutes separates the two by a wide
margin without making an owner wait an hour to see it.

SHAPE. A disposable side file beside the runtime record (the same pattern as
``runtime_record`` and ``wake_diagnostics.json``): closed code, count, first
and last wall-clock time, and the receiver's pid so ``status`` can ignore a
file left by a process that has since exited. NO feed ref, message id, body,
account or exception text — the status is meant to be pasted into a chat.
Never raises: a receiver must not fail to deliver because a diagnostic could
not be written.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Consecutive identical persistence refusals before the receiver reports a jam.
JAM_THRESHOLD = 10

#: The store-code shape that may be written down (mirrors
#: ``loop._STORE_CODE_PATTERN``; duplicated, not imported, so this module stays
#: import-light for ``status``).
_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def jam_path(config_dir: Path | str) -> Path:
    return Path(config_dir) / "team_bridge" / "connector-jam.json"


def _persistence_code(failure: BaseException) -> str | None:
    from local_operator.team_bridge.store import StoreRefused

    if not isinstance(failure, StoreRefused):
        return None
    code = getattr(failure, "code", None)
    return code if isinstance(code, str) and _CODE.fullmatch(code) else None


class JamTracker:
    """Per-loop counter of consecutive identical persistence refusals."""

    def __init__(self, config_dir: Path | str, *, threshold: int = JAM_THRESHOLD) -> None:
        self._path = jam_path(config_dir)
        self._threshold = threshold
        self._code: str | None = None
        self._count = 0
        self._since = 0.0
        self._reported = False
        # A file from an earlier process is stale by definition: this process
        # has not failed yet. Clear it so ``status`` never shows an old jam.
        self._clear()

    def note_success(self) -> None:
        if self._code is not None or self._reported:
            self._reset()

    def note_failure(self, failure: BaseException) -> None:
        try:
            code = _persistence_code(failure)
            if code is None:
                # Not a persistence refusal: neither evidence of a jam nor of
                # its end (the store was never reached). Leave the count alone.
                return
            now = time.time()
            if code != self._code:
                self._reset()
                self._code, self._since = code, now
            self._count += 1
            if self._count >= self._threshold:
                if not self._reported:
                    # ONE content-free line when the jam is first declared.
                    logger.warning(
                        "bridge connector jammed: stage=persist code=%s count=%d",
                        code,
                        self._count,
                    )
                self._reported = True
                self._write(
                    {
                        "pid": os.getpid(),
                        "code": code,
                        "count": self._count,
                        "since": self._since,
                        "last": now,
                    }
                )
        except Exception:  # noqa: BLE001 — a diagnostic never fails the receiver
            logger.debug("bridge connector jam tracker skipped", exc_info=True)

    def _reset(self) -> None:
        if self._reported:
            logger.info("bridge connector unjammed: stage=persist code=%s", self._code)
            self._clear()
        self._code, self._count, self._since, self._reported = None, 0, 0.0, False

    def _clear(self) -> None:
        try:
            self._path.unlink(missing_ok=True)
        except OSError:
            pass

    def _write(self, payload: dict[str, Any]) -> None:
        temporary = self._path.with_name(f"{self._path.name}.{os.getpid()}.tmp")
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            os.replace(temporary, self._path)
        except OSError:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def read_jam(config_dir: Path | str) -> dict[str, Any] | None:
    """The live receiver's jam, or ``None``. Read-only; never raises.

    A file whose pid is provably gone is ignored (the receiver exited mid-jam;
    the next one starts by clearing it).
    """

    try:
        payload = json.loads(jam_path(config_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    pid, code, count = payload.get("pid"), payload.get("code"), payload.get("count")
    if not (isinstance(pid, int) and isinstance(code, str) and _CODE.fullmatch(code)):
        return None
    if not isinstance(count, int):
        return None
    try:
        from local_operator.procstate import pid_liveness

        if pid_liveness(pid) is False:
            return None
    except Exception:  # noqa: BLE001
        pass
    since, last = payload.get("since"), payload.get("last")
    return {
        "code": code,
        "count": count,
        "since": since if isinstance(since, (int, float)) else None,
        "last": last if isinstance(last, (int, float)) else None,
    }
