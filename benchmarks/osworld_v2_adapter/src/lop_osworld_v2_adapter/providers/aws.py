"""AwsProvider: the one module in this adapter that spends money.

It launches the OSWorld guest on EC2 WITH boto3 DIRECTLY, then hands the
instance id to upstream's ``DesktopEnv(path_to_vm=<id>)`` so every guest
interaction (setup, screenshots, actions, evaluation) is upstream code called
unmodified. It does NOT use OSWorld's own ``AWSVMManager._allocate_vm``, and
the reasons are each a released-contract requirement, not a preference:

- OSWorld's ``run_instances`` carries no ``ClientToken`` (manager.py:222-266),
  so a retried ``reset_start`` double-allocates. Ours is minted from the
  episode id (``ProvisioningPlan.client_token``).
- OSWorld tags only ``Name`` and only when an env var is set (manager.py:254).
  The cleanup contract (``cleanup.py``) resolves the instance BY TAG from a
  persisted descriptor, so ``lop:episode`` and ``lop:adapter`` must be in
  ``TagSpecifications`` at creation — and the operator's TTL role is scoped by
  ``aws:ResourceTag/lop:adapter``, so without the tag the lease cannot fire.
- OSWorld treats EventBridge TTL failure as a logged WARNING
  (manager.py:274-276). The previous pilot lost paid instances to SIGKILL, a
  sleeping laptop and a ``/tmp`` purge because teardown was foreground-only.
  Here the TTL lease is created immediately after ``run_instances`` and its
  failure is FATAL to ``allocate`` (terminate what was launched, then raise).
- OSWorld installs process-wide SIGINT/SIGTERM handlers inside ``_allocate_vm``
  (manager.py:181-206), which is unacceptable inside an asyncio worker whose
  parent owns signal delivery.
- ``DesktopEnv.close()`` calls ``terminate_instances`` and returns without
  confirming (provider.py:279-285). It is NEVER called; ``terminate`` here
  polls ``describe_instances`` until the state is ``terminated`` and reports
  ``terminate-unconfirmed`` (rescue-required) if it is not reached in time.

Credentials arrive as ``AwsCredentials`` (from ``ResetStartParams.secrets`` /
``BeginRescueParams.secrets``), never from ``os.environ``: the worker's
environment is stripped and MUST stay that way. Every client is built with an
explicit ``region_name`` — the operator's default profile region is not
us-east-1, and a client that fell back to the profile would launch the
release-pinned AMI in a region where it does not exist.

Every boto3 / HTTP call is blocking and runs under ``asyncio.to_thread`` so the
worker's event loop keeps answering the parent's RPC while a waiter sleeps.

``allocate`` also runs one GUEST PREPARATION step between readiness and
upstream's reset: the released AMI ships ~93% full and its own snapd fills the
remainder on a clock, which destroyed 7 of 8 episodes in a 424-466s window. See
``guest_disk`` for the measurements, for why ffmpeg was NOT the cause, and for
why that step can never fail an allocation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from lop_osworld_v2_adapter import guest_disk, guest_loss, guest_supervisor, scoring
from lop_osworld_v2_adapter.cleanup import (
    EVIDENCE_INSTANCE_ABSENT,
    EVIDENCE_INSTANCE_TERMINATED,
    EVIDENCE_SCHEDULE_ABSENT,
    EVIDENCE_SCHEDULE_DELETE_FAILED,
    EVIDENCE_SCHEDULE_DELETED,
    EVIDENCE_TERMINATE_DENIED,
    EVIDENCE_TERMINATE_UNCONFIRMED,
)
from lop_osworld_v2_adapter.observation import SCREENSHOT_CAUSE_KEY
from lop_osworld_v2_adapter.providers.base import (
    DEFAULT_ACTION_DELAY_S,
    GUEST_COMMAND_TIMEOUT_S,
    GUEST_TRANSPORT_MARGIN_S,
    TRANSPORT_RETRY_ATTEMPTS,
    GuestServerRecovered,
    InstructionUnavailable,
    bounded_observation_cause,
    guest_deadline_for,
    transport_retry_delay,
)
from lop_osworld_v2_adapter.provisioning import ProvisioningPlan
from lop_osworld_v2_adapter.taskfile import TaskDescriptor

logger = logging.getLogger(__name__)

# The tag every resource this adapter creates carries. It is both the leak
# detector's filter (``audit``) and the condition key on the operator's TTL
# role, so it must match ``provisioning.resolve`` exactly.
ADAPTER_TAG_KEY = "lop:adapter"
ADAPTER_TAG_VALUE = "osworld-v2"
EPISODE_TAG_KEY = "lop:episode"

# States in which an instance still exists and can be terminated. ``terminated``
# is deliberately absent: an instance already there is ``instance-absent`` for
# cleanup purposes, and AWS keeps terminated instances visible for an hour.
LIVE_STATES = ("pending", "running", "stopping", "stopped", "shutting-down")

# Default lease when the budget carries no wall cap (DESIGN §B.5): long enough
# for a 500-step episode at ~3 s/action plus setup, short enough that a dead
# controller does not bill for OSWorld's own 5-hour default.
DEFAULT_TTL_SECONDS = 7200
# Slack added to the wall budget so the lease never fires on a legitimately
# slow but in-budget episode.
TTL_SLACK_SECONDS = 15 * 60

# The EventBridge Scheduler universal target for terminating instances.
_TERMINATE_TARGET_ARN = "arn:aws:scheduler:::aws-sdk:ec2:terminateInstances"

# The OSWorld guest control port and the readiness endpoint upstream's own
# ``SetupController.ensure_ready`` probes (setup.py:58-70).
GUEST_PORT = 5000
_READY_PATH = "/terminal"
# The guest's command endpoint. ``shell: false`` means the server execs the argv
# directly, so a pipeline arrives as an explicit ``bash -c`` -- the same shape
# upstream's own SetupController posts (setup.py:418).
_EXECUTE_PATH = "/execute"
# The readiness probe taken before a retry is scheduled. It is EVIDENCE (see
# ``execute``), not a veto: it rides the same host->guest path as the request
# that just failed, so when that PATH is down the probe fails for the same
# reason and says nothing about whether the server is listening. The retry
# bound lives in ``providers.base`` (``TRANSPORT_RETRY_ATTEMPTS``), where every
# remote provider can share it.
_EXECUTE_READINESS_TIMEOUT_S = 2.0

# Where the episode's guest-preparation evidence is written, inside the
# episode-owned cache root the adapter mints in ``reset_start``. It sits there
# rather than in the artifact root because the bundle verifier refuses any entry
# in the artifact root that is not a digest-named artifact.
GUEST_PREPARATION_FILENAME = "guest-preparation.json"


class AllocationError(RuntimeError):
    """``allocate`` could not reach a fully-leased, ready guest."""


class GuestExecutionError(RuntimeError):
    """Input may have partially committed; never replay the failed batch.

    ``delivery`` is set only for a TRANSPORT failure and is one of
    ``guest_loss.DELIVERY_NOT_SENT`` / ``DELIVERY_AMBIGUOUS``: whether the
    failing request provably never left this host. It is for in-process readers
    and tests; across the worker boundary only the message text travels (the
    ``delivery=`` token in it), because the RPC error envelope has no adapter
    field for it and the harness must stay benchmark-neutral.
    """

    def __init__(self, message: str, *, delivery: str | None = None) -> None:
        super().__init__(message)
        self.delivery = delivery


def _exit_description(returncode: int) -> str:
    """Describe a guest exit status: an ordinary code, or death by signal.

    ``subprocess`` reports a signal-killed child as a NEGATIVE returncode, so
    ``exit -15`` in a diagnostic is really "SIGTERM killed it" wearing a
    misleading costume. A reader who does not know that convention reads -15 as
    an exit code and looks for a program that returned one; naming the signal
    removes a research step that has already cost a full investigation.

    Deliberately says WHAT happened and not WHY. A signal death has many
    possible originators — the OS, a supervisor, another process, or the
    command's own actions — and this layer cannot distinguish them. Naming a
    specific cause here would encode a guess into every future incident report,
    which is the failure mode that made the last one expensive.

    Leaks nothing: the output is a signal NAME from the stdlib's fixed
    enumeration or a bare integer, never command text, guest stdout, or stderr.
    That is the same discipline ``execute`` applies to transport errors.
    """

    if returncode >= 0:
        return f"exit {returncode}"
    try:
        name = signal.Signals(-returncode).name
    except ValueError:
        # An unrecognised signal number is still worth reporting as a signal
        # rather than as a negative exit code.
        return f"terminated by signal {-returncode}"
    return f"terminated by signal {name}"


class ReadinessTimeout(AllocationError):
    """The instance ran but the guest service never answered in time."""


@dataclass(frozen=True)
class AwsCredentials:
    """The resolved AWS secrets. Never logged, never placed in the environment."""

    access_key_id: str
    secret_access_key: str
    session_token: str | None = None

    def __repr__(self) -> str:  # pragma: no cover - defensive against accidental echo
        return "AwsCredentials(<redacted>)"


@dataclass
class _Clients:
    """The injectable I/O surface, so tests use botocore's Stubber.

    ``ec2`` and ``scheduler`` are boto3 clients; ``http_get`` takes a URL and
    a timeout and returns the HTTP status code (or raises); ``http_post_json``
    posts a JSON body and returns the decoded response (or raises). Production
    builds all four from the credentials; tests hand in stubbed clients and
    fake probers.

    ``http_post_json`` is REQUIRED rather than optional-with-a-default: it is
    what drives the guest's control server for pre-episode disk reclamation
    (``guest_disk``), and a default that quietly did nothing would make that
    preparation silently absent in exactly the builds nobody checked.
    """

    ec2: Any
    scheduler: Any
    http_get: Callable[[str, float], int]
    http_post_json: Callable[[str, dict[str, Any], float], dict[str, Any]]


def build_clients(credentials: AwsCredentials, region: str) -> _Clients:
    """Construct real clients from the resolved secrets.

    ``region_name`` is passed explicitly on every client. The session is
    built from the credential VALUES, so boto3's own provider chain (env vars,
    ``~/.aws``, instance metadata) is never consulted — which is what keeps
    the worker's stripped environment authoritative.
    """

    import requests  # type: ignore[import-not-found]
    from boto3.session import Session  # type: ignore[import-not-found]

    session = Session(
        aws_access_key_id=credentials.access_key_id,
        aws_secret_access_key=credentials.secret_access_key,
        aws_session_token=credentials.session_token,
        region_name=region,
    )

    def http_get(url: str, timeout: float) -> int:
        return int(requests.get(url, timeout=timeout).status_code)

    def http_post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        # ``allow_redirects=False``: a guest command is a non-idempotent POST.
        # Following a 3xx would re-send it to a host the guest chose, and when
        # THAT hop refused the connection ``requests`` raised a bare connect
        # error that hid the first, delivered request (``guest_loss``, review
        # round 1 C1). The control server never redirects, so a 3xx is already
        # an anomaly; it is surfaced as an HTTP error, which classifies as
        # ``ambiguous`` because the server answered.
        response = requests.post(
            url,
            json=payload,
            timeout=timeout,
            headers={"Content-Type": "application/json"},
            allow_redirects=False,
        )
        if 300 <= response.status_code < 400:
            # ``raise_for_status`` passes 3xx; without this the body would be
            # parsed as if it were an answer.
            raise requests.HTTPError("guest control server answered a redirect", response=response)
        response.raise_for_status()
        decoded = response.json()
        if not isinstance(decoded, dict):
            raise ValueError("guest control server returned a non-object body")
        return decoded

    return _Clients(
        ec2=session.client("ec2", region_name=region),
        scheduler=session.client("scheduler", region_name=region),
        http_get=http_get,
        http_post_json=http_post_json,
    )


def ttl_seconds_for(wall_budget_ms: int | None, *, override: int | None = None) -> int:
    """Derive the lease length from the episode's wall budget.

    An explicit operator override (``OSWORLD_TTL_SECONDS`` infra) wins; a
    capped wall budget produces ``budget + slack``; an uncapped one falls back
    to ``DEFAULT_TTL_SECONDS``. The result is never below the slack itself, so
    a tiny budget cannot produce a lease that fires during boot.
    """

    if override is not None:
        return max(int(override), TTL_SLACK_SECONDS)
    if wall_budget_ms is None:
        return DEFAULT_TTL_SECONDS
    return max(int(wall_budget_ms // 1000) + TTL_SLACK_SECONDS, TTL_SLACK_SECONDS)


def _episode_id_from_ref(instance_ref: str) -> str:
    """``lop-ep-<episode_id>`` -> ``<episode_id>``; the ref shape from cleanup.py."""

    prefix = "lop-ep-"
    if not instance_ref.startswith(prefix):
        raise ValueError(f"instance ref {instance_ref!r} is not a lop-ep- tag ref")
    return instance_ref[len(prefix) :]


def _tag_filters(episode_id: str | None) -> list[dict[str, Any]]:
    filters: list[dict[str, Any]] = [
        {"Name": f"tag:{ADAPTER_TAG_KEY}", "Values": [ADAPTER_TAG_VALUE]},
        {"Name": "instance-state-name", "Values": list(LIVE_STATES)},
    ]
    if episode_id is not None:
        filters.insert(0, {"Name": f"tag:{EPISODE_TAG_KEY}", "Values": [episode_id]})
    return filters


def _instances(response: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        instance
        for reservation in response.get("Reservations", [])
        for instance in reservation.get("Instances", [])
    ]


def _error_code(error: Exception) -> str:
    response = getattr(error, "response", None)
    if isinstance(response, dict):
        return str(response.get("Error", {}).get("Code", ""))
    return ""


#: The first-failure segment: how an annotated diagnostic names the failure
#: upstream's retry loop swallowed. The harness keeps only the HEAD of a wire
#: message (``rpc.MAX_DETAIL_MESSAGE`` = 512, ``worker._control_safe``'s
#: ``[:limit]``), so the segment is bounded AND placed first -- appended to a
#: variable-length tail it can be cut off silently, which is the exact
#: swallow this attachment exists to undo (review round 1, R1-F1 measured it:
#: a 500-char last-failure reason pushed a tail-appended note past the cut
#: entirely).
_FIRST_FAILURE_MARKER = "[first reset setup attempt failed with: "
_FIRST_FAILURE_NOTE_LIMIT = 180


def _first_failure_segment(note: str) -> str:
    """``note`` as the wire-bounded head segment of an annotated message."""

    if len(note) > _FIRST_FAILURE_NOTE_LIMIT:
        note = note[: _FIRST_FAILURE_NOTE_LIMIT - 3] + "..."
    return f"{_FIRST_FAILURE_MARKER}{note}]"


class UpstreamAllocationRefused(RuntimeError):
    """Upstream OSWorld tried to allocate, revert, or release an instance.

    Every upstream launch path (``AWSVMManager._allocate_vm``,
    ``AWSProvider.revert_to_snapshot``) issues ``run_instances`` with NO
    ``ClientToken``, NO ``lop:adapter``/``lop:episode`` tag (``Name`` only,
    and only when ``AWS_INSTANCE_NAME`` is set) and NO TTL schedule (our
    ``ENABLE_TTL=false``). An instance it created would be invisible to the
    tag audit, unreachable by descriptor-driven rescue, and unleased -- the
    exact leak this provider exists to prevent. ``stop_emulator`` is the
    unconfirmed terminate we also never want. So ``_seal_upstream`` replaces
    every one of those methods on the live env with a raiser -- except
    inside the first reset, where upstream's own retry loop asks to revert a
    setup attempt that failed before anything ran (see ``_seal_upstream``)
    -- and this is what it raises whenever it refuses: BEFORE any boto3
    call, from inside the worker, surfacing to the runner as an adapter
    error whose cleanup path terminates OUR tagged instance by tag and
    confirms it.
    """


# Files, not logger names, identify a judge failure. ``llm_metrics.py`` logs
# its swallowed exceptions on the SHARED ``desktopenv.env`` logger -- the
# same one ``DesktopEnv.reset``/``evaluate`` use for benign setup-retry
# ERRORs ("Environment setup failed, retrying (1/5)...") -- so filtering by
# logger name either misses the judge (drop ``desktopenv.env``) or trips on a
# good score after a recoverable retry (keep it). ``LogRecord.pathname`` is
# set by the caller's frame and is what actually distinguishes them. The
# pinned checkout is what makes matching a basename safe.
_JUDGE_SOURCE_FILES = frozenset({"llm_metrics.py", "model_client.py"})
# The loggers those files write to; the handler is attached here only.
_JUDGE_LOGGERS = ("desktopenv.eval_model", "llm_metrics", "desktopenv.env")


class _JudgeErrorCapture(logging.Handler):
    """Records ERROR records emitted FROM OSWorld's judge modules.

    ``llm_metrics`` swallows every judge exception into ``0.0`` after a
    ``logger.error("Error in compare_...")`` (llm_metrics.py:176-178). The
    only observable trace of a failed judge call is that log record, so
    capturing it is what lets ``evaluate`` raise ``ScoringUnavailable``
    instead of returning a silent zero. Records from any other file --
    setup retries, "No evaluator configured", metric file-not-found -- are
    deliberately NOT captured: those are either benign or already the
    score's own honest content.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if os.path.basename(record.pathname) not in _JUDGE_SOURCE_FILES:
            return
        try:
            self.messages.append(record.getMessage())
        except Exception:  # pragma: no cover - a broken format must not mask the error
            self.messages.append(record.msg if isinstance(record.msg, str) else "judge error")


# The tab getter's own namespace and module, both pinned (OSWorld-V2 @ d578d2d,
# ``desktop_env/evaluators/getters/chrome.py`` -- logger at line 32,
# ``get_activate_tab_json`` at 2054-2148). Both halves are checked, for the
# reason recorded at ``_SCREENSHOT_LOGGER``: attachment alone would accept any
# record routed to that logger, and a bare ``chrome.py`` basename is not unique
# in a package this size. An upstream rename disarms the capture, and it then
# degrades to no catch rather than to a wrong one.
_GETTER_LOGGER = "desktopenv.getters.chrome"
_GETTER_SOURCE_FILE = "chrome.py"

#: The tagged getter this guard is about. ``chrome.py`` holds ~a dozen page
#: getters emitting ERRORs from the same file and logger, most of them
#: legitimate readings an evaluator may score (an empty tab list, a page that
#: would not load). The tag names the ONE getter whose matched-but-unreadable
#: outcome is an apparatus failure rather than a reading.
_GETTER_TAG = "[ACTIVATE_TAB_JSON]"

#: The tag's one terminal outcome that is the AGENT's to own: no page matched
#: at all, so the artifact the evaluator asked for is absent from the guest and
#: a 0.0 is the honest reading of an absent artifact.
#:
#: Every OTHER terminal failure of this getter is the EVALUATOR's, in one of two
#: shapes, and both must not be sealed as a miss: it selected a target it could
#: not interrogate (the ``getJSON`` helper is missing, or invoking it failed), so
#: its ``None`` says nothing about the agent's work; or it refused to run at all
#: because its own configuration is missing (``Missing required config
#: 'tab_prefix'``), which is evaluator misconfiguration and equally not the
#: agent's. The captured line is upstream's own text, so it names which shape it
#: was rather than leaving a reader to infer it.
_GETTER_NO_TARGET = "No tab url starts with"


class _UnreadGetterTargetCapture(logging.Handler):
    """Records the tab getter's "matched a target, could not read it" errors.

    THE DEFECT THIS EXISTS FOR (arm 1830 r1, task_009; record:
    ``scripts/logs/ZERO-CLASSIFICATION-1830-009-013.md``). ``tab_prefix`` for
    that task is the bare directory ``file:///home/user/Desktop/HKU-RIMS-System/``
    and the getter takes the FIRST page whose URL starts with it -- which can be
    Chrome's own directory listing for that folder. A listing runs no scripts,
    so the page helper the getter needs (``common.js:87``, loaded by every app
    page) is absent there. The getter logs an ERROR and returns ``None``, the
    task reads that ``None`` as "no result", and the episode is archived as a
    SCORED 0.0 -- a failure the agent did not commit, since the evaluator never
    read the page holding its work.

    WHAT THIS DOES AND DOES NOT CLAIM. It does not claim the agent's answers
    were right; nothing here reads them. It claims only that the evaluation
    could not be performed as intended, which is exactly what must not be
    reported as a miss.

    WHY THE SELECTION ITSELF IS NOT FIXED HERE. The first-match loop, the
    prefix, and the getter are all upstream OSWorld source installed from the
    pinned git dependency, and this tree does not rewrite upstream task,
    evaluator or framework source (docs/benchmarks/osworld_2/dependency-policy.md).
    The durable fix -- prefer a prefix-matching tab whose page defines the
    helper, falling back to the current rule -- belongs upstream; this guard is
    the legibility boundary that holds on our side until it lands, and until
    then it covers EVERY task using this getter and every prefix, not one task.

    A SECOND SHAPE REACHES THE SAME GUARD, and it is not a matcher failure: the
    getter returns ``None`` without looking at any page when its own config
    carries no ``tab_prefix``. That is an evaluator-authoring defect, not a
    reading of the agent's work, so it is unscoreable for the same reason and is
    captured by the same rule (see ``_GETTER_NO_TARGET`` for the rule and its
    one exemption).
    """

    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if os.path.basename(record.pathname) != _GETTER_SOURCE_FILE:
            return
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - a broken format must not mask the error
            # A record whose message will not format cannot be checked against
            # the tag, and ``chrome.py`` holds ~a dozen page getters logging
            # ERRORs from this same file and logger. Capturing it would raise on
            # a record this guard cannot attribute to the tagged getter at all,
            # so it is dropped -- the same failure direction as an upstream
            # rename: no catch rather than a wrong one.
            return
        if _GETTER_TAG not in message or _GETTER_NO_TARGET in message:
            return
        self.messages.append(message)


# The upstream frame path, pinned: ``desktop_env/controllers/python.py``
# (OSWorld-V2 @ d578d2d, ``get_screenshot`` at 455-481). The handler is attached
# to the one logger those records are written on, and each record is
# additionally checked for the MODULE it came from. Neither half is enough on
# its own: attachment alone would accept any record routed to that logger from
# elsewhere, and a bare ``python.py`` basename is not unique in a package this
# size. Together they name one function, which is what makes matching safe here
# (an upstream rename disarms the capture, and it then degrades to no cause
# rather than to a wrong one).
_SCREENSHOT_LOGGER = "desktopenv.pycontroller"
_SCREENSHOT_SOURCE_FILE = "python.py"

# Upstream's per-attempt failure lines, matched as PREFIXES. A record is
# reduced to a closed KIND plus, when the pinned format carries one, an integer
# status code -- nothing else in the record is read, and upstream's own words
# are never copied (see ``_ScreenshotFailureCapture``).
_SCREENSHOT_RECORD_KINDS = (
    ("Failed to get screenshot. Status code: ", "status"),
    ("Invalid screenshot payload", "payload"),
    ("An error occurred while trying to get the screenshot", "transport"),
    ("Failed to get screenshot.", "exhausted"),
)

#: How many upstream failure records a bounded cause may name. The COUNT is
#: always the true total; only the per-record detail is capped, so the cap can
#: never understate how long the guest was failing.
MAX_CAUSE_RECORDS = 8


class _ScreenshotFailureCapture(logging.Handler):
    """Records upstream's FAILED screenshot attempts during one ``observe``.

    ``get_screenshot`` swallows its own failure: it retries ``retry_times``
    times with ``retry_interval`` sleeps, logs each attempt at ERROR and
    returns ``None``. By the time the adapter sees that ``None`` the reason is
    gone, and "environment returned no screenshot frame" is all the bundle can
    say -- which is how two canary episodes (285 and 248 model cycles, $1.08
    and $2.24) died with nothing left to read.

    WHAT IS RECORDED, AND WHAT IS DELIBERATELY NOT. Each record contributes a
    closed KIND and, when the pinned format has one, an integer status code;
    its message and ``args`` are never copied. That is the point: upstream's
    text can carry a URL, a credential or a guest path, and a provider that
    clipped it to a wire bound would sever any secret straddling the cut -- the
    leak the harness closes by scanning BEFORE bounding (``worker._redacted``),
    an ordering this side cannot reproduce. Structured facts contain no
    third-party bytes, so bounding them is safe by construction. Upstream's own
    words are surfaced by the HARNESS instead, as the worker's stderr tail.
    """

    def __init__(self, origin: str | None = None) -> None:
        super().__init__(level=logging.ERROR)
        #: ``scheme://host:port`` of the guest, handed to the classifier so a
        #: failure on a REDIRECT target is not read as "never connected". The
        #: screenshot GET is upstream's own call (it follows redirects and is
        #: idempotent), so its semantics are deliberately left alone; only the
        #: diagnostic claim is kept honest.
        self.origin = origin
        #: (kind, status_or_None, monotonic_seconds) per failed attempt.
        self.records: list[tuple[str, int | None, float]] = []
        #: For each ``transport`` record whose exception PROVES the connection was
        #: never established, the closed sub-kind (``refused`` / ``timeout`` /
        #: ``dns`` / ``connect_failed``, from ``guest_loss``). Read from the
        #: exception OBJECT upstream passed as the log argument, never from its
        #: text, so the value-free guarantee above is unchanged. It is what
        #: separates "nothing is listening" (refused: the control server is
        #: GONE) from "listening but too slow" (a read timeout) -- the 043 loss
        #: read as plain ``transport`` for nine attempts.
        self.not_sent_kinds: list[str] = []
        self.started = time.monotonic()

    def emit(self, record: logging.LogRecord) -> None:
        if os.path.basename(record.pathname) != _SCREENSHOT_SOURCE_FILE:
            return
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - a broken format must not mask the error
            return
        for prefix, kind in _SCREENSHOT_RECORD_KINDS:
            if not message.startswith(prefix):
                continue
            status: int | None = None
            if kind == "status":
                tail = message[len(prefix) :].strip()
                # A malformed code is not a reason to drop the record: the KIND
                # is still known, only the OPTIONAL number is lost.
                status = int(tail) if tail.isdigit() else None
            arguments = record.args
            if kind == "transport" and isinstance(arguments, tuple) and arguments:
                cause = arguments[0]
                if isinstance(cause, BaseException):
                    verdict = guest_loss.classify_transport_failure(cause, origin=self.origin)
                    if verdict.delivery == guest_loss.DELIVERY_NOT_SENT:
                        self.not_sent_kinds.append(verdict.kind)
            self.records.append((kind, status, time.monotonic()))
            return

    def cause(self, *, elapsed_ms: int) -> str | None:
        """One bounded, value-free line, or None when upstream said nothing.

        ``elapsed_ms`` is the whole ``_get_obs`` call and ``first_ms`` the time
        from its start to upstream's first failure record. Those readings are
        the fact the harness could not otherwise obtain, and they separate the
        two mechanisms it had to guess between: a refused or instantly-erroring
        request leaves upstream's own 5 s retry sleep standing (a gap near
        5000 ms), while a request that burned its 10 s client timeout adds that
        much again (a gap near 15000 ms).
        """

        if not self.records:
            return None
        records = self.records[:MAX_CAUSE_RECORDS]
        parts = [
            f"upstream_failures={len(self.records)}",
            "kinds=" + ",".join(kind for kind, _, _ in records),
        ]
        codes = [str(status) for _, status, _ in records if status is not None]
        if codes:
            parts.append("codes=" + ",".join(codes))
        if self.not_sent_kinds:
            # Placed beside ``kinds`` (the head of the line) because the harness
            # keeps only the head of a wire message. ``n:kind[+kind]`` -- how
            # many failed reads never connected, and why -- so a refused guest
            # is recognisable on the FIRST retry record instead of after nine.
            parts.append(
                f"not_sent={len(self.not_sent_kinds)}:" + "+".join(sorted(set(self.not_sent_kinds)))
            )
        parts.append(f"first_ms={_since_ms(self.started, records[0][2])}")
        gaps = [
            _since_ms(records[index - 1][2], records[index][2]) for index in range(1, len(records))
        ]
        if gaps:
            parts.append("gaps_ms=" + ",".join(str(gap) for gap in gaps))
        parts.append(f"elapsed_ms={max(0, elapsed_ms)}")
        return bounded_observation_cause("screenshot unavailable: " + " ".join(parts))


def _since_ms(earlier: float, later: float) -> int:
    """A non-negative millisecond delta between two monotonic readings."""

    return max(0, int((later - earlier) * 1000))


def _read_reset_instruction(env: Any) -> str:
    """Read the task instruction ONCE, post-reset, via the official accessor.

    ``DesktopEnv._task_get(task_config, key)`` (desktop_env.py:222-231 on the
    pinned v2.1 source; the same shape on the 2.0 checkout) is the accessor
    upstream's own ``_set_task_info`` reads the instruction through, for both
    mapping and attribute ``task_config`` shapes -- so this calls the SAME
    function the environment itself trusts, never a re-implementation and
    never a second parser. It is deliberately NOT a mirror of the caching
    code: the point is to ask the live object again AFTER setup, because the
    cached value is the pre-setup one.

    Every failure is a refusal (``InstructionUnavailable``), never a value:
    an absent accessor, a getter exception, a non-string value, or text that
    is empty or whitespace-only. Carrying on would either run the model
    against a stale or fabricated instruction, or let the episode die later
    somewhere less attributable -- both make the episode's record dishonest
    about which task was actually run. The original exception is chained,
    never swallowed, and the VALUE is never echoed (only its type): the
    instruction is task content and this message may travel further than the
    observation does.
    """

    getter = getattr(env, "_task_get", None)
    if not callable(getter):
        raise InstructionUnavailable(
            "the pinned DesktopEnv exposes no _task_get accessor, so the task "
            "instruction cannot be read from the official runtime; refusing "
            "before any model request"
        )
    try:
        value = getter(getattr(env, "task_config", None), "instruction")
    except Exception as error:
        raise InstructionUnavailable(
            "reading the task instruction from the environment failed: "
            f"{type(error).__name__}: {error}"
        ) from error
    if not isinstance(value, str) or not value.strip():
        raise InstructionUnavailable(
            "the environment served no usable task instruction "
            f"(got {type(value).__name__}): the reset observation would carry "
            "no task text; refusing before any model request"
        )
    return value


class AwsProvider:
    """Satisfies ``EnvironmentProvider`` against a real EC2 guest."""

    def __init__(
        self,
        credentials: AwsCredentials,
        *,
        region: str,
        lease_ref: str,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        readiness_timeout_s: float = 600.0,
        action_delay_s: float = DEFAULT_ACTION_DELAY_S,
        terminate_timeout_s: float = 55.0,
        clients: _Clients | None = None,
        desktop_env_factory: Callable[..., Any] | None = None,
        task_factory: Callable[[TaskDescriptor], Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._region = region
        self._lease_ref = lease_ref
        self._ttl_seconds = ttl_seconds
        self._readiness_timeout_s = readiness_timeout_s
        self._action_delay_s = action_delay_s
        # Below the release-instance action's 60 s ``timeout_ms`` so the
        # provider reports ``terminate-unconfirmed`` itself rather than having
        # the parent time the whole call out and record nothing.
        self._terminate_timeout_s = terminate_timeout_s
        self._clients = clients if clients is not None else build_clients(credentials, region)
        # Retained ONLY for _install_default_session; never logged (see
        # AwsCredentials.__repr__), dropped with the provider.
        self._credentials = credentials
        self._desktop_env_factory = desktop_env_factory
        self._task_factory = task_factory
        self._sleep = sleep
        self._instance_id: str | None = None
        self._public_ip: str | None = None
        self._env: Any = None
        # The authoritative task instruction, read ONCE at the reset seam
        # through the official accessor (see ``_read_reset_instruction``) and
        # then served on every observation. None until allocate succeeds; a
        # teardown-only provider never allocates and never needs it.
        self._runtime_instruction: str | None = None
        # The episode-scoped, absolute, workspace-external cache root the
        # adapter mints in reset_start. None until allocate; a rescue
        # (teardown-only) provider never allocates and never needs one.
        self._cache_root: Path | None = None
        self._schedule_created = False
        self._supervisor_install_attempted = False
        self._supervisor_unit: str | None = None
        self._supervisor_password = ""
        self._supervisor_restart: int | None = None
        self._supervisor_records: list[Any] = []
        # The last watchdog summary the guest reported. Kept so the evidence of
        # a LATER unreachable read still says what the watchdog had done: an
        # episode that only survived because the watchdog revived the server
        # must not look like one whose server never died.
        self._supervisor_watchdog: dict[str, Any] | None = None
        # Once a recovery was VERIFIED, a later watchdog-only change must not
        # overwrite that snapshot with a ``verified_none`` observation: the
        # episode did restart, and the evidence of it is the more important
        # fact. The flag is per-episode, like the evidence file.
        self._supervisor_recovery_verified = False
        # The provider's OWN secret values, for the exact-value half of the
        # console scrub (``guest_loss``): console text is the one guest-authored
        # artifact this module persists, and the AWS credentials, the session
        # token and the guest client password are the values it holds that the
        # shape table could not recognise by spelling alone. ``allocate`` adds
        # the client password; a teardown-only provider never sees it.
        self._known_secret_values: tuple[str, ...] = tuple(
            value
            for value in (
                credentials.access_key_id,
                credentials.secret_access_key,
                credentials.session_token,
            )
            if value
        )
        # Stages whose console evidence is already on disk. Each stage is taken
        # at most once per provider, and only on a lost-guest path (see
        # ``_capture_guest_console``): a healthy episode never spends the calls.
        self._console_stages_done: set[str] = set()
        # Every retry taken for a provably-unsent request, and how many of them
        # were followed by a send that succeeded. The count answers the one
        # question a post-mortem cannot answer from an episode that merely
        # survived: how much of the apparent transport failure rate was a
        # network blip rather than a dead guest. Persisted on each event (see
        # ``_record_transport_retry``) so it is part of the episode's evidence,
        # not only of the log.
        self._transport_retries: list[dict[str, Any]] = []
        self._transport_retry_recovered = 0
        # The reason behind the most recent boolean probe, for the recovery
        # evidence. ``_probe_control_server`` is boolean by contract (its async
        # RPC caller acts on the boolean alone) and is patched directly by
        # tests, so the record is a side channel that survives that contract.
        self._last_probe_record: dict[str, Any] | None = None
        # Reset-retry tracking for ``_seal_upstream``. Upstream's ``reset``
        # loop writes ``is_environment_used = True`` before task-specific
        # setup and does not clear it when an attempt fails, so its OWN retry
        # re-enters the "used" branch and calls ``_revert_to_snapshot`` -- a
        # retry-hygiene revert the sealed refusal must not turn into an
        # episode death. ``_reset_in_flight`` is true only while the ONE
        # ``env.reset`` this adapter makes (in ``_start_desktop_env``) is
        # executing; ``_reset_completed`` becomes true only once that call
        # returns (and is re-armed with each window); and ``_reset_first_failure``
        # keeps the first failure that triggered a retry, so the trigger is
        # never swallowed.
        self._reset_in_flight = False
        self._reset_completed = False
        self._reset_first_failure: str | None = None

    @classmethod
    def for_teardown(
        cls,
        credentials: AwsCredentials,
        *,
        region: str,
        lease_ref: str,
        clients: _Clients | None = None,
    ) -> "AwsProvider":
        """A provider that can ONLY terminate/delete/describe.

        Built by ``begin_rescue`` from the descriptor's infra values plus the
        freshly delivered secrets. It never allocates, so it needs no plan,
        task, or DesktopEnv — teardown resolves everything by tag.
        """

        return cls(credentials, region=region, lease_ref=lease_ref, clients=clients)

    # ------------------------------------------------------------------
    # allocate
    # ------------------------------------------------------------------

    async def allocate(
        self, plan: ProvisioningPlan, task: TaskDescriptor, *, cache_root: Path
    ) -> None:
        # Typed as required (matching the Protocol), and re-checked at
        # runtime: a DesktopEnv built without a cache root would fall back to
        # upstream's cwd-relative default and re-open the digest-break
        # defect. The type stops a caller that forgets it; this stops one
        # that passes None dynamically.
        if cache_root is None:
            raise AllocationError(
                "allocate needs the adapter's episode cache root (cache_root); "
                "without it upstream writes land in the digest-pinned workspace"
            )
        self._cache_root = cache_root
        if plan.client_password:
            self._known_secret_values = (*self._known_secret_values, plan.client_password)
        if plan.region != self._region:
            raise AllocationError(
                f"plan region {plan.region!r} differs from provider region {self._region!r}"
            )
        instance_id = await asyncio.to_thread(self._run_instance, plan)
        self._instance_id = instance_id
        # THE LEASE IS CREATED BEFORE READINESS, and its failure is fatal. A
        # laptop that dies during the guest's boot is exactly the case the
        # lease exists for; a warning here would be the old pilot's leak.
        try:
            await asyncio.to_thread(self._create_lease, instance_id, plan.scheduler_role_arn)
        except Exception as error:
            await asyncio.to_thread(self._terminate_best_effort, instance_id)
            raise AllocationError(
                f"TTL lease {self._lease_ref!r} could not be created; "
                f"instance {instance_id} was terminated"
            ) from error
        self._schedule_created = True
        self._public_ip = await asyncio.to_thread(self._wait_ready, instance_id)
        # BEFORE upstream's reset, and therefore before the episode's first
        # observation: the guest ships ~93% full and its own snapd fills the
        # rest on a clock, which killed 7 of 8 episodes in a 424-466s window.
        # See ``guest_disk`` for the measurements, and for why a reclamation
        # that did NOT land now refuses the episode here instead of being
        # recorded and stepped over.
        await asyncio.to_thread(self._prepare_guest_disk, plan)
        await asyncio.to_thread(self._install_guest_supervisor, plan)
        await asyncio.to_thread(self._start_desktop_env, plan, task)

    def _install_guest_supervisor(self, plan: ProvisioningPlan) -> None:
        """Optional preparation only; installation must never reject an episode."""
        if self._supervisor_install_attempted:
            return
        self._supervisor_install_attempted = True
        status: dict[str, Any] = {"installed": False, "status": "supervision unavailable"}
        try:
            result = self._run_guest_command(
                guest_supervisor.install_command(plan.client_password), timeout=15.0
            )
            if result.returncode == 0:
                response = guest_supervisor.parse_output(
                    result.stdout, guest_supervisor.is_install_payload
                )
                if isinstance(response, dict) and response.get("installed") is True:
                    self._supervisor_unit = guest_supervisor.installed_unit(result.stdout)
                    self._supervisor_password = plan.client_password
                    status = {
                        "installed": True,
                        "unit": self._supervisor_unit,
                        "status": "supervised",
                    }
                    # ADDITIVE ONLY, and never fabricated: the guest refuses to
                    # install when its watchdog is inactive or shares the
                    # server's cgroup, so a missing proof is recorded as absent
                    # rather than asserted as success.
                    watchdog = guest_supervisor.watchdog_install_status(response)
                    if watchdog is not None:
                        status["watchdog"] = watchdog
                else:
                    reason = (
                        response.get("reason", "invalid response")
                        if isinstance(response, dict)
                        else "invalid response"
                    )
                    safe = guest_loss._scrub(str(reason)[:512], self._known_secret_values)
                    status["cause"] = safe or "installation reason withheld"
                    # ADDITIVE ONLY: a unit-discovery refusal records why the
                    # guest's units looked the way they did. The outcome and the
                    # cause above are already final, and every failure below is
                    # recorded as an absent/undetermined layout, never raised.
                    if str(reason) in guest_supervisor.LAYOUT_REFUSALS:
                        try:
                            status["layout"] = self._capture_supervisor_layout(plan.client_password)
                        except Exception:
                            # The capture is additive evidence. A defect in the
                            # diagnostic itself must not replace the refusal
                            # reason the guest already gave (the outer handler
                            # writes an exception TYPE into cause).
                            status["layout"] = {"schema_version": 1, "status": "error"}
            else:
                status["cause"] = "privilege escalation or installation command failed"
        except Exception as error:
            # Neither guest output nor exception text is safe to persist.
            status["cause"] = type(error).__name__
        self._write_supervisor_evidence("guest-supervisor-setup.json", status)

    def _write_supervisor_evidence(self, name: str, value: object) -> None:
        try:
            if self._cache_root is not None:
                (self._cache_root / name).write_text(json.dumps(value, sort_keys=True) + "\n")
        except Exception:
            logger.warning("guest supervisor evidence unavailable")

    def _capture_supervisor_layout(self, password: str) -> dict[str, Any]:
        """Bounded, fail-open topology capture for a unit-discovery refusal.

        The guest's install script already said WHICH refusal fired; this says
        what the guest's process tree and units actually look like, so a
        refusal can be designed against rather than guessed at. Bounded by ONE
        shared deadline (``guest_loss._run_bounded``, the console-capture
        helper): a hung transport, a hung guest, a malformed document or a
        scrub it cannot vouch for each become a status-only record, and the
        episode continues with the unchanged refusal.
        """
        budget = guest_supervisor.LAYOUT_BUDGET_S
        # The socket deadline is NOT the shared budget. The guest's subprocess
        # deadline is ``guest_deadline_for`` of this value (margin subtracted),
        # and it must exceed LAYOUT_SOURCE's own 3 s deadline or a loaded guest
        # is killed before it can print; ``_run_bounded`` still abandons the
        # whole capture at ``budget``, which stays the authority.
        socket_timeout = guest_supervisor.LAYOUT_GUEST_DEADLINE_S + GUEST_TRANSPORT_MARGIN_S
        try:
            outcome = guest_loss._run_bounded(
                {
                    "layout": lambda: self._run_guest_command(
                        guest_supervisor.layout_command(
                            password, port=guest_supervisor.ACTION_SERVER_PORT
                        ),
                        timeout=socket_timeout,
                    )
                },
                budget,
            )["layout"]
        except Exception:
            return guest_supervisor.empty_layout("error")
        state, value = outcome
        if state == "timeout":
            return guest_supervisor.empty_layout("timeout")
        if state != "ok" or getattr(value, "returncode", None) != 0:
            return guest_supervisor.empty_layout("error")
        try:
            layout = guest_supervisor.layout_record(
                value.stdout, port=guest_supervisor.ACTION_SERVER_PORT
            )
        except Exception:
            # Includes SupervisorOutputUnparseable: guest text never explains
            # itself into the record, only this fixed status does.
            return guest_supervisor.empty_layout("unparseable")
        try:
            safe = guest_loss._scrub(json.dumps(layout), self._known_secret_values)
            if safe is None:
                return guest_supervisor.empty_layout("withheld")
            scrubbed = json.loads(safe)
        except Exception:
            return guest_supervisor.empty_layout("withheld")
        return scrubbed if isinstance(scrubbed, dict) else guest_supervisor.empty_layout("withheld")

    def _recover_guest_supervisor(
        self, delivery: str = "pre_action"
    ) -> guest_supervisor.ReceiptResult:
        """Only a verified receipt may authorize input on an existing binding.

        A failed read is not evidence of no restart. Retry it once, then record
        the uncertainty and force the same fresh-frame boundary as a restart.
        Unreachable-server exhaustion remains distinct from receipt uncertainty.

        The exhaustion branch records EVERY probe's reason, not just the fact
        that all of them failed: six silent probes used to collapse to
        ``read_failed: true``, which names a path failure and not its element.
        The record is evidence and nothing else -- the decision is still the
        boolean, and the six-attempt budget is unchanged.
        """

        if self._supervisor_unit is None:
            return guest_supervisor.ReceiptResult("verified_none")
        probes: list[dict[str, Any]] = []
        for _ in range(6):
            # Clear BEFORE the call and read after: the record must belong to
            # THIS probe. ``_probe_control_server`` is boolean by contract and
            # is patched by callers (tests patch it), so a stale record left by
            # an earlier probe elsewhere (``execute``'s readiness check) would
            # otherwise be reported as this probe's reason.
            self._last_probe_record = None
            answered = self._probe_control_server(0.5)
            record = self._last_probe_record
            probes.append({"probe": len(probes) + 1, **(record or {"answered": answered})})
            if answered:
                break
            self._sleep(0.5)
        else:
            self._write_supervisor_evidence(
                "guest-supervisor-recovery.json",
                {
                    "recovered": False,
                    "read_failed": True,
                    "receipt_unavailable": True,
                    "records_withheld": False,
                    "cause": "supervisor recovery budget exhausted; server unreachable",
                    "probe_outcomes": probes,
                    "records": self._supervisor_records,
                    "watchdog": self._watchdog_evidence(),
                    "final_guest_record": "not retrieved: action-server transport unavailable",
                    "guest_record_path": (
                        f"/var/lib/lop-action-supervisor/{self._supervisor_unit}.json"
                    ),
                },
            )
            return guest_supervisor.ReceiptResult("read_failed", recovery_exhausted=True)

        receipt_failure = "receipt_read_failed"
        for attempt in range(2):
            try:
                result = self._run_guest_command(
                    guest_supervisor.report_command(
                        self._supervisor_unit, self._supervisor_password
                    ),
                    timeout=5.0,
                )
                if result.returncode != 0:
                    raise ValueError("receipt command failed")
                records, watchdog = guest_supervisor.receipt_report(result.stdout)
                # The watchdog repairs the server THROUGH a channel the host
                # does not have, so its record is the only trace of a death the
                # episode survived. Surface it whenever it changes: a completed
                # episode that needed three revivals must not read like one that
                # needed none. Between interventions the summary is unchanged,
                # so a healthy episode writes no extra evidence.
                # The FIRST read is a silent baseline: it establishes what the
                # watchdog had done before this action, and writing it would
                # create a ``recovered: false`` file for an episode that has
                # nothing to recover from. Only a CHANGE is evidence.
                watchdog_changed = (
                    self._supervisor_watchdog is not None and watchdog != self._supervisor_watchdog
                )
                self._supervisor_watchdog = watchdog
                if not records or records[-1]["restart"] == self._supervisor_restart:
                    if watchdog_changed and not self._supervisor_recovery_verified:
                        self._write_supervisor_evidence(
                            "guest-supervisor-recovery.json",
                            {
                                "recovered": False,
                                "read_failed": False,
                                "receipt_unavailable": False,
                                "records_withheld": False,
                                "observed": "verified_none",
                                "watchdog": watchdog,
                                "records": self._supervisor_records,
                            },
                        )
                    return guest_supervisor.ReceiptResult("verified_none")
                # Keep the console evidence's fail-closed treatment. A withheld
                # stderr payload is explicitly marked, never a silent clean [].
                safe = guest_loss._scrub(json.dumps(records), self._known_secret_values)
                self._supervisor_records = json.loads(safe) if safe is not None else []
                # A RECEIPT ONLY SAYS systemd's MainPID was replaced. Under
                # ``KillMode=process`` every other holder survives that restart
                # and can keep answering the action port with pre-restart screen
                # state, so the receipt alone is NOT recovery. Bounce every
                # process in the unit's cgroup and prove the listener is one
                # that did not hold the port before; an unproved listener fails
                # closed instead of authorizing input against a survivor.
                verification = self._restart_supervisor_unit(self._supervisor_unit)
                if verification["status"] != "verified":
                    status = str(verification["status"])
                    self._write_supervisor_evidence(
                        "guest-supervisor-recovery.json",
                        {
                            "recovered": False,
                            "read_failed": False,
                            "receipt_unavailable": False,
                            "records_withheld": safe is None,
                            "cause": guest_supervisor.restart_cause(status),
                            "cause_status": status,
                            "restart_verification": verification,
                            "failed_action_delivery": delivery,
                            "remaining_batch": "not_sent",
                            "records": self._supervisor_records,
                            "watchdog": self._watchdog_evidence(),
                        },
                    )
                    # The existing recovery failure, with a fixed cause: the
                    # caller stops before posting the pending action.
                    return guest_supervisor.ReceiptResult("read_failed", recovery_exhausted=True)
                # Commit the generation the restart itself produced. Without
                # this, the next poll would read the same record as a second
                # restart and bounce a healthy server in a loop.
                self._supervisor_restart = verification["latest_restart"]
                self._supervisor_recovery_verified = True
                self._write_supervisor_evidence(
                    "guest-supervisor-recovery.json",
                    {
                        "recovered": True,
                        "read_failed": False,
                        "receipt_unavailable": False,
                        "records_withheld": safe is None,
                        "restart_verification": verification,
                        "failed_action_delivery": delivery,
                        "remaining_batch": "not_sent",
                        "records": self._supervisor_records,
                        "watchdog": watchdog,
                    },
                )
                return guest_supervisor.ReceiptResult("new_restart")
            except Exception as error:
                receipt_failure = (
                    "systemd_generation_disagreement"
                    if isinstance(error, guest_supervisor.GenerationMismatch)
                    else "receipt_read_failed"
                )
                if attempt == 0:
                    self._sleep(0.1)
        self._write_supervisor_evidence(
            "guest-supervisor-recovery.json",
            {
                "recovered": False,
                "read_failed": True,
                "receipt_unavailable": True,
                "records_withheld": False,
                "read_attempts": 2,
                "cause": "supervisor receipt unavailable",
                "receipt_failure": receipt_failure,
                "failed_action_delivery": delivery,
                "remaining_batch": "not_sent",
                "records": self._supervisor_records,
                "watchdog": self._watchdog_evidence(),
            },
        )
        return guest_supervisor.ReceiptResult("read_failed")

    def _watchdog_evidence(self) -> dict[str, Any]:
        """The last watchdog summary we read, or an explicit absent record.

        Never fabricated: ``absent`` means no summary was ever readable in this
        episode, which is itself the diagnostic a missing record carries.
        """

        return self._supervisor_watchdog or guest_supervisor.empty_watchdog("absent")

    def _restart_supervisor_unit(self, unit: str) -> dict[str, Any]:
        """Detached restart + freshness proof; never raises.

        THE DISPATCH MAY LOSE ITS RESPONSE. The supervised unit is the process
        serving this request, so the all-process kill terminates our HTTP
        connection -- and, if the restart ran in the target cgroup, would
        terminate the restart itself. The restart therefore runs as a transient
        service and writes its verdict to ``RESTART_RESULT_PATH``; this method
        treats the dispatch as fire-and-forget (a transport error is expected,
        a clean non-zero return is a failure) and reads the result from a LATER
        command once the server answers again. A token ties the result to this
        dispatch, so a stale file cannot be read as a fresh restart.
        """

        token = time.time_ns()
        try:
            dispatched = self._run_guest_command(
                guest_supervisor.restart_command(unit, self._supervisor_password, token),
                timeout=guest_supervisor.RESTART_DISPATCH_TIMEOUT_S,
            )
            if dispatched.returncode != 0:
                return guest_supervisor.empty_restart("command_failed")
        except Exception:
            # Expected: the kill closes the connection that carried the request.
            pass
        deadline = time.monotonic() + guest_supervisor.RESTART_RESULT_BUDGET_S
        while time.monotonic() < deadline:
            try:
                result = self._run_guest_command(
                    guest_supervisor.restart_result_command(self._supervisor_password),
                    timeout=guest_supervisor.RESTART_RESULT_TIMEOUT_S,
                )
                if result.returncode == 0 and result.stdout.strip():
                    record = guest_supervisor.restart_record(result.stdout)
                    if record["token"] == token:
                        return record
            except Exception:
                # The server is down for part of this window, or the file holds
                # a malformed/stale document; keep polling until the budget ends.
                pass
            self._sleep(guest_supervisor.RESTART_RESULT_INTERVAL_S)
        return guest_supervisor.empty_restart("result_unavailable")

    def _run_guest_command(
        self, command: Sequence[str], timeout: float
    ) -> guest_disk.CommandResult:
        """One command in the guest, through its own control server.

        Raises on any transport failure; ``guest_disk`` catches that and records
        the guest as unreachable. The response shape is upstream's own
        (``{"status", "output", "error", "returncode"}``, setup.py:425-441), and
        a missing ``returncode`` reads as a failure rather than a success: a
        server that did not say the command succeeded did not say it succeeded.
        """

        url = f"http://{self._public_ip}:{GUEST_PORT}{_EXECUTE_PATH}"
        # ``timeout`` is sent in the BODY as well as applied to the socket, and
        # the two are deliberately NOT the same number. The body value is the
        # guest's own ``subprocess.run`` deadline; omitting it let the guest
        # default to 120 s, outside our socket deadline, so a slow command
        # expired on our side while still running on the guest's — an unknown
        # outcome the no-retry policy can only turn into a dead episode.
        # ``guest_deadline_for`` keeps the guest strictly first for whatever
        # socket deadline this caller actually passed.
        body = self._clients.http_post_json(
            url,
            {
                "command": list(command),
                "shell": False,
                "timeout": guest_deadline_for(timeout),
            },
            timeout,
        )
        returncode = body.get("returncode")
        return guest_disk.CommandResult(
            returncode=returncode if type(returncode) is int else 1,
            stdout=str(body.get("output") or ""),
            stderr=str(body.get("error") or ""),
        )

    def _prepare_guest_disk(self, plan: ProvisioningPlan) -> None:
        """Reclaim the guest's root filesystem, record it, and refuse a dead guest.

        THE REPORT IS ALWAYS WRITTEN FIRST, then the refusal is raised, so the
        evidence of WHY the episode ended is in the bundle even though the
        episode never started. Keeping the partial record is the point: the four
        paid episodes this refusal exists for left nothing to read but a
        transport error ~400 s later.

        WHAT IS REFUSED, exactly: ``GuestDiskReport.blocking_steps`` -- a
        reclamation step that did not land (failed, unreachable, or skipped for
        want of budget), with a failed ``snap refresh --hold`` covered by its
        successful ``refresh.hold`` fallback. A step-side failure is the
        measured signature of a wrong ``OSWORLD_CLIENT_PASSWORD`` (every
        privileged step returned ``sudo: no password was provided`` / ``1
        incorrect password attempt``) and the guest then fills its root
        filesystem within minutes, so walking on buys nothing but model spend.

        WHAT IS NOT REFUSED: a guest that reclaimed successfully and is still
        short of free space. That is the normal completed state (measured: ~2.2
        GB free after a successful hold+clear, on the run that scored), the
        threshold is 12 GiB, and aborting on the shortfall would refuse every
        healthy episode while protecting nothing -- the hold is what stops the
        clock, not the number of bytes free at t+0.

        The blanket ``except`` on the probe itself stays, on top of
        ``guest_disk``'s own per-step fail-soft: it catches a contract violation
        (or an injected runner that misbehaves) rather than a normal failure
        mode, and there is then no report to write or to act on.
        """

        report: guest_disk.GuestDiskReport | None = None
        try:
            report = guest_disk.prepare_guest_disk(
                self._run_guest_command,
                client_password=plan.client_password,
                clock=time.monotonic,
            )
        except Exception:
            return
        if self._cache_root is not None:
            try:
                (self._cache_root / GUEST_PREPARATION_FILENAME).write_bytes(report.to_json_bytes())
            except OSError:
                # An unwritable cache root is the adapter's problem to surface
                # elsewhere (upstream writes there too); losing the hygiene report
                # must not be the thing that ends the episode. The refusal below
                # still fires: it does not depend on the file being readable.
                pass
        blockers = report.blocking_steps()
        if blockers:
            names = ", ".join(step.name for step in blockers)
            detail = "; ".join(
                f"{step.name} {step.status}"
                + (f" rc={step.returncode}" if step.returncode is not None else "")
                + (f" ({step.detail})" if step.detail else "")
                for step in blockers
            )
            raise AllocationError(
                f"guest disk reclamation did not land at preparation: {names}. "
                "The episode is refused here, before any model spend, because a "
                "guest whose snapd is still filling the root filesystem dies of "
                "'no space left on device' a few hundred seconds in, and reports "
                "it as an unrelated transport error. Check OSWORLD_CLIENT_PASSWORD "
                "first: a value this image rejects fails every privileged step "
                "with 'sudo: no password was provided' / '1 incorrect password "
                f"attempt'. Full step record: {GUEST_PREPARATION_FILENAME}. "
                f"Failed steps: {detail}"
            )

    def _run_instance(self, plan: ProvisioningPlan) -> str:
        ec2 = self._clients.ec2
        volume_gb = plan.volume_gb
        if volume_gb is None:
            volume_gb = self._resolve_root_volume_size(plan.ami_id)
        else:
            # A pinned size is checked against the AMI's OWN snapshot before
            # run_instances. AWS refuses a root volume smaller than the
            # snapshot it is restored from, and does it with an
            # InvalidBlockDeviceMapping message that names neither the AMI's
            # size nor the one asked for -- so the operator learns only that
            # something was wrong, mid-launch. Checking here costs one
            # read-only describe_images on the path that already issues one in
            # the None branch, and turns that into a message carrying both
            # numbers. It cannot move to ``provisioning.resolve``: prepare is
            # declarative and issues no I/O at all, which is exactly what lets
            # it run before anything is allocated.
            self._refuse_volume_smaller_than_ami(plan.ami_id, volume_gb)
        tags = [{"Key": key, "Value": value} for key, value in plan.tags]
        response = ec2.run_instances(
            MaxCount=1,
            MinCount=1,
            ImageId=plan.ami_id,
            InstanceType=plan.instance_type,
            EbsOptimized=True,
            InstanceInitiatedShutdownBehavior="terminate",
            ClientToken=plan.client_token,
            NetworkInterfaces=[
                {
                    "DeviceIndex": 0,
                    "SubnetId": plan.subnet_id,
                    "AssociatePublicIpAddress": True,
                    "Groups": [plan.security_group_id],
                }
            ],
            BlockDeviceMappings=[
                {
                    "DeviceName": "/dev/sda1",
                    "Ebs": {
                        "VolumeSize": volume_gb,
                        "VolumeType": "gp3",
                        "Throughput": 1000,
                        "Iops": 4000,
                        "DeleteOnTermination": True,
                    },
                }
            ],
            # Both the instance AND its root volume carry the tags: the
            # audit lists volumes too, and a volume that outlived its
            # instance (DeleteOnTermination is a default, not a law) would
            # otherwise be invisible to the leak detector.
            TagSpecifications=[
                {"ResourceType": "instance", "Tags": tags},
                {"ResourceType": "volume", "Tags": tags},
            ],
        )
        instances = response.get("Instances", [])
        if len(instances) != 1 or "InstanceId" not in instances[0]:
            raise AllocationError("run_instances returned no instance")
        return str(instances[0]["InstanceId"])

    def _resolve_root_volume_size(self, ami_id: str) -> int:
        """The AMI's own root BDM size, mirroring OSWorld's resolver.

        OSWorld defaults to 40 GiB and grows to the AMI's root size
        (manager.py:105-140). The release AMI is 30 GiB, so the result is 40;
        the lookup exists so a future AMI larger than 40 does not fail to
        boot. The snapshot lookup is skipped: the BDM already carries the size.
        """

        default = 40
        return max(default, self._ami_root_volume_size(ami_id))

    def _ami_root_volume_size(self, ami_id: str) -> int:
        """The AMI's root BlockDeviceMapping size in GiB, with no floor applied.

        Split out from ``_resolve_root_volume_size`` because the two callers
        need different numbers from the same lookup. The default path wants
        OSWorld's 40 GiB floor; the override check wants the AMI's REAL size,
        since that -- not the floor -- is what AWS refuses to shrink below. One
        function returning max(40, ami) would accept a 35 GiB override against
        a 30 GiB AMI and reject one against a 45 GiB AMI with the wrong number
        in the message.

        Returns 0 when the image declares no EBS root mapping, which makes the
        floor check vacuous rather than wrong: an AMI whose size cannot be read
        must not manufacture a constraint the operator cannot satisfy.
        """

        response = self._clients.ec2.describe_images(ImageIds=[ami_id])
        images = response.get("Images", [])
        if not images:
            raise AllocationError(f"AMI {ami_id} is not visible in {self._region}")
        image = images[0]
        root = image.get("RootDeviceName")
        size = 0
        for mapping in image.get("BlockDeviceMappings", []):
            ebs = mapping.get("Ebs")
            if not ebs:
                continue
            if root is None or mapping.get("DeviceName") == root:
                size = max(size, int(ebs.get("VolumeSize", 0)))
                break
        return size

    def _refuse_volume_smaller_than_ami(self, ami_id: str, volume_gb: int) -> None:
        """Fail a root volume smaller than the AMI's snapshot, naming both sizes.

        EBS cannot restore a snapshot into a volume smaller than itself, so
        this launch is doomed; the only question is whether the operator finds
        out from a message naming the two numbers or from botocore's
        InvalidBlockDeviceMapping, which names neither.
        """

        ami_gb = self._ami_root_volume_size(ami_id)
        if ami_gb and volume_gb < ami_gb:
            raise AllocationError(
                f"root volume {volume_gb} GiB is smaller than AMI {ami_id}'s own "
                f"snapshot ({ami_gb} GiB); EBS cannot restore a snapshot into a "
                f"smaller volume, so raise AWS_ROOT_VOLUME_SIZE to at least {ami_gb}"
            )

    def _create_lease(self, instance_id: str, role_arn: str) -> None:
        fire_at = datetime.now(timezone.utc) + timedelta(seconds=self._ttl_seconds)
        self._clients.scheduler.create_schedule(
            Name=self._lease_ref,
            ScheduleExpression=f"at({fire_at.strftime('%Y-%m-%dT%H:%M:%S')})",
            FlexibleTimeWindow={"Mode": "OFF"},
            ActionAfterCompletion="DELETE",
            State="ENABLED",
            Description=f"lop osworld-v2 TTL terminate for {instance_id}",
            Target={
                "Arn": _TERMINATE_TARGET_ARN,
                "RoleArn": role_arn,
                "Input": json.dumps({"InstanceIds": [instance_id]}),
            },
        )

    def _terminate_best_effort(self, instance_id: str) -> None:
        # Used only on the lease-failure path inside allocate. Errors are
        # swallowed because the AllocationError about to be raised is the
        # signal; the runner's cleanup will re-terminate by tag and confirm.
        try:
            self._clients.ec2.terminate_instances(InstanceIds=[instance_id])
        except Exception:
            pass

    def _wait_ready(self, instance_id: str) -> str:
        """Wait for ``running``, then for the guest service to answer."""

        ec2 = self._clients.ec2
        waiter = ec2.get_waiter("instance_running")
        waiter.wait(InstanceIds=[instance_id], WaiterConfig={"Delay": 5, "MaxAttempts": 60})
        response = ec2.describe_instances(InstanceIds=[instance_id])
        found = _instances(response)
        if not found:
            raise AllocationError(f"instance {instance_id} vanished after running")
        public_ip = found[0].get("PublicIpAddress")
        if not public_ip:
            raise AllocationError(f"instance {instance_id} has no public IP")
        url = f"http://{public_ip}:{GUEST_PORT}{_READY_PATH}"
        deadline = time.monotonic() + self._readiness_timeout_s
        while True:
            try:
                if self._clients.http_get(url, 5.0) == 200:
                    return str(public_ip)
            except Exception:
                pass
            if time.monotonic() >= deadline:
                raise ReadinessTimeout(
                    f"guest {instance_id} at {public_ip} did not answer "
                    f"{_READY_PATH} within {self._readiness_timeout_s:.0f}s"
                )
            self._sleep(5.0)

    def _start_desktop_env(self, plan: ProvisioningPlan, task: TaskDescriptor) -> None:
        # Imports are deferred to here (and injected in tests): desktop_env
        # pulls gymnasium/pyautogui and reads the environment at import.
        if self._desktop_env_factory is not None:
            factory = self._desktop_env_factory
        else:
            from lop_osworld_v2_adapter import vendor_bridge

            factory = vendor_bridge.load_desktop_env()
        if self._task_factory is not None:
            task_instance = self._task_factory(task)
        else:
            from lop_osworld_v2_adapter import vendor_bridge

            task_instance = vendor_bridge.instantiate_task(task.module_name, task.task_id)
        # Upstream builds its own boto3 clients ambiently (``boto3.client(
        # 'ec2', region_name=...)`` in start_emulator/get_ip_address) with
        # NO credentials argument. The worker environment is stripped, so
        # those calls would fail with NoCredentialsError -- unless the
        # process default session carries the delivered values. This is the
        # one place credentials leave our Session object, and it is still
        # in-process memory, never the environment.
        self._install_default_session(plan.region)
        env = factory(
            provider_name="aws",
            region=plan.region,
            # path_to_vm=<instance id> makes DesktopEnv skip its own
            # allocation (desktop_env.py:110-113) and adopt ours.
            path_to_vm=self._instance_id,
            snapshot_name=plan.ami_id,
            # THE FIX for the rescue-wedge defect: upstream's default is
            # cache_dir="cache", resolved against the cwd — and the worker's
            # cwd IS the digest-pinned workspace. Routing the base under the
            # episode's owned root (absolute, outside the workspace) keeps
            # _download_setup / reset_cache_dir / getter downloads off the
            # pin, so the rescue worker's digest re-check still matches.
            cache_dir=str(self._cache_root),
            action_space="pyautogui",
            screen_size=plan.screen,
            headless=True,
            require_a11y_tree=False,
            os_type="Ubuntu",
            enable_proxy=plan.enable_proxy,
            client_password=plan.client_password,
            instance_type=plan.instance_type,
            use_public_ip=True,
        )
        self._seal_upstream(env)
        # The ONE reset call is the window in which upstream's retry loop
        # runs; ``_reset_completed`` (set only after reset RETURNS) is what
        # keeps a later revert -- a second reset on a genuinely used env --
        # sealed. All three window fields are re-armed together, completion
        # included: a provider that ever served a SECOND environment must
        # classify that env's first-reset retry inside ITS window, or env #2
        # would die on the sealed revert -- the original defect's shape
        # (review round 1, R1-F2; latent today, since every reset_start
        # builds a fresh provider). See ``_seal_upstream`` for the retry
        # discrimination.
        self._reset_first_failure = None
        self._reset_completed = False
        self._reset_in_flight = True
        try:
            env.reset(task_config=task_instance)
        except Exception as error:
            # The retry loop's own failure, with the FIRST attempt's error
            # attached: a run that dies here must be able to name the
            # transient failure it died retrying through.
            self._enrich_reset_failure(error)
            raise
        else:
            self._reset_completed = True
        finally:
            self._reset_in_flight = False
        # THE POST-RESET INSTRUCTION READ (why it is here and must not move).
        # Upstream's ``_set_task_info`` (called before the setup step this
        # call just drove -- :370 vs :375 on the v2.1 pin, :364 vs :368 on the
        # 2.0 checkout) caches the mapping instruction into ``self.instruction``,
        # and ``_get_obs`` (:455-460) returns that cache -- so for a task whose
        # instruction is computed from setup/profile state, the reset frame
        # would carry the STALE pre-setup value. The published hybrid baselines
        # read the task mapping AFTER setup for exactly this reason. One read,
        # once, at the one seam where "setup has just succeeded" is a fact; the
        # value is cached for the observations and re-read NEVER (a per-frame
        # getter would re-invoke task code and could disagree with itself).
        # Because this runs inside ``allocate``, a missing/blank/unreadable
        # instruction refuses BEFORE any observation exists, therefore before
        # any model request: this must not be softened into a fallback, and a
        # future edit that moves it before ``env.reset`` reads the stale
        # cache, while one that drops it silently reverts the model to that
        # same stale value.
        self._runtime_instruction = _read_reset_instruction(env)
        self._env = env

    def _install_default_session(self, region: str) -> None:
        if self._credentials is None:
            return
        try:
            import boto3  # type: ignore[import-not-found]
        except ImportError:  # pragma: no cover - only reachable without the paid extra
            return
        boto3.setup_default_session(
            aws_access_key_id=self._credentials.access_key_id,
            aws_secret_access_key=self._credentials.secret_access_key,
            aws_session_token=self._credentials.session_token,
            region_name=region,
        )

    def _seal_upstream(self, env: Any) -> None:
        """Replace every upstream allocate/revert/release path with a raiser.

        Applied to the LIVE env object (not the class) immediately after
        construction and before ``reset``: ``DesktopEnv.__init__`` has
        already called ``provider.start_emulator``/``get_ip_address`` on our
        instance, which is the only upstream provider work we want. From
        here on, ``_revert_to_snapshot`` (a second ``reset`` once the env is
        used), ``provider.revert_to_snapshot`` (the untagged relaunch),
        ``manager.get_vm_path`` (upstream's own allocation), and
        ``close``/``stop_emulator`` (unconfirmed terminate) all raise
        ``UpstreamAllocationRefused`` BEFORE any boto3 call.

        ONE of those paths is not an unconditional raiser, because upstream
        asks for it as retry hygiene rather than as a revert of used state:
        its reset loop writes ``is_environment_used = True`` before
        task-specific setup and does not clear it when an attempt fails, so
        the loop's own retry calls ``_revert_to_snapshot`` (and then
        ``_start_emulator``) on an environment where nothing has run yet.
        Inside that first reset the revert is SKIPPED and
        ``_start_emulator`` is skipped with it -- the environment is our own
        tagged instance, it has no completed setup to revert to, and
        skipping performs no boto3 call at all. Everywhere else -- a
        completed reset, a revert with no reset in flight, the direct
        provider/manager paths -- the refusal stands unchanged.

        The env is sealed at the instance level so the seal cannot leak
        into another ``DesktopEnv`` in the same process, and it survives
        upstream re-assigning ``self.provider``/``self.manager`` in
        ``_revert_to_snapshot`` only because that method is itself sealed
        first. Missing attributes are tolerated so an upstream refactor
        fails loudly at the seal's own test rather than silently here.
        """

        def refuse(name: str) -> Callable[..., Any]:
            def _refused(*_args: Any, **_kwargs: Any) -> Any:
                message = (
                    f"upstream DesktopEnv.{name} would allocate, revert, or release an "
                    "EC2 instance outside the adapter's tagged, leased, "
                    "rescue-resolvable path; refused before any boto3 call"
                )
                if self._reset_first_failure is not None:
                    # A refusal on an env whose reset already had to recover
                    # from a failed first attempt names that attempt -- and
                    # the segment LEADS, so the wire's head-keep cannot drop
                    # it behind the (fixed-size) refusal sentence.
                    message = f"{_first_failure_segment(self._reset_first_failure)} {message}"
                raise UpstreamAllocationRefused(message)

            return _refused

        def guarded_revert_to_snapshot() -> None:
            # Upstream's retry loop calls this at the top of every attempt
            # after a failed one; inside the FIRST reset that is the loop
            # retrying its own setup, not a used environment being reverted
            # (nothing has run, and no reset has completed). Skip so the
            # retry can proceed; refuse -- as before this guard -- anywhere
            # else.
            if self._reset_retry_in_flight():
                self._note_retry_skip("_revert_to_snapshot")
                return
            refuse("_revert_to_snapshot")()

        env._revert_to_snapshot = guarded_revert_to_snapshot
        env.close = refuse("close")
        env._save_state = refuse("_save_state")
        provider = getattr(env, "provider", None)
        if provider is not None:
            provider.revert_to_snapshot = refuse("provider.revert_to_snapshot")
            provider.stop_emulator = refuse("provider.stop_emulator")
            provider.save_state = refuse("provider.save_state")
        manager = getattr(env, "manager", None)
        if manager is not None:
            manager.get_vm_path = refuse("manager.get_vm_path")

        # Follows the guarded revert: the loop calls ``_start_emulator``
        # right after a revert, and the instance the retry would "start" is
        # our own, already running and readied in ``allocate``. Skipped only
        # under the same predicate; everywhere else the upstream method runs
        # exactly as it did before this seal existed -- it only ever starts
        # OUR instance, and it is exactly the method ``__init__`` relies on.
        start_emulator = getattr(env, "_start_emulator", None)
        if callable(start_emulator):

            def guarded_start_emulator() -> Any:
                if self._reset_retry_in_flight():
                    self._note_retry_skip("_start_emulator")
                    return None
                return start_emulator()

            env._start_emulator = guarded_start_emulator

        # Capture whatever makes the loop retry, so the trigger of the
        # hazard above is persistable: an exception out of ``_setup_task``,
        # an unsuccessful ``_setup_task`` return, a control server that
        # never answered (``ensure_ready`` False), or a failed proxy setup
        # all leave ``is_environment_used`` set the same way.
        setup_task = getattr(env, "_setup_task", None)
        if callable(setup_task):

            def tracked_setup_task(*args: Any, **kwargs: Any) -> Any:
                try:
                    outcome = setup_task(*args, **kwargs)
                except Exception as error:
                    self._record_reset_setup_failure(f"{type(error).__name__}: {error}")
                    raise
                if isinstance(outcome, tuple) and outcome and not outcome[0]:
                    self._record_reset_setup_failure("_setup_task reported an unsuccessful setup")
                return outcome

            env._setup_task = tracked_setup_task

        setup_controller = getattr(env, "setup_controller", None)
        if setup_controller is not None:
            for name, label in (
                ("ensure_ready", "the guest control server was not ready"),
                ("_proxy_setup", "proxy setup failed"),
            ):
                original = getattr(setup_controller, name, None)
                if callable(original):
                    setattr(
                        setup_controller,
                        name,
                        self._track_controller_result(original, label),
                    )

    def _reset_retry_in_flight(self) -> bool:
        """True while the FIRST ``env.reset`` of this env is still running.

        The discrimination the guarded revert needs, in one sentence: a loop
        retrying its first setup has nothing to revert TO -- no reset has
        completed (so no step is reachable; steps only happen after reset
        returns) and no setup has succeeded -- while a revert of a genuinely
        used environment always follows a completed reset or runs outside
        one. ``_reset_completed`` closes the window the moment reset
        returns.
        """

        return self._reset_in_flight and not self._reset_completed

    def _record_reset_setup_failure(self, failure: str) -> None:
        """Keep the FIRST setup failure of the in-flight reset.

        First, not last: the loop's final error already names the last
        failure, and the first is the one nothing else records -- it is the
        trigger the retry exists to survive, and the fact a reader needs to
        tell a transient hiccup from a persistent fault.
        """

        if self._reset_in_flight and self._reset_first_failure is None:
            self._reset_first_failure = failure

    def _track_controller_result(
        self, original: Callable[..., Any], label: str
    ) -> Callable[..., Any]:
        """Wrap a ``SetupController`` result so a ``False`` is recorded.

        The loop's non-raising continue paths (``ensure_ready`` and
        ``_proxy_setup`` returning False) leave ``is_environment_used`` set
        exactly as a raised setup error does, so they are equally good
        triggers of the sealed revert and would otherwise stay invisible.
        """

        def tracked(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            if result is False:
                self._record_reset_setup_failure(label)
            return result

        return tracked

    def _note_retry_skip(self, name: str) -> None:
        """Record one sealed call skipped for the retry's sake."""

        logger.warning(
            "upstream reset is retrying its first setup; skipping %s instead of "
            "dying on the sealed revert. The environment is this adapter's own "
            "tagged instance with no completed setup to revert to, so the retry "
            "re-runs setup on it. First failure: %s",
            name,
            self._reset_first_failure or "not captured",
        )

    def _enrich_reset_failure(self, error: Exception) -> None:
        """Attach the first setup attempt's failure to the reset's error.

        ``reset`` raising after its retries already names the LAST failure;
        without this, the FIRST -- the reason the loop was retrying at all --
        would survive only in ``_reset_first_failure`` and the skip log. The
        segment leads the message (bounded, ahead of the variable-length
        text: see ``_first_failure_segment``) and the message is rewritten
        through ``args``, the one surface every exception accepts: the class,
        the traceback and the cause chain (the parts the wire diagnostic
        renders) all stay intact.
        """

        note = self._reset_first_failure
        if note is None or _FIRST_FAILURE_MARKER in str(error):
            return
        try:
            error.args = (
                f"{_first_failure_segment(note)} {error}",
                *error.args[1:],
            )
        except Exception:  # pragma: no cover - ``args`` accepts any tuple
            logger.exception("could not attach the first setup failure to %r", error)

    # ------------------------------------------------------------------
    # episode I/O
    # ------------------------------------------------------------------

    def _require_env(self) -> Any:
        if self._env is None:
            raise AllocationError("provider has no live DesktopEnv (allocate first)")
        return self._env

    async def observe(self) -> dict[str, Any]:
        env = self._require_env()
        if self._supervisor_unit is not None:
            # Consume the generation BEFORE reading the frame. Consuming it
            # afterwards could bless a screenshot taken before that restart.
            await asyncio.to_thread(self._recover_guest_supervisor, "observation")
        # Attached only for the duration of upstream's own retry loop, exactly as
        # ``evaluate`` captures judge errors. The records are the ONLY trace of
        # why a capture failed (``get_screenshot`` logs and returns None), and
        # this call is the last place they exist.
        capture = _ScreenshotFailureCapture(self._guest_origin())
        screenshot_logger = logging.getLogger(_SCREENSHOT_LOGGER)
        screenshot_logger.addHandler(capture)
        started = time.monotonic()
        try:
            raw = await asyncio.to_thread(env._get_obs)
        finally:
            screenshot_logger.removeHandler(capture)
        observation = dict(raw)
        if self._runtime_instruction is not None:
            # The environment's own post-setup read is the instruction the
            # model must see (see ``_read_reset_instruction``): override
            # whatever ``_get_obs`` cached at ``_set_task_info`` time. This
            # copies the value read ONCE at the reset seam -- the accessor is
            # never called per frame, and never from here.
            observation["instruction"] = self._runtime_instruction
        if observation.get("screenshot") is None:
            # A frameless read with no captured record adds NOTHING: an absent
            # cause must stay absent rather than be invented, and the builder
            # then fails with the same fixed text it always did.
            cause = capture.cause(elapsed_ms=_since_ms(started, time.monotonic()))
            if cause is not None:
                observation[SCREENSHOT_CAUSE_KEY] = cause
            # The guest is failing to show its screen: the first moment a lost
            # guest is visible. Take the console now, while the instance exists
            # and BEFORE the harness spends minutes retrying; the read below
            # never raises and runs once. It does not change what this method
            # returns, so the committed-action / re-read handling is untouched.
            await asyncio.to_thread(
                self._capture_guest_console, guest_loss.STAGE_OBSERVE, "observe-frameless"
            )
        return observation

    def _guest_origin(self) -> str | None:
        """``scheme://host:port`` the guest transport targets, or None before allocate."""

        if self._public_ip is None:
            return None
        return f"http://{self._public_ip}:{GUEST_PORT}"

    def _capture_guest_console(self, stage: str, trigger: str) -> None:
        """Persist the guest's console tail and screenshot, once per stage. Never raises.

        Called only from lost-guest paths, BEFORE anything terminates the
        instance (see ``guest_loss`` for what is captured, why it is safe to
        persist, and why it cannot block). Needs the instance id and the episode
        cache root; a teardown-only provider has neither and does nothing.
        """

        if stage in self._console_stages_done:
            return
        self._console_stages_done.add(stage)
        if self._instance_id is None:
            return
        try:
            guest_loss.capture_guest_console(
                self._clients.ec2,
                self._instance_id,
                self._cache_root,
                stage=stage,
                trigger=trigger,
                known_values=self._known_secret_values,
            )
        except Exception:  # pragma: no cover - capture_guest_console never raises
            logger.warning("guest console capture failed unexpectedly")

    def _execute_guest_health_cause(self) -> str:
        cause = "readiness check failed"
        try:
            if self._instance_id is not None:
                health = guest_loss.capture_guest_health(
                    self._clients.ec2, self._instance_id, self._cache_root, self._run_guest_command
                )
                cause = health["cause"]
        except Exception:
            # Keep the delivery verdict even if diagnostics violate fail-open.
            logger.warning("guest health capture failed unexpectedly")
        return cause

    def transport_retry_summary(self) -> dict[str, Any]:
        """How much of this episode's transport failure was recovered by a retry.

        ``retries`` counts scheduled re-sends, ``recovered`` those followed by a
        send that succeeded. The same numbers are on disk after every event
        (``guest-transport-retries.json`` in the episode cache root), so a run
        that only survived because a path blinked can be told apart from one
        that never failed.
        """

        return {
            "retries": len(self._transport_retries),
            "recovered": self._transport_retry_recovered,
            "causes": sorted({str(event["cause"]) for event in self._transport_retries}),
        }

    def _persist_transport_retries(self) -> None:
        """Write the retry ledger. Fail-open: evidence never breaks an episode."""

        self._write_supervisor_evidence(
            "guest-transport-retries.json",
            {
                "recovered": self._transport_retry_recovered,
                "retries": len(self._transport_retries),
                "events": self._transport_retries,
            },
        )

    def _record_transport_retry(
        self,
        *,
        index: int,
        attempt: int,
        verdict: guest_loss.TransportVerdict,
        delay_s: float,
        ready: bool | None,
        error: BaseException,
    ) -> None:
        """One structured event per retry, logged AND persisted.

        Values only. Exception text can echo the guest's address or the command,
        so it is reduced to its type name, and the transport verdict to the
        classifier's closed vocabulary (see ``guest_loss.failure_record``).
        ``attempt`` names the send ABOUT to happen, so ``attempt=2`` on the first
        retry reads as it always has in this module's logs.
        """

        event = {
            "statement": index,
            "attempt": attempt + 1,
            "cause": verdict.kind or "unknown",
            "delivery": verdict.delivery,
            "delay_ms": int(round(delay_s * 1000)),
            "readiness_passed": ready,
            "error_class": type(error).__name__,
        }
        self._transport_retries.append(event)
        self._persist_transport_retries()
        logger.warning(
            "guest transport retry attempt=%d delivery=%s error_class=%s "
            "readiness_passed=%s cause=%s delay_ms=%d",
            attempt + 1,
            verdict.delivery,
            event["error_class"],
            ready,
            event["cause"],
            event["delay_ms"],
        )

    async def execute(self, statements: list[str], *, settle: bool = True) -> None:
        env = self._require_env()

        def run() -> None:
            for index, statement in enumerate(statements):
                # A restart can finish BETWEEN calls without any refusal. Check
                # each action, not just the first statement of a long batch.
                if self._supervisor_unit is not None:
                    recovered = self._recover_guest_supervisor()
                    if recovered.recovery_exhausted:
                        # Do not send an action just to discover the server came
                        # back after the deadline: that would bypass re-binding.
                        self._capture_guest_console(
                            guest_loss.STAGE_EXECUTE, "supervisor-exhausted"
                        )
                        cause = self._execute_guest_health_cause()
                        verdict = guest_loss.TransportVerdict(guest_loss.DELIVERY_NOT_SENT)
                        message = guest_loss.describe_failed_action(index, verdict)
                        raise GuestExecutionError(
                            message + "; supervisor recovery budget exhausted; "
                            f"guest down: {cause}; no transport retry",
                            delivery=verdict.delivery,
                        )
                    if recovered.state != "verified_none":
                        raise GuestServerRecovered(
                            "action-server restart or receipt unavailable; refresh binding"
                        )
                # Upstream retries failed HTTP requests and logs their full body.
                # A lost response can already have typed text, so reuse our
                # single-shot transport, preserving the controller's input patch.
                script = env.controller.pkgs_prefix.format(command=statement)
                from local_operator.computer_input import python_source_argv

                command = python_source_argv(script)
                ready: bool | None = None
                supervisor_exhausted = False
                for attempt in range(1, TRANSPORT_RETRY_ATTEMPTS + 1):
                    try:
                        result = self._run_guest_command(command, timeout=GUEST_COMMAND_TIMEOUT_S)
                        if attempt > 1:
                            # The send that succeeded was itself a retry: this is
                            # the count that says how much of the episode's
                            # apparent transport loss was a network blip.
                            self._transport_retry_recovered += 1
                            self._persist_transport_retries()
                        break
                    except Exception as error:
                        # Only positive connect-phase evidence permits another
                        # attempt. A lost response may already have typed text.
                        verdict = guest_loss.classify_transport_failure(
                            error, origin=self._guest_origin()
                        )
                        if self._supervisor_unit is not None:
                            recovered = self._recover_guest_supervisor(verdict.delivery)
                            supervisor_exhausted = recovered.recovery_exhausted
                            if supervisor_exhausted:
                                ready = False
                            elif recovered.state != "verified_none":
                                raise GuestServerRecovered(
                                    "action-server restart or receipt unavailable; refresh binding"
                                ) from None
                        if ready is None and verdict.delivery == guest_loss.DELIVERY_NOT_SENT:
                            # Evidence for the terminal message, NOT a veto. This
                            # probe rides the same host->guest path as the request
                            # that just failed, so a path failure fails it too and
                            # "not ready" cannot distinguish a dead server from a
                            # transient route. Gating the retry on it -- which is
                            # what this code used to do -- turned exactly that
                            # blip into a dead episode, and the recorded message
                            # (connection could not be established, delivery=not_sent)
                            # is the EHOSTUNREACH shape no other cause produces.
                            ready = self._probe_control_server(_EXECUTE_READINESS_TIMEOUT_S)
                        # None means the classifier refused a retry (the request
                        # may have been applied) or the bound is spent. That is
                        # the ONLY gate; it takes a verdict, never a flag.
                        delay = (
                            None
                            if supervisor_exhausted
                            else transport_retry_delay(verdict, failed_attempt=attempt)
                        )
                        if delay is not None:
                            self._record_transport_retry(
                                index=index,
                                attempt=attempt,
                                verdict=verdict,
                                delay_s=delay,
                                ready=ready,
                                error=error,
                            )
                            self._sleep(delay)
                            continue
                        # Capture only on terminal failure, before termination;
                        # a recovered refusal is not a lost-guest incident.
                        self._capture_guest_console(
                            guest_loss.STAGE_EXECUTE, f"execute-{verdict.delivery}"
                        )
                        message = guest_loss.describe_failed_action(index, verdict)
                        if attempt > 1:
                            # Say how much was spent trying: the terminal line is
                            # otherwise indistinguishable from one that never
                            # retried, and the difference is a diagnosis.
                            message += f"; transport retries exhausted ({attempt})"
                        if ready is False:
                            cause = self._execute_guest_health_cause()
                            if supervisor_exhausted:
                                cause += "; supervisor recovery budget exhausted"
                            message += f"; guest down: {cause}"
                            if attempt == 1:
                                message += "; no transport retry"
                        raise GuestExecutionError(message, delivery=verdict.delivery) from None
                else:
                    # The last failed attempt raises above; falling through
                    # would mean the configured budget allowed no initial send.
                    raise AssertionError("guest execution requires at least one attempt")
                if result.returncode != 0:
                    raise GuestExecutionError(
                        f"guest action {index + 1} failed "
                        f"({_exit_description(result.returncode)}); "
                        "input may be partial; batch not retried"
                    )
            # Let the desktop settle after the batch, exactly as OSWorld's
            # own ``step(pause=...)`` does. An empty batch still settles so a
            # pure WAIT advances the guest clock.
            #
            # ``settle=False`` is for a NON-FINAL run of an interleaved batch.
            # Such a batch is delivered as several ordered runs, and settling
            # after every one would both multiply the pause and stack it on
            # top of the wait the model explicitly asked for -- turning a
            # requested 2 s into 5 s. One settle per batch, on the last run.
            if settle:
                self._sleep(self._action_delay_s)

        await asyncio.to_thread(run)

    async def probe_control_server(self, timeout: float) -> bool:
        """One bounded GET to the guest's readiness endpoint; never raises.

        The same endpoint, and the same meaning, as the allocation readiness
        check (``_READY_PATH``, upstream's own ``ensure_ready`` probe), asked
        again in the one state where the answer changes what the harness does:
        after an ambiguous delivery (see ``guest_loss``), where a control
        server that answers is the difference between continuing the episode
        with an UNKNOWN-outcome warning and ending it exactly as before. Read-
        only by construction: a GET, no statement, no observation build.

        Bounded on both ends: ``requests`` gets ``timeout`` (applied to the
        connect and read phases separately, so this call's own worst case
        is roughly twice it), and the RPC layer's outer deadline cancels
        whatever remains -- see ``_PROBE_TIMEOUT_S`` in ``adapter.py`` for
        the chosen arithmetic -- while the call runs on a worker thread so
        a hung connect cannot block the worker's event loop (and with it
        the cancel path).

        Deliberately boolean: refused, timed out, a DNS failure and a non-200
        status all collapse into False because the caller's decision is fail
        closed and nothing acts on the taxonomy. The URL is never surfaced --
        it names the guest's address.
        """

        return await asyncio.to_thread(self._probe_control_server, timeout)

    def _probe_control_server(self, timeout: float) -> bool:
        """Shared read-only probe for the worker thread and async RPC path."""

        answered, record = self._probe_control_server_outcome(timeout)
        self._last_probe_record = record
        return answered

    def _probe_control_server_outcome(self, timeout: float) -> tuple[bool, dict[str, Any]]:
        """The probe answer plus a value-free account of why it was that answer.

        EVIDENCE ONLY. The boolean is the whole decision surface -- a recovery
        that gives up after six silent probes records ``read_failed``, and that
        must stay exactly as it was -- but six false answers used to be
        indistinguishable from one another in the evidence. ``kind`` names which
        element of the path failed and ``errno`` the OS's reason; neither can
        carry the guest's address or the command, which is why the exception's
        text is still discarded. A non-200 status is reported as itself: the
        server answered, so there is no transport fault to classify.
        """

        origin = self._guest_origin()
        if origin is None:
            return False, {"answered": False, "cause": "no_origin"}
        try:
            status = int(self._clients.http_get(f"{origin}{_READY_PATH}", timeout))
        except Exception as error:
            # Discard exception text: it may contain guest addresses or state.
            return False, {"answered": False, **guest_loss.failure_record(error, origin=origin)}
        return status == 200, {"answered": status == 200, "status": status}

    async def evaluate(self) -> Any:
        env = self._require_env()

        def run() -> Any:
            capture = _JudgeErrorCapture()
            loggers = [logging.getLogger(name) for name in _JUDGE_LOGGERS]
            for logger in loggers:
                logger.addHandler(capture)
            # A second handler, on the getter's own logger: a page the getter
            # selected but could not interrogate is the same class of failure as
            # a judge that never answered -- upstream logs it, swallows it into
            # a falsy return, and the task scores 0.0. See
            # ``_UnreadGetterTargetCapture``.
            getter_logger = logging.getLogger(_GETTER_LOGGER)
            getter_capture = _UnreadGetterTargetCapture()
            getter_logger.addHandler(getter_capture)
            try:
                raw = env.evaluate()
            finally:
                for logger in loggers:
                    logger.removeHandler(capture)
                getter_logger.removeHandler(getter_capture)
            if capture.messages:
                # A recorded ERROR means upstream swallowed an exception into
                # a score. Returning that score would seal a silent zero.
                raise scoring.ScoringUnavailable(
                    "judge/evaluator backend failed: " + "; ".join(capture.messages[:3])
                )
            if getter_capture.messages:
                # Same reasoning, different instrument: the getter produced no
                # reading -- it selected a page it could not interrogate, or it
                # refused to run at all on its own missing config -- so the score
                # it fed says nothing about the agent's work and must not be
                # sealed as a miss. The captured line is upstream's own, so it
                # names which of the two shapes it was.
                raise scoring.ScoringUnavailable(
                    "page getter produced no reading: " + "; ".join(getter_capture.messages[:3])
                )
            return raw

        return await asyncio.to_thread(run)

    async def respond(self, prompt: str) -> str | None:
        env = self._env
        simulator = getattr(env, "user_simulator", None) if env is not None else None
        if simulator is None:
            return None
        answer = await asyncio.to_thread(simulator.respond, prompt)
        # None is a refusal, not the public string "None". Other objects may
        # contain private simulator state; never stringify them into replies or
        # diagnostics. Public string content is validated by the adapter wire.
        if answer is not None and not isinstance(answer, str):
            raise TypeError("simulator response must be a string or None")
        return answer

    # ------------------------------------------------------------------
    # teardown (usable with only credentials + refs; see for_teardown)
    # ------------------------------------------------------------------

    async def terminate(self, instance_ref: str) -> str:
        return await asyncio.to_thread(self._terminate, instance_ref)

    def _terminate(self, instance_ref: str) -> str:
        episode_id = _episode_id_from_ref(instance_ref)
        ec2 = self._clients.ec2
        try:
            live = _instances(ec2.describe_instances(Filters=_tag_filters(episode_id)))
            if not live:
                return EVIDENCE_INSTANCE_ABSENT
            ids = [str(instance["InstanceId"]) for instance in live]
            ec2.terminate_instances(InstanceIds=ids)
        except Exception as error:
            if _error_code(error) == "UnauthorizedOperation":
                return EVIDENCE_TERMINATE_DENIED
            raise
        # Confirm. ``shutting-down`` is NOT confirmation: it has been
        # observed to linger, and the mapping in adapter._terminate_status
        # accepts only positive evidence of ``terminated``.
        deadline = time.monotonic() + self._terminate_timeout_s
        while True:
            response = ec2.describe_instances(InstanceIds=ids)
            states = {str(i.get("State", {}).get("Name")) for i in _instances(response)}
            if states and states <= {"terminated"}:
                return EVIDENCE_INSTANCE_TERMINATED
            if time.monotonic() >= deadline:
                return EVIDENCE_TERMINATE_UNCONFIRMED
            self._sleep(2.0)

    async def delete_schedule(self, lease_ref: str) -> str:
        return await asyncio.to_thread(self._delete_schedule, lease_ref)

    def _delete_schedule(self, lease_ref: str) -> str:
        try:
            self._clients.scheduler.delete_schedule(Name=lease_ref)
        except Exception as error:
            if _error_code(error) == "ResourceNotFoundException":
                return EVIDENCE_SCHEDULE_ABSENT
            return EVIDENCE_SCHEDULE_DELETE_FAILED
        return EVIDENCE_SCHEDULE_DELETED

    async def describe(self, instance_ref: str) -> dict[str, Any] | None:
        episode_id = _episode_id_from_ref(instance_ref)

        def run() -> dict[str, Any] | None:
            found = _instances(
                self._clients.ec2.describe_instances(Filters=_tag_filters(episode_id))
            )
            return found[0] if found else None

        return await asyncio.to_thread(run)

    # ------------------------------------------------------------------
    # audit (read-only, account-wide, no descriptor)
    # ------------------------------------------------------------------

    @staticmethod
    def audit(clients: _Clients) -> list[dict[str, Any]]:
        """Every live resource carrying the adapter tag, or ``[]``.

        The proof's assertion after every episode. Deliberately NOT a
        terminator: teardown happens only through a descriptor-driven rescue
        so every termination has a receipt. Instances, volumes and schedules
        are all listed so a leak of any kind shows.
        """

        found: list[dict[str, Any]] = []
        ec2 = clients.ec2
        for instance in _instances(ec2.describe_instances(Filters=_tag_filters(None))):
            tags = {t["Key"]: t["Value"] for t in instance.get("Tags", [])}
            found.append(
                {
                    "kind": "instance",
                    "id": instance.get("InstanceId"),
                    "state": instance.get("State", {}).get("Name"),
                    "episode": tags.get(EPISODE_TAG_KEY),
                    "launched_at": (
                        instance["LaunchTime"].isoformat()
                        if isinstance(instance.get("LaunchTime"), datetime)
                        else instance.get("LaunchTime")
                    ),
                }
            )
        volumes = ec2.describe_volumes(
            Filters=[{"Name": f"tag:{ADAPTER_TAG_KEY}", "Values": [ADAPTER_TAG_VALUE]}]
        )
        for volume in volumes.get("Volumes", []):
            tags = {t["Key"]: t["Value"] for t in volume.get("Tags", [])}
            found.append(
                {
                    "kind": "volume",
                    "id": volume.get("VolumeId"),
                    "state": volume.get("State"),
                    "episode": tags.get(EPISODE_TAG_KEY),
                }
            )
        schedules = clients.scheduler.list_schedules(NamePrefix="lop-ttl-")
        for schedule in schedules.get("Schedules", []):
            found.append(
                {
                    "kind": "schedule",
                    "id": schedule.get("Name"),
                    "state": schedule.get("State"),
                    "episode": str(schedule.get("Name", ""))[len("lop-ttl-") :] or None,
                }
            )
        return found
