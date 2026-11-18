"""The remote-compute library: what a surface calls, with no argparse and no output.

WHY THIS IS A SEPARATE MODULE FROM THE CLIENT. The client is a transport with
typed errors; the CLI is a command line; the desktop GUI's server endpoints are
neither. What they share is a SEQUENCE — submit, then remember locally; read,
then refresh the local record; cancel; delete; list — and that sequence is what
this module owns. It takes an injected
:class:`~local_operator.remote_compute.client.RemoteComputeClient`,
because the credential decision belongs to whoever owns the process (the CLI
pins one per invocation; a server pins one per request), and it PRINTS NOTHING:
every method answers a value or raises a typed error from
:mod:`local_operator.remote_compute.errors`, so each surface keeps its own copy,
locale and exit code.

WHY THE LOCAL RECORD IS WRITTEN HERE RATHER THAN BY EACH SURFACE. "What did I
send, and where is it now?" has to be answerable while Radient is unreachable,
and two surfaces writing that answer their own way is how a panel and a CLI come
to disagree about which runs exist. One write path, one file: see
:mod:`local_operator.remote_compute.records`.

THE RECORD IS BEST-EFFORT AND NEVER FAILS A CALL. A run that was submitted is
submitted even if this disk is full; losing the row is a cosmetic loss, while
raising out of a successful create would tell the operator their work did not
start. So the record write is guarded, and the write's own failure is swallowed
with the reason stated here rather than in a bare ``except``.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

from local_operator.remote_compute import records, results
from local_operator.remote_compute.client import (
    LOGS_PAGE_DEFAULT,
    CreatedRun,
    LogPage,
    RemoteComputeClient,
    run_id,
    run_network,
    run_status,
)
from local_operator.remote_compute.defaults import RunDefaults, run_defaults
from local_operator.remote_compute.errors import RemoteComputeError
from local_operator.remote_compute.repo import RepoRef, detect_repo
from local_operator.remote_compute.request import RunCreateRequest, RunSpec

#: The follower's idle wait, floor and ceiling.
#:
#: AN EMPTY PAGE IS THE NORMAL CASE for a live run: the reader is caught up and
#: the run has not printed anything yet. So the wait doubles while nothing is
#: new and resets the moment a line arrives. The ceiling is the number that
#: matters — a quiet run costs one request every five seconds, which any session
#: log can be polled at, and a run that starts printing is seen within five
#: seconds. These are PACING CHOICES, not measurements: no distribution of
#: inter-line gaps for this endpoint exists in this repository, and the "live"
#: target is the order a person notices text appear, not a latency budget.
LOGS_IDLE_MIN_SECONDS = 1.0
LOGS_IDLE_MAX_SECONDS = 5.0

#: How many consecutive empty pages with an unchanged token mean "caught up".
#:
#: TWO, because that is the contract's own definition of it: one empty page can be
#: a page the server had just filled, and an unchanged token is what distinguishes
#: "nothing new" from "this reader is behind".
LOGS_EMPTY_PAGES_BEFORE_CAUGHT_UP = 2

#: How many consecutive transient failures the follower retries, and its wait cap.
#:
#: THE INVARIANT IS "A STREAM THAT CANNOT CONTINUE REPORTS IT": the follower tails
#: a run, so a refusal it can never satisfy has to arrive at the surface as an
#: error instead of an endless retry, because the failure a GUI shows then is "the
#: log stopped advancing" — a state it can paint — rather than a panel that is
#: silently dead.
#:
#: THE BOUND, and its honesty: with the schedule in :func:`_retry_delay`, five
#: attempts wait at most 1 + 2 + 4 + 8 + 8 = 23 s before the refusal is raised.
#: Both numbers are CHOSEN, not measured: this repository holds no distribution of
#: transient 503s on this path, so they are the interval at which someone watching
#: a log panel concludes it is stuck (a product judgement), and 8 s is also the
#: point past which honouring a server's ``Retry-After`` means a live tail has
#: stopped being live.
LOGS_RETRY_ATTEMPTS = 5
LOGS_RETRY_CAP_SECONDS = 8.0

#: The sleep the follower waits with. A seam so a test does not wait the schedule
#: it is asserting — never a tuning knob for production.
_Sleeper = Callable[[float], Awaitable[None]]


@dataclasses.dataclass(frozen=True)
class SubmittedRun:
    """A run the server accepted, plus what was recorded about it locally."""

    created: CreatedRun
    record: records.RunRecord | None

    @property
    def run(self) -> dict[str, Any]:
        """The run as the server returned it — what a receipt may quote."""
        return self.created.run

    @property
    def replayed(self) -> bool:
        """Whether the server answered an earlier submission with this run."""
        return self.created.replayed


class RemoteRuns:
    """The shared run API: client calls plus the local record, over one client."""

    def __init__(self, client: RemoteComputeClient, *, config_dir: Path | None = None) -> None:
        self.client = client
        #: Where the local record lives; ``None`` resolves the app's config dir
        #: per call, exactly as :func:`records.records_path` does, so a test that
        #: redirects the config directory is honoured without rebuilding this.
        self.config_dir = config_dir

    # -- writes -------------------------------------------------------------

    async def create_run(
        self,
        request: RunCreateRequest,
        *,
        idempotency_key: str,
        recorded_from: "RunSubmitMetadata | None" = None,
    ) -> SubmittedRun:
        """Submit one run and record it locally.

        THE RECORD's repo, sha, size and summary come from ``recorded_from`` when
        the caller knows what it submitted (a GUI that detected the repository,
        a CLI that was handed a spec) and from the request body otherwise — the
        body carries ``repo_url``, ``repo_sha`` and ``size`` already, and a
        prompt only as free text that must be summarised.
        """
        created = await self.client.create_run(request, idempotency_key=idempotency_key)
        record = self._record_submission(created, request=request, metadata=recorded_from)
        return SubmittedRun(created=created, record=record)

    def _record_submission(
        self,
        created: CreatedRun,
        *,
        request: RunCreateRequest,
        metadata: "RunSubmitMetadata | None",
    ) -> records.RunRecord | None:
        body = request.to_json()
        try:
            return records.record_submission(
                run_id=run_id(created.run),
                repo_url=_text(metadata.repo_url if metadata else None)
                or _text(body.get("repo_url")),
                repo_sha=_text(metadata.repo_sha if metadata else None)
                or _text(body.get("repo_sha")),
                # The FULL prompt goes in and ``records`` summarises it, so the
                # scrub-and-cap rule has one implementation rather than two.
                prompt=_text(metadata.prompt if metadata else None) or _text(body.get("prompt")),
                size=_text(metadata.size if metadata else None) or _text(body.get("size")),
                api_origin=self.client.api_url,
                network_id=_optional(created.run.get("network_id")),
                status=_optional(created.run.get("status")),
                config_dir=self.config_dir,
            )
        except OSError:
            # See the module docstring: the run is submitted either way, so a
            # record that cannot be written is reported by its absence, not by
            # failing a create that already happened at the far end.
            return None

    # -- reads --------------------------------------------------------------

    async def get_run(
        self, identifier: str, *, expected_network: str | None = None
    ) -> dict[str, Any]:
        """One run by id, refreshing the local record's status if it is recorded."""
        run = await self.client.get_run(identifier, expected_network=expected_network)
        self._record_status(run)
        return run

    async def list_runs(
        self,
        *,
        status: str | None = None,
        network_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """One page of the account's runs, optionally narrowed by status."""
        return await self.client.list_runs(
            status=status, network_id=network_id, limit=limit, offset=offset
        )

    def local_runs(self) -> list[records.RunRecord]:
        """The recorded runs, newest first — the panel's offline answer.

        Synchronous on purpose: it reads one small local file and touches no
        network, so a surface renders it before it awaits anything.
        """
        return records.load(self.config_dir)

    async def follow_logs(
        self,
        identifier: str,
        *,
        limit: int = LOGS_PAGE_DEFAULT,
        sleep: _Sleeper | None = None,
    ) -> AsyncIterator[LogPage]:
        """Tail a run's logs page by page, until the run is finished.

        ONE STOP RULE, THE CONTRACT'S: an EMPTY page ends the stream only when the
        run is TERMINAL. A terminal run whose page still carries lines has more to
        read, and an empty page on a running run is the ordinary "nothing yet" —
        so neither of those ends it.

        AN UNCHANGED TOKEN ON TWO CONSECUTIVE EMPTY PAGES MEANS "NO NEW LINES",
        and it paces the stream rather than ending it: only once that has happened
        does the wait grow, because until then the page may simply be the one the
        server had just filled. A token that CHANGES while the page is empty is
        the reader being behind, so the next page is asked for at once.

        THE TWO REFUSALS THAT STOP RATHER THAN RETRY. A coded ``not_configured``
        503 is the deployment saying logs are not readable here, and a 400 is a
        bad or foreign token; no retry repairs either, so both are raised as they
        arrive. Only an UNCODED 503 is treated as transient — a hop that could not
        answer rather than the far end refusing — and it is retried inside
        :data:`LOGS_RETRY_ATTEMPTS` with :func:`_retry_delay` before being raised.
        The code, not the status, is what separates those cases: a coded 503 is a
        considered answer and is never retried.
        """
        waiter = sleep if sleep is not None else asyncio.sleep
        token: str | None = None
        unchanged_empty = 0
        failures = 0
        idle = LOGS_IDLE_MIN_SECONDS
        while True:
            sent = token
            try:
                page = await self.client.run_logs(identifier, next_token=token, limit=limit)
            except RemoteComputeError as failure:
                if not _transient(failure) or failures + 1 >= LOGS_RETRY_ATTEMPTS:
                    raise
                failures += 1
                await waiter(_retry_delay(failure, failures))
                continue
            failures = 0
            token = page.next_token
            unchanged_empty = unchanged_empty + 1 if page.empty and token == sent else 0
            yield page
            if page.terminal and page.empty:
                return
            if not page.empty:
                # Lines arrived, so there may be more behind them: ask again at
                # once and start the idle schedule over.
                idle = LOGS_IDLE_MIN_SECONDS
                continue
            caught_up = unchanged_empty >= LOGS_EMPTY_PAGES_BEFORE_CAUGHT_UP
            await waiter(idle if caught_up else LOGS_IDLE_MIN_SECONDS)
            if caught_up:
                idle = min(idle * 2.0, LOGS_IDLE_MAX_SECONDS)

    # -- results -----------------------------------------------------------

    async def fetch_results(
        self, identifier: str, destination: Path | str
    ) -> results.FetchedResults:
        """Download and extract this run's results into a NEW ``destination``.

        The mechanics — the presigned download, the size check, the refusals —
        belong to :mod:`local_operator.remote_compute.results`; this method is here
        because a surface holds one object for a run's whole life.
        """
        return await results.fetch_results(self.client, identifier, Path(destination))

    async def apply_bundle(self, identifier: str, repo_dir: Path | str) -> results.AppliedBundle:
        """Fetch this run's bundle into ``repo_dir`` as ``refs/remotes/radient/lop/<id>``."""
        return await results.apply_bundle(self.client, identifier, repo_dir)

    def restore_session(self, extracted: Path | str) -> results.RestoredSession | None:
        """Offer this run's session for ``lop --resume`` from an extracted tree.

        ``self.config_dir`` is the root the caller was handed, so a daemon serving a
        relocated root restores into THAT root rather than the process default.
        """
        return results.restore_session(Path(extracted), config_dir=self.config_dir)

    # -- cancel / delete ----------------------------------------------------

    async def cancel_run(self, identifier: str) -> dict[str, Any]:
        """Cancel a run and refresh its recorded status from the answer."""
        run = await self.client.cancel_run(identifier)
        if run:
            self._record_status(run)
        return run

    async def delete_run(self, identifier: str) -> dict[str, Any]:
        """Delete a run, marking it deleted in the local record.

        The row is KEPT and marked (see :func:`records.mark_deleted`): a panel
        that dropped it would look like this device forgot work it submitted.
        """
        run = await self.client.delete_run(identifier)
        try:
            records.mark_deleted(identifier, config_dir=self.config_dir)
        except OSError:
            # Same reasoning as the submit path: the server's delete succeeded,
            # and a record that cannot be updated does not undo it.
            pass
        return run

    def _record_status(self, run: dict[str, Any]) -> None:
        try:
            records.record_status(run, api_origin=self.client.api_url, config_dir=self.config_dir)
        except OSError:
            return


@dataclasses.dataclass(frozen=True)
class RunSubmitMetadata:
    """What the caller knows about a submission that the body may not carry.

    A ``RunCreateRequest`` body carries the wire fields — including the prompt,
    which must never be stored verbatim — so a caller that already has the parts
    (a GUI that detected a repository, a CLI holding its own spec) passes them
    here and the record does not have to re-derive them from the body it is
    about to send.
    """

    repo_url: str = ""
    repo_sha: str = ""
    prompt: str = ""
    size: str = ""


def build_spec(
    *,
    prompt: str,
    repo: RepoRef | None = None,
    repo_url: str | None = None,
    repo_sha: str | None = None,
    hosting: str,
    model: str,
    defaults: RunDefaults | None = None,
    size: str | None = None,
    max_hours: float | None = None,
    budget_usd: float | None = None,
    name: str | None = None,
) -> RunSpec:
    """A run spec from either detected repository facts or explicit ones.

    ONE CONSTRUCTOR, so the defaults cannot be applied twice or differently: the
    GUI detects a repository and passes the :class:`RepoRef`, the CLI is handed
    ``--repo``/``--sha``, and both arrive at the same :class:`RunSpec` with the
    same default size and hours.
    """
    chosen = defaults if defaults is not None else run_defaults()
    resolved_url = repo_url if repo_url is not None else (repo.url if repo else "")
    resolved_sha = repo_sha if repo_sha is not None else (repo.sha if repo else "")
    return RunSpec(
        prompt=prompt,
        repo_url=resolved_url or "",
        repo_sha=resolved_sha or "",
        hosting=hosting,
        model=model,
        size=size if size is not None else chosen.size,
        max_hours=max_hours if max_hours is not None else chosen.max_hours,
        budget_usd=budget_usd,
        name=name,
    )


def repository_for(directory: Path | str, *, refresh: bool = False) -> RepoRef:
    """The repository and HEAD commit of ``directory``; see :func:`repo.detect_repo`."""
    return detect_repo(directory, refresh=refresh)


def _transient(failure: RemoteComputeError) -> bool:
    """Whether a log read failed in a way a retry can answer.

    AN UNCODED 503, and nothing else. The status alone is not enough: the contract
    answers ``not_configured`` with a 503 too, and that is the deployment saying
    logs are not readable here — retrying it would turn a definite answer into a
    minute of waiting. A 429 is deliberately not here either: it is the account's
    rate limit, and a tail that waited it out would spend the quota the limit
    exists to protect, so the surface gets the refusal and decides.
    """
    return failure.code is None and failure.status == 503


def _retry_delay(failure: RemoteComputeError, attempt: int) -> float:
    """The wait before retry number ``attempt`` (1-based), capped.

    THE SERVER'S OWN INSTRUCTION WINS WHEN THERE IS ONE, up to
    :data:`LOGS_RETRY_CAP_SECONDS`: an hour-long ``Retry-After`` is the server
    saying stop, and a live tail that honoured it would hold a panel open on
    nothing. Otherwise the wait doubles from the idle floor, which is the same
    first step the idle schedule takes — one schedule to reason about, not two.
    """
    if failure.retry_after is not None:
        return min(failure.retry_after, LOGS_RETRY_CAP_SECONDS)
    return min(LOGS_IDLE_MIN_SECONDS * (2.0 ** (attempt - 1)), LOGS_RETRY_CAP_SECONDS)


def _text(value: object) -> str:
    """``value`` when it is a non-empty string, else ``""``."""
    return value if isinstance(value, str) and value else ""


def _optional(value: object) -> str | None:
    """``value`` when it is a non-empty string, else ``None``."""
    return value if isinstance(value, str) and value else None


__all__ = [
    "LOGS_EMPTY_PAGES_BEFORE_CAUGHT_UP",
    "LOGS_IDLE_MAX_SECONDS",
    "LOGS_IDLE_MIN_SECONDS",
    "LOGS_RETRY_ATTEMPTS",
    "LOGS_RETRY_CAP_SECONDS",
    "RemoteRuns",
    "RunSubmitMetadata",
    "SubmittedRun",
    "build_spec",
    "repository_for",
    "run_defaults",
    "run_network",
    "run_status",
]
