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

import dataclasses
from pathlib import Path
from typing import Any

from local_operator.remote_compute import records
from local_operator.remote_compute.client import (
    CreatedRun,
    RemoteComputeClient,
    run_id,
    run_network,
    run_status,
)
from local_operator.remote_compute.defaults import RunDefaults, run_defaults
from local_operator.remote_compute.repo import RepoRef, detect_repo
from local_operator.remote_compute.request import RunCreateRequest, RunSpec


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


def _text(value: object) -> str:
    """``value`` when it is a non-empty string, else ``""``."""
    return value if isinstance(value, str) and value else ""


def _optional(value: object) -> str | None:
    """``value`` when it is a non-empty string, else ``None``."""
    return value if isinstance(value, str) and value else None


__all__ = [
    "RemoteRuns",
    "RunSubmitMetadata",
    "SubmittedRun",
    "build_spec",
    "repository_for",
    "run_defaults",
    "run_network",
    "run_status",
]
