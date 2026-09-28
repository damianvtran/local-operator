"""One sync run across the two updateable agent families: seeds and hub pulls.

WHY A COORDINATOR, AND WHY HERE
===============================

"Built-in agents and hub-pulled agents have an update method to pull the
latest" is one user-facing verb, but the two families answer it with two very
different mechanisms — a packaged markdown seed compared inside the registry,
and a network re-fetch of a marketplace listing — and each mechanism lives in
its own module (:mod:`local_operator.agent_profiles` and
:mod:`local_operator.agents`). This module is the ONE place that runs both,
so the ``agent`` tool, the ``sync`` CLI command and the desktop route cannot
drift into three slightly different definitions of "sync" (the rule the
desktop side already spells as "no second derivation per surface").

It deliberately owns no classification of its own: every verdict comes out of
:func:`local_operator.agent_profiles.sync_installed_seeds` or
:func:`local_operator.agents.sync_hub_agents`, and this module only orders
them, adds the requested-names-coverage note, and renders. A third
classification comparator is exactly the kind of thing that would later
disagree with the other two.

The clients that reach the hub are resolved here too, because all three
surfaces need the same two questions answered: which API root (config.yml's
``values.radient_base_url`` when set, else ``resolve_radient_api_base_url``),
and is there a credential at all (the store, then ``RADIENT_API_KEY``). With
none, sync does not fail — the hub rows report ``unavailable`` with the reason
and the seed arm still runs (design §9.2).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence, Union

from local_operator.agent_profiles import SeedSyncVerdict, sync_installed_seeds
from local_operator.agents import HubSyncVerdict, sync_hub_agents

#: One entry of a sync run. Two dataclasses and not one: the seed and hub arms
#: carry different evidence (versions vs a marketplace id), and flattening
#: them would make every field optional for half the rows.
SyncEntry = Union[SeedSyncVerdict, HubSyncVerdict]


@dataclass(frozen=True)
class SyncReport:
    """Every verdict of one run, ordered: seeds first (by name), then hub rows.

    One entry per requested name, even when the answer is "nothing to
    update" — an explicit ``--name X`` that silently produced nothing would
    read as success, so the arms answer ``not-installed`` for names neither
    owns (a mistyped name, a starter that was never installed, a role the
    operator authored) and a seed placeholder is dropped only where a hub
    verdict for the same name supersedes it (a hub row would otherwise get a
    true-but-useless "not installed from a packaged starter" beside its real
    verdict).
    """

    entries: tuple[SyncEntry, ...] = ()

    def counts(self) -> dict[str, int]:
        """Entry counts by outcome, for summaries."""
        counts = {
            "up-to-date": 0,
            "updated": 0,
            "diverged": 0,
            "unavailable": 0,
            "not-installed": 0,
        }
        for entry in self.entries:
            if entry.applied or entry.verdict in ("outdated-clean", "updated"):
                counts["updated"] += 1
            elif entry.verdict == "up-to-date":
                counts["up-to-date"] += 1
            elif entry.verdict == "unavailable":
                counts["unavailable"] += 1
            elif entry.verdict in ("outdated-diverged", "diverged"):
                counts["diverged"] += 1
            else:
                counts["not-installed"] += 1
        return counts

    def render(self) -> str:
        """The human-readable form shared by the ``agent`` tool and the CLI.

        Lines are ``<name>: <sentence>``, with the echo of any replaced text
        UNDER its line and indented, so an applied update stays recoverable by
        copy-paste — the guarantee ``op='reset'`` makes, kept here because the
        clean arm overwrites without asking. The tool runs the result through
        ``spill_truncate``; the CLI prints it in full, which is the point of an
        echo.
        """

        if not self.entries:
            return "nothing to sync: no installed seed or hub-pulled profiles."
        lines = [_render_entry(entry) for entry in self.entries]
        counts = self.counts()
        summary = ", ".join(f"{count} {label}" for label, count in counts.items() if count)
        return f"{summary}.\n\n" + "\n".join(lines)


def _render_entry(entry: SyncEntry) -> str:
    """One verdict as a line, with the replacement echo beneath it when applied.

    Built from the STRUCTURED fields rather than the arm's own ``detail``
    sentence wherever the structure exists (versions, diverged field names,
    the marketplace id), because these lines are also what a reader diffing
    two runs compares; the free-text detail is used only for the outcomes that
    have no structure to render (up-to-date notes, refusals' reasons).
    """

    if isinstance(entry, SeedSyncVerdict):
        if entry.verdict == "not-installed":
            line = f"{entry.name}: not installed — {entry.detail}"
        elif entry.verdict == "up-to-date":
            line = f"{entry.name}: up-to-date — {entry.detail}"
        elif entry.verdict == "outdated-clean":
            if entry.installed_version and entry.packaged_version:
                if entry.installed_version == entry.packaged_version:
                    # An unbumped body move is a real update (agent review
                    # round 1, M1). "1.0.0 -> 1.0.0" would read as a no-op, so
                    # the receipt says what actually moved instead.
                    transition = f" ({entry.packaged_version}, text moved)"
                else:
                    transition = f" ({entry.installed_version} -> {entry.packaged_version})"
            else:
                transition = ""
            line = f"{entry.name}: updated to the packaged starter{transition}"
        else:
            fields = ", ".join(entry.diverged_fields) or "unknown fields"
            if entry.applied:
                line = f"{entry.name}: updated (forced over local edits) — replaced {fields}"
            else:
                line = (
                    f"{entry.name}: differs from the packaged starter in {fields} — "
                    f"{entry.detail}"
                )
        if entry.applied and entry.replaced_instructions:
            line += "\n  your instructions were:\n" + _indent(entry.replaced_instructions)
        for field, value in entry.replaced_fields:
            line += f"\n  your {field}: {value}"
        return line

    if entry.verdict == "up-to-date":
        return f"{entry.name}: hub up-to-date — {entry.detail or 'matches the marketplace listing'}"
    if entry.verdict == "updated":
        line = f"{entry.name}: hub updated (listing {entry.hub_id}) — {entry.detail}"
        if entry.applied and entry.replaced_instructions:
            line += "\n  your instructions were:\n" + _indent(entry.replaced_instructions)
            if entry.replaced_description:
                line += "\n  your description was: " + entry.replaced_description
        return line
    if entry.verdict == "diverged":
        return f"{entry.name}: hub listing {entry.hub_id} changed — {entry.detail}"
    return f"{entry.name}: hub unavailable — {entry.reason or entry.detail}"


def _indent(text: str) -> str:
    """Indent an echoed text block beneath its entry line, keeping it copyable."""

    return "\n".join("    " + row for row in (text or "").splitlines())


def sync_agent_profiles(
    registry: Any,
    *,
    radient_client: Any | None = None,
    names: Sequence[str] | None = None,
    force: bool = False,
) -> SyncReport:
    """Run the seed arm and the hub arm over one registry and merge the verdicts.

    ``radient_client=None`` is not an error: it means the caller could not
    resolve a Radient credential (or chose not to), and hub rows report
    ``unavailable`` with the reason while the seed arm runs normally. Any
    per-row hub failure inside the arm degrades the same way — a reachable hub
    is optional enrichment, never a precondition for syncing packaged seeds.

    ``names`` filters BOTH families by the local row name, so ``sync --name X``
    answers for whichever kind of profile X is. Every requested name gets an
    entry: the arms answer ``not-installed`` for names neither family owns,
    and the result is merged so a hub row contributes one verdict, not a seed
    placeholder beside it.
    """

    seed_verdicts = sync_installed_seeds(registry, names=names, force=force)
    hub_verdicts = sync_hub_agents(
        registry, radient_client=radient_client, names=names, force=force
    )
    # Case-folded on both sides: the seed arm answers in the folded seed-name
    # space while the hub arm answers in the row's own spelling, so ``Hunter``
    # and ``hunter`` must collide here or the row gets two lines.
    hub_names = {entry.name.lower() for entry in hub_verdicts}
    # A name that resolved to a hub row also produced a seed-arm
    # ``not-installed`` placeholder ("not installed from a packaged starter").
    # That sentence is true of a hub row and useless next to its real verdict;
    # the hub verdict wins the slot.
    entries: list[SyncEntry] = [
        entry
        for entry in seed_verdicts
        if not (entry.verdict == "not-installed" and entry.name.lower() in hub_names)
    ]
    entries.extend(hub_verdicts)
    return SyncReport(entries=tuple(entries))


def sync_payload(report: SyncReport) -> dict[str, Any]:
    """The JSON form of a report, for the desktop route.

    Every field of every verdict is carried through (``asdict``) plus a
    ``kind`` discriminator, so the UI can render an update without a second
    call — the demo of an update that only names the profile is what sends a
    user hunting through a diff to find out what changed.
    """

    return {
        "entries": [
            {"kind": "seed" if isinstance(entry, SeedSyncVerdict) else "hub", **asdict(entry)}
            for entry in report.entries
        ],
        "summary": report.counts(),
    }


def hub_base_url(config_dir: Path) -> str:
    """The Radient Agent Hub API root for a config root, config.yml included.

    For surfaces that do not hold a ``ConfigManager`` already (the ``agent``
    tool). The RULE is not repeated here: it lives in
    ``providers.radient_credentials.configured_radient_base_url``, which the
    CLI's ``_radient_hub_base_url`` delegates to as well — one configuration
    resolving two destinations is the bug the single-place rule exists to
    prevent (agent review round 1, n2).
    """

    from local_operator.config import ConfigManager
    from local_operator.providers.radient_credentials import configured_radient_base_url

    return configured_radient_base_url(ConfigManager(config_dir))


async def resolve_hub_client(
    config_dir: Path, *, base_url: str | None = None, store: Any | None = None
) -> Any | None:
    """A Radient client when a credential resolves, else None.

    ``None`` is the caller's signal to pass ``radient_client=None`` into
    :func:`sync_agent_profiles` and let the hub rows degrade; it is never an
    error, because using local updates must not depend on a marketplace login.
    ``store`` threads the route's shared ``AuthStore`` so a refresh persists
    in the one place the login owns (the async resolver's contract).
    """

    from local_operator.clients.radient import RadientClient
    from local_operator.providers.radient_credentials import resolve_radient_credential

    url = base_url or hub_base_url(config_dir)
    api_key = await resolve_radient_credential(config_dir, url, store=store)
    if not api_key:
        return None
    return RadientClient(api_key=api_key, base_url=url)


def resolve_hub_client_sync(config_dir: Path, *, base_url: str | None = None) -> Any | None:
    """The CLI bridge: same resolution, for callers with no running event loop.

    Mirrors the async resolver above through
    :func:`~local_operator.providers.radient_credentials.resolve_radient_credential_sync`,
    which raises inside a running loop rather than pretending to work — the CLI
    is the only sync host that reaches this.
    """

    from local_operator.clients.radient import RadientClient
    from local_operator.providers.radient_credentials import (
        resolve_radient_credential_sync,
    )

    url = base_url or hub_base_url(config_dir)
    api_key = resolve_radient_credential_sync(config_dir, url)
    if not api_key:
        return None
    return RadientClient(api_key=api_key, base_url=url)
