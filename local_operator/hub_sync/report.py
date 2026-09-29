"""Item reports and the ONE prose renderer (design A8/A9, B5).

Pure and light on purpose: ``agent_sync`` (which the CLI, the ``agent`` tool and
the desktop route all import) needs these types at module scope, and must not
drag the model stack or the registries in with them. Every surface renders
through :func:`render_report`; none re-derives wording, so the CLI, the tool and
the route cannot disagree about what a merge did.

WORDING RULES (A9.5): the words "overwritten"/"replaced" are reserved for an
explicit ``--replace``, which always echoes the replaced text. A merge is
"updated from the hub", and every removal it honours is NAMED, never only
counted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

ItemOutcome = Literal[
    "up-to-date",
    "available",
    "unchanged",
    "merged",
    "needs-review",
    "refused",
    "failed",
    "skipped",
    "unavailable",
    "would-merge",
]


@dataclass(frozen=True)
class ItemMergeReport:
    """A8's item-level report: one linked item, every field's merge result."""

    kind: Literal["agent", "team"]
    local_id: str
    name: str
    hub_id: str
    outcome: ItemOutcome
    fields: tuple[Any, ...] = ()  # MergeResult (not imported: keeps this module light)
    applied: bool = False
    backup: str | None = None
    #: B4.3 class when the item did not settle (``merge-refused``, ``model-unavailable``…).
    error_class: str | None = None
    message: str = ""
    warnings: tuple[str, ...] = ()
    #: Verbatim texts an explicit ``--replace`` discarded: the echo that keeps it recoverable.
    replaced: dict[str, Any] = field(default_factory=dict)
    classification: str | None = None
    #: ``skipped: <class>`` reason during update-all after a systemic failure.
    skipped_reason: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "local_id": self.local_id,
            "name": self.name,
            "hub_id": self.hub_id,
            "outcome": self.outcome,
            "fields": [f.to_json() if hasattr(f, "to_json") else f for f in self.fields],
            "applied": self.applied,
            "backup": self.backup,
            "error_class": self.error_class,
            "message": self.message,
            "warnings": list(self.warnings),
            "replaced": self.replaced,
            "classification": self.classification,
            "skipped_reason": self.skipped_reason,
        }


@dataclass(frozen=True)
class ApplyReport:
    reports: tuple[ItemMergeReport, ...] = ()

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.reports:
            out[r.outcome] = out.get(r.outcome, 0) + 1
        return out

    def to_json(self) -> dict[str, Any]:
        return {"reports": [r.to_json() for r in self.reports], "counts": self.counts()}


@dataclass(frozen=True)
class HubMergeEntry:
    """Adapter so a report rides ``agent_sync.SyncReport`` beside seed verdicts."""

    report: ItemMergeReport

    @property
    def name(self) -> str:
        return self.report.name

    @property
    def verdict(self) -> str:
        return self.report.outcome

    @property
    def applied(self) -> bool:
        return self.report.applied


# -- rendering ---------------------------------------------------------------------------


def _indent(text: str, pad: str = "    ") -> str:
    return "\n".join(pad + row for row in (text or "").splitlines())


def _plural(n: int, one: str, many: str | None = None) -> str:
    return f"{n} {one if n == 1 else (many or one + 's')}"


def _label(region: Any) -> str:
    """A region's name for prose: its heading, else the first 60 chars of its text."""

    heading = (getattr(region, "heading", "") or "").strip()
    if heading:
        return heading
    for text in (region.base, region.local, region.remote):
        if isinstance(text, str) and text.strip():
            first = " ".join(text.split())
            return first[:60] + ("…" if len(first) > 60 else "")
    return getattr(region, "name", "") or "(text)"


def _roll_up(report: ItemMergeReport) -> str:
    counts = {"taken-remote": 0, "kept-local": 0, "combined": 0, "removal-honored": 0}
    removals: list[str] = []
    for f in report.fields:
        for region in f.regions:
            if region.provenance in counts:
                counts[region.provenance] += 1
            if region.provenance == "removal-honored":
                who = (
                    "you removed"
                    if region.removed_by == "local"
                    else ("the hub removed" if region.removed_by == "remote" else "both removed")
                )
                extra = {
                    "local": "the hub still had it",
                    "remote": "it stays gone",
                    "both": "it stays gone",
                }.get(region.removed_by or "", "")
                removals.append(f'{who} "{_label(region)}"; {extra}'.rstrip("; "))
    bits: list[str] = []
    if counts["taken-remote"]:
        bits.append(f"{_plural(counts['taken-remote'], 'section')} taken from the hub")
    if counts["kept-local"]:
        bits.append(f"{counts['kept-local']} kept yours")
    if counts["combined"]:
        bits.append(f"{counts['combined']} combined")
    if counts["removal-honored"]:
        bits.append(f"{_plural(counts['removal-honored'], 'removal')} honored")
    line = ", ".join(bits) if bits else "no section changed"
    if removals:
        line += " (" + "; ".join(removals) + ")"
    return line


def _unresolved_lines(report: ItemMergeReport, style: str) -> list[str]:
    lines: list[str] = []
    for f in report.fields:
        for region in f.regions:
            if region.provenance != "unresolved":
                continue
            kind = (
                "removed it while the other side edited"
                if region.note == "removal-vs-edit"
                else "both changed"
            )
            lines.append(f'  {f.field}: "{_label(region)}" — {kind}')
            for tag, text in (("yours", region.local), ("hub", region.remote)):
                shown = text if text is not None else "(removed)"
                if not isinstance(shown, str):
                    shown = str(shown)
                lines.append(f"    {tag}:")
                lines.append(_indent(shown, "      "))
    return lines


def render_entry(report: ItemMergeReport, *, style: Literal["cli", "tool"] = "cli") -> str:
    """One item as ``<name>: <sentence>`` plus indented evidence (A9)."""

    name = report.name
    o = report.outcome
    if o == "up-to-date" or o == "unchanged":
        return f"{name}: hub up-to-date"
    if o == "available":
        cls = f" ({report.classification})" if report.classification else ""
        return f"{name}: hub update available{cls}" + (
            f" — {report.message}" if report.message else ""
        )
    if o == "unavailable":
        return f"{name}: hub unavailable — {report.message}"
    if o == "skipped":
        return f"{name}: skipped — {report.skipped_reason or report.message}"
    if o == "failed":
        return f"{name}: hub update failed ({report.error_class or 'error'}) — {report.message}"
    if o == "refused":
        return f"{name}: hub update refused — {report.message}"
    if report.replaced:
        head = f"{name}: replaced with the hub copy (explicit replace)"
        if report.backup:
            head += f"; backup {report.backup}"
        tail = [f"  your {k} was:\n{_indent(str(v))}" for k, v in report.replaced.items() if v]
        return "\n".join([head, *tail])
    if o == "needs-review":
        what = ", ".join(
            sorted(
                {
                    f'"{_label(r)}"'
                    for f in report.fields
                    for r in f.regions
                    if r.provenance == "unresolved"
                }
            )
        )
        head = (
            f"{name}: the hub and your copy both changed {what or 'this item'} — not applied."
            if not report.message
            else f"{name}: {report.message}"
        )
        hint = "Review, then apply with --prefer local|remote (or edit and retry)."
        if style == "tool":
            hint = "Ask the user, then re-run with resolve='local'|'remote' (or edit and retry)."
        return "\n".join([head, *_unresolved_lines(report, style), f"  {hint}"])
    if o == "would-merge":
        return f"{name}: would update from the hub — {_roll_up(report)} (dry run; nothing written)"
    # merged
    line = f"{name}: updated from the hub — {_roll_up(report)}"
    if report.backup:
        line += f"\n  your previous text is saved at {report.backup}"
    for w in report.warnings:
        line += f"\n  note: {w}"
    return line


def render_report(
    reports: "ApplyReport | Sequence[ItemMergeReport]", *, style: Literal["cli", "tool"] = "cli"
) -> str:
    items = reports.reports if isinstance(reports, ApplyReport) else tuple(reports)
    return "\n".join(render_entry(r, style=style) for r in items)
