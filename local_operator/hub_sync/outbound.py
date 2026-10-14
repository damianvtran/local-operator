"""Paired outbound projection: the two-stage merge before any paid publication.

A push must not overwrite remote-only content the hub already accepted, and must
not lose the operator's local edits made since the last accepted publication. One
merge cannot do both, so the addendum freezes TWO stages over the SAME ancestry
core:

* ``C = merge(A, L, P)`` — apply edits made SINCE the captured input A, while
  preserving the remote-only content already present in the accepted document P.
* ``M = merge(P, C, R_latest)`` — reconcile any concurrent remote change on top.

``B`` (the inbound integrated baseline) is deliberately NOT substituted for A or
P: the outbound pair is the same origin's own projection, not the inbound
baseline. Generalization can rewrite what the server accepted, which is exactly
why A is the captured local input and P is the ORIGINAL accepted document rather
than a fresh fetch.

If M is canonically equal to the current remote R, there is nothing to publish:
the caller settles a no-op and makes no paid call. Equality is checked on the
canonical field projection, not raw bytes, so whitespace-only rewrites the hub
would normalize do not trigger a billed publication.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from local_operator.hub_sync.identity import Kind
from local_operator.hub_sync.merge import (
    FieldInput,
    MergeOptions,
    MergeResult,
    merge_field,
)
from local_operator.hub_sync.provenance import fingerprint_snapshot
from local_operator.hub_sync.segment import norm

#: Field shape per kind for the projection. Kept beside the adapters so a new
#: kind declares its transferable fields in ONE place (``adapters.canonical_fields``
#: supplies the values; this only names which fields take part).
_TEXT_FIELDS = {
    "agent": (("instructions", "markdown"), ("description", "text")),
    "team": (
        ("description", "text"),
        ("manager", "scalar"),
        ("members", "roster"),
        ("instructions", "markdown"),
        ("project", "markdown"),
    ),
}


@dataclass(frozen=True)
class ProjectionResult:
    """The projected outbound document plus whether a paid publish is needed."""

    fields: Mapping[str, Any]
    no_op: bool
    results: tuple[MergeResult, ...]
    warnings: tuple[str, ...]
    unresolved: tuple[str, ...]


def _empty(kind: str) -> Any:
    return [] if kind == "roster" else ""


def _one_field(
    name: str,
    merge_kind: str,
    base: Mapping[str, Any] | None,
    local: Mapping[str, Any],
    remote: Mapping[str, Any],
    *,
    opts: MergeOptions,
) -> MergeResult:
    empty = _empty(merge_kind)
    return merge_field(
        FieldInput(
            field=name,
            kind=merge_kind,  # type: ignore[arg-type]
            base=(base or {}).get(name, empty) if base is not None else None,
            local=local.get(name, empty),
            remote=remote.get(name, empty),
        ),
        opts,
    )


def project_outbound(
    kind: Kind,
    *,
    captured_local: Mapping[str, Any],
    accepted: Mapping[str, Any],
    current_local: Mapping[str, Any],
    remote_latest: Mapping[str, Any],
    opts: MergeOptions = MergeOptions(),
) -> ProjectionResult:
    """Two-stage projection; returns the document to publish and a no-op flag.

    ``captured_local`` is A, ``accepted`` is P, ``current_local`` is L, and
    ``remote_latest`` is the freshly fetched R. Raises ``KeyError`` for a kind
    with no declared projection rather than guessing a field shape.
    """
    specs = _TEXT_FIELDS.get(kind)
    if specs is None:
        raise KeyError(kind)

    # Stage 1: C = merge(A, L, P) -- local edits since A, keeping remote-only P.
    stage_one = [
        _one_field(name, mk, captured_local, current_local, accepted, opts=opts)
        for name, mk in specs
    ]
    c_fields = {r.field: r.merged for r in stage_one}

    # Stage 2: M = merge(P, C, R_latest) -- reconcile concurrent remote changes.
    stage_two = [
        _one_field(name, mk, accepted, c_fields, remote_latest, opts=opts) for name, mk in specs
    ]
    m_fields = {r.field: r.merged for r in stage_two}

    results = tuple(stage_one + stage_two)
    warnings = tuple(dict.fromkeys(w for r in results for w in r.warnings))
    unresolved = tuple(r.field for r in results if r.outcome == "needs-review")

    # No-op when the projected document is canonically the remote we already have.
    # Both sides go through the ONE canonical projection so a whitespace-only
    # rewrite the hub would normalize does not trigger a billed publication.
    no_op = fingerprint_snapshot(kind, m_fields) == fingerprint_snapshot(
        kind, canonical_fields(kind, remote_latest)
    )
    return ProjectionResult(
        fields=m_fields,
        no_op=no_op,
        results=results,
        warnings=warnings,
        unresolved=unresolved,
    )


def canonical_fields(kind: Kind, fields: Mapping[str, Any]) -> Mapping[str, Any]:
    """Normalize a document to the engine's canonical form, per field kind.

    The merge core normalizes text (LF, trimmed, single blank line) and rosters; a
    raw remote document is not yet in that form. Applying the SAME normalization
    here is what makes the no-op comparison meaningful rather than byte-exact.
    """
    specs = _TEXT_FIELDS.get(kind)
    if specs is None:
        raise KeyError(kind)
    from local_operator.hub_sync.provenance import normalize_roster

    out: dict[str, Any] = {}
    for name, merge_kind in specs:
        value = fields.get(name, _empty(merge_kind))
        if merge_kind == "roster":
            out[name] = normalize_roster(value)
        elif merge_kind == "scalar":
            out[name] = str(value or "").strip()
        else:
            out[name] = norm(str(value or ""))
    return out


def derived_no_op(kind: Kind, projected: Mapping[str, Any], remote: Mapping[str, Any]) -> bool:
    """Canonical equality for the settle-no-op decision, reusable by callers."""

    return fingerprint_snapshot(kind, projected) == fingerprint_snapshot(kind, remote)


def refuses(projection: ProjectionResult) -> str | None:
    """The named refusal class if the projection cannot be published, else None."""

    for r in projection.results:
        if r.outcome == "refused":
            return "merge-refused"
        if r.outcome == "needs-review":
            return r.engine.failure_class or "merge-refused"
    return None
