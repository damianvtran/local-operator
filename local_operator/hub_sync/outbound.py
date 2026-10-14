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
#: Skills do not fit a flat spec list: their content is a SET of files merged per
#: path, so they take the dedicated branch in ``project_outbound`` and share the
#: per-file helper below with the inbound skill apply.
_TEXT_FIELDS = {
    "agent": (("instructions", "markdown"), ("description", "text")),
    "team": (
        ("description", "text"),
        ("manager", "scalar"),
        ("members", "roster"),
        ("instructions", "markdown"),
        ("project", "markdown"),
    ),
    "skill": (("description", "text"),),
}


#: Files that exist as EMPTY content on both sides stay: "absent" is ``None``,
#: an empty string is a real (if pointless) file. Kept as ONE predicate so the
#: inbound apply and the outbound projection classify removal identically.
def skill_file_present(local: Any, remote: Any, merged: Any) -> bool:
    if merged is None:
        return False
    content = str(merged)
    if content != "":
        return True
    return local == "" and remote == ""


def merge_skill_file(
    path: str,
    *,
    base: str | None,
    local: str | None,
    remote: str | None,
    opts: MergeOptions,
    base_present: bool | None = None,
) -> tuple[str | None, MergeResult]:
    """Three-way merge ONE manifest file through the shared engine (no fork).

    A file participates exactly as a ``text`` field whose ABSENCE is ``None`` --
    the engine's own add/remove table then decides adds, one-sided removals and
    the edit-vs-delete conflict deterministically. Returns ``(merged content or
    None when the file is gone, result)``; callers treat ``unresolved`` as a
    refusal rather than writing either side.
    """
    if base_present is False and local is not None and remote is not None and local != remote:
        # Both sides ADDED this path with different content: a conflict at the
        # FILE level, decided before the text merge can union them (QN4).
        from local_operator.hub_sync.merge import file_add_conflict

        return None, file_add_conflict(path, local, remote)
    result = merge_field(
        FieldInput(
            field=path,
            kind="text",
            base=base,
            local=local,
            remote=remote,
            base_present=base_present,
        ),
        opts,
    )
    if not skill_file_present(local, remote, result.merged):
        return None, result
    return str(result.merged), result


@dataclass(frozen=True)
class ProjectionResult:
    """The projected outbound document plus whether a paid publish is needed."""

    fields: Mapping[str, Any]
    no_op: bool
    results: tuple[MergeResult, ...]
    warnings: tuple[str, ...]
    unresolved: tuple[str, ...]


def merge_skill_document(
    base: Mapping[str, Any] | None,
    local: Mapping[str, Any],
    remote: Mapping[str, Any],
    *,
    opts: MergeOptions = MergeOptions(),
) -> tuple[Mapping[str, Any] | None, list[MergeResult]]:
    """Three-way merge a skill MANIFEST through the shared engine, per file.

    The ONE skill merge used by both directions: the inbound apply and the two
    outbound projection stages. The manifest's ``description`` is a ``text``
    field; every file is one more ``text`` field keyed by its path (absence =
    ``None``), so adds, one-sided removals and edit-vs-delete conflicts go
    through the engine's own deterministic table -- there is no second merge
    implementation for skills.

    Returns ``(merged manifest or None, results)``. ``None`` means the merge is
    not writable: an unresolved file, an invalid result, or a document the
    constructor refuses (which is exactly the removal-of-SKILL.md case). The
    merged document's NAME is the REMOTE spelling: the local alias is
    display-only and must never travel (renaming it is not a remote rename).
    """
    from local_operator.hub_sync.skill_document import (
        SkillDocument,
        SkillFile,
        SkillValidationError,
    )

    try:
        local_doc = SkillDocument.from_json(dict(local))
        remote_doc = SkillDocument.from_json(dict(remote))
        base_doc = SkillDocument.from_json(dict(base)) if base is not None else None
    except SkillValidationError:
        return None, []

    results: list[MergeResult] = []
    description = merge_field(
        FieldInput(
            field="description",
            kind="text",
            base=base_doc.description if base_doc is not None else None,
            local=local_doc.description,
            remote=remote_doc.description,
        ),
        opts,
    )
    results.append(description)

    def files_of(doc: SkillDocument | None) -> dict[str, str]:
        return {file.path: file.content for file in doc.files} if doc is not None else {}

    base_files, local_files, remote_files = (
        files_of(base_doc),
        files_of(local_doc),
        files_of(remote_doc),
    )
    merged_files: dict[str, str] = {}
    for path in sorted(set(base_files) | set(local_files) | set(remote_files)):
        content, result = merge_skill_file(
            path,
            base=base_files.get(path),
            local=local_files.get(path),
            remote=remote_files.get(path),
            opts=opts,
            # A KNOWN base document makes each file's absence from it a FACT
            # (an add), not an unknown baseline (QN4); an unknown base document
            # keeps the legacy held-for-review meaning.
            base_present=None if base_doc is None else path in base_files,
        )
        results.append(result)
        if content is not None:
            merged_files[path] = content
    if any(result.outcome in ("needs-review", "refused") for result in results):
        return None, results

    merged_description = str(description.merged or "")
    try:
        merged = SkillDocument(
            remote_doc.name,
            merged_description,
            tuple(SkillFile(path, content) for path, content in sorted(merged_files.items())),
        )
    except SkillValidationError:
        # The per-file merge may have produced a SKILL.md whose frontmatter
        # description no longer matches the merged description field (they are
        # the same fact stored twice). Re-render the scalar; anything still
        # invalid (a removed SKILL.md, bad paths, caps) refuses instead.
        try:
            merged = SkillDocument(
                remote_doc.name,
                merged_description,
                tuple(SkillFile(path, content) for path, content in sorted(merged_files.items())),
            ).with_description(merged_description)
        except SkillValidationError:
            return None, results
    return merged.to_json(), results


def _project_skill(
    *,
    captured_local: Mapping[str, Any],
    accepted: Mapping[str, Any],
    current_local: Mapping[str, Any],
    remote_latest: Mapping[str, Any],
    opts: MergeOptions,
) -> ProjectionResult:
    """The two outbound stages over a skill manifest (C then M), no-op compared on M."""

    from local_operator.hub_sync.skill_document import (
        SkillDocument,
        SkillValidationError,
    )

    stage_one, results_one = merge_skill_document(
        captured_local, current_local, accepted, opts=opts
    )
    if stage_one is None:
        return ProjectionResult(
            fields={},
            no_op=False,
            results=tuple(results_one),
            warnings=(),
            unresolved=tuple(r.field for r in results_one if r.outcome == "needs-review")
            or ("document",),
        )
    stage_two, results_two = merge_skill_document(accepted, stage_one, remote_latest, opts=opts)
    results = results_one + results_two
    if stage_two is None:
        return ProjectionResult(
            fields={},
            no_op=False,
            results=tuple(results),
            warnings=(),
            unresolved=tuple(r.field for r in results if r.outcome == "needs-review")
            or ("document",),
        )
    try:
        remote_canonical = SkillDocument.from_json(dict(remote_latest)).to_json()
    except SkillValidationError:
        remote_canonical = None
    no_op = remote_canonical is not None and fingerprint_snapshot(
        "skill", stage_two
    ) == fingerprint_snapshot("skill", remote_canonical)
    warnings = tuple(dict.fromkeys(w for r in results for w in r.warnings))
    unresolved = tuple(r.field for r in results if r.outcome == "needs-review")
    return ProjectionResult(
        fields=stage_two,
        no_op=no_op,
        results=tuple(results),
        warnings=warnings,
        unresolved=unresolved,
    )


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
    if kind == "skill":
        return _project_skill(
            captured_local=captured_local,
            accepted=accepted,
            current_local=current_local,
            remote_latest=remote_latest,
            opts=opts,
        )
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
    if kind == "skill":
        from local_operator.hub_sync.skill_document import SkillDocument

        return SkillDocument.from_json(dict(fields)).to_json()
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
    if projection.unresolved:
        # A stage that could not even build its documents (an invalid remote
        # latest, say) returns NO per-file results; the sentinel field names
        # keeps that from reading as "nothing to merge" and publishing empty
        # (round-2 R15).
        return "merge-refused"
    return None
