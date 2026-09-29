"""The three-way merge core (design A4-A8, B2.2): deterministic, model-agnostic.

THE SHAPE OF THE GUARANTEE. Everything this module decides on its own (which
side edited an atom, which side deleted it, whether a shortening is a removal)
is a pure function of three texts. A model can only ever be asked about the
ONE case the table cannot decide (both sides edited the same atom
differently), it can only PROPOSE, and :func:`validate_proposal` accepts or
rejects the proposal deterministically. That is what turns "never blindly
overwrite either side's edits or removals" from a prompt instruction into a
checkable property, and it is why both directions of the hub protocol can run
the same fixture (``tests/fixtures/hub_merge/vectors.json``) with no model.

WHAT LIVES HERE AND WHY NOT IN ``resolver.py``. The request/proposal
dataclasses and the :class:`Resolver` protocol are defined in this module and
re-exported by ``resolver.py``: the merge core must be importable (and unit
testable with a scripted resolver) without touching the model stack, and the
resolver needs these types, so the light half owns them.

BASELINE SEMANTICS. ``base`` is the last remote text this machine integrated
(design A2). After a pull-merge the baseline becomes the REMOTE text merged in,
not the merged result: local edits that survived the merge are then still
"local edits since base", and a section the user deleted stays a local removal
against the very remote text they declined to keep. That differs from the
design's literal "B := text now identical on both sides" only when local edits
survive a merge (then no such text exists). Recorded in the design doc as the
amendment under A2.2 and Implementation note 9.
"""

from __future__ import annotations

import dataclasses
import difflib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Protocol, Sequence

from local_operator.hub_sync import segment as seg

Provenance = Literal[
    "unchanged", "kept-local", "taken-remote", "combined", "removal-honored", "unresolved"
]
Outcome = Literal["unchanged", "merged", "needs-review", "refused"]

#: "strongest wins" order for a region's roll-up (A8): a tidy region must never
#: hide an unresolved atom.
_STRENGTH: dict[str, int] = {
    "unchanged": 0,
    "kept-local": 1,
    "taken-remote": 2,
    "removal-honored": 3,
    "combined": 4,
    "unresolved": 5,
}

#: The default cap when a caller supplies none: the team brief cap, the largest
#: text field the registries hold.
DEFAULT_MAX_CHARS = 32_768

#: A local shortening counts as a deliberate removal (A5) at or below this
#: fraction of the base atom's tokens.
_REDUCED_RATIO = 0.85


# -- resolver protocol and its data ---------------------------------------------


@dataclass(frozen=True)
class Removal:
    """Text that MUST NOT reappear in a proposal, and who removed it."""

    text: str
    removed_by: Literal["local", "remote", "both"]
    atom_id: str = ""


@dataclass(frozen=True)
class ConflictRequest:
    """One conflict group: one atom edited differently on both sides (A6.1).

    A group is one aligned atom triple. The prompt therefore carries that atom's
    three texts plus its immediate neighbours as read-only context — never the
    field — which is what keeps a 30-section brief with three conflicts at three
    small calls (B2.6) and makes the recombination mechanical rather than the
    model's job.
    """

    field: str
    heading: str | None
    base: str | None
    local: str
    remote: str
    removals: tuple[Removal, ...] = ()
    keep_verbatim: tuple[str, ...] = ()
    max_chars: int = DEFAULT_MAX_CHARS
    #: Validator violations of the previous proposal, appended to the re-ask (B2.5).
    feedback: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConflictProposal:
    """What a resolver returns for a group (A6.1). ``covers``/``drops`` are ids."""

    text: str
    covers: tuple[str, ...] = ()
    drops: tuple[str, ...] = ()
    notes: str = ""
    #: Model calls this proposal cost (retries included) and the model that made it.
    attempts: int = 1
    model: str | None = None


ResolverErrorClass = Literal[
    "provider-error", "model-unavailable", "prompt-too-long", "invalid-output", "cancelled"
]


class ResolverError(Exception):
    """A resolver could not produce a proposal; ``cls`` is the B4.3 failure class."""

    def __init__(
        self,
        cls: ResolverErrorClass,
        detail: str = "",
        *,
        subclass: str | None = None,
        attempts: int = 0,
    ) -> None:
        super().__init__(f"{cls}: {detail}" if detail else cls)
        self.cls = cls
        self.detail = detail
        #: For ``provider-error``: quota | timeout | offline | transient.
        self.subclass = subclass
        self.attempts = attempts


class Resolver(Protocol):
    def resolve(self, req: ConflictRequest) -> ConflictProposal: ...


# -- inputs and outputs -----------------------------------------------------------


@dataclass(frozen=True)
class FieldInput:
    field: str
    kind: Literal["markdown", "text", "roster", "scalar"]
    base: "str | list[Any] | None"  # None = baseline unknown for this field
    local: "str | list[Any]"
    remote: "str | list[Any]"


@dataclass(frozen=True)
class MergeOptions:
    prefer: Literal["none", "local", "remote"] = "none"
    allow_llm: bool = True
    acknowledge_unknown_baseline: bool = False
    max_chars: int | None = None
    #: Roster cap (teams: 64 slots). Refusal text is the hub's own vocabulary.
    max_items: int | None = None
    resolver: Resolver | None = None


@dataclass(frozen=True)
class AtomReport:
    id: str
    provenance: str
    removed_by: str | None = None
    note: str = ""


@dataclass(frozen=True)
class RegionReport:
    id: str
    heading: str
    provenance: str
    removed_by: str | None
    atoms: tuple[AtomReport, ...]
    base: Any
    local: Any
    remote: Any
    result: Any
    dropped: Any = None
    note: str = ""
    #: The region's plain name (heading without ``#``s; the field name for
    #: scalars/rosters) — the key vectors and roll-up sentences use.
    name: str = ""


@dataclass(frozen=True)
class EngineInfo:
    mode: Literal["deterministic", "llm", "replace"] = "deterministic"
    model: str | None = None
    attempts: int = 0
    chunks: int = 0
    #: B4.3 class of the failure that left a group unresolved, if any.
    failure_class: str | None = None


@dataclass(frozen=True)
class MergeResult:
    """A8 in dataclass form. ``merged`` equals the LOCAL value unless ``outcome == "merged"``."""

    field: str
    outcome: Outcome
    merged: "str | list[Any]"
    regions: tuple[RegionReport, ...] = ()
    warnings: tuple[str, ...] = ()
    # ``dataclasses.field``, not the bare name: this class declares a FIELD called
    # ``field`` (the merged field's name), which shadows the imported function inside
    # the class body and made the defaults unbound.
    counts: Mapping[str, int] = dataclasses.field(default_factory=dict)
    engine: EngineInfo = dataclasses.field(default_factory=EngineInfo)
    #: The hub's rule text when ``outcome == "refused"``.
    refusal: str = ""
    kind: str = "markdown"

    @property
    def unresolved(self) -> tuple[RegionReport, ...]:
        return tuple(r for r in self.regions if r.provenance == "unresolved")

    def to_json(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "outcome": self.outcome,
            "merged": self.merged,
            "regions": [
                {
                    "id": r.id,
                    "heading": r.heading,
                    "name": r.name,
                    "provenance": r.provenance,
                    "removed_by": r.removed_by,
                    "atoms": [
                        {
                            "id": a.id,
                            "provenance": a.provenance,
                            "removed_by": a.removed_by,
                            "note": a.note,
                        }
                        for a in r.atoms
                    ],
                    "base": r.base,
                    "local": r.local,
                    "remote": r.remote,
                    "result": r.result,
                    "dropped": r.dropped,
                    "note": r.note,
                }
                for r in self.regions
            ],
            "warnings": list(self.warnings),
            "counts": dict(self.counts),
            "engine": {
                "mode": self.engine.mode,
                "model": self.engine.model,
                "attempts": self.engine.attempts,
                "chunks": self.engine.chunks,
                "failure_class": self.engine.failure_class,
            },
            "refusal": self.refusal,
        }


def _zero_counts() -> dict[str, int]:
    return {p: 0 for p in _STRENGTH}


# -- the decision table (A4.2), on states ----------------------------------------


@dataclass(frozen=True)
class _Decision:
    prov: str
    take: Literal["l", "r", "none", "conflict", "removal-vs-edit"]
    removed_by: str | None = None


def decide(b_present: bool, local: str | None, remote: str | None, lr_equal: bool) -> _Decision:
    """A4.2 verbatim. ``local``/``remote``: ``None`` (absent), ``"same"``, ``"mod"`` or ``"add"``.

    One function for atoms, headings and roster slots so the three cannot grow
    three slightly different tables.
    """

    if b_present:
        if local == "same" and remote == "same":
            return _Decision("unchanged", "l")
        if local == "mod" and remote == "same":
            return _Decision("kept-local", "l")
        if local == "same" and remote == "mod":
            return _Decision("taken-remote", "r")
        if local == "mod" and remote == "mod":
            return _Decision("kept-local", "l") if lr_equal else _Decision("unresolved", "conflict")
        if local is None and remote == "same":
            return _Decision("removal-honored", "none", "local")
        if local == "same" and remote is None:
            return _Decision("removal-honored", "none", "remote")
        if local is None and remote is None:
            return _Decision("removal-honored", "none", "both")
        # A removal against an edit: someone's intentional work would be lost.
        return _Decision("unresolved", "removal-vs-edit")
    if local == "add" and remote is None:
        return _Decision("kept-local", "l")
    if local is None and remote == "add":
        return _Decision("taken-remote", "r")
    if lr_equal:
        return _Decision("kept-local", "l")
    return _Decision("unresolved", "conflict")


# -- key tokens, removed spans, validators (A5, A6.2) ---------------------------

_URL_RE = re.compile(r"https?://[^\s)>\]\"']+")
_NUM_RE = re.compile(r"(?<![\w.])\d[\d.,:%/-]*\d%?|(?<![\w.])\d(?![\w.])")
_CODE_RE = re.compile(r"`[^`\n]+`")
_TEMPLATE_RE = re.compile(r"\{\{[^}\n]*\}\}")
# Identifier-shaped: has a joiner (snake_case, dotted, kebab), a call, or camelCase.
# A bare word is NOT an identifier; matching every word would make V2 demand
# the whole sentence verbatim.
_IDENT_RE = re.compile(
    r"\b[A-Za-z_][A-Za-z0-9]*(?:[._-][A-Za-z0-9]+)+\b|\b\w+\(\)|\b[a-z]+[A-Z][A-Za-z0-9]*\b"
)
_FENCE_LINE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_HEADING_LINE_RE = re.compile(r"^#{1,6}[ \t]+\S")


def key_tokens(text: str) -> set[str]:
    """Numbers, URLs, identifiers, code spans and ``{{…}}`` tokens (V2)."""

    found: set[str] = set()
    for rx in (_URL_RE, _CODE_RE, _TEMPLATE_RE, _IDENT_RE, _NUM_RE):
        for m in rx.finditer(text):
            token = m.group().rstrip(".,;:")
            if token:
                found.add(token)
    return found


def _removed_runs(base: str, edited: str) -> list[list[str]]:
    """Word runs of ``base`` that ``edited`` no longer has (A5 ``removed_spans``)."""

    bw, ew = seg.words(base), seg.words(edited)
    runs: list[list[str]] = []
    for tag, i1, i2, _, _ in difflib.SequenceMatcher(None, bw, ew, autojunk=False).get_opcodes():
        if tag in ("delete", "replace") and i2 > i1:
            runs.append(bw[i1:i2])
    return runs


def is_reduced(base: str, edited: str) -> bool:
    """A5: a strict shortening of the base atom that only DROPS words."""

    bw, ew = seg.words(base), seg.words(edited)
    if not bw or not ew or len(ew) >= len(bw):
        return False
    return len(ew) <= _REDUCED_RATIO * len(bw) and seg.is_subsequence(ew, bw)


def _has_run(words: Sequence[str], run: Sequence[str]) -> bool:
    n = len(run)
    if n == 0 or n > len(words):
        return False
    return any(list(words[i : i + n]) == list(run) for i in range(len(words) - n + 1))


def _key_like(word: str) -> bool:
    return any(ch.isdigit() for ch in word) or "_" in word


def regrows(text: str, removed: str) -> bool:
    """Does ``text`` bring back ``removed`` (V3)?

    A contiguous run of >= 4 words of the removed span (the whole span when it is
    shorter), or a sentence within 0.85 similarity of it. A single removed
    stop-word is not policed (it would forbid the word everywhere); a single
    key-like token (digits, underscore) still is.
    """

    rw = seg.words(removed)
    if not rw:
        return False
    tw = seg.words(text)
    k = min(4, len(rw))
    if len(rw) >= 2 or _key_like(rw[0]):
        if any(_has_run(tw, rw[i : i + k]) for i in range(len(rw) - k + 1)):
            return True
    if len(rw) >= 3:
        for region in seg.segment(text):
            for atom in region.atoms:
                if (
                    difflib.SequenceMatcher(None, atom.norm, seg.collapse_ws(removed)).ratio()
                    >= 0.85
                ):
                    return True
    return False


def validate_proposal(req: ConflictRequest, proposal: ConflictProposal) -> list[str]:
    """V1-V6 for one conflict group; an empty list means accepted.

    Deterministic and run on EVERY proposal, model-made or fallback: the model
    is never trusted, only checked.
    """

    problems: list[str] = []
    text = proposal.text
    if not text.strip():
        return ["V1: the proposal is empty"]
    ids = {"l1", "r1"} | ({"b1"} if req.base is not None else set())
    covers, drops = set(proposal.covers), set(proposal.drops)
    unknown = (covers | drops) - ids
    if unknown:
        problems.append(f"V1: unknown atom ids {sorted(unknown)}")
    # V2 coverage: both edited atoms carry through (a Δ/added atom may not be dropped, V4).
    for atom_id in ("l1", "r1"):
        if atom_id in drops:
            problems.append(f"V4: {atom_id} is an edited atom and may not be dropped")
        elif atom_id not in covers:
            problems.append(f"V2: {atom_id} is neither covered nor justified")
    base_tokens = key_tokens(req.base or "")
    for atom_id, side in (("l1", req.local), ("r1", req.remote)):
        if atom_id not in covers:
            continue
        missing = sorted(t for t in key_tokens(side) - base_tokens if t not in text)
        if missing:
            problems.append(f"V2: {atom_id} key tokens missing verbatim: {missing}")
    # V2 (content). Key tokens alone cannot see a dropped PLAIN-prose edit: a
    # proposal of just the base sentence "covers" both sides and passes. So a covered
    # side's newly added words (not in base) must mostly survive in the text. A
    # paraphrase that reuses none of them is refused too; that errs toward
    # needs-review, the safe direction, and never toward losing an edit.
    base_words = set(seg.words(req.base or ""))
    text_words = set(seg.words(text))
    for atom_id, side in (("l1", req.local), ("r1", req.remote)):
        if atom_id not in covers:
            continue
        added = [w for w in dict.fromkeys(seg.words(side)) if w not in base_words and len(w) >= 4]
        if added and sum(w in text_words for w in added) * 2 < len(added):
            problems.append(f"V2: {atom_id} added text is not carried: {added[:6]}")
    if "b1" in drops and req.base is not None:
        if not (is_reduced(req.base, req.local) or is_reduced(req.base, req.remote)):
            problems.append("V4: b1 was not removed or shortened by either side")
    # V3 no-regrow.
    for removal in req.removals:
        if regrows(text, removal.text):
            label = removal.atom_id or "a removed span"
            problems.append(
                f"V3: {label} was removed by {removal.removed_by} and reappears: "
                f"{removal.text[:80]!r}"
            )
    # V5 size.
    if len(text) > req.max_chars:
        problems.append(f"V5: {len(text)} chars exceeds the {req.max_chars} allowed here")
    # V6 structure: balanced fences, no new/renamed headings.
    in_fence = False
    fences = 0
    for line in text.split("\n"):
        if _FENCE_LINE_RE.match(line):
            fences += 1
            in_fence = not in_fence
        elif not in_fence and _HEADING_LINE_RE.match(line):
            problems.append("V6: the proposal contains a heading; headings may not change")
            break
    if fences % 2:
        problems.append("V6: fenced code blocks are unbalanced")
    return problems


def additive_fallback(req: ConflictRequest) -> ConflictProposal | None:
    """B2.7 deterministic no-model resolution: the sides are additive (one contains the other).

    Returns the longer side as a proposal, for the caller to VALIDATE like any
    other: a longer side that regrows something the shorter side deleted fails
    V3 and the group stays unresolved.
    """

    a, b = seg.collapse_ws(req.local), seg.collapse_ws(req.remote)
    if a and a in b:
        return ConflictProposal(req.remote, covers=("l1", "r1"), notes="additive")
    if b and b in a:
        return ConflictProposal(req.local, covers=("l1", "r1"), notes="additive")
    return None


# -- text merge --------------------------------------------------------------------


@dataclass
class _Triple:
    b: seg.Atom | None
    l: seg.Atom | None
    r: seg.Atom | None
    prov: str = "unchanged"
    take: str = "l"
    removed_by: str | None = None
    text: str | None = None  # a resolved conflict's text
    note: str = ""
    dropped: str | None = None
    l_index: int | None = None
    r_index: int | None = None
    atom_id: str = ""
    failure: str | None = None
    l_reduced: bool = False
    r_reduced: bool = False

    @property
    def is_conflict(self) -> bool:
        return self.take == "conflict"


@dataclass
class _Region:
    key: str
    name: str
    b: seg.Region | None
    l: seg.Region | None
    r: seg.Region | None
    triples: list[_Triple] = field(default_factory=list)
    heading: _Triple | None = None
    region_id: str = ""


def _mod(pair: tuple[int, bool] | None) -> str | None:
    if pair is None:
        return None
    return "mod" if pair[1] else "same"


def _triples_for(
    b_atoms: Sequence[seg.Atom],
    l_atoms: Sequence[seg.Atom],
    r_atoms: Sequence[seg.Atom],
    *,
    has_base: bool,
    counter: list[int],
) -> list[_Triple]:
    """Align B<->L and B<->R, then L<->R for what B does not explain (A4.1)."""

    pl = seg.align(b_atoms, l_atoms)
    pr = seg.align(b_atoms, r_atoms)
    used_l: set[int] = set()
    used_r: set[int] = set()
    triples: list[_Triple] = []

    def new_id() -> str:
        counter[0] += 1
        return f"a{counter[0]}"

    for i, b in enumerate(b_atoms):
        lp, rp = pl.pairs.get(i), pr.pairs.get(i)
        li = lp[0] if lp else None
        ri = rp[0] if rp else None
        if li is not None:
            used_l.add(li)
        if ri is not None:
            used_r.add(ri)
        la = l_atoms[li] if li is not None else None
        ra = r_atoms[ri] if ri is not None else None
        lr_equal = la is not None and ra is not None and la.norm == ra.norm
        d = decide(True, _mod(lp), _mod(rp), lr_equal)
        t = _Triple(
            b, la, ra, d.prov, d.take, d.removed_by, l_index=li, r_index=ri, atom_id=new_id()
        )
        t.l_reduced = bool(la and _mod(lp) == "mod" and is_reduced(b.norm, la.norm))
        t.r_reduced = bool(ra and _mod(rp) == "mod" and is_reduced(b.norm, ra.norm))
        if d.take == "removal-vs-edit":
            t.note = "removal-vs-edit"
        triples.append(t)

    rest_l = [j for j in range(len(l_atoms)) if j not in used_l]
    rest_r = [j for j in range(len(r_atoms)) if j not in used_r]
    direct = seg.align([l_atoms[j] for j in rest_l], [r_atoms[j] for j in rest_r])
    matched_r: set[int] = set()
    for a_idx, j in enumerate(rest_l):
        hit = direct.pairs.get(a_idx)
        if hit is not None:
            rj = rest_r[hit[0]]
            matched_r.add(rj)
            la, ra = l_atoms[j], r_atoms[rj]
            d = decide(False, "add", "add", la.norm == ra.norm)
            triples.append(
                _Triple(None, la, ra, d.prov, d.take, l_index=j, r_index=rj, atom_id=new_id())
            )
        else:
            d = decide(False, "add", None, False)
            triples.append(
                _Triple(None, l_atoms[j], None, d.prov, d.take, l_index=j, atom_id=new_id())
            )
    for rj in rest_r:
        if rj in matched_r:
            continue
        d = decide(False, None, "add", False)
        t = _Triple(None, None, r_atoms[rj], d.prov, d.take, r_index=rj, atom_id=new_id())
        if not has_base:
            t.note = "baseline-unknown"
        triples.append(t)
    return triples


def _order(triples: list[_Triple], r_count: int) -> list[_Triple]:
    """Output order: L's order; L-less triples after their nearest placed R predecessor."""

    out = sorted((t for t in triples if t.l_index is not None), key=lambda t: t.l_index or 0)
    by_r = {t.r_index: t for t in triples if t.r_index is not None}
    for t in sorted(
        (t for t in triples if t.l_index is None and t.r_index is not None),
        key=lambda t: t.r_index or 0,
    ):
        pos = 0
        for j in range((t.r_index or 0) - 1, -1, -1):
            prev = by_r.get(j)
            if prev is not None and prev in out:
                pos = out.index(prev) + 1
                break
        out.insert(pos, t)
    # B-only triples (removed on both sides) never emit; keep them out of the list.
    return out


def _emitted_atom(t: _Triple) -> seg.Atom | None:
    if t.take == "l":
        return t.l
    if t.take == "r":
        return t.r
    if t.take == "text" and t.text is not None:
        src = t.l or t.r
        kind = src.kind if src else "sentence"
        lead = (src.lead if src else "\n") or "\n"
        return seg.Atom(kind, t.text, lead)
    return None


def _heading_triple(
    b: seg.Region | None, l_reg: seg.Region | None, r_reg: seg.Region | None
) -> _Triple:
    """The heading is one more three-way atom; a demotion is an edit, not delete+add."""

    def state(x: seg.Region | None) -> str | None:
        if x is None:
            return None
        if b is None:
            return "add"
        return "same" if x.heading_norm == b.heading_norm else "mod"

    lr_equal = l_reg is not None and r_reg is not None and l_reg.heading_norm == r_reg.heading_norm
    d = decide(b is not None, state(l_reg), state(r_reg), lr_equal)
    t = _Triple(None, None, None, d.prov, d.take, d.removed_by)
    return t


class _Ctx:
    def __init__(self, inp: FieldInput, opts: MergeOptions) -> None:
        self.inp = inp
        self.opts = opts
        self.attempts = 0
        self.chunks = 0
        self.model: str | None = None
        self.llm_used = False
        self.failure: str | None = None
        self.systemic_failure: ResolverError | None = None
        self.region_counter = [0]
        self.atom_counter = [0]


def _regions_for(inp: FieldInput, opts: MergeOptions, ctx: _Ctx) -> list[_Region]:
    has_base = inp.base is not None
    bsegs = seg.segment(inp.base if isinstance(inp.base, str) else "") if has_base else ()
    lsegs = seg.segment(inp.local if isinstance(inp.local, str) else "")
    rsegs = seg.segment(inp.remote if isinstance(inp.remote, str) else "")
    bmap = {r.key: r for r in bsegs}
    lmap = {r.key: r for r in lsegs}
    rmap = {r.key: r for r in rsegs}

    order = [r.key for r in lsegs]
    rkeys = [r.key for r in rsegs]
    for idx, key in enumerate(rkeys):
        if key in order:
            continue
        pos = 0
        for j in range(idx - 1, -1, -1):
            if rkeys[j] in order:
                pos = order.index(rkeys[j]) + 1
                break
        order.insert(pos, key)
    for r in bsegs:
        if r.key not in order:
            order.append(r.key)

    out: list[_Region] = []
    for key in order:
        b, l, r = bmap.get(key), lmap.get(key), rmap.get(key)
        name = (l or r or b).name  # type: ignore[union-attr]
        region = _Region(key, name, b, l, r)
        ctx.region_counter[0] += 1
        region.region_id = f"r{ctx.region_counter[0]}"
        region.heading = _heading_triple(b, l, r) if (l or r or b) and key != "" else None
        region.triples = _triples_for(
            b.atoms if b else (),
            l.atoms if l else (),
            r.atoms if r else (),
            has_base=has_base,
            counter=ctx.atom_counter,
        )
        out.append(region)
    return out


def _removal_context(region: _Region) -> list[Removal]:
    return [
        Removal(t.b.norm, t.removed_by or "local", t.atom_id)  # type: ignore[union-attr,arg-type]
        for t in region.triples
        if t.take == "none" and t.b is not None
    ]


def _build_request(ctx: _Ctx, region: _Region, t: _Triple, out_len: int) -> ConflictRequest:
    assert t.l is not None and t.r is not None
    removals = _removal_context(region)
    if t.b is not None:
        # A side's SHORTENING is a removal — but only of tokens the OTHER side
        # left alone; if the other side changed those very tokens, the rewrite is
        # legitimately free to say something new there (A5).
        if t.l_reduced:
            for run in _removed_runs(t.b.norm, t.l.norm):
                if _has_run(seg.words(t.r.norm), run):
                    removals.append(Removal(" ".join(run), "local", "l1"))
        if t.r_reduced:
            for run in _removed_runs(t.b.norm, t.r.norm):
                if _has_run(seg.words(t.l.norm), run):
                    removals.append(Removal(" ".join(run), "remote", "r1"))
    cap = ctx.opts.max_chars or DEFAULT_MAX_CHARS
    room = max(200, cap - max(0, out_len - len(t.l.text)))
    neighbours = tuple(
        x.l.norm for x in region.triples if x.take == "l" and x.l is not None and x is not t
    )[:6]
    return ConflictRequest(
        field=ctx.inp.field,
        heading=region.name or None,
        base=t.b.norm if t.b is not None else None,
        local=t.l.norm,
        remote=t.r.norm,
        removals=tuple(removals),
        keep_verbatim=neighbours,
        max_chars=room,
    )


def _resolve_conflict(
    ctx: _Ctx, req: ConflictRequest
) -> tuple[ConflictProposal | None, str | None]:
    """A validated proposal for ``req``, or ``(None, failure_class)``.

    The ladder (B2.7): model (validated, re-asked ONCE with the violations) ->
    additive fallback (validated) -> unresolved. A systemic model failure stops
    further calls in this merge so N conflicts don't each burn a retry budget.
    """

    opts = ctx.opts
    failure: str | None = None
    if opts.allow_llm and opts.resolver is not None and ctx.systemic_failure is None:
        current = req
        for round_no in range(2):
            try:
                proposal = opts.resolver.resolve(current)
            except ResolverError as err:
                ctx.attempts += err.attempts
                if err.cls in ("model-unavailable", "provider-error", "cancelled"):
                    ctx.systemic_failure = err
                failure = err.cls if not err.subclass else f"{err.cls}/{err.subclass}"
                if err.cls == "cancelled":
                    raise
                break
            ctx.attempts += proposal.attempts
            ctx.chunks += 1
            ctx.model = proposal.model or ctx.model
            problems = validate_proposal(req, proposal)
            if not problems:
                ctx.llm_used = True
                return proposal, None
            failure = "invalid-output"
            current = ConflictRequest(**{**req.__dict__, "feedback": tuple(problems)})
    elif opts.allow_llm and opts.resolver is not None and ctx.systemic_failure is not None:
        e = ctx.systemic_failure
        failure = e.cls if not e.subclass else f"{e.cls}/{e.subclass}"
    fallback = additive_fallback(req)
    if fallback is not None and not validate_proposal(req, fallback):
        return fallback, None
    return None, failure


def _apply_prefer(t: _Triple, prefer: str) -> None:
    """Explicit caller precedence over an UNRESOLVED triple only (A4.3, A6.3)."""

    if prefer == "local":
        t.take = "l" if t.l is not None else "none"
        t.prov = "kept-local"
        t.dropped = t.r.text if t.r is not None else None
        t.removed_by = "local" if t.l is None else None
    else:
        t.take = "r" if t.r is not None else "none"
        t.prov = "taken-remote"
        t.dropped = t.l.text if t.l is not None else None
        t.removed_by = "remote" if t.r is None else None
    t.note = f"prefer={prefer}"


def _assemble(regions: list[_Region]) -> tuple[str, dict[str, seg.Region]]:
    """Render the merged text; also each region's output ``Region`` for reports."""

    rendered: list[seg.Region] = []
    per_region: dict[str, seg.Region] = {}
    for region in regions:
        heading_t = region.heading
        atoms: list[seg.Atom] = []
        r_total = len(region.r.atoms) if region.r else 0
        for t in _order(region.triples, r_total):
            a = _emitted_atom(t)
            if a is not None:
                atoms.append(a)
        heading = ""
        if region.key != "":
            take = heading_t.take if heading_t is not None else "l"
            if take == "conflict" or take == "removal-vs-edit":
                take = "l" if region.l is not None else "r"
            src = {"l": region.l, "r": region.r}.get(take)
            if src is not None:
                heading = src.heading
            elif atoms:
                # Structure needs a heading once atoms survive under it, even if
                # one side removed the heading line itself.
                fallback = region.r or region.l or region.b
                heading = fallback.heading if fallback else ""
            if not heading and not atoms:
                continue
        if not heading and not atoms:
            continue
        lead = (region.l.lead if region.l else (region.r.lead if region.r else "\n\n")) or "\n\n"
        out = seg.Region(region.key, region.name, heading, tuple(atoms), lead)
        rendered.append(out)
        per_region[region.region_id] = out
    return seg.render(rendered), per_region


def _region_rollup(region: _Region) -> tuple[str, str | None]:
    provs = [t.prov for t in region.triples]
    removed = {t.removed_by for t in region.triples if t.prov == "removal-honored" and t.removed_by}
    # A heading removal only counts while the section really is gone: under an
    # explicit ``prefer`` an atom of the section can survive it, and reporting
    # "removal honored" for a region whose text was emitted would misdescribe it.
    survivors = any(t.take in ("l", "r", "text") for t in region.triples)
    if region.heading is not None and not (region.heading.prov == "removal-honored" and survivors):
        provs.append(region.heading.prov)
        if region.heading.prov == "removal-honored" and region.heading.removed_by:
            removed.add(region.heading.removed_by)
    prov = max(provs, key=lambda p: _STRENGTH[p]) if provs else "unchanged"
    if not removed:
        removed_by = None
    elif len(removed) == 1:
        removed_by = next(iter(removed))
    else:
        removed_by = "both"
    return prov, removed_by


def _text_of(region: seg.Region | None) -> str | None:
    return seg.render([region]) if region is not None else None


def _merge_text(inp: FieldInput, opts: MergeOptions) -> MergeResult:
    ctx = _Ctx(inp, opts)
    has_base = inp.base is not None
    regions = _regions_for(inp, opts, ctx)
    local_text = seg.norm(inp.local if isinstance(inp.local, str) else "")
    remote_text = seg.norm(inp.remote if isinstance(inp.remote, str) else "")
    warnings: list[str] = []
    if not has_base:
        warnings.append("baseline-unknown")

    # 1) conflicts -> model/fallback; 2) leftovers -> prefer or unresolved.
    provisional, _ = _assemble(regions)
    for region in regions:
        for t in region.triples:
            if t.is_conflict:
                req = _build_request(ctx, region, t, len(provisional))
                proposal, failure = _resolve_conflict(ctx, req)
                if proposal is not None:
                    t.take, t.prov, t.text = "text", "combined", proposal.text.strip("\n")
                    t.note = proposal.notes
                else:
                    t.prov, t.failure = "unresolved", failure
                    if failure and ctx.failure is None:
                        ctx.failure = failure
    for region in regions:
        for t in region.triples:
            if t.prov == "unresolved" and opts.prefer != "none":
                _apply_prefer(t, opts.prefer)
        if region.heading is not None and region.heading.prov == "unresolved":
            if opts.prefer != "none":
                _apply_prefer(region.heading, opts.prefer)

    merged, out_regions = _assemble(regions)
    unresolved = any(t.prov == "unresolved" for region in regions for t in region.triples) or any(
        region.heading is not None and region.heading.prov == "unresolved" for region in regions
    )

    reports: list[RegionReport] = []
    counts = _zero_counts()
    for region in regions:
        prov, removed_by = _region_rollup(region)
        counts[prov] += 1
        atom_reports = tuple(
            AtomReport(
                t.atom_id, t.prov, t.removed_by if t.prov == "removal-honored" else None, t.note
            )
            for t in region.triples
        )
        dropped = [t.dropped for t in region.triples if t.dropped]
        if region.heading is not None and region.heading.dropped:
            dropped.append(region.heading.dropped)
        note = ""
        if any(t.note == "removal-vs-edit" for t in region.triples):
            note = "removal-vs-edit"
        elif any(t.note == "baseline-unknown" for t in region.triples):
            note = "baseline-unknown"
        elif any(t.take == "conflict" or t.prov == "unresolved" for t in region.triples):
            failures = {t.failure for t in region.triples if t.failure}
            note = "conflict" + (f" ({', '.join(sorted(failures))})" if failures else "")
        reports.append(
            RegionReport(
                id=region.region_id,
                heading=(
                    region.l.heading
                    if region.l
                    else (region.r.heading if region.r else (region.b.heading if region.b else ""))
                ),
                provenance=prov,
                removed_by=removed_by,
                atoms=atom_reports,
                base=_text_of(region.b),
                local=_text_of(region.l),
                remote=_text_of(region.r),
                result=_text_of(out_regions.get(region.region_id)),
                dropped="\n".join(dropped) if dropped else None,
                note=note,
                name=region.name,
            )
        )

    lc, rc, mc = len(local_text), len(remote_text), len(merged)
    big = max(lc, rc)
    if big and mc < 0.5 * big and lc >= 0.5 * big and rc >= 0.5 * big:
        warnings.append("large-shrink")

    refusal = ""
    cap = opts.max_chars
    if cap is not None and len(merged) > cap and not unresolved:
        refusal = f"must be at most {cap} characters (submitted {len(merged)})"

    engine = EngineInfo(
        mode="llm" if ctx.llm_used else "deterministic",
        model=ctx.model,
        attempts=ctx.attempts,
        chunks=ctx.chunks,
        failure_class=ctx.failure,
    )

    if refusal:
        outcome: Outcome = "refused"
        final = local_text
    elif unresolved or (not has_base and not opts.acknowledge_unknown_baseline):
        outcome, final = "needs-review", local_text
    elif seg.norm(merged) == local_text:
        outcome, final = "unchanged", local_text
    else:
        outcome, final = "merged", merged
    return MergeResult(
        field=inp.field,
        outcome=outcome,
        merged=final,
        regions=tuple(reports),
        warnings=tuple(warnings),
        counts=counts,
        engine=engine,
        refusal=refusal,
        kind=inp.kind,
    )


# -- scalar and roster ---------------------------------------------------------------


def _single_region(
    inp: FieldInput,
    name: str,
    prov: str,
    removed_by: str | None,
    result: Any,
    dropped: Any = None,
    note: str = "",
) -> RegionReport:
    return RegionReport(
        id="r1",
        heading=name,
        provenance=prov,
        removed_by=removed_by,
        atoms=(AtomReport("a1", prov, removed_by, note),),
        base=inp.base,
        local=inp.local,
        remote=inp.remote,
        result=result,
        dropped=dropped,
        note=note,
        name=name,
    )


def _merge_scalar(inp: FieldInput, opts: MergeOptions) -> MergeResult:
    """Whole-value three-way, no model: a manager cannot be "combined"."""

    local, remote = str(inp.local or "").strip(), str(inp.remote or "").strip()
    has_base = inp.base is not None
    base = str(inp.base or "").strip() if has_base else None
    warnings: list[str] = [] if has_base else ["baseline-unknown"]
    if has_base:
        d = decide(
            True,
            "same" if local == base else "mod",
            "same" if remote == base else "mod",
            local == remote,
        )
    else:
        d = _Decision("kept-local", "l") if local == remote else _Decision("unresolved", "conflict")
    dropped = None
    if d.take == "conflict":
        if opts.prefer == "none":
            outcome, merged, prov = "needs-review", local, "unresolved"
        else:
            merged = local if opts.prefer == "local" else remote
            dropped = remote if opts.prefer == "local" else local
            prov = "kept-local" if opts.prefer == "local" else "taken-remote"
            outcome = "merged" if merged != local else "unchanged"
    else:
        merged = remote if d.take == "r" else local
        prov = d.prov
        outcome = "merged" if merged != local else "unchanged"
        if not has_base and not opts.acknowledge_unknown_baseline and merged != local:
            outcome, merged = "needs-review", local
    if opts.max_chars is not None and len(merged) > opts.max_chars and outcome == "merged":
        return MergeResult(
            inp.field,
            "refused",
            local,
            (),
            tuple(warnings),
            _zero_counts(),
            EngineInfo(),
            refusal=f"must be at most {opts.max_chars} characters (submitted {len(merged)})",
            kind="scalar",
        )
    counts = _zero_counts()
    counts[prov] += 1
    return MergeResult(
        field=inp.field,
        outcome=outcome,  # type: ignore[arg-type]
        merged=merged if outcome == "merged" else local,
        regions=(_single_region(inp, inp.field, prov, None, merged, dropped),),
        warnings=tuple(warnings),
        counts=counts,
        kind="scalar",
    )


def _slot_key(slot: Mapping[str, Any]) -> tuple[str, str]:
    return (str(slot.get("kind") or "agent"), str(slot.get("role") or "").strip().casefold())


def _slot(slot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "role": str(slot.get("role") or "").strip(),
        "kind": str(slot.get("kind") or "agent"),
        "count": int(slot.get("count") or 1),
    }


def _merge_roster(inp: FieldInput, opts: MergeOptions) -> MergeResult:
    """Set merge by ``(kind, casefold(role))`` (A7): no text, no model."""

    has_base = inp.base is not None
    base_slots: list[Any] = list(inp.base or []) if has_base else []  # type: ignore[arg-type]
    base = {_slot_key(s): _slot(s) for s in base_slots}
    local_list = [_slot(s) for s in inp.local]  # type: ignore[union-attr]
    remote_list = [_slot(s) for s in inp.remote]  # type: ignore[union-attr]
    local = {_slot_key(s): s for s in local_list}
    remote = {_slot_key(s): s for s in remote_list}
    warnings: list[str] = [] if has_base else ["baseline-unknown"]

    order = [_slot_key(s) for s in local_list]
    for s in remote_list:
        if _slot_key(s) not in order:
            order.append(_slot_key(s))
    for key in base:
        if key not in order:
            order.append(key)

    reports: list[RegionReport] = []
    counts = _zero_counts()
    merged: list[dict[str, Any]] = []
    unresolved = False
    for n, key in enumerate(order, 1):
        b, l, r = base.get(key), local.get(key), remote.get(key)
        if has_base and b is not None:
            ls = None if l is None else ("same" if l["count"] == b["count"] else "mod")
            rs = None if r is None else ("same" if r["count"] == b["count"] else "mod")
        else:
            ls = None if l is None else "add"
            rs = None if r is None else "add"
        d = decide(
            b is not None, ls, rs, l is not None and r is not None and l["count"] == r["count"]
        )
        prov, removed_by, pick, dropped = d.prov, d.removed_by, d.take, None
        if pick in ("conflict", "removal-vs-edit"):
            if opts.prefer == "none":
                unresolved = True
                prov, pick = "unresolved", "l"
            else:
                pick = "l" if opts.prefer == "local" else "r"
                prov = "kept-local" if pick == "l" else "taken-remote"
                loser = r if pick == "l" else l
                dropped = None if loser is None else {"count": loser["count"]}
                if (l if pick == "l" else r) is None:
                    pick = "none"
        chosen = {"l": l, "r": r}.get(pick)
        if chosen is not None:
            merged.append(chosen)
        counts[prov] += 1
        label = f"{key[0]}/{(l or r or b or {}).get('role', key[1])}"

        def cnt(x: dict[str, Any] | None) -> Any:
            return None if x is None else {"count": x["count"]}

        reports.append(
            RegionReport(
                id=f"r{n}",
                heading=label,
                provenance=prov,
                removed_by=removed_by if prov == "removal-honored" else None,
                atoms=(AtomReport("a1", prov, removed_by if prov == "removal-honored" else None),),
                base=cnt(b),
                local=cnt(l),
                remote=cnt(r),
                result=cnt(chosen),
                dropped=dropped,
                note="removal-vs-edit" if d.take == "removal-vs-edit" else "",
                name=label,
            )
        )

    refusal = ""
    if opts.max_items is not None and len(merged) > opts.max_items and not unresolved:
        refusal = f"must hold at most {opts.max_items} items (submitted {len(merged)})"
    if refusal:
        outcome: Outcome = "refused"
        final: list[dict[str, Any]] = local_list
    elif unresolved or (not has_base and not opts.acknowledge_unknown_baseline):
        outcome, final = "needs-review", local_list
    elif [_key_count(s) for s in merged] == [_key_count(s) for s in local_list]:
        outcome, final = "unchanged", local_list
    else:
        outcome, final = "merged", merged
    return MergeResult(
        field=inp.field,
        outcome=outcome,
        merged=final,
        regions=tuple(reports),
        warnings=tuple(warnings),
        counts=counts,
        refusal=refusal,
        kind="roster",
    )


def _key_count(slot: Mapping[str, Any]) -> tuple[str, str, int]:
    return (*_slot_key(slot), int(slot.get("count") or 1))


# -- public entry points -----------------------------------------------------------


def merge_field(inp: FieldInput, opts: MergeOptions = MergeOptions()) -> MergeResult:
    """Sync. Deterministic core; calls ``opts.resolver`` ONLY for conflicts.

    Every proposal is validated (:func:`validate_proposal`); nothing the model
    says lands unchecked. Never raises for a model failure: a failed group is
    ``unresolved`` with ``engine.failure_class`` set (B2.7).
    """

    if inp.kind == "roster":
        return _merge_roster(inp, opts)
    if inp.kind == "scalar":
        return _merge_scalar(inp, opts)
    return _merge_text(inp, opts)


def replace_field(inp: FieldInput, *, take: Literal["local", "remote"]) -> MergeResult:
    """The explicit ``--replace`` path (B5.5): one region, dropped text echoed.

    Never reached by auto-update, the tool, or update-all — those cannot express
    it. The only function that can discard a side wholesale, and it says so.
    """

    keep, drop = (inp.remote, inp.local) if take == "remote" else (inp.local, inp.remote)
    if isinstance(keep, str):
        keep_n, drop_n, local_n = seg.norm(keep), seg.norm(str(drop)), seg.norm(str(inp.local))
    else:
        keep_n, drop_n, local_n = keep, drop, inp.local
    changed = keep_n != local_n
    prov = "taken-remote" if take == "remote" else "kept-local"
    counts = _zero_counts()
    counts[prov] += 1
    return MergeResult(
        field=inp.field,
        outcome="merged" if changed else "unchanged",
        merged=keep_n if changed else local_n,
        regions=(
            _single_region(
                inp,
                inp.field,
                prov,
                None,
                keep_n,
                dropped=drop_n if drop_n != keep_n else None,
                note="replaced",
            ),
        ),
        counts=counts,
        engine=EngineInfo(mode="replace"),
        kind=inp.kind,
    )


def dumps(result: MergeResult) -> str:
    return json.dumps(result.to_json(), ensure_ascii=False, sort_keys=True)
