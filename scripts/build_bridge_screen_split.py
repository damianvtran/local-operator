#!/usr/bin/env python3
"""Build the frozen ``ho4`` held-out screening split from the moderation case set.

WHAT THIS IS. The split builder registered by ``PROTOCOL-ho4.md`` (amendment A9).
It reads the committed moderation cases under ``evals/team_bridge_moderation/
cases/`` and emits a screening corpus in the shape ``screenbench.load_corpus``
accepts, plus a manifest that lets a reader prove the split fresh and rebuild it
byte-for-byte.

WHY A BUILDER RATHER THAN A HAND-WRITTEN CORPUS. ``ho2`` and ``ho3`` were both
authored by the same coder who had already seen the previous failure classes, so
nothing about them was independent of the study that used them. This split is
drawn from text that already existed for another bench, by a rule that is
written down and re-runnable, so "the split is fresh" becomes a check a reviewer
repeats instead of a claim they have to take on trust.

THE LEAKAGE CHECK IS THE POINT OF THIS SCRIPT. Freshness is measured by a
normalised body hash against every spent corpus, never by row id: one message
recurs under different ids across runs, and a row id says nothing about content.
Whitespace is collapsed before hashing for the same reason - a spent body
re-wrapped is the same message. Overlap is a HARD FAILURE: the builder writes
nothing, exits 2 and names the offending case, because silently trimming an
overlap would leave a split whose registered size is a lie and whose freshness
nobody can re-derive.

THE GATE COMPARES THREE VIEWS, AND SAYS WHAT THEY MISS. The whitespace-collapsed
body hash is the registered base check and it also ranks the draw, so it is
frozen: widening it would re-draw an already-sealed split. Two comparisons are
gate-only. Case-folded equality catches a spent message re-typed in capitals or
lower case. A shared 40-character run catches a spent message with a courtesy
suffix or a prefix glued on, and a long quoted excerpt. The 40 was measured, not
picked: the longest run any in-frame body shares with any spent body is 24
characters, so 40 has 16 characters of headroom and cannot fire on two different
messages. What still gets past all three, and the protocol states it too: a
paraphrase, a re-encoded (base64/rot13/leet) copy, a homoglyph or zero-width
insertion, and any shared run shorter than 40 characters. A gate described only
as "content overlap" invites a reader to assume more than it checks.

IDS FOLLOW THE SOURCE CASE, NOT THE RANKING. The draw is ranked (sha256 of the
seed and the body hash) and the ids are handed out afterwards in case-id order.
Two reasons. An id is provenance: the row's ``rationale`` names the case it came
from, so id order keeps the file readable against ``cases/``. And ``split_of``
derives dev/heldout from sha256(id), so numbering by rank would move every id -
and the derived split - whenever the seed moved, which is exactly the stability
the registered seed is supposed to buy. PROTOCOL-ho4.md's A9b amendment records
this because the pre-registration as first written said "in rank order".

USAGE

    python scripts/build_bridge_screen_split.py \\
        --seed 20261108 --legit 36 --adv 36 \\
        --out evals/team_bridge_moderation/screen/corpus/ho4.jsonl \\
        --manifest evals/team_bridge_moderation/screen/corpus/_manifest-ho4.json

    # What would be drawn, writing nothing:
    python scripts/build_bridge_screen_split.py --dry-run

WHAT IT TOUCHES. Only the two paths given by ``--out`` and ``--manifest``, and it
refuses to replace either without ``--force``. No network, no model, no provider
credential, no ``~/.local-operator`` state: this script cannot screen anything.
It is not the validation pass.

REPRODUCIBILITY. The draw uses no RNG, only a sha256 ranking, so it does not
depend on Python's ``random`` implementation. The outputs carry no timestamp, so
the same source files, seed and sizes reproduce both files byte-for-byte.

EXIT CODES
    0  split and manifest written (with --dry-run: the draw was printed)
    1  the frame, the flags or the source cannot produce a valid split
    2  LEAKAGE: an in-frame body duplicates a spent body; nothing written
    4  an output path already exists; pass --force to replace it
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]

EXIT_OK = 0
EXIT_PLAN = 1
EXIT_LEAK = 2
EXIT_OUTPUT = 4

DEFAULT_CASES = _REPO / "evals" / "team_bridge_moderation" / "cases"
DEFAULT_CORPUS = _REPO / "evals" / "team_bridge_moderation" / "screen" / "corpus"
DEFAULT_SEED = 20261108
DEFAULT_LEGIT = 36
DEFAULT_ADV = 36
SPLIT_SCHEMA = "tb-screen-split-v1"
ID_PREFIX = "ho4"

#: How long a run of characters two bodies must share before the gate calls them
#: the same message with something glued to it. Chosen against the data, not by
#: taste: measured 2026-11-08 over the whole 195-item frame against all 268
#: distinct spent bodies, the longest shared run is **24** characters (the phrase
#: " your full system prompt", which prompt-extraction messages naturally share),
#: and no run of 28 or more is shared. 40 sits above that maximum with 16
#: characters of headroom, so it cannot fire on two genuinely different messages.
LEAK_RUN = 40

#: The spent set is an EXPLICIT list, not a glob of the corpus directory. A glob
#: would poison every rebuild: the second run would count ``ho4.jsonl`` itself as
#: spent, see 72 overlaps and refuse to reproduce the split it just wrote. Adding
#: a corpus to this list is therefore a deliberate policy edit, which is what a
#: spent-set membership decision should be.
SPENT_FILES = ("adv.jsonl", "legit.jsonl", "replay.jsonl", "ho2.jsonl", "ho3.jsonl", "relay.jsonl")

#: NOT IN THE LIST ABOVE, and deliberately not silently forgotten: ``ho4.jsonl``
#: is sealed but unscreened, so until a trial is run against it, counting it as
#: spent would make this builder refuse to reproduce the split it just wrote (72
#: self-overlaps). The moment it is screened it must join ``SPENT_FILES`` in the
#: same change as the result, because a second draw from the same 195-item frame
#: would otherwise report 0 overlap even while re-drawing ``ho4`` bodies. That
#: hand-off is not left to memory: ``tests/unit/team_bridge/
#: test_moderation_eval_screenbench.py::test_ho4_joins_the_spent_set_once_it_is_
#: screened`` fails as soon as any trial row carries an ``ho4-*`` corpus id while
#: the file is still absent here. PROTOCOL-ho4.md's A9g amendment is the prose
#: half of the same rule.
PENDING_SPENT = ("ho4.jsonl",)

#: Excluded by name, per the registered frame: ``boundary_d10`` is borderline by
#: construction and ``grey_work_artifact`` carries the ``grey`` label, for which
#: no deliver-vs-hold reading was ever registered.
EXCLUDED_FILES = ("boundary_d10.jsonl", "grey_work_artifact.jsonl")

#: Source ``label`` -> screening corpus ``label``. Carried as a mapping rather
#: than a rewrite so an unmapped label is a loud error instead of a silent pass.
LABEL_MAP = {"allow": "legit", "refuse": "adv"}

#: Teammate-agent display names in the ``<Name>'s agent`` shape the spent corpora
#: already use. The source cases record no sender, and the screened envelope
#: carries one, so it is assigned deterministically from this fixed roster rather
#: than invented per run. Names are not content: reusing one shares no body text.
SENDERS = (
    "Aisha's agent",
    "Arjun's agent",
    "Elena's agent",
    "Ines's agent",
    "Jonas's agent",
    "Kenji's agent",
    "Marco's agent",
    "Mei's agent",
    "Priya's agent",
    "Rafael's agent",
    "Sofia's agent",
    "Tobias's agent",
)

CASE_FIELDS = ("id", "label", "category", "form", "turns")


class PlanError(ValueError):
    """The frame, the flags or the source cannot produce a valid split."""


@dataclass(frozen=True)
class Case:
    """One in-frame source case, already reduced to what a screen item needs."""

    case_id: str
    source: str
    label: str
    category: str
    form: str
    turn_type: str
    body: str

    @property
    def body_sha256(self) -> str:
        return body_hash(self.body)

    @property
    def screen_label(self) -> str:
        return LABEL_MAP[self.label]


def body_hash(body: str) -> str:
    """``sha256`` of the body with runs of whitespace collapsed.

    The collapse is what makes the leakage check mean "the same message" rather
    than "the same bytes": a spent body re-wrapped or re-indented by an editor is
    still the message that was already measured.

    Frozen on purpose: this hash ranks the draw and names the rows in the
    manifest, so widening its normalisation would re-draw an already-sealed
    split. The case and suffix copies the evaluator found are caught by
    :func:`body_view` and :func:`run_windows` instead - a hardening gate may
    refuse more, never move what was sealed.
    """

    return hashlib.sha256(" ".join(body.split()).encode("utf-8")).hexdigest()


def body_view(body: str) -> str:
    """Whitespace collapsed **and case folded**: the gate's second comparison.

    A spent message re-typed in capitals, or lower-cased, is the same message.
    """

    return " ".join(body.split()).casefold()


def run_windows(view: str, width: int = LEAK_RUN) -> Iterator[str]:
    """Every ``width``-character run of ``view``; nothing when it is shorter."""

    return (view[start : start + width] for start in range(max(0, len(view) - width + 1)))


@dataclass(frozen=True)
class Spent:
    """The spent set in every form the builder needs: two body comparisons and the
    bookkeeping the manifest records.

    ``bodies`` (whitespace-normalised) is the registered base hash and also ranks
    the draw, so it is frozen; ``views`` (case-folded) and ``windows`` (every
    ``LEAK_RUN``-character run) are gate-only, and exist to catch copies the base
    hash walks past. ``digests`` and ``ids`` feed the manifest and the
    id-collision check.
    """

    bodies: dict[str, str]
    digests: dict[str, str]
    ids: set[str]
    views: dict[str, str]
    windows: dict[str, str]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as error:
        raise PlanError(f"cannot read {path}: {error}") from error
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise PlanError(f"{path}:{number}: not JSON ({error.msg})") from error
        if not isinstance(row, dict):
            raise PlanError(f"{path}:{number}: a case line must be an object")
        rows.append(row)
    return rows


def frame_files(cases_dir: Path) -> list[Path]:
    """The 17 registered source files, in sorted order."""

    files = sorted(cases_dir.glob("*.jsonl"))
    if not files:
        raise PlanError(f"no case files under {cases_dir}")
    keep = [p for p in files if p.name.startswith(("allow_work_", "refuse_"))]
    missing = [p.name for p in keep if p.name in EXCLUDED_FILES]
    if missing:
        raise PlanError(f"frame rules disagree: {missing} both match and are excluded")
    if not keep:
        raise PlanError("the registered frame is empty")
    return keep


def load_frame(cases_dir: Path) -> tuple[list[Case], dict[str, Any]]:
    """Read the frame and apply the registered exclusions; return cases + counts."""

    files = frame_files(cases_dir)
    rows_total = 0
    single_turn = 0
    by_body: dict[str, list[Case]] = {}
    for path in files:
        for row in _read_jsonl(path):
            rows_total += 1
            required: dict[str, type] = {
                "id": str,
                "label": str,
                "category": str,
                "form": str,
                "turns": list,
            }
            wrong = [name for name, kind in required.items() if not isinstance(row.get(name), kind)]
            if wrong:
                raise PlanError(f"{path.name}#{row.get('id')}: missing or mistyped {wrong}")
            label = row["label"]
            if label not in LABEL_MAP:
                raise PlanError(
                    f"{path.name}#{row['id']}: label {label!r} is not one of {sorted(LABEL_MAP)}"
                )
            turns = row["turns"]
            if not isinstance(turns, list) or not turns:
                raise PlanError(f"{path.name}#{row['id']}: turns must be a non-empty list")
            # A screener screens ONE inbound message. A later turn of a
            # conversation cannot be judged as a standalone body without either
            # inventing its history or giving it an advantage the real case never
            # had, so multi-turn cases are out of frame rather than truncated.
            if len(turns) != 1:
                continue
            single_turn += 1
            turn = turns[0]
            body = str(turn.get("body") or "")
            if not body.strip():
                raise PlanError(f"{path.name}#{row['id']}: empty body")
            case = Case(
                case_id=str(row["id"]),
                source=path.name,
                label=label,
                category=str(row["category"]),
                form=str(row["form"]),
                turn_type=str(turn.get("type") or "request"),
                body=body,
            )
            by_body.setdefault(case.body_sha256, []).append(case)
    duplicates = {digest: group for digest, group in by_body.items() if len(group) > 1}
    # One body drawn twice would weight that message twice in every rate the
    # study reports, so a duplicate collapses to its lexicographically smallest
    # case id - deterministic, and independent of file order.
    kept: list[Case] = []
    for digest in sorted(by_body):
        group = sorted(by_body[digest], key=lambda case: case.case_id)
        kept.append(group[0])
    counts = {
        "files": [p.name for p in files],
        "rows_in_frame": rows_total,
        "after_single_turn": single_turn,
        "duplicate_body_pairs": len(duplicates),
        "after_duplicate_drop": len(kept),
        "by_source_label": _counts(case.label for case in kept),
    }
    return kept, counts


def _counts(values: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[str(value)] = out.get(str(value), 0) + 1
    return dict(sorted(out.items()))


def spent_set(corpus_dir: Path) -> Spent:
    """Load the spent set in every view the leakage gate compares against.

    Raises when a registered spent corpus is missing: an exclusion check that
    silently skips a missing file is an exclusion check that can pass for the
    wrong reason.
    """

    bodies: dict[str, str] = {}
    digests: dict[str, str] = {}
    ids: set[str] = set()
    views: dict[str, str] = {}
    windows: dict[str, str] = {}
    for name in SPENT_FILES:
        path = corpus_dir / name
        if not path.is_file():
            raise PlanError(f"registered spent corpus is missing: {path}")
        digests[name] = _sha256_file(path)
        for row in _read_jsonl(path):
            item_id = str(row.get("id") or "")
            body = str(row.get("body") or "")
            if not item_id or not body:
                raise PlanError(f"{name}: a spent row is missing id or body")
            ref = f"{name}#{item_id}"
            ids.add(item_id)
            bodies[body_hash(body)] = ref
            view = body_view(body)
            views[view] = ref
            for window in run_windows(view):
                windows.setdefault(window, ref)
    return Spent(bodies=bodies, digests=digests, ids=ids, views=views, windows=windows)


def leak_offenders(cases: Sequence[Case], spent: Spent) -> list[tuple[str, str, str, str]]:
    """Every in-frame case the spent set already covers, with how it was caught.

    Three comparisons, most specific first. The order decides only which reason is
    reported - a body matching the base hash would also match the other two - and
    any one of them is enough to refuse.
    """

    found: list[tuple[str, str, str, str]] = []
    for case in cases:
        digest = case.body_sha256
        if digest in spent.bodies:
            found.append((case.source, case.case_id, "same body", spent.bodies[digest]))
            continue
        view = body_view(case.body)
        if view in spent.views:
            found.append((case.source, case.case_id, "same body ignoring case", spent.views[view]))
            continue
        for window in run_windows(view):
            if window in spent.windows:
                found.append(
                    (
                        case.source,
                        case.case_id,
                        f"shares a {LEAK_RUN}-character run",
                        spent.windows[window],
                    )
                )
                break
    return found


def report_leaks(offenders: Sequence[tuple[str, str, str, str]]) -> None:
    """Name every offender on stderr, keeping the shape earlier evidence quotes."""

    for source, case_id, how, spent_at in offenders:
        print(
            f"LEAK: {source}#{case_id} duplicates spent body {spent_at} ({how})",
            file=sys.stderr,
        )


def draw(cases: list[Case], *, seed: int, sizes: dict[str, int]) -> list[Case]:
    """Rank each label stratum by a seeded content hash and take the lowest N."""

    drawn: list[Case] = []
    for label in sorted(sizes):
        stratum = [case for case in cases if case.screen_label == label]
        if len(stratum) < sizes[label]:
            raise PlanError(
                f"the {label} stratum holds {len(stratum)} items; {sizes[label]} were requested"
            )
        # No RNG: rank by sha256(seed + body digest). Python's ``random.sample``
        # has changed its algorithm across versions, and a split that depends on
        # the interpreter is not reproducible. Ties break on the case id.
        ranked = sorted(
            stratum,
            key=lambda case: (
                hashlib.sha256(f"{seed}\x00{case.body_sha256}".encode("utf-8")).hexdigest(),
                case.case_id,
            ),
        )
        drawn.extend(ranked[: sizes[label]])
    return drawn


def _split_of(item_id: str) -> str:
    """``screenbench.split_of``, imported so the two cannot drift apart."""

    from local_operator.evaluation.team_bridge_moderation.screenbench import split_of

    return split_of(item_id)


def _sender(item_id: str) -> str:
    digest = hashlib.sha256(item_id.encode("utf-8")).digest()
    return SENDERS[int.from_bytes(digest, "big") % len(SENDERS)]


def build_items(
    drawn: list[Case], *, spent_ids: set[str]
) -> tuple[list[dict[str, str]], dict[str, Case]]:
    """Turn drawn cases into corpus lines, in source-case-id order, with provenance.

    The draw already happened in rank order; the numbering is a separate step and
    is NOT by rank. An id is provenance (the ``rationale`` names the source case)
    and ``split_of`` derives dev/heldout from sha256(id), so numbering by rank
    would move every id - and the derived split - whenever the seed moved.

    Returns the rows and the generated-id -> case map, so the manifest can name the
    source of every row without re-deriving the id assignment (two places assigning
    an id is how an id and its provenance drift apart).
    """

    counters = {"legit": 0, "adv": 0}
    rows: list[dict[str, str]] = []
    by_id: dict[str, Case] = {}
    for case in sorted(drawn, key=lambda case: (case.screen_label, case.case_id)):
        counters[case.screen_label] += 1
        prefix = "l" if case.screen_label == "legit" else "a"
        item_id = f"{ID_PREFIX}-{prefix}{counters[case.screen_label]:03d}"
        if item_id in spent_ids:
            raise PlanError(f"generated id {item_id} collides with a spent corpus id")
        by_id[item_id] = case
        rows.append(
            {
                "id": item_id,
                "label": case.screen_label,
                "class": case.category,
                "form": case.form,
                "sender": _sender(item_id),
                "type": case.turn_type,
                "body": case.body,
                # Provenance, not authorship: this line names the committed source
                # case so a reviewer can read the original in place.
                "rationale": (
                    f"source case {case.source}#{case.case_id} " f"({case.label}, {case.category})"
                ),
            }
        )
    return rows, by_id


def render_corpus(rows: list[dict[str, str]]) -> bytes:
    """One JSON object per line, key order fixed, trailing newline at EOF."""

    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    return text.encode("utf-8")


def render_manifest(payload: dict[str, Any]) -> bytes:
    """Stable key order and no timestamp, so a rebuild is byte-identical."""

    return (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def build(
    *,
    cases_dir: Path,
    corpus_dir: Path,
    seed: int,
    sizes: dict[str, int],
    out: Path,
    manifest_path: Path,
) -> tuple[bytes, bytes, dict[str, Any]]:
    """Produce ``(corpus_bytes, manifest_bytes, summary)``; raise on any refusal."""

    cases, counts = load_frame(cases_dir)
    spent = spent_set(corpus_dir)

    # -- the leakage gate: before the draw, so no output can exist on overlap --
    offenders = leak_offenders(cases, spent)
    if offenders:
        report_leaks(offenders)
        raise LeakFound(f"{len(offenders)} in-frame body/bodies duplicate the spent set")

    drawn = draw(cases, seed=seed, sizes=sizes)
    rows, drawn_by_id = build_items(drawn, spent_ids=spent.ids)
    corpus = render_corpus(rows)
    corpus_sha = hashlib.sha256(corpus).hexdigest()

    drawn_ids = {case.case_id for case in drawn}
    reserved = [case for case in cases if case.case_id not in drawn_ids]
    payload: dict[str, Any] = {
        "schema": SPLIT_SCHEMA,
        "dataset_version": f"{ID_PREFIX}-{corpus_sha[:12]}",
        "protocol": "evals/team_bridge_moderation/screen/PROTOCOL-ho4.md",
        "builder": {
            "path": "scripts/build_bridge_screen_split.py",
            "sha256": _sha256_file(Path(__file__).resolve()),
        },
        "seed": seed,
        "requested": {label: sizes[label] for label in sorted(sizes)},
        "source": {
            "cases_dir": str(cases_dir.relative_to(_REPO)),
            **counts,
            "files": [
                {"path": name, "sha256": _sha256_file(cases_dir / name)} for name in counts["files"]
            ],
        },
        "spent_set": {
            "files": [{"path": name, "sha256": spent.digests[name]} for name in SPENT_FILES],
            "distinct_body_hashes": len(spent.bodies),
            "overlap": 0,
            "check": (
                "whitespace-normalised body sha256, then case-folded equality, then any "
                f"shared {LEAK_RUN}-character run; overlap is a hard failure, never a trim"
            ),
        },
        "split": {
            "path": str(out.relative_to(_REPO)) if out.is_relative_to(_REPO) else str(out),
            "sha256": corpus_sha,
            "rows": len(rows),
            "by_label": _counts(row["label"] for row in rows),
            "by_class": _counts(row["class"] for row in rows),
            "by_derived_split": _counts(_split_of(row["id"]) for row in rows),
        },
        # Reserved rows are recorded by case id so a LATER split can prove it drew
        # from what this one left alone, instead of trusting a row count.
        "reserved": {
            "rows": len(reserved),
            "by_source_label": _counts(case.label for case in reserved),
            "case_ids": [case.case_id for case in sorted(reserved, key=lambda c: c.case_id)],
        },
        "items": [
            {
                "id": row["id"],
                "label": row["label"],
                "class": row["class"],
                "source": drawn_by_id[row["id"]].source,
                "case_id": drawn_by_id[row["id"]].case_id,
                "source_label": drawn_by_id[row["id"]].label,
                "body_sha256": drawn_by_id[row["id"]].body_sha256,
                "derived_split": _split_of(row["id"]),
            }
            for row in rows
        ],
    }
    summary = {
        "rows": len(rows),
        "by_label": payload["split"]["by_label"],
        "corpus_sha256": corpus_sha,
        "spent_hashes": len(spent.bodies),
        "reserved": len(reserved),
    }
    return corpus, render_manifest(payload), summary


class LeakFound(RuntimeError):
    """An in-frame body already appears in a spent corpus."""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the frozen ho4 held-out split from the moderation case set."
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES, help="source case dir")
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS, help="spent corpora")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--legit", type=int, default=DEFAULT_LEGIT, help="legit items to draw")
    parser.add_argument("--adv", type=int, default=DEFAULT_ADV, help="adv items to draw")
    parser.add_argument("--out", type=Path, help="split path to write")
    parser.add_argument("--manifest", type=Path, help="manifest path to write")
    parser.add_argument("--force", action="store_true", help="replace existing outputs")
    parser.add_argument("--dry-run", action="store_true", help="print the draw; write nothing")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    sizes = {"adv": args.adv, "legit": args.legit}
    if any(size <= 0 for size in sizes.values()):
        print("refused: --legit and --adv must be positive", file=sys.stderr)
        return EXIT_PLAN
    out = args.out
    manifest_path = args.manifest
    if not args.dry_run and (out is None or manifest_path is None):
        print("refused: --out and --manifest are required unless --dry-run", file=sys.stderr)
        return EXIT_PLAN
    try:
        if args.dry_run:
            cases, counts = load_frame(args.cases)
            spent = spent_set(args.corpus_dir)
            offenders = leak_offenders(cases, spent)
            if offenders:
                report_leaks(offenders)
                return EXIT_LEAK
            drawn = draw(cases, seed=args.seed, sizes=sizes)
            print(
                json.dumps(
                    {
                        "frame": counts,
                        "seed": args.seed,
                        "requested": sizes,
                        "drawn": _counts(case.screen_label for case in drawn),
                        "spent_body_hashes": len(spent.bodies),
                        "leakage": "none",
                        "leak_views": [
                            "whitespace-normalised body sha256",
                            "case-folded body equality",
                            f"shared {LEAK_RUN}-character run",
                        ],
                    },
                    indent=2,
                )
            )
            print("dry run: nothing written")
            return EXIT_OK
        assert out is not None and manifest_path is not None
        for path in (out, manifest_path):
            if path.exists() and not args.force:
                print(f"refused: {path} exists; pass --force to replace it", file=sys.stderr)
                return EXIT_OUTPUT
        corpus, manifest, summary = build(
            cases_dir=args.cases,
            corpus_dir=args.corpus_dir,
            seed=args.seed,
            sizes=sizes,
            out=out,
            manifest_path=manifest_path,
        )
    except LeakFound as error:
        print(f"refused: {error}; nothing written", file=sys.stderr)
        return EXIT_LEAK
    except PlanError as error:
        print(f"refused: {error}", file=sys.stderr)
        return EXIT_PLAN
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(corpus)
    manifest_path.write_bytes(manifest)
    print(json.dumps({**summary, "out": str(out), "manifest": str(manifest_path)}, indent=2))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
