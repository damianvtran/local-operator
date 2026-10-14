"""Capacity + retention recovery: reclaim disk without killing a live fleet.

WHY THIS EXISTS. Every install generation is a whole venv, and iteration 1 froze
the automatic GC at the promotion tail (``update.py`` returns right after the flip
with no prune) because discovery/argv/open-file absence is not a lifetime-pin
proof: an old process can load an asset later with no handle open now. Freezing GC
is the safe POSITIVE rule, but it makes disk exhaustion possible in principle —
so this module is the OTHER half: it reclaims a generation ONLY when positive
evidence says nothing can still be running from it, and never by killing a
process or by guessing that an absence means a death.

THE THREE PIECES, and each states a boundary rather than a mechanism:

* :func:`capacity_request` — the TRIGGER. Measure generation count and free bytes
  against policy settings; when either threshold is crossed, write the durable
  ``handoff.capacity`` sub-map on the EXISTING operation record (no new queue, no
  new lock) and report it idempotently.
* :func:`reclaimable_now` — the SAFE-GC PREDICATE, positive-evidence only, built
  around :func:`update.prune_generations` and never a second deletion path.
* :func:`run_capacity_recovery` — the CONTROLLER: pick the least-critical ELIGIBLE
  cohort, request its checkpointed move, bounded wait, RE-CENSUS, reclaim only what
  is now positively unreferenced, then leave the caller to retry. It NEVER idles
  the whole fleet, never kills, and never deletes an unknown root.

WHAT THIS DOES NOT DO, stated so nobody reads more into it than it claims:
* It does not stop an arbitrary external ``rm`` from deleting a generation. The
  protection is that OUR deletion path consults the census; a noncooperator is out
  of scope here exactly as it is for the install mutex.
* It does not promise unlimited disk. Unknown or unresolvable roots are retained
  permanently, so exhaustion remains possible in principle and the honest answer
  at that point is a capacity error, not a deletion.
* Reclaiming a root a live record still names requires that record to become
  POSITIVELY dead (birth-proven, via ``procstate.same_birth``); an unresolved
  birth is doubt, and doubt is not death.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from local_operator import install_transaction

#: Policy, not correctness. These bound WHEN recovery is attempted; they never
#: decide WHAT is safe to delete — the predicate does that. Kept here rather than
#: in :mod:`local_operator.install_defaults` because they are read inside the
#: already-heavy ``update`` import path this module is loaded from.
MAX_GENERATIONS = 8
MIN_FREE_BYTES = 2 * 1024**3

#: The recount bound: a re-census after a cohort move that still finds the cohort
#: referenced is reported ``partial`` rather than looped on. Bounded so a cohort
#: that cannot checkpoint ends the attempt with an honest error, not a spin.
RECOVERY_ATTEMPTS = 3


@dataclass(frozen=True)
class CapacityMeasurement:
    """The two readings the trigger is decided from, and which crossed."""

    generations: int
    free_bytes: int
    threshold: str  # "", "count", "free" or "both"

    @property
    def over(self) -> bool:
        return bool(self.threshold)


def measure_capacity(
    *,
    root: Path | None = None,
    max_generations: int = MAX_GENERATIONS,
    min_free_bytes: int = MIN_FREE_BYTES,
) -> CapacityMeasurement:
    """Count generations and free bytes under the install root.

    ``shutil.disk_usage`` reports the filesystem the stable root lives on, which is
    where every generation is written; the free-space floor therefore reads the
    right volume without enumerating trees. Count is the number of directory
    entries that are generations (not the pointer, not the launcher).
    """
    from local_operator import update as update_mod

    stable = update_mod.stable_root() if root is None else root
    generations = update_mod.generations_dir() if root is None else root / "generations"
    count = 0
    if generations.is_dir():
        count = sum(1 for path in generations.iterdir() if path.is_dir() and not path.is_symlink())
    try:
        free = shutil.disk_usage(stable).free
    except OSError:
        # An unreadable free reading is NOT "plenty of room": treat it as over the
        # free threshold so the caller takes the safe path (attempt recovery rather
        # than skip it). Reclaiming is still guarded by the predicate, so an
        # over-eager trigger can delete nothing unsafe.
        free = -1
    crossed = []
    if count > max_generations:
        crossed.append("count")
    if free < min_free_bytes:
        crossed.append("free")
    threshold = "both" if len(crossed) == 2 else (crossed[0] if crossed else "")
    return CapacityMeasurement(count, free, threshold)


def capacity_request(
    root: Path,
    identifier: str,
    *,
    max_generations: int = MAX_GENERATIONS,
    min_free_bytes: int = MIN_FREE_BYTES,
    measurement: CapacityMeasurement | None = None,
) -> dict[str, Any] | None:
    """Return the durable capacity entry when pressure is present, else ``None``.

    IDEMPOTENT BY RECORD, not by memory: re-entering capacity while an OPEN request
    (``state == "requested"``) is already on the operation returns that entry
    unchanged instead of writing a second one — the durable record is the single
    source of "a request is in flight", so two callers serialise through the
    existing transaction rather than a new lock.
    """
    reading = measurement or measure_capacity(
        root=root, max_generations=max_generations, min_free_bytes=min_free_bytes
    )
    if not reading.over:
        return None
    existing = install_transaction.capacity_status(root, identifier)
    if existing is not None and existing.get("state") == "requested":
        return existing
    return install_transaction.note_capacity(
        root, identifier, state="requested", threshold=reading.threshold
    )


@dataclass(frozen=True)
class ReclaimDecision:
    """One generation's eligibility, with the reason it was kept when it was."""

    generation: Path
    reclaimable: bool
    reason: str


@dataclass(frozen=True)
class ReclaimPlan:
    """What :func:`reclaimable_now` decided, reusing ``prune_generations``."""

    decisions: tuple[ReclaimDecision, ...] = ()
    references_complete: bool = True

    @property
    def reclaimable(self) -> tuple[Path, ...]:
        return tuple(d.generation for d in self.decisions if d.reclaimable)


def reclaimable_now(
    *,
    keep: int | None = None,
    referenced: Any = None,
) -> ReclaimPlan:
    """The generations positive evidence says nothing can run from, and the rest.

    THIS IS ``prune_generations`` WEARING A DECISION-ONLY FACE. It asks the same
    function with ``keep=0`` so the answer is exactly the set the retention policy
    would allow deleting regardless of the count margin — "not the pointer's
    target, not the running tree, named by no record, marker present-or-aged, no
    protected attempt" — and turns each :class:`PruneDecision` into a
    :class:`ReclaimDecision`. It is run with ``dry_run=True``, so it makes the
    DECISION without performing a tree or staging mutation (no generation removal,
    no staging-link sweep, no attestation). There is deliberately NO second
    predicate: a second implementation of "safe to delete" is how the two drift and
    the safer one stops being the one that runs.

    ``keep=0`` is the key choice: the count margin is a retention preference, and
    the controller's question is not "what would a prune keep" but "what is
    provably unreferenced". Passing the margin here would let a small fleet hold
    trees that are in fact reclaimable, and passing the default would make the
    answer depend on a number that is not evidence.

    AN INCOMPLETE CENSUS RECLAIMS NOTHING: with ``references_complete`` False every
    decision is a keep (``prune_generations`` enforces this), so the plan is honest
    rather than tempting.
    """
    from local_operator import update as update_mod

    if referenced is None:
        referenced = update_mod.referenced_install_roots()
    plan = update_mod.prune_generations(
        keep=0 if keep is None else keep,
        referenced=referenced,
        dry_run=True,
    )
    decisions: list[ReclaimDecision] = []
    for decision in plan.decisions:
        decisions.append(
            ReclaimDecision(
                generation=decision.path,
                # A generation is reclaimable exactly when the prune would REMOVE
                # it: the decision is the one authority for both.
                reclaimable=bool(decision.removed),
                reason=decision.reason,
            )
        )
    return ReclaimPlan(tuple(decisions), plan.references_complete)


@dataclass
class Cohort:
    """One eligible install root and the live records that name it."""

    root: Path
    records: list[tuple[Any, Path]] = field(default_factory=list)

    @property
    def referenced_roots(self) -> int:
        return len(self.records)


def eligible_cohorts(
    grouped: dict[Path, list[tuple[Any, Path]]] | None = None,
    *,
    current: Path | None = None,
) -> list[Cohort]:
    """Roots a live, positively-identified record names whose build is not current.

    ELIGIBLE is deliberately narrow: only a root a LIVE record names (so there is a
    session to ask to move), and only when it is not the pointer's target (asking
    the current build to move off itself would be absurd). Least-critical is the
    order the controller walks: FEWEST referenced roots first (the least to
    displace), then oldest generation NAME (name order is creation order, the same
    fact ``prune_generations`` relies on).
    """
    from local_operator import update as update_mod

    groups = update_mod.live_records_by_root() if grouped is None else grouped
    resolved_current = update_mod.current_generation()
    if resolved_current is not None:
        resolved_current = Path(update_mod._real(resolved_current))
    if current is not None:
        resolved_current = Path(update_mod._real(current))
    cohorts = [
        Cohort(root=root, records=list(records))
        for root, records in groups.items()
        if records and root != resolved_current
    ]
    cohorts.sort(key=lambda cohort: (cohort.referenced_roots, cohort.root.name))
    return cohorts


def run_capacity_recovery(
    *,
    request_move: Callable[[Cohort], bool] | None = None,
    wait_move: Callable[[Cohort], bool] | None = None,
    reclaim: Callable[[Sequence[Path]], Any] | None = None,
    max_attempts: int = RECOVERY_ATTEMPTS,
) -> str:
    """Advance one capacity request: pick a cohort, move it, re-census, reclaim.

    THE FOUR STEPS, each bounded, and the whole method returning a STATE rather
    than looping forever:

    1. pick the least-critical eligible cohort (never the whole fleet);
    2. ask that ONE cohort to checkpoint/move and wait a BOUNDED time for it (the
       caller supplies ``request_move``/``wait_move``; this module never spawns or
       kills — the move is the §3 checkpointed path's job);
    3. RE-CENSUS: only a cohort whose roots are now positively unreferenced is
       reclaimable, and the census is re-read here rather than trusted from step 2;
    4. reclaim through the predicate, then stop and let the caller retry its
       preparation — the queued work is the caller's, and this method does not
       re-enter it.

    Returns one of ``"reclaimed"``, ``"moved-unreclaimed"``, ``"partial"`` or
    ``"nothing-eligible"``. A cohort that will not move within the bound is
    ``"partial"`` (honest: nothing reclaimed), never a kill and never a delete of a
    root that did not confirm.
    """
    cohorts = eligible_cohorts()
    if not cohorts:
        return "nothing-eligible"
    plan = reclaimable_now()
    if not plan.reclaimable:
        # Nothing is currently provable-safe. If a cohort exists we may still move
        # it so a FUTURE census can reclaim; if the move cannot happen, partial.
        cohort = cohorts[0]
        if (request_move is not None and request_move(cohort)) and (
            wait_move is None or wait_move(cohort)
        ):
            return "moved-unreclaimed"
        return "partial"
    cohort = cohorts[0]
    moved = True
    if request_move is not None:
        moved = bool(request_move(cohort)) and (wait_move is None or bool(wait_move(cohort)))
    if not moved:
        return "partial"
    # RE-CENSUS (step 3): a fresh read, not the one that chose the cohort.
    fresh = reclaimable_now()
    keep = set(plan.reclaimable)
    now = [path for path in fresh.reclaimable if path in keep]
    if not now:
        return "moved-unreclaimed"
    # ONLY A REAL RECLAIM REPORTS "reclaimed". A caller that passed no ``reclaim``
    # asked for the DECISION, so it gets "moved-unreclaimed": the cohort moved and
    # the roots are provably unreferenced, but nothing was deleted — claiming
    # "reclaimed" there would report a deletion that never happened.
    if reclaim is None:
        return "moved-unreclaimed"
    reclaim(now)
    return "reclaimed"


def capacity_error(measurement: CapacityMeasurement) -> str:
    """The honest sentence a pinned-but-unreclaimable fleet gets.

    It names the pressure and the refusal together, because "cannot prepare more"
    with no reason reads as a bug and invites a force.
    """
    return (
        "capacity: "
        f"{measurement.generations} generations, {measurement.free_bytes} bytes free; "
        "no generation is provably unreferenced, so nothing is reclaimed. Free disk "
        "or move an idle session to another device; this tool will not delete a root "
        "it cannot prove dead."
    )


def now() -> float:  # pragma: no cover - trivial seam
    return time.time()
