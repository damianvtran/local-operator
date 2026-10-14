"""Mesh rolling updates, S1: the ``update`` capability and a SINGLE-peer update.

WHAT THIS SLICE IS (``docs/design/mesh-rolling-updates.md`` §8.2 item 1, plus the
trigger note ``docs/design/mesh-update-propagation.md`` §3/§10). The capability
row and its words (``types.py``), the one peer op ``net_update`` (slow pool) and
its local half ``peer_update``, the CLI verb ``lop network update <peer>``, and
the refusal vocabulary with the tests that pin it. It is ONLY the single-peer
half: no rollout record, no trigger hook, no policy key, no serial pass — those
are S2 (the member pass) and S3 (orchestration) and are deliberately absent.

THE AUTHORITY BOUND, restated here because every line below serves it (RU §2):
*a granted peer may cause THIS device to install a build this device could have
obtained itself — a version published on the device's own channel, strictly not
older than what is installed — and nothing else.* The peer never delivers code,
never names a ref, and never runs a command:

* the frame carries a target IDENTITY only — ``{"version": …, "source_ref": …}``
  — where ``source_ref`` is the origin's own build ref as provenance and is
  NEVER fetched, checked out or installed from (RU §2 rule 4);
* the member resolves the VERSION through its own installer against its own
  channel (``update.install_into_generation``, the pinned-PyPI shape the
  onboarding lane uses) and refuses a version its own channel cannot serve;
* monotonicity is enforced here: ``already_on_target`` is a no-op receipt and a
  target not strictly newer than the installed build is refused
  (``ahead_of_target``) — the standing path has no downgrades;
* an editable/source checkout takes no standing path at all
  (``editable_install``, "dev-tree skew is out of scope by design",
  ``update.py``'s own words), and a second update while one is in flight is
  refused by name (``update_in_progress``).

NEVER FORCED, and it is a structural property rather than a sentence: the
handler has no force parameter, the frame has no force key, and the one gate
before the install is a FRESH probe of this device's own session registry — a
member with any live ``busy`` record answers ``busy`` with its own reason and
NOTHING is touched (RU §4.3: the member's own registry is the sole authority on
the member's busyness; the origin cannot see, and must not infer, it). A
registry that cannot be read is treated as "cannot prove idle" and refuses in
the same shape, because the unsafe direction of a failed read is an install
over a running turn.

OPEN QUESTION (verify at implementation), carried from the design rather than
invented here: the reply map's origin-side ``no_grant`` skip (RU §3, first row)
is learned from "the member answering ``not_authorised``" — but the authoriser's
refusal frame is deliberately CODELESS (``wire.refusal_frame``: "which of
membership, epoch or capability failed" is withheld from the remote peer), so
the wire carries no machine token the origin can branch on for that row. S1
therefore reports the member's own sentence verbatim under a plain ``refused``
code and the ``no_grant`` remedy is composed by :func:`no_grant_remedy` for the
caller that CAN tell (a coded ``not_authorised`` from a handler, or S3's driver
reading the member's durable ``authorisation_refused`` audit). Resolving this is
a named S3 decision: either the authoriser learns to carry a code for this op
(a two-part-discipline extension, security-reviewed) or the record's first-row
classification moves off the wire entirely.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, Mapping, Sequence

from local_operator.network import wire
from local_operator.network.types import MeshRefusal

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from local_operator.network.relay import PeerLink, RelayServer

#: The owner-side deadline for one ``net_update`` request (seconds).
#:
#: The install runs INSIDE the bounded call when the member is idle (RU §3:
#: "a call that observes idle may perform the install inside its own bounded
#: deadline"), and the bound is the same 900 s the onboarding runner gives that
#: step (``onboard.STEP_TIMEOUTS["install"]``) — the one number this repo has
#: already measured for a pinned wheel install on a loaded host.
#:
#: THE WHOLE PASS RIDES INSIDE IT (agent review round 1, M2): the tail
#: (retention + sweep ≤ 60 s + re-engage ≤ 120 s) is clamped by
#: ``relay.slow_op_remaining_s()`` less :data:`TAIL_RESERVE_S`, because
#: ``_dispatch_slow`` DROPS a reply that lands after this deadline — so
#: install + tail + reserve fits the 900 s by construction, and a member that
#: cannot fit its tail degrades to notes rather than losing the receipt.
UPDATE_OP_DEADLINE_S = 900.0

#: The LOCAL control-socket budget for one ``peer_update`` hop (seconds).
#:
#: The control hop must OUTLAST the peer's slow-op deadline plus its reply
#: margin (``relay.SLOW_REPLY_MARGIN_S``) or the CLI would give up a moment
#: before the member's own sentence arrives, and report a bare timeout for an
#: install that was about to answer. The +30 is the control socket's own slack.
UPDATE_CONTROL_TIMEOUT_S = UPDATE_OP_DEADLINE_S + 30.0

#: How long the member waits for its own channel's answer about one version
#: (seconds). The read is a single small JSON document; the bound exists so a
#: hung index cannot stall the slow op before the idle gate, and it is the same
#: 5 s the updater's own PyPI read uses.
PUBLISHED_CHECK_TIMEOUT_S = 5.0

#: Where the in-flight marker lives, under the config root.
# Kept for readers of historical mesh metadata only; mutations use INSTALLROOT.
UPDATE_LOCK_RELPATH: tuple[str, ...] = ("network", "update.lock")

#: The install shapes the receipt's ``method`` may disclose (RU §3: "``method``
#: discloses the install shape on ``done``"). S1 shipped the generation-layout
#: pinned install; S2 adds the classic in-place fallback (§8.1: "the classic
#: uv-tool fallback ships until [the lane's `--to` verb] lands") as the
#: DISCLOSED LAST RESORT — taken only where the generation layout cannot run on
#: this member at all (``update.perform_upgrade``'s own documented degradation
#: boundary), and only after this pass's drain has proved no turn is live.
METHOD_GENERATION = "uv-tool-generation"
METHOD_CLASSIC = "uv-tool-classic"

#: The member pass's bounds (RU §4.2 steps 4–5), ALL SPENT INSIDE THE OP'S
#: DEADLINE and clamped by what is left of it (agent review round 1, M2):
#: ``relay.slow_op_remaining_s()`` is the sanctioned read of "what the
#: requester is still listening for", and ``_dispatch_slow`` drops a reply that
#: lands after the deadline — so install + retention + sweep + re-engage +
#: reserve must fit ``UPDATE_OP_DEADLINE_S`` by construction, and a member
#: that cannot fit its tail degrades to notes rather than losing the receipt to
#: an overrun.
#:
#: The sweep is "bounded at ~60 s" (RU §4.2 step 4). The budget is spent
#: SEQUENTIALLY across the live set — ``control.refresh_all`` is deliberately
#: sequential (each dial may wait out its timeout) — so the per-call timeout is
#: this budget divided by the live count, clamped to the control module's own
#: default; when the floor would push the sequential TOTAL past the budget
#: (live > budget/floor), only the targets that fit are asked and the rest are
#: named in the note (QA round 1, Q-2). A sweep that cannot run at all degrades
#: to counts; it never fails the pass (the runtimes also retire on their own
#: watch).
REFRESH_SWEEP_DEADLINE_S = 60.0
#: The per-dial floor for the sweep. Below this a dial would return before a
#: healthy runtime could answer, so the floor is not lowered to fit a huge live
#: set — the set is chunked instead (see above).
REFRESH_MIN_PER_CALL_S = 0.5

#: The re-engage budget (RU §4.2 step 5): the same 60 s ``relay.ENGAGE_DEADLINE_S``
#: gives its own warm path, per session, and this is the TOTAL the loop may
#: spend before it stops engaging — the op's deadline must still hold after the
#: install and the sweep, and a member with many dead sessions (exactly the
#: classic path's world) must not spend it one successor at a time. The budget
#: is checked before each engage AND the engage's own deadline is clamped to
#: what is left of it (agent review round 1, M2's tail note), so the total
#: cannot exceed the budget by one full engage.
REENGAGE_DEADLINE_S = 60.0
REENGAGE_BUDGET_S = 120.0

#: What the tail leaves unspent for the receipt write and the services thread's
#: start (seconds), subtracted from the remaining slow-op budget before the
#: tail is clamped to it.
TAIL_RESERVE_S = 30.0


# ---------------------------------------------------------------------------
# Runners (the seams a test or a later slice may stand in front of)
# ---------------------------------------------------------------------------


def detect_install_kind() -> Any:
    """This device's install kind, through the updater's own reader.

    A module-level function so a test process (which is the repo ``.venv``, i.e.
    EDITABLE, while the real member relay is an installed uv tool) can stand in
    the member's kind without lying about the host.
    """
    from local_operator import update as update_mod

    return update_mod.install_kind()


def generation_layout_supported() -> bool:
    """Whether this device can host the generation layout at all."""
    from local_operator import update as update_mod

    return bool(update_mod.generation_layout_supported())


def current_member_version() -> str:
    """The version of the build a FRESH ``lop`` would load on this device.

    ``disk_build`` rather than this process's metadata, because the monotonicity
    gate is about the INSTALL the member would move from — during a mixed-
    generation window the relay itself may be a generation behind the pointer,
    and comparing against the relay's own build would let a flip through that
    moves the member BACKWARD from what its own ``lop`` reports.
    """
    from local_operator import update as update_mod

    try:
        build = update_mod.disk_build()
    except Exception:  # noqa: BLE001 — an unreadable install answers through metadata
        build = None
    if build is not None and build.version:
        return str(build.version)
    try:
        return str(update_mod.installed_version() or "")
    except Exception:  # noqa: BLE001 — a version readout must never refuse the op
        return ""


def origin_target() -> dict[str, str]:
    """This device's own build stamp — the ONLY target identity that travels.

    ``{"version", "source_ref"}`` (``update.BuildStamp``'s two halves, the same
    shape ``relay.build_stamp`` announces in the hello). ``source_ref`` is
    identity and provenance; it is never fetched by the member (RU §2 rule 4).
    """
    from local_operator import update as update_mod

    version = ""
    ref = ""
    try:
        version = str(update_mod.installed_version() or "")
    except Exception:  # noqa: BLE001 — see current_member_version
        version = ""
    try:
        ref = str(update_mod.source_ref() or "")
    except Exception:  # noqa: BLE001
        ref = ""
    return {"version": version, "source_ref": ref}


def build_target_from_stamp(stamp: Mapping[str, Any] | None) -> dict[str, str]:
    """The target object for a frame, from a build-stamp mapping.

    Kept separate from :func:`origin_target` so a test can name a target without
    touching this device's own install, and so the frame-shape rule (two string
    keys, nothing else) has one implementation both callers share.
    """
    source = stamp if isinstance(stamp, Mapping) else {}
    return {
        "version": str(source.get("version") or "")[:40],
        "source_ref": str(source.get("source_ref") or "")[:80],
    }


def check_published(version: str, *, client: Any | None = None) -> tuple[bool, str]:
    """Whether ``version`` is resolvable on THIS device's own channel.

    Returns ``(attempt, why)``: ``attempt`` is ``False`` ONLY when the channel
    POSITIVELY says this exact version does not exist — the index answers 404 for
    its per-version document — and then ``why`` is the reason clause. Every other
    answer is ``(True, "")``: a 5xx, a transport error, an unreadable body. That
    asymmetry is deliberate and is the whole shape of this function: an
    INCONCLUSIVE read is not "absent", the installer is the final authority on
    what it can resolve, and its own tail is what the failed receipt carries. A
    refusal keyed on an inconclusive read would make the op depend on a second,
    weaker read of the same network the installer is about to use anyway.

    THE CHECK IS ON THE STATUS CODE, not on a reason string: the updater's own
    ``_pypi_release_document`` folds "not served" and "unreachable" into prose
    and was never the right reader for a decision (agent review round 1, m3).
    ``client`` is the test seam (an httpx-like object with ``get``); the real
    import stays function-local, like the updater's, so this module's import
    surface stays stdlib-only.
    """
    from local_operator import update as update_mod

    url = update_mod.PYPI_VERSION_JSON_URL.format(version=version)
    try:
        if client is None:
            import httpx

            response = httpx.get(url, timeout=PUBLISHED_CHECK_TIMEOUT_S)
        else:
            response = client.get(url, timeout=PUBLISHED_CHECK_TIMEOUT_S)
    except Exception:  # noqa: BLE001 — inconclusive means "attempt it"
        return True, ""
    if int(getattr(response, "status_code", 0)) == 404:
        return False, "the member's channel does not serve this version"
    return True, ""


def install_build(
    version: str, *, runner: Callable[[list[str], dict[str, str]], int] | None = None
) -> str:
    """Install ``version``; returns the receipt's ``method``. THE ONLY CHOICE OF SHAPE.

    TWO SHAPES, in the design's order (RU §4.2 step 3):

    - THE GENERATION ROUTE FIRST — ``install_into_generation``, the pinned-PyPI,
      verified-tree, atomic-pointer install for an exact version: the same
      function the onboarding lane's ``lop update --to <version>`` routes
      through. (§8.1's verb is not on main — verified 2026-10-11 — so the pass
      calls the function directly; the effect on a repo-less member is exactly
      the lane's pin mechanics. Stated in the PR.) ``source=None`` is what
      makes it a PyPI resolution; no source directory, no ref, ever.

    - THE CLASSIC IN-PLACE FALLBACK LAST, and only where the generation layout
      cannot run at all (``perform_upgrade``'s own degradation boundary — the
      caller reaches here on such a member). An in-place rewrite, safe here
      because the drain succeeded moments before (RU §4.2 step 3: "*only*
      then") — and disclosed as ``METHOD_CLASSIC`` on the receipt, so an
      operator can tell the two shapes apart afterwards.

    ``runner`` is the test seam both shapes take (it receives ``(argv, env)``
    because the environment IS the mechanism under test for the generation
    route; the classic route passes the inherited environment).
    """
    from local_operator import update as update_mod

    if install_shape() == METHOD_GENERATION:
        _install_generation(version, runner=runner, update_mod=update_mod)
        return METHOD_GENERATION
    _install_classic(version, runner=runner, update_mod=update_mod)
    return METHOD_CLASSIC


def install_shape() -> str:
    """Which shape :func:`install_build` takes — the ONE condition, named.

    THE FAILURE HANDLER NEEDS THE SAME LABEL THE INSTALL WOULD HAVE DISCLOSED
    (agent review round 1, m2): a failed classic-arm install used to be
    labelled ``uv-tool-generation`` by a hardcoded literal, so the one shape
    whose disclosure matters most — the in-place rewrite that can cut a turn —
    was misnamed exactly when it went wrong.
    """
    return METHOD_GENERATION if generation_layout_supported() else METHOD_CLASSIC


class _CapturedInstall:
    """Run one installer argv, KEEPING its output for a failed receipt (QA Q-3).

    ``update._run_installer_env`` inherits this process's stdio, so uv's own
    words on a failed generation install went to the relay's log and the
    receipt carried only the updater's ``installer exited N``. The decision
    (agent review round 1, Q-3) was to capture, in the family's
    ``onboard._install_failure_detail`` discipline: the tail is flattened,
    head-kept and bounded, so a failed receipt can read ``installer exited 1 —
    installer output: … no space left on device``.

    An injected ``runner`` (tests) replaces this object entirely, exactly as it
    replaces ``_run_installer_env`` upstream.
    """

    def __init__(self) -> None:
        self.output = ""

    def __call__(self, argv: list[str], env: dict[str, str]) -> int:
        from local_operator.install_transaction import run_installer

        code = run_installer(argv, env=env, on_output=self._remember_output)
        if code == 0:
            self.output = ""
        return code

    def _remember_output(self, output: str) -> None:
        self.output = _flatten_output(output)


def _flatten_output(*chunks: str) -> str:
    """Flatten installer output and bound it, the failure-tail family's shape.

    Whitespace collapses to single spaces (a receipt line is ONE line) and the
    tail is head-kept when long — the head names the resolver or the errno,
    which is the part a reader needs (``onboard._install_failure_detail``).
    """
    flat = " ".join(" ".join((chunk or "").split()) for chunk in chunks).strip()
    if len(flat) > 300:
        flat = flat[:300].rstrip() + "…"
    return flat


def _install_generation(version: str, *, runner: Any, update_mod: Any) -> None:
    """``install_into_generation`` plus its retention tail and output capture.

    THE PRUNE IS NOT OPTIONAL DECORATION (agent review round 1, M1):
    ``install_into_generation``'s own docstring says "PRUNING IS THE CALLER'S",
    every other generation install prunes immediately after it
    (``perform_upgrade``, the CLI tail), and a member the mesh rolls is one
    that gets re-rolled — without this it grows one whole venv per rollout,
    unbounded (``DEFAULT_KEEP_GENERATIONS`` = 2 exists for exactly this). Run
    best-effort, exactly as ``perform_upgrade`` runs it: a failed prune may not
    fail a settled install, it only logs its own line.

    SEQUENCED WITH #2181 (``fix/prune-running-generation``, MERGED 2026-10-10):
    that fix is what makes the eviction pick safe for the calling process's own
    tree, and this branch is rebased on it; the call shape below is
    ``perform_upgrade``'s own (``prune_generations(referenced=…,
    actor=ACTOR_UPGRADE)`` plus the notice lines), so the guard is the merged
    one rather than a second-guess here.

    ``_CapturedInstall`` carries uv's output onto the failure path (QA Q-3);
    an injected runner replaces it, as upstream.
    """
    capture = _CapturedInstall()
    try:
        update_mod.install_into_generation(source=None, version=version, runner=runner or capture)
    except Exception as exc:  # noqa: BLE001 — re-raised with the installer's own tail
        if capture.output:
            raise update_mod.UpdateError(
                f"{' '.join(str(exc).split())} — installer output: {capture.output}"
            ) from exc
        raise
    try:
        plan = update_mod.prune_generations(
            referenced=update_mod.referenced_install_roots(), actor=update_mod.ACTOR_UPGRADE
        )
    except Exception:  # noqa: BLE001 — see the docstring
        logger.debug("the generation prune after the install did not run", exc_info=True)
    else:
        for line in update_mod.prune_notice_lines(plan):
            logger.info("%s", line)


def _install_classic(
    version: str,
    *,
    runner: Callable[[list[str], dict[str, str]], int] | None,
    update_mod: Any,
) -> None:
    """The classic uv-tool in-place reinstall, with ``perform_upgrade``'s own discipline.

    MIRRORS ``update.perform_upgrade``'s classic branch step for step, because
    that branch owns the one hazard this shape commits ("THE IN-PLACE INSTALL IS
    THE ONE HAZARD THIS FUNCTION STILL COMMITS"): the tree the running fleet
    imports from is rewritten where it stands, so runtimes can die mid-turn and
    cannot record anything themselves. Hence the same attestation pair —
    ``note_doomed_runtimes`` staged IMMEDIATELY BEFORE the installer runs (after
    the fact, the party that could say what happened is the party that was
    killed), withdrawn on both failure shapes so the marker cannot narrate a
    later death as this install's doing. The marker (``.lop-source``) is written
    best-effort after exit 0, exactly as there.

    Unlike the generation route this touches the LIVE tree, which is why it
    exists only on members where the layout cannot run, and only under the
    drain the caller has already proven.

    THE DRAIN IS A PROBE, NOT A FENCE (agent review round 1, m4): a turn that
    STARTS during the in-place rewrite can still be cut mid-turn — the one
    hazard the generation layout exists to remove. That residual is accepted
    only for members where the layout cannot run at all (RU §8.4 names the
    platform out of scope) and is named in the PR; the attestation pair above
    is what keeps a death recorded rather than silent.
    """
    argv = update_mod.installer_argv(update_mod.InstallKind.UV_TOOL, version=version)
    root = Path(sys.prefix)
    attested = update_mod.note_doomed_runtimes(
        root, mechanism=update_mod.MECHANISM_IN_PLACE_INSTALL, actor=update_mod.ACTOR_UPGRADE
    )
    output = ""
    try:
        if runner is not None:
            code = runner(argv, dict(os.environ))
        else:
            # Capture, where ``_run_installer_env`` inherits stdio (QA Q-3):
            # the receipt's tail must carry uv's own words, not just the exit
            # status the updater layers over them.
            proc = subprocess.run(argv, check=False, capture_output=True, text=True)
            code = int(proc.returncode)
            if code != 0:
                output = _flatten_output(proc.stdout, proc.stderr)
    except BaseException:
        update_mod.withdraw_involuntary_stops(
            attested, mechanism=update_mod.MECHANISM_IN_PLACE_INSTALL
        )
        raise
    if code != 0:
        update_mod.withdraw_involuntary_stops(
            attested, mechanism=update_mod.MECHANISM_IN_PLACE_INSTALL
        )
        tail = f"installer exited {code}"
        if output:
            tail += f" — installer output: {output}"
        raise update_mod.UpdateError(tail)
    if not update_mod.write_source_marker(root, version=version):
        # Deliberately not fatal, the same call ``perform_upgrade`` makes: the
        # install SUCCEEDED, and a failed marker only costs accuracy in the
        # labels (it logs there for the same reason).
        logger.warning("updated to %s but could not record it in %s/.lop-source", version, root)


# ---------------------------------------------------------------------------
# Target validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UpdateTarget:
    """The frame's target identity: a version, and the origin's ref as provenance."""

    version: str
    source_ref: str = ""


def target_from_frame(frame: Mapping[str, Any]) -> UpdateTarget:
    """The target a ``net_update`` frame carries, or a protocol refusal.

    Shape-strict on purpose (the ``peer_int`` discipline): a frame whose target
    is missing or not an object is a malformed frame, not an update, and the
    refusal says which half was wrong. ``source_ref`` is capped like every other
    bounded wire string and may be empty — it is provenance, never a fetch
    instruction.
    """
    raw = frame.get("target")
    if not isinstance(raw, Mapping):
        raise MeshRefusal("bad_request", "a net_update frame must carry a target object")
    target = build_target_from_stamp(raw)
    if not target["version"]:
        raise MeshRefusal(
            "bad_request", "the update target named no version, so nothing was attempted"
        )
    return UpdateTarget(version=target["version"], source_ref=target["source_ref"])


def _version_tuple(value: str) -> tuple[int, int, int] | None:
    from local_operator import update as update_mod

    return update_mod.parse_version(value)


# ---------------------------------------------------------------------------
# The member's own facts: busy, lock
# ---------------------------------------------------------------------------


def busy_sessions(root: Path) -> tuple[int, str]:
    """``(count, detail)`` for the member's live BUSY sessions — its own registry.

    THE PREDICATE IS THE FLEET TOOL'S OWN (RU §4.3, note Q1): a live session
    record with ``busy`` set, which is the bit ``RuntimeServer._publish_busy``
    writes from ``is_conversationally_active()`` — deliberately the narrow
    predicate, so the rolling drain and the operator's measured local discipline
    cannot disagree about what "idle" means.

    Reader mode (``reap=False``, ``check_zombie=False``): this is a probe, not
    the sweep that owns that namespace. An unreadable registry returns
    ``(-1, why)`` — "cannot prove idle", which the caller refuses in the same
    shape as busy; the unsafe direction of a failed read is an install over a
    running turn.
    """
    try:
        from local_operator.session.runtime import registry

        rows = registry.scan(root, reap=False, check_zombie=False)
    except Exception as exc:  # noqa: BLE001 — see the docstring: fail toward "not idle"
        return -1, (
            f"this device could not read its own session registry ({exc}), "
            "so nothing was touched"
        )
    busy = [record for record, state in rows if state == "live" and bool(record.busy)]
    if not busy:
        return 0, ""
    return len(busy), f"{len(busy)} session(s) are busy"


@contextmanager
def update_lock(root: Path, target: UpdateTarget) -> Iterator[None]:
    """Mesh admission uses the SAME install root as CLI/TUI, not config root.

    Preparation and service convergence are deliberately outside this short
    transaction; the common installer owns candidate registration and fresh
    promotion. A config-profile lock cannot exclude another profile's writer.
    """
    from local_operator import install_transaction, update

    try:
        with install_transaction.transaction(
            update.stable_root(), operation=f"mesh admission {target.version}", timeout=0
        ):
            yield
    except install_transaction.LockUnsupported as exc:
        raise MeshRefusal("update_lock_unsupported", str(exc)) from exc
    except install_transaction.InstallTransactionError as exc:
        raise MeshRefusal("update_in_progress", str(exc)) from exc


# ---------------------------------------------------------------------------
# The member executor (S1's call, S2's snapshot/drain/install/roll/re-engage)
# ---------------------------------------------------------------------------


def live_session_targets(root: Path) -> list[Any]:
    """The member's LIVE set at the start of a pass — RU §4.2 step 1, the snapshot.

    The registry is the same authority the drain trusts (§4.3) and this is the
    same selector the CLI's rotation sweep uses (``control._rotation_targets``,
    private access the way every cross-module seam here is): live AND wedged
    records, the caller's own excluded. Best-effort by design: a registry that
    cannot be read yields an empty set, because the DRAIN's own fresh read
    (under the lock, moments before the install) is the gate that refuses on
    unreadability — this read only sizes counts and picks re-engage candidates,
    and one fault must not fail the pass twice.
    """
    from local_operator.session.runtime import control

    try:
        return control._rotation_targets(root, own_pid=os.getpid())  # noqa: SLF001
    except Exception:  # noqa: BLE001 — see the docstring: the drain is the gate
        return []


def refresh_runtimes(root: Path, *, budget: float | None = None) -> tuple[list[Any], str]:
    """Ask every live runtime to move to the build on disk — RU §4.2 step 4. NEVER raises.

    ``control.refresh_all`` IS ``lop refresh --all``: sequential asks, each
    runtime judging its own ``may_refresh`` and answering with its own
    sentence — nothing is signalled, nothing is stopped, and a busy runtime
    answers ``busy`` and keeps its intact tree.

    THE SWEEP SIZES ON AND DIALLS THE SAME SET (agent review round 2, F1).
    ``live_session_targets`` is the ask set AND the sizer, read here at sweep
    time; ``only_pids`` is ALWAYS passed, so ``refresh_all`` cannot dial a
    record this call did not size for — the earlier shape passed ``None`` in
    the ≤-askable branch, letting its re-read dial an unbounded live set at up
    to 10 s per dial (the overrun this budget exists to close, reachable
    exactly where the sizing read is best-effort). A runtime that STARTS
    mid-pass is deliberately not asked: it loaded the build already on disk,
    so there is nothing to move it to.

    THE ARITHMETIC (QA round 1, Q-2 + agent review round 1, M2): per-call is
    the budget divided across the live set, floored at
    :data:`REFRESH_MIN_PER_CALL_S` and capped at the control module's default;
    the dialled set is capped at ``ceil(nominal / per_call)`` targets, so the
    sequential total is bounded by ``nominal`` plus at most ONE per-call
    timeout — for ANY live count, including the under-read case the old shape
    left open. ``nominal`` is the remaining slow-op time when the caller
    clamped it, the nominal :data:`REFRESH_SWEEP_DEADLINE_S` otherwise.

    Returns ``(outcomes, note)``; a sweep that could not run at all returns no
    outcomes and a note, never an exception — the install already succeeded,
    the runtimes also retire on their own watch, and the receipt is where the
    degradation is said (RU §4.2 step 4: the receipt says how many, honestly).
    """
    from local_operator.session.runtime import control

    nominal = REFRESH_SWEEP_DEADLINE_S if budget is None else min(REFRESH_SWEEP_DEADLINE_S, budget)
    if nominal <= 0:
        return [], "no time remained for the runtime sweep"
    live = live_session_targets(root)
    slots = max(len(live), 1)
    per_call = min(max(nominal / slots, REFRESH_MIN_PER_CALL_S), control.DEFAULT_REFRESH_TIMEOUT_S)
    askable = max(1, math.ceil(nominal / per_call - 1e-9))
    only = frozenset(int(getattr(t, "pid", 0) or 0) for t in live[:askable])
    note = ""
    if len(live) > askable:
        note = (
            f"the live set ({len(live)}) exceeds the sweep budget; {len(live) - askable} not asked"
        )
    try:
        outcomes = asyncio.run(
            control.refresh_all(timeout_s=per_call, own_pid=os.getpid(), only_pids=only, _root=root)
        )
    except Exception as exc:  # noqa: BLE001 — degrade to counts, never fail the pass
        return [], f"the runtime sweep could not run ({exc})"
    return list(outcomes), note


#: The sweep methods that mean "this session's runtime is leaving or gone",
#: and so the re-engage candidates (RU §4.2 step 5): a retirement just
#: committed (``moved``/``draining``), or nothing a successor can reuse answered
#: (``unreachable``/``unsupported`` — a dead or pre-op runtime). ``busy``,
#: ``kept``, ``current`` and ``unsettled`` keep a working runtime and are left
#: alone. Deliberately a closure over the control module's own vocabulary —
#: its ``RefreshOutcome`` docstring is the source of these names.
_MOVED_METHODS = frozenset({"moved", "draining"})
_DISPLACED_METHODS = frozenset({"moved", "draining", "unreachable", "unsupported"})
#: The methods that mean a runtime is still there (agent review round 1, m1):
#: ``kept`` in the receipt counts THESE, not ``len(outcomes) - moved`` — the old
#: subtraction counted displaced sessions twice ("1 kept, 1 re-engaged" for one
#: session) and called the dead ones kept.
_KEPT_METHODS = frozenset({"busy", "current", "kept", "unsettled"})


def count_pass_outcomes(outcomes: Sequence[Any]) -> tuple[int, int]:
    """``(moved, kept)`` for the receipt — the two numbers the design freezes.

    ``moved`` is a retirement actually committed; ``kept`` is a working runtime
    left alone. Anything else (``unreachable``/``unsupported``) is neither and
    rides the receipt's own "not answerable" count — never the kept count.
    """
    moved = sum(1 for o in outcomes if str(getattr(o, "method", "")) in _MOVED_METHODS)
    kept = sum(1 for o in outcomes if str(getattr(o, "method", "")) in _KEPT_METHODS)
    return moved, kept


def reengage_displaced(
    root: Path,
    records: Sequence[Any],
    outcomes: Sequence[Any],
    *,
    budget: float | None = None,
) -> tuple[int, str]:
    """Ensure a runtime exists for displaced sessions — RU §4.2 step 5. NEVER raises.

    THE HALF "NOTHING ELSE PERFORMS FOR UNWATCHED SESSIONS": an attached viewer
    re-engages its session when its runtime retires, but a session nobody is
    watching keeps no successor — the sweep (§4.2 step 4) has just asked its
    runtime to leave, so this step walks the displaced set and engages one.
    The engage is the relay's own speculative shape (``WarmErrand``: "start the
    runtime, deliver nothing") — nothing is claimed about a death that did not
    happen, and no invented message shape is posted into a session's inbox.

    THE DESIGN'S "TRUTHFUL ONE-LINE NUDGE" rides the receipt and the member's
    audit (``<n> re-engaged``), not the session's store: v1 delivers the
    sentence where an operator reads pass outcomes, and the open question —
    whether the line should also land in the session's inbox or attention row —
    is named in the PR rather than invented silently here.

    ``budget`` is the remaining time the caller's clamp left (None = the
    nominal :data:`REENGAGE_BUDGET_S`); the loop's total is the SMALLER of
    that and the nominal (agent review round 2, F2), and it is checked before
    each engage AND passed into the engage as its own deadline too, so the
    total cannot exceed it by one full engage (agent review round 1, M2's
    tail note).
    """
    by_id = {str(getattr(record, "session_id", "")): record for record in records}
    engaged = 0
    failed = 0
    skipped = 0
    # THE NOMINAL TOTAL IS A CAP, NOT JUST A DEFAULT (agent review round 2,
    # F2): a clotted slow-op remainder must not let the loop — and the pass
    # lock behind it — run past the documented budget, so the caller's
    # remaining time can only TIGHTEN it.
    total = REENGAGE_BUDGET_S if budget is None else min(REENGAGE_BUDGET_S, max(0.0, budget))
    deadline = time.monotonic() + total
    for outcome in outcomes:
        if str(getattr(outcome, "method", "")) not in _DISPLACED_METHODS:
            continue
        left = deadline - time.monotonic()
        # One second, not zero: a sub-second engage cannot outlive the dial it
        # makes, so it is skipped and said rather than started and counted as
        # a failure.
        if left < 1.0:
            skipped += 1
            continue
        record = by_id.get(str(getattr(outcome, "session_id", "")))
        if record is None:
            failed += 1
            continue
        reason = _engage_successor(record, root=root, deadline_s=min(REENGAGE_DEADLINE_S, left))
        if reason:
            failed += 1
        else:
            engaged += 1
    notes = []
    if failed:
        notes.append(f"{failed} session(s) could not be re-engaged")
    if skipped:
        notes.append(f"{skipped} not re-engaged inside the budget")
    return engaged, "; ".join(notes)


def _engage_successor(record: Any, *, root: Path, deadline_s: float) -> str:
    """Engage one successor runtime for ``record``; return "" or the failure reason.

    The call shape is the relay's own speculative engage (``WarmErrand``:
    "start the runtime, deliver nothing") under the caller's deadline, and the
    session must hold its directory on THIS device first — the same pre-check
    ``relay._engage_locally`` makes for its create path, because an id with no
    directory here is a registry read that raced, not a session to spawn for.
    NO MODEL SAMPLE RIDES THE ERRAND (agent review round 1, n3: the record
    publishes a label, not the birth sample ``relay`` passes), so the
    successor boots on the device default. A failure is reported to the caller,
    never raised: one session that cannot come back must not fail a pass whose
    install already succeeded.
    """
    from local_operator.session.runtime.launch import WarmErrand, engage_runtime

    session_id = str(record.session_id)
    if not (root / "sessions" / session_id).is_dir():
        return f"this device does not hold a session {session_id}"
    try:
        asyncio.run(
            engage_runtime(
                session_id,
                str(record.cwd or Path.home()),
                WarmErrand(),
                config_dir=root,
                deadline_s=deadline_s,
            )
        )
    except (TimeoutError, RuntimeError, ConnectionError, OSError) as exc:
        return " ".join(str(exc).split())[:200]
    return ""


def roll_services_after_reply() -> None:
    """Kick the supervised daemons and THIS relay onto the new build — AFTER the reply.

    the note's Q4 recommendation, recorded — WITH ITS PROBABILISTIC HALF
    NAMED (agent review round 1, m3): the receipt VALUE is composed before
    this call is made and the stage runs on a daemon thread, so this call
    returns immediately and the caller's write of the frame is not behind an
    interpreter startup. The WRITE ITSELF belongs to the slow-dispatch
    machinery above this module, so a kickstart still races it by the child's
    startup time — a real margin (fork + interpreter start before the stage's
    relay step can land), not an ordering; the losing race is covered by the
    origin's own re-ask (``already_on_target``) and, once S3 lands, by the
    rollout record's re-probe. Never raises — the upgrade already succeeded,
    so the worst outcome here is a warning (the stage's own contract).
    """

    def _run() -> None:
        try:
            from local_operator import update as update_mod

            update_mod.refresh_service_daemons_after_upgrade()
        except Exception:  # noqa: BLE001 — see the docstring: never fail a settled pass
            logger.warning("the services roll after the update did not run", exc_info=True)

    threading.Thread(target=_run, name="mesh-update-services-roll", daemon=True).start()


def _refused(code: str, sentence: str, *, current: str = "") -> dict[str, Any]:
    """The reply's ``refused`` arm, in the ONE shape RU §3 freezes."""
    return {
        "state": "refused",
        "code": code,
        "reason": sentence,
        "method": "",
        "version": current,
        "sessions": {"moved": 0, "kept": 0},
        "updated_at": time.time(),
    }


def _audit_member(
    server: "RelayServer",
    link: "PeerLink | None",
    event: str,
    *,
    target: UpdateTarget,
    outcome: str = "ok",
    cause: str = "",
    rollout: str = "",
    detail: Mapping[str, Any] | None = None,
) -> None:
    """One member-side row. Best-effort: a record is not a gate (the mcpdefs rule)."""
    try:
        from local_operator.network.audit import AuditEvent

        fields: dict[str, Any] = {
            "rollout": rollout,
            "target": target.version,
        }
        if detail:
            fields.update({k: v for k, v in detail.items()})
        server.audit.record(
            AuditEvent(
                event=event,
                actor=link.device_id if link is not None else "self",
                subject=str(getattr(link, "network_id", "") or ""),
                outcome=outcome,
                network_id=str(getattr(link, "network_id", "") or ""),
                epoch=getattr(link, "epoch", None),
                cause=cause,
                detail=fields,
            )
        )
    except Exception:  # noqa: BLE001 — see the docstring
        pass


def member_execute(
    server: "RelayServer",
    link: "PeerLink | None",
    frame: Mapping[str, Any],
    *,
    runner: Callable[[list[str], dict[str, str]], int] | None = None,
) -> dict[str, Any]:
    """The ``net_update`` handler's body: apply-or-answer, one fresh probe.

    ORDER IS THE DESIGN (RU §3): target validation first (the no-action and
    refusal outcomes need no idle), THEN the fresh busy probe — so every reply
    state has exactly one producer and a ``busy`` answer costs milliseconds.
    """
    target = target_from_frame(frame)
    # THE ROLLOUT ID RIDES THE FRAME (S3a): the member never acts on it, its
    # rows just name the rollout they belong to (see ``update_peer``'s note).
    rollout = str(frame.get("rollout") or "")[:80]
    _audit_member(
        server,
        link,
        "update_requested",
        target=target,
        rollout=rollout,
        detail={"capability": "update"},
    )

    try:
        return _execute_validated(server, link, target, rollout=rollout, runner=runner)
    except MeshRefusal as refusal:
        _audit_member(
            server,
            link,
            "update_refused",
            target=target,
            rollout=rollout,
            outcome="refused",
            cause="policy",
            detail={"code": refusal.code},
        )
        return _refused(refusal.code, refusal.sentence)
    except Exception as exc:  # noqa: BLE001 — a handler bug must not close the link
        return {
            "state": "failed",
            "code": "",
            "reason": f"the update failed on this device: {exc}",
            "method": "",
            "version": "",
            "sessions": {"moved": 0, "kept": 0},
            "updated_at": time.time(),
        }


def _execute_validated(
    server: "RelayServer",
    link: "PeerLink | None",
    target: UpdateTarget,
    *,
    rollout: str = "",
    runner: Callable[[list[str], dict[str, str]], int] | None,
) -> dict[str, Any]:
    """Validation, the never-force gate, the install, the tail, and the receipt.

    THE LOCK COVERS THE WHOLE PASS, the tail included (agent review round 1,
    m5): "one at a time on the member" (RU §4.4) is not just the install.
    """
    from local_operator import update as update_mod

    kind = detect_install_kind()
    if kind is update_mod.InstallKind.EDITABLE:
        raise MeshRefusal(
            "editable_install",
            "this device runs from a development tree; updates are out of scope there "
            "(`lop update` by hand is the path)",
        )
    if not generation_layout_supported() and kind is not update_mod.InstallKind.UV_TOOL:
        # A non-uv-tool install without the generation layout keeps S1's refusal
        # verbatim; the CARVE-OUT is the uv-tool member, which takes the
        # classic in-place arm inside ``install_build`` (RU §4.2 step 3's
        # last resort; ``perform_upgrade``'s own degradation boundary).
        raise MeshRefusal(
            "editable_install",
            "this device runs from a development tree; updates are out of scope there "
            "(`lop update` by hand is the path)",
        )

    target_version = target.version
    if _version_tuple(target_version) is None:
        raise MeshRefusal(
            "target_not_published",
            f"the target {target_version!r} is not a version this device can resolve; "
            "nothing was installed",
        )

    current = current_member_version()
    if current and _version_tuple(current) is not None:
        mine = _version_tuple(current)
        theirs = _version_tuple(target_version)
        assert mine is not None and theirs is not None  # narrowed above
        if theirs == mine:
            return {
                "state": "already_on_target",
                "code": "",
                "reason": f"this device already runs {current}",
                "method": "",
                "version": current,
                "sessions": {"moved": 0, "kept": 0},
                "updated_at": time.time(),
            }
        if theirs < mine:
            # A SKIP, NOT A FAILURE (RU §3's reply map): ``ahead_of_target`` is a
            # STATE — the member was done by hand past the fleet and is left
            # alone until the origin catches up — so it answers in its own state
            # rather than through the ``refused`` arm.
            #
            # AND IT IS STILL A ROW ON THE MEMBER (agent review round 1, m4): the
            # design's audit table lists "executor refusal by rule (ahead, not
            # published, editable, in progress)" as ``update_refused``, and the
            # member's own log is the only place an operator can see WHY a
            # rollout skipped this device. ``already_on_target`` writes nothing —
            # it is the idempotent no-op, and a row per re-ask would be exactly
            # the per-poll log the taxonomy forbids.
            _audit_member(
                server,
                link,
                "update_refused",
                target=target,
                outcome="refused",
                cause="policy",
                detail={"code": "ahead_of_target"},
            )
            return {
                "state": "ahead_of_target",
                "code": "",
                "reason": (
                    f"this device runs {current} and the target is {target_version} — an "
                    "update never moves backwards"
                ),
                "method": "",
                "version": current,
                "sessions": {"moved": 0, "kept": 0},
                "updated_at": time.time(),
            }

    # TARGET VALIDATION RUNS FIRST, BEFORE ANY IDLE GATE, and that is RU §3's
    # own order ("the no-action and refusal outcomes need no idle"). The channel
    # check belongs to that validation, and it sits HERE rather than after the
    # busy probe on purpose: a target this device's channel cannot serve is a
    # TERMINAL refusal the origin should learn on its first ask, not a member
    # that keeps answering ``busy`` for as long as it happens to be working. It
    # costs one bounded read (``PUBLISHED_CHECK_TIMEOUT_S``) on every ask that
    # reaches it — including a busy member's re-ask (agent review round 2, m7:
    # "only on a path that was going to install" was true only before the
    # reorder). A terminal refusal on the first ask is worth that read; it is
    # the cost of the design's order, not an oversight.
    published, why = check_published(target_version)
    if not published:
        raise MeshRefusal(
            "target_not_published",
            f"the target {target_version} is not resolvable on this device's channel "
            f"({why}); nothing was installed",
        )

    root = Path(server.root)
    # RU §4.2 step 1, BEFORE the drain: the snapshot the tail's counting and
    # re-engage candidates come from (agent review n1 — the call used to sit
    # after the install while its docstring said "at the start of a pass").
    snapshot = live_session_targets(root)
    with update_lock(root, target):
        count, detail = busy_sessions(root)
        if count != 0:
            reason = (
                f"{detail}; the update waits — nothing has been touched"
                if count > 0
                else f"{detail} — the update waits rather than risk a running turn"
            )
            return {
                "state": "busy",
                "code": "",
                "reason": reason,
                "method": "",
                "version": current,
                "sessions": {"moved": 0, "kept": max(count, 0)},
                "updated_at": time.time(),
            }
    _audit_member(server, link, "update_started", target=target, rollout=rollout)
    try:
        method = install_build(target_version, runner=runner)
    except Exception as exc:  # noqa: BLE001 — the installer's own tail is the receipt
        tail = " ".join(str(exc).split())
        if len(tail) > 300:
            tail = tail[:300].rstrip() + "…"
        _audit_member(
            server,
            link,
            "update_failed",
            target=target,
            rollout=rollout,
            outcome="failed",
            cause="internal",
            detail={"method": install_shape()},
        )
        return {
            "state": "failed",
            "code": "",
            "reason": tail or "the install did not finish; nothing was made current",
            "method": install_shape(),
            "version": "",
            "sessions": {"moved": 0, "kept": 0},
            "updated_at": time.time(),
        }

    version_now = current_member_version()
    # Another registered candidate may complete before or during the unlocked
    # convergence tail. A newer current does not undo this successful promotion.
    from local_operator.update import parse_version

    observed = parse_version(version_now)
    requested = parse_version(target_version)
    if version_now != target_version and not (observed and requested and observed > requested):
        _audit_member(
            server,
            link,
            "update_failed",
            target=target,
            rollout=rollout,
            outcome="failed",
            cause="internal",
            detail={"method": method},
        )
        return {
            "state": "failed",
            "code": "",
            "reason": (
                f"the install finished but this device still reports "
                f"{version_now or 'nothing'}; nothing was advertised"
            ),
            "method": method,
            "version": version_now,
            "sessions": {"moved": 0, "kept": 0},
            "updated_at": time.time(),
        }
    _audit_member(
        server,
        link,
        "update_completed",
        target=target,
        rollout=rollout,
        detail={"method": method, "from": current, "version": target_version},
    )
    # Convergence is outside the installation transaction. Reconcile current
    # AFTER it: the promoted generation and this relay's serving image are not
    # necessarily what a fresh process will load when the receipt is composed.
    from local_operator.network import relay

    remaining = relay.slow_op_remaining_s()
    # None outside a slow op (direct calls, cells): the nominal budgets
    # apply. Inside one, the tail is clamped to what the requester is
    # still listening for, less the reserve for the receipt write —
    # ``_dispatch_slow`` DROPS a reply that lands after the deadline
    # (agent review round 1, M2).
    tail_budget = None if remaining is None else max(0.0, remaining - TAIL_RESERVE_S)
    tail_started = time.monotonic()
    outcomes, sweep_note = refresh_runtimes(root, budget=tail_budget)
    moved, kept = count_pass_outcomes(outcomes)
    left = (
        None if tail_budget is None else max(0.0, tail_budget - (time.monotonic() - tail_started))
    )
    engaged, engage_note = reengage_displaced(root, snapshot, outcomes, budget=left)
    other = len(outcomes) - moved - kept
    now_current = current_member_version()
    serving = origin_target().get("version", "")
    reason = f"this device moved from {current or 'an unknown build'} to {target_version}"
    if now_current != target_version:
        reason = (
            f"this pass promoted {target_version}; current is now "
            f"{now_current or 'unconfirmed'}; serving relay is {serving or 'unconfirmed'}"
        )
    if outcomes:
        reason += f"; {moved} session(s) retired for the new build, {kept} kept"
        if other > 0:
            reason += f", {other} not answerable"
    if engaged:
        reason += f", {engaged} re-engaged"
    for note in (sweep_note, engage_note):
        if note:
            reason += f" ({note})"
    receipt = {
        "state": "done",
        "code": "",
        "reason": reason,
        "method": method,
        "version": now_current,
        "promoted": target_version,
        "now_current": now_current,
        "serving": serving,
        "sessions": {"moved": moved, "kept": kept},
        "updated_at": time.time(),
    }
    # THE RECEIPT VALUE IS COMPOSED BEFORE THE ROLL IS STARTED (agent review
    # round 1, m3): the WRITE of the frame belongs to the slow-dispatch
    # machinery above this module, so what this ordering buys is that the roll
    # starts only after nothing is left to compute — and the thread's own
    # startup is the margin for the kickstart/`done`-frame race on the wire.
    # See `roll_services_after_reply`: a margin, not an ordering.
    roll_services_after_reply()
    return receipt


# ---------------------------------------------------------------------------
# The origin's single-peer half
# ---------------------------------------------------------------------------


def no_grant_remedy(*, label: str, network: str, requester: str) -> str:
    """RU §2's ``update_not_granted`` sentence, composed where it is actionable.

    The remedy runs ON THE MEMBER (its operator decides), so the sentence names
    the member and the command to run THERE — including the requester's device
    id, because ``member grant`` takes an id and not a name (the Aida round-2
    finding the mobility sibling records).
    """
    network_part = f" {network}" if network else " <network>"
    return (
        f"{label} has not granted `update` to this device; its operator can grant it "
        f"in the Mesh tab on {label}, or run `lop network member grant{network_part} "
        f"{requester} update` there."
    )


def classify_member_answer(reply: Mapping[str, Any] | None) -> tuple[str, str]:
    """``(code, message)`` for one member answer, as the origin must report it.

    The CODED refusals cross the wire (``wire.error_from``) and pass straight
    through — ``update_in_progress`` and the member's rule refusals are machine
    tokens. A CODELESS refusal is the authoriser's, and the remote peer is
    deliberately not told which guard fired; for THIS op the origin reports it
    as ``refused`` with the member's own sentence, and never guesses
    ``no_grant`` from it (see this module's open question — the design's
    first-row classification needs a signal the wire does not carry today).
    """
    if reply is None:
        return "no_answer", "the member did not answer before the bound"
    if str(reply.get("op")) == "error":
        code = str(reply.get("code") or "")
        message = str(reply.get("message") or "the member refused the update")
        return (code or "refused"), message
    detail = reply.get("detail")
    if isinstance(detail, Mapping):
        # The member's own code when it carried one (a rule refusal rides a
        # normal ack), else the state's name.
        return str(detail.get("code") or detail.get("state") or "failed"), ""
    return "failed", ""


def update_peer(server: "RelayServer", device_id: str, *, rollout: str = "") -> dict[str, Any]:
    """Ask ONE member to move to this device's build. The ``peer_update`` body.

    ``rollout`` is the record id this ask belongs to (S3a): it rides the frame
    so the MEMBER's audit rows can name the rollout their update was part of
    (note §6: "every later row of that rollout repeats ``rollout``", so one
    ``lop network log`` filter reconstructs the whole story). It is prose-free
    provenance, never an instruction: the member audits it, nothing acts on it.

    The origin half of the single-peer verb: dial, check the feature string
    BEFORE the first request (RU §5 — an old member never has to compose a
    refusal it did not ask to make), send the target identity, and report the
    member's own words. It sends NO code, NO ref to fetch, and NO command.

    THE TWO REASONS THIS FUNCTION RENDERS, AND WHY THEY DIFFER (round 3, M2):
    the dial seam's token (the unreachable line) is a wire TOKEN and goes through
    the mesh's one gloss, ``resume.peer_reason_words`` (imported locally, the way
    the network CLI spells it); the member's reply sentence is
    authored-for-a-reader PROSE and is passed through VERBATIM — "report the
    member's own words" is the design's contract (RU §3), and round 3 measured
    the gloss re-wording receipts into false states. ``test_reason_surfaces.py``
    holds both: the gloss call satisfies it for the token, and the member's read
    is a DECLARED entry there (its comment carries the same record).
    """
    from local_operator.resume import peer_reason_words  # the mesh's one gloss

    # The one dial seam, and the reason it returns.
    link, dial_reason = server._ensure_link_with_reason(device_id)  # noqa: SLF001
    label = server._member_name(device_id) or device_id  # noqa: SLF001 — the mesh's own name
    if link is None:
        # THE SEAM'S OWN REASON RIDES THE SENTENCE, IN WORDS, WHEN IT EXISTS
        # (m5 + Q-R3-1): ``no_endpoint``, ``not_a_member`` and the dial's refusal
        # codes are different facts with different remedies, and dropping them
        # left the operator to guess between "that device is off" and "this
        # device was never told where it lives". The TOKEN itself stays the
        # ``--json`` register; the line reads the gloss — and an EMPTY reason
        # stays a MISSING parenthetical (the CI round made it unconditional,
        # which would have rendered the gloss's reachability default onto a line
        # that had no reason to say it).
        suffix = f" ({peer_reason_words(dial_reason)})" if dial_reason else ""
        return {
            "ok": False,
            "code": "unreachable",
            "message": (
                f"{label} is not answering right now, so it was not asked to update{suffix}"
            ),
            "device_id": device_id,
            "name": label,
        }
    if wire.MESH_UPDATE_V1 not in link.capabilities:
        return {
            "ok": False,
            "code": "predates_rolling_updates",
            "message": (
                f"{label} predates mesh rolling updates; update it there by hand "
                f"(`lop update` on {label}), then re-check"
            ),
            "device_id": device_id,
            "name": label,
        }
    target = build_target_from_stamp(origin_target())
    if not target["version"]:
        # NO DEAD SECOND CONDITION HERE (agent review round 1, nit): the branch
        # used to read ``detect_install_kind() is not None and not version``,
        # whose first clause is always true — the honest statement is simply
        # "this device has no version to offer".
        return {
            "ok": False,
            "code": "editable_install",
            "message": (
                "this device has no published version to offer (a development or "
                "unreadable install); members are not asked to follow it"
            ),
            "device_id": device_id,
            "name": label,
        }
    if _version_tuple(target["version"]) is None:
        return {
            "ok": False,
            "code": "target_not_published",
            "message": (
                f"this device's own build reports {target['version']}, which "
                "is not a resolvable target; nothing was sent"
            ),
            "device_id": device_id,
            "name": label,
        }
    try:
        reply = link.request(
            {
                "op": "net_update",
                "req": server._next_relay_req(),  # noqa: SLF001 — the one req counter
                "locality": "remote",
                "target": target,
                "rollout": rollout,
            },
            timeout=server.slow_request_timeout("net_update"),
        )
    except Exception as exc:  # noqa: BLE001 — a dial/probe failure is reported, not raised
        return {
            "ok": False,
            "code": "no_answer",
            "message": f"{label} did not answer ({exc}); ask again when it is reachable",
            "device_id": device_id,
            "name": label,
        }
    if reply is None:
        return {
            "ok": False,
            "code": "no_answer",
            "message": (
                f"{label} stopped answering before it replied; nothing is claimed about "
                "whether its update began — re-run `lop network update` to re-ask it"
            ),
            "device_id": device_id,
            "name": label,
        }
    code, message = classify_member_answer(reply)
    detail_out: dict[str, Any] = {}
    if str(reply.get("op")) == "error":
        if code == "not_authorised":
            # A coded not_authorised (a handler's, or a future coded authoriser
            # frame) is the grant's absence, said in the member's own words plus
            # the remedy that runs there.
            return {
                "ok": False,
                "code": "no_grant",
                "message": no_grant_remedy(
                    label=label,
                    network=_shared_network_name(server, device_id),
                    requester=str(server.identity.device_id),
                ),
                "detail": {"member_sentence": message},
                "device_id": device_id,
                "name": label,
            }
        return {
            "ok": False,
            "code": code,
            "message": message,
            "device_id": device_id,
            "name": label,
        }
    raw_detail = reply.get("detail")
    detail_out = dict(raw_detail) if isinstance(raw_detail, Mapping) else {}
    state = str(detail_out.get("state") or "")
    # THE MEMBER'S OWN CODE TRAVELS, not the state's name. A rule refusal answers
    # a NORMAL ack whose detail carries BOTH halves (``state: refused``,
    # ``code: editable_install``) — collapsing the code to the state here would
    # leave the caller unable to tell ``editable_install`` from
    # ``update_in_progress`` without reading English, which is exactly the
    # branch this register exists to make possible (and which the CLI's lines
    # and a `--json` consumer both read).
    inner_code = str(detail_out.get("code") or "")
    ok = state in ("done", "already_on_target", "ahead_of_target")
    return {
        "ok": ok,
        "code": inner_code or state or "failed",
        "state": state,
        # THE MEMBER'S OWN WORDS, VERBATIM (round 3, M2/Q-R3-2): this read was
        # routed through ``resume.peer_reason_words`` in the CI-round fix, and
        # the gloss RE-WORDED the member's receipts into false states — `uv tool
        # install failed: disk full` lost its leading clause, and `…: E2BIG` /
        # `[Errno 13] Permission denied: '…'` read as "it did not answer" (the
        # member DID answer). The member's updater authors this field as prose
        # for a reader, and the design's contract is "report the member's own
        # words" (RU §3) — so the tail is a PASS-THROUGH, and the guardian
        # (``test_reason_surfaces.py``) carries it as a DECLARED read whose entry
        # records this. The emptiness guard stays: a missing reason is a missing
        # message, never a default sentence.
        "message": str(detail_out.get("reason") or ""),
        "method": str(detail_out.get("method") or ""),
        "version": str(detail_out.get("version") or ""),
        "sessions": detail_out.get("sessions") or {"moved": 0, "kept": 0},
        "device_id": device_id,
        "name": label,
    }


def _shared_network_name(server: "RelayServer", device_id: str) -> str:
    """A network name this device and ``device_id`` share, for a remedy sentence."""
    try:
        from local_operator.network import store

        for record in store.list_networks(server.root):
            if any(member.device_id == device_id for member in record.active_members()):
                return str(record.name or record.network_id)
    except Exception:  # noqa: BLE001 — naming a network is decoration on a remedy
        pass
    return ""


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def make_handler(server: "RelayServer") -> Callable[["PeerLink", dict[str, Any]], dict[str, Any]]:
    """The ``net_update`` peer-op handler for one relay.

    It never issues a request over the link it is serving (``PeerLink.request``
    refuses that): every fact it reads is this device's own, and every act it
    performs is this device's own install.
    """

    def _handle(link: "PeerLink", frame: dict[str, Any]) -> dict[str, Any]:
        return member_execute(server, link, frame)

    return _handle


def local_update_handler(server: "RelayServer") -> Callable[[dict[str, Any]], dict[str, Any]]:
    """The ``peer_update`` local verb: ask ONE peer, named by the caller."""

    def _handle(frame: dict[str, Any]) -> dict[str, Any]:
        peer = str(frame.get("peer") or "")
        if not peer:
            raise MeshRefusal("peer_required", "name a device: `lop network update <peer>`")
        device_id = server._resolve_peer(peer)  # noqa: SLF001 — the one name resolver
        return update_peer(server, device_id)

    return _handle


def install(server: "RelayServer") -> None:
    """Register this slice's ops on ``server`` (relay construction calls this).

    ``net_update`` is SLOW: when the member is idle the handler may run a whole
    install inside its own bounded deadline, far past the 10 s inline budget, so
    it must not run on a link's reader. It starts no thread and registers no
    start hook — the rolling driver is S3's.
    """
    server.register_ops(
        {"net_update": make_handler(server)},
        local_handlers={"peer_update": local_update_handler(server)},
        slow={"net_update": UPDATE_OP_DEADLINE_S},
    )
