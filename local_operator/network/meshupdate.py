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

import fcntl
import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, Mapping

from local_operator.network import wire
from local_operator.network.types import MeshRefusal

if TYPE_CHECKING:  # pragma: no cover - typing only
    from local_operator.network.relay import PeerLink, RelayServer

#: The owner-side deadline for one ``net_update`` request (seconds).
#:
#: The install runs INSIDE the bounded call when the member is idle (RU §3:
#: "a call that observes idle may perform the install inside its own bounded
#: deadline"), and the bound is the same 900 s the onboarding runner gives that
#: step (``onboard.STEP_TIMEOUTS["install"]``) — the one number this repo has
#: already measured for a pinned wheel install on a loaded host.
UPDATE_OP_DEADLINE_S = 900.0

#: The LOCAL control-socket budget for one ``peer_update`` hop (seconds).
#:
#: The control hop must OUTLAST the peer's slow-op deadline plus its reply
#: margin (``relay.SLOW_REPLY_MARGIN_S``) or the CLI would give up a moment
#: before the member's own sentence arrives, and report a bare timeout for an
#: install that was about to answer. The +30 is the control socket's own slack.
UPDATE_CONTROL_TIMEOUT_S = UPDATE_OP_DEADLINE_S + 30.0

#: Where the in-flight marker lives, under the config root.
UPDATE_LOCK_RELPATH: tuple[str, ...] = ("network", "update.lock")

#: The install shapes the receipt's ``method`` may disclose (RU §3: "``method``
#: discloses the install shape on ``done``"). S1 ships the generation-layout
#: pinned install and nothing else; the classic in-place fallback is S2's
#: disclosed last resort and MUST NOT appear here until the drain exists.
METHOD_GENERATION = "uv-tool-generation"


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


def check_published(version: str) -> tuple[bool, str]:
    """Whether ``version`` is resolvable on THIS device's own channel.

    Returns ``(present, why)``: ``present`` is ``True`` when the version must be
    attempted, ``False`` only when the channel POSITIVELY says the version does
    not exist (then ``why`` is the reason clause). An unreachable or unreadable
    channel is ``True`` — inconclusive is not "absent": the installer itself is
    the final authority, and its own tail is carried on ``failed``. That is a
    deliberate asymmetry: refusing on an inconclusive read would make the whole
    op depend on a second, weaker read of the same network the installer is
    about to use anyway.

    A module-level function so tests never touch the network.
    """
    from local_operator import update as update_mod

    try:
        document, why = update_mod._pypi_release_document(version)  # noqa: SLF001 — the same read
    except Exception:  # noqa: BLE001 — see above: inconclusive means "attempt it"
        return True, ""
    if document is None and why == "PyPI does not serve it":
        return False, why
    return True, ""


def install_build(version: str, *, runner: Callable[[list[str], dict[str, str]], int] | None = None) -> str:
    """Install ``version`` into its own generation. Returns the receipt's method.

    THE ONLY PLACE THE MEMBER'S INSTALL SHAPE IS CHOSEN, and it is deliberately
    one shape: ``install_into_generation`` — the pinned-PyPI, verified-tree,
    atomic-pointer install the onboarding lane already uses for an exact version
    (``onboard.step_install``'s ``lop-update <tag>`` / ``uv tool install
    --force --refresh local-operator==<tag>`` spellings, resolved through THIS
    device's own uv). ``source=None`` is what makes it a PyPI resolution; no
    source directory, no ref, ever.

    ``runner`` is the test seam ``install_into_generation`` documents (it
    receives ``(argv, env)`` because the environment is the mechanism under
    test). A real member passes nothing and gets uv.
    """
    from local_operator import update as update_mod

    update_mod.install_into_generation(source=None, version=version, runner=runner)
    return METHOD_GENERATION


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
        raise MeshRefusal("bad_request", "the update target named no version, so nothing was attempted")
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
        return -1, f"this device could not read its own session registry ({exc}), so nothing was touched"
    busy = [record for record, state in rows if state == "live" and bool(record.busy)]
    if not busy:
        return 0, ""
    return len(busy), f"{len(busy)} session(s) are busy"


@contextmanager
def update_lock(root: Path, target: UpdateTarget) -> Iterator[None]:
    """Hold the member's single update lock for the duration of an install.

    ONE at a time on the member too (RU §4.4): a pid-carrying file lock, taken
    NON-blockingly so a second distinct target while one is in flight is refused
    ``update_in_progress`` instead of queued behind a 15-minute install. The
    file is written before the lock is released, so a reader can name the pid
    and the target that hold it — a stale holder is superseded by the lock
    itself (an flock dies with its process), which is why no age bound is
    needed here where the approvals store's reloadable record needs one.

    The lock is taken BEFORE the busy probe, and the probe runs under it: two
    concurrent asks must not both observe idle and both install.
    """
    path = root.joinpath(*UPDATE_LOCK_RELPATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+", encoding="utf-8")  # noqa: SIM115 — held for the whole install
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            holder = "another update"
            try:
                handle.seek(0)
                payload = json.loads(handle.read() or "{}")
                pid = str(payload.get("pid") or "")
                held = str(payload.get("version") or "")
                if pid:
                    holder = f"pid {pid}" + (f" moving to {held}" if held else "")
            except Exception:  # noqa: BLE001 — the holder's name is decoration
                pass
            raise MeshRefusal(
                "update_in_progress",
                f"an update is already running here ({holder}); retry when it settles",
            ) from None
        handle.seek(0)
        handle.truncate()
        handle.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "version": target.version,
                    "started_at": time.time(),
                }
            )
        )
        handle.flush()
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        handle.close()


# ---------------------------------------------------------------------------
# The member executor (S1's single call; S2's drain/re-engage pass extends it)
# ---------------------------------------------------------------------------


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
    detail: Mapping[str, Any] | None = None,
) -> None:
    """One member-side row. Best-effort: a record is not a gate (the mcpdefs rule)."""
    try:
        from local_operator.network.audit import AuditEvent

        fields: dict[str, Any] = {
            "rollout": "",
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
    _audit_member(server, link, "update_requested", target=target, detail={"capability": "update"})

    try:
        return _execute_validated(server, link, target, runner=runner)
    except MeshRefusal as refusal:
        _audit_member(
            server,
            link,
            "update_refused",
            target=target,
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
    runner: Callable[[list[str], dict[str, str]], int] | None,
) -> dict[str, Any]:
    """Validation, the never-force gate, the install, and the receipt."""
    from local_operator import update as update_mod

    kind = detect_install_kind()
    if kind is update_mod.InstallKind.EDITABLE or not generation_layout_supported():
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

    root = Path(server.root)
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
        # THE CHANNEL CHECK COMES AFTER THE IDLE GATE, and that ordering is the
        # reply-time contract: a busy member answers in milliseconds (RU §3)
        # without an HTTP call, and the no-action branches above never pay one
        # either. It goes LAST among the refusals because it is the only one that
        # can be INCONCLUSIVE — see ``check_published``: only a positive "the
        # channel does not serve this version" refuses; an unreachable index
        # falls through to the installer, which is the authority on what it can
        # resolve and whose own tail becomes the receipt.
        published, why = check_published(target_version)
        if not published:
            raise MeshRefusal(
                "target_not_published",
                f"the target {target_version} is not resolvable on this device's channel "
                f"({why}); nothing was installed",
            )
        _audit_member(server, link, "update_started", target=target)
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
                outcome="failed",
                cause="internal",
                detail={"method": METHOD_GENERATION},
            )
            return {
                "state": "failed",
                "code": "",
                "reason": tail or "the install did not finish; nothing was made current",
                "method": METHOD_GENERATION,
                "version": "",
                "sessions": {"moved": 0, "kept": 0},
                "updated_at": time.time(),
            }

    version_now = current_member_version()
    if version_now != target_version:
        _audit_member(
            server,
            link,
            "update_failed",
            target=target,
            outcome="failed",
            cause="internal",
            detail={"method": method},
        )
        return {
            "state": "failed",
            "code": "",
            "reason": (
                f"the install finished but this device still reports {version_now or 'nothing'}; "
                "nothing was advertised"
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
        detail={"method": method, "from": current, "version": target_version},
    )
    return {
        "state": "done",
        "code": "",
        "reason": f"this device moved from {current or 'an unknown build'} to {target_version}",
        "method": method,
        "version": target_version,
        "sessions": {"moved": 0, "kept": 0},
        "updated_at": time.time(),
    }


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
    state = str(detail.get("state") or "") if isinstance(detail, Mapping) else ""
    return (state or "failed"), ""


def update_peer(server: "RelayServer", device_id: str) -> dict[str, Any]:
    """Ask ONE member to move to this device's build. The ``peer_update`` body.

    The origin half of the single-peer verb: dial, check the feature string
    BEFORE the first request (RU §5 — an old member never has to compose a
    refusal it did not ask to make), send the target identity, and report the
    member's own words. It sends NO code, NO ref to fetch, and NO command.
    """
    link = server._ensure_link(device_id)  # noqa: SLF001 — the one dial seam
    label = server._member_name(device_id) or device_id  # noqa: SLF001 — the mesh's own name
    if link is None:
        return {
            "ok": False,
            "code": "unreachable",
            "message": f"{label} is not answering right now, so it was not asked to update",
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
    if _version_tuple(target["version"]) is None:
        if detect_install_kind() is not None and not target["version"]:
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
        return {
            "ok": False,
            "code": "target_not_published",
            "message": (
                f"this device's own build reports {target['version'] or 'no version'}, which "
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
    ok = state in ("done", "already_on_target", "ahead_of_target")
    return {
        "ok": ok,
        "code": state or "failed",
        "state": state,
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
