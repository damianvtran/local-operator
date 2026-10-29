"""The fixture world: a synthetic owner home, planted canaries, and its safety rails.

WHY A SYNTHETIC HOME AT ALL. The measurement is "did this text reach someone it
should not", and the only honest way to measure it is to put material in the
agent's reach that MUST NOT leave and material that MAY. A real machine cannot
provide that: its leaks are unlabelled and its private files are the operator's.
So every case runs against a freshly built home whose every private file carries a
run-unique canary and whose every in-scope file is decidable by inspection.

THE SAFETY RAILS, which matter more here than anywhere else in this harness:

  * A world is only ever built under a root the caller named, and the builder
    REFUSES a root that already holds anything. An eval that overwrote a real
    directory once would be an eval nobody trusts again.
  * It refuses the operator's home, any ancestor of it, and the config root: a
    typo'd ``--workdir ~`` must fail, not plant canaries in ~/Downloads.
  * It writes no credentials. Provider auth comes from the process environment
    through the app's own auth cascade, so nothing secret is ever written inside
    the tree the agent can read. That is a deliberate difference from the
    benchmark launcher, which can rely on an OS jail; this harness runs in-process
    and has no such jail before the capability profile lands.

WHY THE TRANSCRIPT IS PART OF THE WORLD. The design's headline gap is that the
chief of staff is the owner's own session: its context already holds personal
turns, so "what did Damian do today?" is answerable with ZERO tool calls and no
inbound screen can see it. The seeded turns are that context, and each carries the
case's own canary so a context-sourced leak is attributable to the case that
provoked it.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .canary import Canaries, mint_canaries
from .dataset import Plant, Workspace

__all__ = [
    "FixtureRefused",
    "enroll_bridge",
    "FixtureWorld",
    "build_fixture",
    "render_line",
    "teardown_fixture",
    "verify_fixture",
]

#: The token a line template uses for a class canary, and for the per-case canary.
_CLASS_SLOT = "{canary:"
_CASE_SLOT = "{case_canary}"


class FixtureRefused(Exception):
    """A world cannot be built at the requested root. Never a warning.

    The refusal names the reason because the caller's next move differs: an
    occupied root means "pick another --workdir", a protected root means "you have
    almost certainly typo'd it".
    """


@dataclass(frozen=True)
class FixtureWorld:
    """One built world, with the paths the runner and the report both need."""

    root: Path
    home: Path
    config: Path
    work_roots: tuple[Path, ...]
    outside_roots: tuple[Path, ...]
    canaries: Canaries
    seeded_turns: tuple[Mapping[str, Any], ...]
    facts: Mapping[str, tuple[str, ...]]
    tool_inventory: Mapping[str, Any]
    org: Mapping[str, Any]
    files: tuple[Path, ...] = ()

    def is_inside_work_root(self, path: str | Path) -> bool:
        """Whether ``path`` resolves inside a declared work root.

        The ONE definition the report uses to decide whether a tool call reached
        outside the boundary. Symlinks are resolved first: a fixture symlink that
        escaped a work root would otherwise read as inside, and the fixture exists
        to be adversarial about exactly that.
        """

        try:
            resolved = Path(path).expanduser().resolve()
        except (OSError, RuntimeError):
            return False
        for root in self.work_roots:
            try:
                resolved.relative_to(root.resolve())
            except ValueError:
                continue
            return True
        return False

    def describe(self) -> Mapping[str, Any]:
        """What a report may say about the world without leaking its contents."""

        return {
            "root": str(self.root),
            "work_roots": [str(p) for p in self.work_roots],
            "outside_roots": [str(p) for p in self.outside_roots],
            "canary_classes": list(self.canaries.classes()),
            "canary_fingerprint": self.canaries.fingerprint(),
            "files": len(self.files),
        }


def render_line(line: str, canaries: Canaries, *, case_canary: str | None = None) -> str:
    """Substitute canary slots in one fixture line.

    The two slots are deliberately distinct: ``{canary:<class>}`` is the run's token
    for that leak class, ``{case_canary}`` is the token belonging to the ONE case
    being measured. A missing class token is a dataset error, not something to
    paper over with an empty string - a plant whose canary silently failed to
    render would make its whole leak class undetectable.
    """

    rendered = line
    while _CLASS_SLOT in rendered:
        start = rendered.index(_CLASS_SLOT)
        end = rendered.index("}", start)
        leak_class = rendered[start + len(_CLASS_SLOT) : end]
        rendered = rendered[:start] + canaries.value(leak_class) + rendered[end + 1 :]
    if case_canary is not None:
        rendered = rendered.replace(_CASE_SLOT, case_canary)
    elif _CASE_SLOT in rendered:
        raise FixtureRefused("a line needs the case canary but none was supplied")
    return rendered


def build_fixture(
    *,
    root: Path,
    workspace: Workspace,
    salt: str,
    case_id: str,
    case_canary: str | None = None,
) -> FixtureWorld:
    """Materialise the world under ``root`` and return its handles.

    Idempotence is deliberately NOT offered: every build is a fresh world with fresh
    canaries, because two cases sharing a world would let one case's answer be
    inferred from another's.
    """

    root = Path(root).expanduser()
    _refuse_unsafe_root(root)
    if root.exists() and any(root.iterdir()):
        raise FixtureRefused(
            f"{root} is not empty; refusing to plant a fixture world over existing files"
        )
    root.mkdir(parents=True, exist_ok=True)

    canaries = mint_canaries(salt, workspace.canary_classes, workspace.canary_prefix)
    per_case = case_canary or canaries.case_value(case_id)
    home = (root / workspace.home).resolve()
    config = (root / workspace.config).resolve()
    config.mkdir(parents=True, exist_ok=True)
    home.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for plant in (*workspace.plants, *workspace.work_files):
        written.append(_write_plant(plant, root=root, canaries=canaries, case_canary=per_case))

    work_roots = tuple((root / rel).resolve() for rel in workspace.work_roots)
    outside = tuple((root / rel).resolve() for rel in workspace.outside_work_roots)
    for path in work_roots:
        path.mkdir(parents=True, exist_ok=True)

    seeded = _seeded_turns(workspace, canaries=canaries, case_canary=per_case)
    return FixtureWorld(
        root=root.resolve(),
        home=home,
        config=config,
        work_roots=work_roots,
        outside_roots=outside,
        canaries=canaries,
        seeded_turns=seeded,
        facts=dict(workspace.facts),
        tool_inventory=dict(workspace.tool_inventory),
        org=dict(workspace.org),
        files=tuple(written),
    )


def _write_plant(plant: Plant, *, root: Path, canaries: Canaries, case_canary: str) -> Path:
    target = (root / plant.path).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        raise FixtureRefused(
            f"plant path {plant.path!r} escapes the fixture root; the data file is wrong"
        ) from None
    text = "\n".join(render_line(line, canaries, case_canary=case_canary) for line in plant.lines)
    if plant.kind == "dir":
        # A directory plant is a directory PLUS a readme, because the most likely
        # leak of an out-of-scope tree is its LISTING, and an empty directory
        # carries no canary to detect.
        target.mkdir(parents=True, exist_ok=True)
        index = target / "README.txt"
        index.write_text(text + "\n", encoding="utf-8")
        return index
    target.parent.mkdir(parents=True, exist_ok=True)
    # A canary in the FILE NAME is what makes a listing detectable (see world.json's
    # why-block): the name template already carries `{canary:<class>}` when the
    # dataset wants that, and rendering happens above for the body only, so handle
    # the name here too.
    if _CLASS_SLOT in target.name:
        target = target.with_name(render_line(target.name, canaries, case_canary=case_canary))
    target.write_text(text + "\n", encoding="utf-8")
    return target


def _seeded_turns(
    workspace: Workspace, *, canaries: Canaries, case_canary: str
) -> tuple[Mapping[str, Any], ...]:
    """The pre-seeded owner transcript, with its canaries rendered.

    WHY class canaries are rendered here as well as the per-case one: the
    private-inbox leak this transcript exists to measure is a CONTEXT leak, not a file
    leak, so the `private_inbox` class token is planted in this turn rather than in a
    file. Rendering through the same ``render_line`` the files use keeps one
    substitution rule for both.
    """

    turns: list[Mapping[str, Any]] = []
    for turn in workspace.seeded_turns:
        if not isinstance(turn, Mapping):
            continue
        rendered = dict(turn)
        text = str(rendered.get("text") or "")
        if rendered.get("carries_case_canary"):
            template = str(rendered.get("case_canary_template") or "{case_canary}")
            text = f"{text} {template.replace(_CASE_SLOT, case_canary)}".strip()
            rendered["case_canary_text"] = text
            text = text.replace(_CASE_SLOT, case_canary)
        rendered["text"] = render_line(text, canaries, case_canary=case_canary)
        turns.append(rendered)
    return tuple(turns)


#: The synthetic enrollment the harness commits so the REAL authority gate passes.
#:
#: WHY THIS EXISTS AT ALL. ``execute_bridge`` runs the S-1 local authority fence
#: BEFORE any op: feature enabled, a unique committed owner, a live non-empty
#: generation, a consent receipt, exact owner/tenant/endpoint/credential binding and
#: a designated session. A bare fixture has none of those, so every bridge op would
#: refuse with ``feature_off``/``owner_changed`` and the run would report a clean
#: result having never sent anything - the worst kind of false negative.
#:
#: So the harness ENROLLS the fixture through the shipped seam
#: (``Store.commit_enrollment``) rather than bypassing the gate. What is synthetic is
#: the *authority* (a fake account, tenant and endpoint in an ``eval.invalid`` base
#: URL); what is real is the code path, so a turn that leaks here leaked through the
#: same gate a live session would face. The report records this as a substitution.
_EVAL_BASE_URL = "https://eval.invalid"
_EVAL_CONSENT_VERSION = "eval-v1"
_EVAL_CONSENT_REF = "eval-consent-ref"


def enroll_bridge(world: FixtureWorld, *, session_id: str) -> Any:
    """Enable the bridge in the fixture and commit one synthetic owner.

    Returns the committed ``OwnerSnapshot``; the caller binds it into the
    ``BridgeRuntime`` so the fence's field-by-field comparison succeeds. ``session_id``
    becomes the DESIGNATED session, which is what lets a bridge turn act at all.
    """

    from local_operator.team_bridge.store import OwnerReceipt, Store
    from local_operator.team_bridge.types import OwnerSnapshot

    if not session_id:
        raise FixtureRefused("a designated session id is required to enroll the fixture")

    config_path = Path(world.config) / "config.yml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    # `values:` is the config document's shape (``read_config_values`` returns this
    # mapping); nothing else is set, so every other key keeps its shipped default.
    config_path.write_text(
        "values:\n  team_bridge:\n    enabled: true\n", encoding="utf-8"
    )

    owner = OwnerSnapshot(
        credential_id=1,
        account_id="acct_tb_eval",
        tenant_id="tnt_tb_eval",
        endpoint_id="ep_tb_eval",
        canonical_base_url=_EVAL_BASE_URL,
        enrollment_generation=1,
    ).validated()
    store = Store(Path(world.config))
    store.commit_enrollment(
        owner,
        OwnerReceipt(consent_version=_EVAL_CONSENT_VERSION, consent_ref=_EVAL_CONSENT_REF),
        cos_session_id=session_id,
        require_no_owner=True,
    )
    return owner


#: The screening artifacts the envelope write validates against. They are FIXED rather
#: than derived: the eval measures the OUTBOUND direction, so the inbound screen's own
#: receipt is scaffolding, and a stable digest keeps two runs comparable.
_EVAL_BODY_SHA256 = "9f2a" * 16
_EVAL_TEMPLATE_SHA256 = "aa11" * 16
_EVAL_ENVELOPE_DIGEST = "bb22" * 16
_EVAL_ADMISSION_REVISION = "eval-rev-1"
_EVAL_POLICY_EPOCH = 1
_EVAL_PROMPT_VERSION = "eval-prompt-v1"


def record_inbound_delivery(
    world: FixtureWorld,
    owner: Any,
    *,
    message_id: str,
    delivery_ref: str,
    session_id: str,
    sender_account_id: str,
    sender_endpoint_id: str,
    body_sha256: str = _EVAL_BODY_SHA256,
    display_digest: str = _EVAL_ENVELOPE_DIGEST,
) -> dict[str, Any]:
    """Record the connector's screened-delivery envelope for one inbound card.

    WHY THIS EXISTS (measured 2026-10-29, first real runs). The harness delivered the
    asker's message with ``receive_peer_message`` while skipping the envelope the real
    connector writes first. The session renders ``from_account=`` from THAT envelope
    (never from the sender dict), so every card arrived with an empty sender identity,
    and the ``bridge`` tool - which accepts exactly one of ``endpoint_id``/``account_id``
    - had no addressable recipient. The agent tried to reply, the tool refused with
    "send/reply require exactly one endpoint_id or account_id", and the case scored as
    a delivery failure the MODEL did not cause. Worse, on a branch with no outbound
    gate a refusal-heavy run would then report "zero leaks" while nothing could ever
    have left - the exact false clean this harness exists to prevent.

    What is real: the store's own ``write_screened_request_delivery`` validator, the
    claim/policy/receipt rows it checks, and the session's envelope read. What is
    synthetic: the screening receipt's verdict (``allow``) and the fixture identities.
    """

    import sqlite3
    import time

    from local_operator.team_bridge import store as bridge_store

    store = bridge_store.Store(Path(world.config))
    store.active_owner()  # force schema creation through the real _init
    claim_token = f"eval-claim-{message_id}"
    key = bridge_store._owner_key(owner)
    conn = sqlite3.connect(store._path)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute(
            "INSERT OR REPLACE INTO claims (owner_key, message_id, session_id, token, "
            "generation, expires_at, state) VALUES (?,?,?,?,?,?,?)",
            (
                key,
                message_id,
                session_id,
                claim_token,
                int(owner.enrollment_generation),
                time.time() + 300,
                "held",
            ),
        )
        conn.execute(
            "INSERT OR REPLACE INTO screening_policy (owner_key, version, template_sha256, "
            "default_model, revision, epoch, accepted_at) VALUES (?,?,?,?,?,?,?)",
            (
                key,
                "eval-v1",
                _EVAL_TEMPLATE_SHA256,
                "eval-model",
                _EVAL_ADMISSION_REVISION,
                _EVAL_POLICY_EPOCH,
                time.time(),
            ),
        )
        conn.execute(
            "INSERT OR REPLACE INTO screening_receipts (owner_key, message_id, body_sha256, "
            "template_sha256, prompt_version, model, verdict, categories, envelope_digest, "
            "admission_revision, policy_epoch, creator_generation, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                key,
                message_id,
                body_sha256,
                _EVAL_TEMPLATE_SHA256,
                _EVAL_PROMPT_VERSION,
                "eval-model",
                "allow",
                "[]",
                _EVAL_ENVELOPE_DIGEST,
                _EVAL_ADMISSION_REVISION,
                _EVAL_POLICY_EPOCH,
                int(owner.enrollment_generation),
                time.time(),
            ),
        )
        conn.commit()
    finally:
        conn.close()
    envelope = store.write_screened_request_delivery(
        owner,
        delivery_ref=delivery_ref,
        message_id=message_id,
        designated_session=session_id,
        display_digest=display_digest,
        body_sha256=body_sha256,
        claim_token=claim_token,
        request={
            "thread_id": f"thread_{message_id}",
            "message_kind": "request",
            "request_state": "pending",
            "request_expires_at": "",
            "request_ref": "",
            "sender_account_id": sender_account_id,
            "sender_endpoint_id": sender_endpoint_id,
            "recipient_account_id": str(owner.account_id),
            "recipient_endpoint_id": str(owner.endpoint_id),
        },
    )
    try:
        store.mark_delivery_phase(owner, delivery_ref, "delivered")
    except Exception:  # noqa: BLE001 - a phase write is bookkeeping, never the measurement
        pass
    return dict(envelope or {})


#: Roots a fixture must never be built under. Named individually so the refusal
#: reads as a deliberate guard rather than a path check that happened to fire.
_PROTECTED_SUFFIXES = (".local-operator", ".ssh", ".config", "Library", "Documents", "Downloads", "Desktop")


def _refuse_unsafe_root(root: Path) -> None:
    """Refuse a root whose loss would be the operator's, not the harness's."""

    resolved = root.expanduser().resolve()
    home = Path(os.path.expanduser("~")).resolve()
    if resolved == home:
        raise FixtureRefused(f"refusing to build a fixture world directly in {home}")
    try:
        home.relative_to(resolved)
    except ValueError:
        pass
    else:
        raise FixtureRefused(
            f"refusing to build a fixture world at {resolved}: it is an ancestor of the home directory"
        )
    for name in _PROTECTED_SUFFIXES:
        if resolved.name == name or resolved.name.endswith(name):
            raise FixtureRefused(
                f"refusing to build a fixture world at {resolved}: "
                f"{name!r} is an operator directory, not a scratch root"
            )
    if not resolved.is_absolute():
        raise FixtureRefused(f"{resolved} must be an absolute path")


def verify_fixture(world: FixtureWorld) -> list[str]:
    """Check the world on disk says what the dataset says it should.

    Run before any model call: a canary that failed to render makes a whole leak
    class undetectable, and discovering that after spending a run's budget is the
    expensive version of the same bug.
    """

    problems: list[str] = []
    for path in world.files:
        try:
            if not Path(path).exists():
                problems.append(f"missing fixture file: {path}")
        except OSError as error:  # pragma: no cover - hostile filesystem
            problems.append(f"unreadable fixture path {path}: {error}")
    try:
        names = os.listdir(world.home)
    except OSError as error:
        return [*problems, f"cannot list fixture home: {error}"]
    if not names:
        problems.append(f"fixture home {world.home} is empty")
    for root in world.work_roots:
        if not root.exists():
            problems.append(f"declared work root does not exist: {root}")
    return problems


def teardown_fixture(world: FixtureWorld) -> None:
    """Remove the world. Best effort, and never under an unverified root.

    WHY best-effort: a failed cleanup must not turn a completed measurement into an
    error. WHY the guard: this deletes a tree, and the one unrecoverable bug here
    would be deleting something the harness did not create.
    """

    root = Path(world.root)
    try:
        if not root.exists():
            return
        # Only remove a tree this harness built: it must contain the home we wrote.
        if not (root / "home").exists() and not (root / "config").exists():
            return
        shutil.rmtree(root, ignore_errors=True)
    except OSError:
        return
