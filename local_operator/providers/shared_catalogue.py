"""Machine-scoped shared catalogue receipts, never a borrowed credential cache.

Only explicit device holders can feed machine-wide pickers. Session/derived
permissions require a real session context and deliberately do not enter here.
A catalogue receipt is model metadata, not evidence of inference entitlement.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any, Callable

from local_operator.network import store
from local_operator.network.credentials import lifecycle, reconcile
from local_operator.network.credentials.catalogue import (
    CAPABILITY,
    MAX_BYTES,
    validate_response,
)
from local_operator.network.credentials.client import MeshCredentialClient
from local_operator.network.types import MeshRefusal

_locks_guard = threading.Lock()
_locks: dict[str, threading.Lock] = {}

#: Leg-1 budget for a PICKER refresh (``controller.shared_catalogue``): a list
#: that is drawn while the user waits keeps the short bound it always had, and a
#: slow owner there only leaves the shared rows out of one listing.
LISTING_FETCH_S = 3.0

#: Leg-1 budget for a SELECTION refresh (:func:`prepare_selection_sync`).
#:
#: WHY IT IS LONGER THAN THE LISTING'S (operator report, 2026-11-07): a peer
#: create with ``--model openai/...`` on a device-scoped share answered
#: "catalogue metadata unavailable" and ran its first turn on the device default.
#: The selection reused the listing's 3 s cap, but a selection is a cold owner
#: round trip — the borrower's own relay, the WAN link to the owner (dialled if
#: it had closed), the owner's authority re-reads on either side of a native
#: models GET (measured 0.3-1.0 s from the owner alone) — and the two-relay
#: rig shows the exact sentence the moment that round trip passes 3.0 s. Every
#: OTHER bound on this path is wider: :func:`prepare_selection` allows
#: :data:`SELECTION_WAIT_S` and
#: the runtime's whole switch 8 s (``serving._MODEL_SWITCH_RESOLVE_S``), and
#: the relay's own owner hop allows ``client.CLIENT_DEADLINE_S``. So the 3 s
#: leg-1 cap was the only thing that could fail a selection the outer budget
#: would have let finish.
#:
#: CONSTRAINT: strictly below :data:`SELECTION_WAIT_S`, so a slow owner surfaces
#: as this fetch's own typed refusal rather than the outer timeout racing it (a
#: timeout there abandons the worker thread mid-write). It is a TOTAL budget: the
#: wait for the per-receipt lock (a picker refresh may hold it) is spent from it
#: too, or a lock wait plus a full fetch would overrun the outer wait (review
#: round 1, M1 on #3554: 3 s + 5.5 s against 6.1 s).
SELECTION_FETCH_S = 5.5

#: The outer ``wait_for`` :func:`prepare_selection` puts around the whole worker.
SELECTION_WAIT_S = 6.1

#: The longest a refresh waits for another refresh of the same receipt.
LOCK_WAIT_S = 3.0


def device_scope_share_command(device: str, network_id: str, key: str = "openai") -> str:
    """The ONE spelling of the remedy for a catalogue a session-scoped share cannot feed.

    WHY IT EXISTS. A model switch on a borrow-only device refused with "catalogue
    needs a device-scoped share" and nothing else, while readiness said the peer
    was fine (a turn there borrows per session and works). The gate itself is the
    design (``docs/design/mesh-catalogue-metadata.md``, "Normal consumers"): a
    receipt here is machine-wide, so only an explicit device-scoped holder may
    populate it. Widening a share is the OWNER's access decision, never this
    device's, so the product names the exact command and changes nothing.

    Shared by the refusal below and ``network.readiness`` so the sentence a
    switch prints and the one ``lop network ready`` predicts cannot drift. It
    names the device by ID (``--with`` accepts a name or an id; an id cannot be
    ambiguous) and the network explicitly, because the share verb refuses to
    guess one on an owner that belongs to several.
    """
    return (
        f"lop network credential share {key} --with {device} --network {network_id} "
        "--scope device"
    )


def _device_scope_refusal(
    client: MeshCredentialClient, entry: Any, root: Path | None
) -> MeshRefusal:
    """The strict refusal, carrying who must decide and what they would run.

    Keeps the historical leading clause (``catalogue needs a device-scoped
    share``) so existing readers keyed on it still match. Names only device and
    network identifiers already on this device's placement copy; reads no
    credential.
    """
    owner = str(getattr(entry, "owner_device_name", "") or "") or str(entry.owner_device)
    try:
        member = store.load(client.network_id, root).member(client.self_device)
        here = (member.name if member is not None else "") or client.self_device
    except (OSError, ValueError):
        here = client.self_device
    command = device_scope_share_command(client.self_device, client.network_id)
    # THE COMMAND SITS ON ITS OWN LINE, IN BACKTICKS (design round 1, D1). Every
    # reader of this sentence was checked for a newline: the peer wire and
    # ``--json`` carry it JSON-escaped; ``network.cli._pilot_slash`` prints the
    # receipt text verbatim; the TUI notice wraps per line; approval copy's
    # ``_raw_reason`` folds it to a space and the audit writer drops it — neither
    # of which receives this sentence on the /model path. The backticks matter
    # for ``peer_model.refusal_detail``, which appends ``; still on <model>``
    # straight after the last character: without a delimiter that suffix would
    # read as part of a command a person copies. Both devices are NAMED (D2):
    # the receipt is often read on the owner, where "here" meant the wrong one.
    return MeshRefusal(
        "catalogue_unsupported",
        "catalogue needs a device-scoped share: "
        f"{here} borrows openai from {owner} per session, which runs turns but does not "
        f"share the model list a switch reads. Allowing model switching on {here} is "
        f"{owner}'s decision: it lends openai to every session on {here}. To allow it, "
        f"run on {owner}:\n`{command}`",
    )


def _authority(
    root: Path | None = None, *, native_local: Callable[[], bool | None] | None = None
) -> tuple[str, MeshCredentialClient | None]:
    """Which rung answers openai's catalogue here, BEFORE any share is consulted.

    ``local``: a native login, an environment/stored key or the local-only mark
    outranks every share, so no switch is ever refused for share scope.
    ``none``: this device does not borrow openai (no client, not a holder, or it
    owns the entry). ``borrows``: the share decides. ``unknown``: the native
    store could not be read. Split out of :func:`_share` so the readiness fact
    a viewer predicts from (:func:`catalogue_authority`) IS the gate's own
    first rungs, not a second spelling of them (review round 1, MINOR-1).
    Metadata only: never resolves or refreshes a bearer.

    ``native_local`` replaces ONLY the native-login rung's reader ("is there an
    enabled openai row here that did not arrive over the mesh"; ``None`` =
    unreadable). Readiness passes its own no-write store, because a report must
    not open ``auth.db`` through a reader that can touch a WAL sidecar (its
    ``test_reads_create_nothing`` family); the gate keeps its own reader.
    """
    from local_operator.paths import config_dir
    from local_operator.providers.auth_store import AuthStore
    from local_operator.providers.registry import (
        resolve_env_key,
        stored_provider_env_keys,
    )

    if native_local is not None:
        own = native_local()
    else:
        native = AuthStore.native_oauth_inventory((root or config_dir()) / "auth.db")
        own = (
            None if native is None else any(row[0] == "openai" and row[6] is None for row in native)
        )
    if own is None:
        return "unknown", None
    if own:
        return "local", None
    if resolve_env_key("openai") or "openai" in stored_provider_env_keys(root):
        return "local", None
    client = MeshCredentialClient.for_this_device(root)
    if client is None or not client.should_borrow("openai"):
        return "none", None
    from local_operator.network.credentials.sync import MARK_LOCAL_ONLY, SyncState

    if SyncState.load(client.network_id, root).mark_for("openai") == MARK_LOCAL_ONLY:
        return "local", None
    return "borrows", client


def catalogue_authority(
    root: Path | None = None, *, native_local: Callable[[], bool | None] | None = None
) -> str | None:
    """The readiness fact: ``local``/``none``/``borrows``, or ``None`` if unreadable."""
    try:
        state, _client = _authority(root, native_local=native_local)
    except (OSError, ValueError):
        return None
    return None if state == "unknown" else state


def _share(
    root: Path | None = None, *, strict: bool = False
) -> tuple[MeshCredentialClient, int, int] | None:
    """Structural eligibility of the approved owner share; latch aside.

    Every rung that decides whether THIS device may consume the share lives
    here — native login, env/stored key, borrowability, local-only ownership,
    holder presence, device scope and network/owner/member liveness — so
    :func:`source` and :func:`approved` cannot drift apart. It does NOT include
    the lifecycle/receipt authority gate: that belongs to :func:`source` alone,
    because a frame asking "is a share pending" must not be answered by a
    receipt question while a reader offering model ids must.
    """
    try:
        # Machine-wide discovery must not outrank a native login, legacy key,
        # environment key or local-only mark (:func:`_authority`).
        state, client = _authority(root)
        if state != "borrows" or client is None:
            return None
        entry = client.entry("openai")
        holder = entry.holder(client.self_device) if entry is not None else None
        if entry is None or entry.provider != "openai" or holder is None:
            return None
        if holder.scope != "device" or reconcile.is_derived(holder):
            if strict:
                raise _device_scope_refusal(client, entry, root)
            return None
        network = store.load(client.network_id, root)
        owner = network.member(entry.owner_device)
        self_member = network.member(client.self_device)
        if (
            network.trust != "active"
            or owner is None
            or not owner.active
            or self_member is None
            or not self_member.active
        ):
            return None
        return client, network.epoch, entry.doc_rev
    except MeshRefusal as error:
        if strict and error.code == "catalogue_unsupported":
            raise
        return None
    except (OSError, ValueError):
        return None


def source(
    root: Path | None = None, *, strict: bool = False
) -> tuple[MeshCredentialClient, int, int] | None:
    """May this device consume the share NOW; never resolve a bearer or dial.

    Structural eligibility plus the lifecycle gate: a pending or overflowed
    baseline means this device cannot prove the share still reflects current
    permission, so no reader may advertise stale positive authority from a
    receipt that predates it.
    """
    found = _share(root, strict=strict)
    if found is None:
        return None
    client, _epoch, _revision = found
    state = lifecycle.LifecycleStore(client.network_id, root)
    if state.restriction_pending() or state.load().needs_baseline:
        return None
    return found


def approved(root: Path | None = None) -> bool:
    """Whether an approved owner share EXISTS, receipt or latch aside.

    The same structural question as :func:`source` MINUS the lifecycle gate: its
    one caller paints an empty catalogue while a share is pending, and asking
    :func:`source` there fell through to the bundled static listing whenever a
    latch hid the receipt — rows that branch exists to suppress. Every other
    exclusion is kept, so a device with its OWN credential, a session-scoped or
    derived share, or a dead trust/owner edge still answers False and the frame
    keeps painting the user's own rows. It grants nothing: no receipt, no offered
    id, no authority.
    """
    return _share(root) is not None


def _path(client: MeshCredentialClient, root: Path | None, cache_dir: Path | None) -> Path:
    namespace = f"{client.network_id}:{client.owner_of('openai')}:openai:catalogue"
    tag = hashlib.sha256(namespace.encode()).hexdigest()
    directory = (
        (cache_dir / "mesh-catalogues")
        if cache_dir is not None
        else store.network_root(root) / "catalogues"
    )
    return directory / (tag + ".json")


def _current(document: dict[str, Any], client: MeshCredentialClient, root: Path | None) -> bool:
    """Apply any newer owner lifecycle/withdrawal observation before offering IDs."""
    state = lifecycle.LifecycleStore(client.network_id, root)
    # The baseline/overflow latch is the canonical "cannot prove permission"
    # signal; per-document revisions cannot see it, so a stale positive receipt
    # would otherwise survive every observation that could not be persisted.
    if state.restriction_pending():
        return False
    observed = state.load()
    if observed.needs_baseline:
        return False
    for (owner, key, scope), revisions in observed.high_water.items():
        opaque = hashlib.sha256(f"{client.network_id}:{owner}:{scope}".encode()).hexdigest()
        if (
            owner != client.owner_of("openai")
            or key != "openai"
            or opaque != document["account_scope"]
        ):
            continue
        scope_state = observed.observation_states.get((owner, key, scope))
        if scope_state is not None and scope_state != lifecycle.STATE_AVAILABLE:
            return False
        if revisions > (document["login_chain_revision"], document["material_generation"]):
            return False
    return True


def cached(root: Path | None = None, *, cache_dir: Path | None = None) -> dict[str, Any] | None:
    """No-network reader shared by picker and model-ID validation.

    One current receipt per route, containing its account/incarnation/revisions,
    avoids unioning old accounts or orphan files into a current offered-ID set.
    Missing/expired/foreign/unreadable receipts are not positive evidence.
    """
    found = source(root)
    if found is None:
        return None
    client, epoch, revision = found
    try:
        path = _path(client, root, cache_dir)
        if path.stat().st_size > MAX_BYTES + 2048:
            return None
        envelope = json.loads(path.read_text())
        if set(envelope) != {"document", "epoch", "placement_revision", "received_at", "purpose"}:
            return None
        if (
            type(envelope["epoch"]) is not int
            or type(envelope["placement_revision"]) is not int
            or type(envelope["received_at"]) not in (int, float)
        ):
            return None
        document = validate_response(
            envelope["document"],
            network=client.network_id,
            owner=client.owner_of("openai"),
            key="openai",
        )
        if (
            document["validator"]
            != hashlib.sha256(json.dumps(document["models"], sort_keys=True).encode()).hexdigest()
        ):
            return None
        age = time.time() - envelope["received_at"]
        if (
            envelope["epoch"] != epoch
            or envelope["placement_revision"] != revision
            or envelope["purpose"] != "catalogue"
            or not 0 <= age < document["ttl"]
            or not _current(document, client, root)
        ):
            return None
        return document
    except (OSError, ValueError, TypeError, KeyError):
        return None


def refresh(
    root: Path | None = None,
    *,
    fresh: bool = False,
    cache_dir: Path | None = None,
    fetch_s: float = LISTING_FETCH_S,
    deadline: float | None = None,
) -> dict[str, Any] | None:
    """Bounded worker-thread fetch; coalesce normal TTL reads, no auth fallback.

    Selection calls use fresh=True: cached metadata cannot replace the current
    owner's permission check at a model mutation boundary. Inference itself does
    not call this function. A failed refresh never republishes an old receipt.
    """
    found = source(root, strict=True)
    if found is None:
        return None
    client, epoch, revision = found
    path = _path(client, root, cache_dir)
    with _locks_guard:
        lock = _locks.setdefault(str(path), threading.Lock())
    # ``deadline`` (a ``time.monotonic()`` instant) makes ``fetch_s`` a TOTAL:
    # the lock wait and the fetch are both paid from it (see SELECTION_FETCH_S).
    # Without one a listing keeps its historical lock-wait-then-fetch bounds.
    lock_wait = LOCK_WAIT_S if deadline is None else min(LOCK_WAIT_S, deadline - time.monotonic())
    if lock_wait <= 0 or not lock.acquire(timeout=lock_wait):
        return None
    try:
        if not fresh:
            receipt = cached(root, cache_dir=cache_dir)
            if receipt is not None:
                return receipt
        # A validated cached document needs no dial, but a real refresh must
        # distinguish an old own relay from an unreachable owner before leg 1.
        record = store.find_own_relay(root)
        if record is None:
            raise MeshRefusal("owner_offline", "catalogue relay unavailable")
        if CAPABILITY not in record.capabilities:
            raise MeshRefusal("catalogue_unsupported", "catalogue relay is unsupported")
        # ``fetch_s`` is the caller's leg-1 budget: see LISTING_FETCH_S and
        # SELECTION_FETCH_S for why a picker and a selection differ.
        budget = fetch_s if deadline is None else min(fetch_s, deadline - time.monotonic())
        if budget <= 0:
            return None
        client.deadline_s = min(client.deadline_s, budget)
        document = client.fetch_catalogue_metadata("openai", ttl=60)
        current = source(root)
        if current is None:
            return None
        now_client, now_epoch, now_revision = current
        if (now_client.network_id, now_client.owner_of("openai"), now_epoch, now_revision) != (
            client.network_id,
            client.owner_of("openai"),
            epoch,
            revision,
        ):
            return None
        if not _current(document, now_client, root):
            return None
        store._write_private_json(
            path,
            {
                "document": document,
                "epoch": epoch,
                "placement_revision": revision,
                "received_at": time.time(),
                "purpose": "catalogue",
            },
        )
        return document
    except (MeshRefusal, OSError, ValueError):
        # A fresh authoritative refusal must fence a previously positive TTL
        # receipt before another reader offers it. This is an owned cache file,
        # not a credential/placement write; preserve no untrusted error prose.
        store._write_private_json(path, {"purpose": "catalogue", "unavailable": True})
        raise
    finally:
        lock.release()


def rows(document: dict[str, Any]) -> list[Any]:
    from local_operator.model.discovery import DiscoveredModel

    # Native silence must not resurrect a public API context/output budget.
    # Modality None keeps the existing native parser's unknown semantics.
    return [
        DiscoveredModel(
            **{key: value for key, value in row.items() if key != "provider"},
            authoritative_fields=(
                "context_window",
                "default_context_window",
                "max_context_window",
                "max_tokens",
            ),
        )
        for row in document["models"]
    ]


async def prepare_selection(
    provider: str,
    root: Path | None = None,
    *,
    deadline: float | None = None,
    local_ready: bool = False,
) -> None:
    """Refresh only at an explicit model mutation, off the owner event loop.

    Offline offered-ID readers are not an authority check. A direct cold
    selection must reach the approved owner before it can consume that cache;
    inference still uses the existing grant/budget/credential-selection path.
    """
    provider = provider.strip().lower()
    if provider != "openai":
        return
    import asyncio

    timeout = (
        SELECTION_WAIT_S
        if deadline is None
        else max(0.001, min(SELECTION_WAIT_S, deadline - time.monotonic()))
    )
    try:
        await asyncio.wait_for(
            asyncio.to_thread(prepare_selection_sync, provider, root, local_ready=local_ready),
            timeout,
        )
    except TimeoutError as error:
        raise MeshRefusal("owner_offline", "catalogue metadata unavailable") from error


def prepare_selection_sync(
    provider: str, root: Path | None = None, *, local_ready: bool = False
) -> None:
    """Worker-only twin for existing synchronous/off-loop birth validators.

    ``local_ready`` is a caller-supplied answer to "this pair already resolves
    locally", which preparation honours by returning before any owner RPC. No
    host can compute it today for this provider: the override tiers live on a
    store INSTANCE that a serving session or a peer switch does not hold, and
    the warm path answers None for the catalogue provider. It is a seam, not a
    behaviour change, and it is never a bypass — the caller's own validation
    still runs afterwards.
    """
    provider = provider.strip().lower()
    if provider != "openai" or local_ready or source(root, strict=True) is None:
        return
    try:
        if (
            refresh(
                root,
                fresh=True,
                fetch_s=SELECTION_FETCH_S,
                deadline=time.monotonic() + SELECTION_FETCH_S,
            )
            is None
        ):
            raise MeshRefusal("owner_offline", "catalogue metadata unavailable")
    except MeshRefusal:
        raise
    except (OSError, ValueError) as error:
        # The contract this function states is ONE refusal type. A damaged or
        # unreadable store raised raw OSError/ValueError out of ``refresh``,
        # which both hosts answer as a generic error string with no code.
        raise MeshRefusal("owner_offline", "catalogue metadata unavailable") from error
