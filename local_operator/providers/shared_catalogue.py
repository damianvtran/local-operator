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
from typing import Any

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
        # Machine-wide discovery must not outrank a native login, legacy key or
        # environment key. Read metadata only; never resolve/refresh a bearer.
        from local_operator.paths import config_dir
        from local_operator.providers.auth_store import AuthStore
        from local_operator.providers.registry import (
            resolve_env_key,
            stored_provider_env_keys,
        )

        native = AuthStore.native_oauth_inventory((root or config_dir()) / "auth.db")
        if native is None or any(row[0] == "openai" and row[6] is None for row in native):
            return None
        if resolve_env_key("openai") or "openai" in stored_provider_env_keys(root):
            return None
        client = MeshCredentialClient.for_this_device(root)
        if client is None or not client.should_borrow("openai"):
            return None
        from local_operator.network.credentials.sync import MARK_LOCAL_ONLY, SyncState

        if SyncState.load(client.network_id, root).mark_for("openai") == MARK_LOCAL_ONLY:
            return None
        entry = client.entry("openai")
        holder = entry.holder(client.self_device) if entry is not None else None
        if entry is None or entry.provider != "openai" or holder is None:
            return None
        if holder.scope != "device" or reconcile.is_derived(holder):
            if strict:
                raise MeshRefusal("catalogue_unsupported", "catalogue needs a device-scoped share")
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
    root: Path | None = None, *, fresh: bool = False, cache_dir: Path | None = None
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
    if not lock.acquire(timeout=3):
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
        client.deadline_s = min(client.deadline_s, 3)
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

    timeout = 6.1 if deadline is None else max(0.001, min(6.1, deadline - time.monotonic()))
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
        if refresh(root, fresh=True) is None:
            raise MeshRefusal("owner_offline", "catalogue metadata unavailable")
    except MeshRefusal:
        raise
    except (OSError, ValueError) as error:
        # The contract this function states is ONE refusal type. A damaged or
        # unreadable store raised raw OSError/ValueError out of ``refresh``,
        # which both hosts answer as a generic error string with no code.
        raise MeshRefusal("owner_offline", "catalogue metadata unavailable") from error
