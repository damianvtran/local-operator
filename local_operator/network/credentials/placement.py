"""The placement document: who owns each credential, and who may borrow it.

``<config>/network/credentials/<network_id>/placement.json``, 0600, written only
by the device that OWNS an entry. This file is the whole authorisation model for
the broker: a device not in an entry's ``holders`` is refused, and absence of an
entry is a refusal too. It carries no credential material at all — not a token,
not a refresh token, not a prefix, not a hash of either (design §2.1, A5).

WHY THE DOCUMENT IS SAFE TO SYNC, and why those rules are enforced here rather
than trusted:

* **Single-writer rows.** Only ``entry.owner_device`` may write that entry.
  :meth:`PlacementDocument.merge` refuses a peer's version of an entry it does not
  own, so a synced document can never take ownership of a credential away from the
  device holding it. Ownership MOVES only through an explicit local act on the new
  owner, which is the same rule stated as code.
* **Per-entry ``doc_rev``, not per-document.** Two devices declaring different
  providers must not have one's whole file win, and a tie inside one key needs a
  total order: the higher ``doc_rev`` wins that key, and an equal revision loses to
  the local copy (so a replayed frame is a no-op rather than a rewrite).
* **No material, checked on every write.** :func:`_assert_no_material` walks the
  serialised payload and refuses a forbidden field name, so "the placement document
  never carries a token" is a property of the writer rather than a promise in a
  docstring. It does NOT reuse ``audit.FORBIDDEN_DETAIL_KEYS`` verbatim, and the
  reason is worth stating: in the audit's namespace ``key`` means KEY MATERIAL,
  while here ``key`` is the credential's NAME (``openai``) — the first field of
  every entry. A shared list would have refused every legitimate document, which is
  how a safety check gets deleted instead of fixed.

WHERE THE FILES LIVE, and why not beside the network record. ``store`` keeps
``<config>/network/networks/<id>.json`` and its ``.secrets.json``, and
``store.purge_network_artifacts`` deletes those two by name. A third file in that
directory would be left behind by every ``lop network disconnect`` — a stale
placement document is not material, but it IS authority, and a device that left a
network should not keep a copy of who may borrow what. Keeping them under
``credentials/<network_id>/`` with the rest of this slice's state means the purge
question is this module's to answer, and :func:`forget_network` answers it.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from local_operator.network.credentials.types import (
    DEVICE_BOUND_PROVIDERS,
    CredentialPlacementEntry,
    Holder,
    credential_key_for_mcp,
    credential_key_for_provider,
)

#: The document's integer schema, matching the transport's whole-file convention
#: (``mesh-transport-identity.md`` §4.2/§4.3): a whole file carries an integer
#: ``schema``, a standalone log record a namespaced string one.
PLACEMENT_SCHEMA = 1

PLACEMENT_FILENAME = "placement.json"
STATE_FILENAME = "placement.state.json"

DIR_MODE = 0o700
FILE_MODE = 0o600

#: Field names that must never appear in a placement document, anywhere.
#:
#: NOT ``audit.FORBIDDEN_DETAIL_KEYS`` — see the module docstring: that list
#: forbids ``key`` because in an audit record ``key`` is key material, while here
#: it is the credential's name. A list shared between two namespaces that disagree
#: about one of its entries is a list that gets relaxed the first time it fires
#: wrongly, so these are the fields that would be material in THIS document.
_FORBIDDEN_PLACEMENT_KEYS: frozenset[str] = frozenset(
    {
        "access",
        "access_token",
        "refresh_token",
        "token",
        "id_token",
        "secret",
        "client_secret",
        "private_key",
        "password",
        "data",
        "creds",
        "credentials_data",
    }
)


#: One lock per document, and it is the STORE'S lock rather than a second registry
#: here. The relay thread merges a peer's document while the CLI (same process)
#: declares or revokes, and a read-modify-write across those two threads loses one
#: of the two updates; two registries for one file would serialise nothing, so
#: :func:`_write_lock` — an RLock, keyed by path — is reused for the whole
#: read-modify-write rather than only for the write. See :func:`mutate`.
def _document_lock(target: Path) -> Any:
    from local_operator.network import store

    return store._write_lock(target)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def credentials_root(root: Path | None = None) -> Path:
    """``<config>/network/credentials``. DOES NOT CREATE IT — see :func:`credentials_dir_for`.

    A read that mkdirs is a read with a side effect, and this one would be paid on
    every session construction (``build_auth_store`` asks whether any placement
    exists). ``paths.config_dir`` may be read-only in a container, and creating
    ``<config>/network`` on a device that has never joined a network is a visible
    footprint for a feature it is not using.
    """
    return _config_root(root) / "network" / "credentials"


def credentials_dir_for(network_id: str, root: Path | None = None) -> Path:
    """The per-network directory, created 0700. The WRITE path only."""
    path = credentials_root(root) / network_id
    _ensure_private_dir(path)
    return path


def _config_root(root: Path | None) -> Path:
    if root is not None:
        return Path(root)
    from local_operator.paths import config_dir

    return config_dir()


def _ensure_private_dir(path: Path) -> Path:
    """``mkdir -p`` at 0700, INCLUDING every intermediate directory.

    The intermediates matter: ``store._stage_private`` chmods only its target's
    immediate parent, so ``<config>/network`` created implicitly by
    ``mkdir(parents=True)`` would take the umask's mode (commonly 0755) and publish
    the existence and names of the placement documents even though each file is
    0600. Names here are device ids and credential keys — not material, but not
    something to leave world-readable either.
    """
    path.mkdir(parents=True, exist_ok=True)
    for parent in [path, *path.parents]:
        try:
            if parent.is_dir():
                os.chmod(parent, DIR_MODE)
        except OSError:  # pragma: no cover - a parent we do not own
            break
    return path


def network_credentials_dir(network_id: str, root: Path | None = None) -> Path:
    return credentials_dir_for(network_id, root)


def placement_path(network_id: str, root: Path | None = None) -> Path:
    """The document's path. NEVER CREATES anything."""
    return credentials_root(root) / network_id / PLACEMENT_FILENAME


def placement_state_path(network_id: str, root: Path | None = None) -> Path:
    """The observation document's path. NEVER CREATES anything."""
    return credentials_root(root) / network_id / STATE_FILENAME


def has_any_placement(root: Path | None = None) -> bool:
    """Whether this device holds any placement document at all.

    THE CHEAP PREDICATE, and the one ``build_auth_store`` needs: it runs on every
    session construction, and a synthetic home has no ``credentials`` directory. A
    device with no document cannot borrow anything, so it gets the plain store and
    runs byte-identical code (design §5.1).

    IT MUST NOT CREATE THE DIRECTORY, and that is why it globs a bare path rather
    than calling :func:`credentials_root`'s mkdir'ing sibling: this predicate gates
    the identity read that follows it, and both must be side-effect free for a
    device that has never joined a network. ``Path.glob`` on a missing directory
    yields nothing rather than raising.
    """
    try:
        return any((credentials_root(root)).glob(f"*/{PLACEMENT_FILENAME}"))
    except OSError:
        return False


def forget_network(network_id: str, root: Path | None = None) -> int:
    """Delete this network's placement and state documents. Returns files removed.

    Called by ``lop network disconnect``/``rm`` (this slice's ``credential``
    verbs) so a device that left a network does not keep a copy of WHO MAY BORROW
    WHAT from it. Nothing here is material, but the holder list is authority, and
    an authority kept past the membership that justified it is the kind of leftover
    an uninstall is supposed to prevent.
    """
    import shutil

    directory = credentials_root(root) / network_id
    if not directory.is_dir():
        return 0
    removed = 0
    for name in (PLACEMENT_FILENAME, STATE_FILENAME):
        if (directory / name).exists():
            removed += 1
    shutil.rmtree(directory, ignore_errors=True)
    return removed


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _assert_no_material(payload: Any, *, where: str = "placement") -> None:
    """Refuse a payload carrying a field that could be credential material.

    Raises ``ValueError`` rather than dropping the field: a writer that tried to
    put a token in the placement document has misunderstood the model, and a
    silent drop would leave it believing the value had been recorded.
    """
    if isinstance(payload, dict):
        for name, value in payload.items():
            if str(name) in _FORBIDDEN_PLACEMENT_KEYS:
                raise ValueError(
                    f"{where} document would carry the field {name!r}, which can hold "
                    "credential material; the placement document carries WHO owns a "
                    "credential, never the credential (design §2.1)"
                )
            _assert_no_material(value, where=where)
    elif isinstance(payload, list):
        for item in payload:
            _assert_no_material(item, where=where)


def refuse_device_bound(key: str, provider: str) -> None:
    """Refuse to place a provider whose grants are device-bound.

    Called on DECLARE and on GRANT, both, because either one alone leaves a way in:
    a document written before this rule existed would otherwise still be honoured,
    and the check is cheap (a set membership on a name).
    """
    from local_operator.network.types import MeshRefusal

    if provider in DEVICE_BOUND_PROVIDERS:
        from local_operator.network.credentials.types import device_bound_refusal

        raise MeshRefusal("device_bound", device_bound_refusal(provider))


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------


class PlacementDocument:
    """This device's view of who owns which credential.

    Loaded from disk, mutated locally or merged from a peer, saved back. Every
    mutation that writes goes through :meth:`save`, which is the only writer and
    therefore the only place ``_assert_no_material`` has to be armed.
    """

    def __init__(
        self,
        network_id: str,
        entries: dict[str, CredentialPlacementEntry] | None = None,
        *,
        root: Path | None = None,
        epoch: int = 0,
        written_by: str = "",
    ) -> None:
        self.network_id = network_id
        self.entries: dict[str, CredentialPlacementEntry] = dict(entries or {})
        self.root = root
        self.epoch = epoch
        self.written_by = written_by

    # -- reads --------------------------------------------------------------

    @property
    def path(self) -> Path:
        return placement_path(self.network_id, self.root)

    def entry(self, key: str) -> CredentialPlacementEntry | None:
        return self.entries.get(key)

    def owner_of(self, key: str) -> str:
        entry = self.entries.get(key)
        return entry.owner_device if entry is not None else ""

    def is_holder(self, key: str, device: str) -> bool:
        entry = self.entries.get(key)
        return entry is not None and entry.is_holder(device)

    def keys_owned_by(self, device: str) -> list[str]:
        return [key for key, row in sorted(self.entries.items()) if row.owner_device == device]

    def borrowable_keys(self, device: str) -> list[str]:
        """Keys ``device`` is a holder for and does NOT own.

        A holder for a key it owns is not borrowing — the local cascade already
        answers, and treating that as a borrow would put an owner's own request on
        the wire.
        """
        return [
            key
            for key, row in sorted(self.entries.items())
            if row.owner_device != device and row.is_holder(device)
        ]

    def has_remote_owner_for(self, device: str) -> bool:
        """Whether any key is owned elsewhere and held here. The wrap predicate."""
        return bool(self.borrowable_keys(device))

    # -- writes -------------------------------------------------------------

    def declare(
        self,
        key: str,
        *,
        owner_device: str,
        owner_device_name: str = "",
        provider: str = "",
        kind: str = "oauth-rotating",
        identity_label: str = "",
        by: str = "",
    ) -> CredentialPlacementEntry:
        """Declare (or re-declare) ownership of ``key`` on ``owner_device``.

        ``by`` is the device making the call, and it MUST be ``owner_device``: a
        device may only ever declare what IT holds. That is the single-writer rule
        at the only place it can be broken locally, and it is checked here rather
        than trusted to the caller.
        """
        from local_operator.network.types import MeshRefusal

        actor = by or owner_device
        if actor != owner_device:
            raise MeshRefusal(
                "not_owner",
                f"this device cannot declare ownership of {key!r} on behalf of "
                f"{owner_device}: a credential is declared by the device that holds it",
            )
        resolved_provider = provider or (key if not key.startswith("mcp:") else "")
        refuse_device_bound(key, resolved_provider)
        existing = self.entries.get(key)
        entry = CredentialPlacementEntry(
            key=key,
            provider=resolved_provider,
            kind=kind,
            owner_device=owner_device,
            owner_device_name=owner_device_name,
            identity_label=identity_label,
            holders=list(existing.holders) if existing is not None else [],
            declared_at=time.time(),
            doc_rev=(existing.doc_rev + 1) if existing is not None else 1,
        )
        if entry.holder(owner_device) is None:
            # The owner is ALWAYS its own holder at device scope: a device that
            # could not borrow what it holds would take the broker rung for its own
            # requests, and the grant narrows nothing but adds a hop.
            entry.holders.append(
                Holder(
                    device=owner_device,
                    scope="device",
                    granted_at=entry.declared_at,
                    granted_by=owner_device,
                )
            )
        self.entries[key] = entry
        return entry

    def grant(
        self,
        key: str,
        device: str,
        *,
        scope: str = "session",
        by: str = "",
    ) -> CredentialPlacementEntry:
        """Add ``device`` to ``key``'s holders. OWNER-DEVICE-ONLY."""
        from local_operator.network.types import MeshRefusal

        entry = self.entries.get(key)
        if entry is None:
            raise MeshRefusal(
                "no_local_credential",
                f"this device has no placement for {key!r}, so there is nothing to share; "
                f"sign in here first ('lop login {key}') and share it again",
            )
        if by and by != entry.owner_device:
            raise MeshRefusal(
                "not_owner",
                f"only the device that owns {key!r} may change who may borrow it",
            )
        refuse_device_bound(key, entry.provider)
        if scope not in ("session", "device"):
            raise MeshRefusal("bad_request", f"scope must be 'session' or 'device', not {scope!r}")
        existing = entry.holder(device)
        if existing is None:
            entry.holders.append(
                Holder(
                    device=device,
                    scope=scope,
                    granted_at=time.time(),
                    granted_by=entry.owner_device,
                )
            )
        else:
            # A re-grant WIDENS (session → device) or is a no-op. It never narrows:
            # narrowing is `revoke`, and making a re-share silently reduce an
            # existing device-scope grant would break a working peer for a typo.
            if scope == "device":
                existing.scope = "device"
            existing.granted_at = time.time()
            existing.granted_by = entry.owner_device
        entry.doc_rev += 1
        return entry

    def revoke(self, key: str, device: str, *, by: str = "") -> CredentialPlacementEntry:
        """Remove ``device`` from ``key``'s holders. OWNER-DEVICE-ONLY.

        The owner may not revoke itself: its own device-scope holder row is what
        keeps the entry coherent, and a document whose owner is not a holder is one
        no reader can interpret.
        """
        from local_operator.network.types import MeshRefusal

        entry = self.entries.get(key)
        if entry is None:
            raise MeshRefusal("not_a_holder", f"this device has no placement for {key!r}")
        if by and by != entry.owner_device:
            raise MeshRefusal(
                "not_owner", f"only the device that owns {key!r} may change who may borrow it"
            )
        if device == entry.owner_device:
            raise MeshRefusal(
                "not_owner",
                f"{device} owns {key!r} and cannot revoke its own access to it; stop "
                "sharing it by revoking the OTHER devices, or sign out here",
            )
        before = len(entry.holders)
        entry.holders = [row for row in entry.holders if row.device != device]
        if len(entry.holders) == before:
            raise MeshRefusal(
                "not_a_holder",
                f"{device} was not borrowing {entry.provider or key!r} from this device",
            )
        entry.doc_rev += 1
        return entry

    def merge(self, incoming: dict[str, Any], *, from_device: str) -> list[str]:
        """Merge a peer's document. Returns the keys that changed.

        THE OWNER-WRITE RULE IS ENFORCED HERE, and this is the only place a
        document that did not originate locally enters memory. Three refusals:

        * an entry whose ``owner_device`` is not the SENDING device is dropped — a
          peer may only tell us about the credentials IT owns, so a relayed claim
          about a third device is not a membership statement, it is a rumour;
        * an entry for a key we own locally is dropped — we are the authority for
          our own rows, and accepting a peer's version would let a stale document
          overwrite who may borrow what this device holds;
        * a lower-or-equal ``doc_rev`` for a key we already have is dropped, so a
          replayed frame cannot revert a newer revoke.
        """
        changed: list[str] = []
        for keyword in incoming.get("credentials") or []:
            if not isinstance(keyword, dict) or not keyword.get("key"):
                continue
            candidate = CredentialPlacementEntry.from_json(keyword)
            if candidate.owner_device != from_device:
                continue
            local = self.entries.get(candidate.key)
            if local is not None and local.owner_device != candidate.owner_device:
                continue
            if local is not None and candidate.doc_rev <= local.doc_rev:
                continue
            self.entries[candidate.key] = candidate
            changed.append(candidate.key)
        if changed:
            self.epoch = max(self.epoch, int(incoming.get("epoch") or 0))
        return changed

    # -- persistence --------------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        payload = {
            "schema": PLACEMENT_SCHEMA,
            "network_id": self.network_id,
            "epoch": self.epoch,
            "written_by": self.written_by,
            "written_at": time.time(),
            "credentials": [row.to_json() for _, row in sorted(self.entries.items())],
        }
        _assert_no_material(payload)
        return payload

    def save(self) -> Path:
        """Write the document. The ONE writer, and therefore the one authorisation point."""
        from local_operator.network import store

        payload = self.to_json()
        _ensure_private_dir(self.path.parent)
        with store._write_lock(self.path):
            return store._write_private_json(self.path, payload)

    @classmethod
    def load(
        cls,
        network_id: str,
        root: Path | None = None,
        *,
        self_device: str = "",
    ) -> PlacementDocument:
        """Read the document, or an empty one. Never raises for a missing file.

        A corrupt or unreadable document yields an EMPTY document, which refuses
        every borrow rather than allowing one. That is the safe direction: the
        failure mode of "I cannot read who may borrow what" must be "nobody may",
        and the caller's refusal sentence then names the owner as unknown rather
        than serving a credential on a guess.
        """
        path = placement_path(network_id, root)
        document = cls(network_id, root=root, written_by=self_device)
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return document
        try:
            payload = json.loads(raw)
        except ValueError:
            return document
        if not isinstance(payload, dict):
            return document
        document.epoch = int(payload.get("epoch") or 0)
        document.written_by = str(payload.get("written_by") or self_device)
        for row in payload.get("credentials") or []:
            if not isinstance(row, dict) or not row.get("key"):
                continue
            entry = CredentialPlacementEntry.from_json(row)
            document.entries[entry.key] = entry
        return document

    @classmethod
    def resolve(
        cls,
        root: Path | None = None,
        *,
        self_device: str = "",
        network_id: str = "",
    ) -> PlacementDocument | None:
        """The document for ``network_id``, or for whichever network this device is in.

        ``None`` when this device has no document at all — the 0-peer case, and the
        one every caller must handle by taking the plain local path.

        IT ENUMERATES THE DOCUMENTS, NOT THE NETWORK RECORDS, and the difference is
        not stylistic: a device that has been SHARED a credential may hold a placement
        file without yet holding a network record for it (a share that arrived before
        the record was re-read, or a root where only the placement was restored).
        Gating on ``store.list_networks`` made that device answer "nothing is shared
        with me" — the exact state this slice exists to remove — so the files are the
        authority for the question the files answer.
        """
        if not has_any_placement(root):
            return None
        if network_id:
            return cls.load(network_id, root, self_device=self_device)
        for path in sorted(credentials_root(root).glob(f"*/{PLACEMENT_FILENAME}")):
            candidate = cls.load(path.parent.name, root, self_device=self_device)
            if candidate.entries:
                return candidate
        return None


def placement_for_store(
    root: Path | None = None,
    *,
    self_device: str = "",
) -> tuple[str, PlacementDocument] | None:
    """``(network_id, document)`` when this device may borrow something, else ``None``.

    THE ONE PREDICATE THAT DECIDES WHETHER A BORROWER-SIDE WRAPPER EXISTS, and it
    is deliberately narrower than "a document exists": it also requires a key this
    device is a holder for and does not own. A device that owns everything it knows
    about — or holds nothing — can never reach the broker rung, so wrapping its
    store would add a delegation layer that can only ever delegate to itself.

    Reads the identity WITHOUT minting one: ``identity.load`` returns ``None`` for a
    device that has no keypair, and a device with no keypair is not in a network.
    Using ``load_or_mint`` here would create a keypair on every machine that so much
    as constructed a session.
    """
    from local_operator.network.identity import load as load_identity

    if not has_any_placement(root):
        return None
    identity = load_identity(root)
    if identity is None:
        return None
    device = self_device or identity.device_id
    document = PlacementDocument.resolve(root, self_device=device)
    if document is None or not document.has_remote_owner_for(device):
        return None
    return document.network_id, document


def placement_entries_for(
    *,
    provider: str = "",
    mcp_url: str = "",
    root: Path | None = None,
) -> tuple[str, CredentialPlacementEntry] | None:
    """The placement entry for a provider or an MCP server URL, if this device has one.

    The MCP path's lookup: ``mcp/manager.py`` knows a server URL and nothing else,
    so the key is built here rather than at the call site — one spelling of
    ``mcp:<url>`` for the whole slice.
    """
    key = credential_key_for_mcp(mcp_url) if mcp_url else credential_key_for_provider(provider)
    from local_operator.network import store

    for record in store.list_networks(root):
        document = PlacementDocument.load(record.network_id, root)
        entry = document.entry(key)
        if entry is not None:
            return record.network_id, entry
    return None


@contextmanager
def mutate(
    network_id: str, root: Path | None = None, *, self_device: str = ""
) -> Iterator[PlacementDocument]:
    """Read-modify-write the document under its lock.

    The ``store.mutate`` shape, for the same stated reason (``store``'s module
    docstring): a caller that loads, edits and saves without holding the target's
    lock writes back a document that predates another writer's save, which silently
    REVERTS a share or a revoke. The CLI and the relay are the two writers here.
    """
    path = placement_path(network_id, root)
    with _document_lock(path):
        document = PlacementDocument.load(network_id, root, self_device=self_device)
        yield document
        document.save()
