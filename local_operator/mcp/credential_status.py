"""One read-only answer per MCP server: is it usable, and if not, WHY not?

THE GAP THIS CLOSES (design §5). ``/mcp`` could say ``auth-required`` and name the
server, and nothing else. On a member that is four different problems wearing one
word: never signed in here, signed in on a device whose refresh failed, a share
whose copy has not arrived, or a reference naming a key no store on this device
holds. The operator's next action differs for each, so the status has to say
which one it is — and it has to say it without a dial, without a value, and
without writing anything.

ONE PASS PER QUESTION, because ``describe_servers`` runs this on every catalog
read and the desktop polls it: the network records and placement documents are
walked once, the OAuth grant store is opened once (one row read per OAuth
server), the encrypted store is opened once and shared by every reference, the
user-scope ``mcp.json`` is read once for un-adopted literals, and the durable
evidence document is read once per network. No network call, no relay, no
credential broker, no write. A surface may call this while a connect is failing;
it must not be a second failure.

THE SEVEN STATES (design §5). ``synced``, ``expired``, ``needs_operator_login``,
``not_yet_synced``, ``missing_source``, ``revoked``, ``unknown``. Each is a
remedy, not a mood:

* ``synced``             works here (a local grant, an owner's share, or every
                         reference resolves);
* ``expired``            a source exists but its token is past expiry / was
                         refused a refresh — the OWNER signs in again;
* ``needs_operator_login`` nothing anywhere has ever been signed in;
* ``not_yet_synced``     the server came from a peer and the share/copy has not
                         landed;
* ``missing_source``     a reference names a key no store here holds, or the
                         owner has not adopted its literals yet;
* ``revoked``            this device was a holder and is not any more;
* ``unknown``            a store could not be read, so no claim is made.

VALUE-FREE BY CONSTRUCTION. Nothing here reads a secret's bytes: the store is
asked whether a NAME exists, the grant store whether a grant row exists and is
tombstoned. Reasons carry key names, device names and commands — never a value,
never a token. ``test_credential_status`` asserts a sentinel never appears.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from local_operator.mcp.secret_refs import public_secret_refs

#: The published state vocabulary. Ordered roughly by how far the operator is
#: from a working server, for a client that has to pick one word.
CREDENTIAL_STATES: tuple[str, ...] = (
    "synced",
    "expired",
    "needs_operator_login",
    "not_yet_synced",
    "missing_source",
    "revoked",
    "unknown",
)

#: How many missing key names a reason may list before it summarises. The string
#: rides one row's detail column and a toast tail; an unbounded list of every
#: reference is how one server's row wraps the whole block.
_MAX_NAMED_KEYS = 4

#: A hard ceiling on any reason. Every sentence below is short by construction;
#: this is the structural guarantee rather than a hope about the copy.
_MAX_REASON = 200


def server_states(
    cwd: str,
    *,
    base: Path | None = None,
    configs: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """``{server_name: {state, reason, owner, shape, refs}}`` for ``cwd``'s servers.

    ``configs`` is the merged ``{name: MCPServerConfig}`` mapping when the caller
    already has it (``describe_servers`` does — one load, shared); passed as
    ``None`` this loads it once itself. ``base`` is the config directory the
    stores live under (the caller's injected root in tests); ``None`` means the
    machine's own.

    NEVER RAISES for a store that cannot be read: that is what ``unknown`` is
    for, and a status read that fails the whole catalog is worse than a row that
    admits it does not know.
    """
    from local_operator.mcp.config import load_all_mcp_configs
    from local_operator.paths import config_dir

    root = Path(base) if base is not None else config_dir()
    if configs is None:
        loaded, _sources = load_all_mcp_configs(cwd)
        configs = loaded
    # NO EARLY RETURN ON AN EMPTY CONFIG MAP: a server the discovery loader
    # dropped (disabled, or shaped in a way it rejects) is exactly the case the
    # supplement below exists for, and returning here hid it completely.
    configs = dict(configs or {})
    mesh = _mesh_context(root)
    grants = _GrantIndex(root)
    secrets = _SecretIndex(root)
    literals = _literal_keys_by_server(root)
    origins = _peer_origins(root)
    rows: dict[str, dict[str, Any]] = {}
    for name, cfg in configs.items():
        try:
            rows[str(name)] = _server_row(
                str(name),
                cfg,
                mesh=mesh,
                grants=grants,
                secrets=secrets,
                literals=literals.get(str(name)) or [],
                origin_peer=bool(origins.get(str(name))),
            )
        except Exception:  # noqa: BLE001 — one row must not fail the document
            rows[str(name)] = _row("unknown", "the credential state could not be read")
    # A CONFIGURED SERVER THE DISCOVERY LOADER DROPPED IS STILL CONFIGURED.
    # ``load_all_mcp_configs`` filters disabled/invalid entries out; the
    # user-scope document is where the writers put servers, and gitlab lived
    # there while this pass listed it nowhere — so ``/mcp`` could not name a
    # server the CLI was repairing at all. One extra read of the SAME document
    # the readiness fact reads, and only for names no config covered.
    try:
        from local_operator.network import readiness as readiness_mod

        fact = readiness_mod.mcp_servers_fact(root)
    except Exception:  # noqa: BLE001 — a missing list must not fail the document
        fact = {}
    for entry in fact.get("servers") or []:
        if not isinstance(entry, Mapping):
            continue
        name = str(entry.get("name") or "")
        if not name or name in rows:
            continue
        try:
            rows[name] = _fact_row(entry, mesh=mesh, grants=grants)
        except Exception:  # noqa: BLE001 — one row must not fail the document
            rows[name] = _row("unknown", "the credential state could not be read")
    return rows


# ---------------------------------------------------------------------------
# One server
# ---------------------------------------------------------------------------


def _server_row(
    name: str,
    cfg: Any,
    *,
    mesh: dict[str, Any],
    grants: "_GrantIndex",
    secrets: "_SecretIndex",
    literals: list[str],
    origin_peer: bool,
) -> dict[str, Any]:
    refs_raw = public_secret_refs(cfg)
    field_bindings = {
        str(binding.get("field") or "")
        for ref in refs_raw
        for binding in ref.get("bindings") or []
        if isinstance(binding, Mapping)
    }
    url = str(getattr(cfg, "url", None) or "")
    auth_type = getattr(getattr(cfg, "auth", None), "type", None)
    oauth_key = f"mcp:{url}" if url else ""
    placement = mesh["entries"].get(oauth_key) if oauth_key else None
    grant_live = url and grants.state(url) == "present"
    # A LITERAL is still a header/env credential, and saying "none" for a server
    # whose ``Authorization`` is a literal is the exact blindness adoption exists
    # to fix — so the shape reads the raw maps as well as the reference bindings.
    raw_fields = {
        field
        for field in ("env", "headers")
        if isinstance(getattr(cfg, field, None), Mapping) and getattr(cfg, field)
    }
    has_refs = bool(field_bindings or raw_fields)
    # A PLACEMENT ROW IS NOT PROOF OF OAUTH. ``--repair`` mints an ``mcp:<url>``
    # entry for a header/env server too (it is how the holder population is
    # addressed), so treating its presence as OAuth classified slack as an OAuth
    # server and IGNORED its ``Authorization`` reference — the status then said
    # "needs sign-in" for a server whose key was simply not copied yet. The
    # config's own ``auth.type``, or a live OAuth grant on a server that declares
    # no references, is the distinguishing fact.
    oauth_shape = bool(url) and (
        auth_type == "oauth" or (grant_live and not has_refs)
    )
    shape = (
        "oauth"
        if oauth_shape
        else (
            "header-ref"
            if "headers" in field_bindings or "headers" in raw_fields
            else ("env-ref" if "env" in field_bindings or "env" in raw_fields else "none")
        )
    )

    refs = [
        {"id": str(ref.get("id") or ""), "set": secrets.has(str(ref.get("id") or ""))}
        for ref in refs_raw
    ]

    # LITERALS FIRST, for every shape: a user-scope literal is invisible to the
    # definitions channel, so the mirror on every peer names a key no store
    # holds. Saying "missing" there would send the operator to look for a share
    # that cannot exist; the fix is adoption at the source.
    if literals:
        return _row(
            "missing_source",
            "literal not adopted; run lop network mcp credentials --adopt",
            refs=refs,
            shape=shape,
        )

    mesh_gone = mesh["lost_membership"]
    if shape == "oauth":
        return _oauth_row(
            name,
            url=url,
            placement=placement,
            holder=bool(placement and placement.get("holder")),
            grant=grants,
            evidence=mesh["evidence"].get(oauth_key) or {},
            origin_peer=origin_peer,
            mesh_gone=mesh_gone
            and (placement is not None or grant_live or bool(mesh["evidence"].get(oauth_key))),
            refs=refs,
        )

    if not refs:
        # Nothing to resolve and no OAuth: the server needs no credential of the
        # kinds this document knows. Vacuous "synced" is the honest answer.
        return _row("synced", "", shape=shape)

    missing = [row["id"] for row in refs if row["set"] is not True]
    if not missing:
        return _row("synced", "", shape=shape, refs=refs)
    if secrets.unavailable:
        return _row(
            "unknown",
            "the encrypted secret store could not be read",
            shape=shape,
            refs=refs,
        )
    holder = mesh["holder_of_secret"]
    if any(holder(key) for key in missing):
        return _row(
            "not_yet_synced",
            "not yet synced — run lop network mcp credentials --repair",
            shape=shape,
            refs=refs,
        )
    if mesh["evidence_revoked"](missing):
        owner = _owner_of_rows(mesh["evidence"].get(f"secret:{key}") for key in missing)
        return _row(
            "revoked",
            _bounded(
                f"{owner}: access was revoked; ask its owner for a new share"
                if owner
                else "access was revoked; ask its owner for a new share"
            ),
            shape=shape,
            refs=refs,
        )
    if origin_peer:
        return _row(
            "not_yet_synced",
            "not yet synced — run lop network mcp credentials --repair",
            shape=shape,
            refs=refs,
        )
    return _row(
        "missing_source",
        f"missing {_named(missing)}; set them here or share them from their owner",
        shape=shape,
        refs=refs,
    )


def _fact_row(
    entry: Mapping[str, Any], *, mesh: dict[str, Any], grants: "_GrantIndex"
) -> dict[str, Any]:
    """A status row for a configured server the discovery loader did not return.

    Everything is taken from the readiness fact (names and booleans only): the
    refs and their ``set`` states, whether a placement entry was ever pulled.
    The GRANT half comes from the shared ``_GrantIndex``, not the fact's
    ``has_row``: that boolean is ANY auth row, so a registration-only
    (``client_info``) row would render ``synced`` while the same URL answers
    ``server_has_stored_grant`` False (review round 3, M3). NO network, no dial
    — the same contract as every other row here.
    """
    refs = [
        {"id": str(ref.get("id") or ""), "set": ref.get("set") is True}
        for ref in entry.get("refs") or []
        if isinstance(ref, Mapping)
    ]
    bindings = {
        str(binding.get("field") or "")
        for ref in entry.get("refs") or []
        if isinstance(ref, Mapping)
        for binding in ref.get("bindings") or []
        if isinstance(binding, Mapping)
    }
    url = str(entry.get("url") or "")
    grant_state = grants.state(url) if url else "absent"
    live_grant = grant_state == "present" and not grants.dead(url)
    # The fact's auth hint is a boolean ``auth_*`` field; another key in the same
    # family carries a diagnostics dict, so the check is by TYPE as well as name.
    auth_declared = any(
        str(field).startswith("auth_") and isinstance(value, bool) and value
        for field, value in entry.items()
    )
    has_refs = bool(refs or bindings)
    oauth_shape = bool(url) and (auth_declared or (live_grant and not has_refs))
    if oauth_shape:
        shape = "oauth"
    elif "headers" in bindings:
        shape = "header-ref"
    elif "env" in bindings:
        shape = "env-ref"
    else:
        shape = "none"
    if entry.get("literal_keys"):
        return _row(
            "missing_source",
            "literal not adopted; run lop network mcp credentials --adopt",
            shape=shape,
            refs=refs,
        )
    if refs:
        missing = [ref["id"] for ref in refs if ref["set"] is not True]
        if not missing:
            return _row("synced", "", shape=shape, refs=refs)
        if entry.get("has_placement"):
            return _row(
                "not_yet_synced",
                "not yet synced — run lop network mcp credentials --repair",
                shape=shape,
                refs=refs,
            )
        return _row(
            "missing_source",
            f"missing {_named(missing)}; set them here or share them from their owner",
            shape=shape,
            refs=refs,
        )
    if shape == "none":
        # Nothing to authenticate: the vacuous ``synced`` rule, not a login.
        return _row("synced", "", shape=shape, refs=refs)
    if grant_state == "unknown":
        return _row("unknown", "the OAuth grant store could not be read", shape=shape, refs=refs)
    if live_grant:
        return _row("synced", "", shape=shape, refs=refs)
    if grant_state == "present":
        return _row(
            "expired",
            "sign-in expired here; sign in again on this device",
            shape=shape,
            refs=refs,
        )
    key = f"mcp:{url}" if url else ""
    placed = mesh["entries"].get(key) if key else None
    if placed and placed.get("holder"):
        owner_name = str(getattr(placed.get("entry"), "owner_device_name", "") or "")
        return _row(
            "synced",
            _bounded(
                f"borrows from {owner_name}"
                if owner_name
                else "borrows from a shared device"
            ),
            owner=str(getattr(placed.get("entry"), "owner_device", "") or ""),
            shape=shape,
            refs=refs,
        )
    if placed:
        return _row(
            "not_yet_synced",
            "not yet synced — run lop network mcp credentials --repair",
            shape=shape,
            refs=refs,
        )
    return _row(
        "needs_operator_login",
        "no sign-in found here or on a shared device",
        shape=shape,
        refs=refs,
    )


def _oauth_row(
    name: str,
    *,
    url: str,
    placement: dict[str, Any] | None,
    holder: bool,
    grant: "_GrantIndex",
    evidence: dict[str, Any],
    origin_peer: bool,
    mesh_gone: bool,
    refs: list[dict[str, Any]],
) -> dict[str, Any]:
    owner = _entry_field(placement, "owner_device")
    owner_name = _entry_field(placement, "owner_device_name") or owner
    if mesh_gone:
        return _row(
            "revoked",
            _with_owner("access was revoked; this device is no longer a member", placement),
            owner=owner,
            shape="oauth",
            refs=refs,
        )
    state = grant.state(url)
    if state == "unknown":
        return _row("unknown", "the OAuth grant store could not be read", shape="oauth", refs=refs)
    if state == "present":
        if grant.dead(url):
            return _row(
                "expired",
                "sign-in expired here; sign in again on this device",
                shape="oauth",
                refs=refs,
            )
        return _row("synced", "", shape="oauth", refs=refs)
    evidence_state = str(evidence.get("state") or "")
    code = str(evidence.get("code") or "")
    owner_name = str(evidence.get("owner_name") or owner_name or evidence.get("owner_device") or "")
    if evidence_state in ("expired", "unauthorized"):
        # ``http_401`` means the server refused a bearer the OWNER had just served,
        # which is a different sentence from a refresh the owner could not perform:
        # one says the sign-in is wrong there, the other that it needs a person.
        # The generic code ALREADY says "sign in again"; prefixing another one
        # produced "…: sign in again — sign in again there" (node, round 2).
        if code == "refresh_failed":
            detail = "the token refresh failed there"
        elif code == "http_401":
            detail = "the server rejected the sign-in there"
        else:
            detail = ""
        if detail:
            sentence = (
                f"{owner_name}: sign in again — {detail}"
                if owner_name
                else f"sign in again — {detail}"
            )
        else:
            sentence = f"{owner_name}: sign in again" if owner_name else "sign in again"
        return _row(
            "expired",
            _bounded(sentence),
            owner=owner or str(evidence.get("owner_device") or ""),
            shape="oauth",
            refs=refs,
        )
    if evidence_state == "revoked":
        return _row(
            "revoked",
            _bounded(
                f"{owner_name}: access was revoked; ask its owner for a new share"
                if owner_name
                else "access was revoked; ask its owner for a new share"
            ),
            owner=owner or str(evidence.get("owner_device") or ""),
            shape="oauth",
            refs=refs,
        )
    if evidence_state == "synced":
        # A SUCCESSFUL VALIDATION THAT LEFT NO LOCAL GRANT (a borrow lives only
        # in the asking process's memory). Without this arm the very next status
        # pass reads "no sign-in found" beside a proven working borrow — the
        # gitlab contradiction measured on cloud-node-1.
        return _row(
            "synced",
            _bounded(
                f"borrows from {owner_name}"
                if owner_name
                else "borrows from a shared device"
            ),
            owner=owner or str(evidence.get("owner_device") or ""),
            shape="oauth",
            refs=refs,
        )
    if evidence_state == "unavailable":
        # A TRANSIENT-ONLY VERDICT: the owner could not be reached, so nothing is
        # proved about the credential either way. Saying "synced" here would make
        # an unreachable check look like a working server; the next successful
        # borrow overwrites this row.
        return _row(
            "not_yet_synced",
            _bounded(
                f"{owner_name}: the owner is unreachable; try again later"
                if owner_name
                else "the owner is unreachable; try again later"
            ),
            owner=owner or str(evidence.get("owner_device") or ""),
            shape="oauth",
            refs=refs,
        )
    if holder:
        return _row(
            "synced",
            _bounded(
                f"borrows from {owner_name}" if owner_name else "borrows from a shared device"
            ),
            owner=owner,
            shape="oauth",
            refs=refs,
        )
    if origin_peer:
        return _row(
            "not_yet_synced",
            "not yet synced — run lop network mcp credentials --repair",
            shape="oauth",
            refs=refs,
        )
    return _row(
        "needs_operator_login",
        "no sign-in found here or on a shared device",
        shape="oauth",
        refs=refs,
    )


# ---------------------------------------------------------------------------
# The one-pass readers
# ---------------------------------------------------------------------------


def _mesh_context(root: Path) -> dict[str, Any]:
    """Network records, placement entries and durable evidence, read ONCE.

    ``lost_membership`` answers a narrower question than "no network records":
    only a device that HAS mesh artifacts (a placement file or evidence) and is no
    longer a live member of the network those artifacts belong to has lost access.
    A zero-peer device with a purely local sign-in is not revoked by the absence
    of a network it never joined — the design keeps that path byte-identical.
    """
    entries: dict[str, dict[str, Any]] = {}
    evidence: dict[str, dict[str, Any]] = {}
    has_records = False
    any_live = False
    try:
        from local_operator.network import store
        from local_operator.network.credentials import mcp_state
        from local_operator.network.credentials.placement import PlacementDocument

        records = store.list_networks(root)
    except Exception:  # noqa: BLE001 — no readable network state is the zero-peer answer
        # RETURN the empty answer rather than falling through: the artifacts-only
        # block below needs ``PlacementDocument``/``mcp_state``, which are the very
        # imports that just failed, and pyright is right that they would be
        # unbound (review round 5, type-check).
        def _no_holder(ref_name: str) -> bool:
            return False

        def _no_revoked(secret_names: list[str]) -> bool:
            return False

        return {
            "entries": entries,
            "evidence": evidence,
            "has_records": False,
            "lost_membership": False,
            "holder_of_secret": _no_holder,
            "evidence_revoked": _no_revoked,
        }
    for record in records:
        has_records = True
        try:
            member = record.self_member()
            live = bool(member is not None and member.active)
            document = PlacementDocument.load(
                record.network_id, root, self_device=str(record.self_device_id or "")
            )
            rows = mcp_state.load(record.network_id, root)
        except Exception:  # noqa: BLE001 — one unreadable network must not hide the rest
            continue
        any_live = any_live or live
        self_id = str(record.self_device_id or "")
        for key, entry in document.entries.items():
            entries.setdefault(
                key,
                {
                    "entry": entry,
                    "network_id": record.network_id,
                    "live": live,
                    "holder": bool(self_id and entry.is_holder(self_id)),
                },
            )
        for key, row in rows.items():
            evidence.setdefault(key, row)
    if not records:
        # THE RECORD IS GONE BUT THE ARTIFACTS ARE NOT (``lop network rm``, a
        # restored root, or a removal whose record write raced this read). The
        # brief's rule is explicit — "network record missing or self member
        # inactive -> revoked for any server with a placement or local grant" — so
        # the documents are enumerated directly, exactly as ``PlacementDocument
        # .resolve`` does for the same reason. No live network holds them, hence
        # ``live=False`` for every row.
        from local_operator.network.credentials.placement import (
            PLACEMENT_FILENAME,
            credentials_root,
        )

        for path in sorted(credentials_root(root).glob(f"*/{PLACEMENT_FILENAME}")):
            network_id = path.parent.name
            try:
                document = PlacementDocument.load(network_id, root)
                rows = mcp_state.load(network_id, root)
            except Exception:  # noqa: BLE001 — see the loop above
                continue
            for key, entry in document.entries.items():
                entries.setdefault(
                    key,
                    {
                        "entry": entry,
                        "network_id": network_id,
                        "live": False,
                        # No record means no self id to check a holder row against.
                        "holder": False,
                    },
                )
            for key, row in rows.items():
                evidence.setdefault(key, row)
    lost = bool(entries or evidence) and not any_live

    def holder_of_secret(ref_name: str) -> bool:
        placement = entries.get(f"secret:{ref_name}")
        return bool(placement and placement["live"] and _entry_field(placement, "owner_device"))

    def evidence_revoked(secret_names: list[str]) -> bool:
        """Whether any named secret key's latest evidence says the share was revoked.

        PER KEY, deliberately: evidence is a fact about one credential. A single
        revoked entry elsewhere in the document must not paint this server's row
        revoked, which is exactly what a document-wide flag would do.
        """
        if not secret_names:
            return False
        return any(
            str((evidence.get(f"secret:{name}") or {}).get("state") or "") == "revoked"
            for name in secret_names
        )

    return {
        "entries": entries,
        "evidence": evidence,
        "has_records": has_records,
        "lost_membership": lost,
        "holder_of_secret": holder_of_secret,
        "evidence_revoked": evidence_revoked,
    }


class _GrantIndex:
    """One OAuth grant store, one row read per server URL, three-valued answers."""

    def __init__(self, root: Path) -> None:
        self._store: Any = None
        self._unreadable = False
        self._cache: dict[str, tuple[str, bool]] = {}
        # NO STORE FILE, NO STORE OBJECT. ``AuthStore``'s constructor CREATES its
        # sqlite database, and a status read that creates the grant store it is
        # describing is a side effect the caller never asked for — the machine
        # with no MCP login would grow an empty ``auth.db`` the first time
        # somebody opened ``/mcp``. A missing file is a definite absence.
        #
        # ``db_path`` is passed EXPLICITLY: ``config_dir`` does not relocate the
        # database (``AuthStore`` defaults it to the ambient ``default_db_path()``),
        # so a base-rooted read that only passed ``config_dir`` would describe the
        # operator's real store from a test or an injected root.
        database = root / "auth.db"
        if not database.exists():
            return
        try:
            from local_operator.providers.auth_store import AuthStore

            self._store = AuthStore(db_path=database, config_dir=root)
        except Exception:  # noqa: BLE001 — an unopenable store is "unknown", not "absent"
            self._unreadable = True

    def state(self, url: str) -> str:
        """``present`` | ``absent`` | ``unknown`` — a grant ROW carrying tokens."""
        if not url:
            return "absent"
        if url in self._cache:
            return self._cache[url][0]
        if self._unreadable:
            return "unknown"
        if self._store is None:
            # NO STORE FILE IS A DEFINITE ABSENCE (the constructor's own rule).
            # Falling through would resolve the AMBIENT store from an injected
            # root and describe somebody else's login as this device's.
            return "absent"
        try:
            from local_operator.mcp.auth import McpTokenStorage, server_has_stored_grant

            # PRESENCE AND USABILITY ARE DIFFERENT QUESTIONS (review M4). A
            # tombstoned row answers ``server_has_stored_grant`` False — by
            # design, it cannot serve a bearer — but it still PROVES a sign-in
            # existed here and was refused. Treating it as absent rendered
            # "needs operator login" for a server whose grant simply expired.
            usable = server_has_stored_grant(url, self._store)
            tombstoned = False if usable else McpTokenStorage(url, self._store).grant_is_dead()
            state = "present" if (usable or tombstoned) else "absent"
        except Exception:  # noqa: BLE001 — see :meth:`__init__`
            return "unknown"
        self._cache[url] = (state, False)
        return state

    def dead(self, url: str) -> bool:
        """Whether the stored grant's refresh token was rejected as invalid."""
        if not url or self._store is None:
            return False
        if url in self._cache and self._cache[url][1]:
            return True
        try:
            from local_operator.mcp.auth import McpTokenStorage

            dead = bool(McpTokenStorage(url, self._store).grant_is_dead())
        except Exception:  # noqa: BLE001 — a tombstone we cannot read is not one
            return False
        if dead and url in self._cache:
            self._cache[url] = (self._cache[url][0], True)
        return dead


class _SecretIndex:
    """One encrypted store, shared by every reference's presence check."""

    def __init__(self, root: Path) -> None:
        self._store: Any = None
        self.unavailable = False
        self._cache: dict[str, bool | None] = {}
        try:
            from local_operator.secrets import access
            from local_operator.secrets.keys import store_path

            if store_path(root).exists():
                self._store = access.open_store(root)
        except Exception:  # noqa: BLE001 — reported as ``unknown`` per reference
            self.unavailable = True

    def has(self, name: str) -> bool | None:
        """``True``/``False`` when decidable, ``None`` when the store cannot be read."""
        if not name:
            return None
        if name in self._cache:
            return self._cache[name]
        if self.unavailable:
            return None
        if self._store is None:
            # No store file at all is a definite absence, not an unknown one —
            # the same answer ``credential_source`` gives.
            self._cache[name] = False
            return False
        try:
            from local_operator.secrets.errors import SecretNotFound

            try:
                self._store.describe(name)
                value: bool | None = True
            except SecretNotFound:
                value = False
        except Exception:  # noqa: BLE001 — a store that fails mid-read is unknown
            value = None
        self._cache[name] = value
        return value


def _literal_keys_by_server(root: Path) -> dict[str, list[str]]:
    """Un-adopted ``env``/``headers`` literals in the user-scope ``mcp.json``.

    Names only — ``literal_credential_keys`` never returns a value, and this
    function only groups by server so one row can say "adopt these".
    """
    try:
        from local_operator.mcp.credential_sources import literal_credential_keys

        rows = literal_credential_keys(root)
    except Exception:  # noqa: BLE001 — an unreadable config is not a credential fact
        return {}
    grouped: dict[str, list[str]] = {}
    for row in rows:
        server = str(row.get("server") or "")
        key = str(row.get("key") or "")
        if server and key and bool(row.get("has_value")):
            grouped.setdefault(server, []).append(key)
    return grouped


def _peer_origins(root: Path) -> dict[str, str]:
    """``{server: origin_device}`` from the definitions provenance index.

    The index is written by ``network.mcpdefs`` when it lands a mirror; an empty
    origin is a server this device authored. An unreadable index reads as no
    mirrors, which is the conservative direction: it says "not from a peer", so
    the row falls through to the local answer rather than promising a sync.
    """
    try:
        from local_operator.network import definitions

        rows = definitions.read_index(root).get("servers")
    except Exception:  # noqa: BLE001 — see the docstring
        return {}
    if not isinstance(rows, Mapping):
        return {}
    out: dict[str, str] = {}
    for name, record in rows.items():
        if isinstance(record, Mapping):
            origin = str(record.get("origin") or "")
            if origin:
                out[str(name)] = origin
    return out


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _row(
    state: str,
    reason: str,
    *,
    owner: str = "",
    shape: str = "none",
    refs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "state": state if state in CREDENTIAL_STATES else "unknown",
        "reason": _bounded(reason),
        "owner": str(owner or "")[:128],
        "shape": shape,
        "refs": list(refs or []),
    }


def _entry_field(placement: dict[str, Any] | None, field: str) -> str:
    if not placement:
        return ""
    entry = placement.get("entry")
    return str(getattr(entry, field, "") or "")


def _with_owner(reason: str, placement: dict[str, Any] | None) -> str:
    name = _entry_field(placement, "owner_device_name") or _entry_field(placement, "owner_device")
    return f"{name}: {reason}" if name else reason


def _owner_of_rows(rows: Any) -> str:
    """The first owner name/device among ``rows``, or ``""``."""
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        name = str(row.get("owner_name") or row.get("owner_device") or "")
        if name:
            return name
    return ""


def _named(keys: list[str]) -> str:
    unique = sorted(dict.fromkeys(keys))
    shown = ", ".join(unique[:_MAX_NAMED_KEYS])
    if len(unique) > _MAX_NAMED_KEYS:
        return f"{shown} (+{len(unique) - _MAX_NAMED_KEYS} more)"
    return shown


def _bounded(reason: str) -> str:
    text = " ".join(str(reason or "").split())
    return text[:_MAX_REASON]
