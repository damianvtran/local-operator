"""Receipt-bound initial adoption, shared by CLI and desktop skill pulls.

An occupied native alias is not permission to replace a folder. Preview takes a
restorable snapshot BEFORE a model sees text, then captures a proposal. Confirm
reuses those exact reviewed bytes, never reruns the model, and revalidates local
bytes, remote revision, ownership and policy. Normal Hub document validation is
unchanged: legacy paths and opaque assets enter only through adoption storage.
"""

from __future__ import annotations

import secrets
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from local_operator.config import read_config_values
from local_operator.config_writer import config_write_lock
from local_operator.hub_sync import policies as pol
from local_operator.hub_sync import provenance as prov
from local_operator.hub_sync.identity import LinkError
from local_operator.hub_sync.skill_document import SkillDocument

PREVIEWS = "adoption-previews"


def mint_adoption_receipt() -> str:
    """A preview receipt the printed CLI command can always accept (issue #3394).

    The CLI tells users to confirm a reviewed preview with
    ``--adopt-receipt <receipt> --yes``. ``secrets.token_urlsafe`` uses the
    URL-safe base64 alphabet, which includes ``-``, so about 1 receipt in 64
    began with ``-``; argparse then read that separate value as an option and
    refused with "expected one argument". Drawing again until the first
    character is not ``-`` keeps the full 24 random bytes (192 bits) and the
    same alphabet, so existing receipts and their lookup are unaffected; only
    the leading character is constrained.
    """
    while True:
        token = secrets.token_urlsafe(24)
        if not token.startswith("-"):
            return token


def _refusal(code: str, message: str) -> LinkError:
    # Message construction stays at the point of use so live locale changes do
    # not freeze English at module import time. The code is a stable protocol.
    from local_operator.i18n.keys import wire_skill_adoption as copy

    key = {
        "alias-occupied": copy.preview_required,
        "adoption-preview-changed": copy.preview_changed,
        "adoption-needs-choice": copy.needs_choice,
    }.get(code)
    return LinkError(code, key().envelope()["text"] if key else message)


def _gate(ctx: Any, resolved: Any) -> tuple[Any, Any]:
    """The SAME bounded raw-tree and durable ownership gates on both arms."""
    from local_operator.hub_sync.skill_adoption import read_tree
    from local_operator.hub_sync.skill_storage import ManagedSkillStorage

    storage = ManagedSkillStorage(ctx.config_dir)
    with config_write_lock(ctx.config_dir):
        destination = storage.adoption_subject(resolved.identity, resolved.alias)
        tree = read_tree(destination)
        values = read_config_values(ctx.config_dir)
        if values is None and (ctx.config_dir / "config.yml").exists():
            raise LinkError("invalid-link-policy", "configuration is unreadable")
        policy = pol.read_policy(values or {}, resolved.identity)
        return tree, policy


def adopt_skill(
    ctx: Any,
    resolved: Any,
    *,
    dry_run: bool,
    confirm: bool,
    receipt: str | None,
    strategy: Literal["merge", "keep-local", "take-hub"],
    allow_llm: bool,
    subscribe: bool | None,
    auto_push: bool | None,
) -> dict[str, Any]:
    """Preview or confirm adoption. Caller holds the ordinary hub update lease."""
    from local_operator.hub_sync import service as svc
    from local_operator.hub_sync.skill_adoption import (
        AdoptionError,
        build_proposal,
        durable_snapshot,
    )
    from local_operator.hub_sync.skill_storage import ManagedSkillStorage

    try:
        tree, policy = _gate(ctx, resolved)
        remote = SkillDocument.from_json(resolved.canonical)
        if dry_run:
            # Snapshot raw paths/bytes before any path mapping or paid resolver
            # work. A failed or declined preview therefore remains restorable.
            snapshot = durable_snapshot(ctx.config_dir, resolved.alias, tree)
            proposal = build_proposal(
                resolved.alias,
                tree,
                remote,
                resolver=svc._resolver_for(ctx, allow_llm),
                allow_llm=allow_llm,
                strategy=strategy,
            )
            token = mint_adoption_receipt()
            payload = proposal.to_json()
            captured = {
                "preview_receipt": token,
                "expires_at": (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat(),
                "identity": resolved.identity.to_json(),
                "alias": resolved.alias,
                "remote_revision": resolved.revision,
                "remote_fingerprint": remote.fingerprint,
                "local_tree_hash": tree.tree_hash,
                "policy_revision": policy.revision,
                "subscribe": subscribe,
                "auto_push": auto_push,
                "strategy": strategy,
                "snapshot": snapshot,
                "adoption": payload,
                "document": proposal.document.to_json() if proposal.document else None,
            }
            svc._write_publication_preview(ctx.config_dir, captured, PREVIEWS)
            return {
                "reports": [
                    {
                        "kind": "skill",
                        "local_id": resolved.identity.local_id,
                        "name": resolved.alias,
                        "hub_id": resolved.identity.hub_id,
                        "outcome": "needs-review",
                        "applied": False,
                    }
                ],
                "adoption": payload,
                "adoption_receipt": token,
                "snapshot": snapshot,
                "confirmation_required": True,
                "dry_run": True,
                "form": "document",
                "link": None,
                "links": svc.links_snapshot(ctx),
            }
        if not confirm or not receipt:
            raise _refusal("alias-occupied", "review an adoption preview first")
        captured = svc._read_publication_preview(ctx.config_dir, receipt, PREVIEWS)
        if not captured:
            raise _refusal("adoption-preview-changed", "the adoption preview is missing")
        expiry = svc._parse_iso(str(captured.get("expires_at") or ""))
        expected = {
            "identity": resolved.identity.to_json(),
            "alias": resolved.alias,
            "remote_revision": resolved.revision,
            "remote_fingerprint": remote.fingerprint,
            "local_tree_hash": tree.tree_hash,
            "policy_revision": policy.revision,
            "subscribe": subscribe,
            "auto_push": auto_push,
            "strategy": strategy,
        }
        if (
            expiry is None
            or expiry < datetime.now(timezone.utc)
            or any(captured.get(key) != value for key, value in expected.items())
        ):
            raise _refusal("adoption-preview-changed", "the adoption preview changed")
        proposal = captured["adoption"]
        if proposal.get("state") == "needs-choice":
            raise _refusal("adoption-needs-choice", "choose a safe adoption strategy")
        # A first adoption must not mint an outbound receipt or a public egress
        # grant. Manual push remains available; auto-push has no agreed point.
        if auto_push:
            raise LinkError("no-receipt", "adopt first, then explicitly publish local changes")
        document = (
            SkillDocument.from_json(captured["document"]).with_name(resolved.alias)
            if captured.get("document") is not None
            else None
        )
        basis = "adopt-" + ("equal" if proposal.get("state") == "equal" else strategy)
        baseline = replace(
            prov.make_record(
                "skill",
                resolved.identity.local_id,
                resolved.identity.hub_id,
                resolved.identity.tenant_id,
                resolved.canonical,
                basis,
            ),
            identity=resolved.identity,
            local_alias=resolved.alias,
            remote_revision=resolved.revision,
            lineage="clean",
            content_state="present",
            legacy_subscribe_hint=subscribe,
        )
        # Exact normalized equality (and keep-local) links without touching any
        # authored bytes; the storage records its canonical read projection.
        write_document = (
            None if proposal.get("state") == "equal" or strategy == "keep-local" else document
        )
        transaction = ManagedSkillStorage(ctx.config_dir).adopt(
            resolved.identity,
            resolved.alias,
            write_document,
            baseline,
            expected_tree_hash=tree.tree_hash,
            snapshot=captured["snapshot"],
            expected_policy_revision=policy.revision,
            basis=basis,
            carried_paths=tuple(
                str(row.get("local_path") or row["path"])
                for row in proposal.get("files", [])
                if row.get("action") == "carried" and strategy != "take-hub"
            ),
        )
        svc._refresh_config(ctx)
        # Report the actual post-adoption direction immediately, rather than
        # waiting for the next runner tick to discover unpublished local work.
        # The ancestor is the fetched Hub revision, never our consolidation.
        from local_operator.hub_sync import check as chk

        installed = ManagedSkillStorage(ctx.config_dir).snapshot(resolved.identity.local_id)
        local = installed.document.with_name(remote.name)
        classification = chk.classify(remote.fingerprint, local.fingerprint, remote.fingerprint)
        svc.record_checks(
            ctx,
            [
                chk.ItemCheck(
                    "skill",
                    resolved.identity.local_id,
                    resolved.alias,
                    resolved.identity.hub_id,
                    resolved.identity.tenant_id,
                    "up-to-date",
                    classification=classification,
                    local_fp=local.fingerprint,
                    remote_fp=remote.fingerprint,
                    local=local.to_json(),
                    remote=resolved.canonical,
                    base=resolved.canonical,
                    remote_revision=resolved.revision,
                    link_id=resolved.identity.link_id,
                )
            ],
        )
        links = svc.links_snapshot(ctx)
        link = next(row for row in links["links"] if row["link_id"] == resolved.identity.link_id)
        return {
            "reports": [
                {
                    "kind": "skill",
                    "local_id": resolved.identity.local_id,
                    "name": resolved.alias,
                    "hub_id": resolved.identity.hub_id,
                    "outcome": "merged",
                    "applied": True,
                    "operation_id": transaction.operation_id,
                    "direction": "pull",
                    "remote_revision": resolved.revision,
                    "model_used": proposal.get("model"),
                    "backup": captured["snapshot"]["path"],
                }
            ],
            "adoption": proposal,
            "snapshot": captured["snapshot"],
            "link": link,
            "links": links,
        }
    except AdoptionError as error:
        raise LinkError(error.code, str(error)) from None
