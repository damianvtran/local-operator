"""The bundle carrier through the SYNC paths: install, check, apply, publish.

The core carrier (format, validation, storage) is exercised by
``test_skill_bundle``; ``test_skill_bundle_integration`` pins storage/adapters.
This file pins the SERVICE integration the desktop route and the CLI drive:
a bundle install verifies the archive before parsing, a public install refuses
without the publisher/hash confirmation, the check classifies tree hashes, an
apply is fast-forward-only, and a publication uploads before it publishes with
a durable upload record.

Everything runs against an isolated config root and a fake credentialed client,
so every assertion reads a PERSISTED effect or the exact wire answer.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.config import ConfigManager
from local_operator.hub_sync import check as chk
from local_operator.hub_sync import provenance as prov
from local_operator.hub_sync import service as svc
from local_operator.hub_sync.identity import LinkError
from local_operator.hub_sync.skill_bundle import build_bundle, tree_hash_of
from local_operator.providers.radient_credentials import configured_radient_base_url

HUB_ID = "sk-bundle-1"
NAME = "pdf-forms"
PUBLISHER = {"account_id": "acct-9", "name": "Ada"}


def _authored(root: Path, *, body: str = "# Body\n") -> Path:
    source = root / "skills" / NAME
    source.mkdir(parents=True, exist_ok=True)
    (source / "SKILL.md").write_text(
        "---\n"
        f"name: {NAME}\n"
        "description: Fill and validate PDF forms.\n"
        "---\n\n" + body,
        encoding="utf-8",
    )
    references = source / "references"
    references.mkdir(exist_ok=True)
    (references / "guide.md").write_text("alpha\n" * 20, encoding="utf-8")
    return source


def _manifest_document(
    built: Any, *, with_publisher: bool = True, visibility: str = "public"
) -> dict[str, Any]:
    document: dict[str, Any] = {
        "document_version": 1,
        "id": HUB_ID,
        "name": built.name,
        "description": built.description,
        "bundle": built.bundle.to_json(),
        "visibility": visibility,
        "tenant_id": None,
        "revision": '"hub-rev-1"',
        "content_revision": 1,
        "sync_contract": {"revision_cas": 1, "skills": 1, "skill_bundles": 1},
    }
    if with_publisher:
        document["published_by"] = dict(PUBLISHER)
    return document


class RemoteBundle:
    """The fetch + download seam for ONE bundle row (and a publish seam)."""

    def __init__(self, base: str, document: dict[str, Any], data: bytes) -> None:
        self.base_url = base
        self.api_key = None
        self.document = document
        self.data = data
        self.downloads: list[str] = []
        self.uploads: list[bytes] = []
        self.published: list[dict[str, Any]] = []
        self.upload_error: BaseException | None = None

    # -- reads -----------------------------------------------------------------
    def get_skill(self, skill_id: str, **_kw: Any) -> dict[str, Any]:
        return dict(self.document)

    def download_skill_bundle(self, skill_id: str, *, dest: Any = None, **_kw: Any) -> Any:
        self.downloads.append(skill_id)
        if dest is not None:
            dest.write(self.data)
        return SimpleNamespace(data=None, not_modified=False, etag=None, server_hash=None)

    # -- publication -----------------------------------------------------------
    def upload_skill_bundle(self, data: bytes, **_kw: Any) -> dict[str, Any]:
        if self.upload_error is not None:
            raise self.upload_error
        self.uploads.append(bytes(data))
        # A real server re-hashes the streamed bytes; the fake must answer for
        # the bytes it was GIVEN, never for whatever document it happens to hold.
        digest = hashlib.sha256(bytes(data)).hexdigest()
        return {"hash": digest, "size": len(data)}

    def publish_skill_document(self, document: dict[str, Any], **_kw: Any) -> dict[str, Any]:
        self.published.append(dict(document))
        return {
            "accepted_document": {**document, "id": HUB_ID},
            "revision": '"hub-rev-2"',
            "content_revision": 2,
            "operation_id": "op-bundle-publish",
            "sync_contract": {"revision_cas": 1, "skills": 1, "skill_bundles": 1},
        }


def _ctx(root: Path, client: Any) -> svc.HubSyncContext:
    manager = ConfigManager(root)
    return svc.HubSyncContext(
        config_dir=root,
        config_manager=manager,
        client_for_tenant=lambda _tenant: client,
    )


def _rig(tmp_path: Path, *, publisher: bool = True) -> tuple[Path, str, Any, RemoteBundle]:
    root = tmp_path / ".local-operator"
    source = _authored(tmp_path)
    built = build_bundle(source)
    base = configured_radient_base_url(ConfigManager(root))
    remote = RemoteBundle(base, _manifest_document(built, with_publisher=publisher), built.data)
    return root, base, built, remote


def _install(ctx: svc.HubSyncContext, base: str, **fields: Any) -> dict[str, Any]:
    return svc.install_skill(
        ctx,
        hub_source=base,
        hub_id=HUB_ID,
        visibility="public",
        tenant_id=None,
        alias=NAME,
        **fields,
    )


# -- install -------------------------------------------------------------------------


def test_install_downloads_verifies_extracts_and_writes_the_baseline(tmp_path: Path) -> None:
    root, base, built, remote = _rig(tmp_path)
    ctx = _ctx(root, remote)

    result = _install(ctx, base, confirm=True)

    destination = root / "skills" / NAME
    assert (destination / "SKILL.md").is_file()
    assert (destination / "references" / "guide.md").is_file()
    # The installed tree's identity is the MANIFEST's tree hash, recomputed from
    # the bytes on disk -- not the archive hash.
    assert tree_hash_of(destination) == built.tree_hash
    assert remote.downloads == [HUB_ID]
    record = prov.read_baseline(root, "skill", result["link"]["local_id"])
    assert record is not None
    assert "bundle" in record.fields
    assert record.fingerprint == built.tree_hash
    assert record.content_state == "present"
    # The download staging directory is private and removed on success.
    downloads = root / "hub" / "downloads"
    assert not downloads.exists() or list(downloads.iterdir()) == []
    # The confirmation's named fact travels in the answer.
    assert result["reports"][0]["publisher"] == {"id": PUBLISHER["account_id"], "name": "Ada"}


def test_install_refuses_a_tampered_archive_and_writes_nothing(tmp_path: Path) -> None:
    root, base, built, remote = _rig(tmp_path)
    tampered = bytearray(built.data)
    tampered[len(tampered) // 2] ^= 0xFF
    remote.data = bytes(tampered)
    ctx = _ctx(root, remote)

    with pytest.raises(LinkError) as caught:
        _install(ctx, base, confirm=True)

    assert caught.value.code == "bundle-hash-mismatch"
    assert not (root / "skills" / NAME).exists()
    assert [r for r in prov.iter_baselines(root) if r.kind == "skill"] == []


def test_install_requires_a_named_publisher(tmp_path: Path) -> None:
    root, base, _built, remote = _rig(tmp_path, publisher=False)
    ctx = _ctx(root, remote)

    with pytest.raises(LinkError) as caught:
        _install(ctx, base, confirm=True)

    assert caught.value.code == "bundle-confirmation-required"
    assert not (root / "skills" / NAME).exists()
    assert remote.downloads == []


def test_install_refuses_without_confirmation_when_the_row_names_a_publisher(
    tmp_path: Path,
) -> None:
    root, base, _built, remote = _rig(tmp_path)
    ctx = _ctx(root, remote)

    with pytest.raises(LinkError) as caught:
        _install(ctx, base)

    assert caught.value.code == "bundle-confirmation-required"
    assert remote.downloads == []


def test_install_confirmation_must_match_the_hash(tmp_path: Path) -> None:
    root, base, built, remote = _rig(tmp_path)
    ctx = _ctx(root, remote)

    with pytest.raises(LinkError) as caught:
        _install(
            ctx,
            base,
            confirm_publisher_id=PUBLISHER["account_id"],
            confirm_bundle_hash="0" * 64,
        )

    assert caught.value.code == "bundle-confirmation-mismatch"
    assert not (root / "skills" / NAME).exists()
    assert built.bundle.hash == remote.document["bundle"]["hash"]


def test_install_dry_run_reports_the_summary_and_writes_nothing(tmp_path: Path) -> None:
    root, base, built, remote = _rig(tmp_path)
    ctx = _ctx(root, remote)

    result = _install(ctx, base, dry_run=True)

    assert result["dry_run"] is True
    assert result["bundle"]["bundle"]["hash"] == built.bundle.hash
    assert result["publisher"] == {"id": PUBLISHER["account_id"], "name": "Ada"}
    assert not (root / "skills" / NAME).exists()
    assert remote.downloads == []


# -- check + apply -------------------------------------------------------------------


def _check(root: Path, remote: RemoteBundle) -> list[chk.ItemCheck]:
    return chk.check_hub_skills(config_dir=root, client_for_tenant=lambda _t: remote)


def _install_then_republish(tmp_path: Path) -> tuple[Path, str, Any, RemoteBundle, Any]:
    root, base, built, remote = _rig(tmp_path)
    ctx = _ctx(root, remote)
    _install(ctx, base, confirm=True)
    record = [r for r in prov.iter_baselines(root) if r.kind == "skill"][0]
    return root, base, built, remote, record


def test_check_classifies_a_newer_remote_as_available_and_apply_fast_forwards(
    tmp_path: Path,
) -> None:
    root, base, built, remote, _record = _install_then_republish(tmp_path)
    # Republish: a new tree (same SKILL.md, different reference body).
    newer = build_bundle(_authored(tmp_path, body="# Second version\n"))
    remote.document = _manifest_document(newer)
    remote.data = newer.data
    ctx = _ctx(root, remote)

    checks = _check(root, remote)
    assert len(checks) == 1
    assert checks[0].classification is not None
    assert checks[0].classification.state == "remote-only"
    assert checks[0].remote_fp == newer.tree_hash

    report = svc.apply_one(ctx, checks[0], auto=False)
    assert report.outcome == "merged"
    assert report.applied is True
    assert tree_hash_of(root / "skills" / NAME) == newer.tree_hash
    record = prov.read_baseline(root, "skill", report.local_id)
    assert record is not None
    assert record.fingerprint == newer.tree_hash


def test_a_local_edit_is_kept_and_never_overwritten(tmp_path: Path) -> None:
    root, base, _built, remote, _record = _install_then_republish(tmp_path)
    edited = root / "skills" / NAME / "references" / "guide.md"
    edited.write_text("locally edited\n", encoding="utf-8")
    newer = build_bundle(_authored(tmp_path, body="# Second version\n"))
    remote.document = _manifest_document(newer)
    remote.data = newer.data
    ctx = _ctx(root, remote)

    checks = _check(root, remote)
    assert checks[0].classification is not None
    assert checks[0].classification.state == "both-changed"

    report = svc.apply_one(ctx, checks[0], auto=False)
    assert report.outcome != "merged"
    assert report.error_class == "bundle-conflict"
    assert edited.read_text(encoding="utf-8") == "locally edited\n"


def test_a_local_only_edit_is_up_to_date_on_the_pull_side(tmp_path: Path) -> None:
    root, base, _built, remote, _record = _install_then_republish(tmp_path)
    edited = root / "skills" / NAME / "references" / "guide.md"
    edited.write_text("locally edited\n", encoding="utf-8")
    ctx = _ctx(root, remote)

    checks = _check(root, remote)
    assert checks[0].classification is not None
    assert checks[0].classification.state == "local-only"
    assert checks[0].verdict == "up-to-date"

    report = svc.apply_one(ctx, checks[0], auto=False)
    assert report.outcome in ("unchanged", "up-to-date")
    assert edited.read_text(encoding="utf-8") == "locally edited\n"


# -- publication ---------------------------------------------------------------------


def _published_rig(tmp_path: Path) -> tuple[Path, str, Any, RemoteBundle]:
    """A rig whose AUTHORED source is the discovery root's own skill directory.

    ``default_skill_roots`` includes ``<config>/skills``, which is also the
    managed root: publishing an already-installed skill is the ADOPT path, and
    that is exactly what this rig exercises.
    """

    root = tmp_path / ".local-operator"
    source = _authored(root)
    built = build_bundle(source)
    base = configured_radient_base_url(ConfigManager(root))
    return root, base, built, RemoteBundle(base, _manifest_document(built), built.data)


def _preview(ctx: svc.HubSyncContext) -> dict[str, Any]:
    return svc.preview_skill_publication(
        ctx,
        source_skill=NAME,
        local_alias=NAME,
        visibility="public",
        tenant_id=None,
        subscribe=True,
        auto_push=False,
        confirm_public_egress=False,
        form="bundle",
    )


def test_publish_uploads_then_publishes_and_materializes_the_link(tmp_path: Path) -> None:
    root, base, built, client = _published_rig(tmp_path)
    ctx = _ctx(root, client)
    svc.record_server_capabilities(
        ctx, {"revision_cas": 1, "skills": 1, "skill_bundles": 1}, source=base
    )

    preview = _preview(ctx)
    assert preview["form"] == "bundle"
    assert preview["bundle"]["file_count"] == built.file_count
    assert preview.get("reason") is None

    result = svc.publish_skill(
        ctx,
        source_skill=NAME,
        local_alias=NAME,
        visibility="public",
        tenant_id=None,
        subscribe=True,
        auto_push=False,
        confirm_public_egress=False,
        preview_receipt=preview["preview_receipt"],
        form="bundle",
    )

    assert len(client.uploads) == 1
    assert len(client.published) == 1
    assert "bundle" in client.published[0]
    link = result["link"]
    record = prov.read_baseline(root, "skill", link["local_id"])
    assert record is not None
    assert "bundle" in record.fields
    assert record.fingerprint == built.tree_hash
    assert tree_hash_of(root / "skills" / NAME) == built.tree_hash


def test_a_failed_upload_publishes_nothing(tmp_path: Path) -> None:
    from local_operator.clients._http import APIError

    root, base, built, client = _published_rig(tmp_path)
    client.upload_error = APIError("storage down", status_code=503)
    ctx = _ctx(root, client)
    svc.record_server_capabilities(
        ctx, {"revision_cas": 1, "skills": 1, "skill_bundles": 1}, source=base
    )
    preview = _preview(ctx)

    with pytest.raises(LinkError):
        svc.publish_skill(
            ctx,
            source_skill=NAME,
            local_alias=NAME,
            visibility="public",
            tenant_id=None,
            subscribe=True,
            auto_push=False,
            confirm_public_egress=False,
            preview_receipt=preview["preview_receipt"],
            form="bundle",
        )

    assert client.published == []
    assert [r for r in prov.iter_baselines(root) if r.kind == "skill"] == []


def test_a_retry_after_the_accepted_publish_does_not_upload_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, base, built, client = _published_rig(tmp_path)
    ctx = _ctx(root, client)
    svc.record_server_capabilities(
        ctx, {"revision_cas": 1, "skills": 1, "skill_bundles": 1}, source=base
    )
    preview = _preview(ctx)
    receipt = preview["preview_receipt"]

    # Crash AFTER the hub accepted the row (the local link never materialized):
    # the receipt keeps the accepted answer and is NOT completed, which is the
    # exact state a retry reconciles.
    real = svc._materialize_published_skill

    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise LinkError("local-concurrent-edit", "simulated crash after acceptance")

    monkeypatch.setattr(svc, "_materialize_published_skill", boom)
    first = svc.publish_skill(
        ctx,
        source_skill=NAME,
        local_alias=NAME,
        visibility="public",
        tenant_id=None,
        subscribe=True,
        auto_push=False,
        confirm_public_egress=False,
        preview_receipt=receipt,
        form="bundle",
    )
    assert first["local_link_receipt"]["state"] == "pending-reconciliation"
    assert len(client.uploads) == 1
    assert len(client.published) == 1

    monkeypatch.setattr(svc, "_materialize_published_skill", real)
    second = svc.publish_skill(
        ctx,
        source_skill=NAME,
        local_alias=NAME,
        visibility="public",
        tenant_id=None,
        subscribe=True,
        auto_push=False,
        confirm_public_egress=False,
        preview_receipt=receipt,
        form="bundle",
    )

    # Reconcile, never re-publish: the blob is not re-sent and the manifest is
    # not published twice.
    assert len(client.uploads) == 1
    assert len(client.published) == 1
    assert second["link"]["local_id"]


def test_a_server_without_the_bundle_capability_is_refused_before_any_upload(
    tmp_path: Path,
) -> None:
    root, base, built, client = _published_rig(tmp_path)
    ctx = _ctx(root, client)
    # Advertises skills + CAS but NOT the bundle carrier.
    svc.record_server_capabilities(ctx, {"revision_cas": 1, "skills": 1}, source=base)

    preview = _preview(ctx)

    assert preview["reason"] == "bundle-unsupported-server"
    assert client.uploads == []


# -- the pre-commit upload a republish needs -------------------------------------------


def test_the_push_upload_hook_sends_the_rebuilt_archive_bytes(tmp_path: Path) -> None:
    root, base, built, remote = _rig(tmp_path)
    ctx = _ctx(root, remote)
    result = _install(ctx, base, confirm=True)
    record = prov.read_baseline(root, "skill", result["link"]["local_id"])
    assert record is not None and record.identity is not None

    hook = svc.bundle_upload_hook(ctx, record.identity, record)
    hook(dict(record.fields))

    assert len(remote.uploads) == 1
    # The bytes are the deterministic archive for the INSTALLED tree, and their
    # hash is exactly the manifest hash the commit will send.
    assert hashlib.sha256(remote.uploads[0]).hexdigest() == built.bundle.hash
    assert remote.uploads[0] == built.data


def test_the_push_upload_hook_refuses_a_locally_edited_tree(tmp_path: Path) -> None:
    root, base, _built, remote = _rig(tmp_path)
    ctx = _ctx(root, remote)
    result = _install(ctx, base, confirm=True)
    record = prov.read_baseline(root, "skill", result["link"]["local_id"])
    assert record is not None and record.identity is not None
    # The check's projected document still describes the OLD tree; the managed
    # directory has moved on. Nothing may be uploaded for a tree that was not
    # the one reviewed.
    stale = dict(record.fields)
    (root / "skills" / NAME / "references" / "guide.md").write_text("edited\n", encoding="utf-8")

    hook = svc.bundle_upload_hook(ctx, record.identity, record)
    with pytest.raises(LinkError) as caught:
        hook(stale)

    assert caught.value.code == "bundle-local-changes"
    assert remote.uploads == []


# -- ``lop skills push``: the manual republish -----------------------------------------


def test_the_cli_push_uploads_the_new_archive_then_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The operator's evidence: modify a file, push, and the hub gets the new tree."""

    from local_operator import cli
    from local_operator.hub_sync.skill_bundle import bundle_manifest_from_directory
    from tests.unit.hub_sync.test_push import FakeTransport

    root, base, _built, remote = _rig(tmp_path)
    ctx = _ctx(root, remote)
    result = _install(ctx, base, confirm=True)
    record = prov.read_baseline(root, "skill", result["link"]["local_id"])
    assert record is not None and record.identity is not None
    svc.record_server_capabilities(
        ctx, {"revision_cas": 1, "skills": 1, "skill_bundles": 1}, source=base
    )
    # Local edit -> the check's manifest for this tree is NEW.
    (root / "skills" / NAME / "references" / "guide.md").write_text("v2\n", encoding="utf-8")
    newer = bundle_manifest_from_directory(root / "skills" / NAME)
    assert newer.bundle.hash != _built.bundle.hash
    transport = FakeTransport(accepted=newer.to_json())

    monkeypatch.setattr(
        "local_operator.hub_sync.service.build_clients_sync", lambda _cm: (lambda _t: remote, "ok")
    )
    monkeypatch.setattr(
        "local_operator.hub_sync.service.push_transport", lambda _ctx, _rec: (transport, None)
    )

    code = cli.skills_command(
        __import__("argparse").Namespace(
            skills_command="push", name=NAME, yes=True, json=True
        ),
        root,
    )

    captured = capsys.readouterr()
    assert code == 0, captured.out + captured.err
    assert len(remote.uploads) == 1
    assert hashlib.sha256(remote.uploads[0]).hexdigest() == newer.bundle.hash
    assert transport.commit_calls[0]["document"]["bundle"]["hash"] == newer.bundle.hash
    assert transport.preview_calls == []
