"""Committed audit-input manifests bind the bytes the auditors judged.

WHY. A core-gap audit bundle used to record ``input_hashes.audited_patch_file``,
the SHA-256 of a scratch file that lives outside the repository. Nobody reading
the PR could re-hash it, so "the audited patch equals the shipped catalogue" was
self-attested. ``i18n/audits/evidence/<locale>/_input/cg1.json`` commits that
input instead: one SHA-256 per shipped key, taken from the audited values. This
test is the verifier the receipt otherwise lacks: it recomputes every hash from
the shipped catalogue, so a value edited after the audit (or a manifest edited to
hide it) fails here rather than passing on a claim.

The cases are discovered from the committed manifests, so a locale that has no
manifest contributes none and a new locale needs no edit here. Hash rule (stated
in each manifest): SHA-256 of the UTF-8 bytes of the JSON-decoded value string,
unnormalised — the same string the runtime renders.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = ROOT / "i18n/audits/evidence"
CATALOGUES = ROOT / "local_operator/i18n/catalogues"
MANIFESTS = sorted(EVIDENCE.glob("*/_input/cg1.json"))


def _value_sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


# The parsed manifest is heterogeneous: note strings sit beside the namespaces mapping.
def _body_sha(manifest: dict[str, Any]) -> str:
    """Self-hash: a file cannot contain its own file hash, so the manifest hashes
    its canonical body (every field except ``manifest_sha256``, sorted keys,
    compact separators, UTF-8)."""
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    text = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _superseding_records(evidence: Path) -> list[Path]:
    """The audit records a ledger row's ``Evidence:`` pointer names.

    A row cites either one record file or the namespace's evidence directory,
    whose per-bundle records are the audits that own it. The cg1 bundle is
    excluded: it is the audit being superseded, so pointing the byte pin back at
    it would let a post-audit edit pass.
    """
    if evidence.is_file():
        return [evidence]
    if evidence.is_dir():
        return sorted(p for p in evidence.glob("*/record.json") if p.parent.name != "cg1")
    return []


def _record_binding(record: dict[str, Any], locale: str) -> tuple[Any, Any]:
    """The (English source, audited target) hashes a record binds.

    Two committed shapes exist and both state the same two facts: the later-wave
    audits put ``source_sha256``/``target_sha256`` at the top level, while the
    VI/startup packets keep them under ``input_hashes`` (``en`` for the English
    source, ``<locale>`` or ``<locale>_target`` for the audited bytes). Reading
    either keeps the pin on the bytes an auditor judged instead of forcing a
    rewrite of committed evidence.
    """
    if "source_sha256" in record or "target_sha256" in record:
        return record.get("source_sha256"), record.get("target_sha256")
    hashes = record.get("input_hashes") or {}
    source = (hashes.get("en") or {}).get("sha256") if isinstance(hashes.get("en"), dict) else None
    target = hashes.get(locale) or hashes.get(f"{locale}_target")
    return source, (target or {}).get("sha256") if isinstance(target, dict) else None


def _superseded(locale: str, namespace: str) -> bool:
    """Whether a LATER passed audit owns this namespace's shipped values.

    The ledger is append-only and its last row per (locale, namespace) is what
    ships. cg1's committed manifest is the byte pin only for the bytes cg1
    audited, so it stops governing as soon as the shipped catalogue differs from
    cg1's own ``target_sha256`` — whether the English source moved on (a later PR
    rewrote copy) or only the target did (a human correction that left English
    alone, as in vi/wire.errors.sdk_startup). Comparing cg1 against either case
    would forbid every later copy change in that namespace.

    Superseded therefore means exactly: cg1 no longer owns the shipped bytes, AND
    the ledger's last row for the namespace is ``passed`` at the CURRENT English
    source, cites evidence, and every record that evidence names binds that same
    English source with a target equal to the shipped catalogue. The byte pin
    MOVES to those records — it is never dropped (review R1) — so a value edited
    after the audit still fails here. Anything less returns False and the cg1
    manifest comparison decides.
    """
    from local_operator.i18n import catalogues

    receipt_path = EVIDENCE / locale / namespace / "cg1" / "receipt.json"
    if not receipt_path.exists():
        return False
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    shipped = catalogues.catalogue_sha256(locale, namespace)
    if receipt.get("target_sha256") == shipped:
        # cg1 still owns the shipped values; the manifest comparison is the check.
        return False
    current = catalogues.catalogue_sha256("en", namespace)
    ledger = json.loads((ROOT / "i18n" / "ledger" / f"{locale}.json").read_text(encoding="utf-8"))
    rows = ledger.get("entries", ledger.get("rows", []))
    last = [row for row in rows if row.get("namespace") == namespace][-1:]
    if not last:
        return False
    row = last[0]
    evidence = str(row.get("note", "")).rsplit("Evidence: ", 1)[-1].split()[0:1]
    if not (row.get("status") == "passed" and row.get("source_sha256") == current and evidence):
        return False
    records = _superseding_records(ROOT / evidence[0])
    if not records:
        return False
    for record_path in records:
        source, target = _record_binding(
            json.loads(record_path.read_text(encoding="utf-8")), locale
        )
        assert source == current, f"{record_path}: source hash is stale"
        assert target == shipped, (
            f"{locale}/{namespace}: shipped values differ from the superseding audit "
            f"({record_path} target_sha256)"
        )
    return True


def test_manifest_equals_shipped_catalogue_values_byte_for_byte(path: Path) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    locale = path.parent.parent.name
    assert manifest["locale"] == locale and manifest["audit_id"] == "cg1"
    assert manifest["manifest_sha256"] == _body_sha(manifest), "manifest self-hash is stale"
    mismatched: list[str] = []
    for namespace, keys in manifest["namespaces"].items():
        if _superseded(locale, namespace):
            continue
        shipped = json.loads(
            (CATALOGUES / locale / f"{namespace}.json").read_text(encoding="utf-8")
        )
        # Both directions: a key added or dropped after the audit is as much a
        # change to the audited input as an edited value.
        assert set(keys) == set(shipped), f"{locale}/{namespace}: key set differs from audit input"
        mismatched += [
            f"{locale}/{key}" for key, sha in keys.items() if sha != _value_sha(shipped[key])
        ]
    assert not mismatched, f"shipped values differ from the audited input: {mismatched[:5]}"
    remade = manifest["remade_by_remediation"]
    flat = {key for keys in manifest["namespaces"].values() for key in keys}
    assert set(remade) <= flat, "a remediation entry names a key the manifest does not carry"


@pytest.mark.parametrize("path", MANIFESTS, ids=lambda p: p.parent.parent.name)
def test_every_bundle_binds_the_committed_manifest(path: Path) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    locale = path.parent.parent.name
    file_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    rel = path.relative_to(ROOT).as_posix()
    bundles = sorted((EVIDENCE / locale).glob("*/cg1/record.json"))
    assert {b.parent.parent.name for b in bundles} == set(manifest["namespaces"])
    for record_path in bundles:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        assert "audited_patch_file" not in record["input_hashes"], record_path
        assert record["input_hashes"]["audited_input_manifest"] == file_sha, record_path
        assert record["input_paths"]["audited_input_manifest"] == rel, record_path
        receipt = json.loads((record_path.parent / "receipt.json").read_text(encoding="utf-8"))
        for name, sha in receipt["hashes"].items():
            actual = hashlib.sha256((record_path.parent / name).read_bytes()).hexdigest()
            assert actual == sha, f"{record_path.parent}: receipt stale for {name}"
