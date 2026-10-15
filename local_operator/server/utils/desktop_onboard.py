"""Narrow publication of existing signed-run receipts, never raw authority records."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from local_operator.network import approvals


def bound_root(root: Path) -> Path:
    """The existing engine has ambient-root steps; never silently switch profiles.

    Normal desktop construction uses ConfigManager at paths.config_dir(). Until
    every existing runner/secret subprocess is root-parameterized, a custom app
    object with a different root must refuse BEFORE request/admission/SSH work.
    Do not mutate os.environ: concurrent requests would inherit another account.
    """
    return approvals.enrollment.bound_profile(root)


def detail(root: Path, approval_id: str) -> dict[str, Any]:
    """Local-only, allowlisted and bounded; references/signature/raw data stay local."""
    record = approvals.load_record(approval_id, root=root)
    ref = str((record.get("credential_ref") or {}).get("ref") or "")

    def sentence(value: Any) -> str:
        text = str(value or "")[:4000]
        if ref:
            text = text.replace(ref, "[credential reference]")
        text = text.replace(str(root), "[profile]")
        # OS diagnostics can name internal file paths; the run card needs the
        # failure fact, not that local path or a temporary key filename.
        return re.sub(r"(?<![\w:])(?:~?/[^\s\"']+|[A-Za-z]:\\[^\s\"']+)", "[path]", text)

    receipts = []
    for row in (record.get("receipts") or [])[-100:]:
        if not isinstance(row, dict):
            continue
        receipts.append(
            {
                "run_id": str(row.get("run_id") or ""),
                "step": str(row.get("step") or ""),
                "at": float(row.get("at") or 0),
                "ok": bool(row.get("ok")),
                "detail": sentence(row.get("detail")),
                "digest": str(row.get("digest") or ""),
            }
        )
    if record.get("kind") == approvals.enrollment.KIND:
        # Only typed, bounded observation facts, never arbitrary step data.
        for source, public in zip((record.get("receipts") or [])[-100:], receipts):
            if isinstance(source, dict) and source.get("step") == "verify" and "data" in source:
                public["data"] = approvals.enrollment.public_report(source["data"])
            elif (
                isinstance(source, dict)
                and isinstance(source.get("data"), dict)
                and source["data"].get("failure_kind") == "changed_facts"
            ):
                public["data"] = approvals.enrollment.failure_report(
                    record, source["data"], root=root
                )
    run = record.get("run") or {}
    if not isinstance(run, dict):
        run = {}
    state = str(record.get("state") or "")
    onboard = record.get("kind") in (
        approvals.KIND_DEVICE_ONBOARD,
        approvals.enrollment.KIND,
    ) and not any(
        key in (record.get("what") or {})
        for key in ("credential_inheritance", "credential_finite_share")
    )
    next_action = (
        ("retry" if state == "failed" else "approve" if state == "requested" else None)
        if onboard
        else "existing_approval_surface"
    )
    replacement = None
    if state == "failed" and receipts:
        observation = receipts[-1].get("data") or {}
        if observation.get("failure_kind") == "changed_facts":
            replacement = observation.get("replacement_id")
            next_action = "review_replacement" if replacement else "start_new_review"
    return {
        "approval_id": record["approval_id"],
        "replacement_id": replacement,
        "state": state,
        "kind": record.get("kind"),
        "onboarding": onboard,
        "authority_version": record.get("authority_version"),
        "scope": "network_only" if record.get("kind") == approvals.enrollment.KIND else None,
        "request_digest": (
            record.get("request_digest")
            if record.get("kind") == approvals.enrollment.KIND
            else None
        ),
        "expires_at": record.get("expires_at"),
        "run_id": run.get("run_id", ""),
        "run_active": state == "connecting" and not approvals.run_is_stale(record),
        "receipts": receipts,
        "next": next_action,
    }
