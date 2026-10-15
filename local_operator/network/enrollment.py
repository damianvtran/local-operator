"""The CLOSED network-only authority codec, not a second approval state machine.

Frozen format/domain contract: LOH-3 security freeze SHA256
b076eff4ed6feaa093df41eb5df746b52cf968970d70c24ad91fd74016f8f11e.
Legacy readers must fail BEFORE presence even on a copied/renamed pending file.
That property comes from the envelope's absent top-level ID, NOT from signature
framing alone (old clients sign before checking a request commitment). Deliberate
unwrapping loses that structural property; domains then prevent execution replay.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from local_operator.network.types import MeshRefusal

KIND = "device_onboard_network_only"
FORMAT = "lop-network-enrollment"
VERSION = 1
DIRECTORY_NAME = "enrollment-approvals"
REQUEST_DOMAIN = b"lop-network-enrollment-request-v1\0"
DECISION_DOMAIN = b"lop-network-enrollment-v1\0"
ID = re.compile(r"apn_[0123456789abcdefghjkmnpqrstvwxyz]{13}\Z")
DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
CONFIRMATION = "network_enrollment_confirmation_required"
REMEDY = (
    "Review the network-enrollment-only card, its target, operator anchor, device-control "
    "grants and original window, then confirm that exact snapshot. Account logins are not shared."
)


def require(condition: bool, message: str = "The network enrollment authority is invalid") -> None:
    if not condition:
        raise MeshRefusal("network_enrollment_invalid", message)


def _object(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MeshRefusal("network_enrollment_invalid", "Expected a structured enrollment fact")
    return value


def _array(value: Any) -> list[Any]:
    if not isinstance(value, list):
        raise MeshRefusal("network_enrollment_invalid", "Expected a list of enrollment facts")
    return value


def bound_profile(root: Path | None) -> Path:
    """Existing runner subprocesses use ambient config; never silently switch it."""
    from local_operator.paths import config_dir

    ambient = config_dir()
    selected = ambient if root is None else root
    if selected.resolve() != ambient.resolve():
        raise MeshRefusal(
            "onboard_root_mismatch",
            "Restart this backend with LOCAL_OPERATOR_CONFIG_DIR set to its configured profile "
            "before adding a remote server; no setup changes were made",
        )
    return selected


def ssh_target(value: str) -> str:
    """Only unambiguous inputs to the EXISTING host/port parser, not new IPv6 syntax."""
    require(
        isinstance(value, str) and len(value) <= 253 and value.count(":") <= 1,
        "Use a hostname or IPv4 address with an optional SSH port for this onboarding path",
    )
    host, separator, port = value.partition(":")
    require(re.fullmatch(r"[A-Za-z0-9_.][A-Za-z0-9_.-]*", host) is not None)
    if separator:
        require(
            port.isascii() and port.isdigit() and 1 <= int(port) <= 65535,
            "Choose an SSH port between1 and65535",
        )
    return value


def credential_reference(value: Any) -> str:
    """Validate a bootstrap REFERENCE without resolving a key or reading a path.

    Reuse stored-secret name grammar for names/agent references; path references
    stay paths. Obvious pasted PEM/JSON/key material and multiline input must not
    be persisted in a supposedly non-secret card. This is new-kind policy ONLY.
    """
    from local_operator.secrets.crypto import InvalidSecretName, validate_name

    require(
        isinstance(value, str) and 0 < len(value) <= 512,
        "Use a bootstrap credential reference, not private key material",
    )
    require(
        not any(ord(c) < 32 or ord(c) == 127 for c in value),
        "Use a single-line bootstrap credential reference",
    )
    upper = value.lstrip().upper()
    require(
        "-----BEGIN" not in upper and "PRIVATE KEY-----" not in upper,
        "Use a bootstrap credential reference, not private key material",
    )
    require(
        not upper.startswith(
            ("-----BEGIN", "PRIVATE KEY", "OPENSSH PRIVATE", "RSA PRIVATE", "{", "[")
        ),
        "Use a bootstrap credential reference, not private key material",
    )
    if value.startswith("path:"):
        require(bool(value[5:].strip()), "Choose a nonempty bootstrap key path reference")
    else:
        name = value[6:] if value.startswith("agent:") else value
        try:
            normalized = validate_name(name)
        except InvalidSecretName:
            raise MeshRefusal(
                "network_enrollment_invalid", "Choose a valid stored bootstrap credential name"
            ) from None
        require(normalized == name, "Use the exact canonical stored bootstrap credential name")
    return value


def restricted(record: Mapping[str, Any]) -> bool:
    return record.get("kind") == KIND or str(record.get("approval_id") or "").startswith("apn_")


def validate(record: Mapping[str, Any], *, approval_id: str | None = None) -> None:
    """Strict authority coherence before decode, signing, every writer and run.

    Mutable lifecycle keys remain the existing store's vocabulary. Immutable
    scope/target fields are closed: an unknown field cannot become implicit
    provisioning, an admin convenience, or credential inheritance permission.
    """
    aid = record.get("approval_id")
    require(isinstance(aid, str) and ID.fullmatch(aid) is not None)
    require(approval_id is None or aid == approval_id)
    require(record.get("kind") == KIND)
    require(
        isinstance(record.get("state"), str)
        and record["state"]
        in {
            "requested",
            "approved",
            "connecting",
            "connected",
            "denied",
            "expired",
            "withdrawn",
            "failed",
        }
    )
    require(
        isinstance(record.get("receipts"), list)
        and all(isinstance(r, dict) for r in record["receipts"])
    )
    require(type(record.get("schema")) is int and record["schema"] == 1)
    require(type(record.get("authority_version")) is int and record["authority_version"] == VERSION)
    request_id = record.get("request_id")
    require(
        isinstance(request_id, str)
        and re.fullmatch(r"req_[A-Za-z0-9_-]{1,124}", request_id) is not None
    )
    for key in ("created_at", "expires_at"):
        value = record.get(key)
        valid = False
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            try:
                valid = math.isfinite(value) and value > 0
            except OverflowError:
                valid = False
        require(valid)
    require(record["expires_at"] > record["created_at"])
    what = _object(record.get("what"))
    require(
        isinstance(what, dict)
        and set(what)
        == {
            "connect",
            "install",
            "anchor",
            "network_id",
            "role",
            "unattended",
            "grant",
            "provisioning",
        }
    )
    require(what["connect"] is True and what["install"] is True)
    require(what["provisioning"] == "network_only" and what["role"] == "drive")
    require(type(what["unattended"]) is bool)
    require(
        isinstance(what["network_id"], str)
        and bool(what["network_id"])
        and len(what["network_id"]) <= 128
    )
    require(type(what["grant"]) is list and what["grant"] in ([], ["approve"]))
    anchor = _object(what["anchor"])
    require(isinstance(anchor, dict) and set(anchor) == {"key_id", "spki_fp", "statement_digest"})
    require(all(isinstance(v, str) and 0 < len(v) <= 256 for v in anchor.values()))
    target = _object(record.get("device"))
    require(
        isinstance(target, dict)
        and set(target)
        == {"device_id", "name", "fingerprint", "host", "user", "transport", "host_key_fp"}
    )
    require(
        all(
            isinstance(v, str) and len(v) <= 512 and not any(ord(c) < 32 for c in v)
            for v in target.values()
        )
    )
    require(
        target["transport"] == "ssh" and bool(target["host"]) and not target["host"].startswith("-")
    )
    require(re.fullmatch(r"[A-Za-z0-9_.:][A-Za-z0-9_.:-]*", target["host"]) is not None)
    require(re.fullmatch(r"(?:[A-Za-z0-9_.][A-Za-z0-9_.-]*)?", target["user"]) is not None)
    require(re.fullmatch(r"SHA256:[A-Za-z0-9+/=]+", target["host_key_fp"]) is not None)
    ssh_target(target["host"])
    ref = _object(record.get("credential_ref"))
    require(isinstance(ref, dict) and set(ref) == {"kind", "ref"} and ref["kind"] == "ssh")
    require(
        isinstance(ref["ref"], str)
        and 0 < len(ref["ref"]) <= 512
        and "\n" not in ref["ref"]
        and "\r" not in ref["ref"]
    )
    credential_reference(ref["ref"])
    requester = _object(record.get("requested_by"))
    require(
        isinstance(requester, dict) and set(requester) == {"session_id", "device_id", "surface"}
    )
    require(all(isinstance(v, str) and len(v) <= 256 for v in requester.values()))
    require(requester["surface"] in {"cli", "desktop"})
    require("machine" not in record)


def commitment(record: Mapping[str, Any]) -> str:
    """Exact NEW framing. Never retry a failed commitment with a legacy domain."""
    from local_operator.network.approvals import _immutable_request, canonical_json

    validate(record)
    payload = {
        "authority_version": VERSION,
        "approval_id": record["approval_id"],
        "request_id": record["request_id"],
        "request": _immutable_request(record),
    }
    try:
        encoded = canonical_json(payload)
    except (UnicodeError, ValueError, TypeError):
        raise MeshRefusal(
            "network_enrollment_invalid", "The enrollment snapshot cannot be encoded canonically"
        ) from None
    return "sha256:" + hashlib.sha256(REQUEST_DOMAIN + encoded).hexdigest()


def checked(record: Mapping[str, Any], *, approval_id: str | None = None) -> dict[str, Any]:
    validate(record, approval_id=approval_id)
    require(
        record.get("request_digest") == commitment(record),
        "The network enrollment snapshot changed; review a fresh card",
    )
    return dict(record)


def decode(data: Any, approval_id: str) -> dict[str, Any]:
    require(isinstance(data, dict) and set(data) == {"format", "version", "record"})
    require(
        data["format"] == FORMAT and type(data["version"]) is int and data["version"] == VERSION
    )
    require(isinstance(data["record"], dict))
    return checked(data["record"], approval_id=approval_id)


def encode(record: Mapping[str, Any]) -> dict[str, Any]:
    return {"format": FORMAT, "version": VERSION, "record": checked(record)}


def ready_evidence(report: Any, *, device_id: str, network_id: str) -> list[dict[str, str]]:
    """Closed known-producer evidence, NOT filtering an empty set of failures.

    Identity/network/member positives also come from the runner's fresh pinned
    SSH identity + both-sided active-join check. This requires live targeted
    reachability, build and operator authority from the real readiness producer.
    Git/model/tool shortcomings remain visible, never healed with account access.
    """
    require(
        isinstance(report, dict)
        and report.get("identity_present") is True
        and type(report.get("ok")) is bool
        and "schema" not in report
        and "version" not in report
    )
    rows = _array(report.get("checks"))
    require(isinstance(rows, list) and bool(rows))
    allowed = {
        "build",
        "operator_authority",
        "git_identity",
        "model_credential",
        "mcp_servers",
        "mcp_credential",
        "tooling",
        "selected_model_credential",
    }
    required: set[str] = set()
    warnings: list[dict[str, str]] = []
    for row in rows:
        require(
            isinstance(row, dict)
            and type(row.get("ok")) is bool
            and isinstance(row.get("class"), str)
            and row["class"] in {"admission", "equipment"}
        )
        check = row.get("check")
        require(
            isinstance(check, str)
            and check in {"identity", "network", "membership", "reachability", "readiness"}
        )
        if check in {"identity", "network", "membership"}:
            if check == "identity" or row.get("network_id") == network_id:
                require(row["ok"] is True)
            continue
        if check == "readiness":
            require(isinstance(row.get("capability"), str) and row["capability"] in allowed)
        if row.get("device_id") != device_id:
            continue
        if check == "reachability" and row["ok"] is True:
            observed = row.get("observed")
            require(
                row["class"] == "admission"
                and isinstance(observed, dict)
                and observed.get("outcome") in {"connected", "connected_link", "connected_unpinned"}
            )
            required.add("reachability")
        if check == "readiness":
            capability = row["capability"]
            if capability in {"build", "operator_authority"}:
                require(
                    row["ok"] is True
                    and isinstance(row.get("code", ""), str)
                    and row.get("code") not in {"unknown", "not_asked", "peer_too_old"}
                )
                require(row["class"] == ("admission" if capability == "build" else "equipment"))
                # Both positive producers use SOURCE_PEER: build_row composes
                # the peer's handshake stamp; operator_row uses its fresh facts.
                require(row.get("source") == "peer")
                required.add(capability)
            elif row["ok"] is False:
                warnings.append(
                    {
                        "capability": capability,
                        "code": str(row.get("code") or ""),
                        "detail": str(row.get("detail") or "")[:500],
                    }
                )
    require(
        required == {"reachability", "build", "operator_authority"},
        "Enrollment verification lacks positive targeted reachability, build or operator evidence",
    )
    return warnings


def public_report(data: Mapping[str, Any]) -> dict[str, Any]:
    """Bounded NEW-kind observation, never authority or a credential diagnostic.

    Legacy terminal receipts cap prose at400 and store no step data. Preserve
    that format unchanged: only the new envelope's final receipt carries these
    allowlisted facts, associated with its existing run_id. This prevents a
    reopened UI from guessing ready_ok from truncated prose or connected state.
    """
    from local_operator.network import readiness

    data = _object(data)
    codes = {
        value
        for key, value in vars(readiness).items()
        if key.startswith("CODE_") and isinstance(value, str)
    } | {""}
    require(
        data.get("enrollment_verified") is True
        and type(data.get("ready_ok")) is bool
        and data.get("task_ready") is False
    )
    rows = _array(data.get("equipment_warnings"))
    require(isinstance(rows, list) and len(rows) <= 32)
    public = []
    for row in rows:
        require(
            isinstance(row, dict)
            and row.get("capability")
            in {
                "git_identity",
                "model_credential",
                "mcp_servers",
                "mcp_credential",
                "tooling",
                "selected_model_credential",
            }
            and row.get("code", "") in codes
        )
        public.append({"capability": row["capability"], "code": row.get("code", "")})
    return {
        "enrollment_verified": True,
        "ready_ok": data["ready_ok"],
        "task_ready": False,
        "equipment_warnings": public,
    }


def failure_report(
    source: Mapping[str, Any], data: Mapping[str, Any], *, root: Any
) -> dict[str, Any]:
    """Only a coherent restricted replacement ID, never a raw contradiction record.

    Missing/expired/unrelated replacements yield a new-review action, not an
    invented retry. This is local navigation evidence, not signed authority;
    following it can NEVER approve or execute the replacement.
    """
    from local_operator.network import approvals

    require(source.get("kind") == KIND and data.get("failure_kind") == "changed_facts")
    candidate = data.get("replacement_id")
    replacement = None
    if (
        isinstance(candidate, str)
        and ID.fullmatch(candidate)
        and candidate != source.get("approval_id")
    ):
        try:
            target = approvals.load_record(candidate, root=root)
            same = all(
                target.get(k) == source.get(k) for k in ("what", "requested_by", "credential_ref")
            )
            old_device = dict(source.get("device") or {})
            new_device = dict(target.get("device") or {})
            changed_fp = new_device.pop("host_key_fp", None)
            old_fp = old_device.pop("host_key_fp", None)
            if (
                target.get("kind") == KIND
                and target.get("state") not in {"expired", "denied", "withdrawn"}
                and target.get("request_id") != source.get("request_id")
                and same
                and old_device == new_device
                and changed_fp
                and changed_fp != old_fp
            ):
                replacement = candidate
        except MeshRefusal:
            replacement = None
    return {"failure_kind": "changed_facts", "replacement_id": replacement}
