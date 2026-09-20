"""Operator authority: the operator, or the operator's device, verified offline.

WHAT THIS PACKAGE REPLACES. The first revision of issue #1310 gave the loosening
authority to the process that SPAWNED the runtime. That closed the hole — a
same-uid tool child could read the session record, dial the loopback control
socket and set its own gate to ``auto`` — but it cost capability the operator will
not accept:

* the phone could never loosen in any session, because the relay's spawn path
  never passes ``--operator-fd``, so its capability was always ``None``;
* a pane attached to a runtime another process started could not loosen;
* the desktop app could loosen only for sessions its own backend spawned;
* a supervised ``lop exec --control`` run could be denied but not approved;
* a background-engaged run parked a card nobody could answer for up to 24 h.

The authority is therefore a fact about the OPERATOR or the OPERATOR'S DEVICE:

    an operator key (ES256) whose private half sits in the OS presence store, so
    every signature costs a human gesture, and whose public half is pinned in a
    ROOT-OWNED file the gated subject cannot substitute.

The seam, the class predicate, the op set, the typed refusal and the proof
machinery all stay exactly where they were; what changed is what "authorised"
means. See ``docs/design/approval-authority.md`` §2.1-§2.4.

    >>> operator_authority_level()          # 'operator-presence' | ...
    >>> load_anchor().usable                # is the pinned key readable?
    >>> sign_challenge(...)                 # the one signing entry point

HONESTY IS A FEATURE OF THIS PACKAGE. Nothing here claims a boundary it cannot
demonstrate: a host with no presence store reports ``file-only``, a host with no
anchor reports ``spawn-capability-only``, and a host whose anchor is not
root-owned reports that instead of trusting it. ``operator_authority_level()`` is
the single answer to "how strong is authority on this host", and it ABSORBS the
earlier per-host capability guarantee rather than sitting beside it — two
answers to one question is how one of them ends up stale.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from local_operator.harness.approval import operator_cap_guarantee
from local_operator.operator.keychain import (
    CNG_PRESENCE,
    FILE_ONLY,
    SECURE_ENCLAVE,
    KeyBackendError,
    KeyHandle,
    choose_backend,
    default_file_path,
)
from local_operator.operator.sign import (
    Signature,
    anchor_for_handle,
    create_key,
    sign_challenge,
)
from local_operator.operator.trust import (
    AnchorCache,
    AnchorLoad,
    OperatorAnchor,
    anchor_bytes,
    anchor_dir,
    anchor_path,
    install_commands,
    is_presence_backend,
    load_anchor,
    load_staged_anchor,
    staging_path,
)
from local_operator.operator.verify import (
    ACTIONS,
    DeviceCert,
    key_id_for,
    signed_message,
    verify_device_cert,
    verify_signature,
)

logger = logging.getLogger(__name__)

#: The levels, as a closed set the seam and the copy both read.
#:
#: * ``operator-presence`` — a root-owned anchor whose private half costs a human
#:   gesture per signature. The level the design is arguing for.
#: * ``operator-file-only`` — a root-owned anchor whose private half is a 0600
#:   file: the anchor stands, but the PRESENCE claim does not, because any
#:   process running as the operator can sign without a gesture.
#: * ``anchor-unpinned`` — a file is installed where the anchor belongs and is
#:   not root-owned, so it is refused rather than trusted. Reported distinctly
#:   because it is either a mistake or an attack, and both need telling.
#: * ``spawn-capability-only`` — no anchor. The previous revision's model: only
#:   the process that spawned the runtime may loosen.
#: * ``unreported`` — the anchor could not be read at all.
LEVEL_OPERATOR_PRESENCE = "operator-presence"
LEVEL_OPERATOR_FILE_ONLY = "operator-file-only"
LEVEL_ANCHOR_UNPINNED = "anchor-unpinned"
LEVEL_SPAWN_ONLY = "spawn-capability-only"
LEVEL_UNREPORTED = "unreported"

#: Reported once per process, at the first runtime that asks (as the capability
#: guarantee was): the answer is a property of the host, and a daemon spawning
#: twenty runtimes must not write twenty identical lines into a bounded log.
_REPORTED = False


def authority_level_load(*, uid: int | None = None) -> tuple[str, AnchorLoad]:
    """The level plus the load that produced it, for callers that need the why."""
    loaded = load_anchor(uid)
    if loaded.exists and not loaded.root_owned:
        return LEVEL_ANCHOR_UNPINNED, loaded
    if not loaded.usable:
        return LEVEL_SPAWN_ONLY, loaded
    return (
        (
            LEVEL_OPERATOR_PRESENCE
            if loaded.anchor and loaded.anchor.presence
            else LEVEL_OPERATOR_FILE_ONLY
        ),
        loaded,
    )


def operator_authority_level(*, uid: int | None = None) -> str:
    """How strong THIS host's operator authority is. Absorbs the capability guarantee.

    The previous revision answered a narrower question — "does the OS stop a
    same-uid process from reading the spawn capability out of this process's
    memory?" (:func:`local_operator.harness.approval.operator_cap_guarantee`) —
    and the refusal copy and the record both quoted it. The question authority
    now turns on is bigger, because a loosening no longer requires that
    capability: it can also be a signature. This function is the ONE answer, and
    it folds the old one in rather than leaving a second, narrower level to be
    quoted out of context.

    Note what it does NOT do: it reads the anchor's own recorded backend rather
    than re-probing the OS. Probes drift from reality (a keychain that was
    available at init and is not at use time), and the anchor is the record of
    what was actually created — a mismatch surfaces as a failed signature and an
    honest refusal, not as an inflated level.
    """
    level, _ = authority_level_load(uid=uid)
    return level


def operator_authority_report(*, uid: int | None = None) -> dict[str, Any]:
    """The level with everything a report line needs, including the residual.

    ``capability_guarantee`` is the absorbed value: it answers the OLD question
    (same-uid memory reach), which still bounds how much the spawn path is worth
    on a host where its memory is readable. Keeping it visible here — rather than
    deleting it — is deliberate: a host can be ``operator-presence`` AND have a
    weak capability boundary, and a reader deserves both facts.
    """
    level, loaded = authority_level_load(uid=uid)
    return {
        "level": level,
        "reason": loaded.reason,
        "anchor_path": str(loaded.path),
        "anchor_installed": loaded.exists,
        "anchor_root_owned": loaded.root_owned,
        "backend": loaded.anchor.backend if loaded.anchor else "",
        "presence": bool(loaded.anchor and loaded.anchor.presence),
        "key_id": loaded.anchor.key_id if loaded.anchor else "",
        "capability_guarantee": operator_cap_guarantee(),
        "presence_enforced_by_os": bool(
            loaded.anchor and is_presence_backend(loaded.anchor.backend)
        ),
    }


def report_operator_authority(*, uid: int | None = None) -> str:
    """Log the authority level once per process and return it."""
    global _REPORTED
    level = operator_authority_level(uid=uid)
    if not _REPORTED:
        _REPORTED = True
        report = operator_authority_report(uid=uid)
        logger.info(
            "operator authority: %s (anchor %s: %s; spawn-capability guarantee %s)",
            level,
            report["anchor_path"],
            report["reason"],
            report["capability_guarantee"],
        )
        if level == LEVEL_OPERATOR_FILE_ONLY:
            logger.warning(
                "operator authority: the anchor's private half is a 0600 FILE, so any "
                "process running as this user can sign for the operator — loosening is "
                "authorised but NOT presence-gated on this host"
            )
    return level


def reset_reported_for_tests() -> None:
    """Clear the once-per-process flag. Tests only."""
    global _REPORTED
    _REPORTED = False


def default_config_root() -> Path:
    """The operator's config root, for key material that is not the anchor."""
    from local_operator.paths import config_dir

    return config_dir()


def load_signer_for_anchor(anchor: OperatorAnchor, *, config_root: Path) -> Any:
    """Load the private half the ANCHOR names, not the one the ladder prefers.

    A host that installed a Secure Enclave key and later lost the enclave must
    fail to sign rather than quietly sign with a file key the anchor does not
    name: the signature would not verify, and the operator would be told "no key"
    at the moment when the true answer is "your key is not reachable".
    """
    backend = choose_backend(anchor.backend, config_root=config_root)
    signer = backend.load()
    if signer is None:
        raise KeyBackendError(
            f"the anchor names a {anchor.backend} key that is not reachable on this host"
        )
    return signer


__all__ = [
    "ACTIONS",
    "AnchorCache",
    "AnchorLoad",
    "CNG_PRESENCE",
    "DeviceCert",
    "FILE_ONLY",
    "KeyBackendError",
    "KeyHandle",
    "LEVEL_ANCHOR_UNPINNED",
    "LEVEL_OPERATOR_FILE_ONLY",
    "LEVEL_OPERATOR_PRESENCE",
    "LEVEL_SPAWN_ONLY",
    "LEVEL_UNREPORTED",
    "OperatorAnchor",
    "SECURE_ENCLAVE",
    "Signature",
    "anchor_bytes",
    "anchor_dir",
    "anchor_for_handle",
    "anchor_path",
    "authority_level_load",
    "create_key",
    "default_config_root",
    "default_file_path",
    "install_commands",
    "is_presence_backend",
    "key_id_for",
    "load_anchor",
    "load_signer_for_anchor",
    "load_staged_anchor",
    "operator_authority_level",
    "operator_authority_report",
    "report_operator_authority",
    "reset_reported_for_tests",
    "sign_challenge",
    "signed_message",
    "staging_path",
    "verify_device_cert",
    "verify_signature",
]
