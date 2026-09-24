"""The ONE signing entry point for operator authority (issue #1310, revision 2).

WHAT THIS IS FOR. A console that did not spawn the runtime behind a session — an
attached pane, the desktop app for a session its backend did not start, the CLI
for a background-started run, the phone's relay (stage D) — may still loosen that
session's gate. It does so by signing a per-action challenge with the operator
key, and the boundary is not the CALLER'S IDENTITY: it is that only a human can
answer the OS prompt the signature requires.

    lop operator sign --challenge <hex> --purpose loosen --session <id>

WHY ONE ENTRY POINT RATHER THAN A LIBRARY EACH SURFACE CALLS. The prompt-copy
rule ("name the session and the effect"), the single-use challenge rule and the
once-per-action rule all have to hold identically for the TUI's attached pane,
the desktop backend and the CLI — and each of those is a different process with
its own idea of what it is doing. One function, with the copy built from the
purpose it was given, is the only shape in which they cannot drift.

IN-PROCESS USE IS ALLOWED, AND IS NOT A HOLE. The TUI and the desktop backend
call this directly instead of shelling out to the CLI. That is safe for exactly
one reason: the presence enforcement is in the OS call (``SecKeyCreateSignature``
on a ``kSecAccessControlUserPresence`` key, or CNG's ``NCRYPT_UI_POLICY``), so an
in-process caller gets the same prompt a subprocess would. What the subprocess
form adds is a process boundary that can be *observed*; it adds nothing to the
boundary itself. The residual — a model's tool child CAN invoke this and raise a
prompt — is recorded in the design document rather than papered over: a prompt
can be spammed and misread, only a human can answer it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from local_operator.operator.keychain import (
    FILE_ONLY,
    KeyBackendError,
    KeyHandle,
    Signer,
    choose_backend,
)
from local_operator.operator.trust import (
    OperatorAnchor,
    anchor_path,
    is_presence_backend,
)
from local_operator.operator.verify import (
    ACTIONS,
    decode_point,
    key_id_for,
    signed_message,
)


@dataclass(frozen=True)
class Signature:
    """What a surface gets back: the signature, and which key produced it.

    ``key_id`` travels with it so the runtime can refuse a signature whose key
    does not match the anchor BEFORE it spends a verification on it, and so a
    refusal can name the key it did not recognise.
    """

    sig: str
    key_id: str

    def as_json(self) -> dict[str, str]:
        return {"sig": self.sig, "key_id": self.key_id}


def effect_copy(*, purpose: str, session_id: str, request_id: str = "") -> str:
    """The ONE sentence naming what a signature is about to authorise.

    Built here rather than at each call site because the point of it is that a
    person can tell what they are approving: the design's residual risk is a
    prompt that gets spammed and misread, and a prompt that does not say which
    session it unlocks is exactly the prompt that gets misread. The CLI prints
    this before signing; the TUI and desktop show it in their own confirmation.
    """
    target = session_id or "this session"
    if purpose == "approve":
        card = f" card {request_id}" if request_id else " the parked approval"
        return f"Authorise the operator key to APPROVE{card} in {target}"
    return f"Authorise the operator key to LOOSEN the approval gate of {target}"


def sign_message(signer: Signer, message: bytes) -> Signature:
    """Sign one message and return the wire form.

    Split from :func:`sign_challenge` so the message construction is not
    duplicated by a caller that already has the bytes — and so a test can sign a
    message the runtime did not mint, which is how the replay tests build their
    forged frames.
    """
    signature = signer.sign(message)
    return Signature(sig=signature.hex(), key_id=signer.handle.key_id)


def load_signer(*, config_root: Path, backend_name: str | None = None) -> Signer | None:
    """The operator's private key, or ``None`` when there is none to load.

    ``backend_name`` comes from the anchor (so a host whose key was created as
    ``secure-enclave`` does not silently look in the file fallback); ``None``
    asks the presence ladder, which is what ``lop operator init`` does before an
    anchor exists.
    """
    backend = choose_backend(backend_name or "auto", config_root=config_root)
    loader: Any = getattr(backend, "load", None)
    if loader is None:  # pragma: no cover — every backend defines ``load``
        return None
    return loader()


def resolve_backend_name(config_root: Path) -> str:
    """Which backend holds the private half, in the order the facts allow.

    1. an INSTALLED, root-owned anchor names it — the authoritative answer;
    2. otherwise the STAGED anchor does — a hint that lets
       ``lop operator init --backend file-only && lop operator sign`` work before
       the privileged install step;
    3. otherwise the host's presence ladder.
    """
    from local_operator.operator.trust import load_anchor, load_staged_anchor

    installed = load_anchor()
    if installed.usable and installed.anchor is not None:
        return installed.anchor.backend
    staged = load_staged_anchor(config_root)
    if staged is not None:
        return staged.backend
    return "auto"


def sign_challenge(
    *,
    challenge: str,
    purpose: str,
    config_root: Path,
    session_id: str = "",
    request_id: str = "",
    backend_name: str | None = None,
) -> Signature:
    """Sign the runtime's challenge for one action. This call prompts.

    Raises :class:`KeyBackendError` when there is no usable key or when the OS
    refuses the gesture — never a silent ``None``, because the surfaces that call
    it must be able to tell "the operator said no" from "there is no key here",
    and only the second one has a remedy the copy can name.
    """
    if purpose not in ACTIONS:
        raise KeyBackendError(f"unknown signing purpose {purpose!r}")
    if not challenge:
        raise KeyBackendError("no challenge to sign")
    signer = load_signer(
        config_root=config_root,
        backend_name=backend_name or resolve_backend_name(config_root),
    )
    if signer is None:
        raise KeyBackendError("no operator key on this machine — run `lop operator init` first")
    message = signed_message(
        action=purpose,
        session_id=session_id,
        request_id=request_id,
        challenge=challenge,
    )
    try:
        return sign_message(signer, message)
    finally:
        signer.close()


def create_key(*, config_root: Path, preference: str = "auto") -> KeyHandle:
    """Create the operator's private key in the best store this host offers.

    Returns the handle so the caller can report the LEVEL it actually got: the
    difference between ``secure-enclave`` and ``file-only`` is the difference
    between a presence gate and none, and a setup step that did not say which one
    it achieved would be the exact overclaim the design forbids.
    """
    backend = choose_backend(preference, config_root=config_root)
    creator: Any = getattr(backend, "create", None)
    if creator is None:  # pragma: no cover — every backend defines ``create``
        raise KeyBackendError(f"backend {preference!r} cannot create a key")
    return creator()


def anchor_for_handle(handle: KeyHandle, *, label: str = "") -> OperatorAnchor:
    """The anchor a freshly created key warrants."""
    return OperatorAnchor(
        key_id=handle.key_id,
        spki=handle.spki,
        backend=handle.backend,
        presence=is_presence_backend(handle.backend),
        label=label,
        created_at=int(time.time()),
    )


def issue_device_cert(
    *,
    device_spki: bytes,
    device_id: str,
    label: str,
    signer: Signer,
    lifetime_s: int = 90 * 24 * 3600,
) -> str:
    """An operator-signed certificate for a device's public key (stage D's wire).

    Lives here rather than in the pairing code because the STATEMENT is the
    operator's and only the operator's key can make it: a device that could mint
    its own certificate would make the whole device tier decorative. The runtime
    side of the same format is :func:`local_operator.operator.verify.
    verify_device_cert`, so the two are pinned against each other by a test
    rather than by a reader's care.
    """
    from local_operator.operator.verify import DeviceCert

    now = int(time.time())
    if decode_point(device_spki) is None:
        raise KeyBackendError("a device certificate needs a P-256 device public key")
    cert = DeviceCert(
        device_id=device_id,
        spki=device_spki,
        label=label,
        issued_at=now,
        not_after=now + int(lifetime_s),
    )
    return cert.encode(signature=signer.sign(cert.payload()))


def describe_level(handle: KeyHandle) -> str:
    """One line, for ``lop operator init``'s output, that does not overclaim."""
    if handle.presence:
        return f"{handle.backend}: every signature requires a human gesture " "(the strong level)"
    if handle.backend == FILE_ONLY:
        return (
            "file-only: the key is a 0600 file under your config dir, so ANY "
            "process running as you can sign for you. This is NOT a boundary — "
            "it is reported as a lower level rather than counted as protection. "
            "Pair a phone (stage D) or use a host with a presence store."
        )
    return f"{handle.backend}: reported as-is; this build makes no claim about it"


def key_id_of(spki: bytes) -> str:
    """The identifier a frame's ``operator_key_id`` must carry for this key."""
    return key_id_for(spki)


def anchor_location() -> str:
    """Where the anchor must be installed for the runtime to trust it."""
    return str(anchor_path())
