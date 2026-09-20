"""``lop pair``'s verbs: the one place a device becomes a signer.

THE FLOW, and which step is allowed to do what:

1. mint a CODE (``devices.begin_pairing``) and print it — this process holds the
   operator's private key, so this is the process that will authorise;
2. wait for a device to claim the code through the relay's pairing endpoint,
   which leaves a PENDING request naming the device's public point. The relay
   checks the code and writes the request; that is the whole of its part, and it
   is a courier's part because it has no operator key and no way to obtain a
   signature;
3. the operator confirms — in this terminal, unless ``--yes`` — and the ONE
   presence-gated signing entry point produces the certificate;
4. the certificate is VERIFIED AGAINST OUR OWN KEY before it is written. A
   certificate we cannot verify is one the runtime will refuse, and writing it
   would leave a paired phone that silently cannot sign; failing here names the
   real problem at the moment it can still be fixed;
5. burn the code and drop the pending request, so one code is one device.

WHY THE CODE IS BURNED RATHER THAN REUSED FOR A RETRY. The code's whole job is
to bound which phone can ask. A code that survived its first use would let a
second phone pair on a gesture the operator already made for the first — and the
operator's memory of that gesture is one device, not two.

``--yes`` IS NOT A HOLE, and the reason it is still a flag worth having: on a
host with a presence store the signature prompts regardless, so ``--yes``
removes a redundant confirmation rather than a boundary. On a ``file-only`` host
the OS prompt does not exist, and ``--yes`` therefore does grant without a
gesture — which is exactly what ``lop operator status`` already reports about
that host, rather than a claim this flag quietly breaks.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from local_operator.operator import devices
from local_operator.operator.keychain import KeyBackendError
from local_operator.operator.sign import (
    issue_device_cert,
    load_signer,
    resolve_backend_name,
)
from local_operator.operator.verify import read_device_cert, verify_device_cert
from local_operator.paths import config_dir

#: How often the waiting loop looks for a claimed code. Named rather than
#: inlined so the test that drives this flow can state what it is racing.
POLL_INTERVAL_S = 0.25


def dispatch(args: argparse.Namespace) -> int:
    return _pair(args)


def _pending_target(root: Path, only: str) -> dict[str, object] | None:
    """The pending request to approve: the named one, or the oldest.

    ``only`` is a device id, and a requested id that is not pending is answered
    with ``None`` rather than falling back to the first request: an operator who
    named a device and silently got a different one would be approving something
    they did not read.
    """
    pending = devices.list_pending(root)
    if not pending:
        return None
    if not only:
        return pending[0]
    for row in pending:
        if row.get("device_id") == only:
            return row
    return None


def _confirm(row: dict[str, object]) -> bool:
    """Ask, on the terminal, naming what is being authorised (never just "ok?").

    THE PROMISE IS QUALIFIED AT THE PROMPT (UX round 8, U8-2). This is the moment
    consent is given, and on a host whose anchor is staged but not installed the
    sentence below promised authority that cannot be exercised yet — the receipt
    carried the qualification, which the operator reads AFTER answering. One
    predicate, one surface earlier, so the answer is given with the same facts the
    receipt states.
    """
    from local_operator.operator import operator_authority_unusable

    name = str(row.get("name") or "")
    device_id = str(row.get("device_id") or "")
    sys.stdout.write(
        f"\nA device is asking to become an operator signer:\n"
        f"  name      : {name or '(unnamed)'}\n"
        f"  device id : {device_id}\n"
        f"This lets that device APPROVE parked tool calls and LOOSEN this session's "
        f"approval gate, from anywhere it can reach this machine.\n"
    )
    if operator_authority_unusable():
        # Same wording as the receipt's qualification, because a reader who sees
        # both should not have to reconcile two sentences about one host.
        sys.stdout.write(
            "  ...once this machine's operator authority is installed: run "
            "`lop operator install` here (one privileged step).\n"
        )
    sys.stdout.flush()
    try:
        answer = input("Authorise it? [y/N] ").strip().lower()
    except EOFError:
        return False
    return answer in ("y", "yes")


def _pair(args: argparse.Namespace) -> int:
    root = config_dir()
    timeout = max(0.0, float(getattr(args, "timeout", 300.0)))
    code = devices.begin_pairing(root, ttl_s=max(int(timeout) + 60, devices.PAIRING_TTL_S))

    sys.stdout.write(
        "pairing code (valid for a few minutes, and consumed by the first device "
        "that uses it):\n\n"
        f"    {code}\n\n"
        "On your phone: open the mobile portal, choose Pair this phone, and enter "
        "the code above.\n"
    )
    sys.stdout.flush()
    if getattr(args, "code_only", False):
        return 0

    only = str(getattr(args, "device", "") or "")
    yes = bool(getattr(args, "yes", False))
    deadline = time.monotonic() + timeout
    row: dict[str, object] | None = None
    while True:
        row = _pending_target(root, only)
        if row is not None:
            break
        if time.monotonic() >= deadline:
            break
        time.sleep(POLL_INTERVAL_S)

    if row is None:
        sys.stdout.write(
            "\nNo device claimed the code before the wait ended. Nothing was paired; "
            "run `lop pair` again when the phone is ready.\n"
        )
        return 1

    if not yes and not _confirm(row):
        # A declined request is DROPPED, and the code is left live so the operator
        # can retry from the phone without minting a new one: declining is a
        # statement about this request, not about the code.
        devices.drop_pending(root, str(row.get("device_id") or ""))
        sys.stdout.write("\nDeclined. Nothing was paired.\n")
        return 1

    device_id = str(row.get("device_id") or "")
    spki = _decode_spki(str(row.get("spki") or ""))
    if spki is None:
        devices.drop_pending(root, device_id)
        print("the request named a public key this build cannot read; refused", file=sys.stderr)
        return 1

    try:
        signer = load_signer(config_root=root, backend_name=resolve_backend_name(root))
    except KeyBackendError as exc:  # pragma: no cover — a broken backend
        print(f"could not load the operator key: {exc}", file=sys.stderr)
        return 1
    if signer is None:
        print(
            "no operator key on this machine — run `lop operator init` (and "
            "`lop operator install`) first",
            file=sys.stderr,
        )
        return 1

    try:
        certificate = issue_device_cert(
            device_spki=spki,
            device_id=device_id,
            label=str(row.get("name") or ""),
            signer=signer,
        )
        operator_key_id = signer.handle.key_id
        operator_spki = signer.handle.spki
    except KeyBackendError as exc:
        print(f"could not sign the device certificate: {exc}", file=sys.stderr)
        return 1
    finally:
        signer.close()

    parsed = read_device_cert(certificate)
    verified = (
        None
        if parsed is None
        else verify_device_cert(
            certificate, operator_spki=operator_spki, now=int(time.time()), parsed=parsed
        )
    )
    if parsed is None or verified is None:
        # NEVER install a certificate the runtime would refuse. This can only
        # happen if the signer and the verifier disagree, which is a bug worth
        # surfacing here rather than as a phone whose signatures mysteriously
        # never verify.
        print(
            "the certificate this machine just signed does not verify against its own "
            "operator key; nothing was installed",
            file=sys.stderr,
        )
        return 1

    stored = devices.write_device_cert(
        root,
        certificate=certificate,
        parsed=parsed,
        operator_key_id=operator_key_id,
        name=str(row.get("name") or ""),
    )
    devices.drop_pending(root, device_id)
    devices.clear_pairing(root)

    # THE RECEIPT IS QUALIFIED BY THE HOST'S OWN STATE (UX round 6, U2). It used to
    # promise authority unconditionally, and on a host between `lop operator init`
    # (which only STAGES the anchor) and `lop operator install` — the default state
    # for anyone following the pairing instructions — that promise was false: the
    # runtime has no key to verify the device's signatures against, and refuses every
    # one of them (U1). The refusal now names the install step; the surface that
    # hands out the device has to as well, or the two disagree about the same fact.
    from local_operator.operator import operator_authority_unusable

    if operator_authority_unusable():
        promise = (
            "It can act once this machine's operator authority is installed: run\n"
            "  `lop operator install` here (one privileged step). Until then the runtime has\n"
            "  no key to verify this device's signatures against, and refuses every one of\n"
            "  them. Revoke it with `lop operator devices --revoke <device id>`.\n"
        )
    else:
        promise = (
            "The phone can now approve parked cards and loosen a running gate. Revoke it "
            "with `lop operator devices --revoke <device id>`.\n"
        )

    sys.stdout.write(
        f"\nPaired {stored.name or stored.device_id}.\n"
        f"  device id : {stored.device_id}\n"
        f"  scopes    : {', '.join(stored.scope)}\n"
        f"  expires   : {time.strftime('%Y-%m-%d', time.localtime(stored.not_after))}\n" + promise
    )
    return 0


def _decode_spki(encoded: str) -> bytes | None:
    """The request's public point, decoded and bounded before it is signed.

    Thin alias over the store's decoder, kept as a name here because the CALLER
    contract is what this module documents: a malformed key must fail before the
    operator makes a human gesture. One implementation, so the relay that
    accepted the request and this verb that signs it cannot disagree about which
    bytes were requested.
    """
    return devices.decode_spki(encoded)


def describe_devices(args: argparse.Namespace) -> int:
    """``lop operator devices`` — paired phones, pending requests, and revocation.

    Revocation is expressed in the ROOT-OWNED ANCHOR rather than in a file under
    the config root, and that is the whole point of it: a revocation list the
    revoked subject could edit is not a revocation, it is a note. Writing it goes
    through the same privileged install step as the anchor itself, so a device the
    operator has revoked cannot be un-revoked by the device.

    Until the staged anchor is installed, the revocation is INERT — the runtime
    keeps checking the anchor it can read — and this says so rather than claiming
    a revocation that has not taken effect.
    """
    from local_operator.operator.handlers import install_anchor

    root = config_dir()
    revoke = str(getattr(args, "revoke", "") or "")

    paired = devices.list_devices(root)
    pending = devices.list_pending(root)

    if revoke:
        if not _stage_anchor_with_revocation(root, revoke):
            print(
                "no installed operator anchor to record a revocation in — run "
                "`lop operator install` on this machine first (the anchor `lop operator "
                "init` staged is not trusted until it is installed)",
                file=sys.stderr,
            )
            return 1
        devices.record_revocation(root, revoke)
        code = install_anchor(root, print_only=bool(getattr(args, "print_only", False)))
        if code != 0:
            return code
        # WHAT THIS NOW SAYS, and why it changed with R6-1. The old sentence left the
        # operator with the impression that the install step alone was the whole
        # story, and it was: `AnchorCache` pinned the anchor and its revocation list
        # at first need, so a device revoked after a runtime started stayed honoured
        # by that runtime for as long as it lived. The cache now re-reads the anchor
        # under a bound (`trust.ANCHOR_REFRESH_S`), so the sentence can state the
        # window instead of implying a runtime restart is needed.
        from local_operator.operator.trust import ANCHOR_REFRESH_S

        print(
            f"revoked {revoke}. Revocation lives in the anchor; until the install step "
            "above completes it has not taken effect. Once installed, a session already "
            f"running picks it up within {int(ANCHOR_REFRESH_S)}s and a new one at once."
        )

    if not paired and not pending:
        print("no paired devices. Run `lop pair` to add one.")
        return 0
    for device in paired:
        expires = time.strftime("%Y-%m-%d", time.localtime(device.not_after))
        state = "revoked" if _anchor_revokes(root, device.device_id) else "active"
        print(
            f"  {device.device_id}  {device.name or '(unnamed)':<20} "
            f"{state:<8} expires {expires}  scopes {','.join(device.scope)}"
        )
    for row in pending:
        print(
            f"  {row.get('device_id')}  {row.get('name') or '(unnamed)':<20} "
            "PENDING — run `lop pair --yes` to approve"
        )
    return 0


def _stage_anchor_with_revocation(root: Path, device_id: str) -> bool:
    """Stage the current anchor with ``device_id`` marked revoked. ``False`` when
    there is no installed anchor to add a revocation to.

    Rebuilt from the installed anchor rather than from a fresh statement, so the
    operator's key, label and creation time survive; only the revocation list is
    added to.
    """
    from dataclasses import replace

    from local_operator.operator.handlers import stage_anchor
    from local_operator.operator.trust import load_anchor

    loaded = load_anchor()
    if not loaded.usable or loaded.anchor is None:
        return False
    entries = [dict(entry) for entry in loaded.anchor.devices]
    for entry in entries:
        if entry.get("device_id") == device_id:
            entry["revoked"] = True
            break
    else:
        entries.append({"device_id": device_id, "revoked": True})
    stage_anchor(root, replace(loaded.anchor, devices=tuple(entries)))
    return True


def _anchor_revokes(root: Path, device_id: str) -> bool:
    from local_operator.operator.trust import device_is_revoked, load_anchor

    loaded = load_anchor()
    if not loaded.usable or loaded.anchor is None:
        return False
    return device_is_revoked(loaded.anchor, device_id)


__all__ = ["POLL_INTERVAL_S", "describe_devices", "dispatch"]
