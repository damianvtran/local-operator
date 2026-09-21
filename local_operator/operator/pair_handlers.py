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

    ``--authorise`` is the INVERSE VERB and it exists because ``--revoke`` had
    none (R9-2/Q9-1): the relay records a revocation in two places, and the local
    half had no shipped way to remove an entry, so a revoked device could not be
    brought back by any route the product named — following the route it did name
    (create and install a new anchor) left the phone refused. Both halves are
    lifted here, and the same privileged install step carries the anchor half, so
    this verb is exactly as host-side-only as ``--revoke`` is: the phone has no
    reach into either.
    """
    from local_operator.operator.handlers import install_anchor

    root = config_dir()
    revoke = str(getattr(args, "revoke", "") or "")
    authorise = str(getattr(args, "authorise", "") or "")

    paired = devices.list_devices(root)
    pending = devices.list_pending(root)

    if revoke:
        if not _stage_anchor_revocation(root, revoke, revoked=True):
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
            f"running picks it up within {int(ANCHOR_REFRESH_S)}s and a new one at once. "
            "To bring this device back, run "
            f"`{devices.AUTHORISE_COMMAND.format(device_id=revoke)}` (needs the same "
            "install step)."
        )

    if authorise and not bool(getattr(args, "print_only", False)):
        # THE TWO HALVES, and every FACT is read before either is changed, so the
        # receipt can say what was true rather than what the code just did: running
        # `--authorise` twice would otherwise report the same lift twice. Local record
        # first, then the anchor's entry through the same privileged install step
        # `--revoke` uses (host-side only, like every other half of this).
        recorded_here = devices.is_revoked_here(root, authorise)
        recorded_in_anchor = _anchor_revokes(root, authorise)

        cleared = devices.forget_revocation(root, authorise)
        if recorded_here and not cleared:
            # The record named this device and its entry is still there: say the clear
            # did not happen (agent review round 10, NIT-1) — and do not let the receipt
            # below claim the lift anyway (round 11, U6/Q11-1/R11-1).
            print(
                f"the local revocation record still names {authorise} — it could not be "
                "rewritten. Check the permissions on the operator directory under your "
                "config root and run this again.",
                file=sys.stderr,
            )
        staged = _stage_anchor_revocation(root, authorise, revoked=False)
        if not recorded_here and not recorded_in_anchor:
            if cleared:
                # The record was there but did not APPLY (it is stamped with a key that
                # is no longer installed) and it has now gone: "nothing recorded a
                # revocation" would be wrong about a record that did exist (NIT-2).
                print(
                    f"a record naming {authorise} was on disk but did not apply under the "
                    "installed anchor; it is now removed, and nothing was refusing this "
                    "device."
                )
            else:
                print(f"nothing recorded a revocation of {authorise} on this machine.")
        elif staged:
            code = install_anchor(root, print_only=False)
            if code != 0:
                return code
            from local_operator.operator.trust import ANCHOR_REFRESH_S

            if recorded_here and not cleared:
                # THE HALF THAT DID NOT HAPPEN (round 11, U6 = Q11-1, R11-1). The success
                # sentence was printed unconditionally, so with the operator root
                # unwritable the same run said "a new pairing request is accepted at once"
                # two lines above a row listing the device as revoked — and the phone was
                # still refused (403 on a fresh code, measured by QA and the reviewer).
                # `is_revoked` is the OR of the two halves, so the local record alone
                # still refuses this device; the sentence says that instead.
                print(
                    f"the anchor no longer revokes {authorise}, but the local record still "
                    "does — so this device is STILL refused. Fix the permissions on the "
                    "operator directory under your config root, then run this again to "
                    "finish it."
                )
            else:
                print(
                    f"authorised {authorise}. A session already running picks this up within "
                    f"{int(ANCHOR_REFRESH_S)}s; a new pairing request is accepted at once. "
                    "Pair the phone again with `lop pair`."
                )
        else:
            if recorded_here and not cleared:
                print(
                    "no installed operator anchor to lift a revocation from, and the local "
                    "record could not be cleared either, so this device is STILL refused — "
                    "run `lop operator install` on this machine and fix the permissions on "
                    "the operator directory, then run this again",
                    file=sys.stderr,
                )
            else:
                print(
                    "no installed operator anchor to lift a revocation from, so only the "
                    "local record was cleared — run `lop operator install` on this machine "
                    "if you expected an anchor here",
                    file=sys.stderr,
                )

    elif authorise:
        # A PREVIEW ACTS ON NOTHING, AND STILL PRINTS WHAT THE FLAG PROMISES
        # (UX round 11, U5; UX/QA/review round 12, U7 = Q12-1 = R12-1). Round 11's fix
        # was right about the first half and lost the second: the local clear sat above
        # the old `print_only` branch, so the dry run mutated the state — and the early
        # `return` that replaced it also removed the privileged command the flag's own
        # help advertises ("print the privileged command instead of running it"), the
        # listing rows below, and the next step. Measured by both roles: 0 `sudo install`
        # lines against `--revoke --print-only`'s one, and no device table at all.
        #
        # So this arm writes nothing and PRINTS the three things: what it would do, the
        # privileged command, and the route to actually take it. It deliberately does not
        # `return`, so the listing at the end of this verb runs exactly as it does for
        # every other invocation.
        #
        # AND IT DESCRIBES THE RUN THIS VERB WILL ACTUALLY PERFORM IN THIS STATE (design
        # round 13, D1 — MAJOR). The first version promised a statement write and a
        # privileged step unconditionally, and with a local record but NO USABLE ANCHOR
        # the real run does neither: it clears the local record, adds nothing to an
        # anchor it cannot read, and says "run `lop operator install`". A preview that
        # promises more than the run performs is the class of defect rounds 10–12 were
        # about, so the branch below is keyed on the same fact the real path keys on.
        #
        # WHY DESCRIBE RATHER THAN REFUSE, WHICH IS THE ONE JUDGEMENT CALL HERE. The
        # sibling `--revoke --print-only` refuses (rc 1) in that state, and consistency
        # with it was one of the two options. It is the wrong one, because the real runs
        # of the two verbs differ there for a reason: a revocation that is not in the
        # anchor is not a revocation (the anchor is the list the runtime reads), while
        # CLEARING the local record really does lift the refusal — `is_revoked` is the OR
        # of the two halves, and with no usable anchor the local record is the only holder
        # left. Refusing here would strand an operator whose phone is refused and who does
        # not need a privileged install to fix it. What has to agree is the PREVIEW and
        # ITS OWN REAL RUN; both `--revoke` pairs agree too, in the other direction (its
        # real run refuses as well). The cell that pins this drives both and compares them.
        from local_operator.operator import staging_path
        from local_operator.operator.trust import load_anchor

        command = devices.AUTHORISE_COMMAND.format(device_id=authorise)
        recorded_here = devices.is_revoked_here(root, authorise)
        recorded_in_anchor = _anchor_revokes(root, authorise)
        loaded = load_anchor()
        anchor_usable = bool(loaded.usable and loaded.anchor is not None)
        staged_path = staging_path(root)

        if not recorded_here and not recorded_in_anchor:
            print(
                f"preview: nothing recorded a revocation of {authorise} on this machine, "
                f"so `{command}` would have nothing to lift and no privileged step "
                "would be needed."
            )
        elif not anchor_usable:
            print(
                f"preview: `{command}` would clear the local revocation record — "
                f"{authorise} could pair again — and do nothing else: there is no usable "
                "installed operator anchor on this machine, so no anchor statement would "
                f"be written and no privileged step would be needed. Run `{command}` "
                "without --print-only to do exactly that, and run `lop operator install` "
                "if you expected an anchor here."
            )
        else:
            halves = " and ".join(
                part
                for part, present in (
                    ("the local revocation record", recorded_here),
                    ("the anchor's revocation entry", recorded_in_anchor),
                )
                if present
            )
            print(
                f"preview: `{command}` would clear {halves}, write the anchor statement "
                f"without the entry to {staged_path}, and need ONE privileged step to "
                "install it."
            )
            if staged_path.exists():
                # THE CAVEAT TRAVELS WITH THE COMMAND (design round 13, D2): it used to sit
                # fifteen rows below it in a 44-column block, so a reader could copy the
                # command without ever meeting the correction.
                print(
                    "NOTHING has been changed by this run: the statement is not on disk "
                    "yet, so the command below would install the anchor unchanged."
                )
                install_anchor(root, print_only=True)
                print(
                    f"Next step: run `{command}` without --print-only — it writes the "
                    "statement and takes that step itself — then pair the phone again with "
                    "`lop pair`."
                )
            else:
                # A REMEDY ON THE SAME STREAM, AND NO INTERLEAVING (design round 13, D5).
                # Calling `install_anchor` here would print its own "nothing staged at …;
                # run `lop operator init` first" to STDERR — a fact about ITS caller
                # (`lop operator install`), not about this verb, whose real run stages the
                # lifted statement itself before it installs — and in a piped run that
                # stderr line arrives BEFORE this stdout block, with the old stdout line
                # then repeating the fact and naming no route at all. So: no call, one
                # remedy, in the right order, and the remedy is the route that exists.
                print(
                    "There is no staged anchor statement to install, so there is no "
                    f"privileged command to print yet: run `{command}` without "
                    "--print-only and it writes the statement and takes that step itself."
                )

    for device in paired:
        expires = time.strftime("%Y-%m-%d", time.localtime(device.not_after))
        state = "revoked" if _anchor_revokes(root, device.device_id) else "active"
        print(
            f"  {device.device_id}  {device.name or '(unnamed)':<20} "
            f"{state:<8} expires {expires}  scopes {','.join(device.scope)}"
        )
    # A REVOKED DEVICE KEEPS ITS ROW (UX round 10, U1). It had none: the revocation
    # clears the certificate, so it cannot be in `paired`, and nothing printed the
    # anchor-only revocations — which left the id every remedy names readable nowhere
    # but a receipt from an earlier day. The row carries the command, because being
    # brought back is the reason the device is listed at all.
    listed = {device.device_id for device in paired}
    for device_id in devices.revoked_ids(root):
        if device_id in listed:
            continue
        print(
            f"  {device_id}  {'':<20} {'revoked':<8} certificate cleared — "
            f"`{devices.AUTHORISE_COMMAND.format(device_id=device_id)}` brings it back"
        )
    for row in pending:
        print(
            f"  {row.get('device_id')}  {row.get('name') or '(unnamed)':<20} "
            "PENDING — run `lop pair --yes` to approve"
        )
    return 0


def _stage_anchor_revocation(root: Path, device_id: str, *, revoked: bool) -> bool:
    """Stage the current anchor with ``device_id`` marked revoked (or un-revoked).

    ``False`` when there is no installed anchor to change the revocation list of.

    Rebuilt from the installed anchor rather than from a fresh statement, so the
    operator's key, label and creation time survive; only the revocation list is
    added to. Un-revoking DROPS the entry rather than writing ``"revoked": False``
    — the list is a list of things that are revoked, and an entry that says
    "present, but not revoked" is a shape a future reader has to know to read.
    """
    from dataclasses import replace

    from local_operator.operator.handlers import stage_anchor
    from local_operator.operator.trust import load_anchor

    loaded = load_anchor()
    if not loaded.usable or loaded.anchor is None:
        return False
    entries = [dict(entry) for entry in loaded.anchor.devices]
    if revoked:
        for entry in entries:
            if entry.get("device_id") == device_id:
                entry["revoked"] = True
                break
        else:
            entries.append({"device_id": device_id, "revoked": True})
    else:
        entries = [entry for entry in entries if entry.get("device_id") != device_id]
    stage_anchor(root, replace(loaded.anchor, devices=tuple(entries)))
    return True


def _anchor_revokes(root: Path, device_id: str) -> bool:
    from local_operator.operator.trust import device_is_revoked, load_anchor

    loaded = load_anchor()
    if not loaded.usable or loaded.anchor is None:
        return False
    return device_is_revoked(loaded.anchor, device_id)


__all__ = ["POLL_INTERVAL_S", "describe_devices", "dispatch"]
