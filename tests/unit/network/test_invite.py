"""Invites: minting, the tag, single use, expiry, and the device binding."""

from __future__ import annotations

import re
import time
from dataclasses import replace
from pathlib import Path

import pytest

from local_operator.network import invite as invite_mod
from local_operator.network import store, types, wire
from local_operator.network.handshake import pair_timeout_seconds

NETWORK = "n_0123456789abcdef01234567"
MATERIAL = wire.b64u(b"s" * 32)


def _record(**overrides: object) -> types.NetworkRecord:
    record = types.NetworkRecord(
        network_id=NETWORK,
        name="home-net",
        epoch=3,
        self_device_id="d_" + "a" * 32,
        self_role="admin",
        self_capabilities=sorted(types.capabilities_for_role("admin")),
        listen={"address": "127.0.0.1", "port": 4097, "advertised": ["127.0.0.1:4097"]},
    )
    for key, value in overrides.items():
        setattr(record, key, value)
    return record


def test_mint_encode_decode_round_trip() -> None:
    record = _record()
    minted = invite_mod.mint(record, MATERIAL, role="drive", ttl_s=600.0, now=100.0)
    assert minted.token.startswith("lop1.")
    envelope = invite_mod.decode(minted.token)
    assert envelope.network_id == NETWORK
    assert envelope.invite_id == minted.record.invite_id
    assert envelope.role == "drive"
    assert envelope.material == MATERIAL
    assert envelope.hosts == ["127.0.0.1:4097"]
    invite_mod.check_tag(
        minted.token,
        envelope,
        wire.invite_key(MATERIAL, NETWORK, minted.record.invite_id),
    )


def test_capabilities_are_resolved_at_mint_and_carried() -> None:
    """A later change to ``ROLE_CAPABILITIES`` must not widen an existing invite."""
    minted = invite_mod.mint(_record(), MATERIAL, role="read", ttl_s=60.0)
    assert set(minted.record.capabilities) == set(types.capabilities_for_role("read"))
    assert "prompt" not in minted.record.capabilities


def test_an_unknown_role_is_refused() -> None:
    with pytest.raises(types.PairingRefusal):
        invite_mod.mint(_record(), MATERIAL, role="operator", ttl_s=60.0)


def test_a_tampered_token_fails_its_tag() -> None:
    """The negative case: rewrite the payload's network id and the MAC no longer
    verifies, which is what makes the token unforgeable without the material."""
    minted = invite_mod.mint(_record(), MATERIAL, role="drive", ttl_s=60.0)
    envelope = invite_mod.decode(minted.token)
    forged = invite_mod.InviteEnvelope(
        network_id="n_ffffffffffffffffffffffff",
        network_name=envelope.network_name,
        epoch=envelope.epoch,
        material=envelope.material,
        inviter_device_id=envelope.inviter_device_id,
        inviter_name=envelope.inviter_name,
        invite_id=envelope.invite_id,
        issued_at=envelope.issued_at,
        ttl_s=envelope.ttl_s,
        role=envelope.role,
        capabilities=list(envelope.capabilities),
        hosts=list(envelope.hosts),
    )
    forged_token = invite_mod.encode(forged, MATERIAL)
    with pytest.raises(types.PairingRefusal) as excinfo:
        invite_mod.check_tag(
            forged_token,
            invite_mod.decode(forged_token),
            wire.invite_key(MATERIAL, NETWORK, envelope.invite_id),
        )
    assert excinfo.value.code == invite_mod.REASON_TAG


def test_a_token_whose_invite_id_changed_fails_its_tag() -> None:
    """Re-labelling a token with another invite's id changes the derived key."""
    minted = invite_mod.mint(_record(), MATERIAL, role="drive", ttl_s=60.0)
    parts = minted.token.split(".")
    envelope = invite_mod.decode(minted.token)
    forged = invite_mod.InviteEnvelope(
        network_id=envelope.network_id,
        network_name=envelope.network_name,
        epoch=envelope.epoch,
        material=envelope.material,
        inviter_device_id=envelope.inviter_device_id,
        inviter_name=envelope.inviter_name,
        invite_id="different-id",
        issued_at=envelope.issued_at,
        ttl_s=envelope.ttl_s,
        role=envelope.role,
    )
    forged_token = invite_mod.encode(forged, MATERIAL)
    assert len(parts) == 3
    with pytest.raises(types.PairingRefusal):
        invite_mod.check_tag(
            forged_token,
            invite_mod.decode(forged_token),
            wire.invite_key(MATERIAL, NETWORK, envelope.invite_id),
        )


def test_nonsense_tokens_are_refused_with_a_named_reason() -> None:
    for text in ("", "hello", "lop1.only-two-parts", "lop2.a.b"):
        with pytest.raises(types.PairingRefusal) as excinfo:
            invite_mod.decode(text)
        assert excinfo.value.code in (invite_mod.REASON_INVALID, "invite_version")


def test_open_token_needs_a_network_this_device_has() -> None:
    minted = invite_mod.mint(_record(), MATERIAL, role="drive", ttl_s=60.0)
    with pytest.raises(types.PairingRefusal) as excinfo:
        invite_mod.open_token(minted.token, material_for=lambda envelope: None)
    assert excinfo.value.code == invite_mod.REASON_TAG
    envelope = invite_mod.open_token(minted.token, material_for=lambda envelope: MATERIAL)
    assert envelope.invite_id == minted.record.invite_id


# ---------------------------------------------------------------------------
# Single use, expiry, binding
# ---------------------------------------------------------------------------


def _with_invite(**overrides: object) -> tuple[types.NetworkRecord, types.InviteRecord]:
    record = _record()
    kwargs: dict[str, object] = {"role": "drive", "ttl_s": 600.0, "now": 100.0}
    kwargs.update(overrides)
    minted = invite_mod.mint(record, MATERIAL, **kwargs)  # type: ignore[arg-type]
    record.invites.append(minted.record)
    return record, minted.record


def test_a_second_redemption_of_a_live_invite_is_refused() -> None:
    record, _invite = _with_invite()
    invite_id = record.invites[0].invite_id
    first = invite_mod.claim(record, invite_id, device_id="d_" + "b" * 32, epoch=3, now=110.0)
    assert first.capabilities == frozenset(types.capabilities_for_role("drive"))
    invite_mod.mark_redeemed(record, invite_id, device_id="d_" + "b" * 32, now=110.0)
    with pytest.raises(types.PairingRefusal) as excinfo:
        invite_mod.claim(record, invite_id, device_id="d_" + "b" * 32, epoch=3, now=111.0)
    assert excinfo.value.code == invite_mod.REASON_IN_USE


def test_a_consumed_invite_is_dead_even_for_the_same_device() -> None:
    record, _invite = _with_invite()
    invite_id = record.invites[0].invite_id
    invite_mod.mark_redeemed(record, invite_id, device_id="d_" + "b" * 32, now=110.0)
    invite_mod.consume(record, invite_id, outcome="sas_mismatch", now=111.0)
    with pytest.raises(types.PairingRefusal) as excinfo:
        invite_mod.claim(record, invite_id, device_id="d_" + "b" * 32, epoch=3, now=112.0)
    assert excinfo.value.code == invite_mod.REASON_USED


def test_expiry_follows_the_minting_clock_only() -> None:
    """The token carries a DURATION, so no cross-host clock comparison exists.

    The inviter enforces freshness against its own clock — ``minted_at + ttl_s`` —
    and a joiner whose clock is an hour wrong is unaffected, because it enforces
    nothing.
    """
    record, _invite = _with_invite(ttl_s=600.0)
    invite_id = record.invites[0].invite_id
    with pytest.raises(types.PairingRefusal) as excinfo:
        invite_mod.claim(record, invite_id, device_id="d_" + "b" * 32, epoch=3, now=100.0 + 601.0)
    assert excinfo.value.code == invite_mod.REASON_EXPIRED
    # One second earlier it is still good, so the boundary is where it is claimed.
    assert invite_mod.claim(record, invite_id, device_id="d_" + "b" * 32, epoch=3, now=699.0)


def test_an_epoch_rotation_invalidates_the_invite() -> None:
    record, _invite = _with_invite()
    record.epoch = 4
    with pytest.raises(types.PairingRefusal) as excinfo:
        invite_mod.claim(
            record, record.invites[0].invite_id, device_id="d_" + "b" * 32, epoch=3, now=110.0
        )
    assert excinfo.value.code == invite_mod.REASON_EPOCH


def test_a_device_bound_invite_refuses_any_other_device() -> None:
    """The optional binding: a token stolen in transit is worthless to a different
    device when the invite names one — and the attempt is TERMINAL.

    ``claim`` alone refuses and leaves the invite standing (that is what lets an
    ordinary failure be retried); the inviter's path goes through
    :func:`claim_or_consume`, which is where design §5.1/§5.4's rule lives: a bound
    invite presented by another device IS the leak the binding exists to contain,
    so it is consumed rather than left redeemable.
    """
    record, _invite = _with_invite(device_id="d_" + "c" * 32)
    invite_id = record.invites[0].invite_id
    with pytest.raises(types.PairingRefusal) as excinfo:
        invite_mod.claim_or_consume(
            record, invite_id, device_id="d_" + "b" * 32, epoch=3, now=110.0
        )
    assert excinfo.value.code == invite_mod.REASON_DEVICE
    assert record.invites[0].state == "consumed"
    # The named device is now too late as well: consuming is terminal, which is
    # the price the design accepts for a binding that actually contains a leak.
    with pytest.raises(types.PairingRefusal) as excinfo:
        invite_mod.claim(record, invite_id, device_id="d_" + "c" * 32, epoch=3, now=110.0)
    assert excinfo.value.code == invite_mod.REASON_USED


def test_claim_alone_leaves_the_invite_standing() -> None:
    """``claim`` is the VALIDATOR, and a refused attempt does not burn the invite.

    That property is what makes an honest retry work after an ordinary failure, and
    it is deliberately unchanged: the one TERMINAL refusal in this family lives in
    ``claim_or_consume``.
    """
    record, _invite = _with_invite(device_id="d_" + "c" * 32)
    invite_id = record.invites[0].invite_id
    with pytest.raises(types.PairingRefusal) as excinfo:
        invite_mod.claim(record, invite_id, device_id="d_" + "b" * 32, epoch=3, now=110.0)
    assert excinfo.value.code == invite_mod.REASON_DEVICE
    assert record.invites[0].state == "minted"


def test_the_binding_travels_in_the_envelope() -> None:
    record = _record()
    minted = invite_mod.mint(record, MATERIAL, role="drive", ttl_s=60.0, device_id="d_" + "c" * 32)
    envelope = invite_mod.decode(minted.token)
    assert envelope.device_id == "d_" + "c" * 32
    assert envelope.payload()["device_id"] == "d_" + "c" * 32


def test_a_redemption_for_an_invite_this_device_never_minted_is_refused() -> None:
    record = _record()
    with pytest.raises(types.PairingRefusal) as excinfo:
        invite_mod.claim(record, "never-minted", device_id="d_" + "b" * 32, epoch=3)
    assert excinfo.value.code == invite_mod.REASON_INVALID


def test_the_token_is_written_to_a_0600_file_and_not_returned_by_the_store(root: Path) -> None:
    """The CLI's contract, held at the store: the token has a FILE, never stdout."""
    record = _record()
    minted = invite_mod.mint(record, MATERIAL, role="drive", ttl_s=60.0)
    path = store.save_invite_token(minted.record.invite_id, minted.token, root)
    assert path.exists()
    assert path.read_text(encoding="utf-8").strip() == minted.token
    assert oct(path.stat().st_mode & 0o777) == "0o600"


def test_consume_before_the_announcing_frame_makes_a_replay_impossible() -> None:
    """``consumed`` is written BEFORE the frame that announces it, on both sides."""
    record, _invite = _with_invite()
    invite_id = record.invites[0].invite_id
    invite_mod.mark_redeemed(record, invite_id, device_id="d_" + "b" * 32)
    invite_mod.consume(record, invite_id, outcome="admitted")
    assert record.invites[0].state == "consumed"
    assert record.invites[0].outcome == "admitted"


def _printed_window(prompt: str) -> int:
    """The seconds the joiner's prompt promises, read off the sentence a person reads."""
    match = re.search(r"you have (\d+)s", prompt)
    assert match is not None, prompt
    return int(match.group(1))


def test_the_prompt_promises_the_window_the_joiner_actually_gets() -> None:
    """An AGED token: the number printed is what is LEFT, not what was minted.

    Measured through these same functions on the pre-fix code at 480 s of invite age:
    the prompt promised 180 s while the joiner's own read waited 120 s, and at 560 s it
    promised 40 s. A prompt that overstates the window is worse than no window at all,
    because a person budgets their attention on it (agent review round 1, MAJOR 2). The
    prompt and the read are one number now, and this cell reads BOTH — a fresh-token cell
    alone cannot see the drift, which is why it shipped.
    """
    record = _record()
    minted = invite_mod.mint(record, MATERIAL, role="drive", ttl_s=600.0)

    # 480 s into a 600 s token: the read waits 120 s, and the minted duration the prompt
    # used to print is 180 s. The tolerance is one second because the prompt recomputes
    # "now" a moment after this line does.
    aged = replace(minted.envelope, issued_at=time.time() - 480.0)
    read = pair_timeout_seconds(invite_mod.remaining_seconds(aged))
    prompt = invite_mod.joiner_prompt(aged, "481926", "K7QM-3XPD")
    printed = _printed_window(prompt)
    assert abs(printed - read) <= 1, prompt
    assert printed < 180, prompt  # the pre-fix prompt printed the MINTED duration
    # Both halves of the sentence agree with each other, so the words can never again
    # describe a longer window than the digits beside them.
    assert f"({invite_mod._minutes(printed)})" in prompt, prompt  # noqa: SLF001

    # …and nearer the end of its life the promise shrinks with it, still matching the
    # read rather than the minted duration.
    nearly_done = replace(minted.envelope, issued_at=time.time() - 560.0)
    late = _printed_window(invite_mod.joiner_prompt(nearly_done, "481926", "K7QM-3XPD"))
    assert abs(late - pair_timeout_seconds(invite_mod.remaining_seconds(nearly_done))) <= 1
    assert late < 60, late

    # A FRESH token still promises the full confirm budget: the fix must not shorten the
    # ordinary window, only stop the prompt from promising more than remains.
    fresh = _printed_window(invite_mod.joiner_prompt(minted.envelope, "481926", "K7QM-3XPD"))
    assert fresh == int(pair_timeout_seconds(600.0)), fresh


def test_a_delay_does_not_spend_a_code_guess(root: Path) -> None:
    """``attempts`` bounds GUESSES, so a timeout costs the token nothing.

    Before this, one timeout counted and three were the budget: two slow humans plus one
    typo exhausted the three forgiven failures the typo path exists to have, and the
    person at the OTHER device had to mint a fresh invite (agent review round 1, NIT 2).
    """
    record, invite = _with_invite()
    invite_id = invite.invite_id
    for _ in range(5):
        invite_mod.release(record, invite_id, outcome="timeout", spent_an_attempt=False)
    stored = record.invites[0]
    assert stored.attempts == 0, stored
    assert stored.state == "minted", stored
    assert not invite_mod.failures_exhausted(record, invite_id)

    # A compared-and-disagreed code is the one that spends it, and three of those still
    # leave the token usable while the fourth reports the budget spent.
    for expected in (1, 2, 3):
        invite_mod.release(record, invite_id, outcome="sas_mismatch", spent_an_attempt=True)
        assert record.invites[0].attempts == expected
    assert invite_mod.failures_exhausted(record, invite_id)


def test_the_prompt_names_the_code_and_the_fingerprint() -> None:
    record = _record()
    minted = invite_mod.mint(record, MATERIAL, role="drive", ttl_s=60.0)
    prompt = invite_mod.joiner_prompt(minted.envelope, "481926", "K7QM-3XPD")
    assert "481 926" in prompt
    assert "K7QM-3XPD" in prompt
    inviter = invite_mod.inviter_prompt(
        minted.envelope, "d_" + "b" * 32, "damian-ec2", "481926", derived="731502"
    )
    # BOTH codes are shown, because the inviter's human is the comparator: their own
    # derivation is what the other screen must show. Printing only the transcribed
    # value left them with nothing to compare against (the defect this fixes).
    assert "YOUR screen shows 731 502" in inviter
    assert "481 926" in inviter
    assert "damian-ec2" in inviter
