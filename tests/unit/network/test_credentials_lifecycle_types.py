"""Negative and positive cases for ``validate_lifecycle_frame``.

A lifecycle frame is metadata only, so the validator rejects (never truncates) and
runs before the frame is applied; each test pins one rejection.
"""

import pytest

from local_operator.network.credentials.types import validate_lifecycle_frame


def _frame(**extra):
    return {"event": "revoked", "credential": "openai", "contexts": ["a"], **extra}


def test_minimal_frame_passes():
    assert validate_lifecycle_frame(_frame()) is None


def test_oversized_frame_rejected():
    with pytest.raises(ValueError, match="bytes"):
        validate_lifecycle_frame(_frame(note="x" * 16_400))


def test_excess_depth_rejected():
    with pytest.raises(ValueError, match="nests deeper"):
        validate_lifecycle_frame(_frame(a={"b": {"c": {"d": {"e": 1}}}}))


def test_depth_four_is_allowed():
    # the top-level mapping is depth 1, so a -> b -> c is depth 4: the last allowed.
    validate_lifecycle_frame(_frame(a={"b": {"c": {"d": 1}}}))


def test_too_many_contexts_rejected():
    with pytest.raises(ValueError, match="contexts"):
        validate_lifecycle_frame(_frame(contexts=[str(i) for i in range(9)]))


def test_duplicate_keys_rejected():
    # A decoded mapping cannot hold two identical keys, so the ambiguity that can
    # reach the validator is a case/whitespace collision, including nested ones.
    with pytest.raises(ValueError, match="duplicate"):
        validate_lifecycle_frame({"event": "x", "Event": "y"})
    with pytest.raises(ValueError, match="duplicate"):
        validate_lifecycle_frame(_frame(meta={"id": 1, " ID": 2}))


@pytest.mark.parametrize(
    "key", ["secret", "access_token", "Password", "private_key", "bearer", "api-key"]
)
def test_secret_looking_key_rejected(key):
    with pytest.raises(ValueError, match="credential material"):
        validate_lifecycle_frame(_frame(meta={key: "v"}))


def _link_pair():
    """A dialer/listener ``LinkCrypto`` pair over one key schedule (as test_wire_crypto)."""
    import os

    from local_operator.network import wire

    keys = wire.link_keys(os.urandom(32), b"t" * 32, b"l" * 16)
    return wire.LinkCrypto(keys, role="dialer"), wire.LinkCrypto(keys, role="listener")


def test_decode_refuses_oversized_lifecycle_frame():
    from local_operator.network import wire

    dialer, listener = _link_pair()
    frame = _frame(op="net_broker", kind="lifecycle_announce", note="x" * 16_400)
    with pytest.raises(wire.LinkCryptoError, match="lifecycle frame"):
        listener.open(dialer.seal(frame)[4:])


def test_decode_passes_small_lifecycle_frame():
    dialer, listener = _link_pair()
    frame = _frame(op="net_broker", kind="lifecycle_announce")
    assert listener.open(dialer.seal(frame)[4:]) == frame


def test_decode_leaves_non_lifecycle_frame_untouched():
    """Discriminating: the same oversized body under another kind is not validated."""
    dialer, listener = _link_pair()
    frame = _frame(op="net_broker", kind="grant", note="x" * 16_400)
    assert listener.open(dialer.seal(frame)[4:]) == frame
