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
