"""Pattern guard coverage: what is refused, what stays storable, and that it is bounded.

The guard is a HEURISTIC, not proof — these tests pin the classes it claims, and the
benign neighbours it must not break (QA Q2), so a loosening or a new false positive
fails by name.
"""

import pytest

from local_operator.memory import (
    MemoryError,
    MemoryIdentity,
    MemoryStore,
    guard,
    validate_fact,
)


def refused(text: str, **kwargs) -> bool:
    try:
        guard(text, **kwargs)
    except MemoryError:
        return True
    return False


@pytest.mark.parametrize(
    "text",
    [
        "SSN 123-45-6789",  # plain control
        "SSN 1\u200b23-45-6789",  # zero-width space (QA Q1)
        "SSN 1\u200d23\u201045\u22126789",  # joiner + Unicode hyphen/minus
        "SSN \uff11\uff12\uff13-\uff14\uff15-\uff16\uff17\uff18\uff19",  # full-width digits
        "SSN 123\u00a045\u00a06789",  # NBSP separators
        "SSN 123 - 45 - 6789",
        "p\u0430ssport number AB1234567",  # Cyrillic a in the keyword
        "pass\u200bport AB1234567",
        "The staging password is hunter2",
        "My login is bob / pa55w0rd!",
        "password: Tr0ub4dor&3",
        "ssh passphrase correct horse battery staple",
        "IBAN GB82WEST12345698765432",
        "iban gb82 west 1234 5698 7654 32",
        "tax id 12-3456789",
        "Account number 12345678 at Barclays",
        "card 4111 1111 1111 1111",
        "card 4111-1111-1111-1111",
    ],
)
def test_sensitive_shapes_are_refused(text):
    assert refused(text), text


@pytest.mark.parametrize(
    "text",
    [
        "Build finished at 1792028832000",  # unix ms: 13 digits, not a card (Q2)
        "order 1234567890123 shipped",
        "tracking 9400111899223033005282",
        "password policy: rotate quarterly",
        "the password manager is 1Password",
        "uses secret GITHUB_TOKEN for CI",
        "login is SERVICE_ACCOUNT_NAME",
        "account owner is the platform team",
        "IBAN is required for payouts",
        "commit a3bf4f1b2c3d",
        "sha 0123456789abcdef0123456789abcdef01234567",
        "revision: 0123456789abcdef0123456789abcdef01234567",
    ],
)
def test_benign_neighbours_stay_storable(text):
    assert not refused(text, opaque=True), text


@pytest.mark.parametrize(
    "text",
    [
        "0123456789abcdef0123456789abcdef01234567",  # unnamed 40-hex = API-key shaped
        "token-ish aB3dE5gH7jK9mN1pQ3sT5vX7zA9cE1gI3kM5oQ7sU9wY",
    ],
)
def test_opaque_tokens_are_refused_for_stores_but_searchable(text):
    assert refused(text, opaque=True)
    assert not refused(text), "a search query may carry an id or SHA"


def test_search_of_a_long_numeric_id_is_not_refused(tmp_path):
    store = MemoryStore(tmp_path, MemoryIdentity("s1"), origin="dev")
    store.put("global", {"title": "Order", "body": "order 1234567890123 shipped"})
    assert store.search("order 1234567890123")


def test_guard_is_structurally_linear_on_pathological_input():
    # Wall-clock free: a bounded-quantifier regression would show as a quadratic count of
    # characters examined, so assert the structural bound instead of a timing.
    from local_operator import memory

    for pattern in (memory._ANCHORED, memory._ACCOUNT, memory._PROSE_SECRET, memory._SSN):
        assert (
            "+" not in pattern.pattern.replace("[+", "").replace("\\+", "").replace("{0,1}", "")
            or pattern is memory._ANCHORED
        ), pattern.pattern
    assert not refused("a" * 8000) and not refused("password " * 900)
    assert not refused("- " * 4000, opaque=True)


def test_validate_fact_refuses_with_a_name_reference_example():
    with pytest.raises(MemoryError, match=r"GITHUB_TOKEN"):
        validate_fact({"title": "t", "body": "SSN 123-45-6789"})
