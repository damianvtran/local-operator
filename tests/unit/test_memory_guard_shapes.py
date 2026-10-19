"""Pattern guard coverage: what is refused, what stays storable, and that it is bounded.

The guard is a HEURISTIC, not proof — these tests pin the classes it claims, and the
benign neighbours it must not break (QA Q2), so a loosening or a new false positive
fails by name.
"""

import hashlib

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


@pytest.mark.parametrize(
    "device_id",
    [
        # Minted as ``d_`` + sha256 hex[:32] (network/identity); the hex after
        # ``d_`` can satisfy the IBAN shape + mod-97 check, so a default-origin
        # store must not be refused as if the id were user-supplied fact text.
        "d_cd474787b1adaa67f8cb5934196c4834",
        "d_ba65518d4cd0f218ddd960d22f3dc6c6",
    ],
)
def test_iban_shaped_device_id_is_structural_not_fact_text(tmp_path, device_id):
    store = MemoryStore(tmp_path, MemoryIdentity("s1"), origin=device_id)
    item = store.put("global", {"title": "Host", "body": "staging host is stage-2"})
    corrected = store.put(
        "global",
        {"title": "Host", "body": "staging host is stage-3"},
        entity=item["id"],
        expected_version=item["version"],
    )
    # A correction genuinely produced a NEW head while preserving the replaced
    # revision in history (the store contract). Comparing corrected["version"] to
    # item["version"] directly would be trivially true — both are independently
    # minted UUIDv7 revisions — so the meaningful check is that each revision
    # still resolves to its own body.
    assert corrected["version"] != item["version"]
    assert store.show("global", item["id"])["fact"]["body"] == "staging host is stage-3"
    assert (
        store.show("global", item["id"], version=item["version"])["fact"]["body"]
        == "staging host is stage-2"
    )


DEVICE_ID = "d_ef73a4312e5c4686196b30c035c8a05a"
UUID_ID = "019a5e3c-9f2b-7c41-8d5e-6f7a8b9c0d1e"


@pytest.mark.parametrize(
    "query",
    [
        # Minted as ``d_`` + sha256 hex[:32] (network/identity.device_id_for; the
        # truncation is the 128-bit id of §3.1). Each hex tail satisfies the IBAN
        # shape + mod-97 check by chance, which is what refused the read: search()
        # ran the free-text STORE guard over a query that IS an identifier.
        DEVICE_ID,
        "d_cd474787b1adaa67f8cb5934196c4834",
        "d_ba65518d4cd0f218ddd960d22f3dc6c6",
    ],
)
def test_a_minted_device_id_query_is_not_guarded_as_fact_text(tmp_path, query):
    """A recall whose query IS a minted identifier is a lookup, not fact text.

    search() persists nothing, so the exemption is scoped to this read: the
    journal and provider-payload copy of the same arguments stays guarded and
    fails closed (tests/unit/session/test_memory_privacy.py pins the bytes).
    """
    assert refused(query), "precondition: the free-text guard itself refuses the value"
    store = MemoryStore(tmp_path, MemoryIdentity("s1"), origin="dev")
    store.put("global", {"title": "Host", "body": "staging host is stage-2"})
    assert store.search(query) == []
    assert store.search("staging host"), "the same store still answers a prose query"


def test_the_uuid_reference_the_read_path_consumes_is_not_guarded(tmp_path):
    """The other minted reference shape (memory._id()), pinned at the same site.

    Not a fix but a boundary: a canonical UUID already passes the free-text guard
    (its hyphens break every shape rule), and listing it as a reference keeps a
    lookup by the ``SCOPE/ID`` the CLI prints from ever failing for being an id.
    """
    assert not refused(UUID_ID)
    store = MemoryStore(tmp_path, MemoryIdentity("s1"), origin="dev")
    store.put("global", {"title": "Host", "body": "staging host is stage-2"})
    assert store.search(UUID_ID) == []


@pytest.mark.parametrize(
    "query",
    [
        # The v1 rejection, pinned by name: each of these IS one safe CHARSET
        # segment, so a charset-only exemption (safe_segment) admitted all three
        # as if they were identifiers. The exemption is the minted-reference
        # SHAPE, so the real card/IBAN/SSN spelling keeps the refusal.
        "4111111111111111",
        "GB82WEST12345698765432",
        "123-45-6789",
    ],
)
def test_a_bare_secret_shaped_query_is_still_refused(tmp_path, query):
    assert refused(query)
    store = MemoryStore(tmp_path, MemoryIdentity("s1"), origin="dev")
    with pytest.raises(MemoryError, match="Sensitive value refused"):
        store.search(query)


@pytest.mark.parametrize(
    "query",
    [
        f"{DEVICE_ID} extra",  # a reference inside a multi-token query is prose
        "the SSN is 123-45-6789",
        "card 4111 1111 1111 1111",
        "my login is bob / pa55w0rd!",
    ],
)
def test_a_prose_query_around_a_secret_is_still_refused(tmp_path, query):
    assert refused(query)
    store = MemoryStore(tmp_path, MemoryIdentity("s1"), origin="dev")
    with pytest.raises(MemoryError, match="Sensitive value refused"):
        store.search(query)


def test_the_reference_classifier_admits_only_minted_identifier_shapes():
    """Both sides of the boundary the two query exemptions rest on, by name.

    Admitting too much re-opens the v1 rejection (a charset is not a shape);
    admitting too little restores the refusal on a benign identifier lookup. The
    admitted set is exactly the shapes this codebase mints as references, each
    one cited at its source; a shape nothing here mints as a reference is not a
    reference, which is why a bare 64-hex digest is excluded even though
    ``body_sha`` is spelled that way (it is an integrity value over the op
    payload, never a handle any surface hands back).
    """
    from local_operator.memory import _id, structural_reference
    from local_operator.network.identity import device_id_for, mint_instance_id

    for value in (
        DEVICE_ID,  # network.identity.device_id_for, held verbatim from the refusal
        device_id_for(b"synthetic-public-key"),  # the live minter, not an exemplar
        UUID_ID,  # memory._id()
        _id(),  # the live minter
    ):
        assert structural_reference(value), value
    for value in (
        "4111111111111111",  # card/Luhn
        "GB82WEST12345698765432",  # IBAN/mod-97
        "123-45-6789",  # SSN
        "e5b8f0a2c7d94f13b6a8c0d2e4f60718",  # bare 32-hex: no minted source
        "d_ef73a4312e5c4686196b30c035c8a05",  # 31 hex: truncated id
        "d_ef73a4312e5c4686196b30c035c8a05aa",  # 33 hex
        "D_ef73a4312e5c4686196b30c035c8a05a",  # minting is lowercase
        "d_EF73A4312E5C4686196B30C035C8A05A",  # uppercase hex
        "ef73a4312e5c4686196b30c035c8a05a",  # the tail without its d_ prefix
        mint_instance_id(),  # identity: a relay process id, not a minted reference
        f"{DEVICE_ID} extra",  # multi-token
        f"{DEVICE_ID},",
        f" {DEVICE_ID}",
        f"{DEVICE_ID} ",
        UUID_ID + " ",
        "019a5e3c9f2b7c418d5e6f7a8b9c0d1e",  # uuid without hyphens
        "a" * 64,  # bare 64-hex: body_sha shape, not a reference
        hashlib.sha256(b"fact-one").hexdigest(),
        "the SSN is 123-45-6789",
        "",
        None,
        12345,
        [DEVICE_ID],
    ):
        assert not structural_reference(value), repr(value)


@pytest.mark.parametrize(
    "fact",
    [
        {"title": "Payout", "body": "IBAN GB82WEST12345698765432"},
        {"title": "IBAN GB82WEST12345698765432", "body": "payout details"},
        {"title": "Card", "body": "card 4111 1111 1111 1111"},
        {"title": "card 4111-1111-1111-1111", "body": "payment details"},
    ],
)
def test_real_iban_and_card_in_fact_payload_still_refused(tmp_path, fact):
    # The structural exemption must not weaken the free-text guard: a real
    # IBAN/card supplied as fact content is still refused, even on a store
    # whose device id happens to be IBAN-shaped.
    store = MemoryStore(tmp_path, MemoryIdentity("s1"), origin="d_cd474787b1adaa67f8cb5934196c4834")
    with pytest.raises(MemoryError, match="Sensitive value refused"):
        store.put("global", fact)
