"""Pattern guard coverage: what is refused, what stays storable, and that it is bounded.

The guard is a HEURISTIC, not proof — these tests pin the classes it claims, and the
benign neighbours it must not break (QA Q2), so a loosening or a new false positive
fails by name.
"""

import hashlib
import json

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
    assert not refused(text), text


@pytest.mark.parametrize(
    "text",
    [
        "0123456789abcdef0123456789abcdef01234567",  # unnamed 40-hex = API-key shaped
        "token-ish aB3dE5gH7jK9mN1pQ3sT5vX7zA9cE1gI3kM5oQ7sU9wY",
    ],
)
def test_unnamed_opaque_tokens_are_refused_by_the_one_rule(text):
    # One rule for every caller: there is no longer a plain pass that admits what the
    # store refuses. A search for an id or SHA names it ("commit <sha>") or is a minted
    # structural reference (``structural_reference``), both of which pass.
    assert refused(text)
    assert not refused(f"commit {text}") or not text[:1].isdigit()


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
    assert not refused("- " * 4000)


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


# --- One rule, explainable refusals (the TestFlight-feedback regression) ---------------
#
# An agent's ``memory op=store`` of TestFlight feedback (prose + ids lifted from
# screenshots) was refused because a tunnel hostname label and an attachment digest are
# long hex runs, and the refusal named nothing, so the agent could not correct it. These
# pin: public identifiers pass, real credential shapes are refused with field + kind +
# span and never the value, and ``guard`` agrees with the tool path on one corpus.

from local_operator.memory import SensitiveValueError  # noqa: E402
from local_operator.memory_privacy import (  # noqa: E402
    refusal_message,
    retained_arguments,
)

TUNNEL_HEX = "3933a4f1c983a20eaa12228c90054b9f"
DIGEST64 = hashlib.sha256(b"synthetic screenshot").hexdigest()
# Synthetic credentials only, assembled so no literal secret sits in the source.
SYNTH_GITHUB = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
SYNTH_JWT = (
    "eyJhbGciOiJIUzI1NiJ9" + "." + "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IlN5bnRoIn0"
    "." + "c2lnbmF0dXJlLXN5bnRoZXRpYy12YWx1ZS0xMjM0NQ"
)
SYNTH_PEM = (
    "-----BEGIN "
    + "PRIVATE KEY-----\n"
    + "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC7syntheticSYNTHETICkeyMaterial0\n"
    + "-----END "
    + "PRIVATE KEY-----"
)
SYNTH_BEARER_VALUE = "abcDEF123456ghiJKL789012mnoPQR345678stu"
SYNTH_AWS = "AKIA" + "IOSFODNN7SYNTHETIC"[:16]


def feedback_text(attachment: str) -> str:
    return "\n".join(
        [
            "TestFlight feedback build 412 (from screenshots):",
            "1. Login button clipped on iPhone SE.",
            f"2. Pairing fails against tunnel {TUNNEL_HEX}-lop.radienthq.com after sleep.",
            f"3. Host https://{TUNNEL_HEX}-lop.radienthq.com/health returns 502.",
            f"4. Screenshot: {attachment}",
            "5. Session id 8024f1204e16 shows a stale title.",
            f"6. Crash digest sha256 {DIGEST64}.",
            "7. Ask id 0b9e7c2a-5d41-4f7e-9c3b-2a1d6e8f0a4b never resolves.",
            "8. Job 3f2a91 stalls at 99%.",
            "9. Dark mode contrast on the settings sheet is too low.",
        ]
    )


PUBLIC_CORPUS = [
    f"tunnel {TUNNEL_HEX}-lop.radienthq.com",
    f"https://{TUNNEL_HEX}-lop.radienthq.com/",
    f"see {TUNNEL_HEX}.lop.example.com.",
    f"screenshot ~/.local-operator/attachments/{DIGEST64}.bin",
    f"commit {DIGEST64[:40]}",
    f"sha256: {DIGEST64}",
    f"session id {TUNNEL_HEX}",
    f"job_id={TUNNEL_HEX}",
    "ask 0b9e7c2a-5d41-4f7e-9c3b-2a1d6e8f0a4b",
    feedback_text(f"~/.local-operator/attachments/{DIGEST64}.bin"),
]

SECRET_CORPUS = [
    (f"key {SYNTH_GITHUB} for CI", "github-token"),
    (f"the JWT is {SYNTH_JWT}", "jwt"),
    (f"deploy key:\n{SYNTH_PEM}", "pem-private-key"),
    (f"Authorization: Bearer {SYNTH_BEARER_VALUE}", "authorization-bearer"),
    (f"aws {SYNTH_AWS}", "aws-access-key-id"),
    (f"api_key={TUNNEL_HEX}", "credential-assignment"),
    (f"token {TUNNEL_HEX}", "named-secret"),
    (f"bare {TUNNEL_HEX}", "opaque-hex"),
    (f"/docs/{TUNNEL_HEX}", "opaque-hex"),
]


@pytest.mark.parametrize("text", PUBLIC_CORPUS)
def test_public_identifiers_and_locators_are_storable(text):
    assert not refused(text), text


@pytest.mark.parametrize("text,kind", SECRET_CORPUS)
def test_credential_shapes_are_refused_with_field_kind_and_span_never_value(text, kind):
    with pytest.raises(SensitiveValueError) as caught:
        validate_fact({"title": "T", "body": text})
    message = str(caught.value)
    span = next(s for s in caught.value.spans if s.kind == kind)
    assert span.field == "body"
    assert f"body: {kind} at chars {span.start}-{span.end}" in message
    secret = text[span.start : span.end]
    assert len(secret) >= 16 and secret not in message


@pytest.mark.parametrize("text", PUBLIC_CORPUS + [text for text, _ in SECRET_CORPUS])
def test_guard_and_the_tool_path_agree(text):
    """One rule: a direct ``guard`` call predicts the loop pre-check and the store."""
    direct = refused(text)
    loop = refusal_message("memory", {"op": "store", "text": text}) is not None
    store = False
    try:
        validate_fact({"title": "T", "body": text})
    except SensitiveValueError:
        store = True
    except MemoryError:
        pass  # the bare fact lacks source metadata; only the privacy verdict matters
    assert direct is loop is store, (text, direct, loop, store)


@pytest.mark.parametrize("text,kind", SECRET_CORPUS)
def test_retention_masks_only_the_offending_span(text, kind):
    args = {"op": "store", "text": f"keep this context. {text} and this tail."}
    kept, raw, masked = retained_arguments("memory", args)
    assert masked is True
    assert "memory_arguments_redacted" not in kept
    assert kept["op"] == "store"
    assert kept["text"].startswith("keep this context.")
    assert kept["text"].endswith("and this tail.")
    assert f"[redacted:{kind}]" in kept["text"]
    assert raw is not None and json.loads(raw) == kept
    for span in _spans_of(args["text"]):
        assert span.start is not None
        assert args["text"][span.start : span.end] not in kept["text"]
        assert args["text"][span.start : span.end] not in raw


def _spans_of(text):
    try:
        guard(text, field="text")
    except SensitiveValueError as exc:
        return exc.spans
    return []


def test_a_folded_only_match_keeps_the_whole_call_redaction():
    # Offsets cannot be trusted when invisible characters moved them: fail closed.
    args = {"op": "store", "text": "SSN 1\u200b23-45-6789"}
    kept, _raw, masked = retained_arguments("memory", args)
    assert masked is True and kept == {"memory_arguments_redacted": True, "op": "store"}


def test_low_variety_runs_are_padding_not_keys():
    assert not refused("a" * 64)
    assert refused("a3bf" * 10)


def test_a_live_attachment_blob_path_is_a_locator_and_nothing_else_in_config_is():
    from local_operator.paths import config_dir
    from local_operator.session.attachments import attachments_dir

    blob = attachments_dir() / f"{DIGEST64}.bin"
    assert not refused(f"screenshot {blob}")
    assert not refused(f"screenshot {attachments_dir() / TUNNEL_HEX}.bin.")
    # Only a digest-named direct child of the store is lifted; the config-dir root
    # verdict still refuses a non-digest name and anything else under the config dir.
    assert refused(f"see {attachments_dir() / 'notes.txt'}")
    assert refused(f"see {attachments_dir() / 'nested' / DIGEST64}.bin")
    assert refused(f"see {config_dir() / 'memory' / 'bank.sqlite'}")
    with pytest.raises(SensitiveValueError, match=r"Sensitive path refused \(value: "):
        guard(f"see {config_dir() / 'memory'}")
