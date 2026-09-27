"""Whose sentence does the operator read when a borrow is refused?

AUDIT ROUND 2, F1. The owner puts a DIAGNOSTIC on the wire (``owner._refuse``'s own
comment: "the peer gets the CODE and the owner's own words; the SENTENCE the operator
reads is rendered where the operator is — client.py → messages.py"). The requester used
to render a sentence only when the wire message was EMPTY, so the diagnostic won: after
``lop network credential revoke`` the borrower showed

    damians-MacBook-Pro does not share 'zai' with d_1d2a4f5aae0daffc36653e3659260

— a raw device id where the design's §4.6 sentence names the owner by name, says the
share remedy, and names no id at all. Measured on the two-device loopback rig.

Every cell here drives ``MeshCredentialClient._detail_from_owner``, the leg-2 entry
point the wire itself reaches, and asserts on the STRING a person would read. The
catalogue's own table is pinned separately
(``test_credentials_refusal_catalogue.py``) so that a fix to the rendering rule cannot
hide behind an import error.
"""

from __future__ import annotations

import pytest

from local_operator.network.credentials.client import MeshCredentialClient
from local_operator.network.credentials.placement import PlacementDocument

OWNER = "d_" + "a" * 32
BORROWER = "d_" + "b" * 32
NETWORK = "n_" + "c" * 24


def _client(root) -> MeshCredentialClient:
    document = PlacementDocument(NETWORK, root=root, written_by=OWNER)
    document.declare(
        "zai",
        owner_device=OWNER,
        owner_device_name="damian-mbp",
        provider="zai",
        by=OWNER,
    )
    document.grant("zai", BORROWER, scope="session", by=OWNER)
    return MeshCredentialClient(
        root=root,
        self_device=BORROWER,
        network_id=NETWORK,
        placement=document,
    )


def _reply(code: str, message: str) -> dict[str, object]:
    """A leg-2 ``ack`` carrying a refusal, exactly as the owner's relay frames one."""
    return {
        "op": "ack",
        "req": 7,
        "detail": {
            "kind": "error",
            "code": code,
            "key": "zai",
            "message": message,
            "owner_device": OWNER,
            "owner_device_name": "damian-mbp",
        },
    }


def test_the_owner_s_diagnostic_never_becomes_the_sentence_the_operator_reads(tmp_path) -> None:
    """F1's headline: the catalogue's sentence wins for a classified code."""
    client = _client(tmp_path)
    owner_words = f"damian-mbp does not share 'zai' with {BORROWER!r}"

    detail = client._detail_from_owner(  # noqa: SLF001 — the leg-2 entry point, as the wire uses it
        _reply("not_a_holder", owner_words), "zai", "zai"
    )

    sentence = str(detail.get("message") or "")
    assert sentence, "a refusal must carry a sentence"
    assert BORROWER not in sentence, sentence
    assert sentence.startswith("damian-mbp does not share 'zai' with this device."), sentence
    assert "lop network credential share zai" in sentence, sentence
    assert "lop login zai" in sentence, sentence


def test_an_unclassified_code_still_keeps_the_owner_s_own_words(tmp_path) -> None:
    """The unclassified arm: a newer owner's reason survives — as the DETAIL.

    REVIEW ROUND 1, R2. The first version of this cell asserted only that the owner's
    words appeared somewhere in the sentence, which is true under BOTH
    implementations: the old one returned the owner's message verbatim, the new one
    interpolates it into ``messages._generic``'s parenthesis. That made it useless as
    a regression test. The discriminating questions are whether THIS build's remedy is
    present (the old arm produced none) and whether the owner's words are the WHOLE
    sentence (the defect: a bare class name reaching an operator with no owner).
    """
    client = _client(tmp_path)

    detail = client._detail_from_owner(  # noqa: SLF001
        _reply("quota_window_exhausted", "the account's weekly window is spent"),
        "zai",
        "zai",
    )

    sentence = str(detail.get("message") or "")
    assert sentence != "the account's weekly window is spent", sentence
    assert "the account's weekly window is spent" in sentence, sentence
    assert "damian-mbp declined to lend 'zai'" in sentence, sentence
    assert "Nothing was changed on either device" in sentence, sentence
    assert "run 'lop login zai' here" in sentence, sentence


def test_an_internal_diagnostic_never_becomes_the_whole_sentence(tmp_path) -> None:
    """The exact shape R2 measured: a broker crash put a bare ``TimeoutError`` up.

    ``owner._refuse``'s catch-all sends ``exc.__class__.__name__`` as the message, so
    before the fix the operator read that class name alone — no owner, no remedy —
    contradicting this module's own promise that a refusal always says what to do.
    """
    client = _client(tmp_path)

    detail = client._detail_from_owner(  # noqa: SLF001
        _reply("internal", "TimeoutError"), "zai", "zai"
    )

    sentence = str(detail.get("message") or "")
    assert sentence != "TimeoutError", sentence
    assert "damian-mbp declined to lend 'zai'" in sentence, sentence
    assert "run 'lop login zai' here" in sentence, sentence
    # The diagnostic is not thrown away — it is the parenthesis, where a diagnostic
    # belongs, and the one surface that reads it is a person reading a log.
    assert "TimeoutError" in sentence, sentence


def test_an_older_owner_that_sends_no_message_still_gets_the_catalogue_sentence(tmp_path) -> None:
    """The pre-existing arm: no message on the wire at all."""
    client = _client(tmp_path)
    detail = client._detail_from_owner(  # noqa: SLF001
        {"op": "ack", "req": 7, "detail": {"kind": "error", "code": "revoked", "key": "zai"}},
        "zai",
        "zai",
    )
    sentence = str(detail.get("message") or "")
    assert "no longer shares" in sentence, sentence


@pytest.mark.parametrize(
    "code", ["not_a_holder", "revoked", "owner_offline", "no_local_credential"]
)
def test_no_sentence_a_person_reads_carries_a_device_id(tmp_path, code: str) -> None:
    """§4's copy rule, asserted structurally: no raw ``d_…`` id in operator-facing text."""
    client = _client(tmp_path)
    detail = client._detail_from_owner(  # noqa: SLF001
        _reply(code, f"internal shorthand naming {BORROWER}"),
        "zai",
        "zai",
    )
    sentence = str(detail.get("message") or "")
    assert BORROWER not in sentence, sentence
    assert "d_" not in sentence, sentence
