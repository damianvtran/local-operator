"""A running session's borrowed credential heals, or says precisely why it cannot.

THE PROBLEM THIS LOCKS DOWN. A session on a borrower device asks its owner for a
credential, and later a turn fails because the borrowed bearer stopped working. The
owner may already hold a working login — it signed in again, or the share was
re-made — and the correct move is to ask once more, through the grant path that
already exists. What used to happen instead: the borrower kept answering from its
OWN cached refusal (``state.py``: 300 s for the codes that mean "the owner has
nothing right now"), so a repaired login stayed invisible for minutes and the
operator restarted the session.

WHAT IS PROVEN HERE, and what is not. The first two tests drive both relays on
loopback with real pairing, the real share/revoke CLI and a real
``MeshCredentialClient``: the cached refusal is real, the re-share is real, and the
repair clears exactly the memo that hid it. The rest are table tests over the
classification: three causes, three different repairs, and the command printed is
the one that fixes it. No test asserts, prints or stores a credential value; the
bearer is only ever compared for identity.
"""

from __future__ import annotations

import pathlib
from typing import Any

import pytest

from local_operator.network.credentials.healing import (
    CAUSE_BY_CODE,
    CAUSE_GRANT,
    CAUSE_OWNER_LOGIN,
    CAUSE_UNKNOWN,
    CAUSE_UNREACHABLE,
    HealOutcome,
    cause_for_code,
    cause_for_evidence,
    repair_for,
)
from local_operator.network.credentials.types import (
    BROKER_ERROR_CODES,
    BrokerError,
    CredentialRef,
    Grant,
)
from tests.unit.network.test_credentials_owner import (  # noqa: F401 — fixtures by import
    STUB_PROVIDER,
    idp,
    stub_provider,
)
from tests.unit.network.test_credentials_real_link import (  # noqa: F401 — fixtures by import
    _borrower_client,
    _lop_network,
    _pull,
    _share,
    mesh,
)

# The ``mesh`` fixture above is built on these, so importing them is what registers
# the ``devices`` and relay-shaping fixtures it asks for by name.
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixtures by import
    _pair,
    devices,
    serve_shaped_relay,
)

# ---------------------------------------------------------------------------
# The classification: one refusal, one cause, one repair
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "cause"),
    [
        # The owner has no login to lend until a person signs in THERE.
        ("interactive_required", CAUSE_OWNER_LOGIN),
        ("no_local_credential", CAUSE_OWNER_LOGIN),
        ("refresh_failed", CAUSE_OWNER_LOGIN),
        # Permission, not login: signing in anywhere changes nothing.
        ("not_a_holder", CAUSE_GRANT),
        ("revoked", CAUSE_GRANT),
        ("consent_withdrawn", CAUSE_GRANT),
        # No answer: nothing is proved about the credential.
        ("owner_offline", CAUSE_UNREACHABLE),
    ],
)
def test_each_refusal_classifies_into_the_action_that_actually_fixes_it(
    code: str, cause: str
) -> None:
    assert cause_for_code(code) == cause


def test_every_broker_code_is_classified_or_deliberately_unknown() -> None:
    """A new code must not silently join one of the three named causes.

    ``CAUSE_BY_CODE`` may not be complete — some codes are about behaviour
    (quota, rate) rather than about the credential — but every code it does name
    must be a real broker code, and every code it does NOT name must read as
    ``unknown`` rather than as the nearest plausible cause.
    """
    for code in CAUSE_BY_CODE:
        assert code in BROKER_ERROR_CODES, code
    unclassified = sorted(BROKER_ERROR_CODES - set(CAUSE_BY_CODE))
    for code in unclassified:
        assert cause_for_code(code) == CAUSE_UNKNOWN, code


def test_the_mcp_repair_names_the_peer_and_the_share_verb() -> None:
    repair = repair_for(
        "mcp:https://example.test/mcp",
        cause=CAUSE_GRANT,
        owner_device="d_owner",
        owner_name="owner-laptop",
        peer_name="node-b",
    )
    assert repair.cause == CAUSE_GRANT
    assert repair.commands == (
        "lop network credential share mcp:https://example.test/mcp --with node-b",
    )
    # The command is on the OWNER, and the sentence says so: a borrower cannot
    # re-share a credential it does not own.
    assert "on macbook" not in repair.sentence
    assert "on <owner>" not in repair.sentence
    assert "owner" in repair.sentence.lower()


def test_an_unreachable_owner_has_no_command_and_invalidates_nothing() -> None:
    repair = repair_for(
        "mcp:https://linear.example.test/mcp",
        cause=CAUSE_UNREACHABLE,
        owner_name="macbook",
        peer_name="node-b",
    )
    assert repair.commands == ()
    assert "cannot be reached" in repair.sentence
    assert "Nothing was changed" in repair.sentence or "nothing" in repair.sentence


def test_the_evidence_mapping_reads_the_durable_states() -> None:
    # ``http_401`` is the one state only the MCP layer can observe: the server
    # refused a bearer the owner had just served.
    assert cause_for_evidence("unauthorized", "http_401") == CAUSE_OWNER_LOGIN
    assert cause_for_evidence("expired", "refresh_failed") == CAUSE_OWNER_LOGIN
    assert cause_for_evidence("revoked", "not_a_holder") == CAUSE_GRANT
    assert cause_for_evidence("unavailable", "owner_offline") == CAUSE_UNREACHABLE
    assert cause_for_evidence("", "") == ""


def test_a_heal_outcome_never_carries_a_bearer() -> None:
    """The receipt is a receipt: the bearer stays in the grant cache, not here."""
    outcome = HealOutcome(key="openai", recovered=True)
    assert "access_token" not in outcome.as_dict()
    assert set(outcome.as_dict()) == {"credential_name", "recovered", "code", "cause", "reason"}


# ---------------------------------------------------------------------------
# The real link: a refusal is cached here, and the repair clears exactly that
# ---------------------------------------------------------------------------


#: The codes an owner whose login cannot be refreshed answers with: a transient
#: outage (``refresh_failed``) or a login that needs a person (``interactive_required``).
_LOGIN_GONE_CODES = frozenset({"refresh_failed", "interactive_required", "no_local_credential"})


def _rig(request: pytest.FixtureRequest) -> tuple[Any, Any]:
    """The two-relay rig and its rotating endpoint, BY NAME.

    ``getfixturevalue`` rather than parameters named after the fixtures, which is the
    pattern this repo's real-link modules already use: a parameter would shadow the
    module-level import that makes the fixture reachable at all.
    """
    return request.getfixturevalue("mesh"), request.getfixturevalue("idp")


def _dead_login(endpoint: Any) -> None:
    """The loopback token endpoint refuses every exchange from now on.

    That is what an owner whose refresh token was revoked sees, and it is the state
    the operator's repair ends — :func:`_owner_signs_in` is the other side of it.
    """
    endpoint.invalid = True


def _owner_signs_in(rig: Any, endpoint: Any) -> None:
    """What the operator does on the OWNING device: a fresh login there.

    The endpoint accepts exchanges again and the owner's row is replaced with one
    whose refresh token the endpoint honours — the same two facts a real ``lop login``
    produces. Nothing is done on the borrower, which is the point of the test below.
    """
    from local_operator.providers.auth_store import AuthStore

    endpoint.invalid = False
    endpoint.current_refresh = "refresh-relogin"
    auth = AuthStore(config_dir=rig.a.root)
    auth.upsert_credential(
        STUB_PROVIDER,
        {
            "access": "access-0",
            "expires": 0,  # expired, so the owner must exercise the new refresh token
            "refresh": "refresh-relogin",
            "email": "owner@example.test",
        },
    )


def test_the_next_resolve_recovers_once_the_owner_signs_in_again(
    request: pytest.FixtureRequest,
) -> None:
    """A RUNNING session heals with no borrower-side action, on two real relays.

    The owner's login cannot be refreshed, so the ask is refused with the code that
    means "the owner has nothing to offer". The operator then signs in again on the
    owning device — and the NEXT resolve from this same client is served, because a
    provider borrow is re-resolved with the owner on every request rather than served
    from a grant cache. That is the property a heal must not break, and the reason
    the borrower needs no new state of its own.
    """
    rig, endpoint = _rig(request)
    _dead_login(endpoint)
    _share(rig)
    _pull(rig)
    client = _borrower_client(rig)
    refused = client.request_grant_sync(STUB_PROVIDER, session_id="sess-heal")
    assert isinstance(refused, BrokerError), refused
    assert refused.code in _LOGIN_GONE_CODES, refused

    _owner_signs_in(rig, endpoint)
    recovered = client.request_grant_sync(STUB_PROVIDER, session_id="sess-heal")
    assert type(recovered).__name__ == "Grant", recovered


def test_a_withdrawn_share_is_refused_again_and_never_widened(
    request: pytest.FixtureRequest,
) -> None:
    """Healing re-ASKS; it never assumes, and never widens who may borrow.

    The share is withdrawn on the owner. Dropping this device's cached observation
    changes nothing about that: the next dial is refused ``not_a_holder`` again, by
    the owner — which is why the sentence for that code names a grant remedy rather
    than a sign-in.
    """
    rig = request.getfixturevalue("mesh")
    _share(rig)
    _pull(rig)
    client = _borrower_client(rig)
    assert (
        type(client.request_grant_sync(STUB_PROVIDER, session_id="sess-heal")).__name__ == "Grant"
    )

    assert _lop_network("credential", "revoke", STUB_PROVIDER, "--from", rig.b.identity.name) == 0
    refused = client.request_grant_sync(STUB_PROVIDER, session_id="sess-heal")
    assert isinstance(refused, BrokerError), refused
    assert refused.code == "not_a_holder", refused

    client.forget_refusal(STUB_PROVIDER)
    again = client.request_grant_sync(STUB_PROVIDER, session_id="sess-heal")
    assert isinstance(again, BrokerError), again
    assert again.code == "not_a_holder", again
    repair = repair_for(
        again.key,
        cause=cause_for_code(again.code),
        code=again.code,
        owner_device=again.owner_device,
        owner_name=again.owner_device_name,
        peer_name=rig.b.identity.name,
    )
    assert repair.cause == CAUSE_GRANT
    assert repair.commands and repair.commands[0].startswith("lop network credential share ")


# ---------------------------------------------------------------------------
# The MCP half: a refused borrow is recorded as EVIDENCE, not swallowed
# ---------------------------------------------------------------------------


def test_a_refused_mcp_borrow_records_the_cause_durably(tmp_path: pathlib.Path) -> None:
    """The 401 alone cannot say why no bearer was offered; the refusal code can.

    ``BrokeredBearerAuth`` used to return ``""`` on a refused borrow, so the connect
    failed as a bare challenge and every surface could only guess between "the owner
    signed out", "the share was withdrawn" and "the owner is away". This drives the
    real class with a broker stand-in and reads the durable evidence back.
    """
    from local_operator.network.credentials import mcp_state
    from local_operator.network.credentials.mcp_bearer import BrokeredBearerAuth
    from local_operator.network.credentials.types import (
        BrokerError,
        credential_key_for_mcp,
    )

    url = "https://mcp.example.test/mcp"
    key = credential_key_for_mcp(url)

    class Refusing:
        network_id = "n_heal"
        root = tmp_path
        self_device = "d_peer"

        def request_grant_sync(
            self, requested: str, provider: str = "", session_id: str = ""
        ) -> Any:
            return BrokerError(
                code="interactive_required",
                key=requested,
                owner_device="d_owner",
                owner_device_name="macbook",
            )

        def owner_of(self, requested: str) -> str:
            return "d_owner"

        def owner_label(self, requested: str) -> str:
            return "macbook"

    bearer = BrokeredBearerAuth(url=url, key=key, client=Refusing())

    # The borrow is refused, and the borrower connects unauthenticated exactly as
    # before (a refused borrow never raises into the connect) …
    assert bearer._blocking_bearer() == ""  # noqa: SLF001 — the borrow seam under test

    evidence = mcp_state.load("n_heal", tmp_path)[key]
    assert evidence["state"] == "expired"
    assert evidence["code"] == "interactive_required"
    assert evidence["owner_name"] == "macbook"


def test_repair_asks_only_borrowable_keys_of_its_own_network(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--repair`` asks what this device may ask, in the network it can reach.

    Two ways the first version over-reached, both from selecting on
    ``entry.is_holder(self)`` across every placement document: a key this device
    OWNS (no borrow exists — the local cascade serves it) and a key belonging to
    ANOTHER network (a client dials one network's relay). This drives the real
    ``_repair_borrowed`` over a three-key, two-network root and records what was
    asked (review round 2, R2-3).
    """
    from local_operator.network import cli as network_cli
    from local_operator.network.credentials.placement import PlacementDocument

    doc_a = PlacementDocument("n_first", root=tmp_path, written_by="d_self")
    doc_a.declare(
        "meshtest",
        owner_device="d_owner",
        owner_device_name="owner-b",
        provider="meshtest",
        identity_label="owner@example.test",
        by="d_owner",
    )
    doc_a.grant("meshtest", "d_self", scope="session", by="d_owner")
    # Owned here: no borrow exists, so asking would be a request to be served a
    # credential this device already holds.
    doc_a.declare(
        "openai",
        owner_device="d_self",
        owner_device_name="self",
        provider="openai",
        identity_label="me@example.test",
        by="d_self",
    )
    doc_a.save()

    doc_b = PlacementDocument("n_other", root=tmp_path, written_by="d_self")
    doc_b.declare(
        "anthropic",
        owner_device="d_owner",
        owner_device_name="owner-b",
        provider="anthropic",
        identity_label="owner@example.test",
        by="d_owner",
    )
    doc_b.grant("anthropic", "d_self", scope="session", by="d_owner")
    doc_b.save()

    asked: list[str] = []

    class FakeClient:
        network_id = "n_first"
        self_device = "d_self"

        def forget_refusal(self, key):
            return None

        def owner_of(self, key):
            return "d_owner"

        def owner_label(self, key):
            return "owner-b"

        async def grant_async(self, key, **kwargs):
            asked.append(key)
            return BrokerError(
                code="not_a_holder",
                key=key,
                owner_device="d_owner",
                owner_device_name="owner-b",
            )

        def close(self):
            return None

    fake = FakeClient()
    monkeypatch.setattr(network_cli, "_placement_documents", lambda: [doc_a, doc_b])
    monkeypatch.setattr(
        "local_operator.network.credentials.client.MeshCredentialClient.for_this_device",
        classmethod(lambda cls, *args, **kwargs: fake),
    )

    outcomes, skipped = network_cli._repair_borrowed(  # noqa: SLF001 — the seam under test
        "d_self", "node-a"
    )

    assert asked == ["meshtest"], asked
    assert outcomes and outcomes[0]["cause"] == CAUSE_GRANT, outcomes
    # AND EVERY KEY IT DID NOT ASK IS NAMED WITH ITS REASON (QA round 3): a repair
    # that prints nothing is indistinguishable from one that repaired nothing.
    reasons = {row["credential_name"]: row["reason"] for row in skipped}
    assert "owned here" in reasons["openai"], reasons
    assert "in network 'n_other'" in reasons["anthropic"], reasons


def test_repair_with_nothing_askable_says_so(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No placement rows at all must produce a REASON, not silence.

    QA round 3's finding: ``lop network credentials --repair`` with a live refusal
    in hand printed its network summary and exited 0 — no repair line, no skip, and
    nothing to tell "no key was askable" from "the selection missed one". This is the
    empty case (nothing shared with this device in this network), which is the one an
    operator with only a durable refusal record hits.
    """
    from local_operator.network import cli as network_cli

    class FakeClient:
        network_id = "n_first"
        self_device = "d_self"

        def forget_refusal(self, key):  # pragma: no cover - no key is asked
            raise AssertionError("no key should have been asked")

        def close(self):
            return None

    monkeypatch.setattr(network_cli, "_placement_documents", lambda: [])
    monkeypatch.setattr(
        "local_operator.network.credentials.client.MeshCredentialClient.for_this_device",
        classmethod(lambda cls, *args, **kwargs: FakeClient()),
    )

    outcomes, skipped = network_cli._repair_borrowed("d_self", "node-a")  # noqa: SLF001

    assert outcomes == [], outcomes
    assert len(skipped) == 1, skipped
    assert skipped[0]["credential_name"] == "", skipped
    assert "no placement rows in 'n_first'" in skipped[0]["reason"], skipped


def test_repair_still_repairs_the_borrowable_key(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The speaking path must not turn into a reporting-only path.

    A holder key that is not already working is still ASKED, and a grant from the
    owner is still reported as ``recovered`` (review round 3's second half: fixing
    the silence must not cost the repair).
    """
    from local_operator.network import cli as network_cli
    from local_operator.network.credentials.placement import PlacementDocument

    document = PlacementDocument("n_first", root=tmp_path, written_by="d_self")
    document.declare(
        "meshtest",
        owner_device="d_owner",
        owner_device_name="owner-b",
        provider="meshtest",
        identity_label="owner@example.test",
        by="d_owner",
    )
    document.grant("meshtest", "d_self", scope="session", by="d_owner")
    document.save()

    asked: list[str] = []

    class FakeClient:
        network_id = "n_first"
        self_device = "d_self"

        def forget_refusal(self, key):
            return None

        def owner_of(self, key):
            return "d_owner"

        def owner_label(self, key):
            return "owner-b"

        async def grant_async(self, key, **kwargs):
            asked.append(key)
            return Grant(
                access_token="bearer-for-the-test",
                kind="bearer",
                token_expires_at_ms=0,
                grant_expires_at_ms=0,
                credential_ref=CredentialRef(
                    owner_device="d_owner",
                    owner_device_name="owner-b",
                    provider="meshtest",
                    kind="oauth",
                    credential_id=1,
                ),
                served_by="d_owner",
            )

        def close(self):
            return None

    monkeypatch.setattr(network_cli, "_placement_documents", lambda: [document])
    monkeypatch.setattr(
        "local_operator.network.credentials.client.MeshCredentialClient.for_this_device",
        classmethod(lambda cls, *args, **kwargs: FakeClient()),
    )

    outcomes, skipped = network_cli._repair_borrowed("d_self", "node-a")  # noqa: SLF001

    assert asked == ["meshtest"], asked
    assert [row["recovered"] for row in outcomes] == [True], outcomes
    assert skipped == [], skipped
