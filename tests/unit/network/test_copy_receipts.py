"""What the S4 copy receipts SAY (design review round 1: D1-D8, N1).

The copy path's receipts are the only place an operator learns what ending a copy
did and did not do, so their sentences are asserted as a CONTRACT, state by
state, through the real verbs (``_lop_network`` parses and dispatches exactly as
typed). Nothing here dials a relay: the receipts are composed from the owner's
own ledger and placement document, and a cell that needed two relays to read a
sentence would be a slow way to read it.

WHAT IS PINNED, and the finding each cell answers:

- the ceiling is printed in EVERY revoke state of a stored secret, tense-free, and never
  borrows the tombstone's verb (D1, D2) — and the guide and the design quote the SAME words
  (D2: "move ALL of them together"), so a future edit that moves one fails here;
- a provider key's revoke keeps its OWN closing sentence, and the guide and the design
  scope the ceiling claim to say so (the author's own sweep of the round's universal
  claims found the first draft's "every" false on that arm); the boundary read is on the
  JOINED output, so the ceiling in either form fails it (F5);
- a receipt says what the LEDGER holds, never what the world holds (D1);
- one secret has one spelling in prose (D5) while ``--json`` keeps the
  placement key an integration reads;
- ``mark`` receipts parse, carry the reachability qualifier, and do not claim a
  mark reaches a device that is already approved (D6, and the overclaim the
  remediation found: ``mark sync`` writes no holder row);
- a bare ``secret:`` on ``mark`` is refused BY NAME, never with ``named ''`` and a
  ``lop secret set `` remedy that cannot run (O-1);
- the removal receipt names a copy a copy, in the right number (D8);
- the two docs sentences that contradicted the ceiling and the at-rest facts
  are gone (D3, D4).

No broker daemon is started by any cell: ``ensure_broker`` is stubbed (the same
seam six sibling files use), because a store opened without ``create`` spawns a
detached daemon that outlives the run — the leak class review round 5's F4
measured on the relay-based files.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from local_operator.network import relay as relay_mod
from local_operator.network import store as network_store
from local_operator.network import wire
from local_operator.network.credentials import messages as messages_mod
from local_operator.network.credentials import offers as offers_mod
from local_operator.network.credentials import placement as placement_mod
from local_operator.network.credentials import sync
from local_operator.network.credentials.types import BrokerError
from local_operator.network.identity import mint
from local_operator.network.types import MeshRefusal, NetworkRecord, SecretState
from tests.unit.network.test_credentials_real_link import _lop_network, _parser

REPO = Path(__file__).resolve().parents[3]

OWNER_NAME = "owner-laptop"
MEMBER = "d_" + "0" * 31 + "2"
MEMBER_NAME = "device-b"
NETWORK = "n_copy_receipts"
#: A name that is not a provider's and is not a forge credential: the receipts under
#: test are about ANY store secret, and a GitHub-flavoured name would invite the
#: reader to think they are about the brokered ``github`` credential.
SECRET = "CRM_API_KEY"
KEY = f"secret:{SECRET}"
DIGEST = "sha256:" + "a" * 64
CEILING = messages_mod.COPY_CEILING_LINES


@pytest.fixture()
def owner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """An owner device with one store secret, one other member, and NO relay."""
    from local_operator.secrets import access
    from local_operator.secrets import client as secrets_client

    root = tmp_path / "owner"
    root.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    # No daemon: ``access.open_store`` WITHOUT ``create`` asks the broker first and
    # spawns one when none runs. Declining it falls back to the on-disk key, which
    # is the same authority (``master_key_for``'s own contract) — and leaves no
    # process behind.
    monkeypatch.setattr(secrets_client, "ensure_broker", lambda *a, **k: False)
    identity = mint(root, name=OWNER_NAME)
    record = NetworkRecord(
        network_id=NETWORK,
        name="testnet",
        created_by=identity.device_id,
        self_device_id=identity.device_id,
        self_role="admin",
        self_capabilities=["admin"],
    )
    relay_mod.admit(
        record,
        device_id=identity.device_id,
        public_key=identity.public_key,
        name=identity.name,
        role="admin",
        capabilities=["admin"],
        added_by=identity.device_id,
        added_via="self",
        root=root,
        persist=False,
    )
    relay_mod.admit(
        record,
        device_id=MEMBER,
        public_key="a" * 43,
        name=MEMBER_NAME,
        role="drive",
        capabilities=["drive", "broker_credential"],
        added_by=identity.device_id,
        added_via="invite",
        root=root,
        persist=False,
    )
    network_store.save(record, root)
    network_store.save_secrets(
        SecretState(network_id=NETWORK, epoch=1, secret=wire.b64u(b"0" * 32)), root
    )
    access.open_store(root, create=True).set(SECRET, b"a-stored-value")
    return SimpleNamespace(root=root, identity=identity)


def _placement(owner: SimpleNamespace, *, member_holds: bool) -> None:
    """The owner's placement document: the key declared, the member a holder or not."""
    me = owner.identity.device_id
    document = placement_mod.PlacementDocument(NETWORK, root=owner.root, written_by=me)
    document.declare(
        KEY,
        owner_device=me,
        owner_device_name=OWNER_NAME,
        provider=SECRET,
        kind="store-secret",
        by=me,
    )
    if member_holds:
        document.grant(KEY, MEMBER, scope="session", by=me)
    document.save()


def _ledger(owner: SimpleNamespace, state: bool | None) -> None:
    """``None`` writes no row; ``False`` a live (confirmed) row; ``True`` a wiped one."""
    if state is None:
        return
    with sync.mutate(NETWORK, owner.root) as document:
        document.record_ack(MEMBER, KEY, gen=1, digest=DIGEST, at=time.time(), wiped=state)


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, list[str], str]:
    code = _lop_network(*argv)
    captured = capsys.readouterr()
    return code, captured.out.splitlines(), captured.err


# ---------------------------------------------------------------------------
# D2 — the ceiling: tense-free, one verb for one act, quoted everywhere alike
# ---------------------------------------------------------------------------


def test_the_ceiling_is_tense_free_and_never_borrows_the_removal_verb() -> None:
    """D2: printed after states where NOTHING was removed yet, so it cannot say so.

    The old sentence opened "This removed the copies it could reach" — printed
    straight after "a wipe notice … is queued" and "could NOT be confirmed
    deleted", and on the ``member rm`` receipt directly under "removed
    device-b from home-net": one verb for the tombstone and for the wipe, and a
    completed-action claim on the arms that say nothing completed.
    """
    assert len(CEILING) == 2
    assert messages_mod.COPY_CEILING_SENTENCE == " ".join(CEILING)
    for line in CEILING:
        assert "removed" not in line, line
    assert CEILING[0].startswith("This ends the copies the owner can still reach")
    assert "rotating the secret at its source" in CEILING[1]


def test_the_guide_and_the_design_quote_the_ceiling_verbatim() -> None:
    """D2's "move ALL of them together", made structural.

    The sentence lives in the receipts, the guide's credentials paragraph and the
    design's §2.3. Three copies of one commitment drift the first time one is
    edited from memory; reading the two documents back through the constant is
    what makes the next edit fail here instead of in an operator's hands.
    """
    sentence = " ".join(messages_mod.COPY_CEILING_SENTENCE.split())
    for relative in (
        "local_operator/guides/network/GUIDE.md",
        "docs/design/mesh-consent-provisioning.md",
    ):
        text = " ".join((REPO / relative).read_text(encoding="utf-8").split())
        assert sentence in text, f"{relative} does not carry the ceiling sentence verbatim"


# ---------------------------------------------------------------------------
# D1 — every revoke state prints the ceiling; the receipt reads the LEDGER
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ledger", "headline"),
    [
        (None, f"no copy of '{SECRET}' is recorded on {MEMBER_NAME}"),
        (False, f"a wipe notice for '{SECRET}' is queued for {MEMBER_NAME}"),
        (True, f"the copy of '{SECRET}' on {MEMBER_NAME} was already wiped"),
    ],
    ids=["no-row", "live-row", "wiped-row"],
)
def test_every_revoke_state_prints_the_ceiling_after_its_state_line(
    owner: SimpleNamespace,
    capsys: pytest.CaptureFixture[str],
    ledger: bool | None,
    headline: str,
) -> None:
    """D1: the state that claims the MOST used to be the only one without the ceiling.

    ``no copy … was confirmed … so there is nothing to wipe there`` turned an
    absent LEDGER row into a statement about the world, and the receipt then
    withheld the one sentence that says what a lost confirmation leaves behind.
    The ledger row is written from the member's REPLY — the channel the removal
    path documents as losable — so "no row" is "not confirmed", never "not there".
    """
    _placement(owner, member_holds=True)
    _ledger(owner, ledger)
    code, lines, _err = _run(capsys, "credential", "revoke", SECRET, "--from", MEMBER_NAME)
    assert code == 0, lines
    assert any(line.startswith(headline) for line in lines), lines
    # The ceiling comes AFTER the state line, one fact per line.
    for line in CEILING:
        assert line in lines, lines
    assert lines.index(CEILING[0]) > max(
        index for index, line in enumerate(lines) if line.startswith(headline)
    )
    # The old world-claims are gone, whichever state is printing.
    text = "\n".join(lines)
    assert "nothing to wipe there" not in text
    assert "was confirmed on" not in text


def test_the_no_row_arm_says_a_lost_confirmation_is_untracked(
    owner: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    """D1's second fact: what the ledger CANNOT see is named, not implied away."""
    _placement(owner, member_holds=True)
    _code, lines, _err = _run(capsys, "credential", "revoke", SECRET, "--from", MEMBER_NAME)
    assert "a copy whose confirmation never arrived is not tracked here" in lines, lines


def test_a_provider_key_revoke_keeps_its_own_closing_sentence_and_the_docs_say_so(
    owner: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    """The boundary of the ceiling: it closes a stored secret's revoke, not a provider key's.

    ``credential revoke`` on a provider API key (class 4) predates S4 and closes with its own
    sentence for the same fact - "a copy of the key taken out of that device never expires: to
    end it, rotate the key at the provider" - which says the second half of the ceiling in
    that credential's own words and omits the first. The guide and the design doc therefore
    SCOPE the claim ("on a stored secret", "keeps the closing sentence it had before S4"), and
    this cell is what keeps that scoping true: the day the class-4 arm prints the two lines
    too, prints the JOINED sentence (F5), or loses its own sentence, the documents that
    describe the boundary fail with it.
    """
    me = owner.identity.device_id
    document = placement_mod.PlacementDocument(NETWORK, root=owner.root, written_by=me)
    document.declare(
        "openai",
        owner_device=me,
        owner_device_name=OWNER_NAME,
        provider="openai",
        kind="api-key-static",
        by=me,
    )
    document.grant("openai", MEMBER, scope="device", by=me)
    document.save()
    with sync.mutate(NETWORK, owner.root) as state:
        state.record_ack(MEMBER, "openai", gen=1, digest=DIGEST, at=time.time(), wiped=False)

    code, lines, _err = _run(capsys, "credential", "revoke", "openai", "--from", MEMBER_NAME)
    assert code == 0, lines
    # F5: read the JOINED output, not whole list items. A ceiling printed as ONE
    # line — the joined sentence, the call the old class-2 branch made — has no
    # item equal to either line, so per-item membership cannot see it; a substring
    # read sees both forms.
    joined = "\n".join(lines)
    for ceiling_line in CEILING:
        assert ceiling_line not in joined, lines
    assert (
        "a copy of the key taken out of that device never expires: to end it, "
        "rotate the 'openai' key at the provider"
    ) in lines, lines
    assert "`credential revoke` on a provider API key closes with its own sentence" in (
        _credentials_section()
    )
    design = " ".join(
        (REPO / "docs/design/mesh-consent-provisioning.md").read_text(encoding="utf-8").split()
    )
    assert (
        "the provider-key (class-4) `credential revoke` keeps the closing sentence it had "
        "before S4"
    ) in design


@pytest.mark.parametrize(
    ("ledger", "wipe", "confirmed", "wiped"),
    [(None, "none", False, False), (False, "queued", True, False), (True, "done", True, True)],
    ids=["no-row", "live-row", "wiped-row"],
)
def test_the_json_receipt_keeps_the_placement_key_and_the_copy_block(
    owner: SimpleNamespace,
    capsys: pytest.CaptureFixture[str],
    ledger: bool | None,
    wipe: str,
    confirmed: bool,
    wiped: bool,
) -> None:
    """D5's boundary and D1's "keep --json as is": identifiers are NOT prose.

    The prose names the secret the way ``lop secret list`` prints it; the machine
    surface keeps ``secret:<NAME>`` — the key ``credential share|revoke`` and the
    listing's rows are matched on — and the three-field copy block unchanged.
    """
    _placement(owner, member_holds=True)
    _ledger(owner, ledger)
    code = _lop_network("credential", "revoke", SECRET, "--from", MEMBER_NAME, "--json")
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["key"] == KEY
    assert payload["copy"] == {"confirmed": confirmed, "wiped": wiped, "wipe": wipe}


def test_one_secret_has_one_spelling_in_the_revoke_prose(
    owner: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    """D5: ``mark`` says ``CRM_API_KEY``; the revoke receipt used to say ``secret:CRM_API_KEY``."""
    _placement(owner, member_holds=True)
    _ledger(owner, False)
    _code, lines, _err = _run(capsys, "credential", "revoke", SECRET, "--from", MEMBER_NAME)
    assert lines[0] == f"revoked '{SECRET}' from {MEMBER_NAME}", lines
    assert "secret:" not in "\n".join(lines), lines


def test_the_renderer_returns_lines_and_a_provider_key_keeps_its_own_name() -> None:
    """The class-4 call site passes a provider name: it must come out as typed."""
    lines = messages_mod.render_copy_revoke_notice(MEMBER_NAME, "openai", copied=True, wiped=False)
    assert lines == [
        f"a wipe notice for 'openai' is queued for {MEMBER_NAME}: the copy is deleted on its "
        "next contact"
    ]
    no_row = messages_mod.render_copy_revoke_notice(MEMBER_NAME, KEY, copied=False, wiped=False)
    assert no_row[0] == (
        f"no copy of '{SECRET}' is recorded on {MEMBER_NAME}, so there is no wipe to send"
    )


def test_render_key_name_strips_only_the_secret_prefix() -> None:
    """D5's helper: one spelling for a store secret, every other key exactly as typed."""
    assert messages_mod.render_key_name(KEY) == SECRET
    for typed in ("openai", "github", "mcp:https://mcp.example/sse"):
        assert messages_mod.render_key_name(typed) == typed


@pytest.mark.parametrize("code", ["local_only", "copy_stale"])
def test_the_two_copy_refusals_name_the_secret_the_way_the_operator_types_it(code: str) -> None:
    """The same D5 rule on the borrower's side of the wire.

    ``client.py`` composes ``label = provider or key`` and a store secret has no provider,
    so these two S4 arms used to receive the placement key and print ``'secret:CRM_API_KEY'``
    beside an owner-side receipt that says ``'CRM_API_KEY'``. The remedy the ``local_only``
    arm quotes is a command the owner pastes; ``mark`` accepts the bare name.
    """
    error = BrokerError(
        code=code, key=KEY, owner_device=MEMBER, owner_device_name=OWNER_NAME, message="x"
    )
    rendered = messages_mod.render_broker_error(error, key=KEY, owner_name=OWNER_NAME)
    assert f"'{SECRET}'" in rendered, rendered
    assert "secret:" not in rendered, rendered
    if code == "local_only":
        assert f"('lop network credential mark {SECRET} default')" in rendered, rendered


# ---------------------------------------------------------------------------
# D8 — the removal receipt: one noun, the right number, the ceiling on every arm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("copies", "wiped", "timed_out", "expected"),
    [
        (2, 2, 0, [f"2 copies on {MEMBER_NAME} were deleted (the ending is confirmed)"]),
        (1, 1, 0, [f"1 copy on {MEMBER_NAME} was deleted (the ending is confirmed)"]),
        (
            3,
            1,
            0,
            [
                f"2 of 3 copies on {MEMBER_NAME} could NOT be confirmed deleted; a removed "
                "member is never contacted again"
            ],
        ),
        (
            2,
            0,
            1,
            [
                f"1 of 2 copies on {MEMBER_NAME} timed out — the member may still complete the "
                "deletion; it is never contacted again, so it cannot be re-checked",
                f"1 of 2 copies on {MEMBER_NAME} could NOT be confirmed deleted; a removed "
                "member is never contacted again",
            ],
        ),
        (
            1,
            0,
            1,
            [
                f"1 of 1 copy on {MEMBER_NAME} timed out — the member may still complete the "
                "deletion; it is never contacted again, so it cannot be re-checked"
            ],
        ),
    ],
    ids=["all-n", "all-one", "unreached", "timed-and-unreached", "timed-one"],
)
def test_the_removal_endings_name_a_copy_a_copy_and_carry_the_ceiling(
    copies: int, wiped: int, timed_out: int, expected: list[str]
) -> None:
    """D8: "copied secret(s)" was a third noun for what the family calls a copy.

    And the CONFIRMED arm used to be the one removal state without the ceiling —
    the same shape as D1 (the state that claims the most withholds the sentence
    about what it did not do), so the ceiling rides every arm here too.
    """
    lines = messages_mod.render_removal_endings(
        MEMBER_NAME, copies=copies, wiped=wiped, timed_out=timed_out
    )
    assert lines == [*expected, *CEILING]
    assert "copied secret" not in "\n".join(lines)


def test_a_member_with_no_copies_gets_no_ending_lines() -> None:
    assert messages_mod.render_removal_endings(MEMBER_NAME, copies=0, wiped=0) == []


# ---------------------------------------------------------------------------
# D6 + the sync overclaim — the mark receipts
# ---------------------------------------------------------------------------


def test_marking_sync_says_it_reaches_future_approvals_and_names_the_share_verb(
    owner: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    """The remediation's own finding: "approved devices get it" overclaimed.

    ``mark … sync`` writes the mark and NOTHING ELSE (see the next cell), so it
    changes what a device approved from now on is preselected with; a device that
    is approved but not yet sharing it needs the share verb (one that already shares
    it needs nothing, which is why the receipt says "not yet sharing it" and not
    "already approved"). The receipt says both.
    """
    _code, lines, _err = _run(capsys, "credential", "mark", SECRET, "sync")
    assert lines[0].startswith(f"'{SECRET}' is marked sync:"), lines
    assert "approved from now on" in lines[0], lines
    assert f"lop network credential share {SECRET} --with <device>" in lines[1], lines
    assert "not yet sharing it" in lines[1], lines


def test_a_sync_mark_does_not_reach_a_device_that_is_already_approved(
    owner: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    """The premise of the sentence above, on the owner's REAL announce computation.

    The member is active and the key is declared, but the member holds no grant.
    After ``mark … sync`` the owner owes it NOTHING (no holder row, no announce);
    after the share verb it owes exactly one present-value announce. If a future
    change makes the mark sweep existing members, this cell fails and the
    receipt and the guide are stale — which is the point of pinning it.
    """
    _placement(owner, member_holds=False)
    engine = sync.SyncEngine(
        SimpleNamespace(),
        root=owner.root,
        self_device=owner.identity.device_id,
        self_device_name=OWNER_NAME,
    )
    assert _lop_network("credential", "mark", SECRET, "sync") == 0
    capsys.readouterr()
    _network, owed = engine._pending_announces(MEMBER)  # noqa: SLF001 — the tick's own read
    assert owed == [], "a sync mark must not create an obligation to an unshared device"

    assert _lop_network("credential", "share", SECRET, "--with", MEMBER_NAME) == 0
    capsys.readouterr()
    _network, owed = engine._pending_announces(MEMBER)  # noqa: SLF001
    assert [(frame["key"], frame["value_state"]) for frame in owed] == [(KEY, "present")]


def test_marking_local_only_with_a_copy_outstanding_is_two_parseable_lines(
    owner: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    """D6: "N existing copies end with a wipe notice" read as if the copies grew a notice.

    And the surface where the policy is SET was the only one with no word about a
    holder that never reconnects.
    """
    _ledger(owner, False)
    _code, lines, _err = _run(capsys, "credential", "mark", SECRET, sync.MARK_LOCAL_ONLY)
    assert lines[0].startswith(f"'{SECRET}' is marked {sync.MARK_LOCAL_ONLY}:"), lines
    assert lines[1] == "1 existing copy will be wiped the next time its holder is reached", lines
    assert "a holder that never reconnects keeps its copy" in lines[2], lines
    assert "rotating the secret at its source" in lines[2], lines
    assert "end with a wipe notice" not in "\n".join(lines)


def test_the_local_only_receipt_pluralises_to_their_holders(
    owner: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    """The plural arm of D6's line: two ledger rows are two copies, held by two holders.

    The cell above only ever sees one row, so the ``copies`` / ``their holders are``
    branch had no reader. A second member is admitted so the ledger can honestly hold
    two rows for one key.
    """
    other = "d_" + "0" * 31 + "3"
    record = network_store.load(NETWORK, owner.root)
    relay_mod.admit(
        record,
        device_id=other,
        public_key="b" * 43,
        name="device-c",
        role="drive",
        capabilities=["drive", "broker_credential"],
        added_by=owner.identity.device_id,
        added_via="invite",
        root=owner.root,
        persist=False,
    )
    network_store.save(record, owner.root)
    _ledger(owner, False)
    with sync.mutate(NETWORK, owner.root) as document:
        document.record_ack(other, KEY, gen=1, digest=DIGEST, at=time.time(), wiped=False)
    _code, lines, _err = _run(capsys, "credential", "mark", SECRET, sync.MARK_LOCAL_ONLY)
    assert lines[1] == "2 existing copies will be wiped the next time their holders are reached"
    assert "a holder that never reconnects keeps its copy" in lines[2], lines


def test_clearing_a_mark_leaves_a_wiped_holder_served_again(
    owner: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    """Why the ``default`` receipt says "still gets it" and never "keeps it".

    Derived on the owner's real announce computation: a holder whose copy a kill-switch
    mark ended is owed a PRESENT announce again the moment the mark is cleared, because
    ``default`` clears the mark and nothing else — the grant never went away. A receipt
    that told the operator "a device already sharing it keeps it" described the state
    before the wipe, not after.
    """
    _placement(owner, member_holds=True)
    engine = sync.SyncEngine(
        SimpleNamespace(),
        root=owner.root,
        self_device=owner.identity.device_id,
        self_device_name=OWNER_NAME,
    )
    assert _lop_network("credential", "mark", SECRET, sync.MARK_LOCAL_ONLY) == 0
    capsys.readouterr()
    _network, owed = engine._pending_announces(MEMBER)  # noqa: SLF001 — the tick's own read
    assert owed == [], "the kill switch withholds the value from a holder"
    assert _lop_network("credential", "mark", SECRET, "default") == 0
    capsys.readouterr()
    _network, owed = engine._pending_announces(MEMBER)  # noqa: SLF001
    assert [(frame["key"], frame["value_state"]) for frame in owed] == [(KEY, "present")]


def test_marking_local_only_with_no_copy_outstanding_is_one_line(
    owner: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    _code, lines, _err = _run(capsys, "credential", "mark", SECRET, sync.MARK_LOCAL_ONLY)
    assert len(lines) == 1, lines


def test_clearing_a_mark_does_not_claim_a_device_that_already_shares_it_loses_it(
    owner: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    """Same overclaim family: an unmarked secret is still copied to a device that shares it."""
    _code, lines, _err = _run(capsys, "credential", "mark", SECRET, "default")
    assert lines[0].startswith(f"'{SECRET}' has no mark:"), lines
    assert "approved from now on" in lines[0], lines
    assert "a device already sharing it still gets it" in lines[1], lines


def test_the_mark_refusal_quotes_a_runnable_remedy(
    owner: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    """D8: the family's remedies are commands you can paste; this one ended in ``...``."""
    code, _lines, err = _run(capsys, "credential", "mark", "NO_SUCH_SECRET", "sync")
    assert code == 1
    assert "holds no secret named 'NO_SUCH_SECRET'" in err
    assert "store it first ('lop secret set NO_SUCH_SECRET')" in err
    assert "..." not in err


def test_a_bare_secret_prefix_on_mark_is_refused_by_name(
    owner: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    """O-1: the degenerate ``secret:`` spelling is refused like the empty argument.

    The name inside a bare ``secret:`` is empty, so the no-such-secret arm used to
    render ``named ''`` and a ``lop secret set `` remedy that cannot run — the one
    residual of D8's own class (a remedy is a command you can paste).
    """
    code, _lines, err = _run(capsys, "credential", "mark", "secret:", "sync")
    assert code == 1, err
    assert "give the secret to mark by name" in err, err
    assert "named ''" not in err, err
    assert "lop secret set " not in err, err


# ---------------------------------------------------------------------------
# N1, D7 — the small ones
# ---------------------------------------------------------------------------


def test_the_rotation_lock_sentence_uses_the_spaced_duration_form() -> None:
    """N1: ``wait 26s`` against every neighbouring ``45 s`` (``render_retry_after``)."""
    record = NetworkRecord(network_id=NETWORK, name="home-net")
    record.rotation_lock_until = 1000.0 + 26.4
    with pytest.raises(MeshRefusal) as excinfo:
        relay_mod.rotation_lock_refusal(record, now=1000.0)
    assert excinfo.value.code == "rotation_in_progress"
    assert "wait 26 s and try again" in excinfo.value.sentence
    assert "26s" not in excinfo.value.sentence


def test_the_listing_keeps_the_placement_key_as_the_row_identifier(
    owner: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    """D5's boundary from the other side: a listing ROW is an identifier, not prose.

    The sentences say ``CRM_API_KEY``; the row says ``secret:CRM_API_KEY`` because that is the
    key column an operator pastes back into ``share|revoke`` (both ALSO accept the bare name)
    and the field ``--json`` integrations match on. Pinned so a future "make everything the
    bare name" edit has to meet the decision instead of silently breaking the identifier.
    """
    _placement(owner, member_holds=True)
    code, lines, _err = _run(capsys, "credentials")
    assert code == 0, lines
    assert any(line.lstrip().startswith(KEY) for line in lines), lines
    assert _lop_network("credentials", "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    names = [row["credential_name"] for net in payload["networks"] for row in net["credentials"]]
    assert names == [KEY]


def test_the_revoke_verb_names_the_copy_it_ends(capsys: pytest.CaptureFixture[str]) -> None:
    """D7: the verb ends a COPY on a stored secret, which its one-line help never said."""
    with pytest.raises(SystemExit):
        _parser().parse_args(["network", "credential", "--help"])
    assert "end its copy of a stored secret" in " ".join(capsys.readouterr().out.split())


# ---------------------------------------------------------------------------
# D3, D4 — the two guide sentences that said the opposite of a fact
# ---------------------------------------------------------------------------


def _guide() -> str:
    return " ".join((REPO / "local_operator/guides/network/GUIDE.md").read_text("utf-8").split())


def test_the_guide_no_longer_says_a_mark_ends_copies_that_already_left() -> None:
    """D3: the parenthetical carried the ceiling's own words on the OPPOSITE claim."""
    guide = _guide()
    assert "ends copies that already left" not in guide
    assert "ends the copies this device can still reach" in guide


def _credentials_section() -> str:
    """The guide's "Credentials on a peer" section, whitespace-normalised.

    Sliced by its heading BEFORE normalising, because a phrase the pairing chapter already
    carries (``not offered`` appears there) would otherwise let a cell pass without the
    credentials section ever saying it.
    """
    raw = (REPO / "local_operator/guides/network/GUIDE.md").read_text(encoding="utf-8")
    return " ".join(raw.split("## Credentials on a peer", 1)[1].split("\n## ", 1)[0].split())


def test_the_guide_quotes_the_label_the_share_list_actually_prints() -> None:
    """The pair the design review asked to make true: the guide says what the card says.

    The guide called an unmarked, undeclared secret "offered"; the share list prints
    ``not offered`` for a ``share: false`` row (``offers.render_rows``). The label is derived
    from the renderer here, not retyped, so the cell follows the code if the label moves.
    """
    row = {"key": KEY, "kind": "store-secret", "label": ""}
    unticked = offers_mod.render_rows([{**row, "share": False}])[0].rsplit("   ", 1)[1]
    ticked = offers_mod.render_rows([{**row, "share": True}])[0].rsplit("   ", 1)[1]
    assert (unticked, ticked) == ("not offered", "will be served")
    assert f"`{unticked}`" in _credentials_section()
    design = " ".join(
        (REPO / "docs/design/mesh-consent-provisioning.md").read_text(encoding="utf-8").split()
    )
    assert f"prints such a row as `{unticked}`" in design


def test_the_guide_scopes_values_never_travel_to_the_push_and_calls_github_brokered() -> None:
    """Two coherence verdicts the review round forced, made structural.

    "Values never travel" sat in the MCP paragraph while this PR's own section teaches that
    store secrets are copied, so the claim is scoped to what it is true of (the PUSH). And
    the guide called a GitHub PAT a copied class's example while its GitHub section describes
    the ``github`` credential as brokered: the credentials section now says which is which.
    """
    guide = _guide()
    assert "**Values never travel**" not in guide
    assert "**The push itself carries no values**" in guide
    section = _credentials_section()
    assert "The `github` credential itself is brokered, never copied" in section
    assert "a `GITHUB_TOKEN` marked `sync` puts that token on the device as an ordinary secret" in (
        section
    )


def test_the_guide_gives_each_copied_class_its_own_resting_place_by_listing_kind() -> None:
    """D4: "re-sealed … never plaintext" is true of ``store-secret`` only.

    A copied ``api-key-static`` login rests in the node's 0600 ``auth.db`` — the
    row a local sign-in writes — so the guide must say which class each guarantee
    belongs to, in the kinds ``lop network credentials`` actually prints, and must
    not hold a GitHub PAT up as the copied class's example (that credential is the
    brokered one, further down the same chapter).
    """
    guide = _guide()
    assert "`store-secret`" in guide and "`api-key-static`" in guide
    assert "0600 `auth.db`" in guide
    assert "static keys (a GitHub PAT" not in guide
